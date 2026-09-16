import os

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8001").rstrip("/")

QUERY_API_URL = f"{API_BASE_URL}/query"
UPLOAD_API_URL = f"{API_BASE_URL}/upload"
FEEDBACK_API_URL = f"{API_BASE_URL}/feedback"
ANALYTICS_API_URL = f"{API_BASE_URL}/analytics"
DOCUMENTS_API_URL = f"{API_BASE_URL}/documents"


st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="📚",
    layout="centered"
)

st.title("Enterprise Knowledge AI")


# =========================
# Session State
# =========================

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "last_log_id" not in st.session_state:
    st.session_state.last_log_id = None

if "last_latency" not in st.session_state:
    st.session_state.last_latency = None

if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False


# =========================
# 上传文件
# =========================

st.subheader("Upload document")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    if st.button(
        "Upload PDF",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Uploading and processing..."
            ):

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf"
                    )
                }

                response = requests.post(
                    UPLOAD_API_URL,
                    files=files,
                    timeout=600
                )

            if response.status_code == 200:

                data = response.json()

                st.success(
                    "Document uploaded successfully."
                )

                st.write(
                    "Filename:",
                    data.get("filename")
                )

                st.write(
                    "Chunks created:",
                    data.get("chunks_created")
                )

            else:

                st.error(
                    f"Upload failed: "
                    f"{response.status_code}"
                )

                st.code(
                    response.text
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "Cannot connect to FastAPI."
            )

        except requests.exceptions.Timeout:

            st.error(
                "Upload timed out."
            )

        except Exception as error:

            st.error(
                f"Upload failed: {error}"
            )


st.divider()

# =========================
# 提问
# =========================

st.subheader("Ask your documents")


# =========================
# 获取文档列表
# =========================

documents = []

try:
    documents_response = requests.get(
        DOCUMENTS_API_URL,
        timeout=30
    )

    if documents_response.status_code == 200:
        documents = (
            documents_response
            .json()
            .get(
                "documents",
                []
            )
        )

except Exception:
    documents = []


# =========================
# 文档选择
# =========================

selected_document_id = None


if documents:

    document_options = {
        document["filename"]:
            document["id"]
        for document in documents
    }

    selected_document_name = (
        st.selectbox(
            "Select document",
            options=list(
                document_options.keys()
            )
        )
    )

    selected_document_id = (
        document_options[
            selected_document_name
        ]
    )

else:

    st.warning(
        "No documents found."
    )


# =========================
# 输入问题
# =========================

question = st.text_input(
    "Question",
    placeholder=(
        "What is the maximum input "
        "current per MPPT?"
    )
)


# =========================
# Ask
# =========================

if st.button(
    "Ask",
    use_container_width=True
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    elif selected_document_id is None:

        st.warning(
            "Please select a document."
        )

    else:

        try:

            with st.spinner(
                "Searching selected document..."
            ):

                response = requests.post(
                    QUERY_API_URL,

                    json={
                        "question":
                            question,

                        "document_id":
                            selected_document_id
                    },

                    timeout=180
                )

            if response.status_code == 200:

                data = response.json()

                st.session_state.last_answer = (
                    data.get(
                        "answer",
                        "No answer returned."
                    )
                )

                st.session_state.last_sources = (
                    data.get(
                        "sources",
                        []
                    )
                )

                st.session_state.last_log_id = (
                    data.get(
                        "log_id"
                    )
                )

                st.session_state.last_latency = (
                    data.get(
                        "latency"
                    )
                )

                st.session_state.feedback_submitted = False

            else:

                st.error(
                    f"Query failed: "
                    f"{response.status_code}"
                )

                st.code(
                    response.text
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "Cannot connect to FastAPI."
            )

        except requests.exceptions.Timeout:

            st.error(
                "Query timed out."
            )

        except Exception as error:

            st.error(
                f"Query failed: {error}"
            )


# =========================
# 显示回答
# =========================

if st.session_state.last_answer:

    st.divider()

    st.subheader("Answer")

    st.success(
        st.session_state.last_answer
    )

    if (
        st.session_state.last_latency
        is not None
    ):

        st.caption(
            f"Response time: "
            f"{st.session_state.last_latency} seconds"
        )


    # =========================
    # Sources
    # =========================

    sources = (
        st.session_state.last_sources
    )

    if sources:

        with st.expander(
            f"View sources "
            f"({len(sources)})"
        ):

            for (
                index,
                source
            ) in enumerate(
                sources,
                start=1
            ):

                st.markdown(
                    f"""
**Source {index}**

- Document ID: `{source.get("document_id")}`
- Page: `{source.get("page_number")}`
- Chunk: `{source.get("chunk_index")}`
- Similarity: `{source.get("similarity")}`
"""
                )


    # =========================
    # Feedback
    # =========================

    log_id = (
        st.session_state.last_log_id
    )

    if log_id:

        st.write(
            "Was this answer helpful?"
        )

        if not (
            st.session_state
            .feedback_submitted
        ):

            col1, col2 = (
                st.columns(2)
            )

            with col1:

                if st.button(
                    "👍 Helpful",
                    use_container_width=True,
                    key=f"helpful_{log_id}"
                ):

                    feedback_response = (
                        requests.post(
                            FEEDBACK_API_URL,

                            json={
                                "log_id":
                                    log_id,

                                "feedback":
                                    "helpful"
                            },

                            timeout=30
                        )
                    )

                    if (
                        feedback_response
                        .status_code
                        == 200
                    ):

                        st.session_state.feedback_submitted = True

                        st.success(
                            "Thanks for your feedback."
                        )

            with col2:

                if st.button(
                    "👎 Not Helpful",
                    use_container_width=True,
                    key=f"not_helpful_{log_id}"
                ):

                    feedback_response = (
                        requests.post(
                            FEEDBACK_API_URL,

                            json={
                                "log_id":
                                    log_id,

                                "feedback":
                                    "not_helpful"
                            },

                            timeout=30
                        )
                    )

                    if (
                        feedback_response
                        .status_code
                        == 200
                    ):

                        st.session_state.feedback_submitted = True

                        st.success(
                            "Thanks for your feedback."
                        )

        else:

            st.info(
                "Feedback submitted."
            )

#分析板块
st.subheader("Analytics")

if st.button(
    "Refresh Analytics",
    use_container_width=True
):

    try:

        response = requests.get(
            ANALYTICS_API_URL,
            timeout=30
        )

        if response.status_code == 200:

            analytics = response.json()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Total Questions",
                    analytics.get(
                        "total_questions",
                        0
                    )
                )

            with col2:
                st.metric(
                    "Average Response Time",
                    f"{analytics.get('average_latency', 0)} s"
                )

            with col3:
                st.metric(
                    "Helpful Rate",
                    f"{analytics.get('helpful_rate', 0)}%"
                )

            st.subheader(
                "Recent Questions"
            )

            recent_logs = analytics.get(
                "recent_logs",
                []
            )

            if recent_logs:
                st.dataframe(
                    recent_logs,
                    use_container_width=True
                )
            else:
                st.info(
                    "No query logs yet."
                )

            st.subheader(
                "Not Helpful Questions"
            )

            bad_questions = analytics.get(
                "not_helpful_questions",
                []
            )

            if bad_questions:
                st.dataframe(
                    bad_questions,
                    use_container_width=True
                )
            else:
                st.success(
                    "No negative feedback yet."
                )

        else:
            st.error(
                f"Analytics failed: "
                f"{response.status_code}"
            )

            st.code(
                response.text
            )

    except Exception as error:

        st.error(
            f"Analytics failed: {error}"
        )
