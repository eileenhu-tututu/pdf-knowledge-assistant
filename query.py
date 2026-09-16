import os

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client


load_dotenv()


SILICONFLOW_API_KEY = os.getenv(
    "SILICONFLOW_API_KEY"
)

SILICONFLOW_BASE_URL = os.getenv(
    "SILICONFLOW_BASE_URL",
    "https://api.siliconflow.cn/v1"
)

SUPABASE_URL = os.getenv(
    "SUPABASE_URL"
)

SUPABASE_KEY = os.getenv(
    "SUPABASE_KEY"
)


EMBEDDING_MODEL = (
    "Qwen/Qwen3-Embedding-8B"
)

LLM_MODEL = (
    "deepseek-ai/DeepSeek-V3.2"
)

DEFAULT_TOP_K = 3
SUMMARY_TOP_K = 10


# =========================
# Clients
# =========================

def create_siliconflow_client():

    if not SILICONFLOW_API_KEY:
        raise RuntimeError(
            "SILICONFLOW_API_KEY is missing."
        )

    return OpenAI(
        api_key=
            SILICONFLOW_API_KEY,

        base_url=
            SILICONFLOW_BASE_URL,

        http_client=
            httpx.Client(
                trust_env=False,
                timeout=180.0
            )
    )


def create_supabase_client():

    if (
        not SUPABASE_URL
        or not SUPABASE_KEY
    ):
        raise RuntimeError(
            "SUPABASE_URL or SUPABASE_KEY is missing."
        )

    return create_client(
        SUPABASE_URL,
        SUPABASE_KEY
    )


supabase = (
    create_supabase_client()
)


# =========================
# Query Router
# =========================

def is_series_summary_question(
    question
):
    """
    判断是不是系列级汇总问题。
    """

    q = question.lower()

    keywords = [
        "how many models",
        "available models",
        "what models",
        "which models",
        "model lineup",
        "product lineup",
        "power ratings",
        "available power ratings",
        "what power ratings",
        "series models",
        "series lineup"
    ]

    return any(
        keyword in q
        for keyword in keywords
    )


# =========================
# Embedding
# =========================

def create_query_embedding(
    question,
    client
):

    response = (
        client
        .embeddings
        .create(
            model=
                EMBEDDING_MODEL,

            input=
                question,

            encoding_format=
                "float"
        )
    )

    return (
        response
        .data[0]
        .embedding
    )


# =========================
# Vector Search
# =========================

def search_top_chunks(
    query_embedding,
    match_count=DEFAULT_TOP_K,
    document_id=None
):
    """
    document_id 不为空时，
    只搜索指定文档。
    """

    result = (
        supabase
        .rpc(
            "match_document_chunks",
            {
                "query_embedding":
                    query_embedding,

                "match_count":
                    match_count,

                "filter_document_id":
                    document_id
            }
        )
        .execute()
    )

    return (
        result.data
        or []
    )


# =========================
# Summary Search
# =========================

def get_series_summary_chunks(
    document_id=None
):
    """
    查 series summary chunk。

    如果指定 document_id，
    只从该文档查。
    """

    query = (
        supabase
        .table(
            "document_chunks"
        )
        .select(
            "id, "
            "document_id, "
            "page_number, "
            "chunk_index, "
            "content"
        )
        .ilike(
            "content",
            "%Product family: X3-AELIO%"
        )
    )

    if document_id is not None:

        query = query.eq(
            "document_id",
            document_id
        )

    response = (
        query.execute()
    )

    return (
        response.data
        or []
    )


# =========================
# Context
# =========================

def build_context(
    chunks
):

    contexts = []

    for (
        index,
        chunk
    ) in enumerate(
        chunks,
        start=1
    ):

        content = chunk.get(
            "content",
            ""
        )

        page_number = chunk.get(
            "page_number",
            "Unknown"
        )

        chunk_index = chunk.get(
            "chunk_index",
            "Unknown"
        )

        similarity = chunk.get(
            "similarity"
        )

        context = (
            f"[Context {index}]\n"
            f"Page: "
            f"{page_number}\n"
            f"Chunk: "
            f"{chunk_index}\n"
        )

        if similarity is not None:

            context += (
                f"Similarity: "
                f"{similarity}\n"
            )

        context += content

        contexts.append(
            context
        )

    return "\n\n".join(
        contexts
    )


# =========================
# LLM
# =========================

def generate_answer(
    question,
    context,
    client
):

    system_prompt = """
You are an enterprise document assistant.

Answer using ONLY the provided context.

Rules:

1. Do not invent information.

2. If the context contains the answer,
   answer clearly and directly.

3. Pay close attention to the exact
   product model.

4. Do not mix specifications from
   different product models.

5. If the question asks for:
   - available models
   - number of models
   - power ratings
   - product lineup
   use the series summary information
   when available.

6. Distinguish between:
   - number of product models
   - number of unique power ratings

7. If the context is insufficient,
   say:
   "I cannot find enough information
   in the provided documents."

8. Keep answers concise.
"""

    user_prompt = f"""
Question:

{question}

Context:

{context}
"""

    response = (
        client
        .chat
        .completions
        .create(
            model=
                LLM_MODEL,

            messages=[
                {
                    "role":
                        "system",

                    "content":
                        system_prompt
                },
                {
                    "role":
                        "user",

                    "content":
                        user_prompt
                }
            ],

            temperature=0
        )
    )

    return (
        response
        .choices[0]
        .message
        .content
    )


# =========================
# Main RAG
# =========================

def answer_question(
    question,
    document_id=None
):

    client = (
        create_siliconflow_client()
    )

    # =====================
    # Route 1:
    # Summary Query
    # =====================

    if is_series_summary_question(
        question
    ):

        print(
            "Query route: "
            "SERIES SUMMARY"
        )

        summary_chunks = (
            get_series_summary_chunks(
                document_id=
                    document_id
            )
        )

        if summary_chunks:

            top_chunks = (
                summary_chunks
            )

        else:

            print(
                "Summary chunk not found. "
                "Using vector fallback."
            )

            query_embedding = (
                create_query_embedding(
                    question,
                    client
                )
            )

            top_chunks = (
                search_top_chunks(
                    query_embedding,
                    match_count=
                        SUMMARY_TOP_K,
                    document_id=
                        document_id
                )
            )

    # =====================
    # Route 2:
    # Parameter Query
    # =====================

    else:

        print(
            "Query route: "
            "VECTOR SEARCH"
        )

        query_embedding = (
            create_query_embedding(
                question,
                client
            )
        )

        top_chunks = (
            search_top_chunks(
                query_embedding,
                match_count=
                    DEFAULT_TOP_K,
                document_id=
                    document_id
            )
        )

    # =====================
    # No Results
    # =====================

    if not top_chunks:

        return {
            "answer":
                (
                    "I cannot find enough "
                    "information in the "
                    "provided documents."
                ),

            "sources":
                []
        }

    # =====================
    # Build Context
    # =====================

    context = (
        build_context(
            top_chunks
        )
    )

    # =====================
    # Generate Answer
    # =====================

    answer = (
        generate_answer(
            question,
            context,
            client
        )
    )

    # =====================
    # Sources
    # =====================

    sources = []

    for chunk in top_chunks:

        sources.append({
            "document_id":
                chunk.get(
                    "document_id"
                ),

            "page_number":
                chunk.get(
                    "page_number"
                ),

            "chunk_index":
                chunk.get(
                    "chunk_index"
                ),

            "similarity":
                chunk.get(
                    "similarity"
                )
        })

    return {
        "answer":
            answer,

        "sources":
            sources
    }