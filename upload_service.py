import os
import re
from pathlib import Path

import fitz
import httpx
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client


# ============================================================
# 配置
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
ENV_FILE = PROJECT_DIR / ".env"

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Embedding 分批
EMBEDDING_BATCH_SIZE = 20

# Supabase 分批写入
DB_BATCH_SIZE = 10


MODEL_PATTERN = re.compile(
    r"^X3-AELIO-\d+(?:\.\d+)?K(?:-P)?$",
    re.IGNORECASE
)


SECTION_KEYWORDS = [
    "PV INPUT",
    "AC INPUT & OUTPUT (ON-GRID)",
    "BATTERY",
    "EPS (OFF-GRID) OUTPUT",
    "EFFICIENCY",
    "ENVIRONMENT LIMIT",
    "GENERAL",
    "PROTECTION",
]


# ============================================================
# 环境变量
# ============================================================

def load_environment():
    load_dotenv(
        dotenv_path=ENV_FILE,
        override=True
    )


def create_siliconflow_client():
    load_environment()

    api_key = os.getenv(
        "SILICONFLOW_API_KEY"
    )

    base_url = os.getenv(
        "SILICONFLOW_BASE_URL",
        "https://api.siliconflow.cn/v1"
    )

    if not api_key:
        raise RuntimeError(
            "SILICONFLOW_API_KEY is missing."
        )

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.Client(
            trust_env=False,
            timeout=180.0
        ),
    )


def create_supabase_client():
    load_environment()

    url = os.getenv(
        "SUPABASE_URL"
    )

    key = os.getenv(
        "SUPABASE_KEY"
    )

    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL or SUPABASE_KEY is missing."
        )

    return create_client(
        url,
        key
    )


# ============================================================
# 文本处理
# ============================================================

def clean_text(value):
    if value is None:
        return ""

    value = str(value)

    value = value.replace(
        "\n",
        " "
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


def split_text(text):
    chunks = []

    start = 0

    while start < len(text):

        end = (
            start
            + CHUNK_SIZE
        )

        chunk = (
            text[start:end]
            .strip()
        )

        if chunk:
            chunks.append(
                chunk
            )

        start += (
            CHUNK_SIZE
            - CHUNK_OVERLAP
        )

    return chunks


# ============================================================
# 型号识别
# ============================================================

def detect_models_from_words(page):
    """
    从 PDF word 坐标中识别完整子型号。
    """

    models = []

    words = page.get_text(
        "words"
    )

    for word in words:

        x0, y0, x1, y1, text, *_ = word

        text = clean_text(
            text
        )

        if MODEL_PATTERN.match(
            text
        ):

            models.append({
                "model":
                    text.upper(),

                "x":
                    (
                        x0 + x1
                    ) / 2,

                "y":
                    (
                        y0 + y1
                    ) / 2
            })

    # 去重
    unique_models = {}

    for model in models:
        unique_models[
            model["model"]
        ] = model

    models = list(
        unique_models.values()
    )

    models.sort(
        key=lambda item:
            item["x"]
    )

    return models


# ============================================================
# Series Summary
# ============================================================

def build_series_summary_chunk(
    models,
    filename,
    page_number
):
    """
    创建系列级 summary chunk。

    用于回答：
    - How many models are available?
    - What models are available?
    - What power ratings are available?
    """

    if not models:
        return None

    model_names = [
        model["model"]
        for model in models
    ]

    power_ratings = []

    for model_name in model_names:

        match = re.search(
            r"X3-AELIO-(\d+(?:\.\d+)?)K",
            model_name,
            re.IGNORECASE
        )

        if match:

            rating = (
                match.group(1)
            )

            if (
                rating
                not in power_ratings
            ):
                power_ratings.append(
                    rating
                )

    models_text = "\n".join(
        f"- {model}"
        for model
        in model_names
    )

    power_text = "\n".join(
        f"- {rating} kW"
        for rating
        in power_ratings
    )

    content = (
        f"Document: {filename}\n"
        f"Product family: X3-AELIO\n"
        f"Number of models: "
        f"{len(model_names)}\n\n"
        f"Available models:\n"
        f"{models_text}\n\n"
        f"Available power ratings:\n"
        f"{power_text}"
    )

    return {
        "page_number":
            page_number,

        "content":
            content,

        "chunk_type":
            "series_summary"
    }


# ============================================================
# Section 识别
# ============================================================

def detect_sections(page):
    """
    找 Section 并去重。
    """

    sections = []

    for section in (
        SECTION_KEYWORDS
    ):

        matches = (
            page.search_for(
                section
            )
        )

        for rect in matches:

            sections.append({
                "section":
                    section,

                "y":
                    rect.y0
            })

    sections.sort(
        key=lambda item:
            item["y"]
    )

    cleaned_sections = []

    for item in sections:

        if not cleaned_sections:

            cleaned_sections.append(
                item
            )

            continue

        last = (
            cleaned_sections[-1]
        )

        if (
            item["section"]
            == last["section"]
            and
            abs(
                item["y"]
                - last["y"]
            ) < 25
        ):
            continue

        cleaned_sections.append(
            item
        )

    return cleaned_sections


def get_section_for_y(
    y,
    sections
):
    """
    根据参数的 Y 坐标，
    找它上方最近的 Section。
    """

    current_section = (
        "UNKNOWN"
    )

    for section in sections:

        if (
            section["y"]
            <= y
        ):

            current_section = (
                section["section"]
            )

        else:
            break

    return current_section


# ============================================================
# PDF words 按行组合
# ============================================================

def group_words_by_line(
    page,
    tolerance=3
):
    words = page.get_text(
        "words"
    )

    sorted_words = sorted(
        words,
        key=lambda w: (
            w[1],
            w[0]
        )
    )

    lines = []

    for word in sorted_words:

        x0, y0, x1, y1, text, *_ = word

        center_y = (
            y0 + y1
        ) / 2

        placed = False

        for line in lines:

            if abs(
                line["y"]
                - center_y
            ) <= tolerance:

                line[
                    "words"
                ].append({
                    "x0":
                        x0,

                    "x1":
                        x1,

                    "x":
                        (
                            x0 + x1
                        ) / 2,

                    "text":
                        clean_text(
                            text
                        )
                })

                placed = True

                break

        if not placed:

            lines.append({
                "y":
                    center_y,

                "words": [{
                    "x0":
                        x0,

                    "x1":
                        x1,

                    "x":
                        (
                            x0 + x1
                        ) / 2,

                    "text":
                        clean_text(
                            text
                        )
                }]
            })

    for line in lines:

        line[
            "words"
        ].sort(
            key=lambda item:
                item["x"]
        )

    return lines


# ============================================================
# 型号/value 匹配
# ============================================================

def find_nearest_model(
    x,
    models
):
    """
    根据 X 坐标找到最近的型号列。
    """

    if not models:
        return None

    return min(
        models,
        key=lambda model:
            abs(
                model["x"]
                - x
            )
    )


def is_probably_parameter_line(
    line,
    models
):
    """
    判断是否为：
    Parameter + Value
    """

    if not models:
        return False

    first_model_x = min(
        model["x"]
        for model in models
    )

    left_words = [
        word
        for word
        in line["words"]
        if (
            word["x"]
            < first_model_x - 30
        )
    ]

    right_words = [
        word
        for word
        in line["words"]
        if (
            word["x"]
            >= first_model_x - 30
        )
    ]

    return bool(
        left_words
        and right_words
    )


def extract_parameter_name(
    line,
    models
):
    """
    型号列左侧文字作为 Parameter。
    """

    first_model_x = min(
        model["x"]
        for model in models
    )

    parameter_words = []

    for word in (
        line["words"]
    ):

        if (
            word["x"]
            < first_model_x - 30
        ):

            parameter_words.append(
                word["text"]
            )

    return clean_text(
        " ".join(
            parameter_words
        )
    )


def extract_values_for_models(
    line,
    models
):
    """
    根据 X 坐标，
    将 Value 匹配给对应型号。
    """

    if not models:
        return {}

    first_model_x = min(
        model["x"]
        for model in models
    )

    values = {
        model["model"]: []
        for model
        in models
    }

    for word in (
        line["words"]
    ):

        if (
            word["x"]
            < first_model_x - 30
        ):
            continue

        nearest_model = (
            find_nearest_model(
                word["x"],
                models
            )
        )

        if nearest_model:

            values[
                nearest_model[
                    "model"
                ]
            ].append(
                word["text"]
            )

    final_values = {}

    for (
        model_name,
        parts
    ) in values.items():

        value = clean_text(
            " ".join(
                parts
            )
        )

        if value:

            final_values[
                model_name
            ] = value

    return final_values


# ============================================================
# 核心结构化 chunk
# ============================================================

def extract_structured_chunks(
    page,
    filename,
    page_number
):
    """
    生成 parameter-level chunks：

    Document
    Product model
    Section
    Parameter
    Value
    """

    models = (
        detect_models_from_words(
            page
        )
    )

    sections = (
        detect_sections(
            page
        )
    )

    print(
        f"[Page {page_number}] "
        f"Detected models:",
        [
            model["model"]
            for model
            in models
        ]
    )

    if len(models) < 2:
        return []

    lines = (
        group_words_by_line(
            page
        )
    )

    chunks = []

    for line in lines:

        y = (
            line["y"]
        )

        # 跳过型号标题行
        if any(
            abs(
                y
                - model["y"]
            ) < 5
            for model
            in models
        ):
            continue

        if not (
            is_probably_parameter_line(
                line,
                models
            )
        ):
            continue

        parameter = (
            extract_parameter_name(
                line,
                models
            )
        )

        if not parameter:
            continue

        # 避免正文误判
        if len(parameter) > 120:
            continue

        values = (
            extract_values_for_models(
                line,
                models
            )
        )

        if not values:
            continue

        # ------------------------------------------------
        # 如果只识别到一个 value，
        # 暂时认为它是所有型号共享值。
        #
        # 这就是你目前高召回版本的逻辑，
        # 先继续保留。
        # ------------------------------------------------

        if (
            len(values) == 1
            and
            len(models) > 1
        ):

            shared_value = next(
                iter(
                    values.values()
                )
            )

            values = {
                model["model"]:
                    shared_value
                for model
                in models
            }

        section = (
            get_section_for_y(
                y,
                sections
            )
        )

        for (
            model_name,
            value
        ) in values.items():

            if not value:
                continue

            content = (
                f"Document: {filename}\n"
                f"Product model: "
                f"{model_name}\n"
                f"Section: "
                f"{section}\n"
                f"Parameter: "
                f"{parameter}\n"
                f"Value: "
                f"{value}"
            )

            chunks.append({
                "page_number":
                    page_number,

                "content":
                    content,

                "chunk_type":
                    "structured"
            })

    return chunks


# ============================================================
# 页面处理
# ============================================================

def extract_page_chunks(
    page,
    filename,
    page_number
):
    """
    结构化参数 chunks
    +
    1 个 series summary chunk
    """

    structured_chunks = (
        extract_structured_chunks(
            page,
            filename,
            page_number
        )
    )

    # ============================================
    # 新增 Series Summary Chunk
    # ============================================

    models = (
        detect_models_from_words(
            page
        )
    )

    summary_chunk = (
        build_series_summary_chunk(
            models,
            filename,
            page_number
        )
    )

    if summary_chunk:

        structured_chunks.append(
            summary_chunk
        )

        print(
            f"[Page {page_number}] "
            f"Series summary chunk added."
        )

    if structured_chunks:

        print(
            f"[Page {page_number}] "
            f"Structured chunks: "
            f"{len(structured_chunks)}"
        )

        return (
            structured_chunks
        )

    # ============================================
    # fallback 普通文本
    # ============================================

    text = page.get_text(
        "text"
    )

    if not text.strip():
        return []

    text_chunks = (
        split_text(
            text
        )
    )

    print(
        f"[Page {page_number}] "
        f"Using text chunks: "
        f"{len(text_chunks)}"
    )

    return [
        {
            "page_number":
                page_number,

            "content":
                chunk,

            "chunk_type":
                "text"
        }
        for chunk
        in text_chunks
    ]


# ============================================================
# Embedding
# ============================================================

def create_embeddings(
    texts,
    client,
    batch_size=EMBEDDING_BATCH_SIZE
):
    """
    Embedding 分批生成。
    """

    if not texts:
        return []

    all_embeddings = []

    total = (
        len(texts)
    )

    for start in range(
        0,
        total,
        batch_size
    ):

        end = min(
            start
            + batch_size,
            total
        )

        batch = (
            texts[
                start:end
            ]
        )

        print(
            f"Embedding batch "
            f"{start + 1}-{end} "
            f"/ {total}..."
        )

        response = (
            client
            .embeddings
            .create(
                model=
                    EMBEDDING_MODEL,

                input=
                    batch,

                encoding_format=
                    "float"
            )
        )

        batch_embeddings = [
            item.embedding
            for item
            in response.data
        ]

        if (
            len(
                batch_embeddings
            )
            != len(batch)
        ):

            raise RuntimeError(
                "Embedding count mismatch."
            )

        all_embeddings.extend(
            batch_embeddings
        )

        print(
            f"Embedding batch "
            f"{start + 1}-{end} "
            f"finished."
        )

    return all_embeddings


# ============================================================
# Supabase 分批插入
# ============================================================

def insert_chunks_in_batches(
    supabase,
    chunks,
    batch_size=DB_BATCH_SIZE
):
    """
    防止 statement timeout。
    """

    total = (
        len(chunks)
    )

    print(
        f"\nStarting Supabase insert: "
        f"{total} chunks"
    )

    for start in range(
        0,
        total,
        batch_size
    ):

        end = min(
            start
            + batch_size,
            total
        )

        batch = (
            chunks[
                start:end
            ]
        )

        print(
            f"Inserting chunks "
            f"{start + 1}-{end} "
            f"/ {total}..."
        )

        (
            supabase
            .table(
                "document_chunks"
            )
            .insert(
                batch
            )
            .execute()
        )

        print(
            f"Supabase batch "
            f"{start + 1}-{end} "
            f"finished."
        )

    print(
        "All Supabase inserts finished."
    )


# ============================================================
# 主入口
# ============================================================

def process_pdf(
    file_bytes,
    filename
):
    """
    PDF
    ↓
    Parameter chunks
    ↓
    Series summary chunk
    ↓
    Embedding
    ↓
    Supabase
    """

    siliconflow = (
        create_siliconflow_client()
    )

    supabase = (
        create_supabase_client()
    )

    print(
        f"\nStarting upload: "
        f"{filename}"
    )

    # ============================================
    # 创建 document
    # ============================================

    document_response = (
        supabase
        .table(
            "documents"
        )
        .insert({
            "filename":
                filename
        })
        .execute()
    )

    if not (
        document_response.data
    ):

        raise RuntimeError(
            "Failed to create "
            "document record."
        )

    document_id = (
        document_response
        .data[0]["id"]
    )

    # ============================================
    # 打开 PDF
    # ============================================

    pdf = fitz.open(
        stream=file_bytes,
        filetype="pdf"
    )

    all_chunks = []

    chunk_index = 1

    try:

        # ========================================
        # 逐页处理
        # ========================================

        for (
            page_number,
            page
        ) in enumerate(
            pdf,
            start=1
        ):

            print(
                f"\nProcessing page "
                f"{page_number}..."
            )

            page_chunks = (
                extract_page_chunks(
                    page,
                    filename,
                    page_number
                )
            )

            if not page_chunks:

                print(
                    f"[Page "
                    f"{page_number}] "
                    f"No chunks found."
                )

                continue

            contents = [
                chunk[
                    "content"
                ]
                for chunk
                in page_chunks
            ]

            # ====================================
            # Embedding
            # ====================================

            print(
                f"[Page {page_number}] "
                f"Creating embeddings "
                f"for "
                f"{len(contents)} "
                f"chunks..."
            )

            embeddings = (
                create_embeddings(
                    contents,
                    siliconflow
                )
            )

            print(
                f"[Page {page_number}] "
                f"All embeddings finished."
            )

            if (
                len(embeddings)
                != len(
                    page_chunks
                )
            ):

                raise RuntimeError(
                    "Embedding count does "
                    "not match chunk count."
                )

            # ====================================
            # 准备 DB rows
            # ====================================

            for (
                chunk,
                embedding
            ) in zip(
                page_chunks,
                embeddings
            ):

                all_chunks.append({
                    "document_id":
                        document_id,

                    "page_number":
                        chunk[
                            "page_number"
                        ],

                    "chunk_index":
                        chunk_index,

                    "content":
                        chunk[
                            "content"
                        ],

                    "embedding":
                        embedding
                })

                chunk_index += 1

        if not all_chunks:

            raise RuntimeError(
                "No chunks were created "
                "from this PDF."
            )

        # ========================================
        # 写入 Supabase
        # ========================================

        insert_chunks_in_batches(
            supabase,
            all_chunks,
            DB_BATCH_SIZE
        )

    except Exception as error:

        print(
            "\nUpload failed:",
            error
        )

        # ========================================
        # 清理失败上传的数据
        # ========================================

        try:

            (
                supabase
                .table(
                    "document_chunks"
                )
                .delete()
                .eq(
                    "document_id",
                    document_id
                )
                .execute()
            )

        except Exception as cleanup_error:

            print(
                "Chunk cleanup failed:",
                cleanup_error
            )

        try:

            (
                supabase
                .table(
                    "documents"
                )
                .delete()
                .eq(
                    "id",
                    document_id
                )
                .execute()
            )

        except Exception as cleanup_error:

            print(
                "Document cleanup failed:",
                cleanup_error
            )

        raise

    finally:

        pdf.close()

    # ============================================
    # 完成
    # ============================================

    print(
        f"\nUpload completed: "
        f"{filename}"
    )

    print(
        f"Total chunks: "
        f"{len(all_chunks)}"
    )

    return {
        "document_id":
            document_id,

        "filename":
            filename,

        "chunks_created":
            len(
                all_chunks
            )
    }
