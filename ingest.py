"""Extract PDF text, create chunks, and generate embeddings."""

import json
import os
import re
from pathlib import Path

import fitz  # PyMuPDF is imported as "fitz"
import httpx
from dotenv import load_dotenv
from openai import APIConnectionError, AuthenticationError, OpenAI


# Using the script's location makes this work even when VS Code uses a
# different working directory.
DOCUMENTS_DIR = Path(__file__).resolve().parent / "documents"
ENV_FILE = Path(__file__).resolve().parent / ".env"
OUTPUT_FILE = Path(__file__).resolve().parent / "chunks.json"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
EMBEDDING_BATCH_SIZE = 32

# Product model names such as X3-ULT-15K or X3-ULT-19.9K.
# The header is detected by position, so this also works for similar model names.
MODEL_NAME_PATTERN = re.compile(r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9.]+){2,}\b")


def clean_text(text: str) -> str:
    """Replace repeated whitespace with single spaces."""
    return " ".join(text.split())


def normalize_parameter_name(parameter: str) -> str:
    """Correct known source-document spelling errors for better retrieval."""
    replacements = {
        "Demensions": "Dimensions",
        "Cetifications": "Certifications",
    }

    normalized = parameter
    for incorrect, correct in replacements.items():
        normalized = normalized.replace(incorrect, correct)
    return normalized


def find_model_headers(page: fitz.Page, table) -> list[dict]:
    """Find model names immediately above a table and record their x positions."""
    parameter_column_right_edges = [
        row.cells[0][2]
        for row in table.rows
        if row.cells and row.cells[0] is not None
    ]
    if not parameter_column_right_edges:
        return []

    data_start_x = min(parameter_column_right_edges)
    table_x0, table_y0, table_x1, _ = table.bbox
    header_area = fitz.Rect(
        data_start_x,
        max(0, table_y0 - 60),
        table_x1,
        table_y0,
    )

    models = []
    seen_names = set()
    for word in page.get_text("words", clip=header_area):
        x0, _, x1, _, text = word[:5]
        model_name = clean_text(text)

        if MODEL_NAME_PATTERN.fullmatch(model_name) and model_name not in seen_names:
            models.append({"name": model_name, "center_x": (x0 + x1) / 2})
            seen_names.add(model_name)

    models.sort(key=lambda model: model["center_x"])
    return models


def extract_table_chunks(
    page: fitz.Page,
    filename: str,
    page_number: int,
) -> list[dict]:
    """Turn PDF table rows into self-contained model/parameter chunks."""
    table_chunks = []
    tables = page.find_tables().tables

    for table_number, table in enumerate(tables, start=1):
        models = find_model_headers(page, table)

        # A table without model headers cannot be mapped safely by product model.
        if not models:
            continue

        section = "Uncategorized"

        for row_number, row in enumerate(table.rows, start=1):
            if not row.cells or row.cells[0] is None:
                continue

            parameter_cell = row.cells[0]
            source_parameter = clean_text(
                page.get_textbox(fitz.Rect(parameter_cell))
            )

            value_cells = [cell for cell in row.cells[1:] if cell is not None]

            # Section rows have an empty first cell and a title spanning the data area.
            if not source_parameter:
                if value_cells:
                    possible_section = clean_text(
                        page.get_textbox(fitz.Rect(value_cells[0]))
                    )
                    if possible_section:
                        section = possible_section
                continue

            parameter = normalize_parameter_name(source_parameter)

            for value_cell in value_cells:
                value = clean_text(page.get_textbox(fitz.Rect(value_cell)))
                if not value:
                    continue

                cell_x0, _, cell_x1, _ = value_cell
                covered_models = [
                    model
                    for model in models
                    if cell_x0 - 0.5 <= model["center_x"] <= cell_x1 + 0.5
                ]

                # A merged cell is expanded to every model column it covers.
                for model in covered_models:
                    safe_model_name = re.sub(r"[^A-Za-z0-9]+", "-", model["name"])
                    chunk_id = (
                        f"{Path(filename).stem}-p{page_number}-t{table_number}-"
                        f"r{row_number}-{safe_model_name}"
                    )
                    chunk_text = (
                        f"Document: {filename}\n"
                        f"Product model: {model['name']}\n"
                        f"Section: {section}\n"
                        f"Parameter: {parameter}\n"
                        f"Value: {value}"
                    )

                    table_chunks.append(
                        {
                            "filename": filename,
                            "page_number": page_number,
                            "chunk_id": chunk_id,
                            "content_type": "table_row",
                            "section": section,
                            "model": model["name"],
                            "parameter": parameter,
                            "source_parameter": source_parameter,
                            "value": value,
                            "text": chunk_text,
                        }
                    )

    return table_chunks


def extract_pdf_pages(documents_dir: Path) -> list[dict]:
    """Read every PDF and preserve structured tables when possible."""
    pages = []

    if not documents_dir.is_dir():
        print(f"Documents folder not found: {documents_dir}")
        return pages

    pdf_files = sorted(
        file_path
        for file_path in documents_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() == ".pdf"
    )

    if not pdf_files:
        print(f"No PDF files found in: {documents_dir}")
        return pages

    for pdf_path in pdf_files:
        print(f"Reading: {pdf_path.name}")

        # The context manager closes the PDF automatically after reading it.
        with fitz.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf):
                page_number = page_index + 1
                page_record = {
                    "filename": pdf_path.name,
                    "page_number": page_number,
                    "text": page.get_text("text").strip(),
                    "table_chunks": extract_table_chunks(
                        page,
                        pdf_path.name,
                        page_number,
                    ),
                }
                pages.append(page_record)

                if page_record["table_chunks"]:
                    print(
                        f"  Page {page_number}: created "
                        f"{len(page_record['table_chunks'])} structured table chunks"
                    )

    return pages


def split_pages_into_chunks(
    pages: list[dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Split page text into overlapping character-based chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least 0 and smaller than chunk_size")

    chunks = []
    step_size = chunk_size - overlap

    for page in pages:
        # Structured table chunks already contain model, parameter, and value.
        # Do not also add the flattened table text, which would hurt retrieval.
        if page.get("table_chunks"):
            chunks.extend(page["table_chunks"])
            continue

        page_text = page["text"]

        # Skip pages that contain no extractable text.
        if not page_text:
            continue

        for chunk_index, start in enumerate(range(0, len(page_text), step_size), start=1):
            chunk_text = page_text[start : start + chunk_size].strip()

            if not chunk_text:
                continue

            chunk = {
                "filename": page["filename"],
                "page_number": page["page_number"],
                "chunk_id": (
                    f"{page['filename']}-page-{page['page_number']}-chunk-{chunk_index}"
                ),
                "text": chunk_text,
            }
            chunks.append(chunk)

    return chunks


def print_debug_chunks(chunks: list[dict], count: int = 5) -> None:
    """Print the first few chunks for debugging."""
    print(f"\nCreated {len(chunks)} chunk(s) in total.")

    for chunk in chunks[:count]:
        print("\n" + "=" * 70)
        print(f"File: {chunk['filename']}")
        print(f"Page: {chunk['page_number']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print("-" * 70)
        print(chunk["text"])


def create_siliconflow_client() -> OpenAI:
    """Load settings from .env and create the SiliconFlow API client."""
    if not ENV_FILE.is_file():
        raise FileNotFoundError(f"Environment file not found: {ENV_FILE}")

    load_dotenv(dotenv_path=ENV_FILE, override=True)

    api_key = (os.getenv("SILICONFLOW_API_KEY") or "").strip()
    base_url = (
        os.getenv("SILICONFLOW_BASE_URL") or "https://api.siliconflow.cn/v1"
    ).strip()

    if not api_key:
        raise RuntimeError("SILICONFLOW_API_KEY is missing from the .env file")

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.Client(trust_env=False, timeout=60.0),
    )


def add_embeddings(chunks: list[dict], client: OpenAI) -> list[dict]:
    """Generate and attach SiliconFlow embeddings in small batches."""
    chunks_with_embeddings = []
    total_chunks = len(chunks)

    for start in range(0, total_chunks, EMBEDDING_BATCH_SIZE):
        batch = chunks[start : start + EMBEDDING_BATCH_SIZE]
        batch_end = start + len(batch)
        print(f"Creating embeddings {start + 1}-{batch_end}/{total_chunks}")

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[chunk["text"] for chunk in batch],
            encoding_format="float",
        )

        # Sort by API response index so vectors stay paired with their input chunks.
        embedding_results = sorted(response.data, key=lambda item: item.index)
        if len(embedding_results) != len(batch):
            raise RuntimeError("SiliconFlow returned an unexpected number of embeddings")

        for chunk, embedding_result in zip(batch, embedding_results):
            chunk_with_embedding = chunk.copy()
            chunk_with_embedding["embedding"] = embedding_result.embedding
            chunks_with_embeddings.append(chunk_with_embedding)

    return chunks_with_embeddings


def save_chunks(chunks: list[dict], output_file: Path) -> None:
    """Save all chunk metadata, text, and embeddings to a JSON file."""
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(chunks)} chunk(s) to: {output_file}")


if __name__ == "__main__":
    extracted_pages = extract_pdf_pages(DOCUMENTS_DIR)
    extracted_chunks = split_pages_into_chunks(extracted_pages)
    print_debug_chunks(extracted_chunks, count=5)

    if extracted_chunks:
        siliconflow_client = create_siliconflow_client()
        try:
            embedded_chunks = add_embeddings(extracted_chunks, siliconflow_client)
        except AuthenticationError as error:
            raise SystemExit(
                "SiliconFlow authentication failed. Check SILICONFLOW_API_KEY in .env."
            ) from error
        except APIConnectionError as error:
            raise SystemExit(
                "Could not connect to SiliconFlow. Check your DNS, VPN, proxy, "
                "or internet connection, then run ingest.py again."
            ) from error

        save_chunks(embedded_chunks, OUTPUT_FILE)
