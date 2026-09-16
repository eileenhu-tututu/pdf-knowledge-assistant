import json
import os

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

filename = chunks[0]["filename"]

document_result = (
    supabase
    .table("documents")
    .insert({
        "filename": filename
    })
    .execute()
)

document_id = document_result.data[0]["id"]

for index, chunk in enumerate(chunks, start=1):

    supabase.table("document_chunks").insert({
        "document_id": document_id,
        "page_number": chunk["page_number"],
        "chunk_index": index,
        "content": chunk["text"],
        "embedding": chunk["embedding"]
    }).execute()

print("Migration complete.")