import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase = create_client(url, key)

response = supabase.table("documents").select("*").execute()

print(response.data)
response = (
    supabase
    .table("documents")
    .insert({
        "filename": "test.pdf"
    })
    .execute()
)

print(response.data)