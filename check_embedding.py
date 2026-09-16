import json

with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(len(chunks[0]["embedding"]))