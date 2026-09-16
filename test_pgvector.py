from query import create_siliconflow_client, create_query_embedding, search_top_chunks

client = create_siliconflow_client()

question = "What is the warranty period?"

query_embedding = create_query_embedding(question, client)

results = search_top_chunks(query_embedding)

print(results)