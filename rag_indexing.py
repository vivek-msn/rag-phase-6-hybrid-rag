import chromadb
import os
from dotenv import load_dotenv
from google import genai
from sentence_transformers import CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client_gemini = genai.Client(api_key=api_key)

document = """
Retrieval-Augmented Generation, also known as RAG,
is a technique that improves the responses generated
by Large Language Models.

RAG first retrieves relevant information from a
knowledge base or vector database.

The retrieved information is then provided to the
Large Language Model as additional context.

This allows the model to generate more accurate and
context-aware responses.
"""

# print(document)

text_splitter =  RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30
)

chunks =text_splitter.split_text(document)

for i, chunk in enumerate(chunks):
    print(f"\n-- Chunk {i} ---")
    print(chunk)

# chunks = [
#     document[i:i + chunk_size]
#     for i in range(0, len(document), chunk_size)
# ]

ids = [
    f"chunk_{i}"
    for i in range(len(chunks))
]

metadatas = [
    {
        "source": "rag_notes",
        "chunk_number": i
    }
    for i in range(len(chunks))
]

# print(metadatas)

client = chromadb.PersistentClient(
    path="./rag_chroma_db"
)

# collection = client.get_or_create_collection(
#     name="rag_documents"
# )

# Remove the old collection containing broken chunks

try:
    client.delete_collection("rag_documents")
except Exception:
    pass

# Create a fresh collection for the new chunks
collection = client.get_or_create_collection(
    name="rag_documents"
)

collection.add(
    ids=ids,
    documents=chunks,
    metadatas=metadatas
)

data = collection.get(
    include=["documents", "metadatas", "embeddings"]
)

model = CrossEncoder("BAAI/bge-reranker-base")

query = "How Does RAG improve LLM resposne?"

n_results = 4

results =  collection.query(
    query_texts=[query],
    n_results=n_results
)

retrieved_chunks = results["documents"][0]

pairs = [(query, chunk) for chunk in retrieved_chunks]

scores = model.predict(pairs)

ranked = list(zip(retrieved_chunks, scores))

ranked = sorted(
    ranked,
    key=lambda x:x[1],
    reverse=True
)

top_results = ranked[:2]

reranked_chunks = [chunk for chunk,score in top_results]

context = "\n\n".join(reranked_chunks)

for i, (chunk, score) in enumerate(top_results):
    print(f"\n -- Reranked Result {i + 1} --")
    print("Score:", score)
    print("Chunk:", chunk)

# for i in range(n_results):

#     print(f"\n--Result {i + 1} ---")

#     print("ID:", results["ids"][0][i])
#     print("Documents:", results["documents"][0][i])
#     print("Distance:", results["distances"][0][i])
#     print("MetaData:", results["metadatas"][0][i])

# context = "\n\n".join(retrieved_chunks)

# print(retrieved_chunks)

prompt = f"""
Answer the questions using only the provided context.

Context:
{context}

Question:
{query}

Answer:
"""

response = client_gemini.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents=prompt
)

print("\nGenerated Answer:\n")
print(response.text)
print("\nRAG Prompt:\n")
print(prompt)