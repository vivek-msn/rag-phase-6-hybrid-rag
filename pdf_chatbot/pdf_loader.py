from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# Configuration
PDF_PATH ="rag_guide.pdf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# PDF Text Extraction

reader = PdfReader(PDF_PATH)

print("Number of pages:", len(reader.pages))

full_text = ""

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text()

    if text:
        full_text += text + "\n"

# print(f"\n--- Full PDF Text ---")
# print(full_text)

# Text Chunking

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

chunks = text_splitter.split_text(full_text)

print("\nNumber of chunks:", len(chunks))


# Generate Embeddings

model = SentenceTransformer(EMBEDDING_MODEL)

embeddings = model.encode(chunks)

print("\nEmbedding shape:", embeddings.shape)


# ChromaDB

client = chromadb.PersistentClient(
    path="pdf_chroma_db"
    )

collection = client.get_or_create_collection(
    name="pdf_documents"
)

# Store Chunks in ChromaDB

ids = [f"chunk_{i}" for i in range(len(chunks))]

metadatas = [
    {
        "source": PDF_PATH,
        "chunk_index": i
    }
    for i in range(len(chunks))
]

collection.add(
    ids=ids,
    documents=chunks,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)

print("\nTotal documents in chromaDB:", collection.count())

# Display Chunks
# for i, chunk in enumerate(chunks, start=1):
#     print(f"\n--- Chunk {i} ---")
#     print(chunk)