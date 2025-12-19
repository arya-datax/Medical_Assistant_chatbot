from src.helper import load_pdf, text_split, download_embedding_model
from pinecone import Pinecone, ServerlessSpec
from pinecone import Pinecone  
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os


# loading key from .env
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
llm_key = os.getenv("LLM_API_KEY")


extracted_data = load_pdf("Data")
text_chunks = text_split(extracted_data)
embedding = download_embedding_model()

# Pinecone creation and calculating embedding fro chunks and upset in Pinecone database
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=pinecone_api_key)

index_name = "medical-bot-minillm"
dimension = 384  # all-MiniLM-L6-v2

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# calculating embedding
texts = [chunk.page_content for chunk in text_chunks]
embedding_text = embedding.client.encode(texts, convert_to_numpy=True)

# Batch Upsert
index = pc.Index(index_name)

vectors = [
    (
        f"chunk_{i}",          # vector id
        embedding_text[i].tolist(),  # embedding vector
        {"text": texts[i]} # metadata
    )
    for i in range(len(text_chunks))
]

# =========================
# 8. Batch upsert (FAST)
# =========================
def batch_upsert(index, vectors, batch_size=200):
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i + batch_size])

batch_upsert(index, vectors, batch_size=200)

# =========================
# 9. Verify insertion
# =========================
stats = index.describe_index_stats()
print("Index stats:", stats)

print("✅ Embeddings successfully stored in Pinecone")
