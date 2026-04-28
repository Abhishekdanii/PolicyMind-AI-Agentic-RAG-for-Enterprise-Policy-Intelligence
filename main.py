from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
import anthropic
import PyPDF2
import io
import uuid
import os
import re

# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(title="Company Policy Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── ChromaDB setup ─────────────────────────────────────────────────────────
CHROMA_PATH = "./chroma_store"
COLLECTION_NAME = "company_policies"

# Use default sentence-transformer embedding
embedding_fn = embedding_functions.DefaultEmbeddingFunction()

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Create or get collection on startup
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"},
)

print(f"✅ ChromaDB initialized at '{CHROMA_PATH}' | Collection: '{COLLECTION_NAME}'")
print(f"   Documents in DB: {collection.count()}")

# ── Anthropic client ───────────────────────────────────────────────────────
anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


# ── Helpers ────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    """Split text into overlapping chunks."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 30]


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract text from PDF or plain-text file."""
    if filename.lower().endswith(".pdf"):
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return "\n".join(
            page.extract_text() or "" for page in reader.pages
        )
    return file_bytes.decode("utf-8", errors="ignore")


# ── Request models ─────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str


# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "docs_in_db": collection.count()}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Accept a PDF or TXT file, chunk it, and store every chunk
    in ChromaDB with its metadata.
    """
    allowed = {".pdf", ".txt"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Only {allowed} files are supported.")

    raw = await file.read()
    text = extract_text(raw, file.filename)

    if not text.strip():
        raise HTTPException(422, "Could not extract any text from the file.")

    chunks = chunk_text(text)

    ids, docs, metas = [], [], []
    for i, chunk in enumerate(chunks):
        ids.append(str(uuid.uuid4()))
        docs.append(chunk)
        metas.append({"source": file.filename, "chunk_index": i})

    # Upsert in batches of 100
    BATCH = 100
    for b in range(0, len(ids), BATCH):
        collection.upsert(
            ids=ids[b : b + BATCH],
            documents=docs[b : b + BATCH],
            metadatas=metas[b : b + BATCH],
        )

    return {
        "message": "File ingested successfully.",
        "filename": file.filename,
        "chunks_stored": len(chunks),
        "total_docs_in_db": collection.count(),
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    1. Similarity-search ChromaDB for top-3 matching chunks.
    2. Build a RAG prompt and call Claude.
    3. Return the answer + source chunks.
    """
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty.")

    if collection.count() == 0:
        raise HTTPException(404, "No documents in the knowledge base yet. Please upload a policy file first.")

    # ── Vector search ──────────────────────────────────────────────────────
    results = collection.query(
        query_texts=[req.query],
        n_results=min(3, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    top_chunks = results["documents"][0]        # list of 3 strings
    top_metas  = results["metadatas"][0]        # list of 3 dicts
    top_scores = results["distances"][0]        # cosine distances (lower = better)

    context_block = "\n\n".join(
        f"[Excerpt {i+1} | source: {m['source']}]\n{doc}"
        for i, (doc, m) in enumerate(zip(top_chunks, top_metas))
    )

    # ── Static RAG system prompt ───────────────────────────────────────────
    system_prompt = (
        "You are a helpful HR assistant for a company. "
        "Your job is to answer employee questions strictly based on the company policy documents provided. "
        "You will be given 3 relevant excerpts retrieved from the policy knowledge base. "
        "Use ONLY these excerpts to compose your answer — do not add information from outside sources. "
        "If the excerpts do not contain enough information to answer the question, say so honestly. "
        "Keep your answer concise, professional, and easy to understand. "
        "Always cite which excerpt(s) you used (e.g. 'According to Excerpt 2...')."
    )

    user_message = (
        f"Employee question: {req.query}\n\n"
        f"Relevant policy excerpts retrieved from the vector database:\n\n"
        f"{context_block}\n\n"
        "Please answer the employee's question using the excerpts above."
    )

    # ── Call Claude ────────────────────────────────────────────────────────
    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        answer = response.content[0].text
    except Exception as e:
        raise HTTPException(500, f"LLM call failed: {str(e)}")

    # ── Return ─────────────────────────────────────────────────────────────
    return {
        "answer": answer,
        "sources": [
            {
                "text": doc,
                "source": meta["source"],
                "chunk_index": meta["chunk_index"],
                "similarity_score": round(1 - score, 4),   # convert distance → similarity
            }
            for doc, meta, score in zip(top_chunks, top_metas, top_scores)
        ],
    }


@app.get("/api/db-stats")
def db_stats():
    return {"total_documents": collection.count(), "collection": COLLECTION_NAME}
