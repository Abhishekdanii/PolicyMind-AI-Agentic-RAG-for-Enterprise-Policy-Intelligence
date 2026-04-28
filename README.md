# PolicyMind – Company Policy Chatbot

A RAG (Retrieval-Augmented Generation) chatbot for company policies, powered by ChromaDB + Claude.

## Architecture

```
Frontend (index.html)
    │
    ├── POST /api/upload  → chunk PDF/TXT → store in ChromaDB
    └── POST /api/chat    → vector search → top-3 chunks → Claude LLM → answer
```

## Quick Start

### 1. Backend

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...

# Start the server
uvicorn main:app --reload --port 8000
```

ChromaDB data is persisted to `./chroma_store/` automatically on first run.

### 2. Frontend

Just open `frontend/index.html` in your browser — no build step needed.

Or serve it:
```bash
cd frontend
python3 -m http.server 3000
# Visit http://localhost:3000
```

---

## API Reference

### `GET /`
Health check + doc count.

### `POST /api/upload`
Upload and index a policy file.

- **Body**: `multipart/form-data` with field `file` (PDF or TXT)
- **Response**:
```json
{
  "message": "File ingested successfully.",
  "filename": "employee_handbook.pdf",
  "chunks_stored": 47,
  "total_docs_in_db": 47
}
```

### `POST /api/chat`
Ask a question about policies.

- **Body**: `{ "query": "How many casual leaves do I get?" }`
- **Response**:
```json
{
  "answer": "According to Excerpt 1, employees are entitled to 12 casual leaves per year...",
  "sources": [
    {
      "text": "Employees are entitled to 12 casual leaves...",
      "source": "employee_handbook.pdf",
      "chunk_index": 5,
      "similarity_score": 0.91
    }
  ]
}
```

### `GET /api/db-stats`
Returns total chunk count in ChromaDB.

---

## How it works

1. **Upload** – File is chunked into 500-char overlapping segments, embedded via ChromaDB's default sentence-transformer, and stored.
2. **Chat** – User query is embedded and similarity-searched against the DB (top 3 chunks).
3. **RAG** – The 3 chunks are passed as context to Claude with a static system prompt instructing it to answer only from the provided excerpts.
4. **Answer** – Claude returns a cited, professional answer shown in the chat UI.

## Project Structure

```
├── backend/
│   ├── main.py           # FastAPI app
│   ├── requirements.txt
│   └── chroma_store/     # Auto-created, persisted vector DB
└── frontend/
    └── index.html        # Single-file React-style UI
```
