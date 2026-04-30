# ClaimClear — Insurance Claim Assistant Backend

AI-powered insurance claim assistant using **Groq LLM** + **ChromaDB RAG** + **FastAPI**.

## Architecture

```
frontend/index.html  ←→  FastAPI (main.py)
                              ├── /api/claims      — CRUD + AI explanation
                              ├── /api/policies    — Policy list + RAG Q&A
                              ├── /api/chat        — Multi-turn RAG chat
                              └── /api/index       — ChromaDB indexing
                                      │
                              Groq (llama3-70b-8192)
                              ChromaDB (cloud/local)
                              sentence-transformers (all-MiniLM-L6-v2)
                              PyMuPDF + pytesseract (OCR)
```

## Quick Start

### 1. Clone and install
```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and fill in:
#   GROQ_API_KEY=gsk_...
#   CHROMA_HOST=... (or leave empty for local ChromaDB)
#   CHROMA_API_KEY=... (for cloud ChromaDB)
```

### 3. Run
```bash
uvicorn main:app --reload --port 8000
```

On first run, the backend will:
1. Connect to ChromaDB
2. OCR all 5 policy PDFs in `data/policies/`
3. Chunk text into ~800-char overlapping segments
4. Embed with `sentence-transformers/all-MiniLM-L6-v2`
5. Store in ChromaDB collection `insurance_policies`

This takes **2–5 minutes** on first run. Subsequent restarts are instant (cached).

### 4. Open frontend
Open `frontend/index.html` in your browser (or serve via `http://localhost:8000/`).

## API Docs
Visit `http://localhost:8000/docs` for interactive Swagger UI.

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `GROQ_API_KEY` | Your Groq API key | `gsk_...` |
| `GROQ_MODEL` | Groq model ID | `llama3-70b-8192` |
| `CHROMA_HOST` | ChromaDB Cloud host (empty = local) | `api.trychroma.com` |
| `CHROMA_PORT` | ChromaDB port | `443` |
| `CHROMA_SSL` | Use SSL for ChromaDB | `true` |
| `CHROMA_API_KEY` | ChromaDB Cloud API key | `ck-...` |
| `CHROMA_PERSIST_DIR` | Local ChromaDB path | `./data/chroma_db` |
| `CHROMA_COLLECTION_NAME` | Collection name | `insurance_policies` |
| `POLICIES_DIR` | Path to policy PDFs | `data/policies` |

## Key Features

- **RAG Q&A**: All policy questions are answered using semantic search over indexed PDFs
- **Guardrails**: Off-topic queries (food, sports, coding) are blocked before hitting the LLM
- **Claim Explanation**: Structured JSON response with denial reasons, appeal steps, required documents
- **Appeal Letter**: IRDAI-compliant formal appeal draft
- **Multi-turn Chat**: Conversation history maintained per session
- **Auto-indexing**: PDFs indexed at startup, skipped if already indexed

## Folder Structure

```
backend/
├── main.py                     # FastAPI entry point
├── requirements.txt
├── .env.example
├── data/
│   └── policies/               # Insurance PDF files
├── app/
│   ├── api/
│   │   ├── claims.py           # /api/claims routes
│   │   ├── policies.py         # /api/policies routes
│   │   ├── chat.py             # /api/chat route
│   │   └── index.py            # /api/index routes
│   ├── core/
│   │   └── config.py           # Settings (pydantic-settings)
│   ├── models/
│   │   └── schemas.py          # All Pydantic schemas
│   └── services/
│       ├── ai_service.py       # Groq LLM calls + RAG
│       ├── vector_store.py     # ChromaDB operations
│       ├── pdf_extractor.py    # OCR text extraction
│       ├── indexer.py          # PDF → chunks → ChromaDB pipeline
│       ├── policy_registry.py  # Policy metadata registry
│       └── guardrails.py       # Topic enforcement
└── tests/
    └── test_api.py
```
