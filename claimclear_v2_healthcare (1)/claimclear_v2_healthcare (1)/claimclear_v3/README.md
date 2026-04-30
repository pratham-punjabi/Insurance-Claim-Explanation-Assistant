# ClaimClear v2 — Healthcare Insurance Assistant

AI-powered healthcare insurance claim assistant. **Healthcare domain only.**

## How Indexing Works (Important!)

```
First run:
  PDF files → text extraction (PyMuPDF / OCR) → chunk into ~800 char pieces
           → embed with sentence-transformers (all-MiniLM-L6-v2)
           → store in ChromaDB at data/chroma_db/

Every subsequent run:
  ChromaDB already has the chunks on disk →
  _rebuild_index_status() sees them → is_indexed() = True →
  ALL policies SKIPPED (no re-chunking, no re-embedding)
```

**So: heavy work runs ONCE. Every restart after that is instant.**

To force re-index: `POST /api/index` with `{"force_reindex": true}`

## Quick Start

```bash
cd backend
pip install -r requirements.txt
# Edit .env — set GROQ_API_KEY
uvicorn main:app --reload --port 8000
# First run: chunks + embeds all 5 policy PDFs (~30-60 seconds)
# All future runs: index is reused, starts in seconds
```

Open: http://localhost:8000

## Features

| Feature | Description |
|---|---|
| Dashboard | Stats + recent health claims |
| My Claims | All claims, filters (All/Approved/Denied/Pending), counts |
| Explain Denial | RAG-powered AI analysis + appeal steps |
| AI Chat | Multi-turn health insurance chat with RAG context |
| Compare Policies | Upload 2–5 PDFs → AI comparison table + recommendation |
| Claim Predictor | AI predicts claim approval/denial likelihood |

## Architecture

- **Backend**: FastAPI + Groq LLM (llama-3.3-70b-versatile)
- **Vector DB**: ChromaDB (local persistent at `data/chroma_db/`)
- **Embeddings**: sentence-transformers `all-MiniLM-L6-v2` (one-time)
- **PDF Extraction**: PyMuPDF + OCR fallback (pytesseract)
- **Frontend**: Single-file vanilla HTML/CSS/JS
