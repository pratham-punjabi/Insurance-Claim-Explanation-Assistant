"""
ClaimClear — Healthcare Insurance Claim Assistant API
=====================================================
FastAPI application entry point.

Startup sequence:
  1. Load config from .env
  2. Connect to ChromaDB (local persist or cloud)
  3. Auto-index all policy PDFs  ← RUNS ONLY ONCE
       • ChromaDB stores chunks on disk (data/chroma_db/)
       • On every subsequent restart, _rebuild_index_status() reads
         existing chunk metadata from disk, so is_indexed() returns
         True for all already-stored policies — they are SKIPPED.
       • Re-indexing only happens if you set force=True via /api/index
         or manually delete the chroma_db folder.
  4. Register all routers
  5. Serve API

Run:
  uvicorn main:app --reload --port 8000
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("claimclear")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 62)
    logger.info("  ClaimClear AI — Healthcare Insurance Assistant")
    logger.info("=" * 62)
    logger.info(f"  Environment : {settings.app_env}")
    logger.info(f"  Groq Model  : {settings.groq_model}")
    logger.info(f"  ChromaDB    : {'Cloud → ' + settings.chroma_host if settings.use_cloud_chroma else 'Local → ' + settings.chroma_persist_dir}")
    logger.info(f"  Policies dir: {settings.policies_dir}")
    logger.info("-" * 62)
    logger.info("  NOTE: PDFs are chunked & embedded ONCE on first run.")
    logger.info("  On subsequent restarts, existing index is reused.")
    logger.info("  To force re-index: POST /api/index with force_reindex=true")
    logger.info("=" * 62)

    # ── One-time policy indexing ───────────────────────────────────────────────
    # ChromaDB persists chunks to disk (data/chroma_db/).
    # _rebuild_index_status() inside vector_store reads existing metadata
    # so is_indexed() returns True for already-stored policies → they skip.
    # Net result: chunking + embedding runs ONLY for new / not-yet-indexed PDFs.
    try:
        from app.services.indexer import index_all_policies
        logger.info("Checking policy index…")
        result = await index_all_policies(force=False)   # <-- force=False is key
        if result.indexed:
            logger.info(f"  ✓ Newly indexed : {result.indexed} ({result.total_chunks} chunks)")
        if result.skipped:
            logger.info(f"  ↷ Already indexed (skipped): {result.skipped}")
        if result.errors:
            logger.warning(f"  ✗ Errors: {result.errors}")
        logger.info("Policy index ready.")
    except Exception as e:
        logger.error(f"Startup indexing failed (app will still run): {e}")

    yield

    logger.info("ClaimClear AI — Shutting down.")


app = FastAPI(
    title="ClaimClear — Healthcare Insurance Assistant API",
    description=(
        "AI-powered healthcare insurance claim assistant using Groq LLM + ChromaDB RAG. "
        "Healthcare domain only. "
        "Explains health policies, analyses denials, compares policies, predicts claim outcomes."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.api.claims   import router as claims_router
from app.api.policies import router as policies_router
from app.api.chat     import router as chat_router
from app.api.index    import router as index_router

app.include_router(claims_router,   prefix="/api")
app.include_router(policies_router, prefix="/api")
app.include_router(chat_router,     prefix="/api")
app.include_router(index_router,    prefix="/api")

frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(str(frontend_dir / "index.html"))


@app.get("/health", tags=["health"])
async def health():
    from app.services import vector_store
    stats = vector_store.get_index_stats()
    return {
        "status": "ok",
        "domain": "Healthcare Insurance Only",
        "groq_model": settings.groq_model,
        "chroma_mode": "cloud" if settings.use_cloud_chroma else "local",
        "index": stats,
    }
