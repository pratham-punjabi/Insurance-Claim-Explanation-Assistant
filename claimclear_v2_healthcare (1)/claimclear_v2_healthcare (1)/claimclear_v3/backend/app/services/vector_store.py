"""
ChromaDB Vector Store Service
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Handles:
  • Connecting to ChromaDB (cloud or local persist)
  • Chunking + embedding policy PDFs with sentence-transformers
  • Storing chunks with rich metadata
  • Semantic retrieval for RAG queries
  • Index status tracking
"""
from __future__ import annotations
import hashlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Lazy-loaded globals ────────────────────────────────────────────────────────
_chroma_client = None
_collection = None
_embedder = None

# Track which policies are indexed (policy_id -> chunk_count)
_index_status: dict[str, int] = {}


def _get_embedder():
    """Lazy-load sentence-transformers embedder (downloads model on first call)."""
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: all-MiniLM-L6-v2")
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Embedding model loaded.")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    return _embedder


def _get_client():
    """Lazy-load ChromaDB client based on config."""
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client

    from app.core.config import settings
    import chromadb

    if settings.use_cloud_chroma:
        logger.info(f"Connecting to ChromaDB Cloud: {settings.chroma_host}:{settings.chroma_port}")
        _chroma_client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
            ssl=settings.chroma_ssl,
            headers={"X-Chroma-Token": settings.chroma_api_key} if settings.chroma_api_key else {},
        )
    else:
        persist_dir = Path(settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using local ChromaDB at: {persist_dir}")
        _chroma_client = chromadb.PersistentClient(path=str(persist_dir))

    return _chroma_client


def _get_collection():
    """Get or create the ChromaDB collection."""
    global _collection
    if _collection is not None:
        return _collection

    from app.core.config import settings
    import chromadb

    client = _get_client()
    _collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"description": "Insurance policy document chunks for RAG"},
    )
    logger.info(f"Collection '{settings.chroma_collection_name}' ready. Count: {_collection.count()}")

    # Rebuild index status from existing data
    _rebuild_index_status()
    return _collection


def _rebuild_index_status():
    """Rebuild in-memory index status from ChromaDB metadata.
    Called once when the collection is first initialized so that
    policies already stored on disk are not re-indexed on every restart.
    """
    global _index_status
    try:
        col = _get_collection()
        total = col.count()
        if total == 0:
            _index_status = {}
            logger.info("ChromaDB collection is empty — nothing to restore.")
            return
        results = col.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for meta in results.get("metadatas", []):
            pid = meta.get("policy_id", "")
            if pid:
                counts[pid] = counts.get(pid, 0) + 1
        _index_status = counts
        logger.info(f"Restored index status from ChromaDB ({total} chunks): {_index_status}")
    except Exception as e:
        logger.warning(f"Could not rebuild index status: {e}")


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using sentence-transformers."""
    model = _get_embedder()
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
    return embeddings.tolist()


def _chunk_id(policy_id: str, chunk_index: int, text: str) -> str:
    """Deterministic chunk ID based on content hash."""
    h = hashlib.md5(f"{policy_id}:{chunk_index}:{text[:100]}".encode()).hexdigest()[:8]
    return f"{policy_id}_chunk_{chunk_index:04d}_{h}"


# ── Public API ────────────────────────────────────────────────────────────────

def is_indexed(policy_id: str) -> bool:
    """Return True if this policy has been indexed in ChromaDB."""
    return _index_status.get(policy_id, 0) > 0


def index_policy(
    policy_id: str,
    policy_name: str,
    insurer: str,
    policy_type: str,
    chunks: list[str],
    force: bool = False,
) -> int:
    """
    Embed and store policy chunks in ChromaDB.
    Returns number of chunks stored.
    Skips if already indexed (unless force=True).
    """
    if is_indexed(policy_id) and not force:
        logger.info(f"Policy '{policy_id}' already indexed ({_index_status[policy_id]} chunks). Skipping.")
        return _index_status[policy_id]

    col = _get_collection()

    # Delete existing chunks for this policy if re-indexing
    if force and is_indexed(policy_id):
        logger.info(f"Re-indexing: removing old chunks for '{policy_id}'")
        existing = col.get(where={"policy_id": policy_id})
        if existing["ids"]:
            col.delete(ids=existing["ids"])

    if not chunks:
        logger.warning(f"No chunks to index for policy '{policy_id}'")
        return 0

    logger.info(f"Indexing {len(chunks)} chunks for policy '{policy_id}'…")
    embeddings = _embed(chunks)

    ids = [_chunk_id(policy_id, i, c) for i, c in enumerate(chunks)]
    metadatas = [
        {
            "policy_id": policy_id,
            "policy_name": policy_name,
            "insurer": insurer,
            "policy_type": policy_type,
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]

    # ChromaDB upsert in batches of 100.
    # upsert (not add) silently overwrites existing IDs, eliminating
    # the "duplicate ID" warning that add() raises when the collection
    # already contains chunks from a previous run.
    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        col.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=chunks[start:end],
            metadatas=metadatas[start:end],
        )

    _index_status[policy_id] = len(chunks)
    logger.info(f"Indexed {len(chunks)} chunks for '{policy_id}'")
    return len(chunks)


def retrieve(
    query: str,
    policy_ids: list[str] | None = None,
    n_results: int = 5,
) -> list[dict[str, Any]]:
    """
    Semantic search over indexed policy chunks.
    Returns list of {text, policy_id, policy_name, score} dicts.

    Args:
        query: Natural language query
        policy_ids: Filter to specific policies (None = search all)
        n_results: Number of top chunks to return
    """
    col = _get_collection()

    if col.count() == 0:
        logger.warning("ChromaDB collection is empty — no policies indexed yet.")
        return []

    query_embedding = _embed([query])[0]

    where_filter = None
    if policy_ids and len(policy_ids) == 1:
        where_filter = {"policy_id": policy_ids[0]}
    elif policy_ids and len(policy_ids) > 1:
        where_filter = {"policy_id": {"$in": policy_ids}}

    kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": min(n_results, max(col.count(), 1)),
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        kwargs["where"] = where_filter

    results = col.query(**kwargs)

    output = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        output.append({
            "text": doc,
            "policy_id": meta.get("policy_id", ""),
            "policy_name": meta.get("policy_name", ""),
            "insurer": meta.get("insurer", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "score": round(1 - dist, 4),   # cosine similarity approx
        })

    return output


def get_index_stats() -> dict[str, Any]:
    """Return stats about what's indexed."""
    try:
        col = _get_collection()
        return {
            "total_chunks": col.count(),
            "policies": _index_status,
            "collection": col.name,
        }
    except Exception as e:
        return {"error": str(e), "total_chunks": 0, "policies": {}}


def delete_policy_index(policy_id: str) -> int:
    """Remove all chunks for a policy from ChromaDB."""
    col = _get_collection()
    existing = col.get(where={"policy_id": policy_id})
    if existing["ids"]:
        col.delete(ids=existing["ids"])
        count = len(existing["ids"])
        _index_status.pop(policy_id, None)
        logger.info(f"Deleted {count} chunks for '{policy_id}'")
        return count
    return 0
