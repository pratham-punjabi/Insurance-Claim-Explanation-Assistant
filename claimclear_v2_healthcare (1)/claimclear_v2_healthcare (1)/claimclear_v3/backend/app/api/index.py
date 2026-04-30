"""
Index management endpoints
  POST   /api/index        — manually trigger (re)indexing
  GET    /api/index/stats  — current ChromaDB stats
  DELETE /api/index/{id}   — remove a policy from the index
"""
from fastapi import APIRouter, HTTPException
from app.models.schemas import IndexRequest, IndexResponse
from app.services.indexer import index_all_policies, index_policies
from app.services import vector_store

router = APIRouter(prefix="/index", tags=["index"])


@router.post("", response_model=IndexResponse)
async def trigger_index(body: IndexRequest):
    """
    Trigger indexing of all or specific policies.
    With force_reindex=False (default): skips already-indexed policies.
    With force_reindex=True: re-chunks and re-embeds everything.
    """
    try:
        if body.policy_ids:
            return await index_policies(body.policy_ids, force=body.force_reindex)
        return await index_all_policies(force=body.force_reindex)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def index_stats():
    """Get current ChromaDB index statistics."""
    return vector_store.get_index_stats()


@router.delete("/{policy_id}")
async def delete_policy_index(policy_id: str):
    """Remove a specific policy's chunks from ChromaDB."""
    try:
        count = vector_store.delete_policy_index(policy_id)
        return {"policy_id": policy_id, "chunks_deleted": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
