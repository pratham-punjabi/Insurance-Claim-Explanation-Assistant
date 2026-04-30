"""
Policy Indexer
━━━━━━━━━━━━━━
Orchestrates the full pipeline:
  PDF → OCR text → chunks → embeddings → ChromaDB

Called at startup (auto-index all) and via /api/index endpoint.
"""
from __future__ import annotations
import logging
from pathlib import Path

from app.core.config import settings
from app.models.schemas import IndexResponse
from app.services.pdf_extractor import extract_text_from_pdf, chunk_text
from app.services.policy_registry import get_all_policies, resolve_pdf_path
from app.services import vector_store

logger = logging.getLogger(__name__)


async def index_all_policies(force: bool = False) -> IndexResponse:
    """Index all policies from the registry."""
    policies = get_all_policies()
    policy_ids = [p.id for p in policies]
    return await index_policies(policy_ids, force=force)


async def index_policies(policy_ids: list[str], force: bool = False) -> IndexResponse:
    """Index specific policies by ID."""
    from app.services.policy_registry import get_policy

    indexed: list[str] = []
    skipped: list[str] = []
    errors: dict[str, str] = {}
    total_chunks = 0

    for policy_id in policy_ids:
        info = get_policy(policy_id)
        if not info:
            errors[policy_id] = "Policy not found in registry"
            continue

        # Skip if already indexed and not forced
        if vector_store.is_indexed(policy_id) and not force:
            logger.info(f"Skipping '{policy_id}' — already indexed")
            skipped.append(policy_id)
            continue

        pdf_path = resolve_pdf_path(info.filename, settings.policies_path)
        if not pdf_path.exists():
            errors[policy_id] = f"PDF not found: {info.filename}"
            logger.error(f"PDF not found: {pdf_path}")
            continue

        try:
            logger.info(f"Processing {info.name}…")

            # Step 1: Extract text (OCR if needed)
            full_text = extract_text_from_pdf(pdf_path)
            if not full_text.strip():
                errors[policy_id] = "No text could be extracted from PDF"
                continue

            # Step 2: Chunk text
            chunks = chunk_text(full_text, chunk_size=800, overlap=150)
            if not chunks:
                errors[policy_id] = "No chunks generated from extracted text"
                continue

            logger.info(f"Generated {len(chunks)} chunks for '{policy_id}'")

            # Step 3: Embed + store in ChromaDB
            n = vector_store.index_policy(
                policy_id=policy_id,
                policy_name=info.name,
                insurer=info.insurer,
                policy_type=info.policy_type,
                chunks=chunks,
                force=force,
            )
            total_chunks += n
            indexed.append(policy_id)
            logger.info(f"✓ Indexed '{policy_id}': {n} chunks stored")

        except Exception as e:
            logger.error(f"Error indexing '{policy_id}': {e}", exc_info=True)
            errors[policy_id] = str(e)

    return IndexResponse(
        indexed=indexed,
        skipped=skipped,
        errors=errors,
        total_chunks=total_chunks,
    )
