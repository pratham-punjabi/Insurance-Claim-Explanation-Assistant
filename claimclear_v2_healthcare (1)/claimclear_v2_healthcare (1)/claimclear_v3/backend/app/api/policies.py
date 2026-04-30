"""Healthcare Policy routes + Policy Comparison (upload-based or by name)"""
import tempfile
import os
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List
from pydantic import BaseModel
from app.models.schemas import (
    PolicyExplainRequest, PolicyExplainResponse,
    PolicyInfo, PolicyListResponse, PolicyCompareResponse,
)
from app.services.policy_registry import get_all_policies, get_policy
from app.services.ai_service import explain_policy, compare_policies, compare_policies_by_name, _extract_pdf_text
from app.services import vector_store

router = APIRouter(prefix="/policies", tags=["policies"])

MAX_COMPARE = 5


@router.get("", response_model=PolicyListResponse)
async def list_policies():
    """List all registered healthcare insurance policies with index status."""
    policies = get_all_policies()
    stats = vector_store.get_index_stats()
    indexed_ids = set(stats.get("policies", {}).keys())
    for p in policies:
        p.indexed = p.id in indexed_ids
    return PolicyListResponse(policies=policies)


@router.get("/{policy_id}", response_model=PolicyInfo)
async def get_one_policy(policy_id: str):
    info = get_policy(policy_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Policy '{policy_id}' not found")
    return info


@router.post("/{policy_id}/explain", response_model=PolicyExplainResponse)
async def explain_policy_endpoint(policy_id: str, body: PolicyExplainRequest):
    """Ask a question about a registered health policy using RAG."""
    if not get_policy(policy_id):
        raise HTTPException(status_code=404, detail=f"Policy '{policy_id}' not found")
    try:
        return await explain_policy(policy_id, body.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", response_model=PolicyCompareResponse)
async def compare_policies_endpoint(
    files: List[UploadFile] = File(...),
    user_condition: str = Form(default=""),
):
    """
    Upload 2–5 healthcare policy PDFs to compare.
    AI extracts text directly from the uploaded files (no indexing needed).
    """
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Upload at least 2 healthcare policy PDFs to compare.")
    if len(files) > MAX_COMPARE:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_COMPARE} PDFs allowed.")

    for f in files:
        if not f.filename or not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"'{f.filename}' is not a PDF. Upload healthcare PDFs only.")

    pdf_texts: dict[str, str] = {}
    tmp_files: list[str] = []

    try:
        for upload in files:
            content = await upload.read()
            if len(content) > 20 * 1024 * 1024:
                raise HTTPException(status_code=400, detail=f"'{upload.filename}' exceeds 20 MB.")

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            tmp_files.append(tmp_path)

            policy_name = Path(upload.filename).stem.replace("_", " ").replace("-", " ")
            text = _extract_pdf_text(tmp_path, max_chars=10000)

            if not text.strip():
                raise HTTPException(status_code=400, detail=f"Could not extract text from '{upload.filename}'. Ensure it's a text-based PDF.")

            # Basic content validation — must look like a health insurance document
            text_lower = text.lower()
            health_kws = ["health", "hospitalisation", "hospitalization", "medical", "illness",
                          "treatment", "insured", "premium", "policy", "coverage", "claim",
                          "hospital", "surgery", "diagnosis"]
            if not any(kw in text_lower for kw in health_kws):
                raise HTTPException(
                    status_code=400,
                    detail=f"'{upload.filename}' doesn't appear to be a healthcare insurance policy. Upload health insurance PDFs only."
                )

            pdf_texts[policy_name] = text

        return await compare_policies(pdf_texts, user_condition=user_condition)

    finally:
        for p in tmp_files:
            try:
                os.unlink(p)
            except Exception:
                pass


class CompareByNameRequest(BaseModel):
    policy_names: List[str]
    user_condition: str = ""


@router.post("/compare-by-name", response_model=PolicyCompareResponse)
async def compare_policies_by_name_endpoint(body: CompareByNameRequest):
    """
    Compare 2–5 insurance policies by name.
    AI uses its knowledge (and optionally web search via Groq) to compare them.
    Works for any insurer worldwide — not limited to uploaded/RAG policies.
    """
    if len(body.policy_names) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 policy names to compare.")
    if len(body.policy_names) > MAX_COMPARE:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_COMPARE} policies allowed.")

    cleaned = [n.strip() for n in body.policy_names if n.strip()]
    if len(cleaned) < 2:
        raise HTTPException(status_code=400, detail="Policy names cannot be empty.")

    try:
        return await compare_policies_by_name(cleaned, user_condition=body.user_condition)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))