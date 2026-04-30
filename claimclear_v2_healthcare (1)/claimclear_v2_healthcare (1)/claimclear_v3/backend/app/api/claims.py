"""Healthcare Claims routes — CRUD + AI explanation + appeal letter + prediction"""
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    Claim, ClaimListResponse, ClaimStatus, ClaimType,
    DashboardStats, ExplanationRequest, ExplanationResponse,
    SubmitClaimRequest, SubmitClaimResponse,
    AppealLetterRequest, AppealLetterResponse,
    ClaimPredictRequest, ClaimPredictResponse,
)
from app.services.ai_service import (
    explain_claim, generate_appeal_letter, analyze_submitted_claim, predict_claim
)

router = APIRouter(prefix="/claims", tags=["claims"])

_CLAIMS: list[Claim] = [
    Claim(id="CLM-2024-0041", title="Hospitalization — Dengue Fever", hospital="Apollo Hospital, Delhi",
          type=ClaimType.health, type_color="#e8f0fe", type_text_color="#1a56db",
          amount="₹2,40,000", status=ClaimStatus.denied, date="15 Nov 2024"),
    Claim(id="CLM-2024-0038", title="Cardiac Treatment — Bypass Surgery", hospital="Fortis Heart Institute",
          type=ClaimType.health, type_color="#e8f0fe", type_text_color="#1a56db",
          amount="₹3,85,000", status=ClaimStatus.approved, date="02 Nov 2024"),
    Claim(id="CLM-2024-0031", title="Knee Replacement Surgery", hospital="AIIMS, Bhopal",
          type=ClaimType.health, type_color="#e8f0fe", type_text_color="#1a56db",
          amount="₹1,95,000", status=ClaimStatus.pending, date="18 Oct 2024"),
    Claim(id="CLM-2024-0025", title="Knee Surgery — Fortis Hospital", hospital="Fortis Hospital, Bhopal",
          type=ClaimType.health, type_color="#e8f0fe", type_text_color="#1a56db",
          amount="₹95,000", status=ClaimStatus.approved, date="05 Oct 2024"),
    Claim(id="CLM-2024-0019", title="Maternity — Normal Delivery", hospital="Bombay Hospital, Mumbai",
          type=ClaimType.health, type_color="#e8f0fe", type_text_color="#1a56db",
          amount="₹62,500", status=ClaimStatus.approved, date="20 Sep 2024"),
    Claim(id="CLM-2024-0012", title="Appendectomy — AIIMS", hospital="AIIMS, Bhopal",
          type=ClaimType.health, type_color="#e8f0fe", type_text_color="#1a56db",
          amount="₹1,80,000", status=ClaimStatus.denied, date="01 Sep 2024"),
    Claim(id="CLM-2024-0007", title="ICU Admission — Pneumonia", hospital="Max Hospital, Delhi",
          type=ClaimType.health, type_color="#e8f0fe", type_text_color="#1a56db",
          amount="₹1,18,000", status=ClaimStatus.approved, date="10 Aug 2024"),
]


def _find(claim_id: str) -> Claim | None:
    return next((c for c in _CLAIMS if c.id == claim_id), None)


@router.get("/stats", response_model=DashboardStats)
async def get_stats():
    approved = sum(1 for c in _CLAIMS if c.status == ClaimStatus.approved)
    denied   = sum(1 for c in _CLAIMS if c.status == ClaimStatus.denied)
    pending  = sum(1 for c in _CLAIMS if c.status == ClaimStatus.pending)
    return DashboardStats(
        total_claims=len(_CLAIMS),
        approved=approved, denied=denied, pending=pending,
        total_covered="₹9,75,500",
        recent_activity=[
            {"id": c.id, "title": c.title, "status": c.status.value,
             "date": c.date, "hospital": c.hospital, "amount": c.amount}
            for c in _CLAIMS[:4]
        ],
    )


@router.get("", response_model=ClaimListResponse)
async def list_claims(status: str | None = None):
    filtered = _CLAIMS if not status or status == "all" else [c for c in _CLAIMS if c.status.value == status]
    return ClaimListResponse(
        claims=filtered, total=len(_CLAIMS),
        approved=sum(1 for c in _CLAIMS if c.status == ClaimStatus.approved),
        denied=sum(1 for c in _CLAIMS if c.status == ClaimStatus.denied),
        pending=sum(1 for c in _CLAIMS if c.status == ClaimStatus.pending),
    )


@router.post("", response_model=SubmitClaimResponse)
async def submit_claim(body: SubmitClaimRequest):
    new_id = f"CLM-2025-{str(uuid.uuid4())[:4].upper()}"
    status_map = {
        "Denied": ClaimStatus.denied,
        "Approved": ClaimStatus.approved,
        "Pending": ClaimStatus.pending,
    }
    new_claim = Claim(
        id=new_id,
        title=f"Health Claim — {body.insurer}",
        hospital=body.hospital or body.insurer,
        type=ClaimType.health, type_color="#e8f0fe", type_text_color="#1a56db",
        amount=f"₹{body.amount}",
        status=status_map.get(body.claim_status, ClaimStatus.pending),
        date=datetime.now().strftime("%d %b %Y"),
    )
    _CLAIMS.insert(0, new_claim)

    summary, req_docs, prediction, confidence, pred_reason = await analyze_submitted_claim(
        body.claim_status, body.amount, body.insurer, body.description
    )
    return SubmitClaimResponse(
        claim_id=new_id,
        message="Health insurance claim submitted and analysed by AI.",
        ai_summary=summary,
        required_documents=req_docs,
        claim_prediction=prediction,
        prediction_confidence=confidence,
        prediction_reason=pred_reason,
    )


# ── /predict MUST be declared before /{claim_id} to avoid routing conflict ────
@router.post("/predict", response_model=ClaimPredictResponse)
async def predict_claim_endpoint(body: ClaimPredictRequest):
    """Predict whether a healthcare claim will be approved or denied."""
    try:
        return await predict_claim(
            description=body.description,
            diagnosis=body.diagnosis,
            hospital=body.hospital,
            amount=body.amount,
            policy_id=body.policy_id,
            days_since_policy_start=body.days_since_policy_start,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{claim_id}", response_model=Claim)
async def get_claim(claim_id: str):
    claim = _find(claim_id)
    if not claim:
        raise HTTPException(status_code=404, detail=f"Claim '{claim_id}' not found")
    return claim


@router.post("/{claim_id}/explain", response_model=ExplanationResponse)
async def explain_claim_endpoint(claim_id: str, body: ExplanationRequest):
    try:
        return await explain_claim(
            claim_id=claim_id, claim_title=body.claim_title,
            claim_type="Health", amount=body.amount,
            status=body.status, insurer=body.insurer,
            denial_reason=body.denial_reason, policy_id=body.policy_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{claim_id}/appeal-letter", response_model=AppealLetterResponse)
async def appeal_letter(claim_id: str, body: AppealLetterRequest):
    try:
        return await generate_appeal_letter(
            claim_id=claim_id, claim_title=body.claim_title,
            claimant_name=body.claimant_name, insurer=body.insurer,
            denial_reason=body.denial_reason, amount=body.amount,
            policy_id=body.policy_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
