"""All Pydantic request/response schemas for ClaimClear — Healthcare Insurance Only"""
from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class ClaimStatus(str, Enum):
    approved = "approved"
    denied = "denied"
    pending = "pending"
    processing = "processing"


class ClaimType(str, Enum):
    health = "Health"


class PolicyInfo(BaseModel):
    id: str
    name: str
    filename: str
    insurer: str
    policy_type: str
    description: str
    indexed: bool = False


class PolicyListResponse(BaseModel):
    policies: list[PolicyInfo]


class PolicyExplainRequest(BaseModel):
    policy_id: str
    question: str = "Give me a complete overview of this policy — coverage, exclusions, and key terms."


class PolicyExplainResponse(BaseModel):
    policy_id: str
    policy_name: str
    answer: str
    sources: list[str] = Field(default_factory=list)
    key_terms: list[dict[str, str]] = Field(default_factory=list)


class Claim(BaseModel):
    id: str
    title: str
    hospital: str
    type: ClaimType = ClaimType.health
    type_color: str = "#e8f0fe"
    type_text_color: str = "#1a56db"
    amount: str
    status: ClaimStatus
    date: str


class ClaimListResponse(BaseModel):
    claims: list[Claim]
    total: int
    approved: int
    denied: int
    pending: int


class SubmitClaimRequest(BaseModel):
    claim_type: str = "Health Insurance"
    claim_status: str
    amount: str
    insurer: str
    description: str
    hospital: str = ""


class SubmitClaimResponse(BaseModel):
    claim_id: str
    message: str
    ai_summary: str
    required_documents: list[str] = Field(default_factory=list)
    claim_prediction: str = ""
    prediction_confidence: str = ""
    prediction_reason: str = ""


class ExplanationRequest(BaseModel):
    claim_id: str
    claim_title: str
    claim_type: str = "Health"
    amount: str
    status: str
    insurer: str
    denial_reason: str = ""
    policy_id: str = ""


class DenialReason(BaseModel):
    code: str
    title: str
    detail: str
    is_appealable: bool


class AppealStep(BaseModel):
    step_number: int
    title: str
    description: str
    is_done: bool = False


class ExplanationResponse(BaseModel):
    claim_id: str
    summary: str
    denial_reasons: list[DenialReason]
    appeal_steps: list[AppealStep]
    key_terms: list[dict[str, str]]
    required_documents: list[str]
    has_appeal_angle: bool
    appeal_confidence: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    claim_id: str
    claim_context: dict[str, Any]
    messages: list[ChatMessage]
    policy_id: str = ""


class ChatResponse(BaseModel):
    reply: str
    suggested_questions: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class AppealLetterRequest(BaseModel):
    claim_id: str
    claim_title: str
    claimant_name: str
    insurer: str
    denial_reason: str
    amount: str
    policy_id: str = ""


class AppealLetterResponse(BaseModel):
    claim_id: str
    letter: str
    subject: str


class IndexRequest(BaseModel):
    policy_ids: list[str] = Field(default_factory=list)
    force_reindex: bool = False


class IndexResponse(BaseModel):
    indexed: list[str]
    skipped: list[str]
    errors: dict[str, str]
    total_chunks: int


class DashboardStats(BaseModel):
    total_claims: int
    approved: int
    denied: int
    pending: int
    total_covered: str
    recent_activity: list[dict[str, str]]


# ── Policy Comparison ─────────────────────────────────────────────────────────
class PolicyFeature(BaseModel):
    category: str
    values: dict[str, str]


class PolicyCompareResponse(BaseModel):
    policies: list[str]
    features: list[PolicyFeature]
    recommendation: str
    best_for_condition: str


# ── Claim Prediction ──────────────────────────────────────────────────────────
class ClaimPredictRequest(BaseModel):
    description: str
    diagnosis: str = ""
    hospital: str = ""
    amount: str = ""
    policy_id: str = ""
    days_since_policy_start: int = 0


class ClaimPredictResponse(BaseModel):
    prediction: str
    confidence: str
    confidence_score: int
    reasons: list[str]
    risk_factors: list[str]
    suggestions: list[str]
    verdict_color: str
