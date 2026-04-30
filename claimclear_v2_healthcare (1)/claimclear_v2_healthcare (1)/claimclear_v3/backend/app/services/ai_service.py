"""
AI Service — Healthcare Insurance Only
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All LLM calls use Groq (llama-3.3-70b-versatile).
Context is retrieved from ChromaDB via semantic search (RAG).
PDFs are chunked + embedded ONCE at startup; reused on every query.

Features:
  • explain_policy()          — Answer policy questions via RAG
  • explain_claim()           — Structured denial analysis
  • chat_with_claim()         — Multi-turn RAG chat
  • generate_appeal_letter()  — IRDAI-compliant appeal draft
  • analyze_submitted_claim() — Quick triage for new claims
  • compare_policies()        — Compare up to 5 uploaded healthcare PDFs
  • predict_claim()           — Predict claim approval / denial likelihood
"""
from __future__ import annotations
import json
import logging
import re
from typing import Any

from groq import Groq

from app.core.config import settings
from app.models.schemas import (
    AppealLetterResponse, AppealStep, ChatMessage, DenialReason,
    ExplanationResponse, PolicyExplainResponse,
    PolicyCompareResponse, PolicyFeature,
    ClaimPredictResponse,
)
from app.services import vector_store
from app.services.guardrails import check_input

logger = logging.getLogger(__name__)

_groq_client: Groq | None = None

HEALTHCARE_PREFIX = (
    "You are ClaimClear AI — an expert healthcare insurance advisor specialising "
    "in Indian health insurance policies and IRDAI regulations. "
    "You ONLY handle healthcare/medical insurance topics. "
    "Refuse any non-health-insurance query politely.\n\n"
)


def _get_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=settings.groq_api_key)
    return _groq_client


def _call_groq(system: str, user: str, max_tokens: int = 2048, temperature: float = 0.3) -> str:
    client = _get_client()
    resp = client.chat.completions.create(
        model=settings.groq_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def _call_groq_messages(system: str, messages: list[dict], max_tokens: int = 2048) -> str:
    client = _get_client()
    resp = client.chat.completions.create(
        model=settings.groq_model,
        messages=[{"role": "system", "content": system}] + messages,
        max_tokens=max_tokens,
        temperature=0.35,
    )
    return resp.choices[0].message.content or ""


def _safe_parse_json(text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
    match = re.search(r'\{[\s\S]*\}', cleaned)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {}


def _guard(msg: str) -> str | None:
    allowed, response = check_input(msg)
    return None if allowed else response


# ── RAG retrieval from ChromaDB ───────────────────────────────────────────────

def _rag_context(query: str, policy_ids: list[str] | None = None, n: int = 5) -> tuple[str, list[str]]:
    """
    Retrieve semantically relevant chunks from ChromaDB for the query.
    Returns (formatted_context_string, list_of_source_names).
    Chunks were embedded once at startup and are reused every call.
    """
    try:
        chunks = vector_store.retrieve(query, policy_ids=policy_ids, n_results=n)
        if not chunks:
            return "No relevant policy content found in the knowledge base.", []
        sources = list({c["policy_name"] for c in chunks})
        parts = [
            f"[Source {i}: {c['policy_name']} (relevance: {c['score']:.2f})]\n{c['text']}"
            for i, c in enumerate(chunks, 1)
        ]
        return "\n\n---\n\n".join(parts), sources
    except Exception as e:
        logger.warning(f"RAG retrieval failed: {e}")
        return "Policy knowledge base unavailable.", []


# ── Required health insurance documents ──────────────────────────────────────

HEALTH_REQUIRED_DOCS = [
    "Duly filled and signed claim form",
    "Original hospital bills and payment receipts",
    "Discharge summary from treating hospital",
    "All investigation reports (lab tests, X-rays, MRI, etc.)",
    "Prescriptions and pharmacy bills",
    "Doctor's certificate confirming diagnosis",
    "Policy document and ID proof",
    "KYC documents (Aadhaar / PAN)",
    "Pre-authorisation reference (if applicable)",
    "Previous insurer's policy certificate (for portability claims)",
]


# ── Policy Explanation ────────────────────────────────────────────────────────

async def explain_policy(policy_id: str, question: str) -> PolicyExplainResponse:
    from app.services.policy_registry import get_policy
    info = get_policy(policy_id)
    if not info:
        raise ValueError(f"Unknown policy id: {policy_id}")

    blocked = _guard(question)
    if blocked:
        return PolicyExplainResponse(policy_id=policy_id, policy_name=info.name, answer=blocked)

    context, sources = _rag_context(question, policy_ids=[policy_id], n=6)

    system = HEALTHCARE_PREFIX + (
        "Use the provided healthcare policy excerpts to give accurate, specific answers. "
        "Cite specific policy sections or clauses when available. "
        "Write in clear plain language. Use bullet points for lists.\n\n"
        "At the very end add a JSON block:\n"
        '{"key_terms": [{"term": "term name", "definition": "plain language definition"}]}'
    )
    user = f"""POLICY: {info.name} ({info.insurer})

RELEVANT POLICY EXCERPTS (from ChromaDB):
{context}

USER QUESTION: {question}

Answer based on the policy excerpts above."""

    raw = _call_groq(system, user, max_tokens=1500)

    key_terms: list[dict] = []
    jm = re.search(r'\{[\s\S]*"key_terms"[\s\S]*\}', raw)
    answer_text = raw
    if jm:
        try:
            key_terms = json.loads(jm.group()).get("key_terms", [])
            answer_text = raw[:jm.start()].strip()
        except Exception:
            pass

    return PolicyExplainResponse(
        policy_id=policy_id, policy_name=info.name,
        answer=answer_text, sources=sources, key_terms=key_terms,
    )


# ── Claim Explanation ─────────────────────────────────────────────────────────

async def explain_claim(
    claim_id: str, claim_title: str, claim_type: str,
    amount: str, status: str, insurer: str,
    denial_reason: str, policy_id: str,
) -> ExplanationResponse:

    query = f"health claim denial {denial_reason} waiting period exclusions appeal"
    context, sources = _rag_context(query, policy_ids=[policy_id] if policy_id else None, n=6)

    system = HEALTHCARE_PREFIX + (
        "Analyse the healthcare insurance claim and respond ONLY with valid JSON:\n"
        "{\n"
        '  "summary": "2-3 sentence plain-language explanation",\n'
        '  "denial_reasons": [{"code":"DC-XX","title":"Short title","detail":"Detailed explanation","is_appealable":true}],\n'
        '  "appeal_steps": [{"step_number":1,"title":"Step title","description":"What to do","is_done":false}],\n'
        '  "key_terms": [{"term":"Term","definition":"Plain-language definition"}],\n'
        '  "has_appeal_angle": true,\n'
        '  "appeal_confidence": "High"\n'
        "}"
    )
    user = f"""HEALTHCARE CLAIM:
- Claim ID: {claim_id}
- Title: {claim_title}
- Amount: {amount}
- Status: {status}
- Insurer: {insurer}
- Denial Reason: {denial_reason or "Not provided"}

RELEVANT POLICY KNOWLEDGE (from ChromaDB):
{context}

Provide 2-3 denial reasons, 4-5 appeal steps, 4-5 key health insurance terms, appeal viability (High/Medium/Low)."""

    raw = _call_groq(system, user, max_tokens=2000, temperature=0.2)
    data = _safe_parse_json(raw) or {
        "summary": "Unable to parse AI response. Please try again.",
        "denial_reasons": [], "appeal_steps": [], "key_terms": [],
        "has_appeal_angle": False, "appeal_confidence": "Low",
    }

    return ExplanationResponse(
        claim_id=claim_id,
        summary=data.get("summary", ""),
        denial_reasons=[DenialReason(**r) for r in data.get("denial_reasons", [])],
        appeal_steps=[AppealStep(**s) for s in data.get("appeal_steps", [])],
        key_terms=data.get("key_terms", []),
        required_documents=HEALTH_REQUIRED_DOCS,
        has_appeal_angle=data.get("has_appeal_angle", False),
        appeal_confidence=data.get("appeal_confidence", "Low"),
    )


# ── Multi-turn Chat ───────────────────────────────────────────────────────────

async def chat_with_claim(
    claim_context: dict[str, Any],
    messages: list[ChatMessage],
    policy_id: str,
) -> tuple[str, list[str], list[str]]:

    last_user_msg = next((m.content for m in reversed(messages) if m.role == "user"), "")
    blocked = _guard(last_user_msg)
    if blocked:
        return blocked, [], []

    context, sources = _rag_context(
        last_user_msg, policy_ids=[policy_id] if policy_id else None, n=5
    )

    system = HEALTHCARE_PREFIX + f"""Be conversational, empathetic, and precise. Give actionable health insurance advice.
Reference specific policy clauses when available. Use ₹ for amounts. Keep under 300 words.

CLAIM CONTEXT:
{json.dumps(claim_context, indent=2)}

RELEVANT POLICY KNOWLEDGE (from ChromaDB):
{context}

At the end add: {{"suggested_questions": ["Question 1?", "Question 2?", "Question 3?"]}}"""

    raw = _call_groq_messages(system, [{"role": m.role, "content": m.content} for m in messages], max_tokens=1500)

    suggested: list[str] = []
    jm = re.search(r'\{[\s\S]*"suggested_questions"[\s\S]*\}', raw)
    reply = raw
    if jm:
        try:
            suggested = json.loads(jm.group()).get("suggested_questions", [])
            reply = raw[:jm.start()].strip()
        except Exception:
            pass

    return reply, suggested, sources


# ── Appeal Letter ─────────────────────────────────────────────────────────────

async def generate_appeal_letter(
    claim_id: str, claim_title: str, claimant_name: str,
    insurer: str, denial_reason: str, amount: str, policy_id: str,
) -> AppealLetterResponse:

    context, _ = _rag_context(
        f"appeal portability grievance health insurance {denial_reason}",
        policy_ids=[policy_id] if policy_id else None, n=4,
    )

    system = HEALTHCARE_PREFIX + (
        "Draft formal IRDAI-compliant health insurance appeal letters. "
        'Return ONLY valid JSON: {"subject": "...", "letter": "full letter text"}'
    )
    user = f"""Draft a health insurance claim appeal letter for:
- Claimant: {claimant_name}
- Claim ID: {claim_id}
- Title: {claim_title}
- Insurer: {insurer}
- Amount: {amount}
- Denial Reason: {denial_reason}

Relevant policy knowledge (from ChromaDB):
{context}

Professional, reference IRDAI health insurance regulations, include enclosures list. Under 400 words."""

    raw = _call_groq(system, user, max_tokens=1200, temperature=0.2)
    data = _safe_parse_json(raw)

    return AppealLetterResponse(
        claim_id=claim_id,
        subject=data.get("subject", f"Appeal against Denial of Health Insurance Claim {claim_id}"),
        letter=data.get("letter", raw),
    )


# ── New Claim Triage ──────────────────────────────────────────────────────────

async def analyze_submitted_claim(
    claim_status: str, amount: str, insurer: str, description: str, policy_id: str = "",
) -> tuple[str, list[str], str, str, str]:
    """Returns (summary, req_docs, prediction, confidence, reason)."""
    context, _ = _rag_context(
        f"health insurance claim {description[:200]}",
        policy_ids=[policy_id] if policy_id else None, n=4,
    )

    system = HEALTHCARE_PREFIX + (
        "Given a health insurance claim, provide triage + prediction. Return JSON:\n"
        '{"summary":"...","is_appealable":true,"next_step":"...",'
        '"prediction":"Likely Approved|Likely Denied|Uncertain",'
        '"confidence":"High|Medium|Low","prediction_reason":"..."}'
    )
    user = f"""Health Insurance Claim:
Status: {claim_status}
Amount: ₹{amount}
Insurer: {insurer}
Description: {description}

Relevant Policy Knowledge (from ChromaDB):
{context}"""

    raw = _call_groq(system, user, max_tokens=500, temperature=0.3)
    data = _safe_parse_json(raw)

    summary = data.get("summary", "")
    if data.get("next_step"):
        summary += f"\n\n**Next Step:** {data['next_step']}"

    return (
        summary, HEALTH_REQUIRED_DOCS,
        data.get("prediction", "Uncertain"),
        data.get("confidence", "Low"),
        data.get("prediction_reason", ""),
    )


# ── Policy Comparison (uploaded PDFs — direct text, not from vector DB) ───────

def _extract_pdf_text(pdf_path: str, max_chars: int = 12000) -> str:
    """Extract text from a user-uploaded PDF for comparison. Uses same extractor as indexer."""
    try:
        from app.services.pdf_extractor import extract_text_from_pdf
        from pathlib import Path
        text = extract_text_from_pdf(Path(pdf_path))
        return text[:max_chars]
    except Exception as e:
        logger.warning(f"PDF extraction failed for {pdf_path}: {e}")
        # Fallback to pypdf
        try:
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            parts = []
            for page in reader.pages:
                parts.append(page.extract_text() or "")
                if sum(len(p) for p in parts) >= max_chars:
                    break
            return "\n".join(parts)[:max_chars]
        except Exception as e2:
            logger.warning(f"pypdf fallback also failed: {e2}")
            return ""


async def compare_policies(
    pdf_texts: dict[str, str],
    user_condition: str = "",
) -> PolicyCompareResponse:
    """Compare up to 5 user-uploaded healthcare policy PDFs directly."""
    names = list(pdf_texts.keys())
    combined = "\n\n".join(
        f"=== POLICY: {name} ===\n{text[:3500]}" for name, text in pdf_texts.items()
    )

    system = HEALTHCARE_PREFIX + (
        "Compare multiple healthcare insurance policies. Extract and compare key features. "
        "Return ONLY valid JSON:\n"
        '{"features":[{"category":"Waiting Period","values":{"Policy A":"2 years","Policy B":"3 years"}},'
        '{"category":"Sum Insured","values":{...}},{"category":"Room Rent Limit","values":{...}},'
        '{"category":"Pre-existing Coverage","values":{...}},{"category":"Maternity Cover","values":{...}},'
        '{"category":"Network Hospitals","values":{...}},{"category":"No-claim Bonus","values":{...}},'
        '{"category":"Daycare Procedures","values":{...}},{"category":"Co-payment","values":{...}},'
        '{"category":"Restoration Benefit","values":{...}}],'
        '"recommendation":"Overall comparison and who each policy suits",'
        '"best_for_condition":"Which policy is best for the given condition and why"}'
    )
    cond = f"\nUser condition/profile: {user_condition}" if user_condition else ""
    user = f"Compare these {len(names)} healthcare insurance policies:{cond}\n\n{combined}\n\nExtract all key features and give a clear recommendation."

    raw = _call_groq(system, user, max_tokens=2500, temperature=0.2)
    data = _safe_parse_json(raw)

    return PolicyCompareResponse(
        policies=names,
        features=[PolicyFeature(category=f["category"], values=f["values"]) for f in data.get("features", [])],
        recommendation=data.get("recommendation", ""),
        best_for_condition=data.get("best_for_condition", ""),
    )


async def compare_policies_by_name(
    policy_names: list[str],
    user_condition: str = "",
) -> PolicyCompareResponse:
    """
    Compare insurance policies by name only — no PDF upload required.
    Works for any insurer worldwide. Uses Groq's knowledge to generate
    a structured comparison table.
    """
    condition_clause = f"\nUser's health profile / use case: {user_condition}" if user_condition else ""

    system = (
        "You are an expert global insurance analyst. "
        "The user will give you a list of insurance policy names (from any insurer, any country). "
        "Use your knowledge to compare them thoroughly across all standard insurance dimensions. "
        "If you are uncertain about a value, use your best estimate and mark it with '~'. "
        "Do NOT restrict yourself to only Indian policies — compare any policy worldwide. "
        "Return ONLY valid JSON with this exact structure (no markdown, no backticks, no preamble):\n"
        '{"policies":["Policy A","Policy B"],'
        '"features":['
        '{"category":"Sum Insured Options","values":{"Policy A":"value","Policy B":"value"}},'
        '{"category":"Annual Premium (approx)","values":{...}},'
        '{"category":"Network Hospitals","values":{...}},'
        '{"category":"Room Rent Limit","values":{...}},'
        '{"category":"Pre-Hospitalisation","values":{...}},'
        '{"category":"Post-Hospitalisation","values":{...}},'
        '{"category":"Initial Waiting Period","values":{...}},'
        '{"category":"Pre-Existing Disease Waiting","values":{...}},'
        '{"category":"Maternity Benefit","values":{...}},'
        '{"category":"Day Care Procedures","values":{...}},'
        '{"category":"No Claim Bonus","values":{...}},'
        '{"category":"Restoration Benefit","values":{...}},'
        '{"category":"Critical Illness","values":{...}},'
        '{"category":"OPD Cover","values":{...}},'
        '{"category":"Mental Health Cover","values":{...}},'
        '{"category":"Co-payment Clause","values":{...}},'
        '{"category":"Key Exclusions","values":{...}},'
        '{"category":"Claim Settlement Ratio","values":{...}},'
        '{"category":"Renewability","values":{...}},'
        '{"category":"Unique Features","values":{...}}'
        '],'
        '"recommendation":"Overall recommendation — which policy suits whom and why",'
        '"best_for_condition":"Which policy best fits the user profile and why (empty string if no profile given)"}'
    )

    policy_list = "\n".join(f"{i+1}. {name}" for i, name in enumerate(policy_names))
    user_msg = (
        f"Compare these insurance policies:{condition_clause}\n\n{policy_list}\n\n"
        "Provide a detailed, factual comparison. Use 'N/A' where a feature is not applicable. "
        "Mark uncertain values with '~'. Be specific with numbers."
    )

    raw = _call_groq(system, user_msg, max_tokens=3000, temperature=0.2)
    data = _safe_parse_json(raw)

    # Use the policy names returned by AI if available, else fall back to input names
    returned_names = data.get("policies", policy_names)

    return PolicyCompareResponse(
        policies=returned_names,
        features=[PolicyFeature(category=f["category"], values=f["values"]) for f in data.get("features", [])],
        recommendation=data.get("recommendation", ""),
        best_for_condition=data.get("best_for_condition", ""),
    )


# ── Claim Prediction ──────────────────────────────────────────────────────────

async def predict_claim(
    description: str, diagnosis: str, hospital: str,
    amount: str, policy_id: str, days_since_policy_start: int,
) -> ClaimPredictResponse:
    """Predict health insurance claim approval likelihood using RAG context."""
    context, _ = _rag_context(
        f"health insurance claim approval {diagnosis} {description[:150]} waiting period exclusions",
        policy_ids=[policy_id] if policy_id else None, n=5,
    )

    system = HEALTHCARE_PREFIX + (
        "Analyse the healthcare claim scenario and predict approval likelihood. "
        "Consider: waiting periods, pre-existing conditions, exclusions, sum insured limits, "
        "network hospitals, documentation. Return ONLY valid JSON:\n"
        '{"prediction":"Likely Approved|Likely Denied|Uncertain",'
        '"confidence":"High|Medium|Low","confidence_score":75,'
        '"reasons":["Reason 1","Reason 2"],'
        '"risk_factors":["Risk 1","Risk 2"],'
        '"suggestions":["Suggestion 1","Suggestion 2"]}'
    )
    user = f"""Health Insurance Claim Scenario:
- Description: {description}
- Diagnosis/Condition: {diagnosis or "Not specified"}
- Hospital: {hospital or "Not specified"}
- Claim Amount: ₹{amount or "Not specified"}
- Days since policy started: {days_since_policy_start}

Relevant Policy Knowledge (from ChromaDB):
{context}

Predict claim outcome with detailed reasoning."""

    raw = _call_groq(system, user, max_tokens=1000, temperature=0.2)
    data = _safe_parse_json(raw)
    prediction = data.get("prediction", "Uncertain")
    color_map = {"Likely Approved": "#15803d", "Likely Denied": "#b91c1c", "Uncertain": "#b45309"}

    return ClaimPredictResponse(
        prediction=prediction,
        confidence=data.get("confidence", "Low"),
        confidence_score=data.get("confidence_score", 50),
        reasons=data.get("reasons", []),
        risk_factors=data.get("risk_factors", []),
        suggestions=data.get("suggestions", []),
        verdict_color=color_map.get(prediction, "#b45309"),
    )