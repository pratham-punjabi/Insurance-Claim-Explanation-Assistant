"""
Guardrails — Healthcare Insurance Only
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Enforces that all queries relate to HEALTH/MEDICAL insurance only.
Blocks non-health insurance and completely off-topic queries.
"""
from __future__ import annotations
import re
import logging

logger = logging.getLogger(__name__)

HEALTH_INSURANCE_KEYWORDS = {
    # Core health insurance terms
    "health insurance", "medical insurance", "mediclaim", "hospitalisation", "hospitalization",
    "health policy", "health plan", "health cover", "health coverage",
    "cashless", "reimbursement", "pre-existing", "waiting period", "sum insured",
    "irdai", "tpa", "third party administrator", "network hospital", "empanelled",
    # Medical/clinical
    "hospital", "surgery", "treatment", "diagnosis", "ailment", "disease", "illness",
    "medical", "doctor", "prescription", "icu", "opd", "daycare", "ambulance",
    "critical illness", "maternity", "delivery", "dental", "vision", "organ",
    "physiotherapy", "rehabilitation", "chemotherapy", "dialysis",
    # Claim process
    "claim", "denied", "rejection", "appeal", "grievance", "ombudsman",
    "pre-authorisation", "preauthorization", "discharge summary", "bills",
    "investigation", "copay", "co-pay", "deductible", "premium",
    # Specific insurers / policies
    "lic", "sbi general", "star health", "hdfc ergo", "bajaj allianz",
    "new india assurance", "united india", "national insurance", "oriental insurance",
    "jeevan arogya", "health companion", "max bupa", "care health", "niva bupa",
    "arogya", "mediclaim", "health protection",
    # Actions / question words in context
    "policy", "insurer", "insured", "coverage", "cover", "benefit", "exclusion",
    "portability", "renewal", "lapse", "eligible", "eligibility",
}

BLOCKED_PATTERNS = [
    r"\b(recipe|cook|food|restaurant)\b",
    r"\b(movie|film|song|music|celebrity|actor|actress)\b",
    r"\b(sports|cricket|football|ipl|match|score)\b",
    r"\b(stock|share market|nifty|sensex|crypto|bitcoin)\b",
    r"\b(travel|hotel|booking|flight|trip)\b",
    r"\b(jokes?|funny|meme|entertainment)\b",
    r"\b(dating|relationship|love|romance)\b",
    r"\b(politics?|election|party|minister)\b",
    r"\b(hack|crack|exploit|malware|virus)\b",
    r"\b(write code|programming|python|javascript|html|css)\b",
    # Non-health insurance types (explicitly blocked)
    r"\b(auto insurance|car insurance|motor insurance|vehicle insurance)\b",
    r"\b(home insurance|property insurance|fire insurance)\b",
    r"\b(term insurance|life insurance|endowment|ulip)\b(?!.*health)",
    r"\b(travel insurance)\b",
]

GREETINGS = {
    "hi", "hello", "hey", "hii", "helo", "namaste", "good morning",
    "good evening", "good afternoon", "thanks", "thank you", "ok",
    "okay", "yes", "no", "bye", "goodbye", "help", "start",
}

OFF_TOPIC_RESPONSE = (
    "I'm ClaimClear AI — I specialise exclusively in **healthcare & medical insurance**. "
    "I can help you with:\n"
    "• Understanding your health insurance policy terms and coverage\n"
    "• Explaining why a health claim was denied\n"
    "• Comparing multiple healthcare policies\n"
    "• Predicting whether your claim is likely to be approved\n"
    "• Drafting appeal letters for denied health claims\n"
    "• Identifying required documents for health insurance claims\n\n"
    "Please ask me something related to your healthcare insurance policy or claim."
)

NON_HEALTH_INSURANCE_RESPONSE = (
    "ClaimClear AI handles **healthcare insurance only**. "
    "I'm not able to assist with auto, home, life, or other non-health insurance types. "
    "Please ask me about your health or medical insurance policy."
)


def is_health_insurance_related(text: str) -> tuple[bool, str]:
    normalized = text.lower().strip()

    if normalized in GREETINGS or len(normalized) < 4:
        return True, "greeting"

    # Non-health insurance type — explicit block
    non_health = [
        r"\b(auto|car|motor|vehicle) insurance\b",
        r"\b(home|property|fire) insurance\b",
        r"\btravel insurance\b",
    ]
    for pattern in non_health:
        if re.search(pattern, normalized, re.IGNORECASE):
            logger.info(f"Guardrail: non-health insurance query blocked: {text[:80]}")
            return False, "non_health_insurance"

    # General off-topic blocklist
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            logger.info(f"Guardrail blocked pattern: {text[:80]}")
            return False, "off_topic_pattern"

    # Check health insurance keywords
    for keyword in HEALTH_INSURANCE_KEYWORDS:
        if keyword in normalized:
            return True, "health_keyword"

    # Contextual / ambiguous pass-through
    if any(phrase in normalized for phrase in [
        "my claim", "my policy", "my insurance", "why was", "how do i",
        "what is", "explain", "covered", "denied", "rejected", "appeal",
        "document", "required", "submit", "compare", "better", "best",
        "waiting", "coverage", "limit", "network",
    ]):
        return True, "contextual_match"

    logger.info(f"Guardrail blocked (no health keyword): {text[:80]}")
    return False, "no_health_keywords"


def check_input(user_message: str) -> tuple[bool, str]:
    allowed, reason = is_health_insurance_related(user_message)
    if not allowed:
        if reason == "non_health_insurance":
            return False, NON_HEALTH_INSURANCE_RESPONSE
        return False, OFF_TOPIC_RESPONSE
    return True, ""
