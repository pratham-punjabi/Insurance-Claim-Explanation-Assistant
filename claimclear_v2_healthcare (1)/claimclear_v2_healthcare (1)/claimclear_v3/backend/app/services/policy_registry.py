"""Registry of all known insurance policy PDFs with metadata."""
from __future__ import annotations
from pathlib import Path
from app.models.schemas import PolicyInfo

POLICY_REGISTRY: list[PolicyInfo] = [
    PolicyInfo(
        id="sbi_health",
        name="SBI General Health Insurance — Retail",
        filename="sbi_insurance_policy.pdf",
        insurer="SBI General Insurance",
        policy_type="Health",
        description="Comprehensive retail health insurance covering hospitalisation, room rent, ICU, surgery, OPD, and critical illness. Covers pre/post hospitalisation, ambulance, daycare procedures.",
    ),
    PolicyInfo(
        id="lic_jeevan_arogya",
        name="LIC Jeevan Arogya",
        filename="LIC_JeevanArogyaBrochure_.pdf",
        insurer="Life Insurance Corporation of India",
        policy_type="Health",
        description="Non-linked health plan covering insured and extended family. Provides hospitalisation benefit, major surgical benefit, day care procedures, and other surgical benefits.",
    ),
    PolicyInfo(
        id="lic_health_protection_plus",
        name="LIC Health Protection Plus",
        filename="LIC_S_HEALTH_PROTECTION_PLUS.pdf",
        insurer="Life Insurance Corporation of India",
        policy_type="Health",
        description="Unit-linked health plan with market-linked returns plus comprehensive health coverage. Covers hospitalisation, critical illness, and disability.",
    ),
    PolicyInfo(
        id="long_term_ulip_health",
        name="Long-Term Unit Linked Health Insurance",
        filename="Long_Term_Unit_Linked_Health_Insurance.pdf",
        insurer="IRDAI Approved Insurer",
        policy_type="Health (ULIP)",
        description="Long-term ULIP health plan combining insurance coverage with investment. Suitable for long-horizon policyholders seeking both protection and growth.",
    ),
    PolicyInfo(
        id="health_companion",
        name="Health Companion Plan (GEN617)",
        filename="Health_CompanionHealth_Insurance_Plan_GEN617.pdf",
        insurer="General Insurance",
        policy_type="Health",
        description="Comprehensive health companion plan offering cashless hospitalisation, wellness benefits, maternity cover, and restoration benefit.",
    ),
]

_MAP: dict[str, PolicyInfo] = {p.id: p for p in POLICY_REGISTRY}


def get_policy(policy_id: str) -> PolicyInfo | None:
    return _MAP.get(policy_id)


def get_all_policies() -> list[PolicyInfo]:
    return POLICY_REGISTRY


def resolve_pdf_path(filename: str, policies_dir: Path) -> Path:
    return policies_dir / filename
