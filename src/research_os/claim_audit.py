"""Claim audit helpers for XCIENTIST-style research harnesses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DRIFT_TYPES = {
    "semantic_drift",
    "experimental_drift",
    "mechanistic_drift",
    "insufficient_evidence",
    "no_drift",
}


@dataclass
class ClaimAudit:
    claim_id: str
    related_exp_ids: list[str]
    claimed_improvement: str
    supporting_metrics: dict[str, Any] = field(default_factory=dict)
    required_ablations: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    drift_type: str = "insufficient_evidence"
    audit_result: str = "revise"
    allowed_conclusion: str = ""


def classify_drift_type(
    claim_text: str,
    hypothesis: str,
    has_required_experiments: bool,
    has_mechanistic_evidence: bool,
    missing_evidence: list[str],
) -> str:
    if missing_evidence:
        return "insufficient_evidence"
    if hypothesis and claim_text and hypothesis.lower() not in claim_text.lower():
        return "semantic_drift"
    if not has_required_experiments:
        return "experimental_drift"
    if "because" in claim_text.lower() and not has_mechanistic_evidence:
        return "mechanistic_drift"
    return "no_drift"


def detect_claim_drift(contract: dict[str, Any], claim_text: str, evidence: dict[str, Any]) -> str:
    return classify_drift_type(
        claim_text=claim_text,
        hypothesis=str(contract.get("hypothesis", "")),
        has_required_experiments=bool(evidence.get("has_required_experiments", False)),
        has_mechanistic_evidence=bool(evidence.get("has_mechanistic_evidence", False)),
        missing_evidence=list(evidence.get("missing_evidence", [])),
    )


def audit_claim(
    claim_id: str,
    claim_text: str,
    related_exp_ids: list[str],
    contract: dict[str, Any],
    supporting_metrics: dict[str, Any],
    required_ablations: list[str],
    completed_ablations: list[str],
    evidence: dict[str, Any] | None = None,
) -> ClaimAudit:
    evidence = evidence or {}
    missing_ablations = [item for item in required_ablations if item not in completed_ablations]
    missing_evidence = list(evidence.get("missing_evidence", [])) + [
        f"missing_ablation:{item}" for item in missing_ablations
    ]
    drift_type = detect_claim_drift(contract, claim_text, {**evidence, "missing_evidence": missing_evidence})

    if drift_type == "no_drift":
        audit_result = "allow"
        allowed_conclusion = claim_text
    elif drift_type in {"insufficient_evidence", "experimental_drift"}:
        audit_result = "reject"
        allowed_conclusion = "当前证据不足，不能写入该提升结论。"
    else:
        audit_result = "revise"
        allowed_conclusion = str(contract.get("conclusion_boundary", "请将结论收窄到 validation contract 允许范围。"))

    return ClaimAudit(
        claim_id=claim_id,
        related_exp_ids=related_exp_ids,
        claimed_improvement=claim_text,
        supporting_metrics=supporting_metrics,
        required_ablations=required_ablations,
        missing_evidence=missing_evidence,
        drift_type=drift_type,
        audit_result=audit_result,
        allowed_conclusion=allowed_conclusion,
    )
