"""
XCIENTIST Research Harness - Contracts and Data Structures
Based on: XCIENTIST (arXiv:2606.18874v1)

Core concept: Externalize research synthesis and validation into
inspectable, contract-governed processes.
- Prevent claim drift: runnable artifacts must support the claimed mechanism
- Every experiment records: hypothesis, implementation, metric, ablation,
  risk check, conclusion boundary
- Validation contracts enforce preconditions before experiments proceed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import json


class ClaimStatus(Enum):
    PROPOSED = "proposed"           # Hypothesis stated, not yet tested
    TESTING = "testing"             # Experiment running
    SUPPORTED = "supported"         # Evidence supports claim
    REFUTED = "refuted"             # Evidence contradicts claim
    DRIFTED = "drifted"             # Implementation no longer matches claim
    BOUNDED = "bounded"             # Claim confirmed within explicit boundaries


class RiskLevel(Enum):
    LOW = "low"                     # Well-understood change
    MEDIUM = "medium"               # Novel but bounded change
    HIGH = "high"                   # High uncertainty, needs careful ablation
    CRITICAL = "critical"          # Could break existing best, needs rollback plan


@dataclass
class IdeaContract:
    """
    An explicit, inspectable research idea with evidence grounding.

    XCIENTIST Section 3.2: Ideas must be grounded in literature evidence
    and have explicit mechanisms that can be validated.
    """
    idea_id: str
    task_id: str
    title: str
    hypothesis: str                         # What we believe and why
    mechanism: str                          # How it works (the proposed change)
    literature_grounding: list[str] = field(default_factory=list)  # References/evidence
    expected_effect: str = ""               # Expected impact on metric
    risk_level: RiskLevel = RiskLevel.MEDIUM
    parent_idea_id: Optional[str] = None    # For idea evolution/fusion
    component_changes: list[str] = field(default_factory=list)  # What code modules change
    status: ClaimStatus = ClaimStatus.PROPOSED
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "idea_id": self.idea_id, "task_id": self.task_id,
            "title": self.title, "hypothesis": self.hypothesis,
            "mechanism": self.mechanism,
            "literature_grounding": self.literature_grounding,
            "expected_effect": self.expected_effect,
            "risk_level": self.risk_level.value,
            "parent_idea_id": self.parent_idea_id,
            "component_changes": self.component_changes,
            "status": self.status.value,
            "created_at": self.created_at,
            "metadata": self.metadata
        }


@dataclass
class ValidationContract:
    """
    Pre-execution contract that must be satisfied before experiment runs.

    XCIENTIST Section 3.4: Contract-governed execution ensures that
    implementation, evaluation, ablation and repair produce checkable
    evidence before the workflow proceeds.
    """
    contract_id: str
    idea_id: str
    task_id: str

    # Preconditions
    baseline_score: Optional[float] = None
    benchmark_configured: bool = False
    implementation_references: list[str] = field(default_factory=list)

    # Execution plan
    validation_plan: str = ""               # What experiments to run
    standard_experiments: list[str] = field(default_factory=list)  # Main eval
    ablation_experiments: list[str] = field(default_factory=list)  # Ablation tests
    risk_checks: list[str] = field(default_factory=list)           # Risk verification

    # Postconditions
    min_expected_delta: float = 0.0         # Minimum improvement to accept
    max_acceptable_regression: float = 0.0  # Maximum regression tolerable
    rollback_condition: str = ""            # When to rollback

    # Status
    all_preconditions_met: bool = False
    validation_passed: bool = False
    signed_by: Optional[str] = None         # Human or agent gate
    signed_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "contract_id": self.contract_id,
            "idea_id": self.idea_id, "task_id": self.task_id,
            "baseline_score": self.baseline_score,
            "benchmark_configured": self.benchmark_configured,
            "implementation_references": self.implementation_references,
            "validation_plan": self.validation_plan,
            "standard_experiments": self.standard_experiments,
            "ablation_experiments": self.ablation_experiments,
            "risk_checks": self.risk_checks,
            "min_expected_delta": self.min_expected_delta,
            "max_acceptable_regression": self.max_acceptable_regression,
            "rollback_condition": self.rollback_condition,
            "all_preconditions_met": self.all_preconditions_met,
            "validation_passed": self.validation_passed,
            "signed_by": self.signed_by, "signed_at": self.signed_at
        }


@dataclass
class ExperimentRecord:
    """
    Complete record of a single experiment, binding hypothesis to evidence.

    XCIENTIST Section 3.4: Each experiment must produce a record that links
    the original idea, implementation details, metrics, ablations, risk
    assessment, and conclusion boundary.
    """
    experiment_id: str
    idea_id: str
    task_id: str
    run_id: Optional[str] = None

    # What was done
    implementation_summary: str = ""
    code_changes: list[str] = field(default_factory=list)
    config_changes: dict[str, Any] = field(default_factory=dict)

    # Results
    metric_name: str = "accuracy"
    metric_direction: str = "maximize"
    baseline_value: Optional[float] = None
    experiment_value: Optional[float] = None
    delta: Optional[float] = None

    # Ablation
    ablation_results: dict[str, Any] = field(default_factory=dict)
    ablation_passed: bool = False

    # Risk assessment
    risk_checks_passed: list[bool] = field(default_factory=list)
    rollback_triggered: bool = False

    # Claim binding
    conclusion: str = ""                    # What we concluded
    claim_boundary: str = ""                # Explicit limits of the claim
    claim_status: ClaimStatus = ClaimStatus.TESTING
    evidence_artifacts: list[str] = field(default_factory=list)

    # Metadata
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    reviewer_notes: str = ""

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id, "idea_id": self.idea_id,
            "task_id": self.task_id, "run_id": self.run_id,
            "implementation_summary": self.implementation_summary,
            "code_changes": self.code_changes, "config_changes": self.config_changes,
            "metric_name": self.metric_name, "metric_direction": self.metric_direction,
            "baseline_value": self.baseline_value,
            "experiment_value": self.experiment_value, "delta": self.delta,
            "ablation_results": self.ablation_results,
            "ablation_passed": self.ablation_passed,
            "risk_checks_passed": self.risk_checks_passed,
            "rollback_triggered": self.rollback_triggered,
            "conclusion": self.conclusion, "claim_boundary": self.claim_boundary,
            "claim_status": self.claim_status.value,
            "evidence_artifacts": self.evidence_artifacts,
            "started_at": self.started_at, "finished_at": self.finished_at,
            "reviewer_notes": self.reviewer_notes
        }


@dataclass
class ClaimAudit:
    """
    Post-experiment audit to detect claim drift.

    XCIENTIST Section 3.4.3: Controlling Claim Boundaries.
    Verifies that the implementation actually supports the claimed mechanism.
    """
    audit_id: str
    experiment_id: str
    idea_id: str

    # Audit checks
    mechanism_present_in_code: bool = False     # Is the claimed mechanism in the code?
    metric_improvement_attributable: bool = False  # Can we attribute gain to the mechanism?
    ablation_consistent: bool = False            # Do ablations confirm the mechanism?
    no_data_leakage: bool = False               # No target leakage
    submission_valid: bool = False              # Submission format correct

    # Drift detection
    claim_drift_detected: bool = False
    drift_description: str = ""
    repair_needed: bool = False
    repair_description: str = ""

    # Final verdict
    audit_passed: bool = False
    auditor_notes: str = ""
    audited_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "audit_id": self.audit_id, "experiment_id": self.experiment_id,
            "idea_id": self.idea_id,
            "mechanism_present_in_code": self.mechanism_present_in_code,
            "metric_improvement_attributable": self.metric_improvement_attributable,
            "ablation_consistent": self.ablation_consistent,
            "no_data_leakage": self.no_data_leakage,
            "submission_valid": self.submission_valid,
            "claim_drift_detected": self.claim_drift_detected,
            "drift_description": self.drift_description,
            "repair_needed": self.repair_needed,
            "repair_description": self.repair_description,
            "audit_passed": self.audit_passed,
            "auditor_notes": self.auditor_notes,
            "audited_at": self.audited_at
        }
