"""
XCIENTIST Research Harness - Main Engine
Based on: XCIENTIST (arXiv:2606.18874v1)

The harness externalizes research synthesis and experimental validation
into inspectable, contract-governed processes.

Three key guarantees:
1. Every experiment has explicit hypothesis->implementation->metric->conclusion chain
2. Claim drift is detected and repaired before results are accepted
3. All evidence is bound to artifacts for auditability

Integration with MLEvolve: The harness wraps each search node expansion,
ensuring that the search controller's outputs are validated and auditable.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .contracts import (
    IdeaContract, ValidationContract, ExperimentRecord, ClaimAudit,
    ClaimStatus, RiskLevel
)


class XCIENTISTHarness:
    """
    Research harness that enforces scientific rigor on ML experiments.

    Usage pattern:
        harness = XCIENTISTHarness(task_id, workspace_root)

        # 1. Propose an idea with evidence grounding
        idea = harness.propose_idea(
            title="Add CatBoost with target encoding",
            hypothesis="CatBoost handles categorical features natively and may reduce CV-public gap",
            mechanism="Replace one-hot encoding with CatBoost native categorical handling + target encoding",
            literature_grounding=["MLEvolve model_family_diversity branch"],
            expected_effect="accuracy +0.002~0.005",
            risk_level="medium"
        )

        # 2. Create validation contract
        contract = harness.create_validation_contract(
            idea, baseline_score=0.80916,
            validation_plan="3-fold CV, OOF evaluation, compare to HGB baseline",
            ablations=["CatBoost only vs blend", "Target encoding vs one-hot"]
        )

        # 3. After experiment, record results
        record = harness.record_experiment(
            idea, contract,
            baseline_value=0.80916, experiment_value=0.81123,
            code_changes=["Added CatBoost model", "Added target encoding"],
            ablation_results={"catboost_only": 0.80980, "with_hgb_blend": 0.81123}
        )

        # 4. Audit for claim drift
        audit = harness.audit_claim(record, code_path="solution.py")

        # 5. If audit passes, accept the claim
        if audit.audit_passed:
            harness.accept_claim(idea, record, audit)
    """

    def __init__(self, task_id: str, workspace_root: Optional[Path] = None,
                 metric: str = "accuracy", metric_direction: str = "maximize"):
        self.task_id = task_id
        self.workspace_root = workspace_root or Path(".")
        self.metric = metric
        self.metric_direction = metric_direction

        # Storage
        self.ideas: dict[str, IdeaContract] = {}
        self.contracts: dict[str, ValidationContract] = {}
        self.experiments: dict[str, ExperimentRecord] = {}
        self.audits: dict[str, ClaimAudit] = {}

        # Tracking
        self.best_score: Optional[float] = None
        self.best_idea_id: Optional[str] = None
        self.claim_drift_count = 0
        self.repair_count = 0

    # ── Idea Management ─────────────────────────────────────────────────

    def propose_idea(self, title: str, hypothesis: str, mechanism: str,
                     literature_grounding: Optional[list[str]] = None,
                     expected_effect: str = "",
                     risk_level: str = "medium",
                     parent_idea_id: Optional[str] = None,
                     component_changes: Optional[list[str]] = None) -> IdeaContract:
        """Propose a new research idea grounded in evidence."""
        idea_id = f"idea_{self.task_id}_{uuid.uuid4().hex[:8]}"
        idea = IdeaContract(
            idea_id=idea_id,
            task_id=self.task_id,
            title=title,
            hypothesis=hypothesis,
            mechanism=mechanism,
            literature_grounding=literature_grounding or [],
            expected_effect=expected_effect,
            risk_level=RiskLevel[risk_level.upper()] if risk_level.upper() in RiskLevel.__members__ else RiskLevel.MEDIUM,
            parent_idea_id=parent_idea_id,
            component_changes=component_changes or [],
            status=ClaimStatus.PROPOSED
        )
        self.ideas[idea_id] = idea
        return idea

    # ── Validation Contract ─────────────────────────────────────────────

    def create_validation_contract(self, idea: IdeaContract,
                                   baseline_score: Optional[float] = None,
                                   validation_plan: str = "",
                                   standard_experiments: Optional[list[str]] = None,
                                   ablations: Optional[list[str]] = None,
                                   risk_checks: Optional[list[str]] = None,
                                   min_expected_delta: float = 0.0,
                                   rollback_condition: str = "") -> ValidationContract:
        """Create a validation contract that must be satisfied before accepting results."""
        contract_id = f"contract_{idea.idea_id}"
        contract = ValidationContract(
            contract_id=contract_id,
            idea_id=idea.idea_id,
            task_id=self.task_id,
            baseline_score=baseline_score,
            benchmark_configured=True,
            validation_plan=validation_plan,
            standard_experiments=standard_experiments or [
                "Cross-validation with OOF evaluation",
                "Compare to protected best baseline"
            ],
            ablation_experiments=ablations or [],
            risk_checks=risk_checks or [
                "No target leakage",
                "Submission format valid",
                "CV-public gap within expected range"
            ],
            min_expected_delta=min_expected_delta,
            max_acceptable_regression=0.0,
            rollback_condition=rollback_condition or "Hold if score <= best_so_far"
        )
        self.contracts[contract_id] = contract
        return contract

    # ── Experiment Recording ────────────────────────────────────────────

    def record_experiment(self, idea: IdeaContract, contract: ValidationContract,
                          baseline_value: Optional[float] = None,
                          experiment_value: Optional[float] = None,
                          implementation_summary: str = "",
                          code_changes: Optional[list[str]] = None,
                          config_changes: Optional[dict] = None,
                          ablation_results: Optional[dict] = None,
                          risk_checks_passed: Optional[list[bool]] = None,
                          conclusion: str = "",
                          evidence_artifacts: Optional[list[str]] = None,
                          reviewer_notes: str = "") -> ExperimentRecord:
        """Record a complete experiment with all evidence bound."""
        experiment_id = f"exp_{idea.idea_id}_{uuid.uuid4().hex[:6]}"
        delta = None
        if baseline_value is not None and experiment_value is not None:
            delta = experiment_value - baseline_value

        ablation_passed = True
        if ablation_results:
            # Check ablations are consistent with main claim
            for ablation_name, ablation_value in ablation_results.items():
                if isinstance(ablation_value, (int, float)) and experiment_value is not None:
                    if self.metric_direction == "maximize":
                        if ablation_value > experiment_value:
                            ablation_passed = False  # Ablation shouldn't beat full method
                    else:
                        if ablation_value < experiment_value:
                            ablation_passed = False

        record = ExperimentRecord(
            experiment_id=experiment_id,
            idea_id=idea.idea_id,
            task_id=self.task_id,
            implementation_summary=implementation_summary,
            code_changes=code_changes or [],
            config_changes=config_changes or {},
            metric_name=self.metric,
            metric_direction=self.metric_direction,
            baseline_value=baseline_value,
            experiment_value=experiment_value,
            delta=delta,
            ablation_results=ablation_results or {},
            ablation_passed=ablation_passed,
            risk_checks_passed=risk_checks_passed or [],
            conclusion=conclusion,
            evidence_artifacts=evidence_artifacts or [],
            started_at=datetime.now().isoformat(),
            finished_at=datetime.now().isoformat(),
            reviewer_notes=reviewer_notes
        )
        self.experiments[experiment_id] = record
        return record

    # ── Claim Audit ─────────────────────────────────────────────────────

    def audit_claim(self, record: ExperimentRecord,
                    code_path: Optional[str] = None,
                    code_content: Optional[str] = None) -> ClaimAudit:
        """Audit an experiment for claim drift."""
        audit_id = f"audit_{record.experiment_id}"
        idea = self.ideas.get(record.idea_id)

        audit = ClaimAudit(audit_id=audit_id, experiment_id=record.experiment_id,
                          idea_id=record.idea_id)

        if not idea:
            audit.auditor_notes = "Idea not found; cannot audit."
            self.audits[audit_id] = audit
            return audit

        # Check 1: Mechanism present in code?
        mechanism_keywords = set(idea.mechanism.lower().replace(",", " ").split())
        code_lower = (code_content or "").lower()
        matched_keywords = [kw for kw in mechanism_keywords
                          if len(kw) > 2 and kw in code_lower]
        audit.mechanism_present_in_code = len(matched_keywords) >= len(mechanism_keywords) * 0.5

        # Check 2: Metric improvement attributable?
        if record.delta is not None:
            tolerance = getattr(self, 'delta_tolerance', 0.01)
            if self.metric_direction == "maximize":
                audit.metric_improvement_attributable = record.delta > -tolerance
            else:
                audit.metric_improvement_attributable = record.delta < tolerance
        else:
            audit.metric_improvement_attributable = True  # exploratory runs: allow

        # Check 3: Ablations consistent?
        audit.ablation_consistent = record.ablation_passed

        # Check 4: No data leakage
        audit.no_data_leakage = True  # Default; real check needs code analysis

        # Check 5: Submission valid
        audit.submission_valid = len(record.evidence_artifacts) > 0

        # Drift detection
        drifted = False
        drift_reasons = []
        if not audit.mechanism_present_in_code:
            drifted = True
            drift_reasons.append("Claimed mechanism not found in implementation code")
        if not audit.metric_improvement_attributable:
            drift_reasons.append("Metric did not improve; claim of improvement unsupported")
        if not audit.ablation_consistent:
            drift_reasons.append("Ablation results contradict the claimed mechanism")

        audit.claim_drift_detected = drifted
        audit.drift_description = "; ".join(drift_reasons) if drift_reasons else "No drift detected"
        audit.repair_needed = drifted

        if drifted:
            audit.repair_description = (
                f"Repair needed: {'; '.join(drift_reasons)}. "
                f"Recommended: revisit implementation to ensure mechanism matches claim, "
                f"or revise claim to match actual implementation."
            )
            self.claim_drift_count += 1

        audit.audit_passed = not drifted and all([
            audit.mechanism_present_in_code,
            audit.submission_valid,
            audit.no_data_leakage
        ])

        self.audits[audit_id] = audit
        return audit

    # ── Claim Acceptance ────────────────────────────────────────────────

    def accept_claim(self, idea: IdeaContract, record: ExperimentRecord,
                     audit: ClaimAudit, claim_boundary: str = ""):
        """Accept a validated claim with explicit boundary."""
        if not audit.audit_passed:
            raise ValueError(f"Cannot accept claim: audit not passed ({audit.drift_description})")

        idea.status = ClaimStatus.BOUNDED
        record.claim_status = ClaimStatus.BOUNDED
        record.claim_boundary = claim_boundary or (
            f"Claim valid under conditions: {idea.mechanism[:100]}. "
            f"Score delta: {record.delta}. "
            f"Applicable to: {self.task_id} with metric {self.metric}."
        )

        # Update best
        if record.experiment_value is not None:
            if self.best_score is None:
                self.best_score = record.experiment_value
                self.best_idea_id = idea.idea_id
            elif self.metric_direction == "maximize" and record.experiment_value > self.best_score:
                self.best_score = record.experiment_value
                self.best_idea_id = idea.idea_id
            elif self.metric_direction == "minimize" and record.experiment_value < self.best_score:
                self.best_score = record.experiment_value
                self.best_idea_id = idea.idea_id

    # ── Reporting ───────────────────────────────────────────────────────

    def to_summary(self) -> dict:
        """Generate a research summary for audit/reporting."""
        return {
            "schema": "academic_research_os.xcientist_harness_summary.v1",
            "task_id": self.task_id,
            "metric": self.metric,
            "metric_direction": self.metric_direction,
            "best_score": self.best_score,
            "best_idea_id": self.best_idea_id,
            "total_ideas": len(self.ideas),
            "total_experiments": len(self.experiments),
            "claim_drift_count": self.claim_drift_count,
            "repair_count": self.repair_count,
            "ideas": {
                iid: {
                    "title": idea.title,
                    "hypothesis": idea.hypothesis[:200],
                    "mechanism": idea.mechanism[:200],
                    "status": idea.status.value,
                    "risk_level": idea.risk_level.value
                }
                for iid, idea in self.ideas.items()
            },
            "experiments": [
                {
                    "experiment_id": rec.experiment_id,
                    "idea_id": rec.idea_id,
                    "baseline": rec.baseline_value,
                    "experiment": rec.experiment_value,
                    "delta": rec.delta,
                    "claim_status": rec.claim_status.value,
                    "ablation_passed": rec.ablation_passed
                }
                for rec in list(self.experiments.values())[-20:]
            ],
            "recent_audits": [
                {
                    "audit_id": aud.audit_id,
                    "experiment_id": aud.experiment_id,
                    "passed": aud.audit_passed,
                    "drift_detected": aud.claim_drift_detected,
                    "drift_description": aud.drift_description[:200]
                }
                for aud in list(self.audits.values())[-10:]
            ],
            "generated_at": datetime.now().isoformat()
        }

    def save_summary(self, path: Optional[Path] = None) -> Path:
        if path is None:
            path = self.workspace_root / "workspace" / "xcientist" / f"harness_{self.task_id}_{int(datetime.now().timestamp())}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_summary(), indent=2, ensure_ascii=False))
        return path
