"""XCIENTIST-style validation contracts for experiment claims."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationContract:
    contract_id: str
    exp_id: str
    claim: str
    hypothesis: str
    implementation_requirement: str
    metric: str
    baseline_exp_id: str
    acceptance_criteria: dict[str, Any] = field(default_factory=dict)
    ablation_plan: list[str] = field(default_factory=list)
    risk_checklist: list[dict[str, Any]] = field(default_factory=list)
    conclusion_boundary: str = ""
    required_artifacts: list[str] = field(default_factory=list)


def create_contract(
    contract_id: str,
    exp_id: str,
    claim: str,
    hypothesis: str,
    implementation_requirement: str,
    metric: str,
    baseline_exp_id: str,
    acceptance_criteria: dict[str, Any] | None = None,
    ablation_plan: list[str] | None = None,
    risk_checklist: list[dict[str, Any]] | None = None,
    conclusion_boundary: str = "",
    required_artifacts: list[str] | None = None,
) -> ValidationContract:
    return ValidationContract(
        contract_id=contract_id,
        exp_id=exp_id,
        claim=claim,
        hypothesis=hypothesis,
        implementation_requirement=implementation_requirement,
        metric=metric,
        baseline_exp_id=baseline_exp_id,
        acceptance_criteria=acceptance_criteria or {},
        ablation_plan=ablation_plan or [],
        risk_checklist=risk_checklist or [],
        conclusion_boundary=conclusion_boundary,
        required_artifacts=required_artifacts or [],
    )


def check_required_artifacts(contract: ValidationContract, available_artifacts: list[str]) -> dict[str, Any]:
    available = set(available_artifacts)
    missing = [artifact for artifact in contract.required_artifacts if artifact not in available]
    return {"passed": not missing, "missing_artifacts": missing}


def evaluate_acceptance(contract: ValidationContract, metrics: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {"passed": True, "checks": []}
    for key, expected in contract.acceptance_criteria.items():
        actual = metrics.get(key)
        passed = True
        if isinstance(expected, dict):
            if "min" in expected and (actual is None or float(actual) < float(expected["min"])):
                passed = False
            if "max" in expected and (actual is None or float(actual) > float(expected["max"])):
                passed = False
            if "equals" in expected and actual != expected["equals"]:
                passed = False
        else:
            passed = actual == expected

        results["checks"].append({"criterion": key, "expected": expected, "actual": actual, "passed": passed})
        results["passed"] = results["passed"] and passed
    return results
