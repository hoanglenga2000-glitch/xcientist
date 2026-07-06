"""Tests for XCIENTIST-style validation contracts (claim acceptance + artifacts)."""
from __future__ import annotations

from research_os.validation_contract import (
    check_required_artifacts,
    create_contract,
    evaluate_acceptance,
)


def _contract(**kwargs):
    base = dict(
        contract_id="c1",
        exp_id="exp1",
        claim="CatBoost beats baseline",
        hypothesis="gradient boosting improves OOF",
        implementation_requirement="train CatBoost with 5-fold CV",
        metric="accuracy",
        baseline_exp_id="exp0",
    )
    base.update(kwargs)
    return create_contract(**base)


def test_required_artifacts_all_present():
    contract = _contract(required_artifacts=["oof.json", "submission.csv"])
    out = check_required_artifacts(contract, ["oof.json", "submission.csv", "extra.log"])
    assert out["passed"] is True
    assert out["missing_artifacts"] == []


def test_required_artifacts_missing():
    contract = _contract(required_artifacts=["oof.json", "submission.csv"])
    out = check_required_artifacts(contract, ["oof.json"])
    assert out["passed"] is False
    assert out["missing_artifacts"] == ["submission.csv"]


def test_acceptance_min_threshold_pass_and_fail():
    contract = _contract(acceptance_criteria={"accuracy": {"min": 0.8}})
    assert evaluate_acceptance(contract, {"accuracy": 0.82})["passed"] is True
    assert evaluate_acceptance(contract, {"accuracy": 0.78})["passed"] is False


def test_acceptance_missing_metric_fails():
    contract = _contract(acceptance_criteria={"accuracy": {"min": 0.8}})
    out = evaluate_acceptance(contract, {})
    assert out["passed"] is False


def test_acceptance_max_and_equals():
    contract = _contract(acceptance_criteria={"rmse": {"max": 0.5}, "folds": {"equals": 5}})
    assert evaluate_acceptance(contract, {"rmse": 0.4, "folds": 5})["passed"] is True
    assert evaluate_acceptance(contract, {"rmse": 0.6, "folds": 5})["passed"] is False
    assert evaluate_acceptance(contract, {"rmse": 0.4, "folds": 4})["passed"] is False


def test_acceptance_scalar_equality():
    contract = _contract(acceptance_criteria={"status": "passed"})
    assert evaluate_acceptance(contract, {"status": "passed"})["passed"] is True
    assert evaluate_acceptance(contract, {"status": "failed"})["passed"] is False


def test_checks_are_reported_per_criterion():
    contract = _contract(acceptance_criteria={"accuracy": {"min": 0.8}, "auc": {"min": 0.9}})
    out = evaluate_acceptance(contract, {"accuracy": 0.85, "auc": 0.88})
    assert len(out["checks"]) == 2
    by_name = {c["criterion"]: c for c in out["checks"]}
    assert by_name["accuracy"]["passed"] is True
    assert by_name["auc"]["passed"] is False
