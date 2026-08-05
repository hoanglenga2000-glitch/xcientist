from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "generate_jigsaw_gpt56_audit",
    ROOT / "scripts" / "generate_jigsaw_gpt56_audit.py",
)
assert SPEC and SPEC.loader
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)


def jigsaw_summary() -> dict:
    return {
        "results": [
            {
                "competition_id": audit_module.JIGSAW_ID,
                "promotion_gate": {
                    "schema": "evomind.mlebench_lite.metric_promotion_gate.v1",
                    "threshold": 0.987,
                    "internal_score": 0.9855174292532056,
                    "official_grader_executed": False,
                    "checks": {"aggregate_threshold": False},
                    "evidence": {"cross_fitted_blend_improvement": 0.0013541282931751608},
                },
                "budget": {
                    "seed": 42,
                    "train_rows": 159571,
                    "test_rows": 153164,
                    "folds": 5,
                    "feature_counts": [{"word": 350000, "char": 500000}],
                    "nbsvm_c": 4.0,
                    "fit_workers": 12,
                    "split_strategy": "iterative_multilabel_stratification",
                },
                "per_label_oof": {"toxic": {"cross_fitted_blend_auc": 0.9808}},
            }
        ]
    }


def valid_audit() -> dict:
    return {
        "diagnosis": ["The two current channels do not provide enough independent signal."],
        "ranked_upgrades": [
            {
                "id": "label_stack",
                "expected_oof_auc_gain_low": 0.0004,
                "expected_oof_auc_gain_high": 0.0018,
                "compute_cost": "low",
                "source_changes": ["Add a cross-fitted label-correlation stacker."],
                "validation_contract": ["Every held-out fold is predicted by a meta-model fit elsewhere."],
                "stop_rule": "Stop when complete OOF fails to improve.",
            }
        ],
        "recommended_minimal_patch": {
            "functions_to_add_or_change": ["cross_fit_multilabel_stacker"],
            "algorithm": ["Fit on non-held-out OOF rows."],
            "new_cli_arguments": ["--wave2-jigsaw-stacker-c=0.1"],
            "artifact_changes": ["Persist stacker OOF and coefficients."],
            "acceptance_tests": ["Synthetic leakage sentinel."],
        },
        "experiment_sequence": ["Evaluate the immutable current OOF artifact first."],
        "go_no_go": {
            "implementation_verdict": "GO",
            "reason": "The experiment is cheap and leakage-safe.",
            "minimum_required_observed_gain": 0.0014825708,
        },
    }


def test_prompt_is_compact_source_grounded_and_truthful() -> None:
    prompt = audit_module.build_prompt(jigsaw_summary())
    assert "0.9855174292532056" in prompt
    assert "0.0014825708" in prompt
    assert "cross_fit_multilabel_rank_blend" in prompt
    assert "run_jigsaw" in prompt
    assert "official MLE-Bench private-grader" in prompt
    assert len(prompt) < 45_000


def test_validate_audit_accepts_complete_contract() -> None:
    audit_module.validate_audit(valid_audit())


def test_validate_audit_rejects_missing_ranked_upgrades() -> None:
    payload = valid_audit()
    payload["ranked_upgrades"] = []
    with pytest.raises(ValueError, match="ranked_upgrades"):
        audit_module.validate_audit(payload)


def test_find_jigsaw_result_requires_withheld_official_grader() -> None:
    payload = jigsaw_summary()
    payload["results"][0]["promotion_gate"]["official_grader_executed"] = True
    with pytest.raises(ValueError, match="withheld"):
        audit_module.find_jigsaw_result(payload)
