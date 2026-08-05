from __future__ import annotations

import json
from pathlib import Path

from scripts import watch_spooky_cross_run_meta_completion as watcher


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _plan(tmp_path: Path) -> tuple[dict, Path, Path]:
    current_plan = tmp_path / "current_plan.json"
    _write_json(current_plan, {"schema": "synthetic.current.plan.v1"})
    summary = tmp_path / "summary.json"
    plan = {
        "current": {
            "run_id": "current-run",
            "summary_path": str(summary),
            "plan": {"sha256": watcher.sha256_file(current_plan)},
        },
        "output": {"run_id": "cross-run"},
        "_plan_sha256": "plan-hash",
    }
    return plan, current_plan, summary


def test_completion_snapshot_waits_for_missing_summary(tmp_path: Path):
    plan, _, _ = _plan(tmp_path)
    snapshot = watcher.completion_snapshot(plan)
    assert snapshot["ready"] is False
    assert snapshot["summary_present"] is False
    assert watcher.stable_key(snapshot) is None


def test_completion_snapshot_accepts_zero_signal_terminal_bundle(tmp_path: Path):
    plan, current_plan, summary = _plan(tmp_path)
    bundle = tmp_path / "bundle.npz"
    bundle.write_bytes(b"synthetic-numeric-bundle")
    _write_json(
        summary,
        {
            "status": "single_seed_gate_failed",
            "run_id": "current-run",
            "plan_sha256": watcher.sha256_file(current_plan),
            "prediction_bundle": {
                "path": str(bundle),
                "sha256": watcher.sha256_file(bundle),
            },
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "process_signals_sent": 0,
        },
    )

    snapshot = watcher.completion_snapshot(plan)

    assert snapshot["ready"] is True
    assert snapshot["boundary_ok"] is True
    assert snapshot["bundle_hash_matches"] is True
    assert watcher.stable_key(snapshot) is not None


def test_completion_snapshot_rejects_boundary_or_bundle_drift(tmp_path: Path):
    plan, current_plan, summary = _plan(tmp_path)
    bundle = tmp_path / "bundle.npz"
    bundle.write_bytes(b"bundle")
    payload = {
        "status": "single_seed_gate_passed_confirmation_pending",
        "run_id": "current-run",
        "plan_sha256": watcher.sha256_file(current_plan),
        "prediction_bundle": {
            "path": str(bundle),
            "sha256": watcher.sha256_file(bundle),
        },
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 1,
    }
    _write_json(summary, payload)
    assert watcher.completion_snapshot(plan)["ready"] is False

    payload["process_signals_sent"] = 0
    payload["prediction_bundle"]["sha256"] = "0" * 64
    _write_json(summary, payload)
    assert watcher.completion_snapshot(plan)["ready"] is False


def test_evaluator_command_is_candidate_only(tmp_path: Path):
    command = watcher.build_evaluator_command(
        tmp_path / "python.exe",
        tmp_path / "plan.json",
        tmp_path / "evaluate_spooky_cross_run_meta.py",
    )
    joined = " ".join(command).lower()
    assert "evaluate_spooky_cross_run_meta.py" in joined
    assert "--frozen-plan" in command
    assert "official_grader" not in joined
    assert "kaggle" not in joined


def test_result_validation_requires_exact_once_human_gated_contract():
    plan = {"output": {"run_id": "cross-run"}, "_plan_sha256": "plan-hash"}
    result = {
        "schema": watcher.EXPECTED_RESULT_SCHEMA,
        "status": "single_seed_gate_failed",
        "run_id": "cross-run",
        "plan_sha256": "plan-hash",
        "checks": {
            "truth_and_folds_identical": True,
            "public_csv_truth_identical": True,
            "exact_once_oof": True,
            "finite_normalized_oof": True,
            "finite_normalized_test": True,
        },
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
        "promotion_allowed": False,
        "human_gate_preserved": True,
    }
    assert watcher.result_valid(result, plan) is True
    result["checks"]["exact_once_oof"] = False
    assert watcher.result_valid(result, plan) is False


def test_cpu_environment_hides_cuda_and_invariants_are_zero_signal():
    assert watcher.cpu_only_environment()["CUDA_VISIBLE_DEVICES"] == ""
    assert watcher.invariant_fields() == {
        "resource": "CPU only",
        "process_signals_sent": 0,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "human_gate_preserved": True,
    }
