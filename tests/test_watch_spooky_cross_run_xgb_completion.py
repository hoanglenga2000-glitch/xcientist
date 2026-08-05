"""Tests for the frozen Spooky cross-run XGBoost completion watcher."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts import watch_spooky_cross_run_xgb_completion as watcher


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_completion_snapshot_requires_terminal_hash_valid_boundary(tmp_path: Path):
    current_plan = tmp_path / "current_plan.json"
    _write(current_plan, {"schema": "current"})
    bundle = tmp_path / "bundle.npz"
    bundle.write_bytes(b"stable-bundle")
    summary = tmp_path / "summary.json"
    _write(
        summary,
        {
            "status": "single_seed_gate_failed",
            "run_id": "current-run",
            "plan_sha256": _sha(current_plan),
            "prediction_bundle": {"path": str(bundle), "sha256": _sha(bundle)},
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "process_signals_sent": 0,
        },
    )
    plan = {
        "current": {
            "run_id": "current-run",
            "summary_path": str(summary),
            "plan": {"path": str(current_plan), "sha256": _sha(current_plan)},
        }
    }

    snapshot = watcher.completion_snapshot(plan)

    assert snapshot["ready"] is True
    assert snapshot["bundle_hash_matches"] is True
    assert snapshot["boundary_ok"] is True
    assert watcher.stable_key(snapshot) is not None


def test_completion_snapshot_rejects_process_signal(tmp_path: Path):
    current_plan = tmp_path / "current_plan.json"
    _write(current_plan, {"schema": "current"})
    bundle = tmp_path / "bundle.npz"
    bundle.write_bytes(b"stable-bundle")
    summary = tmp_path / "summary.json"
    _write(
        summary,
        {
            "status": "single_seed_gate_failed",
            "run_id": "current-run",
            "plan_sha256": _sha(current_plan),
            "prediction_bundle": {"path": str(bundle), "sha256": _sha(bundle)},
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "process_signals_sent": 1,
        },
    )
    plan = {
        "current": {
            "run_id": "current-run",
            "summary_path": str(summary),
            "plan": {"path": str(current_plan), "sha256": _sha(current_plan)},
        }
    }

    snapshot = watcher.completion_snapshot(plan)

    assert snapshot["ready"] is False
    assert snapshot["boundary_ok"] is False
    assert watcher.stable_key(snapshot) is None


def test_result_validation_preserves_human_gate():
    plan = {"_plan_sha256": "p" * 64, "output": {"run_id": "xgb-run"}}
    result = {
        "schema": "evomind.spooky.cross_run_xgb_result.v1",
        "run_id": "xgb-run",
        "plan": {"sha256": "p" * 64},
        "checks": {"exact_once": True, "normalized": True},
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
        "promotion_allowed": False,
    }

    assert watcher.result_valid(result, plan) is True
    result["promotion_allowed"] = True
    assert watcher.result_valid(result, plan) is False


def test_cpu_environment_hides_gpu_and_limits_threads():
    environment = watcher.cpu_only_environment()

    assert environment["CUDA_VISIBLE_DEVICES"] == "-1"
    assert environment["OMP_NUM_THREADS"] == "2"
    assert environment["MKL_NUM_THREADS"] == "2"
