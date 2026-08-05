from __future__ import annotations

import json
from pathlib import Path

from scripts import queue_jigsaw_confirmation_after_may as early


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def dependency(tmp_path: Path) -> dict:
    return {
        "watcher_path": str(tmp_path / "may_watcher.json"),
        "watcher_schema": "evomind.local_candidate_verification_watcher.v1",
        "run_id": "may-run",
        "plan": {"path": str(tmp_path / "may_plan.json"), "sha256": "plan-hash"},
    }


def test_may_verification_snapshot_accepts_verified_run_even_when_gate_failed(
    tmp_path: Path,
) -> None:
    dep = dependency(tmp_path)
    write_json(
        Path(dep["watcher_path"]),
        {
            "schema": dep["watcher_schema"],
            "status": "verification_passed",
            "profile": "may2022",
            "run_id": dep["run_id"],
            "plan_sha256": dep["plan"]["sha256"],
            "candidate_ready": False,
            "process_signals_sent": 0,
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        },
    )

    snapshot = early.may_verification_snapshot({"may2022_dependency": dep})

    assert snapshot["ready"] is True
    assert snapshot["candidate_ready"] is False
    assert all(snapshot["checks"].values())


def test_may_verification_snapshot_rejects_unverified_or_signal_drift(
    tmp_path: Path,
) -> None:
    dep = dependency(tmp_path)
    write_json(
        Path(dep["watcher_path"]),
        {
            "schema": dep["watcher_schema"],
            "status": "waiting_for_training_completion",
            "profile": "may2022",
            "run_id": dep["run_id"],
            "plan_sha256": dep["plan"]["sha256"],
            "process_signals_sent": 1,
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        },
    )

    snapshot = early.may_verification_snapshot({"may2022_dependency": dep})

    assert snapshot["ready"] is False
    assert snapshot["checks"]["status"] is False
    assert snapshot["checks"]["signals"] is False


def test_original_queue_guard_allows_only_waiting_for_leaf(tmp_path: Path) -> None:
    path = tmp_path / "original.json"
    control = {"original_after_leaf_queue_status": str(path)}
    valid = {
        "status": early.ORIGINAL_SAFE_STATUS,
        "pid": 123,
        "process_signals_sent": 0,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }
    write_json(path, valid)
    assert early.original_queue_snapshot(control)["safe_for_early_queue"] is True

    valid["status"] = "waiting_for_stable_gpu_idle_seed_40"
    write_json(path, valid)
    assert early.original_queue_snapshot(control)["safe_for_early_queue"] is False


def test_status_payload_preserves_zero_side_effect_invariants() -> None:
    control = {
        "_path": "control.json",
        "_sha256": "control-hash",
        "_base": {"_path": "base.json", "_sha256": "base-hash"},
    }

    payload = early.queue_status(control, "waiting")

    assert payload["process_signals_sent"] == 0
    assert payload["private_labels_used"] is False
    assert payload["official_grader_executed"] is False
    assert payload["kaggle_submission_executed"] is False
