from __future__ import annotations

import json
from pathlib import Path

from scripts import queue_leaf_after_jigsaw_confirmation as early


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_jigsaw_snapshot_accepts_passed_or_failed_terminal_confirmation(
    tmp_path: Path,
) -> None:
    status_path = tmp_path / "jigsaw.json"
    control = {
        "jigsaw_dependency": {
            "status_path": str(status_path),
            "status_schema": "evomind.jigsaw.early_confirmation_queue.v1",
            "control_plan": {"sha256": "control-hash"},
        }
    }
    for status in (
        "confirmation_passed_human_gate_pending",
        "confirmation_failed",
        "training_failed",
    ):
        write_json(
            status_path,
            {
                "schema": control["jigsaw_dependency"]["status_schema"],
                "status": status,
                "control_plan_sha256": "control-hash",
                "process_signals_sent": 0,
                "private_labels_used": False,
                "official_grader_executed": False,
                "kaggle_submission_executed": False,
            },
        )
        assert early.jigsaw_snapshot(control)["ready"] is True


def test_jigsaw_snapshot_rejects_waiting_or_boundary_drift(tmp_path: Path) -> None:
    status_path = tmp_path / "jigsaw.json"
    control = {
        "jigsaw_dependency": {
            "status_path": str(status_path),
            "status_schema": "evomind.jigsaw.early_confirmation_queue.v1",
            "control_plan": {"sha256": "control-hash"},
        }
    }
    write_json(
        status_path,
        {
            "schema": control["jigsaw_dependency"]["status_schema"],
            "status": "waiting_for_may2022_verification",
            "control_plan_sha256": "control-hash",
            "process_signals_sent": 1,
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        },
    )
    snapshot = early.jigsaw_snapshot(control)
    assert snapshot["ready"] is False
    assert snapshot["checks"]["terminal"] is False
    assert snapshot["checks"]["signals"] is False


def test_original_leaf_queue_guard_allows_only_prerequisite_wait(tmp_path: Path) -> None:
    status_path = tmp_path / "leaf.json"
    control = {"original_after_siim_queue_status": str(status_path)}
    payload = {
        "status": early.ORIGINAL_SAFE_STATUS,
        "pid": 123,
        "process_signals_sent": 0,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }
    write_json(status_path, payload)
    assert early.original_queue_snapshot(control)["safe_for_early_queue"] is True
    payload["status"] = "waiting_for_stable_gpu_idle"
    write_json(status_path, payload)
    assert early.original_queue_snapshot(control)["safe_for_early_queue"] is False


def test_status_payload_preserves_zero_side_effect_invariants() -> None:
    control = {
        "_path": "control.json",
        "_sha256": "control-hash",
        "_base": {"_plan_path": "leaf.json", "_plan_sha256": "leaf-hash"},
    }
    payload = early.status_payload(control, "waiting")
    assert payload["process_signals_sent"] == 0
    assert payload["private_labels_used"] is False
    assert payload["official_grader_executed"] is False
    assert payload["kaggle_submission_executed"] is False
