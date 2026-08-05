from __future__ import annotations

import json
from pathlib import Path

from scripts import watch_siim_final_candidate_completion as watcher


def test_default_queue_status_tracks_calibrated_queue() -> None:
    assert watcher.DEFAULT_QUEUE_STATUS.name == "siim_final_candidate_queue_calibrated.json"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_completion_snapshot_accepts_candidate_only_terminal_pair(tmp_path: Path):
    plan = {
        "competition_id": "siim-isic-melanoma-classification",
        "training": {"run_id": "frozen-run"},
    }
    run_dir = tmp_path / "frozen-run"
    write_json(
        run_dir / "summary.json",
        {
            "run_id": "frozen-run",
            "status": "candidate_complete",
            "competition_count": 1,
        },
    )
    write_json(
        run_dir / "siim-isic-melanoma-classification" / "result.json",
        {
            "competition_id": "siim-isic-melanoma-classification",
            "status": "promotion_gate_passed_confirmation_pending",
            "candidate_only": True,
            "official_grader_executed": False,
            "official_grader_withheld": True,
            "kaggle_public_score": None,
            "kaggle_private_score": None,
        },
    )

    snapshot = watcher.completion_snapshot(plan, run_dir)

    assert snapshot["ready"] is True
    assert watcher.stable_key(snapshot) is not None


def test_completion_snapshot_rejects_grader_or_kaggle_evidence(tmp_path: Path):
    plan = {
        "competition_id": "siim-isic-melanoma-classification",
        "training": {"run_id": "frozen-run"},
    }
    run_dir = tmp_path / "frozen-run"
    write_json(
        run_dir / "summary.json",
        {
            "run_id": "frozen-run",
            "status": "candidate_complete",
            "competition_count": 1,
        },
    )
    write_json(
        run_dir / "siim-isic-melanoma-classification" / "result.json",
        {
            "competition_id": "siim-isic-melanoma-classification",
            "status": "promotion_gate_passed_confirmation_pending",
            "candidate_only": True,
            "official_grader_executed": True,
            "official_grader_withheld": False,
            "kaggle_public_score": 0.9,
            "kaggle_private_score": None,
        },
    )

    snapshot = watcher.completion_snapshot(plan, run_dir)

    assert snapshot["ready"] is False
    assert watcher.stable_key(snapshot) is None


def test_verifier_command_contains_only_candidate_verification_paths(tmp_path: Path):
    plan = {
        "_python_path": str(tmp_path / "python.exe"),
        "_verifier_path": str(tmp_path / "verify_siim.py"),
        "_plan_path": str(tmp_path / "plan.json"),
        "_run_dir": str(tmp_path / "run"),
    }
    command = watcher.build_verifier_command(
        plan,
        queue_status=tmp_path / "queue.json",
        output=tmp_path / "run" / "independent_verification.json",
    )
    joined = " ".join(command).lower()

    assert "verify_siim.py" in joined
    assert "official_grader" not in joined
    assert "kaggle" not in joined
    assert "--queue-status" in command
    assert "--output" in command


def test_invariant_fields_are_zero_signal_candidate_only():
    assert watcher.invariant_fields() == {
        "process_signals_sent": 0,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }
