"""Fast contracts for the serial Spooky-after-Leaf GPU queue."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import queue_spooky_after_leaf as queue


def test_parse_compute_apps_ignores_non_pid_rows():
    parsed = queue.parse_compute_apps(
        "81104, C:\\python.exe\nnot-a-pid, ignored\n123, chrome.exe\n"
    )
    assert parsed == [
        {"pid": 81104, "process_name": "C:\\python.exe"},
        {"pid": 123, "process_name": "chrome.exe"},
    ]


def test_leaf_snapshot_accepts_both_terminal_public_gate_outcomes(tmp_path):
    plan = {
        "execution": {
            "serial_dependency": {
                "competition_id": "leaf-classification",
                "run_id": "leaf-fixture",
                "report": str(tmp_path / "leaf.json"),
                "terminal_statuses": ["promotion_gate_passed", "promotion_gate_failed"],
            }
        }
    }
    report = {
        "schema": "evomind.leaf.multibackbone_oof.v1",
        "competition_id": "leaf-classification",
        "run_id": "leaf-fixture",
        "status": "promotion_gate_failed",
        "full_public_train_scope": True,
        "private_labels_used": False,
        "private_scores_used_for_tuning": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "official_score_claimed": False,
    }
    path = Path(plan["execution"]["serial_dependency"]["report"])
    path.write_text(json.dumps(report), encoding="utf-8")
    snapshot = queue.leaf_snapshot(plan)
    assert snapshot["ready"] is True
    assert all(snapshot["checks"].values())


def test_build_command_contains_frozen_runner_but_no_grader_or_submission():
    plan = queue.validate_frozen_plan(queue.DEFAULT_PLAN)
    command = queue.build_training_command(plan)
    joined = " ".join(command).lower()
    assert "run_spooky_transformer_oof.py" in joined
    assert "official_grader" not in joined
    assert "kaggle" not in joined
    assert command[command.index("--seed") + 1] == "42"
    assert command[command.index("--folds") + 1] == "0,1,2,3,4"
