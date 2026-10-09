from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


updater = _load("update_g21_conditional_waiting_gates", ROOT / "scripts/update_g21_conditional_waiting_gates.py")
item_builder = _load(
    "build_g21_open_polymer_legacy_adoption_item_r118_for_gate_update",
    ROOT / "scripts/build_g21_open_polymer_legacy_adoption_item_r118.py",
)


def _goal() -> dict:
    board = json.loads(
        (ROOT / "configs/g21_five_competition_goal_board_bootstrap.json").read_text(encoding="utf-8")
    )
    board.update(
        baseline_policy=updater.POLICY_MODE,
        policy_evidence_sha256=updater.POLICY_EVIDENCE_SHA256,
    )
    by_name = {row["competition"]: row for row in board["competitions"]}
    by_name["open_polymer"].clear()
    by_name["open_polymer"].update(item_builder.build_item())
    board.pop("completion_count", None)
    board.pop("goal_record_status", None)
    assert updater.canonical_sha256(board) == updater.EXPECTED_CURRENT_BOARD_SHA256
    return {
        "id": updater.FIXED_GOAL_ID,
        "run_id": updater.FIXED_RUN_ID,
        "allocation": "G21",
        "status": "blocked",
        "spec_sha256": updater.FIXED_SPEC_SHA256,
        "board_sha256": updater.EXPECTED_CURRENT_BOARD_SHA256,
        "board": board,
    }


def test_build_board_removes_only_nonblocking_human_labels() -> None:
    goal = _goal()
    original_open = deepcopy(next(row for row in goal["board"]["competitions"] if row["competition"] == "open_polymer"))
    board = updater.build_board(goal)
    by_name = {row["competition"]: row for row in board["competitions"]}
    assert by_name["open_polymer"] == original_open
    assert board["completion_count"] == 1
    assert board["goal_record_status"] == "blocked"
    for competition in updater.AVAILABILITY:
        row = by_name[competition]
        assert row["human_baseline_availability"]["status"] == "NOT_VERIFIED_AVAILABLE"
        assert "HUMAN_BASELINE_UNDEFINED" not in row["additional_gates"]
    assert by_name["cure_bench"]["exact_gate"] == "PRETRAINED_ENCODER_MANIFEST_MISSING"
    assert by_name["mindgames"]["exact_gate"] == "RUNTIME_DEPENDENCY_MISSING:trueskill"


def test_build_board_rejects_wrong_current_hash_or_open_polymer_drift() -> None:
    wrong = _goal()
    wrong["board_sha256"] = "0" * 64
    with pytest.raises(updater.GateUpdateError, match="precondition"):
        updater.build_board(wrong)
    drift = _goal()
    next(row for row in drift["board"]["competitions"] if row["competition"] == "open_polymer")[
        "official_metric_claimed"
    ] = True
    with pytest.raises(updater.GateUpdateError, match="Open Polymer"):
        updater.build_board(drift)


def test_updater_source_is_loopback_only_and_has_no_training_or_sqlite_surface() -> None:
    text = (ROOT / "scripts/update_g21_conditional_waiting_gates.py").read_text(encoding="utf-8")
    assert "127.0.0.1" in text and "http.client" in text
    for forbidden in ("sqlite3", "hpc_verify", "hpc_execute_solution", "competition_data_", "kaggle_submit"):
        assert forbidden not in text
