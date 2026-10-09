from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load("adopt_open_polymer_legacy_r118", ROOT / "scripts/adopt_open_polymer_legacy_r118.py")
item_builder = _load(
    "build_g21_open_polymer_legacy_adoption_item_r118",
    ROOT / "scripts/build_g21_open_polymer_legacy_adoption_item_r118.py",
)
fixture = _load("open_polymer_legacy_fixture", ROOT / "tests/test_open_polymer_legacy_adoption_r118.py")


def _goal() -> dict:
    board = json.loads(
        (ROOT / "configs/g21_five_competition_goal_board_bootstrap.json").read_text(encoding="utf-8")
    )
    return {
        "id": runner.FIXED_GOAL_ID,
        "run_id": runner.FIXED_RUN_ID,
        "allocation": runner.FIXED_ALLOCATION,
        "status": "blocked",
        "spec_sha256": runner.FIXED_SPEC_SHA256,
        "board_sha256": runner.INITIAL_BOARD_SHA256,
        "board": board,
    }


def test_runner_constructs_exact_one_item_adoption_and_preserves_other_four() -> None:
    current = _goal()
    item = fixture._pure_item()
    board = runner.build_board(current, item)
    assert board["competitions"][3] == item
    before = {row["competition"]: row for row in current["board"]["competitions"]}
    after = {row["competition"]: row for row in board["competitions"]}
    for name in ("cure_bench", "e2lmc", "mindgames", "ariel_2025"):
        assert after[name] == before[name]
    assert board["baseline_policy"] == runner.POLICY_MODE
    assert board["weather_actions"] == 0


def test_production_item_builder_closes_exact_historical_evidence() -> None:
    item = item_builder.build_item()
    runner.validate_item(item)
    assert {entry["name"] for entry in item["artifact_manifest"]["files"]} == {
        name for name, _, _, _ in item_builder.FILES
    }
    assert set(item["legacy_artifact_bindings"]) == {
        name for name, _, _, _ in item_builder.FILES
    }
    assert item["existing_candidate_adoption"]["source_idempotency_key"] == "open-polymer-v2-baseline-candidate-1"
    assert "holdout_ledger" not in item and "holdout_claim" not in item


def test_runner_rejects_posthoc_ledger_claim_and_wrong_initial_board() -> None:
    item = fixture._pure_item()
    item["holdout_ledger"] = {"unconsumed_at_claim": True}
    with pytest.raises(runner.AdoptionGate, match="posthoc"):
        runner.build_board(_goal(), item)
    wrong = _goal()
    wrong["board_sha256"] = "0" * 64
    with pytest.raises(runner.AdoptionGate, match="precondition"):
        runner.build_board(wrong, fixture._pure_item())


def test_validate_only_never_calls_goal_api(tmp_path: Path) -> None:
    goal = tmp_path / "goal.json"
    item = tmp_path / "item.json"
    goal.write_text(json.dumps(_goal()), encoding="utf-8")
    item.write_text(json.dumps(fixture._pure_item()), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/adopt_open_polymer_legacy_r118.py"),
            "--adoption-item",
            str(item),
            "--fixture-goal",
            str(goal),
            "--validate-only",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    result = json.loads(completed.stdout)
    assert result["status"] == "validated_not_applied"
    assert result["production_action_performed"] is False
    assert result["goal_put_calls"] == 0
    assert result["hpc_accessed"] is False
    assert result["gpu_touched"] is False


def test_runner_source_is_loopback_http_only_and_has_no_sqlite_or_hpc_surface() -> None:
    text = (ROOT / "scripts/adopt_open_polymer_legacy_r118.py").read_text(encoding="utf-8")
    assert "127.0.0.1" in text
    assert "http.client" in text
    for forbidden in ("sqlite3", "10.120.", "ssh ", "hpc_verify", "hpc_execute_solution"):
        assert forbidden not in text
