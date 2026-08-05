from __future__ import annotations

import pytest

from research_os.demo_campaign import (
    DEMO_ITERATIONS,
    DEMO_MAX_COST_USD,
    DEMO_MAX_NODES,
    DEMO_MAX_TOKENS,
    DEMO_MAX_WALL_SECONDS,
    DEMO_RUNNER,
    DEMO_SEARCH_MODE,
    DEMO_TASK_ID,
)
from scripts.evolution_run_cli import _parse_run_contract


def test_search_contract_keeps_legacy_defaults_and_caps_iterations_by_nodes() -> None:
    contract = _parse_run_contract({"iterations": 12, "max_nodes": 4})

    assert contract["search_mode"] == "legacy_uct"
    assert contract["iterations"] == 4
    assert contract["requested_iterations"] == 12
    assert contract["max_nodes"] == 4
    assert contract["max_tokens"] == 2_000_000
    assert contract["max_wall_seconds"] == 43_200.0
    assert contract["max_cost"] is None


def test_experience_mode_accepts_all_hard_budgets() -> None:
    contract = _parse_run_contract(
        {
            "runner": "local",
            "iterations": 24,
            "mcgs": True,
            "search_mode": "experience_mcgs_v1",
            "max_nodes": 24,
            "max_tokens": 700_000,
            "max_wall_seconds": 14_400,
            "max_cost": 125.5,
        }
    )

    assert contract == {
        "runner": "local",
        "search_mode": "experience_mcgs_v1",
        "iterations": 24,
        "requested_iterations": 24,
        "max_nodes": 24,
        "max_tokens": 700_000,
        "max_wall_seconds": 14_400.0,
        "max_cost": 125.5,
        "mcgs": True,
    }


def test_environment_feature_flag_is_opt_in_and_explicit_mode_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXPERIENCE_MCGS_V1", "1")
    assert _parse_run_contract({})["search_mode"] == "experience_mcgs_v1"
    assert _parse_run_contract({"search_mode": "legacy_uct"})["search_mode"] == "legacy_uct"


def test_demo_task_forces_the_single_eight_node_local_experience_contract() -> None:
    contract = _parse_run_contract(
        {
            "task_id": DEMO_TASK_ID,
            "runner": "gpu",
            "iterations": 3,
            "mcgs": False,
            "search_mode": "legacy_uct",
            "max_nodes": 3,
            "max_tokens": 900_000,
            "max_wall_seconds": 9_000,
            "max_cost": 10.0,
        }
    )

    assert contract == {
        "runner": DEMO_RUNNER,
        "search_mode": DEMO_SEARCH_MODE,
        "iterations": DEMO_ITERATIONS,
        "requested_iterations": DEMO_ITERATIONS,
        "max_nodes": DEMO_MAX_NODES,
        "max_tokens": DEMO_MAX_TOKENS,
        "max_wall_seconds": DEMO_MAX_WALL_SECONDS,
        "max_cost": DEMO_MAX_COST_USD,
        "mcgs": True,
    }


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"runner": "shell"}, "runner must be one of"),
        ({"search_mode": "private_feedback"}, "search_mode must be one of"),
        ({"search_mode": "experience_mcgs_v1", "mcgs": False}, "requires mcgs=true"),
        ({"max_nodes": 0}, "max_nodes must be between"),
        ({"max_tokens": True}, "max_tokens must be an integer"),
        ({"max_wall_seconds": 43_201}, "max_wall_seconds must be between"),
        ({"max_cost": -1}, "max_cost must be between"),
    ],
)
def test_search_contract_rejects_unsupported_or_unbounded_input(payload: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _parse_run_contract(payload)
