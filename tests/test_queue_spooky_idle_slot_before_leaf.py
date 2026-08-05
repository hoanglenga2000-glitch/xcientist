"""Contracts for the no-preemption Spooky idle-slot queue."""

from __future__ import annotations

import inspect
from pathlib import Path

from scripts import queue_spooky_idle_slot_before_leaf as queue


def fixture_snapshot(**overrides):
    payload = {
        "spooky_data": {"ready": True},
        "target_complete": False,
        "target_run_exists": False,
        "siim_priority_active": False,
        "leaf_authoritative_active": False,
    }
    payload.update(overrides)
    return payload


def test_classifier_only_allows_spooky_when_both_priority_queues_are_inactive():
    assert queue.classify_opportunity(fixture_snapshot()) == "eligible_for_idle_gate"
    assert (
        queue.classify_opportunity(fixture_snapshot(siim_priority_active=True))
        == "yielded_to_siim_priority"
    )
    assert (
        queue.classify_opportunity(fixture_snapshot(leaf_authoritative_active=True))
        == "yielded_to_leaf_authoritative_queue"
    )


def test_classifier_fails_closed_for_existing_target_or_missing_data():
    assert (
        queue.classify_opportunity(fixture_snapshot(target_run_exists=True))
        == "target_run_already_exists"
    )
    assert (
        queue.classify_opportunity(
            fixture_snapshot(target_run_exists=True, target_complete=True)
        )
        == "target_run_already_complete"
    )
    assert (
        queue.classify_opportunity(fixture_snapshot(spooky_data={"ready": False}))
        == "waiting_for_spooky_data"
    )


def test_live_frozen_plans_produce_an_eligible_or_priority_snapshot():
    spooky = queue.spooky_queue.validate_frozen_plan(queue.DEFAULT_SPOOKY_PLAN)
    siim = queue.siim_queue.validate_frozen_plan(queue.DEFAULT_SIIM_PLAN)
    snapshot = queue.opportunity_snapshot(
        spooky,
        siim,
        data_report_path=queue.DEFAULT_DATA_REPORT,
    )
    assert snapshot["decision"] in {
        "eligible_for_idle_gate",
        "yielded_to_siim_priority",
        "yielded_to_leaf_authoritative_queue",
        "target_run_already_complete",
        "target_run_already_exists",
    }
    assert snapshot["spooky_data"]["ready"] is True


def test_launch_claim_is_exclusive(tmp_path: Path):
    claim = tmp_path / "claim.json"
    assert queue.create_launch_claim(claim, {"run_id": "fixture"}) is True
    assert queue.create_launch_claim(claim, {"run_id": "fixture"}) is False
    assert '"run_id": "fixture"' in claim.read_text(encoding="utf-8")


def test_source_has_no_process_control_or_official_action():
    source = inspect.getsource(queue)
    assert "Stop-Process" not in source
    assert "taskkill" not in source
    assert ".terminate(" not in source
    assert ".kill(" not in source
    assert "os.kill(" not in source
    assert "private_grade(" not in source
    assert '"process_signals_sent": 0' in source
    assert '"official_grader_executed": False' in source
    assert '"kaggle_submission_executed": False' in source


def test_command_reuses_the_frozen_spooky_runner_and_seed():
    plan = queue.spooky_queue.validate_frozen_plan(queue.DEFAULT_SPOOKY_PLAN)
    command = queue.spooky_queue.build_training_command(plan)
    assert "run_spooky_transformer_oof.py" in " ".join(command)
    assert command[command.index("--seed") + 1] == "42"


def test_calibrated_gate_is_hash_bound_to_seed42_plan():
    plan = queue.spooky_queue.validate_frozen_plan(queue.DEFAULT_SPOOKY_PLAN)
    policy = queue.idle_gate.validate_policy(queue.DEFAULT_GATE_POLICY)
    assert (
        policy["bound_execution_evidence"]["seed42_plan"]["sha256"]
        == plan["_sha256"]
    )
    assert policy["requirements"]["minimum_check_interval_seconds"] == 10
