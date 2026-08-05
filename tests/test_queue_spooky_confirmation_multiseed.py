"""Contracts for the frozen Spooky 40/41 confirmation queue."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

from scripts import queue_spooky_confirmation_multiseed as queue


def test_manifest_and_confirmation_plans_are_frozen_and_hash_bound():
    manifest = queue.validate_manifest(queue.DEFAULT_MANIFEST)
    assert manifest["requested_model"] == "gpt-5.6-sol"
    assert manifest["served_model"] == "gpt-5.6-sol"
    assert sorted(manifest["_plans"]) == [40, 41]
    assert manifest["required_seeds"] == [40, 41, 42]
    for seed, plan in manifest["_plans"].items():
        assert plan["training"]["seed"] == seed
        assert plan["inputs"]["duplicate_group_report"]["seed"] == seed
        assert plan["inputs"]["duplicate_group_report"]["group_isolation"] is True
        assert plan["confirmation"]["private_grader_feedback_used"] is False


def test_confirmation_commands_use_unique_run_ids_and_exact_seeds():
    manifest = queue.validate_manifest(queue.DEFAULT_MANIFEST)
    run_ids = set()
    for seed, plan in manifest["_plans"].items():
        command = queue.spooky_queue.build_training_command(plan)
        assert command[command.index("--seed") + 1] == str(seed)
        assert command[command.index("--run-id") + 1] == plan["execution"]["run_id"]
        assert command[command.index("--frozen-plan") + 1] == plan["_path"]
        run_ids.add(plan["execution"]["run_id"])
    assert len(run_ids) == 2


def test_verifier_command_targets_each_confirmation_plan():
    manifest = queue.validate_manifest(queue.DEFAULT_MANIFEST)
    for seed, plan in manifest["_plans"].items():
        command = queue.build_verifier_command(plan, queue.DEFAULT_VERIFIER)
        assert command[command.index("--plan") + 1] == plan["_path"]
        assert command[command.index("--run-dir") + 1].endswith(
            plan["execution"]["run_id"]
        )
        joined = " ".join(command).lower()
        assert "private_grade" not in joined
        assert "kaggle" not in joined
        assert str(seed) in plan["execution"]["run_id"]


def test_missing_seed_run_is_not_ready(tmp_path: Path):
    manifest = queue.validate_manifest(queue.DEFAULT_MANIFEST)
    snapshot = queue.verified_seed_snapshot(40, tmp_path / "missing", manifest["_plans"][40])
    assert snapshot["ready"] is False
    assert snapshot["status"] == "run_missing"
    assert snapshot["run_exists"] is False


def test_failed_seed42_is_terminal_and_cannot_launch_confirmations(tmp_path: Path):
    seed42 = queue.spooky_queue.validate_frozen_plan(queue.DEFAULT_SEED42_PLAN)
    run_dir = tmp_path / "seed42"
    run_dir.mkdir()
    summary = {
        "run_id": seed42["execution"]["run_id"],
        "status": "single_seed_gate_failed",
        "plan_sha256": seed42["_sha256"],
        "candidate_oof_log_loss": 0.3449842458960858,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }
    verification = {
        "status": "failed",
        "plan": {"sha256": seed42["_sha256"]},
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "independent_verification.json").write_text(
        json.dumps(verification), encoding="utf-8"
    )
    np.savez_compressed(run_dir / "spooky_transformer_oof_and_test.npz", placeholder=[1])
    (run_dir / "candidate_submission_withheld.csv").write_text(
        "id,EAP,HPL,MWS\n", encoding="utf-8"
    )

    snapshot = queue.verified_seed_snapshot(42, run_dir, seed42)

    assert snapshot["ready"] is False
    assert snapshot["terminal"] is True
    assert snapshot["status"] == "terminal_single_seed_gate_failed"
    assert snapshot["score"] == 0.3449842458960858
    assert all(snapshot["checks"].values())


def test_current_primary_snapshot_is_explicit_and_read_only():
    siim = queue.siim_queue.validate_frozen_plan(queue.DEFAULT_SIIM_PLAN)
    leaf = queue.leaf_queue.validate_frozen_plan(queue.DEFAULT_LEAF_PLAN)
    may = queue.may_queue.validate_frozen_plan(queue.DEFAULT_MAY_PLAN)
    snapshot = queue.primary_chain_snapshot(
        siim,
        leaf,
        may,
        seed42_ready=False,
    )
    assert isinstance(snapshot["active"], bool)
    assert "siim_staging" in snapshot
    assert "may2022" in snapshot


def test_launch_claim_is_exclusive(tmp_path: Path):
    claim = tmp_path / "claim.json"
    assert queue.create_launch_claim(claim, {"seed": 40}) is True
    assert queue.create_launch_claim(claim, {"seed": 40}) is False


def test_source_never_controls_other_processes_or_calls_official_actions():
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


def test_calibrated_gate_binds_seed42_and_confirmation_manifest():
    manifest = queue.validate_manifest(queue.DEFAULT_MANIFEST)
    seed42 = queue.spooky_queue.validate_frozen_plan(queue.DEFAULT_SEED42_PLAN)
    policy = queue.idle_gate.validate_policy(queue.DEFAULT_GATE_POLICY)
    assert policy["bound_execution_evidence"]["seed42_plan"]["sha256"] == seed42["_sha256"]
    assert (
        policy["bound_execution_evidence"]["confirmation_manifest"]["sha256"]
        == manifest["_sha256"]
    )
