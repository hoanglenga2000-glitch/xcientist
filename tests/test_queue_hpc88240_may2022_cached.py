from __future__ import annotations

import json
from pathlib import Path

from scripts import queue_hpc88240_may2022_cached as queue


def test_current_execution_plan_and_bundle_are_frozen_and_candidate_only() -> None:
    plan = queue.validate_execution_plan()

    assert plan["planner"]["model"] == "gpt-5.6-sol"
    assert plan["target"]["job_id"] == 88240
    assert plan["target"]["gpu"] == "NVIDIA A40"
    assert plan["authorization"]["full_training_approved"] is True
    assert plan["authorization"]["official_private_grader_approved"] is False
    assert plan["execution_contract"]["candidate_only"] is True
    assert plan["execution_contract"]["verified_precomputed_cache_required"] is True
    assert plan["seeds"] == [42, 43, 44]
    assert len(plan["run_ids"]) == 3
    assert plan["multi_seed_execution"]["all_seeds_terminal_before_successor"] is True
    assert plan["launch_argv_template"].count("--candidate-only") == 1
    assert plan["launch_argv_template"].count("--may-precomputed-cache-dir") == 1
    assert plan["launch_argv_template"].count("--may-require-precomputed-cache") == 1
    assert len(plan["public_precomputed_cache"]["manifest_sha256"]) == 64
    assert plan["_bundle_verification"]["passed"] is True


def test_dependency_classifier_accepts_only_verified_public_terminal_states() -> None:
    base = {
        "schema": "evomind.hpc_cactus_persistent_run.v2",
        "status": "verification_passed",
        "exit_code": 0,
        "process_signals_sent": 0,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }
    assert queue.classify_dependency(base) == "ready"
    assert (
        queue.classify_dependency({**base, "status": "verification_complete_gate_failed"})
        == "ready"
    )
    assert queue.classify_dependency({**base, "status": "waiting_for_ranzcr"}) == "waiting"
    assert queue.classify_dependency({**base, "status": "verifier_failed"}) == "failed"
    assert queue.classify_dependency({**base, "official_grader_executed": True}) == "failed"
    assert queue.classify_dependency(None) == "waiting"


def test_launch_claim_is_atomic_and_never_replaced(tmp_path: Path) -> None:
    claim = tmp_path / "claim.json"
    first = {"run_id": "may-s42", "plan_sha256": "a" * 64}
    second = {"run_id": "may-s42", "plan_sha256": "b" * 64}

    assert queue.acquire_claim(claim, first) is True
    assert queue.acquire_claim(claim, second) is False
    assert json.loads(claim.read_text(encoding="utf-8")) == first


def test_waiting_dependency_never_calls_gpu_or_launch(
    tmp_path: Path, monkeypatch
) -> None:
    plan = queue.validate_execution_plan()
    monkeypatch.setattr(queue, "read_remote_dependency", lambda _path: None)
    monkeypatch.setattr(
        queue.ops,
        "sample_gpu_idle_gate",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("GPU gate must not run")),
    )
    monkeypatch.setattr(
        queue.ops,
        "start_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("launch must not run")),
    )

    result = queue.run_once(plan, tmp_path)

    assert result["status"] == "waiting_for_cactus_terminal"
    assert result["process_signals_sent"] == 0
    assert result["official_grader_executed"] is False
    assert result["kaggle_submission_executed"] is False


def test_ready_launch_keeps_human_gate_and_exact_plan_name(
    tmp_path: Path, monkeypatch
) -> None:
    plan = queue.validate_execution_plan()
    dependency = {
        "schema": "evomind.hpc_cactus_persistent_run.v2",
        "status": "verification_passed",
        "exit_code": 0,
        "process_signals_sent": 0,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }
    gates = iter(
        [
            {"schema": queue.ops.EXPECTED_GATE_SCHEMA, "passed": True},
            {"schema": queue.ops.EXPECTED_GATE_SCHEMA, "passed": True},
        ]
    )
    calls = {}
    monkeypatch.setattr(queue.ops, "sample_gpu_idle_gate", lambda **_kwargs: next(gates))
    monkeypatch.setattr(queue.ops, "deploy_bundle", lambda *_args, **_kwargs: {"passed": True})
    monkeypatch.setattr(queue.ops, "cuda_smoke", lambda *_args, **_kwargs: {"passed": True})
    monkeypatch.setattr(
        queue.ops,
        "read_remote_status",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(queue.ops.RemoteOpsError("missing")),
    )

    def start(*_args, **kwargs):
        calls.update(kwargs)
        return {"status": "running", "argv": ["--candidate-only"]}

    monkeypatch.setattr(queue.ops, "start_run", start)

    result = queue.run_ready_launch(plan, evidence_dir=tmp_path, dependency=dependency)

    assert result["status"] == "seed_active"
    assert calls["run_id"] == plan["run_id"]
    assert calls["seed"] == 42
    assert calls["optimization_plan_name"] == Path(plan["_path"]).name
    assert calls["allow_concurrent_with_cpu_light"] is False
    assert calls["runner_contract_args"] == [
        "--may-precomputed-cache-dir",
        plan["public_precomputed_cache"]["path"],
        "--may-require-precomputed-cache",
    ]
    assert result["official_grader_executed"] is False
    assert result["kaggle_submission_executed"] is False


def test_all_three_terminal_is_required_before_successor(tmp_path: Path, monkeypatch) -> None:
    plan = queue.validate_execution_plan()
    dependency = {
        "schema": "evomind.hpc_cactus_persistent_run.v2",
        "status": "verification_passed",
        "exit_code": 0,
        "process_signals_sent": 0,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }
    statuses = {
        run_id: {"process": "stopped", "summary": {"competition_count": 1}}
        for run_id in plan["run_ids"]
    }
    monkeypatch.setattr(queue.ops, "read_remote_status", lambda run_id: statuses[run_id])
    monkeypatch.setattr(
        queue.ops,
        "sample_gpu_idle_gate",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("GPU gate must not run")),
    )

    result = queue.run_ready_launch(plan, evidence_dir=tmp_path, dependency=dependency)

    assert result["status"] == "all_seeds_terminal"
    assert [item["seed"] for item in result["completed_seeds"]] == [42, 43, 44]

