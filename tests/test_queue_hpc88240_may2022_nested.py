from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import queue_hpc88240_may2022_nested as queue


def test_v2_queue_fails_closed_after_cache_backed_v3_supersession() -> None:
    assert queue.SUPERSEDED_BY_PLAN.is_file()
    with pytest.raises(queue.MayQueueError, match="superseded.*cache-backed v3"):
        queue.validate_execution_plan()


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
    with pytest.raises(queue.MayQueueError, match="superseded"):
        queue.validate_execution_plan()


def test_ready_launch_keeps_human_gate_and_exact_plan_name(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        queue.ops,
        "start_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("launch must not run")),
    )
    with pytest.raises(queue.MayQueueError, match="superseded"):
        queue.validate_execution_plan()
