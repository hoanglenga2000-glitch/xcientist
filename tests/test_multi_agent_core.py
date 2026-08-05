from __future__ import annotations

import json
import threading
import time

import pytest

from research_os.agent.multi_agent import (
    AgentResult,
    AgentRoleSpec,
    AgentTask,
    MultiAgentStore,
    MultiAgentSupervisor,
    create_run,
    validate_task_graph,
)


def _role(name: str) -> AgentRoleSpec:
    return AgentRoleSpec(role=name, capabilities=(name,), output_contract=("evidence",))


def test_graph_rejects_cycle_and_missing_dependency():
    roles = {"worker": _role("worker")}
    with pytest.raises(ValueError, match="cycle"):
        validate_task_graph(
            {
                "a": AgentTask("a", "a", "worker", dependencies=("b",)),
                "b": AgentTask("b", "b", "worker", dependencies=("a",)),
            },
            roles,
        )
    with pytest.raises(ValueError, match="missing dependencies"):
        validate_task_graph({"a": AgentTask("a", "a", "worker", dependencies=("missing",))}, roles)


def test_scheduler_fan_out_fan_in_persists_monotonic_ledgers(tmp_path):
    roles = [_role("setup"), _role("worker"), _role("reviewer")]
    tasks = [
        AgentTask("setup", "prepare", "setup"),
        AgentTask("solution_a", "candidate a", "worker", dependencies=("setup",), solution_id="a"),
        AgentTask("solution_b", "candidate b", "worker", dependencies=("setup",), solution_id="b"),
        AgentTask("review", "review raw evidence", "reviewer", dependencies=("solution_a", "solution_b")),
    ]
    run = create_run(objective="benchmark", tasks=tasks, roles=roles, run_id="run-test", max_concurrency=3)
    store = MultiAgentStore(tmp_path / run.run_id)
    active = 0
    max_active = 0
    lock = threading.Lock()

    def execute(task, handoff, _run):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        if task.role == "worker":
            time.sleep(0.03)
        with lock:
            active -= 1
        if task.role == "reviewer":
            assert len(handoff.input_evidence_refs) == 2
        return AgentResult(task_id=task.task_id, conclusion="ok", confidence=0.9)

    supervisor = MultiAgentSupervisor(run, store, {role.role: execute for role in roles})
    result = supervisor.run_until_blocked()

    assert result.status == "completed"
    assert max_active == 2
    assert all(task.status == "completed" for task in result.tasks.values())
    assert (store.run_dir / "run.json").exists()
    assert (store.run_dir / "task_graph.json").exists()
    assert (store.run_dir / "handoffs.jsonl").exists()
    events = [json.loads(line) for line in (store.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    assert all(event["schema"].startswith("evomind.multi_agent.") for event in events)
    assert all(
        {"run_id", "task_id", "agent", "status", "evidence_refs", "artifact_hashes"}.issubset(event) for event in events
    )
    assert all(event["task_id"] for event in events)


def test_failure_retries_then_enters_needs_continuation(tmp_path):
    run = create_run(
        objective="retry",
        tasks=[AgentTask("unstable", "fail", "worker", max_retries=1)],
        roles=[_role("worker")],
        run_id="run-retry",
    )
    attempts = 0

    def fail(task, _handoff, _run):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("provider down")

    result = MultiAgentSupervisor(run, MultiAgentStore(tmp_path / run.run_id), {"worker": fail}).run_until_blocked()

    assert attempts == 2
    assert result.tasks["unstable"].status == "failed"
    assert result.status == "needs_continuation"
    assert result.next_action == "repair_or_resume"


def test_connection_failure_is_normalized_in_event_ledger(tmp_path):
    run = create_run(
        objective="connection failure",
        tasks=[AgentTask("remote", "connect", "worker", max_retries=0)],
        roles=[_role("worker")],
        run_id="run-connection-failure",
    )
    store = MultiAgentStore(tmp_path / run.run_id)

    def fail(_task, _handoff, _run):
        raise EOFError()

    result = MultiAgentSupervisor(run, store, {"worker": fail}).run_until_blocked()

    assert result.status == "needs_continuation"
    events = [json.loads(line) for line in (store.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    failed = next(event for event in events if event["schema"] == "evomind.multi_agent.task.failed.v1")
    assert failed["failure_type"] == "connection"


def test_rejected_agent_result_preserves_declared_failure_type(tmp_path):
    run = create_run(
        objective="review failure",
        tasks=[AgentTask("review", "review", "worker", max_retries=0)],
        roles=[_role("worker")],
        run_id="run-review-failure",
    )
    store = MultiAgentStore(tmp_path / run.run_id)

    def reject(task, _handoff, _run):
        return AgentResult(task.task_id, "rejected", accepted=False, failure_type="evidence")

    MultiAgentSupervisor(run, store, {"worker": reject}).run_until_blocked()
    events = [json.loads(line) for line in (store.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    failed = next(event for event in events if event["schema"] == "evomind.multi_agent.task.failed.v1")
    assert failed["failure_type"] == "evidence"


def test_successful_retry_clears_stale_error(tmp_path):
    run = create_run(
        objective="recover",
        tasks=[AgentTask("unstable", "recover", "worker", max_retries=1)],
        roles=[_role("worker")],
        run_id="run-retry-success",
    )
    attempts = 0

    def recover(task, _handoff, _run):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("transient connection failure")
        return AgentResult(task.task_id, "recovered", confidence=1.0)

    store = MultiAgentStore(tmp_path / run.run_id)
    result = MultiAgentSupervisor(run, store, {"worker": recover}).run_until_blocked()

    assert attempts == 2
    assert result.status == "completed"
    assert result.tasks["unstable"].status == "completed"
    assert result.tasks["unstable"].error == ""
    persisted = store.load()
    assert persisted.tasks["unstable"].error == ""


def test_running_task_deadline_does_not_start_overlapping_retry(tmp_path):
    run = create_run(
        objective="cooperative timeout",
        tasks=[AgentTask("slow", "finish once", "worker", max_retries=1, timeout_seconds=0)],
        roles=[_role("worker")],
        run_id="run-cooperative-timeout",
    )
    attempts = 0

    def finish_once(task, _handoff, _run):
        nonlocal attempts
        attempts += 1
        time.sleep(0.15)
        return AgentResult(task.task_id, "completed after the supervisor deadline")

    store = MultiAgentStore(tmp_path / run.run_id)
    result = MultiAgentSupervisor(run, store, {"worker": finish_once}).run_until_blocked()

    assert attempts == 1
    assert result.status == "completed"
    assert result.tasks["slow"].status == "completed"
    events = [json.loads(line) for line in (store.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    deadline_events = [
        event for event in events if event["schema"] == "evomind.multi_agent.task.deadline_exceeded.v1"
    ]
    assert len(deadline_events) == 1
    assert deadline_events[0]["action"] == "await_cooperative_completion"


def test_store_round_trip_preserves_run(tmp_path):
    run = create_run(
        objective="resume",
        tasks=[AgentTask("one", "one", "worker")],
        roles=[_role("worker")],
        run_id="run-resume",
    )
    store = MultiAgentStore(tmp_path / run.run_id)
    supervisor = MultiAgentSupervisor(
        run,
        store,
        {"worker": lambda task, _handoff, _run: AgentResult(task.task_id, "done")},
    )
    supervisor.initialize()

    loaded = store.load()
    assert loaded.run_id == run.run_id
    assert loaded.tasks["one"].goal == "one"
    assert loaded.seq == run.seq


def test_resume_reopens_tasks_skipped_after_failed_dependency(tmp_path):
    roles = [_role("runner"), _role("reviewer")]
    tasks = [
        AgentTask("train", "train", "runner", max_retries=0),
        AgentTask("review", "review", "reviewer", dependencies=("train",), max_retries=0),
    ]
    run = create_run(objective="resume chain", tasks=tasks, roles=roles, run_id="run-resume-chain")
    store = MultiAgentStore(tmp_path / run.run_id)
    attempts = 0

    def runner(task, _handoff, _run):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("first attempt fails")
        return AgentResult(task.task_id, "trained", ["metrics.json"])

    def reviewer(task, _handoff, _run):
        return AgentResult(task.task_id, "reviewed", ["review.json"])

    supervisor = MultiAgentSupervisor(run, store, {"runner": runner, "reviewer": reviewer})
    first = supervisor.run_until_blocked()
    assert first.tasks["train"].status == "failed"
    assert first.tasks["review"].status == "skipped"

    supervisor.resume(retry_failed=True)
    assert run.tasks["train"].status == "ready"
    assert run.tasks["review"].status == "pending"
    resumed = supervisor.run_until_blocked()

    assert resumed.status == "completed"
    assert resumed.tasks["train"].status == "completed"
    assert resumed.tasks["review"].status == "completed"


def test_resume_recovers_orphaned_running_task_after_process_exit(tmp_path):
    run = create_run(
        objective="recover interrupted process",
        tasks=[AgentTask("train", "train", "runner", max_retries=0)],
        roles=[_role("runner")],
        run_id="run-orphaned-task",
    )
    store = MultiAgentStore(tmp_path / run.run_id)
    supervisor = MultiAgentSupervisor(
        run,
        store,
        {"runner": lambda task, _handoff, _run: AgentResult(task.task_id, "recovered")},
    )
    supervisor.initialize()
    run.status = "running"
    run.tasks["train"].status = "running"
    run.tasks["train"].attempts = 1
    store.save(run)

    recovered_run = store.load()
    recovered = MultiAgentSupervisor(
        recovered_run,
        store,
        {"runner": lambda task, _handoff, _run: AgentResult(task.task_id, "recovered")},
    )
    recovered.resume()
    result = recovered.run_until_blocked()

    assert result.status == "completed"
    assert result.tasks["train"].status == "completed"
    events = [json.loads(line) for line in (store.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["schema"] == "evomind.multi_agent.task.interrupted.v1" for event in events)


def test_dynamic_followup_expands_graph_and_redirects_claim_audit(tmp_path):
    roles = [_role("reviewer"), _role("worker"), _role("auditor")]
    run = create_run(
        objective="review feedback",
        tasks=[
            AgentTask("review", "review", "reviewer"),
            AgentTask("claim_audit", "audit", "auditor", dependencies=("review",)),
        ],
        roles=roles,
        run_id="run-followup",
    )
    store = MultiAgentStore(tmp_path / run.run_id)
    supervisor = MultiAgentSupervisor(run, store, {})
    followup = AgentTask(
        "correction_review",
        "review corrected evidence",
        "reviewer",
        dependencies=("review",),
        payload={"config_hash": "correction-config"},
    )

    supervisor.apply_followups(
        run.tasks["review"],
        AgentResult(
            "review",
            "correction required",
            followup_tasks=[followup],
            dependency_overrides={"claim_audit": ("correction_review",)},
        ),
    )

    assert run.tasks["correction_review"].dependencies == ("review",)
    assert run.tasks["correction_review"].idempotency_key
    assert run.tasks["claim_audit"].dependencies == ("correction_review",)
    persisted = store.load()
    assert persisted.tasks["claim_audit"].dependencies == ("correction_review",)
    event = json.loads((store.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert event["schema"] == "evomind.multi_agent.task_graph.expanded.v1"


@pytest.mark.parametrize("failure", ["unknown_dependency", "cycle", "conflict"])
def test_invalid_followup_is_atomic_and_does_not_pollute_graph(tmp_path, failure):
    roles = [_role("reviewer"), _role("auditor")]
    run = create_run(
        objective="atomic followup",
        tasks=[
            AgentTask("review", "review", "reviewer"),
            AgentTask("claim_audit", "audit", "auditor", dependencies=("review",)),
        ],
        roles=roles,
        run_id=f"run-followup-{failure}",
    )
    supervisor = MultiAgentSupervisor(run, MultiAgentStore(tmp_path / run.run_id), {})
    before = json.dumps(run.to_dict(), ensure_ascii=False, sort_keys=True)
    if failure == "unknown_dependency":
        result = AgentResult(
            "review",
            "bad dependency",
            followup_tasks=[AgentTask("fix", "fix", "reviewer", dependencies=("missing",))],
        )
        expected = "missing dependencies"
    elif failure == "cycle":
        result = AgentResult(
            "review",
            "cyclic dependency",
            followup_tasks=[AgentTask("fix", "fix", "reviewer", dependencies=("claim_audit",))],
            dependency_overrides={"claim_audit": ("fix",)},
        )
        expected = "cycle"
    else:
        result = AgentResult(
            "review",
            "conflict",
            followup_tasks=[AgentTask("claim_audit", "different", "auditor")],
        )
        expected = "conflicting follow-up"

    with pytest.raises(ValueError, match=expected):
        supervisor.apply_followups(run.tasks["review"], result)

    assert json.dumps(run.to_dict(), ensure_ascii=False, sort_keys=True) == before
    assert not (supervisor.store.run_dir / "events.jsonl").exists()


def test_store_on_save_receives_every_persisted_state(tmp_path):
    snapshots = []
    run = create_run(
        objective="pointer sync",
        tasks=[AgentTask("one", "one", "worker")],
        roles=[_role("worker")],
        run_id="run-pointer-sync",
    )
    store = MultiAgentStore(
        tmp_path / run.run_id,
        on_save=lambda saved: snapshots.append((saved.run_id, saved.seq, saved.status)),
    )

    store.save(run)
    store.emit(run, "run.created", status="ready")

    assert snapshots == [
        ("run-pointer-sync", 0, "created"),
        ("run-pointer-sync", 1, "created"),
    ]
    assert store.load().seq == 1
