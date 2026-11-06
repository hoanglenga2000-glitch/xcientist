"""Run supervision v1 (dsh-superagent port): stage, retry, slots, breaker, stall rule, recovery, compaction."""
from __future__ import annotations

import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from evomind_runtime import run_supervision as sv
from evomind_runtime.runtime import AgentRuntime
from evomind_runtime.super_agent_store import SuperAgentStore


def _iso(seconds_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat()


# ---------------------------------------------------------------- stage

def test_stage_follows_dsh_priority_first_match_wins() -> None:
    card = {"kind": "model_unavailable"}
    assert sv.derive_stage("waiting_approval") == "attention"
    assert sv.derive_stage("blocked", attention=card) == "attention"
    assert sv.derive_stage("blocked") == "blocked"
    assert sv.derive_stage("failed") == "failed"
    assert sv.derive_stage("queued", queue_position=3) == "queued"
    assert sv.derive_stage("running", slot_state="queued") == "queued"
    assert sv.derive_stage("running", slot_state="gpu") == "waiting_gpu"
    assert sv.derive_stage("running", gpu_tool_running=True) == "waiting_gpu"
    assert sv.derive_stage("running") == "working"
    assert sv.derive_stage("completed", attention=card) == "done"
    # a live run is acting on its card already: no attention while it works
    assert sv.derive_stage("running", attention=card) == "working"


def test_public_attention_hides_stale_model_card_on_live_run_and_shows_tool_card_after() -> None:
    meta = {"attention": {"kind": "model_unavailable", "options": [], "secret": "x"}}
    assert sv.public_attention("running", meta) is None
    shown = sv.public_attention("blocked", meta)
    assert shown["kind"] == "model_unavailable" and "secret" not in shown
    circuit = {"tool_circuit": {"file_read": {"code": "FileNotFoundError", "count": 10, "opened_at": "x"}}}
    assert sv.public_attention("running", circuit) is None
    assert sv.public_attention("blocked", circuit)["tool"] == "file_read"


# ---------------------------------------------------------------- retry policy

def test_model_retry_retries_passing_failures_with_growing_backoff() -> None:
    text = "Runtime session is durable. Model execution is unavailable: LLMError/empty_model_response."
    delays = [sv.model_retry_decision(text, done).delay_seconds for done in range(3)]
    assert delays == [2.0, 8.0, 30.0]
    assert sv.model_retry_decision(text, 3).retry is False
    assert sv.model_retry_decision("x LLMError/model_empty_response", 0).category == "unavailable"
    assert sv.model_retry_decision("x LLMError/http_429", 0).retry is True
    assert sv.model_retry_decision("x LLMError/model_first_byte_timeout", 0).category == "timeout"


def test_retry_after_wins_but_is_capped() -> None:
    assert sv.model_retry_decision("LLMError/http_429", 0, retry_after_seconds=12).delay_seconds == 12
    assert sv.model_retry_decision("LLMError/http_429", 0, retry_after_seconds=600).delay_seconds == 60


@pytest.mark.parametrize("code", ["http_401", "authentication_error", "model_identity_unconfirmed", "model_contract_drift", "RuntimeError"])
def test_auth_request_identity_and_unknown_failures_never_retry(code: str) -> None:
    decision = sv.model_retry_decision(f"LLMError/{code}", 0)
    assert decision is not None and decision.retry is False


def test_non_model_text_is_not_a_model_failure() -> None:
    assert sv.model_retry_decision("tool step budget reached", 0) is None


# ---------------------------------------------------------------- slots

def _acquire_async(slots: sv.RunSlots, run_id: str, tenant: str = "", priority: str = "normal") -> threading.Thread:
    thread = threading.Thread(target=slots.acquire, args=(run_id, tenant), kwargs={"priority": priority, "poll_seconds": 0.02}, daemon=True)
    thread.start()
    return thread


def _wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not reached")


def test_slots_enforce_global_and_tenant_limits_with_queue_position() -> None:
    slots = sv.RunSlots(total=3, per_tenant=2, reserve=1)
    assert slots.acquire("a1", "tenant_a") and slots.acquire("a2", "tenant_a")
    waiting = _acquire_async(slots, "a3", "tenant_a")
    _wait_for(lambda: slots.state("a3") == "queued")
    assert slots.position("a3") == 1
    # another tenant is not held up by tenant_a's cap
    assert slots.acquire("b1", "tenant_b")
    assert slots.snapshot()["held"] == 3
    slots.release("a1")
    _wait_for(lambda: slots.state("a3") == "held")
    waiting.join(1)


def test_reserved_slot_only_for_a_persons_resume() -> None:
    slots = sv.RunSlots(total=1, per_tenant=5, reserve=1)
    assert slots.acquire("busy", "t1")
    normal = _acquire_async(slots, "new", "t2")
    _wait_for(lambda: slots.state("new") == "queued")
    assert slots.acquire("resume", "t3", priority="human", poll_seconds=0.02)
    assert slots.snapshot()["held_reserve"] == 1
    assert slots.state("new") == "queued"
    slots.release("busy")
    _wait_for(lambda: slots.state("new") == "held")
    normal.join(1)


def test_human_and_continuing_waiters_go_before_new_runs() -> None:
    slots = sv.RunSlots(total=1, per_tenant=5, reserve=0)
    assert slots.acquire("busy")
    _acquire_async(slots, "new")
    _wait_for(lambda: slots.state("new") == "queued")
    _acquire_async(slots, "human", priority="human")
    _wait_for(lambda: slots.position("human") == 1)
    assert slots.position("new") == 2
    slots.release("busy")
    _wait_for(lambda: slots.state("human") == "held")
    assert slots.state("new") == "queued"
    slots.release("human")
    _wait_for(lambda: slots.state("new") == "held")


def test_gpu_tool_gives_slot_back_and_takes_one_again() -> None:
    slots = sv.RunSlots(total=1, per_tenant=5, reserve=0)
    assert slots.acquire("trainer")
    with slots.released_for_gpu("trainer"):
        assert slots.state("trainer") == "gpu"
        assert slots.acquire("chat", poll_seconds=0.02)
        slots.release("chat")
    assert slots.state("trainer") == "held"


def test_abort_while_queued_returns_false() -> None:
    slots = sv.RunSlots(total=1, per_tenant=1, reserve=0)
    assert slots.acquire("busy")
    assert slots.acquire("cancelled", should_abort=lambda: True, poll_seconds=0.01) is False
    assert slots.state("cancelled") == ""


# ---------------------------------------------------------------- breaker

def _failed(args: dict, error: str, at: str = "2026-11-06T00:00:01") -> dict:
    return {"tool_name": "file_write", "status": "failed", "arguments": args, "result": {"error": error}, "created_at": at}


def test_identical_call_refused_after_three_identical_failures() -> None:
    same = {"path": "outputs/a.txt"}
    calls = [_failed(same, "preimage_mismatch") for _ in range(3)]
    verdict = sv.tool_breaker_verdict(calls, "file_write", same)
    assert verdict.refuse and verdict.reason == "identical_call"
    # different arguments still run: the model is allowed to correct itself
    assert not sv.tool_breaker_verdict(calls, "file_write", {"path": "outputs/b.txt"}).refuse


def test_tool_opens_after_ten_same_errors_and_refusals_do_not_count() -> None:
    calls = [_failed({"path": f"p{i}"}, "FileNotFoundError: [Errno 2] C:\\x") for i in range(9)]
    assert not sv.tool_breaker_verdict(calls, "file_write", {"path": "new"}).refuse
    calls.append(_failed({"path": "p9"}, "FileNotFoundError: other path"))
    verdict = sv.tool_breaker_verdict(calls, "file_write", {"path": "new"})
    assert verdict.refuse and verdict.reason == "tool_open" and verdict.code == "FileNotFoundError"
    refusals = [_failed({"path": "x"}, "repeated_identical_failure") for _ in range(20)]
    assert not sv.tool_breaker_verdict(refusals, "file_write", {"path": "x"}).refuse
    assert sv.tool_breaker_verdict([], "file_write", {}, open_tools=["file_write"]).refuse


# ---------------------------------------------------------------- stall rule / recovery note

def test_no_progress_stall_thresholds() -> None:
    now = time.time()
    assert sv.no_progress_stall({"updated_at": _iso(60)}, now) == ""
    assert sv.no_progress_stall({"updated_at": _iso(31 * 60)}, now) == "stalled"
    assert sv.no_progress_stall({"updated_at": _iso(2 * 86400)}, now) == "abandoned"


def test_recovery_note_only_for_known_reasons() -> None:
    assert "check outputs/" in sv.recovery_note("host_restarted")
    assert sv.recovery_note("resume") == ""


# ---------------------------------------------------------------- ledger

def test_ledger_numbers_attempts_and_closes_an_interrupted_one(tmp_path: Path) -> None:
    ledger = sv.RunLedger(SuperAgentStore(tmp_path / "runtime.sqlite3"))
    first = ledger.begin_attempt("run_1", reason="start")
    second = ledger.begin_attempt("run_1", reason="host_restarted")
    assert first.endswith(":a1") and second.endswith(":a2")
    assert ledger.store.get_attempt(first)["interrupted"] is True
    ledger.report(second, "ready", artifacts=1)
    latest = ledger.latest("run_1")
    assert latest["report"] == "ready" and latest["status"] == "ended" and latest["number"] == 2
    with pytest.raises(ValueError):
        ledger.report(second, "done")


# ---------------------------------------------------------------- compaction

def test_compaction_keeps_last_events_of_old_finished_runs_only(tmp_path: Path) -> None:
    db = tmp_path / "runtime.sqlite3"
    connection = sqlite3.connect(db)
    connection.executescript("""
        CREATE TABLE assistant_runs (id TEXT PRIMARY KEY, status TEXT, updated_at TEXT);
        CREATE TABLE events (id INTEGER PRIMARY KEY, session_id TEXT, seq INTEGER, event_type TEXT, payload_json TEXT, created_at TEXT);
    """)
    connection.execute("INSERT INTO assistant_runs VALUES ('old','completed',?)", (_iso(10 * 86400),))
    connection.execute("INSERT INTO assistant_runs VALUES ('new','completed',?)", (_iso(3600),))
    for session in ("old", "new"):
        for seq in range(1, 121):
            kind = "run_progress" if seq % 2 else "tool.completed"
            connection.execute("INSERT INTO events (session_id,seq,event_type,payload_json,created_at) VALUES (?,?,?,?,?)",
                               (session, seq, kind, "{}", "x"))
    connection.commit()
    connection.close()
    result = sv.compact_events(db, keep=50)
    connection = sqlite3.connect(db)
    count = lambda session, kind: connection.execute(
        "SELECT COUNT(*) FROM events WHERE session_id=? AND event_type=?", (session, kind)).fetchone()[0]
    assert result == {"sessions": 1, "deleted": 10}
    assert count("old", "run_progress") == 50 and count("old", "tool.completed") == 60
    assert count("new", "run_progress") == 60
    newest = connection.execute("SELECT MAX(seq) FROM events WHERE session_id='old' AND event_type='run_progress'").fetchone()[0]
    assert newest == 119
    connection.close()


# ---------------------------------------------------------------- integration through the run service

def _done(runtime: AgentRuntime, session_id: str) -> dict:
    runtime.store.add_turn(session_id, "assistant", "完成。")
    return {"status": "completed", "text": "完成。", "model_execution": {"provider": "fixture", "model": "served-model-v1"}}


def _empty() -> dict:
    return {"status": "blocked", "error_code": "empty_model_response",
            "text": "Runtime session is durable. Model execution is unavailable: LLMError/empty_model_response. No tool was executed by the failed model request."}


def test_empty_answers_are_retried_on_the_same_model_then_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sv, "RETRY_DELAYS_SECONDS", (0.0, 0.0, 0.0))
    runtime = AgentRuntime(tmp_path)
    prompts: list[str] = []

    def message(session_id: str, content: str, *, max_steps: int = 24) -> dict:
        prompts.append(content)
        return _empty() if len(prompts) <= 2 else _done(runtime, session_id)

    monkeypatch.setattr(runtime, "message", message)
    run = runtime.assistant.create_run(prompt="你好", conversation_id="retry_same_model")
    done = runtime.assistant.wait(run["id"], timeout=10)
    assert done["status"] == "completed" and len(prompts) == 3
    assert "Recovery note: The previous model request failed" in prompts[-1]
    retries = [item for item in runtime.assistant.events(run["id"]) if item["event_type"] == "retry_scheduled"]
    assert len(retries) == 2
    assert done["stage"] == "done" and done["attention"] is None
    assert done["supervision"] == {"attempts": 3, "last_report": "ready"}
    runtime.close()


def test_exhausted_retries_leave_a_model_unavailable_card(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sv, "RETRY_DELAYS_SECONDS", (0.0, 0.0, 0.0))
    runtime = AgentRuntime(tmp_path)
    calls = 0

    def message(session_id: str, content: str, *, max_steps: int = 24) -> dict:
        nonlocal calls
        calls += 1
        return _empty()

    monkeypatch.setattr(runtime, "message", message)
    run = runtime.assistant.create_run(prompt="你好", conversation_id="retry_exhausted")
    blocked = runtime.assistant.wait(run["id"], timeout=10)
    assert blocked["status"] == "blocked" and calls == 4
    assert blocked["stage"] == "attention"
    assert blocked["attention"]["kind"] == "model_unavailable"
    assert blocked["attention"]["options"][0]["action"] == "resume"
    # a person's resume clears the card and runs again
    monkeypatch.setattr(runtime, "message", lambda session_id, content, max_steps=24: _done(runtime, session_id))
    runtime.assistant.action(run["id"], "resume")
    done = runtime.assistant.wait(run["id"], timeout=10)
    assert done["status"] == "completed" and done["attention"] is None
    runtime.close()


def test_auth_failure_blocks_without_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = AgentRuntime(tmp_path)
    calls = 0

    def message(session_id: str, content: str, *, max_steps: int = 24) -> dict:
        nonlocal calls
        calls += 1
        return {"status": "blocked", "error_code": "http_401", "text": "Model execution is unavailable: LLMError/http_401."}

    monkeypatch.setattr(runtime, "message", message)
    run = runtime.assistant.create_run(prompt="你好", conversation_id="auth_no_retry")
    blocked = runtime.assistant.wait(run["id"], timeout=10)
    assert blocked["status"] == "blocked" and calls == 1
    assert blocked["attention"]["category"] == "auth"
    runtime.close()


def test_runs_beyond_the_slot_limit_queue_with_a_position(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = AgentRuntime(tmp_path)
    runtime.assistant.slots = sv.RunSlots(total=1, per_tenant=5, reserve=0)
    gate = threading.Event()

    def message(session_id: str, content: str, *, max_steps: int = 24) -> dict:
        gate.wait(5)
        return _done(runtime, session_id)

    monkeypatch.setattr(runtime, "message", message)
    first = runtime.assistant.create_run(prompt="一", conversation_id="slots_1")
    _wait_for(lambda: runtime.assistant.slots.state(first["id"]) == "held")
    second = runtime.assistant.create_run(prompt="二", conversation_id="slots_2")
    _wait_for(lambda: runtime.assistant.slots.state(second["id"]) == "queued")
    queued = runtime.assistant.snapshot(second["id"])
    assert queued["status"] == "queued" and queued["stage"] == "queued" and queued["queue_position"] == 1
    gate.set()
    assert runtime.assistant.wait(first["id"], timeout=10)["status"] == "completed"
    assert runtime.assistant.wait(second["id"], timeout=10)["status"] == "completed"
    runtime.close()


def test_breaker_refuses_identical_repeat_through_invoke_tool(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(prompt="读文件", conversation_id="breaker", start=False)
    results = [runtime.invoke_tool(run["id"], "file_read", {"path": "inputs/missing.txt"}) for _ in range(4)]
    errors = [item["result"]["error"].split(":", 1)[0] for item in results]
    assert errors[3] == "repeated_identical_failure"
    assert errors[0] == errors[1] == errors[2] != "repeated_identical_failure"
    runtime.close()


def test_stall_supervisor_fails_an_abandoned_active_run_without_progress(tmp_path: Path) -> None:
    from evomind_runtime.run_supervisor import StallSupervisor

    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(prompt="旧任务", conversation_id="abandoned", start=False)
    runtime.store.update_assistant_run(run["id"], status="recovering")
    with runtime.store._lock, runtime.store._connection:
        runtime.store._connection.execute("UPDATE assistant_runs SET updated_at=? WHERE id=?", (_iso(60 * 86400), run["id"]))
    outcome = StallSupervisor(runtime).tick()
    assert run["id"] in outcome["failed"]
    assert runtime.store.get_assistant_run(run["id"])["status"] == "failed"
    runtime.close()


def test_shadow_plans_are_off_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVOMIND_SUPER_AGENT_SHADOW_PLANS", raising=False)
    assert sv.shadow_plans_enabled() is False
    monkeypatch.setenv("EVOMIND_SUPER_AGENT_SHADOW_PLANS", "1")
    assert sv.shadow_plans_enabled() is True


def test_production_recovery_window_and_background_supervision_start_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """run_python_runtime.py calls recover_incomplete(max_age_seconds=...): it must accept the window and start the monitor."""
    started: list[str] = []

    class Monitor:
        def start(self) -> None:
            started.append("monitor")

    monkeypatch.setattr("evomind_runtime.ops_alerts.monitor_for", lambda runtime: Monitor())
    monkeypatch.setattr("evomind_runtime.run_supervisor.install_job_liveness", lambda root: started.append("liveness"))
    runtime = AgentRuntime(tmp_path)
    old = runtime.assistant.create_run(prompt="旧", conversation_id="window_old", start=False)
    fresh = runtime.assistant.create_run(prompt="新", conversation_id="window_new", start=False)
    for run_id, age in ((old["id"], 10 * 3600), (fresh["id"], 60)):
        runtime.store.update_assistant_run(run_id, status="running")
        with runtime.store._lock, runtime.store._connection:
            runtime.store._connection.execute("UPDATE assistant_runs SET updated_at=? WHERE id=?", (_iso(age), run_id))
    monkeypatch.setattr(runtime.assistant, "start", lambda run_id, **kw: kw.get("reason") == "host_restarted")
    assert runtime.assistant.recover_incomplete(max_age_seconds=6 * 3600) == [fresh["id"]]
    assert runtime.assistant.recover_incomplete(max_age_seconds=6 * 3600) == [fresh["id"]]
    assert started == ["liveness", "monitor"]
    runtime.close()
