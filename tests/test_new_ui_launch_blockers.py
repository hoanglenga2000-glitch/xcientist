"""Launch blockers N01, N02, and N04 on the new-UI runtime contract.

N01: displayed routing matches ``_resolve_transports`` and each model.response.
The transport function itself is not modified.
N02: owned generated files are listed without filesystem paths.
N04: a later plain answer does not mark an undelivered report as delivered.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from datetime import datetime, timedelta, timezone

from evomind_runtime.model_router import ModelRouter, remember_served_model
from evomind_runtime.models import ApprovalRequest, ToolCall, new_id, utc_now
from evomind_runtime.runtime import AgentRuntime
from evomind_runtime.tenant_access import AccessError, AccessStore, Principal, current_principal
from evomind_runtime.user_files import generated_catalog
from evomind_runtime.user_tasks import UserTasks, delivery_status
from research_os.agent import messaging
from research_os.agent.messaging import _resolve_transports


def _clear_model_env(monkeypatch):
    for name in (
        "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "CLAUDE_CODE_MODEL",
        "OPENAI_API_KEY", "OPENAI_MODEL", "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL",
        "EVOLUTION_PRIMARY_PROVIDER", "EVOLUTION_PROVIDER_STRICT",
    ):
        monkeypatch.delenv(name, raising=False)


def test_route_reports_the_strict_transport_not_the_display_alias(monkeypatch):
    _clear_model_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("CLAUDE_CODE_MODEL", "claude-opus-4-8")
    monkeypatch.setenv("OPENAI_API_KEY", "other-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    monkeypatch.setenv("EVOLUTION_PRIMARY_PROVIDER", "anthropic")
    monkeypatch.setenv("EVOLUTION_PROVIDER_STRICT", "1")
    before = [(item.name, item.config.model) for item in _resolve_transports()]
    decision = ModelRouter().route("planning", "economy")
    after = [(item.name, item.config.model) for item in _resolve_transports()]
    assert before == after == [("anthropic", "claude-opus-4-8")]
    assert (decision.provider, decision.model) == before[0]
    assert decision.model != "claude-sonnet-5"
    assert decision.fallback_chain == []


def test_route_follows_strict_primary_when_it_is_not_anthropic(monkeypatch):
    _clear_model_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("CLAUDE_CODE_MODEL", "claude-opus-4-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setenv("EVOLUTION_PRIMARY_PROVIDER", "deepseek")
    monkeypatch.setenv("EVOLUTION_PROVIDER_STRICT", "true")
    transports = _resolve_transports()
    decision = ModelRouter().route("execution", "balanced")
    assert [(item.name, item.config.model) for item in transports] == [("deepseek", "deepseek-chat")]
    assert (decision.provider, decision.model) == ("deepseek", "deepseek-chat")


def test_platform_pick_on_a_new_run_is_the_observed_model(tmp_path, monkeypatch):
    """Platform selection is the strict transport. A stubbed reply echoes that model.

    Personal-profile selection still needs a Windows credential store and is not
    proven here. ``_resolve_transports`` is called, not replaced.
    """
    _clear_model_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9/anthropic")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("CLAUDE_CODE_MODEL", "claude-opus-5-5")
    monkeypatch.setenv("EVOLUTION_PRIMARY_PROVIDER", "anthropic")
    monkeypatch.setenv("EVOLUTION_PROVIDER_STRICT", "1")
    selected = _resolve_transports()
    assert [(item.name, item.config.model) for item in selected] == [("anthropic", "claude-opus-5-5")]
    requested = {}

    def post(url, headers, payload, timeout):
        requested["model"] = payload["model"]
        return {
            "model": payload["model"],
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "你好"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    monkeypatch.setattr(messaging, "_post_json", post)
    runtime = AgentRuntime(tmp_path)
    try:
        run = runtime.assistant.create_run(prompt="你好", start=False)
        assert run["model"] == ""
        assert not (run.get("model_profile"))
        runtime.assistant._execute(run["id"], resume=False)
        snapshot = runtime.assistant.snapshot(run["id"])
        assert requested["model"] == "claude-opus-5-5"
        assert snapshot["model_provider"] == "anthropic"
        assert snapshot["model"] == "claude-opus-5-5"
        observed = [item for item in runtime.assistant.events(run["id"]) if item["event_type"] == "model_observed"]
        assert observed[-1]["payload"]["provider"] == "anthropic"
        assert observed[-1]["payload"]["model"] == "claude-opus-5-5"
        assert (snapshot["model_provider"], snapshot["model"]) == (selected[0].name, selected[0].config.model)
    finally:
        runtime.close()


def test_model_response_is_what_snapshots_and_public_events_show(tmp_path, monkeypatch):
    _clear_model_env(monkeypatch)
    runtime = AgentRuntime(tmp_path)
    try:
        run = runtime.assistant.create_run(prompt="Say hello.", start=False)
        assert run["model"] == ""
        remember_served_model(runtime.store, run["id"], "anthropic", "claude-opus-4-8")
        remember_served_model(runtime.store, "session_without_run", "anthropic", "claude-sonnet-5")
        remember_served_model(runtime.store, run["id"], "", "")
        stored = runtime.store.get_assistant_run(run["id"])
        assert stored["model_provider"] == "anthropic"
        assert stored["model"] == "claude-opus-4-8"
        runtime.store.append_event(run["id"], "model.response", {"provider": "anthropic", "model": "claude-opus-4-8"})
        observed = [item for item in runtime.assistant.events(run["id"]) if item["event_type"] == "model_observed"]
        assert observed[-1]["payload"]["provider"] == "anthropic"
        assert observed[-1]["payload"]["model"] == "claude-opus-4-8"
        assert runtime.assistant.snapshot(run["id"])["model"] == "claude-opus-4-8"
    finally:
        runtime.close()


def _principal(name: str, tenant: str) -> Principal:
    return Principal("tenant_" + tenant * 24, name)


def _owned_run(runtime, tasks, access, principal, task, prompt: str):
    token = current_principal.set(principal)
    try:
        run = runtime.assistant.create_run(
            prompt=prompt, conversation_id=task["conversation_id"], user_task_id=task["id"], start=False,
        )
    finally:
        current_principal.reset(token)
    tasks.link_run(principal, task["id"], run["id"], runtime, access)
    return run


def test_failed_report_stays_undelivered_after_a_later_answer(tmp_path, monkeypatch):
    _clear_model_env(monkeypatch)
    runtime = AgentRuntime(tmp_path)
    access = AccessStore(runtime.runtime_root)
    tasks = UserTasks(runtime.runtime_root)
    alice = _principal("alice", "a")
    bob = _principal("bob", "b")
    try:
        task = tasks.create(alice, {"title": "风险报告", "draft": "", "idempotency_key": "delivery-task"})["task"]
        report = _owned_run(runtime, tasks, access, alice, task, "请生成一份报告")
        answer = _owned_run(runtime, tasks, access, alice, task, "你好")
        assert report["plan"]["require_artifact"] is True
        assert answer["plan"]["require_artifact"] is False
        runtime.store.update_assistant_run(report["id"], status="failed")
        runtime.store.update_assistant_run(answer["id"], status="completed")
        detail = tasks.detail(alice, task["id"], runtime, access)
        assert detail["task"]["status"] == "undelivered"
        assert detail["generated_files"] == []
        assert tasks.list(alice, runtime, access)[0]["status"] == "undelivered"
        with pytest.raises(AccessError):
            tasks.detail(bob, task["id"], runtime, access)
        assert tasks.list(bob, runtime, access) == []

        delivered = _owned_run(runtime, tasks, access, alice, task, "请生成一份报告")
        path = Path(delivered["task_root"]) / "outputs" / "report.md"
        path.write_text("报告正文\n", encoding="utf-8")
        runtime.assistant.publish_path(delivered["id"], path, source_tool_call="artifact_publish")
        runtime.store.update_assistant_run(delivered["id"], status="completed")
        done = tasks.detail(alice, task["id"], runtime, access)
        assert done["task"]["status"] == "delivered"
        assert done["generated_files"][0]["name"] == "report.md"
        assert "path" not in done["generated_files"][0]
        listed = generated_catalog(runtime, access, alice)
        assert [item["name"] for item in listed] == ["report.md"]
        assert "path" not in listed[0]
        assert generated_catalog(runtime, access, bob) == []
    finally:
        runtime.close()


def test_plain_answer_with_no_goal_is_delivered_and_active_work_stays_in_progress():
    assert delivery_status([
        {"status": "completed", "require_artifact": False, "artifact_count": 0},
    ]) == "delivered"
    assert delivery_status([
        {"status": "running", "require_artifact": True, "artifact_count": 0},
        {"status": "failed", "require_artifact": True, "artifact_count": 0},
    ]) == "running"
    assert delivery_status([
        {"status": "failed", "require_artifact": False, "artifact_count": 0},
    ]) == "undelivered"
    assert delivery_status([
        {"status": "cancelled", "require_artifact": False, "artifact_count": 0},
    ]) == "undelivered"
    assert delivery_status([
        {"status": "planning", "require_artifact": False, "artifact_count": 0},
        {"status": "failed", "require_artifact": False, "artifact_count": 0},
    ]) == "planning"
    assert delivery_status([]) == "draft"


def _backdate_planning(runtime, run_id, stamp="2026-10-14T00:00:00+00:00"):
    runtime.store.update_assistant_run(run_id, status="planning")
    runtime.store.update_session(run_id, status="planning")
    with runtime.store._lock, runtime.store._connection:
        runtime.store._connection.execute(
            "UPDATE assistant_runs SET created_at=?, updated_at=? WHERE id=?",
            (stamp, stamp, run_id),
        )
        runtime.store._connection.execute(
            "UPDATE sessions SET created_at=?, updated_at=? WHERE id=?",
            (stamp, stamp, run_id),
        )


def _planning_events(runtime, run_id):
    return [
        item for item in runtime.store.list_events(run_id)
        if item["event_type"] in {"run_blocked", "run_completed", "planning_closed"}
        and (item.get("payload") or {}).get("previous_status") == "planning"
    ]


def test_idle_planning_since_mid_october_closes_without_cancel(tmp_path):
    runtime = AgentRuntime(tmp_path)
    try:
        old = runtime.assistant.create_run(prompt="整理一份方案", start=False)
        _backdate_planning(runtime, old["id"])
        fresh = runtime.assistant.create_run(prompt="刚开始整理", start=False)
        runtime.store.update_assistant_run(fresh["id"], status="planning")
        runtime.store.update_session(fresh["id"], status="planning")

        snapshot = runtime.assistant.snapshot(old["id"])
        stored = runtime.store.get_assistant_run(old["id"])
        assert snapshot["status"] == stored["status"] == "blocked"
        assert stored["error_class"] == "planning_idle"
        assert stored["error_message"] == ""
        assert stored["status"] != "cancelled"
        assert runtime.store.get_session(old["id"])["status"] == "blocked"
        event = _planning_events(runtime, old["id"])[-1]
        assert event["event_type"] == "run_blocked"
        assert event["payload"]["reason"] == "planning_idle"
        assert event["payload"]["previous_status"] == "planning"
        assert runtime.store.get_assistant_run(fresh["id"])["status"] == "planning"
        assert runtime.store.get_session(fresh["id"])["status"] == "planning"
    finally:
        runtime.close()


def test_recover_does_not_restart_idle_planning(tmp_path):
    runtime = AgentRuntime(tmp_path)
    try:
        old = runtime.assistant.create_run(prompt="停在方案里的记录", start=False)
        _backdate_planning(runtime, old["id"])
        newer = (datetime.now(timezone.utc) - timedelta(days=6)).isoformat()
        with runtime.store._lock, runtime.store._connection:
            for index in range(200):
                filler_id = f"run_filler_{index:03d}"
                runtime.store._connection.execute(
                    "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (filler_id, "", "workspace_write", "planning", "filler", "filler", newer, newer, "", "", "{}"),
                )
                runtime.store._connection.execute(
                    "INSERT INTO assistant_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        filler_id, filler_id, "conversation_default", "filler",
                        "", "planning", "{}", "[]", 0, "", "", "", "", newer, newer, "",
                    ),
                )
        recent = runtime.store.list_assistant_runs(limit=200)
        assert old["id"] not in {item["id"] for item in recent}
        assert runtime.store.list_assistant_runs_with_status("planning")[0]["id"] == old["id"]

        recovered = runtime.assistant.recover_incomplete()
        stored = runtime.store.get_assistant_run(old["id"])
        assert old["id"] not in recovered
        assert stored["status"] == "blocked"
        assert stored["error_class"] == "planning_idle"
        assert stored["status"] != "cancelled"
        assert runtime.assistant._threads.get(old["id"]) is None
        assert runtime.store.list_assistant_runs_with_status("planning") == []
    finally:
        runtime.close()


def test_finished_planning_row_completes_instead_of_cancel(tmp_path):
    runtime = AgentRuntime(tmp_path)
    try:
        run = runtime.assistant.create_run(prompt="方案已经做完", start=False)
        plan = dict(run["plan"])
        plan["steps"] = [{**step, "status": "completed"} for step in plan["steps"]]
        runtime.store.update_assistant_run(run["id"], plan=plan)
        _backdate_planning(runtime, run["id"])
        receipt = runtime.assistant.close_idle_planning()
        stored = runtime.store.get_assistant_run(run["id"])
        assert stored["status"] == "completed"
        assert stored["status"] != "cancelled"
        assert runtime.store.get_session(run["id"])["status"] == "completed"
        assert receipt["closed"][0]["reason"] == "planning_already_finished"
        assert _planning_events(runtime, run["id"])[-1]["event_type"] == "run_completed"
    finally:
        runtime.close()


def test_idle_planning_with_a_fresh_confirmation_is_not_cancelled(tmp_path):
    runtime = AgentRuntime(tmp_path)
    try:
        run = runtime.assistant.create_run(prompt="等确认再继续", start=False)
        approval = ApprovalRequest(
            new_id("approval"),
            run["id"],
            new_id("call"),
            "file_write",
            "fixture-fingerprint",
            {"path": "note.txt"},
            {},
            "high",
            False,
            expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        )
        runtime.store.put_approval(approval)
        _backdate_planning(runtime, run["id"])
        runtime.assistant.close_idle_planning()
        stored = runtime.store.get_assistant_run(run["id"])
        assert stored["status"] == "waiting_approval"
        assert stored["status"] != "cancelled"
        assert runtime.store.get_approval(approval.id)["status"] == "pending"
        assert runtime.store.get_session(run["id"])["status"] == "waiting_approval"
        event = _planning_events(runtime, run["id"])[-1]
        assert event["event_type"] == "planning_closed"
        assert event["payload"]["reason"] == "planning_needs_confirmation"
    finally:
        runtime.close()


def test_idle_planning_with_an_unconfirmed_tool_is_blocked_not_cancelled(tmp_path):
    runtime = AgentRuntime(tmp_path)
    try:
        run = runtime.assistant.create_run(prompt="有一步还没核对", start=False)
        runtime.store.put_tool_call(ToolCall(
            new_id("call"), run["id"], "file_write", {"path": "note.txt"}, status="running", started_at=utc_now(),
        ))
        _backdate_planning(runtime, run["id"])
        runtime.assistant.close_idle_planning()
        stored = runtime.store.get_assistant_run(run["id"])
        assert stored["status"] == "blocked"
        assert stored["error_class"] == "execution_reconciliation_required"
        assert stored["status"] != "cancelled"
        event = _planning_events(runtime, run["id"])[-1]
        assert event["payload"]["reason"] == "planning_tool_unconfirmed"
    finally:
        runtime.close()
