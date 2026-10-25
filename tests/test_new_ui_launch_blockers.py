"""Launch blockers N01, N02, and N04 on the new-UI runtime contract.

N01: displayed routing matches ``_resolve_transports`` and each model.response.
The transport function itself is not modified.
N02: owned generated files are listed without filesystem paths.
N04: a later plain answer does not mark an undelivered report as delivered.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from evomind_runtime.model_router import ModelRouter, remember_served_model
from evomind_runtime.runtime import AgentRuntime
from evomind_runtime.tenant_access import AccessError, AccessStore, Principal, current_principal
from evomind_runtime.user_files import generated_catalog
from evomind_runtime.user_tasks import UserTasks, delivery_status
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
    assert delivery_status([]) == "draft"
