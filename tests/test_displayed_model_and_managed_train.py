"""Displayed Claude stays Claude, and a personal baseline uses the managed trainer.

No provider call leaves the machine: model URLs are 127.0.0.1 and the trainer
stops at the local binding file.
"""
import json

from evomind_runtime.model_router import empty_failover_transport
from evomind_runtime.personal_tool_boundary import managed_training_clause, offered
from evomind_runtime.runtime import AgentRuntime
from evomind_runtime.tenant_access import Principal, current_principal
from evomind_runtime.tools import ToolContext, _tenant_tensor_train
from research_os.agent import messaging


TENANT = "tenant_" + "cd" * 12


def _clear(monkeypatch):
    for name in (
        "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL", "CLAUDE_CODE_MODEL",
        "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL",
        "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL",
        "EVOLUTION_PRIMARY_PROVIDER", "EVOLUTION_PROVIDER_STRICT",
    ):
        monkeypatch.delenv(name, raising=False)


def _tri_vendor(monkeypatch, *, live_hosts: bool):
    _clear(monkeypatch)
    anthropic = "https://api.lt4net.org" if live_hosts else "http://127.0.0.1:9/anthropic"
    deepseek = "https://token-plan.example.invalid/v1" if live_hosts else "http://127.0.0.1:9/deepseek"
    openai = "https://api.zhjjq.tech/v1" if live_hosts else "http://127.0.0.1:9/openai"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-fixture")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", anthropic)
    monkeypatch.setenv("CLAUDE_CODE_MODEL", "claude-opus-5-5")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-fixture")
    monkeypatch.setenv("OPENAI_BASE_URL", openai)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-6-astra")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-fixture")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", deepseek)
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4.1-flash")
    monkeypatch.setenv("EVOLUTION_PRIMARY_PROVIDER", "anthropic")
    monkeypatch.setenv("EVOLUTION_PROVIDER_STRICT", "false")


def test_silent_empty_failover_target_is_deepseek_flash(monkeypatch):
    """The old implicit switch lands on the model the live run served. It must not be used."""
    _tri_vendor(monkeypatch, live_hosts=True)
    transport = empty_failover_transport("anthropic", "claude-opus-5-5")
    assert transport is not None
    assert (transport.name, transport.config.model) == ("deepseek", "deepseek-v4.1-flash")


def _run(tmp_path, monkeypatch, post, *, effort="medium"):
    monkeypatch.setattr(messaging.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(messaging, "_post_json", post)
    runtime = AgentRuntime(tmp_path)
    runtime.assistant._interruptible_sleep = lambda *_args, **_kwargs: True
    try:
        created = runtime.assistant.create_run(prompt="做一个基线计划，先不要训练", start=False, reasoning_effort=effort)
        runtime.assistant._execute(created["id"], resume=False)
        snapshot = runtime.assistant.snapshot(created["id"])
        events = runtime.assistant.events(created["id"])
        return snapshot, events
    finally:
        runtime.close()


def test_default_opus_does_not_complete_on_deepseek_when_opus_is_empty(tmp_path, monkeypatch):
    _tri_vendor(monkeypatch, live_hosts=False)
    seen = []

    def post(url, headers, payload, timeout):
        seen.append(str(url))
        if str(url).rstrip("/").endswith("/messages"):
            return {"model": payload["model"], "stop_reason": "end_turn", "content": [],
                    "usage": {"input_tokens": 1, "output_tokens": 1}}
        return {"model": "deepseek-v4.1-flash", "choices": [{"finish_reason": "stop", "message": {"content": "deepseek plan"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    snapshot, events = _run(tmp_path, monkeypatch, post)
    assert snapshot["status"] != "completed"
    assert snapshot["model"] != "deepseek-v4.1-flash"
    assert not any("deepseek" in url for url in seen)
    assert not any(item["event_type"] == "model.empty_failover" for item in events)


def test_default_opus_does_not_fail_over_when_opus_errors(tmp_path, monkeypatch):
    import urllib.error
    _tri_vendor(monkeypatch, live_hosts=False)
    seen = []

    def post(url, headers, payload, timeout):
        seen.append((str(url), payload.get("model")))
        if str(url).rstrip("/").endswith("/messages"):
            raise urllib.error.URLError("fixture refused")
        return {"model": payload["model"], "choices": [{"finish_reason": "stop", "message": {"content": "deepseek plan"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    snapshot, _events = _run(tmp_path, monkeypatch, post)
    assert snapshot["model"] != "deepseek-v4.1-flash"
    assert {model for _url, model in seen} == {"claude-opus-5-5"}


def test_default_opus_is_kept_when_it_answers(tmp_path, monkeypatch):
    _tri_vendor(monkeypatch, live_hosts=False)
    seen = []

    def post(url, headers, payload, timeout):
        seen.append(payload.get("model"))
        return {"model": payload["model"], "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "一个基线计划。"}],
                "usage": {"input_tokens": 1, "output_tokens": 1}}

    snapshot, events = _run(tmp_path, monkeypatch, post)
    assert snapshot["status"] == "completed"
    assert snapshot["model"] == "claude-opus-5-5"
    assert snapshot["model_provider"] == "anthropic"
    assert seen == ["claude-opus-5-5"]
    observed = [item for item in events if item["event_type"] == "model_observed"]
    assert observed[-1]["payload"]["model"] == "claude-opus-5-5"


def test_personal_prompt_uses_managed_trainer_only_when_it_is_offered(tmp_path, monkeypatch):
    monkeypatch.setattr("evomind_runtime.tenant_training.enabled", lambda: None)
    runtime = AgentRuntime(tmp_path)
    token = current_principal.set(Principal(TENANT, "alice"))
    try:
        run = runtime.assistant.create_run(prompt="训练一个基线", start=False)
        withheld = runtime.assistant._execution_prompt(runtime.assistant.store.get_assistant_run(run["id"]), resume=False)
        assert "cannot be requested or approved" in withheld
        assert "call managed_tensor_train" not in withheld
        assert offered(TENANT, "shell_exec") is False
        assert offered(TENANT, "training_route") is False
        assert offered(TENANT, "managed_tensor_train") is False
    finally:
        current_principal.reset(token)
        runtime.close()

    monkeypatch.setattr("evomind_runtime.tenant_training.enabled", lambda: {"tenant_training": True, "service_tenant_id": TENANT})
    assert offered(TENANT, "managed_tensor_train") is True
    assert offered(TENANT, "shell_exec") is False
    clause = managed_training_clause(TENANT)
    assert "creates the approval card" in clause and "metrics.json" in clause
    assert "Do not use shell_exec" in clause
    runtime = AgentRuntime(tmp_path / "on")
    token = current_principal.set(Principal(TENANT, "alice"))
    try:
        run = runtime.assistant.create_run(prompt="训练一个基线", start=False)
        offered_prompt = runtime.assistant._execution_prompt(runtime.assistant.store.get_assistant_run(run["id"]), resume=False)
        assert "creates the approval card" in offered_prompt
        assert "cannot be requested or approved" not in offered_prompt
    finally:
        current_principal.reset(token)
        runtime.close()
    assert managed_training_clause("") == ""


def test_personal_managed_train_asks_for_approval_and_does_not_dispatch(tmp_path, monkeypatch):
    monkeypatch.setattr("evomind_runtime.tenant_training.enabled", lambda: {"tenant_training": True, "service_tenant_id": TENANT})
    runtime = AgentRuntime(tmp_path)
    token = current_principal.set(Principal(TENANT, "alice"))
    try:
        run = runtime.assistant.create_run(prompt="训练一个基线", start=False)
        shell = runtime.invoke_tool(run["id"], "shell_exec", {"argv": ["python", "-c", "print(1)"], "timeout_seconds": 5})
        assert shell["status"] == "failed"
        assert shell["result"]["error"] == "personal_tool_boundary"
        requested = runtime.invoke_tool(run["id"], "managed_tensor_train", {
            "protocol_id": "a" * 64,
            "solution_id": "baseline",
            "configuration_path": "work/train.json",
            "configuration_sha256": "b" * 64,
        })
        assert requested["status"] == "waiting_approval"
        assert requested["approval"]["tool_name"] == "managed_tensor_train"
    finally:
        current_principal.reset(token)
        runtime.close()


def test_approved_tenant_train_stops_when_the_binding_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSTATION_BYOA_STATE_ROOT", str(tmp_path))
    monkeypatch.setattr("evomind_runtime.tenant_training.enabled", lambda: {"tenant_training": True, "service_tenant_id": TENANT})
    context = ToolContext("run_train", tmp_path, tmp_path, tmp_path, tmp_path, None, {
        "http_access_scope": {"tenant_id": TENANT, "owner_principal_id": "alice"},
    })
    context.approval_verified = True
    result = _tenant_tensor_train({}, context)
    assert result.ok is False
    assert result.error == "tenant_training_binding_invalid"
    assert result.content["remote_dispatched"] is False
    assert "98270" not in json.dumps(result.content)
