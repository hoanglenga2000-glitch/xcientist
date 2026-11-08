"""Planning model turns end when their clock is spent.

The clock is a list the tests advance. Nothing here waits on a socket, and
nothing here starts a trainer.
"""
import threading

import pytest

from evomind_runtime import model_transport as mt
from evomind_runtime import run_supervision as sv
from evomind_runtime.model_transport import ModelTransportError, _StreamingClient
from evomind_runtime.personal_tool_boundary import managed_training_clause
from evomind_runtime.runtime import AgentRuntime
from evomind_runtime.tenant_access import Principal, current_principal


TENANT = "tenant_" + "ef" * 12


class _Transport:
    name = "anthropic"

    def __init__(self) -> None:
        self.config = type("C", (), {"model": "claude-opus-5-5"})()

    def build(self, messages, system, tools, max_tokens, temperature):
        return "http://127.0.0.1:9/v1/messages", {}, {"model": self.config.model}


class _Client:
    timeout = 120
    max_retries = 2

    def __init__(self) -> None:
        self.calls = []

    def _resolve_transports(self):
        return [_Transport()]

    def send(self, *args, **kwargs):
        self.calls.append("send")
        return type("Turn", (), {"text": "late", "tool_calls": [], "stop_reason": "end_turn"})()


def _streaming(client: _Client) -> _StreamingClient:
    streaming = _StreamingClient(client, lambda _event: None)
    streaming.during_request = None
    return streaming


def test_spent_stream_clock_does_not_open_another_request(monkeypatch):
    """A stream that has already used the request budget must not start client.send.

    The deadline worker reports that timeout as model_network_error, so the
    code string alone does not say the clock is spent. The clock does.
    """
    clock = [0.0]
    monkeypatch.setattr(mt.time, "monotonic", lambda: clock[0])
    client = _Client()

    def stream(*_args, **_kwargs):
        clock[0] = float(client.timeout)
        raise ModelTransportError("model_network_error", retryable=True)

    monkeypatch.setattr(mt, "post_stream_with_deadline", stream)
    with pytest.raises(ModelTransportError) as raised:
        _streaming(client).send([{"role": "user", "content": "plan"}], system="", tools=[])
    assert client.calls == []
    assert raised.value.code == "model_total_deadline_exceeded"
    assert raised.value.retryable is False


def test_fallback_that_overruns_the_request_clock_is_rejected(monkeypatch):
    """One fast stream failure must not keep a turn that spent three fresh socket budgets."""
    clock = [0.0]
    monkeypatch.setattr(mt.time, "monotonic", lambda: clock[0])
    client = _Client()

    def stream(*_args, **_kwargs):
        raise ModelTransportError("model_network_error", retryable=True)

    def send(*_args, **_kwargs):
        client.calls.append("send")
        clock[0] += float(client.timeout) * (client.max_retries + 1)
        return type("Turn", (), {"text": "late", "tool_calls": [], "stop_reason": "end_turn"})()

    client.send = send
    monkeypatch.setattr(mt, "post_stream_with_deadline", stream)
    with pytest.raises(ModelTransportError) as raised:
        _streaming(client).send([{"role": "user", "content": "plan"}], system="", tools=[])
    assert client.calls == ["send"]
    assert raised.value.code == "model_total_deadline_exceeded"
    assert clock[0] > client.timeout


def test_spent_request_budget_is_not_retried():
    decision = sv.model_retry_decision(
        "LLMError/model_total_deadline_exceeded", 0, code="model_total_deadline_exceeded")
    assert decision is not None
    assert decision.retry is False
    assert decision.category == "timeout"
    assert sv.model_retry_decision("LLMError/http_504", 0).retry is True
    assert sv.model_retry_decision("LLMError/model_network_error", 0).retry is True
    assert sv.model_retry_decision("LLMError/model_first_byte_timeout", 0).retry is True


def test_cancel_during_fallback_does_not_wait_for_the_socket(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(mt.time, "monotonic", lambda: clock[0])
    release = threading.Event()
    client = _Client()

    def stream(*_args, **_kwargs):
        raise ModelTransportError("model_network_error", retryable=True)

    def send(*_args, **_kwargs):
        client.calls.append("send")
        release.wait(2)
        return type("Turn", (), {"text": "late", "tool_calls": [], "stop_reason": "end_turn"})()

    client.send = send
    monkeypatch.setattr(mt, "post_stream_with_deadline", stream)
    streaming = _streaming(client)

    def cancel():
        raise RuntimeError("user_cancel_requested")

    streaming.during_request = cancel
    try:
        with pytest.raises(RuntimeError, match="user_cancel_requested"):
            streaming.send([{"role": "user", "content": "plan"}], system="", tools=[])
        assert clock[0] < client.timeout
    finally:
        release.set()


def test_one_spent_request_finalizes_the_turn(tmp_path, monkeypatch):
    clock = [0.0]
    sleeps = []
    calls = []
    runtime = AgentRuntime(tmp_path)
    runtime.assistant._interruptible_sleep = lambda _run_id, seconds: sleeps.append(seconds) or True

    def message(_session_id, _content, max_steps=12):
        calls.append(clock[0])
        clock[0] += 120.0
        return {
            "status": "blocked",
            "text": "LLMError/model_total_deadline_exceeded.",
            "error_code": "model_total_deadline_exceeded",
        }

    runtime.message = message
    try:
        created = runtime.assistant.create_run(prompt="只写计划，不要训练", start=False)
        runtime.assistant._execute(created["id"], resume=False)
        snapshot = runtime.assistant.snapshot(created["id"])
    finally:
        runtime.close()
    assert calls == [0.0]
    assert sleeps == []
    assert clock[0] == 120.0
    assert clock[0] < sv.planning_turn_ceiling_seconds()
    assert snapshot["status"] == "blocked"
    assert "model_total_deadline_exceeded" in str(snapshot.get("error_message") or snapshot.get("last_error") or "")


def test_passing_retries_remain_available_and_bounded(tmp_path):
    clock = [0.0]
    sleeps = []
    calls = []
    runtime = AgentRuntime(tmp_path)

    def sleep(_run_id, seconds):
        sleeps.append(seconds)
        clock[0] += seconds
        return True

    runtime.assistant._interruptible_sleep = sleep

    def message(_session_id, _content, max_steps=12):
        calls.append(1)
        clock[0] += 1.0
        if len(calls) <= 3:
            return {"status": "blocked", "text": "LLMError/model_network_error.", "error_code": "model_network_error"}
        return {
            "status": "completed",
            "text": "计划已写好，未启动训练。",
            "model_execution": {"provider": "anthropic", "model": "claude-opus-5-5", "native_tool_calls": 0},
        }

    runtime.message = message
    try:
        created = runtime.assistant.create_run(prompt="只写计划", start=False)
        runtime.assistant._execute(created["id"], resume=False)
        snapshot = runtime.assistant.snapshot(created["id"])
    finally:
        runtime.close()
    assert len(calls) == 4
    assert sleeps == [2.0, 8.0, 30.0]
    assert clock[0] == 4.0 + sum(sleeps)
    assert clock[0] < sv.planning_turn_ceiling_seconds()
    assert snapshot["status"] == "completed"
    assert snapshot.get("model") == "claude-opus-5-5"


def test_a_long_approval_is_not_closed_by_the_planning_clock(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "evomind_runtime.tenant_training.enabled",
        lambda: {"tenant_training": True, "service_tenant_id": TENANT},
    )
    clock = [0.0]
    runtime = AgentRuntime(tmp_path)
    token = current_principal.set(Principal(TENANT, "alice"))

    def message(_session_id, _content, max_steps=12):
        clock[0] = sv.planning_turn_ceiling_seconds() + 1800
        return {
            "status": "waiting_approval",
            "text": "需要你先确认这次训练。",
            "approval": {"tool_name": "managed_tensor_train"},
            "model_execution": {"provider": "anthropic", "model": "claude-opus-5-5", "native_tool_calls": 1},
        }

    runtime.message = message
    try:
        created = runtime.assistant.create_run(prompt="批准后训练一个基线", start=False)
        runtime.assistant._execute(created["id"], resume=False)
        snapshot = runtime.assistant.snapshot(created["id"])
        tools = runtime.store.list_tool_calls(created["id"])
    finally:
        current_principal.reset(token)
        runtime.close()
    assert snapshot["status"] == "waiting_approval"
    assert clock[0] > sv.planning_turn_ceiling_seconds()
    assert all(row.get("tool_name") != "managed_tensor_train" or row.get("status") == "waiting_approval" for row in tools)


def test_plan_only_answer_does_not_execute_training(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "evomind_runtime.tenant_training.enabled",
        lambda: {"tenant_training": True, "service_tenant_id": TENANT},
    )
    runtime = AgentRuntime(tmp_path)
    token = current_principal.set(Principal(TENANT, "alice"))

    def message(_session_id, content, max_steps=12):
        assert "managed_tensor_train" in content or "managed_tensor_train" in managed_training_clause(TENANT)
        assert "call managed_tensor_train" in content
        return {
            "status": "completed",
            "text": "已完成计划，未启动训练。",
            "model_execution": {"provider": "anthropic", "model": "claude-opus-5-5", "native_tool_calls": 0},
        }

    runtime.message = message
    try:
        created = runtime.assistant.create_run(prompt="只写计划，不要训练，不要申请计费", start=False)
        runtime.assistant._execute(created["id"], resume=False)
        snapshot = runtime.assistant.snapshot(created["id"])
        tools = [row.get("tool_name") for row in runtime.store.list_tool_calls(created["id"])]
    finally:
        current_principal.reset(token)
        runtime.close()
    assert snapshot["status"] == "completed"
    assert "managed_tensor_train" not in tools
    assert snapshot.get("model") == "claude-opus-5-5"
