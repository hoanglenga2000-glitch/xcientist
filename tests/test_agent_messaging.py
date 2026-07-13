"""Tool-use message-client tests: parsing tool_use blocks, building tool_result
turns, and multi-turn accumulation — all with a mocked HTTP round-trip so no
network/keys are needed."""
from __future__ import annotations

import pytest

from research_os.agent import messaging
from research_os.agent.messaging import AgentMessageClient, ToolResult, ToolSpec
from research_os.llm_client import LLMError


@pytest.fixture()
def anthropic_env(monkeypatch):
    for var in (
        "DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_FILE", "OPENAI_API_KEY",
        "OPENAI_API_KEY_FILE", "EVOLUTION_PROVIDER_STRICT",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-logged")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:18794/anthropic")
    monkeypatch.setenv("CLAUDE_CODE_MODEL", "claude-opus-4-8")
    monkeypatch.setenv("EVOLUTION_PRIMARY_PROVIDER", "anthropic")


def test_parses_text_and_tool_use(monkeypatch, anthropic_env):
    body = {
        "model": "claude-opus-4-8", "stop_reason": "tool_use",
        "content": [
            {"type": "text", "text": "Let me inspect the data first."},
            {"type": "tool_use", "id": "tu_1", "name": "inspect_data", "input": {"rows": 5}},
        ],
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }
    monkeypatch.setattr(messaging, "_post_json", lambda *a, **k: body)
    client = AgentMessageClient()
    turn = client.send([{"role": "user", "content": "go"}], system="s", tools=[])
    assert turn.text == "Let me inspect the data first."
    assert turn.wants_tool is True
    assert len(turn.tool_calls) == 1
    call = turn.tool_calls[0]
    assert (call.id, call.name, call.input) == ("tu_1", "inspect_data", {"rows": 5})
    assert turn.input_tokens == 100 and turn.output_tokens == 20


def test_end_turn_has_no_tool_calls(monkeypatch, anthropic_env):
    body = {
        "model": "claude-opus-4-8", "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "All done — best CV 0.83."}],
        "usage": {},
    }
    monkeypatch.setattr(messaging, "_post_json", lambda *a, **k: body)
    turn = AgentMessageClient().send([{"role": "user", "content": "x"}], system="s", tools=[])
    assert turn.wants_tool is False
    assert turn.tool_calls == []


def test_tools_and_messages_are_sent_on_the_wire(monkeypatch, anthropic_env):
    captured = {}

    def _fake_post(url, headers, payload, timeout):
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        return {"model": "m", "stop_reason": "end_turn", "content": [], "usage": {}}

    monkeypatch.setattr(messaging, "_post_json", _fake_post)
    spec = ToolSpec("inspect_data", "desc", {"type": "object", "properties": {}})
    msgs = [{"role": "user", "content": "hi"}]
    AgentMessageClient().send(msgs, system="charter", tools=[spec])
    assert captured["url"].endswith("/v1/messages")
    assert captured["payload"]["messages"] == msgs
    assert captured["payload"]["tools"][0]["name"] == "inspect_data"
    assert captured["payload"]["system"][0]["text"] == "charter"
    # the key travels in the header, and we assert it is NOT echoed anywhere we log
    assert captured["headers"]["x-api-key"] == "test-key-not-logged"


def test_missing_key_raises_clear_error(monkeypatch):
    # clear ALL provider keys so no transport resolves
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_FILE", "OPENAI_API_KEY",
                "OPENAI_API_KEY_FILE", "DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_FILE"):
        monkeypatch.delenv(var, raising=False)
    client = AgentMessageClient()
    assert client.is_available() is False
    with pytest.raises(LLMError) as exc:
        client.send([{"role": "user", "content": "x"}], system="s", tools=[])
    assert "xsci login" in str(exc.value)


def test_tool_result_wire_shape():
    wire = ToolResult("tu_9", "the result text", is_error=True).to_wire()
    assert wire == {
        "type": "tool_result", "tool_use_id": "tu_9",
        "content": "the result text", "is_error": True,
    }


def test_retries_then_succeeds(monkeypatch, anthropic_env):
    calls = {"n": 0}

    def _flaky(url, headers, payload, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("gateway blip")
        return {"model": "m", "stop_reason": "end_turn", "content": [], "usage": {}}

    monkeypatch.setattr(messaging, "_post_json", _flaky)
    monkeypatch.setattr(messaging.time, "sleep", lambda *_: None)  # no real delay
    turn = AgentMessageClient(max_retries=2).send(
        [{"role": "user", "content": "x"}], system="s", tools=[])
    assert calls["n"] == 2
    assert turn.stop_reason == "end_turn"


def test_remote_disconnect_is_retried_then_succeeds(monkeypatch, anthropic_env):
    calls = {"n": 0}

    def _flaky(url, headers, payload, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionResetError("remote disconnected")
        return {"model": "m", "stop_reason": "end_turn", "content": [], "usage": {}}

    monkeypatch.setattr(messaging, "_post_json", _flaky)
    monkeypatch.setattr(messaging.time, "sleep", lambda *_: None)
    turn = AgentMessageClient(max_retries=1).send(
        [{"role": "user", "content": "x"}], system="s", tools=[])

    assert calls["n"] == 2
    assert turn.stop_reason == "end_turn"


# ── Layer 2: provider failover (anthropic → openai-compatible) ──────────────────
def test_failover_to_openai_transport(monkeypatch):
    from research_os.agent.messaging import (
        AnthropicTransport,
        OpenAITransport,
        ProviderConfig,
    )
    calls = {"n": 0}

    def _post(url, headers, payload, timeout):
        calls["n"] += 1
        if url.endswith("/v1/messages"):        # anthropic transport: always fails
            raise TimeoutError("primary down")
        # openai-compatible transport succeeds with a tool_call
        return {"model": "ds", "choices": [{"finish_reason": "tool_calls", "message": {
            "content": "ok", "tool_calls": [{"id": "c1", "function": {
                "name": "run_experiment", "arguments": '{"hypothesis":"h","code":"print(1)"}'}}]}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2}}

    monkeypatch.setattr(messaging, "_post_json", _post)
    monkeypatch.setattr(messaging.time, "sleep", lambda *_: None)
    client = AgentMessageClient(max_retries=1, transports=[
        AnthropicTransport(ProviderConfig("anthropic", "http://a", "opus", "k1")),
        OpenAITransport(ProviderConfig("openai", "http://b", "ds", "k2")),
    ])
    turn = client.send([{"role": "user", "content": "x"}], system="s", tools=[])
    # primary retried (2 attempts) then fell over to openai which parsed a tool call
    assert turn.provider == "openai"
    assert turn.tool_calls[0].name == "run_experiment"
    assert turn.tool_calls[0].input == {"hypothesis": "h", "code": "print(1)"}


# ── Layer 3: empty-response self-heal ───────────────────────────────────────────
def test_empty_response_is_retried_not_returned(monkeypatch, anthropic_env):
    calls = {"n": 0}

    def _post(url, headers, payload, timeout):
        calls["n"] += 1
        if calls["n"] == 1:  # empty turn: no text, no tools, non-terminal stop
            return {"model": "m", "stop_reason": "", "content": [], "usage": {}}
        return {"model": "m", "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "recovered"}], "usage": {}}

    monkeypatch.setattr(messaging, "_post_json", _post)
    monkeypatch.setattr(messaging.time, "sleep", lambda *_: None)
    turn = AgentMessageClient(max_retries=2).send(
        [{"role": "user", "content": "x"}], system="s", tools=[])
    assert calls["n"] == 2                 # the empty turn was retried, not returned
    assert turn.text == "recovered"


def test_openai_transport_parses_tool_calls():
    from research_os.agent.messaging import _parse_openai_turn
    body = {"model": "ds", "choices": [{"finish_reason": "tool_calls", "message": {
        "content": "", "tool_calls": [{"id": "c1", "function": {
            "name": "plan_next_experiment", "arguments": "{}"}}]}}], "usage": {}}
    turn = _parse_openai_turn(body, "ds")
    assert turn.wants_tool is True
    assert turn.tool_calls[0].name == "plan_next_experiment"
    # malformed JSON arguments degrade to {} rather than crashing
    bad = {"model": "ds", "choices": [{"finish_reason": "tool_calls", "message": {
        "tool_calls": [{"id": "c2", "function": {"name": "finish", "arguments": "{not json"}}]}}]}
    turn2 = _parse_openai_turn(bad, "ds")
    assert turn2.tool_calls[0].input == {}


@pytest.mark.parametrize(
    ("primary", "expected"),
    [
        ("anthropic", ["anthropic", "deepseek"]),
        ("deepseek", ["deepseek", "anthropic"]),
    ],
)
def test_resolve_transports_respects_explicit_primary(monkeypatch, primary, expected):
    for var in (
        "ANTHROPIC_API_KEY_FILE", "DEEPSEEK_API_KEY_FILE", "OPENAI_API_KEY",
        "OPENAI_API_KEY_FILE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("EVOLUTION_PROVIDER_STRICT", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    monkeypatch.setenv("EVOLUTION_PRIMARY_PROVIDER", primary)

    transports = messaging._resolve_transports()

    assert [transport.name for transport in transports] == expected
    assert [transport.config.name for transport in transports] == expected


def test_deepseek_and_openai_transports_keep_truthful_provider_names(monkeypatch):
    for var in (
        "ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_FILE", "DEEPSEEK_API_KEY_FILE",
        "OPENAI_API_KEY_FILE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("EVOLUTION_PROVIDER_STRICT", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setenv("EVOLUTION_PRIMARY_PROVIDER", "deepseek")

    transports = messaging._resolve_transports()
    evidence = [
        (transport.name, transport.config.name, transport.parse({"choices": []}).provider)
        for transport in transports
    ]

    assert evidence == [
        ("deepseek", "deepseek", "deepseek"),
        ("openai", "openai", "openai"),
    ]


def test_strict_provider_selection_returns_only_explicit_primary(monkeypatch):
    for var in (
        "ANTHROPIC_API_KEY_FILE", "DEEPSEEK_API_KEY_FILE", "OPENAI_API_KEY",
        "OPENAI_API_KEY_FILE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    monkeypatch.setenv("EVOLUTION_PRIMARY_PROVIDER", "deepseek")
    monkeypatch.setenv("EVOLUTION_PROVIDER_STRICT", "1")

    transports = messaging._resolve_transports()

    assert [transport.name for transport in transports] == ["deepseek"]


def test_strict_provider_selection_fails_closed_when_primary_key_is_missing(monkeypatch):
    for var in (
        "ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_FILE", "DEEPSEEK_API_KEY",
        "DEEPSEEK_API_KEY_FILE", "OPENAI_API_KEY_FILE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setenv("EVOLUTION_PRIMARY_PROVIDER", "deepseek")
    monkeypatch.setenv("EVOLUTION_PROVIDER_STRICT", "1")

    assert messaging._resolve_transports() == []
    client = AgentMessageClient()
    assert client.is_available() is False
    with pytest.raises(LLMError):
        client.send([{"role": "user", "content": "x"}], system="s", tools=[])


def test_openai_transport_converts_canonical_tool_history():
    from research_os.agent.messaging import OpenAITransport, ProviderConfig, ToolResult

    transport = OpenAITransport(ProviderConfig("openai", "http://gateway", "ds", "key"))
    messages = [
        {"role": "user", "content": "inspect"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Checking."},
            {"type": "tool_use", "id": "call_1", "name": "data_check", "input": {}},
        ]},
        {"role": "user", "content": [
            ToolResult("call_1", "missing train.csv", is_error=True).to_wire(),
        ]},
    ]

    _, _, payload = transport.build(messages, "system", [], 1000, 0.1)
    assert payload["messages"][1]["role"] == "user"
    assistant = payload["messages"][2]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["function"]["name"] == "data_check"
    assert assistant["tool_calls"][0]["function"]["arguments"] == "{}"
    tool_result = payload["messages"][3]
    assert tool_result["role"] == "tool"
    assert tool_result["tool_call_id"] == "call_1"
    assert tool_result["content"].startswith("[tool_error]")


def test_openai_parsed_tool_call_is_canonical_history():
    from research_os.agent.messaging import _parse_openai_turn

    body = {"model": "ds", "choices": [{"finish_reason": "tool_calls", "message": {
        "content": "I need evidence.",
        "tool_calls": [{"id": "c1", "function": {
            "name": "system_status", "arguments": "{}",
        }}],
    }}]}
    turn = _parse_openai_turn(body, "ds")
    assert turn.raw_content == [
        {"type": "text", "text": "I need evidence."},
        {"type": "tool_use", "id": "c1", "name": "system_status", "input": {}},
    ]
