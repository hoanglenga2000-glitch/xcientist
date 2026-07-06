"""Tool-use-capable, multi-turn message client for the deep agent.

The existing ``llm_client.LLMClient`` is single-shot text in / text out — perfect
for the variation generator, useless for an agent that must *call tools and read
their results across turns*. This module adds exactly that missing layer:

  * a ``tools`` parameter (Anthropic-native tool-use) sent on ``/v1/messages``;
  * parsing of the assistant turn into text blocks + ``tool_use`` blocks;
  * a ``stop_reason`` so the caller's loop knows when the model wants a tool vs.
    is done;
  * multi-turn message accumulation driven by the caller (the AgentSession).

Design rules mirror ``llm_client`` / ``gpu_credentials`` deliberately:
  * credentials come only from the environment, never hardcoded, never logged;
  * standard-library HTTP only (urllib) so this runs on the GPU box too;
  * the gateway (Anthropic-compatible ``ANTHROPIC_BASE_URL``) is the target, so
    "wiring the gateway into an agent" is just: point at it and send ``tools``.

Tool-use is Anthropic-shaped here on purpose. DeepSeek's function-calling schema
differs; Phase A targets the Anthropic-compatible gateway and fails with a clear
message if no Anthropic key is configured, rather than silently degrading.
"""
from __future__ import annotations

import time
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Optional

from ..llm_client import LLMError, ProviderConfig, _env, _post_json


@dataclass
class ToolSpec:
    """One tool the agent may call: a name, a description, and a JSON schema for
    its input. Serialized to the Anthropic ``tools`` wire format on each send."""

    name: str
    description: str
    input_schema: dict[str, Any]

    def to_wire(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class ToolCall:
    """A ``tool_use`` block the model emitted: what it wants to run, with args."""

    id: str
    name: str
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """The outcome the caller feeds back for a ToolCall (becomes a user turn)."""

    tool_use_id: str
    content: str
    is_error: bool = False

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": self.content,
            "is_error": self.is_error,
        }


@dataclass
class AssistantTurn:
    """A parsed assistant response: the free text, any tool calls, why it stopped."""

    text: str
    tool_calls: list[ToolCall]
    stop_reason: str
    raw_content: list[dict[str, Any]]  # the exact content blocks, to echo back as history
    provider: str = "anthropic"
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def wants_tool(self) -> bool:
        return self.stop_reason == "tool_use" or bool(self.tool_calls)


def _parse_turn(body: dict[str, Any], fallback_model: str) -> AssistantTurn:
    """Turn a raw ``/v1/messages`` body into a structured assistant turn."""
    content = body.get("content") or []
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append(ToolCall(
                id=block.get("id", ""),
                name=block.get("name", ""),
                input=block.get("input") or {},
            ))
    usage = body.get("usage", {}) or {}
    return AssistantTurn(
        text="".join(text_parts).strip(),
        tool_calls=tool_calls,
        stop_reason=body.get("stop_reason", "") or "",
        raw_content=content,
        provider="anthropic",
        model=body.get("model", fallback_model),
        input_tokens=int(usage.get("input_tokens", 0) or 0),
        output_tokens=int(usage.get("output_tokens", 0) or 0),
    )


class EmptyResponseError(RuntimeError):
    """A turn with no text, no tool calls, and no clean stop — usually a transient
    rate-limit/overload symptom. Treated as retryable / failover-able, not returned."""


def _parse_openai_turn(body: dict[str, Any], fallback_model: str) -> AssistantTurn:
    """Parse an OpenAI-compatible chat/completions body into an AssistantTurn.

    Lets the agent run against an OpenAI-style gateway (tool_calls with a JSON
    ``arguments`` string) as a fallback provider, without the loop caring which
    wire format produced the turn."""
    import json as _json
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = message.get("content") or ""
    tool_calls: list[ToolCall] = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments")
        try:
            args = _json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except (ValueError, TypeError):
            args = {}
        tool_calls.append(ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), input=args))
    finish = choice.get("finish_reason", "") or ""
    stop = "tool_use" if (finish == "tool_calls" or tool_calls) else (finish or "end_turn")
    usage = body.get("usage", {}) or {}
    return AssistantTurn(
        text=(text or "").strip(), tool_calls=tool_calls, stop_reason=stop,
        raw_content=[{"type": "text", "text": text or ""}],  # normalized for history
        provider="openai", model=body.get("model", fallback_model),
        input_tokens=int(usage.get("prompt_tokens", 0) or 0),
        output_tokens=int(usage.get("completion_tokens", 0) or 0),
    )


class Transport:
    """A provider wire format. ``build`` returns (url, headers, payload); ``parse``
    turns the response body into an AssistantTurn. Credentials live on the config
    and are never logged. Subclasses implement one provider's format."""

    name = "base"

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    def build(self, messages, system, tools, max_tokens, temperature):  # pragma: no cover
        raise NotImplementedError

    def parse(self, body: dict[str, Any]) -> AssistantTurn:  # pragma: no cover
        raise NotImplementedError


class AnthropicTransport(Transport):
    name = "anthropic"

    def build(self, messages, system, tools, max_tokens, temperature):
        url = self.config.base_url.rstrip("/") + "/v1/messages"
        headers = {"x-api-key": self.config.api_key, "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        payload = {
            "model": self.config.model, "max_tokens": max_tokens, "temperature": temperature,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": messages, "tools": [t.to_wire() for t in tools],
        }
        return url, headers, payload

    def parse(self, body: dict[str, Any]) -> AssistantTurn:
        return _parse_turn(body, self.config.model)


class OpenAITransport(Transport):
    name = "openai"

    def build(self, messages, system, tools, max_tokens, temperature):
        url = self.config.base_url.rstrip("/") + "/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.config.api_key}", "content-type": "application/json"}
        # System goes as a leading system message; tools use the function wrapper.
        wire_msgs = [{"role": "system", "content": system}, *messages]
        wire_tools = [{"type": "function", "function": {
            "name": t.name, "description": t.description, "parameters": t.input_schema}} for t in tools]
        payload = {"model": self.config.model, "max_tokens": max_tokens,
                   "temperature": temperature, "messages": wire_msgs, "tools": wire_tools}
        return url, headers, payload

    def parse(self, body: dict[str, Any]) -> AssistantTurn:
        return _parse_openai_turn(body, self.config.model)


def _resolve_transports() -> list[Transport]:
    """Build the ordered transport list (primary first) from the environment.

    Anthropic-native is primary (the gateway). An OpenAI-compatible fallback is
    added when OPENAI_API_KEY/OPENAI_BASE_URL are set. DeepSeek's OpenAI-style
    endpoint also works as a fallback when configured that way."""
    transports: list[Transport] = []
    akey = _env("ANTHROPIC_API_KEY")
    if akey:
        transports.append(AnthropicTransport(ProviderConfig(
            "anthropic", _env("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            _env("CLAUDE_CODE_MODEL", "claude-opus-4-8"), akey)))
    okey = _env("OPENAI_API_KEY") or _env("DEEPSEEK_API_KEY")
    if okey:
        base = _env("OPENAI_BASE_URL") or _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        model = _env("OPENAI_MODEL") or _env("DEEPSEEK_MODEL", "deepseek-chat")
        transports.append(OpenAITransport(ProviderConfig("openai", base, model, okey)))
    return transports


class AgentMessageClient:
    """Multi-turn, tool-use client with provider abstraction + 3-layer recovery.

    Layer 1 — per-transport bounded retry (absorbs a transient gateway blip).
    Layer 2 — provider failover: exhaust the primary transport's retries, then try
              the next configured transport (anthropic → openai-compatible).
    Layer 3 — empty/malformed-response self-heal: an empty turn (no text, no tools,
              no clean stop) is treated as a retryable failure rather than returned
              to the loop as a dead turn.

    The caller (AgentSession) owns the ``messages`` list; this client does one
    authenticated round-trip and returns the parsed turn. Credentials/URLs are
    never echoed in errors.
    """

    def __init__(self, *, max_retries: int = 2, timeout: int = 120,
                 transports: Optional[list[Transport]] = None) -> None:
        self.max_retries = max_retries
        self.timeout = timeout
        self._transports = transports  # lazily resolved from env if None

    def _resolve_transports(self) -> list[Transport]:
        if self._transports is None:
            self._transports = _resolve_transports()
        if not self._transports:
            raise LLMError(
                "the deep agent needs an Anthropic-compatible key: run `xsci login "
                "--provider anthropic` (tool-use is sent to ANTHROPIC_BASE_URL)."
            )
        return self._transports

    @property
    def model(self) -> str:
        return self._resolve_transports()[0].config.model

    def is_available(self) -> bool:
        try:
            return bool(self._transports) or bool(_resolve_transports())
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _is_empty(turn: AssistantTurn) -> bool:
        return not turn.text and not turn.tool_calls and turn.stop_reason not in ("end_turn", "stop")

    def send(self, messages: list[dict[str, Any]], *, system: str, tools: list[ToolSpec],
             max_tokens: int = 4096, temperature: float = 0.3) -> AssistantTurn:
        """One round-trip with retry → failover → empty-response self-heal."""
        transports = self._resolve_transports()
        errors: list[str] = []
        for transport in transports:  # Layer 2: provider failover
            url, headers, payload = transport.build(messages, system, tools, max_tokens, temperature)
            for attempt in range(self.max_retries + 1):  # Layer 1: per-transport retry
                try:
                    body = _post_json(url, headers, payload, self.timeout)
                    turn = transport.parse(body)
                    if self._is_empty(turn):  # Layer 3: empty-response self-heal
                        raise EmptyResponseError("empty assistant turn")
                    return turn
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                        ValueError, EmptyResponseError) as exc:
                    # Never echo the exception URL/body verbatim (could carry the key).
                    errors.append(f"{transport.name} attempt {attempt + 1}: {type(exc).__name__}")
                    if attempt < self.max_retries:
                        time.sleep(1.5 * (attempt + 1))
        raise LLMError("agent message round-trip failed: " + "; ".join(errors))
