"""Unified multi-backend LLM client for the evolution engine.

Supports Anthropic-native ``/v1/messages`` plus DeepSeek and generic OpenAI-style
``/v1/chat/completions`` endpoints. Provider selection is controlled through the
``EVOLUTION_PRIMARY_PROVIDER`` / ``EVOLUTION_FALLBACK_PROVIDER`` environment.

Design rules (mirror gpu_credentials.py):
  * Credentials come only from the environment; never hardcoded, never logged.
  * Standard-library HTTP only (urllib) so the same module runs on the GPU box
    without extra pip installs.
  * A single ``generate`` entry point with automatic primary->fallback failover
    and bounded retries, so callers never see a transient gateway error.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


class LLMError(RuntimeError):
    """Raised when all providers fail to return a completion."""


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    raw_usage: dict[str, Any] = field(default_factory=dict)
    request_profile: dict[str, str] = field(default_factory=dict)

    def __repr__(self) -> str:  # keep tokens visible, never echo the prompt/keys
        return (
            f"LLMResponse(provider={self.provider!r}, model={self.model!r}, "
            f"in={self.input_tokens}, out={self.output_tokens}, cache_read={self.cache_read_tokens})"
        )


@dataclass
class LLMStreamEvent:
    """One provider-neutral event from a streaming completion."""

    kind: str
    text: str = ""
    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    request_profile: dict[str, str] = field(default_factory=dict)


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    model: str
    api_key: str  # never logged
    reasoning_effort: Optional[str] = None
    service_tier: Optional[str] = None


def _openai_chat_url(base_url: str) -> str:
    """Return one canonical chat-completions URL for either host or ``/v1`` bases."""

    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    if value:
        return value
    file_var = os.environ.get(f"{name}_FILE")
    if file_var and os.path.exists(file_var):
        with open(file_var, encoding="utf-8") as handle:
            content = handle.read().strip()
        if content:
            return content
    return default


_OPENAI_REASONING_EFFORTS = {"low", "medium", "high", "xhigh", "max", "ultra"}
_OPENAI_SERVICE_TIERS = {"auto", "default", "flex", "priority"}


def openai_request_profile_from_env() -> dict[str, str]:
    """Resolve validated OpenAI performance controls without exposing secrets."""

    reasoning_effort = (_env("OPENAI_REASONING_EFFORT", "") or "").strip().lower()
    service_tier = (_env("OPENAI_SERVICE_TIER", "") or "").strip().lower()
    if reasoning_effort and reasoning_effort not in _OPENAI_REASONING_EFFORTS:
        raise ValueError("OPENAI_REASONING_EFFORT is unsupported")
    if service_tier and service_tier not in _OPENAI_SERVICE_TIERS:
        raise ValueError("OPENAI_SERVICE_TIER is unsupported")
    return {
        **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}),
        **({"service_tier": service_tier} if service_tier else {}),
    }


def openai_request_options(config: ProviderConfig) -> dict[str, str]:
    if config.name != "openai":
        return {}
    return {
        **(
            {"reasoning_effort": config.reasoning_effort}
            if config.reasoning_effort
            else {}
        ),
        **({"service_tier": config.service_tier} if config.service_tier else {}),
    }


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    # Some gateways (Cloudflare, error 1010) reject urllib's default UA; send a normal one.
    headers = {"User-Agent": "research-os-evolution/1.0", **headers}
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_sse(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: int,
) -> Iterator[tuple[str, str]]:
    """Yield ``(event, data)`` records from a standards-compliant SSE response."""

    data = json.dumps(payload).encode("utf-8")
    request_headers = {
        "User-Agent": "research-os-evolution/1.0",
        "Accept": "text/event-stream",
        **headers,
    }
    request = urllib.request.Request(url, data=data, headers=request_headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        event_name = "message"
        data_lines: list[str] = []
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                if data_lines:
                    yield event_name, "\n".join(data_lines)
                event_name = "message"
                data_lines = []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line[6:].strip() or "message"
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if data_lines:
            yield event_name, "\n".join(data_lines)


def _anthropic_stream_events(
    records: Iterator[tuple[str, str]],
    config: ProviderConfig,
) -> Iterator[LLMStreamEvent]:
    model = config.model
    input_tokens = 0
    output_tokens = 0
    for _event_name, raw_data in records:
        if raw_data == "[DONE]":
            yield LLMStreamEvent(
                "done", provider="anthropic", model=model,
                input_tokens=input_tokens, output_tokens=output_tokens,
            )
            return
        body = json.loads(raw_data)
        event_type = str(body.get("type") or "")
        if event_type == "message_start":
            message = body.get("message") or {}
            model = str(message.get("model") or model)
            usage = message.get("usage") or {}
            input_tokens = int(usage.get("input_tokens", 0) or 0)
            yield LLMStreamEvent("start", provider="anthropic", model=model, input_tokens=input_tokens)
        elif event_type == "content_block_start":
            block = body.get("content_block") or {}
            text = str(block.get("text") or block.get("thinking") or "")
            if text:
                kind = "thinking_delta" if block.get("type") == "thinking" else "text_delta"
                yield LLMStreamEvent(kind, text=text, provider="anthropic", model=model)
        elif event_type == "content_block_delta":
            delta = body.get("delta") or {}
            delta_type = str(delta.get("type") or "")
            if delta_type == "thinking_delta":
                text = str(delta.get("thinking") or "")
                if text:
                    yield LLMStreamEvent("thinking_delta", text=text, provider="anthropic", model=model)
            elif delta_type == "text_delta":
                text = str(delta.get("text") or "")
                if text:
                    yield LLMStreamEvent("text_delta", text=text, provider="anthropic", model=model)
        elif event_type == "message_delta":
            usage = body.get("usage") or {}
            output_tokens = int(usage.get("output_tokens", output_tokens) or output_tokens)
        elif event_type == "message_stop":
            yield LLMStreamEvent(
                "done", provider="anthropic", model=model,
                input_tokens=input_tokens, output_tokens=output_tokens,
            )
            return
        elif event_type == "error":
            raise ConnectionError("anthropic streaming response returned an error event")


def _openai_stream_events(
    records: Iterator[tuple[str, str]],
    config: ProviderConfig,
) -> Iterator[LLMStreamEvent]:
    model = config.model
    request_profile = openai_request_options(config)
    input_tokens = 0
    output_tokens = 0
    for _event_name, raw_data in records:
        if raw_data == "[DONE]":
            yield LLMStreamEvent(
                "done", provider=config.name, model=model,
                input_tokens=input_tokens, output_tokens=output_tokens,
                request_profile=request_profile,
            )
            return
        body = json.loads(raw_data)
        model = str(body.get("model") or model)
        usage = body.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens", input_tokens) or input_tokens)
        output_tokens = int(usage.get("completion_tokens", output_tokens) or output_tokens)
        choices = body.get("choices") or []
        for choice in choices:
            delta = choice.get("delta") or {}
            thinking = str(delta.get("reasoning_content") or delta.get("thinking") or "")
            content = str(delta.get("content") or "")
            if thinking:
                yield LLMStreamEvent(
                    "thinking_delta",
                    text=thinking,
                    provider=config.name,
                    model=model,
                    request_profile=request_profile,
                )
            if content:
                yield LLMStreamEvent(
                    "text_delta",
                    text=content,
                    provider=config.name,
                    model=model,
                    request_profile=request_profile,
                )
            if choice.get("finish_reason"):
                yield LLMStreamEvent(
                    "done", provider=config.name, model=model,
                    input_tokens=input_tokens, output_tokens=output_tokens,
                    request_profile=request_profile,
                )
                return


def _stream_provider(
    config: ProviderConfig,
    *,
    system: Optional[str],
    user: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> Iterator[LLMStreamEvent]:
    if config.name == "anthropic":
        url = config.base_url.rstrip("/") + "/v1/messages"
        headers = {
            "x-api-key": config.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": config.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": user}],
            "stream": True,
        }
        if system:
            payload["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        yield from _anthropic_stream_events(_post_sse(url, headers, payload, timeout), config)
        return

    url = _openai_chat_url(config.base_url)
    headers = {"Authorization": f"Bearer {config.api_key}", "content-type": "application/json"}
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    payload = {
        "model": config.model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        **openai_request_options(config),
    }
    yield from _openai_stream_events(_post_sse(url, headers, payload, timeout), config)


def _call_anthropic(config: ProviderConfig, *, system: Optional[str], user: str,
                    max_tokens: int, temperature: float, timeout: int) -> LLMResponse:
    url = config.base_url.rstrip("/") + "/v1/messages"
    headers = {
        "x-api-key": config.api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": config.model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": user}],
    }
    if system:
        # Mark the system block cacheable; harmless if the gateway ignores it.
        payload["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    body = _post_json(url, headers, payload, timeout)
    parts = body.get("content") or []
    text = "".join(part.get("text", "") for part in parts if part.get("type") == "text")
    usage = body.get("usage", {}) or {}
    return LLMResponse(
        text=text, provider="anthropic", model=body.get("model", config.model),
        input_tokens=int(usage.get("input_tokens", 0) or 0),
        output_tokens=int(usage.get("output_tokens", 0) or 0),
        cache_read_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
        raw_usage=usage,
    )


def _call_openai_compatible(config: ProviderConfig, *, system: Optional[str], user: str,
                            max_tokens: int, temperature: float, timeout: int) -> LLMResponse:
    url = _openai_chat_url(config.base_url)
    headers = {"Authorization": f"Bearer {config.api_key}", "content-type": "application/json"}
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    payload = {
        "model": config.model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
        **openai_request_options(config),
    }
    body = _post_json(url, headers, payload, timeout)
    choice = (body.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content", "")
    usage = body.get("usage", {}) or {}
    details = usage.get("prompt_tokens_details", {}) or {}
    return LLMResponse(
        text=text, provider=config.name, model=body.get("model", config.model),
        input_tokens=int(usage.get("prompt_tokens", 0) or 0),
        output_tokens=int(usage.get("completion_tokens", 0) or 0),
        cache_read_tokens=int(details.get("cached_tokens", 0) or 0),
        raw_usage=usage,
        request_profile=openai_request_options(config),
    )


_CALLERS: dict[str, Callable[..., LLMResponse]] = {
    "anthropic": _call_anthropic,
    "deepseek": _call_openai_compatible,
    "openai": _call_openai_compatible,
}


class LLMClient:
    """Multi-backend client with automatic failover and bounded retries.

    Providers are resolved from the environment at construction time. The primary
    provider (default: anthropic/opus) is tried first; on failure the client
    retries, then fails over to the fallback provider (default: deepseek).
    """

    def __init__(self, *, primary: Optional[str] = None, fallback: Optional[str] = None,
                 max_retries: int = 2, timeout: int = 90, temperature: float = 0.4) -> None:
        self.primary_name = (primary or _env("EVOLUTION_PRIMARY_PROVIDER", "anthropic") or "anthropic").lower()
        self.fallback_name = (fallback or _env("EVOLUTION_FALLBACK_PROVIDER", "deepseek") or "deepseek").lower()
        self.max_retries = max_retries
        self.timeout = timeout
        self.temperature = temperature

    def _resolve(self, name: str) -> Optional[ProviderConfig]:
        if name == "anthropic":
            key = _env("ANTHROPIC_API_KEY")
            if not key:
                return None
            return ProviderConfig("anthropic", _env("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
                                  _env("CLAUDE_CODE_MODEL", "claude-opus-4-8"), key)
        if name == "deepseek":
            key = _env("DEEPSEEK_API_KEY")
            if not key:
                return None
            return ProviderConfig("deepseek", _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                                  _env("DEEPSEEK_MODEL", "deepseek-chat"), key)
        if name == "openai":
            key = _env("OPENAI_API_KEY")
            if not key:
                return None
            return ProviderConfig(
                "openai",
                _env("OPENAI_BASE_URL", "https://api.openai.com"),
                _env("OPENAI_MODEL", "gpt-4o"),
                key,
                **openai_request_profile_from_env(),
            )
        return None

    def available_providers(self) -> list[str]:
        strict = (_env("EVOLUTION_PROVIDER_STRICT", "") or "").strip().lower() in {"1", "true", "yes", "on"}
        order = [self.primary_name] if strict else [self.primary_name, self.fallback_name]
        return [name for name in order if self._resolve(name) is not None]

    def generate(self, user: str, *, system: Optional[str] = None, max_tokens: int = 4096,
                 temperature: Optional[float] = None, provider: Optional[str] = None) -> LLMResponse:
        """Return a completion, trying primary then fallback with retries."""
        strict = (_env("EVOLUTION_PROVIDER_STRICT", "") or "").strip().lower() in {"1", "true", "yes", "on"}
        order = [provider] if provider else ([self.primary_name] if strict else [self.primary_name, self.fallback_name])
        temp = self.temperature if temperature is None else temperature
        errors: list[str] = []
        for name in order:
            config = self._resolve(name)
            if config is None:
                errors.append(f"{name}: no API key configured")
                continue
            caller = _CALLERS[name]
            for attempt in range(self.max_retries + 1):
                try:
                    return caller(config, system=system, user=user, max_tokens=max_tokens,
                                  temperature=temp, timeout=self.timeout)
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
                    # Never echo the exception URL/body verbatim (could contain the key).
                    errors.append(f"{name} attempt {attempt + 1}: {type(exc).__name__}")
                    if attempt < self.max_retries:
                        time.sleep(1.5 * (attempt + 1))
        raise LLMError("All LLM providers failed: " + "; ".join(errors))

    def generate_stream(
        self,
        user: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        provider: Optional[str] = None,
    ) -> Iterator[LLMStreamEvent]:
        """Stream a completion with retry/failover only before visible output."""

        strict = (_env("EVOLUTION_PROVIDER_STRICT", "") or "").strip().lower() in {"1", "true", "yes", "on"}
        order = [provider] if provider else ([self.primary_name] if strict else [self.primary_name, self.fallback_name])
        temp = self.temperature if temperature is None else temperature
        errors: list[str] = []
        for name in order:
            config = self._resolve(str(name or ""))
            if config is None:
                errors.append(f"{name}: no API key configured")
                continue
            for attempt in range(self.max_retries + 1):
                emitted_visible = False
                saw_done = False
                try:
                    for event in _stream_provider(
                        config,
                        system=system,
                        user=user,
                        max_tokens=max_tokens,
                        temperature=temp,
                        timeout=self.timeout,
                    ):
                        if event.kind in {"text_delta", "thinking_delta"} and event.text:
                            emitted_visible = True
                        if event.kind == "done":
                            saw_done = True
                        yield event
                    if saw_done:
                        return
                    raise ConnectionError("stream ended without a completion event")
                except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"{name} attempt {attempt + 1}: {type(exc).__name__}")
                    if emitted_visible:
                        raise LLMError("LLM stream interrupted after output started") from None
                    if attempt < self.max_retries:
                        time.sleep(1.5 * (attempt + 1))
        raise LLMError("All streaming LLM providers failed: " + "; ".join(errors))
