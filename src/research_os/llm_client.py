"""Unified multi-provider LLM client for the evolution engine.

Supports Anthropic-native ``/v1/messages`` plus DeepSeek and OpenAI
``/v1/chat/completions`` endpoints with truthful provider attribution.

Design rules (mirror gpu_credentials.py):
  * Credentials come only from the environment; never hardcoded, never logged.
  * Standard-library HTTP only (urllib) so the same module runs on the GPU box
    without extra pip installs.
  * A single ``generate`` entry point with automatic primary->fallback failover
    and bounded retries, so callers never see a transient gateway error.
"""
from __future__ import annotations

import ipaddress
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
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

    def __repr__(self) -> str:  # keep tokens visible, never echo the prompt/keys
        return (
            f"LLMResponse(provider={self.provider!r}, model={self.model!r}, "
            f"in={self.input_tokens}, out={self.output_tokens}, cache_read={self.cache_read_tokens})"
        )


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    model: str
    api_key: str = field(repr=False)


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


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    parsed = urllib.parse.urlsplit(url)
    hostname = parsed.hostname
    is_loopback = hostname == "localhost"
    if hostname and not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            is_loopback = False
    if (
        parsed.scheme not in {"http", "https"}
        or not hostname
        or parsed.username
        or parsed.password
        or (parsed.scheme == "http" and not is_loopback)
    ):
        raise ValueError(
            "LLM endpoint must be an http(s) URL with a hostname and no embedded credentials; "
            "remote endpoints require https"
        )
    data = json.dumps(payload).encode("utf-8")
    # Some gateways (Cloudflare, error 1010) reject urllib's default UA; send a normal one.
    headers = {"User-Agent": "research-os-evolution/1.0", **headers}
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310 - scheme validated above
        return json.loads(response.read().decode("utf-8"))


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


def _call_deepseek(config: ProviderConfig, *, system: Optional[str], user: str,
                   max_tokens: int, temperature: float, timeout: int) -> LLMResponse:
    url = config.base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {config.api_key}", "content-type": "application/json"}
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    payload = {"model": config.model, "max_tokens": max_tokens, "temperature": temperature, "messages": messages}
    body = _post_json(url, headers, payload, timeout)
    choice = (body.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content", "")
    usage = body.get("usage", {}) or {}
    details = usage.get("prompt_tokens_details", {}) or {}
    return LLMResponse(
        text=text, provider="deepseek", model=body.get("model", config.model),
        input_tokens=int(usage.get("prompt_tokens", 0) or 0),
        output_tokens=int(usage.get("completion_tokens", 0) or 0),
        cache_read_tokens=int(details.get("cached_tokens", 0) or 0),
        raw_usage=usage,
    )


def _call_openai(config: ProviderConfig, **kwargs: Any) -> LLMResponse:
    response = _call_deepseek(config, **kwargs)
    response.provider = "openai"
    return response


_CALLERS: dict[str, Callable[..., LLMResponse]] = {
    "anthropic": _call_anthropic,
    "deepseek": _call_deepseek,
    "openai": _call_openai,
}


class LLMClient:
    """Dual-backend client with automatic failover and bounded retries.

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
            )
        return None

    def available_providers(self) -> list[str]:
        order = [self.primary_name, self.fallback_name]
        return [name for name in order if self._resolve(name) is not None]

    def generate(self, user: str, *, system: Optional[str] = None, max_tokens: int = 4096,
                 temperature: Optional[float] = None, provider: Optional[str] = None) -> LLMResponse:
        """Return a completion, trying primary then fallback with retries."""
        order = [provider] if provider else [self.primary_name, self.fallback_name]
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



