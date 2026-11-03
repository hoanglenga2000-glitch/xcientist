"""t238 C7e A1: product-level LLM failover across equivalent backends.

Not the developer-side claude-failover proxy. This is the runtime's own selection between
backends that serve an *equivalent* model (same family/version), configured by the operator:

    EVOMIND_LLM_BACKENDS_FILE=<json>     (absent = today's single backend, behaviour unchanged)

    {"failure_threshold": 3, "open_seconds": 30, "first_byte_timeout_seconds": 20,
     "backends": [{"id": "backup_1", "label": "备用线路 1", "provider": "anthropic",
                   "base_url": "https://...", "model": "claude-sonnet-5-5",
                   "api_key_env": "EVOMIND_BACKUP1_API_KEY",          # or "api_key_file": "<path>"
                   "equivalent_to": ["anthropic:claude-sonnet-5-5"]}]}

Keys are read from the named environment variable / file at call time and never logged. A
backend whose key is missing is skipped (state ``unconfigured``).

Rules (see docs/LLM_FAILOVER_C7E.md):
- implicit/default selection: on 5xx / 429 / timeout / network failure the next healthy
  equivalent backend takes the same request. A per-backend circuit breaker opens after
  ``failure_threshold`` consecutive failures and skips that backend for ``open_seconds``; a
  hung primary is bounded by ``first_byte_timeout_seconds`` (stream reader), so the switch
  lands within 30 s.
- explicit user selection: never silently switched. Only a ``fallback_model_id`` the user chose
  for this run is tried, and the switch is recorded with the visible notice
  「已切换到备选模型「X」」.
- auth / identity / contract / request errors never fail over (they would fail the same way or
  must reach the user).
- every attempt feeds ``ops_metrics.METRICS.record_llm`` → ``llm_error_rate`` (O3).
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import threading
import time
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

PRIMARY_ID = "primary"
# FIX1 M1: each primary model has its own breaker and metric label, `primary:<provider>:<model>`.
# Cardinality is bounded: past MAX_PRIMARY_KEYS distinct primaries the rest share `primary:other`.
MAX_PRIMARY_KEYS = 16
_KEY_PART = re.compile(r"[^a-z0-9._-]+")


def primary_key(provider: str, model: str) -> str:
    from .model_router import _LT4NET_ALIASES

    model = _LT4NET_ALIASES.get(str(model or ""), str(model or ""))
    parts = [_KEY_PART.sub("_", str(value or "").lower())[:48] or "unknown" for value in (provider, model)]
    return f"{PRIMARY_ID}:{parts[0]}:{parts[1]}"


def public_backend_id(backend_id: str) -> str:
    """`primary:anthropic:claude-sonnet-5-5` → `primary_anthropic_claude_sonnet_5_5` (deep-health allow-list)."""
    return re.sub(r"[^a-z0-9_]+", "_", backend_id.lower()).strip("_")[:32] or "backend"
DEFAULT_FAILURE_THRESHOLD = 3
DEFAULT_OPEN_SECONDS = 30.0
DEFAULT_FIRST_BYTE_TIMEOUT = 20.0
FAILOVER_NOTICE = "已切换到备选模型「{label}」"
FAILOVER_NOTICE_NO_LABEL = "已切换到备选模型"  # no 「」 fragment when the operator gave the backend no label
_ID = re.compile(r"^[a-z0-9_]{1,32}$")
_HTTP_IN_TEXT = re.compile(r"HTTPError (\d{3})")


# ---------------------------------------------------------------- error classification


def public_code(exc: BaseException) -> str:
    """A safe, stable code for one model failure (never a URL, key or body)."""
    from .model_transport import ModelTransportError

    if isinstance(exc, ModelTransportError):
        return exc.code
    if isinstance(exc, urllib.error.HTTPError):
        return f"http_{exc.code}"
    text = str(exc)
    if isinstance(exc, ValueError) and re.fullmatch(r"model_[a-z0-9_]{1,80}", text):
        return text
    if isinstance(exc, TimeoutError) or "TimeoutError" in text or "timed out" in text:
        return "model_timeout"
    statuses = _HTTP_IN_TEXT.findall(text)
    if statuses:
        return f"http_{statuses[-1]}"
    return type(exc).__name__


def category_for_code(code: str) -> str:
    """``rate_limited | unavailable | timeout | auth | identity | request | unknown`` for one public code."""
    code = str(code or "")
    status = re.search(r"http_(\d{3})$", code)
    if status:
        value = int(status.group(1))
        if value == 429:
            return "rate_limited"
        if value in (401, 403):
            return "auth"
        if value == 408:
            return "timeout"
        return "unavailable" if value >= 500 else "request"
    if code in {"model_timeout", "model_total_deadline_exceeded", "model_first_byte_timeout", "model_transport_timeout", "TimeoutError"}:
        return "timeout"
    if code in {"model_network_error", "model_transport_failed", "empty_model_response", "model_response_protocol_error",
                "upstream_error", "server_error", "overloaded_error", "provider_error", "model_unavailable", "ConnectionError"}:
        return "unavailable"
    if code in {"rate_limit_error"}:
        return "rate_limited"
    if code in {"model_profile_disabled", "model_address_invalid", "authentication_error", "permission_error"}:
        return "auth"
    if "identity" in code:
        return "identity"
    if code.startswith("model_contract") or code in {"model_route_configuration_drift", "model_request_too_large"}:
        return "request"
    return "unknown"


def category(exc: BaseException) -> str:
    return category_for_code(public_code(exc))


def failover_eligible(exc: BaseException) -> bool:
    from research_os.llm_client import LLMError

    kind = category(exc)
    if kind in {"rate_limited", "unavailable", "timeout"}:
        return True
    # A bare LLMError from the multi-transport client already exhausted its retries on
    # network/HTTP errors without telling which; treat it as an outage unless it named an auth fault.
    return kind == "unknown" and type(exc) is LLMError


def _is_model_error(exc: BaseException) -> bool:
    from research_os.llm_client import LLMError

    return isinstance(exc, (LLMError, TimeoutError, ConnectionError)) or (
        isinstance(exc, ValueError) and str(exc).startswith("model_"))


# ---------------------------------------------------------------- circuit breaker


def failover_notice(label: str) -> str:
    label = str(label or "").strip()
    return FAILOVER_NOTICE.format(label=label) if label else FAILOVER_NOTICE_NO_LABEL


class CircuitBreaker:
    """closed → open after N consecutive failures → half_open after ``open_seconds`` → one trial."""

    def __init__(self, *, failure_threshold: int = DEFAULT_FAILURE_THRESHOLD, open_seconds: float = DEFAULT_OPEN_SECONDS,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.failure_threshold = max(1, int(failure_threshold))
        self.open_seconds = float(open_seconds)
        self.clock = clock
        self._lock = threading.Lock()
        self._failures = 0
        self._opened_at: float | None = None
        self._trial = False

    @property
    def state(self) -> str:
        with self._lock:
            return self._state()

    def _state(self) -> str:
        if self._opened_at is None:
            return "closed"
        return "half_open" if self.clock() - self._opened_at >= self.open_seconds else "open"

    def allow(self) -> bool:
        with self._lock:
            state = self._state()
            if state == "closed":
                return True
            if state == "half_open" and not self._trial:
                self._trial = True
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._failures, self._opened_at, self._trial = 0, None, False

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state() == "half_open" or self._failures >= self.failure_threshold:
                self._opened_at, self._trial = self.clock(), False


# ---------------------------------------------------------------- configuration


@dataclass(frozen=True)
class BackendSpec:
    id: str
    provider: str
    base_url: str
    model: str
    equivalent_to: tuple[str, ...]
    label: str = ""
    api_key_env: str = ""
    api_key_file: str = ""

    def api_key(self) -> str:
        if self.api_key_env:
            return str(os.environ.get(self.api_key_env) or "").strip()
        if self.api_key_file:
            try:
                return Path(self.api_key_file).read_text(encoding="utf-8").strip()
            except OSError:
                return ""
        return ""


@dataclass
class BackendConfig:
    backends: tuple[BackendSpec, ...] = ()
    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD
    open_seconds: float = DEFAULT_OPEN_SECONDS
    first_byte_timeout_seconds: float = DEFAULT_FIRST_BYTE_TIMEOUT
    errors: list[str] = field(default_factory=list)


def load_config(path: str | os.PathLike | None = None) -> BackendConfig:
    """Parse the backends file; any invalid entry is dropped (never crashes model calls)."""
    path = path if path is not None else os.environ.get("EVOMIND_LLM_BACKENDS_FILE") or ""
    if not path:
        return BackendConfig()
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return BackendConfig(errors=["config_unreadable"])
    config = BackendConfig(
        failure_threshold=int(_bounded(raw.get("failure_threshold"), DEFAULT_FAILURE_THRESHOLD, 1, 20)),
        open_seconds=_bounded(raw.get("open_seconds"), DEFAULT_OPEN_SECONDS, 5, 600),
        first_byte_timeout_seconds=_bounded(raw.get("first_byte_timeout_seconds"), DEFAULT_FIRST_BYTE_TIMEOUT, 5, 25),
    )
    backends, seen = [], {PRIMARY_ID}
    for item in raw.get("backends") or []:
        try:
            spec = BackendSpec(
                id=str(item["id"]), provider=str(item["provider"]), base_url=str(item["base_url"]), model=str(item["model"]),
                equivalent_to=tuple(str(value) for value in item.get("equivalent_to") or ()), label=str(item.get("label") or "")[:40],
                api_key_env=str(item.get("api_key_env") or ""), api_key_file=str(item.get("api_key_file") or ""))
        except (KeyError, TypeError, AttributeError):
            config.errors.append("backend_invalid")
            continue
        if (not _ID.fullmatch(spec.id) or spec.id in seen or spec.provider not in {"anthropic", "openai", "deepseek"}
                or not spec.base_url.startswith("https://") or not spec.model or not spec.equivalent_to
                or not (spec.api_key_env or spec.api_key_file)):
            config.errors.append(f"backend_invalid:{spec.id if _ID.fullmatch(spec.id) else '?'}")
            continue
        seen.add(spec.id)
        backends.append(spec)
    config.backends = tuple(backends)
    return config


def _bounded(value: Any, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(high, max(low, number))


def _normalize(model_id: str) -> str:
    from .model_router import _LT4NET_ALIASES

    provider, _, model = str(model_id).partition(":")
    if not model:
        provider, model = "", provider
    model = _LT4NET_ALIASES.get(model, model)
    return f"{provider}:{model}" if provider else model


def _matches(spec: BackendSpec, provider: str, model: str) -> bool:
    wanted = {_normalize(f"{provider}:{model}"), _normalize(model)}
    return any(_normalize(item) in wanted for item in spec.equivalent_to)


# ---------------------------------------------------------------- pool


class BackendPool:
    """Process-wide breakers (one per backend id) and the loaded config."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic, loader: Callable[[], BackendConfig] = load_config) -> None:
        self.clock = clock
        self.loader = loader
        self._lock = threading.Lock()
        self._breakers: dict[str, CircuitBreaker] = {}
        self._primaries: list[str] = []
        self._config: BackendConfig | None = None
        self._config_key: str | None = None

    def primary_id(self, provider: str, model: str) -> str:
        """The breaker/metric key for one primary model (bounded set, see MAX_PRIMARY_KEYS)."""
        key = primary_key(provider, model)
        with self._lock:
            if key in self._primaries:
                return key
            if len(self._primaries) >= MAX_PRIMARY_KEYS:
                return f"{PRIMARY_ID}:other"
            self._primaries.append(key)
            return key

    def config(self) -> BackendConfig:
        key = os.environ.get("EVOMIND_LLM_BACKENDS_FILE") or ""
        with self._lock:
            if self._config is None or key != self._config_key:
                self._config, self._config_key = self.loader(), key
                self._breakers = {}
            return self._config

    def breaker(self, backend_id: str) -> CircuitBreaker:
        config = self.config()
        with self._lock:
            breaker = self._breakers.get(backend_id)
            if breaker is None:
                breaker = CircuitBreaker(failure_threshold=config.failure_threshold, open_seconds=config.open_seconds, clock=self.clock)
                self._breakers[backend_id] = breaker
            return breaker

    def alternates_for(self, provider: str, model: str, base_url: str = "") -> list[BackendSpec]:
        """Configured equivalents of the primary with a key present (never the primary itself)."""
        return [spec for spec in self.config().backends
                if _matches(spec, provider, model) and spec.api_key() and not (spec.base_url == base_url and spec.model == model)]

    def states(self) -> list[dict[str, Any]]:
        """For deep health / ops provider: id, breaker state, ok, error rate. No hosts, no keys.

        FIX2: the per-model primary breakers (FIX1 M1) are aggregated into ONE ``primary`` row:
        the worst state across them (open > half_open > closed), error rate over all primary calls.
        Only when the operator configured alternates are the per-model primaries listed as well
        (bounded by MAX_PRIMARY_KEYS), so an operator can see which model's breaker tripped.
        """
        from .ops_metrics import METRICS

        config = self.config()
        with self._lock:
            primaries = list(self._primaries)
        primary_states = [self.breaker(key).state for key in primaries] or [self.breaker(PRIMARY_ID).state]
        worst = max(primary_states, key=_STATE_RANK.__getitem__)
        rows = [{"id": PRIMARY_ID, "state": worst, "ok": worst != "open",
                 "error_rate": METRICS.llm_error_rate_prefix(PRIMARY_ID) if primaries else METRICS.llm_error_rate(PRIMARY_ID)}]
        if config.backends:
            for key in primaries:
                state = self.breaker(key).state
                rows.append({"id": public_backend_id(key), "state": state, "ok": state != "open",
                             "error_rate": METRICS.llm_error_rate(key)})
        for spec in config.backends:
            configured = bool(spec.api_key())
            state = self.breaker(spec.id).state if configured else "unconfigured"
            rows.append({"id": spec.id, "state": state, "ok": configured and state != "open",
                         "error_rate": METRICS.llm_error_rate(spec.id)})
        return rows

    def reset(self) -> None:
        """Forget breakers, known primaries and the loaded config (tests; never called by the product)."""
        with self._lock:
            self._breakers, self._primaries, self._config, self._config_key = {}, [], None, None


_STATE_RANK = {"closed": 0, "half_open": 1, "open": 2}


POOL = BackendPool()


def ensure_ops_provider() -> None:
    """Publish backend states to deep health (idempotent; called from ``ops_checks`` and the runtime hook)."""
    from . import ops_providers

    if not ops_providers.registered("llm_backends"):
        ops_providers.register("llm_backends", lambda: globals()["POOL"].states())


def transport_for(spec: BackendSpec):
    """A fresh research_os transport for one configured backend (key read now, never stored elsewhere)."""
    from research_os.agent.messaging import AnthropicTransport, DeepSeekTransport, OpenAITransport, ProviderConfig

    kind = {"anthropic": AnthropicTransport, "openai": OpenAITransport, "deepseek": DeepSeekTransport}[spec.provider]
    return kind(ProviderConfig(spec.provider, spec.base_url, spec.model, spec.api_key()))


# ---------------------------------------------------------------- client


@dataclass
class Candidate:
    backend_id: str
    label: str
    client: Any


class FailoverClient:
    """Wraps the run's model client. With no alternates it only records metrics/breaker state
    and re-raises unchanged (today's behaviour). Attribute reads fall through to the primary so
    contract binding, thinking flags and token limits keep seeing the client they expect."""

    def __init__(self, primary: Any, alternates: list[Candidate] | None = None, *, pool: BackendPool | None = None,
                 primary_id: str = PRIMARY_ID, explicit: bool = False,
                 on_switch: Callable[[dict], None] | None = None, first_byte_timeout: float | None = None,
                 budget_seconds: float = 30.0, clock: Callable[[], float] = time.monotonic) -> None:
        self._primary = Candidate(primary_id, "", primary)
        self._alternates = list(alternates or [])
        self._pool = pool or globals()["POOL"]
        self._explicit = explicit
        self._on_switch = on_switch
        self._first_byte_timeout = first_byte_timeout
        self._budget = float(budget_seconds)
        self._clock = clock
        self._switched_to: str | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["_primary"].client, name)

    @property
    def model(self) -> str:
        return self._primary.client.model

    def is_available(self) -> bool:
        return self._primary.client.is_available() or any(item.client.is_available() for item in self._alternates)

    def _order(self) -> list[Candidate]:
        everyone = [self._primary, *self._alternates]
        if not self._alternates:
            return everyone
        allowed = [item for item in everyone if self._pool.breaker(item.backend_id).allow()]
        return allowed or everyone  # all open: try in order rather than fail without a request

    def _attempt(self, candidate: Candidate, guarded: bool, args: tuple, kwargs: dict):
        from . import model_transport

        context = (model_transport.first_byte_timeout(self._first_byte_timeout)
                   if guarded and self._first_byte_timeout else contextlib.nullcontext())
        with context:
            return candidate.client.send(*args, **kwargs)

    def send(self, *args: Any, **kwargs: Any):
        from .ops_metrics import METRICS

        started = self._clock()
        order = self._order()
        last: BaseException | None = None
        for index, candidate in enumerate(order):
            breaker = self._pool.breaker(candidate.backend_id)
            has_next = index + 1 < len(order)
            try:
                turn = self._attempt(candidate, has_next, args, kwargs)
            except BaseException as exc:
                if not _is_model_error(exc):
                    raise  # pause / cancel / programming errors are not model outcomes
                METRICS.record_llm(False, candidate.backend_id)
                eligible = failover_eligible(exc)
                if eligible:
                    breaker.record_failure()
                last = exc
                if not (eligible and has_next and self._clock() - started < self._budget):
                    raise
                continue
            METRICS.record_llm(True, candidate.backend_id)
            breaker.record_success()
            if candidate is not self._primary and self._switched_to != candidate.backend_id:
                self._switched_to = candidate.backend_id
                if self._on_switch is not None:
                    self._on_switch({"to": candidate.backend_id, "label": candidate.label, "explicit": self._explicit,
                                     "reason": public_code(last) if last else "breaker_open",
                                     "notice": failover_notice(candidate.label)})
            return turn
        raise last if last else RuntimeError("no_model_candidate")


def build_failover(primary: Any, *, provider: str, model: str, base_url: str = "",
                   explicit_fallback: tuple[str, str, Any] | None = None, explicit: bool = False,
                   make_client: Callable[[Any], Any] | None = None, on_switch: Callable[[dict], None] | None = None,
                   pool: BackendPool | None = None) -> FailoverClient:
    """The run's client wrapped for A1/O3.

    - ``explicit`` without ``explicit_fallback``: no alternates (no silent fallback).
    - ``explicit_fallback=(id, label, client)``: only the user's own pre-selected alternate.
    - implicit: configured equivalent backends, built with ``make_client(transport)``.
    - a governed client with a pinned execution contract never fails over.
    """
    pool = pool or globals()["POOL"]
    alternates: list[Candidate] = []
    if getattr(primary, "contract", None):
        pass
    elif explicit_fallback is not None:
        alternates.append(Candidate(*explicit_fallback))
    elif not explicit and make_client is not None:
        for spec in pool.alternates_for(provider, model, base_url):
            try:
                alternates.append(Candidate(spec.id, spec.label or _label(spec.model), make_client(transport_for(spec))))
            except Exception:  # noqa: BLE001 - a broken alternate never blocks the primary
                continue
    config = pool.config()
    return FailoverClient(primary, alternates, pool=pool, explicit=explicit, on_switch=on_switch,
                          primary_id=pool.primary_id(provider, model),
                          first_byte_timeout=config.first_byte_timeout_seconds if alternates else None)


def _label(model: str) -> str:
    from .model_router import model_display_name

    return model_display_name(model)


# ---------------------------------------------------------------- runtime hook


def _primary_target(client: Any) -> tuple[str, str, str]:
    for candidate in (getattr(client, "transport", None),):
        if candidate is not None:
            return str(candidate.name or ""), str(candidate.config.model or ""), str(candidate.config.base_url or "")
    inner = getattr(client, "client", client)
    try:
        transport = inner._resolve_transports()[0]
    except Exception:  # noqa: BLE001 - personal clients and fakes may not expose transports
        return "", str(getattr(client, "model", "") or ""), ""
    return str(transport.name or ""), str(transport.config.model or ""), str(transport.config.base_url or "")


def visible_fallback_model_ids() -> set[str]:
    """Platform ids a user may pick as fallback: listed in the picker (hidden ids excluded)."""
    from .model_router import available_platform_models

    return {item["id"] for item in available_platform_models()}


def wrap_run_client(runtime, session_id: str, client: Any, *, explicit: bool, reasoning_effort: str = "",
                    make_client: Callable[[Any], Any] | None = None) -> FailoverClient:
    """Called once per model turn loop from ``AgentRuntime.message`` (runtime.py hook)."""
    ensure_ops_provider()
    from .model_router import apply_reasoning_effort, model_display_name, platform_transport_for, require_served_identity

    session = runtime.get_session(session_id) or {}
    metadata = session.get("metadata") if isinstance(session.get("metadata"), dict) else {}
    provider, model, base_url = _primary_target(client)
    explicit_fallback = None
    fallback_id = str(metadata.get("fallback_model_id") or "")
    if explicit and fallback_id and make_client is not None and fallback_id in visible_fallback_model_ids():
        transport = platform_transport_for(fallback_id)
        if transport is not None:
            try:
                fallback_client = make_client(require_served_identity(apply_reasoning_effort(transport, reasoning_effort)))
                explicit_fallback = ("fallback_" + re.sub(r"[^a-z0-9_]", "_", fallback_id.lower())[:22], model_display_name(transport.config.model), fallback_client)
            except Exception:  # noqa: BLE001
                explicit_fallback = None

    def on_switch(info: dict) -> None:
        record = {"to": info["to"], "label": info["label"], "explicit": info["explicit"], "reason": info["reason"],
                  "notice": info["notice"], **({"fallback_model_id": fallback_id} if info["explicit"] else {})}
        runtime.store.append_event(session_id, "model.failover", record)
        current = dict((runtime.get_session(session_id) or {}).get("metadata") or {})
        current["model_failover"] = record
        runtime.store.update_session(session_id, metadata_json=current)

    return build_failover(client, provider=provider, model=model, base_url=base_url, explicit=explicit,
                          explicit_fallback=explicit_fallback, make_client=None if explicit else make_client,
                          on_switch=on_switch)
