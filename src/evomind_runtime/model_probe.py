"""Authenticated reachability probe for each platform model provider (operator health only).

One minimal real call (max 8 output tokens) per configured provider default, using the same
transport, endpoint and key the runtime calls. An auth rejection (401/403) is a failure, never
"the HTTP layer answered". Results are cached so the 5-minute monitor costs at most one call per
provider per ``ttl`` seconds. No key, URL or response body is ever recorded.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
from pathlib import Path
from typing import Any, Callable

SCHEMA = "evomind.model_provider_probe.v1"
DEFAULT_TTL_SECONDS = 900
_LOCK = threading.Lock()


def _classify(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code in {401, 403}:
            return "auth_failed"
        return f"http_{exc.code}"
    if isinstance(exc, (TimeoutError, urllib.error.URLError, ConnectionError)):
        return "unreachable"
    return type(exc).__name__


def probe_once(transport, post: Callable[..., dict], timeout: float = 30.0) -> dict[str, Any]:
    from .model_router import served_identity_matches

    started = time.monotonic()
    url, headers, payload = transport.build([{"role": "user", "content": "ping"}], "Reply with one word.", [], 8, 0.0)
    payload.pop("tools", None)
    entry: dict[str, Any] = {"provider": transport.name, "model": transport.config.model}
    try:
        body = post(url, headers, payload, timeout)
        served = str(body.get("model") or "") if isinstance(body, dict) else ""
        entry.update(ok=True, state="ok", served_model=served,
                     identity_ok=bool(served) and served_identity_matches(transport.config.model, served))
    except BaseException as exc:  # noqa: BLE001 - every failure is a recorded state, never raised
        entry.update(ok=False, state=_classify(exc))
    entry["latency_ms"] = int((time.monotonic() - started) * 1000)
    return entry


def probe_providers(cache_path: Path, *, ttl: float = DEFAULT_TTL_SECONDS, force: bool = False,
                    transports: Callable[[], list] | None = None, post: Callable[..., dict] | None = None,
                    clock: Callable[[], float] = time.time) -> dict[str, Any]:
    """Cached report: ``ok`` only when every configured provider answered with its key accepted."""

    cache_path = Path(cache_path)
    with _LOCK:
        if not force:
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("schema") == SCHEMA and clock() - float(cached.get("checked_at", 0)) < ttl:
                    return {**cached, "cached": True}
            except (OSError, ValueError, TypeError):
                pass
        if transports is None:
            from .model_router import product_default_transports
            from research_os.agent.messaging import _resolve_transports

            def transports():
                return product_default_transports() or _resolve_transports()
        if post is None:
            from research_os.llm_client import _post_json as post
        providers = [probe_once(item, post) for item in transports() if str(getattr(item, "name", "")) not in {"", "none"}]
        report = {"schema": SCHEMA, "checked_at": clock(), "ok": bool(providers) and all(p["ok"] for p in providers),
                  "providers": providers}
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
            temporary.replace(cache_path)
        except OSError:
            pass
        return {**report, "cached": False}
