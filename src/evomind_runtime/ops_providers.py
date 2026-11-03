"""t238 C7e: None-safe provider registry for operational signals owned by other subsystems.

The GPU scheduler / lease work (C7c) and the LLM backend pool plug in here with one
`register(...)` call; until they do, every lookup returns ``None`` and callers keep the
pre-C7e behaviour. A provider must answer from local state only (cached probe results,
in-memory queues) — it is called from health and metric paths that must never trigger a
remote call (SSH, HPC, model gateway).

Known names and the value each provider returns:

- ``gpu_queue``    → ``{"depth": int, "oldest_wait_seconds": float}``
- ``hpc_probe``    → ``{"ok": bool, "state": str, "age_seconds": float}`` (last cached probe)
- ``gpu_util``     → ``float`` percent 0..100
- ``llm_backends`` → ``[{"id": str, "state": "closed"|"open"|"half_open", "ok": bool, ...}]``
- ``job_liveness`` → callable ``(run_id) -> {"state": "queued"|"running"|..., "heartbeat_age_seconds": float} | None``
"""
from __future__ import annotations

import threading
from typing import Any, Callable

KNOWN = frozenset({"gpu_queue", "hpc_probe", "gpu_util", "llm_backends", "job_liveness"})

_lock = threading.Lock()
_providers: dict[str, Callable[..., Any]] = {}


def register(name: str, provider: Callable[..., Any]) -> Callable[[], None]:
    """Install ``provider`` under ``name``; returns an unregister callable (used by tests)."""
    if name not in KNOWN:
        raise ValueError(f"unknown ops provider: {name}")
    with _lock:
        _providers[name] = provider

    def unregister() -> None:
        with _lock:
            if _providers.get(name) is provider:
                del _providers[name]
    return unregister


def registered(name: str) -> bool:
    with _lock:
        return name in _providers


def call(name: str, *args: Any) -> Any:
    """The provider's answer, or ``None`` when none is registered or it raised."""
    with _lock:
        provider = _providers.get(name)
    if provider is None:
        return None
    try:
        return provider(*args)
    except Exception:  # noqa: BLE001 - a broken provider must not break health/metrics
        return None
