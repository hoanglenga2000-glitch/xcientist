"""t238 C7b O2: operator-only deep health. Each check has its own timeout; the total stays < 1 s.

A slow or failing check yields `status: degraded` naming the component, never a hang. Only volume
labels (`data`, `uploads`) are reported — no absolute host paths, no secrets.
"""
from __future__ import annotations

import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

CHECK_TIMEOUT_SECONDS = 0.25
TOTAL_BUDGET_SECONDS = 0.8
DEFAULT_DISK_LOW_WATER_PCT = 15.0
_STARTED = time.monotonic()
# Shared, small pool: a hung check keeps at most its own worker; later calls still answer in time.
_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="deep-health")
_GATE = threading.BoundedSemaphore(4)


def _low_water_pct() -> float:
    try:
        value = float(os.environ.get("EVOMIND_DISK_LOW_WATER_PCT", "") or DEFAULT_DISK_LOW_WATER_PCT)
    except ValueError:
        return DEFAULT_DISK_LOW_WATER_PCT
    return min(100.0, max(0.0, value))


def default_checks(runtime, disk_usage: Callable[[Any], Any] | None = None) -> dict[str, Callable[[], dict]]:
    usage = disk_usage or getattr(runtime.assistant, "disk_usage", shutil.disk_usage)

    def sqlite() -> dict:
        started = time.monotonic()
        runtime.store.health_probe()
        return {"ok": True, "write_probe": "rolled_back", "ms": round((time.monotonic() - started) * 1000, 1)}

    def disk() -> dict:
        percent = _low_water_pct()
        volumes = {}
        for label, path in (("data", runtime.runtime_root), ("uploads", runtime.assistant.upload_root)):
            info = usage(path)
            total, free = int(getattr(info, "total", 0) or 0), int(info.free)
            volumes[label] = {"total_bytes": total, "free_bytes": free,
                              "free_pct": round(free * 100 / total, 2) if total else None,
                              "low_water": bool(total and percent > 0 and free * 100 < percent * total)}
        return {"ok": not any(volume["low_water"] for volume in volumes.values()), "low_water_pct": percent, "volumes": volumes}

    def queue() -> dict:
        depth = runtime.store.queue_depth()
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=runtime.assistant.COMPLETING_STALE_SECONDS)).isoformat()
        stale = sum(1 for row in runtime.store.list_uploads_with_status("completing") if str(row.get("updated_at") or "") < cutoff)
        return {"ok": True, **depth, "stale_claims": stale, **gpu_queue_fields()}

    def gpu_broker() -> dict:
        # t238 C7c ADDENDUM 2: local state only (no SSH inside the 0.25 s budget); live status is /v1/gpu/pool.
        from .gpu_broker_link import local_state
        return {"ok": True, **local_state(runtime.runtime_root)}

    return {"sqlite": sqlite, "disk": disk, "queue": queue, "gpu_broker": gpu_broker}


# ---------------------------------------------------------------- t238 C7e O2
# Provider-backed components. Values are re-shaped through an allow-list so a provider can
# never leak a host, path or secret into the report. Missing providers = "not_configured"
# (healthy): the runtime never starts a remote probe from here.
_STATE_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")


def _state(value: Any, default: str = "unknown") -> str:
    text = str(value or "").strip().lower()[:32]
    return text if text and set(text) <= _STATE_CHARS else default


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 3) if number == number and abs(number) != float("inf") else None


def gpu_queue_fields() -> dict:
    """Extra queue fields from the GPU scheduler provider; ``{}`` until one is registered."""
    from . import ops_providers
    value = ops_providers.call("gpu_queue")
    if not isinstance(value, dict):
        return {}
    return {"gpu_queue_depth": int(_number(value.get("depth")) or 0),
            "gpu_queue_oldest_wait_seconds": _number(value.get("oldest_wait_seconds"))}


def ops_checks(runtime=None) -> dict[str, Callable[[], dict]]:
    from . import ops_providers

    def hpc() -> dict:
        value = ops_providers.call("hpc_probe")
        if not isinstance(value, dict):
            return {"ok": True, "state": "not_configured"}
        return {"ok": bool(value.get("ok")), "state": _state(value.get("state")), "age_seconds": _number(value.get("age_seconds"))}

    def llm() -> dict:
        from .llm_backends import ensure_ops_provider
        ensure_ops_provider()  # t238 C7e A1: the runtime's own backend pool is always reported
        value = ops_providers.call("llm_backends")
        if not isinstance(value, list) or not value:
            return {"ok": True, "state": "not_configured", "backends": []}
        backends = [{"id": _state(item.get("id"), "backend"), "state": _state(item.get("state")), "ok": bool(item.get("ok")),
                     "error_rate": _number(item.get("error_rate"))} for item in value if isinstance(item, dict)]
        # Degraded only when no backend can take traffic; one open breaker with a healthy
        # alternate is reported per backend but keeps the component ok.
        return {"ok": any(item["ok"] for item in backends), "backends": backends}

    checks = {"hpc": hpc, "llm": llm}
    if runtime is not None:
        from .ops_capacity import health_component
        checks["capacity"] = lambda: health_component(runtime)  # t238 C7e O4 (cached sample only)
    return checks


def deep_health(checks: dict[str, Callable[[], dict]], *, version: str, timeout: float = CHECK_TIMEOUT_SECONDS,
                budget: float = TOTAL_BUDGET_SECONDS, clock: Callable[[], float] = time.monotonic) -> dict:
    started = clock()
    components: dict[str, dict] = {}
    if not _GATE.acquire(blocking=False):
        return {"status": "degraded", "failed": ["busy"], "components": {}, "version": version,
                "uptime_seconds": int(time.monotonic() - _STARTED)}
    try:
        futures = {name: _POOL.submit(check) for name, check in checks.items()}
        for name, future in futures.items():
            remaining = max(0.0, min(timeout, budget - (clock() - started)))
            try:
                result = future.result(timeout=remaining)
                components[name] = result if isinstance(result, dict) else {"ok": bool(result)}
            except FutureTimeout:
                future.cancel()
                components[name] = {"ok": False, "error": "timeout"}
            except Exception as exc:  # noqa: BLE001 - report the class only, never paths or messages
                components[name] = {"ok": False, "error": type(exc).__name__}
    finally:
        _GATE.release()
    failed = [name for name, value in components.items() if not value.get("ok")]
    return {"status": "degraded" if failed else "ok", "failed": failed, "components": components, "version": version,
            "uptime_seconds": int(time.monotonic() - _STARTED), "elapsed_ms": round((clock() - started) * 1000, 1)}
