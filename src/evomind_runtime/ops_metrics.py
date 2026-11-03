"""t238 C7e O3: minimal in-process metrics (bounded memory, thread-safe, injectable clock).

The metric set is fixed: ``queue_depth, queue_wait_seconds, run_stalled_total, gpu_util,
hpc_probe_ok, disk_free_pct, api_p95, llm_error_rate`` (+ capacity gauges from
``ops_capacity``). Event samples (API latency, LLM call outcomes) live in fixed-size deques
trimmed to a sliding window, so memory never grows with traffic. Scheduler/HPC values come
from ``ops_providers`` and are ``None`` (omitted from the Prometheus text) until a provider
is registered.
"""
from __future__ import annotations

import math
import shutil
import threading
import time
from collections import deque
from typing import Any, Callable

WINDOW_SECONDS = 300.0
MAX_SAMPLES = 4096
METRIC_NAMES = ("queue_depth", "queue_wait_seconds", "run_stalled_total", "gpu_util", "hpc_probe_ok",
                "disk_free_pct", "api_p95", "llm_error_rate")
HELP = {
    "queue_depth": "Active (non-terminal) assistant runs plus GPU scheduler queue depth.",
    "queue_wait_seconds": "Oldest wait in the GPU scheduler queue.",
    "run_stalled_total": "Runs that reached the STALLED threshold since process start.",
    "gpu_util": "GPU utilisation percent reported by the scheduler provider.",
    "hpc_probe_ok": "1 when the last cached HPC probe succeeded, else 0.",
    "disk_free_pct": "Lowest free-space percent across the data and uploads volumes.",
    "api_p95": "95th percentile runtime API latency in seconds over the window.",
    "llm_error_rate": "Share of failed LLM calls over the window (0..1).",
    "runtime_db_bytes": "runtime.sqlite3 plus its WAL, bytes.",
    "uploads_bytes": "Bytes under the uploads directory.",
    "data_free_bytes": "Free bytes on the data volume.",
    "growth_bytes_per_day": "DB + uploads growth per day over the last 14 days.",
    "active_users_7d": "Distinct users that created a run in the last 7 days.",
    "growth_bytes_per_active_user_day": "Daily growth divided by active users.",
    "days_to_full": "Free bytes divided by daily growth.",
    "backup_last_failed": "1 when the last scheduled backup was skipped (free-space guard) or failed verification.",
    "backup_age_hours": "Hours since the newest verified scheduled backup.",
}


class Metrics:
    def __init__(self, *, clock: Callable[[], float] = time.monotonic, window_seconds: float = WINDOW_SECONDS,
                 max_samples: int = MAX_SAMPLES) -> None:
        self.clock = clock
        self.window = float(window_seconds)
        self._lock = threading.Lock()
        self._api: deque[tuple[float, float]] = deque(maxlen=max_samples)
        self._llm: deque[tuple[float, bool, str]] = deque(maxlen=max_samples)
        self._stalled = 0

    def _trim(self, samples: deque, now: float) -> None:
        cutoff = now - self.window
        while samples and samples[0][0] < cutoff:
            samples.popleft()

    def observe_api(self, seconds: float) -> None:
        with self._lock:
            now = self.clock()
            self._api.append((now, max(0.0, float(seconds))))
            self._trim(self._api, now)

    def record_llm(self, ok: bool, backend_id: str = "primary") -> None:
        with self._lock:
            now = self.clock()
            self._llm.append((now, bool(ok), str(backend_id)))
            self._trim(self._llm, now)

    def inc_stalled(self, count: int = 1) -> None:
        with self._lock:
            self._stalled += int(count)

    def api_p95(self) -> float | None:
        with self._lock:
            self._trim(self._api, self.clock())
            values = sorted(value for _t, value in self._api)
        if not values:
            return None
        return round(values[min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)], 4)

    def llm_error_rate(self, backend_id: str | None = None) -> float | None:
        with self._lock:
            self._trim(self._llm, self.clock())
            outcomes = [ok for _t, ok, backend in self._llm if backend_id is None or backend == backend_id]
        if not outcomes:
            return None
        return round(outcomes.count(False) / len(outcomes), 4)

    def llm_error_rate_prefix(self, prefix: str) -> float | None:
        """Error rate over every backend id equal to ``prefix`` or starting with ``prefix + ':'`` (FIX2: all primaries)."""
        with self._lock:
            self._trim(self._llm, self.clock())
            outcomes = [ok for _t, ok, backend in self._llm if backend == prefix or backend.startswith(prefix + ":")]
        if not outcomes:
            return None
        return round(outcomes.count(False) / len(outcomes), 4)

    def stalled_total(self) -> int:
        with self._lock:
            return self._stalled


METRICS = Metrics()


def _disk_free_pct(runtime, disk_usage) -> float | None:
    usage = disk_usage or getattr(runtime.assistant, "disk_usage", shutil.disk_usage)
    lowest = None
    for path in (runtime.runtime_root, runtime.assistant.upload_root):
        try:
            info = usage(path)
            total, free = int(getattr(info, "total", 0) or 0), int(info.free)
        except (OSError, ValueError, AttributeError):
            continue
        if total:
            pct = free * 100 / total
            lowest = pct if lowest is None else min(lowest, pct)
    return None if lowest is None else round(lowest, 2)


def collect(runtime=None, *, metrics: Metrics | None = None, disk_usage=None, capacity: str = "cached") -> dict[str, Any]:
    """One snapshot of the fixed metric set (``None`` = unknown / no provider).

    ``capacity="sample"`` (the monitor tick) refreshes the O4 watermarks, rate limited;
    ``"cached"`` (HTTP paths) only reads the last sample; ``"off"`` skips them.
    """
    from . import ops_providers

    metrics = metrics or METRICS
    gpu_queue = ops_providers.call("gpu_queue")
    gpu_queue = gpu_queue if isinstance(gpu_queue, dict) else {}
    hpc = ops_providers.call("hpc_probe")
    gpu_util = ops_providers.call("gpu_util")
    depth = None
    if runtime is not None:
        try:
            depth = int(runtime.store.queue_depth()["active_runs"])
        except Exception:  # noqa: BLE001 - a locked DB must not break the snapshot
            depth = None
    if gpu_queue.get("depth") is not None:
        depth = (depth or 0) + int(gpu_queue["depth"])
    snapshot = {
        "queue_depth": depth,
        "queue_wait_seconds": _float(gpu_queue.get("oldest_wait_seconds")),
        "run_stalled_total": metrics.stalled_total(),
        "gpu_util": _float(gpu_util),
        "hpc_probe_ok": (1 if hpc.get("ok") else 0) if isinstance(hpc, dict) else None,
        "disk_free_pct": _disk_free_pct(runtime, disk_usage) if runtime is not None else None,
        "api_p95": metrics.api_p95(),
        "llm_error_rate": metrics.llm_error_rate(),
    }
    if runtime is not None and capacity != "off":
        from .ops_capacity import CAPACITY_METRICS, tracker_for
        tracker = tracker_for(runtime)
        try:
            values = tracker.sample() if capacity == "sample" else tracker.cached()
        except Exception:  # noqa: BLE001
            values = None
        snapshot.update({name: (values or {}).get(name) for name in CAPACITY_METRICS})
    return snapshot


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def prometheus(snapshot: dict[str, Any], *, prefix: str = "evomind_") -> str:
    """Prometheus text exposition (0.0.4). Unknown values are omitted, never reported as 0."""
    lines: list[str] = []
    for name, value in snapshot.items():
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        kind = "counter" if name.endswith("_total") else "gauge"
        lines.append(f"# HELP {prefix}{name} {HELP.get(name, name)}")
        lines.append(f"# TYPE {prefix}{name} {kind}")
        lines.append(f"{prefix}{name} {value:g}" if isinstance(value, float) else f"{prefix}{name} {value}")
    return "\n".join(lines) + "\n"
