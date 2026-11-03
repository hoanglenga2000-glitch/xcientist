"""t238 C7c ADDENDUM 1: configurable VRAM budget for training on the shared A800 (job 98270).

Nothing here hardcodes how much memory is free: admission uses the live probe's
``memory.free`` minus a reserve, optionally capped by an operator budget. When the owner
frees memory, larger jobs become admissible with no code change.

Config (re-read on every admission, so edits apply without a redeploy):

* file ``<runtime>/config/gpu-vram.json`` (override path: ``EVOMIND_GPU_VRAM_CONFIG``)
* env overrides per key: ``EVOMIND_GPU_VRAM_RESERVE_MB``, ``EVOMIND_GPU_VRAM_BUDGET_MB``,
  ``EVOMIND_JOB_DEFAULT_VRAM_MB``, ``EVOMIND_JOB_MAX_VRAM_MB`` (env wins over the file)

Keys: ``gpu_vram_reserve_mb`` (headroom never given to training), ``gpu_vram_budget_mb``
(hard cap for all tenant training combined; unset = live probe minus reserve),
``job_default_vram_mb``, ``job_max_vram_mb`` (per-job ceiling) and ``tenant_overrides``
(``{tenant_id: {"job_default_vram_mb": …, "job_max_vram_mb": …}}``).

ADDENDUM 2 adds ``gpu_broker_enabled`` (default false) and ``gpu_broker_config_path`` (env
``EVOMIND_GPU_BROKER_ENABLED`` / ``EVOMIND_GPU_BROKER_CONFIG``) and raises the job defaults
(4096 / 40960 MiB) because the owner frees the A800 for training on demand.

Admission rule: grant only if ``request_mb <= min(job_max, budget or inf, probe_free - reserve)``.
A request above ``min(job_max, budget)`` is refused at once (it can never fit); a request that
only exceeds today's free memory stays queued. An unavailable probe fails closed (queued).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

CONFIG_NAME = "gpu-vram.json"
DEFAULTS = {
    "gpu_vram_reserve_mb": 1024,
    "gpu_vram_budget_mb": None,
    "job_default_vram_mb": 4096,
    "job_max_vram_mb": 40960,
}
BROKER_DEFAULTS = {"gpu_broker_enabled": False, "gpu_broker_config_path": None, "gpu_broker_stop_estimate_seconds": 60}
_ENV = {
    "gpu_vram_reserve_mb": "EVOMIND_GPU_VRAM_RESERVE_MB",
    "gpu_vram_budget_mb": "EVOMIND_GPU_VRAM_BUDGET_MB",
    "job_default_vram_mb": "EVOMIND_JOB_DEFAULT_VRAM_MB",
    "job_max_vram_mb": "EVOMIND_JOB_MAX_VRAM_MB",
}
_BROKER_ENV = {"gpu_broker_enabled": "EVOMIND_GPU_BROKER_ENABLED", "gpu_broker_config_path": "EVOMIND_GPU_BROKER_CONFIG"}
_TENANT_KEYS = {"job_default_vram_mb", "job_max_vram_mb"}
MAX_MB = 1 << 20  # sanity bound (1 TiB)


class VramConfigError(ValueError):
    """Invalid config: admission fails closed (the job stays queued, operator sees the code)."""


def _mb(value: Any, key: str, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_MB:
        raise VramConfigError(f"{key}_invalid")
    return value


@dataclass(frozen=True)
class VramConfig:
    gpu_vram_reserve_mb: int = DEFAULTS["gpu_vram_reserve_mb"]
    gpu_vram_budget_mb: int | None = DEFAULTS["gpu_vram_budget_mb"]
    job_default_vram_mb: int = DEFAULTS["job_default_vram_mb"]
    job_max_vram_mb: int = DEFAULTS["job_max_vram_mb"]
    tenant_overrides: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    gpu_broker_enabled: bool = False
    gpu_broker_config_path: str | None = None
    gpu_broker_stop_estimate_seconds: int = 60

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "VramConfig":
        raw = dict(raw or {})
        unknown = set(raw) - set(DEFAULTS) - set(BROKER_DEFAULTS) - {"tenant_overrides", "_comment"}
        if unknown:
            raise VramConfigError("unknown_key")
        values = {key: raw.get(key, default) for key, default in DEFAULTS.items()}
        values["gpu_vram_reserve_mb"] = _mb(values["gpu_vram_reserve_mb"], "gpu_vram_reserve_mb")
        values["gpu_vram_budget_mb"] = _mb(values["gpu_vram_budget_mb"], "gpu_vram_budget_mb", allow_none=True)
        values["job_default_vram_mb"] = _mb(values["job_default_vram_mb"], "job_default_vram_mb")
        values["job_max_vram_mb"] = _mb(values["job_max_vram_mb"], "job_max_vram_mb")
        if values["job_default_vram_mb"] < 1 or values["job_max_vram_mb"] < values["job_default_vram_mb"]:
            raise VramConfigError("job_vram_range_invalid")
        overrides: dict[str, dict[str, int]] = {}
        for tenant, item in (raw.get("tenant_overrides") or {}).items():
            if not isinstance(tenant, str) or not tenant or not isinstance(item, Mapping) or set(item) - _TENANT_KEYS:
                raise VramConfigError("tenant_overrides_invalid")
            overrides[tenant] = {key: _mb(value, key) for key, value in item.items()}
        enabled = raw.get("gpu_broker_enabled", False)
        broker_path = raw.get("gpu_broker_config_path")
        stop_estimate = raw.get("gpu_broker_stop_estimate_seconds", 60)
        if (not isinstance(enabled, bool) or (broker_path is not None and (not isinstance(broker_path, str) or not broker_path))
                or isinstance(stop_estimate, bool) or not isinstance(stop_estimate, int) or not 0 <= stop_estimate <= 3600):
            raise VramConfigError("gpu_broker_settings_invalid")
        return cls(**values, tenant_overrides=overrides, gpu_broker_enabled=enabled,
                   gpu_broker_config_path=broker_path, gpu_broker_stop_estimate_seconds=stop_estimate)

    def for_tenant(self, tenant_id: str) -> tuple[int, int]:
        """(default, max) request MiB for one tenant after its override."""

        item = self.tenant_overrides.get(tenant_id, {})
        default = int(item.get("job_default_vram_mb", self.job_default_vram_mb))
        maximum = int(item.get("job_max_vram_mb", self.job_max_vram_mb))
        return min(default, maximum), maximum

    def ceiling_mb(self, tenant_id: str) -> float:
        """The most a job of this tenant can ever get: min(job_max, budget)."""

        _default, maximum = self.for_tenant(tenant_id)
        return min(maximum, self.gpu_vram_budget_mb if self.gpu_vram_budget_mb is not None else math.inf)

    def admissible_mb(self, tenant_id: str, probe_free_mb: int | None) -> int | None:
        """``min(job_max, budget or inf, probe_free - reserve)``; None when the probe is unavailable."""

        if probe_free_mb is None:
            return None
        live = int(probe_free_mb) - self.gpu_vram_reserve_mb
        return max(0, int(min(self.ceiling_mb(tenant_id), live)))

    def resolve_request(self, tenant_id: str, request_mb: int | None) -> int:
        """Default when unset; refuse at once what can never fit (``request_too_large``)."""

        default, _maximum = self.for_tenant(tenant_id)
        value = default if request_mb is None else int(request_mb)
        if value < 1:
            raise VramConfigError("gpu_request_invalid")
        if value > self.ceiling_mb(tenant_id):
            raise VramConfigError("request_too_large")
        return value


def config_path(runtime_root: Path) -> Path:
    return Path(os.environ.get("EVOMIND_GPU_VRAM_CONFIG") or Path(runtime_root) / "config" / CONFIG_NAME)


def load(runtime_root: Path) -> VramConfig:
    """Re-read on each admission: file (optional) then env overrides. Invalid → VramConfigError."""

    raw: dict[str, Any] = {}
    path = config_path(runtime_root)
    if path.is_file():
        try:
            document = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            raise VramConfigError("config_unreadable") from None
        if not isinstance(document, dict):
            raise VramConfigError("config_unreadable")
        raw.update(document)
    for key, name in _ENV.items():
        text = os.environ.get(name, "").strip()
        if not text:
            continue
        if key == "gpu_vram_budget_mb" and text.lower() in {"none", "unset"}:
            raw[key] = None
            continue
        if not text.isdigit():
            raise VramConfigError(f"{key}_invalid")
        raw[key] = int(text)
    enabled = os.environ.get(_BROKER_ENV["gpu_broker_enabled"], "").strip()
    if enabled:
        if enabled not in {"0", "1"}:
            raise VramConfigError("gpu_broker_settings_invalid")
        raw["gpu_broker_enabled"] = enabled == "1"
    broker_path = os.environ.get(_BROKER_ENV["gpu_broker_config_path"], "").strip()
    if broker_path:
        raw["gpu_broker_config_path"] = broker_path
    return VramConfig.from_mapping(raw)
