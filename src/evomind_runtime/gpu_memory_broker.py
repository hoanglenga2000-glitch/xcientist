"""GPU memory broker for the shared A800 (HPC job 98270) — t238 C7c-B.

Operator-side module. When a queued EvoMind tenant training job needs more VRAM than is free,
the broker gracefully stops (or, if explicitly configured, shrinks) the owner's own services on
that GPU (vLLM first, TTS only if still short), lets the job run, then restarts them with their
ORIGINAL start scripts and verifies they are healthy. The owner authorised exactly this; nothing
else on the machine is touched. See docs/GPU_MEMORY_BROKER_20261101.md.

Safety model (each point is covered by tests/test_gpu_memory_broker*.py):

* every configured command is validated by shape (``bash <abs-script>`` or an allow-listed
  loopback ``curl`` health probe) and every path is confined under ``REMOTE_ROOT``; config load
  fails closed;
* start/stop scripts are sha256-pinned (``FILL_AT_INSTALL`` = record on first use, then pin);
* a persistent SQLite state (``gpu_broker_state`` + ``gpu_broker_leases``) lets ``reconcile()``
  restore services after a host crash; a remote dead-man guard script restores them if the host
  stops heart-beating or ``max_down_seconds`` is hit;
* restore trouble never raises from ``release()``: it sets ``restore_failed``, alerts the
  operator and opens a circuit that refuses further stops until ``reset()``.
"""
from __future__ import annotations

import hashlib
import json
import posixpath
import os
import re
import socket
import sqlite3
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol
from uuid import uuid4

from research_os.remote_paths import TRAINER_MARKER

REMOTE_ROOT = "/hpc2hdd/home/aimslab/jinghw"
DEFAULT_BROKER_DIR = REMOTE_ROOT + "/evomind/gpu_broker"
FILL_AT_INSTALL = "FILL_AT_INSTALL"
GPU_QUERY_CMD = "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits"
TRAINER_PATTERN = "managed_tensor_train"

# Short Chinese, tenant-visible. Everything else in this module is operator-side.
TENANT_MESSAGES = {
    "refused": "排队中，轮到你就开始。",
    "busy": "正在调度，请稍候",
    "failed": "暂时无法开始训练，任务继续排队",
    "expired": "训练时间已到上限，训练已停止",
}


class BrokerError(Exception):
    """Base class; ``code`` is a stable machine reason, ``tenant_message`` short Chinese copy."""

    kind = "failed"

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail

    @property
    def tenant_message(self) -> str:
        return TENANT_MESSAGES[self.kind]


class BrokerConfigError(BrokerError):
    pass


class BrokerRefused(BrokerError):
    kind = "refused"


class BrokerBusy(BrokerError):
    kind = "busy"


class BrokerFailed(BrokerError):
    kind = "failed"


class BrokerLeaseExpired(BrokerError):
    """Raised by ``heartbeat`` when the job must be cancelled (max_down hit / guard took over)."""

    kind = "expired"


class RemoteExec(Protocol):
    def run(self, command: str, timeout: float) -> tuple[int, str, str]: ...


# --------------------------------------------------------------------------------------------
# Path / command confinement (fail closed at config load)
# --------------------------------------------------------------------------------------------

_SAFE_PATH_RE = re.compile(r"/[A-Za-z0-9._/-]+")
_SCRIPT = r"(/[A-Za-z0-9._/-]+\.sh)"
_BASH_RE = re.compile(r"bash " + _SCRIPT)
_BASH_ARGS_RE = re.compile(r"bash " + _SCRIPT + r"((?: [A-Za-z0-9._=-]+)*)")
_URL = r"(http://127\.0\.0\.1:[0-9]{2,5}(?:/[A-Za-z0-9._-]+)*/?)"
_CURL_CODE_RE = re.compile(r"curl -s -m ([0-9]{1,2}) -o /dev/null -w %\{http_code\} " + _URL)
_CURL_BODY_RE = re.compile(r"curl -s -m ([0-9]{1,2}) " + _URL)
_DETECT_RE = re.compile(r"[A-Za-z0-9._*:/= -]{6,160}")
_DETECT_LITERAL_RE = re.compile(r"[A-Za-z0-9_]{4,}")
_NAME_RE = re.compile(r"[a-z][a-z0-9_-]{0,31}")
_JOB_RE = re.compile(r"[A-Za-z0-9._:-]{1,128}")
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_CODE_EXPECT_RE = re.compile(r"[0-9]{3}")
_BODY_EXPECT_RE = re.compile(r'[A-Za-z0-9:_"{}.,-]{1,64}')
_TRAINER_RE = re.compile(r"[A-Za-z0-9._/-]{4,128}")


def confine_path(value: Any, what: str = "path", root: str = REMOTE_ROOT) -> str:
    """Return ``value`` if it is a normalised absolute path strictly under ``root``.

    Rejects ``..``/``.`` segments, ``//``, trailing ``/``, ``~``, ``$VAR``, backticks, shell
    metacharacters, whitespace, NUL/newline, other homes and prefix siblings (``…/jinghw_evil``).
    """
    if not isinstance(value, str) or not value:
        raise BrokerConfigError("path_invalid", f"{what}: must be a non-empty string")
    if any(ch in value for ch in "\x00\r\n"):
        raise BrokerConfigError("path_invalid", f"{what}: control character")
    if not _SAFE_PATH_RE.fullmatch(value):
        raise BrokerConfigError("path_invalid", f"{what}: unsafe characters or not absolute")
    segments = value.split("/")[1:]
    if any(seg in ("", ".", "..") for seg in segments) or posixpath.normpath(value) != value:
        raise BrokerConfigError("path_invalid", f"{what}: not normalised")
    if not value.startswith(root + "/"):
        raise BrokerConfigError("path_outside_root", f"{what}: outside {root}")
    return value


def parse_bash_command(command: Any, what: str, *, allow_args: bool = False) -> str:
    """Validate ``bash <abs-script>`` (optionally with plain fixed args); return the script path."""
    if not isinstance(command, str):
        raise BrokerConfigError("command_invalid", f"{what}: must be a string")
    match = (_BASH_ARGS_RE if allow_args else _BASH_RE).fullmatch(command)
    if not match:
        raise BrokerConfigError("command_invalid", f"{what}: only 'bash <abs-script.sh>' is allowed")
    return confine_path(match.group(1), what)


def parse_health_command(command: Any, what: str) -> tuple[str, str]:
    """Validate an allow-listed loopback curl probe; return ``(mode, url)`` with mode code|body."""
    if not isinstance(command, str):
        raise BrokerConfigError("command_invalid", f"{what}: must be a string")
    for mode, pattern in (("code", _CURL_CODE_RE), ("body", _CURL_BODY_RE)):
        match = pattern.fullmatch(command)
        if match:
            if not 1 <= int(match.group(1)) <= 30:
                raise BrokerConfigError("command_invalid", f"{what}: curl -m must be 1..30")
            url = match.group(2)
            if ".." in url:
                raise BrokerConfigError("command_invalid", f"{what}: '..' in url")
            return mode, url
    raise BrokerConfigError("command_invalid", f"{what}: not an allow-listed curl health form")


def _validate_detect(value: Any, what: str) -> str:
    if not isinstance(value, str) or not _DETECT_RE.fullmatch(value) or ".." in value:
        raise BrokerConfigError("detect_invalid", f"{what}: unsafe process pattern")
    if not _DETECT_LITERAL_RE.search(value):
        raise BrokerConfigError("detect_invalid", f"{what}: pattern too broad")
    return value


def _check_job_id(job_id: Any) -> str:
    if not isinstance(job_id, str) or not _JOB_RE.fullmatch(job_id):
        raise BrokerRefused("job_id_invalid")
    return job_id


def _as_mb(value: Any, what: str = "request_mb") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BrokerRefused("request_invalid", f"{what} must be a non-negative int")
    return value


# --------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------

_SERVICE_KEYS = {
    "name", "kind", "stop_order", "start_order", "resident_mb", "detect", "stop_cmd", "start_cmd",
    "health_cmd", "health_expect", "script_sha256", "protected", "pid_file", "log_file", "escalate",
    "shrink_start_cmd", "shrink_resident_mb",
}
_ESCALATIONS = ("none", "kill_recorded_pid")


def _int_field(raw: Mapping[str, Any], key: str, default: int, *, low: int, high: int, what: str) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise BrokerConfigError("config_invalid", f"{what}.{key}: int in [{low}, {high}] required")
    return value


def _bool_field(raw: Mapping[str, Any], key: str, default: bool, what: str) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise BrokerConfigError("config_invalid", f"{what}.{key}: bool required")
    return value


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], what: str) -> None:
    unknown = sorted(k for k in raw if k not in allowed and not str(k).startswith("_"))
    if unknown:
        raise BrokerConfigError("config_invalid", f"{what}: unknown keys {unknown}")


@dataclass(frozen=True)
class ServiceConfig:
    name: str
    kind: str
    stop_order: int
    start_order: int
    resident_mb: int
    detect: str
    stop_cmd: str
    start_cmd: str
    health_cmd: str
    health_expect: str
    script_sha256: dict[str, str]
    protected: bool = False
    pid_file: str | None = None
    log_file: str | None = None
    escalate: str = "none"
    shrink_start_cmd: str | None = None
    shrink_resident_mb: int | None = None

    @property
    def start_script(self) -> str:
        return parse_bash_command(self.start_cmd, f"{self.name}.start_cmd")

    @property
    def stop_script(self) -> str:
        return parse_bash_command(self.stop_cmd, f"{self.name}.stop_cmd")

    @property
    def shrink_script(self) -> str | None:
        if not self.shrink_start_cmd:
            return None
        return parse_bash_command(self.shrink_start_cmd, f"{self.name}.shrink_start_cmd", allow_args=True)

    @property
    def health(self) -> tuple[str, str]:
        return parse_health_command(self.health_cmd, f"{self.name}.health_cmd")

    @property
    def can_shrink(self) -> bool:
        return bool(self.shrink_start_cmd) and self.shrink_resident_mb is not None

    @classmethod
    def from_dict(cls, raw: Any, index: int) -> "ServiceConfig":
        what = f"services[{index}]"
        if not isinstance(raw, Mapping):
            raise BrokerConfigError("config_invalid", f"{what}: object required")
        _reject_unknown(raw, _SERVICE_KEYS, what)
        name = raw.get("name")
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise BrokerConfigError("config_invalid", f"{what}.name: [a-z][a-z0-9_-]{{0,31}} required")
        what = f"services[{name}]"
        kind = raw.get("kind", name)
        if not isinstance(kind, str) or not _NAME_RE.fullmatch(kind):
            raise BrokerConfigError("config_invalid", f"{what}.kind invalid")
        parse_bash_command(raw.get("stop_cmd"), f"{what}.stop_cmd")
        parse_bash_command(raw.get("start_cmd"), f"{what}.start_cmd")
        mode, _ = parse_health_command(raw.get("health_cmd"), f"{what}.health_cmd")
        expect = raw.get("health_expect", "200" if mode == "code" else None)
        expect_re = _CODE_EXPECT_RE if mode == "code" else _BODY_EXPECT_RE
        if not isinstance(expect, str) or not expect_re.fullmatch(expect):
            raise BrokerConfigError("config_invalid", f"{what}.health_expect does not fit health mode {mode}")
        shas_raw = raw.get("script_sha256", FILL_AT_INSTALL)
        if isinstance(shas_raw, str):
            shas_raw = {"start": shas_raw}
        if not isinstance(shas_raw, Mapping):
            raise BrokerConfigError("config_invalid", f"{what}.script_sha256: string or object required")
        _reject_unknown(shas_raw, {"start", "stop", "shrink"}, f"{what}.script_sha256")
        shas: dict[str, str] = {}
        for role in ("start", "stop", "shrink"):
            value = shas_raw.get(role, FILL_AT_INSTALL)
            if value != FILL_AT_INSTALL and not (isinstance(value, str) and _SHA_RE.fullmatch(value)):
                raise BrokerConfigError("config_invalid", f"{what}.script_sha256.{role}: sha256 hex or {FILL_AT_INSTALL}")
            shas[role] = value
        pid_file = raw.get("pid_file")
        if pid_file is not None:
            confine_path(pid_file, f"{what}.pid_file")
        log_file = raw.get("log_file")
        if log_file is not None:
            confine_path(log_file, f"{what}.log_file")
        escalate = raw.get("escalate", "none")
        if escalate not in _ESCALATIONS:
            raise BrokerConfigError("config_invalid", f"{what}.escalate must be one of {_ESCALATIONS}")
        if escalate == "kill_recorded_pid" and pid_file is None:
            raise BrokerConfigError("config_invalid", f"{what}.escalate kill_recorded_pid needs pid_file")
        shrink_cmd = raw.get("shrink_start_cmd")
        if shrink_cmd is not None:
            parse_bash_command(shrink_cmd, f"{what}.shrink_start_cmd", allow_args=True)
        resident = _int_field(raw, "resident_mb", 0, low=1, high=1 << 20, what=what)
        shrink_resident = raw.get("shrink_resident_mb")
        if shrink_resident is not None:
            shrink_resident = _int_field(raw, "shrink_resident_mb", 0, low=1, high=resident - 1, what=what)
        if (shrink_cmd is None) != (shrink_resident is None):
            raise BrokerConfigError("config_invalid", f"{what}: shrink_start_cmd and shrink_resident_mb go together")
        return cls(
            name=name,
            kind=kind,
            stop_order=_int_field(raw, "stop_order", 0, low=0, high=999, what=what),
            start_order=_int_field(raw, "start_order", 0, low=0, high=999, what=what),
            resident_mb=resident,
            detect=_validate_detect(raw.get("detect"), f"{what}.detect"),
            stop_cmd=raw["stop_cmd"],
            start_cmd=raw["start_cmd"],
            health_cmd=raw["health_cmd"],
            health_expect=expect,
            script_sha256=shas,
            protected=_bool_field(raw, "protected", False, what),
            pid_file=pid_file,
            log_file=log_file,
            escalate=escalate,
            shrink_start_cmd=shrink_cmd,
            shrink_resident_mb=shrink_resident,
        )


_CONFIG_KEYS = {
    "enabled", "allow_stop", "allow_shrink", "reserve_mb", "max_down_seconds", "heartbeat_window_seconds",
    "linger_seconds", "restore_health_timeout_s", "stop_timeout_s", "restore_retries", "poll_interval_s",
    "health_consecutive_ok", "gpu_total_mb", "lock_wait_s", "remote_root", "broker_dir", "trainer_patterns",
    "audit_path", "services",
}


@dataclass(frozen=True)
class BrokerConfig:
    services: tuple[ServiceConfig, ...]
    enabled: bool = False
    allow_stop: bool = True
    allow_shrink: bool = False
    reserve_mb: int = 1024
    max_down_seconds: int = 2700
    heartbeat_window_seconds: int = 600
    linger_seconds: int = 120
    restore_health_timeout_s: int = 900
    stop_timeout_s: int = 90
    restore_retries: int = 2
    poll_interval_s: int = 5
    health_consecutive_ok: int = 2
    gpu_total_mb: int = 81920
    lock_wait_s: float = 5.0
    broker_dir: str = DEFAULT_BROKER_DIR
    trainer_patterns: tuple[str, ...] = (TRAINER_PATTERN, TRAINER_MARKER)
    audit_path: str | None = None

    @property
    def lease_path(self) -> str:
        return self.broker_dir + "/lease.json"

    @property
    def guard_path(self) -> str:
        return self.broker_dir + "/guard.sh"

    @property
    def lock_path(self) -> str:
        return self.broker_dir + "/restore.lock"

    def service(self, name: str) -> ServiceConfig:
        for svc in self.services:
            if svc.name == name:
                return svc
        raise KeyError(name)

    @classmethod
    def from_json(cls, source: str | Path | Mapping[str, Any]) -> "BrokerConfig":
        """Load from a mapping, a JSON string, or a path to a JSON file. Fails closed."""
        if isinstance(source, Mapping):
            return cls.from_dict(source)
        if isinstance(source, Path) or not str(source).lstrip().startswith("{"):
            text = Path(source).read_text(encoding="utf-8")
        else:
            text = str(source)
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise BrokerConfigError("config_invalid", f"json: {exc}") from exc
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Any) -> "BrokerConfig":
        if not isinstance(raw, Mapping):
            raise BrokerConfigError("config_invalid", "top level: object required")
        _reject_unknown(raw, _CONFIG_KEYS, "config")
        if raw.get("remote_root", REMOTE_ROOT) != REMOTE_ROOT:
            raise BrokerConfigError("path_outside_root", f"remote_root is fixed to {REMOTE_ROOT}")
        broker_dir = confine_path(raw.get("broker_dir", DEFAULT_BROKER_DIR), "broker_dir")
        if broker_dir != DEFAULT_BROKER_DIR:  # the guard script hardcodes this directory
            raise BrokerConfigError("config_invalid", f"broker_dir is fixed to {DEFAULT_BROKER_DIR}")
        patterns = raw.get("trainer_patterns", [TRAINER_PATTERN])
        if not isinstance(patterns, list) or not patterns:
            raise BrokerConfigError("config_invalid", "trainer_patterns: non-empty list required")
        for pattern in patterns:
            if not isinstance(pattern, str) or not _TRAINER_RE.fullmatch(pattern) or TRAINER_PATTERN not in pattern \
                    or ".." in pattern:
                raise BrokerConfigError("config_invalid", f"trainer_patterns: must contain {TRAINER_PATTERN!r}")
        audit_path = raw.get("audit_path")
        if audit_path is not None and not isinstance(audit_path, str):
            raise BrokerConfigError("config_invalid", "audit_path: string required")
        raw_services = raw.get("services")
        if not isinstance(raw_services, list):
            raise BrokerConfigError("config_invalid", "services: list required")
        services = tuple(ServiceConfig.from_dict(item, i) for i, item in enumerate(raw_services))
        for attr in ("name", "stop_order", "start_order"):
            values = [getattr(svc, attr) for svc in services]
            if len(values) != len(set(values)):
                raise BrokerConfigError("config_invalid", f"services: duplicate {attr}")
        lock_wait = raw.get("lock_wait_s", 5.0)
        if isinstance(lock_wait, bool) or not isinstance(lock_wait, (int, float)) or not 0 <= lock_wait <= 600:
            raise BrokerConfigError("config_invalid", "lock_wait_s: number in [0, 600] required")
        heartbeat_window = _int_field(raw, "heartbeat_window_seconds", 600, low=60, high=86400, what="config")
        max_down = _int_field(raw, "max_down_seconds", 2700, low=60, high=7 * 86400, what="config")
        if heartbeat_window > max_down:
            raise BrokerConfigError("config_invalid", "heartbeat_window_seconds must be <= max_down_seconds")
        return cls(
            services=services,
            enabled=_bool_field(raw, "enabled", False, "config"),
            allow_stop=_bool_field(raw, "allow_stop", True, "config"),
            allow_shrink=_bool_field(raw, "allow_shrink", False, "config"),
            reserve_mb=_int_field(raw, "reserve_mb", 1024, low=0, high=1 << 20, what="config"),
            max_down_seconds=max_down,
            heartbeat_window_seconds=heartbeat_window,
            linger_seconds=_int_field(raw, "linger_seconds", 120, low=0, high=3600, what="config"),
            restore_health_timeout_s=_int_field(raw, "restore_health_timeout_s", 900, low=10, high=7200, what="config"),
            stop_timeout_s=_int_field(raw, "stop_timeout_s", 90, low=5, high=3600, what="config"),
            restore_retries=_int_field(raw, "restore_retries", 2, low=0, high=5, what="config"),
            poll_interval_s=_int_field(raw, "poll_interval_s", 5, low=1, high=120, what="config"),
            health_consecutive_ok=_int_field(raw, "health_consecutive_ok", 2, low=1, high=10, what="config"),
            gpu_total_mb=_int_field(raw, "gpu_total_mb", 81920, low=1024, high=1 << 22, what="config"),
            lock_wait_s=float(lock_wait),
            broker_dir=broker_dir,
            # The marker is what the real ``bundle_*/work/train_gpu.py`` process carries; never optional.
            trainer_patterns=tuple(dict.fromkeys([*patterns, TRAINER_MARKER])),
            audit_path=audit_path,
        )


# --------------------------------------------------------------------------------------------
# Remote dead-man guard (shipped as gpu_memory_broker_guard.sh; embedded here for upload)
# --------------------------------------------------------------------------------------------

LEASE_TERMINAL_STATES = ("released", "restored", "restore_failed")
LEASE_FIRING_STATES = ("active", "broker_restoring")

# Byte-identical copy of gpu_memory_broker_guard.sh (a test enforces parity).
GUARD_SCRIPT = r'''#!/usr/bin/env bash
# EvoMind GPU memory broker dead-man guard (t238 C7c-B). Uploaded + launched by
# src/evomind_runtime/gpu_memory_broker.py; operator doc: docs/GPU_MEMORY_BROKER_20261101.md.
# When the EvoMind host stops heart-beating (or max_down is hit) it kills ONLY EvoMind trainer
# processes under ROOT, restarts the owner's services with their original start scripts in
# start order, waits for health and records the result in lease.json + guard.log.
# Touches nothing else. Holds no secrets. Never retries in a loop.
set -u
ROOT=/hpc2hdd/home/aimslab/jinghw
DIR="$ROOT/evomind/gpu_broker"
LEASE="$DIR/lease.json"
LOCK="$DIR/restore.lock"
LOG="$DIR/guard.log"
PIDF="$DIR/guard.pid"
POLL=15

now() { date +%s; }
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) guard[$$] $*" >> "$LOG"; }
num() { case "$1" in ''|*[!0-9]*) return 1 ;; esac; return 0; }

field() {
  sed -n "s/^  \"$1\": \"\{0,1\}\([A-Za-z0-9_]*\)\"\{0,1\},\{0,1\}\$/\1/p" "$LEASE" 2>/dev/null | head -n 1
}

items() {
  awk -v k="$1" '
    $0 == "  \"" k "\": [" { on = 1; next }
    on && /^  \]/ { exit }
    on { s = $0; sub(/^ *"/, "", s); sub(/",?$/, "", s); gsub(/\\"/, "\"", s); print s }
  ' "$LEASE" 2>/dev/null
}

set_state() {
  sed "s/^  \"state\": \"[a-z_]*\"/  \"state\": \"$1\"/" "$LEASE" > "$LEASE.g$$" && mv -f "$LEASE.g$$" "$LEASE"
}

safe_script() {
  case "$1" in "$ROOT"/*.sh) ;; *) return 1 ;; esac
  case "$1" in *..*|*//*|*[!A-Za-z0-9._/-]*) return 1 ;; esac
  return 0
}

run_script() {
  s=${1#bash }
  if [ "bash $s" != "$1" ] || ! safe_script "$s"; then log "refused command: $1"; return 1; fi
  if [ -n "${2:-}" ] && [ "$(sha256sum -- "$s" 2>/dev/null | cut -d' ' -f1)" != "$2" ]; then
    log "refused changed script: $s"; return 1
  fi
  bash "$s" </dev/null >/dev/null 2>&1
}

healthy() {
  url=${1%% *}; want=${1#* }
  if [ -z "$want" ] || [ "$want" = "$1" ]; then return 1; fi
  case "$url" in http://127.0.0.1:[0-9]*) ;; *) return 1 ;; esac
  case "$url" in *..*|*[!A-Za-z0-9.:/_-]*) return 1 ;; esac
  case "$want" in
    *[!0-9]*)
      body=$(curl -s -m 5 "$url" 2>/dev/null | tr -d ' \t\r\n')
      case "$body" in *"$want"*) return 0 ;; esac
      return 1 ;;
    *) [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' "$url" 2>/dev/null)" = "$want" ] ;;
  esac
}

trainer_pids() {
  me=$(id -u)
  items kill_patterns | while IFS= read -r pat; do
    case "$pat" in *managed_tensor_train*|*evomind_gpu_trainer*) ;; *) continue ;; esac
    case "$pat" in *..*|*[!A-Za-z0-9._/-]*) continue ;; esac
    pgrep -u "$me" -f -- "$pat" 2>/dev/null
  done | sort -u | while IFS= read -r p; do
    [ "$p" = "$$" ] && continue
    cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
    case "$cmd" in *"$ROOT/"*) echo "$p" ;; esac
  done
}

kill_trainers() {
  for p in $(trainer_pids); do log "TERM trainer pid $p"; kill -TERM "$p" 2>/dev/null; done
  sleep 10
  for p in $(trainer_pids); do log "KILL trainer pid $p"; kill -KILL "$p" 2>/dev/null; done
}

take_lock() {
  mkdir "$LOCK" 2>/dev/null && return 0
  m=$(stat -c %Y "$LOCK" 2>/dev/null) || m=0
  if [ $(( $1 - m )) -gt "$2" ]; then
    log "stale restore lock (age $(( $1 - m ))s), taking over"
    rmdir "$LOCK" 2>/dev/null
    mkdir "$LOCK" 2>/dev/null && return 0
  fi
  return 1
}

fire() {
  set_state restoring
  log "deadline passed (state=$1): forced restore"
  kill_trainers
  ht=$(field health_timeout_s); num "$ht" || ht=900
  ok=1; i=0
  while IFS= read -r cmd <&3; do
    [ -n "$cmd" ] || continue
    log "pre-stop: $cmd"; run_script "$cmd"
  done 3<<EOF
$(items pre_stop)
EOF
  while IFS= read -r cmd <&3; do
    [ -n "$cmd" ] || continue
    i=$((i + 1)); h=$(items health | sed -n "${i}p")
    if healthy "$h"; then log "already healthy: $cmd"; continue; fi
    log "start: $cmd"
    run_script "$cmd" "$(items restore_sha256 | sed -n "${i}p")" || { ok=0; continue; }
    waited=0
    until healthy "$h"; do
      if [ "$waited" -ge "$ht" ]; then log "unhealthy after ${ht}s: $cmd"; ok=0; break; fi
      sleep "$POLL"; waited=$((waited + POLL))
    done
  done 3<<EOF
$(items restore_order)
EOF
  if [ "$ok" = 1 ]; then
    set_state restored; log "restored"
  else
    set_state restore_failed; log "restore_failed: operator action needed (docs/GPU_MEMORY_BROKER_20261101.md)"
  fi
}

if [ -f "$PIDF" ]; then
  old=$(cat "$PIDF" 2>/dev/null)
  if num "$old" && [ "$old" != "$$" ] && kill -0 "$old" 2>/dev/null; then
    case "$(tr '\0' ' ' < "/proc/$old/cmdline" 2>/dev/null)" in *gpu_broker/guard.sh*) exit 0 ;; esac
  fi
fi
echo "$$" > "$PIDF"
log "start"
while :; do
  [ -f "$LEASE" ] || { log "no lease: exit"; break; }
  st=$(field state); dl=$(field deadline_epoch); hx=$(field hard_exit_epoch); bud=$(field restore_budget_s)
  case "$st" in released|restored|restore_failed) log "lease $st: exit"; break ;; esac
  if ! num "$dl" || ! num "$hx" || ! num "$bud"; then log "unreadable lease: exit"; break; fi
  t=$(now)
  if [ "$t" -gt "$hx" ]; then log "hard exit time passed: exit"; break; fi
  if [ "$t" -gt "$dl" ] && { [ "$st" = active ] || [ "$st" = broker_restoring ]; }; then
    if take_lock "$t" "$bud"; then fire "$st"; break; fi
  fi
  sleep "$POLL"
done
[ "$(cat "$PIDF" 2>/dev/null)" = "$$" ] && rm -f "$PIDF"
exit 0
'''


def lease_document_text(doc: Mapping[str, Any]) -> str:
    """Render lease.json in the canonical layout the guard's sed/awk parser expects."""
    return json.dumps(dict(doc), indent=2) + "\n"


def guard_decision(lease: Mapping[str, Any] | None, now: int, lock_age_s: float | None) -> str:
    """Python mirror of one guard poll: ``exit`` | ``fire`` | ``wait`` (same order as the script)."""
    if lease is None:
        return "exit"
    state = lease.get("state")
    if state in LEASE_TERMINAL_STATES:
        return "exit"
    numbers = [lease.get(k) for k in ("deadline_epoch", "hard_exit_epoch", "restore_budget_s")]
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in numbers):
        return "exit"
    deadline, hard_exit, budget = numbers
    if now > hard_exit:
        return "exit"
    if now > deadline and state in LEASE_FIRING_STATES:
        if lock_age_s is None or lock_age_s > budget:
            return "fire"
    return "wait"


# --------------------------------------------------------------------------------------------
# Persistent state (additive tables only)
# --------------------------------------------------------------------------------------------

WINDOW_ROW = "__window__"
LAST_RESTORE_ROW = "__last_restore__"
# window states: idle | down | lingering | restoring | guard_restoring | restore_failed (= circuit open)
_RESTORABLE = ("stopping", "shrinking", "stopped", "shrunk", "restoring")
_SHRINK_STATES = ("shrinking", "shrunk")
GUARD_GRACE_S = 60  # let the remote guard (15 s poll) act first before the broker restores on expiry
_HEREDOC = "EOF_EVOMIND_GPU_BROKER"
_SECRET_LINE_RE = re.compile(r"(?i)(api[_-]?key|token|secret|passw|authorization|bearer|sk-[A-Za-z0-9])")

_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS gpu_broker_state (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        service TEXT NOT NULL UNIQUE,
        state TEXT NOT NULL,
        job_id TEXT,
        stopped_at REAL,
        deadline_at REAL,
        restore_attempts INTEGER NOT NULL DEFAULT 0,
        last_error TEXT,
        updated_at REAL NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS gpu_broker_leases (
        job_id TEXT PRIMARY KEY,
        request_mb INTEGER NOT NULL,
        free_mb_at_grant INTEGER,
        freed_mb INTEGER NOT NULL DEFAULT 0,
        stopped TEXT NOT NULL DEFAULT '[]',
        attached INTEGER NOT NULL DEFAULT 0,
        state TEXT NOT NULL,
        cancel_reason TEXT,
        granted_at REAL NOT NULL,
        heartbeat_at REAL NOT NULL,
        released_at REAL,
        updated_at REAL NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS gpu_broker_script_pins (
        script TEXT PRIMARY KEY,
        sha256 TEXT NOT NULL,
        pinned_at REAL NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS gpu_broker_lock (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        owner TEXT,
        expires_at REAL NOT NULL DEFAULT 0)""",
)


def _pid_alive(pid: int) -> bool:
    """Never ``os.kill(pid, 0)`` on Windows (the EvoMind host): signal 0 is CTRL_C_EVENT there."""
    try:
        import psutil
    except ImportError:
        if os.name != "posix":
            return True  # unknown: wait for the lock to expire instead
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except OSError:
            return True
        return True
    return bool(psutil.pid_exists(pid))


def _lock_owner() -> str:
    return f"{socket.gethostname()}|{os.getpid()}|{uuid4().hex}"


def _lock_owner_dead(owner: str | None) -> bool:
    """Only a same-host owner with a dead pid; another host or an old-format owner waits for expiry."""
    parts = str(owner or "").split("|")
    if len(parts) != 3 or parts[0] != socket.gethostname() or not parts[1].isdigit():
        return False
    return int(parts[1]) != os.getpid() and not _pid_alive(int(parts[1]))


class _StateStore:
    def __init__(self, path: Path, clock: Callable[[], float]) -> None:
        self.path = Path(path)
        self.clock = clock
        self.took_over: str | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            for statement in _SCHEMA:
                conn.execute(statement)
            conn.execute("INSERT OR IGNORE INTO gpu_broker_lock (id, owner, expires_at) VALUES (1, NULL, 0)")

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path), timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def row(self, service: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            found = conn.execute("SELECT * FROM gpu_broker_state WHERE service = ?", (service,)).fetchone()
        return dict(found) if found else None

    def put(self, service: str, **values: Any) -> None:
        values["updated_at"] = self.clock()
        with self._conn() as conn:
            cur = conn.execute(
                f"UPDATE gpu_broker_state SET {', '.join(f'{k} = ?' for k in values)} WHERE service = ?",
                [*values.values(), service],
            )
            if cur.rowcount == 0:
                values.setdefault("state", "up")
                conn.execute(
                    f"INSERT INTO gpu_broker_state (service, {', '.join(values)}) "
                    f"VALUES (?, {', '.join('?' * len(values))})",
                    [service, *values.values()],
                )

    def lease(self, job_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            found = conn.execute("SELECT * FROM gpu_broker_leases WHERE job_id = ?", (job_id,)).fetchone()
        return dict(found) if found else None

    def leases(self, state: str = "active") -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM gpu_broker_leases WHERE state = ? ORDER BY granted_at, job_id", (state,)
            ).fetchall()
        return [dict(r) for r in rows]

    def put_lease(self, job_id: str, *, replace: bool = False, **values: Any) -> None:
        values["updated_at"] = self.clock()
        with self._conn() as conn:
            if replace:
                conn.execute("DELETE FROM gpu_broker_leases WHERE job_id = ?", (job_id,))
                conn.execute(
                    f"INSERT INTO gpu_broker_leases (job_id, {', '.join(values)}) "
                    f"VALUES (?, {', '.join('?' * len(values))})",
                    [job_id, *values.values()],
                )
            else:
                conn.execute(
                    f"UPDATE gpu_broker_leases SET {', '.join(f'{k} = ?' for k in values)} WHERE job_id = ?",
                    [*values.values(), job_id],
                )

    def pin(self, script: str) -> str | None:
        with self._conn() as conn:
            found = conn.execute("SELECT sha256 FROM gpu_broker_script_pins WHERE script = ?", (script,)).fetchone()
        return found["sha256"] if found else None

    def set_pin(self, script: str, sha256: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO gpu_broker_script_pins (script, sha256, pinned_at) VALUES (?, ?, ?)",
                (script, sha256, self.clock()),
            )

    def try_lock(self, owner: str, ttl_s: float) -> bool:
        """Free, expired, or held by a process that is provably dead on this host."""
        now = self.clock()
        with self._conn() as conn:
            current = conn.execute("SELECT owner FROM gpu_broker_lock WHERE id = 1").fetchone()
            dead = current["owner"] if current and _lock_owner_dead(current["owner"]) else None
            cur = conn.execute(
                "UPDATE gpu_broker_lock SET owner = ?, expires_at = ? "
                "WHERE id = 1 AND (owner IS NULL OR expires_at < ? OR owner = ?)",
                (owner, now + ttl_s, now, dead),
            )
        self.took_over = dead if cur.rowcount == 1 else None
        return cur.rowcount == 1

    def unlock(self, owner: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE gpu_broker_lock SET owner = NULL, expires_at = 0 WHERE id = 1 AND owner = ?", (owner,))


# --------------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------------


@dataclass
class MemoryLease:
    job_id: str
    request_mb: int
    freed_mb: int = 0
    stopped: list[str] = field(default_factory=list)
    free_mb_at_grant: int | None = None
    granted_at: float = 0.0
    max_down_at: float | None = None
    attached: bool = False  # True while the job relies on services the broker stopped

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RestoreResult:
    ok: bool
    action: str  # none | still_held | lingering | restored | restored_by_guard | guard_restoring | restore_failed
    restored: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    attempts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    log_tails: dict[str, list[str]] = field(default_factory=dict)
    pending: bool = False
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _job_hash(job_id: str | None) -> str:
    return hashlib.sha256((job_id or "").encode("utf-8")).hexdigest()[:16]


def _redact(lines: list[str]) -> list[str]:
    return ["[redacted line]" if _SECRET_LINE_RE.search(line) else line[:400] for line in lines]


# --------------------------------------------------------------------------------------------
# Broker
# --------------------------------------------------------------------------------------------


class GpuMemoryBroker:
    """Stop/restore the owner's GPU services around tenant training. See module docstring."""

    def __init__(
        self,
        config: BrokerConfig,
        remote: RemoteExec,
        state_db: Path,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        audit: Callable[[dict[str, Any]], None] | None = None,
        alert: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self._remote = remote
        self._clock = clock
        self._sleep = sleep
        self._audit_cb = audit
        self._alert_cb = alert
        self._store = _StateStore(Path(state_db), clock)
        self.audit_path = (
            Path(config.audit_path) if config.audit_path else Path(state_db).with_name("gpu_broker_audit.jsonl")
        )
        self._lock = threading.RLock()

    # ---------------------------------------------------------------- public interface

    def status(self) -> dict[str, Any]:
        """Operator snapshot (tenant_safe=False): free memory, service states, leases, last restore."""
        snapshot: dict[str, Any] = {"tenant_safe": False, "enabled": self.config.enabled}
        try:
            used, total, free = self._probe()
            snapshot.update(used_mb=used, total_mb=total, free_mb=free)
        except Exception as exc:  # noqa: BLE001 - status must not raise
            snapshot.update(used_mb=None, total_mb=None, free_mb=None, probe_error=type(exc).__name__)
        window = self._window()
        snapshot["window"] = {
            "state": window["state"],
            "holder_job_id": window.get("job_id"),
            "stopped_at": window.get("stopped_at"),
            "deadline_at": window.get("deadline_at"),
            "max_down_at": self._max_down_at(window),
        }
        snapshot["circuit_open"] = window["state"] == "restore_failed"
        services = []
        for svc in self.config.services:
            row = self._store.row(svc.name) or {}
            try:
                process_up: bool | None = self._detect_up(svc)
            except Exception:  # noqa: BLE001
                process_up = None
            services.append({
                "name": svc.name,
                "kind": svc.kind,
                "protected": svc.protected,
                "resident_mb": svc.resident_mb,
                "state": row.get("state", "up"),
                "process_up": process_up,
                "restore_attempts": row.get("restore_attempts", 0),
                "last_error": row.get("last_error"),
            })
        snapshot["services"] = services
        snapshot["leases"] = [
            {k: lease[k] for k in ("job_id", "request_mb", "freed_mb", "attached", "granted_at", "heartbeat_at")}
            for lease in self._store.leases("active")
        ]
        last = self._store.row(LAST_RESTORE_ROW)
        snapshot["last_restore"] = (
            {"result": last["state"], "at": last["updated_at"], "detail": last.get("last_error")} if last else None
        )
        return snapshot

    def plan(self, request_mb: int, free_mb: int) -> list[str]:
        """Services (stop_order) that must be stopped; ``name:shrink`` for a shrink. [] = nothing to do."""
        request_mb = _as_mb(request_mb)
        free_mb = _as_mb(free_mb, "free_mb")
        if request_mb <= free_mb - self.config.reserve_mb:
            return []
        candidates = [svc for svc in self.config.services if not svc.protected]
        return [f"{svc.name}:shrink" if mode == "shrink" else svc.name
                for svc, mode in self._plan_steps(request_mb, free_mb, candidates)]

    def acquire(self, job_id: str, request_mb: int) -> MemoryLease:
        job_id = _check_job_id(job_id)
        request_mb = _as_mb(request_mb)
        with self._exclusive():
            existing = self._store.lease(job_id)
            if existing and existing["state"] == "active":
                return self._lease_obj(existing)
            self._housekeeping()
            window = self._window()
            state = window["state"]
            if state in ("restoring", "guard_restoring"):
                raise BrokerBusy("restore_in_progress")
            used, total, free = self._probe()
            fits = request_mb <= free - self.config.reserve_mb
            if state == "restore_failed":
                if fits:
                    return self._grant(job_id, request_mb, free, attached=True)
                raise BrokerRefused("circuit_open", "restore failed earlier; operator reset() required")
            if state == "down":
                if fits:
                    return self._grant(job_id, request_mb, free, attached=True)
                raise BrokerBusy("window_held", "services already lent to another job")
            if state == "lingering":
                if fits:
                    return self._take_over_linger(job_id, request_mb, free)
                steps = self._plan_steps(request_mb, free, self._up_candidates())
                return self._stop_into(job_id, request_mb, steps, free, new_window=False)
            if fits:
                return self._grant(job_id, request_mb, free, attached=False)
            steps = self._plan_steps(request_mb, free, self._up_candidates())
            return self._stop_into(job_id, request_mb, steps, free, new_window=True)

    def heartbeat(self, job_id: str) -> None:
        job_id = _check_job_id(job_id)
        with self._exclusive():
            lease = self._store.lease(job_id)
            if not lease or lease["state"] != "active":
                raise BrokerLeaseExpired((lease or {}).get("cancel_reason") or "lease_not_active")
            now = self._clock()
            self._store.put_lease(job_id, heartbeat_at=now)
            if not lease["attached"]:
                return
            window = self._window()
            if window["state"] in ("restoring", "guard_restoring"):
                self._expire_lease(job_id, "guard_took_over")
                raise BrokerLeaseExpired("guard_took_over")
            if window["state"] != "down":
                return
            if now >= self._max_down_at(window):
                self._expire_lease(job_id, "max_down")
                raise BrokerLeaseExpired("max_down", "max_down_seconds reached; cancel the job")
            try:
                written = self._write_lease("active", now + self.config.heartbeat_window_seconds, unless_locked=True)
                if written:
                    self._launch_guard()
            except Exception as exc:  # noqa: BLE001 - the guard deadline keeps running; caller retries
                self._audit("heartbeat", job_id=job_id, result="error", code=type(exc).__name__)
                raise BrokerFailed("heartbeat_failed", type(exc).__name__) from exc
            if not written:
                self._store.put(WINDOW_ROW, state="guard_restoring")
                self._expire_lease(job_id, "guard_took_over")
                raise BrokerLeaseExpired("guard_took_over")
            self._store.put(WINDOW_ROW, deadline_at=min(now + self.config.heartbeat_window_seconds,
                                                         self._max_down_at(window)))

    def release(self, job_id: str, *, next_needs_mb: int | None = None) -> RestoreResult:
        job_id = _check_job_id(job_id)
        started = self._clock()
        with self._exclusive(wait_s=self._op_budget_s()):
            lease = self._store.lease(job_id)
            if not lease or lease["state"] not in ("active", "expired"):
                return RestoreResult(ok=True, action="none")
            now = self._clock()
            self._store.put_lease(job_id, state="released", released_at=now)
            self._audit("release", job_id=job_id)
            if not lease["attached"]:
                return RestoreResult(ok=True, action="none")
            window = self._window()
            state = window["state"]
            if state == "restore_failed":
                return RestoreResult(ok=False, action="restore_failed", errors=["circuit_open"])
            if state in ("restoring", "guard_restoring"):
                return self._restore_window("released", started)
            others = [x for x in self._store.leases("active") if x["attached"]]
            if others:
                self._store.put(WINDOW_ROW, job_id=others[0]["job_id"])
                return RestoreResult(ok=True, action="still_held")
            if next_needs_mb is not None and self._should_linger(window, next_needs_mb, now):
                until = min(now + self.config.linger_seconds, self._max_down_at(window))
                try:
                    self._write_lease("active", until, unless_locked=True, kill=False)
                except Exception as exc:  # noqa: BLE001 - cannot re-arm the guard: restore instead
                    self._audit("linger", job_id=job_id, result="error", code=type(exc).__name__)
                    return self._restore_window("linger_unarmed", started)
                self._store.put(WINDOW_ROW, state="lingering", deadline_at=until)
                self._audit("linger", job_id=job_id, code=str(int(until - now)))
                return RestoreResult(ok=True, action="lingering", duration_s=self._clock() - started)
            return self._restore_window("released", started)

    def reconcile(self, *, wait_s: float | None = None) -> RestoreResult | None:
        """Start-up / periodic: restore services the broker stopped when no live lease remains.

        ``wait_s`` bounds the lock wait; on ``BrokerBusy`` the caller retries later.
        """
        with self._exclusive(wait_s=self._op_budget_s() if wait_s is None else wait_s):
            return self._housekeeping()

    def reset(self) -> dict[str, Any]:
        """Operator: close the circuit after a restore failure; the next reconcile retries once."""
        with self._exclusive():
            for svc in self.config.services:
                row = self._store.row(svc.name)
                if row and row["state"] == "restore_failed":
                    self._store.put(svc.name, state="stopped", restore_attempts=0)
            window = self._window()
            if window["state"] == "restore_failed":
                attached = [x for x in self._store.leases("active") if x["attached"]]
                if attached:
                    self._store.put(WINDOW_ROW, state="down", deadline_at=self._clock())
                else:  # an already-expired linger is restored by the next reconcile()
                    self._store.put(WINDOW_ROW, state="lingering", deadline_at=self._clock())
            self._audit("reset", result="ok")
        return self.status()

    @contextmanager
    def context(self, job_id: str, request_mb: int) -> Iterator[MemoryLease]:
        """acquire … finally release — also on exception, KeyboardInterrupt, cancel or timeout."""
        lease = self.acquire(job_id, request_mb)
        try:
            yield lease
        finally:
            self.release(job_id)

    # ---------------------------------------------------------------- planning

    def _plan_steps(
        self, request_mb: int, free_mb: int, candidates: list[ServiceConfig]
    ) -> list[tuple[ServiceConfig, str]]:
        cfg = self.config
        if not cfg.enabled:
            raise BrokerRefused("disabled", "broker disabled (config enabled=false)")
        if self._window()["state"] == "restore_failed":
            raise BrokerRefused("circuit_open", "restore failed earlier; operator reset() required")
        if request_mb > cfg.gpu_total_mb - cfg.reserve_mb:
            raise BrokerRefused("exceeds_capacity", f"{request_mb} MiB > total - reserve")
        need = request_mb + cfg.reserve_mb - free_mb
        steps: list[tuple[ServiceConfig, str]] = []
        for svc in sorted(candidates, key=lambda s: s.stop_order):
            if need <= 0:
                break
            if svc.protected:
                continue
            saving = svc.resident_mb - (svc.shrink_resident_mb or 0)
            if cfg.allow_shrink and svc.can_shrink and need <= saving:
                steps.append((svc, "shrink"))
                need -= saving
            elif cfg.allow_stop:
                steps.append((svc, "stop"))
                need -= svc.resident_mb
        if need > 0:
            if not cfg.allow_stop:
                raise BrokerRefused("stop_not_allowed", "allow_stop=false")
            raise BrokerRefused("insufficient_even_if_all_stopped")
        return steps

    def _up_candidates(self) -> list[ServiceConfig]:
        out = []
        for svc in self.config.services:
            if svc.protected or self._svc_state(svc) != "up":
                continue
            if self._detect_up(svc):
                out.append(svc)
        return out

    # ---------------------------------------------------------------- acquire internals

    def _grant(self, job_id: str, request_mb: int, free_mb: int, *, attached: bool) -> MemoryLease:
        now = self._clock()
        self._store.put_lease(
            job_id, replace=True, request_mb=request_mb, free_mb_at_grant=free_mb, freed_mb=0, stopped="[]",
            attached=int(attached), state="active", cancel_reason=None, granted_at=now, heartbeat_at=now,
            released_at=None,
        )
        self._audit("grant", job_id=job_id, code="shared_window" if attached else "fits_free")
        return self._lease_obj(self._store.lease(job_id))

    def _take_over_linger(self, job_id: str, request_mb: int, free_mb: int) -> MemoryLease:
        now = self._clock()
        lease = self._grant(job_id, request_mb, free_mb, attached=True)
        self._store.put(WINDOW_ROW, state="down", job_id=job_id)
        try:
            if not self._write_lease("active", now + self.config.heartbeat_window_seconds, unless_locked=True):
                raise BrokerFailed("guard_took_over")
            self._launch_guard()
        except BaseException as exc:
            self._expire_lease(job_id, "guard_unarmed")
            self._restore_window("takeover_failed", now)
            if isinstance(exc, Exception):
                raise BrokerFailed(getattr(exc, "code", "takeover_failed")) from exc
            raise
        self._store.put(WINDOW_ROW, deadline_at=min(now + self.config.heartbeat_window_seconds,
                                                     self._max_down_at(self._window())))
        self._audit("linger_takeover", job_id=job_id)
        return lease

    def _stop_into(
        self, job_id: str, request_mb: int, steps: list[tuple[ServiceConfig, str]], free_before: int,
        *, new_window: bool,
    ) -> MemoryLease:
        started = self._clock()
        # Pre-flight before touching anything: never stop a service we could not restart.
        for svc, mode in steps:
            for role in ("stop", "start") + (("shrink",) if mode == "shrink" else ()):
                self._verify_script(svc, role)
        now = self._clock()
        if new_window:
            self._store.put(WINDOW_ROW, state="down", job_id=job_id, stopped_at=now,
                            deadline_at=now + self.config.heartbeat_window_seconds, restore_attempts=0,
                            last_error=None)
        else:
            self._store.put(WINDOW_ROW, state="down", job_id=job_id)
        self._store.put_lease(
            job_id, replace=True, request_mb=request_mb, free_mb_at_grant=free_before, freed_mb=0, stopped="[]",
            attached=1, state="active", cancel_reason=None, granted_at=now, heartbeat_at=now, released_at=None,
        )
        done: list[str] = []
        try:
            for svc, mode in steps:
                self._store.put(svc.name, state="stopping" if mode == "stop" else "shrinking", job_id=job_id,
                                stopped_at=now, restore_attempts=0, last_error=None)
            self._install_guard(clear_lock=new_window)
            self._write_lease("active", now + self.config.heartbeat_window_seconds)
            self._launch_guard()
            for svc, mode in steps:
                t0 = self._clock()
                self._stop_service(svc, job_id)
                if mode == "shrink":
                    self._run(svc.shrink_start_cmd or "", self.config.stop_timeout_s + 60)
                    if not self._wait_healthy(svc, self.config.restore_health_timeout_s):
                        raise BrokerFailed("shrink_unhealthy", svc.name)
                    self._store.put(svc.name, state="shrunk")
                else:
                    self._store.put(svc.name, state="stopped")
                done.append(f"{svc.name}:shrink" if mode == "shrink" else svc.name)
                self._audit("shrink" if mode == "shrink" else "stop", job_id=job_id, service=svc.name,
                            duration_s=self._clock() - t0)
            _, _, free_after = self._probe()
            if request_mb > free_after - self.config.reserve_mb:
                raise BrokerFailed("insufficient_after_stop", f"free {free_after} MiB")
        except BaseException as exc:
            code = exc.code if isinstance(exc, BrokerError) else type(exc).__name__
            self._audit("acquire", job_id=job_id, result="failed", code=code, duration_s=self._clock() - started)
            self._store.put_lease(job_id, state="failed", cancel_reason=code, released_at=self._clock())
            self._restore_window("acquire_failed", started)
            if isinstance(exc, BrokerError):
                raise BrokerFailed(exc.code, exc.detail) from exc
            if isinstance(exc, Exception):
                raise BrokerFailed("acquire_failed", type(exc).__name__) from exc
            raise
        previous = json.loads((self._store.lease(job_id) or {}).get("stopped") or "[]")
        self._store.put_lease(job_id, freed_mb=max(0, free_after - free_before), stopped=json.dumps(previous + done))
        self._audit("acquire", job_id=job_id, code=",".join(done), duration_s=self._clock() - started)
        return self._lease_obj(self._store.lease(job_id))

    def _stop_service(self, svc: ServiceConfig, job_id: str) -> None:
        used_before, _, _ = self._probe()
        self._run(svc.stop_cmd, self.config.stop_timeout_s + 30)
        if self._wait_freed(svc, used_before):
            return
        if svc.escalate == "kill_recorded_pid" and svc.pid_file:
            rc, _, _ = self._run(self._escalate_cmd(svc), 30)
            self._audit("escalate", job_id=job_id, service=svc.name, result="ok" if rc == 0 else "error", code=str(rc))
            if self._wait_freed(svc, used_before):
                return
        raise BrokerFailed("stop_timeout", svc.name)

    def _wait_freed(self, svc: ServiceConfig, used_before: int) -> bool:
        deadline = self._clock() + self.config.stop_timeout_s
        expected = int(svc.resident_mb * 0.8)
        while True:
            used, _, _ = self._probe()
            if used_before - used >= expected and not self._detect_up(svc):
                return True
            if self._clock() >= deadline:
                return False
            self._sleep(self.config.poll_interval_s)

    @staticmethod
    def _escalate_cmd(svc: ServiceConfig) -> str:
        # SIGKILL of the recorded pid only, and only if it is ours and its cmdline matches `detect`.
        return (
            f"p=$(cat -- '{svc.pid_file}' 2>/dev/null); case \"$p\" in ''|*[!0-9]*) exit 3;; esac; "
            f"[ \"$p\" -gt 1 ] || exit 3; "
            f"[ \"$(stat -c %u /proc/$p 2>/dev/null)\" = \"$(id -u)\" ] || exit 4; "
            f"tr '\\0' ' ' < /proc/$p/cmdline | grep -Eq -- '{svc.detect}' || exit 5; "
            f"kill -9 \"$p\""
        )

    # ---------------------------------------------------------------- release / restore internals

    def _should_linger(self, window: dict[str, Any], next_needs_mb: int, now: float) -> bool:
        cfg = self.config
        if not cfg.enabled or isinstance(next_needs_mb, bool) or not isinstance(next_needs_mb, int):
            return False
        if self._max_down_at(window) - now < cfg.heartbeat_window_seconds:
            return False
        try:
            _, _, free = self._probe()
        except Exception:  # noqa: BLE001 - when unsure, restore
            return False
        reclaim = 0
        for svc in self.config.services:
            state = self._svc_state(svc)
            if state in _SHRINK_STATES:
                reclaim += svc.resident_mb - (svc.shrink_resident_mb or 0)
            elif state in _RESTORABLE:
                reclaim += svc.resident_mb
        return next_needs_mb > free - reclaim - cfg.reserve_mb

    def _housekeeping(self) -> RestoreResult | None:
        window = self._window()
        state = window["state"]
        now = self._clock()
        if state == "restore_failed":
            return None
        if state == "idle":
            if any(self._svc_state(svc) in _RESTORABLE for svc in self.config.services):
                return self._restore_window("orphaned_stop", now)
            return None
        if state in ("restoring", "guard_restoring"):
            return self._restore_window("resume_restore", now)
        if state == "lingering":
            if now >= (window.get("deadline_at") or 0) or now >= self._max_down_at(window):
                return self._restore_window("linger_expired", now)
            return None
        # state == "down"
        max_down_at = self._max_down_at(window)
        for lease in self._store.leases("active"):
            if not lease["attached"]:
                continue
            if now >= max_down_at:
                self._expire_lease(lease["job_id"], "max_down")
            elif now > lease["heartbeat_at"] + self.config.heartbeat_window_seconds:
                self._expire_lease(lease["job_id"], "heartbeat_lost")
        if any(x["attached"] for x in self._store.leases("active")):
            return None
        if now < (window.get("deadline_at") or 0) + GUARD_GRACE_S:
            return None  # the guard fires at deadline_at; give it the first chance
        return self._restore_window("lease_expired", now)

    def _restore_window(self, reason: str, started: float) -> RestoreResult:
        """Restore every service the broker stopped, in start_order. Never raises ``Exception``."""
        result = RestoreResult(ok=True, action="restored")
        for lease in self._store.leases("active"):
            if lease["attached"]:
                self._expire_lease(lease["job_id"], reason)
        targets = sorted(
            (svc for svc in self.config.services if self._svc_state(svc) in _RESTORABLE),
            key=lambda s: s.start_order,
        )
        if not targets:
            self._close_window(reason, result, started)
            return result
        was_restoring = self._window()["state"] == "restoring"
        self._store.put(WINDOW_ROW, state="restoring")
        try:
            got_lock = self._take_restore_lock(broker_crashed=was_restoring)
        except Exception as exc:  # noqa: BLE001 - remote unreachable: guard deadline still runs; retry later
            self._audit("restore", result="pending", code=f"lock:{type(exc).__name__}")
            return RestoreResult(ok=False, action="restore_pending", pending=True,
                                 errors=[f"lock:{type(exc).__name__}"], duration_s=self._clock() - started)
        if not got_lock:
            return self._adopt_guard_result(targets, reason, started)
        try:
            self._write_lease("broker_restoring", self._clock() + self._restore_budget_s())
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"lease:{type(exc).__name__}")
        for svc in targets:
            self._store.put(svc.name, state="restoring")
            ok, attempts, error, tail = self._restore_service(svc)
            result.attempts[svc.name] = attempts
            if ok:
                result.restored.append(svc.name)
                self._store.put(svc.name, state="up", job_id=None, restore_attempts=attempts, last_error=None)
            else:
                result.failed.append(svc.name)
                result.errors.append(f"{svc.name}:{error}")
                if tail:
                    result.log_tails[svc.name] = tail
                self._store.put(svc.name, state="restore_failed", restore_attempts=attempts, last_error=error)
        try:
            self._write_lease("released" if not result.failed else "restore_failed", self._clock())
            self._run(f"rmdir '{self.config.lock_path}' 2>/dev/null; true", 15)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"unlock:{type(exc).__name__}")
        if result.failed:
            self._open_circuit(reason, result, started)
        else:
            self._close_window(reason, result, started)
        return result

    def _restore_service(self, svc: ServiceConfig) -> tuple[bool, int, str | None, list[str]]:
        cfg = self.config
        state = self._svc_state(svc)
        try:
            needs_stop = state in _SHRINK_STATES
            if not needs_stop and self._detect_up(svc):
                if self._wait_healthy(svc, cfg.stop_timeout_s):
                    return True, 0, None, []  # never stopped (or already back): do not start a second copy
                needs_stop = True
            self._verify_script(svc, "start")
        except BrokerError as exc:
            return False, 0, exc.code, []
        except Exception as exc:  # noqa: BLE001
            return False, 0, f"remote:{type(exc).__name__}", []
        error: str | None = None
        tail: list[str] = []
        for attempt in range(cfg.restore_retries + 1):
            try:
                if needs_stop or attempt > 0:
                    self._run(svc.stop_cmd, cfg.stop_timeout_s + 30)
                    self._wait_gone(svc)
                self._run(svc.start_cmd, 120)
                if self._wait_healthy(svc, cfg.restore_health_timeout_s):
                    self._audit("restore_service", service=svc.name, code=f"attempt={attempt + 1}")
                    return True, attempt + 1, None, []
                error = "unhealthy_after_start"
            except Exception as exc:  # noqa: BLE001
                error = f"remote:{type(exc).__name__}"
            tail = self._log_tail(svc)
            self._audit("restore_service", service=svc.name, result="error", code=f"{error} attempt={attempt + 1}")
        return False, cfg.restore_retries + 1, error, tail

    def _wait_gone(self, svc: ServiceConfig) -> None:
        deadline = self._clock() + self.config.stop_timeout_s
        while self._detect_up(svc) and self._clock() < deadline:
            self._sleep(self.config.poll_interval_s)

    def _adopt_guard_result(self, targets: list[ServiceConfig], reason: str, started: float) -> RestoreResult:
        lease = self._read_lease()
        state = (lease or {}).get("state")
        if state not in ("restored", "restore_failed"):
            self._store.put(WINDOW_ROW, state="guard_restoring")
            return RestoreResult(ok=False, action="guard_restoring", pending=True, errors=[f"guard_state:{state}"],
                                 duration_s=self._clock() - started)
        result = RestoreResult(ok=True, action="restored_by_guard")
        for svc in targets:
            try:
                healthy = self._healthy(svc)
            except Exception:  # noqa: BLE001
                healthy = False
            if healthy:
                result.restored.append(svc.name)
                self._store.put(svc.name, state="up", job_id=None, last_error=None)
            else:
                # no broker retry on top of the guard's attempt: never loop-kill
                result.failed.append(svc.name)
                result.errors.append(f"{svc.name}:guard_{state}")
                result.log_tails[svc.name] = self._log_tail(svc)
                self._store.put(svc.name, state="restore_failed", last_error=f"guard_{state}")
        if result.failed:
            self._open_circuit(reason, result, started)
        else:
            self._close_window(reason, result, started)
        return result

    def _close_window(self, reason: str, result: RestoreResult, started: float) -> None:
        result.duration_s = self._clock() - started
        self._store.put(WINDOW_ROW, state="idle", job_id=None, last_error=None)
        if result.restored or result.action == "restored_by_guard":
            self._store.put(LAST_RESTORE_ROW, state="ok", last_error=json.dumps(
                {"reason": reason, "restored": result.restored, "action": result.action}))
        self._audit("restore", result="ok", code=f"{reason}:{','.join(result.restored)}",
                    duration_s=result.duration_s)

    def _open_circuit(self, reason: str, result: RestoreResult, started: float) -> None:
        result.ok = False
        result.action = "restore_failed"
        result.duration_s = self._clock() - started
        summary = {"reason": reason, "failed": result.failed, "restored": result.restored, "errors": result.errors}
        self._store.put(WINDOW_ROW, state="restore_failed", last_error=json.dumps(summary))
        self._store.put(LAST_RESTORE_ROW, state="restore_failed", last_error=json.dumps(summary))
        self._audit("restore", result="restore_failed", code=",".join(result.errors), duration_s=result.duration_s)
        payload = {
            "level": "critical",
            "code": "gpu_broker_restore_failed",
            "services": list(result.failed),
            "errors": list(result.errors),
            "log_tails": dict(result.log_tails),
            "message_en": "GPU broker could not restore services; circuit open, no further stops until reset(). "
                          "Manual restore: see docs/GPU_MEMORY_BROKER_20261101.md",
            "message_zh": "训练服务未能恢复，已停止继续借用，请按运维文档手动恢复后 reset()",
        }
        if self._alert_cb is not None:
            try:
                self._alert_cb(payload)
            except Exception:  # noqa: BLE001 - alerting must not break restore bookkeeping
                pass

    def _expire_lease(self, job_id: str, reason: str) -> None:
        self._store.put_lease(job_id, state="expired", cancel_reason=reason)
        self._audit("lease_expired", job_id=job_id, code=reason)

    # ---------------------------------------------------------------- remote helpers

    def _run(self, command: str, timeout: float) -> tuple[int, str, str]:
        rc, out, err = self._remote.run(command, timeout)
        return rc, out or "", err or ""

    def _probe(self) -> tuple[int, int, int]:
        rc, out, _ = self._run(GPU_QUERY_CMD, 30)
        lines = [line for line in out.strip().splitlines() if line.strip()]
        if rc != 0 or len(lines) != 1:
            raise BrokerFailed("gpu_probe_failed", f"rc={rc} lines={len(lines)}")
        try:
            used_s, total_s = lines[0].split(",")
            used, total = int(used_s.strip()), int(total_s.strip())
        except ValueError as exc:
            raise BrokerFailed("gpu_probe_failed", "unparseable nvidia-smi output") from exc
        return used, total, total - used

    def _detect_up(self, svc: ServiceConfig) -> bool:
        # "[v]llm…" so the remote shell's own command line never matches the pattern.
        index = next(i for i, ch in enumerate(svc.detect) if ch.isalnum())
        pattern = f"{svc.detect[:index]}[{svc.detect[index]}]{svc.detect[index + 1:]}"
        rc, out, _ = self._run(f"pgrep -u \"$(id -u)\" -f -- '{pattern}'", 15)
        return rc == 0 and bool(out.strip())

    def _healthy(self, svc: ServiceConfig) -> bool:
        mode, _ = svc.health
        rc, out, _ = self._run(svc.health_cmd, 20)
        if mode == "code":
            return out.strip() == svc.health_expect
        return rc == 0 and "".join(svc.health_expect.split()) in "".join(out.split())

    def _wait_healthy(self, svc: ServiceConfig, timeout_s: float) -> bool:
        deadline = self._clock() + timeout_s
        streak = 0
        while True:
            try:
                healthy = self._healthy(svc)
            except Exception:  # noqa: BLE001 - a failed probe counts as unhealthy
                healthy = False
            streak = streak + 1 if healthy else 0
            if streak >= self.config.health_consecutive_ok:
                return True
            if self._clock() >= deadline:
                return False
            self._sleep(self.config.poll_interval_s)

    def _resolve_confined(self, path: str, what: str) -> str:
        rc, out, _ = self._run(f"readlink -f -- '{path}'", 15)
        resolved = out.strip()
        if rc != 0 or not resolved:
            raise BrokerFailed("path_unresolvable", what)
        try:
            return confine_path(resolved, what)
        except BrokerConfigError as exc:
            raise BrokerFailed("path_escape", f"{what} resolves outside {REMOTE_ROOT}") from exc

    def _verify_script(self, svc: ServiceConfig, role: str) -> None:
        script = {"start": svc.start_script, "stop": svc.stop_script, "shrink": svc.shrink_script}[role]
        if script is None:
            raise BrokerFailed("script_missing", f"{svc.name}.{role}")
        self._resolve_confined(script, f"{svc.name}.{role}")
        rc, out, _ = self._run(f"sha256sum -- '{script}'", 30)
        digest = out.split()[0] if rc == 0 and out.split() else ""
        if not _SHA_RE.fullmatch(digest):
            raise BrokerFailed("script_unreadable", f"{svc.name}.{role}")
        expected = svc.script_sha256.get(role, FILL_AT_INSTALL)
        if expected == FILL_AT_INSTALL:
            pinned = self._store.pin(script)
            if pinned is None:
                self._store.set_pin(script, digest)
                self._audit("script_pinned", service=svc.name, code=role)
                pinned = digest
            expected = pinned
        if digest != expected:
            self._audit("script_changed", service=svc.name, result="refused", code=role)
            raise BrokerFailed("script_changed", f"{svc.name}.{role} sha256 differs from the pinned value")

    def _log_tail(self, svc: ServiceConfig) -> list[str]:
        if not svc.log_file:
            return []
        try:
            self._resolve_confined(svc.log_file, f"{svc.name}.log_file")
            _, out, _ = self._run(f"tail -n 40 -- '{svc.log_file}'", 15)
        except Exception:  # noqa: BLE001 - diagnostics only
            return []
        return _redact(out.splitlines()[-40:])

    def _write_remote(self, path: str, content: str, *, unless_locked: bool = False) -> bool:
        if _HEREDOC in content or not content.endswith("\n"):
            raise BrokerFailed("remote_write_invalid", path)
        body = f"cat > '{path}.tmp' <<'{_HEREDOC}'\n{content}{_HEREDOC}\nmv -f '{path}.tmp' '{path}'"
        if unless_locked:
            body = f"if [ -d '{self.config.lock_path}' ]; then echo LOCKED; else\n{body}\nfi"
        rc, out, _ = self._run(body, 30)
        if unless_locked and out.strip() == "LOCKED":
            return False
        if rc != 0:
            raise BrokerFailed("remote_write_failed", path)
        return True

    def _install_guard(self, *, clear_lock: bool) -> None:
        cfg = self.config
        rc, _, _ = self._run(f"mkdir -p '{cfg.broker_dir}' && chmod 700 '{cfg.broker_dir}'", 15)
        if rc != 0:
            raise BrokerFailed("guard_dir_failed")
        if clear_lock:  # a previous window is fully closed (DB says idle): drop its stale lock dir
            self._run(f"rmdir '{cfg.lock_path}' 2>/dev/null; true", 15)
        self._write_remote(cfg.guard_path, GUARD_SCRIPT)
        rc, out, _ = self._run(f"sha256sum -- '{cfg.guard_path}'", 15)
        expected = hashlib.sha256(GUARD_SCRIPT.encode("utf-8")).hexdigest()
        if rc != 0 or not out.split() or out.split()[0] != expected:
            raise BrokerFailed("guard_upload_failed")

    def _launch_guard(self) -> None:
        cfg = self.config
        rc, _, _ = self._run(
            f"cd '{cfg.broker_dir}' && if [ -f guard.pid ] && kill -0 \"$(cat guard.pid)\" 2>/dev/null; "
            f"then echo RUNNING; else setsid nohup bash '{cfg.guard_path}' >/dev/null 2>&1 < /dev/null & "
            f"echo LAUNCHED; fi",
            15,
        )
        if rc != 0:
            raise BrokerFailed("guard_launch_failed")

    def _write_lease(self, state: str, deadline: float, *, unless_locked: bool = False, kill: bool = True) -> bool:
        cfg = self.config
        window = self._window()
        start = window.get("stopped_at") or self._clock()
        max_down = start + cfg.max_down_seconds
        if state == "active":
            deadline = min(deadline, max_down)
        budget = self._restore_budget_s()
        services = sorted(
            (svc for svc in cfg.services if self._svc_state(svc) in _RESTORABLE + ("restore_failed",)),
            key=lambda s: s.start_order,
        )
        doc = {
            "version": 1,
            "state": state,
            "job": _job_hash(window.get("job_id")),
            "started_epoch": int(start),
            "deadline_epoch": int(deadline),
            "max_down_epoch": int(max_down),
            "hard_exit_epoch": int(max(deadline, max_down) + budget + 3600),
            "restore_budget_s": int(budget),
            "health_timeout_s": int(cfg.restore_health_timeout_s),
            "pre_stop": [svc.stop_cmd for svc in services if self._svc_state(svc) in _SHRINK_STATES],
            "restore_order": [svc.start_cmd for svc in services],
            "restore_sha256": [self._start_sha256(svc) for svc in services],
            "health": [f"{svc.health[1]} {''.join(svc.health_expect.split())}" for svc in services],
            "kill_patterns": list(cfg.trainer_patterns) if kill else [],
        }
        return self._write_remote(cfg.lease_path, lease_document_text(doc), unless_locked=unless_locked)

    def _start_sha256(self, svc: ServiceConfig) -> str:
        """Configured or first-use pinned hash of the start script; the guard refuses a changed one."""
        expected = svc.script_sha256.get("start", FILL_AT_INSTALL)
        return (self._store.pin(svc.start_script) or "") if expected == FILL_AT_INSTALL else expected

    def _read_lease(self) -> dict[str, Any] | None:
        try:
            rc, out, _ = self._run(f"cat '{self.config.lease_path}'", 15)
            return json.loads(out) if rc == 0 else None
        except Exception:  # noqa: BLE001
            return None

    def _take_restore_lock(self, *, broker_crashed: bool) -> bool:
        lock = self.config.lock_path
        rc, out, _ = self._run(f"mkdir '{lock}' 2>/dev/null && echo TAKEN || stat -c %Y '{lock}'", 15)
        out = out.strip()
        if out == "TAKEN":
            return True
        age = self._clock() - int(out) if out.isdigit() else 0
        lease_state = (self._read_lease() or {}).get("state")
        # Steal only from ourselves (a broker that died mid-restore) or from a lock older than the budget.
        if (broker_crashed and lease_state == "broker_restoring") or age > self._restore_budget_s():
            rc, out, _ = self._run(f"rmdir '{lock}' 2>/dev/null; mkdir '{lock}' && echo TAKEN", 15)
            return out.strip() == "TAKEN"
        return False

    # ---------------------------------------------------------------- misc

    def _window(self) -> dict[str, Any]:
        return self._store.row(WINDOW_ROW) or {"state": "idle", "job_id": None, "stopped_at": None,
                                               "deadline_at": None}

    def _max_down_at(self, window: dict[str, Any]) -> float:
        start = window.get("stopped_at")
        return (start if start is not None else self._clock()) + self.config.max_down_seconds

    def _svc_state(self, svc: ServiceConfig) -> str:
        row = self._store.row(svc.name)
        return row["state"] if row else "up"

    def _lease_obj(self, row: dict[str, Any] | None) -> MemoryLease:
        if row is None:
            raise BrokerFailed("lease_missing")
        window = self._window()
        return MemoryLease(
            job_id=row["job_id"],
            request_mb=row["request_mb"],
            freed_mb=row["freed_mb"],
            stopped=json.loads(row["stopped"] or "[]"),
            free_mb_at_grant=row["free_mb_at_grant"],
            granted_at=row["granted_at"],
            max_down_at=self._max_down_at(window) if row["attached"] and window.get("stopped_at") else None,
            attached=bool(row["attached"]),
        )

    def _restore_budget_s(self) -> int:
        cfg = self.config
        per_service = (cfg.restore_retries + 1) * (cfg.restore_health_timeout_s + cfg.stop_timeout_s + 150)
        return max(1, len(cfg.services)) * per_service + 300

    def _op_budget_s(self) -> int:
        cfg = self.config
        return max(1, len(cfg.services)) * (2 * cfg.stop_timeout_s + cfg.restore_health_timeout_s + 120) \
            + self._restore_budget_s() + 600

    @contextmanager
    def _exclusive(self, wait_s: float | None = None) -> Iterator[None]:
        wait = self.config.lock_wait_s if wait_s is None else wait_s
        if not self._lock.acquire(timeout=wait if wait > 0 else 0.001):
            raise BrokerBusy("broker_busy", "another broker operation is running")
        try:
            owner = _lock_owner()
            started = self._clock()
            while not self._store.try_lock(owner, self._op_budget_s()):
                if self._clock() - started >= wait:
                    raise BrokerBusy("broker_busy", "state lock held by another broker process")
                self._sleep(0.5)
            if self._store.took_over:
                self._audit("lock_takeover", code="owner_process_dead")
            try:
                yield
            finally:
                self._store.unlock(owner)
        finally:
            self._lock.release()

    def _audit(
        self, action: str, *, job_id: str | None = None, service: str | None = None, result: str = "ok",
        duration_s: float | None = None, code: str | None = None,
    ) -> None:
        record = {
            "ts": datetime.fromtimestamp(self._clock(), timezone.utc).isoformat(timespec="seconds"),
            "action": action,
            "job_id": job_id,
            "service": service,
            "result": result,
            "duration_s": round(duration_s, 3) if duration_s is not None else None,
            "code": code,
        }
        try:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass
        if self._audit_cb is not None:
            try:
                self._audit_cb(record)
            except Exception:  # noqa: BLE001
                pass
