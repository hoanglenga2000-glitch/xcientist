"""t238 C7c ADDENDUM 2: admission → GPU memory broker.

When a queued job needs more than ``probe_free - reserve`` and the broker is enabled, the
broker stops (then later restores) the owner's co-resident services so the job can run.
Everything about those services stays operator-side: tenants only ever see
「正在准备训练…」 or the honest queued line.

* ``open_broker`` builds the broker from the operator config (``gpu_broker_config_path``) with
  the production SSH executor, or returns None (disabled / misconfigured → no broker).
* ``acquire`` runs in the runner thread that holds the job (never in an HTTP request thread);
  the job is dispatched only after it returns.
* ``release_in_background`` restores after the job without delaying the tenant's results,
  keeping the services down only for the broker's linger and only if the next queued job
  needs the memory (``next_needs_mb``).
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

from . import gpu_vram
from .gpu_memory_broker import (
    _RESTORABLE, BrokerBusy, BrokerConfig, BrokerError, BrokerLeaseExpired, BrokerRefused, GpuMemoryBroker,
)

STATE_DB = "gpu_broker_state.sqlite3"
# Tenant copy: never names services, models or card sizes.
FREEING_LINE = "正在准备训练…"
SHORT_LINE = "排队中，轮到你就开始。"
STOPPED_LINE = "这次训练到了时间上限，已停止。用掉的时长按实际计算。"

# After exit status: sshd can send it before the last output, and a background service may never close stdout.
EOF_GRACE_SECONDS = 1.5

BrokerFactory = Callable[[Path, "gpu_vram.VramConfig"], "GpuMemoryBroker | None"]


class SshRemote:
    """``RemoteExec`` over the identity-gated connector; one short-lived client per command."""

    def __init__(self, connect: Callable[[], Any]) -> None:
        self._connect = connect

    def run(self, command: str, timeout: float) -> tuple[int, str, str]:
        """Ends with the command plus a short bounded wait for EOF, so the last output is not lost."""

        client = self._connect()
        try:
            _stdin, stdout, _stderr = client.exec_command(command, timeout=timeout)
            channel = stdout.channel
            out, err = bytearray(), bytearray()
            deadline = time.monotonic() + float(timeout)
            grace_until: float | None = None
            while True:
                exited = channel.exit_status_ready()
                now = time.monotonic()
                if exited and grace_until is None:
                    grace_until = now + EOF_GRACE_SECONDS
                # Decided before the drain, so nothing that arrived before EOF is left behind.
                finished = exited and (channel.eof_received or channel.closed or now >= grace_until)
                while channel.recv_ready():
                    out.extend(channel.recv(65536))
                while channel.recv_stderr_ready():
                    err.extend(channel.recv_stderr(65536))
                if finished:
                    break
                if not exited and now > deadline:
                    raise TimeoutError("remote_command_timeout")
                time.sleep(0.05)
            return channel.recv_exit_status(), out.decode("utf-8", "replace"), err.decode("utf-8", "replace")
        finally:
            client.close()


def _operator_audit(runtime_root: Path) -> Callable[[dict[str, Any]], None]:
    path = Path(runtime_root) / "audit" / "gpu-broker-link.jsonl"

    def write(event: dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"at": time.time(), **event}, sort_keys=True, default=str) + "\n")
        except OSError:
            pass

    return write


def open_broker(runtime_root: Path, vram: "gpu_vram.VramConfig", *, connect: Callable[[], Any] | None = None,
                clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep) -> GpuMemoryBroker | None:
    """None unless ``gpu_broker_enabled`` AND the broker config loads AND its own ``enabled`` is true."""

    if not vram.gpu_broker_enabled or not vram.gpu_broker_config_path or connect is None:
        return None
    try:
        config = BrokerConfig.from_json(Path(vram.gpu_broker_config_path))
    except (OSError, BrokerError, ValueError):
        _operator_audit(runtime_root)({"kind": "broker_config_rejected"})
        return None
    if not config.enabled:
        return None
    audit = _operator_audit(runtime_root)
    return GpuMemoryBroker(config, SshRemote(connect), Path(runtime_root) / STATE_DB, clock=clock, sleep=sleep,
                           audit=audit, alert=lambda event: audit({"kind": "broker_alert", **event}))


def planned(broker: GpuMemoryBroker | None, request_mb: int, free_mb: int) -> list[str] | None:
    """The broker's stop plan, or None when it cannot help (disabled, refused, circuit open, error)."""

    if broker is None:
        return None
    try:
        steps = broker.plan(int(request_mb), int(free_mb))
    except BrokerError:
        return None
    except Exception:  # noqa: BLE001 - an unreachable host means "cannot help"
        return None
    return steps or None


def freeing_minutes(vram: "gpu_vram.VramConfig") -> int:
    return max(1, -(-int(vram.gpu_broker_stop_estimate_seconds) // 60))


def release_in_background(broker: GpuMemoryBroker, job_id: str, next_needs_mb: int | None,
                          *, wait: bool = False) -> threading.Thread:
    """Restore after the job without blocking the caller (``wait`` for deterministic tests)."""

    def run() -> None:
        try:
            broker.release(job_id, next_needs_mb=next_needs_mb)
        except Exception:  # noqa: BLE001 - restore trouble is the broker's circuit/alert, never the tenant's
            pass

    thread = threading.Thread(target=run, name=f"gpu-broker-release-{job_id[-8:]}", daemon=True)
    thread.start()
    if wait:
        thread.join()
    return thread


def local_state(runtime_root: Path) -> dict[str, Any]:
    """Operator health: broker window + service states from the local state DB (read-only, no remote)."""

    import sqlite3

    path = Path(runtime_root) / STATE_DB
    try:
        vram = gpu_vram.load(runtime_root)
        enabled = vram.gpu_broker_enabled
    except gpu_vram.VramConfigError:
        enabled = False
    if not path.is_file():
        return {"enabled": enabled, "window": "idle", "services": {}, "active_leases": 0}
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=0.2)
    try:
        rows = {str(row[0]): str(row[1]) for row in connection.execute("SELECT service, state FROM gpu_broker_state")}
        leases = connection.execute("SELECT COUNT(*) FROM gpu_broker_leases WHERE state='active'").fetchone()[0]
    except sqlite3.Error:
        return {"enabled": enabled, "window": "unknown", "services": {}, "active_leases": 0}
    finally:
        connection.close()
    window = rows.pop("__window__", "idle")
    rows.pop("__last_restore__", None)
    return {"enabled": enabled, "window": window, "circuit_open": window == "restore_failed",
            "services": rows, "active_leases": int(leases)}


def services_returning(runtime_root: Path) -> bool:
    """Local check: a restore is running or pending, or the state cannot be read (fail closed).

    ``restore_failed`` is not "returning": it stays for the operator, and admission treats it as it did.
    """

    state = local_state(runtime_root)
    if state["window"] == "unknown" or state["window"] in {"restoring", "guard_restoring"}:
        return True
    if state["window"] in {"down", "lingering", "restore_failed"}:
        return False  # lent on purpose (jobs join through the broker), or left to the operator
    return any(value in _RESTORABLE for value in state["services"].values())


__all__ = ["BrokerBusy", "BrokerLeaseExpired", "BrokerRefused", "FREEING_LINE", "SHORT_LINE", "STOPPED_LINE",
           "SshRemote", "freeing_minutes", "open_broker", "planned", "release_in_background"]
