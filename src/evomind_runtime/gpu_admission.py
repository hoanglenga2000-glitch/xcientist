"""Admission gate between HPC training tools and the GPU time-share scheduler.

Supersedes the host-local hot-patch of the same module name. The scheduler is
active only when BOTH hold:

* the operator switch is on (env ``EVOMIND_GPU_SCHEDULER=1`` or the flag file
  ``C:\\ProgramData\\EvoMind\\state\\gpu-scheduler.enabled``), and
* the pool config (``EVOMIND_GPU_POOL_CONFIG`` or
  ``C:\\ProgramData\\EvoMind\\state\\gpu-pool.json``) carries a valid ``gpu_pool``
  block for node ``job98270``.

Otherwise every caller keeps the legacy single-binding behaviour unchanged.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from research_os.remote_paths import HPC_HOME_ROOT, TRAINER_MARKER

from .gpu_scheduler import (
    ALLOWED_HPC_JOB_ID, ALLOWED_NODE_ID, CANCELLED, FAILED, FREEING, QUEUED, RUNNING, SUCCEEDED, TERMINAL, TIMEOUT,
    GpuPolicy, GpuScheduler, SchedulerError,
)

DB_NAME = "gpu_scheduler.sqlite3"
_DEFAULT_FLAG = r"C:\ProgramData\EvoMind\state\gpu-scheduler.enabled"
_DEFAULT_POOL_CONFIG = r"C:\ProgramData\EvoMind\state\gpu-pool.json"


def enabled() -> bool:
    """Gate switch: env var wins; otherwise the operator-owned flag file."""

    value = os.environ.get("EVOMIND_GPU_SCHEDULER", "").strip()
    if value:
        return value == "1"
    return Path(os.environ.get("EVOMIND_GPU_SCHEDULER_FLAG", _DEFAULT_FLAG)).is_file()


def pool_config() -> dict[str, Any] | None:
    """The validated ``gpu_pool`` block, or None (absent or invalid → legacy behaviour)."""

    path = Path(os.environ.get("EVOMIND_GPU_POOL_CONFIG", _DEFAULT_POOL_CONFIG))
    try:
        if not path.is_file():
            return None
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        block = document.get("gpu_pool") if isinstance(document, dict) else None
        if not isinstance(block, dict):
            return None
        if block.get("node_id") != ALLOWED_NODE_ID or block.get("hpc_job_id") != ALLOWED_HPC_JOB_ID:
            return None
        tenant = str(block.get("service_tenant_id") or "")
        if not tenant.startswith("tenant_") or len(tenant) != 31:
            return None
        policy = GpuPolicy.from_mapping(block.get("policy") if isinstance(block.get("policy"), dict) else {})
    except (OSError, ValueError, SchedulerError):
        return None
    return {**block, "policy_object": policy, "tenant_training": block.get("tenant_training") is True}


def active() -> dict[str, Any] | None:
    """The pool config when the scheduler governs HPC training, else None (legacy)."""

    if not enabled():
        return None
    return pool_config()


def scheduler(runtime_root: Path, *, config: dict[str, Any] | None = None, clock: Callable[[], float] | None = None) -> GpuScheduler:
    config = config if config is not None else pool_config()
    policy = (config or {}).get("policy_object") or GpuPolicy()
    kwargs = {"clock": clock} if clock is not None else {}
    return GpuScheduler(Path(runtime_root) / DB_NAME, policy=policy, **kwargs)


def run_gpu_cancelled(runtime_root: Path, run_id: str) -> bool:
    """C7e-R1: True when the run's LATEST GPU job was cancelled (a later retry job clears it).

    Read-only and never raises: no scheduler DB, an unreadable one or an unknown run all mean "not cancelled", so
    runs that never used the time-shared card are unaffected.
    """

    path = Path(runtime_root) / DB_NAME
    if not path.is_file():
        return False
    try:
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)
        try:
            row = db.execute("SELECT status FROM train_jobs WHERE run_id=? ORDER BY seq DESC LIMIT 1", (str(run_id),)).fetchone()
        finally:
            db.close()
    except sqlite3.Error:
        return False
    return bool(row and row[0] == CANCELLED)


def startup_reconcile(runtime_root: Path, *, process_alive: Callable[[dict], "bool | None"] | None = None,
                      clock: Callable[[], float] | None = None, broker: Any = None,
                      broker_wait_s: float | None = None) -> dict[str, list] | None:
    """Expire stale leases, repair orphans, keep the queue; restore what the broker left stopped.

    No-op (None) when the scheduler is not active, so the legacy start-up is unchanged. The service
    calls it through ``reconcile_in_background``: a restore takes minutes and must never hold the port.
    """

    config = active()
    if config is None:
        return None
    sched = scheduler(runtime_root, config=config, clock=clock)
    try:
        connect = _service_connector()
        if connect is not None:
            sched.orphan_killer = ssh_orphan_killer(connect)
            process_alive = process_alive or ssh_process_alive(connect)
        report = sched.reconcile(process_alive)
        # ADDENDUM 2: restore services the broker paused before a crash when no live job holds them.
        try:
            broker = broker if broker is not None else _startup_broker(runtime_root)
            if broker is not None:
                result = broker.reconcile() if broker_wait_s is None else broker.reconcile(wait_s=broker_wait_s)
                report["broker"] = [result.action] if result is not None else []
        except Exception as error:  # noqa: BLE001 - operator audit only
            report["broker"] = ["error:" + type(error).__name__]
        sched.audit("startup_reconcile", detail={key: len(value) for key, value in report.items()})
        return report
    finally:
        sched.close()


STARTUP_BROKER_WAIT_SECONDS = 5.0


def reconcile_in_background(runtime_root: Path, *, retry_seconds: float = 30.0, max_attempts: int = 480,
                            reconcile: Callable[..., "dict[str, list] | None"] | None = None) -> threading.Thread:
    """Start-up reconcile off the serving path, with a short lock wait; retried until the broker settles.

    A busy lock, an unreachable host or a pending restore are retried; anything else ends the thread.
    """

    def run() -> None:
        action = reconcile or startup_reconcile
        for _attempt in range(max_attempts):
            try:
                report = action(runtime_root, broker_wait_s=STARTUP_BROKER_WAIT_SECONDS)
            except Exception as error:  # noqa: BLE001 - a scheduler DB error must not stop the service
                report = {"broker": ["error:" + type(error).__name__]}
            outcome = (report or {}).get("broker") or []
            if not any(item == "restore_pending" or str(item).startswith("error:") for item in outcome):
                return
            time.sleep(retry_seconds)

    thread = threading.Thread(target=run, name="gpu-startup-reconcile", daemon=True)
    thread.start()
    return thread


BROKER_HOUSEKEEPING_SECONDS = 60.0


def broker_housekeeping_tick(runtime_root: Path, *, broker: Any = None) -> str | None:
    """One restore retry while the hold applies. Admission never reaches the broker during the hold, so
    this is the retry after a failed SSH call and what adopts a restore the guard finished.

    Returns the broker's action, or None when nothing is held (a local read, no remote call).
    """

    from . import gpu_broker_link, gpu_vram

    if active() is None:
        return None
    try:
        vram = gpu_vram.load(runtime_root)
    except gpu_vram.VramConfigError:
        return None
    if not vram.gpu_broker_enabled or not gpu_broker_link.services_returning(runtime_root):
        return None
    broker = broker if broker is not None else _startup_broker(runtime_root)
    if broker is None:  # the hold stays and the service is up: only the operator log can say why
        gpu_broker_link._operator_audit(runtime_root)({"kind": "broker_housekeeping_no_broker"})
        return "no_broker"
    result = broker.reconcile(wait_s=STARTUP_BROKER_WAIT_SECONDS)
    return result.action if result is not None else "none"


def broker_housekeeping_in_background(runtime_root: Path, *, seconds: float = BROKER_HOUSEKEEPING_SECONDS,
                                      tick: Callable[[Path], Any] | None = None,
                                      stop: threading.Event | None = None) -> threading.Thread:
    """Runs for the life of the service, so the hold has no end point the start-up retry could run out at."""

    stop = stop or threading.Event()

    def run() -> None:
        from .gpu_broker_link import _operator_audit

        action = tick or broker_housekeeping_tick
        while not stop.wait(seconds):
            try:
                action(runtime_root)
            except Exception as error:  # noqa: BLE001 - busy lock or host down: the next round retries
                _operator_audit(runtime_root)({"kind": "broker_housekeeping_failed", "code": type(error).__name__})

    thread = threading.Thread(target=run, name="gpu-broker-housekeeping", daemon=True)
    thread.start()
    return thread


def _startup_broker(runtime_root: Path) -> Any:
    """Production broker for start-up reconcile and the operator status (None when disabled)."""

    from . import gpu_broker_link, gpu_vram

    try:
        vram = gpu_vram.load(runtime_root)
    except gpu_vram.VramConfigError:
        return None
    if not vram.gpu_broker_enabled:
        return None
    return gpu_broker_link.open_broker(runtime_root, vram, connect=_service_connector())


def _service_connector() -> Callable[[], Any] | None:
    """The identity-gated connector for the service binding (job 98270), or None if unavailable."""

    try:
        from .tenant_training import service_identity
        from .tools import _load_bound_hpc_config
        from research_agent_workstation.server.core.gpu_credentials import connect_ssh

        config = pool_config()
        if config is None:
            return None
        identity = service_identity(config)
        bound = _load_bound_hpc_config(identity["credential_profile"], identity["job_id"], "gpu_broker")
        return lambda: connect_ssh(bound, timeout=30)
    except Exception:  # noqa: BLE001 - no broker without a verified binding
        return None


def cancel_run(runtime_root: Path, run_id: str) -> list[str]:
    """A stopped Run stops its GPU job, queued or running. No-op without the scheduler."""

    if active() is None or not (Path(runtime_root) / DB_NAME).exists():
        return []
    sched = scheduler(runtime_root)
    try:
        return sched.cancel_run(run_id)
    finally:
        sched.close()


def node_id_for_job(job_id: int) -> str:
    return f"job{int(job_id)}"


# -- tenant-facing copy (Chinese only, 「你」, no codes, hosts, paths or job ids) --
MESSAGES = {
    "breaker_open": "训练暂时用不了，你的排队位置还在。恢复后在这个任务里重新发起就行。",
    "pool_unavailable": "训练暂时不可用，请稍后再试。",
    "binding_invalid": "训练暂时不可用，请稍后再试。",
    "quota_exhausted": "你今天的训练时长已用完，明天可以继续训练。",
    "quota_exhausted_rolling": "你最近一段时间的训练时长已用完，过几天可以继续训练。",
    "queue_full": "现在排队的训练太多，请等前面的训练结束后再提交。",
    # After the wait ends nothing dispatches by itself: only a new request in the same task resumes the place.
    "insufficient_gpu_memory": "现在训练排满了，这次训练还没有开始。请稍后在这个任务里重新发起。",
    "queue_wait_exceeded": "排队时间较长，这次训练还没有开始。请稍后在这个任务里重新发起。",
    "queue_held": "这次训练没有在排队，也不会自动开始。请在这个任务里重新发起。",
    "tenant_busy": "你已有一个训练在进行中，等它结束后再开始新的训练。",
    "timeout": "这次训练超过了最长时间，已自动停止。可以缩小数据后重新开始。",
    "cancelled": "训练已取消。",
    "lost": "训练中断了，用掉的时长按实际计算。请重新开始训练。",
    "rejected": "你今天的训练时长已用完，明天可以继续训练。",
    "gpu_request_invalid": "训练请求无效，请检查训练设置后再试。",
    "path_rejected": "训练数据的位置不符合要求，已停止。请把数据上传到这个任务后再试。",
    # t238 C7c ADDENDUM 1: never names card sizes or other services.
    "request_too_large": "这次训练太大，换小一点的数据再试。",
    "vram_config_invalid": "训练暂时不可用，请稍后再试。",
    # ADDENDUM 2 (memory broker): tenant copy never names the services that are paused.
    "gpu_memory_short": "排队中，轮到你就开始。",
    "broker_window_expired": "这次训练到了时间上限，已停止。用掉的时长按实际计算。",
}


def message(code: str) -> str:
    return MESSAGES.get(code, MESSAGES["pool_unavailable"])


class GpuAdmissionError(RuntimeError):
    def __init__(self, code: str, *, position: int | None = None, eta_seconds: int | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.position = position
        self.eta_seconds = eta_seconds

    def public(self) -> dict[str, Any]:
        # breaker_open before any job exists: there is no place to keep, so the copy must not claim one.
        text = message("pool_unavailable" if self.code == "breaker_open" and self.position is None else self.code)
        return {"schema": "evomind.gpu_admission.v2", "granted": False, "code": self.code, "queue_position": self.position,
                "eta_seconds": self.eta_seconds, "eta_is_estimate": self.eta_seconds is not None, "message": text}


# -- Q3 health probe + circuit breaker --

@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    free_vram_mb: int | None = None


Probe = Callable[[], ProbeResult]

_BREAKER_SCHEMA = """
CREATE TABLE IF NOT EXISTS gpu_breaker (
  id INTEGER PRIMARY KEY CHECK (id = 1), failures INTEGER NOT NULL DEFAULT 0, opened_at REAL,
  open_until REAL, backoff_seconds REAL NOT NULL DEFAULT 0, updated_at REAL NOT NULL
);
"""


class CircuitBreaker:
    """Shared (SQLite) breaker: ``threshold`` consecutive probe failures open it; after the
    backoff one half-open probe is allowed; success closes it, failure doubles the backoff."""

    def __init__(self, sched: GpuScheduler, *, threshold: int = 3, base_backoff_seconds: float = 30.0,
                 max_backoff_seconds: float = 600.0) -> None:
        self.sched = sched
        self.threshold = threshold
        self.base = base_backoff_seconds
        self.maximum = max_backoff_seconds
        sched._db().executescript(_BREAKER_SCHEMA)

    def _row(self, db) -> dict:
        row = db.execute("SELECT * FROM gpu_breaker WHERE id=1").fetchone()
        return dict(row) if row else {"failures": 0, "opened_at": None, "open_until": None, "backoff_seconds": 0.0}

    def state(self) -> str:
        row = self._row(self.sched._db())
        if row["open_until"] is None:
            return "closed"
        return "half_open" if self.sched.clock() >= float(row["open_until"]) else "open"

    def allow_probe(self) -> bool:
        return self.state() != "open"

    def record(self, ok: bool) -> str:
        now = self.sched.clock()
        with self.sched._tx() as db:
            row = self._row(db)
            if ok:
                failures, opened, until, backoff = 0, None, None, 0.0
                if row["open_until"] is not None:
                    self.sched.audit("breaker_closed", db=db)
            else:
                failures = int(row["failures"]) + 1
                opened, until, backoff = row["opened_at"], row["open_until"], float(row["backoff_seconds"] or 0.0)
                if row["open_until"] is not None:  # a failed half-open probe
                    backoff = min(self.maximum, max(self.base, backoff * 2))
                    opened, until = now, now + backoff
                elif failures >= self.threshold:
                    backoff = self.base
                    opened, until = now, now + backoff
                    self.sched.audit("breaker_opened", detail={"failures": failures}, db=db)
            db.execute("INSERT INTO gpu_breaker(id,failures,opened_at,open_until,backoff_seconds,updated_at) VALUES(1,?,?,?,?,?) "
                       "ON CONFLICT(id) DO UPDATE SET failures=excluded.failures,opened_at=excluded.opened_at,"
                       "open_until=excluded.open_until,backoff_seconds=excluded.backoff_seconds,updated_at=excluded.updated_at",
                       (failures, opened, until, backoff, now))
        return self.state()


class HealthMonitor:
    """Read-only probe behind the breaker, cached so waiters do not hammer the gateway."""

    def __init__(self, breaker: CircuitBreaker, probe: Probe, *, cache_seconds: float = 30.0) -> None:
        self.breaker = breaker
        self.probe = probe
        self.cache_seconds = cache_seconds
        self._cached: tuple[float, ProbeResult] | None = None

    def check(self) -> tuple[str, int | None]:
        """("ok", free MB) | ("open", None) | ("failed", None)."""

        now = self.breaker.sched.clock()
        if self._cached is not None and now - self._cached[0] < self.cache_seconds and self._cached[1].ok:
            return "ok", self._cached[1].free_vram_mb
        if not self.breaker.allow_probe():
            return "open", None
        try:
            result = self.probe()
        except Exception:  # any transport/identity failure counts as unhealthy
            result = ProbeResult(False)
        state = self.breaker.record(bool(result.ok))
        self._cached = (now, result)
        if result.ok:
            return "ok", result.free_vram_mb
        return ("open" if state == "open" else "failed"), None


def ssh_free_vram_probe(connect: Callable[[], Any], expected_gpu_uuid: str = "") -> Probe:
    """Production probe: the identity-gated connector, then one read-only nvidia-smi query.

    ``connect`` is the existing ``gc.connect_ssh`` closure (it fails closed on any identity
    mismatch); nothing is written on the remote.
    """

    def probe() -> ProbeResult:
        client = connect()
        try:
            _stdin, stdout, _stderr = client.exec_command(
                "nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits", timeout=20)
            text = stdout.read().decode("utf-8", "replace")
            if stdout.channel.recv_exit_status() != 0:
                return ProbeResult(False)
            rows = [[part.strip() for part in line.split(",")] for line in text.splitlines() if line.strip()]
            wanted = {item.strip().casefold() for item in str(expected_gpu_uuid or "").split(",") if item.strip()}
            rows = [row for row in rows if len(row) == 2 and (not wanted or row[0].casefold() in wanted)]
            if len(rows) != 1 or not rows[0][1].isdigit():
                return ProbeResult(False)
            return ProbeResult(True, int(rows[0][1]))
        finally:
            client.close()

    return probe


# -- remote process check + orphan cleanup: only this user's processes carrying this job's marker --
_JOB_ID = re.compile(r"gjob_[0-9a-f]{32}")


def _marker_pattern(job: dict) -> str | None:
    """``[e]`` keeps the remote shell running this very command from matching itself."""

    job_id = str(job.get("id") or "")
    if not _JOB_ID.fullmatch(job_id):
        return None
    return shlex.quote(f"[{TRAINER_MARKER[0]}]{TRAINER_MARKER[1:]}={job_id} {HPC_HOME_ROOT}/")


def _remote_exit_code(connect: Callable[[], Any], command: str, timeout: int) -> int:
    """Bounded: ``exec_command(timeout=)`` does not cover the wait for the exit status."""

    from .gpu_broker_link import SshRemote

    return int(SshRemote(connect).run(command, timeout)[0])


def ssh_process_alive(connect: Callable[[], Any]) -> Callable[[dict], "bool | None"]:
    """True / False from the remote process table; None = unknown (never expires a lease by itself)."""

    def alive(job: dict) -> bool | None:
        pattern = _marker_pattern(job)
        if pattern is None:
            return None
        try:
            code = _remote_exit_code(connect, f'pgrep -u "$(id -u)" -f -- {pattern} >/dev/null', 20)
        except Exception:  # noqa: BLE001
            return None
        return {0: True, 1: False}.get(code)

    return alive


def ssh_orphan_killer(connect: Callable[[], Any]) -> Callable[[dict], bool]:
    """TERM, then KILL, then confirm nothing with this job's marker is left."""

    def kill(job: dict) -> bool:
        pattern = _marker_pattern(job)
        if pattern is None:
            return True  # a legacy row was never given a marker
        matcher = f'-u "$(id -u)" -f -- {pattern}'
        command = f"pkill -TERM {matcher}; sleep 5; pkill -KILL {matcher}; sleep 1; ! pgrep {matcher} >/dev/null"
        return _remote_exit_code(connect, command, 40) == 0

    return kill


# -- admission + lease supervision (Q7 cancel / timeout loop) --

class LeaseHandle:
    """A granted lease. ``tick`` heart-beats and decides cancel/timeout; ``release`` books it.

    Kill policy: a cancel kills at once; at the deadline the remote ``timeout`` wrapper gets
    ``kill_grace_seconds`` to stop the job by itself (graceful), then ``kill`` runs. A kill
    that hangs longer than ``kill_timeout_seconds`` does not hold the lease: release marks it
    ``remote_kill_unconfirmed`` and reconcile frees it even if this runner died.
    """

    def __init__(self, sched: GpuScheduler, job: dict, lease: dict, *, heartbeat_seconds: float) -> None:
        self.sched = sched
        self.job_id = str(job["id"])
        self.tenant_id = str(job["tenant_id"])
        self.lease_id = str(lease["id"])
        self.deadline_at = float(job["deadline_at"])
        self.max_run_seconds = int(job["max_run_seconds"])
        self.heartbeat_seconds = heartbeat_seconds
        self.kill: Callable[[], Any] | None = None
        # C7e-R1: marker-based remote kill (``ssh_orphan_killer``); run after ``kill`` once a settled job's worker must stop.
        self.orphan_kill: Callable[[dict], bool] | None = None
        self.cancel_poll_seconds = 2.0  # how soon a settled cancel is noticed (the heartbeat stays at ``heartbeat_seconds``)
        self.outcome = ""  # "", "cancel", "timeout"
        self.kill_started_at: float | None = None
        self.kill_done = threading.Event()
        self._kill_thread: threading.Thread | None = None
        # ADDENDUM 2: memory broker lease (None when no services were paused for this job).
        self.broker: Any = None
        self.broker_wait_release = False
        self.broker_last_beat: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.released: dict | None = None

    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline_at - self.sched.clock())

    def _start_kill(self, sched: GpuScheduler) -> None:
        if self.kill_started_at is not None:
            return
        self.kill_started_at = sched.clock()
        sched.mark_kill_requested(self.job_id)
        sched.audit("kill_started", tenant_id=self.tenant_id, job_id=self.job_id, detail={"outcome": self.outcome})
        kill = self.kill

        def run() -> None:
            try:
                if kill is not None:
                    kill()
            except Exception:
                pass  # unconfirmed; release/reconcile still free the lease
            else:
                self.kill_done.set()
            self._confirm_remote_gone()

        self._kill_thread = threading.Thread(target=run, name=f"gpu-kill-{self.job_id[-8:]}", daemon=True)
        self._kill_thread.start()

    def _confirm_remote_gone(self) -> None:
        """C7e-R1: after a settled cancel, the marker kill confirms nothing of this job is left on the card.

        Confirmed → the orphan record is dropped; unconfirmed/unreachable → it stays and reconcile retries it.
        """

        killer = self.orphan_kill
        if killer is None or self.outcome != "cancel":
            return
        try:
            confirmed = bool(killer({"id": self.job_id, "tenant_id": self.tenant_id}))
        except Exception:  # noqa: BLE001 - unreachable host: reconcile retries
            confirmed = False
        local = None
        try:
            local = GpuScheduler(self.sched.path, policy=self.sched.policy, clock=self.sched.clock)
            local.audit("orphan_killed" if confirmed else "orphan_kill_failed", tenant_id=self.tenant_id, job_id=self.job_id)
            if confirmed:
                local.clear_orphan(self.job_id)
        except Exception:  # noqa: BLE001 - audit/cleanup only; the orphan record still covers it
            pass
        finally:
            if local is not None:
                local.close()

    def _kill_if_settled_by_cancel(self, sched: GpuScheduler) -> None:
        """C7e-R1: a cancel after dispatch settles the job and deletes the lease at once (≤10 s card), so the supervisor
        never sees ``cancel`` any more. LIVE 86ebf36: nothing stopped the worker; it trained to the end and its files
        were attached to the cancelled Run. Work that was handed to the card is stopped here, right away.
        """

        if self.outcome or self.kill_started_at is not None:
            return
        try:
            job = sched.job(self.job_id)
        except KeyError:
            return
        if job["status"] == CANCELLED and job["dispatched_at"] is not None:
            self.outcome = "cancel"
            self._start_kill(sched)

    def watch(self, sched: GpuScheduler | None = None) -> str:
        """Cheap cancel poll between heartbeats: a settled or requested cancel is acted on at once."""

        sched = sched or self.sched
        if sched.lease_state(self.job_id)["action"] in {"cancel", "released"}:
            return self.tick(sched)
        return "run"

    def tick(self, sched: GpuScheduler | None = None) -> str:
        """One supervision step (the thread passes its own connection; tests call it directly)."""

        sched = sched or self.sched
        if not sched.heartbeat(self.lease_id):
            self._kill_if_settled_by_cancel(sched)
            return "released"
        self._broker_heartbeat(sched)
        state = sched.lease_state(self.job_id)["action"]
        if state == "cancel" and self.outcome != "cancel":
            self.outcome = "cancel"
            self._start_kill(sched)
        elif state == "timeout":
            if not self.outcome:
                self.outcome = "timeout"
            if sched.clock() >= self.deadline_at + sched.policy.kill_grace_seconds:
                self._start_kill(sched)
        return self.outcome or state

    def _broker_heartbeat(self, sched: GpuScheduler) -> None:
        """Broker heartbeat (≤ the supervisor cadence, ≤ 60 s). At ``max_down_seconds`` the job is cancelled."""

        if self.broker is None:
            return
        from .gpu_memory_broker import BrokerError, BrokerLeaseExpired

        self.broker_last_beat = sched.clock()
        try:
            self.broker.heartbeat(self.job_id)
        except BrokerLeaseExpired:
            if not self.outcome:
                self.outcome = "broker_expired"
                sched.audit("broker_window_expired", tenant_id=self.tenant_id, job_id=self.job_id)
                self._start_kill(sched)
        except (BrokerError, Exception):  # noqa: BLE001 - a missed beat is retried; the remote guard keeps its deadline
            sched.audit("broker_heartbeat_failed", tenant_id=self.tenant_id, job_id=self.job_id)

    def cancel_requested(self) -> bool:
        """C7c-FIX5: has the tenant (or the stopped run) asked this lease to stop? Read straight from the ledger, not the 30 s tick."""

        if self.outcome in {"cancel", "broker_expired"}:
            return True
        try:  # "released": the job was settled meanwhile (a cancel during preparation settles at once): stop before dispatch too
            return self.sched.lease_state(self.job_id)["action"] in {"cancel", "released"}
        except sqlite3.Error:
            return False

    def start(self, kill: Callable[[], Any] | None = None, *, thread: bool = True) -> "LeaseHandle":
        self.kill = kill
        if not thread:
            return self

        def loop() -> None:
            local = GpuScheduler(self.sched.path, policy=self.sched.policy, clock=self.sched.clock)
            poll = max(0.01, min(float(self.heartbeat_seconds), float(self.cancel_poll_seconds)))
            last_beat = time.monotonic()
            try:
                while not self._stop.wait(poll):
                    try:
                        due = time.monotonic() - last_beat >= self.heartbeat_seconds
                        outcome = self.tick(local) if due else self.watch(local)
                        if due:
                            last_beat = time.monotonic()
                        if outcome == "released":
                            return
                    except Exception:  # noqa: BLE001 - e.g. "database is locked": the next round retries
                        continue
            finally:
                local.close()

        self._thread = threading.Thread(target=loop, name=f"gpu-lease-{self.lease_id[-8:]}", daemon=True)
        self._thread.start()
        return self

    def release(self, ok: bool, detail: dict | None = None) -> dict:
        if self.released is not None:
            return self.released
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        # Late decisions (deadline passed while the work returned) still count.
        if not self.outcome:
            action = self.sched.lease_state(self.job_id)["action"]
            self.outcome = action if action in {"cancel", "timeout"} else ""
        info = dict(detail or {})
        if self._kill_thread is not None:
            self._kill_thread.join(timeout=0)
            if not self.kill_done.is_set():
                info["remote_kill"] = "unconfirmed"
        if self.outcome == "cancel":
            status, info["reason"] = CANCELLED, "cancelled_by_owner"
        elif self.outcome == "timeout":
            status, info["reason"] = TIMEOUT, "max_run_exceeded"
        elif self.outcome == "broker_expired":
            status, info["reason"] = CANCELLED, "broker_window_expired"
        else:
            status = SUCCEEDED if ok else FAILED
        try:
            self.released = self.sched.finish(self.job_id, status, info)
        finally:
            if self.broker is not None:
                # Restore never delays the tenant's results; services stay down only for the linger
                # and only if the next queued job needs that memory.
                from .gpu_broker_link import release_in_background
                try:
                    next_needs = self.sched.next_queued_request()
                except sqlite3.Error:
                    next_needs = None
                release_in_background(self.broker, self.job_id, next_needs, wait=self.broker_wait_release)
        return self.released


class GpuGate:
    """Submit → wait in the fair queue (holding the place) → lease, with honest messages."""

    def __init__(self, runtime_root: Path, config: dict[str, Any], *, probe: Probe,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                 poll_seconds: float = 5.0, heartbeat_seconds: float = 30.0, wait_seconds: float = 900.0,
                 broker_factory: Callable[..., Any] | None = None) -> None:
        self.runtime_root = Path(runtime_root)
        # ADDENDUM 2: (runtime_root, VramConfig) -> GpuMemoryBroker | None; None = no broker.
        self.broker_factory = broker_factory
        self.broker_wait_release = False  # tests: restore synchronously for determinism
        self.sched = scheduler(runtime_root, config=config, clock=clock)
        profile = str(config.get("credential_profile") or "")
        if profile:
            self.sched.ensure_node(profile=profile)
        self.breaker = CircuitBreaker(self.sched)
        self.health = HealthMonitor(self.breaker, probe)
        self.sleep = sleep
        self.poll_seconds = poll_seconds
        self.heartbeat_seconds = min(float(heartbeat_seconds), 60.0)  # broker heartbeat every ≤ 60 s
        self.wait_seconds = wait_seconds
        self.waiter_refresh_seconds = 15.0  # keeps a job live during a slow broker stop

    def close(self) -> None:
        self.sched.close()

    def acquire(self, *, tenant_id: str, owner_id: str, run_id: str, requested_seconds: int | None = None,
                vram_mb: int | None = None, on_wait: Callable[[dict], None] | None = None,
                operator: bool = False) -> LeaseHandle:
        from . import gpu_vram

        sched = self.sched
        if not self.breaker.allow_probe():
            # Fail fast; an existing queued job of this run keeps its place.
            raise GpuAdmissionError("breaker_open", **self._place(tenant_id, run_id))
        try:
            vram = gpu_vram.load(self.runtime_root)
            request_mb = vram.resolve_request(tenant_id, vram_mb)
        except gpu_vram.VramConfigError as error:
            code = str(error) if str(error) in {"request_too_large", "gpu_request_invalid"} else "vram_config_invalid"
            sched.audit("vram_refused", tenant_id=tenant_id, detail={"code": str(error)})
            raise GpuAdmissionError(code) from None
        try:
            job = sched.submit(tenant_id=tenant_id, owner_id=owner_id, run_id=run_id,
                               requested_seconds=requested_seconds, vram_mb=request_mb, budget_exempt=operator)
        except SchedulerError as error:
            raise GpuAdmissionError(str(error)) from None
        if job["status"] == RUNNING:
            raise GpuAdmissionError("tenant_busy")
        give_up = sched.clock() + self.wait_seconds
        last: tuple | None = None
        while True:
            sched.touch_waiter(job["id"])
            status = sched.queue_status(job["id"])
            free: int | None = None
            # position counts live jobs only, so a head without a waiter never stalls the next one.
            if status["position"] == 1 and sched._db().execute("SELECT 1 FROM gpu_leases WHERE node_id=?", (ALLOWED_NODE_ID,)).fetchone() is None:
                health, probe_free = self.health.check()
                if health == "open":
                    raise GpuAdmissionError("breaker_open", position=status["position"], eta_seconds=status["eta_seconds"])
                if health == "ok":
                    try:  # re-read on every admission: config edits apply without a redeploy
                        vram = gpu_vram.load(self.runtime_root)
                    except gpu_vram.VramConfigError:
                        vram = None  # fail closed: stay queued
                    free = vram.admissible_mb(tenant_id, probe_free) if vram is not None else None
                    if vram is not None and self._services_returning(vram):
                        # Free memory right now belongs to the services coming back: no admission until they are up.
                        sched.mark_reason(job["id"], "gpu_memory_short")
                        free = None
                    window_open = vram is not None and free is not None and free >= request_mb and self._broker_window_open(vram)
                    if window_open or (vram is not None and free is not None and free < request_mb):
                        # Short of memory, or memory is currently lent by the broker (lingering/down):
                        # join through the broker so it restores the services after the last job.
                        lease = self._try_broker(job, tenant_id, request_mb, probe_free, vram, on_wait, joining=window_open)
                        if lease is not None:
                            return lease
                        # Broker disabled / refused / circuit open: the honest queued line, job keeps its place.
                        sched.mark_reason(job["id"], "insufficient_gpu_memory")
                        free = None
            if free is not None:
                for grant in sched.schedule(free_vram=free):
                    if grant.job_id == job["id"]:
                        row = sched.job(job["id"])
                        return LeaseHandle(sched, row, sched.lease_for(job["id"]) or {"id": grant.lease_id},
                                           heartbeat_seconds=self.heartbeat_seconds)
            else:
                sched.reconcile()
            row = sched.job(job["id"])
            if row["status"] in TERMINAL:
                raise GpuAdmissionError(str(row["reason"] or row["status"]) if row["status"] != CANCELLED else "cancelled")
            if row["status"] == RUNNING:  # granted by another waiter's schedule() call
                return LeaseHandle(sched, row, sched.lease_for(job["id"]) or {"id": ""}, heartbeat_seconds=self.heartbeat_seconds)
            status = sched.queue_status(job["id"])
            reason = row["reason"] or "queued"
            if on_wait is not None and (status["position"], reason) != last:
                last = (status["position"], reason)
                on_wait({**status, "reason": reason})
            if sched.clock() >= give_up:
                code = "insufficient_gpu_memory" if reason in {"insufficient_gpu_memory", "gpu_memory_short"} else "queue_wait_exceeded"
                raise GpuAdmissionError(code, position=status["position"], eta_seconds=status["eta_seconds"])
            self.sleep(self.poll_seconds)

    def _broker(self, vram: Any) -> Any:
        if self.broker_factory is None or not vram.gpu_broker_enabled:
            return None
        if getattr(self, "_broker_cached", None) is None:
            self._broker_cached = self.broker_factory(self.runtime_root, vram)
        return self._broker_cached

    def _services_returning(self, vram: Any) -> bool:
        """The broker is restoring, or services it stopped are not back yet (unknown state counts)."""

        from .gpu_broker_link import services_returning

        return bool(vram.gpu_broker_enabled) and services_returning(self.runtime_root)

    def _broker_window_open(self, vram: Any) -> bool:
        if self.broker_factory is None or not vram.gpu_broker_enabled:
            return False
        if getattr(self, "_broker_cached", None) is None:
            # Cheap local check first: only open the broker when its state DB shows a lent window.
            from .gpu_broker_link import local_state
            if local_state(self.runtime_root)["window"] not in {"down", "lingering"}:
                return False
        broker = self._broker(vram)
        if broker is None:
            return False
        try:
            return broker._window()["state"] in {"down", "lingering"}
        except Exception:  # noqa: BLE001
            return False

    def _try_broker(self, job: dict, tenant_id: str, request_mb: int, probe_free: int, vram: Any,
                    on_wait: Callable[[dict], None] | None, *, joining: bool = False) -> "LeaseHandle | None":
        """Ask the broker to free memory for the head job (or join its open window), then grant exactly
        this job. None = cannot help: the caller keeps the honest queued line."""

        from . import gpu_broker_link as link
        from .gpu_memory_broker import BrokerError

        broker = self._broker(vram)
        if broker is None:
            return None
        sched = self.sched
        # Two reserves exist (admission, broker): the larger one wins, so a plan never comes back empty
        # while admission still says the job does not fit.
        broker_reserve = getattr(getattr(broker, "config", None), "reserve_mb", vram.gpu_vram_reserve_mb)
        needed_mb = request_mb + max(0, vram.gpu_vram_reserve_mb - broker_reserve)
        if not joining:
            if link.planned(broker, needed_mb, probe_free) is None:
                return None
            sched.mark_freeing(job["id"], True)
            status = sched.queue_status(job["id"])
            if on_wait is not None:
                on_wait({**status, "reason": FREEING, "freeing_minutes": link.freeing_minutes(vram)})
        sched.audit("broker_acquire", tenant_id=tenant_id, job_id=job["id"], detail={"request_mb": request_mb})
        try:
            with self._waiter_alive(job["id"]):  # a stop can outlast the waiter TTL; the job must stay live
                memory = broker.acquire(job["id"], needed_mb)  # runner thread; dispatch only after it returns
        except (BrokerError, Exception) as error:  # noqa: BLE001
            sched.mark_freeing(job["id"], False)
            sched.audit("broker_refused", tenant_id=tenant_id, job_id=job["id"], detail={"code": getattr(error, "code", type(error).__name__)})
            return None
        sched.touch_waiter(job["id"])
        grants = sched.schedule(free_vram=request_mb, only_job=job["id"])
        if not grants:
            sched.mark_freeing(job["id"], False)
            link.release_in_background(broker, job["id"], None)
            return None
        row = sched.job(job["id"])
        handle = LeaseHandle(sched, row, sched.lease_for(job["id"]) or {"id": grants[0].lease_id},
                             heartbeat_seconds=self.heartbeat_seconds)
        handle.broker = broker
        handle.broker_wait_release = self.broker_wait_release
        sched.audit("broker_granted", tenant_id=tenant_id, job_id=job["id"], detail={"freed_mb": getattr(memory, "freed_mb", 0)})
        return handle

    @contextmanager
    def _waiter_alive(self, job_id: str) -> Iterator[None]:
        """Refresh ``waiter_at`` from a helper thread while this runner is blocked in a broker call."""

        done = threading.Event()

        def refresh() -> None:
            local = GpuScheduler(self.sched.path, policy=self.sched.policy, clock=self.sched.clock)
            try:
                while not done.wait(self.waiter_refresh_seconds):
                    try:
                        local.touch_waiter(job_id)
                    except sqlite3.Error:
                        continue  # the next round retries; TTL is several rounds long
            finally:
                local.close()

        thread = threading.Thread(target=refresh, name=f"gpu-waiter-{job_id[-8:]}", daemon=True)
        thread.start()
        try:
            yield
        finally:
            done.set()
            thread.join(timeout=5)

    def _place(self, tenant_id: str, run_id: str) -> dict[str, Any]:
        for row in self.sched.jobs_for_tenant(tenant_id, run_id=run_id, limit=1):
            if row["status"] == QUEUED:
                status = self.sched.queue_status(row["id"])
                return {"position": status["position"], "eta_seconds": status["eta_seconds"]}
        return {}


def waiting_detail(status: dict) -> str:
    """Chinese progress line for a queued job (estimate, never negative)."""

    if status.get("reason") == "insufficient_gpu_memory":
        return "排队中，轮到你就开始。"
    if status.get("reason") == FREEING:
        minutes = int(status.get("freeing_minutes") or 0)
        return f"正在准备训练，预计 {minutes} 分钟内开始。" if minutes > 0 else "正在准备训练…"
    if status.get("reason") == "gpu_memory_short":
        return "排队中，轮到你就开始。"
    position = status.get("position")
    minutes = max(1, -(-int(status.get("eta_seconds") or 0) // 60)) if status.get("eta_seconds") else 0
    if not position:
        return "排队中"
    return f"排队中（第 {position} 位，预计等待约 {minutes} 分钟）" if minutes else f"排队中（第 {position} 位）"
