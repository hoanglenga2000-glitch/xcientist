"""t238 C7c FIX1 H5: 「删除」 reaches every copy training made of a tenant's data.

Training leaves, per Run: ``work/managed-fit-*`` (derived rows), ``<runtime>/tenant_hpc/<run>/`` (raw
outputs) and the remote ``evomind_runs/<run>/`` (bundle with the inputs). The local two go first.
The remote delete is strictly confined: the path comes from ``confine_remote_path``, must not be a
symlink, and must resolve to itself on the remote before ``rm`` runs.

A record under ``tenant_purge_pending/`` exists from before the first delete until every copy is
gone, so a failure or a crash leaves a retry, and the caller reports ``partial``, never a clean
delete. Tenants see one plain sentence; the operator log has the run and an error category.
"""
from __future__ import annotations

import json
import re
import shutil
import stat
import threading
import time
from pathlib import Path
from typing import Any, Callable

from research_agent_workstation.server.core.gpu_credentials import ALLOWED_GPU_REMOTE_ROOT
from research_os.remote_paths import RemotePathError, confine_remote_path, resolve_and_verify, shell_command

PENDING_DIR = "tenant_purge_pending"
CLEANUP_PENDING_MESSAGE = "文件已删除，但训练时留下的副本还没清理完，系统会继续重试。"
RETRY_SECONDS = 600.0
REQUEST_BUDGET_SECONDS = 8.0  # the browser gives a delete 15 s; what does not fit is finished by the retry
RETRY_BUDGET_SECONDS = 300.0
_RUN_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")


def remote_run_path(run_id: str) -> str:
    if not _RUN_ID.fullmatch(str(run_id)):
        raise RemotePathError("run_id_invalid")
    return confine_remote_path(ALLOWED_GPU_REMOTE_ROOT, "evomind_runs", run_id)


def purge_remote_run(client: Any, run_id: str) -> str:
    """"absent" or "removed". Raises on a refused path or a failed delete."""

    path = remote_run_path(run_id)
    sftp = client.open_sftp()
    try:
        try:
            attributes = sftp.lstat(path)
        except FileNotFoundError:
            return "absent"
        if stat.S_ISLNK(attributes.st_mode) or not stat.S_ISDIR(attributes.st_mode):
            raise RemotePathError("symlink_escape")
        if resolve_and_verify(sftp, path) != path:  # a symlinked parent lands somewhere else
            raise RemotePathError("symlink_escape")
    finally:
        sftp.close()
    _stdin, stdout, _stderr = client.exec_command(shell_command("rm", "-rf", "--", path, paths=(2,)), timeout=120)
    if stdout.channel.recv_exit_status() != 0:
        raise RuntimeError("remote_delete_failed")
    return "removed"


def _remove_tree(path: Path) -> bool:
    if path.is_symlink():
        path.unlink()  # never followed
        return True
    if path.is_dir():
        shutil.rmtree(path)
        return True
    return False


def _local_copies(task_root: Path, runtime_root: Path, run_id: str) -> list[Path]:
    work = Path(task_root) / run_id / "work"
    fits = sorted(work.glob("managed-fit-*")) if work.is_dir() and not work.is_symlink() else []
    raw = Path(runtime_root) / "tenant_hpc" / run_id
    return [*fits, *([raw] if raw.exists() or raw.is_symlink() else [])]


def _trained(runtime_root: Path, run_id: str) -> bool:
    """The scheduler saw this Run, so something may be on the remote. Also stops whatever it still has."""

    from . import gpu_admission

    if not (Path(runtime_root) / gpu_admission.DB_NAME).exists():
        return False
    sched = gpu_admission.scheduler(runtime_root)
    try:
        sched.cancel_run(run_id)
        return bool(sched._db().execute("SELECT 1 FROM train_jobs WHERE run_id=? LIMIT 1", (run_id,)).fetchone())
    finally:
        sched.close()


def _connector() -> "Callable[[], Any] | None":
    from . import gpu_admission

    return gpu_admission._service_connector()


def _audit(runtime_root: Path, **event: Any) -> None:
    path = Path(runtime_root) / "audit" / "tenant-purge.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at": time.time(), **event}, sort_keys=True) + "\n")
    except OSError:
        pass


def _record_path(runtime_root: Path, run_id: str) -> Path:
    return Path(runtime_root) / PENDING_DIR / f"{run_id}.json"


def _write_record(runtime_root: Path, run_id: str, record: dict[str, Any]) -> None:
    path = _record_path(runtime_root, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(".tmp")
    scratch.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    scratch.replace(path)


def _read_record(path: Path) -> dict[str, Any] | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def pending(runtime_root: Path, tenant_id: str | None = None) -> list[dict[str, Any]]:
    folder = Path(runtime_root) / PENDING_DIR
    records = [_read_record(path) for path in sorted(folder.glob("*.json"))] if folder.is_dir() else []
    return [item for item in records if item and (tenant_id is None or item.get("tenant_id") == tenant_id)]


class PurgeSession:
    """One delete operation. It connects at most once and spends at most ``budget_seconds`` on the
    remote, so a host that is down costs one bounded wait, not one per run.

    ``defer_remote`` only records the work: start-up recovery runs before the port binds.
    """

    def __init__(self, runtime_root: Path, task_root: Path, *, defer_remote: bool = False,
                 budget_seconds: float = REQUEST_BUDGET_SECONDS) -> None:
        self.runtime_root = Path(runtime_root)
        self.task_root = Path(task_root)
        self.defer_remote = defer_remote
        self._deadline = time.monotonic() + budget_seconds
        self._client: Any = None
        self._unreachable = False
        self._closed = False

    def _remote(self) -> Any:
        if self._client is None and not self._unreachable:
            connect = _connector()
            try:
                self._client = connect() if connect is not None else None
            except Exception:  # noqa: BLE001 - recorded as pending
                self._client = None
            self._unreachable = self._client is None
        return self._client

    def close(self) -> None:
        self._closed = True
        if self._client is not None:
            self._client.close()
            self._client = None

    def purge(self, run_ids: list[str], *, tenant_id: str, actor: str) -> list[str]:
        """The Runs with a copy left after this attempt (each one keeps its retry record)."""

        return [run_id for run_id in run_ids if not self._purge_run(run_id, tenant_id, actor)]

    def _purge_run(self, run_id: str, tenant_id: str, actor: str) -> bool:
        if not _RUN_ID.fullmatch(run_id):
            return True  # no such path was ever built for this id
        path = _record_path(self.runtime_root, run_id)
        record = _read_record(path) if path.exists() else None
        trained = _trained(self.runtime_root, run_id)
        if record is None and not trained and not _local_copies(self.task_root, self.runtime_root, run_id):
            return True
        record = record or {"run_id": run_id, "tenant_id": tenant_id, "actor": actor, "created_at": time.time(), "attempts": 0}
        _write_record(self.runtime_root, run_id, record)
        error = ""
        try:
            for copy in _local_copies(self.task_root, self.runtime_root, run_id):
                _remove_tree(copy)
        except OSError as exc:
            error = "local:" + type(exc).__name__
        if not error:
            error = self._purge_remote(run_id)
        if error:
            _write_record(self.runtime_root, run_id, {**record, "attempts": int(record["attempts"]) + 1, "last_error": error})
            _audit(self.runtime_root, kind="purge_pending", run_id=run_id, tenant_id=record["tenant_id"], error=error)
            return False
        path.unlink(missing_ok=True)
        _audit(self.runtime_root, kind="purged", run_id=run_id, tenant_id=record["tenant_id"], actor=record["actor"])
        return True

    def _purge_remote(self, run_id: str) -> str:
        if self.defer_remote:
            return "remote:deferred"
        outcome: list[str] = []

        def work() -> None:
            client = self._remote()
            if client is None:
                outcome.append("remote:unreachable")
                return
            try:
                purge_remote_run(client, run_id)
                outcome.append("")
            except RemotePathError as exc:
                outcome.append("remote_refused:" + exc.reason)
            except Exception as exc:  # noqa: BLE001 - transport, SFTP or rm failure: retried
                outcome.append("remote:" + type(exc).__name__)
            finally:
                if self._closed and self._client is not None:  # the request stopped waiting for this thread
                    self._client.close()
                    self._client = None

        thread = threading.Thread(target=work, name="tenant-purge-remote", daemon=True)
        thread.start()
        thread.join(max(0.0, self._deadline - time.monotonic()))
        if not outcome:
            self._unreachable = True
            return "remote:timeout"  # the thread may still finish it; then the retry finds it absent
        return outcome[0]


def retry_pending(runtime_root: Path, task_root: Path) -> list[str]:
    session = PurgeSession(runtime_root, task_root, budget_seconds=RETRY_BUDGET_SECONDS)
    try:
        left: list[str] = []
        for record in pending(runtime_root):
            left += session.purge([str(record.get("run_id") or "")], tenant_id=str(record.get("tenant_id") or ""), actor=str(record.get("actor") or ""))
        return left
    finally:
        session.close()


def retry_in_background(runtime_root: Path, task_root: Path, *, seconds: float = RETRY_SECONDS) -> threading.Thread:
    """Keeps the promise in ``CLEANUP_PENDING_MESSAGE``. Nothing pending means no connection at all."""

    def run() -> None:
        while True:
            try:
                retry_pending(runtime_root, task_root)
            except Exception:  # noqa: BLE001 - the next round retries
                pass
            time.sleep(seconds)

    thread = threading.Thread(target=run, name="tenant-purge-retry", daemon=True)
    thread.start()
    return thread
