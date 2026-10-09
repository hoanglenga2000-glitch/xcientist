"""Transactional session-closeout fix: three bundle files, seal re-pinned, DB snapshotted.

Runs on the EvoMind host.  Fixes the permanent "running" session ledger:

* ``http_server.py``  - a settled direct tool call now settles its session too;
* ``runtime.py``      - adds ``AgentRuntime.settle_orphan_sessions``;
* ``run_python_runtime.py`` - sweeps crash-orphaned sessions at startup and
  writes a reconciliation receipt next to the runtime database.

The restart performed here settles the historical backlog through the new
startup hook, so the live database is snapshotted into the backup directory
before the service is stopped.  No training, HPC, budget or campaign state is
touched; ``work_guard`` only proves that.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import http.client
import json
import os
import shutil
import sqlite3
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("C:/ProgramData/EvoMind")
STAGE = ROOT / "staging/session-closeout-fix-20260916"
BACKUP = ROOT / "backups/session-closeout-fix-20260916"
SEAL_PATH = ROOT / "state/bundle-integrity.json"
LIVE_DB = ROOT / "data/workspace/runtime/runtime.sqlite3"

SEAL_SHA256 = "3a95fda20a17b079b918fd80ee0141f6fa3f026e1a4d9c4989899f8fc70f7368"

# bundle-relative path -> (staged file name, sha256 before, sha256 after)
PAYLOADS = {
    "runtime/evomind_runtime/http_server.py": (
        "http_server.py",
        "970bd7d0577731e57a98a8062f4687b537c6d49e387a6b30788d06b30972212a",
        "52413bb4eeec0945191ab6fb243f7b082ea94c1619c0029f05d5e7639069e2ce",
    ),
    "runtime/evomind_runtime/runtime.py": (
        "runtime.py",
        "6a14d2980cd4372ec9dd697c349e54bb91c8fd902e0ea64592bd33ce2c397cc1",
        "45125f61d319196662af951779d5e8dbad426db7a137a6f74585fc7e1b25a794",
    ),
    "runtime/run_python_runtime.py": (
        "run_python_runtime.py",
        "55ef6200195f104c989fcb054dfb68f82c1a9c888be04bd80839c7cd2bd62a67",
        "d8ebb2859f276e8e73043597ab45d8baf9203ba8838785289364f7ab1e9af8f4",
    ),
}

# The one historical Run and the four tool calls that are intentionally left
# untouched: they are terminal evidence owned by retired work, not live work.
LEGACY_RUN = "run_0c57bbde44e94a988843c2f6981f3c65"
LEGACY_CALLS = {
    "call_b4c7d7d8095a451a5d0d5878f6671620",
    "call_ad79487f22e6e88e8acc01bc8cdb9eb9",
    "call_5c0891e12bbd853969ef3aae01accb80",
    "call_d2f39b4909fe4403aa821d191c649721",
}

sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, code):
    if not value:
        raise ValueError(code)


def write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def session_counts():
    with sqlite3.connect("file:" + LIVE_DB.as_posix() + "?mode=ro", uri=True, timeout=30) as con:
        con.execute("PRAGMA query_only=ON")
        return {
            "sessions": dict(con.execute("SELECT status,count(*) FROM sessions GROUP BY 1").fetchall()),
            "tool_calls": dict(con.execute("SELECT status,count(*) FROM tool_calls GROUP BY 1").fetchall()),
        }


def work_guard():
    """Fail closed unless every remaining activity row is known retired evidence."""

    with sqlite3.connect("file:" + LIVE_DB.as_posix() + "?mode=ro", uri=True, timeout=30) as con:
        con.execute("PRAGMA query_only=ON")
        runs = con.execute(
            "SELECT id,status FROM assistant_runs WHERE status IN "
            "('created','queued','planning','running','verifying','recovering','pausing','waiting_approval')"
        ).fetchall()
        require(all(row == (LEGACY_RUN, "recovering") for row in runs), "current_run_prevents_restart")
        calls = con.execute("SELECT id,session_id FROM tool_calls WHERE status='running'").fetchall()
        require(all(row[0] in LEGACY_CALLS for row in calls), "current_tool_prevents_restart")
        require(
            con.execute(
                "SELECT count(*) FROM approvals WHERE status='approved' AND tool_call_id IN "
                "(SELECT id FROM tool_calls WHERE status='waiting_approval')"
            ).fetchone()[0]
            == 0,
            "approved_work_prevents_restart",
        )
    return {"preserved_legacy_runs": len(runs), "preserved_legacy_calls": len(calls)}


@contextmanager
def deployment_lock():
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.ReleaseMutex.argtypes = kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.CreateMutexW(None, False, "Global\\EvoMind-Byoa-V12-Deployment")
    require(handle, "deployment_mutex_unavailable")
    acquired = kernel.WaitForSingleObject(handle, 0)
    try:
        require(acquired == 0, "deployment_mutex_busy")
        yield
    finally:
        if acquired in {0, 0x80}:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def service(action):
    print(json.dumps({"phase": "service_" + action.lower()}), flush=True)
    process = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-File",
            "C:/SecureInput/Invoke-ServiceAccountAction.ps1",
            "-Action",
            action,
            "-TimeoutMinutes",
            "10",
        ],
        capture_output=True,
        timeout=720,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    write(
        BACKUP / (action.lower() + "-" + str(time.time_ns()) + ".json"),
        {
            "exit_code": process.returncode,
            "output_sha256": hashlib.sha256(process.stdout + process.stderr).hexdigest(),
            "output_tail": (process.stdout + process.stderr).decode("utf-8", "replace")[-600:],
        },
    )
    require(process.returncode == 0, "service_" + action.lower() + "_failed")


def running_process_ids():
    import psutil

    ids = {}
    for name in ("pythonw.exe", "node.exe", "cliproxy-7.2.128.exe"):
        ids[name] = sorted(process.pid for process in psutil.process_iter(["name"]) if process.info["name"] == name)
    return ids


def listeners():
    import psutil

    return sorted(
        connection.laddr.port
        for connection in psutil.net_connections("tcp")
        if connection.status == "LISTEN" and connection.laddr.port in {8088, 8765, 65068, 7890}
    )


def http_health(port, path):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        body = response.read()
        return response.status, body
    finally:
        connection.close()


def snapshot_database(destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect("file:" + LIVE_DB.as_posix() + "?mode=ro", uri=True, timeout=60)
    target = sqlite3.connect(str(destination))
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()
    return sha(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    for relative, (source, before, after) in PAYLOADS.items():
        require(sha(ROOT / "bundle" / relative) == before, "active_source_changed:" + relative)
        require(sha(STAGE / source) == after, "candidate_changed:" + source)
    require(sha(SEAL_PATH) == SEAL_SHA256, "seal_changed")
    require(not BACKUP.exists(), "prior_fix_requires_reconciliation")
    config = json.loads((ROOT / "config/node-config.json").read_text(encoding="utf-8-sig"))
    require(config["hpc"]["state"] != "active", "global_hpc_startup_requires_review")
    seal = json.loads(SEAL_PATH.read_text(encoding="utf-8-sig"))
    for row in seal["files"]:
        require(sha(ROOT / "bundle" / row["path"]) == row["sha256"], "existing_bundle_integrity_failed")
    work = work_guard()
    counts_before = session_counts()
    if not args.apply:
        print(json.dumps({
            "status": "ready_for_session_closeout_deploy",
            "work": work,
            "sessions_before": counts_before,
            "production_changed": False,
        }, ensure_ascii=False))
        return

    with deployment_lock():
        require(work_guard() == work, "work_changed")
        BACKUP.mkdir(parents=True)
        subprocess.run(
            ["icacls.exe", str(BACKUP), "/inheritance:r", "/grant:r", "SYSTEM:(OI)(CI)F", "Administrators:(OI)(CI)F"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        preserved = ["state/bundle-integrity.json", *("bundle/" + name for name in PAYLOADS)]
        for relative in preserved:
            saved = BACKUP / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, saved)
        db_snapshot = BACKUP / "runtime-before-closeout.sqlite3"
        snapshot_sha256 = snapshot_database(db_snapshot)
        write(BACKUP / "before.json", {
            "work": work,
            "sessions_before": counts_before,
            "listeners_before": listeners(),
            "processes_before": running_process_ids(),
            "db_snapshot": str(db_snapshot),
            "db_snapshot_sha256": snapshot_sha256,
            "payloads": {name: {"source": values[0], "before": values[1], "after": values[2]}
                         for name, values in PAYLOADS.items()},
        })
        changed = False
        try:
            service("Stop")
            require(work_guard() == work, "work_changed_during_stop")
            import psutil  # noqa: F401 - imported here so the guard runs after Stop

            require(
                not [port for port in listeners() if port in {8088, 8765, 65068}],
                "managed_listeners_not_stopped",
            )
            for relative, (source, _before, _after) in PAYLOADS.items():
                target = ROOT / "bundle" / relative
                temporary = target.with_name(target.name + ".closeout-new")
                require(not temporary.exists(), "temporary_path_collision")
                shutil.copyfile(STAGE / source, temporary)
                subprocess.run(
                    ["icacls.exe", str(temporary), "/inheritance:r", "/grant:r", "SYSTEM:F", "Administrators:F", "EvoMindSvc:R"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
                os.replace(temporary, target)
                changed = True
            for row in seal["files"]:
                payload = PAYLOADS.get(row["path"])
                if payload:
                    file_path = ROOT / "bundle" / row["path"]
                    row.update(size=file_path.stat().st_size, sha256=payload[2])
                else:
                    require(sha(ROOT / "bundle" / row["path"]) == row["sha256"], "unrelated_bundle_file_changed")
            seal["sealed_at_utc"] = datetime.now(timezone.utc).isoformat()
            temporary = SEAL_PATH.with_name("bundle-integrity.closeout-new.json")
            write(temporary, seal)
            os.replace(temporary, SEAL_PATH)
            service("Start")
            require(work_guard() == work, "historical_work_changed")
            status, body = http_health(8088, "/api/healthz")
            health = json.loads(body.decode("utf-8"))
            require(status == 200 and health.get("status") == "ready", "web_health_not_ready")
            runtime_status, _ = http_health(8765, "/healthz")
            require(runtime_status in {200, 401}, "runtime_not_listening")
            receipt_path = ROOT / "data/workspace/runtime/session-reconciliation-at-startup.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {}
            result = {
                "status": "session_closeout_deployed",
                "sessions_after": session_counts(),
                "startup_reconciliation_receipt": str(receipt_path),
                "startup_reconciled_count": receipt.get("reconciled_count"),
                "listeners_after": listeners(),
                "processes_after": running_process_ids(),
                "db_snapshot_sha256": snapshot_sha256,
                "backup": str(BACKUP),
                "work_before": work,
                "work_after": work_guard(),
            }
            write(BACKUP / "result.json", result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except Exception as error:
            rollback = "not_needed"
            if changed:
                try:
                    service("Stop")
                    for relative in preserved:
                        shutil.copy2(BACKUP / relative, ROOT / relative)
                    service("Start")
                    rollback = "prior_code_and_service_restored"
                except Exception:
                    rollback = "manual_reconciliation_required"
            write(BACKUP / "failure.json", {
                "error_type": type(error).__name__,
                "code": str(error)[:200],
                "rollback": rollback,
                "db_snapshot": str(db_snapshot),
                "db_snapshot_sha256": snapshot_sha256,
            })
            raise


if __name__ == "__main__":
    main()
