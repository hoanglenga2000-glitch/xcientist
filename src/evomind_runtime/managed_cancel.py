"""Exact-process cancellation. Missing identity or exit proof is never success."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import select
import signal
import sqlite3
import stat
import sys
from datetime import datetime, timezone


SCHEMA = "evomind.managed_cancel.v1"
PROC = Path("/proc")
ALLOWED_ROOT = Path("/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra")
IDENTITY_FLAGS = (
    "designated_proxy_path_verified", "pinned_gateway_host_key_verified",
    "allocation_role_authenticated", "expected_host_uuid_match",
    "expected_gpu_uuid_match", "expected_gpu_model_and_memory_match",
    "allowed_remote_root_match", "job_container_verified",
)


def require_fresh_identity(runtime_root: Path, session_id: str, job_id: int) -> None:
    database = Path(runtime_root) / "runtime.sqlite3"
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=3)
    try:
        row = connection.execute(
            "SELECT completed_at,result_json FROM tool_calls WHERE session_id=? AND tool_name='hpc_verify' AND status='completed' ORDER BY completed_at DESC LIMIT 1",
            (session_id,),
        ).fetchone()
    finally:
        connection.close()
    try:
        if row is None:
            raise ValueError()
        timestamp = datetime.fromisoformat(str(row[0]).replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - timestamp).total_seconds()
        result = json.loads(row[1])
        receipt = result.get("content") or {}
        if isinstance(receipt, str):
            receipt = json.loads(receipt)
        samples = receipt.get("samples") or []
        valid = (
            result.get("ok") is True and 0 <= age <= 120
            and receipt.get("schema") == "evomind.hpc.identity_receipt.v2"
            and receipt.get("job_id") == job_id
            and receipt.get("samples_requested") == receipt.get("samples_passed") == 5
            and isinstance(samples, list) and len(samples) == 5
            and all(receipt.get(flag) is True for flag in IDENTITY_FLAGS)
            and all(isinstance(sample, dict) and sample.get("job_id") == job_id
                    and all(sample.get(flag) is True for flag in IDENTITY_FLAGS) for sample in samples)
        )
        if not valid:
            raise ValueError()
    except (TypeError, ValueError, AttributeError, KeyError):
        raise ValueError("fresh_complete_hpc_identity_required_for_cancel") from None


def process_identity(pid: int, solution: Path) -> dict | None:
    """Match an actual Python script argument, never a command-line substring."""
    try:
        directory = PROC / str(pid)
        if directory.stat().st_uid != os.geteuid():
            return None
        stat = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        argv = (directory / "cmdline").read_bytes().split(b"\0")
        argv = [item.decode("utf-8", "strict") for item in argv if item]
        if not argv or not re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?", Path(argv[0]).name):
            return None
        cwd = (directory / "cwd").resolve(strict=True)
        scripts = []
        for argument in argv[1:]:
            if argument.startswith("-") or not argument.endswith(".py"):
                continue
            path = (cwd / argument).resolve(strict=True)
            if path.name == "solution.py" and path.is_relative_to(solution):
                scripts.append(path)
        if len(scripts) != 1 or stat[0] in {"Z", "X"}:
            return None
        digest = hashlib.sha256(json.dumps({"argv": argv, "cwd": str(cwd), "start_ticks": stat[19]}, sort_keys=True).encode()).hexdigest()
        return {"pid": pid, "parent_pid": int(stat[1]), "start_ticks": stat[19], "identity_sha256": digest}
    except (OSError, ValueError, IndexError, UnicodeError):
        return None


def _has_children(pid: int) -> bool:
    for entry in PROC.iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[1]) == pid and fields[0] not in {"Z", "X"}:
                return True
        except FileNotFoundError:
            continue
        except (OSError, ValueError, IndexError):
            # An unreadable process tree cannot establish safe cancellation.
            return True
    return False


def canonical_solution(solution: Path) -> Path:
    if not solution.is_absolute() or ".." in solution.parts:
        raise ValueError("cancel_solution_path_rejected")
    # Compare before replacing the caller's path: aliases inside the allowed
    # root can otherwise redirect cancellation to another Run.
    if solution.resolve(strict=True) != solution:
        raise ValueError("cancel_solution_alias_rejected")
    solution.relative_to(ALLOWED_ROOT)
    if "solutions" not in solution.relative_to(ALLOWED_ROOT).parts:
        raise ValueError("cancel_solution_root_rejected")
    return solution


def directory_identity(solution: Path) -> tuple[int, int]:
    value = solution.lstat()
    if not stat.S_ISDIR(value.st_mode):
        raise ValueError("cancel_solution_directory_replaced")
    return value.st_dev, value.st_ino


def cancel_exact(solution: Path, *, send_signal: bool) -> dict:
    solution = canonical_solution(solution)
    directory_before = directory_identity(solution)
    result = {"schema": SCHEMA, "solution_root": str(solution), "status": "unconfirmed", "exit_observed": False, "signals_sent": 0, "other_processes_modified": False}
    identities = [identity for entry in PROC.iterdir() if entry.name.isdecimal()
                  if (identity := process_identity(int(entry.name), solution)) is not None]
    if len(identities) != 1:
        return {**result, "reason": "managed_process_identity_not_unique", "matching_processes": len(identities)}
    identity = identities[0]
    result["identity"] = identity
    if not send_signal:
        return {**result, "status": "observed", "reason": "read_only_probe"}
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        return {**result, "reason": "pidfd_cancellation_not_supported"}
    fd = None
    try:
        if canonical_solution(solution) != solution or directory_identity(solution) != directory_before:
            return {**result, "reason": "managed_solution_directory_changed"}
        fd = os.pidfd_open(identity["pid"], 0)
        if (canonical_solution(solution) != solution or directory_identity(solution) != directory_before
                or process_identity(identity["pid"], solution) != identity):
            return {**result, "reason": "managed_process_identity_changed"}
        if _has_children(identity["pid"]):
            return {**result, "reason": "managed_process_tree_requires_separate_gate"}
        if canonical_solution(solution) != solution or directory_identity(solution) != directory_before:
            return {**result, "reason": "managed_solution_directory_changed"}
        signal.pidfd_send_signal(fd, signal.SIGTERM)
        result["signals_sent"] = 1
        exited, _, _ = select.select([fd], [], [], 10)
        if not exited:
            return {**result, "reason": "managed_process_exit_not_observed"}
        return {**result, "status": "cancelled", "exit_observed": True}
    except (OSError, ValueError):
        return {**result, "reason": "managed_process_cancel_failed"}
    finally:
        if fd is not None:
            os.close(fd)


def confirmed_receipt(output: str, expected_solution: str) -> dict:
    try:
        if len(output.encode()) > 64 * 1024:
            raise ValueError()
        result = json.loads(output)
        if (
            not isinstance(result, dict) or result.get("schema") != SCHEMA
            or result.get("solution_root") != expected_solution
            or result.get("status") != "cancelled" or result.get("exit_observed") is not True
            or type(result.get("signals_sent")) is not int or result["signals_sent"] != 1
            or result.get("other_processes_modified") is not False
            or not isinstance(result.get("identity"), dict)
            or not re.fullmatch(r"[a-f0-9]{64}", str(result["identity"].get("identity_sha256", "")))
            or type(result["identity"].get("pid")) is not int or result["identity"]["pid"] <= 1
        ):
            raise ValueError()
        return result
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise ValueError("managed_cancel_exit_unconfirmed") from None


if __name__ == "__main__":
    try:
        print(json.dumps(cancel_exact(Path(sys.argv[1]), send_signal=sys.argv[2] == "cancel")))
    except Exception:
        print(json.dumps({"schema": SCHEMA, "status": "unconfirmed", "reason": "managed_cancel_contract_error", "exit_observed": False, "signals_sent": 0}))
        raise SystemExit(1)
