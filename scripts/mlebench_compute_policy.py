#!/usr/bin/env python3
"""Fail-closed local GPU launch policy for the MLE-Bench campaign.

The policy is intentionally local to the workstation.  Remote HPC copies of a
training runner do not carry ``workspace/runtime`` and therefore continue to
run normally.  A missing policy preserves legacy behaviour; an existing but
unreadable/invalid policy fails closed so a partial write cannot start a local
CUDA workload.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

POLICY_SCHEMA = "evomind.mlebench.compute_policy.v1"
POLICY_RELATIVE_PATH = Path("workspace") / "runtime" / "mlebench_compute_policy.json"
QUEUE_SUPERSEDED_STATUS = "superseded_by_hpc"
RUNNER_BLOCKED_STATUS = "blocked_by_hpc_only_policy"
_VALID_BACKENDS = {"hpc_only", "local_gpu"}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace one JSON artifact in the destination directory."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _policy_path(project_root: Path, policy_path: Path | None) -> Path:
    if policy_path is not None:
        return Path(policy_path).expanduser().resolve()
    return (Path(project_root).resolve() / POLICY_RELATIVE_PATH).resolve()


def _inactive_missing_record(path: Path) -> dict[str, Any]:
    return {
        "present": False,
        "valid": True,
        "active": False,
        "fail_closed": False,
        "path": str(path),
        "sha256": None,
        "schema": None,
        "compute_backend": None,
        "allow_local_gpu_training": None,
        "hpc_root": None,
        "error": None,
    }


def _fail_closed_record(
    path: Path,
    *,
    error: str,
    payload_bytes: bytes | None = None,
) -> dict[str, Any]:
    return {
        "present": True,
        "valid": False,
        "active": True,
        "fail_closed": True,
        "path": str(path),
        "sha256": _sha256_bytes(payload_bytes) if payload_bytes is not None else None,
        "schema": None,
        "compute_backend": "policy_error_fail_closed",
        "allow_local_gpu_training": False,
        "hpc_root": None,
        "error": error,
    }


def load_compute_policy(
    project_root: Path,
    *,
    policy_path: Path | None = None,
    read_attempts: int = 3,
    retry_seconds: float = 0.02,
) -> dict[str, Any]:
    """Read one coherent policy snapshot and return its enforcement record.

    ``Path.read_bytes`` provides a single-file snapshot even when another
    process atomically replaces the path.  Short retries cover transient
    sharing violations on Windows.  If a policy existed when the read began,
    exhaustion or invalid JSON is fail-closed.
    """

    path = _policy_path(project_root, policy_path)
    if read_attempts < 1:
        raise ValueError("read_attempts must be at least one")
    initially_present = path.is_file()
    if not initially_present:
        return _inactive_missing_record(path)

    raw: bytes | None = None
    last_error: BaseException | None = None
    for attempt in range(read_attempts):
        try:
            raw = path.read_bytes()
            payload = json.loads(raw.decode("utf-8-sig"))
            if not isinstance(payload, dict):
                raise TypeError("compute policy root must be a JSON object")
            last_error = None
            break
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            last_error = exc
            if attempt + 1 < read_attempts:
                time.sleep(max(0.0, retry_seconds))
    if last_error is not None:
        return _fail_closed_record(
            path,
            error=f"{type(last_error).__name__}: {last_error}",
            payload_bytes=raw,
        )

    schema = payload.get("schema")
    backend = payload.get("compute_backend")
    allow_local = payload.get("allow_local_gpu_training")
    contract_errors: list[str] = []
    if schema != POLICY_SCHEMA:
        contract_errors.append("unexpected schema")
    if backend not in _VALID_BACKENDS:
        contract_errors.append("unsupported compute_backend")
    if not isinstance(allow_local, bool):
        contract_errors.append("allow_local_gpu_training must be boolean")
    if backend == "hpc_only" and allow_local is not False:
        contract_errors.append("hpc_only requires allow_local_gpu_training=false")
    if backend == "local_gpu" and allow_local is not True:
        contract_errors.append("local_gpu requires allow_local_gpu_training=true")
    if contract_errors:
        return _fail_closed_record(
            path,
            error="; ".join(contract_errors),
            payload_bytes=raw,
        )

    return {
        "present": True,
        "valid": True,
        "active": backend == "hpc_only" and allow_local is False,
        "fail_closed": False,
        "path": str(path),
        "sha256": _sha256_bytes(raw),
        "schema": schema,
        "compute_backend": backend,
        "allow_local_gpu_training": allow_local,
        "hpc_root": payload.get("hpc_root"),
        "preferred_hpc_job_id": payload.get("preferred_hpc_job_id"),
        "error": None,
    }


def write_queue_superseded_if_hpc_only(
    *,
    project_root: Path,
    status_path: Path,
    schema: str,
    queue_name: str,
    plan_path: Path | None = None,
) -> bool:
    """Write the queue's terminal supersession status when HPC-only is active."""

    policy = load_compute_policy(project_root)
    if not policy["active"]:
        return False
    payload = {
        "schema": schema,
        "created_at": now_iso(),
        "status": QUEUE_SUPERSEDED_STATUS,
        "terminal": True,
        "queue_name": queue_name,
        "pid": os.getpid(),
        "queue_pid": os.getpid(),
        "training_pid": None,
        "training_started": False,
        "plan_path": str(Path(plan_path).resolve()) if plan_path is not None else None,
        "compute_policy": policy,
        "process_signals_sent": 0,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }
    atomic_write_json(status_path, payload)
    return True


def write_runner_blocked_if_hpc_only(
    *,
    project_root: Path,
    evidence_path: Path,
    runner_name: str,
    run_id: str,
    output_root: Path,
) -> Path | None:
    """Persist early runner evidence before any runner-owned Torch/CUDA setup."""

    policy = load_compute_policy(project_root)
    if not policy["active"]:
        return None
    destination = Path(evidence_path).resolve()
    atomic_write_json(
        destination,
        {
            "schema": "evomind.mlebench.local_gpu_training_guard.v1",
            "created_at": now_iso(),
            "status": RUNNER_BLOCKED_STATUS,
            "terminal": True,
            "runner": runner_name,
            "runner_pid": os.getpid(),
            "run_id": run_id,
            "output_root": str(Path(output_root).resolve()),
            "evidence_path": str(destination),
            "training_started": False,
            "cuda_initialization_started": False,
            "compute_policy": policy,
            "process_signals_sent": 0,
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "human_gate_preserved": True,
        },
    )
    return destination

