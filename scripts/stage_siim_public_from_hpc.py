#!/usr/bin/env python3
"""Resume-safe staging of the SIIM public JPEG subset from the dedicated HPC root."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import posixpath
import queue
import shlex
import shutil
import sys
import threading
import time
from datetime import datetime
from importlib import import_module
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

# Keep direct ``python scripts/...`` execution independent of caller PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

gpu_credentials = import_module(
    "research_agent_workstation.server.core.gpu_credentials"
)
ALLOWED_GPU_REMOTE_ROOT = gpu_credentials.ALLOWED_GPU_REMOTE_ROOT
connect_ssh = gpu_credentials.connect_ssh
load_gpu_ssh_config = gpu_credentials.load_gpu_ssh_config


COMPETITION_ID = "siim-isic-melanoma-classification"
REQUIRED_CREDENTIAL_PROFILE = "job90353"
REMOTE_PUBLIC_ROOT = posixpath.join(
    ALLOWED_GPU_REMOTE_ROOT,
    "mlebench_official_data",
    COMPETITION_ID,
    "prepared",
    "public",
)
DEFAULT_DESTINATION = (
    PROJECT_ROOT
    / "workspace"
    / "local_gpu"
    / "mlebench_official_data"
    / COMPETITION_ID
    / "prepared"
    / "public"
)
ALLOWED_CSV_FILES = {"train.csv", "test.csv", "sample_submission.csv"}
ALLOWED_IMAGE_PREFIXES = {"jpeg/train", "jpeg/test"}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_bound_siim_hpc_config():
    """Load only the strict job90353 DPAPI profile before any SSH socket opens.

    SIIM public data staging is part of the governed job90353 workflow.  Using
    the default/legacy profile or a different job profile can silently point at
    the wrong allocation, so fail closed before network I/O unless the caller
    explicitly selected the job-scoped profile and its v2 lifecycle metadata is
    active.
    """

    profile = os.environ.get("EVOMIND_HPC_CREDENTIAL_PROFILE", "").strip()
    if profile != REQUIRED_CREDENTIAL_PROFILE:
        raise RuntimeError(
            "SIIM staging requires "
            f"EVOMIND_HPC_CREDENTIAL_PROFILE={REQUIRED_CREDENTIAL_PROFILE}"
        )
    return load_gpu_ssh_config(strict_named_profile=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_relative_public_path(raw_path: str) -> str:
    candidate = PurePosixPath(str(raw_path))
    if not raw_path or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Unsafe public-data path: {raw_path!r}")
    normalized = candidate.as_posix()
    if normalized in ALLOWED_CSV_FILES:
        return normalized
    parent = candidate.parent.as_posix()
    if parent in ALLOWED_IMAGE_PREFIXES and candidate.suffix.lower() == ".jpg":
        return normalized
    raise ValueError(f"Path is outside the frozen SIIM public subset: {raw_path!r}")


def ensure_local_target(destination: Path, relative_path: str) -> Path:
    root = destination.resolve()
    target = (root / Path(*PurePosixPath(relative_path).parts)).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("Local staging target escaped its destination root") from exc
    return target


def build_remote_manifest_command() -> str:
    """Build a read-only manifest command rooted at the frozen public directory."""
    return (
        f"base={shlex.quote(REMOTE_PUBLIC_ROOT)}; "
        "test -d \"$base/jpeg/train\" && test -d \"$base/jpeg/test\" || exit 41; "
        "for f in train.csv test.csv sample_submission.csv; do "
        "test -f \"$base/$f\" || exit 42; "
        "printf '%s\\t%s\\n' \"$f\" \"$(stat -c %s \"$base/$f\")\"; done; "
        "find \"$base/jpeg\" -mindepth 2 -maxdepth 2 -type f "
        "\\( -path \"$base/jpeg/train/*.jpg\" -o "
        "-path \"$base/jpeg/test/*.jpg\" \\) "
        "-printf '%P\\t%s\\n' | sed -e 's#^#jpeg/#' | sort"
    )


def parse_remote_manifest(output: str) -> list[dict[str, Any]]:
    """Parse and validate the complete allowlisted public-data manifest."""
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in output.splitlines():
        relative, separator, raw_size = line.rpartition("\t")
        if not separator:
            raise RuntimeError("Remote public manifest contains a malformed line")
        relative = validate_relative_public_path(relative)
        try:
            size = int(raw_size)
        except ValueError as exc:
            raise RuntimeError("Remote public manifest contains an invalid size") from exc
        if size < 1 or relative in seen:
            raise RuntimeError("Remote public manifest contains an invalid entry")
        seen.add(relative)
        entries.append({"path": relative, "size": size})
    csv_paths = {item["path"] for item in entries if item["path"].endswith(".csv")}
    image_counts = {
        prefix: sum(item["path"].startswith(prefix + "/") for item in entries)
        for prefix in ALLOWED_IMAGE_PREFIXES
    }
    if csv_paths != ALLOWED_CSV_FILES or any(value < 1 for value in image_counts.values()):
        raise RuntimeError("Remote public manifest is incomplete")
    return entries


def read_remote_manifest() -> list[dict[str, Any]]:
    command = build_remote_manifest_command()
    client = connect_ssh(load_bound_siim_hpc_config(), timeout=30)
    try:
        _, stdout, stderr = client.exec_command(command, timeout=180)
        status = stdout.channel.recv_exit_status()
        output = stdout.read().decode("utf-8", "strict")
        error = stderr.read().decode("utf-8", "replace").strip()
    finally:
        client.close()
    if status:
        raise RuntimeError(f"Remote public manifest failed with exit={status}: {error[-300:]}")
    return parse_remote_manifest(output)


def build_inventory(entries: list[dict[str, Any]]) -> dict[str, Any]:
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema": "evomind.siim.public_staging_inventory.v1",
        "created_at": now_iso(),
        "competition_id": COMPETITION_ID,
        "remote_public_root": REMOTE_PUBLIC_ROOT,
        "file_count": len(entries),
        "total_bytes": sum(int(item["size"]) for item in entries),
        "train_jpeg_count": sum(item["path"].startswith("jpeg/train/") for item in entries),
        "test_jpeg_count": sum(item["path"].startswith("jpeg/test/") for item in entries),
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        "entries": entries,
        "private_paths_requested": False,
        "remote_writes_performed": False,
    }


class Progress:
    def __init__(self, total_files: int, total_bytes: int) -> None:
        self.lock = threading.Lock()
        self.total_files = total_files
        self.total_bytes = total_bytes
        self.completed_files = 0
        self.completed_bytes = 0
        self.downloaded_bytes = 0
        self.resumed_files = 0
        self.skipped_files = 0
        self.errors: list[dict[str, str]] = []

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "total_files": self.total_files,
                "total_bytes": self.total_bytes,
                "completed_files": self.completed_files,
                "completed_bytes": self.completed_bytes,
                "downloaded_bytes_this_run": self.downloaded_bytes,
                "resumed_files": self.resumed_files,
                "skipped_files": self.skipped_files,
                "errors": list(self.errors),
            }


def stage_entries(
    entries: list[dict[str, Any]],
    *,
    destination: Path,
    workers: int,
    heartbeat_path: Path,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    required_total = sum(int(item["size"]) for item in entries)
    existing_complete = 0
    pending: queue.Queue[dict[str, Any]] = queue.Queue()
    progress = Progress(len(entries), required_total)
    for item in entries:
        target = ensure_local_target(destination, item["path"])
        if target.is_file() and target.stat().st_size == int(item["size"]):
            existing_complete += int(item["size"])
            progress.completed_files += 1
            progress.completed_bytes += int(item["size"])
            progress.skipped_files += 1
        else:
            pending.put(item)
    remaining_bytes = required_total - existing_complete
    free_bytes = shutil.disk_usage(destination).free
    if free_bytes < remaining_bytes + 5 * 1024**3:
        raise RuntimeError(
            f"Insufficient free space: free={free_bytes} remaining={remaining_bytes}"
        )

    def worker(worker_id: int) -> None:
        config = load_bound_siim_hpc_config()
        client = connect_ssh(config, timeout=30)
        try:
            with client.open_sftp() as sftp:
                while True:
                    try:
                        item = pending.get_nowait()
                    except queue.Empty:
                        return
                    relative = item["path"]
                    expected_size = int(item["size"])
                    target = ensure_local_target(destination, relative)
                    part = target.with_suffix(target.suffix + ".part")
                    try:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if target.exists() and target.stat().st_size != expected_size:
                            if not part.exists():
                                target.replace(part)
                            else:
                                quarantine = target.with_suffix(
                                    target.suffix + f".size_mismatch.{int(time.time())}"
                                )
                                target.replace(quarantine)
                        offset = part.stat().st_size if part.is_file() else 0
                        if offset > expected_size:
                            quarantine = part.with_suffix(
                                part.suffix + f".oversize.{int(time.time())}"
                            )
                            part.replace(quarantine)
                            offset = 0
                        remote_path = posixpath.join(REMOTE_PUBLIC_ROOT, relative)
                        with sftp.open(remote_path, "rb") as remote_handle:
                            remote_handle.seek(offset)
                            with part.open("ab" if offset else "wb") as local_handle:
                                while True:
                                    block = remote_handle.read(1024 * 1024)
                                    if not block:
                                        break
                                    local_handle.write(block)
                                    with progress.lock:
                                        progress.downloaded_bytes += len(block)
                                local_handle.flush()
                                os.fsync(local_handle.fileno())
                        if part.stat().st_size != expected_size:
                            raise RuntimeError(
                                f"Size mismatch after transfer: {part.stat().st_size}/{expected_size}"
                            )
                        part.replace(target)
                        with progress.lock:
                            progress.completed_files += 1
                            progress.completed_bytes += expected_size
                            if offset:
                                progress.resumed_files += 1
                    except Exception as exc:
                        with progress.lock:
                            progress.errors.append(
                                {
                                    "worker": str(worker_id),
                                    "path": relative,
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            )
                    finally:
                        pending.task_done()
        finally:
            client.close()

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, worker_id) for worker_id in range(workers)]
        while any(not future.done() for future in futures):
            state = progress.snapshot()
            write_json_atomic(
                heartbeat_path,
                {
                    "schema": "evomind.siim.public_staging_heartbeat.v1",
                    "created_at": now_iso(),
                    "status": "running",
                    "pid": os.getpid(),
                    "workers": workers,
                    "destination": str(destination.resolve()),
                    "elapsed_seconds": time.monotonic() - started,
                    **state,
                    "private_paths_requested": False,
                    "remote_writes_performed": False,
                    "process_signals_sent": 0,
                },
            )
            time.sleep(10)
        for future in futures:
            future.result()
    state = progress.snapshot()
    complete = (
        not state["errors"]
        and state["completed_files"] == state["total_files"]
        and state["completed_bytes"] == state["total_bytes"]
    )
    return {
        "status": "size_verified_complete" if complete else "partial_failure",
        "elapsed_seconds": time.monotonic() - started,
        "free_bytes_before": free_bytes,
        **state,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--heartbeat", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= int(args.workers) <= 8:
        raise ValueError("SIIM staging workers must be in [1, 8]")
    destination = args.destination.resolve()
    allowed_local_root = (
        PROJECT_ROOT / "workspace" / "local_gpu" / "mlebench_official_data"
    ).resolve()
    try:
        destination.relative_to(allowed_local_root)
    except ValueError as exc:
        raise ValueError("SIIM staging destination is outside the local data root") from exc
    report_path = args.report or destination.parent.parent / "public_staging_report.json"
    heartbeat_path = args.heartbeat or destination.parent.parent / "public_staging_heartbeat.json"
    entries = read_remote_manifest()
    inventory = build_inventory(entries)
    inventory_path = destination.parent.parent / "public_staging_inventory.json"
    write_json_atomic(inventory_path, inventory)
    if args.inventory_only:
        report = {
            **inventory,
            "status": "inventory_complete",
            "destination": str(destination),
            "inventory_path": str(inventory_path),
            "inventory_sha256": sha256_file(inventory_path),
        }
        write_json_atomic(report_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    staging = stage_entries(
        entries,
        destination=destination,
        workers=int(args.workers),
        heartbeat_path=heartbeat_path,
    )
    report = {
        "schema": "evomind.siim.public_staging.v1",
        "created_at": now_iso(),
        "competition_id": COMPETITION_ID,
        "destination": str(destination),
        "inventory_path": str(inventory_path),
        "inventory_sha256": sha256_file(inventory_path),
        "inventory_manifest_sha256": inventory["manifest_sha256"],
        "workers": int(args.workers),
        **staging,
        "private_paths_requested": False,
        "remote_writes_performed": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
    }
    write_json_atomic(report_path, report)
    write_json_atomic(
        heartbeat_path,
        {
            "schema": "evomind.siim.public_staging_heartbeat.v1",
            "created_at": now_iso(),
            "pid": os.getpid(),
            **report,
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "size_verified_complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
