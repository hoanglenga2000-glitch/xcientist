#!/usr/bin/env python3
"""Read-only, resume-safe staging of the small Leaf public benchmark subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import re
import shlex
import sys
import time
from datetime import datetime
from importlib import import_module
from pathlib import Path, PurePosixPath
from typing import Any

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

COMPETITION_ID = "leaf-classification"
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
ROOT_FILES = {"train.csv", "test.csv", "sample_submission.csv", "description.md"}
IMAGE_PATTERN = re.compile(r"^images/[0-9]+\.jpg$")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_relative_path(raw_path: str) -> str:
    candidate = PurePosixPath(str(raw_path))
    if not raw_path or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Unsafe Leaf public path: {raw_path!r}")
    normalized = candidate.as_posix()
    if normalized in ROOT_FILES or IMAGE_PATTERN.fullmatch(normalized):
        return normalized
    raise ValueError(f"Path is outside the frozen Leaf public subset: {raw_path!r}")


def local_target(destination: Path, relative: str) -> Path:
    root = destination.resolve()
    target = (root / Path(*PurePosixPath(relative).parts)).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("Leaf staging target escaped the destination") from exc
    return target


def remote_manifest() -> list[dict[str, Any]]:
    command = (
        f"base={shlex.quote(REMOTE_PUBLIC_ROOT)}; "
        "test -d \"$base/images\" || exit 41; "
        "find \"$base\" -type f -printf '%P\\t%s\\n' | sort"
    )
    client = connect_ssh(load_gpu_ssh_config(), timeout=30)
    try:
        _, stdout, stderr = client.exec_command(command, timeout=180)
        status = stdout.channel.recv_exit_status()
        output = stdout.read().decode("utf-8", "strict")
        error = stderr.read().decode("utf-8", "replace").strip()
    finally:
        client.close()
    if status:
        raise RuntimeError(f"Leaf remote manifest failed exit={status}: {error[-300:]}")
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in output.splitlines():
        relative, separator, raw_size = line.rpartition("\t")
        if not separator:
            raise RuntimeError("Leaf remote manifest line is malformed")
        relative = validate_relative_path(relative)
        size = int(raw_size)
        if size < 1 or relative in seen:
            raise RuntimeError("Leaf remote manifest entry is invalid")
        seen.add(relative)
        entries.append({"path": relative, "size": size})
    paths = {entry["path"] for entry in entries}
    image_count = sum(entry["path"].startswith("images/") for entry in entries)
    if not ROOT_FILES <= paths or image_count < 1:
        raise RuntimeError("Leaf remote public manifest is incomplete")
    return entries


def stage(entries: list[dict[str, Any]], destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    heartbeat = destination.parent.parent / "public_staging_heartbeat.json"
    total_bytes = sum(int(entry["size"]) for entry in entries)
    completed_files = 0
    completed_bytes = 0
    downloaded_bytes = 0
    skipped_files = 0
    resumed_files = 0
    errors: list[dict[str, str]] = []
    client = connect_ssh(load_gpu_ssh_config(), timeout=30)
    started = time.monotonic()
    try:
        with client.open_sftp() as sftp:
            for index, entry in enumerate(entries):
                relative = entry["path"]
                expected = int(entry["size"])
                target = local_target(destination, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_file() and target.stat().st_size == expected:
                    completed_files += 1
                    completed_bytes += expected
                    skipped_files += 1
                    continue
                part = target.with_suffix(target.suffix + ".part")
                offset = part.stat().st_size if part.is_file() else 0
                if offset > expected:
                    part.replace(part.with_suffix(part.suffix + f".oversize.{int(time.time())}"))
                    offset = 0
                try:
                    remote_path = posixpath.join(REMOTE_PUBLIC_ROOT, relative)
                    with sftp.open(remote_path, "rb") as remote:
                        remote.seek(offset)
                        with part.open("ab" if offset else "wb") as local:
                            while True:
                                block = remote.read(1024 * 1024)
                                if not block:
                                    break
                                local.write(block)
                                downloaded_bytes += len(block)
                            local.flush()
                            os.fsync(local.fileno())
                    if part.stat().st_size != expected:
                        raise RuntimeError(
                            f"Leaf staged size mismatch: {part.stat().st_size}/{expected}"
                        )
                    part.replace(target)
                    completed_files += 1
                    completed_bytes += expected
                    resumed_files += int(offset > 0)
                except Exception as exc:
                    errors.append(
                        {
                            "path": relative,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                if index % 20 == 0 or index + 1 == len(entries):
                    write_json_atomic(
                        heartbeat,
                        {
                            "schema": "evomind.leaf.public_staging_heartbeat.v1",
                            "created_at": now_iso(),
                            "status": "running",
                            "pid": os.getpid(),
                            "completed_files": completed_files,
                            "total_files": len(entries),
                            "completed_bytes": completed_bytes,
                            "total_bytes": total_bytes,
                            "downloaded_bytes_this_run": downloaded_bytes,
                            "skipped_files": skipped_files,
                            "resumed_files": resumed_files,
                            "errors": errors,
                            "private_paths_requested": False,
                            "remote_writes_performed": False,
                            "process_signals_sent": 0,
                        },
                    )
    finally:
        client.close()
    complete = not errors and completed_files == len(entries) and completed_bytes == total_bytes
    return {
        "status": "size_verified_complete" if complete else "partial_failure",
        "elapsed_seconds": time.monotonic() - started,
        "completed_files": completed_files,
        "total_files": len(entries),
        "completed_bytes": completed_bytes,
        "total_bytes": total_bytes,
        "downloaded_bytes_this_run": downloaded_bytes,
        "skipped_files": skipped_files,
        "resumed_files": resumed_files,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    destination = args.destination.resolve()
    allowed = (
        PROJECT_ROOT / "workspace" / "local_gpu" / "mlebench_official_data"
    ).resolve()
    try:
        destination.relative_to(allowed)
    except ValueError as exc:
        raise ValueError("Leaf staging destination is outside the local data root") from exc
    entries = remote_manifest()
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    inventory = {
        "schema": "evomind.leaf.public_staging_inventory.v1",
        "created_at": now_iso(),
        "competition_id": COMPETITION_ID,
        "remote_public_root": REMOTE_PUBLIC_ROOT,
        "file_count": len(entries),
        "total_bytes": sum(int(entry["size"]) for entry in entries),
        "image_count": sum(entry["path"].startswith("images/") for entry in entries),
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        "entries": entries,
        "private_paths_requested": False,
        "remote_writes_performed": False,
    }
    competition_root = destination.parent.parent
    inventory_path = competition_root / "public_staging_inventory.json"
    write_json_atomic(inventory_path, inventory)
    result = stage(entries, destination)
    report = {
        "schema": "evomind.leaf.public_staging.v1",
        "created_at": now_iso(),
        "competition_id": COMPETITION_ID,
        "destination": str(destination),
        "inventory_path": str(inventory_path),
        "inventory_sha256": sha256_file(inventory_path),
        "inventory_manifest_sha256": inventory["manifest_sha256"],
        **result,
        "private_paths_requested": False,
        "remote_writes_performed": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
    }
    report_path = competition_root / "public_staging_report.json"
    heartbeat_path = competition_root / "public_staging_heartbeat.json"
    write_json_atomic(report_path, report)
    write_json_atomic(
        heartbeat_path,
        {
            "schema": "evomind.leaf.public_staging_heartbeat.v1",
            "created_at": now_iso(),
            "pid": os.getpid(),
            **report,
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "size_verified_complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
