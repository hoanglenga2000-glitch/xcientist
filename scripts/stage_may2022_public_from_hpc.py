#!/usr/bin/env python3
"""Stage the frozen May-2022 public CSV subset from the dedicated HPC root.

The transfer is deliberately single-worker, read-only on the remote side, and
resume-safe through ``.part`` files.  Size and SHA-256 are verified before an
atomic rename exposes a completed local CSV.
"""

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
from typing import Any, Sequence

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

COMPETITION_ID = "tabular-playground-series-may-2022"
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
EXPECTED_PUBLIC_FILES = {
    "train.csv": 283_303_880,
    "test.csv": 35_229_612,
    "sample_submission.csv": 1_100_010,
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
BLOCK_SIZE = 1024 * 1024
HEARTBEAT_INTERVAL_SECONDS = 5.0


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
    if (
        not raw_path
        or candidate.is_absolute()
        or ".." in candidate.parts
        or len(candidate.parts) != 1
    ):
        raise ValueError(f"Unsafe May-2022 public path: {raw_path!r}")
    normalized = candidate.as_posix()
    if normalized not in EXPECTED_PUBLIC_FILES:
        raise ValueError(
            f"Path is outside the frozen May-2022 public subset: {raw_path!r}"
        )
    return normalized


def local_target(destination: Path, relative: str) -> Path:
    relative = validate_relative_path(relative)
    root = destination.resolve()
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("May-2022 staging target escaped the destination") from exc
    return target


def build_remote_manifest_command() -> str:
    checks = " ".join(shlex.quote(name) for name in EXPECTED_PUBLIC_FILES)
    return (
        f"base={shlex.quote(REMOTE_PUBLIC_ROOT)}; "
        f"for f in {checks}; do "
        "test -f \"$base/$f\" || exit 42; "
        "size=$(stat -c %s -- \"$base/$f\") || exit 43; "
        "hash=$(sha256sum -- \"$base/$f\" | cut -d' ' -f1) || exit 44; "
        "printf '%s\\t%s\\t%s\\n' \"$f\" \"$size\" \"$hash\"; "
        "done"
    )


def parse_remote_manifest(output: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            raise RuntimeError("May-2022 remote manifest line is malformed")
        relative = validate_relative_path(parts[0])
        try:
            size = int(parts[1])
        except ValueError as exc:
            raise RuntimeError("May-2022 remote manifest size is invalid") from exc
        digest = parts[2].strip().lower()
        if relative in seen:
            raise RuntimeError("May-2022 remote manifest contains a duplicate")
        if size != EXPECTED_PUBLIC_FILES[relative]:
            raise RuntimeError(
                f"May-2022 public size changed for {relative}: "
                f"{size}/{EXPECTED_PUBLIC_FILES[relative]}"
            )
        if not SHA256_PATTERN.fullmatch(digest):
            raise RuntimeError("May-2022 remote manifest SHA-256 is invalid")
        seen.add(relative)
        entries.append({"path": relative, "size": size, "sha256": digest})
    if seen != set(EXPECTED_PUBLIC_FILES):
        raise RuntimeError("May-2022 remote public manifest is incomplete")
    return sorted(entries, key=lambda item: item["path"])


def read_remote_manifest() -> list[dict[str, Any]]:
    client = connect_ssh(load_gpu_ssh_config(), timeout=30)
    try:
        _, stdout, stderr = client.exec_command(
            build_remote_manifest_command(), timeout=900
        )
        status = stdout.channel.recv_exit_status()
        output = stdout.read().decode("utf-8", "strict")
        error = stderr.read().decode("utf-8", "replace").strip()
    finally:
        client.close()
    if status:
        raise RuntimeError(
            f"May-2022 remote manifest failed exit={status}: {error[-300:]}"
        )
    return parse_remote_manifest(output)


def build_inventory(entries: list[dict[str, Any]]) -> dict[str, Any]:
    canonical = json.dumps(
        entries, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema": "evomind.may2022.public_staging_inventory.v1",
        "created_at": now_iso(),
        "competition_id": COMPETITION_ID,
        "remote_public_root": REMOTE_PUBLIC_ROOT,
        "worker_count": 1,
        "file_count": len(entries),
        "total_bytes": sum(int(entry["size"]) for entry in entries),
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        "entries": entries,
        "private_paths_requested": False,
        "remote_writes_performed": False,
        "process_signals_sent": 0,
    }


def _heartbeat_payload(
    *,
    destination: Path,
    entries: list[dict[str, Any]],
    started: float,
    completed_files: int,
    completed_bytes: int,
    downloaded_bytes: int,
    skipped_files: int,
    resumed_files: int,
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "schema": "evomind.may2022.public_staging_heartbeat.v1",
        "created_at": now_iso(),
        "status": "running",
        "pid": os.getpid(),
        "worker_count": 1,
        "destination": str(destination.resolve()),
        "elapsed_seconds": time.monotonic() - started,
        "completed_files": completed_files,
        "total_files": len(entries),
        "completed_bytes": completed_bytes,
        "total_bytes": sum(int(entry["size"]) for entry in entries),
        "downloaded_bytes_this_run": downloaded_bytes,
        "skipped_files": skipped_files,
        "resumed_files": resumed_files,
        "errors": errors,
        "private_paths_requested": False,
        "remote_writes_performed": False,
        "process_signals_sent": 0,
    }


def stage_entries(
    entries: list[dict[str, Any]],
    *,
    destination: Path,
    heartbeat_path: Path,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    completed_files = 0
    completed_bytes = 0
    downloaded_bytes = 0
    skipped_files = 0
    resumed_files = 0
    errors: list[dict[str, str]] = []
    client = connect_ssh(load_gpu_ssh_config(), timeout=30)
    try:
        with client.open_sftp() as sftp:
            for entry in entries:
                relative = str(entry["path"])
                expected_size = int(entry["size"])
                expected_sha256 = str(entry["sha256"])
                target = local_target(destination, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                if (
                    target.is_file()
                    and target.stat().st_size == expected_size
                    and sha256_file(target) == expected_sha256
                ):
                    completed_files += 1
                    completed_bytes += expected_size
                    skipped_files += 1
                    continue
                if target.exists():
                    quarantine = target.with_suffix(
                        target.suffix + f".mismatch.{int(time.time())}"
                    )
                    target.replace(quarantine)
                part = target.with_suffix(target.suffix + ".part")
                offset = part.stat().st_size if part.is_file() else 0
                if offset > expected_size:
                    quarantine = part.with_suffix(
                        part.suffix + f".oversize.{int(time.time())}"
                    )
                    part.replace(quarantine)
                    offset = 0
                try:
                    remote_path = posixpath.join(REMOTE_PUBLIC_ROOT, relative)
                    last_heartbeat = 0.0
                    with sftp.open(remote_path, "rb") as remote:
                        remote.seek(offset)
                        with part.open("ab" if offset else "wb") as local:
                            while True:
                                block = remote.read(BLOCK_SIZE)
                                if not block:
                                    break
                                local.write(block)
                                downloaded_bytes += len(block)
                                now = time.monotonic()
                                if now - last_heartbeat >= HEARTBEAT_INTERVAL_SECONDS:
                                    write_json_atomic(
                                        heartbeat_path,
                                        _heartbeat_payload(
                                            destination=destination,
                                            entries=entries,
                                            started=started,
                                            completed_files=completed_files,
                                            completed_bytes=completed_bytes,
                                            downloaded_bytes=downloaded_bytes,
                                            skipped_files=skipped_files,
                                            resumed_files=resumed_files,
                                            errors=errors,
                                        ),
                                    )
                                    last_heartbeat = now
                            local.flush()
                            os.fsync(local.fileno())
                    actual_size = part.stat().st_size
                    if actual_size != expected_size:
                        raise RuntimeError(
                            f"May-2022 staged size mismatch: {actual_size}/{expected_size}"
                        )
                    actual_sha256 = sha256_file(part)
                    if actual_sha256 != expected_sha256:
                        quarantine = part.with_suffix(
                            part.suffix + f".sha256_mismatch.{int(time.time())}"
                        )
                        part.replace(quarantine)
                        raise RuntimeError("May-2022 staged SHA-256 mismatch")
                    part.replace(target)
                    completed_files += 1
                    completed_bytes += expected_size
                    resumed_files += int(offset > 0)
                except Exception as exc:
                    errors.append(
                        {"path": relative, "error": f"{type(exc).__name__}: {exc}"}
                    )
                write_json_atomic(
                    heartbeat_path,
                    _heartbeat_payload(
                        destination=destination,
                        entries=entries,
                        started=started,
                        completed_files=completed_files,
                        completed_bytes=completed_bytes,
                        downloaded_bytes=downloaded_bytes,
                        skipped_files=skipped_files,
                        resumed_files=resumed_files,
                        errors=errors,
                    ),
                )
    finally:
        client.close()
    total_bytes = sum(int(entry["size"]) for entry in entries)
    complete = (
        not errors
        and completed_files == len(entries)
        and completed_bytes == total_bytes
    )
    return {
        "status": "hash_verified_complete" if complete else "partial_failure",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--heartbeat", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    destination = args.destination.resolve()
    allowed = (
        PROJECT_ROOT / "workspace" / "local_gpu" / "mlebench_official_data"
    ).resolve()
    try:
        destination.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(
            "May-2022 staging destination is outside the local data root"
        ) from exc
    competition_root = destination.parent.parent
    report_path = (args.report or competition_root / "public_staging_report.json").resolve()
    heartbeat_path = (
        args.heartbeat or competition_root / "public_staging_heartbeat.json"
    ).resolve()
    entries = read_remote_manifest()
    inventory = build_inventory(entries)
    inventory_path = competition_root / "public_staging_inventory.json"
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
    result = stage_entries(
        entries,
        destination=destination,
        heartbeat_path=heartbeat_path,
    )
    report = {
        "schema": "evomind.may2022.public_staging.v1",
        "created_at": now_iso(),
        "competition_id": COMPETITION_ID,
        "destination": str(destination),
        "inventory_path": str(inventory_path),
        "inventory_sha256": sha256_file(inventory_path),
        "inventory_manifest_sha256": inventory["manifest_sha256"],
        "worker_count": 1,
        **result,
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
            "schema": "evomind.may2022.public_staging_heartbeat.v1",
            "created_at": now_iso(),
            "pid": os.getpid(),
            **report,
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "hash_verified_complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
