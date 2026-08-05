#!/usr/bin/env python3
"""Merge a verified SIIM sidecar into the canonical public directory.

The watcher never signals another process and never overwrites an existing
file. Missing canonical files are hard-linked only after the sidecar report is
terminal and every inventory entry in the sidecar has the exact frozen size.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

COMPETITION = "siim-isic-melanoma-classification"
INVENTORY_SCHEMA = "evomind.siim.public_staging_inventory.v1"
REPORT_SCHEMA = "evomind.siim.public_staging.v1"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def relative_path(name: str) -> Path:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe inventory path: {name}")
    if name in {"train.csv", "test.csv", "sample_submission.csv"}:
        return Path(*pure.parts)
    if (
        len(pure.parts) == 3
        and tuple(pure.parts[:2]) in {("jpeg", "train"), ("jpeg", "test")}
        and pure.parts[2].lower().endswith(".jpg")
    ):
        return Path(*pure.parts)
    raise ValueError(f"Non-public SIIM inventory path: {name}")


def load_entries(inventory_path: Path) -> tuple[dict[str, Any], list[tuple[Path, int]]]:
    inventory = read_json(inventory_path)
    if inventory.get("schema") != INVENTORY_SCHEMA:
        raise RuntimeError("Unexpected SIIM inventory schema")
    if inventory.get("competition_id") != COMPETITION:
        raise RuntimeError("Unexpected SIIM competition")
    entries = [
        (relative_path(str(item["path"])), int(item["size"]))
        for item in inventory.get("entries", [])
    ]
    if len(entries) != int(inventory.get("file_count", -1)):
        raise RuntimeError("SIIM inventory file count mismatch")
    if sum(size for _path, size in entries) != int(inventory.get("total_bytes", -1)):
        raise RuntimeError("SIIM inventory byte count mismatch")
    return inventory, entries


def scan(root: Path, entries: Sequence[tuple[Path, int]]) -> dict[str, Any]:
    complete = 0
    complete_bytes = 0
    missing: list[str] = []
    wrong: list[dict[str, Any]] = []
    for relative, expected in entries:
        item = root / relative
        if not item.is_file():
            missing.append(relative.as_posix())
            continue
        actual = item.stat().st_size
        if actual != expected:
            wrong.append(
                {"path": relative.as_posix(), "expected_bytes": expected, "actual_bytes": actual}
            )
            continue
        complete += 1
        complete_bytes += actual
    return {
        "completed_files": complete,
        "completed_bytes": complete_bytes,
        "missing": missing,
        "wrong_size": wrong,
    }


def validate_sidecar_report(
    report_path: Path, inventory: dict[str, Any], inventory_path: Path
) -> dict[str, Any]:
    report = read_json(report_path)
    checks = {
        "schema": report.get("schema") == REPORT_SCHEMA,
        "status": report.get("status") == "size_verified_complete",
        "competition": report.get("competition_id") == COMPETITION,
        "files": report.get("completed_files") == inventory.get("file_count"),
        "bytes": report.get("completed_bytes") == inventory.get("total_bytes"),
        "inventory": report.get("inventory_manifest_sha256")
        == inventory.get("manifest_sha256"),
        "errors": not report.get("errors"),
        "private": report.get("private_paths_requested") is False,
        "signals": report.get("process_signals_sent") == 0,
        "grader": report.get("official_grader_executed") is False,
        "submission": report.get("kaggle_submission_executed") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"SIIM sidecar report failed verification: {checks}")
    return {
        "path": str(report_path),
        "sha256": sha256(report_path),
        "inventory_path": str(inventory_path),
        "inventory_sha256": sha256(inventory_path),
        "checks": checks,
    }


def merge_sidecar(
    *,
    inventory_path: Path,
    sidecar: Path,
    sidecar_report_path: Path,
    destination: Path,
    output: Path,
) -> dict[str, Any]:
    inventory, entries = load_entries(inventory_path)
    report_evidence = validate_sidecar_report(sidecar_report_path, inventory, inventory_path)
    sidecar_scan = scan(sidecar, entries)
    if sidecar_scan["missing"] or sidecar_scan["wrong_size"]:
        raise RuntimeError("SIIM sidecar tree differs from the frozen inventory")

    linked = 0
    already = 0
    linked_bytes = 0
    races = 0
    for relative, expected in entries:
        source = sidecar / relative
        target = destination / relative
        if target.is_file():
            if target.stat().st_size != expected:
                raise RuntimeError(f"Canonical SIIM file has wrong size: {relative.as_posix()}")
            already += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
            linked += 1
            linked_bytes += expected
        except FileExistsError:
            races += 1
            if not target.is_file() or target.stat().st_size != expected:
                raise RuntimeError(f"Canonical SIIM merge race failed: {relative.as_posix()}")

    final = scan(destination, entries)
    complete = (
        final["completed_files"] == inventory["file_count"]
        and final["completed_bytes"] == inventory["total_bytes"]
        and not final["missing"]
        and not final["wrong_size"]
    )
    if not complete:
        raise RuntimeError("Canonical SIIM tree is incomplete after sidecar merge")

    payload = {
        "schema": REPORT_SCHEMA,
        "created_at": now_iso(),
        "status": "size_verified_complete",
        "source": "rate_aware_kaggle_sidecar_v2_hardlink_merge",
        "competition_id": COMPETITION,
        "destination": str(destination.resolve()),
        "total_files": int(inventory["file_count"]),
        "total_bytes": int(inventory["total_bytes"]),
        "completed_files": final["completed_files"],
        "completed_bytes": final["completed_bytes"],
        "inventory_manifest_sha256": inventory["manifest_sha256"],
        "sidecar_report": report_evidence,
        "already_present_files": already,
        "linked_files": linked,
        "linked_bytes": linked_bytes,
        "merge_races_verified": races,
        "errors": [],
        "private_paths_requested": False,
        "remote_writes_performed": False,
        "process_signals_sent": 0,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "human_gate_preserved": True,
    }
    write_json_atomic(output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--sidecar-report", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--deadline-hours", type=int, default=240)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.poll_seconds < 5 or not 1 <= args.deadline_hours <= 240:
        raise ValueError("Invalid SIIM sidecar watcher timing")
    deadline = datetime.now().astimezone() + timedelta(hours=args.deadline_hours)
    while datetime.now().astimezone() < deadline:
        if not args.sidecar_report.is_file():
            write_json_atomic(
                args.status,
                {
                    "schema": "evomind.siim.sidecar_merge_watcher.v1",
                    "created_at": now_iso(),
                    "status": "waiting_for_sidecar_completion",
                    "sidecar_report": str(args.sidecar_report),
                    "process_signals_sent": 0,
                },
            )
            time.sleep(args.poll_seconds)
            continue
        report = read_json(args.sidecar_report)
        if report.get("status") != "size_verified_complete":
            write_json_atomic(
                args.status,
                {
                    "schema": "evomind.siim.sidecar_merge_watcher.v1",
                    "created_at": now_iso(),
                    "status": "sidecar_terminal_not_complete",
                    "sidecar_status": report.get("status"),
                    "sidecar_report_sha256": sha256(args.sidecar_report),
                    "process_signals_sent": 0,
                },
            )
            return 2
        payload = merge_sidecar(
            inventory_path=args.inventory.resolve(),
            sidecar=args.sidecar.resolve(),
            sidecar_report_path=args.sidecar_report.resolve(),
            destination=args.destination.resolve(),
            output=args.output.resolve(),
        )
        write_json_atomic(
            args.status,
            {
                "schema": "evomind.siim.sidecar_merge_watcher.v1",
                "created_at": now_iso(),
                "status": "size_verified_complete",
                "output": str(args.output.resolve()),
                "output_sha256": sha256(args.output.resolve()),
                "completed_files": payload["completed_files"],
                "completed_bytes": payload["completed_bytes"],
                "process_signals_sent": 0,
            },
        )
        return 0
    write_json_atomic(
        args.status,
        {
            "schema": "evomind.siim.sidecar_merge_watcher.v1",
            "created_at": now_iso(),
            "status": "deadline_expired",
            "process_signals_sent": 0,
        },
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
