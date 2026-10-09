"""Administrative reconciliation for the EvoMind GPU execution ledger.

The runtime refuses every new GPU reservation while any ledger row is left in a
non-terminal ``uncertain`` (remote settlement lost) or ``exceeded`` (the
operation ran past its reservation) state.  No recovery path shipped with the
runtime, so a single overshoot permanently blocks the study.  This script adds
the missing, auditable reconciliation path.

Usage (dry run by default)::

    python reconcile_gpu_budget_ledger.py --db <path> --list
    python reconcile_gpu_budget_ledger.py --db <path> \
        --settle <operation-id> --final-status passed \
        --reason "..." [--evidence <path/url>] --apply

``--apply`` copies the SQLite database (plus -wal/-shm) to a timestamped
backup, flips exactly one row to a terminal status, and writes a JSON receipt
next to the database under ``gpu-ledger-reconciliations/``.  Charged seconds
are never altered, so the study accounting stays truthful.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path

GPU_LIMIT_SECONDS = 12 * 3600
ENGINEERING_LIMIT_SECONDS = 8 * 3600
TERMINAL = {"passed", "failed"}
OPEN_STATES = {"uncertain", "exceeded"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(connection: sqlite3.Connection) -> list[tuple]:
    return connection.execute(
        "select id, study, run_id, kind, reserved, charged, status, boot_id, "
        "controller_pid, created from gpu_operations order by created"
    ).fetchall()


def _summary(rows: list[tuple]) -> dict:
    committed = sum(r[4] if r[6] == "reserved" else r[5] for r in rows)
    engineering = sum(
        (r[4] if r[6] == "reserved" else r[5]) for r in rows if r[3] == "engineering"
    )
    return {
        "operations": len(rows),
        "charged_seconds": round(committed, 1),
        "engineering_seconds": round(engineering, 1),
        "gpu_limit_seconds": GPU_LIMIT_SECONDS,
        "engineering_limit_seconds": ENGINEERING_LIMIT_SECONDS,
        "gpu_remaining_seconds": round(GPU_LIMIT_SECONDS - committed, 1),
        "engineering_remaining_seconds": round(
            ENGINEERING_LIMIT_SECONDS - engineering, 1
        ),
        "reconciliation_required": any(r[6] in OPEN_STATES for r in rows),
        "pending_reserved": [r[0] for r in rows if r[6] == "reserved"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="path to gpu_budget.sqlite3")
    parser.add_argument("--list", action="store_true", help="print every ledger row")
    parser.add_argument("--settle", help="operation id (or unique prefix) to reconcile")
    parser.add_argument("--final-status", choices=sorted(TERMINAL))
    parser.add_argument("--reason", default="")
    parser.add_argument("--evidence", default="")
    parser.add_argument("--operator", default="")
    parser.add_argument("--apply", action="store_true", help="write backup + update")
    args = parser.parse_args()

    db = Path(args.db)
    if not db.is_file():
        print(f"ledger not found: {db}", file=sys.stderr)
        return 2
    connection = sqlite3.connect(str(db))
    rows = _rows(connection)

    if args.list or not args.settle:
        for row in rows:
            print(json.dumps({
                "id": row[0], "study": row[1], "run_id": row[2], "kind": row[3],
                "reserved": row[4], "charged": row[5], "status": row[6],
                "boot_id": row[7][:12], "pid": row[8],
                "created": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(row[9])),
            }, ensure_ascii=False))
        print(json.dumps(_summary(rows), ensure_ascii=False))
        connection.close()
        return 0

    matches = [r for r in rows if r[0].startswith(args.settle)]
    if len(matches) != 1:
        print(f"expected exactly one operation for prefix {args.settle!r}, "
              f"found {len(matches)}", file=sys.stderr)
        connection.close()
        return 2
    row = matches[0]
    if row[6] not in OPEN_STATES:
        print(f"operation {row[0]} is already terminal ({row[6]}); nothing to do")
        connection.close()
        return 0
    before = {
        "id": row[0], "study": row[1], "run_id": row[2], "kind": row[3],
        "reserved": row[4], "charged": row[5], "status": row[6], "boot_id": row[7],
        "controller_pid": row[8], "created": row[9],
    }
    print("about to reconcile:")
    print(json.dumps(before, ensure_ascii=False, indent=1))
    if not args.apply:
        print("dry run only - re-run with --apply to write backup and settle")
        connection.close()
        return 0

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backups = []
    for suffix in ("", "-wal", "-shm"):
        source = Path(str(db) + suffix)
        if source.is_file():
            target = source.with_name(f"{source.name}.bak-{stamp}")
            shutil.copy2(source, target)
            backups.append({"path": str(target), "sha256": _sha256(target),
                            "bytes": target.stat().st_size})

    connection.execute("BEGIN IMMEDIATE")
    connection.execute(
        "update gpu_operations set status=? where id=? and status in ('uncertain','exceeded')",
        (args.final_status, row[0]),
    )
    connection.commit()
    after_rows = _rows(connection)
    connection.close()

    receipt = {
        "schema": "evomind.gpu_budget_reconciliation.v1",
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "database": str(db),
        "backups": backups,
        "operation_before": before,
        "operation_after": {**before, "status": args.final_status},
        "final_status": args.final_status,
        "reason": args.reason,
        "evidence": args.evidence,
        "operator": args.operator,
        "summary_after": _summary(after_rows),
    }
    receipt_dir = db.parent / "gpu-ledger-reconciliations"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"{stamp}-{row[0][:12]}.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print("reconciled:", row[0], "->", args.final_status)
    print("receipt:", receipt_path)
    print(json.dumps(_summary(after_rows), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
