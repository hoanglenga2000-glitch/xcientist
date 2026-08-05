from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _iso_from_sqlite(value: Any) -> str:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    text = str(value or "")
    if text.endswith("+00:00"):
        return text[:-6] + "Z"
    return text


def _metadata(value: Any) -> dict[str, Any]:
    try:
        decoded = json.loads(str(value)) if value else {}
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _record(row: sqlite3.Row) -> dict[str, Any]:
    result: dict[str, Any] = {
        "action_id": row["id"],
        "action": row["action"],
    }
    if row["task_id"] is not None:
        result["task_id"] = row["task_id"]
    if row["run_id"] is not None:
        result["run_id"] = row["run_id"]
    result["message"] = row["message"]
    if row["artifact_path"] is not None:
        result["artifact"] = row["artifact_path"]
    result["metadata"] = _metadata(row["metadata_json"])
    result["at"] = _iso_from_sqlite(row["created_at"])
    return result


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def reconcile(database: Path, runtime_root: Path) -> dict[str, Any]:
    database = database.resolve()
    runtime_root = runtime_root.resolve()
    if not database.is_file():
        raise FileNotFoundError(f"SQLite database does not exist: {database}")

    with sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        foreign_key_violations = len(connection.execute("PRAGMA foreign_key_check").fetchall())
        rows = connection.execute(
            """
            SELECT id, action, task_id, run_id, message, artifact_path, metadata_json, created_at
            FROM action_logs
            ORDER BY created_at ASC, id ASC
            """
        ).fetchall()

    records = [_record(row) for row in rows]
    canonical_text = "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records)
    canonical = canonical_text.encode("utf-8")
    mirror = runtime_root / "action_log.jsonl"
    previous = mirror.read_bytes() if mirror.is_file() else b""
    archive: Path | None = None
    if previous != canonical:
        if previous:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            archive = runtime_root / f"action_log.pre-reconcile.{stamp}.jsonl"
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mirror, archive)
        _atomic_write(mirror, canonical)

    checkpoint = {
        "schema": "research_workstation.action_log_checkpoint.v1",
        "sqlite_is_canonical": True,
        "action_count": len(records),
        "last_action_id": records[-1]["action_id"] if records else None,
        "mirror_bytes": len(canonical),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write(runtime_root / "action_log.checkpoint.json", (json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    (runtime_root / "action_log.reconcile-required.json").unlink(missing_ok=True)

    return {
        "schema": "research_workstation.action_log_reconcile.v1",
        "ok": quick_check == "ok" and foreign_key_violations == 0,
        "sqlite_is_canonical": True,
        "quick_check": quick_check,
        "foreign_key_violations": foreign_key_violations,
        "action_count": len(records),
        "previous_mirror_bytes": len(previous),
        "mirror_bytes": len(canonical),
        "mirror_sha256": hashlib.sha256(canonical).hexdigest(),
        "rewritten": previous != canonical,
        "archive_path": str(archive) if archive else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild the JSONL action mirror from canonical SQLite audit rows.")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = reconcile(args.database, args.runtime_root)
    if args.json:
        _atomic_write(args.json.resolve(), (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
