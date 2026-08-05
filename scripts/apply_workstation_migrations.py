from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TABLES = {"tasks", "experiment_runs", "action_logs", "workflows", "gates", "evidence", "reports", "connector_statuses", "settings"}


def validate_additive_migration(name: str, sql: str) -> None:
    """Reject destructive release migrations so code rollback stays valid."""
    without_comments = re.sub(r"--[^\n]*|/\*.*?\*/", " ", sql, flags=re.DOTALL)
    forbidden = {
        "DROP": r"(?:^|;)\s*DROP\s+(?:TABLE|INDEX|VIEW|TRIGGER|COLUMN)\b",
        "TRUNCATE": r"(?:^|;)\s*TRUNCATE\b",
        "DELETE": r"(?:^|;)\s*DELETE\s+FROM\b",
        "UPDATE": r"(?:^|;)\s*UPDATE\s+[\"`\[]?[A-Za-z_]",
        "RENAME": r"(?:^|;)\s*ALTER\s+TABLE\b[^;]*\bRENAME\b",
        "WRITABLE_SCHEMA": r"(?:^|;)\s*PRAGMA\s+writable_schema\b",
    }
    matches = [label for label, pattern in forbidden.items() if re.search(pattern, without_comments, flags=re.IGNORECASE)]
    if matches:
        raise RuntimeError(f"migration {name} is not additive: {', '.join(matches)}")


def migration_root() -> Path:
    candidates = [
        ROOT / "app" / "prisma" / "migrations",
        ROOT / "web" / "research-agent-workstation" / "prisma" / "migrations",
        ROOT / "prisma" / "migrations",
    ]
    found = next((path for path in candidates if path.is_dir()), None)
    if not found:
        raise SystemExit("MIGRATION_FAILED: migration directory not found")
    return found


def default_database() -> Path:
    configured = os.environ.get("DATABASE_URL", "").strip()
    if configured.startswith("file:"):
        raw = configured[5:]
        candidate = Path(raw)
        if candidate.is_absolute():
            return candidate.resolve()
    if (ROOT / "app" / "server.js").is_file():
        return (ROOT / "user-data" / "prisma" / "workstation.db").resolve()
    return (ROOT / "web" / "research-agent-workstation" / "prisma" / "workstation.db").resolve()


def backup_database(database: Path, backup_dir: Path) -> Path | None:
    if not database.is_file():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    destination = backup_dir / f"workstation-{time.strftime('%Y%m%d-%H%M%S')}.db"
    source_connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    target_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(target_connection)
    finally:
        target_connection.close()
        source_connection.close()
    return destination


def table_names(connection: sqlite3.Connection) -> set[str]:
    return {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply additive SQLite workstation migrations with backup and integrity verification.")
    parser.add_argument("--database", type=Path, default=default_database())
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    database = args.database.expanduser().resolve()
    migrations = migration_root()
    backup_dir = (args.backup_dir or database.parent / "backups").expanduser().resolve()
    ordered = sorted(path for path in migrations.iterdir() if path.is_dir() and (path / "migration.sql").is_file())
    if not ordered:
        raise SystemExit("MIGRATION_FAILED: no migration SQL files found")
    if args.dry_run:
        print(json.dumps({"status": "planned", "database": str(database), "migrations": [path.name for path in ordered]}, ensure_ascii=False, indent=2))
        return 0
    database_existed = database.is_file()
    database.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_database(database, backup_dir)
    connection = sqlite3.connect(database, timeout=30)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=30000")
    applied: list[dict[str, str]] = []
    adopted: list[str] = []
    try:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS _workstation_migrations (name TEXT PRIMARY KEY, checksum TEXT NOT NULL, applied_at TEXT NOT NULL)"
        )
        connection.commit()
        for directory in ordered:
            sql_path = directory / "migration.sql"
            sql = sql_path.read_text(encoding="utf-8")
            validate_additive_migration(directory.name, sql)
            checksum = hashlib.sha256(sql.encode("utf-8")).hexdigest()
            existing = connection.execute("SELECT checksum FROM _workstation_migrations WHERE name=?", (directory.name,)).fetchone()
            if existing:
                if existing[0] != checksum:
                    raise RuntimeError(f"migration checksum mismatch: {directory.name}")
                continue
            current_tables = table_names(connection)
            if directory.name.endswith("baseline") and EXPECTED_TABLES.issubset(current_tables):
                connection.execute(
                    "INSERT INTO _workstation_migrations(name, checksum, applied_at) VALUES (?, ?, datetime('now'))",
                    (directory.name, checksum),
                )
                connection.commit()
                adopted.append(directory.name)
                continue
            safe_sql = sql.replace("CREATE INDEX \"", "CREATE INDEX IF NOT EXISTS \"")
            connection.executescript("BEGIN IMMEDIATE;\n" + safe_sql + "\nCOMMIT;")
            connection.execute(
                "INSERT INTO _workstation_migrations(name, checksum, applied_at) VALUES (?, ?, datetime('now'))",
                (directory.name, checksum),
            )
            connection.commit()
            applied.append({"name": directory.name, "checksum": checksum})
        quick = connection.execute("PRAGMA quick_check").fetchone()[0]
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        if quick != "ok" or foreign_keys:
            raise RuntimeError(f"post-migration integrity failed: quick_check={quick}, foreign_keys={len(foreign_keys)}")
    except BaseException:
        connection.close()
        if backup and backup.is_file():
            failed = database.with_suffix(database.suffix + ".failed")
            if database.exists():
                failed.unlink(missing_ok=True)
                database.replace(failed)
            shutil.copy2(backup, database)
        elif not database_existed and database.exists():
            failed = database.with_suffix(database.suffix + ".failed")
            failed.unlink(missing_ok=True)
            database.replace(failed)
        raise
    finally:
        try:
            connection.close()
        except Exception:
            pass
    result = {
        "status": "passed",
        "database": str(database),
        "backup": str(backup) if backup else None,
        "applied": applied,
        "adopted": adopted,
        "quick_check": "ok",
        "foreign_key_violations": 0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
