#!/usr/bin/env python3
"""Apply strictly additive, auditable SQLite release migrations.

The release installer and release verifier both call this module.  Migration
SQL is split with a quote/comment-aware scanner, admitted by a narrow DDL
allowlist, executed under SQLite's authorizer, and checked against a complete
schema fingerprint after every file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

LEDGER_SQL = """
CREATE TABLE IF NOT EXISTS "_prisma_migrations" (
  "id" TEXT PRIMARY KEY NOT NULL,
  "checksum" TEXT NOT NULL,
  "finished_at" DATETIME,
  "migration_name" TEXT NOT NULL,
  "logs" TEXT,
  "rolled_back_at" DATETIME,
  "started_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "applied_steps_count" INTEGER NOT NULL DEFAULT 0
)
"""

BASELINE_TABLES = {
    "tasks", "experiment_runs", "action_logs", "workflows", "gates",
    "evidence", "reports", "connector_statuses", "settings",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def ensure_no_links(path: Path, *, allow_missing: bool = True) -> Path:
    resolved = Path(os.path.abspath(os.path.normpath(str(path.expanduser()))))
    cursor = Path(resolved.anchor)
    for part in resolved.parts[1:]:
        cursor /= part
        if not cursor.exists():
            if allow_missing:
                continue
            raise RuntimeError(f"managed path does not exist: {cursor}")
        if cursor.is_symlink() or getattr(cursor, "is_junction", lambda: False)():
            raise RuntimeError(f"managed path traverses a symlink/reparse point: {cursor}")
    return resolved


def split_sql_statements(sql: str) -> list[str]:
    """Split SQL on semicolons outside strings, identifiers, and comments."""
    statements: list[str] = []
    buffer: list[str] = []
    state = "normal"
    index = 0
    while index < len(sql):
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < len(sql) else ""
        if state == "line_comment":
            if char in "\r\n":
                state = "normal"
                buffer.append(" ")
            index += 1
            continue
        if state == "block_comment":
            if char == "*" and next_char == "/":
                state = "normal"
                buffer.append(" ")
                index += 2
            else:
                index += 1
            continue
        if state == "single":
            buffer.append(char)
            if char == "'":
                if next_char == "'":
                    buffer.append(next_char)
                    index += 2
                    continue
                state = "normal"
            index += 1
            continue
        if state in {"double", "backtick", "bracket"}:
            buffer.append(char)
            closing = {"double": '"', "backtick": "`", "bracket": "]"}[state]
            if char == closing:
                if state != "bracket" and next_char == closing:
                    buffer.append(next_char)
                    index += 2
                    continue
                state = "normal"
            index += 1
            continue
        if char == "-" and next_char == "-":
            state = "line_comment"
            index += 2
            continue
        if char == "/" and next_char == "*":
            state = "block_comment"
            index += 2
            continue
        if char == "'":
            state = "single"
            buffer.append(char)
        elif char == '"':
            state = "double"
            buffer.append(char)
        elif char == "`":
            state = "backtick"
            buffer.append(char)
        elif char == "[":
            state = "bracket"
            buffer.append(char)
        elif char == ";":
            statement = "".join(buffer).strip()
            if statement:
                statements.append(statement)
            buffer.clear()
        else:
            buffer.append(char)
        index += 1
    if state not in {"normal", "line_comment"}:
        raise RuntimeError("migration SQL contains an unterminated quote or comment")
    tail = "".join(buffer).strip()
    if tail:
        statements.append(tail)
    return statements


IDENTIFIER = r'(?:"(?:[^"]|"")+"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*)'
CREATE_TABLE = re.compile(
    rf"^CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?{IDENTIFIER}\s*\(", re.IGNORECASE | re.DOTALL
)
CREATE_INDEX = re.compile(
    rf"^CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?{IDENTIFIER}\s+ON\s+{IDENTIFIER}\s*\(",
    re.IGNORECASE | re.DOTALL,
)
ALTER_ADD = re.compile(
    rf"^ALTER\s+TABLE\s+{IDENTIFIER}\s+ADD\s+(?:COLUMN\s+)?{IDENTIFIER}(?:\s|$)", re.IGNORECASE | re.DOTALL
)


def validate_statement(name: str, statement: str) -> None:
    normalized = statement.lstrip()
    upper = normalized.upper()
    first = re.match(r"[A-Z_]+", upper)
    label = first.group(0) if first else "UNKNOWN"
    if upper.startswith("ALTER TABLE") and not ALTER_ADD.match(normalized):
        if re.search(r"\bDROP\b", upper):
            label = "ALTER_DROP"
        elif re.search(r"\bRENAME\b", upper):
            label = "RENAME"
        else:
            label = "ALTER_NON_ADD"
        raise RuntimeError(f"migration {name} is not additive: {label}")
    if CREATE_TABLE.match(normalized):
        if re.match(r"^CREATE\s+(?:VIRTUAL|TEMP|TEMPORARY)\b", normalized, re.IGNORECASE):
            raise RuntimeError(f"migration {name} is not additive: CREATE_VARIANT")
        return
    if CREATE_INDEX.match(normalized):
        return
    if ALTER_ADD.match(normalized):
        return
    if upper.startswith("CREATE TRIGGER"):
        label = "TRIGGER"
    elif upper.startswith("WITH"):
        label = "CTE_DML"
    elif label in {"BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE", "END"}:
        label = "TRANSACTION_CONTROL"
    elif label == "DROP":
        label = "DROP"
    elif label == "PRAGMA":
        label = "PRAGMA"
    elif label in {"INSERT", "UPDATE", "DELETE", "REPLACE", "ATTACH", "DETACH", "VACUUM", "REINDEX", "ANALYZE"}:
        pass
    raise RuntimeError(f"migration {name} is not additive: {label}")


def validate_additive_migration(name: str, sql: str) -> None:
    statements = split_sql_statements(sql)
    if not statements:
        raise RuntimeError(f"migration {name} contains no SQL statements")
    for statement in statements:
        validate_statement(name, statement)


def _rows(connection: sqlite3.Connection, sql: str, parameters: Iterable[Any] = ()) -> list[list[Any]]:
    return [list(row) for row in connection.execute(sql, tuple(parameters)).fetchall()]


def schema_snapshot(connection: sqlite3.Connection) -> dict[str, Any]:
    objects = _rows(
        connection,
        "SELECT type, name, tbl_name, COALESCE(sql, '') FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name",
    )
    tables: dict[str, Any] = {}
    for object_type, name, _table, _sql in objects:
        if object_type != "table":
            continue
        quoted = str(name).replace('"', '""')
        index_rows = _rows(connection, f'PRAGMA index_list("{quoted}")')
        index_xinfo: dict[str, list[list[Any]]] = {}
        for index_row in index_rows:
            index_name = str(index_row[1]).replace('"', '""')
            index_xinfo[str(index_row[1])] = _rows(connection, f'PRAGMA index_xinfo("{index_name}")')
        tables[str(name)] = {
            "columns": _rows(connection, f'PRAGMA table_xinfo("{quoted}")'),
            "foreign_keys": _rows(connection, f'PRAGMA foreign_key_list("{quoted}")'),
            "indexes": index_rows,
            "index_xinfo": index_xinfo,
        }
    table_options = {
        str(row[1]): {"without_rowid": int(row[4]), "strict": int(row[5])}
        for row in _rows(connection, "PRAGMA table_list")
        if row[0] == "main" and row[2] == "table" and not str(row[1]).startswith("sqlite_")
    }
    payload = {"objects": objects, "tables": tables, "table_options": table_options}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"sha256": sha256(canonical), "schema": payload}


def _contract_sha256(contract: dict[str, Any]) -> str:
    canonical = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(canonical)


def _constraint_markers(create_sql: str) -> dict[str, int]:
    return {
        "check": len(re.findall(r"\bCHECK\s*\(", create_sql, re.IGNORECASE)),
        "unique": len(re.findall(r"\bUNIQUE(?:\s+CONSTRAINT)?\b", create_sql, re.IGNORECASE)),
    }


def _table_contract(connection: sqlite3.Connection, table: str) -> dict[str, Any]:
    quoted = table.replace('"', '""')
    create_row = connection.execute(
        "SELECT COALESCE(sql, '') FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    if create_row is None:
        raise RuntimeError(f"required baseline table is missing: {table}")
    indexes: dict[str, Any] = {}
    for row in _rows(connection, f'PRAGMA index_list("{quoted}")'):
        index_name = str(row[1])
        quoted_index = index_name.replace('"', '""')
        indexes[index_name] = {
            "unique": int(row[2]),
            "origin": str(row[3]),
            "partial": int(row[4]),
            "columns": _rows(connection, f'PRAGMA index_xinfo("{quoted_index}")'),
        }
    options_row = next(
        (
            row for row in _rows(connection, "PRAGMA table_list")
            if row[0] == "main" and row[1] == table and row[2] == "table"
        ),
        None,
    )
    if options_row is None:
        raise RuntimeError(f"required baseline table metadata is missing: {table}")
    return {
        "columns": _rows(connection, f'PRAGMA table_xinfo("{quoted}")'),
        "foreign_keys": _rows(connection, f'PRAGMA foreign_key_list("{quoted}")'),
        "indexes": indexes,
        "options": {"without_rowid": int(options_row[4]), "strict": int(options_row[5])},
        "constraints": _constraint_markers(str(create_row[0])),
    }


def build_canonical_baseline_contract(name: str, sql: str) -> dict[str, Any]:
    """Execute the canonical baseline in an isolated scratch database."""
    validate_additive_migration(name, sql)
    scratch = sqlite3.connect(":memory:", isolation_level=None)
    try:
        scratch.execute("PRAGMA foreign_keys=ON")
        scratch.execute("BEGIN IMMEDIATE")
        scratch.set_authorizer(migration_authorizer)
        for statement in split_sql_statements(sql):
            scratch.execute(statement)
        scratch.set_authorizer(None)
        scratch.execute("COMMIT")
        tables = sorted(
            row[0]
            for row in scratch.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        )
        if not tables:
            raise RuntimeError(f"baseline migration {name} created no tables")
        contract = {"tables": {table: _table_contract(scratch, table) for table in tables}}
        return {"sha256": _contract_sha256(contract), "contract": contract}
    except Exception:
        scratch.set_authorizer(None)
        if scratch.in_transaction:
            scratch.execute("ROLLBACK")
        raise
    finally:
        scratch.close()


def verify_baseline_compatibility(connection: sqlite3.Connection, name: str, sql: str) -> dict[str, Any]:
    """Fail closed unless the existing schema satisfies the full baseline contract."""
    expected = build_canonical_baseline_contract(name, sql)
    errors: list[str] = []
    actual_projection: dict[str, Any] = {"tables": {}}
    for table, required in expected["contract"]["tables"].items():
        try:
            actual = _table_contract(connection, table)
        except RuntimeError as error:
            errors.append(str(error))
            continue
        projected_indexes: dict[str, Any] = {}
        for index_name, required_index in required["indexes"].items():
            actual_index = actual["indexes"].get(index_name)
            if actual_index != required_index:
                errors.append(f"{table}: required index drift: {index_name}")
            elif actual_index is not None:
                projected_indexes[index_name] = actual_index
        required_implicit = {
            index_name: value for index_name, value in required["indexes"].items() if value["origin"] != "c"
        }
        actual_implicit = {
            index_name: value for index_name, value in actual["indexes"].items() if value["origin"] != "c"
        }
        if actual_implicit != required_implicit:
            errors.append(f"{table}: primary/unique constraint index drift")
        for field in ("columns", "foreign_keys", "options", "constraints"):
            if actual[field] != required[field]:
                errors.append(f"{table}: {field} drift")
        actual_projection["tables"][table] = {
            "columns": actual["columns"],
            "foreign_keys": actual["foreign_keys"],
            "indexes": projected_indexes,
            "options": actual["options"],
            "constraints": actual["constraints"],
        }
    actual_sha256 = _contract_sha256(actual_projection)
    if errors or actual_sha256 != expected["sha256"]:
        detail = "; ".join(errors[:12]) or "required schema fingerprint mismatch"
        raise RuntimeError(f"baseline adoption rejected for {name}: {detail}")
    return {
        "migration": name,
        "required_tables": sorted(expected["contract"]["tables"]),
        "canonical_schema_sha256": expected["sha256"],
        "actual_required_schema_sha256": actual_sha256,
        "database_schema_sha256": schema_snapshot(connection)["sha256"],
    }


def verify_additive_schema(before: dict[str, Any], after: dict[str, Any], migration: str) -> None:
    before_schema = before["schema"]
    after_schema = after["schema"]
    before_objects = {(row[0], row[1]): row for row in before_schema["objects"]}
    after_objects = {(row[0], row[1]): row for row in after_schema["objects"]}
    missing = sorted(set(before_objects) - set(after_objects))
    if missing:
        raise RuntimeError(f"migration {migration} removed schema objects: {missing}")
    for key, row in before_objects.items():
        if key[0] != "table" and after_objects[key] != row:
            raise RuntimeError(f"migration {migration} modified existing schema object: {key}")
    for key in set(after_objects) - set(before_objects):
        if key[0] not in {"table", "index"}:
            raise RuntimeError(f"migration {migration} created forbidden schema object: {key}")
    for table, prior in before_schema["tables"].items():
        current = after_schema["tables"].get(table)
        if current is None:
            raise RuntimeError(f"migration {migration} removed table: {table}")
        old_columns = prior["columns"]
        if current["columns"][: len(old_columns)] != old_columns:
            raise RuntimeError(f"migration {migration} changed existing columns: {table}")
        if current["foreign_keys"][: len(prior["foreign_keys"])] != prior["foreign_keys"]:
            raise RuntimeError(f"migration {migration} changed existing foreign keys: {table}")


DENIED_AUTHOR_ACTIONS = {
    value
    for name in (
        "SQLITE_DELETE", "SQLITE_ATTACH", "SQLITE_DETACH", "SQLITE_PRAGMA",
        "SQLITE_DROP_TABLE", "SQLITE_DROP_INDEX", "SQLITE_DROP_VIEW", "SQLITE_DROP_TRIGGER",
        "SQLITE_CREATE_TRIGGER", "SQLITE_CREATE_VIEW", "SQLITE_CREATE_VTABLE", "SQLITE_DROP_VTABLE",
        "SQLITE_ANALYZE",
    )
    if (value := getattr(sqlite3, name, None)) is not None
}


def migration_authorizer(action: int, arg1: str | None, _arg2: str | None, _database: str | None, _source: str | None) -> int:
    if action in DENIED_AUTHOR_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action in {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE} and (arg1 or "") not in {"sqlite_master", "sqlite_temp_master"}:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def existing_tables(connection: sqlite3.Connection) -> set[str]:
    return {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def record_migration(connection: sqlite3.Connection, name: str, checksum: str, steps: int) -> None:
    stable_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"research-workstation:{name}:{checksum}"))
    connection.execute(
        """INSERT INTO "_prisma_migrations"
        (id, checksum, finished_at, migration_name, started_at, applied_steps_count)
        VALUES (?, ?, CURRENT_TIMESTAMP, ?, CURRENT_TIMESTAMP, ?)""",
        (stable_id, checksum, name, steps),
    )


def backup_database(database: Path, backup_root: Path, schema_hash: str) -> dict[str, Any] | None:
    if not database.is_file():
        return None
    backup_root = ensure_no_links(backup_root)
    backup_root.mkdir(parents=True, exist_ok=True)
    ensure_no_links(backup_root, allow_missing=False)
    directory = backup_root / f"migration-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex}"
    directory.mkdir()
    backup = directory / "workstation.db"
    source = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    target = sqlite3.connect(backup)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()
    digest = file_sha256(backup)
    receipt = {
        "schema": "evomind.sqlite_migration_backup.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "database": str(database),
        "backup": str(backup),
        "backup_sha256": digest,
        "schema_sha256": schema_hash,
    }
    receipt_path = directory / "backup-receipt.json"
    atomic_json(receipt_path, receipt)
    return {**receipt, "receipt": str(receipt_path), "receipt_sha256": file_sha256(receipt_path)}


def restore_database(database: Path, snapshot: dict[str, Any]) -> None:
    receipt_path = Path(snapshot["receipt"]).resolve()
    if file_sha256(receipt_path) != snapshot["receipt_sha256"]:
        raise RuntimeError("migration backup receipt hash mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "evomind.sqlite_migration_backup.v1" or Path(receipt.get("database", "")).resolve() != database:
        raise RuntimeError("migration backup receipt target mismatch")
    backup = Path(receipt["backup"]).resolve()
    if backup.parent != receipt_path.parent or file_sha256(backup) != receipt["backup_sha256"]:
        raise RuntimeError("migration backup payload hash mismatch")
    staged = database.with_name(f"{database.name}.{uuid.uuid4().hex}.restore")
    shutil.copy2(backup, staged)
    if file_sha256(staged) != receipt["backup_sha256"]:
        staged.unlink(missing_ok=True)
        raise RuntimeError("staged migration rollback hash mismatch")
    nonce = uuid.uuid4().hex
    quarantined: dict[str, Path] = {}
    committed = False
    try:
        for suffix in ("", "-wal", "-shm"):
            current = Path(f"{database}{suffix}")
            if not current.exists():
                continue
            quarantine = current.with_name(f"{current.name}.{nonce}.failed")
            current.replace(quarantine)
            quarantined[suffix] = quarantine
        staged.replace(database)
        if file_sha256(database) != receipt["backup_sha256"]:
            raise RuntimeError("restored migration database hash mismatch")
        committed = True
    except BaseException:
        database.unlink(missing_ok=True)
        for suffix, quarantine in quarantined.items():
            if quarantine.exists():
                quarantine.replace(Path(f"{database}{suffix}"))
        raise
    finally:
        staged.unlink(missing_ok=True)
        if committed:
            for quarantine in quarantined.values():
                quarantine.unlink(missing_ok=True)


def migrate(database: Path, migrations: Path, backup_root: Path | None = None) -> dict[str, object]:
    database = ensure_no_links(database)
    migrations = ensure_no_links(migrations, allow_missing=False)
    if backup_root is not None:
        backup_root = ensure_no_links(backup_root)
        if database.parent.name.casefold() == "prisma" and database.parent.parent.name.casefold() == "data":
            expected_backup = database.parent.parent.parent / "backups" / "database"
            if os.path.normcase(str(backup_root)) != os.path.normcase(str(expected_backup)):
                raise RuntimeError("migration backup root does not match the canonical EvoMind layout")
    database_existed = database.is_file()
    database.parent.mkdir(parents=True, exist_ok=True)
    migration_dirs = sorted(path for path in migrations.iterdir() if path.is_dir())
    result: dict[str, object] = {
        "database": str(database),
        "applied": [],
        "adopted": [],
        "adoption_receipts": [],
        "skipped": [],
    }
    connection = sqlite3.connect(database, timeout=30, isolation_level=None)
    snapshot: dict[str, Any] | None = None
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA journal_mode=WAL")
        pre_migration_schema = schema_snapshot(connection)
        if backup_root is not None and database_existed:
            snapshot = backup_database(database, backup_root, pre_migration_schema["sha256"])
        connection.execute(LEDGER_SQL)
        result["backup"] = snapshot
        result["schema_before_sha256"] = pre_migration_schema["sha256"]
        for directory in migration_dirs:
            sql_path = directory / "migration.sql"
            if not sql_path.is_file():
                continue
            raw = sql_path.read_bytes()
            checksum = sha256(raw)
            recorded = connection.execute(
                'SELECT checksum, finished_at, rolled_back_at FROM "_prisma_migrations" WHERE migration_name = ?',
                (directory.name,),
            ).fetchone()
            if recorded:
                if recorded[0] != checksum or recorded[2] is not None or recorded[1] is None:
                    raise RuntimeError(f"Migration ledger mismatch: {directory.name}")
                result["skipped"].append(directory.name)  # type: ignore[union-attr]
                continue

            sql = raw.decode("utf-8")
            validate_additive_migration(directory.name, sql)
            statements = split_sql_statements(sql)
            tables = existing_tables(connection)
            if "baseline" in directory.name and BASELINE_TABLES.issubset(tables):
                adoption_receipt = verify_baseline_compatibility(connection, directory.name, sql)
                connection.execute("BEGIN IMMEDIATE")
                record_migration(connection, directory.name, checksum, len(statements))
                connection.execute("COMMIT")
                result["adopted"].append(directory.name)  # type: ignore[union-attr]
                result["adoption_receipts"].append(adoption_receipt)  # type: ignore[union-attr]
                continue

            if "performance_indexes" in directory.name:
                statements = [re.sub(r"^CREATE\s+INDEX\s+", "CREATE INDEX IF NOT EXISTS ", item, count=1, flags=re.IGNORECASE) for item in statements]
            before = schema_snapshot(connection)
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.set_authorizer(migration_authorizer)
                for statement in statements:
                    connection.execute(statement)
                connection.set_authorizer(None)
                after = schema_snapshot(connection)
                verify_additive_schema(before, after, directory.name)
                record_migration(connection, directory.name, checksum, len(statements))
                connection.execute("COMMIT")
            except Exception:
                connection.set_authorizer(None)
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
            result["applied"].append(directory.name)  # type: ignore[union-attr]

        quick = connection.execute("PRAGMA quick_check").fetchone()[0]
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        if quick != "ok" or foreign_keys:
            raise RuntimeError(f"SQLite verification failed: quick={quick}, foreign_keys={len(foreign_keys)}")
        final_schema = schema_snapshot(connection)
        result.update({
            "schema_after_sha256": final_schema["sha256"],
            "quick_check": quick,
            "foreign_key_violations": 0,
            "ok": True,
        })
        return result
    except BaseException:
        connection.close()
        if snapshot:
            restore_database(database, snapshot)
        elif not database_existed:
            database.unlink(missing_ok=True)
            Path(f"{database}-wal").unlink(missing_ok=True)
            Path(f"{database}-shm").unlink(missing_ok=True)
        raise
    finally:
        try:
            connection.close()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--migrations", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = migrate(
        args.database.expanduser().resolve(),
        args.migrations.expanduser().resolve(),
        args.backup_dir.expanduser().resolve() if args.backup_dir else None,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.json:
        atomic_json(args.json.expanduser().resolve(), result)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
