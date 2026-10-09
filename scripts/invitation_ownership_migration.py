"""Additive ownership migration; never rewrites research records or metrics."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import uuid


SCHEMA = "evomind.invitation_ownership_migration.v1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@contextmanager
def read_database(path: Path):
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("BEGIN")
        yield connection
    finally:
        connection.close()


def logical_digest(connection: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    for line in connection.iterdump():
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def plan(database: Path, binding_path: Path) -> dict:
    binding = json.loads(binding_path.read_text(encoding="utf-8-sig"))
    tenant, owner = binding.get("tenant_id"), binding.get("username")
    if (binding.get("schema") != "evomind.tenant_hpc_binding.v1"
            or not isinstance(tenant, str) or not re.fullmatch(r"tenant_[a-f0-9]{24}", tenant)
            or not isinstance(owner, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", owner)):
        raise ValueError("migration_binding_identity_invalid")
    rows = set()
    matched = set()
    withheld = []
    conversations = {}
    upload_owners = {}
    with read_database(database) as connection:
        for record in connection.execute("SELECT r.id,r.session_id,r.conversation_id,s.metadata_json FROM assistant_runs r LEFT JOIN sessions s ON s.id=r.session_id"):
            metadata = json.loads(record["metadata_json"]) if record["metadata_json"] else {}
            identity = metadata.get("managed_hpc_identity") or {}
            owned = identity.get("tenant_id") == tenant and identity.get("owner_principal_id") == owner
            conversations.setdefault(record["conversation_id"], set()).add(owned)
            if owned:
                matched.add(record["id"])
                rows.add(("session", record["session_id"]))
                rows.add(("run", record["id"]))
            else:
                withheld.append(record["id"])
        for conversation, owners in conversations.items():
            if owners == {True}:
                rows.add(("conversation", conversation))
        for attachment in connection.execute("SELECT id,run_id,upload_id FROM attachments"):
            owned = attachment["run_id"] in matched
            if owned:
                rows.add(("attachment", attachment["id"]))
            if attachment["upload_id"]:
                upload_owners.setdefault(attachment["upload_id"], set()).add(owned)
        for upload_id, owners in upload_owners.items():
            if owners == {True}:
                rows.add(("upload", upload_id))
        source_digest = logical_digest(connection)
    result = {
        "schema": SCHEMA, "owner_source": "current_binding.username_and_tenant_id",
        "principal": {"tenant_id": tenant, "owner_id": owner},
        "binding_sha256": sha(binding_path), "source_logical_sha256": source_digest,
        "matched_run_count": len(matched), "withheld_run_ids": sorted(withheld),
        "withheld_conversation_ids": sorted(name for name, owners in conversations.items() if owners != {True}),
        "ownership_rows": [{"kind": kind, "resource_id": resource_id} for kind, resource_id in sorted(rows)],
        "research_records_modified": False, "historical_results_regraded": False,
    }
    result["plan_content_sha256"] = hashlib.sha256(canonical(result).encode()).hexdigest()
    return result


def apply(database: Path, binding: Path, acl: Path, frozen: dict) -> dict:
    if frozen != plan(database, binding):
        raise ValueError("migration_source_or_plan_changed")
    if acl.is_symlink() or acl.resolve(strict=False) != acl.absolute():
        raise ValueError("migration_acl_alias_rejected")
    acl.parent.mkdir(parents=True, exist_ok=True)
    principal = frozen["principal"]
    connection = sqlite3.connect(acl, timeout=5)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS resource_owners (
                kind TEXT NOT NULL,resource_id TEXT NOT NULL,tenant_id TEXT NOT NULL,owner_id TEXT NOT NULL,
                PRIMARY KEY(kind,resource_id)
            );
            CREATE INDEX IF NOT EXISTS resource_owners_principal ON resource_owners(tenant_id,owner_id,kind);
            CREATE TABLE IF NOT EXISTS ownership_migrations (
                plan_sha256 TEXT PRIMARY KEY, source_sha256 TEXT NOT NULL, row_count INTEGER NOT NULL
            );
        """)
        connection.execute("BEGIN IMMEDIATE")
        for row in frozen["ownership_rows"]:
            values = (row["kind"], row["resource_id"], principal["tenant_id"], principal["owner_id"])
            connection.execute("INSERT OR IGNORE INTO resource_owners VALUES(?,?,?,?)", values)
            existing = connection.execute("SELECT tenant_id,owner_id FROM resource_owners WHERE kind=? AND resource_id=?", values[:2]).fetchone()
            if existing != values[2:]:
                raise ValueError("migration_existing_owner_conflict")
        connection.execute("INSERT OR IGNORE INTO ownership_migrations VALUES(?,?,?)", (frozen["plan_content_sha256"], frozen["source_logical_sha256"], len(frozen["ownership_rows"])))
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    with read_database(database) as source:
        if logical_digest(source) != frozen["source_logical_sha256"]:
            raise ValueError("migration_runtime_data_changed")
    return {"schema": SCHEMA, "status": "applied", "plan_sha256": frozen["plan_content_sha256"],
            "ownership_rows": len(frozen["ownership_rows"]), "matched_runs": frozen["matched_run_count"],
            "withheld_runs": len(frozen["withheld_run_ids"]), "withheld_conversations": len(frozen["withheld_conversation_ids"]),
            "source_logical_sha256": frozen["source_logical_sha256"], "research_records_modified": False}


def write_json(path: Path, value: dict):
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("plan", "apply"))
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--acl", type=Path)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.mode == "plan":
        frozen = plan(args.database, args.binding)
        if args.plan.exists():
            raise ValueError("migration_plan_already_exists")
        write_json(args.plan, frozen)
        result = {"schema": SCHEMA, "status": "planned", "plan_file_sha256": sha(args.plan),
                  "matched_runs": frozen["matched_run_count"], "withheld_runs": len(frozen["withheld_run_ids"]),
                  "ownership_rows": len(frozen["ownership_rows"])}
    else:
        if args.acl is None or not args.expected_plan_sha256 or sha(args.plan) != args.expected_plan_sha256:
            raise ValueError("migration_plan_hash_required")
        result = apply(args.database, args.binding, args.acl, json.loads(args.plan.read_text()))
    if args.result:
        write_json(args.result, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
