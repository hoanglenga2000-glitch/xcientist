from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/invitation_ownership_migration.py"
    spec = importlib.util.spec_from_file_location("invitation_ownership_migration", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@pytest.fixture
def source(tmp_path):
    database = tmp_path / "runtime.sqlite3"
    binding = tmp_path / "binding.json"
    identity = {"tenant_id": "tenant_" + "a" * 24, "owner_principal_id": "owner"}
    binding.write_text(json.dumps({"schema": "evomind.tenant_hpc_binding.v1", "tenant_id": identity["tenant_id"], "username": "owner"}))
    connection = sqlite3.connect(database)
    try:
        connection.executescript("CREATE TABLE sessions(id PRIMARY KEY,metadata_json); CREATE TABLE assistant_runs(id PRIMARY KEY,session_id,conversation_id); CREATE TABLE attachments(id PRIMARY KEY,run_id,upload_id);")
        connection.execute("INSERT INTO sessions VALUES(?,?)", ("run_owned", json.dumps({"managed_hpc_identity": identity})))
        connection.execute("INSERT INTO sessions VALUES(?,?)", ("run_unknown", "{}"))
        connection.execute("INSERT INTO assistant_runs VALUES(?,?,?)", ("run_owned", "run_owned", "mixed"))
        connection.execute("INSERT INTO assistant_runs VALUES(?,?,?)", ("run_unknown", "run_unknown", "mixed"))
        connection.execute("INSERT INTO attachments VALUES(?,?,?)", ("attachment_owned", "run_owned", "upload_owned"))
        connection.commit()
    finally:
        connection.close()
    return database, binding, tmp_path / "acl.sqlite3"


def test_migration_adds_only_attributable_ownership_and_preserves_source(source):
    database, binding, acl = source
    migration = module()
    before = migration.sha(database)
    frozen = migration.plan(database, binding)
    assert frozen["matched_run_count"] == 1
    assert frozen["withheld_run_ids"] == ["run_unknown"]
    assert frozen["withheld_conversation_ids"] == ["mixed"]
    first = migration.apply(database, binding, acl, frozen)
    second = migration.apply(database, binding, acl, frozen)
    assert first == second
    assert migration.sha(database) == before
    connection = sqlite3.connect(acl)
    try:
        rows = connection.execute("SELECT kind,resource_id FROM resource_owners").fetchall()
        assert ("run", "run_owned") in rows
        assert ("run", "run_unknown") not in rows
        assert ("conversation", "mixed") not in rows
    finally:
        connection.close()


def test_changed_source_rejects_frozen_plan(source):
    database, binding, acl = source
    migration = module()
    frozen = migration.plan(database, binding)
    connection = sqlite3.connect(database)
    try:
        connection.execute("UPDATE assistant_runs SET conversation_id='changed' WHERE id='run_owned'")
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(ValueError, match="migration_source_or_plan_changed"):
        migration.apply(database, binding, acl, frozen)


def test_owner_conflict_rolls_back_without_reassignment(source):
    database, binding, acl = source
    migration = module()
    frozen = migration.plan(database, binding)
    migration.apply(database, binding, acl, frozen)
    connection = sqlite3.connect(acl)
    try:
        connection.execute("UPDATE resource_owners SET owner_id='another' WHERE kind='run'")
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(ValueError, match="migration_existing_owner_conflict"):
        migration.apply(database, binding, acl, frozen)
