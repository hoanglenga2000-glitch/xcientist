from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sqlite3

import pytest


def history_fixture(tmp_path, monkeypatch):
    transaction = module()
    root = tmp_path / 'node'
    (root / 'bundle/runtime').mkdir(parents=True)
    launcher = root / 'bundle/runtime/run_python_runtime.py'
    launcher.write_text('fixture manual startup')
    monkeypatch.setattr(transaction, 'MANUAL_LAUNCHER_SHA256', transaction.sha(launcher))
    (root / 'config').mkdir()
    (root / 'config/node-config.json').write_text(json.dumps({'hpc': {'state': 'blocked'}}))
    data = root / 'data/workspace/runtime'
    data.mkdir(parents=True)
    database = data / 'runtime.sqlite3'
    with sqlite3.connect(database) as con:
        con.executescript('''
            CREATE TABLE assistant_runs(id TEXT PRIMARY KEY, session_id TEXT, status TEXT, updated_at TEXT);
            CREATE TABLE sessions(id TEXT PRIMARY KEY, status TEXT, metadata_json TEXT);
            CREATE TABLE tool_calls(id TEXT PRIMARY KEY, session_id TEXT, status TEXT);
            CREATE TABLE approvals(id TEXT PRIMARY KEY, session_id TEXT, tool_call_id TEXT, status TEXT);
            CREATE TABLE events(id TEXT PRIMARY KEY, session_id TEXT, type TEXT);
            INSERT INTO assistant_runs VALUES('paused1','paused1','paused','old');
            INSERT INTO sessions VALUES('paused1','paused','{"user_pause_requested":true}');
            INSERT INTO assistant_runs VALUES('recovering1','recovering1','recovering','old');
            INSERT INTO sessions VALUES('recovering1','recovering','{}');
            INSERT INTO tool_calls VALUES('call1','recovering1','completed');
        ''')
    return transaction, root, database


def test_quiescent_history_requires_explicit_receipt_and_preserves_rows(tmp_path, monkeypatch):
    transaction, root, database = history_fixture(tmp_path, monkeypatch)
    before = database.read_bytes()
    with pytest.raises(RuntimeError, match='nonterminal_runs_prevent_cutover'):
        transaction.assert_idle(root)
    receipt = transaction.quiescent_history(root, recovering_ids=['recovering1'])
    transaction.assert_idle(root, preservation=receipt)
    assert receipt['preserved_runs'] == 2
    assert database.read_bytes() == before


@pytest.mark.parametrize('change', [
    "UPDATE assistant_runs SET status='running' WHERE id='paused1'",
    "UPDATE sessions SET metadata_json='{}' WHERE id='paused1'",
    "UPDATE tool_calls SET status='running' WHERE id='call1'",
    "UPDATE assistant_runs SET updated_at='changed' WHERE id='recovering1'",
    "INSERT INTO events VALUES('new-event','recovering1','changed')",
    "INSERT INTO assistant_runs VALUES('new','new','queued','now')",
    "INSERT INTO approvals VALUES('a','recovering1','call1','pending')",
])
def test_preservation_rejects_activity_or_evidence_drift(tmp_path, monkeypatch, change):
    transaction, root, database = history_fixture(tmp_path, monkeypatch)
    receipt = transaction.quiescent_history(root, recovering_ids=['recovering1'])
    with sqlite3.connect(database) as con:
        con.execute(change)
    with pytest.raises(RuntimeError):
        transaction.assert_idle(root, preservation=receipt)


def test_preservation_rejects_unreviewed_recovery_and_changed_launcher(tmp_path, monkeypatch):
    transaction, root, database = history_fixture(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match='unreviewed_recovering_run'):
        transaction.quiescent_history(root)
    receipt = transaction.quiescent_history(root, recovering_ids=['recovering1'])
    (root / 'bundle/runtime/run_python_runtime.py').write_text('automatic recovery')
    with pytest.raises(RuntimeError, match='manual_startup_contract_changed'):
        transaction.assert_idle(root, preservation=receipt)


def module():
    path = Path(os.environ['EVOMIND_TRANSACTION_TEST_SOURCE']) if os.environ.get('EVOMIND_TRANSACTION_TEST_SOURCE') else Path(__file__).resolve().parents[1] / "scripts/invitation_release_transaction.py"
    spec = importlib.util.spec_from_file_location("invitation_release_transaction", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@pytest.mark.parametrize('fail', [False, True])
def test_code_only_delivery_never_replaces_ownership_or_rolls_back_new_user_data(tmp_path, fail):
    transaction = module()
    root = tmp_path / 'node'
    runtime = root / 'bundle/runtime/evomind_runtime'
    runtime.mkdir(parents=True)
    (runtime / 'version.py').write_text('old')
    (root / 'web-overlays/old').mkdir(parents=True)
    (root / 'config').mkdir()
    (root / 'config/node-config.json').write_text(json.dumps({'web_runtime_root': str(root / 'web-overlays/old')}))
    data = root / 'data/workspace/runtime'
    data.mkdir(parents=True)
    (data / 'principal_access.sqlite3').write_bytes(b'original verified owners')
    (data / 'user_tasks.sqlite3').write_bytes(b'original user tasks')
    staged = root / 'staging/runtime'
    staged.mkdir(parents=True)
    (staged / 'version.py').write_text('new')
    web = root / 'staging/web'
    web.mkdir()
    (web / 'server.js').write_text('candidate')

    def verify(build):
        if build == 'new':
            (data / 'user_tasks.sqlite3').write_bytes(b'new user work must survive')
            if fail:
                raise RuntimeError('fixture_canary_failure')
        return {'build_id': build}

    result = transaction.switch_files(root, staged, web, None, root / 'backups/one', 'new',
        stop=lambda: None, start=lambda: None, reseal=lambda: None, verify=verify, preflight=lambda: None)
    assert (data / 'principal_access.sqlite3').read_bytes() == b'original verified owners'
    assert (data / 'user_tasks.sqlite3').read_bytes() == b'new user work must survive'
    assert result['status'] == ('failed' if fail else 'canary_activated')
    if fail:
        assert result['rollback_status'] == 'restored_and_verified'


def test_scoped_seal_preserves_unrelated_files_and_handles_runtime_replacement(tmp_path):
    transaction = module()
    root = tmp_path / 'node'
    runtime = root / 'bundle/runtime/evomind_runtime'
    runtime.mkdir(parents=True)
    (runtime / 'old.py').write_text('old runtime')
    launcher = root / 'bundle/runtime/launcher.py'
    launcher.write_text('stable launcher')
    files = [p for p in (root / 'bundle').rglob('*') if p.is_file()]
    baseline = {'schema': 'evomind.windows_bundle_integrity.v1', 'bundle_root': str(root / 'bundle'),
                'files': [{'path': p.relative_to(root / 'bundle').as_posix(), 'size': p.stat().st_size, 'sha256': transaction.sha(p)} for p in files]}
    transaction.bundle_manifest(root, baseline, allow_runtime_change=False)
    (runtime / 'old.py').rename(runtime / 'new.py')
    (runtime / 'new.py').write_text('new runtime')
    with pytest.raises(RuntimeError, match='unrelated_bundle_file_changed'):
        transaction.bundle_manifest(root, baseline, allow_runtime_change=False)
    updated = transaction.bundle_manifest(root, baseline, allow_runtime_change=True)
    rows = {row['path']: row for row in updated['files']}
    assert 'runtime/evomind_runtime/old.py' not in rows
    assert rows['runtime/evomind_runtime/new.py']['sha256'] == transaction.sha(runtime / 'new.py')
    assert launcher.read_text() == 'stable launcher'
    launcher.write_text('unexpected startup change')
    with pytest.raises(RuntimeError, match='unrelated_bundle_file_changed'):
        transaction.bundle_manifest(root, baseline, allow_runtime_change=True)


def test_scoped_seal_rejects_unlisted_unrelated_files(tmp_path):
    transaction = module()
    root = tmp_path / 'node'
    (root / 'bundle').mkdir(parents=True)
    (root / 'bundle/extra.py').write_text('unexpected')
    baseline = {'schema': 'evomind.windows_bundle_integrity.v1', 'bundle_root': str(root / 'bundle'), 'files': []}
    with pytest.raises(RuntimeError, match='unrelated_bundle_file_changed'):
        transaction.bundle_manifest(root, baseline, allow_runtime_change=True)


@pytest.mark.parametrize("fail", [False, True, "preflight", "start"])
def test_cutover_and_rollback_preserve_research_data(tmp_path, fail):
    transaction = module()
    root = tmp_path / "root"
    runtime = root / "bundle/runtime/evomind_runtime"
    runtime.mkdir(parents=True)
    (runtime / "version.py").write_text("old")
    (root / "web-overlays/old").mkdir(parents=True)
    (root / "config").mkdir()
    config = {"web_runtime_root": str(root / "web-overlays/old"), "secrets_root": "unchanged-private-store"}
    (root / "config/node-config.json").write_text(json.dumps(config))
    data = root / "data/workspace/runtime"
    data.mkdir(parents=True)
    (data / "runtime.sqlite3").write_bytes(b"historical research data")
    staged_runtime = root / "staging/runtime"
    staged_web = root / "staging/web"
    staged_runtime.mkdir(parents=True)
    staged_web.mkdir()
    (staged_runtime / "version.py").write_text("new")
    (staged_web / "server.js").write_text("candidate")
    acl = root / "staging/acl.sqlite3"
    acl.write_bytes(b"new owner indexes")
    actions = []

    def start():
        actions.append("start")
        if fail == "start" and actions.count("start") == 1:
            raise RuntimeError("fixture_receipt_failure_after_start")

    def verify(build):
        if fail and build == "new":
            raise RuntimeError("fixture_canary_failure")
        return {"build_id": build, "status": "ready"}

    def preflight():
        actions.append("preflight")
        if fail == "preflight":
            raise RuntimeError("fixture_preflight_rejection")

    if fail == "preflight":
        with pytest.raises(RuntimeError, match="fixture_preflight_rejection"):
            transaction.switch_files(root, staged_runtime, staged_web, acl, root / "backups/transaction", "new",
                                     stop=lambda: actions.append("stop"), start=lambda: actions.append("start"),
                                     reseal=lambda: actions.append("seal"), verify=verify, preflight=preflight)
        assert actions == ["preflight"]
        assert (runtime / "version.py").read_text() == "old"
        assert json.loads((root / "config/node-config.json").read_text()) == config
        assert not (root / "backups/transaction").exists()
        return

    result = transaction.switch_files(root, staged_runtime, staged_web, acl, root / "backups/transaction", "new",
                                     stop=lambda: actions.append("stop"), start=start,
                                     reseal=lambda: actions.append("seal"), verify=verify, preflight=preflight)
    assert actions[:2] == ["preflight", "stop"]
    assert (data / "runtime.sqlite3").read_bytes() == b"historical research data"
    current = json.loads((root / "config/node-config.json").read_text())
    assert current["secrets_root"] == config["secrets_root"]
    if fail:
        assert actions == ["preflight", "stop", "seal", "start", "stop", "seal", "start"]
        assert result["rollback_status"] == "restored_and_verified"
        assert (runtime / "version.py").read_text() == "old"
        assert current == config
    else:
        assert result["status"] == "canary_activated"
        assert (runtime / "version.py").read_text() == "new"
        assert Path(current["web_runtime_root"]).name == "new"


@pytest.mark.parametrize("rejected", [False, True])
def test_production_preflight_uses_real_resolver_without_writing_live_config(tmp_path, monkeypatch, rejected):
    from types import SimpleNamespace

    transaction = module()
    root = tmp_path / "root"
    (root / "config").mkdir(parents=True)
    candidate = root / "web-overlays/candidate"
    candidate.mkdir(parents=True)
    lib = root / "bundle/scripts/lib"
    lib.mkdir(parents=True)
    (lib / "Runtime.ps1").write_text("fixture-contract")
    (lib / "Common.ps1").write_text("fixture-common")
    config = root / "config/node-config.json"
    data = root / "data"
    (data / "prisma").mkdir(parents=True)
    with sqlite3.connect(data / "prisma/workstation.db") as connection:
        connection.execute("CREATE TABLE fixture (id INTEGER PRIMARY KEY)")
    database_identity = transaction.database_schema_sha256(data / "prisma/workstation.db")
    config.write_text(json.dumps({"web_runtime_root": str(root / "web-overlays/current"), "data_root": str(data)}))
    original = config.read_bytes()

    def run(argv, **kwargs):
        assert "Resolve-WebRuntimeIdentity -Config $cfg" in argv[-1]
        assert "Start-ManagedProcess" not in argv[-1]
        assert kwargs["env"]["EVOMIND_PREFLIGHT_CANDIDATE"] == str(candidate)
        value = ({"status": "failed", "error_code": "WEB_OVERLAY_RUNTIME_BUILD_MANIFEST_REJECTED"} if rejected else
                 {"status": "passed", "verified": True, "build_id": "candidate", "source_identity_sha256": "1" * 64,
                  "database_schema_sha256": database_identity, "prisma_schema_source_sha256": "2" * 64})
        return SimpleNamespace(returncode=int(rejected), stdout=json.dumps(value).encode())

    monkeypatch.setattr(transaction.subprocess, "run", run)
    receipt = tmp_path / "preflight.json"
    if rejected:
        with pytest.raises(RuntimeError, match="production_startup_preflight_rejected"):
            transaction.production_preflight(root, candidate, "candidate", "1" * 64, database_identity, receipt, schema_source_identity="2" * 64)
    else:
        assert transaction.production_preflight(root, candidate, "candidate", "1" * 64, database_identity, receipt, schema_source_identity="2" * 64)["status"] == "passed"
    assert config.read_bytes() == original
    assert json.loads(receipt.read_text())["status"] == ("failed" if rejected else "passed")
