from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from evomind_runtime import dependency_lock
from evomind_runtime.hpc_runtime_overlay import HpcRuntime
from evomind_runtime.runtime import AgentRuntime
from evomind_runtime.tenant_access import AccessError, AccessStore, Principal


PINS = {"xgboost": "2.1.3"}
TARGET = "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/runtime/fixture/site-packages"


def good_receipt():
    return {
        "schema": "evomind.dependency_install.v1", "status": "ready", "target": TARGET,
        "requested_pins": PINS, "hashes_before_install": True, "reused": False,
        "lock_sha256": "b" * 64,
    }


@pytest.mark.parametrize("payload", [
    "training complete", "[]", "null", "{}", "x" * (64 * 1024 + 1),
    '{"status":"ready","status":"failed"}',
    "log output\n" + json.dumps(good_receipt()),
    json.dumps(good_receipt()) + json.dumps(good_receipt()),
], ids=["plain-text", "array", "null", "empty-object", "oversized", "duplicate-key", "prefixed-log", "multiple-documents"])
def test_dependency_receipt_rejects_non_contract_output(payload):
    with pytest.raises(ValueError, match="^managed_dependency_receipt_invalid$"):
        dependency_lock.validate_install_receipt(payload, TARGET, PINS)


@pytest.mark.parametrize(("field", "value"), [
    ("schema", "legacy"), ("status", "failed"), ("target", "/tmp/other"),
    ("requested_pins", {"xgboost": "1.0.0"}), ("hashes_before_install", False),
    ("reused", "true"), ("lock_sha256", "B" * 64), ("lock_sha256", 123),
])
def test_dependency_receipt_is_bound_to_exact_installation(field, value):
    receipt = good_receipt()
    receipt[field] = value
    with pytest.raises(ValueError, match="^managed_dependency_receipt_invalid$"):
        dependency_lock.validate_install_receipt(json.dumps(receipt), TARGET, PINS)


def test_hash_first_install_produces_valid_receipt_and_verified_reuse(tmp_path, monkeypatch):
    wheel = b"local-fixture-wheel-not-a-real-package"
    digest = hashlib.sha256(wheel).hexdigest()
    wheel_name = "xgboost-2.1.3-py3-none-any.whl"
    calls = []

    def pip_fixture(argv, **kwargs):
        calls.append(argv)
        assert kwargs["env"]["PYTHONNOUSERSITE"] == "1"
        if "--dry-run" in argv:
            report = {"install": [{"metadata": {"name": "xgboost", "version": "2.1.3"}, "download_info": {"url": "https://files.pythonhosted.org/packages/" + wheel_name, "archive_info": {"hashes": {"sha256": digest}}}}]}
            Path(argv[argv.index("--report") + 1]).write_text(json.dumps(report))
        elif "download" in argv:
            assert "--require-hashes" in argv
            Path(argv[argv.index("--dest") + 1], wheel_name).write_bytes(wheel)
        else:
            assert "--no-index" in argv and "--require-hashes" in argv
            target = Path(argv[argv.index("--target") + 1])
            target.mkdir(parents=True)
            (target / "fixture_module.py").write_text("value = 1\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(dependency_lock.subprocess, "run", pip_fixture)
    first = dependency_lock.prepare_target(tmp_path, PINS)
    validated = dependency_lock.validate_install_receipt(json.dumps(first), str(tmp_path / "site-packages"), PINS)
    assert validated["reused"] is False
    assert len(calls) == 3
    assert validated["lock_sha256"] == hashlib.sha256((tmp_path / "dependency-lock.json").read_bytes()).hexdigest()
    assert dependency_lock.prepare_target(tmp_path, PINS)["reused"] is True
    assert len(calls) == 3
    (tmp_path / "site-packages" / "fixture_module.py").write_text("tampered")
    with pytest.raises(ValueError, match="installed_dependency_integrity_mismatch"):
        dependency_lock.prepare_target(tmp_path, PINS)


def test_invalid_dependency_receipt_preserves_target_and_closes_transport(tmp_path, monkeypatch):
    script = tmp_path / "train.py"
    script.write_text("import xgboost\n")
    client = SimpleNamespace(closed=False)
    client.close = lambda: setattr(client, "closed", True)
    runtime = HpcRuntime(run_id="run_receipt_rejected", local_run_dir=tmp_path / "hpc", connector=lambda: client)
    original = runtime.remote_solution_deps
    monkeypatch.setattr(runtime, "_exec", lambda *_args, **_kwargs: (0, "training complete", ""))
    with pytest.raises(ValueError, match="managed_dependency_receipt_invalid"):
        runtime.prepare_solution_environment(script)
    assert runtime.remote_solution_deps == original
    assert client.closed is True


def test_runtime_close_waits_for_worker_and_does_not_replay_deferred_resume(tmp_path, monkeypatch):
    runtime = AgentRuntime(tmp_path)
    entered, release = threading.Event(), threading.Event()
    executions = []

    def execute(run_id, *, resume):
        executions.append(run_id)
        entered.set()
        assert release.wait(timeout=5)
        runtime.store.append_event(run_id, "fixture_shutdown_settled", {})

    monkeypatch.setattr(runtime.assistant, "_execute", execute)
    run = runtime.assistant.create_run(prompt="Local lifecycle fixture", start=False)
    try:
        assert runtime.assistant.start(run["id"])
        assert entered.wait(timeout=2)
        runtime.assistant._start_or_defer(run["id"], resume=True)
        assert runtime.close(timeout=0) is False
        assert runtime.store.get_session(run["id"]) is not None
        assert runtime.assistant.start(run["id"], resume=True) is False
        assert runtime.assistant._start_or_defer(run["id"], resume=True) is False
        release.set()
        assert runtime.close(timeout=2) is True
        assert runtime.close(timeout=0) is True
        assert executions == [run["id"]]
    finally:
        release.set()
        runtime.close(timeout=2)


def test_acl_closes_connections_on_success_and_rollback(tmp_path, monkeypatch):
    import evomind_runtime.tenant_access as access_module

    connections = []
    original_connect = sqlite3.connect

    def connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(access_module.sqlite3, "connect", connect)
    store = AccessStore(tmp_path)
    alice = Principal("tenant_" + "a" * 24, "alice")
    bob = Principal("tenant_" + "b" * 24, "bob")
    store.bind("upload", "upload_fixture", alice)
    with pytest.raises(AccessError):
        store.bind("upload", "upload_fixture", bob)
    assert store.owns("upload", "upload_fixture", alice)
    assert not store.owns("upload", "upload_fixture", bob)
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            connection.execute("SELECT 1")
