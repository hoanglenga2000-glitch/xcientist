from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from evomind_runtime import managed_cancel as cancel
from evomind_runtime.hpc_runtime_overlay import HpcRuntime


def receipt():
    return {"schema": cancel.SCHEMA, "solution_root": "/allowed/solution", "status": "cancelled", "exit_observed": True,
            "signals_sent": 1, "other_processes_modified": False, "identity": {"pid": 100, "identity_sha256": "a" * 64}}


@pytest.mark.parametrize("field,value", [("status", "unconfirmed"), ("exit_observed", False), ("signals_sent", 0),
    ("signals_sent", True), ("other_processes_modified", True), ("solution_root", "/another/solution"), ("identity", {})])
def test_cancel_never_accepts_unproven_success(field, value):
    data = receipt()
    data[field] = value
    with pytest.raises(ValueError, match="managed_cancel_exit_unconfirmed"):
        cancel.confirmed_receipt(json.dumps(data), "/allowed/solution")


def test_cancel_requires_exit_and_identity_receipt():
    assert cancel.confirmed_receipt(json.dumps(receipt()), "/allowed/solution")["exit_observed"] is True
    with pytest.raises(ValueError, match="managed_cancel_exit_unconfirmed"):
        cancel.confirmed_receipt("", "/allowed/solution")


def test_overlay_zero_exit_is_not_cancellation_proof(tmp_path, monkeypatch):
    client = SimpleNamespace(closed=False)
    client.close = lambda: setattr(client, "closed", True)
    runtime = HpcRuntime(run_id="run_cancel_fixture", local_run_dir=tmp_path, connector=lambda: client)
    commands = []

    def execute(_client, command, *, timeout):
        commands.append(command)
        return 0, "", ""

    monkeypatch.setattr(runtime, "_exec", execute)
    with pytest.raises(ValueError, match="managed_cancel_exit_unconfirmed"):
        runtime.cancel(solution_id="fixture")
    assert client.closed
    assert "pkill" not in commands[0] and "|| true" not in commands[0]


@pytest.mark.parametrize("fault", ["none", "expired", "missing_proxy", "wrong_job"])
def test_cancel_requires_fresh_complete_profile_bound_gate(tmp_path, fault):
    flags = {key: True for key in cancel.IDENTITY_FLAGS}
    samples = [{**flags, "job_id": 93207} for _ in range(5)]
    evidence = {**flags, "schema": "evomind.hpc.identity_receipt.v2", "job_id": 93207,
                "samples_requested": 5, "samples_passed": 5, "samples": samples}
    if fault == "missing_proxy":
        samples[4].pop("designated_proxy_path_verified")
    if fault == "wrong_job":
        evidence["job_id"] = 1
    timestamp = datetime.now(timezone.utc) - timedelta(seconds=180 if fault == "expired" else 1)
    connection = sqlite3.connect(tmp_path / "runtime.sqlite3")
    try:
        connection.execute("CREATE TABLE tool_calls(session_id,tool_name,status,completed_at,result_json)")
        connection.execute("INSERT INTO tool_calls VALUES(?,?,?,?,?)", ("run_fixture", "hpc_verify", "completed", timestamp.isoformat(), json.dumps({"ok": True, "content": evidence})))
        connection.commit()
    finally:
        connection.close()
    if fault == "none":
        cancel.require_fresh_identity(tmp_path, "run_fixture", 93207)
    else:
        with pytest.raises(ValueError, match="fresh_complete_hpc_identity_required_for_cancel"):
            cancel.require_fresh_identity(tmp_path, "run_fixture", 93207)


@pytest.mark.parametrize("exits", [True, False])
def test_pidfd_cancellation_requires_exit_observation(tmp_path, monkeypatch, exits):
    solution = tmp_path / "solutions" / "fixture"
    solution.mkdir(parents=True)
    proc = tmp_path / "proc"
    (proc / "100").mkdir(parents=True)
    monkeypatch.setattr(cancel, "ALLOWED_ROOT", tmp_path)
    monkeypatch.setattr(cancel, "PROC", proc)
    identity = {"pid": 100, "parent_pid": 1, "start_ticks": "123", "identity_sha256": "b" * 64}
    monkeypatch.setattr(cancel, "process_identity", lambda *_: identity)
    monkeypatch.setattr(cancel, "_has_children", lambda _: False)
    monkeypatch.setattr(cancel.os, "pidfd_open", lambda *_: 901, raising=False)
    sent, closed = [], []
    monkeypatch.setattr(cancel.signal, "pidfd_send_signal", lambda fd, sig: sent.append((fd, sig)), raising=False)
    monkeypatch.setattr(cancel.os, "close", lambda fd: closed.append(fd))
    monkeypatch.setattr(cancel.select, "select", lambda *_: ([901] if exits else [], [], []))
    result = cancel.cancel_exact(solution, send_signal=True)
    assert result["exit_observed"] is exits
    assert (result["status"] == "cancelled") is exits
    assert len(sent) == 1 and closed == [901]


def test_directory_alias_is_rejected_before_process_inspection(tmp_path, monkeypatch):
    solution = tmp_path / "solutions" / "requested"
    other = tmp_path / "solutions" / "another_run"
    solution.mkdir(parents=True)
    other.mkdir()
    original_resolve = Path.resolve
    monkeypatch.setattr(cancel, "ALLOWED_ROOT", tmp_path)
    monkeypatch.setattr(Path, "resolve", lambda path, strict=False: other if path == solution else original_resolve(path, strict=strict))
    monkeypatch.setattr(cancel, "process_identity", lambda *_: pytest.fail("must reject before examining any process"))
    with pytest.raises(ValueError, match="cancel_solution_alias_rejected"):
        cancel.cancel_exact(solution, send_signal=True)


def test_directory_replacement_after_pidfd_open_sends_no_signal(tmp_path, monkeypatch):
    solution = tmp_path / "solutions" / "fixture"
    solution.mkdir(parents=True)
    proc = tmp_path / "proc"
    (proc / "100").mkdir(parents=True)
    monkeypatch.setattr(cancel, "ALLOWED_ROOT", tmp_path)
    monkeypatch.setattr(cancel, "PROC", proc)
    identity = {"pid": 100, "parent_pid": 1, "start_ticks": "123", "identity_sha256": "b" * 64}
    monkeypatch.setattr(cancel, "process_identity", lambda *_: identity)
    monkeypatch.setattr(cancel, "_has_children", lambda _: False)
    replaced = False

    def opened(*_):
        nonlocal replaced
        replaced = True
        return 901

    monkeypatch.setattr(cancel, "directory_identity", lambda _: (1, 2 if replaced else 1))
    monkeypatch.setattr(cancel.os, "pidfd_open", opened, raising=False)
    monkeypatch.setattr(cancel.signal, "pidfd_send_signal", lambda *_: pytest.fail("directory changed before signal"), raising=False)
    closed = []
    monkeypatch.setattr(cancel.os, "close", closed.append)
    result = cancel.cancel_exact(solution, send_signal=True)
    assert result["signals_sent"] == 0 and result["exit_observed"] is False
    assert closed == [901]
