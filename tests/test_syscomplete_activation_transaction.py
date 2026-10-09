"""Disposable fixture activation: no service, launcher, model or HPC execution."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import zipfile

import pytest

from scripts import syscomplete_activation_transaction as tx
from evomind_runtime.models import ToolCall, ToolResult
from evomind_runtime.runtime import AgentRuntime
from invitation_offline_plugin import invitation_offline_boundary  # noqa: F401


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tx.canonical(value) + "\n", encoding="utf-8")
    return {"path": str(path), "sha256": tx.sha(path)}


def package(path, payloads, name, fields):
    rows = [{"path": key, "bytes": len(value), "sha256": tx.digest_bytes(value)} for key, value in sorted(payloads.items())]
    manifest = {**fields, "files": rows, "file_count": len(rows)}
    with zipfile.ZipFile(path, "w") as archive:
        for key, value in payloads.items():
            archive.writestr(key, value)
        archive.writestr(name, tx.canonical(manifest).encode())
    return {"path": str(path), "sha256": tx.sha(path), "bytes": path.stat().st_size}, manifest


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "node"
    stage = root / "staging/release-fixture"
    stage.mkdir(parents=True)
    runtime = AgentRuntime(root / "data", runtime_root=root / "data/workspace/runtime")
    run = runtime.assistant.create_run(prompt="Historical fixture; preserve model and evidence", start=False, _reserved_run_id=tx.PAUSE_RUN)
    runtime.store.update_assistant_run(run["id"], status="recovering", model="gpt-6-astra", model_provider="openai")
    for index in range(22):
        call = ToolCall(f"call_fixture_{index}", run["id"], "file_read", {"path": f"fixture-{index}.txt"}, status="completed")
        runtime.store.put_tool_call(call, ToolResult(call.id, True, {"fixture": index}, "Historical synthetic receipt"))
    runtime.store.add_turn(run["id"], "assistant", "Historical result must remain intact")
    runtime.close()
    data = root / "data/workspace/runtime"
    with sqlite3.connect(data / "gpu_budget.sqlite3") as connection:
        connection.execute("CREATE TABLE gpu_operations(id TEXT PRIMARY KEY,study TEXT,run_id TEXT,kind TEXT,reserved REAL,charged REAL,status TEXT,boot_id TEXT,controller_pid INTEGER,created REAL)")
        connection.execute("INSERT INTO gpu_operations VALUES('old','study',?,'engineering',120,100,'passed','old-boot',123,1)", (tx.PAUSE_RUN,))
    code = root / tx.RUNTIME_REL
    code.mkdir(parents=True)
    (code / "version.py").write_bytes(b"OLD_CODE\n")
    old_web = root / "web-overlays/overlay-invitation-beta-aaaaaaaaaaaa-sys1"
    old_web.mkdir(parents=True)
    (old_web / "server.js").write_bytes(b"old-web")
    launcher_path = root / tx.LAUNCHER_REL
    launcher_path.parent.mkdir(parents=True)
    launcher_path.write_bytes(b"old-launcher")
    config = {"web_runtime_root": str(old_web), "data_root": str(root / "data"),
              "unchanged_private_field": "synthetic-value", "hpc": {"state": "blocked"}}
    put(root / tx.CONFIG_REL, config)
    protected = root / "config/research-control/policy.json"
    put(protected, {"study_id": "existing", "gpu_hours": 12, "engineering_gpu_hours": 3})
    old_seal = root / "bundle/integrity.json"
    put(old_seal, {"identity": "old"})
    new_seal = stage / "new-integrity.json"
    put(new_seal, {"identity": "candidate"})
    build = "overlay-invitation-beta-bbbbbbbbbbbb-sys1"
    runtime_zip, runtime_manifest = package(stage / "runtime.zip", {"evomind_runtime/version.py": b"NEW_CODE\n"},
        "runtime-hotfix-manifest.json", {"release": build})
    web_zip, _ = package(stage / "web.zip", {"server.js": b"new-web"}, "operational-overlay-manifest.json", {"overlay_id": build})
    acceptance_zip, _ = package(stage / "acceptance.zip", {"fixture.json": b"{}"}, "acceptance-manifest.json", {"build_id": build})
    launcher_candidate = stage / "launcher/candidate/Start-Node.ps1"
    launcher_candidate.parent.mkdir(parents=True)
    launcher_candidate.write_bytes(b"synthetic-bound-gpt55-report-launcher")
    profile = {"model": "gpt-5.5", "provider_strict": True, "protocol": "responses_stream_v1",
               "endpoint": "https://api.pezayo.com/v1", "reasoning_effort": "low", "service_tier": "omit"}
    launcher = {"schema": "evomind.launcher_candidate.v1", "production_changed": False,
                "candidate": {"path": "candidate/Start-Node.ps1", "sha256": tx.sha(launcher_candidate)}, "profile": profile}
    launcher_ref = put(stage / "launcher/candidate-manifest.json", launcher)
    delivery = {"schema": "evomind.syscomplete_delivery.v1", "status": "three_packages_built_not_deployed",
        "production_deployed": False, "build_id": build, "runtime": runtime_zip, "web": web_zip, "acceptance": acceptance_zip,
        "external_launcher_binding": {"manifest_sha256": launcher_ref["sha256"], "candidate_sha256": tx.sha(launcher_candidate)}}
    delivery_ref = put(stage / "delivery.json", delivery)
    isolation_rows = []
    for name, checks in tx.REQUIRED_CHECKS.items():
        values = [{"name": value, "passed": True} for value in sorted(checks)]
        if name == "acceptance-result.json":
            next(value for value in values if value["name"] == "candidate_identity").update(
                web_sha256=web_zip["sha256"], runtime_sha256=runtime_zip["sha256"])
        receipt = {"status": "passed", "build_id": build, "checks": values,
                   "web_sha256": web_zip["sha256"], "runtime_sha256": runtime_zip["sha256"]}
        reference = put(stage / "isolation" / name, receipt)
        isolation_rows.append({"path": name, "bytes": Path(reference["path"]).stat().st_size, "sha256": reference["sha256"]})
    isolation = {"schema": "evomind.syscomplete_server_isolation_receipts.v1", "status": "passed", "build_id": build,
        "production_unchanged": True, "frozen_harnesses_modified": False, "total_checks": 64, "files": isolation_rows}
    isolation_ref = put(stage / "isolation/manifest.json", isolation)
    candidate = {"files": [*runtime_manifest["files"], {"path": "run_model_endurance_acceptance.py", "sha256": "e" * 64}]}
    candidate_ref = put(stage / "endurance-candidate.json", candidate)
    cases = [{"run_id": f"run_{index:032x}", "index": index, "status": "completed", "passed": True,
              "case_timeout": False, "output_sha256": "a" * 64, "tool_evidence": {"successful_tool_count": 3,
                "required_tool_steps_passed": True, "hash_check_passed": True, "readback_passed": True}}
             for index in range(1, 21)]
    endurance = {"schema": "evomind.model_service_endurance.v1", "model": "gpt-5.5", "status": "completed",
        "service_soak_passed": True, "runtime_closed_cleanly": True, "preflight_only": False, "injected_faults": 0,
        "gpu_actions": 0, "candidate_manifest_sha256": candidate_ref["sha256"], "elapsed_seconds": 5400,
        "required_seconds": 5400, "native_tools_executed": 60, "cases": cases, "harness_sha256": "e" * 64,
        "wire_protocol": "responses", "reasoning_effort": "low", "service_tier": "omit",
        "provider_target": "api.pezayo.com", "route": "configured_upstream", "real_transport_failures": 0}
    endurance_ref = put(stage / "endurance.json", endurance)
    deps_source = stage / "report-staged/site-packages"
    deps_source.mkdir(parents=True)
    (deps_source / "synthetic_dep.py").write_bytes(b"SYNTHETIC_ONLY=1\n")
    installed = tx.tree(deps_source)
    installed_ref = put(stage / "installed-files.json", installed)
    deps_target = root / "report-envs/gpt55-fixture/site-packages"
    dep_common = {"schema": "evomind.report_dependency_delivery_acceptance.v1", "status": "staged_not_activated",
        "stage_progress": "complete", "base_venv_unchanged": True, "base_tree_sha256_before": "a" * 64,
        "base_tree_sha256_after": "a" * 64, "production_changed": False, "activated": False,
        "installed_manifest_sha256": installed_ref["sha256"], "target": str(deps_source),
        "descriptor_sha256": "d" * 64, "inheritance_sha256": "f" * 64, "acceptance_manifest_sha256": "c" * 64}
    install_ref = put(stage / "install-receipt.json", {**dep_common, "phase": "install"})
    render_ref = put(stage / "render-receipt.json", {**dep_common, "phase": "render", "report": {
        "model_requests": 0, "hpc_actions": 0, "network_attempts": 0, "pdf_pages": 2, "editable_word_tables": 2}})
    dependency_binding = {"schema": "evomind.syscomplete_launcher_dependency_binding.v1", "status": "passed",
        "launcher_manifest_sha256": launcher_ref["sha256"], "runtime_sha256": runtime_zip["sha256"],
        "dependency_tree_sha256": installed["tree_sha256"], "target": str(deps_target), "config_key": "report_dependency_root",
        "consumer_roles": ["python_runtime", "web"], "path_precedence_verified": True, "profile_unchanged": True}
    binding_ref = put(stage / "dependency-launch-binding.json", dependency_binding)
    before = tx.execution_snapshot(root)
    workers = {"schema": "evomind.syscomplete_quiescence_inventory.v1", "run_id": tx.PAUSE_RUN,
        "tool_receipts_sha256": before["tool_receipts_sha256"], "budget_sha256": before["budget_sha256"],
        "all_started_actions_accounted": True, "unknown_workers": [], "active_external_workers": [],
        "calls": [{"call_id": value, "state": "not_external"} for value in before["tool_ids"]]}
    spec = {"schema": tx.SCHEMA, "intent": "canary_validation_pending", "root": str(root), "stage": str(stage),
        "transaction_id": "fixture_transaction", "pause": {"run_id": tx.PAUSE_RUN, "authorization_sha256": "a" * 64},
        "delivery": delivery_ref, "launcher": launcher_ref, "isolation": isolation_ref, "endurance": endurance_ref,
        "endurance_candidate": candidate_ref, "worker_inventory": put(stage / "workers.json", workers),
        "dependencies": {"source": str(deps_source), "target": str(deps_target), "installed_manifest": installed_ref,
                         "install_receipt": install_ref, "render_receipt": render_ref, "launcher_binding": binding_ref},
        "baseline": {"runtime_tree_sha256": tx.tree(code)["tree_sha256"], "web_tree_sha256": tx.tree(old_web)["tree_sha256"],
            "config_sha256": tx.sha(root / tx.CONFIG_REL), "launcher_sha256": tx.sha(launcher_path),
            "protected_files": [{"path": protected.relative_to(root).as_posix(), "sha256": tx.sha(protected)}]},
        "seals": [{"target": "bundle/integrity.json", "before_sha256": tx.sha(old_seal),
                   "source": str(new_seal), "after_sha256": tx.sha(new_seal)}]}
    return spec


class Operations:
    def __init__(self, spec, fail=None):
        self.root, self.spec, self.fail = Path(spec["root"]), spec, fail
        self.actions, self.locked, self.stopped, self.starts = [], False, False, 0

    @contextmanager
    def mutex(self):
        assert not self.locked
        self.locked = True
        self.actions.append("mutex_enter")
        try:
            yield
        finally:
            self.actions.append("mutex_exit")
            self.locked = False

    def protect_backup(self, path):
        assert self.locked and path.is_dir()
        self.actions.append("private_backup")

    def checkpoint(self, phase):
        assert self.locked
        self.actions.append(phase)
        if phase == self.fail:
            raise RuntimeError("fixture_phase_failure")

    def preflight(self, plan):
        assert self.locked and Path(plan["web_target"]).is_dir()

    def stop(self):
        assert self.locked
        self.actions.append("stop")
        if self.fail == "stop" or self.fail == "rollback_stop" and self.starts:
            raise RuntimeError("fixture_stop_unconfirmed")
        self.stopped = True
        return {"stopped": True, "identity_verified": True, "managed_processes_remaining": []}

    def assert_stopped(self):
        assert self.stopped
        if self.fail == "stop_observation":
            raise RuntimeError("fixture_unknown_process")

    def verify_integrity(self, build):
        assert self.stopped and self.locked
        self.actions.append("verify_integrity")

    def start(self):
        assert self.locked and self.stopped
        self.actions.append("start")
        self.starts += 1
        self.stopped = False
        if self.fail == "partial_start":
            raise RuntimeError("fixture_partial_start")

    def verify_ready(self, build):
        if self.fail in {"ready", "rollback_stop", "configuration_drift"}:
            # A new research record after startup must survive code rollback.
            with sqlite3.connect(self.root / "data/workspace/runtime/runtime.sqlite3") as connection:
                connection.execute("CREATE TABLE IF NOT EXISTS new_research(id TEXT PRIMARY KEY,value TEXT)")
                connection.execute("INSERT INTO new_research VALUES('new_run','must survive rollback')")
            if self.fail == "configuration_drift":
                config_path = self.root / tx.CONFIG_REL
                value = json.loads(config_path.read_text())
                value["hpc"]["state"] = "frozen_by_startup"
                config_path.write_text(tx.canonical(value))
            raise RuntimeError("fixture_canary_failure")
        return {"status": "ready", "build_id": build}


def mutate_reference(reference, mutate):
    path = Path(reference["path"])
    value = json.loads(path.read_text())
    mutate(value)
    reference["sha256"] = put(path, value)["sha256"]


def test_read_only_plan_does_not_stop_write_pause_or_install(prepared):
    spec = prepared
    before = tx.execution_snapshot(spec["root"])
    database = Path(spec["root"]) / "data/workspace/runtime/runtime.sqlite3"
    before_bytes = database.read_bytes()
    plan = tx.validate(spec)
    assert plan["status"] == "planned_not_activated" and plan["release_verdict"] == "HOLD"
    assert tx.execution_snapshot(spec["root"]) == before and database.read_bytes() == before_bytes
    assert not Path(plan["backup"]).exists() and not Path(spec["dependencies"]["target"]).exists()


def test_activation_is_only_canary_pending_and_preserves_model_budget_history(prepared):
    spec = prepared
    before = tx.execution_snapshot(spec["root"])
    ops = Operations(spec)
    plan = tx.validate(spec)
    result = tx.activate(spec, plan, ops)
    assert result["status"] == "canary_validation_pending" and result["release_verdict"] == "HOLD"
    assert result["research_database_restored"] is False and result["ownership_migration"] is False
    after = tx.execution_snapshot(spec["root"])
    assert after["status"] == "paused"
    for field in ("research_identity_sha256", "tool_receipts_sha256", "budget_sha256", "charged_seconds"):
        assert after[field] == before[field]
    assert ops.actions.index("service_stopped") < ops.actions.index("sqlite_backed_up") < ops.actions.index("competition_paused") < ops.actions.index("runtime_switched")
    backup = Path(plan["backup"])
    assert (backup / "sqlite/workspace/runtime/report-jobs.sqlite3").exists()
    with tx.read_only(backup / "sqlite/workspace/runtime/runtime.sqlite3") as connection:
        assert connection.execute("SELECT status,model FROM assistant_runs WHERE id=?", (tx.PAUSE_RUN,)).fetchone()[:] == ("recovering", "gpt-6-astra")
    with tx.read_only(Path(spec["root"]) / "data/workspace/runtime/runtime.sqlite3") as connection:
        metadata = json.loads(connection.execute("SELECT metadata_json FROM sessions WHERE id=?", (before["session_id"],)).fetchone()[0])
        assert metadata["user_pause_requested"] is True
        event = json.loads(connection.execute("SELECT payload_json FROM events WHERE event_type='run_paused'").fetchone()[0])
        assert event["source"] == "user_requested_competition_pause" and event["external_actions_dispatched"] == 0
    with pytest.raises(tx.Hold, match="existing_transaction"):
        tx.activate(spec, plan, ops)


@pytest.mark.parametrize("phase", ["before", "candidate_preflight_passed", "service_stopped", "sqlite_backed_up", "competition_paused",
                                   "runtime_switched", "launcher_switched", "configuration_switched", "integrity_seals_switched",
                                   "canary_validation_pending", "partial_start", "ready"])
def test_each_failure_stage_never_restores_research_databases(prepared, phase):
    spec = prepared
    plan, ops = tx.validate(spec), Operations(spec, phase)
    result = tx.activate(spec, plan, ops)
    assert result["status"] == "HOLD" and result["research_database_restored"] is False
    assert tx.tree(Path(spec["root"]) / tx.RUNTIME_REL)["tree_sha256"] == spec["baseline"]["runtime_tree_sha256"]
    assert tx.sha(Path(spec["root"]) / tx.CONFIG_REL) == spec["baseline"]["config_sha256"]
    assert tx.sha(Path(spec["root"]) / tx.LAUNCHER_REL) == spec["baseline"]["launcher_sha256"]
    for seal in spec["seals"]:
        assert tx.sha(Path(spec["root"]) / seal["target"]) == seal["before_sha256"]
    if "competition_paused" in ops.actions:
        assert tx.execution_snapshot(spec["root"])["status"] == "paused"
    if phase == "ready":
        with tx.read_only(Path(spec["root"]) / "data/workspace/runtime/runtime.sqlite3") as connection:
            assert connection.execute("SELECT value FROM new_research WHERE id='new_run'").fetchone()[0] == "must survive rollback"
    assert ops.starts <= 1  # rollback never restarts old code without pause proof


@pytest.mark.parametrize("phase", ["stop", "stop_observation", "rollback_stop", "configuration_drift"])
def test_uncertain_stop_or_external_config_drift_is_manual_hold(prepared, phase):
    ops = Operations(prepared, phase)
    plan = tx.validate(prepared)
    result = tx.activate(prepared, plan, ops)
    assert result["status"] == "HOLD"
    if phase in {"stop", "stop_observation"}:
        assert "runtime_switched" not in ops.actions and "sqlite_backed_up" not in ops.actions
    else:
        assert result["rollback_status"] == "HOLD_manual_reconciliation"
    if phase == "configuration_drift":
        assert json.loads((Path(prepared["root"]) / tx.CONFIG_REL).read_text())["hpc"]["state"] == "frozen_by_startup"


@pytest.mark.parametrize("change", ["short", "nineteen", "failed_case", "duplicate_case", "wrong_model", "preflight", "injected",
                                    "wrong_candidate", "wrong_harness", "bool_duration", "tools_49"])
def test_model_gate_rejects_short_failed_or_unbound_evidence_before_stop(prepared, change):
    def mutate(report):
        if change == "short": report["elapsed_seconds"] = 5399.999
        if change == "nineteen": report["cases"].pop()
        if change == "failed_case": report["cases"][-1]["passed"] = False
        if change == "duplicate_case": report["cases"][-1]["run_id"] = report["cases"][0]["run_id"]
        if change == "wrong_model": report["model"] = "gpt-6-astra"
        if change == "preflight": report["preflight_only"] = True
        if change == "injected": report["injected_faults"] = 1
        if change == "wrong_candidate": report["candidate_manifest_sha256"] = "0" * 64
        if change == "wrong_harness": report["harness_sha256"] = "0" * 64
        if change == "bool_duration": report["elapsed_seconds"] = True
        if change == "tools_49":
            for index, case in enumerate(report["cases"]): case["tool_evidence"]["successful_tool_count"] = 3 if index < 9 else 2
            report["native_tools_executed"] = 49
    mutate_reference(prepared["endurance"], mutate)
    with pytest.raises(tx.Hold):
        tx.validate(prepared)
    assert not (Path(prepared["root"]) / "backups").exists()


@pytest.mark.parametrize("change", ["missing", "duplicate", "string_passed", "wrong_package"])
def test_server_isolation_gate_requires_exact_64_strict_checks(prepared, change):
    manifest = json.loads(Path(prepared["isolation"]["path"]).read_text())
    row = next(item for item in manifest["files"] if item["path"] == "acceptance-result.json")
    path = Path(prepared["isolation"]["path"]).parent / row["path"]
    receipt = json.loads(path.read_text())
    if change == "missing": receipt["checks"].pop()
    if change == "duplicate": receipt["checks"][-1] = deepcopy(receipt["checks"][0])
    if change == "string_passed": receipt["checks"][0]["passed"] = "true"
    if change == "wrong_package": next(item for item in receipt["checks"] if item["name"] == "candidate_identity")["runtime_sha256"] = "0" * 64
    put(path, receipt)
    row.update(bytes=path.stat().st_size, sha256=tx.sha(path))
    prepared["isolation"]["sha256"] = put(Path(prepared["isolation"]["path"]), manifest)["sha256"]
    with pytest.raises(tx.Hold):
        tx.validate(prepared)


@pytest.mark.parametrize("status", ["running", "unknown", "cancelled"])
def test_nonterminal_or_unknown_tool_never_becomes_a_paused_idle_exception(prepared, status):
    database = Path(prepared["root"]) / "data/workspace/runtime/runtime.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE tool_calls SET status=? WHERE id='call_fixture_0'", (status,))
    with pytest.raises(tx.Hold, match="unknown_tool_execution"):
        tx.validate(prepared)
    with tx.read_only(database) as connection:
        assert connection.execute("SELECT status FROM tool_calls WHERE id='call_fixture_0'").fetchone()[0] == status


@pytest.mark.parametrize("status", ["reserved", "uncertain", "exceeded", "unknown"])
def test_gpu_budget_uncertainty_cannot_be_repaired_by_cutover(prepared, status):
    database = Path(prepared["root"]) / "data/workspace/runtime/gpu_budget.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE gpu_operations SET status=?", (status,))
    with pytest.raises(tx.Hold, match="gpu_budget_or_worker"):
        tx.validate(prepared)
    with tx.read_only(database) as connection:
        assert connection.execute("SELECT status,charged FROM gpu_operations").fetchone()[:] == (status, 100)


def test_missing_session_mapping_cannot_pass_by_an_empty_tool_set(prepared):
    database = Path(prepared["root"]) / "data/workspace/runtime/runtime.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE assistant_runs SET session_id='different_session' WHERE id=?", (tx.PAUSE_RUN,))
    with pytest.raises(tx.Hold, match="tool_count"):
        tx.validate(prepared)


def test_dependency_shadow_module_or_missing_launcher_binding_is_hold(prepared):
    (Path(prepared["dependencies"]["source"]) / "unexpected.py").write_bytes(b"unexpected")
    with pytest.raises(tx.Hold, match="dependency_tree_changed"):
        tx.validate(prepared)


def test_dependency_binding_cannot_select_another_valid_directory(prepared):
    mutate_reference(prepared["dependencies"]["launcher_binding"], lambda value: value.update(
        target=str(Path(prepared["root"]) / "report-envs/another/site-packages")))
    with pytest.raises(tx.Hold, match="launcher_dependency_consumption"):
        tx.validate(prepared)


def test_declared_unknown_worker_blocks_even_when_all_tool_rows_are_terminal(prepared):
    mutate_reference(prepared["worker_inventory"], lambda value: value.update(unknown_workers=["unknown-worker"]))
    with pytest.raises(tx.Hold, match="external_worker_quiescence"):
        tx.validate(prepared)


def test_plan_drift_and_create_only_backups_prevent_repeated_cutover(prepared):
    plan = tx.validate(prepared)
    (Path(prepared["root"]) / tx.RUNTIME_REL / "version.py").write_bytes(b"changed-after-plan")
    ops = Operations(prepared)
    with pytest.raises(tx.Hold, match="active_runtime_changed"):
        tx.activate(prepared, plan, ops)
    assert ops.actions == []


def test_sqlite_backup_observes_committed_wal_rows_and_is_create_only(prepared, tmp_path):
    database = Path(prepared["root"]) / "data/workspace/runtime/report-jobs.sqlite3"
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA wal_autocheckpoint=0")
        connection.execute("CREATE TABLE committed_wal(id INTEGER PRIMARY KEY,value TEXT)")
        connection.commit()
        connection.execute("INSERT INTO committed_wal VALUES(1,'committed before backup')")
        connection.commit()
        backup = tmp_path / "sqlite-backups"
        backup.mkdir()
        result = tx.backup_databases(prepared["root"], backup)
        assert result["report_jobs"] == "backed_up"
        with tx.read_only(backup / "sqlite/workspace/runtime/report-jobs.sqlite3") as copy:
            assert copy.execute("SELECT value FROM committed_wal").fetchone()[0] == "committed before backup"
        with pytest.raises(FileExistsError):
            tx.backup_databases(prepared["root"], backup)
    finally:
        connection.close()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/bad", "a\\b", "CON.txt", "bad. "])
def test_archive_and_write_paths_reject_windows_alias_forms(name):
    with pytest.raises(tx.Hold):
        tx.safe_relative(name)
