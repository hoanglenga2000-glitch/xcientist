"""Hash-bound GPT-5.5 canary activation; default CLI mode is read-only.

No ownership migration, model request, GPU command or research-DB restoration.
The apply path is an explicit operator action.  Its service adapter invokes the
existing managed launcher (which itself can start/reverify the HPC bridge), so
that existing startup behavior requires explicit acknowledgement in the spec.
Fixture tests inject operations and never run a service/PowerShell launcher.

An interrupted transaction is not automatically resumed: its create-only backup
and phase receipts require reconciliation. Even success is HOLD pending real
CUDA/Chrome acceptance, never a declaration of full release readiness.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import uuid
from urllib.parse import urlsplit
import zipfile


PRODUCTION_ROOT = Path("C:/ProgramData/EvoMind")
MUTEX_NAME = "Global\\EvoMind-Byoa-V12-Deployment"
PAUSE_RUN = "run_0c57bbde44e94a988843c2f6981f3c65"
SCHEMA = "evomind.syscomplete_activation_spec.v1"
MODEL = "gpt-5.5"
RUNTIME_REL = "bundle/runtime/evomind_runtime"
CONFIG_REL = "config/node-config.json"
LAUNCHER_REL = "bundle/scripts/Start-Node.ps1"
RUNTIME_MANIFEST_REL = "bundle/runtime/runtime-hotfix-manifest.json"
REQUIRED_CHECKS = {
    "acceptance-result.json": {
        "candidate_identity", "web_health_identity", "unauthenticated_denied", "real_login_endpoint", "session_status",
        "missing_csrf_denied", "generic_mutation_denied", "upload_created", "foreign_upload_denied", "chunk_uploaded",
        "upload_verified", "upload_completion_idempotent", "run_created", "run_creation_idempotent",
        "changed_idempotent_payload_denied", "principal_scoped_request_keys", "independent_default_conversations",
        "fixture_run_completed", "no_default_task_context", "foreign_run_denied", "foreign_run_events_denied",
        "event_cursor_no_replay", "fixture_artifact_published", "artifact_bytes_verified", "foreign_artifact_denied",
        "readonly_five_client_latency", "restart_restores_same_run", "restart_restores_artifact",
        "restart_preserves_owner_scope", "restart_preserves_request_key", "single_fixture_invocation", "production_unchanged",
    },
    "project-acceptance.json": {
        "project_unauthenticated_denied", "project_csrf_required", "project_create", "project_replay", "project_payload_conflict",
        "project_foreign_read_denied", "project_foreign_list_empty", "project_owner_spoof_denied", "project_foreign_run_denied",
        "project_run_binding", "project_run_replay", "project_run_fingerprint", "project_survives_restart",
        "project_run_survives_restart", "project_single_invocation", "production_unchanged",
    },
    "tensor-acceptance.json": {"independent_tensor_reload", "tampered_model_rejected", "self_assertion_rejected",
        "foreign_protocol_rejected", "uncertain_gpu_budget_blocks_replacement", "managed_training_permission_contract"},
    "async-approval-acceptance.json": {"approval_ack_before_action_settlement", "approved_action_started",
        "same_run_progress_readable_during_action", "request_identity_context_preserved", "inflight_replay_does_not_repeat_action",
        "coordinator_resumes_after_settlement", "terminal_replay_does_not_redispatch", "one_durable_tool_execution",
        "local_workers_drained", "production_unchanged"},
}
LOCAL_TOOLS = {"file_read", "file_write", "file_list", "file_search", "file_patch", "file_copy", "file_move", "file_delete",
               "artifact_list", "artifact_publish", "artifact_preview", "artifact_bundle", "artifact_import", "attachment_list", "attachment_read",
               "report_generate", "report_status", "directory_hash", "runtime_health", "verified_context", "memory_search"}


class Hold(ValueError):
    pass


def require(condition, code):
    if not condition:
        raise Hold(code)


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_value(value):
    require(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value), "hash_contract_invalid")
    return value


def safe_relative(value):
    require(isinstance(value, str) and value and "\\" not in value and ":" not in value, "relative_path_invalid")
    path = PurePosixPath(value)
    require(not path.is_absolute() and str(path) == value and all(part not in {"", ".", ".."}
        and not part.endswith((".", " ")) and not re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", part)
        for part in path.parts), "relative_path_invalid")
    return path


def clean_path(path, root=None):
    path = Path(path).absolute()
    require(path.resolve(strict=False) == path, "path_alias_rejected")
    for part in (path, *path.parents):
        if part.exists():
            info = part.lstat()
            require(not part.is_symlink() and not (getattr(info, "st_file_attributes", 0) & 0x400), "path_alias_rejected")
    if root is not None:
        try:
            path.relative_to(Path(root).absolute())
        except ValueError:
            raise Hold("path_scope_rejected") from None
        require(path != Path(root).absolute(), "broad_target_rejected")
    return path


def leaf(root, relative):
    return clean_path(Path(root).joinpath(*safe_relative(relative).parts), root)


def checked_ref(reference):
    require(set(reference) == {"path", "sha256"}, "receipt_reference_invalid")
    path = clean_path(reference["path"])
    require(path.is_file() and sha(path) == hash_value(reference["sha256"]), "receipt_hash_mismatch")
    return json.loads(path.read_text(encoding="utf-8-sig")), path


def tree(root):
    root = clean_path(root)
    require(root.is_dir(), "tree_missing")
    rows = []
    for path in sorted(root.rglob("*")):
        clean_path(path, root)
        if path.is_file():
            rows.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)})
    require(len(rows) <= 50000, "tree_size_limit")
    return {"files": rows, "file_count": len(rows), "tree_sha256": digest_bytes(canonical(rows).encode())}


def archive_manifest(path, expected_sha, name):
    require(sha(clean_path(path)) == hash_value(expected_sha), "package_hash_mismatch")
    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        names = [item.filename for item in members]
        require(0 < len(members) <= 50000 and sum(item.file_size for item in members) <= 2 * 1024**3,
                "package_expansion_limit")
        require(len(names) == len({value.casefold() for value in names}) and name in names, "package_file_set_invalid")
        for item in archive.infolist():
            safe_relative(item.filename.rstrip("/"))
            require(not stat.S_ISLNK(item.external_attr >> 16) and not (item.flag_bits & 1), "package_alias_or_encryption_rejected")
        manifest = json.loads(archive.read(name))
        rows = manifest.get("files", [])
        require(manifest.get("file_count") == len(rows) and len(rows) > 0
                and {row["path"] for row in rows} == set(names) - {name}
                and len(rows) == len({row["path"].casefold() for row in rows}), "package_manifest_file_set_invalid")
        for row in rows:
            safe_relative(row["path"])
            require(type(row.get("bytes")) is int and row["bytes"] >= 0, "package_size_invalid")
            info = archive.getinfo(row["path"])
            with archive.open(info) as handle:
                digest = hashlib.sha256()
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            require(info.file_size == row["bytes"] and digest.hexdigest() == hash_value(row["sha256"]), "package_payload_mismatch")
        return manifest


def verify_isolation(reference, delivery):
    manifest, path = checked_ref(reference)
    require(manifest.get("schema") == "evomind.syscomplete_server_isolation_receipts.v1"
            and manifest.get("status") == "passed" and manifest.get("build_id") == delivery["build_id"]
            and manifest.get("production_unchanged") is True and manifest.get("frozen_harnesses_modified") is False
            and manifest.get("total_checks") == 64, "server_isolation_gate_failed")
    rows = manifest["files"]
    require(len(rows) == len({row["path"] for row in rows}), "isolation_duplicate_receipt")
    actual = {}
    for row in rows:
        item = leaf(path.parent, row["path"])
        require(item.stat().st_size == row["bytes"] and sha(item) == row["sha256"], "isolation_receipt_changed")
        actual[row["path"]] = item
    require(set(REQUIRED_CHECKS).issubset(actual), "isolation_receipt_missing")
    for name, expected in REQUIRED_CHECKS.items():
        receipt = json.loads(actual[name].read_text(encoding="utf-8-sig"))
        checks = receipt.get("checks", [])
        require(receipt.get("status") == "passed" and len(checks) == len(expected)
                and {row.get("name") for row in checks} == expected and all(row.get("passed") is True for row in checks),
                "isolation_required_check_failed")
        identity = next((row for row in checks if row["name"] == "candidate_identity"), receipt)
        require(identity.get("web_sha256") == delivery["web"]["sha256"]
                and identity.get("runtime_sha256") == delivery["runtime"]["sha256"], "isolation_package_binding_mismatch")
        if "build_id" in receipt:
            require(receipt["build_id"] == delivery["build_id"], "isolation_build_binding_mismatch")
    return {"checks": 64, "receipt_sha256": reference["sha256"]}


def verify_endurance(reference, candidate_ref, runtime_manifest, profile):
    report, _ = checked_ref(reference)
    candidate, path = checked_ref(candidate_ref)
    require(report.get("schema") == "evomind.model_service_endurance.v1" and report.get("model") == MODEL
            and report.get("status") == "completed" and report.get("service_soak_passed") is True
            and report.get("runtime_closed_cleanly") is True and report.get("preflight_only") is False
            and report.get("injected_faults") == 0 and report.get("gpu_actions") == 0
            and report.get("candidate_manifest_sha256") == candidate_ref["sha256"], "model_endurance_gate_failed")
    duration = report.get("elapsed_seconds")
    require(type(duration) in {int, float} and math.isfinite(duration) and duration >= 5400
            and report.get("required_seconds") == 5400, "model_endurance_duration_failed")
    cases = report.get("cases", [])
    require(isinstance(cases, list) and len(cases) >= 20 and len({row.get("run_id") for row in cases}) == len(cases)
            and len({row.get("index") for row in cases}) == len(cases), "model_case_coverage_failed")
    successful = 0
    for case in cases:
        proof = case.get("tool_evidence", {})
        count = proof.get("successful_tool_count")
        require(case.get("passed") is True and case.get("status") == "completed" and case.get("case_timeout") is False
                and proof.get("required_tool_steps_passed") is True and proof.get("hash_check_passed") is True
                and proof.get("readback_passed") is True and type(count) is int and count > 0,
                "model_case_evidence_failed")
        hash_value(case.get("output_sha256"))
        if case.get("needs_report"):
            require(proof.get("report_job_evidence", {}).get("passed") is True, "model_report_case_failed")
        successful += count
    require(type(report.get("native_tools_executed")) is int and report["native_tools_executed"] == successful
            and successful >= 50, "model_successful_tool_coverage_failed")
    rows = candidate.get("files", [])
    require(len(rows) == len({row["path"] for row in rows}), "model_candidate_duplicate_path")
    bound = {row["path"]: row for row in rows}
    for row in runtime_manifest["files"]:
        require(row["path"] in bound and bound[row["path"]]["sha256"] == row["sha256"], "model_runtime_candidate_mismatch")
    harness = bound.get("run_model_endurance_acceptance.py", {})
    require(report.get("harness_sha256") == harness.get("sha256") and bool(harness), "model_harness_binding_failed")
    require(report.get("wire_protocol") == "responses" and profile.get("protocol") == "responses_stream_v1"
            and report.get("reasoning_effort") == profile.get("reasoning_effort")
            and report.get("service_tier") == profile.get("service_tier") == "omit"
            and report.get("provider_target") == urlsplit(profile["endpoint"]).netloc
            and report.get("route") == "configured_upstream", "model_launcher_profile_mismatch")
    return {"elapsed_seconds": duration, "cases": len(cases), "successful_tools": successful,
            "real_transport_failures": report.get("real_transport_failures"), "receipt_sha256": reference["sha256"]}


def verify_dependencies(spec, delivery, launcher_ref):
    deps = spec["dependencies"]
    installed, _ = checked_ref(deps["installed_manifest"])
    install, _ = checked_ref(deps["install_receipt"])
    render, _ = checked_ref(deps["render_receipt"])
    source = clean_path(deps["source"])
    final = clean_path(deps["target"], Path(spec["root"]) / "report-envs")
    require(final.name == "site-packages" and final.parent.parent == Path(spec["root"]) / "report-envs", "dependency_target_scope_invalid")
    require(tree(source) == installed, "dependency_tree_changed")
    for phase, receipt in (("install", install), ("render", render)):
        require(receipt.get("schema") == "evomind.report_dependency_delivery_acceptance.v1"
                and receipt.get("phase") == phase and receipt.get("status") == "staged_not_activated"
                and receipt.get("stage_progress") == "complete" and receipt.get("base_venv_unchanged") is True
                and receipt.get("base_tree_sha256_before") == receipt.get("base_tree_sha256_after")
                and receipt.get("production_changed") is False and receipt.get("activated") is False
                and receipt.get("installed_manifest_sha256") == deps["installed_manifest"]["sha256"]
                and clean_path(receipt["target"]) == source, "dependency_delivery_receipt_invalid")
    for field in ("descriptor_sha256", "inheritance_sha256", "acceptance_manifest_sha256"):
        hash_value(install.get(field))
        require(install.get(field) == render.get(field), "dependency_phase_binding_mismatch")
    report = render.get("report", {})
    require(report.get("model_requests") == 0 and report.get("hpc_actions") == 0 and report.get("network_attempts") == 0
            and type(report.get("pdf_pages")) is int and report["pdf_pages"] > 0
            and type(report.get("editable_word_tables")) is int and report["editable_word_tables"] > 0,
            "dependency_render_gate_failed")
    binding, _ = checked_ref(deps["launcher_binding"])
    require(binding.get("schema") == "evomind.syscomplete_launcher_dependency_binding.v1" and binding.get("status") == "passed"
            and binding.get("launcher_manifest_sha256") == launcher_ref["sha256"]
            and binding.get("runtime_sha256") == delivery["runtime"]["sha256"]
            and binding.get("dependency_tree_sha256") == installed["tree_sha256"]
            and clean_path(binding["target"]) == final and binding.get("config_key") == "report_dependency_root"
            and binding.get("consumer_roles") == ["python_runtime", "web"]
            and binding.get("path_precedence_verified") is True and binding.get("profile_unchanged") is True,
            "launcher_dependency_consumption_unproved")
    return {"source": str(source), "target": str(final), "tree_sha256": installed["tree_sha256"],
            "descriptor_sha256": install["descriptor_sha256"], "installed_files": installed["file_count"]}


@contextmanager
def read_only(database):
    connection = sqlite3.connect(clean_path(database).as_uri() + "?mode=ro", uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    try:
        yield connection
    finally:
        connection.close()


def execution_snapshot(root, expected_count=22):
    runtime = Path(root) / "data/workspace/runtime"
    with read_only(runtime / "runtime.sqlite3") as connection:
        runs = [dict(row) for row in connection.execute("SELECT * FROM assistant_runs ORDER BY id")]
        target = next((row for row in runs if row["id"] == PAUSE_RUN), None)
        require(target is not None and target["status"] in {"recovering", "paused"}, "competition_run_state_requires_review")
        require(all(row["id"] == PAUSE_RUN or row["status"] in {"completed", "failed", "cancelled", "blocked"} for row in runs),
                "other_nonterminal_run_prevents_activation")
        calls = [dict(row) for row in connection.execute("SELECT * FROM tool_calls WHERE session_id=? ORDER BY id", (target["session_id"],))]
        require(len(calls) == expected_count and expected_count == 22, "competition_tool_count_mismatch")
        require(all(row["status"] in {"completed", "failed"} for row in calls), "unknown_tool_execution_prevents_activation")
        require(connection.execute("SELECT count(*) FROM tool_calls WHERE status NOT IN ('completed','failed')").fetchone()[0] == 0,
                "unsettled_tool_prevents_activation")
        session = connection.execute("SELECT * FROM sessions WHERE id=?", (target["session_id"],)).fetchone()
        require(session is not None, "competition_session_missing")
        protected = {"run": {key: value for key, value in target.items() if key not in {"status", "updated_at"}},
                     "session": {key: value for key, value in dict(session).items() if key not in {"status", "updated_at", "metadata_json"}},
                     "metadata": {key: value for key, value in json.loads(session["metadata_json"]).items() if key != "user_pause_requested"},
                     "tools": calls}
    with read_only(runtime / "gpu_budget.sqlite3") as connection:
        budgets = [dict(row) for row in connection.execute("SELECT * FROM gpu_operations ORDER BY id")]
        require(all(row["status"] in {"passed", "failed"} for row in budgets), "gpu_budget_or_worker_reconciliation_required")
        require(all(type(row["charged"]) in {int, float} and math.isfinite(row["charged"]) and row["charged"] >= 0 for row in budgets),
                "gpu_budget_invalid")
    reports = runtime / "report-jobs.sqlite3"
    if reports.exists():
        with read_only(reports) as connection:
            require(connection.execute("SELECT count(*) FROM jobs WHERE status NOT IN ('ready','partial','failed')").fetchone()[0] == 0,
                    "report_job_reconciliation_required")
    return {"run_id": PAUSE_RUN, "session_id": target["session_id"], "status": target["status"], "tool_count": len(calls),
            "research_identity_sha256": digest_bytes(canonical(protected).encode()), "tool_receipts_sha256": digest_bytes(canonical(calls).encode()),
            "budget_sha256": digest_bytes(canonical(budgets).encode()), "charged_seconds": sum(row["charged"] for row in budgets),
            "tool_ids": [row["id"] for row in calls], "tool_names": {row["id"]: row["tool_name"] for row in calls}}


def verify_worker_inventory(reference, snapshot):
    receipt, _ = checked_ref(reference)
    require(receipt.get("schema") == "evomind.syscomplete_quiescence_inventory.v1"
            and receipt.get("run_id") == PAUSE_RUN and receipt.get("tool_receipts_sha256") == snapshot["tool_receipts_sha256"]
            and receipt.get("budget_sha256") == snapshot["budget_sha256"] and receipt.get("all_started_actions_accounted") is True
            and receipt.get("unknown_workers") == [] and receipt.get("active_external_workers") == [], "external_worker_quiescence_unproved")
    rows = receipt.get("calls", [])
    require(len(rows) == snapshot["tool_count"] and {row.get("call_id") for row in rows} == set(snapshot["tool_ids"]),
            "worker_inventory_coverage_invalid")
    for row in rows:
        name = snapshot["tool_names"][row["call_id"]]
        if row.get("state") == "not_external":
            require(name in LOCAL_TOOLS, "unrecognized_tool_requires_worker_receipt")
        else:
            require(row.get("state") == "terminal_verified" and row.get("identity_verified") is True,
                    "unknown_external_worker")
            checked_ref(row["evidence"])


def launcher_support(spec, launcher, manifest_path):
    """Resolve the two v2 support files against explicit pre-change identities."""
    if launcher.get("schema") == "evomind.launcher_candidate.v1":
        return []
    require(launcher.get("schema") == "evomind.launcher_candidate.v2"
            and launcher.get("activation_claim") is False, "launcher_contract_invalid")
    root = Path(spec["root"])
    target = clean_path(spec["dependencies"]["target"], root / "report-envs")
    allowed = {
        "candidate/Start-Node.ps1": root / LAUNCHER_REL,
        "candidate/lib/verify_report_dependencies.py": root / "bundle/scripts/lib/verify_report_dependencies.py",
        "candidate/report-dependency-activation.json": target.parent / "report-dependency-activation.json",
    }
    rows = launcher.get("deployment_files", [])
    require(len(rows) == 3 and {row.get("path") for row in rows} == set(allowed), "launcher_support_file_set_invalid")
    baselines = spec["baseline"].get("launcher_support_files", [])
    require(len(baselines) == 2 and len({row.get("target") for row in baselines}) == 2, "launcher_support_baseline_required")
    before = {row["target"]: row["sha256"] for row in baselines}
    result = []
    for row in rows:
        destination = clean_path(row["target"], root)
        require(destination == allowed[row["path"]], "launcher_support_target_mismatch")
        source = leaf(manifest_path.parent, row["path"])
        require(sha(source) == hash_value(row["sha256"]), "launcher_support_source_changed")
        if row["path"] == "candidate/Start-Node.ps1":
            require(row["sha256"] == launcher["candidate"]["sha256"], "launcher_support_identity_mismatch")
            continue
        relative = destination.relative_to(root).as_posix()
        require(relative in before, "launcher_support_baseline_missing")
        actual = sha(destination) if destination.exists() else None
        require(actual == before[relative], "launcher_support_target_changed")
        if actual is not None:
            hash_value(actual)
        result.append({"target": relative, "source": str(source), "before_sha256": actual, "after_sha256": row["sha256"],
                       "dependency_binding": row["path"].endswith("report-dependency-activation.json")})
        if result[-1]["dependency_binding"]:
            require(actual is None, "dependency_binding_already_exists")
    return result


def required_report_export_checks():
    expected = {"report_real_owner_login", "report_fixture_run_created", "report_fixture_run_completed",
                "report_unauthenticated_get", "report_unauthenticated_post", "report_csrf_required",
                "report_ack_p95", "three_frozen_report_jobs_only"}
    expected.update("report_idempotent_resume_"+str(index) for index in range(1, 21))
    for kind in ("diagnostic", "training", "inference"):
        expected.update(kind+"_"+name for name in ("report_accepted", "report_terminal", "execution_status_preserved",
            "not_scientific_validation", "required_formats_published", "editable_word", "readable_pdf"))
        for artifact in ("report.html", "report.pdf", "report.docx", "report-bundle.zip"):
            expected.add(kind+"_"+artifact+"_download")
            for index in range(2):
                expected.add(kind+"_"+artifact+"_other_"+str(index))
                expected.add(kind+"_"+artifact+"_preview_other_"+str(index))
        for index in range(2):
            expected.update(kind+"_"+action+"_other_"+str(index) for action in ("list", "read", "generate", "resume"))
    return expected


def verify_report_export_gate(reference, delivery):
    report, _ = checked_ref(reference)
    require(report.get("schema") == "evomind.report_export_http_acceptance.v1"
            and report.get("status") == "passed" and report.get("fixture_only") is True
            and report.get("production_unchanged") is True and report.get("report_site_unchanged") is True
            and report.get("cleanup") == "owned_fixture_processes_stopped"
            and report.get("build_id") == delivery["build_id"]
            and report.get("web_sha256") == delivery["web"]["sha256"]
            and report.get("runtime_sha256") == delivery["runtime"]["sha256"], "report_export_gate_failed")
    expected = required_report_export_checks()
    checks = report.get("checks", [])
    require(len(checks) == len(expected) == 133 and {row.get("name") for row in checks} == expected
            and all(row.get("passed") is True for row in checks), "report_export_coverage_incomplete")
    latency = next(row for row in checks if row["name"] == "report_ack_p95")
    value = latency.get("p95_ms")
    require(type(value) in {int, float} and math.isfinite(value) and 0 <= value <= 1000
            and latency.get("samples") == 20, "report_export_ack_latency_failed")
    return {"checks": 133, "p95_ms": value, "receipt_sha256": reference["sha256"]}


def validate(spec):
    require(spec.get("schema") == SCHEMA and spec.get("intent") == "canary_validation_pending"
            and spec.get("pause", {}).get("run_id") == PAUSE_RUN, "activation_scope_invalid")
    hash_value(spec["pause"]["authorization_sha256"])
    root = clean_path(spec["root"])
    stage = clean_path(spec["stage"], root / "staging")
    require(re.fullmatch(r"[a-zA-Z0-9_-]{8,80}", spec["transaction_id"]), "transaction_id_invalid")
    backup = leaf(root, "backups/" + spec["transaction_id"])
    require(not backup.exists(), "existing_transaction_requires_reconciliation")
    delivery, _ = checked_ref(spec["delivery"])
    require(delivery.get("schema") == "evomind.syscomplete_delivery.v1"
            and delivery.get("status") == "three_packages_built_not_deployed" and delivery.get("production_deployed") is False,
            "delivery_contract_invalid")
    build = delivery["build_id"]
    require(re.fullmatch(r"overlay-invitation-beta-[a-f0-9]{12}-sys\d+", build), "build_id_invalid")
    web_target = leaf(root, "web-overlays/" + build)
    require(not web_target.exists(), "candidate_target_already_exists")
    manifests = {kind: archive_manifest(stage / (kind + ".zip"), delivery[kind]["sha256"], name)
                 for kind, name in (("web", "operational-overlay-manifest.json"), ("runtime", "runtime-hotfix-manifest.json"),
                                    ("acceptance", "acceptance-manifest.json"))}
    require(manifests["web"].get("overlay_id") == build and manifests["runtime"].get("release") == build
            and manifests["acceptance"].get("build_id") == build, "package_build_mismatch")
    launcher, launcher_path = checked_ref(spec["launcher"])
    require(launcher.get("schema") in {"evomind.launcher_candidate.v1", "evomind.launcher_candidate.v2"} and launcher.get("production_changed") is False,
            "launcher_contract_invalid")
    candidate_launcher = leaf(launcher_path.parent, launcher["candidate"]["path"])
    require(sha(candidate_launcher) == launcher["candidate"]["sha256"]
            and launcher["profile"].get("model") == MODEL and launcher["profile"].get("provider_strict") is True,
            "launcher_profile_invalid")
    external = delivery.get("external_launcher_binding", {})
    require(external.get("manifest_sha256") == spec["launcher"]["sha256"]
            and external.get("candidate_sha256") == launcher["candidate"]["sha256"], "delivery_launcher_binding_mismatch")
    gates = {"isolation": verify_isolation(spec["isolation"], delivery),
             "endurance": verify_endurance(spec["endurance"], spec["endurance_candidate"], manifests["runtime"], launcher["profile"]),
             "dependencies": verify_dependencies(spec, delivery, spec["launcher"])}
    support = launcher_support(spec, launcher, launcher_path)
    if support:
        require(external.get("deployment_files") == launcher["deployment_files"], "delivery_launcher_support_mismatch")
        gates["report_exports"] = verify_report_export_gate(spec["report_exports"], delivery)
    require(not Path(gates["dependencies"]["target"]).exists(), "dependency_target_already_exists")
    snapshot = execution_snapshot(root)
    verify_worker_inventory(spec["worker_inventory"], snapshot)
    baseline = spec["baseline"]
    require(tree(root / RUNTIME_REL)["tree_sha256"] == baseline["runtime_tree_sha256"], "active_runtime_changed")
    require(sha(root / CONFIG_REL) == baseline["config_sha256"] and sha(root / LAUNCHER_REL) == baseline["launcher_sha256"],
            "active_config_or_launcher_changed")
    config = json.loads((root / CONFIG_REL).read_text(encoding="utf-8-sig"))
    old_web = clean_path(config["web_runtime_root"], root / "web-overlays")
    require(tree(old_web)["tree_sha256"] == baseline["web_tree_sha256"], "active_web_changed")
    require(clean_path(config["data_root"]) == root / "data", "data_root_out_of_scope")
    protected = baseline.get("protected_files", [])
    require(protected and len(protected) == len({row["path"] for row in protected}), "protected_baseline_missing")
    for row in protected:
        require(sha(leaf(root, row["path"])) == hash_value(row["sha256"]), "protected_configuration_changed")
    seals = spec.get("seals", [])
    require(seals and len(seals) == len({row["target"] for row in seals}), "exact_integrity_seals_required")
    if root == PRODUCTION_ROOT:
        require({row["target"] for row in seals} == {RUNTIME_MANIFEST_REL, "state/bundle-integrity.json"},
                "production_integrity_seal_coverage_invalid")
    for row in seals:
        target = leaf(root, row["target"])
        require((target.is_relative_to(root / "bundle") or row["target"] == "state/bundle-integrity.json") and target.suffix in {".json", ".sha256"}
                and re.search(r"manifest|integrity|sha256", target.name, re.I)
                and not target.is_relative_to(root / RUNTIME_REL), "seal_target_scope_rejected")
        require(sha(target) == hash_value(row["before_sha256"]) and sha(clean_path(row["source"])) == hash_value(row["after_sha256"]),
                "seal_identity_changed")
    return {"schema": "evomind.syscomplete_activation_plan.v1", "status": "planned_not_activated", "release_verdict": "HOLD",
            "script_sha256": sha(Path(__file__)), "spec_sha256": digest_bytes(canonical(spec).encode()), "build_id": build,
            "root": str(root), "stage": str(stage), "backup": str(backup), "old_web_root": str(old_web),
            "web_target": str(web_target), "launcher_source": str(candidate_launcher), "launcher_sha256": launcher["candidate"]["sha256"],
            "launcher_support_files": support,
            "gates": gates, "research_before": snapshot, "baseline": baseline,
            "research_databases_must_never_be_restored": True, "ownership_migration": False}


def write_new(path, data):
    path = clean_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_bytes(path, data):
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    write_new(temporary, data)
    os.replace(temporary, path)


def unpack(path, target):
    target.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            if not item.is_dir():
                write_new(leaf(target, item.filename), archive.read(item))


def verify_extracted(package, target, manifest_name):
    with zipfile.ZipFile(package) as archive:
        raw = archive.read(manifest_name)
        expected = json.loads(raw)["files"] + [{"path": manifest_name, "bytes": len(raw), "sha256": digest_bytes(raw)}]
    by_name = lambda rows: {row["path"]: (row["bytes"], row["sha256"]) for row in rows}
    require(by_name(tree(target)["files"]) == by_name(expected), "extracted_candidate_changed")


def bound_bytes(path, expected):
    payload = clean_path(path).read_bytes()
    require(digest_bytes(payload) == hash_value(expected), "staged_control_file_changed")
    return payload


def backup_databases(root, backup):
    data = Path(root) / "data"
    databases = [path for path in sorted(data.rglob("*")) if path.is_file() and path.suffix in {".sqlite3", ".sqlite", ".db"}]
    require(len(databases) <= 64 and data / "workspace/runtime/runtime.sqlite3" in databases
            and data / "workspace/runtime/gpu_budget.sqlite3" in databases, "research_database_inventory_invalid")
    rows = []
    for path in databases:
        relative = path.relative_to(data).as_posix()
        destination = leaf(backup, "sqlite/" + relative)
        write_new(destination, b"")
        with read_only(path) as source, sqlite3.connect(destination) as copy:
            require(source.execute("PRAGMA quick_check").fetchone()[0] == "ok", "research_database_corrupt")
            source.backup(copy)
            require(copy.execute("PRAGMA quick_check").fetchone()[0] == "ok", "research_backup_corrupt")
        rows.append({"path": relative, "backup_sha256": sha(destination), "bytes": destination.stat().st_size})
    return {"files": rows, "report_jobs": "backed_up" if any(row["path"].endswith("/report-jobs.sqlite3") for row in rows) else "absent_before_activation"}


def pause_competition(root, before, authorization_sha256, transaction_id):
    runtime = Path(root) / "data/workspace/runtime"
    with sqlite3.connect(runtime / "runtime.sqlite3", timeout=5) as connection:
        connection.execute("BEGIN IMMEDIATE")
        current = execution_snapshot(root)
        require(current == before, "research_changed_before_pause")
        session = connection.execute("SELECT metadata_json FROM sessions WHERE id=?", (before["session_id"],)).fetchone()
        metadata = json.loads(session[0])
        metadata["user_pause_requested"] = True
        now = datetime.now(timezone.utc).isoformat()
        connection.execute("UPDATE sessions SET status='paused',metadata_json=?,updated_at=? WHERE id=?",
                           (canonical(metadata), now, before["session_id"]))
        connection.execute("UPDATE assistant_runs SET status='paused',updated_at=? WHERE id=?", (now, PAUSE_RUN))
        seq = connection.execute("SELECT COALESCE(MAX(seq),0)+1 FROM events WHERE session_id=?", (before["session_id"],)).fetchone()[0]
        payload = {"run_id": PAUSE_RUN, "status": "paused", "previous_status": before["status"], "new_work_blocked": True,
                   "source": "user_requested_competition_pause", "authorization_sha256": authorization_sha256,
                   "transaction_id": transaction_id, "external_actions_dispatched": 0, "budget_reset": False}
        connection.execute("INSERT INTO events VALUES(?,?,?,?,?,?)", ("evt_" + uuid.uuid4().hex, before["session_id"], seq, "run_paused", canonical(payload), now))
    after = execution_snapshot(root)
    require(after["status"] == "paused" and all(after[field] == before[field] for field in
            ("research_identity_sha256", "tool_receipts_sha256", "budget_sha256", "charged_seconds")), "pause_changed_research_identity")
    return after


def activate(spec, expected_plan, operations):
    """Only explicit operations injection or the guarded Windows CLI can mutate."""
    require(expected_plan == validate(spec), "activation_plan_or_live_state_changed")
    root, backup, stage = map(Path, (expected_plan["root"], expected_plan["backup"], expected_plan["stage"]))
    result = {"schema": "evomind.syscomplete_activation_result.v1", "status": "HOLD", "release_verdict": "HOLD",
              "research_database_restored": False, "ownership_migration": False, "build_id": expected_plan["build_id"]}
    stopped, stop_attempted, started_attempt, runtime_moved = False, False, False, False
    file_backups, phases = [], []

    def record(phase, **values):
        entry = {"phase": phase, **values}
        write_new(backup / f"{len(phases):03d}-{phase}.json", (canonical(entry) + "\n").encode())
        phases.append(entry)
        operations.checkpoint(phase)

    with operations.mutex():
        require(expected_plan == validate(spec), "activation_state_changed_under_mutex")
        backup.mkdir(parents=True, exist_ok=False)
        operations.protect_backup(backup)
        write_new(backup / "plan.json", (canonical(expected_plan) + "\n").encode())
        try:
            record("before", baseline=expected_plan["baseline"], research=expected_plan["research_before"])
            unpack(stage / "runtime.zip", backup / "candidate-runtime")
            unpack(stage / "web.zip", Path(expected_plan["web_target"]))
            dependency_target = Path(expected_plan["gates"]["dependencies"]["target"])
            shutil.copytree(spec["dependencies"]["source"], dependency_target, symlinks=False)
            require(tree(dependency_target)["tree_sha256"] == expected_plan["gates"]["dependencies"]["tree_sha256"], "copied_dependency_drift")
            # This new immutable directory is not yet consumed by production.
            # Place its signed-by-hash binding so the read-only launch verifier
            # can run before stopping services; never replace an existing file.
            for row in expected_plan.get("launcher_support_files", []):
                if row["dependency_binding"]:
                    target = leaf(root, row["target"])
                    require(not target.exists(), "dependency_binding_already_exists")
                    write_new(target, bound_bytes(row["source"], row["after_sha256"]))
                    file_backups.append((target, None, None, row["after_sha256"]))
            operations.preflight(expected_plan)
            record("candidate_preflight_passed")
            stop_attempted = True
            stop_receipt = operations.stop()
            require(stop_receipt.get("stopped") is True and stop_receipt.get("identity_verified") is True, "service_stop_unconfirmed")
            operations.assert_stopped()
            stopped = True
            fresh = execution_snapshot(root)
            require(fresh == expected_plan["research_before"], "research_changed_during_service_stop")
            verify_worker_inventory(spec["worker_inventory"], fresh)
            record("service_stopped", receipt=stop_receipt)
            backups = backup_databases(root, backup)
            record("sqlite_backed_up", backup=backups)
            paused = pause_competition(root, fresh, spec["pause"]["authorization_sha256"], spec["transaction_id"])
            record("competition_paused", research=paused)
            # Gate receipts and candidate bytes are rechecked after service stop.
            # validate() cannot be rerun after create-only targets now exist.
            for kind in ("web", "runtime", "acceptance"):
                delivery, _ = checked_ref(spec["delivery"])
                require(sha(stage / (kind + ".zip")) == delivery[kind]["sha256"], "candidate_changed_after_stop")
            for reference in (spec["isolation"], spec["endurance"], spec["endurance_candidate"], spec["launcher"],
                              *[spec["dependencies"][name] for name in ("installed_manifest", "install_receipt", "render_receipt", "launcher_binding")]):
                checked_ref(reference)
            verify_extracted(stage / "runtime.zip", backup / "candidate-runtime", "runtime-hotfix-manifest.json")
            verify_extracted(stage / "web.zip", Path(expected_plan["web_target"]), "operational-overlay-manifest.json")
            require(tree(dependency_target)["tree_sha256"] == expected_plan["gates"]["dependencies"]["tree_sha256"], "copied_dependency_drift")
            require(tree(root / RUNTIME_REL)["tree_sha256"] == spec["baseline"]["runtime_tree_sha256"], "runtime_changed_after_stop")
            for row in spec["baseline"]["protected_files"]:
                require(sha(leaf(root, row["path"])) == row["sha256"], "protected_configuration_changed_after_stop")
            for relative, source, expected_after in [
                (LAUNCHER_REL, expected_plan["launcher_source"], expected_plan["launcher_sha256"]),
                *[(row["target"], row["source"], row["after_sha256"]) for row in expected_plan.get("launcher_support_files", []) if not row["dependency_binding"]],
                *[(row["target"], row["source"], row["after_sha256"]) for row in spec["seals"]],
            ]:
                target = leaf(root, relative)
                old = target.read_bytes() if target.exists() else None
                expected_before = spec["baseline"]["launcher_sha256"] if relative == LAUNCHER_REL else next(
                    row["before_sha256"] for row in [*spec["seals"], *expected_plan.get("launcher_support_files", [])] if row["target"] == relative)
                require((digest_bytes(old) if old is not None else None) == expected_before, "control_file_changed_after_stop")
                saved = leaf(backup, "files/" + relative) if old is not None else None
                if saved is not None:
                    write_new(saved, old)
                file_backups.append((target, saved, expected_before, expected_after))
                require(sha(clean_path(source)) == expected_after, "staged_control_file_changed")
            config_path = root / CONFIG_REL
            old_config = config_path.read_bytes()
            require(digest_bytes(old_config) == spec["baseline"]["config_sha256"], "configuration_changed_after_stop")
            config = json.loads(old_config.decode("utf-8-sig"))
            config.update(web_runtime_root=expected_plan["web_target"], report_dependency_root=str(dependency_target))
            new_config = (canonical(config) + "\n").encode()
            saved_config = leaf(backup, "files/" + CONFIG_REL)
            write_new(saved_config, old_config)
            file_backups.append((config_path, saved_config, digest_bytes(old_config), digest_bytes(new_config)))
            (root / RUNTIME_REL).rename(backup / "original-runtime")
            runtime_moved = True
            (backup / "candidate-runtime/evomind_runtime").rename(root / RUNTIME_REL)
            record("runtime_switched", after=tree(root / RUNTIME_REL))
            for row in expected_plan.get("launcher_support_files", []):
                if not row["dependency_binding"]:
                    atomic_bytes(leaf(root, row["target"]), bound_bytes(row["source"], row["after_sha256"]))
            if expected_plan.get("launcher_support_files"):
                record("launcher_support_switched")
            atomic_bytes(root / LAUNCHER_REL, bound_bytes(expected_plan["launcher_source"], expected_plan["launcher_sha256"]))
            record("launcher_switched", sha256=sha(root / LAUNCHER_REL))
            atomic_bytes(config_path, new_config)
            record("configuration_switched", sha256=sha(config_path))
            for row in spec["seals"]:
                atomic_bytes(leaf(root, row["target"]), bound_bytes(row["source"], row["after_sha256"]))
            record("integrity_seals_switched", files=[{"path": row["target"], "sha256": sha(leaf(root, row["target"]))} for row in spec["seals"]])
            operations.verify_integrity(expected_plan["build_id"])
            started_attempt = True
            operations.start()
            stopped = False
            accepted = operations.verify_ready(expected_plan["build_id"])
            require(accepted.get("status") == "ready" and accepted.get("build_id") == expected_plan["build_id"], "candidate_readiness_failed")
            for target, _saved, _before, after in file_backups:
                require(sha(target) == after, "control_file_changed_during_startup")
            for row in spec["baseline"]["protected_files"]:
                require(sha(leaf(root, row["path"])) == row["sha256"], "protected_configuration_changed_during_startup")
            result.update(status="canary_validation_pending", verification=accepted,
                          pending=["real_managed_cuda_acceptance", "real_chrome_end_to_end", "whitelist_gray_validation"])
            record("canary_validation_pending", verification=accepted)
        except Exception as error:
            result.update(status="HOLD", release_verdict="HOLD")
            result["error_code"] = str(error) if re.fullmatch(r"[a-z_]{1,160}", str(error)) else "activation_phase_failed"
            try:
                if started_attempt:
                    receipt = operations.stop()
                    require(receipt.get("stopped") is True and receipt.get("identity_verified") is True, "rollback_stop_unconfirmed")
                    operations.assert_stopped()
                    stopped = True
                if runtime_moved:
                    require(stopped, "rollback_requires_confirmed_stop")
                    # Never overwrite external/user/configuration changes, including
                    # an HPC freeze written by startup. Stop and retain evidence.
                    for target, saved, before, after in file_backups:
                        require((sha(saved) if saved is not None else None) == before
                                and (sha(target) if target.exists() else None) in {before, after}, "rollback_control_drift_requires_review")
                    if (root / RUNTIME_REL).exists():
                        (root / RUNTIME_REL).rename(backup / "failed-runtime")
                    shutil.copytree(backup / "original-runtime", root / RUNTIME_REL)
                    for target, saved, before, _after in reversed(file_backups):
                        if saved is None:
                            if target.exists():
                                retained = leaf(backup, "new-support/" + target.relative_to(root).as_posix())
                                retained.parent.mkdir(parents=True, exist_ok=True)
                                target.rename(retained)
                            require(not target.exists(), "rollback_new_file_removal_failed")
                        else:
                            atomic_bytes(target, saved.read_bytes())
                            require(sha(target) == before, "rollback_file_identity_failed")
                    require(tree(root / RUNTIME_REL)["tree_sha256"] == spec["baseline"]["runtime_tree_sha256"], "rollback_runtime_identity_failed")
                    result["rollback_status"] = "code_restored_service_held"
                    # Old code may replay paused Runs on startup. No default restart.
                    if spec.get("rollback_pause_safe_receipt"):
                        proof, _ = checked_ref(spec["rollback_pause_safe_receipt"])
                        require(proof.get("status") == "passed" and proof.get("paused_run_preserved") is True
                                and proof.get("runtime_tree_sha256") == spec["baseline"]["runtime_tree_sha256"]
                                and proof.get("launcher_sha256") == spec["baseline"]["launcher_sha256"], "rollback_pause_safety_unproved")
                        operations.verify_integrity(Path(expected_plan["old_web_root"]).name)
                        operations.start()
                        stopped = False
                        operations.verify_ready(Path(expected_plan["old_web_root"]).name)
                        result["rollback_status"] = "restored_and_verified_pause_safe"
                else:
                    result["rollback_status"] = ("no_code_switched_service_held" if stopped else
                        "no_code_switched_stop_state_requires_review" if stop_attempted else "no_code_switched_service_unchanged")
            except Exception as rollback_error:
                result.update(rollback_status="HOLD_manual_reconciliation", rollback_error=type(rollback_error).__name__)
        finally:
            result.update(services_confirmed_stopped=stopped, backup=str(backup), phase_count=len(phases))
            write_new(backup / "result.json", (canonical(result) + "\n").encode())
    return result


@contextmanager
def deployment_mutex():
    require(os.name == "nt", "windows_managed_activation_required")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.ReleaseMutex.argtypes = kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.CreateMutexW(None, False, MUTEX_NAME)
    require(bool(handle), "deployment_mutex_unavailable")
    acquired = False
    try:
        answer = kernel.WaitForSingleObject(handle, 0)
        acquired = answer in {0, 0x80}
        require(answer == 0, "deployment_mutex_busy_or_abandoned")
        yield
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--plan-out", type=Path)
    args = parser.parse_args()
    # Operational service adapters intentionally remain a reviewed integration
    # boundary. This standalone command only validates and writes a new plan.
    # Parent may call activate(spec, plan, approved_operations) after all gates.
    spec, _ = checked_ref({"path": str(args.spec), "sha256": args.spec_sha256})
    plan = validate(spec)
    if args.plan_out:
        write_new(args.plan_out, (canonical(plan) + "\n").encode())
    print(canonical({"status": plan["status"], "release_verdict": "HOLD", "build_id": plan["build_id"],
                     "plan_sha256": digest_bytes((canonical(plan) + "\n").encode()), "production_changed": False}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        code = str(error) if re.fullmatch(r"[a-z_]{1,160}", str(error)) else "activation_input_rejected"
        print(canonical({"status": "HOLD", "error_code": code, "production_changed": False}))
        raise SystemExit(2)
