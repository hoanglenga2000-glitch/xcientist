"""Build an immutable candidate from a hash-bound live baseline, not a dirty tree.

This command creates staging artifacts only. It never changes production
services, credentials, profiles, databases, or historical Run ownership.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid
import zipfile


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "research-agent-workstation"
WEB_PATCHES = (
    # Path A-diff: keep 514 AppShell workstation UI; no invitation /workspace shell.
    # Excluded: page/home-client/AppShell/Sidebar/navigation/WorkspaceToolbar/login/workspace/task-workspace + visual chrome.
    "package.json",
    "package-lock.json",
    "src/components/workstation/screens/AssistantScreen.tsx",
    "src/components/workstation/screens/AssistantReportPanel.tsx",
    "src/lib/assistant-report.ts",
    "src/lib/assistant-report.test.ts",
    "src/lib/server/report-assistant-bridge-contract.test.ts",
    "src/app/api/assistant/runs/[runId]/route.ts",
    "src/app/api/assistant/runs/[runId]/reports/route.ts",
    "src/components/workstation/screens/DataKaggleScreen.tsx",
    "src/components/workstation/screens/OverviewScreen.tsx",
    "src/components/workstation/screens/SettingsScreen.tsx",
    "src/components/workstation/screens/ExecutionBudgetPanel.tsx",
    "src/components/workstation/screens/execution-budget.ts",
    "src/components/workstation/screens/execution-budget.test.mjs",
    "src/app/api/assistant/execution-budget/route.ts",
    "src/components/workstation/screens/RunProgressPanel.tsx",
    "src/components/workstation/screens/run-submission.ts",
    "src/components/workstation/screens/assistant-presentation.ts",
    "src/components/workstation/screens/assistant-restoration.ts",
    "src/components/workstation/screens/assistant-restoration.test.mjs",
    "src/components/workstation/screens/assistant-history.ts",
    "src/components/workstation/screens/assistant-history.test.mjs",
    "src/components/workstation/screens/run-progress-presentation.ts",
    "src/components/workstation/screens/run-progress-presentation.test.mjs",
    "src/components/workstation/hpc-quick-enrollment-contract.test.ts",
    "src/components/workstation/TenantHpcQuickEnrollment.tsx",
    "src/app/api/hpc/byoa/binding/route.ts",
    "src/app/api/hpc/byoa/enrollment-identity/route.ts",
    "src/app/api/hpc/byoa/enrollment-envelope/route.ts",
    "src/app/api/hpc/byoa/public-key/route.ts",
    "src/lib/server/byoa-enrollment-contract.test.ts",
    "src/lib/server/service-clock.ts",
    "src/lib/server/service-clock.test.ts",
    "src/lib/server/invitation-workspace-contract.test.ts",
    "src/lib/server/assistant-super-agent-contract.test.ts",
    "src/lib/server/evolution-integrity.test.ts",
    "scripts/lib/loopback-session.mjs",
    "src/components/workstation/screens/ProjectsScreen.tsx",
    "src/app/api/assistant/projects/route.ts",
    "src/app/api/assistant/projects/[projectId]/route.ts",
    "src/lib/server/assistant-runtime.ts",
    "src/lib/connector-presentation.ts",
    "src/lib/connector-presentation.test.mjs",
    "src/lib/server/scoped-runtime-fetch.ts",
    "src/lib/server/runtime-principal.ts",
    "src/app/api/runtime/[...path]/route.ts",
    "src/app/api/assistant/runs/route.ts",
    "src/app/api/assistant/runs/[runId]/actions/route.ts",
    "src/app/api/assistant/run-requests/[key]/route.ts",
    "src/proxy.ts",
    "src/components/workstation/LocalSessionBootstrap.tsx",
    "src/lib/server/local-session.ts",
    "src/lib/server/local-session-security.test.ts",
    "src/app/api/session/status/route.ts",
    "src/lib/server/summary.ts",
    "src/app/api/assistant/approvals/[approvalId]/route.ts",
    "src/app/api/assistant/artifacts/[artifactId]/route.ts",
    "src/app/api/assistant/files/[[...path]]/route.ts",
    "src/app/api/assistant/model-profiles/[[...path]]/route.ts",
    "src/app/api/assistant/run-secrets/envelope/route.ts",
    "src/app/api/assistant/run-secrets/intent/route.ts",
    "src/app/api/assistant/runs/[runId]/events/route.ts",
    "src/app/api/assistant/tasks/[[...path]]/route.ts",
    "src/app/api/assistant/uploads/[uploadId]/chunks/[index]/route.ts",
    "src/app/api/assistant/uploads/[uploadId]/complete/route.ts",
    "src/app/api/assistant/uploads/route.ts",
    "src/app/api/auth/login/route.ts",
    "src/app/api/auth/logout/route.ts",
    "src/app/api/super-agent/status/route.ts",
    "src/components/workstation/SuperAgentV1Panel.tsx",
    "src/components/workstation/TenantHpcEnrollment.tsx",
    "src/components/workstation/deepevo-ui-contract.test.ts",
    "src/components/workstation/screens/AssistantControlledCredentials.tsx",
    "src/components/workstation/screens/assistant-activity.d.mts",
    "src/components/workstation/screens/assistant-activity.mjs",
    "src/components/workstation/screens/assistant-activity.test.ts",
    "src/components/workstation/screens/assistant-presentation.test.mjs",
    "src/components/workstation/screens/controlled-secret-binding.test.ts",
    "src/components/workstation/screens/controlled-secret-binding.ts",
    "src/components/workstation/screens/data-kaggle-csrf.test.ts",
    "src/components/workstation/screens/run-submission.test.mjs",
    "src/components/workstation/status-truthfulness.test.mjs",
    "src/lib/server/account-registry.d.mts",
    "src/lib/server/account-registry.mjs",
    "src/lib/server/assistant-managed-hpc.ts",
    "src/lib/server/assistant-run-secrets.test.ts",
    "src/lib/server/assistant-run-secrets.ts",
    "src/lib/server/managed-connector-projection.test.ts",
    "src/lib/server/managed-connector-projection.ts",
    "src/lib/server/product-account-api.test.mjs",
    "src/lib/server/public-auth.test.ts",
    "src/lib/server/public-auth.ts",
    "src/lib/server/runtime-principal.test.mjs",
    "src/lib/server/tenant-byoa-lifecycle.test.ts",
    "src/lib/server/tenant-byoa.test.ts",
    "src/lib/server/tenant-byoa.ts",
    "src/lib/server/tenant-hpc-profile-lifecycle.test.ts",
    "src/lib/server/tenant-hpc-profile-lifecycle.ts",
    "src/lib/server/tenant-identity.d.mts",
    "src/lib/server/tenant-identity.mjs",
    "src/lib/server/user-preferences.test.mjs",
    "src/lib/server/user-preferences.ts",
    "src/lib/super-agent-v1.test.ts",
    "src/lib/super-agent-v1.ts",
    "src/lib/system-debug-ui.test.mjs",
    "src/lib/task-context.test.ts",
    "src/proxy.test.ts",
    "scripts/run-node-tests.mjs",
    "src/app/api/artifacts/route.ts",
    "src/app/api/assistant/stream/route.ts",
    "src/app/api/healthz/route.ts",
    "src/app/api/settings/route.ts",
    "src/components/workstation/AiControlConsole.tsx",
    "src/components/workstation/OverviewBoardEnhanced.tsx",
    "src/components/workstation/Screens.tsx",
    "src/components/workstation/UserResearchJourney.tsx",
    "src/components/workstation/layout/RunContextBar.tsx",
    "src/components/workstation/localization/index.ts",
    "src/components/workstation/screens/EvidenceLedgerScreen.tsx",
    "src/components/workstation/screens/GatesScreen.test.ts",
    "src/components/workstation/screens/GatesScreen.tsx",
    "src/components/workstation/screens/GpuHpcScreen.tsx",
    "src/components/workstation/screens/ReportStudioScreen.tsx",
    "src/components/workstation/screens/RuntimeScreen.tsx",
    "src/components/workstation/screens/WorkflowScreen.tsx",
    "src/lib/security/request-boundary.test.ts",
    "src/lib/security/request-boundary.ts",
    "src/lib/server/runtime-version-contract.test.ts",
    "src/lib/server/runtime-version-contract.ts",
    "src/lib/server/runtime-version.ts",
    "src/lib/task-context.ts",
)
RUNTIME_PATCHES = (
    "runtime.py",
    "store.py",
    "assistant_runs.py",
    "assistant_history.py",
    "http_server.py",
    "tools.py",
    "policy.py",
    "hpc_runtime_overlay.py",
    "execution_progress.py",
    "managed_model_download.py",
    "dependency_lock.py",
    "research_knowledge.py",
    "model_evidence.py",
    "aibuild_engine.py",
    "research_budget.py",
    "budget_web.py",
    "managed_scheduler.py",
    "tenant_access.py",
    "managed_cancel.py",
    "run_requests.py",
    "user_projects.py",
    "training_control.py",
    "managed_tensor_train.py",
    "tensor_inference.py",
    "ev_calibration_control.py",
    "ev_calibration_diagnostics.py",
    "message_journal.py",
    "model_descriptor_store.py",
    "report_document.py",
    "report_figures.py",
    "report_jobs.py",
    "report_render.py",
    "responses_transport.py",
    "run_control.py",
    "model_transport.py",
    "model_deadline_worker.py",
    "chat_stream_transport.py",
    "siim_calibration_budget.py",
    "siim_calibration_control.py",
    "siim_calibration_web.py",
    "siim_candidate_contract.py",
    "siim_confirmed_recovery.py",
    "siim_hpc_runtime.py",
    "siim_recovery_windows.py",
    "siim_request_limits.py",
    "agent_kernel_v2.py",
    "aibuild_model_recovery.py",
    "capabilities.py",
    "competition_data.py",
    "competition_goal.py",
    "connectors.py",
    "credential_leases.py",
    "directory_broker.py",
    "ecosystem.py",
    "ev_calibration_runner.py",
    "evidence_memory.py",
    "goal_board.py",
    "goal_release_scope.py",
    "model_profile_secrets.py",
    "model_profiles.py",
    "personal_model_client.py",
    "personal_model_http.py",
    "personal_tool_boundary.py",
    "recovery.py",
    "remote_connectors.py",
    "run_secrets.py",
    "siim_calibration_runner.py",
    "siim_dataloader_runtime.py",
    "siim_worker_isolation.py",
    "super_agent_runtime.py",
    "super_agent_store.py",
    "super_agent_tools.py",
    "tool_package_runtime.py",
    "user_files.py",
    "user_tasks.py",
)
MAX_SOURCE_BYTES = 512 * 1024 * 1024
TEST_FIXTURES = ("canonical_json_f64_v1.json",)
# Base commit of the hash-bound live baseline. Deploy scripts bind
# runtime-build-manifest.commit_hash to it, so it is NOT the source of the
# overlaid files; that is recorded separately as ``source_commit``.
BASELINE_COMMIT = "664a636ddd419a66f73cc10820c0a16429784866"
WEB_SOURCE_PREFIX = "web/research-agent-workstation/"
RUNTIME_SOURCE_PREFIX = "src/evomind_runtime/"
FIXTURE_SOURCE_PREFIX = "tests/fixtures/"


def _git(root: Path, *argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *argv], capture_output=True, timeout=120)


def resolve_commit(root: Path, ref: str) -> str:
    if not ref or ref.startswith("-") or any(ch.isspace() for ch in ref):
        raise ValueError("source_git_ref_invalid")
    done = _git(root, "rev-parse", "--verify", "--quiet", ref + "^{commit}")
    commit = done.stdout.decode("ascii", "replace").strip()
    if done.returncode or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("source_git_ref_unresolved")
    return commit


def export_git_source(root: Path, ref: str, paths: list[str], destination: Path) -> dict:
    """Materialise ``paths`` exactly as committed at ``ref`` (never the worktree).

    ``git cat-file --filters`` applies the same smudge/eol filters a checkout
    would, so the bytes match a clean checkout of that commit on this machine.
    """
    commit = resolve_commit(root, ref)
    wanted = sorted({safe_name(path) for path in paths})
    listing = _git(root, "ls-tree", "-r", "-z", "--name-only", commit, "--", *wanted)
    if listing.returncode:
        raise ValueError("source_git_ref_listing_failed")
    present = {item.decode("utf-8") for item in listing.stdout.split(b"\0") if item}
    missing = [path for path in wanted if path not in present]
    if missing:
        raise ValueError("source_git_ref_missing_path:" + missing[0])
    destination.mkdir(parents=True, exist_ok=False)
    for relative in wanted:
        blob = _git(root, "cat-file", "--filters", f"{commit}:{relative}")
        if blob.returncode:
            raise ValueError("source_git_ref_read_failed:" + relative)
        target = destination.joinpath(*PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(blob.stdout)
    return {"source_mode": "git_ref", "source_ref": ref, "source_commit": commit, "source_commit_dirty": False}


def worktree_source_provenance(root: Path, paths: list[str]) -> dict:
    """Legacy worktree staging: record HEAD and whether the used files differ from it."""
    try:
        commit = resolve_commit(root, "HEAD")
    except (ValueError, OSError, subprocess.SubprocessError):
        return {"source_mode": "worktree", "source_commit": None, "source_commit_dirty": True}
    status = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--", *sorted(set(paths)))
    dirty = bool(status.returncode or status.stdout.strip(b"\0"))
    return {"source_mode": "worktree", "source_commit": commit, "source_commit_dirty": dirty}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_file(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def safe_name(name: str) -> str:
    path = PurePosixPath(name)
    if (not name or "\\" in name or path.is_absolute() or ".." in path.parts
            or any(":" in part for part in path.parts) or path.as_posix() != name):
        raise ValueError("candidate_archive_path_rejected")
    return name


def unpack_baseline(archive: Path, expected_sha: str, destination: Path, expected_build: str) -> dict:
    if sha(archive) != expected_sha:
        raise ValueError("baseline_archive_hash_mismatch")
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        names = [safe_name(entry.filename) for entry in entries]
        if (len(names) != len(set(name.casefold() for name in names))
                or sum(entry.file_size for entry in entries) > MAX_SOURCE_BYTES
                or any(((entry.external_attr >> 16) & 0xF000) == 0xA000 for entry in entries)):
            raise ValueError("baseline_archive_contract_rejected")
        manifest = json.loads(bundle.read("baseline-manifest.json"))
        if manifest.get("schema") != "evomind.invitation_baseline.v1" or manifest.get("web_build_id") != expected_build:
            raise ValueError("baseline_identity_mismatch")
        declared = {row["path"]: row for row in manifest["files"]}
        if len(declared) != manifest["file_count"] or set(names) != {*declared, "baseline-manifest.json"}:
            raise ValueError("baseline_file_set_mismatch")
        for name, row in declared.items():
            payload = bundle.read(name)
            if len(payload) != row["bytes"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
                raise ValueError("baseline_file_hash_mismatch")
            path = destination.joinpath(*PurePosixPath(name).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(payload)
    return manifest


def files(root: Path, prefix: str = "") -> list[dict]:
    result = []
    for directory, subdirs, names in os.walk(root, followlinks=False):
        for name in subdirs:
            if (Path(directory) / name).is_symlink():
                raise ValueError("candidate_directory_symlink")
        subdirs[:] = [name for name in subdirs if name != "__pycache__"]
        for name in names:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError("candidate_file_symlink")
            if name.endswith((".pyc", ".pyo")):
                continue
            relative = prefix + path.relative_to(root).as_posix()
            result.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha(path)})
    return sorted(result, key=lambda row: row["path"])


def tree_sha(rows: list[dict]) -> str:
    return hashlib.sha256("\n".join(f"{row['path']}|{row['bytes']}|{row['sha256']}" for row in sorted(rows, key=lambda item: item["path"])).encode()).hexdigest()


def refresh_build_inputs(work: Path, inputs: list[dict], changes: list[dict]) -> list[dict]:
    changed = {row["path"]: row for row in changes}
    result = []
    for row in inputs:
        path = work / safe_name(row["path"])
        change = changed.get(row["path"])
        expected = row
        if change is not None:
            if change.get("before_sha256") != row["sha256"]:
                raise ValueError("candidate_build_input_parent_mismatch:" + row["path"])
            expected = change
        if sha(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError("candidate_build_input_changed:" + row["path"])
        result.append({**row, "sha256": expected["sha256"], "bytes": expected["bytes"]})
    return result


def canonicalize_dependency_registry(path: Path, before_sha256: str) -> dict | None:
    """Keep staged lockfiles on the official registry without changing integrity pins."""
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    packages = value.get("packages")
    if not isinstance(packages, dict):
        raise ValueError("candidate_dependency_lock_invalid")
    changed = False
    for package in packages.values():
        if not isinstance(package, dict):
            continue
        resolved = package.get("resolved")
        if isinstance(resolved, str) and resolved.startswith("https://registry.npmmirror.com/"):
            package["resolved"] = "https://registry.npmjs.org/" + resolved.split("/", 3)[3]
            changed = True
    if not changed:
        return None
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "path": "web/package-lock.json",
        "before_sha256": before_sha256,
        "sha256": sha(path),
        "bytes": path.stat().st_size,
    }


def stage(args) -> Path:
    work_root = Path(args.work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    work = work_root / ("invitation-" + uuid.uuid4().hex[:12])
    work.mkdir(exist_ok=False)
    baseline = unpack_baseline(Path(args.baseline), args.baseline_sha256, work, args.expected_web_build)
    web = work / "web"
    upstream = json.loads((web / "release-source-manifest.json").read_text(encoding="utf-8-sig"))
    source_paths = ([WEB_SOURCE_PREFIX + name for name in WEB_PATCHES]
                    + [RUNTIME_SOURCE_PREFIX + name for name in (*RUNTIME_PATCHES, "aibuild_model_recovery.py")]
                    + [FIXTURE_SOURCE_PREFIX + name for name in TEST_FIXTURES]
                    + [WEB_SOURCE_PREFIX + safe_name(row["path"]) for row in upstream["build_inputs"]
                       if safe_name(row["path"]) not in WEB_PATCHES])
    if getattr(args, "source_git_ref", None):
        # Clean mode: every overlaid file comes from one committed tree.
        source_root = work / "_git_source"
        provenance = export_git_source(ROOT, args.source_git_ref, source_paths, source_root)
    else:
        # Legacy mode (unchanged inputs): the working tree, with honest provenance.
        source_root = ROOT
        provenance = worktree_source_provenance(ROOT, source_paths)
    web_source = WEB if source_root is ROOT else source_root / "web" / "research-agent-workstation"
    build_inputs = []
    for row in upstream["build_inputs"]:
        name = safe_name(row["path"])
        if "/" in name:
            raise ValueError("unexpected_baseline_build_input")
        target = web / name
        source = target if name in WEB_PATCHES and target.is_file() else web_source / name
        if sha(source) != row["sha256"]:
            raise ValueError("build_input_not_baseline_bound:" + name)
        if source != target:
            shutil.copyfile(source, target)
        build_inputs.append({"path": "web/" + name, "sha256": row["sha256"], "bytes": source.stat().st_size})
    if not any(row["path"] == "web/package.json" for row in build_inputs):
        package = web / "package.json"
        build_inputs.append({"path": "web/package.json", "sha256": sha(package), "bytes": package.stat().st_size})
    lockfile = web / "package-lock.json"
    lock_row = next((row for row in build_inputs if row["path"] == "web/package-lock.json"), None)
    if lock_row is not None:
        lock_change = canonicalize_dependency_registry(lockfile, lock_row["sha256"])
    else:
        lock_change = None
    schema = Path(args.production_schema)
    if sha(schema) != args.production_schema_sha256:
        raise ValueError("production_schema_hash_mismatch")
    (web / "prisma").mkdir()
    shutil.copyfile(schema, web / "prisma/schema.prisma")
    build_inputs.append({"path": "web/prisma/schema.prisma", "sha256": sha(schema), "bytes": schema.stat().st_size})
    changes = [lock_change] if lock_change is not None else []
    for name in WEB_PATCHES:
        source, target = web_source / name, web / name
        target.parent.mkdir(parents=True, exist_ok=True)
        before = sha(target) if target.exists() else None
        shutil.copyfile(source, target)
        changes.append({"path": "web/" + name, "before_sha256": before, "sha256": sha(target), "bytes": target.stat().st_size})
    build_inputs = refresh_build_inputs(work, build_inputs, changes)
    for name in TEST_FIXTURES:
        target = web / "test-fixtures" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        before = sha(target) if target.exists() else None
        shutil.copyfile(source_root / "tests/fixtures" / name, target)
        changes.append({"path": "web/test-fixtures/" + name, "before_sha256": before, "sha256": sha(target), "bytes": target.stat().st_size})
    runtime = work / "runtime/evomind_runtime"
    for name in (*RUNTIME_PATCHES, "aibuild_model_recovery.py"):
        source, target = source_root / "src/evomind_runtime" / name, runtime / name
        before = sha(target) if target.exists() else None
        shutil.copyfile(source, target)
        changes.append({"path": "runtime/evomind_runtime/" + name, "before_sha256": before, "sha256": sha(target), "bytes": target.stat().st_size})
    mirror = web / "support/python-runtime/evomind_runtime"
    if {path.name for path in mirror.glob("*.py")} - {path.name for path in runtime.glob("*.py")}:
        raise ValueError("runtime_mirror_unexpected_files")
    for source in runtime.glob("*.py"):
        shutil.copyfile(source, mirror / source.name)
    changed_names = {row["path"] for row in changes}
    for row in baseline["files"]:
        if row["path"] in changed_names or row["path"].startswith("web/support/python-runtime/evomind_runtime/"):
            continue
        if sha(work / row["path"]) != row["sha256"]:
            raise ValueError("unrelated_baseline_file_changed")
    source_rows = files(web / "src", "web/src/") + files(web / "public", "web/public/") + files(web / "support", "web/support/") + files(web / "scripts", "web/scripts/") + files(web / "test-fixtures", "web/test-fixtures/") + files(runtime, "runtime/evomind_runtime/") + build_inputs
    source_identity = tree_sha(source_rows)
    manifest = {
        "schema": "evomind.invitation_source.v1", "frozen": True,
        "baseline_archive_sha256": args.baseline_sha256, "baseline_web_build_id": args.expected_web_build,
        "build_id": "overlay-invitation-beta-" + source_identity[:12],
        "source_tree_sha256": source_identity, "changes": changes, "files": source_rows,
        "work_root": str(work), "runtime_file_count": len(list(runtime.glob("*.py"))),
        "production_schema_sha256": args.production_schema_sha256,
        "production_deployed": False, "research_ablation_executed": False,
        **provenance,
    }
    receipt = work / "source-receipt.json"
    json_file(receipt, manifest)
    print(json.dumps({"phase": "source_staged", "source_receipt": str(receipt), "build_id": manifest["build_id"], "runtime_file_count": manifest["runtime_file_count"]}), flush=True)
    return receipt


def command_environment(web: Path) -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items()
                   if not re.search(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTHORIZATION", key, re.I)
                   and not key.upper().startswith(("OPENAI_", "ANTHROPIC_", "DEEPSEEK_", "PRISMA_", "NPM_CONFIG_"))}
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "NODE_OPTIONS"):
        environment.pop(name, None)
    fixture = web.parent / "fixture"
    fixture.mkdir(exist_ok=True)
    for name in ("npm-user.conf", "npm-global.conf"):
        (fixture / name).write_text("")
    environment.update({
        "NPM_CONFIG_USERCONFIG": str(fixture / "npm-user.conf"),
        "NPM_CONFIG_GLOBALCONFIG": str(fixture / "npm-global.conf"),
        "NPM_CONFIG_CACHE": str(fixture / "npm-cache"),
        "EVOMIND_TEST_FIXTURE_ROOT": str(web / "test-fixtures"),
        "NEXT_TELEMETRY_DISABLED": "1", "NO_PROXY": "*",
        "DATABASE_URL": "file:" + (fixture / "build.sqlite").as_posix(),
        "WORKSTATION_ROOT": str(fixture), "WORKSTATION_DATA_DIR": str(fixture),
        "WORKSTATION_SESSION_SECRET": secrets.token_urlsafe(48),
        "PYTHONIOENCODING": "utf-8", "PRISMA_HIDE_UPDATE_MESSAGE": "1",
    })
    return environment


def run_command(argv: list[str], cwd: Path, environment: dict[str, str], logs: Path, label: str, timeout: int) -> dict:
    print(json.dumps({"phase": label, "status": "running"}), flush=True)
    started = time.monotonic()
    process = subprocess.run(argv, cwd=cwd, env=environment, capture_output=True, timeout=timeout)
    text = (process.stdout + b"\n" + process.stderr).decode("utf-8", "replace")
    lines = ["[SENSITIVE DIAGNOSTIC LINE WITHHELD]" if re.search(r"api.?key|auth.?token|password|credential|environ\(|口令|凭据|密码", line, re.I) else line for line in text.splitlines()]
    logs.mkdir(exist_ok=True)
    log = logs / (label + ".log")
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {"command": label, "exit_code": process.returncode, "elapsed_seconds": round(time.monotonic() - started, 3), "log_sha256": sha(log)}
    print(json.dumps(result), flush=True)
    if process.returncode:
        raise RuntimeError("candidate_build_command_failed:" + label)
    return result


def copy_release_tree(source: Path, destination: Path) -> None:
    def excluded(_directory, names):
        return [name for name in names if name.startswith(".env") or name in {"__pycache__", "runtime.token"}
                or name.endswith((".pyc", ".pyo", ".sqlite", ".sqlite3", ".db"))]
    shutil.copytree(source, destination, dirs_exist_ok=True, ignore=excluded)


def zip_tree(root: Path, destination: Path) -> dict:
    rows = files(root)
    if len(rows) > 40000 or sum(row["bytes"] for row in rows) > 3 * 1024 ** 3:
        raise ValueError("candidate_release_size_limit")
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for row in rows:
            info = zipfile.ZipInfo(row["path"], date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, (root / row["path"]).read_bytes())
    return {"path": str(destination), "bytes": destination.stat().st_size, "sha256": sha(destination), "file_count": len(rows)}


def unpack_release(archive: Path, expected_sha: str, destination: Path, manifest_name: str) -> dict:
    if sha(archive) != expected_sha:
        raise ValueError("candidate_parent_archive_hash_mismatch")
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        names = [safe_name(entry.filename) for entry in entries]
        if (len(names) > 40000 or len(names) != len({name.casefold() for name in names})
                or sum(entry.file_size for entry in entries) > 3 * 1024 ** 3
                or any(entry.is_dir() or ((entry.external_attr >> 16) & 0xF000) == 0xA000 for entry in entries)):
            raise ValueError("candidate_parent_archive_contract_rejected")
        manifest = json.loads(bundle.read(manifest_name))
        rows = manifest["files"]
        declared = {safe_name(row["path"]): row for row in rows}
        if (len(declared) != len(rows) or len(rows) != manifest["file_count"]
                or set(names) != {*declared, manifest_name}):
            raise ValueError("candidate_parent_file_set_mismatch")
        destination.mkdir(parents=True, exist_ok=False)
        for name in names:
            payload = bundle.read(name)
            if name != manifest_name:
                row = declared[name]
                if hashlib.sha256(payload).hexdigest() != row["sha256"] or len(payload) != row["bytes"]:
                    raise ValueError("candidate_parent_file_hash_mismatch")
            target = destination.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(payload)
    return manifest


def database_schema_sha256(database: Path) -> str:
    """Match runtime-version.ts: hash normalized sqlite_master, not Prisma bytes."""
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    try:
        rows = connection.execute("SELECT type,name,tbl_name,COALESCE(sql,'') FROM sqlite_master WHERE type IN ('table','index','view','trigger') AND name NOT LIKE 'sqlite_%' ORDER BY type,name,tbl_name")
        digest = hashlib.sha256()
        for kind, name, table, sql in rows:
            digest.update(("\0".join((kind, name, table, re.sub(r"\s+", " ", sql.strip()))) + "\n").encode("utf-8"))
        return digest.hexdigest()
    finally:
        connection.close()


def validate_release_identity(release: Path, runtime: Path, source: dict) -> dict:
    operational = json.loads((release / "operational-overlay-manifest.json").read_text(encoding="utf-8-sig"))
    identity = json.loads((release / "runtime-build-manifest.json").read_text(encoding="utf-8-sig"))
    published = json.loads((release / "release-source-manifest.json").read_text(encoding="utf-8-sig"))
    expected_source = source["source_tree_sha256"]
    expected_schema = source["database_schema_sha256"]
    expected_schema_source = source["production_schema_sha256"]
    if (not re.fullmatch(r"[a-f0-9]{64}", expected_source)
            or tree_sha(source["files"]) != expected_source
            or not re.fullmatch(r"[a-f0-9]{64}", expected_schema)
            or not re.fullmatch(r"[a-f0-9]{64}", expected_schema_source)
            or published != {key: value for key, value in source.items() if key != "work_root"}
            or operational.get("schema") != "evomind.web_operational_overlay.v1"
            or operational.get("source_dirty") is not True or operational.get("entrypoint") != "server.js"
            or operational.get("source_identity_sha256") != expected_source
            or operational.get("source_tree_sha256") != expected_source
            or operational.get("overlay_id") != source["build_id"]
            or identity.get("schema") != "evomind.runtime_build.v1"
            or identity.get("source_dirty") is not True
            or identity.get("source_tree_sha256") != expected_source or identity.get("build_id") != source["build_id"]
            or identity.get("commit_hash") != operational.get("base_commit")
            or identity.get("commit_hash") != BASELINE_COMMIT
            or (source.get("source_commit") is not None
                and (identity.get("source_commit") != source["source_commit"]
                     or identity.get("source_commit_dirty") is not bool(source.get("source_commit_dirty", True))
                     or operational.get("source_commit") != source["source_commit"]))
            or identity.get("backend_version") != "0.3.0" or identity.get("frontend_version") != "0.3.0"
            or identity.get("database_schema_sha256") != expected_schema
            or not identity.get("database_schema_version")):
        raise ValueError("candidate_release_identity_mismatch")
    schema_path = release / "node_modules/.prisma/client/schema.prisma"
    if not schema_path.is_file() or sha(schema_path) != expected_schema_source:
        raise ValueError("candidate_generated_schema_mismatch")
    if not (release / "server.js").is_file():
        raise ValueError("candidate_entrypoint_missing")
    for folder, name in ((release, "operational-overlay-manifest.json"), (runtime, "runtime-hotfix-manifest.json")):
        manifest = json.loads((folder / name).read_text(encoding="utf-8-sig"))
        actual = [row for row in files(folder) if row["path"] != name]
        declared = manifest["files"]
        if manifest["file_count"] != len(declared) or actual != sorted(declared, key=lambda row: row["path"]):
            raise ValueError("candidate_release_file_set_or_hash_mismatch")
    runtime_manifest = json.loads((runtime / "runtime-hotfix-manifest.json").read_text())
    if (runtime_manifest.get("release") != source["build_id"]
            or runtime_manifest.get("source_manifest_sha256") != sha(release / "release-source-manifest.json")
            or files(runtime / "evomind_runtime") != files(release / "support/python-runtime/evomind_runtime")):
        raise ValueError("candidate_runtime_source_binding_mismatch")
    return {"status": "passed", "build_id": source["build_id"], "source_identity_sha256": expected_source,
            "database_schema_sha256": expected_schema, "prisma_schema_source_sha256": expected_schema_source,
            "production_startup_preflight": "pending"}


def publish_release_manifests(release: Path, runtime: Path, source: dict, runtime_template: dict, operational_template: dict) -> dict:
    identity = dict(runtime_template)
    identity.update(build_id=source["build_id"], source_tree_sha256=source["source_tree_sha256"], source_dirty=True,
                    database_schema_sha256=source["database_schema_sha256"], build_time=datetime.now(timezone.utc).isoformat())
    provenance = {}
    if source.get("source_commit") is not None:
        provenance = {"source_commit": source["source_commit"],
                      "source_commit_dirty": bool(source.get("source_commit_dirty", True))}
    identity.update(provenance)
    json_file(release / "runtime-build-manifest.json", identity)
    json_file(release / "release-source-manifest.json", {key: value for key, value in source.items() if key != "work_root"})
    operational = dict(operational_template)
    operational.update(overlay_id=source["build_id"], source_dirty=True,
                       source_identity_sha256=source["source_tree_sha256"], source_tree_sha256=source["source_tree_sha256"],
                       changed_source_files=[row for row in source["changes"] if row["path"].startswith("web/")],
                       **provenance)
    operational["files"] = [row for row in files(release) if row["path"] != "operational-overlay-manifest.json"]
    operational["file_count"] = len(operational["files"])
    json_file(release / "operational-overlay-manifest.json", operational)
    runtime_rows = [row for row in files(runtime) if row["path"] != "runtime-hotfix-manifest.json"]
    json_file(runtime / "runtime-hotfix-manifest.json", {
        "schema": "evomind.super_agent_runtime_hotfix.v1", "target": "bundle/runtime/evomind_runtime",
        "release": source["build_id"], "source_manifest_sha256": sha(release / "release-source-manifest.json"),
        "files": runtime_rows, "file_count": len(runtime_rows),
    })
    return validate_release_identity(release, runtime, source)


def repair_manifests(args) -> dict:
    parent = Path(args.candidate_root).resolve()
    parent_result = json.loads((parent / "build-result.json").read_text())
    receipt = parent / "source-receipt.json"
    source = json.loads(receipt.read_text())
    if (parent_result.get("status") != "built" or sha(receipt) != parent_result.get("source_receipt_sha256")
            or parent_result.get("source_manifest_sha256") != args.source_manifest_sha256
            or parent_result.get("build_id") != source.get("build_id")
            or parent_result.get("source_tree_sha256") != source.get("source_tree_sha256")
            or source.get("frozen") is not True or tree_sha(source["files"]) != source["source_tree_sha256"]
            or parent_result["web"]["sha256"] != args.web_sha256
            or parent_result["runtime"]["sha256"] != args.runtime_sha256):
        raise ValueError("candidate_parent_receipt_mismatch")
    work = Path(args.work_root).resolve() / ("invitation-manifests-" + uuid.uuid4().hex[:12])
    work.mkdir(parents=True, exist_ok=False)
    release, runtime = work / "release", work / "runtime"
    operational = unpack_release(parent / "web.zip", args.web_sha256, release, "operational-overlay-manifest.json")
    unpack_release(parent / "runtime.zip", args.runtime_sha256, runtime, "runtime-hotfix-manifest.json")
    if (sha(release / "release-source-manifest.json") != args.source_manifest_sha256
            or json.loads((release / "release-source-manifest.json").read_text())
            != {key: value for key, value in source.items() if key != "work_root"}):
        raise ValueError("candidate_parent_source_manifest_mismatch")
    metadata = {"operational-overlay-manifest.json", "runtime-build-manifest.json", "release-source-manifest.json"}
    before = [row for row in files(release) if row["path"] not in metadata]
    before_runtime = [row for row in files(runtime) if row["path"] != "runtime-hotfix-manifest.json"]
    identity = json.loads((release / "runtime-build-manifest.json").read_text())
    database_identity = database_schema_sha256(Path(args.schema_database))
    if database_identity != args.database_schema_sha256 or identity.get("database_schema_sha256") != database_identity:
        raise ValueError("candidate_parent_database_schema_mismatch")
    recipe = sha(Path(__file__))
    source.update(build_id="overlay-invitation-beta-" + source["source_tree_sha256"][:12] + "-m" + recipe[:8],
                  work_root=str(work), artifact_only=True, database_schema_sha256=database_identity,
                  derivation={"operation": "manifest_only", "parent_web_sha256": args.web_sha256,
                              "parent_runtime_sha256": args.runtime_sha256,
                              "parent_source_manifest_sha256": args.source_manifest_sha256,
                              "parent_build_receipt_sha256": sha(parent / "build-result.json"), "recipe_sha256": recipe})
    json_file(work / "source-receipt.json", source)
    validation = publish_release_manifests(release, runtime, source, identity, operational)
    if (before != [row for row in files(release) if row["path"] not in metadata]
            or before_runtime != [row for row in files(runtime) if row["path"] != "runtime-hotfix-manifest.json"]):
        raise ValueError("manifest_repair_changed_payload")
    result = {
        "schema": "evomind.invitation_candidate_build.v1", "status": "built", "build_id": source["build_id"],
        "baseline_web_build_id": source["baseline_web_build_id"], "source_tree_sha256": source["source_tree_sha256"],
        "source_receipt_sha256": sha(work / "source-receipt.json"),
        "source_manifest_sha256": sha(release / "release-source-manifest.json"),
        "web": zip_tree(release, work / "web.zip"), "runtime": zip_tree(runtime, work / "runtime.zip"),
        "commands": [], "inherited_build_commands": parent_result.get("commands", []), "derivation": source["derivation"],
        "payload_unchanged": True, "release_identity_validation": validation, "production_deployed": False,
        "isolated_runtime_acceptance": "pending", "browser_e2e": "pending", "user_credentials_included": False,
        "production_database_migrated": False,
    }
    json_file(work / "build-result.json", result)
    print(json.dumps(result), flush=True)
    return result


REQUIRED_BUILD_INPUTS = ("package.json", "package-lock.json", "next.config.mjs", "postcss.config.mjs", "tailwind.config.ts", "tsconfig.json", "prisma/schema.prisma")


def validate_build_inputs(work: Path, source: dict) -> None:
    declared = {row["path"]: row for row in source["files"]}
    for name in REQUIRED_BUILD_INPUTS:
        relative = "web/" + name
        path = work / relative
        row = declared.get(relative)
        if row is None or not path.is_file():
            raise ValueError("candidate_required_build_input_missing:" + relative)
        if sha(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError("candidate_required_build_input_changed:" + relative)


def build(receipt: Path) -> dict:
    source = json.loads(receipt.read_text())
    if source.get("artifact_only"):
        raise ValueError("derived_artifact_requires_fresh_source_stage")
    work = Path(source["work_root"])
    validate_build_inputs(work, source)
    web = work / "web"
    lock = json.loads((web / "package-lock.json").read_text())
    for name, row in lock["packages"].items():
        if not name:
            continue
        if not str(row.get("resolved", "")).startswith("https://registry.npmjs.org/") or not str(row.get("integrity", "")).startswith("sha512-"):
            raise ValueError("candidate_dependency_not_official_hash_locked")
    npm = Path(shutil.which("npm.cmd") or shutil.which("npm") or "")
    npm_cli = npm.parent / "node_modules/npm/bin/npm-cli.js"
    node = shutil.which("node")
    if not node or not npm_cli.is_file():
        raise RuntimeError("candidate_node_toolchain_missing")
    environment = command_environment(web)
    logs = work / "logs"
    commands = []
    commands.append(run_command([node, str(npm_cli), "ci", "--ignore-scripts", "--no-audit", "--no-fund", "--registry=https://registry.npmjs.org"], web, environment, logs, "npm-ci", 360))
    prisma = str(web / "node_modules/prisma/build/index.js")
    commands.append(run_command([node, prisma, "generate", "--schema", "prisma/schema.prisma"], web, environment, logs, "prisma-generate", 180))
    commands.append(run_command([node, prisma, "db", "push", "--skip-generate", "--schema", "prisma/schema.prisma"], web, environment, logs, "fixture-schema", 120))
    commands.append(run_command([node, "--test", "--experimental-strip-types", "--disable-warning=MODULE_TYPELESS_PACKAGE_JSON", "src/**/*.test.ts", "src/**/*.test.mjs"], web, environment, logs, "frontend-tests", 180))
    commands.append(run_command([node, str(web / "node_modules/typescript/bin/tsc"), "--noEmit", "--incremental", "false"], web, environment, logs, "typecheck", 180))
    commands.append(run_command([node, str(web / "node_modules/next/dist/bin/next"), "build", "--webpack"], web, environment, logs, "next-build", 360))
    for row in source["files"]:
        if sha(work / row["path"]) != row["sha256"]:
            raise ValueError("candidate_source_changed_during_build:" + row["path"])
    release = work / "release"
    copy_release_tree(web / ".next/standalone", release)
    if not (release / "server.js").is_file() or not (release / "node_modules/next/package.json").is_file():
        raise ValueError("candidate_standalone_layout_rejected")
    for name in ("src", "public", "support", "scripts", "test-fixtures"):
        copy_release_tree(web / name, release / name)
    copy_release_tree(web / ".next/static", release / ".next/static")
    runtime_manifest = json.loads((web / "runtime-build-manifest.json").read_text(encoding="utf-8-sig"))
    source["database_schema_sha256"] = database_schema_sha256(work / "fixture/build.sqlite")
    json_file(receipt, source)
    operational = json.loads((web / "operational-overlay-manifest.json").read_text(encoding="utf-8-sig"))
    runtime = work / "runtime"
    validation = publish_release_manifests(release, runtime, source, runtime_manifest, operational)
    result = {
        "schema": "evomind.invitation_candidate_build.v1", "status": "built", "build_id": source["build_id"],
        "baseline_web_build_id": source["baseline_web_build_id"], "source_tree_sha256": source["source_tree_sha256"],
        "source_receipt_sha256": sha(receipt), "source_manifest_sha256": sha(release / "release-source-manifest.json"),
        "web": zip_tree(release, work / "web.zip"), "runtime": zip_tree(runtime, work / "runtime.zip"),
        "commands": commands, "release_identity_validation": validation, "production_deployed": False, "isolated_runtime_acceptance": "pending", "browser_e2e": "pending",
        "production_database_migrated": False, "user_credentials_included": False,
    }
    json_file(work / "build-result.json", result)
    print(json.dumps(result), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    staging = sub.add_parser("stage")
    staging.add_argument("--baseline", required=True)
    staging.add_argument("--baseline-sha256", required=True)
    staging.add_argument("--expected-web-build", required=True)
    staging.add_argument("--production-schema", required=True)
    staging.add_argument("--production-schema-sha256", required=True)
    staging.add_argument("--work-root", default="D:/EV12")
    staging.add_argument("--build", action="store_true")
    staging.add_argument("--source-git-ref", default=None,
                         help="stage overlaid files from this committed ref instead of the working tree")
    building = sub.add_parser("build")
    building.add_argument("--source-receipt", required=True)
    repair = sub.add_parser("repair-manifests")
    repair.add_argument("--candidate-root", required=True)
    repair.add_argument("--web-sha256", required=True)
    repair.add_argument("--runtime-sha256", required=True)
    repair.add_argument("--source-manifest-sha256", required=True)
    repair.add_argument("--schema-database", required=True)
    repair.add_argument("--database-schema-sha256", required=True)
    repair.add_argument("--work-root", default="D:/EV12")
    args = parser.parse_args()
    if args.mode == "stage":
        receipt = stage(args)
        if args.build:
            build(receipt)
    elif args.mode == "build":
        build(Path(args.source_receipt))
    else:
        repair_manifests(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "failed", "error_class": type(error).__name__, "code": str(error) if re.fullmatch(r"[A-Za-z0-9_:./-]{1,250}", str(error)) else "diagnostic_withheld", "production_deployed": False}), flush=True)
        raise SystemExit(1)
