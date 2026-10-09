"""Offline source-only frontend checks and exact-manifest release assembly.

No SSH, service/config mutation, model requests, GPU work, or deployment. The
runtime delta must be explicitly frozen elsewhere before this tool accepts it.
Report wheels are delivered as a checked wheelhouse; activating/import-verifying
them is a separate server gate, never silently inferred from a successful build.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location("syscomplete_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_script("syscomplete_source_freeze.py")
LEGACY = load_script("build_invitation_release.py")
RECIPE_BYTES = Path(__file__).read_bytes()
LEGACY_RECIPE_BYTES = (ROOT / "scripts/build_invitation_release.py").read_bytes()
FROZEN_SOURCE_SHA = "a5670ed851547dd22e05f280b5677b618d9504fccdeae4cf626458ab94f52f63"
RUNTIME_ALLOWLIST = frozenset({
    "aibuild_engine.py", "aibuild_model_recovery.py", "assistant_runs.py", "dependency_lock.py",
    "execution_progress.py", "http_server.py", "managed_scheduler.py", "message_journal.py",
    "model_transport.py", "policy.py", "report_document.py", "report_figures.py", "report_jobs.py",
    "report_render.py", "responses_transport.py", "run_control.py", "runtime.py", "store.py", "tools.py", "tenant_access.py",
})
REQUIRED_RUNTIME = RUNTIME_ALLOWLIST - {"dependency_lock.py", "tenant_access.py"}
# The bundled artifact Node runtime has no npm. This installed Node/npm pair
# was checked locally; either absolute path can be overridden explicitly.
DEFAULT_NODE_ROOT = Path("D:/下载")
REPORT_REQUIREMENTS = frozenset({"matplotlib", "numpy", "python-docx", "pymupdf", "lxml"})
WEB_CORRECTIONS = frozenset({"web/" + name for name in (*BASE.BACKUP_UI, *BASE.REVIEWED_DIRECT,
    "src/lib/server/report-assistant-bridge-contract.test.ts")})


def require_hash(value: str) -> str:
    if not re.fullmatch(r"[a-f0-9]{64}", str(value)):
        raise ValueError("expected_hash_invalid")
    return value


def checked_json(path: Path, expected: str) -> dict:
    payload = BASE.checked_path(path.parent, path.name).read_bytes()
    if BASE.digest(payload) != require_hash(expected):
        raise ValueError("manifest_hash_mismatch")
    return json.loads(payload.decode("utf-8-sig"))


def validate_rows(rows: list[dict], *, prefix: str | None = None) -> None:
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest_files_missing")
    names = []
    for row in rows:
        name = BASE.safe_relative(row["path"])
        if prefix and not name.startswith(prefix):
            raise ValueError("manifest_file_scope_rejected")
        if (type(row.get("bytes")) is not int or not 0 <= row["bytes"] <= 512 * 1024 * 1024
                or not re.fullmatch(r"[a-f0-9]{64}", str(row.get("sha256", "")))):
            raise ValueError("manifest_file_row_invalid")
        names.append(name.casefold())
    if len(names) != len(set(names)) or len(names) > 40000:
        raise ValueError("manifest_file_set_rejected")


def read_declared(root: Path, row: dict) -> bytes:
    payload = BASE.checked_path(root, row["path"]).read_bytes()
    if BASE.row_for(row["path"], payload) != {key: row[key] for key in ("path", "bytes", "sha256")}:
        raise ValueError("declared_file_changed:" + row["path"])
    return payload


def read_source(root: Path, expected: str) -> tuple[dict, dict[str, bytes]]:
    manifest = checked_json(root / "syscomplete-source-manifest.json", expected)
    if (manifest.get("schema") != "evomind.syscomplete_source_freeze.v1"
            or manifest.get("frozen") is not True or manifest.get("base_build_id") != BASE.BASE_BUILD
            or manifest.get("base_source_tree_sha256") != BASE.BASE_TREE
            or manifest.get("base_web_sha256") != BASE.BASE_WEB
            or manifest.get("base_runtime_sha256") != BASE.BASE_RUNTIME
            or manifest.get("dirty_tree_wholesale_copy") is not False
            or manifest.get("runtime_payload_changed") is not False):
        raise ValueError("source_freeze_identity_rejected")
    rows = manifest["files"]
    validate_rows(rows)
    if len(rows) != manifest["file_count"] or BASE.tree_sha(rows) != manifest["source_tree_sha256"]:
        raise ValueError("source_freeze_tree_mismatch")
    templates = manifest["templates"]
    validate_rows(templates, prefix="web/")
    if {row["path"] for row in templates} != {"web/" + name for name in BASE.TEMPLATES}:
        raise ValueError("source_freeze_templates_rejected")
    payloads = {row["path"]: read_declared(root, row) for row in rows + templates}
    actual = {row["path"] for row in LEGACY.files(root)}
    if actual != {*payloads, "syscomplete-source-manifest.json"}:
        raise ValueError("source_freeze_extra_or_missing_files")
    return manifest, payloads


def runtime_delta(path: Path, expected: str, root: Path, payloads: dict[str, bytes]) -> tuple[dict, list[dict]]:
    manifest = checked_json(path, expected)
    if (manifest.get("schema") != "evomind.syscomplete_runtime_delta.v1"
            or manifest.get("base_source_tree_sha256") != BASE.BASE_TREE
            or manifest.get("model") != "gpt-5.5" or manifest.get("frozen") is not True):
        raise ValueError("runtime_manifest_contract_rejected")
    rows = manifest["files"]
    validate_rows(rows, prefix="evomind_runtime/")
    names = {row["path"].removeprefix("evomind_runtime/") for row in rows}
    if not names.issubset(RUNTIME_ALLOWLIST) or not REQUIRED_RUNTIME.issubset(names):
        raise ValueError("runtime_allowlist_incomplete_or_expanded")
    changes = []
    staged = {}
    for row in rows:
        name = row["path"]
        before = payloads.get("runtime/" + name)
        if "before_sha256" not in row or row["before_sha256"] != (BASE.digest(before) if before is not None else None):
            raise ValueError("runtime_parent_file_mismatch:" + name)
        data = read_declared(root, row)
        staged[name] = data
        for target in ("runtime/" + name, "web/support/python-runtime/" + name):
            previous = payloads.get(target)
            if previous != before:
                raise ValueError("runtime_parent_mirror_mismatch")
            if previous != data:
                changes.append({**BASE.row_for(target, data),
                                "before_sha256": BASE.digest(previous) if previous is not None else None})
    # Both consumers must resolve scheduler types from the same owned package.
    for name in ("aibuild_engine.py", "aibuild_model_recovery.py"):
        text = staged["evomind_runtime/" + name].decode("utf-8")
        if "from .managed_scheduler import " not in text or "from research_os.agent.multi_agent import " in text:
            raise ValueError("managed_scheduler_import_contract_rejected")
    for name, data in staged.items():
        payloads["runtime/" + name] = data
        payloads["web/support/python-runtime/" + name] = data
    return manifest, changes


def amend_web(source_root: Path, source_sha: str, delta_path: Path, delta_sha: str,
              delta_root: Path, destination: Path) -> dict:
    """New source derivative for reviewed compile/test repairs; never edit v1."""
    frozen, payloads = read_source(source_root, source_sha)
    delta = checked_json(delta_path, delta_sha)
    if (set(delta) != {"schema", "frozen", "parent_manifest_sha256", "files"}
            or delta.get("schema") != "evomind.syscomplete_web_delta.v1"
            or delta.get("parent_manifest_sha256") != source_sha or delta.get("frozen") is not True):
        raise ValueError("web_delta_contract_rejected")
    validate_rows(delta["files"], prefix="web/")
    changes = []
    for row in delta["files"]:
        name = row["path"]
        if name not in WEB_CORRECTIONS or set(row) != {"path", "bytes", "sha256", "before_sha256"}:
            raise ValueError("web_delta_scope_rejected")
        previous = payloads.get(name)
        if row["before_sha256"] != (BASE.digest(previous) if previous is not None else None):
            raise ValueError("web_delta_parent_mismatch")
        payload = read_declared(delta_root, {**row, "path": name[4:]})
        payloads[name] = payload
        changes.append({**row, "operation": "reviewed_compile_test_delta", "delta_manifest_sha256": delta_sha})
    work = create_work(destination, payloads, source_root)
    rows = [BASE.row_for(name, payload) for name, payload in sorted(payloads.items())
            if name not in {"web/" + name for name in BASE.TEMPLATES}]
    amended = {**frozen, "destination_root": str(work), "source_tree_sha256": BASE.tree_sha(rows),
               "files": rows, "file_count": len(rows), "web_changes": frozen["web_changes"] + changes,
               "parent_frozen_manifest_sha256": source_sha, "reviewed_web_delta_sha256": delta_sha}
    BASE.write_new(work / "syscomplete-source-manifest.json", BASE.canonical_json(amended))
    return {"status": "amended_source_frozen", "source_root": str(work),
            "manifest_sha256": BASE.sha(work / "syscomplete-source-manifest.json"),
            "source_tree_sha256": amended["source_tree_sha256"], "changes": len(changes),
            "production_deployed": False}


def inherited_report_environment(path: Path, expected: str, inherited: list[dict]) -> tuple[bytes, dict]:
    receipt = checked_json(path, expected)
    if (receipt.get("schema") != "evomind.report_dependency_inheritance.v1"
            or receipt.get("base_build_id") != BASE.BASE_BUILD
            or receipt.get("base_source_tree_sha256") != BASE.BASE_TREE
            or receipt.get("base_source_manifest_sha256") != BASE.BASE_SOURCE_MANIFEST
            or receipt.get("production_changed") is not False or receipt.get("activation_claim") is not False
            or not re.fullmatch(r"3\.12\.\d+", str(receipt.get("python_version", "")))
            or str(receipt.get("platform", "")).replace("-", "_") != "win_amd64"):
        raise ValueError("report_inheritance_identity_rejected")
    venv = str(receipt.get("venv_root", "")).replace("\\", "/").rstrip("/")
    if venv.casefold() != "c:/evomind/releases/664a636ddd419a66f73cc10820c0a16429784866/.venv":
        raise ValueError("report_inheritance_venv_rejected")
    packages = receipt.get("packages", [])
    actual = {}
    file_count = 0
    for package in packages:
        name = str(package["name"]).lower().replace("_", "-")
        if name in actual or package.get("import_passed") is not True:
            raise ValueError("report_inheritance_package_rejected")
        validate_rows(package["files"])
        if any(not row["path"].startswith("Lib/site-packages/")
               and not (name == "pymupdf" and row["path"] == "Scripts/pymupdf.exe")
               for row in package["files"]):
            raise ValueError("report_inheritance_file_scope_rejected")
        module = str(package.get("module_path", "")).replace("\\", "/")
        allowed_module_paths = {(venv + "/" + row["path"]).casefold() for row in package["files"]}
        if module.casefold() not in allowed_module_paths:
            raise ValueError("report_inheritance_import_not_hash_bound")
        actual[name] = package["version"]
        file_count += len(package["files"])
    required = {row["name"].lower().replace("_", "-"): row["version"] for row in inherited}
    if actual != required:
        raise ValueError("report_inherited_version_mismatch")
    payload = path.read_bytes()
    if BASE.digest(payload) != expected:
        raise ValueError("report_inheritance_changed_during_read")
    return payload, {"receipt_sha256": expected, "recorded_file_count": file_count,
                     "python_version": receipt["python_version"], "platform": receipt["platform"],
                     "local_scope": "validated_receipt_not_live_server_recheck"}


def report_dependencies(path: Path, expected: str, root: Path, payloads: dict[str, bytes],
                        inheritance: Path | None = None) -> tuple[dict, list[dict]]:
    descriptor = checked_json(path, expected)
    allowed_fields = {"schema", "python_tag", "platform", "activation", "frozen", "wheels",
                      "distributions", "base_environment_binding"}
    if (set(descriptor) - allowed_fields
            or descriptor.get("schema") != "evomind.syscomplete_report_dependencies.v1"
            or descriptor.get("python_tag") != "cp312" or descriptor.get("platform") != "win_amd64"
            or descriptor.get("activation") != "offline_install_then_verify_imports"
            or descriptor.get("frozen") is not True):
        raise ValueError("report_dependency_contract_rejected")
    wheels = descriptor.get("wheels", [])
    validate_rows(wheels)
    if len(wheels) > 64 or sum(row["bytes"] for row in wheels) > 512 * 1024 * 1024:
        raise ValueError("report_wheelhouse_size_limit")
    if any(set(row) != {"path", "bytes", "sha256"} for row in wheels):
        raise ValueError("report_wheel_row_rejected")
    if any("/" in row["path"] or not row["path"].endswith(".whl") for row in wheels):
        raise ValueError("report_wheel_path_rejected")
    distributions = descriptor.get("distributions", [])
    if (not isinstance(distributions, list) or not distributions
            or any(set(row) != {"name", "version", "source"} for row in distributions)):
        raise ValueError("report_distributions_missing")
    names = {row["name"].lower().replace("_", "-") for row in distributions}
    if len(names) != len(distributions) or not REPORT_REQUIREMENTS.issubset(names):
        raise ValueError("report_required_distributions_missing")
    for row in distributions:
        if (not re.fullmatch(r"[a-zA-Z0-9_.-]+", row["name"])
                or not re.fullmatch(r"[a-zA-Z0-9_.+-]+", row["version"])
                or row["source"] not in {"wheelhouse", "pinned_base_environment"}):
            raise ValueError("report_distribution_rejected")
    wheel_distributions = {}
    for wheel in wheels:
        parts = wheel["path"].removesuffix(".whl").split("-")
        if len(parts) not in {5, 6}:
            raise ValueError("report_wheel_filename_rejected")
        package, version = parts[0].lower().replace("_", "-"), parts[1]
        if package in wheel_distributions:
            raise ValueError("report_duplicate_distribution_wheel")
        wheel_distributions[package] = version
    declared_wheels = {row["name"].lower().replace("_", "-"): row["version"] for row in distributions
                       if row["source"] == "wheelhouse"}
    if declared_wheels != wheel_distributions:
        raise ValueError("report_wheel_distribution_binding_mismatch")
    inherited = [row for row in distributions if row["source"] == "pinned_base_environment"]
    if not inherited and "base_environment_binding" in descriptor:
        raise ValueError("report_unused_base_environment_binding")
    if inherited:
        binding = descriptor.get("base_environment_binding", {})
        if (set(binding) != {"base_build_id", "receipt_sha256"}
                or binding.get("base_build_id") != BASE.BASE_BUILD):
            raise ValueError("report_inherited_environment_unbound")
        require_hash(binding.get("receipt_sha256", ""))
        if inheritance is None:
            raise ValueError("report_inheritance_receipt_required")
        inherited_payload, inherited_result = inherited_report_environment(inheritance, binding["receipt_sha256"], inherited)
    else:
        inherited_payload = inherited_result = None
    changes = []
    for row in wheels:
        if "/" in row["path"] or not row["path"].endswith(".whl"):
            raise ValueError("report_wheel_path_rejected")
        payload = read_declared(root, row)
        name = "web/support/report-wheelhouse/" + row["path"]
        if name in payloads:
            raise ValueError("report_wheel_would_overwrite_parent")
        payloads[name] = payload
        changes.append({**BASE.row_for(name, payload), "before_sha256": None})
    # Strip workstation path hints by allowing only the reviewed descriptor
    # fields above. The manifest stores logical delivery data, not credentials.
    descriptor_bytes = path.read_bytes()
    if BASE.digest(descriptor_bytes) != expected:
        raise ValueError("report_descriptor_changed_during_read")
    name = "web/support/report-wheelhouse/dependency-delivery.json"
    payloads[name] = descriptor_bytes
    changes.append({**BASE.row_for(name, descriptor_bytes), "before_sha256": None})
    if inherited_payload is not None:
        name = "web/support/report-wheelhouse/report-inheritance.json"
        payloads[name] = inherited_payload
        changes.append({**BASE.row_for(name, inherited_payload), "before_sha256": None})
    return {"descriptor_sha256": expected, "activation": descriptor["activation"],
            "python_tag": descriptor["python_tag"], "platform": descriptor["platform"],
            "wheel_count": len(wheels), "distributions": distributions,
            "inherited_environment_receipt": inherited_result,
            "server_import_acceptance": "pending", "activated": False}, changes


def create_work(destination: Path, payloads: dict[str, bytes], source_root: Path) -> Path:
    destination = destination.absolute()
    if (destination.exists() or destination.is_symlink() or destination.resolve(strict=False) != destination
            or destination.is_relative_to(source_root.absolute())):
        raise ValueError("assembly_destination_exists_or_aliased")
    destination.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        BASE.write_new(destination / BASE.safe_relative(name), payload)
    return destination


def save_build_recipes(work: Path) -> None:
    BASE.write_new(work / "recipes/syscomplete_release_assemble.py", RECIPE_BYTES)
    BASE.write_new(work / "recipes/build_invitation_release.py", LEGACY_RECIPE_BYTES)


def validate_source_unchanged(work: Path, source: dict) -> None:
    for row in source["files"]:
        read_declared(work, row)


def make_source(work: Path, frozen: dict, payloads: dict, changes: list, *, runtime_sha: str | None,
                dependencies: dict | None, source_manifest_sha: str) -> dict:
    template_names = {"web/" + name for name in BASE.TEMPLATES}
    rows = [BASE.row_for(name, payload) for name, payload in sorted(payloads.items()) if name not in template_names]
    identity = BASE.tree_sha(rows)
    source = {
        "schema": "evomind.invitation_source.v1", "frozen": True,
        "work_root": str(work.absolute()), "build_id": "overlay-invitation-beta-" + identity[:12] + "-sys1",
        "source_tree_sha256": identity, "files": rows, "changes": changes,
        "baseline_kind": "frozen_verified_release_pair", "baseline_web_build_id": BASE.BASE_BUILD,
        "baseline_archive_sha256": BASE.BASE_WEB, "baseline_runtime_archive_sha256": BASE.BASE_RUNTIME,
        "production_schema_sha256": "15ebee08fb5e042c286fa8ebe1c3718128e23e5c25ed87f865501039d7371f90",
        "database_schema_sha256": "ce878bdbf737662fa2332c5bef1495f68762d73b9f05217cf752cf4014149f1f",
        "runtime_file_count": sum(name.startswith("runtime/evomind_runtime/") for name in payloads),
        "production_deployed": False, "research_ablation_executed": False,
        "derivation": {"kind": "system_completeness_scoped_source", "parent_source_tree_sha256": BASE.BASE_TREE,
                       "web_source_manifest_sha256": source_manifest_sha, "runtime_delta_manifest_sha256": runtime_sha,
                       "assembly_recipe_sha256": BASE.digest(RECIPE_BYTES),
                       "legacy_build_recipe_sha256": BASE.digest(LEGACY_RECIPE_BYTES),
                       "unrelated_workspace_changes_included": False},
        "report_dependency_delivery": dependencies,
        "model_qualification": "pending_external_receipt", "browser_e2e": "pending",
        "server_dependency_activation": "pending", "release_verdict": "HOLD",
    }
    if runtime_sha is None:
        source.update(artifact_only=True, frontend_preflight_only=True)
    return source


def frontend_environment(work: Path, node: Path) -> dict:
    environment = LEGACY.command_environment(work / "web")
    local_appdata = work / "build-local-appdata"
    local_appdata.mkdir(exist_ok=True)
    environment.update(CI="1", LOCALAPPDATA=str(local_appdata), NPM_CONFIG_PROGRESS="false",
                       PATH=str(node.parent) + os.pathsep + environment.get("PATH", ""),
                       PYTHONUTF8="1", PYTHONNOUSERSITE="1")
    return environment


def validate_npm_lock(web: Path) -> None:
    lock = BASE.read_json(web / "package-lock.json")
    for name, row in lock["packages"].items():
        if name and (not str(row.get("resolved", "")).startswith("https://registry.npmjs.org/")
                     or not str(row.get("integrity", "")).startswith("sha512-")):
            raise ValueError("candidate_dependency_not_official_hash_locked")


def frontend_checks(work: Path, source: dict, node: Path, npm_cli: Path, *, next_build: bool) -> dict:
    if not node.is_file() or not npm_cli.is_file():
        raise ValueError("explicit_node_toolchain_missing")
    LEGACY.validate_build_inputs(work, source)
    validate_npm_lock(work / "web")
    validate_source_unchanged(work, source)
    environment = frontend_environment(work, node)
    web, logs = work / "web", work / "logs"
    commands = []
    argv = [
        ("npm-ci", [str(node), str(npm_cli), "ci", "--ignore-scripts", "--no-audit", "--no-fund", "--registry=https://registry.npmjs.org"], 360),
        ("prisma-generate", [str(node), str(web / "node_modules/prisma/build/index.js"), "generate", "--schema", "prisma/schema.prisma"], 180),
        ("fixture-schema", [str(node), str(web / "node_modules/prisma/build/index.js"), "db", "push", "--skip-generate", "--schema", "prisma/schema.prisma"], 120),
        ("frontend-tests", [str(node), "--test", "--experimental-strip-types", "--disable-warning=MODULE_TYPELESS_PACKAGE_JSON", "src/**/*.test.ts", "src/**/*.test.mjs"], 180),
        ("typecheck", [str(node), str(web / "node_modules/typescript/bin/tsc"), "--noEmit", "--incremental", "false"], 180),
    ]
    if next_build:
        argv.append(("next-build", [str(node), str(web / "node_modules/next/dist/bin/next"), "build", "--webpack"], 360))
    result = {"schema": "evomind.syscomplete_frontend_preflight.v1", "status": "running",
              "source_tree_sha256": source["source_tree_sha256"], "source_receipt_sha256": BASE.sha(work / "frontend-source-receipt.json"),
              "commands": commands, "production_deployed": False, "runtime_integrated": False}
    started = time.monotonic()
    try:
        for label, command, timeout in argv:
            commands.append(LEGACY.run_command(command, web, environment, logs, label, timeout))
        validate_source_unchanged(work, source)
        result.update(status="passed", frozen_source_unchanged=True, next_build_completed=next_build)
        return result
    finally:
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        if result["status"] == "running":
            result["status"] = "failed"
        LEGACY.json_file(work / "frontend-build-result.json", result)


def assemble(args) -> Path:
    frozen, payloads = read_source(args.source, args.source_manifest_sha256)
    _runtime, runtime_changes = runtime_delta(args.runtime_manifest, args.runtime_manifest_sha256, args.runtime_root, payloads)
    dependencies, dependency_changes = report_dependencies(args.report_dependencies, args.report_dependencies_sha256,
                                                           args.report_wheel_root, payloads,
                                                           getattr(args, "report_inheritance", None))
    work = create_work(args.destination, payloads, args.source)
    save_build_recipes(work)
    for name, path, expected in (
        ("web-source-manifest.json", args.source / "syscomplete-source-manifest.json", args.source_manifest_sha256),
        ("runtime-delta-manifest.json", args.runtime_manifest, args.runtime_manifest_sha256),
        ("report-dependency-delivery.json", args.report_dependencies, args.report_dependencies_sha256),
    ):
        payload = path.read_bytes()
        if BASE.digest(payload) != expected:
            raise ValueError("assembly_input_receipt_changed")
        BASE.write_new(work / "input-receipts" / name, payload)
    if getattr(args, "report_inheritance", None):
        BASE.write_new(work / "input-receipts/report-inheritance.json",
                       payloads["web/support/report-wheelhouse/report-inheritance.json"])
    source = make_source(work, frozen, payloads, frozen["web_changes"] + runtime_changes + dependency_changes,
                         runtime_sha=args.runtime_manifest_sha256, dependencies=dependencies,
                         source_manifest_sha=args.source_manifest_sha256)
    receipt = work / "source-receipt.json"
    BASE.write_new(receipt, BASE.canonical_json(source))
    LEGACY.validate_build_inputs(work, source)
    validate_source_unchanged(work, source)
    if LEGACY.files(work / "runtime/evomind_runtime") != LEGACY.files(work / "web/support/python-runtime/evomind_runtime"):
        raise ValueError("assembled_runtime_mirror_mismatch")
    BASE.write_new(work / "assembly-receipt.json", BASE.canonical_json({
        "schema": "evomind.syscomplete_assembly.v1", "status": "assembled_for_build",
        "build_id": source["build_id"], "source_tree_sha256": source["source_tree_sha256"],
        "source_receipt_sha256": BASE.sha(receipt), "runtime_manifest_sha256": args.runtime_manifest_sha256,
        "report_dependency_descriptor_sha256": args.report_dependencies_sha256,
        "runtime_mirror_verified": True, "production_deployed": False, "release_verdict": "HOLD",
    }))
    return receipt


def verified_launcher_binding(launcher_manifest: Path, expected_sha: str) -> dict:
    launcher = checked_json(launcher_manifest, expected_sha)
    schema = launcher.get("schema")
    if (schema not in {"evomind.launcher_candidate.v1", "evomind.launcher_candidate.v2"}
            or launcher.get("production_changed") is not False
            or launcher.get("profile", {}).get("model") != "gpt-5.5"):
        raise ValueError("external_launcher_binding_rejected")
    if schema == "evomind.launcher_candidate.v1":
        rows = [launcher["candidate"], launcher["patch"]]
        deployment_files = []
        patch_sha = launcher["patch"]["sha256"]
    else:
        rows = launcher.get("files", [])
        validate_rows(rows)
        declared = {row["path"]: row for row in rows}
        deployment_files = launcher.get("deployment_files", [])
        binding = launcher.get("dependency_binding", {})
        expected_targets = {
            "candidate/Start-Node.ps1": "C:/ProgramData/EvoMind/bundle/scripts/Start-Node.ps1",
            "candidate/lib/verify_report_dependencies.py": "C:/ProgramData/EvoMind/bundle/scripts/lib/verify_report_dependencies.py",
            "candidate/report-dependency-activation.json": "C:/ProgramData/EvoMind/report-envs/gpt55-806f87b009cd/report-dependency-activation.json",
        }
        if (len(deployment_files) != 3 or {row.get("path") for row in deployment_files} != set(expected_targets)
                or set(declared) != {*expected_targets, "Start-Node.gpt55-report.patch"}
                or launcher.get("activation_claim") is not False):
            raise ValueError("external_launcher_file_set_rejected")
        for row in deployment_files:
            if (row.get("target") != expected_targets[row["path"]]
                    or row.get("sha256") != declared[row["path"]]["sha256"]):
                raise ValueError("external_launcher_target_binding_rejected")
        if (binding.get("path") != "candidate/report-dependency-activation.json"
                or binding.get("sha256") != declared[binding["path"]]["sha256"]
                or binding.get("target") != expected_targets[binding["path"]]
                or launcher["candidate"]["path"] != "candidate/Start-Node.ps1"
                or launcher["candidate"]["sha256"] != declared["candidate/Start-Node.ps1"]["sha256"]):
            raise ValueError("external_launcher_dependency_binding_rejected")
        patch_sha = declared["Start-Node.gpt55-report.patch"]["sha256"]
    for row in rows:
        path = BASE.checked_path(launcher_manifest.parent, row["path"])
        if BASE.sha(path) != row["sha256"] or ("bytes" in row and path.stat().st_size != row["bytes"]):
            raise ValueError("external_launcher_artifact_changed")
    return {"manifest_sha256": expected_sha, "candidate_sha256": launcher["candidate"]["sha256"],
            "patch_sha256": patch_sha, "deployment_files": deployment_files,
            "included_in_source_bundle": False, "activated": False}


def package_acceptance(work: Path, built: dict, launcher_manifest: Path | None = None,
                       launcher_sha: str | None = None) -> dict:
    """A third, hash-bound acceptance-material ZIP; contains no activated code."""
    source = BASE.read_json(work / "source-receipt.json")
    if (built.get("status") != "built" or built.get("source_receipt_sha256") != BASE.sha(work / "source-receipt.json")
            or built.get("source_tree_sha256") != source["source_tree_sha256"]):
        raise ValueError("acceptance_pack_build_receipt_mismatch")
    LEGACY.validate_release_identity(work / "release", work / "runtime", source)
    for kind in ("web", "runtime"):
        if BASE.sha(work / (kind + ".zip")) != built[kind]["sha256"]:
            raise ValueError("acceptance_pack_archive_mismatch")
    target = work / "acceptance-materials"
    if target.exists():
        raise ValueError("acceptance_pack_already_exists")
    target.mkdir()
    for name in ("source-receipt.json", "assembly-receipt.json", "build-result.json"):
        BASE.write_new(target / name, (work / name).read_bytes())
    for root_name in ("input-receipts", "recipes"):
        for row in LEGACY.files(work / root_name):
            BASE.write_new(target / root_name / row["path"], read_declared(work / root_name, row))
    fixture = work / "fixture/build.sqlite"
    with sqlite3.connect(fixture.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        names = [str(row[0]).replace('"', '""') for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        if any(connection.execute('SELECT count(*) FROM "' + name + '"').fetchone()[0] for name in names):
            raise ValueError("acceptance_fixture_contains_rows")
    BASE.write_new(target / "fixture-schema.sqlite", fixture.read_bytes())
    for name in ("verify_invitation_server_candidate.py", "verify_invitation_extensions.py", "verify_invitation_async_approval.py"):
        BASE.write_new(target / name, (ROOT / "scripts" / name).read_bytes())
    external_launcher = None
    if launcher_manifest is not None:
        external_launcher = verified_launcher_binding(launcher_manifest, launcher_sha or "")
        BASE.write_new(target / "external-launcher-candidate-manifest.json", launcher_manifest.read_bytes())
    fixture_sha = BASE.sha(target / "fixture-schema.sqlite")
    stage = "C:/ProgramData/EvoMind/staging/invitation-" + source["source_tree_sha256"][:12] + "-sys1"
    python = "C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/.venv/Scripts/python.exe"
    commands = [[python, stage + "/verify_invitation_server_candidate.py", "verify", "--stage-root", stage,
                 "--web-sha256", built["web"]["sha256"], "--runtime-sha256", built["runtime"]["sha256"],
                 "--fixture-sha256", fixture_sha, "--build-id", source["build_id"]],
                [python, stage + "/verify_invitation_extensions.py", "--stage-root", stage,
                 "--web-sha256", built["web"]["sha256"], "--runtime-sha256", built["runtime"]["sha256"],
                 "--build-id", source["build_id"]],
                [python, stage + "/verify_invitation_async_approval.py", "--stage-root", stage,
                 "--web-sha256", built["web"]["sha256"], "--runtime-sha256", built["runtime"]["sha256"],
                 "--build-id", source["build_id"]]]
    BASE.write_new(target / "server-isolation-commands.json", BASE.canonical_json({
        "commands": commands, "stage_root": stage, "execution_status": "not_run",
        "scope": "isolated_fixture_no_model_or_gpu_no_production_mutation",
        "report_dependency_activation": "separate_gate_not_proven_by_these_existing_fixtures",
        "external_launcher_binding": external_launcher,
    }))
    rows = LEGACY.files(target)
    BASE.write_new(target / "acceptance-manifest.json", BASE.canonical_json({
        "schema": "evomind.syscomplete_acceptance_materials.v1", "build_id": source["build_id"],
        "source_tree_sha256": source["source_tree_sha256"], "web_sha256": built["web"]["sha256"],
        "runtime_sha256": built["runtime"]["sha256"], "fixture_sha256": fixture_sha,
        "files": rows, "file_count": len(rows), "external_launcher_binding": external_launcher,
        "production_deployed": False, "release_verdict": "HOLD", "real_chrome_e2e": "pending",
    }))
    artifact = LEGACY.zip_tree(target, work / "acceptance.zip")
    result = {"schema": "evomind.syscomplete_delivery.v1", "status": "three_packages_built_not_deployed",
              "build_id": source["build_id"], "source_tree_sha256": source["source_tree_sha256"],
              "web": built["web"], "runtime": built["runtime"], "acceptance": artifact,
              "fixture_sha256": fixture_sha, "external_launcher_binding": external_launcher,
              "production_deployed": False, "release_verdict": "HOLD"}
    BASE.write_new(work / "delivery-result.json", BASE.canonical_json(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    amendment = sub.add_parser("amend-web")
    amendment.add_argument("--source", type=Path, required=True)
    amendment.add_argument("--source-manifest-sha256", required=True)
    amendment.add_argument("--delta", type=Path, required=True)
    amendment.add_argument("--delta-sha256", required=True)
    amendment.add_argument("--delta-root", type=Path, required=True)
    amendment.add_argument("--destination", type=Path, required=True)
    for name in ("frontend", "assemble"):
        mode = sub.add_parser(name)
        mode.add_argument("--source", type=Path, default=Path("D:/EV12/syscomplete-websource-v1"))
        mode.add_argument("--source-manifest-sha256", default=FROZEN_SOURCE_SHA)
        mode.add_argument("--destination", type=Path, required=True)
        mode.add_argument("--node", type=Path, default=DEFAULT_NODE_ROOT / "node.exe")
        mode.add_argument("--npm-cli", type=Path, default=DEFAULT_NODE_ROOT / "node_modules/npm/bin/npm-cli.js")
        if name == "frontend":
            mode.add_argument("--next-build", action="store_true")
        else:
            mode.add_argument("--runtime-root", type=Path, required=True)
            mode.add_argument("--runtime-manifest", type=Path, required=True)
            mode.add_argument("--runtime-manifest-sha256", required=True)
            mode.add_argument("--report-dependencies", type=Path, required=True)
            mode.add_argument("--report-dependencies-sha256", required=True)
            mode.add_argument("--report-wheel-root", type=Path, required=True)
            mode.add_argument("--report-inheritance", type=Path)
            mode.add_argument("--launcher-manifest", type=Path)
            mode.add_argument("--launcher-manifest-sha256")
            mode.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.mode == "amend-web":
        print(json.dumps(amend_web(args.source, args.source_manifest_sha256, args.delta, args.delta_sha256,
                                   args.delta_root, args.destination)), flush=True)
    elif args.mode == "frontend":
        frozen, payloads = read_source(args.source, args.source_manifest_sha256)
        work = create_work(args.destination, payloads, args.source)
        save_build_recipes(work)
        source = make_source(work, frozen, payloads, frozen["web_changes"], runtime_sha=None,
                             dependencies=None, source_manifest_sha=args.source_manifest_sha256)
        BASE.write_new(work / "frontend-source-receipt.json", BASE.canonical_json(source))
        print(json.dumps(frontend_checks(work, source, args.node, args.npm_cli, next_build=args.next_build)), flush=True)
    else:
        receipt = assemble(args)
        print(json.dumps({"status": "assembled_for_build", "source_receipt": str(receipt),
                          "source_receipt_sha256": BASE.sha(receipt), "release_verdict": "HOLD"}), flush=True)
        if args.build:
            # Use the original release build/identity validation path unchanged.
            # Bind its PATH-based discovery to the explicitly selected toolchain.
            if not args.node.is_file() or not args.npm_cli.is_file():
                raise ValueError("explicit_node_toolchain_missing")
            expected_cli = args.node.parent / "node_modules/npm/bin/npm-cli.js"
            if args.npm_cli.absolute() != expected_cli.absolute():
                raise ValueError("legacy_build_requires_adjacent_npm_cli_use_frontend_preflight_toolchain")
            os.environ["PATH"] = str(args.node.parent) + os.pathsep + os.environ.get("PATH", "")
            built = LEGACY.build(receipt)
            print(json.dumps(package_acceptance(receipt.parent, built, args.launcher_manifest,
                                                 args.launcher_manifest_sha256)), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        code = str(error) if re.fullmatch(r"[A-Za-z0-9_:/.[\]-]{1,240}", str(error)) else "diagnostic_withheld"
        print(json.dumps({"status": "failed", "error_class": type(error).__name__, "error_code": code,
                          "production_deployed": False}), flush=True)
        sys.exit(1)
