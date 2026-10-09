from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any
import uuid


SCHEMA = "evomind.g21_terminal_preserving_probe_runtime_deployment_approval.r120.v1"
RELEASE_SCHEMA = "evomind.g21_terminal_preserving_probe_runtime.r120.v1"
RECEIPT_SCHEMA = "evomind.g21_terminal_preserving_probe_runtime_test_receipt.r120.v1"
FIXED_RUN_ID = "run_7b1efb878afb40f396db431e91f093a5"
FIXED_ALLOCATION = "G21"
FIXED_WEB_BUILD_ID = "overlay-controlled-secret-binding-r112-standalone-f38c40f2dbdc"
BASELINE_ZIP_SHA256 = "9368c7962632fb976e7a8ba8ede0d3108989e96c00c4dd75bca154008d438206"
BASELINE_TREE_SHA256 = "906868d22a841f591fb84c7a505b75acee3c8df58f179e83d3d4585601d03975"
BASELINE_MANIFEST_SHA256 = "12f80d02d6fd87e413c57bc606d09a27f90c3edac3b4b1d7462826374666b9ac"
CANDIDATE_TREE_SHA256 = "5998ea21823c121c23a421bb145d8e2266bb79ab59eccc91a16122c3626b61ce"
SOURCE_RAW_SHA256 = "519820314f63d2e5ae6ba354e044ea7df55c544a6c0ab85de43102279af0af4d"
PACKAGED_HTTP_SERVER_SHA256 = "6d2d17f1257736683c4bc590f346c183c9feb25e06cce7a77d63d35843e23a9d"
CHANGED_FILE = "evomind_runtime/http_server.py"
TRANSPORT_MODE = "single_scp_envelope"
SHA_RE = re.compile(r"^[a-f0-9]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def regular(value: str, label: str) -> Path:
    path = Path(value).resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label}_REGULAR_FILE_REQUIRED")
    return path


def copy_bound(source: Path, output: Path) -> dict[str, Any]:
    target = output / source.name
    if target.exists():
        raise ValueError(f"R120_INPUT_COLLISION:{source.name}")
    shutil.copy2(source, target)
    return {"filename": target.name, "bytes": target.stat().st_size, "sha256": sha256_file(target)}


def require_sha(value: Any, label: str) -> str:
    text = str(value or "")
    if not SHA_RE.fullmatch(text):
        raise ValueError(f"{label}_SHA_REJECTED")
    return text


def build(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    paths = {
        "candidate": regular(args.candidate_zip, "CANDIDATE"),
        "source_manifest": regular(args.source_manifest, "SOURCE_MANIFEST"),
        "build_result": regular(args.build_result, "BUILD_RESULT"),
        "test_receipt": regular(args.test_receipt, "TEST_RECEIPT"),
        "verifier": regular(args.verifier, "VERIFIER"),
        "wrapper": regular(args.wrapper, "WRAPPER"),
        "remote_deployer": regular(args.remote_deployer, "REMOTE_DEPLOYER"),
        "probe_wrapper": regular(args.probe_wrapper, "PROBE_WRAPPER"),
        "probe_remote_runner": regular(args.probe_remote_runner, "PROBE_REMOTE_RUNNER"),
    }
    manifest = load_object(paths["source_manifest"])
    build_result = load_object(paths["build_result"])
    receipt = load_object(paths["test_receipt"])
    candidate_sha = sha256_file(paths["candidate"])
    manifest_sha = sha256_file(paths["source_manifest"])
    build_sha = sha256_file(paths["build_result"])
    candidate_tree = require_sha(manifest.get("candidate_runtime_tree_sha256"), "CANDIDATE_TREE")
    if candidate_tree != CANDIDATE_TREE_SHA256:
        raise ValueError("R120_CANDIDATE_TREE_REJECTED")
    source_raw_sha = require_sha(manifest.get("source_http_server_raw_sha256"), "SOURCE_HTTP_SERVER_RAW")
    packaged_sha = require_sha(manifest.get("source_http_server_packaged_sha256"), "SOURCE_HTTP_SERVER_PACKAGED")
    if source_raw_sha != SOURCE_RAW_SHA256 or packaged_sha != PACKAGED_HTTP_SERVER_SHA256:
        raise ValueError("R120_HTTP_SERVER_SOURCE_BINDING_REJECTED")
    if (
        manifest.get("schema") != RELEASE_SCHEMA
        or manifest.get("run_id") != FIXED_RUN_ID
        or manifest.get("allocation") != FIXED_ALLOCATION
        or manifest.get("baseline_zip_sha256") != BASELINE_ZIP_SHA256
        or manifest.get("baseline_runtime_tree_sha256") != BASELINE_TREE_SHA256
        or manifest.get("baseline_manifest_sha256") != BASELINE_MANIFEST_SHA256
        or int(manifest.get("candidate_runtime_file_count", -1)) != 32
        or int(manifest.get("changed_file_count", -1)) != 1
        or int(manifest.get("unchanged_file_count", -1)) != 31
        or manifest.get("changed_files") != [CHANGED_FILE]
        or manifest.get("terminal_preserving_direct_tool") != "hpc_asset_probe"
        or manifest.get("terminal_preserving_fixed_run") != FIXED_RUN_ID
        or manifest.get("cancel_workaround_forbidden") is not True
        or bool(manifest.get("production_deployable"))
        or bool(manifest.get("production_deployed"))
    ):
        raise ValueError("R120_SOURCE_MANIFEST_REJECTED")
    if (
        build_result.get("schema") != RELEASE_SCHEMA
        or build_result.get("status") != "built_and_verified"
        or build_result.get("zip_sha256") != candidate_sha
        or build_result.get("source_manifest_sha256") != manifest_sha
        or build_result.get("candidate_runtime_tree_sha256") != candidate_tree
        or bool(build_result.get("production_deployed"))
        or bool(build_result.get("hpc_accessed"))
        or bool(build_result.get("gpu_touched"))
        or int(build_result.get("remote_writes", -1)) != 0
    ):
        raise ValueError("R120_BUILD_RESULT_REJECTED")
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("status") != "verified"
        or receipt.get("candidate_zip_sha256") != candidate_sha
        or receipt.get("source_manifest_sha256") != manifest_sha
        or receipt.get("build_result_sha256") != build_sha
        or receipt.get("candidate_runtime_tree_sha256") != candidate_tree
        or receipt.get("source_http_server_raw_sha256") != source_raw_sha
        or receipt.get("source_http_server_packaged_sha256") != packaged_sha
        or len(receipt.get("verification_rounds", [])) != 2
        or bool(receipt.get("production_deployed"))
        or bool(receipt.get("hpc_accessed"))
        or bool(receipt.get("gpu_touched"))
        or int(receipt.get("remote_writes", -1)) != 0
    ):
        raise ValueError("R120_TEST_RECEIPT_REJECTED")
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise ValueError(f"OUTPUT_DIRECTORY_EXISTS:{output}")
    output.mkdir(parents=True)
    copied = {name: copy_bound(path, output) for name, path in paths.items()}
    now = datetime.now(timezone.utc)
    valid_minutes = int(args.valid_minutes)
    if valid_minutes < 5 or valid_minutes > 60:
        raise ValueError("VALID_MINUTES_REJECTED")
    approval = {
        "schema": SCHEMA,
        "status": "validation_only_non_authorizing" if args.validate_only else "approved_for_single_runtime_transaction",
        "one_time": True,
        "deployment_id": uuid.uuid4().hex,
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(minutes=valid_minutes)).isoformat(),
        "run_id": FIXED_RUN_ID,
        "allocation": FIXED_ALLOCATION,
        "web_build_id": FIXED_WEB_BUILD_ID,
        "preflight": {
            "run_status_required": "cancelled",
            "run_terminal_required": True,
            "effective_active_required": 0,
            "pending_approvals_required": 0,
            "current_runtime_file_count": 32,
            "current_runtime_tree_sha256": BASELINE_TREE_SHA256,
            "no_concurrent_transaction": True,
        },
        "baseline": {
            "release": "r119-g21-hpc-asset-probe-r118-base",
            "zip_sha256": BASELINE_ZIP_SHA256,
            "runtime_tree_sha256": BASELINE_TREE_SHA256,
            "manifest_sha256": BASELINE_MANIFEST_SHA256,
            "runtime_file_count": 32,
        },
        "candidate": {
            **copied["candidate"],
            "source_manifest_filename": copied["source_manifest"]["filename"],
            "source_manifest_sha256": manifest_sha,
            "build_result_filename": copied["build_result"]["filename"],
            "build_result_sha256": build_sha,
            "installed_runtime_tree_sha256": candidate_tree,
            "source_http_server_raw_sha256": source_raw_sha,
            "source_http_server_packaged_sha256": packaged_sha,
            "runtime_file_count": 32,
            "changed_file_count": 1,
            "unchanged_file_count": 31,
            "changed_files": [CHANGED_FILE],
        },
        "verification": {
            "test_receipt_filename": copied["test_receipt"]["filename"],
            "test_receipt_sha256": copied["test_receipt"]["sha256"],
            "candidate_verifier_filename": copied["verifier"]["filename"],
            "candidate_verifier_sha256": copied["verifier"]["sha256"],
        },
        "production_invariants": {
            "goal_spec_sha256": "fbb9939dff28c355c2798bf32fce104845337064f42f9c280f2b6906cccf4036",
            "goal_board_sha256": "6b908173083889ac21f8afea52528b4b5e35d2ff862a880a6065d65b169f443c",
            "goal_mutation_forbidden": True,
            "database_content_mutation_forbidden": True,
            "hpc_access_forbidden": True,
            "terminal_preserving_tool": "hpc_asset_probe",
            "cancel_workaround_forbidden": True,
        },
        "deployer": {
            "wrapper_filename": copied["wrapper"]["filename"],
            "wrapper_sha256": copied["wrapper"]["sha256"],
            "remote_runner_filename": copied["remote_deployer"]["filename"],
            "remote_runner_sha256": copied["remote_deployer"]["sha256"],
            "service_action_path": "C:\\SecureInput\\Invoke-ServiceAccountAction.ps1",
            "allowed_service_actions": ["Stop", "Start"],
            "direct_scheduled_task_start_forbidden": True,
            "direct_process_kill_forbidden": True,
            "direct_sqlite_forbidden": True,
        },
        "probe_runner": {
            "wrapper_filename": copied["probe_wrapper"]["filename"],
            "wrapper_sha256": copied["probe_wrapper"]["sha256"],
            "remote_runner_filename": copied["probe_remote_runner"]["filename"],
            "remote_runner_sha256": copied["probe_remote_runner"]["sha256"],
            "direct_tool_calls": 1,
            "cancel_calls": 0,
            "preserve_terminal_required": True,
        },
        "rollback": {
            "restore_runtime_tree_sha256": BASELINE_TREE_SHA256,
            "restore_runtime_file_count": 32,
            "restore_database_and_wal_shm": True,
            "health_3_of_3_required": True,
        },
        "bound_inputs": copied,
        "transport": {"mode": TRANSPORT_MODE, "single_scp_upload": True, "envelope_extract_required": True},
        "boundaries": {
            "runtime_only": True, "web_unchanged": True, "hpc_access_forbidden": True,
            "gpu_touch_forbidden": True, "weather_forbidden": True, "training_forbidden_during_deployment": True,
        },
        "production_deployable": not args.validate_only,
    }
    path = output / "evomind-runtime-r120-terminal-preserving-probe-deployment-approval.json"
    path.write_text(json.dumps(approval, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path, approval


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Create a bound R120 runtime-only deployment approval")
    value.add_argument("--candidate-zip", required=True)
    value.add_argument("--source-manifest", required=True)
    value.add_argument("--build-result", required=True)
    value.add_argument("--test-receipt", required=True)
    value.add_argument("--verifier", required=True)
    value.add_argument("--wrapper", required=True)
    value.add_argument("--remote-deployer", required=True)
    value.add_argument("--probe-wrapper", required=True)
    value.add_argument("--probe-remote-runner", required=True)
    value.add_argument("--output-dir", required=True)
    value.add_argument("--valid-minutes", type=int, default=20)
    value.add_argument("--validate-only", action="store_true")
    return value


def main() -> int:
    try:
        path, approval = build(parser().parse_args())
    except Exception as exc:
        print(json.dumps({"schema": SCHEMA, "status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({"schema": SCHEMA, "status": "created", "approval_filename": path.name, "approval_sha256": sha256_file(path), "expires_at": approval["expires_at"], "production_action_performed": False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
