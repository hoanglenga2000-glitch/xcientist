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


SCHEMA = "evomind.g21_hpc_asset_probe_runtime_deployment_approval.r121.v1"
RELEASE_SCHEMA = "evomind.g21_hpc_asset_probe_runtime.r121.v1"
RECEIPT_SCHEMA = "evomind.g21_hpc_asset_probe_runtime_test_receipt.r121.v1"
FIXED_RUN_ID = "run_7b1efb878afb40f396db431e91f093a5"
FIXED_ALLOCATION = "G21"
FIXED_WEB_BUILD_ID = "overlay-controlled-secret-binding-r112-standalone-f38c40f2dbdc"
BASELINE_ZIP_SHA256 = "979224d6baf2d779357d2d2a7ffacd57637289dce11a716857c15fe913eafa3c"
BASELINE_TREE_SHA256 = "5998ea21823c121c23a421bb145d8e2266bb79ab59eccc91a16122c3626b61ce"
BASELINE_MANIFEST_SHA256 = "ba78b0f3c0759e86d8af5e89f10a1adc07a8c547f7afa11aedf684fa477125e4"
CANDIDATE_TREE_SHA256 = "a9595ab4ed9a86786ca65f4456893d64741b449ec84cf8c9ddb42e9bf4110dc7"
PACKAGED_TOOLS_SHA256 = "8176c799277f7c56f67232dbfc3f4fda9c5fad88e1aa280087d94393433e97fd"
GOAL_SPEC_SHA256 = "fbb9939dff28c355c2798bf32fce104845337064f42f9c280f2b6906cccf4036"
GOAL_BOARD_SHA256 = "6b908173083889ac21f8afea52528b4b5e35d2ff862a880a6065d65b169f443c"
CHANGED_FILE = "evomind_runtime/tools.py"
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
        raise ValueError(f"R121_INPUT_COLLISION:{source.name}")
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
    }
    manifest = load_object(paths["source_manifest"])
    build_result = load_object(paths["build_result"])
    receipt = load_object(paths["test_receipt"])
    candidate_sha = sha256_file(paths["candidate"])
    manifest_sha = sha256_file(paths["source_manifest"])
    build_sha = sha256_file(paths["build_result"])
    candidate_tree = require_sha(manifest.get("candidate_runtime_tree_sha256"), "CANDIDATE_TREE")
    packaged_tools = require_sha(manifest.get("packaged_tools_sha256"), "PACKAGED_TOOLS")
    if candidate_tree != CANDIDATE_TREE_SHA256 or packaged_tools != PACKAGED_TOOLS_SHA256:
        raise ValueError("R121_CANDIDATE_HASH_BINDING_REJECTED")
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
        or manifest.get("output_bound", {}).get("model_file_details_included") is not False
        or manifest.get("output_bound", {}).get("model_metadata_closure_complete") is not True
        or bool(manifest.get("production_deployable"))
        or bool(manifest.get("production_deployed"))
    ):
        raise ValueError("R121_SOURCE_MANIFEST_REJECTED")
    if (
        build_result.get("schema") != RELEASE_SCHEMA
        or build_result.get("status") != "built_and_verified"
        or build_result.get("zip_sha256") != candidate_sha
        or build_result.get("source_manifest_sha256") != manifest_sha
        or build_result.get("candidate_runtime_tree_sha256") != candidate_tree
        or build_result.get("packaged_tools_sha256") != packaged_tools
        or bool(build_result.get("production_deployed"))
        or bool(build_result.get("hpc_accessed"))
        or bool(build_result.get("gpu_touched"))
        or int(build_result.get("remote_writes", -1)) != 0
    ):
        raise ValueError("R121_BUILD_RESULT_REJECTED")
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("status") != "verified"
        or receipt.get("candidate_zip_sha256") != candidate_sha
        or receipt.get("source_manifest_sha256") != manifest_sha
        or receipt.get("build_result_sha256") != build_sha
        or receipt.get("candidate_runtime_tree_sha256") != candidate_tree
        or receipt.get("packaged_tools_sha256") != packaged_tools
        or len(receipt.get("verification_rounds", [])) != 2
        or bool(receipt.get("production_deployed"))
        or bool(receipt.get("hpc_accessed"))
        or bool(receipt.get("gpu_touched"))
        or int(receipt.get("remote_writes", -1)) != 0
    ):
        raise ValueError("R121_TEST_RECEIPT_REJECTED")
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise ValueError(f"OUTPUT_DIRECTORY_EXISTS:{output}")
    output.mkdir(parents=True)
    copied = {name: copy_bound(path, output) for name, path in paths.items()}
    valid_minutes = int(args.valid_minutes)
    if valid_minutes < 5 or valid_minutes > 60:
        raise ValueError("VALID_MINUTES_REJECTED")
    now = datetime.now(timezone.utc)
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
            "release": "r120-g21-terminal-preserving-probe-r119-base",
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
            "packaged_tools_sha256": packaged_tools,
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
            "goal_spec_sha256": GOAL_SPEC_SHA256,
            "goal_board_sha256": GOAL_BOARD_SHA256,
            "goal_mutation_forbidden": True,
            "database_content_mutation_forbidden": True,
            "hpc_access_forbidden": True,
            "gpu_touch_forbidden": True,
            "tool_name": "hpc_asset_probe",
            "tool_read_only": True,
            "output_bound_fix": True,
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
        "rollback": {
            "restore_runtime_tree_sha256": BASELINE_TREE_SHA256,
            "restore_runtime_file_count": 32,
            "restore_database_and_wal_shm": True,
            "health_3_of_3_required": True,
        },
        "bound_inputs": copied,
        "transport": {"mode": TRANSPORT_MODE, "single_scp_upload": True, "envelope_extract_required": True},
        "boundaries": {
            "runtime_only": True,
            "web_unchanged": True,
            "hpc_access_forbidden": True,
            "gpu_touch_forbidden": True,
            "weather_forbidden": True,
            "training_forbidden_during_deployment": True,
        },
        "production_deployable": not args.validate_only,
    }
    path = output / "evomind-runtime-r121-hpc-asset-probe-deployment-approval.json"
    path.write_text(json.dumps(approval, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path, approval


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Create a bound R121 output-bound asset-probe runtime deployment approval")
    value.add_argument("--candidate-zip", required=True)
    value.add_argument("--source-manifest", required=True)
    value.add_argument("--build-result", required=True)
    value.add_argument("--test-receipt", required=True)
    value.add_argument("--verifier", required=True)
    value.add_argument("--wrapper", required=True)
    value.add_argument("--remote-deployer", required=True)
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
