from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


SCHEMA = "evomind.managed_hpc_identity_rebind_runtime_deployment_approval.r123.v1"
RELEASE_SCHEMA = "evomind.managed_hpc_identity_rebind_runtime.r123.v1"
RECEIPT_SCHEMA = "evomind.managed_hpc_identity_rebind_runtime_test_receipt.r123.v1"
FIXED_RUN_ID = "run_7b1efb878afb40f396db431e91f093a5"
TARGET_ALLOCATION = "G24"
TARGET_JOB_ID = 93015
WEB_BUILD_ID = "overlay-assistant-interaction-r122-1393046c9de7"
BASELINE_ZIP_SHA256 = "f927d3538b6b1f1a02ed4fb57312d2663296e98baa56940660c321058440f4d2"
BASELINE_TREE_SHA256 = "a9595ab4ed9a86786ca65f4456893d64741b449ec84cf8c9ddb42e9bf4110dc7"
BASELINE_MANIFEST_SHA256 = "d644dc05a01cc4fec0201716bbfdc7a108a7f412c9cc98edfe95ef7a3bac8da8"
CHANGED_FILES = [
    "evomind_runtime/assistant_runs.py",
    "evomind_runtime/http_server.py",
    "evomind_runtime/store.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular(path: str, code: str) -> Path:
    value = Path(path).resolve()
    if not value.is_file() or value.is_symlink():
        raise ValueError(f"R123_{code}_REGULAR_FILE_REQUIRED")
    return value


def load_object(path: Path, code: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"R123_{code}_OBJECT_REQUIRED")
    return value


def copy_bound(source: Path, output: Path) -> dict[str, Any]:
    target = output / source.name
    shutil.copy2(source, target)
    if sha256_file(target) != sha256_file(source):
        raise ValueError(f"R123_BOUND_COPY_HASH_REJECTED:{source.name}")
    return {"filename": target.name, "bytes": target.stat().st_size, "sha256": sha256_file(target)}


def verified_rounds(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    for item in value:
        if not isinstance(item, dict) or int(item.get("py_compile", 0)) != 32:
            return False
        smoke = item.get("contract_smoke")
        if not isinstance(smoke, dict) or smoke.get("isolated_python") is not True:
            return False
        if smoke.get("candidate_import_origin") is not True:
            return False
        hashes = smoke.get("candidate_import_hashes")
        if not isinstance(hashes, dict) or set(hashes) != {"assistant_runs", "store", "http_server"}:
            return False
        junit = smoke.get("junit")
        if (
            not isinstance(junit, dict)
            or int(junit.get("tests", 0)) < 1
            or int(junit.get("failures", -1)) != 0
            or int(junit.get("errors", -1)) != 0
        ):
            return False
    return True


def create(args: argparse.Namespace) -> dict[str, Any]:
    deployment_id = str(args.deployment_id or "").lower()
    if len(deployment_id) != 32 or any(char not in "0123456789abcdef" for char in deployment_id):
        raise ValueError("R123_DEPLOYMENT_ID_REJECTED")
    paths = {
        "candidate": regular(args.candidate_zip, "CANDIDATE"),
        "source_manifest": regular(args.source_manifest, "SOURCE_MANIFEST"),
        "build_result": regular(args.build_result, "BUILD_RESULT"),
        "test_receipt": regular(args.test_receipt, "TEST_RECEIPT"),
        "verifier": regular(args.verifier, "VERIFIER"),
        "wrapper": regular(args.wrapper, "WRAPPER"),
        "remote_deployer": regular(args.remote_deployer, "REMOTE_DEPLOYER"),
    }
    manifest = load_object(paths["source_manifest"], "SOURCE_MANIFEST")
    build = load_object(paths["build_result"], "BUILD_RESULT")
    receipt = load_object(paths["test_receipt"], "TEST_RECEIPT")
    candidate_sha = sha256_file(paths["candidate"])
    manifest_sha = sha256_file(paths["source_manifest"])
    build_sha = sha256_file(paths["build_result"])
    receipt_sha = sha256_file(paths["test_receipt"])
    tree = str(manifest.get("candidate_runtime_tree_sha256") or "")
    if (
        manifest.get("schema") != RELEASE_SCHEMA
        or manifest.get("run_id") != FIXED_RUN_ID
        or manifest.get("target_allocation") != TARGET_ALLOCATION
        or manifest.get("target_job_id") != TARGET_JOB_ID
        or manifest.get("baseline_zip_sha256") != BASELINE_ZIP_SHA256
        or manifest.get("baseline_runtime_tree_sha256") != BASELINE_TREE_SHA256
        or manifest.get("baseline_manifest_sha256") != BASELINE_MANIFEST_SHA256
        or manifest.get("candidate_runtime_file_count") != 32
        or manifest.get("changed_file_count") != 3
        or manifest.get("unchanged_file_count") != 29
        or manifest.get("changed_files") != CHANGED_FILES
        or manifest.get("production_deployable") is not False
        or manifest.get("production_deployed") is not False
        or manifest.get("hpc_accessed") is not False
        or manifest.get("gpu_touched") is not False
    ):
        raise ValueError("R123_SOURCE_MANIFEST_REJECTED")
    if (
        build.get("schema") != RELEASE_SCHEMA
        or build.get("status") != "built_and_verified"
        or build.get("zip_sha256") != candidate_sha
        or build.get("source_manifest_sha256") != manifest_sha
        or build.get("candidate_runtime_tree_sha256") != tree
        or not verified_rounds(build.get("verification_rounds"))
        or build.get("production_deployable") is not False
        or build.get("production_deployed") is not False
    ):
        raise ValueError("R123_BUILD_RESULT_REJECTED")
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("status") != "verified"
        or receipt.get("candidate_zip_sha256") != candidate_sha
        or receipt.get("source_manifest_sha256") != manifest_sha
        or receipt.get("build_result_sha256") != build_sha
        or receipt.get("candidate_runtime_tree_sha256") != tree
        or receipt.get("changed_files") != CHANGED_FILES
        or not verified_rounds(receipt.get("verification_rounds"))
        or receipt.get("production_deployable") is not False
        or receipt.get("production_deployed") is not False
        or receipt.get("hpc_accessed") is not False
        or receipt.get("gpu_touched") is not False
    ):
        raise ValueError("R123_TEST_RECEIPT_REJECTED")

    output = Path(args.output_dir).resolve()
    if output.exists():
        raise ValueError(f"R123_APPROVAL_OUTPUT_EXISTS:{output}")
    output.mkdir(parents=True)
    copied = {name: copy_bound(path, output) for name, path in paths.items()}
    created = datetime.now(timezone.utc)
    expires = created + timedelta(minutes=45)
    status = "validation_only_non_authorizing" if args.validate_only else "approved_for_single_runtime_transaction"
    approval = {
        "schema": SCHEMA,
        "status": status,
        "one_time": True,
        "deployment_id": deployment_id,
        "created_at": created.isoformat(),
        "expires_at": expires.isoformat(),
        "run_id": FIXED_RUN_ID,
        "target_allocation": TARGET_ALLOCATION,
        "target_job_id": TARGET_JOB_ID,
        "web_build_id": WEB_BUILD_ID,
        "preflight": {
            "run_status_required": "cancelled",
            "run_terminal_required": True,
            "effective_active_required": 0,
            "pending_approvals_required": 0,
            "current_runtime_tree_sha256": BASELINE_TREE_SHA256,
            "binding_state_required": "active",
            "profile_state_required": "active",
            "profile_tombstones_allowed": False,
        },
        "baseline": {
            "release": "r121-g21-hpc-asset-probe-r120-base",
            "zip_sha256": BASELINE_ZIP_SHA256,
            "runtime_tree_sha256": BASELINE_TREE_SHA256,
            "manifest_sha256": BASELINE_MANIFEST_SHA256,
        },
        "candidate": {
            "filename": copied["candidate"]["filename"],
            "bytes": copied["candidate"]["bytes"],
            "sha256": candidate_sha,
            "source_manifest_sha256": manifest_sha,
            "build_result_sha256": build_sha,
            "installed_runtime_tree_sha256": tree,
            "runtime_file_count": 32,
            "changed_file_count": 3,
            "unchanged_file_count": 29,
            "changed_files": list(CHANGED_FILES),
        },
        "verification": {
            "test_receipt_sha256": receipt_sha,
            "independent_rounds": 2,
            "py_compile_each_round": 32,
            "candidate_import_origin_verified": True,
            "long_history_idempotency_regression": True,
            "http_action_idempotency_route": True,
            "hpc_accessed": False,
            "gpu_touched": False,
        },
        "bound_inputs": copied,
        "deployer": {
            "wrapper_sha256": copied["wrapper"]["sha256"],
            "remote_runner_sha256": copied["remote_deployer"]["sha256"],
        },
        "transport": {"mode": "single_scp_envelope", "remote_host_alias": "evomind-shanghai"},
        "boundaries": {
            "runtime_only": True,
            "web_immutable": True,
            "goal_immutable": True,
            "g24_binding_immutable": True,
            "byoa_immutable": True,
            "profile_immutable": True,
            "hpc_access_forbidden": True,
            "training_forbidden": True,
            "weather_forbidden": True,
        },
        "production_deployable": not args.validate_only,
        "production_deployed": False,
    }
    approval_path = output / "evomind-runtime-r123-managed-hpc-identity-rebind-deployment-approval.json"
    approval_path.write_text(json.dumps(approval, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "schema": SCHEMA,
        "status": status,
        "deployment_id": deployment_id,
        "approval_filename": approval_path.name,
        "approval_sha256": sha256_file(approval_path),
        "candidate_sha256": candidate_sha,
        "source_manifest_sha256": manifest_sha,
        "build_result_sha256": build_sha,
        "test_receipt_sha256": receipt_sha,
        "candidate_runtime_tree_sha256": tree,
        "production_deployable": not args.validate_only,
        "production_deployed": False,
        "hpc_accessed": False,
        "gpu_touched": False,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Create the bound one-time R123 runtime deployment approval")
    value.add_argument("--deployment-id", required=True)
    value.add_argument("--candidate-zip", required=True)
    value.add_argument("--source-manifest", required=True)
    value.add_argument("--build-result", required=True)
    value.add_argument("--test-receipt", required=True)
    value.add_argument("--verifier", required=True)
    value.add_argument("--wrapper", required=True)
    value.add_argument("--remote-deployer", required=True)
    value.add_argument("--output-dir", required=True)
    value.add_argument("--validate-only", action="store_true")
    return value


def main() -> int:
    try:
        result = create(parser().parse_args())
    except Exception as exc:
        print(json.dumps({"schema": SCHEMA, "status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
