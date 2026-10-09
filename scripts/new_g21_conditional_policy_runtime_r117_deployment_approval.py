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


SCHEMA = "evomind.g21_conditional_policy_runtime_deployment_approval.r117.v1"
RELEASE_SCHEMA = "evomind.g21_conditional_policy_runtime.r117.v1"
RECEIPT_SCHEMA = "evomind.g21_conditional_policy_runtime_test_receipt.r117.v1"
FIXED_RUN_ID = "run_7b1efb878afb40f396db431e91f093a5"
FIXED_ALLOCATION = "G21"
FIXED_WEB_BUILD_ID = "overlay-controlled-secret-binding-r112-standalone-f38c40f2dbdc"
BASELINE_ZIP_SHA256 = "2aa31006f33bb53de56afe6888c9eb03bc8597fa5334e3831e6fd788fda877c2"
BASELINE_TREE_SHA256 = "7e7b0babec6146640dc56cb42c05bbe37765bca2aca6348f42c5bf569ab69b38"
OLD_SPEC_FILE_SHA256 = "380a3b2c037c8067a4618717cb7aaf8847d323d4be401a88d64009468132848f"
OLD_SPEC_CANONICAL_SHA256 = "a569bae731723fc7046817df737876d6dc7674eadd9f7b812fe7b4a87e521655"
NEW_SPEC_FILE_SHA256 = "da8c3fbe8b14107dfadf96ad906b4cf982bc714ebf6420225a1e3722cd41a752"
NEW_SPEC_CANONICAL_SHA256 = "fbb9939dff28c355c2798bf32fce104845337064f42f9c280f2b6906cccf4036"
POLICY_EVIDENCE_FILE_SHA256 = "47df860192502afcaacd7b1b6bc7fc4d86110c2e665278c8b7b3ea421893b7a7"
POLICY_EVIDENCE_CANONICAL_SHA256 = "f81701f13aeebef146659e091cacfc4e39ae18df345ee6b58be2ad5ef40829f0"
MIGRATION_ID = "g21-conditional-strong-baseline-v2"
CHANGED_FILES = {
    "evomind_runtime/competition_goal.py",
    "evomind_runtime/goal_board.py",
    "evomind_runtime/http_server.py",
    "evomind_runtime/runtime.py",
    "evomind_runtime/store.py",
}
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


def regular(path: Path, label: str) -> Path:
    value = path.resolve()
    if not value.is_file() or value.is_symlink():
        raise ValueError(f"{label}_REGULAR_FILE_REQUIRED")
    return value


def copy_bound(source: Path, output: Path) -> dict[str, Any]:
    destination = output / source.name
    if destination.exists():
        raise ValueError(f"APPROVAL_INPUT_COLLISION:{source.name}")
    shutil.copy2(source, destination)
    return {"filename": destination.name, "bytes": destination.stat().st_size, "sha256": sha256_file(destination)}


def assert_hash(value: Any, label: str) -> str:
    text = str(value)
    if not SHA_RE.fullmatch(text):
        raise ValueError(f"{label}_SHA_REQUIRED")
    return text


def build(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    candidate = regular(Path(args.candidate_zip), "CANDIDATE")
    manifest_path = regular(Path(args.source_manifest), "MANIFEST")
    build_result_path = regular(Path(args.build_result), "BUILD_RESULT")
    receipt_path = regular(Path(args.test_receipt), "TEST_RECEIPT")
    old_spec = regular(Path(args.old_spec), "OLD_SPEC")
    new_spec = regular(Path(args.new_spec), "NEW_SPEC")
    evidence = regular(Path(args.policy_evidence), "POLICY_EVIDENCE")
    verifier = regular(Path(args.verifier), "VERIFIER")
    wrapper = regular(Path(args.wrapper), "WRAPPER")
    runner = regular(Path(args.remote_runner), "REMOTE_RUNNER")
    manifest = load_object(manifest_path)
    build_result = load_object(build_result_path)
    receipt = load_object(receipt_path)
    candidate_sha = sha256_file(candidate)
    manifest_sha = sha256_file(manifest_path)
    build_result_sha = sha256_file(build_result_path)
    if (
        manifest.get("schema") != RELEASE_SCHEMA
        or manifest.get("run_id") != FIXED_RUN_ID
        or manifest.get("allocation") != FIXED_ALLOCATION
        or manifest.get("baseline_zip_sha256") != BASELINE_ZIP_SHA256
        or manifest.get("baseline_runtime_tree_sha256") != BASELINE_TREE_SHA256
        or int(manifest.get("candidate_runtime_file_count", -1)) != 32
        or int(manifest.get("changed_file_count", -1)) != 5
        or int(manifest.get("unchanged_file_count", -1)) != 27
        or set(manifest.get("changed_files", [])) != CHANGED_FILES
        or bool(manifest.get("production_deployable"))
        or bool(manifest.get("production_deployed"))
    ):
        raise ValueError("SOURCE_MANIFEST_REJECTED")
    candidate_tree = assert_hash(manifest.get("candidate_runtime_tree_sha256"), "CANDIDATE_TREE")
    if (
        build_result.get("schema") != RELEASE_SCHEMA
        or build_result.get("status") != "built_and_verified"
        or build_result.get("zip_sha256") != candidate_sha
        or build_result.get("source_manifest_sha256") != manifest_sha
        or build_result.get("candidate_runtime_tree_sha256") != candidate_tree
        or bool(build_result.get("production_deployable"))
        or bool(build_result.get("production_deployed"))
        or bool(build_result.get("hpc_accessed"))
        or bool(build_result.get("gpu_touched"))
        or int(build_result.get("remote_writes", -1)) != 0
    ):
        raise ValueError("BUILD_RESULT_REJECTED")
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("status") != "verified"
        or receipt.get("run_id") != FIXED_RUN_ID
        or receipt.get("allocation") != FIXED_ALLOCATION
        or receipt.get("candidate_zip_sha256") != candidate_sha
        or receipt.get("source_manifest_sha256") != manifest_sha
        or receipt.get("build_result_sha256") != build_result_sha
        or receipt.get("candidate_runtime_tree_sha256") != candidate_tree
        or receipt.get("verifier", {}).get("sha256") != sha256_file(verifier)
        or len(receipt.get("verification_rounds", [])) != 2
        or bool(receipt.get("production_deployed"))
        or bool(receipt.get("hpc_accessed"))
        or bool(receipt.get("gpu_touched"))
        or int(receipt.get("remote_writes", -1)) != 0
    ):
        raise ValueError("TEST_RECEIPT_REJECTED")
    bindings = {
        "old_goal_spec_file_sha256": OLD_SPEC_FILE_SHA256,
        "old_goal_spec_canonical_sha256": OLD_SPEC_CANONICAL_SHA256,
        "new_goal_spec_file_sha256": NEW_SPEC_FILE_SHA256,
        "new_goal_spec_canonical_sha256": NEW_SPEC_CANONICAL_SHA256,
        "policy_evidence_file_sha256": POLICY_EVIDENCE_FILE_SHA256,
        "policy_evidence_canonical_sha256": POLICY_EVIDENCE_CANONICAL_SHA256,
    }
    for key, value in bindings.items():
        if manifest.get(key) != value or receipt.get(key) != value:
            raise ValueError(f"POLICY_BINDING_REJECTED:{key}")
    for path, expected, label in (
        (old_spec, OLD_SPEC_FILE_SHA256, "OLD_SPEC"),
        (new_spec, NEW_SPEC_FILE_SHA256, "NEW_SPEC"),
        (evidence, POLICY_EVIDENCE_FILE_SHA256, "POLICY_EVIDENCE"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"{label}_SHA_REJECTED")
    now = datetime.now(timezone.utc)
    valid_minutes = int(args.valid_minutes)
    if valid_minutes < 5 or valid_minutes > 60:
        raise ValueError("VALID_MINUTES_REJECTED")
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise ValueError(f"OUTPUT_DIRECTORY_EXISTS:{output}")
    output.mkdir(parents=True)
    copied = {
        "candidate": copy_bound(candidate, output),
        "source_manifest": copy_bound(manifest_path, output),
        "build_result": copy_bound(build_result_path, output),
        "test_receipt": copy_bound(receipt_path, output),
        "old_spec": copy_bound(old_spec, output),
        "new_spec": copy_bound(new_spec, output),
        "policy_evidence": copy_bound(evidence, output),
        "verifier": copy_bound(verifier, output),
        "wrapper": copy_bound(wrapper, output),
        "remote_runner": copy_bound(runner, output),
    }
    approval = {
        "schema": SCHEMA,
        "status": "approved_for_single_runtime_transaction",
        "one_time": True,
        "deployment_id": uuid.uuid4().hex,
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(minutes=valid_minutes)).isoformat(),
        "run_id": FIXED_RUN_ID,
        "allocation": FIXED_ALLOCATION,
        "web_build_id": FIXED_WEB_BUILD_ID,
        "preflight": {
            "run_terminal_required": True,
            "effective_active_required": 0,
            "pending_approvals_required": 0,
            "current_runtime_file_count": 32,
            "current_runtime_tree_sha256": BASELINE_TREE_SHA256,
            "no_concurrent_transaction": True,
        },
        "candidate": {
            **copied["candidate"],
            "source_manifest_filename": copied["source_manifest"]["filename"],
            "source_manifest_sha256": copied["source_manifest"]["sha256"],
            "build_result_filename": copied["build_result"]["filename"],
            "build_result_sha256": copied["build_result"]["sha256"],
            "installed_runtime_tree_sha256": candidate_tree,
            "runtime_file_count": 32,
            "changed_file_count": 5,
            "unchanged_file_count": 27,
            "changed_files": sorted(CHANGED_FILES),
        },
        "baseline": {
            "release": "r116-g21-goal-r115-base-v2",
            "zip_sha256": BASELINE_ZIP_SHA256,
            "runtime_tree_sha256": BASELINE_TREE_SHA256,
            "runtime_file_count": 32,
        },
        "policy_migration": {
            "migration_id": MIGRATION_ID,
            "old_spec_filename": copied["old_spec"]["filename"],
            "old_spec_file_sha256": OLD_SPEC_FILE_SHA256,
            "old_spec_canonical_sha256": OLD_SPEC_CANONICAL_SHA256,
            "new_spec_filename": copied["new_spec"]["filename"],
            "new_spec_file_sha256": NEW_SPEC_FILE_SHA256,
            "new_spec_canonical_sha256": NEW_SPEC_CANONICAL_SHA256,
            "policy_evidence_filename": copied["policy_evidence"]["filename"],
            "policy_evidence_file_sha256": POLICY_EVIDENCE_FILE_SHA256,
            "policy_evidence_canonical_sha256": POLICY_EVIDENCE_CANONICAL_SHA256,
            "managed_put_only": True,
            "isolated_body_required": True,
            "idempotent_replay_required": True,
            "event_type": "goal.policy_migrated",
            "exact_event_count": 1,
        },
        "verification": {
            "test_receipt_filename": copied["test_receipt"]["filename"],
            "test_receipt_sha256": copied["test_receipt"]["sha256"],
            "candidate_verifier_filename": copied["verifier"]["filename"],
            "candidate_verifier_sha256": copied["verifier"]["sha256"],
        },
        "deployer": {
            "wrapper_filename": copied["wrapper"]["filename"],
            "wrapper_sha256": copied["wrapper"]["sha256"],
            "remote_runner_filename": copied["remote_runner"]["filename"],
            "remote_runner_sha256": copied["remote_runner"]["sha256"],
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
        "boundaries": {
            "runtime_only": True,
            "web_unchanged": True,
            "hpc_access_forbidden": True,
            "gpu_touch_forbidden": True,
            "weather_forbidden": True,
            "training_forbidden_during_deployment": True,
        },
    }
    path = output / "evomind-runtime-r117-g21-conditional-policy-deployment-approval.json"
    path.write_text(json.dumps(approval, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path, approval


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Create one fresh R117 conditional-policy deployment approval")
    value.add_argument("--candidate-zip", required=True)
    value.add_argument("--source-manifest", required=True)
    value.add_argument("--build-result", required=True)
    value.add_argument("--test-receipt", required=True)
    value.add_argument("--old-spec", required=True)
    value.add_argument("--new-spec", required=True)
    value.add_argument("--policy-evidence", required=True)
    value.add_argument("--verifier", required=True)
    value.add_argument("--wrapper", required=True)
    value.add_argument("--remote-runner", required=True)
    value.add_argument("--output-dir", required=True)
    value.add_argument("--valid-minutes", type=int, default=20)
    return value


def main() -> int:
    try:
        path, approval = build(parser().parse_args())
    except Exception as exc:
        print(json.dumps({"schema": SCHEMA, "status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "status": "created",
                "approval_filename": path.name,
                "approval_sha256": sha256_file(path),
                "expires_at": approval["expires_at"],
                "production_action_performed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
