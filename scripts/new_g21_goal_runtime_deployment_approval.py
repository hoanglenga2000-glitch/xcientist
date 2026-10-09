"""Create the single-use, hash-bound R116 Goal runtime deployment approval.

This command is local-only.  It validates a frozen R115-derived candidate and
copies the candidate, its receipts, and the three non-secret bootstrap evidence
documents into one deployment-input directory.  It never contacts production
or HPC and it never changes the candidate build result's fail-closed
``production_deployable=false`` declaration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "evomind.g21_goal_runtime_deployment_approval.v1"
MANIFEST_SCHEMA = "evomind.g21_goal_runtime_overlay.clean.v1"
TEST_RECEIPT_SCHEMA = "evomind.g21_goal_runtime_test_receipt.v1"
FIXED_RUN_ID = "run_7b1efb878afb40f396db431e91f093a5"
FIXED_ALLOCATION = "G21"
FIXED_WEB_BUILD_ID = "overlay-controlled-secret-binding-r112-standalone-f38c40f2dbdc"
R115_ZIP_SHA256 = "fc65d7134f9f53fa6eda21ee2fcd804a330d9184a83d5206a6b7469a217d89ba"
R115_RUNTIME_TREE_SHA256 = "b92cc96dae5ac1c1ed3641999eec56eda782b4959ac8709a0c0584ac1438955a"
GOAL_SPEC_FILE_SHA256 = "380a3b2c037c8067a4618717cb7aaf8847d323d4be401a88d64009468132848f"
HUMAN_BASELINE_FILE_SHA256 = "5b6ed3d912ad1cd4655c96f832629cf9603a4b0ad58ab6e27c01861fadd69c5f"
REJECTED_CANDIDATE_ZIP_SHA256 = "37929a1c9baf682e4f946450f834f10fe3b7a78323084ae96948bd0e92f8de8d"
REJECTION_EVIDENCE_SHA256 = "8900c32d41a47561715dc215ec2e560e338c2d164d6fa410554b7a2b83b5fb83"
GOAL_MANIFEST_NAME = "goal-runtime-overlay-manifest.json"
BASELINE_MANIFEST_NAME = "runtime-hotfix-manifest.json"
EXPECTED_CHANGED_FILES = {
    "evomind_runtime/__init__.py",
    "evomind_runtime/assistant_runs.py",
    "evomind_runtime/http_server.py",
    "evomind_runtime/runtime.py",
    "evomind_runtime/store.py",
    "evomind_runtime/competition_goal.py",
    "evomind_runtime/goal_board.py",
}
SHA_RE = re.compile(r"^[a-f0-9]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path.name}")
    return value


def safe_zip_entries(path: Path) -> list[zipfile.ZipInfo]:
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
    names: set[str] = set()
    for entry in entries:
        name = entry.filename.replace("\\", "/")
        pure = PurePosixPath(name)
        if (
            not name
            or name.startswith("/")
            or "\\" in entry.filename
            or pure.is_absolute()
            or any(part in {"", ".", ".."} for part in pure.parts)
            or re.match(r"^[A-Za-z]:", name)
        ):
            raise ValueError(f"CANDIDATE_ZIP_UNSAFE:{name}")
        key = name.casefold()
        if key in names:
            raise ValueError(f"CANDIDATE_ZIP_DUPLICATE:{name}")
        names.add(key)
        unix_mode = (entry.external_attr >> 16) & 0o170000
        if unix_mode == 0o120000:
            raise ValueError(f"CANDIDATE_ZIP_SYMLINK:{name}")
    return entries


def zip_member_bytes(path: Path, name: str) -> bytes:
    with zipfile.ZipFile(path) as archive:
        matches = [item for item in archive.infolist() if item.filename.replace("\\", "/") == name]
        if len(matches) != 1:
            raise ValueError(f"CANDIDATE_MEMBER_COUNT_REJECTED:{name}")
        return archive.read(matches[0])


def verify_manifest_and_candidate(candidate_zip: Path, source_manifest: Path, build_result: Path) -> dict[str, Any]:
    if sha256_file(candidate_zip) == REJECTED_CANDIDATE_ZIP_SHA256:
        raise ValueError("REJECTED_CANDIDATE_ZIP_SHA256")
    entries = safe_zip_entries(candidate_zip)
    names = {item.filename.replace("\\", "/") for item in entries if not item.is_dir()}
    if GOAL_MANIFEST_NAME not in names or BASELINE_MANIFEST_NAME in names:
        raise ValueError("CANDIDATE_MANIFEST_LAYOUT_REJECTED")
    embedded_raw = zip_member_bytes(candidate_zip, GOAL_MANIFEST_NAME)
    embedded_sha = hashlib.sha256(embedded_raw).hexdigest()
    manifest = json.loads(embedded_raw.decode("utf-8-sig"))
    external_manifest = load_object(source_manifest)
    if external_manifest != manifest or sha256_file(source_manifest) != embedded_sha:
        raise ValueError("CANDIDATE_EXTERNAL_MANIFEST_MISMATCH")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("run_id") != FIXED_RUN_ID
        or str(manifest.get("allocation") or "").casefold() != FIXED_ALLOCATION.casefold()
        or manifest.get("target") != "bundle/runtime/evomind_runtime"
        or manifest.get("baseline_zip_sha256") != R115_ZIP_SHA256
        or manifest.get("baseline_runtime_file_count") != 30
        or manifest.get("candidate_runtime_file_count") != 32
        or manifest.get("file_count") != 32
        or manifest.get("changed_file_count") != 7
        or set(manifest.get("changed_files") or []) != EXPECTED_CHANGED_FILES
        or manifest.get("production_deployable") is not False
        or manifest.get("production_deployed") is not False
        or manifest.get("reviewed_goal_only") is not True
        or manifest.get("hpc_accessed") is not False
        or manifest.get("gpu_touched") is not False
        or manifest.get("remote_writes") != 0
    ):
        raise ValueError("CANDIDATE_MANIFEST_CONTRACT_REJECTED")
    listed: dict[str, dict[str, Any]] = {}
    for item in manifest.get("files") or []:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("CANDIDATE_MANIFEST_ENTRY_REJECTED")
        relative = item["path"]
        if relative in listed or not SHA_RE.fullmatch(str(item.get("sha256") or "")):
            raise ValueError("CANDIDATE_MANIFEST_ENTRY_REJECTED")
        listed[relative] = item
    payload_names = names - {GOAL_MANIFEST_NAME}
    if set(listed) != payload_names or len([name for name in payload_names if name.endswith(".py")]) != 32:
        raise ValueError("CANDIDATE_FILESET_REJECTED")
    for name, item in listed.items():
        raw = zip_member_bytes(candidate_zip, name)
        if len(raw) != int(item.get("bytes", -1)) or hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError(f"CANDIDATE_FILE_HASH_REJECTED:{name}")
    result = load_object(build_result)
    rounds = result.get("verification_rounds")
    if (
        result.get("schema") != MANIFEST_SCHEMA
        or result.get("status") != "built_and_verified"
        or result.get("zip_sha256") != sha256_file(candidate_zip)
        or result.get("manifest_sha256") != embedded_sha
        or result.get("baseline_zip_sha256") != R115_ZIP_SHA256
        or result.get("candidate_runtime_file_count") != 32
        or result.get("changed_file_count") != 7
        or not isinstance(rounds, list)
        or len(rounds) != 2
        or any(not isinstance(item, dict) or item.get("status") != "passed" for item in rounds)
        or result.get("production_deployable") is not False
        or result.get("production_deployed") is not False
        or result.get("hpc_accessed") is not False
        or result.get("gpu_touched") is not False
        or result.get("remote_writes") != 0
    ):
        raise ValueError("CANDIDATE_BUILD_RESULT_REJECTED")
    runtime_rows = []
    for name, item in sorted(listed.items()):
        if not name.startswith("evomind_runtime/"):
            raise ValueError(f"CANDIDATE_TARGET_SCOPE_REJECTED:{name}")
        relative = name.removeprefix("evomind_runtime/")
        runtime_rows.append(f"{relative}|{int(item['bytes'])}|{item['sha256']}")
    runtime_tree_sha = hashlib.sha256("\n".join(runtime_rows).encode("utf-8")).hexdigest()
    if manifest.get("candidate_runtime_tree_sha256") != runtime_tree_sha:
        raise ValueError("CANDIDATE_PRODUCTION_TREE_HASH_REJECTED")
    if result.get("candidate_runtime_tree_sha256") != runtime_tree_sha:
        raise ValueError("CANDIDATE_BUILD_RESULT_TREE_HASH_REJECTED")
    return {
        "manifest": manifest,
        "embedded_manifest_sha256": embedded_sha,
        "runtime_tree_sha256": runtime_tree_sha,
        "zip_sha256": sha256_file(candidate_zip),
        "source_manifest_sha256": sha256_file(source_manifest),
        "build_result_sha256": sha256_file(build_result),
    }


def verify_bootstrap(
    spec_path: Path, human_path: Path, board_path: Path,
    bootstrap_artifact_path: Path, manifest: dict[str, Any], build_result: dict[str, Any],
) -> dict[str, str]:
    spec = load_object(spec_path)
    human = load_object(human_path)
    board = load_object(board_path)
    bootstrap_artifact = load_object(bootstrap_artifact_path)
    spec_raw = sha256_file(spec_path)
    human_raw = sha256_file(human_path)
    board_raw = sha256_file(board_path)
    if spec_raw != GOAL_SPEC_FILE_SHA256 or human_raw != HUMAN_BASELINE_FILE_SHA256:
        raise ValueError("BOOTSTRAP_FIXED_EVIDENCE_HASH_REJECTED")
    if (
        spec.get("schema") != "evomind.five_competition_goal.v1"
        or spec.get("run_id") != FIXED_RUN_ID
        or spec.get("allocation") != FIXED_ALLOCATION
        or spec.get("human_baseline_evidence_sha256") != human_raw
        or human.get("schema") != "evomind.five_competition_human_baseline_audit.v1"
        or human.get("run_id") != FIXED_RUN_ID
        or human.get("allocation") != FIXED_ALLOCATION
        or human.get("classification") != "HUMAN_BASELINE_UNDEFINED"
        or human.get("completion", {}).get("training_authorized_by_this_artifact") is not False
        or board.get("schema") != "evomind.goal-board.v2"
        or board.get("run_id") != FIXED_RUN_ID
        or board.get("allocation") != FIXED_ALLOCATION
        or board.get("goal_record_status") != "blocked"
        or board.get("completion_count") != 0
        or board.get("weather_actions") != 0
        or {item.get("competition") for item in board.get("competitions") or []}
        != {"cure_bench", "e2lmc", "mindgames", "open_polymer", "ariel_2025"}
    ):
        raise ValueError("BOOTSTRAP_EVIDENCE_CONTRACT_REJECTED")
    if manifest.get("goal_spec_sha256") != spec_raw or manifest.get("human_baseline_evidence_sha256") != human_raw:
        raise ValueError("CANDIDATE_BOOTSTRAP_HASH_REJECTED")
    if manifest.get("goal_board_file_sha256") not in (None, board_raw):
        raise ValueError("CANDIDATE_BOARD_FILE_HASH_REJECTED")
    if manifest.get("goal_board_sha256") not in (None, canonical_sha256(board)):
        raise ValueError("CANDIDATE_BOARD_CANONICAL_HASH_REJECTED")
    bootstrap_artifact_sha = sha256_file(bootstrap_artifact_path)
    if (
        bootstrap_artifact.get("schema") != "evomind.g21_goal_runtime_bootstrap.v1"
        or bootstrap_artifact.get("run_id") != FIXED_RUN_ID
        or bootstrap_artifact.get("allocation") != FIXED_ALLOCATION
        or bootstrap_artifact.get("goal_record_status") != "blocked"
        or bootstrap_artifact.get("goal_spec") != spec
        or bootstrap_artifact.get("human_baseline_evidence") != human
        or bootstrap_artifact.get("initial_goal_board") != board
        or bootstrap_artifact.get("goal_spec_file_sha256") != spec_raw
        or bootstrap_artifact.get("human_baseline_evidence_file_sha256") != human_raw
        or bootstrap_artifact.get("goal_board_file_sha256") != board_raw
        or bootstrap_artifact.get("goal_board_sha256") != canonical_sha256(board)
        or bootstrap_artifact.get("production_deployable") is not False
        or manifest.get("bootstrap_artifact_sha256") != bootstrap_artifact_sha
        or build_result.get("bootstrap_artifact_sha256") != bootstrap_artifact_sha
    ):
        raise ValueError("BOOTSTRAP_ARTIFACT_CONTRACT_REJECTED")
    return {
        "goal_spec_file_sha256": spec_raw,
        "goal_spec_sha256": canonical_sha256(spec),
        "human_baseline_file_sha256": human_raw,
        "goal_board_file_sha256": board_raw,
        "goal_board_sha256": canonical_sha256(board),
        "bootstrap_artifact_sha256": bootstrap_artifact_sha,
    }


def verify_test_receipt(
    path: Path, *, candidate_zip_sha256: str, manifest_sha256: str,
    bootstrap_artifact_sha256: str, verifier_sha256: str,
) -> dict[str, Any]:
    value = load_object(path)
    checks = value.get("checks")
    test_artifacts = value.get("test_artifacts")
    targeted = test_artifacts.get("targeted_goal_builder_junit") if isinstance(test_artifacts, dict) else None
    regression = test_artifacts.get("runtime_regression_junit") if isinstance(test_artifacts, dict) else None
    try:
        generated_at = datetime.fromisoformat(str(value.get("generated_at_utc") or "").replace("Z", "+00:00"))
        if generated_at.tzinfo is None:
            raise ValueError
    except ValueError as exc:
        raise ValueError(f"TEST_RECEIPT_TIME_REJECTED:{path.name}") from exc
    age = datetime.now(timezone.utc) - generated_at.astimezone(timezone.utc)
    if (
        value.get("schema") != TEST_RECEIPT_SCHEMA
        or value.get("status") != "passed"
        or value.get("run_id") != FIXED_RUN_ID
        or value.get("allocation") != FIXED_ALLOCATION
        or value.get("candidate_zip_sha256") != candidate_zip_sha256
        or value.get("source_manifest_sha256") != manifest_sha256
        or not isinstance(checks, dict)
        or checks.get("dual_extract_rounds") != 2
        or checks.get("python_files_compiled") != 32
        or checks.get("goal_http_smoke_rounds") != 2
        or checks.get("baseline_unchanged_files") != 25
        or checks.get("changed_files") != 7
        or checks.get("unsafe_entries") != 0
        or checks.get("duplicate_entries") != 0
        or checks.get("symlinks") != 0
        or checks.get("secret_scan") != "passed"
        or checks.get("git_diff_check") != "passed"
        or checks.get("hpc_accessed") is not False
        or checks.get("gpu_touched") is not False
        or checks.get("remote_writes") != 0
        or not isinstance(targeted, dict) or int(targeted.get("tests", 0)) < 55
        or int(targeted.get("failures", -1)) != 0 or int(targeted.get("errors", -1)) != 0
        or not SHA_RE.fullmatch(str(targeted.get("sha256") or ""))
        or not {
            "tests.test_g21_goal_runtime_overlay_clean", "tests.test_goal_board_http",
            "tests.test_goal_board_persistence", "tests.test_competition_goal",
            "tests.test_goal_release_scope", "tests.test_g21_goal_patch_scope",
        } <= set(targeted.get("required_classnames") or [])
        or not isinstance(regression, dict) or int(regression.get("tests", 0)) < 112
        or int(regression.get("failures", -1)) != 0 or int(regression.get("errors", -1)) != 0
        or not SHA_RE.fullmatch(str(regression.get("sha256") or ""))
        or not {
            "tests.test_evomind_runtime", "tests.test_evomind_runtime_http_server",
            "tests.test_assistant_run_service", "tests.test_large_tool_result_artifacts",
        } <= set(regression.get("required_classnames") or [])
        or value.get("bootstrap_artifact_sha256") != bootstrap_artifact_sha256
        or (value.get("verifier") or {}).get("filename") != "verify_g21_goal_runtime_r116_candidate.py"
        or (value.get("verifier") or {}).get("sha256") != verifier_sha256
        or age.total_seconds() < -300 or age > timedelta(hours=24)
    ):
        raise ValueError(f"TEST_RECEIPT_REJECTED:{path.name}")
    return value


def copy_input(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    if destination.exists():
        raise FileExistsError(f"APPROVAL_INPUT_EXISTS:{destination}")
    shutil.copy2(source, destination)


def build_approval(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        key: Path(getattr(args, key)).resolve(strict=True)
        for key in (
            "candidate_zip", "source_manifest", "build_result", "goal_spec",
            "human_baseline", "goal_board", "bootstrap_artifact", "rejection_evidence",
            "candidate_verifier", "wrapper", "remote_runner",
        )
    }
    baseline_zip = Path(args.baseline_zip).resolve(strict=True)
    if sha256_file(baseline_zip) != R115_ZIP_SHA256:
        raise ValueError("R115_BASELINE_ZIP_HASH_REJECTED")
    rejection = load_object(paths["rejection_evidence"])
    rejected_files = rejection.get("candidate_files") or []
    if (
        sha256_file(paths["rejection_evidence"]) != REJECTION_EVIDENCE_SHA256
        or rejection.get("schema") != "evomind.g21_goal_runtime_candidate_rejection.v1"
        or rejection.get("status") != "NOT_DEPLOYABLE"
        or rejection.get("run_id") != FIXED_RUN_ID
        or not any(item.get("sha256") == REJECTED_CANDIDATE_ZIP_SHA256 for item in rejected_files if isinstance(item, dict))
        or rejection.get("production_deployed") is not False
    ):
        raise ValueError("REJECTED_CANDIDATE_EVIDENCE_REJECTED")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"APPROVAL_OUTPUT_EXISTS:{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    candidate = verify_manifest_and_candidate(paths["candidate_zip"], paths["source_manifest"], paths["build_result"])
    bootstrap = verify_bootstrap(
        paths["goal_spec"], paths["human_baseline"], paths["goal_board"],
        paths["bootstrap_artifact"], candidate["manifest"], load_object(paths["build_result"]),
    )
    receipt_paths = [Path(value).resolve(strict=True) for value in args.test_receipt]
    if not receipt_paths:
        raise ValueError("INDEPENDENT_TEST_RECEIPT_REQUIRED")
    for path in receipt_paths:
        verify_test_receipt(
            path,
            candidate_zip_sha256=candidate["zip_sha256"],
            manifest_sha256=candidate["source_manifest_sha256"],
            bootstrap_artifact_sha256=bootstrap["bootstrap_artifact_sha256"],
            verifier_sha256=sha256_file(paths["candidate_verifier"]),
        )
    deployment_id = (args.deployment_id or uuid.uuid4().hex).lower()
    if not re.fullmatch(r"[a-f0-9]{32}", deployment_id):
        raise ValueError("DEPLOYMENT_ID_REJECTED")
    now = datetime.now(timezone.utc)
    expires = now + timedelta(hours=args.valid_hours)
    copied_names = {
        "candidate_zip": "runtime.zip",
        "source_manifest": "source-manifest.json",
        "build_result": "build-result.json",
        "goal_spec": "goal-spec.json",
        "human_baseline": "human-baseline.json",
        "goal_board": "goal-board.json",
        "bootstrap_artifact": "bootstrap.json",
        "rejection_evidence": "rejected-candidate.json",
    }
    for key, filename in copied_names.items():
        copy_input(paths[key], output.parent / filename)
    receipts = []
    for index, path in enumerate(receipt_paths, 1):
        filename = f"test-receipt-{index}.json"
        copy_input(path, output.parent / filename)
        receipts.append({"filename": filename, "sha256": sha256_file(path), "schema": TEST_RECEIPT_SCHEMA})
    approval = {
        "schema": SCHEMA,
        "status": "approved",
        "single_use": True,
        "deployment_id": deployment_id,
        "issued_at_utc": now.isoformat(),
        "expires_at_utc": expires.isoformat(),
        "scope": {
            "mode": "runtime_only",
            "run_id": FIXED_RUN_ID,
            "allocation": FIXED_ALLOCATION,
            "expected_web_build_id": FIXED_WEB_BUILD_ID,
            "expected_current_runtime_tree_sha256": R115_RUNTIME_TREE_SHA256,
            "hpc_accessed": False,
            "gpu_touched": False,
        },
        "baseline": {
            "release": "r115-progress-parser-r114-base",
            "zip_sha256": R115_ZIP_SHA256,
            "runtime_tree_sha256": R115_RUNTIME_TREE_SHA256,
            "runtime_file_count": 30,
        },
        "candidate": {
            "release": "r116-g21-goal-r115-base",
            "zip_filename": copied_names["candidate_zip"],
            "zip_sha256": candidate["zip_sha256"],
            "source_manifest_filename": copied_names["source_manifest"],
            "source_manifest_sha256": candidate["source_manifest_sha256"],
            "build_result_filename": copied_names["build_result"],
            "build_result_sha256": candidate["build_result_sha256"],
            "embedded_manifest_filename": GOAL_MANIFEST_NAME,
            "embedded_manifest_sha256": candidate["embedded_manifest_sha256"],
            "installed_runtime_tree_sha256": candidate["runtime_tree_sha256"],
            "runtime_file_count": 32,
            "changed_file_count": 7,
        },
        "bootstrap": {
            "goal_spec_filename": copied_names["goal_spec"],
            "goal_spec_file_sha256": bootstrap["goal_spec_file_sha256"],
            "goal_spec_sha256": bootstrap["goal_spec_sha256"],
            "human_baseline_filename": copied_names["human_baseline"],
            "human_baseline_file_sha256": bootstrap["human_baseline_file_sha256"],
            "goal_board_filename": copied_names["goal_board"],
            "goal_board_file_sha256": bootstrap["goal_board_file_sha256"],
            "goal_board_sha256": bootstrap["goal_board_sha256"],
            "artifact_filename": copied_names["bootstrap_artifact"],
            "artifact_sha256": bootstrap["bootstrap_artifact_sha256"],
            "goal_record_status": "blocked",
        },
        "test_receipts": receipts,
        "deployer": {
            "wrapper_filename": paths["wrapper"].name,
            "wrapper_sha256": sha256_file(paths["wrapper"]),
            "remote_runner_filename": paths["remote_runner"].name,
            "remote_runner_sha256": sha256_file(paths["remote_runner"]),
            "service_action_path": r"C:\SecureInput\Invoke-ServiceAccountAction.ps1",
        },
        "verification": {
            "candidate_verifier_filename": paths["candidate_verifier"].name,
            "candidate_verifier_sha256": sha256_file(paths["candidate_verifier"]),
        },
        "rejected_candidate": {
            "zip_sha256": REJECTED_CANDIDATE_ZIP_SHA256,
            "evidence_filename": copied_names["rejection_evidence"],
            "evidence_sha256": REJECTION_EVIDENCE_SHA256,
            "status": "NOT_DEPLOYABLE",
        },
        "prohibited": {
            "web_replacement": True,
            "direct_scheduled_task_start": True,
            "process_kill": True,
            "direct_sqlite_write": True,
            "hpc_access": True,
            "chrome": True,
            "training": True,
        },
    }
    output.write_text(json.dumps(approval, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {"approval_path": str(output), "approval_sha256": sha256_file(output), "deployment_id": deployment_id}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-zip", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--build-result", required=True)
    parser.add_argument("--test-receipt", action="append", default=[])
    parser.add_argument("--baseline-zip", default=r"D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\evomind-runtime-r115-progress-parser-r114-base-20260829.zip")
    parser.add_argument("--goal-spec", default=str(root / "configs" / "g21_five_competition_goal.json"))
    parser.add_argument("--human-baseline", default=str(root / "configs" / "g21_five_competition_human_baseline_gate.json"))
    parser.add_argument("--goal-board", default=str(root / "configs" / "g21_five_competition_goal_board_bootstrap.json"))
    parser.add_argument("--bootstrap-artifact", required=True)
    parser.add_argument("--rejection-evidence", default=r"D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\r116-g21-goal-r115-base-20260831\NOT_DEPLOYABLE-37929a1c.json")
    parser.add_argument("--wrapper", default=str(root / "scripts" / "Deploy-G21GoalRuntimeR116.ps1"))
    parser.add_argument("--remote-runner", default=str(root / "scripts" / "Deploy-G21GoalRuntimeR116Remote.ps1"))
    parser.add_argument("--candidate-verifier", default=str(root / "scripts" / "verify_g21_goal_runtime_r116_candidate.py"))
    parser.add_argument("--deployment-id", default="")
    parser.add_argument("--valid-hours", type=int, default=6, choices=range(1, 25))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(build_approval(args), ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
