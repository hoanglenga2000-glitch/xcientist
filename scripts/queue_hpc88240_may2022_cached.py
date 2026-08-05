#!/usr/bin/env python3
"""Queue the cache-backed May-2022 candidate after the active A40 chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tarfile
import time
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts import mlebench_remote_ops as ops  # noqa: E402

DEFAULT_PLAN = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "may2022_nested_selection_execution_multiseed_hpc88240_v8_cache_taxi_route_20260728.json"
)
DEFAULT_EVIDENCE_DIR = (
    PROJECT_ROOT
    / "workspace"
    / "hpc"
    / "job88240_may2022_nested_queue_v8_multiseed_cache_taxi_route"
)
READY_DEPENDENCY_STATUSES = frozenset(
    {
        "verification_passed",
        "verification_complete_gate_failed",
        "v1_candidate_preserved",
        "target_candidate_exists",
    }
)
FAILED_DEPENDENCY_STATUSES = frozenset(
    {
        "blocked_by_ranzcr_integrity_failure",
        "frozen_artifact_missing",
        "frozen_artifact_drift",
        "target_run_exists",
        "aggregation_failed",
        "verifier_failed",
        "verification_contract_failed",
    }
)


class MayQueueError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MayQueueError(message)


def _bundle_manifest(bundle: Path) -> dict[str, Any]:
    with tarfile.open(bundle, "r:gz") as archive:
        member = archive.getmember("bundle_manifest.json")
        handle = archive.extractfile(member)
        _require(handle is not None, "May queue bundle manifest is unreadable")
        return json.loads(handle.read().decode("utf-8"))


def validate_execution_plan(path: Path = DEFAULT_PLAN) -> dict[str, Any]:
    plan_path = Path(path).resolve()
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    _require(
        payload.get("schema")
        == "evomind.mlebench.may2022_nested_selection_execution_plan.v1",
        "Unexpected May queue execution-plan schema",
    )
    _require(
        payload.get("status") == "approved_waiting_for_hpc_gpu_and_serial_queue",
        "May queue execution plan is not in the approved waiting state",
    )
    _require(
        payload.get("competition_id") == "tabular-playground-series-may-2022",
        "Unexpected May queue competition",
    )
    _require(payload.get("seeds") == [42, 43, 44], "May queue confirmation seeds changed")
    _require(
        len(payload.get("run_ids") or []) == 3
        and len(set(payload["run_ids"])) == 3
        and payload.get("run_id") == payload["run_ids"][0],
        "May queue multiseed run IDs are invalid",
    )
    multiseed = payload.get("multi_seed_execution") or {}
    _require(
        multiseed.get("required") is True
        and multiseed.get("all_seeds_terminal_before_successor") is True
        and multiseed.get("confirmation_seeds") == payload["seeds"]
        and multiseed.get("run_ids") == payload["run_ids"],
        "May queue multiseed execution contract changed",
    )
    authorization = payload.get("authorization") or {}
    _require(
        authorization.get("full_training_approved") is True
        and authorization.get("substantial_compute_approval_recorded") is True
        and authorization.get("candidate_generation_approved") is True,
        "May queue full candidate compute was not approved",
    )
    _require(
        authorization.get("official_private_grader_approved") is False
        and authorization.get("kaggle_submission_approved") is False,
        "May queue grading boundary changed",
    )
    target = payload.get("target") or {}
    _require(target.get("job_id") == 88240, "May queue target job changed")
    _require(target.get("gpu") == "NVIDIA A40", "May queue target GPU changed")
    _require(
        target.get("remote_root") == ops.ALLOWED_GPU_REMOTE_ROOT,
        "May queue remote root changed",
    )
    contract = payload.get("execution_contract") or {}
    _require(
        contract.get("candidate_only") is True
        and contract.get("private_labels_used") is False
        and contract.get("official_grader_executed") is False
        and contract.get("kaggle_submission_executed") is False
        and contract.get("process_signals_allowed") is False
        and contract.get("other_processes_may_be_modified") is False,
        "May queue execution contract changed",
    )
    parent = Path(str((payload.get("parent_plan") or {}).get("path", ""))).resolve()
    _require(parent.is_file(), "May queue parent plan is missing")
    parent_hash = sha256_file(parent)
    _require(
        parent_hash == payload.get("parent_plan_sha256")
        == (payload.get("parent_plan") or {}).get("sha256"),
        "May queue parent plan hash drifted",
    )
    source_records = payload.get("source_identity") or []
    _require(len(source_records) >= 7, "May queue source identity is incomplete")
    for record in source_records:
        local_path = Path(str(record.get("path", ""))).resolve()
        try:
            local_path.relative_to(PROJECT_ROOT.resolve())
        except ValueError as exc:
            raise MayQueueError("May queue source escaped the project root") from exc
        _require(local_path.is_file(), "May queue source is missing")
        _require(
            local_path.stat().st_size == int(record.get("bytes", -1))
            and sha256_file(local_path) == record.get("sha256"),
            f"May queue source hash drifted: {record.get('relative_path')}",
        )
    cache = payload.get("public_precomputed_cache") or {}
    _require(
        cache.get("required") is True
        and cache.get("status") == "completed"
        and cache.get("visibility_mode") == "PUBLIC_ONLY"
        and cache.get("private_labels_used") is False
        and cache.get("official_grader_executed") is False
        and cache.get("kaggle_submission_executed") is False,
        "May queue public-cache contract changed",
    )
    cache_path = ops.ensure_remote_path(str(cache.get("path") or ""))
    _require(
        cache_path.startswith(ops.REMOTE_MAY2022_CACHE_ROOT + "/")
        and PurePosixPath(cache_path).name == "cache",
        "May queue public-cache path changed",
    )
    cache_manifest_sha = str(cache.get("manifest_sha256") or "")
    _require(
        len(cache_manifest_sha) == 64
        and all(character in "0123456789abcdef" for character in cache_manifest_sha),
        "May queue public-cache manifest hash is invalid",
    )
    cache_collection = cache.get("collection_evidence") or {}
    collection_path = Path(str(cache_collection.get("path") or "")).resolve()
    _require(
        collection_path.is_file()
        and collection_path.stat().st_size == int(cache_collection.get("bytes", -1))
        and sha256_file(collection_path) == cache_collection.get("sha256"),
        "May queue public-cache collection evidence drifted",
    )
    serial = payload.get("serial_queue") or {}
    dependency_path = str(serial.get("dependency_status_path", ""))
    try:
        PurePosixPath(dependency_path).relative_to(
            PurePosixPath(ops.ALLOWED_GPU_REMOTE_ROOT)
        )
    except ValueError as exc:
        raise MayQueueError("May queue dependency path escaped the remote root") from exc
    _require(
        dependency_path.endswith("/full_cactus_status_v2.json")
        and serial.get("launch_only_after_dependency_terminal_and_gpu_idle") is True
        and serial.get("natural_retirement_only") is True,
        "May queue serial dependency contract changed",
    )
    argv = list(payload.get("launch_argv_template") or [])
    _require(argv.count("--candidate-only") == 1, "May queue candidate-only flag is missing")
    _require(
        argv.count("--may-precomputed-cache-dir") == 1
        and argv.count("--may-require-precomputed-cache") == 1,
        "May queue verified-cache flags are missing",
    )
    _require(
        not any("kaggle" in value.lower() or "grader" in value.lower() for value in argv),
        "May queue launch template contains a grading or submission command",
    )

    bundle_verification = ops.verify_local_bundle(ops.DEFAULT_BUNDLE)
    manifest = _bundle_manifest(ops.DEFAULT_BUNDLE)
    files = manifest.get("files") or {}
    plan_relative = (
        "plans/may2022_nested_selection_execution_multiseed_hpc88240_v8_cache_taxi_route_20260728.json"
    )
    _require(
        files.get(plan_relative) == sha256_file(plan_path),
        "May queue bundle does not bind the execution plan",
    )
    for record in source_records:
        relative = str(record["relative_path"])
        if relative in files:
            _require(
                files[relative] == record["sha256"],
                f"May queue bundle source differs: {relative}",
            )
    _require(
        manifest.get("may2022_public_cache_manifest_sha256") == cache_manifest_sha
        and manifest.get("may2022_cached_execution_plan_sha256") == sha256_file(plan_path)
        and cache.get("feature_builder_sha256")
        == files.get("scripts/mlebench_medal_recovery_adapters.py"),
        "May queue bundle/cache binding differs",
    )
    payload["_path"] = str(plan_path)
    payload["_sha256"] = sha256_file(plan_path)
    payload["_bundle_verification"] = bundle_verification
    return payload


def classify_dependency(payload: Mapping[str, Any] | None) -> str:
    if payload is None:
        return "waiting"
    if payload.get("schema") != "evomind.hpc_cactus_persistent_run.v2":
        return "failed"
    if (
        payload.get("process_signals_sent") != 0
        or payload.get("private_labels_used") is not False
        or payload.get("official_grader_executed") is not False
        or payload.get("kaggle_submission_executed") is not False
    ):
        return "failed"
    status_value = str(payload.get("status", ""))
    if status_value in READY_DEPENDENCY_STATUSES:
        return "ready"
    if status_value in FAILED_DEPENDENCY_STATUSES or (
        payload.get("exit_code") not in {None, 0}
    ):
        return "failed"
    return "waiting"


def read_remote_dependency(path: str) -> dict[str, Any] | None:
    client = ops._connect()
    try:
        with client.open_sftp() as sftp:
            try:
                item = sftp.lstat(path)
            except OSError:
                return None
            _require(
                stat.S_ISREG(item.st_mode) and not stat.S_ISLNK(item.st_mode),
                "May queue dependency evidence is unsafe",
            )
            _require(int(item.st_size) <= 1024 * 1024, "May queue dependency is too large")
            with sftp.open(path, "rb") as handle:
                return json.loads(handle.read().decode("utf-8"))
    finally:
        client.close()


def acquire_claim(path: Path, payload: Mapping[str, Any]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return True


def _status_base(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "evomind.hpc88240.may2022_cached_queue.v1",
        "created_at": now_iso(),
        "plan_path": plan["_path"],
        "plan_sha256": plan["_sha256"],
        "bundle": plan["_bundle_verification"],
        "process_signals_sent": 0,
        "other_processes_modified": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }


def remote_run_terminal(status: Mapping[str, Any]) -> bool:
    process = status.get("process")
    checkpoint = status.get("checkpoint") if isinstance(status.get("checkpoint"), dict) else {}
    summary = status.get("summary") if isinstance(status.get("summary"), dict) else {}
    completed = checkpoint.get("completed") if isinstance(checkpoint.get("completed"), dict) else {}
    return bool(
        process == "stopped"
        and (
            completed
            or summary.get("status") in {"candidate_complete", "partial_failure", "completed"}
            or summary.get("competition_count") == 1
        )
    )


def run_ready_launch(
    plan: Mapping[str, Any],
    *,
    evidence_dir: Path,
    dependency: Mapping[str, Any],
) -> dict[str, Any]:
    gate_path = evidence_dir / "gpu_gate_current.json"
    status_path = evidence_dir / "status_current.json"
    completed: list[dict[str, Any]] = []
    for seed, run_id in zip(plan["seeds"], plan["run_ids"], strict=True):
        try:
            remote = ops.read_remote_status(str(run_id))
        except (ops.RemoteOpsError, OSError, ValueError, json.JSONDecodeError):
            remote = None
        if remote is not None and remote_run_terminal(remote):
            completed.append({"seed": seed, "run_id": run_id, "status": remote})
            continue
        if remote is not None and remote.get("process") == "running":
            result = {
                **_status_base(plan),
                "status": "seed_active",
                "dependency": dict(dependency),
                "completed_seeds": completed,
                "active_seed": {"seed": seed, "run_id": run_id, "status": remote},
            }
            write_json_atomic(status_path, result)
            return result

        first_gate = ops.sample_gpu_idle_gate(interval_seconds=5)
        write_json_atomic(gate_path, first_gate)
        if first_gate.get("passed") is not True:
            result = {
                **_status_base(plan),
                "status": "waiting_for_gpu_idle",
                "dependency": dict(dependency),
                "completed_seeds": completed,
                "next_seed": seed,
                "gpu_gate": first_gate,
            }
            write_json_atomic(status_path, result)
            return result
        claim_path = evidence_dir / f"launch_claim_s{seed}.json"
        claim = {
            "schema": "evomind.hpc88240.may2022_cached_launch_claim.v1",
            "created_at": now_iso(),
            "run_id": run_id,
            "seed": seed,
            "plan_sha256": plan["_sha256"],
            "bundle_sha256": plan["_bundle_verification"]["sha256"],
            "process_signals_sent": 0,
        }
        if not acquire_claim(claim_path, claim):
            result = {
                **_status_base(plan),
                "status": "launch_claim_already_exists",
                "seed": seed,
                "run_id": run_id,
                "launch_claim": str(claim_path.resolve()),
            }
            write_json_atomic(status_path, result)
            return result
        deployment = ops.deploy_bundle(ops.DEFAULT_BUNDLE, gate_path, max_gate_age=600)
        cuda_smoke = ops.cuda_smoke(ops.DEFAULT_BUNDLE, gate_path, max_gate_age=600)
        final_gate = ops.sample_gpu_idle_gate(interval_seconds=5)
        write_json_atomic(gate_path, final_gate)
        _require(final_gate.get("passed") is True, "May queue final GPU gate did not pass")
        start = ops.start_run(
            ops.DEFAULT_BUNDLE,
            gate_path,
            run_id=str(run_id),
            waves=["Wave0"],
            competitions=["tabular-playground-series-may-2022"],
            seed=int(seed),
            optimization_plan_name=Path(str(plan["_path"])).name,
            resume=False,
            allow_concurrent_with_cpu_light=False,
            max_gate_age=600,
            runner_performance_overrides=["--may-mlp-batch-size", "4096"],
            runner_contract_args=[
                "--may-precomputed-cache-dir",
                str((plan.get("public_precomputed_cache") or {})["path"]),
                "--may-require-precomputed-cache",
            ],
        )
        result = {
            **_status_base(plan),
            "status": "seed_active",
            "dependency": dict(dependency),
            "completed_seeds": completed,
            "active_seed": {"seed": seed, "run_id": run_id, "start": start},
            "deployment": deployment,
            "cuda_smoke": cuda_smoke,
            "gpu_gate": final_gate,
            "launch_claim": str(claim_path.resolve()),
        }
        write_json_atomic(status_path, result)
        return result
    result = {
        **_status_base(plan),
        "status": "all_seeds_terminal",
        "dependency": dict(dependency),
        "completed_seeds": completed,
    }
    write_json_atomic(status_path, result)
    return result


def run_once(plan: Mapping[str, Any], evidence_dir: Path) -> dict[str, Any]:
    status_path = evidence_dir / "status_current.json"
    dependency_path = str(plan["serial_queue"]["dependency_status_path"])
    dependency = read_remote_dependency(dependency_path)
    classification = classify_dependency(dependency)
    if classification == "failed":
        result = {
            **_status_base(plan),
            "status": "blocked_by_dependency_integrity_failure",
            "dependency": dependency,
        }
        write_json_atomic(status_path, result)
        return result
    if classification == "waiting":
        result = {
            **_status_base(plan),
            "status": "waiting_for_cactus_terminal",
            "dependency": dependency,
        }
        write_json_atomic(status_path, result)
        return result
    assert dependency is not None
    return run_ready_launch(plan, evidence_dir=evidence_dir, dependency=dependency)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--deadline-hours", type=float, default=240.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.poll_seconds < 15 or args.deadline_hours <= 0:
        raise ValueError("May queue timing contract is invalid")
    evidence_dir = Path(args.evidence_dir).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    deadline = datetime.now().astimezone() + timedelta(hours=args.deadline_hours)
    while True:
        try:
            plan = validate_execution_plan(args.plan)
            result = run_once(plan, evidence_dir)
        except Exception as exc:
            result = {
                "schema": "evomind.hpc88240.may2022_nested_queue_failure.v1",
                "created_at": now_iso(),
                "status": "transient_or_contract_failure",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "process_signals_sent": 0,
                "other_processes_modified": False,
                "official_grader_executed": False,
                "kaggle_submission_executed": False,
            }
            write_json_atomic(evidence_dir / "failure_current.json", result)
            if args.once:
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 1
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            if result["status"] in {
                "all_seeds_terminal",
                "launch_claim_already_exists",
                "blocked_by_dependency_integrity_failure",
            }:
                return 0 if result["status"] != "blocked_by_dependency_integrity_failure" else 5
            if args.once:
                return 0
        if datetime.now().astimezone() >= deadline:
            timeout = {
                "schema": "evomind.hpc88240.may2022_cached_queue.v1",
                "created_at": now_iso(),
                "status": "timeout_waiting_for_serial_queue_or_gpu",
                "deadline": deadline.isoformat(),
                "process_signals_sent": 0,
                "other_processes_modified": False,
                "official_grader_executed": False,
                "kaggle_submission_executed": False,
            }
            write_json_atomic(evidence_dir / "status_current.json", timeout)
            return 4
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

