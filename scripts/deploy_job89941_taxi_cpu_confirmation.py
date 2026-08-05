#!/usr/bin/env python3
"""Launch Taxi CPU confirmation seeds only after a verified parent passes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.collect_job89941_taxi_cpu_candidate import collect_validated_plan  # noqa: E402
from scripts.deploy_job89941_taxi_cpu_candidate import (  # noqa: E402
    REMOTE_UNIFIED_SITE_PACKAGES,
    TaxiCpuDeploymentError,
    confined_remote,
    connect_job89941,
    launch,
    remote_preflight,
    sha256_file,
    status,
    upload_sources,
    write_json_atomic,
)

PLAN_SCHEMA = "evomind.hpc.job89941_taxi_cpu_confirmation_plan.v1"
DEFAULT_PLAN_44 = (
    PROJECT_ROOT / "workspace" / "mlebench_plans" / "taxi_cpu_lightgbm_confirmation_s44_job89941_20260728.json"
)
DEFAULT_PLAN_45 = (
    PROJECT_ROOT / "workspace" / "mlebench_plans" / "taxi_cpu_lightgbm_confirmation_s45_job89941_20260728.json"
)
DEFAULT_EVIDENCE_ROOT = PROJECT_ROOT / "workspace" / "hpc" / "job89941_taxi_cpu_confirmation"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TaxiCpuDeploymentError(f"JSON object required: {path}")
    return payload


def validate_confirmation_plan(plan_path: Path) -> dict[str, Any]:
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    if plan.get("schema") != PLAN_SCHEMA or plan.get("status") != "approved_waiting_for_parent_gate":
        raise TaxiCpuDeploymentError("Taxi confirmation plan schema/status changed")
    runtime = plan.get("runtime") or {}
    if (
        runtime.get("seed") not in {44, 45}
        or runtime.get("threads") != 60
        or runtime.get("iterations") != 1400
        or runtime.get("cuda_visible_devices") != ""
        or runtime.get("pythonpath") != REMOTE_UNIFIED_SITE_PACKAGES
    ):
        raise TaxiCpuDeploymentError("Taxi confirmation runtime changed")
    dependency = plan.get("dependency") or {}
    if (
        dependency.get("launch_only_after_completed_and_verified") is not True
        or dependency.get("parent_candidate_ready_required") is not True
        or dependency.get("parent_seed") != runtime.get("seed") - 1
        or len(str(dependency.get("parent_plan_sha256") or "")) != 64
    ):
        raise TaxiCpuDeploymentError("Taxi confirmation dependency changed")
    boundary = plan.get("boundary") or {}
    for key, expected in {
        "visibility_mode": "PUBLIC_ONLY",
        "candidate_only": True,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "gpu_used": False,
        "process_signals_allowed": False,
        "other_processes_may_be_modified": False,
    }.items():
        if boundary.get(key) != expected:
            raise TaxiCpuDeploymentError(f"Taxi confirmation boundary changed: {key}")
    for key in (
        "remote_run_dir",
        "remote_source",
        "remote_plan",
        "remote_state",
        "remote_log",
        "remote_output_dir",
    ):
        confined_remote(str(plan.get(key) or ""))
    for name in ("candidate_source", "deployment_source"):
        record = plan.get(name) or {}
        path = Path(str(record.get("path") or "")).resolve()
        if not path.is_file() or path.stat().st_size != record.get("bytes") or sha256_file(path) != record.get("sha256"):
            raise TaxiCpuDeploymentError(f"Taxi confirmation source changed: {name}")
    for name in ("base_cache", "route_stat_cache"):
        record = plan.get(name) or {}
        confined_remote(str(record.get("remote_path") or ""))
        local = Path(str(record.get("local_evidence_path") or "")).resolve()
        if not local.is_file() or local.stat().st_size != record.get("bytes") or sha256_file(local) != record.get("sha256"):
            raise TaxiCpuDeploymentError(f"Taxi confirmation cache changed: {name}")
    plan["_path"] = str(plan_path)
    plan["_sha256"] = sha256_file(plan_path)
    return plan


def parent_gate(plan: Mapping[str, Any]) -> dict[str, Any]:
    dependency = plan["dependency"]
    path = Path(str(dependency["collection_path"])).resolve()
    if not path.is_file():
        return {"ready": False, "status": "parent_collection_missing"}
    value = read_json(path)
    result = value.get("result") if isinstance(value.get("result"), dict) else {}
    ready = bool(
        value.get("status") == "completed_and_verified"
        and value.get("candidate_ready_for_multiseed_confirmation") is True
        and value.get("plan_sha256") == dependency.get("parent_plan_sha256")
        and result.get("seed") == dependency.get("parent_seed")
        and result.get("passed") is True
        and result.get("candidate_only") is True
        and result.get("official_grader_executed") is False
        and result.get("kaggle_submission_executed") is False
    )
    return {
        "ready": ready,
        "status": value.get("status"),
        "path": str(path),
        "parent_plan_sha256": value.get("plan_sha256"),
        "parent_seed": result.get("seed"),
        "parent_passed": result.get("passed"),
    }


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


def deploy_confirmation(plan: Mapping[str, Any], evidence_dir: Path) -> dict[str, Any]:
    gate = parent_gate(plan)
    if gate.get("ready") is not True:
        report = {
            "schema": "evomind.hpc.job89941_taxi_cpu_confirmation_queue.v1",
            "created_at": now_iso(),
            "status": "waiting_for_parent_gate",
            "plan_path": plan["_path"],
            "plan_sha256": plan["_sha256"],
            "parent_gate": gate,
            "process_signals_sent": 0,
            "other_processes_modified": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        }
        write_json_atomic(evidence_dir / "status_current.json", report)
        return report
    claim_path = evidence_dir / "launch_claim.json"
    claim = {
        "schema": "evomind.hpc.job89941_taxi_cpu_confirmation_claim.v1",
        "created_at": now_iso(),
        "run_id": plan["run_id"],
        "seed": plan["runtime"]["seed"],
        "plan_sha256": plan["_sha256"],
        "parent_gate": gate,
        "process_signals_sent": 0,
    }
    if not acquire_claim(claim_path, claim):
        return {
            "status": "launch_claim_already_exists",
            "claim_path": str(claim_path.resolve()),
            "plan_sha256": plan["_sha256"],
        }
    client = connect_job89941()
    try:
        preflight = remote_preflight(client, plan)
        uploads = upload_sources(client, plan)
        launched = launch(client, plan)
        observed = status(client, plan)
    finally:
        client.close()
    report = {
        "schema": "evomind.hpc.job89941_taxi_cpu_confirmation_queue.v1",
        "created_at": now_iso(),
        "status": "launched",
        "plan_path": plan["_path"],
        "plan_sha256": plan["_sha256"],
        "parent_gate": gate,
        "preflight": preflight,
        "uploads": uploads,
        "launch": launched,
        "remote_status": observed,
        "claim_path": str(claim_path.resolve()),
        "process_signals_sent": 0,
        "other_processes_modified": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }
    write_json_atomic(evidence_dir / "status_current.json", report)
    return report


def collect_confirmation(plan: Mapping[str, Any], evidence_dir: Path) -> dict[str, Any]:
    return collect_validated_plan(plan, evidence_dir)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "deploy", "collect"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    plan = validate_confirmation_plan(args.plan)
    if args.action == "validate":
        result = {"status": "validated", "plan_sha256": plan["_sha256"]}
    elif args.action == "deploy":
        result = deploy_confirmation(plan, args.evidence_dir.resolve())
    else:
        result = collect_confirmation(plan, args.evidence_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
