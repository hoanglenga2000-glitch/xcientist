#!/usr/bin/env python3
"""Serially execute Taxi CPU seeds 44 and 45 after verified parent gates."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.deploy_job89941_taxi_cpu_confirmation import (  # noqa: E402
    DEFAULT_EVIDENCE_ROOT,
    DEFAULT_PLAN_44,
    DEFAULT_PLAN_45,
    collect_confirmation,
    deploy_confirmation,
    parent_gate,
    validate_confirmation_plan,
)
from scripts.deploy_job89941_taxi_cpu_candidate import write_json_atomic  # noqa: E402


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def classify_collection(value: Mapping[str, Any]) -> str:
    status = value.get("status")
    if status == "waiting_for_terminal_result":
        return "running"
    if status == "completed_and_verified":
        return (
            "passed"
            if value.get("candidate_ready_for_multiseed_confirmation") is True
            else "gate_failed"
        )
    return "failed"


def classify_parent_gate(value: Mapping[str, Any]) -> str:
    if value.get("ready") is True:
        return "ready"
    if (
        value.get("status") == "completed_and_verified"
        and value.get("parent_passed") is False
    ):
        return "gate_failed"
    return "waiting"


def watch_chain(
    *,
    plan_paths: list[Path],
    evidence_root: Path,
    poll_seconds: int,
    deadline_hours: float,
) -> dict[str, Any]:
    plans = [validate_confirmation_plan(path) for path in plan_paths]
    deadline = datetime.now().astimezone() + timedelta(hours=deadline_hours)
    chain_path = evidence_root / "chain_current.json"
    completed: list[dict[str, Any]] = []
    for plan in plans:
        seed = int(plan["runtime"]["seed"])
        evidence_dir = evidence_root.parent / f"job89941_taxi_cpu_confirmation_s{seed}"
        while True:
            gate = parent_gate(plan)
            gate_classification = classify_parent_gate(gate)
            if gate_classification != "ready":
                report = {
                    "schema": "evomind.hpc.job89941_taxi_cpu_confirmation_chain.v1",
                    "created_at": now_iso(),
                    "status": (
                        "waiting_for_parent_gate"
                        if gate_classification == "waiting"
                        else f"stopped_before_seed_{seed}_parent_gate_failed"
                    ),
                    "completed": completed,
                    "current_seed": seed,
                    "parent_gate": gate,
                    "process_signals_sent": 0,
                    "other_processes_modified": False,
                    "official_grader_executed": False,
                    "kaggle_submission_executed": False,
                }
                write_json_atomic(chain_path, report)
                if gate_classification == "gate_failed":
                    return report
                if datetime.now().astimezone() >= deadline:
                    report["status"] = "deadline_reached"
                    write_json_atomic(chain_path, report)
                    return report
                time.sleep(poll_seconds)
                continue
            claim_path = evidence_dir / "launch_claim.json"
            if not claim_path.exists():
                deploy_confirmation(plan, evidence_dir)
            collection = collect_confirmation(plan, evidence_dir)
            classification = classify_collection(collection)
            report = {
                "schema": "evomind.hpc.job89941_taxi_cpu_confirmation_chain.v1",
                "created_at": now_iso(),
                "status": f"seed_{seed}_{classification}",
                "completed": completed,
                "current_seed": seed,
                "parent_gate": gate,
                "collection": collection,
                "process_signals_sent": 0,
                "other_processes_modified": False,
                "official_grader_executed": False,
                "kaggle_submission_executed": False,
            }
            write_json_atomic(chain_path, report)
            if classification == "running":
                if datetime.now().astimezone() >= deadline:
                    report["status"] = "deadline_reached_process_left_running"
                    write_json_atomic(chain_path, report)
                    return report
                time.sleep(poll_seconds)
                continue
            if classification != "passed":
                report["status"] = f"stopped_after_seed_{seed}_{classification}"
                write_json_atomic(chain_path, report)
                return report
            completed.append(
                {
                    "seed": seed,
                    "plan_sha256": plan["_sha256"],
                    "collection_path": str((evidence_dir / "collection_current.json").resolve()),
                    "random_oof_rmse": (collection.get("result") or {}).get("random_oof_rmse"),
                    "temporal_rmse": (collection.get("result") or {}).get("temporal_rmse"),
                    "geographic_rmse": (collection.get("result") or {}).get("geographic_rmse"),
                }
            )
            break
    report = {
        "schema": "evomind.hpc.job89941_taxi_cpu_confirmation_chain.v1",
        "created_at": now_iso(),
        "status": "all_confirmation_seeds_passed",
        "completed": completed,
        "process_signals_sent": 0,
        "other_processes_modified": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }
    write_json_atomic(chain_path, report)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-44", type=Path, default=DEFAULT_PLAN_44)
    parser.add_argument("--plan-45", type=Path, default=DEFAULT_PLAN_45)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--poll-seconds", type=int, default=180)
    parser.add_argument("--deadline-hours", type=float, default=216.0)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    report = watch_chain(
        plan_paths=[args.plan_44, args.plan_45],
        evidence_root=args.evidence_root.resolve(),
        poll_seconds=args.poll_seconds,
        deadline_hours=args.deadline_hours,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
