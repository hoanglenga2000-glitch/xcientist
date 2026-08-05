#!/usr/bin/env python3
"""Use a verified local-GPU idle window for Spooky before SIIM/Leaf are ready.

This is an opportunistic, no-preemption queue.  It yields permanently as soon as
SIIM becomes launchable, any SIIM/Leaf run directory appears, or the authoritative
Spooky-after-Leaf queue can take over.  It never signals another process and it
never invokes the official grader or Kaggle.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import local_rtx4060_idle_gate as idle_gate  # noqa: E402
from scripts import queue_siim_final_candidate as siim_queue  # noqa: E402
from scripts import queue_spooky_after_leaf as spooky_queue  # noqa: E402

DEFAULT_SPOOKY_PLAN = spooky_queue.DEFAULT_PLAN
DEFAULT_SIIM_PLAN = siim_queue.DEFAULT_PLAN
DEFAULT_DATA_REPORT = spooky_queue.DEFAULT_DATA_REPORT
DEFAULT_GATE_POLICY = idle_gate.DEFAULT_POLICY
DEFAULT_STATUS = (
    PROJECT_ROOT
    / "workspace"
    / "local_gpu"
    / "spooky_idle_slot_before_leaf_queue.json"
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def classify_opportunity(snapshot: dict[str, Any]) -> str:
    """Return the single authoritative state for an opportunity snapshot."""

    if snapshot["target_complete"]:
        return "target_run_already_complete"
    if snapshot["target_run_exists"]:
        return "target_run_already_exists"
    if snapshot["siim_priority_active"]:
        return "yielded_to_siim_priority"
    if snapshot["leaf_authoritative_active"]:
        return "yielded_to_leaf_authoritative_queue"
    if not snapshot["spooky_data"]["ready"]:
        return "waiting_for_spooky_data"
    return "eligible_for_idle_gate"


def opportunity_snapshot(
    spooky_plan: dict[str, Any],
    siim_plan: dict[str, Any],
    *,
    data_report_path: Path,
    leaf_report_path: Path | None = None,
) -> dict[str, Any]:
    """Build a fail-closed snapshot without changing any process or run."""

    spooky_data = spooky_queue.data_snapshot(spooky_plan, data_report_path)
    leaf = spooky_queue.leaf_snapshot(spooky_plan, leaf_report_path)
    leaf_run_dir = Path(leaf["report"]).resolve().parent

    siim_staging = siim_queue.staging_snapshot(siim_plan)
    siim_ablation = siim_queue.ablation_snapshot(siim_plan)
    siim_ablation_run_dir = Path(siim_ablation["report"]).resolve().parent
    siim_training = siim_plan["training"]
    siim_final_run_dir = (
        Path(siim_training["output_root"]).resolve() / siim_training["run_id"]
    )

    spooky_execution = spooky_plan["execution"]
    target_run_dir = (
        Path(spooky_execution["output_root"]).resolve()
        / spooky_execution["run_id"]
    )
    target_summary = target_run_dir / "summary.json"
    target_complete = False
    target_summary_status: str | None = None
    if target_summary.is_file():
        try:
            payload = spooky_queue.read_json(target_summary)
            target_summary_status = payload.get("status")
            target_complete = payload.get("stage") == "terminal" or (
                target_summary_status
                in {
                    "single_seed_gate_passed_confirmation_pending",
                    "single_seed_gate_failed",
                }
            )
        except (OSError, json.JSONDecodeError, TypeError):
            target_complete = False

    snapshot: dict[str, Any] = {
        "spooky_data": spooky_data,
        "leaf": leaf,
        "leaf_run_dir": str(leaf_run_dir),
        "leaf_run_exists": leaf_run_dir.exists(),
        "siim_staging": siim_staging,
        "siim_ablation": siim_ablation,
        "siim_ablation_run_dir": str(siim_ablation_run_dir),
        "siim_ablation_run_exists": siim_ablation_run_dir.exists(),
        "siim_final_run_dir": str(siim_final_run_dir),
        "siim_final_run_exists": siim_final_run_dir.exists(),
        "target_run_dir": str(target_run_dir),
        "target_run_exists": target_run_dir.exists(),
        "target_summary": str(target_summary),
        "target_summary_status": target_summary_status,
        "target_complete": target_complete,
    }
    snapshot["siim_priority_active"] = bool(
        siim_staging["ready"]
        or snapshot["siim_ablation_run_exists"]
        or snapshot["siim_final_run_exists"]
    )
    snapshot["leaf_authoritative_active"] = bool(
        leaf["ready"] or snapshot["leaf_run_exists"]
    )
    snapshot["decision"] = classify_opportunity(snapshot)
    snapshot["eligible"] = snapshot["decision"] == "eligible_for_idle_gate"
    return snapshot


def write_status(
    path: Path,
    *,
    status: str,
    deadline: datetime,
    spooky_plan: dict[str, Any],
    siim_plan: dict[str, Any],
    opportunity: dict[str, Any],
    gpu: dict[str, Any] | None,
    idle_checks: int,
    required_idle_checks: int,
    **extra: Any,
) -> None:
    payload: dict[str, Any] = {
        "schema": "evomind.spooky.idle_slot_before_leaf_queue.v1",
        "created_at": now_iso(),
        "status": status,
        "pid": os.getpid(),
        "deadline": deadline.isoformat(),
        "spooky_plan_path": spooky_plan["_path"],
        "spooky_plan_sha256": spooky_plan["_sha256"],
        "siim_plan_path": siim_plan["_plan_path"],
        "siim_plan_sha256": siim_plan["_plan_sha256"],
        "requested_model": spooky_plan["driver"]["requested_model"],
        "served_model": spooky_plan["driver"]["served_model"],
        "opportunity": opportunity,
        "gpu": gpu,
        "consecutive_idle_checks": idle_checks,
        "required_idle_checks": required_idle_checks,
        "gpu_idle_gate": spooky_plan.get("_idle_gate_policy"),
        "idle_gate_evaluation": (gpu or {}).get("_idle_gate_evaluation"),
        "single_gpu_strict_serial": True,
        "preemption_allowed": False,
        "process_signals_sent": 0,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }
    payload.update(extra)
    spooky_queue.write_json_atomic(path.resolve(), payload)


def create_launch_claim(path: Path, payload: dict[str, Any]) -> bool:
    """Atomically claim one opportunistic launch without inspecting other PIDs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0),
        )
    except FileExistsError:
        return False
    try:
        encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
            "utf-8"
        )
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spooky-plan", type=Path, default=DEFAULT_SPOOKY_PLAN)
    parser.add_argument("--siim-plan", type=Path, default=DEFAULT_SIIM_PLAN)
    parser.add_argument("--data-report", type=Path, default=DEFAULT_DATA_REPORT)
    parser.add_argument("--gate-policy", type=Path, default=DEFAULT_GATE_POLICY)
    parser.add_argument("--leaf-report", type=Path)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--priority-grace-seconds", type=int, default=20)
    parser.add_argument("--deadline-hours", type=float, default=36.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        args.poll_seconds < 10
        or args.priority_grace_seconds < 10
        or args.deadline_hours <= 0
    ):
        raise ValueError("Spooky opportunistic queue timing contract is invalid")

    spooky_plan = spooky_queue.validate_frozen_plan(args.spooky_plan)
    siim_plan = siim_queue.validate_frozen_plan(args.siim_plan)
    policy = idle_gate.validate_policy(args.gate_policy)
    if (
        policy["bound_execution_evidence"]["seed42_plan"]["sha256"]
        != spooky_plan["_sha256"]
    ):
        raise RuntimeError("Calibrated idle gate is not bound to this Spooky plan")
    if args.poll_seconds < int(policy["requirements"]["minimum_check_interval_seconds"]):
        raise ValueError("Spooky poll interval is below the calibrated gate minimum")
    spooky_plan["_idle_gate_policy"] = idle_gate.policy_record(policy)
    deadline = datetime.now().astimezone() + timedelta(hours=args.deadline_hours)
    execution = spooky_plan["execution"]
    required_idle = int(policy["requirements"]["consecutive_checks"])
    idle_checks = 0

    while datetime.now().astimezone() < deadline:
        opportunity = opportunity_snapshot(
            spooky_plan,
            siim_plan,
            data_report_path=args.data_report,
            leaf_report_path=args.leaf_report,
        )
        decision = opportunity["decision"]
        if decision in {
            "target_run_already_complete",
            "target_run_already_exists",
            "yielded_to_siim_priority",
            "yielded_to_leaf_authoritative_queue",
        }:
            write_status(
                args.status,
                status=decision,
                deadline=deadline,
                spooky_plan=spooky_plan,
                siim_plan=siim_plan,
                opportunity=opportunity,
                gpu=None,
                idle_checks=idle_checks,
                required_idle_checks=required_idle,
            )
            return 0 if decision != "target_run_already_exists" else 5

        gpu: dict[str, Any] | None = None
        status = decision
        if opportunity["eligible"]:
            if idle_gate.sha256_file(args.gate_policy) != policy["_sha256"]:
                raise RuntimeError("Calibrated idle-gate policy changed while queued")
            gpu = idle_gate.query_gpu()
            evaluation = idle_gate.evaluate_idle(policy, gpu)
            gpu["_idle_gate_evaluation"] = evaluation
            is_idle = evaluation["idle"]
            idle_checks = idle_checks + 1 if is_idle else 0
            status = "waiting_for_stable_gpu_idle"
        else:
            idle_checks = 0

        write_status(
            args.status,
            status=status,
            deadline=deadline,
            spooky_plan=spooky_plan,
            siim_plan=siim_plan,
            opportunity=opportunity,
            gpu=gpu,
            idle_checks=idle_checks,
            required_idle_checks=required_idle,
        )

        if opportunity["eligible"] and idle_checks >= required_idle:
            # Give the 10-second SIIM queue a final priority window, then re-check
            # both prerequisites and the GPU immediately before the launch claim.
            time.sleep(args.priority_grace_seconds)
            launch_spooky = spooky_queue.validate_frozen_plan(args.spooky_plan)
            launch_siim = siim_queue.validate_frozen_plan(args.siim_plan)
            launch_policy = idle_gate.validate_policy(args.gate_policy)
            if launch_spooky["_sha256"] != spooky_plan["_sha256"]:
                raise RuntimeError("Spooky frozen plan changed while queued")
            if launch_siim["_plan_sha256"] != siim_plan["_plan_sha256"]:
                raise RuntimeError("SIIM frozen plan changed while queued")
            if launch_policy["_sha256"] != policy["_sha256"]:
                raise RuntimeError("Calibrated idle-gate policy changed while queued")
            launch_spooky["_idle_gate_policy"] = idle_gate.policy_record(launch_policy)
            final_opportunity = opportunity_snapshot(
                launch_spooky,
                launch_siim,
                data_report_path=args.data_report,
                leaf_report_path=args.leaf_report,
            )
            if not final_opportunity["eligible"]:
                write_status(
                    args.status,
                    status=final_opportunity["decision"],
                    deadline=deadline,
                    spooky_plan=launch_spooky,
                    siim_plan=launch_siim,
                    opportunity=final_opportunity,
                    gpu=None,
                    idle_checks=0,
                    required_idle_checks=required_idle,
                )
                return 0 if final_opportunity["decision"] != "target_run_already_exists" else 5

            final_gpu = idle_gate.query_gpu()
            final_evaluation = idle_gate.evaluate_idle(launch_policy, final_gpu)
            final_gpu["_idle_gate_evaluation"] = final_evaluation
            final_idle = final_evaluation["idle"]
            if not final_idle:
                idle_checks = 0
                write_status(
                    args.status,
                    status="gpu_changed_during_priority_grace",
                    deadline=deadline,
                    spooky_plan=launch_spooky,
                    siim_plan=launch_siim,
                    opportunity=final_opportunity,
                    gpu=final_gpu,
                    idle_checks=0,
                    required_idle_checks=required_idle,
                )
                time.sleep(args.poll_seconds)
                continue

            run_dir = Path(final_opportunity["target_run_dir"])
            output_root = Path(launch_spooky["execution"]["output_root"]).resolve()
            output_root.mkdir(parents=True, exist_ok=True)
            claim_path = output_root / (
                f"{launch_spooky['execution']['run_id']}.opportunistic_launch_claim.json"
            )
            claim = {
                "schema": "evomind.spooky.opportunistic_launch_claim.v1",
                "created_at": now_iso(),
                "queue_pid": os.getpid(),
                "run_id": launch_spooky["execution"]["run_id"],
                "spooky_plan_sha256": launch_spooky["_sha256"],
                "siim_plan_sha256": launch_siim["_plan_sha256"],
                "idle_gate_policy_sha256": launch_policy["_sha256"],
                "process_signals_sent": 0,
                "official_grader_executed": False,
                "kaggle_submission_executed": False,
            }
            if not create_launch_claim(claim_path, claim):
                write_status(
                    args.status,
                    status="launch_claim_already_exists",
                    deadline=deadline,
                    spooky_plan=launch_spooky,
                    siim_plan=launch_siim,
                    opportunity=final_opportunity,
                    gpu=final_gpu,
                    idle_checks=idle_checks,
                    required_idle_checks=required_idle,
                    launch_claim=str(claim_path),
                )
                return 5

            command = spooky_queue.build_training_command(launch_spooky)
            stdout_path = run_dir.with_suffix(".stdout.log")
            stderr_path = run_dir.with_suffix(".stderr.log")
            environment = os.environ.copy()
            python_path = os.pathsep.join([str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)])
            if environment.get("PYTHONPATH"):
                python_path += os.pathsep + environment["PYTHONPATH"]
            environment["PYTHONPATH"] = python_path
            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            with stdout_path.open("ab", buffering=0) as stdout, stderr_path.open(
                "ab", buffering=0
            ) as stderr:
                child = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    creationflags=creation_flags,
                )
            time.sleep(10)
            return_code = child.poll()
            launched = return_code is None
            write_status(
                args.status,
                status="training_launched" if launched else "training_launch_failed",
                deadline=deadline,
                spooky_plan=launch_spooky,
                siim_plan=launch_siim,
                opportunity=final_opportunity,
                gpu=final_gpu,
                idle_checks=idle_checks,
                required_idle_checks=required_idle,
                queue_pid=os.getpid(),
                training_pid=child.pid,
                training_return_code=return_code,
                command=command,
                stdout=str(stdout_path),
                stderr=str(stderr_path),
                launch_claim=str(claim_path),
            )
            return 0 if launched else 6

        time.sleep(args.poll_seconds)

    final_opportunity = opportunity_snapshot(
        spooky_plan,
        siim_plan,
        data_report_path=args.data_report,
        leaf_report_path=args.leaf_report,
    )
    write_status(
        args.status,
        status="timeout_waiting_for_idle_window",
        deadline=deadline,
        spooky_plan=spooky_plan,
        siim_plan=siim_plan,
        opportunity=final_opportunity,
        gpu=None,
        idle_checks=idle_checks,
        required_idle_checks=required_idle,
    )
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
