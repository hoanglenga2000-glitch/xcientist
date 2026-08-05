#!/usr/bin/env python3
"""Launch the frozen Leaf candidate after early Jigsaw confirmation.

This scheduling-only queue reuses the existing Leaf frozen plan and run ID.
The original after-SIIM queue remains untouched and will later fail closed with
``target_run_already_exists`` if this queue has already created the run.
"""
from __future__ import annotations

import argparse
import hashlib
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

from scripts import queue_leaf_after_siim as base  # noqa: E402

CONTROL_SCHEMA = "evomind.leaf.early_after_jigsaw_control.v1"
DEFAULT_CONTROL_PLAN = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "leaf_early_after_jigsaw_control_frozen_plan.json"
)
DEFAULT_STATUS = PROJECT_ROOT / "workspace" / "local_gpu" / "leaf_early_queue.json"
ORIGINAL_SAFE_STATUS = "waiting_for_prerequisites"
JIGSAW_TERMINAL_STATUSES = {
    "confirmation_passed_human_gate_pending",
    "confirmation_failed",
    "training_failed",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def invariant_fields() -> dict[str, Any]:
    return {
        "process_signals_sent": 0,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "human_gate_preserved": True,
        "strict_single_gpu_serial": True,
    }


def validate_control_plan(path: Path) -> dict[str, Any]:
    path = path.resolve()
    control = read_json(path)
    if control.get("schema") != CONTROL_SCHEMA:
        raise RuntimeError("Unexpected Leaf early-queue control schema")
    if control.get("status") != "frozen_waiting_jigsaw_confirmation":
        raise RuntimeError("Leaf early-queue control is not frozen")
    source = control.get("source") or {}
    source_path = Path(source.get("path", "")).resolve()
    if source_path != Path(__file__).resolve() or sha256_file(source_path) != source.get(
        "sha256"
    ):
        raise RuntimeError("Leaf early-queue source drifted")
    base_binding = control.get("base_leaf_plan") or {}
    base_path = Path(base_binding.get("path", "")).resolve()
    if not base_path.is_file() or sha256_file(base_path) != base_binding.get("sha256"):
        raise RuntimeError("Leaf base plan drifted")
    jigsaw_binding = (control.get("jigsaw_dependency") or {}).get("control_plan") or {}
    jigsaw_path = Path(jigsaw_binding.get("path", "")).resolve()
    if not jigsaw_path.is_file() or sha256_file(jigsaw_path) != jigsaw_binding.get(
        "sha256"
    ):
        raise RuntimeError("Jigsaw early control plan drifted")
    if control.get("process_signals_allowed") is not False:
        raise RuntimeError("Leaf early control permits process signals")
    if control.get("official_grader_executed") is not False:
        raise RuntimeError("Leaf early control permits official grading")
    if control.get("kaggle_submission_executed") is not False:
        raise RuntimeError("Leaf early control permits Kaggle submission")
    control["_path"] = str(path)
    control["_sha256"] = sha256_file(path)
    control["_base"] = base.validate_frozen_plan(base_path)
    return control


def jigsaw_snapshot(control: dict[str, Any]) -> dict[str, Any]:
    dependency = control["jigsaw_dependency"]
    path = Path(dependency["status_path"]).resolve()
    payload = read_json(path) if path.is_file() else {}
    status = payload.get("status")
    checks = {
        "present": path.is_file(),
        "schema": payload.get("schema") == dependency["status_schema"],
        "control_hash": payload.get("control_plan_sha256")
        == dependency["control_plan"]["sha256"],
        "terminal": status in JIGSAW_TERMINAL_STATUSES,
        "signals": payload.get("process_signals_sent") == 0,
        "private": payload.get("private_labels_used") is False,
        "grader": payload.get("official_grader_executed") is False,
        "kaggle": payload.get("kaggle_submission_executed") is False,
    }
    return {
        "ready": all(checks.values()),
        "path": str(path),
        "status": status,
        "candidate_ready_for_human_gate": payload.get(
            "candidate_ready_for_human_gate"
        ),
        "checks": checks,
    }


def original_queue_snapshot(control: dict[str, Any]) -> dict[str, Any]:
    path = Path(control["original_after_siim_queue_status"]).resolve()
    payload = read_json(path) if path.is_file() else {}
    status = payload.get("status")
    return {
        "safe_for_early_queue": bool(
            path.is_file()
            and status == ORIGINAL_SAFE_STATUS
            and payload.get("process_signals_sent") == 0
            and payload.get("official_grader_executed") is False
            and payload.get("kaggle_submission_executed") is False
        ),
        "path": str(path),
        "status": status,
        "pid": payload.get("pid"),
        "process_signals_sent": payload.get("process_signals_sent"),
    }


def status_payload(control: dict[str, Any], status: str, **fields: Any) -> dict[str, Any]:
    return {
        "schema": "evomind.leaf.early_training_queue.v1",
        "created_at": now_iso(),
        "status": status,
        "control_plan_path": control["_path"],
        "control_plan_sha256": control["_sha256"],
        "base_plan_path": control["_base"]["_plan_path"],
        "base_plan_sha256": control["_base"]["_plan_sha256"],
        **fields,
        **invariant_fields(),
    }


def control_unchanged(control: dict[str, Any]) -> None:
    if sha256_file(Path(control["_path"])) != control["_sha256"]:
        raise RuntimeError("Leaf early control changed while queue ran")
    plan = control["_base"]
    if sha256_file(Path(plan["_plan_path"])) != plan["_plan_sha256"]:
        raise RuntimeError("Leaf base plan changed while queue ran")


def run_queue(
    control: dict[str, Any], status_path: Path, poll_seconds: int, deadline: datetime
) -> int:
    plan = control["_base"]
    idle_policy = plan["_idle_policy"]
    requirements = idle_policy["requirements"]
    dependency: dict[str, Any] = {}
    original: dict[str, Any] = {}
    while datetime.now().astimezone() < deadline:
        control_unchanged(control)
        dependency = jigsaw_snapshot(control)
        original = original_queue_snapshot(control)
        base.write_json_atomic(
            status_path,
            status_payload(
                control,
                "waiting_for_jigsaw_confirmation",
                pid=os.getpid(),
                deadline=deadline.isoformat(),
                dependency=dependency,
                original_after_siim_queue=original,
                consecutive_idle_checks=0,
                required_idle_checks=requirements["consecutive_checks"],
                gpu_idle_gate=base.calibrated_idle.policy_record(idle_policy),
            ),
        )
        if not original["safe_for_early_queue"]:
            base.write_json_atomic(
                status_path,
                status_payload(
                    control,
                    "superseded_by_original_after_siim_queue",
                    original_after_siim_queue=original,
                ),
            )
            return 6
        if dependency["ready"]:
            break
        time.sleep(poll_seconds)
    else:
        base.write_json_atomic(
            status_path,
            status_payload(
                control,
                "timeout_waiting_for_jigsaw_confirmation",
                dependency=dependency,
                original_after_siim_queue=original,
            ),
        )
        return 4

    run_dir = (
        Path(plan["training"]["output_root"])
        / "runs"
        / plan["training"]["run_id"]
    )
    if run_dir.exists():
        base.write_json_atomic(
            status_path,
            status_payload(
                control,
                "target_run_already_exists",
                run_dir=str(run_dir),
                dependency=dependency,
            ),
        )
        return 5

    consecutive = 0
    while datetime.now().astimezone() < deadline:
        control_unchanged(control)
        original = original_queue_snapshot(control)
        if not original["safe_for_early_queue"]:
            base.write_json_atomic(
                status_path,
                status_payload(
                    control,
                    "superseded_by_original_after_siim_queue",
                    original_after_siim_queue=original,
                ),
            )
            return 6
        if run_dir.exists():
            base.write_json_atomic(
                status_path,
                status_payload(control, "target_run_already_exists", run_dir=str(run_dir)),
            )
            return 5
        gpu = base.calibrated_idle.query_gpu()
        evaluation = base.calibrated_idle.evaluate_idle(idle_policy, gpu)
        consecutive = consecutive + 1 if evaluation["idle"] else 0
        base.write_json_atomic(
            status_path,
            status_payload(
                control,
                "waiting_for_stable_gpu_idle",
                dependency=dependency,
                original_after_siim_queue=original,
                gpu=gpu,
                idle_gate_evaluation=evaluation,
                consecutive_idle_checks=consecutive,
                required_idle_checks=requirements["consecutive_checks"],
                gpu_idle_gate=base.calibrated_idle.policy_record(idle_policy),
            ),
        )
        if consecutive >= int(requirements["consecutive_checks"]):
            final_plan = base.validate_frozen_plan(Path(plan["_plan_path"]))
            final_original = original_queue_snapshot(control)
            final_gpu = base.calibrated_idle.query_gpu()
            final_evaluation = base.calibrated_idle.evaluate_idle(
                final_plan["_idle_policy"], final_gpu
            )
            if (
                final_original["safe_for_early_queue"]
                and final_evaluation["idle"]
                and not run_dir.exists()
            ):
                plan = final_plan
                break
            consecutive = 0
        time.sleep(int(requirements["minimum_check_interval_seconds"]))
    else:
        raise TimeoutError("Timed out waiting for idle GPU before Leaf")

    command = base.build_training_command(plan)
    output_root = Path(plan["training"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = plan["training"]["run_id"]
    stdout_path = output_root / f"{run_id}.stdout.log"
    stderr_path = output_root / f"{run_id}.stderr.log"
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
    base.write_json_atomic(
        status_path,
        status_payload(
            control,
            "training_launched" if launched else "training_launch_failed",
            training_pid=child.pid,
            training_return_code=return_code,
            command=command,
            stdout=str(stdout_path),
            stderr=str(stderr_path),
            dependency=dependency,
            gpu_before_launch=final_gpu,
            idle_gate_evaluation=final_evaluation,
            consecutive_idle_checks=consecutive,
        ),
    )
    return 0 if launched else 7


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-plan", type=Path, default=DEFAULT_CONTROL_PLAN)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--deadline-hours", type=float, default=240.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.poll_seconds < 10 or args.deadline_hours <= 0:
        raise ValueError("Leaf early queue timing contract is invalid")
    control = validate_control_plan(args.control_plan)
    deadline = datetime.now().astimezone() + timedelta(hours=args.deadline_hours)
    return run_queue(control, args.status.resolve(), args.poll_seconds, deadline)


if __name__ == "__main__":
    raise SystemExit(main())
