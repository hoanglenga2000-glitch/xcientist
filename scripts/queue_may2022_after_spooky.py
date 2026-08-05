#!/usr/bin/env python3
"""Launch the frozen May-2022 run after Spooky and stable GPU-idle gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import local_rtx4060_idle_gate as calibrated_idle  # noqa: E402

PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_PLAN = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "may2022_compact_embedding_s42_frozen_plan_v2_20260728.json"
)
DEFAULT_STATUS = PROJECT_ROOT / "workspace" / "local_gpu" / "may2022_training_queue.json"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_frozen_plan(path: Path) -> dict[str, Any]:
    plan_path = Path(path).resolve()
    plan = read_json(plan_path)
    if plan.get("schema") != "evomind.mlebench.may2022_compact_embedding_plan.v1":
        raise RuntimeError("Unexpected May-2022 frozen-plan schema")
    if plan.get("status") != "frozen_before_training":
        raise RuntimeError("May-2022 plan is not frozen before training")
    if plan.get("competition_id") != "tabular-playground-series-may-2022":
        raise RuntimeError("Unexpected May-2022 competition")
    planner = plan.get("planner") or {}
    if (
        planner.get("requested_model") != "gpt-5.6-sol"
        or planner.get("served_model") != "gpt-5.6-sol"
        or planner.get("validation_errors") != []
    ):
        raise RuntimeError("May-2022 plan lacks verified gpt-5.6-sol provenance")
    for path_key, hash_key in (
        ("gateway_evidence", "gateway_evidence_sha256"),
        ("adaptive_controller_evidence", "adaptive_controller_evidence_sha256"),
    ):
        artifact = Path(str(planner.get(path_key, "")))
        if not artifact.is_file() or sha256_file(artifact) != planner.get(hash_key):
            raise RuntimeError(f"May-2022 planner artifact changed: {path_key}")
    runner = Path(str(plan.get("runner_path", "")))
    if not runner.is_file() or sha256_file(runner) != plan.get("runner_sha256"):
        raise RuntimeError("May-2022 compact runner changed after plan freeze")
    smoke = plan.get("cuda_smoke") or {}
    smoke_path = Path(str(smoke.get("path", "")))
    if not smoke_path.is_file() or sha256_file(smoke_path) != smoke.get("sha256"):
        raise RuntimeError("May-2022 CUDA smoke evidence changed after plan freeze")
    execution = plan.get("execution") or {}
    for key in (
        "python",
        "script",
        "public_dir",
        "data_report",
        "output_root",
        "run_id",
        "serial_dependency",
    ):
        if not execution.get(key):
            raise RuntimeError(f"May-2022 plan is missing execution.{key}")
    if (
        execution.get("single_gpu_strict_serial") is not True
        or execution.get("process_signals_allowed") is not False
        or execution.get("automatic_kaggle_submission") is not False
        or execution.get("official_private_grader_before_gate") is not False
    ):
        raise RuntimeError("May-2022 serial execution contract changed")
    if (
        plan.get("private_labels_allowed") is not False
        or plan.get("official_grader_enabled") is not False
        or plan.get("kaggle_submission_enabled") is not False
    ):
        raise RuntimeError("May-2022 plan violates the grading/submission boundary")
    idle_binding = execution.get("idle_gate_policy") or {}
    idle_policy_path = Path(str(idle_binding.get("path", ""))).resolve()
    if (
        not idle_policy_path.is_file()
        or sha256_file(idle_policy_path) != idle_binding.get("sha256")
    ):
        raise RuntimeError("May-2022 calibrated idle-gate binding changed")
    idle_policy = calibrated_idle.validate_policy(idle_policy_path)
    if idle_policy["_sha256"] != idle_binding.get("sha256"):
        raise RuntimeError("May-2022 calibrated idle-gate hash mismatch")
    requirements = idle_policy["requirements"]
    if (
        execution.get("gpu_idle_consecutive_checks")
        != requirements["consecutive_checks"]
        or execution.get("gpu_idle_minimum_check_interval_seconds")
        != requirements["minimum_check_interval_seconds"]
    ):
        raise RuntimeError("May-2022 idle timing differs from the calibrated policy")
    dependency = execution["serial_dependency"]
    dependency_plan = Path(dependency["plan_path"])
    if not dependency_plan.is_file() or sha256_file(dependency_plan) != dependency["plan_sha256"]:
        raise RuntimeError("May-2022 Spooky dependency plan changed")
    plan["_path"] = str(plan_path)
    plan["_sha256"] = sha256_file(plan_path)
    plan["_idle_policy"] = idle_policy
    return plan


def data_snapshot(plan: dict[str, Any]) -> dict[str, Any]:
    execution = plan["execution"]
    report_path = Path(execution["data_report"])
    if not report_path.is_file():
        return {"ready": False, "status": "missing", "report": str(report_path)}
    try:
        report = read_json(report_path)
    except (OSError, json.JSONDecodeError):
        return {"ready": False, "status": "invalid_json", "report": str(report_path)}
    public_dir = Path(execution["public_dir"])
    hashes = plan["public_input_sha256"]
    expected = {
        "train.csv": hashes["train"],
        "test.csv": hashes["test"],
        "sample_submission.csv": hashes["sample"],
    }
    file_hash_checks = {
        name: (public_dir / name).is_file()
        and sha256_file(public_dir / name) == expected_hash
        for name, expected_hash in expected.items()
    }
    checks = {
        "schema": report.get("schema") == "evomind.may2022.public_staging.v1",
        "competition": report.get("competition_id")
        == "tabular-playground-series-may-2022",
        "status": report.get("status") == "hash_verified_complete",
        "files": report.get("completed_files") == 3 == report.get("total_files"),
        "bytes": report.get("completed_bytes") == report.get("total_bytes") == 319_633_502,
        "hashes": all(file_hash_checks.values()),
        "no_errors": report.get("errors") == [],
        "no_private_paths": report.get("private_paths_requested") is False,
        "no_remote_writes": report.get("remote_writes_performed") is False,
        "no_process_signals": report.get("process_signals_sent") == 0,
        "no_official_grader": report.get("official_grader_executed") is False,
        "no_kaggle_submission": report.get("kaggle_submission_executed") is False,
    }
    return {
        "ready": all(checks.values()),
        "status": report.get("status", "missing"),
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "checks": checks,
        "file_hash_checks": file_hash_checks,
    }


def spooky_snapshot(plan: dict[str, Any]) -> dict[str, Any]:
    dependency = plan["execution"]["serial_dependency"]
    report_path = Path(dependency["report"])
    if not report_path.is_file():
        return {"ready": False, "status": "missing", "report": str(report_path)}
    try:
        report = read_json(report_path)
    except (OSError, json.JSONDecodeError):
        return {"ready": False, "status": "invalid_json", "report": str(report_path)}
    checks = {
        "schema": report.get("schema") == "evomind.spooky.transformer_run.v1",
        "run_id": report.get("run_id") == dependency["run_id"],
        "terminal": report.get("status") in dependency["terminal_statuses"],
        "all_folds": report.get("requested_folds") == [0, 1, 2, 3, 4],
        "no_private_labels": report.get("private_labels_used") is False,
        "no_official_grader": report.get("official_grader_executed") is False,
        "no_kaggle_submission": report.get("kaggle_submission_executed") is False,
    }
    return {
        "ready": all(checks.values()),
        "status": report.get("status", "missing"),
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "checks": checks,
    }


def prerequisite_snapshot(plan: dict[str, Any]) -> dict[str, Any]:
    data = data_snapshot(plan)
    spooky = spooky_snapshot(plan)
    return {"data": data, "spooky": spooky, "ready": data["ready"] and spooky["ready"]}


def parse_compute_apps(output: str) -> list[dict[str, Any]]:
    applications: list[dict[str, Any]] = []
    for row in csv.reader(output.splitlines()):
        if len(row) < 2:
            continue
        try:
            pid = int(row[0].strip())
        except ValueError:
            continue
        applications.append({"pid": pid, "process_name": row[1].strip()})
    return applications


def query_gpu() -> dict[str, Any]:
    return calibrated_idle.query_gpu()


def acquire_launch_claim(path: Path, payload: dict[str, Any]) -> bool:
    """Create one local launch claim without replacing an existing claimant."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return True


def build_training_command(plan: dict[str, Any]) -> list[str]:
    execution = plan["execution"]
    return [
        execution["python"],
        execution["script"],
        "--public-dir",
        execution["public_dir"],
        "--plan",
        plan["_path"],
        "--output-root",
        execution["output_root"],
        "--run-id",
        execution["run_id"],
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--deadline-hours", type=float, default=240.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.poll_seconds < 10 or args.deadline_hours <= 0:
        raise ValueError("May-2022 queue timing contract is invalid")
    plan = validate_frozen_plan(args.plan)
    execution = plan["execution"]
    idle_policy = plan["_idle_policy"]
    required_idle = int(idle_policy["requirements"]["consecutive_checks"])
    minimum_interval = int(
        idle_policy["requirements"]["minimum_check_interval_seconds"]
    )
    if args.poll_seconds < minimum_interval:
        raise ValueError("May-2022 polling is faster than the calibrated idle interval")
    deadline = datetime.now().astimezone() + timedelta(hours=args.deadline_hours)
    idle_checks = 0
    while datetime.now().astimezone() < deadline:
        prerequisites = prerequisite_snapshot(plan)
        gpu: dict[str, Any] | None = None
        status = "waiting_for_prerequisites"
        if prerequisites["ready"]:
            gpu = query_gpu()
            evaluation = calibrated_idle.evaluate_idle(idle_policy, gpu)
            gpu["_idle_gate_evaluation"] = evaluation
            is_idle = bool(evaluation["idle"])
            idle_checks = idle_checks + 1 if is_idle else 0
            status = "waiting_for_stable_gpu_idle"
        write_json_atomic(
            args.status,
            {
                "schema": "evomind.may2022.training_queue.v1",
                "created_at": now_iso(),
                "status": status,
                "pid": os.getpid(),
                "deadline": deadline.isoformat(),
                "plan_path": plan["_path"],
                "plan_sha256": plan["_sha256"],
                "requested_model": plan["planner"]["requested_model"],
                "served_model": plan["planner"]["served_model"],
                "prerequisites": prerequisites,
                "gpu": gpu,
                "consecutive_idle_checks": idle_checks,
                "required_idle_checks": required_idle,
                "gpu_idle_gate": calibrated_idle.policy_record(idle_policy),
                "idle_gate_evaluation": (gpu or {}).get("_idle_gate_evaluation"),
                "process_signals_sent": 0,
                "official_grader_executed": False,
                "kaggle_submission_executed": False,
            },
        )
        if prerequisites["ready"] and idle_checks >= required_idle:
            launch_plan = validate_frozen_plan(args.plan)
            if launch_plan["_sha256"] != plan["_sha256"]:
                raise RuntimeError("May-2022 frozen plan changed while queued")
            final_prerequisites = prerequisite_snapshot(launch_plan)
            if not final_prerequisites["ready"]:
                idle_checks = 0
                time.sleep(args.poll_seconds)
                continue
            final_gpu = query_gpu()
            final_evaluation = calibrated_idle.evaluate_idle(
                launch_plan["_idle_policy"], final_gpu
            )
            final_gpu["_idle_gate_evaluation"] = final_evaluation
            if not final_evaluation["idle"]:
                idle_checks = 0
                time.sleep(args.poll_seconds)
                continue
            output_root = Path(execution["output_root"])
            run_dir = output_root / "runs" / execution["run_id"]
            summary_path = run_dir / "summary.json"
            if summary_path.is_file():
                summary = read_json(summary_path)
                if summary.get("status") in {"single_seed_gate_passed", "single_seed_gate_failed"}:
                    write_json_atomic(
                        args.status,
                        {
                            "schema": "evomind.may2022.training_queue.v1",
                            "created_at": now_iso(),
                            "status": "target_run_already_complete",
                            "summary": str(summary_path),
                            "summary_sha256": sha256_file(summary_path),
                            "process_signals_sent": 0,
                            "official_grader_executed": False,
                            "kaggle_submission_executed": False,
                        },
                    )
                    return 0
            if run_dir.exists():
                write_json_atomic(
                    args.status,
                    {
                        "schema": "evomind.may2022.training_queue.v1",
                        "created_at": now_iso(),
                        "status": "target_run_already_exists",
                        "run_dir": str(run_dir),
                        "process_signals_sent": 0,
                        "official_grader_executed": False,
                        "kaggle_submission_executed": False,
                    },
                )
                return 0
            claim_path = output_root / f"{execution['run_id']}.calibrated_launch_claim.json"
            claim = {
                "schema": "evomind.may2022.calibrated_launch_claim.v1",
                "created_at": now_iso(),
                "queue_pid": os.getpid(),
                "run_id": execution["run_id"],
                "plan_path": launch_plan["_path"],
                "plan_sha256": launch_plan["_sha256"],
                "idle_gate": calibrated_idle.policy_record(
                    launch_plan["_idle_policy"]
                ),
                "process_signals_sent": 0,
                "official_grader_executed": False,
                "kaggle_submission_executed": False,
            }
            if not acquire_launch_claim(claim_path, claim):
                write_json_atomic(
                    args.status,
                    {
                        "schema": "evomind.may2022.training_queue.v1",
                        "created_at": now_iso(),
                        "status": "launch_claim_already_exists",
                        "launch_claim": str(claim_path),
                        "process_signals_sent": 0,
                        "official_grader_executed": False,
                        "kaggle_submission_executed": False,
                    },
                )
                return 0
            command = build_training_command(launch_plan)
            output_root.mkdir(parents=True, exist_ok=True)
            stdout_path = output_root / f"{execution['run_id']}.stdout.log"
            stderr_path = output_root / f"{execution['run_id']}.stderr.log"
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
            write_json_atomic(
                args.status,
                {
                    "schema": "evomind.may2022.training_queue.v1",
                    "created_at": now_iso(),
                    "status": "training_launched" if launched else "training_launch_failed",
                    "queue_pid": os.getpid(),
                    "training_pid": child.pid,
                    "training_return_code": return_code,
                    "command": command,
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                    "plan_path": launch_plan["_path"],
                    "plan_sha256": launch_plan["_sha256"],
                    "launch_claim": str(claim_path),
                    "prerequisites": final_prerequisites,
                    "gpu_before_launch": final_gpu,
                    "gpu_idle_gate": calibrated_idle.policy_record(
                        launch_plan["_idle_policy"]
                    ),
                    "idle_gate_evaluation": final_evaluation,
                    "process_signals_sent": 0,
                    "official_grader_executed": False,
                    "kaggle_submission_executed": False,
                },
            )
            return 0 if launched else 6
        time.sleep(args.poll_seconds)
    write_json_atomic(
        args.status,
        {
            "schema": "evomind.may2022.training_queue.v1",
            "created_at": now_iso(),
            "status": "timeout_waiting_for_prerequisites_or_gpu",
            "pid": os.getpid(),
            "deadline": deadline.isoformat(),
            "plan_path": plan["_path"],
            "plan_sha256": plan["_sha256"],
            "process_signals_sent": 0,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        },
    )
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
