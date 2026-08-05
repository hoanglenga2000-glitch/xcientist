#!/usr/bin/env python3
"""Launch the frozen SIIM ablation only after data, verifier, and GPU-idle gates pass."""

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
import mlebench_compute_policy as compute_policy  # noqa: E402

PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_PLAN = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "siim_preprocessing_ablation_frozen_plan_v2_20260728.json"
)
DEFAULT_JIGSAW_WATCHER = (
    PROJECT_ROOT
    / "workspace"
    / "local_gpu"
    / "mlebench_lite_runs"
    / "local4060_jigsaw_xfmr1_s42_20260727_0024"
    / "independent_verification_watcher.json"
)
DEFAULT_STAGING_REPORT = (
    PROJECT_ROOT
    / "workspace"
    / "local_gpu"
    / "mlebench_official_data"
    / "siim-isic-melanoma-classification"
    / "public_staging_report.json"
)
DEFAULT_STATUS = PROJECT_ROOT / "workspace" / "local_gpu" / "siim_training_queue.json"


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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_frozen_plan(plan_path: Path) -> dict[str, Any]:
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    if plan.get("schema") != "evomind.siim.preprocessing_ablation_frozen_plan.v1":
        raise RuntimeError("Unexpected SIIM frozen-plan schema")
    if plan.get("competition_id") != "siim-isic-melanoma-classification":
        raise RuntimeError("Unexpected SIIM frozen-plan competition")
    planner = plan.get("planner", {})
    if (
        planner.get("requested_model") != "gpt-5.6-sol"
        or planner.get("served_model") != "gpt-5.6-sol"
        or not planner.get("parse_ok")
    ):
        raise RuntimeError("SIIM frozen plan lacks verified gpt-5.6-sol provenance")
    training = plan.get("training", {})
    if training.get("private_labels_used") is not False:
        raise RuntimeError("SIIM frozen plan violates the private-label contract")
    if training.get("official_grader_executed") is not False:
        raise RuntimeError("SIIM frozen plan enables the official grader")
    if training.get("kaggle_submission_executed") is not False:
        raise RuntimeError("SIIM frozen plan enables Kaggle submission")
    if training.get("evaluation_seeds") != [40, 41, 42]:
        raise RuntimeError("SIIM external evaluation seeds are not frozen")
    if len(training.get("profiles", [])) != 5:
        raise RuntimeError("SIIM frozen plan does not cover all five profiles")
    for key in ("python", "script", "data_root", "output_root", "torch_home"):
        if not training.get(key):
            raise RuntimeError(f"SIIM frozen plan is missing training.{key}")
    for name, artifact in plan.get("implementation", {}).items():
        path = Path(artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"SIIM implementation artifact changed: {name}")
    planner_artifact = planner["source"]
    planner_path = Path(planner_artifact["path"])
    if not planner_path.is_file() or sha256_file(planner_path) != planner_artifact["sha256"]:
        raise RuntimeError("SIIM gpt-5.6-sol planner evidence changed")
    inventory_artifact = plan["data_contract"]["inventory"]
    inventory_path = Path(inventory_artifact["path"])
    if not inventory_path.is_file() or sha256_file(inventory_path) != inventory_artifact["sha256"]:
        raise RuntimeError("SIIM public inventory changed after plan freeze")
    plan["_idle_policy"] = calibrated_idle.validate_launch_contract(
        plan["launch_contract"]
    )
    may_dependency = plan.get("serial_dependency", {}).get("may2022") or {}
    may_plan_path = Path(str(may_dependency.get("plan_path", ""))).resolve()
    if (
        not may_plan_path.is_file()
        or sha256_file(may_plan_path) != may_dependency.get("plan_sha256")
    ):
        raise RuntimeError("SIIM queue May-2022 serial dependency changed")
    plan["_plan_path"] = str(plan_path)
    plan["_plan_sha256"] = sha256_file(plan_path)
    return plan


def prerequisite_snapshot(
    plan: dict[str, Any],
    *,
    jigsaw_watcher_path: Path,
    staging_report_path: Path,
    may_summary_path: Path | None = None,
) -> dict[str, Any]:
    allowed_jigsaw = set(
        plan["launch_contract"]["wait_for_jigsaw_verification_status"]
    )
    jigsaw = read_json(jigsaw_watcher_path) if Path(jigsaw_watcher_path).is_file() else {}
    staging = read_json(staging_report_path) if Path(staging_report_path).is_file() else {}
    may_dependency = plan["serial_dependency"]["may2022"]
    resolved_may_summary = Path(
        may_summary_path or may_dependency["summary_path"]
    ).resolve()
    may = read_json(resolved_may_summary) if resolved_may_summary.is_file() else {}
    expected = plan["data_contract"]
    staging_errors = staging.get("errors") or []
    jigsaw_ready = jigsaw.get("status") in allowed_jigsaw
    staging_ready = all(
        [
            staging.get("status")
            == plan["launch_contract"]["wait_for_siim_staging_status"],
            staging.get("completed_files") == expected["file_count"],
            staging.get("completed_bytes") == expected["total_bytes"],
            staging.get("inventory_manifest_sha256") == expected["manifest_sha256"],
            not staging_errors,
            staging.get("private_paths_requested") is False,
            staging.get("remote_writes_performed") is False,
            staging.get("process_signals_sent") == 0,
        ]
    )
    may_ready = all(
        [
            may.get("schema") == "evomind.mlebench.may2022_compact_embedding_run.v1",
            may.get("run_id") == may_dependency["run_id"],
            may.get("status") in set(may_dependency["terminal_statuses"]),
            may.get("plan_sha256") == may_dependency["plan_sha256"],
            may.get("private_labels_used") is False,
            may.get("official_grader_executed") is False,
            may.get("kaggle_submission_executed") is False,
            may.get("process_signals_sent") == 0,
        ]
    )
    return {
        "jigsaw_ready": jigsaw_ready,
        "jigsaw_status": jigsaw.get("status", "missing"),
        "jigsaw_watcher": str(Path(jigsaw_watcher_path).resolve()),
        "staging_ready": staging_ready,
        "staging_status": staging.get("status", "missing"),
        "staging_completed_files": staging.get("completed_files", 0),
        "staging_total_files": staging.get("total_files", expected["file_count"]),
        "staging_completed_bytes": staging.get("completed_bytes", 0),
        "staging_total_bytes": staging.get("total_bytes", expected["total_bytes"]),
        "staging_error_count": len(staging_errors),
        "staging_report": str(Path(staging_report_path).resolve()),
        "may2022_ready": may_ready,
        "may2022_status": may.get("status", "missing"),
        "may2022_summary": str(resolved_may_summary),
        "ready": jigsaw_ready and staging_ready and may_ready,
    }


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
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    values = [value.strip() for value in gpu.stdout.splitlines()[0].split(",")]
    utilization, memory_used, memory_total, temperature = map(int, values[:4])
    compute = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    applications = parse_compute_apps(compute.stdout)
    python_compute = [
        app
        for app in applications
        if "python" in Path(app["process_name"]).name.lower()
    ]
    return {
        "utilization_percent": utilization,
        "memory_used_mib": memory_used,
        "memory_total_mib": memory_total,
        "temperature_c": temperature,
        "compute_applications": applications,
        "python_compute_applications": python_compute,
    }


def build_training_command(plan: dict[str, Any]) -> list[str]:
    training = plan["training"]
    return [
        training["python"],
        training["script"],
        "--data-root",
        training["data_root"],
        "--output-root",
        training["output_root"],
        "--run-id",
        training["run_id"],
        "--seed",
        str(training["seed"]),
        "--evaluation-seeds",
        ",".join(str(value) for value in training["evaluation_seeds"]),
        "--folds",
        str(training["folds"]),
        "--image-size",
        str(training["image_size"]),
        "--batch-size",
        str(training["batch_size"]),
        "--workers",
        str(training["workers"]),
        "--manifest-workers",
        str(training["manifest_workers"]),
        "--full-backbone",
        training["full_backbone"],
        "--lesion-backbone",
        training["lesion_backbone"],
        "--linear-alpha",
        str(training["linear_alpha"]),
        "--linear-max-iter",
        str(training["linear_max_iter"]),
        "--minimum-mean-gain",
        str(training["minimum_mean_gain"]),
        "--maximum-worst-fold-regression",
        str(training["maximum_worst_fold_regression"]),
        "--maximum-seed-mean-regression",
        str(training["maximum_seed_mean_regression"]),
        "--minimum-seed-pass-fraction",
        str(training["minimum_seed_pass_fraction"]),
        "--torch-home",
        training["torch_home"],
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--jigsaw-watcher", type=Path, default=DEFAULT_JIGSAW_WATCHER)
    parser.add_argument("--staging-report", type=Path, default=DEFAULT_STAGING_REPORT)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--deadline-hours", type=float, default=36.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if compute_policy.write_queue_superseded_if_hpc_only(
        project_root=PROJECT_ROOT,
        status_path=args.status,
        schema="evomind.siim.training_queue.v1",
        queue_name="queue_siim_after_jigsaw",
        plan_path=args.plan,
    ):
        return 0
    if args.poll_seconds < 10 or args.deadline_hours <= 0:
        raise ValueError("SIIM queue timing contract is invalid")
    plan = validate_frozen_plan(args.plan)
    deadline = datetime.now().astimezone() + timedelta(hours=args.deadline_hours)
    idle_policy = plan["_idle_policy"]
    required_idle = int(idle_policy["requirements"]["consecutive_checks"])
    minimum_interval = int(
        idle_policy["requirements"]["minimum_check_interval_seconds"]
    )
    if args.poll_seconds < minimum_interval:
        raise ValueError("SIIM polling is faster than the calibrated idle interval")
    idle_checks = 0
    while datetime.now().astimezone() < deadline:
        prerequisites = prerequisite_snapshot(
            plan,
            jigsaw_watcher_path=args.jigsaw_watcher,
            staging_report_path=args.staging_report,
        )
        gpu: dict[str, Any] | None = None
        status = "waiting_for_prerequisites"
        if prerequisites["ready"]:
            gpu = calibrated_idle.query_gpu()
            evaluation = calibrated_idle.evaluate_idle(idle_policy, gpu)
            gpu["_idle_gate_evaluation"] = evaluation
            is_idle = bool(evaluation["idle"])
            idle_checks = idle_checks + 1 if is_idle else 0
            status = "waiting_for_stable_gpu_idle"
        write_json_atomic(
            args.status,
            {
                "schema": "evomind.siim.training_queue.v1",
                "created_at": now_iso(),
                "status": status,
                "pid": os.getpid(),
                "deadline": deadline.isoformat(),
                "plan_path": plan["_plan_path"],
                "plan_sha256": plan["_plan_sha256"],
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
            if launch_plan["_plan_sha256"] != plan["_plan_sha256"]:
                raise RuntimeError("SIIM frozen plan changed while queued")
            final_prerequisites = prerequisite_snapshot(
                launch_plan,
                jigsaw_watcher_path=args.jigsaw_watcher,
                staging_report_path=args.staging_report,
            )
            if not final_prerequisites["ready"]:
                idle_checks = 0
                time.sleep(args.poll_seconds)
                continue
            final_gpu = calibrated_idle.query_gpu()
            final_evaluation = calibrated_idle.evaluate_idle(
                launch_plan["_idle_policy"], final_gpu
            )
            if not final_evaluation["idle"]:
                idle_checks = 0
                time.sleep(args.poll_seconds)
                continue
            if compute_policy.write_queue_superseded_if_hpc_only(
                project_root=PROJECT_ROOT,
                status_path=args.status,
                schema="evomind.siim.training_queue.v1",
                queue_name="queue_siim_after_jigsaw",
                plan_path=args.plan,
            ):
                return 0
            command = build_training_command(launch_plan)
            output_root = Path(plan["training"]["output_root"])
            output_root.mkdir(parents=True, exist_ok=True)
            stdout_path = output_root / f"{plan['training']['run_id']}.stdout.log"
            stderr_path = output_root / f"{plan['training']['run_id']}.stderr.log"
            environment = os.environ.copy()
            python_path = os.pathsep.join(
                [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]
            )
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
                    "schema": "evomind.siim.training_queue.v1",
                    "created_at": now_iso(),
                    "status": "training_launched" if launched else "training_launch_failed",
                    "queue_pid": os.getpid(),
                    "training_pid": child.pid,
                    "training_return_code": return_code,
                    "command": command,
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                    "plan_path": plan["_plan_path"],
                    "plan_sha256": plan["_plan_sha256"],
                    "requested_model": plan["planner"]["requested_model"],
                    "served_model": plan["planner"]["served_model"],
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
            "schema": "evomind.siim.training_queue.v1",
            "created_at": now_iso(),
            "status": "timeout_waiting_for_prerequisites_or_gpu",
            "pid": os.getpid(),
            "deadline": deadline.isoformat(),
            "plan_path": plan["_plan_path"],
            "plan_sha256": plan["_plan_sha256"],
            "process_signals_sent": 0,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        },
    )
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
