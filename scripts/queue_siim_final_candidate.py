#!/usr/bin/env python3
"""Queue the final SIIM candidate ahead of downstream single-GPU work."""
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
DEFAULT_PLAN = PROJECT_ROOT / "workspace" / "mlebench_plans" / "siim_final_candidate_frozen_plan_v2_20260728.json"
DEFAULT_STATUS = PROJECT_ROOT / "workspace" / "local_gpu" / "siim_final_candidate_queue.json"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_artifact(name: str, artifact: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(artifact["path"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen artifact {name}: {path}")
    actual = sha256_file(path)
    expected = str(artifact["sha256"]).lower()
    if actual != expected:
        raise RuntimeError(f"Frozen artifact changed for {name}: {path}")
    return {
        "name": name,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": actual,
    }


def validate_frozen_plan(path: Path) -> dict[str, Any]:
    plan_path = path.resolve()
    plan = read_json(plan_path)
    if plan.get("schema") != "evomind.siim.final_candidate_frozen_plan.v1":
        raise RuntimeError("Unexpected SIIM final candidate plan schema")
    if plan.get("status") != "frozen_waiting_prerequisites":
        raise RuntimeError("SIIM final candidate plan is not frozen")
    if plan.get("competition_id") != "siim-isic-melanoma-classification":
        raise RuntimeError("SIIM final candidate plan targets a different competition")
    planner = plan.get("planner") or {}
    if (
        planner.get("requested_model") != "gpt-5.6-sol"
        or planner.get("served_model") != "gpt-5.6-sol"
    ):
        raise RuntimeError("SIIM final candidate plan lacks live gpt-5.6-sol evidence")

    validated: list[dict[str, Any]] = []
    for name, artifact in (planner.get("evidence") or {}).items():
        validated.append(validate_artifact(f"planner.{name}", artifact))
    for name, artifact in (plan.get("implementation") or {}).items():
        validated.append(validate_artifact(f"implementation.{name}", artifact))
    validated.append(
        validate_artifact("serial_dependency.ablation_plan", plan["serial_dependency"]["plan"])
    )
    validated.append(validate_artifact("data_contract.inventory", plan["data_contract"]["inventory"]))

    training = plan["training"]
    if training.get("candidate_only") is not True:
        raise RuntimeError("SIIM final plan must withhold the official grader")
    if training.get("hold_cuda_lease") is not True:
        raise RuntimeError("SIIM final plan must hold the local single-GPU lease")
    if training.get("private_labels_used") is not False:
        raise RuntimeError("SIIM final plan violates the public-data-only contract")
    if training.get("official_grader_executed") is not False:
        raise RuntimeError("SIIM final plan enables the official grader")
    if training.get("kaggle_submission_executed") is not False:
        raise RuntimeError("SIIM final plan enables Kaggle submission")
    if training.get("workers") != 0:
        raise RuntimeError("Windows SIIM final plan must use a process-local DataLoader")
    if training.get("batch_probe_candidates") != [16, 8]:
        raise RuntimeError("SIIM final hardware-probe candidates changed")
    plan["_idle_policy"] = calibrated_idle.validate_launch_contract(
        plan["launch_contract"]
    )

    plan["_plan_path"] = str(plan_path)
    plan["_plan_sha256"] = sha256_file(plan_path)
    plan["_validated_artifacts"] = validated
    return plan


def staging_snapshot(plan: dict[str, Any]) -> dict[str, Any]:
    report_path = Path(plan["data_contract"]["staging_report"]).resolve()
    if not report_path.is_file():
        return {"ready": False, "status": "missing", "report": str(report_path)}
    try:
        report = read_json(report_path)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        return {
            "ready": False,
            "status": "invalid_json",
            "report": str(report_path),
            "error": str(exc),
        }
    expected = plan["data_contract"]
    errors = report.get("errors") or []
    checks = {
        "schema": report.get("schema") == "evomind.siim.public_staging.v1",
        "competition": report.get("competition_id") == plan["competition_id"],
        "status": report.get("status") == "size_verified_complete",
        "files": report.get("completed_files") == expected["file_count"],
        "total_files": report.get("total_files") == expected["file_count"],
        "bytes": report.get("completed_bytes") == expected["total_bytes"],
        "total_bytes": report.get("total_bytes") == expected["total_bytes"],
        "manifest": report.get("inventory_manifest_sha256") == expected["manifest_sha256"],
        "no_errors": not errors,
        "no_private_paths": report.get("private_paths_requested") is False,
        "no_remote_writes": report.get("remote_writes_performed") is False,
        "no_process_signals": report.get("process_signals_sent") == 0,
        "no_official_grader": report.get("official_grader_executed") is False,
        "no_kaggle_submission": report.get("kaggle_submission_executed") is False,
    }
    return {
        "ready": all(checks.values()),
        "status": report.get("status", "unknown"),
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "checks": checks,
        "completed_files": report.get("completed_files", 0),
        "total_files": report.get("total_files", expected["file_count"]),
        "completed_bytes": report.get("completed_bytes", 0),
        "total_bytes": report.get("total_bytes", expected["total_bytes"]),
        "error_count": len(errors),
    }


def ablation_snapshot(plan: dict[str, Any]) -> dict[str, Any]:
    serial = plan["serial_dependency"]
    report_path = Path(serial["report"]).resolve()
    if not report_path.is_file():
        return {"ready": False, "status": "missing", "report": str(report_path)}
    try:
        report = read_json(report_path)
        ablation_plan = read_json(Path(serial["plan"]["path"]).resolve())
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        return {
            "ready": False,
            "status": "invalid_json",
            "report": str(report_path),
            "error": str(exc),
        }
    training = ablation_plan["training"]
    profiles = training["profiles"]
    implementation = plan["implementation"]
    checks = {
        "schema": report.get("schema") == "evomind.siim_preprocessing_ablation.v1",
        "competition": report.get("competition_id") == plan["competition_id"],
        "run_id": report.get("run_id") == training["run_id"],
        "passed": report.get("passed") is True,
        "full_public_train_scope": report.get("full_public_train_scope") is True,
        "patient_content_group_isolation": report.get("patient_content_group_isolation") is True,
        "validation_coverage": report.get("validation_coverage_exactly_once") is True,
        "evaluation_seeds": report.get("evaluation_seeds") == training["evaluation_seeds"],
        "fold_count": report.get("fold_count") == training["folds"],
        "profile_order": report.get("profile_order") == profiles,
        "selected_profile": report.get("selected_profile") in profiles,
        "adapter_hash": report.get("adapter_source_sha256")
        == implementation["adapter"]["sha256"],
        "wave2_hash": report.get("wave2_source_sha256")
        == implementation["wave2"]["sha256"],
        "no_private_labels": report.get("private_labels_used_for_training") is False,
        "no_official_grader": report.get("official_grader_executed") is False,
        "no_kaggle_submission": report.get("kaggle_submission_executed") is False,
        "no_official_score_claim": report.get("official_score_claimed") is False,
    }
    return {
        "ready": all(checks.values()),
        "status": "validated_complete" if all(checks.values()) else "invalid_contract",
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "checks": checks,
        "selected_profile": report.get("selected_profile"),
    }


def prerequisite_snapshot(plan: dict[str, Any]) -> dict[str, Any]:
    staging = staging_snapshot(plan)
    ablation = ablation_snapshot(plan)
    return {
        "staging": staging,
        "ablation": ablation,
        "ready": staging["ready"] and ablation["ready"],
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
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [value.strip() for value in gpu.stdout.splitlines()[0].split(",")]
    compute = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    applications = parse_compute_apps(compute.stdout)
    python_compute = [
        app for app in applications if "python" in Path(app["process_name"]).name.lower()
    ]
    return {
        "name": values[0],
        "utilization_percent": int(values[1]),
        "memory_used_mib": int(values[2]),
        "memory_total_mib": int(values[3]),
        "temperature_c": int(values[4]),
        "compute_applications": applications,
        "python_compute_applications": python_compute,
    }


def run_batch_probes(plan: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    training = plan["training"]
    probe_root = Path(training["probe_output_root"]).resolve()
    probe_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    passing: list[int] = []
    for batch_size in training["batch_probe_candidates"]:
        report_path = probe_root / f"batch_{batch_size}.json"
        command = [
            training["python"],
            plan["implementation"]["batch_probe"]["path"],
            "--report",
            str(report_path),
            "--batch-size",
            str(batch_size),
            "--image-size",
            str(training["image_size"]),
            "--metadata-width",
            str(training["probe_metadata_width"]),
            "--full-backbone",
            training["backbone"],
            "--lesion-backbone",
            training["secondary_backbone"],
            "--torch-home",
            training["torch_home"],
            "--expected-gpu-name",
            plan["resource"]["expected_gpu_name"],
        ]
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        report = read_json(report_path) if report_path.is_file() else {
            "status": "missing_report",
            "error": completed.stderr[-4000:],
        }
        record = {
            "batch_size": batch_size,
            "command": command,
            "return_code": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
            "report": str(report_path),
            "report_sha256": sha256_file(report_path) if report_path.is_file() else None,
            "payload": report,
        }
        records.append(record)
        if (
            completed.returncode == 0
            and report.get("status") == "passed"
            and report.get("batch_size") == batch_size
            and report.get("adapter_source_sha256")
            == plan["implementation"]["adapter"]["sha256"]
            and report.get("process_signals_sent") == 0
            and report.get("official_grader_executed") is False
            and report.get("kaggle_submission_executed") is False
        ):
            passing.append(batch_size)
    if not passing:
        raise RuntimeError("No frozen SIIM batch candidate passed the isolated CUDA probe")
    return max(passing), records


def build_training_command(
    plan: dict[str, Any],
    *,
    selected_profile: str,
    selected_batch_size: int,
    ablation_report: str,
) -> list[str]:
    training = plan["training"]
    command = [
        training["python"],
        plan["implementation"]["full_runner"]["path"],
        "--data-root",
        training["data_root"],
        "--output-root",
        training["output_root"],
        "--allowed-root",
        training["allowed_root"],
        "--official-source-root",
        training["official_source_root"],
        "--waves",
        "Wave0",
        "--competitions",
        plan["competition_id"],
        "--run-id",
        training["run_id"],
        "--seed",
        str(training["seed"]),
        "--phase-a-scope",
        "requested",
        "--optimization-plan",
        plan["planner"]["evidence"]["execution_plan"]["path"],
        "--candidate-only",
        "--hold-cuda-lease",
        "--siim-preprocessing-profile",
        selected_profile,
        "--siim-preprocessing-ablation-report",
        ablation_report,
        "--siim-backbone",
        training["backbone"],
        "--siim-secondary-backbone",
        training["secondary_backbone"],
        "--siim-epochs",
        str(training["epochs"]),
        "--siim-folds",
        str(training["outer_folds"]),
        "--siim-inner-folds",
        str(training["inner_folds"]),
        "--siim-batch-size",
        str(selected_batch_size),
        "--siim-image-size",
        str(training["image_size"]),
        "--siim-workers",
        str(training["workers"]),
        "--siim-learning-rate",
        str(training["learning_rate"]),
        "--siim-metadata-iterations",
        str(training["metadata_iterations"]),
        "--siim-catboost-task-type",
        training["catboost_task_type"],
        "--wave2-fast-kernels",
    ]
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--deadline-hours", type=float, default=240.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if compute_policy.write_queue_superseded_if_hpc_only(
        project_root=PROJECT_ROOT,
        status_path=args.status,
        schema="evomind.siim.final_candidate_queue.v1",
        queue_name="queue_siim_final_candidate",
        plan_path=args.plan,
    ):
        return 0
    if args.poll_seconds < 10 or args.deadline_hours <= 0:
        raise ValueError("SIIM final queue timing contract is invalid")
    plan = validate_frozen_plan(args.plan)
    idle_policy = plan["_idle_policy"]
    required_idle = int(idle_policy["requirements"]["consecutive_checks"])
    minimum_interval = int(
        idle_policy["requirements"]["minimum_check_interval_seconds"]
    )
    if args.poll_seconds < minimum_interval:
        raise ValueError("SIIM final polling is faster than calibrated idle policy")
    deadline = datetime.now().astimezone() + timedelta(hours=args.deadline_hours)
    idle_checks = 0

    while datetime.now().astimezone() < deadline:
        prerequisites = prerequisite_snapshot(plan)
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
            args.status.resolve(),
            {
                "schema": "evomind.siim.final_candidate_queue.v1",
                "created_at": now_iso(),
                "status": status,
                "pid": os.getpid(),
                "deadline": deadline.isoformat(),
                "plan_path": plan["_plan_path"],
                "plan_sha256": plan["_plan_sha256"],
                "validated_artifacts": plan["_validated_artifacts"],
                "requested_model": plan["planner"]["requested_model"],
                "served_model": plan["planner"]["served_model"],
                "prerequisites": prerequisites,
                "gpu": gpu,
                "consecutive_idle_checks": idle_checks,
                "required_idle_checks": required_idle,
                "gpu_idle_gate": calibrated_idle.policy_record(idle_policy),
                "idle_gate_evaluation": (gpu or {}).get("_idle_gate_evaluation"),
                "process_signals_sent": 0,
                "private_labels_used": False,
                "official_grader_executed": False,
                "kaggle_submission_executed": False,
            },
        )
        if prerequisites["ready"] and idle_checks >= required_idle:
            launch_plan = validate_frozen_plan(args.plan)
            if launch_plan["_plan_sha256"] != plan["_plan_sha256"]:
                raise RuntimeError("SIIM final frozen plan changed while queued")
            launch_prerequisites = prerequisite_snapshot(launch_plan)
            if not launch_prerequisites["ready"]:
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

            selected_batch, probes = run_batch_probes(launch_plan)
            selected_profile = launch_prerequisites["ablation"]["selected_profile"]
            training = launch_plan["training"]
            run_dir = Path(training["output_root"]).resolve() / training["run_id"]
            if run_dir.exists():
                write_json_atomic(
                    args.status.resolve(),
                    {
                        "schema": "evomind.siim.final_candidate_queue.v1",
                        "created_at": now_iso(),
                        "status": "target_run_already_exists",
                        "queue_pid": os.getpid(),
                        "run_dir": str(run_dir),
                        "plan_path": launch_plan["_plan_path"],
                        "plan_sha256": launch_plan["_plan_sha256"],
                        "batch_probes": probes,
                        "process_signals_sent": 0,
                        "official_grader_executed": False,
                        "kaggle_submission_executed": False,
                    },
                )
                return 5

            command = build_training_command(
                launch_plan,
                selected_profile=selected_profile,
                selected_batch_size=selected_batch,
                ablation_report=launch_prerequisites["ablation"]["report"],
            )
            if compute_policy.write_queue_superseded_if_hpc_only(
                project_root=PROJECT_ROOT,
                status_path=args.status,
                schema="evomind.siim.final_candidate_queue.v1",
                queue_name="queue_siim_final_candidate",
                plan_path=args.plan,
            ):
                return 0
            output_root = Path(training["output_root"]).resolve()
            output_root.mkdir(parents=True, exist_ok=True)
            stdout_path = output_root / f"{training['run_id']}.stdout.log"
            stderr_path = output_root / f"{training['run_id']}.stderr.log"
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
            time.sleep(15)
            return_code = child.poll()
            launched = return_code is None
            write_json_atomic(
                args.status.resolve(),
                {
                    "schema": "evomind.siim.final_candidate_queue.v1",
                    "created_at": now_iso(),
                    "status": "training_launched" if launched else "training_launch_failed",
                    "queue_pid": os.getpid(),
                    "training_pid": child.pid,
                    "training_return_code": return_code,
                    "run_dir": str(run_dir),
                    "command": command,
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                    "selected_profile": selected_profile,
                    "selected_batch_size": selected_batch,
                    "batch_probes": probes,
                    "plan_path": launch_plan["_plan_path"],
                    "plan_sha256": launch_plan["_plan_sha256"],
                    "validated_artifacts": launch_plan["_validated_artifacts"],
                    "requested_model": launch_plan["planner"]["requested_model"],
                    "served_model": launch_plan["planner"]["served_model"],
                    "prerequisites": launch_prerequisites,
                    "gpu_before_probe": final_gpu,
                    "gpu_idle_gate": calibrated_idle.policy_record(
                        launch_plan["_idle_policy"]
                    ),
                    "idle_gate_evaluation": final_evaluation,
                    "process_signals_sent": 0,
                    "private_labels_used": False,
                    "official_grader_executed": False,
                    "kaggle_submission_executed": False,
                },
            )
            return 0 if launched else 6
        time.sleep(args.poll_seconds)

    write_json_atomic(
        args.status.resolve(),
        {
            "schema": "evomind.siim.final_candidate_queue.v1",
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
