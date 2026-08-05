#!/usr/bin/env python3
"""Launch the frozen Leaf run after SIIM, data, and stable GPU-idle gates pass."""

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
    / "leaf_multibackbone_frozen_plan_v2_20260728.json"
)
DEFAULT_STAGING_REPORT = (
    PROJECT_ROOT
    / "workspace"
    / "local_gpu"
    / "mlebench_official_data"
    / "leaf-classification"
    / "public_staging_report.json"
)
DEFAULT_STATUS = PROJECT_ROOT / "workspace" / "local_gpu" / "leaf_training_queue.json"


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


def validate_artifact(name: str, artifact: dict[str, Any]) -> dict[str, Any]:
    path = Path(artifact.get("path", "")).resolve()
    expected_sha256 = str(artifact.get("sha256", ""))
    expected_bytes = artifact.get("bytes")
    if not path.is_file():
        raise RuntimeError(f"Leaf frozen artifact is missing: {name}")
    actual_bytes = path.stat().st_size
    if expected_bytes is not None and actual_bytes != int(expected_bytes):
        raise RuntimeError(f"Leaf frozen artifact size changed: {name}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"Leaf frozen artifact hash changed: {name}")
    return {
        "name": name,
        "path": str(path),
        "bytes": actual_bytes,
        "sha256": actual_sha256,
    }


def validate_frozen_plan(plan_path: Path) -> dict[str, Any]:
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    if plan.get("schema") != "evomind.leaf.multibackbone_frozen_plan.v1":
        raise RuntimeError("Unexpected Leaf frozen-plan schema")
    if plan.get("competition_id") != "leaf-classification":
        raise RuntimeError("Unexpected Leaf frozen-plan competition")
    planner = plan.get("planner", {})
    if (
        planner.get("requested_model") != "gpt-5.6-sol"
        or planner.get("served_model") != "gpt-5.6-sol"
        or not planner.get("parse_ok")
    ):
        raise RuntimeError("Leaf frozen plan lacks verified gpt-5.6-sol provenance")

    training = plan.get("training", {})
    if training.get("private_labels_used") is not False:
        raise RuntimeError("Leaf frozen plan violates the private-label contract")
    if training.get("official_grader_executed") is not False:
        raise RuntimeError("Leaf frozen plan enables the official grader")
    if training.get("kaggle_submission_executed") is not False:
        raise RuntimeError("Leaf frozen plan enables Kaggle submission")
    if training.get("seeds") != [40, 41, 42]:
        raise RuntimeError("Leaf external evaluation seeds are not frozen")
    if training.get("folds") != 5:
        raise RuntimeError("Leaf grouped fold count is not frozen")
    if training.get("backbones") != ["convnext_small", "efficientnet_v2_s"]:
        raise RuntimeError("Leaf dual-backbone contract changed")
    for key in (
        "python",
        "script",
        "run_id",
        "data_root",
        "output_root",
        "torch_home",
    ):
        if not training.get(key):
            raise RuntimeError(f"Leaf frozen plan is missing training.{key}")

    objective = plan.get("objective", {})
    if objective.get("promotion_mean_log_loss") != 0.0135:
        raise RuntimeError("Leaf mean OOF promotion gate changed")
    if objective.get("promotion_max_seed_log_loss") != 0.0135:
        raise RuntimeError("Leaf worst-seed promotion gate changed")

    data_contract = plan.get("data_contract", {})
    if (
        data_contract.get("file_count") != 994
        or data_contract.get("total_bytes") != 28_617_101
        or data_contract.get("image_count") != 990
    ):
        raise RuntimeError("Leaf public-data contract changed")
    if data_contract.get("private_paths_requested") is not False:
        raise RuntimeError("Leaf plan requests private paths")
    if data_contract.get("remote_writes_performed") is not False:
        raise RuntimeError("Leaf plan permits remote writes")

    validated_artifacts: list[dict[str, Any]] = []
    validated_artifacts.append(
        validate_artifact("planner.source", planner.get("source", {}))
    )
    validated_artifacts.append(
        validate_artifact("data_contract.inventory", data_contract.get("inventory", {}))
    )
    for name, artifact in plan.get("implementation", {}).items():
        validated_artifacts.append(
            validate_artifact(f"implementation.{name}", artifact)
        )
    serial_dependency = plan.get("serial_dependency", {})
    validated_artifacts.append(
        validate_artifact(
            "serial_dependency.siim_plan",
            serial_dependency.get("siim_plan", {}),
        )
    )
    validated_artifacts.append(
        validate_artifact(
            "serial_dependency.siim_final_plan",
            serial_dependency.get("siim_final", {}).get("plan", {}),
        )
    )

    inventory = read_json(Path(data_contract["inventory"]["path"]))
    if (
        inventory.get("schema") != "evomind.leaf.public_staging_inventory.v1"
        or inventory.get("competition_id") != "leaf-classification"
        or inventory.get("file_count") != data_contract["file_count"]
        or inventory.get("total_bytes") != data_contract["total_bytes"]
        or inventory.get("image_count") != data_contract["image_count"]
        or inventory.get("manifest_sha256") != data_contract["manifest_sha256"]
        or inventory.get("private_paths_requested") is not False
        or inventory.get("remote_writes_performed") is not False
    ):
        raise RuntimeError("Leaf frozen inventory content changed")

    siim_plan_path = Path(serial_dependency["siim_plan"]["path"])
    siim_plan = read_json(siim_plan_path)
    if siim_plan.get("schema") != "evomind.siim.preprocessing_ablation_frozen_plan.v1":
        raise RuntimeError("Unexpected serial SIIM frozen-plan schema")
    if siim_plan.get("training", {}).get("run_id") != serial_dependency.get(
        "siim_run_id"
    ):
        raise RuntimeError("Leaf serial dependency points to a different SIIM run")
    if siim_plan.get("training", {}).get("evaluation_seeds") != [40, 41, 42]:
        raise RuntimeError("Leaf serial SIIM seed contract changed")
    if siim_plan.get("training", {}).get("private_labels_used") is not False:
        raise RuntimeError("Leaf serial SIIM plan violates the private-label contract")
    if siim_plan.get("training", {}).get("official_grader_executed") is not False:
        raise RuntimeError("Leaf serial SIIM plan enables the official grader")
    if siim_plan.get("training", {}).get("kaggle_submission_executed") is not False:
        raise RuntimeError("Leaf serial SIIM plan enables Kaggle submission")

    siim_final_dependency = serial_dependency["siim_final"]
    siim_final_plan = read_json(Path(siim_final_dependency["plan"]["path"]))
    if (
        siim_final_plan.get("schema")
        != "evomind.siim.final_candidate_frozen_plan.v1"
        or siim_final_plan.get("training", {}).get("run_id")
        != siim_final_dependency["run_id"]
    ):
        raise RuntimeError("Leaf serial dependency points to a different SIIM final run")
    plan["_idle_policy"] = calibrated_idle.validate_launch_contract(
        plan["launch_contract"]
    )

    plan["_plan_path"] = str(plan_path)
    plan["_plan_sha256"] = sha256_file(plan_path)
    plan["_validated_artifacts"] = validated_artifacts
    plan["_siim_plan"] = siim_plan
    plan["_siim_final_plan"] = siim_final_plan
    return plan


def staging_snapshot(plan: dict[str, Any], report_path: Path) -> dict[str, Any]:
    path = Path(report_path).resolve()
    if not path.is_file():
        return {
            "ready": False,
            "status": "missing",
            "report": str(path),
        }
    try:
        report = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "ready": False,
            "status": "invalid_json",
            "report": str(path),
            "error": str(exc),
        }
    expected = plan["data_contract"]
    errors = report.get("errors") or []
    checks = {
        "schema": report.get("schema") == "evomind.leaf.public_staging.v1",
        "competition": report.get("competition_id") == "leaf-classification",
        "status": report.get("status") == "size_verified_complete",
        "files": report.get("completed_files") == expected["file_count"],
        "total_files": report.get("total_files") == expected["file_count"],
        "bytes": report.get("completed_bytes") == expected["total_bytes"],
        "total_bytes": report.get("total_bytes") == expected["total_bytes"],
        "manifest": report.get("inventory_manifest_sha256")
        == expected["manifest_sha256"],
        "errors": not errors,
        "no_private_paths": report.get("private_paths_requested") is False,
        "no_remote_writes": report.get("remote_writes_performed") is False,
        "no_process_signals": report.get("process_signals_sent") == 0,
        "no_official_grader": report.get("official_grader_executed") is False,
        "no_kaggle_submission": report.get("kaggle_submission_executed") is False,
    }
    return {
        "ready": all(checks.values()),
        "status": report.get("status", "missing"),
        "report": str(path),
        "report_sha256": sha256_file(path),
        "checks": checks,
        "completed_files": report.get("completed_files", 0),
        "total_files": report.get("total_files", expected["file_count"]),
        "completed_bytes": report.get("completed_bytes", 0),
        "total_bytes": report.get("total_bytes", expected["total_bytes"]),
        "error_count": len(errors),
    }


def siim_report_snapshot(
    plan: dict[str, Any], report_path: Path | None = None
) -> dict[str, Any]:
    serial = plan["serial_dependency"]
    path = Path(report_path or serial["siim_report"]).resolve()
    if not path.is_file():
        return {
            "ready": False,
            "status": "missing",
            "report": str(path),
        }
    try:
        report = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "ready": False,
            "status": "invalid_json",
            "report": str(path),
            "error": str(exc),
        }
    siim_plan = plan["_siim_plan"]
    siim_training = siim_plan["training"]
    profiles = siim_training["profiles"]
    checks = {
        "schema": report.get("schema")
        == "evomind.siim_preprocessing_ablation.v1",
        "competition": report.get("competition_id")
        == "siim-isic-melanoma-classification",
        "run_id": report.get("run_id") == serial["siim_run_id"],
        "passed": report.get("passed") is True,
        "full_public_train_scope": report.get("full_public_train_scope") is True,
        "evaluation_seeds": report.get("evaluation_seeds")
        == siim_training["evaluation_seeds"],
        "fold_count": report.get("fold_count") == siim_training["folds"],
        "profile_order": report.get("profile_order") == profiles,
        "selected_profile": report.get("selected_profile") in profiles,
        "adapter_hash": report.get("adapter_source_sha256")
        == siim_plan["implementation"]["adapter"]["sha256"],
        "wave2_hash": report.get("wave2_source_sha256")
        == siim_plan["implementation"]["wave2"]["sha256"],
        "no_private_labels": report.get("private_labels_used_for_training") is False,
        "no_official_grader": report.get("official_grader_executed") is False,
        "no_kaggle_submission": report.get("kaggle_submission_executed") is False,
        "no_official_score_claim": report.get("official_score_claimed") is False,
    }
    return {
        "ready": all(checks.values()),
        "status": "validated_complete" if all(checks.values()) else "invalid_contract",
        "report": str(path),
        "report_sha256": sha256_file(path),
        "checks": checks,
        "selected_profile": report.get("selected_profile"),
    }


def prerequisite_snapshot(
    plan: dict[str, Any],
    *,
    staging_report_path: Path,
    siim_report_path: Path | None = None,
    siim_final_watcher_path: Path | None = None,
) -> dict[str, Any]:
    staging = staging_snapshot(plan, staging_report_path)
    siim = siim_report_snapshot(plan, siim_report_path)
    siim_final = siim_final_snapshot(plan, siim_final_watcher_path)
    return {
        "staging": staging,
        "siim": siim,
        "siim_final": siim_final,
        "ready": staging["ready"] and siim["ready"] and siim_final["ready"],
    }


def siim_final_snapshot(
    plan: dict[str, Any], watcher_path: Path | None = None
) -> dict[str, Any]:
    dependency = plan["serial_dependency"]["siim_final"]
    path = Path(watcher_path or dependency["watcher_status"]).resolve()
    if not path.is_file():
        return {"ready": False, "status": "missing", "watcher": str(path)}
    try:
        watcher = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "ready": False,
            "status": "invalid_json",
            "watcher": str(path),
            "error": str(exc),
        }
    checks = {
        "schema": watcher.get("schema")
        == "evomind.siim.final_candidate_completion_watcher.v1",
        "run_id": watcher.get("run_id") == dependency["run_id"],
        "terminal_status": watcher.get("status") in set(dependency["terminal_statuses"]),
        "plan_hash": watcher.get("plan_sha256") == dependency["plan"]["sha256"],
        "no_process_signals": watcher.get("process_signals_sent") == 0,
        "no_private_labels": watcher.get("private_labels_used") is False,
        "no_official_grader": watcher.get("official_grader_executed") is False,
        "no_kaggle_submission": watcher.get("kaggle_submission_executed") is False,
    }
    return {
        "ready": all(checks.values()),
        "status": watcher.get("status", "missing"),
        "watcher": str(path),
        "watcher_sha256": sha256_file(path),
        "checks": checks,
        "candidate_ready": watcher.get("candidate_ready"),
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
    objective = plan["objective"]
    return [
        training["python"],
        training["script"],
        "--data-root",
        training["data_root"],
        "--output-root",
        training["output_root"],
        "--run-id",
        training["run_id"],
        "--seeds",
        ",".join(str(value) for value in training["seeds"]),
        "--folds",
        str(training["folds"]),
        "--backbones",
        ",".join(training["backbones"]),
        "--image-size",
        str(training["image_size"]),
        "--batch-size",
        str(training["batch_size"]),
        "--workers",
        str(training["workers"]),
        "--manifest-workers",
        str(training["manifest_workers"]),
        "--tta",
        str(training["tta"]),
        "--numeric-c",
        str(training["numeric_c"]),
        "--image-c",
        str(training["image_c"]),
        "--multimodal-c",
        str(training["multimodal_c"]),
        "--promotion-mean-log-loss",
        str(objective["promotion_mean_log_loss"]),
        "--promotion-max-seed-log-loss",
        str(objective["promotion_max_seed_log_loss"]),
        "--torch-home",
        training["torch_home"],
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--staging-report", type=Path, default=DEFAULT_STAGING_REPORT)
    parser.add_argument("--siim-report", type=Path)
    parser.add_argument("--siim-final-watcher", type=Path)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--deadline-hours", type=float, default=96.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if compute_policy.write_queue_superseded_if_hpc_only(
        project_root=PROJECT_ROOT,
        status_path=args.status,
        schema="evomind.leaf.training_queue.v1",
        queue_name="queue_leaf_after_siim",
        plan_path=args.plan,
    ):
        return 0
    if args.poll_seconds < 10 or args.deadline_hours <= 0:
        raise ValueError("Leaf queue timing contract is invalid")
    plan = validate_frozen_plan(args.plan)
    deadline = datetime.now().astimezone() + timedelta(hours=args.deadline_hours)
    idle_policy = plan["_idle_policy"]
    required_idle = int(idle_policy["requirements"]["consecutive_checks"])
    minimum_interval = int(
        idle_policy["requirements"]["minimum_check_interval_seconds"]
    )
    if args.poll_seconds < minimum_interval:
        raise ValueError("Leaf polling is faster than the calibrated idle interval")
    idle_checks = 0
    while datetime.now().astimezone() < deadline:
        prerequisites = prerequisite_snapshot(
            plan,
            staging_report_path=args.staging_report,
            siim_report_path=args.siim_report,
            siim_final_watcher_path=args.siim_final_watcher,
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
                "schema": "evomind.leaf.training_queue.v1",
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
                "official_grader_executed": False,
                "kaggle_submission_executed": False,
            },
        )
        if prerequisites["ready"] and idle_checks >= required_idle:
            launch_plan = validate_frozen_plan(args.plan)
            if launch_plan["_plan_sha256"] != plan["_plan_sha256"]:
                raise RuntimeError("Leaf frozen plan changed while queued")
            launch_prerequisites = prerequisite_snapshot(
                launch_plan,
                staging_report_path=args.staging_report,
                siim_report_path=args.siim_report,
                siim_final_watcher_path=args.siim_final_watcher,
            )
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

            output_root = Path(launch_plan["training"]["output_root"])
            run_dir = output_root / "runs" / launch_plan["training"]["run_id"]
            if run_dir.exists():
                write_json_atomic(
                    args.status,
                    {
                        "schema": "evomind.leaf.training_queue.v1",
                        "created_at": now_iso(),
                        "status": "target_run_already_exists",
                        "queue_pid": os.getpid(),
                        "run_dir": str(run_dir),
                        "plan_path": launch_plan["_plan_path"],
                        "plan_sha256": launch_plan["_plan_sha256"],
                        "process_signals_sent": 0,
                        "official_grader_executed": False,
                        "kaggle_submission_executed": False,
                    },
                )
                return 5

            if compute_policy.write_queue_superseded_if_hpc_only(
                project_root=PROJECT_ROOT,
                status_path=args.status,
                schema="evomind.leaf.training_queue.v1",
                queue_name="queue_leaf_after_siim",
                plan_path=args.plan,
            ):
                return 0
            command = build_training_command(launch_plan)
            output_root.mkdir(parents=True, exist_ok=True)
            run_id = launch_plan["training"]["run_id"]
            stdout_path = output_root / f"{run_id}.stdout.log"
            stderr_path = output_root / f"{run_id}.stderr.log"
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
                    "schema": "evomind.leaf.training_queue.v1",
                    "created_at": now_iso(),
                    "status": (
                        "training_launched" if launched else "training_launch_failed"
                    ),
                    "queue_pid": os.getpid(),
                    "training_pid": child.pid,
                    "training_return_code": return_code,
                    "command": command,
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                    "plan_path": launch_plan["_plan_path"],
                    "plan_sha256": launch_plan["_plan_sha256"],
                    "validated_artifacts": launch_plan["_validated_artifacts"],
                    "requested_model": launch_plan["planner"]["requested_model"],
                    "served_model": launch_plan["planner"]["served_model"],
                    "prerequisites": launch_prerequisites,
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
            "schema": "evomind.leaf.training_queue.v1",
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
