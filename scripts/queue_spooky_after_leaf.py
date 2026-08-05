#!/usr/bin/env python3
"""Launch the frozen Spooky seed-42 run after Leaf and stable GPU-idle gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "spooky_transformer_byte_s42_frozen_plan_v2_20260728.json"
)
DEFAULT_DATA_REPORT = (
    PROJECT_ROOT
    / "workspace"
    / "local_gpu"
    / "mlebench_official_data"
    / "spooky-author-identification"
    / "public_staging_report.json"
)
DEFAULT_STATUS = PROJECT_ROOT / "workspace" / "local_gpu" / "spooky_training_queue.json"


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
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_frozen_plan(path: Path) -> dict[str, Any]:
    plan_path = Path(path).resolve()
    plan = read_json(plan_path)
    if plan.get("schema") != "evomind.mlebench_lite.spooky_transformer_byte_frozen_plan.v1":
        raise RuntimeError("Unexpected Spooky frozen-plan schema")
    if plan.get("status") != "frozen_before_training":
        raise RuntimeError("Spooky plan is not frozen")
    if plan.get("competition_id") != "spooky-author-identification":
        raise RuntimeError("Unexpected Spooky competition")
    driver = plan.get("driver") or {}
    if (
        driver.get("requested_model") != "gpt-5.6-sol"
        or driver.get("served_model") != "gpt-5.6-sol"
    ):
        raise RuntimeError("Spooky plan lacks verified gpt-5.6-sol provenance")
    if plan.get("public_data_only") is not True or plan.get("private_labels_used") is not False:
        raise RuntimeError("Spooky plan violates the public-data contract")
    execution = plan.get("execution") or {}
    if (
        execution.get("single_gpu_strict_serial") is not True
        or execution.get("process_signals_allowed") is not False
        or execution.get("automatic_kaggle_submission") is not False
        or execution.get("official_private_grader_before_gate") is not False
    ):
        raise RuntimeError("Spooky serial execution contract changed")
    for key in ("python", "script", "run_id", "public_dir", "output_root", "hf_cache"):
        if not execution.get(key):
            raise RuntimeError(f"Spooky frozen plan is missing execution.{key}")
    source = plan.get("source") or {}
    runner_path = Path(source.get("runner_path", ""))
    if not runner_path.is_file() or sha256_file(runner_path) != source.get("runner_sha256"):
        raise RuntimeError("Spooky runner changed after plan freeze")
    adapter_path = Path(source.get("frozen_adapter_path", ""))
    if not adapter_path.is_file() or sha256_file(adapter_path) != source.get(
        "frozen_adapter_sha256"
    ):
        raise RuntimeError("Spooky referenced frozen adapter changed")
    audit_path = Path(driver.get("audit_path", ""))
    if not audit_path.is_file() or sha256_file(audit_path) != driver.get("audit_sha256"):
        raise RuntimeError("Spooky gpt-5.6-sol audit changed after plan freeze")
    plan["_path"] = str(plan_path)
    plan["_sha256"] = sha256_file(plan_path)
    return plan


def data_snapshot(plan: dict[str, Any], report_path: Path) -> dict[str, Any]:
    path = Path(report_path).resolve()
    if not path.is_file():
        return {"ready": False, "status": "missing", "report": str(path)}
    try:
        report = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "ready": False,
            "status": "invalid_json",
            "report": str(path),
            "error": str(exc),
        }
    public_dir = Path(plan["execution"]["public_dir"])
    inputs = plan["inputs"]
    expected = {
        "train.csv": inputs["public_train_sha256"],
        "test.csv": inputs["public_test_sha256"],
        "sample_submission.csv": inputs["public_sample_submission_sha256"],
    }
    hash_checks = {
        name: (public_dir / name).is_file()
        and sha256_file(public_dir / name) == expected_hash
        for name, expected_hash in expected.items()
    }
    checks = {
        "schema": report.get("schema") == "evomind.spooky.public_staging.v1",
        "competition": report.get("competition_id") == "spooky-author-identification",
        "status": report.get("status") == "verified_complete",
        "train_rows": report.get("rows", {}).get("train") == inputs["train_rows"],
        "test_rows": report.get("rows", {}).get("test") == inputs["test_rows"],
        "hashes": all(hash_checks.values()),
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
        "file_hash_checks": hash_checks,
    }


def leaf_snapshot(plan: dict[str, Any], report_path: Path | None = None) -> dict[str, Any]:
    dependency = plan["execution"]["serial_dependency"]
    path = Path(report_path or dependency["report"]).resolve()
    if not path.is_file():
        return {"ready": False, "status": "missing", "report": str(path)}
    try:
        report = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "ready": False,
            "status": "invalid_json",
            "report": str(path),
            "error": str(exc),
        }
    checks = {
        "schema": report.get("schema") == "evomind.leaf.multibackbone_oof.v1",
        "competition": report.get("competition_id") == dependency["competition_id"],
        "run_id": report.get("run_id") == dependency["run_id"],
        "terminal": report.get("status") in dependency["terminal_statuses"],
        "full_public_train_scope": report.get("full_public_train_scope") is True,
        "no_private_labels": report.get("private_labels_used") is False,
        "no_private_tuning": report.get("private_scores_used_for_tuning") is False,
        "no_official_grader": report.get("official_grader_executed") is False,
        "no_kaggle_submission": report.get("kaggle_submission_executed") is False,
        "no_official_score_claim": report.get("official_score_claimed") is False,
    }
    return {
        "ready": all(checks.values()),
        "status": report.get("status", "missing"),
        "report": str(path),
        "report_sha256": sha256_file(path),
        "checks": checks,
    }


def prerequisite_snapshot(
    plan: dict[str, Any],
    *,
    data_report_path: Path,
    leaf_report_path: Path | None = None,
) -> dict[str, Any]:
    data = data_snapshot(plan, data_report_path)
    leaf = leaf_snapshot(plan, leaf_report_path)
    return {"data": data, "leaf": leaf, "ready": data["ready"] and leaf["ready"]}


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
        value
        for value in applications
        if "python" in Path(value["process_name"]).name.lower()
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
    execution = plan["execution"]
    training = plan["training"]
    byte = plan["byte_channel"]
    sparse = plan["sparse_channel"]
    ensemble = plan["ensemble"]
    return [
        execution["python"],
        execution["script"],
        "--public-dir",
        execution["public_dir"],
        "--output-root",
        execution["output_root"],
        "--run-id",
        execution["run_id"],
        "--frozen-plan",
        plan["_path"],
        "--hf-cache",
        execution["hf_cache"],
        "--model-id",
        plan["model"]["repo_id"],
        "--model-revision",
        plan["model"]["revision"],
        "--folds",
        ",".join(str(value) for value in range(training["folds"])),
        "--seed",
        str(training["seed"]),
        "--epochs",
        str(training["epochs_per_fold"]),
        "--max-length",
        str(training["max_length"]),
        "--train-batch-size",
        str(training["train_batch_size"]),
        "--eval-batch-size",
        str(training["eval_batch_size"]),
        "--gradient-accumulation-steps",
        str(training["gradient_accumulation_steps"]),
        "--learning-rate",
        str(training["learning_rate"]),
        "--weight-decay",
        str(training["weight_decay"]),
        "--warmup-ratio",
        str(training["warmup_ratio"]),
        "--max-grad-norm",
        str(training["max_grad_norm"]),
        "--num-workers",
        str(training["num_workers"]),
        "--byte-max-length",
        str(byte["max_length"]),
        "--byte-embedding-dim",
        str(byte["embedding_dim"]),
        "--byte-channels",
        str(byte["channels"]),
        "--byte-dropout",
        str(byte["dropout"]),
        "--byte-epochs",
        str(byte["epochs_per_fold"]),
        "--byte-batch-size",
        str(byte["train_batch_size"]),
        "--byte-eval-batch-size",
        str(byte["eval_batch_size"]),
        "--byte-learning-rate",
        str(byte["learning_rate"]),
        "--byte-weight-decay",
        str(byte["weight_decay"]),
        "--byte-label-smoothing",
        str(byte["label_smoothing"]),
        "--sparse-word-features",
        str(sparse["word_max_features"]),
        "--sparse-char-features",
        str(sparse["char_max_features"]),
        "--sparse-c",
        str(sparse["c_value"]),
        "--blend-steps",
        str(ensemble["weight_grid_steps"]),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--data-report", type=Path, default=DEFAULT_DATA_REPORT)
    parser.add_argument("--leaf-report", type=Path)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--deadline-hours", type=float, default=168.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.poll_seconds < 10 or args.deadline_hours <= 0:
        raise ValueError("Spooky queue timing contract is invalid")
    plan = validate_frozen_plan(args.plan)
    deadline = datetime.now().astimezone() + timedelta(hours=args.deadline_hours)
    required_idle = int(plan["execution"]["gpu_idle_consecutive_checks"])
    max_utilization = int(plan["execution"]["gpu_idle_max_utilization_percent"])
    idle_checks = 0
    while datetime.now().astimezone() < deadline:
        prerequisites = prerequisite_snapshot(
            plan,
            data_report_path=args.data_report,
            leaf_report_path=args.leaf_report,
        )
        gpu: dict[str, Any] | None = None
        status = "waiting_for_prerequisites"
        if prerequisites["ready"]:
            gpu = query_gpu()
            is_idle = (
                gpu["utilization_percent"] <= max_utilization
                and not gpu["python_compute_applications"]
            )
            idle_checks = idle_checks + 1 if is_idle else 0
            status = "waiting_for_stable_gpu_idle"
        write_json_atomic(
            args.status,
            {
                "schema": "evomind.spooky.training_queue.v1",
                "created_at": now_iso(),
                "status": status,
                "pid": os.getpid(),
                "deadline": deadline.isoformat(),
                "plan_path": plan["_path"],
                "plan_sha256": plan["_sha256"],
                "requested_model": plan["driver"]["requested_model"],
                "served_model": plan["driver"]["served_model"],
                "prerequisites": prerequisites,
                "gpu": gpu,
                "consecutive_idle_checks": idle_checks,
                "required_idle_checks": required_idle,
                "process_signals_sent": 0,
                "official_grader_executed": False,
                "kaggle_submission_executed": False,
            },
        )
        if prerequisites["ready"] and idle_checks >= required_idle:
            launch_plan = validate_frozen_plan(args.plan)
            if launch_plan["_sha256"] != plan["_sha256"]:
                raise RuntimeError("Spooky frozen plan changed while queued")
            final_prerequisites = prerequisite_snapshot(
                launch_plan,
                data_report_path=args.data_report,
                leaf_report_path=args.leaf_report,
            )
            if not final_prerequisites["ready"]:
                idle_checks = 0
                time.sleep(args.poll_seconds)
                continue
            run_dir = Path(launch_plan["execution"]["output_root"]) / launch_plan[
                "execution"
            ]["run_id"]
            summary_path = run_dir / "summary.json"
            if summary_path.is_file():
                summary = read_json(summary_path)
                if summary.get("stage") == "terminal" or summary.get("status") in {
                    "single_seed_gate_passed_confirmation_pending",
                    "single_seed_gate_failed",
                }:
                    write_json_atomic(
                        args.status,
                        {
                            "schema": "evomind.spooky.training_queue.v1",
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
            command = build_training_command(launch_plan)
            Path(launch_plan["execution"]["output_root"]).mkdir(parents=True, exist_ok=True)
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
            write_json_atomic(
                args.status,
                {
                    "schema": "evomind.spooky.training_queue.v1",
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
                    "prerequisites": final_prerequisites,
                    "gpu_before_launch": gpu,
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
            "schema": "evomind.spooky.training_queue.v1",
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
