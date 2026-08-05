#!/usr/bin/env python3
"""Wait for the frozen SIIM candidate run and execute its independent verifier.

This watcher is deliberately CPU-only.  It never sends process signals and never
invokes the MLE-Bench private grader or Kaggle submission code.
"""
from __future__ import annotations

import argparse
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
    / "siim_final_candidate_frozen_plan.json"
)
DEFAULT_QUEUE_STATUS = (
    PROJECT_ROOT
    / "workspace"
    / "local_gpu"
    / "siim_final_candidate_queue_calibrated.json"
)
DEFAULT_STATUS = (
    PROJECT_ROOT
    / "workspace"
    / "local_gpu"
    / "siim_final_candidate_completion_watcher.json"
)

EXPECTED_PLAN_SCHEMA = "evomind.siim.final_candidate_frozen_plan.v1"
EXPECTED_REPORT_SCHEMA = "evomind.siim.final_candidate_independent_verification.v1"
TERMINAL_RESULT_STATUSES = {
    "promotion_gate_passed_confirmation_pending",
    "promotion_gate_failed",
}
TERMINAL_SUMMARY_STATUSES = {"candidate_complete", "partial_failure"}


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
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def invariant_fields() -> dict[str, Any]:
    return {
        "process_signals_sent": 0,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }


def validate_plan(path: Path) -> dict[str, Any]:
    plan_path = path.resolve()
    plan = read_json(plan_path)
    if plan.get("schema") != EXPECTED_PLAN_SCHEMA:
        raise RuntimeError("Unexpected SIIM final candidate plan schema")
    if plan.get("status") != "frozen_waiting_prerequisites":
        raise RuntimeError("SIIM final candidate plan is not frozen")
    if plan.get("competition_id") != "siim-isic-melanoma-classification":
        raise RuntimeError("SIIM final candidate plan targets another competition")

    training = plan.get("training") or {}
    launch = plan.get("launch_contract") or {}
    if training.get("candidate_only") is not True:
        raise RuntimeError("Candidate-only execution is not frozen on")
    if training.get("private_labels_used") is not False:
        raise RuntimeError("Private-label use is not frozen off")
    if training.get("official_grader_executed") is not False:
        raise RuntimeError("Official grader execution is not frozen off")
    if training.get("kaggle_submission_executed") is not False:
        raise RuntimeError("Kaggle submission is not frozen off")
    if launch.get("automatic_official_grader") is not False:
        raise RuntimeError("Automatic official grading is not frozen off")
    if launch.get("automatic_kaggle_submission") is not False:
        raise RuntimeError("Automatic Kaggle submission is not frozen off")

    verifier = Path(str(plan["verification_contract"]["script"])).resolve()
    verifier_record = plan["implementation"]["verifier"]
    if verifier != Path(str(verifier_record["path"])).resolve():
        raise RuntimeError("Verifier path differs from the frozen implementation")
    if not verifier.is_file():
        raise FileNotFoundError(f"Frozen SIIM verifier is missing: {verifier}")
    verifier_sha256 = sha256_file(verifier)
    if verifier_sha256 != str(verifier_record["sha256"]).lower():
        raise RuntimeError("Frozen SIIM verifier hash changed")

    python_path = Path(str(training["python"])).resolve()
    if not python_path.is_file():
        raise FileNotFoundError(f"Frozen SIIM Python runtime is missing: {python_path}")

    output_root = Path(str(training["output_root"])).resolve()
    run_dir = output_root / str(training["run_id"])
    plan["_plan_path"] = str(plan_path)
    plan["_plan_sha256"] = sha256_file(plan_path)
    plan["_verifier_path"] = str(verifier)
    plan["_verifier_sha256"] = verifier_sha256
    plan["_python_path"] = str(python_path)
    plan["_run_dir"] = str(run_dir)
    return plan


def file_signature(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    stat = path.stat()
    return {
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def completion_snapshot(plan: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    competition_id = str(plan["competition_id"])
    summary_path = run_dir / "summary.json"
    result_path = run_dir / competition_id / "result.json"
    summary: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    errors: list[str] = []
    if summary_path.is_file():
        try:
            summary = read_json(summary_path)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            errors.append(f"summary:{type(exc).__name__}:{exc}")
    if result_path.is_file():
        try:
            result = read_json(result_path)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            errors.append(f"result:{type(exc).__name__}:{exc}")

    summary_status = summary.get("status") if summary else None
    result_status = result.get("status") if result else None
    terminal_pair = (
        summary_status in TERMINAL_SUMMARY_STATUSES
        and result_status in TERMINAL_RESULT_STATUSES
        and summary is not None
        and result is not None
        and summary.get("run_id") == plan["training"]["run_id"]
        and result.get("competition_id") == competition_id
        and summary.get("competition_count") == 1
        and result.get("candidate_only") is True
        and result.get("official_grader_executed") is False
        and result.get("official_grader_withheld") is True
        and result.get("kaggle_public_score") is None
        and result.get("kaggle_private_score") is None
    )
    return {
        "ready": bool(terminal_pair and not errors),
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "result_path": str(result_path),
        "summary_present": summary_path.is_file(),
        "result_present": result_path.is_file(),
        "summary_status": summary_status,
        "result_status": result_status,
        "summary_signature": file_signature(summary_path),
        "result_signature": file_signature(result_path),
        "errors": errors,
    }


def stable_key(snapshot: dict[str, Any]) -> tuple[Any, ...] | None:
    if not snapshot.get("ready"):
        return None
    summary = snapshot.get("summary_signature") or {}
    result = snapshot.get("result_signature") or {}
    return (
        snapshot.get("summary_status"),
        snapshot.get("result_status"),
        summary.get("bytes"),
        summary.get("mtime_ns"),
        summary.get("sha256"),
        result.get("bytes"),
        result.get("mtime_ns"),
        result.get("sha256"),
    )


def build_verifier_command(
    plan: dict[str, Any],
    *,
    queue_status: Path,
    output: Path,
) -> list[str]:
    return [
        plan["_python_path"],
        plan["_verifier_path"],
        "--plan",
        plan["_plan_path"],
        "--run-dir",
        plan["_run_dir"],
        "--queue-status",
        str(queue_status.resolve()),
        "--output",
        str(output.resolve()),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--queue-status", type=Path, default=DEFAULT_QUEUE_STATUS)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--stable-checks", type=int, default=2)
    parser.add_argument("--deadline-hours", type=float, default=240.0)
    parser.add_argument("--max-verifier-attempts", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.poll_seconds < 10:
        raise ValueError("Completion watcher poll interval must be at least 10 seconds")
    if args.stable_checks < 2:
        raise ValueError("At least two stable terminal snapshots are required")
    if args.deadline_hours <= 0 or args.max_verifier_attempts < 1:
        raise ValueError("Completion watcher deadline or retry contract is invalid")

    plan = validate_plan(args.plan)
    plan_sha256 = plan["_plan_sha256"]
    verifier_sha256 = plan["_verifier_sha256"]
    run_dir = Path(plan["_run_dir"])
    independent_report = run_dir / "independent_verification.json"
    stdout_path = run_dir / "independent_verification_watcher.stdout.log"
    stderr_path = run_dir / "independent_verification_watcher.stderr.log"
    deadline = datetime.now().astimezone() + timedelta(hours=args.deadline_hours)
    stable_observations = 0
    previous_key: tuple[Any, ...] | None = None
    verifier_attempts = 0

    while datetime.now().astimezone() < deadline:
        if sha256_file(Path(plan["_plan_path"])) != plan_sha256:
            raise RuntimeError("Frozen SIIM final plan changed while completion watcher ran")
        if sha256_file(Path(plan["_verifier_path"])) != verifier_sha256:
            raise RuntimeError("Frozen SIIM verifier changed while completion watcher ran")

        snapshot = completion_snapshot(plan, run_dir)
        current_key = stable_key(snapshot)
        stable_observations = (
            stable_observations + 1
            if current_key is not None and current_key == previous_key
            else 1
            if current_key is not None
            else 0
        )
        previous_key = current_key

        write_json_atomic(
            args.status.resolve(),
            {
                "schema": "evomind.siim.final_candidate_completion_watcher.v1",
                "created_at": now_iso(),
                "status": (
                    "waiting_for_stable_terminal_artifacts"
                    if snapshot["ready"]
                    else "waiting_for_training_completion"
                ),
                "watcher_pid": os.getpid(),
                "deadline": deadline.isoformat(),
                "run_id": plan["training"]["run_id"],
                "plan_path": plan["_plan_path"],
                "plan_sha256": plan_sha256,
                "verifier_path": plan["_verifier_path"],
                "verifier_sha256": verifier_sha256,
                "completion": snapshot,
                "stable_observations": stable_observations,
                "required_stable_observations": args.stable_checks,
                "verifier_attempts": verifier_attempts,
                **invariant_fields(),
            },
        )

        if stable_observations < args.stable_checks:
            time.sleep(args.poll_seconds)
            continue

        command = build_verifier_command(
            plan,
            queue_status=args.queue_status,
            output=independent_report,
        )
        verifier_attempts += 1
        environment = os.environ.copy()
        python_path = os.pathsep.join([str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)])
        if environment.get("PYTHONPATH"):
            python_path += os.pathsep + environment["PYTHONPATH"]
        environment["PYTHONPATH"] = python_path
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")

        report: dict[str, Any] | None = None
        report_error: str | None = None
        try:
            report = read_json(independent_report)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            report_error = f"{type(exc).__name__}:{exc}"

        report_valid = bool(
            report
            and report.get("schema") == EXPECTED_REPORT_SCHEMA
            and report.get("process_signals_sent") == 0
            and report.get("private_labels_used") is False
            and report.get("official_grader_executed") is False
            and report.get("kaggle_submission_executed") is False
        )
        if not report_valid and verifier_attempts < args.max_verifier_attempts:
            write_json_atomic(
                args.status.resolve(),
                {
                    "schema": "evomind.siim.final_candidate_completion_watcher.v1",
                    "created_at": now_iso(),
                    "status": "verification_retry_pending",
                    "watcher_pid": os.getpid(),
                    "run_id": plan["training"]["run_id"],
                    "verifier_attempts": verifier_attempts,
                    "verifier_exit_code": completed.returncode,
                    "report_error": report_error,
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                    **invariant_fields(),
                },
            )
            stable_observations = 0
            previous_key = None
            time.sleep(args.poll_seconds)
            continue

        terminal_status = (
            str(report.get("status"))
            if report_valid
            else "verification_invocation_failed"
        )
        terminal = {
            "schema": "evomind.siim.final_candidate_completion_watcher.v1",
            "created_at": now_iso(),
            "status": terminal_status,
            "watcher_pid": os.getpid(),
            "run_id": plan["training"]["run_id"],
            "plan_path": plan["_plan_path"],
            "plan_sha256": plan_sha256,
            "verifier_path": plan["_verifier_path"],
            "verifier_sha256": verifier_sha256,
            "verifier_command": command,
            "verifier_attempts": verifier_attempts,
            "verifier_exit_code": completed.returncode,
            "independent_report_path": str(independent_report),
            "independent_report_sha256": (
                sha256_file(independent_report) if independent_report.is_file() else None
            ),
            "independent_report_passed": report.get("passed") if report_valid else False,
            "candidate_ready": report.get("candidate_ready") if report_valid else False,
            "report_error": report_error,
            "summary_path": snapshot["summary_path"],
            "summary_sha256": (snapshot.get("summary_signature") or {}).get("sha256"),
            "result_path": snapshot["result_path"],
            "result_sha256": (snapshot.get("result_signature") or {}).get("sha256"),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            **invariant_fields(),
        }
        write_json_atomic(args.status.resolve(), terminal)
        return 0 if report_valid and report.get("passed") is True else 3

    write_json_atomic(
        args.status.resolve(),
        {
            "schema": "evomind.siim.final_candidate_completion_watcher.v1",
            "created_at": now_iso(),
            "status": "timeout_waiting_for_training_completion",
            "watcher_pid": os.getpid(),
            "deadline": deadline.isoformat(),
            "run_id": plan["training"]["run_id"],
            "plan_path": plan["_plan_path"],
            "plan_sha256": plan_sha256,
            **invariant_fields(),
        },
    )
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
