"""Launch the frozen Spooky cross-run XGBoost evaluator after run completion."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from scripts.evaluate_spooky_cross_run_xgb import (
        EXPECTED_SCHEMA,
        PROJECT_ROOT,
        TERMINAL_CURRENT_STATUSES,
        read_json,
        sha256_file,
        write_json_atomic,
    )
except ImportError:  # Direct ``python scripts/...`` execution.
    from evaluate_spooky_cross_run_xgb import (  # type: ignore[no-redef]
        EXPECTED_SCHEMA,
        PROJECT_ROOT,
        TERMINAL_CURRENT_STATUSES,
        read_json,
        sha256_file,
        write_json_atomic,
    )


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def invariant_fields() -> dict[str, Any]:
    return {
        "resource": "CPU only; maximum two XGBoost worker threads",
        "process_signals_sent": 0,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "human_gate_preserved": True,
    }


def validate_frozen_plan(plan_path: Path) -> dict[str, Any]:
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    if plan.get("schema") != EXPECTED_SCHEMA:
        raise ValueError("Unexpected cross-run XGBoost plan schema")
    if plan.get("status") != "frozen_waiting_current_run":
        raise ValueError("Cross-run XGBoost plan is not frozen")
    for record in plan.get("source", {}).values():
        path = Path(record["path"]).resolve()
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Frozen source drift: {path}")
    manifest = Path(plan["runtime"]["manifest"]["path"]).resolve()
    if not manifest.is_file() or sha256_file(manifest) != plan["runtime"]["manifest"]["sha256"]:
        raise ValueError("Frozen XGBoost runtime manifest drifted")
    plan["_plan_path"] = str(plan_path)
    plan["_plan_sha256"] = sha256_file(plan_path)
    return plan


def completion_snapshot(plan: Mapping[str, Any]) -> dict[str, Any]:
    summary_path = Path(plan["current"]["summary_path"]).resolve()
    snapshot: dict[str, Any] = {
        "ready": False,
        "summary_path": str(summary_path),
        "summary_present": summary_path.is_file(),
        "summary_status": None,
        "summary_run_id": None,
        "summary_signature": None,
        "bundle_path": None,
        "bundle_present": False,
        "bundle_signature": None,
        "bundle_hash_matches": False,
        "boundary_ok": False,
        "errors": [],
    }
    if not summary_path.is_file():
        return snapshot
    try:
        summary = read_json(summary_path)
        snapshot["summary_status"] = summary.get("status")
        snapshot["summary_run_id"] = summary.get("run_id")
        snapshot["summary_signature"] = {
            "bytes": summary_path.stat().st_size,
            "mtime_ns": summary_path.stat().st_mtime_ns,
            "sha256": sha256_file(summary_path),
        }
        prediction = summary.get("prediction_bundle") or {}
        bundle_path = Path(prediction.get("path", "")).resolve()
        snapshot["bundle_path"] = str(bundle_path)
        snapshot["bundle_present"] = bundle_path.is_file()
        if bundle_path.is_file():
            snapshot["bundle_signature"] = {
                "bytes": bundle_path.stat().st_size,
                "mtime_ns": bundle_path.stat().st_mtime_ns,
                "sha256": sha256_file(bundle_path),
            }
            snapshot["bundle_hash_matches"] = (
                prediction.get("sha256") == snapshot["bundle_signature"]["sha256"]
            )
        snapshot["boundary_ok"] = bool(
            summary.get("private_labels_used") is False
            and summary.get("official_grader_executed") is False
            and summary.get("kaggle_submission_executed") is False
            and summary.get("process_signals_sent") == 0
        )
        snapshot["ready"] = bool(
            summary.get("status") in TERMINAL_CURRENT_STATUSES
            and summary.get("run_id") == plan["current"]["run_id"]
            and summary.get("plan_sha256") == plan["current"]["plan"]["sha256"]
            and snapshot["bundle_present"]
            and snapshot["bundle_hash_matches"]
            and snapshot["boundary_ok"]
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        snapshot["errors"].append(f"{type(exc).__name__}:{exc}")
    return snapshot


def stable_key(snapshot: Mapping[str, Any]) -> tuple[Any, ...] | None:
    if not snapshot.get("ready"):
        return None
    summary = snapshot["summary_signature"]
    bundle = snapshot["bundle_signature"]
    return (
        summary["bytes"],
        summary["mtime_ns"],
        summary["sha256"],
        bundle["bytes"],
        bundle["mtime_ns"],
        bundle["sha256"],
    )


def cpu_only_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
        }
    )
    return environment


def result_valid(result: Mapping[str, Any], plan: Mapping[str, Any]) -> bool:
    return bool(
        result.get("schema") == "evomind.spooky.cross_run_xgb_result.v1"
        and result.get("run_id") == plan["output"]["run_id"]
        and result.get("plan", {}).get("sha256") == plan["_plan_sha256"]
        and result.get("private_labels_used") is False
        and result.get("official_grader_executed") is False
        and result.get("kaggle_submission_executed") is False
        and result.get("process_signals_sent") == 0
        and result.get("promotion_allowed") is False
        and all(result.get("checks", {}).values())
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--stable-checks", type=int, default=2)
    parser.add_argument("--deadline-hours", type=float, default=240.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.poll_seconds < 1 or args.stable_checks < 2 or args.deadline_hours <= 0:
        raise ValueError("Watcher timing contract is invalid")
    python_path = args.python.resolve()
    if not python_path.is_file():
        raise FileNotFoundError(python_path)
    plan = validate_frozen_plan(args.plan)
    plan_path = Path(plan["_plan_path"])
    plan_sha256 = plan["_plan_sha256"]
    evaluator = Path(plan["source"]["evaluator"]["path"]).resolve()
    output_dir = Path(plan["output"]["directory"]).resolve()
    result_path = output_dir / "cross_run_xgb_result.json"
    stdout_path = output_dir / "cross_run_xgb_watcher.stdout.log"
    stderr_path = output_dir / "cross_run_xgb_watcher.stderr.log"
    deadline = datetime.now().astimezone() + timedelta(hours=args.deadline_hours)
    previous_key: tuple[Any, ...] | None = None
    stable_observations = 0

    while datetime.now().astimezone() < deadline:
        if sha256_file(plan_path) != plan_sha256:
            raise RuntimeError("Frozen cross-run XGBoost plan changed while watcher ran")
        for record in plan["source"].values():
            if sha256_file(Path(record["path"])) != record["sha256"]:
                raise RuntimeError("Frozen cross-run XGBoost source changed while watcher ran")
        snapshot = completion_snapshot(plan)
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
                "schema": "evomind.spooky.cross_run_xgb_completion_watcher.v1",
                "created_at": now_iso(),
                "status": (
                    "waiting_for_stable_current_run_artifacts"
                    if snapshot["ready"]
                    else "waiting_for_current_run_completion"
                ),
                "watcher_pid": os.getpid(),
                "deadline": deadline.isoformat(),
                "run_id": plan["output"]["run_id"],
                "current_run_id": plan["current"]["run_id"],
                "plan_path": str(plan_path),
                "plan_sha256": plan_sha256,
                "completion": snapshot,
                "stable_observations": stable_observations,
                "required_stable_observations": args.stable_checks,
                **invariant_fields(),
            },
        )
        if stable_observations < args.stable_checks:
            time.sleep(args.poll_seconds)
            continue

        command = [str(python_path), str(evaluator), "--plan", str(plan_path)]
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=cpu_only_environment(),
            capture_output=True,
            text=True,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        result: dict[str, Any] = {}
        error: str | None = None
        try:
            result = read_json(result_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            error = f"{type(exc).__name__}:{exc}"
        valid = result_valid(result, plan)
        write_json_atomic(
            args.status.resolve(),
            {
                "schema": "evomind.spooky.cross_run_xgb_completion_watcher.v1",
                "created_at": now_iso(),
                "status": result.get("status") if valid else "cross_run_xgb_evaluation_failed",
                "watcher_pid": os.getpid(),
                "run_id": plan["output"]["run_id"],
                "current_run_id": plan["current"]["run_id"],
                "plan_path": str(plan_path),
                "plan_sha256": plan_sha256,
                "evaluator_command": command,
                "evaluator_exit_code": completed.returncode,
                "result_path": str(result_path),
                "result_sha256": sha256_file(result_path) if result_path.is_file() else None,
                "result_valid": valid,
                "single_seed_passed": result.get("single_seed_passed") if valid else False,
                "confirmation_required": True,
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "result_error": error,
                **invariant_fields(),
            },
        )
        return 0 if valid and completed.returncode == 0 else 3

    write_json_atomic(
        args.status.resolve(),
        {
            "schema": "evomind.spooky.cross_run_xgb_completion_watcher.v1",
            "created_at": now_iso(),
            "status": "timeout_waiting_for_current_run_completion",
            "watcher_pid": os.getpid(),
            "deadline": deadline.isoformat(),
            "run_id": plan["output"]["run_id"],
            "current_run_id": plan["current"]["run_id"],
            "plan_path": str(plan_path),
            "plan_sha256": plan_sha256,
            **invariant_fields(),
        },
    )
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
