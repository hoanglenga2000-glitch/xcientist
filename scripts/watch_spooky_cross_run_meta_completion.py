#!/usr/bin/env python3
"""Wait for the current Spooky run and execute the frozen CPU-only meta evaluator.

The watcher observes files only.  It does not send process signals, invoke an
official grader, or submit a Kaggle candidate.  A terminal current-run summary
must be stable across multiple polls before the evaluator is launched.
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

try:
    from scripts import evaluate_spooky_cross_run_meta as evaluator
except ImportError:  # Direct ``python scripts/...`` execution.
    import evaluate_spooky_cross_run_meta as evaluator  # type: ignore[no-redef]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = evaluator.DEFAULT_PLAN
DEFAULT_STATUS = (
    PROJECT_ROOT
    / "workspace"
    / "local_gpu"
    / "spooky_cross_run_meta_completion_watcher.json"
)
DEFAULT_PYTHON = (
    PROJECT_ROOT / "workspace" / "release-venv" / "Scripts" / "python.exe"
)
EXPECTED_RESULT_SCHEMA = "evomind.spooky.cross_run_meta_result.v1"
TERMINAL_RESULT_STATUSES = {
    "single_seed_gate_failed",
    "single_seed_gate_passed_confirmation_pending",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def invariant_fields() -> dict[str, Any]:
    return {
        "resource": "CPU only",
        "process_signals_sent": 0,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "human_gate_preserved": True,
    }


def file_signature(path: Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.is_file():
        return None
    stat = path.stat()
    return {
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def validate_frozen_plan(path: Path) -> dict[str, Any]:
    plan = evaluator.validate_plan(Path(path))
    current = plan.get("current") or {}
    if set(current.get("allowed_terminal_statuses") or []) != set(
        evaluator.TERMINAL_CURRENT_STATUSES
    ):
        raise RuntimeError("Current-run terminal status contract drifted")
    resource = plan.get("resource_policy") or {}
    if resource.get("device") != "CPU only":
        raise RuntimeError("Cross-run evaluator is not frozen as CPU only")
    if resource.get("process_signals_allowed") != 0:
        raise RuntimeError("Cross-run plan permits process signals")
    return plan


def completion_snapshot(plan: dict[str, Any]) -> dict[str, Any]:
    current = plan["current"]
    summary_path = Path(current["summary_path"]).resolve()
    summary: dict[str, Any] | None = None
    errors: list[str] = []
    if summary_path.is_file():
        try:
            summary = read_json(summary_path)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            errors.append(f"summary:{type(exc).__name__}:{exc}")

    prediction = (summary or {}).get("prediction_bundle") or {}
    raw_bundle_path = prediction.get("path")
    bundle_path = Path(raw_bundle_path).resolve() if raw_bundle_path else None
    bundle_signature: dict[str, Any] | None = None
    bundle_hash_matches = False
    if bundle_path is not None and bundle_path.is_file():
        try:
            bundle_signature = file_signature(bundle_path)
            bundle_hash_matches = bool(
                bundle_signature
                and prediction.get("sha256") == bundle_signature.get("sha256")
            )
        except OSError as exc:
            errors.append(f"bundle:{type(exc).__name__}:{exc}")

    status = (summary or {}).get("status")
    boundary_ok = bool(
        summary
        and summary.get("private_labels_used") is False
        and summary.get("official_grader_executed") is False
        and summary.get("kaggle_submission_executed") is False
        and summary.get("process_signals_sent") == 0
    )
    ready = bool(
        summary
        and status in evaluator.TERMINAL_CURRENT_STATUSES
        and summary.get("run_id") == current["run_id"]
        and summary.get("plan_sha256") == current["plan"]["sha256"]
        and bundle_hash_matches
        and boundary_ok
        and not errors
    )
    return {
        "ready": ready,
        "summary_path": str(summary_path),
        "summary_present": summary_path.is_file(),
        "summary_status": status,
        "summary_run_id": (summary or {}).get("run_id"),
        "summary_signature": file_signature(summary_path),
        "bundle_path": str(bundle_path) if bundle_path is not None else None,
        "bundle_present": bool(bundle_path and bundle_path.is_file()),
        "bundle_signature": bundle_signature,
        "bundle_hash_matches": bundle_hash_matches,
        "boundary_ok": boundary_ok,
        "errors": errors,
    }


def stable_key(snapshot: dict[str, Any]) -> tuple[Any, ...] | None:
    if not snapshot.get("ready"):
        return None
    summary = snapshot.get("summary_signature") or {}
    bundle = snapshot.get("bundle_signature") or {}
    return (
        snapshot.get("summary_status"),
        summary.get("bytes"),
        summary.get("mtime_ns"),
        summary.get("sha256"),
        bundle.get("bytes"),
        bundle.get("mtime_ns"),
        bundle.get("sha256"),
    )


def build_evaluator_command(
    python_path: Path, plan_path: Path, evaluator_path: Path
) -> list[str]:
    return [
        str(Path(python_path).resolve()),
        str(Path(evaluator_path).resolve()),
        "--frozen-plan",
        str(Path(plan_path).resolve()),
    ]


def cpu_only_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment.setdefault("OMP_NUM_THREADS", "4")
    environment.setdefault("MKL_NUM_THREADS", "4")
    environment.setdefault("OPENBLAS_NUM_THREADS", "4")
    python_path = os.pathsep.join([str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)])
    if environment.get("PYTHONPATH"):
        python_path += os.pathsep + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = python_path
    return environment


def result_valid(result: dict[str, Any], plan: dict[str, Any]) -> bool:
    checks = result.get("checks") or {}
    return bool(
        result.get("schema") == EXPECTED_RESULT_SCHEMA
        and result.get("status") in TERMINAL_RESULT_STATUSES
        and result.get("run_id") == plan["output"]["run_id"]
        and result.get("plan_sha256") == plan["_plan_sha256"]
        and checks.get("truth_and_folds_identical") is True
        and checks.get("public_csv_truth_identical") is True
        and checks.get("exact_once_oof") is True
        and checks.get("finite_normalized_oof") is True
        and checks.get("finite_normalized_test") is True
        and result.get("private_labels_used") is False
        and result.get("official_grader_executed") is False
        and result.get("kaggle_submission_executed") is False
        and result.get("process_signals_sent") == 0
        and result.get("promotion_allowed") is False
        and result.get("human_gate_preserved") is True
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--stable-checks", type=int, default=2)
    parser.add_argument("--deadline-hours", type=float, default=240.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.poll_seconds < 10:
        raise ValueError("Watcher poll interval must be at least 10 seconds")
    if args.stable_checks < 2 or args.deadline_hours <= 0:
        raise ValueError("Watcher stability or deadline contract is invalid")
    python_path = args.python.resolve()
    if not python_path.is_file():
        raise FileNotFoundError(f"Evaluator Python runtime is missing: {python_path}")

    plan = validate_frozen_plan(args.plan)
    plan_path = Path(plan["_plan_path"])
    plan_sha256 = plan["_plan_sha256"]
    evaluator_path = Path(plan["source"]["evaluator"]["path"]).resolve()
    evaluator_sha256 = plan["source"]["evaluator"]["sha256"]
    meta_helper_path = Path(plan["source"]["meta_helper"]["path"]).resolve()
    meta_helper_sha256 = plan["source"]["meta_helper"]["sha256"]
    output_dir = Path(plan["output"]["directory"]).resolve()
    result_path = output_dir / "cross_run_meta_result.json"
    stdout_path = output_dir / "cross_run_meta_watcher.stdout.log"
    stderr_path = output_dir / "cross_run_meta_watcher.stderr.log"
    deadline = datetime.now().astimezone() + timedelta(hours=args.deadline_hours)
    previous_key: tuple[Any, ...] | None = None
    stable_observations = 0

    while datetime.now().astimezone() < deadline:
        if sha256_file(plan_path) != plan_sha256:
            raise RuntimeError("Frozen cross-run plan changed while watcher ran")
        if sha256_file(evaluator_path) != evaluator_sha256:
            raise RuntimeError("Frozen cross-run evaluator changed while watcher ran")
        if sha256_file(meta_helper_path) != meta_helper_sha256:
            raise RuntimeError("Frozen meta helper changed while watcher ran")

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
                "schema": "evomind.spooky.cross_run_meta_completion_watcher.v1",
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
                "evaluator_path": str(evaluator_path),
                "evaluator_sha256": evaluator_sha256,
                "meta_helper_path": str(meta_helper_path),
                "meta_helper_sha256": meta_helper_sha256,
                "completion": snapshot,
                "stable_observations": stable_observations,
                "required_stable_observations": args.stable_checks,
                **invariant_fields(),
            },
        )
        if stable_observations < args.stable_checks:
            time.sleep(args.poll_seconds)
            continue

        command = build_evaluator_command(python_path, plan_path, evaluator_path)
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
        result_error: str | None = None
        try:
            result = read_json(result_path)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            result_error = f"{type(exc).__name__}:{exc}"
        valid = result_valid(result, plan)
        write_json_atomic(
            args.status.resolve(),
            {
                "schema": "evomind.spooky.cross_run_meta_completion_watcher.v1",
                "created_at": now_iso(),
                "status": (
                    result.get("status") if valid else "cross_run_meta_evaluation_failed"
                ),
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
                "confirmation_required": result.get("confirmation_required") if valid else True,
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "result_error": result_error,
                **invariant_fields(),
            },
        )
        return 0 if valid and completed.returncode == 0 else 3

    write_json_atomic(
        args.status.resolve(),
        {
            "schema": "evomind.spooky.cross_run_meta_completion_watcher.v1",
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
