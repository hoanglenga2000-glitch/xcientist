#!/usr/bin/env python3
"""Run Jigsaw confirmation after May 2022 verification, ahead of SIIM/Leaf.

The original after-Leaf queue remains untouched.  This queue reuses the exact
same frozen seed plans, run IDs, token caches, OOF verifier, and result path.
It launches only while the original queue is still waiting for Leaf, so both
queues cannot intentionally enter GPU-idle acquisition at the same time.
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

from scripts import queue_jigsaw_confirmation_after_leaf as base  # noqa: E402

DEFAULT_CONTROL_PLAN = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "jigsaw_early_confirmation_after_may_control_frozen_plan.json"
)
DEFAULT_STATUS = (
    PROJECT_ROOT / "workspace" / "local_gpu" / "jigsaw_confirmation_early_queue.json"
)
CONTROL_SCHEMA = "evomind.jigsaw.early_confirmation_after_may_control.v1"
ORIGINAL_SAFE_STATUS = "waiting_for_leaf_verification"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_control_plan(path: Path) -> dict[str, Any]:
    path = path.resolve()
    control = read_json(path)
    if control.get("schema") != CONTROL_SCHEMA:
        raise RuntimeError("Unexpected Jigsaw early-confirmation control schema")
    if control.get("status") != "frozen_waiting_may_verification":
        raise RuntimeError("Jigsaw early-confirmation control is not frozen")
    source = control.get("source") or {}
    source_path = Path(source.get("path", "")).resolve()
    if source_path != Path(__file__).resolve() or sha256_file(source_path) != source.get(
        "sha256"
    ):
        raise RuntimeError("Jigsaw early-confirmation queue source drifted")
    base_binding = control.get("base_confirmation_plan") or {}
    base_path = Path(base_binding.get("path", "")).resolve()
    if not base_path.is_file() or sha256_file(base_path) != base_binding.get("sha256"):
        raise RuntimeError("Jigsaw base confirmation plan drifted")
    dependency = control.get("may2022_dependency") or {}
    may_plan = dependency.get("plan") or {}
    may_plan_path = Path(may_plan.get("path", "")).resolve()
    if not may_plan_path.is_file() or sha256_file(may_plan_path) != may_plan.get(
        "sha256"
    ):
        raise RuntimeError("May 2022 dependency plan drifted")
    if control.get("process_signals_allowed") is not False:
        raise RuntimeError("Jigsaw early-confirmation control permits process signals")
    if control.get("official_grader_executed") is not False:
        raise RuntimeError("Jigsaw early-confirmation control permits official grading")
    if control.get("kaggle_submission_executed") is not False:
        raise RuntimeError("Jigsaw early-confirmation control permits Kaggle submission")
    control["_path"] = str(path)
    control["_sha256"] = sha256_file(path)
    control["_base"] = base.validate_frozen_plan(base_path)
    return control


def may_verification_snapshot(control: dict[str, Any]) -> dict[str, Any]:
    dependency = control["may2022_dependency"]
    path = Path(dependency["watcher_path"]).resolve()
    payload = read_json(path) if path.is_file() else {}
    checks = {
        "present": path.is_file(),
        "schema": payload.get("schema") == dependency["watcher_schema"],
        "profile": payload.get("profile") == "may2022",
        "run_id": payload.get("run_id") == dependency["run_id"],
        "plan_hash": payload.get("plan_sha256") == dependency["plan"]["sha256"],
        "status": payload.get("status") == "verification_passed",
        "signals": payload.get("process_signals_sent") == 0,
        "private": payload.get("private_labels_used") is False,
        "grader": payload.get("official_grader_executed") is False,
        "kaggle": payload.get("kaggle_submission_executed") is False,
    }
    return {
        "ready": all(checks.values()),
        "path": str(path),
        "status": payload.get("status"),
        "candidate_ready": payload.get("candidate_ready"),
        "checks": checks,
    }


def original_queue_snapshot(control: dict[str, Any]) -> dict[str, Any]:
    path = Path(control["original_after_leaf_queue_status"]).resolve()
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


def queue_status(
    control: dict[str, Any], status: str, **fields: Any
) -> dict[str, Any]:
    return {
        "schema": "evomind.jigsaw.early_confirmation_queue.v1",
        "created_at": now_iso(),
        "status": status,
        "control_plan_path": control["_path"],
        "control_plan_sha256": control["_sha256"],
        "base_plan_path": control["_base"]["_path"],
        "base_plan_sha256": control["_base"]["_sha256"],
        **fields,
        **base.invariant_fields(),
    }


def control_unchanged(control: dict[str, Any]) -> None:
    if sha256_file(Path(control["_path"])) != control["_sha256"]:
        raise RuntimeError("Jigsaw early-confirmation control changed while queue ran")
    base_plan = control["_base"]
    if sha256_file(Path(base_plan["_path"])) != base_plan["_sha256"]:
        raise RuntimeError("Jigsaw base confirmation plan changed while queue ran")


def acquire_idle_gpu(
    control: dict[str, Any],
    status_path: Path,
    deadline: datetime,
    dependency: dict[str, Any],
    completed_seeds: list[int],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], int, dict[str, Any]] | None:
    plan = control["_base"]
    requirements = plan["_idle_policy"]["requirements"]
    consecutive = 0
    while datetime.now().astimezone() < deadline:
        control_unchanged(control)
        original = original_queue_snapshot(control)
        if not original["safe_for_early_queue"]:
            base.write_json_atomic(
                status_path,
                queue_status(
                    control,
                    "superseded_by_original_after_leaf_queue",
                    original_after_leaf_queue=original,
                    completed_seeds=completed_seeds,
                    active_seed=seed,
                ),
            )
            return None
        gpu = base.calibrated_idle.query_gpu()
        evaluation = base.calibrated_idle.evaluate_idle(plan["_idle_policy"], gpu)
        consecutive = consecutive + 1 if evaluation["idle"] else 0
        base.write_json_atomic(
            status_path,
            queue_status(
                control,
                f"waiting_for_stable_gpu_idle_seed_{seed}",
                dependency=dependency,
                original_after_leaf_queue=original,
                confirmation_seeds=[40, 41],
                completed_seeds=completed_seeds,
                active_seed=seed,
                gpu=gpu,
                idle_gate_evaluation=evaluation,
                consecutive_idle_checks=consecutive,
                required_idle_checks=requirements["consecutive_checks"],
                gpu_idle_gate=base.calibrated_idle.policy_record(plan["_idle_policy"]),
            ),
        )
        if consecutive >= int(requirements["consecutive_checks"]):
            # Requery immediately before launch so the last decision is fresh.
            original = original_queue_snapshot(control)
            final_gpu = base.calibrated_idle.query_gpu()
            final_evaluation = base.calibrated_idle.evaluate_idle(
                plan["_idle_policy"], final_gpu
            )
            if original["safe_for_early_queue"] and final_evaluation["idle"]:
                return final_gpu, final_evaluation, consecutive, original
            consecutive = 0
        time.sleep(int(requirements["minimum_check_interval_seconds"]))
    raise TimeoutError(f"Timed out waiting for idle GPU before seed {seed}")


def run_queue(
    control: dict[str, Any], status_path: Path, poll_seconds: int, deadline: datetime
) -> int:
    plan = control["_base"]
    dependency: dict[str, Any] = {}
    original: dict[str, Any] = {}
    while datetime.now().astimezone() < deadline:
        control_unchanged(control)
        dependency = may_verification_snapshot(control)
        original = original_queue_snapshot(control)
        base.write_json_atomic(
            status_path,
            queue_status(
                control,
                "waiting_for_may2022_verification",
                pid=os.getpid(),
                deadline=deadline.isoformat(),
                dependency=dependency,
                original_after_leaf_queue=original,
                confirmation_seeds=[40, 41],
                completed_seeds=[],
                active_seed=None,
                consecutive_idle_checks=0,
                required_idle_checks=plan["_idle_policy"]["requirements"][
                    "consecutive_checks"
                ],
                gpu_idle_gate=base.calibrated_idle.policy_record(plan["_idle_policy"]),
            ),
        )
        if not original["safe_for_early_queue"]:
            base.write_json_atomic(
                status_path,
                queue_status(
                    control,
                    "superseded_by_original_after_leaf_queue",
                    original_after_leaf_queue=original,
                ),
            )
            return 6
        if dependency["ready"]:
            break
        time.sleep(poll_seconds)
    else:
        base.write_json_atomic(
            status_path,
            queue_status(
                control,
                "timeout_waiting_for_may2022_verification",
                dependency=dependency,
                original_after_leaf_queue=original,
            ),
        )
        return 4

    reports: list[dict[str, Any]] = []
    completed_seeds: list[int] = []
    for seed_plan in plan["_seed_plans"]:
        seed = int(seed_plan["_model_seed"])
        existing = base.load_or_verify_existing(plan, seed_plan)
        if existing is not None:
            reports.append(existing)
            completed_seeds.append(seed)
            continue

        while True:
            acquisition = acquire_idle_gpu(
                control, status_path, deadline, dependency, completed_seeds, seed
            )
            if acquisition is None:
                return 6
            _gpu, _evaluation, consecutive, _original = acquisition
            # Close artifact and GPU-state races immediately before launch.
            existing = base.load_or_verify_existing(plan, seed_plan)
            if existing is not None:
                reports.append(existing)
                completed_seeds.append(seed)
                break
            original = original_queue_snapshot(control)
            final_gpu = base.calibrated_idle.query_gpu()
            final_evaluation = base.calibrated_idle.evaluate_idle(
                plan["_idle_policy"], final_gpu
            )
            if original["safe_for_early_queue"] and final_evaluation["idle"]:
                break
        if existing is not None:
            continue

        paths = base.seed_run_paths(plan, seed_plan)
        paths["run_dir"].mkdir(parents=True, exist_ok=True)
        command = base.build_training_command(plan, seed_plan)
        environment = os.environ.copy()
        environment["HF_HOME"] = str(Path(plan["execution"]["hf_cache"]).parent)
        environment["HF_HUB_CACHE"] = plan["execution"]["hf_cache"]
        with paths["train_stdout"].open("w", encoding="utf-8") as stdout_handle, paths[
            "train_stderr"
        ].open("w", encoding="utf-8") as stderr_handle:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
            base.write_json_atomic(
                paths["launch"],
                {
                    "schema": "evomind.jigsaw.confirmation_launch.v1",
                    "created_at": now_iso(),
                    "status": "launched_by_may2022_early_queue",
                    "model_seed": seed,
                    "fold_seed": 42,
                    "pid": process.pid,
                    "command": command,
                    "plan_path": seed_plan["_path"],
                    "plan_sha256": seed_plan["_sha256"],
                    "gpu_preflight": final_gpu,
                    "idle_gate_evaluation": final_evaluation,
                    "consecutive_idle_checks": consecutive,
                    **base.invariant_fields(),
                },
            )
            base.write_json_atomic(
                status_path,
                queue_status(
                    control,
                    f"training_seed_{seed}",
                    training_pid=process.pid,
                    active_seed=seed,
                    completed_seeds=completed_seeds,
                    stdout=str(paths["train_stdout"]),
                    stderr=str(paths["train_stderr"]),
                ),
            )
            return_code = process.wait()
        if return_code != 0:
            base.write_json_atomic(
                status_path,
                queue_status(
                    control,
                    "training_failed",
                    model_seed=seed,
                    training_exit_code=return_code,
                    completed_seeds=completed_seeds,
                ),
            )
            return 7
        report = base.run_verifier(plan, seed_plan)
        reports.append(report)
        completed_seeds.append(seed)

    result = base.aggregate_confirmation(plan, reports)
    result_path = Path(plan["output"]["result"]).resolve()
    base.write_json_atomic(result_path, result)
    base.write_json_atomic(
        status_path,
        queue_status(
            control,
            result["status"],
            completed_seeds=completed_seeds,
            result_path=str(result_path),
            result_sha256=sha256_file(result_path),
            candidate_ready_for_human_gate=result["candidate_ready_for_human_gate"],
        ),
    )
    return 0 if result["candidate_ready_for_human_gate"] else 3


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
        raise ValueError("Jigsaw early-confirmation queue timing contract is invalid")
    control = validate_control_plan(args.control_plan)
    deadline = datetime.now().astimezone() + timedelta(hours=args.deadline_hours)
    return run_queue(control, args.status.resolve(), args.poll_seconds, deadline)


if __name__ == "__main__":
    raise SystemExit(main())
