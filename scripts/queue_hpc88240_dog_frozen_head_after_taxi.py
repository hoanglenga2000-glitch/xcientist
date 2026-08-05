#!/usr/bin/env python3
"""Queue a bounded Dog Breed frozen-backbone recovery after Taxi terminates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import mlebench_remote_ops as ops  # noqa: E402
from scripts import queue_hpc88240_taxi_after_may as taxi_queue  # noqa: E402

PLAN_SCHEMA = "evomind.mlebench.dog_breed_frozen_head_after_taxi_plan.v1"
STATUS_SCHEMA = "evomind.hpc88240.dog_breed_frozen_head_queue.v1"
COMPETITION_ID = "dog-breed-identification"
DOG_PUBLIC_ROOT = (
    f"{ops.ALLOWED_GPU_REMOTE_ROOT}/mlebench_official_data/"
    f"{COMPETITION_ID}/prepared/public"
)
DOG_PUBLIC_INPUT_CONTRACT = {
    "labels.csv": {
        "bytes": 433685,
        "sha256": "79791b843071f193d207b5c289f739d74a2d67fb84965082285cf1f7ee8be664",
    },
    "sample_submission.csv": {
        "bytes": 2613433,
        "sha256": "9e02a595300e98f61fbdf6451a63774b582a16dac7e26f714f757b1ca8b62421",
    },
    "train": {"count": 9199, "bytes": 325166830},
    "test": {"count": 1023, "bytes": 36533540},
}
DEFAULT_PLAN = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "dog_breed_frozen_head_after_taxi_20260728.json"
)
DEFAULT_TAXI_STATUS = (
    PROJECT_ROOT
    / "workspace"
    / "hpc"
    / "job88240_taxi_source_audited_queue_v6_may_v8b_final"
    / "status_current.json"
)
DEFAULT_EVIDENCE_DIR = (
    PROJECT_ROOT / "workspace" / "hpc" / "job88240_dog_frozen_head_after_taxi"
)
TERMINAL_QUEUE_STATUSES = {
    "diagnostic_gate_failed",
    "full_seed_gate_failed",
    "all_full_seeds_terminal",
    "blocked_by_dependency_integrity_failure",
    "launch_claim_already_exists",
}


class DogQueueError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DogQueueError(message)


def validate_plan(path: Path = DEFAULT_PLAN) -> dict[str, Any]:
    resolved = Path(path).resolve()
    plan = json.loads(resolved.read_text(encoding="utf-8"))
    exact = {
        "schema": PLAN_SCHEMA,
        "status": "frozen_waiting_for_taxi_terminal",
        "competition_id": COMPETITION_ID,
        "visibility_mode": "PUBLIC_ONLY",
        "remote_root": ops.ALLOWED_GPU_REMOTE_ROOT,
        "automatic_official_grader": False,
        "kaggle_submission_enabled": False,
    }
    for key, expected in exact.items():
        require(plan.get(key) == expected, f"Dog plan {key} drifted")

    bundle = plan.get("bundle") or {}
    bundle_path = Path(str(bundle.get("path") or "")).resolve()
    require(bundle_path.is_file(), "Dog frozen bundle is missing")
    require(
        bundle_path.stat().st_size == int(bundle.get("bytes", -1))
        and sha256_file(bundle_path) == bundle.get("sha256"),
        "Dog frozen bundle identity drifted",
    )
    dependency = plan.get("serial_dependency") or {}
    taxi_plan_path = Path(str(dependency.get("taxi_plan_path") or "")).resolve()
    require(
        taxi_plan_path.is_file()
        and sha256_file(taxi_plan_path) == dependency.get("taxi_plan_sha256"),
        "Dog Taxi dependency plan drifted",
    )
    taxi_plan = json.loads(taxi_plan_path.read_text(encoding="utf-8"))
    require(
        list(dependency.get("taxi_run_ids") or []) == list(taxi_plan.get("run_ids") or [])
        and dependency.get("all_taxi_seeds_terminal_before_dog") is True,
        "Dog Taxi dependency run inventory drifted",
    )

    diagnostic = plan.get("diagnostic") or {}
    require(
        diagnostic.get("seed") == 46
        and diagnostic.get("backbone") == "convnext_small"
        and diagnostic.get("training_mode") == "frozen_backbone_head"
        and diagnostic.get("head_learning_rate") == 0.001
        and diagnostic.get("fold_limit") == 1
        and diagnostic.get("folds") == 5
        and diagnostic.get("epochs") == 4
        and diagnostic.get("batch_size") == 32
        and diagnostic.get("parent_fold0_epoch1_log_loss") == 0.156225621700287
        and diagnostic.get("parent_fold0_top1_accuracy") == 0.946195652173913,
        "Dog diagnostic contract drifted",
    )
    confirmation = plan.get("confirmation") or {}
    require(
        confirmation.get("seeds") == [46, 47]
        and len(confirmation.get("run_ids") or []) == 2
        and confirmation.get("epochs") == 8
        and confirmation.get("folds") == 5
        and confirmation.get("batch_size") == 32
        and confirmation.get("aggregate_oof_log_loss_maximum") == 0.04
        and confirmation.get("every_seed_oof_log_loss_maximum") == 0.04,
        "Dog confirmation contract drifted",
    )
    confirmation_gate = plan.get("confirmation_gate") or {}
    require(
        confirmation_gate.get("confirmation_seeds") == [46, 47]
        and confirmation_gate.get("every_seed_oof_log_loss_maximum") == 0.04
        and confirmation_gate.get("aggregate_oof_log_loss_maximum") == 0.04,
        "Dog confirmation gate drifted",
    )
    inputs = plan.get("public_inputs") or {}
    require(
        inputs.get("root") == DOG_PUBLIC_ROOT
        and inputs.get("train_count") == 9199
        and inputs.get("test_count") == 1023
        and inputs.get("private_paths_read") == [],
        "Dog public input contract drifted",
    )
    for name, expected in DOG_PUBLIC_INPUT_CONTRACT.items():
        observed = inputs.get(name) or {}
        require(
            observed.get("path") == f"{DOG_PUBLIC_ROOT}/{name}"
            and all(observed.get(key) == value for key, value in expected.items()),
            f"Dog public input identity drifted: {name}",
        )
    sample = plan.get("public_sample_submission") or {}
    expected_sample = DOG_PUBLIC_INPUT_CONTRACT["sample_submission.csv"]
    require(
        sample.get("remote_path") == f"{DOG_PUBLIC_ROOT}/sample_submission.csv"
        and all(sample.get(key) == value for key, value in expected_sample.items()),
        "Dog public sample-submission identity drifted",
    )
    boundaries = plan.get("boundaries") or {}
    require(
        boundaries.get("private_labels_used") is False
        and boundaries.get("official_grader_executed") is False
        and boundaries.get("kaggle_submission_executed") is False
        and boundaries.get("process_signals_sent") == 0
        and boundaries.get("human_gate_preserved") is True,
        "Dog plan boundaries drifted",
    )
    plan["_path"] = str(resolved)
    plan["_sha256"] = sha256_file(resolved)
    plan["_bundle_path"] = str(bundle_path)
    return plan


def _base(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": STATUS_SCHEMA,
        "created_at": now_iso(),
        "plan_path": plan["_path"],
        "plan_sha256": plan["_sha256"],
        "bundle_sha256": plan["bundle"]["sha256"],
        "process_signals_sent": 0,
        "other_processes_modified": False,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }


def _runner_args(plan: Mapping[str, Any], *, diagnostic: bool) -> list[str]:
    source = plan["diagnostic"] if diagnostic else plan["confirmation"]
    values = [
        "--wave2-dog-breed-backbone",
        str(plan["diagnostic"]["backbone"]),
        "--wave2-dog-breed-training-mode",
        str(plan["diagnostic"]["training_mode"]),
        "--wave2-dog-breed-head-learning-rate",
        str(plan["diagnostic"]["head_learning_rate"]),
        "--wave2-dog-breed-epochs",
        str(source["epochs"]),
        "--wave2-dog-breed-batch-size",
        str(source["batch_size"]),
        "--wave2-vision-folds",
        str(source["folds"]),
    ]
    if diagnostic:
        values.extend(
            [
                "--wave2-dog-breed-diagnostic-fold-limit",
                str(source["fold_limit"]),
                "--wave2-dog-breed-diagnostic-parent-fold0-epoch1-log-loss",
                str(source["parent_fold0_epoch1_log_loss"]),
                "--wave2-dog-breed-diagnostic-parent-fold0-top1-accuracy",
                str(source["parent_fold0_top1_accuracy"]),
            ]
        )
    return values


def _collected_result(run_id: str) -> dict[str, Any]:
    result_path = (
        ops.LOCAL_CONTROL_ROOT
        / "collected"
        / run_id
        / COMPETITION_ID
        / "result.json"
    )
    if not result_path.is_file():
        ops.collect_run(run_id)
    require(result_path.is_file(), f"Dog collected result is missing: {run_id}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    require(
        result.get("competition_id") == COMPETITION_ID
        and result.get("official_grader_executed") is False,
        "Dog collected result contract drifted",
    )
    return result


def _launch(
    plan: Mapping[str, Any],
    evidence_dir: Path,
    *,
    run_id: str,
    seed: int,
    diagnostic: bool,
) -> dict[str, Any]:
    gate_path = evidence_dir / "gpu_gate_current.json"
    gate = ops.sample_gpu_idle_gate(interval_seconds=5)
    write_json_atomic(gate_path, gate)
    if gate.get("passed") is not True:
        return {"status": "waiting_for_gpu_idle", "gpu_gate": gate}
    bundle = Path(str(plan["_bundle_path"]))
    deployment = ops.deploy_bundle(bundle, gate_path, max_gate_age=600)
    smoke = ops.cuda_smoke(bundle, gate_path, max_gate_age=600)
    final_gate = ops.sample_gpu_idle_gate(interval_seconds=5)
    write_json_atomic(gate_path, final_gate)
    require(final_gate.get("passed") is True, "Dog final GPU gate failed")
    start = ops.start_run(
        bundle,
        gate_path,
        run_id=run_id,
        waves=["Wave2"],
        competitions=[COMPETITION_ID],
        seed=seed,
        optimization_plan_name="medal_recovery_gpt56_current.json",
        resume=False,
        allow_concurrent_with_cpu_light=False,
        max_gate_age=600,
        runner_performance_overrides=["--wave2-workers", "32"],
        runner_contract_args=_runner_args(plan, diagnostic=diagnostic),
    )
    return {
        "status": "diagnostic_active" if diagnostic else "full_seed_active",
        "deployment": deployment,
        "cuda_smoke": smoke,
        "gpu_gate": final_gate,
        "start": start,
    }


def run_once(
    plan: Mapping[str, Any],
    *,
    taxi_status_path: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    status_path = evidence_dir / "status_current.json"
    taxi_status = json.loads(Path(taxi_status_path).read_text(encoding="utf-8"))
    dependency = plan["serial_dependency"]
    if taxi_status.get("plan_sha256") != dependency["taxi_plan_sha256"]:
        result = {
            **_base(plan),
            "status": "waiting_for_matching_taxi_plan",
            "observed_taxi_plan_sha256": taxi_status.get("plan_sha256"),
        }
        write_json_atomic(status_path, result)
        return result
    if taxi_status.get("status") != "all_seeds_terminal":
        result = {
            **_base(plan),
            "status": "waiting_for_taxi_terminal",
            "taxi_queue_status": taxi_status.get("status"),
        }
        write_json_atomic(status_path, result)
        return result
    completed_taxi = [
        str(item.get("run_id")) for item in taxi_status.get("completed_seeds") or []
    ]
    require(
        completed_taxi == list(dependency["taxi_run_ids"]),
        "Dog queue Taxi terminal inventory drifted",
    )

    diagnostic = plan["diagnostic"]
    diagnostic_run_id = str(diagnostic["run_id"])
    try:
        remote = ops.read_remote_status(diagnostic_run_id)
    except (ops.RemoteOpsError, OSError, ValueError, json.JSONDecodeError):
        remote = None
    if remote is None:
        launched = _launch(
            plan,
            evidence_dir,
            run_id=diagnostic_run_id,
            seed=int(diagnostic["seed"]),
            diagnostic=True,
        )
        result = {**_base(plan), **launched, "diagnostic_run_id": diagnostic_run_id}
        write_json_atomic(status_path, result)
        return result
    if remote.get("process") == "running":
        result = {
            **_base(plan),
            "status": "diagnostic_active",
            "diagnostic_run_id": diagnostic_run_id,
            "remote_status": remote,
        }
        write_json_atomic(status_path, result)
        return result
    if not taxi_queue.remote_run_terminal(remote):
        result = {
            **_base(plan),
            "status": "waiting_for_diagnostic_terminal",
            "diagnostic_run_id": diagnostic_run_id,
            "remote_status": remote,
        }
        write_json_atomic(status_path, result)
        return result

    diagnostic_result = _collected_result(diagnostic_run_id)
    if diagnostic_result.get("continuation_allowed") is not True:
        result = {
            **_base(plan),
            "status": "diagnostic_gate_failed",
            "diagnostic_run_id": diagnostic_run_id,
            "diagnostic_result": diagnostic_result,
            "full_runs_launched": [],
        }
        write_json_atomic(status_path, result)
        return result

    completed = []
    confirmation = plan["confirmation"]
    for seed, run_id in zip(
        confirmation["seeds"], confirmation["run_ids"], strict=True
    ):
        try:
            remote = ops.read_remote_status(str(run_id))
        except (ops.RemoteOpsError, OSError, ValueError, json.JSONDecodeError):
            remote = None
        if remote is not None and taxi_queue.remote_run_terminal(remote):
            result = _collected_result(str(run_id))
            gate = result.get("promotion_gate") or {}
            if gate.get("passed") is not True or float(result.get("cv_score", 1.0)) > float(
                confirmation["every_seed_oof_log_loss_maximum"]
            ):
                stopped = {
                    **_base(plan),
                    "status": "full_seed_gate_failed",
                    "failed_seed": seed,
                    "failed_run_id": run_id,
                    "completed_seeds": completed,
                    "result": result,
                }
                write_json_atomic(status_path, stopped)
                return stopped
            completed.append({"seed": seed, "run_id": run_id, "result": result})
            continue
        if remote is not None and remote.get("process") == "running":
            active = {
                **_base(plan),
                "status": "full_seed_active",
                "completed_seeds": completed,
                "active_seed": {"seed": seed, "run_id": run_id, "remote_status": remote},
            }
            write_json_atomic(status_path, active)
            return active
        launched = _launch(
            plan,
            evidence_dir,
            run_id=str(run_id),
            seed=int(seed),
            diagnostic=False,
        )
        active = {
            **_base(plan),
            **launched,
            "completed_seeds": completed,
            "active_seed": {"seed": seed, "run_id": run_id},
        }
        write_json_atomic(status_path, active)
        return active

    final = {
        **_base(plan),
        "status": "all_full_seeds_terminal",
        "diagnostic_run_id": diagnostic_run_id,
        "completed_seeds": completed,
    }
    write_json_atomic(status_path, final)
    return final


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--taxi-status", type=Path, default=DEFAULT_TAXI_STATUS)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--deadline-hours", type=float, default=720.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.poll_seconds < 30 or args.deadline_hours <= 0:
        raise SystemExit("Dog queue polling/deadline values are invalid")
    plan = validate_plan(args.plan)
    evidence_dir = Path(args.evidence_dir).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    deadline = datetime.now().astimezone() + timedelta(hours=args.deadline_hours)
    while True:
        try:
            result = run_once(
                plan,
                taxi_status_path=Path(args.taxi_status).resolve(),
                evidence_dir=evidence_dir,
            )
        except Exception as exc:
            result = {
                **_base(plan),
                "status": "transient_or_contract_failure",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            write_json_atomic(evidence_dir / "failure_current.json", result)
            if args.once:
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 1
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        if result["status"] in TERMINAL_QUEUE_STATUSES:
            return 0 if result["status"] != "blocked_by_dependency_integrity_failure" else 5
        if args.once:
            return 0
        if datetime.now().astimezone() >= deadline:
            timeout = {
                **_base(plan),
                "status": "timeout_waiting_for_taxi_or_gpu",
                "deadline": deadline.isoformat(),
            }
            write_json_atomic(evidence_dir / "status_current.json", timeout)
            return 4
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
