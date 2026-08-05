#!/usr/bin/env python3
"""Run Jigsaw model seeds 40 and 41 serially after the Leaf chain completes.

The queue preserves the seed-42 outer folds and sparse OOF bundle, varies only
the model-training seed, requires a fresh calibrated RTX 4060 idle gate before
each launch, and never sends process signals or invokes submission/grader code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
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
    / "jigsaw_confirmation_s40_s41_frozen_plan_20260727.json"
)
DEFAULT_STATUS = (
    PROJECT_ROOT / "workspace" / "local_gpu" / "jigsaw_confirmation_queue.json"
)
EXPECTED_PLAN_SCHEMA = "evomind.jigsaw.confirmation_queue_frozen_plan.v1"
EXPECTED_SEED_PLAN_SCHEMA = "evomind.jigsaw.transformer_confirmation_plan.v2"
EXPECTED_REPORT_SCHEMA = "evomind.jigsaw.transformer_independent_verification.v1"
TERMINAL_REPORT_STATUSES = {"promotion_gate_passed", "promotion_gate_failed"}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
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
        "process_signals_sent": 0,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "human_gate_preserved": True,
        "strict_single_gpu_serial": True,
    }


def validate_artifact(name: str, record: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(record.get("path", ""))).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Frozen artifact is missing: {name}: {path}")
    actual = sha256_file(path)
    if actual != record.get("sha256"):
        raise RuntimeError(f"Frozen artifact changed: {name}")
    return {"name": name, "path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def validate_seed_plan(
    record: dict[str, Any],
    *,
    seed: int,
    source: dict[str, Any],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    plan_record = record.get("plan") or {}
    validated = validate_artifact(f"seed_{seed}.plan", plan_record)
    plan_path = Path(validated["path"])
    plan = read_json(plan_path)
    if plan.get("schema") != EXPECTED_SEED_PLAN_SCHEMA:
        raise RuntimeError(f"Seed {seed} plan schema changed")
    if plan.get("status") != "frozen_before_training":
        raise RuntimeError(f"Seed {seed} plan is not frozen")
    if plan.get("competition_id") != "jigsaw-toxic-comment-classification-challenge":
        raise RuntimeError(f"Seed {seed} plan targets another competition")
    training = plan.get("training") or {}
    if (
        training.get("seed") != seed
        or training.get("fold_seed") != 42
        or training.get("model_seed") != seed
        or training.get("folds") != 5
    ):
        raise RuntimeError(f"Seed {seed} plan does not preserve the confirmation split")
    if seed not in (40, 41):
        raise RuntimeError("Only model seeds 40 and 41 are confirmation jobs")
    if plan.get("inputs") != inputs:
        raise RuntimeError(f"Seed {seed} public input contract differs from the queue")
    implementation = plan.get("implementation") or {}
    for key in ("runner", "verifier"):
        if implementation.get(key) != source.get(key):
            raise RuntimeError(f"Seed {seed} implementation binding differs: {key}")
    execution = plan.get("execution") or {}
    if (
        execution.get("automatic_kaggle_submission") is not False
        or execution.get("official_private_grader_before_gate") is not False
        or execution.get("process_signals_allowed") is not False
        or execution.get("human_gate_preserved") is not True
    ):
        raise RuntimeError(f"Seed {seed} execution boundary changed")
    if record.get("run_id") != execution.get("run_id"):
        raise RuntimeError(f"Seed {seed} run ID differs from its frozen plan")
    plan["_path"] = str(plan_path)
    plan["_sha256"] = validated["sha256"]
    plan["_run_id"] = record["run_id"]
    plan["_model_seed"] = seed
    return plan


def validate_frozen_plan(path: Path) -> dict[str, Any]:
    plan_path = Path(path).resolve()
    plan = read_json(plan_path)
    if plan.get("schema") != EXPECTED_PLAN_SCHEMA:
        raise RuntimeError("Unexpected Jigsaw confirmation queue schema")
    if plan.get("status") != "frozen_waiting_leaf":
        raise RuntimeError("Jigsaw confirmation queue plan is not frozen")
    if plan.get("competition_id") != "jigsaw-toxic-comment-classification-challenge":
        raise RuntimeError("Jigsaw confirmation queue targets another competition")
    planner = plan.get("planner") or {}
    if (
        planner.get("requested_model") != "gpt-5.6-sol"
        or planner.get("served_model") != "gpt-5.6-sol"
    ):
        raise RuntimeError("Jigsaw confirmation plan lacks gpt-5.6-sol provenance")
    boundaries = plan.get("boundaries") or {}
    if (
        boundaries.get("private_labels_used") is not False
        or boundaries.get("official_grader_executed") is not False
        or boundaries.get("kaggle_submission_executed") is not False
        or boundaries.get("process_signals_sent") != 0
        or boundaries.get("human_gate_preserved") is not True
    ):
        raise RuntimeError("Jigsaw confirmation boundary changed")

    validated_artifacts = []
    source = plan.get("source") or {}
    for key in ("runner", "verifier", "queue", "idle_gate"):
        validated_artifacts.append(validate_artifact(f"source.{key}", source[key]))
    inputs = plan.get("inputs") or {}
    for key in ("public_train", "public_test", "public_sample", "sparse_bundle"):
        validated_artifacts.append(validate_artifact(f"inputs.{key}", inputs[key]))
    token_cache = plan.get("token_cache") or {}
    for key in ("train", "train_manifest", "test", "test_manifest"):
        validated_artifacts.append(validate_artifact(f"token_cache.{key}", token_cache[key]))
    seed42 = plan.get("seed42_evidence") or {}
    for key in ("audit", "summary", "independent_verification"):
        validated_artifacts.append(validate_artifact(f"seed42.{key}", seed42[key]))
    seed42_audit = read_json(Path(seed42["audit"]["path"]))
    if (
        seed42_audit.get("seed") != 42
        or seed42_audit.get("status") != "single_seed_candidate_passed_confirmation_required"
        or (seed42_audit.get("promotion_gate") or {}).get("passed") is not True
        or seed42_audit.get("private_labels_used") is not False
        or seed42_audit.get("official_grader_executed_for_candidate") is not False
        or seed42_audit.get("kaggle_submission_executed") is not False
        or seed42_audit.get("process_signals_sent") != 0
    ):
        raise RuntimeError("Seed-42 candidate evidence is not confirmation-ready")

    dependency = plan.get("serial_dependency") or {}
    validated_artifacts.append(validate_artifact("serial_dependency.leaf_plan", dependency["plan"]))
    idle_policy = calibrated_idle.validate_launch_contract(plan["launch_contract"])
    if idle_policy["_sha256"] != source["idle_gate_policy"]["sha256"]:
        raise RuntimeError("Jigsaw idle policy binding changed")

    seed_plans: list[dict[str, Any]] = []
    expected_inputs = {
        "public_train_sha256": inputs["public_train"]["sha256"],
        "public_test_sha256": inputs["public_test"]["sha256"],
        "public_sample_submission_sha256": inputs["public_sample"]["sha256"],
        "sparse_bundle_sha256": inputs["sparse_bundle"]["sha256"],
        "sparse_bundle_numeric_arrays_loaded_without_pickle": True,
        "legacy_object_id_arrays_ignored": True,
        "order_contract": "fixed_seed42_outer_folds_plus_public_truth_exact_match",
    }
    records = plan.get("confirmation_runs") or []
    if [record.get("model_seed") for record in records] != [40, 41]:
        raise RuntimeError("Confirmation run order must be seeds 40 then 41")
    for record in records:
        seed_plans.append(
            validate_seed_plan(
                record,
                seed=int(record["model_seed"]),
                source=source,
                inputs=expected_inputs,
            )
        )
    for key in ("training_python", "verification_python", "public_dir", "output_root", "hf_cache"):
        value = Path(str(plan["execution"][key])).resolve()
        if key.endswith("python") and not value.is_file():
            raise FileNotFoundError(value)
        if not key.endswith("python") and not value.exists():
            raise FileNotFoundError(value)

    plan["_path"] = str(plan_path)
    plan["_sha256"] = sha256_file(plan_path)
    plan["_idle_policy"] = idle_policy
    plan["_validated_artifacts"] = validated_artifacts
    plan["_seed_plans"] = seed_plans
    plan["_seed42_audit"] = seed42_audit
    return plan


def leaf_snapshot(plan: dict[str, Any], watcher_path: Path | None = None) -> dict[str, Any]:
    dependency = plan["serial_dependency"]
    path = Path(watcher_path or dependency["watcher_path"]).resolve()
    payload: dict[str, Any] = {}
    errors: list[str] = []
    if path.is_file():
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            errors.append(f"{type(exc).__name__}:{exc}")
    checks = {
        "schema": payload.get("schema") == dependency["watcher_schema"],
        "run_id": payload.get("run_id") == dependency["run_id"],
        "status": payload.get("status") in set(dependency["terminal_statuses"]),
        "plan_hash": payload.get("plan_sha256") == dependency["plan"]["sha256"],
        "signals": payload.get("process_signals_sent") == 0,
        "private": payload.get("private_labels_used") is False,
        "grader": payload.get("official_grader_executed") is False,
        "kaggle": payload.get("kaggle_submission_executed") is False,
    }
    return {
        "ready": bool(payload and all(checks.values()) and not errors),
        "path": str(path),
        "present": path.is_file(),
        "status": payload.get("status", "missing"),
        "candidate_ready": payload.get("candidate_ready"),
        "checks": checks,
        "errors": errors,
    }


def seed_run_paths(plan: dict[str, Any], seed_plan: dict[str, Any]) -> dict[str, Path]:
    run_dir = Path(plan["execution"]["output_root"]).resolve() / seed_plan["_run_id"]
    return {
        "run_dir": run_dir,
        "summary": run_dir / "summary.json",
        "report": run_dir / "independent_verification.json",
        "train_stdout": run_dir / "confirmation_training.stdout.log",
        "train_stderr": run_dir / "confirmation_training.stderr.log",
        "verify_stdout": run_dir / "confirmation_verifier.stdout.log",
        "verify_stderr": run_dir / "confirmation_verifier.stderr.log",
        "launch": run_dir / "confirmation_launch.json",
    }


def build_training_command(plan: dict[str, Any], seed_plan: dict[str, Any]) -> list[str]:
    training = seed_plan["training"]
    model = seed_plan["model"]
    execution = plan["execution"]
    return [
        execution["training_python"],
        plan["source"]["runner"]["path"],
        "--public-dir",
        execution["public_dir"],
        "--sparse-bundle",
        plan["inputs"]["sparse_bundle"]["path"],
        "--output-root",
        execution["output_root"],
        "--run-id",
        seed_plan["_run_id"],
        "--frozen-plan",
        seed_plan["_path"],
        "--model-id",
        model["repo_id"],
        "--model-revision",
        model["revision"],
        "--hf-cache",
        execution["hf_cache"],
        "--token-cache-dir",
        seed_plan["execution"]["token_cache_dir"],
        "--seed",
        str(training["seed"]),
        "--fold-seed",
        str(training["fold_seed"]),
        "--model-seed",
        str(training["model_seed"]),
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
        "--gradient-checkpointing",
        "--bf16",
    ]


def build_verifier_command(
    plan: dict[str, Any], seed_plan: dict[str, Any], report_path: Path
) -> list[str]:
    paths = seed_run_paths(plan, seed_plan)
    return [
        plan["execution"]["verification_python"],
        plan["source"]["verifier"]["path"],
        "--run-dir",
        str(paths["run_dir"]),
        "--public-dir",
        plan["execution"]["public_dir"],
        "--sparse-bundle",
        plan["inputs"]["sparse_bundle"]["path"],
        "--frozen-plan",
        seed_plan["_path"],
        "--report-path",
        str(report_path),
        "--require-complete",
    ]


def report_valid(report: dict[str, Any], seed_plan: dict[str, Any]) -> bool:
    return bool(
        report.get("schema") == EXPECTED_REPORT_SCHEMA
        and report.get("run_id") == seed_plan["_run_id"]
        and report.get("plan_sha256") == seed_plan["_sha256"]
        and report.get("status") in TERMINAL_REPORT_STATUSES
        and report.get("full_contract_valid") is True
        and report.get("private_labels_used") is False
        and report.get("official_grader_executed") is False
        and report.get("kaggle_submission_executed") is False
        and report.get("human_gate_preserved") is True
        and report.get("process_signals_sent") == 0
        and (report.get("seed_contract") or {}).get("fold_assignment_seed") == 42
        and (report.get("seed_contract") or {}).get("model_seed")
        == seed_plan["_model_seed"]
    )


def run_verifier(plan: dict[str, Any], seed_plan: dict[str, Any]) -> dict[str, Any]:
    paths = seed_run_paths(plan, seed_plan)
    command = build_verifier_command(plan, seed_plan, paths["report"])
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    paths["verify_stdout"].write_text(completed.stdout, encoding="utf-8")
    paths["verify_stderr"].write_text(completed.stderr, encoding="utf-8")
    report = read_json(paths["report"]) if paths["report"].is_file() else {}
    if completed.returncode not in (0, 3) or not report_valid(report, seed_plan):
        raise RuntimeError(
            f"Seed {seed_plan['_model_seed']} independent verification failed: "
            f"exit={completed.returncode}"
        )
    return report


def load_or_verify_existing(
    plan: dict[str, Any], seed_plan: dict[str, Any]
) -> dict[str, Any] | None:
    paths = seed_run_paths(plan, seed_plan)
    if paths["report"].is_file():
        report = read_json(paths["report"])
        if report_valid(report, seed_plan):
            return report
    if paths["summary"].is_file():
        summary = read_json(paths["summary"])
        if (
            summary.get("status") in TERMINAL_REPORT_STATUSES
            and summary.get("plan_sha256") == seed_plan["_sha256"]
        ):
            return run_verifier(plan, seed_plan)
    return None


def aggregate_confirmation(
    plan: dict[str, Any], reports: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    seed42 = plan["_seed42_audit"]
    records = [
        {
            "model_seed": 42,
            "status": "promotion_gate_passed",
            "candidate_auc": float(seed42["metrics"]["candidate_auc"]),
            "gain_over_strongest_base": float(
                seed42["metrics"]["gain_over_strongest_base"]
            ),
            "promotion_passed": True,
            "source": seed42["run_dir"],
        }
    ]
    for report in reports:
        records.append(
            {
                "model_seed": int(report["seed_contract"]["model_seed"]),
                "status": report["status"],
                "candidate_auc": float(report["metrics"]["candidate_auc"]),
                "gain_over_strongest_base": float(
                    report["metrics"]["gain_over_strongest_base"]
                ),
                "promotion_passed": report["status"] == "promotion_gate_passed",
                "source": report["run_dir"],
            }
        )
    records.sort(key=lambda item: item["model_seed"])
    values = [item["candidate_auc"] for item in records]
    gains = [item["gain_over_strongest_base"] for item in records]
    gate = plan["confirmation_gate"]
    checks = {
        "exact_model_seeds_40_41_42": [item["model_seed"] for item in records]
        == [40, 41, 42],
        "all_seed_promotion_gates_passed": all(
            item["promotion_passed"] for item in records
        ),
        "minimum_candidate_auc": min(values) >= float(gate["minimum_seed_auc"]),
        "mean_candidate_auc": statistics.fmean(values)
        >= float(gate["minimum_mean_auc"]),
        "minimum_gain": min(gains) >= float(gate["minimum_seed_gain"]),
        "maximum_population_std": statistics.pstdev(values)
        <= float(gate["maximum_population_std"]),
        "public_data_only": True,
        "human_gate_preserved": True,
    }
    passed = all(checks.values())
    return {
        "schema": "evomind.jigsaw.multiseed_confirmation_result.v1",
        "created_at": now_iso(),
        "status": "confirmation_passed_human_gate_pending" if passed else "confirmation_failed",
        "competition_id": "jigsaw-toxic-comment-classification-challenge",
        "plan_path": plan["_path"],
        "plan_sha256": plan["_sha256"],
        "seed_records": records,
        "metrics": {
            "minimum_candidate_auc": min(values),
            "mean_candidate_auc": statistics.fmean(values),
            "maximum_candidate_auc": max(values),
            "candidate_auc_population_std": statistics.pstdev(values),
            "minimum_gain_over_strongest_base": min(gains),
        },
        "confirmation_gate": {**gate, "checks": checks, "passed": passed},
        "candidate_ready_for_human_gate": passed,
        **invariant_fields(),
        "claim_boundary": "Three-seed public OOF confirmation is not an official score or medal.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--deadline-hours", type=float, default=240.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if compute_policy.write_queue_superseded_if_hpc_only(
        project_root=PROJECT_ROOT,
        status_path=args.status,
        schema="evomind.jigsaw.confirmation_queue.v1",
        queue_name="queue_jigsaw_confirmation_after_leaf",
        plan_path=args.plan,
    ):
        return 0
    if args.poll_seconds < 10 or args.deadline_hours <= 0:
        raise ValueError("Jigsaw confirmation queue timing contract is invalid")
    plan = validate_frozen_plan(args.plan)
    status_path = args.status.resolve()
    deadline = datetime.now().astimezone() + timedelta(hours=args.deadline_hours)
    prerequisite: dict[str, Any] = {}

    while datetime.now().astimezone() < deadline:
        if sha256_file(Path(plan["_path"])) != plan["_sha256"]:
            raise RuntimeError("Frozen confirmation plan changed while queue ran")
        prerequisite = leaf_snapshot(plan)
        write_json_atomic(
            status_path,
            {
                "schema": "evomind.jigsaw.confirmation_queue.v1",
                "created_at": now_iso(),
                "status": "waiting_for_leaf_verification",
                "pid": os.getpid(),
                "deadline": deadline.isoformat(),
                "plan_path": plan["_path"],
                "plan_sha256": plan["_sha256"],
                "prerequisite": prerequisite,
                "confirmation_seeds": [40, 41],
                "completed_seeds": [],
                "active_seed": None,
                "consecutive_idle_checks": 0,
                "required_idle_checks": plan["_idle_policy"]["requirements"][
                    "consecutive_checks"
                ],
                "gpu_idle_gate": calibrated_idle.policy_record(plan["_idle_policy"]),
                **invariant_fields(),
            },
        )
        if prerequisite["ready"]:
            break
        time.sleep(args.poll_seconds)
    else:
        write_json_atomic(
            status_path,
            {
                "schema": "evomind.jigsaw.confirmation_queue.v1",
                "created_at": now_iso(),
                "status": "timeout_waiting_for_leaf_verification",
                "pid": os.getpid(),
                "deadline": deadline.isoformat(),
                "plan_path": plan["_path"],
                "plan_sha256": plan["_sha256"],
                **invariant_fields(),
            },
        )
        return 4

    reports: list[dict[str, Any]] = []
    completed_seeds: list[int] = []
    requirements = plan["_idle_policy"]["requirements"]
    for seed_plan in plan["_seed_plans"]:
        seed = int(seed_plan["_model_seed"])
        existing = load_or_verify_existing(plan, seed_plan)
        if existing is not None:
            reports.append(existing)
            completed_seeds.append(seed)
            continue

        consecutive = 0
        gpu: dict[str, Any] | None = None
        evaluation: dict[str, Any] | None = None
        while datetime.now().astimezone() < deadline:
            gpu = calibrated_idle.query_gpu()
            evaluation = calibrated_idle.evaluate_idle(plan["_idle_policy"], gpu)
            consecutive = consecutive + 1 if evaluation["idle"] else 0
            write_json_atomic(
                status_path,
                {
                    "schema": "evomind.jigsaw.confirmation_queue.v1",
                    "created_at": now_iso(),
                    "status": f"waiting_for_stable_gpu_idle_seed_{seed}",
                    "pid": os.getpid(),
                    "deadline": deadline.isoformat(),
                    "plan_path": plan["_path"],
                    "plan_sha256": plan["_sha256"],
                    "prerequisite": prerequisite,
                    "confirmation_seeds": [40, 41],
                    "completed_seeds": completed_seeds,
                    "active_seed": seed,
                    "gpu": gpu,
                    "idle_gate_evaluation": evaluation,
                    "consecutive_idle_checks": consecutive,
                    "required_idle_checks": requirements["consecutive_checks"],
                    "gpu_idle_gate": calibrated_idle.policy_record(plan["_idle_policy"]),
                    **invariant_fields(),
                },
            )
            if consecutive >= int(requirements["consecutive_checks"]):
                break
            time.sleep(int(requirements["minimum_check_interval_seconds"]))
        else:
            raise TimeoutError(f"Timed out waiting for idle GPU before seed {seed}")

        if compute_policy.write_queue_superseded_if_hpc_only(
            project_root=PROJECT_ROOT,
            status_path=args.status,
            schema="evomind.jigsaw.confirmation_queue.v1",
            queue_name="queue_jigsaw_confirmation_after_leaf",
            plan_path=args.plan,
        ):
            return 0
        paths = seed_run_paths(plan, seed_plan)
        paths["run_dir"].mkdir(parents=True, exist_ok=True)
        command = build_training_command(plan, seed_plan)
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
            write_json_atomic(
                paths["launch"],
                {
                    "schema": "evomind.jigsaw.confirmation_launch.v1",
                    "created_at": now_iso(),
                    "status": "launched",
                    "model_seed": seed,
                    "fold_seed": 42,
                    "pid": process.pid,
                    "command": command,
                    "plan_path": seed_plan["_path"],
                    "plan_sha256": seed_plan["_sha256"],
                    "gpu_preflight": gpu,
                    "idle_gate_evaluation": evaluation,
                    "consecutive_idle_checks": consecutive,
                    **invariant_fields(),
                },
            )
            write_json_atomic(
                status_path,
                {
                    "schema": "evomind.jigsaw.confirmation_queue.v1",
                    "created_at": now_iso(),
                    "status": f"training_seed_{seed}",
                    "pid": os.getpid(),
                    "training_pid": process.pid,
                    "active_seed": seed,
                    "completed_seeds": completed_seeds,
                    "plan_path": plan["_path"],
                    "plan_sha256": plan["_sha256"],
                    "seed_plan_path": seed_plan["_path"],
                    "seed_plan_sha256": seed_plan["_sha256"],
                    "stdout": str(paths["train_stdout"]),
                    "stderr": str(paths["train_stderr"]),
                    **invariant_fields(),
                },
            )
            return_code = process.wait()
        if return_code != 0:
            write_json_atomic(
                status_path,
                {
                    "schema": "evomind.jigsaw.confirmation_queue.v1",
                    "created_at": now_iso(),
                    "status": f"training_failed_seed_{seed}",
                    "pid": os.getpid(),
                    "training_exit_code": return_code,
                    "active_seed": seed,
                    "completed_seeds": completed_seeds,
                    "stdout": str(paths["train_stdout"]),
                    "stderr": str(paths["train_stderr"]),
                    **invariant_fields(),
                },
            )
            return 3
        report = run_verifier(plan, seed_plan)
        reports.append(report)
        completed_seeds.append(seed)

    result = aggregate_confirmation(plan, reports)
    output = Path(plan["output"]["result"]).resolve()
    write_json_atomic(output, result)
    write_json_atomic(
        status_path,
        {
            "schema": "evomind.jigsaw.confirmation_queue.v1",
            "created_at": now_iso(),
            "status": result["status"],
            "pid": os.getpid(),
            "plan_path": plan["_path"],
            "plan_sha256": plan["_sha256"],
            "completed_seeds": completed_seeds,
            "active_seed": None,
            "result": str(output),
            "result_sha256": sha256_file(output),
            "candidate_ready_for_human_gate": result[
                "candidate_ready_for_human_gate"
            ],
            **invariant_fields(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
