#!/usr/bin/env python3
"""Run and verify frozen Spooky seeds 40/41, then build the three-seed gate.

The queue waits for a verified seed-42 public OOF run, gives the authoritative
SIIM/Leaf/May candidate chain priority, and uses the same three-check local GPU
idle gate as the seed-42 plan.  It never preempts or signals another process.
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

from scripts import aggregate_spooky_multiseed_gate as aggregate  # noqa: E402
from scripts import local_rtx4060_idle_gate as idle_gate  # noqa: E402
from scripts import queue_leaf_after_siim as leaf_queue  # noqa: E402
from scripts import queue_may2022_after_spooky as may_queue  # noqa: E402
from scripts import queue_siim_final_candidate as siim_queue  # noqa: E402
from scripts import queue_spooky_after_leaf as spooky_queue  # noqa: E402

DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "spooky_confirmation_s4041_frozen_manifest_v2_20260728.json"
)
DEFAULT_SEED42_PLAN = spooky_queue.DEFAULT_PLAN
DEFAULT_SIIM_PLAN = siim_queue.DEFAULT_PLAN
DEFAULT_LEAF_PLAN = leaf_queue.DEFAULT_PLAN
DEFAULT_MAY_PLAN = may_queue.DEFAULT_PLAN
DEFAULT_VERIFIER = PROJECT_ROOT / "scripts" / "verify_spooky_transformer_oof_run.py"
DEFAULT_AGGREGATOR = PROJECT_ROOT / "scripts" / "aggregate_spooky_multiseed_gate.py"
DEFAULT_GATE_POLICY = idle_gate.DEFAULT_POLICY
DEFAULT_STATUS = (
    PROJECT_ROOT / "workspace" / "local_gpu" / "spooky_confirmation_multiseed_queue.json"
)
DEFAULT_AGGREGATE_OUTPUT = (
    PROJECT_ROOT
    / "workspace"
    / "local_gpu"
    / "mlebench_lite_runs"
    / "spooky_multiseed_s404142_20260727_0459"
)


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


def validate_manifest(path: Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    manifest = read_json(manifest_path)
    if manifest.get("schema") != "evomind.spooky.confirmation_plans_manifest.v1":
        raise RuntimeError("Unexpected Spooky confirmation manifest schema")
    if manifest.get("status") != "frozen_before_confirmation_training":
        raise RuntimeError("Spooky confirmation manifest is not frozen")
    if (
        manifest.get("requested_model") != "gpt-5.6-sol"
        or manifest.get("served_model") != "gpt-5.6-sol"
    ):
        raise RuntimeError("Spooky confirmation lacks verified gpt-5.6-sol provenance")
    if manifest.get("required_seeds") != [40, 41, 42]:
        raise RuntimeError("Spooky confirmation seeds changed")
    source_path = Path(manifest["source_seed42_plan"]).resolve()
    if not source_path.is_file() or sha256_file(source_path) != manifest.get(
        "source_seed42_plan_sha256"
    ):
        raise RuntimeError("Spooky seed-42 source plan changed")
    audit_path = Path(manifest["driver_audit_path"]).resolve()
    if not audit_path.is_file() or sha256_file(audit_path) != manifest.get(
        "driver_audit_sha256"
    ):
        raise RuntimeError("Spooky gpt-5.6-sol decision audit changed")

    plans: dict[int, dict[str, Any]] = {}
    records = manifest.get("confirmation_plans") or []
    if [value.get("seed") for value in records] != [40, 41]:
        raise RuntimeError("Spooky confirmation manifest must contain seeds 40 and 41")
    for record in records:
        plan_path = Path(record["path"]).resolve()
        if not plan_path.is_file() or sha256_file(plan_path) != record.get("sha256"):
            raise RuntimeError(f"Spooky confirmation seed {record.get('seed')} plan changed")
        plan = spooky_queue.validate_frozen_plan(plan_path)
        seed = int(record["seed"])
        if plan["training"]["seed"] != seed:
            raise RuntimeError(f"Spooky confirmation seed {seed} plan is inconsistent")
        confirmation = plan.get("confirmation") or {}
        if (
            confirmation.get("seed") != seed
            or confirmation.get("source_seed42_plan_sha256")
            != manifest["source_seed42_plan_sha256"]
            or confirmation.get("predeclared_before_confirmation_training") is not True
            or confirmation.get("private_grader_feedback_used") is not False
        ):
            raise RuntimeError(f"Spooky confirmation seed {seed} contract changed")
        plans[seed] = plan
    manifest["_path"] = str(manifest_path)
    manifest["_sha256"] = sha256_file(manifest_path)
    manifest["_plans"] = plans
    return manifest


def terminal_status(path: Path, statuses: set[str]) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        return {"ready": False, "status": "missing", "path": str(resolved)}
    try:
        payload = read_json(resolved)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        return {
            "ready": False,
            "status": "invalid_json",
            "path": str(resolved),
            "error": str(exc),
        }
    status = payload.get("status")
    return {
        "ready": status in statuses,
        "status": status,
        "path": str(resolved),
        "sha256": sha256_file(resolved),
    }


def primary_chain_snapshot(
    siim_plan: dict[str, Any],
    leaf_plan: dict[str, Any],
    may_plan: dict[str, Any],
    *,
    seed42_ready: bool,
) -> dict[str, Any]:
    """Declare when the primary recovery candidates own the local GPU."""

    staging = siim_queue.staging_snapshot(siim_plan)
    siim_training = siim_plan["training"]
    siim_summary = terminal_status(
        Path(siim_training["output_root"]) / siim_training["run_id"] / "summary.json",
        {"completed", "candidate_ready", "candidate_failed", "failed"},
    )
    # The actual Leaf result path is derived from its own frozen training contract.
    leaf_report_path = (
        Path(leaf_plan["training"]["output_root"])
        / "runs"
        / leaf_plan["training"]["run_id"]
        / "leaf_multibackbone_oof.json"
    )
    leaf = terminal_status(
        leaf_report_path,
        {"promotion_gate_passed", "promotion_gate_failed"},
    )
    may_execution = may_plan["execution"]
    may = terminal_status(
        Path(may_execution["output_root"])
        / "runs"
        / may_execution["run_id"]
        / "summary.json",
        {"single_seed_gate_passed", "single_seed_gate_failed"},
    )
    siim_leaf_chain_complete = siim_summary["ready"] and leaf["ready"]
    reasons: list[str] = []
    if seed42_ready and not may["ready"]:
        reasons.append("may2022_after_verified_spooky")
    if staging["ready"] and not (siim_leaf_chain_complete and may["ready"]):
        reasons.append("siim_leaf_primary_chain")
    return {
        "active": bool(reasons),
        "reasons": reasons,
        "siim_staging": staging,
        "siim_final": siim_summary,
        "leaf": leaf,
        "may2022": may,
        "siim_leaf_chain_complete": siim_leaf_chain_complete,
    }


def verified_seed_snapshot(
    seed: int,
    run_dir: Path,
    plan: dict[str, Any],
) -> dict[str, Any]:
    resolved = Path(run_dir).resolve()
    required = [
        resolved / "summary.json",
        resolved / "independent_verification.json",
        resolved / "spooky_transformer_oof_and_test.npz",
        resolved / "candidate_submission_withheld.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {
            "ready": False,
            "status": "run_missing" if not resolved.exists() else "run_incomplete",
            "run_dir": str(resolved),
            "run_exists": resolved.exists(),
            "missing": missing,
        }
    try:
        summary = read_json(resolved / "summary.json")
        verification = read_json(resolved / "independent_verification.json")
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        return {
            "ready": False,
            "status": "terminal_evidence_unreadable",
            "run_dir": str(resolved),
            "run_exists": True,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if summary.get("status") == "single_seed_gate_failed":
        terminal_checks = {
            "summary_plan_hash": summary.get("plan_sha256") == plan["_sha256"],
            "verification_plan_hash": verification.get("plan", {}).get("sha256")
            == plan["_sha256"],
            "verification_failed": verification.get("status") == "failed",
            "private_labels_unused": summary.get("private_labels_used") is False
            and verification.get("private_labels_used") is False,
            "official_grader_not_executed": summary.get("official_grader_executed") is False
            and verification.get("official_grader_executed") is False,
            "kaggle_submission_not_executed": summary.get("kaggle_submission_executed") is False
            and verification.get("kaggle_submission_executed") is False,
            "process_signals_not_sent": verification.get("process_signals_sent") == 0,
        }
        terminal = all(terminal_checks.values())
        return {
            "ready": False,
            "terminal": terminal,
            "status": (
                "terminal_single_seed_gate_failed"
                if terminal
                else "terminal_failure_contract_invalid"
            ),
            "run_dir": str(resolved),
            "run_exists": True,
            "run_id": summary.get("run_id"),
            "score": summary.get("candidate_oof_log_loss"),
            "checks": terminal_checks,
        }
    try:
        seed_run = aggregate.load_seed_run(seed, resolved)
        checks = {
            **seed_run.checks,
            "verification_plan_hash": verification.get("plan", {}).get("sha256")
            == plan["_sha256"],
            "run_id_matches_plan": seed_run.run_id == plan["execution"]["run_id"],
        }
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError, TypeError) as exc:
        return {
            "ready": False,
            "status": "verification_invalid",
            "run_dir": str(resolved),
            "run_exists": True,
            "error": str(exc),
        }
    return {
        "ready": all(checks.values()),
        "status": "verified_seed_gate_passed" if all(checks.values()) else "checks_failed",
        "run_dir": str(resolved),
        "run_exists": True,
        "run_id": seed_run.run_id,
        "score": seed_run.score,
        "checks": checks,
    }


def write_status(
    path: Path,
    *,
    status: str,
    deadline: datetime,
    manifest: dict[str, Any],
    seed42: dict[str, Any],
    primary: dict[str, Any],
    confirmations: dict[int, dict[str, Any]],
    gpu: dict[str, Any] | None,
    idle_checks: int,
    required_idle_checks: int,
    **extra: Any,
) -> None:
    payload: dict[str, Any] = {
        "schema": "evomind.spooky.confirmation_multiseed_queue.v1",
        "created_at": now_iso(),
        "status": status,
        "pid": os.getpid(),
        "deadline": deadline.isoformat(),
        "manifest_path": manifest["_path"],
        "manifest_sha256": manifest["_sha256"],
        "requested_model": manifest["requested_model"],
        "served_model": manifest["served_model"],
        "seed42": seed42,
        "primary_chain": primary,
        "confirmations": {str(seed): value for seed, value in confirmations.items()},
        "gpu": gpu,
        "consecutive_idle_checks": idle_checks,
        "required_idle_checks": required_idle_checks,
        "gpu_idle_gate": manifest.get("_idle_gate_policy"),
        "idle_gate_evaluation": (gpu or {}).get("_idle_gate_evaluation"),
        "single_gpu_strict_serial": True,
        "preemption_allowed": False,
        "process_signals_sent": 0,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }
    payload.update(extra)
    spooky_queue.write_json_atomic(Path(path).resolve(), payload)


def create_launch_claim(path: Path, payload: dict[str, Any]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0),
        )
    except FileExistsError:
        return False
    try:
        os.write(
            descriptor,
            (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        )
    finally:
        os.close(descriptor)
    return True


def build_verifier_command(plan: dict[str, Any], verifier: Path) -> list[str]:
    execution = plan["execution"]
    run_dir = Path(execution["output_root"]) / execution["run_id"]
    return [
        execution["python"],
        str(Path(verifier).resolve()),
        "--run-dir",
        str(run_dir.resolve()),
        "--public-dir",
        str(Path(execution["public_dir"]).resolve()),
        "--plan",
        plan["_path"],
        "--output",
        str((run_dir / "independent_verification.json").resolve()),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed42-plan", type=Path, default=DEFAULT_SEED42_PLAN)
    parser.add_argument("--siim-plan", type=Path, default=DEFAULT_SIIM_PLAN)
    parser.add_argument("--leaf-plan", type=Path, default=DEFAULT_LEAF_PLAN)
    parser.add_argument("--may-plan", type=Path, default=DEFAULT_MAY_PLAN)
    parser.add_argument("--verifier", type=Path, default=DEFAULT_VERIFIER)
    parser.add_argument("--aggregator", type=Path, default=DEFAULT_AGGREGATOR)
    parser.add_argument("--gate-policy", type=Path, default=DEFAULT_GATE_POLICY)
    parser.add_argument("--aggregate-output", type=Path, default=DEFAULT_AGGREGATE_OUTPUT)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--training-poll-seconds", type=int, default=30)
    parser.add_argument("--priority-grace-seconds", type=int, default=20)
    parser.add_argument("--deadline-hours", type=float, default=240.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        min(args.poll_seconds, args.training_poll_seconds, args.priority_grace_seconds) < 10
        or args.deadline_hours <= 0
    ):
        raise ValueError("Spooky confirmation queue timing contract is invalid")
    for path in (args.verifier, args.aggregator):
        if not Path(path).resolve().is_file():
            raise FileNotFoundError(path)

    manifest = validate_manifest(args.manifest)
    policy = idle_gate.validate_policy(args.gate_policy)
    seed42_plan = spooky_queue.validate_frozen_plan(args.seed42_plan)
    siim_plan = siim_queue.validate_frozen_plan(args.siim_plan)
    leaf_plan = leaf_queue.validate_frozen_plan(args.leaf_plan)
    may_plan = may_queue.validate_frozen_plan(args.may_plan)
    frozen_hashes = {
        "manifest": manifest["_sha256"],
        "seed42": seed42_plan["_sha256"],
        "siim": siim_plan["_plan_sha256"],
        "leaf": leaf_plan["_plan_sha256"],
        "may": may_plan["_sha256"],
        "verifier": sha256_file(args.verifier),
        "aggregator": sha256_file(args.aggregator),
        "idle_gate_policy": policy["_sha256"],
    }
    if manifest["source_seed42_plan_sha256"] != seed42_plan["_sha256"]:
        raise RuntimeError("Confirmation manifest and seed-42 plan disagree")
    if (
        policy["bound_execution_evidence"]["seed42_plan"]["sha256"]
        != seed42_plan["_sha256"]
        or policy["bound_execution_evidence"]["confirmation_manifest"]["sha256"]
        != manifest["_sha256"]
    ):
        raise RuntimeError("Calibrated idle gate is not bound to this confirmation chain")
    if min(args.poll_seconds, args.training_poll_seconds) < int(
        policy["requirements"]["minimum_check_interval_seconds"]
    ):
        raise ValueError("Confirmation poll interval is below the calibrated gate minimum")
    manifest["_idle_gate_policy"] = idle_gate.policy_record(policy)
    deadline = datetime.now().astimezone() + timedelta(hours=args.deadline_hours)
    required_idle = int(policy["requirements"]["consecutive_checks"])
    idle_checks = 0

    while datetime.now().astimezone() < deadline:
        # Every loop rechecks immutable execution evidence before considering GPU work.
        if sha256_file(args.manifest) != frozen_hashes["manifest"]:
            raise RuntimeError("Spooky confirmation manifest changed while queued")
        if sha256_file(args.verifier) != frozen_hashes["verifier"]:
            raise RuntimeError("Spooky verifier changed while queued")
        if sha256_file(args.aggregator) != frozen_hashes["aggregator"]:
            raise RuntimeError("Spooky aggregator changed while queued")
        if idle_gate.sha256_file(args.gate_policy) != frozen_hashes["idle_gate_policy"]:
            raise RuntimeError("Calibrated idle-gate policy changed while queued")
        for seed, plan in manifest["_plans"].items():
            if sha256_file(plan["_path"]) != plan["_sha256"]:
                raise RuntimeError(f"Spooky confirmation seed {seed} plan changed")

        seed42_run = (
            Path(seed42_plan["execution"]["output_root"])
            / seed42_plan["execution"]["run_id"]
        )
        seed42 = verified_seed_snapshot(42, seed42_run, seed42_plan)
        primary = primary_chain_snapshot(
            siim_plan,
            leaf_plan,
            may_plan,
            seed42_ready=seed42["ready"],
        )
        confirmations = {
            seed: verified_seed_snapshot(
                seed,
                Path(plan["execution"]["output_root"]) / plan["execution"]["run_id"],
                plan,
            )
            for seed, plan in manifest["_plans"].items()
        }

        if seed42.get("terminal") is True:
            write_status(
                args.status,
                status="seed42_single_seed_gate_failed_no_confirmation",
                deadline=deadline,
                manifest=manifest,
                seed42=seed42,
                primary=primary,
                confirmations=confirmations,
                gpu=None,
                idle_checks=0,
                required_idle_checks=required_idle,
                frozen_hashes=frozen_hashes,
                promotion_allowed=False,
                confirmation_training_launched=False,
            )
            return 0

        if not seed42["ready"]:
            idle_checks = 0
            write_status(
                args.status,
                status="waiting_for_verified_seed42",
                deadline=deadline,
                manifest=manifest,
                seed42=seed42,
                primary=primary,
                confirmations=confirmations,
                gpu=None,
                idle_checks=0,
                required_idle_checks=required_idle,
                frozen_hashes=frozen_hashes,
            )
            time.sleep(args.poll_seconds)
            continue

        if all(value["ready"] for value in confirmations.values()):
            output_dir = Path(args.aggregate_output).resolve()
            report_path = output_dir / "spooky_multiseed_gate.json"
            if report_path.is_file():
                report = read_json(report_path)
            else:
                report = aggregate.aggregate_runs(
                    {
                        40: Path(confirmations[40]["run_dir"]),
                        41: Path(confirmations[41]["run_dir"]),
                        42: Path(seed42["run_dir"]),
                    },
                    output_dir,
                    required_seeds=(40, 41, 42),
                    mean_threshold=float(manifest["mean_seed_log_loss_threshold"]),
                    max_seed_threshold=float(manifest["maximum_individual_seed_log_loss"]),
                )
            passed = bool(
                report.get("schema") == "evomind.spooky.multiseed_promotion_gate.v1"
                and report.get("promotion_allowed") is True
                and report.get("process_signals_sent") == 0
                and report.get("private_labels_used") is False
                and report.get("official_grader_executed") is False
                and report.get("kaggle_submission_executed") is False
            )
            write_status(
                args.status,
                status="multiseed_gate_passed" if passed else "multiseed_gate_failed",
                deadline=deadline,
                manifest=manifest,
                seed42=seed42,
                primary=primary,
                confirmations=confirmations,
                gpu=None,
                idle_checks=idle_checks,
                required_idle_checks=required_idle,
                aggregate_report=str(report_path),
                aggregate_report_sha256=sha256_file(report_path),
                promotion_allowed=passed,
                frozen_hashes=frozen_hashes,
            )
            return 0 if passed else 3

        if primary["active"]:
            idle_checks = 0
            write_status(
                args.status,
                status="waiting_for_primary_chain_priority",
                deadline=deadline,
                manifest=manifest,
                seed42=seed42,
                primary=primary,
                confirmations=confirmations,
                gpu=None,
                idle_checks=0,
                required_idle_checks=required_idle,
                frozen_hashes=frozen_hashes,
            )
            time.sleep(args.poll_seconds)
            continue

        next_seed = next(seed for seed in (40, 41) if not confirmations[seed]["ready"])
        next_snapshot = confirmations[next_seed]
        if next_snapshot["run_exists"]:
            write_status(
                args.status,
                status="confirmation_run_exists_but_is_not_verified",
                deadline=deadline,
                manifest=manifest,
                seed42=seed42,
                primary=primary,
                confirmations=confirmations,
                gpu=None,
                idle_checks=0,
                required_idle_checks=required_idle,
                next_seed=next_seed,
                frozen_hashes=frozen_hashes,
            )
            return 5

        gpu = idle_gate.query_gpu()
        evaluation = idle_gate.evaluate_idle(policy, gpu)
        gpu["_idle_gate_evaluation"] = evaluation
        is_idle = evaluation["idle"]
        idle_checks = idle_checks + 1 if is_idle else 0
        write_status(
            args.status,
            status="waiting_for_stable_gpu_idle",
            deadline=deadline,
            manifest=manifest,
            seed42=seed42,
            primary=primary,
            confirmations=confirmations,
            gpu=gpu,
            idle_checks=idle_checks,
            required_idle_checks=required_idle,
            next_seed=next_seed,
            frozen_hashes=frozen_hashes,
        )
        if idle_checks < required_idle:
            time.sleep(args.poll_seconds)
            continue

        time.sleep(args.priority_grace_seconds)
        primary = primary_chain_snapshot(
            siim_plan,
            leaf_plan,
            may_plan,
            seed42_ready=True,
        )
        final_gpu = idle_gate.query_gpu()
        final_evaluation = idle_gate.evaluate_idle(policy, final_gpu)
        final_gpu["_idle_gate_evaluation"] = final_evaluation
        if (
            primary["active"]
            or not final_evaluation["idle"]
        ):
            idle_checks = 0
            time.sleep(args.poll_seconds)
            continue

        plan = manifest["_plans"][next_seed]
        run_dir = Path(plan["execution"]["output_root"]) / plan["execution"]["run_id"]
        claim_path = run_dir.with_suffix(".confirmation_launch_claim.json")
        claim = {
            "schema": "evomind.spooky.confirmation_launch_claim.v1",
            "created_at": now_iso(),
            "queue_pid": os.getpid(),
            "seed": next_seed,
            "run_id": plan["execution"]["run_id"],
            "plan_sha256": plan["_sha256"],
            "manifest_sha256": manifest["_sha256"],
            "idle_gate_policy_sha256": policy["_sha256"],
            "process_signals_sent": 0,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        }
        if not create_launch_claim(claim_path, claim):
            write_status(
                args.status,
                status="confirmation_launch_claim_exists",
                deadline=deadline,
                manifest=manifest,
                seed42=seed42,
                primary=primary,
                confirmations=confirmations,
                gpu=final_gpu,
                idle_checks=idle_checks,
                required_idle_checks=required_idle,
                next_seed=next_seed,
                launch_claim=str(claim_path),
                frozen_hashes=frozen_hashes,
            )
            return 5

        command = spooky_queue.build_training_command(plan)
        stdout_path = run_dir.with_suffix(".stdout.log")
        stderr_path = run_dir.with_suffix(".stderr.log")
        environment = os.environ.copy()
        python_path = os.pathsep.join([str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)])
        if environment.get("PYTHONPATH"):
            python_path += os.pathsep + environment["PYTHONPATH"]
        environment["PYTHONPATH"] = python_path
        Path(plan["execution"]["output_root"]).mkdir(parents=True, exist_ok=True)
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
        while child.poll() is None:
            write_status(
                args.status,
                status="confirmation_training_running",
                deadline=deadline,
                manifest=manifest,
                seed42=seed42,
                primary=primary,
                confirmations=confirmations,
                gpu=idle_gate.query_gpu(),
                idle_checks=idle_checks,
                required_idle_checks=required_idle,
                training_seed=next_seed,
                training_pid=child.pid,
                command=command,
                stdout=str(stdout_path),
                stderr=str(stderr_path),
                launch_claim=str(claim_path),
                frozen_hashes=frozen_hashes,
            )
            time.sleep(args.training_poll_seconds)
        if child.returncode != 0:
            write_status(
                args.status,
                status="confirmation_training_failed",
                deadline=deadline,
                manifest=manifest,
                seed42=seed42,
                primary=primary,
                confirmations=confirmations,
                gpu=None,
                idle_checks=idle_checks,
                required_idle_checks=required_idle,
                training_seed=next_seed,
                training_pid=child.pid,
                training_return_code=child.returncode,
                stdout=str(stdout_path),
                stderr=str(stderr_path),
                frozen_hashes=frozen_hashes,
            )
            return 6

        verifier_command = build_verifier_command(plan, args.verifier)
        completed = subprocess.run(
            verifier_command,
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        verifier_stdout = run_dir / "confirmation_verifier.stdout.log"
        verifier_stderr = run_dir / "confirmation_verifier.stderr.log"
        verifier_stdout.write_text(completed.stdout, encoding="utf-8")
        verifier_stderr.write_text(completed.stderr, encoding="utf-8")
        verified = verified_seed_snapshot(next_seed, run_dir, plan)
        if completed.returncode != 0 or not verified["ready"]:
            write_status(
                args.status,
                status="confirmation_verification_failed",
                deadline=deadline,
                manifest=manifest,
                seed42=seed42,
                primary=primary,
                confirmations={**confirmations, next_seed: verified},
                gpu=None,
                idle_checks=0,
                required_idle_checks=required_idle,
                verification_seed=next_seed,
                verifier_command=verifier_command,
                verifier_return_code=completed.returncode,
                verifier_stdout=str(verifier_stdout),
                verifier_stderr=str(verifier_stderr),
                frozen_hashes=frozen_hashes,
            )
            return 7
        idle_checks = 0

    seed42 = {"ready": False, "status": "deadline_reached"}
    primary = {"active": False, "reasons": []}
    confirmations = {seed: {"ready": False} for seed in (40, 41)}
    write_status(
        args.status,
        status="timeout_waiting_for_confirmation_chain",
        deadline=deadline,
        manifest=manifest,
        seed42=seed42,
        primary=primary,
        confirmations=confirmations,
        gpu=None,
        idle_checks=idle_checks,
        required_idle_checks=required_idle,
        frozen_hashes=frozen_hashes,
    )
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
