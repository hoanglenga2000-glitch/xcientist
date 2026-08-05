#!/usr/bin/env python3
"""Wait for a local candidate run and invoke its public-data verifier."""
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
PLAN_ROOT = PROJECT_ROOT / "workspace" / "mlebench_plans"
STATUS_ROOT = PROJECT_ROOT / "workspace" / "local_gpu"

PROFILE_SPECS: dict[str, dict[str, Any]] = {
    "leaf": {
        "plan": PLAN_ROOT / "leaf_multibackbone_frozen_plan.json",
        "verifier": PROJECT_ROOT / "scripts" / "verify_leaf_multibackbone_oof_run.py",
        "queue": STATUS_ROOT / "leaf_training_queue_calibrated.json",
        "terminal_file": "leaf_multibackbone_oof.json",
        "terminal_statuses": {"promotion_gate_passed", "promotion_gate_failed"},
        "report_schema": "evomind.leaf.multibackbone_independent_verification.v1",
    },
    "spooky": {
        "plan": PLAN_ROOT / "spooky_transformer_byte_s42_frozen_plan.json",
        "verifier": PROJECT_ROOT / "scripts" / "verify_spooky_transformer_oof_run.py",
        "queue": STATUS_ROOT / "spooky_training_queue.json",
        "terminal_file": "summary.json",
        "terminal_statuses": {
            "single_seed_gate_passed_confirmation_pending",
            "single_seed_gate_failed",
        },
        "report_schema": "evomind.spooky.independent_verification.v1",
    },
    "may2022": {
        "plan": PLAN_ROOT / "may2022_compact_embedding_s42_frozen_plan.json",
        "verifier": PROJECT_ROOT / "scripts" / "verify_may2022_compact_embedding_run.py",
        "queue": STATUS_ROOT / "may2022_training_queue_calibrated.json",
        "terminal_file": "summary.json",
        "terminal_statuses": {"single_seed_gate_passed", "single_seed_gate_failed"},
        "report_schema": "evomind.mlebench.may2022_compact_independent_verification.v1",
    },
}


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def invariant_fields() -> dict[str, Any]:
    return {
        "process_signals_sent": 0,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }


def frozen_provenance_fields(spec: dict[str, Any]) -> dict[str, Any]:
    """Keep dependency identity fields in every watcher state transition."""
    return {
        "plan_path": str(spec["plan_path"]),
        "plan_sha256": spec["plan_sha256"],
        "verifier_path": str(spec["verifier"]),
        "verifier_sha256": spec["verifier_sha256"],
    }


def resolve_profile(profile: str) -> dict[str, Any]:
    if profile not in PROFILE_SPECS:
        raise ValueError(f"Unknown local verification profile: {profile}")
    spec = dict(PROFILE_SPECS[profile])
    plan_path = Path(spec["plan"]).resolve()
    verifier = Path(spec["verifier"]).resolve()
    queue = Path(spec["queue"]).resolve()
    for path in (plan_path, verifier):
        if not path.is_file():
            raise FileNotFoundError(path)
    plan = read_json(plan_path)
    if profile == "leaf":
        training = plan["training"]
        run_dir = Path(training["output_root"]).resolve() / "runs" / training["run_id"]
        python_path = Path(training["python"]).resolve()
        public_dir = None
        expected_run_id = training["run_id"]
    elif profile == "spooky":
        execution = plan["execution"]
        run_dir = Path(execution["output_root"]).resolve() / execution["run_id"]
        python_path = Path(execution["python"]).resolve()
        public_dir = Path(execution["public_dir"]).resolve()
        expected_run_id = execution["run_id"]
    else:
        execution = plan["execution"]
        run_dir = Path(execution["output_root"]).resolve() / "runs" / execution["run_id"]
        python_path = Path(execution["python"]).resolve()
        public_dir = Path(execution["public_dir"]).resolve()
        expected_run_id = execution["run_id"]
    if not python_path.is_file():
        raise FileNotFoundError(python_path)
    spec.update(
        {
            "profile": profile,
            "plan": plan,
            "plan_path": plan_path,
            "plan_sha256": sha256_file(plan_path),
            "verifier": verifier,
            "verifier_sha256": sha256_file(verifier),
            "queue": queue,
            "python": python_path,
            "run_dir": run_dir,
            "public_dir": public_dir,
            "run_id": expected_run_id,
            "terminal_path": run_dir / spec["terminal_file"],
            "output": run_dir / "independent_verification.json",
        }
    )
    return spec


def terminal_snapshot(spec: dict[str, Any]) -> dict[str, Any]:
    terminal_path = Path(spec["terminal_path"])
    queue_path = Path(spec["queue"])
    terminal: dict[str, Any] | None = None
    queue: dict[str, Any] | None = None
    errors: list[str] = []
    try:
        if terminal_path.is_file():
            terminal = read_json(terminal_path)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        errors.append(f"terminal:{type(exc).__name__}:{exc}")
    try:
        if queue_path.is_file():
            queue = read_json(queue_path)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        errors.append(f"queue:{type(exc).__name__}:{exc}")
    status = terminal.get("status") if terminal else None
    run_id = terminal.get("run_id") if terminal else None
    boundary_ok = bool(
        terminal
        and terminal.get("private_labels_used") is False
        and terminal.get("official_grader_executed") is False
        and terminal.get("kaggle_submission_executed") is False
        and terminal.get("process_signals_sent", 0) == 0
        and queue
        and queue.get("process_signals_sent") == 0
        and queue.get("official_grader_executed") is False
        and queue.get("kaggle_submission_executed") is False
    )
    signature = None
    if terminal_path.is_file():
        stat = terminal_path.stat()
        signature = {
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(terminal_path),
        }
    return {
        "ready": bool(
            terminal
            and status in spec["terminal_statuses"]
            and run_id == spec["run_id"]
            and boundary_ok
            and not errors
        ),
        "terminal_path": str(terminal_path),
        "terminal_present": terminal_path.is_file(),
        "terminal_status": status,
        "terminal_run_id": run_id,
        "terminal_signature": signature,
        "queue_path": str(queue_path),
        "queue_present": queue_path.is_file(),
        "boundary_ok": boundary_ok,
        "errors": errors,
    }


def build_verifier_command(spec: dict[str, Any]) -> list[str]:
    command = [
        str(spec["python"]),
        str(spec["verifier"]),
        "--run-dir",
        str(spec["run_dir"]),
        "--plan",
        str(spec["plan_path"]),
        "--output",
        str(spec["output"]),
    ]
    if spec["profile"] == "spooky":
        command.extend(["--public-dir", str(spec["public_dir"])])
    return command


def report_passed(profile: str, report: dict[str, Any]) -> bool:
    if profile == "spooky":
        return report.get("status") == "passed"
    return report.get("ok") is True and report.get("status") == "verification_passed"


def candidate_ready(profile: str, terminal_status: str | None, report: dict[str, Any]) -> bool:
    if profile == "leaf":
        return report.get("candidate_ready") is True
    if profile == "spooky":
        return terminal_status == "single_seed_gate_passed_confirmation_pending"
    gate = report.get("gate") or {}
    return gate.get("passed") is True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILE_SPECS), required=True)
    parser.add_argument("--status", type=Path)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--stable-checks", type=int, default=2)
    parser.add_argument("--deadline-hours", type=float, default=240.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.poll_seconds < 10 or args.stable_checks < 2 or args.deadline_hours <= 0:
        raise ValueError("Local candidate watcher timing contract is invalid")
    spec = resolve_profile(args.profile)
    status_path = (
        args.status.resolve()
        if args.status
        else STATUS_ROOT / f"{args.profile}_independent_verification_watcher.json"
    )
    deadline = datetime.now().astimezone() + timedelta(hours=args.deadline_hours)
    previous_signature: dict[str, Any] | None = None
    stable_observations = 0

    while datetime.now().astimezone() < deadline:
        if sha256_file(spec["plan_path"]) != spec["plan_sha256"]:
            raise RuntimeError(f"{args.profile} frozen plan changed while watcher ran")
        if sha256_file(spec["verifier"]) != spec["verifier_sha256"]:
            raise RuntimeError(f"{args.profile} verifier changed while watcher ran")
        snapshot = terminal_snapshot(spec)
        signature = snapshot["terminal_signature"] if snapshot["ready"] else None
        stable_observations = (
            stable_observations + 1
            if signature is not None and signature == previous_signature
            else 1
            if signature is not None
            else 0
        )
        previous_signature = signature
        write_json_atomic(
            status_path,
            {
                "schema": "evomind.local_candidate_verification_watcher.v1",
                "created_at": now_iso(),
                "status": (
                    "waiting_for_stable_terminal_artifact"
                    if snapshot["ready"]
                    else "waiting_for_training_completion"
                ),
                "profile": args.profile,
                "watcher_pid": os.getpid(),
                "deadline": deadline.isoformat(),
                "run_id": spec["run_id"],
                **frozen_provenance_fields(spec),
                "completion": snapshot,
                "stable_observations": stable_observations,
                "required_stable_observations": args.stable_checks,
                **invariant_fields(),
            },
        )
        if stable_observations < args.stable_checks:
            time.sleep(args.poll_seconds)
            continue

        command = build_verifier_command(spec)
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
        spec["run_dir"].mkdir(parents=True, exist_ok=True)
        stdout = spec["run_dir"] / "independent_verification_watcher.stdout.log"
        stderr = spec["run_dir"] / "independent_verification_watcher.stderr.log"
        stdout.write_text(completed.stdout, encoding="utf-8")
        stderr.write_text(completed.stderr, encoding="utf-8")
        try:
            report = read_json(spec["output"])
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            report = {"status": "verification_invocation_failed", "error": str(exc)}
        passed = bool(
            report.get("schema") == spec["report_schema"]
            and report_passed(args.profile, report)
            and report.get("process_signals_sent") == 0
            and report.get("private_labels_used") is False
            and report.get("official_grader_executed") is False
            and report.get("kaggle_submission_executed") is False
        )
        write_json_atomic(
            status_path,
            {
                "schema": "evomind.local_candidate_verification_watcher.v1",
                "created_at": now_iso(),
                "status": "verification_passed" if passed else "verification_failed",
                "profile": args.profile,
                "watcher_pid": os.getpid(),
                "run_id": spec["run_id"],
                **frozen_provenance_fields(spec),
                "verifier_command": command,
                "verifier_exit_code": completed.returncode,
                "report_path": str(spec["output"]),
                "report_sha256": sha256_file(spec["output"]) if spec["output"].is_file() else None,
                "candidate_ready": candidate_ready(
                    args.profile, snapshot["terminal_status"], report
                )
                if passed
                else False,
                "terminal_path": snapshot["terminal_path"],
                "terminal_sha256": (snapshot["terminal_signature"] or {}).get("sha256"),
                "stdout": str(stdout),
                "stderr": str(stderr),
                **invariant_fields(),
            },
        )
        return 0 if passed else 3

    write_json_atomic(
        status_path,
        {
            "schema": "evomind.local_candidate_verification_watcher.v1",
            "created_at": now_iso(),
            "status": "timeout_waiting_for_training_completion",
            "profile": args.profile,
            "watcher_pid": os.getpid(),
            "run_id": spec["run_id"],
            "deadline": deadline.isoformat(),
            **frozen_provenance_fields(spec),
            **invariant_fields(),
        },
    )
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
