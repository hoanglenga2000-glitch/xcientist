#!/usr/bin/env python3
"""Wait for the frozen Cactus GPU candidate, then evaluate the CPU sidecar once."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import shlex
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import mlebench_remote_ops as ops
from scripts.deploy_job89941_may2022_cpu_diagnostic import (
    sftp_mkdirs_confined,
    sftp_upload_exact,
)

DEFAULT_PLAN = PROJECT_ROOT / "workspace/mlebench_plans/cactus_cpu_sidecar_frozen_plan_v1_20260728.json"
DEFAULT_EVIDENCE = PROJECT_ROOT / "workspace/hpc/cactus_cpu_sidecar_watcher"
TERMINAL = frozenset({"verification_passed", "verification_complete_gate_failed"})


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_plan(path: Path) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != "evomind.cactus.cpu_sidecar_frozen_plan.v1" or plan.get("status") != "frozen":
        raise RuntimeError("Cactus sidecar plan schema/status drifted")
    for key in ("evaluator", "cpu_collection"):
        record = plan[key]
        local = Path(record["path"]).resolve()
        if not local.is_file() or local.stat().st_size != record["bytes"] or sha256_file(local) != record["sha256"]:
            raise RuntimeError(f"Cactus sidecar {key} drifted")
    for key in ("remote_status", "gpu_bundle", "cpu_bundle", "remote_output"):
        value = ops.ensure_remote_path(plan[key])
        if not value.startswith(ops.ALLOWED_GPU_REMOTE_ROOT + "/"):
            raise RuntimeError(f"Cactus sidecar {key} escaped dedicated root")
    contract = plan.get("contracts") or {}
    if not (
        contract.get("private_labels_used") is False
        and contract.get("official_grader_executed") is False
        and contract.get("kaggle_submission_executed") is False
        and contract.get("process_signals_allowed") is False
    ):
        raise RuntimeError("Cactus sidecar execution boundary drifted")
    plan["_path"] = str(path.resolve())
    plan["_sha256"] = sha256_file(path)
    return plan


def remote_json(client, path: str) -> dict | None:
    source = (
        "import json,pathlib; p=pathlib.Path(" + repr(path) + "); "
        "print(json.dumps(json.loads(p.read_text()) if p.is_file() else None,sort_keys=True))"
    )
    command = f"python3 -c {shlex.quote(source)}"
    code, stdout, stderr = ops._run_remote(client, command, timeout=120)
    if code:
        raise RuntimeError(f"Cactus sidecar remote read failed: {stderr[-500:]}")
    value = json.loads(stdout.strip().splitlines()[-1])
    return value if isinstance(value, dict) else None


def base(plan: dict) -> dict:
    return {
        "schema": "evomind.cactus.cpu_sidecar_watcher.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "plan_path": plan["_path"],
        "plan_sha256": plan["_sha256"],
        "process_signals_sent": 0,
        "other_processes_modified": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }


def run_once(plan: dict, evidence: Path) -> dict:
    status_path = evidence / "status_current.json"
    client = ops._connect()
    try:
        dependency = remote_json(client, plan["remote_status"])
        dependency_status = (dependency or {}).get("status")
        if dependency_status not in TERMINAL:
            result = {**base(plan), "status": "waiting_for_cactus_gpu_terminal", "dependency": dependency}
            write_json(status_path, result)
            return result
        remote_report = posixpath.join(plan["remote_output"], "sidecar_report.json")
        existing = remote_json(client, remote_report)
        if existing is None:
            evaluator = Path(plan["evaluator"]["path"]).resolve()
            remote_script = posixpath.join(posixpath.dirname(plan["remote_output"]), "evaluate_cactus_cpu_sidecar.py")
            with client.open_sftp() as sftp:
                sftp_mkdirs_confined(sftp, posixpath.dirname(plan["remote_output"]))
                sftp_upload_exact(
                    sftp,
                    evaluator,
                    remote_script,
                    expected_bytes=evaluator.stat().st_size,
                    expected_sha256=sha256_file(evaluator),
                )
            argv = [
                "python3", remote_script,
                "--gpu-bundle", plan["gpu_bundle"],
                "--cpu-bundle", plan["cpu_bundle"],
                "--output-dir", plan["remote_output"],
                "--weights", "0,0.025,0.05,0.1,0.2",
                "--promotion-auc", "0.9997",
            ]
            command = "env CUDA_VISIBLE_DEVICES='' " + " ".join(shlex.quote(value) for value in argv)
            code, stdout, stderr = ops._run_remote(client, command, timeout=900)
            if code:
                raise RuntimeError(f"Cactus sidecar evaluation failed: {(stderr or stdout)[-1000:]}")
            existing = remote_json(client, remote_report)
        if existing is None:
            raise RuntimeError("Cactus sidecar report was not materialized")
        collected = evidence / "collected"
        collected.mkdir(parents=True, exist_ok=True)
        with client.open_sftp() as sftp:
            for name in ("sidecar_report.json", "cactus_cpu_sidecar_candidate.npz"):
                sftp.get(posixpath.join(plan["remote_output"], name), str(collected / name))
        result = {
            **base(plan),
            "status": existing.get("status"),
            "candidate_ready": existing.get("candidate_ready"),
            "report": existing,
            "downloads": [
                {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in sorted(collected.iterdir()) if path.is_file()
            ],
        }
        write_json(status_path, result)
        return result
    finally:
        client.close()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--deadline-hours", type=float, default=336.0)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    plan = validate_plan(args.plan.resolve())
    deadline = datetime.now().astimezone() + timedelta(hours=args.deadline_hours)
    while datetime.now().astimezone() < deadline:
        try:
            result = run_once(plan, args.evidence_dir.resolve())
            if result["status"] != "waiting_for_cactus_gpu_terminal":
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 0
        except Exception as exc:
            write_json(
                args.evidence_dir.resolve() / "failure_current.json",
                {**base(plan), "status": "transient_or_contract_failure", "error_type": type(exc).__name__, "error": str(exc)},
            )
        time.sleep(max(30, args.poll_seconds))
    raise RuntimeError("Cactus CPU sidecar watcher deadline exceeded")


if __name__ == "__main__":
    raise SystemExit(main())
