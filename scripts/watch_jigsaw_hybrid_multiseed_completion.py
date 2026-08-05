#!/usr/bin/env python3
"""Retrieve HPC seed41 and build the Jigsaw three-seed Human Gate candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.research_agent_workstation.server.core.gpu_credentials import (
    ALLOWED_GPU_REMOTE_ROOT,
    connect_ssh,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def file_record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def boundary_ok(payload: Mapping[str, Any], *, require_process_signals: bool) -> bool:
    checks = [
        payload.get("private_labels_used") is False,
        payload.get("official_grader_executed") is False,
        payload.get("kaggle_submission_executed") is False,
    ]
    if require_process_signals:
        checks.append(payload.get("process_signals_sent") == 0)
    return all(checks)


def local_seed_snapshot(run_dir: Path, model_seed: int) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    summary_path = run_dir / "summary.json"
    verification_path = run_dir / "independent_verification.json"
    snapshot: dict[str, Any] = {
        "model_seed": model_seed,
        "run_dir": str(run_dir),
        "ready": False,
        "terminal_failure": False,
        "errors": [],
    }
    if not summary_path.is_file() or not verification_path.is_file():
        return snapshot
    try:
        summary = read_json(summary_path)
        verification = read_json(verification_path)
        plan_path = Path(str(summary.get("plan_path", ""))).resolve()
        bundle = summary.get("prediction_bundle") or {}
        bundle_path = Path(str(bundle.get("path", ""))).resolve()
        terminal = summary.get("status") in {
            "promotion_gate_passed",
            "promotion_gate_failed",
        } and verification.get("status") in {
            "promotion_gate_passed",
            "promotion_gate_failed",
        }
        passed = bool(
            summary.get("status") == "promotion_gate_passed"
            and verification.get("status") == "promotion_gate_passed"
            and verification.get("full_contract_valid") is True
        )
        artifacts_ok = bool(
            plan_path.is_file()
            and sha256_file(plan_path) == summary.get("plan_sha256")
            and summary.get("plan_sha256") == verification.get("plan_sha256")
            and bundle_path.is_file()
            and sha256_file(bundle_path) == bundle.get("sha256")
            and boundary_ok(summary, require_process_signals=False)
            and boundary_ok(verification, require_process_signals=(model_seed != 42))
        )
        snapshot.update(
            {
                "run_id": summary.get("run_id"),
                "summary": file_record(summary_path),
                "independent_verification": file_record(verification_path),
                "run_plan": file_record(plan_path) if plan_path.is_file() else None,
                "prediction_bundle": (
                    file_record(bundle_path) if bundle_path.is_file() else None
                ),
                "ready": passed and artifacts_ok,
                "terminal_failure": terminal and not (passed and artifacts_ok),
                "summary_status": summary.get("status"),
                "verification_status": verification.get("status"),
            }
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        snapshot["errors"].append(f"{type(exc).__name__}:{exc}")
    return snapshot


def remote_read_json(sftp: Any, path: str) -> dict[str, Any]:
    with sftp.open(path, "rb") as handle:
        payload = json.loads(handle.read().decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected remote JSON object: {path}")
    return payload


def remote_sha256(client: Any, path: str) -> str:
    command = f"sha256sum -- {shlex.quote(path)}"
    _stdin, stdout, stderr = client.exec_command(command, timeout=600)
    output = stdout.read().decode("utf-8", errors="replace").strip()
    error = stderr.read().decode("utf-8", errors="replace").strip()
    if stdout.channel.recv_exit_status() != 0 or not output:
        raise RuntimeError(f"Remote sha256sum failed: {error}")
    return output.split()[0].lower()


def remote_seed_snapshot(client: Any, remote_base: str, run_id: str) -> dict[str, Any]:
    run_dir = f"{remote_base}/runs/{run_id}"
    status_path = f"{remote_base}/full_seed41_status.json"
    summary_path = f"{run_dir}/summary.json"
    verification_path = f"{run_dir}/independent_verification.json"
    snapshot: dict[str, Any] = {
        "model_seed": 41,
        "run_dir": run_dir,
        "ready": False,
        "terminal_failure": False,
        "errors": [],
    }
    sftp = client.open_sftp()
    try:
        try:
            status = remote_read_json(sftp, status_path)
        except OSError:
            return snapshot
        snapshot["wrapper_status"] = status.get("status")
        if status.get("status") not in {
            "verification_passed",
            "verification_complete_gate_failed",
            "verification_contract_failed",
            "training_failed",
            "verifier_failed",
        }:
            return snapshot
        summary = remote_read_json(sftp, summary_path)
        verification = remote_read_json(sftp, verification_path)
        plan_path = str(summary.get("plan_path", ""))
        bundle = summary.get("prediction_bundle") or {}
        bundle_path = str(bundle.get("path", ""))
        passed = bool(
            status.get("status") == "verification_passed"
            and summary.get("status") == "promotion_gate_passed"
            and verification.get("status") == "promotion_gate_passed"
            and verification.get("full_contract_valid") is True
            and boundary_ok(status, require_process_signals=True)
            and boundary_ok(summary, require_process_signals=False)
            and boundary_ok(verification, require_process_signals=True)
        )
        records = {}
        for name, path, expected in (
            ("run_plan", plan_path, summary.get("plan_sha256")),
            ("summary", summary_path, None),
            ("independent_verification", verification_path, None),
            ("prediction_bundle", bundle_path, bundle.get("sha256")),
            ("wrapper_status_record", status_path, None),
        ):
            stat = sftp.stat(path)
            digest = remote_sha256(client, path)
            if expected is not None and digest != expected:
                raise ValueError(f"Remote {name} hash differs from its source record")
            records[name] = {"path": path, "bytes": int(stat.st_size), "sha256": digest}
        snapshot.update(
            {
                "run_id": summary.get("run_id"),
                **records,
                "ready": passed,
                "terminal_failure": not passed,
                "summary_status": summary.get("status"),
                "verification_status": verification.get("status"),
            }
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        snapshot["errors"].append(f"{type(exc).__name__}:{exc}")
    finally:
        sftp.close()
    return snapshot


def source_stable_key(snapshots: Sequence[Mapping[str, Any]]) -> tuple[Any, ...] | None:
    if not all(snapshot.get("ready") for snapshot in snapshots):
        return None
    values: list[Any] = []
    for snapshot in snapshots:
        values.append(snapshot.get("run_id"))
        for name in (
            "run_plan",
            "summary",
            "independent_verification",
            "prediction_bundle",
        ):
            record = snapshot.get(name) or {}
            values.extend((record.get("bytes"), record.get("sha256")))
    return tuple(values)


def download_remote_file(
    sftp: Any,
    remote: Mapping[str, Any],
    local_path: Path,
    *,
    progress_path: Path,
) -> dict[str, Any]:
    local_path = Path(local_path).resolve()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(remote["bytes"])
    expected_sha = str(remote["sha256"])
    if local_path.is_file() and local_path.stat().st_size == expected_size:
        if sha256_file(local_path) == expected_sha:
            return file_record(local_path)
    partial = local_path.with_suffix(local_path.suffix + ".part")
    offset = partial.stat().st_size if partial.is_file() else 0
    if offset > expected_size:
        raise RuntimeError(f"Oversized local partial download: {partial}")
    started = time.monotonic()
    with sftp.open(str(remote["path"]), "rb") as source:
        source.seek(offset)
        with partial.open("ab") as target:
            transferred = offset
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                target.write(block)
                transferred += len(block)
                elapsed = max(time.monotonic() - started, 1e-6)
                write_json_atomic(
                    progress_path,
                    {
                        "schema": "evomind.jigsaw.remote_download_progress.v1",
                        "created_at": now_iso(),
                        "status": "downloading",
                        "remote_path": remote["path"],
                        "local_path": str(local_path),
                        "transferred_bytes": transferred,
                        "total_bytes": expected_size,
                        "progress_fraction": transferred / expected_size,
                        "throughput_mib_s": (transferred - offset) / elapsed / 1024**2,
                        "process_signals_sent": 0,
                    },
                )
    if partial.stat().st_size != expected_size:
        raise RuntimeError(f"Incomplete remote download: {remote['path']}")
    os.replace(partial, local_path)
    if sha256_file(local_path) != expected_sha:
        raise RuntimeError(f"Downloaded hash mismatch: {local_path}")
    return file_record(local_path)


def materialize_remote_seed(
    client: Any,
    snapshot: Mapping[str, Any],
    download_dir: Path,
    progress_path: Path,
) -> dict[str, Any]:
    download_dir = Path(download_dir).resolve()
    names = {
        "run_plan": "hpc_seed41_frozen_plan.json",
        "summary": "summary.json",
        "independent_verification": "independent_verification.json",
        "prediction_bundle": "jigsaw_transformer_oof_and_test.npz",
        "wrapper_status_record": "hpc_wrapper_status.json",
    }
    local_records: dict[str, Any] = {}
    sftp = client.open_sftp()
    try:
        for name, filename in names.items():
            local_records[name] = download_remote_file(
                sftp,
                snapshot[name],
                download_dir / filename,
                progress_path=progress_path,
            )
    finally:
        sftp.close()
    return {
        "model_seed": 41,
        "run_id": snapshot["run_id"],
        "run_plan": local_records["run_plan"],
        "summary": local_records["summary"],
        "independent_verification": local_records["independent_verification"],
        "prediction_bundle": local_records["prediction_bundle"],
        "wrapper_status_record": local_records["wrapper_status_record"],
    }


def build_aggregation_plan(
    *,
    control_plan_path: Path,
    control_plan: Mapping[str, Any],
    seed40: Mapping[str, Any],
    seed41: Mapping[str, Any],
    seed42: Mapping[str, Any],
    source_evidence_path: Path,
    aggregation_plan_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    aggregator = PROJECT_ROOT / "scripts/aggregate_jigsaw_multiseed_candidate.py"
    verifier = PROJECT_ROOT / "scripts/verify_jigsaw_multiseed_candidate.py"
    watcher = Path(__file__).resolve()
    public = control_plan["inputs"]
    seed_runs = []
    for snapshot in (seed40, seed41, seed42):
        seed_runs.append(
            {
                "model_seed": int(snapshot["model_seed"]),
                "run_id": snapshot["run_id"],
                "run_plan": dict(snapshot["run_plan"]),
                "summary": dict(snapshot["summary"]),
                "independent_verification": dict(
                    snapshot["independent_verification"]
                ),
                "prediction_bundle": dict(snapshot["prediction_bundle"]),
                **(
                    {"boundary_audit": dict(snapshot["boundary_audit"])}
                    if snapshot.get("boundary_audit")
                    else {}
                ),
            }
        )
    output_dir = Path(output_dir).resolve()
    plan = {
        "schema": "evomind.jigsaw.multiseed_confirmation_plan.v1",
        "created_at": now_iso(),
        "status": "frozen_before_aggregation",
        "competition_id": "jigsaw-toxic-comment-classification-challenge",
        "objective": (
            "Probability-average the independently verified public-OOF candidates "
            "for model seeds 40, 41, and 42."
        ),
        "driver": {
            "model": "gpt-5.6-sol",
            "control_plan": file_record(control_plan_path),
            "execution_topology": "local_seed40+hpc_seed41+local_seed42",
        },
        "implementation": {
            "aggregator": file_record(aggregator),
            "verifier": file_record(verifier),
            "watcher": file_record(watcher),
        },
        "public_inputs": {
            "train": {
                "path": public["public_train"]["path"],
                "sha256": public["public_train"]["sha256"],
            },
            "test": {
                "path": public["public_test"]["path"],
                "sha256": public["public_test"]["sha256"],
            },
            "sample_submission": {
                "path": public["public_sample"]["path"],
                "sha256": public["public_sample"]["sha256"],
            },
        },
        "seed_runs": seed_runs,
        "queue_status": file_record(source_evidence_path),
        "aggregation": {
            "method": "arithmetic_mean_probability",
            "model_seeds": [40, 41, 42],
            "numeric_bundle_arrays_only": True,
            "legacy_object_ids_ignored": True,
            "public_csv_id_order_reconstructed": True,
        },
        "confirmation_gate": dict(control_plan["confirmation_gate"]),
        "output": {
            "directory": str(output_dir),
            "result": str(output_dir / "jigsaw_multiseed_confirmation_result.json"),
            "submission_disposition": "withheld_human_gate",
        },
        "boundaries": {
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "process_signals_sent": 0,
            "human_gate_preserved": True,
        },
        "claim_boundary": (
            "Public OOF confirmation only; no official score or medal is claimed."
        ),
    }
    write_json_atomic(aggregation_plan_path.resolve(), plan)
    return plan


def cpu_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
            "PYTHONPATH": os.pathsep.join(
                [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]
            ),
        }
    )
    return environment


def run_command(command: list[str], *, stdout_path: Path, stderr_path: Path) -> int:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=cpu_environment(),
        capture_output=True,
        text=True,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    return completed.returncode


def package_human_gate(
    *,
    plan_path: Path,
    result_path: Path,
    verification_path: Path,
    source_evidence_path: Path,
    package_dir: Path,
) -> dict[str, Any]:
    package_dir = Path(package_dir).resolve()
    package_dir.mkdir(parents=True, exist_ok=True)
    result = read_json(result_path)
    sources = {
        "frozen_plan.json": plan_path,
        "jigsaw_multiseed_confirmation_result.json": result_path,
        "independent_verification.json": verification_path,
        "source_completion_evidence.json": source_evidence_path,
        "candidate_submission_withheld.csv": Path(
            result["submission_withheld"]["path"]
        ),
    }
    for name, source in sources.items():
        shutil.copy2(source, package_dir / name)
    readme = package_dir / "README.md"
    readme.write_text(
        "# Jigsaw three-seed Human Gate candidate\n\n"
        "Public OOF probability-mean confirmation only. No official grader or "
        "Kaggle submission has been executed. Human review and explicit approval "
        "remain required.\n",
        encoding="utf-8",
    )
    files = [file_record(path) for path in sorted(package_dir.iterdir()) if path.is_file()]
    manifest = {
        "schema": "evomind.human_gate.candidate_package.v1",
        "created_at": now_iso(),
        "status": "ready_for_human_review_not_submitted",
        "competition_id": "jigsaw-toxic-comment-classification-challenge",
        "public_oof_metrics": result["metrics"],
        "candidate_ready_for_human_gate": True,
        "files": files,
        "automatic_submission": False,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
        "claim_boundary": (
            "Ready for Human Gate review only; official medal count remains unchanged."
        ),
    }
    write_json_atomic(package_dir / "manifest.json", manifest)
    verification = {
        "schema": "evomind.human_gate.package_verification.v1",
        "created_at": now_iso(),
        "status": "verified",
        "files": [
            file_record(path)
            for path in sorted(package_dir.iterdir())
            if path.is_file() and path.name != "package_verification.json"
        ],
        "candidate_csv_present": (package_dir / "candidate_submission_withheld.csv").is_file(),
        "independent_verification_passed": read_json(
            package_dir / "independent_verification.json"
        ).get("ok")
        is True,
        "automatic_submission": False,
    }
    write_json_atomic(package_dir / "package_verification.json", verification)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-plan", type=Path, required=True)
    parser.add_argument("--local-seed40-run", type=Path, required=True)
    parser.add_argument("--local-seed42-run", type=Path, required=True)
    parser.add_argument("--seed42-boundary-audit", type=Path, required=True)
    parser.add_argument("--remote-base", required=True)
    parser.add_argument("--remote-run-id", required=True)
    parser.add_argument("--remote-download-dir", type=Path, required=True)
    parser.add_argument("--source-evidence", type=Path, required=True)
    parser.add_argument("--aggregation-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verification-output", type=Path, required=True)
    parser.add_argument("--human-gate-dir", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--stable-checks", type=int, default=2)
    parser.add_argument("--deadline-hours", type=float, default=240.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.poll_seconds < 10 or args.stable_checks < 2 or args.deadline_hours <= 0:
        raise ValueError("Watcher timing contract is invalid")
    python_path = args.python.resolve()
    if not python_path.is_file():
        raise FileNotFoundError(python_path)
    remote_base = args.remote_base.rstrip("/")
    if not remote_base.startswith(ALLOWED_GPU_REMOTE_ROOT.rstrip("/") + "/"):
        raise ValueError("Remote watcher base escaped the dedicated HPC root")
    control_plan_path = args.control_plan.resolve()
    control_plan = read_json(control_plan_path)
    if control_plan.get("schema") != "evomind.jigsaw.confirmation_queue_frozen_plan.v1":
        raise ValueError("Unexpected Jigsaw confirmation control-plan schema")
    control_sha = sha256_file(control_plan_path)
    seed42_audit_path = args.seed42_boundary_audit.resolve()
    seed42_audit_sha = sha256_file(seed42_audit_path)
    deadline = datetime.now().astimezone() + timedelta(hours=args.deadline_hours)
    previous_key: tuple[Any, ...] | None = None
    stable_observations = 0
    remote_progress = args.remote_download_dir.resolve() / "download_progress.json"

    while datetime.now().astimezone() < deadline:
        if sha256_file(control_plan_path) != control_sha:
            raise RuntimeError("Frozen control plan changed while watcher ran")
        if sha256_file(seed42_audit_path) != seed42_audit_sha:
            raise RuntimeError("Seed42 boundary audit changed while watcher ran")
        seed40 = local_seed_snapshot(args.local_seed40_run, 40)
        seed42 = local_seed_snapshot(args.local_seed42_run, 42)
        seed42["boundary_audit"] = file_record(seed42_audit_path)
        remote_error: str | None = None
        try:
            client = connect_ssh(timeout=30)
            try:
                seed41 = remote_seed_snapshot(client, remote_base, args.remote_run_id)
            finally:
                client.close()
        except Exception as exc:  # network state is recorded and retried
            remote_error = f"{type(exc).__name__}:{exc}"
            seed41 = {
                "model_seed": 41,
                "run_dir": f"{remote_base}/runs/{args.remote_run_id}",
                "ready": False,
                "terminal_failure": False,
                "errors": [remote_error],
            }
        snapshots = [seed40, seed41, seed42]
        key = source_stable_key(snapshots)
        stable_observations = (
            stable_observations + 1
            if key is not None and key == previous_key
            else 1
            if key is not None
            else 0
        )
        previous_key = key
        terminal_failure = next(
            (snapshot for snapshot in snapshots if snapshot.get("terminal_failure")),
            None,
        )
        status_payload = {
            "schema": "evomind.jigsaw.hybrid_multiseed_completion_watcher.v1",
            "created_at": now_iso(),
            "status": (
                "source_seed_failed"
                if terminal_failure
                else "waiting_for_stable_seed_sources"
                if key is not None
                else "waiting_for_seed_completion"
            ),
            "watcher_pid": os.getpid(),
            "deadline": deadline.isoformat(),
            "control_plan_path": str(control_plan_path),
            "control_plan_sha256": control_sha,
            "sources": snapshots,
            "stable_observations": stable_observations,
            "required_stable_observations": args.stable_checks,
            "remote_error": remote_error,
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "process_signals_sent": 0,
            "human_gate_preserved": True,
        }
        write_json_atomic(args.status.resolve(), status_payload)
        if terminal_failure:
            return 3
        if stable_observations < args.stable_checks:
            time.sleep(args.poll_seconds)
            continue

        client = connect_ssh(timeout=30)
        try:
            seed41_local = materialize_remote_seed(
                client,
                seed41,
                args.remote_download_dir,
                remote_progress,
            )
        finally:
            client.close()
        seed40_record = {
            name: seed40[name]
            for name in (
                "model_seed",
                "run_id",
                "run_plan",
                "summary",
                "independent_verification",
                "prediction_bundle",
            )
        }
        seed42_record = {
            name: seed42[name]
            for name in (
                "model_seed",
                "run_id",
                "run_plan",
                "summary",
                "independent_verification",
                "prediction_bundle",
                "boundary_audit",
            )
        }
        source_evidence = {
            **status_payload,
            "created_at": now_iso(),
            "status": "all_three_seed_sources_stable_and_materialized",
            "sources": [seed40_record, seed41_local, seed42_record],
        }
        write_json_atomic(args.source_evidence.resolve(), source_evidence)
        plan = build_aggregation_plan(
            control_plan_path=control_plan_path,
            control_plan=control_plan,
            seed40=seed40_record,
            seed41=seed41_local,
            seed42=seed42_record,
            source_evidence_path=args.source_evidence.resolve(),
            aggregation_plan_path=args.aggregation_plan.resolve(),
            output_dir=args.output_dir.resolve(),
        )
        aggregator = Path(plan["implementation"]["aggregator"]["path"])
        verifier = Path(plan["implementation"]["verifier"]["path"])
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        aggregate_exit = run_command(
            [str(python_path), str(aggregator), "--plan", str(args.aggregation_plan.resolve())],
            stdout_path=output_dir / "aggregation.stdout.log",
            stderr_path=output_dir / "aggregation.stderr.log",
        )
        result_path = Path(plan["output"]["result"])
        if aggregate_exit not in (0, 3) or not result_path.is_file():
            write_json_atomic(
                args.status.resolve(),
                {**status_payload, "created_at": now_iso(), "status": "aggregation_failed", "aggregation_exit_code": aggregate_exit},
            )
            return 3
        verifier_exit = run_command(
            [
                str(python_path),
                str(verifier),
                "--plan",
                str(args.aggregation_plan.resolve()),
                "--result",
                str(result_path),
                "--output",
                str(args.verification_output.resolve()),
            ],
            stdout_path=output_dir / "verification.stdout.log",
            stderr_path=output_dir / "verification.stderr.log",
        )
        verification = read_json(args.verification_output.resolve())
        if verifier_exit != 0 or verification.get("ok") is not True:
            write_json_atomic(
                args.status.resolve(),
                {**status_payload, "created_at": now_iso(), "status": "independent_verification_failed", "verification_exit_code": verifier_exit, "verification": verification},
            )
            return 3
        package = package_human_gate(
            plan_path=args.aggregation_plan.resolve(),
            result_path=result_path,
            verification_path=args.verification_output.resolve(),
            source_evidence_path=args.source_evidence.resolve(),
            package_dir=args.human_gate_dir.resolve(),
        )
        final = {
            **status_payload,
            "created_at": now_iso(),
            "status": "confirmation_passed_human_gate_pending",
            "aggregation_plan": file_record(args.aggregation_plan.resolve()),
            "result": file_record(result_path),
            "independent_verification": file_record(args.verification_output.resolve()),
            "human_gate_package": {
                "path": str(args.human_gate_dir.resolve()),
                "manifest": file_record(args.human_gate_dir.resolve() / "manifest.json"),
                "status": package["status"],
            },
            "candidate_ready_for_human_gate": True,
        }
        write_json_atomic(args.status.resolve(), final)
        return 0

    write_json_atomic(
        args.status.resolve(),
        {
            "schema": "evomind.jigsaw.hybrid_multiseed_completion_watcher.v1",
            "created_at": now_iso(),
            "status": "timeout_waiting_for_seed_completion",
            "deadline": deadline.isoformat(),
            "process_signals_sent": 0,
        },
    )
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
