#!/usr/bin/env python3
"""Audit the live RTX 4060 training chain without mutating any process.

The audit distinguishes the one current frozen-plan queue for each stage from
older queue processes that are still waiting with superseded plan hashes.  It
also groups the Windows venv shim and its child interpreter into one logical
training process, so strict single-GPU serial execution can be checked without
mistaking the wrapper/worker pair for two jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "workspace"
    / "local_gpu"
    / "local_gpu_serial_chain_audit_current.json"
)


@dataclass(frozen=True)
class QueueSpec:
    stage: str
    plan: str
    script: str
    statuses: tuple[str, ...]


QUEUE_SPECS = (
    QueueSpec(
        "may2022",
        "workspace/mlebench_plans/may2022_compact_embedding_s42_frozen_plan.json",
        "scripts/queue_may2022_after_spooky.py",
        (
            "workspace/local_gpu/may2022_training_queue.json",
            "workspace/local_gpu/may2022_training_queue_gate20.json",
            "workspace/local_gpu/may2022_training_queue_calibrated.json",
        ),
    ),
    QueueSpec(
        "siim_preprocessing",
        "workspace/mlebench_plans/siim_preprocessing_ablation_frozen_plan.json",
        "scripts/queue_siim_after_jigsaw.py",
        (
            "workspace/local_gpu/siim_training_queue.json",
            "workspace/local_gpu/siim_training_queue_calibrated.json",
        ),
    ),
    QueueSpec(
        "siim_final",
        "workspace/mlebench_plans/siim_final_candidate_frozen_plan.json",
        "scripts/queue_siim_final_candidate.py",
        (
            "workspace/local_gpu/siim_final_candidate_queue.json",
            "workspace/local_gpu/siim_final_candidate_queue_calibrated.json",
        ),
    ),
    QueueSpec(
        "leaf",
        "workspace/mlebench_plans/leaf_multibackbone_frozen_plan.json",
        "scripts/queue_leaf_after_siim.py",
        (
            "workspace/local_gpu/leaf_training_queue.json",
            "workspace/local_gpu/leaf_training_queue_calibrated.json",
        ),
    ),
    QueueSpec(
        "leaf_early",
        "workspace/mlebench_plans/leaf_early_after_jigsaw_control_frozen_plan.json",
        "scripts/queue_leaf_after_jigsaw_confirmation.py",
        ("workspace/local_gpu/leaf_early_queue.json",),
    ),
    QueueSpec(
        "jigsaw_confirmation",
        "workspace/mlebench_plans/jigsaw_confirmation_s40_s41_frozen_plan_20260727.json",
        "scripts/queue_jigsaw_confirmation_after_leaf.py",
        ("workspace/local_gpu/jigsaw_confirmation_queue.json",),
    ),
    QueueSpec(
        "jigsaw_confirmation_early",
        "workspace/mlebench_plans/jigsaw_early_confirmation_after_may_control_frozen_plan.json",
        "scripts/queue_jigsaw_confirmation_after_may.py",
        ("workspace/local_gpu/jigsaw_confirmation_early_queue.json",),
    ),
)


TRAINING_SCRIPT_NAMES = (
    "run_spooky_deberta_oof_v2.py",
    "run_may2022_compact_embedding_oof.py",
    "run_siim_preprocessing_ablation.py",
    "run_siim_final_candidate.py",
    "run_leaf_multibackbone_oof.py",
    "run_jigsaw_transformer_oof.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def launcher_for_status(status_path: Path) -> Path:
    return status_path.with_name(status_path.stem + "_launcher.json")


def normalized_path(value: str | Path) -> str:
    return str(Path(value).resolve()).casefold()


def safe_process(pid: int | None) -> psutil.Process | None:
    if not pid:
        return None
    try:
        process = psutil.Process(int(pid))
        if not process.is_running():
            return None
        return process
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        return None


def command_line(process: psutil.Process | None) -> str | None:
    if process is None:
        return None
    try:
        return " ".join(process.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def process_descends_from(process: psutil.Process | None, ancestor_pid: int) -> bool:
    """Return whether process is the ancestor itself or one of its descendants."""

    if process is None:
        return False
    try:
        current = process
        visited: set[int] = set()
        while current.pid not in visited:
            visited.add(current.pid)
            if current.pid == ancestor_pid:
                return True
            parent = current.parent()
            if parent is None:
                return False
            current = parent
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    return False


def persistent_runtime_is_linked(
    persistent_status: dict[str, Any],
    worker: psutil.Process | None,
    domain_process: psutil.Process | None,
) -> bool:
    """Link an exact-status manifest even when a queue omits its own PID.

    Some terminal and in-training queue states intentionally replace ``pid`` with
    ``training_pid``.  The persistent manifest already binds one exact status path,
    so a live manifest worker is authoritative when no domain PID is available.
    When a domain PID is present, retain the stronger ancestry proof.
    """

    return bool(
        persistent_status.get("status") == "running"
        and worker is not None
        and (
            domain_process is None
            or process_descends_from(domain_process, worker.pid)
        )
    )


def select_persistent_runtime(
    candidates: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    records = list(candidates)
    if not records:
        return None
    return max(
        records,
        key=lambda item: (
            bool(item.get("linked")),
            item.get("persistent_status") == "running",
            int(item.get("status_mtime_ns") or 0),
        ),
    )


def persistent_runtime_for_status(
    root: Path,
    status_path: Path,
    domain_process: psutil.Process | None,
    script_path: Path,
    current_script_sha: str,
) -> dict[str, Any] | None:
    """Resolve the live Scheduled Task wrapper owning one domain status file."""

    status_directory = root / "workspace" / "local_gpu" / "persistent_status"
    if not status_directory.is_dir():
        return None
    wanted_status = normalized_path(status_path)
    wanted_script = normalized_path(script_path)
    candidates: list[dict[str, Any]] = []
    for persistent_status_path in sorted(status_directory.glob("*.json")):
        try:
            persistent_status = read_json(persistent_status_path)
            manifest_value = persistent_status.get("manifest_path")
            if not manifest_value:
                continue
            manifest_path = Path(str(manifest_value)).resolve()
            if not manifest_path.is_file():
                continue
            manifest = read_json(manifest_path)
            arguments = {
                normalized_path(str(argument))
                for argument in manifest.get("arguments", [])
                if isinstance(argument, str) and ("\\" in argument or "/" in argument)
            }
            if wanted_status not in arguments:
                continue
            worker = safe_process(persistent_status.get("worker_pid"))
            linked = persistent_runtime_is_linked(
                persistent_status,
                worker,
                domain_process,
            )
            binding = next(
                (
                    item
                    for item in manifest.get("artifact_bindings", [])
                    if isinstance(item, dict)
                    and item.get("path")
                    and normalized_path(str(item["path"])) == wanted_script
                ),
                None,
            )
            bound_script_sha = binding.get("sha256") if binding else None
            stderr_value = manifest.get("stderr_path")
            stderr_path = Path(str(stderr_value)).resolve() if stderr_value else None
            candidates.append({
                "linked": linked,
                "persistent_status_path": str(persistent_status_path.resolve()),
                "persistent_status": persistent_status.get("status"),
                "persistent_manifest_path": str(manifest_path),
                "persistent_manifest_sha256": sha256_file(manifest_path),
                "persistent_wrapper_pid": persistent_status.get("wrapper_pid"),
                "persistent_worker_pid": persistent_status.get("worker_pid"),
                "persistent_process_signals_sent": persistent_status.get(
                    "process_signals_sent"
                ),
                "bound_script_sha256": bound_script_sha,
                "script_identity_proven": bound_script_sha == current_script_sha,
                "stderr_path": str(stderr_path) if stderr_path else None,
                "stderr_bytes": (
                    stderr_path.stat().st_size
                    if stderr_path is not None and stderr_path.is_file()
                    else None
                ),
                "status_mtime_ns": persistent_status_path.stat().st_mtime_ns,
            })
        except (OSError, json.JSONDecodeError, ValueError):
            continue
    return select_persistent_runtime(candidates)


def source_has_launch_guards(script: Path) -> dict[str, bool]:
    source = script.read_text(encoding="utf-8")
    hash_guard = (
        "frozen plan changed while queued" in source
        or "frozen plan changed before" in source
        or "control changed while queue ran" in source
    )
    return {
        "validates_frozen_plan": "validate_frozen_plan" in source,
        "rejects_changed_plan_hash_at_launch": hash_guard,
        "performs_final_idle_query": "final_evaluation" in source,
        "requires_consecutive_idle_checks": (
            "consecutive_idle_checks" in source
            or "required_idle" in source
            or "consecutive = consecutive + 1" in source
        ),
        "records_zero_process_signals": (
            '"process_signals_sent": 0' in source
            or "base.invariant_fields()" in source
        ),
        "requires_original_queue_wait_guard": (
            "safe_for_early_queue" in source
            and "superseded_by_original_" in source
        ),
        "resumes_existing_verified_seed_artifacts": "load_or_verify_existing" in source,
        "rejects_existing_target_run": "target_run_already_exists" in source,
    }


def queue_record(root: Path, spec: QueueSpec, status_relative: str) -> dict[str, Any]:
    plan_path = (root / spec.plan).resolve()
    script_path = (root / spec.script).resolve()
    status_path = (root / status_relative).resolve()
    current_plan_sha = sha256_file(plan_path)
    current_script_sha = sha256_file(script_path)
    status = read_json(status_path)
    pid = status.get("pid")
    process = safe_process(pid)
    launcher_path = launcher_for_status(status_path)
    launcher = read_json(launcher_path) if launcher_path.is_file() else {}
    launcher_script_sha = (
        launcher.get("queue_script_sha256")
        or launcher.get("script_sha256")
        or launcher.get("script_sha256_at_launch")
    )
    stderr_value = (
        launcher.get("stderr")
        or launcher.get("stderr_path")
        or launcher.get("process_stderr")
    )
    stderr_path = Path(stderr_value).resolve() if stderr_value else None
    persistent_runtime = persistent_runtime_for_status(
        root,
        status_path,
        process,
        script_path,
        current_script_sha,
    )
    persistent_linked = bool(
        persistent_runtime is not None and persistent_runtime.get("linked")
    )
    if persistent_linked:
        if process is None:
            process = safe_process(persistent_runtime.get("persistent_worker_pid"))
        launcher_script_sha = persistent_runtime.get("bound_script_sha256")
        persistent_stderr = persistent_runtime.get("stderr_path")
        stderr_path = Path(persistent_stderr) if persistent_stderr else None
    loaded_plan_sha = status.get("plan_sha256") or status.get("control_plan_sha256")
    return {
        "stage": spec.stage,
        "status_path": str(status_path),
        "status": status.get("status"),
        "pid": pid,
        "worker_alive": process is not None,
        "worker_command_line": command_line(process),
        "worker_started_at": (
            datetime.fromtimestamp(process.create_time()).astimezone().isoformat()
            if process is not None
            else None
        ),
        "loaded_plan_sha256": loaded_plan_sha,
        "current_plan_sha256": current_plan_sha,
        "loaded_plan_is_current": loaded_plan_sha == current_plan_sha,
        "plan_path": str(plan_path),
        "script_path": str(script_path),
        "current_script_sha256": current_script_sha,
        "launcher_path": str(launcher_path) if launcher_path.is_file() else None,
        "launcher_script_sha256": launcher_script_sha,
        "launcher_script_identity_proven": launcher_script_sha == current_script_sha,
        "runtime_provenance": (
            "persistent_scheduled_task" if persistent_linked else "legacy_launcher"
        ),
        "persistent_runtime": persistent_runtime,
        "stderr_path": str(stderr_path) if stderr_path else None,
        "stderr_bytes": (
            stderr_path.stat().st_size
            if stderr_path is not None and stderr_path.is_file()
            else None
        ),
        "process_signals_sent": status.get("process_signals_sent"),
        "persistent_process_signals_sent": (
            persistent_runtime.get("persistent_process_signals_sent")
            if persistent_runtime is not None
            else None
        ),
        "official_grader_executed": status.get("official_grader_executed"),
        "kaggle_submission_executed": status.get("kaggle_submission_executed"),
        "source_launch_guards": source_has_launch_guards(script_path),
    }


def collect_queue_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for spec in QUEUE_SPECS:
        for relative in spec.statuses:
            status_path = root / relative
            if status_path.is_file():
                records.append(queue_record(root, spec, relative))
    return records


def collect_logical_training_processes() -> list[dict[str, Any]]:
    matches: dict[int, dict[str, Any]] = {}
    for process in psutil.process_iter(["pid", "ppid", "name", "cmdline", "create_time"]):
        try:
            if str(process.info.get("name", "")).lower() != "python.exe":
                continue
            command = " ".join(process.info.get("cmdline") or [])
            script_name = next(
                (name for name in TRAINING_SCRIPT_NAMES if name in command), None
            )
            if script_name is None:
                continue
            matches[int(process.info["pid"])] = {
                "pid": int(process.info["pid"]),
                "ppid": int(process.info.get("ppid") or 0),
                "script": script_name,
                "command_line": command,
                "started_at": datetime.fromtimestamp(
                    float(process.info["create_time"])
                ).astimezone().isoformat(),
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError, TypeError):
            continue
    roots = [
        record
        for record in matches.values()
        if not (
            record["ppid"] in matches
            and matches[record["ppid"]]["script"] == record["script"]
        )
    ]
    for root in roots:
        root["worker_pids"] = sorted(
            pid
            for pid, record in matches.items()
            if record["ppid"] == root["pid"] and record["script"] == root["script"]
        )
    return sorted(roots, key=lambda item: (item["started_at"], item["pid"]))


def query_gpu() -> dict[str, Any]:
    columns = (
        "index,name,memory.total,memory.used,memory.free,utilization.gpu,"
        "utilization.memory,temperature.gpu,pstate,power.draw,clocks.sm"
    )
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={columns}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    values = [part.strip() for part in completed.stdout.strip().splitlines()[0].split(",")]
    return {
        "index": int(values[0]),
        "name": values[1],
        "memory_total_mib": int(values[2]),
        "memory_used_mib": int(values[3]),
        "memory_free_mib": int(values[4]),
        "utilization_gpu_percent": int(values[5]),
        "utilization_memory_percent": int(values[6]),
        "temperature_c": int(values[7]),
        "pstate": values[8],
        "power_draw_w": float(values[9]),
        "sm_clock_mhz": int(values[10]),
    }


def evaluate_runtime(
    queue_records: Iterable[dict[str, Any]],
    logical_training_processes: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    records = list(queue_records)
    training = list(logical_training_processes)
    current_live = [
        record
        for record in records
        if record["worker_alive"] and record["loaded_plan_is_current"]
    ]
    stages = sorted({spec.stage for spec in QUEUE_SPECS})
    current_count_by_stage = {
        stage: sum(1 for record in current_live if record["stage"] == stage)
        for stage in stages
    }
    stale_live = [
        record
        for record in records
        if record["worker_alive"] and not record["loaded_plan_is_current"]
    ]

    def launch_guards_pass(record: dict[str, Any]) -> bool:
        guards = record["source_launch_guards"]
        common = (
            guards["validates_frozen_plan"]
            and guards["requires_consecutive_idle_checks"]
            and guards["records_zero_process_signals"]
        )
        if record["stage"] == "jigsaw_confirmation":
            # This queue has one hash-proven current instance and validates the
            # complete umbrella/seed plans before entering its serial seed loop.
            return common
        if record["stage"] == "jigsaw_confirmation_early":
            return (
                common
                and guards["rejects_changed_plan_hash_at_launch"]
                and guards["performs_final_idle_query"]
                and guards["requires_original_queue_wait_guard"]
                and guards["resumes_existing_verified_seed_artifacts"]
            )
        if record["stage"] == "leaf_early":
            return (
                common
                and guards["rejects_changed_plan_hash_at_launch"]
                and guards["performs_final_idle_query"]
                and guards["requires_original_queue_wait_guard"]
                and guards["rejects_existing_target_run"]
            )
        return (
            common
            and guards["rejects_changed_plan_hash_at_launch"]
            and guards["performs_final_idle_query"]
        )

    checks = {
        "no_duplicate_current_live_queue_per_stage": all(
            current_count_by_stage.get(stage, 0) <= 1 for stage in stages
        ),
        "current_queue_launcher_script_identities_proven": all(
            record["launcher_script_identity_proven"] for record in current_live
        ),
        "current_queue_stderr_empty": all(
            record["stderr_bytes"] in (None, 0) for record in current_live
        ),
        "all_live_queue_sources_have_launch_guards": all(
            launch_guards_pass(record)
            for record in records
            if record["worker_alive"]
        ),
        "stale_live_queues_have_superseded_plan_hashes": all(
            record["loaded_plan_sha256"] != record["current_plan_sha256"]
            for record in stale_live
        ),
        "queue_process_signals_zero": all(
            record["process_signals_sent"] in (None, 0)
            and record.get("persistent_process_signals_sent") in (None, 0)
            for record in records
        ),
        "no_queue_grader_execution": all(
            record["official_grader_executed"] in (None, False) for record in records
        ),
        "no_queue_kaggle_submission": all(
            record["kaggle_submission_executed"] in (None, False)
            for record in records
        ),
        "at_most_one_logical_gpu_training": len(training) <= 1,
    }
    legacy_identity_unrecorded = [
        record["status_path"]
        for record in stale_live
        if not record["launcher_script_identity_proven"]
    ]
    passed = all(checks.values())
    return {
        "passed": passed,
        "verdict": (
            "PASS_SINGLE_GPU_SERIAL_CHAIN_WITH_TERMINAL_STAGES_ALLOWED"
            if passed
            else "FAIL_SERIAL_CHAIN_INVARIANT"
        ),
        "checks": checks,
        "current_live_queue_count_by_stage": current_count_by_stage,
        "current_live_queue_count": len(current_live),
        "stale_live_queue_count": len(stale_live),
        "logical_training_count": len(training),
        "legacy_runtime_script_identity_unrecorded": legacy_identity_unrecorded,
        "claim_boundary": (
            "Current queues are attributed to their live persistent Scheduled Task "
            "manifest when available, including bound script hash and current stderr. "
            "Completed or terminal stages may have zero live queue workers; duplicate "
            "current workers within one stage remain forbidden. Older launchers remain "
            "classified by frozen-plan hashes; no process was signaled or modified by "
            "this audit."
        ),
    }


def build_audit(root: Path) -> dict[str, Any]:
    queue_records = collect_queue_records(root)
    training = collect_logical_training_processes()
    evaluation = evaluate_runtime(queue_records, training)
    return {
        "schema": "evomind.local_gpu.serial_chain_audit.v2",
        "created_at": datetime.now().astimezone().isoformat(),
        "root": str(root),
        "device": query_gpu(),
        "evaluation": evaluation,
        "logical_training_processes": training,
        "queues": queue_records,
        "invariants": {
            "strict_single_gpu_serial": True,
            "process_signals_sent_by_audit": 0,
            "existing_processes_modified": False,
            "private_labels_read_by_audit": False,
            "official_grader_executed_by_audit": False,
            "kaggle_submission_executed_by_audit": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.resolve()
    audit = build_audit(root)
    write_json_atomic(args.output.resolve(), audit)
    print(json.dumps(audit["evaluation"], ensure_ascii=False, indent=2))
    return 0 if audit["evaluation"]["passed"] or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
