"""Keep the isolated A800 lane busy across the remaining medal-recovery tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts import mlebench_remote_ops as remote_ops

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WATCH_DIR = PROJECT_ROOT / "workspace" / "hpc" / "a800_lane_watch"
LANE_DIR = PROJECT_ROOT / "workspace" / "hpc" / "a800_lane"
RUN_ID_FILE = PROJECT_ROOT / "workspace" / "hpc" / "a800_lane_run_id_current.txt"
MARKER_PATH = PROJECT_ROOT / "workspace" / "hpc" / "a800_recovery_queue_current.json"
DEFAULT_DECISION_REPORT = (
    PROJECT_ROOT / "workspace" / "llm" / "mlebench_gpt56_adaptive_loop_current.json"
)
WATCHER_LAUNCH_PATH = (
    PROJECT_ROOT / "workspace" / "hpc" / "a800_lane_supervisor_launch_current.json"
)
LEGACY_ACTIVE_BUNDLE_SHA256 = (
    "63aaaac2a44ae8de406bb04a0eef3ad2b663f7d782227ab24fb52ceb6f5d621f"
)
LAUNCH_BUNDLE_SHA256 = remote_ops.EXPECTED_BUNDLE_SHA256
# Compatibility alias for callers that mean the bundle required for new launches.
EXPECTED_BUNDLE_SHA256 = LAUNCH_BUNDLE_SHA256


class TerminalTaskFailure(RuntimeError):
    """A remote run stopped without one valid official private grade."""

    def __init__(self, evidence: dict[str, Any]) -> None:
        super().__init__(
            f"{evidence['competition_id']} run {evidence['run_id']} ended "
            f"with {evidence['summary_status']}"
        )
        self.evidence = evidence


@dataclass(frozen=True)
class QueueTask:
    competition_id: str
    wave: str
    run_slug: str
    runner_performance_overrides: tuple[str, ...] = ()


CURRENT_TASK = QueueTask(
    "siim-isic-melanoma-classification",
    "Wave0",
    "siim",
    (
        "--siim-workers",
        "8",
    ),
)
ADAPTIVE_TASK_SPECS = {
    task.competition_id: task
    for task in (
        QueueTask("aptos2019-blindness-detection", "Wave2", "aptos2019"),
        QueueTask(
            "jigsaw-toxic-comment-classification-challenge",
            "Wave2",
            "jigsaw_toxic",
        ),
        QueueTask("aerial-cactus-identification", "Wave0", "aerial_cactus"),
        QueueTask(
            "ranzcr-clip-catheter-line-classification",
            "Wave2",
            "ranzcr",
            ("--wave2-workers", "32"),
        ),
        QueueTask("spooky-author-identification", "Wave0", "spooky"),
        QueueTask("dog-breed-identification", "Wave2", "dog_breed"),
    )
}
NEXT_TASKS: tuple[QueueTask, ...] = ()
PARALLEL_RUNS: dict[str, str] = {}
EXCLUDED_COMPETITIONS: frozenset[str] = frozenset()


def load_adaptive_tasks(
    path: Path,
    *,
    active_run_id: str,
    allow_descendant_active_run: bool = False,
    excluded_competitions: frozenset[str] = frozenset(),
) -> tuple[QueueTask, ...]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if (
        payload.get("schema") != "evomind.mlebench.gpt56_adaptive_loop.v1"
        or payload.get("status") != "passed"
        or payload.get("ok") is not True
        or payload.get("validation_errors")
    ):
        raise RuntimeError("Adaptive decision report did not pass validation")
    decision = payload.get("decision") or {}
    if decision.get("schema") != "evomind.mlebench.adaptive_decision.v1":
        raise RuntimeError("Adaptive decision schema is unsupported")
    active_policy = {
        str(item.get("lane") or ""): item
        for item in decision.get("active_run_policy", [])
        if isinstance(item, dict)
    }
    a800_policy = active_policy.get("a800") or {}
    if a800_policy.get("action") != "continue_active_runs" or (
        not allow_descendant_active_run
        and a800_policy.get("run_id") != active_run_id
    ):
        raise RuntimeError("Adaptive decision does not protect the current A800 run")

    selected: list[QueueTask] = []
    seen: set[str] = set()
    actions = sorted(
        (item for item in decision.get("next_actions", []) if isinstance(item, dict)),
        key=lambda item: int(item.get("priority") or 0),
    )
    for item in actions:
        if item.get("lane") != "a800" or item.get("action") not in {
            "queue_experiment",
            "retry_failed_after_active",
        }:
            continue
        competition_id = str(item.get("competition_id") or "")
        if competition_id in seen:
            raise RuntimeError("Adaptive A800 queue contains a duplicate competition")
        task = ADAPTIVE_TASK_SPECS.get(competition_id)
        if task is None:
            raise RuntimeError("Adaptive A800 queue contains an unsupported competition")
        if item.get("requires_active_run_completion") is not True:
            raise RuntimeError("Adaptive A800 action may interrupt the active run")
        if competition_id in excluded_competitions:
            seen.add(competition_id)
            continue
        selected.append(task)
        seen.add(competition_id)
    unsupported_exclusions = excluded_competitions - ADAPTIVE_TASK_SPECS.keys()
    if unsupported_exclusions:
        raise RuntimeError("Excluded A800 competition is unsupported")
    if not selected and not excluded_competitions:
        raise RuntimeError("Adaptive decision contains no executable A800 tasks")
    return tuple(selected)


def parse_parallel_runs(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        competition_id, separator, run_id = value.partition("=")
        if not separator or not competition_id or not run_id:
            raise RuntimeError("Parallel run must use COMPETITION_ID=RUN_ID")
        if competition_id in parsed:
            raise RuntimeError("Parallel competition is duplicated")
        parsed[competition_id] = run_id
    return parsed


def read_run_launch_evidence(
    run_id: str,
    competition_id: str,
    *,
    require_cpu_light: bool | None = None,
) -> dict[str, Any]:
    launch_path = LANE_DIR / run_id / "launch_summary.json"
    if not launch_path.is_file():
        raise RuntimeError("A800 run launch evidence is missing")
    launch = json.loads(launch_path.read_text(encoding="utf-8-sig"))
    valid = (
        launch.get("schema") == "evomind.mlebench.a800_lane_launch.v1"
        and launch.get("competition_id") == competition_id
        and launch.get("run_id") == run_id
        and launch.get("start_passed") is True
    )
    if require_cpu_light is not None:
        valid = valid and launch.get("cpu_light_concurrent") is require_cpu_light
    if not valid:
        raise RuntimeError("A800 run launch evidence is invalid")
    bundle_sha256 = str(launch.get("bundle_sha256") or "")
    if bundle_sha256 not in remote_ops.TRUSTED_CONCURRENT_RELEASE_SHA256S:
        raise RuntimeError("A800 run used an untrusted bundle release")
    return launch


def bundle_marker_fields(active_bundle_sha256: str) -> dict[str, str]:
    return {
        "bundle_sha256": LAUNCH_BUNDLE_SHA256,
        "active_bundle_sha256": active_bundle_sha256,
        "launch_bundle_sha256": LAUNCH_BUNDLE_SHA256,
    }


def remaining_competition_ids(
    current_task: QueueTask,
    completed: list[dict[str, Any]],
) -> list[str]:
    completed_ids = {
        str(record.get("competition_id") or "")
        for record in completed
        if isinstance(record, dict)
    }
    remaining: list[str] = []
    seen: set[str] = set()
    for task in (current_task, *NEXT_TASKS):
        competition_id = task.competition_id
        if (
            competition_id in seen
            or competition_id in completed_ids
            or competition_id in PARALLEL_RUNS
            or competition_id in EXCLUDED_COMPETITIONS
        ):
            continue
        remaining.append(competition_id)
        seen.add(competition_id)
    return remaining


def resume_queue_state(
    marker_path: Path,
    *,
    current_run_id: str,
    current_competition_id: str,
    selected_tasks: tuple[QueueTask, ...],
    parallel_runs: dict[str, str],
) -> tuple[QueueTask, tuple[QueueTask, ...], list[dict[str, Any]], str]:
    marker = json.loads(marker_path.read_text(encoding="utf-8-sig"))
    if (
        marker.get("schema") != "evomind.mlebench.a800_recovery_queue.v1"
        or marker.get("status") != "waiting_for_official_completion"
        or marker.get("current_run_id") != current_run_id
        or marker.get("current_competition_id") != current_competition_id
    ):
        raise RuntimeError("A800 queue resume marker does not match the active run")

    marker_launch_bundle = marker.get("launch_bundle_sha256")
    if marker_launch_bundle is not None and marker_launch_bundle != LAUNCH_BUNDLE_SHA256:
        raise RuntimeError("A800 queue resume launch bundle is stale")
    active_bundle_sha256 = str(
        marker.get("active_bundle_sha256") or marker.get("bundle_sha256") or ""
    )
    active_launch = read_run_launch_evidence(
        current_run_id,
        current_competition_id,
        require_cpu_light=False,
    )
    if active_launch.get("bundle_sha256") != active_bundle_sha256:
        raise RuntimeError("A800 queue resume active bundle evidence is inconsistent")

    selected_ids = [task.competition_id for task in selected_tasks]
    current_task = ADAPTIVE_TASK_SPECS.get(current_competition_id)
    if current_task is None:
        raise RuntimeError("A800 queue resume competition is unsupported")
    if current_competition_id in selected_ids:
        current_index = selected_ids.index(current_competition_id)
        pending_tasks = selected_tasks[current_index + 1 :]
    else:
        pending_tasks = selected_tasks

    for competition_id, run_id in parallel_runs.items():
        if competition_id == current_competition_id or competition_id not in selected_ids:
            raise RuntimeError("Parallel run is not a validated adaptive task")
        launch = read_run_launch_evidence(
            run_id,
            competition_id,
            require_cpu_light=True,
        )
        if launch.get("bundle_sha256") not in {
            active_bundle_sha256,
            LAUNCH_BUNDLE_SHA256,
        }:
            raise RuntimeError("Parallel run bundle is outside the active release handoff")

    next_tasks = tuple(
        task
        for task in pending_tasks
        if task.competition_id not in parallel_runs
    )
    completed = marker.get("completed") or []
    if not isinstance(completed, list):
        raise RuntimeError("A800 queue resume completed records are invalid")
    return current_task, next_tasks, list(completed), active_bundle_sha256


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raise_remote_nofile(profile_dir: Path, pid: int) -> dict[str, Any]:
    """Raise only the owned training process to the container's existing hard limit."""

    from src.research_agent_workstation.server.core.gpu_credentials import connect_ssh
    from workspace.hpc.probe_hpc_gpu_profile import _load_profile

    if pid <= 1:
        raise RuntimeError("Remote training PID is invalid")
    client = connect_ssh(_load_profile(profile_dir.resolve()))
    try:
        command = (
            f"set -e; prlimit --pid {pid} --nofile=65535:65535; "
            f"grep -i 'open files' /proc/{pid}/limits"
        )
        _stdin, stdout, stderr = client.exec_command(command, timeout=30)
        output = stdout.read().decode("utf-8", "replace").strip()
        error = stderr.read().decode("utf-8", "replace").strip()
        exit_code = stdout.channel.recv_exit_status()
    finally:
        client.close()
    passed = exit_code == 0 and "65535" in output and not error
    if not passed:
        raise RuntimeError("Remote training process file limit was not raised")
    return {
        "passed": True,
        "pid": pid,
        "soft_limit": 65535,
        "hard_limit": 65535,
    }


def read_valid_completion(
    run_id: str,
    competition_id: str,
) -> dict[str, Any] | None:
    path = WATCH_DIR / f"{run_id}_completion.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    results = payload.get("official_results") or []
    if payload.get("run_id") != run_id:
        raise RuntimeError(f"{competition_id} completion belongs to another run")
    valid = (
        payload.get("process") != "running"
        and payload.get("summary_status") == "passed"
        and int(payload.get("official_grade_count") or 0) == 1
        and len(results) == 1
        and results[0].get("competition_id") == competition_id
        and results[0].get("mle_private_grader_score") is not None
        and (payload.get("collection") or {}).get("passed") is True
    )
    return payload if valid else None


def read_terminal_failure(
    run_id: str,
    competition_id: str,
) -> dict[str, Any] | None:
    status_path = LANE_DIR / run_id / "status_current.json"
    if not status_path.is_file():
        return None
    payload = json.loads(status_path.read_text(encoding="utf-8-sig"))
    if payload.get("process") == "running":
        return None
    summary = payload.get("summary") or {}
    if not summary:
        return None
    results = [
        result
        for result in summary.get("results", [])
        if result.get("competition_id") == competition_id
    ]
    result = results[-1] if results else {}
    if summary.get("status") == "passed" and result.get("status") == "passed":
        return None
    traceback_lines = [
        line.strip()
        for line in str(result.get("traceback") or "").splitlines()
        if line.strip()
    ]
    return {
        "schema": "evomind.mlebench.a800_terminal_failure.v1",
        "created_at": now_iso(),
        "run_id": run_id,
        "competition_id": competition_id,
        "process": payload.get("process"),
        "summary_status": summary.get("status"),
        "result_status": result.get("status"),
        "failure_tail": traceback_lines[-1][-500:] if traceback_lines else "",
        "status_path": str(status_path.resolve()),
        "human_gate_preserved": True,
        "kaggle_submission_enabled": False,
    }


def wait_for_completion(
    run_id: str,
    task: QueueTask,
    *,
    poll_seconds: int,
    completed: list[dict[str, Any]],
    active_bundle_sha256: str,
) -> dict[str, Any]:
    while True:
        completion = read_valid_completion(run_id, task.competition_id)
        if completion is not None:
            return completion
        failure = read_terminal_failure(run_id, task.competition_id)
        if failure is not None:
            raise TerminalTaskFailure(failure)
        write_json_atomic(
            MARKER_PATH,
            {
                "schema": "evomind.mlebench.a800_recovery_queue.v1",
                "created_at": now_iso(),
                "pid": os.getpid(),
                "status": "waiting_for_official_completion",
                "current_run_id": run_id,
                "current_competition_id": task.competition_id,
                "completed": completed,
                "remaining": remaining_competition_ids(task, completed),
                "parallel_runs": PARALLEL_RUNS,
                "excluded_competitions": sorted(EXCLUDED_COMPETITIONS),
                **bundle_marker_fields(active_bundle_sha256),
                "campaign_mutation_enabled": False,
                "kaggle_submission_enabled": False,
            },
        )
        time.sleep(max(10, poll_seconds))


def wait_for_task_with_recovery(
    run_id: str,
    task: QueueTask,
    *,
    profile_dir: Path,
    bundle: Path,
    poll_seconds: int,
    launch_attempts: int,
    run_attempts: int,
    completed: list[dict[str, Any]],
    active_bundle_sha256: str,
) -> tuple[dict[str, Any], str, str]:
    current_run_id = run_id
    current_bundle_sha256 = active_bundle_sha256
    failures: list[dict[str, Any]] = []
    for run_attempt in range(1, max(1, run_attempts) + 1):
        try:
            completion = wait_for_completion(
                current_run_id,
                task,
                poll_seconds=poll_seconds,
                completed=completed,
                active_bundle_sha256=current_bundle_sha256,
            )
            return completion, current_run_id, current_bundle_sha256
        except TerminalTaskFailure as exc:
            failure = dict(exc.evidence)
            failure["run_attempt"] = run_attempt
            failure["run_attempts_allowed"] = max(1, run_attempts)
            failures.append(failure)
            write_json_atomic(
                LANE_DIR / current_run_id / "terminal_failure_recovery.json",
                failure,
            )
            if run_attempt >= max(1, run_attempts):
                write_json_atomic(
                    MARKER_PATH,
                    {
                        "schema": "evomind.mlebench.a800_recovery_queue.v1",
                        "created_at": now_iso(),
                        "pid": os.getpid(),
                        "status": "run_retry_exhausted",
                        "current_run_id": current_run_id,
                        "current_competition_id": task.competition_id,
                        "run_failures": failures,
                        "completed": completed,
                        **bundle_marker_fields(current_bundle_sha256),
                        "campaign_mutation_enabled": False,
                        "kaggle_submission_enabled": False,
                    },
                )
                raise
            write_json_atomic(
                MARKER_PATH,
                {
                    "schema": "evomind.mlebench.a800_recovery_queue.v1",
                    "created_at": now_iso(),
                    "pid": os.getpid(),
                    "status": "terminal_failure_retrying",
                    "current_run_id": current_run_id,
                    "current_competition_id": task.competition_id,
                    "next_run_attempt": run_attempt + 1,
                    "run_failures": failures,
                    "completed": completed,
                    **bundle_marker_fields(current_bundle_sha256),
                    "campaign_mutation_enabled": False,
                    "kaggle_submission_enabled": False,
                },
            )
            current_run_id, _watcher_pid = launch_task(
                task,
                profile_dir=profile_dir,
                bundle=bundle,
                launch_attempts=launch_attempts,
                completed=completed,
            )
            current_bundle_sha256 = LAUNCH_BUNDLE_SHA256
    raise RuntimeError("A800 recovery loop exhausted without a terminal outcome")


def launch_task(
    task: QueueTask,
    *,
    profile_dir: Path,
    bundle: Path,
    launch_attempts: int,
    completed: list[dict[str, Any]],
) -> tuple[str, int]:
    last_error: Exception | None = None
    for attempt in range(1, max(1, launch_attempts) + 1):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"a800_{task.run_slug}_recovery_s42_{timestamp}"
        evidence_dir = LANE_DIR / run_id
        write_json_atomic(
            MARKER_PATH,
            {
                "schema": "evomind.mlebench.a800_recovery_queue.v1",
                "created_at": now_iso(),
                "pid": os.getpid(),
                "status": "launching",
                "attempt": attempt,
                "next_run_id": run_id,
                "next_competition_id": task.competition_id,
                "runner_performance_overrides": list(task.runner_performance_overrides),
                "completed": completed,
                "remaining": remaining_competition_ids(task, completed),
                "parallel_runs": PARALLEL_RUNS,
                "excluded_competitions": sorted(EXCLUDED_COMPETITIONS),
                **bundle_marker_fields(LAUNCH_BUNDLE_SHA256),
                "campaign_mutation_enabled": False,
                "kaggle_submission_enabled": False,
            },
        )
        command = [
            sys.executable,
            str(PROJECT_ROOT / "workspace" / "hpc" / "launch_mlebench_a800_lane.py"),
            "--profile-dir",
            str(profile_dir),
            "--bundle",
            str(bundle),
            "--competition",
            task.competition_id,
            "--wave",
            task.wave,
            "--seed",
            "42",
            "--run-id",
            run_id,
            "--optimization-plan-name",
            "medal_recovery_gpt56_current.json",
            "--gate-interval-seconds",
            "3",
            "--evidence-dir",
            str(evidence_dir),
            "--allow-concurrent-with-cpu-light",
        ]
        for value in task.runner_performance_overrides:
            command.append(f"--runner-performance-override={value}")
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(PROJECT_ROOT)
        try:
            completed_launch = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=600,
            )
            launch_summary = json.loads(
                (evidence_dir / "launch_summary.json").read_text(encoding="utf-8-sig")
            )
            if launch_summary.get("start_passed") is not True:
                raise RuntimeError("A800 launch summary did not pass")
            remote_pid = int(launch_summary.get("remote_pid") or 0)
            nofile = raise_remote_nofile(profile_dir, remote_pid)
            RUN_ID_FILE.write_text(run_id + "\n", encoding="utf-8")

            stdout_path = (
                PROJECT_ROOT
                / "workspace"
                / "hpc"
                / f"a800_lane_supervisor_{run_id}.stdout.log"
            )
            stderr_path = (
                PROJECT_ROOT
                / "workspace"
                / "hpc"
                / f"a800_lane_supervisor_{run_id}.stderr.log"
            )
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr_handle:
                watcher = subprocess.Popen(
                    [
                        sys.executable,
                        str(PROJECT_ROOT / "workspace" / "hpc" / "a800_lane_supervisor.py"),
                        "--profile-dir",
                        str(profile_dir),
                        "--run-id",
                        run_id,
                        "--evidence-dir",
                        str(WATCH_DIR),
                        "--poll-seconds",
                        "60",
                    ],
                    cwd=PROJECT_ROOT,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    creationflags=creationflags,
                )
            write_json_atomic(
                WATCHER_LAUNCH_PATH,
                {
                    "schema": "evomind.mlebench.a800_lane_watch.launch.v1",
                    "created_at": now_iso(),
                    "pid": watcher.pid,
                    "run_id": run_id,
                    "poll_seconds": 60,
                    "profile_secret_policy": "DPAPI only; no secret material persisted",
                    "campaign_mutation_enabled": False,
                    "kaggle_submission_enabled": False,
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                },
            )
            write_json_atomic(
                MARKER_PATH,
                {
                    "schema": "evomind.mlebench.a800_recovery_queue.v1",
                    "created_at": now_iso(),
                    "pid": os.getpid(),
                    "status": "started_and_watched",
                    "run_id": run_id,
                    "competition_id": task.competition_id,
                    "runner_performance_overrides": list(
                        task.runner_performance_overrides
                    ),
                    "remote_pid": launch_summary.get("remote_pid"),
                    "remote_nofile": nofile,
                    "watcher_pid": watcher.pid,
                    "completed": completed,
                    "remaining": remaining_competition_ids(task, completed),
                    "parallel_runs": PARALLEL_RUNS,
                    "excluded_competitions": sorted(EXCLUDED_COMPETITIONS),
                    **bundle_marker_fields(LAUNCH_BUNDLE_SHA256),
                    "launch_stdout_tail": completed_launch.stdout[-2000:],
                    "campaign_mutation_enabled": False,
                    "kaggle_submission_enabled": False,
                },
            )
            return run_id, watcher.pid
        except Exception as exc:
            last_error = exc
            write_json_atomic(
                MARKER_PATH,
                {
                    "schema": "evomind.mlebench.a800_recovery_queue.v1",
                    "created_at": now_iso(),
                    "pid": os.getpid(),
                    "status": "launch_retry_pending",
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "next_run_id": run_id,
                    "next_competition_id": task.competition_id,
                    "completed": completed,
                    "remaining": remaining_competition_ids(task, completed),
                    "parallel_runs": PARALLEL_RUNS,
                    "excluded_competitions": sorted(EXCLUDED_COMPETITIONS),
                    **bundle_marker_fields(LAUNCH_BUNDLE_SHA256),
                    "campaign_mutation_enabled": False,
                    "kaggle_submission_enabled": False,
                },
            )
            if attempt < max(1, launch_attempts):
                time.sleep(60)
    assert last_error is not None
    raise last_error


def completion_record(completion: dict[str, Any]) -> dict[str, Any]:
    official = completion["official_results"][0]
    return {
        "competition_id": official.get("competition_id"),
        "run_id": completion.get("run_id"),
        "metric": official.get("metric"),
        "direction": official.get("direction"),
        "cv_score": official.get("cv_score"),
        "mle_private_grader_score": official.get("mle_private_grader_score"),
        "upstream_report": official.get("upstream_report"),
    }


def main() -> int:
    global EXCLUDED_COMPETITIONS, NEXT_TASKS, PARALLEL_RUNS

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--current-run-id", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--launch-attempts", type=int, default=3)
    parser.add_argument("--run-attempts", type=int, default=1)
    parser.add_argument("--decision-report", type=Path, default=DEFAULT_DECISION_REPORT)
    parser.add_argument("--resume-marker", type=Path)
    parser.add_argument("--current-competition-id")
    parser.add_argument("--parallel-run", action="append", default=[])
    parser.add_argument("--exclude-competition", action="append", default=[])
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    if sha256_file(bundle) != LAUNCH_BUNDLE_SHA256:
        raise RuntimeError("Pinned A800 recovery bundle SHA256 mismatch")
    profile_dir = args.profile_dir.resolve()
    EXCLUDED_COMPETITIONS = frozenset(str(value) for value in args.exclude_competition)
    unsupported_exclusions = EXCLUDED_COMPETITIONS - ADAPTIVE_TASK_SPECS.keys()
    if unsupported_exclusions:
        raise RuntimeError("Excluded A800 competition is unsupported")
    if args.current_competition_id in EXCLUDED_COMPETITIONS:
        raise RuntimeError("The active A800 competition cannot be excluded")
    selected_tasks = load_adaptive_tasks(
        args.decision_report.resolve(),
        active_run_id=args.current_run_id,
        allow_descendant_active_run=args.resume_marker is not None,
        excluded_competitions=EXCLUDED_COMPETITIONS,
    )
    PARALLEL_RUNS = parse_parallel_runs(args.parallel_run)
    if args.resume_marker is not None:
        if not args.current_competition_id:
            raise RuntimeError("Queue resume requires --current-competition-id")
        current_task, NEXT_TASKS, completed, active_bundle_sha256 = resume_queue_state(
            args.resume_marker.resolve(),
            current_run_id=args.current_run_id,
            current_competition_id=args.current_competition_id,
            selected_tasks=selected_tasks,
            parallel_runs=PARALLEL_RUNS,
        )
    else:
        if args.current_competition_id or PARALLEL_RUNS:
            raise RuntimeError("Current competition and parallel runs require --resume-marker")
        NEXT_TASKS = selected_tasks
        completed = []
        current_task = CURRENT_TASK
        active_bundle_sha256 = read_run_launch_evidence(
            args.current_run_id,
            current_task.competition_id,
            require_cpu_light=False,
        ).get("bundle_sha256")
        if not isinstance(active_bundle_sha256, str):
            raise RuntimeError("Active A800 run bundle evidence is missing")
    terminal_failures: list[dict[str, Any]] = []
    current_run_id = args.current_run_id

    while NEXT_TASKS:
        next_task = NEXT_TASKS[0]
        try:
            completion, current_run_id, active_bundle_sha256 = wait_for_task_with_recovery(
                current_run_id,
                current_task,
                profile_dir=profile_dir,
                bundle=bundle,
                poll_seconds=args.poll_seconds,
                launch_attempts=args.launch_attempts,
                run_attempts=args.run_attempts,
                completed=completed,
                active_bundle_sha256=active_bundle_sha256,
            )
        except TerminalTaskFailure as exc:
            terminal_failures.append(dict(exc.evidence))
        else:
            completed.append(completion_record(completion))
        NEXT_TASKS = NEXT_TASKS[1:]
        current_run_id, _watcher_pid = launch_task(
            next_task,
            profile_dir=profile_dir,
            bundle=bundle,
            launch_attempts=args.launch_attempts,
            completed=completed,
        )
        current_task = next_task
        active_bundle_sha256 = LAUNCH_BUNDLE_SHA256

    try:
        completion, current_run_id, active_bundle_sha256 = wait_for_task_with_recovery(
            current_run_id,
            current_task,
            profile_dir=profile_dir,
            bundle=bundle,
            poll_seconds=args.poll_seconds,
            launch_attempts=args.launch_attempts,
            run_attempts=args.run_attempts,
            completed=completed,
            active_bundle_sha256=active_bundle_sha256,
        )
    except TerminalTaskFailure as exc:
        terminal_failures.append(dict(exc.evidence))
    else:
        completed.append(completion_record(completion))
    write_json_atomic(
        MARKER_PATH,
        {
            "schema": "evomind.mlebench.a800_recovery_queue.v1",
            "created_at": now_iso(),
            "pid": os.getpid(),
            "status": "completed",
            "completed": completed,
            "terminal_failures": terminal_failures,
            "parallel_runs": PARALLEL_RUNS,
            "excluded_competitions": sorted(EXCLUDED_COMPETITIONS),
            "decision_report": str(args.decision_report.resolve()),
            **bundle_marker_fields(active_bundle_sha256),
            "campaign_mutation_enabled": False,
            "campaign_integration_required": True,
            "kaggle_submission_enabled": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
