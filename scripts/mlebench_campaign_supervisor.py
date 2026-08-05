#!/usr/bin/env python3
"""Run the Job 89441 MLE-Bench coverage campaign as a gated, resumable queue.

The supervisor never starts a task until the authoritative three-sample GPU
idle gate passes.  Every task is a separate remote run so its official private
grader evidence can be collected and merged into the progress ledger before
the next task starts.  Kaggle submission remains outside this workflow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
CONTROL_ROOT = PROJECT_ROOT / "workspace" / "hpc" / "mlebench_remote_ops"
STATE_PATH = CONTROL_ROOT / "job89441_full_campaign_current.json"
LOCK_PATH = CONTROL_ROOT / "job89441_full_campaign.lock.json"
LOG_PATH = CONTROL_ROOT / "job89441_full_campaign.log"
PROGRESS_JSON = PROJECT_ROOT / "workspace" / "mlebench_progress" / "lite11_current.json"
PROGRESS_MD = PROJECT_ROOT / "workspace" / "mlebench_progress" / "lite11_current.md"
MULTISEED_JSON = (
    PROJECT_ROOT / "workspace" / "mlebench_progress" / "job89441_multiseed_current.json"
)
MULTISEED_MD = (
    PROJECT_ROOT / "workspace" / "mlebench_progress" / "job89441_multiseed_current.md"
)
LOW_SPLIT = PROJECT_ROOT / "external-projects" / "mle-bench" / "experiments" / "splits" / "low.txt"
LEADERBOARD_README = PROJECT_ROOT / "external-projects" / "mle-bench" / "README.md"
WAVE1_PLAN = PROJECT_ROOT / "workspace" / "mlebench_plans" / "wave1_gpt56_current.json"
WAVE2_PLAN = PROJECT_ROOT / "workspace" / "mlebench_plans" / "wave2_gpt56_current.json"
RECOVERY_PLAN = (
    PROJECT_ROOT / "workspace" / "mlebench_plans" / "medal_recovery_gpt56_current.json"
)
CANDIDATE_GATES = (
    PROJECT_ROOT / "workspace" / "mlebench_plans" / "medal_candidate_gates_current.json"
)
OPENAI_GATEWAY_EVIDENCE = (
    PROJECT_ROOT / "workspace" / "llm" / "openai_gateway_smoke_current.json"
)
NATIVE_TOOL_LOOP_EVIDENCE = (
    PROJECT_ROOT / "workspace" / "llm" / "evomind_native_tool_loop_current.json"
)
A800_QUEUE_MARKER = PROJECT_ROOT / "workspace" / "hpc" / "a800_recovery_queue_current.json"
A800_DECISION_REPORT = (
    PROJECT_ROOT / "workspace" / "llm" / "mlebench_gpt56_adaptive_loop_current.json"
)
A800_WATCH_DIR = PROJECT_ROOT / "workspace" / "hpc" / "a800_lane_watch"
A800_COLLECTION_ROOT = CONTROL_ROOT / "collected"
A800_INTEGRATION_DIR = PROJECT_ROOT / "workspace" / "hpc" / "a800_campaign_integration"
DATA_INVENTORY_EVIDENCE = (
    PROJECT_ROOT / "workspace" / "mlebench_lite_inventory_current.json"
)
REQUIRED_LLM_MODEL = "gpt-5.6-sol"
PRIMARY_SEED = 42
CONFIRMATION_SEEDS = (43, 44)
ALL_SEEDS = (PRIMARY_SEED, *CONFIRMATION_SEEDS)
LITE_TOTAL = 22
MEDALS_REQUIRED = 18
MAX_TASK_ATTEMPTS = 3
TRANSIENT_REMOTE_ERROR_NAMES = frozenset(
    {
        "AuthenticationException",
        "BannerTimeoutError",
        "ChannelException",
        "EOFError",
        "NoValidConnectionsError",
        "ProxyCommandFailure",
        "SSHException",
    }
)


@dataclass(frozen=True)
class CampaignTask:
    competition_id: str
    wave: str
    slug: str
    optimization_plan_name: str
    purpose: str


# Run the unscored Lite competitions in the exact gpt-5.6-sol Wave 2
# plan order: quick audio/text/small-vision evidence first, progressively
# larger vision tasks next, and the full Histopath image set last.  This
# maximizes early official-grade throughput after an idle GPU window opens
# instead of spending the first window on the longest ingestion/training path.
# After coverage, execute every bounded conversion target in the exact
# gpt-5.6-sol recovery-plan order.  The complete queue is mathematically
# capable of reaching 18/22; the old Leaf-only tail was capped at 15/22.
CAMPAIGN_TASKS: tuple[CampaignTask, ...] = (
    CampaignTask(
        "mlsp-2013-birds",
        "Wave2",
        "mlsp_birds",
        "wave2_gpt56_current.json",
        "official_coverage",
    ),
    CampaignTask(
        "the-icml-2013-whale-challenge-right-whale-redux",
        "Wave2",
        "right_whale",
        "wave2_gpt56_current.json",
        "official_coverage",
    ),
    CampaignTask(
        "plant-pathology-2020-fgvc7",
        "Wave2",
        "plant_pathology",
        "wave2_gpt56_current.json",
        "official_coverage",
    ),
    CampaignTask(
        "jigsaw-toxic-comment-classification-challenge",
        "Wave2",
        "jigsaw_toxic",
        "medal_recovery_gpt56_current.json",
        "official_medal_conversion",
    ),
    CampaignTask(
        "aptos2019-blindness-detection",
        "Wave2",
        "aptos2019",
        "wave2_gpt56_current.json",
        "official_coverage",
    ),
    CampaignTask(
        "dog-breed-identification",
        "Wave2",
        "dog_breed",
        "wave2_gpt56_current.json",
        "official_coverage",
    ),
    CampaignTask(
        "ranzcr-clip-catheter-line-classification",
        "Wave2",
        "ranzcr",
        "wave2_gpt56_current.json",
        "official_coverage",
    ),
    CampaignTask(
        "histopathologic-cancer-detection",
        "Wave2",
        "histopathologic",
        "wave2_gpt56_current.json",
        "official_coverage",
    ),
    CampaignTask(
        "spooky-author-identification",
        "Wave0",
        "spooky_recovery",
        "medal_recovery_gpt56_current.json",
        "official_medal_conversion",
    ),
    CampaignTask(
        "new-york-city-taxi-fare-prediction",
        "Wave1",
        "taxi_recovery",
        "medal_recovery_gpt56_current.json",
        "official_medal_conversion",
    ),
    CampaignTask(
        "siim-isic-melanoma-classification",
        "Wave0",
        "siim_recovery",
        "medal_recovery_gpt56_current.json",
        "official_medal_conversion",
    ),
    CampaignTask(
        "tabular-playground-series-may-2022",
        "Wave0",
        "may2022_recovery",
        "medal_recovery_gpt56_current.json",
        "official_medal_conversion",
    ),
    CampaignTask(
        "aerial-cactus-identification",
        "Wave0",
        "aerial_recovery",
        "medal_recovery_gpt56_current.json",
        "official_medal_conversion",
    ),
    CampaignTask(
        "leaf-classification",
        "Wave1",
        "leaf_multimodal",
        "medal_recovery_gpt56_current.json",
        "official_medal_conversion",
    ),
)


# Seed 42 keeps the bounded 14-task coverage/recovery campaign above.  Once it
# proves 22/22 official grades and at least 18 medals, the frozen configuration
# is repeated across all 22 Lite tasks for Seeds 43 and 44.  The order matches
# the authoritative low split so coverage drift is easy to detect.
_CONFIRMATION_EXTRA_TASKS: tuple[CampaignTask, ...] = (
    CampaignTask(
        "denoising-dirty-documents",
        "Wave1",
        "denoising_recovery",
        "medal_recovery_gpt56_current.json",
        "seed_confirmation",
    ),
    CampaignTask(
        "detecting-insults-in-social-commentary",
        "Wave1",
        "insults",
        "wave1_gpt56_current.json",
        "seed_confirmation",
    ),
    CampaignTask(
        "dogs-vs-cats-redux-kernels-edition",
        "Wave1",
        "dogs_recovery",
        "medal_recovery_gpt56_current.json",
        "seed_confirmation",
    ),
    CampaignTask(
        "nomad2018-predict-transparent-conductors",
        "Wave2",
        "nomad",
        "wave2_gpt56_current.json",
        "seed_confirmation",
    ),
    CampaignTask(
        "random-acts-of-pizza",
        "Wave1",
        "pizza",
        "wave1_gpt56_current.json",
        "seed_confirmation",
    ),
    CampaignTask(
        "tabular-playground-series-dec-2021",
        "Wave1",
        "dec2021",
        "wave1_gpt56_current.json",
        "seed_confirmation",
    ),
    CampaignTask(
        "text-normalization-challenge-english-language",
        "Wave2",
        "text_en",
        "wave2_gpt56_current.json",
        "seed_confirmation",
    ),
    CampaignTask(
        "text-normalization-challenge-russian-language",
        "Wave2",
        "text_ru",
        "wave2_gpt56_current.json",
        "seed_confirmation",
    ),
)
_CONFIRMATION_TASK_BY_ID = {
    task.competition_id: task for task in (*CAMPAIGN_TASKS, *_CONFIRMATION_EXTRA_TASKS)
}
_CONFIRMATION_ORDER = (
    "aerial-cactus-identification",
    "aptos2019-blindness-detection",
    "denoising-dirty-documents",
    "detecting-insults-in-social-commentary",
    "dog-breed-identification",
    "dogs-vs-cats-redux-kernels-edition",
    "histopathologic-cancer-detection",
    "jigsaw-toxic-comment-classification-challenge",
    "leaf-classification",
    "mlsp-2013-birds",
    "new-york-city-taxi-fare-prediction",
    "nomad2018-predict-transparent-conductors",
    "plant-pathology-2020-fgvc7",
    "random-acts-of-pizza",
    "ranzcr-clip-catheter-line-classification",
    "siim-isic-melanoma-classification",
    "spooky-author-identification",
    "tabular-playground-series-dec-2021",
    "tabular-playground-series-may-2022",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
    "the-icml-2013-whale-challenge-right-whale-redux",
)
CONFIRMATION_TASKS: tuple[CampaignTask, ...] = tuple(
    CampaignTask(
        competition_id,
        _CONFIRMATION_TASK_BY_ID[competition_id].wave,
        _CONFIRMATION_TASK_BY_ID[competition_id].slug,
        _CONFIRMATION_TASK_BY_ID[competition_id].optimization_plan_name,
        "seed_confirmation",
    )
    for competition_id in _CONFIRMATION_ORDER
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_log(event: str, **fields: Any) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"at": utc_now(), "event": event, **fields}
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))) and (
                exit_code.value == still_active
            )
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def a800_reserved_competitions(
    marker_path: Path = A800_QUEUE_MARKER,
    decision_path: Path = A800_DECISION_REPORT,
) -> set[str]:
    """Return validated tasks owned by the live isolated A800 controller."""

    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8-sig"))
        decision_report = json.loads(decision_path.read_text(encoding="utf-8-sig"))
        controller_pid = int(marker.get("pid") or 0)
    except (OSError, ValueError, json.JSONDecodeError):
        return set()
    if (
        marker.get("schema") != "evomind.mlebench.a800_recovery_queue.v1"
        or marker.get("status") == "completed"
        or not process_exists(controller_pid)
        or decision_report.get("schema")
        != "evomind.mlebench.gpt56_adaptive_loop.v1"
        or decision_report.get("status") != "passed"
        or decision_report.get("ok") is not True
        or decision_report.get("validation_errors")
    ):
        return set()
    decision = decision_report.get("decision") or {}
    if decision.get("schema") != "evomind.mlebench.adaptive_decision.v1":
        return set()

    reserved = {
        str(item.get("competition_id") or "")
        for item in decision.get("next_actions", [])
        if isinstance(item, dict)
        and item.get("lane") == "a800"
        and item.get("action") in {"queue_experiment", "retry_failed_after_active"}
        and item.get("requires_active_run_completion") is True
    }
    current = str(marker.get("current_competition_id") or "")
    if current:
        reserved.add(current)
    remaining = marker.get("remaining")
    if isinstance(remaining, list):
        reserved.update(item for item in remaining if isinstance(item, str))
    reserved.discard("")
    return reserved


def acquire_lock() -> None:
    if LOCK_PATH.is_file():
        try:
            existing = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
            existing_pid = int(existing.get("pid", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            existing_pid = 0
        if process_exists(existing_pid):
            raise RuntimeError("A live Job 89441 campaign supervisor already owns the lock")
    atomic_write_json(
        LOCK_PATH,
        {
            "schema": "evomind.mlebench_campaign.lock.v1",
            "pid": os.getpid(),
            "created_at": utc_now(),
        },
    )


def release_lock() -> None:
    if not LOCK_PATH.is_file():
        return
    try:
        payload = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        if int(payload.get("pid", 0)) == os.getpid():
            LOCK_PATH.unlink()
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return


def load_remote_ops() -> Any:
    sys.path.insert(0, str(SCRIPTS_ROOT))
    import mlebench_remote_ops  # type: ignore[import-not-found]

    return mlebench_remote_ops


def task_payloads(
    tasks: tuple[CampaignTask, ...] = CAMPAIGN_TASKS,
) -> list[dict[str, Any]]:
    return [asdict(task) for task in tasks]


def seed_progress_paths(seed: int) -> tuple[Path, Path]:
    if seed == PRIMARY_SEED:
        return PROGRESS_JSON, PROGRESS_MD
    if seed not in CONFIRMATION_SEEDS:
        raise ValueError(f"Unsupported campaign seed: {seed}")
    root = PROJECT_ROOT / "workspace" / "mlebench_progress"
    return (
        root / f"job89441_seed{seed}_current.json",
        root / f"job89441_seed{seed}_current.md",
    )


def tasks_for_seed(seed: int) -> tuple[CampaignTask, ...]:
    if seed == PRIMARY_SEED:
        return CAMPAIGN_TASKS
    if seed in CONFIRMATION_SEEDS:
        return CONFIRMATION_TASKS
    raise ValueError(f"Unsupported campaign seed: {seed}")


def validate_confirmation_queue(low_split: Path = LOW_SPLIT) -> dict[str, Any]:
    expected = tuple(
        line.strip()
        for line in low_split.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    actual = tuple(task.competition_id for task in CONFIRMATION_TASKS)
    if len(expected) != LITE_TOTAL or len(set(expected)) != LITE_TOTAL:
        raise RuntimeError("Authoritative Lite split is not 22 unique competitions")
    if actual != expected or len(set(actual)) != LITE_TOTAL:
        raise RuntimeError("Confirmation queue diverges from the authoritative Lite split")
    return {
        "passed": True,
        "competition_count": len(actual),
        "low_split_sha256": sha256_file(low_split),
    }


def campaign_reachability(
    progress: dict[str, Any],
    recovery_plan: dict[str, Any],
) -> dict[str, Any]:
    """Prove the queued search space can still reach the 18-medal objective."""

    coverage = [task.competition_id for task in CAMPAIGN_TASKS if task.purpose == "official_coverage"]
    recovery = [
        task.competition_id
        for task in CAMPAIGN_TASKS
        if task.purpose == "official_medal_conversion"
    ]
    if len({task.competition_id for task in CAMPAIGN_TASKS}) != len(CAMPAIGN_TASKS):
        raise RuntimeError("Campaign queue contains duplicate competition IDs")
    remaining = [str(value) for value in progress.get("remaining_competition_ids", [])]
    missing_coverage = sorted(set(remaining) - set(coverage))
    if missing_coverage:
        raise RuntimeError("Campaign queue omits currently unscored Lite competitions")
    planned_recovery = [str(value) for value in recovery_plan.get("priority_order", [])]
    if recovery != planned_recovery:
        raise RuntimeError("Campaign recovery queue diverges from the gpt-5.6-sol priority order")
    current_medals = int(progress.get("any_medal_count", 0))
    scored_medals = {
        str(item.get("competition_id"))
        for item in progress.get("results", [])
        if item.get("any_medal") is True
    }
    available_conversions = [value for value in recovery if value not in scored_medals]
    maximum_reachable = current_medals + len(remaining) + len(available_conversions)
    target = int(progress.get("medals_required_to_strictly_exceed_top", 18))
    if maximum_reachable < target:
        raise RuntimeError("Campaign queue is mathematically unable to reach the medal target")
    return {
        "current_medals": current_medals,
        "remaining_unscored": len(remaining),
        "available_conversion_targets": len(available_conversions),
        "maximum_reachable_medals": maximum_reachable,
        "target_medals": target,
        "reachable": True,
        "coverage_queue": coverage,
        "recovery_queue": recovery,
        "gpt56_priority_order_preserved": True,
    }


def current_campaign_reachability() -> dict[str, Any]:
    progress = json.loads(PROGRESS_JSON.read_text(encoding="utf-8"))
    recovery_plan = json.loads(RECOVERY_PLAN.read_text(encoding="utf-8"))
    return campaign_reachability(progress, recovery_plan)


def outcomes_for_seed(state: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    if seed == PRIMARY_SEED:
        return state.get("outcomes", [])
    return (state.get("confirmation_outcomes") or {}).get(str(seed), [])


def outcome_completes_task(item: dict[str, Any]) -> bool:
    if item.get("finalized") is not True or item.get("retry_required") is True:
        return False
    return bool(
        int(item.get("official_grade_count") or 0) > 0
        or item.get("promotion_gate_withheld") is True
        or item.get("evidence_stop")
    )


def completed_competitions(state: dict[str, Any], seed: int = PRIMARY_SEED) -> set[str]:
    return {
        str(item["competition_id"])
        for item in outcomes_for_seed(state, seed)
        if outcome_completes_task(item)
    }


def next_task(state: dict[str, Any], seed: int = PRIMARY_SEED) -> CampaignTask | None:
    completed = completed_competitions(state, seed)
    outcomes = outcomes_for_seed(state, seed)
    reserved = (
        a800_reserved_competitions()
        if seed == PRIMARY_SEED
        and state.get("a800_reservation_coordination_enabled") is True
        else set()
    )
    for task in tasks_for_seed(seed):
        if task.competition_id in completed or task.competition_id in reserved:
            continue
        attempts = sum(
            1
            for item in outcomes
            if item.get("competition_id") == task.competition_id
            and item.get("finalized") is True
            and item.get("run_id")
        )
        if attempts >= MAX_TASK_ATTEMPTS:
            raise RuntimeError(
                f"Seed {seed} {task.competition_id} exhausted "
                f"{MAX_TASK_ATTEMPTS} recorded attempts without an official grade"
            )
        return task
    return None


def record_outcome(state: dict[str, Any], seed: int, outcome: dict[str, Any]) -> None:
    if seed == PRIMARY_SEED:
        state.setdefault("outcomes", []).append(outcome)
        return
    confirmation = state.setdefault("confirmation_outcomes", {})
    confirmation.setdefault(str(seed), []).append(outcome)


def prefetched_runs_for_seed(state: dict[str, Any], seed: int) -> dict[str, Any]:
    prefetched = state.setdefault("prefetched_runs", {})
    return prefetched.setdefault(str(seed), {})


def next_gpu_successor(
    state: dict[str, Any],
    seed: int,
    current_competition_id: str,
    cpu_concurrent_competitions: Iterable[str],
) -> CampaignTask | None:
    """Return the next unfinished GPU task after a trusted CPU-only task."""

    cpu_ids = set(cpu_concurrent_competitions)
    if current_competition_id not in cpu_ids:
        return None
    completed = completed_competitions(state, seed)
    found_current = False
    for task in tasks_for_seed(seed):
        if task.competition_id == current_competition_id:
            found_current = True
            continue
        if not found_current or task.competition_id in completed:
            continue
        if task.competition_id in cpu_ids:
            return None
        return task
    return None


def forget_prefetched_run(state: dict[str, Any], seed: int, competition_id: str) -> None:
    by_seed = (state.get("prefetched_runs") or {}).get(str(seed)) or {}
    by_seed.pop(competition_id, None)


def maybe_prelaunch_gpu_successor(
    remote: Any,
    state: dict[str, Any],
    args: argparse.Namespace,
    *,
    seed: int,
    current_task: CampaignTask,
) -> dict[str, Any] | None:
    """Fill an idle GPU while the owned CPU-only task continues to run."""

    candidate = next_gpu_successor(
        state,
        seed,
        current_task.competition_id,
        remote.CPU_LIGHT_CONCURRENT_COMPETITIONS,
    )
    if candidate is None:
        return None
    prefetched = prefetched_runs_for_seed(state, seed)
    if candidate.competition_id in prefetched:
        return prefetched[candidate.competition_id]

    wait_for_idle(
        remote,
        state,
        poll_seconds=args.poll_seconds,
        gate_interval_seconds=args.gate_interval_seconds,
        max_idle_wait_minutes=args.max_idle_wait_minutes,
        reason=f"Waiting to prelaunch Seed {seed} {candidate.competition_id}",
    )
    run_id = (
        f"job89441_{candidate.slug}_s{seed}_concurrent_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    start_payload = retry_remote_call(
        lambda: remote.start_run(
            remote.DEFAULT_BUNDLE,
            remote.DEFAULT_GATE_REPORT,
            run_id=run_id,
            waves=[candidate.wave],
            competitions=[candidate.competition_id],
            seed=seed,
            optimization_plan_name=candidate.optimization_plan_name,
            resume=False,
            allow_concurrent_with_cpu_light=True,
            max_gate_age=600,
        ),
        state,
        status="prelaunching_gpu_successor_retry",
        detail=f"Prelaunching Seed {seed} {candidate.competition_id}",
        event="concurrent_remote_start_failed",
        failure_key="concurrent_start_failures",
        poll_seconds=args.run_poll_seconds,
    )
    record = {
        "competition_id": candidate.competition_id,
        "run_id": run_id,
        "seed": seed,
        "wave": candidate.wave,
        "started_at": utc_now(),
        "passed": start_payload.get("passed") is True,
        "idempotent_reuse": bool(
            (start_payload.get("remote") or {}).get("idempotent_reuse")
        ),
        "concurrent_with": current_task.competition_id,
    }
    prefetched[candidate.competition_id] = record
    state["latest_concurrent_start"] = record
    write_state(
        state,
        "concurrent_remote_run_started",
        f"CPU {current_task.competition_id}; GPU {candidate.competition_id}",
    )
    append_log("concurrent_remote_run_started", **record)
    return record


def validated_primary_recovery_stops(
    candidate_gates: Path = CANDIDATE_GATES,
) -> dict[str, dict[str, Any]]:
    """Load evidence-backed Seed 42 recovery stops and verify their artifacts."""

    payload = json.loads(candidate_gates.read_text(encoding="utf-8"))
    recovery_ids = {
        task.competition_id
        for task in CAMPAIGN_TASKS
        if task.purpose == "official_medal_conversion"
    }
    stops: dict[str, dict[str, Any]] = {}
    for item in payload.get("not_promoted", []):
        if item.get("stop_rule_triggered") is not True:
            continue
        competition_id = str(item.get("competition_id") or "")
        if competition_id not in recovery_ids:
            raise RuntimeError("Evidence stop targets a task outside the Seed 42 recovery queue")
        evidence = item.get("evidence") or {}
        relative_path = Path(str(evidence.get("path") or ""))
        expected_sha = str(evidence.get("sha256") or "").lower()
        evidence_path = relative_path if relative_path.is_absolute() else PROJECT_ROOT / relative_path
        if not evidence_path.is_file() or len(expected_sha) != 64:
            raise RuntimeError("Evidence stop artifact is missing or has no SHA256")
        actual_sha = sha256_file(evidence_path)
        if actual_sha != expected_sha:
            raise RuntimeError("Evidence stop artifact SHA256 mismatch")
        stops[competition_id] = {
            "competition_id": competition_id,
            "reason": str(item.get("reason") or "bounded_public_oof_stop_rule"),
            "evidence": {
                "path": str(relative_to_project(evidence_path)),
                "sha256": actual_sha,
            },
        }
    return stops


def apply_primary_recovery_stops(
    state: dict[str, Any],
    stops: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Finalize bounded recovery stops without creating an official grade."""

    completed = completed_competitions(state, PRIMARY_SEED)
    applied: list[dict[str, Any]] = []
    for task in CAMPAIGN_TASKS:
        stop = stops.get(task.competition_id)
        if stop is None or task.competition_id in completed:
            continue
        outcome = {
            "competition_id": task.competition_id,
            "seed": PRIMARY_SEED,
            "wave": task.wave,
            "purpose": task.purpose,
            "run_id": None,
            "finalized": True,
            "status": "evidence_stop_rule_triggered",
            "official_grade_count": 0,
            "official_grader_executed": False,
            "official_results": [],
            "evidence_stop": stop,
            "completed_at": utc_now(),
        }
        record_outcome(state, PRIMARY_SEED, outcome)
        append_log("task_skipped_by_evidence_stop", outcome=outcome)
        applied.append(outcome)
    if stops and isinstance(state.get("reachability"), dict):
        reachability = dict(state["reachability"])
        previously_applied = {
            str(value)
            for value in reachability.get("evidence_stopped_recovery_targets", [])
        }
        stopped_ids = [
            task.competition_id
            for task in CAMPAIGN_TASKS
            if task.competition_id in stops
        ]
        newly_applied_count = len(set(stopped_ids) - previously_applied)
        reachability["evidence_stopped_recovery_targets"] = stopped_ids
        reachability["available_conversion_targets"] = max(
            0,
            int(reachability.get("available_conversion_targets", 0)) - newly_applied_count,
        )
        reachability["maximum_reachable_medals"] = max(
            0,
            int(reachability.get("maximum_reachable_medals", 0)) - newly_applied_count,
        )
        reachability["reachable"] = (
            reachability["maximum_reachable_medals"]
            >= int(reachability.get("target_medals", MEDALS_REQUIRED))
        )
        if not reachability["reachable"]:
            raise RuntimeError("Evidence stops make the official medal target unreachable")
        state["reachability"] = reachability
    return applied


def remote_status_finished(payload: dict[str, Any]) -> bool:
    """Only a stopped process is final; a written summary may precede exit."""

    return payload.get("process") == "stopped"


def is_transient_remote_exception(exc: BaseException) -> bool:
    """Classify transport failures without persisting exception text or secrets."""

    return isinstance(exc, (ConnectionError, TimeoutError)) or (
        type(exc).__name__ in TRANSIENT_REMOTE_ERROR_NAMES
    )


def retry_remote_call(
    operation: Any,
    state: dict[str, Any],
    *,
    status: str,
    detail: str,
    event: str,
    failure_key: str,
    poll_seconds: int,
) -> Any:
    """Retry transient SSH/control failures while keeping state evidence current."""

    failures = 0
    while True:
        try:
            result = operation()
            state.pop(failure_key, None)
            state.pop("last_error_type", None)
            return result
        except Exception as exc:
            if not is_transient_remote_exception(exc):
                raise
            failures += 1
            state[failure_key] = failures
            state["last_error_type"] = type(exc).__name__
            write_state(state, status, detail)
            append_log(
                event,
                error_type=type(exc).__name__,
                failures=failures,
            )
            time.sleep(min(poll_seconds * max(1, failures), 600))


def performance_evidence_from_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Return compact per-task runtime and accelerator evidence without raw samples."""

    evidence: list[dict[str, Any]] = []
    for result in summary.get("results", []):
        telemetry = result.get("gpu_telemetry") or {}
        evidence.append(
            {
                "competition_id": result.get("competition_id"),
                "runtime_seconds_total": result.get("runtime_seconds_total"),
                "runtime_seconds_model": result.get("runtime_seconds_model"),
                "runtime_seconds_feature_engineering": result.get(
                    "runtime_seconds_feature_engineering"
                ),
                "model_family": result.get("model_family"),
                "gpu_sample_count": telemetry.get("sample_count"),
                "gpu_peak_memory_used_mib": telemetry.get("peak_memory_used_mib"),
                "gpu_peak_utilization_percent": telemetry.get(
                    "peak_utilization_gpu_percent"
                ),
                "torch_peak_memory_allocated_mib": result.get(
                    "torch_peak_memory_allocated_mib"
                ),
            }
        )
    return evidence


def official_results_from_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for result in summary.get("results", []):
        score = result.get("mle_private_grader_score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            continue
        if not math.isfinite(float(score)):
            continue
        if result.get("valid_submission") is not True:
            continue
        if result.get("official_grader_executed") is not True:
            continue
        valid.append(result)
    return valid


def intentional_promotion_withholds(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Return bounded branches that deliberately withheld private grading."""

    return [
        result
        for result in summary.get("results", [])
        if result.get("official_grader_withheld") is True
        and isinstance(result.get("promotion_gate"), dict)
        and result["promotion_gate"].get("passed") is False
    ]


def summarize_gate(gate: dict[str, Any]) -> dict[str, Any]:
    samples = []
    for item in gate.get("samples", []):
        samples.append(
            {
                "captured_at": item.get("captured_at"),
                "gpu_name": item.get("gpu_name"),
                "memory_used_mib": item.get("memory_used_mib"),
                "utilization_percent": item.get("utilization_percent"),
                "compute_process_count": len(item.get("compute_apps") or []),
                "idle": item.get("idle") is True,
            }
        )
    return {
        "created_at": gate.get("created_at"),
        "passed": gate.get("passed") is True,
        "dedicated_root_writable": gate.get("dedicated_root_writable") is True,
        "samples": samples,
    }


def write_state(state: dict[str, Any], status: str, detail: str) -> None:
    state["status"] = status
    state["detail"] = detail
    state["updated_at"] = utc_now()
    state["supervisor_pid"] = os.getpid()
    state["human_gate_preserved"] = True
    state["kaggle_submission_enabled"] = False
    atomic_write_json(STATE_PATH, state)


def wait_for_idle(
    remote: Any,
    state: dict[str, Any],
    *,
    poll_seconds: int,
    gate_interval_seconds: int,
    max_idle_wait_minutes: int,
    reason: str,
) -> dict[str, Any]:
    deadline = time.monotonic() + max_idle_wait_minutes * 60
    failures = 0
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError("GPU idle gate wait reached its configured deadline")
        try:
            gate = remote.sample_gpu_idle_gate(interval_seconds=gate_interval_seconds)
            atomic_write_json(remote.DEFAULT_GATE_REPORT, gate)
            state["latest_gpu_gate"] = summarize_gate(gate)
            failures = 0
            state.pop("gate_probe_failures", None)
            state.pop("last_error_type", None)
        except Exception as exc:  # sanitized state/log: exception text is excluded
            failures += 1
            state["gate_probe_failures"] = failures
            state["last_error_type"] = type(exc).__name__
            write_state(state, "waiting_for_gpu_probe", reason)
            append_log("gpu_gate_probe_failed", error_type=type(exc).__name__, failures=failures)
            time.sleep(min(poll_seconds * max(1, failures), 300))
            continue
        if gate.get("passed") is True:
            write_state(state, "gpu_idle_gate_passed", reason)
            append_log("gpu_idle_gate_passed", reason=reason)
            return gate
        write_state(state, "waiting_for_gpu_idle", reason)
        append_log("gpu_busy", reason=reason, gate=state["latest_gpu_gate"])
        time.sleep(poll_seconds)


def plan_evidence() -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for label, path in (
        ("wave1", WAVE1_PLAN),
        ("wave2", WAVE2_PLAN),
        ("medal_recovery", RECOVERY_PLAN),
    ):
        evidence[label] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(path),
        }
        payload = json.loads(path.read_text(encoding="utf-8"))
        evidence[label]["provider"] = (payload.get("planner") or {}).get("provider")
        evidence[label]["model"] = (payload.get("planner") or {}).get("model")
        evidence[label]["status"] = payload.get("status")
    return evidence


def llm_runtime_evidence() -> dict[str, Any]:
    gateway = json.loads(OPENAI_GATEWAY_EVIDENCE.read_text(encoding="utf-8"))
    native = json.loads(NATIVE_TOOL_LOOP_EVIDENCE.read_text(encoding="utf-8"))
    execution = native.get("llm_execution") or {}
    gateway_passed = bool(
        gateway.get("schema") == "evomind.openai_gateway_smoke.v1"
        and gateway.get("status") == "passed"
        and gateway.get("ok") is True
        and gateway.get("requested_model") == REQUIRED_LLM_MODEL
        and (gateway.get("models") or {}).get("requested_model_present") is True
        and (gateway.get("nonstream") or {}).get("served_model") == REQUIRED_LLM_MODEL
        and (gateway.get("stream") or {}).get("ok") is True
        and (gateway.get("tool_call") or {}).get("ok") is True
    )
    native_passed = bool(
        native.get("schema") == "evomind.native_tool_loop_smoke.v1"
        and native.get("status") == "passed"
        and native.get("ok") is True
        and native.get("requested_model") == REQUIRED_LLM_MODEL
        and execution.get("provider") == "openai"
        and execution.get("model") == REQUIRED_LLM_MODEL
        and execution.get("native_tool_loop") is True
        and int(execution.get("native_tool_calls", 0)) >= 1
        and int(execution.get("tool_rounds", 0)) >= 2
    )
    if not gateway_passed or not native_passed:
        raise RuntimeError("Pinned gpt-5.6-sol runtime evidence is incomplete")
    return {
        "passed": True,
        "required_model": REQUIRED_LLM_MODEL,
        "gateway": {
            "path": str(relative_to_project(OPENAI_GATEWAY_EVIDENCE)),
            "sha256": sha256_file(OPENAI_GATEWAY_EVIDENCE),
            "created_at": gateway.get("created_at"),
            "served_model": (gateway.get("nonstream") or {}).get("served_model"),
            "stream_first_delta_ms": (gateway.get("stream") or {}).get("first_delta_ms"),
            "tool_call_count": (gateway.get("tool_call") or {}).get("tool_call_count"),
        },
        "native_tool_loop": {
            "path": str(relative_to_project(NATIVE_TOOL_LOOP_EVIDENCE)),
            "sha256": sha256_file(NATIVE_TOOL_LOOP_EVIDENCE),
            "created_at": native.get("created_at"),
            "provider": execution.get("provider"),
            "model": execution.get("model"),
            "native_tool_calls": execution.get("native_tool_calls"),
            "tool_rounds": execution.get("tool_rounds"),
        },
    }


def data_inventory_evidence() -> dict[str, Any]:
    inventory = json.loads(DATA_INVENTORY_EVIDENCE.read_text(encoding="utf-8"))
    summary = inventory.get("summary") or {}
    rows = inventory.get("competitions") or []
    low_ids = {
        value.strip()
        for value in LOW_SPLIT.read_text(encoding="utf-8").splitlines()
        if value.strip()
    }
    row_ids = {
        str(row.get("competition_id") or "")
        for row in rows
        if isinstance(row, dict)
    }
    remote = inventory.get("remote") or {}
    source = inventory.get("source") or {}
    passed = bool(
        inventory.get("schema") == "evomind.mlebench_lite_inventory.v1"
        and int(source.get("competition_count", 0)) == LITE_TOTAL
        and len(rows) == LITE_TOTAL
        and len(row_ids) == LITE_TOTAL
        and row_ids == low_ids
        and int(summary.get("official_prepared", 0)) == LITE_TOTAL
        and int(summary.get("raw_present_unverified", -1)) == 0
        and int(summary.get("partial_by_size", -1)) == 0
        and int(summary.get("missing", -1)) == 0
        and all(
            isinstance(row, dict)
            and row.get("official_prepared") is True
            and row.get("status") == "official_prepared"
            for row in rows
        )
        and remote.get("root") == "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra"
        and int(remote.get("disk_free", 0)) > 0
    )
    if not passed:
        raise RuntimeError("Current HPC Lite-22 data inventory is incomplete")
    return {
        "passed": True,
        "path": str(relative_to_project(DATA_INVENTORY_EVIDENCE)),
        "sha256": sha256_file(DATA_INVENTORY_EVIDENCE),
        "created_at": inventory.get("created_at"),
        "competition_count": len(rows),
        "official_prepared": int(summary["official_prepared"]),
        "missing": int(summary["missing"]),
        "partial_by_size": int(summary["partial_by_size"]),
        "raw_present_unverified": int(summary["raw_present_unverified"]),
        "remote_root": remote.get("root"),
        "disk_free_bytes": int(remote["disk_free"]),
        "remote_repo_commit": source.get("remote_repo_commit"),
        "low_split_sha256": sha256_file(LOW_SPLIT),
    }


def relative_to_project(path: Path) -> Path:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        return resolved


def refresh_official_progress(summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    valid = official_results_from_summary(summary)
    if not valid:
        return {
            "updated": False,
            "reason": "no_valid_official_private_grade",
            "summary": str(relative_to_project(summary_path)),
        }

    progress = json.loads(PROGRESS_JSON.read_text(encoding="utf-8"))
    sources = [Path(item["path"]) for item in progress.get("sources", [])]
    relative_summary = relative_to_project(summary_path)
    if relative_summary not in sources:
        sources.append(relative_summary)
    missing = [str(path) for path in sources if not (PROJECT_ROOT / path).is_file()]
    if missing:
        raise RuntimeError("An authoritative progress source is missing")

    history = PROGRESS_JSON.parent / "history"
    history.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for current in (PROGRESS_JSON, PROGRESS_MD):
        if current.is_file():
            target = history / f"{current.stem}_{stamp}{current.suffix}"
            target.write_bytes(current.read_bytes())

    command = [
        sys.executable,
        str(SCRIPTS_ROOT / "build_mlebench_lite_progress_report.py"),
    ]
    for source in sources:
        command.extend(("--summary", str(source)))
    command.extend(
        (
            "--low-split",
            str(relative_to_project(LOW_SPLIT)),
            "--leaderboard-readme",
            str(relative_to_project(LEADERBOARD_README)),
            "--output-json",
            str(relative_to_project(PROGRESS_JSON)),
            "--output-md",
            str(relative_to_project(PROGRESS_MD)),
        )
    )
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "build_mlebench_candidate_gates.py"),
            "--root",
            str(PROJECT_ROOT),
            "--output",
            str(PROJECT_ROOT / "workspace" / "mlebench_plans" / "medal_candidate_gates_current.json"),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    updated = json.loads(PROGRESS_JSON.read_text(encoding="utf-8"))
    return {
        "updated": True,
        "summary": str(relative_summary),
        "official_results": [
            {
                "competition_id": item.get("competition_id"),
                "score": item.get("mle_private_grader_score"),
                "any_medal": bool(
                    ((item.get("private_grader") or {}).get("upstream_report") or {}).get("any_medal")
                ),
            }
            for item in valid
        ],
        "scored_competitions": updated.get("scored_competitions"),
        "any_medal_count": updated.get("any_medal_count"),
        "target_medals": updated.get("medals_required_to_strictly_exceed_top"),
    }


def seed42_confirmation_eligibility(progress: dict[str, Any]) -> dict[str, Any]:
    scored = int(progress.get("scored_competitions", 0))
    valid = int(progress.get("valid_official_grades", 0))
    remaining = int(progress.get("remaining_competitions", LITE_TOTAL))
    medals = int(progress.get("any_medal_count", 0))
    coverage_complete = scored == LITE_TOTAL and remaining == 0
    official_complete = coverage_complete and valid == LITE_TOTAL
    medal_target_met = medals >= MEDALS_REQUIRED
    return {
        "seed": PRIMARY_SEED,
        "scored_competitions": scored,
        "valid_official_grades": valid,
        "remaining_competitions": remaining,
        "any_medal_count": medals,
        "coverage_complete": coverage_complete,
        "official_complete": official_complete,
        "medal_target_met": medal_target_met,
        "eligible_for_confirmation_seeds": official_complete and medal_target_met,
    }


def refresh_confirmation_progress(state: dict[str, Any], seed: int) -> dict[str, Any]:
    if seed not in CONFIRMATION_SEEDS:
        raise ValueError("Only confirmation seeds have isolated progress reports")
    summary_paths: list[Path] = []
    for outcome in outcomes_for_seed(state, seed):
        value = ((outcome.get("collection") or {}).get("summary_path"))
        if not value:
            raise RuntimeError("A finalized confirmation outcome has no summary path")
        path = Path(str(value))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise RuntimeError("A confirmation summary is missing")
        summary_paths.append(path)
    if not summary_paths:
        return {"updated": False, "reason": "no_confirmation_summaries", "seed": seed}

    output_json, output_md = seed_progress_paths(seed)
    command = [
        sys.executable,
        str(SCRIPTS_ROOT / "build_mlebench_lite_progress_report.py"),
    ]
    for path in summary_paths:
        command.extend(("--summary", str(relative_to_project(path))))
    command.extend(
        (
            "--low-split",
            str(relative_to_project(LOW_SPLIT)),
            "--leaderboard-readme",
            str(relative_to_project(LEADERBOARD_README)),
            "--output-json",
            str(relative_to_project(output_json)),
            "--output-md",
            str(relative_to_project(output_md)),
        )
    )
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    report = json.loads(output_json.read_text(encoding="utf-8"))
    report["seed"] = seed
    report["campaign_id"] = state.get("campaign_id")
    atomic_write_json(output_json, report)
    return {
        "updated": True,
        "seed": seed,
        "summary_count": len(summary_paths),
        "report": str(relative_to_project(output_json)),
        "scored_competitions": report.get("scored_competitions"),
        "valid_official_grades": report.get("valid_official_grades"),
        "any_medal_count": report.get("any_medal_count"),
    }


def build_multiseed_report_payload(
    reports: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    seed_rows: list[dict[str, Any]] = []
    low_hashes: set[str] = set()
    leaderboard_hashes: set[str] = set()
    for seed in ALL_SEEDS:
        report = reports.get(seed)
        if report is None:
            seed_rows.append(
                {
                    "seed": seed,
                    "report_available": False,
                    "official_complete": False,
                    "target_met": False,
                }
            )
            continue
        scored = int(report.get("scored_competitions", 0))
        valid = int(report.get("valid_official_grades", 0))
        remaining = int(report.get("remaining_competitions", LITE_TOTAL))
        medals = int(report.get("any_medal_count", 0))
        official_complete = scored == LITE_TOTAL and valid == LITE_TOTAL and remaining == 0
        percentage = medals / LITE_TOTAL * 100.0
        low_hash = str((report.get("low_split_source") or {}).get("sha256") or "")
        leaderboard_hash = str(
            (report.get("leaderboard_source") or {}).get("sha256") or ""
        )
        if low_hash:
            low_hashes.add(low_hash)
        if leaderboard_hash:
            leaderboard_hashes.add(leaderboard_hash)
        seed_rows.append(
            {
                "seed": seed,
                "report_available": True,
                "scored_competitions": scored,
                "valid_official_grades": valid,
                "remaining_competitions": remaining,
                "any_medal_count": medals,
                "any_medal_percentage": percentage,
                "official_complete": official_complete,
                "target_met": official_complete and medals >= MEDALS_REQUIRED,
                "low_split_sha256": low_hash,
                "leaderboard_source_sha256": leaderboard_hash,
            }
        )

    source_contract_consistent = (
        len(reports) == len(ALL_SEEDS)
        and len(low_hashes) == 1
        and len(leaderboard_hashes) == 1
    )
    leaderboard_comparable = source_contract_consistent and all(
        row["official_complete"] for row in seed_rows
    )
    stable_target_met = leaderboard_comparable and all(row["target_met"] for row in seed_rows)
    percentages = [
        float(row["any_medal_percentage"])
        for row in seed_rows
        if row.get("official_complete") is True
    ]
    percentage_mean = (
        math.fsum(percentages) / len(percentages) if percentages else None
    )
    percentage_sample_sd = None
    percentage_sem = None
    if len(percentages) >= 2 and percentage_mean is not None:
        variance = math.fsum(
            (value - percentage_mean) ** 2 for value in percentages
        ) / (len(percentages) - 1)
        percentage_sample_sd = math.sqrt(variance)
        percentage_sem = percentage_sample_sd / math.sqrt(len(percentages))
    status = (
        "passed"
        if stable_target_met
        else "target_not_reproduced"
        if leaderboard_comparable
        else "incomplete"
    )
    return {
        "schema": "evomind.mlebench_lite.multiseed.v1",
        "created_at": utc_now(),
        "job": "89441",
        "seeds": seed_rows,
        "required_seeds": list(ALL_SEEDS),
        "total_competitions_per_seed": LITE_TOTAL,
        "medals_required_per_seed": MEDALS_REQUIRED,
        "target_percentage": MEDALS_REQUIRED / LITE_TOTAL * 100.0,
        "public_top_reference_percentage": 80.30,
        "source_contract_consistent": source_contract_consistent,
        "leaderboard_comparable": leaderboard_comparable,
        "stable_target_met": stable_target_met,
        "complete_seed_count": len(percentages),
        "minimum_complete_seed_percentage": min(percentages) if percentages else None,
        "mean_complete_seed_percentage": percentage_mean,
        "sample_sd_complete_seed_percentage": percentage_sample_sd,
        "sem_complete_seed_percentage": percentage_sem,
        "status": status,
        "human_gate_preserved": True,
        "kaggle_submission_enabled": False,
    }


def write_multiseed_report(state: dict[str, Any]) -> dict[str, Any]:
    reports: dict[int, dict[str, Any]] = {}
    report_paths: dict[str, str] = {}
    for seed in ALL_SEEDS:
        path, _ = seed_progress_paths(seed)
        if not path.is_file():
            continue
        reports[seed] = json.loads(path.read_text(encoding="utf-8"))
        report_paths[str(seed)] = str(relative_to_project(path))
    payload = build_multiseed_report_payload(reports)
    payload["campaign_id"] = state.get("campaign_id")
    payload["seed_report_paths"] = report_paths
    payload["frozen_configuration"] = {
        "bundle_sha256": (state.get("bundle") or {}).get("sha256"),
        "planner_evidence": state.get("planner_evidence"),
        "llm_runtime_evidence": state.get("llm_runtime_evidence"),
        "data_inventory_evidence": state.get("data_inventory_evidence"),
        "confirmation_queue_contract": state.get("confirmation_queue_contract"),
    }
    atomic_write_json(MULTISEED_JSON, payload)
    lines = [
        "# Job 89441 MLE-Bench Lite Multi-Seed Report",
        "",
        f"- Status: **{payload['status']}**",
        f"- Leaderboard comparable: **{payload['leaderboard_comparable']}**",
        f"- Stable target met: **{payload['stable_target_met']}**",
        f"- Target: **{MEDALS_REQUIRED}/{LITE_TOTAL} medals per seed**",
        (
            "- Complete-seed medal rate: "
            f"**mean {payload['mean_complete_seed_percentage']:.2f}% / "
            f"sample SD {payload['sample_sd_complete_seed_percentage']:.2f} pp / "
            f"SEM {payload['sem_complete_seed_percentage']:.2f} pp**"
            if payload["complete_seed_count"] >= 2
            else "- Complete-seed medal rate statistics: **pending at least two complete seeds**"
        ),
        "",
        "| Seed | Official coverage | Valid grades | Medals | Medal rate | Target |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in payload["seeds"]:
        rate = row.get("any_medal_percentage")
        lines.append(
            f"| {row['seed']} | {row.get('scored_competitions', 0)}/{LITE_TOTAL} | "
            f"{row.get('valid_official_grades', 0)}/{LITE_TOTAL} | "
            f"{row.get('any_medal_count', 0)}/{LITE_TOTAL} | "
            f"{rate:.2f}% | {'YES' if row.get('target_met') else 'NO'} |"
            if rate is not None
            else f"| {row['seed']} | 0/{LITE_TOTAL} | 0/{LITE_TOTAL} | 0/{LITE_TOTAL} | -- | NO |"
        )
    lines.extend(
        [
            "",
            "Only complete official private-grader evidence for all three frozen seeds is leaderboard comparable.",
            "No Kaggle submission was made.",
            "",
        ]
    )
    MULTISEED_MD.parent.mkdir(parents=True, exist_ok=True)
    MULTISEED_MD.write_text("\n".join(lines), encoding="utf-8")
    return payload


def monitor_and_finalize(
    remote: Any,
    state: dict[str, Any],
    task: CampaignTask,
    run_id: str,
    *,
    seed: int = PRIMARY_SEED,
    run_poll_seconds: int,
) -> dict[str, Any]:
    while True:
        status = retry_remote_call(
            lambda: remote.read_remote_status(run_id),
            state,
            status="monitoring_remote_retry",
            detail=f"Monitoring {task.competition_id}",
            event="remote_status_failed",
            failure_key="status_probe_failures",
            poll_seconds=run_poll_seconds,
        )

        summary = status.get("summary") or {}
        checkpoint = status.get("checkpoint") or {}
        state["latest_remote_status"] = {
            "run_id": run_id,
            "process": status.get("process"),
            "summary_status": summary.get("status"),
            "summary_passed": summary.get("passed"),
            "summary_failed": summary.get("failed"),
            "checkpoint_completed": checkpoint.get("completed"),
            "checkpoint_remaining": checkpoint.get("remaining"),
        }
        if seed == PRIMARY_SEED:
            integrate_external_a800_completions(state)
        write_state(state, "monitoring_remote_run", f"Monitoring {task.competition_id}")
        if remote_status_finished(status):
            break
        time.sleep(run_poll_seconds)

    summary = status.get("summary") or {}
    if not official_results_from_summary(summary) and not intentional_promotion_withholds(summary):
        write_state(state, "regrading_remote_run", f"Regrading {task.competition_id}")
        try:
            remote.regrade_run(remote.DEFAULT_BUNDLE, run_id, force=False)
            status = remote.read_remote_status(run_id)
            summary = status.get("summary") or {}
        except Exception as exc:  # retain the run and collect all available failure evidence
            state["last_regrade_error_type"] = type(exc).__name__
            append_log("remote_regrade_failed", run_id=run_id, error_type=type(exc).__name__)

    write_state(state, "collecting_remote_run", f"Collecting {task.competition_id}")
    collection = retry_remote_call(
        lambda: remote.collect_run(run_id, include_checkpoints=False),
        state,
        status="collecting_remote_retry",
        detail=f"Collecting {task.competition_id}",
        event="remote_collection_failed",
        failure_key="collection_failures",
        poll_seconds=run_poll_seconds,
    )
    local_root = Path(collection["local_root"])
    summary_path = local_root / "summary.json"
    progress_refresh: dict[str, Any]
    if summary_path.is_file() and seed == PRIMARY_SEED:
        try:
            progress_refresh = refresh_official_progress(summary_path)
        except Exception as exc:  # collection remains authoritative even if ledger refresh needs repair
            progress_refresh = {
                "updated": False,
                "reason": "progress_refresh_failed",
                "error_type": type(exc).__name__,
            }
            append_log("progress_refresh_failed", run_id=run_id, error_type=type(exc).__name__)
    elif summary_path.is_file():
        progress_refresh = {
            "updated": False,
            "reason": "confirmation_seed_refresh_deferred",
            "seed": seed,
        }
    else:
        progress_refresh = {"updated": False, "reason": "summary_not_collected"}

    valid = official_results_from_summary(summary)
    outcome = {
        "competition_id": task.competition_id,
        "seed": seed,
        "wave": task.wave,
        "purpose": task.purpose,
        "run_id": run_id,
        "finalized": True,
        "remote_summary_status": summary.get("status"),
        "remote_release_sha256": Path(
            str(((status.get("state") or {}).get("release") or "unknown"))
        ).name,
        "official_grade_count": len(valid),
        "promotion_gate_withheld": bool(intentional_promotion_withholds(summary)),
        "promotion_gate_results": [
            {
                "status": item.get("status"),
                "promotion_gate": item.get("promotion_gate"),
                "submission_path": item.get("submission_path"),
                "submission_sha256": item.get("submission_sha256"),
            }
            for item in intentional_promotion_withholds(summary)
        ],
        "official_results": [
            {
                "score": item.get("mle_private_grader_score"),
                "cv_score": item.get("cv_score"),
                "metric": item.get("metric"),
                "direction": item.get("direction"),
                "any_medal": bool(
                    ((item.get("private_grader") or {}).get("upstream_report") or {}).get("any_medal")
                ),
            }
            for item in valid
        ],
        "performance_evidence": performance_evidence_from_summary(summary),
        "collection": {
            "local_root": str(relative_to_project(local_root)),
            "summary_path": (
                str(relative_to_project(summary_path)) if summary_path.is_file() else None
            ),
            "file_count": collection.get("file_count"),
            "passed": collection.get("passed") is True,
        },
        "progress_refresh": progress_refresh,
        "completed_at": utc_now(),
    }
    append_log("task_finalized", outcome=outcome)
    return outcome


def integrate_outcome(
    state: dict[str, Any],
    seed: int,
    outcome: dict[str, Any],
) -> dict[str, Any]:
    record_outcome(state, seed, outcome)
    write_state(
        state,
        "task_collected",
        f"Collected Seed {seed} {outcome['competition_id']}",
    )
    if seed in CONFIRMATION_SEEDS:
        try:
            outcome["progress_refresh"] = refresh_confirmation_progress(state, seed)
        except Exception as exc:
            outcome["progress_refresh"] = {
                "updated": False,
                "reason": "confirmation_progress_refresh_failed",
                "error_type": type(exc).__name__,
                "seed": seed,
            }
            append_log(
                "confirmation_progress_refresh_failed",
                seed=seed,
                error_type=type(exc).__name__,
            )
    return outcome


def integrate_external_a800_completions(
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Merge collected A800 official grades while this supervisor owns the lock."""

    existing_run_ids = {
        str(item.get("run_id") or "")
        for item in outcomes_for_seed(state, PRIMARY_SEED)
        if item.get("run_id")
    }
    if not A800_WATCH_DIR.is_dir():
        return []

    integrated: list[dict[str, Any]] = []
    completion_paths = sorted(
        A800_WATCH_DIR.glob("*_completion.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    task_by_id = {
        task.competition_id: task for task in tasks_for_seed(PRIMARY_SEED)
    }
    for completion_path in completion_paths:
        try:
            completion = json.loads(completion_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if completion.get("schema") != "evomind.mlebench.a800_lane_watch.completion.v1":
            continue

        run_id = str(completion.get("run_id") or "")
        if (
            not run_id
            or run_id in existing_run_ids
            or Path(run_id).name != run_id
            or "/" in run_id
            or "\\" in run_id
        ):
            continue
        completion_results = completion.get("official_results") or []
        if (
            completion.get("process") == "running"
            or completion.get("summary_status") != "passed"
            or int(completion.get("official_grade_count") or 0) != 1
            or len(completion_results) != 1
            or (completion.get("collection") or {}).get("passed") is not True
        ):
            continue

        competition_id = str(completion_results[0].get("competition_id") or "")
        task = task_by_id.get(competition_id)
        if task is None:
            continue
        collection_root = (A800_COLLECTION_ROOT / run_id).resolve()
        try:
            collection_root.relative_to(A800_COLLECTION_ROOT.resolve())
        except ValueError:
            continue
        summary_path = collection_root / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        valid = official_results_from_summary(summary)
        if (
            summary.get("status") != "passed"
            or len(valid) != 1
            or valid[0].get("competition_id") != competition_id
        ):
            continue

        try:
            progress_refresh = refresh_official_progress(summary_path)
        except Exception as exc:
            append_log(
                "a800_progress_refresh_failed",
                run_id=run_id,
                competition_id=competition_id,
                error_type=type(exc).__name__,
            )
            continue

        outcome = {
            "competition_id": task.competition_id,
            "seed": PRIMARY_SEED,
            "wave": task.wave,
            "purpose": task.purpose,
            "run_id": run_id,
            "finalized": True,
            "remote_summary_status": summary.get("status"),
            "official_grade_count": len(valid),
            "promotion_gate_withheld": False,
            "promotion_gate_results": [],
            "official_results": [
                {
                    "score": item.get("mle_private_grader_score"),
                    "cv_score": item.get("cv_score"),
                    "metric": item.get("metric"),
                    "direction": item.get("direction"),
                    "any_medal": bool(
                        ((item.get("private_grader") or {}).get("upstream_report") or {}).get(
                            "any_medal"
                        )
                    ),
                }
                for item in valid
            ],
            "performance_evidence": performance_evidence_from_summary(summary),
            "collection": {
                "local_root": str(relative_to_project(collection_root)),
                "summary_path": str(relative_to_project(summary_path)),
                "file_count": int(
                    (completion.get("collection") or {}).get("file_count") or 0
                ),
                "passed": True,
            },
            "watcher": {
                "completion_path": str(relative_to_project(completion_path)),
                "completion_created_at": completion.get("created_at"),
            },
            "progress_refresh": progress_refresh,
            "completed_at": utc_now(),
            "external_lane": "A800",
        }
        integrate_outcome(state, PRIMARY_SEED, outcome)
        existing_run_ids.add(run_id)
        record = {
            "schema": "evomind.mlebench_campaign.external_lane_integration.v1",
            "created_at": utc_now(),
            "run_id": run_id,
            "competition_id": competition_id,
            "idempotent_reuse": False,
            "official_grade_count": len(valid),
            "score": valid[0].get("mle_private_grader_score"),
            "any_medal": outcome["official_results"][0]["any_medal"],
            "campaign_current_task_preserved": state.get("current_task"),
            "campaign_current_run_preserved": state.get("current_run_id"),
            "kaggle_submission_enabled": False,
        }
        A800_INTEGRATION_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write_json(A800_INTEGRATION_DIR / f"{run_id}.json", record)
        atomic_write_json(A800_INTEGRATION_DIR / "current.json", record)
        append_log("a800_outcome_integrated", **record)
        integrated.append(record)
    return integrated


def new_state(campaign_id: str, remote: Any) -> dict[str, Any]:
    bundle = remote.verify_local_bundle(remote.DEFAULT_BUNDLE)
    return {
        "schema": "evomind.mlebench_campaign.v2",
        "campaign_id": campaign_id,
        "created_at": utc_now(),
        "job": "89441",
        "seed": PRIMARY_SEED,
        "current_seed": PRIMARY_SEED,
        "queue": task_payloads(),
        "confirmation_queue": task_payloads(CONFIRMATION_TASKS),
        "confirmation_queue_contract": validate_confirmation_queue(),
        "outcomes": [],
        "confirmation_outcomes": {str(seed): [] for seed in CONFIRMATION_SEEDS},
        "current_task": None,
        "current_run_id": None,
        "bundle": {
            "path": str(relative_to_project(remote.DEFAULT_BUNDLE)),
            "sha256": bundle["sha256"],
            "passed": bundle["passed"],
        },
        "planner_evidence": plan_evidence(),
        "llm_runtime_evidence": llm_runtime_evidence(),
        "data_inventory_evidence": data_inventory_evidence(),
        "reachability": current_campaign_reachability(),
        "official_success_target": {
            "total_competitions": LITE_TOTAL,
            "medals_required": MEDALS_REQUIRED,
            "percentage": MEDALS_REQUIRED / LITE_TOTAL * 100.0,
            "public_top_reference_percentage": 80.30,
            "minimum_seeds_for_final_comparison": len(ALL_SEEDS),
            "confirmation_seeds": list(CONFIRMATION_SEEDS),
        },
    }


def run_seed_queue(
    remote: Any,
    state: dict[str, Any],
    args: argparse.Namespace,
    seed: int,
) -> None:
    state["current_seed"] = seed
    while True:
        if seed == PRIMARY_SEED:
            integrate_external_a800_completions(state)
        task = next_task(state, seed)
        if task is None:
            break
        prefetched = prefetched_runs_for_seed(state, seed).get(task.competition_id)
        if prefetched:
            run_id = str(prefetched["run_id"])
        else:
            wait_for_idle(
                remote,
                state,
                poll_seconds=args.poll_seconds,
                gate_interval_seconds=args.gate_interval_seconds,
                max_idle_wait_minutes=args.max_idle_wait_minutes,
                reason=f"Waiting to start Seed {seed} {task.competition_id}",
            )
            run_id = (
                f"job89441_{task.slug}_s{seed}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            )
        state["current_task"] = task.competition_id
        state["current_run_id"] = run_id
        state["current_run_started"] = bool(prefetched)
        if prefetched:
            write_state(
                state,
                "adopting_prefetched_remote_run",
                f"Adopting Seed {seed} {task.competition_id}",
            )
            append_log(
                "prefetched_remote_run_adopted",
                run_id=run_id,
                competition_id=task.competition_id,
                seed=seed,
            )
            start_payload = {"passed": True, "remote": {"idempotent_reuse": True}}
        else:
            write_state(
                state,
                "starting_remote_run",
                f"Starting Seed {seed} {task.competition_id}",
            )
            start_payload = retry_remote_call(
                lambda: remote.start_run(
                    remote.DEFAULT_BUNDLE,
                    remote.DEFAULT_GATE_REPORT,
                    run_id=run_id,
                    waves=[task.wave],
                    competitions=[task.competition_id],
                    seed=seed,
                    optimization_plan_name=task.optimization_plan_name,
                    resume=False,
                    allow_concurrent_with_cpu_light=False,
                    max_gate_age=600,
                ),
                state,
                status="starting_remote_retry",
                detail=f"Starting Seed {seed} {task.competition_id}",
                event="remote_start_failed",
                failure_key="start_failures",
                poll_seconds=args.run_poll_seconds,
            )
        state["current_run_started"] = True
        state["latest_remote_start"] = {
            "run_id": run_id,
            "passed": start_payload.get("passed") is True,
            "idempotent_reuse": bool(
                (start_payload.get("remote") or {}).get("idempotent_reuse")
            ),
        }
        write_state(
            state,
            "remote_run_started",
            f"Started Seed {seed} {task.competition_id}",
        )
        append_log(
            "remote_run_started",
            run_id=run_id,
            competition_id=task.competition_id,
            seed=seed,
        )
        maybe_prelaunch_gpu_successor(
            remote,
            state,
            args,
            seed=seed,
            current_task=task,
        )
        outcome = monitor_and_finalize(
            remote,
            state,
            task,
            run_id,
            seed=seed,
            run_poll_seconds=args.run_poll_seconds,
        )
        integrate_outcome(state, seed, outcome)
        forget_prefetched_run(state, seed, task.competition_id)
        state["current_task"] = None
        state["current_run_id"] = None
        state["current_run_started"] = False
        write_state(
            state,
            "task_finalized",
            f"Finalized Seed {seed} {task.competition_id}",
        )


def run_campaign(args: argparse.Namespace) -> int:
    remote = load_remote_ops()
    if args.profile_dir is not None:
        from src.research_agent_workstation.server.core.gpu_credentials import (
            connect_ssh,
        )
        from workspace.hpc.probe_hpc_gpu_profile import _load_profile

        profile = _load_profile(args.profile_dir.resolve())
        remote._connect = lambda: connect_ssh(profile, timeout=25)
    if args.resume and STATE_PATH.is_file():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        state["schema"] = "evomind.mlebench_campaign.v2"
        state["resumed_at"] = utc_now()
        state.setdefault("current_seed", PRIMARY_SEED)
        state["queue"] = task_payloads()
        state["confirmation_queue"] = task_payloads(CONFIRMATION_TASKS)
        state["confirmation_queue_contract"] = validate_confirmation_queue()
        state.setdefault(
            "confirmation_outcomes",
            {str(seed): [] for seed in CONFIRMATION_SEEDS},
        )
        for seed in CONFIRMATION_SEEDS:
            state["confirmation_outcomes"].setdefault(str(seed), [])
        state["reachability"] = current_campaign_reachability()
        state["planner_evidence"] = plan_evidence()
        state["llm_runtime_evidence"] = llm_runtime_evidence()
        state["data_inventory_evidence"] = data_inventory_evidence()
        bundle = remote.verify_local_bundle(remote.DEFAULT_BUNDLE)
        previous_bundle_sha = (state.get("bundle") or {}).get("sha256")
        state["bundle"] = {
            "path": str(relative_to_project(remote.DEFAULT_BUNDLE)),
            "sha256": bundle["sha256"],
            "passed": bundle["passed"],
        }
        if previous_bundle_sha != bundle["sha256"]:
            state["release_deployed"] = False
            state.pop("deployment", None)
            state.pop("cuda_smoke", None)
    else:
        campaign_id = args.campaign_id or datetime.now().strftime(
            "job89441_full_s42_%Y%m%d_%H%M%S"
        )
        state = new_state(campaign_id, remote)

    evidence_stops = validated_primary_recovery_stops()
    state["a800_reservation_coordination_enabled"] = bool(
        args.respect_a800_reservations
    )
    state["credential_profile"] = {
        "path": str(args.profile_dir.resolve()) if args.profile_dir else None,
        "secret_policy": "DPAPI only; no secret material persisted",
    }
    state["validated_primary_recovery_stops"] = list(evidence_stops.values())
    apply_primary_recovery_stops(state, evidence_stops)

    write_state(state, "initializing", "Initializing gated full campaign")
    append_log("campaign_started", campaign_id=state["campaign_id"], resume=bool(args.resume))

    write_state(state, "staging_vision_weights", "Verifying pinned vision weight cache")
    vision_weight_cache = remote.stage_vision_weights()
    state["vision_weight_cache"] = {
        "passed": vision_weight_cache.get("passed") is True,
        "schema": vision_weight_cache.get("schema"),
        "weight_cache": vision_weight_cache.get("weight_cache"),
    }
    write_state(state, "vision_weights_ready", "Pinned vision weight cache verified")

    current_task_id = state.get("current_task")
    current_run_id = state.get("current_run_id")
    active_seed = int(state.get("current_seed", PRIMARY_SEED))
    if (
        current_task_id
        and current_run_id
        and current_task_id not in completed_competitions(state, active_seed)
    ):
        task = next(
            item
            for item in tasks_for_seed(active_seed)
            if item.competition_id == current_task_id
        )
        if not state.get("current_run_started"):
            wait_for_idle(
                remote,
                state,
                poll_seconds=args.poll_seconds,
                gate_interval_seconds=args.gate_interval_seconds,
                max_idle_wait_minutes=args.max_idle_wait_minutes,
                reason=f"Waiting to resume start Seed {active_seed} {task.competition_id}",
            )
            start_payload = retry_remote_call(
                lambda: remote.start_run(
                    remote.DEFAULT_BUNDLE,
                    remote.DEFAULT_GATE_REPORT,
                    run_id=str(current_run_id),
                    waves=[task.wave],
                    competitions=[task.competition_id],
                    seed=active_seed,
                    optimization_plan_name=task.optimization_plan_name,
                    resume=False,
                    allow_concurrent_with_cpu_light=False,
                    max_gate_age=600,
                ),
                state,
                status="resuming_remote_start_retry",
                detail=f"Resuming start Seed {active_seed} {task.competition_id}",
                event="remote_resume_start_failed",
                failure_key="start_failures",
                poll_seconds=args.run_poll_seconds,
            )
            state["current_run_started"] = True
            state["latest_remote_start"] = {
                "run_id": str(current_run_id),
                "passed": start_payload.get("passed") is True,
                "idempotent_reuse": bool(
                    (start_payload.get("remote") or {}).get("idempotent_reuse")
                ),
            }
            write_state(
                state,
                "remote_run_started",
                f"Started Seed {active_seed} {task.competition_id}",
            )
        maybe_prelaunch_gpu_successor(
            remote,
            state,
            args,
            seed=active_seed,
            current_task=task,
        )
        outcome = monitor_and_finalize(
            remote,
            state,
            task,
            str(current_run_id),
            seed=active_seed,
            run_poll_seconds=args.run_poll_seconds,
        )
        integrate_outcome(state, active_seed, outcome)
        forget_prefetched_run(state, active_seed, task.competition_id)
        state["current_task"] = None
        state["current_run_id"] = None
        state["current_run_started"] = False
        write_state(
            state,
            "task_finalized",
            f"Finalized Seed {active_seed} {task.competition_id}",
        )

    if not state.get("release_deployed"):
        wait_for_idle(
            remote,
            state,
            poll_seconds=args.poll_seconds,
            gate_interval_seconds=args.gate_interval_seconds,
            max_idle_wait_minutes=args.max_idle_wait_minutes,
            reason="Waiting to deploy the pinned release",
        )
        write_state(state, "deploying_release", "Deploying the pinned release")
        deployment = remote.deploy_bundle(
            remote.DEFAULT_BUNDLE,
            remote.DEFAULT_GATE_REPORT,
            max_gate_age=600,
        )
        state["deployment"] = {
            "passed": deployment.get("passed") is True,
            "release": deployment.get("remote_release"),
        }
        wait_for_idle(
            remote,
            state,
            poll_seconds=args.poll_seconds,
            gate_interval_seconds=args.gate_interval_seconds,
            max_idle_wait_minutes=args.max_idle_wait_minutes,
            reason="Waiting for CUDA smoke",
        )
        write_state(state, "cuda_smoke", "Running CUDA smoke")
        smoke = remote.cuda_smoke(
            remote.DEFAULT_BUNDLE,
            remote.DEFAULT_GATE_REPORT,
            max_gate_age=600,
        )
        state["cuda_smoke"] = {
            "passed": smoke.get("passed") is True,
            "remote": smoke.get("smoke"),
        }
        state["release_deployed"] = True
        write_state(state, "release_ready", "Pinned release and CUDA smoke passed")

    run_seed_queue(remote, state, args, PRIMARY_SEED)

    write_state(
        state,
        "seed42_queue_completed",
        "All unscored competitions and all bounded medal conversions were finalized",
    )
    append_log("seed_queue_completed", campaign_id=state["campaign_id"], seed=PRIMARY_SEED)

    seed42_progress = json.loads(PROGRESS_JSON.read_text(encoding="utf-8"))
    eligibility = seed42_confirmation_eligibility(seed42_progress)
    state["seed42_confirmation_eligibility"] = eligibility
    state["multiseed_report"] = write_multiseed_report(state)
    if not eligibility["eligible_for_confirmation_seeds"]:
        write_state(
            state,
            "seed42_target_not_met",
            "Seed 42 requires 22 valid official grades and at least 18 medals before confirmation seeds",
        )
        append_log("confirmation_seeds_blocked", eligibility=eligibility)
        return 2

    state["confirmation_seeds_authorized"] = True
    write_state(
        state,
        "confirmation_seeds_ready",
        "Seed 42 official target passed; frozen Seeds 43 and 44 are authorized",
    )
    for seed in CONFIRMATION_SEEDS:
        if outcomes_for_seed(state, seed):
            refresh_confirmation_progress(state, seed)
        run_seed_queue(remote, state, args, seed)
        refresh_confirmation_progress(state, seed)
        state["multiseed_report"] = write_multiseed_report(state)
        write_state(
            state,
            "confirmation_seed_completed",
            f"Seed {seed} completed all {LITE_TOTAL} confirmation tasks",
        )
        append_log("seed_queue_completed", campaign_id=state["campaign_id"], seed=seed)

    multiseed = write_multiseed_report(state)
    state["multiseed_report"] = multiseed
    if multiseed["stable_target_met"]:
        write_state(
            state,
            "multiseed_campaign_completed",
            "All three frozen seeds exceeded the public Lite top reference",
        )
        append_log("campaign_completed", campaign_id=state["campaign_id"])
        return 0
    write_state(
        state,
        "multiseed_target_not_reproduced",
        "All configured seeds completed but the official target was not reproduced on every seed",
    )
    append_log("campaign_target_not_reproduced", campaign_id=state["campaign_id"])
    return 3


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--gate-interval-seconds", type=int, default=5)
    parser.add_argument("--run-poll-seconds", type=int, default=120)
    parser.add_argument("--max-idle-wait-minutes", type=int, default=2880)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--respect-a800-reservations", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if min(
        args.poll_seconds,
        args.gate_interval_seconds,
        args.run_poll_seconds,
        args.max_idle_wait_minutes,
    ) <= 0:
        parser.error("All timing arguments must be positive")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        payload = {
            "schema": "evomind.mlebench_campaign.dry_run.v2",
            "job": "89441",
            "primary_seed": PRIMARY_SEED,
            "primary_queue": task_payloads(),
            "primary_queue_count": len(CAMPAIGN_TASKS),
            "confirmation_seeds": list(CONFIRMATION_SEEDS),
            "confirmation_queue": task_payloads(CONFIRMATION_TASKS),
            "confirmation_queue_count": len(CONFIRMATION_TASKS),
            "confirmation_queue_contract": validate_confirmation_queue(),
            "confirmation_requires": {
                "scored_competitions": LITE_TOTAL,
                "valid_official_grades": LITE_TOTAL,
                "any_medal_count": MEDALS_REQUIRED,
            },
            "reachability": current_campaign_reachability(),
            "human_gate_preserved": True,
            "kaggle_submission_enabled": False,
            "planner_evidence": plan_evidence(),
            "llm_runtime_evidence": llm_runtime_evidence(),
            "data_inventory_evidence": data_inventory_evidence(),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    acquire_lock()
    try:
        return run_campaign(args)
    except Exception as exc:  # persistent sanitized failure evidence
        state: dict[str, Any]
        try:
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {"schema": "evomind.mlebench_campaign.v2", "job": "89441"}
        state["last_error_type"] = type(exc).__name__
        write_state(state, "failed", "Campaign supervisor stopped with a recorded error")
        append_log("campaign_failed", error_type=type(exc).__name__)
        return 1
    finally:
        release_lock()


if __name__ == "__main__":
    raise SystemExit(main())
