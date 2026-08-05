#!/usr/bin/env python3
"""Run a fail-closed, evidence-driven EvoMind MLE-Bench decision turn."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evomind_runtime import AgentRuntime  # noqa: E402

REQUIRED_MODEL = "gpt-5.6-sol"
DECISION_SCHEMA = "evomind.mlebench.adaptive_decision.v1"
REPORT_SCHEMA = "evomind.mlebench.gpt56_adaptive_loop.v1"
REQUIRED_EVIDENCE = (
    "official_progress",
    "campaign_state",
    "ranzcr_failure",
    "a800_status",
    "a800_parallel_status",
    "a40_taxi_status",
    "local4060_dog_status",
    "dog_gpt56_audit",
    "taxi_source_audit",
    "recovery_deep_review",
    "human_gate_candidates",
    "job88240_gpu_gate",
    "job89508_resource_probe",
    "job88240_serial_chain",
    "job89941_may_status",
    "job89941_taxi_cpu_candidate",
    "job89941_taxi_cpu_progress",
    "job89941_taxi_cpu_confirmation",
    "job89771_taxi_longhaul_expert",
    "job89771_taxi_midhaul_expert",
    "job89941_leaf_status",
    "cactus_cpu_sidecar_status",
    "cactus_human_gate_postrun",
    "ranzcr_human_gate_postrun",
    "siim_human_gate_postrun",
    "may2022_execution_queue",
    "may2022_supersession_seal",
    "staged_human_gate_candidates",
)
ALLOWED_ACTIONS = {
    "continue_active_runs",
    "queue_experiment",
    "collect_and_grade",
    "retry_failed_after_active",
    "hold_for_evidence",
}
ALLOWED_LANES = {"a800", "a40", "local4060", "cpu"}
ACTIVE_RUN_LANES = {"a800", "a800_parallel", "a40", "local4060", "cpu"}
ACTIVE_RUN_MAX_AGE_SECONDS = 15 * 60
RUNTIME_MAX_STEPS = max(12, len(REQUIRED_EVIDENCE) + 4)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json(path: Path) -> tuple[dict[str, Any], str]:
    data = path.read_bytes()
    value = json.loads(data.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value, _sha256(data)


def _source_record(path: Path, digest: str) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": digest}


def _brief_result(item: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "competition_id",
        "status",
        "metric",
        "direction",
        "cv_score",
        "mle_private_grader_score",
        "official_grader_executed",
        "official_grader_withheld",
        "score_scope",
        "proxy_score",
        "valid_submission",
        "any_medal",
        "gold_medal",
        "silver_medal",
        "bronze_medal",
        "bronze_threshold",
        "margin_to_bronze_positive_is_medal",
        "error_type",
        "failure_taxonomy",
        "validation_errors",
    )
    result = {key: item.get(key) for key in keys if key in item}
    grader = item.get("private_grader")
    if isinstance(grader, dict):
        result["private_grader"] = {key: grader.get(key) for key in ("status", "score", "error_type") if key in grader}
    gate = item.get("promotion_gate")
    if isinstance(gate, dict):
        result["promotion_gate"] = gate
    return result


def _normalise_progress(value: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "schema",
        "created_at",
        "lite_total_competitions",
        "scored_competitions",
        "remaining_competitions",
        "valid_official_grades",
        "any_medal_count",
        "gold_medal_count",
        "scored_subset_any_medal_percentage",
        "padded_full_lite_any_medal_percentage",
        "main_leaderboard_lite_top_percentage",
        "medals_required_to_strictly_exceed_top",
        "medal_deficit",
        "maximum_medals_if_all_unscored_tasks_medal",
        "minimum_scored_non_medals_that_must_also_be_converted",
        "scored_non_medal_count",
        "remaining_competition_ids",
        "claim_boundary",
    )
    return {
        "captured_at": _now(),
        "source": source,
        "progress": {key: value.get(key) for key in keys if key in value},
        "results": [_brief_result(item) for item in value.get("results", []) if isinstance(item, dict)],
    }


def _normalise_campaign(value: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    queue = []
    for item in value.get("queue", []):
        if isinstance(item, dict):
            queue.append(
                {
                    key: item.get(key)
                    for key in (
                        "competition_id",
                        "wave",
                        "slug",
                        "purpose",
                        "optimization_plan_name",
                    )
                    if key in item
                }
            )
    outcomes = []
    for item in value.get("outcomes", []):
        if not isinstance(item, dict):
            continue
        outcome = {
            key: item.get(key)
            for key in (
                "competition_id",
                "seed",
                "wave",
                "purpose",
                "run_id",
                "finalized",
                "remote_summary_status",
                "official_grade_count",
                "promotion_gate_withheld",
                "completed_at",
                "status",
                "official_grader_executed",
                "mle_private_grader_score",
                "error_type",
            )
            if key in item
        }
        collection = item.get("collection")
        if isinstance(collection, dict):
            outcome["collection"] = {
                key: collection.get(key)
                for key in ("passed", "summary_path", "file_count")
                if key in collection
            }
        progress_refresh = item.get("progress_refresh")
        if isinstance(progress_refresh, dict):
            outcome["progress_refresh"] = {
                key: progress_refresh.get(key)
                for key in ("updated", "reason", "summary")
                if key in progress_refresh
            }
        promotion_results = []
        for result in item.get("promotion_gate_results", []):
            if not isinstance(result, dict):
                continue
            gate = result.get("promotion_gate")
            brief_gate = {}
            if isinstance(gate, dict):
                brief_gate = {
                    key: gate.get(key)
                    for key in (
                        "schema",
                        "name",
                        "metric",
                        "direction",
                        "operator",
                        "threshold",
                        "fold_threshold",
                        "internal_score",
                        "passed",
                        "checks",
                        "claim_boundary",
                    )
                    if key in gate
                }
            promotion_results.append(
                {
                    "status": result.get("status"),
                    "promotion_gate": brief_gate,
                }
            )
        if promotion_results:
            outcome["promotion_gate_results"] = promotion_results
        outcomes.append(outcome)
    state_keys = (
        "schema",
        "campaign_id",
        "updated_at",
        "status",
        "detail",
        "seed",
        "current_seed",
        "current_task",
        "current_run_id",
        "official_success_target",
        "latest_remote_status",
        "human_gate_preserved",
        "kaggle_submission_enabled",
    )
    return {
        "captured_at": _now(),
        "source": source,
        "campaign": {key: value.get(key) for key in state_keys if key in value},
        "queue": queue,
        "outcomes": outcomes,
    }


def _blocked_collect_and_grade_competitions(campaign_snapshot: dict[str, Any]) -> set[str]:
    """Return latest collected outcomes that require a new experiment before grading."""

    latest: dict[str, dict[str, Any]] = {}
    for item in campaign_snapshot.get("outcomes", []):
        if not isinstance(item, dict):
            continue
        competition_id = str(item.get("competition_id") or "")
        if competition_id:
            latest[competition_id] = item
    return {
        competition_id
        for competition_id, item in latest.items()
        if item.get("promotion_gate_withheld") is True
        and int(item.get("official_grade_count") or 0) == 0
        and isinstance(item.get("collection"), dict)
        and item["collection"].get("passed") is True
    }


def _normalise_run_summary(value: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    return {
        "captured_at": _now(),
        "source": source,
        "summary": {
            key: value.get(key)
            for key in ("schema", "run_id", "status", "passed", "failed", "completed_at")
            if key in value
        },
        "results": [_brief_result(item) for item in value.get("results", []) if isinstance(item, dict)],
    }


def _normalise_a800(
    heartbeat: dict[str, Any],
    heartbeat_source: dict[str, Any],
    queue: dict[str, Any],
    queue_source: dict[str, Any],
) -> dict[str, Any]:
    return {
        "captured_at": _now(),
        "sources": {
            "heartbeat": heartbeat_source,
            "queue": queue_source,
        },
        "heartbeat": heartbeat,
        "queue": queue,
    }


def _normalise_a40(value: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    state = value.get("state") if isinstance(value.get("state"), dict) else {}
    summary = value.get("summary") if isinstance(value.get("summary"), dict) else None
    manifest = value.get("manifest") if isinstance(value.get("manifest"), dict) else {}
    checkpoint = (
        value.get("checkpoint") if isinstance(value.get("checkpoint"), dict) else None
    )
    compact_summary = None
    if summary is not None:
        compact_summary = {
            key: summary.get(key)
            for key in (
                "schema",
                "run_id",
                "status",
                "passed",
                "failed",
                "completed_at",
                "official_private_grader_rate",
                "valid_submission_rate",
                "claim_boundary",
            )
            if key in summary
        }
        compact_summary["results"] = [
            _brief_result(item)
            for item in summary.get("results", [])
            if isinstance(item, dict)
        ]
    return {
        "captured_at": _now(),
        "source": source,
        "process": value.get("process"),
        "log_tail": value.get("log_tail"),
        "checkpoint": (
            {
                key: checkpoint.get(key)
                for key in (
                    "schema",
                    "run_id",
                    "completed",
                    "remaining",
                    "requested",
                    "updated_at",
                )
                if key in checkpoint
            }
            if checkpoint is not None
            else None
        ),
        "summary": compact_summary,
        "run": {
            key: state.get(key)
            for key in ("run_id", "pid", "status", "performance_contract", "concurrent_with")
            if key in state
        },
        "manifest": {
            key: manifest.get(key)
            for key in (
                "run_id",
                "requested_competitions",
                "seed",
                "phase_a_status",
                "human_gate_preserved",
                "kaggle_submission_enabled",
            )
            if key in manifest
        },
    }


def _normalise_a800_parallel(value: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    return {
        "captured_at": _now(),
        "source": source,
        "heartbeat": value,
    }


def _normalise_local4060_dog(
    plan: dict[str, Any],
    plan_source: dict[str, Any],
    queue: dict[str, Any],
    queue_source: dict[str, Any],
) -> dict[str, Any]:
    return {
        "captured_at": _now(),
        "sources": {"recovery_plan": plan_source, "live_queue": queue_source},
        "competition_id": plan.get("competition_id"),
        "promotion_threshold": plan.get("promotion_threshold"),
        "active_parent": plan.get("active_parent"),
        "frozen_teacher_evidence": plan.get("frozen_teacher_evidence"),
        "material_revision": plan.get("material_revision"),
        "launch_policy": plan.get("launch_policy"),
        "live_queue": queue,
        "human_gate_preserved": plan.get("human_gate_preserved"),
        "kaggle_submission_enabled": plan.get("kaggle_submission_enabled"),
    }


def _normalise_dog_audit(value: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    audit = value.get("audit") if isinstance(value.get("audit"), dict) else {}
    return {
        "captured_at": _now(),
        "source": source,
        "schema": value.get("schema"),
        "created_at": value.get("created_at"),
        "ok": value.get("ok"),
        "planner": value.get("planner"),
        "truth_boundary_acknowledgement": audit.get("truth_boundary_acknowledgement"),
        "risk_assessment": audit.get("risk_assessment"),
        "bounded_experiments": audit.get("bounded_experiments"),
        "recommended_next_run": audit.get("recommended_next_run"),
        "campaign_impact": audit.get("campaign_impact"),
    }


def _normalise_recovery_deep_review(
    value: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    """Compact the latest full-source GPT-5.6 audit into controller evidence."""

    review = value.get("review") if isinstance(value.get("review"), dict) else {}
    audit = (
        review.get("audit_summary")
        if isinstance(review.get("audit_summary"), dict)
        else {}
    )
    competition_rows: list[dict[str, Any]] = []
    for item in review.get("per_competition", []):
        if not isinstance(item, dict):
            continue
        competition_rows.append({
            "competition_id": item.get("competition_id"),
            "deployment_verdict": item.get("deployment_verdict"),
            "estimated_medal_probability_after_upgrade": item.get(
                "estimated_medal_probability_after_upgrade"
            ),
            "compute_cost": item.get("compute_cost"),
            "stop_rule": item.get("stop_rule"),
            "confirmed_source_defects": list(item.get("confirmed_source_defects") or [])[:3],
            "highest_value_upgrades": list(item.get("highest_value_upgrades") or [])[:3],
            "validation_contract": list(item.get("validation_contract") or [])[:3],
        })
    return {
        "captured_at": _now(),
        "source": source,
        "schema": value.get("schema"),
        "created_at": value.get("created_at"),
        "parse_ok": value.get("parse_ok"),
        "ok": value.get("ok"),
        "planner": value.get("planner"),
        "audit_summary": {
            "current_coverage": audit.get("current_coverage"),
            "required_coverage": audit.get("required_coverage"),
            "critical_findings": list(audit.get("critical_findings") or []),
        },
        "per_competition": competition_rows,
        "implementation_order": review.get("implementation_order"),
        "exact_medal_paths": review.get("exact_medal_paths"),
        "resource_schedule": review.get("resource_schedule"),
        "global_stop_rules": review.get("global_stop_rules"),
    }


def _normalise_human_gate_candidates(root: Path) -> dict[str, Any]:
    """Collect immutable, verified candidate packages without reading predictions."""

    candidates: list[dict[str, Any]] = []
    sources: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted(Path(root).glob("*/manifest.json")):
        manifest, manifest_sha = _load_json(manifest_path)
        package_dir = manifest_path.parent
        verification_path = package_dir / "package_verification.json"
        verification: dict[str, Any] = {}
        verification_sha = ""
        if verification_path.is_file():
            verification, verification_sha = _load_json(verification_path)
        sources[f"{package_dir.name}/manifest"] = _source_record(
            manifest_path, manifest_sha
        )
        if verification_sha:
            sources[f"{package_dir.name}/verification"] = _source_record(
                verification_path, verification_sha
            )
        files = manifest.get("files") if isinstance(manifest.get("files"), list) else []
        submission = next(
            (
                item
                for item in files
                if isinstance(item, dict)
                and str(item.get("path") or item.get("name") or "").endswith(
                    "candidate_submission_withheld.csv"
                )
            ),
            None,
        )
        candidates.append(
            {
                "package": package_dir.name,
                "competition_id": manifest.get("competition_id"),
                "status": manifest.get("status"),
                "created_at": manifest.get("created_at"),
                "candidate_ready_for_human_gate": manifest.get(
                    "candidate_ready_for_human_gate"
                ),
                "public_oof_metrics": manifest.get("public_oof_metrics"),
                "package_verification_status": verification.get("status"),
                "package_verification_passed": (
                    verification.get("status") == "verified"
                    and verification.get("candidate_csv_present") is True
                    and verification.get("independent_verification_passed") is True
                ),
                "candidate_submission": submission,
                "automatic_submission": manifest.get("automatic_submission"),
                "private_labels_used": manifest.get("private_labels_used"),
                "official_grader_executed": manifest.get(
                    "official_grader_executed"
                ),
                "kaggle_submission_executed": manifest.get(
                    "kaggle_submission_executed"
                ),
                "claim_boundary": manifest.get("claim_boundary"),
            }
        )
    return {
        "captured_at": _now(),
        "sources": sources,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "rule": (
            "A verified Human Gate candidate is gradeable evidence, not an official "
            "score or medal; execution still requires explicit Human Gate approval."
        ),
    }


def _normalise_gpu_gate(
    value: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    samples = []
    for item in value.get("samples", []):
        if not isinstance(item, dict):
            continue
        samples.append(
            {
                key: item.get(key)
                for key in (
                    "index",
                    "captured_at",
                    "gpu_name",
                    "memory_used_mib",
                    "utilization_percent",
                    "compute_apps",
                    "idle",
                )
                if key in item
            }
        )
    return {
        "captured_at": _now(),
        "source": source,
        "schema": value.get("schema"),
        "observed_at": value.get("created_at"),
        "passed": value.get("passed"),
        "read_only_gate_passed": value.get("read_only_gate_passed"),
        "dedicated_root_writable": value.get("dedicated_root_writable"),
        "preflight": value.get("preflight"),
        "policy": value.get("policy"),
        "samples": samples,
        "rule": "Do not launch GPU work while this fresh read-only gate is false.",
    }


def _normalise_job89508_resource_probe(
    value: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    training_gate = (
        value.get("training_gate") if isinstance(value.get("training_gate"), dict) else {}
    )
    authoritative_gate = (
        training_gate.get("authoritative_three_sample_gate")
        if isinstance(training_gate.get("authoritative_three_sample_gate"), dict)
        else {}
    )
    gpu = value.get("gpu") if isinstance(value.get("gpu"), dict) else {}
    identity = value.get("identity") if isinstance(value.get("identity"), dict) else {}
    runtime = value.get("runtime") if isinstance(value.get("runtime"), dict) else {}

    samples = []
    for item in authoritative_gate.get("samples", []):
        if not isinstance(item, dict):
            continue
        samples.append(
            {
                key: item.get(key)
                for key in (
                    "captured_at",
                    "memory_used_mib",
                    "utilization_gpu_percent",
                    "compute_apps",
                    "idle",
                )
                if key in item
            }
        )

    existing_processes = []
    for item in value.get("existing_processes", []):
        if not isinstance(item, dict):
            continue
        existing_processes.append(
            {
                key: item.get(key)
                for key in (
                    "pid",
                    "state",
                    "cpu_percent_probe",
                    "command",
                    "cuda_device",
                    "nvidia_fd",
                    "classification",
                    "do_not_touch",
                )
                if key in item
            }
        )

    return {
        "captured_at": _now(),
        "source": source,
        "schema": value.get("schema"),
        "observed_at": value.get("captured_at"),
        "job_id": value.get("job_id"),
        "identity": {
            "user": identity.get("user"),
            "hostname": identity.get("hostname"),
            "remote_root": identity.get("remote_root"),
            "root_exists": identity.get("root_exists"),
            "root_readable": identity.get("root_readable"),
            "root_writable": identity.get("root_writable"),
        },
        "gpu": {
            "index": gpu.get("index"),
            "name": gpu.get("name"),
            "memory_total_mib": gpu.get("memory_total_mib"),
            "memory_used_mib": gpu.get("memory_used_mib"),
            "memory_free_mib": gpu.get("memory_free_mib"),
            "utilization_gpu_percent": gpu.get("utilization_gpu_percent"),
            "compute_apps": gpu.get("compute_apps"),
            "device_nodes": gpu.get("device_nodes"),
        },
        "existing_processes": existing_processes,
        "runtime": {
            "default_python": runtime.get("default_python"),
            "default_user_site_numpy": runtime.get("default_user_site_numpy"),
            "default_torch_import": runtime.get("default_torch_import"),
            "alternate_python": runtime.get("alternate_python"),
            "alternate_torch": runtime.get("alternate_torch"),
            "alternate_cuda_available": runtime.get("alternate_cuda_available"),
            "unified_runtime_verified": runtime.get("unified_runtime_verified"),
        },
        "training_gate": {
            "training_start_allowed": training_gate.get("training_start_allowed"),
            "passed": authoritative_gate.get("passed"),
            "samples_required": authoritative_gate.get("samples_required"),
            "sample_interval_seconds": authoritative_gate.get(
                "sample_interval_seconds"
            ),
            "compute_apps_required_empty": authoritative_gate.get(
                "compute_apps_required_empty"
            ),
            "reason_codes": training_gate.get("reason_codes"),
            "signals_sent": training_gate.get("signals_sent"),
            "remote_files_modified": training_gate.get("remote_files_modified"),
            "parallel_training_started": training_gate.get(
                "parallel_training_started"
            ),
            "samples": samples,
        },
        "rule": (
            "Job 89508 may not start parallel A800 training until compute apps are empty, "
            "NVIDIA device handles are released, the unified runtime is verified, and a "
            "fresh three-sample gate reports training_start_allowed=true."
        ),
    }


def _normalise_job88240_serial_chain(
    value: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    stages = []
    for item in value.get("statuses", []):
        if not isinstance(item, dict):
            continue
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        stages.append(
            {
                "path": item.get("path"),
                "exists": item.get("exists"),
                "schema": payload.get("schema"),
                "observed_at": payload.get("created_at"),
                "status": payload.get("status"),
                "wrapper_pid": payload.get("wrapper_pid"),
                "run_id": payload.get("run_id"),
                "completed_seeds": payload.get("completed_seeds"),
                "exit_code": payload.get("exit_code"),
                "process_signals_sent": payload.get("process_signals_sent"),
                "official_grader_executed": payload.get("official_grader_executed"),
                "kaggle_submission_executed": payload.get("kaggle_submission_executed"),
            }
        )
    raw = str(value.get("remote_probe_stdout") or "")
    gpu_section, _, remainder = raw.partition("__APPS__")
    app_section, _, _process_section = remainder.partition("__PROCS__")
    if not raw:
        gpu_section = str(value.get("gpu_stdout") or "")
        app_section = str(value.get("apps_stdout") or "")
    local_value = (
        value.get("local_successor_chain")
        if isinstance(value.get("local_successor_chain"), dict)
        else {}
    )
    local_successors = []
    for item in local_value.get("successors", []):
        if not isinstance(item, dict):
            continue
        wrapper = (
            item.get("wrapper_process")
            if isinstance(item.get("wrapper_process"), dict)
            else {}
        )
        worker = (
            item.get("worker_process")
            if isinstance(item.get("worker_process"), dict)
            else {}
        )
        local_successors.append(
            {
                "name": item.get("name"),
                "task_id": item.get("task_id"),
                "launcher_status": item.get("launcher_status"),
                "launcher_observed_at": item.get("launcher_observed_at"),
                "wrapper_pid": wrapper.get("pid"),
                "wrapper_live": wrapper.get("exists"),
                "worker_pid": worker.get("pid"),
                "worker_live": worker.get("exists"),
                "evidence_status_path": item.get("evidence_status_path"),
                "evidence_observed_at": item.get("evidence_observed_at"),
                "evidence_status": item.get("evidence_status"),
                "plan_sha256": item.get("plan_sha256"),
            }
        )
    return {
        "captured_at": _now(),
        "source": source,
        "schema": value.get("schema"),
        "observed_at": value.get("created_at"),
        "stages": stages,
        "gpu_lines": [line.strip() for line in gpu_section.splitlines() if line.strip()],
        "compute_apps": [
            line.strip() for line in app_section.splitlines() if line.strip()
        ],
        "local_successor_chain": {
            "passed": local_value.get("passed"),
            "expected_successor_count": local_value.get("expected_successor_count"),
            "observed_successor_count": local_value.get("observed_successor_count"),
            "all_launchers_running": local_value.get("all_launchers_running"),
            "all_wrappers_live": local_value.get("all_wrappers_live"),
            "all_workers_live": local_value.get("all_workers_live"),
            "all_evidence_statuses_present": local_value.get(
                "all_evidence_statuses_present"
            ),
            "successors": local_successors,
        },
        "remote_probe_exit_code": value.get("remote_probe_exit_code"),
        "process_signals_sent": value.get("process_signals_sent"),
        "other_processes_modified": value.get("other_processes_modified"),
        "rule": (
            "This is the authoritative already-scheduled Job 88240 chain. Do not "
            "queue duplicate Leaf, SIIM, RANZCR, Cactus, May, or Taxi work while "
            "their remote wrappers or local successor workers remain live; preserve "
            "natural serial progression and never signal them."
        ),
    }


def _normalise_may2022_supersession_seal(
    value: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    queues = []
    for item in value.get("superseded_queues") or []:
        if not isinstance(item, dict):
            continue
        queues.append(
            {
                key: item.get(key)
                for key in (
                    "queue_dir",
                    "status_before_seal",
                    "loaded_plan_sha256",
                    "current_plan_path_sha256",
                    "plan_path_hash_matches_loaded_status",
                    "claim_path",
                    "claim_sha256",
                )
            }
        )
    return {
        "captured_at": _now(),
        "source": source,
        "schema": value.get("schema"),
        "observed_at": value.get("created_at"),
        "status": value.get("status"),
        "authoritative_queue_dir": value.get("authoritative_queue_dir"),
        "authoritative_plan_sha256": value.get("authoritative_plan_sha256"),
        "authoritative_bundle_sha256": value.get("authoritative_bundle_sha256"),
        "superseded_queue_count": value.get("superseded_queue_count"),
        "superseded_queues": queues,
        "authoritative_queue_sealed": value.get("authoritative_queue_sealed"),
        "watchers_stopped": value.get("watchers_stopped"),
        "process_signals_sent": value.get("process_signals_sent"),
        "other_processes_modified": value.get("other_processes_modified"),
        "official_grader_executed": value.get("official_grader_executed"),
        "kaggle_submission_executed": value.get("kaggle_submission_executed"),
        "rule": (
            "Only the declared authoritative May queue may launch. Every listed "
            "superseded queue is locally fail-closed by an immutable claim; do not "
            "remove those claims or queue a duplicate run."
        ),
    }


def _normalise_taxi_cpu_candidate_status(
    value: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    remote = (
        value.get("remote_status")
        if isinstance(value.get("remote_status"), dict)
        else value
    )
    state = remote.get("state") if isinstance(remote.get("state"), dict) else {}
    process = remote.get("process") if isinstance(remote.get("process"), dict) else {}
    result = remote.get("result") if isinstance(remote.get("result"), dict) else None
    compact_result = None
    if result is not None:
        compact_result = {
            key: result.get(key)
            for key in (
                "schema",
                "created_at",
                "status",
                "passed",
                "competition_id",
                "model_family",
                "seed",
                "folds",
                "random_oof_rmse",
                "temporal_rmse",
                "geographic_rmse",
                "promotion_gate",
                "candidate_only",
                "private_labels_used",
                "official_grader_executed",
                "kaggle_submission_executed",
                "gpu_used",
                "claim_boundary",
            )
        }
    return {
        "captured_at": _now(),
        "source": source,
        "schema": value.get("schema"),
        "observed_at": value.get("created_at") or remote.get("observed_at"),
        "status": value.get("status"),
        "run_id": state.get("run_id"),
        "plan_sha256": state.get("plan_sha256") or value.get("plan_sha256"),
        "process": process,
        "result": compact_result,
        "process_signals_sent": remote.get("process_signals_sent"),
        "other_processes_modified": remote.get("other_processes_modified"),
        "official_grader_executed": remote.get("official_grader_executed"),
        "kaggle_submission_executed": remote.get("kaggle_submission_executed"),
        "rule": (
            "This independent job89941 CPU candidate may run concurrently with the "
            "A40 chain because it uses no GPU. Treat its OOF result as candidate evidence "
            "only; never count it as a medal or duplicate it while the process is live."
        ),
    }


def _normalise_taxi_cpu_confirmation_chain(
    value: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    completed = []
    for item in value.get("completed") or []:
        if isinstance(item, dict):
            completed.append(
                {
                    key: item.get(key)
                    for key in (
                        "seed",
                        "plan_sha256",
                        "collection_path",
                        "random_oof_rmse",
                        "temporal_rmse",
                        "geographic_rmse",
                    )
                }
            )
    return {
        "captured_at": _now(),
        "source": source,
        "schema": value.get("schema"),
        "observed_at": value.get("created_at"),
        "status": value.get("status"),
        "current_seed": value.get("current_seed"),
        "parent_gate": value.get("parent_gate"),
        "completed": completed,
        "process_signals_sent": value.get("process_signals_sent"),
        "other_processes_modified": value.get("other_processes_modified"),
        "official_grader_executed": value.get("official_grader_executed"),
        "kaggle_submission_executed": value.get("kaggle_submission_executed"),
        "rule": (
            "Taxi CPU seeds 44 and 45 are a strict serial successor chain. Do not "
            "launch or recommend a duplicate: each seed requires the preceding "
            "independently collected candidate to pass every frozen gate."
        ),
    }


def _normalise_taxi_cpu_progress(
    value: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    sample = value.get("sample") if isinstance(value.get("sample"), dict) else {}
    delta = (
        value.get("delta_from_previous")
        if isinstance(value.get("delta_from_previous"), dict)
        else {}
    )
    return {
        "captured_at": _now(),
        "source": source,
        "schema": value.get("schema"),
        "observed_at": value.get("created_at"),
        "status": value.get("status"),
        "plan_sha256": value.get("plan_sha256"),
        "sample": {
            key: sample.get(key)
            for key in (
                "captured_at_epoch",
                "pid",
                "process_exists",
                "process_state",
                "utime_ticks",
                "stime_ticks",
                "process_status",
                "io",
                "log_bytes",
                "result_exists",
                "loadavg",
            )
        },
        "delta_from_previous": delta,
        "stalled": value.get("stalled"),
        "process_signals_sent": value.get("process_signals_sent"),
        "other_processes_modified": value.get("other_processes_modified"),
        "official_grader_executed": value.get("official_grader_executed"),
        "kaggle_submission_executed": value.get("kaggle_submission_executed"),
        "rule": (
            "Use CPU tick, IO, log, or result deltas as the liveness authority. "
            "A live process with making_progress=true is not stalled even when its "
            "buffered training log has not grown."
        ),
    }


def _normalise_taxi_expert_evidence(
    value: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    result = value.get("result") if isinstance(value.get("result"), dict) else {}
    launch = value.get("launch") if isinstance(value.get("launch"), dict) else {}
    independent = (
        value.get("independent_verification")
        if isinstance(value.get("independent_verification"), dict)
        else {}
    )
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    experiments = []
    for item in value.get("experiments") or []:
        if isinstance(item, dict):
            experiments.append(
                {
                    key: item.get(key)
                    for key in (
                        "name",
                        "status",
                        "passed",
                        "model_family",
                        "seed",
                        "random_oof_rmse",
                        "official_grader_executed",
                        "kaggle_submission_executed",
                        "gpu_used",
                    )
                    if key in item
                }
            )
    return {
        "captured_at": _now(),
        "source": source,
        "schema": value.get("schema"),
        "observed_at": value.get("created_at"),
        "status": value.get("status"),
        "job_id": value.get("job_id"),
        "run_id": launch.get("run_id") or result.get("run_id"),
        "plan_sha256": value.get("plan_sha256") or launch.get("plan_sha256"),
        "process_status": value.get("process_status"),
        "result": {
            "status": result.get("status"),
            "passed": result.get("passed"),
            "model_family": result.get("model_family"),
            "seed": result.get("seed"),
            "folds": result.get("folds"),
            "distance_threshold_km": result.get("distance_threshold_km"),
            "expert_weight": result.get("expert_weight"),
            "longhaul_rows": result.get("longhaul_rows"),
            "metrics": metrics,
            "candidate_only": result.get("candidate_only"),
            "private_labels_used": result.get("private_labels_used"),
            "official_grader_executed": result.get("official_grader_executed"),
            "kaggle_submission_executed": result.get("kaggle_submission_executed"),
            "gpu_used": result.get("gpu_used"),
            "claim_boundary": result.get("claim_boundary"),
        }
        if result
        else None,
        "independent_verification": {
            key: independent.get(key)
            for key in (
                "verifier_node",
                "rows",
                "folds",
                "longhaul_rows",
                "recomputed_base_rmse",
                "recomputed_blended_rmse",
                "recomputed_base_longhaul_rmse",
                "recomputed_expert_longhaul_rmse",
                "fixed_weight_grid_rmse",
                "diagnostic_global_optimal_weight",
                "diagnostic_global_optimal_rmse",
                "official_grader_executed",
                "kaggle_submission_executed",
                "gpu_used",
            )
            if key in independent
        },
        "candidate_ready_for_nested_production_confirmation": value.get(
            "candidate_ready_for_nested_production_confirmation"
        ),
        "campaign": {
            "best_single_oof_rmse": value.get("best_single_oof_rmse"),
            "best_single_experiment": value.get("best_single_experiment"),
            "best_disjoint_combination_oof_rmse": value.get(
                "best_disjoint_combination_oof_rmse"
            ),
            "weighted_unweighted_leave_one_fold_blend_rmse": value.get(
                "weighted_unweighted_leave_one_fold_blend_rmse"
            ),
            "promotion_contract": value.get("promotion_contract"),
            "promotion_passed": value.get("promotion_passed"),
            "candidate_ready_for_human_gate": value.get(
                "candidate_ready_for_human_gate"
            ),
            "stop_reason": value.get("stop_reason"),
            "next_authority": value.get("next_authority"),
            "experiments": experiments,
        }
        if experiments
        else None,
        "process_signals_sent": value.get("process_signals_sent"),
        "other_processes_modified": value.get("other_processes_modified"),
        "official_grader_executed": value.get("official_grader_executed"),
        "kaggle_submission_executed": value.get("kaggle_submission_executed"),
        "rule": (
            "job89771 is a CPU-only, PUBLIC_ONLY Taxi expert lane. A cross-node OOF "
            "improvement may justify a nested production successor, but it is not an "
            "official score or medal. If the campaign status is diagnostic_stop_no_promotion, "
            "obey its stop reason and next authority; do not duplicate a closed CPU sweep."
        ),
    }


def _normalise_cpu_status(
    value: dict[str, Any],
    source: dict[str, Any],
    *,
    result_audit: dict[str, Any] | None = None,
    result_audit_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "captured_at": _now(),
        "sources": {
            "status": source,
            **(
                {"result_audit": result_audit_source}
                if result_audit_source is not None
                else {}
            ),
        },
        "schema": value.get("schema"),
        "observed_at": value.get("observed_at"),
        "status": value.get("status"),
        "job_id": value.get("job_id"),
        "run_id": value.get("run_id"),
        "process": value.get("process"),
        "result_present": value.get("result_present"),
        "result_valid": value.get("result_valid"),
        "process_signals_sent": value.get("process_signals_sent"),
        "claim_age_seconds": value.get("claim_age_seconds"),
        "plan_sha256": value.get("plan_sha256"),
    }
    if result_audit is not None:
        payload["result_audit"] = {
            key: result_audit.get(key)
            for key in (
                "schema",
                "created_at",
                "status",
                "all_integrity_checks_passed",
                "metrics",
                "data_boundary",
                "verdict",
                "executable_next_step",
            )
            if key in result_audit
        }
    return payload


def _normalise_leaf_status(
    value: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    verification = (
        value.get("verification")
        if isinstance(value.get("verification"), dict)
        else {}
    )
    return {
        "captured_at": _now(),
        "source": source,
        "schema": value.get("schema"),
        "observed_at": value.get("observed_at"),
        "status": value.get("status"),
        "process": value.get("process"),
        "process_signals_sent": value.get("process_signals_sent"),
        "run_id": verification.get("run_id")
        or (value.get("state") or {}).get("run_id"),
        "verification": {
            key: verification.get(key)
            for key in (
                "status",
                "ok",
                "candidate_ready",
                "seed_results",
                "ensemble_log_loss",
                "promotion_gate",
                "candidate_sha256",
                "candidate_only",
                "private_labels_used",
                "official_grader_executed",
                "kaggle_submission_executed",
                "claim_boundary",
            )
            if key in verification
        },
    }


def _normalise_cactus_cpu_sidecar(
    value: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    report = value.get("report") if isinstance(value.get("report"), dict) else {}
    return {
        "captured_at": _now(),
        "source": source,
        "schema": value.get("schema"),
        "created_at": value.get("created_at"),
        "status": value.get("status"),
        "candidate_ready": value.get("candidate_ready"),
        "dependency": value.get("dependency"),
        "report": {
            key: report.get(key)
            for key in (
                "gpu_baseline_auc",
                "nested_auc",
                "nested_gain",
                "full_oof_best",
                "full_best_gain",
                "candidate_ready",
                "claim_boundary",
            )
            if key in report
        },
        "process_signals_sent": value.get("process_signals_sent"),
        "other_processes_modified": value.get("other_processes_modified"),
        "official_grader_executed": value.get("official_grader_executed"),
        "kaggle_submission_executed": value.get("kaggle_submission_executed"),
    }


def _normalise_cactus_human_gate_postrun(
    value: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    return {
        "captured_at": _now(),
        "source": source,
        "schema": value.get("schema"),
        "created_at": value.get("created_at"),
        "status": value.get("status"),
        "remote_status": value.get("remote_status"),
        "approved": value.get("approved"),
        "automatic_approval": value.get("automatic_approval"),
        "process_signals_sent": value.get("process_signals_sent"),
        "other_processes_modified": value.get("other_processes_modified"),
        "official_grader_executed": value.get("official_grader_executed"),
        "kaggle_submission_executed": value.get("kaggle_submission_executed"),
    }


def _normalise_taxi_source_audit(
    value: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    checks = value.get("checks") if isinstance(value.get("checks"), dict) else {}
    return {
        "captured_at": _now(),
        "source": source,
        "schema": value.get("schema"),
        "observed_at": value.get("created_at"),
        "status": value.get("status"),
        "competition_id": value.get("competition_id"),
        "passed": value.get("passed"),
        "ready_for_candidate_execution": value.get("ready_for_candidate_execution"),
        "candidate_metric_available": value.get("candidate_metric_available"),
        "external_seed_results_available": value.get("external_seed_results_available"),
        "official_grader_executed": value.get("official_grader_executed"),
        "kaggle_submission_executed": value.get("kaggle_submission_executed"),
        "private_labels_used": value.get("private_labels_used"),
        "checks": {
            name: {
                key: record.get(key)
                for key in (
                    "passed",
                    "validation_target_rows_used",
                    "iteration_budget_uses_oof_only",
                    "test_targets_used",
                    "plan_sha256",
                    "bundle_sha256",
                    "runner_sha256",
                )
                if key in record
            }
            for name, record in checks.items()
            if isinstance(record, dict)
        },
        "runner_source": value.get("source"),
        "frozen_plan": value.get("plan"),
        "claim_boundary": value.get("claim_boundary"),
    }


def _normalise_may_execution_queue(
    value: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    bundle = value.get("bundle") if isinstance(value.get("bundle"), dict) else {}
    dependency = (
        value.get("dependency")
        if isinstance(value.get("dependency"), dict)
        else None
    )
    plan_source = None
    public_cache: dict[str, Any] = {}
    performance_contract: dict[str, Any] = {}
    raw_plan_path = str(value.get("plan_path") or "")
    if raw_plan_path:
        try:
            plan_path = Path(raw_plan_path).expanduser().resolve()
            plan_path.relative_to(PROJECT_ROOT.resolve())
            if plan_path.is_file():
                plan_value, plan_sha = _load_json(plan_path)
                plan_source = _source_record(plan_path, plan_sha)
                raw_cache = plan_value.get("public_precomputed_cache")
                if isinstance(raw_cache, dict):
                    public_cache = {
                        key: raw_cache.get(key)
                        for key in (
                            "required",
                            "path",
                            "manifest_path",
                            "manifest_sha256",
                            "status",
                            "visibility_mode",
                            "seed",
                            "folds",
                            "train_rows",
                            "test_rows",
                            "feature_count",
                            "feature_schema_sha256",
                            "feature_builder_sha256",
                            "private_labels_used",
                            "official_grader_executed",
                            "kaggle_submission_executed",
                        )
                        if key in raw_cache
                    }
                raw_performance = plan_value.get("performance_contract")
                if isinstance(raw_performance, dict):
                    performance_contract = dict(raw_performance)
        except (OSError, ValueError, json.JSONDecodeError):
            plan_source = None
    return {
        "captured_at": _now(),
        "source": source,
        "schema": value.get("schema"),
        "observed_at": value.get("created_at"),
        "status": value.get("status"),
        "plan_path": value.get("plan_path"),
        "plan_sha256": value.get("plan_sha256"),
        "plan_source": plan_source,
        "public_precomputed_cache": public_cache,
        "performance_contract": performance_contract,
        "bundle": {
            key: bundle.get(key)
            for key in (
                "sha256",
                "archive_member_count",
                "manifest_hash_count",
                "human_gate_preserved",
                "passed",
            )
            if key in bundle
        },
        "dependency": (
            {
                key: dependency.get(key)
                for key in (
                    "schema",
                    "created_at",
                    "status",
                    "completed_seeds",
                    "exit_code",
                    "process_signals_sent",
                    "official_grader_executed",
                    "kaggle_submission_executed",
                )
                if key in dependency
            }
            if dependency is not None
            else None
        ),
        "process_signals_sent": value.get("process_signals_sent"),
        "other_processes_modified": value.get("other_processes_modified"),
        "official_grader_executed": value.get("official_grader_executed"),
        "kaggle_submission_executed": value.get("kaggle_submission_executed"),
        "rule": (
            "Treat the May queue as execution authority: it may train only after the "
            "existing A40 serial chain reaches a valid terminal state and the fresh "
            "three-sample GPU gate passes. A required verified public cache forbids a "
            "duplicate feature rebuild; candidate-only and Human Gate remain mandatory."
        ),
    }


def _normalise_staged_human_gate_candidates(
    value: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    candidates = []
    for item in value.get("candidates", []):
        if not isinstance(item, dict):
            continue
        candidates.append(
            {
                key: item.get(key)
                for key in (
                    "run_id",
                    "competition_id",
                    "candidate_manifest",
                    "submission",
                    "staging_verification",
                    "approval_template",
                    "approved",
                    "official_grader_executed",
                    "kaggle_submission_executed",
                )
                if key in item
            }
        )
    return {
        "captured_at": _now(),
        "source": source,
        "schema": value.get("schema"),
        "created_at": value.get("created_at"),
        "status": value.get("status"),
        "requested_action": value.get("requested_action"),
        "candidates": candidates,
        "exact_approval_phrase": value.get("exact_approval_phrase"),
        "automatic_approval": value.get("automatic_approval"),
        "official_grader_executed": value.get("official_grader_executed"),
        "kaggle_submission_executed": value.get("kaggle_submission_executed"),
        "claim_boundary": value.get("claim_boundary"),
    }


def _discover_ranzcr_summary(workspace: Path) -> Path:
    root = workspace / "workspace" / "hpc" / "mlebench_remote_ops" / "collected"
    candidates = sorted(
        root.glob("*ranzcr*/summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No collected RANZCR summary was found")
    return candidates[0]


def _discover_a800_heartbeat(workspace: Path) -> Path:
    run_id = (workspace / "workspace" / "hpc" / "a800_lane_run_id_current.txt").read_text(encoding="utf-8-sig").strip()
    if not run_id:
        raise RuntimeError("The current A800 run id is empty")
    return workspace / "workspace" / "hpc" / "a800_lane_watch" / f"{run_id}_heartbeat_current.json"


def _discover_a800_parallel_heartbeat(workspace: Path, queue_path: Path) -> Path:
    queue, _digest = _load_json(queue_path)
    parallel = queue.get("parallel_runs")
    watch_root = workspace / "workspace" / "hpc" / "a800_lane_watch"
    if isinstance(parallel, dict) and parallel:
        run_ids = sorted(str(value) for value in parallel.values() if value)
        if len(run_ids) != 1:
            raise RuntimeError("Expected exactly one current A800 parallel run")
        return watch_root / f"{run_ids[0]}_heartbeat_current.json"

    # Older live queue supervisors omit ``parallel_runs`` from transient retry
    # markers even though the parallel watcher continues to refresh its heartbeat.
    # Recover from that evidence instead of dropping the lane from the controller.
    candidates = sorted(
        watch_root.glob("*parallel*_heartbeat_current.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "The A800 recovery queue and watcher directory have no parallel run"
        )
    return candidates[0]


def build_evidence_snapshots(
    workspace: Path,
    *,
    progress_path: Path,
    campaign_path: Path,
    ranzcr_path: Path,
    a800_heartbeat_path: Path,
    a800_parallel_heartbeat_path: Path,
    a800_queue_path: Path,
    a40_status_path: Path,
    local_dog_plan_path: Path,
    local_dog_queue_path: Path,
    dog_audit_path: Path,
    taxi_source_audit_path: Path,
    recovery_deep_review_path: Path,
    human_gate_root: Path,
    job88240_gpu_gate_path: Path,
    job89508_resource_probe_path: Path,
    job88240_serial_chain_path: Path,
    job89941_may_status_path: Path,
    job89941_may_audit_path: Path | None,
    job89941_taxi_cpu_candidate_path: Path,
    job89941_taxi_cpu_progress_path: Path,
    job89941_taxi_cpu_confirmation_path: Path,
    job89771_taxi_longhaul_expert_path: Path,
    job89771_taxi_midhaul_expert_path: Path,
    job89941_leaf_status_path: Path,
    cactus_cpu_sidecar_status_path: Path,
    cactus_human_gate_postrun_path: Path,
    ranzcr_human_gate_postrun_path: Path,
    siim_human_gate_postrun_path: Path,
    may2022_execution_queue_path: Path,
    may2022_supersession_seal_path: Path,
    staged_human_gate_request_path: Path,
    snapshot_dir: Path,
) -> tuple[dict[str, dict[str, Any]], set[str], dict[str, str]]:
    progress, progress_sha = _load_json(progress_path)
    campaign, campaign_sha = _load_json(campaign_path)
    ranzcr, ranzcr_sha = _load_json(ranzcr_path)
    a800_heartbeat, a800_heartbeat_sha = _load_json(a800_heartbeat_path)
    a800_parallel, a800_parallel_sha = _load_json(a800_parallel_heartbeat_path)
    a800_queue, a800_queue_sha = _load_json(a800_queue_path)
    a40, a40_sha = _load_json(a40_status_path)
    local_dog_plan, local_dog_plan_sha = _load_json(local_dog_plan_path)
    local_dog_queue, local_dog_queue_sha = _load_json(local_dog_queue_path)
    dog_audit, dog_audit_sha = _load_json(dog_audit_path)
    taxi_source_audit, taxi_source_audit_sha = _load_json(taxi_source_audit_path)
    recovery_deep_review, recovery_deep_review_sha = _load_json(
        recovery_deep_review_path
    )
    job88240_gpu_gate, job88240_gpu_gate_sha = _load_json(
        job88240_gpu_gate_path
    )
    job89508_resource_probe, job89508_resource_probe_sha = _load_json(
        job89508_resource_probe_path
    )
    job88240_serial_chain, job88240_serial_chain_sha = _load_json(
        job88240_serial_chain_path
    )
    job89941_may_status, job89941_may_status_sha = _load_json(
        job89941_may_status_path
    )
    job89941_taxi_cpu_candidate, job89941_taxi_cpu_candidate_sha = _load_json(
        job89941_taxi_cpu_candidate_path
    )
    job89941_taxi_cpu_progress, job89941_taxi_cpu_progress_sha = _load_json(
        job89941_taxi_cpu_progress_path
    )
    job89941_taxi_cpu_confirmation, job89941_taxi_cpu_confirmation_sha = _load_json(
        job89941_taxi_cpu_confirmation_path
    )
    job89771_taxi_longhaul_expert, job89771_taxi_longhaul_expert_sha = _load_json(
        job89771_taxi_longhaul_expert_path
    )
    job89771_taxi_midhaul_expert, job89771_taxi_midhaul_expert_sha = _load_json(
        job89771_taxi_midhaul_expert_path
    )
    job89941_may_audit: dict[str, Any] | None = None
    job89941_may_audit_source: dict[str, Any] | None = None
    if job89941_may_audit_path is not None and job89941_may_audit_path.is_file():
        job89941_may_audit, job89941_may_audit_sha = _load_json(
            job89941_may_audit_path
        )
        job89941_may_audit_source = _source_record(
            job89941_may_audit_path, job89941_may_audit_sha
        )
    job89941_leaf_status, job89941_leaf_status_sha = _load_json(
        job89941_leaf_status_path
    )
    cactus_cpu_sidecar_status, cactus_cpu_sidecar_status_sha = _load_json(
        cactus_cpu_sidecar_status_path
    )
    cactus_human_gate_postrun, cactus_human_gate_postrun_sha = _load_json(
        cactus_human_gate_postrun_path
    )
    ranzcr_human_gate_postrun, ranzcr_human_gate_postrun_sha = _load_json(
        ranzcr_human_gate_postrun_path
    )
    siim_human_gate_postrun, siim_human_gate_postrun_sha = _load_json(
        siim_human_gate_postrun_path
    )
    may2022_execution_queue, may2022_execution_queue_sha = _load_json(
        may2022_execution_queue_path
    )
    may2022_supersession_seal, may2022_supersession_seal_sha = _load_json(
        may2022_supersession_seal_path
    )
    staged_human_gate_request, staged_human_gate_request_sha = _load_json(
        staged_human_gate_request_path
    )

    payloads = {
        "official_progress": _normalise_progress(progress, _source_record(progress_path, progress_sha)),
        "campaign_state": _normalise_campaign(campaign, _source_record(campaign_path, campaign_sha)),
        "ranzcr_failure": _normalise_run_summary(ranzcr, _source_record(ranzcr_path, ranzcr_sha)),
        "a800_status": _normalise_a800(
            a800_heartbeat,
            _source_record(a800_heartbeat_path, a800_heartbeat_sha),
            a800_queue,
            _source_record(a800_queue_path, a800_queue_sha),
        ),
        "a800_parallel_status": _normalise_a800_parallel(
            a800_parallel,
            _source_record(a800_parallel_heartbeat_path, a800_parallel_sha),
        ),
        "a40_taxi_status": _normalise_a40(a40, _source_record(a40_status_path, a40_sha)),
        "local4060_dog_status": _normalise_local4060_dog(
            local_dog_plan,
            _source_record(local_dog_plan_path, local_dog_plan_sha),
            local_dog_queue,
            _source_record(local_dog_queue_path, local_dog_queue_sha),
        ),
        "dog_gpt56_audit": _normalise_dog_audit(
            dog_audit,
            _source_record(dog_audit_path, dog_audit_sha),
        ),
        "taxi_source_audit": _normalise_taxi_source_audit(
            taxi_source_audit,
            _source_record(taxi_source_audit_path, taxi_source_audit_sha),
        ),
        "recovery_deep_review": _normalise_recovery_deep_review(
            recovery_deep_review,
            _source_record(recovery_deep_review_path, recovery_deep_review_sha),
        ),
        "human_gate_candidates": _normalise_human_gate_candidates(
            human_gate_root
        ),
        "job88240_gpu_gate": _normalise_gpu_gate(
            job88240_gpu_gate,
            _source_record(job88240_gpu_gate_path, job88240_gpu_gate_sha),
        ),
        "job89508_resource_probe": _normalise_job89508_resource_probe(
            job89508_resource_probe,
            _source_record(
                job89508_resource_probe_path, job89508_resource_probe_sha
            ),
        ),
        "job88240_serial_chain": _normalise_job88240_serial_chain(
            job88240_serial_chain,
            _source_record(
                job88240_serial_chain_path, job88240_serial_chain_sha
            ),
        ),
        "job89941_may_status": _normalise_cpu_status(
            job89941_may_status,
            _source_record(job89941_may_status_path, job89941_may_status_sha),
            result_audit=job89941_may_audit,
            result_audit_source=job89941_may_audit_source,
        ),
        "job89941_taxi_cpu_candidate": _normalise_taxi_cpu_candidate_status(
            job89941_taxi_cpu_candidate,
            _source_record(
                job89941_taxi_cpu_candidate_path,
                job89941_taxi_cpu_candidate_sha,
            ),
        ),
        "job89941_taxi_cpu_progress": _normalise_taxi_cpu_progress(
            job89941_taxi_cpu_progress,
            _source_record(
                job89941_taxi_cpu_progress_path,
                job89941_taxi_cpu_progress_sha,
            ),
        ),
        "job89941_taxi_cpu_confirmation": _normalise_taxi_cpu_confirmation_chain(
            job89941_taxi_cpu_confirmation,
            _source_record(
                job89941_taxi_cpu_confirmation_path,
                job89941_taxi_cpu_confirmation_sha,
            ),
        ),
        "job89771_taxi_longhaul_expert": _normalise_taxi_expert_evidence(
            job89771_taxi_longhaul_expert,
            _source_record(
                job89771_taxi_longhaul_expert_path,
                job89771_taxi_longhaul_expert_sha,
            ),
        ),
        "job89771_taxi_midhaul_expert": _normalise_taxi_expert_evidence(
            job89771_taxi_midhaul_expert,
            _source_record(
                job89771_taxi_midhaul_expert_path,
                job89771_taxi_midhaul_expert_sha,
            ),
        ),
        "job89941_leaf_status": _normalise_leaf_status(
            job89941_leaf_status,
            _source_record(job89941_leaf_status_path, job89941_leaf_status_sha),
        ),
        "cactus_cpu_sidecar_status": _normalise_cactus_cpu_sidecar(
            cactus_cpu_sidecar_status,
            _source_record(
                cactus_cpu_sidecar_status_path, cactus_cpu_sidecar_status_sha
            ),
        ),
        "cactus_human_gate_postrun": _normalise_cactus_human_gate_postrun(
            cactus_human_gate_postrun,
            _source_record(
                cactus_human_gate_postrun_path, cactus_human_gate_postrun_sha
            ),
        ),
        "ranzcr_human_gate_postrun": _normalise_cactus_human_gate_postrun(
            ranzcr_human_gate_postrun,
            _source_record(
                ranzcr_human_gate_postrun_path, ranzcr_human_gate_postrun_sha
            ),
        ),
        "siim_human_gate_postrun": _normalise_cactus_human_gate_postrun(
            siim_human_gate_postrun,
            _source_record(siim_human_gate_postrun_path, siim_human_gate_postrun_sha),
        ),
        "may2022_execution_queue": _normalise_may_execution_queue(
            may2022_execution_queue,
            _source_record(
                may2022_execution_queue_path, may2022_execution_queue_sha
            ),
        ),
        "may2022_supersession_seal": _normalise_may2022_supersession_seal(
            may2022_supersession_seal,
            _source_record(
                may2022_supersession_seal_path, may2022_supersession_seal_sha
            ),
        ),
        "staged_human_gate_candidates": _normalise_staged_human_gate_candidates(
            staged_human_gate_request,
            _source_record(
                staged_human_gate_request_path, staged_human_gate_request_sha
            ),
        ),
    }

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, str] = {}
    for name, payload in payloads.items():
        path = snapshot_dir / f"{name}.json"
        _write_json_atomic(path, payload)
        data = path.read_bytes()
        records[name] = {
            "path": str(path.resolve()),
            "relative_path": path.resolve().relative_to(workspace).as_posix(),
            "sha256": _sha256(data),
            "line_count": len(data.decode("utf-8").splitlines()),
        }
        source = payload.get("source")
        if isinstance(source, dict):
            source_hashes[name] = str(source.get("sha256") or "")
        else:
            sources = payload.get("sources") or {}
            combined = json.dumps(sources, sort_keys=True, ensure_ascii=False).encode("utf-8")
            source_hashes[name] = _sha256(combined)

    competition_ids = {
        str(item.get("competition_id"))
        for item in progress.get("results", [])
        if isinstance(item, dict) and item.get("competition_id")
    }
    competition_ids.update(str(value) for value in progress.get("remaining_competition_ids", []) if value)
    return records, competition_ids, source_hashes


def _extract_decision_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schema") == DECISION_SCHEMA:
            return value
    raise ValueError("The model did not return the required adaptive-decision JSON object")


def _fresh_runtime_evidence(
    payload: dict[str, Any],
    *,
    now: datetime,
    max_age_seconds: int = ACTIVE_RUN_MAX_AGE_SECONDS,
) -> bool:
    """Reject a cached ``running`` marker after its observation window expires."""

    for key in ("observed_at", "updated_at", "created_at"):
        raw = payload.get(key)
        if not raw:
            continue
        try:
            timestamp = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            return False
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        age = (now - timestamp.astimezone(timezone.utc)).total_seconds()
        return -300 <= age <= max_age_seconds
    return False


def _active_runs_from_snapshots(
    snapshot_dir: Path,
    *,
    now: datetime | None = None,
) -> dict[str, str]:
    a800 = json.loads((snapshot_dir / "a800_status.json").read_text(encoding="utf-8"))
    a800_parallel = json.loads(
        (snapshot_dir / "a800_parallel_status.json").read_text(encoding="utf-8")
    )
    a40 = json.loads((snapshot_dir / "a40_taxi_status.json").read_text(encoding="utf-8"))
    local_dog = json.loads(
        (snapshot_dir / "local4060_dog_status.json").read_text(encoding="utf-8")
    )
    may_status = json.loads(
        (snapshot_dir / "job89941_may_status.json").read_text(encoding="utf-8")
    )
    observed_now = now or datetime.now(timezone.utc)
    runs: dict[str, str] = {}
    a800_heartbeat = a800.get("heartbeat") or {}
    if a800_heartbeat.get("process") == "running" and _fresh_runtime_evidence(
        a800_heartbeat,
        now=observed_now,
    ):
        runs["a800"] = str(a800_heartbeat.get("run_id") or "")
    parallel_heartbeat = a800_parallel.get("heartbeat") or {}
    if parallel_heartbeat.get("process") == "running" and _fresh_runtime_evidence(
        parallel_heartbeat,
        now=observed_now,
    ):
        runs["a800_parallel"] = str(parallel_heartbeat.get("run_id") or "")
    if a40.get("process") == "running":
        runs["a40"] = str((a40.get("run") or {}).get("run_id") or "")
    queue = local_dog.get("live_queue") or {}
    queue_status = str(queue.get("status") or "")
    if queue_status == "recovery_launched":
        runs["local4060"] = str(queue.get("recovery_run_id") or "")
    elif queue_status.startswith("waiting_for_parent"):
        runs["local4060"] = str((local_dog.get("active_parent") or {}).get("run_id") or "")
    if (
        may_status.get("status") == "running"
        and isinstance(may_status.get("process"), dict)
        and may_status["process"].get("running") is True
        and _fresh_runtime_evidence(may_status, now=observed_now)
    ):
        runs["cpu"] = str(may_status.get("run_id") or "")
    return {lane: run_id for lane, run_id in runs.items() if run_id}


def validate_decision(
    decision: dict[str, Any],
    *,
    competition_ids: set[str],
    source_hashes: dict[str, str],
    active_runs: dict[str, str],
    blocked_collect_and_grade: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    if decision.get("schema") != DECISION_SCHEMA:
        errors.append("schema mismatch")
    assessment = decision.get("leaderboard_assessment")
    if not isinstance(assessment, dict):
        errors.append("leaderboard_assessment is required")
    evidence = decision.get("evidence_source_sha256")
    if not isinstance(evidence, dict):
        errors.append("evidence_source_sha256 is required")
    else:
        for name in REQUIRED_EVIDENCE:
            if evidence.get(name) != source_hashes.get(name):
                errors.append(f"source hash mismatch: {name}")

    policies = decision.get("active_run_policy")
    if not isinstance(policies, list):
        errors.append("active_run_policy must be a list")
        policies = []
    seen_lanes: set[str] = set()
    for item in policies:
        if not isinstance(item, dict):
            errors.append("active_run_policy entries must be objects")
            continue
        lane = str(item.get("lane") or "")
        seen_lanes.add(lane)
        if lane not in ACTIVE_RUN_LANES:
            errors.append(f"invalid active lane: {lane}")
        if item.get("run_id") != active_runs.get(lane):
            errors.append(f"active run mismatch: {lane}")
        if item.get("action") != "continue_active_runs":
            errors.append(f"active run must continue: {lane}")
    for lane, run_id in active_runs.items():
        if run_id and lane not in seen_lanes:
            errors.append(f"missing active run policy: {lane}")

    actions = decision.get("next_actions")
    if not isinstance(actions, list) or not actions:
        errors.append("next_actions must be a non-empty list")
        actions = []
    if len(actions) > 8:
        errors.append("next_actions exceeds the maximum of 8")
    priorities: list[int] = []
    for item in actions:
        if not isinstance(item, dict):
            errors.append("next_actions entries must be objects")
            continue
        action = str(item.get("action") or "")
        competition_id = str(item.get("competition_id") or "")
        lane = str(item.get("lane") or "")
        if action not in ALLOWED_ACTIONS:
            errors.append(f"invalid action: {action}")
        if competition_id not in competition_ids:
            errors.append(f"unknown competition: {competition_id}")
        if action == "collect_and_grade" and competition_id in (
            blocked_collect_and_grade or set()
        ):
            errors.append(
                f"collect_and_grade requires a new experiment after promotion gate withholding: {competition_id}"
            )
        if lane not in ALLOWED_LANES:
            errors.append(f"invalid lane: {lane}")
        if item.get("requires_active_run_completion") is not True:
            errors.append(f"action may interrupt active run: {competition_id}")
        priority = item.get("priority")
        if not isinstance(priority, int) or isinstance(priority, bool):
            errors.append(f"invalid priority: {competition_id}")
        else:
            priorities.append(priority)
    if priorities and sorted(priorities) != list(range(1, len(priorities) + 1)):
        errors.append("priorities must be unique and contiguous from 1")
    return errors


def enforce_authoritative_active_run_policy(
    decision: dict[str, Any], active_runs: dict[str, str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Make process-freshness evidence, not model inference, own active-run state."""

    expected = [
        {"lane": lane, "run_id": run_id, "action": "continue_active_runs"}
        for lane, run_id in active_runs.items()
    ]
    observed = decision.get("active_run_policy")
    if observed == expected:
        return decision, []
    enforced = deepcopy(decision)
    enforced["active_run_policy"] = expected
    return enforced, [
        {
            "field": "active_run_policy",
            "reason": "deterministic freshness-derived runtime authority",
            "model_value": observed,
            "enforced_value": expected,
        }
    ]


def collect_tool_evidence(
    events: list[dict[str, Any]],
    snapshots: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    expected_by_path = {str(Path(record["path"]).resolve()): (name, record) for name, record in snapshots.items()}
    started_names: dict[str, str] = {}
    observed: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for event in events:
        payload = event.get("payload") or {}
        if event.get("event_type") == "tool.started":
            tool_name = str(payload.get("tool_name") or "")
            if tool_name != "file_read":
                errors.append(f"unexpected tool call: {tool_name}")
            arguments = payload.get("arguments") or {}
            path = str(Path(str(arguments.get("path") or "")).resolve())
            started_names[str(payload.get("id") or "")] = path
        elif event.get("event_type") == "tool.completed":
            call_id = str(payload.get("tool_call_id") or "")
            content = payload.get("content") or {}
            path = str(Path(str(content.get("path") or started_names.get(call_id, ""))).resolve())
            if path not in expected_by_path:
                errors.append(f"unexpected file_read path: {path}")
                continue
            name, record = expected_by_path[path]
            digest = str(content.get("sha256") or "")
            if digest != record["sha256"]:
                errors.append(f"observed snapshot hash mismatch: {name}")
            observed[name] = {
                "path": path,
                "sha256": digest,
                "line_count": content.get("line_count"),
                "tool_call_id": call_id,
            }
    for name in REQUIRED_EVIDENCE:
        if name not in observed:
            errors.append(f"required evidence was not read: {name}")
    return observed, errors


def _validate_environment() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not loaded in the secure runtime environment")
    if os.environ.get("OPENAI_MODEL", "") != REQUIRED_MODEL:
        raise RuntimeError(f"OPENAI_MODEL must be {REQUIRED_MODEL}")
    if os.environ.get("EVOLUTION_PRIMARY_PROVIDER", "") != "openai":
        raise RuntimeError("EVOLUTION_PRIMARY_PROVIDER must be openai")
    if os.environ.get("EVOLUTION_PROVIDER_STRICT", "").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        raise RuntimeError("EVOLUTION_PROVIDER_STRICT must be enabled")
    parsed = urlparse(os.environ.get("OPENAI_BASE_URL", ""))
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.port != 65068
        or parsed.path.rstrip("/") != "/v1"
    ):
        raise RuntimeError("OPENAI_BASE_URL must use the approved loopback gateway")


def _prompt(
    snapshots: dict[str, dict[str, Any]],
    source_hashes: dict[str, str],
    active_runs: dict[str, str],
    competition_ids: set[str],
    blocked_collect_and_grade: set[str],
) -> str:
    reads = "\n".join(
        f"- {name}: call file_read with path={record['relative_path']!r}, start_line=1, end_line={record['line_count']}"
        for name, record in snapshots.items()
    )
    return f"""You are the evidence-driven MLE-Bench controller inside EvoMind.
Use only the file_read tool. Read every evidence file listed below before deciding.
Do not call file_list, file_search, shell, process, write, browser, desktop, MCP, or research tools.
Issue all required file_read calls together in one parallel tool-call batch in the first
model turn. After that single evidence batch, return the final JSON in the next turn;
do not serialize the reads across separate model turns.

Required reads:
{reads}

Objective: strictly exceed the published MLE-Bench Lite top medal rate. The verified target is at least 18 medals out of 22, followed by seeds 43 and 44 confirmation. Never treat CV, a proxy metric, or a promotion gate as an official private grade. Never interrupt any active run. Do not submit to Kaggle.
Use the compact full-source recovery review as the implementation-readiness authority:
rank a source-audited GO path above a smaller raw official-score gap when the latter is
NO-GO, and obey each competition's bounded stop rule and validation contract.
Treat human_gate_candidates as the current candidate authority. When a package is
independently verified and ready for Human Gate, prefer collect_and_grade over training
another configuration, while keeping it out of the official medal count until grading.
Treat job88240_gpu_gate as the current GPU launch authority: when its fresh gate is
false, queue GPU work without launching it or hold for evidence. Preserve every fresh
job89941_may_status CPU run unchanged until it reaches a terminal verified state.
Treat job89508_resource_probe as the newer A800 safety interlock for job 89508. When
training_start_allowed=false, compute apps are present, NVIDIA device handles remain
open, or the default runtime contract fails, do not recommend starting a parallel A800
training process from job 89508. Only hold or queue behind a fresh passing
three-sample gate, and never let older A800 heartbeat evidence override this probe.
Treat job88240_serial_chain as the authoritative existing remote schedule. Do not
recommend duplicate Leaf, SIIM, RANZCR, or Cactus queues while those wrappers remain
live; preserve their natural order and place May only after the Cactus terminal state.
Treat staged_human_gate_candidates as the one-shot execution authority: approved=false
means request explicit approval and do not execute a grader. Use job89941_leaf_status as
the current Leaf authority; a verified ensemble metric does not override a failed
predeclared per-seed promotion gate.
Treat taxi_source_audit as the current Taxi source-readiness authority. When it reports
ready_for_candidate_execution=true but candidate_metric_available=false, the old failed
candidate no longer proves the new source path is NO-GO; queue or hold the new bounded
three-seed experiment according to current compute gates, without claiming a score.
Treat job89941_taxi_cpu_candidate as an additional no-GPU candidate channel. Preserve a
live matching process without duplication; if it becomes terminal, use its PUBLIC_ONLY
OOF and stress gates as candidate evidence only and keep Official Grader behind Human Gate.
Use job89941_taxi_cpu_progress as the read-only liveness authority. Buffered log size
alone is not a stall signal when CPU/IO deltas prove continued progress.
Treat job89941_taxi_cpu_confirmation as the only seeds 44/45 CPU successor authority.
It may advance strictly after the preceding collected seed passes every frozen gate;
never queue a parallel or duplicate confirmation run.
Treat job89771_taxi_longhaul_expert as cross-node verified diagnostic evidence from a
distinct CPU-only expert path. Treat job89771_taxi_midhaul_expert as the current 89771
campaign authority: when it reports diagnostic_stop_no_promotion, obey its stop reason,
do not queue another Taxi CPU sweep, and preserve the source-audited A40 successor.
Even a large PUBLIC_ONLY OOF improvement remains diagnostic until nested stress
validation and the production gate.
Treat may2022_execution_queue as the current May GPU execution authority. A waiting
queue is already scheduled work, not permission to race the existing A40 serial chain;
do not recommend a duplicate launch. When it binds a completed PUBLIC_ONLY cache, keep
that cache mandatory and reserve A40 time for fitting rather than rebuilding features.
Preserve its candidate-only and Human Gate fields.
Treat may2022_supersession_seal as the duplicate-prevention authority. Its listed old
queues are fail-closed without process termination, and only its declared authoritative
queue may launch; never remove, replace, or bypass a supersession claim.
Treat cactus_cpu_sidecar_status as a deferred heterogeneous Cactus blend authority.
While it waits for the existing Cactus GPU terminal state, preserve it without adding
another GPU queue. If its nested sidecar gate later fails, do not promote the CPU blend;
if it passes, treat it as candidate-only until Human Gate.
Treat cactus_human_gate_postrun as the authoritative Cactus candidate-conversion
workflow. While it waits, do not duplicate collection. If it reaches
verified_human_approval_pending, request approval and do not execute the grader.
Treat ranzcr_human_gate_postrun and siim_human_gate_postrun as their authoritative
candidate-conversion workflows under the same Human Gate rule. Waiting states are
already scheduled work; do not duplicate collection, training, or grading.

Active runs that must continue unchanged:
{json.dumps(active_runs, ensure_ascii=False, sort_keys=True)}
This exact freshness-derived map is authoritative. active_run_policy must contain
exactly one entry for each map item and no extra entries. When the map is {{}}, return
an empty active_run_policy even if an older evidence file still says "running".

Allowed competition IDs:
{json.dumps(sorted(competition_ids), ensure_ascii=False)}

Allowed actions: {json.dumps(sorted(ALLOWED_ACTIONS))}
Allowed lanes: {json.dumps(sorted(ALLOWED_LANES))}
Competition IDs that must not use collect_and_grade in this decision because their
latest collected outcome has official_grade_count=0 and promotion_gate_withheld=true:
{json.dumps(sorted(blocked_collect_and_grade), ensure_ascii=False)}
Exact evidence source hashes to echo after reading:
{json.dumps(source_hashes, ensure_ascii=False, sort_keys=True)}

After all tool results, return exactly one JSON object and no markdown using this contract:
{{
  "schema": "{DECISION_SCHEMA}",
  "leaderboard_assessment": {{
    "official_scored": 0,
    "official_medals": 0,
    "target_medals": 18,
    "medal_deficit": 0,
    "strictly_above_top": false
  }},
  "active_run_policy": [
    {json.dumps([{"lane": lane, "run_id": run_id, "action": "continue_active_runs"} for lane, run_id in active_runs.items()], ensure_ascii=False)[1:-1]}
  ],
  "next_actions": [
    {{
      "priority": 1,
      "action": "queue_experiment",
      "competition_id": "...",
      "lane": "a800",
      "expected_medal_gain": 1,
      "requires_active_run_completion": true,
      "rationale": "evidence-grounded concise reason"
    }}
  ],
  "evidence_source_sha256": {json.dumps(source_hashes, ensure_ascii=False, sort_keys=True)},
  "stop_conditions": ["official 22/22 coverage", "at least 18 official medals", "seeds 43 and 44 confirmed"]
}}

Return between 1 and 8 next_actions, inclusive. Rank them by expected official medal
conversion per GPU-hour. Include collection/grading only when the current evidence is
gradeable, and include a bounded RANZCR recovery only after active runs. Every next
action must set requires_active_run_completion=true. For every competition ID in the
explicit blocked list above, select queue_experiment, retry_failed_after_active, or
hold_for_evidence as appropriate; never select collect_and_grade. The active_run_policy
already preserves running experiments, so do not add a duplicate collection action for
their stale pre-run outcomes.
"""


def main() -> int:
    started_monotonic = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--ranzcr-summary", type=Path)
    parser.add_argument("--a800-heartbeat", type=Path)
    parser.add_argument("--a800-parallel-heartbeat", type=Path)
    parser.add_argument("--a800-queue", type=Path)
    parser.add_argument("--a40-status", type=Path)
    parser.add_argument("--local-dog-plan", type=Path)
    parser.add_argument("--local-dog-queue", type=Path)
    parser.add_argument("--dog-audit", type=Path)
    parser.add_argument("--taxi-source-audit", type=Path)
    parser.add_argument("--recovery-deep-review", type=Path)
    parser.add_argument("--human-gate-root", type=Path)
    parser.add_argument("--job88240-gpu-gate", type=Path)
    parser.add_argument("--job89508-resource-probe", type=Path)
    parser.add_argument("--job88240-serial-chain", type=Path)
    parser.add_argument("--job89941-may-status", type=Path)
    parser.add_argument("--job89941-may-audit", type=Path)
    parser.add_argument("--job89941-taxi-cpu-candidate", type=Path)
    parser.add_argument("--job89941-taxi-cpu-progress", type=Path)
    parser.add_argument("--job89941-taxi-cpu-confirmation", type=Path)
    parser.add_argument("--job89771-taxi-longhaul-expert", type=Path)
    parser.add_argument("--job89771-taxi-midhaul-expert", type=Path)
    parser.add_argument("--job89941-leaf-status", type=Path)
    parser.add_argument("--cactus-cpu-sidecar-status", type=Path)
    parser.add_argument("--cactus-human-gate-postrun", type=Path)
    parser.add_argument("--ranzcr-human-gate-postrun", type=Path)
    parser.add_argument("--siim-human-gate-postrun", type=Path)
    parser.add_argument("--may2022-execution-queue", type=Path)
    parser.add_argument("--may2022-supersession-seal", type=Path)
    parser.add_argument("--staged-human-gate-request", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    progress_path = (args.progress or workspace / "workspace" / "mlebench_progress" / "lite11_current.json").resolve()
    campaign_path = (
        args.campaign or workspace / "workspace" / "hpc" / "mlebench_remote_ops" / "job89441_full_campaign_current.json"
    ).resolve()
    ranzcr_path = (args.ranzcr_summary or _discover_ranzcr_summary(workspace)).resolve()
    a800_heartbeat_path = (args.a800_heartbeat or _discover_a800_heartbeat(workspace)).resolve()
    a800_queue_path = (
        args.a800_queue or workspace / "workspace" / "hpc" / "a800_recovery_queue_current.json"
    ).resolve()
    a800_parallel_heartbeat_path = (
        args.a800_parallel_heartbeat
        or _discover_a800_parallel_heartbeat(workspace, a800_queue_path)
    ).resolve()
    a40_status_path = (
        args.a40_status
        or workspace
        / "workspace"
        / "hpc"
        / "mlebench_remote_ops"
        / "job89441_taxi_recovery_s42_20260726_121933_status_current.json"
    ).resolve()
    local_dog_plan_path = (
        args.local_dog_plan
        or workspace
        / "workspace"
        / "mlebench_plans"
        / "dog_breed_imagenet_head_recovery_current.json"
    ).resolve()
    local_dog_queue_path = (
        args.local_dog_queue
        or workspace
        / "workspace"
        / "local_gpu"
        / "local_dog_recovery_queue_current.json"
    ).resolve()
    dog_audit_path = (
        args.dog_audit
        or workspace
        / "workspace"
        / "mlebench_plans"
        / "dog_breed_gpt56_audit_current.json"
    ).resolve()
    taxi_source_audit_path = (
        args.taxi_source_audit
        or workspace
        / "workspace"
        / "mlebench_plans"
        / "taxi_source_audit_readiness_current.json"
    ).resolve()
    recovery_deep_review_path = (
        args.recovery_deep_review
        or workspace
        / "workspace"
        / "mlebench_plans"
        / "medal_recovery_deep_gpt56_review_current.json"
    ).resolve()
    human_gate_root = (
        args.human_gate_root or workspace / "workspace" / "human_gate"
    ).resolve()
    job88240_gpu_gate_path = (
        args.job88240_gpu_gate
        or workspace / "workspace" / "hpc" / "mlebench_remote_ops" / "gpu_gate_current.json"
    ).resolve()
    job89508_resource_probe_path = (
        args.job89508_resource_probe
        or workspace / "workspace" / "hpc" / "job89508_live_resource_probe_current.json"
    ).resolve()
    job88240_serial_chain_path = (
        args.job88240_serial_chain
        or workspace
        / "workspace"
        / "hpc"
        / "job88240_serial_chain_readonly_current.json"
    ).resolve()
    job89941_may_status_path = (
        args.job89941_may_status
        or workspace
        / "workspace"
        / "hpc"
        / "job89941_may2022_cpu_diagnostic"
        / "status_current.json"
    ).resolve()
    job89941_may_audit_path = (
        args.job89941_may_audit
        or workspace
        / "workspace"
        / "hpc"
        / "job89941_may2022_cpu_diagnostic"
        / "may2022_v2_result_audit_current.json"
    ).resolve()
    job89941_taxi_cpu_candidate_path = (
        args.job89941_taxi_cpu_candidate
        or workspace
        / "workspace"
        / "hpc"
        / "job89941_taxi_cpu_candidate_s43_v3"
        / "watcher_current.json"
    ).resolve()
    job89941_taxi_cpu_progress_path = (
        args.job89941_taxi_cpu_progress
        or workspace
        / "workspace"
        / "hpc"
        / "job89941_taxi_cpu_candidate_s43_v3"
        / "progress_current.json"
    ).resolve()
    job89941_taxi_cpu_confirmation_path = (
        args.job89941_taxi_cpu_confirmation
        or workspace
        / "workspace"
        / "hpc"
        / "job89941_taxi_cpu_confirmation"
        / "chain_current.json"
    ).resolve()
    job89771_taxi_longhaul_expert_path = (
        args.job89771_taxi_longhaul_expert
        or workspace
        / "workspace"
        / "hpc"
        / "job89771_taxi_longhaul_expert"
        / "collection_current.json"
    ).resolve()
    job89771_taxi_midhaul_expert_path = (
        args.job89771_taxi_midhaul_expert
        or workspace
        / "workspace"
        / "hpc"
        / "job89771_taxi_expert_campaign_current.json"
    ).resolve()
    job89941_leaf_status_path = (
        args.job89941_leaf_status
        or workspace
        / "workspace"
        / "hpc"
        / "job89941_leaf_decision_confirmation"
        / "status_current.json"
    ).resolve()
    cactus_cpu_sidecar_status_path = (
        args.cactus_cpu_sidecar_status
        or workspace
        / "workspace"
        / "hpc"
        / "cactus_cpu_sidecar_watcher"
        / "status_current.json"
    ).resolve()
    cactus_human_gate_postrun_path = (
        args.cactus_human_gate_postrun
        or workspace
        / "workspace"
        / "hpc"
        / "job88240_cactus_human_gate_postrun"
        / "status_current.json"
    ).resolve()
    ranzcr_human_gate_postrun_path = (
        args.ranzcr_human_gate_postrun
        or workspace
        / "workspace"
        / "hpc"
        / "job88240_ranzcr_human_gate_postrun"
        / "status_current.json"
    ).resolve()
    siim_human_gate_postrun_path = (
        args.siim_human_gate_postrun
        or workspace
        / "workspace"
        / "hpc"
        / "job88240_siim_human_gate_postrun"
        / "status_current.json"
    ).resolve()
    may2022_execution_queue_path = (
        args.may2022_execution_queue
        or workspace
        / "workspace"
        / "hpc"
        / "job88240_may2022_nested_queue_v8b_multiseed_final"
        / "status_current.json"
    ).resolve()
    may2022_supersession_seal_path = (
        args.may2022_supersession_seal
        or workspace
        / "workspace"
        / "hpc"
        / "may2022_supersession_seal_current.json"
    ).resolve()
    staged_human_gate_request_path = (
        args.staged_human_gate_request
        or workspace
        / "workspace"
        / "human_gate_staged_runs"
        / "approval_request_current.json"
    ).resolve()
    output = (args.output or workspace / "workspace" / "llm" / "mlebench_gpt56_adaptive_loop_current.json").resolve()
    snapshot_dir = workspace / "workspace" / "llm" / "mlebench_gpt56_adaptive_inputs"

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "created_at": _now(),
        "provider": "openai",
        "requested_model": REQUIRED_MODEL,
        "controller_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).read_bytes()),
        },
        "secret_policy": "The API key is loaded from DPAPI/environment and is never persisted in this report.",
        "human_gate_preserved": True,
        "kaggle_submission_enabled": False,
    }
    runtime: AgentRuntime | None = None
    try:
        _validate_environment()
        snapshots, competition_ids, source_hashes = build_evidence_snapshots(
            workspace,
            progress_path=progress_path,
            campaign_path=campaign_path,
            ranzcr_path=ranzcr_path,
            a800_heartbeat_path=a800_heartbeat_path,
            a800_parallel_heartbeat_path=a800_parallel_heartbeat_path,
            a800_queue_path=a800_queue_path,
            a40_status_path=a40_status_path,
            local_dog_plan_path=local_dog_plan_path,
            local_dog_queue_path=local_dog_queue_path,
            dog_audit_path=dog_audit_path,
            taxi_source_audit_path=taxi_source_audit_path,
            recovery_deep_review_path=recovery_deep_review_path,
            human_gate_root=human_gate_root,
            job88240_gpu_gate_path=job88240_gpu_gate_path,
            job89508_resource_probe_path=job89508_resource_probe_path,
            job88240_serial_chain_path=job88240_serial_chain_path,
            job89941_may_status_path=job89941_may_status_path,
            job89941_may_audit_path=job89941_may_audit_path,
            job89941_taxi_cpu_candidate_path=job89941_taxi_cpu_candidate_path,
            job89941_taxi_cpu_progress_path=job89941_taxi_cpu_progress_path,
            job89941_taxi_cpu_confirmation_path=job89941_taxi_cpu_confirmation_path,
            job89771_taxi_longhaul_expert_path=job89771_taxi_longhaul_expert_path,
            job89771_taxi_midhaul_expert_path=job89771_taxi_midhaul_expert_path,
            job89941_leaf_status_path=job89941_leaf_status_path,
            cactus_cpu_sidecar_status_path=cactus_cpu_sidecar_status_path,
            cactus_human_gate_postrun_path=cactus_human_gate_postrun_path,
            ranzcr_human_gate_postrun_path=ranzcr_human_gate_postrun_path,
            siim_human_gate_postrun_path=siim_human_gate_postrun_path,
            may2022_execution_queue_path=may2022_execution_queue_path,
            may2022_supersession_seal_path=may2022_supersession_seal_path,
            staged_human_gate_request_path=staged_human_gate_request_path,
            snapshot_dir=snapshot_dir,
        )
        active_runs = _active_runs_from_snapshots(snapshot_dir)
        campaign_snapshot = json.loads(
            (snapshot_dir / "campaign_state.json").read_text(encoding="utf-8")
        )
        blocked_collect_and_grade = _blocked_collect_and_grade_competitions(
            campaign_snapshot
        )
        runtime = AgentRuntime(
            workspace,
            runtime_root=workspace / "workspace" / "runtime" / "mlebench_gpt56_adaptive",
        )
        session = runtime.create_session(
            objective="Read current MLE-Bench evidence and select the next bounded medal-recovery actions.",
            title="MLE-Bench GPT-5.6 adaptive controller",
            permission_level="observe",
        )
        result = runtime.message(
            session["id"],
            _prompt(
                snapshots,
                source_hashes,
                active_runs,
                competition_ids,
                blocked_collect_and_grade,
            ),
            max_steps=RUNTIME_MAX_STEPS,
        )
        events = runtime.store.list_events(session["id"], limit=1000)
        observed, tool_errors = collect_tool_evidence(events, snapshots)
        model_execution = result.get("model_execution") or {}
        raw_decision = _extract_decision_json(str(result.get("text") or ""))
        decision, deterministic_adjustments = enforce_authoritative_active_run_policy(
            raw_decision, active_runs
        )
        decision_errors = validate_decision(
            decision,
            competition_ids=competition_ids,
            source_hashes=source_hashes,
            active_runs=active_runs,
            blocked_collect_and_grade=blocked_collect_and_grade,
        )
        errors = [*tool_errors, *decision_errors]
        if result.get("status") != "completed":
            errors.append(f"runtime status is {result.get('status')}")
        if model_execution.get("provider") != "openai":
            errors.append("served provider is not openai")
        if model_execution.get("model") != REQUIRED_MODEL:
            errors.append("served model is not gpt-5.6-sol")
        if int(model_execution.get("native_tool_calls", 0)) < len(REQUIRED_EVIDENCE):
            errors.append("native tool-call count is below the required evidence count")

        report.update(
            {
                "status": "passed" if not errors else "failed",
                "ok": not errors,
                "session_id": session["id"],
                "model_execution": model_execution,
                "snapshots": snapshots,
                "observed_tool_evidence": observed,
                "decision": decision,
                "raw_model_active_run_policy": raw_decision.get(
                    "active_run_policy"
                ),
                "deterministic_adjustments": deterministic_adjustments,
                "validation_errors": errors,
                "response": {
                    "characters": len(str(result.get("text") or "")),
                    "sha256": _sha256(str(result.get("text") or "").encode("utf-8")),
                },
            }
        )
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "ok": False,
                "error_type": type(exc).__name__,
            }
        )
    finally:
        if runtime is not None:
            runtime.close()

    report["runtime_seconds"] = round(time.perf_counter() - started_monotonic, 3)
    _write_json_atomic(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": report.get("status"),
                "ok": report.get("ok"),
                "model_execution": report.get("model_execution"),
                "decision": report.get("decision"),
                "validation_errors": report.get("validation_errors", []),
                "error_type": report.get("error_type"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
