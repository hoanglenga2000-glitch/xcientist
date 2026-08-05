from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.run_mlebench_gpt56_adaptive_loop import (
    DECISION_SCHEMA,
    REQUIRED_EVIDENCE,
    RUNTIME_MAX_STEPS,
    _active_runs_from_snapshots,
    _blocked_collect_and_grade_competitions,
    _discover_a800_parallel_heartbeat,
    _extract_decision_json,
    _fresh_runtime_evidence,
    _normalise_a40,
    _normalise_campaign,
    _normalise_cpu_status,
    _normalise_job89508_resource_probe,
    _normalise_job88240_serial_chain,
    _normalise_may2022_supersession_seal,
    _normalise_may_execution_queue,
    _normalise_recovery_deep_review,
    _normalise_taxi_cpu_candidate_status,
    _normalise_taxi_cpu_confirmation_chain,
    _normalise_taxi_cpu_progress,
    _normalise_taxi_expert_evidence,
    _normalise_taxi_source_audit,
    _prompt,
    collect_tool_evidence,
    enforce_authoritative_active_run_policy,
    validate_decision,
)

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_step_budget_covers_all_reads_and_final_decision():
    assert RUNTIME_MAX_STEPS >= len(REQUIRED_EVIDENCE) + 1


def test_authoritative_active_policy_drops_stale_model_inference():
    raw = {"active_run_policy": [{"lane": "a800", "run_id": "stale"}]}
    enforced, adjustments = enforce_authoritative_active_run_policy(raw, {})
    assert enforced["active_run_policy"] == []
    assert raw["active_run_policy"] == [{"lane": "a800", "run_id": "stale"}]
    assert adjustments[0]["reason"] == "deterministic freshness-derived runtime authority"


def test_cpu_status_accepts_missing_or_terminal_result_audit():
    status = {
        "schema": "cpu-status.v1",
        "status": "running",
        "run_id": "may-v2",
        "process": {"running": True},
    }
    source = {"path": "status.json", "sha256": "a" * 64}
    running = _normalise_cpu_status(status, source)
    assert "result_audit" not in running
    assert running["sources"] == {"status": source}

    audit_source = {"path": "audit.json", "sha256": "b" * 64}
    terminal = _normalise_cpu_status(
        {**status, "status": "completed"},
        source,
        result_audit={
            "status": "verified_negative_result",
            "metrics": {"gain_over_incumbent": -0.001},
            "verdict": {"promote_this_lightgbm_candidate": False},
        },
        result_audit_source=audit_source,
    )
    assert terminal["sources"]["result_audit"] == audit_source
    assert terminal["result_audit"]["metrics"]["gain_over_incumbent"] == -0.001


def test_job89508_probe_preserves_failed_gate_and_do_not_touch_processes():
    snapshot = _normalise_job89508_resource_probe(
        {
            "schema": "evomind.hpc.job89508.live_resource_probe.v1",
            "captured_at": "2026-07-28T14:40:39+08:00",
            "job_id": 89508,
            "identity": {
                "user": "aimslab",
                "hostname": "ef72fa298c6d",
                "remote_root": "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra",
                "root_exists": True,
                "root_readable": True,
                "root_writable": True,
            },
            "gpu": {
                "index": 0,
                "name": "NVIDIA A800-SXM4-80GB",
                "memory_total_mib": 81920,
                "memory_used_mib": 602,
                "memory_free_mib": 80619,
                "utilization_gpu_percent": 1,
                "compute_apps": [{"pid": 2020455, "used_memory_mib": 596}],
            },
            "existing_processes": [
                {
                    "pid": 1324428,
                    "state": "R",
                    "cpu_percent_probe": 677,
                    "command": "FlexMS_run.py --device cuda:0",
                    "nvidia_fd": "/dev/nvidia1",
                    "classification": "other_project_active",
                    "do_not_touch": True,
                }
            ],
            "runtime": {
                "default_python": "3.10.12",
                "default_user_site_numpy": "broken_namespace_missing_ndarray",
                "default_torch_import": "failed_due_to_numpy_ndarray",
                "unified_runtime_verified": False,
            },
            "training_gate": {
                "training_start_allowed": False,
                "reason_codes": [
                    "COMPUTE_APP_PRESENT",
                    "OTHER_PROJECT_CUDA_PROCESS_ACTIVE",
                    "DEFAULT_RUNTIME_IMPORT_CONTRACT_FAILED",
                ],
                "signals_sent": 0,
                "remote_files_modified": False,
                "parallel_training_started": False,
                "authoritative_three_sample_gate": {
                    "samples_required": 3,
                    "sample_interval_seconds": 15,
                    "compute_apps_required_empty": True,
                    "passed": False,
                    "samples": [
                        {
                            "captured_at": "2026-07-28T14:40:09+08:00",
                            "memory_used_mib": 602,
                            "utilization_gpu_percent": 0,
                            "compute_apps": ["2020455, [Not Found], 596"],
                            "idle": False,
                        }
                    ],
                },
            },
        },
        {"path": "job89508.json", "sha256": "a" * 64},
    )

    assert snapshot["job_id"] == 89508
    assert snapshot["training_gate"]["training_start_allowed"] is False
    assert snapshot["training_gate"]["passed"] is False
    assert snapshot["training_gate"]["samples"][0]["idle"] is False
    assert snapshot["existing_processes"][0]["do_not_touch"] is True
    assert "parallel A800 training" in snapshot["rule"]


def test_taxi_source_readiness_is_distinct_from_candidate_metric() -> None:
    normalized = _normalise_taxi_source_audit(
        {
            "schema": "evomind.mlebench.taxi_source_audit_readiness.v1",
            "status": "source_ready_candidate_execution_pending",
            "passed": True,
            "ready_for_candidate_execution": True,
            "candidate_metric_available": False,
            "external_seed_results_available": False,
            "checks": {
                "duplicate_safe_oof": {"passed": True},
                "fold_local_route_statistics": {
                    "passed": True,
                    "validation_target_rows_used": 0,
                },
            },
        },
        {"path": "taxi.json", "sha256": "a" * 64},
    )

    assert normalized["ready_for_candidate_execution"] is True
    assert normalized["candidate_metric_available"] is False
    assert normalized["checks"]["fold_local_route_statistics"] == {
        "passed": True,
        "validation_target_rows_used": 0,
    }


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def test_cli_help_bootstraps_src_without_pythonpath():
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_mlebench_gpt56_adaptive_loop.py"),
            "--help",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "evidence-driven EvoMind MLE-Bench decision turn" in completed.stdout


def test_active_run_detection_rejects_stale_running_heartbeats(tmp_path: Path):
    now = datetime(2026, 7, 27, 15, 0, tzinfo=timezone.utc)
    stale = {
        "process": "running",
        "run_id": "stale-a800",
        "created_at": (now - timedelta(hours=1)).isoformat(),
    }
    fresh = {
        "process": "running",
        "run_id": "fresh-parallel",
        "created_at": (now - timedelta(minutes=5)).isoformat(),
    }
    assert _fresh_runtime_evidence(stale, now=now) is False
    assert _fresh_runtime_evidence(fresh, now=now) is True

    (tmp_path / "a800_status.json").write_text(
        json.dumps({"heartbeat": stale}), encoding="utf-8"
    )
    (tmp_path / "a800_parallel_status.json").write_text(
        json.dumps({"heartbeat": fresh}), encoding="utf-8"
    )
    (tmp_path / "a40_taxi_status.json").write_text(
        json.dumps({"process": "stopped", "run": {}}), encoding="utf-8"
    )
    (tmp_path / "local4060_dog_status.json").write_text(
        json.dumps({"live_queue": {"status": "closed"}}), encoding="utf-8"
    )
    (tmp_path / "job89941_may_status.json").write_text(
        json.dumps({"status": "completed", "process": {"running": False}}),
        encoding="utf-8",
    )

    assert _active_runs_from_snapshots(tmp_path, now=now) == {
        "a800_parallel": "fresh-parallel"
    }


def _decision(source_hashes: dict[str, str]) -> dict:
    return {
        "schema": DECISION_SCHEMA,
        "leaderboard_assessment": {
            "official_scored": 19,
            "official_medals": 12,
            "target_medals": 18,
            "medal_deficit": 6,
            "strictly_above_top": False,
        },
        "active_run_policy": [
            {"lane": "a800", "run_id": "run-a800", "action": "continue_active_runs"},
            {
                "lane": "a800_parallel",
                "run_id": "run-jigsaw",
                "action": "continue_active_runs",
            },
            {"lane": "a40", "run_id": "run-a40", "action": "continue_active_runs"},
            {
                "lane": "local4060",
                "run_id": "run-dog",
                "action": "continue_active_runs",
            },
        ],
        "next_actions": [
            {
                "priority": 1,
                "action": "queue_experiment",
                "competition_id": "jigsaw",
                "lane": "a800",
                "expected_medal_gain": 1,
                "requires_active_run_completion": True,
                "rationale": "closest verified medal gap",
            }
        ],
        "evidence_source_sha256": dict(source_hashes),
        "stop_conditions": ["18 official medals"],
    }


def test_extract_decision_json_accepts_fenced_object():
    hashes = {name: _hash(name) for name in REQUIRED_EVIDENCE}
    encoded = json.dumps(_decision(hashes))
    result = _extract_decision_json(f"```json\n{encoded}\n```")
    assert result["schema"] == DECISION_SCHEMA


def test_validate_decision_enforces_active_runs_hashes_and_allowlists():
    hashes = {name: _hash(name) for name in REQUIRED_EVIDENCE}
    valid = _decision(hashes)
    assert (
        validate_decision(
            valid,
            competition_ids={"jigsaw"},
            source_hashes=hashes,
            active_runs={
                "a800": "run-a800",
                "a800_parallel": "run-jigsaw",
                "a40": "run-a40",
                "local4060": "run-dog",
            },
        )
        == []
    )

    invalid = _decision(hashes)
    invalid["next_actions"][0]["action"] = "interrupt_and_replace"
    invalid["active_run_policy"][0]["run_id"] = "wrong-run"
    invalid["evidence_source_sha256"]["official_progress"] = "bad-hash"
    errors = validate_decision(
        invalid,
        competition_ids={"jigsaw"},
        source_hashes=hashes,
        active_runs={
            "a800": "run-a800",
            "a800_parallel": "run-jigsaw",
            "a40": "run-a40",
            "local4060": "run-dog",
        },
    )
    assert "invalid action: interrupt_and_replace" in errors
    assert "active run mismatch: a800" in errors
    assert "source hash mismatch: official_progress" in errors


def test_campaign_snapshot_preserves_collection_and_promotion_truth():
    campaign = {
        "outcomes": [
            {
                "competition_id": "aptos",
                "run_id": "aptos-run",
                "finalized": True,
                "remote_summary_status": "partial_failure",
                "official_grade_count": 0,
                "promotion_gate_withheld": True,
                "collection": {
                    "passed": True,
                    "summary_path": "collected/aptos/summary.json",
                    "file_count": 10,
                },
                "progress_refresh": {
                    "updated": False,
                    "reason": "no_valid_official_private_grade",
                },
                "promotion_gate_results": [
                    {
                        "status": "promotion_gate_failed",
                        "promotion_gate": {
                            "name": "aptos_cross_fitted_qwk",
                            "internal_score": 0.914,
                            "threshold": 0.92,
                            "passed": False,
                            "checks": {"aggregate_threshold": False},
                        },
                    }
                ],
            }
        ]
    }
    snapshot = _normalise_campaign(campaign, {"path": "campaign.json", "sha256": "abc"})
    outcome = snapshot["outcomes"][0]
    assert outcome["official_grade_count"] == 0
    assert outcome["promotion_gate_withheld"] is True
    assert outcome["collection"]["passed"] is True
    assert outcome["progress_refresh"]["reason"] == "no_valid_official_private_grade"
    assert outcome["promotion_gate_results"][0]["promotion_gate"]["passed"] is False
    assert _blocked_collect_and_grade_competitions(snapshot) == {"aptos"}


def test_validate_decision_rejects_recollection_after_withheld_promotion_gate():
    hashes = {name: _hash(name) for name in REQUIRED_EVIDENCE}
    invalid = _decision(hashes)
    invalid["next_actions"][0].update(
        {
            "action": "collect_and_grade",
            "competition_id": "aptos",
            "lane": "cpu",
        }
    )
    errors = validate_decision(
        invalid,
        competition_ids={"aptos"},
        source_hashes=hashes,
        active_runs={
            "a800": "run-a800",
            "a800_parallel": "run-jigsaw",
            "a40": "run-a40",
            "local4060": "run-dog",
        },
        blocked_collect_and_grade={"aptos"},
    )
    assert (
        "collect_and_grade requires a new experiment after promotion gate withholding: aptos"
        in errors
    )


def test_collect_tool_evidence_requires_all_exact_snapshot_reads(tmp_path: Path):
    snapshots = {}
    events = []
    for index, name in enumerate(REQUIRED_EVIDENCE, 1):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"name": name}) + "\n", encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        snapshots[name] = {"path": str(path), "sha256": digest}
        call_id = f"call-{index}"
        events.extend(
            [
                {
                    "event_type": "tool.started",
                    "payload": {
                        "id": call_id,
                        "tool_name": "file_read",
                        "arguments": {"path": str(path)},
                    },
                },
                {
                    "event_type": "tool.completed",
                    "payload": {
                        "tool_call_id": call_id,
                        "content": {
                            "path": str(path),
                            "sha256": digest,
                            "line_count": 1,
                        },
                    },
                },
            ]
        )

    observed, errors = collect_tool_evidence(events, snapshots)
    assert errors == []
    assert set(observed) == set(REQUIRED_EVIDENCE)

    _observed, missing_errors = collect_tool_evidence(events[:-2], snapshots)
    assert f"required evidence was not read: {REQUIRED_EVIDENCE[-1]}" in missing_errors


def test_prompt_exposes_action_limit_and_blocked_collection_contract():
    hashes = {name: _hash(name) for name in REQUIRED_EVIDENCE}
    snapshots = {
        name: {"relative_path": f"snapshots/{name}.json", "line_count": 1}
        for name in REQUIRED_EVIDENCE
    }
    prompt = _prompt(
        snapshots,
        hashes,
        {"a800": "run-a800"},
        {"aptos", "jigsaw"},
        {"aptos"},
    )

    assert "Return between 1 and 8 next_actions" in prompt
    assert json.dumps(["aptos"], ensure_ascii=False) in prompt
    assert "never select collect_and_grade" in prompt
    assert "full-source recovery review" in prompt
    assert "recovery_deep_review.json" in prompt
    assert "one parallel tool-call batch" in prompt
    assert "job89508_resource_probe" in prompt
    assert "newer A800 safety interlock" in prompt


def test_recovery_deep_review_snapshot_preserves_go_no_go_and_bounded_order():
    value = {
        "schema": "deep-review.v1",
        "created_at": "2026-07-26T20:00:00+08:00",
        "parse_ok": True,
        "ok": True,
        "planner": {"model": "gpt-5.6-sol", "native": True},
        "review": {
            "audit_summary": {
                "current_coverage": 9,
                "required_coverage": 9,
                "critical_findings": ["SIIM uses untouched nested OOF"],
            },
            "per_competition": [
                {
                    "competition_id": "siim",
                    "deployment_verdict": "GO",
                    "estimated_medal_probability_after_upgrade": 0.5,
                    "compute_cost": "high",
                    "stop_rule": "two predetermined seeds",
                    "confirmed_source_defects": ["d1", "d2", "d3", "d4"],
                    "highest_value_upgrades": ["u1", "u2", "u3", "u4"],
                    "validation_contract": ["v1", "v2", "v3", "v4"],
                }
            ],
            "implementation_order": ["jigsaw", "siim"],
            "exact_medal_paths": {"base_case": 3},
            "resource_schedule": ["run SIIM after Jigsaw"],
            "global_stop_rules": ["18 official medals"],
        },
    }
    snapshot = _normalise_recovery_deep_review(
        value,
        {"path": "deep.json", "sha256": "abc"},
    )
    siim = snapshot["per_competition"][0]
    assert snapshot["ok"] is True
    assert snapshot["audit_summary"]["current_coverage"] == 9
    assert snapshot["implementation_order"] == ["jigsaw", "siim"]
    assert siim["deployment_verdict"] == "GO"
    assert siim["estimated_medal_probability_after_upgrade"] == 0.5
    assert siim["confirmed_source_defects"] == ["d1", "d2", "d3"]
    assert siim["highest_value_upgrades"] == ["u1", "u2", "u3"]
    assert siim["validation_contract"] == ["v1", "v2", "v3"]


def test_may_execution_queue_snapshot_preserves_serial_and_human_gates() -> None:
    value = {
        "schema": "evomind.hpc88240.may2022_nested_queue.v1",
        "created_at": "2026-07-28T00:23:15+08:00",
        "status": "waiting_for_cactus_terminal",
        "plan_path": "plan.json",
        "plan_sha256": "a" * 64,
        "bundle": {
            "sha256": "b" * 64,
            "archive_member_count": 702,
            "manifest_hash_count": 701,
            "human_gate_preserved": True,
            "passed": True,
        },
        "dependency": None,
        "process_signals_sent": 0,
        "other_processes_modified": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }

    snapshot = _normalise_may_execution_queue(
        value, {"path": "queue.json", "sha256": "c" * 64}
    )

    assert snapshot["status"] == "waiting_for_cactus_terminal"
    assert snapshot["bundle"]["human_gate_preserved"] is True
    assert snapshot["dependency"] is None
    assert snapshot["process_signals_sent"] == 0
    assert snapshot["official_grader_executed"] is False
    assert snapshot["kaggle_submission_executed"] is False


def test_may_cached_queue_snapshot_exposes_verified_cache_without_predictions() -> None:
    root = Path(__file__).resolve().parents[1]
    plan_path = (
        root
        / "workspace"
        / "mlebench_plans"
        / "may2022_nested_selection_execution_s42_hpc88240_v3_cache_20260728.json"
    )
    value = {
        "schema": "evomind.hpc88240.may2022_cached_queue.v1",
        "created_at": "2026-07-28T01:25:59+08:00",
        "status": "waiting_for_cactus_terminal",
        "plan_path": str(plan_path),
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "bundle": {"human_gate_preserved": True, "passed": True},
        "dependency": None,
        "process_signals_sent": 0,
        "other_processes_modified": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }

    snapshot = _normalise_may_execution_queue(
        value, {"path": "queue-v3.json", "sha256": "c" * 64}
    )

    cache = snapshot["public_precomputed_cache"]
    assert cache["required"] is True
    assert cache["status"] == "completed"
    assert cache["visibility_mode"] == "PUBLIC_ONLY"
    assert cache["train_rows"] == 800_000
    assert cache["feature_count"] == 276
    assert cache["official_grader_executed"] is False
    assert len(cache["manifest_sha256"]) == 64
    assert snapshot["performance_contract"]["feature_stage_speedup"] > 1.0
    assert snapshot["plan_source"]["sha256"] == snapshot["plan_sha256"]


def test_job88240_serial_chain_snapshot_prevents_duplicate_queue_inference() -> None:
    value = {
        "schema": "evomind.hpc88240.serial_chain_readonly_probe.v1",
        "created_at": "2026-07-28T00:46:04+08:00",
        "statuses": [
            {
                "path": "/dedicated/full_siim_status.json",
                "exists": True,
                "payload": {
                    "schema": "evomind.hpc_siim_persistent_run.v1",
                    "created_at": "2026-07-28T00:46:03+08:00",
                    "status": "waiting_for_leaf",
                    "wrapper_pid": 692154,
                    "process_signals_sent": 0,
                    "official_grader_executed": False,
                    "kaggle_submission_executed": False,
                },
            }
        ],
        "remote_probe_exit_code": 0,
        "remote_probe_stdout": (
            "0, NVIDIA A40, 13021, 100\n__APPS__\n"
            "695731, /existing/python, 13012\n__PROCS__\n"
        ),
        "process_signals_sent": 0,
        "other_processes_modified": False,
        "local_successor_chain": {
            "passed": True,
            "expected_successor_count": 4,
            "observed_successor_count": 4,
            "all_launchers_running": True,
            "all_wrappers_live": True,
            "all_workers_live": True,
            "all_evidence_statuses_present": True,
            "successors": [
                {
                    "name": "may_queue",
                    "task_id": "job88240_final_successor_may_queue",
                    "launcher_status": "running",
                    "launcher_observed_at": "2026-07-28T12:47:38+08:00",
                    "wrapper_process": {"pid": 4268, "exists": True},
                    "worker_process": {"pid": 3308, "exists": True},
                    "evidence_status_path": "may/status_current.json",
                    "evidence_observed_at": "2026-07-28T12:47:40+08:00",
                    "evidence_status": "waiting_for_cactus_terminal",
                    "plan_sha256": "e" * 64,
                }
            ],
        },
    }

    snapshot = _normalise_job88240_serial_chain(
        value, {"path": "serial.json", "sha256": "d" * 64}
    )

    assert snapshot["stages"][0]["status"] == "waiting_for_leaf"
    assert snapshot["stages"][0]["wrapper_pid"] == 692154
    assert snapshot["gpu_lines"] == ["0, NVIDIA A40, 13021, 100"]
    assert snapshot["compute_apps"] == ["695731, /existing/python, 13012"]
    assert snapshot["local_successor_chain"]["passed"] is True
    assert snapshot["local_successor_chain"]["successors"][0]["worker_live"] is True
    assert "Do not queue duplicate" in snapshot["rule"]
    assert snapshot["process_signals_sent"] == 0


def test_may2022_supersession_snapshot_preserves_unique_launch_authority() -> None:
    snapshot = _normalise_may2022_supersession_seal(
        {
            "schema": "evomind.local_queue_supersession_seal_report.v1",
            "created_at": "2026-07-28T05:45:33+08:00",
            "status": "sealed",
            "authoritative_queue_dir": "final-v8b",
            "authoritative_plan_sha256": "a" * 64,
            "authoritative_bundle_sha256": "b" * 64,
            "superseded_queue_count": 1,
            "superseded_queues": [
                {
                    "queue_dir": "legacy-v8",
                    "status_before_seal": "waiting_for_cactus_terminal",
                    "loaded_plan_sha256": "c" * 64,
                    "current_plan_path_sha256": "a" * 64,
                    "plan_path_hash_matches_loaded_status": False,
                    "claim_path": "legacy-v8/launch_claim_s42.json",
                    "claim_sha256": "d" * 64,
                    "ignored_field": "large-or-unstable",
                }
            ],
            "authoritative_queue_sealed": False,
            "watchers_stopped": 0,
            "process_signals_sent": 0,
            "other_processes_modified": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        },
        {"path": "seal.json", "sha256": "e" * 64},
    )

    assert snapshot["status"] == "sealed"
    assert snapshot["superseded_queue_count"] == 1
    assert snapshot["authoritative_queue_sealed"] is False
    assert snapshot["watchers_stopped"] == 0
    assert snapshot["superseded_queues"][0]["claim_sha256"] == "d" * 64
    assert "ignored_field" not in snapshot["superseded_queues"][0]
    assert "Only the declared authoritative May queue" in snapshot["rule"]


def test_taxi_cpu_candidate_snapshot_is_candidate_only() -> None:
    snapshot = _normalise_taxi_cpu_candidate_status(
        {
            "schema": "evomind.hpc.job89941_taxi_cpu_candidate_watcher.v1",
            "created_at": "2026-07-28T06:09:00+08:00",
            "status": "running",
            "remote_status": {
                "state": {"run_id": "taxi-cpu-43", "plan_sha256": "a" * 64},
                "process": {"pid": 123, "running": True, "cmdline_matches": True},
                "result": None,
                "process_signals_sent": 0,
                "other_processes_modified": False,
                "official_grader_executed": False,
                "kaggle_submission_executed": False,
            },
        },
        {"path": "watcher.json", "sha256": "b" * 64},
    )
    assert snapshot["status"] == "running"
    assert snapshot["run_id"] == "taxi-cpu-43"
    assert snapshot["process"]["running"] is True
    assert snapshot["result"] is None
    assert "candidate evidence only" in snapshot["rule"]


def test_taxi_cpu_confirmation_snapshot_preserves_serial_successor() -> None:
    snapshot = _normalise_taxi_cpu_confirmation_chain(
        {
            "schema": "evomind.hpc.job89941_taxi_cpu_confirmation_chain.v1",
            "created_at": "2026-07-28T06:24:06+08:00",
            "status": "waiting_for_parent_gate",
            "current_seed": 44,
            "parent_gate": {"ready": False, "parent_seed": None},
            "completed": [],
            "process_signals_sent": 0,
            "other_processes_modified": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        },
        {"path": "chain.json", "sha256": "c" * 64},
    )
    assert snapshot["current_seed"] == 44
    assert snapshot["parent_gate"]["ready"] is False
    assert snapshot["completed"] == []
    assert "strict serial successor" in snapshot["rule"]


def test_taxi_cpu_progress_snapshot_uses_cpu_delta_not_log_only() -> None:
    snapshot = _normalise_taxi_cpu_progress(
        {
            "schema": "evomind.hpc.job89941_taxi_cpu_progress.v1",
            "created_at": "2026-07-28T06:39:52+08:00",
            "status": "running",
            "plan_sha256": "a" * 64,
            "sample": {
                "pid": 388586,
                "process_exists": True,
                "process_state": "R",
                "utime_ticks": 746938,
                "log_bytes": 0,
                "result_exists": False,
            },
            "delta_from_previous": {
                "utime_ticks": 4168,
                "log_bytes": 0,
                "making_progress": True,
            },
            "stalled": False,
            "process_signals_sent": 0,
            "other_processes_modified": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        },
        {"path": "progress.json", "sha256": "b" * 64},
    )
    assert snapshot["sample"]["log_bytes"] == 0
    assert snapshot["delta_from_previous"]["making_progress"] is True
    assert snapshot["stalled"] is False
    assert "buffered training log" in snapshot["rule"]


def test_taxi_expert_snapshot_preserves_diagnostic_boundary() -> None:
    snapshot = _normalise_taxi_expert_evidence(
        {
            "schema": "evomind.hpc.job89771_taxi_longhaul_expert_collection.v1",
            "created_at": "2026-07-28T10:40:00+08:00",
            "status": "completed_and_cross_node_verified",
            "result": {
                "status": "diagnostic_improvement_confirmed",
                "passed": True,
                "model_family": "LightGBM_CPU_cross_fitted_longhaul_expert",
                "seed": 46,
                "folds": 3,
                "distance_threshold_km": 6.0,
                "expert_weight": 0.75,
                "longhaul_rows": 681643,
                "metrics": {
                    "base_random_oof_rmse": 4.529,
                    "blended_random_oof_rmse": 3.089,
                },
                "candidate_only": True,
                "private_labels_used": False,
                "official_grader_executed": False,
                "kaggle_submission_executed": False,
                "gpu_used": False,
            },
            "independent_verification": {
                "verifier_node": "job89941_cpu",
                "recomputed_blended_rmse": 3.089,
                "diagnostic_global_optimal_rmse": 2.970,
                "official_grader_executed": False,
                "kaggle_submission_executed": False,
                "gpu_used": False,
            },
            "candidate_ready_for_nested_production_confirmation": True,
            "process_signals_sent": 0,
            "other_processes_modified": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        },
        {"path": "collection.json", "sha256": "a" * 64},
    )

    assert snapshot["status"] == "completed_and_cross_node_verified"
    assert snapshot["result"]["metrics"]["blended_random_oof_rmse"] == 3.089
    assert (
        snapshot["independent_verification"]["diagnostic_global_optimal_rmse"]
        == 2.970
    )
    assert snapshot["candidate_ready_for_nested_production_confirmation"] is True
    assert "not an official score or medal" in snapshot["rule"]


def test_taxi_expert_campaign_snapshot_closes_failed_cpu_sweep() -> None:
    snapshot = _normalise_taxi_expert_evidence(
        {
            "schema": "evomind.hpc.job89771_taxi_expert_campaign.v1",
            "created_at": "2026-07-28T12:32:00+08:00",
            "status": "diagnostic_stop_no_promotion",
            "best_single_oof_rmse": 2.869,
            "best_single_experiment": "shorthaul_s48",
            "best_disjoint_combination_oof_rmse": 2.866,
            "promotion_contract": {"maximum_random_oof_rmse": 2.85},
            "promotion_passed": False,
            "candidate_ready_for_human_gate": False,
            "stop_reason": "No bounded CPU expert reached the promotion threshold.",
            "next_authority": "Preserve the source-audited A40 Taxi path.",
            "experiments": [
                {
                    "name": "shorthaul_s48",
                    "status": "diagnostic_improvement_insufficient",
                    "passed": False,
                    "random_oof_rmse": 2.869,
                    "official_grader_executed": False,
                    "kaggle_submission_executed": False,
                    "gpu_used": False,
                }
            ],
            "process_signals_sent": 0,
            "other_processes_modified": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        },
        {"path": "campaign.json", "sha256": "b" * 64},
    )

    assert snapshot["status"] == "diagnostic_stop_no_promotion"
    assert snapshot["campaign"]["promotion_passed"] is False
    assert snapshot["campaign"]["experiments"][0]["name"] == "shorthaul_s48"
    assert "do not duplicate a closed CPU sweep" in snapshot["rule"]


def test_normalise_a40_compacts_terminal_summary_without_losing_gate_truth():
    value = {
        "process": "stopped",
        "state": {"run_id": "taxi-run", "pid": 123, "status": "running"},
        "manifest": {"run_id": "taxi-run", "requested_competitions": ["taxi"]},
        "checkpoint": {
            "schema": "checkpoint.v1",
            "run_id": "taxi-run",
            "completed": {"taxi": "promotion_gate_failed"},
            "remaining": [],
        },
        "summary": {
            "schema": "summary.v1",
            "run_id": "taxi-run",
            "status": "partial_failure",
            "passed": 0,
            "failed": 1,
            "results": [
                {
                    "competition_id": "taxi",
                    "status": "promotion_gate_failed",
                    "cv_score": 2.87,
                    "official_grader_executed": False,
                    "official_grader_withheld": True,
                    "promotion_gate": {"passed": False, "threshold": 2.74},
                    "gpu_telemetry": {"samples": [{"used": i} for i in range(10000)]},
                }
            ],
        },
        "log_tail": "status=promotion_gate_failed",
    }

    snapshot = _normalise_a40(value, {"path": "status.json", "sha256": "abc"})
    result = snapshot["summary"]["results"][0]
    assert snapshot["process"] == "stopped"
    assert snapshot["checkpoint"]["completed"] == {"taxi": "promotion_gate_failed"}
    assert result["official_grader_withheld"] is True
    assert result["promotion_gate"]["passed"] is False
    assert "gpu_telemetry" not in result
    assert len(json.dumps(snapshot)) < 5000


def test_parallel_heartbeat_discovery_recovers_from_transient_queue_marker(
    tmp_path: Path,
):
    queue_path = tmp_path / "workspace" / "hpc" / "queue.json"
    queue_path.parent.mkdir(parents=True)
    queue_path.write_text(
        json.dumps({"status": "launch_retry_pending"}), encoding="utf-8"
    )
    watch_root = tmp_path / "workspace" / "hpc" / "a800_lane_watch"
    watch_root.mkdir(parents=True)
    heartbeat = watch_root / "jigsaw_parallel_heartbeat_current.json"
    heartbeat.write_text(json.dumps({"process": "running"}), encoding="utf-8")

    assert _discover_a800_parallel_heartbeat(tmp_path, queue_path) == heartbeat
