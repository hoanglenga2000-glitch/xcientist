from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_local_gpu_serial_chain.py"
SPEC = importlib.util.spec_from_file_location("audit_local_gpu_serial_chain", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def record(stage: str, *, current: bool, alive: bool = True, signals: int = 0):
    return {
        "stage": stage,
        "status_path": f"{stage}.json",
        "worker_alive": alive,
        "loaded_plan_is_current": current,
        "loaded_plan_sha256": "current" if current else "old",
        "current_plan_sha256": "current",
        "launcher_script_identity_proven": current,
        "stderr_bytes": 0,
        "source_launch_guards": {
            "validates_frozen_plan": True,
            "rejects_changed_plan_hash_at_launch": True,
            "performs_final_idle_query": True,
            "requires_consecutive_idle_checks": True,
            "records_zero_process_signals": True,
            "requires_original_queue_wait_guard": True,
            "resumes_existing_verified_seed_artifacts": True,
            "rejects_existing_target_run": True,
        },
        "process_signals_sent": signals,
        "persistent_process_signals_sent": 0,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }


def current_records():
    return [record(spec.stage, current=True) for spec in MODULE.QUEUE_SPECS]


def test_one_current_queue_per_stage_passes_with_stale_legacy_waiters():
    records = current_records()
    records.append(record("may2022", current=False))
    result = MODULE.evaluate_runtime(records, [{"pid": 10}])
    assert result["passed"] is True
    assert result["stale_live_queue_count"] == 1
    assert result["logical_training_count"] == 1
    assert (
        result["verdict"]
        == "PASS_SINGLE_GPU_SERIAL_CHAIN_WITH_TERMINAL_STAGES_ALLOWED"
    )


def test_completed_stage_may_have_zero_live_queue_workers():
    records = [
        item for item in current_records() if item["stage"] != "may2022"
    ]
    result = MODULE.evaluate_runtime(records, [{"pid": 10}])

    assert result["passed"] is True
    assert result["current_live_queue_count_by_stage"]["may2022"] == 0
    assert result["checks"]["no_duplicate_current_live_queue_per_stage"] is True


def test_two_current_queues_for_one_stage_fail():
    records = current_records()
    records.append(record("may2022", current=True))
    result = MODULE.evaluate_runtime(records, [])
    assert result["passed"] is False
    assert result["checks"]["no_duplicate_current_live_queue_per_stage"] is False


def test_persistent_runtime_links_exact_manifest_when_domain_pid_is_omitted():
    class Worker:
        pid = 123

    assert MODULE.persistent_runtime_is_linked(
        {"status": "running"}, Worker(), None
    ) is True
    assert MODULE.persistent_runtime_is_linked(
        {"status": "completed"}, Worker(), None
    ) is False


def test_two_logical_training_jobs_fail():
    result = MODULE.evaluate_runtime(
        current_records(), [{"pid": 10}, {"pid": 20}]
    )
    assert result["passed"] is False
    assert result["checks"]["at_most_one_logical_gpu_training"] is False


def test_nonzero_process_signal_fails():
    records = current_records()
    records[0]["process_signals_sent"] = 1
    result = MODULE.evaluate_runtime(records, [])
    assert result["passed"] is False
    assert result["checks"]["queue_process_signals_zero"] is False


def test_nonzero_persistent_wrapper_signal_fails():
    records = current_records()
    records[0]["persistent_process_signals_sent"] = 1
    result = MODULE.evaluate_runtime(records, [])
    assert result["passed"] is False
    assert result["checks"]["queue_process_signals_zero"] is False


def test_current_launcher_without_script_identity_fails():
    records = current_records()
    records[0]["launcher_script_identity_proven"] = False
    result = MODULE.evaluate_runtime(records, [])
    assert result["passed"] is False
    assert (
        result["checks"]["current_queue_launcher_script_identities_proven"]
        is False
    )


def test_persistent_runtime_prefers_linked_running_candidate_over_old_failure():
    selected = MODULE.select_persistent_runtime(
        [
            {
                "persistent_status": "failed",
                "linked": False,
                "status_mtime_ns": 100,
                "name": "old",
            },
            {
                "persistent_status": "running",
                "linked": True,
                "status_mtime_ns": 90,
                "name": "current",
            },
        ]
    )

    assert selected is not None
    assert selected["name"] == "current"


def test_production_queue_sources_keep_serial_launch_guards():
    for spec in MODULE.QUEUE_SPECS:
        guards = MODULE.source_has_launch_guards(ROOT / spec.script)
        assert guards["validates_frozen_plan"] is True
        assert guards["requires_consecutive_idle_checks"] is True
        assert guards["records_zero_process_signals"] is True
        if spec.stage != "jigsaw_confirmation":
            assert guards["rejects_changed_plan_hash_at_launch"] is True
            assert guards["performs_final_idle_query"] is True
        if spec.stage == "jigsaw_confirmation_early":
            assert guards["requires_original_queue_wait_guard"] is True
            assert guards["resumes_existing_verified_seed_artifacts"] is True
        if spec.stage == "leaf_early":
            assert guards["requires_original_queue_wait_guard"] is True
            assert guards["rejects_existing_target_run"] is True
