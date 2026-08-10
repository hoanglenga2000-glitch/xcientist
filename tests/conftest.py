"""Shared pytest fixtures and path setup for the research workstation tests."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (SRC, ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


PORTABLE_SOURCE_CI = os.environ.get("EVOMIND_PORTABLE_SOURCE_CI", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# These tests deliberately bind immutable plans to workstation-local evidence
# that is not part of the source release.  They remain mandatory in the local
# production suite and are skipped only in the explicit portable-source job.
LOCAL_FROZEN_EVIDENCE_TESTS = frozenset(
    {
        "tests/test_audit_human_gate_readiness.py::test_cli_writes_machine_readable_readiness_audit",
        "tests/test_audit_human_gate_readiness.py::test_live_human_gate_readiness_audit_is_candidate_only",
        "tests/test_collect_job89941_taxi_cpu_candidate.py::test_collection_plan_accepts_frozen_launched_source_after_checkout_advances",
        "tests/test_deploy_job89941_leaf_decision_confirmation.py::test_all_remote_python_payloads_compile",
        "tests/test_deploy_job89941_leaf_decision_confirmation.py::test_argv_environment_and_stage_paths_are_cpu_only_and_confined",
        "tests/test_deploy_job89941_leaf_decision_confirmation.py::test_collection_reconnects_after_transport_drop_and_resumes",
        "tests/test_deploy_job89941_leaf_decision_confirmation.py::test_collection_rejects_existing_drifted_file",
        "tests/test_deploy_job89941_leaf_decision_confirmation.py::test_collection_rejects_final_sha_mismatch_without_atomic_promotion",
        "tests/test_deploy_job89941_leaf_decision_confirmation.py::test_collection_rejects_oversized_partial_before_connect",
        "tests/test_deploy_job89941_leaf_decision_confirmation.py::test_collection_resumes_existing_partial_from_exact_offset",
        "tests/test_deploy_job89941_leaf_decision_confirmation.py::test_collection_reuses_existing_exact_file_without_connecting",
        "tests/test_deploy_job89941_leaf_decision_confirmation.py::test_current_frozen_plan_is_complete_and_hash_verified",
        "tests/test_deploy_job89941_leaf_decision_confirmation.py::test_launch_is_exclusive_detached_and_contains_no_signal_api",
        "tests/test_deploy_job89941_leaf_decision_confirmation.py::test_plan_rejects_source_hash_drift",
        "tests/test_deploy_job89941_leaf_decision_confirmation.py::test_readonly_preflight_hashes_public_files_without_writes",
        "tests/test_deploy_job89941_leaf_decision_confirmation.py::test_staged_smoke_loads_override_manifest_and_both_caches",
        "tests/test_evaluate_spooky_cross_run_xgb.py::test_runtime_manifest_is_isolated_and_hash_verified",
        "tests/test_local_rtx4060_idle_gate.py::test_calibrated_idle_requires_every_hardware_check",
        "tests/test_local_rtx4060_idle_gate.py::test_frozen_policy_binds_baseline_and_execution_evidence",
        "tests/test_local_rtx4060_idle_gate.py::test_launch_contract_is_hash_bound_to_calibrated_policy",
        "tests/test_queue_hpc88240_may2022_cached.py::test_all_three_terminal_is_required_before_successor",
        "tests/test_queue_hpc88240_may2022_cached.py::test_current_execution_plan_and_bundle_are_frozen_and_candidate_only",
        "tests/test_queue_hpc88240_may2022_cached.py::test_ready_launch_keeps_human_gate_and_exact_plan_name",
        "tests/test_queue_hpc88240_may2022_cached.py::test_waiting_dependency_never_calls_gpu_or_launch",
        "tests/test_queue_leaf_after_siim.py::test_current_frozen_plan_and_every_declared_hash_are_verified",
        "tests/test_queue_leaf_after_siim.py::test_gpu_parser_and_command_preserve_serial_withheld_contract",
        "tests/test_queue_leaf_after_siim.py::test_leaf_waits_for_terminal_siim_final_verification",
        "tests/test_queue_leaf_after_siim.py::test_prerequisites_require_complete_leaf_data_and_siim_report",
        "tests/test_queue_leaf_after_siim.py::test_siim_report_fails_closed_on_grader_or_missing_report",
        "tests/test_queue_may2022_after_spooky.py::test_calibrated_idle_gate_rejects_active_python_and_accepts_wddm_baseline",
        "tests/test_queue_may2022_after_spooky.py::test_frozen_plan_and_command_preserve_serial_gpu_contract",
        "tests/test_queue_siim_after_jigsaw.py::test_calibrated_gate_rejects_active_python_and_accepts_wddm_baseline",
        "tests/test_queue_siim_after_jigsaw.py::test_current_frozen_plan_is_complete_and_hash_verified",
        "tests/test_queue_siim_after_jigsaw.py::test_gpu_parser_and_command_preserve_serial_contract",
        "tests/test_queue_siim_after_jigsaw.py::test_prerequisite_snapshot_requires_exact_complete_contract",
        "tests/test_queue_siim_final_candidate.py::test_current_plan_uses_calibrated_wddm_gate",
        "tests/test_queue_spooky_after_leaf.py::test_build_command_contains_frozen_runner_but_no_grader_or_submission",
        "tests/test_queue_spooky_confirmation_multiseed.py::test_calibrated_gate_binds_seed42_and_confirmation_manifest",
        "tests/test_queue_spooky_confirmation_multiseed.py::test_confirmation_commands_use_unique_run_ids_and_exact_seeds",
        "tests/test_queue_spooky_confirmation_multiseed.py::test_current_primary_snapshot_is_explicit_and_read_only",
        "tests/test_queue_spooky_confirmation_multiseed.py::test_failed_seed42_is_terminal_and_cannot_launch_confirmations",
        "tests/test_queue_spooky_confirmation_multiseed.py::test_manifest_and_confirmation_plans_are_frozen_and_hash_bound",
        "tests/test_queue_spooky_confirmation_multiseed.py::test_missing_seed_run_is_not_ready",
        "tests/test_queue_spooky_confirmation_multiseed.py::test_verifier_command_targets_each_confirmation_plan",
        "tests/test_queue_spooky_idle_slot_before_leaf.py::test_calibrated_gate_is_hash_bound_to_seed42_plan",
        "tests/test_queue_spooky_idle_slot_before_leaf.py::test_command_reuses_the_frozen_spooky_runner_and_seed",
        "tests/test_queue_spooky_idle_slot_before_leaf.py::test_live_frozen_plans_produce_an_eligible_or_priority_snapshot",
        "tests/test_watch_may2022_multiseed_completion.py::test_waiting_may_queue_never_collects_or_stages",
    }
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if not PORTABLE_SOURCE_CI:
        return
    skip = pytest.mark.skip(
        reason="requires workstation-local frozen evidence excluded from the portable source release"
    )
    for item in items:
        if item.nodeid.replace("\\", "/") in LOCAL_FROZEN_EVIDENCE_TESTS:
            item.add_marker(pytest.mark.local_frozen_evidence)
            item.add_marker(skip)
