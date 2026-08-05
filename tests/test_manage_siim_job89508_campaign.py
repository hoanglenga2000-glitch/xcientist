"""Contracts for the isolated SIIM job89508 campaign manager and runner."""
from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from paramiko import SSHException

from scripts import manage_siim_job89508_campaign as manager
from scripts import run_siim_job89508_campaign as runner


def test_campaign_plan_binds_disjoint_seeds_budget_and_resource_policy():
    manifest = {"bundle_sha256": "a" * 64}
    plan = manager.build_campaign_plan("siim_contract_test", manifest, "b" * 64)
    assert plan["schema"] == "evomind.siim.hpc_campaign_plan.v1"
    assert plan["ablation_seeds"] == [40, 41, 42]
    assert plan["formal_seeds"] == [43, 44, 45]
    assert {item["seed"] for item in plan["formal_runs"]} == {43, 44, 45}
    assert plan["resource_policy"]["memory_limit_mib"] == 55 * 1024
    assert plan["resource_policy"]["effective_batch_size"] == 384
    assert plan["resource_policy"]["max_workers"] == 8
    assert plan["budget_hours"] == {
        "ablation": 4,
        "formal_training": 72,
        "delivery": 2,
        "total": 78,
    }
    assert plan["budget_policy"] == "user_selected_A_corrected_full_closure"
    assert plan["original_budget_hours"]["total"] == 24
    assert plan["corrected_protocol_version"] == "siim_exact_duplicate_grouping_v2"
    assert plan["remote_storage_policy"] == manager.build_remote_storage_policy(
        "siim_contract_test"
    )
    assert plan["remote_storage_policy"]["allowed_root"] == manager.ALLOWED_GPU_REMOTE_ROOT
    assert plan["remote_storage_policy"]["external_remote_writes"] == "forbidden"
    assert plan["official_submission"] == "forbidden"
    assert plan["private_grader"] == "once_after_candidate_freeze"


def test_legacy_plan_is_readable_for_upgrade_but_not_launchable():
    plan = manager.build_campaign_plan(
        "siim_legacy_upgrade",
        {"bundle_sha256": "a" * 64},
        "b" * 64,
    )
    plan["budget_hours"] = {
        "ablation": 4,
        "formal_training": 18,
        "delivery": 2,
        "total": 24,
    }
    plan.pop("budget_policy")
    plan.pop("original_budget_hours")
    plan.pop("corrected_protocol_version")

    assert manager.validate_campaign_plan_identity(plan, "siim_legacy_upgrade")
    with pytest.raises(manager.CampaignManagerError, match="corrected budget policy"):
        manager.validate_campaign_plan(plan, "siim_legacy_upgrade")


def test_job90353_binding_reaches_all_orchestration_modules():
    environment = os.environ.copy()
    environment.update(
        {
            "EVOMIND_SIIM_HPC_JOB_ID": "90353",
            "EVOMIND_HPC_CREDENTIAL_PROFILE": "job90353",
            "EVOMIND_SIIM_RUN_ID": "evomind_siim_job90353_contract",
        }
    )
    source = """
import json
from scripts import build_siim_workflow_ingress as ingress
from scripts import manage_siim_job89508_campaign as manager
from scripts import prepare_siim_job89508_runtime as runtime
from scripts import run_siim_private_grader_once as grader
print(json.dumps({
    'manager': [manager.HPC_JOB_ID, manager.CREDENTIAL_PROFILE, manager.JOB_TAG,
                str(manager.LOCAL_ROOT), manager.REMOTE_CAMPAIGN_PARENT, manager.DEFAULT_RUN_ID],
    'runtime': [runtime.HPC_JOB_ID, runtime.CREDENTIAL_PROFILE, runtime.JOB_TAG,
                str(runtime.DEFAULT_REQUIREMENTS), runtime.REMOTE_RUNTIME_ROOT],
    'ingress': [ingress.HPC_JOB_ID, ingress.CREDENTIAL_PROFILE, ingress.JOB_TAG,
                str(ingress.DEFAULT_CAMPAIGN_PARENT), str(ingress.DEFAULT_CANDIDATE_PARENT)],
    'grader': [grader.HPC_JOB_ID, grader.CREDENTIAL_PROFILE, grader.JOB_TAG,
               grader.REMOTE_PRIVATE_GRADER_PARENT_RELATIVE],
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=manager.PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    serialized = json.dumps(payload)

    assert payload["manager"][:3] == [90353, "job90353", "job90353"]
    assert payload["runtime"][:3] == [90353, "job90353", "job90353"]
    assert payload["ingress"][:3] == [90353, "job90353", "job90353"]
    assert payload["grader"][:3] == [90353, "job90353", "job90353"]
    assert "job90353" in serialized
    assert "evomind_siim_job90353_contract" in serialized
    assert "siim_job89508/runtime-requirements.txt" not in serialized.replace("\\\\", "/")


def test_bundle_identity_changes_with_source_hash():
    first = manager.bundle_manifest(
        [{"path": "scripts/a.py", "bytes": 1, "sha256": "1" * 64, "local": "a"}]
    )
    second = manager.bundle_manifest(
        [{"path": "scripts/a.py", "bytes": 1, "sha256": "2" * 64, "local": "a"}]
    )
    assert first["bundle_sha256"] != second["bundle_sha256"]
    assert first["official_submission_enabled"] is False
    assert first["private_grader_in_training_enabled"] is False


def test_campaign_runner_enforces_memory_monitoring_resume_and_no_signals():
    source = inspect.getsource(runner)
    assert '"--memory-limit-mib", str(55 * 1024)' in source
    assert '"--siim-memory-limit-mib", str(55 * 1024)' in source
    assert '"--siim-effective-batch-size", "384"' in source
    assert '"--siim-workers", "8"' in source
    assert 'if (ablation_root / "runs" / ablation_id).exists()' in source
    assert 'command.append("--resume")' in source
    assert "resolve_formal_runtime_budget" in source
    assert 'run_root / COMPETITION / "attempts" / "siim_resume_state"' in source
    assert '"--siim-image-content-manifest"' in source
    assert "CORRECTED_FORMAL_BUDGET_SECONDS = 72 * 3600" in source
    assert '"pause_after_epoch"' in source
    assert '"signals_sent": 0' in source
    assert '"other_processes_modified": False' in source
    assert "os.kill" not in source


def test_persisted_formal_budget_is_validated_and_reused_exactly(monkeypatch, tmp_path):
    resume = tmp_path / "siim_resume_state"
    resume.mkdir()
    state = resume / "runtime_budget.json"
    state.write_text(
        json.dumps({
            "schema": "evomind.siim.runtime_budget.v1",
            "started_unix": 100.5,
            "runtime_budget_seconds": 56589.75,
            "model_seed": 43,
        }),
        encoding="utf-8",
    )
    before = state.read_bytes()
    monkeypatch.setattr(runner, "siim_resume_root", lambda _run_root: resume)

    resolved = runner.resolve_formal_runtime_budget(
        tmp_path,
        model_seed=43,
        remaining_seconds=123.0,
    )

    assert resolved == 56589.75
    assert state.read_bytes() == before
    with pytest.raises(runner.CampaignError, match="seed changed"):
        runner.resolve_formal_runtime_budget(
            tmp_path,
            model_seed=44,
            remaining_seconds=123.0,
        )


def test_incompatible_same_run_directory_is_preserved_without_deletion(monkeypatch, tmp_path):
    source = tmp_path / "siim_resume_state"
    source.mkdir()
    checkpoint = source / "stage_outer00_inner00_selection.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(runner, "ensure_within", lambda path, root=runner.ALLOWED_ROOT: Path(path))

    preserved, audit = runner.supersede_directory(
        source,
        reason="fixture_contract_change",
    )

    assert not source.exists()
    assert preserved.is_dir()
    assert (preserved / checkpoint.name).read_bytes() == b"checkpoint"
    assert (preserved / "supersession_record.json").is_file()
    assert audit["files_deleted"] == 0
    assert audit["signals_sent"] == 0
    assert audit["other_processes_modified"] is False


def _corrected_formal_fixture(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        runner,
        "ensure_within",
        lambda path, root=runner.ALLOWED_ROOT: Path(path),
    )
    run_root = tmp_path / "formal_seed_43"
    resume = (
        run_root
        / runner.COMPETITION
        / "attempts"
        / "siim_resume_state"
    )
    resume.mkdir(parents=True)
    bundle = tmp_path / "bundle"
    scripts = bundle / "scripts"
    scripts.mkdir(parents=True)
    adapter = scripts / "mlebench_medal_recovery_adapters.py"
    wave2 = scripts / "mlebench_wave2_adapters.py"
    adapter.write_bytes(b"adapter-v1")
    wave2.write_bytes(b"wave2-v1")
    manifest_sha256 = "a" * 64
    contract = {
        "schema": runner.FORMAL_RESUME_SCHEMA,
        "competition_id": runner.COMPETITION,
        "model_seed": 43,
        "image_content_manifest_sha256": manifest_sha256,
        "leakage_group_policy": runner.FORMAL_GROUP_POLICY,
        "perceptual_edge_policy": runner.FORMAL_PERCEPTUAL_POLICY,
        "outer_folds": 5,
        "inner_folds": 3,
        "model": {"workers": 8, "fast_kernel_mode": True},
        "adapter_source_sha256": runner.sha256_file(adapter),
        "wave2_source_sha256": runner.sha256_file(wave2),
    }
    contract["contract_sha256"] = runner._contract_sha256(contract)
    runner.atomic_json(resume / "resume_contract.json", contract)
    groups = {
        "schema": runner.FORMAL_GROUP_SCHEMA,
        "leakage_group_policy": runner.FORMAL_GROUP_POLICY,
        "perceptual_edge_policy": runner.FORMAL_PERCEPTUAL_POLICY,
        "perceptual_edges_applied_to_groups": 0,
    }
    result = {
        "competition_id": runner.COMPETITION,
        "status": "candidate_ready_confirmation_pending",
        "candidate_only": True,
        "valid_submission": True,
        "official_grader_executed": False,
        "budget": {
            "seed": 43,
            "folds": 5,
            "image_content_manifest_sha256": manifest_sha256,
            "resume_contract_sha256": contract["contract_sha256"],
            "duplicate_group_report": groups,
        },
    }
    return run_root, bundle, manifest_sha256, contract, result


def test_corrected_terminal_result_requires_exact_grouping_and_resume_contract(
    monkeypatch,
    tmp_path,
):
    run_root, bundle, manifest_sha256, _contract, result = _corrected_formal_fixture(
        monkeypatch,
        tmp_path,
    )

    checks = runner.corrected_formal_result_checks(
        run_root,
        result,
        expected_seed=43,
        expected_manifest_sha256=manifest_sha256,
        bundle_root=bundle,
    )

    assert runner.all_checks_pass(checks)
    assert set(runner.TERMINAL_CANDIDATE_STATES) == {
        "candidate_ready_confirmation_pending",
        "promotion_gate_passed_confirmation_pending",
    }
    assert "passed" not in runner.TERMINAL_CANDIDATE_STATES


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (lambda result, contract: result.update(status="passed"), "terminal_status"),
        (lambda result, contract: result["budget"].update(folds=2), "outer_folds_exact"),
        (
            lambda result, contract: result["budget"]["duplicate_group_report"].update(
                perceptual_edges_applied_to_groups=1
            ),
            "perceptual_edges_not_applied",
        ),
        (
            lambda result, contract: contract.update(outer_folds=2),
            "resume_contract_outer",
        ),
    ],
)
def test_incompatible_terminal_result_fails_closed(
    monkeypatch,
    tmp_path,
    mutation,
    failed_check,
):
    run_root, bundle, manifest_sha256, contract, result = _corrected_formal_fixture(
        monkeypatch,
        tmp_path,
    )
    mutation(result, contract)
    runner.atomic_json(
        runner.siim_resume_root(run_root) / "resume_contract.json",
        contract,
    )

    checks = runner.corrected_formal_result_checks(
        run_root,
        result,
        expected_seed=43,
        expected_manifest_sha256=manifest_sha256,
        bundle_root=bundle,
    )

    assert checks[failed_check] is False
    assert not runner.all_checks_pass(checks)


def test_remote_campaign_launch_uses_the_same_isolated_import_environment_as_prepare():
    source = inspect.getsource(manager.launch)
    assert '"PYTHONNOUSERSITE=1"' in source
    assert '"PYTHONPATH="' in source
    assert "{remote['bundle']}:{remote['bundle']}/scripts:{remote['bundle']}/src" in source
    assert source.index('"PYTHONNOUSERSITE=1"') < source.index("remote[\"runtime_python\"]")
    assert "isolated_environment" in source
    assert '"-u", "KAGGLE_USERNAME"' in source
    assert '"-u", "KAGGLE_KEY"' in source


def test_campaign_child_environment_confines_home_cache_temp_and_bytecode(
    monkeypatch, tmp_path: Path
):
    allowed = tmp_path / "operator_remote_root"
    bundle = allowed / f"siim_{runner.BINDING.job_tag}" / "bundles" / ("a" * 64)
    campaign = (
        allowed
        / f"siim_{runner.BINDING.job_tag}"
        / "campaigns"
        / "siim_environment_contract"
    )
    torch_home = allowed / "mlebench_model_cache" / "torch"
    monkeypatch.setattr(runner, "ALLOWED_ROOT", allowed)
    monkeypatch.setenv("KAGGLE_USERNAME", "must_not_propagate")
    monkeypatch.setenv("KAGGLE_KEY", "must_not_propagate")

    environment = runner.command_environment(bundle, torch_home, campaign)

    assert "KAGGLE_USERNAME" not in environment
    assert "KAGGLE_KEY" not in environment
    for name in runner.DEDICATED_ENVIRONMENT_KEYS:
        path = Path(environment[name]).resolve()
        assert path == allowed.resolve() or allowed.resolve() in path.parents
        assert path.is_dir()


def test_remote_and_bundle_paths_cannot_escape_dedicated_root():
    with pytest.raises(manager.CampaignManagerError, match="escaped"):
        manager.ensure_remote("/tmp/outside")
    with pytest.raises(runner.CampaignError, match="escaped"):
        runner.ensure_within(Path("/tmp/outside"))
    with pytest.raises(manager.CampaignManagerError, match="escaped"):
        manager.ensure_remote(
            manager.ALLOWED_GPU_REMOTE_ROOT + "/../outside"
        )


def test_venv_interpreter_validation_preserves_the_lexical_entry_point(tmp_path):
    root = tmp_path / "allowed"
    base = root / "base" / "python"
    base.parent.mkdir(parents=True)
    base.write_bytes(b"fixture")
    entry = root / "venv" / "bin" / "python"
    entry.parent.mkdir(parents=True)
    try:
        entry.symlink_to(base)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows host")

    preserved = runner.ensure_venv_python(entry, root)

    assert preserved == entry.absolute()
    assert preserved != entry.resolve()
    assert preserved.resolve() == base.resolve()


def test_collection_contract_contains_aggregator_material():
    for name in (
        "siim_fold_ensemble.npz",
        "siim_oof_predictions.csv",
        "submission.csv",
        "result.json",
    ):
        assert name in manager.COLLECT_NAMES


def test_mutating_manager_connection_uses_strict_same_connection_identity_gate():
    source = inspect.getsource(manager.connect)
    assert "strict_named_profile=True" in source
    assert "verify_job_container_identity" in source
    assert source.index("verify_job_container_identity") < source.index("return client")


def test_remote_launch_and_children_pin_working_directory_and_umask():
    launch_source = inspect.getsource(manager.launch)
    phase_source = inspect.getsource(runner.run_phase)
    assert "cd --" in launch_source
    assert "umask 077" in launch_source
    assert "cwd=log_root.parent" in phase_source


def test_prepare_retries_a_dropped_ssh_connection(monkeypatch, tmp_path):
    calls = []

    def fake_prepare(run_id):
        calls.append(run_id)
        if len(calls) < 3:
            raise SSHException("fixture drop")
        return {"status": "deployed_not_started"}

    monkeypatch.setattr(manager, "_prepare_once", fake_prepare)
    monkeypatch.setattr(manager, "local_paths", lambda _run_id: {"deploy": tmp_path / "deploy.json"})
    monkeypatch.setattr(manager.time, "sleep", lambda _seconds: None)
    payload = manager.prepare("siim_retry", attempts=3)
    assert payload["connection_attempt"] == 3
    assert len(calls) == 3


def test_rescinded_evolution_reservation_blocks_prepare_before_spec_load(
    monkeypatch, tmp_path
):
    child_id = "siim_child_r2"
    rescission = (
        tmp_path
        / "workspace"
        / "siim_evolution_control"
        / child_id
        / "constraint_rescission.json"
    )
    rescission.parent.mkdir(parents=True)
    rescission.write_text(
        json.dumps(
            {
                "schema": "evomind.siim.constraint_rescission.v1",
                "status": "effective",
                "child_run_id": child_id,
                "active_policy": {"new_candidate_run_allowed": False},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(manager, "PROJECT_ROOT", tmp_path)

    with pytest.raises(manager.CampaignManagerError, match="was rescinded"):
        manager._load_evolution_spec(tmp_path / "missing-spec.json", child_id)


def test_publish_status_keeps_a_hold_visible_without_claiming_training(monkeypatch, tmp_path):
    run_id = "siim_status_test"
    monkeypatch.setattr(manager, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(manager, "LOCAL_ROOT", tmp_path / "hpc")
    run_dir = tmp_path / "workspace" / "evomind_runs" / run_id
    run_dir.mkdir(parents=True)
    local = manager.local_paths(run_id)
    local["gate"].parent.mkdir(parents=True)
    local["gate"].write_text(json.dumps({
        "schema": manager.remote_ops.EXPECTED_GATE_SCHEMA,
        "credential_profile": manager.CREDENTIAL_PROFILE,
        "remote_root": manager.ALLOWED_GPU_REMOTE_ROOT,
        "passed": False,
        "hold_reasons": ["blocked_process_state"],
        "identity": {"gpu_uuids": ["fixture"]},
        "samples": [{
            "captured_at": "2026-07-29T00:00:00+00:00",
            "gpus": [{
                "name": "NVIDIA A800-SXM4-80GB",
                "memory_total_mib": 81920,
                "memory_free_mib": 81917,
                "memory_used_mib": 3,
                "utilization_percent": 0,
            }],
            "compute_apps": [],
            "eligible": False,
            "hold_reasons": ["blocked_process_state"],
        }] * 5,
    }), encoding="utf-8")
    payload = manager.publish_status(run_id)
    runtime = json.loads((run_dir / "hpc_runtime.json").read_text(encoding="utf-8"))
    assert payload["launch_decision"] == "HOLD"
    assert runtime["status"] == "hold"
    assert runtime["signals_sent"] == 0
    assert not (run_dir / "metrics.json").exists()


def test_continuation_record_must_stay_inside_campaign_directory(monkeypatch, tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    defaults = {"root": root, "gate": root / "gpu_gate.json"}
    monkeypatch.setattr(manager, "local_paths", lambda _run_id: defaults)

    selected = manager._local_campaign_record(
        "siim_continuation",
        root / "continuation_gpu_gate_001.json",
        default=defaults["gate"],
        label="gate",
    )
    assert selected == (root / "continuation_gpu_gate_001.json").resolve()
    with pytest.raises(manager.CampaignManagerError, match="direct child"):
        manager._local_campaign_record(
            "siim_continuation",
            tmp_path / "outside.json",
            default=defaults["gate"],
            label="gate",
        )


def test_continuation_gate_does_not_overwrite_initial_or_global_gate(monkeypatch, tmp_path):
    run_id = "siim_continuation"
    monkeypatch.setattr(manager, "LOCAL_ROOT", tmp_path / "campaigns")
    local = manager.local_paths(run_id)
    local["root"].mkdir(parents=True)
    local["gate"].write_text("initial\n", encoding="utf-8")
    global_gate = tmp_path / "gpu_gate_current.json"
    global_gate.write_text("global\n", encoding="utf-8")
    monkeypatch.setattr(manager.remote_ops, "DEFAULT_GATE_REPORT", global_gate)
    payload = {
        "schema": manager.remote_ops.EXPECTED_GATE_SCHEMA,
        "credential_profile": manager.CREDENTIAL_PROFILE,
        "remote_root": manager.ALLOWED_GPU_REMOTE_ROOT,
        "passed": False,
        "hold_reasons": ["fixture_hold"],
        "signals_sent": 0,
        "other_processes_modified": False,
    }
    monkeypatch.setattr(manager.remote_ops, "sample_gpu_idle_gate", lambda **_kwargs: payload)
    class FakeClient:
        @staticmethod
        def close():
            return None

    monkeypatch.setattr(manager, "connect", lambda: (FakeClient(), object()))
    continuation = local["root"] / "continuation_gpu_gate_001.json"

    result = manager.gate(
        run_id,
        interval_seconds=0,
        output_path=continuation,
        publish_default=False,
    )

    assert result == payload
    assert json.loads(continuation.read_text(encoding="utf-8")) == payload
    assert local["gate"].read_text(encoding="utf-8") == "initial\n"
    assert global_gate.read_text(encoding="utf-8") == "global\n"
