from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "deploy_hpc88240_cactus_continuation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("deploy_cactus_hpc", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_data() -> dict:
    return {
        "file_count": 4,
        "total_bytes": 100,
        "core": {
            "train.csv": {"bytes": 1, "sha256": "a" * 64},
            "sample_submission.csv": {"bytes": 1, "sha256": "b" * 64},
            "train.zip": {"bytes": 1, "sha256": "c" * 64},
            "test.zip": {"bytes": 1, "sha256": "d" * 64},
        },
    }


def test_plan_is_highres_multiseed_and_candidate_only() -> None:
    module = load_module()
    plan = module.build_plan(module.source_records(), fake_data())
    assert plan["competition_id"] == "aerial-cactus-identification"
    assert plan["training"]["backbone"] == "convnext_small"
    assert plan["training"]["image_size"] == 384
    assert plan["training"]["folds"] == 5
    assert plan["training"]["seeds"] == [40, 41, 42]
    assert plan["stopping_policy"]["per_seed_minimum_auc"] == 0.9995
    assert plan["training"]["candidate_only"] is True
    assert plan["planner"]["execution_plan"]["remote_path"] == module.REMOTE_OPTIMIZATION_PLAN
    assert plan["data_contract"]["visibility_mode"] == "PUBLIC_ONLY"
    assert plan["data_contract"]["public_only_environment"] == "EVOMIND_MLEBENCH_PUBLIC_ONLY=1"
    assert plan["serial_dependency"]["blocking_terminal_statuses"] == sorted(
        module.RANZCR_FAILURE_TERMINAL_STATUSES
    )
    assert plan["launch_contract"]["v1_supersession"]["mode"] == "observe_only_natural_retirement"
    assert plan["launch_contract"]["process_signals_allowed"] is False


def test_wrapper_waits_for_ranzcr_and_has_no_submission_or_signal_path() -> None:
    module = load_module()
    wrapper = module.render_wrapper("c" * 64, module.source_records(), fake_data())
    assert "waiting_for_ranzcr" in wrapper
    assert "waiting_for_v1_retirement" in wrapper
    assert "v1_candidate_preserved" in wrapper
    for status in module.RANZCR_READY_TERMINAL_STATUSES:
        assert status in wrapper
    for status in module.RANZCR_FAILURE_TERMINAL_STATUSES:
        assert status in wrapper
    assert "--waves Wave0 --competitions aerial-cactus-identification" in wrapper
    assert "--aerial-backbone convnext_small" in wrapper
    assert "--aerial-image-size 384" in wrapper
    assert "--aerial-folds 5" in wrapper
    assert "--optimization-plan \"$OPTIMIZATION_PLAN\"" in wrapper
    assert "EVOMIND_MLEBENCH_PUBLIC_ONLY=1" in wrapper
    assert "wait_for_gpu_idle \"$completed\"" in wrapper
    assert "[ \"$rc\" -ne 0 ] && [ \"$rc\" -ne 3 ]" in wrapper
    assert "single-seed public OOF stop loss" in wrapper
    assert "cactus_multiseed_oof_and_test.npz" in wrapper
    assert "--candidate-only" in wrapper
    assert "candidate result status contract failed" in wrapper
    assert "candidate_only was not preserved" in wrapper
    assert "candidate submission validation failed" in wrapper
    assert "official grader execution detected" in wrapper
    assert "official grader withholding contract failed" in wrapper
    assert "human confirmation contract failed" in wrapper
    assert "candidate summary terminal contract failed" in wrapper
    assert "dtype=np.float64" in wrapper
    assert "probability_dtype':'float64" in wrapper
    assert "evomind.hpc_cactus_persistent_run.v2" in wrapper
    lowered = wrapper.lower()
    assert "pkill" not in lowered
    assert "killall" not in lowered
    assert "taskkill" not in lowered
    assert "kaggle competitions submit" not in lowered
    assert "official_grader_executed':False" in wrapper
    assert "process_signals_sent':0" in wrapper


def test_no_start_argument_is_exposed() -> None:
    module = load_module()
    assert module.build_parser().parse_args(["--no-start"]).no_start is True


def test_evidence_status_is_derived_from_observed_remote_state() -> None:
    module = load_module()
    assert module.deployment_evidence_status(
        start=False, action="new_waiting_wrapper_started", wrapper_status={"status": "waiting"}
    ) == "deployed_not_started"
    assert module.deployment_evidence_status(
        start=True,
        action="new_waiting_wrapper_started",
        wrapper_status={"status": "waiting_for_ranzcr"},
    ) == "started_waiting_for_ranzcr"
    assert module.deployment_evidence_status(
        start=True,
        action="existing_wrapper_reused",
        wrapper_status={"status": "training_seed_41"},
    ) == "reused_training_seed_41"


def test_existing_v2_is_preserved_before_any_redeployment() -> None:
    module = load_module()
    waiting = {"status": "waiting_for_ranzcr", "wrapper_pid": 123}
    assert (
        module.preserve_existing_v2_action(waiting, active=True)
        == "existing_wrapper_reused"
    )
    assert (
        module.preserve_existing_v2_action(waiting, active=False)
        == "terminal_or_active_status_preserved"
    )
    assert (
        module.preserve_existing_v2_action(
            {"status": "frozen_artifact_drift", "wrapper_pid": 123},
            active=False,
        )
        is None
    )


def test_upload_bytes_preflight_reuses_matching_remote_content(monkeypatch) -> None:
    module = load_module()
    payload = b"candidate-only"
    calls: list[tuple] = []
    monkeypatch.setattr(module, "remote_sha256", lambda _client, _remote: module.hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(module.common, "upload_bytes", lambda *args: calls.append(args))
    assert module.upload_bytes_if_changed(object(), payload, "/remote/candidate") == "reused"
    assert calls == []


def test_upload_file_preflight_reuses_matching_remote_content(tmp_path, monkeypatch) -> None:
    module = load_module()
    source = tmp_path / "frozen-plan.json"
    source.write_bytes(b"frozen")
    calls: list[tuple] = []
    digest = module.common.sha256_file(source)
    monkeypatch.setattr(module, "remote_sha256", lambda _client, _remote: digest)
    monkeypatch.setattr(module.common, "upload_file", lambda *args: calls.append(args))
    assert module.upload_file_if_changed(object(), source, "/remote/frozen-plan.json") == "reused"
    assert calls == []
