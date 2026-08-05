"""Fast contract tests for the recoverable MLE-Bench Lite full runner."""
from __future__ import annotations

import inspect
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import run_mlebench_lite_full as full
from scripts import run_mlebench_lite_wave0 as wave0
from scripts.generate_mlebench_wave_plan import (
    competitions_from_progress_report,
    extract_json,
    validate_order,
)


def test_wave_selection_and_explicit_scope():
    assert full.parse_waves("Wave1,Wave1") == ("Wave1",)
    assert full.select_competitions(("Wave1",)) == list(full.WAVE1_COMPETITIONS)
    selected = full.select_competitions(("Wave1",), "leaf-classification,random-acts-of-pizza")
    assert selected == ["leaf-classification", "random-acts-of-pizza"]
    with pytest.raises(ValueError, match="outside selected waves"):
        full.select_competitions(("Wave1",), "spooky-author-identification")
    with pytest.raises(ValueError, match="Unsupported waves"):
        full.parse_waves("Wave9")
    assert full.parse_waves("Wave2") == ("Wave2",)
    assert full.select_competitions(("Wave2",)) == list(full.WAVE2_COMPETITIONS)
    assert len(full.WAVE2_COMPETITIONS) == 11


def test_siim_a800_batch_contract_defaults_to_eight_workers_and_effective_384(tmp_path):
    args = full.parse_args(["--official-source-root", str(tmp_path)])
    assert args.siim_workers == 8
    assert args.siim_effective_batch_size == 384
    assert args.siim_batch_size == 64
    assert args.siim_memory_limit_mib == 0
    assert args.siim_runtime_budget_seconds == 0.0
    assert args.siim_control_file is None
    assert args.siim_image_content_manifest is None


def test_siim_cuda_memory_limit_is_applied_before_training():
    class FakeProperties:
        total_memory = 80 * 1024 * 1024 * 1024

    class FakeCuda:
        def __init__(self):
            self.calls = []

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_properties(_device):
            return FakeProperties()

        def set_per_process_memory_fraction(self, fraction, *, device):
            self.calls.append((fraction, device))

    cuda = FakeCuda()
    contract = full.recovery.configure_siim_cuda_memory_limit(
        SimpleNamespace(cuda=cuda),
        55 * 1024,
    )
    assert contract["memory_limit_mib"] == 55 * 1024
    assert contract["memory_total_mib"] == 80 * 1024
    assert contract["memory_fraction"] == pytest.approx(0.6875)
    assert cuda.calls == [(pytest.approx(0.6875), 0)]


def test_siim_cuda_memory_limit_rejects_invalid_capacity():
    cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_properties=lambda _device: SimpleNamespace(
            total_memory=40 * 1024 * 1024 * 1024
        ),
        set_per_process_memory_fraction=lambda *_args, **_kwargs: None,
    )
    with pytest.raises(RuntimeError, match="memory limit is invalid"):
        full.recovery.configure_siim_cuda_memory_limit(
            SimpleNamespace(cuda=cuda),
            55 * 1024,
        )


def test_siim_epoch_pause_reason_is_fail_closed_and_budget_aware(tmp_path, monkeypatch):
    budget = tmp_path / "runtime_budget.json"
    budget.write_text(
        json.dumps({"started_unix": 100.0, "runtime_budget_seconds": 20.0}),
        encoding="utf-8",
    )
    monkeypatch.setattr(full.recovery.time, "time", lambda: 121.0)
    assert full.recovery.siim_epoch_pause_reason(
        control_file=None,
        budget_state_path=budget,
        runtime_budget_seconds=20.0,
    ) == "runtime_budget_exhausted"

    control = tmp_path / "control.json"
    control.write_text(
        json.dumps({"status": "pause_after_epoch", "reason": "other_gpu_growth"}),
        encoding="utf-8",
    )
    assert full.recovery.siim_epoch_pause_reason(
        control_file=control,
        budget_state_path=budget,
        runtime_budget_seconds=0.0,
    ) == "resource_guard:other_gpu_growth"

    control.write_text("{broken", encoding="utf-8")
    assert full.recovery.siim_epoch_pause_reason(
        control_file=control,
        budget_state_path=budget,
        runtime_budget_seconds=0.0,
    ).startswith("resource_control_invalid:")


def test_requested_phase_a_scope_filters_without_hiding_requested_failures():
    report = {
        "data_root": "DATA",
        "rows": [
            {"competition_id": "dog-breed-identification", "status": "passed"},
            {"competition_id": "ranzcr-clip-catheter-line-classification", "status": "failed"},
            {"competition_id": "siim-isic-melanoma-classification", "status": "failed"},
        ],
    }
    passed = full.scope_phase_a_audit(report, ["dog-breed-identification"])
    assert passed["scope"] == "requested"
    assert passed["competition_count"] == 1
    assert passed["passed"] == 1
    assert passed["status"] == "passed"

    failed = full.scope_phase_a_audit(
        report,
        ["dog-breed-identification", "ranzcr-clip-catheter-line-classification"],
    )
    assert failed["competition_count"] == 2
    assert failed["failed"] == 1
    assert failed["status"] == "failed"

    with pytest.raises(RuntimeError, match="omitted requested competitions"):
        full.scope_phase_a_audit(report, ["missing-competition"])


def test_gpt56_plan_is_validated_and_controls_order(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({
        "schema": "evomind.mlebench_lite.optimization_plan.v1",
        "planner": {"provider": "openai", "model": "gpt-5.6-sol"},
        "competition_order": ["random-acts-of-pizza", "leaf-classification"],
    }), encoding="utf-8")
    plan = full.load_optimization_plan(path, tmp_path)
    ordered = full.apply_plan_order(
        ["leaf-classification", "random-acts-of-pizza", "denoising-dirty-documents"], plan
    )
    assert ordered == [
        "random-acts-of-pizza", "leaf-classification", "denoising-dirty-documents"
    ]
    assert plan["_sha256"]


def test_gpt56_medal_recovery_plan_is_validated_and_controls_priority_order(tmp_path):
    path = tmp_path / "medal-plan.json"
    path.write_text(json.dumps({
        "schema": "evomind.mlebench_lite.medal_recovery_plan.v1",
        "planner": {"provider": "openai", "model": "gpt-5.6-sol"},
        "priority_order": ["dogs-vs-cats-redux-kernels-edition", "leaf-classification"],
    }), encoding="utf-8")
    plan = full.load_optimization_plan(path, tmp_path)
    ordered = full.apply_plan_order(
        ["leaf-classification", "dogs-vs-cats-redux-kernels-edition"], plan
    )
    assert ordered == ["dogs-vs-cats-redux-kernels-edition", "leaf-classification"]
    assert plan["_sha256"]


def test_plan_order_allows_full_plan_for_single_selected_competition():
    plan = {
        "competition_order": [
            "plant-pathology-2020-fgvc7",
            "mlsp-2013-birds",
            "jigsaw-toxic-comment-classification-challenge",
        ]
    }
    ordered = full.apply_plan_order(["mlsp-2013-birds"], plan)
    assert ordered == ["mlsp-2013-birds"]


def test_plan_fails_closed_for_wrong_model(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({
        "schema": "evomind.mlebench_lite.optimization_plan.v1",
        "planner": {"provider": "openai", "model": "other"},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="gpt-5.6-sol"):
        full.load_optimization_plan(path, tmp_path)


def test_attempt_directories_are_monotonic(tmp_path):
    assert full.next_attempt_dir(tmp_path).name == "attempt_001"
    assert full.next_attempt_dir(tmp_path).name == "attempt_002"


def test_promotion_gate_withholds_private_grader_and_persists_submission(tmp_path, monkeypatch):
    spec = SimpleNamespace(metric="log_loss", direction="minimize")
    monkeypatch.setattr(wave0, "get_competition_spec", lambda _competition_id: spec)
    monkeypatch.setattr(
        wave0,
        "validate_submission_file",
        lambda *_args, **_kwargs: {"valid": True},
    )

    def fail_if_graded(*_args, **_kwargs):
        raise AssertionError("private grader must be withheld when the gate fails")

    monkeypatch.setattr(wave0, "private_grade", fail_if_graded)
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    result = wave0.finalize_scored_task(
        competition_id="example",
        submission=pd.DataFrame({"id": [1], "target": [0.5]}),
        cv_score=0.2,
        args=SimpleNamespace(data_root=tmp_path),
        task_dir=task_dir,
        budget={},
        extra={},
        promotion_gate={"passed": False, "name": "example_gate"},
    )
    assert result["status"] == "promotion_gate_failed"
    assert result["official_grader_executed"] is False
    assert result["official_grader_withheld"] is True
    assert (task_dir / "promotion_gate.json").is_file()
    assert (task_dir / "submission.csv").is_file()

    evidence = full.build_evidence_contract(
        SimpleNamespace(seed=42, holdout_fraction=0.2), result
    )
    assert evidence["official_private_grader_required"] is False
    assert evidence["official_private_grader_withheld"] is True
    assert evidence["promotion_gate_evaluated"] is True


def test_candidate_only_withholds_private_grader_after_gate_pass(tmp_path, monkeypatch):
    spec = SimpleNamespace(metric="roc_auc", direction="maximize")
    monkeypatch.setattr(wave0, "get_competition_spec", lambda _competition_id: spec)
    monkeypatch.setattr(
        wave0,
        "validate_submission_file",
        lambda *_args, **_kwargs: {"valid": True},
    )

    def fail_if_graded(*_args, **_kwargs):
        raise AssertionError("candidate-only mode must never invoke the private grader")

    monkeypatch.setattr(wave0, "private_grade", fail_if_graded)
    task_dir = tmp_path / "candidate"
    task_dir.mkdir()
    result = wave0.finalize_scored_task(
        competition_id="example",
        submission=pd.DataFrame({"id": [1], "target": [0.9]}),
        cv_score=0.95,
        args=SimpleNamespace(data_root=tmp_path, candidate_only=True),
        task_dir=task_dir,
        budget={},
        extra={},
        promotion_gate={"passed": True, "name": "example_gate"},
    )

    assert result["status"] == "promotion_gate_passed_confirmation_pending"
    assert result["official_grader_executed"] is False
    assert result["official_grader_withheld"] is True
    assert result["official_grader_confirmation_pending"] is True
    assert result["candidate_only"] is True
    assert result["private_grader"]["reason"] == (
        "explicit_candidate_only_human_confirmation_required"
    )
    assert (task_dir / "submission.csv").is_file()

    evidence = full.build_evidence_contract(
        SimpleNamespace(seed=42, holdout_fraction=0.2, candidate_only=True), result
    )
    assert evidence["official_private_grader_required"] is False
    assert evidence["official_private_grader_confirmation_pending"] is True
    assert evidence["candidate_only"] is True


def test_default_gate_pass_keeps_private_grader_behavior(tmp_path, monkeypatch):
    spec = SimpleNamespace(metric="roc_auc", direction="maximize")
    monkeypatch.setattr(wave0, "get_competition_spec", lambda _competition_id: spec)
    monkeypatch.setattr(
        wave0,
        "validate_submission_file",
        lambda *_args, **_kwargs: {"valid": True},
    )
    calls = []

    def grade(*_args, **_kwargs):
        calls.append(True)
        return {
            "status": "passed",
            "score": 0.95,
            "official_mlebench_grader_executed": True,
        }

    monkeypatch.setattr(wave0, "private_grade", grade)
    task_dir = tmp_path / "graded"
    task_dir.mkdir()
    result = wave0.finalize_scored_task(
        competition_id="example",
        submission=pd.DataFrame({"id": [1], "target": [0.9]}),
        cv_score=0.95,
        args=SimpleNamespace(data_root=tmp_path),
        task_dir=task_dir,
        budget={},
        extra={},
        promotion_gate={"passed": True, "name": "example_gate"},
    )

    assert calls == [True]
    assert result["status"] == "passed"
    assert result["official_grader_executed"] is True
    assert result["official_grader_withheld"] is False
    assert result["official_grader_confirmation_pending"] is False


def test_candidate_only_cli_aliases_are_equivalent():
    candidate = full.parse_args(["--official-source-root", ".", "--candidate-only"])
    withheld = full.parse_args([
        "--official-source-root",
        ".",
        "--withhold-official-grader",
    ])
    default = full.parse_args(["--official-source-root", "."])

    assert candidate.candidate_only is True
    assert withheld.candidate_only is True
    assert default.candidate_only is False
    assert default.hold_cuda_lease is False


def test_local_cuda_lease_cli_is_explicit():
    args = full.parse_args([
        "--official-source-root",
        ".",
        "--candidate-only",
        "--hold-cuda-lease",
    ])
    assert args.candidate_only is True
    assert args.hold_cuda_lease is True


def test_candidate_only_terminal_status_completes_full_run_successfully(tmp_path, monkeypatch):
    monkeypatch.setattr(full, "compute_policy", None)
    competition_id = "siim-isic-melanoma-classification"
    data_root = tmp_path / "data"
    output_root = tmp_path / "runs"
    official_root = tmp_path / "mle-bench"
    for path in (data_root, output_root, official_root):
        path.mkdir()

    monkeypatch.setattr(
        full,
        "audit_lite22_specs",
        lambda _root: {
            "schema": "test",
            "data_root": str(data_root),
            "competition_count": 1,
            "passed": 1,
            "failed": 0,
            "status": "passed",
            "rows": [{"competition_id": competition_id, "status": "passed"}],
        },
    )
    monkeypatch.setattr(
        full,
        "export_specs",
        lambda path: path.write_text("{}", encoding="utf-8"),
    )
    monkeypatch.setattr(full.wave0, "environment_report", lambda: {"cuda_available": True})

    class Telemetry:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def summary(self):
            return {"sample_count": 0}

    monkeypatch.setattr(full.wave0, "GpuTelemetry", Telemetry)
    monkeypatch.setitem(
        full.RUNNERS,
        competition_id,
        lambda *_args: {
            "competition_id": competition_id,
            "status": "promotion_gate_passed_confirmation_pending",
            "official_grader_executed": False,
            "official_grader_withheld": True,
            "official_grader_confirmation_pending": True,
            "candidate_only": True,
            "promotion_gate": {"passed": True},
            "valid_submission": True,
        },
    )

    return_code = full.main([
        "--data-root",
        str(data_root),
        "--output-root",
        str(output_root),
        "--allowed-root",
        str(tmp_path),
        "--official-source-root",
        str(official_root),
        "--waves",
        "Wave0",
        "--competitions",
        competition_id,
        "--run-id",
        "candidate_only_test",
        "--phase-a-scope",
        "requested",
        "--candidate-only",
    ])

    summary = json.loads(
        (output_root / "candidate_only_test" / "summary.json").read_text(encoding="utf-8")
    )
    assert return_code == 0
    assert summary["status"] == "candidate_complete"
    assert summary["passed"] == 0
    assert summary["candidate_confirmation_pending"] == 1
    assert summary["failed"] == 0


def test_taxi_temporal_stress_split_is_chronological():
    timestamps = pd.Series(pd.date_range("2020-01-01", periods=20, freq="D"))
    fit, valid, evidence = full.build_taxi_temporal_stress_split(
        timestamps, holdout_fraction=0.2
    )
    assert set(fit).isdisjoint(set(valid))
    assert timestamps.iloc[fit].max() < timestamps.iloc[valid].min()
    assert evidence["validation_rows"] == len(valid)


def test_cache_environment_keeps_torch_shared_and_other_caches_run_local(
    tmp_path, monkeypatch
):
    allowed_root = tmp_path / "gpu_tra"
    run_dir = allowed_root / "mlebench_lite_runs" / "run_001"
    run_dir.mkdir(parents=True)
    for name in (*full.RUN_LOCAL_CACHE_ENVIRONMENTS, "TORCH_HOME"):
        monkeypatch.delenv(name, raising=False)

    result = full.configure_cache_environment(run_dir, allowed_root)

    expected_torch = (allowed_root / full.SHARED_TORCH_CACHE_RELATIVE).resolve()
    assert Path(result["shared_torch_home"]) == expected_torch
    assert Path(result["environment"]["TORCH_HOME"]) == expected_torch
    assert expected_torch.is_dir()
    short_temp_root = Path(result["short_temp_root"])
    assert short_temp_root.parent == (allowed_root / full.SHORT_TEMP_RELATIVE).resolve()
    assert len(short_temp_root.name) == 12
    assert short_temp_root.is_dir()
    for name in full.RUN_LOCAL_CACHE_ENVIRONMENTS:
        path = Path(result["environment"][name])
        if name in full.RUN_SHORT_TEMP_ENVIRONMENTS:
            assert path == short_temp_root
        else:
            assert path.is_relative_to((run_dir / "cache").resolve())
        assert path.is_dir()
    assert not Path(result["environment"]["TORCH_HOME"]).is_relative_to(run_dir)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (FileNotFoundError("x"), "data_missing"),
        (RuntimeError("CUDA out of memory"), "resource_oom"),
        (ModuleNotFoundError("x"), "dependency_missing"),
        (ValueError("submission column mismatch"), "submission_or_schema"),
        (TimeoutError("x"), "infrastructure_transient"),
        (RuntimeError("bad fit"), "training_runtime"),
    ],
)
def test_failure_taxonomy(exc, expected):
    assert full.classify_failure(exc) == expected


def test_taxi_features_are_finite_and_include_geo_time():
    frame = pd.DataFrame({
        "pickup_datetime": ["2010-10-01 21:26:11 UTC"],
        "pickup_longitude": [-73.98313],
        "pickup_latitude": [40.76197],
        "dropoff_longitude": [-73.994386],
        "dropoff_latitude": [40.749236],
        "passenger_count": [1],
    })
    features = full.taxi_features(frame)
    assert features.loc[0, "haversine_km"] > 0
    assert features.loc[0, "hour"] == 21
    assert "airport_trip" in features
    assert "bearing_sin" in features
    assert "pickup_jfk_km" in features
    assert "airport_rush_hour" in features
    assert "pickup_borough_proxy" in features
    assert "cross_borough_proxy" in features
    assert "pickup_coordinate_missing" in features
    assert "airport_route_code" in features
    assert len(features.columns) >= 59
    assert np.isfinite(features.to_numpy()).all()

    stationary_jfk = frame.copy()
    stationary_jfk[["pickup_latitude", "dropoff_latitude"]] = 40.6413
    stationary_jfk[["pickup_longitude", "dropoff_longitude"]] = -73.7781
    jfk_features = full.taxi_features(stationary_jfk)
    assert jfk_features.loc[0, "haversine_km"] == pytest.approx(0.0, abs=1e-10)
    assert jfk_features.loc[0, "pickup_jfk_within_3km"] == 1.0
    assert jfk_features.loc[0, "dropoff_jfk_within_3km"] == 1.0
    assert jfk_features.loc[0, "jfk_route"] == 1.0


def test_taxi_loader_reads_in_chunks_and_filters_targets(tmp_path):
    path = tmp_path / "labels.csv"
    pd.DataFrame({
        "key": [f"k{i}" for i in range(8)],
        "fare_amount": [-1, 4, 5, 6, 7, 8, 300, 10],
        "pickup_datetime": ["2010-01-01 00:00:00 UTC"] * 8,
        "pickup_longitude": [-73.9] * 8,
        "pickup_latitude": [40.7] * 8,
        "dropoff_longitude": [-73.8] * 8,
        "dropoff_latitude": [40.8] * 8,
        "passenger_count": [1] * 8,
    }).to_csv(path, index=False)
    frame, scanned = full.load_taxi_training(path, max_rows=4, chunk_rows=3, seed=42)
    assert len(frame) == 4
    assert frame["fare_amount"].between(1, 200).all()
    assert frame["__source_row__"].between(0, 7).all()
    assert frame["__source_row__"].is_unique
    assert scanned == 8
    audit = frame.attrs["cleaning_audit"]
    assert audit["rows_scanned"] == scanned
    assert audit["reservoir_rows_retained"] == len(frame)
    assert audit["rejected_fare"] == 2
    assert audit["rules_are_nonexclusive"] is True


def test_taxi_hash_oof_splits_are_deterministic_disjoint_and_complete():
    source_rows = np.arange(10_000, 10_103, dtype=np.int64)
    first_splits, first_assignment = full.build_taxi_oof_splits(
        source_rows,
        fold_count=3,
        seed=42,
    )
    second_splits, second_assignment = full.build_taxi_oof_splits(
        source_rows,
        fold_count=3,
        seed=42,
    )
    np.testing.assert_array_equal(first_assignment, second_assignment)
    assert len(first_splits) == len(second_splits) == 3
    validation_rows = []
    for fit_indices, valid_indices in first_splits:
        assert set(fit_indices).isdisjoint(set(valid_indices))
        validation_rows.extend(valid_indices.tolist())
    assert sorted(validation_rows) == list(range(len(source_rows)))
    assert set(first_assignment.tolist()) == {0, 1, 2}

    duplicate_splits, duplicate_assignment = full.build_taxi_oof_splits(
        np.asarray([1, 1, 2]),
        fold_count=3,
        seed=42,
    )
    assert len(duplicate_splits) == 2
    assert duplicate_assignment[0] == duplicate_assignment[1]


def test_taxi_duplicate_groups_quarantine_conflicts_and_stay_in_one_fold():
    frame = pd.DataFrame(
        {
            "key": ["same", "same", "ok-a", "ok-b", "ok-c"],
            "fare_amount": [10.0, 11.0, 8.0, 8.0, 12.0],
            "pickup_datetime": [
                "2010-01-01 00:00:00 UTC",
                "2010-01-01 00:00:00 UTC",
                "2010-01-02 00:00:00 UTC",
                "2010-01-02 00:00:00 UTC",
                "2010-01-03 00:00:00 UTC",
            ],
            "pickup_longitude": [-73.9, -73.9, -73.8, -73.8, -73.7],
            "pickup_latitude": [40.7, 40.7, 40.8, 40.8, 40.9],
            "dropoff_longitude": [-73.8, -73.8, -73.7, -73.7, -73.6],
            "dropoff_latitude": [40.8, 40.8, 40.9, 40.9, 41.0],
            "passenger_count": [1, 1, 2, 2, 3],
        }
    )

    safe, groups, audit = full.build_taxi_duplicate_groups(frame)

    assert safe["key"].tolist() == ["ok-a", "ok-b", "ok-c"]
    assert audit["quarantined_rows"] == 2
    assert audit["conflicting_trip_signatures"] == 1
    assert groups[0] == groups[1]
    splits, assignment = full.build_taxi_oof_splits(groups, fold_count=2, seed=42)
    assert assignment[0] == assignment[1]
    for fit_index, valid_index in splits:
        assert not set(groups[fit_index]) & set(groups[valid_index])


def test_taxi_fold_local_route_statistics_never_reads_validation_targets():
    fit = pd.DataFrame(
        {
            "pickup_cell_id": [1, 1, 2],
            "dropoff_cell_id": [3, 3, 4],
            "hour": [8, 8, 9],
            "airport_route_code": [1, 1, 0],
            "passenger_count": [1, 2, 1],
            "pickup_borough_proxy": [0, 0, 1],
            "dropoff_borough_proxy": [1, 1, 2],
        }
    )
    validation = fit.iloc[[0]].copy()
    fit_encoded, validation_encoded = full.build_taxi_fold_local_route_statistics(
        fit, pd.Series([10.0, 14.0, 20.0]), validation, smoothing=1.0
    )
    _, changed_validation = full.build_taxi_fold_local_route_statistics(
        fit, pd.Series([10.0, 14.0, 20.0]), validation, smoothing=1.0
    )

    columns = [column for column in fit_encoded if column.startswith("route_stat_")]
    assert len(columns) == 4
    assert validation_encoded[columns].equals(changed_validation[columns])
    assert np.isfinite(validation_encoded[columns].to_numpy()).all()


def test_taxi_geographic_stress_split_keeps_pickup_cells_disjoint():
    cells = np.repeat(np.arange(10), 3)
    fit, valid, evidence = full.build_taxi_geographic_stress_split(
        cells, holdout_fraction=0.2, seed=42
    )

    assert not set(cells[fit]) & set(cells[valid])
    assert evidence["fit_rows"] + evidence["validation_rows"] == len(cells)
    assert evidence["validation_pickup_cells"] == 2


def test_taxi_runner_persists_full_refit_and_all_stress_contracts():
    source = Path(full.__file__).read_text(encoding="utf-8")
    assert '"full_data_refit": True' in source
    assert "median_outer_fold_selected_iterations" in source
    assert "taxi_full_data_refit.cbm" in source
    assert "geographic_stress_rmse" in source
    assert "taxi_duplicate_audit.json" in source


def test_taxi_public_cache_round_trip_and_tamper_detection(tmp_path, monkeypatch):
    public = (
        tmp_path
        / "data"
        / "new-york-city-taxi-fare-prediction"
        / "prepared"
        / "public"
    )
    public.mkdir(parents=True)
    rows = 30
    labels = pd.DataFrame(
        {
            "key": [f"train-{index}" for index in range(rows)],
            "fare_amount": [8.0 + index / 10 for index in range(rows)],
            "pickup_datetime": [
                f"2010-01-{1 + index % 20:02d} {index % 24:02d}:00:00 UTC"
                for index in range(rows)
            ],
            "pickup_longitude": [-74.0 + (index % 10) * 0.01 for index in range(rows)],
            "pickup_latitude": [40.6 + (index % 10) * 0.01 for index in range(rows)],
            "dropoff_longitude": [-73.9 + (index % 10) * 0.01 for index in range(rows)],
            "dropoff_latitude": [40.7 + (index % 10) * 0.01 for index in range(rows)],
            "passenger_count": [1 + index % 4 for index in range(rows)],
        }
    )
    test = labels.drop(columns="fare_amount").iloc[:5].copy()
    test["key"] = [f"test-{index}" for index in range(len(test))]
    labels.to_csv(public / "labels.csv", index=False)
    test.to_csv(public / "test.csv", index=False)
    pd.DataFrame({"key": test["key"], "fare_amount": 0.0}).to_csv(
        public / "sample_submission.csv", index=False
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    cache = tmp_path / "cache"

    manifest = full.build_taxi_public_feature_cache(
        data_root=tmp_path / "data",
        output_dir=cache,
        allowed_root=tmp_path,
        max_rows=30,
        chunk_rows=7,
        seed=42,
        folds=3,
        holdout_fraction=0.2,
    )
    loaded = full.load_taxi_public_feature_cache(
        cache,
        data_root=tmp_path / "data",
        allowed_root=tmp_path,
        max_rows=30,
        chunk_rows=7,
        seed=42,
        folds=3,
        holdout_fraction=0.2,
    )

    assert manifest["status"] == "completed"
    assert manifest["contracts"]["gpu_used"] is False
    assert loaded["train_features"].shape[0] == rows
    assert loaded["test_features"].shape[0] == len(test)
    assert set(loaded["fold_assignment"].tolist()) == {0, 1, 2}
    with (cache / "target.npy").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(RuntimeError, match="artifact size drifted"):
        full.load_taxi_public_feature_cache(
            cache,
            data_root=tmp_path / "data",
            allowed_root=tmp_path,
            max_rows=30,
            chunk_rows=7,
            seed=42,
            folds=3,
            holdout_fraction=0.2,
        )


def test_scalar_submission_alignment_is_id_keyed_and_finite():
    sample = pd.DataFrame({"key": ["b", "a"], "fare_amount": [0.0, 0.0]})
    aligned = full.align_scalar_submission(
        sample,
        pd.Series(["a", "b"]),
        np.asarray([11.0, 22.0]),
        id_column="key",
        target_column="fare_amount",
    )
    assert aligned["key"].tolist() == ["b", "a"]
    assert aligned["fare_amount"].tolist() == [22.0, 11.0]
    with pytest.raises(RuntimeError, match="ID sets differ"):
        full.align_scalar_submission(
            sample,
            pd.Series(["a", "c"]),
            np.asarray([11.0, 22.0]),
            id_column="key",
            target_column="fare_amount",
        )
    with pytest.raises(RuntimeError, match="finiteness"):
        full.align_scalar_submission(
            sample,
            pd.Series(["a", "b"]),
            np.asarray([np.nan, 22.0]),
            id_column="key",
            target_column="fare_amount",
        )


def test_recovery_compute_defaults_and_runner_contracts():
    import inspect

    args = full.parse_args(["--official-source-root", "."])
    assert args.taxi_max_train_rows == 5_000_000
    assert args.taxi_iterations == 1_400
    assert args.taxi_folds == 3
    assert args.taxi_holdout_fraction == pytest.approx(0.05)
    assert args.taxi_cache_seed == 42
    assert args.taxi_route_stat_cache_dir is None
    assert args.taxi_require_route_stat_cache is False
    assert args.dogs_folds == 5
    assert args.aerial_epochs == 6
    assert args.aerial_folds == 5
    assert args.aerial_image_size == 224
    assert args.wave2_vision_patience == 2
    assert args.phase_a_scope == "all"
    assert args.wave2_dog_breed_batch_size == 32
    assert args.wave2_dog_breed_backbone == "convnext_small"
    assert args.wave2_dog_breed_training_mode == "stability_finetune"
    assert args.wave2_dog_breed_head_learning_rate == pytest.approx(1e-3)
    assert args.wave2_dog_breed_diagnostic_fold_limit == 0
    assert args.wave2_dog_breed_diagnostic_parent_fold0_epoch1_log_loss is None
    assert args.wave2_dog_breed_diagnostic_parent_fold0_top1_accuracy is None
    assert args.wave2_histopath_max_rows == 0
    assert args.denoising_epochs == 10
    assert args.denoising_folds == 3
    assert args.denoising_patch_size == 128
    assert args.denoising_patches_per_image == 48
    assert args.denoising_batch_size == 64
    assert args.denoising_workers == 4
    assert args.denoising_base_channels == 32
    assert args.denoising_learning_rate == pytest.approx(8e-4)
    assert args.denoising_patience == 3
    assert args.siim_epochs == 8
    assert args.siim_folds == 5
    assert args.siim_inner_folds == 3
    assert args.siim_image_size == 384
    assert args.siim_batch_size == 64
    assert args.siim_backbone == "convnext_small"
    assert args.siim_secondary_backbone == "efficientnet_v2_s"
    assert args.siim_catboost_task_type == "GPU"
    assert args.siim_workers == 8
    assert args.siim_metadata_iterations == 700

    vision_signature = inspect.signature(full.wave2._run_vision)
    assert "promotion_contract" in vision_signature.parameters
    dogs_source = inspect.getsource(full.recovery.run_dogs_cats_convnext)
    aerial_source = inspect.getsource(full.recovery.run_aerial_cactus_convnext)
    assert '"threshold": 0.055' in dogs_source
    assert 'getattr(args, "aerial_backbone", "convnext_tiny")' in aerial_source
    assert '"threshold": 1.0' in aerial_source
    assert '"require_zero_binary_ranking_violations": True' in aerial_source
    assert args.dogs_epochs == 6
    assert args.dogs_batch_size == 128
    assert args.dogs_image_size == 288
    assert args.dogs_learning_rate == pytest.approx(2e-4)
    assert args.leaf_svc_c == pytest.approx(10.0)
    assert args.leaf_logistic_c == pytest.approx(10.0)
    assert args.leaf_folds == 5
    assert args.leaf_image_size == 224
    assert args.leaf_embedding_batch_size == 128
    assert args.leaf_embedding_tta == 8
    assert args.leaf_image_svc_c == pytest.approx(8.0)
    assert args.leaf_multimodal_svc_c == pytest.approx(10.0)
    assert args.pizza_nbsvm_c == pytest.approx(4.0)
    assert args.spooky_style_c == pytest.approx(1.0)
    assert args.spooky_raw_char_features == 160_000
    assert args.may_max_train_rows == 0
    assert args.may_folds == 5
    assert args.may_xgb_estimators == 2_600
    assert args.may_xgb_depth == 8
    assert args.may_catboost_iterations == 2_000
    assert args.may_catboost_depth == 8
    assert args.may_learning_rate == pytest.approx(0.035)
    assert args.may_early_stopping == 180
    assert args.may_mlp_epochs == 24
    assert args.may_mlp_width == 768
    assert args.may_mlp_blocks == 5
    assert args.may_mlp_batch_size == 4_096
    assert args.may_mlp_learning_rate == pytest.approx(1e-3)
    assert args.may_mlp_weight_decay == pytest.approx(1e-5)
    assert args.may_mlp_dropout == pytest.approx(0.08)
    assert args.may_mlp_patience == 4
    assert args.may_precomputed_cache_dir is None
    assert args.may_require_precomputed_cache is False
    assert args.wave2_jigsaw_folds == 5
    assert args.wave2_jigsaw_word_features == 350_000
    assert args.wave2_jigsaw_char_features == 500_000
    assert args.wave2_workers == 16
    assert args.wave2_audio_workers == 16
    assert args.wave2_birds_folds == 5
    assert args.wave2_birds_pair_iterations == 800
    assert args.precompute_only == ""
    assert args.wave2_whale_folds == 5
    assert args.wave2_whale_iterations == 1_200
    assert args.wave2_fast_kernels is True
    local_dog_args = full.parse_args([
        "--official-source-root", ".",
        "--phase-a-scope", "requested",
        "--wave2-dog-breed-batch-size", "16",
        "--wave2-dog-breed-diagnostic-fold-limit", "1",
        "--wave2-dog-breed-diagnostic-parent-fold0-epoch1-log-loss", "0.1562256217",
        "--wave2-dog-breed-diagnostic-parent-fold0-top1-accuracy", "0.9461956522",
    ])
    assert local_dog_args.phase_a_scope == "requested"
    assert local_dog_args.wave2_dog_breed_batch_size == 16
    assert local_dog_args.wave2_dog_breed_diagnostic_fold_limit == 1
    assert local_dog_args.wave2_dog_breed_diagnostic_parent_fold0_epoch1_log_loss == pytest.approx(
        0.1562256217
    )
    assert local_dog_args.wave2_dog_breed_diagnostic_parent_fold0_top1_accuracy == pytest.approx(
        0.9461956522
    )
    assert (
        full.RUNNERS["aerial-cactus-identification"]
        is full.recovery.run_aerial_cactus_convnext
    )
    assert full.RUNNERS["denoising-dirty-documents"] is full.recovery.run_denoising_unet
    assert full.RUNNERS["dogs-vs-cats-redux-kernels-edition"] is full.recovery.run_dogs_cats_convnext
    assert full.RUNNERS["leaf-classification"] is full.recovery.run_leaf_multimodal
    assert full.RUNNERS["siim-isic-melanoma-classification"] is full.recovery.run_siim_image_metadata
    assert (
        full.RUNNERS["spooky-author-identification"]
        is full.recovery.run_spooky_nbsvm_oof
    )
    assert (
        full.RUNNERS["tabular-playground-series-may-2022"]
        is full.recovery.run_may2022_gpu_ensemble
    )
    assert "tta_flips" in inspect.signature(full.wave2._run_vision).parameters
    precompute_args = full.parse_args([
        "--official-source-root", ".",
        "--waves", "Wave2",
        "--competitions", "mlsp-2013-birds",
        "--precompute-only", "birds",
    ])
    assert precompute_args.precompute_only == "birds"


def test_metric_promotion_gate_obeys_direction_and_extra_checks():
    passed = wave0.build_metric_promotion_gate(
        name="auc",
        metric="roc_auc",
        direction="maximize",
        score=0.95,
        threshold=0.94,
    )
    failed = wave0.build_metric_promotion_gate(
        name="rmse",
        metric="rmse",
        direction="minimize",
        score=2.8,
        threshold=2.85,
        extra_checks={"temporal": False},
    )
    assert passed["passed"] is True
    assert failed["passed"] is False


def _may2022_frame(strings: list[str], *, include_target: bool) -> pd.DataFrame:
    rows = len(strings)
    payload: dict[str, object] = {"id": np.arange(rows, dtype=np.int64)}
    for index in range(31):
        if index == 27:
            payload["f_27"] = strings
        else:
            payload[f"f_{index:02d}"] = np.zeros(rows, dtype=np.float32)
    payload["f_02"] = np.array([3.0, -3.0, 0.0][:rows], dtype=np.float32)
    payload["f_21"] = np.array([3.0, -3.0, 0.0][:rows], dtype=np.float32)
    payload["f_05"] = np.array([2.6, -2.8, 0.0][:rows], dtype=np.float32)
    payload["f_22"] = np.array([2.6, -2.8, 0.0][:rows], dtype=np.float32)
    payload["f_00"] = np.array([2.0, -2.0, 0.0][:rows], dtype=np.float32)
    payload["f_01"] = np.array([2.0, -2.0, 0.0][:rows], dtype=np.float32)
    payload["f_26"] = np.array([2.0, -2.0, 0.0][:rows], dtype=np.float32)
    if include_target:
        payload["target"] = np.array([1, 0, 1][:rows], dtype=np.int8)
    return pd.DataFrame(payload)


def test_may2022_feature_builder_recovers_string_and_geometric_signal():
    train = _may2022_frame(
        ["ABCDEFGHIJ", "AAAAAAAAAA", "JIHGFEDCBA"], include_target=True
    )
    test = _may2022_frame(
        ["ABCDEFGHIJ", "AAAAAAAAAA", "JIHGFEDCBA"], include_target=False
    )
    train_features, test_features, diagnostics = full.recovery.build_may2022_features(
        train, test
    )
    assert list(train_features.columns) == list(test_features.columns)
    assert train_features.shape == test_features.shape
    assert diagnostics["target_derived_features"] == 0
    assert diagnostics["base_numeric_feature_count"] == 30
    assert diagnostics["f27_position_one_hot_count"] == 200
    assert "target" not in train_features
    assert "id" not in train_features
    assert train_features.loc[0, "f27_unique_count"] == 10
    assert train_features.loc[1, "f27_unique_count"] == 1
    assert train_features.loc[1, "f27_count_A"] == 10
    assert train_features.loc[0, "f27_pos_0_A"] == 1
    assert train_features.loc[0, "f27_pos_0_B"] == 0
    assert train_features.loc[0, "interaction_f02_f21"] == 1
    assert train_features.loc[1, "interaction_f02_f21"] == -1
    assert train_features.loc[0, "interaction_f05_f22"] == 1
    assert train_features.loc[1, "interaction_f05_f22"] == -1
    assert train_features.loc[0, "interaction_f00_f01_f26"] == 1
    assert train_features.loc[1, "interaction_f00_f01_f26"] == -1
    assert all(pd.api.types.is_numeric_dtype(train_features[c]) for c in train_features)
    assert np.isfinite(train_features.to_numpy()).all()


def test_may2022_feature_builder_rejects_malformed_f27():
    train = _may2022_frame(["TOO_SHORT"], include_target=True)
    test = _may2022_frame(["ABCDEFGHIJ"], include_target=False)
    with pytest.raises(ValueError, match="must contain 10 characters"):
        full.recovery.build_may2022_features(train, test)


def test_may2022_feature_contract_excludes_id_target_and_hashes_ordered_schema():
    train = _may2022_frame(
        ["ABCDEFGHIJ", "AAAAAAAAAA", "JIHGFEDCBA"], include_target=True
    )
    test = _may2022_frame(
        ["ABCDEFGHIJ", "AAAAAAAAAA", "JIHGFEDCBA"], include_target=False
    )
    baseline, _, diagnostics = full.recovery.build_may2022_features(train, test)
    perturbed = train.copy()
    perturbed["id"] = perturbed["id"] + 10_000
    perturbed["target"] = 1 - perturbed["target"]
    changed, _, changed_diagnostics = full.recovery.build_may2022_features(perturbed, test)
    pd.testing.assert_frame_equal(baseline, changed)
    assert diagnostics["id_feature_count"] == 0
    assert diagnostics["target_feature_count"] == 0
    assert diagnostics["ordered_feature_names"] == list(baseline.columns)
    assert diagnostics["feature_schema_sha256"] == changed_diagnostics["feature_schema_sha256"]
    assert len(diagnostics["feature_schema_sha256"]) == 64


def test_may2022_medal_mode_requires_every_training_row():
    contract = full.recovery.validate_may2022_full_data_contract(
        train_rows_available=900_000,
        train_rows_used=900_000,
    )
    assert contract["full_training_data"] is True
    assert contract["medal_mode_full_data_required"] is True
    with pytest.raises(RuntimeError, match="requires every public training row"):
        full.recovery.validate_may2022_full_data_contract(
            train_rows_available=900_000,
            train_rows_used=899_999,
        )


def test_may2022_oof_blend_is_at_least_as_good_as_best_component():
    target = np.array([0, 0, 0, 1, 1, 1])
    xgboost = np.array([0.05, 0.20, 0.55, 0.45, 0.75, 0.95])
    catboost = np.array([0.10, 0.50, 0.25, 0.65, 0.55, 0.90])
    blend = full.recovery.select_may2022_binary_blend(xgboost, catboost, target)
    assert blend["space"] in {"raw_probability", "rank"}
    assert blend["oof_auc"] >= max(
        blend["xgboost_oof_auc"], blend["catboost_oof_auc"]
    )
    prediction = full.recovery.apply_may2022_binary_blend(xgboost, catboost, blend)
    assert prediction.shape == target.shape
    assert np.isfinite(prediction).all()


def test_may2022_three_model_simplex_blend_is_deterministic():
    target = np.array([0, 0, 0, 1, 1, 1])
    components = {
        "residual_mlp": np.array([0.05, 0.25, 0.55, 0.45, 0.80, 0.95]),
        "xgboost": np.array([0.10, 0.20, 0.50, 0.60, 0.70, 0.90]),
        "catboost": np.array([0.15, 0.45, 0.30, 0.65, 0.55, 0.85]),
    }
    first = full.recovery.select_may2022_multimodel_blend(components, target)
    second = full.recovery.select_may2022_multimodel_blend(components, target)
    assert first == second
    assert sum(first["weights"].values()) == pytest.approx(1.0)
    assert first["oof_auc"] >= max(first["component_oof_auc"].values())
    prediction = full.recovery.apply_may2022_multimodel_blend(components, first)
    assert prediction.shape == target.shape
    assert np.isfinite(prediction).all()


def test_may2022_deployment_score_and_three_seed_confirmation_gate():
    target = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int8)
    folds = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    components = {
        "residual_mlp": np.array([0.01, 0.02, 0.98, 0.99, 0.03, 0.04, 0.96, 0.97]),
        "xgboost": np.array([0.02, 0.01, 0.97, 0.98, 0.04, 0.03, 0.95, 0.96]),
        "catboost": np.array([0.03, 0.04, 0.96, 0.95, 0.01, 0.02, 0.98, 0.99]),
    }
    blend = full.recovery.select_may2022_multimodel_blend(components, target)
    scored = full.recovery.score_may2022_deployment_blend_oof(
        components, target, folds, blend
    )
    assert scored["oof_auc"] == pytest.approx(1.0)
    assert set(scored["fold_auc"]) == {"0", "1"}
    assert all(value == pytest.approx(1.0) for value in scored["fold_auc"].values())

    records = [
        {
            "seed": seed,
            "aggregate_oof_auc": 0.999,
            "confirmation_oof_auc": 0.9988,
            "fold_auc": [0.9986, 0.9987, 0.9989],
        }
        for seed in (42, 43, 44)
    ]
    gate = full.recovery.evaluate_may2022_confirmation_gate(records)
    assert gate["passed"] is True
    assert gate["confirmation_seed_count"] == 3
    assert gate["eligible_seed_count"] == 3
    assert gate["every_seed_eligible"] is True
    assert gate["claim_boundary"].endswith("not an official medal.")
    assert full.recovery.evaluate_may2022_confirmation_gate(records[:1])["passed"] is False
    failed_extra = {**records[-1], "seed": 45, "aggregate_oof_auc": 0.95}
    assert full.recovery.evaluate_may2022_confirmation_gate([*records, failed_extra])["passed"] is False


def test_may2022_single_run_gate_is_fail_closed_and_scores_deployment_rule():
    one_seed = full.recovery.evaluate_may2022_confirmation_gate([
        {
            "seed": 42,
            "aggregate_oof_auc": 0.9991,
            "confirmation_oof_auc": 0.9990,
            "fold_auc": [0.9987, 0.9988, 0.9989],
        }
    ])
    contract = full.recovery.validate_may2022_full_data_contract(
        train_rows_available=800_000,
        train_rows_used=800_000,
    )
    gate = full.recovery.build_may2022_run_promotion_gate(
        seed_gate=one_seed,
        full_data_contract=contract,
        deployment_oof_auc=0.9989,
        deployment_fold_auc=[0.9985, 0.9986, 0.9987],
    )
    assert gate["internal_score"] == pytest.approx(0.9989)
    assert gate["checks"]["exact_deployment_oof_auc"] is True
    assert gate["checks"]["three_seed_confirmation_complete"] is False
    assert gate["passed"] is False


def test_may2022_artifact_manifest_hashes_required_files(tmp_path: Path):
    required = {
        "may2022_fold_assignments.npz",
        "may2022_oof_ensemble.npz",
        "may2022_ensemble_diagnostics.json",
        "may2022_promotion_gate.json",
        "promotion_gate.json",
        "submission.csv",
        "submission_validation.json",
    }
    for name in required:
        (tmp_path / name).write_bytes(name.encode("utf-8"))
    source = tmp_path / "runner_source.py"
    source.write_text("print('runner')\n", encoding="utf-8")
    record = full.recovery.write_may2022_artifact_manifest(
        tmp_path,
        source_paths=(source,),
    )
    manifest = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
    names = {item["relative_path"] for item in manifest["artifacts"]}
    assert required <= names
    assert "may2022_environment_manifest.json" in names
    assert manifest["required_artifacts_present"] is True
    assert record["artifact_count"] == len(manifest["artifacts"])
    assert len(record["sha256"]) == 64


def test_may2022_residual_mlp_cpu_forward_is_finite():
    torch = pytest.importorskip("torch")
    model = full.recovery.build_may2022_residual_mlp(
        12, width=32, block_count=2, dropout=0.0
    ).eval()
    with torch.inference_mode():
        logits = model(torch.rand(5, 12))
    assert logits.shape == (5,)
    assert torch.isfinite(logits).all()


def test_denoising_unet_cpu_forward_is_shape_preserving_and_finite():
    torch = pytest.importorskip("torch")
    model = full.recovery.build_denoising_unet(base_channels=8).eval()
    dirty = torch.rand(2, 1, 32, 40)
    with torch.inference_mode():
        residual = model(dirty)
        restored = torch.clamp(dirty + residual, 0.0, 1.0)
    assert residual.shape == dirty.shape
    assert torch.isfinite(residual).all()
    assert torch.isfinite(restored).all()
    assert float(restored.min()) >= 0.0
    assert float(restored.max()) <= 1.0


def test_denoising_nested_document_folds_keep_outer_oof_untouched():
    names = [f"document_{index:03d}.png" for index in range(30)]
    records = full.recovery.build_denoising_nested_document_folds(
        names, requested_folds=3, seed=42
    )
    assert len(records) == 3
    coverage: dict[str, int] = {name: 0 for name in names}
    for record in records:
        outer_fit = set(record["outer_fit"])
        outer_valid = set(record["outer_valid"])
        inner_fit = set(record["inner_fit"])
        inner_checkpoint = set(record["inner_checkpoint"])
        assert outer_fit.isdisjoint(outer_valid)
        assert inner_fit.isdisjoint(inner_checkpoint)
        assert inner_fit | inner_checkpoint == outer_fit
        assert outer_valid.isdisjoint(inner_fit | inner_checkpoint)
        for name in outer_valid:
            coverage[name] += 1
    assert set(coverage.values()) == {1}
    assert records == full.recovery.build_denoising_nested_document_folds(
        list(reversed(names)), requested_folds=3, seed=42
    )


def test_siim_metadata_features_are_aligned_and_finite():
    train = pd.DataFrame({
        "image_name": ["a", "b", "c"],
        "patient_id": ["p1", "p1", "p2"],
        "sex": ["male", "female", None],
        "age_approx": [50, None, 70],
        "anatom_site_general_challenge": ["torso", "head/neck", "torso"],
        "target": [0, 1, 0],
        "diagnosis": ["unknown", "melanoma", "unknown"],
    })
    test = pd.DataFrame({
        "image_name": ["d", "e"],
        "patient_id": ["p3", "p3"],
        "sex": ["female", "other"],
        "age_approx": [60, 80],
        "anatom_site_general_challenge": ["lower extremity", None],
    })
    train_features, test_features, names = full.recovery.siim_metadata_features(train, test)
    assert train_features.shape[0] == len(train)
    assert test_features.shape[0] == len(test)
    assert train_features.shape[1] == test_features.shape[1] == len(names)
    assert np.isfinite(train_features).all()
    assert np.isfinite(test_features).all()
    assert not any("diagnosis" in name or "target" in name for name in names)


def test_siim_auc_blend_is_finite_and_selects_image_signal():
    truth = np.array([0, 0, 1, 1])
    image = np.array([0.05, 0.2, 0.8, 0.95])
    metadata = np.array([0.6, 0.4, 0.3, 0.5])
    prediction, weight, mode, score = full.recovery.select_siim_auc_blend(
        image, metadata, truth
    )
    assert score == pytest.approx(1.0)
    assert 0.0 <= weight <= 1.0
    assert mode in {"raw", "rank"}
    assert np.isfinite(prediction).all()
    test_prediction = full.recovery.apply_siim_blend(
        image, metadata, image_weight=weight, mode=mode
    )
    assert np.all((test_prediction > 0.0) & (test_prediction < 1.0))


def test_siim_three_channel_blend_and_crossfit_are_deterministic():
    truth = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int8)
    folds = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    channels = {
        "pure_image": np.array([0.1, 0.3, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8]),
        "image_metadata_fusion": np.array([0.01, 0.02, 0.98, 0.99, 0.03, 0.04, 0.96, 0.97]),
        "metadata_catboost": np.array([0.4, 0.6, 0.3, 0.7, 0.5, 0.2, 0.8, 0.4]),
    }
    first = full.recovery.select_siim_multichannel_auc_blend(channels, truth)
    second = full.recovery.select_siim_multichannel_auc_blend(channels, truth)
    assert first == second
    assert sum(first["weights"].values()) == pytest.approx(1.0)
    blended = full.recovery.apply_siim_multichannel_blend(channels, first)
    assert np.isfinite(blended).all()
    crossfit, records = full.recovery.cross_fit_siim_multichannel_auc_blend(
        channels, truth, folds
    )
    assert len(records) == 2
    assert np.isfinite(crossfit).all()
    assert all(record["outer_auc"] == pytest.approx(1.0) for record in records)


def test_siim_fold_seed_replays_python_numpy_and_torch_rngs(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    first_contract = full.recovery.seed_siim_fold(4201)
    first = (random.random(), np.random.random(), torch.rand(3))
    second_contract = full.recovery.seed_siim_fold(4201)
    second = (random.random(), np.random.random(), torch.rand(3))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert first_contract == second_contract
    assert first_contract["deterministic_algorithms_requested"] is True
    assert first_contract["deterministic_warn_only"] is False
    assert first_contract["cublas_workspace_config"] == ":4096:8"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_siim_fold_seed_sets_cublas_workspace_before_importing_torch():
    source = inspect.getsource(full.recovery.seed_siim_fold)
    assert source.index('os.environ["CUBLAS_WORKSPACE_CONFIG"]') < source.index(
        "import torch"
    )


def test_siim_fold_seed_rejects_invalid_cublas_workspace_config(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "invalid")
    with pytest.raises(RuntimeError, match="SIIM deterministic mode requires"):
        full.recovery.seed_siim_fold(4201)


def test_siim_fold_seed_can_enable_fast_a40_kernels(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "leave-fast-mode-unchanged")
    try:
        contract = full.recovery.seed_siim_fold(4201, fast_kernel_mode=True)
        assert contract["fast_kernel_mode"] is True
        assert contract["tf32_enabled"] is True
        assert contract["cudnn_benchmark"] is True
        assert contract["cudnn_deterministic"] is False
        assert contract["deterministic_algorithms_requested"] is False
        assert contract["deterministic_warn_only"] is False
        assert contract["cublas_workspace_config"] == "leave-fast-mode-unchanged"
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == "leave-fast-mode-unchanged"
    finally:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        full.recovery.seed_siim_fold(4201)


def test_siim_jpeg_loader_uses_dct_scaled_decode(tmp_path):
    Image = pytest.importorskip("PIL.Image")

    path = tmp_path / "large_siim.jpg"
    Image.new("RGB", (2048, 1536), color=(80, 120, 160)).save(path, format="JPEG")

    decoded = full.recovery._open_siim_rgb_for_transform(path, decode_size=352)

    assert decoded.mode == "RGB"
    assert decoded.size[0] <= 1024
    assert decoded.size[1] <= 768


def test_siim_jpeg_loader_rejects_invalid_decode_size(tmp_path):
    path = tmp_path / "unused.jpg"
    with pytest.raises(ValueError, match="decode size must be positive"):
        full.recovery._open_siim_rgb_for_transform(path, decode_size=0)


def test_siim_dermoscopy_views_are_deterministic_and_robust():
    from PIL import Image

    pixels = np.zeros((180, 220, 3), dtype=np.uint8)
    pixels[18:162, 24:196] = np.asarray([188, 142, 121], dtype=np.uint8)
    yy, xx = np.mgrid[:180, :220]
    lesion = ((yy - 90) / 38.0) ** 2 + ((xx - 110) / 50.0) ** 2 <= 1.0
    pixels[lesion] = np.asarray([92, 54, 48], dtype=np.uint8)
    pixels[45:140, 108:111] = 4
    source = Image.fromarray(pixels, mode="RGB")

    cropped = full.recovery.siim_crop_dark_border(source)
    assert cropped.width < source.width
    assert cropped.height < source.height
    normalized = full.recovery.siim_shades_of_gray_color_constancy(cropped)
    repaired = full.recovery.siim_suppress_dark_hairs(normalized)
    assert np.asarray(repaired)[30:120, repaired.width // 2].mean() > np.asarray(normalized)[
        30:120, normalized.width // 2
    ].mean()

    first = full.recovery.prepare_siim_dermoscopy_views(source)
    second = full.recovery.prepare_siim_dermoscopy_views(source)
    assert set(first) == {"full_image", "lesion_focus"}
    for name in first:
        assert first[name].mode == "RGB"
        np.testing.assert_array_equal(np.asarray(first[name]), np.asarray(second[name]))
    assert first["lesion_focus"].width <= first["full_image"].width
    assert first["lesion_focus"].height <= first["full_image"].height
    assert (
        first["lesion_focus"].width < first["full_image"].width
        or first["lesion_focus"].height < first["full_image"].height
    )
    assert first["lesion_focus"].width >= int(first["full_image"].width * 0.52)
    assert first["lesion_focus"].height >= int(first["full_image"].height * 0.52)


def test_siim_paired_transform_replays_one_rng_draw_for_both_views():
    torch = pytest.importorskip("torch")

    class RandomTransform:
        def __call__(self, _image):
            return torch.tensor(
                [torch.rand(1).item(), random.random(), np.random.random()],
                dtype=torch.float64,
            )

    transform = RandomTransform()
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)
    full_view, lesion_view = full.recovery.apply_siim_paired_transform(
        transform, object(), object()
    )
    after_pair = (
        torch.rand(1).item(),
        random.random(),
        np.random.random(),
    )

    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)
    expected = transform(object())
    after_single = (
        torch.rand(1).item(),
        random.random(),
        np.random.random(),
    )

    torch.testing.assert_close(full_view, expected)
    torch.testing.assert_close(lesion_view, expected)
    np.testing.assert_allclose(after_pair, after_single, rtol=0.0, atol=0.0)


def test_siim_dermoscopy_profiles_are_deterministic_and_distinct():
    from PIL import Image

    pixels = np.zeros((180, 220, 3), dtype=np.uint8)
    pixels[18:162, 24:196] = np.asarray([188, 142, 121], dtype=np.uint8)
    yy, xx = np.mgrid[:180, :220]
    lesion = ((yy - 90) / 38.0) ** 2 + ((xx - 110) / 50.0) ** 2 <= 1.0
    pixels[lesion] = np.asarray([92, 54, 48], dtype=np.uint8)
    pixels[45:140, 108:111] = 4
    source = Image.fromarray(pixels, mode="RGB")

    outputs = {}
    for profile in full.recovery.SIIM_PREPROCESSING_PROFILES:
        first = full.recovery.prepare_siim_dermoscopy_views(source, profile=profile)
        second = full.recovery.prepare_siim_dermoscopy_views(source, profile=profile)
        assert set(first) == {"full_image", "lesion_focus"}
        for view_name in first:
            np.testing.assert_array_equal(
                np.asarray(first[view_name]),
                np.asarray(second[view_name]),
            )
        outputs[profile] = first

    np.testing.assert_array_equal(
        np.asarray(outputs["raw_multiview_v1"]["full_image"]),
        np.asarray(source),
    )
    assert outputs["raw_multiview_v1"]["full_image"].size != outputs[
        "border_multiview_v1"
    ]["full_image"].size
    assert not np.array_equal(
        np.asarray(outputs["color_multiview_v1"]["full_image"]),
        np.asarray(outputs["hair_multiview_v1"]["full_image"]),
    )
    assert outputs["robust_multiview_v1"]["lesion_focus"].size != outputs[
        "hair_multiview_v1"
    ]["lesion_focus"].size


def test_siim_dermoscopy_profiles_reject_unknown_profile():
    from PIL import Image

    with pytest.raises(ValueError, match="Unsupported SIIM preprocessing profile"):
        full.recovery.prepare_siim_dermoscopy_views(
            Image.new("RGB", (32, 32)),
            profile="unknown_multiview",
        )


def test_siim_image_content_manifest_hashes_exact_duplicates(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    train_paths = [tmp_path / "train_a.jpg", tmp_path / "train_b.jpg"]
    test_paths = [tmp_path / "test_c.jpg"]
    for path in train_paths:
        Image.new("RGB", (32, 32), color=(20, 40, 60)).save(path, format="JPEG")
    Image.new("RGB", (32, 32), color=(90, 100, 110)).save(
        test_paths[0], format="JPEG"
    )

    manifest, report = full.recovery.build_siim_image_content_manifest(
        train_paths,
        test_paths,
        train_ids=["a", "b"],
        test_ids=["c"],
        workers=2,
    )

    assert manifest["source"].tolist() == ["train", "train", "test"]
    assert manifest.loc[0, "content_sha256"] == manifest.loc[1, "content_sha256"]
    assert manifest.loc[2, "content_sha256"] != manifest.loc[0, "content_sha256"]
    assert manifest["decoded_pixel_sha256"].str.len().eq(64).all()
    assert manifest["perceptual_dhash64"].str.len().eq(16).all()
    assert manifest.loc[:1, "duplicate_scope"].tolist() == [
        "within_train+within_train_decoded_pixels",
        "within_train+within_train_decoded_pixels",
    ]
    assert report["train_duplicate_hashes"] == 1
    assert report["train_duplicate_rows"] == 2
    assert report["private_labels_used"] is False

    manifest_path = tmp_path / "siim_image_content_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    reused, reuse_report = full.recovery.load_precomputed_siim_image_content_manifest(
        manifest_path,
        train_ids=["a", "b"],
        test_ids=["c"],
    )
    assert reused["image_name"].tolist() == ["a", "b", "c"]
    assert reused["perceptual_dhash64"].astype(str).str.len().eq(16).all()
    assert reuse_report["precomputed_manifest_reused"] is True
    assert reuse_report["ordered_content_vector_sha256"] == report[
        "ordered_content_vector_sha256"
    ]


def test_siim_perceptual_hash_candidates_do_not_union_cross_patient_rows(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    first = tmp_path / "quality_70.jpg"
    second = tmp_path / "quality_95.jpg"
    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(64, dtype=np.uint8)[None, :] * 3
    pixels[:, :, 1] = np.arange(64, dtype=np.uint8)[:, None] * 3
    pixels[:, :, 2] = 80
    image = Image.fromarray(pixels, mode="RGB")
    image.save(first, format="JPEG", quality=70)
    image.save(second, format="JPEG", quality=95)
    manifest, _ = full.recovery.build_siim_image_content_manifest(
        [first, second],
        [],
        train_ids=["a", "b"],
        test_ids=[],
        workers=1,
    )
    assert manifest.loc[0, "content_sha256"] != manifest.loc[1, "content_sha256"]
    assert manifest.loc[0, "perceptual_dhash64"] == manifest.loc[1, "perceptual_dhash64"]

    metadata = pd.DataFrame({
        "image_name": ["a", "b"],
        "patient_id": ["p0", "p1"],
    })
    groups, report = full.recovery.build_siim_content_connected_groups(
        metadata,
        np.asarray([0, 1], dtype=np.int8),
        manifest["content_sha256"].to_numpy(dtype=str),
        decoded_pixel_sha256=manifest["decoded_pixel_sha256"].to_numpy(dtype=str),
        perceptual_dhash64=manifest["perceptual_dhash64"].to_numpy(dtype=str),
        perceptual_max_distance=1,
    )
    assert groups[0] != groups[1]
    assert report["perceptual_hamming_max_distance"] == 1
    assert report["perceptual_exact_candidate_edges"] == 1
    assert report["perceptual_edges_applied_to_groups"] == 0
    assert report["perceptual_edge_policy"] == (
        "audit_only_no_cross_patient_union_v1"
    )


def test_siim_strict_patient_folds_reject_silent_fold_count_degradation():
    metadata = pd.DataFrame({
        "image_name": [f"image_{index}" for index in range(20)],
        "patient_id": [f"p{index}" for index in range(20)],
    })
    target = np.asarray([index % 2 for index in range(20)], dtype=np.int8)
    groups = np.asarray(
        ["giant"] * 16 + [f"tail_{index}" for index in range(4)],
        dtype=str,
    )

    with pytest.raises(RuntimeError, match="requested fold count"):
        full.recovery.build_siim_patient_folds(
            metadata,
            target,
            requested_folds=5,
            seed=43,
            group_values=groups,
        )


def test_siim_formal_fold_contract_is_exactly_five_outer_by_three_inner(tmp_path):
    assert full.recovery.validate_siim_formal_fold_contract(
        outer_folds=5,
        inner_folds=3,
    ) == (5, 3)
    with pytest.raises(RuntimeError, match="exactly 5 outer x 3 inner"):
        full.recovery.validate_siim_formal_fold_contract(
            outer_folds=4,
            inner_folds=3,
        )
    with pytest.raises(RuntimeError, match="exactly 5 outer x 3 inner"):
        full.recovery.validate_siim_formal_fold_contract(
            outer_folds=5,
            inner_folds=2,
        )
    with pytest.raises(SystemExit):
        full.parse_args([
            "--official-source-root",
            str(tmp_path),
            "--siim-folds",
            "4",
        ])
    with pytest.raises(SystemExit):
        full.parse_args([
            "--official-source-root",
            str(tmp_path),
            "--siim-inner-folds",
            "2",
        ])


def _siim_terminal_resume_fixture(tmp_path: Path):
    task_root = tmp_path / full.SIIM_COMPETITION_ID
    resume = task_root / "attempts" / "siim_resume_state"
    resume.mkdir(parents=True)
    manifest = tmp_path / "siim_image_content_manifest.csv"
    manifest.write_text("image_name,sha256\nfixture,a\n", encoding="utf-8")
    manifest_sha256 = full.sha256_file(manifest)
    contract = {
        "schema": "evomind.siim.formal_resume_contract.v1",
        "competition_id": full.SIIM_COMPETITION_ID,
        "model_seed": 43,
        "image_content_manifest_sha256": manifest_sha256,
        "leakage_group_policy": full.recovery.SIIM_LEAKAGE_GROUP_POLICY,
        "perceptual_edge_policy": full.recovery.SIIM_PERCEPTUAL_EDGE_POLICY,
        "outer_folds": 5,
        "inner_folds": 3,
        "model": {"workers": 8, "fast_kernel_mode": True},
        "adapter_source_sha256": full.sha256_file(Path(full.recovery.__file__)),
        "wave2_source_sha256": full.sha256_file(Path(full.wave2.__file__)),
    }
    contract["contract_sha256"] = full._resume_contract_sha256(contract)
    (resume / "resume_contract.json").write_text(
        json.dumps(contract),
        encoding="utf-8",
    )
    result = {
        "status": "candidate_ready_confirmation_pending",
        "candidate_only": True,
        "valid_submission": True,
        "official_grader_executed": False,
        "budget": {
            "seed": 43,
            "folds": 5,
            "image_content_manifest_sha256": manifest_sha256,
            "resume_contract_sha256": contract["contract_sha256"],
            "duplicate_group_report": {
                "schema": "evomind.siim_patient_content_connected_groups.v3",
                "leakage_group_policy": full.recovery.SIIM_LEAKAGE_GROUP_POLICY,
                "perceptual_edge_policy": full.recovery.SIIM_PERCEPTUAL_EDGE_POLICY,
                "perceptual_edges_applied_to_groups": 0,
            },
        },
    }
    args = SimpleNamespace(seed=43, siim_image_content_manifest=manifest)
    return task_root, result, args, contract


def test_siim_resume_accepts_only_terminal_result_bound_to_current_contract(tmp_path):
    task_root, result, args, _contract = _siim_terminal_resume_fixture(tmp_path)

    checks = full.siim_terminal_resume_checks(task_root, result, args)

    assert checks and all(checks.values())


@pytest.mark.parametrize("mutation", ["passed_status", "two_folds", "changed_contract"])
def test_siim_resume_rejects_legacy_terminal_result(tmp_path, mutation):
    task_root, result, args, contract = _siim_terminal_resume_fixture(tmp_path)
    if mutation == "passed_status":
        result["status"] = "passed"
    elif mutation == "two_folds":
        result["budget"]["folds"] = 2
    else:
        contract["outer_folds"] = 2
        (task_root / "attempts" / "siim_resume_state" / "resume_contract.json").write_text(
            json.dumps(contract),
            encoding="utf-8",
        )

    checks = full.siim_terminal_resume_checks(task_root, result, args)

    assert not all(checks.values())


def test_siim_content_connected_groups_close_patient_duplicate_chains():
    metadata = pd.DataFrame({
        "image_name": [f"image_{index}" for index in range(8)],
        "patient_id": ["p0", "p0", "p1", "p1", "p2", "p2", "p3", "p3"],
    })
    target = np.asarray([0, 1, 1, 0, 0, 1, 0, 1], dtype=np.int8)
    hashes = np.asarray([
        "a" * 64,
        "b" * 64,
        "b" * 64,
        "c" * 64,
        "c" * 64,
        "d" * 64,
        "e" * 64,
        "f" * 64,
    ])

    groups, report = full.recovery.build_siim_content_connected_groups(
        metadata, target, hashes
    )

    assert len(set(groups[:6])) == 1
    assert groups[6] == groups[7]
    assert groups[0] != groups[6]
    assert report["duplicate_hashes"] == 2
    assert report["components_joining_multiple_patients"] == 1
    assert report["largest_component_rows"] == 6
    assert report["exact_duplicate_label_conflicts"] == 0


def test_siim_content_connected_groups_reject_conflicting_exact_duplicates():
    metadata = pd.DataFrame({
        "image_name": ["a", "b"],
        "patient_id": ["p0", "p1"],
    })
    with pytest.raises(RuntimeError, match="conflicting public labels"):
        full.recovery.build_siim_content_connected_groups(
            metadata,
            np.asarray([0, 1], dtype=np.int8),
            np.asarray(["a" * 64, "a" * 64]),
        )


def test_siim_artifact_manifest_hashes_bounded_files(tmp_path):
    first = tmp_path / "first.bin"
    second = tmp_path / "second.json"
    first.write_bytes(b"first")
    second.write_text('{"second": true}', encoding="utf-8")

    path, digest, payload = full.recovery.write_siim_artifact_manifest(
        tmp_path,
        [(first, "checkpoint"), (second, "report")],
        source_sha256="f" * 64,
    )

    assert path.is_file()
    assert digest == full.recovery._path_sha256(path)
    assert payload["artifact_count"] == 2
    assert [record["path"] for record in payload["artifacts"]] == [
        "first.bin",
        "second.json",
    ]
    assert all(len(record["sha256"]) == 64 for record in payload["artifacts"])


def test_siim_fold_harmonization_compares_raw_logit_and_rank_without_outer_fit():
    truth = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int8)
    folds = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int8)
    oof = np.asarray([0.10, 0.20, 0.40, 0.50, 0.80, 0.90], dtype=float)
    test_by_fold = np.asarray([
        [0.10, 0.15, 0.20, 0.25],
        [0.40, 0.45, 0.50, 0.55],
        [0.75, 0.80, 0.85, 0.90],
    ])

    oof_variants, test_variants = full.recovery.build_siim_fold_harmonization_variants(
        oof, test_by_fold, folds
    )
    crossfit, test, records, final = full.recovery.select_siim_crossfit_harmonization(
        oof_variants,
        test_variants,
        truth,
        folds,
    )

    assert set(oof_variants) == {
        "raw_probability",
        "fold_logit_zscore",
        "fold_percentile_rank",
    }
    assert final["oof_auc_by_mode"]["raw_probability"] < 1.0
    assert final["selected_mode"] in {
        "fold_logit_zscore",
        "fold_percentile_rank",
    }
    assert full.recovery.compute_metric("roc_auc", truth, crossfit) == pytest.approx(1.0)
    assert len(records) == 3
    assert np.isfinite(test).all()


def test_siim_fold_harmonization_epsilon_clips_valid_sigmoid_endpoints():
    folds = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int8)
    oof = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float64)
    test_by_fold = np.asarray([
        [0.0, 0.25, 1.0],
        [0.1, 0.50, 0.9],
        [0.0, 0.75, 1.0],
    ], dtype=np.float64)

    oof_variants, test_variants = (
        full.recovery.build_siim_fold_harmonization_variants(
            oof, test_by_fold, folds
        )
    )

    assert np.all((oof_variants["raw_probability"] > 0.0) & (oof_variants["raw_probability"] < 1.0))
    assert np.all((test_variants["raw_probability"] > 0.0) & (test_variants["raw_probability"] < 1.0))
    assert oof_variants["raw_probability"][0] == pytest.approx(1e-6)
    assert oof_variants["raw_probability"][-1] == pytest.approx(1.0 - 1e-6)
    assert np.array_equal(
        np.argsort(oof, kind="stable"),
        np.argsort(oof_variants["raw_probability"], kind="stable"),
    )


@pytest.mark.parametrize("invalid", [float("nan"), -1e-8, 1.0 + 1e-8])
def test_siim_fold_harmonization_still_rejects_nonfinite_or_out_of_range(invalid):
    folds = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int8)
    oof = np.asarray([0.1, 0.2, 0.4, 0.6, 0.8, 0.9], dtype=np.float64)
    test_by_fold = np.asarray([
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
        [0.3, 0.4, 0.5],
    ], dtype=np.float64)
    oof[0] = invalid

    with pytest.raises(RuntimeError, match="finite and closed-unit"):
        full.recovery.build_siim_fold_harmonization_variants(
            oof, test_by_fold, folds
        )


def test_siim_joint_harmonized_blend_does_not_read_held_out_fold_labels():
    truth = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int8)
    folds = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int8)
    image_oof = np.asarray([0.10, 0.20, 0.40, 0.50, 0.80, 0.90])
    fusion_oof = np.asarray([0.12, 0.25, 0.35, 0.52, 0.77, 0.93])
    image_test_by_fold = np.asarray([
        [0.10, 0.15, 0.20],
        [0.40, 0.45, 0.50],
        [0.80, 0.85, 0.90],
    ])
    fusion_test_by_fold = np.asarray([
        [0.11, 0.16, 0.21],
        [0.38, 0.46, 0.54],
        [0.76, 0.84, 0.92],
    ])
    image_variants, image_test_variants = (
        full.recovery.build_siim_fold_harmonization_variants(
            image_oof, image_test_by_fold, folds
        )
    )
    fusion_variants, fusion_test_variants = (
        full.recovery.build_siim_fold_harmonization_variants(
            fusion_oof, fusion_test_by_fold, folds
        )
    )
    metadata_oof = np.asarray([0.2, 0.7, 0.3, 0.6, 0.4, 0.8])
    metadata_test_by_fold = np.asarray([
        [0.20, 0.45, 0.70],
        [0.25, 0.50, 0.75],
        [0.30, 0.55, 0.80],
    ])
    metadata_variants, metadata_test_variants = (
        full.recovery.build_siim_fold_harmonization_variants(
            metadata_oof, metadata_test_by_fold, folds
        )
    )

    first, _, first_records, first_final = (
        full.recovery.cross_fit_siim_harmonized_multichannel_blend(
            image_variants,
            fusion_variants,
            metadata_variants,
            image_test_variants,
            fusion_test_variants,
            metadata_test_variants,
            truth,
            folds,
        )
    )
    mutated = truth.copy()
    mutated[folds == 0] = 1 - mutated[folds == 0]
    changed, _, changed_records, _ = (
        full.recovery.cross_fit_siim_harmonized_multichannel_blend(
            image_variants,
            fusion_variants,
            metadata_variants,
            image_test_variants,
            fusion_test_variants,
            metadata_test_variants,
            mutated,
            folds,
        )
    )

    assert np.array_equal(first[folds == 0], changed[folds == 0])
    assert first_records[0]["weights"] == changed_records[0]["weights"]
    assert first_records[0]["image_harmonization_mode"] == changed_records[0][
        "image_harmonization_mode"
    ]
    assert first_records[0]["fusion_harmonization_mode"] == changed_records[0][
        "fusion_harmonization_mode"
    ]
    assert first_records[0]["metadata_harmonization_mode"] == changed_records[0][
        "metadata_harmonization_mode"
    ]
    assert first_records[0]["grid_denominator"] == 10
    assert first_final["grid_denominator"] == 20
    assert set(first_records[0]["meta_fit_image_auc_by_mode"]) == {
        "raw_probability",
        "fold_logit_zscore",
        "fold_percentile_rank",
    }
    assert set(first_records[0]["meta_fit_fusion_auc_by_mode"]) == {
        "raw_probability",
        "fold_logit_zscore",
        "fold_percentile_rank",
    }
    assert set(first_records[0]["meta_fit_metadata_auc_by_mode"]) == {
        "raw_probability",
        "fold_logit_zscore",
        "fold_percentile_rank",
    }
    assert first_final["harmonization_candidate_count"] == 9


def test_siim_four_channel_harmonized_blend_keeps_lesion_outer_labels_isolated():
    truth = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int8)
    folds = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int8)
    oof_channels = {
        "image": np.asarray([0.10, 0.22, 0.30, 0.55, 0.70, 0.91]),
        "lesion": np.asarray([0.08, 0.27, 0.25, 0.62, 0.68, 0.94]),
        "fusion": np.asarray([0.12, 0.30, 0.28, 0.66, 0.64, 0.96]),
        "metadata": np.asarray([0.20, 0.65, 0.35, 0.72, 0.42, 0.81]),
    }
    test_by_fold = {
        name: np.vstack(
            [
                np.clip(values[:3] + offset, 0.01, 0.99)
                for offset in (-0.02, 0.0, 0.02)
            ]
        )
        for name, values in oof_channels.items()
    }
    variants = {
        name: full.recovery.build_siim_fold_harmonization_variants(
            values,
            test_by_fold[name],
            folds,
        )
        for name, values in oof_channels.items()
    }

    first, _, records, final = full.recovery.cross_fit_siim_harmonized_multichannel_blend(
        variants["image"][0],
        variants["fusion"][0],
        variants["metadata"][0],
        variants["image"][1],
        variants["fusion"][1],
        variants["metadata"][1],
        truth,
        folds,
        lesion_oof_variants=variants["lesion"][0],
        lesion_test_variants=variants["lesion"][1],
    )
    mutated = truth.copy()
    mutated[folds == 0] = 1 - mutated[folds == 0]
    changed, _, changed_records, _ = (
        full.recovery.cross_fit_siim_harmonized_multichannel_blend(
            variants["image"][0],
            variants["fusion"][0],
            variants["metadata"][0],
            variants["image"][1],
            variants["fusion"][1],
            variants["metadata"][1],
            mutated,
            folds,
            lesion_oof_variants=variants["lesion"][0],
            lesion_test_variants=variants["lesion"][1],
        )
    )

    np.testing.assert_array_equal(first[folds == 0], changed[folds == 0])
    assert records[0]["weights"] == changed_records[0]["weights"]
    assert records[0]["lesion_harmonization_mode"] == changed_records[0][
        "lesion_harmonization_mode"
    ]
    assert set(records[0]["weights"]) == {
        "pure_image",
        "lesion_focus",
        "image_metadata_fusion",
        "metadata_catboost",
    }
    assert final["harmonization_candidate_count"] == 12


def test_siim_crossfit_records_replay_five_test_streams_without_new_selection():
    modes = {
        "raw_probability": np.asarray([0.1, 0.8]),
        "fold_logit_zscore": np.asarray([0.2, 0.7]),
        "fold_percentile_rank": np.asarray([0.3, 0.6]),
    }
    records = [
        {
            "fold": fold,
            "image_harmonization_mode": "fold_percentile_rank",
            "fusion_harmonization_mode": "fold_percentile_rank",
            "metadata_harmonization_mode": "raw_probability",
            "lesion_harmonization_mode": "fold_logit_zscore",
            "blend_mode": "raw",
            "weights": {
                "pure_image": 0.4,
                "image_metadata_fusion": 0.3,
                "metadata_catboost": 0.1,
                "lesion_focus": 0.2,
            },
        }
        for fold in range(5)
    ]

    result = full.recovery.apply_siim_crossfit_blend_records_to_test(
        records,
        modes,
        modes,
        modes,
        lesion_test_variants=modes,
    )

    expected = (
        0.7 * modes["fold_percentile_rank"]
        + 0.2 * modes["fold_logit_zscore"]
        + 0.1 * modes["raw_probability"]
    )
    assert result.shape == (5, 2)
    assert np.allclose(result, np.stack([expected] * 5))


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("broadcast_shape", "inconsistent shapes"),
        ("negative_weight", "finite simplex"),
        ("non_unit_weight_sum", "finite simplex"),
        ("out_of_range_probability", "closed-unit probabilities"),
        ("extra_weight_channel", "differ from replay channels"),
    ],
)
def test_siim_crossfit_record_replay_rejects_malformed_frozen_inputs(
    failure: str,
    message: str,
):
    def variants() -> dict[str, np.ndarray]:
        return {
            "raw_probability": np.asarray([0.1, 0.8]),
            "fold_logit_zscore": np.asarray([0.2, 0.7]),
            "fold_percentile_rank": np.asarray([0.3, 0.6]),
        }

    image = variants()
    fusion = variants()
    metadata = variants()
    lesion = variants()
    records = [
        {
            "fold": fold,
            "image_harmonization_mode": "fold_percentile_rank",
            "fusion_harmonization_mode": "fold_percentile_rank",
            "metadata_harmonization_mode": "raw_probability",
            "lesion_harmonization_mode": "fold_logit_zscore",
            "blend_mode": "raw",
            "weights": {
                "pure_image": 0.4,
                "image_metadata_fusion": 0.3,
                "metadata_catboost": 0.1,
                "lesion_focus": 0.2,
            },
        }
        for fold in range(full.recovery.SIIM_FORMAL_OUTER_FOLDS)
    ]
    if failure == "broadcast_shape":
        image["fold_percentile_rank"] = np.asarray([0.3])
    elif failure == "negative_weight":
        records[0]["weights"]["pure_image"] = -0.1
        records[0]["weights"]["image_metadata_fusion"] = 0.8
    elif failure == "non_unit_weight_sum":
        records[0]["weights"]["pure_image"] = 0.5
    elif failure == "out_of_range_probability":
        metadata["raw_probability"] = np.asarray([0.1, 1.2])
    elif failure == "extra_weight_channel":
        records[0]["weights"]["unexpected"] = 0.0

    with pytest.raises(RuntimeError, match=message):
        full.recovery.apply_siim_crossfit_blend_records_to_test(
            records,
            image,
            fusion,
            metadata,
            lesion_test_variants=lesion,
        )


def test_recovery_gpu_runners_expose_a40_throughput_contracts():
    import inspect

    recovery_source = inspect.getsource(full.recovery)
    denoising_train = inspect.getsource(full.recovery._train_denoising_model)
    denoising_restore = inspect.getsource(full.recovery._restore_image)
    siim_runner = inspect.getsource(full.recovery.run_siim_image_metadata)

    assert ".cuda(memory_format=" not in recovery_source
    assert 'model.to(device="cuda", memory_format=torch.channels_last)' in recovery_source
    for source in (denoising_train, denoising_restore, siim_runner):
        assert "torch.bfloat16" in source
        assert "memory_format=torch.channels_last" in source
    assert "fused=True" in denoising_train
    assert "prefetch_factor" in denoising_train
    assert '"train_patches_per_second"' in denoising_train
    assert "fast_kernel_mode=bool(args.wave2_fast_kernels)" in siim_runner
    assert "fused=True" in siim_runner
    assert "prefetch_factor" in siim_runner
    assert "build_siim_image_content_manifest" in siim_runner
    assert "build_siim_content_connected_groups" in siim_runner
    assert "select_siim_channel_epochs" in siim_runner
    assert "capture_epochs=set(selected_channel_epochs.values())" in siim_runner
    assert "write_siim_artifact_manifest" in siim_runner
    assert "build_siim_fold_harmonization_variants" in siim_runner
    assert "select_siim_crossfit_harmonization" in siim_runner
    assert "cross_fit_siim_harmonized_multichannel_blend" in siim_runner
    assert "metadata_test_by_fold" in siim_runner
    assert "metadata_catboost_test_by_fold=metadata_test_by_fold" in siim_runner
    assert "build_siim_multiview_fusion_model" in siim_runner
    assert "full_backbone_name=args.siim_backbone" in siim_runner
    assert "lesion_backbone_name=args.siim_secondary_backbone" in siim_runner
    assert "prepare_siim_dermoscopy_views" in siim_runner
    assert "apply_siim_paired_transform" in siim_runner
    assert "gradient_accumulation_steps" in siim_runner
    assert '"paired_multiview_transform": True' in siim_runner
    assert "SiimPauseRequested" in siim_runner
    assert '"signals_sent": 0' in siim_runner
    assert "lesion_test_by_fold" in siim_runner
    assert "lesion_oof_variants=lesion_oof_variants" in siim_runner
    assert '"task_type": metadata_catboost_task_type' in siim_runner
    assert '"gpu_ram_part": 0.2' in siim_runner
    assert "_open_siim_rgb_for_transform" in siim_runner
    assert '"jpeg_dct_scaled_decode": True' in siim_runner
    assert "model, selection_loader, tta=False" in siim_runner
    assert "outer_valid_loader" in siim_runner
    assert "test_loader" in siim_runner
    assert "required_channels=required_channels" in siim_runner
    assert '"train_images_per_second"' in siim_runner


def test_siim_split_keeps_patient_groups_disjoint():
    patients = [f"p{index}" for index in range(20) for _ in range(2)]
    metadata = pd.DataFrame({
        "image_name": [f"image_{index}" for index in range(len(patients))],
        "patient_id": patients,
    })
    target = np.array([index % 2 for index in range(20) for _ in range(2)])
    train_indices, valid_indices, strategy = full.recovery._siim_split_indices(
        metadata, target, holdout_fraction=0.2, seed=42
    )
    train_patients = set(metadata.iloc[train_indices]["patient_id"])
    valid_patients = set(metadata.iloc[valid_indices]["patient_id"])
    assert not train_patients & valid_patients
    assert strategy.startswith("stratified_group_")


def test_siim_patient_folds_cover_oof_and_crossfit_blend():
    patients = [f"p{index}" for index in range(30) for _ in range(2)]
    metadata = pd.DataFrame({
        "image_name": [f"image_{index}" for index in range(len(patients))],
        "patient_id": patients,
    })
    target = np.asarray([index % 2 for index in range(30) for _ in range(2)])
    splits, groups, strategy = full.recovery.build_siim_patient_folds(
        metadata,
        target,
        requested_folds=5,
        seed=42,
    )
    assert strategy == "stratified_patient_group_5_fold"
    coverage = np.zeros(len(metadata), dtype=np.int8)
    fold_assignment = np.full(len(metadata), -1, dtype=np.int8)
    for fold, (fit_index, valid_index) in enumerate(splits):
        coverage[valid_index] += 1
        fold_assignment[valid_index] = fold
        assert set(groups[fit_index]).isdisjoint(set(groups[valid_index]))
        assert set(target[valid_index].tolist()) == {0, 1}
    assert np.all(coverage == 1)

    image = np.where(target == 1, 0.8, 0.2).astype(float)
    metadata_probability = np.where(target == 1, 0.7, 0.3).astype(float)
    prediction, records = full.recovery.cross_fit_siim_auc_blend(
        image,
        metadata_probability,
        target,
        fold_assignment,
    )
    assert len(records) == 5
    assert np.isfinite(prediction).all()
    assert all(record["outer_auc"] == pytest.approx(1.0) for record in records)


def test_siim_nested_patient_folds_are_disjoint_and_refit_complete():
    patients = [f"p{index}" for index in range(60) for _ in range(2)]
    metadata = pd.DataFrame({
        "image_name": [f"image_{index}" for index in range(len(patients))],
        "patient_id": patients,
    })
    target = np.asarray([index % 2 for index in range(60) for _ in range(2)])
    outer_splits, outer_groups, _ = full.recovery.build_siim_patient_folds(
        metadata,
        target,
        requested_folds=5,
        seed=42,
    )
    plans, nested_groups = full.recovery.build_siim_nested_patient_folds(
        metadata,
        target,
        outer_splits,
        requested_inner_folds=5,
        seed=42,
    )
    assert np.array_equal(outer_groups, nested_groups)
    assert len(plans) == 5
    for plan in plans:
        outer_fit = plan["outer_fit_index"]
        outer_valid = plan["outer_valid_index"]
        refit = plan["refit_index"]
        assert set(nested_groups[outer_fit]).isdisjoint(
            set(nested_groups[outer_valid])
        )
        assert len(plan["inner_folds"]) == 5
        inner_coverage = np.zeros(len(metadata), dtype=np.int8)
        for inner_plan in plan["inner_folds"]:
            inner_fit = inner_plan["inner_fit_index"]
            inner_valid = inner_plan["inner_valid_index"]
            inner_coverage[inner_valid] += 1
            assert set(nested_groups[inner_fit]).isdisjoint(
                set(nested_groups[inner_valid])
            )
            assert np.array_equal(
                np.sort(np.concatenate([inner_fit, inner_valid])), np.sort(outer_fit)
            )
        assert np.all(inner_coverage[outer_fit] == 1)
        assert np.array_equal(np.sort(refit), np.sort(outer_fit))


def test_siim_nested_folds_honor_supplied_content_connected_groups():
    patients = [f"p{index}" for index in range(60) for _ in range(2)]
    metadata = pd.DataFrame({
        "image_name": [f"image_{index}" for index in range(len(patients))],
        "patient_id": patients,
    })
    target = np.asarray([index % 2 for index in range(60) for _ in range(2)])
    connected = np.asarray([
        "connected_even" if index in {0, 2} else f"g{index}"
        for index in range(60)
        for _ in range(2)
    ])
    outer_splits, outer_groups, strategy = full.recovery.build_siim_patient_folds(
        metadata,
        target,
        requested_folds=5,
        seed=42,
        group_values=connected,
    )
    plans, nested_groups = full.recovery.build_siim_nested_patient_folds(
        metadata,
        target,
        outer_splits,
        requested_inner_folds=5,
        seed=42,
        group_values=connected,
    )

    assert strategy == "stratified_patient_content_connected_5_fold"
    assert np.array_equal(outer_groups, nested_groups)
    for plan in plans:
        assert set(nested_groups[plan["outer_fit_index"]]).isdisjoint(
            set(nested_groups[plan["outer_valid_index"]])
        )
        for inner_plan in plan["inner_folds"]:
            assert set(nested_groups[inner_plan["inner_fit_index"]]).isdisjoint(
                set(nested_groups[inner_plan["inner_valid_index"]])
            )


def test_siim_nested_selection_does_not_read_fixed_outer_validation_labels():
    patients = [f"p{index}" for index in range(80) for _ in range(2)]
    metadata = pd.DataFrame({
        "image_name": [f"image_{index}" for index in range(len(patients))],
        "patient_id": patients,
    })
    target = np.asarray([index % 2 for index in range(80) for _ in range(2)])
    outer_splits, _, _ = full.recovery.build_siim_patient_folds(
        metadata,
        target,
        requested_folds=5,
        seed=19,
    )
    original, _ = full.recovery.build_siim_nested_patient_folds(
        metadata,
        target,
        outer_splits,
        requested_inner_folds=5,
        seed=19,
    )
    mutated = target.copy()
    mutated[outer_splits[0][1]] = 1 - mutated[outer_splits[0][1]]
    changed, _ = full.recovery.build_siim_nested_patient_folds(
        metadata,
        mutated,
        outer_splits,
        requested_inner_folds=5,
        seed=19,
    )
    assert len(original[0]["inner_folds"]) == len(changed[0]["inner_folds"]) == 5
    for original_inner, changed_inner in zip(
        original[0]["inner_folds"], changed[0]["inner_folds"], strict=True
    ):
        assert np.array_equal(
            original_inner["inner_fit_index"], changed_inner["inner_fit_index"]
        )
        assert np.array_equal(
            original_inner["inner_valid_index"], changed_inner["inner_valid_index"]
        )


def test_siim_inner_budgets_are_fixed_without_outer_oof_labels():
    history = [
        {
            "epoch": 1,
            "pure_image_inner_validation_auc": 0.88,
            "fusion_inner_validation_auc": 0.81,
        },
        {
            "epoch": 2,
            "pure_image_inner_validation_auc": 0.87,
            "fusion_inner_validation_auc": 0.86,
        },
        {
            "epoch": 3,
            "pure_image_inner_validation_auc": 0.86,
            "fusion_inner_validation_auc": 0.86,
        },
    ]
    outer_truth = np.asarray([0, 1, 0, 1])
    selected_before = full.recovery.select_siim_inner_epoch(history)
    iterations_before = full.recovery.select_siim_metadata_refit_iterations(
        67,
        requested_iterations=700,
    )
    outer_truth[:] = 1 - outer_truth
    selected_after = full.recovery.select_siim_inner_epoch(history)
    iterations_after = full.recovery.select_siim_metadata_refit_iterations(
        67,
        requested_iterations=700,
    )
    assert selected_before == selected_after == 2
    assert full.recovery.select_siim_channel_epochs(history) == {
        "pure_image": 1,
        "image_metadata_fusion": 2,
    }
    assert iterations_before == iterations_after == 68
    assert full.recovery.select_siim_metadata_refit_iterations(
        -1,
        requested_iterations=700,
    ) == 700
    assert full.recovery.select_siim_metadata_refit_iterations(
        0,
        requested_iterations=700,
    ) == 50


def test_siim_epoch_selection_aggregates_every_inner_fold():
    histories = [
        [
            {
                "epoch": 1,
                "full_image_inner_validation_auc": 0.91,
                "lesion_focus_inner_validation_auc": 0.70,
                "pure_image_inner_validation_auc": 0.91,
                "fusion_inner_validation_auc": 0.80,
            },
            {
                "epoch": 2,
                "full_image_inner_validation_auc": 0.86,
                "lesion_focus_inner_validation_auc": 0.78,
                "pure_image_inner_validation_auc": 0.86,
                "fusion_inner_validation_auc": 0.88,
            },
        ],
        [
            {
                "epoch": 1,
                "full_image_inner_validation_auc": 0.60,
                "lesion_focus_inner_validation_auc": 0.74,
                "pure_image_inner_validation_auc": 0.60,
                "fusion_inner_validation_auc": 0.76,
            },
            {
                "epoch": 2,
                "full_image_inner_validation_auc": 0.90,
                "lesion_focus_inner_validation_auc": 0.82,
                "pure_image_inner_validation_auc": 0.90,
                "fusion_inner_validation_auc": 0.84,
            },
        ],
    ]
    aggregated = full.recovery.aggregate_siim_inner_histories(histories)
    assert [record["inner_fold_count"] for record in aggregated] == [2, 2]
    assert aggregated[0]["full_image_inner_validation_auc"] == pytest.approx(0.755)
    assert aggregated[1]["full_image_inner_validation_auc"] == pytest.approx(0.88)
    assert aggregated[0]["fusion_inner_validation_auc_minimum"] == pytest.approx(0.76)
    assert full.recovery.select_siim_channel_epochs(
        aggregated,
        channels=("full_image", "lesion_focus", "image_metadata_fusion"),
    ) == {
        "full_image": 2,
        "lesion_focus": 2,
        "image_metadata_fusion": 2,
    }


def test_siim_inner_history_aggregation_rejects_missing_fold_epoch():
    with pytest.raises(RuntimeError, match="same epochs"):
        full.recovery.aggregate_siim_inner_histories(
            [
                [{"epoch": 1}],
                [{"epoch": 1}, {"epoch": 2}],
            ]
        )


def _siim_seeded_ablation_scores(
    profile_scores: dict[str, list[float]],
    *,
    seeds: tuple[int, ...] = (40, 41, 42),
) -> dict[str, dict[int, list[float]]]:
    seed_offsets = {seed: (index - 1) * 0.0001 for index, seed in enumerate(seeds)}
    return {
        profile: {
            seed: [float(value + seed_offsets[seed]) for value in values]
            for seed in seeds
        }
        for profile, values in profile_scores.items()
    }


def test_siim_preprocessing_ablation_selects_only_contiguous_accepted_steps():
    profile_fold_auc = _siim_seeded_ablation_scores({
        "raw_multiview_v1": [0.800, 0.810, 0.820],
        "border_multiview_v1": [0.802, 0.811, 0.822],
        "color_multiview_v1": [0.803, 0.812, 0.823],
        "hair_multiview_v1": [0.802, 0.811, 0.822],
        "robust_multiview_v1": [0.810, 0.820, 0.830],
    })

    report = full.recovery.select_siim_preprocessing_ablation(
        profile_fold_auc,
        evaluation_seeds=[40, 41, 42],
        image_content_manifest_sha256="a" * 64,
    )

    assert report["profile_order"] == list(full.recovery.SIIM_PREPROCESSING_PROFILES)
    assert [record["profile"] for record in report["profiles"]] == report[
        "profile_order"
    ]
    assert report["fold_count"] == 3
    assert report["evaluation_seed_count"] == 3
    assert report["score_count_per_profile"] == 9
    assert report["selected_profile"] == "color_multiview_v1"
    by_profile = {record["profile"]: record for record in report["profiles"]}
    assert by_profile["border_multiview_v1"]["eligible_for_selection"] is True
    assert by_profile["color_multiview_v1"]["eligible_for_selection"] is True
    assert by_profile["hair_multiview_v1"]["increment_accepted"] is False
    assert by_profile["hair_multiview_v1"]["eligible_for_selection"] is False
    assert by_profile["robust_multiview_v1"]["increment_accepted"] is True
    assert by_profile["robust_multiview_v1"]["eligible_for_selection"] is False
    assert report["private_labels_used_for_training"] is False
    assert report["official_grader_executed"] is False
    assert report["kaggle_submission_executed"] is False


def test_siim_preprocessing_ablation_requires_all_profiles_and_consistent_folds():
    base = {
        profile: [0.80, 0.81, 0.82]
        for profile in full.recovery.SIIM_PREPROCESSING_PROFILES
    }
    complete = _siim_seeded_ablation_scores(base)
    incomplete = dict(complete)
    incomplete.pop("robust_multiview_v1")
    with pytest.raises(RuntimeError, match="evaluate every profile"):
        full.recovery.select_siim_preprocessing_ablation(
            incomplete,
            evaluation_seeds=[40, 41, 42],
            image_content_manifest_sha256="a" * 64,
        )

    with pytest.raises(RuntimeError, match="three distinct external seeds"):
        full.recovery.select_siim_preprocessing_ablation(
            _siim_seeded_ablation_scores(base, seeds=(40, 41)),
            evaluation_seeds=[40, 41],
            image_content_manifest_sha256="a" * 64,
        )

    with pytest.raises(RuntimeError, match="three distinct external seeds"):
        full.recovery.select_siim_preprocessing_ablation(
            complete,
            evaluation_seeds=[40, 40, 42],
            image_content_manifest_sha256="a" * 64,
        )

    mismatched = {profile: dict(values) for profile, values in complete.items()}
    mismatched["raw_multiview_v1"].pop(42)
    with pytest.raises(RuntimeError, match="seed sets differ"):
        full.recovery.select_siim_preprocessing_ablation(
            mismatched,
            evaluation_seeds=[40, 41, 42],
            image_content_manifest_sha256="a" * 64,
        )

    two_fold = {
        profile: {seed: values[:2] for seed, values in seed_values.items()}
        for profile, seed_values in complete.items()
    }
    with pytest.raises(RuntimeError, match="fold AUC arrays are invalid"):
        full.recovery.select_siim_preprocessing_ablation(
            two_fold,
            evaluation_seeds=[40, 41, 42],
            image_content_manifest_sha256="a" * 64,
        )

    one_fold = {
        profile: {seed: values[:1] for seed, values in seed_values.items()}
        for profile, seed_values in complete.items()
    }
    with pytest.raises(RuntimeError, match="fold AUC arrays are invalid"):
        full.recovery.select_siim_preprocessing_ablation(
            one_fold,
            evaluation_seeds=[40, 41, 42],
            image_content_manifest_sha256="a" * 64,
        )

    non_finite = json.loads(json.dumps(complete))
    non_finite["raw_multiview_v1"]["40"][0] = float("nan")
    with pytest.raises(RuntimeError, match="fold AUC arrays are invalid"):
        full.recovery.select_siim_preprocessing_ablation(
            non_finite,
            evaluation_seeds=[40, 41, 42],
            image_content_manifest_sha256="a" * 64,
        )


def test_siim_two_fold_selector_report_is_rejected_by_formal_validator(tmp_path):
    profile_fold_auc = _siim_seeded_ablation_scores({
        profile: [
            0.80 + index * 0.002,
            0.81 + index * 0.002,
        ]
        for index, profile in enumerate(full.recovery.SIIM_PREPROCESSING_PROFILES)
    })
    manifest_sha256 = "b" * 64
    three_fold_scores = {
        profile: {
            seed: [*values, values[-1] + 0.001]
            for seed, values in seed_values.items()
        }
        for profile, seed_values in profile_fold_auc.items()
    }
    report = full.recovery.select_siim_preprocessing_ablation(
        three_fold_scores,
        evaluation_seeds=[40, 41, 42],
        image_content_manifest_sha256=manifest_sha256,
    )
    report["fold_count"] = 2
    report["score_count_per_profile"] = 6
    for record in report["profiles"]:
        record["seed_fold_auc"] = {
            seed: values[:2] for seed, values in record["seed_fold_auc"].items()
        }
    report["leakage_group_policy"] = full.recovery.SIIM_LEAKAGE_GROUP_POLICY
    report["perceptual_edge_policy"] = full.recovery.SIIM_PERCEPTUAL_EDGE_POLICY

    assert full.recovery.MIN_SIIM_ABLATION_FOLDS == 3
    assert report["fold_count"] == 2
    assert report["evaluation_seed_count"] == 3
    assert report["score_count_per_profile"] == 6
    report_path = tmp_path / "siim_preprocessing_ablation_two_fold.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    validated = full.recovery.validate_siim_preprocessing_ablation_report(
        report_path,
        expected_profile="robust_multiview_v1",
        expected_image_content_manifest_sha256=manifest_sha256,
        formal_seed=43,
    )

    assert validated["validated"] is False
    assert validated["checks"]["fold_count_exact"] is False
    assert validated["checks"]["three_distinct_external_seeds"] is True
    assert validated["checks"]["per_seed_fold_arrays_complete"] is False

    mismatched = json.loads(json.dumps(report))
    mismatched["profiles"][-1]["seed_fold_auc"]["42"].append(0.82)
    mismatched_path = tmp_path / "siim_preprocessing_ablation_mismatched_folds.json"
    mismatched_path.write_text(json.dumps(mismatched), encoding="utf-8")
    rejected = full.recovery.validate_siim_preprocessing_ablation_report(
        mismatched_path,
        expected_profile="robust_multiview_v1",
        expected_image_content_manifest_sha256=manifest_sha256,
        formal_seed=43,
    )
    assert rejected["validated"] is False
    assert rejected["checks"]["per_seed_fold_arrays_complete"] is False


def test_siim_preprocessing_ablation_report_validation_contract(tmp_path):
    profile_fold_auc = _siim_seeded_ablation_scores({
        profile: [
            0.80 + index * 0.002,
            0.81 + index * 0.002,
            0.82 + index * 0.002,
        ]
        for index, profile in enumerate(full.recovery.SIIM_PREPROCESSING_PROFILES)
    })
    manifest_sha256 = "b" * 64
    report = full.recovery.select_siim_preprocessing_ablation(
        profile_fold_auc,
        evaluation_seeds=[40, 41, 42],
        image_content_manifest_sha256=manifest_sha256,
    )
    report["leakage_group_policy"] = full.recovery.SIIM_LEAKAGE_GROUP_POLICY
    report["perceptual_edge_policy"] = full.recovery.SIIM_PERCEPTUAL_EDGE_POLICY
    report_path = tmp_path / "siim_preprocessing_ablation.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    validated = full.recovery.validate_siim_preprocessing_ablation_report(
        report_path,
        expected_profile="robust_multiview_v1",
        expected_image_content_manifest_sha256=manifest_sha256,
        formal_seed=43,
    )
    assert validated["validated"] is True
    assert all(validated["checks"].values())

    wrong_manifest = full.recovery.validate_siim_preprocessing_ablation_report(
        report_path,
        expected_profile="robust_multiview_v1",
        expected_image_content_manifest_sha256="c" * 64,
        formal_seed=43,
    )
    assert wrong_manifest["validated"] is False
    assert wrong_manifest["checks"]["image_manifest_matches"] is False

    reused_seed = full.recovery.validate_siim_preprocessing_ablation_report(
        report_path,
        expected_profile="robust_multiview_v1",
        expected_image_content_manifest_sha256=manifest_sha256,
        formal_seed=42,
    )
    assert reused_seed["validated"] is False
    assert reused_seed["checks"]["formal_seed_not_used_for_ablation"] is False

    legacy_report = dict(report)
    legacy_report["evaluation_seeds"] = [42]
    legacy_report["evaluation_seed_count"] = 1
    legacy_report["score_count_per_profile"] = 3
    legacy_report["profiles"] = [
        {
            **record,
            "seed_fold_auc": {"42": record["seed_fold_auc"]["42"]},
        }
        for record in report["profiles"]
    ]
    legacy_path = tmp_path / "siim_preprocessing_ablation_legacy_single_seed.json"
    legacy_path.write_text(json.dumps(legacy_report), encoding="utf-8")
    legacy_validation = full.recovery.validate_siim_preprocessing_ablation_report(
        legacy_path,
        expected_profile="robust_multiview_v1",
        expected_image_content_manifest_sha256=manifest_sha256,
        formal_seed=43,
    )
    assert legacy_validation["validated"] is False
    assert legacy_validation["checks"]["three_distinct_external_seeds"] is False

    missing = full.recovery.validate_siim_preprocessing_ablation_report(
        None,
        expected_profile="robust_multiview_v1",
        expected_image_content_manifest_sha256=manifest_sha256,
        formal_seed=43,
    )
    assert missing == {
        "validated": False,
        "reason": "report_not_configured",
        "selected_profile": "robust_multiview_v1",
        "checks": {"report_configured": False},
    }


def test_siim_runner_uses_inner_selection_and_untouched_outer_oof_contract():
    import inspect

    source = inspect.getsource(full.recovery.run_siim_image_metadata)
    nested_source = inspect.getsource(full.recovery.build_siim_nested_patient_folds)
    assert "selection_loader=inner_valid_loader" in source
    assert "expected_selection_truth=target[inner_valid_index]" in source
    assert "train_metadata[inner_valid_index]" in source
    assert "train_metadata[refit_index]" in source
    assert "target[refit_index]" in source
    assert "outer_valid_loader" in source
    assert "tta=True" in source
    assert "required_channels=required_channels" in source
    assert '"outer_validation_role": "final_oof_only"' in source
    assert '"fixed_epoch_outer_refit": True' in source
    assert '"channel_specific_epoch_selection": True' in source
    assert '"fixed_iteration_metadata_outer_refit": True' in source
    assert '"exact_content_duplicate_grouped": True' in source
    assert '"decoded_pixel_duplicate_grouped": True' in source
    assert '"perceptual_duplicate_candidates_audited": True' in source
    assert '"perceptual_candidates_grouped": False' in source
    assert '"fold_scale_harmonization_cross_fitted": True' in source
    assert '"metadata_test_predictions_by_fold": True' in source
    assert '"artifact_manifest_sha256"' in source
    assert "aggregate_siim_inner_histories(inner_selection_histories)" in source
    assert '"all_inner_folds_aggregated": True' in source
    assert "profile=preprocessing_profile" in source
    assert "validate_siim_preprocessing_ablation_report" in source
    assert "require_requested_folds=True" in source
    assert "require_requested_inner_folds=True" in source
    assert 'task_dir / "siim_resume_contract.json"' in source
    assert '"preprocessing_ablation_validated"' in source
    assert 'task_dir / "siim_preprocessing_ablation_gate.json"' in source
    gate_position = source.index('if not preprocessing_ablation_gate["validated"]')
    assert gate_position < source.index("wave2._image_transforms")
    assert '"inner_folds": inner_fold_records' in nested_source
    assert "selected_inner_fold" not in nested_source
    assert "selection_loader=outer_valid_loader" not in source
    assert "eval_set=(train_metadata[valid_index]" not in source


def test_siim_multiview_runtime_contract_wires_dataset_and_lesion_blend_correctly():
    import ast
    import inspect
    import textwrap

    source = inspect.getsource(full.recovery.run_siim_image_metadata)
    legacy_source = inspect.getsource(
        full.recovery._run_siim_image_metadata_single_holdout_legacy
    )
    assert "prepare_siim_dermoscopy_views" in source
    assert "return full_image, lesion_image, metadata_tensor" in source
    assert "for batch_index, (full_images, lesion_images, metadata, labels) in enumerate" in source
    assert "apply_siim_paired_transform" in source
    assert "stage_id=f\"outer{fold:02d}_inner{inner_fold:02d}_selection\"" in source
    assert "stage_id=f\"outer{fold:02d}_refit\"" in source
    assert "prepare_siim_dermoscopy_views" not in legacy_source
    assert "return image, metadata_tensor" in legacy_source

    tree = ast.parse(textwrap.dedent(source))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    harmonization_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "select_siim_crossfit_harmonization"
    ]
    assert len(harmonization_calls) == 4
    assert all(not node.keywords for node in harmonization_calls)

    blend_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "cross_fit_siim_harmonized_multichannel_blend"
    ]
    assert len(blend_calls) == 1
    assert {keyword.arg for keyword in blend_calls[0].keywords} == {
        "lesion_oof_variants",
        "lesion_test_variants",
    }


def test_binary_logit_calibration_is_finite_and_not_worse():
    from scipy.special import expit
    from sklearn.metrics import log_loss

    truth = np.array([0, 0, 0, 1, 1, 1])
    logits = np.array([-8.0, -3.0, 1.0, -1.0, 3.0, 8.0])
    uncalibrated = log_loss(truth, expit(logits))
    probability, temperature, intercept, calibrated = full.recovery.calibrate_binary_logits(
        logits, truth
    )
    assert calibrated <= uncalibrated + 1e-10
    assert temperature > 0
    assert np.isfinite(intercept)
    assert np.all((probability > 0.0) & (probability < 1.0))
    reapplied = full.recovery.apply_binary_logit_calibration(
        logits, temperature=temperature, intercept=intercept
    )
    assert np.allclose(probability, reapplied)


def test_multiclass_blend_is_normalized_and_selects_signal():
    classes = ["a", "b", "c"]
    truth = np.asarray(["a", "b", "c", "a", "b", "c"])
    strong = np.asarray(
        [
            [0.90, 0.05, 0.05],
            [0.05, 0.90, 0.05],
            [0.05, 0.05, 0.90],
            [0.80, 0.10, 0.10],
            [0.10, 0.80, 0.10],
            [0.10, 0.10, 0.80],
        ]
    )
    weak = np.full_like(strong, 1.0 / 3.0)
    noisy = strong[:, ::-1]
    probability, weights, temperature, score = full.recovery.select_multiclass_logloss_blend(
        [strong, weak, noisy], truth, classes
    )
    assert probability.shape == strong.shape
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert np.isfinite(probability).all()
    assert len(weights) == 3 and sum(weights) == pytest.approx(1.0)
    assert temperature > 0 and score >= 0
    assert weights[0] > weights[2]
    crossfit, records = full.recovery.cross_fit_multiclass_logloss_blend(
        [strong, weak, noisy],
        truth,
        classes,
        np.asarray([0, 0, 0, 1, 1, 1]),
    )
    assert len(records) == 2
    assert np.isfinite(crossfit).all()
    assert np.allclose(crossfit.sum(axis=1), 1.0)


def test_pizza_features_use_only_shared_request_time_fields():
    frame = pd.DataFrame({
        "request_title": ["Need dinner"],
        "request_text_edit_aware": ["shared body"],
        "request_text": ["TRAIN_ONLY_LEAK"],
        "requester_subreddits_at_request": [["food", "help"]],
        "requester_account_age_in_days_at_request": [10.0],
        "requester_upvotes_plus_downvotes_at_request": [12],
        "unix_timestamp_of_request_utc": [1_400_000_000],
        "giver_username_if_known": ["N/A"],
    })
    text = full._pizza_text(frame).iloc[0]
    assert "shared body" in text and "food help" in text
    assert "TRAIN_ONLY_LEAK" not in text
    features = full.pizza_structured_features(frame)
    assert features.shape[1] >= 20
    assert np.isfinite(features.to_numpy()).all()


def test_pizza_nbsvm_fold_is_finite_and_persists_fold_local_state():
    fit_text = pd.Series([
        "please help pizza hungry",
        "please help pizza family",
        "kind person pizza hungry",
        "kind person pizza family",
        "ordinary story update later",
        "ordinary story update today",
        "general post update later",
        "general post update today",
    ])
    labels = pd.Series([1, 1, 1, 1, 0, 0, 0, 0])
    valid, test, artifacts = full.fit_pizza_nbsvm_fold(
        fit_text,
        labels,
        pd.Series(["please pizza hungry", "ordinary update later"]),
        pd.Series(["kind pizza family", "general post today"]),
        word_features=1_000,
        char_features=1_000,
        c_value=1.0,
        seed=42,
    )
    assert valid.shape == (2,)
    assert test.shape == (2,)
    assert np.isfinite(valid).all() and np.isfinite(test).all()
    assert artifacts["word_features"] > 0 and artifacts["char_features"] > 0
    assert {"word_vectorizer", "char_vectorizer", "word_ratio", "char_ratio"} <= set(artifacts)


def test_binary_auc_blend_is_bounded_and_selects_signal():
    labels = pd.Series([0, 0, 1, 1])
    text = np.array([0.1, 0.2, 0.8, 0.9])
    structured = np.array([0.4, 0.3, 0.7, 0.6])
    prediction, weight, mode, score = full.select_binary_auc_blend(text, structured, labels)
    assert score == pytest.approx(1.0)
    assert 0.0 <= weight <= 1.0
    assert mode in {"raw", "rank"}
    assert np.all((prediction > 0) & (prediction < 1))


def test_binary_auc_blend_crossfit_covers_every_outer_fold():
    labels = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    text = np.array([0.1, 0.9, 0.2, 0.8, 0.15, 0.85, 0.25, 0.75])
    structured = np.array([0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.35, 0.65])
    folds = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    prediction, records = full.cross_fit_binary_auc_blend(
        text, structured, labels, folds
    )
    assert len(records) == 2
    assert {record["fold"] for record in records} == {0, 1}
    assert np.isfinite(prediction).all()
    assert np.all((prediction > 0) & (prediction < 1))
    assert all(record["outer_auc"] == pytest.approx(1.0) for record in records)


@pytest.mark.parametrize(
    ("text", "folds", "message"),
    [
        (
            np.array([0.1, 0.9, 0.2, 0.8]),
            np.array([-1, 0, 1, 1]),
            "complete nonnegative fold assignments",
        ),
        (
            np.array([0.1, np.nan, 0.2, 0.8]),
            np.array([0, 0, 1, 1]),
            "non-finite",
        ),
    ],
)
def test_binary_auc_blend_crossfit_fails_closed_on_invalid_inputs(text, folds, message):
    labels = pd.Series([0, 1, 0, 1])
    structured = np.array([0.2, 0.8, 0.3, 0.7])
    with pytest.raises(RuntimeError, match=message):
        full.cross_fit_binary_auc_blend(text, structured, labels, folds)


def test_spooky_multiclass_blend_is_normalized_and_finite():
    classes = ["EAP", "HPL", "MWS"]
    labels = pd.Series(["EAP", "HPL", "MWS"])
    word = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    char = np.array([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
    prediction, weight, temperature, score = wave0.select_multiclass_blend(
        word, char, labels, classes
    )
    assert prediction.shape == (3, 3)
    assert np.allclose(prediction.sum(axis=1), 1.0)
    assert 0.2 <= weight <= 0.8
    assert temperature > 0
    assert np.isfinite(score)


def test_wave_plan_json_extraction_and_exact_set_validation():
    order = list(reversed(full.WAVE1_COMPETITIONS))
    payload = extract_json("```json\n" + json.dumps({"competition_order": order}) + "\n```")
    assert validate_order(payload["competition_order"]) == order
    with pytest.raises(ValueError, match="exactly once"):
        validate_order(order[:-1])


def test_wave_plan_uses_remaining_competitions_from_progress_report(tmp_path):
    path = tmp_path / "progress.json"
    remaining = ["aptos2019-blindness-detection", "jigsaw-toxic-comment-classification-challenge"]
    path.write_text(json.dumps({
        "schema": "evomind.mlebench_lite.progress.v1",
        "remaining_competition_ids": remaining,
    }), encoding="utf-8")
    selected = competitions_from_progress_report(path)
    assert selected == tuple(remaining)
    assert validate_order(list(reversed(remaining)), selected) == list(reversed(remaining))
