#!/usr/bin/env python3
"""Evaluate one bounded fare-weighted Taxi mid-haul expert on CPU."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from diagnose_taxi_cpu_longhaul_expert import (
        CLIP_LOWER,
        CLIP_UPPER,
        TaxiLonghaulDiagnosticError,
        file_record,
        longhaul_mask,
        now_iso,
        rmse,
        validate_inputs,
        write_json_atomic,
    )
except ModuleNotFoundError:
    from scripts.diagnose_taxi_cpu_longhaul_expert import (
        CLIP_LOWER,
        CLIP_UPPER,
        TaxiLonghaulDiagnosticError,
        file_record,
        longhaul_mask,
        now_iso,
        rmse,
        validate_inputs,
        write_json_atomic,
    )

RESULT_SCHEMA = "evomind.mlebench.taxi_cpu_weighted_midhaul_expert_diagnostic.v1"


def fare_weights(
    target: np.ndarray,
    *,
    pivot_fare: float,
    maximum_weight: float,
) -> np.ndarray:
    if pivot_fare <= 0 or maximum_weight < 1:
        raise TaxiLonghaulDiagnosticError("Fare-weight parameters are invalid")
    values = np.asarray(target, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise TaxiLonghaulDiagnosticError("Fare-weight target is invalid")
    result = np.clip(values / pivot_fare, 1.0, maximum_weight)
    return np.asarray(result, dtype=np.float32)


def train_weighted_expert(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    fit_weight: np.ndarray,
    *,
    seed: int,
    iterations: int,
    threads: int,
) -> Any:
    import lightgbm as lgb

    params = {
        "objective": "regression_l2",
        "metric": "rmse",
        "learning_rate": 0.035,
        "num_leaves": 255,
        "max_depth": 16,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.95,
        "bagging_fraction": 0.95,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "max_bin": 511,
        "num_threads": threads,
        "seed": seed,
        "feature_fraction_seed": seed + 1000,
        "bagging_seed": seed + 2000,
        "data_random_seed": seed + 3000,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    dataset = lgb.Dataset(
        fit_x,
        label=fit_y,
        weight=fit_weight,
        free_raw_data=True,
    )
    return lgb.train(
        params,
        dataset,
        num_boost_round=iterations,
        callbacks=[lgb.log_evaluation(100)],
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cache_dir = Path(args.cache_dir).resolve()
    candidate_dir = Path(args.candidate_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise TaxiLonghaulDiagnosticError("Output directory must be new or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache, base_result, bundle_path = validate_inputs(cache_dir, candidate_dir)

    train_features = np.load(cache_dir / "train_features.npy", mmap_mode="r")
    feature_names = list(cache.get("feature_names") or [])
    target = np.asarray(
        np.load(cache_dir / "target.npy", mmap_mode="r"), dtype=np.float64
    )
    cache_folds = np.asarray(
        np.load(cache_dir / "fold_assignment.npy", mmap_mode="r"), dtype=np.int16
    )
    with np.load(bundle_path, allow_pickle=False) as archive:
        base_oof = np.asarray(archive["oof_prediction"], dtype=np.float64)
        bundle_truth = np.asarray(archive["truth"], dtype=np.float64)
        bundle_folds = np.asarray(archive["fold_assignment"], dtype=np.int16)
    if train_features.shape != (5_000_000, len(feature_names)):
        raise TaxiLonghaulDiagnosticError("Taxi train feature shape changed")
    if not np.array_equal(cache_folds, bundle_folds):
        raise TaxiLonghaulDiagnosticError("Taxi candidate fold assignment drifted")
    if not np.array_equal(target.astype(np.float32), bundle_truth.astype(np.float32)):
        raise TaxiLonghaulDiagnosticError("Taxi candidate truth drifted")

    selected = longhaul_mask(
        train_features,
        feature_names,
        distance_threshold_km=args.distance_threshold_km,
    )
    all_weights = fare_weights(
        target,
        pivot_fare=args.weight_pivot_fare,
        maximum_weight=args.maximum_fare_weight,
    )
    expert_oof = np.full(len(target), np.nan, dtype=np.float64)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    fold_records: list[dict[str, Any]] = []
    for fold in sorted(int(value) for value in np.unique(cache_folds)):
        fit = np.flatnonzero((cache_folds != fold) & selected)
        valid = np.flatnonzero((cache_folds == fold) & selected)
        fit_x = np.asarray(train_features[fit], dtype=np.float32)
        valid_x = np.asarray(train_features[valid], dtype=np.float32)
        model = train_weighted_expert(
            fit_x,
            target[fit],
            all_weights[fit],
            seed=args.seed + fold,
            iterations=args.iterations,
            threads=args.threads,
        )
        prediction = np.asarray(
            model.predict(valid_x, num_iteration=args.iterations),
            dtype=np.float64,
        )
        prediction = np.clip(prediction, CLIP_LOWER, CLIP_UPPER)
        expert_oof[valid] = prediction
        model_path = model_dir / f"fold_{fold:02d}.txt"
        model.save_model(str(model_path), num_iteration=args.iterations)
        fold_records.append(
            {
                "fold": fold,
                "fit_weighted_rows": len(fit),
                "valid_weighted_rows": len(valid),
                "base_midhaul_rmse": rmse(target[valid], base_oof[valid]),
                "weighted_expert_midhaul_rmse": rmse(target[valid], prediction),
                "fit_weight_mean": float(all_weights[fit].mean()),
                "fit_weight_maximum": float(all_weights[fit].max()),
                "model": file_record(model_path),
            }
        )
        del fit_x, valid_x, model
    if not np.isfinite(expert_oof[selected]).all():
        raise TaxiLonghaulDiagnosticError("Weighted expert OOF coverage is incomplete")
    successor = base_oof.copy()
    successor[selected] = expert_oof[selected]
    successor = np.clip(successor, CLIP_LOWER, CLIP_UPPER)
    base_rmse = rmse(target, base_oof)
    successor_rmse = rmse(target, successor)
    base_midhaul_rmse = rmse(target[selected], base_oof[selected])
    expert_midhaul_rmse = rmse(target[selected], expert_oof[selected])
    high_fare = selected & (target > args.high_fare_threshold)
    base_high_fare_rmse = rmse(target[high_fare], base_oof[high_fare])
    expert_high_fare_rmse = rmse(target[high_fare], expert_oof[high_fare])
    passed = (
        successor_rmse <= args.maximum_successor_rmse
        and successor_rmse <= base_rmse - args.minimum_overall_improvement
        and expert_high_fare_rmse
        <= base_high_fare_rmse
        * (1.0 - args.minimum_high_fare_relative_improvement)
    )
    bundle_output = output_dir / "weighted_midhaul_expert_oof.npz"
    np.savez_compressed(
        bundle_output,
        truth=target.astype(np.float32),
        base_oof_prediction=base_oof.astype(np.float32),
        expert_oof_prediction=expert_oof.astype(np.float32),
        successor_oof_prediction=successor.astype(np.float32),
        midhaul_mask=selected.astype(np.uint8),
        fold_assignment=cache_folds,
    )
    report = {
        "schema": RESULT_SCHEMA,
        "created_at": now_iso(),
        "status": (
            "diagnostic_improvement_confirmed"
            if passed
            else "diagnostic_improvement_insufficient_stop"
        ),
        "passed": passed,
        "competition_id": "new-york-city-taxi-fare-prediction",
        "model_family": "LightGBM_CPU_fare_weighted_midhaul_expert",
        "seed": args.seed,
        "folds": int(cache["folds"]),
        "iterations": args.iterations,
        "threads": args.threads,
        "distance_threshold_km": args.distance_threshold_km,
        "weight_pivot_fare": args.weight_pivot_fare,
        "maximum_fare_weight": args.maximum_fare_weight,
        "high_fare_threshold": args.high_fare_threshold,
        "midhaul_rows": int(selected.sum()),
        "metrics": {
            "base_random_oof_rmse": base_rmse,
            "successor_random_oof_rmse": successor_rmse,
            "overall_absolute_improvement": base_rmse - successor_rmse,
            "base_midhaul_rmse": base_midhaul_rmse,
            "expert_midhaul_rmse": expert_midhaul_rmse,
            "base_high_fare_rmse": base_high_fare_rmse,
            "expert_high_fare_rmse": expert_high_fare_rmse,
        },
        "diagnostic_gate": {
            "passed": passed,
            "maximum_successor_rmse": args.maximum_successor_rmse,
            "minimum_overall_improvement": args.minimum_overall_improvement,
            "minimum_high_fare_relative_improvement": (
                args.minimum_high_fare_relative_improvement
            ),
            "stop_after_failure": True,
        },
        "fold_records": fold_records,
        "base_cache_manifest": file_record(cache_dir / "cache_manifest.json"),
        "base_candidate_result": file_record(candidate_dir / "result.json"),
        "base_candidate_metrics": {
            "random_oof_rmse": base_result.get("random_oof_rmse"),
            "temporal_rmse": base_result.get("temporal_rmse"),
            "geographic_rmse": base_result.get("geographic_rmse"),
        },
        "diagnostic_bundle": file_record(bundle_output),
        "elapsed_seconds": time.perf_counter() - started,
        "visibility_mode": "PUBLIC_ONLY",
        "candidate_only": True,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "gpu_used": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "claim_boundary": (
            "One bounded weighted public OOF diagnostic; not an official score or medal."
        ),
    }
    write_json_atomic(output_dir / "result.json", report)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--threads", type=int, default=60)
    parser.add_argument("--distance-threshold-km", type=float, default=3.0)
    parser.add_argument("--weight-pivot-fare", type=float, default=20.0)
    parser.add_argument("--maximum-fare-weight", type=float, default=4.0)
    parser.add_argument("--high-fare-threshold", type=float, default=50.0)
    parser.add_argument("--maximum-successor-rmse", type=float, default=2.85)
    parser.add_argument("--minimum-overall-improvement", type=float, default=1.65)
    parser.add_argument(
        "--minimum-high-fare-relative-improvement",
        type=float,
        default=0.05,
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
