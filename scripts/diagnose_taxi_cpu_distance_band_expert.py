#!/usr/bin/env python3
"""Evaluate a PUBLIC_ONLY cross-fitted Taxi distance-band expert on CPU."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:
    from scripts.diagnose_taxi_cpu_longhaul_expert import (
        CACHE_SCHEMA,
        CLIP_LOWER,
        CLIP_UPPER,
        RESULT_SCHEMA as PREDECESSOR_RESULT_SCHEMA,
        TaxiLonghaulDiagnosticError,
        feature_indices,
        file_record,
        now_iso,
        read_json,
        rmse,
        sha256_file,
        train_expert,
        validate_manifest_artifacts,
        write_json_atomic,
    )
except ModuleNotFoundError:
    from diagnose_taxi_cpu_longhaul_expert import (
        CACHE_SCHEMA,
        CLIP_LOWER,
        CLIP_UPPER,
        RESULT_SCHEMA as PREDECESSOR_RESULT_SCHEMA,
        TaxiLonghaulDiagnosticError,
        feature_indices,
        file_record,
        now_iso,
        read_json,
        rmse,
        sha256_file,
        train_expert,
        validate_manifest_artifacts,
        write_json_atomic,
    )

RESULT_SCHEMA = "evomind.mlebench.taxi_cpu_distance_band_expert_diagnostic.v1"


def distance_band_mask(
    features: np.ndarray,
    feature_names: Sequence[str],
    *,
    minimum_distance_km: float,
    maximum_distance_km: float,
    exclude_airport: bool,
) -> np.ndarray:
    if minimum_distance_km < 0 or maximum_distance_km <= minimum_distance_km:
        raise TaxiLonghaulDiagnosticError("Distance-band bounds are invalid")
    indices = feature_indices(feature_names, ("haversine_km", "airport_trip"))
    values = np.asarray(features)
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise TaxiLonghaulDiagnosticError("Taxi feature matrix shape changed")
    distance = np.asarray(values[:, indices["haversine_km"]], dtype=np.float64)
    airport = np.asarray(values[:, indices["airport_trip"]], dtype=np.float64)
    mask = (distance >= minimum_distance_km) & (distance < maximum_distance_km)
    if exclude_airport:
        mask &= airport < 0.5
    if not mask.any() or mask.all():
        raise TaxiLonghaulDiagnosticError("Distance-band mask is degenerate")
    return mask


def validate_inputs(
    cache_dir: Path,
    predecessor_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    cache_path = cache_dir / "cache_manifest.json"
    predecessor_path = predecessor_dir / "result.json"
    cache = read_json(cache_path)
    predecessor = read_json(predecessor_path)
    if (
        cache.get("schema") != CACHE_SCHEMA
        or cache.get("status") != "completed"
        or cache.get("visibility_mode") != "PUBLIC_ONLY"
        or cache.get("train_rows") != 5_000_000
        or cache.get("folds") != 3
    ):
        raise TaxiLonghaulDiagnosticError("Taxi cache contract changed")
    contracts = cache.get("contracts") or {}
    if (
        contracts.get("private_labels_used") is not False
        or contracts.get("official_grader_executed") is not False
        or contracts.get("kaggle_submission_executed") is not False
        or contracts.get("gpu_used") is not False
    ):
        raise TaxiLonghaulDiagnosticError("Taxi cache boundary changed")
    if (
        predecessor.get("schema") != PREDECESSOR_RESULT_SCHEMA
        or predecessor.get("status") != "diagnostic_improvement_confirmed"
        or predecessor.get("passed") is not True
        or predecessor.get("distance_threshold_km") != 3.0
        or predecessor.get("expert_weight") != 1.0
        or predecessor.get("candidate_only") is not True
        or predecessor.get("private_labels_used") is not False
        or predecessor.get("official_grader_executed") is not False
        or predecessor.get("kaggle_submission_executed") is not False
        or predecessor.get("gpu_used") is not False
    ):
        raise TaxiLonghaulDiagnosticError("Taxi predecessor contract changed")
    if (predecessor.get("base_cache_manifest") or {}).get(
        "sha256"
    ) != sha256_file(cache_path):
        raise TaxiLonghaulDiagnosticError("Taxi predecessor cache hash changed")
    bundle_record = predecessor.get("diagnostic_bundle") or {}
    bundle_path = predecessor_dir / "longhaul_expert_oof.npz"
    if (
        not bundle_path.is_file()
        or bundle_path.stat().st_size != int(bundle_record.get("bytes") or -1)
        or sha256_file(bundle_path) != bundle_record.get("sha256")
    ):
        raise TaxiLonghaulDiagnosticError("Taxi predecessor bundle changed")
    validate_manifest_artifacts(cache_dir, cache)
    return cache, predecessor, bundle_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cache_dir = Path(args.cache_dir).resolve()
    predecessor_dir = Path(args.predecessor_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise TaxiLonghaulDiagnosticError("Output directory must be new or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache, predecessor, bundle_path = validate_inputs(cache_dir, predecessor_dir)

    train_features = np.load(cache_dir / "train_features.npy", mmap_mode="r")
    feature_names = list(cache.get("feature_names") or [])
    cache_target = np.asarray(
        np.load(cache_dir / "target.npy", mmap_mode="r"), dtype=np.float64
    )
    cache_folds = np.asarray(
        np.load(cache_dir / "fold_assignment.npy", mmap_mode="r"), dtype=np.int16
    )
    with np.load(bundle_path, allow_pickle=False) as archive:
        target = np.asarray(archive["truth"], dtype=np.float64)
        incumbent = np.asarray(archive["blended_oof_prediction"], dtype=np.float64)
        predecessor_folds = np.asarray(
            archive["fold_assignment"], dtype=np.int16
        )
    if train_features.shape != (5_000_000, len(feature_names)):
        raise TaxiLonghaulDiagnosticError("Taxi train feature shape changed")
    if not np.array_equal(cache_folds, predecessor_folds):
        raise TaxiLonghaulDiagnosticError("Taxi predecessor fold assignment drifted")
    if not np.array_equal(cache_target.astype(np.float32), target.astype(np.float32)):
        raise TaxiLonghaulDiagnosticError("Taxi predecessor truth drifted")

    selected = distance_band_mask(
        train_features,
        feature_names,
        minimum_distance_km=args.minimum_distance_km,
        maximum_distance_km=args.maximum_distance_km,
        exclude_airport=args.exclude_airport,
    )
    expert_oof = np.full(len(target), np.nan, dtype=np.float64)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    fold_records: list[dict[str, Any]] = []
    for fold in sorted(int(value) for value in np.unique(cache_folds)):
        fit = np.flatnonzero((cache_folds != fold) & selected)
        valid = np.flatnonzero((cache_folds == fold) & selected)
        if len(fit) < 10_000 or len(valid) < 1_000:
            raise TaxiLonghaulDiagnosticError("Distance-band fold is too small")
        fit_x = np.asarray(train_features[fit], dtype=np.float32)
        valid_x = np.asarray(train_features[valid], dtype=np.float32)
        model = train_expert(
            fit_x,
            target[fit],
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
                "fit_band_rows": len(fit),
                "valid_band_rows": len(valid),
                "incumbent_band_rmse": rmse(target[valid], incumbent[valid]),
                "expert_band_rmse": rmse(target[valid], prediction),
                "model": file_record(model_path),
            }
        )
        del fit_x, valid_x, model
    if not np.isfinite(expert_oof[selected]).all():
        raise TaxiLonghaulDiagnosticError("Distance-band OOF coverage is incomplete")
    successor = incumbent.copy()
    successor[selected] = expert_oof[selected]
    successor = np.clip(successor, CLIP_LOWER, CLIP_UPPER)
    incumbent_rmse = rmse(target, incumbent)
    successor_rmse = rmse(target, successor)
    incumbent_band_rmse = rmse(target[selected], incumbent[selected])
    expert_band_rmse = rmse(target[selected], expert_oof[selected])
    passed = (
        successor_rmse <= args.maximum_successor_rmse
        and successor_rmse <= incumbent_rmse - args.minimum_overall_improvement
        and expert_band_rmse
        <= incumbent_band_rmse
        * (1.0 - args.minimum_band_relative_improvement)
    )
    bundle_output = output_dir / "distance_band_expert_oof.npz"
    np.savez_compressed(
        bundle_output,
        truth=target.astype(np.float32),
        incumbent_oof_prediction=incumbent.astype(np.float32),
        expert_oof_prediction=expert_oof.astype(np.float32),
        successor_oof_prediction=successor.astype(np.float32),
        distance_band_mask=selected.astype(np.uint8),
        fold_assignment=cache_folds,
    )
    report = {
        "schema": RESULT_SCHEMA,
        "created_at": now_iso(),
        "status": (
            "diagnostic_improvement_confirmed"
            if passed
            else "diagnostic_improvement_insufficient"
        ),
        "passed": passed,
        "competition_id": "new-york-city-taxi-fare-prediction",
        "model_family": "LightGBM_CPU_cross_fitted_distance_band_successor",
        "seed": args.seed,
        "folds": int(cache["folds"]),
        "iterations": args.iterations,
        "threads": args.threads,
        "minimum_distance_km": args.minimum_distance_km,
        "maximum_distance_km": args.maximum_distance_km,
        "exclude_airport": args.exclude_airport,
        "band_rows": int(selected.sum()),
        "band_fraction": float(selected.mean()),
        "metrics": {
            "incumbent_random_oof_rmse": incumbent_rmse,
            "successor_random_oof_rmse": successor_rmse,
            "overall_absolute_improvement": incumbent_rmse - successor_rmse,
            "incumbent_band_rmse": incumbent_band_rmse,
            "expert_band_rmse": expert_band_rmse,
            "band_relative_improvement": (
                (incumbent_band_rmse - expert_band_rmse)
                / incumbent_band_rmse
            ),
        },
        "diagnostic_gate": {
            "passed": passed,
            "maximum_successor_rmse": args.maximum_successor_rmse,
            "minimum_overall_improvement": args.minimum_overall_improvement,
            "minimum_band_relative_improvement": (
                args.minimum_band_relative_improvement
            ),
        },
        "fold_records": fold_records,
        "base_cache_manifest": file_record(cache_dir / "cache_manifest.json"),
        "predecessor_result": file_record(predecessor_dir / "result.json"),
        "predecessor_metrics": predecessor.get("metrics"),
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
            "Cross-fitted public OOF distance-band diagnostic only; "
            "not an official score or medal."
        ),
    }
    write_json_atomic(output_dir / "result.json", report)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--predecessor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--threads", type=int, default=60)
    parser.add_argument("--minimum-distance-km", type=float, default=3.0)
    parser.add_argument("--maximum-distance-km", type=float, default=6.0)
    parser.add_argument("--exclude-airport", action="store_true")
    parser.add_argument("--maximum-successor-rmse", type=float, default=2.85)
    parser.add_argument("--minimum-overall-improvement", type=float, default=0.02)
    parser.add_argument("--minimum-band-relative-improvement", type=float, default=0.03)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
