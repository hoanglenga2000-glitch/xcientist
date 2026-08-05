#!/usr/bin/env python3
"""Evaluate a PUBLIC_ONLY cross-fitted Taxi long-haul expert on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

CACHE_SCHEMA = "evomind.mlebench.taxi_public_feature_cache.v1"
BASE_RESULT_SCHEMA = "evomind.mlebench.taxi_cpu_lightgbm_candidate.v1"
RESULT_SCHEMA = "evomind.mlebench.taxi_cpu_longhaul_expert_diagnostic.v1"
CLIP_LOWER = 2.5
CLIP_UPPER = 100.0
DEFAULT_DISTANCE_THRESHOLD_KM = 6.0
DEFAULT_BLEND_WEIGHT = 0.75


class TaxiLonghaulDiagnosticError(RuntimeError):
    """Raised when a frozen input or diagnostic invariant changes."""


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TaxiLonghaulDiagnosticError(f"JSON object required: {path}")
    return payload


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape or truth.ndim != 1:
        raise TaxiLonghaulDiagnosticError("RMSE arrays are misaligned")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise TaxiLonghaulDiagnosticError("RMSE arrays contain non-finite values")
    return float(np.sqrt(np.mean(np.square(truth - prediction))))


def feature_indices(
    feature_names: Sequence[str],
    required: Sequence[str],
) -> dict[str, int]:
    index = {name: position for position, name in enumerate(feature_names)}
    missing = sorted(set(required) - set(index))
    if missing:
        raise TaxiLonghaulDiagnosticError(
            f"Taxi long-haul diagnostic lacks features: {missing}"
        )
    return {name: index[name] for name in required}


def longhaul_mask(
    features: np.ndarray,
    feature_names: Sequence[str],
    *,
    distance_threshold_km: float,
) -> np.ndarray:
    if distance_threshold_km <= 0:
        raise TaxiLonghaulDiagnosticError("Distance threshold must be positive")
    indices = feature_indices(feature_names, ("haversine_km", "airport_trip"))
    values = np.asarray(features)
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise TaxiLonghaulDiagnosticError("Taxi feature matrix shape changed")
    distance = np.asarray(values[:, indices["haversine_km"]], dtype=np.float64)
    airport = np.asarray(values[:, indices["airport_trip"]], dtype=np.float64)
    mask = (distance >= distance_threshold_km) | (airport >= 0.5)
    if not mask.any() or mask.all():
        raise TaxiLonghaulDiagnosticError("Long-haul mask is degenerate")
    return mask


def blend_longhaul_predictions(
    base_prediction: np.ndarray,
    expert_prediction: np.ndarray,
    mask: np.ndarray,
    *,
    expert_weight: float,
) -> np.ndarray:
    if not 0.0 <= expert_weight <= 1.0:
        raise TaxiLonghaulDiagnosticError("Expert blend weight is outside [0, 1]")
    base = np.asarray(base_prediction, dtype=np.float64)
    expert = np.asarray(expert_prediction, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    if base.shape != expert.shape or base.shape != selected.shape or base.ndim != 1:
        raise TaxiLonghaulDiagnosticError("Long-haul prediction arrays are misaligned")
    result = base.copy()
    result[selected] = (
        (1.0 - expert_weight) * base[selected]
        + expert_weight * expert[selected]
    )
    result = np.clip(result, CLIP_LOWER, CLIP_UPPER)
    if not np.isfinite(result).all():
        raise TaxiLonghaulDiagnosticError("Blended prediction is non-finite")
    return result


def validate_manifest_artifacts(root: Path, manifest: Mapping[str, Any]) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise TaxiLonghaulDiagnosticError("Taxi cache artifacts are missing")
    for record in artifacts:
        relative = Path(str((record or {}).get("relative_path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise TaxiLonghaulDiagnosticError("Taxi cache artifact path is unsafe")
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != int((record or {}).get("bytes") or -1)
            or sha256_file(path) != (record or {}).get("sha256")
        ):
            raise TaxiLonghaulDiagnosticError(
                f"Taxi cache artifact differs: {relative}"
            )


def validate_inputs(
    cache_dir: Path,
    candidate_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    cache_path = cache_dir / "cache_manifest.json"
    result_path = candidate_dir / "result.json"
    cache = read_json(cache_path)
    result = read_json(result_path)
    if (
        cache.get("schema") != CACHE_SCHEMA
        or cache.get("status") != "completed"
        or cache.get("visibility_mode") != "PUBLIC_ONLY"
        or cache.get("train_rows") != 5_000_000
        or cache.get("test_rows") != 9_914
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
        result.get("schema") != BASE_RESULT_SCHEMA
        or result.get("candidate_only") is not True
        or result.get("private_labels_used") is not False
        or result.get("official_grader_executed") is not False
        or result.get("kaggle_submission_executed") is not False
        or result.get("gpu_used") is not False
    ):
        raise TaxiLonghaulDiagnosticError("Taxi base candidate boundary changed")
    expected_cache = result.get("cache_manifest") or {}
    if expected_cache.get("sha256") != sha256_file(cache_path):
        raise TaxiLonghaulDiagnosticError("Taxi base candidate cache hash changed")
    bundle = result.get("candidate_bundle") or {}
    bundle_path = candidate_dir / "taxi_cpu_candidate.npz"
    if (
        not bundle_path.is_file()
        or bundle_path.stat().st_size != int(bundle.get("bytes") or -1)
        or sha256_file(bundle_path) != bundle.get("sha256")
    ):
        raise TaxiLonghaulDiagnosticError("Taxi base candidate bundle changed")
    validate_manifest_artifacts(cache_dir, cache)
    return cache, result, bundle_path


def train_expert(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
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
    dataset = lgb.Dataset(fit_x, label=fit_y, free_raw_data=True)
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

    long_mask = longhaul_mask(
        train_features,
        feature_names,
        distance_threshold_km=args.distance_threshold_km,
    )
    expert_oof = np.full(len(target), np.nan, dtype=np.float64)
    fold_records: list[dict[str, Any]] = []
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for fold in sorted(int(value) for value in np.unique(cache_folds)):
        fit = np.flatnonzero((cache_folds != fold) & long_mask)
        valid = np.flatnonzero((cache_folds == fold) & long_mask)
        if len(fit) < 10_000 or len(valid) < 1_000:
            raise TaxiLonghaulDiagnosticError("Long-haul fold is too small")
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
                "fit_longhaul_rows": len(fit),
                "valid_longhaul_rows": len(valid),
                "base_longhaul_rmse": rmse(target[valid], base_oof[valid]),
                "expert_longhaul_rmse": rmse(target[valid], prediction),
                "model": file_record(model_path),
            }
        )
        del fit_x, valid_x, model
    if not np.isfinite(expert_oof[long_mask]).all():
        raise TaxiLonghaulDiagnosticError("Long-haul expert OOF coverage is incomplete")
    expert_full = base_oof.copy()
    expert_full[long_mask] = expert_oof[long_mask]
    blended = blend_longhaul_predictions(
        base_oof,
        expert_full,
        long_mask,
        expert_weight=args.expert_weight,
    )
    base_rmse = rmse(target, base_oof)
    blended_rmse = rmse(target, blended)
    base_long_rmse = rmse(target[long_mask], base_oof[long_mask])
    expert_long_rmse = rmse(target[long_mask], expert_oof[long_mask])
    diagnostic_passed = (
        blended_rmse <= base_rmse - args.minimum_overall_improvement
        and expert_long_rmse
        <= base_long_rmse * (1.0 - args.minimum_longhaul_relative_improvement)
    )
    bundle_output = output_dir / "longhaul_expert_oof.npz"
    np.savez_compressed(
        bundle_output,
        truth=target.astype(np.float32),
        base_oof_prediction=base_oof.astype(np.float32),
        expert_oof_prediction=expert_oof.astype(np.float32),
        blended_oof_prediction=blended.astype(np.float32),
        longhaul_mask=long_mask.astype(np.uint8),
        fold_assignment=cache_folds,
    )
    report = {
        "schema": RESULT_SCHEMA,
        "created_at": now_iso(),
        "status": (
            "diagnostic_improvement_confirmed"
            if diagnostic_passed
            else "diagnostic_improvement_insufficient"
        ),
        "passed": diagnostic_passed,
        "competition_id": "new-york-city-taxi-fare-prediction",
        "model_family": "LightGBM_CPU_cross_fitted_longhaul_expert",
        "seed": args.seed,
        "folds": int(cache["folds"]),
        "iterations": args.iterations,
        "threads": args.threads,
        "distance_threshold_km": args.distance_threshold_km,
        "expert_weight": args.expert_weight,
        "longhaul_rows": int(long_mask.sum()),
        "longhaul_fraction": float(long_mask.mean()),
        "metrics": {
            "base_random_oof_rmse": base_rmse,
            "blended_random_oof_rmse": blended_rmse,
            "overall_absolute_improvement": base_rmse - blended_rmse,
            "base_longhaul_rmse": base_long_rmse,
            "expert_longhaul_rmse": expert_long_rmse,
            "longhaul_relative_improvement": (
                (base_long_rmse - expert_long_rmse) / base_long_rmse
            ),
        },
        "diagnostic_gate": {
            "passed": diagnostic_passed,
            "minimum_overall_improvement": args.minimum_overall_improvement,
            "minimum_longhaul_relative_improvement": (
                args.minimum_longhaul_relative_improvement
            ),
        },
        "fold_records": fold_records,
        "base_cache_manifest": file_record(cache_dir / "cache_manifest.json"),
        "base_candidate_result": file_record(candidate_dir / "result.json"),
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
            "Cross-fitted public OOF diagnostic only; not an official score or medal."
        ),
    }
    write_json_atomic(output_dir / "result.json", report)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--threads", type=int, default=60)
    parser.add_argument(
        "--distance-threshold-km",
        type=float,
        default=DEFAULT_DISTANCE_THRESHOLD_KM,
    )
    parser.add_argument(
        "--expert-weight",
        type=float,
        default=DEFAULT_BLEND_WEIGHT,
    )
    parser.add_argument("--minimum-overall-improvement", type=float, default=0.10)
    parser.add_argument(
        "--minimum-longhaul-relative-improvement",
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
