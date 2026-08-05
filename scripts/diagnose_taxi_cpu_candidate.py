#!/usr/bin/env python3
"""Train a PUBLIC_ONLY LightGBM Taxi candidate from verified CPU caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

CACHE_SCHEMA = "evomind.mlebench.taxi_public_feature_cache.v1"
ROUTE_SCHEMA = "evomind.mlebench.taxi_route_stat_sidecar.v1"
RESULT_SCHEMA = "evomind.mlebench.taxi_cpu_lightgbm_candidate.v1"
ROUTE_COLUMNS = (
    "route_stat_route_cell",
    "route_stat_route_hour",
    "route_stat_airport_route",
    "route_stat_borough_route",
)
CLIP_LOWER = 2.5
CLIP_UPPER = 100.0


class TaxiCpuCandidateError(RuntimeError):
    """Raised when an input or validation contract changes."""


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
        raise TaxiCpuCandidateError(f"JSON object required: {path}")
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


def validate_manifest_artifacts(root: Path, manifest: Mapping[str, Any]) -> None:
    records = manifest.get("artifacts")
    if not isinstance(records, list) or not records:
        raise TaxiCpuCandidateError("manifest artifact list is missing")
    for record in records:
        if not isinstance(record, dict):
            raise TaxiCpuCandidateError("manifest artifact record changed")
        relative = Path(str(record.get("relative_path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise TaxiCpuCandidateError("manifest artifact path is unsafe")
        path = root / relative
        if not path.is_file():
            raise TaxiCpuCandidateError(f"manifest artifact is missing: {relative}")
        if path.stat().st_size != int(record.get("bytes") or -1):
            raise TaxiCpuCandidateError(f"manifest artifact size differs: {relative}")
        if sha256_file(path) != record.get("sha256"):
            raise TaxiCpuCandidateError(f"manifest artifact hash differs: {relative}")


def validate_inputs(cache_dir: Path, route_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cache_dir = cache_dir.resolve()
    route_dir = route_dir.resolve()
    cache_path = cache_dir / "cache_manifest.json"
    route_path = route_dir / "route_stat_manifest.json"
    cache = read_json(cache_path)
    route = read_json(route_path)
    if (
        cache.get("schema") != CACHE_SCHEMA
        or cache.get("status") != "completed"
        or cache.get("visibility_mode") != "PUBLIC_ONLY"
        or cache.get("train_rows") != 5_000_000
        or cache.get("test_rows") != 9_914
        or cache.get("folds") != 3
    ):
        raise TaxiCpuCandidateError("Taxi base cache contract changed")
    contracts = cache.get("contracts") or {}
    if (
        contracts.get("private_labels_used") is not False
        or contracts.get("official_grader_executed") is not False
        or contracts.get("kaggle_submission_executed") is not False
        or contracts.get("gpu_used") is not False
    ):
        raise TaxiCpuCandidateError("Taxi base cache boundary changed")
    if (
        route.get("schema") != ROUTE_SCHEMA
        or route.get("status") != "completed"
        or route.get("visibility_mode") != "PUBLIC_ONLY"
        or route.get("train_rows") != cache.get("train_rows")
        or route.get("test_rows") != cache.get("test_rows")
        or route.get("folds") != cache.get("folds")
        or tuple(route.get("route_output_columns") or ()) != ROUTE_COLUMNS
        or (route.get("base_cache_manifest") or {}).get("sha256")
        != sha256_file(cache_path)
    ):
        raise TaxiCpuCandidateError("Taxi route-stat cache contract changed")
    route_contracts = route.get("contracts") or {}
    if (
        route_contracts.get("validation_targets_used") != 0
        or route_contracts.get("private_labels_used") is not False
        or route_contracts.get("official_grader_executed") is not False
        or route_contracts.get("kaggle_submission_executed") is not False
        or route_contracts.get("gpu_used") is not False
    ):
        raise TaxiCpuCandidateError("Taxi route-stat boundary changed")
    validate_manifest_artifacts(cache_dir, cache)
    validate_manifest_artifacts(route_dir, route)
    return cache, route


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape or truth.ndim != 1:
        raise TaxiCpuCandidateError("RMSE arrays are misaligned")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise TaxiCpuCandidateError("RMSE arrays contain non-finite values")
    return float(np.sqrt(np.mean(np.square(truth - prediction))))


def attach_route_stats(base: np.ndarray, route: np.ndarray) -> np.ndarray:
    base = np.asarray(base, dtype=np.float32)
    route = np.asarray(route, dtype=np.float32)
    if base.ndim != 2 or route.shape != (len(base), len(ROUTE_COLUMNS)):
        raise TaxiCpuCandidateError("Taxi route-stat shape differs")
    result = np.concatenate((base, route), axis=1, dtype=np.float32)
    if not np.isfinite(result).all():
        raise TaxiCpuCandidateError("Taxi model matrix contains non-finite values")
    return result


def train_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    *,
    seed: int,
    iterations: int,
    threads: int,
) -> tuple[Any, int]:
    import lightgbm as lgb

    params = {
        "objective": "regression_l2",
        "metric": "rmse",
        "learning_rate": 0.035,
        "num_leaves": 127,
        "max_depth": 12,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "max_bin": 255,
        "num_threads": threads,
        "seed": seed,
        "feature_fraction_seed": seed + 1000,
        "bagging_seed": seed + 2000,
        "data_random_seed": seed + 3000,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    train_set = lgb.Dataset(train_x, label=train_y, free_raw_data=True)
    valid_set = lgb.Dataset(valid_x, label=valid_y, reference=train_set, free_raw_data=True)
    model = lgb.train(
        params,
        train_set,
        num_boost_round=iterations,
        valid_sets=[valid_set],
        valid_names=["validation"],
        callbacks=[lgb.early_stopping(100, verbose=True), lgb.log_evaluation(50)],
    )
    return model, int(model.best_iteration or iterations)


def clipped_predict(model: Any, values: np.ndarray, *, iteration: int) -> np.ndarray:
    prediction = np.asarray(
        model.predict(values, num_iteration=iteration), dtype=np.float64
    ).reshape(-1)
    prediction = np.clip(prediction, CLIP_LOWER, CLIP_UPPER)
    if not np.isfinite(prediction).all():
        raise TaxiCpuCandidateError("Taxi model produced non-finite predictions")
    return prediction


def load_split(archive: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    if name not in archive:
        raise TaxiCpuCandidateError(f"Taxi split is missing: {name}")
    values = np.asarray(archive[name], dtype=np.int64)
    if values.ndim != 1 or len(values) == 0:
        raise TaxiCpuCandidateError(f"Taxi split is invalid: {name}")
    return values


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    cache_dir = Path(args.cache_dir).resolve()
    route_dir = Path(args.route_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache, route = validate_inputs(cache_dir, route_dir)

    base_train = np.load(cache_dir / "train_features.npy", mmap_mode="r")
    base_test = np.load(cache_dir / "test_features.npy", mmap_mode="r")
    target = np.asarray(np.load(cache_dir / "target.npy", mmap_mode="r"), dtype=np.float64)
    fold_assignment = np.asarray(
        np.load(cache_dir / "fold_assignment.npy", mmap_mode="r"), dtype=np.int16
    )
    test_key = np.asarray(np.load(cache_dir / "test_key.npy", mmap_mode="r")).astype(str)
    split_archive = np.load(cache_dir / "split_indices.npz", allow_pickle=False)
    if base_train.shape != (5_000_000, 72) or base_test.shape != (9_914, 72):
        raise TaxiCpuCandidateError("Taxi base feature shape changed")
    if target.shape != (len(base_train),) or fold_assignment.shape != target.shape:
        raise TaxiCpuCandidateError("Taxi target/fold shape changed")

    oof = np.full(len(target), np.nan, dtype=np.float64)
    test_folds = np.zeros((len(base_test), int(cache["folds"])), dtype=np.float64)
    fold_records: list[dict[str, Any]] = []
    selected_iterations: list[int] = []
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(int(cache["folds"])):
        fit = load_split(split_archive, f"fold_{fold:02d}_fit")
        valid = load_split(split_archive, f"fold_{fold:02d}_valid")
        route_train = np.load(
            route_dir / f"oof_fold_{fold:02d}_train.npy", mmap_mode="r"
        )
        route_test = np.load(
            route_dir / f"oof_fold_{fold:02d}_test.npy", mmap_mode="r"
        )
        fit_x = attach_route_stats(base_train[fit], route_train[fit])
        valid_x = attach_route_stats(base_train[valid], route_train[valid])
        test_x = attach_route_stats(base_test, route_test)
        model, best = train_model(
            fit_x,
            target[fit],
            valid_x,
            target[valid],
            seed=args.seed + fold,
            iterations=args.iterations,
            threads=args.threads,
        )
        valid_prediction = clipped_predict(model, valid_x, iteration=best)
        test_prediction = clipped_predict(model, test_x, iteration=best)
        oof[valid] = valid_prediction
        test_folds[:, fold] = test_prediction
        model_path = model_dir / f"fold_{fold:02d}.txt"
        model.save_model(str(model_path), num_iteration=best)
        score = rmse(target[valid], valid_prediction)
        selected_iterations.append(best)
        fold_records.append(
            {
                "fold": fold,
                "fit_rows": len(fit),
                "valid_rows": len(valid),
                "best_iteration": best,
                "rmse": score,
                "model": file_record(model_path),
                "validation_targets_used_for_route_statistics": 0,
            }
        )
        del fit_x, valid_x, test_x, model
    if not np.isfinite(oof).all():
        raise TaxiCpuCandidateError("Taxi OOF coverage is incomplete")
    random_oof_rmse = rmse(target, oof)

    stress_scores: dict[str, float] = {}
    stress_bundle: dict[str, np.ndarray] = {}
    for offset, context in enumerate(("temporal", "geographic"), start=100):
        fit = load_split(split_archive, f"{context}_fit")
        valid = load_split(split_archive, f"{context}_valid")
        route_train = np.load(route_dir / f"{context}_train.npy", mmap_mode="r")
        fit_x = attach_route_stats(base_train[fit], route_train[fit])
        valid_x = attach_route_stats(base_train[valid], route_train[valid])
        model, best = train_model(
            fit_x,
            target[fit],
            valid_x,
            target[valid],
            seed=args.seed + offset,
            iterations=args.iterations,
            threads=args.threads,
        )
        prediction = clipped_predict(model, valid_x, iteration=best)
        stress_scores[f"{context}_rmse"] = rmse(target[valid], prediction)
        stress_bundle[f"{context}_valid_index"] = valid.astype(np.int64)
        stress_bundle[f"{context}_truth"] = target[valid].astype(np.float32)
        stress_bundle[f"{context}_prediction"] = prediction.astype(np.float32)
        model_path = model_dir / f"{context}.txt"
        model.save_model(str(model_path), num_iteration=best)
        del fit_x, valid_x, model

    refit_route_train = np.load(route_dir / "refit_train.npy", mmap_mode="r")
    refit_route_test = np.load(route_dir / "refit_test.npy", mmap_mode="r")
    refit_x = attach_route_stats(base_train, refit_route_train)
    refit_test_x = attach_route_stats(base_test, refit_route_test)
    refit_iterations = int(np.median(selected_iterations))
    import lightgbm as lgb

    refit_params = {
        "objective": "regression_l2",
        "learning_rate": 0.035,
        "num_leaves": 127,
        "max_depth": 12,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "max_bin": 255,
        "num_threads": args.threads,
        "seed": args.seed + 500,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    refit_model = lgb.train(
        refit_params,
        lgb.Dataset(refit_x, label=target, free_raw_data=True),
        num_boost_round=refit_iterations,
        callbacks=[lgb.log_evaluation(50)],
    )
    refit_prediction = clipped_predict(
        refit_model, refit_test_x, iteration=refit_iterations
    )
    fold_prediction = np.mean(test_folds, axis=1)
    test_prediction = 0.75 * refit_prediction + 0.25 * fold_prediction
    refit_model_path = model_dir / "refit.txt"
    refit_model.save_model(str(refit_model_path), num_iteration=refit_iterations)

    gates = {
        "random_oof_rmse_at_or_below_2_85": random_oof_rmse <= 2.85,
        "temporal_rmse_at_or_below_3_10": stress_scores["temporal_rmse"] <= 3.10,
        "geographic_rmse_at_or_below_3_35": stress_scores["geographic_rmse"] <= 3.35,
    }
    passed = all(gates.values())
    bundle_path = output_dir / "taxi_cpu_candidate.npz"
    np.savez_compressed(
        bundle_path,
        truth=target.astype(np.float32),
        oof_prediction=oof.astype(np.float32),
        fold_assignment=fold_assignment,
        test_key=test_key,
        fold_test_prediction=test_folds.astype(np.float32),
        refit_test_prediction=refit_prediction.astype(np.float32),
        selected_test_prediction=test_prediction.astype(np.float32),
        **stress_bundle,
    )
    result = {
        "schema": RESULT_SCHEMA,
        "created_at": now_iso(),
        "status": "candidate_complete" if passed else "verification_complete_gate_failed",
        "passed": passed,
        "competition_id": "new-york-city-taxi-fare-prediction",
        "model_family": "LightGBM_CPU_5M_duplicate_safe_route_stats",
        "seed": args.seed,
        "folds": int(cache["folds"]),
        "iterations_requested": args.iterations,
        "threads": args.threads,
        "fold_records": fold_records,
        "random_oof_rmse": random_oof_rmse,
        **stress_scores,
        "refit_iterations": refit_iterations,
        "promotion_gate": {"passed": passed, "checks": gates},
        "cache_manifest": file_record(cache_dir / "cache_manifest.json"),
        "route_stat_manifest": file_record(route_dir / "route_stat_manifest.json"),
        "candidate_bundle": file_record(bundle_path),
        "refit_model": file_record(refit_model_path),
        "elapsed_seconds": time.perf_counter() - started,
        "visibility_mode": "PUBLIC_ONLY",
        "candidate_only": True,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "gpu_used": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "claim_boundary": "PUBLIC OOF candidate evidence; not an official score or medal.",
    }
    write_json_atomic(output_dir / "result.json", result)
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--route-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--iterations", type=int, default=1400)
    parser.add_argument("--threads", type=int, default=60)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
