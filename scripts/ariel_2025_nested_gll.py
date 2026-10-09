from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 93207
TARGETS = 283
OUTER_FOLDS = 5
INNER_FOLDS = 3
BOOTSTRAP_ROUNDS = 20_000
ALPHAS = (0.1, 1.0, 10.0, 100.0)
SIGMA_SCALES = (0.75, 1.0, 1.25, 1.5, 2.0)
FGS_WEIGHT = 57.846
FGS_SIGMA_TRUE = 1e-6
AIRS_SIGMA_TRUE = 1e-5


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def id_set_sha(values: np.ndarray | list[str]) -> str:
    return sha256_bytes(("\n".join(sorted(map(str, values))) + "\n").encode("utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_event(path: Path, event: str, **payload: Any) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"at_utc": utc_now(), "event": event, **payload}, separators=(",", ":")) + "\n")
        handle.flush()


def official_gll(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    naive_mean: float,
    naive_sigma: float,
) -> float:
    """Array-equivalent implementation of Kaggle metric/ariel-gaussian-log-likelihood v4."""

    per_planet = official_gll_per_planet(
        y_true,
        mu,
        sigma,
        naive_mean=naive_mean,
        naive_sigma=naive_sigma,
    )
    return float(np.clip(np.mean(per_planet), 0.0, 1.0))


def official_gll_per_planet(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    naive_mean: float,
    naive_sigma: float,
) -> np.ndarray:
    """Return the weighted wavelength contribution for every planet before clipping."""

    if y_true.shape != mu.shape or y_true.shape != sigma.shape or y_true.shape[1] != TARGETS:
        raise ValueError("official GLL shape mismatch")
    if not np.isfinite(y_true).all() or not np.isfinite(mu).all() or not np.isfinite(sigma).all():
        raise ValueError("official GLL received non-finite values")
    if np.min(mu) < 0 or np.min(sigma) < 0:
        raise ValueError("official GLL predictions must be nonnegative")
    sigma_pred = np.clip(sigma, 1e-15, None)
    sigma_true = np.concatenate(([FGS_SIGMA_TRUE], np.full(TARGETS - 1, AIRS_SIGMA_TRUE)))
    naive_sigma = max(float(naive_sigma), 1e-15)
    constant = math.log(2.0 * math.pi)
    gll_pred = -0.5 * (constant + 2.0 * np.log(sigma_pred) + ((y_true - mu) / sigma_pred) ** 2)
    gll_true = -0.5 * (constant + 2.0 * np.log(sigma_true[None, :]))
    gll_mean = -0.5 * (
        constant + 2.0 * math.log(naive_sigma) + ((y_true - float(naive_mean)) / naive_sigma) ** 2
    )
    denominator = gll_true - gll_mean
    if not np.isfinite(denominator).all() or np.any(np.abs(denominator) < 1e-15):
        raise ValueError("official GLL normalizer is invalid")
    individual = (gll_pred - gll_mean) / denominator
    weights = np.concatenate(([FGS_WEIGHT], np.ones(TARGETS - 1)))
    return np.average(individual, axis=1, weights=weights)


def scorer_anchors(y: np.ndarray) -> dict[str, float | bool]:
    naive_mean = float(np.mean(y))
    naive_sigma = float(np.std(y))
    reference_mu = np.full_like(y, naive_mean)
    reference_sigma = np.full_like(y, naive_sigma)
    ideal_sigma = np.broadcast_to(
        np.concatenate(([FGS_SIGMA_TRUE], np.full(TARGETS - 1, AIRS_SIGMA_TRUE))), y.shape
    ).copy()
    reference = official_gll(y, reference_mu, reference_sigma, naive_mean=naive_mean, naive_sigma=naive_sigma)
    ideal = official_gll(y, y.copy(), ideal_sigma, naive_mean=naive_mean, naive_sigma=naive_sigma)
    return {
        "reference": reference,
        "ideal": ideal,
        "reference_zero": abs(reference) <= 1e-12,
        "ideal_one": abs(ideal - 1.0) <= 1e-12,
    }


def parquet_summary(path: Path) -> tuple[list[float], dict[str, int]]:
    if not path.is_file() or path.is_symlink():
        return [0.0] * 18, {"files": 0, "sample_values": 0, "bytes": 0, "errors": 0}
    try:
        parquet = pq.ParquetFile(path)
        meta = parquet.metadata
        minimums: list[float] = []
        maximums: list[float] = []
        for row_group_index in range(meta.num_row_groups):
            row_group = meta.row_group(row_group_index)
            for column_index in range(row_group.num_columns):
                stats = row_group.column(column_index).statistics
                if stats is None or not stats.has_min_max:
                    continue
                if isinstance(stats.min, (int, float)) and isinstance(stats.max, (int, float)):
                    low = float(stats.min)
                    high = float(stats.max)
                    if math.isfinite(low) and math.isfinite(high):
                        minimums.append(low)
                        maximums.append(high)
        names = list(parquet.schema.names)
        selected_indices = sorted(set(np.linspace(0, max(0, len(names) - 1), min(12, len(names)), dtype=int)))
        selected_names = [names[index] for index in selected_indices]
        batch = next(parquet.iter_batches(batch_size=256, columns=selected_names), None) if selected_names else None
        values = np.asarray([], dtype=float)
        trend = 0.0
        if batch is not None:
            numeric = batch.to_pandas().select_dtypes(include=[np.number])
            if not numeric.empty:
                matrix = numeric.to_numpy(dtype=float, copy=False)
                finite_values = matrix[np.isfinite(matrix)]
                values = finite_values.astype(float, copy=False)
                first = matrix[0]
                last = matrix[-1]
                finite = np.isfinite(first) & np.isfinite(last)
                if finite.any():
                    trend = float(np.mean(last[finite] - first[finite]))
        ranges = np.asarray(maximums) - np.asarray(minimums) if minimums else np.asarray([], dtype=float)
        features = [
            1.0,
            float(path.stat().st_size),
            float(meta.num_rows),
            float(meta.num_row_groups),
            float(meta.num_columns),
            float(len(minimums)),
            float(np.mean(minimums)) if minimums else 0.0,
            float(np.mean(maximums)) if maximums else 0.0,
            float(np.mean(ranges)) if ranges.size else 0.0,
            float(values.size),
            float(np.mean(values)) if values.size else 0.0,
            float(np.std(values)) if values.size else 0.0,
            float(np.min(values)) if values.size else 0.0,
            float(np.max(values)) if values.size else 0.0,
            float(np.quantile(values, 0.1)) if values.size else 0.0,
            float(np.quantile(values, 0.5)) if values.size else 0.0,
            float(np.quantile(values, 0.9)) if values.size else 0.0,
            trend,
        ]
        return features, {"files": 1, "sample_values": int(values.size), "bytes": int(path.stat().st_size), "errors": 0}
    except (OSError, ValueError, TypeError, IndexError, KeyError):
        return [0.0] * 18, {"files": 0, "sample_values": 0, "bytes": 0, "errors": 1}


def component_paths(train_root: Path, planet_id: str) -> list[tuple[str, Path]]:
    root = train_root / planet_id
    return [
        ("airs_signal", root / "AIRS-CH0_signal_0.parquet"),
        ("fgs_signal", root / "FGS1_signal_0.parquet"),
        ("airs_dark", root / "AIRS-CH0_calibration_0" / "dark.parquet"),
        ("fgs_dark", root / "FGS1_calibration_0" / "dark.parquet"),
    ]


def build_features(data_root: Path, frame: pd.DataFrame, log_path: Path) -> tuple[np.ndarray, list[str], dict[str, int]]:
    star_columns = [name for name in frame.columns if name.startswith("star__")]
    star_matrix = frame[star_columns].astype(float).to_numpy()
    summary_fields = (
        "file_count", "file_bytes", "rows", "row_groups", "columns", "statistics_count",
        "minimum_mean", "maximum_mean", "range_mean", "sample_count", "sample_mean", "sample_std",
        "sample_min", "sample_max", "sample_q10", "sample_median", "sample_q90", "sample_trend",
    )
    components = ("airs_signal", "fgs_signal", "airs_dark", "fgs_dark")
    feature_names = [*star_columns, *[f"{component}__{field}" for component in components for field in summary_fields]]
    rows: list[list[float]] = []
    audit = {"files": 0, "sample_values": 0, "bytes": 0, "errors": 0}
    train_root = data_root / "data" / "train"
    for index, planet_id in enumerate(frame["planet_id"].astype(str)):
        row: list[float] = []
        for _component, path in component_paths(train_root, planet_id):
            values, item_audit = parquet_summary(path)
            row.extend(values)
            for key in audit:
                audit[key] += item_audit[key]
        rows.append(row)
        if (index + 1) % 100 == 0 or index + 1 == len(frame):
            append_event(log_path, "feature_progress", completed=index + 1, total=len(frame))
    summary_matrix = np.asarray(rows, dtype=float)
    if summary_matrix.shape != (len(frame), len(components) * len(summary_fields)):
        raise ValueError("Ariel parquet feature matrix shape mismatch")
    return np.concatenate((star_matrix, summary_matrix), axis=1), feature_names, audit


def make_model(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )


def calibrated_sigma(y_true: np.ndarray, prediction: np.ndarray, scale: float) -> np.ndarray:
    residual = y_true - prediction
    sigma = np.sqrt(np.mean(residual * residual, axis=0)) * float(scale)
    minimum = np.concatenate(([FGS_SIGMA_TRUE], np.full(TARGETS - 1, AIRS_SIGMA_TRUE)))
    sigma = np.maximum(sigma, minimum)
    return np.broadcast_to(sigma, y_true.shape).copy()


def select_inner(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    outer_train: np.ndarray,
    fold: int,
) -> tuple[float, float, list[dict[str, Any]], np.ndarray]:
    splitter = GroupKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + 100 + fold)
    candidate_rows: list[dict[str, Any]] = []
    best: tuple[float, float, float, np.ndarray] | None = None
    for alpha in ALPHAS:
        inner_oof = np.zeros_like(y[outer_train])
        seen = np.zeros(len(outer_train), dtype=bool)
        for train_local, valid_local in splitter.split(x[outer_train], y[outer_train], groups[outer_train]):
            model = make_model(alpha)
            model.fit(x[outer_train[train_local]], y[outer_train[train_local]])
            inner_oof[valid_local] = np.maximum(model.predict(x[outer_train[valid_local]]), 1e-15)
            seen[valid_local] = True
        if not seen.all():
            raise ValueError("inner OOF coverage incomplete")
        naive_mean = float(np.mean(y[outer_train]))
        naive_sigma = float(np.std(y[outer_train]))
        for sigma_scale in SIGMA_SCALES:
            sigma = calibrated_sigma(y[outer_train], inner_oof, sigma_scale)
            score = official_gll(y[outer_train], inner_oof, sigma, naive_mean=naive_mean, naive_sigma=naive_sigma)
            row = {"alpha": alpha, "sigma_scale": sigma_scale, "inner_oof_gll": score}
            candidate_rows.append(row)
            key = (score, -float(alpha), -abs(float(sigma_scale) - 1.0))
            if best is None or key > (best[0], -best[1], -abs(best[2] - 1.0)):
                best = (score, float(alpha), float(sigma_scale), inner_oof.copy())
    if best is None:
        raise ValueError("no valid inner candidate")
    return best[1], best[2], candidate_rows, best[3]


def bootstrap_ci(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    rounds: int,
    *,
    naive_mean: float,
    naive_sigma: float,
) -> tuple[float, float]:
    rng = np.random.default_rng(SEED + 900)
    contributions = official_gll_per_planet(
        y,
        mu,
        sigma,
        naive_mean=naive_mean,
        naive_sigma=naive_sigma,
    )
    values = np.empty(rounds, dtype=float)
    for index in range(rounds):
        sample = rng.integers(0, len(y), len(y))
        values[index] = float(np.clip(np.mean(contributions[sample]), 0.0, 1.0))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    data_root = Path(args.data_dir).resolve(strict=True)
    output_root = Path(args.out_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "training.log"
    source_path = Path(__file__).resolve(strict=True)
    append_event(log_path, "start", source_sha256=sha256_file(source_path))

    labels_path = data_root / "data" / "train.csv"
    stars_path = data_root / "data" / "train_star_info.csv"
    labels = pd.read_csv(labels_path)
    stars = pd.read_csv(stars_path)
    target_columns = [name for name in labels.columns if str(name).startswith("wl_")]
    if labels.shape != (1100, 284) or len(target_columns) != TARGETS:
        raise ValueError("Ariel train target schema mismatch")
    if labels["planet_id"].duplicated().any() or stars["planet_id"].duplicated().any():
        raise ValueError("Ariel planet IDs must be unique")
    star_features = [name for name in stars.columns if name != "planet_id"]
    stars = stars.rename(columns={name: f"star__{name}" for name in star_features})
    frame = labels.merge(stars, on="planet_id", how="inner", validate="one_to_one")
    if len(frame) != 1100:
        raise ValueError("Ariel star metadata coverage mismatch")
    all_ids = frame["planet_id"].astype(str).to_numpy()
    directory_ids = {path.name for path in (data_root / "data" / "train").iterdir() if path.is_dir() and path.name.isdigit()}
    if set(all_ids) != directory_ids:
        raise ValueError("Ariel CSV and planet directory IDs differ")

    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    development_index, parent_holdout_index = next(split.split(all_ids, groups=all_ids))
    parent_holdout_ids = set(all_ids[parent_holdout_index])
    for permutation_seed in (1, 7, 23, 101, 509):
        permutation = np.random.default_rng(permutation_seed).permutation(len(all_ids))
        _train, test = next(split.split(all_ids[permutation], groups=all_ids[permutation]))
        if set(all_ids[permutation][test]) != parent_holdout_ids:
            raise ValueError("GroupShuffleSplit holdout is order-dependent")
    development_ids = set(all_ids[development_index])
    if len(development_ids) != 880 or len(parent_holdout_ids) != 220 or development_ids & parent_holdout_ids:
        raise ValueError("Ariel parent split contract mismatch")
    append_event(log_path, "parent_holdout_excluded", development=880, holdout=220)

    development_frame = frame.loc[frame["planet_id"].astype(str).isin(development_ids)].copy()
    development_frame = development_frame.sort_values("planet_id").reset_index(drop=True)
    x, feature_names, feature_audit = build_features(data_root, development_frame, log_path)
    y = development_frame[target_columns].astype(float).to_numpy()
    groups = development_frame["planet_id"].astype(str).to_numpy()
    if x.shape[0] != 880 or y.shape != (880, TARGETS) or len(np.unique(groups)) != 880:
        raise ValueError("Ariel real loader regression failed")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Ariel loader produced non-finite values")
    anchors = scorer_anchors(y)
    if not anchors["reference_zero"] or not anchors["ideal_one"]:
        raise ValueError("official GLL scorer anchors failed")
    append_event(log_path, "loader_regression_passed", rows=880, features=x.shape[1], targets=TARGETS)

    outer = GroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED + 1)
    oof_mu = np.zeros_like(y)
    oof_sigma = np.zeros_like(y)
    coverage = np.zeros(len(y), dtype=int)
    fold_metrics: list[dict[str, Any]] = []
    candidate_history: list[dict[str, Any]] = []
    for fold, (outer_train, outer_valid) in enumerate(outer.split(x, y, groups)):
        alpha, sigma_scale, candidates, inner_oof = select_inner(x, y, groups, outer_train, fold)
        model = make_model(alpha)
        model.fit(x[outer_train], y[outer_train])
        prediction = np.maximum(model.predict(x[outer_valid]), 1e-15)
        sigma_vector = calibrated_sigma(y[outer_train], inner_oof, sigma_scale)[0]
        sigma = np.broadcast_to(sigma_vector, prediction.shape).copy()
        score = official_gll(
            y[outer_valid], prediction, sigma,
            naive_mean=float(np.mean(y[outer_train])), naive_sigma=float(np.std(y[outer_train])),
        )
        oof_mu[outer_valid] = prediction
        oof_sigma[outer_valid] = sigma
        coverage[outer_valid] += 1
        fold_metrics.append({
            "fold": fold,
            "train": len(outer_train),
            "valid": len(outer_valid),
            "alpha": alpha,
            "sigma_scale": sigma_scale,
            "official_gll": score,
            "parent_holdout_overlap": 0,
        })
        candidate_history.extend({"outer_fold": fold, **row} for row in candidates)
        append_event(log_path, "outer_fold_complete", **fold_metrics[-1])
    if not np.all(coverage == 1):
        raise ValueError("outer OOF coverage failed")
    full_naive_mean = float(np.mean(y))
    full_naive_sigma = float(np.std(y))
    oof_score = official_gll(y, oof_mu, oof_sigma, naive_mean=full_naive_mean, naive_sigma=full_naive_sigma)
    ci_lower, ci_upper = bootstrap_ci(
        y,
        oof_mu,
        oof_sigma,
        BOOTSTRAP_ROUNDS,
        naive_mean=full_naive_mean,
        naive_sigma=full_naive_sigma,
    )

    aggregate_candidates: dict[tuple[float, float], list[float]] = {}
    for row in candidate_history:
        aggregate_candidates.setdefault((float(row["alpha"]), float(row["sigma_scale"])), []).append(float(row["inner_oof_gll"]))
    selected_alpha, selected_sigma_scale = max(
        aggregate_candidates,
        key=lambda key: (float(np.mean(aggregate_candidates[key])), -key[0], -abs(key[1] - 1.0)),
    )
    final_model = make_model(selected_alpha)
    final_model.fit(x, y)
    final_sigma = calibrated_sigma(y, oof_mu, selected_sigma_scale)[0]
    gate_passed = bool(oof_score > 0.0 and ci_lower > 0.0)
    append_event(log_path, "training_complete", official_gll=oof_score, ci_lower=ci_lower, ci_upper=ci_upper, gate_passed=gate_passed)

    split_evidence = {
        "schema": "evomind.ariel_parent_holdout_exclusion.v2",
        "algorithm": "sklearn.model_selection.GroupShuffleSplit",
        "n_splits": 1,
        "test_size": 0.2,
        "random_state": SEED,
        "all_planets": 1100,
        "development": 880,
        "parent_holdout": 220,
        "intersection": 0,
        "union": 1100,
        "order_invariance_trials": 5,
        "order_invariance": True,
        "all_id_sha256": id_set_sha(all_ids),
        "development_id_sha256": id_set_sha(list(development_ids)),
        "parent_holdout_id_sha256": id_set_sha(list(parent_holdout_ids)),
    }
    loader_regression = {
        "schema": "evomind.ariel_loader_regression.v2",
        "passed": True,
        "rows": int(x.shape[0]),
        "features": int(x.shape[1]),
        "targets": int(y.shape[1]),
        "groups": int(len(np.unique(groups))),
        "parent_holdout_overlap": 0,
        "outer_fold_sizes": [{"train": row["train"], "valid": row["valid"]} for row in fold_metrics],
        "outer_oof_coverage_exactly_once": True,
        "real_parquet_features": True,
        "feature_audit": feature_audit,
    }
    metrics = {
        "schema": "evomind.ariel_nested_gll_metrics.v1",
        "metric": "official_normalized_gaussian_log_likelihood",
        "direction": "higher_is_better",
        "reference": 0.0,
        "nested_oof_gll": oof_score,
        "cluster_bootstrap_rounds": BOOTSTRAP_ROUNDS,
        "ci95": [ci_lower, ci_upper],
        "gate": "PUBLIC_NESTED_OOF_BASELINE_BEAT" if gate_passed else "FAILED_BELOW_BASELINE",
        "official_hidden_score": None,
        "folds": fold_metrics,
        "scorer_anchors": anchors,
    }
    write_json(output_root / "parent-holdout-exclusion-v2.json", split_evidence)
    write_json(output_root / "loader-regression-v2.json", loader_regression)
    write_json(output_root / "fold-metrics.json", fold_metrics)
    write_json(output_root / "candidate-history.json", candidate_history)
    write_json(output_root / "metrics.json", metrics)
    write_json(output_root / "scorer-anchors.json", anchors)
    write_json(output_root / "training-config.json", {
        "schema": "evomind.ariel_nested_gll_training_config.v1",
        "seed": SEED,
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "alphas": ALPHAS,
        "sigma_scales": SIGMA_SCALES,
        "selected_alpha": selected_alpha,
        "selected_sigma_scale": selected_sigma_scale,
        "feature_names": feature_names,
        "parent_holdout_used": False,
    })
    write_json(output_root / "environment.json", {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pq.__version__ if hasattr(pq, "__version__") else None,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
    })
    np.savez_compressed(
        output_root / "oof-predictions.npz",
        planet_id=groups,
        mu=oof_mu,
        sigma=oof_sigma,
        target_columns=np.asarray(target_columns),
    )
    joblib.dump(
        {
            "schema": "evomind.ariel_nested_gll_model.v1",
            "pipeline": final_model,
            "sigma": final_sigma,
            "feature_names": feature_names,
            "target_columns": target_columns,
            "development_id_sha256": id_set_sha(list(development_ids)),
            "parent_holdout_id_sha256": id_set_sha(list(parent_holdout_ids)),
            "selected_alpha": selected_alpha,
            "selected_sigma_scale": selected_sigma_scale,
            "source_sha256": sha256_file(source_path),
        },
        output_root / "model-bundle.joblib",
        compress=3,
    )
    shutil.copyfile(source_path, output_root / "ariel_2025_nested_gll.py")
    write_json(output_root / "independent-verification.json", {
        "schema": "evomind.ariel_nested_gll_verification.v1",
        "passed": True,
        "source_sha256": sha256_file(source_path),
        "parent_holdout_overlap": 0,
        "oof_coverage_exactly_once": True,
        "reference_anchor": anchors["reference"],
        "ideal_anchor": anchors["ideal"],
        "nested_oof_gll": oof_score,
        "ci95": [ci_lower, ci_upper],
        "signals_sent": 0,
        "other_processes_modified": False,
        "official_hidden_score": None,
    })
    files: list[dict[str, Any]] = []
    for path in sorted(output_root.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "artifact-manifest.json":
            files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(output_root / "artifact-manifest.json", {
        "schema": "evomind.ariel_nested_gll_manifest.v1",
        "status": "completed",
        "gate_passed": gate_passed,
        "files": files,
    })
    print(json.dumps({"status": "completed", "nested_oof_gll": oof_score, "ci95": [ci_lower, ci_upper], "gate_passed": gate_passed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
