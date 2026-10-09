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
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer


SEED = 20260830
BOOTSTRAP_ROUNDS = 500


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def stable_split(ids: pd.Series, salt: str, fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    ranked = sorted(
        ids.astype(str).tolist(),
        key=lambda item: hashlib.sha256(f"{salt}|{item}".encode()).hexdigest(),
    )
    holdout_ids = set(ranked[: max(1, int(round(len(ranked) * fraction)))])
    holdout = ids.astype(str).isin(holdout_ids).to_numpy()
    train = ~holdout
    if not train.any() or not holdout.any():
        raise ValueError("empty split")
    return train, holdout


def engineered_features(frame: pd.DataFrame, feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    raw = frame[feature_names].astype(float).to_numpy()
    parts = [raw, np.sign(raw) * np.log1p(np.abs(raw)), raw * raw]
    names = [*feature_names, *[f"logabs_{name}" for name in feature_names], *[f"sq_{name}" for name in feature_names]]
    for left in range(len(feature_names)):
        for right in range(left + 1, len(feature_names)):
            denominator = np.where(np.abs(raw[:, right]) < 1e-12, np.nan, raw[:, right])
            parts.append((raw[:, left] / denominator)[:, None])
            names.append(f"ratio_{feature_names[left]}_{feature_names[right]}")
    return np.concatenate(parts, axis=1), names


def parquet_metadata_features(data_root: Path, planet_ids: pd.Series) -> tuple[np.ndarray, list[str], dict[str, int]]:
    import pyarrow.parquet as pq

    kinds = ("airs_signal", "fgs_signal", "airs_other", "fgs_other", "other")
    fields = (
        "file_count",
        "file_bytes",
        "rows",
        "row_groups",
        "columns",
        "statistics_count",
        "minimum_mean",
        "maximum_mean",
        "range_mean",
        "sample_values",
        "sample_mean",
        "sample_std",
        "sample_minimum",
        "sample_maximum",
        "sample_q10",
        "sample_median",
        "sample_q90",
        "sample_trend",
    )
    feature_names = [f"parquet_{kind}_{field}" for kind in kinds for field in fields]
    fgs_fields = (
        "count",
        "mean",
        "std",
        "minimum",
        "maximum",
        "q01",
        "q05",
        "q10",
        "q25",
        "median",
        "q75",
        "q90",
        "q95",
        "q99",
        *[f"segment_{index:02d}" for index in range(32)],
    )
    feature_names.extend(f"fgs_full_{field}" for field in fgs_fields)
    rows: list[list[float]] = []
    opened = 0
    errors = 0
    sampled_files = 0
    sampled_values_total = 0
    fgs_full_files_read = 0
    fgs_full_values_materialized = 0
    fgs_full_file_bytes_read = 0

    def kind_of(name: str) -> str:
        folded = name.casefold()
        if "airs" in folded and "signal" in folded:
            return "airs_signal"
        if "fgs" in folded and "signal" in folded:
            return "fgs_signal"
        if "airs" in folded:
            return "airs_other"
        if "fgs" in folded:
            return "fgs_other"
        return "other"

    for planet_id in planet_ids.astype(str):
        directory = data_root / "data" / "train" / planet_id
        candidates = sorted(directory.glob("*.parquet"))
        candidates.sort(key=lambda path: ("signal_0" not in path.name.casefold(), path.name.casefold()))
        selected = candidates[:8]
        accumulators: dict[str, dict[str, float | list[float]]] = {
            kind: {
                "file_count": 0.0,
                "file_bytes": 0.0,
                "rows": 0.0,
                "row_groups": 0.0,
                "columns": 0.0,
                "statistics_count": 0.0,
                "minimums": [],
                "maximums": [],
                "ranges": [],
                "samples": [],
                "trends": [],
            }
            for kind in kinds
        }
        for path in selected:
            kind = kind_of(path.name)
            item = accumulators[kind]
            try:
                parquet_file = pq.ParquetFile(path)
                metadata = parquet_file.metadata
                opened += 1
                item["file_count"] = float(item["file_count"]) + 1.0
                item["file_bytes"] = float(item["file_bytes"]) + float(path.stat().st_size)
                item["rows"] = float(item["rows"]) + float(metadata.num_rows)
                item["row_groups"] = float(item["row_groups"]) + float(metadata.num_row_groups)
                item["columns"] = float(item["columns"]) + float(metadata.num_columns)
                for row_group_index in range(metadata.num_row_groups):
                    row_group = metadata.row_group(row_group_index)
                    for column_index in range(row_group.num_columns):
                        stats = row_group.column(column_index).statistics
                        if stats is None or not stats.has_min_max:
                            continue
                        minimum = stats.min
                        maximum = stats.max
                        if isinstance(minimum, (int, float)) and isinstance(maximum, (int, float)):
                            minimum_value = float(minimum)
                            maximum_value = float(maximum)
                            if math.isfinite(minimum_value) and math.isfinite(maximum_value):
                                item["statistics_count"] = float(item["statistics_count"]) + 1.0
                                assert isinstance(item["minimums"], list)
                                assert isinstance(item["maximums"], list)
                                assert isinstance(item["ranges"], list)
                                item["minimums"].append(minimum_value)
                                item["maximums"].append(maximum_value)
                                item["ranges"].append(maximum_value - minimum_value)
                if "signal_0" in path.name.casefold():
                    names = list(parquet_file.schema.names)
                    if names:
                        indices = sorted(set(np.linspace(0, len(names) - 1, min(8, len(names)), dtype=int)))
                        selected_names = [names[index] for index in indices]
                        batch = next(
                            parquet_file.iter_batches(batch_size=256, columns=selected_names),
                            None,
                        )
                        if batch is not None:
                            numeric = batch.to_pandas().select_dtypes(include=[np.number])
                            values = numeric.to_numpy(dtype=float, copy=False)
                            values = values[np.isfinite(values)]
                            if values.size:
                                sampled_files += 1
                                sampled_values_total += int(values.size)
                                assert isinstance(item["samples"], list)
                                assert isinstance(item["trends"], list)
                                item["samples"].extend(values.tolist())
                                first = numeric.iloc[0].to_numpy(dtype=float)
                                last = numeric.iloc[-1].to_numpy(dtype=float)
                                finite = np.isfinite(first) & np.isfinite(last)
                                if finite.any():
                                    item["trends"].append(float(np.mean(last[finite] - first[finite])))
            except (OSError, ValueError, TypeError):
                errors += 1
        feature_row: list[float] = []
        for kind in kinds:
            item = accumulators[kind]
            minimums = item["minimums"] if isinstance(item["minimums"], list) else []
            maximums = item["maximums"] if isinstance(item["maximums"], list) else []
            ranges = item["ranges"] if isinstance(item["ranges"], list) else []
            samples = np.asarray(item["samples"] if isinstance(item["samples"], list) else [], dtype=float)
            trends = item["trends"] if isinstance(item["trends"], list) else []
            feature_row.extend(
                [
                    float(item["file_count"]),
                    float(item["file_bytes"]),
                    float(item["rows"]),
                    float(item["row_groups"]),
                    float(item["columns"]),
                    float(item["statistics_count"]),
                    float(np.mean(minimums)) if minimums else 0.0,
                    float(np.mean(maximums)) if maximums else 0.0,
                    float(np.mean(ranges)) if ranges else 0.0,
                    float(samples.size),
                    float(np.mean(samples)) if samples.size else 0.0,
                    float(np.std(samples)) if samples.size else 0.0,
                    float(np.min(samples)) if samples.size else 0.0,
                    float(np.max(samples)) if samples.size else 0.0,
                    float(np.quantile(samples, 0.1)) if samples.size else 0.0,
                    float(np.quantile(samples, 0.5)) if samples.size else 0.0,
                    float(np.quantile(samples, 0.9)) if samples.size else 0.0,
                    float(np.mean(trends)) if trends else 0.0,
                ]
            )
        fgs_candidates = sorted(
            path
            for path in candidates
            if "fgs1" in path.name.casefold() and "signal_0" in path.name.casefold()
        )
        fgs_values = np.asarray([], dtype=float)
        if fgs_candidates:
            fgs_path = fgs_candidates[0]
            try:
                table = pq.read_table(fgs_path)
                numeric = table.to_pandas().select_dtypes(include=[np.number])
                if not numeric.empty:
                    fgs_values = numeric.iloc[:, 0].to_numpy(dtype=float, copy=False)
                    fgs_values = fgs_values[np.isfinite(fgs_values)]
                    if fgs_values.size > 1_000_000:
                        indices = np.linspace(0, fgs_values.size - 1, 1_000_000, dtype=int)
                        fgs_values = fgs_values[indices]
                    fgs_full_files_read += 1
                    fgs_full_values_materialized += int(fgs_values.size)
                    fgs_full_file_bytes_read += int(fgs_path.stat().st_size)
            except (OSError, ValueError, TypeError):
                errors += 1
        if fgs_values.size:
            segments = [
                float(np.mean(segment)) if segment.size else 0.0
                for segment in np.array_split(fgs_values, 32)
            ]
            feature_row.extend(
                [
                    float(fgs_values.size),
                    float(np.mean(fgs_values)),
                    float(np.std(fgs_values)),
                    float(np.min(fgs_values)),
                    float(np.max(fgs_values)),
                    float(np.quantile(fgs_values, 0.01)),
                    float(np.quantile(fgs_values, 0.05)),
                    float(np.quantile(fgs_values, 0.10)),
                    float(np.quantile(fgs_values, 0.25)),
                    float(np.quantile(fgs_values, 0.50)),
                    float(np.quantile(fgs_values, 0.75)),
                    float(np.quantile(fgs_values, 0.90)),
                    float(np.quantile(fgs_values, 0.95)),
                    float(np.quantile(fgs_values, 0.99)),
                    *segments,
                ]
            )
        else:
            feature_row.extend([0.0] * len(fgs_fields))
        rows.append(feature_row)
    return np.asarray(rows, dtype=float), feature_names, {
        "parquet_metadata_files_opened": opened,
        "parquet_metadata_errors": errors,
        "parquet_data_pages_read": sampled_files,
        "bounded_signal_batches_read": sampled_files,
        "bounded_signal_values_materialized": sampled_values_total,
        "bounded_signal_value_bytes_materialized": sampled_values_total * 8,
        "maximum_rows_per_signal_file": 256,
        "maximum_columns_per_signal_file": 8,
        "fgs_full_files_read": fgs_full_files_read,
        "fgs_full_values_materialized": fgs_full_values_materialized,
        "fgs_full_file_bytes_read": fgs_full_file_bytes_read,
        "airs_full_files_read": 0,
    }


def per_planet_normalized_rmse(y_true: np.ndarray, y_pred: np.ndarray, scale: np.ndarray) -> np.ndarray:
    normalized = (y_pred - y_true) / np.maximum(scale[None, :], 1e-12)
    return np.sqrt(np.mean(normalized * normalized, axis=1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    started = utc_now()
    data_root = Path(args.data_dir).resolve(strict=True)
    output_root = Path(args.out_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    train_path = data_root / "data" / "train.csv"
    star_path = data_root / "data" / "train_star_info.csv"
    test_star_path = data_root / "data" / "test_star_info.csv"
    for path in (train_path, star_path, test_star_path):
        if not path.is_file():
            raise ValueError(f"required Ariel table is missing: {path.name}")

    source_path = Path(__file__).resolve(strict=True)
    source_sha = sha256_file(source_path)
    labels = pd.read_csv(train_path)
    stars = pd.read_csv(star_path)
    test_schema = pd.read_csv(test_star_path, nrows=5)
    wavelength_columns = [name for name in labels.columns if str(name).startswith("wl_")]
    if "planet_id" not in labels or "planet_id" not in stars or len(wavelength_columns) != 283:
        raise ValueError("Ariel target schema mismatch")
    if labels["planet_id"].duplicated().any() or stars["planet_id"].duplicated().any():
        raise ValueError("duplicate planet id")
    frame = labels.merge(stars, on="planet_id", how="inner", validate="one_to_one")
    if len(frame) != len(labels):
        raise ValueError("star metadata does not cover every training planet")
    if frame[wavelength_columns].isna().any().any():
        raise ValueError("missing wavelength target")
    feature_names = [name for name in stars.columns if name != "planet_id"]
    if not feature_names or any(name not in test_schema.columns for name in feature_names):
        raise ValueError("Ariel star feature schema mismatch")

    _previous_development, previous_holdout = stable_split(frame["planet_id"], f"{SEED}|outer", 0.2)
    generation2_pool = np.flatnonzero(~previous_holdout)
    _generation2_development, generation2_holdout_local = stable_split(
        frame.loc[generation2_pool, "planet_id"].reset_index(drop=True),
        f"{SEED}|outer-generation-2",
        0.25,
    )
    generation2_holdout = np.zeros(len(frame), dtype=bool)
    generation2_holdout[generation2_pool[generation2_holdout_local]] = True
    generation3_pool = np.flatnonzero(~(previous_holdout | generation2_holdout))
    _generation3_development, generation3_holdout_local = stable_split(
        frame.loc[generation3_pool, "planet_id"].reset_index(drop=True),
        f"{SEED}|outer-generation-3",
        1.0 / 3.0,
    )
    generation3_holdout = np.zeros(len(frame), dtype=bool)
    generation3_holdout[generation3_pool[generation3_holdout_local]] = True
    previous_three_holdouts = previous_holdout | generation2_holdout | generation3_holdout
    generation4_pool = np.flatnonzero(~previous_three_holdouts)
    _generation4_development, generation4_holdout_local = stable_split(
        frame.loc[generation4_pool, "planet_id"].reset_index(drop=True),
        f"{SEED}|outer-generation-4",
        0.5,
    )
    generation4_holdout = np.zeros(len(frame), dtype=bool)
    generation4_holdout[generation4_pool[generation4_holdout_local]] = True
    previous_holdouts = previous_three_holdouts | generation4_holdout
    generation5_pool = np.flatnonzero(~previous_holdouts)
    if len(generation5_pool) < 5:
        raise ValueError("no untouched generation-5 holdout remains")
    holdout = np.zeros(len(frame), dtype=bool)
    holdout[generation5_pool] = True
    development = ~holdout
    inner_train_local, inner_val_local = stable_split(
        frame.loc[development, "planet_id"].reset_index(drop=True), f"{SEED}|inner-generation-5", 0.2
    )
    if set(frame.loc[development, "planet_id"]) & set(frame.loc[holdout, "planet_id"]):
        raise ValueError("planet leakage")
    if np.any(previous_holdouts & holdout):
        raise ValueError("generation-5 holdout overlaps a previously observed holdout")

    imputer = SimpleImputer(strategy="median")
    raw_all = imputer.fit_transform(frame[feature_names].astype(float))
    engineered_all, engineered_names = engineered_features(
        pd.DataFrame(raw_all, columns=feature_names), feature_names
    )
    parquet_features, parquet_feature_names, parquet_audit = parquet_metadata_features(
        data_root, frame["planet_id"]
    )
    if parquet_features.shape[0] != len(frame) or parquet_features.shape[1] != len(parquet_feature_names):
        raise ValueError("parquet metadata feature shape mismatch")
    engineered_all = np.concatenate([engineered_all, parquet_features], axis=1)
    engineered_names.extend(parquet_feature_names)
    y_all = frame[wavelength_columns].astype(float).to_numpy()
    y_development = y_all[development]
    scale = np.std(y_development, axis=0, ddof=1)
    scale = np.where(scale < 1e-12, 1.0, scale)

    development_indices = np.flatnonzero(development)
    inner_train = development_indices[inner_train_local]
    inner_val = development_indices[inner_val_local]
    baseline_inner = RandomForestRegressor(
        n_estimators=240,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=SEED,
    )
    baseline_inner.fit(raw_all[inner_train], y_all[inner_train])
    baseline_inner_prediction = np.asarray(baseline_inner.predict(raw_all[inner_val]), dtype=float)
    baseline_inner_score = float(
        per_planet_normalized_rmse(
            y_all[inner_val], baseline_inner_prediction, scale
        ).mean()
    )

    candidate_history: list[dict[str, Any]] = []
    inner_target_mean = np.mean(y_all[inner_train], axis=0)
    inner_target_scale = np.std(y_all[inner_train], axis=0, ddof=1)
    inner_target_scale = np.where(inner_target_scale < 1e-12, 1.0, inner_target_scale)
    inner_normalized_targets = (y_all[inner_train] - inner_target_mean) / inner_target_scale
    for index, requested_components in enumerate((8, 16, 32)):
        component_count = min(requested_components, len(inner_train) - 1, y_all.shape[1])
        pca_inner = PCA(n_components=component_count, random_state=SEED + index)
        target_scores = pca_inner.fit_transform(inner_normalized_targets)
        model = ExtraTreesRegressor(
            n_estimators=180,
            min_samples_leaf=2,
            max_features=0.8,
            n_jobs=-1,
            random_state=SEED + index + 1,
        )
        model.fit(engineered_all[inner_train], target_scores)
        reconstructed = inner_target_mean + inner_target_scale * pca_inner.inverse_transform(
            model.predict(engineered_all[inner_val])
        )
        for blend_weight in (0.25, 0.5, 0.75, 1.0):
            blended = blend_weight * reconstructed + (1.0 - blend_weight) * baseline_inner_prediction
            score = float(per_planet_normalized_rmse(y_all[inner_val], blended, scale).mean())
            candidate_history.append(
                {
                    "candidate_id": f"pca-{component_count}-blend-{blend_weight}",
                    "pca_components": component_count,
                    "blend_weight": blend_weight,
                    "inner_score": score,
                }
            )
    selected = min(candidate_history, key=lambda item: (item["inner_score"], item["candidate_id"]))

    baseline = RandomForestRegressor(
        n_estimators=360,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=SEED,
    )
    candidate = ExtraTreesRegressor(
        n_estimators=320,
        min_samples_leaf=2,
        max_features=0.8,
        n_jobs=-1,
        random_state=SEED + 100,
    )
    baseline.fit(raw_all[development], y_all[development])
    target_mean = np.mean(y_all[development], axis=0)
    target_scale = np.std(y_all[development], axis=0, ddof=1)
    target_scale = np.where(target_scale < 1e-12, 1.0, target_scale)
    pca = PCA(n_components=int(selected["pca_components"]), random_state=SEED + 100)
    target_scores = pca.fit_transform((y_all[development] - target_mean) / target_scale)
    candidate.fit(engineered_all[development], target_scores)
    baseline_prediction = np.asarray(baseline.predict(raw_all[holdout]), dtype=float)
    pca_prediction = target_mean + target_scale * pca.inverse_transform(
        candidate.predict(engineered_all[holdout])
    )
    candidate_prediction = (
        float(selected["blend_weight"]) * pca_prediction
        + (1.0 - float(selected["blend_weight"])) * baseline_prediction
    )
    y_holdout = y_all[holdout]
    baseline_per_planet = per_planet_normalized_rmse(y_holdout, baseline_prediction, scale)
    candidate_per_planet = per_planet_normalized_rmse(y_holdout, candidate_prediction, scale)
    baseline_score = float(baseline_per_planet.mean())
    candidate_score = float(candidate_per_planet.mean())

    rng = np.random.default_rng(SEED + 200)
    differences: list[float] = []
    for _ in range(BOOTSTRAP_ROUNDS):
        indices = rng.integers(0, len(y_holdout), len(y_holdout))
        differences.append(float(candidate_per_planet[indices].mean() - baseline_per_planet[indices].mean()))
    difference = np.asarray(differences)
    standard_error = float(difference.std(ddof=1))
    ci_lower = float(np.quantile(difference, 0.025))
    ci_upper = float(np.quantile(difference, 0.975))
    minimum_margin = max(0.01 * baseline_score, standard_error)
    gate_passed = bool(candidate_score < baseline_score and ci_upper < -minimum_margin)

    task_contract = {
        "schema": "evomind.ariel_2025.task_contract.v2",
        "task": "283-output exoplanet transmission spectrum regression",
        "features": feature_names,
        "targets": wavelength_columns,
        "metric": "mean per-planet RMSE normalized by development wavelength standard deviation",
        "direction": "lower_is_better",
        "entity_group": "planet_id",
        "signal_parquet_used": "full_fgs1_plus_bounded_airs_statistics",
        "hidden_test_labels_used": False,
    }
    baseline_evidence = {
        "schema": "evomind.ariel_2025.baseline_evidence.v2",
        "classification": "PUBLICLY_REPRODUCIBLE_STRONG_BASELINE",
        "method": "multi-output RandomForest on official train_star_info features",
        "implementation": "solution.py",
        "implementation_sha256": source_sha,
        "train_sha256": sha256_file(train_path),
        "star_info_sha256": sha256_file(star_path),
        "metric": task_contract["metric"],
        "reproduction_value": baseline_score,
    }
    split_manifest = {
        "schema": "evomind.ariel_2025.split_manifest.v2",
        "seed": SEED,
        "development_planets": int(development.sum()),
        "holdout_planets": int(holdout.sum()),
        "planet_overlap": 0,
        "generation": 5,
        "previous_holdout_planets": int(previous_holdouts.sum()),
        "previous_holdout_overlap": 0,
        "previous_holdouts_promoted_to_development": True,
        "development_planet_sha256": sha256_bytes("\n".join(map(str, sorted(frame.loc[development, "planet_id"]))).encode()),
        "holdout_planet_sha256": sha256_bytes("\n".join(map(str, sorted(frame.loc[holdout, "planet_id"]))).encode()),
        "holdout_blind_for_selection": True,
    }
    dataset_audit = {
        "schema": "evomind.ariel_2025.dataset_audit.v2",
        "training_planets": len(frame),
        "wavelength_targets": len(wavelength_columns),
        "star_features": feature_names,
        "missing_star_values": {name: int(frame[name].isna().sum()) for name in feature_names},
        "missing_targets": int(frame[wavelength_columns].isna().sum().sum()),
        "test_schema_rows_read": len(test_schema),
        "test_labels_read": False,
        "signal_parquet_bytes_read": None,
        **parquet_audit,
    }
    metrics = {
        "schema": "evomind.ariel_2025.metrics.v2",
        "baseline": baseline_score,
        "candidate": candidate_score,
        "absolute_improvement": baseline_score - candidate_score,
        "relative_improvement": (baseline_score - candidate_score) / max(baseline_score, 1e-12),
        "inner_baseline": baseline_inner_score,
        "candidate_history": candidate_history,
        "bootstrap": {
            "rounds": BOOTSTRAP_ROUNDS,
            "difference_candidate_minus_baseline_mean": float(difference.mean()),
            "standard_error": standard_error,
            "ci95_lower": ci_lower,
            "ci95_upper": ci_upper,
        },
    }
    comparison = {
        "schema": "evomind.ariel_2025.candidate_vs_baseline.v2",
        "baseline": baseline_score,
        "candidate": candidate_score,
        "minimum_margin": minimum_margin,
        "difference_candidate_minus_baseline_ci95_upper": ci_upper,
        "gate_passed": gate_passed,
    }
    holdout_frame = pd.DataFrame(
        {
            "planet_id": frame.loc[holdout, "planet_id"].astype(str).to_numpy(),
            "baseline_per_planet_normalized_rmse": baseline_per_planet,
            "candidate_per_planet_normalized_rmse": candidate_per_planet,
        }
    )
    recomputed_baseline = float(holdout_frame["baseline_per_planet_normalized_rmse"].mean())
    recomputed_candidate = float(holdout_frame["candidate_per_planet_normalized_rmse"].mean())
    independent = {
        "schema": "evomind.ariel_2025.independent_verification.v2",
        "holdout_planets": len(holdout_frame),
        "baseline_recomputed": recomputed_baseline,
        "candidate_recomputed": recomputed_candidate,
        "baseline_match": abs(recomputed_baseline - baseline_score) < 1e-12,
        "candidate_match": abs(recomputed_candidate - candidate_score) < 1e-12,
        "planet_overlap": 0,
        "test_labels_used": False,
        "gate_passed": gate_passed,
    }

    shutil.copyfile(source_path, output_root / "solution.py")
    write_json(output_root / "task-contract-v2.json", task_contract)
    write_json(output_root / "baseline-evidence.json", baseline_evidence)
    write_json(output_root / "baseline-reproduction.json", {**baseline_evidence, "completed_at_utc": utc_now()})
    write_json(output_root / "dataset-audit.json", dataset_audit)
    write_json(output_root / "split-manifest.json", split_manifest)
    write_json(
        output_root / "training-config.json",
        {
            "schema": "evomind.ariel_2025.training_config.v2",
            "seed": SEED,
            "baseline": baseline_evidence["method"],
            "candidate": "target-normalized PCA plus ExtraTrees using engineered star, full FGS1 and bounded AIRS statistics, blended with the frozen RandomForest baseline",
            "selected_candidate": selected,
            "bootstrap_rounds": BOOTSTRAP_ROUNDS,
            "single_gpu_solution": True,
            "external_submission": False,
        },
    )
    with (output_root / "candidate-history.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for item in candidate_history:
            handle.write(json.dumps(item, separators=(",", ":")) + "\n")
    with (output_root / "training.log").open("w", encoding="utf-8", newline="\n") as handle:
        for event in (
            {"at_utc": started, "event": "start", "source_sha256": source_sha},
            {"at_utc": utc_now(), "event": "baseline_complete", "metric": baseline_score},
            {"at_utc": utc_now(), "event": "candidate_complete", "metric": candidate_score},
            {"at_utc": utc_now(), "event": "independent_verification", "gate_passed": gate_passed},
        ):
            handle.write(json.dumps(event, separators=(",", ":")) + "\n")
    holdout_frame.to_csv(output_root / "holdout-metrics.csv", index=False)
    write_json(output_root / "metrics.json", metrics)
    write_json(output_root / "candidate-vs-baseline.json", comparison)
    write_json(output_root / "independent-verification.json", independent)
    write_json(
        output_root / "environment-lock.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    )
    joblib.dump(
        {
            "imputer": imputer,
            "baseline": baseline,
            "candidate": candidate,
            "pca": pca,
            "target_mean": target_mean,
            "target_scale": target_scale,
            "blend_weight": float(selected["blend_weight"]),
            "engineered_names": engineered_names,
        },
        output_root / "model-bundle.joblib",
        compress=3,
    )
    write_json(
        output_root / "retrospective-memory.json",
        {
            "schema": "evomind.retrospective_memory.v2",
            "baseline": baseline_evidence["method"],
            "candidate": "generation-5 target-PCA ExtraTrees blend with full FGS1 statistics",
            "gate_passed": gate_passed,
            "signal_parquet_used": "full_fgs1_plus_bounded_airs_statistics",
            "next_step_if_failed": "Request an external blind evaluator or new labeled holdout; all internal planet holdouts have been consumed exactly once.",
            "memory_writeback_allowed": gate_passed,
        },
    )

    files: list[dict[str, Any]] = []
    for path in sorted(output_root.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name not in {"artifact-manifest.json", "artifact-manifest-receipt.json"}:
            files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "schema": "evomind.ariel_2025.artifact_manifest.v2",
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "source_sha256": source_sha,
        "exit_code": 0,
        "gate_passed": gate_passed,
        "files": files,
    }
    write_json(output_root / "artifact-manifest.json", manifest)
    write_json(
        output_root / "artifact-manifest-receipt.json",
        {
            "artifact": "artifact-manifest.json",
            "bytes": (output_root / "artifact-manifest.json").stat().st_size,
            "sha256": sha256_file(output_root / "artifact-manifest.json"),
        },
    )
    print(json.dumps({"status": "completed", **comparison}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
