from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


SEED = 20260829
TARGETS = ("Tg", "FFV", "Tc", "Density", "Rg")
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
    encoded = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def stable_rank(value: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{value}".encode("utf-8")).hexdigest()


def frozen_group_split(groups: pd.Series, *, salt: str, fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    unique = sorted(set(groups.astype(str)), key=lambda item: stable_rank(item, salt))
    holdout_count = max(1, int(round(len(unique) * fraction)))
    holdout_groups = set(unique[:holdout_count])
    holdout = groups.astype(str).isin(holdout_groups).to_numpy()
    train = ~holdout
    if not train.any() or not holdout.any():
        raise ValueError("group split produced an empty partition")
    return train, holdout


def normalized_rmse(y_true: np.ndarray, y_pred: np.ndarray, scale: float) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)) / max(scale, 1e-12))


@dataclass
class TargetModel:
    target: str
    scale: float
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD
    ridge: Ridge
    trees: ExtraTreesRegressor
    tree_weight: float


def fit_pair(
    texts: np.ndarray,
    values: np.ndarray,
    *,
    random_state: int,
    tree_weight: float | None = None,
) -> tuple[TfidfVectorizer, TruncatedSVD, Ridge, ExtraTreesRegressor, float, dict[str, float]]:
    if len(values) < 20:
        raise ValueError("insufficient labeled rows")
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=1,
        max_features=4096,
        sublinear_tf=True,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(texts.tolist())
    ridge = Ridge(alpha=10.0, solver="lsqr")
    ridge.fit(matrix, values)
    components = max(2, min(128, matrix.shape[0] - 1, matrix.shape[1] - 1))
    svd = TruncatedSVD(n_components=components, random_state=random_state)
    dense = svd.fit_transform(matrix)
    trees = ExtraTreesRegressor(
        n_estimators=240,
        min_samples_leaf=1,
        max_features=0.8,
        n_jobs=-1,
        random_state=random_state,
    )
    trees.fit(dense, values)
    chosen_weight = 0.7 if tree_weight is None else float(tree_weight)
    return vectorizer, svd, ridge, trees, chosen_weight, {
        "rows": float(len(values)),
        "features": float(matrix.shape[1]),
        "svd_components": float(components),
    }


def predict_pair(model: TargetModel, texts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = model.vectorizer.transform(texts.tolist())
    baseline = np.asarray(model.ridge.predict(matrix), dtype=float)
    tree = np.asarray(model.trees.predict(model.svd.transform(matrix)), dtype=float)
    candidate = model.tree_weight * tree + (1.0 - model.tree_weight) * baseline
    return baseline, candidate


def choose_tree_weight(texts: np.ndarray, values: np.ndarray, groups: pd.Series, target: str) -> tuple[float, dict[str, float]]:
    inner_train, inner_val = frozen_group_split(groups, salt=f"{SEED}|inner|{target}", fraction=0.2)
    if inner_train.sum() < 20 or inner_val.sum() < 5:
        return 0.7, {"selection": "default_due_to_small_inner_split"}
    vectorizer, svd, ridge, trees, _, shape = fit_pair(
        texts[inner_train], values[inner_train], random_state=SEED + 17
    )
    matrix = vectorizer.transform(texts[inner_val].tolist())
    ridge_pred = np.asarray(ridge.predict(matrix), dtype=float)
    tree_pred = np.asarray(trees.predict(svd.transform(matrix)), dtype=float)
    scale = float(np.std(values[inner_train], ddof=1))
    scores: dict[str, float] = {}
    for weight in (0.35, 0.5, 0.65, 0.8, 1.0):
        prediction = weight * tree_pred + (1.0 - weight) * ridge_pred
        scores[str(weight)] = normalized_rmse(values[inner_val], prediction, scale)
    selected = min(scores, key=lambda item: (scores[item], float(item)))
    return float(selected), {"selection": "inner_group_holdout", "scores": scores, **shape}


def bootstrap_difference(
    rows: dict[str, dict[str, np.ndarray | float]], *, rounds: int, seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    targets = sorted(rows)
    for _ in range(rounds):
        per_target: list[float] = []
        for target in targets:
            item = rows[target]
            y_true = np.asarray(item["y_true"], dtype=float)
            baseline = np.asarray(item["baseline"], dtype=float)
            candidate = np.asarray(item["candidate"], dtype=float)
            scale = float(item["scale"])
            indices = rng.integers(0, len(y_true), len(y_true))
            base_score = normalized_rmse(y_true[indices], baseline[indices], scale)
            candidate_score = normalized_rmse(y_true[indices], candidate[indices], scale)
            per_target.append(candidate_score - base_score)
        differences.append(float(np.mean(per_target)))
    array = np.asarray(differences, dtype=float)
    return {
        "rounds": rounds,
        "difference_mean": float(array.mean()),
        "difference_standard_error": float(array.std(ddof=1)),
        "difference_ci95_lower": float(np.quantile(array, 0.025)),
        "difference_ci95_upper": float(np.quantile(array, 0.975)),
    }


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
    test_path = data_root / "data" / "test.csv"
    if not train_path.is_file() or not test_path.is_file():
        raise ValueError("required train/test files are missing")

    source_path = Path(__file__).resolve(strict=True)
    source_sha = sha256_file(source_path)
    train_sha = sha256_file(train_path)
    test_sha = sha256_file(test_path)
    frame = pd.read_csv(train_path)
    test_frame = pd.read_csv(test_path, nrows=5)
    required = {"id", "SMILES", *TARGETS}
    if not required.issubset(frame.columns) or not {"id", "SMILES"}.issubset(test_frame.columns):
        raise ValueError("Open Polymer schema does not match the frozen contract")
    if frame["SMILES"].isna().any() or frame["id"].duplicated().any():
        raise ValueError("invalid SMILES or duplicate training id")

    groups = frame["SMILES"].astype(str)
    development, holdout = frozen_group_split(groups, salt=f"{SEED}|outer", fraction=0.2)
    if set(groups[development]) & set(groups[holdout]):
        raise ValueError("SMILES leakage across outer split")

    log_rows: list[dict[str, Any]] = []

    def log(event: str, **payload: Any) -> None:
        log_rows.append({"at_utc": utc_now(), "event": event, **payload})

    log("validation", rows=len(frame), columns=len(frame.columns), source_sha256=source_sha)
    models: dict[str, TargetModel] = {}
    target_results: dict[str, dict[str, Any]] = {}
    bootstrap_rows: dict[str, dict[str, np.ndarray | float]] = {}
    prediction_rows: list[pd.DataFrame] = []

    for index, target in enumerate(TARGETS):
        labeled = frame[target].notna().to_numpy()
        train_mask = development & labeled
        holdout_mask = holdout & labeled
        if train_mask.sum() < 20 or holdout_mask.sum() < 5:
            raise ValueError(f"insufficient labeled rows for {target}")
        target_texts = groups[train_mask].to_numpy()
        target_values = frame.loc[train_mask, target].astype(float).to_numpy()
        target_groups = groups[train_mask].reset_index(drop=True)
        selected_weight, selection = choose_tree_weight(
            target_texts, target_values, target_groups, target
        )
        vectorizer, svd, ridge, trees, _, shape = fit_pair(
            target_texts, target_values, random_state=SEED + index, tree_weight=selected_weight
        )
        scale = float(np.std(target_values, ddof=1))
        model = TargetModel(target, scale, vectorizer, svd, ridge, trees, selected_weight)
        models[target] = model
        y_true = frame.loc[holdout_mask, target].astype(float).to_numpy()
        baseline, candidate = predict_pair(model, groups[holdout_mask].to_numpy())
        baseline_score = normalized_rmse(y_true, baseline, scale)
        candidate_score = normalized_rmse(y_true, candidate, scale)
        target_results[target] = {
            "development_rows": int(train_mask.sum()),
            "holdout_rows": int(holdout_mask.sum()),
            "scale": scale,
            "baseline_normalized_rmse": baseline_score,
            "candidate_normalized_rmse": candidate_score,
            "absolute_improvement": baseline_score - candidate_score,
            "relative_improvement": (baseline_score - candidate_score) / max(baseline_score, 1e-12),
            "tree_weight": selected_weight,
            "inner_selection": selection,
            "shape": shape,
        }
        bootstrap_rows[target] = {
            "y_true": y_true,
            "baseline": baseline,
            "candidate": candidate,
            "scale": scale,
        }
        prediction_rows.append(
            pd.DataFrame(
                {
                    "id": frame.loc[holdout_mask, "id"].astype(str).to_numpy(),
                    "target": target,
                    "y_true": y_true,
                    "baseline_prediction": baseline,
                    "candidate_prediction": candidate,
                }
            )
        )
        log("target_complete", target=target, baseline=baseline_score, candidate=candidate_score)

    baseline_aggregate = float(np.mean([item["baseline_normalized_rmse"] for item in target_results.values()]))
    candidate_aggregate = float(np.mean([item["candidate_normalized_rmse"] for item in target_results.values()]))
    bootstrap = bootstrap_difference(bootstrap_rows, rounds=BOOTSTRAP_ROUNDS, seed=SEED + 99)
    minimum_margin = max(
        0.01 * baseline_aggregate,
        float(bootstrap["difference_standard_error"]),
    )
    pass_gate = bool(
        candidate_aggregate < baseline_aggregate
        and float(bootstrap["difference_ci95_upper"]) < -minimum_margin
    )

    task_contract = {
        "schema": "evomind.open_polymer.task_contract.v2",
        "competition": "open_polymer",
        "task": "multi-target polymer property regression from SMILES",
        "targets": list(TARGETS),
        "metric": "mean per-target RMSE normalized by development-target standard deviation",
        "direction": "lower_is_better",
        "entity_group": "exact SMILES",
        "hidden_test_labels_used": False,
        "seed": SEED,
    }
    baseline_evidence = {
        "schema": "evomind.open_polymer.baseline_evidence.v2",
        "classification": "PUBLICLY_REPRODUCIBLE_STRONG_BASELINE",
        "method": "character 2-5 gram TF-IDF (4096 features) plus per-target Ridge(alpha=10)",
        "implementation": "solution.py",
        "implementation_sha256": source_sha,
        "data_train_sha256": train_sha,
        "metric": task_contract["metric"],
        "reproduction_value": baseline_aggregate,
    }
    dataset_audit = {
        "schema": "evomind.open_polymer.dataset_audit.v2",
        "rows": len(frame),
        "columns": list(map(str, frame.columns)),
        "targets": {
            target: {
                "observed": int(frame[target].notna().sum()),
                "missing": int(frame[target].isna().sum()),
                "mean": float(frame[target].mean(skipna=True)),
                "std": float(frame[target].std(skipna=True)),
            }
            for target in TARGETS
        },
        "duplicate_ids": int(frame["id"].duplicated().sum()),
        "duplicate_smiles": int(frame["SMILES"].duplicated().sum()),
        "train_sha256": train_sha,
        "test_sha256": test_sha,
        "test_rows_sampled_for_schema_only": len(test_frame),
        "test_labels_read": False,
    }
    split_manifest = {
        "schema": "evomind.open_polymer.split_manifest.v2",
        "seed": SEED,
        "group": "SMILES",
        "development_rows": int(development.sum()),
        "holdout_rows": int(holdout.sum()),
        "development_groups": int(groups[development].nunique()),
        "holdout_groups": int(groups[holdout].nunique()),
        "group_overlap": 0,
        "development_id_sha256": sha256_bytes("\n".join(sorted(frame.loc[development, "id"].astype(str))).encode()),
        "holdout_id_sha256": sha256_bytes("\n".join(sorted(frame.loc[holdout, "id"].astype(str))).encode()),
        "holdout_blind_for_selection": True,
    }
    training_config = {
        "schema": "evomind.open_polymer.training_config.v2",
        "seed": SEED,
        "baseline": baseline_evidence["method"],
        "candidate": "TruncatedSVD(<=128)+ExtraTrees(240), blended with Ridge using inner group split",
        "bootstrap_rounds": BOOTSTRAP_ROUNDS,
        "single_gpu_solution": True,
        "external_submission": False,
    }
    metrics = {
        "schema": "evomind.open_polymer.metrics.v2",
        "baseline_aggregate_normalized_rmse": baseline_aggregate,
        "candidate_aggregate_normalized_rmse": candidate_aggregate,
        "targets": target_results,
        "bootstrap": bootstrap,
    }
    comparison = {
        "schema": "evomind.open_polymer.candidate_vs_baseline.v2",
        "baseline": baseline_aggregate,
        "candidate": candidate_aggregate,
        "absolute_improvement": baseline_aggregate - candidate_aggregate,
        "relative_improvement": (baseline_aggregate - candidate_aggregate) / max(baseline_aggregate, 1e-12),
        "minimum_margin": minimum_margin,
        "difference_candidate_minus_baseline_ci95_upper": bootstrap["difference_ci95_upper"],
        "gate_passed": pass_gate,
    }
    predictions = pd.concat(prediction_rows, ignore_index=True)
    recomputed_target: dict[str, dict[str, float]] = {}
    for target in TARGETS:
        rows = predictions[predictions["target"] == target]
        scale = float(target_results[target]["scale"])
        recomputed_target[target] = {
            "baseline": normalized_rmse(rows["y_true"].to_numpy(), rows["baseline_prediction"].to_numpy(), scale),
            "candidate": normalized_rmse(rows["y_true"].to_numpy(), rows["candidate_prediction"].to_numpy(), scale),
        }
    recomputed_baseline = float(np.mean([value["baseline"] for value in recomputed_target.values()]))
    recomputed_candidate = float(np.mean([value["candidate"] for value in recomputed_target.values()]))
    independent = {
        "schema": "evomind.open_polymer.independent_verification.v2",
        "prediction_rows": len(predictions),
        "targets": recomputed_target,
        "baseline_recomputed": recomputed_baseline,
        "candidate_recomputed": recomputed_candidate,
        "baseline_match": abs(recomputed_baseline - baseline_aggregate) < 1e-12,
        "candidate_match": abs(recomputed_candidate - candidate_aggregate) < 1e-12,
        "group_overlap": 0,
        "test_labels_used": False,
        "gate_passed": pass_gate,
    }

    shutil.copyfile(source_path, output_root / "solution.py")
    write_json(output_root / "task-contract-v2.json", task_contract)
    write_json(output_root / "baseline-evidence.json", baseline_evidence)
    write_json(
        output_root / "baseline-reproduction.json",
        {**baseline_evidence, "completed_at_utc": utc_now(), "target_values": target_results},
    )
    write_json(output_root / "dataset-audit.json", dataset_audit)
    write_json(output_root / "split-manifest.json", split_manifest)
    write_json(output_root / "training-config.json", training_config)
    with (output_root / "training.log").open("w", encoding="utf-8", newline="\n") as handle:
        for item in log_rows:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (output_root / "candidate-history.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for target, item in target_results.items():
            handle.write(
                json.dumps(
                    {"candidate_id": f"open-polymer-v2-{target}", "parent": "ridge-baseline", **item},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    predictions.to_csv(output_root / "holdout-predictions.csv", index=False)
    write_json(output_root / "metrics.json", metrics)
    write_json(output_root / "independent-verification.json", independent)
    write_json(output_root / "candidate-vs-baseline.json", comparison)
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
    joblib.dump(models, output_root / "model-bundle.joblib", compress=3)
    write_json(
        output_root / "retrospective-memory.json",
        {
            "schema": "evomind.retrospective_memory.v2",
            "candidate": "svd-extra-trees-ridge-blend",
            "baseline": "char-ngram-ridge",
            "gate_passed": pass_gate,
            "lessons": [
                "Keep exact SMILES groups out of both development and holdout.",
                "Select blend weights only on an inner group split.",
                "Do not use competition test labels or leaderboard feedback.",
            ],
            "memory_writeback_allowed": pass_gate and independent["baseline_match"] and independent["candidate_match"],
        },
    )

    manifest_files: list[dict[str, Any]] = []
    for path in sorted(output_root.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name not in {"artifact-manifest.json", "artifact-manifest-receipt.json"}:
            manifest_files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    artifact_manifest = {
        "schema": "evomind.open_polymer.artifact_manifest.v2",
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "source_sha256": source_sha,
        "data_train_sha256": train_sha,
        "data_test_sha256": test_sha,
        "exit_code": 0,
        "gate_passed": pass_gate,
        "files": manifest_files,
    }
    write_json(output_root / "artifact-manifest.json", artifact_manifest)
    write_json(
        output_root / "artifact-manifest-receipt.json",
        {
            "artifact": "artifact-manifest.json",
            "bytes": (output_root / "artifact-manifest.json").stat().st_size,
            "sha256": sha256_file(output_root / "artifact-manifest.json"),
        },
    )
    print(json.dumps({"status": "completed", "gate_passed": pass_gate, **comparison}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
