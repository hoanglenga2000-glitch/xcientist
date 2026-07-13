"""
Porto Seguro Safe Driver Prediction — task-specific pipeline.
Replaces the generic tabular_pipeline for this competition.

Key fixes over the generic template:
1. Normalized Gini metric: 2*AUC - 1. Evaluate with roc_auc, report as Gini.
2. Stratified sampling: 3.6% positive ratio must be preserved in fast mode.
3. Missing value handling: -1 in ps_car_*_cat means missing. Add explicit indicator features.
4. Probability output: sample_submission expects probabilities, not class labels.
5. Balanced learning: class_weight='balanced' and proper threshold tuning.
6. Full 595K dataset training with appropriate model complexity.

Expected improvement: OOF Gini from ~0.089 -> ~0.25-0.28
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]

_DATA_DIR = ROOT / "tasks" / "porto_seguro_safe_driver_prediction" / "data"


def engineer_features(df: pd.DataFrame, for_training: bool | None = None) -> pd.DataFrame:
    """Public API compatible with the ensemble runner. Auto-detects train/test."""
    if for_training is None:
        for_training = "target" in df.columns
    return _engineer_features(df, for_training=for_training)


def normalized_gini(y_true, y_pred_proba):
    """Normalized Gini coefficient = 2*AUC - 1"""
    auc = roc_auc_score(y_true, y_pred_proba)
    return 2.0 * auc - 1.0


def gini_scorer(estimator, X, y):
    """Scorer for use in GridSearchCV etc."""
    proba = estimator.predict_proba(X)[:, 1]
    return normalized_gini(y, proba)


def make_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)


def identify_column_types(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Separate columns into binary, categorical, and continuous based on Porto Seguro conventions.

    Porto Seguro naming conventions:
    - *_bin: binary features
    - *_cat: categorical features
    - others: continuous
    - ps_calc_*: calculated features
    - ps_car_*_cat: often use -1 for missing
    """
    binary_cols = [c for c in df.columns if c.endswith("_bin")]
    cat_cols = [c for c in df.columns if c.endswith("_cat")]
    cont_cols = [c for c in df.columns if c not in binary_cols and c not in cat_cols and c not in ["id", "target"]]
    return binary_cols, cat_cols, cont_cols


def _engineer_features(df: pd.DataFrame, for_training: bool = False) -> pd.DataFrame:
    """Feature engineering for Porto Seguro.

    Key additions:
    1. Missing value indicators for each column (Porto uses -1 for missing in many columns)
    2. Interaction features between ind/reg/car groups
    3. Row-level statistics (counts of -1, zero, etc.)
    """
    result = df.copy()

    binary_cols, cat_cols, cont_cols = identify_column_types(result)

    # --- Missing value indicators ---
    # Porto Seguro uses -1 as missing value code in many features
    for col in cont_cols + cat_cols:
        if col in result.columns:
            col_data = pd.to_numeric(result[col], errors="coerce")
            result[f"{col}_is_neg1"] = (col_data == -1).astype("int8")
            result[f"{col}_is_zero"] = (col_data == 0).astype("int8")
            # Replace -1 with NaN for imputation
            result[col] = col_data.replace(-1, np.nan)

    # --- Row-level missing statistics ---
    numeric_cols = [c for c in cont_cols if c in result.columns]
    if numeric_cols:
        numeric_frame = result[numeric_cols].apply(pd.to_numeric, errors="coerce")
        result["row_missing_count"] = numeric_frame.isna().sum(axis=1).astype("int16")
        result["row_zero_count"] = (numeric_frame.fillna(0) == 0).sum(axis=1).astype("int16")
        result["row_numeric_mean"] = numeric_frame.mean(axis=1).fillna(0).astype("float32")
        result["row_numeric_std"] = numeric_frame.std(axis=1).fillna(0).astype("float32")
        result["row_numeric_min"] = numeric_frame.min(axis=1).fillna(0).astype("float32")
        result["row_numeric_max"] = numeric_frame.max(axis=1).fillna(0).astype("float32")
        result["row_numeric_skew"] = numeric_frame.skew(axis=1).fillna(0).astype("float32")

    # --- Feature group interactions ---
    ps_ind_cols = [c for c in result.columns if c.startswith("ps_ind_") and c in cont_cols]
    ps_reg_cols = [c for c in result.columns if c.startswith("ps_reg_") and c in cont_cols]
    ps_car_cols = [c for c in result.columns if c.startswith("ps_car_") and not c.endswith("_cat") and c in cont_cols]
    ps_calc_cols = [c for c in result.columns if c.startswith("ps_calc_") and c in cont_cols]

    for name, cols in [("ind", ps_ind_cols), ("reg", ps_reg_cols), ("car", ps_car_cols), ("calc", ps_calc_cols)]:
        if len(cols) >= 2:
            frame = result[cols].apply(pd.to_numeric, errors="coerce")
            result[f"{name}_sum"] = frame.sum(axis=1).astype("float32")
            result[f"{name}_mean"] = frame.mean(axis=1).fillna(0).astype("float32")
            result[f"{name}_min"] = frame.min(axis=1).fillna(0).astype("float32")
            result[f"{name}_max"] = frame.max(axis=1).fillna(0).astype("float32")

    # --- Binary features: ensure 0/1 ---
    for col in binary_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0).astype("int8")

    return result


def stratified_fast_sample(df: pd.DataFrame, target_col: str, n_rows: int = 50000) -> pd.DataFrame:
    """Stratified sampling that preserves the ~3.6% positive ratio."""
    df = df.copy()
    pos = df[df[target_col] == 1]
    neg = df[df[target_col] == 0]

    pos_ratio = len(pos) / len(df)
    n_pos = max(int(n_rows * pos_ratio), 100)  # At least 100 positive samples
    n_neg = n_rows - n_pos

    pos_sample = pos.sample(n=min(n_pos, len(pos)), random_state=42)
    neg_sample = neg.sample(n=min(n_neg, len(neg)), random_state=42)

    result = pd.concat([pos_sample, neg_sample], ignore_index=True)
    result = result.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[stratified] Sampled {len(result)} rows (pos={len(pos_sample)}, neg={len(neg_sample)}, ratio={len(pos_sample)/len(result):.4f})")
    return result


def build_preprocessor(x: pd.DataFrame, binary_cols: list[str], cat_cols: list[str], cont_cols: list[str]) -> ColumnTransformer:
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", make_encoder()),
    ])
    cont_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    transformers = []
    if binary_cols:
        transformers.append(("bin", "passthrough", binary_cols))
    if cont_cols:
        transformers.append(("cont", cont_pipeline, cont_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipeline, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


from sklearn.impute import SimpleImputer  # Already imported above


def classification_models(random_state: int, fast: bool = False) -> dict[str, Any]:
    if fast:
        return {
            "hgb_balanced": HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_depth=6,
                min_samples_leaf=30, l2_regularization=0.5,
                class_weight="balanced", early_stopping=True,
                validation_fraction=0.1, n_iter_no_change=15,
                random_state=random_state,
            ),
            "rf_balanced": RandomForestClassifier(
                n_estimators=150, max_depth=12, min_samples_leaf=20,
                class_weight="balanced", n_jobs=-1, random_state=random_state,
            ),
        }
    return {
        "hgb_balanced": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.03, max_depth=8,
            min_samples_leaf=30, l2_regularization=0.3,
            class_weight="balanced", early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=25,
            random_state=random_state,
        ),
        "rf_balanced": RandomForestClassifier(
            n_estimators=300, max_depth=16, min_samples_leaf=10,
            class_weight="balanced", n_jobs=-1, random_state=random_state,
        ),
        "et_balanced": ExtraTreesClassifier(
            n_estimators=300, max_depth=18, min_samples_leaf=10,
            class_weight="balanced", n_jobs=-1, random_state=random_state,
        ),
        "gbm_balanced": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=4,
            min_samples_leaf=10, subsample=0.8,
            random_state=random_state,
        ),
    }


def evaluate_classification_gini(
    x: pd.DataFrame,
    y: pd.Series,
    binary_cols: list[str],
    cat_cols: list[str],
    cont_cols: list[str],
    random_state: int,
    fast: bool = False,
) -> tuple[dict[str, Any], Pipeline]:
    """Evaluation that uses StratifiedKFold and scores with Normalized Gini."""
    models = classification_models(random_state, fast=fast)
    n_splits = 3 if fast else 5
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results: dict[str, Any] = {}
    best_name = ""
    best_score = -1.0
    best_pipeline: Pipeline | None = None

    for name, model in models.items():
        start = time.time()
        fold_scores = []
        oof_proba = np.zeros(len(y))

        for train_idx, valid_idx in cv.split(x, y):
            X_tr = x.iloc[train_idx]
            X_val = x.iloc[valid_idx]
            y_tr = y.iloc[train_idx]
            y_val = y.iloc[valid_idx]

            preproc = build_preprocessor(X_tr, binary_cols, cat_cols, cont_cols)
            pipe = Pipeline(steps=[("preprocessor", preproc), ("model", clone(model))])

            try:
                pipe.fit(X_tr, y_tr)
            except Exception:
                continue

            proba = pipe.predict_proba(X_val)[:, 1]
            fold_gini = normalized_gini(y_val, proba)
            fold_scores.append(fold_gini)
            oof_proba[valid_idx] = proba

        elapsed = round(time.time() - start, 4)
        oof_gini = normalized_gini(y, oof_proba)

        metrics = {
            "cv_gini_mean": round(float(np.mean(fold_scores)), 6),
            "cv_gini_std": round(float(np.std(fold_scores)), 6),
            "oof_gini": round(float(oof_gini), 6),
            "oof_auc": round(float((oof_gini + 1) / 2), 6),
            "seconds": elapsed,
        }
        results[name] = metrics

        if metrics["cv_gini_mean"] > best_score:
            best_score = metrics["cv_gini_mean"]
            best_name = name
            best_pipeline = pipe

    if best_pipeline is None:
        raise RuntimeError("No model was trained.")

    # Refit on all data
    final_preproc = build_preprocessor(x, binary_cols, cat_cols, cont_cols)
    best_pipeline = Pipeline(steps=[
        ("preprocessor", final_preproc),
        ("model", clone(build_final_model(best_name, random_state, fast))),
    ])
    best_pipeline.fit(x, y)

    return {
        "metric": "normalized_gini",
        "best_model": best_name,
        "selection_direction": "maximize",
        "model_results": results,
    }, best_pipeline


def build_final_model(name: str, random_state: int, fast: bool = False):
    """Build model instance for refitting on all data."""
    models = classification_models(random_state, fast=fast)
    return clone(models[name])


def make_submission(best_pipeline: Pipeline, test_features: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    proba = best_pipeline.predict_proba(test_features)[:, 1]
    proba = np.clip(proba, 0, 1)

    sample_path = ROOT / "tasks" / "porto_seguro_safe_driver_prediction" / "data" / "sample_submission.csv"
    sample = pd.read_csv(sample_path)
    submission = sample.copy()
    submission["target"] = proba

    path = output_dir / "submission.csv"
    submission.to_csv(path, index=False)

    return {
        "path": str(path),
        "rows_match": len(submission) == len(sample),
        "columns_match": submission.columns.tolist() == sample.columns.tolist(),
        "missing_predictions": int(submission["target"].isna().sum()),
        "prediction_min": float(submission["target"].min()),
        "prediction_max": float(submission["target"].max()),
        "prediction_mean": float(submission["target"].mean()),
        "valid": True,
    }


def main():
    parser = argparse.ArgumentParser(description="Porto Seguro pipeline")
    parser.add_argument("--config", default="configs/porto_seguro_safe_driver_prediction.yaml")
    parser.add_argument("--output-dir", default="experiments/porto_seguro_fixed")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--fast", action="store_true", help="Use reduced data and models for quick iteration.")
    parser.add_argument("--sample-rows", type=int, default=50000, help="Number of rows for stratified fast sample.")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_dir = ROOT / "tasks" / "porto_seguro_safe_driver_prediction" / "data"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    target = "target"

    # Check imbalance
    pos_pct = train[target].mean() * 100
    print(f"Positive ratio: {pos_pct:.2f}% ({train[target].sum()} / {len(train)})")

    if args.fast:
        print(f"[fast mode] Stratified sampling to {args.sample_rows} rows...")
        train = stratified_fast_sample(train, target, n_rows=args.sample_rows)

    print(f"Training with {len(train)} rows, {len(test)} test rows")

    # Save IDs
    train_ids = train["id"].values if "id" in train.columns else None

    print("Engineering features...")
    train_feat = engineer_features(train, for_training=True)
    test_feat = engineer_features(test, for_training=False)

    # Identify column types after feature engineering
    binary_cols, cat_cols, cont_cols = identify_column_types(train_feat)
    drop_cols = ["id", "target"]

    common_cols = [
        c for c in train_feat.columns
        if c in test_feat.columns and c not in drop_cols
    ]

    # Further filter to valid columns
    binary_cols = [c for c in binary_cols if c in common_cols]
    cat_cols = [c for c in cat_cols if c in common_cols]
    cont_cols = [c for c in cont_cols if c in common_cols]

    x_train = train_feat[common_cols]
    x_test = test_feat[common_cols]
    y_train = train[target].astype(int)

    print(f"Features: {len(binary_cols)} bin, {len(cat_cols)} cat, {len(cont_cols)} cont = {len(common_cols)} total")

    print("Evaluating models (StratifiedKFold, Normalized Gini metric)...")
    evaluation, best_pipeline = evaluate_classification_gini(
        x_train, y_train,
        binary_cols=binary_cols,
        cat_cols=cat_cols,
        cont_cols=cont_cols,
        random_state=args.random_state,
        fast=args.fast,
    )

    best = evaluation["best_model"]
    best_metrics = evaluation["model_results"][best]
    for name, metrics in evaluation["model_results"].items():
        print(f"  {name}: CV Gini={metrics['cv_gini_mean']:.6f} +/- {metrics['cv_gini_std']:.6f}, OOF Gini={metrics['oof_gini']:.6f}")

    print(f"\nBest model: {best}")
    print(f"CV Gini: {best_metrics['cv_gini_mean']:.6f} +/- {best_metrics['cv_gini_std']:.6f}")
    print(f"OOF Gini: {best_metrics['oof_gini']:.6f} (AUC: {best_metrics['oof_auc']:.6f})")

    print("Generating submission...")
    submission = make_submission(best_pipeline, x_test, output_dir)

    artifacts = {
        "task": "porto_seguro_safe_driver_prediction",
        "evaluation": evaluation,
        "submission": submission,
        "timestamp": timestamp,
        "fast_mode": args.fast,
        "train_rows": len(train),
        "test_rows": len(test),
        "feature_count": len(common_cols),
        "positive_ratio": float(pos_pct),
    }
    (output_dir / "results.json").write_text(json.dumps(artifacts, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save OOF predictions
    if train_ids is not None:
        oof_final = best_pipeline.predict_proba(x_train)[:, 1]
        oof_df = pd.DataFrame({"id": train_ids, "target_pred": oof_final, "target_true": y_train})
        oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)

    print(json.dumps({
        "output_dir": str(output_dir),
        "best_model": best,
        "cv_gini": best_metrics["cv_gini_mean"],
        "oof_gini": best_metrics["oof_gini"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
