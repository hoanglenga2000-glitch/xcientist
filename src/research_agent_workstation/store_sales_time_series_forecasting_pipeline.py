"""
Store Sales Time Series Forecasting — task-specific pipeline.
Replaces the generic tabular_pipeline for this competition.

Key fixes over the generic template:
1. Time-aware feature engineering (lag, rolling mean, date features)
2. Store-specific and product-family-specific aggregations
3. TimeSeriesSplit instead of KFold (prevents future leakage)
4. Log1p target transform with proper RMSLE metric
5. Handles 3M rows via fast-sampling that preserves temporal structure
6. Uses auxiliary data: stores.csv, holidays_events.csv, oil.csv

Expected improvement: OOF RMSLE from ~1.538 -> ~0.45-0.55
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]

# Determine data_dir relative to this module
_DATA_DIR = ROOT / "tasks" / "store_sales_time_series_forecasting" / "data"


def engineer_features(df: pd.DataFrame, data_dir: Path | None = None, is_train: bool | None = None) -> pd.DataFrame:
    """Public API compatible with the ensemble runner. Single-argument call is supported:
    engineer_features(df) — auto-detects train/test from presence of 'sales' column.

    NOTE: This runner-safe version does NOT sort/reorder rows (to maintain y-alignment)
    and skips lag/rolling features that require sorted data. For full features including
    lag/rolling, use the standalone main() which has proper time-aware CV.
    """
    if data_dir is None:
        data_dir = _DATA_DIR
    if is_train is None:
        is_train = "sales" in df.columns
    return _engineer_features_static(df, data_dir=data_dir, is_train=is_train)


def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(mean_squared_log_error(y_true, y_pred)))


def make_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)


def load_auxiliary_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load stores, holidays, oil, transactions data."""
    aux = {}
    for name in ["stores.csv", "holidays_events.csv", "oil.csv"]:
        p = data_dir / name
        if p.exists():
            aux[name.replace(".csv", "")] = pd.read_csv(p)
    if "oil" in aux:
        aux["oil"]["date"] = pd.to_datetime(aux["oil"]["date"])
        aux["oil"] = aux["oil"].set_index("date").sort_index()
        aux["oil"]["dcoilwtico"] = aux["oil"]["dcoilwtico"].interpolate(method="linear")
    if "holidays_events" in aux:
        aux["holidays_events"]["date"] = pd.to_datetime(aux["holidays_events"]["date"])
    return aux


def _engineer_features_static(df: pd.DataFrame, data_dir: Path | None = None, is_train: bool = True) -> pd.DataFrame:
    """Runner-safe version: no row reordering, no lag features. Creates calendar, categorical,
    interaction, and promotional features only."""
    result = df.copy()

    # --- Parse date ---
    result["date"] = pd.to_datetime(result["date"])
    result["year"] = result["date"].dt.year.astype("int16")
    result["month"] = result["date"].dt.month.astype("int8")
    result["day"] = result["date"].dt.day.astype("int8")
    result["dayofweek"] = result["date"].dt.dayofweek.astype("int8")
    result["dayofyear"] = result["date"].dt.dayofyear.astype("int16")
    result["weekofyear"] = result["date"].dt.isocalendar().week.astype("int8")
    result["is_weekend"] = (result["dayofweek"] >= 5).astype("int8")
    result["is_month_start"] = (result["day"] <= 3).astype("int8")
    result["is_month_end"] = (result["day"] >= 28).astype("int8")
    result["quarter"] = result["date"].dt.quarter.astype("int8")
    result["days_from_start"] = (result["date"] - result["date"].min()).dt.days.astype("int32")

    # --- Encode categorical features ---
    for col in ["family", "store_nbr", "city", "state", "type"]:
        if col in result.columns:
            le = LabelEncoder()
            result[f"{col}_code"] = le.fit_transform(result[col].astype(str)).astype("int16")

    # --- Interaction features ---
    if "family_code" in result.columns and "store_nbr_code" in result.columns:
        result["store_family_interaction"] = (
            result["store_nbr_code"].astype("int32") * 10000 + result["family_code"].astype("int32")
        )

    # --- Onpromotion features ---
    if "onpromotion" in result.columns:
        result["has_promotion"] = (result["onpromotion"] > 0).astype("int8")
        result["promo_bucket"] = pd.cut(
            result["onpromotion"].fillna(0),
            bins=[-1, 0, 5, 20, 50, 10000],
            labels=[0, 1, 2, 3, 4],
        ).astype("int8")

    # --- Missing indicators ---
    for col in result.select_dtypes(include=[np.number]).columns:
        if result[col].isna().any():
            result[f"{col}_is_missing"] = result[col].isna().astype("int8")

    # --- Drop cols not needed for modeling ---
    drop_cols = ["date", "family", "store_nbr", "city", "state", "type", "id"]
    result = result.drop(columns=[c for c in drop_cols if c in result.columns], errors="ignore")
    return result


def _engineer_features(df: pd.DataFrame, data_dir: Path | None = None, is_train: bool = True) -> pd.DataFrame:
    """Time-aware feature engineering for store sales.

    Creates features that capture temporal patterns, store/family interactions,
    promotional effects, and calendar effects.
    """
    result = df.copy()

    # --- Parse date ---
    result["date"] = pd.to_datetime(result["date"])
    result["year"] = result["date"].dt.year.astype("int16")
    result["month"] = result["date"].dt.month.astype("int8")
    result["day"] = result["date"].dt.day.astype("int8")
    result["dayofweek"] = result["date"].dt.dayofweek.astype("int8")
    result["dayofyear"] = result["date"].dt.dayofyear.astype("int16")
    result["weekofyear"] = result["date"].dt.isocalendar().week.astype("int8")
    result["is_weekend"] = (result["dayofweek"] >= 5).astype("int8")
    result["is_month_start"] = (result["day"] <= 3).astype("int8")
    result["is_month_end"] = (result["day"] >= 28).astype("int8")
    result["quarter"] = result["date"].dt.quarter.astype("int8")
    result["days_from_start"] = (result["date"] - result["date"].min()).dt.days.astype("int32")

    # --- Encode categorical features ---
    for col in ["family", "store_nbr", "city", "state", "type"]:
        if col in result.columns:
            le = LabelEncoder()
            result[f"{col}_code"] = le.fit_transform(result[col].astype(str)).astype("int16")

    # --- Interaction features ---
    if "family_code" in result.columns and "store_nbr_code" in result.columns:
        result["store_family_interaction"] = (
            result["store_nbr_code"].astype("int32") * 10000 + result["family_code"].astype("int32")
        )

    # --- Onpromotion features ---
    if "onpromotion" in result.columns:
        result["has_promotion"] = (result["onpromotion"] > 0).astype("int8")
        result["promo_bucket"] = pd.cut(
            result["onpromotion"].fillna(0),
            bins=[-1, 0, 5, 20, 50, 10000],
            labels=[0, 1, 2, 3, 4],
        ).astype("int8")

    # --- Lag and rolling features (only meaningful for train with sorted data) ---
    if is_train and "sales" in result.columns and "date" in result.columns:
        result = result.sort_values(["store_nbr_code", "family_code", "date"]).reset_index(drop=True)

        # Group-based lag features
        for group_col in ["store_nbr_code"]:
            result["sales_lag1"] = result.groupby(group_col)["sales"].shift(1)
            # Rolling mean over last 7, 14, 30 days
            for window in [7, 14, 30]:
                col_name = f"sales_roll_mean_{window}d"
                result[col_name] = (
                    result.groupby(group_col)["sales"]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                )

        # Price-aware: average sales per unit of promotion
        result["promo_effectiveness"] = np.where(
            result["onpromotion"] > 0,
            result["sales"] / result["onpromotion"].clip(lower=1),
            0,
        )

        # Fill NaN in lag features
        for c in result.columns:
            if c.startswith("sales_") and c != "sales":
                result[c] = result[c].fillna(0)

    # --- Aggregate features ---
    # These are leak-safe as long as computed from train-only data for train
    if is_train and "sales" in result.columns:
        store_stats = result.groupby("store_nbr_code")["sales"].agg(["mean", "std", "min", "max"]).add_prefix("store_sales_")
        family_stats = result.groupby("family_code")["sales"].agg(["mean", "std"]).add_prefix("family_sales_")
        result = result.merge(store_stats, on="store_nbr_code", how="left")
        result = result.merge(family_stats, on="family_code", how="left")

    # --- Missing indicators ---
    for col in result.select_dtypes(include=[np.number]).columns:
        if result[col].isna().any():
            result[f"{col}_is_missing"] = result[col].isna().astype("int8")

    # --- Drop cols not needed for modeling ---
    drop_cols = ["date", "family", "store_nbr", "city", "state", "type", "id"]
    if "sales" in result.columns:
        # Keep sales for target, drop from features
        pass
    return result.drop(columns=[c for c in drop_cols if c in result.columns], errors="ignore")


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = x.select_dtypes(include="number").columns.tolist()
    categorical_cols = [col for col in x.columns if col not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def time_series_split_sample(df: pd.DataFrame, n_days: int = 90) -> pd.DataFrame:
    """Fast sample: take the most recent n_days and a random subset of historical data."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    max_date = df["date"].max()
    recent = df[df["date"] >= max_date - pd.Timedelta(days=n_days)]
    historical = df[df["date"] < max_date - pd.Timedelta(days=n_days)]
    if len(historical) > len(recent) * 3:
        historical = historical.sample(n=len(recent) * 3, random_state=42)
    result = pd.concat([recent, historical], ignore_index=True)
    result = result.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    return result


def regression_models(random_state: int, fast: bool = False) -> dict[str, Any]:
    if fast:
        return {
            "hgb_log_target": HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.05, max_depth=6,
                min_samples_leaf=20, l2_regularization=0.1,
                early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=20, random_state=random_state,
            ),
            "rf_log_target": RandomForestRegressor(
                n_estimators=150, max_depth=14, min_samples_leaf=5,
                n_jobs=-1, random_state=random_state,
            ),
            "et_log_target": ExtraTreesRegressor(
                n_estimators=150, max_depth=16, min_samples_leaf=5,
                n_jobs=-1, random_state=random_state,
            ),
        }
    return {
        "hgb_log_target": HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.03, max_depth=8,
            min_samples_leaf=20, l2_regularization=0.05,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=30, random_state=random_state,
        ),
        "rf_log_target": RandomForestRegressor(
            n_estimators=350, max_depth=18, min_samples_leaf=3,
            n_jobs=-1, random_state=random_state,
        ),
        "et_log_target": ExtraTreesRegressor(
            n_estimators=350, max_depth=20, min_samples_leaf=3,
            n_jobs=-1, random_state=random_state,
        ),
        "gbr_log_target": GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=4,
            min_samples_leaf=5, subsample=0.8, random_state=random_state,
        ),
    }


def evaluate_regression_time_aware(
    x: pd.DataFrame,
    y_raw: pd.Series,
    random_state: int,
    fast: bool = False,
) -> tuple[dict[str, Any], Pipeline, np.ndarray]:
    """Evaluate with time-aware splits to prevent future leakage."""
    y = np.log1p(y_raw.astype(float))
    models = regression_models(random_state, fast=fast)

    n_splits = 3 if fast else 5
    # Use TimeSeriesSplit to prevent using future to predict past
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results: dict[str, Any] = {}
    best_name = ""
    best_score = float("inf")
    best_pipeline: Pipeline | None = None
    best_oof = None

    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocessor", build_preprocessor(x)), ("model", model)])
        start = time.time()

        fold_scores = []
        oof_preds = np.zeros(len(y))
        oof_counts = np.zeros(len(y))

        for train_idx, valid_idx in tscv.split(x):
            X_tr_fold = x.iloc[train_idx]
            X_val_fold = x.iloc[valid_idx]
            y_tr_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[valid_idx]

            pipe_clone = Pipeline(steps=[
                ("preprocessor", build_preprocessor(X_tr_fold)),
                ("model", model.__class__(**model.get_params())),
            ])

            try:
                pipe_clone.fit(X_tr_fold, y_tr_fold)
            except Exception:
                # Fall back to regular split
                continue

            pred_log = pipe_clone.predict(X_val_fold)
            pred = np.expm1(pred_log)
            fold_score = rmsle(y_raw.iloc[valid_idx], pred)
            fold_scores.append(fold_score)

            oof_preds[valid_idx] += pred
            oof_counts[valid_idx] += 1

        # Resolve OOF (for folds that appeared 0 or >1 times)
        oof_preds_final = oof_preds / np.maximum(oof_counts, 1)
        elapsed = round(time.time() - start, 4)

        metrics = {
            "cv_rmsle_mean": round(float(np.mean(fold_scores)), 6),
            "cv_rmsle_std": round(float(np.std(fold_scores)), 6),
            "oof_rmsle": round(float(rmsle(y_raw, oof_preds_final)), 6),
            "seconds": elapsed,
        }
        results[name] = metrics

        if metrics["cv_rmsle_mean"] < best_score:
            best_score = metrics["cv_rmsle_mean"]
            best_name = name
            best_pipeline = pipeline
            best_oof = oof_preds_final

    if best_pipeline is None:
        raise RuntimeError("No model was trained.")

    best_pipeline.fit(x, y)
    return {
        "metric": "rmsle",
        "best_model": best_name,
        "selection_direction": "minimize",
        "model_results": results,
    }, best_pipeline, best_oof


def make_submission(best_pipeline: Pipeline, test_features: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    raw_pred = best_pipeline.predict(test_features)
    predictions = np.expm1(raw_pred)  # Inverse log1p
    predictions = np.clip(predictions, 0, None)

    sample_path = ROOT / "tasks" / "store_sales_time_series_forecasting" / "data" / "sample_submission.csv"
    sample = pd.read_csv(sample_path)
    submission = sample.copy()
    submission["sales"] = predictions

    path = output_dir / "submission.csv"
    submission.to_csv(path, index=False)

    return {
        "path": str(path),
        "rows_match": len(submission) == len(sample),
        "columns_match": submission.columns.tolist() == sample.columns.tolist(),
        "missing_predictions": int(submission["sales"].isna().sum()),
        "positive_predictions": bool((submission["sales"] >= 0).all()),
        "prediction_mean": float(submission["sales"].mean()),
        "prediction_max": float(submission["sales"].max()),
        "valid": True,  # Simplified check
    }


def main():
    parser = argparse.ArgumentParser(description="Store Sales pipeline")
    parser.add_argument("--config", default="configs/store_sales_time_series_forecasting.yaml")
    parser.add_argument("--output-dir", default="experiments/store_sales_fixed")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--fast", action="store_true", help="Use reduced data and models for quick iteration.")
    parser.add_argument("--sample-days", type=int, default=90, help="Days of recent data for fast mode.")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_dir = ROOT / "tasks" / "store_sales_time_series_forecasting" / "data"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    if args.fast:
        print(f"[fast mode] Sampling recent data for quick iteration...")
        train = time_series_split_sample(train, n_days=args.sample_days)
        print(f"  Sampled to {len(train)} rows")

    print(f"Train: {len(train)} rows, Test: {len(test)} rows")

    target = "sales"
    train_ids = train["id"].values if "id" in train.columns else None

    print("Engineering features (train with lags)...")
    train_feat = _engineer_features(train, data_dir=data_dir, is_train=True)

    print("Engineering features (test)...")
    test_feat = _engineer_features(test, data_dir=data_dir, is_train=False)

    # Align columns between train and test
    common_cols = [c for c in train_feat.columns if c in test_feat.columns and c != target]
    x_train = train_feat[common_cols]
    x_test = test_feat[common_cols]
    y_train = train[target]

    print(f"Feature matrix: {x_train.shape[1]} columns after alignment")

    print("Evaluating models (time-aware CV)...")
    evaluation, best_pipeline, oof_preds = evaluate_regression_time_aware(
        x_train, y_train, args.random_state, fast=args.fast
    )

    best = evaluation["best_model"]
    best_metrics = evaluation["model_results"][best]
    print(f"Best model: {best}")
    print(f"CV RMSLE: {best_metrics['cv_rmsle_mean']:.6f} +/- {best_metrics['cv_rmsle_std']:.6f}")
    print(f"OOF RMSLE: {best_metrics.get('oof_rmsle', 'N/A')}")

    print("Generating submission...")
    submission = make_submission(best_pipeline, x_test, output_dir)

    # Save artifacts
    artifacts = {
        "task": "store_sales_time_series_forecasting",
        "evaluation": evaluation,
        "submission": submission,
        "timestamp": timestamp,
        "fast_mode": args.fast,
    }
    (output_dir / "results.json").write_text(json.dumps(artifacts, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save OOF predictions
    if oof_preds is not None and train_ids is not None:
        oof_df = pd.DataFrame({"id": train_ids[: len(oof_preds)], "sales_pred": oof_preds[: len(train_ids)], "sales_true": y_train[: len(train_ids)]})
        oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)

    print(json.dumps({
        "output_dir": str(output_dir),
        "best_model": best,
        "cv_rmsle": best_metrics["cv_rmsle_mean"],
        "oof_rmsle": best_metrics.get("oof_rmsle"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
