"""
Fix bike_sharing_demand Kaggle submission.

ROOT CAUSE:
  The datetime column was treated as a categorical string and one-hot encoded,
  creating ~10886 binary features (one per unique training datetime).
  In 5-fold shuffled CV, train/test folds share many datetimes -> RMSLE 0.154 (good).
  But Kaggle test set has ZERO overlapping datetimes -> all datetime features = 0
  -> model predicts nearly constant value (~357) -> RMSLE 1.919 (12x worse).

FIX:
  1. Extract time-based numeric features from datetime (hour, day, month, year, dayofweek)
  2. Drop the raw datetime string column from features
  3. Re-train ensemble with proper features
  4. Generate corrected submission.csv
"""

import json
import math
import time
import warnings
from datetime import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(r"D:\桌面\codex\科研港科技")
DATA_DIR = ROOT / "tasks" / "bike_sharing_demand" / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_DIR = ROOT / "experiments" / "bike_sharing_demand" / "fix_datetime_leak"
EXISTING_SUB = ROOT / "experiments" / "bike_sharing_demand" / "wr_2026-06-24T21-11-31.153454_19104660" / "submission.csv"

RANDOM_STATE = 42
N_FOLDS = 5
RIDGE_ALPHA = 18.0

# ── Helpers ────────────────────────────────────────────────────────────────

def rmsle(y_true, y_pred):
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return float(math.sqrt(mean_squared_log_error(y_true, y_pred)))


def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract numeric and categorical time features from datetime column."""
    result = df.copy()
    dt_series = pd.to_datetime(result["datetime"])
    result["hour"] = dt_series.dt.hour.astype(int)
    result["day"] = dt_series.dt.day.astype(int)
    result["month"] = dt_series.dt.month.astype(int)
    result["year"] = dt_series.dt.year.astype(int)
    result["dayofweek"] = dt_series.dt.dayofweek.astype(int)
    result["is_weekend"] = (result["dayofweek"] >= 5).astype(int)
    # Cyclical encoding for hour (peak hours ~8am and ~5pm)
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24.0)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24.0)
    # Cyclical encoding for month (seasonality)
    result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12.0)
    result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12.0)
    # Drop raw datetime column — THIS IS THE CRITICAL FIX
    result = result.drop(columns=["datetime"])
    return result


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = x.select_dtypes(include="number").columns.tolist()
    categorical_cols = [col for col in x.columns if col not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ], remainder="drop")


# ── Diagnostic: check existing submission ─────────────────────────────────
def diagnose_existing():
    """Analyze the broken submission to confirm the root cause."""
    print("=" * 60)
    print("DIAGNOSIS: Existing Submission")
    print("=" * 60)

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sub = pd.read_csv(EXISTING_SUB)

    train_dt = set(pd.to_datetime(train["datetime"]).dt.strftime("%Y-%m-%d %H"))
    test_dt = set(pd.to_datetime(test["datetime"]).dt.strftime("%Y-%m-%d %H"))
    overlap = train_dt & test_dt

    print(f"Train unique hours: {len(train_dt)}")
    print(f"Test unique hours:  {len(test_dt)}")
    print(f"Overlap hours:      {len(overlap)}")
    print()
    print(f"Train count stats: mean={train['count'].mean():.1f}, std={train['count'].std():.1f}")
    print(f"Existing sub stats: mean={sub['count'].mean():.1f}, std={sub['count'].std():.1f}")
    print(f"  min={sub['count'].min():.1f}, max={sub['count'].max():.1f}")
    print()
    print("ROOT CAUSE: Zero datetime overlap + datetime one-hot encoded as categorical")
    print("-> CV RMSLE 0.154 (shuffled CV sees overlapping datetimes)")
    print("-> Kaggle RMSLE ~1.919 (test datetimes all unseen -> all-zero features)")
    print()


# ── Main fix pipeline ──────────────────────────────────────────────────────
def run_fix():
    """Re-train with proper datetime features and generate corrected submission."""
    print("=" * 60)
    print("FIX: Re-training with proper datetime features")
    print("=" * 60)

    # Load data
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample = pd.read_csv(SAMPLE_PATH)
    print(f"Train: {train.shape}, Test: {test.shape}")

    # Drop casual/registered (not available in test set)
    train_features = train.drop(columns=["casual", "registered", "count"])
    train_target = train["count"].astype(float)
    test_features = test.copy()

    # Extract datetime features and drop raw datetime
    print("Extracting datetime features (hour, day, month, year, dayofweek, cyclical)...")
    X_train = extract_datetime_features(train_features)
    X_test = extract_datetime_features(test_features)
    print(f"Features after extraction: {X_train.shape[1]} columns")
    print(f"Columns: {list(X_train.columns)}")

    # The target transform: log1p for RMSLE optimization
    y_train_log = np.log1p(train_target)

    # ── Model definitions ──────────────────────────────────────────────
    models = {
        "random_forest_log_target": RandomForestRegressor(
            n_estimators=260, max_depth=18, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "extra_trees_log_target": ExtraTreesRegressor(
            n_estimators=320, max_depth=20, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "gradient_boosting_log_target": GradientBoostingRegressor(
            n_estimators=700, learning_rate=0.035, max_depth=3,
            min_samples_leaf=3, subsample=0.85, random_state=RANDOM_STATE,
        ),
    }

    # ── Cross-validation with stacking ─────────────────────────────────
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    preprocessor = build_preprocessor(X_train)

    oof_preds = {}      # model_name -> oof predictions
    cv_scores = {}      # model_name -> cv RMSLE
    test_preds = {}     # model_name -> test predictions

    t_start = time.time()

    for name, model in models.items():
        print(f"\nTraining {name}...")
        oof = np.zeros(len(X_train))
        test_fold_preds = np.zeros(len(X_test))
        fold_scores = []

        for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X_train)):
            # Clone and fit
            fold_pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model),
            ])
            fold_pipe.fit(X_train.iloc[train_idx], y_train_log.iloc[train_idx])

            # OOF predictions (inverse transform from log1p space)
            fold_raw = fold_pipe.predict(X_train.iloc[valid_idx])
            fold_pred = np.expm1(fold_raw)
            oof[valid_idx] = fold_pred

            # Score
            fold_rmsle = rmsle(train_target.iloc[valid_idx], fold_pred)
            fold_scores.append(fold_rmsle)

            # Test predictions for this fold
            test_fold_raw = fold_pipe.predict(X_test)
            test_fold_pred = np.expm1(test_fold_raw)
            test_fold_preds += test_fold_pred / N_FOLDS

        oof_preds[name] = oof
        test_preds[name] = test_fold_preds
        cv_scores[name] = {
            "mean": float(np.mean(fold_scores)),
            "std": float(np.std(fold_scores)),
        }
        oof_rmsle_val = rmsle(train_target, oof)
        print(f"  CV RMSLE: {cv_scores[name]['mean']:.6f} +/- {cv_scores[name]['std']:.6f}")
        print(f"  OOF RMSLE: {oof_rmsle_val:.6f}")

    # ── Stacking ensemble (RidgeCV on OOF predictions) ─────────────────
    print("\nBuilding stacking ensemble...")
    oof_df = pd.DataFrame({
        name: oof_preds[name]
        for name in models
    })
    # Stack in log space for RMSLE
    oof_log_df = np.log1p(oof_df)

    stack_meta = RidgeCV(alphas=[0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    stack_meta.fit(oof_log_df, y_train_log)
    print(f"  RidgeCV best alpha: {stack_meta.alpha_}")

    # OOF stacking predictions (in log space, then inverse transform)
    oof_stack_log = stack_meta.predict(oof_log_df)
    oof_stack = np.expm1(oof_stack_log)
    oof_stack_rmsle = rmsle(train_target, oof_stack)
    print(f"  Stack OOF RMSLE: {oof_stack_rmsle:.6f}")

    # Test stacking predictions
    test_stack_df = pd.DataFrame({
        name: test_preds[name]
        for name in models
    })
    test_stack_log = np.log1p(test_stack_df)
    test_stack_raw = stack_meta.predict(test_stack_log)
    test_stack = np.expm1(test_stack_raw)

    # ── Also compute simple weighted blend ─────────────────────────────
    weights = {"random_forest_log_target": 0.05, "extra_trees_log_target": 0.05, "gradient_boosting_log_target": 0.90}
    test_blend = sum(
        weights[name] * test_preds[name]
        for name in models
    )

    # ── Generate corrected submission ──────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use stacking ensemble for submission
    submission = sample.copy()
    submission["count"] = np.clip(test_stack, 0, None)

    sub_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(sub_path, index=False)
    print(f"\nCorrected submission saved to: {sub_path}")

    # ── Validation checks ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUBMISSION VALIDATION")
    print("=" * 60)
    checks = {
        "rows_match": len(submission) == len(sample),
        "columns_match": submission.columns.tolist() == sample.columns.tolist(),
        "missing": int(submission["count"].isna().sum()),
        "negative": int((submission["count"] < 0).sum()),
        "min": float(submission["count"].min()),
        "max": float(submission["count"].max()),
        "mean": float(submission["count"].mean()),
        "std": float(submission["count"].std()),
    }
    for k, v in checks.items():
        print(f"  {k}: {v}")

    all_ok = checks["rows_match"] and checks["columns_match"] and checks["missing"] == 0 and checks["negative"] == 0
    print(f"\n  VALID: {all_ok}")

    # ── Metrics summary ────────────────────────────────────────────────
    metrics = {
        "task": "bike_sharing_demand",
        "fix": "datetime_feature_extraction",
        "cv_scores": {k: v for k, v in cv_scores.items()},
        "stack_oof_rmsle": oof_stack_rmsle,
        "submission_checks": checks,
        "submission_stats": {
            "min": checks["min"],
            "max": checks["max"],
            "mean": checks["mean"],
            "std": checks["std"],
        },
        "train_count_stats": {
            "mean": float(train_target.mean()),
            "std": float(train_target.std()),
            "min": float(train_target.min()),
            "max": float(train_target.max()),
        },
        "seconds": round(time.time() - t_start, 2),
    }

    metrics_path = OUTPUT_DIR / "fix_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nMetrics saved to: {metrics_path}")

    # ── Compare with broken submission ─────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON: Broken vs Fixed Submission")
    print("=" * 60)
    broken = pd.read_csv(EXISTING_SUB)
    print(f"{'Metric':<20} {'Broken':>12} {'Fixed':>12} {'Training':>12}")
    print("-" * 60)
    print(f"{'mean':<20} {broken['count'].mean():>12.2f} {submission['count'].mean():>12.2f} {train_target.mean():>12.2f}")
    print(f"{'std':<20} {broken['count'].std():>12.2f} {submission['count'].std():>12.2f} {train_target.std():>12.2f}")
    print(f"{'min':<20} {broken['count'].min():>12.2f} {submission['count'].min():>12.2f} {train_target.min():>12.2f}")
    print(f"{'max':<20} {broken['count'].max():>12.2f} {submission['count'].max():>12.2f} {train_target.max():>12.2f}")
    print(f"\nBroken submission RMSLE (CV estimate): {0.154:.4f}  (optimistic due to datetime leakage)")
    print(f"Broken submission RMSLE (Kaggle):      {1.919:.4f}  (actual, 12x worse)")
    print(f"Fixed OOF RMSLE:                       {oof_stack_rmsle:.4f}  (honest CV estimate)")

    return metrics


if __name__ == "__main__":
    diagnose_existing()
    metrics = run_fix()
    print("\n" + "=" * 60)
    print("FIX COMPLETE")
    print("=" * 60)
    print(f"Corrected submission: {OUTPUT_DIR / 'submission.csv'}")
    print(f"Expected Kaggle RMSLE: ~{metrics['stack_oof_rmsle']:.4f}")
    print(f"Previous Kaggle RMSLE: 1.919")
    print(f"Improvement: {1.919 - metrics['stack_oof_rmsle']:.4f} RMSLE reduction")
