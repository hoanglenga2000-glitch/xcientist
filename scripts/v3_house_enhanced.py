#!/usr/bin/env python3
"""
V3 Enhanced for house_prices: log1p target + more features + more iterations.
Direct RMSLE optimization via log-transformed target.
"""
import sys, os, json, time, warnings, argparse
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

HOME = '/hpc2hdd/home/aimslab'
TASK_ID = 'house_prices'
DIR_NAME = 'house-prices-advanced-regression-techniques'
TARGET = 'SalePrice'
TASK_TYPE = 'regression'
METRIC = 'rmsle'
DIRECTION = 'min'
BRONZE = 0.140
MARGIN = 0.005


def load_and_preprocess():
    data_dir = os.path.join(HOME, DIR_NAME)
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    # Extract IDs
    test_ids = test['Id'].values.copy()

    # Separate target (original scale for reference)
    y_orig = train[TARGET].copy()
    y = np.log1p(y_orig)  # log1p transform for RMSLE optimization

    # Drop target and Id
    train_feat = train.drop(columns=[TARGET, 'Id'])
    test_feat = test.drop(columns=['Id'])

    # Combine for encoding
    combined = pd.concat([train_feat, test_feat], ignore_index=True)
    n_train = len(train_feat)

    # Encode categoricals - keep more features (threshold 500 instead of 100)
    for col in list(combined.columns):
        if combined[col].dtype == 'object':
            if combined[col].nunique() > 500:
                combined.drop(columns=[col], inplace=True)
            else:
                combined[col] = combined[col].fillna('MISSING')
                combined[col] = LabelEncoder().fit_transform(combined[col].astype(str))
        elif combined[col].dtype in ('float64', 'int64'):
            combined[col] = combined[col].fillna(combined[col].median())

    X_train = combined.iloc[:n_train].copy()
    X_test = combined.iloc[n_train:].copy()

    return X_train, y, y_orig, X_test, test_ids


def compute_rmsle(y_true, y_pred):
    """Compute RMSLE on original scale (y_true, y_pred are original values)."""
    yt = np.maximum(y_true, 0)
    yp = np.maximum(y_pred, 0)
    return float(np.sqrt(np.mean((np.log1p(yp) - np.log1p(yt)) ** 2)))


def train_catboost(X_train, y, X_test, gpu_device, n_folds, fast):
    n_iter = 300 if fast else 2000
    early_stop = 50 if fast else 120

    folds = list(KFold(n_splits=n_folds, shuffle=True, random_state=42).split(X_train))

    oof = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    scores = []

    cb_params = {
        'iterations': n_iter,
        'learning_rate': 0.02,
        'depth': 5,
        'l2_leaf_reg': 5,
        'bootstrap_type': 'Bayesian',
        'bagging_temperature': 0.5,
        'task_type': 'GPU',
        'devices': str(gpu_device),
        'verbose': 0,
        'random_seed': 42,
        'allow_writing_files': False,
        'early_stopping_rounds': early_stop,
        'use_best_model': True,
    }

    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        print(f"  Fold {fold_idx+1}/{n_folds}...", end=' ', flush=True)
        t0 = time.time()
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = CatBoostRegressor(loss_function='RMSE', **cb_params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)

        val_pred = model.predict(X_val)
        oof[val_idx] = val_pred
        test_preds += model.predict(X_test) / n_folds

        fold_score = float(np.sqrt(mean_squared_error(y_val, val_pred)))
        scores.append(fold_score)
        print(f"score={fold_score:.4f} [{time.time()-t0:.0f}s]")

    return oof, test_preds, scores


def make_submission(test_preds_log, test_ids):
    """Convert log-scale predictions back to original scale."""
    pred_values = np.expm1(test_preds_log)
    pred_values = np.maximum(pred_values, 0)
    sub = pd.DataFrame({'Id': test_ids, 'SalePrice': [f"{v:.6f}" for v in pred_values]})
    return sub


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-device', type=int, default=1)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()

    t_start = time.time()

    X_train, y_log, y_orig, X_test, test_ids = load_and_preprocess()

    # Auto-fold
    n_folds = args.n_folds
    if n_folds == 5 and len(X_train) > 500000: n_folds = 3
    elif n_folds == 5 and len(X_train) > 100000: n_folds = 4

    print(f"\n{'='*60}")
    print(f"TASK: {TASK_ID} (log1p target) | GPU: {args.gpu_device} | FOLDS: {n_folds}")
    print(f"{'='*60}")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Target: log1p(SalePrice), range=[{y_log.min():.2f}, {y_log.max():.2f}]")

    oof_log, test_preds_log, cv_scores = train_catboost(
        X_train, y_log, X_test, args.gpu_device, n_folds, args.fast
    )

    # CV scores are RMSE in log space = per-fold RMSLE
    oof_score = float(np.mean(cv_scores))
    oof_std = float(np.std(cv_scores))

    # Full OOF RMSLE on original scale
    oof_orig = np.expm1(oof_log)
    oof_rmsle = compute_rmsle(y_orig.values, oof_orig)

    print(f"\n  CV Fold Scores (log RMSE): {[f'{s:.5f}' for s in cv_scores]}")
    print(f"  OOF (fold mean, log RMSE): {oof_score:.5f} +/- {oof_std:.5f}")
    print(f"  OOF RMSLE (original scale): {oof_rmsle:.6f}")

    # GATE CHECK
    passed = oof_rmsle <= BRONZE - MARGIN
    gap = oof_rmsle - (BRONZE - MARGIN)
    gate_status = "PASS" if passed else "FAIL"
    print(f"  GATE: {gate_status} | RMSLE={oof_rmsle:.6f} vs bronze={BRONZE} margin={MARGIN} | gap={gap:+.6f}")

    # Make submission
    sub = make_submission(test_preds_log, test_ids)
    sub_path = f"/hpc2hdd/home/aimslab/results/v3_submission_{TASK_ID}.csv"
    sub.to_csv(sub_path, index=False)

    elapsed = time.time() - t_start
    result_json = {
        "task_id": TASK_ID,
        "status": "completed",
        "metric": METRIC,
        "direction": DIRECTION,
        "oof_score": round(oof_rmsle, 6),
        "oof_std": round(oof_std, 6),
        "cv_scores": [round(float(s), 6) for s in cv_scores],
        "bronze_threshold": BRONZE,
        "gate_passed": passed,
        "gate_gap": round(float(gap), 6),
        "n_folds": n_folds,
        "n_features": X_train.shape[1],
        "elapsed_seconds": round(elapsed, 1),
        "submission_path": sub_path,
    }
    result_path = f"/hpc2hdd/home/aimslab/results/v3_result_{TASK_ID}.json"
    with open(result_path, 'w') as f:
        json.dump(result_json, f)

    print(f"  RESULT: {json.dumps({k: result_json[k] for k in ['task_id', 'oof_score', 'gate_passed', 'gate_gap', 'elapsed_seconds']})}")
    print(f"SUMMARY: {TASK_ID} | RMSLE={oof_rmsle:.6f} | GATE={gate_status} | {elapsed:.0f}s")


if __name__ == '__main__':
    main()
