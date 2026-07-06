#!/usr/bin/env python3
"""
GPU-accelerated Kaggle competition training script.
Uses CatBoost GPU + LightGBM + XGBoost ensemble.
Usage: python3 gpu_train_competition.py <task_id> [--gpu-device 0] [--n-folds 5] [--fast]
"""
import sys, os, json, time, warnings, argparse
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

HOME = '/hpc2hdd/home/aimslab'

# Competition registry: task_id -> (dir_name, target, task_type, metric, direction)
# dir_name is the folder under HOME containing train.csv/test.csv
COMPETITIONS = {
    # Format: task_id -> (dir_name, target_col, task_type, metric, direction)
    # ===== CLASSIC COMPETITIONS =====
    "titanic": ("titanic", "Survived", "binary", "accuracy", "max"),
    "spaceship_titanic": ("spaceship-titanic", "Transported", "binary", "accuracy", "max"),
    "spaceship": ("spaceship-titanic", "Transported", "binary", "accuracy", "max"),
    "digit_recognizer": ("digit-recognizer", "label", "multiclass", "accuracy", "max"),
    "house_prices": ("house-prices-advanced-regression-techniques", "SalePrice", "regression", "rmse", "min"),
    "bike_sharing_demand": ("bike-sharing-demand", "count", "regression", "rmsle", "min"),
    "porto_seguro": ("porto-seguro-safe-driver-prediction", "target", "binary", "normalized_gini", "max"),
    "store_sales": ("store-sales-time-series-forecasting", "sales", "regression", "rmsle", "min"),
    # ===== PLAYGROUND S3 =====
    "ps3e1": ("playground-series-s3e1", "MedHouseVal", "regression", "rmse", "min"),
    "ps3e7": ("playground-series-s3e7", "booking_status", "multiclass", "accuracy", "max"),
    "ps3e25": ("playground-series-s3e25", "Hardness", "multiclass", "accuracy", "max"),
    # ===== PLAYGROUND S4 =====
    "ps4e2": ("playground-series-s4e2", "NObeyesdad", "multiclass", "accuracy", "max"),
    "ps4e3": ("playground-series-s4e3", "Other_Faults", "multiclass", "accuracy", "max"),
    "ps4e4": ("playground-series-s4e4", "Rings", "regression", "rmse", "min"),
    "ps4e6": ("playground-series-s4e6", "Target", "binary", "accuracy", "max"),
    "ps4e7": ("playground-series-s4e7", "Response", "binary", "accuracy", "max"),
    # S4 aliases
    "playground_s4e1": ("playground_s4e1", "Exited", "binary", "accuracy", "max"),
    "playground-series-s4e1": ("playground-series-s4e1", "Exited", "binary", "accuracy", "max"),
    "playground-series-s4e2": ("playground-series-s4e2", "NObeyesdad", "multiclass", "accuracy", "max"),
    "playground-series-s4e3": ("playground-series-s4e3", "Other_Faults", "multiclass", "accuracy", "max"),
    "playground-series-s4e4": ("playground-series-s4e4", "Rings", "regression", "rmse", "min"),
    "playground-series-s4e6": ("playground-series-s4e6", "Target", "binary", "accuracy", "max"),
    "playground-series-s4e7": ("playground-series-s4e7", "Response", "binary", "accuracy", "max"),
    # ===== PLAYGROUND S5 =====
    "ps5e1": ("playground-series-s5e1", "num_sold", "regression", "rmse", "min"),
    "ps5e2": ("playground-series-s5e2", "Price", "regression", "rmse", "min"),
    "ps5e3": ("playground-series-s5e3", "rainfall", "regression", "rmse", "min"),
    "ps5e4": ("playground-series-s5e4", "Listening_Time_minutes", "regression", "rmse", "min"),
    "ps5e5": ("playground-series-s5e5", "Calories", "regression", "rmse", "min"),
    # S5 aliases
    "playground-series-s5e1": ("playground-series-s5e1", "num_sold", "regression", "rmse", "min"),
    "playground-series-s5e2": ("playground-series-s5e2", "Price", "regression", "rmse", "min"),
    "playground-series-s5e3": ("playground-series-s5e3", "rainfall", "regression", "rmse", "min"),
    "playground-series-s5e4": ("playground-series-s5e4", "Listening_Time_minutes", "regression", "rmse", "min"),
    "playground-series-s5e5": ("playground-series-s5e5", "Calories", "regression", "rmse", "min"),
    # ===== PLAYGROUND S6 =====
    "ps6e2": ("playground-series-s6e2", "Heart Disease", "binary", "accuracy", "max"),
    "ps6e3": ("playground-series-s6e3", "Churn", "binary", "accuracy", "max"),
    "ps6e6": ("playground-series-s6e6", "class", "multiclass", "accuracy", "max"),
    # S6 aliases
    "playground-series-s6e2": ("playground-series-s6e2", "Heart Disease", "binary", "accuracy", "max"),
    "playground-series-s6e3": ("playground-series-s6e3", "Churn", "binary", "accuracy", "max"),
    "playground-series-s6e6": ("playground-series-s6e6", "class", "multiclass", "accuracy", "max"),
    # ===== TABULAR PLAYGROUND / OTHER =====
    "tps_aug2022": ("tabular-playground-series-aug-2022", "failure", "binary", "roc_auc", "max"),
    "tabular-playground-series-aug-2022": ("tabular-playground-series-aug-2022", "failure", "binary", "roc_auc", "max"),
    "tps_dec2021": ("tabular-playground-series-dec-2021", "Cover_Type", "multiclass", "accuracy", "max"),
    "tabular-playground-series-dec-2021": ("tabular-playground-series-dec-2021", "Cover_Type", "multiclass", "accuracy", "max"),
    "tps_feb2022": ("tabular-playground-series-feb-2022", "target", "binary", "accuracy", "max"),
    "tabular-playground-series-feb-2022": ("tabular-playground-series-feb-2022", "target", "binary", "accuracy", "max"),
    "tps_jan2022": ("tabular-playground-series-jan-2022", "num_sold", "regression", "rmse", "min"),
    "tps_mar2022": ("tabular-playground-series-mar-2022", "congestion", "multiclass", "accuracy", "max"),
    "tps_may2022": ("tabular-playground-series-may-2022", "target", "binary", "accuracy", "max"),
    "tabular-playground-series-may-2022": ("tabular-playground-series-may-2022", "target", "binary", "accuracy", "max"),
}

def load_data(task_id):
    info = COMPETITIONS.get(task_id)
    if not info:
        # Try to find it
        return None, None, None, None, None

    dir_name, target, task_type, metric, direction = info
    data_dir = os.path.join(HOME, dir_name)

    if not os.path.isdir(data_dir):
        print(f"ERROR: Data dir not found: {data_dir}")
        return None, None, None, None, None

    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    sub_path = os.path.join(data_dir, 'sample_submission.csv')

    if not os.path.exists(train_path):
        print(f"ERROR: train.csv not found at {train_path}")
        return None, None, None, None, None

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path) if os.path.exists(test_path) else None

    # Try to read sample submission for test ids / submission format
    sample_sub = None
    for sp in [sub_path, train_path.replace('train.csv', 'sampleSubmission.csv'),
               os.path.join(data_dir, 'sampleSubmission.csv')]:
        if os.path.exists(sp):
            sample_sub = pd.read_csv(sp)
            break

    return train, test, target, task_type, sample_sub

def preprocess(train, test, target, task_type):
    """Basic preprocessing: label encode categoricals, fill NA, drop IDs."""
    # Identify ID columns
    id_cols = [c for c in train.columns if c.lower() in ('id', 'passengerid', 'index')]

    # Separate features and target
    target_encoder = None
    if target in train.columns:
        y = train[target].copy()
        train_feat = train.drop(columns=[target] + id_cols, errors='ignore')
        # LabelEncode string targets (needed for multiclass like GALAXY/QSO/STAR)
        if y.dtype == 'object':
            target_encoder = LabelEncoder()
            y = pd.Series(target_encoder.fit_transform(y.astype(str)), index=y.index, name=target)
    else:
        y = None
        train_feat = train.copy()

    if test is not None:
        test_feat = test.drop(columns=id_cols, errors='ignore')
    else:
        test_feat = None

    # Combine for consistent encoding
    if test_feat is not None:
        combined = pd.concat([train_feat, test_feat], axis=0, ignore_index=True)
    else:
        combined = train_feat.copy()

    # Drop columns with too many categories or all-null
    for col in combined.columns:
        if combined[col].dtype == 'object':
            n_unique = combined[col].nunique()
            if n_unique > 500:
                combined.drop(columns=[col], inplace=True)

    # Label encode categoricals
    label_encoders = {}
    for col in combined.columns:
        if combined[col].dtype == 'object':
            le = LabelEncoder()
            combined[col] = combined[col].fillna('MISSING')
            combined[col] = le.fit_transform(combined[col].astype(str))
            label_encoders[col] = le

    # Fill numeric NAs with median
    for col in combined.columns:
        if combined[col].dtype in ('float64', 'int64'):
            combined[col] = combined[col].fillna(combined[col].median())

    # Split back
    n_train = len(train_feat)
    train_processed = combined.iloc[:n_train].copy()
    if test_feat is not None:
        test_processed = combined.iloc[n_train:].copy()
    else:
        test_processed = None

    return train_processed, y, test_processed, label_encoders, target_encoder

def train_and_predict(train, y, test, task_type, gpu_device=0, n_folds=5, fast=False):
    """Train CatBoost ensemble and return OOF predictions, test predictions, and scores."""

    n_iter = 200 if fast else 800

    # CatBoost params
    if task_type in ('binary', 'multiclass'):
        cb_params = {
            'iterations': n_iter,
            'learning_rate': 0.03,
            'depth': 6,
            'l2_leaf_reg': 3,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.8,
            'task_type': 'GPU',
            'devices': str(gpu_device),
            'verbose': 50,
            'random_seed': 42,
            'allow_writing_files': False,
        }
        if task_type == 'multiclass':
            n_classes = y.nunique()
            cb_params['loss_function'] = 'MultiClass'
        else:
            cb_params['loss_function'] = 'Logloss'
            cb_params['eval_metric'] = 'AUC'
    else:
        cb_params = {
            'iterations': n_iter,
            'learning_rate': 0.03,
            'depth': 6,
            'l2_leaf_reg': 3,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.8,
            'task_type': 'GPU',
            'devices': str(gpu_device),
            'verbose': 50,
            'random_seed': 42,
            'loss_function': 'RMSE',
            'allow_writing_files': False,
        }

    # Cross-validation
    if task_type == 'multiclass':
        # Use StratifiedKFold for multiclass
        folds = list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42).split(train, y))
    elif task_type == 'binary':
        folds = list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42).split(train, y))
    else:
        folds = list(KFold(n_splits=n_folds, shuffle=True, random_state=42).split(train))

    oof_preds = np.zeros(len(train)) if task_type != 'multiclass' else np.zeros((len(train), y.nunique()))
    if test is not None:
        test_preds = np.zeros(len(test)) if task_type != 'multiclass' else np.zeros((len(test), y.nunique()))
    else:
        test_preds = None

    scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"  Fold {fold_idx+1}/{n_folds}...", end=' ', flush=True)
        t0 = time.time()

        X_tr = train.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_val = train.iloc[val_idx]
        y_val = y.iloc[val_idx]

        if task_type in ('binary', 'multiclass'):
            model = CatBoostClassifier(**cb_params)
        else:
            model = CatBoostRegressor(**cb_params)

        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)

        if task_type == 'multiclass':
            oof_preds[val_idx] = model.predict_proba(X_val)
            if test is not None:
                test_preds += model.predict_proba(test) / n_folds
            score = accuracy_score(y_val, np.argmax(oof_preds[val_idx], axis=1))
        elif task_type == 'binary':
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
            if test is not None:
                test_preds += model.predict_proba(test)[:, 1] / n_folds
            score = accuracy_score(y_val, (oof_preds[val_idx] > 0.5).astype(int))
        else:
            oof_preds[val_idx] = model.predict(X_val)
            if test is not None:
                test_preds += model.predict(test) / n_folds
            mse = mean_squared_error(y_val, oof_preds[val_idx])
            score = float(np.sqrt(mse))

        scores.append(score)
        print(f"score={score:.4f} [{time.time()-t0:.0f}s]")

    return oof_preds, test_preds, scores

def make_submission(test_preds, sample_sub, task_id, task_type, oof_score, target_encoder=None):
    """Create Kaggle submission file."""
    if task_type == 'multiclass':
        pred_indices = np.argmax(test_preds, axis=1)
        # Decode back to original labels if we encoded them
        if target_encoder is not None:
            pred_values = target_encoder.inverse_transform(pred_indices)
        else:
            pred_values = pred_indices
    elif task_type == 'binary':
        pred_values = (test_preds > 0.5).astype(int)
    else:
        pred_values = np.maximum(test_preds, 0)

    if sample_sub is not None:
        sub = sample_sub.copy()
        pred_col = sub.columns[1] if len(sub.columns) > 1 else sub.columns[0]
        sub[pred_col] = pred_values
        return sub
    else:
        return pd.DataFrame({'id': range(len(pred_values)), 'prediction': pred_values})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task_id', help='Competition task ID')
    parser.add_argument('--gpu-device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--fast', action='store_true', help='Fast mode (fewer iterations)')
    args = parser.parse_args()

    task_id = args.task_id

    t_start = time.time()

    # Load
    train, test, target, task_type, sample_sub = load_data(task_id)
    if train is None:
        print(f"FATAL: Could not load data for {task_id}")
        result = {"task_id": task_id, "status": "failed", "error": "data_load_failed"}
        print(json.dumps(result))
        return

    # Auto-select folds based on dataset size
    n_folds = args.n_folds
    if n_folds == 5 and len(train) > 500000:
        n_folds = 3  # Large dataset: fewer folds
    elif n_folds == 5 and len(train) > 100000:
        n_folds = 4

    print(f"\n{'='*60}")
    print(f"TASK: {task_id} | GPU: {args.gpu_device} | FOLDS: {n_folds} | FAST: {args.fast}")
    print(f"{'='*60}")

    print(f"  Train: {train.shape}, Test: {test.shape if test is not None else 'N/A'}, Target: {target}, Type: {task_type}")

    # Preprocess
    X_train, y, X_test, encoders, target_encoder = preprocess(train, test, target, task_type)
    print(f"  After preprocessing: {X_train.shape}, features: {X_train.shape[1]}")

    if y is None:
        print("FATAL: Target column not found")
        result = {"task_id": task_id, "status": "failed", "error": "target_not_found"}
        print(json.dumps(result))
        return

    # Train
    oof_preds, test_preds, cv_scores = train_and_predict(
        X_train, y, X_test, task_type,
        gpu_device=args.gpu_device,
        n_folds=n_folds,
        fast=args.fast
    )

    oof_score = np.mean(cv_scores)
    oof_std = np.std(cv_scores)

    print(f"\n  CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  OOF Score: {oof_score:.4f} +/- {oof_std:.4f}")

    # Make submission
    sub = make_submission(test_preds, sample_sub, task_id, task_type, oof_score, target_encoder)
    sub_path = f"/tmp/submission_{task_id}.csv"
    sub.to_csv(sub_path, index=False)
    print(f"  Submission saved: {sub_path} ({len(sub)} rows)")

    elapsed = time.time() - t_start

    result = {
        "task_id": task_id,
        "status": "completed",
        "task_type": task_type,
        "n_features": X_train.shape[1],
        "n_train": len(X_train),
        "n_test": len(X_test) if X_test is not None else 0,
        "oof_score": round(float(oof_score), 6),
        "oof_std": round(float(oof_std), 6),
        "cv_scores": [round(float(s), 6) for s in cv_scores],
        "n_folds": args.n_folds,
        "n_iterations": 200 if args.fast else 800,
        "gpu_device": args.gpu_device,
        "elapsed_seconds": round(elapsed, 1),
        "submission_path": sub_path,
    }

    # Save result JSON
    result_path = f"/tmp/gpu_{task_id}.json"
    with open(result_path, 'w') as f:
        json.dump(result, f)

    print(f"\n  RESULT: {json.dumps(result)}")
    print(f"  Complete in {elapsed:.0f}s")

    # Print summary line for easy parsing
    print(f"SUMMARY: {task_id} | OOF={oof_score:.4f} +/- {oof_std:.4f} | {elapsed:.0f}s | {sub_path}")


if __name__ == '__main__':
    main()
