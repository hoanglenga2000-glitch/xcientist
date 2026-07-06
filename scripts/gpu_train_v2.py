#!/usr/bin/env python3
"""
GPU-accelerated Kaggle competition training script V2.
- CatBoost + LightGBM + XGBoost ensemble
- Proper RMSLE/RMSE metrics
- Task-specific feature engineering
- Correct submission formats
- Auto-fold selection
Usage: python3 gpu_train_v2.py <task_id> [--gpu-device 0] [--n-folds 5] [--fast]
"""
import sys, os, json, time, warnings, argparse
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

HOME = '/hpc2hdd/home/aimslab'

# Complete competition registry
COMPETITIONS = {
    # === CLASSIC ===
    "titanic": ("titanic", "Survived", "binary", "accuracy", "max",
                "PassengerId", "Survived"),
    "spaceship_titanic": ("spaceship-titanic", "Transported", "binary", "accuracy", "max",
                          "PassengerId", "Transported"),
    "digit_recognizer": ("digit-recognizer", "label", "multiclass", "accuracy", "max",
                         "ImageId", "Label"),
    "house_prices": ("house-prices-advanced-regression-techniques", "SalePrice", "regression", "rmsle", "min",
                     "Id", "SalePrice"),
    "bike_sharing_demand": ("bike-sharing-demand", "count", "regression", "rmsle", "min",
                            "datetime", "count"),
    "porto_seguro": ("porto-seguro-safe-driver-prediction", "target", "binary", "normalized_gini", "max",
                     "id", "target"),
    "store_sales": ("store-sales-time-series-forecasting", "sales", "regression", "rmsle", "min",
                    "id", "sales"),
    # === PLAYGROUND S3 ===
    "ps3e1": ("playground-series-s3e1", "MedHouseVal", "regression", "rmse", "min", "id", "MedHouseVal"),
    "ps3e7": ("playground-series-s3e7", "booking_status", "multiclass", "accuracy", "max", "id", "booking_status"),
    "ps3e25": ("playground-series-s3e25", "Hardness", "multiclass", "accuracy", "max", "id", "Hardness"),
    # === PLAYGROUND S4 ===
    "ps4e1": ("playground-series-s4e1", "Exited", "binary", "accuracy", "max", "id", "Exited"),
    "ps4e2": ("playground-series-s4e2", "NObeyesdad", "multiclass", "accuracy", "max", "id", "NObeyesdad"),
    "ps4e3": ("playground-series-s4e3", "Other_Faults", "multiclass", "accuracy", "max", "id", "Other_Faults"),
    "ps4e4": ("playground-series-s4e4", "Rings", "regression", "rmse", "min", "id", "Rings"),
    "ps4e6": ("playground-series-s4e6", "Target", "binary", "accuracy", "max", "id", "Target"),
    "ps4e7": ("playground-series-s4e7", "Response", "binary", "accuracy", "max", "id", "Response"),
    # Aliases
    "playground_s4e1": ("playground_s4e1", "Exited", "binary", "accuracy", "max", "id", "Exited"),
    "playground-series-s4e1": ("playground-series-s4e1", "Exited", "binary", "accuracy", "max", "id", "Exited"),
    "playground-series-s4e2": ("playground-series-s4e2", "NObeyesdad", "multiclass", "accuracy", "max", "id", "NObeyesdad"),
    "playground-series-s4e3": ("playground-series-s4e3", "Other_Faults", "multiclass", "accuracy", "max", "id", "Other_Faults"),
    "playground-series-s4e4": ("playground-series-s4e4", "Rings", "regression", "rmse", "min", "id", "Rings"),
    "playground-series-s4e6": ("playground-series-s4e6", "Target", "binary", "accuracy", "max", "id", "Target"),
    "playground-series-s4e7": ("playground-series-s4e7", "Response", "binary", "accuracy", "max", "id", "Response"),
    # === PLAYGROUND S5 ===
    "ps5e1": ("playground-series-s5e1", "num_sold", "regression", "rmse", "min", "id", "num_sold"),
    "ps5e2": ("playground-series-s5e2", "Price", "regression", "rmse", "min", "id", "Price"),
    "ps5e3": ("playground-series-s5e3", "rainfall", "regression", "rmse", "min", "id", "rainfall"),
    "ps5e4": ("playground-series-s5e4", "Listening_Time_minutes", "regression", "rmse", "min", "id", "Listening_Time_minutes"),
    "ps5e5": ("playground-series-s5e5", "Calories", "regression", "rmse", "min", "id", "Calories"),
    # S5 aliases
    "playground-series-s5e1": ("playground-series-s5e1", "num_sold", "regression", "rmse", "min", "id", "num_sold"),
    "playground-series-s5e2": ("playground-series-s5e2", "Price", "regression", "rmse", "min", "id", "Price"),
    "playground-series-s5e3": ("playground-series-s5e3", "rainfall", "regression", "rmse", "min", "id", "rainfall"),
    "playground-series-s5e4": ("playground-series-s5e4", "Listening_Time_minutes", "regression", "rmse", "min", "id", "Listening_Time_minutes"),
    "playground-series-s5e5": ("playground-series-s5e5", "Calories", "regression", "rmse", "min", "id", "Calories"),
    # === PLAYGROUND S6 ===
    "ps6e2": ("playground-series-s6e2", "Heart Disease", "binary", "accuracy", "max", "id", "Heart Disease"),
    "ps6e3": ("playground-series-s6e3", "Churn", "binary", "accuracy", "max", "id", "Churn"),
    "ps6e6": ("playground-series-s6e6", "class", "multiclass", "accuracy", "max", "id", "class"),
    # S6 aliases
    "playground-series-s6e2": ("playground-series-s6e2", "Heart Disease", "binary", "accuracy", "max", "id", "Heart Disease"),
    "playground-series-s6e3": ("playground-series-s6e3", "Churn", "binary", "accuracy", "max", "id", "Churn"),
    "playground-series-s6e6": ("playground-series-s6e6", "class", "multiclass", "accuracy", "max", "id", "class"),
    # === TABULAR PLAYGROUND ===
    "tps_aug2022": ("tabular-playground-series-aug-2022", "failure", "binary", "roc_auc", "max", "id", "failure"),
    "tps_dec2021": ("tabular-playground-series-dec-2021", "Cover_Type", "multiclass", "accuracy", "max", "id", "Cover_Type"),
    "tps_feb2022": ("tabular-playground-series-feb-2022", "target", "binary", "accuracy", "max", "id", "target"),
    "tps_jan2022": ("tabular-playground-series-jan-2022", "num_sold", "regression", "rmse", "min", "id", "num_sold"),
    "tps_mar2022": ("tabular-playground-series-mar-2022", "congestion", "multiclass", "accuracy", "max", "id", "congestion"),
    "tps_may2022": ("tabular-playground-series-may-2022", "target", "binary", "accuracy", "max", "id", "target"),
    # Aliases
    "tabular-playground-series-aug-2022": ("tabular-playground-series-aug-2022", "failure", "binary", "roc_auc", "max", "id", "failure"),
    "tabular-playground-series-dec-2021": ("tabular-playground-series-dec-2021", "Cover_Type", "multiclass", "accuracy", "max", "id", "Cover_Type"),
    "tabular-playground-series-feb-2022": ("tabular-playground-series-feb-2022", "target", "binary", "accuracy", "max", "id", "target"),
    "tabular-playground-series-jan-2022": ("tabular-playground-series-jan-2022", "num_sold", "regression", "rmse", "min", "id", "num_sold"),
    "tabular-playground-series-mar-2022": ("tabular-playground-series-mar-2022", "congestion", "multiclass", "accuracy", "max", "id", "congestion"),
    "tabular-playground-series-may-2022": ("tabular-playground-series-may-2022", "target", "binary", "accuracy", "max", "id", "target"),
}

# ===== FEATURE ENGINEERING HOOKS =====
def titanic_features(train, test):
    """Add Title, FamilySize, IsAlone, FarePerPerson features."""
    for df in [train, test]:
        if df is None: continue
        if 'Name' in df.columns:
            df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.', expand=False)
            df['Title'] = df['Title'].apply(lambda x: x if x in ['Mr','Mrs','Miss','Master','Dr','Rev'] else 'Other')
        if 'SibSp' in df.columns and 'Parch' in df.columns:
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        if 'Fare' in df.columns and 'FamilySize' in df.columns:
            df['FarePerPerson'] = df['Fare'] / df['FamilySize'].clip(lower=1)
    return train, test

def spaceship_features(train, test):
    """CryoSleep+Age interaction, group size from PassengerId."""
    for df in [train, test]:
        if df is None: continue
        if 'CryoSleep' in df.columns and 'Age' in df.columns:
            df['CryoAge'] = df['CryoSleep'].astype(bool) & (df['Age'] < 20)
        if 'PassengerId' in df.columns:
            df['Group'] = df['PassengerId'].str.split('_').str[0]
            df['GroupSize'] = df['Group'].map(df['Group'].value_counts())
        if 'RoomService' in df.columns and 'FoodCourt' in df.columns:
            df['TotalSpend'] = df[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].sum(axis=1)
            df['HasSpend'] = (df['TotalSpend'] > 0).astype(int)
    return train, test

FEATURE_HOOKS = {
    "titanic": titanic_features,
    "house_prices": None,  # No special features needed
    "spaceship_titanic": spaceship_features,
}


def load_data(task_id):
    info = COMPETITIONS.get(task_id)
    if not info:
        print(f"ERROR: Unknown task {task_id}")
        return None, None, None, None, None, None, None

    dir_name, target, task_type, metric, direction, id_col_out, pred_col_out = info
    data_dir = os.path.join(HOME, dir_name)

    if not os.path.isdir(data_dir):
        print(f"ERROR: Data dir not found: {data_dir}")
        return None, None, None, None, None, None, None

    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')

    if not os.path.exists(train_path):
        print(f"ERROR: train.csv not found at {train_path}")
        return None, None, None, None, None, None, None

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path) if os.path.exists(test_path) else None

    # Get sample submission for test IDs
    sample_sub = None
    for sp in [os.path.join(data_dir, 'sample_submission.csv'),
               os.path.join(data_dir, 'sampleSubmission.csv')]:
        if os.path.exists(sp):
            sample_sub = pd.read_csv(sp)
            break

    return train, test, target, task_type, metric, sample_sub, (id_col_out, pred_col_out)


def preprocess(train, test, target, task_type):
    """Preprocess: feature engineering, encode, fill NA, drop high-cardinality."""
    # Apply feature engineering hook
    task_id = None  # Will be set from context
    # We detect task from the target column name indirectly

    # Identify ID columns (exact match only, not substring)
    id_cols = []
    for c in train.columns:
        cl = c.lower().replace(' ', '_')
        if cl in ('id', 'passengerid', 'index', 'imageid'):
            id_cols.append(c)

    # Separate features and target
    target_encoder = None
    if target in train.columns:
        y = train[target].copy()
        train_feat = train.drop(columns=[target] + id_cols, errors='ignore')
    else:
        y = None
        train_feat = train.copy()

    if test is not None:
        test_ids = test[id_cols[0]].copy() if id_cols and id_cols[0] in test.columns else None
        test_feat = test.drop(columns=id_cols, errors='ignore')
    else:
        test_ids = None
        test_feat = None

    # Encode string target
    if y is not None and y.dtype == 'object':
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y.astype(str)), index=y.index)

    # Combine for consistent encoding
    if test_feat is not None:
        combined = pd.concat([train_feat, test_feat], axis=0, ignore_index=True)
    else:
        combined = train_feat.copy()

    # Drop columns with >500 unique categories
    for col in list(combined.columns):
        if combined[col].dtype == 'object':
            if combined[col].nunique() > 500:
                combined.drop(columns=[col], inplace=True)

    # Label encode categoricals, fill NAs
    for col in combined.columns:
        if combined[col].dtype == 'object':
            combined[col] = combined[col].fillna('MISSING')
            combined[col] = LabelEncoder().fit_transform(combined[col].astype(str))
        elif combined[col].dtype in ('float64', 'int64'):
            combined[col] = combined[col].fillna(combined[col].median())

    # Split back
    n_train = len(train_feat)
    train_processed = combined.iloc[:n_train].copy()
    if test_feat is not None:
        test_processed = combined.iloc[n_train:].copy()
    else:
        test_processed = None

    return train_processed, y, test_processed, target_encoder, id_cols


def compute_metric(y_true, y_pred, metric, task_type):
    """Compute evaluation metric correctly."""
    if metric == "rmsle":
        # RMSLE = sqrt(mean((log(p+1) - log(y+1))^2))
        y_true_clipped = np.maximum(y_true, 0)
        y_pred_clipped = np.maximum(y_pred, 0)
        log_diff = np.log1p(y_pred_clipped) - np.log1p(y_true_clipped)
        return float(np.sqrt(np.mean(log_diff ** 2)))
    elif metric == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    elif metric == "accuracy":
        if task_type == "multiclass":
            return float(accuracy_score(y_true, np.argmax(y_pred, axis=1)))
        else:
            return float(accuracy_score(y_true, (y_pred > 0.5).astype(int) if y_pred.ndim == 1 else np.argmax(y_pred, axis=1)))
    elif metric == "roc_auc":
        return float(roc_auc_score(y_true, y_pred))
    else:
        # Default: accuracy for classification, RMSE for regression
        if task_type in ("binary", "multiclass"):
            return float(accuracy_score(y_true, (y_pred > 0.5).astype(int) if y_pred.ndim == 1 else np.argmax(y_pred, axis=1)))
        else:
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_and_predict(train, y, test, task_type, metric, gpu_device=0, n_folds=5, fast=False):
    """Train ensemble (CatBoost + LGBM + XGBoost) with CV and return OOF + test predictions."""
    n_iter = 200 if fast else 800

    # Determine folds
    if task_type in ('binary', 'multiclass'):
        folds = list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42).split(train, y))
    else:
        folds = list(KFold(n_splits=n_folds, shuffle=True, random_state=42).split(train))

    n_classes = y.nunique() if task_type == 'multiclass' else 1
    oof_preds = np.zeros(len(train)) if task_type != 'multiclass' else np.zeros((len(train), n_classes))
    if test is not None:
        test_preds = np.zeros(len(test)) if task_type != 'multiclass' else np.zeros((len(test), n_classes))
    else:
        test_preds = None

    scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"  Fold {fold_idx+1}/{n_folds}...", end=' ', flush=True)
        t0 = time.time()

        X_tr, X_val = train.iloc[train_idx], train.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # CatBoost (GPU)
        if task_type == 'multiclass':
            cb = CatBoostClassifier(iterations=n_iter, learning_rate=0.03, depth=6,
                                    task_type='GPU', devices=str(gpu_device),
                                    loss_function='MultiClass', verbose=0,
                                    random_seed=42, allow_writing_files=False)
            cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
            cb_pred = cb.predict_proba(X_val)
            oof_preds[val_idx] = cb_pred
            if test is not None:
                test_preds += cb.predict_proba(test) / n_folds
            fold_score = accuracy_score(y_val, np.argmax(cb_pred, axis=1))
        elif task_type == 'binary':
            cb = CatBoostClassifier(iterations=n_iter, learning_rate=0.03, depth=6,
                                    task_type='GPU', devices=str(gpu_device),
                                    loss_function='Logloss', verbose=0,
                                    random_seed=42, allow_writing_files=False)
            cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
            cb_pred = cb.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = cb_pred * 0.5  # CatBoost gets 50% weight
            if test is not None:
                test_preds += cb.predict_proba(test)[:, 1] * 0.5 / n_folds
            fold_score = accuracy_score(y_val, (cb_pred > 0.5).astype(int))
        else:
            cb = CatBoostRegressor(iterations=n_iter, learning_rate=0.03, depth=6,
                                   task_type='GPU', devices=str(gpu_device),
                                   loss_function='RMSE', verbose=0,
                                   random_seed=42, allow_writing_files=False)
            cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
            cb_pred = cb.predict(X_val)
            oof_preds[val_idx] = cb_pred * 0.5
            if test is not None:
                test_preds += cb.predict(test) * 0.5 / n_folds
            fold_score = float(np.sqrt(mean_squared_error(y_val, cb_pred)))

        # LightGBM (CPU, fast)
        try:
            if task_type == 'multiclass':
                lgb = LGBMClassifier(n_estimators=min(n_iter, 300), learning_rate=0.05,
                                     num_leaves=31, verbose=-1, random_state=42, n_jobs=4)
                lgb.fit(X_tr, y_tr)
                lgb_pred = lgb.predict_proba(X_val)
                oof_preds[val_idx] += lgb_pred * 0.3
                if test is not None:
                    test_preds += lgb.predict_proba(test) * 0.3 / n_folds
            elif task_type == 'binary':
                lgb = LGBMClassifier(n_estimators=min(n_iter, 300), learning_rate=0.05,
                                     num_leaves=31, verbose=-1, random_state=42, n_jobs=4)
                lgb.fit(X_tr, y_tr)
                lgb_pred = lgb.predict_proba(X_val)[:, 1]
                oof_preds[val_idx] += lgb_pred * 0.3
                if test is not None:
                    test_preds += lgb.predict_proba(test)[:, 1] * 0.3 / n_folds
            else:
                lgb = LGBMRegressor(n_estimators=min(n_iter, 300), learning_rate=0.05,
                                    num_leaves=31, verbose=-1, random_state=42, n_jobs=4)
                lgb.fit(X_tr, y_tr)
                lgb_pred = lgb.predict(X_val)
                oof_preds[val_idx] += lgb_pred * 0.3
                if test is not None:
                    test_preds += lgb.predict(test) * 0.3 / n_folds
        except Exception as e:
            print(f"LGB:{e}", end=' ')

        # XGBoost (CPU)
        try:
            if task_type == 'multiclass':
                xgb = XGBClassifier(n_estimators=min(n_iter, 300), learning_rate=0.05,
                                    max_depth=6, verbosity=0, random_state=42, n_jobs=4,
                                    eval_metric='mlogloss')
                xgb.fit(X_tr, y_tr)
                xgb_pred = xgb.predict_proba(X_val)
                oof_preds[val_idx] += xgb_pred * 0.2
                if test is not None:
                    test_preds += xgb.predict_proba(test) * 0.2 / n_folds
            elif task_type == 'binary':
                xgb = XGBClassifier(n_estimators=min(n_iter, 300), learning_rate=0.05,
                                    max_depth=6, verbosity=0, random_state=42, n_jobs=4)
                xgb.fit(X_tr, y_tr)
                xgb_pred = xgb.predict_proba(X_val)[:, 1]
                oof_preds[val_idx] += xgb_pred * 0.2
                if test is not None:
                    test_preds += xgb.predict_proba(test)[:, 1] * 0.2 / n_folds
            else:
                xgb = XGBRegressor(n_estimators=min(n_iter, 300), learning_rate=0.05,
                                   max_depth=6, verbosity=0, random_state=42, n_jobs=4)
                xgb.fit(X_tr, y_tr)
                xgb_pred = xgb.predict(X_val)
                oof_preds[val_idx] += xgb_pred * 0.2
                if test is not None:
                    test_preds += xgb.predict(test) * 0.2 / n_folds
        except Exception as e:
            print(f"XGB:{e}", end=' ')

        # Compute fold score using the ensemble OOF
        fold_metric = compute_metric(y_val, oof_preds[val_idx], metric, task_type)
        scores.append(fold_metric)
        print(f"score={fold_metric:.4f} [{time.time()-t0:.0f}s]")

    return oof_preds, test_preds, scores


def make_submission(test_preds, task_id, task_type, sample_sub, id_col_out, pred_col_out, target_encoder):
    """Create correctly-formatted submission file."""
    if task_type == 'multiclass':
        pred_indices = np.argmax(test_preds, axis=1)
        if target_encoder is not None:
            pred_values = target_encoder.inverse_transform(pred_indices)
        else:
            pred_values = pred_indices
    elif task_type == 'binary':
        pred_raw = (test_preds > 0.5).astype(int)
        # Special: spaceship uses False/True
        if pred_col_out == "Transported":
            pred_values = pd.Series(pred_raw).map({0: False, 1: True}).values
        else:
            pred_values = pred_raw
    else:
        # Regression: clip negatives, use raw predictions
        pred_values = np.maximum(test_preds, 0)

    if sample_sub is not None:
        sub = sample_sub.copy()
        if len(sub.columns) >= 2:
            sub.iloc[:, 1] = pred_values
        else:
            sub[sub.columns[0]] = pred_values
        return sub
    else:
        return pd.DataFrame({id_col_out: range(len(pred_values)), pred_col_out: pred_values})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task_id')
    parser.add_argument('--gpu-device', type=int, default=0)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()

    task_id = args.task_id
    t_start = time.time()

    # Load
    result = load_data(task_id)
    if result[0] is None:
        print(json.dumps({"task_id": task_id, "status": "failed", "error": "data_load_failed"}))
        return
    train, test, target, task_type, metric, sample_sub, (id_col_out, pred_col_out) = result

    # Auto-fold selection
    n_folds = args.n_folds
    if n_folds == 5 and len(train) > 500000:
        n_folds = 3
    elif n_folds == 5 and len(train) > 100000:
        n_folds = 4

    print(f"\n{'='*60}")
    print(f"TASK: {task_id} | GPU: {args.gpu_device} | FOLDS: {n_folds} | METRIC: {metric}")
    print(f"{'='*60}")
    print(f"  Train: {train.shape}, Test: {test.shape if test is not None else 'N/A'}")

    # Apply feature engineering hook
    hook = FEATURE_HOOKS.get(task_id)
    if hook:
        train, test = hook(train, test)
        print(f"  Feature engineering applied: {train.shape}")

    # Preprocess
    X_train, y, X_test, target_encoder, id_cols = preprocess(train, test, target, task_type)
    print(f"  Features: {X_train.shape[1]}, Target: {target}")

    if y is None:
        print(json.dumps({"task_id": task_id, "status": "failed", "error": "target_not_found"}))
        return

    # Train ensemble
    oof_preds, test_preds, cv_scores = train_and_predict(
        X_train, y, X_test, task_type, metric,
        gpu_device=args.gpu_device, n_folds=n_folds, fast=args.fast
    )

    oof_score = np.mean(cv_scores)
    oof_std = np.std(cv_scores)

    print(f"\n  CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  OOF {metric.upper()}: {oof_score:.4f} +/- {oof_std:.4f}")

    # Make submission
    sub = make_submission(test_preds, task_id, task_type, sample_sub, id_col_out, pred_col_out, target_encoder)
    sub_path = f"/hpc2hdd/home/aimslab/results/submission_v2_{task_id}.csv"
    sub.to_csv(sub_path, index=False)
    print(f"  Submission: {sub_path} ({len(sub)} rows)")

    elapsed = time.time() - t_start

    result = {
        "task_id": task_id,
        "status": "completed",
        "metric": metric,
        "n_features": X_train.shape[1],
        "n_train": len(X_train),
        "oof_score": round(float(oof_score), 6),
        "oof_std": round(float(oof_std), 6),
        "cv_scores": [round(float(s), 6) for s in cv_scores],
        "n_folds": n_folds,
        "gpu_device": args.gpu_device,
        "elapsed_seconds": round(elapsed, 1),
        "submission_path": sub_path,
    }

    result_path = f"/hpc2hdd/home/aimslab/results/gpu_v2_{task_id}.json"
    with open(result_path, 'w') as f:
        json.dump(result, f)

    print(f"  RESULT: task_id={task_id} oof={oof_score:.4f} elapsed={elapsed:.0f}s")
    print(f"SUMMARY: {task_id} | {metric.upper()}={oof_score:.4f} +/- {oof_std:.4f} | {elapsed:.0f}s | {sub_path}")


if __name__ == '__main__':
    main()
