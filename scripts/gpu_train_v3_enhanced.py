#!/usr/bin/env python3
"""
GPU V3: Anti-overfitting CatBoost training with system gate checks.
- Single CatBoost GPU model (no ensemble overfitting)
- Conservative params (lr=0.02, depth=5, L2=5, early_stopping=50)
- Proper RMSLE/RMSE metrics
- Correct submission formats with test IDs from test.csv
- Gate check: only pass if OOF clears bronze with safety margin
"""
import sys, os, json, time, warnings, argparse
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

HOME = '/hpc2hdd/home/aimslab'

# Competition registry: task_id -> (dir, target, type, metric, direction, bronze_threshold, safety_margin)
COMPETITIONS = {
    "titanic":               ("titanic", "Survived", "binary", "accuracy", "max", 0.794, 0.010),
    "spaceship_titanic":     ("spaceship-titanic", "Transported", "binary", "accuracy", "max", 0.795, 0.005),
    "digit_recognizer":      ("digit-recognizer", "label", "multiclass", "accuracy", "max", 0.986, 0.005),
    "house_prices":          ("house-prices-advanced-regression-techniques", "SalePrice", "regression", "rmsle", "min", 0.140, 0.005),
    "bike_sharing_demand":   ("bike-sharing-demand", "count", "regression", "rmsle", "min", 0.480, 0.020),
    "porto_seguro":          ("porto-seguro-safe-driver-prediction", "target", "binary", "accuracy", "max", 0.285, 0.010),
    "store_sales":           ("store-sales-time-series-forecasting", "sales", "regression", "rmsle", "min", 0.500, 0.020),
    # S3
    "ps3e1":  ("playground-series-s3e1", "MedHouseVal", "regression", "rmse", "min", 0.600, 0.010),
    "ps3e7":  ("playground-series-s3e7", "booking_status", "multiclass", "accuracy", "max", 0.800, 0.010),
    "ps3e25": ("playground-series-s3e25", "Hardness", "multiclass", "accuracy", "max", 0.700, 0.010),
    # S4
    "ps4e1":  ("playground-series-s4e1", "Exited", "binary", "accuracy", "max", 0.750, 0.010),
    "ps4e2":  ("playground-series-s4e2", "NObeyesdad", "multiclass", "accuracy", "max", 0.750, 0.010),
    "ps4e3":  ("playground-series-s4e3", "Other_Faults", "multiclass", "accuracy", "max", 0.700, 0.010),
    "ps4e4":  ("playground-series-s4e4", "Rings", "regression", "rmse", "min", 0.500, 0.020),
    "ps4e6":  ("playground-series-s4e6", "Target", "binary", "accuracy", "max", 0.750, 0.010),
    "ps4e7":  ("playground-series-s4e7", "Response", "binary", "accuracy", "max", 0.600, 0.010),
    # S5
    "ps5e1":  ("playground-series-s5e1", "num_sold", "regression", "rmse", "min", 0.600, 0.020),
    "ps5e2":  ("playground-series-s5e2", "Price", "regression", "rmse", "min", 0.800, 0.020),
    "ps5e3":  ("playground-series-s5e3", "rainfall", "regression", "rmse", "min", 0.700, 0.010),
    "ps5e4":  ("playground-series-s5e4", "Listening_Time_minutes", "regression", "rmse", "min", 0.600, 0.020),
    "ps5e5":  ("playground-series-s5e5", "Calories", "regression", "rmse", "min", 0.600, 0.020),
    # S6
    "ps6e2":  ("playground-series-s6e2", "Heart Disease", "binary", "accuracy", "max", 0.800, 0.010),
    "ps6e3":  ("playground-series-s6e3", "Churn", "binary", "accuracy", "max", 0.800, 0.010),
    "ps6e6":  ("playground-series-s6e6", "class", "multiclass", "accuracy", "max", 0.400, 0.020),
    # Tabular
    "tps_aug2022": ("tabular-playground-series-aug-2022", "failure", "binary", "roc_auc", "max", 0.842, 0.010),
    "tps_dec2021": ("tabular-playground-series-dec-2021", "Cover_Type", "multiclass", "accuracy", "max", 0.800, 0.010),
    "tps_feb2022": ("tabular-playground-series-feb-2022", "target", "binary", "accuracy", "max", 0.800, 0.005),
    "tps_jan2022": ("tabular-playground-series-jan-2022", "num_sold", "regression", "rmse", "min", 0.600, 0.020),
    "tps_mar2022": ("tabular-playground-series-mar-2022", "congestion", "multiclass", "accuracy", "max", 0.700, 0.010),
    "tps_may2022": ("tabular-playground-series-may-2022", "target", "binary", "accuracy", "max", 0.750, 0.010),
}

# Kaggle submission format: (id_col_in_test, pred_col_name, special_format)
SUBMISSION_FORMATS = {
    "titanic": ("PassengerId", "Survived", "int"),
    "spaceship_titanic": ("PassengerId", "Transported", "bool"),
    "digit_recognizer": ("ImageId", "Label", "int"),
    "house_prices": ("Id", "SalePrice", "float"),
    "bike_sharing_demand": ("datetime", "count", "float"),
    "porto_seguro": ("id", "target", "int"),
    "store_sales": ("id", "sales", "float"),
}

# Feature engineering

def titanic_features(train, test):
    for df in [df for df in [train, test] if df is not None]:
        if 'Name' in df.columns:
            df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.', expand=False).fillna('Other')
            # More granular titles
            rare_titles = ['Dr','Rev','Col','Major','Lady','Sir','Don','Jonkheer','Capt','Countess']
            df['Title'] = df['Title'].apply(lambda x: x if x not in rare_titles else 'Rare')
            df['TitleLen'] = df['Name'].str.len()
        if 'SibSp' in df.columns and 'Parch' in df.columns:
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
            df['FamilyType'] = pd.cut(df['FamilySize'], bins=[0,1,2,4,100], labels=['Alone','Small','Medium','Large'])
        if 'Fare' in df.columns:
            df['Fare'] = df['Fare'].fillna(df['Fare'].median())
            df['FareBin'] = pd.qcut(df['Fare'], 5, labels=False, duplicates='drop')
            df['FareLog'] = np.log1p(df['Fare'])
        if 'Age' in df.columns:
            df['Age'] = df['Age'].fillna(df['Age'].median())
            df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,18,25,35,50,65,100], labels=False)
        if 'Age' in df.columns and 'Fare' in df.columns:
            df['AgeFare'] = df['Age'] * df['FareLog'] / 10
        if 'Ticket' in df.columns:
            df['TicketPrefix'] = df['Ticket'].str.extract(r'^([A-Za-z\./]+)', expand=False).fillna('NUM')
            df['TicketPrefix'] = df['TicketPrefix'].apply(lambda x: x if x in ['A','CA','PC','STON','SOTON'] else ('NUM' if x=='NUM' else 'Other'))
            df['TicketLen'] = df['Ticket'].str.len()
        if 'Cabin' in df.columns:
            df['CabinDeck'] = df['Cabin'].str[0].fillna('U')
            df['HasCabin'] = df['Cabin'].notna().astype(int)
            df['CabinNum'] = df['Cabin'].str.extract(r'(\d+)', expand=False).fillna(0).astype(int)
            df['CabinSide'] = df['Cabin'].str[-1].fillna('U')
        if 'Embarked' in df.columns:
            df['Embarked'] = df['Embarked'].fillna('S')
        if 'Pclass' in df.columns:
            df['Pclass'] = df['Pclass'].astype(str)
    return train, test

def spaceship_features(train, test):
    for df in [df for df in [train, test] if df is not None]:
        if 'CryoSleep' in df.columns and 'Age' in df.columns:
            df['CryoSleep'] = df['CryoSleep'].fillna(False).astype(bool)
            df['CryoAge'] = df['CryoSleep'].astype(int) & (df['Age'].fillna(0) < 20).astype(int)
        spend_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
        if all(c in df.columns for c in spend_cols):
            for c in spend_cols: df[c] = df[c].fillna(0)
            df['TotalSpend'] = df[spend_cols].sum(axis=1)
            df['HasSpend'] = (df['TotalSpend'] > 0).astype(int)
            df['LogTotalSpend'] = np.log1p(df['TotalSpend'])
        if 'PassengerId' in df.columns:
            df['Group'] = df['PassengerId'].str.split('_').str[0]
            df['GroupSize'] = df['Group'].map(df['Group'].value_counts())
        if 'Cabin' in df.columns:
            df['Deck'] = df['Cabin'].str[0].fillna('U')
            df['CabinNum'] = df['Cabin'].str.extract(r'(\d+)', expand=False).fillna(0).astype(int)
            df['CabinSide'] = df['Cabin'].str[-1].fillna('U')
    return train, test

FEATURE_HOOKS = {
    "titanic": titanic_features,
    "spaceship_titanic": spaceship_features,
}


def load_and_preprocess(task_id):
    info = COMPETITIONS.get(task_id)
    if not info: return None
    dir_name, target, task_type, metric, direction, bronze, margin = info
    data_dir = os.path.join(HOME, dir_name)
    if not os.path.isdir(data_dir): return None

    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    if not os.path.exists(train_path): return None

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path) if os.path.exists(test_path) else None

    # Apply feature hook
    hook = FEATURE_HOOKS.get(task_id)
    if hook: train, test = hook(train, test)

    # Identify ID columns
    id_cols = [c for c in train.columns if c.lower().replace(' ','_') in ('id','passengerid','imageid')]

    # Extract test IDs
    test_ids = None
    if test is not None and id_cols:
        tid = id_cols[0]
        test_ids = test[tid].values.copy() if tid in test.columns else None

    # Separate target
    target_encoder = None
    y = train[target].copy() if target in train.columns else None
    if y is None: return None
    train_feat = train.drop(columns=[target] + id_cols, errors='ignore')
    test_feat = test.drop(columns=id_cols, errors='ignore') if test is not None else None

    # Encode string target
    if y.dtype == 'object':
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y.astype(str)), index=y.index)

    # Combine and encode
    combined = pd.concat([train_feat, test_feat] if test_feat is not None else [train_feat], ignore_index=True)

    for col in list(combined.columns):
        if combined[col].dtype == 'object':
            if combined[col].nunique() > 100:
                combined.drop(columns=[col], inplace=True)
            else:
                combined[col] = combined[col].fillna('MISSING')
                combined[col] = LabelEncoder().fit_transform(combined[col].astype(str))
        elif combined[col].dtype in ('float64', 'int64'):
            combined[col] = combined[col].fillna(combined[col].median())

    n_train = len(train_feat)
    X_train = combined.iloc[:n_train].copy()
    X_test = combined.iloc[n_train:].copy() if test_feat is not None else None

    # Get submission format
    fmt = SUBMISSION_FORMATS.get(task_id)
    if fmt:
        id_col_out, pred_col_out, val_fmt = fmt
    else:
        id_col_out = id_cols[0] if id_cols else 'id'
        pred_col_out = target
        val_fmt = 'int'

    return X_train, y, X_test, task_type, metric, direction, bronze, margin, test_ids, id_col_out, pred_col_out, val_fmt, target_encoder


def compute_oof_metric(y_true, y_pred, metric, task_type):
    if metric == "rmsle":
        yt = np.maximum(y_true, 0)
        yp = np.maximum(y_pred, 0)
        return float(np.sqrt(np.mean((np.log1p(yp) - np.log1p(yt)) ** 2)))
    elif metric == "rmse":
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    elif metric == "accuracy":
        if task_type == "multiclass":
            return float(accuracy_score(y_true, np.argmax(y_pred, axis=1)))
        return float(accuracy_score(y_true, (y_pred > 0.5).astype(int)))
    elif metric == "roc_auc":
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_pred))
    return 0.0


def train_catboost(X_train, y, X_test, task_type, gpu_device, n_folds, fast):
    n_iter = 300 if fast else 1000
    early_stop = 30 if fast else 80

    if task_type in ('binary', 'multiclass'):
        folds = list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42).split(X_train, y))
    else:
        folds = list(KFold(n_splits=n_folds, shuffle=True, random_state=42).split(X_train))

    n_classes = y.nunique() if task_type == 'multiclass' else 1
    oof = np.zeros(len(X_train)) if task_type != 'multiclass' else np.zeros((len(X_train), n_classes))
    test_preds = np.zeros(len(X_test)) if task_type != 'multiclass' and X_test is not None else (
        np.zeros((len(X_test), n_classes)) if X_test is not None and task_type == 'multiclass' else None)
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

        if task_type == 'multiclass':
            model = CatBoostClassifier(loss_function='MultiClass', **cb_params)
        elif task_type == 'binary':
            model = CatBoostClassifier(loss_function='Logloss', eval_metric='Accuracy', **cb_params)
        else:
            model = CatBoostRegressor(loss_function='RMSE', **cb_params)

        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)

        if task_type == 'multiclass':
            val_pred = model.predict_proba(X_val)
            oof[val_idx] = val_pred
            if X_test is not None: test_preds += model.predict_proba(X_test) / n_folds
            fold_score = compute_oof_metric(y_val, val_pred, "accuracy", task_type)
        elif task_type == 'binary':
            val_pred = model.predict_proba(X_val)[:, 1]
            oof[val_idx] = val_pred
            if X_test is not None: test_preds += model.predict_proba(X_test)[:, 1] / n_folds
            fold_score = compute_oof_metric(y_val, val_pred, "accuracy", task_type)
        else:
            val_pred = model.predict(X_val)
            oof[val_idx] = val_pred
            if X_test is not None: test_preds += model.predict(X_test) / n_folds
            fold_score = compute_oof_metric(y_val, val_pred, "rmse", task_type)

        scores.append(fold_score)
        print(f"score={fold_score:.4f} [{time.time()-t0:.0f}s]")

    return oof, test_preds, scores


def gate_check(oof_score, bronze_threshold, direction, safety_margin):
    """Gate check: OOF must clear bronze with safety margin."""
    if direction == "max":
        passes = oof_score >= bronze_threshold + safety_margin
        gap = bronze_threshold + safety_margin - oof_score
    else:
        passes = oof_score <= bronze_threshold - safety_margin
        gap = oof_score - (bronze_threshold - safety_margin)
    return passes, gap


def make_submission(test_preds, test_ids, id_col_out, pred_col_out, val_fmt, task_type, target_encoder):
    if task_type == 'multiclass':
        pred_indices = np.argmax(test_preds, axis=1)
        if target_encoder is not None:
            pred_values = target_encoder.inverse_transform(pred_indices)
        else:
            pred_values = pred_indices
    elif task_type == 'binary':
        pred_raw = (test_preds > 0.5).astype(int)
        if val_fmt == 'bool':
            pred_values = ['True' if p == 1 else 'False' for p in pred_raw]
        else:
            pred_values = pred_raw
    else:
        pred_values = np.maximum(test_preds, 0)
        if val_fmt == 'float':
            pred_values = [f"{v:.6f}" for v in pred_values]

    if test_ids is not None:
        ids = test_ids[:len(pred_values)]
    else:
        ids = range(len(pred_values))

    sub = pd.DataFrame({id_col_out: ids, pred_col_out: pred_values})
    return sub


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task_id')
    parser.add_argument('--gpu-device', type=int, default=0)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()

    tid = args.task_id
    t_start = time.time()

    result = load_and_preprocess(tid)
    if result is None:
        print(json.dumps({"task_id": tid, "status": "failed", "error": "data_load"}))
        return

    X_train, y, X_test, task_type, metric, direction, bronze, margin, test_ids, id_col_out, pred_col_out, val_fmt, target_encoder = result

    # Auto-fold
    n_folds = args.n_folds
    if n_folds == 5 and len(X_train) > 500000: n_folds = 3
    elif n_folds == 5 and len(X_train) > 100000: n_folds = 4

    print(f"\n{'='*60}")
    print(f"TASK: {tid} | GPU: {args.gpu_device} | FOLDS: {n_folds} | BRONZE: {direction}>={bronze}")
    print(f"{'='*60}")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape if X_test is not None else 'N/A'}")

    oof, test_preds, cv_scores = train_catboost(X_train, y, X_test, task_type, args.gpu_device, n_folds, args.fast)

    oof_score = float(np.mean(cv_scores))
    oof_std = float(np.std(cv_scores))

    # Compute final OOF metric using the correct metric
    final_metric = compute_oof_metric(y, oof, metric, task_type)
    print(f"\n  CV Fold Scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  OOF (fold mean): {oof_score:.4f} +/- {oof_std:.4f}")
    print(f"  OOF ({metric}): {final_metric:.4f}")

    # GATE CHECK
    passed, gap = gate_check(final_metric, bronze, direction, margin)
    gate_status = "PASS" if passed else "FAIL"
    print(f"  GATE: {gate_status} | {metric}={final_metric:.4f} vs bronze={bronze} margin={margin} | gap={gap:+.4f}")

    # Make submission
    sub = make_submission(test_preds, test_ids, id_col_out, pred_col_out, val_fmt, task_type, target_encoder)
    sub_path = f"/hpc2hdd/home/aimslab/results/v3_submission_{tid}.csv"
    sub.to_csv(sub_path, index=False)

    elapsed = time.time() - t_start
    result_json = {
        "task_id": tid, "status": "completed",
        "metric": metric, "direction": direction,
        "oof_score": round(final_metric, 6),
        "oof_std": round(oof_std, 6),
        "cv_scores": [round(float(s), 6) for s in cv_scores],
        "bronze_threshold": bronze,
        "gate_passed": passed,
        "gate_gap": round(float(gap), 6),
        "n_folds": n_folds, "n_features": X_train.shape[1],
        "elapsed_seconds": round(elapsed, 1),
        "submission_path": sub_path,
    }
    result_path = f"/hpc2hdd/home/aimslab/results/v3_result_{tid}.json"
    with open(result_path, 'w') as f: json.dump(result_json, f)

    print(f"  RESULT: {json.dumps({k:result_json[k] for k in ['task_id','oof_score','gate_passed','gate_gap','elapsed_seconds']})}")
    print(f"SUMMARY: {tid} | {metric}={final_metric:.4f} | GATE={gate_status} | {elapsed:.0f}s")


if __name__ == '__main__':
    main()
