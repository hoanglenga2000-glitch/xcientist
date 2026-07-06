#!/usr/bin/env python3
"""
V4 Targeted: Combined approach — single CatBoost with higher iterations
for competitions where V3 was close to bronze.
Also handles remaining untrained competitions.
"""
import sys, os, json, time, warnings, argparse, glob
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

HOME = '/hpc2hdd/home/aimslab'

COMPETITIONS = {
    "titanic": ("titanic", "Survived", "binary", "accuracy", "max", 0.794),
    "spaceship_titanic": ("spaceship-titanic", "Transported", "binary", "accuracy", "max", 0.795),
    "ps4e1": ("playground-series-s4e1", "Exited", "binary", "accuracy", "max", 0.750),
    "ps4e6": ("playground-series-s4e6", "Target", "binary", "accuracy", "max", 0.750),
    "ps3e25": ("playground-series-s3e25", "Hardness", "multiclass", "accuracy", "max", 0.700),
    "ps5e1": ("playground-series-s5e1", "num_sold", "regression", "rmse", "min", 0.600),
    "tps_dec2021": ("tabular-playground-series-dec-2021", "Cover_Type", "multiclass", "accuracy", "max", 0.800),
    "tps_jan2022": ("tabular-playground-series-jan-2022", "num_sold", "regression", "rmse", "min", 0.600),
    "tps_mar2022": ("tabular-playground-series-mar-2022", "congestion", "multiclass", "accuracy", "max", 0.700),
    "bike_sharing_demand": ("bike-sharing-demand", "count", "regression", "rmsle", "min", 0.480),
    "house_prices": ("house-prices-advanced-regression-techniques", "SalePrice", "regression", "rmsle", "min", 0.140),
    "store_sales": ("store-sales-time-series-forecasting", "sales", "regression", "rmsle", "min", 0.500),
    "porto_seguro": ("porto-seguro-safe-driver-prediction", "target", "binary", "accuracy", "max", 0.285),
}

def load_and_preprocess(task_id):
    info = COMPETITIONS.get(task_id)
    if not info: return None
    dir_name, target, task_type, metric, direction, bronze = info
    data_dir = os.path.join(HOME, dir_name)
    if not os.path.isdir(data_dir): return None

    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    if not os.path.exists(train_path): return None

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path) if os.path.exists(test_path) else None

    # Feature engineering for titanic
    if task_id == "titanic":
        for df in [df for df in [train, test] if df is not None]:
            if 'Name' in df.columns:
                df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.', expand=False).fillna('Other')
                df['Title'] = df['Title'].apply(lambda x: x if x in ['Mr','Mrs','Miss','Master','Dr','Rev','Col','Major'] else 'Other')
            if 'SibSp' in df.columns and 'Parch' in df.columns:
                df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
                df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
                df['FamilyGroup'] = pd.cut(df['FamilySize'], bins=[0,1,2,4,20], labels=['Solo','Small','Medium','Large'])
            if 'Fare' in df.columns:
                df['Fare'] = df['Fare'].fillna(df['Fare'].median())
                df['FareBin'] = pd.qcut(df['Fare'], 5, labels=False, duplicates='drop')
                df['FarePerPerson'] = df['Fare'] / df['FamilySize'].clip(lower=1)
            if 'Age' in df.columns:
                df['Age'] = df['Age'].fillna(df['Age'].median())
                df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,18,25,35,50,65,100], labels=False)
            if 'Cabin' in df.columns and df['Cabin'].notna().any():
                df['HasCabin'] = df['Cabin'].notna().astype(int)
                df['CabinDeck'] = df['Cabin'].str[0].fillna('U')
            if 'Embarked' in df.columns:
                df['Embarked'] = df['Embarked'].fillna('S')
            if 'Sex' in df.columns:
                df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
            if 'Ticket' in df.columns:
                df['TicketPrefix'] = df['Ticket'].str.extract(r'^([A-Za-z]+)', expand=False).fillna('NUM')
                df['TicketLen'] = df['Ticket'].str.len()

    # Feature engineering for spaceship
    if task_id == "spaceship_titanic":
        for df in [df for df in [train, test] if df is not None]:
            if 'CryoSleep' in df.columns:
                df['CryoSleep'] = df['CryoSleep'].fillna(False).astype(bool)
            spend_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
            if all(c in df.columns for c in spend_cols):
                for c in spend_cols: df[c] = df[c].fillna(0)
                df['TotalSpend'] = df[spend_cols].sum(axis=1)
                df['HasSpend'] = (df['TotalSpend'] > 0).astype(int)
                df['LogTotalSpend'] = np.log1p(df['TotalSpend'])
            if 'PassengerId' in df.columns:
                df['Group'] = df['PassengerId'].str.split('_').str[0]
                df['GroupSize'] = df['Group'].map(df['Group'].value_counts())
            if 'HomePlanet' in df.columns:
                df['HomePlanet'] = df['HomePlanet'].fillna('Unknown')
            if 'Destination' in df.columns:
                df['Destination'] = df['Destination'].fillna('Unknown')
            if 'VIP' in df.columns:
                df['VIP'] = df['VIP'].fillna(False).astype(int)
            if 'Age' in df.columns:
                df['Age'] = df['Age'].fillna(df['Age'].median())
            if 'Cabin' in df.columns:
                df['Deck'] = df['Cabin'].str[0].fillna('U')
                df['CabinNum'] = df['Cabin'].str.extract(r'(\d+)', expand=False).fillna(0).astype(int)
                df['Side'] = df['Cabin'].str[-1].fillna('U')

    # ID columns
    id_cols = [c for c in train.columns if c.lower().replace(' ','_') in ('id','passengerid','imageid')]

    # Test IDs for submission
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
            if combined[col].nunique() > 200:
                combined.drop(columns=[col], inplace=True)
            else:
                combined[col] = combined[col].fillna('MISSING')
                combined[col] = LabelEncoder().fit_transform(combined[col].astype(str))
        elif combined[col].dtype in ('float64', 'int64'):
            combined[col] = combined[col].fillna(combined[col].median())

    n_train = len(train_feat)
    X_train = combined.iloc[:n_train].copy()
    X_test = combined.iloc[n_train:].copy() if test_feat is not None else None

    # Output column names
    id_col_out = id_cols[0] if id_cols else 'id'
    pred_col_out = target
    val_fmt = 'bool' if task_id == 'spaceship_titanic' else 'int'

    return X_train, y, X_test, task_type, metric, direction, bronze, test_ids, id_col_out, pred_col_out, val_fmt, target_encoder

def compute_metric(y_true, y_pred, metric, task_type):
    if metric == "rmsle":
        yt = np.maximum(y_true, 0); yp = np.maximum(y_pred, 0)
        return float(np.sqrt(np.mean((np.log1p(yp) - np.log1p(yt)) ** 2)))
    elif metric == "rmse":
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    elif metric == "accuracy":
        if task_type == "multiclass":
            return float(accuracy_score(y_true, np.argmax(y_pred, axis=1)))
        return float(accuracy_score(y_true, (y_pred > 0.5).astype(int)))
    return 0.0

def train_hybrid(X_train, y, X_test, task_type, metric, gpu_device, n_folds, fast):
    """CatBoost (GPU, more iterations) + LGBM blend."""
    n_iter = 400 if fast else 1500
    es = 50 if fast else 100

    if task_type in ('binary', 'multiclass'):
        folds = list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42).split(X_train, y))
    else:
        folds = list(KFold(n_splits=n_folds, shuffle=True, random_state=42).split(X_train))

    n_classes = y.nunique() if task_type == 'multiclass' else 1
    oof = np.zeros(len(X_train)) if task_type != 'multiclass' else np.zeros((len(X_train), n_classes))
    test_preds = None
    if X_test is not None:
        test_preds = np.zeros(len(X_test)) if task_type != 'multiclass' else np.zeros((len(X_test), n_classes))
    scores = []

    cb_params = {
        'iterations': n_iter, 'learning_rate': 0.02, 'depth': 6,
        'l2_leaf_reg': 3, 'bootstrap_type': 'Bayesian', 'bagging_temperature': 0.5,
        'task_type': 'GPU', 'devices': str(gpu_device), 'verbose': 0,
        'random_seed': 42, 'allow_writing_files': False,
        'early_stopping_rounds': es, 'use_best_model': True,
    }

    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        print(f"  Fold {fold_idx+1}/{n_folds}...", end=' ', flush=True)
        t0 = time.time()
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        # CatBoost (GPU)
        if task_type == 'multiclass':
            cb = CatBoostClassifier(loss_function='MultiClass', **cb_params)
            cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
            cb_pred = cb.predict_proba(X_val)
            oof[val_idx] = cb_pred * 0.6
            if X_test is not None: test_preds += cb.predict_proba(X_test) * 0.6 / n_folds
        elif task_type == 'binary':
            cb = CatBoostClassifier(loss_function='Logloss', eval_metric='Accuracy', **cb_params)
            cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
            cb_pred = cb.predict_proba(X_val)[:, 1]
            oof[val_idx] = cb_pred * 0.6
            if X_test is not None: test_preds += cb.predict_proba(X_test)[:, 1] * 0.6 / n_folds
        else:
            cb = CatBoostRegressor(loss_function='RMSE', **cb_params)
            cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
            cb_pred = cb.predict(X_val)
            oof[val_idx] = cb_pred * 0.6
            if X_test is not None: test_preds += cb.predict(X_test) * 0.6 / n_folds

        # LightGBM (CPU) — 40% weight
        try:
            if task_type == 'multiclass':
                lgb = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=42, n_jobs=4)
                lgb.fit(X_tr, y_tr)
                oof[val_idx] += lgb.predict_proba(X_val) * 0.4
                if X_test is not None: test_preds += lgb.predict_proba(X_test) * 0.4 / n_folds
            elif task_type == 'binary':
                lgb = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=42, n_jobs=4)
                lgb.fit(X_tr, y_tr)
                oof[val_idx] += lgb.predict_proba(X_val)[:, 1] * 0.4
                if X_test is not None: test_preds += lgb.predict_proba(X_test)[:, 1] * 0.4 / n_folds
            else:
                lgb = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=42, n_jobs=4)
                lgb.fit(X_tr, y_tr)
                oof[val_idx] += lgb.predict(X_val) * 0.4
                if X_test is not None: test_preds += lgb.predict(X_test) * 0.4 / n_folds
        except:
            pass

        fold_score = compute_metric(y_val, oof[val_idx], metric, task_type)
        scores.append(fold_score)
        print(f"score={fold_score:.4f} [{time.time()-t0:.0f}s]")

    return oof, test_preds, scores

def make_submission(test_preds, test_ids, id_col_out, pred_col_out, val_fmt, task_type, target_encoder):
    if task_type == 'multiclass':
        pred_indices = np.argmax(test_preds, axis=1)
        pred_values = target_encoder.inverse_transform(pred_indices) if target_encoder else pred_indices
    elif task_type == 'binary':
        pred_raw = (test_preds > 0.5).astype(int)
        pred_values = ['True' if p == 1 else 'False' for p in pred_raw] if val_fmt == 'bool' else pred_raw
    else:
        pred_values = np.maximum(test_preds, 0)

    ids = test_ids[:len(pred_values)] if test_ids is not None else range(len(pred_values))
    return pd.DataFrame({id_col_out: ids, pred_col_out: pred_values})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task_id')
    parser.add_argument('--gpu-device', type=int, default=0)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()

    tid = args.task_id; t0 = time.time()
    r = load_and_preprocess(tid)
    if r is None:
        print(json.dumps({"task_id": tid, "status": "failed"}))
        return
    X_train, y, X_test, task_type, metric, direction, bronze, test_ids, id_col_out, pred_col_out, val_fmt, target_encoder = r

    n_folds = args.n_folds
    if n_folds == 5 and len(X_train) > 500000: n_folds = 3
    elif n_folds == 5 and len(X_train) > 100000: n_folds = 4

    print(f"\n{'='*60}")
    print(f"V4: {tid} | GPU:{args.gpu_device} | FOLDS:{n_folds} | TARGET:{bronze}")
    print(f"{'='*60}")
    print(f"  Train: {X_train.shape}, Features: {X_train.shape[1]}")

    oof, test_preds, scores = train_hybrid(X_train, y, X_test, task_type, metric, args.gpu_device, n_folds, args.fast)

    final_metric = compute_metric(y, oof, metric, task_type)
    oof_mean = float(np.mean(scores))
    passed = (direction == "max" and final_metric >= bronze) or (direction == "min" and final_metric <= bronze)

    print(f"\n  CV: {[f'{s:.4f}' for s in scores]}")
    print(f"  OOF: {final_metric:.4f} | GATE: {'PASS' if passed else 'FAIL'} | Bronze: {bronze}")

    sub = make_submission(test_preds, test_ids, id_col_out, pred_col_out, val_fmt, task_type, target_encoder)
    sub_path = f"/hpc2hdd/home/aimslab/results/v4_submission_{tid}.csv"
    sub.to_csv(sub_path, index=False)

    result = {
        "task_id": tid, "status": "completed", "metric": metric,
        "oof_score": round(final_metric, 6), "oof_mean": round(oof_mean, 6),
        "gate_passed": passed, "n_features": X_train.shape[1],
        "elapsed_seconds": round(time.time()-t0, 1), "submission_path": sub_path,
    }
    with open(f"/hpc2hdd/home/aimslab/results/v4_result_{tid}.json", 'w') as f:
        json.dump(result, f)

    print(f"SUMMARY: {tid} | {metric}={final_metric:.4f} | GATE={'PASS' if passed else 'FAIL'} | {time.time()-t0:.0f}s")

if __name__ == '__main__':
    main()
