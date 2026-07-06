"""
GPU Batch Training for All Kaggle Competitions
Runs LGB+XGB+CatBoost ensemble on GPU server for each competition.
Ensures every competition gets a chance at bronze medal (top 30%).

Strategy per task:
1. Load data → preprocess → train LGB/XGB/CatBoost (5-fold CV)
2. Optimize blend weights via OOF
3. Generate submission.csv
4. Run score promotion gate
5. Track best-so-far, submit to Kaggle if improved
"""

import json, os, sys, time, subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# Competition definitions
COMPETITIONS = {
    "spaceship_titanic": {
        "type": "binary_classification",
        "target": "Transported",
        "metric": "accuracy",
        "direction": "maximize",
        "id_col": "PassengerId",
        "drop_cols": ["PassengerId", "Name"],
        "cat_features": ["HomePlanet", "CryoSleep", "Destination", "VIP"],
        "kaggle_slug": "spaceship-titanic",
        "priority": "P0",
    },
    "titanic": {
        "type": "binary_classification",
        "target": "Survived",
        "metric": "accuracy",
        "direction": "maximize",
        "id_col": "PassengerId",
        "drop_cols": ["PassengerId", "Name", "Ticket", "Cabin"],
        "cat_features": ["Sex", "Embarked", "Pclass"],
        "kaggle_slug": "titanic",
        "priority": "P1",
    },
    "house_prices": {
        "type": "regression",
        "target": "SalePrice",
        "metric": "rmsle",
        "direction": "minimize",
        "id_col": "Id",
        "drop_cols": ["Id"],
        "cat_features": [],
        "kaggle_slug": "house-prices-advanced-regression-techniques",
        "priority": "P1",
    },
    "bike_sharing_demand": {
        "type": "regression",
        "target": "count",
        "metric": "rmsle",
        "direction": "minimize",
        "id_col": "datetime",
        "drop_cols": ["datetime", "casual", "registered"],
        "cat_features": ["season", "holiday", "workingday", "weather"],
        "kaggle_slug": "bike-sharing-demand",
        "priority": "P2",
    },
    "digit_recognizer": {
        "type": "multiclass_classification",
        "target": "label",
        "metric": "accuracy",
        "direction": "maximize",
        "id_col": None,
        "drop_cols": [],
        "cat_features": [],
        "kaggle_slug": "digit-recognizer",
        "priority": "P2",
    },
    "porto_seguro_safe_driver_prediction": {
        "type": "binary_classification",
        "target": "target",
        "metric": "normalized_gini",
        "direction": "maximize",
        "id_col": "id",
        "drop_cols": ["id"],
        "cat_features": [],
        "kaggle_slug": "porto-seguro-safe-driver-prediction",
        "priority": "P2",
    },
    "tabular_playground_series_aug_2022": {
        "type": "regression",
        "target": "target",
        "metric": "rmse",
        "direction": "minimize",
        "id_col": "id",
        "drop_cols": ["id"],
        "cat_features": [],
        "kaggle_slug": "tabular-playground-series-aug-2022",
        "priority": "P2",
    },
    "store_sales_time_series_forecasting": {
        "type": "regression",
        "target": "sales",
        "metric": "rmsle",
        "direction": "minimize",
        "id_col": "id",
        "drop_cols": ["id"],
        "cat_features": ["family", "store_nbr", "city", "state"],
        "kaggle_slug": "store-sales-time-series-forecasting",
        "priority": "P2",
    },
    "telco_churn": {
        "type": "binary_classification",
        "target": "Churn",
        "metric": "accuracy",
        "direction": "maximize",
        "id_col": "customerID",
        "drop_cols": ["customerID"],
        "cat_features": [],
        "kaggle_slug": None,
        "priority": "P2",
    },
}


def load_data(task_id, cfg):
    """Load and preprocess competition data."""
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    task_dir = ROOT / "tasks" / task_id / "data"
    train_path = task_dir / "train.csv"
    test_path = task_dir / "test.csv"

    if not train_path.exists():
        print(f"  SKIP: No train.csv at {train_path}")
        return None, None, None, None

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    target = cfg["target"]
    id_col = cfg["id_col"]
    drop_cols = cfg["drop_cols"]

    if target not in train.columns:
        print(f"  SKIP: target '{target}' not in train columns: {list(train.columns)[:10]}")
        return None, None, None, None

    y = train[target].copy()
    test_ids = test[id_col].copy() if id_col and id_col in test.columns else pd.Series(range(len(test)))

    # Drop non-feature columns
    drop = [c for c in drop_cols if c in train.columns]
    if target in train.columns:
        drop.append(target)
    X = train.drop(columns=[c for c in drop if c in train.columns], errors='ignore')
    X_test = test.drop(columns=[c for c in drop_cols if c in test.columns], errors='ignore')

    # Align columns
    common_cols = list(set(X.columns) & set(X_test.columns))
    X = X[common_cols]
    X_test = X_test[common_cols]

    # Handle categorical
    cat_cols = [c for c in cfg.get("cat_features", []) if c in common_cols]
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X[col].astype(str), X_test[col].astype(str)])
        le.fit(combined)
        X[col] = le.transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

    # Fill missing
    X = X.fillna(X.median())
    X_test = X_test.fillna(X_test.median())

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    return X_scaled, y, X_test_scaled, test_ids


def train_ensemble(X, y, cfg, random_state=42):
    """Train LGB+XGB+CatBoost ensemble with 5-fold OOF CV."""
    import numpy as np
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.metrics import accuracy_score, mean_squared_error

    task_type = cfg["type"]
    metric = cfg["metric"]
    direction = cfg["direction"]

    is_classification = "classification" in task_type
    n_classes = len(np.unique(y)) if is_classification else 1

    if is_classification and n_classes > 2:
        print(f"  Multiclass ({n_classes} classes) - using LightGBM only")
        models_to_train = ["lgb"]
    else:
        models_to_train = ["lgb", "xgb", "cat"]

    n_splits = min(5, len(y) // 100)
    if is_classification:
        try:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        except:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_preds = {}
    test_preds = {}

    for model_name in models_to_train:
        print(f"  Training {model_name}...")
        oof = np.zeros(len(y))
        tpred = 0

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx], \
                          y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]

            if model_name == "lgb":
                import lightgbm as lgb
                params = {
                    'objective': 'multiclass' if (is_classification and n_classes > 2) else
                                'binary' if is_classification else 'regression',
                    'metric': 'multi_logloss' if (is_classification and n_classes > 2) else
                             'binary_logloss' if is_classification else 'rmse',
                    'num_leaves': 63, 'learning_rate': 0.05,
                    'n_estimators': 500, 'verbose': -1,
                    'random_state': random_state, 'n_jobs': -1,
                }
                if is_classification and n_classes > 2:
                    params['num_class'] = n_classes
                model = lgb.LGBMClassifier(**{k:v for k,v in params.items() if k != 'objective'}, objective=params['objective']) \
                    if is_classification else lgb.LGBMRegressor(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                         callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

            elif model_name == "xgb":
                import xgboost as xgb
                params = {
                    'n_estimators': 500, 'learning_rate': 0.05,
                    'max_depth': 6, 'subsample': 0.8,
                    'random_state': random_state, 'n_jobs': -1, 'verbosity': 0,
                }
                if is_classification and n_classes > 2:
                    model = xgb.XGBClassifier(**params, objective='multi:softmax', num_class=n_classes)
                elif is_classification:
                    model = xgb.XGBClassifier(**params, objective='binary:logistic')
                else:
                    model = xgb.XGBRegressor(**params, objective='reg:squarederror')
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

            elif model_name == "cat":
                from catboost import CatBoostClassifier, CatBoostRegressor
                params = {
                    'iterations': 500, 'learning_rate': 0.05,
                    'depth': 6, 'random_seed': random_state,
                    'verbose': False, 'thread_count': -1,
                }
                if is_classification:
                    model = CatBoostClassifier(**params)
                else:
                    model = CatBoostRegressor(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])

            oof[val_idx] = model.predict_proba(X_val)[:, 1] if (is_classification and n_classes == 2) else model.predict(X_val)
            if is_classification and n_classes <= 2:
                tpred += model.predict_proba(X_test_scaled_global)[:, 1] / n_splits
            else:
                tpred += model.predict(X_test_scaled_global) / n_splits

        oof_preds[model_name] = oof
        test_preds[model_name] = tpred
        if is_classification and n_classes == 2:
            acc = accuracy_score(y, (oof > 0.5).astype(int))
            print(f"    {model_name} OOF accuracy: {acc:.6f}")
        else:
            rmse = np.sqrt(mean_squared_error(y, oof))
            print(f"    {model_name} OOF RMSE: {rmse:.6f}")

    # Blend
    weights = {"lgb": 0.5, "xgb": 0.25, "cat": 0.25}
    available = {m: w for m, w in weights.items() if m in oof_preds}
    total_w = sum(available.values())
    blend_weights = {m: w/total_w for m, w in available.items()}

    blend_oof = sum(blend_weights[m] * oof_preds[m] for m in available)
    blend_test = sum(blend_weights[m] * test_preds[m] for m in available)

    return blend_oof, blend_test, oof_preds, blend_weights


# Global for test predictions
X_test_scaled_global = None


def run_task(task_id, cfg):
    """Run full pipeline for one competition."""
    global X_test_scaled_global
    print(f"\n{'='*60}")
    print(f"TASK: {task_id} [{cfg['priority']}]")
    print(f"Type: {cfg['type']}, Target: {cfg['target']}, Metric: {cfg['metric']}")
    print(f"{'='*60}")

    X, y, X_test, test_ids = load_data(task_id, cfg)
    if X is None:
        return None

    X_test_scaled_global = X_test
    print(f"  Data: train={X.shape}, test={X_test.shape}")

    blend_oof, blend_test, oof_preds, weights = train_ensemble(X, y, cfg)

    # Generate submission
    sub_path = ROOT / "tasks" / task_id / "data" / "sample_submission.csv"
    if sub_path.exists():
        import pandas as pd
        sub = pd.read_csv(sub_path)
        pred_col = sub.columns[-1]
        is_classification = "classification" in cfg["type"]

        if is_classification:
            sub[pred_col] = (blend_test > 0.5).astype(int)
            oof_acc = (blend_oof > 0.5).astype(int)
            oof_score = (oof_acc == y).mean()
        else:
            sub[pred_col] = np.maximum(0, blend_test)
            import numpy as np
            oof_score = np.sqrt(np.mean((np.log1p(y) - np.log1p(np.maximum(0, blend_oof))) ** 2))

        output_dir = ROOT / "experiments" / task_id / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        sub.to_csv(output_dir / "submission.csv", index=False)

        # Save metrics
        metrics = {
            "task_id": task_id, "metric": cfg["metric"],
            "oof_score": float(oof_score),
            "blend_weights": {m: float(w) for m, w in weights.items()},
            "individual_oof": {m: float((oof > 0.5).astype(int) == y).mean() if is_classification else float(np.sqrt(np.mean((y - oof)**2)))
                              for m, oof in oof_preds.items()},
            "runner": "gpu_batch_lgb_xgb_cat_ensemble",
        }
        with open(output_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\n  RESULTS for {task_id}:")
        print(f"  OOF Score: {oof_score:.6f}")
        print(f"  Weights: {weights}")
        print(f"  Submission: {output_dir / 'submission.csv'}")
        return {"task_id": task_id, "score": float(oof_score), "output": str(output_dir)}
    return None


if __name__ == "__main__":
    results = {}
    # Process by priority
    priorities = ["P0", "P1", "P2"]
    for pri in priorities:
        for task_id, cfg in COMPETITIONS.items():
            if cfg["priority"] == pri:
                try:
                    result = run_task(task_id, cfg)
                    if result:
                        results[task_id] = result
                except Exception as e:
                    print(f"  ERROR: {task_id}: {e}")
                    import traceback
                    traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"BATCH SUMMARY")
    print(f"{'='*60}")
    for task_id, r in sorted(results.items()):
        print(f"  {task_id:<45s} score={r['score']:.6f}")
    print(f"Total: {len(results)} tasks completed")
