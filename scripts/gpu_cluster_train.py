"""Single-file GPU training: no config dependencies, works on bare cluster."""
import pandas as pd, numpy as np, json, sys, os
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_log_error
from datetime import datetime

# Task definitions (hardcoded, no config file needed)
TASKS = {
    "house_prices": {"target": "SalePrice", "metric": "rmsle", "is_clf": False},
    "spaceship_titanic": {"target": "Transported", "metric": "accuracy", "is_clf": True},
    "telco_churn": {"target": "Churn", "metric": "accuracy", "is_clf": True},
    "titanic": {"target": "Survived", "metric": "accuracy", "is_clf": True},
    "bike_sharing_demand": {"target": "count", "metric": "rmsle", "is_clf": False},
    "tabular_playground_series_aug_2022": {"target": "failure", "metric": "roc_auc", "is_clf": True},
}

HOME = "/hpc2hdd/home/aimslab"

task_id = sys.argv[1]
gpu_count = int(sys.argv[2]) if len(sys.argv) > 2 else 1

task = TASKS[task_id]
target = task["target"]
is_clf = task["is_clf"]
metric_name = task["metric"]

# Find data on shared filesystem
data_dir = f"{HOME}/{task_id}"
train_path = f"{data_dir}/train.csv"
test_path = f"{data_dir}/test.csv"
if not os.path.exists(train_path):
    train_path = f"{HOME}/tasks/{task_id}/data/train.csv"
    test_path = f"{HOME}/tasks/{task_id}/data/test.csv"

print(f"Task: {task_id} | GPU: {gpu_count} | Data: {train_path}")
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
train = train.fillna(-999)
test = test.fillna(-999)

# Label-encode objects
for c in train.columns:
    if train[c].dtype == 'object' and c != target:
        le = LabelEncoder()
        all_vals = list(train[c].astype(str).unique()) + list(test[c].astype(str).unique())
        le.fit(all_vals)
        train[c] = le.transform(train[c].astype(str))
        test[c] = le.transform(test[c].astype(str))

# Handle target
if target in train.columns:
    y = train[target].astype(int).values if is_clf else train[target].astype(float).values
    drop_cols = [target] + [c for c in ['id','Id','ID','PassengerId','ImageId','datetime','casual','registered'] if c in train.columns]
else:
    y = np.zeros(len(train))
    drop_cols = []

X_cols = [c for c in train.columns if c not in drop_cols and c in test.columns]
X = train[X_cols].values.astype(np.float32)
X_test = test[X_cols].values.astype(np.float32)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

print(f"X: {X.shape}, X_test: {X_test.shape}, y: {len(y)}")

# Train
n_folds = 5 if len(y) > 500 else 10
cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42) if is_clf else KFold(n_splits=n_folds, shuffle=True, random_state=42)
oof = np.zeros(len(y))
test_preds = np.zeros(len(X_test))

for fold, (tr, va) in enumerate(cv.split(X, y)):
    if is_clf:
        m = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=7,
                               task_type="GPU", devices=list(range(min(gpu_count, 4))),
                               random_seed=42+fold, verbose=50 if fold==0 else 0)
        m.fit(X[tr], y[tr], verbose=50 if fold==0 else 0)
        oof[va] = m.predict_proba(X[va])[:, 1]
        test_preds += m.predict_proba(X_test)[:, 1] / n_folds
        acc = accuracy_score(y[va], (oof[va] > 0.5).astype(int))
        print(f"Fold{fold+1}: acc={acc:.4f}")
    else:
        m = CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=7,
                              task_type="GPU", devices=list(range(min(gpu_count, 4))),
                              random_seed=42+fold, verbose=50 if fold==0 else 0)
        m.fit(X[tr], y[tr], verbose=50 if fold==0 else 0)
        oof[va] = m.predict(X[va])
        test_preds += m.predict(X_test) / n_folds
        rmsle = np.sqrt(mean_squared_log_error(np.abs(y[va]), np.abs(np.clip(oof[va], 0, None))))
        print(f"Fold{fold+1}: rmsle={rmsle:.4f}")

# Final
if is_clf:
    score = float(accuracy_score(y, (oof > 0.5).astype(int)))
    pred = (test_preds > 0.5).astype(int)
else:
    score = float(np.sqrt(mean_squared_log_error(np.abs(y), np.abs(np.clip(oof, 0, None)))))
    pred = np.clip(test_preds, 0, None)

print(f"SCORE: {task_id} {metric_name}={score:.5f}")

json.dump({
    "task": task_id, "gpus": gpu_count, "score": score, "metric": metric_name,
    "time": datetime.now().isoformat(), "pred_shape": pred.shape
}, open(f"/tmp/gpu_result_{task_id}.json", "w"))
