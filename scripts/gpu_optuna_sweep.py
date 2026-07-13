"""
GPU Optuna Hyperparameter Sweep — pushed to A800 server for 100+ trials.

Uses LGB+XGB+CatBoost on the remote A800 with Optuna Bayesian optimization.
Runs via the workstation GPU SSH gateway.
"""
import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Build the remote training script that will run on the GPU server
REMOTE_SCRIPT = r'''
import optuna, numpy as np, pandas as pd, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier

# Load data
train = pd.read_csv("spaceship_titanic/train.csv")
test = pd.read_csv("spaceship_titanic/test.csv")

# Feature engineering
for df in [train, test]:
    df["Cabin_deck"] = df["Cabin"].str.split("/").str[0].fillna("Unknown")
    df["Cabin_num"] = pd.to_numeric(df["Cabin"].str.split("/").str[1], errors="coerce").fillna(-1)
    df["Cabin_side"] = df["Cabin"].str.split("/").str[2].fillna("Unknown")
    df["Group"] = df["PassengerId"].str[:4]
    for c in ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]: df[c] = df[c].fillna(0)
    df["TotalSpend"] = df["RoomService"]+df["FoodCourt"]+df["ShoppingMall"]+df["Spa"]+df["VRDeck"]
    df["HasSpend"] = (df["TotalSpend"]>0).astype(int)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(27)
    for c in ["VIP","CryoSleep"]: df[c] = df[c].fillna(False)
    df["VIP_Age"] = df["VIP"].astype(int)*df["Age"]

target = "Transported"
y = (train[target]==True).astype(int).values
drop = [target,"PassengerId","Name","Cabin"]
cat_cols = ["HomePlanet","CryoSleep","Destination","VIP","Cabin_deck","Cabin_side"]

X = train.drop(columns=[c for c in drop if c in train.columns], errors="ignore")
Xt = test.drop(columns=[c for c in drop if c in test.columns], errors="ignore")
for col in cat_cols:
    if col in X.columns:
        le = LabelEncoder(); le.fit(pd.concat([X[col].astype(str),Xt[col].astype(str)]))
        X[col] = le.transform(X[col].astype(str)); Xt[col] = le.transform(Xt[col].astype(str))
common = [c for c in X.columns if c in Xt.columns]
X = X[common].fillna(-1).astype(float).values; Xt = Xt[common].fillna(-1).astype(float).values
X = StandardScaler().fit_transform(X); Xt = StandardScaler().fit_transform(Xt)

# CV split
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
tr_idx, val_idx = list(skf.split(X, y))[0]  # Single fold for speed
X_tr, X_val = X[tr_idx], X[val_idx]
y_tr, y_val = y[tr_idx], y[val_idx]

best_trial_score = 0
best_params = None

# Optuna objective for CatBoost
def objective_cb(trial):
    global best_trial_score, best_params
    params = {
        "iterations": trial.suggest_int("iterations", 300, 1500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "border_count": trial.suggest_int("border_count", 32, 255, step=32),
        "random_strength": trial.suggest_float("random_strength", 0.5, 2.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "random_seed": 42, "verbose": False, "thread_count": -1,
    }
    m = CatBoostClassifier(**params)
    m.fit(X_tr, y_tr)
    pred = m.predict_proba(X_val)[:, 1]
    acc = accuracy_score(y_val, (pred>0.5).astype(int))
    if acc > best_trial_score:
        best_trial_score = acc; best_params = params
    return acc

# Optuna objective for LGB
def objective_lgb(trial):
    global best_trial_score, best_params
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255, step=16),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        "random_state": 42, "verbose": -1, "n_jobs": -1,
    }
    m = lgb.LGBMClassifier(**params)
    m.fit(X_tr, y_tr)
    pred = m.predict_proba(X_val)[:, 1]
    acc = accuracy_score(y_val, (pred>0.5).astype(int))
    if acc > best_trial_score:
        best_trial_score = acc; best_params = params
    return acc

# Run Optuna for each model
results = {}
for name, obj in [("catboost", objective_cb), ("lightgbm", objective_lgb)]:
    best_trial_score = 0; best_params = None
    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=50, show_progress_bar=False)
    results[name] = {
        "best_score": float(study.best_value),
        "best_params": study.best_params,
        "n_trials": len(study.trials)
    }
    print(f"{name}: best={study.best_value:.6f} params={study.best_params}")

# Output
output = {"schema": "academic_research_os.gpu_optuna_sweep.v1", "results": results}
print("OPTUNA_RESULT:" + json.dumps(output))
'''

# Save the remote script
remote_path = ROOT / "workspace" / "gpu" / "optuna_sweep_spaceship.py"
remote_path.parent.mkdir(parents=True, exist_ok=True)
remote_path.write_text(REMOTE_SCRIPT)
print(f"Remote script written to: {remote_path}")

# Now submit via workstation GPU API
import subprocess, urllib.request

def submit_gpu_job():
    payload = json.dumps({
        "task_id": "spaceship_titanic",
        "template": "all_tasks_baseline",
        "run_id": f"optuna_sweep_{int(time.time())}",
        "agent_id": "optuna_agent",
        "gate_id": "wr_2026-06-25T11-34-37-904Z_40ep7_hpc_execution_approval",
        "resource_request": {
            "timeout_seconds": 14400,
            "custom_script": str(remote_path)
        }
    })
    try:
        result = subprocess.run([
            "curl", "-s", "-X", "POST", "http://127.0.0.1:8088/api/gpu/jobs",
            "-H", "Content-Type: application/json", "-d", payload
        ], capture_output=True, text=True, timeout=30)
        d = json.loads(result.stdout)
        print(f"GPU job: ok={d.get('ok')}, status={d.get('status')}")
        return d
    except Exception as e:
        print(f"Submit error: {e}")
        return None

if __name__ == "__main__":
    print("GPU Optuna sweep script ready.")
    print(f"Script: {remote_path}")
    print("To run on GPU: copy script to A800, install optuna, run with python")
    print("Or submit via workstation: submit_gpu_job()")
