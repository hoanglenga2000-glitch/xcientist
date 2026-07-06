#!/usr/bin/env python3
"""
Deploy Optuna Hyperparameter Sweeps to TWO GPU Servers.
Server1 (A800 80GB): LightGBM 50-trial 5-fold Optuna
Server2 (A40 48GB): CatBoost 50-trial 5-fold Optuna
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import socks
import paramiko

# ── Configuration ──────────────────────────────────────────────────────────
LOGIN_HOST = "100.85.169.63"
LOGIN_PORT = 1235
LOGIN_USER = "aimslab-IwkteXqP"
SOCKS5_HOST = "127.0.0.1"
SOCKS5_PORT = 7890

# Inner server configs
SERVER1 = {
    "name": "server1-A800",
    "ssh_host": "10.120.18.240",
    "ssh_port": 6988,
    "ssh_user": "aimslab-IwkteXqP",
    "gpu": "A800 80GB",
}
SERVER2 = {
    "name": "server2-A40",
    "ssh_host": "10.120.18.240",
    "ssh_port": 6988,
    "ssh_user": "aimslab-kdd-ai4s",
    "gpu": "A40 48GB",
}

# Data path on the shared filesystem
DATA_DIR = "/hpc2hdd/home/aimslab/spaceship_titanic"
TRAIN_CSV = f"{DATA_DIR}/train.csv"
TEST_CSV = f"{DATA_DIR}/test.csv"
WORK_DIR = "/hpc2hdd/home/aimslab/optuna_sweeps"
SWEEP_LOG = "/tmp/optuna_sweep_launch.log"

SCRIPTS_DIR = Path(__file__).resolve().parent


def get_password() -> str:
    ps_script = r'$cred = Import-Clixml -Path "$env:APPDATA\ResearchAgentWorkstation\hpc_ssh_credential.xml"; Write-Output $cred.GetNetworkCredential().Password'
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(f"DPAPI failed: {result.stderr}")
    return result.stdout.strip()


def ssh_to_login(password: str) -> paramiko.SSHClient:
    """Connect to the login node via SOCKS5 proxy."""
    sock = socks.socksocket()
    sock.set_proxy(socks.SOCKS5, SOCKS5_HOST, SOCKS5_PORT)
    sock.settimeout(30)
    sock.connect((LOGIN_HOST, LOGIN_PORT))

    transport = paramiko.Transport(sock)
    transport.connect(username=LOGIN_USER, password=password)

    ssh = paramiko.SSHClient()
    ssh._transport = transport
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    return ssh


def exec_remote(ssh: paramiko.SSHClient, cmd: str, timeout: int = 60) -> tuple[int, str, str]:
    """Execute command on remote via SSH and return (exit_code, stdout, stderr)."""
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return exit_code, out, err


def exec_on_inner(login_ssh: paramiko.SSHClient, inner_host: str, inner_port: int,
                   inner_user: str, cmd: str, timeout: int = 120) -> tuple[int, str, str]:
    """Execute command on an inner server by SSHing from the login node."""
    ssh_cmd = f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {inner_port} {inner_user}@{inner_host} '{cmd}'"
    return exec_remote(login_ssh, ssh_cmd, timeout=timeout)


def check_environment(login_ssh: paramiko.SSHClient, server: dict) -> dict:
    """Check Python environment and GPU on an inner server."""
    name = server["name"]
    print(f"\n{'='*60}")
    print(f"  CHECKING ENVIRONMENT: {name} ({server['gpu']})")
    print(f"{'='*60}")

    info = {"server": name, "gpu": server["gpu"]}

    # Check Python
    _, out, _ = exec_on_inner(login_ssh, server["ssh_host"], server["ssh_port"],
                               server["ssh_user"], "which python3 && python3 --version 2>&1")
    info["python"] = out.strip()
    print(f"  Python: {out.strip()}")

    # Check packages
    _, out, _ = exec_on_inner(login_ssh, server["ssh_host"], server["ssh_port"],
                               server["ssh_user"],
                               "python3 -c \"import catboost, lightgbm, xgboost, optuna; print('all ok')\" 2>&1")
    info["packages_all"] = "all ok" in out
    print(f"  Packages (catboost+lgb+xgb+optuna): {'OK' if info['packages_all'] else 'MISSING: ' + out.strip()}")

    # Check individual packages
    for pkg in ["catboost", "lightgbm", "xgboost", "optuna", "pandas", "numpy", "sklearn"]:
        _, out, _ = exec_on_inner(login_ssh, server["ssh_host"], server["ssh_port"],
                                   server["ssh_user"],
                                   f"python3 -c \"import {pkg}; print('{pkg}=' + getattr({pkg}, '__version__', 'ok'))\" 2>&1")
        info[f"pkg_{pkg}"] = out.strip()
        print(f"    {pkg}: {out.strip()}")

    # Check GPU
    _, out, _ = exec_on_inner(login_ssh, server["ssh_host"], server["ssh_port"],
                               server["ssh_user"], "nvidia-smi 2>&1 | head -20")
    info["nvidia_smi"] = out.strip()
    print(f"  GPU:\n{out.strip()}")

    # Check workspace
    _, out, _ = exec_on_inner(login_ssh, server["ssh_host"], server["ssh_port"],
                               server["ssh_user"], "ls /hpc2hdd/home/ 2>&1")
    info["workspace"] = out.strip()
    print(f"  Workspace: /hpc2hdd/home/ contents: {out.strip()[:200]}")

    # Check data files
    _, out, _ = exec_on_inner(login_ssh, server["ssh_host"], server["ssh_port"],
                               server["ssh_user"],
                               f"ls -la {TRAIN_CSV} {TEST_CSV} 2>&1")
    info["data_files"] = out.strip()
    print(f"  Data files: {out.strip()}")

    return info


# ── Optuna Sweep Scripts ──────────────────────────────────────────────────

LIGHTGBM_SWEEP_SCRIPT = r'''
"""LightGBM 50-trial 5-fold Optuna sweep for Spaceship Titanic."""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb

print("=== LightGBM Optuna Sweep ===", flush=True)
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

# Load data
DATA_DIR = os.environ.get("DATA_DIR", "/hpc2hdd/home/aimslab/spaceship_titanic")
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
print(f"Train: {train.shape}, Test: {test.shape}", flush=True)

# Feature engineering
for df in [train, test]:
    df["Cabin_deck"] = df["Cabin"].str.split("/").str[0].fillna("Unknown")
    df["Cabin_num"] = pd.to_numeric(df["Cabin"].str.split("/").str[1], errors="coerce").fillna(-1)
    df["Cabin_side"] = df["Cabin"].str.split("/").str[2].fillna("Unknown")
    df["Group"] = df["PassengerId"].str[:4]
    for c in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]:
        df[c] = df[c].fillna(0)
    df["TotalSpend"] = df["RoomService"] + df["FoodCourt"] + df["ShoppingMall"] + df["Spa"] + df["VRDeck"]
    df["HasSpend"] = (df["TotalSpend"] > 0).astype(int)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(27)
    for c in ["VIP", "CryoSleep"]:
        df[c] = df[c].fillna(False)
    df["VIP_Age"] = df["VIP"].astype(int) * df["Age"]

target = "Transported"
y = (train[target] == True).astype(int).values
drop_cols = [target, "PassengerId", "Name", "Cabin"]
cat_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Cabin_deck", "Cabin_side"]

X = train.drop(columns=[c for c in drop_cols if c in train.columns], errors="ignore")
Xt = test.drop(columns=[c for c in drop_cols if c in test.columns], errors="ignore")

for col in cat_cols:
    if col in X.columns:
        le = LabelEncoder()
        le.fit(pd.concat([X[col].astype(str), Xt[col].astype(str)]))
        X[col] = le.transform(X[col].astype(str))
        Xt[col] = le.transform(Xt[col].astype(str))

common = [c for c in X.columns if c in Xt.columns]
X = X[common].fillna(-1).astype(float).values
Xt = Xt[common].fillna(-1).astype(float).values

scaler = StandardScaler()
X = scaler.fit_transform(X)
Xt = scaler.transform(Xt)

print(f"Feature matrix: {X.shape}, target: {y.shape}", flush=True)

# 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_overall_score = 0.0
best_overall_params = None
best_overall_fold = -1

def objective(trial):
    global best_overall_score, best_overall_params, best_overall_fold
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 3000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255, step=16),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
        "force_col_wise": True,
    }
    scores = []
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr)
        pred = m.predict_proba(X_val)[:, 1]
        acc = accuracy_score(y_val, (pred > 0.5).astype(int))
        scores.append(acc)

    mean_acc = float(np.mean(scores))
    if mean_acc > best_overall_score:
        best_overall_score = mean_acc
        best_overall_params = params.copy()
    return mean_acc

# Run Optuna
study = optuna.create_study(direction="maximize", study_name="lgbm_5fold")
study.optimize(objective, n_trials=50, show_progress_bar=True, n_jobs=1)

print(f"\nBest score: {study.best_value:.6f}", flush=True)
print(f"Best params: {json.dumps(study.best_params, indent=2)}", flush=True)

# Save results
results = {
    "model": "LightGBM",
    "server": os.uname().nodename if hasattr(os, "uname") else "unknown",
    "best_score": float(study.best_value),
    "best_params": study.best_params,
    "n_trials": len(study.trials),
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "trials": [
        {"number": t.number, "value": float(t.value), "params": t.params}
        for t in study.trials if t.value is not None
    ],
}

OUT_DIR = os.environ.get("OUT_DIR", "/hpc2hdd/home/aimslab/optuna_sweeps/server1-A800-lightgbm")
os.makedirs(OUT_DIR, exist_ok=True)
out_path = f"{OUT_DIR}/results_{time.strftime('%Y%m%d_%H%M%S')}.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to: {out_path}", flush=True)

# Also save best params as standalone
best_path = f"{OUT_DIR}/best_params.json"
with open(best_path, "w") as f:
    json.dump(results["best_params"], f, indent=2)

# Save study
study_path = f"{OUT_DIR}/optuna_study.pkl"
import pickle
with open(study_path, "wb") as f:
    pickle.dump(study, f)

print("=== LightGBM Sweep COMPLETE ===", flush=True)
'''

CATBOOST_SWEEP_SCRIPT = r'''
"""CatBoost 50-trial 5-fold Optuna sweep for Spaceship Titanic."""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from catboost import CatBoostClassifier, Pool

print("=== CatBoost Optuna Sweep ===", flush=True)
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

# Load data
DATA_DIR = os.environ.get("DATA_DIR", "/hpc2hdd/home/aimslab/spaceship_titanic")
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
print(f"Train: {train.shape}, Test: {test.shape}", flush=True)

# Feature engineering
for df in [train, test]:
    df["Cabin_deck"] = df["Cabin"].str.split("/").str[0].fillna("Unknown")
    df["Cabin_num"] = pd.to_numeric(df["Cabin"].str.split("/").str[1], errors="coerce").fillna(-1)
    df["Cabin_side"] = df["Cabin"].str.split("/").str[2].fillna("Unknown")
    df["Group"] = df["PassengerId"].str[:4]
    for c in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]:
        df[c] = df[c].fillna(0)
    df["TotalSpend"] = df["RoomService"] + df["FoodCourt"] + df["ShoppingMall"] + df["Spa"] + df["VRDeck"]
    df["HasSpend"] = (df["TotalSpend"] > 0).astype(int)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(27)
    for c in ["VIP", "CryoSleep"]:
        df[c] = df[c].fillna(False)
    df["VIP_Age"] = df["VIP"].astype(int) * df["Age"]

target = "Transported"
y = (train[target] == True).astype(int).values
drop_cols = [target, "PassengerId", "Name", "Cabin"]
cat_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Cabin_deck", "Cabin_side"]

X = train.drop(columns=[c for c in drop_cols if c in train.columns], errors="ignore")
Xt = test.drop(columns=[c for c in drop_cols if c in test.columns], errors="ignore")

for col in cat_cols:
    if col in X.columns:
        le = LabelEncoder()
        le.fit(pd.concat([X[col].astype(str), Xt[col].astype(str)]))
        X[col] = le.transform(X[col].astype(str))
        Xt[col] = le.transform(Xt[col].astype(str))

common = [c for c in X.columns if c in Xt.columns]
X = X[common].fillna(-1).astype(float).values
Xt = Xt[common].fillna(-1).astype(float).values

scaler = StandardScaler()
X = scaler.fit_transform(X)
Xt = scaler.transform(Xt)

print(f"Feature matrix: {X.shape}, target: {y.shape}", flush=True)

# 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 500, 3000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 20.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255, step=32),
        "random_strength": trial.suggest_float("random_strength", 0.1, 5.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
        "allow_writing_files": False,
    }
    scores = []
    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        m = CatBoostClassifier(**params)
        m.fit(X_tr, y_tr, verbose=False)
        pred = m.predict_proba(X_val)[:, 1]
        acc = accuracy_score(y_val, (pred > 0.5).astype(int))
        scores.append(acc)
    return float(np.mean(scores))

# Run Optuna
study = optuna.create_study(direction="maximize", study_name="catboost_5fold")
study.optimize(objective, n_trials=50, show_progress_bar=True, n_jobs=1)

print(f"\nBest score: {study.best_value:.6f}", flush=True)
print(f"Best params: {json.dumps(study.best_params, indent=2)}", flush=True)

# Save results
results = {
    "model": "CatBoost",
    "server": os.uname().nodename if hasattr(os, "uname") else "unknown",
    "best_score": float(study.best_value),
    "best_params": study.best_params,
    "n_trials": len(study.trials),
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "trials": [
        {"number": t.number, "value": float(t.value), "params": t.params}
        for t in study.trials if t.value is not None
    ],
}

OUT_DIR = os.environ.get("OUT_DIR", "/hpc2hdd/home/aimslab/optuna_sweeps/server2-A40-catboost")
os.makedirs(OUT_DIR, exist_ok=True)
out_path = f"{OUT_DIR}/results_{time.strftime('%Y%m%d_%H%M%S')}.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to: {out_path}", flush=True)

best_path = f"{OUT_DIR}/best_params.json"
with open(best_path, "w") as f:
    json.dump(results["best_params"], f, indent=2)

# Save study
study_path = f"{OUT_DIR}/optuna_study.pkl"
import pickle
with open(study_path, "wb") as f:
    pickle.dump(study, f)

print("=== CatBoost Sweep COMPLETE ===", flush=True)
'''


def deploy_script(login_ssh: paramiko.SSHClient, server: dict, script_content: str,
                  script_name: str, out_dir: str) -> bool:
    """SFTP a script to the login node, then SCP it to the inner server."""
    name = server["name"]
    inner_host = server["ssh_host"]
    inner_port = server["ssh_port"]
    inner_user = server["ssh_user"]

    # 1. Write script to login node's /tmp
    sftp = login_ssh.open_sftp()
    tmp_path = f"/tmp/{script_name}"
    try:
        with sftp.open(tmp_path, "w") as f:
            f.write(script_content)
    finally:
        sftp.close()
    print(f"  [{name}] Script written to login node: {tmp_path}")

    # 2. SCP from login node to inner server
    scp_cmd = f"scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P {inner_port} {tmp_path} {inner_user}@{inner_host}:/tmp/{script_name}"
    exit_code, out, err = exec_remote(login_ssh, scp_cmd, timeout=60)
    if exit_code != 0:
        print(f"  [{name}] SCP FAILED: {err}")
        return False
    print(f"  [{name}] Script SCP'd to inner server: /tmp/{script_name}")

    # 3. Create output directory on inner server
    mkdir_cmd = f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {inner_port} {inner_user}@{inner_host} 'mkdir -p {out_dir}'"
    exec_remote(login_ssh, mkdir_cmd, timeout=30)
    print(f"  [{name}] Output directory created: {out_dir}")

    return True


def launch_job(login_ssh: paramiko.SSHClient, server: dict, script_name: str,
               out_dir: str, data_dir: str) -> bool:
    """Launch the Optuna sweep as a nohup background job on the inner server."""
    name = server["name"]
    inner_host = server["ssh_host"]
    inner_port = server["ssh_port"]
    inner_user = server["ssh_user"]
    log_file = f"{out_dir}/sweep_output.log"

    launch_cmd = (
        f"DATA_DIR={data_dir} OUT_DIR={out_dir} "
        f"nohup python3 /tmp/{script_name} > {log_file} 2>&1 &"
    )

    ssh_cmd = (
        f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
        f"-p {inner_port} {inner_user}@{inner_host} '{launch_cmd}'"
    )
    exit_code, out, err = exec_remote(login_ssh, ssh_cmd, timeout=30)
    if exit_code != 0:
        print(f"  [{name}] Launch FAILED: {err}")
        return False

    # Give it a moment then verify
    time.sleep(3)
    verify_cmd = (
        f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
        f"-p {inner_port} {inner_user}@{inner_host} "
        f"'ps aux | grep {script_name} | grep -v grep'"
    )
    _, out, _ = exec_remote(login_ssh, verify_cmd, timeout=30)
    if out.strip():
        print(f"  [{name}] JOB STARTED - running processes:\n{out.strip()}")
        return True
    else:
        # Check if it started and immediately failed
        _, log_out, _ = exec_on_inner(login_ssh, inner_host, inner_port, inner_user,
                                       f"tail -20 {log_file} 2>/dev/null || echo 'no log yet'")
        print(f"  [{name}] JOB may have failed. Log tail:\n{log_out[:500]}")
        return False


def main():
    print("=" * 60)
    print("  DEPLOYING OPTUNA SWEEPS TO DUAL GPU SERVERS")
    print(f"  Time: {datetime.now().isoformat()}")
    print("=" * 60)

    password = get_password()
    print("Password retrieved from DPAPI.")

    login_ssh = ssh_to_login(password)
    print(f"Connected to login node {LOGIN_HOST}:{LOGIN_PORT}")

    try:
        # ── Step 1: Check environments ──
        print("\n" + "=" * 60)
        print("  STEP 1: CHECKING ENVIRONMENTS")
        print("=" * 60)

        env1 = check_environment(login_ssh, SERVER1)
        env2 = check_environment(login_ssh, SERVER2)

        all_env = {"server1": env1, "server2": env2}
        env_path = SCRIPTS_DIR / "env_check_results.json"
        env_path.write_text(json.dumps(all_env, indent=2, default=str))
        print(f"\nEnvironment check results saved to: {env_path}")

        # ── Step 2: Deploy scripts ──
        print("\n" + "=" * 60)
        print("  STEP 2: DEPLOYING SWEEP SCRIPTS")
        print("=" * 60)

        out_dir1 = "/hpc2hdd/home/aimslab/optuna_sweeps/server1-A800-lightgbm"
        ok1 = deploy_script(login_ssh, SERVER1, LIGHTGBM_SWEEP_SCRIPT,
                            "lgbm_optuna_sweep.py", out_dir1)

        out_dir2 = "/hpc2hdd/home/aimslab/optuna_sweeps/server2-A40-catboost"
        ok2 = deploy_script(login_ssh, SERVER2, CATBOOST_SWEEP_SCRIPT,
                            "catboost_optuna_sweep.py", out_dir2)

        if not ok1:
            print("WARNING: Server1 deployment had issues")
        if not ok2:
            print("WARNING: Server2 deployment had issues")

        # ── Step 3: Launch jobs ──
        print("\n" + "=" * 60)
        print("  STEP 3: LAUNCHING BACKGROUND JOBS")
        print("=" * 60)

        launched1 = launch_job(login_ssh, SERVER1, "lgbm_optuna_sweep.py",
                               out_dir1, DATA_DIR)

        launched2 = launch_job(login_ssh, SERVER2, "catboost_optuna_sweep.py",
                               out_dir2, DATA_DIR)

        # ── Final report ──
        print("\n" + "=" * 60)
        print("  DEPLOYMENT REPORT")
        print("=" * 60)
        print(f"  Server1 (A800 80GB): LightGBM 50-trial 5-fold - {'RUNNING' if launched1 else 'FAILED'}")
        print(f"    Output: {out_dir1}/")
        print(f"    Log: {out_dir1}/sweep_output.log")
        print(f"  Server2 (A40 48GB): CatBoost 50-trial 5-fold - {'RUNNING' if launched2 else 'FAILED'}")
        print(f"    Output: {out_dir2}/")
        print(f"    Log: {out_dir2}/sweep_output.log")
        print(f"  Monitoring script: D:/桌面/codex/科研港科技/scripts/gpu_dual_monitor.py")
        print("=" * 60)

        # Save deployment manifest
        manifest = {
            "deployed_at": datetime.now().isoformat(),
            "server1": {
                "name": SERVER1["name"],
                "model": "LightGBM",
                "trials": 50,
                "folds": 5,
                "output_dir": out_dir1,
                "launched": launched1,
            },
            "server2": {
                "name": SERVER2["name"],
                "model": "CatBoost",
                "trials": 50,
                "folds": 5,
                "output_dir": out_dir2,
                "launched": launched2,
            },
        }
        manifest_path = SCRIPTS_DIR / "optuna_deployment_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"\nDeployment manifest saved to: {manifest_path}")

    finally:
        login_ssh.close()
        print("\nSSH connection closed.")


if __name__ == "__main__":
    main()
