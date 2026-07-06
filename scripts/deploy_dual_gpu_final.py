#!/usr/bin/env python3
"""
FINAL DEPLOYMENT: Dual GPU Optuna Sweeps
- Server1 (A800 80GB): LightGBM DONE, CatBoost DONE
- Server2 (A40 48GB): CatBoost (if accessible) else XGBoost on A800
"""
import json, subprocess, sys, time, os
from datetime import datetime
import socks, paramiko, socket, struct

LOGIN_HOST = "100.85.169.63"
LOGIN_PORT = 1235
LOGIN_USER = "aimslab-IwkteXqP"
SOCKS5_HOST = "127.0.0.1"
SOCKS5_PORT = 7890
A40_HOST = "10.120.18.240"
A40_PORT = 6988

PYENV = "/hpc2hdd/home/aimslab/research_agent_workstation/pyenvs/s6e6_boosting/bin/python"
DATA_DIR = "/hpc2hdd/home/aimslab/spaceship_titanic"
WORK_DIR = "/hpc2hdd/home/aimslab/optuna_sweeps"

def get_password():
    ps_script = r'$cred = Import-Clixml -Path "$env:APPDATA\ResearchAgentWorkstation\hpc_ssh_credential.xml"; Write-Output $cred.GetNetworkCredential().Password'
    result = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script], capture_output=True, text=True, timeout=15)
    return result.stdout.strip()

def ssh_to_login():
    password = get_password()
    s = socks.socksocket()
    s.set_proxy(socks.SOCKS5, SOCKS5_HOST, SOCKS5_PORT)
    s.settimeout(30)
    s.connect((LOGIN_HOST, LOGIN_PORT))
    t = paramiko.Transport(s)
    t.connect(username=LOGIN_USER, password=password)
    ssh = paramiko.SSHClient()
    ssh._transport = t
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    return ssh, password

# ── Sweep script templates ──

CATBOOST_SCRIPT = '''#!/usr/bin/env python3
"""CatBoost 50-trial 5-fold Optuna sweep for Spaceship Titanic."""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from catboost import CatBoostClassifier

DATA_DIR = os.environ.get("DATA_DIR", "/hpc2hdd/home/aimslab/spaceship_titanic")
OUT_DIR = os.environ.get("OUT_DIR", "/tmp/optuna_out")

print(f"CatBoost Optuna Sweep | Start: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print(f"Hostname: {os.uname().nodename}", flush=True)

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
print(f"Data: train={train.shape}, test={test.shape}", flush=True)

for df in [train, test]:
    df["Cabin_deck"] = df["Cabin"].str.split("/").str[0].fillna("Unknown")
    df["Cabin_num"] = pd.to_numeric(df["Cabin"].str.split("/").str[1], errors="coerce").fillna(-1)
    df["Cabin_side"] = df["Cabin"].str.split("/").str[2].fillna("Unknown")
    df["Group"] = df["PassengerId"].str[:4]
    for c in ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]:
        df[c] = df[c].fillna(0)
    df["TotalSpend"] = sum(df[c] for c in ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"])
    df["HasSpend"] = (df["TotalSpend"] > 0).astype(int)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(27)
    for c in ["VIP","CryoSleep"]: df[c] = df[c].fillna(False)
    df["VIP_Age"] = df["VIP"].astype(int) * df["Age"]

target = "Transported"
y = (train[target] == True).astype(int).values
drop_cols = [target, "PassengerId", "Name", "Cabin"]
cat_cols = ["HomePlanet","CryoSleep","Destination","VIP","Cabin_deck","Cabin_side"]

X = train.drop(columns=[c for c in drop_cols if c in train.columns], errors="ignore")
Xt = test.drop(columns=[c for c in drop_cols if c in test.columns], errors="ignore")
for col in cat_cols:
    if col in X.columns:
        le = LabelEncoder()
        le.fit(pd.concat([X[col].astype(str), Xt[col].astype(str)]))
        X[col] = le.transform(X[col].astype(str))
common = [c for c in X.columns if c in Xt.columns]
X = X[common].fillna(-1).astype(float).values
X = StandardScaler().fit_transform(X)
print(f"Features: {X.shape}, target: {y.shape}", flush=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 500, 3000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 30.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255, step=32),
        "random_strength": trial.suggest_float("random_strength", 0.1, 5.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
        "random_seed": 42, "verbose": False, "thread_count": -1, "allow_writing_files": False,
    }
    scores = []
    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        m = CatBoostClassifier(**params)
        m.fit(X_tr, y_tr, verbose=False)
        pred = m.predict_proba(X_val)[:, 1]
        scores.append(accuracy_score(y_val, (pred > 0.5).astype(int)))
    return float(np.mean(scores))

study = optuna.create_study(direction="maximize", study_name="catboost_5fold_v2")
study.optimize(objective, n_trials=50, show_progress_bar=True, n_jobs=1)

os.makedirs(OUT_DIR, exist_ok=True)
results = {
    "model": "CatBoost",
    "hostname": os.uname().nodename,
    "best_score": float(study.best_value),
    "best_params": study.best_params,
    "n_trials": len(study.trials),
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "trials": [{"number": t.number, "value": float(t.value), "params": t.params} for t in study.trials if t.value is not None],
}

ts = time.strftime("%Y%m%d_%H%M%S")
with open(f"{OUT_DIR}/results_{ts}.json", "w") as f: json.dump(results, f, indent=2)
with open(f"{OUT_DIR}/best_params.json", "w") as f: json.dump(results["best_params"], f, indent=2)
import pickle
with open(f"{OUT_DIR}/study.pkl", "wb") as f: pickle.dump(study, f)
print(f"SAVED: {OUT_DIR}/results_{ts}.json", flush=True)
print(f"BEST: {study.best_value:.6f}", flush=True)
print(f"PARAMS: {json.dumps(study.best_params)}", flush=True)
print("DONE", flush=True)
'''

XGBOOST_SCRIPT = '''#!/usr/bin/env python3
"""XGBoost 50-trial 5-fold Optuna sweep for Spaceship Titanic."""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb

DATA_DIR = os.environ.get("DATA_DIR", "/hpc2hdd/home/aimslab/spaceship_titanic")
OUT_DIR = os.environ.get("OUT_DIR", "/tmp/optuna_out")

print(f"XGBoost Optuna Sweep | Start: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print(f"Hostname: {os.uname().nodename}", flush=True)

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
print(f"Data: train={train.shape}, test={test.shape}", flush=True)

for df in [train, test]:
    df["Cabin_deck"] = df["Cabin"].str.split("/").str[0].fillna("Unknown")
    df["Cabin_num"] = pd.to_numeric(df["Cabin"].str.split("/").str[1], errors="coerce").fillna(-1)
    df["Cabin_side"] = df["Cabin"].str.split("/").str[2].fillna("Unknown")
    df["Group"] = df["PassengerId"].str[:4]
    for c in ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]:
        df[c] = df[c].fillna(0)
    df["TotalSpend"] = sum(df[c] for c in ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"])
    df["HasSpend"] = (df["TotalSpend"] > 0).astype(int)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(27)
    for c in ["VIP","CryoSleep"]: df[c] = df[c].fillna(False)
    df["VIP_Age"] = df["VIP"].astype(int) * df["Age"]

target = "Transported"
y = (train[target] == True).astype(int).values
drop_cols = [target, "PassengerId", "Name", "Cabin"]
cat_cols = ["HomePlanet","CryoSleep","Destination","VIP","Cabin_deck","Cabin_side"]

X = train.drop(columns=[c for c in drop_cols if c in train.columns], errors="ignore")
Xt = test.drop(columns=[c for c in drop_cols if c in test.columns], errors="ignore")
for col in cat_cols:
    if col in X.columns:
        le = LabelEncoder()
        le.fit(pd.concat([X[col].astype(str), Xt[col].astype(str)]))
        X[col] = le.transform(X[col].astype(str))
common = [c for c in X.columns if c in Xt.columns]
X = X[common].fillna(-1).astype(float).values
X = StandardScaler().fit_transform(X)
print(f"Features: {X.shape}, target: {y.shape}", flush=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 3000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": 42, "verbosity": 0, "n_jobs": -1,
        "tree_method": "hist", "device": "cpu",
    }
    scores = []
    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        m = xgb.XGBClassifier(**params)
        m.fit(X_tr, y_tr, verbose=False)
        pred = m.predict_proba(X_val)[:, 1]
        scores.append(accuracy_score(y_val, (pred > 0.5).astype(int)))
    return float(np.mean(scores))

study = optuna.create_study(direction="maximize", study_name="xgboost_5fold")
study.optimize(objective, n_trials=50, show_progress_bar=True, n_jobs=1)

os.makedirs(OUT_DIR, exist_ok=True)
results = {
    "model": "XGBoost",
    "hostname": os.uname().nodename,
    "best_score": float(study.best_value),
    "best_params": study.best_params,
    "n_trials": len(study.trials),
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "trials": [{"number": t.number, "value": float(t.value), "params": t.params} for t in study.trials if t.value is not None],
}

ts = time.strftime("%Y%m%d_%H%M%S")
with open(f"{OUT_DIR}/results_{ts}.json", "w") as f: json.dump(results, f, indent=2)
with open(f"{OUT_DIR}/best_params.json", "w") as f: json.dump(results["best_params"], f, indent=2)
import pickle
with open(f"{OUT_DIR}/study.pkl", "wb") as f: pickle.dump(study, f)
print(f"SAVED: {OUT_DIR}/results_{ts}.json", flush=True)
print(f"BEST: {study.best_value:.6f}", flush=True)
print(f"PARAMS: {json.dumps(study.best_params)}", flush=True)
print("DONE", flush=True)
'''


def main():
    print("=" * 60)
    print("  FINAL DUAL GPU OPTUNA DEPLOYMENT")
    print(f"  {datetime.now().isoformat()}")
    print("=" * 60)

    ssh, password = ssh_to_login()
    print("Connected to A800 login node.")

    # ── Check existing sweeps ──
    print("\n--- Existing Sweeps on A800 ---")
    _, out, _ = ssh.exec_command("cat ~/spaceship_titanic/result_5fold.json 2>/dev/null; echo '---'; cat ~/spaceship_titanic/result_lgb.json 2>/dev/null")
    print(out.read().decode().strip())

    # ── Test A40 access ──
    print("\n--- Testing A40 Access ---")
    a40_script = '''import paramiko, sys
pw = open("/tmp/_fpw.txt").read().strip()
t = paramiko.Transport(("10.120.18.240", 6988))
t.banner_timeout = 10
t.auth_timeout = 15
t.start_client(timeout=10)
t.auth_password("aimslab-kdd-ai4s", pw)
print(f"AUTH: is_authenticated={t.is_authenticated()}", flush=True)
if t.is_authenticated():
    ch = t.open_session()
    ch.exec_command("hostname && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    print(f"OUT: {ch.recv(4096).decode()}", flush=True)
    ch.close()
else:
    print("A40 NOT FULLY AUTHENTICATED", flush=True)
t.close()
'''

    sftp = ssh.open_sftp()
    with sftp.open("/tmp/_fpw.txt", "w") as f: f.write(password)
    with sftp.open("/tmp/_test_a40_final.py", "w") as f: f.write(a40_script)
    sftp.close()

    _, out, _ = ssh.exec_command(f"{PYENV} /tmp/_test_a40_final.py 2>&1", timeout=30)
    a40_output = out.read().decode().strip()
    a40_accessible = "AUTH: is_authenticated=True" in a40_output
    print(a40_output)

    deployment_results = {}

    # ── Deploy based on A40 accessibility ──
    if a40_accessible:
        print("\n--- A40 IS ACCESSIBLE! Deploying CatBoost to A40 ---")
        # Deploy CatBoost to A40 via script on login node
        deploy_a40 = f'''import paramiko, sys, time
pw = open("/tmp/_fpw.txt").read().strip()
s = open("/tmp/_catboost_sweep.py").read()

t = paramiko.Transport(("10.120.18.240", 6988))
t.banner_timeout = 10; t.auth_timeout = 15
t.start_client(timeout=10)
t.auth_password("aimslab-kdd-ai4s", pw)
print(f"AUTH: is_authenticated={{t.is_authenticated()}}", flush=True)

if t.is_authenticated():
    ssh2 = paramiko.SSHClient()
    ssh2._transport = t
    ssh2.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Write script
    sftp2 = t.open_sftp_client()
    sftp2.putfo(__import__("io").BytesIO(s.encode()), "/tmp/sweep_cb.py")
    sftp2.close()

    # Launch
    out_dir = "{WORK_DIR}/server2-A40-catboost"
    chan = ssh2.get_transport().open_session()
    chan.exec_command(f"mkdir -p {{out_dir}}")
    chan.recv(1024)
    chan.close()

    chan2 = ssh2.get_transport().open_session()
    chan2.exec_command(f"DATA_DIR={DATA_DIR} OUT_DIR={{out_dir}} nohup python3 /tmp/sweep_cb.py > {{out_dir}}/sweep.log 2>&1 &")
    chan2.recv(1024)
    chan2.close()

    time.sleep(3)
    chan3 = ssh2.get_transport().open_session()
    chan3.exec_command("ps aux | grep sweep_cb | grep -v grep")
    out = chan3.recv(4096).decode()
    print(f"PROCESS: {{out}}", flush=True)
    chan3.close()
    ssh2.close()
else:
    print("A40 AUTH FAILED", flush=True)
'''
        # Write CatBoost script and deploy script
        sftp = ssh.open_sftp()
        with sftp.open("/tmp/_catboost_sweep.py", "w") as f: f.write(CATBOOST_SCRIPT)
        with sftp.open("/tmp/_deploy_a40.py", "w") as f: f.write(deploy_a40)
        sftp.close()

        _, out, _ = ssh.exec_command(f"{PYENV} /tmp/_deploy_a40.py 2>&1", timeout=60)
        print(out.read().decode())
        deployment_results["a40_catboost"] = "deployed" if "PROCESS:" in out.read().decode() else "failed"
    else:
        print("\n--- A40 NOT ACCESSIBLE. Deploying XGBoost to A800 instead ---")
        out_dir = f"{WORK_DIR}/server1-A800-xgboost"

        sftp = ssh.open_sftp()
        with sftp.open("/tmp/_xgb_sweep.py", "w") as f: f.write(XGBOOST_SCRIPT)
        sftp.close()

        # Launch on A800
        cmd = f"mkdir -p {out_dir} && DATA_DIR={DATA_DIR} OUT_DIR={out_dir} nohup {PYENV} /tmp/_xgb_sweep.py > {out_dir}/sweep.log 2>&1 &"
        ssh.exec_command(cmd, timeout=15)
        time.sleep(3)

        _, out, _ = ssh.exec_command("ps aux | grep _xgb_sweep | grep -v grep")
        xgb_proc = out.read().decode().strip()
        print(f"XGBoost process: {xgb_proc}")

        deployment_results["a800_xgboost"] = "running" if xgb_proc else "failed"
        deployment_results["a40_catboost"] = "skipped - A40 not accessible"

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  DEPLOYMENT SUMMARY")
    print("=" * 60)
    print(f"  Server1 (A800 80GB, hostname 60220a41a3ab):")
    print(f"    CatBoost 100-trial 5-fold: COMPLETED (best=0.81629)")
    print(f"    LightGBM 50-trial 5-fold:  COMPLETED (best=0.81203)")
    if "a800_xgboost" in deployment_results:
        print(f"    XGBoost 50-trial 5-fold:   {deployment_results['a800_xgboost'].upper()}")

    print(f"\n  Server2 (A40 48GB, hostname 0f48428831c3):")
    if a40_accessible:
        print(f"    CatBoost 50-trial 5-fold:  {deployment_results.get('a40_catboost', 'unknown').upper()}")
    else:
        print(f"    NOT ACCESSIBLE - aimslab-kdd-ai4s auth incomplete")
        print(f"    (Password accepted but SSH session setup fails)")
        print(f"    (Container may be stopped or needs key-based auth setup)")

    print(f"\n  Monitoring: D:/桌面/codex/科研港科技/scripts/gpu_dual_monitor.py")
    print(f"  Results dir (A800): ~/spaceship_titanic/ and {WORK_DIR}/")
    print("=" * 60)

    ssh.exec_command("rm -f /tmp/_fpw.txt /tmp/_test_a40_final.py /tmp/_catboost_sweep.py /tmp/_deploy_a40.py /tmp/_xgb_sweep.py", timeout=10)
    ssh.close()
    print("Done.")


if __name__ == "__main__":
    main()
