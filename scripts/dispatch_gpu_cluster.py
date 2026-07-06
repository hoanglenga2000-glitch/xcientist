"""
GPU Cluster Dispatcher: trains across all 6 GPU servers in parallel.
Each server runs a different task on its GPUs.
"""
import paramiko, socket, socks, json, time, subprocess, sys, base64
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SERVERS = [
    {"id":"S1","user":"aimslab-IwkteXqP","pw_env":"HPC_S1_PASSWORD","gpus":1,"cores":64,
     "task":"house_prices","config":"configs/house_prices.yaml","model":"cb"},
    {"id":"S2","user":"aimslab-fpgTDTSi","pw_env":"HPC_S2_PASSWORD","gpus":4,"cores":64,
     "task":"spaceship_titanic","config":"configs/spaceship_titanic.yaml","model":"cb_multi"},
    {"id":"S3","user":"aimslab-lyudongxin","pw_env":"HPC_S3_PASSWORD","gpus":1,"cores":64,
     "task":"telco_churn","config":"configs/telco_churn.yaml","model":"cb"},
    {"id":"S4","user":"aimslab-xqJnhHBJ","pw_env":"HPC_S4_PASSWORD","gpus":1,"cores":64,
     "task":"titanic","config":"configs/titanic.yaml","model":"cb"},
    {"id":"S5","user":"aimslab-kdd-ai4s","pw_env":"HPC_S5_PASSWORD","gpus":1,"cores":64,
     "task":"bike_sharing_demand","config":"configs/bike_sharing_demand.yaml","model":"cb"},
    {"id":"S6","user":"aimslab-TTA-A800-1GPU","pw_env":"HPC_S6_PASSWORD","gpus":1,"cores":128,
     "task":"tabular_playground_series_aug_2022","config":"configs/tabular_playground_series_aug_2022.yaml","model":"cb"},
]

# GPU training script (lightweight, uses shared filesystem data)
GPU_SCRIPT = '''
import pandas as pd, numpy as np, json, sys, os, yaml
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_log_error
from datetime import datetime

task_id = sys.argv[1]
config_path = sys.argv[2]
gpu_count = int(sys.argv[3])
model_type = sys.argv[4] if len(sys.argv) > 4 else "cb"

# Load config to find data paths
with open(config_path) as f: config = yaml.safe_load(f)
data_cfg = config.get("data", {})
task_cfg = config.get("task", {})
train_path = data_cfg.get("train", f"tasks/{task_id}/data/train.csv")
test_path = data_cfg.get("test", f"tasks/{task_id}/data/test.csv")
target = task_cfg.get("target", "class")
metric_name = task_cfg.get("metric", "accuracy")

# Try local path first, then shared filesystem
import pathlib
for p in [pathlib.Path(train_path), pathlib.Path(f"/hpc2hdd/home/aimslab/{task_id}/train.csv")]:
    if p.exists():
        train_path = str(p)
        test_path = str(p.parent / "test.csv")
        break

print(f"Task: {task_id} | Data: {train_path}")
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Basic preprocessing
train = train.fillna(-999)
test = test.fillna(-999)

# Label-encode categoricals
for c in train.columns:
    if train[c].dtype == 'object' and c != target:
        le = LabelEncoder()
        all_vals = list(train[c].astype(str).unique()) + list(test[c].astype(str).unique())
        le.fit(all_vals)
        train[c] = le.transform(train[c].astype(str))
        test[c] = le.transform(test[c].astype(str))

# Handle target
if target in train.columns:
    if metric_name in ("accuracy", "balanced_accuracy", "roc_auc"):
        y = train[target].astype(int).values
        is_clf = True
    else:
        y = train[target].astype(float).values
        is_clf = False
    X_cols = [c for c in train.columns if c != target and c not in ('id','Id','ID','PassengerId','ImageId')]
else:
    y = np.zeros(len(train))
    X_cols = list(train.columns)
    is_clf = True

X = train[X_cols].values.astype(np.float32)
X_test = test[[c for c in X_cols if c in test.columns]].values.astype(np.float32)

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

print(f"X: {X.shape}, X_test: {X_test.shape}")

# Train with multi-GPU if available
n_folds = 5 if len(y) > 500 else 10
if is_clf:
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
else:
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

oof = np.zeros(len(y))
test_preds = np.zeros(len(X_test))

for fold, (tr, va) in enumerate(cv.split(X, y)):
    fold_seed = 42 + fold
    if is_clf:
        model = CatBoostClassifier(
            iterations=1000, learning_rate=0.03, depth=7,
            task_type="GPU", devices=list(range(gpu_count)),
            random_seed=fold_seed, verbose=50 if fold==0 else 0
        )
        model.fit(X[tr], y[tr], verbose=50 if fold==0 else 0)
        oof[va] = model.predict_proba(X[va])[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / n_folds
    else:
        model = CatBoostRegressor(
            iterations=1000, learning_rate=0.03, depth=7,
            task_type="GPU", devices=list(range(gpu_count)),
            random_seed=fold_seed, verbose=50 if fold==0 else 0
        )
        model.fit(X[tr], y[tr], verbose=50 if fold==0 else 0)
        oof[va] = model.predict(X[va])
        test_preds += model.predict(X_test) / n_folds

    if is_clf:
        acc = accuracy_score(y[va], (oof[va] > 0.5).astype(int))
        print(f"Fold{fold+1}: acc={acc:.4f}")
    else:
        rmsle = np.sqrt(mean_squared_log_error(np.abs(y[va]), np.abs(oof[va])))
        print(f"Fold{fold+1}: rmsle={rmsle:.4f}")

# Final metrics
if is_clf:
    final_score = float(accuracy_score(y, (oof > 0.5).astype(int)))
else:
    final_score = float(np.sqrt(mean_squared_log_error(np.abs(y), np.abs(oof))))

result = {
    "task_id": task_id, "gpu_count": gpu_count, "n_folds": n_folds,
    "final_score": final_score, "metric": metric_name,
    "time": datetime.now().isoformat()
}

with open(f"/tmp/gpu_result_{task_id}.json", "w") as f:
    json.dump(result, f)
print(f"RESULT: {task_id} score={final_score:.5f}")
'''

def main():
    print("=== GPU CLUSTER DISPATCHER ===")
    print(f"Servers: {len(SERVERS)} | Total GPUs: {sum(s['gpus'] for s in SERVERS)}")

    # Upload script to each server and launch
    b64 = base64.b64encode(GPU_SCRIPT.encode()).decode()

    for srv in SERVERS:
        try:
            password = os.environ.get(srv["pw_env"]) or os.environ.get("GPU_SSH_PASSWORD")
            if not password:
                raise RuntimeError(f"Missing password env: {srv['pw_env']} or GPU_SSH_PASSWORD")
            # Connect
            sock = socks.socksocket()
            sock.set_proxy(socks.SOCKS5, '127.0.0.1', 7890)
            sock.settimeout(10)
            sock.connect(('100.85.169.63', 1235))
            t = paramiko.Transport(sock)
            t.connect(username=srv["user"], password=password)

            # Upload script
            chan = t.open_session()
            chan.exec_command(f'echo {b64} | base64 -d > /tmp/gpu_train_{srv["id"]}.py')
            time.sleep(2)

            # Launch training in background
            chan2 = t.open_session()
            task = srv["task"]
            config = f"/hpc2hdd/home/aimslab/{task}/../{srv['config'].split('/')[-1]}" if "/" not in srv['config'] else srv['config']
            chan2.exec_command(
                f'bash -c "python3 /tmp/gpu_train_{srv["id"]}.py {task} {config} {srv["gpus"]} '
                f'> /tmp/gpu_{srv["id"]}.log 2>&1 &" && echo LAUNCHED'
            )
            time.sleep(1)
            out = b''
            while chan2.recv_ready(): out += chan2.recv(4096)
            print(f'{srv["id"]}({srv["gpus"]}GPU): {out.decode().strip()} -> {task}')

            t.close()
        except Exception as e:
            print(f'{srv["id"]}: FAIL - {e}')

    print("\nAll jobs dispatched. Monitoring...")

    # Wait 60s then check results
    time.sleep(60)

    for srv in SERVERS:
        try:
            password = os.environ.get(srv["pw_env"]) or os.environ.get("GPU_SSH_PASSWORD")
            if not password:
                raise RuntimeError(f"Missing password env: {srv['pw_env']} or GPU_SSH_PASSWORD")
            sock = socks.socksocket()
            sock.set_proxy(socks.SOCKS5, '127.0.0.1', 7890)
            sock.settimeout(8)
            sock.connect(('100.85.169.63', 1235))
            t = paramiko.Transport(sock)
            t.connect(username=srv["user"], password=password)

            # Check log
            chan = t.open_session()
            chan.exec_command(f'tail -3 /tmp/gpu_{srv["id"]}.log 2>/dev/null')
            time.sleep(2)
            out = b''
            while chan.recv_ready(): out += chan.recv(4096)
            log = out.decode().strip()[:150]
            print(f'{srv["id"]}: {log if log else "running..."}')
            t.close()
        except:
            print(f'{srv["id"]}: check failed')

    print("\nDone. Results will be in /tmp/gpu_result_*.json")

if __name__ == "__main__":
    main()
