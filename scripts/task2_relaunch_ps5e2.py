"""Task 2: Relaunch ps5e2 on 87318 with log1p enabled.

Strategy: Change ps5e2 metric from "rmse" to "rmsle" in COMPETITIONS dict.
This causes:
1. log1p target transform during training (line 334-336: np.log1p on target)
2. expm1 on predictions for Kaggle submission (line 370-371)
3. OOF evaluation uses RMSLE formula (line 259: rmsle calc)
Net result: Model trains on log-scale prices, predictions are raw scale for Kaggle.

The log1p logic (confirmed from grep):
  line 279: log1p_target = (metric == "rmsle") and not no_log1p
  line 334-336: if log1p_target: y_tr = np.log1p(np.maximum(y_tr, 0))
  line 365-366: if log1p_target: val_pred = np.expm1(val_pred)
  line 370-371: if log1p_target: test_p = np.expm1(test_p)
"""
import sys
sys.path.insert(0, r"D:\桌面\codex\科研港科技\scripts")
from hpc_connect_v2 import exec_cmd, exec_async
import time

jid = "87318"
script_path = "/hpc2hdd/home/aimslab/gpu_train_v3.py"

# Step 1: Check if anything is running
print("=" * 60)
print("Step 1: Check if ps5e2 or anything is running on 87318")
print("=" * 60)
out, err, code = exec_cmd(jid, "ps aux | grep 'gpu_train_v3' | grep -v grep")
print(f"Running processes:\n{out if out else '(none)'}")

# Step 2: Change ps5e2 metric from "rmse" to "rmsle"
print("\n" + "=" * 60)
print("Step 2: Modify ps5e2 metric from rmse to rmsle")
print("=" * 60)

# First verify the line exists
out, err, code = exec_cmd(jid,
    "grep -n 'ps5e2.*Price.*rmse' /hpc2hdd/home/aimslab/gpu_train_v3.py")
print(f"Before: {out.strip()}")

# Use sed to change the metric in-place
sed_cmd = (
    "sed -i 's/\"ps5e2\":  (\"playground-series-s5e2\", \"Price\", \"regression\", \"rmse\"/"
    "\"ps5e2\":  (\"playground-series-s5e2\", \"Price\", \"regression\", \"rmsle\"/' "
    + script_path
)
out, err, code = exec_cmd(jid, sed_cmd)
print(f"sed result: code={code}, err={err[:200] if err else 'none'}")

# Verify the change
out, err, code = exec_cmd(jid,
    "grep -n 'ps5e2.*Price' /hpc2hdd/home/aimslab/gpu_train_v3.py")
print(f"After: {out.strip()}")

# Step 3: Launch ps5e2
print("\n" + "=" * 60)
print("Step 3: Launch ps5e2 training with log1p (via rmsle metric)")
print("=" * 60)

launch_cmd = (
    f"cd /hpc2hdd/home/aimslab && "
    f"nohup /usr/bin/python3 {script_path} ps5e2 "
    f"--depth=6 --l2=5 --lr=0.03 --iter=1000 --n-folds=5 "
    f"> v3_log_ps5e2_v2.txt 2>&1 &"
)
print(f"Launch command: {launch_cmd}")

out, err, code = exec_cmd(jid, launch_cmd)
print(f"Launch result: code={code}, err={err[:200] if err else 'none'}")

# Step 4: Verify launch
time.sleep(3)
print("\n" + "=" * 60)
print("Step 4: Verify launch")
print("=" * 60)
out, err, code = exec_cmd(jid, "ps aux | grep 'gpu_train_v3.*ps5e2' | grep -v grep")
print(f"ps5e2 process:\n{out if out else '(not found - may need more time)'}")

out, err, code = exec_cmd(jid, "tail -10 /hpc2hdd/home/aimslab/v3_log_ps5e2_v2.txt 2>/dev/null")
print(f"Log tail:\n{out[:500] if out else '(empty or not yet created)'}")

print("\nDONE. ps5e2 relauched with rmsle metric (log1p enabled).")
print("After training completes, predictions will be expm1'd back to raw scale.")
print("The log1p transform should bring RMSE much closer to the bronze gate of 0.8.")
