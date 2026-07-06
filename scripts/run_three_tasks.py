"""Execute all three parallel tasks across the HPC GPU cluster."""
import sys, json
sys.path.insert(0, r"D:\桌面\codex\科研港科技\scripts")
from hpc_connect_v2 import exec_cmd, exec_async, INSTANCES
import time

results = {}

# ═══════════════════════════════════════════════════════════════════════
# TASK 1: Check spaceship_titanic Kaggle score via 87384
# ═══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TASK 1: spaceship-titanic Kaggle score on 87384")
print("=" * 60)
try:
    out, err, code = exec_cmd("87384",
        "kaggle competitions submissions -c spaceship-titanic 2>/dev/null | head -10")
    results["task1_kaggle_score"] = {"out": out, "err": err, "code": code}
    print(f"OUTPUT:\n{out}")
    if err:
        print(f"STDERR: {err}")
except Exception as e:
    results["task1_kaggle_score"] = {"error": str(e)}
    print(f"ERROR: {e}")

# ═══════════════════════════════════════════════════════════════════════
# TASK 2a: Check log1p logic in gpu_train_v3.py on 87318
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 2a: Check gpu_train_v3.py log1p logic on 87318")
print("=" * 60)
try:
    out, err, code = exec_cmd("87318",
        "grep -n 'log1p\\|log_scale\\|LOG_TASKS\\|expm1\\|no.log1p' /hpc2hdd/home/aimslab/gpu_train_v3.py | head -30")
    results["task2_log1p_logic"] = {"out": out, "err": err, "code": code}
    print(f"OUTPUT:\n{out}")
    if err:
        print(f"STDERR: {err}")
except Exception as e:
    results["task2_log1p_logic"] = {"error": str(e)}
    print(f"ERROR: {e}")

# Also check the competition registry for ps5e2 entry
print("\n--- ps5e2 entry in registry ---")
try:
    out, err, code = exec_cmd("87318",
        "grep -n 'ps5e2\\|ps5e1\\|ps5e3' /hpc2hdd/home/aimslab/gpu_train_v3.py | head -10")
    results["task2_ps5e2_registry"] = {"out": out, "err": err, "code": code}
    print(f"OUTPUT:\n{out}")
    if err:
        print(f"STDERR: {err}")
except Exception as e:
    results["task2_ps5e2_registry"] = {"error": str(e)}
    print(f"ERROR: {e}")

# Check also the full competition task list
print("\n--- Competition task dict ---")
try:
    out, err, code = exec_cmd("87318",
        "grep -A100 'COMPETITIONS\\s*=\\s*{' /hpc2hdd/home/aimslab/gpu_train_v3.py | head -60")
    results["task2_competitions"] = {"out": out, "err": err, "code": code}
    print(f"OUTPUT:\n{out}")
    if err:
        print(f"STDERR: {err}")
except Exception as e:
    results["task2_competitions"] = {"error": str(e)}
    print(f"ERROR: {e}")

# ═══════════════════════════════════════════════════════════════════════
# TASK 3: Check progress of 5 running tasks
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 3: Check progress of running tasks")
print("=" * 60)

# 3a: 87617 - ps6e6 and ps4e3
print("\n--- [87617] ps6e6 tail ---")
try:
    out, err, code = exec_cmd("87617",
        "tail -20 /hpc2hdd/home/aimslab/v3_log_ps6e6.txt 2>/dev/null")
    results["task3_87617_ps6e6"] = {"out": out, "err": err, "code": code}
    print(f"OUTPUT:\n{out}")
except Exception as e:
    results["task3_87617_ps6e6"] = {"error": str(e)}
    print(f"ERROR: {e}")

print("\n--- [87617] ps4e3 tail ---")
try:
    out, err, code = exec_cmd("87617",
        "tail -20 /hpc2hdd/home/aimslab/v3_log_ps4e3.txt 2>/dev/null")
    results["task3_87617_ps4e3"] = {"out": out, "err": err, "code": code}
    print(f"OUTPUT:\n{out}")
except Exception as e:
    results["task3_87617_ps4e3"] = {"error": str(e)}
    print(f"ERROR: {e}")

# 3b: 87384 - ps4e7 and ps3e25
print("\n--- [87384] ps4e7 tail ---")
try:
    out, err, code = exec_cmd("87384",
        "tail -20 /hpc2hdd/home/aimslab/v3_log_ps4e7.txt 2>/dev/null")
    results["task3_87384_ps4e7"] = {"out": out, "err": err, "code": code}
    print(f"OUTPUT:\n{out}")
except Exception as e:
    results["task3_87384_ps4e7"] = {"error": str(e)}
    print(f"ERROR: {e}")

print("\n--- [87384] ps3e25 tail ---")
try:
    out, err, code = exec_cmd("87384",
        "tail -20 /hpc2hdd/home/aimslab/v3_log_ps3e25.txt 2>/dev/null")
    results["task3_87384_ps3e25"] = {"out": out, "err": err, "code": code}
    print(f"OUTPUT:\n{out}")
except Exception as e:
    results["task3_87384_ps3e25"] = {"error": str(e)}
    print(f"ERROR: {e}")

# 3c: 87519 - tps_mar2022
print("\n--- [87519] tps_mar2022 tail ---")
try:
    out, err, code = exec_cmd("87519",
        "tail -20 /hpc2hdd/home/aimslab/v3_log_tps_mar2022.txt 2>/dev/null")
    results["task3_87519_tps"] = {"out": out, "err": err, "code": code}
    print(f"OUTPUT:\n{out}")
except Exception as e:
    results["task3_87519_tps"] = {"error": str(e)}
    print(f"ERROR: {e}")

# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("COMPLETE - All results stored in results dict")
print("=" * 60)

# Print structured summary
for k, v in results.items():
    out_str = v.get("out", "").strip()[:200]
    err_str = v.get("err", "").strip()[:100] if isinstance(v.get("err"), str) else ""
    code = v.get("code", "N/A")
    error = v.get("error", "")
    if error:
        print(f"  [{k}] ERROR: {error[:150]}")
    elif "No such file" in err_str or out_str == "":
        print(f"  [{k}] code={code} | out={out_str[:100]} | err={err_str}")
    else:
        print(f"  [{k}] code={code} | {out_str[:150]}")
