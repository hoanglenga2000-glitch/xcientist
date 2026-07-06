"""
Robust Self-Evolving Loop v3: fast mode, per-task error isolation, auto-continue.
"""
import json, subprocess, sys, time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
PYTHON = r"C:\codex-python\python.exe"

TASKS = [
    {"id":"house_prices","bronze":0.140,"metric":"rmsle","dir":"minimize","budget":3},
    {"id":"telco_churn","bronze":0.800,"metric":"accuracy","dir":"maximize","budget":3},
    {"id":"titanic","bronze":0.794,"metric":"accuracy","dir":"maximize","budget":5},
    {"id":"bike_sharing_demand","bronze":0.480,"metric":"rmsle","dir":"minimize","budget":3},
    {"id":"spaceship_titanic","bronze":0.795,"metric":"accuracy","dir":"maximize","budget":3},
    {"id":"store_sales_time_series_forecasting","bronze":0.500,"metric":"rmsle","dir":"minimize","budget":3},
    {"id":"porto_seguro_safe_driver_prediction","bronze":0.285,"metric":"normalized_gini","dir":"maximize","budget":3},
    {"id":"tabular_playground_series_aug_2022","bronze":0.842,"metric":"roc_auc","dir":"maximize","budget":3},
    {"id":"digit_recognizer","bronze":0.986,"metric":"accuracy","dir":"maximize","budget":2},
]

def run_one_ensemble(task_id, branch_type, seed, n_folds):
    """Run single fast ensemble, return score or None."""
    config = f"configs/{task_id}.yaml"
    run_id = f"robust_{task_id}_{branch_type}_{int(time.time())}"
    cmd = [PYTHON, str(ROOT/"scripts"/"run_local_sklearn_ensemble.py"),
           "--config", str(ROOT/config), "--output-base", "experiments",
           "--task-id", task_id, "--run-id", run_id,
           "--n-folds", str(n_folds), "--branch-type", branch_type,
           "--random-state", str(seed), "--fast", "--sample-rows", "5000"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(ROOT))
        if r.returncode != 0: return None
        # Parse score from stdout
        for line in r.stdout.split('\n'):
            if '"best_validation_score"' in line:
                try:
                    d = json.loads(line)
                    return d.get("best_validation_score") or d.get("ensemble",{}).get("best_validation_score")
                except: pass
        # Fallback: read metrics from experiment dir
        exp_dir = ROOT / "experiments" / task_id
        dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()], reverse=True)
        for d in dirs[:5]:
            mp = d / "metrics.json"
            if mp.exists():
                try:
                    m = json.loads(mp.read_text())
                    s = m.get("ensemble",{}).get("best_validation_score")
                    if s: return float(s)
                except: pass
        return None
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None

def main():
    print("=" * 60)
    print("ROBUST EVOLUTION v3: Fast Mode, Error-Isolated")
    print(f"Tasks: {len(TASKS)} | Mode: fast (5000 samples)")
    print("=" * 60)

    state = {"started": datetime.now().isoformat(), "results": {}}
    MAX_ROUNDS = 10

    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n### ROUND {round_num}/{MAX_ROUNDS} ###")
        round_improved = 0

        for task in TASKS:
            tid = task["id"]
            if tid not in state["results"]:
                state["results"][tid] = {"best_score": None, "bronze_reached": False, "rounds": 0, "scores": []}

            ts = state["results"][tid]
            if ts["bronze_reached"]:
                continue

            # Try 3 different strategies per round
            strategies = [
                ("baseline", SEEDS[round_num % len(SEEDS)], 5),
                ("feature_engineering", SEEDS[(round_num+1) % len(SEEDS)], 5),
                ("ensemble_blend", SEEDS[(round_num+2) % len(SEEDS)], 10),
            ]

            for branch, seed, folds in strategies:
                score = run_one_ensemble(tid, branch, seed, folds)
                if score is None:
                    continue

                ts["rounds"] += 1
                ts["scores"].append(score)
                prev = ts["best_score"]
                is_better = (prev is None) or (task["dir"]=="maximize" and score > prev) or (task["dir"]=="minimize" and score < prev)

                if is_better:
                    ts["best_score"] = score
                    is_bronze = (task["dir"]=="maximize" and score >= task["bronze"]) or (task["dir"]=="minimize" and score <= task["bronze"])
                    if is_bronze:
                        ts["bronze_reached"] = True
                    delta = f"{score-prev:+.5f}" if prev else "new"
                    icon = "🏅" if is_bronze else "↑"
                    print(f"  {tid}: {score:.5f} {icon} ({delta}) [{branch}]")
                    round_improved += 1

        # Count bronze
        bronze = sum(1 for v in state["results"].values() if v["bronze_reached"])
        print(f"  Bronze: {bronze}/{len(TASKS)} | Improved: {round_improved}")

        # Save state
        state["last_updated"] = datetime.now().isoformat()
        state["bronze_count"] = bronze
        state["rounds"] = round_num
        sp = ROOT / "workspace" / f"robust_evolution_state_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        sp.write_text(json.dumps(state, indent=2))

        if bronze >= len(TASKS):
            print(f"\nALL BRONZE!")
            break

    print(f"\nFINAL: {state['bronze_count']}/{len(TASKS)} bronze")

SEEDS = [42, 3407, 12345, 777, 2048, 9999, 5555, 1111, 666, 8888]
if __name__ == "__main__":
    main()
