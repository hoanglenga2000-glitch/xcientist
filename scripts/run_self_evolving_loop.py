"""
Self-Evolving Training Loop: Autonomous MCGS search with cross-task memory.
Continuously runs until bronze on all tasks or budget exhausted.
MLEvolve-style: 500-step search, 12h budget → 65.3% medal rate.

Architecture:
  while not (all_bronze or time_up):
    for each task:
      1. Load cross-task memory from previous tasks
      2. MCGS search with UCT-guided branch exploration
      3. Score promotion gate → update best-so-far
      4. Claim audit → prevent drift
      5. Memory persist → accumulate for next tasks
      6. If bronze reached → auto Kaggle submit (if quota available)
      7. Rotate to next task (carrying over learned patterns)

Resource needs: CPU (no GPU required for tabular ensemble),
Kaggle API quota (5/day/competition), persistent disk for memory.
"""
import json, subprocess, sys, time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
PYTHON = r"C:\codex-python\python.exe"

# All competitions with verified bronze thresholds (Agent 2 corrected)
TASKS = [
    {"id": "house_prices", "bronze": 0.140, "metric": "rmsle", "dir": "minimize", "budget": 5},
    {"id": "telco_churn", "bronze": 0.800, "metric": "accuracy", "dir": "maximize", "budget": 5},
    {"id": "titanic", "bronze": 0.794, "metric": "accuracy", "dir": "maximize", "budget": 8},
    {"id": "bike_sharing_demand", "bronze": 0.480, "metric": "rmsle", "dir": "minimize", "budget": 5},
    {"id": "spaceship_titanic", "bronze": 0.795, "metric": "accuracy", "dir": "maximize", "budget": 3},
    {"id": "store_sales_time_series_forecasting", "bronze": 0.500, "metric": "rmsle", "dir": "minimize", "budget": 5},
    {"id": "porto_seguro_safe_driver_prediction", "bronze": 0.285, "metric": "normalized_gini", "dir": "maximize", "budget": 5},
    {"id": "tabular_playground_series_aug_2022", "bronze": 0.842, "metric": "roc_auc", "dir": "maximize", "budget": 5},
    {"id": "digit_recognizer", "bronze": 0.986, "metric": "accuracy", "dir": "maximize", "budget": 2},
]

def is_bronze(score, threshold, direction):
    if score is None: return False
    if direction == "maximize": return score >= threshold
    return score <= threshold

def run_mcgs_cycle(task, round_num):
    """Run one MCGS evolution cycle on a task."""
    task_id = task["id"]
    config = f"configs/{task_id}.yaml"
    budget = task["budget"]

    print(f"\n  [{task_id}] Round {round_num} (budget={budget})...", end=" ", flush=True)

    cmd = [PYTHON, str(ROOT / "scripts" / "run_workstation_mcgs.py"),
           "--config", str(ROOT / config),
           "--output-base", "experiments",
           "--task-id", task_id,
           "--budget-nodes", str(budget)]

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, cwd=str(ROOT))
    if r.returncode != 0:
        return None, None

    # Parse best score
    best_line = [l for l in r.stdout.split("\n") if "Best=" in l]
    if best_line:
        parts = best_line[0].split()
        try:
            score = float(parts[1])
            return score, r.stdout
        except:
            pass
    return None, None

def try_kaggle_submit(task_id, score, direction, bronze):
    """Submit to Kaggle if score improved."""
    # Map task_id to kaggle competition slug
    slug_map = {
        "titanic": "titanic",
        "house_prices": "house-prices-advanced-regression-techniques",
        "spaceship_titanic": "spaceship-titanic",
        "bike_sharing_demand": "bike-sharing-demand",
        "digit_recognizer": "digit-recognizer",
        "store_sales_time_series_forecasting": "store-sales-time-series-forecasting",
        "porto_seguro_safe_driver_prediction": "porto-seguro-safe-driver-prediction",
        "tabular_playground_series_aug_2022": "tabular-playground-series-aug-2022",
        "telco_churn": None,  # Not on Kaggle
    }
    slug = slug_map.get(task_id)
    if not slug:
        return False

    # Find latest submission
    exp_dir = ROOT / "experiments" / task_id
    dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("mcgs_")], reverse=True)
    if not dirs:
        return False

    for d in dirs[:2]:
        sp = d / "submission.csv"
        if not sp.exists():
            continue
        msg = f"MCGS-evolve: {task_id} score={score:.5f} bronze<={bronze}"
        r = subprocess.run(["python", "-m", "kaggle", "competitions", "submit",
                           "-c", slug, "-f", str(sp), "-m", msg],
                          capture_output=True, text=True, timeout=30)
        if "Success" in r.stdout:
            return True
        elif "Bad Request" in str(r.stdout):
            return False  # Rate limited
    return False

def main():
    print("=" * 60)
    print("SELF-EVOLVING LOOP: Autonomous Bronze Pursuit")
    print(f"Tasks: {len(TASKS)} | Started: {datetime.now().isoformat()}")
    print("=" * 60)

    state = {
        "started_at": datetime.now().isoformat(),
        "tasks": {},
        "rounds_completed": 0,
        "bronze_count": 0,
        "total_mcgs_steps": 0,
    }

    for t in TASKS:
        state["tasks"][t["id"]] = {
            "best_score": None, "best_round": 0, "bronze_reached": False,
            "kaggle_submitted": False, "rounds": 0, "bronze_threshold": t["bronze"],
            "direction": t["dir"],
        }

    MAX_ROUNDS = 10  # Total evolution rounds across all tasks
    state_path = ROOT / "workspace" / f"self_evolving_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n{'#'*60}")
        print(f"ROUND {round_num}/{MAX_ROUNDS}")
        print(f"{'#'*60}")

        round_improvements = 0
        for task in TASKS:
            tid = task["id"]
            ts = state["tasks"][tid]

            # Skip if already bronze
            if ts["bronze_reached"]:
                continue

            t0 = time.time()
            score, output = run_mcgs_cycle(task, round_num)
            elapsed = time.time() - t0

            ts["rounds"] += 1
            state["total_mcgs_steps"] += task["budget"]

            if score is not None:
                prev = ts["best_score"]
                better = (prev is None) or \
                         (task["dir"] == "maximize" and score > prev) or \
                         (task["dir"] == "minimize" and score < prev)

                if better:
                    ts["best_score"] = score
                    ts["best_round"] = round_num
                    delta = score - prev if prev else 0
                    print(f"  {tid}: {score:.5f} {'↑' if task['dir']=='maximize' else '↓'} "
                          f"({delta:+.5f}) {elapsed:.0f}s")
                    round_improvements += 1
                else:
                    print(f"  {tid}: {score:.5f} (no change) {elapsed:.0f}s")

                # Check bronze
                if is_bronze(score, task["bronze"], task["dir"]):
                    ts["bronze_reached"] = True
                    state["bronze_count"] += 1
                    print(f"    🏅 BRONZE REACHED! {score:.5f} >= {task['bronze']}")

                    # Auto-submit to Kaggle
                    if not ts["kaggle_submitted"]:
                        submitted = try_kaggle_submit(tid, score, task["dir"], task["bronze"])
                        ts["kaggle_submitted"] = submitted
                        if submitted:
                            print(f"    ✅ Kaggle submitted")
                        else:
                            print(f"    ⚠️ Kaggle submit skipped (rate limited / no slug)")
            else:
                print(f"  {tid}: FAILED {elapsed:.0f}s")

        state["rounds_completed"] = round_num

        # Persist state
        state["last_updated"] = datetime.now().isoformat()
        state_path.write_text(json.dumps(state, indent=2))

        # Check termination
        if state["bronze_count"] >= len(TASKS):
            print(f"\n🏆 ALL {len(TASKS)} COMPETITIONS REACHED BRONZE!")
            break

        if round_improvements == 0:
            print(f"\n  No improvements this round. Memory should help next round.")
        else:
            print(f"\n  {round_improvements} tasks improved this round.")

    # Final report
    print(f"\n{'='*60}")
    print(f"EVOLUTION COMPLETE: {state['bronze_count']}/{len(TASKS)} bronze")
    print(f"Rounds: {state['rounds_completed']} | Total MCGS steps: {state['total_mcgs_steps']}")
    print(f"State: {state_path}")

    for tid, ts in state["tasks"].items():
        medal = "🏅" if ts["bronze_reached"] else "  "
        score = f"{ts['best_score']:.5f}" if ts['best_score'] else "no_data"
        print(f"  {medal} {tid}: {score} (rounds={ts['rounds']})")

if __name__ == "__main__":
    main()
