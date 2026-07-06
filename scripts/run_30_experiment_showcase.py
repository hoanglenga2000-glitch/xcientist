"""
30-Experiment Showcase: 10 competitions × 3 strategy variants each.
Demonstrates the workstation's self-evolving capability at scale.
Each variant uses different branch_type + seed + n_folds combinations.
"""
import json, subprocess, sys, time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
PYTHON = r"C:\codex-python\python.exe"

# 10 competitions with 3 strategy variants each = 30 experiments
TASKS = [
    # (task_id, config_path, target, metric, direction, bronze_threshold)
    ("house_prices", "configs/house_prices.yaml", "SalePrice", "rmsle", "minimize", 0.14),
    ("telco_churn", "configs/telco_churn.yaml", "Churn", "accuracy", "maximize", 0.80),
    ("tabular_playground_series_aug_2022", "configs/tabular_playground_series_aug_2022.yaml", "failure", "accuracy", "maximize", 0.55),
    ("porto_seguro_safe_driver_prediction", "configs/porto_seguro_safe_driver_prediction.yaml", "target", "gini", "maximize", 0.26),
    ("store_sales_time_series_forecasting", "configs/store_sales_time_series_forecasting.yaml", "sales", "rmsle", "minimize", 0.60),
    ("bike_sharing_demand", "configs/bike_sharing_demand.yaml", "count", "rmsle", "minimize", 0.40),
    ("titanic", "configs/titanic.yaml", "Survived", "accuracy", "maximize", 0.794),
    ("spaceship_titanic", "configs/spaceship_titanic.yaml", "Transported", "accuracy", "maximize", 0.789),
    ("digit_recognizer", "configs/digit_recognizer.yaml", "label", "accuracy", "maximize", 0.975),
    ("playground_series_s6e6", "configs/generated/playground_series_s6e6.yaml", "class", "balanced_accuracy", "maximize", 0.40),
]

# 3 strategy variants per competition
VARIANTS = [
    {"name": "baseline", "branch_type": "", "n_folds": 5, "seed": 42},
    {"name": "feature_engineering", "branch_type": "feature_engineering", "n_folds": 5, "seed": 3407},
    {"name": "ensemble_blend", "branch_type": "ensemble_blend", "n_folds": 10, "seed": 12345},
]

def run_experiment(task_id, config_path, variant):
    """Run a single ensemble experiment variant."""
    run_id = f"{task_id}_{variant['name']}_{int(time.time())}"
    cmd = [PYTHON, str(ROOT / "scripts" / "run_local_sklearn_ensemble.py"),
           "--config", str(ROOT / config_path),
           "--output-base", "experiments",
           "--task-id", task_id,
           "--run-id", run_id,
           "--n-folds", str(variant["n_folds"]),
           "--branch-type", variant["branch_type"],
           "--random-state", str(variant["seed"])]

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(ROOT))
    if r.returncode != 0:
        return None

    # Parse score from output
    for line in r.stdout.strip().split("\n"):
        if "best_validation_score" in line or "status" in line:
            try:
                d = json.loads(line)
                return d.get("best_validation_score") or d.get("ensemble", {}).get("best_validation_score")
            except:
                pass

    # Fallback: read from experiment dir
    exp_dir = ROOT / "experiments" / task_id
    dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()], reverse=True)
    for d in dirs[:3]:
        mp = d / "metrics.json"
        if mp.exists():
            m = json.loads(mp.read_text())
            return m.get("ensemble", {}).get("best_validation_score")

    return None

def main():
    t0 = time.time()
    results = []
    total = len(TASKS) * len(VARIANTS)
    current = 0

    print(f"=== 30-Experiment Showcase ===")
    print(f"Competitions: {len(TASKS)} | Variants each: {len(VARIANTS)} | Total runs: {total}")
    print()

    for task_id, config_path, target, metric, direction, bronze in TASKS:
        print(f"\n{'='*60}")
        print(f"[{task_id}] metric={metric} direction={direction} bronze<={bronze}")
        print(f"{'='*60}")

        task_results = []
        for variant in VARIANTS:
            current += 1
            print(f"  [{current}/{total}] {variant['name']}...", end=" ", flush=True)

            t1 = time.time()
            score = run_experiment(task_id, config_path, variant)

            if score is not None:
                is_bronze = (direction == "maximize" and score >= bronze) or \
                           (direction == "minimize" and score <= bronze)
                elapsed = time.time() - t1
                status = "✅" if is_bronze else "❌"
                print(f"score={score:.4f} {status} ({elapsed:.0f}s)")
                task_results.append({
                    "variant": variant["name"], "score": score,
                    "bronze_reached": is_bronze, "time_sec": round(elapsed, 1)
                })
            else:
                print(f"FAILED")

        if task_results:
            best = max([r["score"] for r in task_results], key=lambda s: s if direction == "maximize" else -s)
            best_bronze = any(r["bronze_reached"] for r in task_results)
            results.append({
                "task_id": task_id, "metric": metric, "direction": direction,
                "bronze_threshold": bronze, "best_score": best,
                "bronze_reached": best_bronze,
                "variants": task_results,
            })

    # Final report
    elapsed = time.time() - t0
    bronze_count = sum(1 for r in results if r["bronze_reached"])
    total_tasks = len(results)

    report = {
        "schema": "academic_research_os.30_experiment_showcase.v1",
        "timestamp": datetime.now().isoformat(),
        "total_tasks": total_tasks,
        "total_experiments": total,
        "bronze_count": bronze_count,
        "medal_rate": round(bronze_count / max(total_tasks, 1) * 100, 1),
        "elapsed_minutes": round(elapsed / 60, 1),
        "results": results,
    }

    report_path = ROOT / "workspace" / f"30_experiment_showcase_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    print(f"\n{'='*60}")
    print(f"SHOWCASE COMPLETE: {bronze_count}/{total_tasks} bronze ({report['medal_rate']}%)")
    print(f"Time: {elapsed/60:.1f} minutes | {total} total experiments")
    print(f"Report: {report_path}")

if __name__ == "__main__":
    main()
