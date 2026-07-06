"""
Full optimization: train ALL competitions with full data, gate check, submit only if bronze-level.
"""
import subprocess, json, time, os, yaml
from pathlib import Path
from datetime import datetime

ROOT = Path(r'D:\桌面\codex\科研港科技')
PYTHON = r'C:\codex-python\python.exe'

# All competitions with verified Kaggle bronze thresholds
TASKS = [
    # (task_id, kaggle_slug, metric, direction, bronze_threshold)
    ("house_prices", "house-prices-advanced-regression-techniques", "rmsle", "min", 0.140),
    ("titanic", "titanic", "accuracy", "max", 0.794),
    ("spaceship_titanic", "spaceship-titanic", "accuracy", "max", 0.795),
    ("bike_sharing_demand", "bike-sharing-demand", "rmsle", "min", 0.480),
    ("digit_recognizer", "digit-recognizer", "accuracy", "max", 0.986),
    ("playground_series_s6e6", "playground-series-s6e6", "accuracy", "max", 0.400),
    ("ps3e7", "playground-series-s3e7", "accuracy", "max", 0.800),
    ("ps3e1", "playground-series-s3e1", "rmse", "min", 0.600),
    ("tps_feb2022", "tabular-playground-series-feb-2022", "accuracy", "max", 0.800),
    ("tps_dec2021", "tabular-playground-series-dec-2021", "accuracy", "max", 0.850),
    ("tps_may2022", "tabular-playground-series-may-2022", "accuracy", "max", 0.750),
    ("playground_s4e1", "playground-series-s4e1", "accuracy", "max", 0.750),
    ("ps4e7", "playground-series-s4e7", "accuracy", "max", 0.600),
    ("tabular_playground_series_aug_2022", "tabular-playground-series-aug-2022", "roc_auc", "max", 0.842),
    ("porto_seguro_safe_driver_prediction", "porto-seguro-safe-driver-prediction", "normalized_gini", "max", 0.285),
    ("store_sales_time_series_forecasting", "store-sales-time-series-forecasting", "rmsle", "min", 0.500),
    ("ps6e2", "playground-series-s6e2", "accuracy", "max", 0.800),
    ("ps6e3", "playground-series-s6e3", "accuracy", "max", 0.800),
    ("ps3e25", "playground-series-s3e25", "accuracy", "max", 0.700),
]

def run_full_training(task_id, n_folds=5):
    """Run full ensemble training, return (oof_score, submission_path)."""
    config = ROOT / f"configs/{task_id}.yaml"
    if not config.exists():
        return None, None

    run_id = f"full_opt_{int(time.time())}"
    cmd = [PYTHON, str(ROOT / "scripts/run_local_sklearn_ensemble.py"),
           "--config", str(config), "--output-base", "experiments",
           "--task-id", task_id, "--n-folds", str(n_folds),
           "--random-state", "42", "--run-id", run_id]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(ROOT))
        if r.returncode != 0:
            return None, None

        score = None
        for line in r.stdout.split('\n'):
            if 'best_validation_score' in line:
                try:
                    d = json.loads(line)
                    score = d.get("best_validation_score") or d.get("ensemble", {}).get("best_validation_score")
                except: pass

        if score is None:
            exp_dir = ROOT / "experiments" / task_id
            dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()], reverse=True)
            for d in dirs[:5]:
                mp = d / "metrics.json"
                if mp.exists():
                    m = json.loads(mp.read_text())
                    score = m.get("ensemble", {}).get("best_validation_score")
                    if score: break

        # Find submission
        exp_dir = ROOT / "experiments" / task_id
        dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()], reverse=True)
        sub_path = None
        for d in dirs[:5]:
            sp = d / "submission.csv"
            if sp.exists():
                sub_path = str(sp)
                break

        return score, sub_path
    except Exception as e:
        return None, None

def is_bronze(score, threshold, direction):
    if score is None: return False
    if direction == "max": return score >= threshold
    return score <= threshold

def main():
    print("=" * 60)
    print("FULL OPTIMIZE & GATE CHECK")
    print(f"Tasks: {len(TASKS)} | Mode: FULL | Started: {datetime.now()}")
    print("=" * 60)

    results = {}
    medals = 0
    submitted = 0

    for task_id, slug, metric, direction, bronze in TASKS:
        config = ROOT / f"configs/{task_id}.yaml"
        if not config.exists():
            print(f"  {task_id}: SKIP (no config)")
            continue

        # Use 3-fold for large datasets, 5-fold for small
        n_folds = 3 if task_id in ("spaceship_titanic", "store_sales_time_series_forecasting",
                                    "porto_seguro_safe_driver_prediction", "ps6e2", "ps6e3") else 5

        print(f"  {task_id}...", end=" ", flush=True)
        t0 = time.time()
        score, sub_path = run_full_training(task_id, n_folds)
        elapsed = time.time() - t0

        if score is None:
            print(f"FAIL [{elapsed:.0f}s]")
            results[task_id] = {"status": "failed"}
            continue

        bronze_ok = is_bronze(score, bronze, direction)
        if bronze_ok: medals += 1
        icon = "🏅" if bronze_ok else "  "

        # Gate check: only submit if bronze
        if bronze_ok and sub_path:
            r = subprocess.run(["python", "-m", "kaggle", "competitions", "submit",
                "-c", slug, "-f", sub_path, "-m", f"FULL-OPT OOF={score:.4f} bronze<={bronze}"],
                capture_output=True, text=True, timeout=30)
            sub_ok = "Success" in r.stdout
            if sub_ok: submitted += 1
            print(f"{icon} OOF={score:.4f}>={bronze} Kaggle={'OK' if sub_ok else 'FAIL'} [{elapsed:.0f}s]")
        else:
            status = f"OOF={score:.4f}"
            if not bronze_ok:
                gap = bronze - score if direction == "max" else score - bronze
                status += f" need {gap:+.4f}"
            print(f"{icon} {status} [{elapsed:.0f}s]")

        results[task_id] = {"oof": score, "bronze": bronze_ok, "submitted": bronze_ok and sub_path is not None}

    # Report
    total = len([r for r in results.values() if r.get("status") != "failed"])
    print(f"\n{'='*60}")
    print(f"RESULTS: {medals}/{total} bronze ({medals/max(total,1)*100:.0f}%) | {submitted} submitted to Kaggle")

    report = {"timestamp": datetime.now().isoformat(), "total": total, "medals": medals,
              "submitted": submitted, "results": {k: v for k, v in results.items()}}
    rp = ROOT / "workspace" / f"full_optimize_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    rp.write_text(json.dumps(report, indent=2))
    print(f"Report: {rp}")

if __name__ == "__main__":
    main()
