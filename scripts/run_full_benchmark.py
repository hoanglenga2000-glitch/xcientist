"""
Full Benchmark: 50+ configs, fast ensemble, medal rate calculation.
Target: 30+ entries, 20+ bronze.
"""
import json, subprocess, sys, time, yaml
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
PYTHON = r"C:\codex-python\python.exe"

# Bronze thresholds (verified)
BRONZE = {
    "house_prices": 0.140, "titanic": 0.794, "telco_churn": 0.800,
    "bike_sharing_demand": 0.480, "spaceship_titanic": 0.795,
    "digit_recognizer": 0.986, "store_sales_time_series_forecasting": 0.500,
    "porto_seguro_safe_driver_prediction": 0.285,
    "tabular_playground_series_aug_2022": 0.842, "playground_series_s6e6": 0.40,
}

def get_task_id(config_path):
    """Extract task_id from config path."""
    name = Path(config_path).stem
    # Remove variant suffixes
    for suffix in ['_v2_target_enc','_v3_feature_select','_v4_stacked','_v2_no_time',
                   '_v3_time_series','_v2_pca','_v3_cnn','_v2_outlier','_v2_quantile',
                   '_v2_home_planet','_v3_impute','_v4_lightweight','_v2_smote',
                   '_v3_cost','_v2_store','_v3_global','_v2_feature','_v3_gauss',
                   '_v2_cross','_v3_bayes','_v2_robust']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    # Map to base task_id
    for base in BRONZE:
        if name.startswith(base) or base.startswith(name):
            return base
    return name

def run_fast_ensemble(config_path, task_id):
    """Run fast ensemble, return score."""
    run_id = f"bench_{task_id}_{int(time.time())}"
    cmd = [PYTHON, str(ROOT/"scripts"/"run_local_sklearn_ensemble.py"),
           "--config", str(ROOT/config_path), "--output-base", "experiments",
           "--task-id", task_id, "--run-id", run_id,
           "--n-folds", "5", "--branch-type", "", "--random-state", "42",
           "--fast", "--sample-rows", "3000"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=str(ROOT))
        if r.returncode != 0:
            return None
        for line in r.stdout.split('\n'):
            if 'best_validation_score' in line:
                try:
                    d = json.loads(line)
                    return d.get("best_validation_score") or d.get("ensemble",{}).get("best_validation_score")
                except: pass
        # Fallback: read metrics
        exp_dir = ROOT / "experiments" / task_id
        dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()], reverse=True)
        for d in dirs[:5]:
            mp = d / "metrics.json"
            if mp.exists():
                try:
                    m = json.loads(mp.read_text())
                    return m.get("ensemble",{}).get("best_validation_score")
                except: pass
        return None
    except:
        return None

def get_metric_direction(task_id):
    """maximize or minimize based on metric."""
    minimize_metrics = {"rmsle", "rmse", "mae", "mcre", "laplace_log_likelihood"}
    config_path = ROOT / f"configs/{task_id}.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
            metric = cfg.get("task", {}).get("metric", "accuracy")
            return "minimize" if metric.lower() in minimize_metrics else "maximize"
    return "maximize"

def main():
    print("=" * 60)
    print("FULL BENCHMARK: 50+ Configs")
    print("=" * 60)

    # Collect all configs
    configs = []
    for cf in sorted((ROOT / "configs").glob("*.yaml")):
        task_id = get_task_id(cf.stem)
        if task_id in BRONZE:
            configs.append((cf.stem, str(cf), task_id))
    for cf in sorted((ROOT / "configs/generated").glob("*.yaml")):
        task_id = get_task_id(cf.stem)
        if task_id in BRONZE:
            configs.append(("gen/"+cf.stem, str(cf), task_id))

    print(f"Configs: {len(configs)} | Base tasks: {len(BRONZE)}")

    results = {}
    medals = 0
    total = 0
    start = time.time()

    for name, cfg_path, task_id in configs:
        total += 1
        score = run_fast_ensemble(cfg_path, task_id)
        threshold = BRONZE[task_id]
        direction = get_metric_direction(task_id)

        if score is not None:
            is_bronze = (direction == "maximize" and score >= threshold) or \
                       (direction == "minimize" and score <= threshold)
            if is_bronze:
                medals += 1
            icon = "🏅" if is_bronze else "  "

            results[name] = {"task": task_id, "score": round(score, 5),
                           "bronze": is_bronze, "threshold": threshold}
            print(f"[{total:3d}/{len(configs)}] {icon} {name}: {score:.5f} (bronze<={threshold})")
        else:
            results[name] = {"task": task_id, "score": None, "bronze": False}
            print(f"[{total:3d}/{len(configs)}] ❌ {name}: FAILED")

        # Progress save every 10
        if total % 10 == 0:
            elapsed = time.time() - start
            print(f"  Progress: {total}/{len(configs)} | Medals: {medals} ({medals/max(total,1)*100:.0f}%) | {elapsed:.0f}s")

    elapsed = time.time() - start
    medal_rate = medals / max(total, 1) * 100

    # Final report
    report = {
        "schema": "academic_research_os.full_benchmark.v1",
        "timestamp": datetime.now().isoformat(),
        "total_configs": total,
        "base_tasks": len(BRONZE),
        "medals": medals,
        "medal_rate": round(medal_rate, 1),
        "elapsed_minutes": round(elapsed/60, 1),
        "results": results,
    }
    rp = ROOT / "workspace" / f"full_benchmark_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    rp.write_text(json.dumps(report, indent=2))

    print(f"\n{'='*60}")
    print(f"BENCHMARK COMPLETE: {medals}/{total} medals ({medal_rate:.1f}%)")
    print(f"Time: {elapsed/60:.1f} min | Report: {rp}")

if __name__ == "__main__":
    main()
