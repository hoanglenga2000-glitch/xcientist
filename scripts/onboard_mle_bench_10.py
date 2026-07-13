"""
Onboard 10 MLE-Bench tabular competitions into the workstation.
Downloads data via Kaggle API, creates configs, and registers tasks.
"""
import json, os, subprocess, sys, yaml
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = ROOT / "tasks"
CONFIGS_DIR = ROOT / "configs"

# 10 MLE-Bench competitions focusing on tabular/structured data
COMPETITIONS = [
    # (kaggle_slug, task_id, task_type, target, metric, direction, time_budget_h)
    ("house-prices-advanced-regression-techniques", "house_prices", "regression", "SalePrice", "rmsle", "minimize", 3),
    ("tabular-playground-series-dec-2021", "tps_dec_2021", "classification", "Cover_Type", "accuracy", "maximize", 4),
    ("new-york-city-taxi-fare-prediction", "taxi_fare", "regression", "fare_amount", "rmse", "minimize", 6),
    ("ventilator-pressure-prediction", "ventilator", "regression", "pressure", "mae", "minimize", 6),
    ("champs-scalar-coupling", "champs", "regression", "scalar_coupling_constant", "mae", "minimize", 6),
    ("osic-pulmonary-fibrosis-progression", "osic", "regression", "FVC", "laplace_log_likelihood", "minimize", 8),
    ("stanford-covid-vaccine", "covid_vaccine", "regression", "reactivity", "mcre", "minimize", 6),
    ("predict-volcanic-eruptions-ingv-oe", "volcanic", "classification", "eruption", "accuracy", "maximize", 4),
    ("nomad2018-predict-transparent-conductors", "nomad", "regression", "band_gap", "mae", "minimize", 6),
    ("petfinder-pawpularity-score", "petfinder", "regression", "Pawpularity", "rmse", "minimize", 4),
]

def download_competition(slug, task_id):
    """Download competition data via Kaggle API."""
    task_dir = TASKS_DIR / task_id / "data"
    task_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading {slug}...")
    r = subprocess.run(
        ["python", "-m", "kaggle", "competitions", "download", "-c", slug, "-p", str(task_dir)],
        capture_output=True, text=True, timeout=120
    )
    if r.returncode != 0:
        # Try kagglehub as fallback
        print(f"    Kaggle CLI failed: {r.stderr[:100]}. Trying kagglehub...")
        try:
            import kagglehub
            path = kagglehub.competition_download(slug)
            print(f"    Downloaded to: {path}")
        except Exception as e:
            print(f"    kagglehub also failed: {e}")
            return False

    # Unzip any .zip files
    for zf in task_dir.glob("*.zip"):
        subprocess.run(["python", "-c", f"import zipfile; zipfile.ZipFile('{zf}').extractall('{task_dir}')"],
                       capture_output=True, timeout=60)
        zf.unlink()
        print(f"    Extracted {zf.name}")

    # Check for train/test CSV
    csvs = list(task_dir.glob("*.csv"))
    print(f"    Files: {[c.name for c in csvs]}")
    return len(csvs) > 0

def create_config(task_id, slug, task_type, target, metric, direction, time_budget):
    """Create workstation config YAML for the task."""
    config = {
        "task": {
            "name": task_id,
            "competition": slug,
            "type": task_type + ("_classification" if task_type == "classification" else ""),
            "target": target,
            "metric": metric,
        },
        "data": {
            "task_dir": f"tasks/{task_id}",
            "train": f"tasks/{task_id}/data/train.csv",
            "test": f"tasks/{task_id}/data/test.csv",
            "sample_submission": f"tasks/{task_id}/data/sample_submission.csv",
        },
        "workflow": [
            "task_understanding", "preliminary_eda", "data_quality_check",
            "feature_engineering", "model_validation", "submission_generation",
            "report_and_review"
        ],
        "feature_engineering": {
            "preset": "tabular_basic",
            "target_transform": None,
            "drop_columns": [],
        },
        "thresholds": {
            "require_submission_schema_valid": True,
            "require_no_missing_predictions": True,
            "require_train_test_features_match": True,
        },
        "scaffold": {
            "time_budget_minutes": time_budget * 60,
            "validation_strategy": "5-fold stratified cross-validation",
            "first_stage_models": ["random_forest", "extra_trees", "hist_gradient_boosting"],
        },
    }
    config_path = CONFIGS_DIR / f"{task_id}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"  Config: {config_path}")
    return True

def main():
    print(f"=== Onboarding {len(COMPETITIONS)} MLE-Bench Competitions ===\n")

    results = []
    for slug, task_id, task_type, target, metric, direction, budget in COMPETITIONS:
        print(f"\n[{task_id}] {slug} ({task_type}, {metric})")

        # Download
        ok = download_competition(slug, task_id)

        # Config
        if ok:
            create_config(task_id, slug, task_type, target, metric, direction, budget)
            results.append({"task_id": task_id, "slug": slug, "status": "onboarded"})
        else:
            results.append({"task_id": task_id, "slug": slug, "status": "download_failed"})

    # Summary
    print(f"\n=== RESULTS ===")
    onboarded = [r for r in results if r["status"] == "onboarded"]
    failed = [r for r in results if r["status"] != "onboarded"]
    print(f"Onboarded: {len(onboarded)}/{len(COMPETITIONS)}")
    for r in onboarded:
        print(f"  ✅ {r['task_id']}: {r['slug']}")
    for r in failed:
        print(f"  ❌ {r['task_id']}: {r['slug']} ({r['status']})")

    # Save manifest
    manifest = {
        "schema": "academic_research_os.mle_bench_10_onboard.v1",
        "timestamp": datetime.now().isoformat(),
        "total": len(COMPETITIONS),
        "onboarded": len(onboarded),
        "competitions": results,
        "next_action": "Trigger run-ensemble-experiment for each onboarded task via workstation API"
    }
    manifest_path = ROOT / "workspace" / "mle_bench_10_onboard_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")

if __name__ == "__main__":
    main()
