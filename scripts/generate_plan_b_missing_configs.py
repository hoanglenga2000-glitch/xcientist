import yaml, json, os
from pathlib import Path

ROOT = Path(r"D:\桌面\codex\科研港科技")
CONFIGS_DIR = ROOT / "configs"
GEN_DIR = CONFIGS_DIR / "generated"
GEN_DIR.mkdir(parents=True, exist_ok=True)

def deep_merge(base, updates):
    result = base.copy() if base else {}
    for k, v in updates.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result

# ============================================================
# Create base configs for 3 missing tasks
# ============================================================

base_configs = {
    "playground_series_s6e6": {
        "task": {
            "name": "playground_series_s6e6",
            "competition": "Playground Series - Season 6 Episode 6",
            "type": "multiclass_classification",
            "target": "class",
            "metric": "accuracy",
            "id_column": "id",
            "prediction_column": "class",
        },
        "data": {
            "task_dir": "tasks/playground_series_s6e6",
            "train": "tasks/playground_series_s6e6/data/train.csv",
            "test": "tasks/playground_series_s6e6/data/test.csv",
            "sample_submission": "tasks/playground_series_s6e6/data/sample_submission.csv",
        },
        "workflow": [
            "task_understanding", "preliminary_eda", "data_quality_check",
            "feature_engineering", "model_validation", "submission_generation",
            "report_and_review"
        ],
        "feature_engineering": {
            "preset": "tabular_basic",
            "target_transform": None,
            "drop_columns": ["id"],
            "numeric_strategy": "median",
            "categorical_strategy": "onehot",
        },
        "thresholds": {
            "require_submission_schema_valid": True,
            "require_no_missing_predictions": True,
            "require_train_test_features_match": True,
            "expected_submission_columns": ["id", "class"],
        },
        "scaffold": {
            "time_budget_minutes": 15,
            "validation_strategy": "5-fold stratified cross-validation",
            "first_stage_models": ["random_forest", "extra_trees", "hist_gradient_boosting", "xgboost"],
        },
    },
    "home_data_ml_course": {
        "task": {
            "name": "home_data_ml_course",
            "competition": "Home Data for ML Course",
            "type": "regression",
            "target": "SalePrice",
            "metric": "rmsle",
            "id_column": "Id",
            "prediction_column": "SalePrice",
        },
        "data": {
            "task_dir": "tasks/home_data_ml_course",
            "train": "tasks/home_data_ml_course/data/train.csv",
            "test": "tasks/home_data_ml_course/data/test.csv",
            "sample_submission": "tasks/home_data_ml_course/data/sample_submission.csv",
        },
        "workflow": [
            "task_understanding", "preliminary_eda", "data_quality_check",
            "feature_engineering", "model_validation", "submission_generation",
            "report_and_review"
        ],
        "feature_engineering": {
            "preset": "tabular_basic",
            "target_transform": "log1p",
            "drop_columns": ["Id"],
            "numeric_strategy": "median",
            "categorical_strategy": "onehot",
        },
        "thresholds": {
            "require_submission_schema_valid": True,
            "require_no_missing_predictions": True,
            "require_train_test_features_match": True,
            "require_positive_predictions": True,
            "expected_submission_rows": 1459,
        },
        "scaffold": {
            "time_budget_minutes": 15,
            "validation_strategy": "5-fold KFold cross-validation on log1p target",
            "first_stage_models": ["ridge_log_target", "random_forest_log_target", "gradient_boosting_log_target"],
        },
    },
    "kaggle_new_competition_smoke": {
        "task": {
            "name": "kaggle_new_competition_smoke",
            "competition": "Kaggle New Competition Smoke Test",
            "type": "regression",
            "target": "SalePrice",
            "metric": "rmsle",
            "id_column": "Id",
            "prediction_column": "SalePrice",
        },
        "data": {
            "task_dir": "tasks/kaggle_new_competition_smoke",
            "train": "tasks/kaggle_new_competition_smoke/data/train.csv",
            "test": "tasks/kaggle_new_competition_smoke/data/test.csv",
            "sample_submission": "tasks/kaggle_new_competition_smoke/data/sample_submission.csv",
        },
        "workflow": [
            "task_understanding", "preliminary_eda", "data_quality_check",
            "feature_engineering", "model_validation", "submission_generation",
            "report_and_review"
        ],
        "feature_engineering": {
            "preset": "tabular_basic",
            "target_transform": "log1p",
            "drop_columns": ["Id"],
            "numeric_strategy": "median",
            "categorical_strategy": "label_encoding",
        },
        "thresholds": {
            "require_submission_schema_valid": True,
            "require_no_missing_predictions": True,
            "require_train_test_features_match": True,
            "require_positive_predictions": True,
            "expected_submission_rows": 1459,
        },
        "scaffold": {
            "time_budget_minutes": 12,
            "validation_strategy": "5-fold KFold cross-validation on log1p target",
            "first_stage_models": ["random_forest_log_target", "extra_trees_log_target"],
        },
    },
}

# Write base configs
for task_id, config in base_configs.items():
    config_path = CONFIGS_DIR / f"{task_id}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"Created base config: {task_id}.yaml")

# ============================================================
# Now create variants for these 3 tasks
# ============================================================

def load_config(task_id):
    cfg_path = CONFIGS_DIR / f"{task_id}.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    return None

variants_extra = [
    ("playground_s6e6_v2_robust_scaler", "playground_series_s6e6", {
        "task": {"name": "playground_s6e6_v2_robust_scaler", "competition": "Playground S6E6 - Robust Scaler Variant"},
        "feature_engineering": {
            "preset": "tabular_robust",
            "target_transform": None,
            "drop_columns": ["id"],
            "numeric_strategy": "median",
            "categorical_strategy": "onehot",
            "scaler": "robust",
        },
        "scaffold": {
            "time_budget_minutes": 12,
            "first_stage_models": ["logistic_regression", "random_forest", "hist_gradient_boosting"],
            "validation_strategy": "5-fold stratified CV with robust scaling",
        },
    }),
    ("home_data_v2_outlier_robust", "home_data_ml_course", {
        "task": {"name": "home_data_v2_outlier_robust", "competition": "Home Data - Outlier Robust Variant"},
        "feature_engineering": {
            "preset": "tabular_basic",
            "target_transform": "log1p",
            "drop_columns": ["Id"],
            "numeric_strategy": "median",
            "categorical_strategy": "label_encoding",
            "outlier_handling": "winsorize_0.01",
        },
        "scaffold": {
            "time_budget_minutes": 12,
            "first_stage_models": ["random_forest", "extra_trees", "hist_gradient_boosting"],
            "validation_strategy": "5-fold KFold CV with winsorized features",
        },
    }),
    ("kaggle_smoke_v2_quantile_transform", "kaggle_new_competition_smoke", {
        "task": {"name": "kaggle_smoke_v2_quantile_transform", "competition": "Kaggle Smoke - Quantile Transform Variant"},
        "feature_engineering": {
            "preset": "tabular_quantile",
            "target_transform": "quantile",
            "drop_columns": ["Id"],
            "numeric_strategy": "median",
            "categorical_strategy": "onehot",
            "scaler": "quantile_uniform",
        },
        "scaffold": {
            "time_budget_minutes": 15,
            "first_stage_models": ["ridge", "random_forest", "hist_gradient_boosting"],
            "validation_strategy": "5-fold KFold CV with quantile-transformed features",
        },
    }),
]

results_extra = []
for variant_id, base_task_id, modifications in variants_extra:
    base_config = load_config(base_task_id)
    config = deep_merge(base_config, modifications)
    config["data"] = {
        "task_dir": f"tasks/{base_task_id}",
        "train": f"tasks/{base_task_id}/data/train.csv",
        "test": f"tasks/{base_task_id}/data/test.csv",
        "sample_submission": f"tasks/{base_task_id}/data/sample_submission.csv",
        "variant_of": base_task_id,
    }
    config_path = GEN_DIR / f"{variant_id}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    results_extra.append({"variant_id": variant_id, "base_task": base_task_id})
    print(f"OK variant: {variant_id} (base: {base_task_id})")

print(f"\n=== SUMMARY ===")
print(f"Base configs created: {len(base_configs)}")
print(f"Extra variants created: {len(results_extra)}")
