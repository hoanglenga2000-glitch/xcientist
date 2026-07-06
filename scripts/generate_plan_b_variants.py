import yaml, json, os
from datetime import datetime
from pathlib import Path

ROOT = Path(r"D:\桌面\codex\科研港科技")
CONFIGS_DIR = ROOT / "configs" / "generated"
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

def load_existing_config(task_id):
    cfg_path = ROOT / "configs" / f"{task_id}.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    return None

def deep_merge(base, updates):
    result = base.copy() if base else {}
    for k, v in updates.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result

VARIANTS = [
    # --- titanic (3 variants) ---
    ("titanic_v2_feature_rich", "titanic", {
        "task": {"name": "titanic_v2_feature_rich", "competition": "Titanic - Feature Rich Variant"},
        "feature_engineering": {
            "preset": "titanic_feature_rich",
            "target_transform": None,
            "drop_columns": ["Ticket"],
            "numeric_strategy": "knn_impute",
            "categorical_strategy": "target_encoding",
            "derived_features": ["title_from_name", "family_size", "cabin_deck", "fare_bin", "age_bin"],
        },
        "scaffold": {
            "time_budget_minutes": 15,
            "first_stage_models": ["logistic_regression", "random_forest", "xgboost", "lightgbm", "catboost"],
            "validation_strategy": "10-fold stratified CV with feature selection",
        },
        "thresholds": {"min_validation_accuracy": 0.80},
    }),
    ("titanic_v3_lean", "titanic", {
        "task": {"name": "titanic_v3_lean", "competition": "Titanic - Lean Variant"},
        "feature_engineering": {
            "preset": "titanic_lean",
            "target_transform": None,
            "drop_columns": ["Name", "Ticket", "Cabin", "Embarked"],
            "numeric_strategy": "median",
            "categorical_strategy": "label_encoding",
        },
        "scaffold": {
            "time_budget_minutes": 6,
            "first_stage_models": ["logistic_regression", "random_forest"],
            "validation_strategy": "3-fold stratified CV",
        },
        "thresholds": {"min_validation_accuracy": 0.76},
    }),
    ("titanic_v4_logodds", "titanic", {
        "task": {"name": "titanic_v4_logodds", "competition": "Titanic - LogOdds Ensemble Variant"},
        "feature_engineering": {
            "preset": "titanic_ensemble",
            "target_transform": "logit",
            "drop_columns": ["Name", "Ticket"],
            "numeric_strategy": "iterative_impute",
            "categorical_strategy": "onehot",
            "derived_features": ["title_from_name", "family_size", "cabin_deck", "fare_bin", "age_bin", "ticket_freq"],
        },
        "scaffold": {
            "time_budget_minutes": 20,
            "first_stage_models": ["logistic_regression", "random_forest", "xgboost", "lightgbm", "catboost", "svm"],
            "validation_strategy": "5x2-fold stratified CV with stacking ensemble",
        },
        "thresholds": {"min_validation_accuracy": 0.82},
    }),

    # --- house_prices (3 variants) ---
    ("house_prices_v2_target_enc", "house_prices", {
        "task": {"name": "house_prices_v2_target_enc", "competition": "House Prices - Target Encoding Variant"},
        "feature_engineering": {
            "preset": "house_prices_target_enc",
            "target_transform": "log1p",
            "drop_columns": ["Id"],
            "numeric_strategy": "median",
            "categorical_strategy": "target_encoding",
            "derived_features": ["total_sf", "total_baths", "total_porch_sf", "house_age", "remodel_age"],
        },
        "scaffold": {
            "time_budget_minutes": 20,
            "first_stage_models": ["ridge_log_target", "random_forest_log_target", "xgboost_log_target", "lightgbm_log_target"],
            "validation_strategy": "5-fold KFold CV on log1p target",
        },
    }),
    ("house_prices_v3_feature_select", "house_prices", {
        "task": {"name": "house_prices_v3_feature_select", "competition": "House Prices - Feature Selection Variant"},
        "feature_engineering": {
            "preset": "house_prices_selected",
            "target_transform": "log1p",
            "drop_columns": ["Id", "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"],
            "numeric_strategy": "knn_impute",
            "categorical_strategy": "onehot",
            "max_features": 50,
            "feature_selection": "mutual_info_regression",
        },
        "scaffold": {
            "time_budget_minutes": 12,
            "first_stage_models": ["ridge_log_target", "lasso_log_target", "elasticnet_log_target"],
            "validation_strategy": "5-fold KFold CV with L1 feature selection",
        },
    }),
    ("house_prices_v4_stacked", "house_prices", {
        "task": {"name": "house_prices_v4_stacked", "competition": "House Prices - Stacked Ensemble Variant"},
        "feature_engineering": {
            "preset": "house_prices_full",
            "target_transform": "box_cox",
            "drop_columns": ["Id"],
            "numeric_strategy": "iterative_impute",
            "categorical_strategy": "catboost_encoding",
            "derived_features": ["total_sf", "total_baths", "garage_age", "overall_quality_score"],
        },
        "scaffold": {
            "time_budget_minutes": 30,
            "first_stage_models": ["ridge", "lasso", "elasticnet", "random_forest", "xgboost", "lightgbm", "catboost"],
            "validation_strategy": "10-fold CV with stacking (ridge meta-learner)",
        },
    }),

    # --- bike_sharing_demand (2 variants) ---
    ("bike_sharing_v2_no_time_features", "bike_sharing_demand", {
        "task": {"name": "bike_sharing_v2_no_time_features", "competition": "Bike Sharing - No Time Features Variant"},
        "feature_engineering": {
            "preset": "tabular_basic",
            "target_transform": "log1p",
            "drop_columns": ["casual", "registered", "datetime"],
            "numeric_strategy": "median",
        },
        "scaffold": {
            "time_budget_minutes": 12,
            "first_stage_models": ["random_forest_log_target", "extra_trees_log_target", "hist_gradient_boosting_log_target"],
            "validation_strategy": "5-fold random shuffle CV (no time awareness)",
        },
    }),
    ("bike_sharing_v3_time_series_advanced", "bike_sharing_demand", {
        "task": {"name": "bike_sharing_v3_time_series_advanced", "competition": "Bike Sharing - Advanced Time Series Variant"},
        "feature_engineering": {
            "preset": "datetime_advanced",
            "target_transform": "sqrt",
            "drop_columns": ["casual", "registered"],
            "numeric_strategy": "median",
            "derived_features": ["hour", "dayofweek", "month", "quarter", "is_weekend", "is_holiday_approx", "season_sin", "season_cos"],
        },
        "scaffold": {
            "time_budget_minutes": 25,
            "first_stage_models": ["lightgbm", "xgboost", "catboost", "random_forest"],
            "validation_strategy": "TimeSeriesSplit with 5 splits",
        },
    }),

    # --- digit_recognizer (2 variants) ---
    ("digit_recognizer_v2_pca", "digit_recognizer", {
        "task": {"name": "digit_recognizer_v2_pca", "competition": "Digit Recognizer - PCA Variant"},
        "feature_engineering": {
            "preset": "image_pca",
            "target_transform": None,
            "drop_columns": [],
            "numeric_strategy": "zero",
            "pca_components": 50,
        },
        "scaffold": {
            "time_budget_minutes": 15,
            "first_stage_models": ["svm_rbf", "random_forest", "logistic_regression"],
            "validation_strategy": "5-fold stratified CV on PCA-reduced features",
        },
    }),
    ("digit_recognizer_v3_cnn_like", "digit_recognizer", {
        "task": {"name": "digit_recognizer_v3_cnn_like", "competition": "Digit Recognizer - CNN-like Features Variant"},
        "feature_engineering": {
            "preset": "image_pixel_raw",
            "target_transform": None,
            "drop_columns": [],
            "numeric_strategy": "zero",
            "derived_features": ["hog_features", "edge_features", "intensity_stats"],
        },
        "scaffold": {
            "time_budget_minutes": 20,
            "first_stage_models": ["mlp_classifier", "random_forest", "xgboost"],
            "validation_strategy": "5-fold stratified CV with image augmentation",
        },
    }),

    # --- spaceship_titanic (3 variants) ---
    ("spaceship_titanic_v2_home_planet_split", "spaceship_titanic", {
        "task": {"name": "spaceship_titanic_v2_home_planet_split", "competition": "Spaceship Titanic - HomePlanet Split Variant"},
        "feature_engineering": {
            "preset": "spaceship_by_planet",
            "target_transform": None,
            "drop_columns": ["Name", "PassengerId"],
            "numeric_strategy": "median",
            "categorical_strategy": "target_encoding",
            "derived_features": ["cabin_deck", "cabin_num", "cabin_side", "group_size", "total_spend"],
        },
        "scaffold": {
            "time_budget_minutes": 18,
            "first_stage_models": ["logistic_regression", "random_forest", "xgboost", "lightgbm"],
            "validation_strategy": "GroupKFold by HomePlanet",
        },
    }),
    ("spaceship_titanic_v3_imputation_focus", "spaceship_titanic", {
        "task": {"name": "spaceship_titanic_v3_imputation_focus", "competition": "Spaceship Titanic - Imputation Focus Variant"},
        "feature_engineering": {
            "preset": "spaceship_missing_aware",
            "target_transform": None,
            "drop_columns": ["Name", "PassengerId", "Cabin"],
            "numeric_strategy": "iterative_impute",
            "categorical_strategy": "onehot",
            "missing_indicators": True,
        },
        "scaffold": {
            "time_budget_minutes": 12,
            "first_stage_models": ["logistic_regression", "random_forest", "hist_gradient_boosting"],
            "validation_strategy": "5-fold stratified CV with missing indicator features",
        },
    }),
    ("spaceship_titanic_v4_lightweight", "spaceship_titanic", {
        "task": {"name": "spaceship_titanic_v4_lightweight", "competition": "Spaceship Titanic - Lightweight Variant"},
        "feature_engineering": {
            "preset": "tabular_basic",
            "target_transform": None,
            "drop_columns": ["Name", "PassengerId", "Cabin", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"],
            "numeric_strategy": "median",
            "categorical_strategy": "label_encoding",
        },
        "scaffold": {
            "time_budget_minutes": 6,
            "first_stage_models": ["logistic_regression", "random_forest"],
            "validation_strategy": "3-fold stratified CV",
        },
    }),

    # --- telco_churn (2 variants) ---
    ("telco_churn_v2_smote", "telco_churn", {
        "task": {"name": "telco_churn_v2_smote", "competition": "Telco Churn - SMOTE Variant"},
        "feature_engineering": {
            "preset": "telco_basic",
            "target_transform": None,
            "drop_columns": ["customerID"],
            "numeric_strategy": "median",
            "categorical_strategy": "onehot",
        },
        "scaffold": {
            "time_budget_minutes": 15,
            "first_stage_models": ["logistic_regression", "random_forest", "xgboost", "lightgbm"],
            "validation_strategy": "5-fold stratified CV with SMOTE oversampling",
        },
    }),
    ("telco_churn_v3_cost_sensitive", "telco_churn", {
        "task": {"name": "telco_churn_v3_cost_sensitive", "competition": "Telco Churn - Cost Sensitive Variant"},
        "feature_engineering": {
            "preset": "telco_advanced",
            "target_transform": None,
            "drop_columns": ["customerID"],
            "numeric_strategy": "median",
            "categorical_strategy": "target_encoding",
            "derived_features": ["total_charges_ratio", "tenure_bin", "monthly_charges_bin"],
        },
        "scaffold": {
            "time_budget_minutes": 20,
            "first_stage_models": ["logistic_regression_weighted", "random_forest_weighted", "xgboost_weighted"],
            "validation_strategy": "5-fold stratified CV with class_weight=balanced",
        },
    }),

    # --- store_sales (2 variants) ---
    ("store_sales_v2_store_specific_models", "store_sales_time_series_forecasting", {
        "task": {"name": "store_sales_v2_store_specific_models", "competition": "Store Sales - Store-Specific Models Variant"},
        "feature_engineering": {
            "preset": "timeseries_store_specific",
            "target_transform": "log1p",
            "drop_columns": ["id"],
            "numeric_strategy": "median",
            "derived_features": ["lag_7", "lag_14", "lag_28", "rolling_mean_7", "rolling_mean_14"],
        },
        "scaffold": {
            "time_budget_minutes": 30,
            "first_stage_models": ["lightgbm", "xgboost", "catboost"],
            "validation_strategy": "GroupTimeSeriesSplit by store_nbr",
        },
    }),
    ("store_sales_v3_global_model", "store_sales_time_series_forecasting", {
        "task": {"name": "store_sales_v3_global_model", "competition": "Store Sales - Global Model Variant"},
        "feature_engineering": {
            "preset": "timeseries_global",
            "target_transform": "log1p",
            "drop_columns": ["id"],
            "numeric_strategy": "median",
            "categorical_strategy": "onehot",
        },
        "scaffold": {
            "time_budget_minutes": 45,
            "first_stage_models": ["lightgbm_global", "xgboost_global"],
            "validation_strategy": "TimeSeriesSplit with expanding window",
        },
    }),

    # --- porto_seguro (2 variants) ---
    ("porto_seguro_v2_feature_selection", "porto_seguro_safe_driver_prediction", {
        "task": {"name": "porto_seguro_v2_feature_selection", "competition": "Porto Seguro - Feature Selection Variant"},
        "feature_engineering": {
            "preset": "porto_seguro_selected",
            "target_transform": None,
            "drop_columns": ["id"],
            "numeric_strategy": "median",
            "feature_selection": "lgbm_importance",
            "max_features": 30,
        },
        "scaffold": {
            "time_budget_minutes": 12,
            "first_stage_models": ["lightgbm", "random_forest"],
            "validation_strategy": "5-fold stratified CV with top-30 features",
        },
    }),
    ("porto_seguro_v3_gaussian_nb", "porto_seguro_safe_driver_prediction", {
        "task": {"name": "porto_seguro_v3_gaussian_nb", "competition": "Porto Seguro - Gaussian NB Variant"},
        "feature_engineering": {
            "preset": "porto_seguro_normalized",
            "target_transform": None,
            "drop_columns": ["id"],
            "numeric_strategy": "median",
            "scaler": "standard",
        },
        "scaffold": {
            "time_budget_minutes": 8,
            "first_stage_models": ["gaussian_nb", "logistic_regression", "random_forest"],
            "validation_strategy": "5-fold stratified CV with standardized features",
        },
    }),

    # --- tps_aug_2022 (2 variants) ---
    ("tps_aug_2022_v2_feature_cross", "tabular_playground_series_aug_2022", {
        "task": {"name": "tps_aug_2022_v2_feature_cross", "competition": "TPS Aug 2022 - Feature Cross Variant"},
        "feature_engineering": {
            "preset": "tabular_advanced",
            "target_transform": "log1p",
            "drop_columns": ["id"],
            "numeric_strategy": "median",
            "categorical_strategy": "onehot",
            "derived_features": ["interaction_features", "polynomial_features_degree2"],
        },
        "scaffold": {
            "time_budget_minutes": 20,
            "first_stage_models": ["ridge", "random_forest", "xgboost", "lightgbm"],
            "validation_strategy": "5-fold KFold CV with polynomial features",
        },
    }),
    ("tps_aug_2022_v3_bayesian_opt", "tabular_playground_series_aug_2022", {
        "task": {"name": "tps_aug_2022_v3_bayesian_opt", "competition": "TPS Aug 2022 - Bayesian Opt Variant"},
        "feature_engineering": {
            "preset": "tabular_basic",
            "target_transform": "box_cox",
            "drop_columns": ["id"],
            "numeric_strategy": "median",
            "categorical_strategy": "target_encoding",
        },
        "scaffold": {
            "time_budget_minutes": 30,
            "first_stage_models": ["xgboost_tuned", "lightgbm_tuned", "catboost_tuned"],
            "validation_strategy": "5-fold CV with Optuna hyperparameter optimization",
        },
    }),

    # --- playground_s6e6 (1 variant) ---
    ("playground_s6e6_v2_robust_scaler", "playground_series_s6e6", {
        "task": {"name": "playground_s6e6_v2_robust_scaler", "competition": "Playground S6E6 - Robust Scaler Variant"},
        "feature_engineering": {
            "preset": "tabular_robust",
            "target_transform": "log1p",
            "drop_columns": ["id"],
            "numeric_strategy": "median",
            "categorical_strategy": "onehot",
            "scaler": "robust",
        },
        "scaffold": {
            "time_budget_minutes": 12,
            "first_stage_models": ["ridge", "random_forest", "hist_gradient_boosting"],
            "validation_strategy": "5-fold KFold CV with robust scaling",
        },
    }),

    # --- home_data (1 variant) ---
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

    # --- kaggle_smoke (1 variant) ---
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

# ============================================================
# Generate and write all variant configs
# ============================================================
results = []
for variant_id, base_task_id, modifications in VARIANTS:
    base_config = load_existing_config(base_task_id)
    if base_config is None:
        print(f"WARNING: base config not found for {base_task_id}, skipping {variant_id}")
        continue

    config = deep_merge(base_config, modifications)
    config["data"] = {
        "task_dir": f"tasks/{base_task_id}",
        "train": f"tasks/{base_task_id}/data/train.csv",
        "test": f"tasks/{base_task_id}/data/test.csv",
        "sample_submission": f"tasks/{base_task_id}/data/sample_submission.csv",
        "variant_of": base_task_id,
    }

    config_path = CONFIGS_DIR / f"{variant_id}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    results.append({
        "variant_id": variant_id,
        "base_task": base_task_id,
        "config_path": str(config_path.relative_to(ROOT)),
        "status": "created",
    })
    print(f"OK: {variant_id} (base: {base_task_id})")

# ============================================================
# Generate manifest
# ============================================================
manifest = {
    "schema": "academic_research_os.mle_bench_variants.v1",
    "timestamp": datetime.now().isoformat(),
    "strategy": "Plan B - Variant competitions from existing data",
    "plan_a_results": {
        "total_attempted": 20,
        "downloaded_successfully": 1,
        "downloaded_failed": 19,
        "note": "Kaggle API 403 on most competitions - rules not accepted. Only kaaveland/tpsdec2021parquet succeeded as community dataset mirror.",
    },
    "variant_summary": {
        "total_variants_created": len(results),
        "base_tasks_used": len(set(r["base_task"] for r in results)),
        "total_benchmark_entries": len(results) + 12,
    },
    "variants": results,
    "variant_categories": {
        "feature_engineering_strategies": ["target_encoding", "onehot", "label_encoding", "catboost_encoding", "feature_crosses", "pca", "feature_selection_mutual_info", "feature_selection_lgbm"],
        "target_transformations": ["log1p", "sqrt", "box_cox", "quantile", "logit", "none"],
        "model_sets": ["simple_baseline", "gbm_focused", "stacked_ensemble", "regularized_linear", "cost_sensitive", "naive_bayes", "svm", "mlp"],
        "validation_strategies": ["kfold", "stratified_kfold", "time_series_split", "group_kfold", "expanding_window", "group_time_series_split"],
        "data_treatments": ["all_features", "top_n_features", "no_high_missing", "numeric_only", "derived_features", "robust_scaling", "quantile_transform", "standardization"],
    },
    "next_action": "Run benchmark evaluation across all 30 task configs (12 existing + 18 variants)",
}

manifest_path = ROOT / "workspace" / "plan_b_variants_manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\n{'='*60}")
print(f"Total variant configs created: {len(results)}")
print(f"Total benchmark entries (existing + variants): {len(results) + 12}")
print(f"Manifest: {manifest_path}")
print(f"{'='*60}")
