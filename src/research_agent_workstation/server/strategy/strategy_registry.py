from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..schemas.task import TaskProfile


@dataclass(slots=True)
class StrategyRecommendation:
    strategy_id: str
    task_type: str
    data_scale: str
    metric_type: str
    primary_templates: list[str]
    fallback_templates: list[str]
    ensemble_suggestion: str
    hpc_required: bool
    confidence: float
    rationale: str
    evidence_refs: list[str] = field(default_factory=list)


class StrategyRegistry:
    """Auto-recommends training templates based on task characteristics."""

    TEMPLATES = {
        "sklearn_baseline": {
            "models": ["logistic_regression", "random_forest", "extra_trees", "gradient_boosting"],
            "validation": "5fold_stratified_cv",
            "hpc_required": False,
            "target_metric": "balanced_accuracy",
            "description": "Sklearn baseline suite with 5-fold CV",
        },
        "lightgbm_multiclass": {
            "models": ["lightgbm"],
            "validation": "5fold_stratified_cv_multiseed",
            "hpc_required": True,
            "target_metric": "balanced_accuracy",
            "description": "LightGBM GBDT multiclass with multi-seed CV",
            "seeds": [42, 2025, 260612],
            "params": {
                "objective": "multiclass",
                "metric": "multi_logloss",
                "boosting_type": "gbdt",
                "n_estimators": 1500,
                "learning_rate": 0.03,
                "num_leaves": 63,
                "subsample": 0.78,
                "colsample_bytree": 0.65,
            },
        },
        "xgboost_multiclass": {
            "models": ["xgboost"],
            "validation": "5fold_stratified_cv_multiseed",
            "hpc_required": True,
            "target_metric": "balanced_accuracy",
            "description": "XGBoost GPU multiclass with multi-seed CV",
            "seeds": [42, 2025, 260612],
            "params": {
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "n_estimators": 1800,
                "learning_rate": 0.03,
                "max_depth": 8,
                "tree_method": "hist",
            },
        },
        "catboost_multiclass": {
            "models": ["catboost"],
            "validation": "5fold_stratified_cv_multiseed",
            "hpc_required": True,
            "target_metric": "balanced_accuracy",
            "description": "CatBoost GPU multiclass with multi-seed CV",
            "seeds": [42, 2025, 260612],
            "params": {
                "iterations": 2000,
                "learning_rate": 0.03,
                "depth": 8,
                "task_type": "GPU",
            },
        },
        "boosting_ensemble_blend": {
            "models": ["lightgbm", "xgboost", "catboost"],
            "validation": "oof_weighted_blend_grid_search",
            "hpc_required": True,
            "target_metric": "balanced_accuracy",
            "description": "OOF-governed weighted blend of LGB+XGB+CAT with grid search over blend weights",
        },
        "sklearn_ensemble_rf_hgb_et": {
            "models": ["random_forest", "hist_gradient_boosting", "extra_trees"],
            "validation": "5fold_stratified_cv_multiseed_plus_logistic_stack",
            "hpc_required": False,
            "target_metric": "balanced_accuracy",
            "description": "Sklearn RF+HGB+ET ensemble with logistic regression stacking",
            "seeds": [42, 3407, 12345],
        },
        # ── MLEvolve-Style Strategies (Progressive MCGS + Retrospective Memory) ──
        "mlevolve_progressive_mcgs": {
            "models": ["auto_select_by_mcgs"],
            "validation": "progressive_mcgs_5fold_oof",
            "hpc_required": False,
            "target_metric": "balanced_accuracy",
            "description": "Progressive MCGS with graph-based cross-branch search, entropy schedule, and retrospective memory",
            "search_engine": "mlevolve_search.MCEvolveSearchEngine",
            "expansion_types": ["primary", "intra_branch", "cross_branch", "aggregation"],
            "coding_modes": ["base", "diff", "stepwise"],
            "budget_hours": 12,
            "seeds": [42, 3407, 20250627],
        },
        "mlevolve_boosting_ensemble": {
            "models": ["lightgbm", "xgboost", "catboost"],
            "validation": "5fold_stratified_cv_multiseed_plus_optuna",
            "hpc_required": True,
            "target_metric": "accuracy",
            "description": "LGB+XGB+CatBoost ensemble with MLEvolve-guided Optuna tuning, cross-branch reference, and claim audit",
            "search_engine": "mlevolve_search.MCEvolveSearchEngine",
            "memory": "retrospective_memory.RetrospectiveMemory",
            "harness": "research_harness.XCIENTISTHarness",
            "budget_hours": 12,
            "ensemble_method": "weighted_blend_with_calibration",
        },
        "mlevolve_model_family_diversity": {
            "models": ["catboost", "hist_gradient_boosting", "extra_trees", "logistic_regression"],
            "validation": "3fold_stratified_cv_oof",
            "hpc_required": False,
            "target_metric": "accuracy",
            "description": "Model family diversity branch: CatBoost+HGB+ET+LR with different encoding strategies, MLEvolve-guided fusion",
            "search_engine": "mlevolve_search.MCEvolveSearchEngine",
            "expansion_type": "cross_branch",
            "coding_mode": "diff",
            "branch_type": "model_family",
        },
        "mlevolve_feature_interaction": {
            "models": ["lightgbm", "hist_gradient_boosting"],
            "validation": "5fold_stratified_cv_oof",
            "hpc_required": False,
            "target_metric": "accuracy",
            "description": "Feature interaction exploitation: group/cabin/aggregation features with MLEvolve stepwise coding",
            "expansion_type": "intra_branch",
            "coding_mode": "stepwise",
            "branch_type": "feature_engineering",
        },
    }

    DATA_SCALE_THRESHOLDS = {
        "small": 5000,
        "medium": 50000,
        "large": 200000,
        "xlarge": 1000000,
    }

    @classmethod
    def classify_scale(cls, train_rows: int) -> str:
        if train_rows <= cls.DATA_SCALE_THRESHOLDS["small"]:
            return "small"
        if train_rows <= cls.DATA_SCALE_THRESHOLDS["medium"]:
            return "medium"
        if train_rows <= cls.DATA_SCALE_THRESHOLDS["large"]:
            return "large"
        return "xlarge"

    @classmethod
    def recommend(
        cls,
        task_profile: TaskProfile,
        data_quality: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> list[StrategyRecommendation]:
        task_type = task_profile.task_type
        metric = task_profile.metric
        train_rows = data_quality.get("train_rows", 0) if data_quality else 0
        scale = cls.classify_scale(train_rows)

        recommendations: list[StrategyRecommendation] = []

        if task_type == "classification":
            # Always start with sklearn baseline (fast, no HPC needed)
            recommendations.append(StrategyRecommendation(
                strategy_id="s1_sklearn_baseline",
                task_type=task_type,
                data_scale=scale,
                metric_type=metric,
                primary_templates=["sklearn_baseline"],
                fallback_templates=[],
                ensemble_suggestion="none",
                hpc_required=False,
                confidence=0.95,
                rationale="Sklearn baselines establish a reproducible reference with minimal compute.",
                evidence_refs=["EXP001", "EXP002"],
            ))

            # For medium+ scale classification, recommend boosting
            if scale in ("medium", "large", "xlarge"):
                recommendations.append(StrategyRecommendation(
                    strategy_id="s2_boosting_single_models",
                    task_type=task_type,
                    data_scale=scale,
                    metric_type=metric,
                    primary_templates=["lightgbm_multiclass", "xgboost_multiclass", "catboost_multiclass"],
                    fallback_templates=["sklearn_ensemble_rf_hgb_et"],
                    ensemble_suggestion="Run all three boosting models then blend via OOF grid search",
                    hpc_required=True,
                    confidence=0.90,
                    rationale="Boosting models typically outperform sklearn on tabular classification with sufficient data.",
                    evidence_refs=["EXP003", "EXP004", "EXP006"],
                ))

            # Ensemble recommendation after single models
            recommendations.append(StrategyRecommendation(
                strategy_id="s3_ensemble_blend",
                task_type=task_type,
                data_scale=scale,
                metric_type=metric,
                primary_templates=["boosting_ensemble_blend"],
                fallback_templates=["sklearn_ensemble_rf_hgb_et"],
                ensemble_suggestion="Weighted OOF blend + logistic stacker, pick best method by balanced_accuracy",
                hpc_required=True,
                confidence=0.85,
                rationale="Ensemble blending consistently improves over single models in tabular competitions.",
                evidence_refs=["EXP007", "EXP010", "EXP011"],
            ))

            # If historical evidence exists, factor it in
            if history:
                best_historical = max(
                    (h for h in history if h.get("score")),
                    key=lambda h: h["score"],
                    default=None,
                )
                if best_historical:
                    recommendations.append(StrategyRecommendation(
                        strategy_id="s4_history_guided",
                        task_type=task_type,
                        data_scale=scale,
                        metric_type=metric,
                        primary_templates=["boosting_ensemble_blend"],
                        fallback_templates=["lightgbm_multiclass"],
                        ensemble_suggestion="Use historical best as target; start with same model family",
                        hpc_required=True,
                        confidence=0.80,
                        rationale=f"Historical best score {best_historical['score']} guides model selection.",
                        evidence_refs=[best_historical.get("experiment_id", "unknown")],
                    ))

        elif task_type == "regression":
            recommendations.append(StrategyRecommendation(
                strategy_id="s1_regression_baseline",
                task_type=task_type,
                data_scale=scale,
                metric_type=metric,
                primary_templates=["sklearn_baseline"],
                fallback_templates=[],
                ensemble_suggestion="none",
                hpc_required=False,
                confidence=0.95,
                rationale="Sklearn regression baselines with RMSLE/MAE/RMSE validation.",
            ))

        return recommendations

    @classmethod
    def get_template(cls, template_name: str) -> dict[str, Any] | None:
        return cls.TEMPLATES.get(template_name)

    @classmethod
    def is_hpc_required(cls, template_name: str) -> bool:
        template = cls.TEMPLATES.get(template_name, {})
        return template.get("hpc_required", False)

    @classmethod
    def list_all_templates(cls) -> list[str]:
        return sorted(cls.TEMPLATES.keys())

    @classmethod
    def list_hpc_templates(cls) -> list[str]:
        return [name for name, tmpl in cls.TEMPLATES.items() if tmpl.get("hpc_required")]

    @classmethod
    def list_local_templates(cls) -> list[str]:
        return [name for name, tmpl in cls.TEMPLATES.items() if not tmpl.get("hpc_required")]
