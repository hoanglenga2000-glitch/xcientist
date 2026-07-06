from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EnsembleTemplate:
    template_id: str
    name: str
    model_family: list[str]
    description: str
    validation_strategy: str
    hpc_required: bool
    command_template: str
    required_inputs: list[str]
    expected_outputs: list[str]
    params: dict[str, Any] = field(default_factory=dict)
    seeds: list[int] = field(default_factory=list)
    approved: bool = True
    risk_level: str = "medium"


class EnsembleTemplateRegistry:
    """Whitelisted ensemble templates for workstation-managed training.

    Only templates registered here can be dispatched through the workstation
    HPC/GPU execution gate. Bypassing this registry is not allowed.
    """

    TEMPLATES: dict[str, EnsembleTemplate] = {
        "exp007_style_lgb_xgb_cat_blend": EnsembleTemplate(
            template_id="exp007_style_lgb_xgb_cat_blend",
            name="EXP007-Style LGB+XGB+CAT OOF Weighted Blend",
            model_family=["lightgbm", "xgboost", "catboost"],
            description="OOF-governed three-model weighted blend with grid search over blend weights. "
            "Uses saved OOF/test class probabilities from single-model runs. "
            "Historically achieved 0.96659 public on playground_series_s6e6.",
            validation_strategy="oof_weighted_blend_grid_search",
            hpc_required=True,
            command_template=(
                "python scripts/run_hpc_kaggle_boosting_ensemble.py "
                "--host {gpu_host} --port {gpu_port} --user {gpu_user} "
                "--proxy-host {proxy_host} --proxy-port {proxy_port} "
                "--password-env {password_env} "
                "--remote-root {remote_root} "
                "--local-artifact-dir {local_artifact_dir} "
                "--timeout-seconds {timeout_seconds}"
            ),
            required_inputs=[
                "train.csv",
                "test.csv",
                "sample_submission.csv",
                "exp003_lightgbm_oof.npz (or equivalent)",
                "exp004_xgboost_oof.npz (or equivalent)",
                "exp006_catboost_oof.npz (or equivalent)",
            ],
            expected_outputs=[
                "metrics.json",
                "submission.csv",
                "report.md",
                "manifest.json",
                "oof_predictions.csv",
                "weight_grid.csv",
            ],
            params={
                "weight_step": 0.02,
                "min_catboost_weight": 0.0,
                "max_catboost_weight": 0.20,
                "error_sample_rows": 1000,
            },
            seeds=[42, 2025, 260612],
            approved=True,
            risk_level="medium",
        ),
        "sklearn_rf_hgb_et_ensemble": EnsembleTemplate(
            template_id="sklearn_rf_hgb_et_ensemble",
            name="Sklearn RF+HGB+ET Ensemble with Logistic Stacking",
            model_family=["random_forest", "hist_gradient_boosting", "extra_trees"],
            description="Local sklearn ensemble: RF+HGB+ET with 5-fold CV, multi-seed, "
            "logistic regression stacking, and OOF blend grid search. "
            "No HPC required; runs on local CPU.",
            validation_strategy="5fold_stratified_cv_multiseed",
            hpc_required=False,
            command_template=(
                "python scripts/run_local_sklearn_ensemble.py "
                "--config {config_path} "
                "--output-base {output_base} "
                "--n-folds 5 "
                "--seeds 42,3407,12345 "
                "--random-state {random_state}"
            ),
            required_inputs=[
                "train.csv",
                "test.csv",
                "sample_submission.csv",
                "config YAML",
            ],
            expected_outputs=[
                "metrics.json",
                "submission.csv",
                "report.md",
                "manifest.json",
                "oof_predictions.csv",
                "per_class_metrics.csv",
            ],
            params={
                "n_folds": 5,
                "rf_n_estimators": 400,
                "rf_max_depth": 18,
                "hgb_max_iter": 300,
                "hgb_learning_rate": 0.05,
                "et_n_estimators": 400,
                "stacker_C": 1.0,
            },
            seeds=[42, 3407, 12345],
            approved=True,
            risk_level="low",
        ),
        "lightgbm_optuna_cv": EnsembleTemplate(
            template_id="lightgbm_optuna_cv",
            name="LightGBM Optuna Hyperparameter Search",
            model_family=["lightgbm"],
            description="Governed LightGBM Optuna TPE search over learning rate, leaves/depth, "
            "sampling, regularization, and class weighting. "
            "Bounded search space; full data on HPC.",
            validation_strategy="stratified_kfold_optuna_tpe",
            hpc_required=True,
            command_template=(
                "python scripts/run_hpc_exp016_lgbm_optuna_search.py "
                "--host {gpu_host} --port {gpu_port} --user {gpu_user} "
                "--proxy-host {proxy_host} --proxy-port {proxy_port} "
                "--password-env {password_env} "
                "--remote-root {remote_root} "
                "--local-artifact-dir {local_artifact_dir}"
            ),
            required_inputs=[
                "train.csv",
                "test.csv",
                "sample_submission.csv",
            ],
            expected_outputs=[
                "optuna_study_results.json",
                "metrics.json",
                "submission.csv",
                "report.md",
            ],
            params={
                "n_trials": 100,
                "timeout_seconds": 7200,
            },
            seeds=[260612],
            approved=True,
            risk_level="medium",
        ),
    }

    @classmethod
    def get(cls, template_id: str) -> EnsembleTemplate | None:
        return cls.TEMPLATES.get(template_id)

    @classmethod
    def list_approved(cls) -> list[EnsembleTemplate]:
        return [t for t in cls.TEMPLATES.values() if t.approved]

    @classmethod
    def list_local(cls) -> list[EnsembleTemplate]:
        return [t for t in cls.TEMPLATES.values() if not t.hpc_required]

    @classmethod
    def list_hpc(cls) -> list[EnsembleTemplate]:
        return [t for t in cls.TEMPLATES.values() if t.hpc_required]

    @classmethod
    def list_by_model(cls, model: str) -> list[EnsembleTemplate]:
        return [t for t in cls.TEMPLATES.values() if model in t.model_family]

    @classmethod
    def template_ids(cls) -> list[str]:
        return sorted(cls.TEMPLATES.keys())

    @classmethod
    def resolve_command(
        cls,
        template_id: str,
        **kwargs: str | int,
    ) -> str | None:
        template = cls.get(template_id)
        if template is None:
            return None
        try:
            return template.command_template.format(**kwargs)
        except KeyError:
            return None
