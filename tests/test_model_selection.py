"""Tests for data-profile-driven model strategy selection."""
from __future__ import annotations

import json

from research_os.model_selection import (
    DataProfile,
    ModelStrategy,
    TrainingPlan,
    recommend_model_strategy,
    resolve_training_plan,
)


def test_small_tabular_prefers_catboost():
    # titanic-like: ~900 rows, small.
    profile = DataProfile(task_type="classification", n_rows=891, n_cols=11, metric="accuracy", n_classes=2)
    strat = recommend_model_strategy(profile)
    assert strat.primary_model == "catboost"
    assert "catboost" in strat.model_families


def test_large_tabular_prefers_lightgbm():
    profile = DataProfile(task_type="classification", n_rows=500_000, n_cols=30, metric="roc_auc", n_classes=2)
    strat = recommend_model_strategy(profile)
    assert strat.primary_model == "lightgbm"
    assert set(strat.model_families) >= {"lightgbm", "xgboost", "catboost"}


def test_pixel_like_recommends_neural_net():
    # digit-recognizer: 42000 rows, 784 pixel columns.
    profile = DataProfile(
        task_type="classification", n_rows=42000, n_cols=784, metric="accuracy", n_classes=10, is_pixel_like=True
    )
    strat = recommend_model_strategy(profile)
    assert strat.primary_model == "neural_net"
    assert strat.hyperparams.get("arch") == "cnn"
    assert "normalize_pixels" in strat.feature_engineering
    assert any("pixel" in n or "CNN" in n for n in strat.notes)


def test_high_dim_non_pixel_still_flags_nn():
    profile = DataProfile(task_type="regression", n_rows=10000, n_cols=800, metric="rmse")
    strat = recommend_model_strategy(profile)
    assert strat.primary_model == "neural_net"
    assert any("high-dimensional" in n for n in strat.notes)


def test_high_categorical_ratio_adds_target_encoding():
    profile = DataProfile(
        task_type="classification", n_rows=50000, n_cols=20, metric="roc_auc", n_classes=2, categorical_ratio=0.5
    )
    strat = recommend_model_strategy(profile)
    assert "target_encoding" in strat.feature_engineering


def test_stagnation_escalates_to_stacking_and_exploit():
    profile = DataProfile(task_type="classification", n_rows=50000, n_cols=20, metric="accuracy", n_classes=2)
    strat = recommend_model_strategy(profile, rounds_without_improvement=3)
    assert strat.ensemble_strategy == "stacking"
    assert strat.exploration_mode == "exploit"


def test_first_round_is_explore_single():
    profile = DataProfile(task_type="classification", n_rows=50000, n_cols=20, metric="accuracy", n_classes=2)
    strat = recommend_model_strategy(profile)
    assert strat.exploration_mode == "explore"
    assert strat.ensemble_strategy == "single"


def test_best_score_switches_to_blend_exploit():
    profile = DataProfile(task_type="regression", n_rows=50000, n_cols=20, metric="rmse")
    strat = recommend_model_strategy(profile, best_score_so_far=0.42)
    assert strat.ensemble_strategy == "weighted_blend"
    assert strat.exploration_mode == "exploit"


def test_hyperparams_scale_with_size():
    small = recommend_model_strategy(
        DataProfile(task_type="classification", n_rows=1000, n_cols=10, metric="accuracy", n_classes=2)
    )
    large = recommend_model_strategy(
        DataProfile(task_type="classification", n_rows=300_000, n_cols=10, metric="roc_auc", n_classes=2)
    )
    # CatBoost depth (small) vs LightGBM leaves (large) both present and sane.
    assert small.hyperparams["depth"] in (6, 8)
    assert large.hyperparams["num_leaves"] >= 31


def test_strategy_is_json_serializable():
    profile = DataProfile(task_type="classification", n_rows=1000, n_cols=10, metric="accuracy", n_classes=2)
    strat = recommend_model_strategy(profile)
    assert isinstance(strat, ModelStrategy)
    json.dumps(strat.to_dict())


def test_memory_hits_recorded_in_notes():
    profile = DataProfile(task_type="classification", n_rows=50000, n_cols=20, metric="accuracy", n_classes=2)
    strat = recommend_model_strategy(profile, memory_hits=[{"strategy": "x"}, {"strategy": "y"}])
    assert any("retrospective-memory" in n for n in strat.notes)


# ── Modality detection ──

from research_os.model_selection import detect_modality, profile_from_frame_stats  # noqa: E402


def test_detect_modality_image_from_pixels():
    cols = ["label"] + [f"pixel{i}" for i in range(784)]
    assert detect_modality(cols) == "image"


def test_detect_modality_text_from_column_name():
    assert detect_modality(["id", "text", "target"]) == "text"
    assert detect_modality(["id", "comment_text", "toxic"]) == "text"


def test_detect_modality_text_from_length():
    assert detect_modality(["id", "content", "y"], avg_text_length=350) == "text"


def test_detect_modality_time_series():
    assert detect_modality(["date", "sales"]) == "time_series"
    assert detect_modality(["id", "value"], has_time_column=True) == "time_series"


def test_detect_modality_tabular_default():
    assert detect_modality(["id", "age", "fare", "pclass", "survived"]) == "tabular"


def test_detect_modality_pixel_priority_over_few_cols():
    # A handful of pixel columns among many features should NOT be image.
    cols = ["pixel0", "pixel1", "age", "fare", "income"]
    assert detect_modality(cols) == "tabular"


def test_profile_from_frame_stats_flags_pixel_for_image():
    cols = ["label"] + [f"pixel{i}" for i in range(784)]
    profile = profile_from_frame_stats(
        task_type="classification", column_names=cols, n_rows=42000, metric="accuracy", n_classes=10
    )
    assert profile.is_pixel_like is True
    # And the resulting strategy should recommend a neural net.
    strat = recommend_model_strategy(profile)
    assert strat.primary_model == "neural_net"


def test_profile_from_frame_stats_tabular_not_pixel():
    profile = profile_from_frame_stats(
        task_type="classification", column_names=["id", "age", "fare"], n_rows=891, metric="accuracy", n_classes=2
    )
    assert profile.is_pixel_like is False


# ── resolve_training_plan (P1: recommendation -> executable plan) ──

def test_resolve_plan_uses_available_gbdt_primary():
    profile = DataProfile(
        task_type="classification", n_rows=595212, n_cols=58,
        metric="normalized_gini", n_classes=2, categorical_ratio=0.35,
    )
    plan = resolve_training_plan(recommend_model_strategy(profile), profile)
    assert isinstance(plan, TrainingPlan)
    assert plan.executable_model in ("lightgbm", "catboost", "xgboost")
    assert plan.deferred_model is None
    # high-cardinality cats -> target encoding recommended
    assert "target_encoding" in plan.feature_engineering


def test_resolve_plan_defers_neural_and_falls_back_to_gbdt():
    profile = DataProfile(
        task_type="classification", n_rows=42000, n_cols=784,
        metric="accuracy", n_classes=10, is_pixel_like=True,
    )
    strategy = recommend_model_strategy(profile)
    assert strategy.primary_model == "neural_net"
    plan = resolve_training_plan(strategy, profile)
    # neural_net is not executable without a backend -> deferred, GBDT fallback chosen.
    assert plan.deferred_model == "neural_net"
    assert plan.executable_model in ("lightgbm", "catboost", "xgboost")
    assert any("not executable" in n for n in plan.notes)


def test_resolve_plan_runs_cnn_when_neural_available():
    profile = DataProfile(
        task_type="classification", n_rows=42000, n_cols=784,
        metric="accuracy", n_classes=10, is_pixel_like=True,
    )
    strategy = recommend_model_strategy(profile)
    plan = resolve_training_plan(strategy, profile, neural_available=True)
    # With a neural backend, the recommended net runs as a CNN (image modality).
    assert plan.executable_model == "cnn"
    assert plan.deferred_model is None
    assert any("CNN" in n for n in plan.notes)


def test_resolve_plan_neural_regression_still_defers():
    # Neural backend present but task is regression -> CNN branch not used; defer.
    profile = DataProfile(
        task_type="regression", n_rows=10000, n_cols=784,
        metric="rmse", is_pixel_like=True,
    )
    strategy = recommend_model_strategy(profile)
    plan = resolve_training_plan(strategy, profile, neural_available=True)
    assert plan.executable_model in ("lightgbm", "catboost", "xgboost")
    assert plan.deferred_model == "neural_net"


def test_resolve_plan_small_data_enables_multi_seed():
    profile = DataProfile(task_type="classification", n_rows=891, n_cols=10, metric="accuracy", n_classes=2)
    plan = resolve_training_plan(recommend_model_strategy(profile), profile)
    assert plan.multi_seed is True
    assert len(plan.seeds) == 5


def test_resolve_plan_large_data_no_multi_seed():
    profile = DataProfile(task_type="regression", n_rows=1_000_000, n_cols=20, metric="rmse")
    plan = resolve_training_plan(recommend_model_strategy(profile), profile)
    assert plan.multi_seed is False
    assert plan.seeds == []


def test_resolve_plan_respects_available_models_allowlist():
    profile = DataProfile(
        task_type="classification", n_rows=595212, n_cols=58,
        metric="normalized_gini", n_classes=2, categorical_ratio=0.35,
    )
    # Only catboost installed -> must fall back to it, no infeasible ensemble members.
    plan = resolve_training_plan(
        recommend_model_strategy(profile), profile, available_models=["catboost"]
    )
    assert plan.executable_model == "catboost"
    assert plan.ensemble_models == []


def test_resolve_plan_ensemble_excludes_primary_and_infeasible():
    profile = DataProfile(task_type="regression", n_rows=50_000, n_cols=30, metric="rmse")
    plan = resolve_training_plan(
        recommend_model_strategy(profile), profile,
        available_models=["catboost", "lightgbm", "xgboost"],
    )
    assert plan.executable_model not in plan.ensemble_models
    assert all(m in ("catboost", "lightgbm", "xgboost") for m in plan.ensemble_models)


def test_resolve_plan_to_dict_is_json_serializable():
    profile = DataProfile(task_type="classification", n_rows=891, n_cols=10, metric="accuracy", n_classes=2)
    plan = resolve_training_plan(recommend_model_strategy(profile), profile)
    payload = json.dumps(plan.to_dict())
    assert "executable_model" in payload

