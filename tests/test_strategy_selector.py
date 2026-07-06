"""Tests for the gold-medal strategy recommender."""
from __future__ import annotations

import json

from research_os.strategy_selector import (
    StrategyRecommendation,
    TaskProfile,
    recommend_strategies,
)


def test_high_cardinality_triggers_target_encoding():
    p = TaskProfile(modality="tabular", task_type="classification", train_size=50000,
                    n_features=40, n_high_cardinality_features=5)
    rec = recommend_strategies(p)
    assert "target_encoding" in rec.strategies


def test_low_cardinality_no_target_encoding():
    p = TaskProfile(modality="tabular", task_type="classification", train_size=50000,
                    n_features=40, n_high_cardinality_features=1)
    rec = recommend_strategies(p)
    assert "target_encoding" not in rec.strategies


def test_small_data_triggers_multi_seed():
    p = TaskProfile(modality="tabular", task_type="classification", train_size=891, n_features=10)
    rec = recommend_strategies(p)
    assert "multi_seed_ensemble" in rec.strategies


def test_large_test_triggers_pseudo_labeling():
    p = TaskProfile(modality="tabular", task_type="classification", train_size=1000,
                    test_size=10000, n_features=10)
    rec = recommend_strategies(p)
    assert "pseudo_labeling" in rec.strategies


def test_image_triggers_tta():
    p = TaskProfile(modality="image", task_type="classification", train_size=42000)
    rec = recommend_strategies(p)
    assert "test_time_augmentation" in rec.strategies


def test_rmsle_regression_triggers_log1p():
    p = TaskProfile(modality="tabular", task_type="regression", train_size=50000, metric="rmsle")
    rec = recommend_strategies(p)
    assert "log1p_target" in rec.strategies


def test_positive_target_triggers_log1p():
    p = TaskProfile(modality="tabular", task_type="regression", train_size=50000,
                    metric="rmse", target_is_positive=True)
    rec = recommend_strategies(p)
    assert "log1p_target" in rec.strategies


def test_time_series_forces_ts_cv_and_lags():
    p = TaskProfile(modality="time_series", task_type="regression", train_size=50000, metric="rmsle")
    rec = recommend_strategies(p)
    assert "time_series_cv" in rec.strategies
    assert "lag_features" in rec.strategies


def test_has_time_column_forces_ts_cv_even_if_tabular():
    p = TaskProfile(modality="tabular", task_type="regression", train_size=50000, has_time_column=True)
    rec = recommend_strategies(p)
    assert "time_series_cv" in rec.strategies


def test_model_diversity_triggers_stacking():
    p = TaskProfile(modality="tabular", task_type="classification", train_size=50000,
                    n_features=20, n_model_families=3)
    rec = recommend_strategies(p)
    assert "oof_stacking" in rec.strategies


def test_strategies_are_deduped_and_ordered():
    p = TaskProfile(modality="time_series", task_type="regression", train_size=500,
                    test_size=5000, metric="rmsle", n_features=20,
                    n_high_cardinality_features=4, n_model_families=3, has_time_column=True)
    rec = recommend_strategies(p)
    assert len(rec.strategies) == len(set(rec.strategies))
    # time-series guard should come first
    assert rec.strategies[0] == "time_series_cv"


def test_expected_gains_present_for_known_strategies():
    p = TaskProfile(modality="image", task_type="classification", train_size=42000)
    rec = recommend_strategies(p)
    assert "test_time_augmentation" in rec.expected_gains


def test_recommendation_json_serializable():
    p = TaskProfile(modality="tabular", task_type="classification", train_size=5000, n_features=20)
    rec = recommend_strategies(p)
    assert isinstance(rec, StrategyRecommendation)
    json.dumps(rec.to_dict())
