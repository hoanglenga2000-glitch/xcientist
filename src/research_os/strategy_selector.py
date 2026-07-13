"""Gold-medal strategy recommender.

Given a task profile, recommend the high-leverage techniques that Kaggle
gold-medal solutions and the MLEvolve paper repeatedly rely on. Pure, deterministic
logic so it is fully unit-testable and can be consulted by the trainer or the
onboarding pipeline before a run.

Strategies (with the condition that triggers them):
  target_encoding        - tabular with several high-cardinality categoricals
  multi_seed_ensemble    - small training set (high variance)
  pseudo_labeling        - test set much larger than train
  oof_stacking           - multiple diverse model families in play
  test_time_augmentation - image modality
  log1p_target           - RMSLE / right-skewed positive regression target
  feature_crossing       - tabular with a modest number of features
  time_series_cv         - temporal data (never shuffle!)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TaskProfile:
    modality: str  # tabular | image | text | time_series | multimodal
    task_type: str  # classification | regression
    train_size: int
    test_size: int = 0
    metric: str = ""
    n_features: int = 0
    n_high_cardinality_features: int = 0
    n_model_families: int = 1
    has_time_column: bool = False
    target_is_positive: bool = False


@dataclass
class StrategyRecommendation:
    strategies: list[str] = field(default_factory=list)
    rationale: dict[str, str] = field(default_factory=dict)
    expected_gains: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategies": list(self.strategies),
            "rationale": dict(self.rationale),
            "expected_gains": dict(self.expected_gains),
        }


# Expected-gain notes sourced from the Phase 3 strategy table.
_EXPECTED_GAINS = {
    "target_encoding": "+0.5-2% on the primary metric for high-cardinality cats",
    "multi_seed_ensemble": "+0.5-1% via variance reduction on small data",
    "pseudo_labeling": "+0.3-1% when unlabeled test data is plentiful",
    "oof_stacking": "+0.3-0.8% when base models are diverse",
    "test_time_augmentation": "+0.5-2% on image tasks",
    "log1p_target": "-30-70% error on RMSLE regression",
    "feature_crossing": "+0.2-0.5% on tabular",
    "tfidf_ngrams": "strong fast baseline on text before escalating to transformers",
    "multimodal_fusion": "fuse per-modality encoders to beat any single-modality model",
    "audio_spectrogram": "log-mel spectrogram + CNN is the reliable audio baseline",
    "time_series_cv": "prevents optimistic CV / leakage on temporal data",
}


def recommend_strategies(
    profile: TaskProfile,
    *,
    small_data_threshold: int = 10_000,
    high_cardinality_threshold: int = 3,
    pseudo_label_ratio: float = 3.0,
) -> StrategyRecommendation:
    """Return an ordered, de-duplicated list of recommended strategies."""
    strategies: list[str] = []
    rationale: dict[str, str] = {}

    def add(name: str, why: str) -> None:
        if name not in strategies:
            strategies.append(name)
            rationale[name] = why

    modality = profile.modality

    # Temporal data first — it constrains CV and unlocks lag features.
    if modality == "time_series" or profile.has_time_column:
        add("time_series_cv", "temporal data must use forward-chaining CV, never random shuffle")
        add("lag_features", "generate lag/rolling features from the time index")

    if modality == "tabular" or modality == "multimodal":
        if profile.n_high_cardinality_features >= high_cardinality_threshold:
            add("target_encoding", f"{profile.n_high_cardinality_features} high-cardinality categorical features")
        if 0 < profile.n_features <= 100:
            add("feature_crossing", f"modest feature count ({profile.n_features}) supports crossing")

    if modality == "image":
        add("test_time_augmentation", "image modality benefits from TTA at inference")

    if modality == "text":
        add("tfidf_ngrams", "text modality: TF-IDF word+char n-grams with a linear model is a strong, fast baseline")

    if modality == "multimodal":
        add("multimodal_fusion", "multimodal data: encode each modality then fuse (concat embeddings or blend probabilities)")

    if modality == "audio":
        add("audio_spectrogram", "audio modality: convert waveforms to log-mel spectrograms and model as images")

    # Regression target shaping.
    if profile.task_type == "regression":
        if profile.metric == "rmsle" or profile.target_is_positive:
            add("log1p_target", "RMSLE / positive skewed target benefits from log1p")

    # Data-size driven techniques.
    if 0 < profile.train_size < small_data_threshold:
        add("multi_seed_ensemble", f"small train set ({profile.train_size}) -> average seeds to cut variance")
    if profile.test_size > profile.train_size * pseudo_label_ratio and profile.train_size > 0:
        add("pseudo_labeling", "test set much larger than train -> exploit unlabeled data")

    # Stacking when there is model diversity to exploit.
    if profile.n_model_families >= 3:
        add("oof_stacking", f"{profile.n_model_families} model families -> stack OOF predictions")

    expected = {name: _EXPECTED_GAINS[name] for name in strategies if name in _EXPECTED_GAINS}
    return StrategyRecommendation(strategies=strategies, rationale=rationale, expected_gains=expected)
