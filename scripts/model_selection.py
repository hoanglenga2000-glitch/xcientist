"""Data-profile-driven model strategy selection (MLEvolve-style).

Replaces the workstation's fixed-CatBoost default with a decision that adapts the
model family, hyperparameters, feature engineering, and ensemble strategy to the
task profile. It is deliberately pure and deterministic so it can be unit-tested
offline and consulted by the GPU trainer before a run.

This complements ``mlevolve_controller`` (which chooses the *search* mode:
Base/Stepwise/Diff and exploration stage). Here we choose *what to train*.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DataProfile:
    """Compact description of a competition dataset."""

    task_type: str  # "classification" | "regression"
    n_rows: int
    n_cols: int
    metric: str
    n_classes: Optional[int] = None
    # Fraction of feature columns that are high-cardinality categoricals.
    categorical_ratio: float = 0.0
    # True when features look like dense pixels / embeddings (e.g. digit-recognizer).
    is_pixel_like: bool = False


@dataclass
class ModelStrategy:
    """A concrete, trainer-consumable recommendation."""

    model_families: list[str]
    primary_model: str
    hyperparams: dict[str, Any]
    feature_engineering: list[str] = field(default_factory=list)
    ensemble_strategy: str = "single"  # single | weighted_blend | stacking
    exploration_mode: str = "explore"  # explore | exploit
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_families": list(self.model_families),
            "primary_model": self.primary_model,
            "hyperparams": dict(self.hyperparams),
            "feature_engineering": list(self.feature_engineering),
            "ensemble_strategy": self.ensemble_strategy,
            "exploration_mode": self.exploration_mode,
            "notes": list(self.notes),
        }


@dataclass
class TrainingPlan:
    """An *executable* plan: the recommendation resolved against what a trainer
    can actually run right now. Neural/CNN families are deferred (P2) with a safe
    gradient-boosting fallback, so wiring this into the trainer never makes a run
    inert or crash on an unavailable family.
    """

    executable_model: str            # family the trainer will actually run
    hyperparams: dict[str, Any]
    ensemble_models: list[str] = field(default_factory=list)  # feasible secondaries (P2)
    ensemble_strategy: str = "single"
    feature_engineering: list[str] = field(default_factory=list)
    multi_seed: bool = False
    seeds: list[int] = field(default_factory=list)
    deferred_model: Optional[str] = None   # e.g. "cnn"/"neural_net" recommended but not yet executable
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "executable_model": self.executable_model,
            "hyperparams": dict(self.hyperparams),
            "ensemble_models": list(self.ensemble_models),
            "ensemble_strategy": self.ensemble_strategy,
            "feature_engineering": list(self.feature_engineering),
            "multi_seed": self.multi_seed,
            "seeds": list(self.seeds),
            "deferred_model": self.deferred_model,
            "notes": list(self.notes),
        }


# Families the tabular GPU trainer can execute today (P1). CNN/neural are P2.
_EXECUTABLE_GBDT = ("catboost", "lightgbm", "xgboost")
_DEFAULT_SEEDS = [42, 123, 456, 789, 1024]


def resolve_training_plan(
    strategy: ModelStrategy,
    profile: DataProfile,
    *,
    available_models: Optional[list[str]] = None,
    neural_available: bool = False,
    small_data_rows: int = 5_000,  # == _SMALL_DATA_ROWS (defined below; literal avoids fwd-ref)
) -> TrainingPlan:
    """Resolve a ModelStrategy into an executable TrainingPlan.

    Rules:
      * If the primary is a runnable GBDT and available, use it directly.
      * If the primary is neural/CNN and ``neural_available`` (torch present) AND the
        task is classification, run it as ``cnn`` (unlocks the image modality).
      * Otherwise a neural/CNN or unavailable family is deferred with a safe GBDT
        fallback, recording the deferral in ``deferred_model`` + notes.
      * Secondary ensemble models are filtered to available GBDTs.
      * Small tabular data (< ``small_data_rows``) enables multi-seed averaging,
        which is the cheap, high-yield variance reducer for tiny train sets.
    """
    available = list(available_models) if available_models is not None else list(_EXECUTABLE_GBDT)
    notes = list(strategy.notes)

    primary = strategy.primary_model
    deferred: Optional[str] = None
    _NEURAL = ("neural_net", "cnn", "vit", "resnet18", "vit_tiny")

    if primary in _EXECUTABLE_GBDT and primary in available:
        executable = primary
        hyperparams = dict(strategy.hyperparams)
    elif primary in _NEURAL and neural_available and profile.task_type == "classification":
        # Execute the recommended neural net as a CNN (image modality unlocked).
        executable = "cnn"
        hyperparams = dict(strategy.hyperparams)
        notes.append("neural backend available -> running CNN branch")
    else:
        # Neural/CNN without a backend, or an unavailable family -> defer + GBDT fallback.
        if primary not in _EXECUTABLE_GBDT:
            deferred = primary
            notes.append(f"'{primary}' recommended but not executable here; falling back to gradient boosting")
        else:
            notes.append(f"'{primary}' unavailable in this environment; falling back")
        fallback = next(
            (m for m in strategy.model_families if m in _EXECUTABLE_GBDT and m in available),
            None,
        )
        if fallback is None:
            fallback = "catboost" if "catboost" in available else (available[0] if available else "catboost")
        executable = fallback
        hyperparams = _base_hyperparams(executable, profile)

    # Feasible secondary models for a later ensemble (P2), excluding the primary.
    ensemble_models = [
        m for m in strategy.model_families
        if m in _EXECUTABLE_GBDT and m in available and m != executable
    ]

    # Small-data variance control: multi-seed averaging.
    multi_seed = profile.n_rows < small_data_rows
    seeds = list(_DEFAULT_SEEDS) if multi_seed else []
    if multi_seed:
        notes.append(f"small data ({profile.n_rows} rows): multi-seed averaging over {len(seeds)} seeds")

    return TrainingPlan(
        executable_model=executable,
        hyperparams=hyperparams,
        ensemble_models=ensemble_models,
        ensemble_strategy=strategy.ensemble_strategy,
        feature_engineering=list(strategy.feature_engineering),
        multi_seed=multi_seed,
        seeds=seeds,
        deferred_model=deferred,
        notes=notes,
    )


# Size thresholds (rows) that switch the default family.
_SMALL_DATA_ROWS = 5_000
_LARGE_DATA_ROWS = 200_000
_HIGH_DIM_COLS = 500


def _base_hyperparams(model: str, profile: DataProfile) -> dict[str, Any]:
    """Sensible starting hyperparameters per family, scaled by data size."""
    if model == "catboost":
        depth = 6 if profile.n_rows < _LARGE_DATA_ROWS else 8
        return {"depth": depth, "learning_rate": 0.05, "iterations": 2000, "early_stopping_rounds": 100}
    if model == "lightgbm":
        leaves = 31 if profile.n_rows < _LARGE_DATA_ROWS else 127
        return {"num_leaves": leaves, "learning_rate": 0.05, "n_estimators": 3000, "early_stopping_rounds": 100}
    if model == "xgboost":
        depth = 6 if profile.n_rows < _LARGE_DATA_ROWS else 8
        return {"max_depth": depth, "learning_rate": 0.05, "n_estimators": 3000, "early_stopping_rounds": 100}
    return {}


def recommend_model_strategy(
    profile: DataProfile,
    *,
    best_score_so_far: Optional[float] = None,
    rounds_without_improvement: int = 0,
    memory_hits: Optional[list[dict[str, Any]]] = None,
) -> ModelStrategy:
    """Choose a model strategy for the given data profile.

    Heuristics:
      * pixel-like / very high dimensional -> recommend a neural net branch
        (gradient boosting is a weak default there; flagged in notes).
      * small tabular data -> CatBoost primary (robust, less overfit).
      * large tabular data -> LightGBM primary (fast) with XGBoost/CatBoost blend.
      * stagnation (no improvement for >=2 rounds) -> switch explore->exploit and
        move from single model to an ensemble.
    """
    notes: list[str] = []
    feature_engineering: list[str] = []

    # High-cardinality categoricals benefit from target encoding.
    if profile.categorical_ratio >= 0.3:
        feature_engineering.append("target_encoding")

    if profile.is_pixel_like or profile.n_cols >= _HIGH_DIM_COLS:
        notes.append(
            "high-dimensional/pixel-like input: gradient boosting is a weak baseline; "
            "prefer a CNN/MLP branch for competitive scores"
        )
        families = ["neural_net", "lightgbm"]
        primary = "neural_net"
        hyperparams = {"arch": "mlp", "hidden": [512, 256], "dropout": 0.2, "epochs": 20}
        if profile.is_pixel_like:
            hyperparams = {"arch": "cnn", "epochs": 15, "batch_size": 128}
            feature_engineering.append("normalize_pixels")
        return ModelStrategy(
            model_families=families,
            primary_model=primary,
            hyperparams=hyperparams,
            feature_engineering=feature_engineering,
            ensemble_strategy="single",
            exploration_mode="explore",
            notes=notes,
        )

    # Tabular families ordered by data size.
    if profile.n_rows < _SMALL_DATA_ROWS:
        families = ["catboost", "lightgbm"]
        primary = "catboost"
        notes.append("small data: CatBoost primary to limit overfitting")
    elif profile.n_rows >= _LARGE_DATA_ROWS:
        families = ["lightgbm", "xgboost", "catboost"]
        primary = "lightgbm"
        notes.append("large data: LightGBM primary for training speed")
    else:
        families = ["lightgbm", "catboost", "xgboost"]
        primary = "lightgbm"

    stagnant = rounds_without_improvement >= 2
    if stagnant:
        ensemble_strategy = "stacking"
        exploration_mode = "exploit"
        notes.append(f"stagnation ({rounds_without_improvement} rounds): escalate to stacking + exploit")
    elif best_score_so_far is not None:
        ensemble_strategy = "weighted_blend"
        exploration_mode = "exploit"
    else:
        ensemble_strategy = "single"
        exploration_mode = "explore"

    if memory_hits:
        notes.append(f"reused {len(memory_hits)} retrospective-memory strategy record(s)")

    return ModelStrategy(
        model_families=families,
        primary_model=primary,
        hyperparams=_base_hyperparams(primary, profile),
        feature_engineering=feature_engineering,
        ensemble_strategy=ensemble_strategy,
        exploration_mode=exploration_mode,
        notes=notes,
    )


# ── Modality detection (feeds DataProfile / the trainer's branch choice) ──

def detect_modality(
    column_names: list[str],
    n_rows: int = 0,
    *,
    avg_text_length: float = 0.0,
    has_time_column: bool = False,
) -> str:
    """Infer the dataset modality from column names and light stats.

    Returns one of: "image" | "text" | "time_series" | "tabular".
    Pure and deterministic so it can be unit-tested without loading data.
    """
    names = [str(c).lower() for c in column_names]
    n_cols = len(names)

    # Image: many pixel-like columns.
    pixel_cols = [c for c in names if c.startswith("pixel") or c.startswith("px")]
    if len(pixel_cols) >= 100 or (n_cols >= 100 and len(pixel_cols) / max(n_cols, 1) > 0.8):
        return "image"

    # Text: an explicit text/comment/body column, or long average strings.
    text_hints = ("text", "comment", "body", "tweet", "review", "content", "excerpt", "essay")
    if any(any(h in c for h in text_hints) for c in names):
        return "text"
    if avg_text_length >= 200:
        return "text"

    # Time series: an explicit date/time column.
    time_hints = ("date", "datetime", "timestamp", "time", "month", "day", "year")
    if has_time_column or any(c in time_hints or c.endswith("_date") for c in names):
        return "time_series"

    return "tabular"


def profile_from_frame_stats(
    *,
    task_type: str,
    column_names: list[str],
    n_rows: int,
    metric: str,
    n_classes: Optional[int] = None,
    categorical_ratio: float = 0.0,
    avg_text_length: float = 0.0,
    has_time_column: bool = False,
) -> DataProfile:
    """Build a DataProfile with modality-aware pixel flagging."""
    modality = detect_modality(
        column_names, n_rows, avg_text_length=avg_text_length, has_time_column=has_time_column
    )
    return DataProfile(
        task_type=task_type,
        n_rows=n_rows,
        n_cols=len(column_names),
        metric=metric,
        n_classes=n_classes,
        categorical_ratio=categorical_ratio,
        is_pixel_like=(modality == "image"),
    )
