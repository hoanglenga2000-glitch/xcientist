#!/usr/bin/env python3
"""Auto-detect Kaggle competition metadata from raw data.

Pure, dependency-light inference helpers that turn a raw train dataframe (plus an
optional sample_submission and test dataframe) into a COMPETITIONS-registry entry
compatible with scripts/gpu_batch_trainer_v1.py.

Every heuristic is conservative and overridable; the goal is a *safe default* that
a human or the trainer can correct, not a black box. No network, no GPU, no Kaggle
calls here -- that lives in onboard_new_competition.py.

Registry entry shape (matches gpu_batch_trainer_v1.py COMPETITIONS):
    {
        "type": "classification" | "regression",
        "metric": "accuracy"|"roc_auc"|"normalized_gini"|"rmse"|"rmsle"|"mae",
        "target": str,
        "id_col": str | None,
        "drop_cols": list[str],
        "higher_is_better": bool,
        "bronze": float | None,
    }
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

try:  # pandas is a control-plane dependency; import lazily-friendly for tooling
    import pandas as pd
except Exception:  # pragma: no cover - only hit when pandas missing
    pd = None  # type: ignore

# Common Kaggle id-column names, checked case-insensitively.
_ID_NAME_HINTS = ("id", "index", "row_id", "rowid", "passengerid", "key", "uid")
# Column names that are almost always the TARGET, never an id -- guards the
# "first column is unique" fallback from grabbing a target on small/edge data.
_TARGET_NAME_HINTS = ("label", "target", "class", "y", "outcome", "prediction", "pred")
# Metrics where a higher score is better.
_HIGHER_IS_BETTER = {"accuracy", "roc_auc", "normalized_gini", "f1", "map", "r2"}
# Column-name tokens that hint at a right-skewed positive regression target
# (RMSLE is the usual Kaggle metric for these).
_RMSLE_TARGET_HINTS = ("price", "sales", "count", "demand", "revenue", "amount", "sepsis")


@dataclass
class CompetitionConfig:
    """A detected registry entry plus the evidence behind each decision."""

    type: str
    metric: str
    target: str
    id_col: Optional[str]
    drop_cols: list[str]
    higher_is_better: bool
    bronze: Optional[float] = None
    n_classes: Optional[int] = None
    rationale: dict[str, str] = field(default_factory=dict)

    def to_registry_entry(self) -> dict[str, Any]:
        """Return only the keys the trainer's COMPETITIONS dict expects."""
        return {
            "type": self.type,
            "metric": self.metric,
            "target": self.target,
            "id_col": self.id_col,
            "drop_cols": list(self.drop_cols),
            "higher_is_better": self.higher_is_better,
            "bronze": self.bronze,
        }


def detect_id_column(df: "pd.DataFrame", sample_submission: "pd.DataFrame | None" = None) -> Optional[str]:
    """Find the identifier column.

    Priority:
      1. The first column of sample_submission (Kaggle's authoritative id column).
      2. A column whose name matches a known id hint AND is unique per row.
      3. The first column if it is unique and not obviously the target.
    Returns None when nothing looks like an id (e.g. pixel data with no id).
    """
    if sample_submission is not None and len(sample_submission.columns) > 0:
        candidate = str(sample_submission.columns[0])
        if candidate in df.columns:
            return candidate
        # sample_submission id may not exist in train (that's fine) -- still authoritative.
        return candidate

    n = len(df)
    for col in df.columns:
        if str(col).lower() in _ID_NAME_HINTS and df[col].nunique(dropna=False) == n:
            return str(col)

    if len(df.columns) > 0:
        first = df.columns[0]
        # Never treat a target-like name as an id, even if it is unique on small data.
        if str(first).lower() not in _TARGET_NAME_HINTS and df[first].nunique(dropna=False) == n and n > 1:
            return str(first)
    return None


def detect_target_column(
    df: "pd.DataFrame",
    sample_submission: "pd.DataFrame | None" = None,
    id_col: Optional[str] = None,
) -> Optional[str]:
    """Infer the target column.

    Priority:
      1. sample_submission's prediction column name, if present in train.
      2. The last column of train that is neither the id nor an obvious feature id.
    """
    if sample_submission is not None and len(sample_submission.columns) >= 2:
        for col in sample_submission.columns[1:]:
            if str(col) in df.columns:
                return str(col)
    # Fall back to the last non-id column.
    candidates = [c for c in df.columns if c != id_col]
    if not candidates:
        return None
    return str(candidates[-1])


def detect_task_type(
    df: "pd.DataFrame",
    target_col: str,
    classification_max_classes: int = 50,
    max_unique_ratio: float = 0.5,
) -> str:
    """Classification vs regression.

    Rules:
      - Non-numeric target -> classification.
      - Numeric integer target with few unique values (<= threshold) AND enough
        repetition (unique/rows ratio below `max_unique_ratio`) -> classification.
        The ratio guard rejects the degenerate "every row unique" case, which is
        regression (e.g. an integer-valued price), not a many-class problem.
      - Otherwise regression.
    """
    series = df[target_col].dropna()
    if series.empty:
        return "regression"
    is_numeric = pd.api.types.is_numeric_dtype(series)
    if not is_numeric:
        return "classification"
    n = len(series)
    nunique = series.nunique()
    looks_integer = bool((series == series.round()).all())
    unique_ratio = nunique / n if n else 1.0
    if nunique <= classification_max_classes and looks_integer and unique_ratio <= max_unique_ratio:
        return "classification"
    return "regression"


def detect_metric(df: "pd.DataFrame", target_col: str, task_type: str) -> str:
    """Pick a sensible default metric.

    classification:
      - binary + imbalanced (minority < 35%)      -> roc_auc
      - binary + balanced                         -> accuracy
      - multiclass                                -> accuracy
    regression:
      - all-positive target whose name hints at a
        count/price/sales quantity                -> rmsle
      - otherwise                                 -> rmse
    """
    series = df[target_col].dropna()
    if task_type == "classification":
        nunique = series.nunique()
        if nunique == 2:
            counts = series.value_counts(normalize=True)
            minority = float(counts.min()) if len(counts) else 0.5
            return "roc_auc" if minority < 0.35 else "accuracy"
        return "accuracy"
    # regression
    all_positive = bool((series >= 0).all()) and pd.api.types.is_numeric_dtype(series)
    name_hint = any(tok in str(target_col).lower() for tok in _RMSLE_TARGET_HINTS)
    if all_positive and name_hint:
        return "rmsle"
    return "rmse"


def detect_drop_columns(
    df: "pd.DataFrame",
    target_col: str,
    id_col: Optional[str],
    high_cardinality_ratio: float = 0.9,
) -> list[str]:
    """Identify leakage-prone / non-informative columns to drop.

    Drops:
      - the id column (never a feature),
      - free-text / near-unique string columns (name, ticket, etc.):
        object dtype whose distinct-ratio exceeds high_cardinality_ratio,
      - columns that are entirely null.
    The target is never dropped.
    """
    drop: list[str] = []
    n = max(len(df), 1)
    if id_col and id_col in df.columns:
        drop.append(id_col)
    for col in df.columns:
        if col == target_col or col == id_col:
            continue
        series = df[col]
        if series.isna().all():
            drop.append(str(col))
            continue
        if (series.dtype == object or pd.api.types.is_string_dtype(series)):
            ratio = series.nunique(dropna=True) / n
            if ratio >= high_cardinality_ratio:
                drop.append(str(col))
    # De-dup while preserving order.
    seen: set[str] = set()
    ordered: list[str] = []
    for col in drop:
        if col not in seen:
            seen.add(col)
            ordered.append(col)
    return ordered


def detect_competition_config(
    train_df: "pd.DataFrame",
    sample_submission: "pd.DataFrame | None" = None,
    *,
    target_override: Optional[str] = None,
    id_override: Optional[str] = None,
    bronze: Optional[float] = None,
) -> CompetitionConfig:
    """End-to-end detection producing a COMPETITIONS-compatible config."""
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas is required for competition detection")

    id_col = id_override if id_override is not None else detect_id_column(train_df, sample_submission)
    target_col = target_override or detect_target_column(train_df, sample_submission, id_col)
    if target_col is None or target_col not in train_df.columns:
        raise ValueError(
            f"Could not determine a target column present in train data "
            f"(detected={target_col!r}, columns={list(train_df.columns)[:10]}...)"
        )

    task_type = detect_task_type(train_df, target_col)
    metric = detect_metric(train_df, target_col, task_type)
    drop_cols = detect_drop_columns(train_df, target_col, id_col)
    higher = metric in _HIGHER_IS_BETTER
    n_classes = int(train_df[target_col].dropna().nunique()) if task_type == "classification" else None

    return CompetitionConfig(
        type=task_type,
        metric=metric,
        target=target_col,
        id_col=id_col,
        drop_cols=drop_cols,
        higher_is_better=higher,
        bronze=bronze,
        n_classes=n_classes,
        rationale={
            "id_col": f"from {'sample_submission' if sample_submission is not None else 'heuristic'}",
            "target": "override" if target_override else "detected",
            "task_type": f"{task_type} (nunique={train_df[target_col].dropna().nunique()})",
            "metric": f"{metric} (higher_is_better={higher})",
            "drop_cols": f"{len(drop_cols)} column(s) flagged as id/high-cardinality/empty",
        },
    )
