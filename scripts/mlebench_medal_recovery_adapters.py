#!/usr/bin/env python3
"""High-value medal-recovery adapters for already-scored MLE-Bench Lite tasks."""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata as importlib_metadata
import io
import json
import math
import os
import platform
import random
import re
import shutil
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from research_os.mlebench_phase_a import compute_metric, resolve_competition

try:
    import mlebench_wave2_adapters as wave2
    import run_mlebench_lite_wave0 as wave0
except ModuleNotFoundError:
    from scripts import mlebench_wave2_adapters as wave2
    from scripts import run_mlebench_lite_wave0 as wave0


MAY2022_F27_ALPHABET = "ABCDEFGHIJKLMNOPQRST"
MAY2022_F27_WIDTH = 10
MAY2022_AGGREGATE_PROMOTION_AUC = 0.9985
MAY2022_FOLD_PROMOTION_AUC = 0.99818
MAY2022_CONFIRMATION_SEED_AUC = 0.9983
MAY2022_CONFIRMATION_SEEDS_REQUIRED = 3
MAY2022_INNER_VALIDATION_FRACTION = 0.20
MAY2022_PUBLIC_CACHE_SCHEMA = "evomind.mlebench.may2022_public_feature_cache.v1"
MAY2022_PUBLIC_CACHE_FEATURE_NAMES_SCHEMA = (
    "evomind.mlebench.may2022_public_feature_names.v1"
)
MAY2022_PUBLIC_CACHE_DATA_ARTIFACTS = (
    "train_features.npy",
    "test_features.npy",
    "train_id.npy",
    "test_id.npy",
    "target.npy",
    "feature_names.json",
    "may2022_fold_assignments.npz",
    "may2022_nested_fold_indices.npz",
    "feature_diagnostics.json",
)
MIN_SIIM_ABLATION_FOLDS = 3
SIIM_FORMAL_OUTER_FOLDS = 5
SIIM_FORMAL_INNER_FOLDS = 3
SIIM_LEAKAGE_GROUP_POLICY = "patient_exact_file_decoded_pixel_v2"
SIIM_PERCEPTUAL_EDGE_POLICY = "audit_only_no_cross_patient_union_v1"
SIIM_PREPROCESSING_PROFILES = (
    "raw_multiview_v1",
    "border_multiview_v1",
    "color_multiview_v1",
    "hair_multiview_v1",
    "robust_multiview_v1",
)
SIIM_PREPROCESSING_PROFILE_STEPS = {
    "raw_multiview_v1": [],
    "border_multiview_v1": ["dark_border_crop"],
    "color_multiview_v1": ["dark_border_crop", "shades_of_gray_color_constancy"],
    "hair_multiview_v1": [
        "dark_border_crop",
        "shades_of_gray_color_constancy",
        "dark_hair_median_repair",
    ],
    "robust_multiview_v1": [
        "dark_border_crop",
        "shades_of_gray_color_constancy",
        "dark_hair_median_repair",
        "skin_deviation_lesion_focus_crop",
    ],
}


class SiimPauseRequested(RuntimeError):
    """Epoch-boundary pause with an atomic, same-Run resume checkpoint."""

    def __init__(self, reason: str, checkpoint_path: Path) -> None:
        super().__init__(reason)
        self.reason = str(reason)
        self.checkpoint_path = Path(checkpoint_path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def siim_epoch_pause_reason(
    *,
    control_file: Path | None,
    budget_state_path: Path,
    runtime_budget_seconds: float,
) -> str | None:
    """Return a fail-closed epoch-boundary pause reason, if any."""

    budget_seconds = float(runtime_budget_seconds)
    if budget_seconds > 0:
        try:
            state = json.loads(budget_state_path.read_text(encoding="utf-8"))
            started_unix = float(state["started_unix"])
            configured = float(state["runtime_budget_seconds"])
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            return f"runtime_budget_state_invalid:{type(exc).__name__}"
        if abs(configured - budget_seconds) > 1e-6:
            return "runtime_budget_changed_after_start"
        if time.time() - started_unix >= budget_seconds:
            return "runtime_budget_exhausted"

    if control_file is None:
        return None
    try:
        payload = json.loads(Path(control_file).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return f"resource_control_invalid:{type(exc).__name__}"
    status = str(payload.get("status", "continue")).strip().lower()
    if status in {"continue", "clear", "go", "running"}:
        return None
    if status in {"pause", "pause_requested", "pause_after_epoch", "hold"}:
        reason = str(payload.get("reason") or "resource_guard_requested")
        return f"resource_guard:{reason}"
    return f"resource_control_unknown_status:{status or 'empty'}"


def _encode_may2022_f27(values: pd.Series) -> np.ndarray:
    """Encode the fixed-width ``f_27`` strings without learning from labels."""

    strings = values.fillna("").astype(str)
    lengths = strings.str.len().to_numpy()
    if not np.all(lengths == MAY2022_F27_WIDTH):
        observed = sorted(int(value) for value in np.unique(lengths))
        raise ValueError(
            f"May 2022 f_27 must contain {MAY2022_F27_WIDTH} characters; "
            f"observed lengths={observed}"
        )
    joined = "".join(strings.tolist())
    try:
        raw = np.frombuffer(joined.encode("ascii"), dtype=np.uint8).astype(np.int16)
    except UnicodeEncodeError as exc:
        raise ValueError("May 2022 f_27 contains non-ASCII characters") from exc
    encoded = raw.reshape(len(strings), MAY2022_F27_WIDTH) - ord("A")
    if encoded.size and (encoded.min() < 0 or encoded.max() >= len(MAY2022_F27_ALPHABET)):
        observed = "".join(sorted(set(joined)))
        raise ValueError(f"May 2022 f_27 contains unexpected alphabet={observed!r}")
    return encoded.astype(np.int8, copy=False)


def build_may2022_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build deterministic numeric features for TPS May 2022.

    The public data generator hides substantial signal in the ten positions of
    ``f_27``.  A raw high-cardinality string is nearly useless to a baseline
    tree model, so every position is represented both as an ordinal code and a
    one-hot categorical indicator.  Per-letter counts, repetition statistics,
    and the three published geometric interactions are also materialized.
    No target-derived aggregate is used, keeping the transform fold-safe.
    """

    required = {
        "f_00", "f_01", "f_02", "f_05", "f_21", "f_22", "f_26", "f_27"
    }
    for name, frame in (("train", train), ("test", test)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"May 2022 {name} data is missing required columns: {missing}")

    numeric_columns = [
        column
        for column in test.columns
        if column.startswith("f_") and column != "f_27"
    ]
    if len(numeric_columns) != 30:
        raise ValueError(
            f"May 2022 expected 30 numeric f_* columns besides f_27; "
            f"found {len(numeric_columns)}"
        )

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        numeric = frame[numeric_columns].apply(pd.to_numeric, errors="raise").astype(np.float32)
        encoded = _encode_may2022_f27(frame["f_27"])
        engineered: dict[str, np.ndarray] = {}

        counts: list[np.ndarray] = []
        for letter_index, letter in enumerate(MAY2022_F27_ALPHABET):
            count = (encoded == letter_index).sum(axis=1).astype(np.int8)
            counts.append(count)
            engineered[f"f27_count_{letter}"] = count
        count_matrix = np.column_stack(counts)
        engineered["f27_unique_count"] = (count_matrix > 0).sum(axis=1).astype(np.int8)
        engineered["f27_repeat_count"] = (
            MAY2022_F27_WIDTH - engineered["f27_unique_count"]
        ).astype(np.int8)
        engineered["f27_adjacent_equal_count"] = (
            encoded[:, 1:] == encoded[:, :-1]
        ).sum(axis=1).astype(np.int8)
        engineered["f27_ascending_steps"] = (
            encoded[:, 1:] > encoded[:, :-1]
        ).sum(axis=1).astype(np.int8)
        engineered["f27_descending_steps"] = (
            encoded[:, 1:] < encoded[:, :-1]
        ).sum(axis=1).astype(np.int8)
        engineered["f27_first_last_equal"] = (encoded[:, 0] == encoded[:, -1]).astype(np.int8)
        engineered["f27_first_last_delta"] = (
            encoded[:, -1].astype(np.int16) - encoded[:, 0].astype(np.int16)
        )
        engineered["f27_code_mean"] = encoded.mean(axis=1).astype(np.float32)
        engineered["f27_code_std"] = encoded.std(axis=1).astype(np.float32)
        engineered["f27_weighted_checksum"] = (
            encoded.astype(np.int32) * np.arange(1, MAY2022_F27_WIDTH + 1, dtype=np.int32)
        ).sum(axis=1)

        for position in range(MAY2022_F27_WIDTH):
            engineered[f"f27_pos_{position}_code"] = encoded[:, position]
            for letter_index, letter in enumerate(MAY2022_F27_ALPHABET):
                engineered[f"f27_pos_{position}_{letter}"] = (
                    encoded[:, position] == letter_index
                ).astype(np.int8)

        sum_02_21 = numeric["f_02"].to_numpy() + numeric["f_21"].to_numpy()
        sum_05_22 = numeric["f_05"].to_numpy() + numeric["f_22"].to_numpy()
        sum_00_01_26 = (
            numeric["f_00"].to_numpy()
            + numeric["f_01"].to_numpy()
            + numeric["f_26"].to_numpy()
        )
        engineered["sum_f02_f21"] = sum_02_21
        engineered["sum_f05_f22"] = sum_05_22
        engineered["sum_f00_f01_f26"] = sum_00_01_26
        engineered["interaction_f02_f21"] = (
            (sum_02_21 > 5.2).astype(np.int8) - (sum_02_21 < -5.3).astype(np.int8)
        )
        engineered["interaction_f05_f22"] = (
            (sum_05_22 > 5.1).astype(np.int8) - (sum_05_22 < -5.4).astype(np.int8)
        )
        engineered["interaction_f00_f01_f26"] = (
            (sum_00_01_26 > 5.0).astype(np.int8)
            - (sum_00_01_26 < -5.0).astype(np.int8)
        )

        extra = pd.DataFrame(engineered, index=frame.index)
        result = pd.concat([numeric, extra], axis=1)
        if result.columns.duplicated().any():
            raise RuntimeError("May 2022 feature builder produced duplicate columns")
        if any(not pd.api.types.is_numeric_dtype(result[column]) for column in result.columns):
            raise RuntimeError("May 2022 feature builder produced a nonnumeric column")
        if not np.isfinite(result.to_numpy(dtype=np.float32, copy=False)).all():
            raise RuntimeError("May 2022 feature builder produced nonfinite values")
        return result.astype(np.float32, copy=False)

    train_features = transform(train)
    test_features = transform(test)
    if list(train_features.columns) != list(test_features.columns):
        raise RuntimeError("May 2022 train/test engineered schemas differ")
    diagnostics = {
        "feature_count": len(train_features.columns),
        "base_numeric_feature_count": len(numeric_columns),
        "f27_position_code_count": MAY2022_F27_WIDTH,
        "f27_position_one_hot_count": MAY2022_F27_WIDTH * len(MAY2022_F27_ALPHABET),
        "f27_letter_count": len(MAY2022_F27_ALPHABET),
        "published_interaction_count": 3,
        "target_derived_features": 0,
        "f27_alphabet": MAY2022_F27_ALPHABET,
        "f27_width": MAY2022_F27_WIDTH,
        "ordered_feature_names": [str(column) for column in train_features.columns],
        "ordered_feature_dtypes": [str(dtype) for dtype in train_features.dtypes],
    }
    schema_payload = {
        "columns": diagnostics["ordered_feature_names"],
        "dtypes": diagnostics["ordered_feature_dtypes"],
    }
    diagnostics["feature_schema_sha256"] = hashlib.sha256(
        json.dumps(schema_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    diagnostics["id_feature_count"] = sum(
        str(column).lower() == "id" for column in train_features.columns
    )
    diagnostics["target_feature_count"] = sum(
        str(column).lower() == "target" for column in train_features.columns
    )
    return train_features, test_features, diagnostics


def validate_may2022_full_data_contract(
    *, train_rows_available: int, train_rows_used: int
) -> dict[str, Any]:
    """Make full public-training-row use a hard recovery invariant."""

    available = int(train_rows_available)
    used = int(train_rows_used)
    if available <= 0 or used <= 0:
        raise RuntimeError("May 2022 full-data contract requires positive row counts")
    if used != available:
        raise RuntimeError(
            "May 2022 medal mode requires every public training row: "
            f"used={used} available={available}"
        )
    return {
        "train_rows_available": available,
        "train_rows_used": used,
        "full_training_data": True,
        "medal_mode_full_data_required": True,
    }


def _may2022_index_sha256(indices: np.ndarray | Sequence[int]) -> str:
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    return hashlib.sha256(values.tobytes()).hexdigest()


def freeze_may2022_refit_budget(
    *,
    selected_iteration: int,
    requested_budget: int,
    zero_based_iteration: bool,
) -> dict[str, Any]:
    """Convert an inner-selection checkpoint into a fixed outer-refit budget."""

    requested = int(requested_budget)
    if requested < 1:
        raise RuntimeError("May 2022 requested iteration budget must be positive")
    selected = int(selected_iteration)
    if selected < -1:
        raise RuntimeError("May 2022 selected iteration must be -1 or non-negative")
    if not zero_based_iteration and selected == 0:
        raise RuntimeError("May 2022 one-based selected iteration must be positive or -1")
    fallback = selected == -1
    if fallback:
        frozen = requested
    else:
        frozen = selected + 1 if zero_based_iteration else selected
        frozen = min(max(1, frozen), requested)
    return {
        "selected_iteration": selected,
        "selected_iteration_zero_based": bool(zero_based_iteration),
        "requested_budget": requested,
        "frozen_refit_budget": int(frozen),
        "fallback_to_requested_budget": fallback,
    }


def build_may2022_nested_fold_plans(
    target: np.ndarray | pd.Series,
    outer_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    seed: int,
    inner_validation_fraction: float = MAY2022_INNER_VALIDATION_FRACTION,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Build deterministic inner splits wholly contained in each outer-train fold."""

    from sklearn.model_selection import StratifiedShuffleSplit

    labels = np.asarray(target, dtype=np.int8).reshape(-1)
    row_count = len(labels)
    fraction = float(inner_validation_fraction)
    if row_count < 4 or set(np.unique(labels).tolist()) != {0, 1}:
        raise RuntimeError("May 2022 nested selection requires a non-empty binary target")
    if not 0.05 <= fraction <= 0.50:
        raise RuntimeError("May 2022 inner validation fraction must be between 0.05 and 0.50")
    if len(outer_splits) < 2:
        raise RuntimeError("May 2022 nested selection requires at least two outer folds")

    all_rows = np.arange(row_count, dtype=np.int64)
    outer_coverage = np.zeros(row_count, dtype=np.int16)
    plans: list[dict[str, Any]] = []
    for outer_fold, (raw_outer_train, raw_outer_valid) in enumerate(outer_splits):
        outer_train = np.sort(np.asarray(raw_outer_train, dtype=np.int64).reshape(-1))
        outer_valid = np.sort(np.asarray(raw_outer_valid, dtype=np.int64).reshape(-1))
        for name, values in (("outer train", outer_train), ("outer validation", outer_valid)):
            if (
                not len(values)
                or len(np.unique(values)) != len(values)
                or values.min() < 0
                or values.max() >= row_count
            ):
                raise RuntimeError(f"May 2022 {name} indices are invalid")
        if np.intersect1d(outer_train, outer_valid).size:
            raise RuntimeError("May 2022 outer train and validation rows overlap")
        if not np.array_equal(np.sort(np.concatenate((outer_train, outer_valid))), all_rows):
            raise RuntimeError("May 2022 outer fold does not partition every training row")
        if set(labels[outer_train].tolist()) != {0, 1} or set(labels[outer_valid].tolist()) != {0, 1}:
            raise RuntimeError("May 2022 outer fold does not contain both classes")
        outer_coverage[outer_valid] += 1

        inner_seed = int(seed + (outer_fold + 1) * 10_007)
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=fraction,
            random_state=inner_seed,
        )
        inner_train_local, inner_valid_local = next(
            splitter.split(np.zeros(len(outer_train), dtype=np.int8), labels[outer_train])
        )
        inner_train = np.sort(outer_train[np.asarray(inner_train_local, dtype=np.int64)])
        inner_valid = np.sort(outer_train[np.asarray(inner_valid_local, dtype=np.int64)])
        if np.intersect1d(inner_train, inner_valid).size:
            raise RuntimeError("May 2022 inner train and validation rows overlap")
        if np.intersect1d(inner_train, outer_valid).size or np.intersect1d(
            inner_valid, outer_valid
        ).size:
            raise RuntimeError("May 2022 outer validation rows entered inner model selection")
        if not np.array_equal(
            np.sort(np.concatenate((inner_train, inner_valid))), outer_train
        ):
            raise RuntimeError("May 2022 inner split does not partition outer-train rows")
        if set(labels[inner_train].tolist()) != {0, 1} or set(labels[inner_valid].tolist()) != {
            0,
            1,
        }:
            raise RuntimeError("May 2022 inner split does not contain both classes")

        plans.append(
            {
                "outer_fold": outer_fold,
                "outer_train_index": outer_train,
                "outer_valid_index": outer_valid,
                "inner_train_index": inner_train,
                "inner_valid_index": inner_valid,
                "evidence": {
                    "outer_fold": outer_fold,
                    "inner_split_seed": inner_seed,
                    "inner_validation_fraction": fraction,
                    "outer_train_rows": int(len(outer_train)),
                    "outer_valid_rows": int(len(outer_valid)),
                    "inner_train_rows": int(len(inner_train)),
                    "inner_valid_rows": int(len(inner_valid)),
                    "outer_train_index_sha256": _may2022_index_sha256(outer_train),
                    "outer_valid_index_sha256": _may2022_index_sha256(outer_valid),
                    "inner_train_index_sha256": _may2022_index_sha256(inner_train),
                    "inner_valid_index_sha256": _may2022_index_sha256(inner_valid),
                    "iteration_budget_selection_label_scope": "inner_validation_only",
                    "outer_validation_labels_used_for_selection": False,
                },
            }
        )
    if not np.all(outer_coverage == 1):
        raise RuntimeError("May 2022 outer folds must cover every row exactly once")
    return plans, outer_coverage


def _may2022_atomic_json(path: Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _may2022_atomic_npy(path: Path, values: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _may2022_atomic_npz(path: Path, **values: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **values)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _may2022_confined_path(path: Path, root: Path, *, label: str) -> Path:
    resolved_root = Path(root).expanduser().resolve()
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(f"May 2022 {label} escaped the allowed root: {resolved}") from exc
    return resolved


def _may2022_file_record(path: Path, *, relative_path: str | None = None) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise RuntimeError(f"May 2022 cache file is missing, not regular, or a symlink: {resolved}")
    return {
        "path": str(resolved),
        "relative_path": relative_path or resolved.name,
        "bytes": resolved.stat().st_size,
        "sha256": _may2022_sha256(resolved),
    }


def _may2022_public_input_paths(data_root: Path) -> dict[str, Path]:
    resolved = resolve_competition(
        "tabular-playground-series-may-2022",
        Path(data_root),
        require_private=False,
    )
    public_root = Path(resolved.public_dir).resolve()
    inputs = {
        "train.csv": (public_root / "train.csv").resolve(),
        "test.csv": (public_root / "test.csv").resolve(),
        "sample_submission.csv": Path(resolved.sample_submission_path).resolve(),
    }
    for name, path in inputs.items():
        try:
            path.relative_to(public_root)
        except ValueError as exc:
            raise RuntimeError(f"May 2022 public input escaped prepared/public: {name}") from exc
        if "private" in {part.lower() for part in path.parts}:
            raise RuntimeError(f"May 2022 public cache rejected a private path: {name}")
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"May 2022 public input is missing or unsafe: {name}")
    return inputs


def build_may2022_public_feature_cache(
    *,
    data_root: Path,
    output_dir: Path,
    allowed_root: Path,
    seed: int,
    folds: int,
    source_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    """Materialize the full public-only May feature/fold cache on a CPU node.

    ``cache_manifest.json`` is written last and acts as the completion marker.
    The three public CSVs are hashed before and after construction so a mutable
    input can never produce an apparently valid cache.
    """

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("May 2022 public cache requires CUDA_VISIBLE_DEVICES to be empty")
    requested_folds = int(folds)
    if requested_folds < 2:
        raise RuntimeError("May 2022 public cache requires at least two folds")
    if int(seed) < 0:
        raise RuntimeError("May 2022 public cache seed must be non-negative")

    root = Path(allowed_root).expanduser().resolve()
    data_root = _may2022_confined_path(Path(data_root), root, label="data root")
    cache_root = _may2022_confined_path(Path(output_dir), root, label="cache output")
    if cache_root == root:
        raise RuntimeError("May 2022 cache output must be below the allowed root")
    cache_root.mkdir(parents=True, exist_ok=True)
    if cache_root.is_symlink():
        raise RuntimeError("May 2022 cache output cannot be a symlink")
    manifest_path = cache_root / "cache_manifest.json"
    if manifest_path.exists():
        raise RuntimeError("May 2022 cache completion marker already exists")

    inputs = _may2022_public_input_paths(data_root)
    input_before = {
        name: _may2022_file_record(path, relative_path=f"prepared/public/{name}")
        for name, path in inputs.items()
    }
    started = time.perf_counter()
    train = pd.read_csv(inputs["train.csv"])
    test = pd.read_csv(inputs["test.csv"])
    sample = pd.read_csv(inputs["sample_submission.csv"])
    if "id" not in train or "id" not in test or "target" not in train:
        raise RuntimeError("May 2022 public cache input schema is incomplete")
    if list(sample.columns) != ["id", "target"]:
        raise RuntimeError("May 2022 sample submission schema must be exactly id,target")
    train_id = train["id"].to_numpy(dtype=np.int64, copy=True)
    test_id = test["id"].to_numpy(dtype=np.int64, copy=True)
    target = train["target"].to_numpy(dtype=np.int8, copy=True)
    if len(np.unique(train_id)) != len(train_id) or len(np.unique(test_id)) != len(test_id):
        raise RuntimeError("May 2022 public cache found duplicate IDs")
    if set(np.unique(target).tolist()) != {0, 1}:
        raise RuntimeError("May 2022 public cache target is not binary")
    if set(sample["id"].astype(str)) != set(pd.Series(test_id).astype(str)):
        raise RuntimeError("May 2022 sample/test ID sets differ")

    train_features, test_features, feature_diagnostics = build_may2022_features(train, test)
    train_matrix = np.ascontiguousarray(
        train_features.to_numpy(dtype=np.float32, copy=False)
    )
    test_matrix = np.ascontiguousarray(test_features.to_numpy(dtype=np.float32, copy=False))
    fold_count = min(requested_folds, int(np.bincount(target).min()))
    if fold_count != requested_folds:
        raise RuntimeError("May 2022 public cache cannot satisfy the requested fold count")

    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=int(seed))
    splits = list(splitter.split(train_matrix, target))
    nested_plans, outer_coverage = build_may2022_nested_fold_plans(
        target,
        splits,
        seed=int(seed),
    )
    fold_assignment = np.full(len(target), -1, dtype=np.int8)
    nested_payload: dict[str, np.ndarray] = {}
    nested_evidence: list[dict[str, Any]] = []
    for plan in nested_plans:
        fold_index = int(plan["outer_fold"])
        fold_assignment[plan["outer_valid_index"]] = fold_index
        nested_evidence.append(dict(plan["evidence"]))
        for name in (
            "outer_train_index",
            "outer_valid_index",
            "inner_train_index",
            "inner_valid_index",
        ):
            nested_payload[f"fold_{fold_index:02d}_{name}"] = np.asarray(
                plan[name], dtype=np.int64
            )
    if np.any(fold_assignment < 0) or not np.all(outer_coverage == 1):
        raise RuntimeError("May 2022 public cache fold coverage is incomplete")

    _may2022_atomic_npy(cache_root / "train_features.npy", train_matrix)
    _may2022_atomic_npy(cache_root / "test_features.npy", test_matrix)
    _may2022_atomic_npy(cache_root / "train_id.npy", train_id)
    _may2022_atomic_npy(cache_root / "test_id.npy", test_id)
    _may2022_atomic_npy(cache_root / "target.npy", target)
    feature_names_payload = {
        "schema": MAY2022_PUBLIC_CACHE_FEATURE_NAMES_SCHEMA,
        "ordered_feature_names": list(feature_diagnostics["ordered_feature_names"]),
        "ordered_feature_dtypes": list(feature_diagnostics["ordered_feature_dtypes"]),
        "feature_schema_sha256": feature_diagnostics["feature_schema_sha256"],
    }
    _may2022_atomic_json(cache_root / "feature_names.json", feature_names_payload)
    _may2022_atomic_npz(
        cache_root / "may2022_fold_assignments.npz",
        id=train_id,
        target=target,
        fold=fold_assignment,
        seed=np.asarray([int(seed)], dtype=np.int64),
    )
    _may2022_atomic_npz(
        cache_root / "may2022_nested_fold_indices.npz",
        **nested_payload,
    )
    elapsed = time.perf_counter() - started
    diagnostics_payload = {
        "schema": "evomind.mlebench.may2022_public_feature_diagnostics.v1",
        "feature_engineering": feature_diagnostics,
        "train_rows": int(len(train_matrix)),
        "test_rows": int(len(test_matrix)),
        "feature_count": int(train_matrix.shape[1]),
        "seed": int(seed),
        "folds": fold_count,
        "nested_fold_evidence": nested_evidence,
        "build_seconds": elapsed,
        "visibility_mode": "PUBLIC_ONLY",
        "target_derived_features": 0,
        "private_labels_used": False,
        "gpu_used": False,
        "cuda_visible_devices": "",
    }
    _may2022_atomic_json(cache_root / "feature_diagnostics.json", diagnostics_payload)

    input_after = {
        name: _may2022_file_record(path, relative_path=f"prepared/public/{name}")
        for name, path in inputs.items()
    }
    for name in inputs:
        if (
            input_before[name]["bytes"] != input_after[name]["bytes"]
            or input_before[name]["sha256"] != input_after[name]["sha256"]
        ):
            raise RuntimeError(f"May 2022 public input changed while building cache: {name}")

    resolved_sources = {Path(__file__).resolve(), *(Path(path).resolve() for path in source_paths)}
    source_records = []
    for path in sorted(resolved_sources, key=str):
        record = _may2022_file_record(path)
        record["role"] = "feature_builder" if path == Path(__file__).resolve() else "support"
        source_records.append(record)
    artifact_records = [
        _may2022_file_record(cache_root / name, relative_path=name)
        for name in MAY2022_PUBLIC_CACHE_DATA_ARTIFACTS
    ]
    manifest = {
        "schema": MAY2022_PUBLIC_CACHE_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "completed",
        "competition_id": "tabular-playground-series-may-2022",
        "visibility_mode": "PUBLIC_ONLY",
        "seed": int(seed),
        "folds": fold_count,
        "train_rows": int(len(train_matrix)),
        "test_rows": int(len(test_matrix)),
        "feature_count": int(train_matrix.shape[1]),
        "feature_schema_sha256": feature_diagnostics["feature_schema_sha256"],
        "inputs": [input_after[name] for name in sorted(input_after)],
        "sources": source_records,
        "artifacts": artifact_records,
        "contracts": {
            "full_public_training_rows": True,
            "public_files_read": sorted(inputs),
            "private_files_read": [],
            "target_derived_features": 0,
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "process_signals_sent": 0,
            "other_processes_modified": False,
            "gpu_used": False,
            "cuda_visible_devices": "",
            "cache_manifest_written_last": True,
        },
        "build_seconds": elapsed,
        "claim_boundary": "Public feature cache evidence is not an official score or medal.",
    }
    _may2022_atomic_json(manifest_path, manifest)
    return {
        "path": str(manifest_path),
        "bytes": manifest_path.stat().st_size,
        "sha256": _may2022_sha256(manifest_path),
        "train_rows": len(train_matrix),
        "test_rows": len(test_matrix),
        "feature_count": train_matrix.shape[1],
        "build_seconds": elapsed,
    }


def _may2022_require_manifest_record(
    records: Sequence[dict[str, Any]], name: str
) -> dict[str, Any]:
    matches = [record for record in records if record.get("relative_path") == name]
    if len(matches) != 1:
        raise RuntimeError(f"May 2022 cache manifest record is missing or duplicated: {name}")
    return matches[0]


def load_may2022_public_feature_cache(
    cache_dir: Path,
    *,
    data_root: Path,
    allowed_root: Path,
    seed: int,
    folds: int,
    verify_finite: bool = True,
) -> dict[str, Any]:
    """Validate and memory-map a CPU-built May cache for GPU-only fitting."""

    root = Path(allowed_root).expanduser().resolve()
    cache_root = _may2022_confined_path(Path(cache_dir), root, label="cache input")
    manifest_path = cache_root / "cache_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError("May 2022 cache has no safe completion manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exact = {
        "schema": MAY2022_PUBLIC_CACHE_SCHEMA,
        "status": "completed",
        "competition_id": "tabular-playground-series-may-2022",
        "visibility_mode": "PUBLIC_ONLY",
        "seed": int(seed),
        "folds": int(folds),
    }
    for key, expected in exact.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"May 2022 cache manifest {key} does not match {expected!r}")
    contracts = manifest.get("contracts")
    required_contracts = {
        "full_public_training_rows": True,
        "public_files_read": ["sample_submission.csv", "test.csv", "train.csv"],
        "private_files_read": [],
        "target_derived_features": 0,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
        "other_processes_modified": False,
        "gpu_used": False,
        "cuda_visible_devices": "",
        "cache_manifest_written_last": True,
    }
    if not isinstance(contracts, dict):
        raise RuntimeError("May 2022 cache contracts are missing")
    for key, expected in required_contracts.items():
        if contracts.get(key) != expected:
            raise RuntimeError(f"May 2022 cache contract failed: {key}")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or {
        record.get("relative_path") for record in artifacts if isinstance(record, dict)
    } != set(MAY2022_PUBLIC_CACHE_DATA_ARTIFACTS):
        raise RuntimeError("May 2022 cache artifact inventory is incomplete or unexpected")
    for name in MAY2022_PUBLIC_CACHE_DATA_ARTIFACTS:
        record = _may2022_require_manifest_record(artifacts, name)
        actual = _may2022_file_record(cache_root / name, relative_path=name)
        if actual["bytes"] != record.get("bytes") or actual["sha256"] != record.get("sha256"):
            raise RuntimeError(f"May 2022 cache artifact hash failed: {name}")

    source_records = manifest.get("sources")
    if not isinstance(source_records, list):
        raise RuntimeError("May 2022 cache source identity is missing")
    builder_records = [
        record for record in source_records
        if isinstance(record, dict) and record.get("role") == "feature_builder"
    ]
    if len(builder_records) != 1:
        raise RuntimeError("May 2022 cache must bind exactly one feature builder")
    current_builder = _may2022_file_record(Path(__file__))
    if (
        builder_records[0].get("bytes") != current_builder["bytes"]
        or builder_records[0].get("sha256") != current_builder["sha256"]
    ):
        raise RuntimeError("May 2022 cache feature-builder source drifted")

    data_root = _may2022_confined_path(Path(data_root), root, label="data root")
    inputs = _may2022_public_input_paths(data_root)
    input_records = manifest.get("inputs")
    if not isinstance(input_records, list):
        raise RuntimeError("May 2022 cache input inventory is missing")
    for name, path in inputs.items():
        expected = _may2022_require_manifest_record(
            input_records, f"prepared/public/{name}"
        )
        actual = _may2022_file_record(path, relative_path=f"prepared/public/{name}")
        if actual["bytes"] != expected.get("bytes") or actual["sha256"] != expected.get("sha256"):
            raise RuntimeError(f"May 2022 cache public input drifted: {name}")

    train_matrix = np.load(cache_root / "train_features.npy", mmap_mode="r", allow_pickle=False)
    test_matrix = np.load(cache_root / "test_features.npy", mmap_mode="r", allow_pickle=False)
    train_id = np.load(cache_root / "train_id.npy", mmap_mode="r", allow_pickle=False)
    test_id = np.load(cache_root / "test_id.npy", mmap_mode="r", allow_pickle=False)
    target = np.load(cache_root / "target.npy", mmap_mode="r", allow_pickle=False)
    if train_matrix.dtype != np.float32 or test_matrix.dtype != np.float32:
        raise RuntimeError("May 2022 cache feature matrices must be float32")
    if train_id.dtype != np.int64 or test_id.dtype != np.int64 or target.dtype != np.int8:
        raise RuntimeError("May 2022 cache ID or target dtype is invalid")
    if (
        train_matrix.ndim != 2
        or test_matrix.ndim != 2
        or train_matrix.shape[1] != test_matrix.shape[1]
        or len(train_id) != len(target)
        or len(train_matrix) != len(target)
        or len(test_matrix) != len(test_id)
    ):
        raise RuntimeError("May 2022 cache array shapes are inconsistent")
    if (
        manifest.get("train_rows") != len(train_matrix)
        or manifest.get("test_rows") != len(test_matrix)
        or manifest.get("feature_count") != train_matrix.shape[1]
    ):
        raise RuntimeError("May 2022 cache manifest dimensions are inconsistent")
    if len(np.unique(train_id)) != len(train_id) or len(np.unique(test_id)) != len(test_id):
        raise RuntimeError("May 2022 cache IDs are duplicated")
    if set(np.unique(target).tolist()) != {0, 1}:
        raise RuntimeError("May 2022 cache target is invalid")
    if verify_finite:
        for name, matrix in (("train", train_matrix), ("test", test_matrix)):
            for start in range(0, len(matrix), 65_536):
                if not np.isfinite(matrix[start : start + 65_536]).all():
                    raise RuntimeError(f"May 2022 cache {name} matrix contains nonfinite values")

    feature_names_payload = json.loads(
        (cache_root / "feature_names.json").read_text(encoding="utf-8")
    )
    if feature_names_payload.get("schema") != MAY2022_PUBLIC_CACHE_FEATURE_NAMES_SCHEMA:
        raise RuntimeError("May 2022 cache feature-name schema is invalid")
    feature_names = feature_names_payload.get("ordered_feature_names")
    dtypes = feature_names_payload.get("ordered_feature_dtypes")
    if (
        not isinstance(feature_names, list)
        or len(feature_names) != train_matrix.shape[1]
        or len(set(feature_names)) != len(feature_names)
        or dtypes != ["float32"] * len(feature_names)
    ):
        raise RuntimeError("May 2022 cache feature-name contract failed")
    schema_payload = {"columns": feature_names, "dtypes": dtypes}
    schema_sha = hashlib.sha256(
        json.dumps(schema_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if (
        feature_names_payload.get("feature_schema_sha256") != schema_sha
        or manifest.get("feature_schema_sha256") != schema_sha
    ):
        raise RuntimeError("May 2022 cache feature schema hash failed")

    with np.load(cache_root / "may2022_fold_assignments.npz", allow_pickle=False) as bundle:
        fold_id = np.asarray(bundle["id"], dtype=np.int64)
        fold_target = np.asarray(bundle["target"], dtype=np.int8)
        fold_assignment = np.asarray(bundle["fold"], dtype=np.int8)
        fold_seed = np.asarray(bundle["seed"], dtype=np.int64)
    if (
        not np.array_equal(fold_id, train_id)
        or not np.array_equal(fold_target, target)
        or not np.array_equal(fold_seed, np.asarray([int(seed)], dtype=np.int64))
    ):
        raise RuntimeError("May 2022 cache fold assignment identity failed")

    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    expected_splits = list(splitter.split(np.zeros(len(target), dtype=np.int8), target))
    nested_plans, coverage = build_may2022_nested_fold_plans(
        target,
        expected_splits,
        seed=int(seed),
    )
    expected_fold_assignment = np.full(len(target), -1, dtype=np.int8)
    with np.load(cache_root / "may2022_nested_fold_indices.npz", allow_pickle=False) as bundle:
        expected_keys: set[str] = set()
        for plan in nested_plans:
            fold_index = int(plan["outer_fold"])
            expected_fold_assignment[plan["outer_valid_index"]] = fold_index
            for name in (
                "outer_train_index",
                "outer_valid_index",
                "inner_train_index",
                "inner_valid_index",
            ):
                key = f"fold_{fold_index:02d}_{name}"
                expected_keys.add(key)
                if key not in bundle or not np.array_equal(
                    np.asarray(bundle[key], dtype=np.int64),
                    np.asarray(plan[name], dtype=np.int64),
                ):
                    raise RuntimeError(f"May 2022 cache nested fold drifted: {key}")
        if set(bundle.files) != expected_keys:
            raise RuntimeError("May 2022 cache nested fold inventory is unexpected")
    if (
        not np.all(coverage == 1)
        or not np.array_equal(fold_assignment, expected_fold_assignment)
    ):
        raise RuntimeError("May 2022 cache deterministic fold coverage failed")

    diagnostics = json.loads(
        (cache_root / "feature_diagnostics.json").read_text(encoding="utf-8")
    )
    if (
        diagnostics.get("visibility_mode") != "PUBLIC_ONLY"
        or diagnostics.get("private_labels_used") is not False
        or diagnostics.get("gpu_used") is not False
        or diagnostics.get("feature_engineering", {}).get("feature_schema_sha256") != schema_sha
    ):
        raise RuntimeError("May 2022 cache diagnostics contract failed")
    return {
        "cache_root": cache_root,
        "manifest": manifest,
        "manifest_record": _may2022_file_record(manifest_path),
        "train_features": train_matrix,
        "test_features": test_matrix,
        "train_id": train_id,
        "test_id": test_id,
        "target": target,
        "feature_names": feature_names,
        "feature_diagnostics": diagnostics["feature_engineering"],
        "fold_assignment": fold_assignment,
        "nested_plans": nested_plans,
    }


def _may2022_mean_scale_from_indices(
    feature_matrix: np.ndarray,
    indices: np.ndarray,
    *,
    chunk_rows: int = 65_536,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(indices, dtype=np.int64).reshape(-1)
    if not len(rows):
        raise RuntimeError("May 2022 normalization rows are empty")
    feature_sum = np.zeros(feature_matrix.shape[1], dtype=np.float64)
    feature_square_sum = np.zeros(feature_matrix.shape[1], dtype=np.float64)
    for start in range(0, len(rows), max(1, int(chunk_rows))):
        values = feature_matrix[rows[start : start + chunk_rows]]
        feature_sum += values.sum(axis=0, dtype=np.float64)
        feature_square_sum += np.square(values, dtype=np.float64).sum(axis=0)
    mean = feature_sum / len(rows)
    variance = np.maximum(feature_square_sum / len(rows) - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def build_may2022_xgboost_parameters(
    args: Any,
    *,
    seed: int,
    estimator_budget: int,
    selection: bool,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "n_estimators": int(estimator_budget),
        "max_depth": args.may_xgb_depth,
        "learning_rate": args.may_learning_rate,
        "subsample": 0.92,
        "colsample_bytree": 0.90,
        "min_child_weight": 2.0,
        "reg_alpha": 0.02,
        "reg_lambda": 3.0,
        "max_bin": 256,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": "cuda",
        "random_state": int(seed),
        "n_jobs": max(1, min(16, os.cpu_count() or 4)),
    }
    if selection:
        parameters["early_stopping_rounds"] = int(args.may_early_stopping)
    return parameters


def build_may2022_catboost_parameters(
    args: Any,
    *,
    seed: int,
    iteration_budget: int,
    selection: bool,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "iterations": int(iteration_budget),
        "depth": args.may_catboost_depth,
        "learning_rate": args.may_learning_rate,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "task_type": "GPU",
        "devices": "0",
        "gpu_ram_part": 0.60,
        "random_seed": int(seed),
        "border_count": 254,
        "l2_leaf_reg": 3.0,
        "random_strength": 0.35,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.35,
        "verbose": args.may_verbose_eval,
        "allow_writing_files": False,
    }
    if selection:
        parameters.update(
            {
                "od_type": "Iter",
                "od_wait": int(args.may_early_stopping),
            }
        )
    return parameters


def validate_may2022_nested_selection_contract(
    fold_records: Sequence[dict[str, Any]],
    oof_coverage: np.ndarray,
    *,
    expected_rows: int,
    expected_folds: int,
) -> dict[str, Any]:
    """Fail closed unless every outer prediction used inner-selected fixed budgets."""

    records = list(fold_records)
    coverage = np.asarray(oof_coverage, dtype=np.int16).reshape(-1)
    if len(records) != int(expected_folds):
        raise RuntimeError("May 2022 nested selection fold count is incomplete")
    if coverage.shape != (int(expected_rows),) or not np.all(coverage == 1):
        raise RuntimeError("May 2022 nested selection OOF coverage is not exactly one")
    seen_folds: set[int] = set()
    for record in records:
        fold = int(record.get("fold", -1))
        if fold in seen_folds:
            raise RuntimeError("May 2022 nested selection fold is duplicated")
        seen_folds.add(fold)
        nested = record.get("nested_selection")
        if not isinstance(nested, dict):
            raise RuntimeError("May 2022 nested selection provenance is missing")
        if nested.get("outer_validation_labels_used_for_selection") is not False:
            raise RuntimeError("May 2022 outer validation labels entered budget selection")
        if nested.get("all_budgets_frozen_before_outer_prediction") is not True:
            raise RuntimeError("May 2022 budgets were not frozen before outer prediction")
        if nested.get("refit_uses_complete_outer_train") is not True:
            raise RuntimeError("May 2022 outer refit did not use complete outer-train rows")
        if nested.get("iteration_budget_selection_label_scope") != "inner_validation_only":
            raise RuntimeError("May 2022 iteration budget selection label scope is invalid")
        if int(nested.get("inner_train_rows", -1)) + int(
            nested.get("inner_valid_rows", -1)
        ) != int(nested.get("outer_train_rows", -2)):
            raise RuntimeError("May 2022 nested split row counts are inconsistent")
        if int(nested.get("outer_train_rows", -1)) + int(
            nested.get("outer_valid_rows", -1)
        ) != int(expected_rows):
            raise RuntimeError("May 2022 outer split row counts are inconsistent")
        if int(nested.get("outer_fold", -1)) != fold - 1:
            raise RuntimeError("May 2022 nested outer-fold provenance is inconsistent")
        if not math.isclose(
            float(nested.get("inner_validation_fraction", math.nan)),
            MAY2022_INNER_VALIDATION_FRACTION,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("May 2022 inner validation fraction provenance is invalid")
        inner_split_seed = nested.get("inner_split_seed")
        if isinstance(inner_split_seed, bool) or not isinstance(inner_split_seed, (int, np.integer)):
            raise RuntimeError("May 2022 inner split seed provenance is invalid")
        for name in (
            "outer_train_index_sha256",
            "outer_valid_index_sha256",
            "inner_train_index_sha256",
            "inner_valid_index_sha256",
        ):
            digest = str(nested.get(name) or "").lower()
            if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
                raise RuntimeError(f"May 2022 nested split hash is invalid: {name}")
        budgets = nested.get("budgets")
        if not isinstance(budgets, dict) or set(budgets) != {
            "residual_mlp",
            "xgboost",
            "catboost",
        }:
            raise RuntimeError("May 2022 frozen model budgets are incomplete")
        for model_name, budget in budgets.items():
            if not isinstance(budget, dict):
                raise RuntimeError(f"May 2022 {model_name} budget provenance is invalid")
            integer_fields = (
                "selected_iteration",
                "requested_budget",
                "frozen_refit_budget",
                "selection_seed",
            )
            if any(
                isinstance(budget.get(name), bool)
                or not isinstance(budget.get(name), (int, np.integer))
                for name in integer_fields
            ):
                raise RuntimeError(f"May 2022 {model_name} budget provenance is incomplete")
            if not isinstance(budget.get("selected_iteration_zero_based"), bool) or not isinstance(
                budget.get("fallback_to_requested_budget"), bool
            ):
                raise RuntimeError(f"May 2022 {model_name} budget basis is invalid")
            selected = int(budget["selected_iteration"])
            requested = int(budget["requested_budget"])
            frozen = int(budget["frozen_refit_budget"])
            if not 1 <= frozen <= requested:
                raise RuntimeError(f"May 2022 {model_name} frozen budget is invalid")
            expected_budget = freeze_may2022_refit_budget(
                selected_iteration=selected,
                requested_budget=requested,
                zero_based_iteration=budget["selected_iteration_zero_based"],
            )
            if frozen != expected_budget["frozen_refit_budget"] or budget.get(
                "fallback_to_requested_budget"
            ) is not expected_budget["fallback_to_requested_budget"]:
                raise RuntimeError(f"May 2022 {model_name} frozen budget provenance is inconsistent")
            if budget.get("outer_validation_labels_used_for_selection") is not False:
                raise RuntimeError(
                    f"May 2022 {model_name} used outer validation labels for selection"
                )
            selection_inner_auc = float(budget.get("selection_inner_auc", math.nan))
            if not math.isfinite(selection_inner_auc) or not 0.0 <= selection_inner_auc <= 1.0:
                raise RuntimeError(f"May 2022 {model_name} inner-selection AUC is invalid")
    if seen_folds != set(range(1, int(expected_folds) + 1)):
        raise RuntimeError("May 2022 nested selection fold IDs are incomplete")
    return {
        "schema": "evomind.mlebench.may2022_nested_selection_contract.v1",
        "passed": True,
        "fold_count": len(records),
        "expected_rows": int(expected_rows),
        "oof_coverage_min": int(coverage.min()),
        "oof_coverage_max": int(coverage.max()),
        "oof_coverage_sha256": hashlib.sha256(coverage.tobytes()).hexdigest(),
        "outer_validation_labels_used_for_selection": False,
        "all_budgets_frozen_before_outer_prediction": True,
        "refit_uses_complete_outer_train": True,
    }


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=np.float64))
    return series.rank(method="average", pct=True).to_numpy(dtype=np.float64)


def _simplex_weights(component_count: int, denominator: int = 20) -> list[tuple[float, ...]]:
    integer_weights: list[tuple[int, ...]] = []

    def visit(prefix: tuple[int, ...], remaining: int) -> None:
        if len(prefix) == component_count - 1:
            integer_weights.append((*prefix, remaining))
            return
        for value in range(remaining + 1):
            visit((*prefix, value), remaining - value)

    visit((), denominator)
    return [tuple(value / denominator for value in row) for row in integer_weights]


def select_may2022_multimodel_blend(
    components: dict[str, np.ndarray],
    labels: np.ndarray | pd.Series,
) -> dict[str, Any]:
    """Select a fixed-grid nonnegative raw/rank OOF blend."""

    from sklearn.metrics import roc_auc_score

    names = list(components)
    if not 2 <= len(names) <= 4:
        raise ValueError("May 2022 blend expects between two and four components")
    arrays = {name: np.asarray(components[name], dtype=np.float64) for name in names}
    target = np.asarray(labels)
    if target.ndim != 1 or any(value.shape != target.shape for value in arrays.values()):
        raise ValueError("May 2022 blend inputs must have identical one-dimensional shapes")
    candidates: list[dict[str, Any]] = []
    weight_grid = _simplex_weights(len(names), denominator=20)
    for space in ("raw_probability", "rank"):
        values = arrays if space == "raw_probability" else {
            name: _rank_normalize(value) for name, value in arrays.items()
        }
        for row in weight_grid:
            probability = sum(weight * values[name] for name, weight in zip(names, row))
            candidates.append({
                "space": space,
                "weights": {name: float(weight) for name, weight in zip(names, row)},
                "oof_auc": float(roc_auc_score(target, probability)),
            })
    candidates.sort(
        key=lambda item: (
            -item["oof_auc"],
            0 if item["space"] == "raw_probability" else 1,
            sum(abs(weight - 1.0 / len(names)) for weight in item["weights"].values()),
        )
    )
    best = dict(candidates[0])
    best["component_oof_auc"] = {
        name: float(roc_auc_score(target, value)) for name, value in arrays.items()
    }
    best["grid_denominator"] = 20
    best["candidate_count"] = len(candidates)
    return best


def apply_may2022_multimodel_blend(
    components: dict[str, np.ndarray],
    blend: dict[str, Any],
) -> np.ndarray:
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in components.items()}
    if blend["space"] == "rank":
        arrays = {name: _rank_normalize(value) for name, value in arrays.items()}
    return sum(float(blend["weights"][name]) * arrays[name] for name in blend["weights"])


def cross_fit_may2022_multimodel_blend(
    components: dict[str, np.ndarray],
    labels: np.ndarray | pd.Series,
    folds: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Select May 2022 blend parameters without each outer validation fold."""

    target = np.asarray(labels, dtype=np.int8)
    fold_values = np.asarray(folds)
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in components.items()}
    if target.ndim != 1 or fold_values.shape != target.shape:
        raise RuntimeError("May 2022 cross-fit labels or folds have invalid shape")
    if any(value.shape != target.shape for value in arrays.values()):
        raise RuntimeError("May 2022 cross-fit component shapes differ")
    prediction = np.full(len(target), np.nan, dtype=np.float64)
    records: list[dict[str, Any]] = []
    for fold in sorted(int(value) for value in np.unique(fold_values)):
        validation = fold_values == fold
        fitting = ~validation
        if set(target[fitting].tolist()) != {0, 1} or set(target[validation].tolist()) != {0, 1}:
            raise RuntimeError("May 2022 cross-fit fold does not contain both classes")
        selected = select_may2022_multimodel_blend(
            {name: value[fitting] for name, value in arrays.items()},
            target[fitting],
        )
        prediction[validation] = apply_may2022_multimodel_blend(
            {name: value[validation] for name, value in arrays.items()},
            selected,
        )
        records.append({
            "fold": fold,
            "space": selected["space"],
            "weights": selected["weights"],
            "meta_fit_auc": selected["oof_auc"],
            "outer_auc": float(compute_metric("roc_auc", target[validation], prediction[validation])),
        })
    if not np.isfinite(prediction).all():
        raise RuntimeError("May 2022 cross-fit blend did not cover every row")
    return prediction, records


def score_may2022_deployment_blend_oof(
    components: dict[str, np.ndarray],
    labels: np.ndarray | pd.Series,
    folds: np.ndarray,
    blend: dict[str, Any],
) -> dict[str, Any]:
    """Score the exact globally selected blend that will be applied to test rows."""

    target = np.asarray(labels, dtype=np.int8)
    fold_values = np.asarray(folds)
    if target.ndim != 1 or fold_values.shape != target.shape:
        raise RuntimeError("May 2022 deployment blend labels or folds have invalid shape")
    prediction = apply_may2022_multimodel_blend(components, blend)
    if prediction.shape != target.shape or not np.isfinite(prediction).all():
        raise RuntimeError("May 2022 deployment blend OOF prediction is invalid")
    fold_auc = {
        str(int(fold)): float(
            compute_metric("roc_auc", target[fold_values == fold], prediction[fold_values == fold])
        )
        for fold in sorted(np.unique(fold_values))
    }
    return {
        "oof_probability": prediction,
        "oof_auc": float(compute_metric("roc_auc", target, prediction)),
        "fold_auc": fold_auc,
    }


def evaluate_may2022_confirmation_gate(
    seed_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate the predeclared three-seed OOF promotion contract.

    Every supplied frozen confirmation seed must be eligible.  This prevents a
    failed seed from being hidden by adding more successful records and keeps
    the gate aligned with the final MLE-Bench reporting contract.
    """

    normalized: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    for record in seed_records:
        seed = int(record["seed"])
        if seed in seen_seeds:
            raise RuntimeError(f"May 2022 confirmation seed is duplicated: {seed}")
        seen_seeds.add(seed)
        aggregate_auc = float(record["aggregate_oof_auc"])
        confirmation_auc = float(record.get("confirmation_oof_auc", aggregate_auc))
        fold_auc = [float(value) for value in record["fold_auc"]]
        if not fold_auc or not np.isfinite([aggregate_auc, confirmation_auc, *fold_auc]).all():
            raise RuntimeError("May 2022 confirmation evidence contains invalid AUC values")
        checks = {
            "aggregate_oof_auc": aggregate_auc >= MAY2022_AGGREGATE_PROMOTION_AUC,
            "confirmation_oof_auc": confirmation_auc >= MAY2022_CONFIRMATION_SEED_AUC,
            "all_folds": min(fold_auc) >= MAY2022_FOLD_PROMOTION_AUC,
        }
        normalized.append({
            "seed": seed,
            "aggregate_oof_auc": aggregate_auc,
            "confirmation_oof_auc": confirmation_auc,
            "minimum_fold_auc": min(fold_auc),
            "fold_auc": fold_auc,
            "checks": checks,
            "eligible": bool(all(checks.values())),
        })
    eligible = [record for record in normalized if record["eligible"]]
    enough_distinct_seeds = len(normalized) >= MAY2022_CONFIRMATION_SEEDS_REQUIRED
    every_seed_eligible = bool(normalized) and len(eligible) == len(normalized)
    return {
        "schema": "evomind.mlebench.may2022_confirmation_gate.v2",
        "thresholds": {
            "aggregate_oof_auc": MAY2022_AGGREGATE_PROMOTION_AUC,
            "each_confirmation_seed_oof_auc": MAY2022_CONFIRMATION_SEED_AUC,
            "every_fold_oof_auc": MAY2022_FOLD_PROMOTION_AUC,
            "confirmation_seeds_required": MAY2022_CONFIRMATION_SEEDS_REQUIRED,
        },
        "seed_records": normalized,
        "confirmation_seed_count": len(normalized),
        "eligible_seed_count": len(eligible),
        "enough_distinct_seeds": enough_distinct_seeds,
        "every_seed_eligible": every_seed_eligible,
        "passed": bool(enough_distinct_seeds and every_seed_eligible),
        "claim_boundary": "Internal OOF promotion evidence is not an official medal.",
    }


def build_may2022_run_promotion_gate(
    *,
    seed_gate: dict[str, Any],
    full_data_contract: dict[str, Any],
    deployment_oof_auc: float,
    deployment_fold_auc: Sequence[float],
) -> dict[str, Any]:
    """Build a fail-closed gate for the exact rule deployed to test rows."""

    aggregate_auc = float(deployment_oof_auc)
    fold_auc = [float(value) for value in deployment_fold_auc]
    if not fold_auc or not np.isfinite([aggregate_auc, *fold_auc]).all():
        raise RuntimeError("May 2022 deployment promotion evidence is invalid")
    seed_records = list(seed_gate.get("seed_records") or [])
    checks = {
        "exact_deployment_oof_auc": aggregate_auc >= MAY2022_AGGREGATE_PROMOTION_AUC,
        "every_deployment_fold_oof_auc": min(fold_auc) >= MAY2022_FOLD_PROMOTION_AUC,
        "full_training_data": bool(full_data_contract.get("full_training_data")),
        "three_seed_confirmation_complete": seed_gate.get("passed") is True,
        "confirmation_seed_count": (
            len(seed_records) >= MAY2022_CONFIRMATION_SEEDS_REQUIRED
        ),
        "every_confirmation_seed_eligible": bool(seed_records) and all(
            record.get("eligible") is True for record in seed_records
        ),
    }
    return {
        "schema": "evomind.mlebench_lite.may2022_deployment_gate.v2",
        "name": "may2022_exact_deployment_three_seed_oof_gate",
        "metric": "roc_auc",
        "direction": "maximize",
        "threshold": MAY2022_AGGREGATE_PROMOTION_AUC,
        "internal_score": aggregate_auc,
        "deployment_fold_auc": fold_auc,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "confirmation_gate": seed_gate,
        "official_grader_executed": False,
        "claim_boundary": "Internal OOF promotion evidence is not an official medal.",
    }


def _may2022_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_may2022_artifact_manifest(
    task_dir: Path,
    *,
    source_paths: Sequence[Path] = (),
    optimization_plan: Path | None = None,
) -> dict[str, Any]:
    """Hash every runner artifact plus source and dependency identities."""

    task_root = task_dir.resolve()
    dependency_names = (
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "xgboost",
        "catboost",
    )
    dependencies: dict[str, str | None] = {}
    for name in dependency_names:
        try:
            dependencies[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            dependencies[name] = None
    environment_path = task_root / "may2022_environment_manifest.json"
    wave0.write_json(
        environment_path,
        {
            "schema": "evomind.mlebench.may2022_environment_manifest.v1",
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "dependencies": dependencies,
        },
    )

    required_names = {
        "may2022_fold_assignments.npz",
        "may2022_oof_ensemble.npz",
        "may2022_ensemble_diagnostics.json",
        "may2022_promotion_gate.json",
        "promotion_gate.json",
        "submission.csv",
        "submission_validation.json",
        "may2022_environment_manifest.json",
    }
    missing = sorted(name for name in required_names if not (task_root / name).is_file())
    if missing:
        raise RuntimeError(f"May 2022 artifact manifest is missing required files: {missing}")

    artifact_records = []
    for path in sorted(task_root.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name == "may2022_artifact_manifest.json":
            continue
        artifact_records.append({
            "path": str(path),
            "relative_path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _may2022_sha256(path),
        })

    source_records = []
    for path in sorted({Path(value).resolve() for value in source_paths}, key=str):
        if not path.is_file():
            raise RuntimeError(f"May 2022 source identity is missing: {path}")
        source_records.append({
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _may2022_sha256(path),
        })

    plan_record = None
    if optimization_plan is not None:
        plan_path = Path(optimization_plan).resolve()
        if not plan_path.is_file():
            raise RuntimeError(f"May 2022 optimization plan is missing: {plan_path}")
        plan_record = {
            "path": str(plan_path),
            "bytes": plan_path.stat().st_size,
            "sha256": _may2022_sha256(plan_path),
        }

    manifest_path = task_root / "may2022_artifact_manifest.json"
    payload = {
        "schema": "evomind.mlebench.may2022_artifact_manifest.v1",
        "artifact_count": len(artifact_records),
        "artifacts": artifact_records,
        "sources": source_records,
        "optimization_plan": plan_record,
        "required_artifacts_present": True,
        "later_orchestrator_artifacts_not_in_scope": ["result.json", "summary.json"],
        "claim_boundary": "Artifact integrity evidence is not an official medal.",
    }
    wave0.write_json(manifest_path, payload)
    return {
        "path": str(manifest_path),
        "bytes": manifest_path.stat().st_size,
        "sha256": _may2022_sha256(manifest_path),
        "artifact_count": len(artifact_records),
        "required_artifacts_present": True,
    }


def align_scalar_submission_by_id(
    sample: pd.DataFrame,
    test_ids: pd.Series,
    prediction: np.ndarray,
    *,
    id_column: str,
    target_column: str,
) -> pd.DataFrame:
    """Align finite scalar predictions to sample order with a one-to-one ID join."""

    if id_column not in sample or target_column not in sample:
        raise RuntimeError("Submission schema is missing the ID or target column")
    if sample[id_column].duplicated().any() or test_ids.duplicated().any():
        raise RuntimeError("Submission or prediction IDs are duplicated")
    values = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if len(values) != len(test_ids) or not np.isfinite(values).all():
        raise RuntimeError("Scalar prediction shape or finiteness is invalid")
    sample_keys = sample[id_column].astype(str)
    test_keys = test_ids.astype(str)
    if set(sample_keys) != set(test_keys):
        raise RuntimeError("Submission and prediction ID sets differ")
    keyed = pd.DataFrame({id_column: test_keys.to_numpy(), target_column: values})
    aligned = sample.copy()
    original_ids = aligned[id_column].copy()
    aligned[id_column] = sample_keys.to_numpy()
    aligned = aligned.drop(columns=[target_column]).merge(
        keyed,
        on=id_column,
        how="left",
        validate="one_to_one",
    )
    aligned[id_column] = original_ids.to_numpy()
    if aligned[target_column].isna().any():
        raise RuntimeError("Submission ID join produced missing predictions")
    return aligned[list(sample.columns)]


def select_may2022_binary_blend(
    xgboost_probability: np.ndarray,
    catboost_probability: np.ndarray,
    labels: np.ndarray | pd.Series,
) -> dict[str, Any]:
    """Backward-compatible two-component view of the general blend search."""

    selected = select_may2022_multimodel_blend(
        {"xgboost": xgboost_probability, "catboost": catboost_probability}, labels
    )
    return {
        **selected,
        "xgboost_weight": selected["weights"]["xgboost"],
        "catboost_weight": selected["weights"]["catboost"],
        "xgboost_oof_auc": selected["component_oof_auc"]["xgboost"],
        "catboost_oof_auc": selected["component_oof_auc"]["catboost"],
    }


def apply_may2022_binary_blend(
    xgboost_probability: np.ndarray,
    catboost_probability: np.ndarray,
    blend: dict[str, Any],
) -> np.ndarray:
    return apply_may2022_multimodel_blend(
        {"xgboost": xgboost_probability, "catboost": catboost_probability},
        {
            "space": blend["space"],
            "weights": {
                "xgboost": float(blend["xgboost_weight"]),
                "catboost": float(blend["catboost_weight"]),
            },
        },
    )


def build_may2022_residual_mlp(
    input_features: int,
    *,
    width: int = 768,
    block_count: int = 5,
    dropout: float = 0.08,
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
):
    """Construct the fold-standardized residual MLP selected by GPT-5.6 review."""

    import torch
    from torch import nn

    class ResidualBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, width * 2),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(width * 2, width),
                nn.Dropout(dropout),
            )

        def forward(self, value):
            return value + 0.5 * self.layers(value)

    class ResidualMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            mean_value = np.zeros(input_features, dtype=np.float32) if mean is None else mean
            scale_value = np.ones(input_features, dtype=np.float32) if scale is None else scale
            self.register_buffer("feature_mean", torch.as_tensor(mean_value, dtype=torch.float32))
            self.register_buffer("feature_scale", torch.as_tensor(scale_value, dtype=torch.float32))
            self.input = nn.Sequential(
                nn.Linear(input_features, width),
                nn.SiLU(inplace=True),
                nn.LayerNorm(width),
            )
            self.blocks = nn.ModuleList(ResidualBlock() for _ in range(block_count))
            self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

        def forward(self, value):
            value = torch.clamp((value - self.feature_mean) / self.feature_scale, -10.0, 10.0)
            hidden = self.input(value)
            for block in self.blocks:
                hidden = block(hidden)
            return self.output(hidden).squeeze(-1)

    return ResidualMLP()


def _may2022_predict_mlp(model: Any, features: Any, indices: np.ndarray, batch_size: int) -> np.ndarray:
    import torch

    index_tensor = torch.as_tensor(indices, dtype=torch.long, device="cuda")
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(index_tensor), batch_size):
            batch_index = index_tensor[start:start + batch_size]
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(features[batch_index])
            predictions.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(predictions)


def _select_may2022_mlp_budget(
    *,
    feature_tensor: Any,
    train_index: np.ndarray,
    valid_index: np.ndarray,
    target: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    args: Any,
    seed: int,
    fold: int,
    logger: Any,
) -> dict[str, Any]:
    import torch
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_may2022_residual_mlp(
        feature_tensor.shape[1],
        width=args.may_mlp_width,
        block_count=args.may_mlp_blocks,
        dropout=args.may_mlp_dropout,
        mean=mean,
        scale=scale,
    ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.may_mlp_learning_rate,
        weight_decay=args.may_mlp_weight_decay,
        fused=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.may_mlp_epochs, eta_min=args.may_mlp_learning_rate * 0.05
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    train_indices = torch.as_tensor(train_index, dtype=torch.long, device="cuda")
    target_tensor = torch.as_tensor(target, dtype=torch.float32, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    best_auc = -math.inf
    best_epoch = 0
    history: list[dict[str, Any]] = []
    stale_epochs = 0

    for epoch in range(1, args.may_mlp_epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_indices), generator=generator, device="cuda")
        shuffled = train_indices[permutation]
        loss_sum = 0.0
        row_count = 0
        for start in range(0, len(shuffled), args.may_mlp_batch_size):
            batch_index = shuffled[start:start + args.may_mlp_batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(feature_tensor[batch_index])
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, target_tensor[batch_index]
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(batch_index)
            row_count += len(batch_index)
        scheduler.step()
        valid_probability = _may2022_predict_mlp(
            model, feature_tensor, valid_index, args.may_mlp_batch_size * 4
        )
        valid_auc = float(roc_auc_score(target[valid_index], valid_probability))
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / max(row_count, 1),
            "valid_auc": valid_auc,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        logger.info(
            "[tabular-playground-series-may-2022] fold=%d mlp_epoch=%d loss=%.6f auc=%.7f",
            fold,
            epoch,
            record["train_loss"],
            valid_auc,
        )
        if valid_auc > best_auc + 1e-7:
            best_auc = valid_auc
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.may_mlp_patience:
                break

    if best_epoch < 1:
        raise RuntimeError("May 2022 residual MLP inner selector produced no budget")
    budget = freeze_may2022_refit_budget(
        selected_iteration=best_epoch,
        requested_budget=args.may_mlp_epochs,
        zero_based_iteration=False,
    )
    del model, optimizer, scheduler, scaler, train_indices, target_tensor
    torch.cuda.empty_cache()
    gc.collect()
    return {
        **budget,
        "selection_inner_auc": best_auc,
        "selection_epochs_ran": len(history),
        "selection_history": history,
        "selection_seed": int(seed),
        "outer_validation_labels_used_for_selection": False,
    }


def _refit_may2022_mlp_fold(
    *,
    feature_tensor: Any,
    test_tensor: Any,
    train_index: np.ndarray,
    valid_index: np.ndarray,
    target: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    args: Any,
    seed: int,
    fold: int,
    frozen_epochs: int,
    checkpoint_path: Path,
    logger: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Freshly refit the MLP on all outer-train rows for a frozen epoch budget."""

    import torch

    epoch_budget = int(frozen_epochs)
    if not 1 <= epoch_budget <= int(args.may_mlp_epochs):
        raise RuntimeError("May 2022 residual MLP refit epoch budget is invalid")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_may2022_residual_mlp(
        feature_tensor.shape[1],
        width=args.may_mlp_width,
        block_count=args.may_mlp_blocks,
        dropout=args.may_mlp_dropout,
        mean=mean,
        scale=scale,
    ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.may_mlp_learning_rate,
        weight_decay=args.may_mlp_weight_decay,
        fused=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(args.may_mlp_epochs)),
        eta_min=args.may_mlp_learning_rate * 0.05,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    train_indices = torch.as_tensor(train_index, dtype=torch.long, device="cuda")
    target_tensor = torch.as_tensor(target, dtype=torch.float32, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    history: list[dict[str, Any]] = []
    for epoch in range(1, epoch_budget + 1):
        model.train()
        permutation = torch.randperm(len(train_indices), generator=generator, device="cuda")
        shuffled = train_indices[permutation]
        loss_sum = 0.0
        row_count = 0
        for start in range(0, len(shuffled), args.may_mlp_batch_size):
            batch_index = shuffled[start : start + args.may_mlp_batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(feature_tensor[batch_index])
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, target_tensor[batch_index]
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(batch_index)
            row_count += len(batch_index)
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / max(row_count, 1),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        logger.info(
            "[tabular-playground-series-may-2022] fold=%d mlp_refit_epoch=%d loss=%.6f",
            fold,
            epoch,
            record["train_loss"],
        )

    valid_probability = _may2022_predict_mlp(
        model, feature_tensor, valid_index, args.may_mlp_batch_size * 4
    )
    test_indices = np.arange(len(test_tensor), dtype=np.int64)
    test_probability = _may2022_predict_mlp(
        model, test_tensor, test_indices, args.may_mlp_batch_size * 4
    )
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    torch.save(
        {
            "state_dict": state,
            "seed": seed,
            "fold": fold,
            "fixed_epoch_budget": epoch_budget,
            "outer_train_rows": int(len(train_index)),
            "outer_validation_labels_used_for_training_or_selection": False,
            "history": history,
        },
        checkpoint_path,
    )
    del model, optimizer, scheduler, scaler, train_indices, target_tensor, state
    torch.cuda.empty_cache()
    gc.collect()
    return valid_probability, test_probability, {
        "mlp_refit_epochs": epoch_budget,
        "mlp_refit_seed": int(seed),
        "mlp_refit_history": history,
        "mlp_outer_auc_scored_after_refit": True,
    }


def run_may2022_gpu_ensemble(args: Any, task_dir: Path, logger: Any) -> dict[str, Any]:
    """Full-data residual-MLP/XGBoost/CatBoost recovery for TPS May 2022."""

    from catboost import CatBoostClassifier
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier

    competition_id = "tabular-playground-series-may-2022"
    resolved = resolve_competition(competition_id, args.data_root)
    sample = pd.read_csv(resolved.sample_submission_path)
    cache_value = str(getattr(args, "may_precomputed_cache_dir", "") or "").strip()
    require_cache = bool(getattr(args, "may_require_precomputed_cache", False))
    cache_bundle: dict[str, Any] | None = None
    if cache_value:
        cache_bundle = load_may2022_public_feature_cache(
            Path(cache_value),
            data_root=Path(args.data_root),
            allowed_root=Path(args.allowed_root),
            seed=int(args.seed),
            folds=int(args.may_folds),
        )
        target = np.asarray(cache_bundle["target"], dtype=np.int8)
        train = pd.DataFrame({
            "id": np.asarray(cache_bundle["train_id"], dtype=np.int64),
            "target": target,
        })
        test = pd.DataFrame({
            "id": np.asarray(cache_bundle["test_id"], dtype=np.int64),
        })
        feature_names = list(cache_bundle["feature_names"])
        train_features = pd.DataFrame(
            cache_bundle["train_features"], columns=feature_names, copy=False
        )
        test_features = pd.DataFrame(
            cache_bundle["test_features"], columns=feature_names, copy=False
        )
        feature_diagnostics = dict(cache_bundle["feature_diagnostics"])
        feature_diagnostics["precomputed_cache"] = cache_bundle["manifest_record"]
        fold_count = int(args.may_folds)
        nested_plans = list(cache_bundle["nested_plans"])
        fold_assignment = np.asarray(cache_bundle["fold_assignment"], dtype=np.int8)
        planned_oof_coverage = np.ones(len(target), dtype=np.int16)
    else:
        if require_cache:
            raise RuntimeError("May 2022 requires a verified precomputed public cache")
        train = pd.read_csv(resolved.public_dir / "train.csv")
        test = pd.read_csv(resolved.public_dir / "test.csv")
        if "id" not in train or "id" not in test:
            raise RuntimeError("May 2022 train or test ID column is missing")
        if train["id"].duplicated().any() or test["id"].duplicated().any():
            raise RuntimeError("May 2022 train or test IDs are duplicated")
        target = train["target"].astype(np.int8).to_numpy()
        train_features, test_features, feature_diagnostics = build_may2022_features(
            train, test
        )
        fold_count = min(int(args.may_folds), int(np.bincount(target).min()))
        if fold_count < 2:
            raise RuntimeError("May 2022 requires at least two stratified folds")
        splitter = StratifiedKFold(
            n_splits=fold_count, shuffle=True, random_state=args.seed
        )
        splits = list(splitter.split(train_features, target))
        nested_plans, planned_oof_coverage = build_may2022_nested_fold_plans(
            target,
            splits,
            seed=args.seed,
        )
        fold_assignment = np.full(len(train), -1, dtype=np.int8)

    available_rows = len(train)
    if args.may_max_train_rows and len(train) > args.may_max_train_rows:
        raise RuntimeError(
            "May 2022 medal mode forbids training-row truncation: "
            f"requested={args.may_max_train_rows} available={available_rows}"
        )
    full_data_contract = validate_may2022_full_data_contract(
        train_rows_available=available_rows,
        train_rows_used=len(train),
    )

    logger.info(
        "[%s] rows=%d/%d test=%d features=%d folds=%d",
        competition_id,
        len(train),
        available_rows,
        len(test),
        len(train_features.columns),
        fold_count,
    )
    started = time.perf_counter()
    nested_index_payload: dict[str, np.ndarray] = {}
    for plan in nested_plans:
        fold_index = int(plan["outer_fold"])
        fold_assignment[plan["outer_valid_index"]] = fold_index
        for name in (
            "outer_train_index",
            "outer_valid_index",
            "inner_train_index",
            "inner_valid_index",
        ):
            nested_index_payload[f"fold_{fold_index:02d}_{name}"] = np.asarray(
                plan[name], dtype=np.int64
            )
    if np.any(fold_assignment < 0) or not np.all(planned_oof_coverage == 1):
        raise RuntimeError("May 2022 fold assignment did not cover every training row")
    if cache_bundle is not None:
        for name in ("may2022_fold_assignments.npz", "may2022_nested_fold_indices.npz"):
            shutil.copy2(cache_bundle["cache_root"] / name, task_dir / name)
        feature_matrix = np.asarray(cache_bundle["train_features"])
        test_matrix = np.asarray(cache_bundle["test_features"])
        if not feature_matrix.flags.c_contiguous or not test_matrix.flags.c_contiguous:
            raise RuntimeError("May 2022 precomputed feature matrices are not C-contiguous")
    else:
        np.savez_compressed(
            task_dir / "may2022_fold_assignments.npz",
            id=train["id"].to_numpy(),
            target=target,
            fold=fold_assignment,
            seed=np.asarray([args.seed]),
        )
        np.savez_compressed(
            task_dir / "may2022_nested_fold_indices.npz",
            **nested_index_payload,
        )
        feature_matrix = np.ascontiguousarray(
            train_features.to_numpy(dtype=np.float32, copy=False)
        )
        test_matrix = np.ascontiguousarray(
            test_features.to_numpy(dtype=np.float32, copy=False)
        )
    import torch

    feature_tensor = torch.as_tensor(feature_matrix, dtype=torch.float32, device="cuda")
    test_tensor = torch.as_tensor(test_matrix, dtype=torch.float32, device="cuda")
    mlp_oof = np.full(len(train), np.nan, dtype=np.float64)
    xgboost_oof = np.full(len(train), np.nan, dtype=np.float64)
    catboost_oof = np.full(len(train), np.nan, dtype=np.float64)
    mlp_test = np.zeros(len(test), dtype=np.float64)
    xgboost_test = np.zeros(len(test), dtype=np.float64)
    catboost_test = np.zeros(len(test), dtype=np.float64)
    oof_coverage = np.zeros(len(train), dtype=np.int8)
    component_oof_coverage = {
        "residual_mlp": np.zeros(len(train), dtype=np.int8),
        "xgboost": np.zeros(len(train), dtype=np.int8),
        "catboost": np.zeros(len(train), dtype=np.int8),
    }
    fold_records: list[dict[str, Any]] = []

    for plan in nested_plans:
        fold = int(plan["outer_fold"]) + 1
        train_index = np.asarray(plan["outer_train_index"], dtype=np.int64)
        valid_index = np.asarray(plan["outer_valid_index"], dtype=np.int64)
        inner_train_index = np.asarray(plan["inner_train_index"], dtype=np.int64)
        inner_valid_index = np.asarray(plan["inner_valid_index"], dtype=np.int64)
        fold_seed = int(args.seed + fold * 101)
        inner_mean, inner_scale = _may2022_mean_scale_from_indices(
            feature_matrix, inner_train_index
        )
        fold_mean, fold_scale = _may2022_mean_scale_from_indices(feature_matrix, train_index)

        mlp_budget = _select_may2022_mlp_budget(
            feature_tensor=feature_tensor,
            train_index=inner_train_index,
            valid_index=inner_valid_index,
            target=target,
            mean=inner_mean,
            scale=inner_scale,
            args=args,
            seed=fold_seed,
            fold=fold,
            logger=logger,
        )
        inner_x_train = train_features.iloc[inner_train_index]
        inner_x_valid = train_features.iloc[inner_valid_index]
        inner_y_train = target[inner_train_index]
        inner_y_valid = target[inner_valid_index]

        xgboost_selector = XGBClassifier(
            **build_may2022_xgboost_parameters(
                args,
                seed=fold_seed,
                estimator_budget=args.may_xgb_estimators,
                selection=True,
            )
        )
        xgboost_selector.fit(
            inner_x_train,
            inner_y_train,
            eval_set=[(inner_x_valid, inner_y_valid)],
            verbose=args.may_verbose_eval,
        )
        xgb_inner_probability = xgboost_selector.predict_proba(inner_x_valid)[:, 1]
        raw_xgboost_best = getattr(xgboost_selector, "best_iteration", -1)
        xgboost_best = -1 if raw_xgboost_best is None else int(raw_xgboost_best)
        xgboost_budget = freeze_may2022_refit_budget(
            selected_iteration=xgboost_best,
            requested_budget=args.may_xgb_estimators,
            zero_based_iteration=True,
        )
        xgboost_budget.update(
            {
                "selection_inner_auc": float(
                    compute_metric("roc_auc", inner_y_valid, xgb_inner_probability)
                ),
                "selection_seed": fold_seed,
                "outer_validation_labels_used_for_selection": False,
            }
        )

        catboost_selector = CatBoostClassifier(
            **build_may2022_catboost_parameters(
                args,
                seed=fold_seed + 17,
                iteration_budget=args.may_catboost_iterations,
                selection=True,
            )
        )
        catboost_selector.fit(
            inner_x_train,
            inner_y_train,
            eval_set=(inner_x_valid, inner_y_valid),
            use_best_model=True,
        )
        cat_inner_probability = catboost_selector.predict_proba(inner_x_valid)[:, 1]
        raw_catboost_best = catboost_selector.get_best_iteration()
        catboost_best = -1 if raw_catboost_best is None else int(raw_catboost_best)
        catboost_budget = freeze_may2022_refit_budget(
            selected_iteration=catboost_best,
            requested_budget=args.may_catboost_iterations,
            zero_based_iteration=True,
        )
        catboost_budget.update(
            {
                "selection_inner_auc": float(
                    compute_metric("roc_auc", inner_y_valid, cat_inner_probability)
                ),
                "selection_seed": fold_seed + 17,
                "outer_validation_labels_used_for_selection": False,
            }
        )
        del (
            xgboost_selector,
            xgb_inner_probability,
            catboost_selector,
            cat_inner_probability,
            inner_x_train,
            inner_x_valid,
            inner_y_train,
            inner_y_valid,
        )
        gc.collect()

        nested_selection = {
            **plan["evidence"],
            "budgets": {
                "residual_mlp": mlp_budget,
                "xgboost": xgboost_budget,
                "catboost": catboost_budget,
            },
            "all_budgets_frozen_before_outer_prediction": True,
            "refit_uses_complete_outer_train": True,
            "outer_validation_labels_used_for_selection": False,
        }

        mlp_valid, mlp_test_fold, mlp_refit_record = _refit_may2022_mlp_fold(
            feature_tensor=feature_tensor,
            test_tensor=test_tensor,
            train_index=train_index,
            valid_index=valid_index,
            target=target,
            mean=fold_mean,
            scale=fold_scale,
            args=args,
            seed=fold_seed + 1_000_003,
            fold=fold,
            frozen_epochs=int(mlp_budget["frozen_refit_budget"]),
            checkpoint_path=task_dir / f"may2022_residual_mlp_fold_{fold:02d}.pt",
            logger=logger,
        )
        mlp_oof[valid_index] = mlp_valid
        mlp_test += mlp_test_fold / fold_count
        component_oof_coverage["residual_mlp"][valid_index] += 1

        x_train = train_features.iloc[train_index]
        x_valid = train_features.iloc[valid_index]
        y_train = target[train_index]
        y_valid = target[valid_index]
        xgboost_model = XGBClassifier(
            **build_may2022_xgboost_parameters(
                args,
                seed=fold_seed,
                estimator_budget=int(xgboost_budget["frozen_refit_budget"]),
                selection=False,
            )
        )
        xgboost_model.fit(x_train, y_train, verbose=False)
        xgb_valid = xgboost_model.predict_proba(x_valid)[:, 1]
        xgb_test_fold = xgboost_model.predict_proba(test_features)[:, 1]
        xgboost_oof[valid_index] = xgb_valid
        xgboost_test += xgb_test_fold / fold_count
        component_oof_coverage["xgboost"][valid_index] += 1
        xgboost_path = task_dir / f"may2022_xgboost_fold_{fold:02d}.json"
        xgboost_model.save_model(str(xgboost_path))
        xgboost_auc = float(compute_metric("roc_auc", y_valid, xgb_valid))
        del xgboost_model, xgb_valid, xgb_test_fold
        gc.collect()

        catboost_model = CatBoostClassifier(
            **build_may2022_catboost_parameters(
                args,
                seed=fold_seed + 17,
                iteration_budget=int(catboost_budget["frozen_refit_budget"]),
                selection=False,
            )
        )
        catboost_model.fit(x_train, y_train, use_best_model=False)
        cat_valid = catboost_model.predict_proba(x_valid)[:, 1]
        cat_test_fold = catboost_model.predict_proba(test_features)[:, 1]
        catboost_oof[valid_index] = cat_valid
        catboost_test += cat_test_fold / fold_count
        component_oof_coverage["catboost"][valid_index] += 1
        catboost_path = task_dir / f"may2022_catboost_fold_{fold:02d}.cbm"
        catboost_model.save_model(str(catboost_path))
        catboost_auc = float(compute_metric("roc_auc", y_valid, cat_valid))
        mlp_auc = float(compute_metric("roc_auc", y_valid, mlp_valid))
        fold_record = {
            "fold": fold,
            "seed": fold_seed,
            "train_rows": len(train_index),
            "valid_rows": len(valid_index),
            "mlp_auc": mlp_auc,
            "mlp_best_epoch": int(mlp_budget["selected_iteration"]),
            "mlp_epochs_ran": int(mlp_budget["selection_epochs_ran"]),
            "mlp_history": mlp_budget["selection_history"],
            **mlp_refit_record,
            "xgboost_auc": xgboost_auc,
            "xgboost_best_iteration": xgboost_best,
            "xgboost_refit_estimators": int(xgboost_budget["frozen_refit_budget"]),
            "catboost_auc": catboost_auc,
            "catboost_best_iteration": catboost_best,
            "catboost_refit_iterations": int(catboost_budget["frozen_refit_budget"]),
            "nested_selection": nested_selection,
        }
        fold_records.append(fold_record)
        logger.info(
            "[%s] fold=%d mlp_auc=%.7f xgboost_auc=%.7f catboost_auc=%.7f",
            competition_id,
            fold,
            mlp_auc,
            xgboost_auc,
            catboost_auc,
        )
        oof_coverage[valid_index] += 1
        del (
            catboost_model, cat_valid, cat_test_fold, x_train, x_valid, y_train, y_valid,
            mlp_valid, mlp_test_fold, fold_mean, fold_scale, inner_mean, inner_scale
        )
        gc.collect()

    if not np.all(oof_coverage == 1):
        raise RuntimeError("May 2022 OOF coverage must equal one for every training row")
    if any(not np.all(coverage == 1) for coverage in component_oof_coverage.values()):
        raise RuntimeError("May 2022 component OOF coverage must equal one for every row")
    if any(
        not np.isfinite(values).all()
        for values in (mlp_oof, xgboost_oof, catboost_oof, mlp_test, xgboost_test, catboost_test)
    ):
        raise RuntimeError("May 2022 nested refit produced non-finite predictions")
    nested_selection_contract = validate_may2022_nested_selection_contract(
        fold_records,
        oof_coverage,
        expected_rows=len(train),
        expected_folds=fold_count,
    )
    nested_selection_path = task_dir / "may2022_nested_selection.json"
    wave0.write_json(
        nested_selection_path,
        {
            "schema": "evomind.mlebench.may2022_nested_selection.v1",
            "seed": args.seed,
            "inner_validation_fraction": MAY2022_INNER_VALIDATION_FRACTION,
            "contract": nested_selection_contract,
            "folds": [record["nested_selection"] for record in fold_records],
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "claim_boundary": "Nested public OOF selection evidence is not an official medal.",
        },
    )
    nested_indices_path = task_dir / "may2022_nested_fold_indices.npz"
    nested_selection_sha256 = _may2022_sha256(nested_selection_path)
    nested_indices_sha256 = _may2022_sha256(nested_indices_path)
    del feature_tensor, test_tensor
    torch.cuda.empty_cache()
    oof_components = {
        "residual_mlp": mlp_oof,
        "xgboost": xgboost_oof,
        "catboost": catboost_oof,
    }
    test_components = {
        "residual_mlp": mlp_test,
        "xgboost": xgboost_test,
        "catboost": catboost_test,
    }
    oof_probability, crossfit_blends = cross_fit_may2022_multimodel_blend(
        oof_components, target, fold_assignment
    )
    blend = select_may2022_multimodel_blend(oof_components, target)
    deployment_oof = score_may2022_deployment_blend_oof(
        oof_components, target, fold_assignment, blend
    )
    test_probability = apply_may2022_multimodel_blend(test_components, blend)
    cv_score = float(compute_metric("roc_auc", target, oof_probability))
    crossfit_fold_auc: list[float] = []
    for fold_record in fold_records:
        fold_mask = fold_assignment == (fold_record["fold"] - 1)
        fold_record["blend_auc"] = float(
            compute_metric("roc_auc", target[fold_mask], oof_probability[fold_mask])
        )
        fold_record["deployment_blend_auc"] = float(
            deployment_oof["fold_auc"][str(fold_record["fold"] - 1)]
        )
        crossfit_fold_auc.append(float(fold_record["blend_auc"]))
    deployment_fold_auc = [
        float(deployment_oof["fold_auc"][str(fold)])
        for fold in sorted(int(value) for value in np.unique(fold_assignment))
    ]
    seed_gate = evaluate_may2022_confirmation_gate([{
        "seed": args.seed,
        "aggregate_oof_auc": deployment_oof["oof_auc"],
        "confirmation_oof_auc": deployment_oof["oof_auc"],
        "fold_auc": deployment_fold_auc,
    }])
    wave0.write_json(task_dir / "may2022_promotion_gate.json", seed_gate)
    sample = align_scalar_submission_by_id(
        sample,
        test["id"],
        test_probability,
        id_column="id",
        target_column="target",
    )
    np.savez_compressed(
        task_dir / "may2022_oof_ensemble.npz",
        id=train["id"].to_numpy(),
        target=target,
        fold_assignment=fold_assignment,
        mlp_oof=mlp_oof,
        xgboost_oof=xgboost_oof,
        catboost_oof=catboost_oof,
        oof_probability=oof_probability,
        mlp_test=mlp_test,
        xgboost_test=xgboost_test,
        catboost_test=catboost_test,
        test_probability=test_probability,
        test_id=test["id"].to_numpy(),
        oof_coverage=oof_coverage,
        mlp_oof_coverage=component_oof_coverage["residual_mlp"],
        xgboost_oof_coverage=component_oof_coverage["xgboost"],
        catboost_oof_coverage=component_oof_coverage["catboost"],
    )
    wave0.write_json(
        task_dir / "may2022_ensemble_diagnostics.json",
        {
            "feature_engineering": feature_diagnostics,
            "folds": fold_records,
            "blend": blend,
            "crossfit_blends": crossfit_blends,
            "crossfit_oof_auc": cv_score,
            "deployment_blend_oof_auc": deployment_oof["oof_auc"],
            "deployment_blend_fold_auc": deployment_oof["fold_auc"],
            "single_seed_gate": seed_gate,
            "nested_selection_contract": nested_selection_contract,
            "nested_selection_path": str(nested_selection_path),
            "nested_selection_sha256": nested_selection_sha256,
            "nested_fold_indices_path": str(nested_indices_path),
            "nested_fold_indices_sha256": nested_indices_sha256,
        },
    )
    budget = {
        "seed": args.seed,
        "folds": fold_count,
        "train_rows_available": available_rows,
        "train_rows_used": len(train),
        "test_rows": len(test),
        **full_data_contract,
        "feature_count": len(train_features.columns),
        "precomputed_public_cache_used": cache_bundle is not None,
        "precomputed_public_cache_manifest": (
            cache_bundle["manifest_record"] if cache_bundle is not None else None
        ),
        "mlp_epochs": args.may_mlp_epochs,
        "mlp_width": args.may_mlp_width,
        "mlp_blocks": args.may_mlp_blocks,
        "mlp_batch_size": args.may_mlp_batch_size,
        "mlp_learning_rate": args.may_mlp_learning_rate,
        "mlp_patience": args.may_mlp_patience,
        "xgboost_estimators": args.may_xgb_estimators,
        "xgboost_depth": args.may_xgb_depth,
        "catboost_iterations": args.may_catboost_iterations,
        "catboost_depth": args.may_catboost_depth,
        "learning_rate": args.may_learning_rate,
        "early_stopping_rounds": args.may_early_stopping,
        "inner_validation_fraction": MAY2022_INNER_VALIDATION_FRACTION,
        "nested_iteration_budget_selection": True,
        "outer_validation_labels_used_for_iteration_selection": False,
        "fixed_budget_complete_outer_train_refit": True,
        "nested_selection_contract": nested_selection_contract,
        "nested_selection_path": str(nested_selection_path),
        "nested_selection_sha256": nested_selection_sha256,
        "nested_fold_indices_path": str(nested_indices_path),
        "nested_fold_indices_sha256": nested_indices_sha256,
        "exact_oof_coverage": True,
        "component_oof_coverage": {
            name: {
                "minimum": int(coverage.min()),
                "maximum": int(coverage.max()),
                "sha256": hashlib.sha256(coverage.tobytes()).hexdigest(),
            }
            for name, coverage in component_oof_coverage.items()
        },
        "blend": blend,
        "full_fold_ensemble": True,
        "cross_fitted_meta_validation": True,
        "exact_deployment_blend_oof_scored": True,
        "single_seed_promotion_eligible": seed_gate["seed_records"][0]["eligible"],
        "three_seed_confirmation_pending": not seed_gate["passed"],
        "explicit_id_join": True,
    }
    private_grade_gate = build_may2022_run_promotion_gate(
        seed_gate=seed_gate,
        full_data_contract=full_data_contract,
        deployment_oof_auc=deployment_oof["oof_auc"],
        deployment_fold_auc=deployment_fold_auc,
    )
    private_grade_gate["checks"].update(
        {
            "nested_iteration_budget_selection": nested_selection_contract["passed"] is True,
            "outer_validation_labels_excluded_from_iteration_selection": (
                nested_selection_contract["outer_validation_labels_used_for_selection"] is False
            ),
            "fixed_budget_complete_outer_train_refit": (
                nested_selection_contract["refit_uses_complete_outer_train"] is True
            ),
            "exact_component_oof_coverage": all(
                np.all(coverage == 1) for coverage in component_oof_coverage.values()
            ),
        }
    )
    private_grade_gate["passed"] = bool(all(private_grade_gate["checks"].values()))
    private_grade_gate["nested_selection_evidence"] = {
        "path": str(nested_selection_path),
        "sha256": nested_selection_sha256,
        "fold_indices_path": str(nested_indices_path),
        "fold_indices_sha256": nested_indices_sha256,
        "contract": nested_selection_contract,
    }
    result = wave0.finalize_scored_task(
        competition_id=competition_id,
        submission=sample,
        cv_score=cv_score,
        args=args,
        task_dir=task_dir,
        budget=budget,
        promotion_gate=private_grade_gate,
        extra={
            "runtime_seconds_model": time.perf_counter() - started,
            "model_family": (
                f"full_data_{fold_count}fold_nested_budget_refit_residual_MLP_"
                "XGBoost_GPU_CatBoost_GPU_raw_or_rank_OOF_blend"
            ),
            "feature_engineering": feature_diagnostics,
            "precomputed_public_cache_used": cache_bundle is not None,
            "precomputed_public_cache_manifest": (
                cache_bundle["manifest_record"] if cache_bundle is not None else None
            ),
            "folds": fold_records,
            "blend": blend,
            "deployment_blend_oof_auc": deployment_oof["oof_auc"],
            "confirmation_gate": seed_gate,
            "nested_selection_contract": nested_selection_contract,
            "nested_selection_evidence": private_grade_gate["nested_selection_evidence"],
            "crossfit_blends": crossfit_blends,
            "budget": budget,
        },
    )
    plan_value = getattr(args, "optimization_plan", None)
    artifact_manifest = write_may2022_artifact_manifest(
        task_dir,
        source_paths=(Path(__file__), Path(wave0.__file__), Path(wave2.__file__)),
        optimization_plan=Path(plan_value) if plan_value else None,
    )
    result["artifact_manifest"] = artifact_manifest
    return result


def build_denoising_unet(base_channels: int = 32):
    import torch
    from torch import nn
    from torch.nn import functional as functional

    class Block(nn.Module):
        def __init__(self, input_channels: int, output_channels: int) -> None:
            super().__init__()
            groups = min(8, output_channels)
            while output_channels % groups:
                groups -= 1
            self.layers = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 3, padding=1),
                nn.GroupNorm(groups, output_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(output_channels, output_channels, 3, padding=1),
                nn.GroupNorm(groups, output_channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, value):
            return self.layers(value)

    class ResidualUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder1 = Block(1, base_channels)
            self.encoder2 = Block(base_channels, base_channels * 2)
            self.encoder3 = Block(base_channels * 2, base_channels * 4)
            self.bottleneck = Block(base_channels * 4, base_channels * 8)
            self.decoder3 = Block(base_channels * 12, base_channels * 4)
            self.decoder2 = Block(base_channels * 6, base_channels * 2)
            self.decoder1 = Block(base_channels * 3, base_channels)
            self.output = nn.Conv2d(base_channels, 1, 1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, value):
            encoder1 = self.encoder1(value)
            encoder2 = self.encoder2(self.pool(encoder1))
            encoder3 = self.encoder3(self.pool(encoder2))
            bottleneck = self.bottleneck(self.pool(encoder3))
            decoder3 = functional.interpolate(
                bottleneck, size=encoder3.shape[-2:], mode="bilinear", align_corners=False
            )
            decoder3 = self.decoder3(torch.cat([decoder3, encoder3], dim=1))
            decoder2 = functional.interpolate(
                decoder3, size=encoder2.shape[-2:], mode="bilinear", align_corners=False
            )
            decoder2 = self.decoder2(torch.cat([decoder2, encoder2], dim=1))
            decoder1 = functional.interpolate(
                decoder2, size=encoder1.shape[-2:], mode="bilinear", align_corners=False
            )
            decoder1 = self.decoder1(torch.cat([decoder1, encoder1], dim=1))
            return torch.tanh(self.output(decoder1)) * 0.5

    return ResidualUNet()


def _restore_image(model: Any, dirty: np.ndarray, *, tta: bool) -> np.ndarray:
    import torch
    from torch.nn import functional

    height, width = dirty.shape
    pad_height = (-height) % 8
    pad_width = (-width) % 8
    tensor = (
        torch.from_numpy(dirty.astype(np.float32))[None, None]
        .cuda()
        .contiguous(memory_format=torch.channels_last)
    )
    if pad_height or pad_width:
        tensor = functional.pad(tensor, (0, pad_width, 0, pad_height), mode="reflect")
    variants = [(tensor, ())]
    if tta:
        variants.extend([
            (torch.flip(tensor, dims=[3]), (3,)),
            (torch.flip(tensor, dims=[2]), (2,)),
            (torch.flip(tensor, dims=[2, 3]), (2, 3)),
        ])
    restored = []
    model.eval()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.inference_mode():
        for variant, flip_dims in variants:
            variant = variant.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                prediction = torch.clamp(variant + model(variant), 0.0, 1.0)
            if flip_dims:
                prediction = torch.flip(prediction, dims=list(flip_dims))
            restored.append(prediction.float())
    result = torch.stack(restored).mean(dim=0)[0, 0, :height, :width]
    return result.cpu().numpy()


def _read_gray(path: Path) -> np.ndarray:
    from PIL import Image

    if not path.is_file():
        raise FileNotFoundError(f"Grayscale image is missing: {path}")
    with Image.open(path) as image:
        values = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    if values.ndim != 2 or not np.isfinite(values).all():
        raise RuntimeError(f"Grayscale image is invalid: {path}")
    return values


def _train_denoising_model(
    *,
    names: list[str],
    dirty_dir: Path,
    clean_dir: Path,
    args: Any,
    epochs: int,
    seed: int,
    logger: Any,
    validation_names: list[str] | None = None,
    checkpoint_path: Path | None = None,
):
    import torch
    from torch.nn import functional
    from torch.utils.data import DataLoader, Dataset

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    fast_kernel_mode = bool(args.wave2_fast_kernels)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = fast_kernel_mode
    torch.backends.cudnn.deterministic = not fast_kernel_mode
    torch.use_deterministic_algorithms(not fast_kernel_mode, warn_only=True)
    if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    dirty_images = [_read_gray(dirty_dir / name) for name in names]
    clean_images = [_read_gray(clean_dir / name) for name in names]
    for name, dirty, clean in zip(names, dirty_images, clean_images):
        if dirty.shape != clean.shape:
            raise RuntimeError(f"Image pair shape mismatch: {name}")

    class PairedPatches(Dataset):
        def __init__(self) -> None:
            generator = np.random.default_rng(seed)
            records: list[tuple[int, int, int, bool, bool]] = []
            for image_index, dirty in enumerate(dirty_images):
                height, width = dirty.shape
                if height < args.denoising_patch_size or width < args.denoising_patch_size:
                    raise RuntimeError("Denoising patch size exceeds a training image")
                for _ in range(args.denoising_patches_per_image):
                    top = int(generator.integers(0, height - args.denoising_patch_size + 1))
                    left = int(generator.integers(0, width - args.denoising_patch_size + 1))
                    records.append((image_index, top, left,
                                    bool(generator.integers(0, 2)), bool(generator.integers(0, 2))))
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int):
            image_index, top, left, flip_h, flip_v = self.records[index]
            size = args.denoising_patch_size
            dirty = dirty_images[image_index][top:top + size, left:left + size]
            clean = clean_images[image_index][top:top + size, left:left + size]
            if flip_h:
                dirty, clean = np.flip(dirty, 1), np.flip(clean, 1)
            if flip_v:
                dirty, clean = np.flip(dirty, 0), np.flip(clean, 0)
            return (
                torch.from_numpy(np.ascontiguousarray(dirty))[None],
                torch.from_numpy(np.ascontiguousarray(clean))[None],
            )

    workers = min(args.denoising_workers, max(0, (os.cpu_count() or 4) // 4))
    generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": args.denoising_batch_size,
        "shuffle": True,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
        "generator": generator,
        "worker_init_fn": wave2._seed_vision_worker,
    }
    if workers > 0:
        loader_options["prefetch_factor"] = wave2.VISION_DATALOADER_PREFETCH_FACTOR
    loader = DataLoader(
        PairedPatches(),
        **loader_options,
    )
    model = build_denoising_unet(args.denoising_base_channels).to(
        device="cuda", memory_format=torch.channels_last
    )
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.denoising_learning_rate,
            weight_decay=1e-4,
            fused=True,
        )
        fused_adamw = True
    except (RuntimeError, TypeError):
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.denoising_learning_rate, weight_decay=1e-4
        )
        fused_adamw = False
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = amp_dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    if validation_names and checkpoint_path is None:
        raise ValueError("A checkpoint path is required for denoising validation")
    history: list[dict[str, Any]] = []
    best_epoch = epochs
    best_score: float | None = None
    epochs_without_improvement = 0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        torch.cuda.synchronize()
        epoch_started = time.perf_counter()
        model.train()
        total_loss = 0.0
        seen = 0
        for dirty, clean in loader:
            dirty = dirty.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
            clean = clean.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                prediction = torch.clamp(dirty + model(dirty), 0.0, 1.0)
                pixel_loss = 0.65 * functional.l1_loss(prediction, clean) + 0.35 * functional.mse_loss(
                    prediction, clean
                )
                horizontal = functional.l1_loss(
                    prediction[..., :, 1:] - prediction[..., :, :-1],
                    clean[..., :, 1:] - clean[..., :, :-1],
                )
                vertical = functional.l1_loss(
                    prediction[..., 1:, :] - prediction[..., :-1, :],
                    clean[..., 1:, :] - clean[..., :-1, :],
                )
                loss = pixel_loss + 0.08 * (horizontal + vertical)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach()) * len(dirty)
            seen += len(dirty)
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - epoch_started
        scheduler.step()
        epoch_loss = total_loss / max(1, seen)
        record = {
            "epoch": float(epoch + 1),
            "train_loss": epoch_loss,
            "train_seconds": train_seconds,
            "train_patches_per_second": seen / max(train_seconds, 1e-9),
        }
        if validation_names:
            squared_error = 0.0
            pixel_count = 0
            for name in validation_names:
                dirty = _read_gray(dirty_dir / name)
                clean = _read_gray(clean_dir / name)
                prediction = _restore_image(model, dirty, tta=True)
                squared_error += float(np.square(prediction - clean).sum())
                pixel_count += clean.size
            validation_rmse = math.sqrt(squared_error / max(1, pixel_count))
            record["validation_rmse"] = validation_rmse
            if best_score is None or validation_rmse < best_score - 1e-8:
                best_score = validation_rmse
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                epochs_without_improvement += 1
            log_suffix = f" valid_rmse={validation_rmse:.7f}"
        else:
            log_suffix = ""
        torch.cuda.synchronize()
        record["epoch_seconds"] = time.perf_counter() - epoch_started
        record["peak_memory_allocated_mib"] = int(
            torch.cuda.max_memory_allocated() / 2**20
        )
        logger.info(
            "[denoising-dirty-documents] epoch=%d train_loss=%.7f "
            "patches_per_second=%.1f epoch_seconds=%.1f%s",
            epoch + 1,
            epoch_loss,
            record["train_patches_per_second"],
            record["epoch_seconds"],
            log_suffix,
        )
        history.append(record)
        if validation_names and epochs_without_improvement >= args.denoising_patience:
            logger.info(
                "[denoising-dirty-documents] early_stop epoch=%d best_epoch=%d",
                epoch + 1,
                best_epoch,
            )
            break
    if validation_names:
        assert checkpoint_path is not None and best_score is not None
        model.load_state_dict(torch.load(checkpoint_path, map_location="cuda", weights_only=True))
    telemetry = {
        "amp_dtype": str(amp_dtype).removeprefix("torch."),
        "channels_last": True,
        "cudnn_benchmark": fast_kernel_mode,
        "deterministic_algorithms_requested": not fast_kernel_mode,
        "dataloader_workers": workers,
        "dataloader_prefetch_factor": (
            wave2.VISION_DATALOADER_PREFETCH_FACTOR if workers > 0 else None
        ),
        "fused_adamw": fused_adamw,
        "peak_memory_allocated_mib": int(torch.cuda.max_memory_allocated() / 2**20),
    }
    return model, history, best_epoch, best_score, telemetry


def validate_denoising_submission_coordinates(
    sample: pd.DataFrame, test_dir: Path
) -> pd.DataFrame:
    """Validate exact one-based, full-image pixel coverage for every test document."""

    if "id" not in sample or sample["id"].duplicated().any():
        raise RuntimeError("Denoising submission IDs are missing or duplicated")
    parts = sample["id"].astype(str).str.rsplit("_", n=2, expand=True)
    if parts.shape[1] != 3 or bool((parts[0].astype(str).str.len() == 0).any()):
        raise RuntimeError("Unexpected denoising submission id format")
    try:
        rows = pd.to_numeric(parts[1], errors="raise").astype(np.int64)
        columns = pd.to_numeric(parts[2], errors="raise").astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Denoising submission coordinates must be integers") from exc
    coordinates = pd.DataFrame({
        "sample_position": np.arange(len(sample), dtype=np.int64),
        "image_id": parts[0].astype(str).to_numpy(),
        "row": rows.to_numpy(),
        "column": columns.to_numpy(),
    })
    expected_images = {path.stem for path in test_dir.glob("*.png") if path.is_file()}
    observed_images = set(coordinates["image_id"])
    if not expected_images or observed_images != expected_images:
        raise RuntimeError("Denoising submission/test image ID sets differ")
    for image_id, group in coordinates.groupby("image_id", sort=False):
        image = _read_gray(test_dir / f"{image_id}.png")
        height, width = image.shape
        row = group["row"].to_numpy()
        column = group["column"].to_numpy()
        if (
            np.any(row < 1)
            or np.any(row > height)
            or np.any(column < 1)
            or np.any(column > width)
        ):
            raise RuntimeError("Denoising submission coordinate is out of bounds")
        linear = (row - 1) * width + (column - 1)
        if len(group) != height * width or len(np.unique(linear)) != height * width:
            raise RuntimeError("Denoising submission does not cover every pixel exactly once")
    return coordinates


def build_denoising_nested_document_folds(
    names: list[str], *, requested_folds: int, seed: int
) -> list[dict[str, Any]]:
    """Create untouched outer document folds plus inner checkpoint folds."""

    from sklearn.model_selection import KFold

    ordered = np.asarray(sorted(str(name) for name in names), dtype=object)
    if len(set(ordered.tolist())) != len(ordered):
        raise RuntimeError("Denoising document names are duplicated")
    outer_count = min(int(requested_folds), len(ordered))
    if outer_count < 2:
        raise RuntimeError("Denoising recovery requires at least two document folds")
    outer = KFold(n_splits=outer_count, shuffle=True, random_state=int(seed))
    records: list[dict[str, Any]] = []
    outer_coverage: Counter[str] = Counter()
    for fold, (outer_fit_index, outer_valid_index) in enumerate(outer.split(ordered), start=1):
        outer_fit = ordered[outer_fit_index]
        outer_valid = ordered[outer_valid_index]
        inner_count = min(5, len(outer_fit))
        if inner_count < 2:
            raise RuntimeError("Denoising outer fit has too few documents for inner selection")
        inner = KFold(
            n_splits=inner_count,
            shuffle=True,
            random_state=int(seed + fold * 10_007),
        )
        inner_fit_index, inner_valid_index = next(inner.split(outer_fit))
        record = {
            "fold": fold,
            "outer_fit": outer_fit.tolist(),
            "outer_valid": outer_valid.tolist(),
            "inner_fit": outer_fit[inner_fit_index].tolist(),
            "inner_checkpoint": outer_fit[inner_valid_index].tolist(),
        }
        if set(record["outer_fit"]) & set(record["outer_valid"]):
            raise RuntimeError("Denoising outer document folds overlap")
        if set(record["inner_fit"]) & set(record["inner_checkpoint"]):
            raise RuntimeError("Denoising inner document folds overlap")
        if set(record["inner_fit"]) | set(record["inner_checkpoint"]) != set(record["outer_fit"]):
            raise RuntimeError("Denoising inner folds do not partition the outer fit")
        outer_coverage.update(record["outer_valid"])
        records.append(record)
    if set(outer_coverage) != set(ordered.tolist()) or any(
        count != 1 for count in outer_coverage.values()
    ):
        raise RuntimeError("Denoising outer OOF folds do not cover every document exactly once")
    return records


def run_denoising_unet(args: Any, task_dir: Path, logger: Any) -> dict[str, Any]:
    import torch

    competition_id = "denoising-dirty-documents"
    resolved = resolve_competition(competition_id, args.data_root)
    dirty_dir = resolved.public_dir / "train"
    clean_dir = resolved.public_dir / "train_cleaned"
    test_dir = resolved.public_dir / "test"
    dirty_names = {path.name for path in dirty_dir.glob("*.png") if path.is_file()}
    clean_names = {path.name for path in clean_dir.glob("*.png") if path.is_file()}
    if not dirty_names or dirty_names != clean_names:
        raise RuntimeError("Denoising dirty/clean document manifests differ")
    names = sorted(dirty_names)
    fold_count = min(int(args.denoising_folds), len(names))
    if fold_count < 2:
        raise RuntimeError("Denoising recovery requires at least two document folds")
    sample = pd.read_csv(resolved.sample_submission_path).reset_index(drop=True)
    coordinates = validate_denoising_submission_coordinates(sample, test_dir)
    test_image_ids = list(dict.fromkeys(coordinates["image_id"].tolist()))
    test_dirty = {image_id: _read_gray(test_dir / f"{image_id}.png") for image_id in test_image_ids}
    test_sum_no_tta = {
        image_id: np.zeros_like(image, dtype=np.float64)
        for image_id, image in test_dirty.items()
    }
    test_sum_tta = {
        image_id: np.zeros_like(image, dtype=np.float64)
        for image_id, image in test_dirty.items()
    }
    nested_folds = build_denoising_nested_document_folds(
        names, requested_folds=fold_count, seed=args.seed
    )
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    oof_dir = task_dir / "denoising_oof_restorations"
    oof_dir.mkdir(parents=True, exist_ok=True)
    fold_assignment: dict[str, int] = {}
    fold_records: list[dict[str, Any]] = []
    performance_records: list[dict[str, Any]] = []
    document_records: list[dict[str, Any]] = []
    global_squared_error_no_tta = 0.0
    global_squared_error_tta = 0.0
    global_pixel_count = 0
    for fold_spec in nested_folds:
        fold = int(fold_spec["fold"])
        fit_names = list(fold_spec["outer_fit"])
        valid_names = list(fold_spec["outer_valid"])
        inner_fit_names = list(fold_spec["inner_fit"])
        inner_checkpoint_names = list(fold_spec["inner_checkpoint"])
        for name in valid_names:
            if name in fold_assignment:
                raise RuntimeError("Denoising document appears in multiple validation folds")
            fold_assignment[name] = fold
        selection_checkpoint = task_dir / f"denoising_selection_fold_{fold:02d}.pt"
        (
            selection_model,
            selection_history,
            best_epoch,
            best_score,
            selection_performance,
        ) = _train_denoising_model(
            names=inner_fit_names,
            dirty_dir=dirty_dir,
            clean_dir=clean_dir,
            args=args,
            epochs=args.denoising_epochs,
            seed=args.seed + fold * 101,
            logger=logger,
            validation_names=inner_checkpoint_names,
            checkpoint_path=selection_checkpoint,
        )
        if best_score is None:
            raise RuntimeError("Denoising inner fold did not produce a selection checkpoint")
        del selection_model
        torch.cuda.empty_cache()
        (
            model,
            refit_history,
            refit_epochs,
            refit_score,
            refit_performance,
        ) = _train_denoising_model(
            names=fit_names,
            dirty_dir=dirty_dir,
            clean_dir=clean_dir,
            args=args,
            epochs=best_epoch,
            seed=args.seed + fold * 101 + 50_000,
            logger=logger,
        )
        if refit_epochs != best_epoch or refit_score is not None:
            raise RuntimeError("Denoising fixed-epoch outer refit contract failed")
        refit_checkpoint = task_dir / f"denoising_refit_fold_{fold:02d}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "fold": fold,
                "selected_epoch": best_epoch,
                "selection_seed": args.seed + fold * 101,
                "refit_seed": args.seed + fold * 101 + 50_000,
                "outer_validation_documents_used_for_selection": False,
            },
            refit_checkpoint,
        )
        fold_squared_error_no_tta = 0.0
        fold_squared_error_tta = 0.0
        fold_pixel_count = 0
        for name in valid_names:
            dirty = _read_gray(dirty_dir / name)
            clean = _read_gray(clean_dir / name)
            prediction_no_tta = _restore_image(model, dirty, tta=False)
            prediction_tta = _restore_image(model, dirty, tta=True)
            if (
                prediction_no_tta.shape != clean.shape
                or prediction_tta.shape != clean.shape
                or not np.isfinite(prediction_no_tta).all()
                or not np.isfinite(prediction_tta).all()
            ):
                raise RuntimeError(f"Denoising OOF restoration is invalid: {name}")
            squared_error_no_tta = float(np.square(prediction_no_tta - clean).sum())
            squared_error_tta = float(np.square(prediction_tta - clean).sum())
            pixel_count = int(clean.size)
            fold_squared_error_no_tta += squared_error_no_tta
            fold_squared_error_tta += squared_error_tta
            fold_pixel_count += pixel_count
            global_squared_error_no_tta += squared_error_no_tta
            global_squared_error_tta += squared_error_tta
            global_pixel_count += pixel_count
            np.savez_compressed(
                oof_dir / f"{Path(name).stem}.npz",
                no_tta=prediction_no_tta.astype(np.float32),
                tta=prediction_tta.astype(np.float32),
            )
            document_records.append({
                "document": name,
                "fold": fold,
                "pixels": pixel_count,
                "rmse_no_tta": math.sqrt(squared_error_no_tta / pixel_count),
                "rmse_tta": math.sqrt(squared_error_tta / pixel_count),
            })
        for image_id, dirty in test_dirty.items():
            test_sum_no_tta[image_id] += _restore_image(model, dirty, tta=False)
            test_sum_tta[image_id] += _restore_image(model, dirty, tta=True)
        fold_records.append({
            "fold": fold,
            "train_documents": len(fit_names),
            "validation_documents": len(valid_names),
            "inner_train_documents": len(inner_fit_names),
            "inner_checkpoint_documents": len(inner_checkpoint_names),
            "best_epoch": best_epoch,
            "inner_checkpoint_validation_rmse_tta": float(best_score),
            "outer_validation_rmse_no_tta": math.sqrt(
                fold_squared_error_no_tta / fold_pixel_count
            ),
            "outer_validation_rmse_tta": math.sqrt(
                fold_squared_error_tta / fold_pixel_count
            ),
            "outer_validation_used_for_checkpoint_selection": False,
            "selection_history": selection_history,
            "refit_history": refit_history,
            "selection_performance": selection_performance,
            "refit_performance": refit_performance,
        })
        performance_records.append({
            "fold": fold,
            "selection": selection_performance,
            "refit": refit_performance,
        })
        del model
        torch.cuda.empty_cache()
    if set(fold_assignment) != set(names):
        raise RuntimeError("Denoising OOF folds did not cover every training document")
    cv_no_tta = math.sqrt(global_squared_error_no_tta / global_pixel_count)
    cv_tta = math.sqrt(global_squared_error_tta / global_pixel_count)
    selected_inference_mode = "tta" if cv_tta <= cv_no_tta else "no_tta"
    cv_score = cv_tta if selected_inference_mode == "tta" else cv_no_tta
    pd.DataFrame({
        "document": names,
        "fold": [fold_assignment[name] for name in names],
    }).to_csv(task_dir / "denoising_fold_manifest.csv", index=False)
    pd.DataFrame(document_records).to_csv(task_dir / "denoising_oof_metrics.csv", index=False)
    predicted_values = np.empty(len(sample), dtype=np.float32)
    for image_id, group in coordinates.groupby("image_id", sort=False):
        selected_sum = (
            test_sum_tta[image_id]
            if selected_inference_mode == "tta"
            else test_sum_no_tta[image_id]
        )
        prediction = (selected_sum / fold_count).astype(np.float32)
        rows = group["row"].to_numpy(dtype=np.int64) - 1
        columns = group["column"].to_numpy(dtype=np.int64) - 1
        positions = group["sample_position"].to_numpy(dtype=np.int64)
        predicted_values[positions] = prediction[rows, columns]
    if not np.isfinite(predicted_values).all():
        raise RuntimeError("Denoising inference did not produce every sample value")
    sample["value"] = np.clip(predicted_values, 0.0, 1.0)
    wave0.write_json(task_dir / "denoising_history.json", {
        "fold_contract": "inner_checkpoint_selection_then_fixed_epoch_outer_refit",
        "folds": fold_records,
        "global_full_resolution_oof_rmse_no_tta": cv_no_tta,
        "global_full_resolution_oof_rmse_tta": cv_tta,
        "selected_inference_mode": selected_inference_mode,
        "selected_global_full_resolution_oof_rmse": cv_score,
        "performance": performance_records,
    })
    budget = {
        "seed": args.seed,
        "folds": fold_count,
        "train_images": len(names),
        "epochs_requested": args.denoising_epochs,
        "early_stopping_patience": args.denoising_patience,
        "patch_size": args.denoising_patch_size,
        "patches_per_image": args.denoising_patches_per_image,
        "batch_size": args.denoising_batch_size,
        "base_channels": args.denoising_base_channels,
        "learning_rate": args.denoising_learning_rate,
        "full_resolution_oof": True,
        "full_resolution_oof_no_tta_and_tta": True,
        "nested_checkpoint_selection": True,
        "outer_validation_used_for_checkpoint_selection": False,
        "fold_checkpoint_test_ensemble": True,
        "sample_coordinate_validation": True,
        "tta_flips_compared": True,
        "selected_inference_mode": selected_inference_mode,
        "fast_kernel_mode": bool(args.wave2_fast_kernels),
        "amp_dtype": performance_records[0]["selection"]["amp_dtype"],
        "channels_last": True,
        "dataloader_prefetch_factor": performance_records[0]["selection"][
            "dataloader_prefetch_factor"
        ],
        "fused_adamw_all_runs": bool(all(
            record[stage]["fused_adamw"]
            for record in performance_records
            for stage in ("selection", "refit")
        )),
    }
    promotion_gate = wave0.build_metric_promotion_gate(
        name="denoising_full_resolution_oof_rmse",
        metric="rmse",
        direction="minimize",
        score=float(cv_score),
        threshold=0.042,
        evidence={
            "global_full_resolution_oof_rmse_no_tta": float(cv_no_tta),
            "global_full_resolution_oof_rmse_tta": float(cv_tta),
            "selected_inference_mode": selected_inference_mode,
            "fold_count": fold_count,
        },
    )
    return wave0.finalize_scored_task(
        competition_id=competition_id,
        submission=sample,
        cv_score=float(cv_score),
        args=args,
        task_dir=task_dir,
        budget=budget,
        promotion_gate=promotion_gate,
        extra={
            "runtime_seconds_model": time.perf_counter() - started,
            "model_family": (
                f"{fold_count}fold_nested_document_OOF_paired_patch_residual_"
                f"UNet_L1_MSE_edge_{selected_inference_mode}"
            ),
            "folds": fold_records,
            "full_resolution_oof_rmse_no_tta": cv_no_tta,
            "full_resolution_oof_rmse_tta": cv_tta,
            "torch_peak_memory_allocated_mib": int(torch.cuda.max_memory_allocated() / 2**20),
            "budget": budget,
        },
    )


def siim_metadata_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build aligned, non-target SIIM metadata features for image fusion/blending."""

    required = {"image_name", "sex", "age_approx", "anatom_site_general_challenge"}
    missing = sorted((required - set(train.columns)) | (required - set(test.columns)))
    if missing:
        raise RuntimeError(f"SIIM metadata columns missing: {missing}")
    combined = pd.concat(
        [train.assign(_source=0), test.assign(_source=1)], ignore_index=True, sort=False
    )
    train_rows = len(train)
    age = pd.to_numeric(combined["age_approx"], errors="coerce")
    train_age = age.iloc[:train_rows]
    age_median = float(train_age.median()) if train_age.notna().any() else 0.0
    age_mean = float(train_age.fillna(age_median).mean())
    age_std = float(train_age.fillna(age_median).std())
    if not np.isfinite(age_std) or age_std < 1e-6:
        age_std = 1.0

    features = pd.DataFrame(index=combined.index)
    filled_age = age.fillna(age_median)
    features["age_scaled"] = (filled_age - age_mean) / age_std
    features["age_squared_scaled"] = np.square(features["age_scaled"])
    features["age_missing"] = age.isna().astype(np.float32)

    patient = combined.get("patient_id", combined["image_name"]).fillna("__missing__").astype(str)
    patient_counts = patient.map(patient.value_counts()).astype(float)
    features["patient_image_count_log1p"] = np.log1p(patient_counts)
    features["patient_has_multiple_images"] = (patient_counts > 1).astype(np.float32)

    categorical = combined[["sex", "anatom_site_general_challenge"]].copy()
    categorical = categorical.fillna("__missing__").astype(str)
    encoded = pd.get_dummies(
        categorical,
        columns=list(categorical.columns),
        prefix=["sex", "site"],
        dtype=np.float32,
    )
    features = pd.concat([features, encoded], axis=1)
    matrix = features.to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise RuntimeError("SIIM metadata feature matrix contains non-finite values")
    names = [str(column) for column in features.columns]
    return matrix[:train_rows], matrix[train_rows:], names


def select_siim_multichannel_auc_blend(
    components: dict[str, np.ndarray],
    truth: np.ndarray,
    *,
    grid_denominator: int = 20,
) -> dict[str, Any]:
    """Select a deterministic raw/rank simplex blend for two to four SIIM channels."""

    names = list(components)
    labels = np.asarray(truth, dtype=np.int8)
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in components.items()}
    if not 2 <= len(names) <= 4:
        raise ValueError("SIIM blend expects between two and four channels")
    if labels.ndim != 1 or any(value.shape != labels.shape for value in arrays.values()):
        raise ValueError("SIIM blend arrays must have identical one-dimensional shapes")
    if not np.isfinite([*labels, *(value for array in arrays.values() for value in array)]).all():
        raise ValueError("SIIM blend arrays must be finite")
    grid_denominator = int(grid_denominator)
    if grid_denominator < 1:
        raise ValueError("SIIM blend grid denominator must be positive")
    candidates: list[dict[str, Any]] = []
    for mode in ("raw", "rank"):
        values = arrays if mode == "raw" else {
            name: _rank_normalize(value) for name, value in arrays.items()
        }
        for weights in _simplex_weights(len(names), denominator=grid_denominator):
            prediction = sum(
                weight * values[name] for name, weight in zip(names, weights)
            )
            candidates.append({
                "mode": mode,
                "weights": {name: float(weight) for name, weight in zip(names, weights)},
                "oof_auc": float(compute_metric("roc_auc", labels, prediction)),
            })
    candidates.sort(key=lambda item: (
        -item["oof_auc"],
        0 if item["mode"] == "raw" else 1,
        sum(abs(weight - 1.0 / len(names)) for weight in item["weights"].values()),
    ))
    selected = dict(candidates[0])
    selected["component_oof_auc"] = {
        name: float(compute_metric("roc_auc", labels, value))
        for name, value in arrays.items()
    }
    selected["grid_denominator"] = grid_denominator
    selected["candidate_count"] = len(candidates)
    return selected


def apply_siim_multichannel_blend(
    components: dict[str, np.ndarray], blend: dict[str, Any]
) -> np.ndarray:
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in components.items()}
    if blend["mode"] == "rank":
        arrays = {name: _rank_normalize(value) for name, value in arrays.items()}
    elif blend["mode"] != "raw":
        raise ValueError(f"Unsupported SIIM blend mode: {blend['mode']}")
    expected = set(blend["weights"])
    if set(arrays) != expected:
        raise RuntimeError("SIIM blend channels differ from the fitted blend")
    prediction = sum(
        float(blend["weights"][name]) * arrays[name] for name in blend["weights"]
    )
    if not np.isfinite(prediction).all():
        raise RuntimeError("SIIM multichannel blend produced non-finite values")
    return np.clip(prediction, 1e-6, 1.0 - 1e-6)


def select_siim_auc_blend(
    image_probability: np.ndarray,
    metadata_probability: np.ndarray,
    truth: np.ndarray,
) -> tuple[np.ndarray, float, str, float]:
    components = {
        "image": np.asarray(image_probability, dtype=np.float64),
        "metadata": np.asarray(metadata_probability, dtype=np.float64),
    }
    selected = select_siim_multichannel_auc_blend(components, np.asarray(truth))
    prediction = apply_siim_multichannel_blend(components, selected)
    return (
        prediction,
        float(selected["weights"]["image"]),
        str(selected["mode"]),
        float(selected["oof_auc"]),
    )


def apply_siim_blend(
    image_probability: np.ndarray,
    metadata_probability: np.ndarray,
    *,
    image_weight: float,
    mode: str,
) -> np.ndarray:
    return apply_siim_multichannel_blend(
        {
            "image": np.asarray(image_probability, dtype=np.float64),
            "metadata": np.asarray(metadata_probability, dtype=np.float64),
        },
        {
            "mode": mode,
            "weights": {
                "image": float(image_weight),
                "metadata": 1.0 - float(image_weight),
            },
        },
    )


def build_siim_fusion_model(
    metadata_width: int,
    *,
    backbone_name: str = "convnext_small",
):
    import torch
    from torch import nn

    backbone, pretrained, weight_identity = wave2._vision_model(
        1,
        backbone=backbone_name,
    )
    backbone_spec = wave2.VISION_BACKBONE_SPECS[backbone_name]
    classifier_index = int(backbone_spec["classifier"].rsplit(".", maxsplit=1)[-1])
    image_width = int(backbone.classifier[classifier_index].in_features)
    backbone.classifier[classifier_index] = nn.Identity()

    class FusionModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.metadata = nn.Sequential(
                nn.LayerNorm(metadata_width),
                nn.Linear(metadata_width, 64),
                nn.SiLU(inplace=True),
                nn.Dropout(0.15),
                nn.Linear(64, 32),
                nn.SiLU(inplace=True),
            )
            self.image_output = nn.Sequential(
                nn.Dropout(0.25),
                nn.Linear(image_width, 128),
                nn.SiLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
            )
            self.output = nn.Sequential(
                nn.Dropout(0.25),
                nn.Linear(image_width + 32, 128),
                nn.SiLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
            )

        def forward_channels(self, image, metadata):
            image_features = self.backbone(image)
            metadata_features = self.metadata(metadata)
            image_logit = self.image_output(image_features)[:, 0]
            fusion_logit = self.output(
                torch.cat([image_features, metadata_features], dim=1)
            )[:, 0]
            return image_logit, fusion_logit

        def forward(self, image, metadata):
            _, fusion_logit = self.forward_channels(image, metadata)
            return fusion_logit

    return FusionModel(), pretrained, weight_identity


def build_siim_multiview_fusion_model(
    metadata_width: int,
    *,
    full_backbone_name: str = "convnext_small",
    lesion_backbone_name: str = "efficientnet_v2_s",
):
    """Build distinct full-image and lesion-focus backbones plus metadata fusion."""

    import torch
    from torch import nn

    if full_backbone_name == lesion_backbone_name:
        raise ValueError("SIIM multiview backbones must be distinct")

    def prepare_backbone(name: str):
        backbone, pretrained, identity = wave2._vision_model(1, backbone=name)
        spec = wave2.VISION_BACKBONE_SPECS[name]
        classifier_index = int(spec["classifier"].rsplit(".", maxsplit=1)[-1])
        width = int(backbone.classifier[classifier_index].in_features)
        backbone.classifier[classifier_index] = nn.Identity()
        return backbone, width, pretrained, identity

    full_backbone, full_width, full_pretrained, full_identity = prepare_backbone(
        full_backbone_name
    )
    lesion_backbone, lesion_width, lesion_pretrained, lesion_identity = prepare_backbone(
        lesion_backbone_name
    )

    class MultiViewFusionModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.full_backbone = full_backbone
            self.lesion_backbone = lesion_backbone
            self.metadata = nn.Sequential(
                nn.LayerNorm(metadata_width),
                nn.Linear(metadata_width, 64),
                nn.SiLU(inplace=True),
                nn.Dropout(0.15),
                nn.Linear(64, 32),
                nn.SiLU(inplace=True),
            )
            self.full_image_output = nn.Sequential(
                nn.Dropout(0.25),
                nn.Linear(full_width, 128),
                nn.SiLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
            )
            self.lesion_image_output = nn.Sequential(
                nn.Dropout(0.25),
                nn.Linear(lesion_width, 128),
                nn.SiLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
            )
            self.output = nn.Sequential(
                nn.Dropout(0.25),
                nn.Linear(full_width + lesion_width + 32, 192),
                nn.SiLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(192, 1),
            )

        def forward_channels(self, full_image, lesion_image, metadata):
            full_features = self.full_backbone(full_image)
            lesion_features = self.lesion_backbone(lesion_image)
            metadata_features = self.metadata(metadata)
            return {
                "full_image": self.full_image_output(full_features)[:, 0],
                "lesion_focus": self.lesion_image_output(lesion_features)[:, 0],
                "image_metadata_fusion": self.output(
                    torch.cat(
                        [full_features, lesion_features, metadata_features], dim=1
                    )
                )[:, 0],
            }

        def forward(self, full_image, lesion_image, metadata):
            return self.forward_channels(full_image, lesion_image, metadata)[
                "image_metadata_fusion"
            ]

    weight_identity = {
        "schema": "evomind.siim.multiview_pretrained_weights.v1",
        "full_image": {
            "backbone": full_backbone_name,
            "identity": full_identity,
        },
        "lesion_focus": {
            "backbone": lesion_backbone_name,
            "identity": lesion_identity,
        },
    }
    return (
        MultiViewFusionModel(),
        bool(full_pretrained and lesion_pretrained),
        weight_identity,
    )


def seed_siim_fold(seed: int, *, fast_kernel_mode: bool = False) -> dict[str, Any]:
    """Seed every SIIM RNG and configure either replayable or fast CUDA kernels."""

    fast_kernel_mode = bool(fast_kernel_mode)
    cublas_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if not fast_kernel_mode:
        supported_workspace_configs = {":4096:8", ":16:8"}
        if (
            cublas_workspace_config is not None
            and cublas_workspace_config not in supported_workspace_configs
        ):
            raise RuntimeError(
                "SIIM deterministic mode requires CUBLAS_WORKSPACE_CONFIG to be "
                "':4096:8' or ':16:8'; "
                f"observed={cublas_workspace_config!r}"
            )
        cublas_workspace_config = cublas_workspace_config or ":4096:8"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_workspace_config

    import torch

    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        torch.cuda.manual_seed_all(value)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = fast_kernel_mode
    torch.backends.cudnn.deterministic = not fast_kernel_mode
    torch.use_deterministic_algorithms(not fast_kernel_mode, warn_only=False)
    if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    return {
        "seed": value,
        "python_seeded": True,
        "numpy_seeded": True,
        "torch_seeded": True,
        "torch_cuda_seeded": cuda_available,
        "fast_kernel_mode": fast_kernel_mode,
        "tf32_enabled": True,
        "cudnn_benchmark": fast_kernel_mode,
        "cudnn_deterministic": not fast_kernel_mode,
        "deterministic_algorithms_requested": not fast_kernel_mode,
        "deterministic_warn_only": False,
        "cublas_workspace_config": cublas_workspace_config,
    }


def _siim_split_indices(
    metadata: pd.DataFrame, target: np.ndarray, *, holdout_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray, str]:
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split

    indices = np.arange(len(metadata))
    if "patient_id" in metadata.columns and metadata["patient_id"].notna().any():
        groups = metadata["patient_id"].fillna(metadata["image_name"]).astype(str).to_numpy()
        folds = max(2, min(10, round(1.0 / max(holdout_fraction, 0.05))))
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        try:
            train_indices, valid_indices = next(splitter.split(indices, target, groups))
            if len(np.unique(target[valid_indices])) == 2:
                return train_indices, valid_indices, f"stratified_group_{folds}_fold"
        except ValueError:
            pass
    train_indices, valid_indices = train_test_split(
        indices,
        test_size=holdout_fraction,
        random_state=seed,
        stratify=target,
    )
    return train_indices, valid_indices, "stratified_random_holdout"


def verify_siim_image_manifest(paths: list[Path], *, workers: int) -> None:
    """Decode-check every SIIM image with bounded parallel I/O."""

    from PIL import Image

    def verify(path: Path) -> None:
        try:
            with Image.open(path) as handle:
                handle.verify()
        except Exception as exc:
            raise RuntimeError(f"SIIM image decode verification failed: {path}") from exc

    worker_count = max(1, workers)
    if worker_count == 1:
        for path in paths:
            verify(path)
        return
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="siim-image-verify",
    ) as executor:
        for _ in executor.map(verify, paths):
            pass


def build_siim_image_content_manifest(
    train_paths: list[Path],
    test_paths: list[Path],
    *,
    train_ids: list[str],
    test_ids: list[str],
    workers: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Decode-check and hash every SIIM image for duplicate-aware validation.

    Exact image bytes are the only duplicate signal used here.  The hashes are
    computed without labels and may therefore be reused by every outer/inner
    split.  The returned frame preserves the caller's train/test order.
    """

    from PIL import Image

    if len(train_paths) != len(train_ids) or len(test_paths) != len(test_ids):
        raise ValueError("SIIM image paths and IDs have inconsistent lengths")
    if len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids):
        raise ValueError("SIIM image content manifest IDs must be unique per source")

    items = [
        ("train", str(image_id), Path(path))
        for image_id, path in zip(train_ids, train_paths)
    ] + [
        ("test", str(image_id), Path(path))
        for image_id, path in zip(test_ids, test_paths)
    ]

    def inspect(item: tuple[str, str, Path]) -> dict[str, Any]:
        source, image_id, path = item
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                pixel_array = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
                pixel_digest = hashlib.sha256()
                pixel_digest.update(
                    f"RGB:{rgb.width}x{rgb.height}:".encode("ascii")
                )
                pixel_digest.update(pixel_array.tobytes())
                gray = rgb.convert("L").resize(
                    (9, 8),
                    resample=Image.Resampling.LANCZOS,
                )
                gray_array = np.asarray(gray, dtype=np.uint8)
                difference = gray_array[:, 1:] > gray_array[:, :-1]
                perceptual_value = 0
                for bit in difference.reshape(-1).tolist():
                    perceptual_value = (perceptual_value << 1) | int(bit)
        except Exception as exc:
            raise RuntimeError(
                f"SIIM image content verification failed: {path}"
            ) from exc
        return {
            "source": source,
            "image_name": image_id,
            "filename": path.name,
            "bytes": int(path.stat().st_size),
            "content_sha256": digest.hexdigest(),
            "decoded_pixel_sha256": pixel_digest.hexdigest(),
            "perceptual_dhash64": f"{perceptual_value:016x}",
            "decoded_width": int(pixel_array.shape[1]),
            "decoded_height": int(pixel_array.shape[0]),
        }

    # Full decoded-pixel hashing is more memory intensive than JPEG verify().
    worker_count = max(1, min(int(workers), 8))
    if worker_count == 1:
        records = [inspect(item) for item in items]
    else:
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="siim-image-content",
        ) as executor:
            records = list(executor.map(inspect, items))
    manifest = pd.DataFrame.from_records(records)
    if len(manifest) != len(items) or manifest["content_sha256"].isna().any():
        raise RuntimeError("SIIM image content manifest is incomplete")

    train_hashes = manifest.loc[manifest["source"] == "train", "content_sha256"]
    test_hashes = manifest.loc[manifest["source"] == "test", "content_sha256"]
    train_pixel_hashes = manifest.loc[
        manifest["source"] == "train", "decoded_pixel_sha256"
    ]
    test_pixel_hashes = manifest.loc[
        manifest["source"] == "test", "decoded_pixel_sha256"
    ]
    train_counts = train_hashes.value_counts()
    test_counts = test_hashes.value_counts()
    train_duplicate_hashes = set(train_counts[train_counts > 1].index)
    test_duplicate_hashes = set(test_counts[test_counts > 1].index)
    cross_source_hashes = set(train_hashes) & set(test_hashes)
    train_pixel_counts = train_pixel_hashes.value_counts()
    test_pixel_counts = test_pixel_hashes.value_counts()
    train_duplicate_pixel_hashes = set(
        train_pixel_counts[train_pixel_counts > 1].index
    )
    test_duplicate_pixel_hashes = set(test_pixel_counts[test_pixel_counts > 1].index)
    cross_source_pixel_hashes = set(train_pixel_hashes) & set(test_pixel_hashes)
    duplicate_scope = []
    for record in manifest.itertuples(index=False):
        scopes: list[str] = []
        if record.content_sha256 in train_duplicate_hashes:
            scopes.append("within_train")
        if record.content_sha256 in test_duplicate_hashes:
            scopes.append("within_test")
        if record.content_sha256 in cross_source_hashes:
            scopes.append("train_test")
        if record.decoded_pixel_sha256 in train_duplicate_pixel_hashes:
            scopes.append("within_train_decoded_pixels")
        if record.decoded_pixel_sha256 in test_duplicate_pixel_hashes:
            scopes.append("within_test_decoded_pixels")
        if record.decoded_pixel_sha256 in cross_source_pixel_hashes:
            scopes.append("train_test_decoded_pixels")
        duplicate_scope.append("+".join(scopes) if scopes else "unique")
    manifest["duplicate_scope"] = duplicate_scope
    vector_sha256 = hashlib.sha256(
        "\n".join(
            f"{row.source}\t{row.image_name}\t{row.content_sha256}\t"
            f"{row.decoded_pixel_sha256}\t{row.perceptual_dhash64}\t{row.bytes}"
            for row in manifest.itertuples(index=False)
        ).encode("utf-8")
    ).hexdigest()
    report = {
        "schema": "evomind.siim_image_content_manifest.v1",
        "file_hash_algorithm": "sha256_exact_file_bytes",
        "decoded_pixel_hash_algorithm": "sha256_rgb_dimensions_plus_pixels",
        "perceptual_hash_algorithm": "dhash64_lanczos_grayscale_9x8",
        "train_rows": len(train_paths),
        "test_rows": len(test_paths),
        "train_duplicate_hashes": len(train_duplicate_hashes),
        "train_duplicate_rows": int(train_hashes.isin(train_duplicate_hashes).sum()),
        "test_duplicate_hashes": len(test_duplicate_hashes),
        "test_duplicate_rows": int(test_hashes.isin(test_duplicate_hashes).sum()),
        "cross_source_duplicate_hashes": len(cross_source_hashes),
        "cross_source_train_rows": int(train_hashes.isin(cross_source_hashes).sum()),
        "cross_source_test_rows": int(test_hashes.isin(cross_source_hashes).sum()),
        "train_decoded_pixel_duplicate_hashes": len(train_duplicate_pixel_hashes),
        "train_decoded_pixel_duplicate_rows": int(
            train_pixel_hashes.isin(train_duplicate_pixel_hashes).sum()
        ),
        "test_decoded_pixel_duplicate_hashes": len(test_duplicate_pixel_hashes),
        "test_decoded_pixel_duplicate_rows": int(
            test_pixel_hashes.isin(test_duplicate_pixel_hashes).sum()
        ),
        "cross_source_decoded_pixel_hashes": len(cross_source_pixel_hashes),
        "cross_source_decoded_pixel_train_rows": int(
            train_pixel_hashes.isin(cross_source_pixel_hashes).sum()
        ),
        "cross_source_decoded_pixel_test_rows": int(
            test_pixel_hashes.isin(cross_source_pixel_hashes).sum()
        ),
        "ordered_content_vector_sha256": vector_sha256,
        "private_labels_used": False,
    }
    return manifest, report


def load_precomputed_siim_image_content_manifest(
    manifest_path: Path,
    *,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    train_paths: Sequence[Path] | None = None,
    test_paths: Sequence[Path] | None = None,
    expected_sha256: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a previously hashed public-image manifest with strict order checks.

    The precomputed manifest is an immutable artifact from the same governed
    campaign's ablation stage.  Reusing it avoids decoding and hashing all
    33,126 JPEGs after every epoch-boundary continuation while preserving the
    exact content identities used by the preprocessing gate.
    """

    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None and source_sha256 != str(expected_sha256):
        raise RuntimeError("SIIM precomputed image-content manifest hash changed")
    manifest = pd.read_csv(
        path,
        dtype={
            "source": "string",
            "image_name": "string",
            "filename": "string",
            "content_sha256": "string",
            "decoded_pixel_sha256": "string",
            "perceptual_dhash64": "string",
            "duplicate_scope": "string",
        },
    )
    required = {
        "source",
        "image_name",
        "filename",
        "bytes",
        "content_sha256",
        "decoded_pixel_sha256",
        "perceptual_dhash64",
        "decoded_width",
        "decoded_height",
        "duplicate_scope",
    }
    if not required <= set(manifest):
        raise RuntimeError("SIIM precomputed image-content manifest schema changed")
    expected_sources = ["train"] * len(train_ids) + ["test"] * len(test_ids)
    expected_ids = [str(value) for value in train_ids] + [str(value) for value in test_ids]
    if manifest["source"].astype(str).tolist() != expected_sources:
        raise RuntimeError("SIIM precomputed image-content manifest source order changed")
    if manifest["image_name"].astype(str).tolist() != expected_ids:
        raise RuntimeError("SIIM precomputed image-content manifest image order changed")
    if manifest["image_name"].duplicated().any():
        raise RuntimeError("SIIM precomputed image-content manifest IDs are duplicated")
    if (train_paths is None) != (test_paths is None):
        raise ValueError("SIIM precomputed manifest paths must be supplied together")
    if train_paths is not None and test_paths is not None:
        expected_paths = [Path(value) for value in train_paths] + [
            Path(value) for value in test_paths
        ]
        if len(expected_paths) != len(manifest):
            raise RuntimeError("SIIM precomputed manifest path count changed")
        if manifest["filename"].astype(str).tolist() != [
            value.name for value in expected_paths
        ]:
            raise RuntimeError("SIIM precomputed manifest filenames changed")
        expected_sizes = [int(value.stat().st_size) for value in expected_paths]
        observed_sizes = pd.to_numeric(manifest["bytes"], errors="coerce").tolist()
        if observed_sizes != expected_sizes:
            raise RuntimeError("SIIM precomputed manifest file sizes changed")
    if not manifest["content_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise RuntimeError("SIIM precomputed exact-content hashes are invalid")
    if not manifest["decoded_pixel_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise RuntimeError("SIIM precomputed decoded-pixel hashes are invalid")
    if not manifest["perceptual_dhash64"].astype(str).str.fullmatch(r"[0-9a-f]{16}").all():
        raise RuntimeError("SIIM precomputed perceptual hashes are invalid")
    for numeric in ("bytes", "decoded_width", "decoded_height"):
        values = pd.to_numeric(manifest[numeric], errors="coerce")
        if values.isna().any() or bool((values <= 0).any()):
            raise RuntimeError(f"SIIM precomputed manifest field is invalid: {numeric}")

    train_rows = manifest["source"].eq("train")
    test_rows = manifest["source"].eq("test")
    train_hashes = manifest.loc[train_rows, "content_sha256"].astype(str)
    test_hashes = manifest.loc[test_rows, "content_sha256"].astype(str)
    train_pixels = manifest.loc[train_rows, "decoded_pixel_sha256"].astype(str)
    test_pixels = manifest.loc[test_rows, "decoded_pixel_sha256"].astype(str)
    train_counts = train_hashes.value_counts()
    test_counts = test_hashes.value_counts()
    train_pixel_counts = train_pixels.value_counts()
    test_pixel_counts = test_pixels.value_counts()
    train_duplicate_hashes = set(train_counts[train_counts > 1].index)
    test_duplicate_hashes = set(test_counts[test_counts > 1].index)
    train_duplicate_pixels = set(train_pixel_counts[train_pixel_counts > 1].index)
    test_duplicate_pixels = set(test_pixel_counts[test_pixel_counts > 1].index)
    cross_hashes = set(train_hashes) & set(test_hashes)
    cross_pixels = set(train_pixels) & set(test_pixels)
    vector_sha256 = hashlib.sha256(
        "\n".join(
            f"{row.source}\t{row.image_name}\t{row.content_sha256}\t"
            f"{row.decoded_pixel_sha256}\t{row.perceptual_dhash64}\t{row.bytes}"
            for row in manifest.itertuples(index=False)
        ).encode("utf-8")
    ).hexdigest()
    report = {
        "schema": "evomind.siim_image_content_manifest.v1",
        "file_hash_algorithm": "sha256_exact_file_bytes",
        "decoded_pixel_hash_algorithm": "sha256_rgb_dimensions_plus_pixels",
        "perceptual_hash_algorithm": "dhash64_lanczos_grayscale_9x8",
        "train_rows": len(train_ids),
        "test_rows": len(test_ids),
        "train_duplicate_hashes": len(train_duplicate_hashes),
        "train_duplicate_rows": int(train_hashes.isin(train_duplicate_hashes).sum()),
        "test_duplicate_hashes": len(test_duplicate_hashes),
        "test_duplicate_rows": int(test_hashes.isin(test_duplicate_hashes).sum()),
        "cross_source_duplicate_hashes": len(cross_hashes),
        "cross_source_train_rows": int(train_hashes.isin(cross_hashes).sum()),
        "cross_source_test_rows": int(test_hashes.isin(cross_hashes).sum()),
        "train_decoded_pixel_duplicate_hashes": len(train_duplicate_pixels),
        "train_decoded_pixel_duplicate_rows": int(train_pixels.isin(train_duplicate_pixels).sum()),
        "test_decoded_pixel_duplicate_hashes": len(test_duplicate_pixels),
        "test_decoded_pixel_duplicate_rows": int(test_pixels.isin(test_duplicate_pixels).sum()),
        "cross_source_decoded_pixel_hashes": len(cross_pixels),
        "cross_source_decoded_pixel_train_rows": int(train_pixels.isin(cross_pixels).sum()),
        "cross_source_decoded_pixel_test_rows": int(test_pixels.isin(cross_pixels).sum()),
        "ordered_content_vector_sha256": vector_sha256,
        "precomputed_manifest_reused": True,
        "precomputed_manifest_path": str(path),
        "precomputed_manifest_sha256": source_sha256,
        "private_labels_used": False,
    }
    return manifest, report


def build_siim_content_connected_groups(
    metadata: pd.DataFrame,
    target: np.ndarray,
    content_sha256: np.ndarray,
    *,
    decoded_pixel_sha256: np.ndarray | None = None,
    perceptual_dhash64: np.ndarray | None = None,
    perceptual_max_distance: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Close patient/exact-duplicate edges and audit perceptual candidates."""

    required = {"image_name", "patient_id"}
    if not required <= set(metadata):
        raise RuntimeError("SIIM content-connected metadata is incomplete")
    labels = np.asarray(target, dtype=np.int8)
    hashes = np.asarray(content_sha256, dtype=str)
    pixel_hashes = (
        np.asarray(decoded_pixel_sha256, dtype=str)
        if decoded_pixel_sha256 is not None
        else None
    )
    perceptual_hashes = (
        np.asarray(perceptual_dhash64, dtype=str)
        if perceptual_dhash64 is not None
        else None
    )
    if labels.shape != (len(metadata),) or hashes.shape != (len(metadata),):
        raise RuntimeError("SIIM content-connected arrays have invalid shapes")
    if pixel_hashes is not None and pixel_hashes.shape != (len(metadata),):
        raise RuntimeError("SIIM decoded-pixel hashes have an invalid shape")
    if perceptual_hashes is not None and perceptual_hashes.shape != (len(metadata),):
        raise RuntimeError("SIIM perceptual hashes have an invalid shape")
    if set(labels.tolist()) != {0, 1}:
        raise RuntimeError("SIIM content-connected target is invalid")
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes.tolist()):
        raise RuntimeError("SIIM image content hashes are invalid")
    if pixel_hashes is not None and any(
        not re.fullmatch(r"[0-9a-f]{64}", value) for value in pixel_hashes.tolist()
    ):
        raise RuntimeError("SIIM decoded-pixel hashes are invalid")
    if perceptual_hashes is not None and any(
        not re.fullmatch(r"[0-9a-f]{16}", value) for value in perceptual_hashes.tolist()
    ):
        raise RuntimeError("SIIM perceptual hashes are invalid")
    perceptual_max_distance = int(perceptual_max_distance)
    if not 0 <= perceptual_max_distance <= 2:
        raise ValueError("SIIM perceptual Hamming threshold must be between 0 and 2")

    parent = list(range(len(metadata)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            if left_root > right_root:
                left_root, right_root = right_root, left_root
            parent[right_root] = left_root

    patient = metadata["patient_id"].fillna("").astype(str).str.strip()
    patient_keys = patient.where(
        patient.ne(""),
        "__image__" + metadata["image_name"].astype(str),
    ).to_numpy(dtype=str)
    first_patient: dict[str, int] = {}
    first_hash: dict[str, int] = {}
    first_pixel_hash: dict[str, int] = {}
    hash_labels: dict[str, int] = {}
    pixel_hash_labels: dict[str, int] = {}
    for index, (patient_key, content_hash, label) in enumerate(
        zip(patient_keys, hashes, labels)
    ):
        if patient_key in first_patient:
            union(index, first_patient[patient_key])
        else:
            first_patient[patient_key] = index
        if content_hash in hash_labels and hash_labels[content_hash] != int(label):
            raise RuntimeError(
                "SIIM exact-content duplicate has conflicting public labels"
            )
        hash_labels.setdefault(content_hash, int(label))
        if content_hash in first_hash:
            union(index, first_hash[content_hash])
        else:
            first_hash[content_hash] = index
        if pixel_hashes is not None:
            pixel_hash = pixel_hashes[index]
            if (
                pixel_hash in pixel_hash_labels
                and pixel_hash_labels[pixel_hash] != int(label)
            ):
                raise RuntimeError(
                    "SIIM decoded-pixel duplicate has conflicting public labels"
                )
            pixel_hash_labels.setdefault(pixel_hash, int(label))
            if pixel_hash in first_pixel_hash:
                union(index, first_pixel_hash[pixel_hash])
            else:
                first_pixel_hash[pixel_hash] = index

    perceptual_exact_candidate_edges = 0
    perceptual_near_edges = 0
    if perceptual_hashes is not None:
        representative_by_value: dict[int, int] = {}
        chunk_buckets: dict[tuple[int, int], set[int]] = {}
        for index, raw_value in enumerate(perceptual_hashes.tolist()):
            value = int(raw_value, 16)
            if value in representative_by_value:
                # dHash is deliberately only a candidate-generation signal.
                # Across patients it has a high collision/chain rate on this
                # corpus, so applying transitive unions collapses unrelated
                # lesions into a giant component. Patient identity plus exact
                # file/decoded-pixel equality remain the isolation boundary.
                perceptual_exact_candidate_edges += 1
                continue
            candidates: set[int] = set()
            chunks = [
                (value >> shift) & 0xFFFF for shift in (0, 16, 32, 48)
            ]
            for position, chunk in enumerate(chunks):
                candidates.update(chunk_buckets.get((position, chunk), set()))
            for candidate_value in candidates:
                if (value ^ candidate_value).bit_count() <= perceptual_max_distance:
                    perceptual_near_edges += 1
            representative_by_value[value] = index
            for position, chunk in enumerate(chunks):
                chunk_buckets.setdefault((position, chunk), set()).add(value)

    component_members: dict[int, list[int]] = {}
    for index in range(len(metadata)):
        component_members.setdefault(find(index), []).append(index)
    groups = np.empty(len(metadata), dtype=object)
    component_records: list[dict[str, Any]] = []
    for members in component_members.values():
        image_names = sorted(metadata.iloc[members]["image_name"].astype(str).tolist())
        group_id = "siim_cc_" + hashlib.sha256(
            "\n".join(image_names).encode("utf-8")
        ).hexdigest()[:20]
        groups[np.asarray(members, dtype=np.int64)] = group_id
        component_records.append({
            "group_id": group_id,
            "rows": len(members),
            "patient_groups": len(set(patient_keys[members].tolist())),
            "content_hashes": len(set(hashes[members].tolist())),
            "decoded_pixel_hashes": (
                len(set(pixel_hashes[members].tolist()))
                if pixel_hashes is not None
                else None
            ),
            "perceptual_hashes": (
                len(set(perceptual_hashes[members].tolist()))
                if perceptual_hashes is not None
                else None
            ),
        })
    duplicate_counts = pd.Series(hashes).value_counts()
    pixel_duplicate_counts = (
        pd.Series(pixel_hashes).value_counts() if pixel_hashes is not None else None
    )
    report = {
        "schema": "evomind.siim_patient_content_connected_groups.v3",
        "leakage_group_policy": SIIM_LEAKAGE_GROUP_POLICY,
        "rows": len(metadata),
        "patient_groups_before_closure": len(set(patient_keys.tolist())),
        "connected_groups": len(component_records),
        "duplicate_hashes": int((duplicate_counts > 1).sum()),
        "duplicate_rows": int(duplicate_counts[duplicate_counts > 1].sum()),
        "decoded_pixel_duplicate_hashes": (
            int((pixel_duplicate_counts > 1).sum())
            if pixel_duplicate_counts is not None
            else 0
        ),
        "decoded_pixel_duplicate_rows": (
            int(pixel_duplicate_counts[pixel_duplicate_counts > 1].sum())
            if pixel_duplicate_counts is not None
            else 0
        ),
        "perceptual_hash_algorithm": (
            "dhash64_lanczos_grayscale_9x8"
            if perceptual_hashes is not None
            else None
        ),
        "perceptual_hamming_max_distance": (
            perceptual_max_distance if perceptual_hashes is not None else None
        ),
        "perceptual_unique_hashes": (
            len(set(perceptual_hashes.tolist()))
            if perceptual_hashes is not None
            else 0
        ),
        "perceptual_near_edges": perceptual_near_edges,
        "perceptual_exact_candidate_edges": perceptual_exact_candidate_edges,
        "perceptual_edge_policy": SIIM_PERCEPTUAL_EDGE_POLICY,
        "perceptual_edges_applied_to_groups": 0,
        "components_joining_multiple_patients": sum(
            record["patient_groups"] > 1 for record in component_records
        ),
        "largest_component_rows": max(
            (record["rows"] for record in component_records), default=0
        ),
        "ordered_group_vector_sha256": hashlib.sha256(
            "\n".join(str(value) for value in groups).encode("utf-8")
        ).hexdigest(),
        "exact_duplicate_label_conflicts": 0,
        "decoded_pixel_duplicate_label_conflicts": 0,
        "private_labels_used": False,
        "components": sorted(component_records, key=lambda item: item["group_id"]),
    }
    return np.asarray(groups, dtype=str), report


def _open_siim_rgb_for_transform(path: Path, *, decode_size: int):
    """Decode a JPEG near its training size instead of materializing all source pixels."""

    if decode_size < 1:
        raise ValueError("SIIM decode size must be positive")
    from PIL import Image

    with Image.open(path) as handle:
        handle.draft("RGB", (decode_size, decode_size))
        return handle.convert("RGB")


def siim_crop_dark_border(image: Any, *, threshold: int = 12):
    """Remove deterministic near-black dermoscope borders without shrinking valid skin."""

    from PIL import Image

    rgb = image.convert("RGB")
    values = np.asarray(rgb, dtype=np.uint8)
    luminance = values.astype(np.float32).mean(axis=2)
    foreground = luminance > int(threshold)
    rows = np.flatnonzero(foreground.mean(axis=1) >= 0.08)
    columns = np.flatnonzero(foreground.mean(axis=0) >= 0.08)
    if not len(rows) or not len(columns):
        return rgb.copy()
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(columns[0]), int(columns[-1]) + 1
    if (bottom - top) < 0.55 * values.shape[0] or (right - left) < 0.55 * values.shape[1]:
        return rgb.copy()
    pad_y = max(1, int(round((bottom - top) * 0.015)))
    pad_x = max(1, int(round((right - left) * 0.015)))
    top, bottom = max(0, top - pad_y), min(values.shape[0], bottom + pad_y)
    left, right = max(0, left - pad_x), min(values.shape[1], right + pad_x)
    return Image.fromarray(values[top:bottom, left:right], mode="RGB")


def siim_shades_of_gray_color_constancy(image: Any, *, power: int = 6):
    """Apply deterministic Shades-of-Gray illumination normalization."""

    from PIL import Image

    if int(power) < 1:
        raise ValueError("SIIM color-constancy power must be positive")
    values = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    illuminant = np.power(
        np.maximum(np.mean(np.power(np.clip(values, 1e-6, 1.0), int(power)), axis=(0, 1)), 1e-12),
        1.0 / int(power),
    )
    reference = float(np.mean(illuminant))
    corrected = np.clip(values * (reference / np.maximum(illuminant, 1e-6)), 0.0, 1.0)
    return Image.fromarray(np.rint(corrected * 255.0).astype(np.uint8), mode="RGB")


def siim_suppress_dark_hairs(
    image: Any,
    *,
    difference_threshold: int = 18,
    darkness_threshold: int = 150,
):
    """Replace narrow dark-hair candidates with a deterministic local median."""

    from PIL import Image, ImageFilter

    rgb = image.convert("RGB")
    median = rgb.filter(ImageFilter.MedianFilter(size=7))
    values = np.asarray(rgb, dtype=np.uint8)
    median_values = np.asarray(median, dtype=np.uint8)
    gray = values.astype(np.float32).mean(axis=2)
    median_gray = median_values.astype(np.float32).mean(axis=2)
    hair = (median_gray - gray >= int(difference_threshold)) & (
        gray <= int(darkness_threshold)
    )
    mask = Image.fromarray((hair.astype(np.uint8) * 255), mode="L").filter(
        ImageFilter.MaxFilter(size=3)
    )
    expanded = np.asarray(mask, dtype=np.uint8) > 0
    repaired = values.copy()
    repaired[expanded] = median_values[expanded]
    return Image.fromarray(repaired, mode="RGB")


def siim_lesion_focus_crop(image: Any, *, minimum_side_fraction: float = 0.52):
    """Crop around skin-color deviation while retaining a bounded contextual margin."""

    from PIL import Image

    fraction = float(minimum_side_fraction)
    if not 0.35 <= fraction <= 0.9:
        raise ValueError("SIIM lesion-focus minimum side fraction is invalid")
    rgb = image.convert("RGB")
    values = np.asarray(rgb, dtype=np.float32)
    height, width = values.shape[:2]
    band_y = max(1, int(round(height * 0.08)))
    band_x = max(1, int(round(width * 0.08)))
    border = np.concatenate(
        [
            values[:band_y].reshape(-1, 3),
            values[-band_y:].reshape(-1, 3),
            values[:, :band_x].reshape(-1, 3),
            values[:, -band_x:].reshape(-1, 3),
        ],
        axis=0,
    )
    skin_reference = np.median(border, axis=0)
    color_distance = np.linalg.norm(values - skin_reference.reshape(1, 1, 3), axis=2)
    threshold = float(np.percentile(color_distance, 68.0))
    candidate = color_distance >= max(8.0, threshold)
    yy, xx = np.mgrid[:height, :width]
    center_gate = (
        ((yy - (height - 1) / 2.0) / max(1.0, height * 0.52)) ** 2
        + ((xx - (width - 1) / 2.0) / max(1.0, width * 0.52)) ** 2
        <= 1.0
    )
    # ``siim_crop_dark_border`` intentionally retains a narrow contextual pad.
    # That pad can still contain near-black scope pixels and must not be allowed
    # to stretch the lesion bounding box back to the complete frame.
    guard_y = max(1, int(round(height * 0.04)))
    guard_x = max(1, int(round(width * 0.04)))
    interior_gate = np.zeros((height, width), dtype=bool)
    interior_gate[guard_y : height - guard_y, guard_x : width - guard_x] = True
    candidate &= center_gate & interior_gate
    rows, columns = np.flatnonzero(candidate.any(axis=1)), np.flatnonzero(candidate.any(axis=0))
    if not len(rows) or not len(columns):
        side_height, side_width = int(height * 0.78), int(width * 0.78)
        top, left = (height - side_height) // 2, (width - side_width) // 2
        return rgb.crop((left, top, left + side_width, top + side_height))
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(columns[0]), int(columns[-1]) + 1
    desired_height = max(bottom - top, int(round(height * fraction)))
    desired_width = max(right - left, int(round(width * fraction)))
    margin_y = int(round(desired_height * 0.12))
    margin_x = int(round(desired_width * 0.12))
    desired_height = min(height, desired_height + 2 * margin_y)
    desired_width = min(width, desired_width + 2 * margin_x)
    center_y, center_x = (top + bottom) / 2.0, (left + right) / 2.0
    top = int(round(center_y - desired_height / 2.0))
    left = int(round(center_x - desired_width / 2.0))
    top = min(max(0, top), height - desired_height)
    left = min(max(0, left), width - desired_width)
    return Image.fromarray(
        values[top : top + desired_height, left : left + desired_width].astype(np.uint8),
        mode="RGB",
    )


def _siim_center_focus_crop(image: Any, *, fraction: float = 0.78) -> Any:
    width, height = image.size
    crop_width = max(1, int(round(width * fraction)))
    crop_height = max(1, int(round(height * fraction)))
    left = max(0, (width - crop_width) // 2)
    top = max(0, (height - crop_height) // 2)
    return image.crop((left, top, left + crop_width, top + crop_height))


def prepare_siim_dermoscopy_views(
    image: Any,
    *,
    profile: str = "robust_multiview_v1",
) -> dict[str, Any]:
    """Return deterministic dermoscopy views for one frozen ablation profile."""

    if profile not in SIIM_PREPROCESSING_PROFILES:
        raise ValueError(f"Unsupported SIIM preprocessing profile: {profile}")
    full_image = image.copy()
    if profile != "raw_multiview_v1":
        full_image = siim_crop_dark_border(full_image)
    if profile in {"color_multiview_v1", "hair_multiview_v1", "robust_multiview_v1"}:
        full_image = siim_shades_of_gray_color_constancy(full_image)
    if profile in {"hair_multiview_v1", "robust_multiview_v1"}:
        full_image = siim_suppress_dark_hairs(full_image)
    lesion_focus = (
        siim_lesion_focus_crop(full_image)
        if profile == "robust_multiview_v1"
        else _siim_center_focus_crop(full_image)
    )
    return {"full_image": full_image, "lesion_focus": lesion_focus}


def apply_siim_paired_transform(
    transform: Any,
    full_image: Any,
    lesion_image: Any,
) -> tuple[Any, Any]:
    """Apply one sampled augmentation to both SIIM image views.

    The full-frame and lesion-focus branches must see the same random crop,
    flips, rotation, and colour-jitter draw.  Calling a torchvision transform
    twice without preserving its RNG state silently creates two unrelated
    geometries.  This helper samples the transform once logically: it replays
    the same Python/NumPy/Torch CPU RNG state for the second view, then restores
    the post-sample state so the worker RNG stream advances exactly once.

    Both views enter the shared transform's fixed-size resize before any random
    crop, so replaying the RNG state also replays identical crop coordinates.
    The function remains compatible with deterministic evaluation transforms.
    """

    import torch

    torch_before = torch.get_rng_state()
    python_before = random.getstate()
    numpy_before = np.random.get_state()

    full_tensor = transform(full_image)
    torch_after = torch.get_rng_state()
    python_after = random.getstate()
    numpy_after = np.random.get_state()

    torch.set_rng_state(torch_before)
    random.setstate(python_before)
    np.random.set_state(numpy_before)
    try:
        lesion_tensor = transform(lesion_image)
    finally:
        # A pair consumes one augmentation draw, not two.  Restoring the state
        # after the first application keeps subsequent samples deterministic.
        torch.set_rng_state(torch_after)
        random.setstate(python_after)
        np.random.set_state(numpy_after)

    full_shape = getattr(full_tensor, "shape", None)
    lesion_shape = getattr(lesion_tensor, "shape", None)
    if full_shape is not None and lesion_shape is not None and full_shape != lesion_shape:
        raise RuntimeError("SIIM paired transforms produced different tensor shapes")
    return full_tensor, lesion_tensor


def select_siim_preprocessing_ablation(
    profile_fold_auc: dict[str, dict[int | str, Sequence[float]]],
    *,
    evaluation_seeds: Sequence[int],
    image_content_manifest_sha256: str,
    minimum_mean_gain: float = 0.0005,
    maximum_worst_fold_regression: float = 0.002,
    maximum_seed_mean_regression: float = 0.001,
    minimum_seed_pass_fraction: float = 2.0 / 3.0,
) -> dict[str, Any]:
    """Select the longest profile with stable gains over external seeds."""

    if set(profile_fold_auc) != set(SIIM_PREPROCESSING_PROFILES):
        raise RuntimeError("SIIM preprocessing ablation must evaluate every profile")
    normalized_seeds = [int(value) for value in evaluation_seeds]
    if len(normalized_seeds) < 3 or len(set(normalized_seeds)) != len(normalized_seeds):
        raise RuntimeError(
            "SIIM preprocessing ablation requires at least three distinct external seeds"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", str(image_content_manifest_sha256)):
        raise RuntimeError("SIIM preprocessing ablation image-manifest SHA256 is invalid")
    if maximum_seed_mean_regression < 0.0:
        raise ValueError("SIIM maximum seed-mean regression must be non-negative")
    if not 0.0 < minimum_seed_pass_fraction <= 1.0:
        raise ValueError("SIIM minimum seed pass fraction must be in (0, 1]")
    fold_count: int | None = None
    profile_records: list[dict[str, Any]] = []
    for profile in SIIM_PREPROCESSING_PROFILES:
        raw_seed_values = profile_fold_auc[profile]
        try:
            supplied_seeds = {int(value) for value in raw_seed_values}
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "SIIM preprocessing ablation requires per-seed fold AUC arrays"
            ) from exc
        if supplied_seeds != set(normalized_seeds):
            raise RuntimeError("SIIM preprocessing ablation seed sets differ")
        seed_fold_auc: dict[str, list[float]] = {}
        seed_mean_auc: dict[str, float] = {}
        seed_worst_fold_auc: dict[str, float] = {}
        flattened: list[float] = []
        for seed in normalized_seeds:
            raw_values = (
                raw_seed_values[seed]
                if seed in raw_seed_values
                else raw_seed_values[str(seed)]
            )
            values = np.asarray(raw_values, dtype=np.float64)
            if (
                values.ndim != 1
                or len(values) < MIN_SIIM_ABLATION_FOLDS
                or not np.isfinite(values).all()
            ):
                raise RuntimeError("SIIM preprocessing ablation fold AUC arrays are invalid")
            if np.any((values < 0.0) | (values > 1.0)):
                raise RuntimeError(
                    "SIIM preprocessing ablation AUC values are outside [0, 1]"
                )
            if fold_count is None:
                fold_count = len(values)
            elif len(values) != fold_count:
                raise RuntimeError("SIIM preprocessing ablation fold counts differ")
            key = str(seed)
            seed_fold_auc[key] = values.tolist()
            seed_mean_auc[key] = float(values.mean())
            seed_worst_fold_auc[key] = float(values.min())
            flattened.extend(values.tolist())
        all_values = np.asarray(flattened, dtype=np.float64)
        profile_records.append({
            "profile": profile,
            "steps": SIIM_PREPROCESSING_PROFILE_STEPS[profile],
            "seed_fold_auc": seed_fold_auc,
            "seed_mean_auc": seed_mean_auc,
            "seed_worst_fold_auc": seed_worst_fold_auc,
            "fold_auc": all_values.tolist(),
            "mean_auc": float(all_values.mean()),
            "worst_fold_auc": float(all_values.min()),
        })

    selected_profile = SIIM_PREPROCESSING_PROFILES[0]
    predecessor_accepted = True
    required_seed_passes = int(
        math.ceil(len(normalized_seeds) * float(minimum_seed_pass_fraction))
    )
    for index, record in enumerate(profile_records):
        if index == 0:
            record.update({
                "predecessor_profile": None,
                "mean_gain": None,
                "worst_fold_delta": None,
                "seed_mean_gain": None,
                "seed_stability_pass_count": len(normalized_seeds),
                "seed_stability_required_count": required_seed_passes,
                "increment_accepted": True,
                "eligible_for_selection": True,
            })
            continue
        predecessor = profile_records[index - 1]
        mean_gain = float(record["mean_auc"] - predecessor["mean_auc"])
        worst_fold_delta = float(
            record["worst_fold_auc"] - predecessor["worst_fold_auc"]
        )
        seed_mean_gain = {
            str(seed): float(
                record["seed_mean_auc"][str(seed)]
                - predecessor["seed_mean_auc"][str(seed)]
            )
            for seed in normalized_seeds
        }
        seed_stability_pass_count = sum(
            gain >= -maximum_seed_mean_regression for gain in seed_mean_gain.values()
        )
        increment_accepted = bool(
            mean_gain >= minimum_mean_gain
            and worst_fold_delta >= -maximum_worst_fold_regression
            and seed_stability_pass_count >= required_seed_passes
        )
        eligible = bool(predecessor_accepted and increment_accepted)
        record.update({
            "predecessor_profile": predecessor["profile"],
            "mean_gain": mean_gain,
            "worst_fold_delta": worst_fold_delta,
            "seed_mean_gain": seed_mean_gain,
            "seed_stability_pass_count": seed_stability_pass_count,
            "seed_stability_required_count": required_seed_passes,
            "increment_accepted": increment_accepted,
            "eligible_for_selection": eligible,
        })
        predecessor_accepted = eligible
        if eligible:
            selected_profile = str(record["profile"])

    return {
        "schema": "evomind.siim_preprocessing_ablation.v1",
        "passed": True,
        "selected_profile": selected_profile,
        "profile_order": list(SIIM_PREPROCESSING_PROFILES),
        "profiles": profile_records,
        "fold_count": fold_count,
        "evaluation_seed_count": len(normalized_seeds),
        "evaluation_seeds": normalized_seeds,
        "score_count_per_profile": int((fold_count or 0) * len(normalized_seeds)),
        "minimum_mean_gain": float(minimum_mean_gain),
        "maximum_worst_fold_regression": float(maximum_worst_fold_regression),
        "maximum_seed_mean_regression": float(maximum_seed_mean_regression),
        "minimum_seed_pass_fraction": float(minimum_seed_pass_fraction),
        "minimum_seed_pass_count": required_seed_passes,
        "image_content_manifest_sha256": str(image_content_manifest_sha256),
        "private_labels_used_for_training": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "formal_seed_exclusion_required": True,
        "claim_boundary": (
            "Public-train preprocessing ablation only; not an official score or medal."
        ),
    }


def validate_siim_preprocessing_ablation_report(
    report_path: Path | None,
    *,
    expected_profile: str,
    expected_image_content_manifest_sha256: str,
    formal_seed: int,
) -> dict[str, Any]:
    """Validate an immutable SIIM preprocessing-selection report for promotion."""

    if report_path is None:
        return {
            "validated": False,
            "reason": "report_not_configured",
            "selected_profile": expected_profile,
            "checks": {"report_configured": False},
        }
    path = Path(report_path).resolve()
    if not path.is_file():
        return {
            "validated": False,
            "reason": "report_missing",
            "path": str(path),
            "selected_profile": expected_profile,
            "checks": {"report_configured": True, "report_exists": False},
        }
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    profile_names = [
        str(record.get("profile")) for record in payload.get("profiles", [])
    ]
    evaluation_seeds = [int(value) for value in payload.get("evaluation_seeds", [])]
    expected_seed_keys = {str(value) for value in evaluation_seeds}
    reported_fold_count = int(payload.get("fold_count") or 0)

    def valid_seed_fold_array(values: Any) -> bool:
        if not isinstance(values, list) or len(values) != reported_fold_count:
            return False
        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError):
            return False
        return bool(
            array.ndim == 1
            and len(array) >= MIN_SIIM_ABLATION_FOLDS
            and np.isfinite(array).all()
            and np.all((array >= 0.0) & (array <= 1.0))
        )

    per_seed_fold_arrays_complete = bool(payload.get("profiles")) and all(
        set(record.get("seed_fold_auc", {})) == expected_seed_keys
        and all(
            valid_seed_fold_array(values)
            for values in record.get("seed_fold_auc", {}).values()
        )
        for record in payload.get("profiles", [])
    )
    checks = {
        "report_configured": True,
        "report_exists": True,
        "schema_valid": payload.get("schema")
        == "evomind.siim_preprocessing_ablation.v1",
        "ablation_passed": payload.get("passed") is True,
        "selected_profile_matches": payload.get("selected_profile") == expected_profile,
        "all_profiles_evaluated": profile_names == list(SIIM_PREPROCESSING_PROFILES),
        "fold_count_exact": reported_fold_count == MIN_SIIM_ABLATION_FOLDS,
        "leakage_group_policy_valid": payload.get("leakage_group_policy")
        == SIIM_LEAKAGE_GROUP_POLICY,
        "perceptual_edge_policy_valid": payload.get("perceptual_edge_policy")
        == SIIM_PERCEPTUAL_EDGE_POLICY,
        "three_distinct_external_seeds": bool(
            len(evaluation_seeds) >= 3
            and len(set(evaluation_seeds)) == len(evaluation_seeds)
            and int(payload.get("evaluation_seed_count") or 0) == len(evaluation_seeds)
        ),
        "per_seed_fold_arrays_complete": per_seed_fold_arrays_complete,
        "score_count_matches": int(payload.get("score_count_per_profile") or 0)
        == reported_fold_count * len(evaluation_seeds),
        "image_manifest_matches": payload.get("image_content_manifest_sha256")
        == expected_image_content_manifest_sha256,
        "private_labels_excluded": payload.get("private_labels_used_for_training") is False,
        "official_grader_excluded": payload.get("official_grader_executed") is False,
        "kaggle_submission_excluded": payload.get("kaggle_submission_executed") is False,
        "formal_seed_not_used_for_ablation": int(formal_seed) not in evaluation_seeds,
    }
    return {
        "validated": bool(all(checks.values())),
        "reason": "validated" if all(checks.values()) else "contract_check_failed",
        "path": str(path),
        "sha256": _path_sha256(path),
        "selected_profile": expected_profile,
        "evaluation_seeds": evaluation_seeds,
        "checks": checks,
    }


def _run_siim_image_metadata_single_holdout_legacy(
    args: Any, task_dir: Path, logger: Any
) -> dict[str, Any]:
    import torch
    from catboost import CatBoostClassifier
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    competition_id = "siim-isic-melanoma-classification"
    resolved = resolve_competition(competition_id, args.data_root)
    train = pd.read_csv(resolved.public_dir / "train.csv")
    test = pd.read_csv(resolved.public_dir / "test.csv")
    sample = pd.read_csv(resolved.sample_submission_path)
    if not {"image_name", "patient_id", "target"} <= set(train):
        raise RuntimeError("SIIM training schema is incomplete")
    if "image_name" not in test or list(sample.columns) != ["image_name", "target"]:
        raise RuntimeError("SIIM test or submission schema is invalid")
    if (
        train["image_name"].duplicated().any()
        or test["image_name"].duplicated().any()
        or sample["image_name"].duplicated().any()
    ):
        raise RuntimeError("SIIM train, test, or submission image IDs are duplicated")
    ordered_test = sample[["image_name"]].merge(
        test,
        on="image_name",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if len(ordered_test) != len(test) or bool((ordered_test["_merge"] != "both").any()):
        raise RuntimeError("SIIM submission and test image ID sets differ")
    test = ordered_test.drop(columns="_merge")
    target = train["target"].astype(np.int64).to_numpy()
    if set(target.tolist()) != {0, 1}:
        raise RuntimeError("SIIM target must contain both binary classes")
    train_metadata, test_metadata, metadata_names = siim_metadata_features(train, test)
    train_paths = [
        resolved.public_dir / "jpeg" / "train" / f"{value}.jpg"
        for value in train["image_name"].astype(str)
    ]
    test_paths = [
        resolved.public_dir / "jpeg" / "test" / f"{value}.jpg"
        for value in test["image_name"].astype(str)
    ]
    missing_train = [str(path) for path in train_paths if not path.is_file()]
    missing_test = [str(path) for path in test_paths if not path.is_file()]
    if missing_train or missing_test:
        raise FileNotFoundError(
            "SIIM image manifest is incomplete: "
            f"train_missing={len(missing_train)} test_missing={len(missing_test)}"
        )
    train_indices, valid_indices, split_strategy = _siim_split_indices(
        train, target, holdout_fraction=args.holdout_fraction, seed=args.seed
    )
    train_transform, eval_transform = wave2._image_transforms(args.siim_image_size)

    class Images(Dataset):
        def __init__(self, indices: np.ndarray | None, *, training: bool) -> None:
            self.indices = indices
            self.training = training

        def __len__(self) -> int:
            return len(self.indices) if self.indices is not None else len(test_paths)

        def __getitem__(self, item: int):
            if self.indices is None:
                path, metadata = test_paths[item], test_metadata[item]
                label = None
            else:
                index = int(self.indices[item])
                path, metadata, label = train_paths[index], train_metadata[index], target[index]
            transform = train_transform if self.training else eval_transform
            image = transform(
                _open_siim_rgb_for_transform(
                    path,
                    decode_size=args.siim_image_size + 32,
                )
            )
            metadata_tensor = torch.from_numpy(np.asarray(metadata, dtype=np.float32))
            if label is None:
                return image, metadata_tensor
            return image, metadata_tensor, torch.tensor(float(label), dtype=torch.float32)

    train_target = target[train_indices]
    positives = max(1, int(train_target.sum()))
    negatives = max(1, len(train_target) - positives)
    ratio = negatives / positives
    generator = torch.Generator().manual_seed(args.seed)
    workers = max(0, min(int(args.siim_workers), 8, max(1, (os.cpu_count() or 4) // 2)))
    loader_options = {
        "batch_size": args.siim_batch_size,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        Images(train_indices, training=True),
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    valid_loader = DataLoader(Images(valid_indices, training=False), shuffle=False, **loader_options)
    test_loader = DataLoader(Images(None, training=False), shuffle=False, **loader_options)

    model, pretrained, weight_identity = build_siim_fusion_model(train_metadata.shape[1])
    if not pretrained:
        raise RuntimeError("SIIM medal mode requires verified pretrained vision weights")
    model = model.cuda()
    backbone_parameters = list(model.backbone.parameters())
    head_parameters = [
        *model.metadata.parameters(),
        *model.image_output.parameters(),
        *model.output.parameters(),
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.siim_learning_rate * 0.25},
            {"params": head_parameters, "lr": args.siim_learning_rate},
        ],
        weight_decay=2e-4,
    )
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(min(12.0, math.sqrt(ratio)), dtype=torch.float32, device="cuda")
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.siim_epochs)
    )
    try:
        scaler = torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler()

    def infer(loader: DataLoader, *, tta: bool) -> tuple[np.ndarray, np.ndarray | None]:
        predictions: list[np.ndarray] = []
        truths: list[np.ndarray] = []
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                images, metadata = batch[:2]
                images = images.cuda(non_blocking=True)
                metadata = metadata.cuda(non_blocking=True)
                variants = [images]
                if tta:
                    variants.extend(
                        [
                            torch.flip(images, dims=[3]),
                            torch.flip(images, dims=[2]),
                            torch.flip(images, dims=[2, 3]),
                        ]
                    )
                values = []
                for variant in variants:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        values.append(torch.sigmoid(model(variant, metadata)))
                predictions.append(torch.stack(values).mean(dim=0).float().cpu().numpy())
                if len(batch) == 3:
                    truths.append(batch[2].numpy())
        return np.concatenate(predictions), (np.concatenate(truths) if truths else None)

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    best_score = -math.inf
    best_path = task_dir / "siim_fusion_best.pt"
    history: list[dict[str, float]] = []
    for epoch in range(args.siim_epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, metadata, labels in train_loader:
            images = images.cuda(non_blocking=True)
            metadata = metadata.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = loss_fn(model(images, metadata), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach()) * len(images)
            seen += len(images)
        scheduler.step()
        valid_probability, valid_truth = infer(valid_loader, tta=True)
        assert valid_truth is not None
        score = compute_metric("roc_auc", valid_truth, valid_probability)
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_path)
        history.append(
            {"epoch": float(epoch + 1), "train_loss": running_loss / max(1, seen), "cv_score": score}
        )
        logger.info("[%s] epoch=%d cv_auc=%.6f", competition_id, epoch + 1, score)
    model.load_state_dict(torch.load(best_path, map_location="cuda", weights_only=True))
    image_valid, valid_truth = infer(valid_loader, tta=True)
    image_test, _ = infer(test_loader, tta=True)
    assert valid_truth is not None

    metadata_model = CatBoostClassifier(
        iterations=args.siim_metadata_iterations,
        depth=7,
        learning_rate=0.04,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=args.seed,
        l2_leaf_reg=5.0,
        od_type="Iter",
        od_wait=60,
        allow_writing_files=False,
        verbose=100,
        thread_count=max(1, min(24, os.cpu_count() or 4)),
    )
    metadata_model.fit(
        train_metadata[train_indices],
        target[train_indices],
        eval_set=(train_metadata[valid_indices], target[valid_indices]),
        use_best_model=True,
    )
    metadata_valid = metadata_model.predict_proba(train_metadata[valid_indices])[:, 1]
    best_metadata_iterations = max(50, int(metadata_model.get_best_iteration()) + 1)
    metadata_full = CatBoostClassifier(
        iterations=best_metadata_iterations,
        depth=7,
        learning_rate=0.04,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        random_seed=args.seed,
        l2_leaf_reg=5.0,
        allow_writing_files=False,
        verbose=100,
        thread_count=max(1, min(24, os.cpu_count() or 4)),
    )
    metadata_full.fit(train_metadata, target)
    metadata_test = metadata_full.predict_proba(test_metadata)[:, 1]
    metadata_full.save_model(str(task_dir / "siim_metadata_catboost.cbm"))

    _, image_weight, blend_mode, cv_score = select_siim_auc_blend(
        image_valid, metadata_valid, valid_truth.astype(int)
    )
    test_probability = apply_siim_blend(
        image_test,
        metadata_test,
        image_weight=image_weight,
        mode=blend_mode,
    )
    sample = align_scalar_submission_by_id(
        sample,
        test["image_name"],
        test_probability,
        id_column="image_name",
        target_column="target",
    )
    holdout_blend = apply_siim_blend(
        image_valid,
        metadata_valid,
        image_weight=image_weight,
        mode=blend_mode,
    )
    split_manifest = pd.DataFrame({
        "image_name": train["image_name"].astype(str),
        "patient_id": train["patient_id"].fillna("__missing_patient__").astype(str),
        "target": target,
        "split": "fit",
    })
    split_manifest.loc[valid_indices, "split"] = "validation"
    split_manifest.to_csv(task_dir / "siim_split_manifest.csv", index=False)
    pd.DataFrame({
        "image_name": train.iloc[valid_indices]["image_name"].astype(str).to_numpy(),
        "patient_id": train.iloc[valid_indices]["patient_id"].fillna("__missing_patient__").astype(str).to_numpy(),
        "target": valid_truth.astype(int),
        "image_probability": image_valid,
        "metadata_probability": metadata_valid,
        "blended_probability": holdout_blend,
    }).to_csv(task_dir / "siim_holdout_predictions.csv", index=False)
    component_frame = pd.DataFrame(
        {
            "image_name": sample["image_name"],
            "image_probability": image_test,
            "metadata_probability": metadata_test,
            "blended_probability": test_probability,
        }
    )
    component_frame.to_csv(task_dir / "siim_test_components.csv", index=False)
    wave0.write_json(
        task_dir / "siim_training_history.json",
        {
            "history": history,
            "image_valid_auc": compute_metric("roc_auc", valid_truth, image_valid),
            "metadata_valid_auc": compute_metric("roc_auc", valid_truth, metadata_valid),
            "blend_valid_auc": cv_score,
            "image_weight": image_weight,
            "blend_mode": blend_mode,
            "metadata_features": metadata_names,
            "pretrained_weight_identity": weight_identity,
        },
    )
    budget = {
        "seed": args.seed,
        "holdout_fraction": args.holdout_fraction,
        "split_strategy": split_strategy,
        "train_images": len(train),
        "test_images": len(test),
        "epochs": args.siim_epochs,
        "image_size": args.siim_image_size,
        "batch_size": args.siim_batch_size,
        "vision_backbone": args.siim_backbone,
        "secondary_vision_backbone": args.siim_secondary_backbone,
        "view_backbones": {
            "full_image": args.siim_backbone,
            "lesion_focus": args.siim_secondary_backbone,
        },
        "metadata_iterations_requested": args.siim_metadata_iterations,
        "metadata_iterations_refit": best_metadata_iterations,
        "pretrained_weights_loaded": pretrained,
        "pretrained_weight_identity": weight_identity,
        "image_manifest_complete": True,
        "imbalance_strategy": "loss_pos_weight_only",
        "explicit_image_id_join": True,
        "tta_flips": True,
        "image_weight": image_weight,
        "blend_mode": blend_mode,
    }
    return wave0.finalize_scored_task(
        competition_id=competition_id,
        submission=sample,
        cv_score=float(cv_score),
        args=args,
        task_dir=task_dir,
        budget=budget,
        extra={
            "runtime_seconds_model": time.perf_counter() - started,
            "model_family": "ConvNeXt_Tiny_ImageNet1K_metadata_fusion_plus_CatBoost_blend",
            "image_valid_auc": compute_metric("roc_auc", valid_truth, image_valid),
            "metadata_valid_auc": compute_metric("roc_auc", valid_truth, metadata_valid),
            "metadata_feature_count": len(metadata_names),
            "torch_peak_memory_allocated_mib": int(torch.cuda.max_memory_allocated() / 2**20),
            "budget": budget,
        },
    )


def build_siim_patient_folds(
    metadata: pd.DataFrame,
    target: np.ndarray,
    *,
    requested_folds: int,
    seed: int,
    group_values: np.ndarray | None = None,
    require_requested_folds: bool = True,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, str]:
    """Build deterministic leakage-grouped folds with complete OOF coverage."""

    from sklearn.model_selection import StratifiedGroupKFold

    required = {"image_name", "patient_id"}
    if not required <= set(metadata):
        raise RuntimeError("SIIM patient-fold metadata is incomplete")
    labels = np.asarray(target, dtype=np.int8)
    if labels.shape != (len(metadata),) or set(labels.tolist()) != {0, 1}:
        raise RuntimeError("SIIM patient-fold target is invalid")
    requested_folds = int(requested_folds)
    if requested_folds < 2:
        raise RuntimeError("SIIM patient-group validation requires at least two folds")
    if group_values is None:
        patient = metadata["patient_id"]
        missing = patient.isna() | patient.fillna("").astype(str).str.strip().eq("")
        groups = patient.fillna("").astype(str).to_numpy(dtype=object)
        groups[missing.to_numpy()] = (
            "__image__" + metadata.loc[missing, "image_name"].astype(str)
        ).to_numpy()
        strategy_group = "patient_group"
    else:
        groups = np.asarray(group_values, dtype=str)
        if groups.shape != (len(metadata),) or any(not value for value in groups.tolist()):
            raise RuntimeError("SIIM supplied leakage groups are invalid")
        strategy_group = "patient_content_connected"
    unique_groups = np.unique(groups)
    positive_groups = len(np.unique(groups[labels == 1]))
    negative_groups = len(np.unique(groups[labels == 0]))
    maximum_folds = min(
        requested_folds,
        len(unique_groups),
        positive_groups,
        negative_groups,
    )
    if require_requested_folds and maximum_folds != requested_folds:
        raise RuntimeError(
            "SIIM strict patient-group validation cannot satisfy the requested fold count: "
            f"requested_folds={requested_folds} unique_groups={len(unique_groups)} "
            f"positive_groups={positive_groups} negative_groups={negative_groups} "
            f"maximum_supported_folds={maximum_folds}"
        )
    indices = np.arange(len(metadata))
    candidate_fold_counts = (
        [requested_folds]
        if require_requested_folds
        else range(maximum_folds, 1, -1)
    )
    for fold_count in candidate_fold_counts:
        splitter = StratifiedGroupKFold(
            n_splits=fold_count,
            shuffle=True,
            random_state=seed,
        )
        try:
            splits = list(splitter.split(indices, labels, groups))
        except ValueError:
            continue
        coverage = np.zeros(len(metadata), dtype=np.int8)
        valid = True
        for fit_index, valid_index in splits:
            coverage[valid_index] += 1
            if (
                set(labels[fit_index].tolist()) != {0, 1}
                or set(labels[valid_index].tolist()) != {0, 1}
                or set(groups[fit_index]) & set(groups[valid_index])
            ):
                valid = False
                break
        if valid and np.all(coverage == 1):
            return (
                splits,
                np.asarray(groups, dtype=str),
                f"stratified_{strategy_group}_{fold_count}_fold",
            )
    detail = (
        f"requested_folds={requested_folds} unique_groups={len(unique_groups)} "
        f"positive_groups={positive_groups} negative_groups={negative_groups} "
        f"maximum_supported_folds={maximum_folds}"
    )
    raise RuntimeError(
        "SIIM could not construct valid stratified patient-group folds: " + detail
    )


def validate_siim_formal_fold_contract(
    *,
    outer_folds: int,
    inner_folds: int,
) -> tuple[int, int]:
    """Fail closed unless the formal SIIM path uses exactly 5 x 3 folds."""

    normalized = (int(outer_folds), int(inner_folds))
    required = (SIIM_FORMAL_OUTER_FOLDS, SIIM_FORMAL_INNER_FOLDS)
    if normalized != required:
        raise RuntimeError(
            "SIIM formal validation requires exactly "
            f"{required[0]} outer x {required[1]} inner folds; "
            f"received {normalized[0]} outer x {normalized[1]} inner folds"
        )
    return normalized


def build_siim_nested_patient_folds(
    metadata: pd.DataFrame,
    target: np.ndarray,
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    requested_inner_folds: int,
    seed: int,
    group_values: np.ndarray | None = None,
    require_requested_inner_folds: bool = True,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Build patient-disjoint inner selection folds inside fixed outer folds.

    The outer validation labels are deliberately never passed to the inner
    splitter.  Every available inner fold is returned so epoch and iteration
    budgets can be aggregated across all configured inner validation partitions.
    Each returned ``refit_index`` is the complete outer-fit set for the fresh
    refit before untouched outer OOF evaluation.
    """

    labels = np.asarray(target, dtype=np.int8)
    if labels.shape != (len(metadata),):
        raise RuntimeError("SIIM nested-fold target shape is invalid")
    if group_values is None:
        patient = metadata["patient_id"]
        missing = patient.isna() | patient.fillna("").astype(str).str.strip().eq("")
        groups = patient.fillna("").astype(str).to_numpy(dtype=object)
        groups[missing.to_numpy()] = (
            "__image__" + metadata.loc[missing, "image_name"].astype(str)
        ).to_numpy()
    else:
        groups = np.asarray(group_values, dtype=str)
        if groups.shape != (len(metadata),) or any(not value for value in groups.tolist()):
            raise RuntimeError("SIIM supplied nested leakage groups are invalid")
    all_indices = np.arange(len(metadata), dtype=np.int64)
    plans: list[dict[str, Any]] = []
    outer_coverage = np.zeros(len(metadata), dtype=np.int8)
    for outer_fold, (outer_fit_raw, outer_valid_raw) in enumerate(outer_splits):
        outer_fit = np.asarray(outer_fit_raw, dtype=np.int64)
        outer_valid = np.asarray(outer_valid_raw, dtype=np.int64)
        if (
            len(outer_fit) == 0
            or len(outer_valid) == 0
            or np.intersect1d(outer_fit, outer_valid).size
            or not np.array_equal(
                np.sort(np.concatenate([outer_fit, outer_valid])), all_indices
            )
        ):
            raise RuntimeError("SIIM outer fold is not a complete disjoint partition")
        outer_coverage[outer_valid] += 1
        inner_metadata = metadata.iloc[outer_fit].reset_index(drop=True)
        inner_splits, _, inner_strategy = build_siim_patient_folds(
            inner_metadata,
            labels[outer_fit],
            requested_folds=requested_inner_folds,
            seed=seed + outer_fold * 1009 + 37,
            group_values=np.asarray(groups[outer_fit], dtype=str),
            require_requested_folds=require_requested_inner_folds,
        )
        inner_fold_records: list[dict[str, Any]] = []
        inner_validation_coverage = np.zeros(len(outer_fit), dtype=np.int8)
        for inner_fold, (inner_fit_local_raw, inner_valid_local_raw) in enumerate(
            inner_splits
        ):
            inner_fit_local = np.asarray(inner_fit_local_raw, dtype=np.int64)
            inner_valid_local = np.asarray(inner_valid_local_raw, dtype=np.int64)
            inner_fit = outer_fit[inner_fit_local]
            inner_valid = outer_fit[inner_valid_local]
            inner_validation_coverage[inner_valid_local] += 1
            if (
                np.intersect1d(inner_fit, inner_valid).size
                or not np.array_equal(
                    np.sort(np.concatenate([inner_fit, inner_valid])),
                    np.sort(outer_fit),
                )
                or set(groups[inner_fit]) & set(groups[inner_valid])
                or set(groups[outer_fit]) & set(groups[outer_valid])
            ):
                raise RuntimeError("SIIM nested patient-fold isolation failed")
            inner_fold_records.append({
                "inner_fold": inner_fold,
                "inner_fit_index": inner_fit,
                "inner_valid_index": inner_valid,
            })
        if not np.all(inner_validation_coverage == 1):
            raise RuntimeError(
                "SIIM inner validation folds must cover every outer-fit row exactly once"
            )
        plans.append({
            "outer_fold": outer_fold,
            "outer_fit_index": outer_fit,
            "outer_valid_index": outer_valid,
            "inner_folds": inner_fold_records,
            "refit_index": outer_fit.copy(),
            "inner_strategy": inner_strategy,
        })
    if not np.all(outer_coverage == 1):
        raise RuntimeError("SIIM nested outer folds must cover every row exactly once")
    return plans, np.asarray(groups, dtype=str)


def select_siim_inner_epoch(
    history: list[dict[str, Any]],
    *,
    channel: str = "image_metadata_fusion",
) -> int:
    """Select one channel's fixed refit epoch from inner validation only."""

    metric_by_channel = {
        "pure_image": "pure_image_inner_validation_auc",
        "full_image": "full_image_inner_validation_auc",
        "lesion_focus": "lesion_focus_inner_validation_auc",
        "image_metadata_fusion": "fusion_inner_validation_auc",
    }
    if channel not in metric_by_channel:
        raise ValueError(f"Unsupported SIIM epoch-selection channel: {channel}")
    metric_name = metric_by_channel[channel]

    candidates: list[tuple[float, int]] = []
    for record in history:
        epoch = int(record.get("epoch", 0))
        score = float(record.get(metric_name, math.nan))
        if epoch > 0 and math.isfinite(score):
            candidates.append((score, epoch))
    if not candidates:
        raise RuntimeError("SIIM inner history contains no finite epoch-selection score")
    # Prefer the earliest epoch on an exact tie to avoid unnecessary refit work.
    return max(candidates, key=lambda item: (item[0], -item[1]))[1]


def select_siim_channel_epochs(
    history: list[dict[str, Any]],
    *,
    channels: Sequence[str] = ("pure_image", "image_metadata_fusion"),
) -> dict[str, int]:
    """Select independent channel budgets from one leakage-safe inner run."""

    return {
        channel: select_siim_inner_epoch(history, channel=channel)
        for channel in channels
    }


def aggregate_siim_inner_histories(
    histories: Sequence[Sequence[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Aggregate epoch metrics over every configured SIIM inner fold."""

    if not histories or any(not history for history in histories):
        raise RuntimeError("SIIM inner-fold histories must all be non-empty")
    metric_names = (
        "full_image_inner_validation_auc",
        "lesion_focus_inner_validation_auc",
        "pure_image_inner_validation_auc",
        "fusion_inner_validation_auc",
    )
    expected_epochs = [int(record.get("epoch", 0)) for record in histories[0]]
    if not expected_epochs or expected_epochs != list(range(1, len(expected_epochs) + 1)):
        raise RuntimeError("SIIM inner-fold history epochs are incomplete or unordered")
    if any(
        [int(record.get("epoch", 0)) for record in history] != expected_epochs
        for history in histories[1:]
    ):
        raise RuntimeError("SIIM inner-fold histories do not share the same epochs")

    aggregated: list[dict[str, Any]] = []
    for offset, epoch in enumerate(expected_epochs):
        fold_records = [history[offset] for history in histories]
        record: dict[str, Any] = {
            "epoch": epoch,
            "stage": "inner_selection_aggregate",
            "inner_fold_count": len(fold_records),
            "inner_fold_records": [dict(value) for value in fold_records],
        }
        for metric_name in metric_names:
            values = np.asarray(
                [float(value.get(metric_name, math.nan)) for value in fold_records],
                dtype=np.float64,
            )
            if not np.isfinite(values).all():
                raise RuntimeError(
                    f"SIIM inner-fold histories contain invalid {metric_name} values"
                )
            record[metric_name] = float(values.mean())
            record[f"{metric_name}_minimum"] = float(values.min())
            record[f"{metric_name}_std"] = float(values.std(ddof=0))
        aggregated.append(record)
    return aggregated


def select_siim_metadata_refit_iterations(
    best_iteration: int,
    *,
    requested_iterations: int,
) -> int:
    """Convert inner CatBoost early stopping into a fixed outer-refit budget."""

    requested = int(requested_iterations)
    if requested <= 0:
        raise ValueError("SIIM metadata iterations must be positive")
    best = int(best_iteration)
    if best < 0:
        return requested
    minimum = min(50, requested)
    return min(requested, max(minimum, best + 1))


def cross_fit_siim_multichannel_auc_blend(
    components: dict[str, np.ndarray],
    truth: np.ndarray,
    folds: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Select multichannel blending without each outer patient fold."""

    labels = np.asarray(truth, dtype=np.int8)
    fold_values = np.asarray(folds)
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in components.items()}
    if labels.shape != fold_values.shape or any(value.shape != labels.shape for value in arrays.values()):
        raise RuntimeError("SIIM cross-fit arrays have inconsistent shapes")
    prediction = np.full(len(labels), np.nan, dtype=np.float64)
    records: list[dict[str, Any]] = []
    for fold in sorted(int(value) for value in np.unique(fold_values)):
        validation = fold_values == fold
        fitting = ~validation
        if set(labels[fitting].tolist()) != {0, 1} or set(labels[validation].tolist()) != {0, 1}:
            raise RuntimeError("SIIM cross-fit fold does not contain both classes")
        selected = select_siim_multichannel_auc_blend(
            {name: value[fitting] for name, value in arrays.items()}, labels[fitting]
        )
        prediction[validation] = apply_siim_multichannel_blend(
            {name: value[validation] for name, value in arrays.items()}, selected
        )
        records.append({
            "fold": fold,
            "weights": selected["weights"],
            "mode": selected["mode"],
            "meta_fit_auc": selected["oof_auc"],
            "component_fit_auc": selected["component_oof_auc"],
            "outer_auc": float(
                compute_metric("roc_auc", labels[validation], prediction[validation])
            ),
        })
    if not np.isfinite(prediction).all():
        raise RuntimeError("SIIM cross-fit blend did not cover every row")
    return prediction, records


def cross_fit_siim_auc_blend(
    image_probability: np.ndarray,
    metadata_probability: np.ndarray,
    truth: np.ndarray,
    folds: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Backward-compatible two-channel SIIM cross-fit blend."""

    prediction, records = cross_fit_siim_multichannel_auc_blend(
        {
            "image": np.asarray(image_probability, dtype=np.float64),
            "metadata": np.asarray(metadata_probability, dtype=np.float64),
        },
        truth,
        folds,
    )
    for record in records:
        record["image_weight"] = record["weights"]["image"]
    return prediction, records


def build_siim_fold_harmonization_variants(
    oof_probability: np.ndarray,
    test_probability_by_fold: np.ndarray,
    folds: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build raw, logit-normalized, and percentile-rank fold aggregations."""

    oof = np.asarray(oof_probability, dtype=np.float64)
    test_by_fold = np.asarray(test_probability_by_fold, dtype=np.float64)
    fold_values = np.asarray(folds)
    unique_folds = sorted(int(value) for value in np.unique(fold_values))
    if oof.ndim != 1 or fold_values.shape != oof.shape:
        raise RuntimeError("SIIM harmonization OOF arrays have inconsistent shapes")
    if test_by_fold.ndim != 2 or test_by_fold.shape[0] != len(unique_folds):
        raise RuntimeError("SIIM harmonization fold-test array has an invalid shape")
    if (
        not np.isfinite(oof).all()
        or not np.isfinite(test_by_fold).all()
        or np.any((oof < 0.0) | (oof > 1.0))
        or np.any((test_by_fold < 0.0) | (test_by_fold > 1.0))
    ):
        raise RuntimeError("SIIM harmonization probabilities must be finite and closed-unit")
    # BF16/float32 sigmoid can legitimately saturate to an exact probability of
    # zero or one.  Epsilon clipping is monotonic, preserves every strict rank
    # and endpoint tie for ROC-AUC, and is required only for the logit-space
    # harmonization candidates.  Applying the same clipped arrays to the raw
    # candidate keeps all downstream probability contracts open-unit.
    oof = np.clip(oof, 1e-6, 1.0 - 1e-6)
    test_by_fold = np.clip(test_by_fold, 1e-6, 1.0 - 1e-6)

    rank_oof = np.full_like(oof, np.nan)
    logit_oof = np.full_like(oof, np.nan)
    rank_test_folds = np.zeros_like(test_by_fold)
    logit_test_folds = np.zeros_like(test_by_fold)

    def logit_zscore(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1.0 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped))
        scale = float(np.std(logits))
        if not math.isfinite(scale) or scale <= 1e-8:
            raise RuntimeError("SIIM fold logits are constant during harmonization")
        normalized = (logits - float(np.mean(logits))) / scale
        return np.clip(1.0 / (1.0 + np.exp(-normalized)), 1e-6, 1.0 - 1e-6)

    for row, fold in enumerate(unique_folds):
        validation = fold_values == fold
        rank_oof[validation] = _rank_normalize(oof[validation])
        logit_oof[validation] = logit_zscore(oof[validation])
        rank_test_folds[row] = _rank_normalize(test_by_fold[row])
        logit_test_folds[row] = logit_zscore(test_by_fold[row])
    if not np.isfinite(rank_oof).all() or not np.isfinite(logit_oof).all():
        raise RuntimeError("SIIM fold harmonization did not cover every OOF row")
    oof_variants = {
        "raw_probability": oof,
        "fold_logit_zscore": logit_oof,
        "fold_percentile_rank": rank_oof,
    }
    test_variants = {
        "raw_probability": np.mean(test_by_fold, axis=0),
        "fold_logit_zscore": np.mean(logit_test_folds, axis=0),
        "fold_percentile_rank": np.mean(rank_test_folds, axis=0),
    }
    return oof_variants, test_variants


def select_siim_crossfit_harmonization(
    oof_variants: dict[str, np.ndarray],
    test_variants: dict[str, np.ndarray],
    truth: np.ndarray,
    folds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Select fold-scale harmonization without scoring each held-out fold."""

    preferred_modes = (
        "raw_probability",
        "fold_logit_zscore",
        "fold_percentile_rank",
    )
    if set(oof_variants) != set(preferred_modes) or set(test_variants) != set(
        preferred_modes
    ):
        raise RuntimeError("SIIM harmonization variants are incomplete")
    labels = np.asarray(truth, dtype=np.int8)
    fold_values = np.asarray(folds)
    arrays = {
        mode: np.asarray(oof_variants[mode], dtype=np.float64)
        for mode in preferred_modes
    }
    tests = {
        mode: np.asarray(test_variants[mode], dtype=np.float64)
        for mode in preferred_modes
    }
    if labels.shape != fold_values.shape or any(
        values.shape != labels.shape for values in arrays.values()
    ):
        raise RuntimeError("SIIM harmonization selection arrays have invalid shapes")
    test_shape = next(iter(tests.values())).shape
    if any(values.shape != test_shape for values in tests.values()):
        raise RuntimeError("SIIM harmonization test variants have inconsistent shapes")

    def select(indices: np.ndarray) -> tuple[str, dict[str, float]]:
        scores = {
            mode: float(compute_metric("roc_auc", labels[indices], arrays[mode][indices]))
            for mode in preferred_modes
        }
        mode = max(
            preferred_modes,
            key=lambda value: (scores[value], -preferred_modes.index(value)),
        )
        return mode, scores

    crossfit = np.full(len(labels), np.nan, dtype=np.float64)
    records: list[dict[str, Any]] = []
    for fold in sorted(int(value) for value in np.unique(fold_values)):
        validation = fold_values == fold
        fitting = ~validation
        mode, scores = select(fitting)
        crossfit[validation] = arrays[mode][validation]
        records.append({
            "fold": fold,
            "selected_mode": mode,
            "meta_fit_auc_by_mode": scores,
            "outer_auc": float(
                compute_metric("roc_auc", labels[validation], crossfit[validation])
            ),
        })
    if not np.isfinite(crossfit).all():
        raise RuntimeError("SIIM harmonization cross-fit did not cover every row")
    final_mode, final_scores = select(np.ones(len(labels), dtype=bool))
    final = {
        "selected_mode": final_mode,
        "oof_auc_by_mode": final_scores,
        "cross_fitted_auc": float(compute_metric("roc_auc", labels, crossfit)),
    }
    return crossfit, tests[final_mode], records, final


def cross_fit_siim_harmonized_multichannel_blend(
    image_oof_variants: dict[str, np.ndarray],
    fusion_oof_variants: dict[str, np.ndarray],
    metadata_oof_variants: dict[str, np.ndarray],
    image_test_variants: dict[str, np.ndarray],
    fusion_test_variants: dict[str, np.ndarray],
    metadata_test_variants: dict[str, np.ndarray],
    truth: np.ndarray,
    folds: np.ndarray,
    *,
    lesion_oof_variants: dict[str, np.ndarray] | None = None,
    lesion_test_variants: dict[str, np.ndarray] | None = None,
    crossfit_grid_denominator: int = 10,
    final_grid_denominator: int = 20,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Select channel harmonization, then blend, without each outer fold.

    Harmonization is selected independently for the image, optional lesion,
    fusion, and metadata channels on the meta-fit partition.  Only the selected channels enter the
    simplex search.  This preserves the leakage boundary while avoiding a
    Cartesian set of redundant blend-grid searches on the roughly 33K-row data.
    """

    preferred_modes = (
        "raw_probability",
        "fold_logit_zscore",
        "fold_percentile_rank",
    )
    expected = set(preferred_modes)
    has_lesion = lesion_oof_variants is not None or lesion_test_variants is not None
    if (lesion_oof_variants is None) != (lesion_test_variants is None):
        raise RuntimeError("SIIM lesion harmonization variants are incomplete")
    if (
        set(image_oof_variants) != expected
        or set(fusion_oof_variants) != expected
        or set(metadata_oof_variants) != expected
        or set(image_test_variants) != expected
        or set(fusion_test_variants) != expected
        or set(metadata_test_variants) != expected
        or (
            has_lesion
            and (
                set(lesion_oof_variants or {}) != expected
                or set(lesion_test_variants or {}) != expected
            )
        )
    ):
        raise RuntimeError("SIIM joint harmonization variants are incomplete")
    labels = np.asarray(truth, dtype=np.int8)
    fold_values = np.asarray(folds)
    image_oof_arrays = {
        mode: np.asarray(image_oof_variants[mode], dtype=np.float64)
        for mode in preferred_modes
    }
    fusion_oof_arrays = {
        mode: np.asarray(fusion_oof_variants[mode], dtype=np.float64)
        for mode in preferred_modes
    }
    metadata_oof_arrays = {
        mode: np.asarray(metadata_oof_variants[mode], dtype=np.float64)
        for mode in preferred_modes
    }
    lesion_oof_arrays = (
        {
            mode: np.asarray((lesion_oof_variants or {})[mode], dtype=np.float64)
            for mode in preferred_modes
        }
        if has_lesion
        else {}
    )
    image_test_arrays = {
        mode: np.asarray(image_test_variants[mode], dtype=np.float64)
        for mode in preferred_modes
    }
    fusion_test_arrays = {
        mode: np.asarray(fusion_test_variants[mode], dtype=np.float64)
        for mode in preferred_modes
    }
    metadata_test_arrays = {
        mode: np.asarray(metadata_test_variants[mode], dtype=np.float64)
        for mode in preferred_modes
    }
    lesion_test_arrays = (
        {
            mode: np.asarray((lesion_test_variants or {})[mode], dtype=np.float64)
            for mode in preferred_modes
        }
        if has_lesion
        else {}
    )
    if (
        labels.ndim != 1
        or labels.shape != fold_values.shape
        or any(values.shape != labels.shape for values in image_oof_arrays.values())
        or any(values.shape != labels.shape for values in fusion_oof_arrays.values())
        or any(values.shape != labels.shape for values in metadata_oof_arrays.values())
        or any(values.shape != labels.shape for values in lesion_oof_arrays.values())
    ):
        raise RuntimeError("SIIM joint harmonization OOF arrays have invalid shapes")
    test_shape = next(iter(metadata_test_arrays.values())).shape
    if (
        len(test_shape) != 1
        or any(values.shape != test_shape for values in image_test_arrays.values())
        or any(values.shape != test_shape for values in fusion_test_arrays.values())
        or any(values.shape != test_shape for values in metadata_test_arrays.values())
        or any(values.shape != test_shape for values in lesion_test_arrays.values())
    ):
        raise RuntimeError("SIIM joint harmonization test arrays have invalid shapes")
    if not np.isfinite(
        [
            *(value for array in image_oof_arrays.values() for value in array),
            *(value for array in fusion_oof_arrays.values() for value in array),
            *(value for array in metadata_oof_arrays.values() for value in array),
            *(value for array in lesion_oof_arrays.values() for value in array),
            *(value for array in image_test_arrays.values() for value in array),
            *(value for array in fusion_test_arrays.values() for value in array),
            *(value for array in metadata_test_arrays.values() for value in array),
            *(value for array in lesion_test_arrays.values() for value in array),
        ]
    ).all():
        raise RuntimeError("SIIM joint harmonization arrays must be finite")
    crossfit_grid_denominator = int(crossfit_grid_denominator)
    final_grid_denominator = int(final_grid_denominator)
    if crossfit_grid_denominator < 1 or final_grid_denominator < 1:
        raise ValueError("SIIM harmonized blend grid denominators must be positive")

    def select(indices: np.ndarray, *, grid_denominator: int) -> dict[str, Any]:
        image_scores = {
            mode: float(
                compute_metric(
                    "roc_auc", labels[indices], image_oof_arrays[mode][indices]
                )
            )
            for mode in preferred_modes
        }
        fusion_scores = {
            mode: float(
                compute_metric(
                    "roc_auc", labels[indices], fusion_oof_arrays[mode][indices]
                )
            )
            for mode in preferred_modes
        }
        metadata_scores = {
            mode: float(
                compute_metric(
                    "roc_auc", labels[indices], metadata_oof_arrays[mode][indices]
                )
            )
            for mode in preferred_modes
        }
        lesion_scores = (
            {
                mode: float(
                    compute_metric(
                        "roc_auc", labels[indices], lesion_oof_arrays[mode][indices]
                    )
                )
                for mode in preferred_modes
            }
            if has_lesion
            else {}
        )
        image_mode = max(
            preferred_modes,
            key=lambda value: (image_scores[value], -preferred_modes.index(value)),
        )
        fusion_mode = max(
            preferred_modes,
            key=lambda value: (fusion_scores[value], -preferred_modes.index(value)),
        )
        metadata_mode = max(
            preferred_modes,
            key=lambda value: (metadata_scores[value], -preferred_modes.index(value)),
        )
        lesion_mode = (
            max(
                preferred_modes,
                key=lambda value: (
                    lesion_scores[value],
                    -preferred_modes.index(value),
                ),
            )
            if has_lesion
            else None
        )
        blend_components = {
            "pure_image": image_oof_arrays[image_mode][indices],
            "image_metadata_fusion": fusion_oof_arrays[fusion_mode][indices],
            "metadata_catboost": metadata_oof_arrays[metadata_mode][indices],
        }
        if has_lesion:
            assert lesion_mode is not None
            blend_components["lesion_focus"] = lesion_oof_arrays[lesion_mode][indices]
        selected = select_siim_multichannel_auc_blend(
            blend_components,
            labels[indices],
            grid_denominator=grid_denominator,
        )
        selected.update({
            "image_harmonization_mode": image_mode,
            "fusion_harmonization_mode": fusion_mode,
            "metadata_harmonization_mode": metadata_mode,
            "lesion_harmonization_mode": lesion_mode,
            "image_harmonization_auc_by_mode": image_scores,
            "fusion_harmonization_auc_by_mode": fusion_scores,
            "metadata_harmonization_auc_by_mode": metadata_scores,
            "lesion_harmonization_auc_by_mode": lesion_scores,
            "harmonization_candidate_count": (4 if has_lesion else 3)
            * len(preferred_modes),
        })
        return selected

    def components(
        selection: dict[str, Any],
        *,
        indices: np.ndarray | None,
        test: bool,
    ) -> dict[str, np.ndarray]:
        image_values = image_test_arrays if test else image_oof_arrays
        fusion_values = fusion_test_arrays if test else fusion_oof_arrays
        metadata_values = metadata_test_arrays if test else metadata_oof_arrays
        lesion_values = lesion_test_arrays if test else lesion_oof_arrays
        values = {
            "pure_image": image_values[selection["image_harmonization_mode"]],
            "image_metadata_fusion": fusion_values[
                selection["fusion_harmonization_mode"]
            ],
            "metadata_catboost": metadata_values[
                selection["metadata_harmonization_mode"]
            ],
        }
        if has_lesion:
            lesion_mode = selection.get("lesion_harmonization_mode")
            if lesion_mode is None:
                raise RuntimeError("SIIM lesion harmonization mode is missing")
            values["lesion_focus"] = lesion_values[lesion_mode]
        if indices is not None:
            values = {name: value[indices] for name, value in values.items()}
        return values

    crossfit = np.full(len(labels), np.nan, dtype=np.float64)
    records: list[dict[str, Any]] = []
    for fold in sorted(int(value) for value in np.unique(fold_values)):
        validation = fold_values == fold
        fitting = ~validation
        selected = select(
            fitting,
            grid_denominator=crossfit_grid_denominator,
        )
        crossfit[validation] = apply_siim_multichannel_blend(
            components(selected, indices=validation, test=False), selected
        )
        records.append({
            "fold": fold,
            "image_harmonization_mode": selected["image_harmonization_mode"],
            "fusion_harmonization_mode": selected["fusion_harmonization_mode"],
            "metadata_harmonization_mode": selected["metadata_harmonization_mode"],
            "lesion_harmonization_mode": selected.get("lesion_harmonization_mode"),
            "weights": selected["weights"],
            "blend_mode": selected["mode"],
            "grid_denominator": selected["grid_denominator"],
            "meta_fit_auc": selected["oof_auc"],
            "meta_fit_image_auc_by_mode": selected[
                "image_harmonization_auc_by_mode"
            ],
            "meta_fit_fusion_auc_by_mode": selected[
                "fusion_harmonization_auc_by_mode"
            ],
            "meta_fit_metadata_auc_by_mode": selected[
                "metadata_harmonization_auc_by_mode"
            ],
            "meta_fit_lesion_auc_by_mode": selected.get(
                "lesion_harmonization_auc_by_mode", {}
            ),
            "outer_auc": float(
                compute_metric("roc_auc", labels[validation], crossfit[validation])
            ),
        })
    if not np.isfinite(crossfit).all():
        raise RuntimeError("SIIM joint harmonized blend did not cover every OOF row")
    final = select(
        np.ones(len(labels), dtype=bool),
        grid_denominator=final_grid_denominator,
    )
    test_probability = apply_siim_multichannel_blend(
        components(final, indices=None, test=True), final
    )
    final["cross_fitted_auc"] = float(
        compute_metric("roc_auc", labels, crossfit)
    )
    return crossfit, test_probability, records, final


def apply_siim_crossfit_blend_records_to_test(
    records: Sequence[Mapping[str, Any]],
    image_test_variants: Mapping[str, np.ndarray],
    fusion_test_variants: Mapping[str, np.ndarray],
    metadata_test_variants: Mapping[str, np.ndarray],
    *,
    lesion_test_variants: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Replay each public meta-fit fold's fixed blend on the matching test streams.

    The ordinary SIIM submission uses a final blend fitted on all public OOF
    rows.  Evolution protocols that rank a seed's *cross-fitted* final score
    need the five corresponding fold-specific test streams instead.  This
    helper only replays already-selected public-OOF modes/weights; it performs
    no new search and reads no private labels.
    """

    required_modes = {
        "raw_probability",
        "fold_logit_zscore",
        "fold_percentile_rank",
    }
    variant_sets = [image_test_variants, fusion_test_variants, metadata_test_variants]
    if lesion_test_variants is not None:
        variant_sets.append(lesion_test_variants)
    if any(set(values) != required_modes for values in variant_sets):
        raise RuntimeError("SIIM crossfit test harmonization variants are incomplete")
    expected_variant_shape: tuple[int, ...] | None = None
    for variants in variant_sets:
        for mode in required_modes:
            values = np.asarray(variants[mode], dtype=np.float64)
            if values.ndim != 1 or values.size == 0:
                raise RuntimeError("SIIM crossfit test variants must be non-empty vectors")
            if expected_variant_shape is None:
                expected_variant_shape = values.shape
            elif values.shape != expected_variant_shape:
                raise RuntimeError("SIIM crossfit test variants have inconsistent shapes")
            if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
                raise RuntimeError(
                    "SIIM crossfit test variants must be finite closed-unit probabilities"
                )
    normalized = sorted((dict(record) for record in records), key=lambda item: int(item["fold"]))
    folds = [int(record["fold"]) for record in normalized]
    if (
        folds != list(range(len(normalized)))
        or len(normalized) != SIIM_FORMAL_OUTER_FOLDS
    ):
        raise RuntimeError("SIIM crossfit test records must cover folds 0..4 exactly once")
    outputs: list[np.ndarray] = []
    expected_shape: tuple[int, ...] | None = None
    for record in normalized:
        components = {
            "pure_image": np.asarray(
                image_test_variants[str(record["image_harmonization_mode"])],
                dtype=np.float64,
            ),
            "image_metadata_fusion": np.asarray(
                fusion_test_variants[str(record["fusion_harmonization_mode"])],
                dtype=np.float64,
            ),
            "metadata_catboost": np.asarray(
                metadata_test_variants[str(record["metadata_harmonization_mode"])],
                dtype=np.float64,
            ),
        }
        lesion_mode = record.get("lesion_harmonization_mode")
        if lesion_test_variants is not None:
            if lesion_mode is None:
                raise RuntimeError("SIIM crossfit test lesion mode is missing")
            components["lesion_focus"] = np.asarray(
                lesion_test_variants[str(lesion_mode)], dtype=np.float64
            )
        blend = {
            "mode": str(record["blend_mode"]),
            "weights": {
                str(name): float(value)
                for name, value in dict(record.get("weights") or {}).items()
            },
        }
        weights = blend["weights"]
        if set(weights) != set(components):
            raise RuntimeError("SIIM crossfit test weights differ from replay channels")
        weight_values = np.asarray(list(weights.values()), dtype=np.float64)
        if (
            not np.isfinite(weight_values).all()
            or np.any(weight_values < 0.0)
            or not math.isclose(float(weight_values.sum()), 1.0, abs_tol=1e-8)
        ):
            raise RuntimeError("SIIM crossfit test weights must be a finite simplex")
        probability = apply_siim_multichannel_blend(components, blend)
        if expected_shape is None:
            expected_shape = probability.shape
        elif probability.shape != expected_shape:
            raise RuntimeError("SIIM crossfit test fold predictions changed shape")
        outputs.append(probability)
    result = np.stack(outputs, axis=0)
    if not np.isfinite(result).all():
        raise RuntimeError("SIIM crossfit test fold predictions contain non-finite values")
    return result


def write_siim_artifact_manifest(
    task_dir: Path,
    artifacts: list[tuple[Path, str]],
    *,
    source_sha256: str,
) -> tuple[Path, str, dict[str, Any]]:
    """Write one top-level SHA256 inventory for every material SIIM artifact."""

    root = Path(task_dir).resolve()
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_path, role in artifacts:
        path = Path(raw_path).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise RuntimeError("SIIM artifact is outside its task directory") from exc
        if relative in seen:
            raise RuntimeError(f"SIIM artifact manifest path is duplicated: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"SIIM artifact is missing: {path}")
        seen.add(relative)
        records.append({
            "path": relative,
            "role": str(role),
            "bytes": int(path.stat().st_size),
            "sha256": _path_sha256(path),
        })
    if not records:
        raise RuntimeError("SIIM artifact manifest cannot be empty")
    payload = {
        "schema": "evomind.siim_artifact_manifest.v1",
        "hash_algorithm": "sha256",
        "source_sha256": str(source_sha256),
        "artifact_count": len(records),
        "artifacts": sorted(records, key=lambda item: item["path"]),
    }
    path = root / "siim_artifact_manifest.json"
    wave0.write_json(path, payload)
    return path, _path_sha256(path), payload


def configure_siim_cuda_memory_limit(torch_module: Any, memory_limit_mib: int) -> dict[str, float | int]:
    """Apply the governed SIIM CUDA ceiling before allocating model tensors."""

    limit_mib = int(memory_limit_mib)
    if limit_mib < 0:
        raise RuntimeError("SIIM CUDA memory limit must be non-negative")
    if limit_mib == 0:
        return {
            "memory_limit_mib": 0,
            "memory_total_mib": 0,
            "memory_fraction": 0.0,
        }
    if not torch_module.cuda.is_available():
        raise RuntimeError("SIIM CUDA memory limit requires an available CUDA device")
    properties = torch_module.cuda.get_device_properties(0)
    total_mib = int(properties.total_memory // (1024 * 1024))
    if total_mib <= 0 or limit_mib > total_mib:
        raise RuntimeError(
            f"SIIM CUDA memory limit is invalid: {limit_mib} MiB for {total_mib} MiB GPU"
        )
    fraction = float(limit_mib) / float(total_mib)
    torch_module.cuda.set_per_process_memory_fraction(fraction, device=0)
    return {
        "memory_limit_mib": limit_mib,
        "memory_total_mib": total_mib,
        "memory_fraction": fraction,
    }


def run_siim_image_metadata(args: Any, task_dir: Path, logger: Any) -> dict[str, Any]:
    """Patient-grouped fold ensemble with complete image/metadata OOF predictions."""

    formal_outer_folds, formal_inner_folds = validate_siim_formal_fold_contract(
        outer_folds=args.siim_folds,
        inner_folds=args.siim_inner_folds,
    )

    import torch
    from catboost import CatBoostClassifier
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    memory_limit_mib = int(getattr(args, "siim_memory_limit_mib", 0) or 0)
    memory_contract = configure_siim_cuda_memory_limit(torch, memory_limit_mib)

    competition_id = "siim-isic-melanoma-classification"
    resolved = resolve_competition(competition_id, args.data_root)
    train = pd.read_csv(resolved.public_dir / "train.csv").reset_index(drop=True)
    test = pd.read_csv(resolved.public_dir / "test.csv").reset_index(drop=True)
    sample = pd.read_csv(resolved.sample_submission_path).reset_index(drop=True)
    if not {"image_name", "patient_id", "target"} <= set(train):
        raise RuntimeError("SIIM training schema is incomplete")
    if "image_name" not in test or list(sample.columns) != ["image_name", "target"]:
        raise RuntimeError("SIIM test or submission schema is invalid")
    if (
        train["image_name"].duplicated().any()
        or test["image_name"].duplicated().any()
        or sample["image_name"].duplicated().any()
    ):
        raise RuntimeError("SIIM train, test, or submission image IDs are duplicated")
    ordered_test = sample[["image_name"]].merge(
        test,
        on="image_name",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if len(ordered_test) != len(test) or bool((ordered_test["_merge"] != "both").any()):
        raise RuntimeError("SIIM submission and test image ID sets differ")
    test = ordered_test.drop(columns="_merge")
    target = train["target"].astype(np.int64).to_numpy()
    if set(target.tolist()) != {0, 1}:
        raise RuntimeError("SIIM target must contain both binary classes")
    train_metadata, test_metadata, metadata_names = siim_metadata_features(train, test)
    train_paths = [
        resolved.public_dir / "jpeg" / "train" / f"{value}.jpg"
        for value in train["image_name"].astype(str)
    ]
    test_paths = [
        resolved.public_dir / "jpeg" / "test" / f"{value}.jpg"
        for value in test["image_name"].astype(str)
    ]
    missing_train = [str(path) for path in train_paths if not path.is_file()]
    missing_test = [str(path) for path in test_paths if not path.is_file()]
    if missing_train or missing_test:
        raise FileNotFoundError(
            "SIIM image manifest is incomplete: "
            f"train_missing={len(missing_train)} test_missing={len(missing_test)}"
        )
    workers = max(0, min(int(args.siim_workers), 8, max(1, (os.cpu_count() or 4) // 2)))
    physical_batch_size = int(args.siim_batch_size)
    effective_batch_size = int(args.siim_effective_batch_size)
    if physical_batch_size < 1 or effective_batch_size < physical_batch_size:
        raise RuntimeError("SIIM physical/effective batch sizes are invalid")
    if effective_batch_size % physical_batch_size:
        raise RuntimeError(
            "SIIM physical batch must divide the fixed effective batch exactly"
        )
    gradient_accumulation_steps = effective_batch_size // physical_batch_size
    resume_root = task_dir.parent / "siim_resume_state"
    resume_root.mkdir(parents=True, exist_ok=True)
    existing_stage_checkpoints = list(resume_root.glob("stage_*.pt"))
    if existing_stage_checkpoints and not bool(getattr(args, "resume", False)):
        raise RuntimeError(
            "SIIM resume checkpoints already exist; explicit --resume is required"
        )
    runtime_budget_seconds = float(
        getattr(args, "siim_runtime_budget_seconds", 0.0) or 0.0
    )
    if runtime_budget_seconds < 0:
        raise RuntimeError("SIIM runtime budget must be non-negative")
    budget_state_path = resume_root / "runtime_budget.json"
    if not budget_state_path.is_file():
        _write_json_atomic(
            budget_state_path,
            {
                "schema": "evomind.siim.runtime_budget.v1",
                "started_unix": time.time(),
                "runtime_budget_seconds": runtime_budget_seconds,
                "model_seed": int(args.seed),
            },
        )
    control_file = getattr(args, "siim_control_file", None)
    if control_file is not None:
        control_file = wave0.ensure_within(Path(control_file), args.allowed_root)
    content_manifest_path = task_dir / "siim_image_content_manifest.csv"
    configured_content_manifest = getattr(args, "siim_image_content_manifest", None)
    if configured_content_manifest is not None:
        content_manifest_source_path = wave0.ensure_within(
            Path(configured_content_manifest), args.allowed_root
        )
        image_content_manifest, image_content_report = (
            load_precomputed_siim_image_content_manifest(
                content_manifest_source_path,
                train_ids=train["image_name"].astype(str).tolist(),
                test_ids=test["image_name"].astype(str).tolist(),
                train_paths=train_paths,
                test_paths=test_paths,
            )
        )
        if content_manifest_source_path.resolve() != content_manifest_path.resolve():
            temporary_manifest = content_manifest_path.with_name(
                f".{content_manifest_path.name}.{os.getpid()}.tmp"
            )
            shutil.copy2(content_manifest_source_path, temporary_manifest)
            os.replace(temporary_manifest, content_manifest_path)
        image_content_report["reuse_role"] = "same_run_ablation_manifest"
    else:
        image_content_manifest, image_content_report = build_siim_image_content_manifest(
            train_paths,
            test_paths,
            train_ids=train["image_name"].astype(str).tolist(),
            test_ids=test["image_name"].astype(str).tolist(),
            workers=workers,
        )
        image_content_manifest.to_csv(content_manifest_path, index=False)
    content_manifest_sha256 = _path_sha256(content_manifest_path)
    preprocessing_profile = str(args.siim_preprocessing_profile)
    preprocessing_ablation_source = None
    if args.siim_preprocessing_ablation_report is not None:
        preprocessing_ablation_source = wave0.ensure_within(
            args.siim_preprocessing_ablation_report,
            args.allowed_root,
        )
    preprocessing_ablation_gate = validate_siim_preprocessing_ablation_report(
        preprocessing_ablation_source,
        expected_profile=preprocessing_profile,
        expected_image_content_manifest_sha256=content_manifest_sha256,
        formal_seed=args.seed,
    )
    preprocessing_ablation_gate_path = (
        task_dir / "siim_preprocessing_ablation_gate.json"
    )
    wave0.write_json(preprocessing_ablation_gate_path, preprocessing_ablation_gate)
    if not preprocessing_ablation_gate["validated"]:
        raise RuntimeError(
            "SIIM preprocessing ablation gate failed before GPU training: "
            f"{preprocessing_ablation_gate['reason']}"
        )
    train_content_hashes = image_content_manifest.loc[
        image_content_manifest["source"] == "train", "content_sha256"
    ].to_numpy(dtype=str)
    train_pixel_hashes = image_content_manifest.loc[
        image_content_manifest["source"] == "train", "decoded_pixel_sha256"
    ].to_numpy(dtype=str)
    train_perceptual_hashes = image_content_manifest.loc[
        image_content_manifest["source"] == "train", "perceptual_dhash64"
    ].to_numpy(dtype=str)
    leakage_groups, duplicate_group_report = build_siim_content_connected_groups(
        train,
        target,
        train_content_hashes,
        decoded_pixel_sha256=train_pixel_hashes,
        perceptual_dhash64=train_perceptual_hashes,
        perceptual_max_distance=1,
    )
    image_content_report.update({
        "manifest": content_manifest_path.name,
        "manifest_sha256": content_manifest_sha256,
    })
    duplicate_group_report["image_content_manifest"] = image_content_report
    duplicate_group_path = task_dir / "siim_duplicate_connected_groups.json"
    wave0.write_json(duplicate_group_path, duplicate_group_report)
    duplicate_group_sha256 = _path_sha256(duplicate_group_path)

    splits, patient_groups, split_strategy = build_siim_patient_folds(
        train,
        target,
        requested_folds=formal_outer_folds,
        seed=args.seed,
        group_values=leakage_groups,
        require_requested_folds=True,
    )
    nested_plans, nested_patient_groups = build_siim_nested_patient_folds(
        train,
        target,
        splits,
        requested_inner_folds=formal_inner_folds,
        seed=args.seed,
        group_values=leakage_groups,
        require_requested_inner_folds=True,
    )
    if not np.array_equal(patient_groups, nested_patient_groups):
        raise RuntimeError("SIIM outer and nested patient-group identities differ")
    fold_count = len(splits)
    if fold_count != formal_outer_folds:
        raise RuntimeError("SIIM formal outer-fold count changed")
    if any(len(plan["inner_folds"]) != formal_inner_folds for plan in nested_plans):
        raise RuntimeError("SIIM formal inner-fold count changed")

    def index_vector_sha256(values: np.ndarray) -> str:
        normalized = np.asarray(values, dtype="<i8")
        return hashlib.sha256(normalized.tobytes(order="C")).hexdigest()

    resume_contract_base = {
        "schema": "evomind.siim.formal_resume_contract.v1",
        "competition_id": "siim-isic-melanoma-classification",
        "model_seed": int(args.seed),
        "train_csv_sha256": _path_sha256(resolved.public_dir / "train.csv"),
        "test_csv_sha256": _path_sha256(resolved.public_dir / "test.csv"),
        "sample_submission_sha256": _path_sha256(
            resolved.public_dir / "sample_submission.csv"
        ),
        "image_content_manifest_sha256": content_manifest_sha256,
        "preprocessing_ablation_sha256": preprocessing_ablation_gate["sha256"],
        "preprocessing_profile": preprocessing_profile,
        "leakage_group_policy": SIIM_LEAKAGE_GROUP_POLICY,
        "perceptual_edge_policy": SIIM_PERCEPTUAL_EDGE_POLICY,
        "ordered_group_vector_sha256": duplicate_group_report[
            "ordered_group_vector_sha256"
        ],
        "split_strategy": split_strategy,
        "outer_folds": formal_outer_folds,
        "inner_folds": formal_inner_folds,
        "split_indices": [
            {
                "outer_fold": int(plan["outer_fold"]),
                "outer_fit_sha256": index_vector_sha256(plan["outer_fit_index"]),
                "outer_valid_sha256": index_vector_sha256(plan["outer_valid_index"]),
                "inner": [
                    {
                        "inner_fold": int(inner["inner_fold"]),
                        "fit_sha256": index_vector_sha256(inner["inner_fit_index"]),
                        "valid_sha256": index_vector_sha256(inner["inner_valid_index"]),
                    }
                    for inner in plan["inner_folds"]
                ],
            }
            for plan in nested_plans
        ],
        "model": {
            "primary_backbone": str(args.siim_backbone),
            "secondary_backbone": str(args.siim_secondary_backbone),
            "epochs": int(args.siim_epochs),
            "image_size": int(args.siim_image_size),
            "physical_batch_size": physical_batch_size,
            "effective_batch_size": effective_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": float(args.siim_learning_rate),
            "metadata_iterations": int(args.siim_metadata_iterations),
            "catboost_task_type": str(args.siim_catboost_task_type),
            "workers": workers,
            "fast_kernel_mode": bool(args.wave2_fast_kernels),
            "metadata_feature_schema_sha256": hashlib.sha256(
                json.dumps(
                    metadata_names,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        },
        "adapter_source_sha256": _path_sha256(Path(__file__)),
        "wave2_source_sha256": _path_sha256(Path(wave2.__file__)),
    }
    resume_contract_sha256 = hashlib.sha256(
        json.dumps(
            resume_contract_base,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    resume_contract = {
        **resume_contract_base,
        "contract_sha256": resume_contract_sha256,
    }
    resume_contract_path = resume_root / "resume_contract.json"
    if resume_contract_path.is_file():
        persisted_resume_contract = json.loads(
            resume_contract_path.read_text(encoding="utf-8")
        )
        if persisted_resume_contract != resume_contract:
            raise RuntimeError("SIIM formal resume contract changed")
    elif existing_stage_checkpoints:
        raise RuntimeError("SIIM legacy checkpoints lack a formal resume contract")
    else:
        _write_json_atomic(resume_contract_path, resume_contract)
    resume_contract_artifact_path = task_dir / "siim_resume_contract.json"
    shutil.copy2(resume_contract_path, resume_contract_artifact_path)

    train_transform, eval_transform = wave2._image_transforms(args.siim_image_size)

    class Images(Dataset):
        def __init__(self, indices: np.ndarray | None, *, training: bool) -> None:
            self.indices = indices
            self.training = training

        def __len__(self) -> int:
            return len(self.indices) if self.indices is not None else len(test_paths)

        def __getitem__(self, item: int):
            if self.indices is None:
                path, metadata = test_paths[item], test_metadata[item]
                label = None
            else:
                index = int(self.indices[item])
                path, metadata, label = train_paths[index], train_metadata[index], target[index]
            transform = train_transform if self.training else eval_transform
            views = prepare_siim_dermoscopy_views(
                _open_siim_rgb_for_transform(
                    path,
                    decode_size=args.siim_image_size + 32,
                ),
                profile=preprocessing_profile,
            )
            full_image, lesion_image = apply_siim_paired_transform(
                transform,
                views["full_image"],
                views["lesion_focus"],
            )
            metadata_tensor = torch.from_numpy(np.asarray(metadata, dtype=np.float32))
            if label is None:
                return full_image, lesion_image, metadata_tensor
            return (
                full_image,
                lesion_image,
                metadata_tensor,
                torch.tensor(float(label), dtype=torch.float32),
            )

    loader_options = {
        "batch_size": physical_batch_size,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
        "worker_init_fn": wave2._seed_vision_worker,
    }
    if workers > 0:
        loader_options["prefetch_factor"] = wave2.VISION_DATALOADER_PREFETCH_FACTOR
    image_oof = np.zeros(len(train), dtype=np.float64)
    lesion_oof = np.zeros(len(train), dtype=np.float64)
    fusion_oof = np.zeros(len(train), dtype=np.float64)
    metadata_oof = np.zeros(len(train), dtype=np.float64)
    image_test = np.zeros(len(test), dtype=np.float64)
    lesion_test = np.zeros(len(test), dtype=np.float64)
    fusion_test = np.zeros(len(test), dtype=np.float64)
    image_test_by_fold = np.zeros((fold_count, len(test)), dtype=np.float64)
    lesion_test_by_fold = np.zeros((fold_count, len(test)), dtype=np.float64)
    fusion_test_by_fold = np.zeros((fold_count, len(test)), dtype=np.float64)
    metadata_test_by_fold = np.zeros((fold_count, len(test)), dtype=np.float64)
    fold_assignment = np.full(len(train), -1, dtype=np.int16)
    oof_coverage = np.zeros(len(train), dtype=np.int8)
    fold_records: list[dict[str, Any]] = []
    pretrained_flags: list[bool] = []
    pretrained_weight_identities: list[dict[str, Any]] = []
    determinism_records: list[dict[str, Any]] = []
    nested_patient_records: list[dict[str, Any]] = []
    siim_fold_artifacts: list[tuple[Path, str]] = []
    siim_fold_artifacts.append(
        (preprocessing_ablation_gate_path, "preprocessing_ablation_gate")
    )
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    metadata_catboost_task_type = str(args.siim_catboost_task_type).upper()
    metadata_catboost_runtime: dict[str, Any] = {
        "task_type": metadata_catboost_task_type,
        "thread_count": max(1, min(24, os.cpu_count() or 4)),
    }
    if metadata_catboost_task_type == "GPU":
        metadata_catboost_runtime.update({
            "devices": "0",
            "gpu_ram_part": 0.2,
            "border_count": 128,
        })

    def infer(
        model: Any,
        loader: DataLoader,
        *,
        tta: bool,
        required_channels: set[str] | None = None,
    ) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
        channel_predictions: dict[str, list[np.ndarray]] = {
            "full_image": [],
            "lesion_focus": [],
            "image_metadata_fusion": [],
        }
        truths: list[np.ndarray] = []
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                full_images, lesion_images, metadata = batch[:3]
                full_images = full_images.cuda(non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
                lesion_images = lesion_images.cuda(non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
                metadata = metadata.cuda(non_blocking=True)
                variants = [(full_images, lesion_images)]
                if tta:
                    variants.extend([
                        (
                            torch.flip(full_images, dims=[3]),
                            torch.flip(lesion_images, dims=[3]),
                        ),
                        (
                            torch.flip(full_images, dims=[2]),
                            torch.flip(lesion_images, dims=[2]),
                        ),
                        (
                            torch.flip(full_images, dims=[2, 3]),
                            torch.flip(lesion_images, dims=[2, 3]),
                        ),
                    ])
                values: dict[str, list[Any]] = {
                    name: [] for name in channel_predictions
                }
                for full_variant, lesion_variant in variants:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        logits = model.forward_channels(
                            full_variant,
                            lesion_variant,
                            metadata,
                        )
                        for name in values:
                            values[name].append(torch.sigmoid(logits[name]))
                for name in channel_predictions:
                    channel_predictions[name].append(
                        torch.stack(values[name]).mean(dim=0).float().cpu().numpy()
                    )
                if len(batch) == 4:
                    truths.append(batch[3].numpy())
        probabilities = {
            name: np.concatenate(values) for name, values in channel_predictions.items()
        }
        truth = np.concatenate(truths) if truths else None
        required = required_channels or set(channel_predictions)
        for name, probability in probabilities.items():
            if name in required and (
                not np.isfinite(probability).all()
                or float(np.std(probability)) <= 1e-8
            ):
                raise RuntimeError(f"SIIM {name} model produced invalid or constant predictions")
        return probabilities, truth

    def make_loader(
        indices: np.ndarray | None,
        *,
        training: bool,
        generator_seed: int,
    ) -> DataLoader:
        return DataLoader(
            Images(indices, training=training),
            shuffle=training,
            generator=torch.Generator().manual_seed(generator_seed),
            **loader_options,
        )

    def initialize_training_state(
        training_index: np.ndarray,
        *,
        stage_seed: int,
        epoch_count: int,
    ) -> dict[str, Any]:
        determinism = seed_siim_fold(
            stage_seed,
            fast_kernel_mode=bool(args.wave2_fast_kernels),
        )
        model, pretrained, weight_identity = build_siim_multiview_fusion_model(
            train_metadata.shape[1],
            full_backbone_name=args.siim_backbone,
            lesion_backbone_name=args.siim_secondary_backbone,
        )
        if not pretrained:
            raise RuntimeError("SIIM medal mode requires verified pretrained vision weights")
        model = model.to(device="cuda", memory_format=torch.channels_last)
        training_target = target[training_index]
        positives = max(1, int(training_target.sum()))
        negatives = max(1, len(training_target) - positives)
        ratio = negatives / positives
        parameter_groups = [
            {
                "params": [
                    *model.full_backbone.parameters(),
                    *model.lesion_backbone.parameters(),
                ],
                "lr": args.siim_learning_rate * 0.25,
            },
            {
                "params": [
                    *model.metadata.parameters(),
                    *model.full_image_output.parameters(),
                    *model.lesion_image_output.parameters(),
                    *model.output.parameters(),
                ],
                "lr": args.siim_learning_rate,
            },
        ]
        try:
            optimizer = torch.optim.AdamW(
                parameter_groups,
                weight_decay=2e-4,
                fused=True,
            )
            fused_adamw = True
        except (RuntimeError, TypeError):
            optimizer = torch.optim.AdamW(parameter_groups, weight_decay=2e-4)
            fused_adamw = False
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                min(12.0, math.sqrt(ratio)), dtype=torch.float32, device="cuda"
            )
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epoch_count)
        )
        use_scaler = amp_dtype == torch.float16
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
        return {
            "model": model,
            "optimizer": optimizer,
            "loss_fn": loss_fn,
            "scheduler": scheduler,
            "scaler": scaler,
            "use_scaler": use_scaler,
            "pretrained": pretrained,
            "weight_identity": weight_identity,
            "fused_adamw": fused_adamw,
            "determinism": determinism,
        }

    def train_fixed_epochs(
        state: dict[str, Any],
        train_loader: DataLoader,
        *,
        epoch_count: int,
        stage: str,
        stage_id: str,
        fold: int,
        selection_loader: DataLoader | None = None,
        expected_selection_truth: np.ndarray | None = None,
        capture_epochs: set[int] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
        history: list[dict[str, Any]] = []
        captured_state_dicts: dict[int, dict[str, Any]] = {}
        model = state["model"]
        optimizer = state["optimizer"]
        loss_fn = state["loss_fn"]
        scheduler = state["scheduler"]
        scaler = state["scaler"]
        use_scaler = bool(state["use_scaler"])
        normalized_stage_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", stage_id).strip("._")
        if not normalized_stage_id or normalized_stage_id != stage_id:
            raise RuntimeError("SIIM resume stage identifier is invalid")
        stage_checkpoint_path = resume_root / f"stage_{stage_id}.pt"
        start_epoch = 0
        if bool(getattr(args, "resume", False)) and stage_checkpoint_path.is_file():
            checkpoint = torch.load(
                stage_checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            expected_contract = {
                "schema": "evomind.siim.epoch_resume.v2",
                "stage_id": stage_id,
                "stage": stage,
                "outer_fold": int(fold),
                "epoch_count": int(epoch_count),
                "model_seed": int(args.seed),
                "physical_batch_size": physical_batch_size,
                "effective_batch_size": effective_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "resume_contract_sha256": resume_contract_sha256,
                "source_sha256": _path_sha256(Path(__file__)),
            }
            for name, expected in expected_contract.items():
                if checkpoint.get(name) != expected:
                    raise RuntimeError(f"SIIM resume checkpoint contract changed: {name}")
            history = list(checkpoint.get("history", []))
            captured_state_dicts = dict(checkpoint.get("captured_state_dicts", {}))
            if checkpoint.get("status") == "complete":
                return history, captured_state_dicts
            if checkpoint.get("status") != "in_progress":
                raise RuntimeError("SIIM resume checkpoint status is invalid")
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for optimizer_state in optimizer.state.values():
                for name, value in list(optimizer_state.items()):
                    if torch.is_tensor(value):
                        optimizer_state[name] = value.cuda(non_blocking=False)
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if use_scaler and checkpoint.get("scaler_state_dict") is not None:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = int(checkpoint["next_epoch"])
            if not 0 <= start_epoch < epoch_count:
                raise RuntimeError("SIIM resume next epoch is invalid")

        for epoch in range(start_epoch, epoch_count):
            torch.cuda.synchronize()
            epoch_started = time.perf_counter()
            model.train()
            running_loss = 0.0
            seen = 0
            optimizer.zero_grad(set_to_none=True)
            batch_count = len(train_loader)
            for batch_index, (full_images, lesion_images, metadata, labels) in enumerate(
                train_loader
            ):
                full_images = full_images.cuda(non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
                lesion_images = lesion_images.cuda(non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
                metadata = metadata.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model.forward_channels(
                        full_images,
                        lesion_images,
                        metadata,
                    )
                    full_image_loss = loss_fn(logits["full_image"], labels)
                    lesion_focus_loss = loss_fn(logits["lesion_focus"], labels)
                    fusion_loss = loss_fn(logits["image_metadata_fusion"], labels)
                    loss = (
                        0.20 * full_image_loss
                        + 0.20 * lesion_focus_loss
                        + 0.60 * fusion_loss
                    )
                    group_start = (
                        batch_index // gradient_accumulation_steps
                    ) * gradient_accumulation_steps
                    group_steps = min(
                        gradient_accumulation_steps,
                        batch_count - group_start,
                    )
                    backward_loss = loss / max(1, group_steps)
                should_step = (
                    (batch_index + 1) % gradient_accumulation_steps == 0
                    or batch_index + 1 == batch_count
                )
                if use_scaler:
                    scaler.scale(backward_loss).backward()
                    if should_step:
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    backward_loss.backward()
                    if should_step:
                        optimizer.step()
                if should_step:
                    optimizer.zero_grad(set_to_none=True)
                running_loss += float(loss.detach()) * len(full_images)
                seen += len(full_images)
            scheduler.step()
            torch.cuda.synchronize()
            train_seconds = time.perf_counter() - epoch_started
            record: dict[str, Any] = {
                "epoch": epoch + 1,
                "stage": stage,
                "train_loss": running_loss / max(1, seen),
                "train_seconds": train_seconds,
                "train_images_per_second": seen / max(train_seconds, 1e-9),
                "physical_batch_size": physical_batch_size,
                "effective_batch_size": effective_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "peak_memory_allocated_mib": int(
                    torch.cuda.max_memory_allocated() / 2**20
                ),
            }
            if selection_loader is not None:
                selection_probability, selection_truth = infer(
                    model, selection_loader, tta=False
                )
                if (
                    selection_truth is None
                    or expected_selection_truth is None
                    or not np.array_equal(
                        selection_truth.astype(int), expected_selection_truth
                    )
                ):
                    raise RuntimeError(
                        "SIIM inner-validation labels or order differ from the fold manifest"
                    )
                record.update({
                    "full_image_inner_validation_auc": float(
                        compute_metric(
                            "roc_auc",
                            selection_truth,
                            selection_probability["full_image"],
                        )
                    ),
                    "lesion_focus_inner_validation_auc": float(
                        compute_metric(
                            "roc_auc",
                            selection_truth,
                            selection_probability["lesion_focus"],
                        )
                    ),
                    "pure_image_inner_validation_auc": float(
                        compute_metric(
                            "roc_auc",
                            selection_truth,
                            selection_probability["full_image"],
                        )
                    ),
                    "fusion_inner_validation_auc": float(
                        compute_metric(
                            "roc_auc",
                            selection_truth,
                            selection_probability["image_metadata_fusion"],
                        )
                    ),
                    "checkpoint_selection_tta": False,
                })
            history.append(record)
            if capture_epochs and epoch + 1 in capture_epochs:
                captured_state_dicts[epoch + 1] = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
            logger.info(
                "[%s] fold=%d stage=%s epoch=%d images_per_second=%.1f "
                "inner_fusion_auc=%s",
                competition_id,
                fold + 1,
                stage,
                epoch + 1,
                seen / max(train_seconds, 1e-9),
                (
                    f"{record['fusion_inner_validation_auc']:.6f}"
                    if "fusion_inner_validation_auc" in record
                    else "not_evaluated"
                ),
            )
            stage_complete = epoch + 1 == epoch_count
            checkpoint_payload: dict[str, Any] = {
                "schema": "evomind.siim.epoch_resume.v2",
                "status": "complete" if stage_complete else "in_progress",
                "stage_id": stage_id,
                "stage": stage,
                "outer_fold": int(fold),
                "epoch_count": int(epoch_count),
                "next_epoch": int(epoch + 1),
                "model_seed": int(args.seed),
                "physical_batch_size": physical_batch_size,
                "effective_batch_size": effective_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "resume_contract_sha256": resume_contract_sha256,
                "history": history,
                "captured_state_dicts": captured_state_dicts,
                "source_sha256": _path_sha256(Path(__file__)),
            }
            if not stage_complete:
                checkpoint_payload.update({
                    "model_state_dict": {
                        name: tensor.detach().cpu()
                        for name, tensor in model.state_dict().items()
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if use_scaler else None,
                })
            temporary_checkpoint = stage_checkpoint_path.with_name(
                f".{stage_checkpoint_path.name}.{os.getpid()}.tmp"
            )
            torch.save(checkpoint_payload, temporary_checkpoint)
            os.replace(temporary_checkpoint, stage_checkpoint_path)

            pause_reason = siim_epoch_pause_reason(
                control_file=control_file,
                budget_state_path=budget_state_path,
                runtime_budget_seconds=runtime_budget_seconds,
            )
            if pause_reason is not None:
                pause_state_path = resume_root / "pause_state.json"
                _write_json_atomic(
                    pause_state_path,
                    {
                        "schema": "evomind.siim.pause_state.v1",
                        "status": "paused_after_epoch",
                        "reason": pause_reason,
                        "stage_id": stage_id,
                        "stage": stage,
                        "outer_fold": int(fold),
                        "completed_epoch": int(epoch + 1),
                        "checkpoint": str(stage_checkpoint_path),
                        "same_run_resume_required": True,
                        "other_processes_modified": False,
                        "signals_sent": 0,
                    },
                )
                raise SiimPauseRequested(pause_reason, stage_checkpoint_path)
        if capture_epochs and set(captured_state_dicts) != set(capture_epochs):
            raise RuntimeError("SIIM refit did not capture every selected channel epoch")
        return history, captured_state_dicts

    for plan in nested_plans:
        fold = int(plan["outer_fold"])
        fit_index = np.asarray(plan["outer_fit_index"], dtype=np.int64)
        valid_index = np.asarray(plan["outer_valid_index"], dtype=np.int64)
        inner_fold_plans = list(plan["inner_folds"])
        if len(inner_fold_plans) < 2:
            raise RuntimeError("SIIM epoch selection requires at least two inner folds")
        refit_index = np.asarray(plan["refit_index"], dtype=np.int64)
        if not np.array_equal(np.sort(refit_index), np.sort(fit_index)):
            raise RuntimeError("SIIM fixed-epoch refit must cover the complete outer-fit set")
        fold_seed = args.seed + fold * 101
        refit_seed = fold_seed + 20_000
        fold_assignment[valid_index] = fold
        oof_coverage[valid_index] += 1

        inner_selection_histories: list[list[dict[str, Any]]] = []
        selection_seeds: list[int] = []
        selection_peak_memory_mib = 0
        selection_weight_identity: dict[str, Any] | None = None
        for inner_plan in inner_fold_plans:
            inner_fold = int(inner_plan["inner_fold"])
            inner_fit_index = np.asarray(inner_plan["inner_fit_index"], dtype=np.int64)
            inner_valid_index = np.asarray(
                inner_plan["inner_valid_index"], dtype=np.int64
            )
            selection_seed = fold_seed + 10_000 + inner_fold * 1_000
            selection_seeds.append(selection_seed)
            inner_train_loader = make_loader(
                inner_fit_index,
                training=True,
                generator_seed=selection_seed + 1,
            )
            inner_valid_loader = make_loader(
                inner_valid_index,
                training=False,
                generator_seed=selection_seed + 2,
            )
            torch.cuda.reset_peak_memory_stats()
            selection_state = initialize_training_state(
                inner_fit_index,
                stage_seed=selection_seed,
                epoch_count=args.siim_epochs,
            )
            inner_history, _ = train_fixed_epochs(
                selection_state,
                inner_train_loader,
                epoch_count=args.siim_epochs,
                stage="inner_selection",
                stage_id=f"outer{fold:02d}_inner{inner_fold:02d}_selection",
                fold=fold,
                selection_loader=inner_valid_loader,
                expected_selection_truth=target[inner_valid_index],
            )
            for record in inner_history:
                record["inner_fold"] = inner_fold
            inner_selection_histories.append(inner_history)
            selection_peak_memory_mib = max(
                selection_peak_memory_mib,
                int(torch.cuda.max_memory_allocated() / 2**20),
            )
            current_weight_identity = selection_state["weight_identity"]
            if selection_weight_identity is None:
                selection_weight_identity = current_weight_identity
            elif current_weight_identity != selection_weight_identity:
                raise RuntimeError("SIIM inner-fold pretrained weights differ")
            determinism_records.append({
                "outer_fold": fold,
                "inner_fold": inner_fold,
                "stage": "inner_selection",
                **selection_state["determinism"],
            })
            del selection_state, inner_train_loader, inner_valid_loader
            torch.cuda.empty_cache()
        if selection_weight_identity is None:
            raise RuntimeError("SIIM inner-fold selection produced no weight identity")
        selection_history = aggregate_siim_inner_histories(inner_selection_histories)
        selected_channel_epochs = select_siim_channel_epochs(
            selection_history,
            channels=("full_image", "lesion_focus", "image_metadata_fusion"),
        )
        selected_image_epoch = selected_channel_epochs["full_image"]
        selected_lesion_epoch = selected_channel_epochs["lesion_focus"]
        selected_fusion_epoch = selected_channel_epochs["image_metadata_fusion"]
        selected_epoch = max(selected_channel_epochs.values())

        outer_train_loader = make_loader(
            refit_index,
            training=True,
            generator_seed=refit_seed + 1,
        )
        outer_valid_loader = make_loader(
            valid_index,
            training=False,
            generator_seed=refit_seed + 2,
        )
        test_loader = make_loader(
            None,
            training=False,
            generator_seed=refit_seed + 3,
        )
        torch.cuda.reset_peak_memory_stats()
        refit_state = initialize_training_state(
            refit_index,
            stage_seed=refit_seed,
            epoch_count=selected_epoch,
        )
        if refit_state["weight_identity"] != selection_weight_identity:
            raise RuntimeError("SIIM selection and refit pretrained weights differ")
        refit_history, refit_state_dicts = train_fixed_epochs(
            refit_state,
            outer_train_loader,
            epoch_count=selected_epoch,
            stage="outer_refit",
            stage_id=f"outer{fold:02d}_refit",
            fold=fold,
            capture_epochs=set(selected_channel_epochs.values()),
        )
        model = refit_state["model"]
        pretrained = bool(refit_state["pretrained"])
        weight_identity = refit_state["weight_identity"]
        fused_adamw = bool(refit_state["fused_adamw"])
        pretrained_flags.append(pretrained)
        pretrained_weight_identities.append(weight_identity)
        determinism_records.append({
            "outer_fold": fold,
            "stage": "outer_refit",
            **refit_state["determinism"],
        })
        checkpoint_path = task_dir / f"siim_fusion_fold_{fold + 1:02d}.pt"
        torch.save(
            {
                "state_dict_by_epoch": refit_state_dicts,
                "outer_fold": fold,
                "selection_seeds": selection_seeds,
                "refit_seed": refit_seed,
                "selected_epoch": selected_epoch,
                "selected_channel_epochs": selected_channel_epochs,
                "refit_epochs": selected_epoch,
                "outer_fit_rows": len(refit_index),
                "pretrained_weights_loaded": pretrained,
                "pretrained_weight_identity": weight_identity,
            },
            checkpoint_path,
        )
        checkpoint_sha256 = _path_sha256(checkpoint_path)
        siim_fold_artifacts.append((checkpoint_path, "vision_channel_checkpoint"))
        channels_by_epoch: dict[int, set[str]] = {}
        for channel, epoch in selected_channel_epochs.items():
            channels_by_epoch.setdefault(int(epoch), set()).add(channel)
        fold_valid_by_channel: dict[str, np.ndarray] = {}
        fold_test_by_channel: dict[str, np.ndarray] = {}
        valid_truth: np.ndarray | None = None
        for epoch, required_channels in sorted(channels_by_epoch.items()):
            model.load_state_dict(refit_state_dicts[epoch])
            valid_probabilities, epoch_truth = infer(
                model,
                outer_valid_loader,
                tta=True,
                required_channels=required_channels,
            )
            test_probabilities, _ = infer(
                model,
                test_loader,
                tta=True,
                required_channels=required_channels,
            )
            if valid_truth is None:
                valid_truth = epoch_truth
            elif epoch_truth is None or not np.array_equal(valid_truth, epoch_truth):
                raise RuntimeError("SIIM channel-specific OOF label orders differ")
            for channel in required_channels:
                fold_valid_by_channel[channel] = valid_probabilities[channel]
                fold_test_by_channel[channel] = test_probabilities[channel]
        if valid_truth is None or not np.array_equal(
            valid_truth.astype(int), target[valid_index]
        ):
            raise RuntimeError("SIIM untouched outer OOF labels or order differ")
        fold_image_valid = fold_valid_by_channel["full_image"]
        fold_lesion_valid = fold_valid_by_channel["lesion_focus"]
        fold_fusion_valid = fold_valid_by_channel["image_metadata_fusion"]
        fold_image_test = fold_test_by_channel["full_image"]
        fold_lesion_test = fold_test_by_channel["lesion_focus"]
        fold_fusion_test = fold_test_by_channel["image_metadata_fusion"]
        image_auc = float(compute_metric("roc_auc", valid_truth, fold_image_valid))
        lesion_auc = float(compute_metric("roc_auc", valid_truth, fold_lesion_valid))
        fusion_auc = float(compute_metric("roc_auc", valid_truth, fold_fusion_valid))
        if image_auc <= 0.5 or lesion_auc <= 0.5 or fusion_auc <= 0.5:
            raise RuntimeError("SIIM positive-class orientation is noncompetitive or inverted")
        image_oof[valid_index] = fold_image_valid
        lesion_oof[valid_index] = fold_lesion_valid
        fusion_oof[valid_index] = fold_fusion_valid
        image_test_by_fold[fold] = fold_image_test
        lesion_test_by_fold[fold] = fold_lesion_test
        fusion_test_by_fold[fold] = fold_fusion_test
        image_test += fold_image_test / fold_count
        lesion_test += fold_lesion_test / fold_count
        fusion_test += fold_fusion_test / fold_count
        fold_peak_memory_mib = int(torch.cuda.max_memory_allocated() / 2**20)
        del (
            refit_state,
            refit_state_dicts,
            model,
            outer_train_loader,
            outer_valid_loader,
            test_loader,
        )
        torch.cuda.empty_cache()

        metadata_inner_best_iterations: list[int] = []
        metadata_inner_refit_candidates: list[int] = []
        for inner_plan, selection_seed in zip(
            inner_fold_plans, selection_seeds, strict=True
        ):
            inner_fit_index = np.asarray(inner_plan["inner_fit_index"], dtype=np.int64)
            inner_valid_index = np.asarray(
                inner_plan["inner_valid_index"], dtype=np.int64
            )
            metadata_selector = CatBoostClassifier(
                iterations=args.siim_metadata_iterations,
                depth=7,
                learning_rate=0.04,
                loss_function="Logloss",
                eval_metric="AUC",
                auto_class_weights="Balanced",
                random_seed=selection_seed + 17,
                l2_leaf_reg=5.0,
                od_type="Iter",
                od_wait=60,
                allow_writing_files=False,
                verbose=100,
                **metadata_catboost_runtime,
            )
            metadata_selector.fit(
                train_metadata[inner_fit_index],
                target[inner_fit_index],
                eval_set=(
                    train_metadata[inner_valid_index],
                    target[inner_valid_index],
                ),
                use_best_model=True,
            )
            metadata_inner_best_iteration = int(metadata_selector.get_best_iteration())
            metadata_inner_best_iterations.append(metadata_inner_best_iteration)
            metadata_inner_refit_candidates.append(
                select_siim_metadata_refit_iterations(
                    metadata_inner_best_iteration,
                    requested_iterations=args.siim_metadata_iterations,
                )
            )
            del metadata_selector
        metadata_refit_iterations = int(
            math.ceil(float(np.median(metadata_inner_refit_candidates)))
        )
        metadata_model = CatBoostClassifier(
            iterations=metadata_refit_iterations,
            depth=7,
            learning_rate=0.04,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=refit_seed + 17,
            l2_leaf_reg=5.0,
            allow_writing_files=False,
            verbose=100,
            **metadata_catboost_runtime,
        )
        metadata_model.fit(
            train_metadata[refit_index],
            target[refit_index],
        )
        fold_metadata_valid = metadata_model.predict_proba(train_metadata[valid_index])[:, 1]
        fold_metadata_test = metadata_model.predict_proba(test_metadata)[:, 1]
        metadata_oof[valid_index] = fold_metadata_valid
        metadata_test_by_fold[fold] = fold_metadata_test
        metadata_auc = float(
            compute_metric("roc_auc", target[valid_index], fold_metadata_valid)
        )
        metadata_path = task_dir / f"siim_metadata_fold_{fold + 1:02d}.cbm"
        metadata_model.save_model(str(metadata_path))
        metadata_sha256 = _path_sha256(metadata_path)
        siim_fold_artifacts.append((metadata_path, "metadata_catboost_model"))
        outer_group_roles = {
            "outer_fit": fit_index,
            "outer_valid": valid_index,
            "refit": refit_index,
        }
        nested_inner_records = []
        for inner_plan in inner_fold_plans:
            inner_fit_index = np.asarray(inner_plan["inner_fit_index"], dtype=np.int64)
            inner_valid_index = np.asarray(
                inner_plan["inner_valid_index"], dtype=np.int64
            )
            nested_inner_records.append({
                "inner_fold": int(inner_plan["inner_fold"]),
                "roles": {
                    "inner_fit": {
                        "rows": len(inner_fit_index),
                        "leakage_groups": sorted(
                            set(patient_groups[inner_fit_index].tolist())
                        ),
                    },
                    "inner_valid": {
                        "rows": len(inner_valid_index),
                        "leakage_groups": sorted(
                            set(patient_groups[inner_valid_index].tolist())
                        ),
                    },
                },
            })
        nested_patient_records.append({
            "outer_fold": fold,
            "inner_strategy": plan["inner_strategy"],
            "all_inner_folds_aggregated": True,
            "inner_folds": nested_inner_records,
            "roles": {
                role: {
                    "rows": len(indices),
                    "leakage_groups": sorted(set(patient_groups[indices].tolist())),
                }
                for role, indices in outer_group_roles.items()
            },
        })
        fold_records.append({
            "fold": fold,
            "seed": fold_seed,
            "selection_seeds": selection_seeds,
            "refit_seed": refit_seed,
            "train_rows": len(fit_index),
            "valid_rows": len(valid_index),
            "inner_fold_count": len(inner_fold_plans),
            "inner_train_rows_by_fold": [
                len(inner_plan["inner_fit_index"]) for inner_plan in inner_fold_plans
            ],
            "inner_valid_rows_by_fold": [
                len(inner_plan["inner_valid_index"]) for inner_plan in inner_fold_plans
            ],
            "refit_rows": len(refit_index),
            "positive_train_rows": int(target[fit_index].sum()),
            "positive_valid_rows": int(target[valid_index].sum()),
            "selected_epoch": selected_epoch,
            "selected_channel_epochs": selected_channel_epochs,
            "selected_pure_image_epoch": selected_image_epoch,
            "selected_full_image_epoch": selected_image_epoch,
            "selected_lesion_focus_epoch": selected_lesion_epoch,
            "selected_image_metadata_fusion_epoch": selected_fusion_epoch,
            "refit_epoch_count": selected_epoch,
            "pure_image_auc": image_auc,
            "full_image_auc": image_auc,
            "lesion_focus_auc": lesion_auc,
            "fusion_auc": fusion_auc,
            "metadata_auc": metadata_auc,
            "metadata_inner_best_iterations": metadata_inner_best_iterations,
            "metadata_inner_refit_candidates": metadata_inner_refit_candidates,
            "metadata_inner_aggregation": "ceil_median_fixed_budget",
            "metadata_refit_iterations": metadata_refit_iterations,
            "metadata_catboost_task_type": metadata_catboost_task_type,
            "vision_checkpoint_sha256": checkpoint_sha256,
            "metadata_model_sha256": metadata_sha256,
            "pretrained_weight_identity": weight_identity,
            "amp_dtype": str(amp_dtype).removeprefix("torch."),
            "channels_last": True,
            "fused_adamw": fused_adamw,
            "dataloader_prefetch_factor": (
                wave2.VISION_DATALOADER_PREFETCH_FACTOR if workers > 0 else None
            ),
            "jpeg_dct_scaled_decode": True,
            "jpeg_draft_decode_size": args.siim_image_size + 32,
            "preprocessing_profile": preprocessing_profile,
            "dermoscopy_preprocessing": SIIM_PREPROCESSING_PROFILE_STEPS[
                preprocessing_profile
            ],
            "preprocessing_ablation_validated": bool(
                preprocessing_ablation_gate["validated"]
            ),
            "full_image_backbone": args.siim_backbone,
            "lesion_focus_backbone": args.siim_secondary_backbone,
            "checkpoint_selection_tta": False,
            "final_oof_test_tta": True,
            "outer_validation_role": "final_oof_only",
            "selection_peak_memory_allocated_mib": selection_peak_memory_mib,
            "peak_memory_allocated_mib": fold_peak_memory_mib,
            "selection_history": selection_history,
            "inner_selection_histories": inner_selection_histories,
            "refit_history": refit_history,
        })
        del metadata_model
        gc.collect()

    if np.any(fold_assignment < 0) or not np.all(oof_coverage == 1):
        raise RuntimeError("SIIM OOF coverage must equal one for every training row")
    metadata_test = np.mean(metadata_test_by_fold, axis=0)
    image_oof_variants, image_test_variants = build_siim_fold_harmonization_variants(
        image_oof,
        image_test_by_fold,
        fold_assignment,
    )
    lesion_oof_variants, lesion_test_variants = build_siim_fold_harmonization_variants(
        lesion_oof,
        lesion_test_by_fold,
        fold_assignment,
    )
    (
        image_harmonized_oof,
        image_harmonized_test,
        image_harmonization_records,
        image_harmonization_final,
    ) = select_siim_crossfit_harmonization(
        image_oof_variants,
        image_test_variants,
        target,
        fold_assignment,
    )
    (
        lesion_harmonized_oof,
        lesion_harmonized_test,
        lesion_harmonization_records,
        lesion_harmonization_final,
    ) = select_siim_crossfit_harmonization(
        lesion_oof_variants,
        lesion_test_variants,
        target,
        fold_assignment,
    )
    fusion_oof_variants, fusion_test_variants = build_siim_fold_harmonization_variants(
        fusion_oof,
        fusion_test_by_fold,
        fold_assignment,
    )
    metadata_oof_variants, metadata_test_variants = (
        build_siim_fold_harmonization_variants(
            metadata_oof,
            metadata_test_by_fold,
            fold_assignment,
        )
    )
    (
        metadata_harmonized_oof,
        metadata_harmonized_test,
        metadata_harmonization_records,
        metadata_harmonization_final,
    ) = select_siim_crossfit_harmonization(
        metadata_oof_variants,
        metadata_test_variants,
        target,
        fold_assignment,
    )
    (
        fusion_harmonized_oof,
        fusion_harmonized_test,
        fusion_harmonization_records,
        fusion_harmonization_final,
    ) = select_siim_crossfit_harmonization(
        fusion_oof_variants,
        fusion_test_variants,
        target,
        fold_assignment,
    )
    (
        crossfit_probability,
        test_probability,
        crossfit_records,
        final_blend,
    ) = cross_fit_siim_harmonized_multichannel_blend(
        image_oof_variants,
        fusion_oof_variants,
        metadata_oof_variants,
        image_test_variants,
        fusion_test_variants,
        metadata_test_variants,
        target,
        fold_assignment,
        lesion_oof_variants=lesion_oof_variants,
        lesion_test_variants=lesion_test_variants,
    )
    crossfit_test_by_fold = apply_siim_crossfit_blend_records_to_test(
        crossfit_records,
        image_test_variants,
        fusion_test_variants,
        metadata_test_variants,
        lesion_test_variants=lesion_test_variants,
    )
    seed_final_oof_variants, seed_final_test_variants = (
        build_siim_fold_harmonization_variants(
            crossfit_probability,
            crossfit_test_by_fold,
            fold_assignment,
        )
    )
    cv_score = float(compute_metric("roc_auc", target, crossfit_probability))
    sample = align_scalar_submission_by_id(
        sample,
        test["image_name"],
        test_probability,
        id_column="image_name",
        target_column="target",
    )
    submission_path = task_dir / "submission.csv"
    sample.to_csv(submission_path, index=False)
    oof_path = task_dir / "siim_oof_predictions.csv"
    pd.DataFrame({
        "image_name": train["image_name"].astype(str),
        "patient_id": train["patient_id"].fillna("").astype(str),
        "leakage_group": patient_groups,
        "target": target,
        "fold": fold_assignment,
        "pure_image_probability": image_oof,
        "pure_image_harmonized_probability": image_harmonized_oof,
        "pure_image_fold_logit_zscore_probability": image_oof_variants[
            "fold_logit_zscore"
        ],
        "pure_image_fold_percentile_rank_probability": image_oof_variants[
            "fold_percentile_rank"
        ],
        "lesion_focus_probability": lesion_oof,
        "lesion_focus_harmonized_probability": lesion_harmonized_oof,
        "lesion_focus_fold_logit_zscore_probability": lesion_oof_variants[
            "fold_logit_zscore"
        ],
        "lesion_focus_fold_percentile_rank_probability": lesion_oof_variants[
            "fold_percentile_rank"
        ],
        "image_metadata_fusion_probability": fusion_oof,
        "image_metadata_fusion_harmonized_probability": fusion_harmonized_oof,
        "image_metadata_fusion_fold_logit_zscore_probability": fusion_oof_variants[
            "fold_logit_zscore"
        ],
        "image_metadata_fusion_fold_percentile_rank_probability": fusion_oof_variants[
            "fold_percentile_rank"
        ],
        "metadata_catboost_probability": metadata_oof,
        "metadata_catboost_harmonized_probability": metadata_harmonized_oof,
        "metadata_catboost_fold_logit_zscore_probability": metadata_oof_variants[
            "fold_logit_zscore"
        ],
        "metadata_catboost_fold_percentile_rank_probability": metadata_oof_variants[
            "fold_percentile_rank"
        ],
        "blended_probability": crossfit_probability,
        "seed_final_fold_percentile_rank_probability": seed_final_oof_variants[
            "fold_percentile_rank"
        ],
    }).to_csv(oof_path, index=False)
    test_components_path = task_dir / "siim_test_components.csv"
    pd.DataFrame({
        "image_name": test["image_name"].astype(str),
        "pure_image_probability": image_test,
        "pure_image_harmonized_probability": image_harmonized_test,
        "pure_image_fold_logit_zscore_probability": image_test_variants[
            "fold_logit_zscore"
        ],
        "pure_image_fold_percentile_rank_probability": image_test_variants[
            "fold_percentile_rank"
        ],
        "lesion_focus_probability": lesion_test,
        "lesion_focus_harmonized_probability": lesion_harmonized_test,
        "lesion_focus_fold_logit_zscore_probability": lesion_test_variants[
            "fold_logit_zscore"
        ],
        "lesion_focus_fold_percentile_rank_probability": lesion_test_variants[
            "fold_percentile_rank"
        ],
        "image_metadata_fusion_probability": fusion_test,
        "image_metadata_fusion_harmonized_probability": fusion_harmonized_test,
        "image_metadata_fusion_fold_logit_zscore_probability": fusion_test_variants[
            "fold_logit_zscore"
        ],
        "image_metadata_fusion_fold_percentile_rank_probability": fusion_test_variants[
            "fold_percentile_rank"
        ],
        "metadata_catboost_probability": metadata_test,
        "metadata_catboost_harmonized_probability": metadata_harmonized_test,
        "metadata_catboost_fold_logit_zscore_probability": metadata_test_variants[
            "fold_logit_zscore"
        ],
        "metadata_catboost_fold_percentile_rank_probability": metadata_test_variants[
            "fold_percentile_rank"
        ],
        "blended_probability": test_probability,
        "seed_final_fold_percentile_rank_probability": seed_final_test_variants[
            "fold_percentile_rank"
        ],
    }).to_csv(test_components_path, index=False)
    fold_ensemble_path = task_dir / "siim_fold_ensemble.npz"
    np.savez_compressed(
        fold_ensemble_path,
        train_id=train["image_name"].fillna("").astype(str).to_numpy(dtype=np.str_),
        patient_id=train["patient_id"].fillna("").astype(str).to_numpy(dtype=np.str_),
        leakage_group=np.asarray(patient_groups, dtype=np.str_),
        target=target,
        fold=fold_assignment,
        pure_image_oof=image_oof,
        pure_image_harmonized_oof=image_harmonized_oof,
        pure_image_logit_zscore_oof=image_oof_variants["fold_logit_zscore"],
        pure_image_percentile_rank_oof=image_oof_variants["fold_percentile_rank"],
        lesion_focus_oof=lesion_oof,
        lesion_focus_harmonized_oof=lesion_harmonized_oof,
        lesion_focus_logit_zscore_oof=lesion_oof_variants["fold_logit_zscore"],
        lesion_focus_percentile_rank_oof=lesion_oof_variants["fold_percentile_rank"],
        image_metadata_fusion_oof=fusion_oof,
        image_metadata_fusion_harmonized_oof=fusion_harmonized_oof,
        image_metadata_fusion_logit_zscore_oof=fusion_oof_variants[
            "fold_logit_zscore"
        ],
        image_metadata_fusion_percentile_rank_oof=fusion_oof_variants[
            "fold_percentile_rank"
        ],
        metadata_catboost_oof=metadata_oof,
        metadata_catboost_harmonized_oof=metadata_harmonized_oof,
        metadata_catboost_logit_zscore_oof=metadata_oof_variants[
            "fold_logit_zscore"
        ],
        metadata_catboost_percentile_rank_oof=metadata_oof_variants[
            "fold_percentile_rank"
        ],
        crossfit_probability=crossfit_probability,
        seed_final_test_by_crossfit_fold=crossfit_test_by_fold,
        seed_final_fold_percentile_rank_oof=seed_final_oof_variants[
            "fold_percentile_rank"
        ],
        test_id=test["image_name"].fillna("").astype(str).to_numpy(dtype=np.str_),
        pure_image_test=image_test,
        pure_image_test_by_fold=image_test_by_fold,
        pure_image_harmonized_test=image_harmonized_test,
        pure_image_logit_zscore_test=image_test_variants["fold_logit_zscore"],
        pure_image_percentile_rank_test=image_test_variants["fold_percentile_rank"],
        lesion_focus_test=lesion_test,
        lesion_focus_test_by_fold=lesion_test_by_fold,
        lesion_focus_harmonized_test=lesion_harmonized_test,
        lesion_focus_logit_zscore_test=lesion_test_variants["fold_logit_zscore"],
        lesion_focus_percentile_rank_test=lesion_test_variants["fold_percentile_rank"],
        image_metadata_fusion_test=fusion_test,
        image_metadata_fusion_test_by_fold=fusion_test_by_fold,
        image_metadata_fusion_harmonized_test=fusion_harmonized_test,
        image_metadata_fusion_logit_zscore_test=fusion_test_variants[
            "fold_logit_zscore"
        ],
        image_metadata_fusion_percentile_rank_test=fusion_test_variants[
            "fold_percentile_rank"
        ],
        metadata_catboost_test=metadata_test,
        metadata_catboost_test_by_fold=metadata_test_by_fold,
        metadata_catboost_harmonized_test=metadata_harmonized_test,
        metadata_catboost_logit_zscore_test=metadata_test_variants[
            "fold_logit_zscore"
        ],
        metadata_catboost_percentile_rank_test=metadata_test_variants[
            "fold_percentile_rank"
        ],
        test_probability=test_probability,
        seed_final_fold_percentile_rank_test=seed_final_test_variants[
            "fold_percentile_rank"
        ],
    )
    nested_manifest_path = task_dir / "siim_nested_patient_folds.json"
    wave0.write_json(nested_manifest_path, {
        "schema": "evomind.siim_nested_patient_folds.v2",
        "outer_split_strategy": split_strategy,
        "outer_fold_count": fold_count,
        "requested_inner_fold_count": formal_inner_folds,
        "all_inner_folds_aggregated": True,
        "outer_validation_role": "final_oof_only",
        "leakage_group_definition": (
            "patient_plus_exact_file_bytes_plus_exact_decoded_pixels__"
            "dhash64_candidates_audited_without_cross_patient_union"
        ),
        "leakage_group_policy": SIIM_LEAKAGE_GROUP_POLICY,
        "perceptual_edge_policy": SIIM_PERCEPTUAL_EDGE_POLICY,
        "resume_contract": resume_contract_artifact_path.name,
        "resume_contract_sha256": resume_contract_sha256,
        "duplicate_group_manifest": duplicate_group_path.name,
        "duplicate_group_manifest_sha256": duplicate_group_sha256,
        "fixed_epoch_outer_refit": True,
        "channel_specific_epoch_selection": True,
        "fixed_iteration_metadata_outer_refit": True,
        "folds": nested_patient_records,
    })
    nested_manifest_sha256 = _path_sha256(nested_manifest_path)
    training_history_path = task_dir / "siim_training_history.json"
    wave0.write_json(training_history_path, {
        "split_strategy": split_strategy,
        "selection_protocol": (
            "all_nested_patient_inner_folds_mean_auc_select_then_fresh_outer_refit"
        ),
        "requested_inner_fold_count": formal_inner_folds,
        "all_inner_folds_aggregated": True,
        "outer_validation_role": "final_oof_only",
        "leakage_group_definition": (
            "patient_plus_exact_file_bytes_plus_exact_decoded_pixels__"
            "dhash64_candidates_audited_without_cross_patient_union"
        ),
        "leakage_group_policy": SIIM_LEAKAGE_GROUP_POLICY,
        "perceptual_edge_policy": SIIM_PERCEPTUAL_EDGE_POLICY,
        "resume_contract": resume_contract_artifact_path.name,
        "resume_contract_sha256": resume_contract_sha256,
        "image_content_manifest": content_manifest_path.name,
        "image_content_manifest_sha256": content_manifest_sha256,
        "duplicate_group_manifest": duplicate_group_path.name,
        "duplicate_group_manifest_sha256": duplicate_group_sha256,
        "duplicate_group_report": duplicate_group_report,
        "nested_patient_manifest": nested_manifest_path.name,
        "nested_patient_manifest_sha256": nested_manifest_sha256,
        "folds": fold_records,
        "crossfit_blends": crossfit_records,
        "final_blend": final_blend,
        "fold_scale_harmonization": {
            "pure_image": {
                "crossfit_records": image_harmonization_records,
                "final": image_harmonization_final,
            },
            "lesion_focus": {
                "crossfit_records": lesion_harmonization_records,
                "final": lesion_harmonization_final,
            },
            "image_metadata_fusion": {
                "crossfit_records": fusion_harmonization_records,
                "final": fusion_harmonization_final,
            },
            "metadata_catboost": {
                "crossfit_records": metadata_harmonization_records,
                "final": metadata_harmonization_final,
            },
        },
        "cross_fitted_auc": cv_score,
        "final_meta_fit_auc": final_blend["oof_auc"],
        "metadata_features": metadata_names,
        "pretrained_weight_identities": pretrained_weight_identities,
        "determinism_records": determinism_records,
        "performance": {
            "amp_dtype": str(amp_dtype).removeprefix("torch."),
            "fast_kernel_mode": bool(args.wave2_fast_kernels),
            "channels_last": True,
            "paired_multiview_transform": True,
            "paired_transform_rng_streams": ["python", "numpy", "torch_cpu"],
            "physical_batch_size": physical_batch_size,
            "effective_batch_size": effective_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "dataloader_workers": workers,
            "dataloader_prefetch_factor": (
                wave2.VISION_DATALOADER_PREFETCH_FACTOR if workers > 0 else None
            ),
            "jpeg_dct_scaled_decode": True,
            "jpeg_draft_decode_size": args.siim_image_size + 32,
            "preprocessing_profile": preprocessing_profile,
            "dermoscopy_preprocessing": SIIM_PREPROCESSING_PROFILE_STEPS[
                preprocessing_profile
            ],
            "preprocessing_ablation": preprocessing_ablation_gate,
            "view_backbones": {
                "full_image": args.siim_backbone,
                "lesion_focus": args.siim_secondary_backbone,
            },
            "fused_adamw_all_folds": bool(
                all(record["fused_adamw"] for record in fold_records)
            ),
            "checkpoint_selection_tta": False,
            "final_oof_test_tta": True,
            "peak_memory_allocated_mib_by_fold": [
                record["peak_memory_allocated_mib"] for record in fold_records
            ],
        },
    })
    artifact_manifest_path, artifact_manifest_sha256, artifact_manifest = (
        write_siim_artifact_manifest(
            task_dir,
            [
                (content_manifest_path, "image_content_manifest"),
                (resume_contract_artifact_path, "formal_resume_contract"),
                (duplicate_group_path, "duplicate_connected_groups"),
                (nested_manifest_path, "nested_leakage_group_folds"),
                (training_history_path, "training_history"),
                (oof_path, "outer_oof_predictions"),
                (test_components_path, "test_component_predictions"),
                (fold_ensemble_path, "fold_ensemble_arrays"),
                (submission_path, "candidate_submission"),
                *siim_fold_artifacts,
            ],
            source_sha256=_path_sha256(Path(__file__)),
        )
    )
    budget = {
        "seed": args.seed,
        "folds": fold_count,
        "split_strategy": split_strategy,
        "train_images": len(train),
        "test_images": len(test),
        "epochs_per_fold": args.siim_epochs,
        "image_size": args.siim_image_size,
        "batch_size": physical_batch_size,
        "physical_batch_size": physical_batch_size,
        "effective_batch_size": effective_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "cuda_memory_limit_mib": memory_contract["memory_limit_mib"],
        "cuda_memory_total_mib": memory_contract["memory_total_mib"],
        "cuda_memory_fraction": memory_contract["memory_fraction"],
        "dataloader_workers": workers,
        "runtime_budget_seconds": runtime_budget_seconds,
        "runtime_budget_state": str(budget_state_path),
        "epoch_resume_root": str(resume_root),
        "resume_contract_sha256": resume_contract_sha256,
        "resource_control_file": str(control_file) if control_file is not None else None,
        "epoch_boundary_pause_and_resume": True,
        "other_processes_modified": False,
        "signals_sent": 0,
        "vision_backbone": args.siim_backbone,
        "metadata_iterations_requested": args.siim_metadata_iterations,
        "metadata_catboost_task_type": metadata_catboost_task_type,
        "pretrained_weights_loaded_all_folds": bool(all(pretrained_flags)),
        "pretrained_weight_identities": pretrained_weight_identities,
        "image_manifest_decoded": True,
        "image_content_hash_algorithms": [
            "sha256_exact_file_bytes",
            "sha256_rgb_dimensions_plus_pixels",
            "dhash64_lanczos_grayscale_9x8_hamming_le_1",
        ],
        "image_content_manifest": content_manifest_path.name,
        "image_content_manifest_sha256": content_manifest_sha256,
        "duplicate_connected_groups": True,
        "duplicate_group_manifest": duplicate_group_path.name,
        "duplicate_group_manifest_sha256": duplicate_group_sha256,
        "duplicate_group_report": duplicate_group_report,
        "imbalance_strategy": "loss_pos_weight_only",
        "full_image_oof": True,
        "lesion_focus_oof": True,
        "full_image_and_lesion_test_predictions_by_fold": True,
        "full_fusion_oof": True,
        "full_metadata_oof": True,
        "metadata_test_predictions_by_fold": True,
        "fold_checkpoint_test_ensemble": True,
        "cross_fitted_meta_validation": True,
        "fold_scale_harmonization_cross_fitted": True,
        "fold_scale_harmonization_modes": [
            "raw_probability",
            "fold_logit_zscore",
            "fold_percentile_rank",
        ],
        "pure_image_final_harmonization_mode": image_harmonization_final[
            "selected_mode"
        ],
        "lesion_focus_final_harmonization_mode": lesion_harmonization_final[
            "selected_mode"
        ],
        "fusion_final_harmonization_mode": fusion_harmonization_final[
            "selected_mode"
        ],
        "metadata_final_harmonization_mode": metadata_harmonization_final[
            "selected_mode"
        ],
        "nested_patient_selection": True,
        "outer_validation_final_oof_only": True,
        "fixed_epoch_outer_refit": True,
        "channel_specific_epoch_selection": True,
        "fixed_iteration_metadata_outer_refit": True,
        "nested_patient_manifest": nested_manifest_path.name,
        "nested_patient_manifest_sha256": nested_manifest_sha256,
        "selected_epochs": [record["selected_epoch"] for record in fold_records],
        "selected_channel_epochs": [
            record["selected_channel_epochs"] for record in fold_records
        ],
        "metadata_refit_iterations": [
            record["metadata_refit_iterations"] for record in fold_records
        ],
        "vision_checkpoint_sha256_by_fold": [
            record["vision_checkpoint_sha256"] for record in fold_records
        ],
        "metadata_model_sha256_by_fold": [
            record["metadata_model_sha256"] for record in fold_records
        ],
        "explicit_image_id_join": True,
        "tta_flips": True,
        "checkpoint_selection_tta": False,
        "final_oof_test_tta": True,
        "amp_dtype": str(amp_dtype).removeprefix("torch."),
        "fast_kernel_mode": bool(args.wave2_fast_kernels),
        "channels_last": True,
        "paired_multiview_transform": True,
        "paired_transform_rng_streams": ["python", "numpy", "torch_cpu"],
        "dataloader_prefetch_factor": (
            wave2.VISION_DATALOADER_PREFETCH_FACTOR if workers > 0 else None
        ),
        "jpeg_dct_scaled_decode": True,
        "jpeg_draft_decode_size": args.siim_image_size + 32,
        "preprocessing_profile": preprocessing_profile,
        "dermoscopy_preprocessing": SIIM_PREPROCESSING_PROFILE_STEPS[
            preprocessing_profile
        ],
        "preprocessing_ablation": preprocessing_ablation_gate,
        "fused_adamw_all_folds": bool(
            all(record["fused_adamw"] for record in fold_records)
        ),
        "blend_weights": final_blend["weights"],
        "blend_mode": final_blend["mode"],
        "deterministic_algorithms_requested": not bool(args.wave2_fast_kernels),
        "fold_seeds": [int(args.seed + fold * 101) for fold in range(fold_count)],
        "dataloader_worker_seeding": "torch_initial_seed_to_python_numpy",
        "determinism_records": determinism_records,
        "artifact_manifest": artifact_manifest_path.name,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "artifact_count": artifact_manifest["artifact_count"],
    }
    promotion_gate = wave0.build_metric_promotion_gate(
        name="siim_patient_grouped_cross_fitted_auc",
        metric="roc_auc",
        direction="maximize",
        score=float(cv_score),
        threshold=0.942,
        extra_checks={
            "preprocessing_ablation_validated": bool(
                preprocessing_ablation_gate["validated"]
            ),
        },
        evidence={
            "folds": fold_records,
            "patient_grouped": True,
            "exact_content_duplicate_grouped": True,
            "decoded_pixel_duplicate_grouped": True,
            "perceptual_duplicate_candidates_audited": True,
            "perceptual_candidates_grouped": False,
            "fold_scale_harmonization_cross_fitted": True,
            "metadata_test_predictions_by_fold": True,
            "nested_patient_selection": True,
            "all_inner_folds_aggregated": True,
            "requested_inner_fold_count": formal_inner_folds,
            "outer_validation_role": "final_oof_only",
            "fixed_epoch_outer_refit": True,
            "channel_specific_epoch_selection": True,
            "fixed_iteration_metadata_outer_refit": True,
            "duplicate_group_manifest_sha256": duplicate_group_sha256,
            "nested_patient_manifest_sha256": nested_manifest_sha256,
            "artifact_manifest_sha256": artifact_manifest_sha256,
            "image_harmonization": image_harmonization_final,
            "lesion_harmonization": lesion_harmonization_final,
            "fusion_harmonization": fusion_harmonization_final,
            "metadata_harmonization": metadata_harmonization_final,
            "pretrained_weight_identities": pretrained_weight_identities,
            "view_backbones": {
                "full_image": args.siim_backbone,
                "lesion_focus": args.siim_secondary_backbone,
            },
            "preprocessing_profile": preprocessing_profile,
            "dermoscopy_preprocessing": SIIM_PREPROCESSING_PROFILE_STEPS[
                preprocessing_profile
            ],
            "preprocessing_ablation": preprocessing_ablation_gate,
            "determinism_records": determinism_records,
        },
    )
    return wave0.finalize_scored_task(
        competition_id=competition_id,
        submission=sample,
        cv_score=cv_score,
        args=args,
        task_dir=task_dir,
        budget=budget,
        promotion_gate=promotion_gate,
        extra={
            "runtime_seconds_model": time.perf_counter() - started,
            "model_family": (
                f"{fold_count}fold_patient_content_grouped_{args.siim_backbone}_full_plus_"
                f"{args.siim_secondary_backbone}_lesion_ImageNet1K_"
                "multiview_metadata_fusion_plus_CatBoost_crossfit_blend"
            ),
            "folds": fold_records,
            "crossfit_blends": crossfit_records,
            "image_harmonization": image_harmonization_final,
            "lesion_harmonization": lesion_harmonization_final,
            "fusion_harmonization": fusion_harmonization_final,
            "metadata_harmonization": metadata_harmonization_final,
            "pure_image_oof_auc": float(compute_metric("roc_auc", target, image_oof)),
            "pure_image_harmonized_crossfit_oof_auc": float(
                compute_metric("roc_auc", target, image_harmonized_oof)
            ),
            "lesion_focus_oof_auc": float(
                compute_metric("roc_auc", target, lesion_oof)
            ),
            "lesion_focus_harmonized_crossfit_oof_auc": float(
                compute_metric("roc_auc", target, lesion_harmonized_oof)
            ),
            "image_metadata_fusion_oof_auc": float(
                compute_metric("roc_auc", target, fusion_oof)
            ),
            "image_metadata_fusion_harmonized_crossfit_oof_auc": float(
                compute_metric("roc_auc", target, fusion_harmonized_oof)
            ),
            "metadata_catboost_oof_auc": float(
                compute_metric("roc_auc", target, metadata_oof)
            ),
            "metadata_catboost_harmonized_crossfit_oof_auc": float(
                compute_metric("roc_auc", target, metadata_harmonized_oof)
            ),
            "final_blend": final_blend,
            "duplicate_group_report": duplicate_group_report,
            "artifact_manifest": artifact_manifest_path.name,
            "artifact_manifest_sha256": artifact_manifest_sha256,
            "metadata_feature_count": len(metadata_names),
            "torch_peak_memory_allocated_mib": max(
                record["peak_memory_allocated_mib"] for record in fold_records
            ),
            "budget": budget,
        },
    )


def calibrate_binary_logits(
    logits: np.ndarray, truth: np.ndarray
) -> tuple[np.ndarray, float, float, float]:
    """Fit temperature and intercept against validation log-loss."""

    from scipy.optimize import minimize
    from scipy.special import expit
    from sklearn.metrics import log_loss

    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(truth, dtype=np.int64).reshape(-1)
    if values.shape != labels.shape or len(np.unique(labels)) != 2:
        raise ValueError("Binary calibration requires aligned logits with both classes")

    def objective(parameters: np.ndarray) -> float:
        temperature = math.exp(float(parameters[0]))
        probability = expit(values / temperature + float(parameters[1]))
        return float(log_loss(labels, np.clip(probability, 1e-7, 1.0 - 1e-7)))

    uncalibrated_score = objective(np.asarray([0.0, 0.0]))
    fitted = minimize(
        objective,
        x0=np.asarray([0.0, 0.0]),
        method="L-BFGS-B",
        bounds=[(-2.3, 2.3), (-3.0, 3.0)],
    )
    temperature = math.exp(float(fitted.x[0]))
    intercept = float(fitted.x[1])
    calibrated = expit(values / temperature + intercept)
    calibrated_score = float(log_loss(labels, np.clip(calibrated, 1e-7, 1.0 - 1e-7)))
    if not fitted.success or calibrated_score > uncalibrated_score + 1e-10:
        temperature, intercept, calibrated_score = 1.0, 0.0, uncalibrated_score
        calibrated = expit(values)
    return calibrated, temperature, intercept, calibrated_score


def apply_binary_logit_calibration(
    logits: np.ndarray, *, temperature: float, intercept: float
) -> np.ndarray:
    from scipy.special import expit

    if not np.isfinite(temperature) or temperature <= 0 or not np.isfinite(intercept):
        raise ValueError("Invalid binary calibration parameters")
    probability = expit(np.asarray(logits, dtype=np.float64) / temperature + intercept)
    if not np.isfinite(probability).all():
        raise RuntimeError("Binary calibration produced non-finite probabilities")
    return np.clip(probability, 1e-6, 1.0 - 1e-6)


def _run_dogs_cats_convnext_single_holdout_legacy(
    args: Any,
    task_dir: Path,
    logger: Any,
) -> dict[str, Any]:
    import torch
    from PIL import Image
    from sklearn.model_selection import train_test_split
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    competition_id = "dogs-vs-cats-redux-kernels-edition"
    resolved = resolve_competition(competition_id, args.data_root)
    train_dir = resolved.public_dir / "train"
    test_dir = resolved.public_dir / "test"
    if not train_dir.is_dir():
        train_dir = task_dir / "cache" / "train"
        wave0.safe_extract_zip(resolved.public_dir / "train.zip", train_dir)
        if (train_dir / "train").is_dir():
            train_dir = train_dir / "train"
    if not test_dir.is_dir():
        test_dir = task_dir / "cache" / "test"
        wave0.safe_extract_zip(resolved.public_dir / "test.zip", test_dir)
        if (test_dir / "test").is_dir():
            test_dir = test_dir / "test"

    paths = sorted(train_dir.glob("*.jpg"))
    labels = np.asarray(
        [1 if path.name.lower().startswith("dog.") else 0 for path in paths],
        dtype=np.int64,
    )
    if len(paths) == 0 or len(np.unique(labels)) != 2:
        raise RuntimeError("Dogs-vs-Cats training images or labels are incomplete")
    indices = np.arange(len(paths))
    train_indices, valid_indices = train_test_split(
        indices,
        test_size=args.holdout_fraction,
        random_state=args.seed,
        stratify=labels,
    )

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [
            transforms.Resize((args.dogs_image_size + 32, args.dogs_image_size + 32)),
            transforms.RandomResizedCrop(args.dogs_image_size, scale=(0.78, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
            transforms.ToTensor(),
            normalize,
        ]
    )
    evaluate_transform = transforms.Compose(
        [
            transforms.Resize((args.dogs_image_size + 24, args.dogs_image_size + 24)),
            transforms.CenterCrop(args.dogs_image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    class Images(Dataset):
        def __init__(self, selected_paths: list[Path], targets: np.ndarray | None, training: bool) -> None:
            self.paths = selected_paths
            self.targets = targets
            self.transform = train_transform if training else evaluate_transform

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, index: int):
            with Image.open(self.paths[index]) as handle:
                image = self.transform(handle.convert("RGB"))
            if self.targets is None:
                return image
            return image, torch.tensor(float(self.targets[index]), dtype=torch.float32)

    workers = min(args.dogs_workers, max(0, (os.cpu_count() or 4) // 4))
    loader_options = {
        "batch_size": args.dogs_batch_size,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        Images([paths[index] for index in train_indices], labels[train_indices], True),
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    valid_loader = DataLoader(
        Images([paths[index] for index in valid_indices], labels[valid_indices], False),
        shuffle=False,
        **loader_options,
    )
    model, pretrained, _ = wave2._vision_model(1)
    model = model.cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.dogs_learning_rate, weight_decay=2e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.dogs_epochs)
    )
    loss_fn = nn.BCEWithLogitsLoss()
    try:
        scaler = torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler()

    def infer_logits(loader: DataLoader, *, tta: bool) -> tuple[np.ndarray, np.ndarray | None]:
        values: list[np.ndarray] = []
        truths: list[np.ndarray] = []
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                images = batch[0] if isinstance(batch, (tuple, list)) else batch
                images = images.cuda(non_blocking=True)
                variants = [images, torch.flip(images, dims=[3])] if tta else [images]
                outputs = []
                for variant in variants:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs.append(model(variant)[:, 0])
                values.append(torch.stack(outputs).mean(dim=0).float().cpu().numpy())
                if isinstance(batch, (tuple, list)):
                    truths.append(batch[1].numpy())
        return np.concatenate(values), (np.concatenate(truths) if truths else None)

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    best_score = math.inf
    best_epoch = 0
    best_temperature = 1.0
    best_intercept = 0.0
    best_path = task_dir / "dogs_cats_convnext_best.pt"
    history: list[dict[str, float]] = []
    for epoch in range(args.dogs_epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, targets in train_loader:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                smoothed_targets = targets * 0.98 + 0.01
                loss = loss_fn(model(images)[:, 0], smoothed_targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach()) * len(images)
            seen += len(images)
        scheduler.step()
        valid_logits, valid_truth = infer_logits(valid_loader, tta=True)
        assert valid_truth is not None
        _, temperature, intercept, score = calibrate_binary_logits(valid_logits, valid_truth)
        if score < best_score:
            best_score = score
            best_epoch = epoch + 1
            best_temperature = temperature
            best_intercept = intercept
            torch.save(model.state_dict(), best_path)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": running_loss / max(1, seen),
                "valid_log_loss_calibrated": score,
                "temperature": temperature,
                "intercept": intercept,
            }
        )
        logger.info(
            "[%s] epoch=%d log_loss=%.6f temperature=%.4f intercept=%.4f",
            competition_id,
            epoch + 1,
            score,
            temperature,
            intercept,
        )

    model.load_state_dict(torch.load(best_path, map_location="cuda", weights_only=True))
    valid_logits, valid_truth = infer_logits(valid_loader, tta=True)
    assert valid_truth is not None
    valid_probability = apply_binary_logit_calibration(
        valid_logits,
        temperature=best_temperature,
        intercept=best_intercept,
    )
    sample = pd.read_csv(resolved.sample_submission_path)
    test_paths = [test_dir / f"{int(identifier)}.jpg" for identifier in sample["id"]]
    if any(not path.is_file() for path in test_paths[: min(64, len(test_paths))]):
        raise FileNotFoundError("Dogs-vs-Cats test image path probe failed")
    test_loader = DataLoader(
        Images(test_paths, None, False),
        shuffle=False,
        **loader_options,
    )
    test_logits, _ = infer_logits(test_loader, tta=True)
    test_probability = apply_binary_logit_calibration(
        test_logits,
        temperature=best_temperature,
        intercept=best_intercept,
    )
    sample["label"] = test_probability
    np.savez_compressed(
        task_dir / "dogs_cats_validation_and_test.npz",
        valid_indices=valid_indices,
        valid_truth=valid_truth,
        valid_logits=valid_logits,
        valid_probability=valid_probability,
        test_logits=test_logits,
        test_probability=test_probability,
    )
    wave0.write_json(
        task_dir / "dogs_cats_training_history.json",
        {
            "history": history,
            "best_epoch": best_epoch,
            "best_log_loss": best_score,
            "temperature": best_temperature,
            "intercept": best_intercept,
        },
    )
    budget = {
        "seed": args.seed,
        "holdout_fraction": args.holdout_fraction,
        "train_images": len(paths),
        "validation_images": len(valid_indices),
        "epochs_requested": args.dogs_epochs,
        "best_epoch": best_epoch,
        "batch_size": args.dogs_batch_size,
        "image_size": args.dogs_image_size,
        "learning_rate": args.dogs_learning_rate,
        "pretrained_weights_loaded": pretrained,
        "temperature": best_temperature,
        "intercept": best_intercept,
        "horizontal_flip_tta": True,
    }
    return wave0.finalize_scored_task(
        competition_id=competition_id,
        submission=sample,
        cv_score=float(best_score),
        args=args,
        task_dir=task_dir,
        budget=budget,
        extra={
            "runtime_seconds_model": time.perf_counter() - started,
            "model_family": "ConvNeXt_Tiny_ImageNet1K_finetune_temperature_bias_calibration_TTA",
            "history": history,
            "torch_peak_memory_allocated_mib": int(torch.cuda.max_memory_allocated() / 2**20),
            "budget": budget,
        },
    )


def multiclass_temperature_scale(probability: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Multiclass probabilities must be a finite matrix")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Multiclass temperature must be positive")
    powered = np.power(np.clip(values, 1e-12, 1.0), 1.0 / temperature)
    return powered / np.maximum(powered.sum(axis=1, keepdims=True), 1e-12)


def apply_multiclass_logloss_blend(
    components: list[np.ndarray], weights: list[float], temperature: float
) -> np.ndarray:
    if len(components) != len(weights) or not components:
        raise ValueError("Multiclass blend component/weight cardinality mismatch")
    matrices = [np.asarray(component, dtype=np.float64) for component in components]
    if any(matrix.shape != matrices[0].shape for matrix in matrices):
        raise ValueError("Multiclass probability components must have identical shapes")
    if not np.isclose(sum(weights), 1.0, atol=1e-8) or any(weight < 0 for weight in weights):
        raise ValueError("Multiclass blend weights must form a nonnegative simplex")
    raw = sum(float(weight) * matrix for weight, matrix in zip(weights, matrices))
    raw /= np.maximum(raw.sum(axis=1, keepdims=True), 1e-12)
    return multiclass_temperature_scale(raw, temperature)


def select_multiclass_logloss_blend(
    components: list[np.ndarray], truth: np.ndarray, classes: list[str]
) -> tuple[np.ndarray, list[float], float, float]:
    from sklearn.metrics import log_loss

    if len(components) != 3:
        raise ValueError("Leaf recovery expects exactly three probability components")
    matrices = [np.asarray(component, dtype=np.float64) for component in components]
    if any(matrix.shape != matrices[0].shape for matrix in matrices):
        raise ValueError("Leaf probability components must have identical shapes")
    best: tuple[np.ndarray, list[float], float, float] | None = None
    grid = np.linspace(0.0, 1.0, 21)
    temperatures = np.geomspace(0.25, 2.5, 41)
    for first in grid:
        for second in grid:
            third = 1.0 - first - second
            if third < -1e-12:
                continue
            weights = [float(first), float(second), float(max(0.0, third))]
            for temperature in temperatures:
                calibrated = apply_multiclass_logloss_blend(
                    matrices, weights, float(temperature)
                )
                score = float(log_loss(truth, calibrated, labels=classes))
                if best is None or score < best[3] - 1e-12:
                    best = (calibrated, weights, float(temperature), score)
    assert best is not None
    return best


def cross_fit_multiclass_logloss_blend(
    components: list[np.ndarray],
    truth: np.ndarray,
    classes: list[str],
    folds: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Select Leaf blend/temperature without each outer validation fold."""

    from sklearn.metrics import log_loss

    target = np.asarray(truth)
    fold_values = np.asarray(folds)
    matrices = [np.asarray(component, dtype=np.float64) for component in components]
    if fold_values.shape != target.shape or any(matrix.shape[0] != len(target) for matrix in matrices):
        raise RuntimeError("Leaf cross-fit arrays have inconsistent row counts")
    prediction = np.full_like(matrices[0], np.nan, dtype=np.float64)
    records: list[dict[str, Any]] = []
    for fold in sorted(int(value) for value in np.unique(fold_values)):
        validation = fold_values == fold
        fitting = ~validation
        _, weights, temperature, fitting_score = select_multiclass_logloss_blend(
            [matrix[fitting] for matrix in matrices], target[fitting], classes
        )
        prediction[validation] = apply_multiclass_logloss_blend(
            [matrix[validation] for matrix in matrices], weights, temperature
        )
        records.append({
            "fold": fold,
            "weights": weights,
            "temperature": temperature,
            "meta_fit_log_loss": fitting_score,
            "outer_log_loss": float(
                log_loss(target[validation], prediction[validation], labels=classes)
            ),
        })
    if not np.isfinite(prediction).all():
        raise RuntimeError("Leaf cross-fit blend did not cover every row")
    return prediction, records


def _aligned_multiclass_probability(
    model: Any, features: pd.DataFrame, classes: list[str]
) -> np.ndarray:
    raw = np.asarray(model.predict_proba(features), dtype=np.float64)
    model_classes = [str(value) for value in model.classes_]
    class_index = {value: index for index, value in enumerate(model_classes)}
    if set(model_classes) != set(classes):
        raise RuntimeError("Leaf model classes do not match the sample submission")
    aligned = np.column_stack([raw[:, class_index[value]] for value in classes])
    aligned = np.clip(aligned, 1e-12, 1.0)
    return aligned / np.maximum(aligned.sum(axis=1, keepdims=True), 1e-12)


def _leaf_id_text(value: Any) -> str:
    """Render a Leaf CSV ID exactly as the corresponding image stem."""

    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    if not text:
        raise ValueError("Leaf image ID is empty")
    return text


def resolve_leaf_image_paths(public_dir: Path, identifiers: Any) -> list[Path]:
    """Resolve and validate the explicit CSV-ID to image-path join."""

    image_dir = public_dir / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Leaf image directory is missing: {image_dir}")
    paths = [image_dir / f"{_leaf_id_text(value)}.jpg" for value in identifiers]
    if len({str(path.resolve()) for path in paths}) != len(paths):
        raise RuntimeError("Leaf image IDs resolve to duplicate paths")
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Leaf image join is incomplete: missing={len(missing)} first={missing[:5]}"
        )
    return paths


def leaf_d4_variant_codes(count: int) -> tuple[tuple[int, bool], ...]:
    """Return deterministic rotation/flip codes for frozen-image embedding TTA."""

    if count == 1:
        return ((0, False),)
    if count == 4:
        return tuple((rotation, False) for rotation in range(4))
    if count == 8:
        return tuple(
            (rotation, flipped)
            for flipped in (False, True)
            for rotation in range(4)
        )
    raise ValueError("Leaf embedding TTA must be one of 1, 4, or 8")


def extract_leaf_convnext_embeddings(
    paths: list[Path],
    *,
    image_size: int,
    batch_size: int,
    worker_count: int,
    tta_count: int,
    seed: int,
    logger: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract orientation-robust frozen ConvNeXt embeddings at high GPU occupancy."""

    import torch
    from PIL import Image, ImageOps
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    if not torch.cuda.is_available():
        raise RuntimeError("Leaf multimodal recovery requires CUDA for frozen embeddings")
    if image_size < 32 or batch_size < 1:
        raise ValueError("Leaf embedding image size and batch size must be positive")
    variants = leaf_d4_variant_codes(tta_count)
    normalize = transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    )
    to_tensor = transforms.ToTensor()
    resampling = getattr(Image, "Resampling", Image).BICUBIC

    class LeafImages(Dataset):
        def __len__(self) -> int:
            return len(paths)

        def __getitem__(self, index: int) -> Any:
            with Image.open(paths[index]) as handle:
                image = handle.convert("RGB")
                contained = ImageOps.contain(
                    image,
                    (image_size, image_size),
                    method=resampling,
                )
            canvas = Image.new("RGB", (image_size, image_size), color=(255, 255, 255))
            canvas.paste(
                contained,
                ((image_size - contained.width) // 2, (image_size - contained.height) // 2),
            )
            return normalize(to_tensor(canvas))

    workers = min(max(0, worker_count), max(0, (os.cpu_count() or 4) // 4))
    generator = torch.Generator().manual_seed(seed)
    loader_options = (
        {"prefetch_factor": wave2.VISION_DATALOADER_PREFETCH_FACTOR}
        if workers > 0
        else {}
    )
    loader = DataLoader(
        LeafImages(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=generator,
        worker_init_fn=wave2._seed_vision_worker,
        **loader_options,
    )

    model, pretrained, weight_identity = wave2._vision_model(
        1,
        require_pretrained=True,
    )
    embedding_dimension = int(model.classifier[2].in_features)
    model.classifier[2] = nn.Identity()
    model = model.to(device="cuda", memory_format=torch.channels_last).eval()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    previous_benchmark = bool(torch.backends.cudnn.benchmark)
    previous_deterministic = bool(torch.are_deterministic_algorithms_enabled())
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)
    torch.cuda.reset_peak_memory_stats()
    rows: list[np.ndarray] = []
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for batch_index, images in enumerate(loader):
                images = images.cuda(
                    non_blocking=True,
                    memory_format=torch.channels_last,
                )
                aggregate = None
                for rotation, flipped in variants:
                    transformed = torch.rot90(images, rotation, dims=(2, 3))
                    if flipped:
                        transformed = torch.flip(transformed, dims=(3,))
                    transformed = transformed.contiguous(memory_format=torch.channels_last)
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        value = model(transformed)
                    aggregate = value.float() if aggregate is None else aggregate + value.float()
                assert aggregate is not None
                aggregate /= len(variants)
                aggregate = torch.nn.functional.normalize(aggregate, dim=1)
                rows.append(aggregate.cpu().numpy())
                logger.info(
                    "[leaf-classification] embedding batch=%d/%d rows=%d",
                    batch_index + 1,
                    len(loader),
                    len(images),
                )
        embeddings = np.concatenate(rows).astype(np.float32, copy=False)
        if embeddings.shape != (len(paths), embedding_dimension):
            raise RuntimeError("Leaf ConvNeXt embedding matrix has an unexpected shape")
        if not np.isfinite(embeddings).all():
            raise RuntimeError("Leaf ConvNeXt embeddings contain non-finite values")
        peak_memory_mib = int(torch.cuda.max_memory_allocated() / 2**20)
    finally:
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=True)
        del model
        torch.cuda.empty_cache()

    digest = hashlib.sha256(np.ascontiguousarray(embeddings).tobytes()).hexdigest()
    return embeddings, {
        "pretrained": pretrained,
        "pretrained_weight_identity": weight_identity,
        "embedding_dimension": embedding_dimension,
        "image_size": image_size,
        "batch_size": batch_size,
        "workers": workers,
        "prefetch_factor": (
            wave2.VISION_DATALOADER_PREFETCH_FACTOR if workers > 0 else None
        ),
        "tta_count": len(variants),
        "tta_codes": [
            {"rotation_quarters": rotation, "horizontal_flip": flipped}
            for rotation, flipped in variants
        ],
        "channels_last": True,
        "amp_dtype": str(amp_dtype),
        "tf32_allowed": True,
        "cudnn_benchmark_during_extraction": True,
        "peak_memory_allocated_mib": peak_memory_mib,
        "embedding_sha256": digest,
        "runtime_seconds": time.perf_counter() - started,
    }


def run_leaf_multimodal(args: Any, task_dir: Path, logger: Any) -> dict[str, Any]:
    """Five-fold numeric, image, and multimodal Leaf ensemble."""

    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import log_loss
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import Normalizer, StandardScaler
    from sklearn.svm import SVC

    competition_id = "leaf-classification"
    resolved = resolve_competition(competition_id, args.data_root)
    train = pd.read_csv(resolved.public_dir / "train.csv")
    test = pd.read_csv(resolved.public_dir / "test.csv")
    sample = pd.read_csv(resolved.sample_submission_path)
    if list(sample.columns[:1]) != ["id"] or "id" not in train or "id" not in test:
        raise RuntimeError("Leaf ID or sample-submission schema is invalid")
    if train["id"].duplicated().any() or test["id"].duplicated().any():
        raise RuntimeError("Leaf train or test IDs are duplicated")
    feature_columns = [column for column in train.columns if column not in {"id", "species"}]
    classes = [str(column) for column in sample.columns[1:]]
    target = train["species"].astype(str).to_numpy()
    if set(target) != set(classes):
        raise RuntimeError("Leaf training labels do not match submission classes")
    numeric_train = train[feature_columns].to_numpy(dtype=np.float64)
    numeric_test = test[feature_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_train).all() or not np.isfinite(numeric_test).all():
        raise RuntimeError("Leaf train or test numeric features contain non-finite values")

    train_paths = resolve_leaf_image_paths(resolved.public_dir, train["id"].tolist())
    test_paths = resolve_leaf_image_paths(resolved.public_dir, test["id"].tolist())
    train_manifest = verify_image_decode_manifest(train_paths, split="train")
    test_manifest = verify_image_decode_manifest(test_paths, split="test")
    manifest = pd.concat([train_manifest, test_manifest], ignore_index=True)
    manifest.insert(
        2,
        "id",
        pd.concat([train["id"], test["id"]], ignore_index=True),
    )
    manifest.to_csv(task_dir / "leaf_image_decode_manifest.csv", index=False)

    all_paths = train_paths + test_paths
    all_embeddings, embedding_report = extract_leaf_convnext_embeddings(
        all_paths,
        image_size=args.leaf_image_size,
        batch_size=args.leaf_embedding_batch_size,
        worker_count=args.wave2_workers,
        tta_count=args.leaf_embedding_tta,
        seed=args.seed,
        logger=logger,
    )
    image_train = all_embeddings[: len(train)]
    image_test = all_embeddings[len(train) :]
    multimodal_train = np.concatenate([numeric_train, image_train], axis=1)
    multimodal_test = np.concatenate([numeric_test, image_test], axis=1)
    np.savez_compressed(
        task_dir / "leaf_convnext_embeddings.npz",
        train_id=train["id"].to_numpy(),
        test_id=test["id"].to_numpy(),
        train_embedding=image_train,
        test_embedding=image_test,
    )

    numeric_count = numeric_train.shape[1]
    image_count = image_train.shape[1]

    def builders(seed: int) -> list[tuple[str, str, Any]]:
        numeric_group = Pipeline([
            ("scale", StandardScaler()),
            ("normalize", Normalizer()),
        ])
        image_group = Normalizer()
        multimodal_preprocessor = ColumnTransformer(
            [
                ("numeric", numeric_group, slice(0, numeric_count)),
                (
                    "image",
                    image_group,
                    slice(numeric_count, numeric_count + image_count),
                ),
            ],
            transformer_weights={"numeric": 1.0, "image": 1.0},
            sparse_threshold=0.0,
        )
        return [
            (
                "numeric_rbf_svc",
                "numeric",
                Pipeline([
                    ("scale", StandardScaler()),
                    (
                        "model",
                        SVC(
                            C=args.leaf_svc_c,
                            gamma="scale",
                            probability=True,
                            random_state=seed,
                            cache_size=4096,
                        ),
                    ),
                ]),
            ),
            (
                "image_cosine_rbf_svc",
                "image",
                Pipeline([
                    ("normalize", Normalizer()),
                    (
                        "model",
                        SVC(
                            C=args.leaf_image_svc_c,
                            gamma="scale",
                            probability=True,
                            random_state=seed,
                            cache_size=4096,
                        ),
                    ),
                ]),
            ),
            (
                "multimodal_group_balanced_rbf_svc",
                "multimodal",
                Pipeline([
                    ("groups", multimodal_preprocessor),
                    (
                        "model",
                        SVC(
                            C=args.leaf_multimodal_svc_c,
                            gamma="scale",
                            probability=True,
                            random_state=seed,
                            cache_size=4096,
                        ),
                    ),
                ]),
            ),
        ]

    feature_sets = {
        "numeric": (numeric_train, numeric_test),
        "image": (image_train, image_test),
        "multimodal": (multimodal_train, multimodal_test),
    }
    minimum_class_count = int(pd.Series(target).value_counts().min())
    fold_count = max(2, min(args.leaf_folds, minimum_class_count))
    splitter = StratifiedKFold(
        n_splits=fold_count,
        shuffle=True,
        random_state=args.seed,
    )
    component_names = [name for name, _, _ in builders(args.seed)]
    oof_components = [
        np.zeros((len(train), len(classes)), dtype=np.float64)
        for _ in component_names
    ]
    fold_assignment = np.full(len(train), -1, dtype=np.int16)
    fold_records: list[dict[str, Any]] = []
    fold_model_dir = task_dir / "leaf_multimodal_fold_models"
    fold_model_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for fold, (train_indices, valid_indices) in enumerate(
        splitter.split(numeric_train, target)
    ):
        fold_assignment[valid_indices] = fold
        record: dict[str, Any] = {
            "fold": fold + 1,
            "train_rows": len(train_indices),
            "valid_rows": len(valid_indices),
        }
        for model_index, (name, feature_kind, model) in enumerate(
            builders(args.seed + fold)
        ):
            train_features = feature_sets[feature_kind][0]
            model.fit(train_features[train_indices], target[train_indices])
            probability = _aligned_multiclass_probability(
                model,
                train_features[valid_indices],
                classes,
            )
            oof_components[model_index][valid_indices] = probability
            record[f"{name}_log_loss"] = float(
                log_loss(target[valid_indices], probability, labels=classes)
            )
            joblib.dump(
                model,
                fold_model_dir / f"fold_{fold}_{name}.joblib",
                compress=3,
            )
        fold_records.append(record)
        logger.info(
            "[%s] fold=%d numeric=%.6f image=%.6f multimodal=%.6f",
            competition_id,
            fold + 1,
            record["numeric_rbf_svc_log_loss"],
            record["image_cosine_rbf_svc_log_loss"],
            record["multimodal_group_balanced_rbf_svc_log_loss"],
        )

    if np.any(fold_assignment < 0):
        raise RuntimeError("Leaf multimodal fold assignment did not cover every row")
    crossfit_probability, crossfit_records = cross_fit_multiclass_logloss_blend(
        oof_components,
        target,
        classes,
        fold_assignment,
    )
    _, weights, temperature, final_meta_fit_score = select_multiclass_logloss_blend(
        oof_components,
        target,
        classes,
    )
    cv_score = float(log_loss(target, crossfit_probability, labels=classes))

    final_components: list[np.ndarray] = []
    final_models: dict[str, Any] = {}
    for name, feature_kind, model in builders(args.seed):
        train_features, test_features = feature_sets[feature_kind]
        model.fit(train_features, target)
        final_models[name] = model
        final_components.append(
            _aligned_multiclass_probability(model, test_features, classes)
        )
    test_probability = apply_multiclass_logloss_blend(
        final_components,
        weights,
        temperature,
    )
    sample = align_multiclass_submission(
        sample,
        test["id"],
        test_probability,
        classes,
    )
    joblib.dump(
        final_models,
        task_dir / "leaf_multimodal_ensemble.joblib",
        compress=3,
    )
    np.savez_compressed(
        task_dir / "leaf_multimodal_oof_components.npz",
        id=train["id"].to_numpy(),
        target=target,
        classes=np.asarray(classes),
        fold_assignment=fold_assignment,
        numeric=oof_components[0],
        image=oof_components[1],
        multimodal=oof_components[2],
        crossfit_probability=crossfit_probability,
        weights=np.asarray(weights),
        temperature=np.asarray([temperature]),
        test_id=test["id"].to_numpy(),
        test_probability=test_probability,
    )
    diagnostics = {
        "schema": "evomind.mlebench_lite.leaf_multimodal.v1",
        "folds": fold_records,
        "component_names": component_names,
        "weights": dict(zip(component_names, weights)),
        "temperature": temperature,
        "cross_fitted_oof_log_loss": cv_score,
        "final_meta_fit_log_loss": final_meta_fit_score,
        "crossfit_blends": crossfit_records,
        "embedding": embedding_report,
        "explicit_id_join": True,
        "private_labels_used": False,
        "private_scores_used_for_tuning": False,
    }
    wave0.write_json(task_dir / "leaf_multimodal_diagnostics.json", diagnostics)
    budget = {
        "seed": args.seed,
        "folds": fold_count,
        "train_rows": len(train),
        "test_rows": len(test),
        "numeric_feature_count": numeric_count,
        "image_embedding_count": image_count,
        "class_count": len(classes),
        "numeric_svc_c": args.leaf_svc_c,
        "image_svc_c": args.leaf_image_svc_c,
        "multimodal_svc_c": args.leaf_multimodal_svc_c,
        "blend_weights": weights,
        "temperature": temperature,
        "full_refit": True,
        "fold_models_persisted": True,
        "cross_fitted_meta_validation": True,
        "explicit_id_join": True,
        "frozen_convnext_embeddings": True,
    }
    promotion_gate = wave0.build_metric_promotion_gate(
        name="leaf_multimodal_cross_fitted_log_loss",
        metric="log_loss",
        direction="minimize",
        score=float(cv_score),
        threshold=0.0135,
        evidence={
            "final_meta_fit_log_loss": float(final_meta_fit_score),
            "cross_fitted_oof_log_loss": float(cv_score),
            "probability_rows_sum_to_one": bool(
                np.allclose(crossfit_probability.sum(axis=1), 1.0, atol=1e-8)
            ),
        },
    )
    return wave0.finalize_scored_task(
        competition_id=competition_id,
        submission=sample,
        cv_score=cv_score,
        args=args,
        task_dir=task_dir,
        budget=budget,
        promotion_gate=promotion_gate,
        extra={
            "runtime_seconds_model": time.perf_counter() - started,
            "model_family": (
                f"{fold_count}fold_numeric_image_multimodal_ConvNeXt_"
                "RBF_SVC_crossfit_blend"
            ),
            "folds": fold_records,
            "crossfit_blends": crossfit_records,
            "embedding": embedding_report,
            "budget": budget,
        },
    )


def run_leaf_oof_ensemble(args: Any, task_dir: Path, logger: Any) -> dict[str, Any]:
    import joblib
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    competition_id = "leaf-classification"
    resolved = resolve_competition(competition_id, args.data_root)
    train = pd.read_csv(resolved.public_dir / "train.csv")
    test = pd.read_csv(resolved.public_dir / "test.csv")
    sample = pd.read_csv(resolved.sample_submission_path)
    if list(sample.columns[:1]) != ["id"] or "id" not in train or "id" not in test:
        raise RuntimeError("Leaf ID or sample-submission schema is invalid")
    if train["id"].duplicated().any() or test["id"].duplicated().any():
        raise RuntimeError("Leaf train or test IDs are duplicated")
    feature_columns = [column for column in train.columns if column not in {"id", "species"}]
    classes = [str(column) for column in sample.columns[1:]]
    target = train["species"].astype(str).to_numpy()
    if set(target) != set(classes):
        raise RuntimeError("Leaf training labels do not match submission classes")
    train_features = train[feature_columns].astype(np.float64)
    test_features = test[feature_columns].astype(np.float64)
    if not np.isfinite(train_features.to_numpy()).all() or not np.isfinite(test_features.to_numpy()).all():
        raise RuntimeError("Leaf train or test features contain non-finite values")

    def builders(seed: int):
        return [
            (
                "rbf_svc",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "model",
                            SVC(
                                C=args.leaf_svc_c,
                                gamma="scale",
                                probability=True,
                                random_state=seed,
                                cache_size=4096,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "shrinkage_lda",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "model",
                            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                        ),
                    ]
                ),
            ),
            (
                "multinomial_logistic",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "model",
                            LogisticRegression(
                                C=args.leaf_logistic_c,
                                solver="lbfgs",
                                max_iter=2500,
                                random_state=seed,
                            ),
                        ),
                    ]
                ),
            ),
        ]

    minimum_class_count = int(pd.Series(target).value_counts().min())
    fold_count = max(2, min(args.leaf_folds, minimum_class_count))
    splitter = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=args.seed)
    oof_components = [np.zeros((len(train), len(classes)), dtype=np.float64) for _ in range(3)]
    fold_assignment = np.full(len(train), -1, dtype=np.int16)
    fold_records: list[dict[str, Any]] = []
    fold_model_dir = task_dir / "leaf_fold_models"
    fold_model_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for fold, (train_indices, valid_indices) in enumerate(splitter.split(train_features, target)):
        fold_assignment[valid_indices] = fold
        record: dict[str, Any] = {"fold": fold + 1, "train_rows": len(train_indices), "valid_rows": len(valid_indices)}
        for model_index, (name, model) in enumerate(builders(args.seed + fold)):
            model.fit(train_features.iloc[train_indices], target[train_indices])
            probability = _aligned_multiclass_probability(
                model, train_features.iloc[valid_indices], classes
            )
            oof_components[model_index][valid_indices] = probability
            from sklearn.metrics import log_loss

            record[f"{name}_log_loss"] = float(
                log_loss(target[valid_indices], probability, labels=classes)
            )
            joblib.dump(
                model,
                fold_model_dir / f"fold_{fold}_{name}.joblib",
                compress=3,
            )
        fold_records.append(record)
        logger.info(
            "[%s] fold=%d svc=%.6f lda=%.6f logistic=%.6f",
            competition_id,
            fold + 1,
            record["rbf_svc_log_loss"],
            record["shrinkage_lda_log_loss"],
            record["multinomial_logistic_log_loss"],
        )

    if np.any(fold_assignment < 0):
        raise RuntimeError("Leaf fold assignment did not cover every training row")
    crossfit_probability, crossfit_records = cross_fit_multiclass_logloss_blend(
        oof_components, target, classes, fold_assignment
    )
    _, weights, temperature, final_meta_fit_score = select_multiclass_logloss_blend(
        oof_components, target, classes
    )
    from sklearn.metrics import log_loss

    cv_score = float(log_loss(target, crossfit_probability, labels=classes))
    final_components: list[np.ndarray] = []
    final_models: dict[str, Any] = {}
    for name, model in builders(args.seed):
        model.fit(train_features, target)
        final_models[name] = model
        final_components.append(_aligned_multiclass_probability(model, test_features, classes))
    test_probability = apply_multiclass_logloss_blend(
        final_components, weights, temperature
    )
    sample = align_multiclass_submission(sample, test["id"], test_probability, classes)
    joblib.dump(final_models, task_dir / "leaf_oof_ensemble.joblib", compress=3)
    np.savez_compressed(
        task_dir / "leaf_oof_components.npz",
        id=train["id"].to_numpy(),
        target=target,
        classes=np.asarray(classes),
        fold_assignment=fold_assignment,
        svc=oof_components[0],
        lda=oof_components[1],
        logistic=oof_components[2],
        crossfit_probability=crossfit_probability,
        weights=np.asarray(weights),
        temperature=np.asarray([temperature]),
        test_id=test["id"].to_numpy(),
        test_probability=test_probability,
    )
    wave0.write_json(
        task_dir / "leaf_ensemble_diagnostics.json",
        {
            "folds": fold_records,
            "weights": {
                "rbf_svc": weights[0],
                "shrinkage_lda": weights[1],
                "multinomial_logistic": weights[2],
            },
            "temperature": temperature,
            "oof_log_loss": cv_score,
            "final_meta_fit_log_loss": final_meta_fit_score,
            "crossfit_blends": crossfit_records,
        },
    )
    budget = {
        "seed": args.seed,
        "folds": fold_count,
        "train_rows": len(train),
        "test_rows": len(test),
        "feature_count": len(feature_columns),
        "class_count": len(classes),
        "svc_c": args.leaf_svc_c,
        "logistic_c": args.leaf_logistic_c,
        "blend_weights": weights,
        "temperature": temperature,
        "full_refit": True,
        "fold_models_persisted": True,
        "cross_fitted_meta_validation": True,
        "explicit_id_join": True,
    }
    promotion_gate = wave0.build_metric_promotion_gate(
        name="leaf_numeric_cross_fitted_log_loss",
        metric="log_loss",
        direction="minimize",
        score=float(cv_score),
        threshold=0.0135,
        evidence={
            "final_meta_fit_log_loss": float(final_meta_fit_score),
            "cross_fitted_oof_log_loss": float(cv_score),
            "probability_rows_sum_to_one": bool(
                np.allclose(crossfit_probability.sum(axis=1), 1.0, atol=1e-8)
            ),
        },
    )
    return wave0.finalize_scored_task(
        competition_id=competition_id,
        submission=sample,
        cv_score=float(cv_score),
        args=args,
        task_dir=task_dir,
        budget=budget,
        promotion_gate=promotion_gate,
        extra={
            "runtime_seconds_model": time.perf_counter() - started,
            "model_family": (
                f"{fold_count}fold_RBF_SVC_shrinkage_LDA_"
                "multinomial_Logistic_OOF_blend_temperature"
            ),
            "folds": fold_records,
            "crossfit_blends": crossfit_records,
            "budget": budget,
        },
    )


_DOGS_CATS_NAME = re.compile(r"^(cat|dog)\.(\d+)\.jpg$", re.IGNORECASE)


def strict_dogs_cats_labels(paths: list[Path]) -> np.ndarray:
    """Validate the complete competition filename contract before assigning labels."""

    labels: list[int] = []
    observed: set[tuple[str, int]] = set()
    for path in paths:
        match = _DOGS_CATS_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected Dogs-vs-Cats training filename: {path.name}")
        key = (match.group(1).lower(), int(match.group(2)))
        if key in observed:
            raise ValueError(f"Duplicate Dogs-vs-Cats training identity: {path.name}")
        observed.add(key)
        labels.append(1 if key[0] == "dog" else 0)
    result = np.asarray(labels, dtype=np.int64)
    if len(result) < 2 or set(result.tolist()) != {0, 1}:
        raise RuntimeError("Dogs-vs-Cats requires complete cat and dog training classes")
    return result


def _resolve_zipped_image_directory(
    *,
    public_dir: Path,
    task_dir: Path,
    name: str,
) -> Path:
    direct = public_dir / name
    if direct.is_dir():
        return direct
    archive = public_dir / f"{name}.zip"
    extracted = task_dir / "cache" / name
    wave0.safe_extract_zip(archive, extracted)
    nested = extracted / name
    result = nested if nested.is_dir() else extracted
    if not result.is_dir():
        raise FileNotFoundError(f"Extracted image directory is missing: {result}")
    return result


def _decode_manifest_record(
    item: tuple[int, Path, str, int | None],
) -> dict[str, Any]:
    """Decode and hash one image without sharing PIL state across workers."""

    from PIL import Image

    index, path, split, label = item
    if not path.is_file():
        raise FileNotFoundError(f"Image decode manifest is missing {path}")
    try:
        encoded = path.read_bytes()
        digest = hashlib.sha256(encoded).hexdigest()
        with Image.open(io.BytesIO(encoded)) as handle:
            handle.verify()
        with Image.open(io.BytesIO(encoded)) as handle:
            source_mode = handle.mode
            rgb = handle.convert("RGB")
            rgb.load()
            width, height = rgb.size
    except Exception as exc:
        raise RuntimeError(f"Image decode failed for {path.name}") from exc
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Image has invalid dimensions: {path.name}")
    return {
        "split": split,
        "index": index,
        "path": str(path),
        "filename": path.name,
        "sha256": digest,
        "width": int(width),
        "height": int(height),
        "source_mode": str(source_mode),
        "rgb_decode": True,
        "label": label,
    }


def verify_image_decode_manifest(
    paths: list[Path],
    *,
    split: str,
    labels: np.ndarray | None = None,
    workers: int | None = None,
) -> pd.DataFrame:
    """Decode and hash every image through a bounded parallel I/O pipeline."""

    if labels is not None and len(labels) != len(paths):
        raise RuntimeError("Image decode manifest label cardinality mismatch")
    normalized = [str(path.resolve()) for path in paths]
    if len(set(normalized)) != len(normalized):
        raise RuntimeError("Image decode manifest contains duplicate paths")
    if not paths:
        return pd.DataFrame.from_records([])
    label_values = None if labels is None else np.asarray(labels).reshape(-1)
    items = [
        (
            index,
            path,
            split,
            None if label_values is None else int(label_values[index]),
        )
        for index, path in enumerate(paths)
    ]
    worker_count = max(1, min(int(workers or 64), len(items)))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        records = list(pool.map(_decode_manifest_record, items, chunksize=16))
    return pd.DataFrame.from_records(records)


def run_dogs_cats_convnext(args: Any, task_dir: Path, logger: Any) -> dict[str, Any]:
    """Verified-pretrained five-fold Dogs-vs-Cats ensemble with full OOF coverage."""

    competition_id = "dogs-vs-cats-redux-kernels-edition"
    resolved = resolve_competition(competition_id, args.data_root)
    train_dir = _resolve_zipped_image_directory(
        public_dir=resolved.public_dir,
        task_dir=task_dir,
        name="train",
    )
    test_dir = _resolve_zipped_image_directory(
        public_dir=resolved.public_dir,
        task_dir=task_dir,
        name="test",
    )
    train_paths = sorted(train_dir.glob("*.jpg"))
    labels = strict_dogs_cats_labels(train_paths)
    train = pd.DataFrame({"path": [str(path) for path in train_paths], "label": labels})
    sample = pd.read_csv(resolved.sample_submission_path)
    if list(sample.columns) != ["id", "label"] or sample["id"].duplicated().any():
        raise RuntimeError("Dogs-vs-Cats sample-submission schema or IDs are invalid")
    try:
        test_ids = sample["id"].map(lambda value: int(value)).tolist()
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Dogs-vs-Cats sample IDs must be integer image IDs") from exc
    test_paths = [test_dir / f"{identifier}.jpg" for identifier in test_ids]
    train_decode_manifest = verify_image_decode_manifest(
        train_paths, split="train", labels=labels, workers=args.dogs_workers
    )
    conflicts = train_decode_manifest.groupby("sha256")["label"].nunique()
    if bool((conflicts > 1).any()):
        raise RuntimeError("Content-identical Dogs-vs-Cats images have conflicting labels")
    duplicate_mask = train_decode_manifest["sha256"].duplicated(keep=False)
    duplicate_groups = (
        train_decode_manifest["sha256"].astype(str).to_numpy()
        if bool(duplicate_mask.any())
        else None
    )
    train_decode_manifest["duplicate_group"] = duplicate_mask
    train_decode_manifest["split_group_id"] = train_decode_manifest["sha256"].astype(str)
    test_decode_manifest = verify_image_decode_manifest(
        test_paths, split="test", workers=args.dogs_workers
    )
    test_decode_manifest["duplicate_group"] = False
    test_decode_manifest["split_group_id"] = ""
    cross_split_hashes = sorted(
        set(train_decode_manifest["sha256"].astype(str))
        & set(test_decode_manifest["sha256"].astype(str))
    )
    wave0.write_json(task_dir / "dogs_image_hash_audit.json", {
        "schema": "evomind.mlebench_lite.dogs_image_hash_audit.v1",
        "train_images": len(train_decode_manifest),
        "test_images": len(test_decode_manifest),
        "train_duplicate_rows": int(duplicate_mask.sum()),
        "train_duplicate_hashes": int(
            train_decode_manifest.loc[duplicate_mask, "sha256"].nunique()
        ),
        "train_test_exact_match_count": len(cross_split_hashes),
        "train_test_exact_match_hashes": cross_split_hashes,
        "label_transfer_used": False,
    })
    decode_manifest = pd.concat([
        train_decode_manifest,
        test_decode_manifest,
    ], ignore_index=True)
    decode_manifest.to_csv(task_dir / "dogs_image_decode_manifest.csv", index=False)
    return wave2._run_vision(
        args=args,
        task_dir=task_dir,
        logger=logger,
        competition_id=competition_id,
        train_frame=train,
        sample=sample,
        train_paths=train_paths,
        test_paths=test_paths,
        target_columns=["label"],
        label_column="label",
        mode="binary",
        binary_metric="log_loss",
        epochs=args.dogs_epochs,
        image_size=args.dogs_image_size,
        batch_size=args.dogs_batch_size,
        vertical_flip=False,
        tta_flips=True,
        fold_count=args.dogs_folds,
        learning_rate=args.dogs_learning_rate,
        worker_count=args.dogs_workers,
        group_values=duplicate_groups,
        promotion_contract={
            "name": "dogs_single_seed_oof_log_loss",
            "metric": "log_loss",
            "direction": "minimize",
            "threshold": 0.055,
            "require_all_folds": False,
        },
    )


def _path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_exact_image_hash_groups(
    paths: list[Path],
    labels: np.ndarray,
) -> tuple[np.ndarray | None, pd.DataFrame]:
    """Return exact-content groups only when duplicates exist; reject label conflicts."""

    if len(paths) != len(labels):
        raise ValueError("Image hash manifest path/label cardinality mismatch")
    hashes = [_path_sha256(path) for path in paths]
    manifest = pd.DataFrame({
        "path": [str(path) for path in paths],
        "sha256": hashes,
        "label": np.asarray(labels, dtype=int),
    })
    conflicts = manifest.groupby("sha256")["label"].nunique()
    if bool((conflicts > 1).any()):
        raise RuntimeError("Content-identical training images have conflicting labels")
    duplicate_rows = int(manifest["sha256"].duplicated(keep=False).sum())
    manifest["duplicate_group"] = manifest["sha256"].duplicated(keep=False)
    return (np.asarray(hashes) if duplicate_rows else None), manifest


def run_aerial_cactus_convnext(args: Any, task_dir: Path, logger: Any) -> dict[str, Any]:
    """Pretrained stratified fold ensemble for the perfect-AUC cactus target."""

    competition_id = "aerial-cactus-identification"
    resolved = resolve_competition(competition_id, args.data_root)
    train = pd.read_csv(resolved.public_dir / "train.csv").reset_index(drop=True)
    sample = pd.read_csv(resolved.sample_submission_path)
    if list(train.columns) != ["id", "has_cactus"]:
        raise RuntimeError("Aerial Cactus training schema is unsupported")
    if list(sample.columns) != ["id", "has_cactus"] or sample["id"].duplicated().any():
        raise RuntimeError("Aerial Cactus sample-submission schema or IDs are invalid")
    labels = train["has_cactus"].astype(int).to_numpy()
    if set(labels.tolist()) != {0, 1}:
        raise RuntimeError("Aerial Cactus requires both binary classes")
    train_dir = resolved.public_dir / "train"
    test_dir = resolved.public_dir / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        cache = task_dir / "cache"
        train_dir = cache / "train"
        test_dir = cache / "test"
        wave0.safe_extract_zip(resolved.public_dir / "train.zip", train_dir)
        wave0.safe_extract_zip(resolved.public_dir / "test.zip", test_dir)
    train_paths = [train_dir / str(identifier) for identifier in train["id"]]
    test_paths = [test_dir / str(identifier) for identifier in sample["id"]]
    missing = [str(path) for path in train_paths + test_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Aerial Cactus image manifest is incomplete: missing={len(missing)}")
    groups, hash_manifest = build_exact_image_hash_groups(train_paths, labels)
    hash_manifest.to_csv(task_dir / "aerial_exact_hash_manifest.csv", index=False)
    return wave2._run_vision(
        args=args,
        task_dir=task_dir,
        logger=logger,
        competition_id=competition_id,
        train_frame=train,
        sample=sample,
        train_paths=train_paths,
        test_paths=test_paths,
        target_columns=["has_cactus"],
        label_column="has_cactus",
        mode="binary",
        binary_metric="roc_auc",
        epochs=args.aerial_epochs,
        backbone=getattr(args, "aerial_backbone", "convnext_tiny"),
        image_size=args.aerial_image_size,
        batch_size=args.aerial_batch_size,
        vertical_flip=True,
        tta_flips=True,
        group_values=groups,
        fold_count=args.aerial_folds,
        promotion_contract={
            "name": "aerial_perfect_oof_ranking",
            "metric": "roc_auc",
            "direction": "maximize",
            "threshold": 1.0,
            "require_all_folds": True,
            "require_zero_binary_ranking_violations": True,
        },
    )


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(values)
    probability = exponent / np.maximum(exponent.sum(axis=1, keepdims=True), 1e-12)
    if not np.isfinite(probability).all():
        raise RuntimeError("Multiclass softmax produced non-finite values")
    return probability


_SPOOKY_FUNCTION_WORDS = (
    "a", "about", "after", "all", "and", "as", "at", "be", "been", "before",
    "but", "by", "could", "do", "for", "from", "had", "has", "have", "he",
    "her", "him", "his", "i", "if", "in", "into", "is", "it", "its", "me",
    "my", "no", "not", "of", "on", "or", "our", "she", "so", "than", "that",
    "the", "their", "them", "then", "there", "they", "this", "to", "up", "was",
    "we", "were", "what", "when", "which", "who", "with", "would", "you", "your",
)
_SPOOKY_PUNCTUATION = {
    "period": ".",
    "comma": ",",
    "semicolon": ";",
    "colon": ":",
    "question": "?",
    "exclamation": "!",
    "apostrophe": "'",
    "quote": '"',
    "dash": "-",
    "left_parenthesis": "(",
    "right_parenthesis": ")",
}


def build_spooky_stylometric_features(texts: pd.Series) -> pd.DataFrame:
    """Create deterministic label-free author-style features."""

    records: list[dict[str, float]] = []
    for value in texts.fillna("").astype(str):
        lower = value.lower()
        words = re.findall(r"[a-z]+(?:'[a-z]+)?", lower)
        word_count = max(1, len(words))
        char_count = max(1, len(value))
        alphabetic_count = max(1, sum(character.isalpha() for character in value))
        word_lengths = np.asarray([len(word) for word in words], dtype=np.float64)
        sentence_count = max(1, len(re.findall(r"[.!?]+", value)))
        token_counts = Counter(words)
        character_counts = Counter(value)
        character_probability = np.asarray(
            [count / char_count for count in character_counts.values()],
            dtype=np.float64,
        )
        record: dict[str, float] = {
            "char_count_log1p": float(np.log1p(len(value))),
            "word_count_log1p": float(np.log1p(len(words))),
            "sentence_count_log1p": float(np.log1p(sentence_count)),
            "words_per_sentence": float(len(words) / sentence_count),
            "mean_word_length": float(word_lengths.mean()) if len(word_lengths) else 0.0,
            "std_word_length": float(word_lengths.std()) if len(word_lengths) else 0.0,
            "max_word_length": float(word_lengths.max()) if len(word_lengths) else 0.0,
            "vocabulary_richness": float(len(token_counts) / word_count),
            "hapax_ratio": float(sum(count == 1 for count in token_counts.values()) / word_count),
            "uppercase_ratio": float(sum(character.isupper() for character in value) / alphabetic_count),
            "digit_ratio": float(sum(character.isdigit() for character in value) / char_count),
            "whitespace_ratio": float(sum(character.isspace() for character in value) / char_count),
            "newline_ratio": float(value.count("\n") / char_count),
            "character_entropy": float(
                -np.sum(character_probability * np.log(np.maximum(character_probability, 1e-12)))
            ),
            "quoted_span_count_log1p": float(np.log1p(value.count('"') // 2)),
            "contraction_ratio": float(sum("'" in word for word in words) / word_count),
        }
        for name, punctuation in _SPOOKY_PUNCTUATION.items():
            count = value.count(punctuation)
            record[f"punct_{name}_per_char"] = float(count / char_count)
            record[f"punct_{name}_per_word"] = float(count / word_count)
        for word in _SPOOKY_FUNCTION_WORDS:
            record[f"function_{word}"] = float(token_counts.get(word, 0) / word_count)
        records.append(record)
    result = pd.DataFrame.from_records(records, index=texts.index).astype(np.float64)
    if result.empty or not np.isfinite(result.to_numpy()).all():
        raise RuntimeError("Spooky stylometric features are empty or non-finite")
    return result


def fit_spooky_stylometric_channel(
    fit_frame: pd.DataFrame,
    fit_labels: np.ndarray,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    classes: list[str],
    *,
    c_value: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit a fold-local scaled multinomial style classifier."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if list(fit_frame.columns) != list(valid_frame.columns) or list(fit_frame.columns) != list(test_frame.columns):
        raise RuntimeError("Spooky stylometric train/validation/test schemas differ")
    scaler = StandardScaler()
    fit_values = scaler.fit_transform(fit_frame)
    valid_values = scaler.transform(valid_frame)
    test_values = scaler.transform(test_frame)
    model = LogisticRegression(
        C=c_value,
        max_iter=1_000,
        solver="lbfgs",
        random_state=seed,
    )
    model.fit(fit_values, fit_labels)
    class_index = {str(value): index for index, value in enumerate(model.classes_)}
    if set(class_index) != set(classes):
        raise RuntimeError("Spooky stylometric model class order is incomplete")
    order = [class_index[value] for value in classes]
    valid_probability = model.predict_proba(valid_values)[:, order]
    test_probability = model.predict_proba(test_values)[:, order]
    return valid_probability, test_probability, {
        "scaler": scaler,
        "model": model,
        "classes": classes,
        "feature_columns": list(fit_frame.columns),
    }


def fit_spooky_nbsvm_channel(
    fit_matrix: Any,
    fit_labels: np.ndarray,
    valid_matrix: Any,
    test_matrix: Any,
    classes: list[str],
    *,
    c_value: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    """Fit fold-local one-vs-rest NB-SVM models for one sparse channel."""

    from sklearn.linear_model import LogisticRegression

    valid_logits = np.zeros((valid_matrix.shape[0], len(classes)), dtype=np.float64)
    test_logits = np.zeros((test_matrix.shape[0], len(classes)), dtype=np.float64)
    models: list[Any] = []
    for class_index, class_name in enumerate(classes):
        binary = (fit_labels == class_name).astype(np.int8)
        ratio = wave2.compute_nb_log_count_ratio(fit_matrix, binary)
        model = LogisticRegression(
            C=c_value,
            max_iter=500,
            solver="liblinear",
            random_state=seed + class_index,
        )
        model.fit(fit_matrix.multiply(ratio), binary)
        valid_logits[:, class_index] = model.decision_function(valid_matrix.multiply(ratio))
        test_logits[:, class_index] = model.decision_function(test_matrix.multiply(ratio))
        models.append({"class": class_name, "ratio": ratio, "model": model})
    return _stable_softmax(valid_logits), _stable_softmax(test_logits), models


def normalize_spooky_text(value: Any) -> str:
    """Return a stable key for duplicate-aware Spooky Author folds."""

    normalized = unicodedata.normalize("NFKC", str(value)).lower()
    normalized = normalized.replace("\u2018", "'").replace("\u2019", "'")
    return " ".join(re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+", normalized))


def build_spooky_duplicate_groups(
    texts: pd.Series,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Group normalized duplicates and reject contradictory public labels."""

    normalized = texts.fillna("").astype(str).map(normalize_spooky_text)
    hashes = normalized.map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
    if labels is not None:
        label_values = np.asarray(labels).astype(str)
        if len(label_values) != len(hashes):
            raise ValueError("Spooky duplicate labels do not match the text rows")
        conflicts = (
            pd.DataFrame({"text_hash": hashes, "label": label_values})
            .groupby("text_hash", sort=False)["label"]
            .nunique()
        )
        if bool((conflicts > 1).any()):
            raise RuntimeError("Spooky normalized duplicate group has conflicting labels")
    group_ids, unique_hashes = pd.factorize(hashes, sort=True)
    counts = pd.Series(group_ids).value_counts()
    report = {
        "schema": "evomind.mlebench.spooky_duplicate_groups.v1",
        "rows": len(texts),
        "unique_groups": int(len(unique_hashes)),
        "duplicate_groups": int((counts > 1).sum()),
        "duplicate_rows": int((counts - 1).clip(lower=0).sum()),
        "maximum_group_size": int(counts.max()) if len(counts) else 0,
        "conflicting_label_groups": 0,
        "normalization": "NFKC_lower_word_number_tokens_v1",
    }
    return group_ids.astype(np.int32, copy=False), report


def apply_spooky_probability_blend(
    word_probability: np.ndarray,
    char_probability: np.ndarray,
    *,
    word_weight: float,
    temperature: float,
) -> np.ndarray:
    if not 0.0 <= word_weight <= 1.0 or temperature <= 0:
        raise ValueError("Invalid Spooky blend parameters")
    blended = (
        word_weight * np.asarray(word_probability, dtype=np.float64)
        + (1.0 - word_weight) * np.asarray(char_probability, dtype=np.float64)
    )
    return _stable_softmax(np.log(np.clip(blended, 1e-12, 1.0)) / temperature)


def select_spooky_probability_blend(
    word_probability: np.ndarray,
    char_probability: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
) -> tuple[float, float, float]:
    from sklearn.metrics import log_loss

    best: tuple[float, float, float] | None = None
    for word_weight in np.linspace(0.2, 0.8, 7):
        for temperature in (0.75, 0.9, 1.0, 1.1, 1.25):
            probability = apply_spooky_probability_blend(
                word_probability,
                char_probability,
                word_weight=float(word_weight),
                temperature=float(temperature),
            )
            score = float(log_loss(labels, probability, labels=classes))
            candidate = (score, float(word_weight), float(temperature))
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return best


def cross_fit_spooky_probability_blend(
    word_oof: np.ndarray,
    char_oof: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    folds: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    prediction = np.zeros_like(word_oof, dtype=np.float64)
    records: list[dict[str, float]] = []
    for fold in sorted(int(value) for value in np.unique(folds)):
        validation = folds == fold
        fitting = ~validation
        _, word_weight, temperature = select_spooky_probability_blend(
            word_oof[fitting], char_oof[fitting], labels[fitting], classes
        )
        prediction[validation] = apply_spooky_probability_blend(
            word_oof[validation],
            char_oof[validation],
            word_weight=word_weight,
            temperature=temperature,
        )
        records.append({
            "fold": float(fold),
            "word_weight": word_weight,
            "temperature": temperature,
        })
    return prediction, records


def apply_spooky_multicomponent_blend(
    components: list[np.ndarray],
    *,
    weights: list[float] | tuple[float, ...],
    temperature: float,
    mode: str = "arithmetic",
) -> np.ndarray:
    if len(components) < 2 or len(components) != len(weights) or temperature <= 0:
        raise ValueError("Invalid Spooky multicomponent blend contract")
    weight_array = np.asarray(weights, dtype=np.float64)
    if np.any(weight_array < 0) or not np.isclose(weight_array.sum(), 1.0, atol=1e-8):
        raise ValueError("Spooky multicomponent weights must be nonnegative and sum to one")
    matrices = [np.asarray(component, dtype=np.float64) for component in components]
    if any(matrix.shape != matrices[0].shape for matrix in matrices):
        raise RuntimeError("Spooky multicomponent probability shapes differ")
    if any(not np.isfinite(matrix).all() for matrix in matrices):
        raise RuntimeError("Spooky multicomponent probabilities contain non-finite values")
    if mode == "arithmetic":
        blended = np.zeros_like(matrices[0], dtype=np.float64)
        for weight, matrix in zip(weight_array, matrices, strict=True):
            blended += weight * matrix
        logits = np.log(np.clip(blended, 1e-12, 1.0))
    elif mode == "log_probability":
        logits = np.zeros_like(matrices[0], dtype=np.float64)
        for weight, matrix in zip(weight_array, matrices, strict=True):
            logits += weight * np.log(np.clip(matrix, 1e-12, 1.0))
    else:
        raise ValueError(f"Unsupported Spooky blend mode: {mode}")
    return _stable_softmax(logits / temperature)


def _spooky_simplex_weights(component_count: int, steps: int = 10) -> list[list[float]]:
    if component_count < 2 or steps < 1:
        raise ValueError("Spooky simplex requires at least two components and one step")
    values: list[list[float]] = []

    def visit(prefix: list[int], remaining_components: int, remaining_steps: int) -> None:
        if remaining_components == 1:
            values.append([*(value / steps for value in prefix), remaining_steps / steps])
            return
        for value in range(remaining_steps + 1):
            visit([*prefix, value], remaining_components - 1, remaining_steps - value)

    visit([], component_count, steps)
    return values


def select_spooky_multicomponent_blend(
    components: list[np.ndarray],
    labels: np.ndarray,
    classes: list[str],
) -> tuple[float, list[float], float, str]:
    from sklearn.metrics import log_loss

    if not 2 <= len(components) <= 4:
        raise ValueError("Spooky production blend requires two to four distinct channels")
    best: tuple[float, list[float], float, str] | None = None
    for weights in _spooky_simplex_weights(len(components)):
        for mode in ("arithmetic", "log_probability"):
            for temperature in (0.7, 0.8, 0.9, 1.0, 1.1, 1.25):
                probability = apply_spooky_multicomponent_blend(
                    components,
                    weights=weights,
                    temperature=temperature,
                    mode=mode,
                )
                score = float(log_loss(labels, probability, labels=classes))
                candidate = (score, weights, float(temperature), mode)
                if best is None or (
                    candidate[0], candidate[1], candidate[2], candidate[3]
                ) < (best[0], best[1], best[2], best[3]):
                    best = candidate
    assert best is not None
    return best


def cross_fit_spooky_multicomponent_blend(
    components: list[np.ndarray],
    labels: np.ndarray,
    classes: list[str],
    folds: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    prediction = np.zeros_like(components[0], dtype=np.float64)
    records: list[dict[str, Any]] = []
    for fold in sorted(int(value) for value in np.unique(folds)):
        validation = folds == fold
        fitting = ~validation
        _, weights, temperature, mode = select_spooky_multicomponent_blend(
            [component[fitting] for component in components],
            labels[fitting],
            classes,
        )
        prediction[validation] = apply_spooky_multicomponent_blend(
            [component[validation] for component in components],
            weights=weights,
            temperature=temperature,
            mode=mode,
        )
        records.append({
            "fold": fold,
            "weights": weights,
            "temperature": temperature,
            "mode": mode,
        })
    if not np.isfinite(prediction).all():
        raise RuntimeError("Spooky cross-fitted multicomponent blend is incomplete")
    return prediction, records


def align_multiclass_submission(
    sample: pd.DataFrame,
    test_ids: pd.Series,
    probability: np.ndarray,
    classes: list[str],
    *,
    id_column: str = "id",
) -> pd.DataFrame:
    if sample[id_column].duplicated().any() or test_ids.duplicated().any():
        raise RuntimeError("Submission or prediction IDs are duplicated")
    if set(sample[id_column].astype(str)) != set(test_ids.astype(str)):
        raise RuntimeError("Submission and prediction ID sets differ")
    if probability.shape != (len(test_ids), len(classes)):
        raise RuntimeError("Multiclass prediction shape is invalid")
    keyed = pd.DataFrame(probability, columns=classes)
    keyed[id_column] = test_ids.astype(str).to_numpy()
    aligned = sample[[id_column]].copy()
    original_ids = aligned[id_column].copy()
    aligned[id_column] = aligned[id_column].astype(str)
    aligned = aligned.merge(keyed, on=id_column, how="left", validate="one_to_one")
    aligned[id_column] = original_ids.to_numpy()
    if aligned[classes].isna().any().any():
        raise RuntimeError("Submission ID join produced missing probabilities")
    return aligned[[id_column, *classes]]


def run_spooky_nbsvm_oof(args: Any, task_dir: Path, logger: Any) -> dict[str, Any]:
    """Duplicate-safe four-channel NB-SVM with a cross-fitted meta blend."""

    from joblib import dump
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import log_loss
    from sklearn.model_selection import StratifiedGroupKFold

    competition_id = "spooky-author-identification"
    resolved = resolve_competition(competition_id, args.data_root)
    train = pd.read_csv(resolved.public_dir / "train.csv").reset_index(drop=True)
    test = pd.read_csv(resolved.public_dir / "test.csv").reset_index(drop=True)
    sample = pd.read_csv(resolved.sample_submission_path)
    required_train = {"id", "text", "author"}
    required_test = {"id", "text"}
    if not required_train <= set(train) or not required_test <= set(test):
        raise RuntimeError("Spooky Author input schema is incomplete")
    if train["id"].duplicated().any() or test["id"].duplicated().any() or sample["id"].duplicated().any():
        raise RuntimeError("Spooky Author train, test, or submission IDs are duplicated")
    if set(test["id"].astype(str)) != set(sample["id"].astype(str)):
        raise RuntimeError("Spooky Author test and submission ID sets differ")
    classes = [column for column in sample.columns if column != "id"]
    if set(classes) != set(train["author"].astype(str).unique()):
        raise RuntimeError("Spooky Author class columns differ from training labels")
    labels = train["author"].astype(str).to_numpy()
    train_text = train["text"].fillna("").astype(str)
    test_text = test["text"].fillna("").astype(str)
    duplicate_groups, duplicate_report = build_spooky_duplicate_groups(train_text, labels)
    group_labels = pd.DataFrame({"group": duplicate_groups, "label": labels}).drop_duplicates()
    minimum_class_groups = int(group_labels["label"].value_counts().min())
    effective_folds = min(int(args.spooky_folds), minimum_class_groups)
    if effective_folds < 2:
        raise RuntimeError("Spooky Author requires at least two duplicate groups in every class")
    splitter = StratifiedGroupKFold(
        n_splits=effective_folds,
        shuffle=True,
        random_state=args.seed,
    )
    train_style = build_spooky_stylometric_features(train_text)
    test_style = build_spooky_stylometric_features(test_text)
    word_oof = np.zeros((len(train), len(classes)), dtype=np.float64)
    char_oof = np.zeros_like(word_oof)
    raw_char_oof = np.zeros_like(word_oof)
    style_oof = np.zeros_like(word_oof)
    word_test = np.zeros((len(test), len(classes)), dtype=np.float64)
    char_test = np.zeros_like(word_test)
    raw_char_test = np.zeros_like(word_test)
    style_test = np.zeros_like(word_test)
    fold_assignment = np.full(len(train), -1, dtype=np.int16)
    fold_records: list[dict[str, Any]] = []
    model_dir = task_dir / "spooky_fold_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for fold, (fit_indices, valid_indices) in enumerate(
        splitter.split(train_text, labels, duplicate_groups)
    ):
        if set(duplicate_groups[fit_indices]) & set(duplicate_groups[valid_indices]):
            raise RuntimeError("Spooky duplicate group crossed a fold boundary")
        fold_assignment[valid_indices] = fold
        word = TfidfVectorizer(
            strip_accents="unicode",
            sublinear_tf=True,
            ngram_range=(1, 3),
            min_df=2,
            max_features=args.spooky_word_features,
        )
        char = TfidfVectorizer(
            analyzer="char_wb",
            sublinear_tf=True,
            ngram_range=(3, 6),
            min_df=2,
            max_features=args.spooky_char_features,
        )
        raw_char = TfidfVectorizer(
            analyzer="char",
            sublinear_tf=True,
            ngram_range=(2, 5),
            min_df=2,
            max_features=args.spooky_raw_char_features,
        )
        fit_word = word.fit_transform(train_text.iloc[fit_indices])
        valid_word = word.transform(train_text.iloc[valid_indices])
        test_word = word.transform(test_text)
        fit_char = char.fit_transform(train_text.iloc[fit_indices])
        valid_char = char.transform(train_text.iloc[valid_indices])
        test_char = char.transform(test_text)
        fit_raw_char = raw_char.fit_transform(train_text.iloc[fit_indices])
        valid_raw_char = raw_char.transform(train_text.iloc[valid_indices])
        test_raw_char = raw_char.transform(test_text)
        valid_word_probability, test_word_probability, word_models = fit_spooky_nbsvm_channel(
            fit_word,
            labels[fit_indices],
            valid_word,
            test_word,
            classes,
            c_value=args.spooky_nbsvm_c,
            seed=args.seed + fold * 10,
        )
        valid_char_probability, test_char_probability, char_models = fit_spooky_nbsvm_channel(
            fit_char,
            labels[fit_indices],
            valid_char,
            test_char,
            classes,
            c_value=args.spooky_nbsvm_c,
            seed=args.seed + fold * 10 + 1000,
        )
        valid_raw_char_probability, test_raw_char_probability, raw_char_models = (
            fit_spooky_nbsvm_channel(
                fit_raw_char,
                labels[fit_indices],
                valid_raw_char,
                test_raw_char,
                classes,
                c_value=args.spooky_nbsvm_c,
                seed=args.seed + fold * 10 + 2000,
            )
        )
        valid_style_probability, test_style_probability, style_model = (
            fit_spooky_stylometric_channel(
                train_style.iloc[fit_indices],
                labels[fit_indices],
                train_style.iloc[valid_indices],
                test_style,
                classes,
                c_value=args.spooky_style_c,
                seed=args.seed + fold * 10 + 3000,
            )
        )
        word_oof[valid_indices] = valid_word_probability
        char_oof[valid_indices] = valid_char_probability
        raw_char_oof[valid_indices] = valid_raw_char_probability
        style_oof[valid_indices] = valid_style_probability
        word_test += test_word_probability / effective_folds
        char_test += test_char_probability / effective_folds
        raw_char_test += test_raw_char_probability / effective_folds
        style_test += test_style_probability / effective_folds
        fold_score = float(log_loss(
            labels[valid_indices],
            (
                valid_word_probability
                + valid_char_probability
                + valid_raw_char_probability
                + valid_style_probability
            ) / 4.0,
            labels=classes,
        ))
        fold_records.append({
            "fold": fold,
            "train_rows": len(fit_indices),
            "valid_rows": len(valid_indices),
            "word_features": int(fit_word.shape[1]),
            "char_features": int(fit_char.shape[1]),
            "raw_char_features": int(fit_raw_char.shape[1]),
            "style_features": int(train_style.shape[1]),
            "fixed_equal_blend_log_loss": fold_score,
        })
        dump(
            {
                "word_vectorizer": word,
                "char_vectorizer": char,
                "raw_char_vectorizer": raw_char,
                "word_models": word_models,
                "char_models": char_models,
                "raw_char_models": raw_char_models,
                "style_model": style_model,
                "classes": classes,
            },
            model_dir / f"fold_{fold}.joblib",
            compress=3,
        )
        logger.info("[%s] fold=%d/%d log_loss=%.6f", competition_id, fold + 1, effective_folds, fold_score)
    if np.any(fold_assignment < 0):
        raise RuntimeError("Spooky Author OOF coverage is incomplete")
    wave0.write_json(task_dir / "spooky_duplicate_groups.json", duplicate_report)
    components_oof = [word_oof, char_oof, raw_char_oof, style_oof]
    crossfit_oof, crossfit_records = cross_fit_spooky_multicomponent_blend(
        components_oof, labels, classes, fold_assignment
    )
    cv_score = float(log_loss(labels, crossfit_oof, labels=classes))
    _, final_weights, final_temperature, final_mode = select_spooky_multicomponent_blend(
        components_oof, labels, classes
    )
    test_probability = apply_spooky_multicomponent_blend(
        [word_test, char_test, raw_char_test, style_test],
        weights=final_weights,
        temperature=final_temperature,
        mode=final_mode,
    )
    if not np.isfinite(crossfit_oof).all() or not np.isfinite(test_probability).all():
        raise RuntimeError("Spooky Author blended probabilities are non-finite")
    if not np.allclose(crossfit_oof.sum(axis=1), 1.0, atol=1e-8):
        raise RuntimeError("Spooky Author OOF probability rows do not sum to one")
    if not np.allclose(test_probability.sum(axis=1), 1.0, atol=1e-8):
        raise RuntimeError("Spooky Author test probability rows do not sum to one")
    submission = align_multiclass_submission(sample, test["id"], test_probability, classes)
    np.savez_compressed(
        task_dir / "spooky_oof_and_test.npz",
        train_id=train["id"].astype(str).to_numpy(),
        test_id=test["id"].astype(str).to_numpy(),
        labels=labels,
        classes=np.asarray(classes),
        fold=fold_assignment,
        duplicate_group=duplicate_groups,
        word_oof=word_oof,
        char_oof=char_oof,
        raw_char_oof=raw_char_oof,
        style_oof=style_oof,
        crossfit_blended_oof=crossfit_oof,
        word_test=word_test,
        char_test=char_test,
        raw_char_test=raw_char_test,
        style_test=style_test,
        final_test_probability=test_probability,
    )
    budget = {
        "seed": args.seed,
        "folds": effective_folds,
        "train_rows": len(train),
        "test_rows": len(test),
        "word_max_features": args.spooky_word_features,
        "char_max_features": args.spooky_char_features,
        "raw_char_max_features": args.spooky_raw_char_features,
        "nbsvm_c": args.spooky_nbsvm_c,
        "style_c": args.spooky_style_c,
        "final_component_weights": {
            "word": final_weights[0],
            "character_word_boundary": final_weights[1],
            "character_raw": final_weights[2],
            "style": final_weights[3],
        },
        "final_temperature": final_temperature,
        "final_mode": final_mode,
        "duplicate_groups": duplicate_report,
        "cross_fitted_meta_cv": True,
        "fold_models_saved": True,
    }
    promotion_gate = wave0.build_metric_promotion_gate(
        name="spooky_cross_fitted_multichannel_log_loss",
        metric="log_loss",
        direction="minimize",
        score=float(cv_score),
        threshold=0.28,
        evidence={
            "cross_fitted_meta_cv": True,
            "fold_count": effective_folds,
            "probability_rows_sum_to_one": bool(
                np.allclose(crossfit_oof.sum(axis=1), 1.0, atol=1e-8)
            ),
        },
    )
    return wave0.finalize_scored_task(
        competition_id=competition_id,
        submission=submission,
        cv_score=cv_score,
        args=args,
        task_dir=task_dir,
        budget=budget,
        promotion_gate=promotion_gate,
        extra={
            "runtime_seconds_model": time.perf_counter() - started,
            "model_family": "duplicate_safe_word_charwb_rawchar_NBSVM_plus_stylometry_crossfit_blend",
            "folds": fold_records,
            "crossfit_blend": crossfit_records,
            "budget": budget,
        },
    )


RUNNERS = {
    "aerial-cactus-identification": run_aerial_cactus_convnext,
    "denoising-dirty-documents": run_denoising_unet,
    "dogs-vs-cats-redux-kernels-edition": run_dogs_cats_convnext,
    # Keep the standalone recovery registry aligned with the authoritative
    # production entrypoint.  The multimodal runner owns the verified image
    # embedding, numeric channel, fold-local calibration, and deployment
    # contract; the older numeric-only helper remains available as an explicit
    # diagnostic function but must not be selected implicitly.
    "leaf-classification": run_leaf_multimodal,
    "siim-isic-melanoma-classification": run_siim_image_metadata,
    "spooky-author-identification": run_spooky_nbsvm_oof,
    "tabular-playground-series-may-2022": run_may2022_gpu_ensemble,
}
