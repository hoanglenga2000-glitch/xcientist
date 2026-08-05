#!/usr/bin/env python3
"""Deterministic Wave 2 adapters for the remaining MLE-Bench Lite tasks.

The module is imported by ``run_mlebench_lite_full.py``.  It never submits to
Kaggle and writes only below the caller-provided attempt directory.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import re
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from research_os.mlebench_phase_a import compute_metric, resolve_competition

try:
    import run_mlebench_lite_wave0 as wave0
    import russian_transliteration as ru_transliteration
except ModuleNotFoundError:  # imported as ``scripts.mlebench_wave2_adapters``
    from scripts import run_mlebench_lite_wave0 as wave0
    from scripts import russian_transliteration as ru_transliteration


WAVE2_COMPETITIONS = (
    "aptos2019-blindness-detection",
    "dog-breed-identification",
    "histopathologic-cancer-detection",
    "jigsaw-toxic-comment-classification-challenge",
    "mlsp-2013-birds",
    "nomad2018-predict-transparent-conductors",
    "plant-pathology-2020-fgvc7",
    "ranzcr-clip-catheter-line-classification",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
    "the-icml-2013-whale-challenge-right-whale-redux",
)
WAVE2_PROMOTION_CONTRACTS: dict[str, dict[str, Any]] = {
    "aptos2019-blindness-detection": {
        "name": "aptos_cross_fitted_qwk",
        "metric": "quadratic_weighted_kappa",
        "direction": "maximize",
        "threshold": 0.920,
        "fold_threshold": 0.900,
        "require_all_folds": True,
    },
    "dog-breed-identification": {
        "name": "dog_breed_cross_fitted_log_loss",
        "metric": "multiclass_log_loss",
        "direction": "minimize",
        "threshold": 0.040,
        "minimum_teacher_blend_improvement": 0.0005,
    },
    "histopathologic-cancer-detection": {
        "name": "histopath_duplicate_aware_auc",
        "metric": "roc_auc",
        "direction": "maximize",
        "threshold": 0.9775,
        "fold_threshold": 0.974,
        "require_all_folds": True,
    },
    "jigsaw-toxic-comment-classification-challenge": {
        "name": "jigsaw_exact_six_column_mean_auc",
        "metric": "mean_columnwise_roc_auc",
        "direction": "maximize",
        "threshold": 0.9870,
        "minimum_blend_improvement": 0.0003,
    },
    "mlsp-2013-birds": {
        "name": "birds_exact_pooled_pair_auc",
        "metric": "roc_auc",
        "direction": "maximize",
        "threshold": 0.890,
    },
    "plant-pathology-2020-fgvc7": {
        "name": "plant_exact_four_class_mean_auc",
        "metric": "mean_columnwise_roc_auc",
        "direction": "maximize",
        "threshold": 0.978,
        "minimum_class_score": 0.950,
    },
    "ranzcr-clip-catheter-line-classification": {
        "name": "ranzcr_grouped_mean_auc",
        "metric": "mean_columnwise_roc_auc",
        "direction": "maximize",
        "threshold": 0.9725,
    },
    "the-icml-2013-whale-challenge-right-whale-redux": {
        "name": "right_whale_duplicate_aware_auc",
        "metric": "roc_auc",
        "direction": "maximize",
        "threshold": 0.930,
    },
}
JIGSAW_TARGET_COLUMNS = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)
JIGSAW_STACKER_FEATURE_SET = "hybrid30"
JIGSAW_STACKER_C_GRID = (0.03, 0.1, 0.3, 1.0)
JIGSAW_STACKER_CLASS_WEIGHT = "balanced"
CONVNEXT_TINY_WEIGHT_FILENAME = "convnext_tiny-983f1562.pth"
CONVNEXT_TINY_WEIGHT_SHA256 = (
    "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d"
)
CONVNEXT_SMALL_WEIGHT_FILENAME = "convnext_small-0c510722.pth"
CONVNEXT_SMALL_WEIGHT_SHA256 = (
    "0c510722adfd92966a2bd72b92f785ca05966bbac03cafe2f7a90b1f54bfab9a"
)
EFFICIENTNET_V2_S_WEIGHT_FILENAME = "efficientnet_v2_s-dd5fe13b.pth"
EFFICIENTNET_V2_S_WEIGHT_SHA256 = (
    "dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74"
)
VISION_BACKBONE_SPECS: dict[str, dict[str, str]] = {
    "convnext_tiny": {
        "filename": CONVNEXT_TINY_WEIGHT_FILENAME,
        "sha256": CONVNEXT_TINY_WEIGHT_SHA256,
        "classifier": "classifier.2",
    },
    "convnext_small": {
        "filename": CONVNEXT_SMALL_WEIGHT_FILENAME,
        "sha256": CONVNEXT_SMALL_WEIGHT_SHA256,
        "classifier": "classifier.2",
    },
    "efficientnet_v2_s": {
        "filename": EFFICIENTNET_V2_S_WEIGHT_FILENAME,
        "sha256": EFFICIENTNET_V2_S_WEIGHT_SHA256,
        "classifier": "classifier.1",
    },
}
VISION_DATALOADER_PREFETCH_FACTOR = 4
DOG_BREED_TEACHER_CACHE_SCHEMA = "evomind.dog_breed.imagenet_teacher.v1"
DOG_BREED_BLEND_FAMILIES = (
    "arithmetic_probability",
    "geometric_log_probability",
)
DOG_BREED_BLEND_COARSE_WEIGHT_COUNT = 21
DOG_BREED_BLEND_COARSE_TEMPERATURE_COUNT = 17
DOG_BREED_BLEND_REFINEMENT_COUNT = 9
DOG_BREED_CLASS_BIAS_L2_GRID = (1.0, 10.0, 100.0)
DOG_BREED_STAB_WARMUP_FRACTION = 0.10
DOG_BREED_STAB_MIN_LR_FACTOR = 0.05
DOG_BREED_STAB_WEIGHT_DECAY = 2e-4
DOG_BREED_STAB_EPOCH_ONE_MAX_LOG_LOSS = 0.090
DOG_BREED_STAB_MIN_RELATIVE_EPOCH_ONE_IMPROVEMENT = 0.25
DOG_BREED_STAB_MIN_EPOCH_IMPROVEMENT = 0.002
DOG_BREED_STAB_CONTINUATION_MAX_LOG_LOSS = 0.055
DOG_BREED_STAB_MIN_TOP1_IMPROVEMENT = 0.005
DOG_BREED_STAB_MAX_EPOCHS = 4
DOG_BREED_STAB_MAX_WALL_SECONDS = 90 * 60
DOG_BREED_TRAINING_MODES = (
    "stability_finetune",
    "frozen_backbone_head",
)
DOG_BREED_FROZEN_HEAD_WEIGHT_DECAY = 1e-4


class _DogBreedTeacherImages:
    """Spawn-picklable image dataset for the frozen ImageNet teacher."""

    def __init__(self, paths: Sequence[Path], transform: Any) -> None:
        self.paths = [Path(path) for path in paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        from PIL import Image

        with Image.open(self.paths[index]) as handle:
            return self.transform(handle.convert("RGB"))


class _VisionImages:
    """Spawn-picklable dataset shared by Wave 2 vision training loaders."""

    def __init__(
        self,
        paths: Sequence[Path],
        targets: np.ndarray | None,
        transform: Any,
        target_mode: str,
    ) -> None:
        self.paths = [Path(path) for path in paths]
        self.targets = targets
        self.transform = transform
        self.target_mode = target_mode

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        import torch
        from PIL import Image

        with Image.open(self.paths[index]) as handle:
            image = self.transform(handle.convert("RGB"))
        if self.targets is None:
            return image
        value = self.targets[index]
        if self.target_mode in {"ordinal", "multiclass"}:
            return image, torch.tensor(int(value), dtype=torch.long)
        return image, torch.tensor(value, dtype=torch.float32)
AUDIO_FEATURE_CACHE_VERSION = "audio_1572_v2"


def _finite_probability(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise RuntimeError("Model produced non-finite probabilities")
    return np.clip(result, 1e-6, 1.0 - 1e-6)


def _binary_log_loss_array(target: np.ndarray, probability: np.ndarray) -> float:
    truth = np.asarray(target, dtype=np.float64).reshape(-1)
    predicted = _finite_probability(probability).reshape(-1)
    if truth.shape != predicted.shape or set(np.unique(truth).tolist()) != {0.0, 1.0}:
        raise RuntimeError("Binary calibration target/probability contract is invalid")
    return float(-np.mean(truth * np.log(predicted) + (1.0 - truth) * np.log1p(-predicted)))


def _sigmoid_array(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    if not np.isfinite(logits).all():
        raise RuntimeError("Binary calibration logits contain non-finite values")
    clipped = np.clip(logits, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_binary_temperature_intercept(
    logits: np.ndarray,
    target: np.ndarray,
    *,
    max_iterations: int = 80,
) -> dict[str, float]:
    """Fit a deterministic two-parameter Platt/temperature calibrator.

    Newton steps are accepted only when they improve training log loss.  The
    positive scale constraint preserves class orientation; symmetric clipping
    is selected from a fixed grid using only the calibrator-fit rows.
    """

    x = np.asarray(logits, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or len(x) < 4 or not np.isfinite(x).all():
        raise RuntimeError("Binary calibration inputs are incomplete or non-finite")
    if set(np.unique(y).tolist()) != {0.0, 1.0}:
        raise RuntimeError("Binary calibration requires both target classes")

    scale = 1.0
    intercept = 0.0
    ridge = 1e-6

    def objective(candidate_scale: float, candidate_intercept: float) -> float:
        probability = _sigmoid_array(candidate_scale * x + candidate_intercept)
        penalty = ridge * ((candidate_scale - 1.0) ** 2 + candidate_intercept**2)
        return _binary_log_loss_array(y, probability) + penalty

    best = objective(scale, intercept)
    for _ in range(max_iterations):
        probability = _sigmoid_array(scale * x + intercept)
        residual = probability - y
        weight = np.maximum(probability * (1.0 - probability), 1e-8)
        gradient = np.asarray([
            float(np.dot(x, residual)) + 2.0 * ridge * (scale - 1.0),
            float(residual.sum()) + 2.0 * ridge * intercept,
        ])
        hessian = np.asarray([
            [float(np.dot(x * x, weight)) + 2.0 * ridge, float(np.dot(x, weight))],
            [float(np.dot(x, weight)), float(weight.sum()) + 2.0 * ridge],
        ])
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        accepted = False
        multiplier = 1.0
        for _ in range(24):
            candidate_scale = float(np.clip(scale - multiplier * step[0], 0.05, 20.0))
            candidate_intercept = float(np.clip(intercept - multiplier * step[1], -10.0, 10.0))
            score = objective(candidate_scale, candidate_intercept)
            if score <= best - 1e-12:
                scale, intercept, best = candidate_scale, candidate_intercept, score
                accepted = True
                break
            multiplier *= 0.5
        if not accepted or float(np.linalg.norm(multiplier * step)) < 1e-7:
            break

    calibrated = _sigmoid_array(scale * x + intercept)
    clip_grid = (0.0, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2)
    clip_floor = min(
        clip_grid,
        key=lambda value: _binary_log_loss_array(
            y,
            np.clip(calibrated, value, 1.0 - value) if value else calibrated,
        ),
    )
    return {
        "scale": float(scale),
        "temperature": float(1.0 / scale),
        "intercept": float(intercept),
        "clip_floor": float(clip_floor),
        "fit_log_loss": _binary_log_loss_array(
            y,
            np.clip(calibrated, clip_floor, 1.0 - clip_floor) if clip_floor else calibrated,
        ),
    }


def apply_binary_temperature_intercept(
    logits: np.ndarray,
    calibration: dict[str, float],
) -> np.ndarray:
    probability = _sigmoid_array(
        float(calibration["scale"]) * np.asarray(logits, dtype=np.float64)
        + float(calibration["intercept"])
    )
    clip_floor = float(calibration.get("clip_floor", 0.0))
    if clip_floor:
        probability = np.clip(probability, clip_floor, 1.0 - clip_floor)
    return _finite_probability(probability)


def cross_fit_binary_logloss_calibration(
    oof_logits: np.ndarray,
    test_logits: np.ndarray,
    target: np.ndarray,
    fold_assignment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Calibrate each held-out fold and preserve its test-model path.

    A two-dimensional ``test_logits`` array contains one test prediction column
    per OOF fold model.  Each column is calibrated with the matching
    cross-fitted calibrator before probabilities are averaged.  One-dimensional
    input remains supported for nested candidate selection, where only one
    external prediction vector exists.
    """

    logits = np.asarray(oof_logits, dtype=np.float64).reshape(-1)
    test = np.asarray(test_logits, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64).reshape(-1)
    folds = np.asarray(fold_assignment, dtype=np.int64).reshape(-1)
    if not (logits.shape == truth.shape == folds.shape):
        raise RuntimeError("Cross-fitted calibration arrays have inconsistent shapes")
    if not np.isfinite(logits).all() or not np.isfinite(test).all() or np.any(folds < 0):
        raise RuntimeError("Cross-fitted calibration arrays are incomplete or non-finite")
    unique_folds = np.unique(folds)
    if len(unique_folds) < 2 or set(np.unique(truth).tolist()) != {0.0, 1.0}:
        raise RuntimeError("Cross-fitted calibration requires multiple folds and both classes")
    if test.ndim == 1:
        test_components = test.reshape(-1, 1)
        per_fold_test_components = False
    elif test.ndim == 2 and test.shape[1] == len(unique_folds):
        test_components = test
        per_fold_test_components = True
    else:
        raise RuntimeError(
            "Test logits must be one vector or one component per OOF fold"
        )

    calibrated_oof = np.full(len(truth), np.nan, dtype=np.float64)
    calibrated_test_components: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for component_index, fold in enumerate(unique_folds):
        validation = folds == fold
        fitting = ~validation
        if set(np.unique(truth[fitting]).tolist()) != {0.0, 1.0}:
            raise RuntimeError("A calibration training split is missing a target class")
        fitted = fit_binary_temperature_intercept(logits[fitting], truth[fitting])
        calibrated_oof[validation] = apply_binary_temperature_intercept(logits[validation], fitted)
        if per_fold_test_components:
            calibrated_test_components.append(
                apply_binary_temperature_intercept(
                    test_components[:, component_index], fitted
                )
            )
        records.append({
            "fold": float(fold),
            "test_component_index": component_index if per_fold_test_components else None,
            **fitted,
            "validation_log_loss": _binary_log_loss_array(
                truth[validation], calibrated_oof[validation]
            ),
        })
    if not np.isfinite(calibrated_oof).all():
        raise RuntimeError("Cross-fitted calibration did not cover every OOF row")
    final = fit_binary_temperature_intercept(logits, truth)
    if per_fold_test_components:
        calibrated_test = np.mean(
            np.column_stack(calibrated_test_components), axis=1
        )
        test_aggregation = "per_fold_calibrate_then_probability_average"
    else:
        calibrated_test = apply_binary_temperature_intercept(
            test_components[:, 0], final
        )
        test_aggregation = "global_oof_calibration_single_external_vector"
    final = {
        **final,
        "cross_fitted_log_loss": _binary_log_loss_array(truth, calibrated_oof),
        "test_component_count": int(test_components.shape[1]),
        "test_aggregation": test_aggregation,
    }
    return calibrated_oof, calibrated_test, records, final


def optimize_ordinal_thresholds(expected: np.ndarray, truth: np.ndarray) -> list[float]:
    """Coordinate-search four monotonic thresholds for five ordinal classes."""

    expected = np.asarray(expected, dtype=float)
    truth = np.asarray(truth, dtype=int)
    thresholds = np.asarray([0.5, 1.5, 2.5, 3.5], dtype=float)

    def score(candidate: np.ndarray) -> float:
        ordered = np.maximum.accumulate(candidate)
        predicted = np.digitize(expected, ordered)
        return compute_metric("quadratic_weighted_kappa", truth, predicted)

    best = score(thresholds)
    for step in (0.5, 0.2, 0.08, 0.03):
        improved = True
        while improved:
            improved = False
            for index in range(4):
                for delta in (-step, step):
                    candidate = thresholds.copy()
                    candidate[index] += delta
                    if index and candidate[index] <= candidate[index - 1] + 0.02:
                        continue
                    if index < 3 and candidate[index] >= candidate[index + 1] - 0.02:
                        continue
                    candidate_score = score(candidate)
                    if candidate_score > best + 1e-9:
                        thresholds, best, improved = candidate, candidate_score, True
    return thresholds.tolist()


def cross_fit_ordinal_thresholds(
    expected: np.ndarray,
    truth: np.ndarray,
    fold_assignment: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Fit ordinal thresholds away from each held-out fold and score every row once."""

    expected = np.asarray(expected, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth, dtype=np.int64).reshape(-1)
    folds = np.asarray(fold_assignment, dtype=np.int16).reshape(-1)
    if expected.shape != truth.shape or truth.shape != folds.shape:
        raise ValueError("Ordinal cross-fit inputs must have matching one-dimensional shapes")
    if not np.isfinite(expected).all() or np.any(folds < 0):
        raise ValueError("Ordinal cross-fit inputs must be finite and fully assigned")
    prediction = np.full(len(truth), -1, dtype=np.int64)
    records: list[dict[str, Any]] = []
    for fold in sorted(np.unique(folds).tolist()):
        fitting = folds != fold
        validation = folds == fold
        if not np.any(fitting) or not np.any(validation):
            raise ValueError("Ordinal cross-fit produced an empty fitting or validation partition")
        thresholds = optimize_ordinal_thresholds(expected[fitting], truth[fitting])
        fold_prediction = np.digitize(expected[validation], thresholds)
        prediction[validation] = fold_prediction
        records.append({
            "fold": int(fold),
            "fit_rows": int(fitting.sum()),
            "validation_rows": int(validation.sum()),
            "thresholds": thresholds,
            "fold_score": float(
                compute_metric(
                    "quadratic_weighted_kappa",
                    truth[validation],
                    fold_prediction,
                )
            ),
        })
    if np.any(prediction < 0):
        raise RuntimeError("Ordinal cross-fit prediction coverage is incomplete")
    return prediction, records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_classifier_from_imagenet_rows(
    source_classifier: Any,
    output_count: int,
    source_indices: Sequence[int],
) -> tuple[Any, dict[str, Any]]:
    """Build a task head from exact rows of a pretrained ImageNet classifier."""
    import torch

    if not isinstance(source_classifier, torch.nn.Linear):
        raise TypeError("ImageNet row initialization requires a linear classifier")
    indices = np.asarray(list(source_indices), dtype=np.int64)
    if indices.ndim != 1 or len(indices) != int(output_count):
        raise ValueError("ImageNet classifier row count must match the task output count")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("ImageNet classifier source rows must be unique")
    if len(indices) and (
        int(indices.min()) < 0 or int(indices.max()) >= source_classifier.out_features
    ):
        raise ValueError("ImageNet classifier source row is outside the pretrained head")

    replacement = torch.nn.Linear(
        source_classifier.in_features,
        int(output_count),
        bias=source_classifier.bias is not None,
        device=source_classifier.weight.device,
        dtype=source_classifier.weight.dtype,
    )
    tensor_indices = torch.as_tensor(
        indices,
        dtype=torch.long,
        device=source_classifier.weight.device,
    )
    with torch.no_grad():
        replacement.weight.copy_(source_classifier.weight.index_select(0, tensor_indices))
        if source_classifier.bias is not None:
            assert replacement.bias is not None
            replacement.bias.copy_(source_classifier.bias.index_select(0, tensor_indices))
    return replacement, {
        "mode": "exact_pretrained_imagenet_classifier_rows",
        "output_count": int(output_count),
        "source_output_count": int(source_classifier.out_features),
        "source_indices": indices.tolist(),
        "unique_source_rows": True,
        "weights_copied": True,
        "bias_copied": source_classifier.bias is not None,
    }


def _vision_model(
    output_count: int,
    *,
    backbone: str = "convnext_tiny",
    require_pretrained: bool = True,
    preserve_imagenet_classifier: bool = False,
    classifier_source_indices: Sequence[int] | None = None,
):
    import torch
    from torchvision.models import (
        ConvNeXt_Small_Weights,
        ConvNeXt_Tiny_Weights,
        EfficientNet_V2_S_Weights,
        convnext_small,
        convnext_tiny,
        efficientnet_v2_s,
    )

    registry = {
        "convnext_tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT),
        "convnext_small": (convnext_small, ConvNeXt_Small_Weights.DEFAULT),
        "efficientnet_v2_s": (efficientnet_v2_s, EfficientNet_V2_S_Weights.DEFAULT),
    }
    if backbone not in registry or backbone not in VISION_BACKBONE_SPECS:
        raise ValueError(f"Unsupported vision backbone: {backbone}")

    model_factory, weights = registry[backbone]
    spec = VISION_BACKBONE_SPECS[backbone]
    weight_path = Path(torch.hub.get_dir()) / "checkpoints" / spec["filename"]
    actual_sha256 = _sha256_file(weight_path) if weight_path.is_file() else None
    weight_identity = {
        "backbone": backbone,
        "enum": str(weights),
        "filename": spec["filename"],
        "path": str(weight_path),
        "sha256": actual_sha256,
        "expected_sha256": spec["sha256"],
    }

    if not weight_path.is_file():
        if require_pretrained:
            raise RuntimeError(
                f"Wave 2 medal mode requires the pinned {backbone} ImageNet weights in TORCH_HOME"
            )
        model = model_factory(weights=None)
        pretrained = False
    else:
        pretrained = True
    if weight_path.is_file() and actual_sha256 != spec["sha256"]:
        raise RuntimeError(f"Cached {backbone} weights do not match the pinned SHA256")

    if pretrained:
        try:
            model = model_factory(weights=weights)
        except Exception as exc:
            if require_pretrained:
                raise RuntimeError(
                    f"Wave 2 medal mode could not load the pinned {backbone} ImageNet weights"
                ) from exc
            model = model_factory(weights=None)
            pretrained = False
    if require_pretrained and not weight_identity["sha256"]:
        raise RuntimeError(f"Loaded {backbone} pretrained weights are missing a cache-file hash")
    classifier_index = int(spec["classifier"].rsplit(".", maxsplit=1)[-1])
    if preserve_imagenet_classifier and classifier_source_indices is not None:
        raise ValueError(
            "Preserving the full ImageNet classifier and selecting source rows are exclusive"
        )
    if preserve_imagenet_classifier:
        category_count = len(weights.meta.get("categories", ()))
        if output_count != category_count:
            raise RuntimeError(
                "Preserved ImageNet classifier output count does not match its category metadata"
            )
    elif classifier_source_indices is not None:
        if not pretrained:
            raise RuntimeError("ImageNet classifier row initialization requires pretrained weights")
        replacement, initialization = initialize_classifier_from_imagenet_rows(
            model.classifier[classifier_index],
            output_count,
            classifier_source_indices,
        )
        model.classifier[classifier_index] = replacement
        weight_identity["classifier_initialization"] = initialization
    else:
        input_features = int(model.classifier[classifier_index].in_features)
        model.classifier[classifier_index] = torch.nn.Linear(input_features, output_count)
    return model, pretrained, weight_identity


def _resolve_named_module(model: Any, dotted_path: str) -> Any:
    current = model
    for component in dotted_path.split("."):
        current = current[int(component)] if component.isdigit() else getattr(current, component)
    return current


def build_dog_breed_stability_optimizer_groups(
    model: Any,
    *,
    base_learning_rate: float,
    classifier_path: str = "classifier.2",
    weight_decay: float = DOG_BREED_STAB_WEIGHT_DECAY,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build disjoint DB-STAB parameter groups without disturbing classifier LayerNorm."""

    learning_rate = float(base_learning_rate)
    decay = float(weight_decay)
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise RuntimeError("Dog Breed base learning rate must be positive")
    if not np.isfinite(decay) or decay < 0.0:
        raise RuntimeError("Dog Breed weight decay must be non-negative")

    head = _resolve_named_module(model, classifier_path)
    head_parameter_ids = {
        id(parameter) for parameter in head.parameters(recurse=False) if parameter.requires_grad
    }
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not named_parameters or not head_parameter_ids:
        raise RuntimeError("Dog Breed DB-STAB trainable parameters are incomplete")

    buckets: dict[str, list[tuple[str, Any]]] = {
        "pretrained_decay": [],
        "pretrained_no_decay": [],
        "task_head_decay": [],
        "task_head_no_decay": [],
    }
    for name, parameter in named_parameters:
        is_head = id(parameter) in head_parameter_ids
        no_decay = name.endswith(".bias") or int(parameter.ndim) <= 1
        family = "task_head" if is_head else "pretrained"
        buckets[f"{family}_{'no_decay' if no_decay else 'decay'}"].append(
            (name, parameter)
        )

    ordered_groups: list[dict[str, Any]] = []
    group_contracts: list[dict[str, Any]] = []
    assigned_ids: list[int] = []
    for group_name in (
        "pretrained_decay",
        "pretrained_no_decay",
        "task_head_decay",
        "task_head_no_decay",
    ):
        entries = buckets[group_name]
        if not entries:
            continue
        is_head = group_name.startswith("task_head")
        no_decay = group_name.endswith("no_decay")
        group_learning_rate = learning_rate if is_head else learning_rate * 0.1
        group_weight_decay = 0.0 if no_decay else decay
        parameters = [parameter for _, parameter in entries]
        ordered_groups.append({
            "params": parameters,
            "lr": group_learning_rate,
            "weight_decay": group_weight_decay,
            "group_name": group_name,
        })
        assigned_ids.extend(id(parameter) for parameter in parameters)
        group_contracts.append({
            "group_name": group_name,
            "learning_rate": group_learning_rate,
            "weight_decay": group_weight_decay,
            "parameter_names": [name for name, _ in entries],
            "parameter_tensors": len(entries),
        })

    expected_ids = [id(parameter) for _, parameter in named_parameters]
    groups_disjoint = len(assigned_ids) == len(set(assigned_ids))
    groups_cover_all = set(assigned_ids) == set(expected_ids)
    head_names = [name for name, parameter in named_parameters if id(parameter) in head_parameter_ids]
    classifier_norm_names = [
        name
        for name, parameter in named_parameters
        if name.startswith("classifier.")
        and id(parameter) not in head_parameter_ids
        and int(parameter.ndim) <= 1
    ]
    if not groups_disjoint or not groups_cover_all:
        raise RuntimeError("Dog Breed DB-STAB optimizer groups are not disjoint and complete")
    if not head_names or not classifier_norm_names:
        raise RuntimeError("Dog Breed DB-STAB classifier head/normalization split is incomplete")
    if any(
        contract["learning_rate"] != learning_rate * 0.1
        or contract["weight_decay"] != 0.0
        for contract in group_contracts
        if any(name in classifier_norm_names for name in contract["parameter_names"])
    ):
        raise RuntimeError("Dog Breed classifier normalization escaped the 0.1x/no-decay group")

    return ordered_groups, {
        "schema": "evomind.dog_breed.stability_optimizer.v1",
        "mode": "classifier_linear_1.0x_pretrained_and_classifier_norm_0.1x",
        "base_learning_rate": learning_rate,
        "pretrained_learning_rate": learning_rate * 0.1,
        "task_head_learning_rate": learning_rate,
        "decay_weight_decay": decay,
        "no_decay_weight_decay": 0.0,
        "classifier_path": classifier_path,
        "task_head_parameter_names": head_names,
        "classifier_norm_parameter_names": classifier_norm_names,
        "groups": group_contracts,
        "groups_disjoint": groups_disjoint,
        "groups_cover_all_trainable_parameters": groups_cover_all,
        "trainable_parameter_tensors": len(named_parameters),
    }


def build_dog_breed_frozen_head_optimizer_groups(
    model: Any,
    *,
    base_learning_rate: float,
    classifier_path: str = "classifier.2",
    weight_decay: float = DOG_BREED_FROZEN_HEAD_WEIGHT_DECAY,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Freeze the pretrained feature extractor and train only its mapped linear head."""

    learning_rate = float(base_learning_rate)
    decay = float(weight_decay)
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise RuntimeError("Dog Breed frozen-head learning rate must be positive")
    if not np.isfinite(decay) or decay < 0.0:
        raise RuntimeError("Dog Breed frozen-head weight decay must be non-negative")

    head = _resolve_named_module(model, classifier_path)
    head_parameter_ids = {id(parameter) for parameter in head.parameters(recurse=False)}
    if not head_parameter_ids:
        raise RuntimeError("Dog Breed frozen-head classifier has no direct parameters")
    named_parameters = list(model.named_parameters())
    for _name, parameter in named_parameters:
        parameter.requires_grad = id(parameter) in head_parameter_ids
    trainable = [
        (name, parameter)
        for name, parameter in named_parameters
        if parameter.requires_grad
    ]
    frozen = [name for name, parameter in named_parameters if not parameter.requires_grad]
    if not trainable or not frozen:
        raise RuntimeError("Dog Breed frozen-head parameter partition is incomplete")
    if {id(parameter) for _name, parameter in trainable} != head_parameter_ids:
        raise RuntimeError("Dog Breed frozen-head trainable set escaped the mapped classifier")

    decay_parameters = [
        (name, parameter) for name, parameter in trainable if int(parameter.ndim) > 1
    ]
    no_decay_parameters = [
        (name, parameter) for name, parameter in trainable if int(parameter.ndim) <= 1
    ]
    groups = []
    contracts = []
    for group_name, entries, group_decay in (
        ("mapped_head_decay", decay_parameters, decay),
        ("mapped_head_no_decay", no_decay_parameters, 0.0),
    ):
        if not entries:
            continue
        groups.append(
            {
                "params": [parameter for _name, parameter in entries],
                "lr": learning_rate,
                "weight_decay": group_decay,
                "group_name": group_name,
            }
        )
        contracts.append(
            {
                "group_name": group_name,
                "learning_rate": learning_rate,
                "weight_decay": group_decay,
                "parameter_names": [name for name, _parameter in entries],
            }
        )
    assigned = [id(parameter) for group in groups for parameter in group["params"]]
    groups_disjoint = len(assigned) == len(set(assigned))
    groups_cover_all = set(assigned) == head_parameter_ids
    if not groups_disjoint or not groups_cover_all:
        raise RuntimeError("Dog Breed frozen-head groups are not disjoint and complete")
    return groups, {
        "schema": "evomind.dog_breed.frozen_backbone_head_optimizer.v1",
        "mode": "frozen_backbone_imagenet_head_only",
        "classifier_path": classifier_path,
        "base_learning_rate": learning_rate,
        "decay_weight_decay": decay,
        "no_decay_weight_decay": 0.0,
        "trainable_parameter_names": [name for name, _parameter in trainable],
        "frozen_parameter_tensor_count": len(frozen),
        "trainable_parameter_tensor_count": len(trainable),
        "groups": contracts,
        "groups_disjoint": groups_disjoint,
        "groups_cover_all_trainable_parameters": groups_cover_all,
        "backbone_frozen": True,
        "imagenet_mapped_head_only": True,
    }


def dog_breed_stability_lr_factor(
    step: int,
    *,
    total_steps: int,
    warmup_fraction: float = DOG_BREED_STAB_WARMUP_FRACTION,
    minimum_factor: float = DOG_BREED_STAB_MIN_LR_FACTOR,
) -> float:
    """Return the per-update 10% warmup plus cosine-to-5% DB-STAB factor."""

    count = int(total_steps)
    fraction = float(warmup_fraction)
    floor = float(minimum_factor)
    if count <= 1 or not 0.0 < fraction < 1.0 or not 0.0 < floor <= 1.0:
        raise RuntimeError("Dog Breed DB-STAB scheduler contract is invalid")
    warmup_steps = max(1, min(count - 1, int(math.ceil(count * fraction))))
    current = max(0, min(int(step), count - 1))
    if current < warmup_steps:
        return float(current + 1) / float(warmup_steps)
    decay_steps = count - warmup_steps
    progress = float(current - warmup_steps + 1) / float(decay_steps)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_dog_breed_stability_scheduler(
    optimizer: Any,
    *,
    total_steps: int,
) -> tuple[Any, dict[str, Any]]:
    import torch

    count = int(total_steps)
    warmup_steps = max(
        1,
        min(count - 1, int(math.ceil(count * DOG_BREED_STAB_WARMUP_FRACTION))),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: dog_breed_stability_lr_factor(
            step,
            total_steps=count,
        ),
    )
    return scheduler, {
        "schema": "evomind.dog_breed.per_update_scheduler.v1",
        "mode": "linear_warmup_10pct_then_cosine_to_5pct",
        "total_steps": count,
        "warmup_steps": warmup_steps,
        "warmup_fraction": DOG_BREED_STAB_WARMUP_FRACTION,
        "minimum_learning_rate_factor": DOG_BREED_STAB_MIN_LR_FACTOR,
        "first_update_factor": dog_breed_stability_lr_factor(0, total_steps=count),
        "peak_factor": dog_breed_stability_lr_factor(
            warmup_steps - 1,
            total_steps=count,
        ),
        "final_update_factor": dog_breed_stability_lr_factor(
            count - 1,
            total_steps=count,
        ),
        "step_unit": "optimizer_update",
    }


def normalize_dog_breed_name(value: str) -> str:
    """Normalize Kaggle and torchvision dog labels to one strict join key."""

    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def build_dog_breed_imagenet_mapping(
    classes: Sequence[str],
    imagenet_categories: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map all Dog Breed columns to the pinned ImageNet canine classifier span."""

    class_names = [str(value) for value in classes]
    categories = [str(value) for value in imagenet_categories]
    if not class_names or len(set(class_names)) != len(class_names):
        raise RuntimeError("Dog Breed classes must be non-empty and unique")
    try:
        canine_start = categories.index("Chihuahua")
        canine_end = categories.index("African hunting dog")
    except ValueError as exc:
        raise RuntimeError("Pinned ImageNet metadata is missing the canine category span") from exc
    if canine_end <= canine_start:
        raise RuntimeError("Pinned ImageNet canine category span is malformed")

    canine_lookup: dict[str, list[tuple[int, str]]] = {}
    for index in range(canine_start, canine_end + 1):
        name = categories[index]
        canine_lookup.setdefault(normalize_dog_breed_name(name), []).append((index, name))

    mapped: list[int] = []
    mapped_names: dict[str, str] = {}
    missing: list[str] = []
    ambiguous: dict[str, list[str]] = {}
    for class_name in class_names:
        matches = canine_lookup.get(normalize_dog_breed_name(class_name), [])
        if not matches:
            missing.append(class_name)
            continue
        if len(matches) != 1:
            ambiguous[class_name] = [name for _, name in matches]
            continue
        index, category = matches[0]
        mapped.append(index)
        mapped_names[class_name] = category
    if missing or ambiguous or len(mapped) != len(class_names) or len(set(mapped)) != len(mapped):
        raise RuntimeError(
            "Dog Breed ImageNet mapping is incomplete or ambiguous: "
            f"missing={missing[:5]} ambiguous={list(ambiguous)[:5]}"
        )
    return np.asarray(mapped, dtype=np.int64), {
        "coverage_count": len(mapped),
        "class_count": len(class_names),
        "coverage_fraction": len(mapped) / len(class_names),
        "canine_span": [canine_start, canine_end],
        "category_indices": mapped,
        "category_names_by_class": mapped_names,
        "missing_classes": missing,
        "ambiguous_classes": ambiguous,
    }


def _normalize_multiclass_probability(values: np.ndarray) -> np.ndarray:
    probability = np.asarray(values, dtype=np.float64)
    if probability.ndim != 2 or not np.isfinite(probability).all():
        raise RuntimeError("Multiclass probabilities must be a finite matrix")
    if np.any(probability < 0.0):
        raise RuntimeError("Multiclass probabilities cannot be negative")
    probability = np.clip(probability, 1e-12, None)
    row_sum = probability.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0.0):
        raise RuntimeError("Multiclass probability rows must have positive mass")
    return probability / row_sum


def _indexed_multiclass_log_loss(target: np.ndarray, probability: np.ndarray) -> float:
    matrix = _normalize_multiclass_probability(probability)
    truth = np.asarray(target, dtype=np.int64).reshape(-1)
    if len(truth) != len(matrix) or (
        len(truth) and (int(truth.min()) < 0 or int(truth.max()) >= matrix.shape[1])
    ):
        raise RuntimeError("Multiclass target/probability alignment is invalid")
    return float(-np.mean(np.log(np.clip(matrix[np.arange(len(truth)), truth], 1e-12, 1.0))))


def _multiclass_expected_calibration_error(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    bins: int = 15,
) -> float:
    matrix = _normalize_multiclass_probability(probability)
    truth = np.asarray(target, dtype=np.int64).reshape(-1)
    if len(truth) != len(matrix) or bins < 2:
        raise RuntimeError("Multiclass ECE arrays are invalid")
    confidence = matrix.max(axis=1)
    correct = (matrix.argmax(axis=1) == truth).astype(np.float64)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    total = max(1, len(truth))
    error = 0.0
    for index in range(int(bins)):
        lower, upper = edges[index], edges[index + 1]
        selected = (confidence > lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        if selected.any():
            error += float(selected.sum()) / total * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return float(error)


def apply_dog_breed_probability_blend(
    fine_tuned_probability: np.ndarray,
    teacher_probability: np.ndarray,
    *,
    fine_tuned_weight: float,
    temperature: float,
    blend_family: str = "arithmetic_probability",
) -> np.ndarray:
    fine_tuned = _normalize_multiclass_probability(fine_tuned_probability)
    teacher = _normalize_multiclass_probability(teacher_probability)
    if fine_tuned.shape != teacher.shape:
        raise RuntimeError("Dog Breed blend components must have identical shapes")
    if not np.isfinite(fine_tuned_weight) or not 0.0 <= fine_tuned_weight <= 1.0:
        raise RuntimeError("Dog Breed fine-tuned blend weight must be inside [0, 1]")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise RuntimeError("Dog Breed blend temperature must be positive")
    if blend_family not in DOG_BREED_BLEND_FAMILIES:
        raise ValueError(f"Unsupported Dog Breed blend family: {blend_family}")
    if blend_family == "arithmetic_probability":
        mixture = (
            float(fine_tuned_weight) * fine_tuned
            + (1.0 - float(fine_tuned_weight)) * teacher
        )
        log_probability = np.log(np.clip(mixture, 1e-12, 1.0))
    else:
        log_probability = (
            float(fine_tuned_weight) * np.log(np.clip(fine_tuned, 1e-12, 1.0))
            + (1.0 - float(fine_tuned_weight))
            * np.log(np.clip(teacher, 1e-12, 1.0))
        )
    log_probability /= float(temperature)
    log_probability -= log_probability.max(axis=1, keepdims=True)
    scaled = np.exp(log_probability)
    return scaled / np.maximum(scaled.sum(axis=1, keepdims=True), 1e-12)


def select_dog_breed_probability_blend(
    fine_tuned_probability: np.ndarray,
    teacher_probability: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Jointly fit a bounded blend family, weight, and temperature."""

    fine_tuned = _normalize_multiclass_probability(fine_tuned_probability)
    teacher = _normalize_multiclass_probability(teacher_probability)
    truth = np.asarray(target, dtype=np.int64).reshape(-1)
    if fine_tuned.shape != teacher.shape or len(truth) != len(fine_tuned):
        raise RuntimeError("Dog Breed blend fitting arrays are misaligned")

    best_prediction: np.ndarray | None = None
    best_report: dict[str, Any] | None = None
    evaluated: set[tuple[str, float, float]] = set()

    def evaluate(blend_family: str, fine_weight: float, temperature: float) -> None:
        nonlocal best_prediction, best_report
        key = (blend_family, round(float(fine_weight), 12), round(float(temperature), 12))
        if key in evaluated:
            return
        evaluated.add(key)
        prediction = apply_dog_breed_probability_blend(
            fine_tuned,
            teacher,
            fine_tuned_weight=float(fine_weight),
            temperature=float(temperature),
            blend_family=blend_family,
        )
        score = _indexed_multiclass_log_loss(truth, prediction)
        family_rank = DOG_BREED_BLEND_FAMILIES.index(blend_family)
        candidate_order = (
            float(score),
            family_rank,
            abs(math.log(float(temperature))),
            -float(fine_weight),
            float(temperature),
        )
        current_order = None if best_report is None else best_report["selection_order"]
        if current_order is None or candidate_order < tuple(current_order):
            best_prediction = prediction
            best_report = {
                "blend_family": blend_family,
                "fine_tuned_weight": float(fine_weight),
                "teacher_weight": 1.0 - float(fine_weight),
                "temperature": float(temperature),
                "fitting_log_loss": float(score),
                "selection_order": list(candidate_order),
            }

    coarse_weights = np.linspace(0.0, 1.0, DOG_BREED_BLEND_COARSE_WEIGHT_COUNT)
    coarse_temperatures = np.unique(
        np.concatenate(
            [
                np.geomspace(0.5, 2.0, DOG_BREED_BLEND_COARSE_TEMPERATURE_COUNT),
                np.asarray([1.0]),
            ]
        )
    )
    for blend_family in DOG_BREED_BLEND_FAMILIES:
        for fine_weight in coarse_weights:
            for temperature in coarse_temperatures:
                evaluate(blend_family, float(fine_weight), float(temperature))

    assert best_report is not None
    coarse_weight_step = 1.0 / max(1, DOG_BREED_BLEND_COARSE_WEIGHT_COUNT - 1)
    coarse_log_temperature_step = math.log(4.0) / max(
        1, DOG_BREED_BLEND_COARSE_TEMPERATURE_COUNT - 1
    )
    refined_weights = np.unique(
        np.clip(
            float(best_report["fine_tuned_weight"])
            + np.linspace(
                -coarse_weight_step,
                coarse_weight_step,
                DOG_BREED_BLEND_REFINEMENT_COUNT,
            ),
            0.0,
            1.0,
        )
    )
    refined_temperatures = np.unique(
        np.clip(
            np.exp(
                math.log(float(best_report["temperature"]))
                + np.linspace(
                    -coarse_log_temperature_step,
                    coarse_log_temperature_step,
                    DOG_BREED_BLEND_REFINEMENT_COUNT,
                )
            ),
            0.5,
            2.0,
        )
    )
    refined_family = str(best_report["blend_family"])
    for fine_weight in refined_weights:
        for temperature in refined_temperatures:
            evaluate(refined_family, float(fine_weight), float(temperature))

    assert best_prediction is not None and best_report is not None
    best_report.pop("selection_order", None)
    best_report.update(
        {
            "fine_tuned_only_log_loss": _indexed_multiclass_log_loss(truth, fine_tuned),
            "teacher_only_log_loss": _indexed_multiclass_log_loss(truth, teacher),
            "candidate_count": len(evaluated),
            "search_contract": "joint_family_weight_temperature_coarse_to_fine_v1",
            "available_blend_families": list(DOG_BREED_BLEND_FAMILIES),
            "coarse_weight_count": DOG_BREED_BLEND_COARSE_WEIGHT_COUNT,
            "coarse_temperature_count": int(len(coarse_temperatures)),
            "refinement_count_per_axis": DOG_BREED_BLEND_REFINEMENT_COUNT,
        }
    )
    return best_prediction, best_report


def cross_fit_dog_breed_probability_blend(
    fine_tuned_probability: np.ndarray,
    teacher_probability: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Fit Dog Breed blend parameters without each outer validation fold."""

    fine_tuned = _normalize_multiclass_probability(fine_tuned_probability)
    teacher = _normalize_multiclass_probability(teacher_probability)
    truth = np.asarray(target, dtype=np.int64).reshape(-1)
    fold_values = np.asarray(folds, dtype=np.int16).reshape(-1)
    if (
        fine_tuned.shape != teacher.shape
        or len(truth) != len(fine_tuned)
        or fold_values.shape != truth.shape
        or len(np.unique(fold_values)) < 2
    ):
        raise RuntimeError("Dog Breed cross-fit blend arrays are invalid")
    prediction = np.full_like(fine_tuned, np.nan, dtype=np.float64)
    records: list[dict[str, Any]] = []
    for fold in sorted(int(value) for value in np.unique(fold_values)):
        validation = fold_values == fold
        fitting = ~validation
        _, parameters = select_dog_breed_probability_blend(
            fine_tuned[fitting], teacher[fitting], truth[fitting]
        )
        prediction[validation] = apply_dog_breed_probability_blend(
            fine_tuned[validation],
            teacher[validation],
            fine_tuned_weight=parameters["fine_tuned_weight"],
            temperature=parameters["temperature"],
            blend_family=str(parameters["blend_family"]),
        )
        records.append({
            "fold": fold,
            "fitting_rows": int(fitting.sum()),
            "validation_rows": int(validation.sum()),
            **parameters,
            "outer_log_loss": _indexed_multiclass_log_loss(
                truth[validation], prediction[validation]
            ),
        })
    if not np.isfinite(prediction).all():
        raise RuntimeError("Dog Breed cross-fit blend did not cover every row")
    return prediction, records


def apply_dog_breed_class_bias_calibration(
    probability: np.ndarray,
    class_bias: np.ndarray,
) -> np.ndarray:
    """Apply a sum-zero class bias in log-probability space."""

    matrix = _normalize_multiclass_probability(probability)
    bias = np.asarray(class_bias, dtype=np.float64).reshape(-1)
    if matrix.shape[1] != len(bias):
        raise RuntimeError("Dog Breed class-bias dimensions are misaligned")
    if not np.isfinite(bias).all() or abs(float(bias.sum())) > 1e-8:
        raise RuntimeError("Dog Breed class bias must be finite and sum to zero")
    logits = np.log(np.clip(matrix, 1e-12, 1.0)) + bias.reshape(1, -1)
    logits -= logits.max(axis=1, keepdims=True)
    calibrated = np.exp(logits)
    calibrated /= np.maximum(calibrated.sum(axis=1, keepdims=True), 1e-12)
    if not np.isfinite(calibrated).all():
        raise RuntimeError("Dog Breed class-bias calibration produced non-finite values")
    return calibrated


def fit_dog_breed_class_bias(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    l2_penalty: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one regularized sum-zero class-bias vector on public OOF rows."""

    from scipy.optimize import minimize

    matrix = _normalize_multiclass_probability(probability)
    truth = np.asarray(target, dtype=np.int64).reshape(-1)
    if len(matrix) != len(truth) or matrix.shape[1] < 2:
        raise RuntimeError("Dog Breed class-bias fitting arrays are invalid")
    if len(truth) and (int(truth.min()) < 0 or int(truth.max()) >= matrix.shape[1]):
        raise RuntimeError("Dog Breed class-bias targets are outside the class range")
    penalty = float(l2_penalty)
    if not np.isfinite(penalty) or penalty <= 0.0:
        raise RuntimeError("Dog Breed class-bias L2 penalty must be positive")

    base_logits = np.log(np.clip(matrix, 1e-12, 1.0))
    class_count = matrix.shape[1]
    class_totals = np.bincount(truth, minlength=class_count).astype(np.float64)

    def expand(free_bias: np.ndarray) -> np.ndarray:
        free = np.asarray(free_bias, dtype=np.float64)
        return np.concatenate([free, np.asarray([-float(free.sum())])])

    def objective(free_bias: np.ndarray) -> tuple[float, np.ndarray]:
        bias = expand(free_bias)
        shifted = base_logits + bias.reshape(1, -1)
        shifted -= shifted.max(axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        calibrated = exp_shifted / np.maximum(exp_shifted.sum(axis=1, keepdims=True), 1e-12)
        loss = -float(
            np.log(np.clip(calibrated[np.arange(len(truth)), truth], 1e-12, 1.0)).sum()
        )
        loss += 0.5 * penalty * float(np.dot(bias, bias))
        gradient_bias = calibrated.sum(axis=0) - class_totals + penalty * bias
        gradient_free = gradient_bias[:-1] - gradient_bias[-1]
        return loss, gradient_free

    fitted = minimize(
        objective,
        np.zeros(class_count - 1, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 250, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50},
    )
    bias = expand(np.asarray(fitted.x, dtype=np.float64))
    if not fitted.success or not np.isfinite(bias).all() or abs(float(bias.sum())) > 1e-8:
        raise RuntimeError(
            "Dog Breed class-bias optimizer failed: "
            f"status={fitted.status} message={fitted.message}"
        )
    calibrated = apply_dog_breed_class_bias_calibration(matrix, bias)
    report = {
        "l2_penalty": penalty,
        "fitting_rows": int(len(truth)),
        "bias_l2_norm": float(np.linalg.norm(bias)),
        "bias_sum": float(bias.sum()),
        "optimizer_success": bool(fitted.success),
        "optimizer_iterations": int(fitted.nit),
        "optimizer_objective": float(fitted.fun),
        "fitting_log_loss": _indexed_multiclass_log_loss(truth, calibrated),
        "private_labels_used": False,
    }
    return bias, report


def select_dog_breed_class_bias_penalty(
    probability: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
    *,
    l2_grid: Sequence[float] = DOG_BREED_CLASS_BIAS_L2_GRID,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose L2 on a deterministic inner holdout, then refit on all supplied rows."""

    matrix = _normalize_multiclass_probability(probability)
    truth = np.asarray(target, dtype=np.int64).reshape(-1)
    fold_values = np.asarray(folds, dtype=np.int16).reshape(-1)
    unique_folds = sorted(int(value) for value in np.unique(fold_values))
    penalties = tuple(float(value) for value in l2_grid)
    if (
        len(matrix) != len(truth)
        or fold_values.shape != truth.shape
        or len(unique_folds) < 2
        or not penalties
        or any(not np.isfinite(value) or value <= 0.0 for value in penalties)
    ):
        raise RuntimeError("Dog Breed nested class-bias penalty selection is invalid")

    inner_validation_fold = unique_folds[-1]
    inner_validation = fold_values == inner_validation_fold
    inner_fitting = ~inner_validation
    candidates: list[dict[str, Any]] = []
    for penalty in penalties:
        bias, fit_report = fit_dog_breed_class_bias(
            matrix[inner_fitting],
            truth[inner_fitting],
            l2_penalty=penalty,
        )
        inner_prediction = apply_dog_breed_class_bias_calibration(
            matrix[inner_validation], bias
        )
        candidates.append({
            "l2_penalty": penalty,
            "inner_validation_log_loss": _indexed_multiclass_log_loss(
                truth[inner_validation], inner_prediction
            ),
            "inner_fit": fit_report,
        })
    selected = min(
        candidates,
        key=lambda item: (float(item["inner_validation_log_loss"]), -float(item["l2_penalty"])),
    )
    final_bias, final_fit = fit_dog_breed_class_bias(
        matrix,
        truth,
        l2_penalty=float(selected["l2_penalty"]),
    )
    return final_bias, {
        "selection_contract": "deterministic_nested_inner_holdout_fixed_l2_grid_v1",
        "l2_grid": list(penalties),
        "inner_validation_fold": inner_validation_fold,
        "inner_fitting_rows": int(inner_fitting.sum()),
        "inner_validation_rows": int(inner_validation.sum()),
        "candidates": candidates,
        "selected_l2_penalty": float(selected["l2_penalty"]),
        "final_fit": final_fit,
        "private_labels_used": False,
    }


def cross_fit_dog_breed_class_bias_calibration(
    probability: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
    *,
    l2_grid: Sequence[float] = DOG_BREED_CLASS_BIAS_L2_GRID,
    enforce_early_checkpoint: bool = True,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Nested cross-fit Dog Breed class bias without using outer-fold labels."""

    incumbent = _normalize_multiclass_probability(probability)
    truth = np.asarray(target, dtype=np.int64).reshape(-1)
    fold_values = np.asarray(folds, dtype=np.int16).reshape(-1)
    unique_folds = sorted(int(value) for value in np.unique(fold_values))
    if (
        len(incumbent) != len(truth)
        or fold_values.shape != truth.shape
        or len(unique_folds) < 3
    ):
        raise RuntimeError("Dog Breed class-bias cross-fit arrays are invalid")

    candidate = incumbent.copy()
    processed = np.zeros(len(truth), dtype=bool)
    records: list[dict[str, Any]] = []
    early_checkpoint: dict[str, Any] | None = None
    for outer_index, outer_fold in enumerate(unique_folds):
        validation = fold_values == outer_fold
        fitting = ~validation
        bias, selection = select_dog_breed_class_bias_penalty(
            incumbent[fitting],
            truth[fitting],
            fold_values[fitting],
            l2_grid=l2_grid,
        )
        candidate[validation] = apply_dog_breed_class_bias_calibration(
            incumbent[validation], bias
        )
        processed[validation] = True
        incumbent_loss = _indexed_multiclass_log_loss(
            truth[validation], incumbent[validation]
        )
        candidate_loss = _indexed_multiclass_log_loss(
            truth[validation], candidate[validation]
        )
        records.append({
            "outer_fold": outer_fold,
            "fitting_rows": int(fitting.sum()),
            "validation_rows": int(validation.sum()),
            "selected_l2_penalty": float(selection["selected_l2_penalty"]),
            "bias_l2_norm": float(np.linalg.norm(bias)),
            "incumbent_outer_log_loss": incumbent_loss,
            "candidate_outer_log_loss": candidate_loss,
            "outer_degradation": candidate_loss - incumbent_loss,
            "penalty_selection": selection,
            "private_labels_used": False,
        })
        if enforce_early_checkpoint and outer_index == 1:
            checkpoint_mask = processed.copy()
            weighted_incumbent = _indexed_multiclass_log_loss(
                truth[checkpoint_mask], incumbent[checkpoint_mask]
            )
            weighted_candidate = _indexed_multiclass_log_loss(
                truth[checkpoint_mask], candidate[checkpoint_mask]
            )
            early_checkpoint = {
                "evaluated_outer_folds": unique_folds[:2],
                "weighted_incumbent_log_loss": weighted_incumbent,
                "weighted_candidate_log_loss": weighted_candidate,
                "weighted_improvement": weighted_incumbent - weighted_candidate,
                "maximum_fold_degradation": max(
                    float(record["outer_degradation"]) for record in records
                ),
            }
            early_checkpoint["passed"] = bool(
                weighted_candidate <= 0.060
                and weighted_incumbent - weighted_candidate >= 0.005
                and early_checkpoint["maximum_fold_degradation"] <= 0.002
            )
            if not early_checkpoint["passed"]:
                return candidate, records, {
                    "schema": "evomind.dog_breed.class_bias_crossfit.v1",
                    "status": "early_checkpoint_failed",
                    "candidate_complete": False,
                    "processed_rows": int(processed.sum()),
                    "total_rows": int(len(truth)),
                    "early_checkpoint": early_checkpoint,
                    "final_deployment": None,
                    "private_labels_used": False,
                }

    if not processed.all() or not np.isfinite(candidate).all():
        raise RuntimeError("Dog Breed class-bias cross-fit did not cover every row")
    final_bias, final_selection = select_dog_breed_class_bias_penalty(
        incumbent,
        truth,
        fold_values,
        l2_grid=l2_grid,
    )
    return candidate, records, {
        "schema": "evomind.dog_breed.class_bias_crossfit.v1",
        "status": "complete",
        "candidate_complete": True,
        "processed_rows": int(processed.sum()),
        "total_rows": int(len(truth)),
        "early_checkpoint": early_checkpoint,
        "final_deployment": {
            "class_bias": final_bias.tolist(),
            **final_selection,
        },
        "private_labels_used": False,
    }


def select_dog_breed_probability_replacement(
    incumbent_probability: np.ndarray,
    candidate_probability: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
    *,
    candidate_complete: bool,
    minimum_improvement: float = 0.0005,
    maximum_fold_degradation: float = 0.003,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select a candidate only when it beats and does not destabilize the incumbent."""

    incumbent_raw = np.asarray(incumbent_probability, dtype=np.float64)
    candidate_raw = np.asarray(candidate_probability, dtype=np.float64)
    truth = np.asarray(target, dtype=np.int64).reshape(-1)
    fold_values = np.asarray(folds, dtype=np.int16).reshape(-1)
    if (
        incumbent_raw.shape != candidate_raw.shape
        or len(incumbent_raw) != len(truth)
        or fold_values.shape != truth.shape
        or not np.isfinite(incumbent_raw).all()
    ):
        raise RuntimeError("Dog Breed replacement arrays are invalid")
    incumbent = _normalize_multiclass_probability(incumbent_raw)
    candidate_valid = bool(candidate_complete and np.isfinite(candidate_raw).all())
    candidate = (
        _normalize_multiclass_probability(candidate_raw)
        if candidate_valid
        else incumbent.copy()
    )
    incumbent_loss = _indexed_multiclass_log_loss(truth, incumbent)
    candidate_loss = _indexed_multiclass_log_loss(truth, candidate)
    fold_records: list[dict[str, Any]] = []
    for fold in sorted(int(value) for value in np.unique(fold_values)):
        mask = fold_values == fold
        fold_incumbent = _indexed_multiclass_log_loss(truth[mask], incumbent[mask])
        fold_candidate = _indexed_multiclass_log_loss(truth[mask], candidate[mask])
        fold_records.append({
            "fold": fold,
            "rows": int(mask.sum()),
            "incumbent_log_loss": fold_incumbent,
            "candidate_log_loss": fold_candidate,
            "degradation": fold_candidate - fold_incumbent,
        })
    aggregate_improvement = incumbent_loss - candidate_loss
    worst_fold_degradation = max(
        (float(record["degradation"]) for record in fold_records),
        default=float("inf"),
    )
    replacement_selected = bool(
        candidate_valid
        and aggregate_improvement >= float(minimum_improvement)
        and worst_fold_degradation <= float(maximum_fold_degradation)
    )
    selected = candidate.copy() if replacement_selected else incumbent_raw.copy()
    return selected, {
        "incumbent_oof_log_loss": incumbent_loss,
        "candidate_oof_log_loss": candidate_loss,
        "aggregate_improvement": aggregate_improvement,
        "minimum_improvement": float(minimum_improvement),
        "worst_fold_degradation": worst_fold_degradation,
        "maximum_allowed_fold_degradation": float(maximum_fold_degradation),
        "candidate_complete": candidate_valid,
        "replacement_selected": replacement_selected,
        "fallback_preserved_incumbent_exactly": not replacement_selected,
        "fold_records": fold_records,
        "private_labels_used": False,
    }


def _image_transforms(
    size: int,
    *,
    center_patch: bool = False,
    monochrome: bool = False,
    vertical_flip: bool = True,
    preprocessing_profile: str = "standard",
):
    from torchvision import transforms

    prefix: list[Any] = []
    if monochrome:
        prefix.append(transforms.Grayscale(num_output_channels=3))
    if preprocessing_profile == "retina":
        prefix.append(transforms.Lambda(prepare_retina_image))
    elif preprocessing_profile == "full_frame":
        prefix.append(transforms.Lambda(pad_to_square_image))
    elif preprocessing_profile != "standard":
        raise ValueError(f"Unsupported vision preprocessing profile: {preprocessing_profile}")
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if center_patch:
        # Histopathologic labels describe the centre of each 96x96 tile.  A
        # RandomResizedCrop after CenterCrop can shift that labelled region out
        # of view, so keep the centre geometry fixed and augment orientation.
        train = transforms.Compose([
            *prefix,
            transforms.CenterCrop(64),
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
            transforms.ToTensor(),
            normalize,
        ])
        evaluate = transforms.Compose([
            *prefix,
            transforms.CenterCrop(64),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            normalize,
        ])
    elif preprocessing_profile in {"retina", "full_frame"}:
        # Preserve the complete retinal/leaf frame.  RandomResizedCrop can
        # remove peripheral lesions, so these task-specific profiles augment
        # orientation and colour without discarding image borders.
        train = transforms.Compose([
            *prefix,
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.15 if vertical_flip else 0.0),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
            transforms.ToTensor(),
            normalize,
        ])
        evaluate = transforms.Compose([
            *prefix,
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train = transforms.Compose([
            *prefix,
            transforms.Resize((size + 32, size + 32)),
            transforms.RandomResizedCrop(size, scale=(0.78, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.15 if vertical_flip else 0.0),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ])
        evaluate = transforms.Compose([
            *prefix,
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            normalize,
        ])
    return train, evaluate


def load_or_compute_dog_breed_imagenet_teacher(
    *,
    train_paths: list[Path],
    test_paths: list[Path],
    classes: Sequence[str],
    source_manifest_sha256: str,
    cache_path: Path,
    workers: int,
    batch_size: int,
    logger: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Cache frozen ImageNet canine probabilities for Dog Breed train/test rows."""

    import torch
    from torch.utils.data import DataLoader
    from torchvision.models import ConvNeXt_Small_Weights

    class_names = [str(value) for value in classes]
    weights = ConvNeXt_Small_Weights.DEFAULT
    categories = [str(value) for value in weights.meta["categories"]]
    category_indices, mapping = build_dog_breed_imagenet_mapping(class_names, categories)
    identity_payload = {
        "schema": DOG_BREED_TEACHER_CACHE_SCHEMA,
        "source_manifest_sha256": str(source_manifest_sha256),
        "classes_sha256": hashlib.sha256(
            json.dumps(class_names, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "train_rows": len(train_paths),
        "test_rows": len(test_paths),
        "weight_sha256": CONVNEXT_SMALL_WEIGHT_SHA256,
        "weight_enum": str(weights),
        "category_indices": category_indices.tolist(),
        "tta": "identity_plus_horizontal_flip_probability_average",
    }
    identity_sha256 = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    contract_path = cache_path.with_suffix(".json")
    if cache_path.is_file() and contract_path.is_file():
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            with np.load(cache_path, allow_pickle=False) as cached:
                train_probability = np.asarray(cached["train_probability"], dtype=np.float64)
                test_probability = np.asarray(cached["test_probability"], dtype=np.float64)
                cached_indices = np.asarray(cached["category_indices"], dtype=np.int64)
            cache_valid = bool(
                contract.get("identity_sha256") == identity_sha256
                and train_probability.shape == (len(train_paths), len(class_names))
                and test_probability.shape == (len(test_paths), len(class_names))
                and np.array_equal(cached_indices, category_indices)
                and np.isfinite(train_probability).all()
                and np.isfinite(test_probability).all()
                and np.allclose(train_probability.sum(axis=1), 1.0, atol=1e-6)
                and np.allclose(test_probability.sum(axis=1), 1.0, atol=1e-6)
            )
            if cache_valid:
                contract["cache_hit"] = True
                return train_probability, test_probability, contract
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    if not torch.cuda.is_available():
        raise RuntimeError("Dog Breed ImageNet teacher requires CUDA")

    transform = weights.transforms()
    effective_workers = max(0, min(int(workers), max(0, (os.cpu_count() or 4) // 2)))
    loader_options = (
        {"prefetch_factor": VISION_DATALOADER_PREFETCH_FACTOR}
        if effective_workers > 0
        else {}
    )

    def loader_for(paths: list[Path], seed: int) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(seed)
        return DataLoader(
            _DogBreedTeacherImages(paths, transform),
            batch_size=max(1, int(batch_size)),
            shuffle=False,
            num_workers=effective_workers,
            pin_memory=True,
            persistent_workers=effective_workers > 0,
            generator=generator,
            worker_init_fn=_seed_vision_worker,
            **loader_options,
        )

    model, pretrained, weight_identity = _vision_model(
        len(categories),
        backbone="convnext_small",
        require_pretrained=True,
        preserve_imagenet_classifier=True,
    )
    if not pretrained or weight_identity.get("sha256") != CONVNEXT_SMALL_WEIGHT_SHA256:
        raise RuntimeError("Dog Breed teacher did not load the pinned ImageNet classifier")
    model = model.to(device="cuda", memory_format=torch.channels_last)
    model.eval()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    selected_indices = torch.tensor(category_indices, dtype=torch.long, device="cuda")

    def infer(
        paths: list[Path],
        seed: int,
        inference_model: Any,
        selected_category_indices: Any,
    ) -> np.ndarray:
        predictions: list[np.ndarray] = []
        loader = loader_for(paths, seed)
        with torch.inference_mode():
            for images in loader:
                images = images.cuda(non_blocking=True, memory_format=torch.channels_last)
                variants = (images, torch.flip(images, dims=[3]))
                variant_probabilities = []
                for variant in variants:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        logits = inference_model(variant)
                    canine_logits = torch.index_select(
                        logits.float(),
                        1,
                        selected_category_indices,
                    )
                    variant_probabilities.append(torch.softmax(canine_logits, dim=1))
                probability = torch.stack(variant_probabilities).mean(dim=0)
                predictions.append(probability.cpu().numpy())
        del loader
        return _normalize_multiclass_probability(np.concatenate(predictions))

    started = time.perf_counter()
    logger.info("[dog-breed-identification] computing frozen ImageNet teacher probabilities")
    train_probability = infer(train_paths, 310_001, model, selected_indices)
    test_probability = infer(test_paths, 310_002, model, selected_indices)
    runtime_seconds = time.perf_counter() - started
    del model, selected_indices
    torch.cuda.empty_cache()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        train_probability=np.asarray(train_probability, dtype=np.float32),
        test_probability=np.asarray(test_probability, dtype=np.float32),
        category_indices=category_indices,
    )
    temporary.replace(cache_path)
    contract = {
        "schema": DOG_BREED_TEACHER_CACHE_SCHEMA,
        "identity_sha256": identity_sha256,
        "identity": identity_payload,
        "cache_path": str(cache_path),
        "cache_hit": False,
        "mapping": mapping,
        "mapping_complete": mapping["coverage_count"] == len(class_names),
        "teacher_classifier_preserved": True,
        "teacher_backbone": "convnext_small",
        "teacher_weight_identity": weight_identity,
        "teacher_transform": str(transform),
        "teacher_tta": "identity_plus_horizontal_flip_probability_average",
        "batch_size": max(1, int(batch_size)),
        "workers": effective_workers,
        "runtime_seconds": runtime_seconds,
        "private_labels_used": False,
        "created_at": wave0.utc_now(),
    }
    wave0.write_json(contract_path, contract)
    return train_probability, test_probability, contract


def pad_to_square_image(image: Any, *, fill: tuple[int, int, int] = (0, 0, 0)) -> Any:
    """Pad a PIL image to a square without cropping any source pixel."""

    from PIL import Image

    source = image.convert("RGB")
    width, height = source.size
    if width <= 0 or height <= 0:
        raise RuntimeError("Vision preprocessing received an empty image")
    side = max(width, height)
    if width == height:
        return source.copy()
    result = Image.new("RGB", (side, side), color=fill)
    result.paste(source, ((side - width) // 2, (side - height) // 2))
    return result


def prepare_retina_image(image: Any, *, black_threshold: int = 12) -> Any:
    """Deterministically remove black retinal borders and normalize contrast.

    The crop is derived solely from image pixels.  Rows/columns must contain a
    small fraction of non-black pixels, which rejects isolated border noise
    while retaining the full retinal support.  The result is padded rather than
    stretched before the downstream fixed-size resize.
    """

    from PIL import Image, ImageOps

    source = image.convert("RGB")
    values = np.asarray(source, dtype=np.uint8)
    if values.ndim != 3 or values.shape[2] != 3:
        raise RuntimeError("Retina preprocessing requires an RGB image")
    active = values.max(axis=2) > int(black_threshold)
    minimum_row_pixels = max(1, int(math.ceil(values.shape[1] * 0.02)))
    minimum_col_pixels = max(1, int(math.ceil(values.shape[0] * 0.02)))
    rows = np.flatnonzero(active.sum(axis=1) >= minimum_row_pixels)
    columns = np.flatnonzero(active.sum(axis=0) >= minimum_col_pixels)
    if rows.size and columns.size:
        cropped = source.crop(
            (int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1)
        )
    else:
        cropped = source
    normalized = ImageOps.autocontrast(cropped, cutoff=1)
    if not isinstance(normalized, Image.Image):
        raise RuntimeError("Retina preprocessing did not return a PIL image")
    return pad_to_square_image(normalized)


def make_vision_splits(
    *,
    sample_count: int,
    requested_folds: int,
    seed: int,
    mode: str,
    target: np.ndarray,
    groups: np.ndarray | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    """Build valid vision folds, falling back when grouping is degenerate."""

    from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

    if sample_count < 2:
        raise RuntimeError("Vision cross-validation requires at least two training rows")
    indices = np.arange(sample_count)
    effective_folds = max(2, min(requested_folds, sample_count))
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != sample_count:
            raise RuntimeError("Vision group cardinality mismatch")
        unique_group_count = len(np.unique(groups))
        if unique_group_count >= 2:
            grouped_folds = min(effective_folds, unique_group_count)
            if mode in {"binary", "ordinal", "multiclass"}:
                for candidate_folds in range(grouped_folds, 1, -1):
                    splitter = StratifiedGroupKFold(
                        n_splits=candidate_folds,
                        shuffle=True,
                        random_state=seed,
                    )
                    candidate = list(splitter.split(indices, target, groups))
                    if mode != "binary" or all(
                        np.unique(target[train_indices]).size == 2
                        and np.unique(target[valid_indices]).size == 2
                        for train_indices, valid_indices in candidate
                    ):
                        return candidate, "stratified_group_kfold"
                raise RuntimeError("Grouped binary vision CV could not form two-class folds")
            if mode == "multilabel":
                matrix = np.asarray(target, dtype=np.int8)
                if matrix.ndim != 2 or not set(np.unique(matrix).tolist()) <= {0, 1}:
                    raise RuntimeError("Grouped multilabel vision targets must be a binary matrix")
                signatures = np.asarray(
                    ["".join(str(int(value)) for value in row) for row in matrix],
                    dtype=object,
                )
                for candidate_folds in range(grouped_folds, 1, -1):
                    for attempt in range(12):
                        splitter = StratifiedGroupKFold(
                            n_splits=candidate_folds,
                            shuffle=True,
                            random_state=seed + attempt * 997,
                        )
                        candidate = list(splitter.split(indices, signatures, groups))
                        if vision_splits_are_scoreable(
                            mode=mode,
                            target=matrix,
                            splits=candidate,
                        ):
                            return candidate, "multilabel_stratified_group_kfold"
                raise RuntimeError(
                    "Grouped multilabel vision CV could not form fully scoreable folds"
                )
            splitter = GroupKFold(n_splits=grouped_folds)
            return list(splitter.split(indices, groups=groups)), "group_kfold"
        splitter = KFold(n_splits=effective_folds, shuffle=True, random_state=seed)
        return list(splitter.split(indices)), "kfold_single_group_fallback"
    if mode in {"binary", "ordinal", "multiclass"}:
        splitter = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
        return list(splitter.split(indices, target)), "stratified_kfold"
    splitter = KFold(n_splits=effective_folds, shuffle=True, random_state=seed)
    return list(splitter.split(indices)), "kfold"


def vision_splits_are_scoreable(
    *,
    mode: str,
    target: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> bool:
    values = np.asarray(target)
    if not splits:
        return False
    if mode == "binary":
        return all(
            np.unique(values[training_indices]).size == 2
            and np.unique(values[validation_indices]).size == 2
            for training_indices, validation_indices in splits
        )
    if mode == "multilabel":
        if values.ndim != 2:
            return False
        return all(
            all(
                np.unique(values[training_indices, label]).size == 2
                and np.unique(values[validation_indices, label]).size == 2
                for label in range(values.shape[1])
            )
            for training_indices, validation_indices in splits
        )
    return True


def binary_ranking_violations(
    truth: np.ndarray,
    probability: np.ndarray,
) -> dict[str, int]:
    """Count positive/negative ranking errors without materializing all pairs."""

    labels = np.asarray(truth).reshape(-1)
    scores = np.asarray(probability, dtype=np.float64).reshape(-1)
    if labels.shape != scores.shape or not np.isfinite(scores).all():
        raise RuntimeError("Binary ranking gate inputs are misaligned or non-finite")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise RuntimeError("Binary ranking gate requires both target classes")
    negative = np.sort(scores[labels == 0])
    positive = scores[labels == 1]
    greater = int(
        sum(len(negative) - int(np.searchsorted(negative, value, side="right")) for value in positive)
    )
    ties = int(
        sum(
            int(np.searchsorted(negative, value, side="right"))
            - int(np.searchsorted(negative, value, side="left"))
            for value in positive
        )
    )
    return {
        "strict_inversions": greater,
        "positive_negative_ties": ties,
        "non_strict_violations": greater + ties,
        "positive_negative_pairs": int(len(positive) * len(negative)),
    }


def evaluate_vision_promotion_gate(
    contract: dict[str, Any],
    *,
    metric: str,
    cv_score: float,
    fold_scores: Iterable[float],
    truth: np.ndarray,
    oof_probability: np.ndarray,
    extra_checks: dict[str, bool] | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a predeclared vision gate before any private scoring call."""

    expected_metric = str(contract["metric"])
    direction = str(contract["direction"])
    threshold = float(contract["threshold"])
    fold_threshold = float(contract.get("fold_threshold", threshold))
    value = float(cv_score)
    folds = [float(item) for item in fold_scores]
    finite = (
        math.isfinite(value)
        and math.isfinite(threshold)
        and math.isfinite(fold_threshold)
        and bool(folds)
    )
    finite = finite and all(math.isfinite(item) for item in folds)
    if direction == "minimize":
        aggregate_passed = finite and value <= threshold
        fold_passed = finite and all(item <= fold_threshold for item in folds)
        operator = "<="
    elif direction == "maximize":
        aggregate_passed = finite and value >= threshold
        fold_passed = finite and all(item >= fold_threshold for item in folds)
        operator = ">="
    else:
        raise ValueError(f"Unsupported vision promotion direction: {direction}")
    checks: dict[str, bool] = {
        "metric_matches_contract": metric == expected_metric,
        "finite_internal_metrics": finite,
        "aggregate_threshold": aggregate_passed,
    }
    if bool(contract.get("require_all_folds")):
        checks["every_fold_threshold"] = fold_passed
    ranking: dict[str, int] | None = None
    if bool(contract.get("require_zero_binary_ranking_violations")):
        ranking = binary_ranking_violations(truth, oof_probability)
        checks["zero_positive_negative_ranking_violations"] = (
            ranking["non_strict_violations"] == 0
        )
    checks.update(extra_checks or {})
    return {
        "schema": "evomind.mlebench_lite.vision_promotion_gate.v1",
        "name": str(contract["name"]),
        "metric": expected_metric,
        "direction": direction,
        "operator": operator,
        "threshold": threshold,
        "fold_threshold": fold_threshold,
        "internal_score": value,
        "fold_scores": folds,
        "checks": checks,
        "ranking": ranking,
        "evidence": evidence or {},
        "passed": bool(all(checks.values())),
        "official_grader_executed": False,
        "claim_boundary": "Internal promotion evidence is not an official medal.",
    }


def _run_vision(
    *,
    args: argparse.Namespace,
    task_dir: Path,
    logger: Any,
    competition_id: str,
    train_frame: pd.DataFrame,
    sample: pd.DataFrame,
    train_paths: list[Path],
    test_paths: list[Path],
    target_columns: list[str],
    mode: str,
    epochs: int,
    backbone: str = "convnext_tiny",
    label_column: str | None = None,
    image_size: int | None = None,
    batch_size: int | None = None,
    center_patch: bool = False,
    monochrome: bool = False,
    vertical_flip: bool = True,
    tta_flips: bool = False,
    group_values: Iterable[Any] | None = None,
    fold_count: int | None = None,
    binary_metric: str = "roc_auc",
    learning_rate: float | None = None,
    worker_count: int | None = None,
    promotion_contract: dict[str, Any] | None = None,
    group_contract: dict[str, Any] | None = None,
    test_probability_override: np.ndarray | None = None,
    preprocessing_profile: str = "standard",
    early_stopping_min_epochs: int = 3,
    dog_breed_teacher_cache_path: Path | None = None,
    dog_breed_teacher_source_manifest_sha256: str | None = None,
    diagnostic_fold_limit: int = 0,
    diagnostic_parent_fold0_epoch1_log_loss: float | None = None,
    diagnostic_parent_fold0_top1_accuracy: float | None = None,
    dog_breed_training_mode: str = "stability_finetune",
) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    if competition_id in WAVE2_PROMOTION_CONTRACTS and promotion_contract is None:
        raise RuntimeError(f"Missing frozen promotion contract for {competition_id}")
    dog_breed_teacher_requested = bool(
        dog_breed_teacher_cache_path is not None
        or dog_breed_teacher_source_manifest_sha256 is not None
    )
    if competition_id == "dog-breed-identification":
        if (
            mode != "multiclass"
            or dog_breed_teacher_cache_path is None
            or not dog_breed_teacher_source_manifest_sha256
        ):
            raise RuntimeError("Dog Breed requires its complete ImageNet teacher contract")
        if dog_breed_training_mode not in DOG_BREED_TRAINING_MODES:
            raise RuntimeError("Dog Breed training mode is not registered")
    elif dog_breed_teacher_requested:
        raise RuntimeError("Dog Breed ImageNet teacher options cannot be used by another task")
    elif dog_breed_training_mode != "stability_finetune":
        raise RuntimeError("Dog Breed training mode cannot be used by another task")
    diagnostic_fold_limit = int(diagnostic_fold_limit)
    diagnostic_mode = diagnostic_fold_limit > 0
    if diagnostic_mode:
        parent_epoch_one = float(diagnostic_parent_fold0_epoch1_log_loss or math.nan)
        parent_top1 = float(diagnostic_parent_fold0_top1_accuracy or math.nan)
        if competition_id != "dog-breed-identification" or diagnostic_fold_limit != 1:
            raise RuntimeError("DB-STAB1 diagnostic is restricted to Dog Breed fixed fold zero")
        if (
            not np.isfinite(parent_epoch_one)
            or parent_epoch_one <= 0.0
            or not np.isfinite(parent_top1)
            or not 0.0 <= parent_top1 <= 1.0
        ):
            raise RuntimeError("DB-STAB1 requires explicit parent fold-zero baselines")
        epochs = min(int(epochs), DOG_BREED_STAB_MAX_EPOCHS)
    else:
        parent_epoch_one = math.nan
        parent_top1 = math.nan
    if len(train_frame) != len(train_paths) or len(sample) != len(test_paths):
        raise RuntimeError("Image metadata/path cardinality mismatch")
    missing_train = [str(path) for path in train_paths if not path.is_file()]
    missing_test = [str(path) for path in test_paths if not path.is_file()]
    if missing_train or missing_test:
        raise FileNotFoundError(
            f"Image path manifest failed: train_missing={len(missing_train)} test_missing={len(missing_test)}"
        )

    if mode in {"binary", "ordinal", "multiclass"}:
        raw_target = train_frame[label_column or target_columns[0]].to_numpy()
        if mode == "multiclass":
            classes = list(target_columns)
            class_to_index = {value: index for index, value in enumerate(classes)}
            target = np.asarray([class_to_index[str(value)] for value in raw_target], dtype=np.int64)
            output_count = len(classes)
        elif mode == "ordinal":
            target = raw_target.astype(np.int64)
            classes = list(range(int(target.max()) + 1))
            output_count = len(classes)
        else:
            target = raw_target.astype(np.float32)
            classes = [0, 1]
            output_count = 1
    else:
        target = train_frame[target_columns].to_numpy(dtype=np.float32)
        classes = target_columns
        output_count = len(target_columns)

    groups = None if group_values is None else np.asarray(list(group_values))
    splits, split_strategy = make_vision_splits(
        sample_count=len(train_frame),
        requested_folds=fold_count or args.wave2_vision_folds,
        seed=args.seed,
        mode=mode,
        target=target,
        groups=groups,
    )
    planned_folds = len(splits)
    if not vision_splits_are_scoreable(mode=mode, target=target, splits=splits):
        raise RuntimeError("Vision folds are not scoreable before GPU allocation")
    group_overlap_by_fold = {
        str(fold): (
            0
            if groups is None
            else len(
                set(groups[training_indices].tolist())
                & set(groups[validation_indices].tolist())
            )
        )
        for fold, (training_indices, validation_indices) in enumerate(splits)
    }
    if groups is not None and any(group_overlap_by_fold.values()):
        raise RuntimeError("Vision group leakage detected before GPU allocation")
    if diagnostic_mode:
        splits = splits[:diagnostic_fold_limit]
    effective_folds = len(splits)

    effective_size = image_size or args.wave2_image_size
    effective_batch = batch_size or args.wave2_batch_size
    train_transform, eval_transform = _image_transforms(
        effective_size,
        center_patch=center_patch,
        monochrome=monochrome,
        vertical_flip=vertical_flip,
        preprocessing_profile=preprocessing_profile,
    )

    requested_workers = args.wave2_workers if worker_count is None else worker_count
    workers = min(requested_workers, max(0, (os.cpu_count() or 4) // 2))

    def loader_for(
        selected: np.ndarray | None,
        *,
        train_mode: bool,
        include_targets: bool,
        loader_seed: int,
    ) -> DataLoader:
        if selected is None:
            paths = test_paths
            values = None
        else:
            paths = [train_paths[int(index)] for index in selected]
            values = target[selected] if include_targets else None
        generator = torch.Generator()
        generator.manual_seed(loader_seed)
        loader_options = (
            {"prefetch_factor": VISION_DATALOADER_PREFETCH_FACTOR}
            if workers > 0
            else {}
        )
        return DataLoader(
            _VisionImages(
                paths,
                values,
                train_transform if train_mode else eval_transform,
                mode,
            ),
            batch_size=effective_batch,
            shuffle=train_mode,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
            generator=generator,
            worker_init_fn=_seed_vision_worker,
            **loader_options,
        )

    def build_loss(training_indices: np.ndarray):
        if mode in {"ordinal", "multiclass"}:
            counts = np.bincount(target[training_indices].astype(int), minlength=output_count)
            weights = np.where(counts > 0, len(training_indices) / (output_count * counts), 0.0)
            label_smoothing = 0.0 if competition_id == "dog-breed-identification" else 0.02
            return nn.CrossEntropyLoss(
                weight=torch.tensor(weights, dtype=torch.float32, device="cuda"),
                label_smoothing=label_smoothing,
            )
        matrix = target[training_indices]
        matrix = matrix.reshape(-1, 1) if matrix.ndim == 1 else matrix
        positive = matrix.sum(axis=0)
        negative = len(matrix) - positive
        pos_weight = np.clip(negative / np.maximum(positive, 1), 1.0, 25.0)
        return nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device="cuda")
        )

    def infer(
        model: Any,
        loader: DataLoader,
        *,
        apply_tta: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        predictions: list[np.ndarray] = []
        raw_logits: list[np.ndarray] = []
        truths: list[np.ndarray] = []
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                images = images.cuda(
                    non_blocking=True,
                    memory_format=torch.channels_last,
                )
                variants = [images]
                if apply_tta:
                    variants.append(torch.flip(images, dims=[3]))
                    if vertical_flip:
                        variants.extend([torch.flip(images, dims=[2]), torch.flip(images, dims=[2, 3])])
                variant_values: list[Any] = []
                variant_logits: list[Any] = []
                for variant in variants:
                    # Keep the expensive backbone forward pass in the same AMP
                    # mode as training, then promote logits before probability
                    # transforms.  This preserves numerically stable FP32
                    # softmax/sigmoid and calibration while removing avoidable
                    # full-FP32 validation and TTA work on tensor-core GPUs.
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        logits = model(variant)
                    logits = logits.float()
                    variant_logits.append(logits)
                    if mode in {"ordinal", "multiclass"}:
                        variant_values.append(torch.softmax(logits, dim=1))
                    else:
                        variant_values.append(torch.sigmoid(logits))
                values = torch.stack(variant_values).mean(dim=0)
                averaged_logits = torch.stack(variant_logits).mean(dim=0)
                predictions.append(values.float().cpu().numpy())
                raw_logits.append(averaged_logits.float().cpu().numpy())
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    truths.append(batch[1].numpy())
        return (
            np.concatenate(predictions),
            (np.concatenate(truths) if truths else None),
            np.concatenate(raw_logits),
        )

    def validation_score(probability: np.ndarray, truth: np.ndarray) -> tuple[float, list[float] | None]:
        if mode == "binary":
            if binary_metric not in {"roc_auc", "log_loss"}:
                raise ValueError(f"Unsupported binary vision metric: {binary_metric}")
            return compute_metric(binary_metric, truth, probability[:, 0]), None
        if mode == "multilabel":
            return compute_metric("mean_columnwise_roc_auc", truth, probability), None
        if mode == "multiclass":
            metric = (
                "mean_columnwise_roc_auc"
                if competition_id == "plant-pathology-2020-fgvc7"
                else "multiclass_log_loss"
            )
            if metric == "mean_columnwise_roc_auc":
                onehot = np.eye(output_count, dtype=np.float32)[truth.astype(int)]
                return compute_metric(metric, onehot, probability), None
            return compute_metric(metric, truth.astype(int), probability), None
        expected = probability @ np.arange(output_count, dtype=float)
        thresholds = optimize_ordinal_thresholds(expected, truth.astype(int))
        prediction = np.digitize(expected, thresholds)
        return compute_metric("quadratic_weighted_kappa", truth.astype(int), prediction), thresholds

    oof_probability = np.zeros((len(train_frame), output_count), dtype=np.float32)
    test_probability = np.zeros((len(sample), output_count), dtype=np.float32)
    oof_logits = np.zeros((len(train_frame), output_count), dtype=np.float32)
    test_logits = np.zeros((len(sample), output_count), dtype=np.float32)
    binary_test_logits_by_fold = (
        np.zeros((len(sample), effective_folds), dtype=np.float32)
        if mode == "binary" and binary_metric == "log_loss"
        else np.empty((len(sample), 0), dtype=np.float32)
    )
    fold_assignment = np.full(len(train_frame), -1, dtype=np.int16)
    histories: dict[str, list[dict[str, Any]]] = {}
    fold_scores: list[float] = []
    fold_checkpoint_selection_scores: list[float] = []
    fold_thresholds: list[list[float] | None] = []
    pretrained_flags: list[bool] = []
    pretrained_weight_identities: list[dict[str, Any]] = []
    checkpoint_hashes: list[str] = []
    peak_memory: list[int] = []
    fold_seeds: list[int] = []
    optimizer_contracts: list[dict[str, Any]] = []
    scheduler_contracts: list[dict[str, Any]] = []
    fold_best_top1_accuracy: list[float] = []
    fold_final_top1_accuracy: list[float] = []
    fold_final_ece: list[float] = []
    diagnostic_stop_reasons: list[str | None] = []
    started = time.perf_counter()
    test_loader = (
        None
        if diagnostic_mode
        else loader_for(
            None,
            train_mode=False,
            include_targets=False,
            loader_seed=args.seed + 90_000,
        )
    )
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    fast_kernel_mode = bool(args.wave2_fast_kernels)
    early_stopping_patience = max(1, int(getattr(args, "wave2_vision_patience", 2)))
    minimum_epochs_before_stop = max(1, min(int(early_stopping_min_epochs), int(epochs)))
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = fast_kernel_mode
    torch.backends.cudnn.deterministic = not fast_kernel_mode
    torch.use_deterministic_algorithms(not fast_kernel_mode, warn_only=True)
    if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    teacher_train_probability = np.empty((0, 0), dtype=np.float64)
    teacher_test_probability = np.empty((0, 0), dtype=np.float64)
    dog_breed_teacher_contract: dict[str, Any] | None = None
    dog_breed_classifier_source_indices: list[int] | None = None
    if competition_id == "dog-breed-identification":
        assert dog_breed_teacher_cache_path is not None
        assert dog_breed_teacher_source_manifest_sha256 is not None
        (
            teacher_train_probability,
            teacher_test_probability,
            dog_breed_teacher_contract,
        ) = load_or_compute_dog_breed_imagenet_teacher(
            train_paths=train_paths,
            test_paths=test_paths,
            classes=target_columns,
            source_manifest_sha256=dog_breed_teacher_source_manifest_sha256,
            cache_path=dog_breed_teacher_cache_path,
            workers=workers,
            batch_size=max(64, effective_batch * 4),
            logger=logger,
        )
        wave0.write_json(
            task_dir / "dog_breed_imagenet_teacher_contract.json",
            dog_breed_teacher_contract,
        )
        dog_breed_classifier_source_indices = [
            int(value)
            for value in dog_breed_teacher_contract["identity"]["category_indices"]
        ]
        if len(dog_breed_classifier_source_indices) != output_count:
            raise RuntimeError("Dog Breed ImageNet classifier initialization is incomplete")

    for fold, (training_indices, validation_indices) in enumerate(splits):
        fold_seed = args.seed + fold * 1_003
        fold_seeds.append(fold_seed)
        random.seed(fold_seed)
        torch.manual_seed(fold_seed)
        torch.cuda.manual_seed_all(fold_seed)
        np.random.seed(fold_seed)
        train_loader = loader_for(
            training_indices,
            train_mode=True,
            include_targets=True,
            loader_seed=fold_seed + 1,
        )
        valid_loader = loader_for(
            validation_indices,
            train_mode=False,
            include_targets=True,
            loader_seed=fold_seed + 2,
        )
        model, pretrained, pretrained_weight_identity = _vision_model(
            output_count,
            backbone=backbone,
            require_pretrained=True,
            classifier_source_indices=dog_breed_classifier_source_indices,
        )
        pretrained_flags.append(pretrained)
        pretrained_weight_identities.append(pretrained_weight_identity)
        model = model.to(device="cuda", memory_format=torch.channels_last)
        loss_fn = build_loss(training_indices)
        effective_learning_rate = args.wave2_learning_rate if learning_rate is None else learning_rate
        if competition_id == "dog-breed-identification":
            if dog_breed_training_mode == "frozen_backbone_head":
                optimizer_parameters, optimizer_contract = (
                    build_dog_breed_frozen_head_optimizer_groups(
                        model,
                        base_learning_rate=effective_learning_rate,
                        classifier_path=VISION_BACKBONE_SPECS[backbone]["classifier"],
                    )
                )
            else:
                optimizer_parameters, optimizer_contract = (
                    build_dog_breed_stability_optimizer_groups(
                        model,
                        base_learning_rate=effective_learning_rate,
                        classifier_path=VISION_BACKBONE_SPECS[backbone]["classifier"],
                    )
                )
            optimizer_contracts.append(optimizer_contract)
        else:
            optimizer_parameters = model.parameters()
            optimizer_contracts.append({
                "mode": "uniform",
                "learning_rate": effective_learning_rate,
            })
        try:
            optimizer = torch.optim.AdamW(
                optimizer_parameters,
                lr=effective_learning_rate,
                weight_decay=2e-4,
                fused=True,
            )
            fused_adamw = True
        except (RuntimeError, TypeError):
            optimizer = torch.optim.AdamW(
                optimizer_parameters, lr=effective_learning_rate, weight_decay=2e-4
            )
            fused_adamw = False
        scheduler_steps_per_update = competition_id == "dog-breed-identification"
        if scheduler_steps_per_update:
            scheduler, scheduler_contract = build_dog_breed_stability_scheduler(
                optimizer,
                total_steps=max(2, int(epochs) * len(train_loader)),
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, epochs),
            )
            scheduler_contract = {
                "schema": "evomind.vision.epoch_scheduler.v1",
                "mode": "cosine_annealing_per_epoch",
                "total_epochs": int(epochs),
                "step_unit": "epoch",
            }
        scheduler_contracts.append(scheduler_contract)
        use_scaler = amp_dtype == torch.float16
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
        minimize_metric = (
            competition_id == "dog-breed-identification"
            or (mode == "binary" and binary_metric == "log_loss")
        )
        best_score = math.inf if minimize_metric else -math.inf
        best_thresholds: list[float] | None = None
        best_path = task_dir / f"vision_fold{fold}_best.pt"
        best_state: dict[str, Any] | None = None
        best_top1_accuracy = 0.0
        history: list[dict[str, Any]] = []
        torch.cuda.reset_peak_memory_stats()
        checkpoint_selection_tta = bool(tta_flips)
        non_improving_epochs = 0
        stopped_early = False
        diagnostic_stop_reason: str | None = None
        previous_epoch_score: float | None = None
        for epoch in range(epochs):
            epoch_started = time.perf_counter()
            model.train()
            running = 0.0
            seen = 0
            for images, targets in train_loader:
                optimizer.zero_grad(set_to_none=True)
                images = images.cuda(
                    non_blocking=True,
                    memory_format=torch.channels_last,
                )
                targets = targets.cuda(non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(images)
                    loss = loss_fn(logits[:, 0], targets) if mode == "binary" else loss_fn(logits, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scheduler_steps_per_update:
                    scheduler.step()
                running += float(loss.detach()) * len(images)
                seen += len(images)
            training_seconds = time.perf_counter() - epoch_started
            if not scheduler_steps_per_update:
                scheduler.step()
            probability, truth, _ = infer(
                model,
                valid_loader,
                apply_tta=checkpoint_selection_tta,
            )
            assert truth is not None
            score, thresholds = validation_score(probability, truth)
            if not np.isfinite(score) or not np.isfinite(running):
                raise RuntimeError("Vision training produced a non-finite score or loss")
            top1_accuracy = (
                float((probability.argmax(axis=1) == truth.astype(int)).mean())
                if mode in {"multiclass", "ordinal"}
                else math.nan
            )
            improved = score < best_score if minimize_metric else score > best_score
            if improved:
                best_score, best_thresholds = score, thresholds
                best_top1_accuracy = top1_accuracy
                non_improving_epochs = 0
                # Keep the best weights in host RAM during training and write
                # one checkpoint per fold. Repeated large GPFS writes on every
                # improving epoch were a measurable avoidable stall.
                best_state = {
                    name: value.detach().to(device="cpu", copy=True)
                    for name, value in model.state_dict().items()
                }
            else:
                non_improving_epochs += 1
            epoch_seconds = time.perf_counter() - epoch_started
            history.append({
                "epoch": epoch + 1,
                "train_loss": running / max(1, seen),
                "cv_score": float(score),
                "epoch_seconds": epoch_seconds,
                "training_seconds": training_seconds,
                "train_images_per_second": seen / max(training_seconds, 1e-9),
                "non_improving_epochs": non_improving_epochs,
                "top1_accuracy": top1_accuracy,
                "learning_rates": {
                    str(group.get("group_name", index)): float(group["lr"])
                    for index, group in enumerate(optimizer.param_groups)
                },
            })
            logger.info(
                "[%s] fold=%d/%d epoch=%d cv=%.6f images_per_second=%.1f",
                competition_id,
                fold + 1,
                effective_folds,
                epoch + 1,
                score,
                seen / max(training_seconds, 1e-9),
            )
            if diagnostic_mode:
                if epoch == 0:
                    relative_improvement = (parent_epoch_one - score) / parent_epoch_one
                    history[-1]["parent_relative_improvement"] = relative_improvement
                    if score > DOG_BREED_STAB_EPOCH_ONE_MAX_LOG_LOSS:
                        diagnostic_stop_reason = "epoch_one_log_loss_above_0_090"
                    elif relative_improvement < DOG_BREED_STAB_MIN_RELATIVE_EPOCH_ONE_IMPROVEMENT:
                        diagnostic_stop_reason = "epoch_one_relative_improvement_below_25pct"
                elif previous_epoch_score is not None:
                    epoch_improvement = previous_epoch_score - score
                    history[-1]["previous_epoch_improvement"] = epoch_improvement
                    if epoch_improvement < DOG_BREED_STAB_MIN_EPOCH_IMPROVEMENT:
                        diagnostic_stop_reason = "per_epoch_improvement_below_0_002"
                if time.perf_counter() - started >= DOG_BREED_STAB_MAX_WALL_SECONDS:
                    diagnostic_stop_reason = "wall_clock_limit_90_minutes"
                previous_epoch_score = float(score)
                if diagnostic_stop_reason is not None:
                    stopped_early = True
                    logger.info(
                        "[%s] %s stop fold=%d epoch=%d reason=%s",
                        competition_id,
                        (
                            "DB-FROZEN-HEAD1"
                            if dog_breed_training_mode == "frozen_backbone_head"
                            else "DB-STAB1"
                        ),
                        fold,
                        epoch + 1,
                        diagnostic_stop_reason,
                    )
                    break
            if (
                epoch + 1 >= minimum_epochs_before_stop
                and non_improving_epochs >= early_stopping_patience
            ):
                stopped_early = True
                logger.info(
                    "[%s] fold=%d/%d early_stop epoch=%d patience=%d",
                    competition_id,
                    fold + 1,
                    effective_folds,
                    epoch + 1,
                    early_stopping_patience,
                )
                break
        if best_state is None:
            raise RuntimeError(f"No best checkpoint was selected for vision fold {fold}")
        torch.save(best_state, best_path)
        model.load_state_dict(best_state)
        del best_state
        checkpoint_hashes.append(_sha256_file(best_path))
        fold_valid, truth, fold_valid_logits = infer(model, valid_loader, apply_tta=tta_flips)
        if diagnostic_mode:
            fold_test = np.empty((0, output_count), dtype=np.float32)
            fold_test_logits = np.empty((0, output_count), dtype=np.float32)
        else:
            assert test_loader is not None
            fold_test, _, fold_test_logits = infer(model, test_loader, apply_tta=tta_flips)
        assert truth is not None
        fold_final_score, _ = validation_score(fold_valid, truth)
        oof_probability[validation_indices] = fold_valid
        if not diagnostic_mode:
            test_probability += fold_test / effective_folds
        oof_logits[validation_indices] = fold_valid_logits
        if not diagnostic_mode:
            test_logits += fold_test_logits / effective_folds
        if binary_test_logits_by_fold.shape[1]:
            binary_test_logits_by_fold[:, fold] = fold_test_logits[:, 0]
        fold_assignment[validation_indices] = fold
        histories[str(fold)] = history
        histories[str(fold)][-1]["stopped_early"] = stopped_early
        fold_scores.append(float(fold_final_score))
        fold_best_top1_accuracy.append(float(best_top1_accuracy))
        final_top1 = float((fold_valid.argmax(axis=1) == truth.astype(int)).mean())
        fold_final_top1_accuracy.append(final_top1)
        fold_final_ece.append(
            _multiclass_expected_calibration_error(truth, fold_valid)
            if mode in {"multiclass", "ordinal"}
            else math.nan
        )
        diagnostic_stop_reasons.append(diagnostic_stop_reason)
        fold_checkpoint_selection_scores.append(float(best_score))
        fold_thresholds.append(best_thresholds)
        peak_memory.append(int(torch.cuda.max_memory_allocated() / 2**20))
        del model, optimizer, scheduler, scaler, train_loader, valid_loader
        torch.cuda.empty_cache()

    if diagnostic_mode:
        best_log_loss = min(fold_scores)
        best_top1 = max(fold_best_top1_accuracy)
        continuation_allowed = bool(
            best_log_loss <= DOG_BREED_STAB_CONTINUATION_MAX_LOG_LOSS
            and best_top1 >= parent_top1 + DOG_BREED_STAB_MIN_TOP1_IMPROVEMENT
            and all(pretrained_flags)
            and all(
                contract["groups_disjoint"]
                and contract["groups_cover_all_trainable_parameters"]
                for contract in optimizer_contracts
            )
            and not any(group_overlap_by_fold.values())
        )
        diagnostic_id = (
            "DB-FROZEN-HEAD1"
            if dog_breed_training_mode == "frozen_backbone_head"
            else "DB-STAB1"
        )
        diagnostic_report = {
            "schema": (
                "evomind.dog_breed.frozen_backbone_head_diagnostic.v1"
                if dog_breed_training_mode == "frozen_backbone_head"
                else "evomind.dog_breed.db_stab1.v1"
            ),
            "competition_id": competition_id,
            "status": "diagnostic_passed" if continuation_allowed else "diagnostic_failed",
            "diagnostic_id": diagnostic_id,
            "training_mode": dog_breed_training_mode,
            "fixed_fold": 0,
            "planned_fold_count": planned_folds,
            "executed_fold_count": effective_folds,
            "epochs_cap": int(epochs),
            "best_fold0_log_loss": best_log_loss,
            "best_fold0_top1_accuracy": best_top1,
            "final_fold0_top1_accuracy": fold_final_top1_accuracy[0],
            "final_fold0_expected_calibration_error": fold_final_ece[0],
            "parent_fold0_epoch1_log_loss": parent_epoch_one,
            "parent_fold0_top1_accuracy": parent_top1,
            "continuation_max_log_loss": DOG_BREED_STAB_CONTINUATION_MAX_LOG_LOSS,
            "minimum_top1_improvement": DOG_BREED_STAB_MIN_TOP1_IMPROVEMENT,
            "continuation_allowed": continuation_allowed,
            "fold_histories": histories,
            "fold_checkpoint_sha256": checkpoint_hashes,
            "pretrained_weight_identities": pretrained_weight_identities,
            "optimizer_contracts": optimizer_contracts,
            "scheduler_contracts": scheduler_contracts,
            "group_overlap_by_fold": group_overlap_by_fold,
            "diagnostic_stop_reasons": diagnostic_stop_reasons,
            "runtime_seconds_model": time.perf_counter() - started,
            "peak_gpu_memory_mib": peak_memory,
            "valid_submission": False,
            "submission_created": False,
            "official_grader_executed": False,
            "official_grader_withheld": True,
            "kaggle_submission_enabled": False,
            "private_labels_used": False,
            "human_gate_preserved": True,
            "claim_boundary": (
                "Fixed public fold-zero diagnostic only; no submission and no official medal claim."
            ),
        }
        diagnostic_name = (
            "dog_breed_frozen_backbone_head_diagnostic.json"
            if dog_breed_training_mode == "frozen_backbone_head"
            else "dog_breed_db_stab1_diagnostic.json"
        )
        wave0.write_json(task_dir / diagnostic_name, diagnostic_report)
        return diagnostic_report

    if np.any(fold_assignment < 0):
        raise RuntimeError("Vision OOF coverage is incomplete")
    selected_oof_probability = np.asarray(oof_probability, dtype=np.float64)
    selected_test_probability = np.asarray(test_probability, dtype=np.float64)
    selected_probability_mode = "fold_probability_average"
    calibration_report: dict[str, Any] | None = None
    if mode == "binary" and binary_metric == "log_loss":
        raw_logit_oof = _sigmoid_array(oof_logits[:, 0])
        raw_logit_test = _sigmoid_array(test_logits[:, 0])
        calibrated_oof, calibrated_test, calibration_folds, final_calibration = (
            cross_fit_binary_logloss_calibration(
                oof_logits[:, 0],
                binary_test_logits_by_fold,
                target,
                fold_assignment,
            )
        )
        deployment_candidates = {
            "fold_probability_average": (
                _binary_log_loss_array(target, oof_probability[:, 0]),
                np.asarray(oof_probability[:, 0], dtype=np.float64),
                np.asarray(test_probability[:, 0], dtype=np.float64),
            ),
            "fold_logit_average": (
                _binary_log_loss_array(target, raw_logit_oof),
                raw_logit_oof,
                raw_logit_test,
            ),
            "cross_fitted_temperature_intercept": (
                _binary_log_loss_array(target, calibrated_oof),
                calibrated_oof,
                calibrated_test,
            ),
        }
        nested_selected_oof = np.full(len(target), np.nan, dtype=np.float64)
        nested_selection_records: list[dict[str, Any]] = []
        for fold in sorted(int(value) for value in np.unique(fold_assignment)):
            validation = fold_assignment == fold
            fitting = ~validation
            nested_candidates: dict[str, tuple[float, np.ndarray]] = {
                "fold_probability_average": (
                    _binary_log_loss_array(target[fitting], oof_probability[fitting, 0]),
                    np.asarray(oof_probability[validation, 0], dtype=np.float64),
                ),
                "fold_logit_average": (
                    _binary_log_loss_array(target[fitting], raw_logit_oof[fitting]),
                    raw_logit_oof[validation],
                ),
            }
            fitting_folds = fold_assignment[fitting]
            if len(np.unique(fitting_folds)) >= 2:
                inner_calibrated_oof, outer_calibrated, _, _ = (
                    cross_fit_binary_logloss_calibration(
                        oof_logits[fitting, 0],
                        oof_logits[validation, 0],
                        target[fitting],
                        fitting_folds,
                    )
                )
                nested_candidates["cross_fitted_temperature_intercept"] = (
                    _binary_log_loss_array(target[fitting], inner_calibrated_oof),
                    outer_calibrated,
                )
            selected_mode = min(
                nested_candidates,
                key=lambda name: nested_candidates[name][0],
            )
            nested_selected_oof[validation] = nested_candidates[selected_mode][1]
            nested_selection_records.append({
                "outer_fold": fold,
                "candidate_fit_log_loss": {
                    name: float(values[0]) for name, values in nested_candidates.items()
                },
                "selected_mode": selected_mode,
            })
        if not np.isfinite(nested_selected_oof).all():
            raise RuntimeError("Nested binary probability-mode selection is incomplete")
        selected_probability_mode = min(
            deployment_candidates,
            key=lambda name: deployment_candidates[name][0],
        )
        selected_oof_probability = nested_selected_oof.reshape(-1, 1)
        selected_test_probability = deployment_candidates[selected_probability_mode][2].reshape(-1, 1)
        calibration_report = {
            "selection_metric": "nested_cross_fitted_oof_log_loss",
            "candidate_oof_log_loss": {
                name: float(values[0]) for name, values in deployment_candidates.items()
            },
            "nested_selected_oof_log_loss": _binary_log_loss_array(
                target, nested_selected_oof
            ),
            "nested_selection_records": nested_selection_records,
            "deployment_probability_mode": selected_probability_mode,
            "fold_calibrations": calibration_folds,
            "final_calibration": final_calibration,
        }
    dog_breed_blend_report: dict[str, Any] | None = None
    dog_breed_class_bias_report: dict[str, Any] | None = None
    dog_breed_class_bias_crossfit_probability = np.empty((0, 0), dtype=np.float64)
    if competition_id == "dog-breed-identification":
        if (
            teacher_train_probability.shape != oof_probability.shape
            or teacher_test_probability.shape != test_probability.shape
            or dog_breed_teacher_contract is None
        ):
            raise RuntimeError("Dog Breed teacher probabilities are incomplete")
        cross_fitted_blend, cross_fitted_records = cross_fit_dog_breed_probability_blend(
            oof_probability,
            teacher_train_probability,
            target,
            fold_assignment,
        )
        _, final_blend_parameters = select_dog_breed_probability_blend(
            oof_probability,
            teacher_train_probability,
            target,
        )
        fine_tuned_log_loss = _indexed_multiclass_log_loss(target, oof_probability)
        teacher_log_loss = _indexed_multiclass_log_loss(target, teacher_train_probability)
        cross_fitted_blend_log_loss = _indexed_multiclass_log_loss(target, cross_fitted_blend)
        minimum_improvement = float(
            promotion_contract.get("minimum_teacher_blend_improvement", 0.0005)
        )
        incumbent_before_blend = selected_oof_probability.copy()
        selected_blend_oof, blend_replacement = select_dog_breed_probability_replacement(
            incumbent_before_blend,
            cross_fitted_blend,
            target,
            fold_assignment,
            candidate_complete=True,
            minimum_improvement=minimum_improvement,
        )
        blend_selected = bool(blend_replacement["replacement_selected"])
        if blend_selected:
            selected_oof_probability = selected_blend_oof
            selected_test_probability = apply_dog_breed_probability_blend(
                test_probability,
                teacher_test_probability,
                fine_tuned_weight=final_blend_parameters["fine_tuned_weight"],
                temperature=final_blend_parameters["temperature"],
                blend_family=str(final_blend_parameters["blend_family"]),
            )
            selected_probability_mode = "cross_fitted_fine_tuned_imagenet_teacher_blend"
        dog_breed_blend_report = {
            "schema": "evomind.dog_breed.imagenet_teacher_blend.v2",
            "selection_metric": "nested_cross_fitted_multiclass_log_loss",
            "calibration_search": "joint_family_weight_temperature_coarse_to_fine_v1",
            "fine_tuned_oof_log_loss": fine_tuned_log_loss,
            "teacher_oof_log_loss": teacher_log_loss,
            "cross_fitted_blend_oof_log_loss": cross_fitted_blend_log_loss,
            "incumbent_oof_log_loss": blend_replacement["incumbent_oof_log_loss"],
            "cross_fitted_blend_improvement": blend_replacement["aggregate_improvement"],
            "minimum_blend_improvement": minimum_improvement,
            "blend_selected": blend_selected,
            "incumbent_safe_replacement": blend_replacement,
            "cross_fitted_records": cross_fitted_records,
            "final_all_oof_parameters_for_test_only": final_blend_parameters,
            "teacher_contract": dog_breed_teacher_contract,
            "private_labels_used": False,
        }
        wave0.write_json(
            task_dir / "dog_breed_imagenet_teacher_blend.json",
            dog_breed_blend_report,
        )
        incumbent_before_class_bias = selected_oof_probability.copy()
        (
            dog_breed_class_bias_crossfit_probability,
            class_bias_records,
            class_bias_crossfit,
        ) = cross_fit_dog_breed_class_bias_calibration(
            incumbent_before_class_bias,
            target,
            fold_assignment,
        )
        selected_class_bias_oof, class_bias_replacement = (
            select_dog_breed_probability_replacement(
                incumbent_before_class_bias,
                dog_breed_class_bias_crossfit_probability,
                target,
                fold_assignment,
                candidate_complete=bool(class_bias_crossfit["candidate_complete"]),
                minimum_improvement=minimum_improvement,
            )
        )
        class_bias_selected = bool(class_bias_replacement["replacement_selected"])
        if class_bias_selected:
            final_deployment = class_bias_crossfit.get("final_deployment") or {}
            final_class_bias = np.asarray(
                final_deployment.get("class_bias", []), dtype=np.float64
            )
            if final_class_bias.shape != (selected_test_probability.shape[1],):
                raise RuntimeError("Dog Breed final class-bias parameters are incomplete")
            selected_oof_probability = selected_class_bias_oof
            selected_test_probability = apply_dog_breed_class_bias_calibration(
                selected_test_probability,
                final_class_bias,
            )
            selected_probability_mode += "_plus_nested_class_bias"
        dog_breed_class_bias_report = {
            "schema": "evomind.dog_breed.class_bias_calibration.v1",
            "selection_metric": "nested_cross_fitted_multiclass_log_loss",
            "fixed_l2_grid": list(DOG_BREED_CLASS_BIAS_L2_GRID),
            "cross_fitted_records": class_bias_records,
            "crossfit_contract": class_bias_crossfit,
            "incumbent_safe_replacement": class_bias_replacement,
            "class_bias_selected": class_bias_selected,
            "private_labels_used": False,
        }
        wave0.write_json(
            task_dir / "dog_breed_class_bias_calibration.json",
            dog_breed_class_bias_report,
        )
    promotion_fold_scores = list(fold_scores)
    if dog_breed_blend_report is not None and dog_breed_blend_report["blend_selected"]:
        promotion_fold_scores = [
            float(record["outer_log_loss"])
            for record in dog_breed_blend_report["cross_fitted_records"]
        ]
    if (
        dog_breed_class_bias_report is not None
        and dog_breed_class_bias_report["class_bias_selected"]
    ):
        promotion_fold_scores = [
            float(record["candidate_log_loss"])
            for record in dog_breed_class_bias_report["incumbent_safe_replacement"][
                "fold_records"
            ]
        ]
    promotion_extra_checks: dict[str, bool] = {
        "pretrained_weights_loaded_all_folds": bool(all(pretrained_flags)),
    }
    if competition_id == "dog-breed-identification":
        promotion_extra_checks["imagenet_classifier_rows_initialized_all_folds"] = bool(
            pretrained_weight_identities
            and all(
                identity.get("classifier_initialization", {}).get("mode")
                == "exact_pretrained_imagenet_classifier_rows"
                and identity.get("classifier_initialization", {}).get("weights_copied") is True
                for identity in pretrained_weight_identities
            )
        )
        promotion_extra_checks["dog_breed_optimizer_contract_all_folds"] = bool(
            len(optimizer_contracts) == effective_folds
            and all(
                contract.get("mode")
                in {
                    "pretrained_backbone_0.1x_classifier_1.0x",
                    "classifier_linear_1.0x_pretrained_and_classifier_norm_0.1x",
                    "frozen_backbone_imagenet_head_only",
                }
                for contract in optimizer_contracts
            )
        )
    promotion_evidence: dict[str, Any] = {
        "split_strategy": split_strategy,
        "group_overlap_by_fold": group_overlap_by_fold,
        "pretrained_weight_identities": pretrained_weight_identities,
    }
    ordinal_crossfit_records: list[dict[str, Any]] = []
    if mode == "ordinal":
        expected = selected_oof_probability @ np.arange(output_count, dtype=float)
        cross_fitted_prediction, ordinal_crossfit_records = cross_fit_ordinal_thresholds(
            expected,
            target,
            fold_assignment,
        )
        cv_score = compute_metric(
            "quadratic_weighted_kappa",
            target.astype(int),
            cross_fitted_prediction,
        )
        final_thresholds = optimize_ordinal_thresholds(expected, target.astype(int))
        promotion_fold_scores = [
            float(record["fold_score"]) for record in ordinal_crossfit_records
        ]
        class_coverage = {
            str(fold): sorted(np.unique(target[fold_assignment == fold]).astype(int).tolist())
            for fold in sorted(np.unique(fold_assignment).tolist())
        }
        required_classes = list(range(output_count))
        promotion_extra_checks["all_classes_present_in_every_fold"] = all(
            values == required_classes for values in class_coverage.values()
        )
        promotion_evidence.update({
            "threshold_protocol": "fit_on_other_folds_score_held_out_fold",
            "cross_fitted_threshold_records": ordinal_crossfit_records,
            "class_coverage_by_fold": class_coverage,
            "final_all_oof_thresholds_for_test_only": final_thresholds,
        })
    else:
        cv_score, final_thresholds = validation_score(selected_oof_probability, target)

    if competition_id == "dog-breed-identification":
        promotion_extra_checks["oof_probability_rows_sum_to_one"] = bool(
            np.allclose(selected_oof_probability.sum(axis=1), 1.0, atol=1e-6)
        )
        promotion_extra_checks["test_probability_rows_sum_to_one"] = bool(
            np.allclose(selected_test_probability.sum(axis=1), 1.0, atol=1e-6)
        )
        duplicate_contract = group_contract or {}
        promotion_extra_checks.update({
            "exact_image_hash_grouped_split": bool(
                groups is not None and "group" in split_strategy
            ),
            "duplicate_group_disjoint": bool(
                groups is not None and not any(group_overlap_by_fold.values())
            ),
            "conflicting_duplicate_hashes_quarantined": bool(
                duplicate_contract.get("conflicting_hash_policy")
                == "group_together_no_test_override"
                and duplicate_contract.get("conflicting_test_overrides") == 0
            ),
            "imagenet_teacher_mapping_complete": bool(
                dog_breed_teacher_contract is not None
                and dog_breed_teacher_contract.get("mapping_complete") is True
            ),
            "imagenet_teacher_probability_contract": bool(
                teacher_train_probability.shape == oof_probability.shape
                and teacher_test_probability.shape == test_probability.shape
                and np.allclose(teacher_train_probability.sum(axis=1), 1.0, atol=1e-6)
                and np.allclose(teacher_test_probability.sum(axis=1), 1.0, atol=1e-6)
            ),
            "teacher_blend_meta_crossfit_complete": bool(
                dog_breed_blend_report is not None
                and len(dog_breed_blend_report["cross_fitted_records"]) == effective_folds
            ),
            "private_labels_excluded_from_teacher": bool(
                dog_breed_blend_report is not None
                and dog_breed_blend_report.get("private_labels_used") is False
            ),
            "class_bias_incumbent_safe_replacement": bool(
                dog_breed_class_bias_report is not None
                and (
                    dog_breed_class_bias_report["class_bias_selected"]
                    or dog_breed_class_bias_report["incumbent_safe_replacement"].get(
                        "fallback_preserved_incumbent_exactly"
                    )
                    is True
                )
            ),
            "class_bias_meta_crossfit_complete_when_selected": bool(
                dog_breed_class_bias_report is not None
                and (
                    not dog_breed_class_bias_report["class_bias_selected"]
                    or dog_breed_class_bias_report["crossfit_contract"].get(
                        "candidate_complete"
                    )
                    is True
                )
            ),
            "private_labels_excluded_from_class_bias": bool(
                dog_breed_class_bias_report is not None
                and dog_breed_class_bias_report.get("private_labels_used") is False
            ),
        })
        promotion_evidence.update({
            "duplicate_contract": duplicate_contract,
            "imagenet_teacher_blend": dog_breed_blend_report,
            "class_bias_calibration": dog_breed_class_bias_report,
        })
    elif competition_id == "plant-pathology-2020-fgvc7":
        onehot = np.eye(output_count, dtype=np.float32)[target.astype(int)]
        class_scores = {
            str(target_columns[index]): float(
                compute_metric("roc_auc", onehot[:, index], selected_oof_probability[:, index])
            )
            for index in range(output_count)
        }
        minimum_class_score = float(promotion_contract["minimum_class_score"])
        promotion_extra_checks["every_class_auc_threshold"] = bool(
            len(class_scores) == 4
            and min(class_scores.values()) >= minimum_class_score
        )
        promotion_evidence.update({
            "class_auc": class_scores,
            "minimum_class_score": minimum_class_score,
        })
    elif competition_id == "ranzcr-clip-catheter-line-classification":
        label_scoreable_by_fold = {
            str(fold): bool(
                all(
                    np.unique(target[fold_assignment == fold, index]).size == 2
                    for index in range(output_count)
                )
            )
            for fold in sorted(np.unique(fold_assignment).tolist())
        }
        promotion_extra_checks["patient_grouped_split"] = bool(
            groups is not None and "group" in split_strategy
        )
        promotion_extra_checks["every_label_scoreable_in_every_fold"] = bool(
            all(label_scoreable_by_fold.values())
        )
        promotion_evidence["label_scoreable_by_fold"] = label_scoreable_by_fold
    elif competition_id == "histopathologic-cancer-detection":
        duplicate_contract = group_contract or {}
        promotion_extra_checks.update(
            histopath_duplicate_promotion_checks(
                groups=groups,
                split_strategy=split_strategy,
                group_overlap_by_fold=group_overlap_by_fold,
                duplicate_contract=duplicate_contract,
            )
        )
        promotion_evidence["duplicate_contract"] = duplicate_contract

    promotion_gate = (
        evaluate_vision_promotion_gate(
            promotion_contract,
            metric=binary_metric if mode == "binary" else str(promotion_contract["metric"]),
            cv_score=float(cv_score),
            fold_scores=promotion_fold_scores,
            truth=target,
            oof_probability=selected_oof_probability[:, 0],
            extra_checks=promotion_extra_checks,
            evidence=promotion_evidence,
        )
        if promotion_contract is not None
        else None
    )
    override_count = 0
    if test_probability_override is not None:
        override = np.asarray(test_probability_override, dtype=np.float64)
        if mode == "binary":
            override = override.reshape(-1)
            if len(override) != len(sample):
                raise RuntimeError("Vision binary test override has invalid cardinality")
            mask = np.isfinite(override)
            if np.any((override[mask] < 0.0) | (override[mask] > 1.0)):
                raise RuntimeError("Vision binary test override is outside the probability range")
            selected_test_probability[mask, 0] = np.clip(
                override[mask], 1e-6, 1.0 - 1e-6
            )
            override_count = int(mask.sum())
        elif mode == "multiclass":
            if override.shape != (len(sample), output_count):
                raise RuntimeError("Vision multiclass test override has invalid shape")
            finite_by_row = np.isfinite(override)
            if np.any(finite_by_row.any(axis=1) != finite_by_row.all(axis=1)):
                raise RuntimeError("Vision multiclass test override contains partial rows")
            mask = finite_by_row.all(axis=1)
            selected_rows = override[mask]
            if (
                np.any((selected_rows < 0.0) | (selected_rows > 1.0))
                or not np.allclose(selected_rows.sum(axis=1), 1.0, atol=1e-6)
            ):
                raise RuntimeError("Vision multiclass test override is not a probability matrix")
            selected_test_probability[mask] = selected_rows
            override_count = int(mask.sum())
        else:
            raise RuntimeError("Vision test overrides are supported only for binary/multiclass tasks")
    if mode == "binary":
        sample[target_columns[0]] = _finite_probability(selected_test_probability[:, 0])
    elif mode == "multilabel":
        sample[target_columns] = _finite_probability(test_probability)
    elif mode == "multiclass":
        normalized = selected_test_probability / np.maximum(
            selected_test_probability.sum(axis=1, keepdims=True), 1e-12
        )
        sample[target_columns] = normalized
    else:
        expected = test_probability @ np.arange(output_count, dtype=float)
        sample[target_columns[0]] = np.digitize(
            expected, final_thresholds or [0.5, 1.5, 2.5, 3.5]
        )
    np.savez_compressed(
        task_dir / "vision_oof_and_test.npz",
        train_id=np.asarray([path.stem for path in train_paths], dtype=np.str_),
        test_id=np.asarray([path.stem for path in test_paths], dtype=np.str_),
        class_names=np.asarray(target_columns, dtype=np.str_),
        truth=target,
        oof_probability=oof_probability,
        test_probability=test_probability,
        oof_logits=oof_logits,
        test_logits=test_logits,
        binary_test_logits_by_fold=binary_test_logits_by_fold,
        selected_oof_probability=selected_oof_probability,
        selected_test_probability=selected_test_probability,
        dog_breed_teacher_train_probability=teacher_train_probability,
        dog_breed_teacher_test_probability=teacher_test_probability,
        dog_breed_class_bias_crossfit_probability=dog_breed_class_bias_crossfit_probability,
        fold=fold_assignment,
    )
    budget = {
        "seed": args.seed,
        "folds": effective_folds,
        "split_strategy": split_strategy,
        "unique_groups": None if groups is None else int(len(np.unique(groups))),
        "train_images": len(train_frame),
        "test_images": len(sample),
        "epochs_per_fold": epochs,
        "epochs_completed_by_fold": {
            fold: len(history) for fold, history in histories.items()
        },
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_epochs": minimum_epochs_before_stop,
        "preprocessing_profile": preprocessing_profile,
        "image_size": effective_size,
        "batch_size": effective_batch,
        "backbone": backbone,
        "dog_breed_training_mode": (
            dog_breed_training_mode
            if competition_id == "dog-breed-identification"
            else None
        ),
        "pretrained_weights_loaded_all_folds": bool(all(pretrained_flags)),
        "pretrained_weight_identities": pretrained_weight_identities,
        "checkpoint_sha256_by_fold": checkpoint_hashes,
        "deterministic_algorithms_requested": not fast_kernel_mode,
        "a40_fast_kernel_mode": fast_kernel_mode,
        "tta_flips": tta_flips,
        "vertical_flip": vertical_flip,
        "amp_dtype": str(amp_dtype),
        "validation_metric": binary_metric if mode == "binary" else None,
        "selected_probability_mode": selected_probability_mode,
        "cross_fitted_calibration": calibration_report is not None,
        "dog_breed_imagenet_teacher": dog_breed_teacher_contract,
        "dog_breed_teacher_blend": dog_breed_blend_report,
        "dog_breed_class_bias_calibration": dog_breed_class_bias_report,
        "optimizer_contracts": optimizer_contracts,
        "learning_rate": args.wave2_learning_rate if learning_rate is None else learning_rate,
        "workers": workers,
        "exact_test_probability_overrides": override_count,
        "group_contract": group_contract,
        "dataloader_prefetch_factor": (
            VISION_DATALOADER_PREFETCH_FACTOR if workers > 0 else None
        ),
        "channels_last": True,
        "fused_adamw": fused_adamw,
        "tf32_allowed": True,
        "bf16_reduced_precision_reduction_allowed": True,
        "epoch_selection_tta": bool(tta_flips),
        "final_oof_test_tta": tta_flips,
        "fold_seeds": fold_seeds,
        "promotion_gate": promotion_gate,
    }
    model_family = f"{backbone}_ImageNet1K_fold_ensemble"
    if (
        competition_id == "dog-breed-identification"
        and dog_breed_training_mode == "frozen_backbone_head"
    ):
        model_family += "_frozen_backbone_mapped_head"
    if dog_breed_blend_report is not None and dog_breed_blend_report["blend_selected"]:
        model_family += "_plus_frozen_ImageNet120_teacher"
    if (
        dog_breed_class_bias_report is not None
        and dog_breed_class_bias_report["class_bias_selected"]
    ):
        model_family += "_plus_nested_class_bias"
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
            "model_family": model_family,
            "fold_scores": fold_scores,
            "fold_checkpoint_selection_scores": fold_checkpoint_selection_scores,
            "fold_histories": histories,
            "fold_ordinal_thresholds": fold_thresholds,
            "ordinal_thresholds": final_thresholds,
            "ordinal_cross_fitted_threshold_records": ordinal_crossfit_records,
            "torch_peak_memory_allocated_mib_by_fold": peak_memory,
            "binary_calibration": calibration_report,
            "dog_breed_imagenet_teacher_blend": dog_breed_blend_report,
            "dog_breed_class_bias_calibration": dog_breed_class_bias_report,
            "budget": budget,
        },
    )

def run_aptos(args: argparse.Namespace, task_dir: Path, logger: Any) -> dict[str, Any]:
    cid = "aptos2019-blindness-detection"
    resolved = resolve_competition(cid, args.data_root)
    train = pd.read_csv(resolved.public_dir / "train.csv")
    sample = pd.read_csv(resolved.sample_submission_path)
    return _run_vision(
        args=args, task_dir=task_dir, logger=logger, competition_id=cid,
        train_frame=train, sample=sample,
        train_paths=[resolved.public_dir / "train_images" / f"{value}.png" for value in train["id_code"]],
        test_paths=[resolved.public_dir / "test_images" / f"{value}.png" for value in sample["id_code"]],
        target_columns=["diagnosis"], mode="ordinal", epochs=args.wave2_aptos_epochs,
        backbone="efficientnet_v2_s",
        image_size=512, batch_size=24, vertical_flip=False, tta_flips=True,
        preprocessing_profile="retina", early_stopping_min_epochs=4,
        fold_count=args.wave2_vision_folds,
        promotion_contract=WAVE2_PROMOTION_CONTRACTS[cid],
    )


def run_dog_breed(args: argparse.Namespace, task_dir: Path, logger: Any) -> dict[str, Any]:
    cid = "dog-breed-identification"
    resolved = resolve_competition(cid, args.data_root)
    train = pd.read_csv(resolved.public_dir / "labels.csv")
    sample = pd.read_csv(resolved.sample_submission_path)
    classes = list(sample.columns[1:])
    train_paths = [resolved.public_dir / "train" / f"{value}.jpg" for value in train["id"]]
    test_paths = [resolved.public_dir / "test" / f"{value}.jpg" for value in sample["id"]]
    train_hashes, test_overrides, duplicate_contract = (
        build_multiclass_image_duplicate_contract(
            train_paths,
            test_paths,
            train["breed"].astype(str).to_numpy(),
            classes,
            workers=args.wave2_workers,
            cache_path=wave0.ensure_within(
                args.output_root
                / "_shared_wave2_cache"
                / cid
                / "exact_image_sha256_groups_v1.npz",
                args.allowed_root,
            ),
        )
    )
    # The generic multiclass adapter expects the target values to equal sample class names.
    return _run_vision(
        args=args, task_dir=task_dir, logger=logger, competition_id=cid,
        train_frame=train, sample=sample,
        train_paths=train_paths,
        test_paths=test_paths,
        target_columns=classes, mode="multiclass", epochs=args.wave2_dog_breed_epochs,
        backbone=getattr(args, "wave2_dog_breed_backbone", "convnext_small"),
        label_column="breed",
        image_size=384, batch_size=args.wave2_dog_breed_batch_size,
        vertical_flip=False, tta_flips=True,
        fold_count=args.wave2_vision_folds,
        group_values=train_hashes,
        group_contract=duplicate_contract,
        test_probability_override=test_overrides,
        dog_breed_teacher_cache_path=wave0.ensure_within(
            args.output_root
            / "_shared_wave2_cache"
            / cid
            / "convnext_small_imagenet1k_teacher_v1.npz",
            args.allowed_root,
        ),
        dog_breed_teacher_source_manifest_sha256=str(
            duplicate_contract["source_manifest_sha256"]
        ),
        diagnostic_fold_limit=int(
            getattr(args, "wave2_dog_breed_diagnostic_fold_limit", 0)
        ),
        diagnostic_parent_fold0_epoch1_log_loss=getattr(
            args,
            "wave2_dog_breed_diagnostic_parent_fold0_epoch1_log_loss",
            None,
        ),
        diagnostic_parent_fold0_top1_accuracy=getattr(
            args,
            "wave2_dog_breed_diagnostic_parent_fold0_top1_accuracy",
            None,
        ),
        dog_breed_training_mode=getattr(
            args,
            "wave2_dog_breed_training_mode",
            "stability_finetune",
        ),
        learning_rate=(
            getattr(args, "wave2_dog_breed_head_learning_rate", 1e-3)
            if getattr(
                args,
                "wave2_dog_breed_training_mode",
                "stability_finetune",
            )
            == "frozen_backbone_head"
            else args.wave2_learning_rate
        ),
        promotion_contract=WAVE2_PROMOTION_CONTRACTS[cid],
    )


def run_histopath(args: argparse.Namespace, task_dir: Path, logger: Any) -> dict[str, Any]:
    cid = "histopathologic-cancer-detection"
    resolved = resolve_competition(cid, args.data_root)
    train = pd.read_csv(resolved.public_dir / "train_labels.csv")
    if args.wave2_histopath_max_rows and len(train) > args.wave2_histopath_max_rows:
        train = train.groupby("label", group_keys=False).sample(
            n=max(1, args.wave2_histopath_max_rows // train["label"].nunique()),
            random_state=args.seed,
        ).head(args.wave2_histopath_max_rows).reset_index(drop=True)
    sample = pd.read_csv(resolved.sample_submission_path)
    train_paths = [resolved.public_dir / "train" / f"{value}.tif" for value in train["id"]]
    test_paths = [resolved.public_dir / "test" / f"{value}.tif" for value in sample["id"]]
    train_hashes, test_overrides, duplicate_contract = build_histopath_duplicate_contract(
        train_paths,
        test_paths,
        train["label"].to_numpy(),
        workers=max(32, args.wave2_workers),
        cache_path=wave0.ensure_within(
            args.output_root
            / "_shared_wave2_cache"
            / cid
            / "exact_image_sha256_groups_v1.npz",
            args.allowed_root,
        ),
    )
    return _run_vision(
        args=args, task_dir=task_dir, logger=logger, competition_id=cid,
        train_frame=train, sample=sample,
        train_paths=train_paths,
        test_paths=test_paths,
        target_columns=["label"], mode="binary", epochs=args.wave2_histopath_epochs,
        backbone="efficientnet_v2_s",
        center_patch=True, image_size=192, batch_size=128, tta_flips=True,
        fold_count=min(3, args.wave2_vision_folds),
        group_values=train_hashes,
        group_contract=duplicate_contract,
        test_probability_override=test_overrides,
        worker_count=32,
        promotion_contract=WAVE2_PROMOTION_CONTRACTS[cid],
    )


def run_plant(args: argparse.Namespace, task_dir: Path, logger: Any) -> dict[str, Any]:
    cid = "plant-pathology-2020-fgvc7"
    resolved = resolve_competition(cid, args.data_root)
    train = pd.read_csv(resolved.public_dir / "train.csv")
    sample = pd.read_csv(resolved.sample_submission_path)
    targets = list(sample.columns[1:])
    train = train.copy()
    train["__class__"] = train[targets].to_numpy().argmax(axis=1)
    class_names = np.asarray(targets, dtype=object)
    train["__label__"] = class_names[train["__class__"].to_numpy()]
    return _run_vision(
        args=args, task_dir=task_dir, logger=logger, competition_id=cid,
        train_frame=train, sample=sample,
        train_paths=[resolved.public_dir / "images" / f"{value}.jpg" for value in train["image_id"]],
        test_paths=[resolved.public_dir / "images" / f"{value}.jpg" for value in sample["image_id"]],
        target_columns=targets, mode="multiclass", epochs=args.wave2_plant_epochs,
        backbone="convnext_small",
        label_column="__label__",
        image_size=448, batch_size=32, tta_flips=True,
        fold_count=min(3, args.wave2_vision_folds),
        preprocessing_profile="full_frame", early_stopping_min_epochs=3,
        promotion_contract=WAVE2_PROMOTION_CONTRACTS[cid],
    )


def run_ranzcr(args: argparse.Namespace, task_dir: Path, logger: Any) -> dict[str, Any]:
    cid = "ranzcr-clip-catheter-line-classification"
    resolved = resolve_competition(cid, args.data_root)
    train = pd.read_csv(resolved.public_dir / "train.csv")
    sample = pd.read_csv(resolved.sample_submission_path)
    targets = list(sample.columns[1:])
    if "PatientID" not in train or train["PatientID"].isna().any():
        raise RuntimeError("RANZCR requires a complete PatientID column before GPU allocation")
    backbone = getattr(args, "wave2_ranzcr_backbone", "efficientnet_v2_s")
    image_size = int(getattr(args, "wave2_ranzcr_image_size", 512))
    batch_size = int(getattr(args, "wave2_ranzcr_batch_size", 24))
    fold_count = int(getattr(args, "wave2_ranzcr_folds", args.wave2_vision_folds))
    learning_rate = float(
        getattr(args, "wave2_ranzcr_learning_rate", args.wave2_learning_rate)
    )
    if backbone not in VISION_BACKBONE_SPECS:
        raise RuntimeError(f"Unsupported frozen RANZCR backbone: {backbone}")
    if image_size < 512 or batch_size < 1 or fold_count < 3 or learning_rate <= 0:
        raise RuntimeError("RANZCR high-resolution profile is invalid")
    return _run_vision(
        args=args, task_dir=task_dir, logger=logger, competition_id=cid,
        train_frame=train, sample=sample,
        train_paths=[resolved.public_dir / "train" / f"{value}.jpg" for value in train["StudyInstanceUID"]],
        test_paths=[resolved.public_dir / "test" / f"{value}.jpg" for value in sample["StudyInstanceUID"]],
        target_columns=targets, mode="multilabel", epochs=args.wave2_ranzcr_epochs,
        backbone=backbone,
        monochrome=True, image_size=image_size, batch_size=batch_size,
        vertical_flip=False, tta_flips=True,
        group_values=train["PatientID"].astype(str),
        fold_count=fold_count,
        worker_count=32,
        learning_rate=learning_rate,
        promotion_contract=WAVE2_PROMOTION_CONTRACTS[cid],
    )


_NOMAD_ELEMENTS = ("Al", "Ga", "In", "O")
_NOMAD_ELEMENT_PAIRS = tuple(itertools.combinations_with_replacement(_NOMAD_ELEMENTS, 2))
_NOMAD_PROPERTIES = {
    "atomic_number": {"Al": 13.0, "Ga": 31.0, "In": 49.0, "O": 8.0},
    "atomic_mass": {"Al": 26.9815, "Ga": 69.723, "In": 114.818, "O": 15.999},
    "covalent_radius": {"Al": 1.21, "Ga": 1.22, "In": 1.42, "O": 0.66},
    "vdw_radius": {"Al": 1.84, "Ga": 1.87, "In": 1.93, "O": 1.52},
    "electronegativity": {"Al": 1.61, "Ga": 1.81, "In": 1.78, "O": 3.44},
}


def parse_nomad_geometry(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Parse one NOMAD ``geometry.xyz`` file with strict finite invariants."""

    lattice: list[list[float]] = []
    coordinates: list[list[float]] = []
    elements: list[str] = []
    coordinate_modes: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="strict").splitlines():
        parts = raw.split()
        if not parts or parts[0].startswith("#"):
            continue
        if parts[0] == "lattice_vector" and len(parts) >= 4:
            lattice.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] in {"atom", "atom_frac"} and len(parts) >= 5:
            element = parts[4]
            if element not in _NOMAD_ELEMENTS:
                raise RuntimeError(f"Unsupported NOMAD element {element!r} in {path}")
            coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elements.append(element)
            coordinate_modes.append(parts[0])
    cell = np.asarray(lattice, dtype=np.float64)
    atoms = np.asarray(coordinates, dtype=np.float64)
    if cell.shape != (3, 3) or atoms.ndim != 2 or atoms.shape[1:] != (3,) or not len(atoms):
        raise RuntimeError(f"Malformed NOMAD geometry: {path}")
    if not np.isfinite(cell).all() or not np.isfinite(atoms).all():
        raise RuntimeError(f"Non-finite NOMAD geometry: {path}")
    if abs(float(np.linalg.det(cell))) <= 1e-6:
        raise RuntimeError(f"Singular NOMAD lattice: {path}")
    modes = set(coordinate_modes)
    if len(modes) != 1:
        raise RuntimeError(f"Mixed NOMAD coordinate modes: {path}")
    if modes == {"atom_frac"}:
        atoms = atoms @ cell
    return cell, atoms, tuple(elements)


def _nomad_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {
            f"{prefix}_{name}": 0.0
            for name in ("count", "min", "mean", "std", "p10", "p25", "p50", "p75", "p90", "max")
        }
    return {
        f"{prefix}_count": float(values.size),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_p10": float(np.percentile(values, 10)),
        f"{prefix}_p25": float(np.percentile(values, 25)),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p75": float(np.percentile(values, 75)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_max": float(values.max()),
    }


def _minimum_image_distances(
    cell: np.ndarray, coordinates: np.ndarray, elements: tuple[str, ...]
) -> tuple[np.ndarray, dict[tuple[str, str], np.ndarray], np.ndarray]:
    fractional = coordinates @ np.linalg.inv(cell)
    fractional -= np.floor(fractional)
    count = len(coordinates)
    matrix = np.zeros((count, count), dtype=np.float64)
    pair_values: dict[tuple[str, str], list[float]] = {pair: [] for pair in _NOMAD_ELEMENT_PAIRS}
    offsets = np.asarray(list(itertools.product((-1, 0, 1), repeat=3)), dtype=np.float64)
    all_values: list[float] = []
    for left in range(count):
        for right in range(left + 1, count):
            delta = fractional[right] - fractional[left]
            center = -np.rint(delta)
            candidates = (delta[None, :] + center[None, :] + offsets) @ cell
            distance = float(np.linalg.norm(candidates, axis=1).min())
            matrix[left, right] = matrix[right, left] = distance
            pair = tuple(sorted((elements[left], elements[right]), key=_NOMAD_ELEMENTS.index))
            pair_values[pair].append(distance)
            all_values.append(distance)
    return (
        np.asarray(all_values, dtype=np.float64),
        {key: np.asarray(value, dtype=np.float64) for key, value in pair_values.items()},
        matrix,
    )


def nomad_structure_features(path: Path) -> dict[str, float]:
    """Extract a fixed-width, periodic crystal descriptor from ``geometry.xyz``."""

    cell, coordinates, elements = parse_nomad_geometry(path)
    result: dict[str, float] = {}
    lengths = np.linalg.norm(cell, axis=1)
    volume = abs(float(np.linalg.det(cell)))
    reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T
    reciprocal_lengths = np.linalg.norm(reciprocal, axis=1)
    singular = np.linalg.svd(cell, compute_uv=False)
    gram = cell @ cell.T
    for index, value in enumerate(cell.ravel()):
        result[f"geom_cell_{index}"] = float(value)
    for index, value in enumerate(gram[np.triu_indices(3)]):
        result[f"geom_gram_{index}"] = float(value)
    for index, value in enumerate(lengths):
        result[f"geom_lattice_length_{index + 1}"] = float(value)
    for index, value in enumerate(reciprocal_lengths):
        result[f"geom_reciprocal_length_{index + 1}"] = float(value)
    for index, value in enumerate(singular):
        result[f"geom_singular_value_{index + 1}"] = float(value)
    result.update({
        "geom_volume": volume,
        "geom_volume_per_atom": volume / len(coordinates),
        "geom_lattice_mean": float(lengths.mean()),
        "geom_lattice_std": float(lengths.std()),
        "geom_lattice_min": float(lengths.min()),
        "geom_lattice_max": float(lengths.max()),
        "geom_lattice_ratio": float(lengths.max() / max(lengths.min(), 1e-12)),
        "geom_reciprocal_mean": float(reciprocal_lengths.mean()),
        "geom_condition": float(singular.max() / max(singular.min(), 1e-12)),
        "geom_atom_count": float(len(coordinates)),
    })
    angle_pairs = ((1, 2), (0, 2), (0, 1))
    for name, (left, right) in zip(("alpha", "beta", "gamma"), angle_pairs):
        cosine = float(np.dot(cell[left], cell[right]) / max(lengths[left] * lengths[right], 1e-12))
        result[f"geom_angle_{name}"] = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    element_array = np.asarray(elements)
    for element in _NOMAD_ELEMENTS:
        count = int(np.sum(element_array == element))
        result[f"geom_count_{element.lower()}"] = float(count)
        result[f"geom_fraction_{element.lower()}"] = float(count / len(elements))
    fractions = np.asarray([result[f"geom_fraction_{element.lower()}"] for element in _NOMAD_ELEMENTS])
    result["geom_composition_entropy"] = float(-np.sum(fractions * np.log(np.clip(fractions, 1e-12, None))))
    for left, right in _NOMAD_ELEMENT_PAIRS:
        result[f"geom_fraction_product_{left.lower()}_{right.lower()}"] = (
            result[f"geom_fraction_{left.lower()}"] * result[f"geom_fraction_{right.lower()}"]
        )
    for property_name, table in _NOMAD_PROPERTIES.items():
        values = np.asarray([table[element] for element in elements], dtype=np.float64)
        result.update({
            f"geom_{property_name}_mean": float(values.mean()),
            f"geom_{property_name}_std": float(values.std()),
            f"geom_{property_name}_min": float(values.min()),
            f"geom_{property_name}_max": float(values.max()),
            f"geom_{property_name}_range": float(values.max() - values.min()),
        })

    all_distances, pair_distances, distance_matrix = _minimum_image_distances(cell, coordinates, elements)
    result.update(_nomad_stats("geom_pair_all", all_distances))
    bins = np.linspace(0.0, 8.0, 41)
    shell_volume = np.maximum((4.0 * np.pi / 3.0) * (bins[1:] ** 3 - bins[:-1] ** 3), 1e-12)
    for pair in (("ALL", "ALL"), *_NOMAD_ELEMENT_PAIRS):
        values = all_distances if pair[0] == "ALL" else pair_distances[pair]
        prefix = f"geom_pair_{pair[0].lower()}_{pair[1].lower()}"
        if pair[0] != "ALL":
            result.update(_nomad_stats(prefix, values))
        histogram, _ = np.histogram(values, bins=bins)
        normalized = histogram / max(1, len(values)) / shell_volume
        for index, value in enumerate(normalized):
            result[f"{prefix}_rdf_{index:02d}"] = float(value)

    if len(coordinates) > 1:
        neighbor = np.sort(np.where(distance_matrix > 0, distance_matrix, np.inf), axis=1)
        for rank in range(6):
            values = neighbor[:, min(rank, neighbor.shape[1] - 1)]
            values = values[np.isfinite(values)]
            statistics = (
                (("mean", values.mean()), ("std", values.std()), ("min", values.min()),
                 ("median", np.median(values)), ("max", values.max()))
                if values.size
                else (("mean", 0.0), ("std", 0.0), ("min", 0.0), ("median", 0.0), ("max", 0.0))
            )
            for name, value in statistics:
                result[f"geom_neighbor_{rank + 1}_{name}"] = float(value)
    else:
        for rank in range(6):
            for name in ("mean", "std", "min", "median", "max"):
                result[f"geom_neighbor_{rank + 1}_{name}"] = 0.0
    for cutoff in (2.0, 2.5, 3.0, 3.5, 4.0):
        result[f"geom_coordination_all_{cutoff:.1f}"] = float(np.sum((distance_matrix > 0) & (distance_matrix <= cutoff)) / max(1, len(coordinates)))
        for center in _NOMAD_ELEMENTS:
            center_mask = element_array == center
            for neighbor_element in _NOMAD_ELEMENTS:
                neighbor_mask = element_array == neighbor_element
                values = distance_matrix[np.ix_(center_mask, neighbor_mask)]
                positive = (values > 0) & (values <= cutoff)
                result[f"geom_coordination_{center.lower()}_{neighbor_element.lower()}_{cutoff:.1f}"] = float(
                    positive.sum() / max(1, int(center_mask.sum()))
                )

    # A periodic Coulomb-like spectrum adds a compact, permutation-invariant
    # descriptor without allowing variable feature widths.
    atomic_numbers = np.asarray([_NOMAD_PROPERTIES["atomic_number"][element] for element in elements])
    coulomb = np.zeros_like(distance_matrix)
    np.fill_diagonal(coulomb, 0.5 * atomic_numbers ** 2.4)
    mask = distance_matrix > 1e-12
    coulomb[mask] = (atomic_numbers[:, None] * atomic_numbers[None, :])[mask] / distance_matrix[mask]
    eigenvalues = np.sort(np.linalg.eigvalsh(coulomb))[::-1]
    for index in range(20):
        result[f"geom_coulomb_eigen_{index:02d}"] = float(eigenvalues[index]) if index < len(eigenvalues) else 0.0
    result["geom_coulomb_trace"] = float(np.trace(coulomb))
    result["geom_coulomb_frobenius"] = float(np.linalg.norm(coulomb))
    result["geom_coulomb_spectral_radius"] = float(np.max(np.abs(eigenvalues)))
    if not np.isfinite(np.fromiter(result.values(), dtype=np.float64)).all():
        raise RuntimeError(f"Non-finite NOMAD features from {path}")
    return result


def _nomad_id(value: Any) -> str:
    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError):
        pass
    return str(value)


def _parallel_nomad_structure_features(
    paths: list[Path],
    workers: int,
) -> list[dict[str, float]]:
    """Extract NOMAD geometry descriptors in stable input order.

    The descriptor contains enough Python-level work that threads do not
    improve throughput.  A bounded process pool does, while ``executor.map``
    preserves row order and therefore the submission/OOF alignment contract.
    """

    if not paths:
        return []
    effective_workers = min(max(1, int(workers)), len(paths), os.cpu_count() or 1)
    if effective_workers == 1:
        return [nomad_structure_features(path) for path in paths]
    chunksize = max(1, len(paths) // (effective_workers * 4))
    with ProcessPoolExecutor(max_workers=effective_workers) as pool:
        return list(pool.map(nomad_structure_features, paths, chunksize=chunksize))


def _nomad_features(
    frame: pd.DataFrame,
    geometry_root: Path | None = None,
    *,
    geometry_workers: int = 1,
) -> pd.DataFrame:
    result = frame.drop(
        columns=[column for column in ("id", "formation_energy_ev_natom", "bandgap_energy_ev") if column in frame],
        errors="ignore",
    ).copy()
    for column in result.columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    a, b, c = (result[f"lattice_vector_{index}_ang"] for index in (1, 2, 3))
    alpha = np.deg2rad(result["lattice_angle_alpha_degree"])
    beta = np.deg2rad(result["lattice_angle_beta_degree"])
    gamma = np.deg2rad(result["lattice_angle_gamma_degree"])
    volume_factor = np.sqrt(np.clip(
        1 + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
        - np.cos(alpha) ** 2 - np.cos(beta) ** 2 - np.cos(gamma) ** 2,
        0,
        None,
    ))
    result["cell_volume"] = a * b * c * volume_factor
    result["volume_per_atom"] = result["cell_volume"] / result["number_of_total_atoms"].clip(lower=1)
    result["lattice_mean"] = (a + b + c) / 3
    result["lattice_std"] = result[[f"lattice_vector_{i}_ang" for i in (1, 2, 3)]].std(axis=1)
    for left in ("al", "ga", "in"):
        for right in ("al", "ga", "in"):
            if left <= right:
                result[f"composition_{left}_{right}"] = result[f"percent_atom_{left}"] * result[f"percent_atom_{right}"]
    if geometry_root is not None:
        geometry_paths: list[Path] = []
        for value in frame["id"]:
            geometry_path = geometry_root / _nomad_id(value) / "geometry.xyz"
            if not geometry_path.is_file():
                raise FileNotFoundError(f"Missing NOMAD geometry for id={_nomad_id(value)}")
            geometry_paths.append(geometry_path)
        rows = _parallel_nomad_structure_features(geometry_paths, geometry_workers)
        geometry = pd.DataFrame(rows, index=result.index)
        result = pd.concat([result, geometry], axis=1)
        result["geom_atom_count_residual"] = result["geom_atom_count"] - result["number_of_total_atoms"]
        for index in (1, 2, 3):
            result[f"geom_lattice_length_residual_{index}"] = (
                result[f"geom_lattice_length_{index}"] - result[f"lattice_vector_{index}_ang"]
            )
        for name in ("alpha", "beta", "gamma"):
            result[f"geom_angle_residual_{name}"] = (
                result[f"geom_angle_{name}"] - result[f"lattice_angle_{name}_degree"]
            )
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def select_nomad_log_blend_weight(
    truth_log: np.ndarray,
    catboost_log: np.ndarray,
    extra_trees_log: np.ndarray,
) -> tuple[float, float]:
    """Select a deterministic CatBoost weight against log-space RMSE."""

    truth = np.asarray(truth_log, dtype=np.float64)
    catboost = np.asarray(catboost_log, dtype=np.float64)
    extra_trees = np.asarray(extra_trees_log, dtype=np.float64)
    if not (truth.shape == catboost.shape == extra_trees.shape) or truth.ndim != 1:
        raise RuntimeError("NOMAD blend arrays have inconsistent shapes")
    if not np.isfinite(np.column_stack((truth, catboost, extra_trees))).all():
        raise RuntimeError("NOMAD blend arrays contain non-finite values")
    candidates = np.linspace(0.0, 1.0, 101)
    losses = np.asarray([
        np.sqrt(np.mean((truth - (weight * catboost + (1.0 - weight) * extra_trees)) ** 2))
        for weight in candidates
    ])
    best_index = int(np.argmin(losses))
    return float(candidates[best_index]), float(losses[best_index])


def cross_fit_nomad_log_blend(
    truth_log: np.ndarray,
    catboost_log: np.ndarray,
    extra_trees_log: np.ndarray,
    fold_assignment: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Choose blend weights without each outer fold and predict that fold."""

    truth = np.asarray(truth_log, dtype=np.float64)
    catboost = np.asarray(catboost_log, dtype=np.float64)
    extra_trees = np.asarray(extra_trees_log, dtype=np.float64)
    folds = np.asarray(fold_assignment)
    if not (truth.shape == catboost.shape == extra_trees.shape == folds.shape):
        raise RuntimeError("NOMAD cross-fit arrays have inconsistent shapes")
    prediction = np.full(truth.shape, np.nan, dtype=np.float64)
    records: list[dict[str, float]] = []
    for fold in sorted(int(value) for value in np.unique(folds)):
        validation = folds == fold
        fitting = ~validation
        if not validation.any() or not fitting.any():
            raise RuntimeError("NOMAD cross-fit fold is empty")
        weight, fitting_rmse = select_nomad_log_blend_weight(
            truth[fitting], catboost[fitting], extra_trees[fitting]
        )
        prediction[validation] = (
            weight * catboost[validation] + (1.0 - weight) * extra_trees[validation]
        )
        records.append({
            "fold": float(fold),
            "catboost_weight": weight,
            "extra_trees_weight": 1.0 - weight,
            "meta_fit_rmsle": fitting_rmse,
            "outer_rmsle": float(np.sqrt(np.mean((truth[validation] - prediction[validation]) ** 2))),
        })
    if not np.isfinite(prediction).all():
        raise RuntimeError("NOMAD cross-fit blend did not cover every row")
    return prediction, records


def run_nomad(args: argparse.Namespace, task_dir: Path, logger: Any) -> dict[str, Any]:
    from catboost import CatBoostRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.model_selection import KFold

    cid = "nomad2018-predict-transparent-conductors"
    resolved = resolve_competition(cid, args.data_root)
    train = pd.read_csv(resolved.public_dir / "train.csv")
    test = pd.read_csv(resolved.public_dir / "test.csv")
    sample = pd.read_csv(resolved.sample_submission_path)
    targets = ["formation_energy_ev_natom", "bandgap_energy_ev"]
    geometry_workers = min(
        max(1, int(args.wave2_nomad_workers)),
        os.cpu_count() or 1,
    )
    feature_started = time.perf_counter()
    x = _nomad_features(
        train,
        resolved.public_dir / "train",
        geometry_workers=geometry_workers,
    )
    x_test = _nomad_features(
        test,
        resolved.public_dir / "test",
        geometry_workers=geometry_workers,
    ).reindex(columns=x.columns, fill_value=0.0)
    feature_runtime_seconds = time.perf_counter() - feature_started
    fold_count = max(2, min(args.wave2_nomad_folds, len(train)))
    folds = KFold(n_splits=fold_count, shuffle=True, random_state=args.seed)
    fold_assignment = np.full(len(train), -1, dtype=np.int16)
    valid_predictions = np.zeros((len(train), 2), dtype=float)
    test_predictions = np.zeros((len(test), 2), dtype=float)
    best_iterations: dict[str, list[int]] = {}
    blend_weights: dict[str, dict[str, float]] = {}
    crossfit_blend_records: dict[str, list[dict[str, float]]] = {}
    target_diagnostics: dict[str, Any] = {}
    started = time.perf_counter()
    for column_index, target in enumerate(targets):
        y = np.log1p(train[target].clip(lower=0).to_numpy())
        oof_cat = np.zeros(len(train), dtype=np.float64)
        oof_extra = np.zeros(len(train), dtype=np.float64)
        test_cat = np.zeros(len(test), dtype=np.float64)
        test_extra = np.zeros(len(test), dtype=np.float64)
        target_best: list[int] = []
        for fold, (train_indices, valid_indices) in enumerate(folds.split(x)):
            fold_assignment[valid_indices] = fold
            model = CatBoostRegressor(
                iterations=args.wave2_nomad_iterations,
                depth=7,
                learning_rate=0.03,
                loss_function="RMSE",
                eval_metric="RMSE",
                random_seed=args.seed + column_index * 101 + fold,
                task_type="GPU",
                devices="0",
                border_count=128,
                l2_leaf_reg=6.0,
                random_strength=0.25,
                od_type="Iter",
                od_wait=150,
                verbose=100,
                allow_writing_files=False,
            )
            model.fit(
                x.iloc[train_indices],
                y[train_indices],
                eval_set=(x.iloc[valid_indices], y[valid_indices]),
                use_best_model=True,
            )
            best = max(100, int(model.get_best_iteration()) + 1)
            target_best.append(best)
            oof_cat[valid_indices] = model.predict(x.iloc[valid_indices])
            test_cat += model.predict(x_test) / fold_count
            model.save_model(str(task_dir / f"nomad_{target}_fold{fold}.cbm"))

            extra = ExtraTreesRegressor(
                n_estimators=args.wave2_nomad_extra_trees,
                max_features=0.85,
                min_samples_leaf=1,
                random_state=args.seed + column_index * 101 + fold,
                n_jobs=-1,
            )
            extra.fit(x.iloc[train_indices], y[train_indices])
            oof_extra[valid_indices] = extra.predict(x.iloc[valid_indices])
            test_extra += extra.predict(x_test) / fold_count
        crossfit_log, crossfit_records = cross_fit_nomad_log_blend(
            y, oof_cat, oof_extra, fold_assignment
        )
        cat_weight, full_meta_fit_rmsle = select_nomad_log_blend_weight(y, oof_cat, oof_extra)
        test_log = cat_weight * test_cat + (1.0 - cat_weight) * test_extra
        valid_predictions[:, column_index] = np.clip(np.expm1(crossfit_log), 0.0, None)
        test_predictions[:, column_index] = np.clip(np.expm1(test_log), 0.0, None)
        best_iterations[target] = target_best
        blend_weights[target] = {"catboost": cat_weight, "extra_trees": 1.0 - cat_weight}
        crossfit_blend_records[target] = crossfit_records
        target_diagnostics[target] = {
            "catboost_oof_rmsle": float(np.sqrt(np.mean((y - oof_cat) ** 2))),
            "extra_trees_oof_rmsle": float(np.sqrt(np.mean((y - oof_extra) ** 2))),
            "cross_fitted_blend_oof_rmsle": float(np.sqrt(np.mean((y - crossfit_log) ** 2))),
            "full_oof_meta_fit_rmsle": full_meta_fit_rmsle,
        }
    if np.any(fold_assignment < 0):
        raise RuntimeError("NOMAD OOF coverage is incomplete")
    cv_score = compute_metric("mean_columnwise_rmsle", train[targets], valid_predictions)
    sample[targets] = test_predictions
    np.savez_compressed(
        task_dir / "nomad_oof_predictions.npz",
        ids=train["id"].to_numpy(),
        truth=train[targets].to_numpy(),
        predictions=valid_predictions,
        fold=fold_assignment,
    )
    budget = {
        "seed": args.seed,
        "train_rows": len(train),
        "test_rows": len(test),
        "feature_count": len(x.columns),
        "geometry_feature_count": int(sum(column.startswith("geom_") for column in x.columns)),
        "geometry_workers": geometry_workers,
        "runtime_seconds_feature_engineering": feature_runtime_seconds,
        "folds": fold_count,
        "best_iterations": best_iterations,
        "blend_weights": blend_weights,
        "cross_fitted_meta_validation": True,
    }
    return wave0.finalize_scored_task(
        competition_id=cid, submission=sample, cv_score=cv_score, args=args, task_dir=task_dir,
        budget=budget, extra={"runtime_seconds_model": time.perf_counter() - started,
                              "model_family": "NOMAD_geometry_CatBoost_GPU_ExtraTrees_5fold_log_blend",
                              "target_diagnostics": target_diagnostics,
                              "crossfit_blend": crossfit_blend_records, "budget": budget},
    )


def compute_nb_log_count_ratio(matrix: Any, labels: np.ndarray) -> np.ndarray:
    """Compute the NB-SVM log-count ratio for one binary target."""

    labels = np.asarray(labels, dtype=np.int8)
    if set(np.unique(labels)) - {0, 1}:
        raise ValueError("NB-SVM labels must be binary")
    binary = matrix.copy().tocsr()
    binary.data = np.ones_like(binary.data)
    positive_count = int(np.sum(labels == 1))
    negative_count = int(np.sum(labels == 0))
    if positive_count == 0 or negative_count == 0:
        raise ValueError("NB-SVM requires both classes")
    positive = (1.0 + np.asarray(binary[labels == 1].sum(axis=0)).ravel()) / (2.0 + positive_count)
    negative = (1.0 + np.asarray(binary[labels == 0].sum(axis=0)).ravel()) / (2.0 + negative_count)
    return np.log(positive / negative).astype(np.float32)


def _fractional_rank(values: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    values = np.asarray(values, dtype=np.float64)
    return (rankdata(values, method="average") - 0.5) / max(1, len(values))


def make_multilabel_stratified_folds(
    targets: np.ndarray,
    *,
    requested_folds: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Iteratively distribute multilabel rows while balancing rare positives.

    This is a small dependency-free implementation of the iterative
    multilabel-stratification idea.  It assigns the rarest remaining label
    first, uses fold capacity as the secondary objective and fails closed if a
    scoreable label is absent from any validation fold.
    """

    matrix = np.asarray(targets)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("Multilabel stratification requires a non-empty 2D target matrix")
    if not np.isin(matrix, (0, 1)).all():
        raise ValueError("Multilabel targets must contain only zero and one")
    matrix = matrix.astype(np.int8, copy=False)
    sample_count, label_count = matrix.shape
    fold_count = max(2, min(int(requested_folds), sample_count))
    rng = np.random.RandomState(seed)

    assignment = np.full(sample_count, -1, dtype=np.int16)
    unassigned = np.ones(sample_count, dtype=bool)
    desired_size = np.full(fold_count, sample_count / fold_count, dtype=np.float64)
    desired_label = np.tile(matrix.sum(axis=0, dtype=np.float64) / fold_count, (fold_count, 1))

    while True:
        remaining_label = matrix[unassigned].sum(axis=0)
        selectable = np.flatnonzero(remaining_label > 0)
        if not len(selectable):
            break
        rarest_count = remaining_label[selectable].min()
        rare_labels = selectable[remaining_label[selectable] == rarest_count]
        label = int(rng.choice(rare_labels))
        rows = np.flatnonzero(unassigned & (matrix[:, label] == 1))
        rng.shuffle(rows)
        rows = rows[np.argsort(-matrix[rows].sum(axis=1), kind="stable")]
        for row in rows:
            if not unassigned[row]:
                continue
            label_need = desired_label[:, label]
            candidates = np.flatnonzero(np.isclose(label_need, label_need.max()))
            size_need = desired_size[candidates]
            candidates = candidates[np.isclose(size_need, size_need.max())]
            fold = int(rng.choice(candidates))
            assignment[row] = fold
            unassigned[row] = False
            desired_size[fold] -= 1.0
            desired_label[fold, matrix[row].astype(bool)] -= 1.0

    remaining = np.flatnonzero(unassigned)
    rng.shuffle(remaining)
    for row in remaining:
        candidates = np.flatnonzero(np.isclose(desired_size, desired_size.max()))
        fold = int(rng.choice(candidates))
        assignment[row] = fold
        desired_size[fold] -= 1.0

    if np.any(assignment < 0):
        raise RuntimeError("Multilabel fold assignment is incomplete")
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    all_indices = np.arange(sample_count)
    for fold in range(fold_count):
        valid = all_indices[assignment == fold]
        train = all_indices[assignment != fold]
        if not len(valid) or not len(train):
            raise RuntimeError("Multilabel stratification produced an empty fold")
        fold_targets = matrix[valid]
        for label in range(label_count):
            positives = int(matrix[:, label].sum())
            negatives = sample_count - positives
            if positives >= fold_count and negatives >= fold_count:
                if fold_targets[:, label].min() == fold_targets[:, label].max():
                    raise RuntimeError(
                        f"Multilabel fold {fold} is not scoreable for label {label}"
                    )
        splits.append((train, valid))
    return splits, assignment


def cross_fit_multilabel_rank_blend(
    oof_word: np.ndarray,
    oof_char: np.ndarray,
    test_word: np.ndarray,
    test_char: np.ndarray,
    truth: np.ndarray,
    fold_assignment: np.ndarray,
    *,
    weight_grid: Iterable[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select word/character rank weights without scoring on fit rows."""

    oof_word = np.asarray(oof_word, dtype=np.float64)
    oof_char = np.asarray(oof_char, dtype=np.float64)
    test_word = np.asarray(test_word, dtype=np.float64)
    test_char = np.asarray(test_char, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.int8)
    fold_assignment = np.asarray(fold_assignment, dtype=np.int16)
    if oof_word.shape != oof_char.shape or oof_word.shape != truth.shape:
        raise ValueError("Jigsaw OOF component and truth shapes must match")
    if test_word.ndim == 2:
        test_word = test_word[None, ...]
        test_char = test_char[None, ...]
    if (
        test_word.ndim != 3
        or test_word.shape != test_char.shape
        or test_word.shape[2] != truth.shape[1]
    ):
        raise ValueError("Jigsaw test component shapes must match the target columns")
    if len(fold_assignment) != len(truth) or np.any(fold_assignment < 0):
        raise ValueError("Jigsaw fold assignment must cover every training row")
    arrays = (oof_word, oof_char, test_word, test_char)
    if any(not np.isfinite(values).all() for values in arrays):
        raise ValueError("Jigsaw blend inputs must be finite")

    grid = np.asarray(list(weight_grid) if weight_grid is not None else np.linspace(0.0, 1.0, 21))
    if grid.ndim != 1 or not len(grid) or np.any((grid < 0.0) | (grid > 1.0)):
        raise ValueError("Jigsaw blend weights must be a non-empty grid inside [0, 1]")
    folds = np.unique(fold_assignment)
    label_count = truth.shape[1]
    cross_fitted = np.zeros_like(oof_word)
    fold_weights = np.zeros((len(folds), label_count), dtype=np.float64)
    final_weights = np.zeros(label_count, dtype=np.float64)
    test_blend = np.zeros_like(test_word[0])

    def choose_weight(word: np.ndarray, char: np.ndarray, labels: np.ndarray) -> float:
        if len(np.unique(labels)) != 2:
            raise RuntimeError("Every meta-training label must contain both classes")
        word_rank = _fractional_rank(word)
        char_rank = _fractional_rank(char)
        scored = [
            (compute_metric("roc_auc", labels, weight * word_rank + (1.0 - weight) * char_rank), float(weight))
            for weight in grid
        ]
        # Prefer a balanced blend when multiple weights have identical AUC.
        return max(scored, key=lambda item: (item[0], -abs(item[1] - 0.5)))[1]

    for label in range(label_count):
        for fold_index, fold in enumerate(folds):
            fit = fold_assignment != fold
            held_out = fold_assignment == fold
            if len(np.unique(truth[held_out, label])) != 2:
                raise RuntimeError(f"Held-out fold {int(fold)} is not scoreable for label {label}")
            weight = choose_weight(oof_word[fit, label], oof_char[fit, label], truth[fit, label])
            fold_weights[fold_index, label] = weight
            cross_fitted[held_out, label] = (
                weight * _fractional_rank(oof_word[held_out, label])
                + (1.0 - weight) * _fractional_rank(oof_char[held_out, label])
            )
        final_weight = choose_weight(oof_word[:, label], oof_char[:, label], truth[:, label])
        final_weights[label] = final_weight
        normalized_test_word = np.mean(
            [_fractional_rank(component[:, label]) for component in test_word], axis=0
        )
        normalized_test_char = np.mean(
            [_fractional_rank(component[:, label]) for component in test_char], axis=0
        )
        test_blend[:, label] = (
            final_weight * normalized_test_word
            + (1.0 - final_weight) * normalized_test_char
        )
    return cross_fitted, test_blend, fold_weights, final_weights


def align_multilabel_submission_by_id(
    sample: pd.DataFrame,
    test: pd.DataFrame,
    prediction: np.ndarray,
    *,
    id_column: str,
    target_columns: list[str],
) -> pd.DataFrame:
    """Join multilabel probabilities to the sample contract by explicit ID."""

    if list(sample.columns) != [id_column, *target_columns]:
        raise RuntimeError("Multilabel sample columns do not match the expected contract")
    if id_column not in test:
        raise RuntimeError(f"Test frame is missing ID column {id_column}")
    if sample[id_column].duplicated().any() or test[id_column].duplicated().any():
        raise RuntimeError("Multilabel sample/test IDs must be unique")
    if set(sample[id_column].astype(str)) != set(test[id_column].astype(str)):
        raise RuntimeError("Multilabel sample/test ID sets differ")
    values = _finite_probability(np.asarray(prediction, dtype=np.float64))
    if values.shape != (len(test), len(target_columns)):
        raise RuntimeError("Multilabel prediction shape does not match the test contract")
    predicted = pd.DataFrame(values, columns=target_columns)
    predicted.insert(0, id_column, test[id_column].astype(str).to_numpy())
    ordered_ids = sample[[id_column]].copy()
    ordered_ids[id_column] = ordered_ids[id_column].astype(str)
    aligned = ordered_ids.merge(predicted, on=id_column, how="left", validate="one_to_one", sort=False)
    if aligned[target_columns].isna().any().any():
        raise RuntimeError("Multilabel ID join produced missing predictions")
    result = sample.copy()
    result[target_columns] = aligned[target_columns].to_numpy(dtype=np.float64)
    return result


def _fit_nbsvm(matrix: Any, labels: np.ndarray, *, c_value: float, max_iter: int, seed: int):
    from sklearn.linear_model import LogisticRegression

    ratio = compute_nb_log_count_ratio(matrix, labels)
    model = LogisticRegression(
        C=c_value,
        max_iter=max_iter,
        solver="liblinear",
        random_state=seed,
    ).fit(matrix.multiply(ratio), labels)
    return model, ratio


def fit_jigsaw_nbsvm_channels(
    word_matrix: Any,
    char_matrix: Any,
    targets: np.ndarray,
    *,
    target_names: Sequence[str],
    c_value: float,
    max_iter: int,
    seed_base: int,
    workers: int,
) -> list[dict[str, Any]]:
    """Fit the fixed Jigsaw label/channel grid with deterministic ordering.

    Each liblinear fit is independent.  ``executor.map`` preserves the task
    order, while every task retains the exact seed previously used by the
    serial loop.  The sparse matrices are shared read-only between threads, so
    this avoids twelve serial solver passes without changing folds, features,
    ratios, coefficients, or prediction order.
    """

    truth = np.asarray(targets, dtype=np.int8)
    names = tuple(str(value) for value in target_names)
    if truth.ndim != 2 or truth.shape[1] != len(names):
        raise ValueError("Jigsaw targets and target names do not share a column contract")
    tasks: list[tuple[str, int, str, Any, np.ndarray, int]] = []
    for index, target in enumerate(names):
        labels = truth[:, index]
        tasks.extend(
            [
                ("word", index, target, word_matrix, labels, seed_base + index),
                ("char", index, target, char_matrix, labels, seed_base + index + 10_000),
            ]
        )

    def fit_one(task: tuple[str, int, str, Any, np.ndarray, int]) -> dict[str, Any]:
        channel, index, target, matrix, labels, seed = task
        model, ratio = _fit_nbsvm(
            matrix,
            labels,
            c_value=c_value,
            max_iter=max_iter,
            seed=seed,
        )
        return {
            "channel": channel,
            "index": index,
            "target": target,
            "model": model,
            "ratio": ratio,
            "seed": seed,
        }

    worker_count = max(1, min(int(workers), len(tasks)))
    if worker_count == 1:
        return [fit_one(task) for task in tasks]
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="jigsaw-nbsvm") as executor:
        return list(executor.map(fit_one, tasks))


def _jigsaw_logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped) - np.log1p(-clipped)


def _hash_index_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.int64).reshape(-1))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def fold_clean_fractional_rank_channels(
    channels: dict[str, np.ndarray],
    fold_assignment: np.ndarray,
    *,
    target_columns: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Rank every channel/label only inside the row's source OOF fold."""

    folds = np.asarray(fold_assignment, dtype=np.int16).reshape(-1)
    if not channels or not len(folds) or np.any(folds < 0):
        raise ValueError("Jigsaw fold-clean ranking requires complete folds and channels")
    target_names = tuple(str(value) for value in target_columns)
    expected_shape = (len(folds), len(target_names))
    normalized: list[np.ndarray] = []
    feature_names: list[str] = []
    unique_folds = sorted(int(value) for value in np.unique(folds))
    for channel_name, raw in channels.items():
        values = np.asarray(raw, dtype=np.float64)
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise ValueError(f"Jigsaw channel {channel_name!r} violates the OOF contract")
        ranked = np.full(expected_shape, np.nan, dtype=np.float64)
        for fold in unique_folds:
            held_out = folds == fold
            for label in range(len(target_names)):
                ranked[held_out, label] = _fractional_rank(values[held_out, label])
        if not np.isfinite(ranked).all():
            raise RuntimeError(f"Jigsaw fold-clean ranking is incomplete for {channel_name}")
        normalized.append(ranked)
        feature_names.extend(
            f"rank:{channel_name}:{target}" for target in target_names
        )
    return np.concatenate(normalized, axis=1), feature_names


def build_jigsaw_meta_features(
    oof_word: np.ndarray,
    oof_char: np.ndarray,
    oof_blend: np.ndarray,
    fold_assignment: np.ndarray,
    *,
    target_columns: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Build the frozen Hybrid30 cross-label stacker feature contract."""

    target_names = tuple(str(value) for value in target_columns)
    ranked, ranked_names = fold_clean_fractional_rank_channels(
        {"word": oof_word, "char_wb": oof_char, "rank_blend": oof_blend},
        fold_assignment,
        target_columns=target_names,
    )
    raw = np.concatenate(
        [np.asarray(oof_word, dtype=np.float64), np.asarray(oof_char, dtype=np.float64)],
        axis=1,
    )
    if raw.shape[1] != 2 * len(target_names) or not np.isfinite(raw).all():
        raise ValueError("Jigsaw Hybrid30 raw channels violate the target contract")
    logit_names = [
        f"logit:{channel}:{target}"
        for channel in ("word", "char_wb")
        for target in target_names
    ]
    features = np.concatenate([ranked, _jigsaw_logit(raw)], axis=1)
    if features.shape[1] != 5 * len(target_names) or not np.isfinite(features).all():
        raise RuntimeError("Jigsaw Hybrid30 feature construction is incomplete")
    return features, [*ranked_names, *logit_names]


def build_jigsaw_fold_test_blends(
    test_word_by_fold: np.ndarray,
    test_char_by_fold: np.ndarray,
    fold_blend_weights: np.ndarray,
) -> np.ndarray:
    """Apply each fold's meta-trained rank weight to its matching test model."""

    word = np.asarray(test_word_by_fold, dtype=np.float64)
    char = np.asarray(test_char_by_fold, dtype=np.float64)
    weights = np.asarray(fold_blend_weights, dtype=np.float64)
    if word.ndim != 3 or word.shape != char.shape:
        raise ValueError("Jigsaw fold test channels must be matching 3-D arrays")
    if weights.shape != (word.shape[0], word.shape[2]):
        raise ValueError("Jigsaw fold blend weights do not match fold/label axes")
    if any(not np.isfinite(values).all() for values in (word, char, weights)):
        raise ValueError("Jigsaw fold test blend inputs must be finite")
    result = np.zeros_like(word, dtype=np.float64)
    for fold in range(word.shape[0]):
        for label in range(word.shape[2]):
            result[fold, :, label] = (
                weights[fold, label] * _fractional_rank(word[fold, :, label])
                + (1.0 - weights[fold, label])
                * _fractional_rank(char[fold, :, label])
            )
    return result


def build_jigsaw_test_meta_features_by_fold(
    test_word_by_fold: np.ndarray,
    test_char_by_fold: np.ndarray,
    test_blend_by_fold: np.ndarray,
    *,
    target_columns: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Mirror Hybrid30 normalization for each matching fold-specific test model."""

    word = np.asarray(test_word_by_fold, dtype=np.float64)
    char = np.asarray(test_char_by_fold, dtype=np.float64)
    blend = np.asarray(test_blend_by_fold, dtype=np.float64)
    if word.ndim != 3 or word.shape != char.shape or word.shape != blend.shape:
        raise ValueError("Jigsaw fold-specific test channels do not share a contract")
    target_names = tuple(str(value) for value in target_columns)
    if word.shape[2] != len(target_names):
        raise ValueError("Jigsaw fold-specific test labels do not match target columns")
    feature_names = [
        *[
            f"rank:{channel}:{target}"
            for channel in ("word", "char_wb", "rank_blend")
            for target in target_names
        ],
        *[
            f"logit:{channel}:{target}"
            for channel in ("word", "char_wb")
            for target in target_names
        ],
    ]
    features = np.zeros(
        (word.shape[0], word.shape[1], 5 * len(target_names)),
        dtype=np.float64,
    )
    for fold in range(word.shape[0]):
        ranked_parts: list[np.ndarray] = []
        for values in (word[fold], char[fold], blend[fold]):
            ranked = np.column_stack(
                [_fractional_rank(values[:, label]) for label in range(values.shape[1])]
            )
            ranked_parts.append(ranked)
        raw = np.concatenate([word[fold], char[fold]], axis=1)
        features[fold] = np.concatenate([*ranked_parts, _jigsaw_logit(raw)], axis=1)
    if not np.isfinite(features).all():
        raise RuntimeError("Jigsaw fold-specific test meta features are incomplete")
    return features, feature_names


def _resolve_jigsaw_stacker_c_grid(value: Any) -> tuple[float, ...]:
    if value is None:
        parsed = list(JIGSAW_STACKER_C_GRID)
    elif isinstance(value, str):
        parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    else:
        parsed = [float(item) for item in value]
    if not parsed or any(not math.isfinite(item) or item <= 0.0 for item in parsed):
        raise ValueError("Jigsaw stacker C grid must contain finite positive values")
    return tuple(dict.fromkeys(parsed))


def cross_fit_jigsaw_label_stacker(
    features: np.ndarray,
    test_features_by_fold: np.ndarray,
    truth: np.ndarray,
    fold_assignment: np.ndarray,
    *,
    target_columns: Sequence[str],
    feature_names: Sequence[str],
    c_grid: Iterable[float] = JIGSAW_STACKER_C_GRID,
    max_iter: int = 300,
    seed: int = 42,
    estimator_factory: Callable[[float, int], Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Nested-CV cross-label stacker with fold-matched test provenance.

    For an outer score fold, both inner C selection and the final meta fit use
    only rows from other outer folds.  Test prediction for that stacker uses
    base predictions from the matching outer-fold model and is averaged only
    after every score fold has produced one test component.
    """

    from sklearn.linear_model import LogisticRegression

    matrix = np.asarray(features, dtype=np.float64)
    test_matrix = np.asarray(test_features_by_fold, dtype=np.float64)
    labels = np.asarray(truth, dtype=np.int8)
    folds = np.asarray(fold_assignment, dtype=np.int16).reshape(-1)
    target_names = tuple(str(value) for value in target_columns)
    names = tuple(str(value) for value in feature_names)
    grid = _resolve_jigsaw_stacker_c_grid(c_grid)
    unique_folds = sorted(int(value) for value in np.unique(folds))
    expected_folds = list(range(len(unique_folds)))
    if unique_folds != expected_folds:
        raise ValueError("Jigsaw stacker requires contiguous zero-based folds")
    if matrix.ndim != 2 or matrix.shape[0] != len(labels) or len(folds) != len(labels):
        raise ValueError("Jigsaw stacker train arrays do not share a row contract")
    if labels.shape != (len(folds), len(target_names)):
        raise ValueError("Jigsaw stacker truth does not match target columns")
    if matrix.shape[1] != len(names):
        raise ValueError("Jigsaw stacker feature names do not match the matrix")
    if test_matrix.ndim != 3 or test_matrix.shape[0] != len(unique_folds):
        raise ValueError("Jigsaw stacker test matrix requires one source per outer fold")
    if test_matrix.shape[2] != matrix.shape[1]:
        raise ValueError("Jigsaw stacker train/test feature widths differ")
    if len(unique_folds) < 3 or any(not np.isfinite(value).all() for value in (matrix, test_matrix)):
        raise ValueError("Jigsaw stacker requires at least three complete finite folds")

    def make_estimator(c_value: float, model_seed: int):
        if estimator_factory is not None:
            return estimator_factory(float(c_value), int(model_seed))
        return LogisticRegression(
            C=float(c_value),
            class_weight=JIGSAW_STACKER_CLASS_WEIGHT,
            max_iter=int(max_iter),
            random_state=int(model_seed),
            solver="liblinear",
        )

    oof = np.full(labels.shape, np.nan, dtype=np.float64)
    write_counts = np.zeros(labels.shape, dtype=np.uint8)
    test_by_fold = np.zeros(
        (len(unique_folds), test_matrix.shape[1], len(target_names)),
        dtype=np.float64,
    )
    records: list[dict[str, Any]] = []
    for outer_index, outer_fold in enumerate(unique_folds):
        outer_fit = folds != outer_fold
        outer_score = folds == outer_fold
        fit_indices = np.flatnonzero(outer_fit)
        score_indices = np.flatnonzero(outer_score)
        inner_folds = [value for value in unique_folds if value != outer_fold]
        for label_index, target in enumerate(target_names):
            candidate_scores: list[dict[str, float]] = []
            for candidate_index, c_value in enumerate(grid):
                inner_prediction = np.full(len(labels), np.nan, dtype=np.float64)
                for inner_fold in inner_folds:
                    inner_score = outer_fit & (folds == inner_fold)
                    inner_fit = outer_fit & (folds != inner_fold)
                    if np.unique(labels[inner_fit, label_index]).size != 2:
                        raise RuntimeError(
                            f"Jigsaw stacker inner fit is not scoreable for {target}"
                        )
                    model = make_estimator(
                        c_value,
                        seed + 100_000 * outer_fold + 1_000 * inner_fold
                        + 10 * label_index + candidate_index,
                    )
                    model.fit(matrix[inner_fit], labels[inner_fit, label_index])
                    inner_prediction[inner_score] = model.predict_proba(
                        matrix[inner_score]
                    )[:, 1]
                if not np.isfinite(inner_prediction[outer_fit]).all():
                    raise RuntimeError("Jigsaw stacker inner OOF prediction is incomplete")
                score = float(
                    compute_metric(
                        "roc_auc",
                        labels[outer_fit, label_index],
                        inner_prediction[outer_fit],
                    )
                )
                candidate_scores.append({"c_value": float(c_value), "inner_auc": score})
            selected = max(
                candidate_scores,
                key=lambda item: (
                    item["inner_auc"],
                    -abs(math.log(item["c_value"] / 0.1)),
                    -item["c_value"],
                ),
            )
            model = make_estimator(
                selected["c_value"],
                seed + 1_000_000 + 10_000 * outer_fold + label_index,
            )
            model.fit(matrix[outer_fit], labels[outer_fit, label_index])
            oof[outer_score, label_index] = model.predict_proba(matrix[outer_score])[:, 1]
            write_counts[outer_score, label_index] += 1
            test_by_fold[outer_index, :, label_index] = model.predict_proba(
                test_matrix[outer_index]
            )[:, 1]
            coefficient = np.asarray(getattr(model, "coef_", []), dtype=np.float64).reshape(-1)
            intercept = np.asarray(getattr(model, "intercept_", []), dtype=np.float64).reshape(-1)
            records.append(
                {
                    "score_fold": outer_fold,
                    "test_source_fold": outer_fold,
                    "target": target,
                    "target_index": label_index,
                    "fit_folds": inner_folds,
                    "fit_rows": int(len(fit_indices)),
                    "score_rows": int(len(score_indices)),
                    "fit_row_ids_sha256": _hash_index_array(fit_indices),
                    "score_row_ids_sha256": _hash_index_array(score_indices),
                    "selected_c": float(selected["c_value"]),
                    "selected_inner_auc": float(selected["inner_auc"]),
                    "candidate_scores": candidate_scores,
                    "coefficient": coefficient.tolist(),
                    "intercept": intercept.tolist(),
                }
            )
    if not np.all(write_counts == 1) or not np.isfinite(oof).all():
        raise RuntimeError("Jigsaw stacker OOF predictions were not written exactly once")
    if not np.isfinite(test_by_fold).all():
        raise RuntimeError("Jigsaw stacker fold-specific test prediction is incomplete")
    averaged_test = np.mean(test_by_fold, axis=0)
    ranked_test = np.column_stack(
        [_fractional_rank(averaged_test[:, label]) for label in range(averaged_test.shape[1])]
    )
    contract = {
        "schema": "evomind.mlebench_lite.jigsaw_stacker.v2",
        "feature_set": JIGSAW_STACKER_FEATURE_SET,
        "feature_names": list(names),
        "feature_count": len(names),
        "target_columns": list(target_names),
        "c_grid": list(grid),
        "class_weight": JIGSAW_STACKER_CLASS_WEIGHT,
        "max_iter": int(max_iter),
        "outer_folds": unique_folds,
        "records": records,
        "test_aggregation": "matching_outer_fold_stacker_probability_average_then_rank",
        "exact_once_oof": True,
        "private_labels_used": False,
    }
    return oof, ranked_test, write_counts, contract


def validate_jigsaw_prediction_provenance(
    *,
    target_columns: Sequence[str],
    fold_assignment: np.ndarray,
    base_write_counts: dict[str, np.ndarray],
    stacker_write_counts: np.ndarray,
    stacker_contract: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed unless base and stack predictions prove exact-once OOF writes."""

    targets = tuple(str(value) for value in target_columns)
    folds = np.asarray(fold_assignment, dtype=np.int16).reshape(-1)
    expected_shape = (len(folds), len(targets))
    if targets != JIGSAW_TARGET_COLUMNS:
        raise RuntimeError("Jigsaw provenance target order is not the exact six-column contract")
    if not len(folds) or np.any(folds < 0):
        raise RuntimeError("Jigsaw provenance fold assignment is incomplete")
    coverage: dict[str, bool] = {}
    for channel, raw in base_write_counts.items():
        values = np.asarray(raw)
        coverage[channel] = bool(values.shape == expected_shape and np.all(values == 1))
    stack_counts = np.asarray(stacker_write_counts)
    coverage["stacker"] = bool(
        stack_counts.shape == expected_shape and np.all(stack_counts == 1)
    )
    if not coverage or not all(coverage.values()):
        raise RuntimeError("Jigsaw prediction provenance failed exact-once coverage")
    unique_folds = sorted(int(value) for value in np.unique(folds))
    records = list(stacker_contract.get("records") or [])
    if len(records) != len(unique_folds) * len(targets):
        raise RuntimeError("Jigsaw stacker provenance record count is incomplete")
    for record in records:
        score_fold = int(record["score_fold"])
        target_index = int(record["target_index"])
        fit_folds = [int(value) for value in record["fit_folds"]]
        expected_fit = np.flatnonzero(folds != score_fold)
        expected_score = np.flatnonzero(folds == score_fold)
        if (
            record["target"] != targets[target_index]
            or score_fold in fit_folds
            or sorted(fit_folds) != [value for value in unique_folds if value != score_fold]
            or int(record["test_source_fold"]) != score_fold
            or record["fit_row_ids_sha256"] != _hash_index_array(expected_fit)
            or record["score_row_ids_sha256"] != _hash_index_array(expected_score)
        ):
            raise RuntimeError("Jigsaw stacker provenance contains a fold leakage mismatch")
    return {
        "schema": "evomind.mlebench_lite.jigsaw_prediction_provenance.v2",
        "target_columns": list(targets),
        "folds": unique_folds,
        "channel_exact_once": coverage,
        "stacker_records": len(records),
        "meta_fit_score_disjoint": True,
        "fold_matched_test_sources": True,
        "private_labels_used": False,
        "passed": True,
    }


def save_jigsaw_prediction_bundle(
    task_dir: Path,
    *,
    target_columns: Sequence[str],
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    truth: np.ndarray,
    fold_assignment: np.ndarray,
    oof_word: np.ndarray,
    oof_char: np.ndarray,
    oof_baseline_blend: np.ndarray,
    test_word_by_fold: np.ndarray,
    test_char_by_fold: np.ndarray,
    test_blend_by_fold: np.ndarray,
    oof_stacker: np.ndarray,
    test_stacker: np.ndarray,
    base_write_counts: dict[str, np.ndarray],
    stacker_write_counts: np.ndarray,
    stacker_contract: dict[str, Any],
    provenance: dict[str, Any],
    promotion_gate: dict[str, Any],
) -> dict[str, Any]:
    """Write an immutable, content-addressed Jigsaw prediction/provenance bundle."""

    task_dir.mkdir(parents=True, exist_ok=True)
    target_names = tuple(str(value) for value in target_columns)
    channel_names = ("word", "char_wb", "baseline_rank_blend")
    oof_by_channel = np.stack(
        [oof_word, oof_char, oof_baseline_blend], axis=0
    ).astype(np.float64, copy=False)
    test_by_fold_channel = np.stack(
        [test_word_by_fold, test_char_by_fold, test_blend_by_fold], axis=1
    ).astype(np.float64, copy=False)
    temporary = task_dir / "jigsaw_prediction_bundle_v2.tmp.npz"
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            oof_by_channel=oof_by_channel,
            test_by_fold_channel=test_by_fold_channel,
            oof_stacker=np.asarray(oof_stacker, dtype=np.float64),
            test_stacker=np.asarray(test_stacker, dtype=np.float64),
            truth=np.asarray(truth, dtype=np.int8),
            fold=np.asarray(fold_assignment, dtype=np.int16),
            train_id=np.asarray(train_ids).astype(str),
            test_id=np.asarray(test_ids).astype(str),
            target_columns=np.asarray(target_names),
            channel_names=np.asarray(channel_names),
            feature_names=np.asarray(stacker_contract["feature_names"]),
            word_write_counts=np.asarray(base_write_counts["word"], dtype=np.uint8),
            char_write_counts=np.asarray(base_write_counts["char_wb"], dtype=np.uint8),
            baseline_blend_write_counts=np.asarray(
                base_write_counts["rank_blend"], dtype=np.uint8
            ),
            stacker_write_counts=np.asarray(stacker_write_counts, dtype=np.uint8),
        )
    bundle_sha256 = hashlib.sha256(temporary.read_bytes()).hexdigest()
    bundle_path = task_dir / f"jigsaw_prediction_bundle_v2_{bundle_sha256}.npz"
    temporary.replace(bundle_path)
    array_hashes = {
        "oof_by_channel": _sha256_array(oof_by_channel),
        "test_by_fold_channel": _sha256_array(test_by_fold_channel),
        "oof_stacker": _sha256_array(np.asarray(oof_stacker, dtype=np.float64)),
        "test_stacker": _sha256_array(np.asarray(test_stacker, dtype=np.float64)),
        "truth": _sha256_array(np.asarray(truth, dtype=np.int8)),
        "fold": _sha256_array(np.asarray(fold_assignment, dtype=np.int16)),
    }
    manifest = {
        "schema": "evomind.mlebench_lite.jigsaw_prediction_bundle.v2",
        "bundle": bundle_path.name,
        "bundle_sha256": bundle_sha256,
        "source_file": Path(__file__).name,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "target_columns": list(target_names),
        "channel_names": list(channel_names),
        "train_rows": int(len(train_ids)),
        "test_rows": int(len(test_ids)),
        "array_sha256": array_hashes,
        "stacker_contract": stacker_contract,
        "prediction_provenance": provenance,
        "promotion_gate": promotion_gate,
        "private_labels_used": False,
        "claim_boundary": "Cross-fitted public-train promotion evidence is not an official medal.",
    }
    manifest_path = task_dir / f"jigsaw_prediction_bundle_v2_{bundle_sha256}.manifest.json"
    manifest_temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest_temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_temporary.replace(manifest_path)
    return {
        "schema": manifest["schema"],
        "bundle": bundle_path.name,
        "bundle_sha256": bundle_sha256,
        "manifest": manifest_path.name,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "passed": True,
    }


def build_jigsaw_promotion_gate(
    contract: dict[str, Any],
    *,
    target_columns: Iterable[str],
    truth: np.ndarray,
    fold_assignment: np.ndarray,
    oof_word: np.ndarray,
    oof_char: np.ndarray,
    oof_blend: np.ndarray,
    declared_oof_channels: dict[str, np.ndarray] | None = None,
    candidate_name: str = "rank_blend",
    oof_write_counts: dict[str, np.ndarray] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind Jigsaw promotion to exact, fold-clean six-label OOF evidence."""

    columns = tuple(str(value) for value in target_columns)
    labels = np.asarray(truth, dtype=np.int8)
    folds = np.asarray(fold_assignment, dtype=np.int16).reshape(-1)
    word = np.asarray(oof_word, dtype=np.float64)
    char = np.asarray(oof_char, dtype=np.float64)
    blend = np.asarray(oof_blend, dtype=np.float64)
    expected_shape = (len(folds), len(columns))
    if labels.shape != expected_shape or any(
        values.shape != expected_shape for values in (word, char, blend)
    ):
        raise RuntimeError("Jigsaw promotion arrays do not share the target-column contract")

    complete_oof = bool(
        len(folds) > 0
        and np.all(folds >= 0)
        and all(np.isfinite(values).all() for values in (word, char, blend))
    )
    unique_folds = sorted(int(value) for value in np.unique(folds[folds >= 0]))
    label_scoreable_by_fold = {
        str(fold): {
            columns[index]: bool(np.unique(labels[folds == fold, index]).size == 2)
            for index in range(len(columns))
        }
        for fold in unique_folds
    }
    every_label_scoreable = bool(
        unique_folds
        and all(all(values.values()) for values in label_scoreable_by_fold.values())
    )

    def label_auc(values: np.ndarray) -> list[float]:
        return [
            (
                float(compute_metric("roc_auc", labels[:, index], values[:, index]))
                if np.unique(labels[:, index]).size == 2
                and np.isfinite(values[:, index]).all()
                else math.nan
            )
            for index in range(len(columns))
        ]

    def fold_clean_rank(values: np.ndarray) -> np.ndarray:
        normalized = np.full_like(values, np.nan, dtype=np.float64)
        for fold in unique_folds:
            held_out = folds == fold
            for index in range(len(columns)):
                normalized[held_out, index] = _fractional_rank(values[held_out, index])
        return normalized

    ranked_word = fold_clean_rank(word)
    ranked_char = fold_clean_rank(char)
    word_auc = label_auc(ranked_word)
    char_auc = label_auc(ranked_char)
    ranked_blend = fold_clean_rank(blend)
    blend_auc = label_auc(ranked_blend)
    declared = {"word": word, "char_wb": char}
    for name, values in (declared_oof_channels or {}).items():
        if name == candidate_name:
            raise RuntimeError("Jigsaw candidate cannot also be a declared base channel")
        array = np.asarray(values, dtype=np.float64)
        if array.shape != expected_shape or not np.isfinite(array).all():
            raise RuntimeError(f"Jigsaw declared channel {name!r} violates the OOF contract")
        declared[str(name)] = array
    declared_label_auc = {
        name: label_auc(fold_clean_rank(values)) for name, values in declared.items()
    }
    declared_mean_auc = {
        name: float(np.mean(scores)) for name, scores in declared_label_auc.items()
    }
    word_mean = declared_mean_auc["word"]
    char_mean = declared_mean_auc["char_wb"]
    blend_mean = float(np.mean(blend_auc))
    strongest_base_name, strongest_base_mean = max(
        declared_mean_auc.items(), key=lambda item: item[1]
    )
    blend_improvement = blend_mean - strongest_base_mean
    minimum_improvement = float(contract["minimum_blend_improvement"])
    exact_once_coverage = True
    coverage_evidence: dict[str, bool] = {}
    if oof_write_counts is not None:
        for name, values in oof_write_counts.items():
            array = np.asarray(values)
            coverage_evidence[str(name)] = bool(
                array.shape == expected_shape and np.all(array == 1)
            )
        exact_once_coverage = bool(coverage_evidence and all(coverage_evidence.values()))
    provenance_valid = bool(provenance is None or provenance.get("passed") is True)
    return wave0.build_metric_promotion_gate(
        name=str(contract["name"]),
        metric=str(contract["metric"]),
        direction=str(contract["direction"]),
        score=blend_mean,
        threshold=float(contract["threshold"]),
        extra_checks={
            "exact_six_target_columns": columns == JIGSAW_TARGET_COLUMNS,
            "complete_cross_fitted_oof_coverage": complete_oof,
            "exact_once_oof_write_coverage": exact_once_coverage,
            "every_label_scoreable_in_every_fold": every_label_scoreable,
            "minimum_cross_fitted_blend_improvement": bool(
                math.isfinite(blend_improvement)
                and blend_improvement >= minimum_improvement
            ),
            "prediction_provenance_validated": provenance_valid,
            "private_labels_unused": True,
        },
        evidence={
            "target_columns": list(columns),
            "per_label_word_auc": dict(zip(columns, word_auc, strict=True)),
            "per_label_char_auc": dict(zip(columns, char_auc, strict=True)),
            "per_label_cross_fitted_candidate_auc": dict(
                zip(columns, blend_auc, strict=True)
            ),
            "candidate_name": candidate_name,
            "per_label_declared_channel_auc": {
                name: dict(zip(columns, scores, strict=True))
                for name, scores in declared_label_auc.items()
            },
            "declared_channel_unweighted_mean_auc": declared_mean_auc,
            "word_unweighted_mean_auc": word_mean,
            "char_unweighted_mean_auc": char_mean,
            "strongest_base_name": strongest_base_name,
            "strongest_base_unweighted_mean_auc": strongest_base_mean,
            "cross_fitted_candidate_unweighted_mean_auc": blend_mean,
            "cross_fitted_candidate_improvement": blend_improvement,
            # Compatibility aliases retained for existing report consumers.
            "per_label_cross_fitted_blend_auc": dict(
                zip(columns, blend_auc, strict=True)
            ),
            "cross_fitted_blend_unweighted_mean_auc": blend_mean,
            "cross_fitted_blend_improvement": blend_improvement,
            "minimum_blend_improvement": minimum_improvement,
            "label_scoreable_by_fold": label_scoreable_by_fold,
            "exact_once_oof_write_coverage": coverage_evidence,
            "prediction_provenance": provenance,
            "private_labels_used": False,
            "promotion_normalization": (
                "same_fold_clean_fractional_rank_for_all_declared_channels_and_candidate"
            ),
        },
    )


def histopath_duplicate_promotion_checks(
    *,
    groups: np.ndarray | None,
    split_strategy: str,
    group_overlap_by_fold: dict[str, int],
    duplicate_contract: dict[str, Any],
) -> dict[str, bool]:
    return {
        "exact_image_hash_grouped_split": bool(
            groups is not None and "group" in split_strategy
        ),
        "duplicate_group_disjoint": bool(
            groups is not None and not any(group_overlap_by_fold.values())
        ),
        "zero_conflicting_duplicate_hash_groups": bool(
            duplicate_contract.get("conflicting_duplicate_hash_groups") == 0
        ),
    }


def run_jigsaw(args: argparse.Namespace, task_dir: Path, logger: Any) -> dict[str, Any]:
    from joblib import dump
    from sklearn.feature_extraction.text import TfidfVectorizer

    cid = "jigsaw-toxic-comment-classification-challenge"
    resolved = resolve_competition(cid, args.data_root)
    train = pd.read_csv(resolved.public_dir / "train.csv")
    test = pd.read_csv(resolved.public_dir / "test.csv")
    sample = pd.read_csv(resolved.sample_submission_path)
    targets = list(sample.columns[1:])
    train_text = train["comment_text"].fillna("").astype(str).reset_index(drop=True)
    test_text = test["comment_text"].fillna("").astype(str).reset_index(drop=True)
    started = time.perf_counter()
    fold_count = max(2, min(args.wave2_jigsaw_folds, len(train)))
    truth = train[targets].to_numpy(dtype=np.int8)
    splits, iterative_assignment = make_multilabel_stratified_folds(
        truth,
        requested_folds=fold_count,
        seed=args.seed,
    )
    fold_count = len(splits)
    oof_word = np.zeros((len(train), len(targets)), dtype=np.float64)
    oof_char = np.zeros_like(oof_word)
    word_write_counts = np.zeros_like(oof_word, dtype=np.uint8)
    char_write_counts = np.zeros_like(oof_word, dtype=np.uint8)
    fold_assignment = np.full(len(train), -1, dtype=np.int16)
    test_word = np.zeros((len(test), len(targets)), dtype=np.float64)
    test_char = np.zeros_like(test_word)
    test_word_by_fold = np.zeros((fold_count, len(test), len(targets)), dtype=np.float64)
    test_char_by_fold = np.zeros_like(test_word_by_fold)
    feature_counts: list[dict[str, int]] = []
    fold_artifacts: list[str] = []
    fold_label_counts: list[dict[str, Any]] = []
    for fold, (train_indices, valid_indices) in enumerate(splits):
        fold_assignment[valid_indices] = fold
        word = TfidfVectorizer(
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w{1,}",
            ngram_range=(1, 2),
            min_df=2,
            max_features=args.wave2_jigsaw_word_features,
            sublinear_tf=True,
            dtype=np.float32,
        )
        char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 6),
            min_df=2,
            max_features=args.wave2_jigsaw_char_features,
            sublinear_tf=True,
            dtype=np.float32,
        )
        word_train = word.fit_transform(train_text.iloc[train_indices])
        word_valid = word.transform(train_text.iloc[valid_indices])
        word_test = word.transform(test_text)
        char_train = char.fit_transform(train_text.iloc[train_indices])
        char_valid = char.transform(train_text.iloc[valid_indices])
        char_test = char.transform(test_text)
        feature_counts.append({"fold": fold, "word": word_train.shape[1], "char": char_train.shape[1]})
        word_models: dict[str, Any] = {}
        char_models: dict[str, Any] = {}
        word_ratios: dict[str, np.ndarray] = {}
        char_ratios: dict[str, np.ndarray] = {}
        fitted = fit_jigsaw_nbsvm_channels(
            word_train,
            char_train,
            truth[train_indices],
            target_names=targets,
            c_value=args.wave2_jigsaw_nbsvm_c,
            max_iter=args.wave2_jigsaw_max_iter,
            seed_base=args.seed + fold * 101,
            workers=args.wave2_jigsaw_fit_workers,
        )
        by_key = {(item["channel"], item["index"]): item for item in fitted}
        for index, target in enumerate(targets):
            word_fit = by_key[("word", index)]
            char_fit = by_key[("char", index)]
            word_model, word_ratio = word_fit["model"], word_fit["ratio"]
            char_model, char_ratio = char_fit["model"], char_fit["ratio"]
            oof_word[valid_indices, index] = word_model.predict_proba(word_valid.multiply(word_ratio))[:, 1]
            oof_char[valid_indices, index] = char_model.predict_proba(char_valid.multiply(char_ratio))[:, 1]
            word_write_counts[valid_indices, index] += 1
            char_write_counts[valid_indices, index] += 1
            fold_test_word = word_model.predict_proba(word_test.multiply(word_ratio))[:, 1]
            fold_test_char = char_model.predict_proba(char_test.multiply(char_ratio))[:, 1]
            test_word_by_fold[fold, :, index] = fold_test_word
            test_char_by_fold[fold, :, index] = fold_test_char
            test_word[:, index] += fold_test_word / fold_count
            test_char[:, index] += fold_test_char / fold_count
            word_models[target], char_models[target] = word_model, char_model
            word_ratios[target], char_ratios[target] = word_ratio, char_ratio
        artifact_path = task_dir / f"jigsaw_fold{fold}_nbsvm.joblib"
        dump(
            {
                "fold": fold,
                "train_indices": train_indices,
                "validation_indices": valid_indices,
                "targets": targets,
                "word_vectorizer": word,
                "char_vectorizer": char,
                "word_models": word_models,
                "char_models": char_models,
                "word_ratios": word_ratios,
                "char_ratios": char_ratios,
            },
            artifact_path,
            compress=3,
        )
        fold_artifacts.append(artifact_path.name)
        fold_label_counts.append({
            "fold": fold,
            "rows": int(len(valid_indices)),
            "positive_by_label": {
                target: int(truth[valid_indices, index].sum())
                for index, target in enumerate(targets)
            },
        })
        logger.info("[%s] fold=%d/%d word=%d char=%d", cid, fold + 1, fold_count,
                    word_train.shape[1], char_train.shape[1])
    if np.any(fold_assignment < 0):
        raise RuntimeError("Jigsaw OOF coverage is incomplete")
    if not np.array_equal(fold_assignment, iterative_assignment):
        raise RuntimeError("Jigsaw fold assignment diverged from the iterative split manifest")
    if not np.all(word_write_counts == 1) or not np.all(char_write_counts == 1):
        raise RuntimeError("Jigsaw base OOF predictions were not written exactly once")
    valid_prediction, test_prediction, fold_blend_weights, final_blend_weights = (
        cross_fit_multilabel_rank_blend(
            oof_word,
            oof_char,
            test_word_by_fold,
            test_char_by_fold,
            truth,
            fold_assignment,
        )
    )
    baseline_valid_prediction = valid_prediction
    baseline_test_prediction = test_prediction
    test_blend_by_fold = build_jigsaw_fold_test_blends(
        test_word_by_fold,
        test_char_by_fold,
        fold_blend_weights,
    )
    baseline_write_counts = np.ones_like(word_write_counts, dtype=np.uint8)
    stacker_disabled = bool(getattr(args, "wave2_jigsaw_disable_stacker", False))
    stacker_contract: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    stacker_write_counts = np.zeros_like(word_write_counts, dtype=np.uint8)
    meta_feature_names: list[str] = []
    if not stacker_disabled:
        meta_features, meta_feature_names = build_jigsaw_meta_features(
            oof_word,
            oof_char,
            baseline_valid_prediction,
            fold_assignment,
            target_columns=targets,
        )
        test_meta_features, test_feature_names = build_jigsaw_test_meta_features_by_fold(
            test_word_by_fold,
            test_char_by_fold,
            test_blend_by_fold,
            target_columns=targets,
        )
        if meta_feature_names != test_feature_names:
            raise RuntimeError("Jigsaw train/test stacker feature contracts diverged")
        valid_prediction, test_prediction, stacker_write_counts, stacker_contract = (
            cross_fit_jigsaw_label_stacker(
                meta_features,
                test_meta_features,
                truth,
                fold_assignment,
                target_columns=targets,
                feature_names=meta_feature_names,
                c_grid=_resolve_jigsaw_stacker_c_grid(
                    getattr(args, "wave2_jigsaw_stacker_c_grid", None)
                ),
                max_iter=int(getattr(args, "wave2_jigsaw_stacker_max_iter", 300)),
                seed=args.seed + 70_001,
            )
        )
        provenance = validate_jigsaw_prediction_provenance(
            target_columns=targets,
            fold_assignment=fold_assignment,
            base_write_counts={
                "word": word_write_counts,
                "char_wb": char_write_counts,
                "rank_blend": baseline_write_counts,
            },
            stacker_write_counts=stacker_write_counts,
            stacker_contract=stacker_contract,
        )
    per_label: dict[str, Any] = {}
    for index, target in enumerate(targets):
        per_label[target] = {
            "word_auc": compute_metric("roc_auc", truth[:, index], oof_word[:, index]),
            "char_auc": compute_metric("roc_auc", truth[:, index], oof_char[:, index]),
            "cross_fitted_blend_auc": compute_metric("roc_auc", truth[:, index], valid_prediction[:, index]),
            "baseline_cross_fitted_rank_blend_auc": compute_metric(
                "roc_auc", truth[:, index], baseline_valid_prediction[:, index]
            ),
            "final_word_weight": float(final_blend_weights[index]),
        }
    cv_score = compute_metric("mean_columnwise_roc_auc", truth, valid_prediction)
    promotion_gate = build_jigsaw_promotion_gate(
        WAVE2_PROMOTION_CONTRACTS[cid],
        target_columns=targets,
        truth=truth,
        fold_assignment=fold_assignment,
        oof_word=oof_word,
        oof_char=oof_char,
        oof_blend=valid_prediction,
        declared_oof_channels=(
            {"baseline_rank_blend": baseline_valid_prediction}
            if not stacker_disabled
            else None
        ),
        candidate_name=("hybrid30_cross_label_stacker" if not stacker_disabled else "rank_blend"),
        oof_write_counts={
            "word": word_write_counts,
            "char_wb": char_write_counts,
            "candidate": (
                stacker_write_counts if not stacker_disabled else baseline_write_counts
            ),
        },
        provenance=provenance,
    )
    prediction_bundle = None
    if not stacker_disabled:
        if stacker_contract is None or provenance is None:
            raise RuntimeError("Jigsaw stacker bundle lacks its provenance contract")
        prediction_bundle = save_jigsaw_prediction_bundle(
            task_dir,
            target_columns=targets,
            train_ids=train["id"].astype(str).to_numpy(),
            test_ids=test["id"].astype(str).to_numpy(),
            truth=truth,
            fold_assignment=fold_assignment,
            oof_word=oof_word,
            oof_char=oof_char,
            oof_baseline_blend=baseline_valid_prediction,
            test_word_by_fold=test_word_by_fold,
            test_char_by_fold=test_char_by_fold,
            test_blend_by_fold=test_blend_by_fold,
            oof_stacker=valid_prediction,
            test_stacker=test_prediction,
            base_write_counts={
                "word": word_write_counts,
                "char_wb": char_write_counts,
                "rank_blend": baseline_write_counts,
            },
            stacker_write_counts=stacker_write_counts,
            stacker_contract=stacker_contract,
            provenance=provenance,
            promotion_gate=promotion_gate,
        )
    sample = align_multilabel_submission_by_id(
        sample,
        test,
        test_prediction,
        id_column="id",
        target_columns=targets,
    )
    np.savez_compressed(
        task_dir / "jigsaw_sparse_oof_and_test.npz",
        oof_word=oof_word,
        oof_char=oof_char,
        oof_blend=baseline_valid_prediction,
        oof_stacker=valid_prediction,
        test_word=test_word,
        test_char=test_char,
        test_word_by_fold=test_word_by_fold,
        test_char_by_fold=test_char_by_fold,
        test_blend=baseline_test_prediction,
        test_stacker=test_prediction,
        test_blend_by_fold=test_blend_by_fold,
        truth=truth,
        fold=fold_assignment,
        word_write_counts=word_write_counts,
        char_write_counts=char_write_counts,
        stacker_write_counts=stacker_write_counts,
        fold_blend_weights=fold_blend_weights,
        final_blend_weights=final_blend_weights,
        train_id=train["id"].astype(str).to_numpy(),
        test_id=test["id"].astype(str).to_numpy(),
    )
    budget = {
        "seed": args.seed, "train_rows": len(train), "test_rows": len(test),
        "folds": fold_count, "feature_counts": feature_counts,
        "nbsvm_c": args.wave2_jigsaw_nbsvm_c,
        "fit_workers": args.wave2_jigsaw_fit_workers,
        "split_strategy": "iterative_multilabel_stratification",
        "fold_label_counts": fold_label_counts,
        "fold_artifacts": fold_artifacts,
        "stacker_enabled": not stacker_disabled,
        "stacker_contract": stacker_contract,
        "prediction_provenance": provenance,
        "prediction_bundle": prediction_bundle,
        "promotion_gate": promotion_gate,
    }
    return wave0.finalize_scored_task(
        competition_id=cid, submission=sample, cv_score=cv_score, args=args, task_dir=task_dir,
        budget=budget, promotion_gate=promotion_gate,
        extra={"runtime_seconds_model": time.perf_counter() - started,
                              "model_family": (
                                  "NB_SVM_word_char_Hybrid30_nested_cross_label_stacker"
                                  if not stacker_disabled
                                  else "NB_SVM_word_char_iterative_cross_fitted_rank_blend"
                              ),
                              "per_label_oof": per_label, "budget": budget},
    )


def _read_audio(path: Path) -> tuple[int, np.ndarray]:
    try:
        import soundfile as sf
        values, rate = sf.read(str(path), always_2d=False)
        return int(rate), np.asarray(values)
    except Exception:
        if path.suffix.lower() in {".aif", ".aiff"}:
            import aifc
            with aifc.open(str(path), "rb") as handle:
                rate = handle.getframerate()
                channels = handle.getnchannels()
                width = handle.getsampwidth()
                raw = handle.readframes(handle.getnframes())
            dtype = ">i2" if width == 2 else ">i1"
            values = np.frombuffer(raw, dtype=dtype).astype(np.float32)
            if channels > 1:
                values = values.reshape(-1, channels).mean(axis=1)
            return int(rate), values
        from scipy.io import wavfile
        rate, values = wavfile.read(path)
        return int(rate), np.asarray(values)


AUDIO_FEATURE_WIDTH = 1572


def resample_audio(values: np.ndarray, source_rate: int, target_rate: int = 16_000) -> np.ndarray:
    """Polyphase-resample audio so features are comparable across source rates."""

    from scipy.signal import resample_poly

    values = np.asarray(values, dtype=np.float32)
    if source_rate <= 0:
        raise ValueError("Audio sample rate must be positive")
    if source_rate == target_rate or not len(values):
        return values
    divisor = math.gcd(int(source_rate), int(target_rate))
    return np.asarray(
        resample_poly(values, target_rate // divisor, source_rate // divisor),
        dtype=np.float32,
    )


@lru_cache(maxsize=8)
def _mel_filterbank(rate: int, n_fft: int, n_mels: int = 128) -> np.ndarray:
    def hz_to_mel(value: np.ndarray | float) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + np.asarray(value) / 700.0)

    def mel_to_hz(value: np.ndarray | float) -> np.ndarray:
        return 700.0 * (10.0 ** (np.asarray(value) / 2595.0) - 1.0)

    mel_points = np.linspace(hz_to_mel(20.0), hz_to_mel(rate / 2.0), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mel_points) / rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    bank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for index in range(n_mels):
        left, center, right = int(bins[index]), int(bins[index + 1]), int(bins[index + 2])
        center = max(center, left + 1)
        right = max(right, center + 1)
        right = min(right, n_fft // 2)
        if center > left:
            bank[index, left:center] = np.linspace(0.0, 1.0, center - left, endpoint=False)
        if right > center:
            bank[index, center:right] = np.linspace(1.0, 0.0, right - center, endpoint=False)
    return bank


def _frame_audio(values: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if len(values) < frame_size:
        values = np.pad(values, (0, frame_size - len(values)))
    frame_count = 1 + max(0, (len(values) - frame_size) // hop_size)
    windows = np.lib.stride_tricks.sliding_window_view(values, frame_size)[::hop_size]
    return np.asarray(windows[:frame_count], dtype=np.float32).copy()


def extract_audio_feature_vector(
    values: np.ndarray,
    source_rate: int,
    *,
    target_rate: int = 16_000,
) -> np.ndarray:
    """Build a fixed 1572-wide log-mel, temporal and spectral descriptor."""

    raw = np.asarray(values)
    if raw.ndim > 1:
        raw = raw.astype(np.float32).mean(axis=1)
    if not len(raw):
        return np.zeros(AUDIO_FEATURE_WIDTH, dtype=np.float32)
    if np.issubdtype(raw.dtype, np.integer):
        raw = raw.astype(np.float32) / max(1.0, float(np.iinfo(raw.dtype).max))
    else:
        raw = raw.astype(np.float32)
    if not np.isfinite(raw).all():
        raise RuntimeError("Audio contains non-finite samples")
    original_rms = float(np.sqrt(np.mean(raw.astype(np.float64) ** 2)))
    original_peak = float(np.max(np.abs(raw)))
    values = resample_audio(raw, source_rate, target_rate)
    values -= float(values.mean())
    robust_scale = float(np.percentile(np.abs(values), 99))
    if robust_scale > 1e-8:
        values /= robust_scale
    values = np.clip(values, -8.0, 8.0)
    signs = np.signbit(values)
    zcr_global = float(np.mean(signs[1:] != signs[:-1])) if len(values) > 1 else 0.0
    absolute = np.abs(values)
    summary = np.asarray([
        len(values) / target_rate,
        original_rms,
        original_peak,
        float(values.mean()),
        float(values.std()),
        float(absolute.mean()),
        float(np.percentile(absolute, 50)),
        float(np.percentile(absolute, 90)),
        float(np.percentile(absolute, 99)),
        zcr_global,
        float(original_peak / max(original_rms, 1e-8)),
        float(np.mean(absolute < 0.01)),
    ], dtype=np.float32)

    n_fft, hop = 1024, 320
    frames = _frame_audio(values, n_fft, hop)
    windowed = frames * np.hanning(n_fft).astype(np.float32)
    spectrum = np.fft.rfft(windowed, axis=1)
    power = np.abs(spectrum).astype(np.float64) ** 2
    mel_power = power @ _mel_filterbank(target_rate, n_fft).T
    logmel = np.log1p(mel_power).astype(np.float32)
    global_mel = np.concatenate([
        logmel.mean(axis=0),
        logmel.std(axis=0),
        np.percentile(logmel, 10, axis=0),
        np.percentile(logmel, 50, axis=0),
        np.percentile(logmel, 90, axis=0),
    ])
    delta = np.diff(logmel, axis=0, prepend=logmel[:1])
    delta_summary = np.concatenate([delta.mean(axis=0), delta.std(axis=0), np.mean(np.abs(delta), axis=0)])
    temporal = np.concatenate([
        segment.mean(axis=0) if len(segment) else np.zeros(logmel.shape[1], dtype=np.float32)
        for segment in np.array_split(logmel, 4, axis=0)
    ])

    frequencies = np.fft.rfftfreq(n_fft, 1.0 / target_rate)
    total_power = np.maximum(power.sum(axis=1), 1e-12)
    distribution = power / total_power[:, None]
    centroid = distribution @ frequencies
    bandwidth = np.sqrt(np.sum(distribution * (frequencies[None, :] - centroid[:, None]) ** 2, axis=1))
    cumulative = np.cumsum(distribution, axis=1)
    rolloff85 = frequencies[np.argmax(cumulative >= 0.85, axis=1)]
    rolloff95 = frequencies[np.argmax(cumulative >= 0.95, axis=1)]
    flatness = np.exp(np.mean(np.log(power + 1e-12), axis=1)) / np.maximum(power.mean(axis=1), 1e-12)
    entropy = -np.sum(distribution * np.log(np.maximum(distribution, 1e-12)), axis=1)
    flux = np.sqrt(np.mean(np.diff(distribution, axis=0, prepend=distribution[:1]) ** 2, axis=1))
    frame_rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    frame_zcr = np.mean(np.signbit(frames[:, 1:]) != np.signbit(frames[:, :-1]), axis=1)
    low = distribution[:, frequencies < 1_000].sum(axis=1)
    middle = distribution[:, (frequencies >= 1_000) & (frequencies < 4_000)].sum(axis=1)
    high = distribution[:, frequencies >= 4_000].sum(axis=1)
    spectral_series = (centroid, bandwidth, rolloff85, rolloff95, flatness, entropy,
                       flux, frame_rms, frame_zcr, low, middle, high)
    spectral_summary = np.asarray(
        [value for series in spectral_series for value in (float(np.mean(series)), float(np.std(series)))],
        dtype=np.float32,
    )
    result = np.concatenate([summary, global_mel, delta_summary, temporal, spectral_summary]).astype(np.float32)
    if result.shape != (AUDIO_FEATURE_WIDTH,):
        raise RuntimeError(f"Unexpected audio feature width: {result.shape}")
    if not np.isfinite(result).all():
        raise RuntimeError("Audio feature extraction produced non-finite values")
    return result


def audio_features(path: Path) -> np.ndarray:
    rate, values = _read_audio(path)
    return extract_audio_feature_vector(values, rate)


def _parallel_audio_features(paths: list[Path], workers: int) -> np.ndarray:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        rows = list(pool.map(audio_features, paths, chunksize=8))
    return np.stack(rows)


def _path_manifest_digest(paths: list[Path]) -> str:
    """Fingerprint audio path metadata for the cached waveform hash manifest."""

    digest = hashlib.sha256(AUDIO_FEATURE_CACHE_VERSION.encode("ascii"))
    for index, path in enumerate(paths):
        stat = path.stat()
        digest.update(
            f"{index}\0{path.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8")
        )
    return digest.hexdigest()


def load_or_compute_audio_path_manifest(
    paths: list[Path],
    workers: int,
    cache_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Cache full waveform SHA256 values behind a cheap path metadata manifest."""

    if not paths:
        raise RuntimeError("Audio path manifest requires at least one input path")
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_fingerprint = _path_manifest_digest(paths)
    manifest_path = (
        cache_dir
        / f"{AUDIO_FEATURE_CACHE_VERSION}_{len(paths)}_{metadata_fingerprint[:20]}_paths.json"
    )
    cache_hit = False
    payload: dict[str, Any] | None = None
    if manifest_path.is_file():
        candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = candidate.get("rows") or []
        if (
            candidate.get("schema") == "evomind.audio_path_manifest.v1"
            and candidate.get("version") == AUDIO_FEATURE_CACHE_VERSION
            and candidate.get("metadata_fingerprint") == metadata_fingerprint
            and len(rows) == len(paths)
            and all(isinstance(row.get("sha256"), str) and len(row["sha256"]) == 64 for row in rows)
        ):
            payload = candidate
            cache_hit = True
    if payload is None:
        hashes = _parallel_sha256(paths, workers).astype(str)
        rows = []
        for index, (path, waveform_hash) in enumerate(zip(paths, hashes, strict=True)):
            stat = path.stat()
            rows.append({
                "index": index,
                "name": path.name,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": str(waveform_hash),
            })
        waveform_digest = hashlib.sha256(
            "\n".join(row["sha256"] for row in rows).encode("ascii")
        ).hexdigest()
        payload = {
            "schema": "evomind.audio_path_manifest.v1",
            "version": AUDIO_FEATURE_CACHE_VERSION,
            "metadata_fingerprint": metadata_fingerprint,
            "waveform_sha256": waveform_digest,
            "rows": rows,
            "path_count": len(rows),
            "created_at": wave0.utc_now(),
        }
        temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(manifest_path)
    return payload, {
        "cache_hit": cache_hit,
        "path": str(manifest_path),
        "metadata_fingerprint": metadata_fingerprint,
        "waveform_sha256": str(payload["waveform_sha256"]),
        "rows": int(payload["path_count"]),
        "workers": max(1, int(workers)),
    }


def load_or_compute_audio_feature_matrix(
    paths: list[Path],
    workers: int,
    cache_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reuse the expensive 1572-wide audio matrix across external seeds."""

    if not paths:
        raise RuntimeError("Audio feature cache requires at least one input path")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path_manifest, path_manifest_contract = load_or_compute_audio_path_manifest(
        paths,
        workers,
        cache_dir / "path_manifests",
    )
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "version": AUDIO_FEATURE_CACHE_VERSION,
                "metadata_fingerprint": path_manifest["metadata_fingerprint"],
                "waveform_sha256": path_manifest["waveform_sha256"],
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    cache_path = cache_dir / f"{AUDIO_FEATURE_CACHE_VERSION}_{len(paths)}_{fingerprint[:20]}.npy"
    cache_hit = cache_path.is_file()
    if not cache_hit:
        matrix = _parallel_audio_features(paths, workers)
        temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, matrix, allow_pickle=False)
        temporary.replace(cache_path)
    matrix = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    if matrix.shape != (len(paths), AUDIO_FEATURE_WIDTH):
        raise RuntimeError(
            f"Cached audio feature matrix has invalid shape {matrix.shape}; "
            f"expected {(len(paths), AUDIO_FEATURE_WIDTH)}"
        )
    if not np.isfinite(matrix).all():
        raise RuntimeError("Cached audio feature matrix contains non-finite values")
    return np.asarray(matrix), {
        "version": AUDIO_FEATURE_CACHE_VERSION,
        "cache_hit": cache_hit,
        "path": str(cache_path),
        "fingerprint": fingerprint,
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "workers": max(1, int(workers)),
        "path_manifest": path_manifest_contract,
    }


def extract_zip_to_shared_cache(archive: Path, cache_root: Path) -> tuple[Path, dict[str, Any]]:
    """Extract a stable archive once and atomically reuse it for later seeds."""

    stat = archive.stat()
    identity = f"{archive.name}\0{stat.st_size}\0{stat.st_mtime_ns}"
    fingerprint = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / f"{archive.stem}_{fingerprint[:20]}"
    marker = target / ".evomind_extract_complete"
    cache_hit = marker.is_file() and marker.read_text(encoding="utf-8") == identity
    if not cache_hit:
        if target.exists():
            if target.parent.resolve() != cache_root.resolve():
                raise RuntimeError("Shared ZIP cache target escaped its cache root")
            shutil.rmtree(target)
        staging = cache_root / f".{target.name}.{os.getpid()}.staging"
        if staging.exists():
            shutil.rmtree(staging)
        try:
            wave0.safe_extract_zip(archive, staging)
            (staging / ".evomind_extract_complete").write_text(identity, encoding="utf-8")
            staging.replace(target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return target, {
        "cache_hit": cache_hit,
        "path": str(target),
        "archive": archive.name,
        "archive_size": int(stat.st_size),
        "fingerprint": fingerprint,
    }


def _parallel_sha256(paths: list[Path], workers: int) -> np.ndarray:
    worker_count = max(1, min(int(workers), len(paths)))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        values = list(pool.map(_sha256_file, paths, chunksize=32))
    return np.asarray(values, dtype=object)


def _file_manifest_entry(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns)


def _parallel_file_manifest(
    paths: list[Path],
    workers: int,
) -> list[tuple[str, int, int]]:
    """Collect stable file metadata without serially stalling large vision runs."""

    worker_count = max(1, min(int(workers), len(paths)))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        return list(pool.map(_file_manifest_entry, paths, chunksize=64))


def build_histopath_duplicate_contract(
    train_paths: list[Path],
    test_paths: list[Path],
    labels: np.ndarray,
    *,
    workers: int,
    cache_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Hash public tiles before CUDA allocation and isolate exact duplicates."""

    target = np.asarray(labels, dtype=np.int8).reshape(-1)
    if len(train_paths) != len(target) or set(np.unique(target).tolist()) != {0, 1}:
        raise RuntimeError("Histopath duplicate hashing requires aligned binary labels")
    all_paths = [*train_paths, *test_paths]
    manifest_digest = hashlib.sha256()
    for resolved_path, size, mtime_ns in _parallel_file_manifest(all_paths, workers):
        manifest_digest.update(resolved_path.encode("utf-8"))
        manifest_digest.update(f"\0{size}\0{mtime_ns}\n".encode("ascii"))
    manifest_sha256 = manifest_digest.hexdigest()
    labels_sha256 = hashlib.sha256(target.tobytes()).hexdigest()
    cache_hit = False
    if cache_path is not None and cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            cache_valid = bool(
                str(cached["manifest_sha256"].item()) == manifest_sha256
                and str(cached["labels_sha256"].item()) == labels_sha256
                and int(cached["train_count"].item()) == len(train_paths)
                and int(cached["test_count"].item()) == len(test_paths)
            )
            if cache_valid:
                train_hashes = cached["train_hashes"].astype(str)
                test_hashes = cached["test_hashes"].astype(str)
                cache_hit = True
    if not cache_hit:
        all_hashes = _parallel_sha256(all_paths, workers).astype(str)
        train_hashes = all_hashes[: len(train_paths)]
        test_hashes = all_hashes[len(train_paths) :]
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            staging = cache_path.with_suffix(".tmp.npz")
            np.savez_compressed(
                staging,
                manifest_sha256=np.asarray(manifest_sha256),
                labels_sha256=np.asarray(labels_sha256),
                train_count=np.asarray(len(train_paths)),
                test_count=np.asarray(len(test_paths)),
                train_hashes=train_hashes,
                test_hashes=test_hashes,
            )
            staging.replace(cache_path)
    grouped = pd.DataFrame({"hash": train_hashes, "label": target}).groupby("hash")["label"]
    label_counts = grouped.nunique()
    conflicts = int((label_counts > 1).sum())
    if conflicts:
        raise RuntimeError(
            f"Histopath exact image hashes contain conflicting labels in {conflicts} groups"
        )
    unanimous_labels = grouped.first().to_dict()
    overrides = np.asarray(
        [float(unanimous_labels.get(value, math.nan)) for value in test_hashes],
        dtype=np.float64,
    )
    return train_hashes, overrides, {
        "hash_algorithm": "sha256_full_tile_bytes",
        "train_rows": len(train_paths),
        "test_rows": len(test_paths),
        "unique_train_image_hashes": int(len(np.unique(train_hashes))),
        "duplicate_train_rows": int(len(train_hashes) - len(np.unique(train_hashes))),
        "conflicting_duplicate_hash_groups": conflicts,
        "exact_train_test_matches": int(np.isfinite(overrides).sum()),
        "exact_test_override": True,
        "cache_hit": cache_hit,
        "cache_path": None if cache_path is None else str(cache_path),
        "source_manifest_sha256": manifest_sha256,
    }


def build_multiclass_image_duplicate_contract(
    train_paths: list[Path],
    test_paths: list[Path],
    labels: np.ndarray,
    classes: Sequence[str],
    *,
    workers: int,
    cache_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Group exact duplicates and build only label-unanimous test overrides.

    Conflicting train labels for identical image bytes stay in one CV group so
    they can never leak across train/validation folds.  Such hashes are excluded
    from train-to-test overrides because no deterministic label can be inferred.
    """

    target = np.asarray(labels, dtype=str).reshape(-1)
    class_names = [str(value) for value in classes]
    if len(train_paths) != len(target) or not class_names or len(set(class_names)) != len(class_names):
        raise RuntimeError("Multiclass duplicate hashing received an invalid label contract")
    unknown = sorted(set(target.tolist()) - set(class_names))
    if unknown:
        raise RuntimeError(f"Multiclass duplicate hashing found unknown labels: {unknown[:5]}")
    all_paths = [*train_paths, *test_paths]
    if not all(path.is_file() for path in all_paths):
        raise FileNotFoundError("Multiclass duplicate hashing found a missing image")
    manifest_digest = hashlib.sha256()
    for resolved_path, size, mtime_ns in _parallel_file_manifest(all_paths, workers):
        manifest_digest.update(resolved_path.encode("utf-8"))
        manifest_digest.update(f"\0{size}\0{mtime_ns}\n".encode("ascii"))
    manifest_sha256 = manifest_digest.hexdigest()
    cache_hit = False
    if cache_path is not None and cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            cache_valid = bool(
                str(cached["manifest_sha256"].item()) == manifest_sha256
                and int(cached["train_count"].item()) == len(train_paths)
                and int(cached["test_count"].item()) == len(test_paths)
            )
            if cache_valid:
                train_hashes = cached["train_hashes"].astype(str)
                test_hashes = cached["test_hashes"].astype(str)
                cache_hit = True
    if not cache_hit:
        all_hashes = _parallel_sha256(all_paths, workers).astype(str)
        train_hashes = all_hashes[: len(train_paths)]
        test_hashes = all_hashes[len(train_paths) :]
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            staging = cache_path.with_suffix(".tmp.npz")
            np.savez_compressed(
                staging,
                manifest_sha256=np.asarray(manifest_sha256),
                train_count=np.asarray(len(train_paths)),
                test_count=np.asarray(len(test_paths)),
                train_hashes=train_hashes,
                test_hashes=test_hashes,
            )
            staging.replace(cache_path)
    grouped = pd.DataFrame({"hash": train_hashes, "label": target}).groupby("hash")["label"]
    distinct_label_counts = grouped.nunique()
    conflicting_hashes = set(
        distinct_label_counts[distinct_label_counts > 1].index.astype(str).tolist()
    )
    conflicts = len(conflicting_hashes)
    unanimous = {
        str(image_hash): str(label)
        for image_hash, label in grouped.first().to_dict().items()
        if str(image_hash) not in conflicting_hashes
    }
    class_index = {value: index for index, value in enumerate(class_names)}
    overrides = np.full((len(test_paths), len(class_names)), np.nan, dtype=np.float64)
    epsilon = 1e-6
    for row, image_hash in enumerate(test_hashes):
        label = unanimous.get(str(image_hash))
        if label is None:
            continue
        overrides[row] = epsilon
        overrides[row, class_index[str(label)]] = 1.0 - epsilon * (len(class_names) - 1)
    conflicting_train_rows = int(
        np.isin(train_hashes, list(conflicting_hashes)).sum()
    ) if conflicting_hashes else 0
    conflicting_test_matches = int(
        np.isin(test_hashes, list(conflicting_hashes)).sum()
    ) if conflicting_hashes else 0
    return train_hashes, overrides, {
        "hash_algorithm": "sha256_full_image_bytes",
        "train_rows": len(train_paths),
        "test_rows": len(test_paths),
        "class_count": len(class_names),
        "unique_train_image_hashes": int(len(np.unique(train_hashes))),
        "duplicate_train_rows": int(len(train_hashes) - len(np.unique(train_hashes))),
        "conflicting_duplicate_hash_groups": conflicts,
        "conflicting_duplicate_train_rows": conflicting_train_rows,
        "conflicting_test_hash_matches": conflicting_test_matches,
        "conflicting_test_overrides": 0,
        "conflicting_hash_policy": "group_together_no_test_override",
        "exact_train_test_matches": int(np.isfinite(overrides).all(axis=1).sum()),
        "exact_test_override": True,
        "cache_hit": cache_hit,
        "cache_path": None if cache_path is None else str(cache_path),
        "source_manifest_sha256": manifest_sha256,
    }


def load_bird_supplemental_features(
    supplemental_dir: Path,
    recording_ids: Iterable[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Align provided MLSP recording/segment features without using hidden labels."""

    requested = np.asarray([int(value) for value in recording_ids], dtype=np.int64)
    if len(np.unique(requested)) != len(requested):
        raise RuntimeError("Bird recording IDs must be unique before feature alignment")
    histogram_path = supplemental_dir / "histogram_of_segments.txt"
    segment_path = supplemental_dir / "segment_features.txt"
    histogram = np.loadtxt(histogram_path, delimiter=",", skiprows=1, ndmin=2)
    segments = np.loadtxt(segment_path, delimiter=",", skiprows=1, ndmin=2)
    if histogram.shape[1] < 2 or segments.shape[1] < 3:
        raise RuntimeError("Bird supplemental feature files have invalid widths")
    if not np.isfinite(histogram).all() or not np.isfinite(segments).all():
        raise RuntimeError("Bird supplemental features contain non-finite values")

    histogram_ids = histogram[:, 0].astype(np.int64)
    if not np.array_equal(histogram[:, 0], histogram_ids):
        raise RuntimeError("Bird histogram recording IDs are not exact integers")
    if len(np.unique(histogram_ids)) != len(histogram_ids):
        raise RuntimeError("Bird histogram features contain duplicate recording IDs")
    histogram_lookup = {int(value): index for index, value in enumerate(histogram_ids)}
    missing_histogram = sorted(set(requested.tolist()) - set(histogram_lookup))
    if missing_histogram:
        raise RuntimeError(
            f"Bird histogram alignment is missing {len(missing_histogram)} recording IDs"
        )

    segment_ids = segments[:, 0].astype(np.int64)
    segment_row_ids = segments[:, 1].astype(np.int64)
    if not np.array_equal(segments[:, 0], segment_ids) or not np.array_equal(
        segments[:, 1], segment_row_ids
    ):
        raise RuntimeError("Bird segment identifiers are not exact integers")
    segment_keys = np.column_stack([segment_ids, segment_row_ids])
    if len(np.unique(segment_keys, axis=0)) != len(segment_keys):
        raise RuntimeError("Bird segment features contain duplicate recording/segment IDs")
    segment_values = segments[:, 2:].astype(np.float32)
    segment_width = int(segment_values.shape[1])
    aggregated: list[np.ndarray] = []
    missing_segment_count = 0
    missing_segment_nonzero_histogram_ids: list[int] = []
    for recording_id in requested:
        values = segment_values[segment_ids == recording_id]
        if not len(values):
            missing_segment_count += 1
            histogram_values = histogram[
                histogram_lookup[int(recording_id)], 1:
            ].astype(np.float32)
            if not np.allclose(histogram_values, 0.0, rtol=0.0, atol=1e-8):
                missing_segment_nonzero_histogram_ids.append(int(recording_id))
            summary = np.zeros(segment_width * 6, dtype=np.float32)
            count_features = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            summary = np.concatenate([
                values.mean(axis=0),
                values.std(axis=0),
                np.percentile(values, 10, axis=0),
                np.percentile(values, 50, axis=0),
                np.percentile(values, 90, axis=0),
                values.max(axis=0),
            ]).astype(np.float32)
            count_features = np.asarray(
                [float(len(values)), math.log1p(len(values)), 0.0],
                dtype=np.float32,
            )
        aggregated.append(np.concatenate([count_features, summary]))

    if missing_segment_nonzero_histogram_ids:
        raise RuntimeError(
            "Bird segment alignment is missing "
            f"{len(missing_segment_nonzero_histogram_ids)} recording IDs "
            "with non-zero histograms"
        )

    aligned_histogram = np.stack([
        histogram[histogram_lookup[int(recording_id)], 1:].astype(np.float32)
        for recording_id in requested
    ])
    aligned_segments = np.stack(aggregated)
    combined = np.concatenate([aligned_histogram, aligned_segments], axis=1)
    if not np.isfinite(combined).all():
        raise RuntimeError("Aligned bird supplemental features are non-finite")
    return combined, {
        "recordings": int(len(requested)),
        "histogram_width": int(aligned_histogram.shape[1]),
        "segment_source_width": segment_width,
        "segment_aggregate_width": int(aligned_segments.shape[1]),
        "combined_width": int(combined.shape[1]),
        "missing_segment_recordings": int(missing_segment_count),
        "missing_segment_policy": "zero_fill_only_when_histogram_all_zero",
        "missing_segment_nonzero_histograms": len(missing_segment_nonzero_histogram_ids),
        "private_labels_used": False,
        "source_files": {
            "histogram_of_segments.txt": {
                "bytes": int(histogram_path.stat().st_size),
                "sha256": _sha256_file(histogram_path),
            },
            "segment_features.txt": {
                "bytes": int(segment_path.stat().st_size),
                "sha256": _sha256_file(segment_path),
            },
        },
        "fingerprint": hashlib.sha256(
            json.dumps(
                {
                    "recording_ids": requested.tolist(),
                    "histogram_sha256": _sha256_file(histogram_path),
                    "segment_sha256": _sha256_file(segment_path),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
    }


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _bird_shared_cache(args: argparse.Namespace) -> Path:
    return wave0.ensure_within(
        args.output_root / "_shared_wave2_cache" / "mlsp-2013-birds",
        args.allowed_root,
    )


def build_bird_training_context(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve and validate the public MLSP Birds split once for every code path."""

    cid = "mlsp-2013-birds"
    resolved = resolve_competition(cid, args.data_root)
    essential = resolved.public_dir / "essential_data"
    supplemental = resolved.public_dir / "supplemental_data"
    mapping = pd.read_csv(essential / "rec_id2filename.txt")
    official_split = pd.read_csv(essential / "CVfolds_2.txt")
    if mapping["rec_id"].duplicated().any() or mapping["filename"].duplicated().any():
        raise RuntimeError("Bird recording-to-filename mapping is not one-to-one")
    if official_split["rec_id"].duplicated().any() or set(official_split["fold"]) != {0, 1}:
        raise RuntimeError("Bird official train/test split manifest is invalid")
    available = {path.stem: path for path in (essential / "src_wavs").glob("*.wav")}
    rec_to_path = {
        int(row.rec_id): available[str(row.filename)]
        for row in mapping.itertuples(index=False)
        if str(row.filename) in available
    }
    labels: dict[int, set[int]] = {}
    for line in (essential / "rec_labels_test_hidden.txt").read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.strip().split(",")
        if not parts or not parts[0] or (len(parts) > 1 and parts[1] == "?"):
            continue
        labels[int(parts[0])] = {int(value) for value in parts[1:] if value}
    train_ids = sorted(set(rec_to_path) & set(labels))
    sample = pd.read_csv(resolved.sample_submission_path)
    sample_ids = sample["Id"].astype(int).to_numpy()
    test_ids = sorted(set((sample_ids // 100).tolist()))
    sample_species = sample_ids % 100
    if np.any((sample_species < 0) | (sample_species >= 19)):
        raise RuntimeError("Bird sample submission contains an invalid species ID")
    sample_pairs = list(zip((sample_ids // 100).tolist(), sample_species.tolist(), strict=True))
    if len(set(sample_pairs)) != len(sample_pairs):
        raise RuntimeError("Bird sample submission contains duplicate recording/species rows")
    split_train_ids = sorted(
        official_split.loc[official_split["fold"] == 0, "rec_id"].astype(int).tolist()
    )
    split_test_ids = sorted(
        official_split.loc[official_split["fold"] == 1, "rec_id"].astype(int).tolist()
    )
    if train_ids != split_train_ids or test_ids != split_test_ids:
        raise RuntimeError("Bird labels/sample IDs do not match the official public split manifest")
    expected_pairs = {(recording_id, species) for recording_id in test_ids for species in range(19)}
    if set(sample_pairs) != expected_pairs:
        raise RuntimeError("Bird sample submission does not cover the complete recording/species grid")
    missing = sorted(set(test_ids) - set(rec_to_path))
    if missing:
        raise FileNotFoundError(f"Bird test audio missing for {len(missing)} recordings")
    truth = np.asarray(
        [[int(species in labels[rec_id]) for species in range(19)] for rec_id in train_ids],
        dtype=np.int8,
    )
    splits, iterative_assignment = make_multilabel_stratified_folds(
        truth,
        requested_folds=args.wave2_birds_folds,
        seed=args.seed,
    )
    fold_assignment = np.full(len(train_ids), -1, dtype=np.int16)
    for fold, (_, valid_indices) in enumerate(splits):
        fold_assignment[valid_indices] = fold
    if not np.array_equal(fold_assignment, iterative_assignment):
        raise RuntimeError("Bird fold assignment diverged from iterative stratification")
    return {
        "competition_id": cid,
        "resolved": resolved,
        "supplemental": supplemental,
        "sample": sample,
        "sample_ids": sample_ids,
        "sample_pairs": sample_pairs,
        "expected_pairs": expected_pairs,
        "rec_to_path": rec_to_path,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "all_ids": train_ids + test_ids,
        "truth": truth,
        "splits": splits,
        "fold_assignment": fold_assignment,
        "shared_cache": _bird_shared_cache(args),
    }


def _birds_precompute_identity(
    *,
    args: argparse.Namespace,
    context: dict[str, Any],
    audio_contract: dict[str, Any],
    supplemental_contract: dict[str, Any],
    feature_count: int,
) -> tuple[str, dict[str, Any]]:
    truth = np.asarray(context["truth"], dtype=np.int8)
    fold_assignment = np.asarray(context["fold_assignment"], dtype=np.int16)
    train_ids = np.asarray(context["train_ids"], dtype=np.int64)
    test_ids = np.asarray(context["test_ids"], dtype=np.int64)
    payload = {
        "schema": "evomind.mlsp_birds.cpu_precompute_identity.v1",
        "competition_id": "mlsp-2013-birds",
        "seed": int(args.seed),
        "folds": int(args.wave2_birds_folds),
        "species_iterations": int(args.wave2_birds_iterations),
        "train_recordings": int(len(train_ids)),
        "test_recordings": int(len(test_ids)),
        "feature_count": int(feature_count),
        "train_ids_sha256": _sha256_array(train_ids),
        "test_ids_sha256": _sha256_array(test_ids),
        "truth_sha256": _sha256_array(truth),
        "fold_assignment_sha256": _sha256_array(fold_assignment),
        "audio_feature_fingerprint": audio_contract["fingerprint"],
        "audio_waveform_sha256": audio_contract["path_manifest"]["waveform_sha256"],
        "supplemental_fingerprint": supplemental_contract["fingerprint"],
    }
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return identity, payload


def load_or_compute_birds_species_precompute(
    *,
    args: argparse.Namespace,
    context: dict[str, Any],
    x_train: np.ndarray,
    x_test: np.ndarray,
    audio_contract: dict[str, Any],
    supplemental_contract: dict[str, Any],
    task_dir: Path,
    logger: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, Any]]:
    """Fit or reuse the CPU-only 19-species channel before reserving A40 time."""

    from catboost import CatBoostClassifier

    truth = np.asarray(context["truth"], dtype=np.int8)
    splits = context["splits"]
    fold_count = len(splits)
    identity, identity_payload = _birds_precompute_identity(
        args=args,
        context=context,
        audio_contract=audio_contract,
        supplemental_contract=supplemental_contract,
        feature_count=x_train.shape[1],
    )
    cache_dir = context["shared_cache"] / "species_precompute"
    cache_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = cache_dir / f"birds_species_{identity[:24]}.npz"
    contract_path = cache_dir / f"birds_species_{identity[:24]}.json"
    if artifact_path.is_file() and contract_path.is_file():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if contract.get("identity_sha256") == identity:
            with np.load(artifact_path, allow_pickle=False) as payload:
                oof_species = payload["oof_species"]
                test_species = payload["test_species"]
                cached_fold = payload["fold_assignment"]
            if (
                oof_species.shape == truth.shape
                and test_species.shape == (len(context["test_ids"]), 19)
                and np.array_equal(cached_fold, context["fold_assignment"])
                and np.isfinite(oof_species).all()
                and np.isfinite(test_species).all()
            ):
                contract["cache_hit"] = True
                return (
                    np.asarray(oof_species, dtype=np.float64),
                    np.asarray(test_species, dtype=np.float64),
                    {str(key): int(value) for key, value in contract["best_iterations"].items()},
                    contract,
                )
    oof_species = np.zeros_like(truth, dtype=np.float64)
    test_species = np.zeros((len(context["test_ids"]), 19), dtype=np.float64)
    best_iterations: dict[str, int] = {}
    species_thread_count = max(1, min(8, int(args.wave2_audio_workers)))
    model_dir = cache_dir / f"models_{identity[:24]}"
    model_dir.mkdir(parents=True, exist_ok=True)
    for fold, (train_indices, valid_indices) in enumerate(splits):
        if logger is not None:
            logger.info("[mlsp-2013-birds] CPU species precompute fold=%s", fold)
        for species in range(19):
            fit_labels = truth[train_indices, species]
            key = f"fold{fold}_species{species}"
            if np.unique(fit_labels).size < 2:
                prior = float(np.mean(fit_labels))
                oof_species[valid_indices, species] = prior
                test_species[:, species] += prior / fold_count
                best_iterations[key] = 0
                continue
            model = CatBoostClassifier(
                iterations=args.wave2_birds_iterations,
                depth=6,
                learning_rate=0.04,
                loss_function="Logloss",
                eval_metric="Logloss",
                random_seed=args.seed + fold * 101 + species,
                verbose=False,
                allow_writing_files=False,
                thread_count=species_thread_count,
                od_type="Iter",
                od_wait=60,
            )
            model.fit(
                x_train[train_indices],
                fit_labels,
                eval_set=(x_train[valid_indices], truth[valid_indices, species]),
                use_best_model=True,
            )
            oof_species[valid_indices, species] = model.predict_proba(
                x_train[valid_indices]
            )[:, 1]
            test_species[:, species] += model.predict_proba(x_test)[:, 1] / fold_count
            model_path = model_dir / f"birds_fold{fold}_species{species:02d}.cbm"
            model.save_model(str(model_path))
            best_iterations[key] = max(1, int(model.get_best_iteration()) + 1)
    if not np.isfinite(oof_species).all() or not np.isfinite(test_species).all():
        raise RuntimeError("Bird species CPU precompute produced non-finite probabilities")
    temporary = artifact_path.with_name(f".{artifact_path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        oof_species=oof_species,
        test_species=test_species,
        fold_assignment=np.asarray(context["fold_assignment"], dtype=np.int16),
        train_recording_ids=np.asarray(context["train_ids"], dtype=np.int64),
        test_recording_ids=np.asarray(context["test_ids"], dtype=np.int64),
    )
    temporary.replace(artifact_path)
    contract = {
        "schema": "evomind.mlsp_birds.cpu_precompute.v1",
        "competition_id": "mlsp-2013-birds",
        "identity_sha256": identity,
        "identity": identity_payload,
        "artifact_path": str(artifact_path),
        "model_dir": str(model_dir),
        "cache_hit": False,
        "thread_count": species_thread_count,
        "best_iterations": best_iterations,
        "oof_shape": list(oof_species.shape),
        "test_shape": list(test_species.shape),
        "audio_feature_cache": audio_contract,
        "supplemental_contract": supplemental_contract,
        "private_labels_used": False,
        "created_at": wave0.utc_now(),
    }
    wave0.write_json(contract_path, contract)
    if task_dir != cache_dir:
        wave0.write_json(task_dir / "birds_species_precompute_contract.json", contract)
    return oof_species, test_species, best_iterations, contract


def build_birds_feature_matrices(
    args: argparse.Namespace,
    context: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    all_ids = context["all_ids"]
    handcrafted, audio_cache_contract = load_or_compute_audio_feature_matrix(
        [context["rec_to_path"][value] for value in all_ids],
        args.wave2_audio_workers,
        context["shared_cache"] / "audio_features",
    )
    provided, supplemental_contract = load_bird_supplemental_features(
        context["supplemental"],
        all_ids,
    )
    features = np.concatenate([handcrafted, provided], axis=1).astype(np.float32)
    x_train, x_test = features[: len(context["train_ids"])], features[len(context["train_ids"]):]
    return x_train, x_test, audio_cache_contract, supplemental_contract


def precompute_birds_cpu_artifacts(
    args: argparse.Namespace,
    task_dir: Path,
    logger: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    context = build_bird_training_context(args)
    x_train, x_test, audio_cache_contract, supplemental_contract = build_birds_feature_matrices(
        args,
        context,
    )
    oof_species, test_species, best_iterations, contract = load_or_compute_birds_species_precompute(
        args=args,
        context=context,
        x_train=x_train,
        x_test=x_test,
        audio_contract=audio_cache_contract,
        supplemental_contract=supplemental_contract,
        task_dir=task_dir,
        logger=logger,
    )
    cv_score = compute_metric("roc_auc", context["truth"].reshape(-1), oof_species.reshape(-1))
    result = {
        "competition_id": "mlsp-2013-birds",
        "status": "passed",
        "precompute_only": True,
        "valid_submission": False,
        "official_grader_executed": False,
        "cv_score": float(cv_score),
        "gpu_training_started": False,
        "cuda_required": False,
        "runtime_seconds_model": time.perf_counter() - started,
        "budget": {
            "seed": args.seed,
            "train_recordings": len(context["train_ids"]),
            "test_recordings": len(context["test_ids"]),
            "feature_count": int(x_train.shape[1]),
            "folds": len(context["splits"]),
            "audio_feature_cache": audio_cache_contract,
            "supplemental_contract": supplemental_contract,
            "species_precompute": contract,
            "best_iterations": best_iterations,
            "species_channel_pooled_oof_auc": float(cv_score),
        },
        "artifact_contract": contract,
        "oof_shape": list(oof_species.shape),
        "test_shape": list(test_species.shape),
    }
    wave0.write_json(task_dir / "birds_cpu_precompute_result.json", result)
    return result


def cross_fit_binary_auc_blend(
    first_oof: np.ndarray,
    second_oof: np.ndarray,
    first_test: np.ndarray,
    second_test: np.ndarray,
    target: np.ndarray,
    fold_assignment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[float], float]:
    """Select a probability blend using only out-of-fold fitting rows."""

    first = _finite_probability(np.asarray(first_oof, dtype=np.float64).reshape(-1))
    second = _finite_probability(np.asarray(second_oof, dtype=np.float64).reshape(-1))
    truth = np.asarray(target, dtype=np.int8).reshape(-1)
    folds = np.asarray(fold_assignment).reshape(-1)
    if not (first.shape == second.shape == truth.shape == folds.shape):
        raise RuntimeError("Binary AUC blend arrays do not share one row contract")
    first_test_values = np.asarray(first_test, dtype=np.float64).reshape(-1)
    second_test_values = np.asarray(second_test, dtype=np.float64).reshape(-1)
    if first_test_values.shape != second_test_values.shape:
        raise RuntimeError("Binary AUC blend test arrays do not share one row contract")
    if (
        set(np.unique(truth).tolist()) != {0, 1}
        or len(np.unique(folds)) < 2
        or np.any(folds < 0)
    ):
        raise RuntimeError("Binary AUC blend requires two classes and multiple folds")
    grid = np.linspace(0.0, 1.0, 21)

    def choose(mask: np.ndarray) -> float:
        if set(np.unique(truth[mask]).tolist()) != {0, 1}:
            raise RuntimeError("Binary AUC blend fitting rows must contain both classes")
        candidates = [
            (
                compute_metric(
                    "roc_auc",
                    truth[mask],
                    weight * first[mask] + (1.0 - weight) * second[mask],
                ),
                float(weight),
            )
            for weight in grid
        ]
        return max(candidates, key=lambda item: (item[0], -abs(item[1] - 0.5)))[1]

    cross_fitted = np.full(len(truth), np.nan, dtype=np.float64)
    weights: list[float] = []
    for fold in sorted(np.unique(folds).tolist()):
        held_out = folds == fold
        fitting = ~held_out
        weight = choose(fitting)
        weights.append(weight)
        cross_fitted[held_out] = weight * first[held_out] + (1.0 - weight) * second[held_out]
    if not np.isfinite(cross_fitted).all():
        raise RuntimeError("Cross-fitted binary AUC blend is incomplete")
    final_weight = choose(np.ones(len(truth), dtype=bool))
    test = (
        final_weight * first_test_values
        + (1.0 - final_weight) * second_test_values
    )
    return _finite_probability(cross_fitted), _finite_probability(test), weights, final_weight


def fit_birds_linear_species_channel(
    x_train: np.ndarray,
    x_test: np.ndarray,
    truth: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    seed: int,
    task_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit a regularized linear bird channel with recording-level OOF isolation."""

    from joblib import dump
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    train = np.asarray(x_train, dtype=np.float32)
    test = np.asarray(x_test, dtype=np.float32)
    target = np.asarray(truth, dtype=np.int8)
    if (
        train.ndim != 2
        or test.ndim != 2
        or target.ndim != 2
        or len(train) != len(target)
        or train.shape[1] != test.shape[1]
        or target.shape[1] != 19
    ):
        raise RuntimeError("Bird linear channel inputs do not share the recording/species contract")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise RuntimeError("Bird linear channel features contain non-finite values")

    regularization_c = 0.01
    oof = np.full(target.shape, np.nan, dtype=np.float64)
    test_prediction = np.zeros((len(test), target.shape[1]), dtype=np.float64)
    fold_artifacts: list[str] = []
    constant_models = 0
    for fold, (train_indices, valid_indices) in enumerate(splits):
        scaler = StandardScaler()
        fold_train = scaler.fit_transform(train[train_indices])
        fold_valid = scaler.transform(train[valid_indices])
        fold_test = scaler.transform(test)
        models: list[Any] = []
        for species in range(target.shape[1]):
            labels = target[train_indices, species]
            if np.unique(labels).size < 2:
                prior = float(np.mean(labels))
                oof[valid_indices, species] = prior
                test_prediction[:, species] += prior / len(splits)
                models.append({"constant_probability": prior})
                constant_models += 1
                continue
            model = LogisticRegression(
                C=regularization_c,
                solver="liblinear",
                max_iter=2_000,
                random_state=seed + fold * 101 + species,
            )
            model.fit(fold_train, labels)
            oof[valid_indices, species] = model.predict_proba(fold_valid)[:, 1]
            test_prediction[:, species] += model.predict_proba(fold_test)[:, 1] / len(splits)
            models.append(model)
        if task_dir is not None:
            artifact = task_dir / f"birds_linear_fold{fold}.joblib"
            dump(
                {
                    "schema": "evomind.mlsp_birds.linear_species_fold.v1",
                    "fold": fold,
                    "regularization_c": regularization_c,
                    "scaler": scaler,
                    "models": models,
                    "train_indices": np.asarray(train_indices, dtype=np.int32),
                    "validation_indices": np.asarray(valid_indices, dtype=np.int32),
                },
                artifact,
                compress=3,
            )
            fold_artifacts.append(artifact.name)
    if not np.isfinite(oof).all() or not np.isfinite(test_prediction).all():
        raise RuntimeError("Bird linear channel produced incomplete or non-finite probabilities")
    return _finite_probability(oof), _finite_probability(test_prediction), {
        "schema": "evomind.mlsp_birds.linear_species_channel.v1",
        "model": "StandardScaler_LogisticRegression",
        "regularization_c": regularization_c,
        "solver": "liblinear",
        "max_iter": 2_000,
        "folds": len(splits),
        "feature_count": int(train.shape[1]),
        "constant_models": constant_models,
        "fold_artifacts": fold_artifacts,
        "private_labels_used": False,
    }


def cross_fit_multichannel_logit_auc_blend(
    oof_channels: np.ndarray,
    test_channels: np.ndarray,
    target: np.ndarray,
    fold_assignment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[list[float]], list[float]]:
    """Cross-fit a three-channel simplex blend in calibrated log-odds space."""

    oof = _finite_probability(np.asarray(oof_channels, dtype=np.float64))
    test = _finite_probability(np.asarray(test_channels, dtype=np.float64))
    truth = np.asarray(target, dtype=np.int8).reshape(-1)
    folds = np.asarray(fold_assignment).reshape(-1)
    if oof.ndim != 2 or test.ndim != 2 or oof.shape[1] != 3 or test.shape[1] != 3:
        raise RuntimeError("Multichannel AUC blend requires exactly three probability channels")
    if oof.shape[0] != len(truth) or not (truth.shape == folds.shape):
        raise RuntimeError("Multichannel AUC blend arrays do not share one row contract")
    if (
        set(np.unique(truth).tolist()) != {0, 1}
        or len(np.unique(folds)) < 2
        or np.any(folds < 0)
    ):
        raise RuntimeError("Multichannel AUC blend requires two classes and multiple folds")

    epsilon = 1e-6
    oof_logits = np.log(np.clip(oof, epsilon, 1.0 - epsilon) / np.clip(1.0 - oof, epsilon, 1.0))
    test_logits = np.log(
        np.clip(test, epsilon, 1.0 - epsilon) / np.clip(1.0 - test, epsilon, 1.0)
    )
    grid = np.linspace(0.0, 1.0, 21)
    simplex = [
        np.asarray([first, second, 1.0 - first - second], dtype=np.float64)
        for first in grid
        for second in grid
        if first + second <= 1.0 + 1e-12
    ]

    def choose(mask: np.ndarray) -> np.ndarray:
        if set(np.unique(truth[mask]).tolist()) != {0, 1}:
            raise RuntimeError("Multichannel AUC blend fitting rows must contain both classes")
        candidates = [
            (
                compute_metric("roc_auc", truth[mask], oof_logits[mask] @ weights),
                -float(np.sum((weights - 1.0 / 3.0) ** 2)),
                weights,
            )
            for weights in simplex
        ]
        return max(candidates, key=lambda item: (item[0], item[1]))[2]

    cross_fitted_logits = np.full(len(truth), np.nan, dtype=np.float64)
    fold_weights: list[list[float]] = []
    for fold in sorted(np.unique(folds).tolist()):
        held_out = folds == fold
        weights = choose(~held_out)
        fold_weights.append(weights.tolist())
        cross_fitted_logits[held_out] = oof_logits[held_out] @ weights
    if not np.isfinite(cross_fitted_logits).all():
        raise RuntimeError("Cross-fitted multichannel AUC blend is incomplete")
    final_weights = choose(np.ones(len(truth), dtype=bool))
    blended_oof = 1.0 / (1.0 + np.exp(-cross_fitted_logits))
    blended_test = 1.0 / (1.0 + np.exp(-(test_logits @ final_weights)))
    return (
        _finite_probability(blended_oof),
        _finite_probability(blended_test),
        fold_weights,
        final_weights.tolist(),
    )


def run_birds(args: argparse.Namespace, task_dir: Path, logger: Any) -> dict[str, Any]:
    from catboost import CatBoostClassifier

    cid = "mlsp-2013-birds"
    started = time.perf_counter()
    context = build_bird_training_context(args)
    sample = context["sample"]
    sample_ids = context["sample_ids"]
    sample_pairs = context["sample_pairs"]
    expected_pairs = context["expected_pairs"]
    train_ids = context["train_ids"]
    test_ids = context["test_ids"]
    truth = context["truth"]
    splits = context["splits"]
    fold_count = len(splits)
    fold_assignment = context["fold_assignment"]
    x_train, x_test, audio_cache_contract, supplemental_contract = build_birds_feature_matrices(
        args,
        context,
    )
    oof_species, test_species, best_iterations, species_precompute_contract = (
        load_or_compute_birds_species_precompute(
            args=args,
            context=context,
            x_train=x_train,
            x_test=x_test,
            audio_contract=audio_cache_contract,
            supplemental_contract=supplemental_contract,
            task_dir=task_dir,
            logger=logger,
        )
    )
    oof_linear, test_linear, linear_channel_contract = fit_birds_linear_species_channel(
        x_train,
        x_test,
        truth,
        splits,
        seed=args.seed,
        task_dir=task_dir,
    )

    onehot_species = np.eye(19, dtype=np.float32)

    def pair_features(recording_indices: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        repeated = np.repeat(matrix[recording_indices], 19, axis=0)
        species_features = np.tile(onehot_species, (len(recording_indices), 1))
        return np.concatenate([repeated, species_features], axis=1)

    test_recording_indices = np.arange(len(x_test))
    x_test_pairs = pair_features(test_recording_indices, x_test)
    oof_pair = np.zeros_like(truth, dtype=np.float64)
    test_pair = np.zeros((len(test_ids), 19), dtype=np.float64)
    pair_best_iterations: list[int] = []
    for fold, (train_indices, valid_indices) in enumerate(splits):
        train_pairs = pair_features(train_indices, x_train)
        valid_pairs = pair_features(valid_indices, x_train)
        train_pair_labels = truth[train_indices].reshape(-1)
        valid_pair_labels = truth[valid_indices].reshape(-1)
        model = CatBoostClassifier(
            iterations=args.wave2_birds_pair_iterations,
            depth=7,
            learning_rate=0.035,
            loss_function="Logloss",
            eval_metric="AUC",
            task_type="GPU",
            devices="0",
            random_seed=args.seed + 20_000 + fold,
            verbose=100,
            allow_writing_files=False,
            od_type="Iter",
            od_wait=80,
        )
        model.fit(
            train_pairs,
            train_pair_labels,
            eval_set=(valid_pairs, valid_pair_labels),
            use_best_model=True,
        )
        oof_pair[valid_indices] = model.predict_proba(valid_pairs)[:, 1].reshape(-1, 19)
        test_pair += model.predict_proba(x_test_pairs)[:, 1].reshape(-1, 19) / fold_count
        model.save_model(str(task_dir / f"birds_pair_fold{fold}.cbm"))
        pair_best_iterations.append(max(1, int(model.get_best_iteration()) + 1))

    pair_folds = np.repeat(fold_assignment, 19)
    blended_oof, blended_test, fold_blend_weights, final_blend_weights = (
        cross_fit_multichannel_logit_auc_blend(
            np.column_stack([
                oof_species.reshape(-1),
                oof_pair.reshape(-1),
                oof_linear.reshape(-1),
            ]),
            np.column_stack([
                test_species.reshape(-1),
                test_pair.reshape(-1),
                test_linear.reshape(-1),
            ]),
            truth.reshape(-1),
            pair_folds,
        )
    )
    cv_score = compute_metric("roc_auc", truth.reshape(-1), blended_oof)
    promotion_contract = WAVE2_PROMOTION_CONTRACTS[cid]
    complete_sample_grid = bool(
        len(sample_pairs) == len(expected_pairs)
        and set(sample_pairs) == expected_pairs
    )
    complete_oof = bool(
        np.all(fold_assignment >= 0)
        and np.isfinite(blended_oof).all()
        and blended_oof.size == truth.size
    )
    private_labels_unused = bool(
        supplemental_contract.get("private_labels_used") is False
    )
    promotion_gate = wave0.build_metric_promotion_gate(
        name=str(promotion_contract["name"]),
        metric=str(promotion_contract["metric"]),
        direction=str(promotion_contract["direction"]),
        score=float(cv_score),
        threshold=float(promotion_contract["threshold"]),
        extra_checks={
            "pooled_recording_species_metric": True,
            "complete_recording_species_sample_grid": complete_sample_grid,
            "complete_cross_fitted_oof_coverage": complete_oof,
            "official_train_test_manifest_verified": True,
            "private_labels_unused": private_labels_unused,
        },
        evidence={
            "metric_reduction": "pooled_recording_species_binary_roc_auc",
            "train_recordings": len(train_ids),
            "test_recordings": len(test_ids),
            "species": 19,
            "expected_sample_pairs": len(expected_pairs),
            "observed_sample_pairs": len(sample_pairs),
            "supplemental_contract": supplemental_contract,
            "private_labels_used": False,
        },
    )
    macro_species_auc = {
        str(species): compute_metric("roc_auc", truth[:, species], oof_species[:, species])
        for species in range(19)
        if np.unique(truth[:, species]).size == 2
    }
    blended_test_matrix = blended_test.reshape(len(test_ids), 19)
    test_lookup = {rec_id: index for index, rec_id in enumerate(test_ids)}
    sample["Probability"] = [
        blended_test_matrix[test_lookup[int(value // 100)], int(value % 100)]
        for value in sample_ids
    ]
    np.savez_compressed(
        task_dir / "birds_oof_and_test.npz",
        truth=truth,
        oof_species=oof_species,
        oof_pair=oof_pair,
        oof_linear=oof_linear,
        oof_blend=blended_oof.reshape(-1, 19),
        test_species=test_species,
        test_pair=test_pair,
        test_linear=test_linear,
        test_blend=blended_test_matrix,
        fold=fold_assignment,
        train_recording_ids=np.asarray(train_ids),
        test_recording_ids=np.asarray(test_ids),
    )
    budget = {"seed": args.seed, "train_recordings": len(train_ids), "test_recordings": len(test_ids),
              "feature_count": x_train.shape[1], "folds": fold_count,
              "split_strategy": "iterative_multilabel_stratification_by_recording",
              "official_metric_reduction": "pooled_recording_species_binary_roc_auc",
              "audio_feature_cache": audio_cache_contract,
              "supplemental_contract": supplemental_contract,
              "species_precompute": species_precompute_contract,
              "linear_channel": linear_channel_contract,
              "best_iterations": best_iterations,
              "pair_best_iterations": pair_best_iterations,
              "fold_blend_weights_species_pair_linear": fold_blend_weights,
              "final_blend_weights_species_pair_linear": final_blend_weights,
              "fold_test_ensemble": True,
              "official_train_test_manifest_verified": True,
              "private_labels_used": False,
              "promotion_gate": promotion_gate}
    return wave0.finalize_scored_task(
        competition_id=cid, submission=sample, cv_score=cv_score, args=args, task_dir=task_dir,
        budget=budget, promotion_gate=promotion_gate,
        extra={"runtime_seconds_model": time.perf_counter() - started,
                              "model_family": "audio_1897plus_CatBoost_species_pair_linear_logit_blend",
                              "macro_species_oof_auc": macro_species_auc,
                              "species_channel_pooled_oof_auc": compute_metric(
                                  "roc_auc", truth.reshape(-1), oof_species.reshape(-1)
                              ),
                              "pair_channel_pooled_oof_auc": compute_metric(
                                   "roc_auc", truth.reshape(-1), oof_pair.reshape(-1)
                               ),
                              "linear_channel_pooled_oof_auc": compute_metric(
                                  "roc_auc", truth.reshape(-1), oof_linear.reshape(-1)
                              ),
                              "budget": budget},
    )


def make_whale_duplicate_aware_splits(
    labels: np.ndarray,
    waveform_hashes: np.ndarray,
    *,
    requested_folds: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str, dict[str, Any]]:
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    target = np.asarray(labels, dtype=np.int8).reshape(-1)
    groups = np.asarray(waveform_hashes, dtype=object).reshape(-1)
    if target.shape != groups.shape or set(np.unique(target).tolist()) != {0, 1}:
        raise RuntimeError("Whale split inputs must contain aligned binary labels and hashes")
    grouped = pd.DataFrame({"hash": groups, "label": target}).groupby("hash")["label"]
    label_counts = grouped.nunique()
    conflicts = int((label_counts > 1).sum())
    if conflicts:
        raise RuntimeError(
            f"Whale exact waveform hashes contain conflicting labels in {conflicts} groups"
        )
    indices = np.arange(len(target))
    duplicate_groups = int(len(groups) - len(np.unique(groups)))
    if not duplicate_groups:
        fold_count = min(int(requested_folds), int(np.bincount(target).min()))
        if fold_count < 2:
            raise RuntimeError("Whale stratified CV requires at least two rows per class")
        splitter = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=seed)
        splits = list(splitter.split(indices, target))
        strategy = "stratified_kfold_no_exact_duplicates"
    else:
        group_labels = pd.DataFrame({"hash": groups, "label": target}).drop_duplicates("hash")
        group_counts = group_labels.groupby("label").size()
        max_folds = min(int(requested_folds), int(group_counts.min()))
        splits = []
        fold_count = 0
        for candidate_folds in range(max_folds, 1, -1):
            splitter = StratifiedGroupKFold(
                n_splits=candidate_folds,
                shuffle=True,
                random_state=seed,
            )
            candidate = list(splitter.split(indices, target, groups))
            if all(
                np.unique(target[train_indices]).size == 2
                and np.unique(target[valid_indices]).size == 2
                for train_indices, valid_indices in candidate
            ):
                splits = candidate
                fold_count = candidate_folds
                break
        if not splits:
            raise RuntimeError("Whale duplicate-aware CV could not form two-class folds")
        strategy = "stratified_group_kfold_exact_waveform_sha256"
    return splits, strategy, {
        "folds": int(fold_count),
        "unique_waveform_hashes": int(len(np.unique(groups))),
        "duplicate_rows": duplicate_groups,
        "conflicting_duplicate_hash_groups": conflicts,
    }


def run_whale(args: argparse.Namespace, task_dir: Path, logger: Any) -> dict[str, Any]:
    from catboost import CatBoostClassifier

    cid = "the-icml-2013-whale-challenge-right-whale-redux"
    resolved = resolve_competition(cid, args.data_root)
    shared_cache = wave0.ensure_within(
        args.output_root / "_shared_wave2_cache" / cid,
        args.allowed_root,
    )
    train_dir, train_extract_contract = extract_zip_to_shared_cache(
        resolved.public_dir / "train2.zip", shared_cache / "archives"
    )
    test_dir, test_extract_contract = extract_zip_to_shared_cache(
        resolved.public_dir / "test2.zip", shared_cache / "archives"
    )
    if (train_dir / "train2").is_dir():
        train_dir = train_dir / "train2"
    if (test_dir / "test2").is_dir():
        test_dir = test_dir / "test2"
    train_paths = sorted(train_dir.glob("*.aif"))
    if not train_paths:
        raise FileNotFoundError("Whale training archive contains no AIF clips")
    label_matches = [re.search(r"_([01])\.aif$", path.name) for path in train_paths]
    if any(match is None for match in label_matches):
        raise RuntimeError("Whale training clip filenames do not encode strict binary labels")
    labels = np.asarray([int(match.group(1)) for match in label_matches if match is not None])
    if args.wave2_whale_max_train and len(train_paths) > args.wave2_whale_max_train:
        rng = np.random.default_rng(args.seed)
        selected = np.sort(rng.choice(len(train_paths), args.wave2_whale_max_train, replace=False))
        train_paths = [train_paths[index] for index in selected]
        labels = labels[selected]
    sample = pd.read_csv(resolved.sample_submission_path)
    if sample["clip"].duplicated().any():
        raise RuntimeError("Whale sample submission contains duplicate clip IDs")
    test_paths = [test_dir / value for value in sample["clip"].astype(str)]
    missing_test = [path.name for path in test_paths if not path.is_file()]
    if missing_test:
        raise FileNotFoundError(f"Whale sample submission references {len(missing_test)} missing clips")
    started = time.perf_counter()
    all_paths = train_paths + test_paths
    all_features, audio_cache_contract = load_or_compute_audio_feature_matrix(
        all_paths,
        args.wave2_audio_workers,
        shared_cache / "audio_features",
    )
    x_train, x_test = all_features[: len(train_paths)], all_features[len(train_paths):]
    waveform_manifest, waveform_manifest_contract = load_or_compute_audio_path_manifest(
        all_paths,
        args.wave2_audio_workers,
        shared_cache / "audio_features" / "path_manifests",
    )
    all_waveform_hashes = np.asarray(
        [row["sha256"] for row in waveform_manifest["rows"]],
        dtype=object,
    )
    waveform_hashes = all_waveform_hashes[: len(train_paths)]
    test_waveform_hashes = all_waveform_hashes[len(train_paths):]
    splits, split_strategy, duplicate_contract = make_whale_duplicate_aware_splits(
        labels,
        waveform_hashes,
        requested_folds=args.wave2_whale_folds,
        seed=args.seed,
    )
    fold_count = len(splits)
    oof = np.zeros(len(train_paths), dtype=np.float64)
    test_prediction = np.zeros(len(test_paths), dtype=np.float64)
    fold_assignment = np.full(len(train_paths), -1, dtype=np.int16)
    fold_scores: list[float] = []
    best_iterations: list[int] = []
    for fold, (train_indices, valid_indices) in enumerate(splits):
        fold_assignment[valid_indices] = fold
        model = CatBoostClassifier(
            iterations=args.wave2_whale_iterations,
            depth=8,
            learning_rate=0.04,
            loss_function="Logloss",
            eval_metric="AUC",
            task_type="GPU",
            devices="0",
            random_seed=args.seed + fold * 101,
            verbose=100,
            allow_writing_files=False,
            od_type="Iter",
            od_wait=100,
        )
        model.fit(
            x_train[train_indices],
            labels[train_indices],
            eval_set=(x_train[valid_indices], labels[valid_indices]),
            use_best_model=True,
        )
        fold_valid = model.predict_proba(x_train[valid_indices])[:, 1]
        oof[valid_indices] = fold_valid
        test_prediction += model.predict_proba(x_test)[:, 1] / fold_count
        fold_scores.append(compute_metric("roc_auc", labels[valid_indices], fold_valid))
        best_iterations.append(max(1, int(model.get_best_iteration()) + 1))
        model.save_model(str(task_dir / f"whale_audio_catboost_fold{fold}.cbm"))
    if np.any(fold_assignment < 0):
        raise RuntimeError("Whale OOF coverage is incomplete")
    cv_score = compute_metric("roc_auc", labels, oof)
    promotion_contract = WAVE2_PROMOTION_CONTRACTS[cid]
    promotion_gate = wave0.build_metric_promotion_gate(
        name=str(promotion_contract["name"]),
        metric=str(promotion_contract["metric"]),
        direction=str(promotion_contract["direction"]),
        score=float(cv_score),
        threshold=float(promotion_contract["threshold"]),
        extra_checks={
            "duplicate_aware_split_protocol": split_strategy in {
                "stratified_group_kfold_exact_waveform_sha256",
                "stratified_kfold_no_exact_duplicates",
            },
            "zero_conflicting_duplicate_hash_groups": (
                duplicate_contract["conflicting_duplicate_hash_groups"] == 0
            ),
            "complete_cross_fitted_oof_coverage": bool(
                np.all(fold_assignment >= 0)
                and np.isfinite(oof).all()
                and len(oof) == len(labels)
            ),
            "private_labels_unused": True,
        },
        evidence={
            "split_strategy": split_strategy,
            "duplicate_contract": duplicate_contract,
            "fold_scores": fold_scores,
            "private_labels_used": False,
        },
    )
    label_by_hash = dict(zip(waveform_hashes.tolist(), labels.tolist(), strict=True))
    exact_test_match = np.asarray([value in label_by_hash for value in test_waveform_hashes])
    if np.any(exact_test_match):
        test_prediction[exact_test_match] = [
            label_by_hash[value] for value in test_waveform_hashes[exact_test_match]
        ]
    sample["probability"] = _finite_probability(test_prediction)
    np.savez_compressed(
        task_dir / "whale_oof_and_test.npz",
        truth=labels,
        oof=oof,
        test=test_prediction,
        fold=fold_assignment,
        train_clip=np.asarray([path.name for path in train_paths]),
        test_clip=np.asarray([path.name for path in test_paths]),
        waveform_sha256=waveform_hashes,
        test_waveform_sha256=test_waveform_hashes,
    )
    budget = {"seed": args.seed, "train_clips": len(train_paths), "test_clips": len(test_paths),
              "feature_count": x_train.shape[1], "folds": fold_count,
              "split_strategy": split_strategy, "fold_scores": fold_scores,
              "best_iterations": best_iterations,
              **duplicate_contract,
              "train_extract_cache": train_extract_contract,
              "test_extract_cache": test_extract_contract,
              "audio_feature_cache": audio_cache_contract,
              "waveform_hash_manifest": waveform_manifest_contract,
              "exact_train_test_hash_matches": int(exact_test_match.sum()),
              "exact_duplicate_test_override": True,
              "fold_test_ensemble": True,
              "private_labels_used": False,
              "promotion_gate": promotion_gate}
    return wave0.finalize_scored_task(
        competition_id=cid, submission=sample, cv_score=cv_score, args=args, task_dir=task_dir,
        budget=budget, promotion_gate=promotion_gate,
        extra={"runtime_seconds_model": time.perf_counter() - started,
                              "model_family": "audio_1572_CatBoost_GPU_duplicate_aware_fold_ensemble",
                              "budget": budget},
    )


def build_normalization_map(frame: pd.DataFrame) -> dict[str, str]:
    pairs = frame.loc[:, ["before", "after"]].astype(str).copy()
    if pairs.empty:
        return {}
    grouped = pairs.groupby(["before", "after"], sort=False).size().rename("count").reset_index()
    grouped = grouped.sort_values(["before", "count", "after"], ascending=[True, False, True])
    best = grouped.drop_duplicates("before", keep="first")
    return dict(zip(best["before"].astype(str), best["after"].astype(str)))


_SMALL_NUMBERS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def english_integer_words(value: int) -> str:
    if value < 0:
        return "minus " + english_integer_words(-value)
    if value < 20:
        return _SMALL_NUMBERS[value]
    if value < 100:
        return _TENS[value // 10] + (" " + _SMALL_NUMBERS[value % 10] if value % 10 else "")
    if value < 1000:
        return _SMALL_NUMBERS[value // 100] + " hundred" + (" " + english_integer_words(value % 100) if value % 100 else "")
    for scale, label in ((10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (1000, "thousand")):
        if value >= scale:
            return english_integer_words(value // scale) + f" {label}" + (" " + english_integer_words(value % scale) if value % scale else "")
    return str(value)


_ORDINAL_SMALL = {
    0: "zeroth", 1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
    11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
    15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth", 19: "nineteenth",
}
_ORDINAL_TENS = {20: "twentieth", 30: "thirtieth", 40: "fortieth", 50: "fiftieth",
                 60: "sixtieth", 70: "seventieth", 80: "eightieth", 90: "ninetieth"}
_ENGLISH_DIGIT_WORDS = ("o", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
_ENGLISH_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}
_ENGLISH_MONTH_ALIASES = {
    "jan": "january", "january": "january",
    "feb": "february", "february": "february",
    "mar": "march", "march": "march",
    "apr": "april", "april": "april",
    "may": "may",
    "jun": "june", "june": "june",
    "jul": "july", "july": "july",
    "aug": "august", "august": "august",
    "sep": "september", "sept": "september", "september": "september",
    "oct": "october", "october": "october",
    "nov": "november", "november": "november",
    "dec": "december", "december": "december",
}
_DIGIT_WORDS_RUSSIAN = ("ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять")
_RUSSIAN_SMALL = (
    "ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять",
    "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
    "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать",
)
_RUSSIAN_TENS = ("", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто")
_RUSSIAN_HUNDREDS = ("", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот")
_RUSSIAN_GENITIVE_SMALL = (
    "нуля", "одного", "двух", "трех", "четырех", "пяти", "шести", "семи", "восьми", "девяти",
    "десяти", "одиннадцати", "двенадцати", "тринадцати", "четырнадцати", "пятнадцати",
    "шестнадцати", "семнадцати", "восемнадцати", "девятнадцати",
)


def _seed_vision_worker(_: int) -> None:
    """Seed Python and NumPy from PyTorch's persisted worker seed."""

    import torch

    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)
_RUSSIAN_GENITIVE_TENS = (
    "", "", "двадцати", "тридцати", "сорока", "пятидесяти", "шестидесяти",
    "семидесяти", "восьмидесяти", "девяноста",
)
_RUSSIAN_GENITIVE_HUNDREDS = (
    "", "ста", "двухсот", "трехсот", "четырехсот", "пятисот", "шестисот",
    "семисот", "восьмисот", "девятисот",
)
_RUSSIAN_ORDINAL_BASE = {
    1: "первый", 2: "второй", 3: "третий", 4: "четвертый", 5: "пятый",
    6: "шестой", 7: "седьмой", 8: "восьмой", 9: "девятый", 10: "десятый",
    11: "одиннадцатый", 12: "двенадцатый", 13: "тринадцатый",
    14: "четырнадцатый", 15: "пятнадцатый", 16: "шестнадцатый",
    17: "семнадцатый", 18: "восемнадцатый", 19: "девятнадцатый",
    20: "двадцатый", 30: "тридцатый", 40: "сороковой", 50: "пятидесятый",
    60: "шестидесятый", 70: "семидесятый", 80: "восьмидесятый",
    90: "девяностый",
}
_RUSSIAN_ORDINAL_HUNDREDS = {
    100: "сотый", 200: "двухсотый", 300: "трехсотый", 400: "четырехсотый",
    500: "пятисотый", 600: "шестисотый", 700: "семисотый",
    800: "восьмисотый", 900: "девятисотый",
}
_RUSSIAN_MONTHS_GENITIVE = (
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
)


def english_ordinal_words(value: int) -> str:
    if value < 0:
        return "minus " + english_ordinal_words(-value)
    if value in _ORDINAL_SMALL:
        return _ORDINAL_SMALL[value]
    if value in _ORDINAL_TENS:
        return _ORDINAL_TENS[value]
    if value < 100:
        return _TENS[value // 10] + " " + _ORDINAL_SMALL[value % 10]
    for scale, cardinal, ordinal in (
        (10**12, "trillion", "trillionth"), (10**9, "billion", "billionth"),
        (10**6, "million", "millionth"), (1000, "thousand", "thousandth"),
        (100, "hundred", "hundredth"),
    ):
        if value >= scale:
            quotient, remainder = divmod(value, scale)
            prefix = english_integer_words(quotient)
            return f"{prefix} {ordinal}" if remainder == 0 else f"{prefix} {cardinal} {english_ordinal_words(remainder)}"
    return str(value)


def english_year_words(value: int) -> str:
    """Render the year convention used by the English normalization corpus."""

    if 2000 <= value <= 2009:
        return "two thousand" + (" " + english_integer_words(value - 2000) if value > 2000 else "")
    if 1000 <= value <= 2099:
        century, tail = divmod(value, 100)
        return english_integer_words(century) + (
            " hundred" if tail == 0 else " " + english_integer_words(tail)
        )
    if 100 <= value <= 999:
        century, tail = divmod(value, 100)
        return english_integer_words(century) + (
            " hundred" if tail == 0 else " " + english_integer_words(tail)
        )
    return english_integer_words(value)


def _roman_integer(token: str) -> int | None:
    compact = token.upper().rstrip(".")
    if not compact or not re.fullmatch(r"[IVXLCDM]+", compact):
        return None
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    previous = 0
    for character in reversed(compact):
        current = values[character]
        total += -current if current < previous else current
        previous = max(previous, current)
    return total if 0 < total <= 3999 else None


def english_date_words(token: str) -> str | None:
    compact = re.sub(r"\s+", " ", token.strip().rstrip(","))
    if re.fullmatch(r"\d{4}", compact):
        value = int(compact)
        if 1000 <= value <= 2099:
            return english_year_words(value)
    decade = re.fullmatch(r"(\d{3,4})s", compact)
    if decade:
        value = int(decade.group(1))
        century, tail = divmod(value, 100)
        decade_names = {
            10: "tens", 20: "twenties", 30: "thirties", 40: "forties",
            50: "fifties", 60: "sixties", 70: "seventies", 80: "eighties",
            90: "nineties",
        }
        if tail in decade_names:
            return f"{english_integer_words(century)} {decade_names[tail]}"

    weekday = ""
    weekday_match = re.match(
        r"(?i)^(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+",
        compact,
    )
    if weekday_match:
        weekday = weekday_match.group(1).lower() + " "
        compact = compact[weekday_match.end():]

    month_pattern = "|".join(sorted(_ENGLISH_MONTH_ALIASES, key=len, reverse=True))
    # Normalize month abbreviations first and then parse canonical month names.
    month_search = re.search(
        rf"(?i)(?<![A-Za-z])({month_pattern})(?:\.)?(?![A-Za-z])",
        compact,
    )
    if month_search:
        raw_month = month_search.group(1).lower()
        canonical = _ENGLISH_MONTH_ALIASES[raw_month]
        compact = compact[:month_search.start()] + canonical + compact[month_search.end():]
        compact = re.sub(r"\s+", " ", compact).strip()
    month_pattern = "|".join(_ENGLISH_MONTHS)
    day_first = re.fullmatch(
        rf"(\d{{1,2}})\s+({month_pattern})(?:\s*,\s*|\s+)(\d{{3,4}})",
        compact,
        flags=re.IGNORECASE,
    )
    if day_first:
        day, month, year = day_first.groups()
        return weekday + f"the {english_ordinal_words(int(day))} of {month.lower()} {english_year_words(int(year))}"
    month_first = re.fullmatch(
        rf"({month_pattern})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:\s*,\s*|\s+)(\d{{3,4}})",
        compact,
        flags=re.IGNORECASE,
    )
    if month_first:
        month, day, year = month_first.groups()
        return weekday + f"{month.lower()} {english_ordinal_words(int(day))} {english_year_words(int(year))}"
    month_year = re.fullmatch(rf"({month_pattern})\s+(\d{{3,4}})", compact, flags=re.IGNORECASE)
    if month_year:
        month, year = month_year.groups()
        return weekday + f"{month.lower()} {english_year_words(int(year))}"
    day_month = re.fullmatch(rf"(\d{{1,2}})\s+({month_pattern})", compact, flags=re.IGNORECASE)
    if day_month:
        day, month = day_month.groups()
        return weekday + f"the {english_ordinal_words(int(day))} of {month.lower()}"
    month_day = re.fullmatch(
        rf"({month_pattern})\s+(\d{{1,2}})(?:st|nd|rd|th)?",
        compact,
        flags=re.IGNORECASE,
    )
    if month_day:
        month, day = month_day.groups()
        return weekday + f"{month.lower()} {english_ordinal_words(int(day))}"
    iso = re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", compact)
    if iso:
        year, month, day = (int(value) for value in iso.groups())
        month_name = next((name for name, number in _ENGLISH_MONTHS.items() if number == month), None)
        if month_name and 1 <= day <= 31:
            return weekday + f"the {english_ordinal_words(day)} of {month_name} {english_year_words(year)}"
    return None


def english_letters_words(token: str) -> str | None:
    compact = token.strip()
    possessive = bool(re.fullmatch(r"[A-Z]{1,12}s", compact))
    letters = re.findall(r"[A-Za-z]", compact[:-1] if possessive else compact)
    if not letters or re.sub(r"[A-Za-z.\-'\s]", "", compact):
        return None
    result = " ".join(character.lower() for character in letters)
    return result + "'s" if possessive else result


def english_digit_words(token: str) -> str | None:
    digits = re.sub(r"\D", "", token)
    if not digits:
        return None
    if digits.startswith("00") and len(digits) == 3:
        return "double o " + _ENGLISH_DIGIT_WORDS[int(digits[2])]
    return " ".join(_ENGLISH_DIGIT_WORDS[int(character)] for character in digits)


def english_telephone_words(token: str) -> str | None:
    groups = re.findall(r"[A-Za-z0-9]+", token)
    if not groups:
        return None
    spoken: list[str] = []
    for group in groups:
        characters = [
            _ENGLISH_DIGIT_WORDS[int(character)] if character.isdigit() else character.lower()
            for character in group
        ]
        spoken.append(" ".join(characters))
    return " sil ".join(spoken)


def english_electronic_words(token: str) -> str | None:
    names = {
        ".": "dot", ":": "colon", "/": "slash", "-": "dash", "_": "underscore",
        "#": "hash", ",": "comma", "@": "at", "&": "and", "%": "percent",
        "+": "plus", "=": "equals", "?": "question mark", "~": "tilde",
    }
    spoken: list[str] = []
    for character in token.strip():
        if character.isalpha():
            spoken.append(character.lower())
        elif character.isdigit():
            spoken.append(_ENGLISH_DIGIT_WORDS[int(character)])
        elif character in names:
            spoken.append(names[character])
    return " ".join(spoken) if spoken else None


def english_measure_words(token: str) -> str | None:
    compact = token.strip()
    percent = re.fullmatch(r"([-+]?\d[\d,.]*)\s*%", compact)
    if percent:
        return _english_cardinal_or_decimal(percent.group(1)) + " percent"
    match = re.fullmatch(
        r"([-+]?\d[\d,.]*)\s*(/)?\s*(km²|km2|m²|m2|kg|km|cm|mm|mi|m|g|lb|ft|hz|mhz|ghz)",
        compact,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    number, per, raw_unit = match.groups()
    value = _english_cardinal_or_decimal(number)
    unit = {
        "kg": "kilograms", "km": "kilometers", "cm": "centimeters", "mm": "millimeters",
        "mi": "miles", "m": "meters", "g": "grams", "lb": "pounds", "ft": "feet",
        "hz": "hertz", "mhz": "megahertz", "ghz": "gigahertz",
        "km²": "square kilometers", "km2": "square kilometers",
        "m²": "square meters", "m2": "square meters",
    }[raw_unit.lower()]
    return f"{value} {'per ' if per else ''}{unit}"


def english_money_words(token: str) -> str | None:
    match = re.fullmatch(
        r"\s*([$£€])\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*(million|billion|thousand)?\s*",
        token,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    symbol, raw_number, scale = match.groups()
    major = {"$": ("dollar", "dollars"), "£": ("pound", "pounds"), "€": ("euro", "euros")}[symbol]
    if scale:
        return f"{_english_cardinal_or_decimal(raw_number)} {scale.lower()} {major[1]}"
    number = raw_number.replace(",", "")
    if "." in number:
        whole, fraction = number.split(".", 1)
        if len(fraction) <= 2:
            whole_value = int(whole)
            cent_value = int(fraction.ljust(2, "0"))
            result = f"{english_integer_words(whole_value)} {major[0] if abs(whole_value) == 1 else major[1]}"
            if cent_value:
                result += f" and {english_integer_words(cent_value)} {'cent' if cent_value == 1 else 'cents'}"
            return result
    numeric = int(number)
    return f"{english_integer_words(numeric)} {major[0] if abs(numeric) == 1 else major[1]}"


def _russian_under_thousand(value: int, *, feminine: bool = False) -> str:
    parts: list[str] = []
    if value >= 100:
        parts.append(_RUSSIAN_HUNDREDS[value // 100])
        value %= 100
    if value < 20:
        if feminine and value in (1, 2):
            parts.append("одна" if value == 1 else "две")
        elif value:
            parts.append(_RUSSIAN_SMALL[value])
    else:
        parts.append(_RUSSIAN_TENS[value // 10])
        if value % 10:
            parts.append(_RUSSIAN_SMALL[value % 10])
    return " ".join(parts)


def russian_integer_words(value: int) -> str:
    if value == 0:
        return _RUSSIAN_SMALL[0]
    if value < 0:
        return "минус " + russian_integer_words(-value)
    parts: list[str] = []
    scales = ((10**9, "миллиард", "миллиарда", "миллиардов", False),
              (10**6, "миллион", "миллиона", "миллионов", False),
              (1000, "тысяча", "тысячи", "тысяч", True))
    for scale, one, few, many, feminine in scales:
        amount, value = divmod(value, scale)
        if not amount:
            continue
        if not (scale == 1000 and amount == 1):
            parts.append(_russian_under_thousand(amount, feminine=feminine))
        tail = amount % 100
        if 11 <= tail <= 14:
            label = many
        elif amount % 10 == 1:
            label = one
        elif amount % 10 in (2, 3, 4):
            label = few
        else:
            label = many
        parts.append(label)
    if value:
        parts.append(_russian_under_thousand(value))
    return " ".join(part for part in parts if part)


def russian_cardinal_genitive_words(value: int) -> str:
    """Conservative genitive cardinal rendering used by explicit preposition contexts."""

    if value == 0:
        return _RUSSIAN_GENITIVE_SMALL[0]
    if value < 0:
        return "минус " + russian_cardinal_genitive_words(-value)
    if value >= 1000:
        thousands, remainder = divmod(value, 1000)
        if thousands == 1:
            prefix = "тысячи"
        else:
            prefix = f"{russian_cardinal_genitive_words(thousands)} тысяч"
        return f"{prefix} {russian_cardinal_genitive_words(remainder)}" if remainder else prefix
    parts: list[str] = []
    if value >= 100:
        parts.append(_RUSSIAN_GENITIVE_HUNDREDS[value // 100])
        value %= 100
    if value < 20:
        if value:
            parts.append(_RUSSIAN_GENITIVE_SMALL[value])
    else:
        parts.append(_RUSSIAN_GENITIVE_TENS[value // 10])
        if value % 10:
            parts.append(_RUSSIAN_GENITIVE_SMALL[value % 10])
    return " ".join(parts)


def _inflect_russian_ordinal_word(word: str, form: str) -> str:
    if form == "masculine_nominative":
        return word
    if word == "третий":
        return {
            "masculine_genitive": "третьего",
            "masculine_prepositional": "третьем",
            "masculine_dative": "третьему",
            "feminine_nominative": "третья",
            "neuter_nominative": "третье",
            "plural_nominative": "третьи",
            "plural_genitive": "третьих",
        }[form]
    if not word.endswith(("ый", "ой")):
        return word
    stem = word[:-2]
    endings = {
        "masculine_genitive": "ого",
        "masculine_prepositional": "ом",
        "masculine_dative": "ому",
        "feminine_nominative": "ая",
        "neuter_nominative": "ое",
        "plural_nominative": "ые",
        "plural_genitive": "ых",
    }
    return stem + endings[form]


def _russian_ordinal_base_phrase(value: int) -> str:
    if value in _RUSSIAN_ORDINAL_BASE:
        return _RUSSIAN_ORDINAL_BASE[value]
    if value in _RUSSIAN_ORDINAL_HUNDREDS:
        return _RUSSIAN_ORDINAL_HUNDREDS[value]
    if value == 1000:
        return "тысячный"
    if value == 2000:
        return "двух тысячный"
    if value < 100:
        tens, unit = divmod(value, 10)
        if unit:
            return f"{_RUSSIAN_TENS[tens]} {_RUSSIAN_ORDINAL_BASE[unit]}"
    if value < 1000:
        remainder = value % 100
        if remainder:
            return f"{_RUSSIAN_HUNDREDS[value // 100]} {_russian_ordinal_base_phrase(remainder)}"
    if value < 1_000_000:
        remainder = value % 1000
        if remainder:
            prefix = russian_integer_words(value - remainder)
            return f"{prefix} {_russian_ordinal_base_phrase(remainder)}"
    return russian_integer_words(value)


def russian_ordinal_words(value: int, form: str = "masculine_nominative") -> str:
    if value <= 0:
        raise ValueError("Russian ordinals require a positive integer")
    phrase = _russian_ordinal_base_phrase(value)
    prefix, separator, final = phrase.rpartition(" ")
    inflected = _inflect_russian_ordinal_word(final if separator else phrase, form)
    return f"{prefix} {inflected}" if separator else inflected


def russian_date_words(token: str, day_form: str = "masculine_genitive") -> str | None:
    compact = re.sub(r"\s+", " ", token.strip())
    year_match = re.fullmatch(r"(\d{3,4})\s*(г\.|гг\.|год|года|году)", compact, flags=re.IGNORECASE)
    if year_match:
        year = int(year_match.group(1))
        suffix = year_match.group(2).lower()
        form = {
            "г.": "masculine_nominative",
            "гг.": "masculine_nominative",
            "год": "masculine_nominative",
            "года": "masculine_genitive",
            "году": "masculine_prepositional",
        }[suffix]
        label = "год" if suffix in {"г.", "гг.", "год"} else suffix
        return f"{russian_ordinal_words(year, form)} {label}"

    month_pattern = "|".join(_RUSSIAN_MONTHS_GENITIVE)
    dotted = re.fullmatch(r"(\d{1,2})[.]([01]?\d)[.](\d{3,4})", compact)
    if dotted:
        day, month_number, year = (int(value) for value in dotted.groups())
        if 1 <= month_number <= 12:
            return (
                f"{russian_ordinal_words(day, 'neuter_nominative')} "
                f"{_RUSSIAN_MONTHS_GENITIVE[month_number - 1]} "
                f"{russian_ordinal_words(year, 'masculine_genitive')} года"
            )
    full_date = re.fullmatch(
        rf"(\d{{1,2}})\s+({month_pattern})(?:\s+(\d{{3,4}}))?(?:\s+(?:года|г\.))?",
        compact,
        flags=re.IGNORECASE,
    )
    if full_date:
        day, month, year = full_date.groups()
        result = f"{russian_ordinal_words(int(day), day_form)} {month.lower()}"
        if year:
            result += f" {russian_ordinal_words(int(year), 'masculine_genitive')} года"
        return result
    return None


def russian_letters_words(token: str) -> str | None:
    compact = token.strip()
    if re.sub(r"[A-Za-zА-Яа-яЁё.\-\s]", "", compact):
        return None
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", compact)
    return " ".join(letter.lower() for letter in letters) if letters else None


def russian_digit_words(token: str) -> str | None:
    digits = re.sub(r"\D", "", token)
    return " ".join(_DIGIT_WORDS_RUSSIAN[int(value)] for value in digits) if digits else None


def russian_telephone_words(token: str) -> str | None:
    groups = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", token)
    spoken: list[str] = []
    for group in groups:
        if group.isdigit():
            leading = len(group) - len(group.lstrip("0"))
            parts = ["ноль"] * leading
            remainder = group[leading:]
            if remainder:
                parts.append(russian_integer_words(int(remainder)))
            spoken.append(" ".join(parts) if parts else "ноль")
        else:
            letters = russian_letters_words(group)
            spoken.append(letters or group.lower())
    return " sil ".join(spoken) if spoken else None


def _russian_feminine_cardinal(value: int) -> str:
    words = russian_integer_words(value)
    if value % 100 not in (11, 12):
        if value % 10 == 1:
            return words.rsplit(" ", 1)[0] + " одна" if " " in words else "одна"
        if value % 10 == 2:
            return words.rsplit(" ", 1)[0] + " две" if " " in words else "две"
    return words


def russian_decimal_words(token: str) -> str | None:
    match = re.fullmatch(r"([-+]?\d[\d ]*),([0-9]+)", token.strip())
    if not match:
        return None
    whole_text, fraction_text = match.groups()
    whole = int(whole_text.replace(" ", ""))
    if set(fraction_text) == {"0"}:
        return russian_integer_words(whole)
    fraction = int(fraction_text)
    whole_words = _russian_feminine_cardinal(whole) if abs(whole) % 10 in (1, 2) else russian_integer_words(whole)
    whole_unit = "целая" if abs(whole) % 10 == 1 and abs(whole) % 100 != 11 else "целых"
    fraction_words = _russian_feminine_cardinal(fraction)
    denominator = {
        1: "десятая" if fraction == 1 else "десятых",
        2: "сотая" if fraction == 1 else "сотых",
        3: "тысячная" if fraction == 1 else "тысячных",
    }.get(len(fraction_text))
    if denominator is None:
        return None
    return f"{whole_words} {whole_unit} и {fraction_words} {denominator}"


def russian_fraction_words(token: str) -> str | None:
    match = re.fullmatch(r"(\d+)[/⁄](\d+)", token.strip())
    if not match:
        return None
    numerator, denominator = (int(value) for value in match.groups())
    if denominator <= 0:
        return None
    numerator_words = _russian_feminine_cardinal(numerator) if numerator in (1, 2) else russian_integer_words(numerator)
    denominator_words = russian_ordinal_words(denominator, "plural_genitive")
    return f"{numerator_words} {denominator_words}"


def russian_time_words(token: str) -> str | None:
    match = re.fullmatch(
        r"(\d{1,2})[:.](\d{2})(?:\s+([A-Za-z]{2,5}))?",
        token.strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    if hour > 24 or minute > 59:
        return None
    hour_label = _russian_plural_unit(hour, "час", "часа", "часов")
    result = f"{russian_integer_words(hour)} {hour_label}"
    if minute:
        minute_label = _russian_plural_unit(minute, "минута", "минуты", "минут")
        result += f" {_russian_feminine_cardinal(minute)} {minute_label}"
    timezone_name = match.group(3)
    if timezone_name:
        result += " по часовому поясу " + " ".join(timezone_name.lower())
    return result


def _russian_plural_unit(value: int, one: str, few: str, many: str) -> str:
    tail = abs(value) % 100
    if 11 <= tail <= 14:
        return many
    return one if abs(value) % 10 == 1 else few if abs(value) % 10 in (2, 3, 4) else many


def russian_measure_words(token: str) -> str | None:
    match = re.fullmatch(
        r"([+]?[0-9][0-9 ]*)\s*(с\.|т\.|%|км\.?|см\.?|мм\.?|кг\.?|м\.?)",
        token.strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    value = int(match.group(1).replace(" ", ""))
    unit = match.group(2).lower()
    if unit == "с.":
        words = _russian_feminine_cardinal(value)
        label = _russian_plural_unit(value, "секунда", "секунды", "секунд")
    elif unit == "т.":
        words = _russian_feminine_cardinal(value)
        label = _russian_plural_unit(value, "тонна", "тонны", "тонн")
    elif unit == "%":
        words = russian_integer_words(value)
        label = _russian_plural_unit(value, "процент", "процента", "процентов")
    else:
        words = russian_integer_words(value)
        singular, few, many = {
            "км": ("километр", "километра", "километров"),
            "см": ("сантиметр", "сантиметра", "сантиметров"),
            "мм": ("миллиметр", "миллиметра", "миллиметров"),
            "кг": ("килограмм", "килограмма", "килограммов"),
            "м": ("метр", "метра", "метров"),
        }[unit.rstrip(".")]
        label = _russian_plural_unit(value, singular, few, many)
    return f"{words} {label}"


def russian_ordinal_token_words(token: str) -> str | None:
    match = re.fullmatch(
        r"[—-]?(\d+)-?(й|ый|го|му|м|я|е|х)",
        token.strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    value = int(match.group(1))
    suffix = match.group(2).lower()
    form = {
        "й": "masculine_nominative", "ый": "masculine_nominative",
        "го": "masculine_genitive", "му": "masculine_dative",
        "м": "masculine_prepositional", "я": "feminine_nominative",
        "е": "neuter_nominative", "х": "plural_genitive",
    }[suffix]
    return russian_ordinal_words(value, form)


def russian_contextual_token_words(
    token: str,
    previous: str | None,
    following: str | None,
) -> str | None:
    """Apply only high-confidence rules that require adjacent source tokens."""

    compact = token.strip()
    previous_value = (previous or "").strip().lower()
    following_value = (following or "").strip().lower()
    integer_match = re.fullmatch(r"[-+]?\d+", compact.replace(" ", ""))
    if integer_match and previous_value in {
        "от", "до", "около", "более", "менее", "свыше", "порядка", "без", "для",
    }:
        return russian_cardinal_genitive_words(int(integer_match.group()))
    if integer_match and following_value in {"год", "года", "году"}:
        value = int(integer_match.group())
        if previous_value == "с":
            return russian_ordinal_words(value, "masculine_genitive")
        if previous_value == "по":
            return russian_ordinal_words(value, "masculine_nominative")
    if integer_match and previous_value == "с" and following_value == "по":
        return russian_ordinal_words(int(integer_match.group()), "masculine_genitive")

    kilometer = re.fullmatch(r"(\d[\d ]*)\s*км\.?", compact, flags=re.IGNORECASE)
    if kilometer:
        value = int(kilometer.group(1).replace(" ", ""))
        if previous_value in {"в", "на"}:
            words = "одном" if value == 1 else russian_cardinal_genitive_words(value)
            label = "километре" if value == 1 else "километрах"
            return f"{words} {label}"
        if previous_value in {"от", "до", "около", "более", "менее", "свыше", "порядка"}:
            return f"{russian_cardinal_genitive_words(value)} километров"

    roman = _roman_integer(compact)
    if roman is not None and following_value.startswith("век"):
        if previous_value == "в":
            form = "masculine_prepositional"
        elif previous_value in {"начале", "конце", "середине"}:
            form = "masculine_genitive"
        else:
            form = "masculine_nominative"
        return russian_ordinal_words(roman, form)
    if roman is not None and following_value in {".", ",", ")", "—", "-"} and re.fullmatch(
        r"[А-ЯЁ][А-Яа-яЁё-]+", previous or ""
    ):
        form = "masculine_genitive" if previous_value.endswith(("а", "я")) else "masculine_nominative"
        return russian_ordinal_words(roman, form)
    return None


def _normalization_token_shape(token: str) -> str:
    if re.fullmatch(r"[-+]?\d[\d.,:/-]*", token):
        return "<NUM>"
    if re.fullmatch(r"[A-Za-z]+", token):
        return "<LATIN>"
    if re.fullmatch(r"[А-Яа-яЁё]+", token):
        return "<CYRILLIC>"
    return token.lower()


def _finalize_normalization_context_counts(
    counts: Counter[tuple[tuple[str, ...], str]],
) -> dict[tuple[str, ...], tuple[str, int, int]]:
    grouped: dict[tuple[str, ...], tuple[str, int, int]] = {}
    totals: Counter[tuple[str, ...]] = Counter()
    for (key, after), count in counts.items():
        totals[key] += count
        current = grouped.get(key)
        if current is None or count > current[1] or (count == current[1] and after < current[0]):
            grouped[key] = (after, count, 0)
    return {key: (after, best_count, totals[key]) for key, (after, best_count, _) in grouped.items()}


def build_normalization_context_maps(
    frame: pd.DataFrame,
) -> dict[str, dict[tuple[str, ...], tuple[str, int, int]]]:
    """Learn sentence-neighbor maps only for tokens with multiple observed outputs."""

    required = {"sentence_id", "before", "after"}
    if frame.empty or not required.issubset(frame.columns):
        return {}
    pairs = frame.loc[:, ["before", "after"]].astype(str).drop_duplicates()
    ambiguous = set(pairs.groupby("before", sort=False)["after"].nunique().loc[lambda value: value > 1].index)
    if not ambiguous:
        return {}
    sentence_ids = frame["sentence_id"].astype(str).tolist()
    before = frame["before"].astype(str).tolist()
    after = frame["after"].astype(str).tolist()
    counters: dict[str, Counter[tuple[tuple[str, ...], str]]] = {
        name: Counter() for name in ("exact", "prev", "next", "shape")
    }
    for index, token in enumerate(before):
        if token not in ambiguous:
            continue
        previous = before[index - 1] if index > 0 and sentence_ids[index - 1] == sentence_ids[index] else "<BOS>"
        following = before[index + 1] if index + 1 < len(before) and sentence_ids[index + 1] == sentence_ids[index] else "<EOS>"
        target = after[index]
        counters["exact"][((token, previous, following), target)] += 1
        counters["prev"][((token, previous), target)] += 1
        counters["next"][((token, following), target)] += 1
        counters["shape"][((token, _normalization_token_shape(previous), _normalization_token_shape(following)), target)] += 1
    return {name: _finalize_normalization_context_counts(counter) for name, counter in counters.items()}


def _normalization_context_prediction(
    token: str,
    previous: str,
    following: str,
    context_maps: dict[str, dict[tuple[str, ...], tuple[str, int, int]]],
) -> str | None:
    if not context_maps:
        return None
    keys = {
        "exact": (token, previous, following),
        "prev": (token, previous),
        "next": (token, following),
        "shape": (token, _normalization_token_shape(previous), _normalization_token_shape(following)),
    }
    thresholds = {
        "exact": (1, 0.50, 4),
        "prev": (2, 0.60, 3),
        "next": (2, 0.60, 3),
        "shape": (3, 0.67, 2),
    }
    candidates: list[tuple[float, int, int, str]] = []
    for name, key in keys.items():
        record = context_maps.get(name, {}).get(key)
        if record is None:
            continue
        after, best_count, total = record
        minimum, minimum_purity, specificity = thresholds[name]
        purity = best_count / total
        if best_count >= minimum and purity >= minimum_purity:
            candidates.append((purity, min(best_count, 20), specificity, after))
    return max(candidates)[3] if candidates else None


def normalize_frame_tokens(
    frame: pd.DataFrame,
    mapping: dict[str, str],
    language: str,
    context_maps: dict[str, dict[tuple[str, ...], tuple[str, int, int]]] | None = None,
) -> list[str]:
    """Normalize a token frame while preserving sentence-local context boundaries."""

    before = frame["before"].astype(str).tolist()
    if "sentence_id" in frame:
        sentence_ids = frame["sentence_id"].astype(str).tolist()
    else:
        sentence_ids = [str(index) for index in range(len(frame))]
    prediction: list[str] = []
    for index, token in enumerate(before):
        same_previous = index > 0 and sentence_ids[index - 1] == sentence_ids[index]
        same_following = index + 1 < len(before) and sentence_ids[index + 1] == sentence_ids[index]
        previous = before[index - 1] if same_previous else None
        following = before[index + 1] if same_following else None
        learned_context = _normalization_context_prediction(
            token,
            previous or "<BOS>",
            following or "<EOS>",
            context_maps or {},
        ) if language == "russian" else None
        contextual = learned_context or (
            russian_contextual_token_words(token, previous, following) if language == "russian" else None
        )
        prediction.append(contextual if contextual is not None else normalize_token(token, mapping, language))
    return prediction


def infer_normalization_class(token: str, language: str) -> str:
    compact = token.strip()
    if language == "english":
        month_pattern = "|".join(sorted(_ENGLISH_MONTH_ALIASES, key=len, reverse=True))
        if (
            re.search(r"(?i)(?:https?://|www\.)", compact)
            or re.fullmatch(r"[^\s@]+@[^\s@]+\.[A-Za-z]{2,}", compact)
            or re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._~-]*\.[A-Za-z]{2,}(?:[/:?#][^\s]*)?",
                compact,
            )
        ):
            return "ELECTRONIC"
        if (
            re.fullmatch(r"\d{4}", compact)
            and 1000 <= int(compact) <= 2099
        ) or (
            re.search(
                rf"(?i)(?<![A-Za-z])(?:{month_pattern})(?:\.)?(?![A-Za-z])",
                compact,
            )
            and re.search(r"\d", compact)
        ) or re.fullmatch(r"\d{3,4}s", compact) or re.fullmatch(
            r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}", compact
        ):
            return "DATE"
        if (
            re.fullmatch(r"(?:[A-Za-z]\.\s*){1,12}", compact)
            or re.fullmatch(r"[A-Z]{2,12}-", compact)
            or re.fullmatch(r"[A-Z]{2,12}s", compact)
            or re.fullmatch(r"[A-Z]{2,4}", compact)
        ):
            return "LETTERS"
    elif language == "russian":
        month_pattern = "|".join(_RUSSIAN_MONTHS_GENITIVE)
        if re.fullmatch(r"\d{1,2}[:.]\d{2}(?:\s+[A-Za-z]{2,5})?", compact):
            return "TIME"
        if re.fullmatch(r"\d{3,4}\s*(?:г\.|гг\.|год|года|году)", compact, flags=re.IGNORECASE) or re.fullmatch(
            r"\d{1,2}[.]\d{1,2}[.]\d{3,4}", compact
        ) or re.fullmatch(
            rf"\d{{1,2}}\s+(?:{month_pattern})(?:\s+\d{{3,4}})?(?:\s+(?:года|г\.))?",
            compact,
            flags=re.IGNORECASE,
        ):
            return "DATE"
        if re.fullmatch(r"[—-]?\d+-?(?:й|ый|го|му|м|я|е|х)", compact, flags=re.IGNORECASE):
            return "ORDINAL"
        if re.fullmatch(r"[-+]?\d[\d ]*,\d+", compact):
            return "DECIMAL"
        if re.fullmatch(
            r"[+]?\d[\d ]*\s*(?:с\.|т\.|%|км\.?|см\.?|мм\.?|кг\.?|м\.?)",
            compact,
            flags=re.IGNORECASE,
        ):
            return "MEASURE"
        if re.fullmatch(r"0\d+", compact):
            return "DIGIT"
        if (
            re.fullmatch(r"(?:[A-Za-zА-Яа-яЁё]\.\s*){1,12}", compact)
            or re.fullmatch(r"[A-ZА-ЯЁ]{2,5}-", compact)
            or re.fullmatch(r"[A-ZА-ЯЁ]{2,5}", compact)
        ):
            return "LETTERS"
    if re.fullmatch(r"[-+]?\d+(?:st|nd|rd|th)", compact, flags=re.IGNORECASE):
        return "ORDINAL"
    if re.fullmatch(r"[-+]?\d+[/⁄]\d+", compact):
        return "FRACTION"
    if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", compact):
        return "TIME"
    if re.fullmatch(
        r"[$£€₽]\s*[-+]?\d[\d,]*(?:\.\d+)?(?:\s*(?:thousand|million|billion))?",
        compact,
        flags=re.IGNORECASE,
    ):
        return "MONEY"
    if re.fullmatch(r"[-+]?\d[\d,.]*\s*%", compact):
        return "MEASURE"
    if re.fullmatch(
        r"[-+]?\d[\d,.]*\s*/?\s*(?:km²|km2|m²|m2|kg|km|cm|mm|mi|m|g|lb|ft|hz|mhz|ghz)",
        compact,
        flags=re.IGNORECASE,
    ):
        return "MEASURE"
    if language == "english" and (
        re.fullmatch(r"0\d+", compact)
        or re.fullmatch(r"\d+-", compact)
    ):
        return "DIGIT"
    if len(re.sub(r"\D", "", compact)) >= 7 and re.fullmatch(r"[+\d() .-]+", compact):
        return "TELEPHONE"
    if re.fullmatch(r"[-+]?(?:\d+\.\d+|\.\d+)", compact.replace(",", "")):
        return "DECIMAL"
    if re.fullmatch(r"[-+]?\d+", compact.replace(",", "")):
        return "CARDINAL"
    return "PLAIN"


def _english_cardinal_or_decimal(token: str) -> str:
    compact = token.strip().replace(",", "")
    if "." not in compact:
        return english_integer_words(int(compact))
    left, right = compact.split(".", 1)
    if not right or not right.isdigit():
        raise ValueError(f"Invalid English decimal: {token!r}")
    sign = ""
    if left in {"+", "-"}:
        sign = "minus " if left == "-" else ""
        left = ""
    left_words = english_integer_words(int(left)) if left else ""
    decimal_words = " ".join(_ENGLISH_DIGIT_WORDS[int(value)] for value in right)
    prefix = f"{left_words} " if left_words else ""
    return f"{sign}{prefix}point {decimal_words}"


def normalize_token(
    token: str,
    mapping: dict[str, str],
    language: str,
    token_class: str | None = None,
) -> str:
    if token in mapping:
        return mapping[token]
    token_class = (token_class or infer_normalization_class(token, language)).upper()
    if language == "english":
        compact = token.strip()
        try:
            if token_class == "ORDINAL":
                numeric = re.sub(r"(?i)(st|nd|rd|th)$", "", compact.replace(",", ""))
                if re.fullmatch(r"[-+]?\d+", numeric):
                    return english_ordinal_words(int(numeric))
                roman = _roman_integer(compact)
                if roman is not None:
                    return english_ordinal_words(roman)
            if token_class == "DATE":
                converted = english_date_words(compact)
                if converted is not None:
                    return converted
            if token_class == "LETTERS":
                converted = english_letters_words(compact)
                if converted is not None:
                    return converted
            if token_class == "DIGIT":
                converted = english_digit_words(compact)
                if converted is not None:
                    return converted
            if token_class == "ELECTRONIC":
                converted = english_electronic_words(compact)
                if converted is not None:
                    return converted
            if token_class == "FRACTION":
                numerator, denominator = (int(value) for value in re.split(r"[/⁄]", compact))
                denominator_words = english_ordinal_words(denominator)
                if abs(numerator) != 1:
                    denominator_words += "s"
                return f"{english_integer_words(numerator)} {denominator_words}"
            if token_class == "TIME":
                hour, minute, *second = (int(value) for value in compact.split(":"))
                result = english_integer_words(hour)
                if minute:
                    result += " " + ("o " if minute < 10 else "") + english_integer_words(minute)
                if second and second[0]:
                    result += " and " + english_integer_words(second[0]) + " seconds"
                return result
            if token_class == "MONEY":
                converted = english_money_words(compact)
                if converted is not None:
                    return converted
            if token_class == "MEASURE":
                converted = english_measure_words(compact)
                if converted is not None:
                    return converted
            if token_class == "TELEPHONE":
                converted = english_telephone_words(compact)
                if converted is not None:
                    return converted
            if token_class in {"CARDINAL", "DECIMAL"}:
                normalized = compact.replace(",", "")
                value = float(normalized)
                if abs(value) <= 10**15:
                    return _english_cardinal_or_decimal(normalized)
        except (ValueError, IndexError, ZeroDivisionError):
            pass
    elif language == "russian":
        compact = token.strip()
        try:
            if token_class == "DATE":
                converted = russian_date_words(compact)
                if converted is not None:
                    return converted
            if token_class == "LETTERS":
                converted = russian_letters_words(compact)
                if converted is not None:
                    return converted
            if token_class == "DIGIT":
                converted = russian_digit_words(compact)
                if converted is not None:
                    return converted
            if token_class == "ORDINAL":
                converted = russian_ordinal_token_words(compact)
                if converted is not None:
                    return converted
            if token_class == "DECIMAL":
                converted = russian_decimal_words(compact)
                if converted is not None:
                    return converted
            if token_class == "FRACTION":
                converted = russian_fraction_words(compact)
                if converted is not None:
                    return converted
            if token_class == "TIME":
                converted = russian_time_words(compact)
                if converted is not None:
                    return converted
            if token_class == "MEASURE":
                converted = russian_measure_words(compact)
                if converted is not None:
                    return converted
            numeric = compact.replace(" ", "")
            if token_class == "CARDINAL" and re.fullmatch(r"[-+]?\d+", numeric):
                return russian_integer_words(int(numeric))
            if token_class == "MONEY" and compact.startswith("₽"):
                return russian_integer_words(int(compact[1:].replace(" ", ""))) + " рублей"
            if token_class == "TELEPHONE":
                converted = russian_telephone_words(compact)
                if converted is not None:
                    return converted
        except (ValueError, IndexError):
            pass
    return token


def _load_normalization_map(path: Path, max_rows: int, chunksize: int) -> tuple[dict[str, str], int]:
    counts: Counter[tuple[str, str]] = Counter()
    seen = 0
    for chunk in pd.read_csv(path, compression="zip", usecols=["before", "after"], dtype=str, keep_default_na=False, chunksize=chunksize):
        counts.update(zip(chunk["before"], chunk["after"]))
        seen += len(chunk)
        if max_rows and seen >= max_rows:
            break
    best: dict[str, tuple[int, str]] = {}
    for (before, after), count in counts.items():
        current = best.get(before)
        if current is None or count > current[0] or (count == current[0] and after < current[1]):
            best[before] = (count, after)
    return {before: value[1] for before, value in best.items()}, seen


def align_text_normalization_submission_by_id(
    sample: pd.DataFrame,
    test: pd.DataFrame,
    prediction: Sequence[str],
) -> pd.DataFrame:
    """Align token predictions to sample order using the official compound ID."""

    required_test = {"sentence_id", "token_id"}
    missing_test = required_test.difference(test.columns)
    if missing_test:
        raise RuntimeError(f"Text normalization test is missing ID columns: {sorted(missing_test)}")
    if "id" not in sample or "after" not in sample:
        raise RuntimeError("Text normalization sample must contain id and after columns")
    if len(prediction) != len(test):
        raise RuntimeError("Text normalization prediction/test row count mismatch")

    test_ids = (
        test["sentence_id"].astype(str)
        + "_"
        + test["token_id"].astype(str)
    )
    sample_ids = sample["id"].astype(str)
    if test_ids.duplicated().any():
        raise RuntimeError("Text normalization test IDs are not unique")
    if sample_ids.duplicated().any():
        raise RuntimeError("Text normalization sample IDs are not unique")
    if set(test_ids) != set(sample_ids):
        raise RuntimeError("Text normalization test/sample ID sets differ")

    keyed = pd.Series(np.asarray(prediction, dtype=object), index=test_ids)
    aligned = sample.copy()
    aligned["after"] = sample_ids.map(keyed)
    if aligned["after"].isna().any():
        raise RuntimeError("Text normalization aligned predictions contain missing values")
    # An empty token is a valid deterministic normalizer output, but an empty CSV
    # field is parsed back as NaN by the official MLE-Bench reader.  Preserve a
    # non-empty serialized value so the strict submission gate and grader see a
    # complete string column.  This affects only truly zero-length predictions;
    # existing whitespace and every non-empty prediction remain unchanged.
    aligned.loc[aligned["after"].eq(""), "after"] = " "
    return aligned


def apply_russian_transliteration_model(
    training_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    mapping: dict[str, str],
    prediction: list[str],
    *,
    epochs: int,
    seed: int,
    device: str,
) -> tuple[list[str], dict[str, Any]]:
    tokens = prediction_frame["before"].astype(str).tolist()
    candidates = [
        token for token in tokens
        if token not in mapping and re.fullmatch(r"[A-Za-z]{2,40}", token)
    ]
    model_prediction, diagnostics = ru_transliteration.predict_russian_transliterations(
        training_frame,
        candidates,
        epochs=epochs,
        seed=seed,
        device=device,
    )
    result = list(prediction)
    applied = 0
    for index, token in enumerate(tokens):
        value = model_prediction.get(token.lower()) if token not in mapping else None
        if value is not None:
            result[index] = value
            applied += 1
    diagnostics = {**diagnostics, "applied_rows": applied}
    return result, diagnostics


def _run_text_normalization(args: argparse.Namespace, task_dir: Path, logger: Any, language: str) -> dict[str, Any]:
    from sklearn.model_selection import GroupKFold

    cid = f"text-normalization-challenge-{language}-language"
    resolved = resolve_competition(cid, args.data_root)
    prefix = "en" if language == "english" else "ru"
    train_path = resolved.public_dir / f"{prefix}_train.csv.zip"
    test_path = resolved.public_dir / f"{prefix}_test_2.csv.zip"
    started = time.perf_counter()
    header = pd.read_csv(train_path, compression="zip", nrows=0).columns.tolist()
    usecols = [column for column in ("sentence_id", "token_id", "class", "before", "after") if column in header]
    cv = pd.read_csv(train_path, compression="zip", usecols=usecols, dtype=str,
                     keep_default_na=False, nrows=args.wave2_normalization_cv_rows)
    groups = cv["sentence_id"].astype(str) if "sentence_id" in cv else pd.Series(np.arange(len(cv)) // 20)
    fold_count = max(2, min(args.wave2_normalization_folds, groups.nunique()))
    cv_prediction = np.empty(len(cv), dtype=object)
    oracle_cv_prediction = np.empty(len(cv), dtype=object)
    fold_assignment = np.full(len(cv), -1, dtype=np.int16)
    transliteration_folds: list[dict[str, Any]] = []
    for fold, (train_indices, valid_indices) in enumerate(GroupKFold(n_splits=fold_count).split(cv, groups=groups)):
        cv_train = cv.iloc[train_indices]
        cv_valid = cv.iloc[valid_indices]
        cv_map = build_normalization_map(cv_train)
        cv_context_maps = build_normalization_context_maps(cv_train) if language == "russian" else {}
        fold_prediction = normalize_frame_tokens(cv_valid, cv_map, language, cv_context_maps)
        if language == "russian" and args.wave2_russian_transliteration_epochs > 0:
            fold_prediction, fold_transliteration = apply_russian_transliteration_model(
                cv_train,
                cv_valid,
                cv_map,
                fold_prediction,
                epochs=args.wave2_russian_transliteration_epochs,
                seed=args.seed + fold,
                device=args.wave2_russian_transliteration_device,
            )
            transliteration_folds.append({"fold": fold, **fold_transliteration})
        cv_prediction[valid_indices] = fold_prediction
        if "class" in cv:
            oracle_cv_prediction[valid_indices] = [
                normalize_token(value, cv_map, language, token_class)
                for value, token_class in zip(
                    cv.iloc[valid_indices]["before"],
                    cv.iloc[valid_indices]["class"],
                )
            ]
        else:
            oracle_cv_prediction[valid_indices] = cv_prediction[valid_indices]
        fold_assignment[valid_indices] = fold
    if np.any(fold_assignment < 0):
        raise RuntimeError("Text-normalization OOF coverage is incomplete")
    cv_score = compute_metric("token_exact_accuracy", cv["after"], cv_prediction)
    changed = cv["before"].to_numpy() != cv["after"].to_numpy()
    diagnostics: dict[str, Any] = {
        "inferred_class_overall_accuracy": float(cv_score),
        "oracle_class_overall_accuracy": float(
            compute_metric("token_exact_accuracy", cv["after"], oracle_cv_prediction)
        ),
        "identity_accuracy": float(np.mean(cv_prediction[~changed] == cv.loc[~changed, "after"])) if np.any(~changed) else None,
        "changed_accuracy": float(np.mean(cv_prediction[changed] == cv.loc[changed, "after"])) if np.any(changed) else None,
        "changed_fraction": float(np.mean(changed)),
        "folds": fold_count,
        "transliteration_folds": transliteration_folds,
    }
    np.savez_compressed(
        task_dir / f"{language}_normalization_oof.npz",
        prediction=cv_prediction.astype(str),
        oracle_prediction=oracle_cv_prediction.astype(str),
        truth=cv["after"].to_numpy(),
        fold=fold_assignment,
    )
    mapping, rows_seen = _load_normalization_map(
        train_path, args.wave2_normalization_max_rows, args.wave2_normalization_chunk_rows
    )
    context_maps = build_normalization_context_maps(cv) if language == "russian" else {}
    test = pd.read_csv(test_path, compression="zip", dtype=str, keep_default_na=False)
    sample = pd.read_csv(resolved.sample_submission_path, dtype=str, keep_default_na=False)
    prediction = normalize_frame_tokens(test, mapping, language, context_maps)
    test_transliteration: dict[str, Any] = {}
    if language == "russian" and args.wave2_russian_transliteration_epochs > 0:
        prediction, test_transliteration = apply_russian_transliteration_model(
            cv,
            test,
            mapping,
            prediction,
            epochs=args.wave2_russian_transliteration_epochs,
            seed=args.seed,
            device=args.wave2_russian_transliteration_device,
        )
    sample = align_text_normalization_submission_by_id(sample, test, prediction)
    budget = {"seed": args.seed, "language": language, "training_rows_scanned": rows_seen,
               "mapping_entries": len(mapping), "test_rows": len(test), "cv_rows": len(cv),
               "cv_folds": fold_count,
               "context_mapping_entries": sum(len(values) for values in context_maps.values()),
               "test_transliteration": test_transliteration}
    return wave0.finalize_scored_task(
        competition_id=cid, submission=sample, cv_score=cv_score, args=args, task_dir=task_dir,
        budget=budget, extra={"runtime_seconds_model": time.perf_counter() - started,
                              "model_family": "grouped_oof_frequency_map_plus_class_rules",
                              "normalization_diagnostics": diagnostics, "budget": budget},
    )


def run_text_normalization_english(args: argparse.Namespace, task_dir: Path, logger: Any) -> dict[str, Any]:
    return _run_text_normalization(args, task_dir, logger, "english")


def run_text_normalization_russian(args: argparse.Namespace, task_dir: Path, logger: Any) -> dict[str, Any]:
    return _run_text_normalization(args, task_dir, logger, "russian")


RUNNERS: dict[str, Callable[[argparse.Namespace, Path, Any], dict[str, Any]]] = {
    "aptos2019-blindness-detection": run_aptos,
    "dog-breed-identification": run_dog_breed,
    "histopathologic-cancer-detection": run_histopath,
    "jigsaw-toxic-comment-classification-challenge": run_jigsaw,
    "mlsp-2013-birds": run_birds,
    "nomad2018-predict-transparent-conductors": run_nomad,
    "plant-pathology-2020-fgvc7": run_plant,
    "ranzcr-clip-catheter-line-classification": run_ranzcr,
    "text-normalization-challenge-english-language": run_text_normalization_english,
    "text-normalization-challenge-russian-language": run_text_normalization_russian,
    "the-icml-2013-whale-challenge-right-whale-redux": run_whale,
}
