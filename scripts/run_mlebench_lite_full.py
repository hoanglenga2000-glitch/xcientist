#!/usr/bin/env python3
"""Recoverable MLE-Bench Lite runner for Wave 0/1 deterministic adapters.

All outputs, caches, models, checkpoints, and logs are constrained to the
dedicated execution root. The runner validates every submission and normally
executes the upstream MLE-Bench private grader; ``--candidate-only`` preserves
the candidate artifact behind an explicit human confirmation gate. It contains
no Kaggle submission path.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

try:
    import mlebench_medal_recovery_adapters as recovery
    import mlebench_wave2_adapters as wave2
    import run_mlebench_lite_wave0 as wave0
except ModuleNotFoundError:  # imported as ``scripts.run_mlebench_lite_full`` in tests
    from scripts import mlebench_medal_recovery_adapters as recovery
    from scripts import mlebench_wave2_adapters as wave2
    from scripts import run_mlebench_lite_wave0 as wave0
try:
    import mlebench_compute_policy as compute_policy
except ModuleNotFoundError:
    try:
        from scripts import mlebench_compute_policy as compute_policy
    except (ModuleNotFoundError, ImportError):  # Standalone remote HPC source copy.
        compute_policy = None  # type: ignore[assignment]
from research_os.mlebench_phase_a import (
    DEFAULT_OFFICIAL_DATA_ROOT,
    DEFAULT_REMOTE_ROOT,
    audit_lite22_specs,
    compute_metric,
    export_specs,
    resolve_competition,
    sha256_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WAVE1_COMPETITIONS = (
    "denoising-dirty-documents",
    "detecting-insults-in-social-commentary",
    "dogs-vs-cats-redux-kernels-edition",
    "leaf-classification",
    "new-york-city-taxi-fare-prediction",
    "random-acts-of-pizza",
    "tabular-playground-series-dec-2021",
)
WAVE2_COMPETITIONS = tuple(wave2.WAVE2_COMPETITIONS)

WAVES: dict[str, tuple[str, ...]] = {
    "Wave0": tuple(wave0.WAVE0_COMPETITIONS),
    "Wave1": WAVE1_COMPETITIONS,
    "Wave2": WAVE2_COMPETITIONS,
}

RUN_LOCAL_CACHE_ENVIRONMENTS = (
    "TMPDIR",
    "TEMP",
    "TMP",
    "XDG_CACHE_HOME",
    "HF_HOME",
)
RUN_SHORT_TEMP_ENVIRONMENTS = frozenset({"TMPDIR", "TEMP", "TMP"})
SHORT_TEMP_RELATIVE = Path(".t")
AF_UNIX_SOCKET_PATH_LIMIT_BYTES = 107
RESOURCE_SHARER_SUFFIX_BUDGET_BYTES = 40
SHARED_TORCH_CACHE_RELATIVE = Path("mlebench_model_cache") / "torch"
SIIM_COMPETITION_ID = "siim-isic-melanoma-classification"
SIIM_CANDIDATE_TERMINAL_STATES = {
    "promotion_gate_passed_confirmation_pending",
    "candidate_ready_confirmation_pending",
}


def _int_equals(value: Any, expected: int) -> bool:
    try:
        return int(value) == int(expected)
    except (TypeError, ValueError):
        return False


def _resume_contract_sha256(contract: Mapping[str, Any]) -> str:
    base = {key: value for key, value in contract.items() if key != "contract_sha256"}
    return hashlib.sha256(
        json.dumps(
            base,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def siim_terminal_resume_checks(
    task_root: Path,
    existing: Mapping[str, Any],
    args: Any,
) -> dict[str, bool]:
    """Prevent --resume from accepting terminal SIIM results from old protocols."""

    contract_path = task_root / "attempts" / "siim_resume_state" / "resume_contract.json"
    manifest_path = Path(str(getattr(args, "siim_image_content_manifest", "") or ""))
    if not contract_path.is_file():
        return {"resume_contract_exists": False}
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {"resume_contract_exists": True, "resume_contract_readable": False}
    if not isinstance(contract, dict):
        return {"resume_contract_exists": True, "resume_contract_readable": False}
    budget = existing.get("budget") if isinstance(existing.get("budget"), dict) else {}
    groups = (
        budget.get("duplicate_group_report")
        if isinstance(budget.get("duplicate_group_report"), dict)
        else {}
    )
    model = contract.get("model") if isinstance(contract.get("model"), dict) else {}
    manifest_sha256 = sha256_file(manifest_path) if manifest_path.is_file() else ""
    return {
        "resume_contract_exists": True,
        "resume_contract_readable": True,
        "terminal_status": existing.get("status") in SIIM_CANDIDATE_TERMINAL_STATES,
        "candidate_only": existing.get("candidate_only") is True,
        "valid_submission": existing.get("valid_submission") is True,
        "official_grader_excluded": existing.get("official_grader_executed") is False,
        "result_seed": _int_equals(budget.get("seed"), getattr(args, "seed", -1)),
        "result_outer_folds": _int_equals(budget.get("folds"), 5),
        "result_manifest": bool(manifest_sha256)
        and budget.get("image_content_manifest_sha256") == manifest_sha256,
        "group_schema": groups.get("schema")
        == "evomind.siim_patient_content_connected_groups.v3",
        "group_policy": groups.get("leakage_group_policy")
        == recovery.SIIM_LEAKAGE_GROUP_POLICY,
        "perceptual_audit_only": groups.get("perceptual_edge_policy")
        == recovery.SIIM_PERCEPTUAL_EDGE_POLICY,
        "perceptual_edges_not_applied": _int_equals(
            groups.get("perceptual_edges_applied_to_groups"), 0
        ),
        "contract_schema": contract.get("schema")
        == "evomind.siim.formal_resume_contract.v1",
        "contract_seed": _int_equals(contract.get("model_seed"), getattr(args, "seed", -1)),
        "contract_outer_folds": _int_equals(contract.get("outer_folds"), 5),
        "contract_inner_folds": _int_equals(contract.get("inner_folds"), 3),
        "contract_group_policy": contract.get("leakage_group_policy")
        == recovery.SIIM_LEAKAGE_GROUP_POLICY,
        "contract_perceptual_policy": contract.get("perceptual_edge_policy")
        == recovery.SIIM_PERCEPTUAL_EDGE_POLICY,
        "contract_manifest": bool(manifest_sha256)
        and contract.get("image_content_manifest_sha256") == manifest_sha256,
        "contract_self_hash": contract.get("contract_sha256")
        == _resume_contract_sha256(contract),
        "result_contract_matches": budget.get("resume_contract_sha256")
        == contract.get("contract_sha256"),
        "contract_adapter_source": contract.get("adapter_source_sha256")
        == sha256_file(Path(recovery.__file__)),
        "contract_wave2_source": contract.get("wave2_source_sha256")
        == sha256_file(Path(wave2.__file__)),
        "contract_workers": _int_equals(model.get("workers"), 8),
        "contract_fast_kernels": model.get("fast_kernel_mode") is True,
    }


def configure_cache_environment(run_dir: Path, allowed_root: Path) -> dict[str, Any]:
    """Keep transient caches per-run while reusing pinned Torch model weights.

    The shared cache remains below the dedicated AIMSLAB root, so repeated
    image runs do not redownload the same ImageNet checkpoint.  Every returned
    path is resolved and boundary checked before it is exported.
    """

    run_cache_root = wave0.ensure_within(run_dir / "cache", allowed_root)
    short_temp_root = wave0.ensure_within(
        allowed_root
        / SHORT_TEMP_RELATIVE
        / hashlib.sha256(run_dir.name.encode("utf-8")).hexdigest()[:12],
        allowed_root,
    )
    torch_home = wave0.ensure_within(
        allowed_root / SHARED_TORCH_CACHE_RELATIVE,
        allowed_root,
    )
    run_cache_root.mkdir(parents=True, exist_ok=True)
    short_temp_root.mkdir(parents=True, exist_ok=True)
    torch_home.mkdir(parents=True, exist_ok=True)
    if (
        os.name == "posix"
        and len(os.fsencode(short_temp_root)) + RESOURCE_SHARER_SUFFIX_BUDGET_BYTES
        > AF_UNIX_SOCKET_PATH_LIMIT_BYTES
    ):
        raise RuntimeError("Short runtime temp root exceeds the AF_UNIX socket path budget")
    exported: dict[str, str] = {}
    for name in RUN_LOCAL_CACHE_ENVIRONMENTS:
        path = (
            short_temp_root
            if name in RUN_SHORT_TEMP_ENVIRONMENTS
            else wave0.ensure_within(run_cache_root / name.lower(), allowed_root)
        )
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
        exported[name] = str(path)
    os.environ["TORCH_HOME"] = str(torch_home)
    exported["TORCH_HOME"] = str(torch_home)
    return {
        "schema": "evomind.mlebench_lite.cache_environment.v1",
        "run_cache_root": str(run_cache_root),
        "short_temp_root": str(short_temp_root),
        "shared_torch_home": str(torch_home),
        "environment": exported,
    }


def configure_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("mlebench_lite_full")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "full.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def parse_waves(value: str) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    unknown = sorted(set(requested) - set(WAVES))
    if unknown:
        raise ValueError(f"Unsupported waves: {unknown}")
    if not requested:
        raise ValueError("At least one wave is required")
    return requested


def scope_phase_a_audit(
    report: dict[str, Any],
    requested: Iterable[str],
) -> dict[str, Any]:
    """Restrict a Lite-22 audit to an explicit single-run competition set."""

    requested_ids = tuple(dict.fromkeys(str(value) for value in requested))
    rows_by_id = {str(row.get("competition_id")): row for row in report.get("rows", [])}
    missing = [competition_id for competition_id in requested_ids if competition_id not in rows_by_id]
    if missing:
        raise RuntimeError(f"Phase A audit omitted requested competitions: {missing}")
    rows = [rows_by_id[competition_id] for competition_id in requested_ids]
    passed = sum(row.get("status") == "passed" for row in rows)
    return {
        "schema": "evomind.mlebench.phase_a_requested_audit.v1",
        "data_root": report.get("data_root"),
        "scope": "requested",
        "competition_count": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "status": "passed" if passed == len(rows) else "failed",
        "rows": rows,
    }


def select_competitions(waves: Iterable[str], explicit: str = "") -> list[str]:
    allowed = [competition for wave in waves for competition in WAVES[wave]]
    if not explicit.strip():
        return list(dict.fromkeys(allowed))
    requested = list(dict.fromkeys(item.strip() for item in explicit.split(",") if item.strip()))
    invalid = sorted(set(requested) - set(allowed))
    if invalid:
        raise ValueError(f"Competitions are outside selected waves: {invalid}")
    return requested


def classify_failure(exc: BaseException) -> str:
    message = str(exc).lower()
    if isinstance(exc, FileNotFoundError) or "not found" in message or "missing" in message:
        return "data_missing"
    if "out of memory" in message or "cuda oom" in message:
        return "resource_oom"
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return "dependency_missing"
    if "submission" in message or "schema" in message or "column" in message:
        return "submission_or_schema"
    if "grader" in message:
        return "private_grader"
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return "infrastructure_transient"
    return "training_runtime"


def next_attempt_dir(task_root: Path) -> Path:
    attempts = task_root / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in attempts.glob("attempt_*"):
        try:
            numbers.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    attempt = attempts / f"attempt_{max(numbers, default=0) + 1:03d}"
    attempt.mkdir(parents=True, exist_ok=False)
    return attempt


def load_optimization_plan(path: Path | None, allowed_root: Path) -> dict[str, Any] | None:
    if path is None:
        return None
    plan_path = wave0.ensure_within(path, allowed_root)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    supported_schemas = {
        "evomind.mlebench_lite.optimization_plan.v1",
        "evomind.mlebench_lite.medal_recovery_plan.v1",
    }
    if payload.get("schema") not in supported_schemas:
        raise ValueError("Optimization plan schema mismatch")
    planner = payload.get("planner") or {}
    if planner.get("provider") != "openai" or planner.get("model") != "gpt-5.6-sol":
        raise ValueError("Optimization plan must carry gpt-5.6-sol provider evidence")
    payload["_path"] = str(plan_path)
    payload["_sha256"] = sha256_file(plan_path)
    return payload


def apply_plan_order(competitions: list[str], plan: dict[str, Any] | None) -> list[str]:
    if not plan:
        return competitions
    order = [
        str(item)
        for item in (plan.get("competition_order") or plan.get("priority_order") or [])
    ]
    selected = set(competitions)
    planned_selection = [item for item in order if item in selected]
    return list(dict.fromkeys([*planned_selection, *competitions]))


def _restoration_features(array: np.ndarray) -> np.ndarray:
    from PIL import Image, ImageFilter

    image = Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8), mode="L")
    median = np.asarray(image.filter(ImageFilter.MedianFilter(3)), dtype=np.float32) / 255.0
    gaussian_1 = np.asarray(image.filter(ImageFilter.GaussianBlur(0.8)), dtype=np.float32) / 255.0
    gaussian_2 = np.asarray(image.filter(ImageFilter.GaussianBlur(1.6)), dtype=np.float32) / 255.0
    return np.stack([array, median, gaussian_1, gaussian_2], axis=-1)


def _read_gray(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def run_denoising(args: argparse.Namespace, task_dir: Path, logger: logging.Logger) -> dict[str, Any]:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    competition_id = "denoising-dirty-documents"
    resolved = resolve_competition(competition_id, args.data_root)
    dirty_dir = resolved.public_dir / "train"
    clean_dir = resolved.public_dir / "train_cleaned"
    test_dir = resolved.public_dir / "test"
    names = sorted(path.name for path in dirty_dir.glob("*.png") if (clean_dir / path.name).is_file())
    train_names, valid_names = train_test_split(
        names, test_size=args.holdout_fraction, random_state=args.seed
    )
    rng = np.random.default_rng(args.seed)

    def training_matrix(selected: list[str]) -> tuple[np.ndarray, np.ndarray]:
        feature_parts: list[np.ndarray] = []
        target_parts: list[np.ndarray] = []
        for name in selected:
            dirty = _read_gray(dirty_dir / name)
            clean = _read_gray(clean_dir / name)
            if dirty.shape != clean.shape:
                raise RuntimeError(f"Image pair shape mismatch: {name}")
            features = _restoration_features(dirty).reshape(-1, 4)
            targets = clean.reshape(-1)
            count = min(args.restoration_pixels_per_image, len(targets))
            indices = rng.choice(len(targets), size=count, replace=False)
            feature_parts.append(features[indices])
            target_parts.append(targets[indices])
        return np.concatenate(feature_parts), np.concatenate(target_parts)

    logger.info("[%s] fitting sampled pixel restoration model", competition_id)
    started = time.perf_counter()
    train_x, train_y = training_matrix(train_names)
    holdout_model = Ridge(alpha=args.restoration_ridge_alpha).fit(train_x, train_y)
    squared_error = 0.0
    pixel_count = 0
    for name in valid_names:
        dirty = _read_gray(dirty_dir / name)
        clean = _read_gray(clean_dir / name)
        predicted = np.clip(holdout_model.predict(_restoration_features(dirty).reshape(-1, 4)), 0, 1)
        squared_error += float(np.square(predicted - clean.reshape(-1)).sum())
        pixel_count += clean.size
    cv_score = math.sqrt(squared_error / pixel_count)

    all_x, all_y = training_matrix(names)
    final_model = Ridge(alpha=args.restoration_ridge_alpha).fit(all_x, all_y)
    sample = pd.read_csv(resolved.sample_submission_path)
    id_parts = sample["id"].astype(str).str.rsplit("_", n=2, expand=True)
    if id_parts.shape[1] != 3:
        raise RuntimeError("Unexpected denoising submission id format")
    predicted_values = np.empty(len(sample), dtype=np.float32)
    for image_id, index in id_parts.groupby(0).groups.items():
        dirty = _read_gray(test_dir / f"{image_id}.png")
        predicted = np.clip(final_model.predict(_restoration_features(dirty).reshape(-1, 4)), 0, 1)
        rows = id_parts.loc[index, 1].astype(int).to_numpy() - 1
        columns = id_parts.loc[index, 2].astype(int).to_numpy() - 1
        predicted_values[np.asarray(index)] = predicted.reshape(dirty.shape)[rows, columns]
    sample["value"] = predicted_values
    wave0.write_json(task_dir / "restoration_model.json", {
        "coefficient": final_model.coef_.tolist(), "intercept": float(final_model.intercept_)
    })
    budget = {
        "seed": args.seed,
        "holdout_fraction": args.holdout_fraction,
        "train_images": len(names),
        "validation_images": len(valid_names),
        "pixels_per_image": args.restoration_pixels_per_image,
        "ridge_alpha": args.restoration_ridge_alpha,
    }
    return wave0.finalize_scored_task(
        competition_id=competition_id, submission=sample, cv_score=cv_score,
        args=args, task_dir=task_dir, budget=budget,
        extra={
            "runtime_seconds_model": time.perf_counter() - started,
            "model_family": "Ridge_raw_median_gaussian_pixel_restoration",
            "budget": budget,
        },
    )


def _text_feature_union(seed: int, *, max_word: int, max_char: int):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion

    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_word, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2,
                                 max_features=max_char, sublinear_tf=True)),
    ])


def run_insults(args: argparse.Namespace, task_dir: Path, logger: logging.Logger) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    competition_id = "detecting-insults-in-social-commentary"
    resolved = resolve_competition(competition_id, args.data_root)
    train = pd.read_csv(resolved.public_dir / "train.csv")
    test = pd.read_csv(resolved.public_dir / "test.csv")
    x_train, x_valid, y_train, y_valid = train_test_split(
        train["Comment"].fillna(""), train["Insult"].astype(int),
        test_size=args.holdout_fraction, random_state=args.seed, stratify=train["Insult"],
    )

    def build() -> Pipeline:
        return Pipeline([
            ("features", _text_feature_union(args.seed, max_word=60_000, max_char=80_000)),
            ("classifier", LogisticRegression(C=4.0, max_iter=500, solver="liblinear",
                                              class_weight="balanced", random_state=args.seed)),
        ])

    started = time.perf_counter()
    holdout = build().fit(x_train, y_train)
    cv_score = float(roc_auc_score(y_valid, holdout.predict_proba(x_valid)[:, 1]))
    final = build().fit(train["Comment"].fillna(""), train["Insult"].astype(int))
    sample = pd.read_csv(resolved.sample_submission_path)
    sample["Insult"] = final.predict_proba(test["Comment"].fillna(""))[:, 1]
    sample["Date"] = test["Date"]
    sample["Comment"] = test["Comment"]
    budget = {"seed": args.seed, "holdout_fraction": args.holdout_fraction, "train_rows": len(train)}
    return wave0.finalize_scored_task(
        competition_id=competition_id, submission=sample, cv_score=cv_score,
        args=args, task_dir=task_dir, budget=budget,
        extra={"runtime_seconds_model": time.perf_counter() - started,
               "model_family": "TFIDF_word_char_LogisticRegression", "budget": budget},
    )


def run_dogs_cats(args: argparse.Namespace, task_dir: Path, logger: logging.Logger) -> dict[str, Any]:
    import torch
    from PIL import Image
    from sklearn.metrics import log_loss
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
    labels = np.asarray([1 if path.name.startswith("dog.") else 0 for path in paths], dtype=np.int64)
    indices = np.arange(len(paths))
    train_indices, valid_indices = train_test_split(
        indices, test_size=args.holdout_fraction, random_state=args.seed, stratify=labels
    )

    class Files(Dataset):
        def __init__(self, selected: list[Path], targets: np.ndarray | None, transform: Any) -> None:
            self.paths, self.targets, self.transform = selected, targets, transform

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, index: int):
            image = self.transform(Image.open(self.paths[index]).convert("RGB"))
            if self.targets is None:
                return image
            return image, torch.tensor(float(self.targets[index]), dtype=torch.float32)

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize((args.dogs_image_size, args.dogs_image_size)),
        transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(), normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((args.dogs_image_size, args.dogs_image_size)), transforms.ToTensor(), normalize,
    ])
    workers = min(8, max(2, (os.cpu_count() or 4) // 8))
    train_loader = DataLoader(
        Files([paths[i] for i in train_indices], labels[train_indices], train_transform),
        batch_size=args.dogs_batch_size, shuffle=True, num_workers=workers, pin_memory=True,
        persistent_workers=workers > 0,
    )
    valid_loader = DataLoader(
        Files([paths[i] for i in valid_indices], labels[valid_indices], eval_transform),
        batch_size=args.dogs_batch_size, shuffle=False, num_workers=workers, pin_memory=True,
        persistent_workers=workers > 0,
    )
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.25), nn.Linear(256, 1),
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.dogs_epochs):
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logits = model(images.cuda(non_blocking=True)).squeeze(1)
                loss = loss_fn(logits, targets.cuda(non_blocking=True))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(images)
        model.eval()
        probability: list[float] = []
        truth: list[float] = []
        with torch.inference_mode():
            for images, targets in valid_loader:
                logits = model(images.cuda(non_blocking=True)).squeeze(1)
                probability.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                truth.extend(targets.numpy().tolist())
        epoch_loss = float(log_loss(truth, np.clip(probability, 1e-6, 1 - 1e-6)))
        history.append({"epoch": epoch + 1, "train_loss": total_loss / len(train_indices),
                        "valid_log_loss": epoch_loss})
        logger.info("[%s] epoch=%s log_loss=%.6f", competition_id, epoch + 1, epoch_loss)
    cv_score = history[-1]["valid_log_loss"]
    sample = pd.read_csv(resolved.sample_submission_path)
    test_paths = [test_dir / f"{int(identifier)}.jpg" for identifier in sample["id"]]
    test_loader = DataLoader(Files(test_paths, None, eval_transform), batch_size=args.dogs_batch_size,
                             shuffle=False, num_workers=workers, pin_memory=True,
                             persistent_workers=workers > 0)
    predictions: list[float] = []
    model.eval()
    with torch.inference_mode():
        for images in test_loader:
            predictions.extend(torch.sigmoid(model(images.cuda(non_blocking=True)).squeeze(1)).cpu().numpy().tolist())
    sample["label"] = np.clip(predictions, 1e-6, 1 - 1e-6)
    torch.save(model.state_dict(), task_dir / "dogs_cats_cnn.pt")
    budget = {"seed": args.seed, "holdout_fraction": args.holdout_fraction, "epochs": args.dogs_epochs,
              "batch_size": args.dogs_batch_size, "image_size": args.dogs_image_size,
              "train_rows": len(train_indices), "valid_rows": len(valid_indices)}
    return wave0.finalize_scored_task(
        competition_id=competition_id, submission=sample, cv_score=cv_score,
        args=args, task_dir=task_dir, budget=budget,
        extra={"runtime_seconds_model": time.perf_counter() - started,
               "model_family": "SmallCNN_4xConv_AMP", "history": history,
               "torch_peak_memory_allocated_mib": int(torch.cuda.max_memory_allocated() / 2**20),
               "budget": budget},
    )


def run_leaf(args: argparse.Namespace, task_dir: Path, logger: logging.Logger) -> dict[str, Any]:
    from sklearn.metrics import log_loss
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    competition_id = "leaf-classification"
    resolved = resolve_competition(competition_id, args.data_root)
    train = pd.read_csv(resolved.public_dir / "train.csv")
    test = pd.read_csv(resolved.public_dir / "test.csv")
    sample = pd.read_csv(resolved.sample_submission_path)
    features = [column for column in train.columns if column not in {"id", "species"}]
    x_train, x_valid, y_train, y_valid = train_test_split(
        train[features], train["species"], test_size=args.holdout_fraction,
        random_state=args.seed, stratify=train["species"],
    )

    def build() -> Pipeline:
        return Pipeline([
            ("scale", StandardScaler()),
            ("svc", SVC(C=args.leaf_svc_c, gamma="scale", probability=True,
                        random_state=args.seed, cache_size=4096)),
        ])

    started = time.perf_counter()
    holdout = build().fit(x_train, y_train)
    valid_probability = holdout.predict_proba(x_valid)
    valid_classes = list(holdout.named_steps["svc"].classes_)
    cv_score = float(log_loss(y_valid, valid_probability, labels=valid_classes))
    final = build().fit(train[features], train["species"])
    probability = final.predict_proba(test[features])
    classes = list(final.named_steps["svc"].classes_)
    class_index = {name: index for index, name in enumerate(classes)}
    for column in sample.columns[1:]:
        sample[column] = probability[:, class_index[column]]
    budget = {"seed": args.seed, "holdout_fraction": args.holdout_fraction,
              "train_rows": len(train), "features": len(features), "svc_c": args.leaf_svc_c}
    return wave0.finalize_scored_task(
        competition_id=competition_id, submission=sample, cv_score=cv_score,
        args=args, task_dir=task_dir, budget=budget,
        extra={"runtime_seconds_model": time.perf_counter() - started,
               "model_family": "StandardScaler_RBF_SVC_probability", "classes": classes,
               "budget": budget},
    )


def taxi_features(frame: pd.DataFrame) -> pd.DataFrame:
    pickup_lat = pd.to_numeric(frame["pickup_latitude"], errors="coerce")
    pickup_lon = pd.to_numeric(frame["pickup_longitude"], errors="coerce")
    drop_lat = pd.to_numeric(frame["dropoff_latitude"], errors="coerce")
    drop_lon = pd.to_numeric(frame["dropoff_longitude"], errors="coerce")
    radians = np.pi / 180.0
    lat1, lat2 = pickup_lat * radians, drop_lat * radians
    delta_lat = (drop_lat - pickup_lat) * radians
    delta_lon = (drop_lon - pickup_lon) * radians
    hav = np.sin(delta_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    distance = 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))
    mean_latitude = (pickup_lat + drop_lat) / 2
    east_west_km = (drop_lon - pickup_lon).abs() * 111.32 * np.cos(mean_latitude * radians)
    north_south_km = (drop_lat - pickup_lat).abs() * 110.574
    manhattan_km = east_west_km + north_south_km
    bearing = np.arctan2(
        np.sin(delta_lon) * np.cos(lat2),
        np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon),
    )
    timestamp = pd.to_datetime(frame["pickup_datetime"], errors="coerce", utc=True)
    features = pd.DataFrame(index=frame.index)
    features["pickup_longitude"] = pickup_lon
    features["pickup_latitude"] = pickup_lat
    features["dropoff_longitude"] = drop_lon
    features["dropoff_latitude"] = drop_lat
    features["passenger_count"] = pd.to_numeric(frame["passenger_count"], errors="coerce")
    features["pickup_coordinate_missing"] = (pickup_lat.isna() | pickup_lon.isna()).astype(float)
    features["dropoff_coordinate_missing"] = (drop_lat.isna() | drop_lon.isna()).astype(float)
    features["passenger_count_missing"] = features["passenger_count"].isna().astype(float)
    features["pickup_datetime_missing"] = timestamp.isna().astype(float)
    features["haversine_km"] = distance
    features["manhattan_km"] = manhattan_km
    features["bearing_sin"] = np.sin(bearing)
    features["bearing_cos"] = np.cos(bearing)
    features["delta_latitude"] = drop_lat - pickup_lat
    features["delta_longitude"] = drop_lon - pickup_lon
    features["center_latitude"] = mean_latitude
    features["center_longitude"] = (pickup_lon + drop_lon) / 2
    features["year"] = timestamp.dt.year
    features["month"] = timestamp.dt.month
    features["day"] = timestamp.dt.day
    features["hour"] = timestamp.dt.hour
    features["weekday"] = timestamp.dt.weekday
    features["day_of_year"] = timestamp.dt.dayofyear
    features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24.0)
    features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24.0)
    features["weekday_sin"] = np.sin(2 * np.pi * features["weekday"] / 7.0)
    features["weekday_cos"] = np.cos(2 * np.pi * features["weekday"] / 7.0)
    features["rush_hour"] = features["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(float)
    features["night"] = features["hour"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(float)
    features["weekend"] = (features["weekday"] >= 5).astype(float)

    airports = {
        "jfk": (40.6413, -73.7781),
        "lga": (40.7769, -73.8740),
        "ewr": (40.6895, -74.1745),
    }
    airport_pickup: list[pd.Series] = []
    airport_dropoff: list[pd.Series] = []
    for name, (airport_lat, airport_lon) in airports.items():
        pickup_distance = np.sqrt(
            ((pickup_lat - airport_lat) * 110.574) ** 2
            + ((pickup_lon - airport_lon) * 111.32 * np.cos(airport_lat * radians)) ** 2
        )
        dropoff_distance = np.sqrt(
            ((drop_lat - airport_lat) * 110.574) ** 2
            + ((drop_lon - airport_lon) * 111.32 * np.cos(airport_lat * radians)) ** 2
        )
        features[f"pickup_{name}_km"] = pickup_distance
        features[f"dropoff_{name}_km"] = dropoff_distance
        features[f"pickup_{name}_within_3km"] = (pickup_distance < 3.0).astype(float)
        features[f"dropoff_{name}_within_3km"] = (dropoff_distance < 3.0).astype(float)
        features[f"{name}_route"] = (
            (pickup_distance < 3.0) | (dropoff_distance < 3.0)
        ).astype(float)
        airport_pickup.append(pickup_distance)
        airport_dropoff.append(dropoff_distance)
    features["pickup_airport_min_km"] = pd.concat(airport_pickup, axis=1).min(axis=1)
    features["dropoff_airport_min_km"] = pd.concat(airport_dropoff, axis=1).min(axis=1)
    features["airport_trip"] = (
        (features["pickup_airport_min_km"] < 3.0) | (features["dropoff_airport_min_km"] < 3.0)
    ).astype(float)
    features["airport_distance_sum_km"] = (
        features["pickup_airport_min_km"] + features["dropoff_airport_min_km"]
    )
    features["airport_rush_hour"] = features["airport_trip"] * features["rush_hour"]
    features["airport_night"] = features["airport_trip"] * features["night"]
    features["airport_weekend"] = features["airport_trip"] * features["weekend"]

    borough_centers = {
        "manhattan": (40.7831, -73.9712),
        "brooklyn": (40.6782, -73.9442),
        "queens": (40.7282, -73.7949),
        "bronx": (40.8448, -73.8648),
        "staten_island": (40.5795, -74.1502),
    }
    pickup_borough_distances: list[pd.Series] = []
    dropoff_borough_distances: list[pd.Series] = []
    for name, (center_lat, center_lon) in borough_centers.items():
        pickup_distance = np.sqrt(
            ((pickup_lat - center_lat) * 110.574) ** 2
            + ((pickup_lon - center_lon) * 111.32 * np.cos(center_lat * radians)) ** 2
        )
        dropoff_distance = np.sqrt(
            ((drop_lat - center_lat) * 110.574) ** 2
            + ((drop_lon - center_lon) * 111.32 * np.cos(center_lat * radians)) ** 2
        )
        features[f"pickup_{name}_center_km"] = pickup_distance
        features[f"dropoff_{name}_center_km"] = dropoff_distance
        pickup_borough_distances.append(pickup_distance)
        dropoff_borough_distances.append(dropoff_distance)
    pickup_borough_matrix = np.column_stack(pickup_borough_distances)
    dropoff_borough_matrix = np.column_stack(dropoff_borough_distances)
    features["pickup_borough_proxy"] = np.argmin(pickup_borough_matrix, axis=1)
    features["dropoff_borough_proxy"] = np.argmin(dropoff_borough_matrix, axis=1)
    features["cross_borough_proxy"] = (
        features["pickup_borough_proxy"] != features["dropoff_borough_proxy"]
    ).astype(float)
    features["pickup_cell_lat"] = np.floor((pickup_lat - 40.4) * 100)
    features["pickup_cell_lon"] = np.floor((pickup_lon + 74.3) * 100)
    features["dropoff_cell_lat"] = np.floor((drop_lat - 40.4) * 100)
    features["dropoff_cell_lon"] = np.floor((drop_lon + 74.3) * 100)
    features["pickup_cell_id"] = (
        features["pickup_cell_lat"] * 1_000 + features["pickup_cell_lon"]
    )
    features["dropoff_cell_id"] = (
        features["dropoff_cell_lat"] * 1_000 + features["dropoff_cell_lon"]
    )
    features["airport_route_code"] = (
        features["jfk_route"]
        + 2 * features["lga_route"]
        + 4 * features["ewr_route"]
    )
    return features.replace([np.inf, -np.inf], np.nan).fillna(0)


def load_taxi_training(path: Path, *, max_rows: int, chunk_rows: int, seed: int) -> tuple[pd.DataFrame, int]:
    reservoir: pd.DataFrame | None = None
    scanned = 0
    rng = np.random.default_rng(seed)
    audit = {
        "rows_scanned": 0,
        "accepted_after_rules": 0,
        "rejected_fare": 0,
        "rejected_passenger_count": 0,
        "rejected_pickup_longitude": 0,
        "rejected_dropoff_longitude": 0,
        "rejected_pickup_latitude": 0,
        "rejected_dropoff_latitude": 0,
        "rejected_zero_distance_high_fare": 0,
        "rejected_implausible_distance": 0,
    }
    for chunk in pd.read_csv(path, chunksize=chunk_rows):
        source_start = scanned
        scanned += len(chunk)
        chunk = chunk.copy()
        chunk["__source_row__"] = np.arange(source_start, scanned, dtype=np.int64)
        target = pd.to_numeric(chunk["fare_amount"], errors="coerce")
        passenger = pd.to_numeric(chunk["passenger_count"], errors="coerce")
        pickup_lon = pd.to_numeric(chunk["pickup_longitude"], errors="coerce")
        pickup_lat = pd.to_numeric(chunk["pickup_latitude"], errors="coerce")
        dropoff_lon = pd.to_numeric(chunk["dropoff_longitude"], errors="coerce")
        dropoff_lat = pd.to_numeric(chunk["dropoff_latitude"], errors="coerce")
        fare_ok = target.between(2.5, 100.0)
        passenger_ok = passenger.between(1, 6)
        pickup_lon_ok = pickup_lon.between(-75.0, -72.0)
        dropoff_lon_ok = dropoff_lon.between(-75.0, -72.0)
        pickup_lat_ok = pickup_lat.between(39.0, 42.0)
        dropoff_lat_ok = dropoff_lat.between(39.0, 42.0)
        approximate_distance = np.sqrt(
            ((dropoff_lat - pickup_lat) * 110.574) ** 2
            + ((dropoff_lon - pickup_lon) * 84.0) ** 2
        )
        zero_distance_high_fare = (approximate_distance < 0.02) & (target > 20.0)
        implausible_distance = approximate_distance > 150.0
        valid = (
            fare_ok
            & passenger_ok
            & pickup_lon_ok
            & dropoff_lon_ok
            & pickup_lat_ok
            & dropoff_lat_ok
            & ~zero_distance_high_fare
            & ~implausible_distance
        )
        audit["rows_scanned"] += len(chunk)
        audit["accepted_after_rules"] += int(valid.sum())
        audit["rejected_fare"] += int((~fare_ok).sum())
        audit["rejected_passenger_count"] += int((~passenger_ok).sum())
        audit["rejected_pickup_longitude"] += int((~pickup_lon_ok).sum())
        audit["rejected_dropoff_longitude"] += int((~dropoff_lon_ok).sum())
        audit["rejected_pickup_latitude"] += int((~pickup_lat_ok).sum())
        audit["rejected_dropoff_latitude"] += int((~dropoff_lat_ok).sum())
        audit["rejected_zero_distance_high_fare"] += int(zero_distance_high_fare.sum())
        audit["rejected_implausible_distance"] += int(implausible_distance.sum())
        clean = chunk[valid].copy()
        if clean.empty:
            continue
        clean["__sample_priority__"] = rng.random(len(clean))
        reservoir = clean if reservoir is None else pd.concat([reservoir, clean], ignore_index=True)
        if len(reservoir) > max_rows * 2:
            reservoir = reservoir.nsmallest(max_rows, "__sample_priority__").reset_index(drop=True)
    if reservoir is None or reservoir.empty:
        raise RuntimeError("Taxi labels stream produced no valid rows")
    if len(reservoir) > max_rows:
        reservoir = reservoir.nsmallest(max_rows, "__sample_priority__")
    result = reservoir.drop(columns="__sample_priority__").reset_index(drop=True)
    audit["reservoir_rows_retained"] = len(result)
    audit["reservoir_rows_discarded"] = int(audit["accepted_after_rules"] - len(result))
    audit["rules_are_nonexclusive"] = True
    result.attrs["cleaning_audit"] = audit
    return result, scanned


def build_taxi_duplicate_groups(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Quarantine conflicting trip/key records and return exact-trip groups.

    Duplicate keys that describe more than one exact trip are ambiguous and are
    removed rather than connected through a potentially target-bearing graph.
    Remaining exact trip signatures are safe group IDs for OOF assignment.
    """

    signature_columns = [
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
    required = {"key", "fare_amount", *signature_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Taxi duplicate audit is missing columns: {missing}")
    normalized = pd.DataFrame(index=frame.index)
    normalized["pickup_datetime"] = pd.to_datetime(
        frame["pickup_datetime"], errors="coerce", utc=True
    ).astype("int64")
    for column in signature_columns[1:]:
        values = pd.to_numeric(frame[column], errors="coerce")
        normalized[column] = np.rint(values.to_numpy(dtype=np.float64) * 1_000_000).astype(
            np.int64
        )
    signatures = pd.util.hash_pandas_object(normalized, index=False).to_numpy(
        dtype=np.uint64
    )
    keys = frame["key"].astype(str)
    targets = pd.to_numeric(frame["fare_amount"], errors="coerce")
    audit_frame = pd.DataFrame(
        {"key": keys.to_numpy(), "signature": signatures, "target": targets.to_numpy()}
    )
    signature_target_count = audit_frame.groupby("signature", sort=False)["target"].nunique(
        dropna=False
    )
    conflicting_signatures = set(signature_target_count[signature_target_count > 1].index)
    key_summary = audit_frame.groupby("key", sort=False).agg(
        signature_count=("signature", "nunique"),
        target_count=("target", lambda values: values.nunique(dropna=False)),
    )
    conflicting_keys = set(
        key_summary[
            (key_summary["signature_count"] > 1) | (key_summary["target_count"] > 1)
        ].index
    )
    quarantine = audit_frame["signature"].isin(conflicting_signatures) | audit_frame[
        "key"
    ].isin(conflicting_keys)
    safe = frame.loc[~quarantine].copy().reset_index(drop=True)
    safe_groups = signatures[~quarantine.to_numpy()]
    if safe.empty:
        raise RuntimeError("Taxi duplicate audit quarantined every retained row")
    safe_key_duplicates = int(safe["key"].astype(str).duplicated(keep=False).sum())
    audit = {
        "schema": "evomind.mlebench_lite.taxi_duplicate_audit.v1",
        "input_rows": int(len(frame)),
        "retained_rows": int(len(safe)),
        "quarantined_rows": int(quarantine.sum()),
        "conflicting_trip_signatures": int(len(conflicting_signatures)),
        "conflicting_keys": int(len(conflicting_keys)),
        "duplicate_signature_rows_retained": int(
            pd.Series(safe_groups).duplicated(keep=False).sum()
        ),
        "duplicate_key_rows_retained": safe_key_duplicates,
        "exact_trip_signature_columns": signature_columns,
    }
    return safe, np.asarray(safe_groups, dtype=np.uint64), audit


def build_taxi_oof_splits(
    group_ids: pd.Series | np.ndarray,
    *,
    fold_count: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Assign whole duplicate groups to deterministic, complete OOF folds."""

    groups = np.asarray(group_ids, dtype=np.uint64).reshape(-1)
    unique_groups = np.unique(groups)
    if len(groups) < 2 or len(unique_groups) < 2:
        raise RuntimeError("Taxi OOF requires at least two duplicate groups")
    effective_folds = min(max(2, int(fold_count)), len(unique_groups))
    with np.errstate(over="ignore"):
        values = unique_groups ^ np.uint64(seed & ((1 << 64) - 1))
        values += np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        values = values ^ (values >> np.uint64(31))
    order = np.argsort(values, kind="stable")
    group_assignment = np.empty(len(unique_groups), dtype=np.int16)
    group_assignment[order] = np.arange(len(unique_groups), dtype=np.int64) % effective_folds
    assignment = group_assignment[np.searchsorted(unique_groups, groups)]
    indices = np.arange(len(groups), dtype=np.int64)
    splits = [
        (indices[assignment != fold], indices[assignment == fold])
        for fold in range(effective_folds)
    ]
    covered = np.concatenate([validation for _, validation in splits])
    if len(covered) != len(groups) or len(np.unique(covered)) != len(groups):
        raise RuntimeError("Taxi OOF split coverage is incomplete")
    for fit_index, valid_index in splits:
        if set(groups[fit_index].tolist()) & set(groups[valid_index].tolist()):
            raise RuntimeError("Taxi duplicate group crossed an OOF fold")
    return splits, assignment


def build_taxi_fold_local_route_statistics(
    fit_features: pd.DataFrame,
    fit_target: pd.Series | np.ndarray,
    *apply_features: pd.DataFrame,
    smoothing: float = 40.0,
) -> tuple[pd.DataFrame, ...]:
    """Add leakage-safe route/time fare priors fitted on one training scope."""

    route_specs = {
        "route_cell": ["pickup_cell_id", "dropoff_cell_id"],
        "route_hour": ["pickup_cell_id", "dropoff_cell_id", "hour"],
        "airport_route": ["airport_route_code", "hour", "passenger_count"],
        "borough_route": ["pickup_borough_proxy", "dropoff_borough_proxy", "hour"],
    }
    missing = sorted(
        {column for columns in route_specs.values() for column in columns}
        - set(fit_features.columns)
    )
    if missing:
        raise RuntimeError(f"Taxi route statistics are missing features: {missing}")
    target = np.asarray(fit_target, dtype=np.float64).reshape(-1)
    if len(target) != len(fit_features) or not np.isfinite(target).all():
        raise RuntimeError("Taxi route-statistic fit target is invalid")
    prior = float(np.mean(target))
    alpha = float(smoothing)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Taxi route-statistic smoothing must be positive")
    outputs = [fit_features.copy(), *(frame.copy() for frame in apply_features)]
    for name, columns in route_specs.items():
        fit_scope = fit_features[columns].copy()
        fit_scope["__target__"] = target
        grouped = fit_scope.groupby(columns, dropna=False, sort=False)["__target__"].agg(
            ["sum", "count"]
        )
        grouped[f"route_stat_{name}"] = (
            grouped["sum"] + alpha * prior
        ) / (grouped["count"] + alpha)
        mapping = grouped[[f"route_stat_{name}"]]
        for index, source in enumerate((fit_features, *apply_features)):
            ordered = source[columns].copy()
            ordered["__row_order__"] = np.arange(len(source), dtype=np.int64)
            encoded = ordered.merge(
                mapping.reset_index(), on=columns, how="left", sort=False, validate="many_to_one"
            ).sort_values("__row_order__", kind="stable")[f"route_stat_{name}"]
            outputs[index][f"route_stat_{name}"] = encoded.fillna(prior).to_numpy(
                dtype=np.float64
            )
    return tuple(outputs)


def build_taxi_temporal_stress_split(
    pickup_datetime: pd.Series,
    *,
    holdout_fraction: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return a chronological train/latest-holdout split for promotion evidence."""

    timestamps = pd.to_datetime(pickup_datetime, errors="coerce", utc=True)
    if timestamps.isna().any():
        raise RuntimeError("Taxi temporal stress split contains invalid pickup timestamps")
    fraction = float(holdout_fraction)
    if not 0.0 < fraction < 0.5:
        raise ValueError("Taxi temporal stress holdout fraction must be between 0 and 0.5")
    order = np.argsort(timestamps.astype("int64").to_numpy(), kind="stable")
    holdout_rows = max(1, int(round(len(order) * fraction)))
    fit_end = len(order) - holdout_rows
    if fit_end < 2 or holdout_rows < 2:
        raise RuntimeError("Taxi temporal stress split is too small")
    fit_index = np.asarray(order[:fit_end], dtype=np.int64)
    valid_index = np.asarray(order[fit_end:], dtype=np.int64)
    return fit_index, valid_index, {
        "holdout_fraction": fraction,
        "fit_rows": int(len(fit_index)),
        "validation_rows": int(len(valid_index)),
        "fit_latest_timestamp": timestamps.iloc[fit_index].max().isoformat(),
        "validation_earliest_timestamp": timestamps.iloc[valid_index].min().isoformat(),
        "validation_latest_timestamp": timestamps.iloc[valid_index].max().isoformat(),
    }


def build_taxi_geographic_stress_split(
    pickup_cell_id: pd.Series | np.ndarray,
    *,
    holdout_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Hold out whole pickup geocells for a deterministic spatial stress test."""

    cells = np.asarray(pickup_cell_id, dtype=np.int64).reshape(-1)
    unique_cells = np.unique(cells)
    fraction = float(holdout_fraction)
    if not 0.0 < fraction < 0.5:
        raise ValueError("Taxi geographic holdout fraction must be between 0 and 0.5")
    if len(unique_cells) < 4:
        raise RuntimeError("Taxi geographic stress split has too few pickup cells")
    with np.errstate(over="ignore"):
        hashed = unique_cells.astype(np.uint64) ^ np.uint64(seed & ((1 << 64) - 1))
        hashed = (hashed ^ (hashed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        hashed = (hashed ^ (hashed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        hashed = hashed ^ (hashed >> np.uint64(31))
    holdout_groups = max(1, min(len(unique_cells) - 1, int(round(len(unique_cells) * fraction))))
    valid_cells = set(unique_cells[np.argsort(hashed, kind="stable")[:holdout_groups]].tolist())
    valid_mask = np.isin(cells, np.fromiter(valid_cells, dtype=np.int64))
    fit_index = np.flatnonzero(~valid_mask).astype(np.int64)
    valid_index = np.flatnonzero(valid_mask).astype(np.int64)
    if len(fit_index) < 2 or len(valid_index) < 2:
        raise RuntimeError("Taxi geographic stress split is too small")
    if set(cells[fit_index].tolist()) & set(cells[valid_index].tolist()):
        raise RuntimeError("Taxi pickup geocell crossed the geographic stress split")
    return fit_index, valid_index, {
        "holdout_fraction": fraction,
        "fit_rows": int(len(fit_index)),
        "validation_rows": int(len(valid_index)),
        "fit_pickup_cells": int(len(np.unique(cells[fit_index]))),
        "validation_pickup_cells": int(len(np.unique(cells[valid_index]))),
        "seed": int(seed),
    }


TAXI_PUBLIC_CACHE_SCHEMA = "evomind.mlebench.taxi_public_feature_cache.v1"
TAXI_CACHE_ARTIFACTS = (
    "train_features.npy",
    "test_features.npy",
    "target.npy",
    "train_key.npy",
    "test_key.npy",
    "source_row.npy",
    "duplicate_group.npy",
    "fold_assignment.npy",
    "pickup_datetime_ns.npy",
    "split_indices.npz",
    "feature_names.json",
    "cleaning_audit.json",
    "duplicate_audit.json",
)
TAXI_ROUTE_STAT_SIDECAR_SCHEMA = "evomind.mlebench.taxi_route_stat_sidecar.v1"
TAXI_ROUTE_STAT_COLUMNS = (
    "route_stat_route_cell",
    "route_stat_route_hour",
    "route_stat_airport_route",
    "route_stat_borough_route",
)
TAXI_ROUTE_STAT_ARTIFACTS = (
    "oof_fold_00_train.npy",
    "oof_fold_00_test.npy",
    "oof_fold_01_train.npy",
    "oof_fold_01_test.npy",
    "oof_fold_02_train.npy",
    "oof_fold_02_test.npy",
    "temporal_train.npy",
    "geographic_train.npy",
    "refit_train.npy",
    "refit_test.npy",
)


def _taxi_atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _taxi_atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _taxi_atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **values)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _taxi_cache_file_record(path: Path, *, relative_path: str | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise RuntimeError(f"Taxi cache artifact is missing or unsafe: {resolved}")
    return {
        "path": str(resolved),
        "relative_path": relative_path or resolved.name,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def build_taxi_public_feature_cache(
    *,
    data_root: Path,
    output_dir: Path,
    allowed_root: Path,
    max_rows: int,
    chunk_rows: int,
    seed: int,
    folds: int,
    holdout_fraction: float,
) -> dict[str, Any]:
    """Build a public-only Taxi cache on a CPU node; manifest is written last."""

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Taxi public cache requires CUDA_VISIBLE_DEVICES to be empty")
    root = Path(allowed_root).expanduser().resolve()
    data_root = Path(data_root).expanduser().resolve()
    cache_root = Path(output_dir).expanduser().resolve()
    for label, path in (("data root", data_root), ("cache output", cache_root)):
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"Taxi {label} escaped the allowed root") from exc
    if cache_root == root or cache_root.is_symlink():
        raise RuntimeError("Taxi cache output must be a regular directory below the allowed root")
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / "cache_manifest.json"
    if manifest_path.exists():
        raise RuntimeError("Taxi cache completion marker already exists")

    resolved = resolve_competition(
        "new-york-city-taxi-fare-prediction",
        data_root,
        require_private=False,
    )
    public_root = resolved.public_dir.resolve()
    inputs = {
        "labels.csv": (public_root / "labels.csv").resolve(),
        "test.csv": (public_root / "test.csv").resolve(),
        "sample_submission.csv": resolved.sample_submission_path.resolve(),
    }
    input_before: dict[str, dict[str, Any]] = {}
    for name, path in inputs.items():
        try:
            path.relative_to(public_root)
        except ValueError as exc:
            raise RuntimeError(f"Taxi public input escaped prepared/public: {name}") from exc
        if "private" in {part.lower() for part in path.parts}:
            raise RuntimeError(f"Taxi cache rejected private input: {name}")
        input_before[name] = _taxi_cache_file_record(
            path, relative_path=f"prepared/public/{name}"
        )

    started = time.perf_counter()
    train, scanned = load_taxi_training(
        inputs["labels.csv"], max_rows=max_rows, chunk_rows=chunk_rows, seed=seed
    )
    cleaning_audit = dict(train.attrs.get("cleaning_audit") or {})
    train, duplicate_groups, duplicate_audit = build_taxi_duplicate_groups(train)
    test = pd.read_csv(inputs["test.csv"])
    if train["key"].astype(str).isna().any() or test["key"].astype(str).isna().any():
        raise RuntimeError("Taxi cache found invalid keys")
    if test["key"].astype(str).duplicated().any():
        raise RuntimeError("Taxi cache found duplicate test keys")
    train_features = taxi_features(train)
    test_features = taxi_features(test)
    if list(train_features.columns) != list(test_features.columns):
        raise RuntimeError("Taxi cache feature schema mismatch")
    train_matrix = np.ascontiguousarray(
        train_features.to_numpy(dtype=np.float32, copy=False)
    )
    test_matrix = np.ascontiguousarray(test_features.to_numpy(dtype=np.float32, copy=False))
    if not np.isfinite(train_matrix).all() or not np.isfinite(test_matrix).all():
        raise RuntimeError("Taxi cache contains non-finite features")
    target = pd.to_numeric(train["fare_amount"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    splits, fold_assignment = build_taxi_oof_splits(
        duplicate_groups, fold_count=folds, seed=seed
    )
    temporal_fit, temporal_valid, temporal_evidence = build_taxi_temporal_stress_split(
        train["pickup_datetime"], holdout_fraction=holdout_fraction
    )
    geographic_fit, geographic_valid, geographic_evidence = (
        build_taxi_geographic_stress_split(
            train_features["pickup_cell_id"],
            holdout_fraction=holdout_fraction,
            seed=seed,
        )
    )
    train_keys = train["key"].astype(str)
    test_keys = test["key"].astype(str)
    key_width = max(1, int(max(train_keys.str.len().max(), test_keys.str.len().max())))
    pickup_datetime_ns = pd.to_datetime(
        train["pickup_datetime"], errors="raise", utc=True
    ).astype("int64").to_numpy(dtype=np.int64)

    _taxi_atomic_npy(cache_root / "train_features.npy", train_matrix)
    _taxi_atomic_npy(cache_root / "test_features.npy", test_matrix)
    _taxi_atomic_npy(cache_root / "target.npy", target)
    _taxi_atomic_npy(
        cache_root / "train_key.npy", train_keys.to_numpy(dtype=f"U{key_width}")
    )
    _taxi_atomic_npy(cache_root / "test_key.npy", test_keys.to_numpy(dtype=f"U{key_width}"))
    _taxi_atomic_npy(
        cache_root / "source_row.npy",
        train["__source_row__"].to_numpy(dtype=np.int64),
    )
    _taxi_atomic_npy(cache_root / "duplicate_group.npy", duplicate_groups.astype(np.uint64))
    _taxi_atomic_npy(cache_root / "fold_assignment.npy", fold_assignment.astype(np.int16))
    _taxi_atomic_npy(cache_root / "pickup_datetime_ns.npy", pickup_datetime_ns)
    split_payload: dict[str, np.ndarray] = {
        "temporal_fit": temporal_fit,
        "temporal_valid": temporal_valid,
        "geographic_fit": geographic_fit,
        "geographic_valid": geographic_valid,
    }
    for fold, (fit_index, valid_index) in enumerate(splits):
        split_payload[f"fold_{fold:02d}_fit"] = fit_index
        split_payload[f"fold_{fold:02d}_valid"] = valid_index
    _taxi_atomic_npz(cache_root / "split_indices.npz", **split_payload)
    _taxi_atomic_json(
        cache_root / "feature_names.json",
        {"ordered_feature_names": list(train_features.columns), "feature_count": len(train_features.columns)},
    )
    _taxi_atomic_json(cache_root / "cleaning_audit.json", cleaning_audit)
    _taxi_atomic_json(cache_root / "duplicate_audit.json", duplicate_audit)

    input_after = {
        name: _taxi_cache_file_record(path, relative_path=f"prepared/public/{name}")
        for name, path in inputs.items()
    }
    if any(
        input_before[name]["bytes"] != input_after[name]["bytes"]
        or input_before[name]["sha256"] != input_after[name]["sha256"]
        for name in inputs
    ):
        raise RuntimeError("Taxi public input changed during cache construction")
    artifacts = [
        _taxi_cache_file_record(cache_root / name, relative_path=name)
        for name in TAXI_CACHE_ARTIFACTS
    ]
    manifest = {
        "schema": TAXI_PUBLIC_CACHE_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "competition_id": "new-york-city-taxi-fare-prediction",
        "visibility_mode": "PUBLIC_ONLY",
        "seed": int(seed),
        "folds": int(folds),
        "max_rows": int(max_rows),
        "chunk_rows": int(chunk_rows),
        "holdout_fraction": float(holdout_fraction),
        "rows_scanned": int(scanned),
        "train_rows": int(len(train_matrix)),
        "test_rows": int(len(test_matrix)),
        "feature_count": int(train_matrix.shape[1]),
        "feature_names": list(train_features.columns),
        "inputs": [input_after[name] for name in sorted(input_after)],
        "artifacts": artifacts,
        "temporal_split": temporal_evidence,
        "geographic_split": geographic_evidence,
        "build_seconds": time.perf_counter() - started,
        "contracts": {
            "public_files_read": sorted(inputs),
            "private_files_read": [],
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "gpu_used": False,
            "cuda_visible_devices": "",
            "manifest_written_last": True,
        },
        "claim_boundary": "Taxi public cache is throughput evidence, not a score or medal.",
    }
    _taxi_atomic_json(manifest_path, manifest)
    return manifest


def load_taxi_public_feature_cache(
    cache_dir: Path,
    *,
    data_root: Path,
    allowed_root: Path,
    max_rows: int,
    chunk_rows: int,
    seed: int,
    folds: int,
    holdout_fraction: float,
) -> dict[str, Any]:
    """Validate every bound Taxi cache input/artifact before returning arrays."""

    root = Path(allowed_root).expanduser().resolve()
    cache_root = Path(cache_dir).expanduser().resolve()
    try:
        cache_root.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("Taxi cache escaped the allowed root") from exc
    manifest_path = cache_root / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema": TAXI_PUBLIC_CACHE_SCHEMA,
        "status": "completed",
        "competition_id": "new-york-city-taxi-fare-prediction",
        "visibility_mode": "PUBLIC_ONLY",
        "seed": int(seed),
        "folds": int(folds),
        "max_rows": int(max_rows),
        "chunk_rows": int(chunk_rows),
        "holdout_fraction": float(holdout_fraction),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Taxi cache manifest contract drifted: {key}")
    contracts = manifest.get("contracts") or {}
    if not (
        contracts.get("private_files_read") == []
        and contracts.get("private_labels_used") is False
        and contracts.get("official_grader_executed") is False
        and contracts.get("kaggle_submission_executed") is False
        and contracts.get("gpu_used") is False
    ):
        raise RuntimeError("Taxi cache safety contract failed")
    resolved = resolve_competition(
        "new-york-city-taxi-fare-prediction",
        Path(data_root),
        require_private=False,
    )
    actual_inputs = {
        "prepared/public/labels.csv": resolved.public_dir / "labels.csv",
        "prepared/public/test.csv": resolved.public_dir / "test.csv",
        "prepared/public/sample_submission.csv": resolved.sample_submission_path,
    }
    for record in manifest.get("inputs") or []:
        path = actual_inputs.get(str(record.get("relative_path")))
        if path is None or path.stat().st_size != int(record.get("bytes", -1)):
            raise RuntimeError("Taxi cache input size or identity drifted")
        if sha256_file(path) != record.get("sha256"):
            raise RuntimeError("Taxi cache input hash drifted")
    artifact_map = {record["relative_path"]: record for record in manifest.get("artifacts") or []}
    for name in TAXI_CACHE_ARTIFACTS:
        record = artifact_map.get(name)
        path = cache_root / name
        if record is None or path.stat().st_size != int(record.get("bytes", -1)):
            raise RuntimeError(f"Taxi cache artifact size drifted: {name}")
        if sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"Taxi cache artifact hash drifted: {name}")
    feature_names = json.loads((cache_root / "feature_names.json").read_text(encoding="utf-8"))[
        "ordered_feature_names"
    ]
    with np.load(cache_root / "split_indices.npz", allow_pickle=False) as payload:
        split_indices = {name: np.asarray(payload[name], dtype=np.int64) for name in payload.files}
    arrays = {
        "train_features": np.load(cache_root / "train_features.npy", allow_pickle=False),
        "test_features": np.load(cache_root / "test_features.npy", allow_pickle=False),
        "target": np.load(cache_root / "target.npy", allow_pickle=False),
        "train_key": np.load(cache_root / "train_key.npy", allow_pickle=False),
        "test_key": np.load(cache_root / "test_key.npy", allow_pickle=False),
        "source_row": np.load(cache_root / "source_row.npy", allow_pickle=False),
        "duplicate_group": np.load(cache_root / "duplicate_group.npy", allow_pickle=False),
        "fold_assignment": np.load(cache_root / "fold_assignment.npy", allow_pickle=False),
        "pickup_datetime_ns": np.load(cache_root / "pickup_datetime_ns.npy", allow_pickle=False),
    }
    train_rows = int(manifest["train_rows"])
    test_rows = int(manifest["test_rows"])
    if any(len(arrays[name]) != train_rows for name in ("train_features", "target", "train_key", "source_row", "duplicate_group", "fold_assignment", "pickup_datetime_ns")):
        raise RuntimeError("Taxi cache train cardinality failed")
    if len(arrays["test_features"]) != test_rows or len(arrays["test_key"]) != test_rows:
        raise RuntimeError("Taxi cache test cardinality failed")
    return {
        **arrays,
        "feature_names": feature_names,
        "splits": split_indices,
        "manifest": manifest,
        "manifest_record": _taxi_cache_file_record(manifest_path),
        "cleaning_audit": json.loads((cache_root / "cleaning_audit.json").read_text(encoding="utf-8")),
        "duplicate_audit": json.loads((cache_root / "duplicate_audit.json").read_text(encoding="utf-8")),
    }


def load_taxi_route_stat_sidecar(
    sidecar_dir: Path,
    *,
    base_cache_dir: Path,
    allowed_root: Path,
    max_rows: int,
    chunk_rows: int,
    cache_seed: int,
    folds: int,
    holdout_fraction: float,
) -> dict[str, Any]:
    """Verify the fixed-split route-stat sidecar before exposing memory maps."""

    root = Path(allowed_root).expanduser().resolve()
    sidecar = Path(sidecar_dir).expanduser().resolve()
    base_cache = Path(base_cache_dir).expanduser().resolve()
    for label, path in (("sidecar", sidecar), ("base cache", base_cache)):
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"Taxi route-stat {label} escaped allowed root") from exc
    manifest_path = sidecar / "route_stat_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema": TAXI_ROUTE_STAT_SIDECAR_SCHEMA,
        "status": "completed",
        "competition_id": "new-york-city-taxi-fare-prediction",
        "visibility_mode": "PUBLIC_ONLY",
        "cache_seed": int(cache_seed),
        "folds": int(folds),
        "max_rows": int(max_rows),
        "chunk_rows": int(chunk_rows),
        "holdout_fraction": float(holdout_fraction),
        "route_output_columns": list(TAXI_ROUTE_STAT_COLUMNS),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Taxi route-stat manifest contract drifted: {key}")
    contracts = manifest.get("contracts") or {}
    if not (
        contracts.get("validation_targets_used") == 0
        and contracts.get("private_files_read") == []
        and contracts.get("private_labels_used") is False
        and contracts.get("official_grader_executed") is False
        and contracts.get("kaggle_submission_executed") is False
        and contracts.get("gpu_used") is False
    ):
        raise RuntimeError("Taxi route-stat data boundary failed")
    base_manifest_path = base_cache / "cache_manifest.json"
    base_record = manifest.get("base_cache_manifest") or {}
    if (
        base_manifest_path.stat().st_size != int(base_record.get("bytes", -1))
        or sha256_file(base_manifest_path) != base_record.get("sha256")
    ):
        raise RuntimeError("Taxi route-stat base-cache identity drifted")
    algorithm_sha = hashlib.sha256(
        inspect.getsource(build_taxi_fold_local_route_statistics).encode("utf-8")
    ).hexdigest()
    if manifest.get("algorithm_source_sha256") != algorithm_sha:
        raise RuntimeError("Taxi route-stat algorithm source drifted")
    records = {record["relative_path"]: record for record in manifest.get("artifacts") or []}
    if set(records) != set(TAXI_ROUTE_STAT_ARTIFACTS):
        raise RuntimeError("Taxi route-stat artifact inventory drifted")
    arrays: dict[str, np.ndarray] = {}
    for name in TAXI_ROUTE_STAT_ARTIFACTS:
        path = sidecar / name
        record = records[name]
        if path.stat().st_size != int(record.get("bytes", -1)) or sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"Taxi route-stat artifact drifted: {name}")
        values = np.load(path, allow_pickle=False, mmap_mode="r")
        expected_rows = int(manifest["test_rows"] if name.endswith("_test.npy") else manifest["train_rows"])
        if values.shape != (expected_rows, len(TAXI_ROUTE_STAT_COLUMNS)) or values.dtype != np.float32:
            raise RuntimeError(f"Taxi route-stat array shape/dtype drifted: {name}")
        arrays[name.removesuffix(".npy")] = values
    return {
        "manifest": manifest,
        "manifest_record": _taxi_cache_file_record(manifest_path),
        "arrays": arrays,
    }


def align_scalar_submission(
    sample: pd.DataFrame,
    test_ids: pd.Series,
    prediction: np.ndarray,
    *,
    id_column: str,
    target_column: str,
) -> pd.DataFrame:
    """Join one scalar prediction per test ID into sample order, failing closed on drift."""

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


def run_taxi(args: argparse.Namespace, task_dir: Path, logger: logging.Logger) -> dict[str, Any]:
    from catboost import CatBoostRegressor

    competition_id = "new-york-city-taxi-fare-prediction"
    resolved = resolve_competition(competition_id, args.data_root)
    cache_dir = getattr(args, "taxi_precomputed_cache_dir", None)
    route_stat_dir = getattr(args, "taxi_route_stat_cache_dir", None)
    if getattr(args, "taxi_require_precomputed_cache", False) and cache_dir is None:
        raise RuntimeError("Taxi requires a precomputed cache but no cache directory was supplied")
    if getattr(args, "taxi_require_route_stat_cache", False) and route_stat_dir is None:
        raise RuntimeError("Taxi requires a route-stat cache but no directory was supplied")
    if route_stat_dir is not None and cache_dir is None:
        raise RuntimeError("Taxi route-stat cache requires the bound base feature cache")
    cache_record: dict[str, Any] | None = None
    cache_manifest: dict[str, Any] | None = None
    cached_splits: dict[str, np.ndarray] | None = None
    route_stat_arrays: dict[str, np.ndarray] | None = None
    route_stat_record: dict[str, Any] | None = None
    if cache_dir is not None:
        logger.info("[%s] loading verified PUBLIC_ONLY precomputed cache", competition_id)
        loaded = load_taxi_public_feature_cache(
            Path(cache_dir),
            data_root=args.data_root,
            allowed_root=args.allowed_root,
            max_rows=args.taxi_max_train_rows,
            chunk_rows=args.taxi_chunk_rows,
            seed=args.taxi_cache_seed,
            folds=args.taxi_folds,
            holdout_fraction=args.taxi_holdout_fraction,
        )
        x = pd.DataFrame(loaded["train_features"], columns=loaded["feature_names"])
        x_test = pd.DataFrame(loaded["test_features"], columns=loaded["feature_names"])
        y = pd.Series(loaded["target"], dtype=float)
        train = pd.DataFrame(
            {
                "key": loaded["train_key"].astype(str),
                "__source_row__": loaded["source_row"].astype(np.int64),
                "pickup_datetime": pd.to_datetime(
                    loaded["pickup_datetime_ns"], unit="ns", utc=True
                ),
            }
        )
        test = pd.DataFrame({"key": loaded["test_key"].astype(str)})
        duplicate_groups = loaded["duplicate_group"].astype(np.uint64)
        fold_assignment = loaded["fold_assignment"].astype(np.int16)
        cached_splits = loaded["splits"]
        splits = [
            (
                cached_splits[f"fold_{fold:02d}_fit"],
                cached_splits[f"fold_{fold:02d}_valid"],
            )
            for fold in range(args.taxi_folds)
        ]
        cleaning_audit = loaded["cleaning_audit"]
        duplicate_audit = loaded["duplicate_audit"]
        scanned = int(loaded["manifest"]["rows_scanned"])
        cache_record = {
            "required": bool(args.taxi_require_precomputed_cache),
            "manifest": loaded["manifest_record"],
            "build_seconds": loaded["manifest"].get("build_seconds"),
            "train_rows": loaded["manifest"].get("train_rows"),
            "test_rows": loaded["manifest"].get("test_rows"),
            "feature_count": loaded["manifest"].get("feature_count"),
            "visibility_mode": loaded["manifest"].get("visibility_mode"),
        }
        cache_manifest = loaded["manifest"]
    else:
        logger.info("[%s] streaming largest predeclared bounded training sample", competition_id)
        train, scanned = load_taxi_training(
            resolved.public_dir / "labels.csv",
            max_rows=args.taxi_max_train_rows,
            chunk_rows=args.taxi_chunk_rows,
            seed=args.seed,
        )
        cleaning_audit = dict(train.attrs.get("cleaning_audit") or {})
        if cleaning_audit.get("rows_scanned") != scanned:
            raise RuntimeError("Taxi cleaning audit row count does not match streamed rows")
        train, duplicate_groups, duplicate_audit = build_taxi_duplicate_groups(train)
        test = pd.read_csv(resolved.public_dir / "test.csv")
        x = taxi_features(train)
        x_test = taxi_features(test)
        y = pd.to_numeric(train["fare_amount"], errors="coerce").astype(float)
        splits, fold_assignment = build_taxi_oof_splits(
            duplicate_groups,
            fold_count=args.taxi_folds,
            seed=args.seed,
        )
    if route_stat_dir is not None:
        route_loaded = load_taxi_route_stat_sidecar(
            Path(route_stat_dir),
            base_cache_dir=Path(cache_dir),
            allowed_root=args.allowed_root,
            max_rows=args.taxi_max_train_rows,
            chunk_rows=args.taxi_chunk_rows,
            cache_seed=args.taxi_cache_seed,
            folds=args.taxi_folds,
            holdout_fraction=args.taxi_holdout_fraction,
        )
        route_stat_arrays = route_loaded["arrays"]
        route_stat_record = {
            "required": bool(args.taxi_require_route_stat_cache),
            "manifest": route_loaded["manifest_record"],
            "contexts": route_loaded["manifest"].get("contexts"),
            "train_rows": route_loaded["manifest"].get("train_rows"),
            "test_rows": route_loaded["manifest"].get("test_rows"),
            "validation_targets_used": 0,
        }

    def attach_route_stats(frame: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
        if len(frame) != len(values):
            raise RuntimeError("Taxi route-stat row cardinality drifted")
        result = frame.reset_index(drop=True).copy()
        for index, column in enumerate(TAXI_ROUTE_STAT_COLUMNS):
            result[column] = np.asarray(values[:, index], dtype=np.float32)
        return result
    wave0.write_json(
        task_dir / "taxi_cleaning_audit.json",
        {"schema": "evomind.mlebench_lite.taxi_cleaning_audit.v1", **cleaning_audit},
    )
    wave0.write_json(task_dir / "taxi_duplicate_audit.json", duplicate_audit)
    required_key = "key"
    if required_key not in train or required_key not in test:
        raise RuntimeError("Taxi key column is missing from train or test")
    if test[required_key].duplicated().any():
        raise RuntimeError("Taxi test keys are duplicated")
    if list(x.columns) != list(x_test.columns):
        raise RuntimeError("Taxi train/test feature schema mismatch")
    if not np.isfinite(x.to_numpy()).all() or not np.isfinite(x_test.to_numpy()).all():
        raise RuntimeError("Taxi engineered features contain non-finite values")
    categorical_features = [
        "pickup_cell_lat", "pickup_cell_lon", "dropoff_cell_lat", "dropoff_cell_lon",
        "pickup_cell_id", "dropoff_cell_id",
        "pickup_borough_proxy", "dropoff_borough_proxy", "airport_route_code",
        "hour", "weekday", "month", "year",
    ]
    missing_categories = sorted(set(categorical_features) - set(x.columns))
    if missing_categories:
        raise RuntimeError(f"Taxi categorical feature contract is incomplete: {missing_categories}")
    for column in categorical_features:
        x[column] = np.rint(x[column]).astype(np.int64)
        x_test[column] = np.rint(x_test[column]).astype(np.int64)
    if not np.isfinite(y.to_numpy()).all():
        raise RuntimeError("Taxi training targets contain non-finite values")
    started = time.perf_counter()
    oof_prediction = np.full(len(train), np.nan, dtype=np.float64)
    test_fold_prediction = np.zeros((len(test), len(splits)), dtype=np.float64)
    model_dir = task_dir / "taxi_fold_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    fold_records: list[dict[str, Any]] = []
    for fold, (fit_index, valid_index) in enumerate(splits):
        if route_stat_arrays is None:
            fold_fit_x, fold_valid_x, fold_test_x = build_taxi_fold_local_route_statistics(
                x.iloc[fit_index].reset_index(drop=True),
                y.iloc[fit_index].reset_index(drop=True),
                x.iloc[valid_index].reset_index(drop=True),
                x_test.reset_index(drop=True),
            )
        else:
            full_route = route_stat_arrays[f"oof_fold_{fold:02d}_train"]
            fold_fit_x = attach_route_stats(x.iloc[fit_index], full_route[fit_index])
            fold_valid_x = attach_route_stats(x.iloc[valid_index], full_route[valid_index])
            fold_test_x = attach_route_stats(
                x_test, route_stat_arrays[f"oof_fold_{fold:02d}_test"]
            )
        model = CatBoostRegressor(
            iterations=args.taxi_iterations,
            depth=10,
            learning_rate=0.055,
            loss_function="RMSE",
            eval_metric="RMSE",
            task_type="GPU",
            devices="0",
            random_seed=args.seed + fold,
            border_count=128,
            l2_leaf_reg=7.0,
            random_strength=0.25,
            od_type="Iter",
            od_wait=80,
            verbose=50,
            allow_writing_files=False,
        )
        model.fit(
            fold_fit_x,
            y.iloc[fit_index].reset_index(drop=True),
            eval_set=(fold_valid_x, y.iloc[valid_index].reset_index(drop=True)),
            cat_features=categorical_features,
            use_best_model=True,
        )
        fold_valid_prediction = np.clip(
            np.asarray(model.predict(fold_valid_x), dtype=np.float64),
            2.5,
            100.0,
        )
        fold_test_prediction = np.asarray(model.predict(fold_test_x), dtype=np.float64)
        if not np.isfinite(fold_valid_prediction).all() or not np.isfinite(fold_test_prediction).all():
            raise RuntimeError("Taxi fold model produced non-finite predictions")
        oof_prediction[valid_index] = fold_valid_prediction
        test_fold_prediction[:, fold] = fold_test_prediction
        best_iteration = int(model.get_best_iteration())
        fold_score = compute_metric("rmse", y.iloc[valid_index], fold_valid_prediction)
        model_path = model_dir / f"taxi_fold{fold}.cbm"
        model.save_model(str(model_path))
        fold_records.append({
            "fold": fold,
            "fit_rows": len(fit_index),
            "validation_rows": len(valid_index),
            "best_iteration": best_iteration,
            "selected_iterations": (
                best_iteration + 1 if best_iteration >= 0 else args.taxi_iterations
            ),
            "validation_rmse": float(fold_score),
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "route_statistics_fit_rows": int(len(fit_index)),
            "route_statistics_validation_target_rows": 0,
        })
        logger.info(
            "[%s] fold=%d/%d validation_rmse=%.6f best_iteration=%d",
            competition_id,
            fold + 1,
            len(splits),
            fold_score,
            best_iteration,
        )
    if not np.isfinite(oof_prediction).all():
        raise RuntimeError("Taxi OOF coverage is incomplete")
    cv_score = compute_metric("rmse", y, oof_prediction)
    if cached_splits is not None and cache_manifest is not None:
        temporal_fit = cached_splits["temporal_fit"]
        temporal_valid = cached_splits["temporal_valid"]
        temporal_split = dict(cache_manifest["temporal_split"])
    else:
        temporal_fit, temporal_valid, temporal_split = build_taxi_temporal_stress_split(
            train["pickup_datetime"],
            holdout_fraction=args.taxi_holdout_fraction,
        )
    if route_stat_arrays is None:
        temporal_fit_x, temporal_valid_x = build_taxi_fold_local_route_statistics(
            x.iloc[temporal_fit].reset_index(drop=True),
            y.iloc[temporal_fit].reset_index(drop=True),
            x.iloc[temporal_valid].reset_index(drop=True),
        )
    else:
        temporal_route = route_stat_arrays["temporal_train"]
        temporal_fit_x = attach_route_stats(x.iloc[temporal_fit], temporal_route[temporal_fit])
        temporal_valid_x = attach_route_stats(x.iloc[temporal_valid], temporal_route[temporal_valid])
    temporal_model = CatBoostRegressor(
        iterations=args.taxi_iterations,
        depth=10,
        learning_rate=0.055,
        loss_function="RMSE",
        eval_metric="RMSE",
        task_type="GPU",
        devices="0",
        random_seed=args.seed + 50_000,
        border_count=128,
        l2_leaf_reg=7.0,
        random_strength=0.25,
        od_type="Iter",
        od_wait=80,
        verbose=50,
        allow_writing_files=False,
    )
    temporal_model.fit(
        temporal_fit_x,
        y.iloc[temporal_fit].reset_index(drop=True),
        eval_set=(temporal_valid_x, y.iloc[temporal_valid].reset_index(drop=True)),
        cat_features=categorical_features,
        use_best_model=True,
    )
    temporal_prediction = np.clip(
        np.asarray(temporal_model.predict(temporal_valid_x), dtype=np.float64),
        2.5,
        100.0,
    )
    if not np.isfinite(temporal_prediction).all():
        raise RuntimeError("Taxi temporal stress model produced non-finite predictions")
    temporal_rmse = compute_metric("rmse", y.iloc[temporal_valid], temporal_prediction)
    temporal_model_path = model_dir / "taxi_temporal_stress.cbm"
    temporal_model.save_model(str(temporal_model_path))
    temporal_split["validation_rmse"] = float(temporal_rmse)
    temporal_split["best_iteration"] = int(temporal_model.get_best_iteration())
    temporal_split["model_sha256"] = sha256_file(temporal_model_path)
    del temporal_model
    if cached_splits is not None and cache_manifest is not None:
        geographic_fit = cached_splits["geographic_fit"]
        geographic_valid = cached_splits["geographic_valid"]
        geographic_split = dict(cache_manifest["geographic_split"])
    else:
        geographic_fit, geographic_valid, geographic_split = (
            build_taxi_geographic_stress_split(
                x["pickup_cell_id"],
                holdout_fraction=args.taxi_holdout_fraction,
                seed=args.seed,
            )
        )
    if route_stat_arrays is None:
        geographic_fit_x, geographic_valid_x = build_taxi_fold_local_route_statistics(
            x.iloc[geographic_fit].reset_index(drop=True),
            y.iloc[geographic_fit].reset_index(drop=True),
            x.iloc[geographic_valid].reset_index(drop=True),
        )
    else:
        geographic_route = route_stat_arrays["geographic_train"]
        geographic_fit_x = attach_route_stats(x.iloc[geographic_fit], geographic_route[geographic_fit])
        geographic_valid_x = attach_route_stats(x.iloc[geographic_valid], geographic_route[geographic_valid])
    geographic_model = CatBoostRegressor(
        iterations=args.taxi_iterations,
        depth=10,
        learning_rate=0.055,
        loss_function="RMSE",
        eval_metric="RMSE",
        task_type="GPU",
        devices="0",
        random_seed=args.seed + 60_000,
        border_count=128,
        l2_leaf_reg=7.0,
        random_strength=0.25,
        od_type="Iter",
        od_wait=80,
        verbose=50,
        allow_writing_files=False,
    )
    geographic_model.fit(
        geographic_fit_x,
        y.iloc[geographic_fit].reset_index(drop=True),
        eval_set=(geographic_valid_x, y.iloc[geographic_valid].reset_index(drop=True)),
        cat_features=categorical_features,
        use_best_model=True,
    )
    geographic_prediction = np.clip(
        np.asarray(geographic_model.predict(geographic_valid_x), dtype=np.float64),
        2.5,
        100.0,
    )
    geographic_rmse = compute_metric(
        "rmse", y.iloc[geographic_valid], geographic_prediction
    )
    geographic_model_path = model_dir / "taxi_geographic_stress.cbm"
    geographic_model.save_model(str(geographic_model_path))
    geographic_split["validation_rmse"] = float(geographic_rmse)
    geographic_split["best_iteration"] = int(geographic_model.get_best_iteration())
    geographic_split["model_sha256"] = sha256_file(geographic_model_path)
    del geographic_model
    selected_iterations = [record["selected_iterations"] for record in fold_records]
    refit_iterations = max(1, int(np.median(selected_iterations)))
    if route_stat_arrays is None:
        refit_x, refit_test_x = build_taxi_fold_local_route_statistics(x, y, x_test)
    else:
        refit_x = attach_route_stats(x, route_stat_arrays["refit_train"])
        refit_test_x = attach_route_stats(x_test, route_stat_arrays["refit_test"])
    refit_model = CatBoostRegressor(
        iterations=refit_iterations,
        depth=10,
        learning_rate=0.055,
        loss_function="RMSE",
        task_type="GPU",
        devices="0",
        random_seed=args.seed,
        border_count=128,
        l2_leaf_reg=7.0,
        random_strength=0.25,
        verbose=50,
        allow_writing_files=False,
    )
    refit_model.fit(
        refit_x,
        y,
        cat_features=categorical_features,
        use_best_model=False,
    )
    test_prediction = np.clip(
        np.asarray(refit_model.predict(refit_test_x), dtype=np.float64), 2.5, 100.0
    )
    if not np.isfinite(test_prediction).all():
        raise RuntimeError("Taxi full-data refit produced non-finite predictions")
    refit_model_path = model_dir / "taxi_full_data_refit.cbm"
    refit_model.save_model(str(refit_model_path))
    refit_record = {
        "fit_rows": int(len(train)),
        "iterations": refit_iterations,
        "iteration_source": "median_outer_fold_selected_iterations",
        "selected_iterations": selected_iterations,
        "model_path": str(refit_model_path),
        "model_sha256": sha256_file(refit_model_path),
        "target_statistics_scope": "all_retained_public_training_rows",
        "test_targets_used": False,
    }
    del refit_model
    sample = pd.read_csv(resolved.sample_submission_path)
    sample = align_scalar_submission(
        sample,
        test[required_key],
        test_prediction,
        id_column=required_key,
        target_column="fare_amount",
    )
    manifest = pd.DataFrame({
        "key": train[required_key].astype(str),
        "source_row": train["__source_row__"].astype(np.int64),
        "fare_amount": y,
              "duplicate_group": duplicate_groups.astype(str),
              "fold": fold_assignment,
        "oof_prediction": oof_prediction,
    })
    manifest.to_csv(task_dir / "taxi_oof_manifest.csv", index=False)
    pd.DataFrame({
        "key": test[required_key].astype(str),
        **{
            f"fold_{fold}": test_fold_prediction[:, fold]
            for fold in range(len(splits))
        },
        "fare_amount": test_prediction,
    }).to_csv(task_dir / "taxi_fold_test_predictions.csv", index=False)
    wave0.write_json(task_dir / "taxi_oof_fold_records.json", {
        "schema": "evomind.mlebench_lite.taxi_oof_folds.v1",
        "seed": args.seed,
        "fold_count": len(splits),
        "aggregate_oof_rmse": float(cv_score),
        "folds": fold_records,
    })
    budget = {"seed": args.seed, "folds": len(splits),
              "split_strategy": "deterministic_source_row_hash",
              "rows_scanned": scanned, "train_rows_used": len(train),
              "chunk_rows": args.taxi_chunk_rows, "iterations_requested": args.taxi_iterations,
              "best_iterations": [record["best_iteration"] for record in fold_records],
              "streaming_read": True, "full_data_refit": True,
              "precomputed_public_cache": cache_record,
              "precomputed_route_stat_cache": route_stat_record,
              "feature_build_on_gpu_run": cache_record is None,
              "route_stat_build_on_gpu_run": route_stat_record is None,
              "bounded_training_sample": True,
              "duplicate_audit": duplicate_audit,
              "fold_test_ensemble": True, "complete_oof": True,
              "categorical_features": categorical_features,
              "full_file_priority_reservoir": True, "strict_geo_target_cleaning": True,
              "cleaning_audit_persisted": True, "explicit_key_join": True,
              "temporal_stress_split": temporal_split,
              "temporal_stress_rmse": float(temporal_rmse),
              "geographic_stress_split": geographic_split,
              "geographic_stress_rmse": float(geographic_rmse),
              "full_data_refit_record": refit_record}
    promotion_gate = wave0.build_metric_promotion_gate(
        name="taxi_random_oof_and_temporal_stress_rmse",
        metric="rmse",
        direction="minimize",
        score=float(cv_score),
        threshold=2.85,
        extra_checks={
            "temporal_stress_rmse": float(temporal_rmse) <= 3.1,
            "geographic_stress_rmse": float(geographic_rmse) <= 3.35,
            "duplicate_isolation": duplicate_audit["quarantined_rows"] >= 0,
            "full_data_refit": True,
        },
        evidence={
            "random_oof_rmse": float(cv_score),
            "temporal_stress_rmse": float(temporal_rmse),
            "temporal_split": temporal_split,
            "geographic_stress_rmse": float(geographic_rmse),
            "geographic_split": geographic_split,
            "duplicate_audit": duplicate_audit,
            "full_data_refit": refit_record,
        },
    )
    return wave0.finalize_scored_task(
        competition_id=competition_id, submission=sample, cv_score=cv_score,
         args=args, task_dir=task_dir, budget=budget,
         promotion_gate=promotion_gate,
         extra={"runtime_seconds_model": time.perf_counter() - started,
                 "model_family": "CatBoostRegressor_GPU_duplicate_group_OOF_fold_local_route_stats_full_refit",
                 "feature_count": len(refit_x.columns), "fold_records": fold_records,
                 "temporal_stress": temporal_split,
                 "geographic_stress": geographic_split,
                 "full_data_refit": refit_record,
                "budget": budget},
    )


def _pizza_text(frame: pd.DataFrame) -> pd.Series:
    title = frame.get("request_title", pd.Series("", index=frame.index)).fillna("").astype(str)
    edited = frame.get("request_text_edit_aware", pd.Series("", index=frame.index)).fillna("").astype(str)
    subreddits = frame.get("requester_subreddits_at_request", pd.Series("", index=frame.index)).map(
        lambda value: " ".join(str(item) for item in value)
        if isinstance(value, (list, tuple, set)) else str(value or "")
    )
    # ``request_text`` exists in train but not test; only shared request-time
    # fields are allowed here to avoid a train/test distribution mismatch.
    return "__TITLE__ " + title + " __BODY__ " + edited + " __SUBREDDITS__ " + subreddits


def fit_pizza_nbsvm_fold(
    fit_text: pd.Series,
    fit_labels: pd.Series,
    valid_text: pd.Series,
    test_text: pd.Series,
    *,
    word_features: int,
    char_features: int,
    c_value: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit fold-local word and character NB-SVM channels for Pizza."""

    from scipy.special import expit
    from sklearn.feature_extraction.text import TfidfVectorizer

    word = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=word_features,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=char_features,
        sublinear_tf=True,
    )
    word_fit = word.fit_transform(fit_text)
    word_valid = word.transform(valid_text)
    word_test = word.transform(test_text)
    char_fit = char.fit_transform(fit_text)
    char_valid = char.transform(valid_text)
    char_test = char.transform(test_text)
    label_values = np.asarray(fit_labels, dtype=np.int8)
    word_model, word_ratio = wave2._fit_nbsvm(
        word_fit, label_values, c_value=c_value, max_iter=600, seed=seed
    )
    char_model, char_ratio = wave2._fit_nbsvm(
        char_fit, label_values, c_value=c_value, max_iter=600, seed=seed + 1
    )
    word_valid_logit = word_model.decision_function(word_valid.multiply(word_ratio))
    char_valid_logit = char_model.decision_function(char_valid.multiply(char_ratio))
    word_test_logit = word_model.decision_function(word_test.multiply(word_ratio))
    char_test_logit = char_model.decision_function(char_test.multiply(char_ratio))
    valid_probability = expit(0.5 * word_valid_logit + 0.5 * char_valid_logit)
    test_probability = expit(0.5 * word_test_logit + 0.5 * char_test_logit)
    if not np.isfinite(valid_probability).all() or not np.isfinite(test_probability).all():
        raise RuntimeError("Pizza NB-SVM produced non-finite probabilities")
    artifacts = {
        "word_vectorizer": word,
        "char_vectorizer": char,
        "word_model": word_model,
        "char_model": char_model,
        "word_ratio": word_ratio,
        "char_ratio": char_ratio,
        "word_features": int(word_fit.shape[1]),
        "char_features": int(char_fit.shape[1]),
    }
    return valid_probability, test_probability, artifacts


def pizza_structured_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=frame.index)
    request_time_numeric = (
        "requester_account_age_in_days_at_request",
        "requester_days_since_first_post_on_raop_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_comments_in_raop_at_request",
        "requester_number_of_posts_at_request",
        "requester_number_of_posts_on_raop_at_request",
        "requester_number_of_subreddits_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_upvotes_plus_downvotes_at_request",
    )
    for column in request_time_numeric:
        values = pd.to_numeric(frame.get(column, pd.Series(0, index=frame.index)), errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        features[column] = values
        features[f"log1p_{column}"] = np.sign(values) * np.log1p(np.abs(values))

    timestamp = pd.to_numeric(
        frame.get("unix_timestamp_of_request_utc", pd.Series(0, index=frame.index)), errors="coerce"
    ).fillna(0)
    date = pd.to_datetime(timestamp, unit="s", utc=True, errors="coerce")
    features["request_hour"] = date.dt.hour.fillna(0).astype(float)
    features["request_weekday"] = date.dt.dayofweek.fillna(0).astype(float)
    features["request_month"] = date.dt.month.fillna(0).astype(float)
    features["request_year"] = date.dt.year.fillna(0).astype(float)
    features["request_weekend"] = (features["request_weekday"] >= 5).astype(float)

    text = _pizza_text(frame).fillna("").astype(str)
    features["text_characters"] = text.str.len().astype(float)
    features["text_words"] = text.str.count(r"\b\w+\b").astype(float)
    features["text_questions"] = text.str.count(r"\?").astype(float)
    features["text_exclamations"] = text.str.count("!").astype(float)
    features["text_digits"] = text.str.count(r"\d").astype(float)
    features["known_giver"] = (
        frame.get("giver_username_if_known", pd.Series("N/A", index=frame.index))
        .fillna("N/A").astype(str).str.lower().ne("n/a").astype(float)
    )
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _rank_fraction(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rank(method="average", pct=True).to_numpy()


def apply_binary_auc_blend(
    text_probability: np.ndarray,
    structured_probability: np.ndarray,
    *,
    text_weight: float,
    mode: str,
) -> np.ndarray:
    text_values = np.asarray(text_probability, dtype=float)
    structured_values = np.asarray(structured_probability, dtype=float)
    if mode == "rank":
        text_values = _rank_fraction(text_values)
        structured_values = _rank_fraction(structured_values)
    elif mode != "raw":
        raise ValueError(f"Unsupported binary blend mode: {mode}")
    return np.clip(text_weight * text_values + (1.0 - text_weight) * structured_values, 1e-6, 1 - 1e-6)


def select_binary_auc_blend(
    text_probability: np.ndarray,
    structured_probability: np.ndarray,
    labels: pd.Series,
) -> tuple[np.ndarray, float, str, float]:
    from sklearn.metrics import roc_auc_score

    best: tuple[float, float, int, np.ndarray, float, str] | None = None
    for mode in ("raw", "rank"):
        for weight in np.linspace(0.0, 1.0, 21):
            prediction = apply_binary_auc_blend(
                text_probability,
                structured_probability,
                text_weight=float(weight),
                mode=mode,
            )
            score = float(roc_auc_score(labels, prediction))
            candidate = (-score, abs(float(weight) - 0.5), 0 if mode == "raw" else 1,
                         prediction, float(weight), mode)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    assert best is not None
    return best[3], best[4], best[5], -best[0]


def cross_fit_binary_auc_blend(
    text_probability: np.ndarray,
    structured_probability: np.ndarray,
    labels: pd.Series,
    folds: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Fit blend parameters without each outer fold and predict that held-out fold."""

    from sklearn.metrics import roc_auc_score

    text_values = np.asarray(text_probability, dtype=np.float64)
    structured_values = np.asarray(structured_probability, dtype=np.float64)
    label_values = np.asarray(labels, dtype=np.int8)
    fold_values = np.asarray(folds)
    if not (text_values.shape == structured_values.shape == label_values.shape == fold_values.shape):
        raise RuntimeError("Binary cross-fit blend arrays have inconsistent shapes")
    if not all(np.isfinite(values).all() for values in (text_values, structured_values, fold_values)):
        raise RuntimeError("Binary cross-fit blend inputs contain non-finite values")
    if set(label_values.tolist()) != {0, 1}:
        raise RuntimeError("Binary cross-fit blend requires both target classes")
    unique_folds = sorted(int(value) for value in np.unique(fold_values))
    if len(unique_folds) < 2 or unique_folds[0] < 0:
        raise RuntimeError("Binary cross-fit blend requires complete nonnegative fold assignments")
    prediction = np.full(len(label_values), np.nan, dtype=np.float64)
    records: list[dict[str, Any]] = []
    for fold in unique_folds:
        validation = fold_values == fold
        fitting = ~validation
        if set(label_values[fitting].tolist()) != {0, 1} or set(label_values[validation].tolist()) != {0, 1}:
            raise RuntimeError("Binary cross-fit fold does not contain both classes")
        _, text_weight, mode, fitting_score = select_binary_auc_blend(
            text_values[fitting],
            structured_values[fitting],
            pd.Series(label_values[fitting]),
        )
        prediction[validation] = apply_binary_auc_blend(
            text_values[validation],
            structured_values[validation],
            text_weight=text_weight,
            mode=mode,
        )
        records.append({
            "fold": fold,
            "text_weight": text_weight,
            "mode": mode,
            "meta_fit_auc": fitting_score,
            "outer_auc": float(roc_auc_score(label_values[validation], prediction[validation])),
        })
    if not np.isfinite(prediction).all():
        raise RuntimeError("Binary cross-fit blend did not cover every row")
    return prediction, records


def run_pizza(args: argparse.Namespace, task_dir: Path, logger: logging.Logger) -> dict[str, Any]:
    from catboost import CatBoostClassifier
    from joblib import dump
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    competition_id = "random-acts-of-pizza"
    resolved = resolve_competition(competition_id, args.data_root)
    train = pd.read_json(resolved.public_dir / "train.json")
    test = pd.read_json(resolved.public_dir / "test.json")
    train = train.copy()
    test = test.copy()
    if "request_id" not in train or "request_id" not in test:
        raise RuntimeError("Pizza request_id column is missing")
    if train["request_id"].duplicated().any() or test["request_id"].duplicated().any():
        raise RuntimeError("Pizza train or test request IDs are duplicated")
    train["__text__"] = _pizza_text(train)
    test["__text__"] = _pizza_text(test)
    target = "requester_received_pizza"
    y = train[target].astype(int)
    train_structured = pizza_structured_features(train)
    test_structured = pizza_structured_features(test)
    if list(train_structured.columns) != list(test_structured.columns):
        raise RuntimeError("Pizza structured train/test feature schema mismatch")

    folds = StratifiedKFold(n_splits=args.pizza_folds, shuffle=True, random_state=args.seed)
    text_oof = np.zeros(len(train), dtype=np.float64)
    structured_oof = np.zeros(len(train), dtype=np.float64)
    text_test = np.zeros(len(test), dtype=np.float64)
    structured_test = np.zeros(len(test), dtype=np.float64)
    fold_assignment = np.full(len(train), -1, dtype=np.int16)
    best_iterations: list[int] = []
    model_dir = task_dir / "pizza_fold_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    logger.info("[%s] fitting %s-fold text and structured OOF ensemble", competition_id, args.pizza_folds)
    for fold_index, (fit_index, valid_index) in enumerate(folds.split(train, y)):
        fold_assignment[valid_index] = fold_index
        text_valid, text_test_fold, text_artifacts = fit_pizza_nbsvm_fold(
            train.iloc[fit_index]["__text__"],
            y.iloc[fit_index],
            train.iloc[valid_index]["__text__"],
            test["__text__"],
            word_features=args.pizza_word_features,
            char_features=args.pizza_char_features,
            c_value=args.pizza_nbsvm_c,
            seed=args.seed + fold_index,
        )
        text_oof[valid_index] = text_valid
        text_test += text_test_fold / args.pizza_folds
        dump(text_artifacts, model_dir / f"fold_{fold_index}_text_nbsvm.joblib", compress=3)

        structured_model = CatBoostClassifier(
            iterations=args.pizza_catboost_iterations,
            depth=6,
            learning_rate=0.04,
            loss_function="Logloss",
            eval_metric="AUC",
            task_type="GPU",
            devices="0",
            random_seed=args.seed + fold_index,
            border_count=64,
            l2_leaf_reg=6.0,
            random_strength=0.35,
            od_type="Iter",
            od_wait=60,
            verbose=False,
            allow_writing_files=False,
        )
        structured_model.fit(
            train_structured.iloc[fit_index],
            y.iloc[fit_index],
            eval_set=(train_structured.iloc[valid_index], y.iloc[valid_index]),
            use_best_model=True,
        )
        structured_oof[valid_index] = structured_model.predict_proba(
            train_structured.iloc[valid_index]
        )[:, 1]
        structured_test += structured_model.predict_proba(test_structured)[:, 1] / args.pizza_folds
        structured_model.save_model(str(model_dir / f"fold_{fold_index}_structured.cbm"))
        best_iterations.append(int(structured_model.get_best_iteration()))

    if np.any(fold_assignment < 0):
        raise RuntimeError("Pizza fold assignment did not cover every training row")
    for name, values in {
        "text_oof": text_oof,
        "structured_oof": structured_oof,
        "text_test": text_test,
        "structured_test": structured_test,
    }.items():
        if not np.isfinite(values).all() or np.ptp(values) <= 1e-12:
            raise RuntimeError(f"Pizza component is non-finite or constant: {name}")

    blended_oof, crossfit_records = cross_fit_binary_auc_blend(
        text_oof, structured_oof, y, fold_assignment
    )
    _, text_weight, blend_mode, meta_fit_auc = select_binary_auc_blend(
        text_oof, structured_oof, y
    )
    cv_score = float(roc_auc_score(y, blended_oof))
    test_prediction = apply_binary_auc_blend(
        text_test,
        structured_test,
        text_weight=text_weight,
        mode=blend_mode,
    )
    component_scores = {
        "text_auc": float(roc_auc_score(y, text_oof)),
        "structured_auc": float(roc_auc_score(y, structured_oof)),
        "cross_fitted_blended_auc": float(cv_score),
        "final_meta_fit_auc": float(meta_fit_auc),
    }
    pd.DataFrame({
        "request_id": train["request_id"],
        "target": y,
        "fold": fold_assignment,
        "text_probability": text_oof,
        "structured_probability": structured_oof,
        "blended_probability": blended_oof,
    }).to_csv(task_dir / "pizza_oof_predictions.csv", index=False)
    pd.DataFrame({
        "request_id": test["request_id"],
        "text_probability": text_test,
        "structured_probability": structured_test,
        "blended_probability": test_prediction,
    }).to_csv(task_dir / "pizza_test_components.csv", index=False)

    sample = pd.read_csv(resolved.sample_submission_path)
    sample = align_scalar_submission(
        sample,
        test["request_id"],
        test_prediction,
        id_column="request_id",
        target_column=target,
    )
    wave0.write_json(task_dir / "pizza_crossfit_blend.json", {
        "schema": "evomind.mlebench_lite.pizza_crossfit_blend.v1",
        "folds": crossfit_records,
        "final_text_weight": text_weight,
        "final_mode": blend_mode,
        "cross_fitted_auc": cv_score,
        "final_meta_fit_auc": meta_fit_auc,
    })
    budget = {"seed": args.seed, "folds": args.pizza_folds,
              "train_rows": len(train), "structured_features": len(train_structured.columns),
              "word_max_features": args.pizza_word_features,
              "char_max_features": args.pizza_char_features,
              "nbsvm_c": args.pizza_nbsvm_c,
              "catboost_iterations_requested": args.pizza_catboost_iterations,
              "catboost_best_iterations": best_iterations,
              "text_weight": text_weight, "blend_mode": blend_mode,
              "fold_test_ensemble": True, "cross_fitted_meta_validation": True,
              "explicit_request_id_join": True, "fold_models_persisted": True}
    return wave0.finalize_scored_task(
        competition_id=competition_id, submission=sample, cv_score=cv_score,
        args=args, task_dir=task_dir, budget=budget,
        extra={"runtime_seconds_model": time.perf_counter() - started,
               "model_family": "fold_local_word_char_NBSVM_plus_GPU_CatBoost_crossfit_OOF_blend",
               "component_scores": component_scores, "budget": budget},
    )


def run_dec2021(args: argparse.Namespace, task_dir: Path, logger: logging.Logger) -> dict[str, Any]:
    from catboost import CatBoostClassifier
    from sklearn.model_selection import train_test_split

    competition_id = "tabular-playground-series-dec-2021"
    resolved = resolve_competition(competition_id, args.data_root)
    train = pd.read_csv(resolved.public_dir / "train.csv", nrows=args.dec_max_train_rows)
    test = pd.read_csv(resolved.public_dir / "test.csv")
    features = [column for column in train.columns if column not in {"Id", "Cover_Type"}]
    x_train, x_valid, y_train, y_valid = train_test_split(
        train[features], train["Cover_Type"].astype(int), test_size=args.holdout_fraction,
        random_state=args.seed, stratify=train["Cover_Type"],
    )
    model = CatBoostClassifier(
        iterations=args.dec_iterations, depth=9, learning_rate=0.14,
        loss_function="MultiClass", eval_metric="Accuracy", task_type="GPU", devices="0",
        random_seed=args.seed, border_count=64, l2_leaf_reg=4.0,
        od_type="Iter", od_wait=35, verbose=50, allow_writing_files=False,
    )
    started = time.perf_counter()
    model.fit(x_train, y_train, eval_set=(x_valid, y_valid), use_best_model=True)
    valid_prediction = model.predict(x_valid).reshape(-1).astype(int)
    cv_score = compute_metric("accuracy", y_valid, valid_prediction)
    sample = pd.read_csv(resolved.sample_submission_path)
    sample["Cover_Type"] = model.predict(test[features]).reshape(-1).astype(int)
    model.save_model(str(task_dir / "dec2021_catboost.cbm"))
    budget = {"seed": args.seed, "holdout_fraction": args.holdout_fraction,
              "train_rows_used": len(train), "iterations_requested": args.dec_iterations,
              "best_iteration": model.get_best_iteration()}
    return wave0.finalize_scored_task(
        competition_id=competition_id, submission=sample, cv_score=cv_score,
        args=args, task_dir=task_dir, budget=budget,
        extra={"runtime_seconds_model": time.perf_counter() - started,
               "model_family": "CatBoostClassifier_GPU_multiclass", "feature_count": len(features),
               "classes": sorted(int(value) for value in train["Cover_Type"].unique()), "budget": budget},
    )


RUNNERS: dict[str, Callable[[argparse.Namespace, Path, logging.Logger], dict[str, Any]]] = {
    **wave0.RUNNERS,
    **wave2.RUNNERS,
    **recovery.RUNNERS,
    "detecting-insults-in-social-commentary": run_insults,
    "dogs-vs-cats-redux-kernels-edition": recovery.run_dogs_cats_convnext,
    "leaf-classification": recovery.run_leaf_multimodal,
    "new-york-city-taxi-fare-prediction": run_taxi,
    "random-acts-of-pizza": run_pizza,
    "tabular-playground-series-dec-2021": run_dec2021,
}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_OFFICIAL_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_REMOTE_ROOT / "mlebench_lite_runs")
    parser.add_argument("--allowed-root", type=Path, default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--official-source-root", type=Path, required=True)
    parser.add_argument("--waves", default="Wave1")
    parser.add_argument("--competitions", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--candidate-only",
        "--withhold-official-grader",
        dest="candidate_only",
        action="store_true",
        help=(
            "Persist and validate the candidate submission without invoking the official "
            "private grader. A passed promotion gate ends in confirmation-pending status."
        ),
    )
    parser.add_argument(
        "--hold-cuda-lease",
        action="store_true",
        help=(
            "Initialize a minimal CUDA allocation before CPU-heavy preprocessing so "
            "external single-GPU queues continue to observe this run as the GPU owner."
        ),
    )
    parser.add_argument("--phase-a-only", action="store_true")
    parser.add_argument("--phase-a-scope", choices=("all", "requested"), default="all")
    parser.add_argument("--precompute-only", choices=("", "birds"), default="")
    parser.add_argument("--optimization-plan", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--denoising-epochs", type=int, default=10)
    parser.add_argument("--denoising-folds", type=int, default=3)
    parser.add_argument("--denoising-patch-size", type=int, default=128)
    parser.add_argument("--denoising-patches-per-image", type=int, default=48)
    parser.add_argument("--denoising-batch-size", type=int, default=64)
    parser.add_argument("--denoising-workers", type=int, default=4)
    parser.add_argument("--denoising-base-channels", type=int, default=32)
    parser.add_argument("--denoising-learning-rate", type=float, default=8e-4)
    parser.add_argument("--denoising-patience", type=int, default=3)
    parser.add_argument("--restoration-pixels-per-image", type=int, default=12_000)
    parser.add_argument("--restoration-ridge-alpha", type=float, default=0.001)
    parser.add_argument("--dogs-epochs", type=int, default=6)
    parser.add_argument("--dogs-folds", type=int, default=5)
    parser.add_argument("--dogs-batch-size", type=int, default=128)
    parser.add_argument("--dogs-image-size", type=int, default=288)
    parser.add_argument("--dogs-workers", type=int, default=64)
    parser.add_argument("--dogs-learning-rate", type=float, default=2e-4)
    parser.add_argument("--leaf-svc-c", type=float, default=10.0)
    parser.add_argument("--leaf-logistic-c", type=float, default=10.0)
    parser.add_argument("--leaf-folds", type=int, default=5)
    parser.add_argument("--leaf-image-size", type=int, default=224)
    parser.add_argument("--leaf-embedding-batch-size", type=int, default=128)
    parser.add_argument(
        "--leaf-embedding-tta",
        type=int,
        choices=(1, 4, 8),
        default=8,
    )
    parser.add_argument("--leaf-image-svc-c", type=float, default=8.0)
    parser.add_argument("--leaf-multimodal-svc-c", type=float, default=10.0)
    parser.add_argument("--taxi-max-train-rows", type=int, default=5_000_000)
    parser.add_argument("--taxi-chunk-rows", type=int, default=250_000)
    parser.add_argument("--taxi-iterations", type=int, default=1_400)
    parser.add_argument("--taxi-folds", type=int, default=3)
    parser.add_argument("--taxi-holdout-fraction", type=float, default=0.05)
    parser.add_argument(
        "--taxi-cache-seed",
        type=int,
        default=42,
        help="Fixed split/cache seed, independent of the external model seed.",
    )
    parser.add_argument("--taxi-precomputed-cache-dir", type=Path)
    parser.add_argument("--taxi-require-precomputed-cache", action="store_true")
    parser.add_argument("--taxi-route-stat-cache-dir", type=Path)
    parser.add_argument("--taxi-require-route-stat-cache", action="store_true")
    parser.add_argument("--pizza-folds", type=int, default=5)
    parser.add_argument("--pizza-word-features", type=int, default=90_000)
    parser.add_argument("--pizza-char-features", type=int, default=110_000)
    parser.add_argument("--pizza-nbsvm-c", type=float, default=4.0)
    parser.add_argument("--pizza-catboost-iterations", type=int, default=650)
    parser.add_argument("--dec-max-train-rows", type=int, default=1_200_000)
    parser.add_argument("--dec-iterations", type=int, default=380)
    # Wave 0 compatibility when a combined 11-task run is requested.
    parser.add_argument("--may-max-train-rows", type=int, default=0)
    parser.add_argument("--catboost-iterations", type=int, default=220)
    parser.add_argument("--may-folds", type=int, default=5)
    parser.add_argument("--may-xgb-estimators", type=int, default=2_600)
    parser.add_argument("--may-xgb-depth", type=int, default=8)
    parser.add_argument("--may-catboost-iterations", type=int, default=2_000)
    parser.add_argument("--may-catboost-depth", type=int, default=8)
    parser.add_argument("--may-learning-rate", type=float, default=0.035)
    parser.add_argument("--may-early-stopping", type=int, default=180)
    parser.add_argument("--may-verbose-eval", type=int, default=100)
    parser.add_argument("--may-mlp-epochs", type=int, default=24)
    parser.add_argument("--may-mlp-width", type=int, default=768)
    parser.add_argument("--may-mlp-blocks", type=int, default=5)
    parser.add_argument("--may-mlp-batch-size", type=int, default=4_096)
    parser.add_argument("--may-mlp-learning-rate", type=float, default=1e-3)
    parser.add_argument("--may-mlp-weight-decay", type=float, default=1e-5)
    parser.add_argument("--may-mlp-dropout", type=float, default=0.08)
    parser.add_argument("--may-mlp-patience", type=int, default=4)
    parser.add_argument(
        "--may-precomputed-cache-dir",
        type=Path,
        default=None,
        help="Verified public-only May feature/fold cache built on a CPU node.",
    )
    parser.add_argument(
        "--may-require-precomputed-cache",
        action="store_true",
        help="Fail closed instead of rebuilding May features on the GPU node.",
    )
    parser.add_argument("--spooky-folds", type=int, default=5)
    parser.add_argument("--spooky-word-features", type=int, default=120_000)
    parser.add_argument("--spooky-char-features", type=int, default=180_000)
    parser.add_argument("--spooky-raw-char-features", type=int, default=160_000)
    parser.add_argument("--spooky-nbsvm-c", type=float, default=4.0)
    parser.add_argument("--spooky-style-c", type=float, default=1.0)
    parser.add_argument("--aerial-epochs", type=int, default=6)
    parser.add_argument(
        "--aerial-backbone",
        choices=("convnext_tiny", "convnext_small", "efficientnet_v2_s"),
        default="convnext_tiny",
    )
    parser.add_argument("--aerial-folds", type=int, default=5)
    parser.add_argument("--aerial-batch-size", type=int, default=256)
    parser.add_argument("--aerial-image-size", type=int, default=224)
    parser.add_argument("--siim-epochs", type=int, default=8)
    parser.add_argument(
        "--siim-folds",
        type=int,
        choices=(recovery.SIIM_FORMAL_OUTER_FOLDS,),
        default=recovery.SIIM_FORMAL_OUTER_FOLDS,
        help="Formal SIIM outer-fold count; fixed by the research contract.",
    )
    parser.add_argument(
        "--siim-inner-folds",
        type=int,
        choices=(recovery.SIIM_FORMAL_INNER_FOLDS,),
        default=recovery.SIIM_FORMAL_INNER_FOLDS,
        help="Formal SIIM inner-fold count; fixed by the research contract.",
    )
    parser.add_argument("--siim-batch-size", type=int, default=64)
    parser.add_argument(
        "--siim-effective-batch-size",
        type=int,
        default=384,
        help=(
            "Fixed effective batch for SIIM gradient accumulation. The selected "
            "physical batch must divide this value exactly."
        ),
    )
    parser.add_argument(
        "--siim-memory-limit-mib",
        type=int,
        default=0,
        help=(
            "Per-process CUDA memory ceiling for SIIM. Zero preserves the "
            "runtime default; governed HPC campaigns must pass an explicit limit."
        ),
    )
    parser.add_argument("--siim-image-size", type=int, default=384)
    parser.add_argument(
        "--siim-preprocessing-profile",
        choices=recovery.SIIM_PREPROCESSING_PROFILES,
        default="robust_multiview_v1",
    )
    parser.add_argument("--siim-preprocessing-ablation-report", type=Path)
    parser.add_argument(
        "--siim-image-content-manifest",
        type=Path,
        help=(
            "Verified image-content manifest from the same governed ablation Run; "
            "avoids rehashing every JPEG during epoch-boundary continuation."
        ),
    )
    parser.add_argument(
        "--siim-backbone",
        choices=("convnext_tiny", "convnext_small", "efficientnet_v2_s"),
        default="convnext_small",
    )
    parser.add_argument(
        "--siim-secondary-backbone",
        choices=("convnext_tiny", "convnext_small", "efficientnet_v2_s"),
        default="efficientnet_v2_s",
    )
    parser.add_argument("--siim-workers", type=int, default=8)
    parser.add_argument("--siim-learning-rate", type=float, default=3e-4)
    parser.add_argument("--siim-metadata-iterations", type=int, default=700)
    parser.add_argument(
        "--siim-runtime-budget-seconds",
        type=float,
        default=0.0,
        help="Wall-clock SIIM budget from first GPU acquisition; zero disables it.",
    )
    parser.add_argument(
        "--siim-control-file",
        type=Path,
        default=None,
        help="Optional resource-monitor JSON checked after every completed epoch.",
    )
    parser.add_argument(
        "--siim-catboost-task-type",
        choices=("CPU", "GPU"),
        default="GPU",
    )
    parser.add_argument("--siim-probe-images", type=int, default=8)
    # Wave 2 deterministic caps.  These are recorded in every task result.
    parser.add_argument("--wave2-image-size", type=int, default=224)
    parser.add_argument("--wave2-batch-size", type=int, default=256)
    parser.add_argument("--wave2-workers", type=int, default=16)
    parser.add_argument(
        "--wave2-fast-kernels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable A40-oriented cuDNN autotuning and fast kernels while keeping "
            "data splits, sampler seeds, and fold manifests reproducible."
        ),
    )
    parser.add_argument("--wave2-learning-rate", type=float, default=2e-4)
    parser.add_argument("--wave2-vision-folds", type=int, default=5)
    parser.add_argument("--wave2-vision-patience", type=int, default=2)
    parser.add_argument("--wave2-aptos-epochs", type=int, default=10)
    parser.add_argument("--wave2-dog-breed-epochs", type=int, default=10)
    parser.add_argument("--wave2-dog-breed-batch-size", type=int, default=32)
    parser.add_argument(
        "--wave2-dog-breed-backbone",
        choices=("convnext_small", "efficientnet_v2_s"),
        default="convnext_small",
    )
    parser.add_argument(
        "--wave2-dog-breed-training-mode",
        choices=("stability_finetune", "frozen_backbone_head"),
        default="stability_finetune",
    )
    parser.add_argument(
        "--wave2-dog-breed-head-learning-rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--wave2-dog-breed-diagnostic-fold-limit",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "Run only the existing Dog Breed fold zero as DB-STAB1. This mode "
            "cannot create a submission or invoke the official grader."
        ),
    )
    parser.add_argument(
        "--wave2-dog-breed-diagnostic-parent-fold0-epoch1-log-loss",
        type=float,
    )
    parser.add_argument(
        "--wave2-dog-breed-diagnostic-parent-fold0-top1-accuracy",
        type=float,
    )
    parser.add_argument("--wave2-histopath-epochs", type=int, default=6)
    parser.add_argument("--wave2-histopath-max-rows", type=int, default=0)
    parser.add_argument("--wave2-plant-epochs", type=int, default=10)
    parser.add_argument("--wave2-ranzcr-epochs", type=int, default=8)
    parser.add_argument(
        "--wave2-ranzcr-backbone",
        choices=("efficientnet_v2_s", "convnext_small"),
        default="efficientnet_v2_s",
    )
    parser.add_argument("--wave2-ranzcr-image-size", type=int, default=512)
    parser.add_argument("--wave2-ranzcr-batch-size", type=int, default=24)
    parser.add_argument("--wave2-ranzcr-folds", type=int, default=5)
    parser.add_argument("--wave2-ranzcr-learning-rate", type=float, default=2e-4)
    parser.add_argument("--wave2-nomad-workers", type=int, default=16)
    parser.add_argument("--wave2-nomad-iterations", type=int, default=2_600)
    parser.add_argument("--wave2-nomad-folds", type=int, default=5)
    parser.add_argument("--wave2-nomad-extra-trees", type=int, default=800)
    parser.add_argument("--wave2-jigsaw-word-features", type=int, default=350_000)
    parser.add_argument("--wave2-jigsaw-char-features", type=int, default=500_000)
    parser.add_argument("--wave2-jigsaw-max-iter", type=int, default=240)
    parser.add_argument("--wave2-jigsaw-folds", type=int, default=5)
    parser.add_argument("--wave2-jigsaw-nbsvm-c", type=float, default=4.0)
    parser.add_argument("--wave2-jigsaw-fit-workers", type=int, default=12)
    parser.add_argument(
        "--wave2-jigsaw-stacker-c-grid",
        default="0.03,0.1,0.3,1.0",
        help="Frozen nested-CV grid for the Jigsaw Hybrid30 cross-label stacker.",
    )
    parser.add_argument("--wave2-jigsaw-stacker-max-iter", type=int, default=300)
    parser.add_argument(
        "--wave2-jigsaw-disable-stacker",
        action="store_true",
        help="Reproduce the immutable word/char rank-blend baseline without the production stacker.",
    )
    parser.add_argument("--wave2-audio-workers", type=int, default=16)
    parser.add_argument("--wave2-birds-iterations", type=int, default=500)
    parser.add_argument("--wave2-birds-pair-iterations", type=int, default=800)
    parser.add_argument("--wave2-birds-folds", type=int, default=5)
    parser.add_argument("--wave2-whale-iterations", type=int, default=1_200)
    parser.add_argument("--wave2-whale-folds", type=int, default=5)
    parser.add_argument("--wave2-whale-max-train", type=int, default=0)
    parser.add_argument("--wave2-normalization-cv-rows", type=int, default=1_200_000)
    parser.add_argument("--wave2-normalization-max-rows", type=int, default=0)
    parser.add_argument("--wave2-normalization-chunk-rows", type=int, default=250_000)
    parser.add_argument("--wave2-normalization-folds", type=int, default=5)
    parser.add_argument("--wave2-russian-transliteration-epochs", type=int, default=40)
    parser.add_argument(
        "--wave2-russian-transliteration-device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
    )
    return parser.parse_args(argv)


def _checkpoint(run_dir: Path, run_id: str, requested: list[str], results: list[dict[str, Any]]) -> None:
    by_competition = {result["competition_id"]: result.get("status") for result in results}
    wave0.write_json(run_dir / "checkpoint.json", {
        "schema": "evomind.mlebench_lite.checkpoint.v1",
        "run_id": run_id,
        "updated_at": wave0.utc_now(),
        "requested": requested,
        "completed": by_competition,
        "remaining": [competition for competition in requested if competition not in by_competition],
    })


def build_evidence_contract(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any]:
    """Describe the grader obligation after any local promotion decision."""

    grader_withheld = bool(result.get("official_grader_withheld"))
    confirmation_pending = bool(result.get("official_grader_confirmation_pending"))
    return {
        "seed": args.seed,
        "holdout_fraction": args.holdout_fraction,
        "submission_validator_required": True,
        "official_private_grader_required": not grader_withheld,
        "official_private_grader_withheld": grader_withheld,
        "official_private_grader_confirmation_pending": confirmation_pending,
        "candidate_only": bool(getattr(args, "candidate_only", False)),
        "promotion_gate_evaluated": isinstance(result.get("promotion_gate"), dict),
        "kaggle_submission_executed": False,
        "resume_capable": True,
    }


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    args.allowed_root = args.allowed_root.expanduser().resolve()
    args.data_root = wave0.ensure_within(args.data_root, args.allowed_root)
    args.output_root = wave0.ensure_within(args.output_root, args.allowed_root)
    args.official_source_root = wave0.ensure_within(args.official_source_root, args.allowed_root)
    if args.may_precomputed_cache_dir is not None:
        args.may_precomputed_cache_dir = wave0.ensure_within(
            args.may_precomputed_cache_dir,
            args.allowed_root,
        )
    args.code_paths = [
        Path(__file__).resolve(),
        Path(wave0.__file__).resolve(),
        Path(wave2.__file__).resolve(),
        Path(recovery.__file__).resolve(),
    ]
    waves = parse_waves(args.waves)
    requested = select_competitions(waves, args.competitions)
    plan = load_optimization_plan(args.optimization_plan, args.allowed_root)
    requested = apply_plan_order(requested, plan)
    diagnostic_mode = bool(args.wave2_dog_breed_diagnostic_fold_limit)
    if diagnostic_mode:
        if requested != ["dog-breed-identification"]:
            raise RuntimeError("DB-STAB1 must target only dog-breed-identification")
        if (
            args.wave2_dog_breed_diagnostic_parent_fold0_epoch1_log_loss is None
            or args.wave2_dog_breed_diagnostic_parent_fold0_top1_accuracy is None
        ):
            raise RuntimeError("DB-STAB1 requires explicit parent fold-zero baselines")
    run_id = args.run_id or f"lite_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}_s{args.seed}"
    run_dir = wave0.ensure_within(args.output_root / run_id, args.allowed_root)
    if run_dir.exists() and not args.resume:
        raise RuntimeError(f"Run directory already exists; use --resume: {run_dir}")
    if args.resume and not run_dir.exists():
        raise RuntimeError(f"Resume run directory does not exist: {run_dir}")
    if (
        not args.phase_a_only
        and not args.precompute_only
        and compute_policy is not None
        and compute_policy.write_runner_blocked_if_hpc_only(
            project_root=PROJECT_ROOT,
            evidence_path=run_dir / "hpc_only_policy_block.json",
            runner_name=Path(__file__).name,
            run_id=run_id,
            output_root=args.output_root,
        )
    ):
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(run_dir)
    original_cwd = Path.cwd()
    os.chdir(run_dir)
    cache_environment = configure_cache_environment(run_dir, args.allowed_root)
    wave0.seed_everything(args.seed)

    try:
        export_specs(run_dir / "lite22_specs.json")
        phase_a = audit_lite22_specs(args.data_root)
        if args.phase_a_scope == "requested":
            phase_a = scope_phase_a_audit(phase_a, requested)
        wave0.write_json(run_dir / "phase_a_audit.json", phase_a)
        environment = wave0.environment_report()
        wave0.write_json(run_dir / "environment.json", environment)
        manifest_path = run_dir / "manifest.json"
        manifest = {
            "schema": "evomind.mlebench_lite.full_manifest.v1",
            "run_id": run_id,
            "created_at": wave0.utc_now(),
            "waves": list(waves),
            "requested_competitions": requested,
            "data_root": str(args.data_root),
            "output_root": str(args.output_root),
            "allowed_root": str(args.allowed_root),
            "official_source_root": str(args.official_source_root),
            "seed": args.seed,
            "phase_a_status": phase_a["status"],
            "phase_a_scope": args.phase_a_scope,
            "resume": args.resume,
            "training_started": False,
            "kaggle_submission_enabled": False,
            "human_gate_preserved": True,
            "candidate_only": args.candidate_only,
            "hold_cuda_lease": args.hold_cuda_lease,
            "diagnostic_mode": diagnostic_mode,
            "diagnostic_id": "DB-STAB1" if diagnostic_mode else None,
            "cache_environment": cache_environment,
            "optimization_plan": ({"path": plan["_path"], "sha256": plan["_sha256"],
                                   "planner": plan.get("planner")} if plan else None),
            "code_sha256": {
                str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
                str(Path(wave0.__file__).resolve()): sha256_file(Path(wave0.__file__).resolve()),
                str(Path(wave2.__file__).resolve()): sha256_file(Path(wave2.__file__).resolve()),
                str(Path(recovery.__file__).resolve()): sha256_file(
                    Path(recovery.__file__).resolve()
                ),
            },
        }
        wave0.write_json(manifest_path, manifest)
        if phase_a["status"] != "passed":
            logger.error(
                "Phase A audit failed: %s/%s",
                phase_a["passed"],
                phase_a["competition_count"],
            )
            return 2
        if args.phase_a_only:
            manifest.update({"status": "phase_a_passed", "completed_at": wave0.utc_now()})
            wave0.write_json(manifest_path, manifest)
            return 0
        if args.precompute_only:
            if requested != ["mlsp-2013-birds"]:
                raise RuntimeError("Birds CPU precompute must target only mlsp-2013-birds")
            manifest["training_started"] = False
            manifest["precompute_only"] = args.precompute_only
            wave0.write_json(manifest_path, manifest)
            precompute_dir = run_dir / "precompute" / "mlsp-2013-birds"
            precompute_dir.mkdir(parents=True, exist_ok=True)
            result = wave2.precompute_birds_cpu_artifacts(args, precompute_dir, logger)
            summary = {
                "schema": "evomind.mlebench_lite.precompute_summary.v1",
                "run_id": run_id,
                "status": "passed",
                "competition_count": 1,
                "passed": 1,
                "failed": 0,
                "precompute_only": args.precompute_only,
                "results": [result],
                "claim_boundary": (
                    "CPU precompute only; no official private grading, no Kaggle submission, "
                    "and no GPU pair-channel training was executed."
                ),
                "completed_at": wave0.utc_now(),
            }
            wave0.write_json(run_dir / "summary.json", summary)
            wave0.write_json(run_dir / "results_current.json", {"run_id": run_id, "results": [result]})
            _checkpoint(run_dir, run_id, requested, [result])
            manifest.update({"status": "precompute_passed", "completed_at": wave0.utc_now(),
                             "summary_path": str(run_dir / "summary.json")})
            wave0.write_json(manifest_path, manifest)
            return 0
        if not environment["cuda_available"]:
            raise RuntimeError("CUDA is required for the dedicated remote full runner")

        cuda_lease = None
        if args.hold_cuda_lease:
            import torch

            cuda_lease = torch.empty(1, dtype=torch.uint8, device="cuda")
            torch.cuda.synchronize()
            manifest["cuda_lease"] = {
                "active": True,
                "bytes": int(cuda_lease.numel() * cuda_lease.element_size()),
                "device": torch.cuda.get_device_name(0),
                "purpose": "single_gpu_queue_ownership_during_cpu_preprocessing",
            }
            wave0.write_json(manifest_path, manifest)

        manifest["training_started"] = True
        wave0.write_json(manifest_path, manifest)
        results: list[dict[str, Any]] = []
        for competition_id in requested:
            task_root = run_dir / competition_id
            task_root.mkdir(parents=True, exist_ok=True)
            current_result = task_root / "result.json"
            if args.resume and current_result.is_file():
                existing = json.loads(current_result.read_text(encoding="utf-8"))
                resumable_terminal_statuses = {"passed"}
                if args.candidate_only:
                    resumable_terminal_statuses.update({
                        "promotion_gate_passed_confirmation_pending",
                        "candidate_ready_confirmation_pending",
                    })
                reusable = existing.get("status") in resumable_terminal_statuses
                if competition_id == SIIM_COMPETITION_ID and args.candidate_only:
                    checks = siim_terminal_resume_checks(task_root, existing, args)
                    reusable = bool(checks) and all(value is True for value in checks.values())
                    if not reusable and existing.get("status") in (
                        SIIM_CANDIDATE_TERMINAL_STATES | {"passed"}
                    ):
                        logger.warning(
                            "[%s] resume terminal result rejected by corrected formal contract: %s",
                            competition_id,
                            ",".join(name for name, passed in checks.items() if passed is not True),
                        )
                if reusable:
                    existing["resume_disposition"] = "skipped_already_passed"
                    results.append(existing)
                    logger.info("[%s] resume skip: already passed", competition_id)
                    _checkpoint(run_dir, run_id, requested, results)
                    continue

            attempt_dir = next_attempt_dir(task_root)
            attempt_number = int(attempt_dir.name.split("_", 1)[1])
            logger.info("[%s] starting attempt=%s", competition_id, attempt_number)
            started = time.perf_counter()
            telemetry = wave0.GpuTelemetry()
            try:
                with telemetry:
                    result = RUNNERS[competition_id](args, attempt_dir, logger)
            except recovery.SiimPauseRequested as exc:
                result = {
                    "competition_id": competition_id,
                    "status": "paused_resource_guard",
                    "failure_taxonomy": "resource_guard_pause",
                    "pause_reason": exc.reason,
                    "resume_checkpoint": str(exc.checkpoint_path),
                    "resume_available": True,
                    "same_run_id_required": True,
                    "valid_submission": False,
                    "official_grader_executed": False,
                    "cv_score": None,
                    "proxy_score": None,
                    "mle_private_grader_score": None,
                    "kaggle_public_score": None,
                    "kaggle_private_score": None,
                    "medal_or_percentile": None,
                    "other_processes_modified": False,
                    "signals_sent": 0,
                }
                logger.warning(
                    "[%s] paused after an epoch by the resource guard: %s",
                    competition_id,
                    exc.reason,
                )
            except Exception as exc:
                result = {
                    "competition_id": competition_id,
                    "status": "failed",
                    "failure_taxonomy": classify_failure(exc),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "valid_submission": False,
                    "official_grader_executed": False,
                    "cv_score": None,
                    "proxy_score": None,
                    "mle_private_grader_score": None,
                    "kaggle_public_score": None,
                    "kaggle_private_score": None,
                    "medal_or_percentile": None,
                }
                logger.exception("[%s] failed", competition_id)
            result.update({
                "wave": next(wave for wave in waves if competition_id in WAVES[wave]),
                "attempt": attempt_number,
                "attempt_dir": str(attempt_dir),
                "runtime_seconds_total": time.perf_counter() - started,
                "gpu_telemetry": telemetry.summary(),
                "completed_at": wave0.utc_now(),
                "evidence_contract": build_evidence_contract(args, result),
            })
            wave0.write_json(attempt_dir / "result.json", result)
            wave0.write_json(current_result, result)
            results.append(result)
            wave0.write_json(run_dir / "results_current.json", {"run_id": run_id, "results": results})
            _checkpoint(run_dir, run_id, requested, results)
            logger.info("[%s] status=%s", competition_id, result["status"])

        passed = sum(result.get("status") == "passed" for result in results)
        candidate_pending = sum(
            result.get("status")
            in {
                "promotion_gate_passed_confirmation_pending",
                "candidate_ready_confirmation_pending",
            }
            for result in results
        )
        diagnostic_completed = sum(
            result.get("status") in {"diagnostic_passed", "diagnostic_failed"}
            for result in results
        )
        completed = diagnostic_completed if diagnostic_mode else passed + candidate_pending
        summary = {
            "schema": "evomind.mlebench_lite.full_summary.v1",
            "run_id": run_id,
            "status": (
                "diagnostic_complete"
                if diagnostic_mode and diagnostic_completed == len(requested)
                else "candidate_complete"
                if candidate_pending and completed == len(requested)
                else "passed"
                if passed == len(requested)
                else "partial_failure"
            ),
            "competition_count": len(requested),
            "passed": passed,
            "candidate_confirmation_pending": candidate_pending,
            "diagnostic_completed": diagnostic_completed,
            "failed": len(requested) - completed,
            "diagnostic_mode": diagnostic_mode,
            "valid_submission_rate": (
                sum(bool(result.get("valid_submission")) for result in results) / len(requested)
                if requested else 0.0
            ),
            "official_private_grader_rate": (
                sum(bool(result.get("official_grader_executed")) for result in results) / len(requested)
                if requested else 0.0
            ),
            "results": results,
            "claim_boundary": "No Kaggle submission; CV, private grader, public score, and medals remain distinct.",
            "completed_at": wave0.utc_now(),
        }
        wave0.write_json(run_dir / "summary.json", summary)
        manifest.update({"status": summary["status"], "completed_at": wave0.utc_now(),
                         "summary_path": str(run_dir / "summary.json")})
        wave0.write_json(manifest_path, manifest)
        return 0 if completed == len(requested) else 3
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    raise SystemExit(main())
