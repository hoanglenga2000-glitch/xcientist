#!/usr/bin/env python3
"""Execute the approved MLE-Bench Lite Phase A audit and Wave 0 baselines.

The script is intended for the dedicated AIMSLAB allocation.  It refuses to
write outside ``/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra`` and contains no
Kaggle submission code.  SIIM is deliberately limited to metadata, image I/O,
a tiny CUDA inference probe, and a clearly labelled constant-schema grader
probe; it does not launch a long training job.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from research_os.mlebench_phase_a import (
    DEFAULT_OFFICIAL_DATA_ROOT,
    DEFAULT_REMOTE_ROOT,
    audit_lite22_specs,
    export_specs,
    get_competition_spec,
    grade_private_submission,
    resolve_competition,
    sha256_file,
    to_proxy_score,
    validate_submission_file,
)

WAVE0_COMPETITIONS = (
    "tabular-playground-series-may-2022",
    "spooky-author-identification",
    "aerial-cactus-identification",
    "siim-isic-melanoma-classification",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_within(path: Path, root: Path) -> Path:
    root = root.expanduser().resolve()
    target = path.expanduser().resolve()
    if target != root and root not in target.parents:
        raise RuntimeError(f"Output path escapes the dedicated remote root: {target}")
    return target


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def configure_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("mlebench_wave0")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "wave0.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def nvidia_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,pstate",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    if result.returncode:
        return {"ok": False, "error": result.stderr.strip(), "captured_at": utc_now()}
    line = result.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in line.split(",")]
    return {
        "ok": True,
        "captured_at": utc_now(),
        "index": int(parts[0]),
        "name": parts[1],
        "memory_total_mib": int(parts[2]),
        "memory_used_mib": int(parts[3]),
        "memory_free_mib": int(parts[4]),
        "utilization_gpu_percent": int(parts[5]),
        "pstate": parts[6],
    }


class GpuTelemetry:
    def __init__(self, interval_seconds: float = 1.0) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "GpuTelemetry":
        def collect() -> None:
            while not self._stop.is_set():
                try:
                    self.samples.append(nvidia_snapshot())
                except Exception as exc:
                    self.samples.append({"ok": False, "captured_at": utc_now(), "error": str(exc)})
                self._stop.wait(self.interval_seconds)

        self._thread = threading.Thread(target=collect, name="gpu-telemetry", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def summary(self) -> dict[str, Any]:
        successful = [sample for sample in self.samples if sample.get("ok")]
        return {
            "sample_count": len(self.samples),
            "peak_memory_used_mib": max((sample["memory_used_mib"] for sample in successful), default=None),
            "peak_utilization_gpu_percent": max(
                (sample["utilization_gpu_percent"] for sample in successful), default=None
            ),
            "first": successful[0] if successful else None,
            "last": successful[-1] if successful else None,
            "samples": self.samples,
        }


def safe_extract_zip(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    target_root = target.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            output = (target / member.filename).resolve()
            if output != target_root and target_root not in output.parents:
                raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
        handle.extractall(target)


def private_grade(
    submission_path: Path,
    competition_id: str,
    args: argparse.Namespace,
    task_dir: Path,
    budget: dict[str, Any],
) -> dict[str, Any]:
    report_path = task_dir / "private_grader.json"
    code_paths = [
        Path(__file__),
        Path(__file__).resolve().parents[1] / "src/research_os/mlebench_phase_a.py",
        *[Path(path) for path in getattr(args, "code_paths", ())],
    ]
    return grade_private_submission(
        submission_path,
        competition_id,
        args.data_root,
        official_source_root=args.official_source_root,
        seed=args.seed,
        budget=budget,
        code_paths=list(dict.fromkeys(code_paths)),
        output_path=report_path,
    )


def build_metric_promotion_gate(
    *,
    name: str,
    metric: str,
    direction: str,
    score: float,
    threshold: float,
    extra_checks: dict[str, bool] | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an immutable pre-private-grader metric gate.

    The helper deliberately accepts only an already computed internal score.
    It never invokes a private grader and therefore keeps model selection and
    final benchmark measurement separated.
    """

    value = float(score)
    boundary = float(threshold)
    finite = math.isfinite(value) and math.isfinite(boundary)
    if direction == "maximize":
        threshold_passed = finite and value >= boundary
        operator = ">="
    elif direction == "minimize":
        threshold_passed = finite and value <= boundary
        operator = "<="
    else:
        raise ValueError(f"Unsupported promotion direction: {direction}")
    checks = {
        "finite_internal_metric": finite,
        "aggregate_threshold": threshold_passed,
        **(extra_checks or {}),
    }
    return {
        "schema": "evomind.mlebench_lite.metric_promotion_gate.v1",
        "name": name,
        "metric": metric,
        "direction": direction,
        "operator": operator,
        "threshold": boundary,
        "internal_score": value,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "official_grader_executed": False,
        "evidence": evidence or {},
        "claim_boundary": "Internal promotion evidence is not an official medal.",
    }


def finalize_scored_task(
    *,
    competition_id: str,
    submission: pd.DataFrame,
    cv_score: float | None,
    args: argparse.Namespace,
    task_dir: Path,
    budget: dict[str, Any],
    extra: dict[str, Any],
    score_scope: str = "model_baseline",
    promotion_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = get_competition_spec(competition_id)
    submission_path = task_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    validation = validate_submission_file(submission_path, spec, args.data_root)
    write_json(task_dir / "submission_validation.json", validation)
    gate_required = promotion_gate is not None
    if gate_required:
        if not isinstance(promotion_gate.get("passed"), bool):
            raise RuntimeError("Promotion gate must contain an explicit boolean passed field")
        write_json(task_dir / "promotion_gate.json", promotion_gate)
    gate_passed = not gate_required or promotion_gate["passed"] is True
    gate_failed = gate_required and not gate_passed
    candidate_only = bool(getattr(args, "candidate_only", False))
    grader_withheld = gate_failed or candidate_only
    if grader_withheld:
        withheld_reason = (
            "internal_promotion_gate_failed"
            if gate_failed
            else "explicit_candidate_only_human_confirmation_required"
        )
        grader = {
            "schema": "evomind.mlebench_lite.private_grade_withheld.v1",
            "status": "withheld",
            "reason": withheld_reason,
            "official_mlebench_grader_executed": False,
            "score": None,
        }
    else:
        grader = private_grade(submission_path, competition_id, args, task_dir, budget)
    if validation.get("valid") is not True:
        status = "failed"
    elif gate_failed:
        status = "promotion_gate_failed"
    elif candidate_only:
        status = (
            "promotion_gate_passed_confirmation_pending"
            if gate_required
            else "candidate_ready_confirmation_pending"
        )
    else:
        status = "passed" if grader.get("status") == "passed" else "failed"
    confirmation_pending = bool(
        candidate_only and gate_passed and validation.get("valid") is True
    )
    return {
        "competition_id": competition_id,
        "status": status,
        "metric": spec.metric,
        "direction": spec.direction,
        "score_scope": score_scope,
        "cv_score": cv_score,
        "proxy_score": to_proxy_score(spec.metric, cv_score) if cv_score is not None else None,
        "mle_private_grader_score": grader.get("score"),
        "official_grader_executed": grader.get("official_mlebench_grader_executed", False),
        "official_grader_withheld": grader_withheld,
        "official_grader_confirmation_pending": confirmation_pending,
        "candidate_only": candidate_only,
        "promotion_gate": promotion_gate,
        "valid_submission": validation.get("valid", False),
        "submission_path": str(submission_path),
        "submission_sha256": sha256_file(submission_path),
        "kaggle_public_score": None,
        "kaggle_private_score": None,
        "medal_or_percentile": None,
        "claim_boundary": (
            "Candidate artifact only; official private grading awaits explicit human confirmation."
            if confirmation_pending
            else "Internal promotion evidence only; the private grader was intentionally withheld."
            if gate_failed
            else "CV and MLE private grading only; no Kaggle submission was made."
        ),
        "validation": validation,
        "private_grader": grader,
        **extra,
    }


def run_may2022(args: argparse.Namespace, task_dir: Path, logger: logging.Logger) -> dict[str, Any]:
    from catboost import CatBoostClassifier
    from sklearn.model_selection import train_test_split

    competition_id = "tabular-playground-series-may-2022"
    resolved = resolve_competition(competition_id, args.data_root)
    public = resolved.public_dir
    train_path = public / "train.csv"
    test_path = public / "test.csv"
    logger.info("[%s] loading train/test", competition_id)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    original_rows = len(train)
    if args.may_max_train_rows and len(train) > args.may_max_train_rows:
        train = train.sample(args.may_max_train_rows, random_state=args.seed).sort_index()

    target = "target"
    features = [column for column in train.columns if column not in {target, "id"}]
    categorical = [column for column in features if not pd.api.types.is_numeric_dtype(train[column])]
    for column in categorical:
        train[column] = train[column].fillna("__MISSING__").astype(str)
        test[column] = test[column].fillna("__MISSING__").astype(str)
    x_train, x_valid, y_train, y_valid = train_test_split(
        train[features], train[target], test_size=args.holdout_fraction, random_state=args.seed, stratify=train[target]
    )
    model = CatBoostClassifier(
        iterations=args.catboost_iterations,
        depth=8,
        learning_rate=0.12,
        loss_function="Logloss",
        eval_metric="AUC",
        task_type="GPU",
        devices="0",
        random_seed=args.seed,
        border_count=64,
        l2_leaf_reg=4.0,
        random_strength=0.5,
        od_type="Iter",
        od_wait=35,
        verbose=50,
        allow_writing_files=False,
    )
    started = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        cat_features=categorical,
        eval_set=(x_valid, y_valid),
        use_best_model=True,
    )
    valid_probability = model.predict_proba(x_valid)[:, 1]
    from research_os.mlebench_phase_a import compute_metric

    cv_score = compute_metric("roc_auc", y_valid, valid_probability)
    test_probability = model.predict_proba(test[features])[:, 1]
    sample = pd.read_csv(resolved.sample_submission_path)
    sample["target"] = test_probability
    model.save_model(str(task_dir / "catboost_model.cbm"))
    runtime = time.perf_counter() - started
    budget = {
        "seed": args.seed,
        "holdout_fraction": args.holdout_fraction,
        "train_rows_available": original_rows,
        "train_rows_used": len(train),
        "iterations_requested": args.catboost_iterations,
        "best_iteration": model.get_best_iteration(),
    }
    return finalize_scored_task(
        competition_id=competition_id,
        submission=sample,
        cv_score=cv_score,
        args=args,
        task_dir=task_dir,
        budget=budget,
        extra={
            "runtime_seconds_model": runtime,
            "model_family": "CatBoostClassifier_GPU",
            "feature_count": len(features),
            "categorical_feature_count": len(categorical),
            "budget": budget,
        },
    )


def normalize_multiclass_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, None)
    return values / values.sum(axis=1, keepdims=True)


def apply_multiclass_blend(
    word_probability: np.ndarray,
    char_probability: np.ndarray,
    *,
    word_weight: float,
    temperature: float,
) -> np.ndarray:
    blended = normalize_multiclass_probabilities(
        float(word_weight) * word_probability + (1.0 - float(word_weight)) * char_probability
    )
    logits = np.log(np.clip(blended, 1e-7, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def select_multiclass_blend(
    word_probability: np.ndarray,
    char_probability: np.ndarray,
    labels: pd.Series,
    classes: list[str],
) -> tuple[np.ndarray, float, float, float]:
    from sklearn.metrics import log_loss

    best: tuple[float, float, float, np.ndarray] | None = None
    for weight in np.linspace(0.2, 0.8, 13):
        for temperature in (0.75, 0.85, 0.95, 1.0, 1.1, 1.2, 1.35):
            prediction = apply_multiclass_blend(
                word_probability,
                char_probability,
                word_weight=float(weight),
                temperature=temperature,
            )
            score = float(log_loss(labels, prediction, labels=classes))
            candidate = (score, abs(float(weight) - 0.5), abs(temperature - 1.0), prediction)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
                best_weight = float(weight)
                best_temperature = float(temperature)
    assert best is not None
    return best[3], best_weight, best_temperature, best[0]


def nbsvm_multiclass_predict(
    train_text: pd.Series,
    train_labels: pd.Series,
    validation_text: pd.Series,
    test_text: pd.Series,
    *,
    classes: list[str],
    analyzer: str,
    ngram_range: tuple[int, int],
    max_features: int,
    c_value: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        lowercase=True,
        strip_accents="unicode" if analyzer == "word" else None,
        ngram_range=ngram_range,
        min_df=2,
        max_features=max_features,
        binary=True,
        use_idf=False,
        norm="l2",
    )
    x_train = vectorizer.fit_transform(train_text.fillna(""))
    x_valid = vectorizer.transform(validation_text.fillna(""))
    x_test = vectorizer.transform(test_text.fillna(""))
    valid_probability = np.zeros((len(validation_text), len(classes)), dtype=np.float64)
    test_probability = np.zeros((len(test_text), len(classes)), dtype=np.float64)
    labels = train_labels.astype(str).to_numpy()
    for class_index, class_name in enumerate(classes):
        binary_labels = (labels == class_name).astype(np.int8)
        positive = np.asarray(x_train[binary_labels == 1].sum(axis=0)).ravel()
        negative = np.asarray(x_train[binary_labels == 0].sum(axis=0)).ravel()
        ratio = np.log((positive + 1.0) / (binary_labels.sum() + 1.0)) - np.log(
            (negative + 1.0) / ((binary_labels == 0).sum() + 1.0)
        )
        classifier = LogisticRegression(
            C=c_value,
            max_iter=500,
            solver="liblinear",
            random_state=seed + class_index,
        )
        classifier.fit(x_train.multiply(ratio).tocsr(), binary_labels)
        valid_probability[:, class_index] = classifier.predict_proba(
            x_valid.multiply(ratio).tocsr()
        )[:, 1]
        test_probability[:, class_index] = classifier.predict_proba(
            x_test.multiply(ratio).tocsr()
        )[:, 1]
    return (
        normalize_multiclass_probabilities(valid_probability),
        normalize_multiclass_probabilities(test_probability),
        int(x_train.shape[1]),
    )


def run_spooky(args: argparse.Namespace, task_dir: Path, logger: logging.Logger) -> dict[str, Any]:
    from sklearn.metrics import log_loss
    from sklearn.model_selection import StratifiedKFold

    competition_id = "spooky-author-identification"
    resolved = resolve_competition(competition_id, args.data_root)
    train = pd.read_csv(resolved.public_dir / "train.csv")
    test = pd.read_csv(resolved.public_dir / "test.csv")
    classes = [str(value) for value in resolved.prediction_columns]
    observed_classes = sorted(train["author"].astype(str).unique())
    if set(classes) != set(observed_classes):
        raise RuntimeError("Spooky author class mapping does not match the submission schema")
    folds = StratifiedKFold(n_splits=args.spooky_folds, shuffle=True, random_state=args.seed)
    word_oof = np.zeros((len(train), len(classes)), dtype=np.float64)
    char_oof = np.zeros_like(word_oof)
    word_test = np.zeros((len(test), len(classes)), dtype=np.float64)
    char_test = np.zeros_like(word_test)
    fold_assignment = np.full(len(train), -1, dtype=np.int16)
    feature_counts: list[dict[str, int]] = []

    logger.info("[%s] fitting %s-fold word/char NB-SVM ensemble", competition_id, args.spooky_folds)
    started = time.perf_counter()
    for fold_index, (fit_index, valid_index) in enumerate(folds.split(train, train["author"])):
        fold_assignment[valid_index] = fold_index
        word_valid, word_fold_test, word_features = nbsvm_multiclass_predict(
            train.loc[fit_index, "text"],
            train.loc[fit_index, "author"],
            train.loc[valid_index, "text"],
            test["text"],
            classes=classes,
            analyzer="word",
            ngram_range=(1, 3),
            max_features=args.spooky_word_features,
            c_value=args.spooky_nbsvm_c,
            seed=args.seed + fold_index * 10,
        )
        char_valid, char_fold_test, char_features = nbsvm_multiclass_predict(
            train.loc[fit_index, "text"],
            train.loc[fit_index, "author"],
            train.loc[valid_index, "text"],
            test["text"],
            classes=classes,
            analyzer="char",
            ngram_range=(3, 5),
            max_features=args.spooky_char_features,
            c_value=args.spooky_nbsvm_c,
            seed=args.seed + fold_index * 10 + 5,
        )
        word_oof[valid_index] = word_valid
        char_oof[valid_index] = char_valid
        word_test += word_fold_test / args.spooky_folds
        char_test += char_fold_test / args.spooky_folds
        feature_counts.append({"fold": fold_index, "word": word_features, "char": char_features})

    blended_oof, word_weight, temperature, cv_score = select_multiclass_blend(
        word_oof, char_oof, train["author"].astype(str), classes
    )
    test_probability = apply_multiclass_blend(
        word_test,
        char_test,
        word_weight=word_weight,
        temperature=temperature,
    )
    component_scores = {
        "word_nbsvm_log_loss": float(log_loss(train["author"], word_oof, labels=classes)),
        "char_nbsvm_log_loss": float(log_loss(train["author"], char_oof, labels=classes)),
        "blended_log_loss": float(cv_score),
    }
    oof_artifact = pd.DataFrame({"id": train["id"], "author": train["author"], "fold": fold_assignment})
    test_artifact = pd.DataFrame({"id": test["id"]})
    for class_index, class_name in enumerate(classes):
        oof_artifact[f"word_{class_name}"] = word_oof[:, class_index]
        oof_artifact[f"char_{class_name}"] = char_oof[:, class_index]
        oof_artifact[f"blend_{class_name}"] = blended_oof[:, class_index]
        test_artifact[f"word_{class_name}"] = word_test[:, class_index]
        test_artifact[f"char_{class_name}"] = char_test[:, class_index]
        test_artifact[f"blend_{class_name}"] = test_probability[:, class_index]
    oof_artifact.to_csv(task_dir / "spooky_oof_predictions.csv", index=False)
    test_artifact.to_csv(task_dir / "spooky_test_components.csv", index=False)

    sample = pd.read_csv(resolved.sample_submission_path)
    class_index = {name: index for index, name in enumerate(classes)}
    for column in resolved.prediction_columns:
        sample[column] = test_probability[:, class_index[column]]
    runtime = time.perf_counter() - started
    budget = {
        "seed": args.seed,
        "folds": args.spooky_folds,
        "train_rows": len(train),
        "word_max_features": args.spooky_word_features,
        "char_max_features": args.spooky_char_features,
        "nbsvm_c": args.spooky_nbsvm_c,
        "word_weight": word_weight,
        "temperature": temperature,
        "feature_counts": feature_counts,
        "full_refit": False,
        "fold_test_ensemble": True,
    }
    return finalize_scored_task(
        competition_id=competition_id,
        submission=sample,
        cv_score=cv_score,
        args=args,
        task_dir=task_dir,
        budget=budget,
        extra={
            "runtime_seconds_model": runtime,
            "model_family": "NB-SVM_word_char_OOF_temperature_ensemble",
            "classes": classes,
            "component_scores": component_scores,
            "budget": budget,
        },
    )


def _run_aerial_smallcnn_legacy(args: argparse.Namespace, task_dir: Path, logger: logging.Logger) -> dict[str, Any]:
    import torch
    from PIL import Image
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    competition_id = "aerial-cactus-identification"
    resolved = resolve_competition(competition_id, args.data_root)
    cache = task_dir / "cache"
    train_dir = cache / "train"
    test_dir = cache / "test"
    logger.info("[%s] extracting image archives into run-local cache", competition_id)
    safe_extract_zip(resolved.public_dir / "train.zip", train_dir)
    safe_extract_zip(resolved.public_dir / "test.zip", test_dir)
    labels = pd.read_csv(resolved.public_dir / "train.csv")
    train_rows, valid_rows = train_test_split(
        labels,
        test_size=args.holdout_fraction,
        random_state=args.seed,
        stratify=labels["has_cactus"],
    )

    class ImageDataset(Dataset):
        def __init__(self, rows: pd.DataFrame, directory: Path, transform: Any, labelled: bool) -> None:
            self.rows = rows.reset_index(drop=True)
            self.directory = directory
            self.transform = transform
            self.labelled = labelled

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> Any:
            row = self.rows.iloc[index]
            image = Image.open(self.directory / str(row["id"])).convert("RGB")
            tensor = self.transform(image)
            if self.labelled:
                return tensor, torch.tensor(float(row["has_cactus"]), dtype=torch.float32)
            return tensor

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.ToTensor()]
    )
    eval_transform = transforms.ToTensor()
    worker_count = min(8, max(2, (os.cpu_count() or 4) // 8))
    train_loader = DataLoader(
        ImageDataset(train_rows, train_dir, train_transform, True),
        batch_size=args.aerial_batch_size,
        shuffle=True,
        num_workers=worker_count,
        pin_memory=True,
        persistent_workers=worker_count > 0,
    )
    valid_loader = DataLoader(
        ImageDataset(valid_rows, train_dir, eval_transform, True),
        batch_size=args.aerial_batch_size,
        shuffle=False,
        num_workers=worker_count,
        pin_memory=True,
        persistent_workers=worker_count > 0,
    )

    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 1),
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats()
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(args.aerial_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images).squeeze(1)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * len(images)
        model.eval()
        true: list[float] = []
        probability: list[float] = []
        with torch.inference_mode():
            for images, targets in valid_loader:
                logits = model(images.cuda(non_blocking=True)).squeeze(1)
                probability.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                true.extend(targets.numpy().tolist())
        epoch_auc = float(roc_auc_score(true, probability))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": running_loss / len(train_loader.dataset),
                "valid_roc_auc": epoch_auc,
            }
        )
        logger.info("[%s] epoch=%s auc=%.6f", competition_id, epoch + 1, epoch_auc)

    cv_score = history[-1]["valid_roc_auc"]
    sample = pd.read_csv(resolved.sample_submission_path)
    test_loader = DataLoader(
        ImageDataset(sample[["id"]], test_dir, eval_transform, False),
        batch_size=args.aerial_batch_size,
        shuffle=False,
        num_workers=worker_count,
        pin_memory=True,
        persistent_workers=worker_count > 0,
    )
    predictions: list[float] = []
    model.eval()
    with torch.inference_mode():
        for images in test_loader:
            logits = model(images.cuda(non_blocking=True)).squeeze(1)
            predictions.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    sample["has_cactus"] = predictions
    torch.save(model.state_dict(), task_dir / "small_cnn_state.pt")
    runtime = time.perf_counter() - started
    process_peak_mib = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
    budget = {
        "seed": args.seed,
        "holdout_fraction": args.holdout_fraction,
        "epochs": args.aerial_epochs,
        "batch_size": args.aerial_batch_size,
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
    }
    return finalize_scored_task(
        competition_id=competition_id,
        submission=sample,
        cv_score=cv_score,
        args=args,
        task_dir=task_dir,
        budget=budget,
        extra={
            "runtime_seconds_model": runtime,
            "model_family": "SmallCNN_3xConv",
            "history": history,
            "torch_peak_memory_allocated_mib": process_peak_mib,
            "budget": budget,
        },
    )


def run_aerial(args: argparse.Namespace, task_dir: Path, logger: logging.Logger) -> dict[str, Any]:
    try:
        import mlebench_wave2_adapters as vision
    except ModuleNotFoundError:
        from scripts import mlebench_wave2_adapters as vision

    competition_id = "aerial-cactus-identification"
    resolved = resolve_competition(competition_id, args.data_root)
    cache = task_dir / "cache"
    train_dir = cache / "train"
    test_dir = cache / "test"
    logger.info("[%s] extracting image archives for pretrained ConvNeXt", competition_id)
    safe_extract_zip(resolved.public_dir / "train.zip", train_dir)
    safe_extract_zip(resolved.public_dir / "test.zip", test_dir)
    labels = pd.read_csv(resolved.public_dir / "train.csv")
    sample = pd.read_csv(resolved.sample_submission_path)
    return vision._run_vision(
        args=args,
        task_dir=task_dir,
        logger=logger,
        competition_id=competition_id,
        train_frame=labels,
        sample=sample,
        train_paths=[train_dir / str(value) for value in labels["id"]],
        test_paths=[test_dir / str(value) for value in sample["id"]],
        target_columns=["has_cactus"],
        mode="binary",
        epochs=args.aerial_epochs,
        backbone=getattr(args, "aerial_backbone", "convnext_tiny"),
        label_column="has_cactus",
        image_size=args.aerial_image_size,
        batch_size=args.aerial_batch_size,
        fold_count=getattr(args, "aerial_folds", None),
        tta_flips=True,
    )


def run_siim_precheck(args: argparse.Namespace, task_dir: Path, logger: logging.Logger) -> dict[str, Any]:
    import torch
    from PIL import Image
    from torch import nn
    from torchvision import transforms

    competition_id = "siim-isic-melanoma-classification"
    resolved = resolve_competition(competition_id, args.data_root)
    train_metadata = pd.read_csv(resolved.public_dir / "train.csv")
    test_metadata = pd.read_csv(resolved.public_dir / "test.csv")
    sample = pd.read_csv(resolved.sample_submission_path)
    image_ids = sample["image_name"].astype(str).head(args.siim_probe_images).tolist()
    image_paths = [resolved.public_dir / "jpeg" / "test" / f"{image_id}.jpg" for image_id in image_ids]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"SIIM JPEG probe files missing: {missing[:3]}")

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    tensors = [transform(Image.open(path).convert("RGB")) for path in image_paths]
    batch = torch.stack(tensors).cuda()
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 1),
    ).cuda()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = torch.sigmoid(model(batch)).squeeze(1).cpu().numpy()
    inference_seconds = time.perf_counter() - started

    # This is intentionally a constant schema/grader probe, not a model claim.
    sample["target"] = 0.5
    budget = {
        "seed": args.seed,
        "mode": "io_metadata_small_batch_inference_only",
        "probe_images": len(image_paths),
        "long_training_started": False,
    }
    return finalize_scored_task(
        competition_id=competition_id,
        submission=sample,
        cv_score=None,
        args=args,
        task_dir=task_dir,
        budget=budget,
        score_scope="constant_0.5_schema_and_private_grader_probe",
        extra={
            "runtime_seconds_model": inference_seconds,
            "model_family": "untrained_tiny_cnn_io_probe_only",
            "train_metadata_rows": len(train_metadata),
            "test_metadata_rows": len(test_metadata),
            "train_metadata_columns": list(train_metadata.columns),
            "test_metadata_columns": list(test_metadata.columns),
            "image_paths": [str(path) for path in image_paths],
            "input_batch_shape": list(batch.shape),
            "probe_output_min": float(outputs.min()),
            "probe_output_max": float(outputs.max()),
            "torch_peak_memory_allocated_mib": int(torch.cuda.max_memory_allocated() / (1024 * 1024)),
            "long_training_started": False,
            "budget": budget,
        },
    )


RUNNERS: dict[str, Callable[[argparse.Namespace, Path, logging.Logger], dict[str, Any]]] = {
    "tabular-playground-series-may-2022": run_may2022,
    "spooky-author-identification": run_spooky,
    "aerial-cactus-identification": run_aerial,
    "siim-isic-melanoma-classification": run_siim_precheck,
}


def environment_report() -> dict[str, Any]:
    import sklearn
    import torch

    return {
        "created_at": utc_now(),
        "python": sys.version,
        "platform": sys.platform,
        "cwd": str(Path.cwd()),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvidia_smi": nvidia_snapshot(),
        "local_gpu_used": False,
        "kaggle_submission_enabled": False,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_OFFICIAL_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_REMOTE_ROOT / "mlebench_lite_runs")
    parser.add_argument("--allowed-root", type=Path, default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--official-source-root", type=Path, required=True)
    parser.add_argument("--phase-a-only", action="store_true")
    parser.add_argument("--competitions", default=",".join(WAVE0_COMPETITIONS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--may-max-train-rows", type=int, default=400_000)
    parser.add_argument("--catboost-iterations", type=int, default=220)
    parser.add_argument("--spooky-folds", type=int, default=5)
    parser.add_argument("--spooky-word-features", type=int, default=120_000)
    parser.add_argument("--spooky-char-features", type=int, default=180_000)
    parser.add_argument("--spooky-nbsvm-c", type=float, default=4.0)
    parser.add_argument("--aerial-epochs", type=int, default=6)
    parser.add_argument(
        "--aerial-backbone",
        choices=("convnext_tiny", "convnext_small", "efficientnet_v2_s"),
        default="convnext_tiny",
    )
    parser.add_argument("--aerial-folds", type=int, default=5)
    parser.add_argument("--aerial-batch-size", type=int, default=256)
    parser.add_argument("--aerial-image-size", type=int, default=224)
    parser.add_argument("--wave2-workers", type=int, default=8)
    parser.add_argument("--wave2-learning-rate", type=float, default=2e-4)
    parser.add_argument("--siim-probe-images", type=int, default=8)
    parser.add_argument("--run-id", default="")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    args.allowed_root = args.allowed_root.expanduser().resolve()
    args.data_root = ensure_within(args.data_root, args.allowed_root)
    args.output_root = ensure_within(args.output_root, args.allowed_root)
    args.official_source_root = ensure_within(args.official_source_root, args.allowed_root)
    run_id = args.run_id or f"wave0_{datetime.now().strftime('%Y%m%d_%H%M%S')}_s{args.seed}"
    run_dir = ensure_within(args.output_root / run_id, args.allowed_root)
    run_dir.mkdir(parents=True, exist_ok=False)
    logger = configure_logging(run_dir)
    original_cwd = Path.cwd()
    os.chdir(run_dir)
    for name in ("TMPDIR", "TEMP", "TMP", "XDG_CACHE_HOME", "HF_HOME", "TORCH_HOME"):
        os.environ[name] = str(run_dir / "cache" / name.lower())
    (run_dir / "cache").mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    try:
        logger.info("Phase A audit started: run_id=%s", run_id)
        export_specs(run_dir / "lite22_specs.json")
        phase_a = audit_lite22_specs(args.data_root)
        write_json(run_dir / "phase_a_audit.json", phase_a)
        environment = environment_report()
        write_json(run_dir / "environment.json", environment)
        manifest = {
            "schema": "evomind.mlebench_lite.wave0_manifest.v1",
            "run_id": run_id,
            "created_at": utc_now(),
            "data_root": str(args.data_root),
            "output_root": str(args.output_root),
            "allowed_root": str(args.allowed_root),
            "official_source_root": str(args.official_source_root),
            "seed": args.seed,
            "phase_a_status": phase_a["status"],
            "training_started": False,
            "kaggle_submission_enabled": False,
            "human_gate_preserved": True,
            "code_sha256": {
                str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
            },
        }
        write_json(run_dir / "manifest.json", manifest)
        if phase_a["status"] != "passed":
            logger.error("Phase A audit failed: %s/22", phase_a["passed"])
            return 2
        if args.phase_a_only:
            manifest.update({"status": "phase_a_passed", "completed_at": utc_now()})
            write_json(run_dir / "manifest.json", manifest)
            logger.info("Phase A audit passed; phase-a-only run complete")
            return 0
        if not environment["cuda_available"]:
            raise RuntimeError("CUDA is required for the approved remote Wave 0 run")

        requested = [item.strip() for item in args.competitions.split(",") if item.strip()]
        invalid = sorted(set(requested) - set(WAVE0_COMPETITIONS))
        if invalid:
            raise RuntimeError(f"Only approved Wave 0 competitions are allowed: {invalid}")

        manifest["training_started"] = True
        manifest["requested_competitions"] = requested
        write_json(run_dir / "manifest.json", manifest)
        results: list[dict[str, Any]] = []
        for competition_id in requested:
            task_dir = run_dir / competition_id
            task_dir.mkdir(parents=True, exist_ok=False)
            logger.info("[%s] starting", competition_id)
            started = time.perf_counter()
            telemetry = GpuTelemetry()
            try:
                with telemetry:
                    result = RUNNERS[competition_id](args, task_dir, logger)
            except Exception as exc:
                result = {
                    "competition_id": competition_id,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "valid_submission": False,
                    "cv_score": None,
                    "proxy_score": None,
                    "mle_private_grader_score": None,
                    "kaggle_public_score": None,
                    "kaggle_private_score": None,
                    "medal_or_percentile": None,
                }
                logger.exception("[%s] failed", competition_id)
            result["runtime_seconds_total"] = time.perf_counter() - started
            result["gpu_telemetry"] = telemetry.summary()
            result["completed_at"] = utc_now()
            write_json(task_dir / "result.json", result)
            results.append(result)
            write_json(run_dir / "results_current.json", {"run_id": run_id, "results": results})
            logger.info("[%s] status=%s", competition_id, result["status"])

        passed = sum(result.get("status") == "passed" for result in results)
        summary = {
            "schema": "evomind.mlebench_lite.wave0_summary.v1",
            "run_id": run_id,
            "status": "passed" if passed == len(results) else "partial_failure",
            "competition_count": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "valid_submission_rate": (
                sum(bool(result.get("valid_submission")) for result in results) / len(results) if results else 0.0
            ),
            "results": results,
            "claim_boundary": "No Kaggle submission; CV/proxy/private-grader fields remain separate.",
            "completed_at": utc_now(),
        }
        write_json(run_dir / "summary.json", summary)
        manifest.update(
            {
                "status": summary["status"],
                "completed_at": utc_now(),
                "training_started": True,
                "summary_path": str(run_dir / "summary.json"),
            }
        )
        write_json(run_dir / "manifest.json", manifest)
        logger.info("Wave 0 complete: passed=%s failed=%s", passed, len(results) - passed)
        return 0 if passed == len(results) else 3
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    raise SystemExit(main())
