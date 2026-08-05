"""Frozen DeBERTa-v3-base plus high-capacity sparse Spooky OOF runner.

The production candidate is built only from the public train/test/sample files.
Every base learner is fitted inside one duplicate-safe outer fold.  The final
regularized logistic meta-model is cross-fitted: the model that scores outer
fold ``f`` is fitted only on OOF rows from the other folds.  No private answer,
official grader, or Kaggle submission path exists in this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scripts.run_spooky_transformer_oof import (
        CLASS_COLUMNS,
        EncodedTextDataset,
        TokenCache,
        _cuda_snapshot,
        _predict_transformer,
        _seed_everything,
        append_jsonl,
        build_duplicate_safe_folds,
        build_or_load_token_cache,
        multiclass_log_loss,
        normalize_probability,
        now_iso,
        sha256_file,
        write_json_atomic,
    )
    from scripts.spooky_leakage_free_meta import (
        assemble_meta_features,
        cross_fit_logistic_meta,
    )
    from scripts.spooky_style_features import extract_style_features
except ImportError:  # Direct ``python scripts/...`` execution.
    from run_spooky_transformer_oof import (  # type: ignore[no-redef]
        CLASS_COLUMNS,
        EncodedTextDataset,
        TokenCache,
        _cuda_snapshot,
        _predict_transformer,
        _seed_everything,
        append_jsonl,
        build_duplicate_safe_folds,
        build_or_load_token_cache,
        multiclass_log_loss,
        normalize_probability,
        now_iso,
        sha256_file,
        write_json_atomic,
    )
    from spooky_leakage_free_meta import (  # type: ignore[no-redef]
        assemble_meta_features,
        cross_fit_logistic_meta,
    )
    from spooky_style_features import extract_style_features  # type: ignore[no-redef]


DEFAULT_MODEL = "microsoft/deberta-v3-base"
DEFAULT_REVISION = "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"


def safe_unicode_ids(values: Sequence[Any]) -> np.ndarray:
    """Return a non-object Unicode array that is safe with ``allow_pickle=False``."""

    return np.asarray([str(value) for value in values], dtype=np.str_)


def index_sha256(indices: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def vocabulary_sha256(vocabulary: Mapping[str, int]) -> str:
    digest = hashlib.sha256()
    for token, index in sorted(vocabulary.items(), key=lambda item: (item[1], item[0])):
        digest.update(str(index).encode("ascii"))
        digest.update(b"\0")
        digest.update(token.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_source_contract(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Verify every source path frozen into the plan before computation starts."""

    source = plan.get("source")
    if not isinstance(source, Mapping) or not source:
        raise ValueError("Frozen Spooky v2 plan has no source contract")
    records: dict[str, dict[str, Any]] = {}
    failures: dict[str, Any] = {}
    for name, descriptor in source.items():
        if not isinstance(descriptor, Mapping):
            failures[str(name)] = "descriptor_not_mapping"
            continue
        path = Path(str(descriptor.get("path", ""))).resolve()
        frozen_hash = str(descriptor.get("sha256", "")).lower()
        exists = path.is_file()
        actual_hash = sha256_file(path).lower() if exists else None
        passed = bool(exists and frozen_hash and actual_hash == frozen_hash)
        records[str(name)] = {
            "path": str(path),
            "frozen_sha256": frozen_hash,
            "actual_sha256": actual_hash,
            "passed": passed,
        }
        if not passed:
            failures[str(name)] = records[str(name)]
    if failures:
        raise ValueError(f"Runtime source diverges from frozen plan: {failures}")
    return records


def _validate_probability(name: str, values: np.ndarray, rows: int) -> np.ndarray:
    result = normalize_probability(values)
    if result.shape != (rows, len(CLASS_COLUMNS)):
        raise RuntimeError(f"{name} has shape {result.shape}, expected {(rows, len(CLASS_COLUMNS))}")
    return result


def fit_sparse_style_fold(
    fit_text: Sequence[str],
    fit_style: np.ndarray,
    fit_labels: np.ndarray,
    valid_text: Sequence[str],
    valid_style: np.ndarray,
    test_text: Sequence[str],
    test_style: np.ndarray,
    *,
    word_features: int,
    char_features: int,
    c_value: float,
    max_iter: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit one fold-local joint word/character/style multinomial LR channel."""

    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    started = time.perf_counter()
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.995,
        max_features=word_features,
        strip_accents="unicode",
        sublinear_tf=True,
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 6),
        min_df=2,
        max_df=0.995,
        max_features=char_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    fit_word = word.fit_transform(fit_text)
    valid_word = word.transform(valid_text)
    test_word = word.transform(test_text)
    fit_char = char.fit_transform(fit_text)
    valid_char = char.transform(valid_text)
    test_char = char.transform(test_text)

    scaler = StandardScaler()
    fit_style_scaled = scaler.fit_transform(np.asarray(fit_style, dtype=np.float64))
    valid_style_scaled = scaler.transform(np.asarray(valid_style, dtype=np.float64))
    test_style_scaled = scaler.transform(np.asarray(test_style, dtype=np.float64))
    fit_matrix = hstack(
        [fit_word, fit_char, csr_matrix(fit_style_scaled.astype(np.float32))],
        format="csr",
        dtype=np.float32,
    )
    valid_matrix = hstack(
        [valid_word, valid_char, csr_matrix(valid_style_scaled.astype(np.float32))],
        format="csr",
        dtype=np.float32,
    )
    test_matrix = hstack(
        [test_word, test_char, csr_matrix(test_style_scaled.astype(np.float32))],
        format="csr",
        dtype=np.float32,
    )
    model = LogisticRegression(
        C=c_value,
        solver="lbfgs",
        max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(fit_matrix, np.asarray(fit_labels, dtype=np.int64))
    if not np.array_equal(model.classes_, np.arange(len(CLASS_COLUMNS))):
        raise RuntimeError("Sparse/style classifier class order changed")
    valid_probability = normalize_probability(model.predict_proba(valid_matrix))
    test_probability = normalize_probability(model.predict_proba(test_matrix))
    contract = {
        "schema": "evomind.spooky.high_capacity_sparse_style_fold.v2",
        "fit_scope": "outer_fit_rows_only",
        "model": "joint_multinomial_logistic_regression",
        "word": {
            "ngram_range": [1, 3],
            "min_df": 2,
            "max_df": 0.995,
            "max_features": word_features,
            "actual_features": int(fit_word.shape[1]),
            "vocabulary_sha256": vocabulary_sha256(word.vocabulary_),
        },
        "char": {
            "analyzer": "char",
            "ngram_range": [2, 6],
            "min_df": 2,
            "max_df": 0.995,
            "max_features": char_features,
            "actual_features": int(fit_char.shape[1]),
            "vocabulary_sha256": vocabulary_sha256(char.vocabulary_),
        },
        "style": {
            "feature_count": int(fit_style_scaled.shape[1]),
            "scaler_fit_scope": "outer_fit_rows_only",
            "scale_mean_sha256": hashlib.sha256(
                np.ascontiguousarray(scaler.mean_, dtype=np.float64).tobytes()
            ).hexdigest(),
            "scale_scale_sha256": hashlib.sha256(
                np.ascontiguousarray(scaler.scale_, dtype=np.float64).tobytes()
            ).hexdigest(),
        },
        "c_value": c_value,
        "max_iter": max_iter,
        "iterations": int(np.max(model.n_iter_)),
        "elapsed_seconds": time.perf_counter() - started,
        "private_labels_used": False,
    }
    return valid_probability, test_probability, contract


def _load_resumable_fold(
    prediction_path: Path,
    metadata_path: Path,
    *,
    plan_sha256: str,
    source_sha256: str,
    valid_indices: np.ndarray,
    test_rows: int,
    diagnostic: bool,
) -> dict[str, Any] | None:
    if not prediction_path.is_file() or not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    if not (
        metadata.get("status") == "passed"
        and metadata.get("plan_sha256") == plan_sha256
        and metadata.get("runner_sha256") == source_sha256
        and bool(metadata.get("diagnostic")) == diagnostic
        and metadata.get("prediction_sha256") == sha256_file(prediction_path)
    ):
        return None
    with np.load(prediction_path, allow_pickle=False) as archive:
        if not np.array_equal(np.asarray(archive["valid_indices"], dtype=np.int64), valid_indices):
            return None
        if not np.array_equal(
            np.asarray(archive["test_indices"], dtype=np.int64), np.arange(test_rows, dtype=np.int64)
        ):
            return None
        for name, rows in (
            ("transformer_valid", len(valid_indices)),
            ("sparse_valid", len(valid_indices)),
            ("transformer_test", test_rows),
            ("sparse_test", test_rows),
        ):
            _validate_probability(name, archive[name], rows)
    return metadata


def train_fold(
    *,
    fold: int,
    model_snapshot: Path,
    train_cache: TokenCache,
    test_cache: TokenCache,
    train_text: Sequence[str],
    test_text: Sequence[str],
    train_style: np.ndarray,
    test_style: np.ndarray,
    labels: np.ndarray,
    fit_indices: np.ndarray,
    valid_indices: np.ndarray,
    run_dir: Path,
    plan: Mapping[str, Any],
    plan_sha256: str,
    runner_sha256: str,
    smoke_train_rows: int,
    smoke_valid_rows: int,
    smoke_test_rows: int,
    max_train_updates: int,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        get_cosine_schedule_with_warmup,
    )

    training = plan["training"]
    sparse = plan["sparse_channel"]
    diagnostic = any((smoke_train_rows, smoke_valid_rows, smoke_test_rows, max_train_updates))
    fit_selected = np.asarray(fit_indices, dtype=np.int64)
    valid_selected = np.asarray(valid_indices, dtype=np.int64)
    test_selected = np.arange(len(test_text), dtype=np.int64)
    if smoke_train_rows:
        fit_selected = fit_selected[:smoke_train_rows]
    if smoke_valid_rows:
        valid_selected = valid_selected[:smoke_valid_rows]
    if smoke_test_rows:
        test_selected = test_selected[:smoke_test_rows]

    prediction_path = run_dir / f"fold_{fold}_predictions.npz"
    metadata_path = run_dir / f"fold_{fold}_result.json"
    resumable = _load_resumable_fold(
        prediction_path,
        metadata_path,
        plan_sha256=plan_sha256,
        source_sha256=runner_sha256,
        valid_indices=valid_selected,
        test_rows=len(test_selected),
        diagnostic=diagnostic,
    )
    if resumable is not None:
        return {**resumable, "resumed": True}

    fold_seed = int(training["seed"]) + fold * 1009
    _seed_everything(fold_seed)
    heartbeat_path = run_dir / "heartbeat.json"
    progress_path = run_dir / "progress.jsonl"
    generator = torch.Generator().manual_seed(fold_seed)
    train_loader = DataLoader(
        EncodedTextDataset(train_cache, fit_selected, labels),
        batch_size=int(training["train_batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(training["num_workers"]),
        pin_memory=True,
        persistent_workers=bool(training["num_workers"]),
    )
    valid_loader = DataLoader(
        EncodedTextDataset(train_cache, valid_selected, labels),
        batch_size=int(training["eval_batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        pin_memory=True,
        persistent_workers=bool(training["num_workers"]),
    )
    test_loader = DataLoader(
        EncodedTextDataset(test_cache, test_selected, None),
        batch_size=int(training["eval_batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        pin_memory=True,
        persistent_workers=bool(training["num_workers"]),
    )
    write_json_atomic(
        heartbeat_path,
        {
            "schema": "evomind.spooky.deberta_oof_heartbeat.v2",
            "created_at": now_iso(),
            "status": "running",
            "stage": "transformer_model_load",
            "fold": fold,
            "plan_sha256": plan_sha256,
            "process_signals_sent": 0,
        },
    )
    config = AutoConfig.from_pretrained(
        model_snapshot,
        local_files_only=True,
        num_labels=len(CLASS_COLUMNS),
        problem_type="single_label_classification",
        id2label={index: value for index, value in enumerate(CLASS_COLUMNS)},
        label2id={value: index for index, value in enumerate(CLASS_COLUMNS)},
    )
    dropout = float(plan["model"]["dropout"])
    for name in ("hidden_dropout_prob", "attention_probs_dropout_prob", "pooler_dropout"):
        if hasattr(config, name):
            setattr(config, name, dropout)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_snapshot,
        config=config,
        local_files_only=True,
    )
    if bool(training["gradient_checkpointing"]):
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    device = torch.device("cuda:0")
    model.to(device)
    use_bf16 = bool(training["mixed_precision"] == "bf16" and torch.cuda.is_bf16_supported())
    if training["mixed_precision"] == "bf16" and not use_bf16:
        raise RuntimeError("Frozen Spooky v2 plan requires BF16, but CUDA reports no BF16 support")

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    parameter_groups = [
        {
            "params": [
                value
                for name, value in model.named_parameters()
                if value.requires_grad and not any(term in name for term in no_decay)
            ],
            "weight_decay": float(training["weight_decay"]),
        },
        {
            "params": [
                value
                for name, value in model.named_parameters()
                if value.requires_grad and any(term in name for term in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(parameter_groups, lr=float(training["learning_rate"]))
    accumulation = int(training["gradient_accumulation_steps"])
    epochs = int(training["epochs_per_fold"])
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_updates = max(1, updates_per_epoch * epochs)
    if max_train_updates:
        total_updates = min(total_updates, max_train_updates)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        int(total_updates * float(training["warmup_ratio"])),
        total_updates,
    )
    optimizer.zero_grad(set_to_none=True)
    update = 0
    epoch_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch_index, batch in enumerate(train_loader):
            batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(**batch).loss / accumulation
            loss.backward()
            total_loss += float(loss.detach().cpu()) * accumulation
            batches += 1
            should_update = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader)
            if should_update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["max_grad_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                if update == 1 or update % 25 == 0 or update >= total_updates:
                    heartbeat = {
                        "schema": "evomind.spooky.deberta_oof_heartbeat.v2",
                        "created_at": now_iso(),
                        "status": "running",
                        "stage": "transformer_training",
                        "fold": fold,
                        "epoch": epoch + 1,
                        "update": update,
                        "total_updates": total_updates,
                        "progress_fraction": update / total_updates,
                        "mean_train_loss": total_loss / max(batches, 1),
                        "elapsed_seconds": time.perf_counter() - started,
                        "plan_sha256": plan_sha256,
                        "process_signals_sent": 0,
                    }
                    write_json_atomic(heartbeat_path, heartbeat)
                    append_jsonl(progress_path, heartbeat)
                if update >= total_updates:
                    break
        valid_at_epoch = _predict_transformer(model, valid_loader, device, use_bf16=use_bf16)
        epoch_records.append(
            {
                "epoch": epoch + 1,
                "updates": update,
                "mean_train_loss": total_loss / max(batches, 1),
                "validation_log_loss_observed_not_selected": multiclass_log_loss(
                    labels[valid_selected], valid_at_epoch
                ),
                "checkpoint_selection_used_outer_fold": False,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        if update >= total_updates:
            break
    transformer_valid = _predict_transformer(model, valid_loader, device, use_bf16=use_bf16)
    transformer_test = _predict_transformer(model, test_loader, device, use_bf16=use_bf16)
    model_artifact: dict[str, Any] | None = None
    if bool(plan["execution"].get("save_fold_models", False)):
        model_dir = run_dir / f"fold_{fold}_deberta_model"
        model.save_pretrained(model_dir, safe_serialization=True)
        files = sorted(path for path in model_dir.rglob("*") if path.is_file())
        model_artifact = {
            "path": str(model_dir),
            "files": [{"name": path.name, "sha256": sha256_file(path)} for path in files],
        }
    del model
    torch.cuda.empty_cache()

    write_json_atomic(
        heartbeat_path,
        {
            "schema": "evomind.spooky.deberta_oof_heartbeat.v2",
            "created_at": now_iso(),
            "status": "running",
            "stage": "fold_local_sparse_style",
            "fold": fold,
            "plan_sha256": plan_sha256,
            "process_signals_sent": 0,
        },
    )
    sparse_valid, sparse_test, sparse_contract = fit_sparse_style_fold(
        [train_text[index] for index in fit_selected],
        train_style[fit_selected],
        labels[fit_selected],
        [train_text[index] for index in valid_selected],
        train_style[valid_selected],
        [test_text[index] for index in test_selected],
        test_style[test_selected],
        word_features=int(sparse["word_max_features"]),
        char_features=int(sparse["char_max_features"]),
        c_value=float(sparse["c_value"]),
        max_iter=int(sparse["max_iter"]),
        random_state=fold_seed + 7000,
    )
    temporary = prediction_path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        valid_indices=valid_selected,
        test_indices=test_selected,
        transformer_valid=transformer_valid,
        transformer_test=transformer_test,
        sparse_valid=sparse_valid,
        sparse_test=sparse_test,
    )
    temporary.replace(prediction_path)
    metadata = {
        "schema": "evomind.spooky.deberta_sparse_style_fold.v2",
        "created_at": now_iso(),
        "status": "passed",
        "fold": fold,
        "seed": fold_seed,
        "diagnostic": diagnostic,
        "plan_sha256": plan_sha256,
        "runner_sha256": runner_sha256,
        "fit_rows": len(fit_selected),
        "valid_rows": len(valid_selected),
        "test_rows": len(test_selected),
        "fit_index_sha256": index_sha256(fit_selected),
        "valid_index_sha256": index_sha256(valid_selected),
        "fit_valid_disjoint": not bool(np.intersect1d(fit_selected, valid_selected).size),
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "transformer_epochs": epoch_records,
        "transformer_model": model_artifact,
        "sparse_style": sparse_contract,
        "component_log_loss": {
            "transformer": multiclass_log_loss(labels[valid_selected], transformer_valid),
            "sparse": multiclass_log_loss(labels[valid_selected], sparse_valid),
        },
        "checkpoint_budget": "fixed_before_outer_fold_training",
        "checkpoint_selection_used_outer_fold": False,
        "mixed_precision": "bf16" if use_bf16 else "fp32",
        "cuda_after": _cuda_snapshot(),
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
    }
    write_json_atomic(metadata_path, metadata)
    return metadata


def build_meta_inputs(
    component_oof: Mapping[str, np.ndarray],
    component_test_by_fold: Mapping[str, np.ndarray],
    train_style: np.ndarray,
    test_style: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    oof_features, contract = assemble_meta_features(
        component_oof,
        style_features=train_style,
    )
    folds = next(iter(component_test_by_fold.values())).shape[0]
    fold_features: list[np.ndarray] = []
    for fold in range(folds):
        values, fold_contract = assemble_meta_features(
            {name: matrix[fold] for name, matrix in component_test_by_fold.items()},
            style_features=test_style,
        )
        if fold_contract["blocks"] != contract["blocks"] or values.shape[1] != oof_features.shape[1]:
            raise RuntimeError("Spooky train/test meta feature contracts diverged")
        fold_features.append(values)
    return oof_features, np.stack(fold_features, axis=0), contract


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--frozen-plan", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--smoke-train-rows", type=int, default=0)
    parser.add_argument("--smoke-valid-rows", type=int, default=0)
    parser.add_argument("--smoke-test-rows", type=int, default=0)
    parser.add_argument("--max-train-updates-per-fold", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen Spooky DeBERTa v2 run")
    plan_path = args.frozen_plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    if plan.get("status") != "frozen_before_training":
        raise ValueError("Spooky DeBERTa v2 plan is not frozen")
    if plan.get("schema") != "evomind.spooky.deberta_oof_frozen_plan.v2":
        raise ValueError("Spooky DeBERTa v2 plan schema is not exact")
    source_records = verify_source_contract(plan)
    runner_sha256 = source_records["runner"]["actual_sha256"]
    plan_sha256 = sha256_file(plan_path)
    training = plan["training"]
    model_spec = plan["model"]
    execution = plan["execution"]
    if model_spec["repo_id"] != DEFAULT_MODEL or model_spec["revision"] != DEFAULT_REVISION:
        raise ValueError("Frozen Spooky v2 model identity is not the reviewed DeBERTa snapshot")
    if args.hf_cache.resolve() != Path(execution["hf_cache"]).resolve():
        raise ValueError("Runtime HuggingFace cache diverges from frozen plan")
    diagnostic = any(
        (
            args.smoke_train_rows,
            args.smoke_valid_rows,
            args.smoke_test_rows,
            args.max_train_updates_per_fold,
        )
    )
    requested = tuple(int(value.strip()) for value in args.folds.split(",") if value.strip())
    frozen_folds = int(training["folds"])
    if not requested or any(value not in range(frozen_folds) for value in requested):
        raise ValueError("Requested Spooky v2 fold is outside the frozen range")
    if diagnostic:
        frozen_diagnostic = plan.get("diagnostic_smoke")
        if not isinstance(frozen_diagnostic, Mapping):
            raise ValueError("Diagnostic arguments are absent from the frozen plan")
        expected_diagnostic = {
            "run_id": args.run_id,
            "folds": list(requested),
            "smoke_train_rows": args.smoke_train_rows,
            "smoke_valid_rows": args.smoke_valid_rows,
            "smoke_test_rows": args.smoke_test_rows,
            "max_train_updates_per_fold": args.max_train_updates_per_fold,
        }
        if expected_diagnostic != frozen_diagnostic:
            raise ValueError("Runtime diagnostic arguments diverge from frozen plan")
    else:
        expected_runtime = {
            "run_id": execution["run_id"],
            "public_dir": Path(execution["public_dir"]).resolve(),
            "output_root": Path(execution["output_root"]).resolve(),
        }
        actual_runtime = {
            "run_id": args.run_id,
            "public_dir": args.public_dir.resolve(),
            "output_root": args.output_root.resolve(),
        }
        if actual_runtime != expected_runtime:
            raise ValueError(
                f"Runtime paths or run ID diverge from frozen plan: "
                f"{{'frozen': {expected_runtime}, 'actual': {actual_runtime}}}"
            )
        if requested != tuple(range(frozen_folds)):
            raise ValueError("Production execution must cover every frozen fold in order")

    torch.backends.cuda.matmul.allow_tf32 = bool(training["tf32"])
    torch.backends.cudnn.allow_tf32 = bool(training["tf32"])
    torch.set_float32_matmul_precision("high")
    run_dir = (args.output_root / args.run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    public_dir = args.public_dir.resolve()
    paths = {
        "public_train_sha256": public_dir / "train.csv",
        "public_test_sha256": public_dir / "test.csv",
        "public_sample_submission_sha256": public_dir / "sample_submission.csv",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    actual_input_hashes = {name: sha256_file(path) for name, path in paths.items()}
    mismatch = {
        name: {"frozen": plan["inputs"].get(name), "actual": value}
        for name, value in actual_input_hashes.items()
        if plan["inputs"].get(name) != value
    }
    if mismatch:
        raise ValueError(f"Runtime public inputs diverge from frozen plan: {mismatch}")
    train = pd.read_csv(paths["public_train_sha256"])
    test = pd.read_csv(paths["public_test_sha256"])
    sample = pd.read_csv(paths["public_sample_submission_sha256"])
    if list(train.columns) != ["id", "text", "author"]:
        raise ValueError("Spooky public train schema is not exact")
    if list(test.columns) != ["id", "text"] or list(sample.columns) != ["id", *CLASS_COLUMNS]:
        raise ValueError("Spooky public test/sample schema is not exact")
    if sample["id"].astype(str).tolist() != test["id"].astype(str).tolist():
        raise ValueError("Spooky sample submission IDs do not match public test order")
    if len(train) != int(plan["inputs"]["train_rows"]) or len(test) != int(
        plan["inputs"]["test_rows"]
    ):
        raise ValueError("Spooky public row counts diverge from frozen plan")
    if list(plan["inputs"]["class_order"]) != list(CLASS_COLUMNS):
        raise ValueError("Spooky frozen class order diverges from runner contract")
    class_to_index = {value: index for index, value in enumerate(CLASS_COLUMNS)}
    mapped = train["author"].map(class_to_index)
    if mapped.isna().any():
        raise ValueError("Spooky public train includes an unexpected author label")
    labels = mapped.to_numpy(dtype=np.int64)
    train_text = train["text"].fillna("").astype(str).tolist()
    test_text = test["text"].fillna("").astype(str).tolist()
    splits, fold_assignment, duplicate_report = build_duplicate_safe_folds(
        train_text,
        labels,
        folds=frozen_folds,
        seed=int(training["seed"]),
    )
    if duplicate_report != plan["inputs"]["duplicate_group_report"]:
        raise ValueError("Runtime duplicate-safe fold contract diverges from frozen plan")
    train_style, train_style_contract = extract_style_features(train_text)
    test_style, test_style_contract = extract_style_features(test_text)
    if train_style_contract["feature_names"] != test_style_contract["feature_names"]:
        raise RuntimeError("Spooky train/test style feature contracts diverged")
    write_json_atomic(run_dir / "duplicate_groups.json", duplicate_report)
    fold_path = run_dir / "fold_assignments.npz"
    np.savez_compressed(
        fold_path,
        train_id=safe_unicode_ids(train["id"].astype(str).tolist()),
        truth=labels,
        fold=fold_assignment,
    )

    model_snapshot = Path(
        snapshot_download(
            repo_id=model_spec["repo_id"],
            revision=model_spec["revision"],
            cache_dir=args.hf_cache.resolve(),
            local_files_only=True,
        )
    ).resolve()
    # The slow SentencePiece tokenizer preserves DeBERTa-v3 byte fallback; the
    # converted fast tokenizer explicitly omits that behavior.
    tokenizer = AutoTokenizer.from_pretrained(model_snapshot, local_files_only=True, use_fast=False)
    train_cache = build_or_load_token_cache(
        train_text,
        tokenizer,
        cache_path=run_dir / "token_cache" / f"train_l{training['max_length']}.npz",
        max_length=int(training["max_length"]),
        source_sha256=actual_input_hashes["public_train_sha256"],
        model_revision=model_spec["revision"],
    )
    test_cache = build_or_load_token_cache(
        test_text,
        tokenizer,
        cache_path=run_dir / "token_cache" / f"test_l{training['max_length']}.npz",
        max_length=int(training["max_length"]),
        source_sha256=actual_input_hashes["public_test_sha256"],
        model_revision=model_spec["revision"],
    )
    manifest = {
        "schema": "evomind.spooky.deberta_oof_run.v2",
        "created_at": now_iso(),
        "run_id": args.run_id,
        "status": "training",
        "diagnostic": diagnostic,
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha256,
        "source_contract": source_records,
        "public_inputs": {
            name: {"path": str(paths[name]), "sha256": digest}
            for name, digest in actual_input_hashes.items()
        },
        "model": {**model_spec, "snapshot": str(model_snapshot)},
        "fold_assignment": {"path": str(fold_path), "sha256": sha256_file(fold_path)},
        "duplicate_group_isolation": duplicate_report,
        "style_feature_contract": train_style_contract,
        "requested_folds": list(requested),
        "cuda_before": _cuda_snapshot(),
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "human_gate_preserved": True,
        "process_signals_sent": 0,
    }
    write_json_atomic(run_dir / "manifest.json", manifest)
    fold_results = []
    for fold in requested:
        fit_indices, valid_indices = splits[fold]
        fold_results.append(
            train_fold(
                fold=fold,
                model_snapshot=model_snapshot,
                train_cache=train_cache,
                test_cache=test_cache,
                train_text=train_text,
                test_text=test_text,
                train_style=train_style,
                test_style=test_style,
                labels=labels,
                fit_indices=fit_indices,
                valid_indices=valid_indices,
                run_dir=run_dir,
                plan=plan,
                plan_sha256=plan_sha256,
                runner_sha256=runner_sha256,
                smoke_train_rows=args.smoke_train_rows,
                smoke_valid_rows=args.smoke_valid_rows,
                smoke_test_rows=args.smoke_test_rows,
                max_train_updates=args.max_train_updates_per_fold,
            )
        )

    summary: dict[str, Any] = {
        **manifest,
        "created_at": now_iso(),
        "status": "diagnostic_complete" if diagnostic else "partial",
        "fold_results": fold_results,
    }
    complete = not diagnostic and requested == tuple(range(frozen_folds))
    if complete:
        component_oof = {
            name: np.full((len(train), len(CLASS_COLUMNS)), np.nan, dtype=np.float64)
            for name in ("transformer", "sparse")
        }
        component_test = {
            name: np.zeros((frozen_folds, len(test), len(CLASS_COLUMNS)), dtype=np.float64)
            for name in component_oof
        }
        component_counts = np.zeros(len(train), dtype=np.uint8)
        for fold in range(frozen_folds):
            with np.load(run_dir / f"fold_{fold}_predictions.npz", allow_pickle=False) as archive:
                valid_indices = np.asarray(archive["valid_indices"], dtype=np.int64)
                for name in component_oof:
                    component_oof[name][valid_indices] = _validate_probability(
                        f"{name}_valid", archive[f"{name}_valid"], len(valid_indices)
                    )
                    component_test[name][fold] = _validate_probability(
                        f"{name}_test", archive[f"{name}_test"], len(test)
                    )
                component_counts[valid_indices] += 1
        if not np.all(component_counts == 1) or any(
            not np.isfinite(values).all() for values in component_oof.values()
        ):
            raise RuntimeError("Spooky v2 component OOF assembly failed exact-once coverage")
        meta_oof, meta_test_by_fold, meta_feature_contract = build_meta_inputs(
            component_oof,
            component_test,
            train_style,
            test_style,
        )
        candidate_oof, candidate_test, candidate_counts, meta_contract = cross_fit_logistic_meta(
            meta_oof,
            meta_test_by_fold,
            labels,
            fold_assignment,
            c_value=float(plan["meta"]["c_value"]),
            max_iter=int(plan["meta"]["max_iter"]),
            random_state=int(training["seed"]) + 9000,
        )
        component_scores = {
            name: multiclass_log_loss(labels, values) for name, values in component_oof.items()
        }
        candidate_score = multiclass_log_loss(labels, candidate_oof)
        threshold = float(plan["promotion_gate"]["single_seed_log_loss_threshold"])
        single_seed_passed = candidate_score <= threshold
        submission = sample.copy()
        submission.loc[:, list(CLASS_COLUMNS)] = candidate_test
        submission_path = run_dir / "candidate_submission_withheld.csv"
        submission.to_csv(submission_path, index=False)
        bundle_path = run_dir / "spooky_deberta_oof_v2.npz"
        np.savez_compressed(
            bundle_path,
            truth=labels,
            fold=fold_assignment,
            transformer_oof=component_oof["transformer"],
            sparse_oof=component_oof["sparse"],
            transformer_test_by_fold=component_test["transformer"],
            sparse_test_by_fold=component_test["sparse"],
            candidate_oof=candidate_oof,
            candidate_test=candidate_test,
            component_write_counts=component_counts,
            candidate_write_counts=candidate_counts,
            train_id=safe_unicode_ids(train["id"].astype(str).tolist()),
            test_id=safe_unicode_ids(test["id"].astype(str).tolist()),
        )
        promotion = {
            "schema": "evomind.spooky.deberta_single_seed_gate.v2",
            "seed": int(training["seed"]),
            "candidate_oof_log_loss": candidate_score,
            "single_seed_threshold": threshold,
            "single_seed_passed": single_seed_passed,
            "three_seed_confirmation_required": True,
            "three_seed_confirmation_complete": False,
            "promotion_allowed": False,
            "checks": {
                "exact_once_component_oof": bool(np.all(component_counts == 1)),
                "exact_once_candidate_oof": bool(np.all(candidate_counts == 1)),
                "all_rows_finite_and_normalized": bool(
                    np.isfinite(candidate_oof).all()
                    and np.allclose(candidate_oof.sum(axis=1), 1.0, atol=1e-7)
                ),
                "duplicate_group_isolation": bool(duplicate_report["group_isolation"]),
                "fixed_neural_budgets": all(
                    not result["checkpoint_selection_used_outer_fold"] for result in fold_results
                ),
                "cross_fit_meta": bool(meta_contract["exact_once_oof"]),
                "private_labels_unused": True,
                "official_grader_not_executed": True,
                "kaggle_submission_not_executed": True,
                "process_signals_sent": 0,
            },
        }
        summary.update(
            {
                "status": (
                    "single_seed_gate_passed_confirmation_pending"
                    if single_seed_passed
                    else "single_seed_gate_failed"
                ),
                "component_oof_log_loss": component_scores,
                "candidate_oof_log_loss": candidate_score,
                "meta_feature_contract": meta_feature_contract,
                "meta_contract": meta_contract,
                "promotion_gate": promotion,
                "prediction_bundle": {"path": str(bundle_path), "sha256": sha256_file(bundle_path)},
                "submission_withheld": {
                    "path": str(submission_path),
                    "sha256": sha256_file(submission_path),
                },
            }
        )
    write_json_atomic(run_dir / "summary.json", summary)
    write_json_atomic(
        run_dir / "heartbeat.json",
        {
            "schema": "evomind.spooky.deberta_oof_heartbeat.v2",
            "created_at": now_iso(),
            "status": summary["status"],
            "stage": "terminal",
            "run_id": args.run_id,
            "diagnostic": diagnostic,
            "plan_sha256": plan_sha256,
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "process_signals_sent": 0,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
