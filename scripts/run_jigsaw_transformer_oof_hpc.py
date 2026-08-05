"""Leakage-controlled Jigsaw transformer OOF training for an isolated HPC GPU.

The runner deliberately consumes only the prepared public train/test/sample files and
an existing public-fold sparse prediction bundle.  It never opens the MLE-Bench private
answers and it never invokes either the official grader or Kaggle submission APIs.

The production candidate is a fold-clean rank blend.  For each held-out outer fold and
target, its transformer weight is selected only on the other folds.  The same weight is
then applied to the held-out OOF rows and to the matching transformer test prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def make_multilabel_stratified_folds(
    targets: np.ndarray,
    *,
    requested_folds: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Dependency-free iterative multilabel fold assignment used by the frozen runner."""

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
    desired_label = np.tile(
        matrix.sum(axis=0, dtype=np.float64) / fold_count, (fold_count, 1)
    )

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

TARGET_COLUMNS = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)
DEFAULT_MODEL = "distilbert/distilroberta-base"
DEFAULT_REVISION = "fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b"
DEFAULT_BLEND_GRID = (0.25, 0.35, 0.45, 0.55, 0.65, 0.75)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def fractional_rank(values: np.ndarray) -> np.ndarray:
    """Return deterministic average ranks scaled to the open interval (0, 1]."""

    from scipy.stats import rankdata

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError("Rank input contains a non-finite value")
    return rankdata(array, method="average") / max(1, len(array))


def fold_rank_matrix(values: np.ndarray, fold_assignment: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    folds = np.asarray(fold_assignment, dtype=np.int16).reshape(-1)
    if matrix.ndim != 2 or len(matrix) != len(folds):
        raise ValueError("Fold-ranked values and fold assignment do not share rows")
    ranked = np.full_like(matrix, np.nan, dtype=np.float64)
    for fold in sorted(int(value) for value in np.unique(folds)):
        mask = folds == fold
        for label in range(matrix.shape[1]):
            ranked[mask, label] = fractional_rank(matrix[mask, label])
    if not np.isfinite(ranked).all():
        raise RuntimeError("Fold ranking is incomplete")
    return ranked


def mean_columnwise_auc(truth: np.ndarray, prediction: np.ndarray) -> float:
    labels = np.asarray(truth, dtype=np.int8)
    scores = np.asarray(prediction, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 2:
        raise ValueError("Jigsaw truth and prediction matrices do not share a contract")
    return float(
        np.mean(
            [roc_auc_score(labels[:, label], scores[:, label]) for label in range(labels.shape[1])]
        )
    )


def _select_blend_weight(
    truth: np.ndarray,
    sparse_rank: np.ndarray,
    transformer_rank: np.ndarray,
    grid: Sequence[float],
) -> tuple[float, list[dict[str, float]]]:
    records = []
    for value in grid:
        weight = float(value)
        prediction = (1.0 - weight) * sparse_rank + weight * transformer_rank
        records.append({"transformer_weight": weight, "auc": float(roc_auc_score(truth, prediction))})
    selected = max(
        records,
        key=lambda record: (
            record["auc"],
            -abs(record["transformer_weight"] - 0.5),
            -record["transformer_weight"],
        ),
    )
    return float(selected["transformer_weight"]), records


def cross_fit_sparse_transformer_rank_blend(
    sparse_oof: np.ndarray,
    transformer_oof: np.ndarray,
    sparse_test: np.ndarray,
    transformer_test_by_fold: np.ndarray,
    truth: np.ndarray,
    fold_assignment: np.ndarray,
    *,
    weight_grid: Iterable[float] = DEFAULT_BLEND_GRID,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build exact-once OOF predictions with outer-fold-isolated weight selection."""

    sparse = np.asarray(sparse_oof, dtype=np.float64)
    transformer = np.asarray(transformer_oof, dtype=np.float64)
    sparse_test_matrix = np.asarray(sparse_test, dtype=np.float64)
    transformer_test = np.asarray(transformer_test_by_fold, dtype=np.float64)
    labels = np.asarray(truth, dtype=np.int8)
    folds = np.asarray(fold_assignment, dtype=np.int16).reshape(-1)
    grid = tuple(sorted({float(value) for value in weight_grid}))
    unique_folds = sorted(int(value) for value in np.unique(folds))
    if sparse.shape != transformer.shape or sparse.shape != labels.shape:
        raise ValueError("Sparse, transformer, and truth OOF matrices must match")
    if sparse_test_matrix.ndim != 2 or sparse_test_matrix.shape[1] != labels.shape[1]:
        raise ValueError("Sparse test predictions do not match target columns")
    if transformer_test.shape != (
        len(unique_folds),
        len(sparse_test_matrix),
        labels.shape[1],
    ):
        raise ValueError("Transformer test predictions require one matching source per fold")
    if unique_folds != list(range(len(unique_folds))) or len(unique_folds) < 3:
        raise ValueError("Blend requires contiguous zero-based folds")
    if not grid or any(not 0.0 <= value <= 1.0 for value in grid):
        raise ValueError("Blend weights must be a non-empty grid inside [0, 1]")
    if any(not np.isfinite(value).all() for value in (sparse, transformer, sparse_test_matrix, transformer_test)):
        raise ValueError("Blend inputs must be finite")

    sparse_rank = fold_rank_matrix(sparse, folds)
    transformer_rank = fold_rank_matrix(transformer, folds)
    sparse_test_rank = np.column_stack(
        [fractional_rank(sparse_test_matrix[:, label]) for label in range(labels.shape[1])]
    )
    transformer_test_rank = np.empty_like(transformer_test, dtype=np.float64)
    for fold in unique_folds:
        transformer_test_rank[fold] = np.column_stack(
            [fractional_rank(transformer_test[fold, :, label]) for label in range(labels.shape[1])]
        )

    oof = np.full_like(sparse, np.nan, dtype=np.float64)
    write_counts = np.zeros_like(labels, dtype=np.uint8)
    test_components = np.zeros_like(transformer_test, dtype=np.float64)
    records: list[dict[str, Any]] = []
    for fold in unique_folds:
        fit_mask = folds != fold
        score_mask = folds == fold
        for label, target in enumerate(TARGET_COLUMNS):
            weight, candidates = _select_blend_weight(
                labels[fit_mask, label],
                sparse_rank[fit_mask, label],
                transformer_rank[fit_mask, label],
                grid,
            )
            oof[score_mask, label] = (
                (1.0 - weight) * sparse_rank[score_mask, label]
                + weight * transformer_rank[score_mask, label]
            )
            write_counts[score_mask, label] += 1
            test_components[fold, :, label] = (
                (1.0 - weight) * sparse_test_rank[:, label]
                + weight * transformer_test_rank[fold, :, label]
            )
            records.append(
                {
                    "score_fold": fold,
                    "fit_folds": [value for value in unique_folds if value != fold],
                    "target": target,
                    "target_index": label,
                    "selected_transformer_weight": weight,
                    "candidate_scores": candidates,
                }
            )
    if not np.all(write_counts == 1) or not np.isfinite(oof).all():
        raise RuntimeError("Transformer blend OOF predictions were not written exactly once")
    averaged_test = np.mean(test_components, axis=0)
    ranked_test = np.column_stack(
        [fractional_rank(averaged_test[:, label]) for label in range(averaged_test.shape[1])]
    )
    contract = {
        "schema": "evomind.mlebench_lite.jigsaw_transformer_blend.v1",
        "weight_definition": "transformer_fraction",
        "weight_grid": list(grid),
        "selection": "non_held_outer_folds_only_per_label",
        "outer_folds": unique_folds,
        "records": records,
        "exact_once_oof": True,
        "test_aggregation": "matching_transformer_fold_plus_public_sparse_test_then_mean_and_rank",
        "private_labels_used": False,
    }
    return oof, ranked_test, write_counts, contract


def load_sparse_bundle(path: Path, *, row_count: int, test_count: int) -> dict[str, np.ndarray]:
    required = {
        "truth",
        "fold",
        "oof_stacker",
        "test_stacker",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"Sparse bundle is missing arrays: {missing}")
        bundle = {name: np.asarray(archive[name]) for name in required}
    if bundle["truth"].shape != (row_count, len(TARGET_COLUMNS)):
        raise ValueError("Sparse truth shape does not match public train rows")
    if bundle["oof_stacker"].shape != bundle["truth"].shape:
        raise ValueError("Sparse stacker OOF shape is invalid")
    if bundle["test_stacker"].shape != (test_count, len(TARGET_COLUMNS)):
        raise ValueError("Sparse stacker test shape is invalid")
    if bundle["fold"].shape != (row_count,):
        raise ValueError("Sparse fold assignment shape is invalid")
    return bundle


@dataclass(frozen=True)
class TokenCache:
    input_ids: np.ndarray
    attention_mask: np.ndarray


def build_or_load_token_cache(
    texts: Sequence[str],
    tokenizer: Any,
    *,
    cache_path: Path,
    max_length: int,
    source_sha256: str,
    model_revision: str,
) -> TokenCache:
    manifest_path = cache_path.with_suffix(".manifest.json")
    expected = {
        "schema": "evomind.jigsaw.transformer_token_cache.v1",
        "rows": len(texts),
        "max_length": int(max_length),
        "source_sha256": source_sha256,
        "model_revision": model_revision,
    }
    if cache_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if (
            all(manifest.get(key) == value for key, value in expected.items())
            and manifest.get("cache_sha256") == sha256_file(cache_path)
        ):
            with np.load(cache_path, allow_pickle=False) as archive:
                input_ids = np.asarray(archive["input_ids"], dtype=np.int32)
                attention_mask = np.asarray(archive["attention_mask"], dtype=np.uint8)
            if input_ids.shape == attention_mask.shape == (len(texts), max_length):
                return TokenCache(input_ids=input_ids, attention_mask=attention_mask)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    input_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    for start in range(0, len(texts), 4096):
        encoded = tokenizer(
            list(texts[start : start + 4096]),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="np",
        )
        input_parts.append(np.asarray(encoded["input_ids"], dtype=np.int32))
        mask_parts.append(np.asarray(encoded["attention_mask"], dtype=np.uint8))
    input_ids = np.concatenate(input_parts, axis=0)
    attention_mask = np.concatenate(mask_parts, axis=0)
    temporary = cache_path.with_suffix(cache_path.suffix + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, input_ids=input_ids, attention_mask=attention_mask)
    temporary.replace(cache_path)
    manifest = {**expected, "created_at": now_iso(), "cache_sha256": sha256_file(cache_path)}
    write_json_atomic(manifest_path, manifest)
    return TokenCache(input_ids=input_ids, attention_mask=attention_mask)


class EncodedDataset:
    def __init__(
        self,
        cache: TokenCache,
        indices: np.ndarray,
        labels: np.ndarray | None,
    ) -> None:
        self.cache = cache
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        import torch

        row = int(self.indices[item])
        output = {
            "input_ids": torch.as_tensor(self.cache.input_ids[row], dtype=torch.long),
            "attention_mask": torch.as_tensor(self.cache.attention_mask[row], dtype=torch.long),
        }
        if self.labels is not None:
            output["labels"] = torch.as_tensor(self.labels[row], dtype=torch.float32)
        return output


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_unicode_ids(values: Sequence[Any]) -> np.ndarray:
    """Store public row IDs without object dtype or pickle requirements."""

    strings = [str(value) for value in values]
    width = max(1, max((len(value) for value in strings), default=1))
    return np.asarray(strings, dtype=f"<U{width}")


def resolve_seed_contract(
    args: argparse.Namespace, plan: dict[str, Any]
) -> tuple[int, int]:
    """Resolve immutable outer folds separately from model randomness.

    Legacy plans use ``training.seed`` for both values.  Confirmation plans
    freeze ``fold_seed`` at 42 so they remain aligned with the existing sparse
    OOF bundle while varying only ``model_seed`` across 40, 41, and 42.
    """

    training = plan.get("training") or {}
    legacy_seed = training.get("seed")
    frozen_fold_seed = training.get("fold_seed", legacy_seed)
    frozen_model_seed = training.get("model_seed", legacy_seed)
    if frozen_fold_seed is None or frozen_model_seed is None:
        raise ValueError("Frozen plan does not define a complete seed contract")
    actual_fold_seed = args.fold_seed if args.fold_seed is not None else args.seed
    actual_model_seed = args.model_seed if args.model_seed is not None else args.seed
    mismatches = {}
    if int(actual_fold_seed) != int(frozen_fold_seed):
        mismatches["fold_seed"] = {
            "frozen": int(frozen_fold_seed),
            "actual": int(actual_fold_seed),
        }
    if int(actual_model_seed) != int(frozen_model_seed):
        mismatches["model_seed"] = {
            "frozen": int(frozen_model_seed),
            "actual": int(actual_model_seed),
        }
    if mismatches:
        raise ValueError(f"Runtime arguments diverge from the frozen plan: {mismatches}")
    args.fold_seed = int(actual_fold_seed)
    args.model_seed = int(actual_model_seed)
    return args.fold_seed, args.model_seed


def resolve_token_cache_dir(
    args: argparse.Namespace, plan: dict[str, Any], run_dir: Path
) -> Path:
    """Resolve a hash-validated shared token cache for confirmation runs."""

    frozen = (plan.get("execution") or {}).get("token_cache_dir")
    requested = args.token_cache_dir
    if frozen:
        frozen_path = Path(str(frozen)).resolve()
        if requested is not None and requested.resolve() != frozen_path:
            raise ValueError("Runtime token cache differs from the frozen plan")
        return frozen_path
    if requested is not None:
        raise ValueError("Runtime token cache is not bound by the frozen plan")
    return (run_dir / "token_cache").resolve()


def _cuda_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
    return {
        "exit_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr_present": bool(completed.stderr.strip()),
    }


def _predict(model: Any, loader: Any, device: Any, *, use_bf16: bool) -> np.ndarray:
    import torch

    model.eval()
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items() if key != "labels"}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(**inputs).logits
            predictions.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(predictions, axis=0)


def train_fold(
    *,
    fold: int,
    model_snapshot: Path,
    train_cache: TokenCache,
    test_cache: TokenCache,
    truth: np.ndarray,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
    plan_sha256: str,
    diagnostic: bool,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup

    prediction_path = output_dir / f"fold_{fold}_predictions.npz"
    metadata_path = output_dir / f"fold_{fold}_result.json"
    heartbeat_path = output_dir / "heartbeat.json"
    progress_path = output_dir / "progress.jsonl"
    if prediction_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        if (
            metadata.get("status") == "passed"
            and metadata.get("plan_sha256") == plan_sha256
            and bool(metadata.get("diagnostic")) == diagnostic
        ):
            with np.load(prediction_path, allow_pickle=False) as archive:
                if np.array_equal(archive["valid_indices"], valid_indices):
                    return metadata

    fold_seed = int(args.model_seed + fold * 1009)
    _seed_everything(fold_seed)
    train_selected = np.asarray(train_indices, dtype=np.int64)
    valid_selected = np.asarray(valid_indices, dtype=np.int64)
    test_selected = np.arange(len(test_cache.input_ids), dtype=np.int64)
    if args.smoke_train_rows:
        train_selected = train_selected[: int(args.smoke_train_rows)]
    if args.smoke_valid_rows:
        valid_selected = valid_selected[: int(args.smoke_valid_rows)]
    if args.smoke_test_rows:
        test_selected = test_selected[: int(args.smoke_test_rows)]

    train_dataset = EncodedDataset(train_cache, train_selected, truth)
    valid_dataset = EncodedDataset(train_cache, valid_selected, truth)
    test_dataset = EncodedDataset(test_cache, test_selected, None)
    write_json_atomic(
        heartbeat_path,
        {
            "schema": "evomind.jigsaw.transformer_heartbeat.v1",
            "created_at": now_iso(),
            "status": "running",
            "stage": "model_load",
            "fold": fold,
            "diagnostic": diagnostic,
            "plan_sha256": plan_sha256,
            "private_labels_used": False,
            "official_grader_executed": False,
        },
    )
    generator = torch.Generator().manual_seed(fold_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=bool(args.num_workers),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=bool(args.num_workers),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=bool(args.num_workers),
    )

    config = AutoConfig.from_pretrained(
        model_snapshot,
        local_files_only=True,
        num_labels=len(TARGET_COLUMNS),
        problem_type="multi_label_classification",
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_snapshot,
        config=config,
        local_files_only=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    device = torch.device("cuda:0")
    model.to(device)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(x in n for x in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(x in n for x in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(groups, lr=args.learning_rate)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_updates = max(1, updates_per_epoch * args.epochs)
    if args.max_train_updates_per_fold:
        total_updates = min(total_updates, int(args.max_train_updates_per_fold))
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_updates)
    use_bf16 = bool(args.bf16 and torch.cuda.is_bf16_supported())
    optimizer.zero_grad(set_to_none=True)
    update = 0
    epoch_records = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        loss_total = 0.0
        batch_count = 0
        for batch_index, batch in enumerate(train_loader):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(**batch).loss / args.gradient_accumulation_steps
            loss.backward()
            loss_total += float(loss.detach().cpu()) * args.gradient_accumulation_steps
            batch_count += 1
            should_update = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
            )
            if should_update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                if update == 1 or update % 100 == 0 or update >= total_updates:
                    heartbeat = {
                        "schema": "evomind.jigsaw.transformer_heartbeat.v1",
                        "created_at": now_iso(),
                        "status": "running",
                        "stage": "training",
                        "fold": fold,
                        "epoch": epoch + 1,
                        "update": update,
                        "total_updates": total_updates,
                        "progress_fraction": update / total_updates,
                        "mean_train_loss": loss_total / max(1, batch_count),
                        "elapsed_seconds": time.perf_counter() - started,
                        "cuda_memory_allocated_mib": round(torch.cuda.memory_allocated() / 1024 / 1024, 2),
                        "cuda_memory_reserved_mib": round(torch.cuda.memory_reserved() / 1024 / 1024, 2),
                        "cuda_peak_memory_mib": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2),
                        "diagnostic": diagnostic,
                        "plan_sha256": plan_sha256,
                        "private_labels_used": False,
                        "official_grader_executed": False,
                    }
                    write_json_atomic(heartbeat_path, heartbeat)
                    append_jsonl(progress_path, heartbeat)
                if update >= total_updates:
                    break
        valid_prediction = _predict(model, valid_loader, device, use_bf16=use_bf16)
        valid_auc = mean_columnwise_auc(truth[valid_selected], valid_prediction)
        epoch_records.append(
            {
                "epoch": epoch + 1,
                "updates": update,
                "mean_train_loss": loss_total / max(1, batch_count),
                "validation_auc": valid_auc,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        write_json_atomic(
            heartbeat_path,
            {
                "schema": "evomind.jigsaw.transformer_heartbeat.v1",
                "created_at": now_iso(),
                "status": "running",
                "stage": "epoch_evaluated",
                "fold": fold,
                "epoch": epoch + 1,
                "update": update,
                "total_updates": total_updates,
                "validation_auc": valid_auc,
                "elapsed_seconds": time.perf_counter() - started,
                "diagnostic": diagnostic,
                "plan_sha256": plan_sha256,
                "private_labels_used": False,
                "official_grader_executed": False,
            },
        )
        if update >= total_updates:
            break
    final_valid_prediction = _predict(model, valid_loader, device, use_bf16=use_bf16)
    final_test_prediction = _predict(model, test_loader, device, use_bf16=use_bf16)
    fold_model_dir = output_dir / f"fold_{fold}_model"
    model.save_pretrained(fold_model_dir, safe_serialization=True)
    del model
    torch.cuda.empty_cache()

    temporary = prediction_path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        valid_indices=valid_selected,
        valid_prediction=final_valid_prediction,
        test_indices=test_selected,
        test_prediction=final_test_prediction,
    )
    temporary.replace(prediction_path)
    metadata = {
        "schema": "evomind.jigsaw.transformer_fold.v1",
        "created_at": now_iso(),
        "status": "passed",
        "fold": fold,
        "plan_sha256": plan_sha256,
        "diagnostic": diagnostic,
        "seed": fold_seed,
        "train_rows": len(train_selected),
        "valid_rows": len(valid_selected),
        "test_rows": len(test_selected),
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "model_dir": str(fold_model_dir),
        "epochs": epoch_records,
        "mixed_precision": "bf16" if use_bf16 else "fp32",
        "cuda_after": _cuda_snapshot(),
        "private_labels_used": False,
        "official_grader_executed": False,
        "process_signals_sent": 0,
    }
    write_json_atomic(metadata_path, metadata)
    fold_complete = {
        "schema": "evomind.jigsaw.transformer_heartbeat.v1",
        "created_at": now_iso(),
        "status": "running",
        "stage": "fold_complete",
        "fold": fold,
        "fold_prediction_sha256": metadata["prediction_sha256"],
        "diagnostic": diagnostic,
        "plan_sha256": plan_sha256,
        "private_labels_used": False,
        "official_grader_executed": False,
        "process_signals_sent": 0,
    }
    write_json_atomic(heartbeat_path, fold_complete)
    append_jsonl(progress_path, fold_complete)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--sparse-bundle", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--frozen-plan", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_REVISION)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--token-cache-dir", type=Path)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold-seed", type=int)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke-train-rows", type=int, default=0)
    parser.add_argument("--smoke-valid-rows", type=int, default=0)
    parser.add_argument("--smoke-test-rows", type=int, default=0)
    parser.add_argument("--max-train-updates-per-fold", type=int, default=0)
    return parser


def validate_frozen_plan_arguments(
    args: argparse.Namespace,
    plan: dict[str, Any],
    *,
    diagnostic: bool,
) -> None:
    training = plan.get("training") or {}
    model = plan.get("model") or {}
    resolve_seed_contract(args, plan)
    expected = {
        "model_id": model.get("repo_id"),
        "model_revision": model.get("revision"),
        "epochs": training.get("epochs_per_fold"),
        "max_length": training.get("max_length"),
        "train_batch_size": training.get("train_batch_size"),
        "eval_batch_size": training.get("eval_batch_size"),
        "gradient_accumulation_steps": training.get("gradient_accumulation_steps"),
        "learning_rate": training.get("learning_rate"),
        "weight_decay": training.get("weight_decay"),
        "warmup_ratio": training.get("warmup_ratio"),
        "max_grad_norm": training.get("max_grad_norm"),
        "num_workers": training.get("num_workers"),
        "gradient_checkpointing": training.get("gradient_checkpointing"),
        "bf16": training.get("mixed_precision") == "bf16",
    }
    mismatches = {}
    for name, frozen in expected.items():
        actual = getattr(args, name)
        matches = math.isclose(actual, frozen, rel_tol=0.0, abs_tol=1e-12) if isinstance(frozen, float) else actual == frozen
        if not matches:
            mismatches[name] = {"frozen": frozen, "actual": actual}
    if mismatches:
        raise ValueError(f"Runtime arguments diverge from the frozen plan: {mismatches}")
    requested_folds = tuple(int(value.strip()) for value in args.folds.split(",") if value.strip())
    if not diagnostic and requested_folds != tuple(range(int(training.get("folds", 0)))):
        raise ValueError("Production execution must cover every frozen fold in order")


def main(argv: Sequence[str] | None = None) -> int:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen Jigsaw transformer run")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    diagnostic = any(
        value
        for value in (
            args.smoke_train_rows,
            args.smoke_valid_rows,
            args.smoke_test_rows,
            args.max_train_updates_per_fold,
        )
    )
    run_dir = (args.output_root / args.run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.frozen_plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    plan_sha256 = sha256_file(plan_path)
    if plan.get("status") != "frozen_before_training":
        raise ValueError("Transformer plan is not frozen")
    validate_frozen_plan_arguments(args, plan, diagnostic=diagnostic)
    token_dir = resolve_token_cache_dir(args, plan, run_dir)

    public_dir = args.public_dir.resolve()
    train_path = public_dir / "train.csv"
    test_path = public_dir / "test.csv"
    sample_path = public_dir / "sample_submission.csv"
    for path in (train_path, test_path, sample_path, args.sparse_bundle):
        if not path.resolve().is_file():
            raise FileNotFoundError(path)
    frozen_inputs = plan.get("inputs") or {}
    actual_input_hashes = {
        "public_train_sha256": sha256_file(train_path),
        "public_test_sha256": sha256_file(test_path),
        "public_sample_submission_sha256": sha256_file(sample_path),
        "sparse_bundle_sha256": sha256_file(args.sparse_bundle.resolve()),
    }
    mismatched_inputs = {
        name: {"frozen": frozen_inputs.get(name), "actual": value}
        for name, value in actual_input_hashes.items()
        if frozen_inputs.get(name) != value
    }
    if mismatched_inputs:
        raise ValueError(f"Runtime inputs diverge from the frozen plan: {mismatched_inputs}")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    targets = tuple(str(value) for value in sample.columns[1:])
    if targets != TARGET_COLUMNS:
        raise ValueError("Jigsaw target order is not the exact six-column contract")
    truth = train[list(TARGET_COLUMNS)].to_numpy(dtype=np.int8)
    sparse = load_sparse_bundle(args.sparse_bundle.resolve(), row_count=len(train), test_count=len(test))
    if not np.array_equal(sparse["truth"], truth):
        raise RuntimeError("Sparse public truth does not match the current public train file")
    splits, fold_assignment = make_multilabel_stratified_folds(
        truth, requested_folds=5, seed=args.fold_seed
    )
    if not np.array_equal(sparse["fold"].astype(np.int16), fold_assignment):
        raise RuntimeError("Transformer folds diverge from the immutable sparse OOF folds")

    model_snapshot = Path(
        snapshot_download(
            repo_id=args.model_id,
            revision=args.model_revision,
            cache_dir=args.hf_cache.resolve(),
        )
    ).resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_snapshot, local_files_only=True, use_fast=True)
    train_text = train["comment_text"].fillna("").astype(str).tolist()
    test_text = test["comment_text"].fillna("").astype(str).tolist()
    train_cache = build_or_load_token_cache(
        train_text,
        tokenizer,
        cache_path=token_dir / f"train_l{args.max_length}.npz",
        max_length=args.max_length,
        source_sha256=sha256_file(train_path),
        model_revision=args.model_revision,
    )
    test_cache = build_or_load_token_cache(
        test_text,
        tokenizer,
        cache_path=token_dir / f"test_l{args.max_length}.npz",
        max_length=args.max_length,
        source_sha256=sha256_file(test_path),
        model_revision=args.model_revision,
    )

    requested_folds = tuple(int(value.strip()) for value in args.folds.split(",") if value.strip())
    if not requested_folds or any(value not in range(5) for value in requested_folds):
        raise ValueError("Requested folds must be a non-empty subset of 0..4")
    manifest = {
        "schema": "evomind.jigsaw.transformer_run.v1",
        "created_at": now_iso(),
        "run_id": args.run_id,
        "status": "training",
        "diagnostic": diagnostic,
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha256,
        "public_inputs": {
            "train": {"path": str(train_path), "sha256": sha256_file(train_path)},
            "test": {"path": str(test_path), "sha256": sha256_file(test_path)},
            "sample": {"path": str(sample_path), "sha256": sha256_file(sample_path)},
            "sparse_bundle": {
                "path": str(args.sparse_bundle.resolve()),
                "sha256": sha256_file(args.sparse_bundle.resolve()),
            },
        },
        "model": {"repo_id": args.model_id, "revision": args.model_revision, "snapshot": str(model_snapshot)},
        "requested_folds": list(requested_folds),
        "seed_contract": {
            "fold_assignment_seed": int(args.fold_seed),
            "model_seed": int(args.model_seed),
            "fold_model_seeds": [int(args.model_seed + fold * 1009) for fold in range(5)],
            "fold_assignment_reused_across_confirmation_seeds": True,
        },
        "cuda_before": _cuda_snapshot(),
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "human_gate_preserved": True,
        "process_signals_sent": 0,
    }
    write_json_atomic(run_dir / "manifest.json", manifest)
    fold_results = []
    for fold in requested_folds:
        train_indices, valid_indices = splits[fold]
        fold_results.append(
            train_fold(
                fold=fold,
                model_snapshot=model_snapshot,
                train_cache=train_cache,
                test_cache=test_cache,
                truth=truth,
                train_indices=train_indices,
                valid_indices=valid_indices,
                output_dir=run_dir,
                args=args,
                plan_sha256=plan_sha256,
                diagnostic=diagnostic,
            )
        )

    summary: dict[str, Any] = {
        **manifest,
        "created_at": now_iso(),
        "status": "diagnostic_complete" if diagnostic else "partial",
        "fold_results": fold_results,
    }
    complete_production = not diagnostic and set(requested_folds) == set(range(5))
    if complete_production:
        transformer_oof = np.full_like(truth, np.nan, dtype=np.float64)
        transformer_test_by_fold = np.zeros((5, len(test), len(TARGET_COLUMNS)), dtype=np.float64)
        write_counts = np.zeros_like(truth, dtype=np.uint8)
        for fold in range(5):
            with np.load(run_dir / f"fold_{fold}_predictions.npz", allow_pickle=False) as archive:
                valid_indices = np.asarray(archive["valid_indices"], dtype=np.int64)
                transformer_oof[valid_indices] = archive["valid_prediction"]
                write_counts[valid_indices] += 1
                transformer_test_by_fold[fold] = archive["test_prediction"]
        if not np.all(write_counts == 1) or not np.isfinite(transformer_oof).all():
            raise RuntimeError("Transformer OOF assembly failed exact-once coverage")
        candidate_oof, candidate_test, candidate_counts, blend_contract = (
            cross_fit_sparse_transformer_rank_blend(
                sparse["oof_stacker"],
                transformer_oof,
                sparse["test_stacker"],
                transformer_test_by_fold,
                truth,
                fold_assignment,
            )
        )
        sparse_auc = mean_columnwise_auc(truth, fold_rank_matrix(sparse["oof_stacker"], fold_assignment))
        transformer_auc = mean_columnwise_auc(truth, fold_rank_matrix(transformer_oof, fold_assignment))
        candidate_auc = mean_columnwise_auc(truth, candidate_oof)
        strongest = max(sparse_auc, transformer_auc)
        promotion = {
            "schema": "evomind.mlebench_lite.jigsaw_transformer_promotion_gate.v1",
            "threshold": 0.987,
            "minimum_gain_over_strongest_base": 0.0003,
            "sparse_auc": sparse_auc,
            "transformer_auc": transformer_auc,
            "candidate_auc": candidate_auc,
            "gain_over_strongest_base": candidate_auc - strongest,
            "checks": {
                "aggregate_threshold": candidate_auc >= 0.987,
                "minimum_gain": candidate_auc - strongest >= 0.0003,
                "exact_once_transformer_oof": bool(np.all(write_counts == 1)),
                "exact_once_candidate_oof": bool(np.all(candidate_counts == 1)),
                "all_six_labels_scoreable": all(
                    np.unique(truth[:, label]).size == 2 for label in range(len(TARGET_COLUMNS))
                ),
                "prediction_provenance_validated": bool(blend_contract["exact_once_oof"]),
                "private_labels_unused": True,
            },
        }
        promotion["passed"] = all(promotion["checks"].values())
        submission = sample.copy()
        submission.loc[:, list(TARGET_COLUMNS)] = candidate_test
        submission_path = run_dir / "candidate_submission_withheld.csv"
        submission.to_csv(submission_path, index=False)
        prediction_path = run_dir / "jigsaw_transformer_oof_and_test.npz"
        np.savez_compressed(
            prediction_path,
            truth=truth,
            fold=fold_assignment,
            transformer_oof=transformer_oof,
            transformer_test_by_fold=transformer_test_by_fold,
            candidate_oof=candidate_oof,
            candidate_test=candidate_test,
            transformer_write_counts=write_counts,
            candidate_write_counts=candidate_counts,
            train_id=safe_unicode_ids(train["id"].tolist()),
            test_id=safe_unicode_ids(test["id"].tolist()),
        )
        summary.update(
            {
                "status": "promotion_gate_passed" if promotion["passed"] else "promotion_gate_failed",
                "transformer_oof_auc": transformer_auc,
                "candidate_oof_auc": candidate_auc,
                "promotion_gate": promotion,
                "blend_contract": blend_contract,
                "prediction_bundle": {"path": str(prediction_path), "sha256": sha256_file(prediction_path)},
                "submission_withheld": {"path": str(submission_path), "sha256": sha256_file(submission_path)},
            }
        )
    write_json_atomic(run_dir / "summary.json", summary)
    write_json_atomic(
        run_dir / "heartbeat.json",
        {
            "schema": "evomind.jigsaw.transformer_heartbeat.v1",
            "created_at": now_iso(),
            "status": summary["status"],
            "stage": "terminal",
            "run_id": args.run_id,
            "diagnostic": diagnostic,
            "plan_sha256": plan_sha256,
            "private_labels_used": False,
            "official_grader_executed": False,
            "process_signals_sent": 0,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
