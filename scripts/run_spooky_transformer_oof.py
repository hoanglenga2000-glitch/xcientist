"""Leakage-controlled Spooky Author transformer, byte-CNN, and sparse OOF runner.

The production candidate uses only the prepared public train/test/sample files.  It
never opens MLE-Bench private answers, runs the official grader, or submits to
Kaggle.  Duplicate-normalized texts are kept in one outer fold.  Neural budgets are
fixed before training, so the final outer validation labels never select a checkpoint.

For every held-out fold the deployment blend and temperature are selected on the
other folds only.  The same fold-specific transform is applied to that fold's test
components before test predictions are averaged.  This makes the scored OOF
topology match the deployed fold ensemble rather than scoring a different meta-model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold

CLASS_COLUMNS = ("EAP", "HPL", "MWS")
DEFAULT_MODEL = "distilbert/distilroberta-base"
DEFAULT_REVISION = "fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b"
DEFAULT_TEMPERATURES = (0.8, 0.9, 1.0, 1.1, 1.2)
DEFAULT_BLEND_STEPS = 20


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
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


def normalize_duplicate_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value)).lower()
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    return " ".join(re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+", text))


def build_duplicate_safe_folds(
    texts: Sequence[str],
    labels: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, dict[str, Any]]:
    target = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(texts) != len(target):
        raise ValueError("Spooky texts and labels do not share rows")
    normalized = [normalize_duplicate_text(value) for value in texts]
    group_keys = [hashlib.sha256(value.encode("utf-8")).hexdigest() for value in normalized]
    unique_keys = {value: index for index, value in enumerate(sorted(set(group_keys)))}
    groups = np.asarray([unique_keys[value] for value in group_keys], dtype=np.int64)
    group_labels: dict[int, set[int]] = {}
    for group, label in zip(groups, target, strict=True):
        group_labels.setdefault(int(group), set()).add(int(label))
    conflicting = sorted(group for group, values in group_labels.items() if len(values) != 1)
    if conflicting:
        raise ValueError(f"Duplicate-normalized groups contain conflicting labels: {conflicting[:10]}")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    assignment = np.full(len(target), -1, dtype=np.int16)
    for fold, (fit, valid) in enumerate(splitter.split(np.zeros(len(target)), target, groups)):
        fit_indices = np.asarray(fit, dtype=np.int64)
        valid_indices = np.asarray(valid, dtype=np.int64)
        if set(groups[fit_indices]).intersection(groups[valid_indices]):
            raise RuntimeError("A duplicate text group crossed an outer fold boundary")
        assignment[valid_indices] = fold
        splits.append((fit_indices, valid_indices))
    if np.any(assignment < 0):
        raise RuntimeError("Spooky outer-fold assignment is incomplete")
    counts = np.bincount(groups)
    report = {
        "schema": "evomind.spooky.duplicate_groups.v1",
        "normalization": "NFKC_lower_word_number_tokens_v1",
        "rows": len(target),
        "unique_groups": int(len(counts)),
        "duplicate_groups": int(np.sum(counts > 1)),
        "duplicate_rows_beyond_first": int(np.sum(np.maximum(counts - 1, 0))),
        "maximum_group_size": int(counts.max(initial=0)),
        "conflicting_label_groups": len(conflicting),
        "folds": folds,
        "seed": seed,
        "group_isolation": True,
    }
    return splits, assignment, report


def normalize_probability(matrix: np.ndarray) -> np.ndarray:
    probability = np.asarray(matrix, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[1] != len(CLASS_COLUMNS):
        raise ValueError("Spooky probabilities must have exactly three class columns")
    if not np.isfinite(probability).all():
        raise ValueError("Spooky probabilities contain a non-finite value")
    probability = np.clip(probability, 1e-7, 1.0)
    return probability / probability.sum(axis=1, keepdims=True)


def multiclass_log_loss(truth: np.ndarray, probability: np.ndarray) -> float:
    return float(
        log_loss(
            np.asarray(truth, dtype=np.int64),
            normalize_probability(probability),
            labels=np.arange(len(CLASS_COLUMNS)),
        )
    )


def _simplex_weights(component_count: int, steps: int) -> list[tuple[float, ...]]:
    if component_count < 1 or steps < 1:
        raise ValueError("Simplex dimensions and steps must be positive")

    def compositions(total: int, parts: int, prefix: tuple[int, ...] = ()) -> Iterable[tuple[int, ...]]:
        if parts == 1:
            yield (*prefix, total)
            return
        for value in range(total + 1):
            yield from compositions(total - value, parts - 1, (*prefix, value))

    return [tuple(value / steps for value in raw) for raw in compositions(steps, component_count)]


def apply_component_blend(
    components: Sequence[np.ndarray],
    weights: Sequence[float],
    *,
    temperature: float,
    mode: str,
) -> np.ndarray:
    if len(components) != len(weights) or not components:
        raise ValueError("Spooky blend components and weights do not match")
    numeric_weights = np.asarray(weights, dtype=np.float64)
    if np.any(numeric_weights < 0.0) or not math.isclose(
        float(numeric_weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("Spooky blend weights must be nonnegative and sum to one")
    if temperature <= 0.0:
        raise ValueError("Spooky blend temperature must be positive")
    stack = np.stack([normalize_probability(value) for value in components], axis=0)
    if mode == "probability":
        pooled = np.tensordot(numeric_weights, stack, axes=(0, 0))
        logits = np.log(np.clip(pooled, 1e-7, 1.0)) / temperature
    elif mode == "log_probability":
        logits = np.tensordot(numeric_weights, np.log(stack), axes=(0, 0)) / temperature
    else:
        raise ValueError(f"Unknown Spooky blend mode: {mode}")
    logits -= logits.max(axis=1, keepdims=True)
    exponential = np.exp(logits)
    return normalize_probability(exponential)


def select_component_blend(
    components: Sequence[np.ndarray],
    truth: np.ndarray,
    *,
    steps: int = DEFAULT_BLEND_STEPS,
    temperatures: Sequence[float] = DEFAULT_TEMPERATURES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    weights = _simplex_weights(len(components), steps)
    for mode in ("probability", "log_probability"):
        for temperature in temperatures:
            for value in weights:
                prediction = apply_component_blend(
                    components,
                    value,
                    temperature=float(temperature),
                    mode=mode,
                )
                candidates.append(
                    {
                        "mode": mode,
                        "temperature": float(temperature),
                        "weights": list(value),
                        "log_loss": multiclass_log_loss(truth, prediction),
                    }
                )
    selected = min(
        candidates,
        key=lambda item: (
            item["log_loss"],
            abs(item["temperature"] - 1.0),
            item["mode"] != "probability",
            tuple(-value for value in item["weights"]),
        ),
    )
    return selected, candidates


def cross_fit_deployment_blend(
    component_oof: dict[str, np.ndarray],
    component_test_by_fold: dict[str, np.ndarray],
    truth: np.ndarray,
    fold_assignment: np.ndarray,
    *,
    steps: int = DEFAULT_BLEND_STEPS,
    temperatures: Sequence[float] = DEFAULT_TEMPERATURES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    names = tuple(component_oof)
    if names != tuple(component_test_by_fold) or len(names) < 2:
        raise ValueError("Spooky OOF/test components require the same ordered names")
    labels = np.asarray(truth, dtype=np.int64).reshape(-1)
    folds = np.asarray(fold_assignment, dtype=np.int16).reshape(-1)
    unique_folds = sorted(int(value) for value in np.unique(folds))
    if unique_folds != list(range(len(unique_folds))) or len(unique_folds) < 3:
        raise ValueError("Spooky deployment blend requires contiguous zero-based folds")
    oof_matrices = {name: normalize_probability(component_oof[name]) for name in names}
    if any(len(value) != len(labels) for value in oof_matrices.values()):
        raise ValueError("Spooky component OOF rows do not match truth")
    test_matrices = {
        name: np.asarray(component_test_by_fold[name], dtype=np.float64) for name in names
    }
    expected_test_rows = test_matrices[names[0]].shape[1]
    expected_test_shape = (len(unique_folds), expected_test_rows, len(CLASS_COLUMNS))
    if any(value.shape != expected_test_shape for value in test_matrices.values()):
        raise ValueError("Spooky test components require one aligned matrix per fold")

    candidate_oof = np.full((len(labels), len(CLASS_COLUMNS)), np.nan, dtype=np.float64)
    write_counts = np.zeros(len(labels), dtype=np.uint8)
    foldwise_test = np.zeros(expected_test_shape, dtype=np.float64)
    records: list[dict[str, Any]] = []
    for fold in unique_folds:
        fit_mask = folds != fold
        score_mask = folds == fold
        selected, candidates = select_component_blend(
            [oof_matrices[name][fit_mask] for name in names],
            labels[fit_mask],
            steps=steps,
            temperatures=temperatures,
        )
        candidate_oof[score_mask] = apply_component_blend(
            [oof_matrices[name][score_mask] for name in names],
            selected["weights"],
            temperature=selected["temperature"],
            mode=selected["mode"],
        )
        write_counts[score_mask] += 1
        foldwise_test[fold] = apply_component_blend(
            [test_matrices[name][fold] for name in names],
            selected["weights"],
            temperature=selected["temperature"],
            mode=selected["mode"],
        )
        records.append(
            {
                "score_fold": fold,
                "fit_folds": [value for value in unique_folds if value != fold],
                "component_order": list(names),
                "selected": selected,
                "candidate_count": len(candidates),
            }
        )
    if not np.all(write_counts == 1) or not np.isfinite(candidate_oof).all():
        raise RuntimeError("Spooky deployment OOF was not written exactly once")
    candidate_test = normalize_probability(foldwise_test.mean(axis=0))
    contract = {
        "schema": "evomind.spooky.exact_deployment_blend.v1",
        "component_order": list(names),
        "folds": unique_folds,
        "selection": "non_held_outer_folds_only",
        "test_aggregation": "apply_fold_specific_transform_then_average",
        "weight_grid_steps": steps,
        "temperatures": [float(value) for value in temperatures],
        "records": records,
        "exact_once_oof": True,
        "private_labels_used": False,
    }
    return candidate_oof, candidate_test, write_counts, contract


def _fit_nbsvm_channel(
    fit_text: Sequence[str],
    fit_labels: np.ndarray,
    valid_text: Sequence[str],
    test_text: Sequence[str],
    *,
    analyzer: str,
    ngram_range: tuple[int, int],
    max_features: int,
    c_value: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.995,
        max_features=max_features,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    fit_matrix = vectorizer.fit_transform(fit_text)
    valid_matrix = vectorizer.transform(valid_text)
    test_matrix = vectorizer.transform(test_text)
    valid_probability = np.zeros((len(valid_text), len(CLASS_COLUMNS)), dtype=np.float64)
    test_probability = np.zeros((len(test_text), len(CLASS_COLUMNS)), dtype=np.float64)
    model_records = []
    labels = np.asarray(fit_labels, dtype=np.int64)
    for class_index, class_name in enumerate(CLASS_COLUMNS):
        binary = labels == class_index
        positive = np.asarray(fit_matrix[binary].sum(axis=0)).reshape(-1) + 1.0
        negative = np.asarray(fit_matrix[~binary].sum(axis=0)).reshape(-1) + 1.0
        ratio = np.log(positive / positive.sum()) - np.log(negative / negative.sum())
        model = LogisticRegression(
            C=c_value,
            solver="liblinear",
            max_iter=400,
            random_state=2300 + class_index,
        )
        model.fit(fit_matrix.multiply(ratio), binary.astype(np.int8))
        valid_probability[:, class_index] = model.predict_proba(
            valid_matrix.multiply(ratio)
        )[:, 1]
        test_probability[:, class_index] = model.predict_proba(test_matrix.multiply(ratio))[:, 1]
        model_records.append(
            {
                "class": class_name,
                "iterations": int(np.max(np.asarray(model.n_iter_))),
            }
        )
    return (
        normalize_probability(valid_probability),
        normalize_probability(test_probability),
        {
            "analyzer": analyzer,
            "ngram_range": list(ngram_range),
            "features": int(len(vectorizer.get_feature_names_out())),
            "models": model_records,
        },
    )


def fit_sparse_fold(
    fit_text: Sequence[str],
    fit_labels: np.ndarray,
    valid_text: Sequence[str],
    test_text: Sequence[str],
    *,
    word_features: int,
    char_features: int,
    c_value: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    word_valid, word_test, word_record = _fit_nbsvm_channel(
        fit_text,
        fit_labels,
        valid_text,
        test_text,
        analyzer="word",
        ngram_range=(1, 2),
        max_features=word_features,
        c_value=c_value,
    )
    char_valid, char_test, char_record = _fit_nbsvm_channel(
        fit_text,
        fit_labels,
        valid_text,
        test_text,
        analyzer="char",
        ngram_range=(2, 6),
        max_features=char_features,
        c_value=c_value,
    )
    return (
        normalize_probability(0.3 * word_valid + 0.7 * char_valid),
        normalize_probability(0.3 * word_test + 0.7 * char_test),
        {
            "schema": "evomind.spooky.sparse_fold.v1",
            "fixed_channel_weights": {"word": 0.3, "raw_char": 0.7},
            "word": word_record,
            "raw_char": char_record,
        },
    )


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
        "schema": "evomind.spooky.transformer_token_cache.v1",
        "rows": len(texts),
        "max_length": int(max_length),
        "source_sha256": source_sha256,
        "model_revision": model_revision,
    }
    if cache_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if all(manifest.get(key) == value for key, value in expected.items()):
            with np.load(cache_path, allow_pickle=False) as archive:
                ids = np.asarray(archive["input_ids"], dtype=np.int32)
                mask = np.asarray(archive["attention_mask"], dtype=np.uint8)
            if ids.shape == mask.shape == (len(texts), max_length):
                return TokenCache(input_ids=ids, attention_mask=mask)
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
    ids = np.concatenate(input_parts, axis=0)
    mask = np.concatenate(mask_parts, axis=0)
    temporary = cache_path.with_suffix(cache_path.suffix + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, input_ids=ids, attention_mask=mask)
    temporary.replace(cache_path)
    write_json_atomic(
        manifest_path,
        {**expected, "created_at": now_iso(), "cache_sha256": sha256_file(cache_path)},
    )
    return TokenCache(input_ids=ids, attention_mask=mask)


class EncodedTextDataset:
    def __init__(
        self,
        cache: TokenCache,
        indices: np.ndarray,
        labels: np.ndarray | None,
    ) -> None:
        self.cache = cache
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)

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
            output["labels"] = torch.as_tensor(self.labels[row], dtype=torch.long)
        return output


def encode_bytes(texts: Sequence[str], *, max_length: int) -> np.ndarray:
    encoded = np.zeros((len(texts), max_length), dtype=np.uint16)
    for row, text in enumerate(texts):
        values = str(text).encode("utf-8", "replace")[:max_length]
        if values:
            encoded[row, : len(values)] = np.frombuffer(values, dtype=np.uint8).astype(np.uint16) + 1
    return encoded


class ByteDataset:
    def __init__(self, values: np.ndarray, labels: np.ndarray | None) -> None:
        self.values = np.asarray(values, dtype=np.uint16)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, item: int) -> dict[str, Any]:
        import torch

        output = {"input_ids": torch.as_tensor(self.values[item], dtype=torch.long)}
        if self.labels is not None:
            output["labels"] = torch.as_tensor(self.labels[item], dtype=torch.long)
        return output


def build_byte_cnn(*, embedding_dim: int, channels: int, dropout: float) -> Any:
    import torch

    class ByteCNN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(257, embedding_dim, padding_idx=0)
            self.convolutions = torch.nn.ModuleList(
                [
                    torch.nn.Conv1d(embedding_dim, channels, kernel_size=kernel)
                    for kernel in (3, 5, 7)
                ]
            )
            self.dropout = torch.nn.Dropout(dropout)
            self.output = torch.nn.Linear(channels * len(self.convolutions), len(CLASS_COLUMNS))

        def forward(self, input_ids: Any) -> Any:
            sequence = self.embedding(input_ids).transpose(1, 2)
            pooled = [torch.amax(torch.nn.functional.gelu(layer(sequence)), dim=2) for layer in self.convolutions]
            return self.output(self.dropout(torch.cat(pooled, dim=1)))

    return ByteCNN()


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def _predict_transformer(model: Any, loader: Any, device: Any, *, use_bf16: bool) -> np.ndarray:
    import torch

    model.eval()
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
                if key != "labels"
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(**inputs).logits
            predictions.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return normalize_probability(np.concatenate(predictions, axis=0))


def _predict_byte(model: Any, loader: Any, device: Any, *, use_bf16: bool) -> np.ndarray:
    import torch

    model.eval()
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            values = batch["input_ids"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(values)
            predictions.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return normalize_probability(np.concatenate(predictions, axis=0))


def train_fold(
    *,
    fold: int,
    model_snapshot: Path,
    train_cache: TokenCache,
    test_cache: TokenCache,
    all_text: Sequence[str],
    test_text: Sequence[str],
    labels: np.ndarray,
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

    fold_seed = int(args.seed + fold * 1009)
    _seed_everything(fold_seed)
    fit_selected = np.asarray(train_indices, dtype=np.int64)
    valid_selected = np.asarray(valid_indices, dtype=np.int64)
    test_selected = np.arange(len(test_text), dtype=np.int64)
    if args.smoke_train_rows:
        fit_selected = fit_selected[: args.smoke_train_rows]
    if args.smoke_valid_rows:
        valid_selected = valid_selected[: args.smoke_valid_rows]
    if args.smoke_test_rows:
        test_selected = test_selected[: args.smoke_test_rows]

    generator = torch.Generator().manual_seed(fold_seed)
    train_loader = DataLoader(
        EncodedTextDataset(train_cache, fit_selected, labels),
        batch_size=args.train_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=bool(args.num_workers),
    )
    valid_loader = DataLoader(
        EncodedTextDataset(train_cache, valid_selected, labels),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=bool(args.num_workers),
    )
    test_loader = DataLoader(
        EncodedTextDataset(test_cache, test_selected, None),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=bool(args.num_workers),
    )
    write_json_atomic(
        heartbeat_path,
        {
            "schema": "evomind.spooky.transformer_heartbeat.v1",
            "created_at": now_iso(),
            "status": "running",
            "stage": "transformer_model_load",
            "fold": fold,
            "plan_sha256": plan_sha256,
            "diagnostic": diagnostic,
            "private_labels_used": False,
            "official_grader_executed": False,
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
    parameter_groups = [
        {
            "params": [
                value
                for name, value in model.named_parameters()
                if value.requires_grad and not any(term in name for term in no_decay)
            ],
            "weight_decay": args.weight_decay,
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
    optimizer = torch.optim.AdamW(parameter_groups, lr=args.learning_rate)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_updates = max(1, updates_per_epoch * args.epochs)
    if args.max_train_updates_per_fold:
        total_updates = min(total_updates, args.max_train_updates_per_fold)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        int(total_updates * args.warmup_ratio),
        total_updates,
    )
    use_bf16 = bool(args.bf16 and torch.cuda.is_bf16_supported())
    optimizer.zero_grad(set_to_none=True)
    update = 0
    epoch_records = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch_index, batch in enumerate(train_loader):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(**batch).loss / args.gradient_accumulation_steps
            loss.backward()
            total_loss += float(loss.detach().cpu()) * args.gradient_accumulation_steps
            batches += 1
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
                if update == 1 or update % 50 == 0 or update >= total_updates:
                    heartbeat = {
                        "schema": "evomind.spooky.transformer_heartbeat.v1",
                        "created_at": now_iso(),
                        "status": "running",
                        "stage": "transformer_training",
                        "fold": fold,
                        "epoch": epoch + 1,
                        "update": update,
                        "total_updates": total_updates,
                        "progress_fraction": update / total_updates,
                        "mean_train_loss": total_loss / max(1, batches),
                        "elapsed_seconds": time.perf_counter() - started,
                        "plan_sha256": plan_sha256,
                        "diagnostic": diagnostic,
                        "private_labels_used": False,
                        "official_grader_executed": False,
                    }
                    write_json_atomic(heartbeat_path, heartbeat)
                    append_jsonl(progress_path, heartbeat)
                if update >= total_updates:
                    break
        valid_probability = _predict_transformer(model, valid_loader, device, use_bf16=use_bf16)
        epoch_records.append(
            {
                "epoch": epoch + 1,
                "updates": update,
                "mean_train_loss": total_loss / max(1, batches),
                "validation_log_loss": multiclass_log_loss(labels[valid_selected], valid_probability),
                "elapsed_seconds": time.perf_counter() - started,
                "checkpoint_selection_used_outer_fold": False,
            }
        )
        if update >= total_updates:
            break
    transformer_valid = _predict_transformer(model, valid_loader, device, use_bf16=use_bf16)
    transformer_test = _predict_transformer(model, test_loader, device, use_bf16=use_bf16)
    transformer_dir = output_dir / f"fold_{fold}_transformer_model"
    model.save_pretrained(transformer_dir, safe_serialization=True)
    del model
    torch.cuda.empty_cache()

    byte_train = encode_bytes([all_text[index] for index in fit_selected], max_length=args.byte_max_length)
    byte_valid = encode_bytes([all_text[index] for index in valid_selected], max_length=args.byte_max_length)
    byte_test = encode_bytes([test_text[index] for index in test_selected], max_length=args.byte_max_length)
    byte_train_loader = DataLoader(
        ByteDataset(byte_train, labels[fit_selected]),
        batch_size=args.byte_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(fold_seed + 17),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=bool(args.num_workers),
    )
    byte_valid_loader = DataLoader(
        ByteDataset(byte_valid, labels[valid_selected]),
        batch_size=args.byte_eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=bool(args.num_workers),
    )
    byte_test_loader = DataLoader(
        ByteDataset(byte_test, None),
        batch_size=args.byte_eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=bool(args.num_workers),
    )
    byte_model = build_byte_cnn(
        embedding_dim=args.byte_embedding_dim,
        channels=args.byte_channels,
        dropout=args.byte_dropout,
    ).to(device)
    byte_optimizer = torch.optim.AdamW(
        byte_model.parameters(),
        lr=args.byte_learning_rate,
        weight_decay=args.byte_weight_decay,
    )
    byte_loss = torch.nn.CrossEntropyLoss(label_smoothing=args.byte_label_smoothing)
    byte_records = []
    for epoch in range(args.byte_epochs):
        byte_model.train()
        total_loss = 0.0
        batches = 0
        for batch in byte_train_loader:
            values = batch["input_ids"].to(device, non_blocking=True)
            target = batch["labels"].to(device, non_blocking=True)
            byte_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss = byte_loss(byte_model(values), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(byte_model.parameters(), 1.0)
            byte_optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        byte_valid_probability = _predict_byte(
            byte_model,
            byte_valid_loader,
            device,
            use_bf16=use_bf16,
        )
        byte_records.append(
            {
                "epoch": epoch + 1,
                "mean_train_loss": total_loss / max(1, batches),
                "validation_log_loss": multiclass_log_loss(
                    labels[valid_selected], byte_valid_probability
                ),
                "checkpoint_selection_used_outer_fold": False,
            }
        )
    byte_valid_probability = _predict_byte(
        byte_model,
        byte_valid_loader,
        device,
        use_bf16=use_bf16,
    )
    byte_test_probability = _predict_byte(
        byte_model,
        byte_test_loader,
        device,
        use_bf16=use_bf16,
    )
    byte_model_path = output_dir / f"fold_{fold}_byte_cnn.pt"
    torch.save(byte_model.state_dict(), byte_model_path)
    del byte_model
    torch.cuda.empty_cache()

    sparse_valid, sparse_test, sparse_record = fit_sparse_fold(
        [all_text[index] for index in fit_selected],
        labels[fit_selected],
        [all_text[index] for index in valid_selected],
        [test_text[index] for index in test_selected],
        word_features=args.sparse_word_features,
        char_features=args.sparse_char_features,
        c_value=args.sparse_c,
    )
    temporary = prediction_path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        valid_indices=valid_selected,
        test_indices=test_selected,
        transformer_valid=transformer_valid,
        transformer_test=transformer_test,
        byte_valid=byte_valid_probability,
        byte_test=byte_test_probability,
        sparse_valid=sparse_valid,
        sparse_test=sparse_test,
    )
    temporary.replace(prediction_path)
    metadata = {
        "schema": "evomind.spooky.transformer_byte_sparse_fold.v1",
        "created_at": now_iso(),
        "status": "passed",
        "fold": fold,
        "seed": fold_seed,
        "plan_sha256": plan_sha256,
        "diagnostic": diagnostic,
        "train_rows": len(fit_selected),
        "valid_rows": len(valid_selected),
        "test_rows": len(test_selected),
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "transformer_model_dir": str(transformer_dir),
        "byte_model": {"path": str(byte_model_path), "sha256": sha256_file(byte_model_path)},
        "transformer_epochs": epoch_records,
        "byte_epochs": byte_records,
        "sparse": sparse_record,
        "component_log_loss": {
            "transformer": multiclass_log_loss(labels[valid_selected], transformer_valid),
            "byte": multiclass_log_loss(labels[valid_selected], byte_valid_probability),
            "sparse": multiclass_log_loss(labels[valid_selected], sparse_valid),
        },
        "checkpoint_budget": "fixed_before_outer_fold_training",
        "checkpoint_selection_used_outer_fold": False,
        "mixed_precision": "bf16" if use_bf16 else "fp32",
        "cuda_after": _cuda_snapshot(),
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }
    write_json_atomic(metadata_path, metadata)
    write_json_atomic(
        heartbeat_path,
        {
            "schema": "evomind.spooky.transformer_heartbeat.v1",
            "created_at": now_iso(),
            "status": "running",
            "stage": "fold_complete",
            "fold": fold,
            "fold_prediction_sha256": metadata["prediction_sha256"],
            "plan_sha256": plan_sha256,
            "diagnostic": diagnostic,
            "private_labels_used": False,
            "official_grader_executed": False,
        },
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--frozen-plan", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_REVISION)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--byte-max-length", type=int, default=768)
    parser.add_argument("--byte-embedding-dim", type=int, default=48)
    parser.add_argument("--byte-channels", type=int, default=128)
    parser.add_argument("--byte-dropout", type=float, default=0.2)
    parser.add_argument("--byte-epochs", type=int, default=6)
    parser.add_argument("--byte-batch-size", type=int, default=128)
    parser.add_argument("--byte-eval-batch-size", type=int, default=256)
    parser.add_argument("--byte-learning-rate", type=float, default=1e-3)
    parser.add_argument("--byte-weight-decay", type=float, default=1e-4)
    parser.add_argument("--byte-label-smoothing", type=float, default=0.05)
    parser.add_argument("--sparse-word-features", type=int, default=100000)
    parser.add_argument("--sparse-char-features", type=int, default=180000)
    parser.add_argument("--sparse-c", type=float, default=4.0)
    parser.add_argument("--blend-steps", type=int, default=DEFAULT_BLEND_STEPS)
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
    byte = plan.get("byte_channel") or {}
    sparse = plan.get("sparse_channel") or {}
    ensemble = plan.get("ensemble") or {}
    model = plan.get("model") or {}
    expected = {
        "model_id": model.get("repo_id"),
        "model_revision": model.get("revision"),
        "seed": training.get("seed"),
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
        "byte_max_length": byte.get("max_length"),
        "byte_embedding_dim": byte.get("embedding_dim"),
        "byte_channels": byte.get("channels"),
        "byte_dropout": byte.get("dropout"),
        "byte_epochs": byte.get("epochs_per_fold"),
        "byte_batch_size": byte.get("train_batch_size"),
        "byte_eval_batch_size": byte.get("eval_batch_size"),
        "byte_learning_rate": byte.get("learning_rate"),
        "byte_weight_decay": byte.get("weight_decay"),
        "byte_label_smoothing": byte.get("label_smoothing"),
        "sparse_word_features": sparse.get("word_max_features"),
        "sparse_char_features": sparse.get("char_max_features"),
        "sparse_c": sparse.get("c_value"),
        "blend_steps": ensemble.get("weight_grid_steps"),
    }
    mismatches = {}
    for name, frozen in expected.items():
        actual = getattr(args, name)
        matches = (
            math.isclose(actual, frozen, rel_tol=0.0, abs_tol=1e-12)
            if isinstance(frozen, float)
            else actual == frozen
        )
        if not matches:
            mismatches[name] = {"frozen": frozen, "actual": actual}
    if mismatches:
        raise ValueError(f"Runtime arguments diverge from the frozen plan: {mismatches}")
    requested = tuple(int(value.strip()) for value in args.folds.split(",") if value.strip())
    if not diagnostic and requested != tuple(range(int(training.get("folds", 0)))):
        raise ValueError("Production execution must cover every frozen fold in order")


def main(argv: Sequence[str] | None = None) -> int:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen Spooky transformer run")
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
        raise ValueError("Spooky plan is not frozen")
    validate_frozen_plan_arguments(args, plan, diagnostic=diagnostic)

    public_dir = args.public_dir.resolve()
    train_path = public_dir / "train.csv"
    test_path = public_dir / "test.csv"
    sample_path = public_dir / "sample_submission.csv"
    for path in (train_path, test_path, sample_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    actual_hashes = {
        "public_train_sha256": sha256_file(train_path),
        "public_test_sha256": sha256_file(test_path),
        "public_sample_submission_sha256": sha256_file(sample_path),
    }
    frozen_inputs = plan.get("inputs") or {}
    mismatch = {
        key: {"frozen": frozen_inputs.get(key), "actual": value}
        for key, value in actual_hashes.items()
        if frozen_inputs.get(key) != value
    }
    if mismatch:
        raise ValueError(f"Runtime inputs diverge from the frozen plan: {mismatch}")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    if list(train.columns) != ["id", "text", "author"]:
        raise ValueError("Spooky public train schema is not exact")
    if list(test.columns) != ["id", "text"] or list(sample.columns) != ["id", *CLASS_COLUMNS]:
        raise ValueError("Spooky test/sample schema is not exact")
    if sample["id"].astype(str).tolist() != test["id"].astype(str).tolist():
        raise ValueError("Spooky sample submission IDs do not match public test order")
    class_to_index = {value: index for index, value in enumerate(CLASS_COLUMNS)}
    labels = train["author"].map(class_to_index)
    if labels.isna().any():
        raise ValueError("Spooky public train includes an unexpected author label")
    target = labels.to_numpy(dtype=np.int64)
    all_text = train["text"].fillna("").astype(str).tolist()
    test_text = test["text"].fillna("").astype(str).tolist()
    splits, fold_assignment, duplicate_report = build_duplicate_safe_folds(
        all_text,
        target,
        folds=int(plan["training"]["folds"]),
        seed=args.seed,
    )
    write_json_atomic(run_dir / "duplicate_groups.json", duplicate_report)
    fold_path = run_dir / "fold_assignments.npz"
    np.savez_compressed(
        fold_path,
        train_id=train["id"].astype(str).to_numpy(),
        truth=target,
        fold=fold_assignment,
    )

    model_snapshot = Path(
        snapshot_download(
            repo_id=args.model_id,
            revision=args.model_revision,
            cache_dir=args.hf_cache.resolve(),
        )
    ).resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_snapshot, local_files_only=True, use_fast=True)
    train_cache = build_or_load_token_cache(
        all_text,
        tokenizer,
        cache_path=run_dir / "token_cache" / f"train_l{args.max_length}.npz",
        max_length=args.max_length,
        source_sha256=actual_hashes["public_train_sha256"],
        model_revision=args.model_revision,
    )
    test_cache = build_or_load_token_cache(
        test_text,
        tokenizer,
        cache_path=run_dir / "token_cache" / f"test_l{args.max_length}.npz",
        max_length=args.max_length,
        source_sha256=actual_hashes["public_test_sha256"],
        model_revision=args.model_revision,
    )
    requested = tuple(int(value.strip()) for value in args.folds.split(",") if value.strip())
    if not requested or any(value not in range(len(splits)) for value in requested):
        raise ValueError("Requested Spooky folds are outside the frozen fold range")
    manifest = {
        "schema": "evomind.spooky.transformer_run.v1",
        "created_at": now_iso(),
        "run_id": args.run_id,
        "status": "training",
        "diagnostic": diagnostic,
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha256,
        "public_inputs": {
            "train": {"path": str(train_path), "sha256": actual_hashes["public_train_sha256"]},
            "test": {"path": str(test_path), "sha256": actual_hashes["public_test_sha256"]},
            "sample": {
                "path": str(sample_path),
                "sha256": actual_hashes["public_sample_submission_sha256"],
            },
        },
        "model": {
            "repo_id": args.model_id,
            "revision": args.model_revision,
            "snapshot": str(model_snapshot),
        },
        "fold_assignment": {"path": str(fold_path), "sha256": sha256_file(fold_path)},
        "duplicate_group_isolation": duplicate_report,
        "requested_folds": list(requested),
        "cuda_before": _cuda_snapshot(),
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "human_gate_preserved": True,
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
                all_text=all_text,
                test_text=test_text,
                labels=target,
                train_indices=fit_indices,
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
    complete = not diagnostic and set(requested) == set(range(len(splits)))
    if complete:
        component_oof = {
            name: np.full((len(train), len(CLASS_COLUMNS)), np.nan, dtype=np.float64)
            for name in ("sparse", "byte", "transformer")
        }
        component_test = {
            name: np.zeros((len(splits), len(test), len(CLASS_COLUMNS)), dtype=np.float64)
            for name in component_oof
        }
        component_counts = np.zeros(len(train), dtype=np.uint8)
        for fold in range(len(splits)):
            with np.load(run_dir / f"fold_{fold}_predictions.npz", allow_pickle=False) as archive:
                valid_indices = np.asarray(archive["valid_indices"], dtype=np.int64)
                for name in component_oof:
                    component_oof[name][valid_indices] = archive[f"{name}_valid"]
                    component_test[name][fold] = archive[f"{name}_test"]
                component_counts[valid_indices] += 1
        if not np.all(component_counts == 1) or any(
            not np.isfinite(value).all() for value in component_oof.values()
        ):
            raise RuntimeError("Spooky component OOF assembly failed exact-once coverage")
        candidate_oof, candidate_test, candidate_counts, blend_contract = cross_fit_deployment_blend(
            component_oof,
            component_test,
            target,
            fold_assignment,
            steps=args.blend_steps,
        )
        component_scores = {
            name: multiclass_log_loss(target, value) for name, value in component_oof.items()
        }
        candidate_score = multiclass_log_loss(target, candidate_oof)
        single_seed_threshold = float(plan["promotion_gate"]["single_seed_log_loss_threshold"])
        single_seed_passed = candidate_score <= single_seed_threshold
        promotion = {
            "schema": "evomind.spooky.single_seed_promotion_gate.v1",
            "seed": args.seed,
            "candidate_oof_log_loss": candidate_score,
            "single_seed_threshold": single_seed_threshold,
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
                    not record["checkpoint_selection_used_outer_fold"]
                    for fold_result in fold_results
                    for record in [fold_result]
                ),
                "private_labels_unused": True,
                "official_grader_not_executed": True,
                "kaggle_submission_not_executed": True,
            },
        }
        submission = sample.copy()
        submission.loc[:, list(CLASS_COLUMNS)] = candidate_test
        submission_path = run_dir / "candidate_submission_withheld.csv"
        submission.to_csv(submission_path, index=False)
        bundle_path = run_dir / "spooky_transformer_oof_and_test.npz"
        np.savez_compressed(
            bundle_path,
            truth=target,
            fold=fold_assignment,
            sparse_oof=component_oof["sparse"],
            byte_oof=component_oof["byte"],
            transformer_oof=component_oof["transformer"],
            sparse_test_by_fold=component_test["sparse"],
            byte_test_by_fold=component_test["byte"],
            transformer_test_by_fold=component_test["transformer"],
            candidate_oof=candidate_oof,
            candidate_test=candidate_test,
            component_write_counts=component_counts,
            candidate_write_counts=candidate_counts,
            train_id=train["id"].astype(str).to_numpy(),
            test_id=test["id"].astype(str).to_numpy(),
        )
        summary.update(
            {
                "status": (
                    "single_seed_gate_passed_confirmation_pending"
                    if single_seed_passed
                    else "single_seed_gate_failed"
                ),
                "component_oof_log_loss": component_scores,
                "candidate_oof_log_loss": candidate_score,
                "promotion_gate": promotion,
                "blend_contract": blend_contract,
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
            "schema": "evomind.spooky.transformer_heartbeat.v1",
            "created_at": now_iso(),
            "status": summary["status"],
            "stage": "terminal",
            "run_id": args.run_id,
            "diagnostic": diagnostic,
            "plan_sha256": plan_sha256,
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
