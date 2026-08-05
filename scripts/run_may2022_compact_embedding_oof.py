#!/usr/bin/env python3
"""Fold-safe compact character-embedding recovery for TPS May 2022.

The outer validation fold is never used for epoch selection.  Each outer fold
first selects an epoch count on an inner split, then trains a fresh model on the
entire outer-fit partition for exactly that many epochs.  Only the fixed public
train/test/sample CSV files are read; official grading and Kaggle submission are
outside this runner.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPETITION_ID = "tabular-playground-series-may-2022"
DEFAULT_PUBLIC_DIR = (
    PROJECT_ROOT
    / "workspace"
    / "local_gpu"
    / "mlebench_official_data"
    / COMPETITION_ID
    / "prepared"
    / "public"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "workspace" / "local_gpu" / "may2022_compact_embedding"
DEFAULT_PLAN = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "may2022_compact_embedding_s42_frozen_plan.json"
)
F27_ALPHABET = "ABCDEFGHIJKLMNOPQRST"
F27_WIDTH = 10
NUMERIC_COLUMNS = tuple(f"f_{index:02d}" for index in range(31) if index != 27)
EXPECTED_TRAIN_ROWS = 800_000
EXPECTED_TEST_ROWS = 100_000
AGGREGATE_PROMOTION_AUC = 0.9985
EVERY_FOLD_PROMOTION_AUC = 0.99818


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def encode_f27(values: pd.Series) -> np.ndarray:
    strings = values.fillna("").astype(str)
    lengths = strings.str.len().to_numpy()
    if not np.all(lengths == F27_WIDTH):
        raise ValueError(
            f"May-2022 f_27 must contain exactly {F27_WIDTH} characters; "
            f"observed={sorted(np.unique(lengths).astype(int).tolist())}"
        )
    joined = "".join(strings.tolist())
    try:
        raw = np.frombuffer(joined.encode("ascii"), dtype=np.uint8).astype(np.int16)
    except UnicodeEncodeError as exc:
        raise ValueError("May-2022 f_27 contains non-ASCII characters") from exc
    encoded = raw.reshape(len(strings), F27_WIDTH) - ord("A")
    if encoded.size and (encoded.min() < 0 or encoded.max() >= len(F27_ALPHABET)):
        raise ValueError("May-2022 f_27 contains characters outside A-T")
    return encoded.astype(np.int8, copy=False)


def build_numeric_features(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    required = {*NUMERIC_COLUMNS, "f_27"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"May-2022 data is missing required columns: {missing}")
    numeric = frame.loc[:, NUMERIC_COLUMNS].apply(pd.to_numeric, errors="raise")
    base = numeric.to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(base).all():
        raise ValueError("May-2022 numeric features contain non-finite values")
    characters = encode_f27(frame["f_27"])
    features: list[np.ndarray] = [base]
    names = list(NUMERIC_COLUMNS)

    count_matrix = np.column_stack(
        [(characters == index).sum(axis=1) for index in range(len(F27_ALPHABET))]
    ).astype(np.float32)
    features.append(count_matrix)
    names.extend(f"f27_count_{letter}" for letter in F27_ALPHABET)
    unique_count = (count_matrix > 0).sum(axis=1).astype(np.float32)
    string_statistics = np.column_stack(
        [
            unique_count,
            F27_WIDTH - unique_count,
            (characters[:, 1:] == characters[:, :-1]).sum(axis=1),
            characters[:, 0] == characters[:, -1],
        ]
    ).astype(np.float32)
    features.append(string_statistics)
    names.extend(
        [
            "f27_unique_count",
            "f27_repeat_count",
            "f27_adjacent_equal_count",
            "f27_first_last_equal",
        ]
    )

    by_name = {name: base[:, index] for index, name in enumerate(NUMERIC_COLUMNS)}
    sum_02_21 = by_name["f_02"] + by_name["f_21"]
    sum_05_22 = by_name["f_05"] + by_name["f_22"]
    sum_00_01_26 = by_name["f_00"] + by_name["f_01"] + by_name["f_26"]
    continuous_interactions = np.column_stack(
        [sum_02_21, sum_05_22, sum_00_01_26]
    ).astype(np.float32)
    signed_interactions = np.column_stack(
        [
            (sum_02_21 > 5.2).astype(np.float32)
            - (sum_02_21 < -5.3).astype(np.float32),
            (sum_05_22 > 5.1).astype(np.float32)
            - (sum_05_22 < -5.4).astype(np.float32),
            (sum_00_01_26 > 5.0).astype(np.float32)
            - (sum_00_01_26 < -5.0).astype(np.float32),
        ]
    ).astype(np.float32)
    features.extend([continuous_interactions, signed_interactions])
    names.extend(["sum_f02_f21", "sum_f05_f22", "sum_f00_f01_f26"])
    names.extend(
        [
            "interaction_f02_f21",
            "interaction_f05_f22",
            "interaction_f00_f01_f26",
        ]
    )
    matrix = np.ascontiguousarray(np.column_stack(features), dtype=np.float32)
    if matrix.shape[1] != len(names) or len(names) != 60:
        raise RuntimeError("May-2022 compact feature contract changed unexpectedly")
    if not np.isfinite(matrix).all():
        raise RuntimeError("May-2022 compact features contain non-finite values")
    return matrix, characters, names


def build_feature_contract(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    train_numeric, train_characters, train_names = build_numeric_features(train)
    test_numeric, test_characters, test_names = build_numeric_features(test)
    if train_names != test_names:
        raise RuntimeError("May-2022 compact train/test feature schemas differ")
    schema = {
        "numeric_feature_names": train_names,
        "numeric_feature_count": len(train_names),
        "character_positions": F27_WIDTH,
        "character_cardinality": len(F27_ALPHABET),
        "character_representation": "position_specific_embedding_without_ordinal_distance",
        "target_derived_features": 0,
        "id_features": 0,
        "published_interactions": {
            "sum_f02_f21": ["f_02", "+", "f_21"],
            "sum_f05_f22": ["f_05", "+", "f_22"],
            "sum_f00_f01_f26": ["f_00", "+", "f_01", "+", "f_26"],
        },
    }
    schema["feature_schema_sha256"] = sha256_json(schema)
    return train_numeric, train_characters, test_numeric, test_characters, schema


def build_outer_folds(target: np.ndarray, *, folds: int, seed: int) -> np.ndarray:
    from sklearn.model_selection import StratifiedKFold

    labels = np.asarray(target, dtype=np.int8)
    assignment = np.full(len(labels), -1, dtype=np.int16)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (_, validation) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        assignment[validation] = fold
    if np.any(assignment < 0):
        raise RuntimeError("May-2022 outer fold assignment did not cover every row")
    return assignment


def build_inner_split(
    outer_fit_indices: np.ndarray,
    target: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import StratifiedShuffleSplit

    outer_fit = np.asarray(outer_fit_indices, dtype=np.int64)
    labels = np.asarray(target, dtype=np.int8)
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=validation_fraction, random_state=seed
    )
    relative_fit, relative_validation = next(
        splitter.split(np.zeros(len(outer_fit)), labels[outer_fit])
    )
    inner_fit = outer_fit[relative_fit]
    inner_validation = outer_fit[relative_validation]
    if set(inner_fit.tolist()) & set(inner_validation.tolist()):
        raise RuntimeError("May-2022 inner fit and validation partitions overlap")
    if len(inner_fit) + len(inner_validation) != len(outer_fit):
        raise RuntimeError("May-2022 inner split did not cover the outer-fit rows")
    return inner_fit, inner_validation


def fold_mean_scale(matrix: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(matrix[indices], dtype=np.float64)
    mean = values.mean(axis=0).astype(np.float32)
    scale = values.std(axis=0).astype(np.float32)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    return mean, scale


def build_model(
    numeric_features: int,
    *,
    width: int,
    blocks: int,
    embedding_dim: int,
    dropout: float,
    mean: np.ndarray,
    scale: np.ndarray,
):
    import torch
    from torch import nn

    class ResidualBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(width)
            self.expand = nn.Linear(width, width * 2)
            self.contract = nn.Linear(width * 2, width)
            self.dropout = nn.Dropout(dropout)
            self.activation = nn.SiLU()

        def forward(self, values):
            hidden = self.expand(self.norm(values))
            hidden = self.activation(hidden)
            hidden = self.dropout(hidden)
            hidden = self.contract(hidden)
            return values + 0.5 * self.dropout(hidden)

    class CompactEmbeddingMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("numeric_mean", torch.as_tensor(mean, dtype=torch.float32))
            self.register_buffer("numeric_scale", torch.as_tensor(scale, dtype=torch.float32))
            self.character_embeddings = nn.ModuleList(
                nn.Embedding(len(F27_ALPHABET), embedding_dim) for _ in range(F27_WIDTH)
            )
            input_width = numeric_features + F27_WIDTH * embedding_dim
            self.stem = nn.Sequential(
                nn.Linear(input_width, width),
                nn.SiLU(),
                nn.LayerNorm(width),
            )
            self.blocks = nn.ModuleList(ResidualBlock() for _ in range(blocks))
            self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

        def forward(self, numeric, characters):
            normalized = torch.clamp(
                (numeric - self.numeric_mean) / self.numeric_scale, -10.0, 10.0
            )
            embedded = torch.cat(
                [layer(characters[:, position]) for position, layer in enumerate(self.character_embeddings)],
                dim=1,
            )
            hidden = self.stem(torch.cat([normalized, embedded], dim=1))
            for block in self.blocks:
                hidden = block(hidden)
            return self.head(hidden).squeeze(1)

    return CompactEmbeddingMLP()


def set_reproducible_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")


def _optimizer(model: Any, plan: dict[str, Any]):
    import torch

    kwargs = {
        "lr": float(plan["training"]["learning_rate"]),
        "weight_decay": float(plan["training"]["weight_decay"]),
    }
    try:
        return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(model.parameters(), **kwargs)


def _predict(
    model: Any,
    numeric_tensor: Any,
    character_tensor: Any,
    indices: np.ndarray,
    *,
    batch_size: int,
    amp_dtype: Any,
) -> np.ndarray:
    import torch

    model.eval()
    output: list[np.ndarray] = []
    index_tensor = torch.as_tensor(indices, dtype=torch.long, device="cuda")
    with torch.inference_mode():
        for start in range(0, len(index_tensor), batch_size):
            batch = index_tensor[start : start + batch_size]
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(numeric_tensor[batch], character_tensor[batch])
            output.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(output)


def _train_epochs(
    *,
    model: Any,
    numeric_tensor: Any,
    character_tensor: Any,
    target_tensor: Any,
    train_indices: np.ndarray,
    evaluation_indices: np.ndarray | None,
    epochs: int,
    scheduler_horizon: int,
    patience: int,
    seed: int,
    plan: dict[str, Any],
    heartbeat: callable,
    phase: str,
) -> tuple[Any, list[dict[str, Any]], int]:
    import torch
    from sklearn.metrics import roc_auc_score

    optimizer = _optimizer(model, plan)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=scheduler_horizon,
        eta_min=float(plan["training"]["learning_rate"]) * 0.05,
    )
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    batch_size = int(plan["training"]["batch_size"])
    generator = torch.Generator(device="cuda").manual_seed(seed)
    train_tensor = torch.as_tensor(train_indices, dtype=torch.long, device="cuda")
    best_auc = -np.inf
    best_epoch = epochs
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_tensor), generator=generator, device="cuda")
        shuffled = train_tensor[permutation]
        loss_sum = 0.0
        row_count = 0
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(numeric_tensor[batch], character_tensor[batch])
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, target_tensor[batch]
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(batch)
            row_count += len(batch)
        scheduler.step()
        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": loss_sum / max(row_count, 1),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if evaluation_indices is not None:
            probability = _predict(
                model,
                numeric_tensor,
                character_tensor,
                evaluation_indices,
                batch_size=batch_size * 4,
                amp_dtype=amp_dtype,
            )
            labels = target_tensor[
                torch.as_tensor(evaluation_indices, dtype=torch.long, device="cuda")
            ].cpu().numpy()
            evaluation_auc = float(roc_auc_score(labels, probability))
            record["inner_validation_auc"] = evaluation_auc
            if evaluation_auc > best_auc + 1e-7:
                best_auc = evaluation_auc
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
        history.append(record)
        heartbeat(phase, epoch, record)
        if evaluation_indices is not None and stale >= patience:
            break
    del optimizer, scheduler, scaler, train_tensor
    return model, history, int(best_epoch)


def load_plan(path: Path) -> tuple[dict[str, Any], str]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != "evomind.mlebench.may2022_compact_embedding_plan.v1":
        raise ValueError("May-2022 compact plan schema is invalid")
    if plan.get("competition_id") != COMPETITION_ID:
        raise ValueError("May-2022 compact plan competition is invalid")
    runner_hash = sha256_file(Path(__file__).resolve())
    if plan.get("runner_sha256") != runner_hash:
        raise RuntimeError("May-2022 compact runner changed after plan freeze")
    if plan.get("private_labels_allowed") is not False:
        raise RuntimeError("May-2022 compact plan must forbid private labels")
    if plan.get("official_grader_enabled") is not False:
        raise RuntimeError("May-2022 compact plan must disable official grading")
    return plan, sha256_file(path)


def validate_public_contract(
    train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame
) -> None:
    if len(train) != EXPECTED_TRAIN_ROWS or len(test) != EXPECTED_TEST_ROWS:
        raise RuntimeError(
            f"May-2022 full-data contract failed: train={len(train)} test={len(test)}"
        )
    for name, frame in (("train", train), ("test", test), ("sample", sample)):
        if "id" not in frame:
            raise RuntimeError(f"May-2022 {name} is missing id")
        if frame["id"].duplicated().any():
            raise RuntimeError(f"May-2022 {name} contains duplicate ids")
    if "target" not in train or "target" not in sample:
        raise RuntimeError("May-2022 train/sample target column is missing")
    if set(train["target"].dropna().astype(int).unique().tolist()) != {0, 1}:
        raise RuntimeError("May-2022 target is not binary")
    if not np.array_equal(test["id"].to_numpy(), sample["id"].to_numpy()):
        raise RuntimeError("May-2022 test/sample IDs are not exactly aligned")


def _fold_result_valid(
    metadata_path: Path,
    result_path: Path,
    *,
    plan_sha256: str,
    input_sha256: dict[str, str],
    fold: int,
    expected_validation: np.ndarray,
    expected_test_rows: int,
) -> bool:
    if not metadata_path.is_file() or not result_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("plan_sha256") != plan_sha256:
            return False
        if metadata.get("input_sha256") != input_sha256 or int(metadata.get("fold")) != fold:
            return False
        with np.load(result_path, allow_pickle=False) as result:
            validation = np.asarray(result["validation_indices"], dtype=np.int64)
            valid_probability = np.asarray(result["validation_probability"])
            test_probability = np.asarray(result["test_probability"])
        return bool(
            np.array_equal(validation, expected_validation)
            and len(valid_probability) == len(expected_validation)
            and len(test_probability) == expected_test_rows
            and np.isfinite(valid_probability).all()
            and np.isfinite(test_probability).all()
        )
    except Exception:
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-dir", type=Path, default=DEFAULT_PUBLIC_DIR)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id")
    return parser


def configure_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"may2022_compact_{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def main(argv: Sequence[str] | None = None) -> int:
    import torch
    from sklearn.metrics import roc_auc_score

    args = build_parser().parse_args(argv)
    plan_path = args.plan.resolve()
    plan, plan_sha256 = load_plan(plan_path)
    public_dir = args.public_dir.resolve()
    allowed = (
        PROJECT_ROOT / "workspace" / "local_gpu" / "mlebench_official_data"
    ).resolve()
    try:
        public_dir.relative_to(allowed)
    except ValueError as exc:
        raise ValueError("May-2022 public directory is outside the local data root") from exc
    paths = {
        "train": public_dir / "train.csv",
        "test": public_dir / "test.csv",
        "sample": public_dir / "sample_submission.csv",
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("May-2022 public CSV staging is incomplete")
    input_sha256 = {name: sha256_file(path) for name, path in paths.items()}
    expected_hashes = plan["public_input_sha256"]
    if input_sha256 != expected_hashes:
        raise RuntimeError("May-2022 public CSV hashes differ from the frozen plan")
    if not torch.cuda.is_available():
        raise RuntimeError("May-2022 compact runner requires CUDA")

    run_id = args.run_id or (
        f"local4060_may2022_compact_embedding_s{plan['seed']}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = (args.output_root.resolve() / "runs" / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_contract_path = run_dir / "run_contract.json"
    run_contract = {
        "schema": "evomind.mlebench.may2022_compact_run_contract.v1",
        "run_id": run_id,
        "competition_id": COMPETITION_ID,
        "plan_sha256": plan_sha256,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "input_sha256": input_sha256,
        "seed": int(plan["seed"]),
        "folds": int(plan["folds"]),
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
    }
    if run_contract_path.is_file():
        existing_contract = json.loads(run_contract_path.read_text(encoding="utf-8"))
        if existing_contract != run_contract:
            raise RuntimeError("May-2022 resume contract differs from the existing run")
    else:
        write_json_atomic(run_contract_path, run_contract)
    logger = configure_logger(run_dir)
    heartbeat_path = run_dir / "heartbeat.json"
    started = time.monotonic()
    write_json_atomic(
        heartbeat_path,
        {
            "schema": "evomind.mlebench.may2022_compact_heartbeat.v1",
            "created_at": now_iso(),
            "status": "loading_public_data",
            "pid": os.getpid(),
            "run_id": run_id,
            "plan_sha256": plan_sha256,
            "process_signals_sent": 0,
        },
    )
    train = pd.read_csv(paths["train"])
    test = pd.read_csv(paths["test"])
    sample = pd.read_csv(paths["sample"])
    validate_public_contract(train, test, sample)
    target = train["target"].to_numpy(dtype=np.int8)
    train_numeric, train_characters, test_numeric, test_characters, feature_contract = (
        build_feature_contract(train, test)
    )
    seed = int(plan["seed"])
    folds = int(plan["folds"])
    assignment = build_outer_folds(target, folds=folds, seed=seed)
    write_npz_atomic(
        run_dir / "fold_manifest.npz",
        id=train["id"].to_numpy(),
        target=target,
        fold_assignment=assignment,
        seed=np.asarray([seed], dtype=np.int64),
    )
    device = torch.device("cuda:0")
    train_numeric_tensor = torch.as_tensor(train_numeric, dtype=torch.float32, device=device)
    train_character_tensor = torch.as_tensor(train_characters, dtype=torch.long, device=device)
    test_numeric_tensor = torch.as_tensor(test_numeric, dtype=torch.float32, device=device)
    test_character_tensor = torch.as_tensor(test_characters, dtype=torch.long, device=device)
    target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device)
    oof_probability = np.full(len(train), np.nan, dtype=np.float64)
    test_probability_sum = np.zeros(len(test), dtype=np.float64)
    fold_records: list[dict[str, Any]] = []

    def heartbeat(fold: int, phase: str, epoch: int, record: dict[str, Any]) -> None:
        write_json_atomic(
            heartbeat_path,
            {
                "schema": "evomind.mlebench.may2022_compact_heartbeat.v1",
                "created_at": now_iso(),
                "status": "training",
                "pid": os.getpid(),
                "run_id": run_id,
                "fold": fold,
                "phase": phase,
                "epoch": epoch,
                "epoch_record": record,
                "completed_folds": len(fold_records),
                "total_folds": folds,
                "elapsed_seconds": time.monotonic() - started,
                "gpu_name": torch.cuda.get_device_name(0),
                "process_signals_sent": 0,
            },
        )

    for fold in range(folds):
        outer_validation = np.flatnonzero(assignment == fold)
        outer_fit = np.flatnonzero(assignment != fold)
        fold_dir = run_dir / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        result_path = fold_dir / "result.npz"
        metadata_path = fold_dir / "metadata.json"
        if _fold_result_valid(
            metadata_path,
            result_path,
            plan_sha256=plan_sha256,
            input_sha256=input_sha256,
            fold=fold,
            expected_validation=outer_validation,
            expected_test_rows=len(test),
        ):
            with np.load(result_path, allow_pickle=False) as result:
                valid_probability = np.asarray(result["validation_probability"], dtype=np.float64)
                fold_test_probability = np.asarray(result["test_probability"], dtype=np.float64)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            fold_record = metadata["fold_record"]
            logger.info("fold=%d resumed from verified artifacts", fold)
        else:
            fold_seed = seed * 10_000 + fold * 101 + 17
            inner_fit, inner_validation = build_inner_split(
                outer_fit,
                target,
                validation_fraction=float(plan["training"]["inner_validation_fraction"]),
                seed=fold_seed,
            )
            inner_mean, inner_scale = fold_mean_scale(train_numeric, inner_fit)
            set_reproducible_seed(fold_seed)
            selection_model = build_model(
                train_numeric.shape[1],
                width=int(plan["architecture"]["width"]),
                blocks=int(plan["architecture"]["blocks"]),
                embedding_dim=int(plan["architecture"]["embedding_dim"]),
                dropout=float(plan["architecture"]["dropout"]),
                mean=inner_mean,
                scale=inner_scale,
            ).cuda()
            selection_model, selection_history, selected_epoch = _train_epochs(
                model=selection_model,
                numeric_tensor=train_numeric_tensor,
                character_tensor=train_character_tensor,
                target_tensor=target_tensor,
                train_indices=inner_fit,
                evaluation_indices=inner_validation,
                epochs=int(plan["training"]["max_selection_epochs"]),
                scheduler_horizon=int(plan["training"]["max_selection_epochs"]),
                patience=int(plan["training"]["patience"]),
                seed=fold_seed,
                plan=plan,
                heartbeat=lambda phase, epoch, record, f=fold: heartbeat(f, phase, epoch, record),
                phase="inner_epoch_selection",
            )
            del selection_model
            torch.cuda.empty_cache()
            gc.collect()

            outer_mean, outer_scale = fold_mean_scale(train_numeric, outer_fit)
            refit_seed = fold_seed + 1_000_003
            set_reproducible_seed(refit_seed)
            model = build_model(
                train_numeric.shape[1],
                width=int(plan["architecture"]["width"]),
                blocks=int(plan["architecture"]["blocks"]),
                embedding_dim=int(plan["architecture"]["embedding_dim"]),
                dropout=float(plan["architecture"]["dropout"]),
                mean=outer_mean,
                scale=outer_scale,
            ).cuda()
            model, refit_history, _ = _train_epochs(
                model=model,
                numeric_tensor=train_numeric_tensor,
                character_tensor=train_character_tensor,
                target_tensor=target_tensor,
                train_indices=outer_fit,
                evaluation_indices=None,
                epochs=selected_epoch,
                scheduler_horizon=int(plan["training"]["max_selection_epochs"]),
                patience=int(plan["training"]["patience"]),
                seed=refit_seed,
                plan=plan,
                heartbeat=lambda phase, epoch, record, f=fold: heartbeat(f, phase, epoch, record),
                phase="outer_fixed_budget_refit",
            )
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            valid_probability = _predict(
                model,
                train_numeric_tensor,
                train_character_tensor,
                outer_validation,
                batch_size=int(plan["training"]["batch_size"]) * 4,
                amp_dtype=amp_dtype,
            )
            fold_test_probability = _predict(
                model,
                test_numeric_tensor,
                test_character_tensor,
                np.arange(len(test), dtype=np.int64),
                batch_size=int(plan["training"]["batch_size"]) * 4,
                amp_dtype=amp_dtype,
            )
            outer_auc = float(roc_auc_score(target[outer_validation], valid_probability))
            checkpoint = {
                "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "fold": fold,
                "seed": seed,
                "selected_epoch": selected_epoch,
                "plan_sha256": plan_sha256,
                "feature_schema_sha256": feature_contract["feature_schema_sha256"],
            }
            checkpoint_tmp = fold_dir / "model.pt.tmp"
            torch.save(checkpoint, checkpoint_tmp)
            checkpoint_tmp.replace(fold_dir / "model.pt")
            fold_record = {
                "fold": fold,
                "outer_fit_rows": len(outer_fit),
                "outer_validation_rows": len(outer_validation),
                "inner_fit_rows": len(inner_fit),
                "inner_validation_rows": len(inner_validation),
                "selected_epoch": selected_epoch,
                "inner_best_auc": max(
                    float(item["inner_validation_auc"]) for item in selection_history
                ),
                "outer_auc": outer_auc,
                "selection_history": selection_history,
                "refit_history": refit_history,
                "outer_validation_used_for_epoch_selection": False,
                "outer_refit_fixed_budget": True,
            }
            write_npz_atomic(
                result_path,
                validation_indices=outer_validation,
                validation_probability=valid_probability.astype(np.float32),
                test_probability=fold_test_probability.astype(np.float32),
            )
            write_json_atomic(
                metadata_path,
                {
                    "schema": "evomind.mlebench.may2022_compact_fold.v1",
                    "created_at": now_iso(),
                    "fold": fold,
                    "plan_sha256": plan_sha256,
                    "input_sha256": input_sha256,
                    "result_sha256": sha256_file(result_path),
                    "checkpoint_sha256": sha256_file(fold_dir / "model.pt"),
                    "fold_record": fold_record,
                    "private_labels_used": False,
                    "official_grader_executed": False,
                    "process_signals_sent": 0,
                },
            )
            del model
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(
                "fold=%d selected_epoch=%d outer_auc=%.8f",
                fold,
                selected_epoch,
                outer_auc,
            )
        oof_probability[outer_validation] = valid_probability
        test_probability_sum += fold_test_probability / folds
        fold_records.append(fold_record)

    if not np.isfinite(oof_probability).all():
        raise RuntimeError("May-2022 compact OOF coverage is incomplete")
    aggregate_auc = float(roc_auc_score(target, oof_probability))
    fold_auc = [float(record["outer_auc"]) for record in fold_records]
    single_seed_passed = bool(
        aggregate_auc >= AGGREGATE_PROMOTION_AUC
        and min(fold_auc) >= EVERY_FOLD_PROMOTION_AUC
    )
    candidate = sample.copy()
    candidate["target"] = test_probability_sum
    candidate_path = run_dir / "candidate_submission_withheld.csv"
    candidate.to_csv(candidate_path, index=False)
    bundle_path = run_dir / "may2022_compact_oof_bundle.npz"
    write_npz_atomic(
        bundle_path,
        id=train["id"].to_numpy(),
        target=target,
        fold_assignment=assignment,
        oof_probability=oof_probability.astype(np.float32),
        test_id=test["id"].to_numpy(),
        test_probability=test_probability_sum.astype(np.float32),
    )
    summary = {
        "schema": "evomind.mlebench.may2022_compact_embedding_run.v1",
        "created_at": now_iso(),
        "status": "single_seed_gate_passed" if single_seed_passed else "single_seed_gate_failed",
        "run_id": run_id,
        "competition_id": COMPETITION_ID,
        "seed": seed,
        "folds": folds,
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha256,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "input_sha256": input_sha256,
        "feature_contract": feature_contract,
        "train_rows": len(train),
        "test_rows": len(test),
        "fold_records": fold_records,
        "exact_deployment_oof_auc": aggregate_auc,
        "minimum_fold_auc": min(fold_auc),
        "maximum_fold_auc": max(fold_auc),
        "single_seed_gate": {
            "aggregate_threshold": AGGREGATE_PROMOTION_AUC,
            "every_fold_threshold": EVERY_FOLD_PROMOTION_AUC,
            "aggregate_passed": aggregate_auc >= AGGREGATE_PROMOTION_AUC,
            "all_folds_passed": min(fold_auc) >= EVERY_FOLD_PROMOTION_AUC,
            "passed": single_seed_passed,
        },
        "multi_seed_confirmation": {
            "required_seeds": [42, 43, 44],
            "passed": False,
            "pending": True,
        },
        "bundle_path": str(bundle_path),
        "bundle_sha256": sha256_file(bundle_path),
        "candidate_submission_path": str(candidate_path),
        "candidate_submission_sha256": sha256_file(candidate_path),
        "submission_withheld": True,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "mixed_precision": "bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        },
        "elapsed_seconds": time.monotonic() - started,
        "claim_boundary": "Public OOF evidence is not an official medal.",
    }
    summary_path = run_dir / "summary.json"
    write_json_atomic(summary_path, summary)
    write_json_atomic(
        heartbeat_path,
        {
            "schema": "evomind.mlebench.may2022_compact_heartbeat.v1",
            "created_at": now_iso(),
            "status": summary["status"],
            "pid": os.getpid(),
            "run_id": run_id,
            "completed_folds": folds,
            "total_folds": folds,
            "exact_deployment_oof_auc": aggregate_auc,
            "summary_path": str(summary_path),
            "process_signals_sent": 0,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
