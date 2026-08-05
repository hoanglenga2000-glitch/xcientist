#!/usr/bin/env python3
"""CPU-only frozen Leaf decision-logit confirmation and withheld candidate builder."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPETITION_ID = "leaf-classification"
SCHEMA = "evomind.leaf.decision_logit_confirmation.v1"
ARTIFACT_SCHEMA = "evomind.leaf.decision_logit_confirmation_artifacts.v1"
INPUT_SCHEMA = "evomind.leaf.decision_logit_confirmation_inputs.v1"
SOURCE_SCHEMA = "evomind.leaf.decision_logit_confirmation_sources.v1"

CONFIRMATION_SEEDS = (43, 44, 45)
FOLD_COUNT = 5
BACKBONES = ("convnext_small", "efficientnet_v2_s")
BACKBONE_WIDTHS = {"convnext_small": 768, "efficientnet_v2_s": 1280}
COMPONENT_NAME = "multimodal_group_balanced_rbf_svc_decision_logit"
SVC_C = 10.0
SVC_GAMMA = "scale"
TEMPERATURE = 0.15
LOG_LOSS_THRESHOLD = 0.0135
FOLD_AGGREGATION = "arithmetic_mean_probabilities"
SEED_AGGREGATION = "arithmetic_mean_probabilities"

DEFAULT_DATA_ROOT = PROJECT_ROOT / "workspace" / "local_gpu" / "mlebench_official_data"
DEFAULT_REFERENCE_REPORT = (
    PROJECT_ROOT
    / "workspace"
    / "local_gpu"
    / "leaf_multibackbone"
    / "leaf_multibackbone_current.json"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "workspace" / "cpu" / "leaf_decision_logit_confirmation"
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    record = {
        "path": (
            path.relative_to(relative_to.resolve()).as_posix()
            if relative_to is not None
            else str(path)
        ),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    return record


def portable_unicode_array(values: Sequence[Any]) -> np.ndarray:
    strings = [str(value) for value in values]
    width = max(1, max((len(value) for value in strings), default=1))
    result = np.asarray(strings, dtype=f"<U{width}")
    if result.dtype.kind != "U" or result.dtype.hasobject:
        raise RuntimeError("Leaf confirmation strings must use fixed-width Unicode")
    return result


def canonical_id(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError("Leaf ID is non-finite")
        if float(value).is_integer():
            return str(int(value))
    text = str(value).strip()
    if not text:
        raise ValueError("Leaf ID is empty")
    return text


def portable_reference_basename(value: Any) -> str:
    """Return a basename from a frozen path produced on Windows or POSIX."""

    text = str(value).strip()
    name = (
        PureWindowsPath(text).name
        if "\\" in text
        else PurePosixPath(text).name
    )
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("Leaf frozen reference path has no portable basename")
    return name


def stable_softmax(logits: np.ndarray, *, temperature: float = TEMPERATURE) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Leaf decision logits must be a finite matrix")
    if temperature != TEMPERATURE:
        raise ValueError(f"Leaf confirmation temperature is frozen at {TEMPERATURE}")
    scaled = values / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    exponential = np.exp(scaled)
    probability = exponential / exponential.sum(axis=1, keepdims=True)
    if not np.isfinite(probability).all() or not np.allclose(
        probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("Leaf stable softmax produced invalid probabilities")
    return probability


def multiclass_log_loss(
    truth: Sequence[Any], probability: np.ndarray, classes: Sequence[str]
) -> float:
    values = np.asarray(probability, dtype=np.float64)
    ordered = [str(value) for value in classes]
    class_index = {value: index for index, value in enumerate(ordered)}
    if len(class_index) != len(ordered):
        raise ValueError("Leaf classes are duplicated")
    try:
        target = np.asarray([class_index[str(value)] for value in truth], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Leaf truth contains an unknown class: {exc}") from exc
    if values.shape != (len(target), len(ordered)):
        raise ValueError("Leaf probability shape differs from truth/classes")
    selected = np.clip(values[np.arange(len(target)), target], 1e-15, 1.0)
    return float(-np.log(selected).mean())


def aligned_decision_logits(
    model: Any, features: np.ndarray, classes: Sequence[str]
) -> np.ndarray:
    raw = np.asarray(model.decision_function(features), dtype=np.float64)
    model_classes = [str(value) for value in model.classes_]
    ordered = [str(value) for value in classes]
    if raw.ndim != 2 or set(model_classes) != set(ordered):
        raise RuntimeError("Leaf decision-logit classes differ from sample columns")
    class_index = {value: index for index, value in enumerate(model_classes)}
    aligned = np.column_stack([raw[:, class_index[value]] for value in ordered])
    if not np.isfinite(aligned).all():
        raise RuntimeError("Leaf aligned decision logits are non-finite")
    return aligned


def fixed_contract() -> dict[str, Any]:
    contract = {
        "schema": "evomind.leaf.decision_logit_confirmation_contract.v1",
        "competition_id": COMPETITION_ID,
        "device": "cpu_only",
        "seeds": list(CONFIRMATION_SEEDS),
        "development_seeds_excluded": [40, 41, 42],
        "folds": FOLD_COUNT,
        "backbones": list(BACKBONES),
        "component": COMPONENT_NAME,
        "svc": {
            "C": SVC_C,
            "gamma": SVC_GAMMA,
            "kernel": "rbf",
            "probability": False,
            "decision_function_shape": "ovr",
        },
        "calibration": {
            "source": "decision_function",
            "function": "stable_softmax",
            "temperature": TEMPERATURE,
            "frozen_before_confirmation": True,
            "confirmation_seed_tuning_allowed": False,
        },
        "fold_aggregation": FOLD_AGGREGATION,
        "seed_aggregation": SEED_AGGREGATION,
        "gate": {
            "per_seed_log_loss_max": LOG_LOSS_THRESHOLD,
            "mean_seed_log_loss_max": LOG_LOSS_THRESHOLD,
            "maximum_seed_log_loss_max": LOG_LOSS_THRESHOLD,
            "ensemble_log_loss_max": LOG_LOSS_THRESHOLD,
        },
        "candidate_only": True,
        "automatic_official_grader": False,
        "automatic_kaggle_submission": False,
    }
    contract["contract_sha256"] = sha256_json(contract)
    return contract


def evaluate_gate(seed_scores: Sequence[float], ensemble_score: float) -> dict[str, Any]:
    scores = [float(value) for value in seed_scores]
    if len(scores) != len(CONFIRMATION_SEEDS) or not np.isfinite(scores).all():
        raise ValueError("Leaf confirmation gate requires three finite seed scores")
    mean_score = float(np.mean(scores))
    maximum_score = float(np.max(scores))
    checks = {
        "all_per_seed_at_or_below_threshold": all(
            score <= LOG_LOSS_THRESHOLD for score in scores
        ),
        "mean_seed_log_loss_at_or_below_threshold": mean_score
        <= LOG_LOSS_THRESHOLD,
        "maximum_seed_log_loss_at_or_below_threshold": maximum_score
        <= LOG_LOSS_THRESHOLD,
        "ensemble_log_loss_at_or_below_threshold": float(ensemble_score)
        <= LOG_LOSS_THRESHOLD,
    }
    return {
        "passed": all(checks.values()),
        "threshold": LOG_LOSS_THRESHOLD,
        "seed_scores": scores,
        "mean_seed_log_loss": mean_score,
        "maximum_seed_log_loss": maximum_score,
        "ensemble_log_loss": float(ensemble_score),
        "checks": checks,
    }


def dependency_versions() -> dict[str, str]:
    names = {
        "joblib": "joblib",
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit_learn": "scikit-learn",
        "scipy": "scipy",
    }
    versions: dict[str, str] = {}
    for key, distribution in names.items():
        try:
            versions[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = "missing"
    return versions


def source_manifest() -> dict[str, Any]:
    paths = [
        Path(__file__).resolve(),
        PROJECT_ROOT / "scripts" / "verify_leaf_decision_logit_confirmation.py",
        PROJECT_ROOT / "scripts" / "run_leaf_multibackbone_oof.py",
        PROJECT_ROOT / "scripts" / "mlebench_medal_recovery_adapters.py",
        PROJECT_ROOT / "scripts" / "mlebench_wave2_adapters.py",
        PROJECT_ROOT / "scripts" / "russian_transliteration.py",
        PROJECT_ROOT / "src" / "research_os" / "mlebench_phase_a.py",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Leaf confirmation source dependencies are missing: {missing}")
    return {
        "schema": SOURCE_SCHEMA,
        "created_at": now_iso(),
        "files": [file_record(path) for path in paths],
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "packages": dependency_versions(),
        },
    }


def resolve_public_dir(data_root: Path) -> Path:
    public_dir = Path(data_root).resolve() / COMPETITION_ID / "prepared" / "public"
    required = [
        public_dir / "train.csv",
        public_dir / "test.csv",
        public_dir / "sample_submission.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Leaf public inputs are missing: {missing}")
    return public_dir


def resolve_cache_records(
    reference: dict[str, Any],
    *,
    cache_dir: Path | None,
    expected_rows: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    metadata_by_backbone = reference.get("embedding_metadata")
    if not isinstance(metadata_by_backbone, dict) or set(metadata_by_backbone) != set(
        BACKBONES
    ):
        raise RuntimeError("Leaf reference report does not contain both frozen backbones")
    arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    for backbone in BACKBONES:
        embedded = metadata_by_backbone[backbone]
        original_cache = Path(str(embedded["cache_path"]))
        original_metadata = Path(str(embedded["metadata_path"]))
        resolved_cache = (
            Path(cache_dir).resolve()
            / portable_reference_basename(embedded["cache_path"])
            if cache_dir is not None
            else original_cache.resolve()
        )
        resolved_metadata = (
            Path(cache_dir).resolve()
            / portable_reference_basename(embedded["metadata_path"])
            if cache_dir is not None
            else original_metadata.resolve()
        )
        if not resolved_cache.is_file() or not resolved_metadata.is_file():
            raise FileNotFoundError(f"Leaf embedding cache is missing for {backbone}")
        metadata = read_json(resolved_metadata)
        if metadata.get("cache_sha256") != sha256_file(resolved_cache):
            raise RuntimeError(f"Leaf embedding cache hash differs for {backbone}")
        if embedded.get("cache_sha256") != metadata.get("cache_sha256"):
            raise RuntimeError(f"Leaf reference/cache metadata hash differs for {backbone}")
        contract = metadata.get("contract") or {}
        if contract.get("backbone") != backbone:
            raise RuntimeError(f"Leaf embedding cache backbone differs: {backbone}")
        if contract.get("private_labels_used") is not False:
            raise RuntimeError("Leaf embedding cache does not preserve the public-only contract")
        array = np.load(resolved_cache, mmap_mode="r", allow_pickle=False)
        if array.dtype.hasobject or array.ndim != 2 or len(array) != expected_rows:
            raise RuntimeError(f"Leaf embedding cache shape/dtype differs for {backbone}")
        if not np.isfinite(array).all():
            raise RuntimeError(f"Leaf embedding cache is non-finite for {backbone}")
        arrays[backbone] = array
        records.append(
            {
                "backbone": backbone,
                "cache": file_record(resolved_cache),
                "metadata": file_record(resolved_metadata),
                "contract": contract,
                "pretrained_weight_identity": metadata.get(
                    "pretrained_weight_identity"
                ),
                "rows": int(array.shape[0]),
                "width": int(array.shape[1]),
                "dtype": str(array.dtype),
            }
        )
    return arrays, records


def validate_reference_manifest(
    reference: dict[str, Any],
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    override_path: Path | None = None,
) -> tuple[Path, pd.DataFrame]:
    manifest_path = (
        Path(override_path).resolve()
        if override_path is not None
        else Path(str(reference["manifest_path"])).resolve()
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Leaf reference image manifest is missing: {manifest_path}")
    if sha256_file(manifest_path) != reference.get("manifest_sha256"):
        raise RuntimeError("Leaf reference image manifest hash differs")
    manifest = pd.read_csv(manifest_path)
    required = {"split", "index", "id", "sha256"}
    if not required <= set(manifest) or len(manifest) != len(train) + len(test):
        raise RuntimeError("Leaf reference image manifest schema/rows differ")
    train_manifest = manifest.iloc[: len(train)].reset_index(drop=True)
    test_manifest = manifest.iloc[len(train) :].reset_index(drop=True)
    if train_manifest["split"].tolist() != ["train"] * len(train):
        raise RuntimeError("Leaf reference train-manifest split differs")
    if test_manifest["split"].tolist() != ["test"] * len(test):
        raise RuntimeError("Leaf reference test-manifest split differs")
    if [canonical_id(value) for value in train_manifest["id"]] != [
        canonical_id(value) for value in train["id"]
    ]:
        raise RuntimeError("Leaf reference train-manifest IDs differ")
    if [canonical_id(value) for value in test_manifest["id"]] != [
        canonical_id(value) for value in test["id"]
    ]:
        raise RuntimeError("Leaf reference test-manifest IDs differ")
    return manifest_path, manifest


def build_component_model(seed: int) -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import Normalizer, StandardScaler
    from sklearn.svm import SVC

    numeric_width = 192
    embedding_widths = tuple(BACKBONE_WIDTHS[value] for value in BACKBONES)
    cursor = numeric_width
    transformers: list[tuple[str, Any, slice]] = [
        (
            "numeric",
            Pipeline([("scale", StandardScaler()), ("normalize", Normalizer())]),
            slice(0, numeric_width),
        )
    ]
    for backbone, width in zip(BACKBONES, embedding_widths, strict=True):
        transformers.append((backbone, Normalizer(), slice(cursor, cursor + width)))
        cursor += width
    preprocessor = ColumnTransformer(
        transformers,
        transformer_weights={name: 1.0 for name, _, _ in transformers},
        sparse_threshold=0.0,
    )
    model = SVC(
        C=SVC_C,
        gamma=SVC_GAMMA,
        kernel="rbf",
        probability=False,
        decision_function_shape="ovr",
        random_state=int(seed),
        cache_size=4096,
    )
    return Pipeline([("groups", preprocessor), ("model", model)])


def build_folds(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    classes: Sequence[str],
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, list[dict[str, Any]]]:
    target_values = np.asarray(target)
    group_values = np.asarray(groups)
    if target_values.ndim != 1 or group_values.ndim != 1:
        raise ValueError("Leaf target/groups must be one-dimensional")
    if len(features) != len(target_values) or len(group_values) != len(target_values):
        raise ValueError("Leaf features/target/groups rows differ")

    ordered_classes = [str(value) for value in classes]
    expected_classes = set(ordered_classes)
    if len(expected_classes) != len(ordered_classes):
        raise ValueError("Leaf fold classes are duplicated")
    target_strings = np.asarray([str(value) for value in target_values])
    group_strings = np.asarray([str(value).strip() for value in group_values])
    if set(target_strings) != expected_classes:
        raise RuntimeError("Leaf fold targets differ from the frozen classes")
    if any(not value for value in group_strings):
        raise RuntimeError("Leaf fold group is empty")

    group_rows: dict[str, list[int]] = {}
    group_class: dict[str, str] = {}
    class_groups: dict[str, list[str]] = {value: [] for value in ordered_classes}
    for row, (class_name, group_name) in enumerate(
        zip(target_strings, group_strings, strict=True)
    ):
        previous = group_class.setdefault(group_name, class_name)
        if previous != class_name:
            raise RuntimeError("Leaf image group spans target classes")
        group_rows.setdefault(group_name, []).append(row)
    for group_name, class_name in group_class.items():
        class_groups[class_name].append(group_name)
    for class_name, class_group_names in class_groups.items():
        if len(class_group_names) < FOLD_COUNT:
            raise RuntimeError(
                f"Leaf class {class_name} has fewer than {FOLD_COUNT} distinct groups"
            )

    assignment = np.full(len(target), -1, dtype=np.int16)
    fold_rows = np.zeros(FOLD_COUNT, dtype=np.int64)
    for class_name in ordered_classes:
        class_fold_rows = np.zeros(FOLD_COUNT, dtype=np.int64)
        class_group_names = sorted(
            class_groups[class_name],
            key=lambda group_name: (
                -len(group_rows[group_name]),
                hashlib.sha256(
                    f"leaf-fold-v1\0{seed}\0{class_name}\0{group_name}".encode()
                ).digest(),
                group_name,
            ),
        )
        fold_priority = sorted(
            range(FOLD_COUNT),
            key=lambda fold: (
                hashlib.sha256(
                    f"leaf-fold-v1\0{seed}\0{class_name}\0fold\0{fold}".encode()
                ).digest(),
                fold,
            ),
        )
        for position, group_name in enumerate(class_group_names):
            rows = np.asarray(group_rows[group_name], dtype=np.int64)
            if position < FOLD_COUNT:
                fold = fold_priority[position]
            else:
                fold = min(
                    range(FOLD_COUNT),
                    key=lambda candidate: (
                        int(class_fold_rows[candidate]),
                        int(fold_rows[candidate]),
                        hashlib.sha256(
                            (
                                f"leaf-fold-v1\0{seed}\0{class_name}\0"
                                f"{group_name}\0{candidate}"
                            ).encode()
                        ).digest(),
                        candidate,
                    ),
                )
            if np.any(assignment[rows] >= 0):
                raise RuntimeError("Leaf group rows were assigned more than once")
            assignment[rows] = fold
            class_fold_rows[fold] += len(rows)
            fold_rows[fold] += len(rows)

    if np.any(assignment < 0):
        raise RuntimeError("Leaf confirmation folds did not cover every row")
    splits = [
        (
            np.flatnonzero(assignment != fold),
            np.flatnonzero(assignment == fold),
        )
        for fold in range(FOLD_COUNT)
    ]
    records: list[dict[str, Any]] = []
    write_count = np.zeros(len(target_values), dtype=np.uint8)
    for fold, (fit_indices, valid_indices) in enumerate(splits):
        fit_classes = set(target_strings[fit_indices])
        valid_classes = set(target_strings[valid_indices])
        fit_groups = set(group_strings[fit_indices])
        valid_groups = set(group_strings[valid_indices])
        overlap = fit_groups & valid_groups
        if fit_classes != expected_classes or valid_classes != expected_classes:
            raise RuntimeError(
                f"Leaf confirmation fold {seed}/{fold} is not scoreable for every class"
            )
        if overlap:
            raise RuntimeError(f"Leaf confirmation group crossed fold {seed}/{fold}")
        write_count[valid_indices] += 1
        records.append(
            {
                "fold": fold,
                "fit_rows": len(fit_indices),
                "valid_rows": len(valid_indices),
                "fit_class_count": len(fit_classes),
                "valid_class_count": len(valid_classes),
                "fit_group_count": len(fit_groups),
                "valid_group_count": len(valid_groups),
                "group_overlap_count": 0,
            }
        )
    if not np.all(write_count == 1):
        raise RuntimeError("Leaf confirmation OOF rows were not assigned exactly once")
    return splits, assignment, records


def align_submission(
    sample: pd.DataFrame,
    test_ids: Sequence[Any],
    probability: np.ndarray,
    classes: Sequence[str],
) -> pd.DataFrame:
    ordered_classes = [str(value) for value in classes]
    expected_columns = ["id", *ordered_classes]
    if sample.columns.tolist() != expected_columns:
        raise RuntimeError("Leaf sample-submission class order differs")
    test_keys = [canonical_id(value) for value in test_ids]
    sample_keys = [canonical_id(value) for value in sample["id"]]
    if len(set(test_keys)) != len(test_keys) or set(test_keys) != set(sample_keys):
        raise RuntimeError("Leaf test/sample ID contract differs")
    if probability.shape != (len(test_keys), len(ordered_classes)):
        raise RuntimeError("Leaf test probability shape differs")
    by_id = {key: probability[index] for index, key in enumerate(test_keys)}
    aligned = np.vstack([by_id[key] for key in sample_keys])
    result = sample.copy()
    result.loc[:, ordered_classes] = aligned
    return result


def build_artifact_manifest(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    artifacts_root = run_dir / "artifacts"
    paths = sorted(path for path in artifacts_root.rglob("*") if path.is_file())
    records = [file_record(path, relative_to=run_dir) for path in paths]
    expected = [str(record["path"]) for record in records]
    required = expected_artifact_paths()
    if expected != required:
        raise RuntimeError(
            f"Leaf confirmation artifact inventory differs: expected={required} actual={expected}"
        )
    return {
        "schema": ARTIFACT_SCHEMA,
        "created_at": now_iso(),
        "artifact_root": "artifacts",
        "expected_paths": expected,
        "artifact_count": len(records),
        "artifacts": records,
        "exact_inventory_required": True,
    }


def expected_artifact_paths() -> list[str]:
    paths = [
        "artifacts/confirmation_report.json",
        "artifacts/ensemble_oof_and_test.npz",
        "artifacts/input_manifest.json",
        "artifacts/source_manifest.json",
        "artifacts/submission_withheld.csv",
    ]
    paths.extend(
        f"artifacts/models/seed_{seed}_fold_{fold}.joblib"
        for seed in CONFIRMATION_SEEDS
        for fold in range(FOLD_COUNT)
    )
    paths.extend(
        f"artifacts/seeds/seed_{seed}.npz" for seed in CONFIRMATION_SEEDS
    )
    return sorted(paths)


def run_confirmation(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    import joblib

    started = time.perf_counter()
    contract = fixed_contract()
    public_dir = resolve_public_dir(args.data_root)
    train_path = public_dir / "train.csv"
    test_path = public_dir / "test.csv"
    sample_path = public_dir / "sample_submission.csv"
    train = pd.read_csv(train_path).reset_index(drop=True)
    test = pd.read_csv(test_path).reset_index(drop=True)
    sample = pd.read_csv(sample_path).reset_index(drop=True)
    if train.columns[:2].tolist() != ["id", "species"] or "id" not in test:
        raise RuntimeError("Leaf public CSV schema differs")
    if train["id"].duplicated().any() or test["id"].duplicated().any():
        raise RuntimeError("Leaf public IDs are duplicated")
    feature_columns = [value for value in train if value not in {"id", "species"}]
    if len(feature_columns) != 192 or test.columns.tolist() != ["id", *feature_columns]:
        raise RuntimeError("Leaf numeric feature schema differs from the frozen 192-column contract")
    numeric_train = train[feature_columns].to_numpy(dtype=np.float64)
    numeric_test = test[feature_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_train).all() or not np.isfinite(numeric_test).all():
        raise RuntimeError("Leaf numeric features are non-finite")
    classes = [str(value) for value in sample.columns[1:]]
    target = portable_unicode_array(train["species"].astype(str).tolist())
    if len(classes) != 99 or set(target) != set(classes):
        raise RuntimeError("Leaf target/sample class contract differs")

    reference_path = Path(args.reference_report).resolve()
    reference = read_json(reference_path)
    if reference.get("competition_id") != COMPETITION_ID:
        raise RuntimeError("Leaf reference report competition differs")
    manifest_path, image_manifest = validate_reference_manifest(
        reference,
        train,
        test,
        override_path=args.reference_image_manifest,
    )
    cache_arrays, cache_records = resolve_cache_records(
        reference,
        cache_dir=args.embedding_cache_dir,
        expected_rows=len(train) + len(test),
    )
    embedding_train = [
        np.asarray(cache_arrays[backbone][: len(train)], dtype=np.float32)
        for backbone in BACKBONES
    ]
    embedding_test = [
        np.asarray(cache_arrays[backbone][len(train) :], dtype=np.float32)
        for backbone in BACKBONES
    ]
    if [value.shape[1] for value in embedding_train] != [
        BACKBONE_WIDTHS[value] for value in BACKBONES
    ]:
        raise RuntimeError("Leaf embedding widths differ from the frozen contract")
    train_features = np.concatenate([numeric_train, *embedding_train], axis=1)
    test_features = np.concatenate([numeric_test, *embedding_test], axis=1)
    groups = image_manifest.iloc[: len(train)]["sha256"].astype(str).to_numpy()

    run_id = args.run_id or f"leaf_decision_logit_s434445_{datetime.now():%Y%m%d_%H%M%S}"
    output_root = Path(args.output_root).resolve()
    run_dir = output_root / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"Leaf confirmation run already exists: {run_dir}")
    artifacts_dir = run_dir / "artifacts"
    model_dir = artifacts_dir / "models"
    seed_dir = artifacts_dir / "seeds"
    model_dir.mkdir(parents=True, exist_ok=False)
    seed_dir.mkdir(parents=True, exist_ok=False)

    sources = source_manifest()
    source_path = artifacts_dir / "source_manifest.json"
    write_json_atomic(source_path, sources)
    input_manifest = {
        "schema": INPUT_SCHEMA,
        "created_at": now_iso(),
        "competition_id": COMPETITION_ID,
        "public_dir": str(public_dir),
        "public_files": {
            "train": file_record(train_path),
            "test": file_record(test_path),
            "sample_submission": file_record(sample_path),
        },
        "reference_report": file_record(reference_path),
        "reference_image_manifest": file_record(manifest_path),
        "embedding_caches": cache_records,
        "train_rows": len(train),
        "test_rows": len(test),
        "feature_columns": feature_columns,
        "feature_columns_sha256": sha256_json(feature_columns),
        "classes": classes,
        "classes_sha256": sha256_json(classes),
        "train_ids_sha256": sha256_json(
            [canonical_id(value) for value in train["id"]]
        ),
        "test_ids_sha256": sha256_json(
            [canonical_id(value) for value in test["id"]]
        ),
        "private_labels_used": False,
    }
    input_path = artifacts_dir / "input_manifest.json"
    write_json_atomic(input_path, input_manifest)

    seed_records: list[dict[str, Any]] = []
    seed_oof_probabilities: list[np.ndarray] = []
    seed_test_probabilities: list[np.ndarray] = []
    for seed in CONFIRMATION_SEEDS:
        splits, fold_assignment, fold_records = build_folds(
            train_features,
            target,
            groups,
            seed=seed,
            classes=classes,
        )
        oof_logits = np.full((len(train), len(classes)), np.nan, dtype=np.float64)
        test_fold_logits = np.empty(
            (FOLD_COUNT, len(test), len(classes)), dtype=np.float64
        )
        model_records: list[dict[str, Any]] = []
        for fold, (fit_indices, valid_indices) in enumerate(splits):
            fold_seed = seed * 100 + fold
            model = build_component_model(fold_seed)
            model.fit(train_features[fit_indices], target[fit_indices])
            oof_logits[valid_indices] = aligned_decision_logits(
                model, train_features[valid_indices], classes
            )
            test_fold_logits[fold] = aligned_decision_logits(
                model, test_features, classes
            )
            model_path = model_dir / f"seed_{seed}_fold_{fold}.joblib"
            joblib.dump(model, model_path, compress=3)
            model_records.append(
                {
                    "fold": fold,
                    "fold_seed": fold_seed,
                    **file_record(model_path, relative_to=run_dir),
                }
            )
        if not np.isfinite(oof_logits).all():
            raise RuntimeError(f"Leaf OOF decision logits are incomplete for seed {seed}")
        oof_probability = stable_softmax(oof_logits)
        test_fold_probability = np.stack(
            [stable_softmax(values) for values in test_fold_logits], axis=0
        )
        test_probability = test_fold_probability.mean(axis=0)
        test_probability /= test_probability.sum(axis=1, keepdims=True)
        seed_score = multiclass_log_loss(target, oof_probability, classes)
        seed_path = seed_dir / f"seed_{seed}.npz"
        np.savez_compressed(
            seed_path,
            seed=np.asarray([seed], dtype=np.int64),
            train_id=portable_unicode_array(
                [canonical_id(value) for value in train["id"]]
            ),
            target=portable_unicode_array(target),
            classes=portable_unicode_array(classes),
            fold_assignment=fold_assignment,
            oof_decision_logits=oof_logits,
            oof_probability=oof_probability,
            test_id=portable_unicode_array(
                [canonical_id(value) for value in test["id"]]
            ),
            test_fold_decision_logits=test_fold_logits,
            test_fold_probability=test_fold_probability,
            test_probability=test_probability,
        )
        seed_record = {
            "seed": seed,
            "cross_fitted_log_loss": seed_score,
            "gate_passed": seed_score <= LOG_LOSS_THRESHOLD,
            "folds": fold_records,
            "models": model_records,
            "bundle": file_record(seed_path, relative_to=run_dir),
        }
        seed_records.append(seed_record)
        seed_oof_probabilities.append(oof_probability)
        seed_test_probabilities.append(test_probability)

    ensemble_oof = np.mean(seed_oof_probabilities, axis=0)
    ensemble_oof /= ensemble_oof.sum(axis=1, keepdims=True)
    ensemble_test = np.mean(seed_test_probabilities, axis=0)
    ensemble_test /= ensemble_test.sum(axis=1, keepdims=True)
    ensemble_score = multiclass_log_loss(target, ensemble_oof, classes)
    gate = evaluate_gate(
        [record["cross_fitted_log_loss"] for record in seed_records],
        ensemble_score,
    )
    candidate = align_submission(sample, test["id"], ensemble_test, classes)
    candidate_path = artifacts_dir / "submission_withheld.csv"
    candidate.to_csv(candidate_path, index=False)
    ensemble_path = artifacts_dir / "ensemble_oof_and_test.npz"
    np.savez_compressed(
        ensemble_path,
        seeds=np.asarray(CONFIRMATION_SEEDS, dtype=np.int64),
        train_id=portable_unicode_array(
            [canonical_id(value) for value in train["id"]]
        ),
        target=portable_unicode_array(target),
        classes=portable_unicode_array(classes),
        oof_probability=ensemble_oof,
        test_id=portable_unicode_array(
            [canonical_id(value) for value in test["id"]]
        ),
        test_probability=ensemble_test,
    )
    report = {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "status": (
            "confirmation_gate_passed_verification_pending"
            if gate["passed"]
            else "confirmation_gate_failed"
        ),
        "run_id": run_id,
        "competition_id": COMPETITION_ID,
        "contract": contract,
        "source_manifest": file_record(source_path, relative_to=run_dir),
        "input_manifest": file_record(input_path, relative_to=run_dir),
        "seed_records": seed_records,
        "ensemble": {
            "cross_fitted_log_loss": ensemble_score,
            "bundle": file_record(ensemble_path, relative_to=run_dir),
            "submission": file_record(candidate_path, relative_to=run_dir),
        },
        "promotion_gate": gate,
        "candidate_only": True,
        "candidate_ready": False,
        "independent_verification_required": True,
        "private_labels_used": False,
        "private_scores_used_for_tuning": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "runtime_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Frozen public-OOF CPU confirmation only; independent verification and "
            "Human Gate are required before any official grader."
        ),
    }
    report_path = artifacts_dir / "confirmation_report.json"
    write_json_atomic(report_path, report)
    artifact_manifest = build_artifact_manifest(run_dir)
    artifact_manifest_path = run_dir / "artifact_manifest.json"
    write_json_atomic(artifact_manifest_path, artifact_manifest)
    return report, run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--reference-report", type=Path, default=DEFAULT_REFERENCE_REPORT)
    parser.add_argument("--reference-image-manifest", type=Path)
    parser.add_argument("--embedding-cache-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report, run_dir = run_confirmation(args)
    output = {
        "ok": True,
        "status": report["status"],
        "run_id": report["run_id"],
        "run_dir": str(run_dir),
        "seed_scores": report["promotion_gate"]["seed_scores"],
        "ensemble_log_loss": report["promotion_gate"]["ensemble_log_loss"],
        "promotion_gate_passed": report["promotion_gate"]["passed"],
        "candidate_only": True,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if report["promotion_gate"]["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
