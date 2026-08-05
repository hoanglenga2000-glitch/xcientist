#!/usr/bin/env python3
"""Independently verify a frozen Leaf CPU decision-logit confirmation run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPETITION_ID = "leaf-classification"
REPORT_SCHEMA = "evomind.leaf.decision_logit_confirmation.v1"
ARTIFACT_SCHEMA = "evomind.leaf.decision_logit_confirmation_artifacts.v1"
INPUT_SCHEMA = "evomind.leaf.decision_logit_confirmation_inputs.v1"
SOURCE_SCHEMA = "evomind.leaf.decision_logit_confirmation_sources.v1"
VERIFICATION_SCHEMA = "evomind.leaf.decision_logit_confirmation_verification.v1"

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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_id(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        require(bool(np.isfinite(value)), "Leaf ID is non-finite")
        if float(value).is_integer():
            return str(int(value))
    text = str(value).strip()
    require(bool(text), "Leaf ID is empty")
    return text


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


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    require(values.ndim == 2, "Leaf decision logits are not a matrix")
    require(np.isfinite(values).all(), "Leaf decision logits are non-finite")
    scaled = values / TEMPERATURE
    scaled -= scaled.max(axis=1, keepdims=True)
    exponential = np.exp(scaled)
    probability = exponential / exponential.sum(axis=1, keepdims=True)
    require(np.isfinite(probability).all(), "Leaf stable softmax is non-finite")
    require(
        np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-12),
        "Leaf stable softmax rows do not sum to one",
    )
    return probability


def normalize_probability(values: np.ndarray) -> np.ndarray:
    probability = np.asarray(values, dtype=np.float64)
    require(probability.ndim == 2, "Leaf probability is not a matrix")
    require(np.isfinite(probability).all(), "Leaf probability is non-finite")
    require((probability >= 0.0).all(), "Leaf probability is negative")
    row_sum = probability.sum(axis=1, keepdims=True)
    require((row_sum > 0.0).all(), "Leaf probability row is empty")
    normalized = probability / row_sum
    require(
        np.allclose(normalized.sum(axis=1), 1.0, rtol=0.0, atol=1e-12),
        "Leaf normalized probability rows differ",
    )
    return normalized


def multiclass_log_loss(
    truth: Sequence[Any], probability: np.ndarray, classes: Sequence[str]
) -> float:
    ordered = [str(value) for value in classes]
    class_index = {value: index for index, value in enumerate(ordered)}
    require(len(class_index) == len(ordered), "Leaf classes are duplicated")
    try:
        target = np.asarray([class_index[str(value)] for value in truth], dtype=np.int64)
    except KeyError as exc:
        raise RuntimeError(f"Leaf truth contains unknown class: {exc}") from exc
    values = normalize_probability(probability)
    require(
        values.shape == (len(target), len(ordered)),
        "Leaf probability shape differs from truth/classes",
    )
    selected = np.clip(values[np.arange(len(target)), target], 1e-15, 1.0)
    return float(-np.log(selected).mean())


def evaluate_gate(seed_scores: Sequence[float], ensemble_score: float) -> dict[str, Any]:
    scores = [float(value) for value in seed_scores]
    require(
        len(scores) == len(CONFIRMATION_SEEDS) and np.isfinite(scores).all(),
        "Leaf gate does not contain exactly three finite seed scores",
    )
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


def verify_gate_record(reported: dict[str, Any], rebuilt: dict[str, Any]) -> None:
    require(reported.get("passed") is rebuilt["passed"], "Leaf gate passed flag differs")
    require(reported.get("checks") == rebuilt["checks"], "Leaf gate checks differ")
    require(
        math.isclose(
            float(reported.get("threshold")),
            float(rebuilt["threshold"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "Leaf gate threshold differs",
    )
    reported_scores = [float(value) for value in reported.get("seed_scores") or []]
    require(len(reported_scores) == len(rebuilt["seed_scores"]), "Leaf gate seed count differs")
    require(
        np.allclose(
            reported_scores,
            rebuilt["seed_scores"],
            rtol=0.0,
            atol=1e-12,
        ),
        "Leaf gate seed scores differ",
    )
    for key in (
        "mean_seed_log_loss",
        "maximum_seed_log_loss",
        "ensemble_log_loss",
    ):
        require(
            math.isclose(
                float(reported.get(key)),
                float(rebuilt[key]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"Leaf gate {key} differs",
        )


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


def resolve_record_path(run_dir: Path, record: dict[str, Any]) -> Path:
    relative = str(record.get("path", ""))
    require(bool(relative), "Leaf artifact record path is empty")
    path = (run_dir / relative).resolve()
    try:
        path.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise RuntimeError("Leaf artifact path escapes run directory") from exc
    return path


def verify_file_record(path: Path, record: dict[str, Any], message: str) -> None:
    require(path.is_file(), f"{message} is missing: {path}")
    require(path.stat().st_size == int(record.get("bytes", -1)), f"{message} size differs")
    require(sha256_file(path) == record.get("sha256"), f"{message} hash differs")


def verify_artifact_manifest(run_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_path = run_dir / "artifact_manifest.json"
    manifest = read_json(manifest_path)
    require(manifest.get("schema") == ARTIFACT_SCHEMA, "Leaf artifact schema differs")
    records = manifest.get("artifacts") or []
    require(isinstance(records, list) and bool(records), "Leaf artifact records are empty")
    expected = [str(value) for value in manifest.get("expected_paths") or []]
    require(expected == sorted(expected), "Leaf expected artifact paths are not sorted")
    require(len(expected) == len(set(expected)), "Leaf expected artifact paths are duplicated")
    require(len(records) == len(expected), "Leaf artifact record count differs")
    require(manifest.get("artifact_count") == len(records), "Leaf artifact count differs")
    require(manifest.get("exact_inventory_required") is True, "Leaf exact inventory is not required")
    require(
        expected == expected_artifact_paths(),
        "Leaf artifact inventory differs from the production expected set",
    )
    by_path: dict[str, dict[str, Any]] = {}
    for record in records:
        relative = str(record.get("path", ""))
        require(relative in expected and relative not in by_path, "Leaf artifact record path differs")
        path = resolve_record_path(run_dir, record)
        verify_file_record(path, record, f"Leaf artifact {relative}")
        by_path[relative] = record
    actual = sorted(
        path.relative_to(run_dir).as_posix()
        for path in (run_dir / "artifacts").rglob("*")
        if path.is_file()
    )
    require(actual == expected, "Leaf actual artifact inventory differs from expected inventory")
    require(sorted(by_path) == expected, "Leaf artifact manifest coverage differs")
    return manifest, by_path


def verify_source_manifest(run_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    record = report.get("source_manifest") or {}
    path = resolve_record_path(run_dir, record)
    verify_file_record(path, record, "Leaf source manifest")
    manifest = read_json(path)
    require(manifest.get("schema") == SOURCE_SCHEMA, "Leaf source-manifest schema differs")
    files = manifest.get("files") or []
    expected_names = {
        "run_leaf_decision_logit_confirmation.py",
        "verify_leaf_decision_logit_confirmation.py",
        "run_leaf_multibackbone_oof.py",
        "mlebench_medal_recovery_adapters.py",
        "mlebench_wave2_adapters.py",
        "russian_transliteration.py",
        "mlebench_phase_a.py",
    }
    require(isinstance(files, list) and len(files) == 7, "Leaf source-manifest files differ")
    require(
        {Path(str(value.get("path", ""))).name for value in files} == expected_names,
        "Leaf source dependency names differ",
    )
    seen: set[str] = set()
    for source_record in files:
        source_path = Path(str(source_record.get("path", ""))).resolve()
        require(str(source_path) not in seen, "Leaf source path is duplicated")
        seen.add(str(source_path))
        verify_file_record(source_path, source_record, "Leaf source dependency")
    runtime = manifest.get("runtime") or {}
    require(runtime.get("packages") == dependency_versions(), "Leaf runtime dependency versions differ")
    require(runtime.get("python") == sys.version, "Leaf Python runtime version differs")
    return manifest


def verify_external_file_record(record: dict[str, Any], message: str) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    verify_file_record(path, record, message)
    return path


def verify_input_manifest(
    run_dir: Path,
    report: dict[str, Any],
    data_root: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    record = report.get("input_manifest") or {}
    path = resolve_record_path(run_dir, record)
    verify_file_record(path, record, "Leaf input manifest")
    manifest = read_json(path)
    require(manifest.get("schema") == INPUT_SCHEMA, "Leaf input-manifest schema differs")
    require(manifest.get("competition_id") == COMPETITION_ID, "Leaf input competition differs")
    public_dir = Path(data_root).resolve() / COMPETITION_ID / "prepared" / "public"
    expected_files = {
        "train": public_dir / "train.csv",
        "test": public_dir / "test.csv",
        "sample_submission": public_dir / "sample_submission.csv",
    }
    records = manifest.get("public_files") or {}
    for key, expected_path in expected_files.items():
        input_record = records.get(key) or {}
        require(
            Path(str(input_record.get("path", ""))).resolve() == expected_path.resolve(),
            f"Leaf {key} path differs",
        )
        verify_file_record(expected_path, input_record, f"Leaf {key}")
    train = pd.read_csv(expected_files["train"]).reset_index(drop=True)
    test = pd.read_csv(expected_files["test"]).reset_index(drop=True)
    sample = pd.read_csv(expected_files["sample_submission"]).reset_index(drop=True)
    require(len(train) == manifest.get("train_rows"), "Leaf train rows differ")
    require(len(test) == manifest.get("test_rows"), "Leaf test rows differ")
    feature_columns = [value for value in train if value not in {"id", "species"}]
    classes = [str(value) for value in sample.columns[1:]]
    require(feature_columns == manifest.get("feature_columns"), "Leaf feature columns differ")
    require(len(feature_columns) == 192, "Leaf confirmation requires 192 numeric features")
    require(test.columns.tolist() == ["id", *feature_columns], "Leaf test feature schema differs")
    require(sha256_json(feature_columns) == manifest.get("feature_columns_sha256"), "Leaf feature hash differs")
    require(classes == manifest.get("classes"), "Leaf input classes differ")
    require(sha256_json(classes) == manifest.get("classes_sha256"), "Leaf class hash differs")
    require(len(classes) == 99, "Leaf confirmation requires 99 classes")
    require(
        sha256_json([canonical_id(value) for value in train["id"]])
        == manifest.get("train_ids_sha256"),
        "Leaf train ID hash differs",
    )
    require(
        sha256_json([canonical_id(value) for value in test["id"]])
        == manifest.get("test_ids_sha256"),
        "Leaf test ID hash differs",
    )
    require(manifest.get("private_labels_used") is False, "Leaf input manifest used private labels")
    reference_report_path = verify_external_file_record(
        manifest.get("reference_report") or {}, "Leaf reference report"
    )
    reference_report = read_json(reference_report_path)
    require(
        reference_report.get("competition_id") == COMPETITION_ID,
        "Leaf reference-report competition differs",
    )
    image_manifest_path = verify_external_file_record(
        manifest.get("reference_image_manifest") or {}, "Leaf reference image manifest"
    )
    image_manifest = pd.read_csv(image_manifest_path)
    require(len(image_manifest) == len(train) + len(test), "Leaf image-manifest rows differ")
    require(
        [canonical_id(value) for value in image_manifest.iloc[: len(train)]["id"]]
        == [canonical_id(value) for value in train["id"]],
        "Leaf image-manifest train IDs differ",
    )
    cache_records = manifest.get("embedding_caches") or []
    require(
        [str(record.get("backbone")) for record in cache_records] == list(BACKBONES),
        "Leaf cache backbone order differs",
    )
    for cache_record in cache_records:
        backbone = str(cache_record.get("backbone"))
        require(
            int(cache_record.get("width", -1)) == BACKBONE_WIDTHS[backbone],
            "Leaf embedding width differs from the production contract",
        )
        cache_path = verify_external_file_record(cache_record.get("cache") or {}, "Leaf embedding cache")
        metadata_path = verify_external_file_record(
            cache_record.get("metadata") or {}, "Leaf embedding metadata"
        )
        metadata = read_json(metadata_path)
        require(metadata.get("cache_sha256") == sha256_file(cache_path), "Leaf cache metadata hash differs")
        require(metadata.get("contract") == cache_record.get("contract"), "Leaf cache contract differs")
        require(
            (metadata.get("contract") or {}).get("backbone") == backbone,
            "Leaf cache contract backbone differs",
        )
        require(
            (metadata.get("contract") or {}).get("private_labels_used") is False,
            "Leaf cache contract used private labels",
        )
        array = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        require(not array.dtype.hasobject, "Leaf embedding cache has object dtype")
        require(
            array.shape
            == (
                len(train) + len(test),
                int(cache_record.get("width", -1)),
            ),
            "Leaf embedding cache shape differs",
        )
        require(np.isfinite(array).all(), "Leaf embedding cache is non-finite")
    return manifest, train, test, sample, image_manifest


def verify_seed_bundle(
    run_dir: Path,
    seed_record: dict[str, Any],
    *,
    train_ids: Sequence[str],
    target: np.ndarray,
    test_ids: Sequence[str],
    classes: Sequence[str],
    groups: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    expected_seed = int(seed_record.get("seed", -1))
    require(expected_seed in CONFIRMATION_SEEDS, "Leaf seed record seed differs")
    bundle_record = seed_record.get("bundle") or {}
    require(
        str(bundle_record.get("path")) == f"artifacts/seeds/seed_{expected_seed}.npz",
        "Leaf seed bundle path differs",
    )
    bundle_path = resolve_record_path(run_dir, bundle_record)
    verify_file_record(bundle_path, bundle_record, f"Leaf seed {expected_seed} bundle")
    try:
        archive_context = np.load(bundle_path, allow_pickle=False)
    except ValueError as exc:
        raise RuntimeError("Leaf seed bundle requires pickle/object arrays") from exc
    with archive_context as archive:
        required = {
            "seed",
            "train_id",
            "target",
            "classes",
            "fold_assignment",
            "oof_decision_logits",
            "oof_probability",
            "test_id",
            "test_fold_decision_logits",
            "test_fold_probability",
            "test_probability",
        }
        require(required <= set(archive.files), "Leaf seed bundle arrays are incomplete")
        for name in required:
            require(not archive[name].dtype.hasobject, f"Leaf seed array {name} has object dtype")
        seed_value = np.asarray(archive["seed"], dtype=np.int64).reshape(-1)
        bundle_train_ids = np.asarray(archive["train_id"]).astype(str).tolist()
        bundle_target = np.asarray(archive["target"]).astype(str)
        bundle_classes = np.asarray(archive["classes"]).astype(str).tolist()
        folds = np.asarray(archive["fold_assignment"], dtype=np.int16)
        oof_logits = np.asarray(archive["oof_decision_logits"], dtype=np.float64)
        reported_oof = normalize_probability(archive["oof_probability"])
        bundle_test_ids = np.asarray(archive["test_id"]).astype(str).tolist()
        test_fold_logits = np.asarray(
            archive["test_fold_decision_logits"], dtype=np.float64
        )
        reported_test_fold = np.asarray(
            archive["test_fold_probability"], dtype=np.float64
        )
        reported_test = normalize_probability(archive["test_probability"])
    require(seed_value.tolist() == [expected_seed], "Leaf bundle seed differs")
    require(bundle_train_ids == list(train_ids), "Leaf bundle train IDs differ")
    require(np.array_equal(bundle_target, target.astype(str)), "Leaf bundle targets differ")
    require(bundle_classes == [str(value) for value in classes], "Leaf bundle classes differ")
    require(bundle_test_ids == list(test_ids), "Leaf bundle test IDs differ")
    require(
        sorted(np.unique(folds).tolist()) == list(range(FOLD_COUNT)),
        "Leaf fold assignment differs",
    )
    require(folds.shape == (len(train_ids),), "Leaf fold-assignment rows differ")
    expected_classes = set(str(value) for value in classes)
    write_count = np.zeros(len(folds), dtype=np.uint8)
    fold_records = seed_record.get("folds") or []
    require(len(fold_records) == FOLD_COUNT, "Leaf fold records differ")
    for fold in range(FOLD_COUNT):
        valid = folds == fold
        fit = ~valid
        write_count[valid] += 1
        require(set(target[valid].astype(str)) == expected_classes, "Leaf validation classes differ")
        require(set(target[fit].astype(str)) == expected_classes, "Leaf fit classes differ")
        require(
            not (set(groups[valid].astype(str)) & set(groups[fit].astype(str))),
            "Leaf image group crossed a fold",
        )
        record = fold_records[fold]
        require(int(record.get("fold", -1)) == fold, "Leaf fold-record order differs")
        require(int(record.get("valid_rows", -1)) == int(valid.sum()), "Leaf fold rows differ")
        require(int(record.get("group_overlap_count", -1)) == 0, "Leaf fold overlap differs")
    require(np.all(write_count == 1), "Leaf OOF rows were not written exactly once")
    rebuilt_oof = stable_softmax(oof_logits)
    require(
        np.allclose(rebuilt_oof, reported_oof, rtol=0.0, atol=1e-12),
        "Leaf OOF probability does not match decision logits",
    )
    require(
        test_fold_logits.shape
        == (FOLD_COUNT, len(test_ids), len(classes)),
        "Leaf test fold-logit shape differs",
    )
    rebuilt_test_fold = np.stack(
        [stable_softmax(test_fold_logits[fold]) for fold in range(FOLD_COUNT)],
        axis=0,
    )
    require(
        np.allclose(rebuilt_test_fold, reported_test_fold, rtol=0.0, atol=1e-12),
        "Leaf test fold probabilities do not match decision logits",
    )
    rebuilt_test = normalize_probability(rebuilt_test_fold.mean(axis=0))
    require(
        np.allclose(rebuilt_test, reported_test, rtol=0.0, atol=1e-12),
        "Leaf seed test probability does not match fold mean",
    )
    score = multiclass_log_loss(target, rebuilt_oof, classes)
    require(
        math.isclose(
            score,
            float(seed_record.get("cross_fitted_log_loss")),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "Leaf seed log loss differs",
    )
    require(
        seed_record.get("gate_passed") is (score <= LOG_LOSS_THRESHOLD),
        "Leaf seed gate differs",
    )
    models = seed_record.get("models") or []
    require(len(models) == FOLD_COUNT, "Leaf seed model inventory differs")
    require([int(value.get("fold", -1)) for value in models] == list(range(FOLD_COUNT)), "Leaf model fold order differs")
    for model_record in models:
        fold = int(model_record.get("fold", -1))
        require(
            int(model_record.get("fold_seed", -1)) == expected_seed * 100 + fold,
            "Leaf fold-model seed differs",
        )
        require(
            str(model_record.get("path"))
            == f"artifacts/models/seed_{expected_seed}_fold_{fold}.joblib",
            "Leaf fold-model path differs",
        )
        model_path = resolve_record_path(run_dir, model_record)
        verify_file_record(model_path, model_record, "Leaf fold model")
    return {
        "seed": expected_seed,
        "cross_fitted_log_loss": score,
        "gate_passed": score <= LOG_LOSS_THRESHOLD,
        "bundle_sha256": sha256_file(bundle_path),
    }, rebuilt_oof, rebuilt_test


def aligned_submission(
    sample: pd.DataFrame,
    test_ids: Sequence[str],
    probability: np.ndarray,
    classes: Sequence[str],
) -> pd.DataFrame:
    ordered = [str(value) for value in classes]
    require(sample.columns.tolist() == ["id", *ordered], "Leaf sample columns differ")
    sample_ids = [canonical_id(value) for value in sample["id"]]
    require(len(set(test_ids)) == len(test_ids), "Leaf test IDs are duplicated")
    require(set(sample_ids) == set(test_ids), "Leaf sample/test ID set differs")
    mapping = {value: probability[index] for index, value in enumerate(test_ids)}
    values = np.vstack([mapping[value] for value in sample_ids])
    result = sample.copy()
    result.loc[:, ordered] = values
    return result


def verify_run(run_dir: Path, data_root: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    manifest, _ = verify_artifact_manifest(run_dir)
    report_path = run_dir / "artifacts" / "confirmation_report.json"
    report = read_json(report_path)
    require(report.get("schema") == REPORT_SCHEMA, "Leaf report schema differs")
    require(report.get("competition_id") == COMPETITION_ID, "Leaf report competition differs")
    require(report.get("contract") == fixed_contract(), "Leaf frozen contract differs")
    require(report.get("candidate_only") is True, "Leaf run is not candidate-only")
    require(report.get("candidate_ready") is False, "Leaf runner prematurely marked candidate ready")
    require(
        report.get("independent_verification_required") is True,
        "Leaf independent verification is not required",
    )
    require(report.get("private_labels_used") is False, "Leaf run used private labels")
    require(
        report.get("private_scores_used_for_tuning") is False,
        "Leaf run used private scores",
    )
    require(report.get("official_grader_executed") is False, "Leaf run executed grader")
    require(report.get("kaggle_submission_executed") is False, "Leaf run submitted to Kaggle")
    sources = verify_source_manifest(run_dir, report)
    inputs, train, test, sample, image_manifest = verify_input_manifest(
        run_dir, report, data_root
    )
    classes = [str(value) for value in sample.columns[1:]]
    target = train["species"].astype(str).to_numpy(dtype=str)
    train_ids = [canonical_id(value) for value in train["id"]]
    test_ids = [canonical_id(value) for value in test["id"]]
    groups = image_manifest.iloc[: len(train)]["sha256"].astype(str).to_numpy()
    seed_records = report.get("seed_records") or []
    require(
        [int(value.get("seed", -1)) for value in seed_records]
        == list(CONFIRMATION_SEEDS),
        "Leaf confirmation seed order differs",
    )
    verified_seeds = []
    seed_oof = []
    seed_test = []
    for seed_record in seed_records:
        result, oof_probability, test_probability = verify_seed_bundle(
            run_dir,
            seed_record,
            train_ids=train_ids,
            target=target,
            test_ids=test_ids,
            classes=classes,
            groups=groups,
        )
        verified_seeds.append(result)
        seed_oof.append(oof_probability)
        seed_test.append(test_probability)
    ensemble_oof = normalize_probability(np.mean(seed_oof, axis=0))
    ensemble_test = normalize_probability(np.mean(seed_test, axis=0))
    ensemble_score = multiclass_log_loss(target, ensemble_oof, classes)
    ensemble_record = report.get("ensemble") or {}
    ensemble_bundle_record = ensemble_record.get("bundle") or {}
    require(
        ensemble_bundle_record.get("path") == "artifacts/ensemble_oof_and_test.npz",
        "Leaf ensemble bundle path differs",
    )
    ensemble_bundle_path = resolve_record_path(run_dir, ensemble_bundle_record)
    verify_file_record(ensemble_bundle_path, ensemble_bundle_record, "Leaf ensemble bundle")
    with np.load(ensemble_bundle_path, allow_pickle=False) as archive:
        for name in archive.files:
            require(not archive[name].dtype.hasobject, f"Leaf ensemble {name} has object dtype")
        require(
            np.asarray(archive["seeds"], dtype=np.int64).tolist()
            == list(CONFIRMATION_SEEDS),
            "Leaf ensemble seeds differ",
        )
        require(np.asarray(archive["train_id"]).astype(str).tolist() == train_ids, "Leaf ensemble train IDs differ")
        require(np.asarray(archive["target"]).astype(str).tolist() == target.astype(str).tolist(), "Leaf ensemble targets differ")
        require(np.asarray(archive["classes"]).astype(str).tolist() == classes, "Leaf ensemble classes differ")
        require(np.asarray(archive["test_id"]).astype(str).tolist() == test_ids, "Leaf ensemble test IDs differ")
        require(
            np.allclose(
                normalize_probability(archive["oof_probability"]),
                ensemble_oof,
                rtol=0.0,
                atol=1e-12,
            ),
            "Leaf ensemble OOF differs",
        )
        require(
            np.allclose(
                normalize_probability(archive["test_probability"]),
                ensemble_test,
                rtol=0.0,
                atol=1e-12,
            ),
            "Leaf ensemble test probability differs",
        )
    require(
        math.isclose(
            ensemble_score,
            float(ensemble_record.get("cross_fitted_log_loss")),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "Leaf ensemble log loss differs",
    )
    candidate_record = ensemble_record.get("submission") or {}
    require(
        candidate_record.get("path") == "artifacts/submission_withheld.csv",
        "Leaf withheld-submission path differs",
    )
    candidate_path = resolve_record_path(run_dir, candidate_record)
    verify_file_record(candidate_path, candidate_record, "Leaf withheld submission")
    expected_candidate = aligned_submission(sample, test_ids, ensemble_test, classes)
    candidate = pd.read_csv(candidate_path)
    require(candidate.columns.tolist() == expected_candidate.columns.tolist(), "Leaf candidate columns differ")
    require(
        [canonical_id(value) for value in candidate["id"]]
        == [canonical_id(value) for value in expected_candidate["id"]],
        "Leaf candidate IDs differ",
    )
    require(
        np.allclose(
            candidate[classes].to_numpy(dtype=np.float64),
            expected_candidate[classes].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        ),
        "Leaf candidate probabilities differ",
    )
    gate = evaluate_gate(
        [value["cross_fitted_log_loss"] for value in verified_seeds],
        ensemble_score,
    )
    verify_gate_record(report.get("promotion_gate") or {}, gate)
    expected_status = (
        "confirmation_gate_passed_verification_pending"
        if gate["passed"]
        else "confirmation_gate_failed"
    )
    require(report.get("status") == expected_status, "Leaf report terminal status differs")
    return {
        "schema": VERIFICATION_SCHEMA,
        "created_at": now_iso(),
        "status": "verification_passed",
        "ok": True,
        "candidate_ready": gate["passed"],
        "run_id": report["run_id"],
        "run_dir": str(run_dir),
        "contract_sha256": report["contract"]["contract_sha256"],
        "artifact_manifest_sha256": sha256_file(run_dir / "artifact_manifest.json"),
        "artifact_count": manifest["artifact_count"],
        "source_manifest_sha256": sha256_file(
            resolve_record_path(run_dir, report["source_manifest"])
        ),
        "input_manifest_sha256": sha256_file(
            resolve_record_path(run_dir, report["input_manifest"])
        ),
        "verified_sources": len(sources["files"]),
        "verified_embedding_caches": len(inputs["embedding_caches"]),
        "seed_results": verified_seeds,
        "ensemble_log_loss": ensemble_score,
        "promotion_gate": gate,
        "candidate_path": str(candidate_path),
        "candidate_sha256": sha256_file(candidate_path),
        "candidate_only": True,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "claim_boundary": (
            "Independent public-OOF verification is not an official medal; "
            "Human Gate remains mandatory."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output or Path(args.run_dir) / "independent_verification.json"
    try:
        report = verify_run(args.run_dir, args.data_root)
    except Exception as exc:
        report = {
            "schema": VERIFICATION_SCHEMA,
            "created_at": now_iso(),
            "status": "verification_failed",
            "ok": False,
            "candidate_ready": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "candidate_only": True,
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        }
    write_json_atomic(Path(output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
