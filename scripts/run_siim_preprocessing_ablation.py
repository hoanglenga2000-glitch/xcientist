#!/usr/bin/env python3
"""Leakage-safe SIIM preprocessing ablation with frozen GPU embeddings.

The runner evaluates every incremental preprocessing profile on the complete
public training set using one immutable set of patient/content-grouped folds.
ImageNet-pretrained backbones are frozen; a fresh fold-local linear head is fit
for each profile/fold.  The script has no private-grader or Kaggle submission
execution path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from research_os.mlebench_phase_a import resolve_competition

try:
    import mlebench_medal_recovery_adapters as recovery
    import mlebench_wave2_adapters as wave2
except ModuleNotFoundError:
    from scripts import mlebench_medal_recovery_adapters as recovery
    from scripts import mlebench_wave2_adapters as wave2

try:
    import mlebench_compute_policy as compute_policy
except ModuleNotFoundError:
    try:
        from scripts import mlebench_compute_policy as compute_policy
    except (ModuleNotFoundError, ImportError):  # Standalone remote HPC source copy.
        compute_policy = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPETITION_ID = "siim-isic-melanoma-classification"
# The corrected bundle changes only leakage-group orchestration in the adapter.
# The actual image decoding, preprocessing, and feature extraction functions are
# unchanged, so the already-hashed A800 embedding cache remains scientifically
# identical and is reused under its original immutable extraction contract.
FROZEN_EMBEDDING_ADAPTER_CONTRACT_SHA256 = (
    "0fdaba52b47973953a9523e7d547f5183eae1bd9b164b206f93b61d2c77d5e8e"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "workspace" / "local_gpu" / "siim_preprocessing_ablation"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_run_directory(run_dir: Path, *, resume: bool) -> str:
    run_dir = Path(run_dir)
    if run_dir.exists():
        if not resume:
            raise FileExistsError(f"SIIM ablation run directory already exists: {run_dir}")
        if not run_dir.is_dir():
            raise RuntimeError(f"SIIM ablation run path is not a directory: {run_dir}")
        return "resumed"
    run_dir.mkdir(parents=True, exist_ok=False)
    return "created"


def build_embedding_cache_contract(
    *,
    profile: str,
    view_name: str,
    backbone: str,
    weight_sha256: str,
    image_vector_sha256: str,
    image_content_manifest_sha256: str,
    image_size: int,
    adapter_source_sha256: str,
    wave2_source_sha256: str,
) -> dict[str, Any]:
    """Return the complete immutable key for one embedding cache."""

    if profile not in recovery.SIIM_PREPROCESSING_PROFILES:
        raise ValueError(f"Unsupported SIIM preprocessing profile: {profile}")
    if view_name not in {"full_image", "lesion_focus"}:
        raise ValueError(f"Unsupported SIIM embedding view: {view_name}")
    sha_fields = {
        "weight_sha256": weight_sha256,
        "image_vector_sha256": image_vector_sha256,
        "image_content_manifest_sha256": image_content_manifest_sha256,
        "adapter_source_sha256": adapter_source_sha256,
        "wave2_source_sha256": wave2_source_sha256,
    }
    invalid = [
        name
        for name, value in sha_fields.items()
        if len(str(value)) != 64 or any(char not in "0123456789abcdef" for char in str(value))
    ]
    if invalid:
        raise ValueError(f"Invalid SHA256 fields in SIIM embedding cache: {invalid}")
    if int(image_size) <= 0:
        raise ValueError("SIIM embedding image size must be positive")
    contract = {
        "schema": "evomind.siim_preprocessing_embedding_cache.v1",
        "competition_id": COMPETITION_ID,
        "profile": profile,
        "profile_steps": recovery.SIIM_PREPROCESSING_PROFILE_STEPS[profile],
        "view_name": view_name,
        "backbone": backbone,
        "weight_sha256": weight_sha256,
        "image_vector_sha256": image_vector_sha256,
        "image_content_manifest_sha256": image_content_manifest_sha256,
        "image_size": int(image_size),
        "adapter_source_sha256": adapter_source_sha256,
        "wave2_source_sha256": wave2_source_sha256,
        "embedding_dtype": "float32",
        "private_labels_used": False,
    }
    contract["cache_key_sha256"] = sha256_json(contract)
    return contract


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the same run directory and reuse only exact verified embedding caches.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--evaluation-seeds",
        default="40,41,42",
        help=(
            "Comma-separated external split/head seeds. At least three distinct "
            "seeds are required."
        ),
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--memory-limit-mib",
        type=int,
        default=0,
        help="Per-process CUDA ceiling; governed HPC runs pass an explicit value.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--manifest-workers", type=int, default=8)
    parser.add_argument(
        "--image-content-manifest",
        type=Path,
        help=(
            "Verified manifest from the superseded same-Run ablation attempt; "
            "it is copied and revalidated instead of rehashing all JPEGs."
        ),
    )
    parser.add_argument(
        "--full-backbone",
        choices=tuple(wave2.VISION_BACKBONE_SPECS),
        default="convnext_small",
    )
    parser.add_argument(
        "--lesion-backbone",
        choices=tuple(wave2.VISION_BACKBONE_SPECS),
        default="efficientnet_v2_s",
    )
    parser.add_argument("--linear-alpha", type=float, default=1e-4)
    parser.add_argument("--linear-max-iter", type=int, default=2_000)
    parser.add_argument("--minimum-mean-gain", type=float, default=0.0005)
    parser.add_argument("--maximum-worst-fold-regression", type=float, default=0.002)
    parser.add_argument("--maximum-seed-mean-regression", type=float, default=0.001)
    parser.add_argument("--minimum-seed-pass-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--torch-home", type=Path)
    parser.add_argument(
        "--no-resume-cache",
        action="store_true",
        help="Recompute embeddings even when an exact verified cache exists.",
    )
    return parser.parse_args(argv)


def parse_evaluation_seeds(raw_value: str | Sequence[int]) -> list[int]:
    """Normalize the external evaluation-seed contract before CUDA is touched."""

    if isinstance(raw_value, str):
        tokens = [token.strip() for token in raw_value.split(",")]
        if not tokens or any(not token for token in tokens):
            raise ValueError("SIIM evaluation seeds must be comma-separated integers")
        try:
            seeds = [int(token) for token in tokens]
        except ValueError as exc:
            raise ValueError(
                "SIIM evaluation seeds must be comma-separated integers"
            ) from exc
    else:
        seeds = [int(value) for value in raw_value]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError(
            "SIIM preprocessing ablation requires at least three distinct "
            "external evaluation seeds"
        )
    return seeds


def configure_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"siim-preprocessing-ablation-{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(run_dir / "ablation.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.handlers[:] = [file_handler, stream_handler]
    return logger


def load_public_dataset(data_root: Path) -> dict[str, Any]:
    resolved = resolve_competition(COMPETITION_ID, Path(data_root).resolve())
    train = pd.read_csv(resolved.public_dir / "train.csv").reset_index(drop=True)
    test = pd.read_csv(resolved.public_dir / "test.csv").reset_index(drop=True)
    sample = pd.read_csv(resolved.sample_submission_path).reset_index(drop=True)
    if not {"image_name", "patient_id", "target"} <= set(train):
        raise RuntimeError("SIIM public training schema is incomplete")
    if "image_name" not in test or list(sample.columns) != ["image_name", "target"]:
        raise RuntimeError("SIIM public test or sample-submission schema is invalid")
    if (
        train["image_name"].duplicated().any()
        or test["image_name"].duplicated().any()
        or sample["image_name"].duplicated().any()
    ):
        raise RuntimeError("SIIM public image IDs are duplicated")
    ordered_test = sample[["image_name"]].merge(
        test,
        on="image_name",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if len(ordered_test) != len(test) or bool((ordered_test["_merge"] != "both").any()):
        raise RuntimeError("SIIM sample submission and public-test image sets differ")
    test = ordered_test.drop(columns="_merge")
    target = train["target"].astype(np.int8).to_numpy()
    if set(target.tolist()) != {0, 1}:
        raise RuntimeError("SIIM public target must contain both classes")
    train_paths = [
        resolved.public_dir / "jpeg" / "train" / f"{value}.jpg"
        for value in train["image_name"].astype(str)
    ]
    test_paths = [
        resolved.public_dir / "jpeg" / "test" / f"{value}.jpg"
        for value in test["image_name"].astype(str)
    ]
    missing_train = sum(not path.is_file() for path in train_paths)
    missing_test = sum(not path.is_file() for path in test_paths)
    if missing_train or missing_test:
        raise FileNotFoundError(
            "SIIM public image manifest is incomplete: "
            f"train_missing={missing_train} test_missing={missing_test}"
        )
    return {
        "resolved": resolved,
        "train": train,
        "test": test,
        "target": target,
        "train_paths": train_paths,
        "test_paths": test_paths,
    }


def build_manifest_and_folds(
    dataset: dict[str, Any],
    *,
    run_dir: Path,
    folds: int,
    evaluation_seeds: Sequence[int],
    workers: int,
    precomputed_manifest: Path | None = None,
) -> dict[str, Any]:
    train = dataset["train"]
    test = dataset["test"]
    target = dataset["target"]
    normalized_seeds = [int(value) for value in evaluation_seeds]
    if len(normalized_seeds) < 3 or len(set(normalized_seeds)) != len(normalized_seeds):
        raise RuntimeError("SIIM ablation folds require three distinct external seeds")
    manifest_path = run_dir / "siim_image_content_manifest.csv"
    if precomputed_manifest is not None:
        source_manifest = Path(precomputed_manifest).resolve()
        manifest, manifest_report = recovery.load_precomputed_siim_image_content_manifest(
            source_manifest,
            train_ids=train["image_name"].astype(str).tolist(),
            test_ids=test["image_name"].astype(str).tolist(),
            train_paths=dataset["train_paths"],
            test_paths=dataset["test_paths"],
        )
        if source_manifest != manifest_path.resolve():
            temporary_manifest = manifest_path.with_name(
                f".{manifest_path.name}.{os.getpid()}.tmp"
            )
            shutil.copy2(source_manifest, temporary_manifest)
            os.replace(temporary_manifest, manifest_path)
        manifest_report["reuse_role"] = "superseded_same_run_ablation_manifest"
    else:
        manifest, manifest_report = recovery.build_siim_image_content_manifest(
            dataset["train_paths"],
            dataset["test_paths"],
            train_ids=train["image_name"].astype(str).tolist(),
            test_ids=test["image_name"].astype(str).tolist(),
            workers=max(1, int(workers)),
        )
        manifest.to_csv(manifest_path, index=False)
    manifest_sha256 = sha256_file(manifest_path)
    train_manifest = manifest.loc[manifest["source"] == "train"].reset_index(drop=True)
    if train_manifest["image_name"].astype(str).tolist() != train[
        "image_name"
    ].astype(str).tolist():
        raise RuntimeError("SIIM train image-content manifest order changed")
    leakage_groups, group_report = recovery.build_siim_content_connected_groups(
        train,
        target,
        train_manifest["content_sha256"].to_numpy(dtype=str),
        decoded_pixel_sha256=train_manifest["decoded_pixel_sha256"].to_numpy(dtype=str),
        perceptual_dhash64=train_manifest["perceptual_dhash64"].to_numpy(dtype=str),
        perceptual_max_distance=1,
    )
    splits_by_seed: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    split_strategy_by_seed: dict[str, str] = {}
    split_records_by_seed: dict[str, list[dict[str, Any]]] = {}
    assignment_paths: dict[int, Path] = {}
    split_contracts: list[dict[str, Any]] = []
    for seed in normalized_seeds:
        split_list, split_groups, split_strategy = recovery.build_siim_patient_folds(
            train,
            target,
            requested_folds=int(folds),
            seed=seed,
            group_values=leakage_groups,
            require_requested_folds=True,
        )
        if not np.array_equal(split_groups, leakage_groups):
            raise RuntimeError("SIIM ablation split group identities changed")
        assignments = np.full(len(train), -1, dtype=np.int16)
        split_records: list[dict[str, Any]] = []
        for fold, (fit_index, valid_index) in enumerate(split_list):
            fit_index = np.asarray(fit_index, dtype=np.int64)
            valid_index = np.asarray(valid_index, dtype=np.int64)
            assignments[valid_index] = fold
            overlap = set(split_groups[fit_index]) & set(split_groups[valid_index])
            if overlap:
                raise RuntimeError(
                    "SIIM ablation patient/content groups overlap across a fold"
                )
            split_records.append({
                "fold": fold,
                "fit_rows": len(fit_index),
                "valid_rows": len(valid_index),
                "fit_positive": int(target[fit_index].sum()),
                "valid_positive": int(target[valid_index].sum()),
                "group_overlap_count": len(overlap),
            })
        if np.any(assignments < 0):
            raise RuntimeError("SIIM ablation folds do not cover every public-train row")
        assignment_frame = pd.DataFrame({
            "image_name": train["image_name"].astype(str),
            "content_connected_group": split_groups,
            "fold": assignments,
        })
        assignment_path = run_dir / f"siim_preprocessing_ablation_folds_s{seed}.csv"
        assignment_frame.to_csv(assignment_path, index=False)
        splits_by_seed[seed] = split_list
        split_strategy_by_seed[str(seed)] = split_strategy
        split_records_by_seed[str(seed)] = split_records
        assignment_paths[seed] = assignment_path
        split_contracts.append({
            "seed": seed,
            "requested_folds": int(folds),
            "actual_folds": len(split_list),
            "strategy": split_strategy,
            "folds": split_records,
            "assignment_path": assignment_path.name,
            "assignment_sha256": sha256_file(assignment_path),
        })
    image_vector_sha256 = hashlib.sha256(
        "\n".join(
            f"{row.image_name}\t{row.content_sha256}"
            for row in train_manifest.itertuples(index=False)
        ).encode("utf-8")
    ).hexdigest()
    manifest_report.update({
        "path": manifest_path.name,
        "sha256": manifest_sha256,
    })
    write_json(run_dir / "siim_image_content_manifest_report.json", manifest_report)
    write_json(run_dir / "siim_duplicate_connected_groups.json", group_report)
    write_json(
        run_dir / "siim_preprocessing_ablation_split_contract.json",
        {
            "schema": "evomind.siim_preprocessing_ablation_splits.v1",
            "evaluation_seed_count": len(normalized_seeds),
            "evaluation_seeds": normalized_seeds,
            "seed_contracts": split_contracts,
            "patient_content_group_isolation": True,
            "leakage_group_policy": recovery.SIIM_LEAKAGE_GROUP_POLICY,
            "perceptual_edge_policy": recovery.SIIM_PERCEPTUAL_EDGE_POLICY,
            "validation_coverage_exactly_once": True,
            "private_labels_used": False,
        },
    )
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "image_vector_sha256": image_vector_sha256,
        "leakage_groups": leakage_groups,
        "splits_by_seed": splits_by_seed,
        "split_strategy_by_seed": split_strategy_by_seed,
        "split_records_by_seed": split_records_by_seed,
        "assignment_paths": assignment_paths,
    }


class SiimEmbeddingDataset:
    """Pickle-safe map dataset; torch base classes are deliberately unnecessary."""

    def __init__(
        self,
        paths: list[Path],
        *,
        profile: str,
        view_name: str,
        image_size: int,
    ) -> None:
        self.paths = [Path(path) for path in paths]
        self.profile = profile
        self.view_name = view_name
        self.image_size = int(image_size)
        _, self.transform = wave2._image_transforms(self.image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = recovery._open_siim_rgb_for_transform(
            self.paths[index],
            decode_size=self.image_size + 32,
        )
        views = recovery.prepare_siim_dermoscopy_views(image, profile=self.profile)
        return self.transform(views[self.view_name]), int(index)


def load_frozen_backbone(backbone_name: str, *, device: Any) -> tuple[Any, dict[str, Any]]:
    import torch
    from torch import nn

    model, pretrained, identity = wave2._vision_model(1, backbone=backbone_name)
    if not pretrained:
        raise RuntimeError(f"SIIM ablation backbone did not load fixed weights: {backbone_name}")
    spec = wave2.VISION_BACKBONE_SPECS[backbone_name]
    classifier_index = int(spec["classifier"].rsplit(".", maxsplit=1)[-1])
    model.classifier[classifier_index] = nn.Identity()
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if identity.get("sha256") != identity.get("expected_sha256"):
        raise RuntimeError(f"SIIM ablation backbone weight SHA256 mismatch: {backbone_name}")
    torch.cuda.empty_cache()
    return model, identity


def verified_embedding_cache(
    *,
    cache_path: Path,
    metadata_path: Path,
    expected_contract: dict[str, Any],
    expected_rows: int,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    if not cache_path.is_file() or not metadata_path.is_file():
        return None, None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        checks = {
            "contract_matches": metadata.get("contract") == expected_contract,
            "rows_match": int(metadata.get("rows", -1)) == int(expected_rows),
            "cache_sha256_matches": metadata.get("cache_sha256") == sha256_file(cache_path),
        }
        array = np.load(cache_path, mmap_mode="r")
        checks["array_shape_valid"] = bool(
            array.ndim == 2 and array.shape[0] == expected_rows and array.shape[1] > 0
        )
        checks["array_dtype_valid"] = array.dtype == np.float32
        if not all(checks.values()):
            return None, {"validated": False, "checks": checks}
        metadata["metadata_path"] = str(metadata_path)
        return array, {"validated": True, "checks": checks, **metadata}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None, None


def extract_view_embeddings(
    *,
    paths: list[Path],
    profile: str,
    view_name: str,
    backbone_name: str,
    model: Any,
    weight_identity: dict[str, Any],
    cache_dir: Path,
    cache_contract_base: dict[str, Any],
    image_size: int,
    batch_size: int,
    workers: int,
    resume_cache: bool,
    logger: logging.Logger,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader

    contract = build_embedding_cache_contract(
        profile=profile,
        view_name=view_name,
        backbone=backbone_name,
        weight_sha256=str(weight_identity["sha256"]),
        image_vector_sha256=cache_contract_base["image_vector_sha256"],
        image_content_manifest_sha256=cache_contract_base[
            "image_content_manifest_sha256"
        ],
        image_size=image_size,
        adapter_source_sha256=cache_contract_base["adapter_source_sha256"],
        wave2_source_sha256=cache_contract_base["wave2_source_sha256"],
    )
    key = contract["cache_key_sha256"]
    stem = f"{profile}__{view_name}__{backbone_name}__{key[:16]}"
    cache_path = cache_dir / f"{stem}.npy"
    metadata_path = cache_dir / f"{stem}.json"
    if resume_cache:
        cached, cache_metadata = verified_embedding_cache(
            cache_path=cache_path,
            metadata_path=metadata_path,
            expected_contract=contract,
            expected_rows=len(paths),
        )
        if cached is not None and cache_metadata is not None:
            logger.info(
                "profile=%s view=%s backbone=%s embedding_cache=verified_reuse",
                profile,
                view_name,
                backbone_name,
            )
            return cached, cache_metadata

    dataset = SiimEmbeddingDataset(
        paths,
        profile=profile,
        view_name=view_name,
        image_size=image_size,
    )
    worker_count = max(0, int(workers))
    loader_options: dict[str, Any] = {
        "batch_size": max(1, int(batch_size)),
        "shuffle": False,
        "num_workers": worker_count,
        "pin_memory": True,
        "persistent_workers": worker_count > 0,
        "worker_init_fn": wave2._seed_vision_worker,
    }
    if worker_count > 0:
        loader_options["prefetch_factor"] = wave2.VISION_DATALOADER_PREFETCH_FACTOR
    loader = DataLoader(dataset, **loader_options)
    output: np.ndarray | None = None
    started = time.perf_counter()
    processed = 0
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.inference_mode():
        for batch_index, (images, indices) in enumerate(loader):
            images = images.to("cuda", non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                features = model(images)
            values = features.detach().to(dtype=torch.float32).cpu().numpy()
            if values.ndim != 2 or not np.isfinite(values).all():
                raise RuntimeError("SIIM frozen backbone produced invalid embeddings")
            if output is None:
                output = np.empty((len(paths), values.shape[1]), dtype=np.float32)
            output[np.asarray(indices, dtype=np.int64)] = values
            processed += len(values)
            if batch_index % 50 == 0 or processed == len(paths):
                elapsed = max(time.perf_counter() - started, 1e-6)
                logger.info(
                    "profile=%s view=%s rows=%d/%d images_per_second=%.1f",
                    profile,
                    view_name,
                    processed,
                    len(paths),
                    processed / elapsed,
                )
    if output is None or processed != len(paths) or not np.isfinite(output).all():
        raise RuntimeError("SIIM embedding extraction did not cover every row")
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, output, allow_pickle=False)
    temporary.replace(cache_path)
    cache_metadata = {
        "schema": "evomind.siim_preprocessing_embedding_artifact.v1",
        "created_at": utc_now(),
        "contract": contract,
        "rows": int(output.shape[0]),
        "width": int(output.shape[1]),
        "dtype": str(output.dtype),
        "cache_path": str(cache_path),
        "metadata_path": str(metadata_path),
        "cache_sha256": sha256_file(cache_path),
        "runtime_seconds": time.perf_counter() - started,
        "torch_peak_memory_allocated_mib": int(torch.cuda.max_memory_allocated() / 2**20),
        "validated": True,
    }
    write_json(metadata_path, cache_metadata)
    return np.load(cache_path, mmap_mode="r"), cache_metadata


def evaluate_frozen_linear_head(
    full_embeddings: np.ndarray,
    lesion_embeddings: np.ndarray,
    target: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    seed: int,
    alpha: float,
    max_iter: int,
) -> dict[str, Any]:
    """Fit a fresh fold-local standardizer and deterministic linear head."""

    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    labels = np.asarray(target, dtype=np.int8)
    full = np.asarray(full_embeddings, dtype=np.float32)
    lesion = np.asarray(lesion_embeddings, dtype=np.float32)
    if (
        full.ndim != 2
        or lesion.ndim != 2
        or full.shape[0] != len(labels)
        or lesion.shape[0] != len(labels)
    ):
        raise RuntimeError("SIIM ablation embedding arrays have invalid shapes")
    if not np.isfinite(full).all() or not np.isfinite(lesion).all():
        raise RuntimeError("SIIM ablation embedding arrays contain non-finite values")
    oof = np.full(len(labels), np.nan, dtype=np.float64)
    coverage = np.zeros(len(labels), dtype=np.int8)
    records: list[dict[str, Any]] = []
    for fold, (fit_raw, valid_raw) in enumerate(splits):
        fit_index = np.asarray(fit_raw, dtype=np.int64)
        valid_index = np.asarray(valid_raw, dtype=np.int64)
        fit_matrix = np.concatenate([full[fit_index], lesion[fit_index]], axis=1)
        valid_matrix = np.concatenate([full[valid_index], lesion[valid_index]], axis=1)
        scaler = StandardScaler(copy=False)
        fit_matrix = scaler.fit_transform(fit_matrix)
        valid_matrix = scaler.transform(valid_matrix)
        classifier = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=float(alpha),
            max_iter=int(max_iter),
            tol=1e-4,
            shuffle=True,
            random_state=int(seed) + fold * 1009,
            class_weight="balanced",
            average=True,
        )
        started = time.perf_counter()
        classifier.fit(fit_matrix, labels[fit_index])
        prediction = classifier.decision_function(valid_matrix)
        auc = float(roc_auc_score(labels[valid_index], prediction))
        oof[valid_index] = prediction
        coverage[valid_index] += 1
        records.append({
            "fold": fold,
            "fit_rows": len(fit_index),
            "valid_rows": len(valid_index),
            "fit_positive": int(labels[fit_index].sum()),
            "valid_positive": int(labels[valid_index].sum()),
            "auc": auc,
            "linear_iterations": int(classifier.n_iter_),
            "runtime_seconds": time.perf_counter() - started,
            "standardizer_fit_scope": "fold_fit_only",
            "head_fit_scope": "fold_fit_only",
        })
    if not np.all(coverage == 1) or not np.isfinite(oof).all():
        raise RuntimeError("SIIM preprocessing ablation OOF coverage is incomplete")
    return {
        "fold_auc": [record["auc"] for record in records],
        "oof_auc": float(roc_auc_score(labels, oof)),
        "folds": records,
        "oof_prediction": oof,
    }


def evaluate_profiles_across_seeds(
    *,
    profile_embeddings: dict[str, dict[str, Any]],
    train_image_names: Sequence[str],
    target: np.ndarray,
    fold_contract: dict[str, Any],
    evaluation_seeds: Sequence[int],
    run_dir: Path,
    alpha: float,
    max_iter: int,
    logger: logging.Logger,
) -> tuple[
    dict[str, dict[int, list[float]]],
    dict[str, dict[str, dict[str, Any]]],
]:
    """Reuse each frozen embedding matrix across every external split seed."""

    normalized_seeds = [int(value) for value in evaluation_seeds]
    profile_seed_fold_auc: dict[str, dict[int, list[float]]] = {
        profile: {} for profile in recovery.SIIM_PREPROCESSING_PROFILES
    }
    profile_evaluations: dict[str, dict[str, dict[str, Any]]] = {
        profile: {} for profile in recovery.SIIM_PREPROCESSING_PROFILES
    }
    oof_dir = run_dir / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    fold_assignments_by_seed = {
        seed: pd.read_csv(fold_contract["assignment_paths"][seed])["fold"]
        for seed in normalized_seeds
    }
    for seed, assignments in fold_assignments_by_seed.items():
        if len(assignments) != len(target):
            raise RuntimeError(
                f"SIIM fold assignment length differs for evaluation seed {seed}"
            )
    for profile in recovery.SIIM_PREPROCESSING_PROFILES:
        full_embeddings = profile_embeddings[profile]["full_image"]
        lesion_embeddings = profile_embeddings[profile]["lesion_focus"]
        for evaluation_seed in normalized_seeds:
            evaluation = evaluate_frozen_linear_head(
                full_embeddings,
                lesion_embeddings,
                target,
                fold_contract["splits_by_seed"][evaluation_seed],
                seed=evaluation_seed,
                alpha=alpha,
                max_iter=max_iter,
            )
            profile_seed_fold_auc[profile][evaluation_seed] = evaluation["fold_auc"]
            oof_path = oof_dir / f"{profile}_s{evaluation_seed}_oof.csv"
            pd.DataFrame({
                "image_name": list(train_image_names),
                "target": target,
                "fold": fold_assignments_by_seed[evaluation_seed],
                "decision_score": evaluation.pop("oof_prediction"),
            }).to_csv(oof_path, index=False)
            evaluation["evaluation_seed"] = evaluation_seed
            evaluation["oof_path"] = oof_path.relative_to(run_dir).as_posix()
            evaluation["oof_sha256"] = sha256_file(oof_path)
            profile_evaluations[profile][str(evaluation_seed)] = evaluation
            logger.info(
                "profile=%s evaluation_seed=%s fold_auc=%s oof_auc=%.6f",
                profile,
                evaluation_seed,
                ",".join(f"{value:.6f}" for value in evaluation["fold_auc"]),
                evaluation["oof_auc"],
            )
    return profile_seed_fold_auc, profile_evaluations


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = Path(args.output_root).resolve()
    run_id = args.run_id or (
        f"siim_preprocessing_ablation_s{args.seed}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    run_dir = output_root / "runs" / run_id
    if compute_policy is not None and compute_policy.write_runner_blocked_if_hpc_only(
        project_root=PROJECT_ROOT,
        evidence_path=run_dir / "hpc_only_policy_block.json",
        runner_name=Path(__file__).name,
        run_id=run_id,
        output_root=output_root,
    ):
        return 0

    evaluation_seeds = parse_evaluation_seeds(args.evaluation_seeds)
    if args.folds < 3:
        raise ValueError("SIIM preprocessing ablation requires at least three folds")
    if args.full_backbone == args.lesion_backbone:
        raise ValueError("SIIM preprocessing ablation backbones must be distinct")
    if args.linear_alpha <= 0 or args.linear_max_iter <= 0:
        raise ValueError("SIIM preprocessing linear-head budget is invalid")
    if args.maximum_seed_mean_regression < 0:
        raise ValueError("SIIM maximum seed-mean regression must be non-negative")
    if not 0.0 < args.minimum_seed_pass_fraction <= 1.0:
        raise ValueError("SIIM minimum seed pass fraction must be in (0, 1]")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.torch_home is not None:
        os.environ["TORCH_HOME"] = str(Path(args.torch_home).resolve())

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("SIIM preprocessing ablation requires CUDA")
    memory_contract = recovery.configure_siim_cuda_memory_limit(
        torch,
        args.memory_limit_mib,
    )
    determinism = recovery.seed_siim_fold(args.seed, fast_kernel_mode=False)
    output_root.mkdir(parents=True, exist_ok=True)
    run_directory_action = prepare_run_directory(run_dir, resume=args.resume)
    cache_dir = output_root / "embedding_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(run_dir)
    started = time.perf_counter()
    logger.info(
        "run_id=%s status=starting run_directory_action=%s",
        run_id,
        run_directory_action,
    )

    gpu_properties = torch.cuda.get_device_properties(0)
    gpu_contract = {
        "name": torch.cuda.get_device_name(0),
        "total_memory_mib": int(gpu_properties.total_memory / 2**20),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "memory_limit_mib": memory_contract["memory_limit_mib"],
        "memory_fraction": memory_contract["memory_fraction"],
        "determinism": determinism,
    }
    write_json(run_dir / "gpu_preflight.json", gpu_contract)

    dataset = load_public_dataset(args.data_root)
    fold_contract = build_manifest_and_folds(
        dataset,
        run_dir=run_dir,
        folds=args.folds,
        evaluation_seeds=evaluation_seeds,
        workers=args.manifest_workers,
        precomputed_manifest=args.image_content_manifest,
    )
    adapter_path = Path(recovery.__file__).resolve()
    wave2_path = Path(wave2.__file__).resolve()
    cache_contract_base = {
        "image_vector_sha256": fold_contract["image_vector_sha256"],
        "image_content_manifest_sha256": fold_contract["manifest_sha256"],
        "adapter_source_sha256": FROZEN_EMBEDDING_ADAPTER_CONTRACT_SHA256,
        "wave2_source_sha256": sha256_file(wave2_path),
    }

    profile_embeddings: dict[str, dict[str, Any]] = {
        profile: {} for profile in recovery.SIIM_PREPROCESSING_PROFILES
    }
    weight_identities: dict[str, Any] = {}
    for view_name, backbone_name in (
        ("full_image", args.full_backbone),
        ("lesion_focus", args.lesion_backbone),
    ):
        torch.cuda.reset_peak_memory_stats()
        model, weight_identity = load_frozen_backbone(backbone_name, device="cuda")
        weight_identities[view_name] = {
            "backbone": backbone_name,
            "identity": weight_identity,
        }
        for profile in recovery.SIIM_PREPROCESSING_PROFILES:
            embeddings, metadata = extract_view_embeddings(
                paths=dataset["train_paths"],
                profile=profile,
                view_name=view_name,
                backbone_name=backbone_name,
                model=model,
                weight_identity=weight_identity,
                cache_dir=cache_dir,
                cache_contract_base=cache_contract_base,
                image_size=args.image_size,
                batch_size=args.batch_size,
                workers=args.workers,
                resume_cache=not args.no_resume_cache,
                logger=logger,
            )
            profile_embeddings[profile][view_name] = embeddings
            profile_embeddings[profile][f"{view_name}_metadata"] = metadata
        del model
        torch.cuda.empty_cache()

    profile_seed_fold_auc, profile_evaluations = evaluate_profiles_across_seeds(
        profile_embeddings=profile_embeddings,
        train_image_names=dataset["train"]["image_name"].astype(str).tolist(),
        target=dataset["target"],
        fold_contract=fold_contract,
        evaluation_seeds=evaluation_seeds,
        run_dir=run_dir,
        alpha=args.linear_alpha,
        max_iter=args.linear_max_iter,
        logger=logger,
    )

    report = recovery.select_siim_preprocessing_ablation(
        profile_seed_fold_auc,
        evaluation_seeds=evaluation_seeds,
        image_content_manifest_sha256=fold_contract["manifest_sha256"],
        minimum_mean_gain=args.minimum_mean_gain,
        maximum_worst_fold_regression=args.maximum_worst_fold_regression,
        maximum_seed_mean_regression=args.maximum_seed_mean_regression,
        minimum_seed_pass_fraction=args.minimum_seed_pass_fraction,
    )
    report.update({
        "created_at": utc_now(),
        "run_id": run_id,
        "competition_id": COMPETITION_ID,
        "full_public_train_scope": True,
        "public_train_rows": len(dataset["train"]),
        "public_test_rows_manifested": len(dataset["test"]),
        "split_strategy_by_seed": fold_contract["split_strategy_by_seed"],
        "split_records_by_seed": fold_contract["split_records_by_seed"],
        "patient_content_group_isolation": True,
        "leakage_group_policy": recovery.SIIM_LEAKAGE_GROUP_POLICY,
        "perceptual_edge_policy": recovery.SIIM_PERCEPTUAL_EDGE_POLICY,
        "validation_coverage_exactly_once": True,
        "frozen_backbone_embeddings": True,
        "pretrained_weight_identity": weight_identities,
        "fixed_pretrained_weight_sha256_verified": True,
        "image_size": int(args.image_size),
        "linear_head": {
            "family": "StandardScaler_fit_only_plus_SGDClassifier_log_loss",
            "alpha": float(args.linear_alpha),
            "max_iter": int(args.linear_max_iter),
            "class_weight": "balanced",
            "average": True,
            "fold_local_fit": True,
        },
        "profile_evaluations": profile_evaluations,
        "adapter_source": str(adapter_path),
        "adapter_source_sha256": sha256_file(adapter_path),
        "embedding_adapter_contract_sha256": cache_contract_base[
            "adapter_source_sha256"
        ],
        "embedding_cache_reuse_basis": (
            "grouping_only_adapter_change__image_preprocessing_and_embedding_"
            "extraction_contract_unchanged"
        ),
        "wave2_source": str(wave2_path),
        "wave2_source_sha256": cache_contract_base["wave2_source_sha256"],
        "gpu": gpu_contract,
        "runtime_seconds": time.perf_counter() - started,
        "official_score_claimed": False,
        "run_directory_action": run_directory_action,
    })
    report_path = run_dir / "siim_preprocessing_ablation.json"
    write_json(report_path, report)

    logger.info(
        "run_id=%s status=completed selected_profile=%s report=%s",
        run_id,
        report["selected_profile"],
        report_path,
    )
    for handler in logger.handlers:
        handler.flush()
        handler.close()
    logger.handlers.clear()

    artifacts: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts.append({
                "path": path.relative_to(run_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    for profile in recovery.SIIM_PREPROCESSING_PROFILES:
        for view_name in ("full_image", "lesion_focus"):
            metadata = profile_embeddings[profile][f"{view_name}_metadata"]
            cache_path = Path(metadata["cache_path"])
            metadata_path = Path(metadata["metadata_path"])
            artifacts.append({
                "path": str(cache_path),
                "bytes": cache_path.stat().st_size,
                "sha256": metadata["cache_sha256"],
                "role": "verified_embedding_cache",
            })
            artifacts.append({
                "path": str(metadata_path),
                "bytes": metadata_path.stat().st_size,
                "sha256": sha256_file(metadata_path),
                "role": "verified_embedding_cache_metadata",
            })
    artifact_manifest = {
        "schema": "evomind.siim_preprocessing_ablation_artifacts.v1",
        "created_at": utc_now(),
        "run_id": run_id,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    artifact_path = run_dir / "artifact_manifest.json"
    write_json(artifact_path, artifact_manifest)
    current_path = output_root / "siim_preprocessing_ablation_current.json"
    shutil.copy2(report_path, current_path)
    print(json.dumps({
        "ok": True,
        "run_id": run_id,
        "selected_profile": report["selected_profile"],
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "artifact_manifest": str(artifact_path),
        "artifact_manifest_sha256": sha256_file(artifact_path),
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
