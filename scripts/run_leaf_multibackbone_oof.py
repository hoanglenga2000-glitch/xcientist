#!/usr/bin/env python3
"""Leakage-safe three-seed Leaf dual-backbone OOF and withheld candidate builder."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (SRC_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

phase_a = import_module("research_os.mlebench_phase_a")
recovery = import_module("scripts.mlebench_medal_recovery_adapters")
wave2 = import_module("scripts.mlebench_wave2_adapters")
resolve_competition = phase_a.resolve_competition

try:
    import mlebench_compute_policy as compute_policy
except ModuleNotFoundError:
    try:
        from scripts import mlebench_compute_policy as compute_policy
    except (ModuleNotFoundError, ImportError):  # Standalone remote HPC source copy.
        compute_policy = None  # type: ignore[assignment]

COMPETITION_ID = "leaf-classification"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "workspace" / "local_gpu" / "leaf_multibackbone"
DEFAULT_BACKBONES = ("convnext_small", "efficientnet_v2_s")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_csv_integers(raw: str, *, minimum_count: int) -> list[int]:
    tokens = [token.strip() for token in str(raw).split(",")]
    if any(not token for token in tokens):
        raise ValueError("Leaf seeds must be comma-separated integers")
    try:
        values = [int(token) for token in tokens]
    except ValueError as exc:
        raise ValueError("Leaf seeds must be comma-separated integers") from exc
    if len(values) < minimum_count or len(set(values)) != len(values):
        raise ValueError(f"Leaf requires at least {minimum_count} distinct seeds")
    return values


def parse_backbones(raw: str) -> list[str]:
    values = [token.strip() for token in str(raw).split(",") if token.strip()]
    if len(values) < 2 or len(set(values)) != len(values):
        raise ValueError("Leaf requires at least two distinct frozen backbones")
    unsupported = [value for value in values if value not in wave2.VISION_BACKBONE_SPECS]
    if unsupported:
        raise ValueError(f"Unsupported Leaf backbones: {unsupported}")
    return values


def portable_unicode_array(values: Sequence[Any]) -> np.ndarray:
    """Return a fixed-width Unicode array that is safe with allow_pickle=False."""

    strings = [str(value) for value in values]
    width = max(1, max((len(value) for value in strings), default=1))
    result = np.asarray(strings, dtype=f"<U{width}")
    if result.dtype.kind != "U":
        raise RuntimeError("Leaf string serialization must use fixed-width Unicode")
    return result


class LeafEmbeddingDataset:
    """Module-level dataset so Windows spawn workers can pickle it."""

    def __init__(self, paths: Sequence[Path], image_size: int) -> None:
        from torchvision import transforms

        self.paths = [Path(path) for path in paths]
        self.image_size = int(image_size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        from PIL import Image, ImageOps

        resampling = getattr(Image, "Resampling", Image).BICUBIC
        with Image.open(self.paths[index]) as handle:
            image = handle.convert("RGB")
            contained = ImageOps.contain(
                image,
                (self.image_size, self.image_size),
                method=resampling,
            )
        canvas = Image.new(
            "RGB",
            (self.image_size, self.image_size),
            color=(255, 255, 255),
        )
        canvas.paste(
            contained,
            (
                (self.image_size - contained.width) // 2,
                (self.image_size - contained.height) // 2,
            ),
        )
        return self.normalize(self.to_tensor(canvas)), int(index)


def leaf_d4_codes(count: int) -> tuple[tuple[int, bool], ...]:
    if count == 1:
        return ((0, False),)
    if count == 4:
        return tuple((rotation, False) for rotation in range(4))
    if count == 8:
        return tuple(
            (rotation, flipped)
            for flipped in (False, True)
            for rotation in range(4)
        )
    raise ValueError("Leaf embedding TTA must be 1, 4, or 8")


def embedding_cache_contract(
    *,
    backbone: str,
    weight_sha256: str,
    image_manifest_sha256: str,
    image_size: int,
    tta_count: int,
    source_sha256: str,
) -> dict[str, Any]:
    if len(weight_sha256) != 64 or len(image_manifest_sha256) != 64:
        raise ValueError("Leaf cache contract contains an invalid SHA256")
    contract = {
        "schema": "evomind.leaf.frozen_embedding_cache.v1",
        "competition_id": COMPETITION_ID,
        "backbone": backbone,
        "weight_sha256": weight_sha256,
        "image_manifest_sha256": image_manifest_sha256,
        "image_size": int(image_size),
        "tta_count": int(tta_count),
        "source_sha256": source_sha256,
        "dtype": "float32",
        "private_labels_used": False,
    }
    contract["cache_key_sha256"] = sha256_json(contract)
    return contract


def _replace_classifier_with_identity(model: Any, backbone: str) -> int:
    import torch

    classifier_path = wave2.VISION_BACKBONE_SPECS[backbone]["classifier"]
    parent_path, index_text = classifier_path.rsplit(".", maxsplit=1)
    parent = wave2._resolve_named_module(model, parent_path)
    index = int(index_text)
    classifier = parent[index]
    dimension = int(classifier.in_features)
    parent[index] = torch.nn.Identity()
    return dimension


def load_or_extract_embeddings(
    *,
    paths: list[Path],
    backbone: str,
    image_manifest_sha256: str,
    cache_dir: Path,
    image_size: int,
    batch_size: int,
    workers: int,
    tta_count: int,
    seed: int,
    source_sha256: str,
    logger: logging.Logger,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available():
        raise RuntimeError("Leaf dual-backbone embedding extraction requires CUDA")
    spec = wave2.VISION_BACKBONE_SPECS[backbone]
    weight_path = Path(torch.hub.get_dir()) / "checkpoints" / spec["filename"]
    if not weight_path.is_file():
        raise FileNotFoundError(f"Pinned Leaf backbone weight is missing: {weight_path}")
    weight_sha256 = sha256_file(weight_path)
    if weight_sha256 != spec["sha256"]:
        raise RuntimeError(f"Pinned Leaf backbone weight hash mismatch: {backbone}")
    contract = embedding_cache_contract(
        backbone=backbone,
        weight_sha256=weight_sha256,
        image_manifest_sha256=image_manifest_sha256,
        image_size=image_size,
        tta_count=tta_count,
        source_sha256=source_sha256,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{backbone}__{contract['cache_key_sha256'][:20]}"
    cache_path = cache_dir / f"{stem}.npy"
    metadata_path = cache_dir / f"{stem}.json"
    if cache_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        cached = np.load(cache_path, mmap_mode="r")
        if (
            metadata.get("contract") == contract
            and metadata.get("rows") == len(paths)
            and metadata.get("cache_sha256") == sha256_file(cache_path)
            and cached.shape == (len(paths), metadata.get("width"))
            and np.isfinite(cached).all()
        ):
            logger.info("backbone=%s embedding_cache=verified_reuse", backbone)
            return cached, metadata

    dataset = LeafEmbeddingDataset(paths, image_size)
    worker_count = min(max(0, int(workers)), 8)
    options: dict[str, Any] = {
        "batch_size": max(1, int(batch_size)),
        "shuffle": False,
        "num_workers": worker_count,
        "pin_memory": True,
        "persistent_workers": worker_count > 0,
        "worker_init_fn": wave2._seed_vision_worker,
        "generator": torch.Generator().manual_seed(int(seed)),
    }
    if worker_count:
        options["prefetch_factor"] = wave2.VISION_DATALOADER_PREFETCH_FACTOR
    loader = DataLoader(dataset, **options)
    model, pretrained, identity = wave2._vision_model(
        1,
        backbone=backbone,
        require_pretrained=True,
    )
    width = _replace_classifier_with_identity(model, backbone)
    model = model.to(device="cuda", memory_format=torch.channels_last).eval()
    variants = leaf_d4_codes(tta_count)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    output = np.empty((len(paths), width), dtype=np.float32)
    covered = np.zeros(len(paths), dtype=bool)
    previous_benchmark = bool(torch.backends.cudnn.benchmark)
    started = time.perf_counter()
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.inference_mode():
            for batch_index, (images, indices) in enumerate(loader):
                images = images.cuda(
                    non_blocking=True,
                    memory_format=torch.channels_last,
                )
                aggregate = None
                for rotation, flipped in variants:
                    transformed = torch.rot90(images, rotation, dims=(2, 3))
                    if flipped:
                        transformed = torch.flip(transformed, dims=(3,))
                    transformed = transformed.contiguous(memory_format=torch.channels_last)
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        values = model(transformed).float()
                    aggregate = values if aggregate is None else aggregate + values
                assert aggregate is not None
                aggregate = torch.nn.functional.normalize(
                    aggregate / len(variants),
                    dim=1,
                )
                index_values = np.asarray(indices, dtype=np.int64)
                output[index_values] = aggregate.cpu().numpy()
                covered[index_values] = True
                if batch_index % 10 == 0 or bool(covered.all()):
                    elapsed = max(time.perf_counter() - started, 1e-6)
                    logger.info(
                        "backbone=%s rows=%d/%d images_per_second=%.2f",
                        backbone,
                        int(covered.sum()),
                        len(paths),
                        int(covered.sum()) / elapsed,
                    )
        if not covered.all() or not np.isfinite(output).all():
            raise RuntimeError("Leaf frozen embeddings are incomplete or non-finite")
        peak_memory = int(torch.cuda.max_memory_allocated() / 2**20)
    finally:
        torch.backends.cudnn.benchmark = previous_benchmark
        del model
        torch.cuda.empty_cache()
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, output, allow_pickle=False)
    temporary.replace(cache_path)
    metadata = {
        "schema": "evomind.leaf.frozen_embedding_artifact.v1",
        "created_at": now_iso(),
        "contract": contract,
        "pretrained": pretrained,
        "pretrained_weight_identity": identity,
        "rows": int(output.shape[0]),
        "width": int(output.shape[1]),
        "cache_path": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "metadata_path": str(metadata_path),
        "amp_dtype": str(amp_dtype),
        "workers": worker_count,
        "batch_size": int(batch_size),
        "runtime_seconds": time.perf_counter() - started,
        "peak_memory_allocated_mib": peak_memory,
        "validated": True,
    }
    write_json_atomic(metadata_path, metadata)
    return np.load(cache_path, mmap_mode="r"), metadata


def apply_blend(
    components: Sequence[np.ndarray],
    weights: Sequence[float],
    temperature: float,
) -> np.ndarray:
    matrices = [np.asarray(value, dtype=np.float64) for value in components]
    if not matrices or len(matrices) != len(weights):
        raise ValueError("Leaf blend component and weight counts differ")
    if any(matrix.shape != matrices[0].shape for matrix in matrices):
        raise ValueError("Leaf blend component shapes differ")
    weight_array = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weight_array).all() or float(weight_array.sum()) <= 0:
        raise ValueError("Leaf blend weights are invalid")
    weight_array = weight_array / weight_array.sum()
    probability = np.zeros_like(matrices[0], dtype=np.float64)
    for weight, matrix in zip(weight_array, matrices):
        probability += float(weight) * np.clip(matrix, 1e-12, 1.0)
    logits = np.log(np.clip(probability, 1e-12, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits)
    return result / result.sum(axis=1, keepdims=True)


def fit_blend(
    components: Sequence[np.ndarray],
    truth: np.ndarray,
    classes: Sequence[str],
) -> dict[str, Any]:
    from scipy.optimize import minimize
    from scipy.special import softmax

    matrices = [np.asarray(value, dtype=np.float64) for value in components]
    target_index = {str(value): index for index, value in enumerate(classes)}
    truth_index = np.asarray([target_index[str(value)] for value in truth], dtype=np.int64)
    count = len(matrices)
    if count < 2 or any(matrix.shape != matrices[0].shape for matrix in matrices):
        raise ValueError("Leaf meta blend requires aligned probability components")

    def objective(parameters: np.ndarray) -> float:
        weights = softmax(parameters[:count])
        temperature = float(np.exp(parameters[-1]))
        probability = apply_blend(matrices, weights, temperature)
        nll = -np.log(
            np.clip(probability[np.arange(len(truth_index)), truth_index], 1e-12, 1.0)
        ).mean()
        regularization = 1e-4 * float(np.square(weights - 1.0 / count).sum())
        return float(nll + regularization)

    starts = [np.zeros(count + 1, dtype=np.float64)]
    for index in range(count):
        start = np.full(count + 1, -2.0, dtype=np.float64)
        start[index] = 2.0
        start[-1] = 0.0
        starts.append(start)
    best: tuple[float, Any] | None = None
    bounds = [(-8.0, 8.0)] * count + [(np.log(0.25), np.log(2.5))]
    for start in starts:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        score = objective(result.x)
        if best is None or score < best[0]:
            best = (score, result)
    assert best is not None
    result = best[1]
    weights = softmax(result.x[:count])
    temperature = float(np.exp(result.x[-1]))
    probability = apply_blend(matrices, weights, temperature)
    nll = -np.log(
        np.clip(probability[np.arange(len(truth_index)), truth_index], 1e-12, 1.0)
    ).mean()
    return {
        "weights": weights.tolist(),
        "temperature": temperature,
        "log_loss": float(nll),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }


def cross_fit_blend(
    components: Sequence[np.ndarray],
    truth: np.ndarray,
    classes: Sequence[str],
    folds: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    from sklearn.metrics import log_loss

    matrices = [np.asarray(value, dtype=np.float64) for value in components]
    prediction = np.full_like(matrices[0], np.nan, dtype=np.float64)
    records: list[dict[str, Any]] = []
    for fold in sorted(int(value) for value in np.unique(folds)):
        validation = folds == fold
        fitting = ~validation
        selected = fit_blend(
            [matrix[fitting] for matrix in matrices],
            truth[fitting],
            classes,
        )
        prediction[validation] = apply_blend(
            [matrix[validation] for matrix in matrices],
            selected["weights"],
            selected["temperature"],
        )
        records.append(
            {
                "fold": fold,
                "weights": selected["weights"],
                "temperature": selected["temperature"],
                "meta_fit_log_loss": selected["log_loss"],
                "outer_log_loss": float(
                    log_loss(
                        truth[validation],
                        prediction[validation],
                        labels=list(classes),
                    )
                ),
            }
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError("Leaf cross-fitted meta blend did not cover every row")
    return prediction, records


def aligned_probability(model: Any, features: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    raw = np.asarray(model.predict_proba(features), dtype=np.float64)
    model_classes = [str(value) for value in model.classes_]
    if set(model_classes) != set(classes):
        raise RuntimeError("Leaf model classes differ from the submission classes")
    index = {value: position for position, value in enumerate(model_classes)}
    result = np.column_stack([raw[:, index[str(value)]] for value in classes])
    result = np.clip(result, 1e-12, 1.0)
    return result / result.sum(axis=1, keepdims=True)


def build_component_specs(
    *,
    numeric_train: np.ndarray,
    numeric_test: np.ndarray,
    embedding_train: dict[str, np.ndarray],
    embedding_test: dict[str, np.ndarray],
    seed: int,
    numeric_c: float,
    image_c: float,
    multimodal_c: float,
) -> list[tuple[str, np.ndarray, np.ndarray, Any]]:
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import Normalizer, StandardScaler
    from sklearn.svm import SVC

    def svc(c_value: float) -> SVC:
        return SVC(
            C=float(c_value),
            gamma="scale",
            probability=True,
            random_state=int(seed),
            cache_size=4096,
        )

    specs: list[tuple[str, np.ndarray, np.ndarray, Any]] = [
        (
            "numeric_rbf_svc",
            numeric_train,
            numeric_test,
            Pipeline([("scale", StandardScaler()), ("model", svc(numeric_c))]),
        )
    ]
    ordered_backbones = list(embedding_train)
    for backbone in ordered_backbones:
        specs.append(
            (
                f"image_{backbone}_rbf_svc",
                embedding_train[backbone],
                embedding_test[backbone],
                Pipeline([("normalize", Normalizer()), ("model", svc(image_c))]),
            )
        )
    image_train = np.concatenate(
        [embedding_train[name] for name in ordered_backbones],
        axis=1,
    )
    image_test = np.concatenate(
        [embedding_test[name] for name in ordered_backbones],
        axis=1,
    )
    specs.append(
        (
            "image_dual_backbone_rbf_svc",
            image_train,
            image_test,
            Pipeline([("normalize", Normalizer()), ("model", svc(image_c))]),
        )
    )
    multimodal_train = np.concatenate([numeric_train, image_train], axis=1)
    multimodal_test = np.concatenate([numeric_test, image_test], axis=1)
    numeric_width = numeric_train.shape[1]
    transformers: list[tuple[str, Any, slice]] = [
        (
            "numeric",
            Pipeline([("scale", StandardScaler()), ("normalize", Normalizer())]),
            slice(0, numeric_width),
        )
    ]
    cursor = numeric_width
    for backbone in ordered_backbones:
        width = embedding_train[backbone].shape[1]
        transformers.append((backbone, Normalizer(), slice(cursor, cursor + width)))
        cursor += width
    multimodal_preprocessor = ColumnTransformer(
        transformers,
        transformer_weights={name: 1.0 for name, _, _ in transformers},
        sparse_threshold=0.0,
    )
    specs.append(
        (
            "multimodal_group_balanced_rbf_svc",
            multimodal_train,
            multimodal_test,
            Pipeline(
                [
                    ("groups", multimodal_preprocessor),
                    ("model", svc(multimodal_c)),
                ]
            ),
        )
    )
    return specs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--seeds", default="40,41,42")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--backbones", default=",".join(DEFAULT_BACKBONES))
    parser.add_argument("--image-size", type=int, default=288)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--manifest-workers", type=int, default=8)
    parser.add_argument("--tta", type=int, choices=(1, 4, 8), default=8)
    parser.add_argument("--numeric-c", type=float, default=10.0)
    parser.add_argument("--image-c", type=float, default=8.0)
    parser.add_argument("--multimodal-c", type=float, default=10.0)
    parser.add_argument("--promotion-mean-log-loss", type=float, default=0.0135)
    parser.add_argument("--promotion-max-seed-log-loss", type=float, default=0.0135)
    parser.add_argument("--torch-home", type=Path)
    return parser


def configure_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"leaf-multibackbone-{run_dir.name}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (
        logging.FileHandler(run_dir / "run.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = args.output_root.resolve()
    run_id = args.run_id or f"leaf_multibackbone_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = output_root / "runs" / run_id
    if compute_policy is not None and compute_policy.write_runner_blocked_if_hpc_only(
        project_root=PROJECT_ROOT,
        evidence_path=run_dir / "hpc_only_policy_block.json",
        runner_name=Path(__file__).name,
        run_id=run_id,
        output_root=output_root,
    ):
        return 0

    import joblib
    from sklearn.metrics import log_loss
    from sklearn.model_selection import StratifiedGroupKFold

    seeds = parse_csv_integers(args.seeds, minimum_count=3)
    backbones = parse_backbones(args.backbones)
    if args.folds < 3:
        raise ValueError("Leaf requires at least three grouped folds")
    if min(args.numeric_c, args.image_c, args.multimodal_c) <= 0:
        raise ValueError("Leaf SVC regularization values must be positive")
    if args.torch_home is not None:
        os.environ["TORCH_HOME"] = str(args.torch_home.resolve())

    output_root.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    logger = configure_logger(run_dir)
    started = time.perf_counter()

    resolved = resolve_competition(COMPETITION_ID, args.data_root.resolve())
    train = pd.read_csv(resolved.public_dir / "train.csv").reset_index(drop=True)
    test = pd.read_csv(resolved.public_dir / "test.csv").reset_index(drop=True)
    sample = pd.read_csv(resolved.sample_submission_path).reset_index(drop=True)
    required_train = {"id", "species"}
    if not required_train <= set(train) or "id" not in test or sample.columns[0] != "id":
        raise RuntimeError("Leaf public-data schema is incomplete")
    if train["id"].duplicated().any() or test["id"].duplicated().any():
        raise RuntimeError("Leaf train or test IDs are duplicated")
    classes = [str(value) for value in sample.columns[1:]]
    target = portable_unicode_array(train["species"].astype(str).tolist())
    if set(target) != set(classes):
        raise RuntimeError("Leaf labels do not equal the submission class set")
    feature_columns = [value for value in train if value not in {"id", "species"}]
    numeric_train = train[feature_columns].to_numpy(dtype=np.float64)
    numeric_test = test[feature_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_train).all() or not np.isfinite(numeric_test).all():
        raise RuntimeError("Leaf numeric features contain non-finite values")

    train_paths = recovery.resolve_leaf_image_paths(resolved.public_dir, train["id"])
    test_paths = recovery.resolve_leaf_image_paths(resolved.public_dir, test["id"])
    train_manifest = recovery.verify_image_decode_manifest(
        train_paths,
        split="train",
        workers=args.manifest_workers,
    )
    test_manifest = recovery.verify_image_decode_manifest(
        test_paths,
        split="test",
        workers=args.manifest_workers,
    )
    train_manifest.insert(2, "id", train["id"])
    train_manifest.insert(3, "species", target)
    test_manifest.insert(2, "id", test["id"])
    manifest = pd.concat([train_manifest, test_manifest], ignore_index=True)
    manifest_path = run_dir / "leaf_image_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    manifest_sha256 = sha256_file(manifest_path)
    conflicts = train_manifest.groupby("sha256")["species"].nunique()
    if bool((conflicts > 1).any()):
        raise RuntimeError("Leaf exact-duplicate images have conflicting labels")
    cross_source = set(train_manifest["sha256"]) & set(test_manifest["sha256"])
    duplicate_audit = {
        "schema": "evomind.leaf.image_duplicate_audit.v1",
        "train_rows": len(train_manifest),
        "test_rows": len(test_manifest),
        "train_duplicate_rows": int(train_manifest["sha256"].duplicated(keep=False).sum()),
        "train_test_exact_hash_count": len(cross_source),
        "train_test_exact_hashes": sorted(cross_source),
        "conflicting_train_hashes": 0,
    }
    write_json_atomic(run_dir / "leaf_image_duplicate_audit.json", duplicate_audit)

    source_sha256 = sha256_file(Path(__file__).resolve())
    all_paths = train_paths + test_paths
    embedding_train: dict[str, np.ndarray] = {}
    embedding_test: dict[str, np.ndarray] = {}
    embedding_metadata: dict[str, Any] = {}
    cache_dir = output_root / "embedding_cache"
    for backbone in backbones:
        embeddings, metadata = load_or_extract_embeddings(
            paths=all_paths,
            backbone=backbone,
            image_manifest_sha256=manifest_sha256,
            cache_dir=cache_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            workers=args.workers,
            tta_count=args.tta,
            seed=seeds[0],
            source_sha256=source_sha256,
            logger=logger,
        )
        embedding_train[backbone] = np.asarray(embeddings[: len(train)])
        embedding_test[backbone] = np.asarray(embeddings[len(train) :])
        embedding_metadata[backbone] = metadata

    group_values = train_manifest["sha256"].astype(str).to_numpy()
    per_seed: list[dict[str, Any]] = []
    seed_test_predictions: list[np.ndarray] = []
    model_root = run_dir / "fold_models"
    model_root.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        splitter = StratifiedGroupKFold(
            n_splits=args.folds,
            shuffle=True,
            random_state=seed,
        )
        initial_specs = build_component_specs(
            numeric_train=numeric_train,
            numeric_test=numeric_test,
            embedding_train=embedding_train,
            embedding_test=embedding_test,
            seed=seed,
            numeric_c=args.numeric_c,
            image_c=args.image_c,
            multimodal_c=args.multimodal_c,
        )
        component_names = [value[0] for value in initial_specs]
        oof = [np.zeros((len(train), len(classes)), dtype=np.float64) for _ in initial_specs]
        test_components = [
            np.zeros((len(test), len(classes)), dtype=np.float64) for _ in initial_specs
        ]
        fold_assignment = np.full(len(train), -1, dtype=np.int16)
        fold_records: list[dict[str, Any]] = []
        for fold, (fit_indices, valid_indices) in enumerate(
            splitter.split(numeric_train, target, group_values)
        ):
            if set(group_values[fit_indices]) & set(group_values[valid_indices]):
                raise RuntimeError("Leaf exact-image group crossed a fold boundary")
            fold_assignment[valid_indices] = fold
            specs = build_component_specs(
                numeric_train=numeric_train,
                numeric_test=numeric_test,
                embedding_train=embedding_train,
                embedding_test=embedding_test,
                seed=seed * 100 + fold,
                numeric_c=args.numeric_c,
                image_c=args.image_c,
                multimodal_c=args.multimodal_c,
            )
            record: dict[str, Any] = {
                "fold": fold,
                "fit_rows": len(fit_indices),
                "valid_rows": len(valid_indices),
                "group_overlap_count": 0,
            }
            fold_dir = model_root / f"s{seed}_f{fold}"
            fold_dir.mkdir(parents=True, exist_ok=False)
            for index, (name, train_features, test_features, model) in enumerate(specs):
                model.fit(train_features[fit_indices], target[fit_indices])
                valid_probability = aligned_probability(
                    model,
                    train_features[valid_indices],
                    classes,
                )
                oof[index][valid_indices] = valid_probability
                test_components[index] += aligned_probability(
                    model,
                    test_features,
                    classes,
                ) / args.folds
                record[f"{name}_log_loss"] = float(
                    log_loss(target[valid_indices], valid_probability, labels=classes)
                )
                joblib.dump(model, fold_dir / f"{name}.joblib", compress=3)
            fold_records.append(record)
            logger.info("seed=%d fold=%d completed", seed, fold)
        if np.any(fold_assignment < 0):
            raise RuntimeError("Leaf fold assignment did not cover every training row")
        crossfit, crossfit_records = cross_fit_blend(
            oof,
            target,
            classes,
            fold_assignment,
        )
        final_blend = fit_blend(oof, target, classes)
        seed_test = apply_blend(
            test_components,
            final_blend["weights"],
            final_blend["temperature"],
        )
        seed_score = float(log_loss(target, crossfit, labels=classes))
        seed_test_predictions.append(seed_test)
        seed_path = run_dir / f"leaf_multibackbone_s{seed}.npz"
        np.savez_compressed(
            seed_path,
            train_id=train["id"].to_numpy(),
            target=portable_unicode_array(target),
            classes=portable_unicode_array(classes),
            fold_assignment=fold_assignment,
            component_names=portable_unicode_array(component_names),
            crossfit_probability=crossfit,
            test_id=test["id"].to_numpy(),
            test_probability=seed_test,
            **{f"oof__{name}": value for name, value in zip(component_names, oof)},
            **{
                f"test__{name}": value
                for name, value in zip(component_names, test_components)
            },
        )
        per_seed.append(
            {
                "seed": seed,
                "cross_fitted_log_loss": seed_score,
                "component_names": component_names,
                "folds": fold_records,
                "crossfit_blend_records": crossfit_records,
                "deployment_blend": final_blend,
                "artifact": str(seed_path),
                "artifact_sha256": sha256_file(seed_path),
            }
        )
        logger.info("seed=%d cross_fitted_log_loss=%.8f", seed, seed_score)

    final_probability = np.mean(seed_test_predictions, axis=0)
    final_probability /= final_probability.sum(axis=1, keepdims=True)
    submission = recovery.align_multiclass_submission(
        sample,
        test["id"],
        final_probability,
        classes,
    )
    submission_path = run_dir / "submission_withheld.csv"
    submission.to_csv(submission_path, index=False)
    seed_scores = [value["cross_fitted_log_loss"] for value in per_seed]
    mean_score = float(np.mean(seed_scores))
    max_score = float(np.max(seed_scores))
    gate_passed = (
        mean_score <= args.promotion_mean_log_loss
        and max_score <= args.promotion_max_seed_log_loss
    )
    report = {
        "schema": "evomind.leaf.multibackbone_oof.v1",
        "created_at": now_iso(),
        "status": "promotion_gate_passed" if gate_passed else "promotion_gate_failed",
        "run_id": run_id,
        "competition_id": COMPETITION_ID,
        "requested_model": "gpt-5.6-sol",
        "served_model": "gpt-5.6-sol",
        "full_public_train_scope": True,
        "train_rows": len(train),
        "test_rows": len(test),
        "class_count": len(classes),
        "numeric_feature_count": len(feature_columns),
        "backbones": backbones,
        "embedding_metadata": embedding_metadata,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "duplicate_audit": duplicate_audit,
        "external_seeds": seeds,
        "folds": args.folds,
        "per_seed": per_seed,
        "mean_cross_fitted_log_loss": mean_score,
        "maximum_seed_cross_fitted_log_loss": max_score,
        "promotion_gate": {
            "passed": gate_passed,
            "mean_log_loss_threshold": args.promotion_mean_log_loss,
            "maximum_seed_log_loss_threshold": args.promotion_max_seed_log_loss,
        },
        "submission_path": str(submission_path),
        "submission_sha256": sha256_file(submission_path),
        "source_sha256": source_sha256,
        "runtime_seconds": time.perf_counter() - started,
        "private_labels_used": False,
        "private_scores_used_for_tuning": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "official_score_claimed": False,
    }
    report_path = run_dir / "leaf_multibackbone_oof.json"
    write_json_atomic(report_path, report)
    artifacts: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts.append(
                {
                    "path": path.relative_to(run_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    artifact_manifest = {
        "schema": "evomind.leaf.multibackbone_artifacts.v1",
        "created_at": now_iso(),
        "run_id": run_id,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    artifact_path = run_dir / "artifact_manifest.json"
    write_json_atomic(artifact_path, artifact_manifest)
    shutil.copy2(report_path, output_root / "leaf_multibackbone_current.json")
    print(
        json.dumps(
            {
                "ok": True,
                "status": report["status"],
                "run_id": run_id,
                "mean_cross_fitted_log_loss": mean_score,
                "maximum_seed_cross_fitted_log_loss": max_score,
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "artifact_manifest": str(artifact_path),
                "artifact_manifest_sha256": sha256_file(artifact_path),
                "official_grader_executed": False,
                "kaggle_submission_executed": False,
            },
            ensure_ascii=False,
        )
    )
    return 0 if gate_passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
