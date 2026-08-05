#!/usr/bin/env python3
"""PUBLIC_ONLY duplicate-safe CPU ensemble diagnostic for Aerial Cactus."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

SCHEMA = "evomind.mlebench.cactus_cpu_ensemble_diagnostic.v1"
COMPETITION = "aerial-cactus-identification"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def image_features(array: np.ndarray) -> np.ndarray:
    """Extract deterministic color, low-resolution, grid, and gradient features."""
    rgb = np.asarray(array, dtype=np.float32) / np.float32(255.0)
    if rgb.shape != (32, 32, 3):
        raise ValueError(f"Expected a 32x32 RGB image, got {rgb.shape}")
    low = rgb.reshape(8, 4, 8, 4, 3).mean(axis=(1, 3)).reshape(-1)
    cells = rgb.reshape(4, 8, 4, 8, 3).transpose(0, 2, 1, 3, 4)
    cell_mean = cells.mean(axis=(2, 3)).reshape(-1)
    cell_std = cells.std(axis=(2, 3)).reshape(-1)
    hist = np.concatenate(
        [np.histogram(rgb[:, :, channel], bins=16, range=(0.0, 1.0))[0] / 1024.0 for channel in range(3)]
    ).astype(np.float32)
    gray = rgb @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    magnitude = np.hypot(gx, gy)
    orientation = (np.arctan2(gy, gx) + np.pi) * np.float32(8.0 / (2.0 * np.pi))
    bins = np.floor(orientation).astype(np.int16) % 8
    hog = []
    for row in range(4):
        for column in range(4):
            region_bins = bins[row * 8 : (row + 1) * 8, column * 8 : (column + 1) * 8]
            region_mag = magnitude[row * 8 : (row + 1) * 8, column * 8 : (column + 1) * 8]
            hog.extend(np.bincount(region_bins.ravel(), weights=region_mag.ravel(), minlength=8))
    global_stats = np.concatenate(
        [rgb.mean(axis=(0, 1)), rgb.std(axis=(0, 1)), rgb.min(axis=(0, 1)), rgb.max(axis=(0, 1))]
    )
    return np.asarray(
        np.concatenate([low, cell_mean, cell_std, hist, np.asarray(hog), global_stats]),
        dtype=np.float32,
    )


def read_zip_features(path: Path, ordered_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    hashes: list[str] = []
    with zipfile.ZipFile(path) as archive:
        members = {Path(info.filename).name: info for info in archive.infolist() if not info.is_dir()}
        if set(ordered_ids) != set(members):
            raise RuntimeError(f"ZIP member identity mismatch for {path.name}")
        for name in ordered_ids:
            raw = archive.read(members[name])
            hashes.append(hashlib.sha256(raw).hexdigest())
            with Image.open(io.BytesIO(raw)) as image:
                features.append(image_features(np.asarray(image.convert("RGB"))))
    return np.ascontiguousarray(features, dtype=np.float32), np.asarray(hashes, dtype="U64")


def build_models(seed: int, threads: int, trees: int):
    from xgboost import XGBClassifier

    extra = ExtraTreesClassifier(
        n_estimators=int(trees),
        max_features=0.70,
        min_samples_leaf=1,
        class_weight="balanced",
        n_jobs=int(threads),
        random_state=int(seed),
    )
    xgb = XGBClassifier(
        n_estimators=int(trees),
        max_depth=6,
        learning_rate=0.035,
        min_child_weight=2.0,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=1.0,
        reg_alpha=0.01,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=int(threads),
        random_state=int(seed + 10_000),
    )
    return extra, xgb


def run(args: argparse.Namespace) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    allowed = args.allowed_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    for label, path in (("data", data_root), ("output", output)):
        try:
            path.relative_to(allowed)
        except ValueError as exc:
            raise RuntimeError(f"Cactus {label} path escaped allowed root") from exc
    public = data_root / COMPETITION / "prepared" / "public"
    inputs = {name: public / name for name in ("train.csv", "train.zip", "test.zip", "sample_submission.csv")}
    if any(not path.is_file() or path.is_symlink() for path in inputs.values()):
        raise RuntimeError("Cactus PUBLIC_ONLY inputs are incomplete or unsafe")
    if any("private" in {part.lower() for part in path.parts} for path in inputs.values()):
        raise RuntimeError("Cactus diagnostic rejected a private path")

    started = time.perf_counter()
    train = pd.read_csv(inputs["train.csv"])
    sample = pd.read_csv(inputs["sample_submission.csv"])
    if list(train.columns) != ["id", "has_cactus"] or list(sample.columns) != ["id", "has_cactus"]:
        raise RuntimeError("Cactus CSV schema drifted")
    train_ids = train["id"].astype(str).tolist()
    test_ids = sample["id"].astype(str).tolist()
    y = train["has_cactus"].to_numpy(dtype=np.int8)
    x, train_hash = read_zip_features(inputs["train.zip"], train_ids)
    x_test, test_hash = read_zip_features(inputs["test.zip"], test_ids)
    groups, uniques = pd.factorize(train_hash, sort=True)
    duplicate_rows = int(len(train_hash) - len(uniques))
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_extra = np.zeros(len(y), dtype=np.float64)
    oof_xgb = np.zeros(len(y), dtype=np.float64)
    test_extra = np.zeros((args.folds, len(test_ids)), dtype=np.float64)
    test_xgb = np.zeros((args.folds, len(test_ids)), dtype=np.float64)
    fold_assignment = np.full(len(y), -1, dtype=np.int16)
    fold_records = []
    for fold, (fit, valid) in enumerate(splitter.split(x, y, groups)):
        extra, xgb = build_models(args.seed + 1009 * fold, args.threads, args.trees)
        extra.fit(x[fit], y[fit])
        xgb.fit(x[fit], y[fit])
        oof_extra[valid] = extra.predict_proba(x[valid])[:, 1]
        oof_xgb[valid] = xgb.predict_proba(x[valid])[:, 1]
        fold_assignment[valid] = fold
        test_extra[fold] = extra.predict_proba(x_test)[:, 1]
        test_xgb[fold] = xgb.predict_proba(x_test)[:, 1]
        fold_records.append(
            {
                "fold": fold,
                "fit_rows": int(len(fit)),
                "valid_rows": int(len(valid)),
                "extra_auc": float(roc_auc_score(y[valid], oof_extra[valid])),
                "xgb_auc": float(roc_auc_score(y[valid], oof_xgb[valid])),
            }
        )
    oof_blend = 0.30 * oof_extra + 0.70 * oof_xgb
    scores = {
        "extra_trees_auc": float(roc_auc_score(y, oof_extra)),
        "xgboost_auc": float(roc_auc_score(y, oof_xgb)),
        "predeclared_blend_auc": float(roc_auc_score(y, oof_blend)),
    }
    test_prediction = 0.30 * test_extra.mean(axis=0) + 0.70 * test_xgb.mean(axis=0)
    label_by_hash: dict[str, float] = {}
    conflicts = 0
    for digest, label in zip(train_hash, y, strict=True):
        if digest in label_by_hash and label_by_hash[digest] != float(label):
            label_by_hash.pop(digest, None)
            conflicts += 1
        elif digest not in label_by_hash:
            label_by_hash[digest] = float(label)
    exact_matches = 0
    for index, digest in enumerate(test_hash):
        if digest in label_by_hash:
            test_prediction[index] = label_by_hash[digest]
            exact_matches += 1

    output.mkdir(parents=True, exist_ok=True)
    bundle = output / "cactus_cpu_oof_and_test.npz"
    np.savez_compressed(
        bundle,
        truth=y,
        fold=fold_assignment,
        extra_oof=oof_extra,
        xgb_oof=oof_xgb,
        candidate_oof=oof_blend,
        candidate_test=test_prediction,
        train_id=np.asarray(train_ids),
        test_id=np.asarray(test_ids),
    )
    submission = sample.copy()
    submission["has_cactus"] = test_prediction
    submission_path = output / "candidate_submission_withheld.csv"
    submission.to_csv(submission_path, index=False)
    report = {
        "schema": SCHEMA,
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "status": "candidate_ready" if scores["predeclared_blend_auc"] >= args.promotion_auc else "diagnostic_gate_failed",
        "competition_id": COMPETITION,
        "visibility_mode": "PUBLIC_ONLY",
        "seed": args.seed,
        "folds": args.folds,
        "trees": args.trees,
        "train_rows": len(y),
        "test_rows": len(test_ids),
        "feature_count": int(x.shape[1]),
        "duplicate_rows": duplicate_rows,
        "conflicting_duplicate_hashes": conflicts,
        "exact_public_train_test_matches": exact_matches,
        "scores": scores,
        "fold_records": fold_records,
        "promotion_auc": args.promotion_auc,
        "candidate_ready": scores["predeclared_blend_auc"] >= args.promotion_auc,
        "inputs": {name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for name, path in inputs.items()},
        "prediction_bundle": {"path": str(bundle), "bytes": bundle.stat().st_size, "sha256": sha256_file(bundle)},
        "submission": {"path": str(submission_path), "bytes": submission_path.stat().st_size, "sha256": sha256_file(submission_path)},
        "runtime_seconds": time.perf_counter() - started,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "gpu_used": False,
        "claim_boundary": "Public duplicate-safe OOF diagnostic only; not an official score or medal.",
    }
    atomic_json(output / "diagnostic_report.json", report)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allowed-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=700)
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--promotion-auc", type=float, default=0.9997)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
