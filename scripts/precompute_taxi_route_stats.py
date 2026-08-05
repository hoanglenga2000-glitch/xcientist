#!/usr/bin/env python3
"""Build the fixed-split Taxi fold-local route-stat sidecar on a CPU node."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

try:
    import run_mlebench_lite_full as full
except ModuleNotFoundError:
    from scripts import run_mlebench_lite_full as full

SCHEMA = "evomind.mlebench.taxi_route_stat_sidecar.v1"
ROUTE_INPUT_COLUMNS = (
    "pickup_cell_id",
    "dropoff_cell_id",
    "hour",
    "airport_route_code",
    "passenger_count",
    "pickup_borough_proxy",
    "dropoff_borough_proxy",
)
ROUTE_OUTPUT_COLUMNS = (
    "route_stat_route_cell",
    "route_stat_route_hour",
    "route_stat_airport_route",
    "route_stat_borough_route",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def route_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.ascontiguousarray(frame[list(ROUTE_OUTPUT_COLUMNS)].to_numpy(dtype=np.float32))


def build_context(
    train_route: pd.DataFrame,
    target: np.ndarray,
    fit_index: np.ndarray,
    valid_index: np.ndarray,
    *,
    test_route: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    apply = [train_route.iloc[valid_index].reset_index(drop=True)]
    if test_route is not None:
        apply.append(test_route.reset_index(drop=True))
    encoded = full.build_taxi_fold_local_route_statistics(
        train_route.iloc[fit_index].reset_index(drop=True),
        target[fit_index],
        *apply,
    )
    train_values = np.empty((len(train_route), len(ROUTE_OUTPUT_COLUMNS)), dtype=np.float32)
    train_values[fit_index] = route_matrix(encoded[0])
    train_values[valid_index] = route_matrix(encoded[1])
    test_values = route_matrix(encoded[2]) if test_route is not None else None
    return train_values, test_values


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    allowed = args.allowed_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    for label, path in (("base cache", args.base_cache_dir.expanduser().resolve()), ("output", output)):
        try:
            path.relative_to(allowed)
        except ValueError as exc:
            raise RuntimeError(f"Taxi route-stat {label} escaped allowed root") from exc
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Taxi route-stat output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    loaded = full.load_taxi_public_feature_cache(
        args.base_cache_dir,
        data_root=args.data_root,
        allowed_root=args.allowed_root,
        max_rows=args.max_rows,
        chunk_rows=args.chunk_rows,
        seed=args.cache_seed,
        folds=args.folds,
        holdout_fraction=args.holdout_fraction,
    )
    names = list(loaded["feature_names"])
    missing = sorted(set(ROUTE_INPUT_COLUMNS) - set(names))
    if missing:
        raise RuntimeError(f"Taxi base cache lacks route inputs: {missing}")
    indices = [names.index(name) for name in ROUTE_INPUT_COLUMNS]
    train_route = pd.DataFrame(loaded["train_features"][:, indices], columns=ROUTE_INPUT_COLUMNS)
    test_route = pd.DataFrame(loaded["test_features"][:, indices], columns=ROUTE_INPUT_COLUMNS)
    target = np.asarray(loaded["target"], dtype=np.float64)
    splits = loaded["splits"]
    artifacts: list[dict[str, Any]] = []

    def save(name: str, values: np.ndarray) -> None:
        path = output / name
        atomic_npy(path, values)
        artifacts.append({"relative_path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})

    for fold in range(args.folds):
        train_values, test_values = build_context(
            train_route,
            target,
            np.asarray(splits[f"fold_{fold:02d}_fit"], dtype=np.int64),
            np.asarray(splits[f"fold_{fold:02d}_valid"], dtype=np.int64),
            test_route=test_route,
        )
        save(f"oof_fold_{fold:02d}_train.npy", train_values)
        assert test_values is not None
        save(f"oof_fold_{fold:02d}_test.npy", test_values)

    for context in ("temporal", "geographic"):
        train_values, _ = build_context(
            train_route,
            target,
            np.asarray(splits[f"{context}_fit"], dtype=np.int64),
            np.asarray(splits[f"{context}_valid"], dtype=np.int64),
        )
        save(f"{context}_train.npy", train_values)

    all_index = np.arange(len(target), dtype=np.int64)
    refit_train, refit_test = full.build_taxi_fold_local_route_statistics(
        train_route,
        target,
        test_route,
    )
    save("refit_train.npy", route_matrix(refit_train))
    save("refit_test.npy", route_matrix(refit_test))
    del all_index

    function_source = inspect.getsource(full.build_taxi_fold_local_route_statistics).encode("utf-8")
    manifest = {
        "schema": SCHEMA,
        "status": "completed",
        "competition_id": "new-york-city-taxi-fare-prediction",
        "visibility_mode": "PUBLIC_ONLY",
        "cache_seed": args.cache_seed,
        "folds": args.folds,
        "max_rows": args.max_rows,
        "chunk_rows": args.chunk_rows,
        "holdout_fraction": args.holdout_fraction,
        "train_rows": len(target),
        "test_rows": len(test_route),
        "route_input_columns": list(ROUTE_INPUT_COLUMNS),
        "route_output_columns": list(ROUTE_OUTPUT_COLUMNS),
        "algorithm_source_sha256": hashlib.sha256(function_source).hexdigest(),
        "base_cache_manifest": loaded["manifest_record"],
        "artifacts": artifacts,
        "contexts": ["oof_fold_00", "oof_fold_01", "oof_fold_02", "temporal", "geographic", "refit"],
        "contracts": {
            "validation_targets_used": 0,
            "private_files_read": [],
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "gpu_used": False,
            "manifest_written_last": True,
        },
        "claim_boundary": "PUBLIC_ONLY throughput sidecar; not a score or medal.",
    }
    manifest_path = output / "route_stat_manifest.json"
    atomic_json(manifest_path, manifest)
    return manifest


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allowed-root", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=5_000_000)
    parser.add_argument("--chunk-rows", type=int, default=250_000)
    parser.add_argument("--cache-seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--holdout-fraction", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
