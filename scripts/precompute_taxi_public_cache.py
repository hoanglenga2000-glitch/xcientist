#!/usr/bin/env python3
"""Build or verify the public-only Taxi feature cache on a CPU node."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALLOWED_ROOT = Path("/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra")
DEFAULT_DATA_ROOT = DEFAULT_ALLOWED_ROOT / "mlebench_official_data"


def _load_runner() -> Any:
    # Enforce CPU-only operation before importing any optional ML runtime.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for path in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        import run_mlebench_lite_full as runner
    except ModuleNotFoundError:
        from scripts import run_mlebench_lite_full as runner
    return runner


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allowed-root", type=Path, default=DEFAULT_ALLOWED_ROOT)
    parser.add_argument("--max-rows", type=int, default=5_000_000)
    parser.add_argument("--chunk-rows", type=int, default=250_000)
    parser.add_argument("--cache-seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--holdout-fraction", type=float, default=0.05)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    runner = _load_runner()
    load_kwargs = {
        "data_root": args.data_root,
        "allowed_root": args.allowed_root,
        "max_rows": args.max_rows,
        "chunk_rows": args.chunk_rows,
        "seed": args.cache_seed,
        "folds": args.folds,
        "holdout_fraction": args.holdout_fraction,
    }
    if args.verify_only:
        loaded = runner.load_taxi_public_feature_cache(
            args.output_dir, **load_kwargs
        )
        payload = {
            "schema": "evomind.mlebench.taxi_public_feature_cache_verification.v1",
            "status": "verified",
            "manifest": loaded["manifest_record"],
            "train_shape": list(loaded["train_features"].shape),
            "test_shape": list(loaded["test_features"].shape),
        }
    else:
        manifest = runner.build_taxi_public_feature_cache(
            output_dir=args.output_dir, **load_kwargs
        )
        loaded = runner.load_taxi_public_feature_cache(
            args.output_dir, **load_kwargs
        )
        payload = {
            "schema": "evomind.mlebench.taxi_public_feature_cache_build.v1",
            "status": "completed_and_verified",
            "build_seconds": manifest["build_seconds"],
            "manifest": loaded["manifest_record"],
            "train_shape": list(loaded["train_features"].shape),
            "test_shape": list(loaded["test_features"].shape),
        }
    payload.update(
        {
            "visibility_mode": "PUBLIC_ONLY",
            "cache_seed": args.cache_seed,
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "gpu_used": False,
        }
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
