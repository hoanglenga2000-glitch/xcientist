#!/usr/bin/env python3
"""Build or verify the public-only May-2022 feature cache on a CPU node."""
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


def _load_recovery() -> Any:
    # This must happen before importing the adapter or any optional ML runtime.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for path in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        import mlebench_medal_recovery_adapters as recovery
    except ModuleNotFoundError:
        from scripts import mlebench_medal_recovery_adapters as recovery
    return recovery


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allowed-root", type=Path, default=DEFAULT_ALLOWED_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    recovery = _load_recovery()
    if args.verify_only:
        loaded = recovery.load_may2022_public_feature_cache(
            args.output_dir,
            data_root=args.data_root,
            allowed_root=args.allowed_root,
            seed=args.seed,
            folds=args.folds,
        )
        payload = {
            "schema": "evomind.mlebench.may2022_public_feature_cache_verification.v1",
            "status": "verified",
            "manifest": loaded["manifest_record"],
            "train_shape": list(loaded["train_features"].shape),
            "test_shape": list(loaded["test_features"].shape),
            "visibility_mode": "PUBLIC_ONLY",
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "gpu_used": False,
        }
    else:
        record = recovery.build_may2022_public_feature_cache(
            data_root=args.data_root,
            output_dir=args.output_dir,
            allowed_root=args.allowed_root,
            seed=args.seed,
            folds=args.folds,
            source_paths=(Path(__file__),),
        )
        loaded = recovery.load_may2022_public_feature_cache(
            args.output_dir,
            data_root=args.data_root,
            allowed_root=args.allowed_root,
            seed=args.seed,
            folds=args.folds,
        )
        payload = {
            "schema": "evomind.mlebench.may2022_public_feature_cache_build.v1",
            "status": "completed_and_verified",
            "record": record,
            "manifest": loaded["manifest_record"],
            "train_shape": list(loaded["train_features"].shape),
            "test_shape": list(loaded["test_features"].shape),
            "visibility_mode": "PUBLIC_ONLY",
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "gpu_used": False,
        }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
