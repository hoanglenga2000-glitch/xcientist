from __future__ import annotations

import os

from scripts import precompute_taxi_public_cache as cli


def test_cli_exposes_frozen_cache_contract_and_verify_mode():
    args = cli.parse_args(["--output-dir", "/tmp/cache", "--verify-only"])

    assert args.verify_only is True
    assert args.max_rows == 5_000_000
    assert args.chunk_rows == 250_000
    assert args.cache_seed == 42
    assert args.folds == 3
    assert args.holdout_fraction == 0.05


def test_runner_import_forces_cpu_only_visibility(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    runner = cli._load_runner()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert runner.TAXI_PUBLIC_CACHE_SCHEMA.endswith(".v1")
