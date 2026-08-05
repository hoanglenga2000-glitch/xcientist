from __future__ import annotations

import numpy as np
import pandas as pd
import hashlib
import inspect
import json

from scripts import precompute_taxi_route_stats as route
from scripts import run_mlebench_lite_full as full


def test_build_context_preserves_original_row_order_and_hides_validation_targets():
    frame = pd.DataFrame(
        {
            "pickup_cell_id": [1, 1, 2, 2, 3, 3],
            "dropoff_cell_id": [4, 4, 5, 5, 6, 6],
            "hour": [8, 8, 9, 9, 10, 10],
            "airport_route_code": [0, 0, 1, 1, 0, 0],
            "passenger_count": [1, 1, 2, 2, 1, 1],
            "pickup_borough_proxy": [1, 1, 2, 2, 3, 3],
            "dropoff_borough_proxy": [2, 2, 3, 3, 4, 4],
        }
    )
    target = np.asarray([10.0, 12.0, 20.0, 22.0, 30.0, 32.0])
    fit = np.asarray([0, 2, 4])
    valid = np.asarray([1, 3, 5])
    first, _ = route.build_context(frame, target, fit, valid)
    changed = target.copy()
    changed[valid] += 10_000
    second, _ = route.build_context(frame, changed, fit, valid)

    assert first.shape == (6, 4)
    np.testing.assert_allclose(first, second)
    assert np.isfinite(first).all()


def test_cli_defaults_bind_taxi_cache_contract():
    args = route.parse_args(
        ["--data-root", "/data", "--base-cache-dir", "/cache", "--output-dir", "/out", "--allowed-root", "/"]
    )
    assert args.max_rows == 5_000_000
    assert args.chunk_rows == 250_000
    assert args.cache_seed == 42
    assert args.folds == 3


def test_runner_loader_verifies_route_sidecar_inventory(tmp_path):
    base = tmp_path / "base"
    sidecar = tmp_path / "route"
    base.mkdir()
    sidecar.mkdir()
    base_manifest = base / "cache_manifest.json"
    base_manifest.write_text('{"fixture":true}\n', encoding="utf-8")
    records = []
    for name in full.TAXI_ROUTE_STAT_ARTIFACTS:
        rows = 2 if name.endswith("_test.npy") else 6
        path = sidecar / name
        np.save(path, np.zeros((rows, 4), dtype=np.float32), allow_pickle=False)
        records.append(
            {
                "relative_path": name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "schema": full.TAXI_ROUTE_STAT_SIDECAR_SCHEMA,
        "status": "completed",
        "competition_id": "new-york-city-taxi-fare-prediction",
        "visibility_mode": "PUBLIC_ONLY",
        "cache_seed": 42,
        "folds": 3,
        "max_rows": 6,
        "chunk_rows": 3,
        "holdout_fraction": 0.2,
        "train_rows": 6,
        "test_rows": 2,
        "route_output_columns": list(full.TAXI_ROUTE_STAT_COLUMNS),
        "algorithm_source_sha256": hashlib.sha256(
            inspect.getsource(full.build_taxi_fold_local_route_statistics).encode()
        ).hexdigest(),
        "base_cache_manifest": {
            "bytes": base_manifest.stat().st_size,
            "sha256": hashlib.sha256(base_manifest.read_bytes()).hexdigest(),
        },
        "artifacts": records,
        "contracts": {
            "validation_targets_used": 0,
            "private_files_read": [],
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "gpu_used": False,
        },
    }
    (sidecar / "route_stat_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    loaded = full.load_taxi_route_stat_sidecar(
        sidecar,
        base_cache_dir=base,
        allowed_root=tmp_path,
        max_rows=6,
        chunk_rows=3,
        cache_seed=42,
        folds=3,
        holdout_fraction=0.2,
    )

    assert set(loaded["arrays"]) == {name.removesuffix(".npy") for name in full.TAXI_ROUTE_STAT_ARTIFACTS}
