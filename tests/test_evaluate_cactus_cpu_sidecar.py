from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts import evaluate_cactus_cpu_sidecar as sidecar


def _bundles(tmp_path: Path) -> tuple[Path, Path]:
    truth = np.tile(np.asarray([0, 1], dtype=np.int8), 50)
    fold = np.arange(100, dtype=np.int16) % 5
    gpu_oof = np.where(truth == 1, 0.7, 0.3) + np.sin(np.arange(100)) * 0.15
    cpu_oof = np.where(truth == 1, 0.8, 0.2) + np.cos(np.arange(100)) * 0.05
    gpu = tmp_path / "gpu.npz"
    cpu = tmp_path / "cpu.npz"
    np.savez_compressed(
        gpu,
        truth=truth,
        selected_oof_probability=gpu_oof,
        selected_test_probability=np.asarray([0.4, 0.6]),
    )
    np.savez_compressed(
        cpu,
        truth=truth,
        fold=fold,
        xgb_oof=cpu_oof,
        candidate_test=np.asarray([0.3, 0.7]),
    )
    return gpu, cpu


def test_nested_sidecar_evaluation_preserves_cardinality(tmp_path: Path):
    gpu, cpu = _bundles(tmp_path)
    report, arrays = sidecar.evaluate(gpu, cpu, [0.0, 0.1, 0.2])

    assert report["nested_auc"] >= report["gpu_baseline_auc"]
    assert arrays["selected_oof_probability"].shape == (100,)
    assert arrays["selected_test_probability"].shape == (2,)


def test_parse_weights_fails_closed():
    assert sidecar.parse_weights("0,0.1,0.2") == [0.0, 0.1, 0.2]
    with pytest.raises(Exception):
        sidecar.parse_weights("0.1,0.2")
