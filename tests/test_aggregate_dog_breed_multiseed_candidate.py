from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts import aggregate_dog_breed_multiseed_candidate as aggregate


def _bundle(path: Path, *, loss: float, reverse_classes: bool = False) -> dict:
    truth = np.arange(9199, dtype=np.int64) % 120
    true_probability = float(np.exp(-loss))
    oof = np.full((9199, 120), (1.0 - true_probability) / 119.0, dtype=np.float32)
    oof[np.arange(9199), truth] = true_probability
    test = np.full((1023, 120), 1.0 / 120.0, dtype=np.float32)
    classes = np.asarray([f"breed_{index:03d}" for index in range(120)], dtype=np.str_)
    if reverse_classes:
        classes = classes[::-1]
    np.savez_compressed(
        path,
        train_id=np.asarray([f"train_{index:05d}" for index in range(9199)], dtype=np.str_),
        test_id=np.asarray([f"test_{index:05d}" for index in range(1023)], dtype=np.str_),
        class_names=classes,
        truth=truth,
        selected_oof_probability=oof,
        selected_test_probability=test,
        fold=np.arange(9199, dtype=np.int16) % 5,
    )
    score = aggregate.multiclass_log_loss(truth, oof)
    return {
        "bundle_path": path,
        "result": {"cv_score": score},
        "seed": 46,
        "run_id": path.stem,
    }


def test_multiseed_recomputes_seed_and_ensemble_log_loss(tmp_path: Path) -> None:
    first = _bundle(tmp_path / "seed46.npz", loss=0.030)
    second = _bundle(tmp_path / "seed47.npz", loss=0.035)

    result = aggregate._load_and_verify([first, second])

    assert result["seed_log_loss"] == pytest.approx([0.030, 0.035], abs=1e-6)
    assert result["ensemble_log_loss"] <= max(result["seed_log_loss"])
    assert result["ensemble_test"].shape == (1023, 120)
    assert np.allclose(result["ensemble_test"].sum(axis=1), 1.0, atol=1e-12)
    assert result["class_names"][0] == "breed_000"


def test_multiseed_rejects_class_order_drift(tmp_path: Path) -> None:
    first = _bundle(tmp_path / "seed46.npz", loss=0.030)
    second = _bundle(tmp_path / "seed47.npz", loss=0.031, reverse_classes=True)

    with pytest.raises(aggregate.DogAggregationError, match="class_names"):
        aggregate._load_and_verify([first, second])


def test_bundle_rejects_probability_rows_that_do_not_sum_to_one(tmp_path: Path) -> None:
    item = _bundle(tmp_path / "invalid.npz", loss=0.030)
    with np.load(item["bundle_path"], allow_pickle=False) as source:
        payload = {name: np.asarray(source[name]) for name in source.files}
    payload["selected_oof_probability"][0] *= 0.5
    np.savez_compressed(item["bundle_path"], **payload)

    with pytest.raises(aggregate.DogAggregationError, match="sum to one"):
        aggregate._load_bundle(item)
