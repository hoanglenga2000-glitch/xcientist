"""Tests for the new-competition onboarding orchestrator.

Uses self-contained temp CSVs so the suite never depends on local datasets,
network, GPU, or the Kaggle CLI. Verifies the orchestrator stays side-effect-free
by default and never mutates the live trainer.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

_ROOT = Path(__file__).resolve().parents[1]


def _load(module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _ROOT / "scripts" / f"{module_name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


onb = _load("onboard_new_competition")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("titanic", "titanic"),
        ("https://www.kaggle.com/c/titanic", "titanic"),
        ("https://www.kaggle.com/competitions/spaceship-titanic/data", "spaceship-titanic"),
        ("kaggle.com/c/house-prices/", "house-prices"),
        ("  playground-series-s6e6  ", "playground-series-s6e6"),
    ],
)
def test_parse_competition_name(raw, expected):
    assert onb.parse_competition_name(raw) == expected


def _make_comp(tmp_path: Path) -> Path:
    data_dir = tmp_path / "mycomp"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(100),
            "feat_a": list(range(100)),
            "feat_b": ["x", "y"] * 50,
            "target": [0, 1] * 50,
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": range(100, 150), "target": [0] * 50}).to_csv(
        data_dir / "sample_submission.csv", index=False
    )
    return data_dir


def test_onboard_detects_config(tmp_path: Path):
    data_dir = _make_comp(tmp_path)
    report = onb.onboard("mycomp", data_dir=data_dir)
    assert report["status"] if "status" in report else True
    entry = report["registry_entry"]
    assert entry["type"] == "classification"
    assert entry["target"] == "target"
    assert entry["id_col"] == "id"
    assert report["n_rows"] == 100


def test_onboard_is_side_effect_free(tmp_path: Path):
    data_dir = _make_comp(tmp_path)
    trainer = _ROOT / "scripts" / "gpu_batch_trainer_v1.py"
    before = trainer.read_bytes()
    onb.onboard("mycomp", data_dir=data_dir)
    # The orchestrator must never mutate the live trainer source.
    assert trainer.read_bytes() == before


def test_onboard_missing_data_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        onb.onboard("ghost", data_dir=tmp_path / "does_not_exist")


def test_onboard_missing_train_raises(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        onb.onboard("empty", data_dir=empty)


def test_onboard_report_is_json_serializable(tmp_path: Path):
    data_dir = _make_comp(tmp_path)
    report = onb.onboard("mycomp", data_dir=data_dir)
    json.dumps(report)  # must not raise


def test_onboard_respects_target_override(tmp_path: Path):
    data_dir = tmp_path / "reg"
    data_dir.mkdir()
    pd.DataFrame(
        {"Id": range(50), "x": range(50), "SalePrice": [float(i * 1000) for i in range(1, 51)]}
    ).to_csv(data_dir / "train.csv", index=False)
    report = onb.onboard("reg", data_dir=data_dir, target_override="SalePrice")
    assert report["registry_entry"]["type"] == "regression"
    assert report["registry_entry"]["target"] == "SalePrice"
