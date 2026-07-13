"""Tests for auto-detection of Kaggle competition metadata.

The strongest validation is reproducing the hand-curated COMPETITIONS registry
entries (titanic, house_prices, digit-recognizer) from synthetic-but-faithful
data, so onboarding a new competition yields configs consistent with the ones a
human already tuned.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

# Load the script module by path (scripts/ is not a package). The module must be
# registered in sys.modules before exec so dataclass introspection can resolve it.
_SPEC = importlib.util.spec_from_file_location(
    "auto_detect_competition",
    Path(__file__).resolve().parents[1] / "scripts" / "auto_detect_competition.py",
)
adc = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = adc
_SPEC.loader.exec_module(adc)


def test_detect_id_column_prefers_sample_submission():
    train = pd.DataFrame({"PassengerId": [1, 2], "Survived": [0, 1], "Age": [22, 38]})
    sample = pd.DataFrame({"PassengerId": [3, 4], "Survived": [0, 0]})
    assert adc.detect_id_column(train, sample) == "PassengerId"


def test_detect_id_column_by_name_and_uniqueness():
    train = pd.DataFrame({"id": [10, 11, 12], "x": [1, 1, 2]})
    assert adc.detect_id_column(train) == "id"


def test_detect_id_column_none_for_pixel_data():
    # digit-recognizer: label + pixel columns, no id column.
    train = pd.DataFrame({"label": [0, 1, 2], "pixel0": [0, 5, 9], "pixel1": [3, 3, 3]})
    assert adc.detect_id_column(train) is None


def test_detect_task_type_string_target_is_classification():
    train = pd.DataFrame({"class": ["STAR", "GALAXY", "QSO", "STAR"]})
    assert adc.detect_task_type(train, "class") == "classification"


def test_detect_task_type_binary_integer_is_classification():
    train = pd.DataFrame({"Survived": [0, 1, 1, 0, 1]})
    assert adc.detect_task_type(train, "Survived") == "classification"


def test_detect_task_type_continuous_is_regression():
    train = pd.DataFrame({"SalePrice": [120000.5, 250000.0, 310000.75, 99000.1]})
    assert adc.detect_task_type(train, "SalePrice") == "regression"


def test_detect_metric_imbalanced_binary_is_roc_auc():
    # 10% positive -> imbalanced -> roc_auc (porto-seguro-like).
    train = pd.DataFrame({"target": [0] * 90 + [1] * 10})
    assert adc.detect_metric(train, "target", "classification") == "roc_auc"


def test_detect_metric_balanced_binary_is_accuracy():
    train = pd.DataFrame({"Transported": [0, 1] * 50})
    assert adc.detect_metric(train, "Transported", "classification") == "accuracy"


def test_detect_metric_positive_named_regression_is_rmsle():
    train = pd.DataFrame({"SalePrice": [100.0, 200.0, 300.0]})
    assert adc.detect_metric(train, "SalePrice", "regression") == "rmsle"


def test_detect_metric_generic_regression_is_rmse():
    train = pd.DataFrame({"MedHouseVal": [-1.2, 0.5, 3.4]})
    assert adc.detect_metric(train, "MedHouseVal", "regression") == "rmse"


def test_detect_drop_columns_flags_id_and_freetext():
    train = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4],
            "Name": ["Alice A", "Bob B", "Cara C", "Dan D"],  # near-unique free text
            "Sex": ["m", "f", "m", "f"],
            "Survived": [0, 1, 0, 1],
        }
    )
    drop = adc.detect_drop_columns(train, target_col="Survived", id_col="PassengerId")
    assert "PassengerId" in drop
    assert "Name" in drop
    assert "Sex" not in drop  # low-cardinality categorical is kept
    assert "Survived" not in drop  # target never dropped


def test_detect_drop_columns_flags_all_null():
    train = pd.DataFrame({"id": [1, 2], "empty": [None, None], "y": [0, 1]})
    drop = adc.detect_drop_columns(train, target_col="y", id_col="id")
    assert "empty" in drop


# ── End-to-end: reproduce hand-curated registry entries ──

def test_e2e_titanic_matches_registry():
    train = pd.DataFrame(
        {
            "PassengerId": range(1, 21),
            "Survived": [0, 1] * 10,
            "Pclass": [3, 1] * 10,
            "Name": [f"Person {i}" for i in range(20)],  # free text
            "Sex": (["male", "female"] * 10),
            "Age": list(range(20, 40)),
        }
    )
    sample = pd.DataFrame({"PassengerId": range(21, 31), "Survived": [0] * 10})
    cfg = adc.detect_competition_config(train, sample)
    assert cfg.type == "classification"
    assert cfg.metric == "accuracy"
    assert cfg.target == "Survived"
    assert cfg.id_col == "PassengerId"
    assert "PassengerId" in cfg.drop_cols
    assert "Name" in cfg.drop_cols
    assert cfg.higher_is_better is True


def test_e2e_house_prices_matches_registry():
    rng = np.random.default_rng(0)
    train = pd.DataFrame(
        {
            "Id": range(1, 101),
            "LotArea": rng.integers(5000, 15000, 100),
            "OverallQual": rng.integers(1, 10, 100),
            "SalePrice": rng.integers(80000, 400000, 100).astype(float),
        }
    )
    sample = pd.DataFrame({"Id": range(101, 151), "SalePrice": [0.0] * 50})
    cfg = adc.detect_competition_config(train, sample)
    assert cfg.type == "regression"
    assert cfg.metric == "rmsle"
    assert cfg.target == "SalePrice"
    assert cfg.id_col == "Id"
    assert cfg.higher_is_better is False


def test_e2e_digit_recognizer_no_id():
    rng = np.random.default_rng(1)
    train = pd.DataFrame({"label": rng.integers(0, 10, 50)})
    for p in range(5):
        train[f"pixel{p}"] = rng.integers(0, 255, 50)
    # digit-recognizer sample_submission has ImageId + Label, but train has no id.
    cfg = adc.detect_competition_config(train, target_override="label")
    assert cfg.type == "classification"
    assert cfg.metric == "accuracy"
    assert cfg.target == "label"
    assert cfg.n_classes is not None and cfg.n_classes <= 10


def test_e2e_to_registry_entry_shape():
    train = pd.DataFrame({"id": range(10), "target": [0, 1] * 5, "f1": range(10)})
    sample = pd.DataFrame({"id": range(10, 20), "target": [0] * 10})
    entry = adc.detect_competition_config(train, sample).to_registry_entry()
    assert set(entry.keys()) == {
        "type", "metric", "target", "id_col", "drop_cols", "higher_is_better", "bronze",
    }
    import json
    json.dumps(entry)  # must be JSON-serializable for registry persistence


def test_e2e_missing_target_raises():
    train = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError):
        adc.detect_competition_config(train, target_override="does_not_exist")
