from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import diagnose_may2022_cpu_feature_model as diagnostic


def frame(rows: int = 6) -> pd.DataFrame:
    payload: dict[str, object] = {"id": np.arange(rows, dtype=np.int64)}
    for index in range(31):
        if index == 27:
            payload["f_27"] = ["ABCDEFGHIJ", "JIHGFEDCBA", "AAAAAAAAAA"] * (rows // 3)
        elif index in list(range(7, 19)) + [29, 30]:
            payload[f"f_{index:02d}"] = np.arange(rows, dtype=np.int64) % 3
        else:
            payload[f"f_{index:02d}"] = np.linspace(-2.0, 2.0, rows)
    payload["target"] = np.arange(rows, dtype=np.int8) % 2
    return pd.DataFrame(payload)


def test_category_codes_return_separate_equal_length_frequencies() -> None:
    fit = pd.Series(["b", "a", "b", "c"])
    validation = pd.Series(["a", "b", "c", "unknown"])

    fit_code, validation_code, fit_frequency, validation_frequency, mapping = (
        diagnostic._fit_category_codes(fit, validation)
    )

    assert mapping == {"a": 0, "b": 1, "c": 2}
    assert fit_code.tolist() == [1, 0, 1, 2]
    assert validation_code.tolist() == [0, 1, 2, -1]
    assert fit_frequency.tolist() == pytest.approx([0.5, 0.25, 0.5, 0.25])
    assert validation_frequency.tolist() == pytest.approx([0.25, 0.5, 0.25, 0.0])


def test_category_codes_support_unequal_fit_and_validation_lengths() -> None:
    fit = pd.Series(["a", "a", "b", "c", "c", "c"])
    validation = pd.Series(["c", "unknown"])

    fit_code, validation_code, fit_frequency, validation_frequency, mapping = (
        diagnostic._fit_category_codes(fit, validation)
    )

    assert mapping == {"a": 0, "b": 1, "c": 2}
    assert fit_code.tolist() == [0, 0, 1, 2, 2, 2]
    assert validation_code.tolist() == [2, -1]
    assert fit_frequency.tolist() == pytest.approx(
        [2 / 6, 2 / 6, 1 / 6, 3 / 6, 3 / 6, 3 / 6]
    )
    assert validation_frequency.tolist() == pytest.approx([3 / 6, 0.0])


def test_fold_features_are_finite_aligned_and_fold_local() -> None:
    fit = frame(6)
    validation = frame(6)
    validation.loc[0, "f_07"] = 999
    train_features, valid_features, categorical, metadata = (
        diagnostic.build_fold_categorical_features(fit, validation)
    )

    assert train_features.columns.tolist() == valid_features.columns.tolist()
    assert np.isfinite(train_features.to_numpy(dtype=np.float64)).all()
    assert np.isfinite(valid_features.to_numpy(dtype=np.float64)).all()
    assert valid_features.loc[0, "cat__f_07"] == -1
    assert valid_features.loc[0, "freq__f_07"] == pytest.approx(0.0)
    assert metadata["target_derived_features"] == 0
    assert metadata["frequency_fit_scope"] == "outer_fit_only"
    assert set(categorical) <= set(train_features.columns)
    assert len(metadata["feature_schema_sha256"]) == 64


def test_fold_features_support_unequal_fit_and_validation_lengths() -> None:
    fit = frame(9)
    validation = frame(6)
    validation.loc[0, "f_07"] = 999

    fit_features, validation_features, _, metadata = (
        diagnostic.build_fold_categorical_features(fit, validation)
    )

    assert len(fit_features) == 9
    assert len(validation_features) == 6
    assert fit_features.columns.tolist() == validation_features.columns.tolist()
    assert validation_features.loc[0, "cat__f_07"] == -1
    assert validation_features.loc[0, "freq__f_07"] == pytest.approx(0.0)
    assert metadata["frequency_fit_scope"] == "outer_fit_only"


def test_raw_schema_rejects_wrong_feature_identity() -> None:
    value = frame(6).rename(columns={"f_03": "f_31"})
    with pytest.raises(RuntimeError, match="raw schema differs"):
        diagnostic.validate_public_frame(value, require_target=True)
