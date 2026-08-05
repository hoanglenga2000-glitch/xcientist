from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "run_may2022_compact_embedding_oof.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "run_may2022_compact_embedding_oof", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frame(rows: int = 20) -> pd.DataFrame:
    data = {
        f"f_{index:02d}": np.linspace(-1.0, 1.0, rows, dtype=np.float32) + index
        for index in range(31)
        if index != 27
    }
    data["f_27"] = ["ABCDEFGHIJ", "AAAAAAAAAA"] * (rows // 2)
    return pd.DataFrame(data)


def test_f27_encoding_is_fixed_width_and_rejects_non_alphabet() -> None:
    module = load_module()
    encoded = module.encode_f27(pd.Series(["ABCDEFGHIJ", "TTTTTTTTTT"]))
    assert encoded.shape == (2, 10)
    assert encoded[0].tolist() == list(range(10))
    assert encoded[1].tolist() == [19] * 10
    with pytest.raises(ValueError, match="exactly"):
        module.encode_f27(pd.Series(["SHORT"]))
    with pytest.raises(ValueError, match="outside A-T"):
        module.encode_f27(pd.Series(["ABCDEFGHIU"]))


def test_compact_feature_contract_has_exact_interactions_and_no_ordinal_statistics() -> None:
    module = load_module()
    data = frame()
    matrix, characters, names = module.build_numeric_features(data)
    assert matrix.shape == (len(data), 60)
    assert characters.shape == (len(data), 10)
    assert "f27_pos_0_code" not in names
    assert "f27_code_mean" not in names
    first = data.iloc[0]
    assert matrix[0, names.index("sum_f02_f21")] == pytest.approx(
        first["f_02"] + first["f_21"]
    )
    assert matrix[0, names.index("sum_f05_f22")] == pytest.approx(
        first["f_05"] + first["f_22"]
    )
    assert matrix[0, names.index("sum_f00_f01_f26")] == pytest.approx(
        first["f_00"] + first["f_01"] + first["f_26"]
    )
    assert names.count("f27_count_A") == 1
    assert np.isfinite(matrix).all()


def test_outer_and_inner_splits_are_deterministic_and_strictly_isolated() -> None:
    module = load_module()
    target = np.tile(np.array([0, 1], dtype=np.int8), 250)
    first = module.build_outer_folds(target, folds=5, seed=42)
    second = module.build_outer_folds(target, folds=5, seed=42)
    assert np.array_equal(first, second)
    assert set(np.unique(first).tolist()) == {0, 1, 2, 3, 4}
    outer_validation = np.flatnonzero(first == 0)
    outer_fit = np.flatnonzero(first != 0)
    inner_fit, inner_validation = module.build_inner_split(
        outer_fit, target, validation_fraction=0.1, seed=123
    )
    assert not (set(inner_fit) & set(inner_validation))
    assert not (set(outer_validation) & set(inner_fit))
    assert not (set(outer_validation) & set(inner_validation))
    assert set(inner_fit) | set(inner_validation) == set(outer_fit)


def test_fold_mean_scale_uses_only_supplied_indices() -> None:
    module = load_module()
    values = np.array([[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]], dtype=np.float32)
    mean, scale = module.fold_mean_scale(values, np.array([0, 1]))
    assert mean.tolist() == pytest.approx([2.0, 3.0])
    assert scale.tolist() == pytest.approx([1.0, 1.0])


def test_compact_embedding_model_cpu_forward_is_finite() -> None:
    torch = pytest.importorskip("torch")
    module = load_module()
    model = module.build_model(
        60,
        width=64,
        blocks=2,
        embedding_dim=4,
        dropout=0.05,
        mean=np.zeros(60, dtype=np.float32),
        scale=np.ones(60, dtype=np.float32),
    )
    numeric = torch.zeros((8, 60), dtype=torch.float32)
    characters = torch.zeros((8, 10), dtype=torch.long)
    output = model(numeric, characters)
    assert output.shape == (8,)
    assert torch.isfinite(output).all()


def test_verified_fold_artifact_rejects_plan_or_index_drift(tmp_path: Path) -> None:
    module = load_module()
    result = tmp_path / "result.npz"
    metadata = tmp_path / "metadata.json"
    validation = np.array([1, 3, 5], dtype=np.int64)
    module.write_npz_atomic(
        result,
        validation_indices=validation,
        validation_probability=np.array([0.1, 0.2, 0.3]),
        test_probability=np.array([0.4, 0.5]),
    )
    module.write_json_atomic(
        metadata,
        {
            "plan_sha256": "plan",
            "input_sha256": {"train": "a", "test": "b", "sample": "c"},
            "fold": 0,
        },
    )
    assert module._fold_result_valid(
        metadata,
        result,
        plan_sha256="plan",
        input_sha256={"train": "a", "test": "b", "sample": "c"},
        fold=0,
        expected_validation=validation,
        expected_test_rows=2,
    )
    assert not module._fold_result_valid(
        metadata,
        result,
        plan_sha256="different",
        input_sha256={"train": "a", "test": "b", "sample": "c"},
        fold=0,
        expected_validation=validation,
        expected_test_rows=2,
    )


def test_run_contract_is_stable_and_forbids_side_effects() -> None:
    module = load_module()
    source_hash = module.sha256_file(MODULE_PATH)
    contract = {
        "schema": "evomind.mlebench.may2022_compact_run_contract.v1",
        "runner_sha256": source_hash,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
    }
    assert contract["runner_sha256"] == source_hash
    assert contract["private_labels_used"] is False
    assert contract["official_grader_executed"] is False
    assert contract["kaggle_submission_executed"] is False
    assert contract["process_signals_sent"] == 0
