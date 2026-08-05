"""Contract tests for the nested Spooky cross-run XGBoost evaluator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from scripts import evaluate_spooky_cross_run_xgb as evaluator


def _load_xgb_classifier():
    package_root = (
        evaluator.PROJECT_ROOT
        / "workspace"
        / "local_gpu"
        / "runtime_packages"
        / "xgboost_3_0_2"
    )
    sys.path.insert(0, str(package_root))
    from xgboost import XGBClassifier

    return XGBClassifier


def _probability(
    rng: np.random.Generator, truth: np.ndarray, *, signal: float
) -> np.ndarray:
    values = rng.uniform(0.1, 0.4, size=(len(truth), 3))
    values[np.arange(len(truth)), truth] += signal
    return values / values.sum(axis=1, keepdims=True)


def test_feature_assembly_has_frozen_probability_transforms_and_style():
    rng = np.random.default_rng(42)
    rows = 10
    truth = np.arange(rows) % 3
    channels = {
        "a": _probability(rng, truth, signal=0.4),
        "b": _probability(rng, truth, signal=0.6),
    }
    style = rng.normal(size=(rows, 28))

    features, contract = evaluator.assemble_xgb_features(
        channels, style, channel_order=("a", "b")
    )

    assert features.shape == (rows, 44)
    assert np.isfinite(features).all()
    assert contract["columns"] == 44
    assert contract["channel_order"] == ["a", "b"]
    assert contract["blocks"][-1]["name"] == "style"


def test_nested_cross_fit_scores_every_outer_row_once_and_is_normalized():
    rng = np.random.default_rng(9042)
    rows = 150
    test_rows = 13
    truth = np.tile(np.arange(3, dtype=np.int64), rows // 3)
    folds = np.tile(np.arange(5, dtype=np.int16), rows // 5)
    features = rng.normal(size=(rows, 12))
    features[:, :3] += np.eye(3)[truth] * 1.5
    test_features = rng.normal(size=(5, test_rows, 12))
    configurations = [
        {
            "name": "conservative",
            "params": {
                "n_estimators": 12,
                "max_depth": 2,
                "learning_rate": 0.08,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "min_child_weight": 2.0,
                "reg_lambda": 4.0,
                "reg_alpha": 0.1,
                "n_jobs": 1,
            },
        },
        {
            "name": "external_faithful",
            "params": {
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1.0,
                "reg_lambda": 3.0,
                "reg_alpha": 0.1,
                "n_jobs": 1,
            },
        },
    ]

    oof, test, counts, contract = evaluator.cross_fit_nested_xgb(
        features,
        test_features,
        truth,
        folds,
        configurations=configurations,
        temperatures=(0.9, 1.0, 1.1),
        inner_validation_fold_offset=1,
        random_state=42000,
        xgb_classifier=_load_xgb_classifier(),
    )

    assert oof.shape == (rows, 3)
    assert test.shape == (test_rows, 3)
    assert np.all(counts == 1)
    assert np.allclose(oof.sum(axis=1), 1.0)
    assert np.allclose(test.sum(axis=1), 1.0)
    assert contract["exact_once_oof"] is True
    assert len(contract["records"]) == 5
    assert all(record["fit_score_disjoint"] for record in contract["records"])


def test_held_outer_labels_do_not_affect_that_fold_predictions():
    rng = np.random.default_rng(7)
    rows = 150
    truth = np.tile(np.arange(3, dtype=np.int64), rows // 3)
    folds = np.tile(np.arange(5, dtype=np.int16), rows // 5)
    features = rng.normal(size=(rows, 8))
    test_features = rng.normal(size=(5, 4, 8))
    configurations = [
        {
            "name": "fixed",
            "params": {
                "n_estimators": 6,
                "max_depth": 2,
                "learning_rate": 0.1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "min_child_weight": 1.0,
                "reg_lambda": 3.0,
                "reg_alpha": 0.0,
                "n_jobs": 1,
            },
        }
    ]
    kwargs = dict(
        configurations=configurations,
        temperatures=(1.0,),
        inner_validation_fold_offset=1,
        random_state=42000,
        xgb_classifier=_load_xgb_classifier(),
    )
    baseline = evaluator.cross_fit_nested_xgb(
        features, test_features, truth, folds, **kwargs
    )
    changed_truth = truth.copy()
    changed_truth[folds == 2] = (changed_truth[folds == 2] + 1) % 3
    changed = evaluator.cross_fit_nested_xgb(
        features, test_features, changed_truth, folds, **kwargs
    )

    assert np.array_equal(baseline[0][folds == 2], changed[0][folds == 2])


def test_runtime_manifest_is_isolated_and_hash_verified():
    manifest_path = (
        evaluator.PROJECT_ROOT
        / "workspace"
        / "mlebench_plans"
        / "xgboost_3_0_2_isolated_runtime_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    package_root = Path(manifest["package_root"])
    digest, count, total = evaluator.tree_sha256(package_root)

    assert manifest["version"] == "3.0.2"
    assert manifest["global_environment_modified"] is False
    assert digest == manifest["tree_sha256"]
    assert count == manifest["file_count"]
    assert total == manifest["total_bytes"]


def test_runtime_tree_hash_ignores_interpreter_bytecode_cache(tmp_path: Path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "module.cpython-312.pyc").write_bytes(b"frozen runtime cache")
    baseline = evaluator.tree_sha256(package)
    (cache / "module.cpython-311.pyc").write_bytes(b"foreign interpreter cache")

    assert evaluator.tree_sha256(package) == baseline
