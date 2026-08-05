"""CPU contract tests for the versioned Spooky DeBERTa OOF runner."""

from __future__ import annotations

import json

import numpy as np

from scripts import run_spooky_deberta_oof_v2 as runner


def test_safe_unicode_ids_load_without_pickle(tmp_path):
    ids = runner.safe_unicode_ids(["id-1", "测试-2", 3])
    assert ids.dtype.kind == "U"
    path = tmp_path / "ids.npz"
    np.savez_compressed(path, ids=ids)
    with np.load(path, allow_pickle=False) as archive:
        assert archive["ids"].tolist() == ["id-1", "测试-2", "3"]


def test_source_contract_rejects_drift(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    plan = {
        "source": {
            "runner": {
                "path": str(source),
                "sha256": runner.sha256_file(source),
            }
        }
    }
    report = runner.verify_source_contract(plan)
    assert report["runner"]["passed"] is True
    source.write_text("VALUE = 2\n", encoding="utf-8")
    try:
        runner.verify_source_contract(plan)
    except ValueError as exc:
        assert "diverges" in str(exc)
    else:
        raise AssertionError("source drift was accepted")


def test_sparse_style_fold_is_fold_local_finite_and_normalized():
    rng = np.random.default_rng(42)
    labels = np.tile(np.arange(3), 30)
    author_tokens = np.asarray(["raven chamber poe", "eldritch arkham horror", "victor windsor life"])
    texts = [
        f"{author_tokens[label]} common token line {index % 7}."
        for index, label in enumerate(labels)
    ]
    style = rng.normal(size=(len(texts), 28)).astype(np.float32)
    fit = np.arange(60)
    valid = np.arange(60, 75)
    test = np.arange(75, 90)
    valid_probability, test_probability, contract = runner.fit_sparse_style_fold(
        [texts[index] for index in fit],
        style[fit],
        labels[fit],
        [texts[index] for index in valid],
        style[valid],
        [texts[index] for index in test],
        style[test],
        word_features=300,
        char_features=500,
        c_value=1.0,
        max_iter=200,
        random_state=42,
    )
    assert valid_probability.shape == (15, 3)
    assert test_probability.shape == (15, 3)
    assert np.all(np.isfinite(valid_probability))
    assert np.allclose(valid_probability.sum(axis=1), 1.0)
    assert np.allclose(test_probability.sum(axis=1), 1.0)
    assert contract["fit_scope"] == "outer_fit_rows_only"
    assert contract["style"]["scaler_fit_scope"] == "outer_fit_rows_only"
    assert contract["private_labels_used"] is False


def test_build_meta_inputs_preserves_fold_specific_base_predictions():
    rng = np.random.default_rng(7)
    rows = 50
    test_rows = 9
    folds = 5
    component_oof = {
        "transformer": runner.normalize_probability(rng.uniform(0.1, 1.0, size=(rows, 3))),
        "sparse": runner.normalize_probability(rng.uniform(0.1, 1.0, size=(rows, 3))),
    }
    component_test = {
        name: np.stack(
            [runner.normalize_probability(rng.uniform(0.1, 1.0, size=(test_rows, 3))) for _ in range(folds)]
        )
        for name in component_oof
    }
    train_style = rng.normal(size=(rows, 28))
    test_style = rng.normal(size=(test_rows, 28))
    oof, test_by_fold, contract = runner.build_meta_inputs(
        component_oof,
        component_test,
        train_style,
        test_style,
    )
    assert oof.shape == (rows, 34)
    assert test_by_fold.shape == (folds, test_rows, 34)
    assert contract["blocks"][-1] == {
        "name": "style",
        "kind": "dense",
        "start": 6,
        "stop": 34,
    }
    assert np.array_equal(test_by_fold[0, :, 6:], test_by_fold[4, :, 6:])


def test_source_contract_descriptor_is_json_serializable(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("pass\n", encoding="utf-8")
    report = runner.verify_source_contract(
        {"source": {"runner": {"path": str(source), "sha256": runner.sha256_file(source)}}}
    )
    json.dumps(report)
