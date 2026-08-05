"""Fast leakage and deployment contracts for the Spooky neural recovery runner."""

from __future__ import annotations

import numpy as np
import pytest

from scripts import run_spooky_transformer_oof as runner


def _blend_fixture():
    rng = np.random.default_rng(20260727)
    folds = np.repeat(np.arange(5), 30)
    truth = (np.arange(len(folds)) + folds) % 3
    component_oof = {}
    component_test = {}
    for index, name in enumerate(("sparse", "byte", "transformer")):
        logits = rng.normal(0.0, 0.7, size=(len(truth), 3))
        logits[np.arange(len(truth)), truth] += 1.2 + index * 0.3
        probability = np.exp(logits - logits.max(axis=1, keepdims=True))
        component_oof[name] = probability / probability.sum(axis=1, keepdims=True)
        raw_test = rng.uniform(0.01, 1.0, size=(5, 17, 3))
        component_test[name] = raw_test / raw_test.sum(axis=2, keepdims=True)
    return component_oof, component_test, truth, folds


def test_duplicate_groups_never_cross_outer_fold():
    texts = [f"unique text {index}" for index in range(90)]
    labels = np.arange(90) % 3
    texts[31] = texts[1]
    labels[31] = labels[1]
    texts[62] = texts[2]
    labels[62] = labels[2]
    splits, assignment, report = runner.build_duplicate_safe_folds(
        texts,
        labels,
        folds=5,
        seed=42,
    )
    assert len(splits) == 5
    assert np.all(assignment >= 0)
    assert assignment[1] == assignment[31]
    assert assignment[2] == assignment[62]
    assert report["group_isolation"] is True


def test_conflicting_duplicate_labels_are_rejected():
    texts = [f"row {index}" for index in range(30)]
    labels = np.arange(30) % 3
    texts[7] = texts[1]
    labels[7] = (labels[1] + 1) % 3
    with pytest.raises(ValueError, match="conflicting labels"):
        runner.build_duplicate_safe_folds(texts, labels, folds=3, seed=42)


def test_exact_deployment_blend_is_finite_normalized_and_exact_once():
    component_oof, component_test, truth, folds = _blend_fixture()
    oof, test, counts, contract = runner.cross_fit_deployment_blend(
        component_oof,
        component_test,
        truth,
        folds,
        steps=5,
        temperatures=(0.9, 1.0, 1.1),
    )
    assert oof.shape == (len(truth), 3)
    assert test.shape == (17, 3)
    assert np.all(counts == 1)
    assert np.isfinite(oof).all() and np.isfinite(test).all()
    assert np.allclose(oof.sum(axis=1), 1.0)
    assert np.allclose(test.sum(axis=1), 1.0)
    assert contract["exact_once_oof"] is True
    assert contract["test_aggregation"] == "apply_fold_specific_transform_then_average"


def test_held_fold_truth_does_not_change_its_meta_model_or_prediction():
    component_oof, component_test, truth, folds = _blend_fixture()
    baseline = runner.cross_fit_deployment_blend(
        component_oof,
        component_test,
        truth,
        folds,
        steps=4,
        temperatures=(0.9, 1.0),
    )
    changed_truth = truth.copy()
    changed_truth[folds == 2] = (changed_truth[folds == 2] + 1) % 3
    changed = runner.cross_fit_deployment_blend(
        component_oof,
        component_test,
        changed_truth,
        folds,
        steps=4,
        temperatures=(0.9, 1.0),
    )
    assert np.array_equal(baseline[0][folds == 2], changed[0][folds == 2])
    baseline_record = [item for item in baseline[3]["records"] if item["score_fold"] == 2]
    changed_record = [item for item in changed[3]["records"] if item["score_fold"] == 2]
    assert baseline_record == changed_record


def test_frozen_plan_rejects_runtime_drift():
    args = runner.build_parser().parse_args(
        [
            "--public-dir",
            ".",
            "--output-root",
            ".",
            "--run-id",
            "fixture",
            "--frozen-plan",
            "plan.json",
            "--hf-cache",
            ".",
            "--learning-rate",
            "3e-5",
        ]
    )
    plan = {
        "model": {"repo_id": runner.DEFAULT_MODEL, "revision": runner.DEFAULT_REVISION},
        "training": {
            "seed": 42,
            "folds": 5,
            "epochs_per_fold": 3,
            "max_length": 192,
            "train_batch_size": 8,
            "eval_batch_size": 64,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "max_grad_norm": 1.0,
            "num_workers": 2,
            "gradient_checkpointing": True,
            "mixed_precision": "bf16",
        },
        "byte_channel": {
            "max_length": 768,
            "embedding_dim": 48,
            "channels": 128,
            "dropout": 0.2,
            "epochs_per_fold": 6,
            "train_batch_size": 128,
            "eval_batch_size": 256,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "label_smoothing": 0.05,
        },
        "sparse_channel": {
            "word_max_features": 100000,
            "char_max_features": 180000,
            "c_value": 4.0,
        },
        "ensemble": {"weight_grid_steps": 20},
    }
    with pytest.raises(ValueError, match="learning_rate"):
        runner.validate_frozen_plan_arguments(args, plan, diagnostic=False)
