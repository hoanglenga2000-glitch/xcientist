"""Fast contracts for the local Jigsaw transformer OOF runner."""

from __future__ import annotations

import numpy as np
import pytest

from scripts import run_jigsaw_transformer_oof as runner


def _fixture():
    rng = np.random.default_rng(42)
    folds = np.repeat(np.arange(5), 20)
    truth = np.column_stack(
        [((np.arange(len(folds)) + label) % (3 + label % 2) == 0).astype(np.int8) for label in range(6)]
    )
    sparse = np.clip(0.2 + 0.6 * truth + rng.normal(0, 0.25, truth.shape), 0.001, 0.999)
    transformer = np.clip(0.15 + 0.7 * truth + rng.normal(0, 0.20, truth.shape), 0.001, 0.999)
    sparse_test = rng.uniform(0.01, 0.99, size=(17, 6))
    transformer_test = rng.uniform(0.01, 0.99, size=(5, 17, 6))
    return sparse, transformer, sparse_test, transformer_test, truth, folds


def test_fold_clean_transformer_blend_is_exact_once_and_finite():
    sparse, transformer, sparse_test, transformer_test, truth, folds = _fixture()
    oof, test, counts, contract = runner.cross_fit_sparse_transformer_rank_blend(
        sparse,
        transformer,
        sparse_test,
        transformer_test,
        truth,
        folds,
    )

    assert oof.shape == truth.shape
    assert test.shape == sparse_test.shape
    assert np.isfinite(oof).all()
    assert np.isfinite(test).all()
    assert np.all(counts == 1)
    assert contract["exact_once_oof"] is True
    assert contract["private_labels_used"] is False
    assert len(contract["records"]) == 5 * 6


def test_held_fold_labels_do_not_change_its_weight_or_prediction():
    sparse, transformer, sparse_test, transformer_test, truth, folds = _fixture()
    baseline = runner.cross_fit_sparse_transformer_rank_blend(
        sparse,
        transformer,
        sparse_test,
        transformer_test,
        truth,
        folds,
    )
    changed_truth = truth.copy()
    changed_truth[folds == 2] = 1 - changed_truth[folds == 2]
    changed = runner.cross_fit_sparse_transformer_rank_blend(
        sparse,
        transformer,
        sparse_test,
        transformer_test,
        changed_truth,
        folds,
    )

    assert np.array_equal(baseline[0][folds == 2], changed[0][folds == 2])
    baseline_records = [r for r in baseline[3]["records"] if r["score_fold"] == 2]
    changed_records = [r for r in changed[3]["records"] if r["score_fold"] == 2]
    assert baseline_records == changed_records


def test_sparse_bundle_rejects_missing_arrays(tmp_path):
    path = tmp_path / "broken.npz"
    np.savez_compressed(path, truth=np.zeros((10, 6), dtype=np.int8))

    with pytest.raises(ValueError, match="missing arrays"):
        runner.load_sparse_bundle(path, row_count=10, test_count=4)


def test_frozen_plan_rejects_runtime_drift():
    args = runner.build_parser().parse_args(
        [
            "--public-dir",
            ".",
            "--sparse-bundle",
            "sparse.npz",
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
            "epochs_per_fold": 2,
            "max_length": 192,
            "train_batch_size": 8,
            "eval_batch_size": 32,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "max_grad_norm": 1.0,
            "num_workers": 2,
            "gradient_checkpointing": True,
            "mixed_precision": "bf16",
        },
    }

    with pytest.raises(ValueError, match="learning_rate"):
        runner.validate_frozen_plan_arguments(args, plan, diagnostic=False)


def test_confirmation_seed_contract_reuses_folds_and_changes_model_seed():
    args = runner.build_parser().parse_args(
        [
            "--public-dir",
            ".",
            "--sparse-bundle",
            "sparse.npz",
            "--output-root",
            ".",
            "--run-id",
            "fixture",
            "--frozen-plan",
            "plan.json",
            "--hf-cache",
            ".",
            "--fold-seed",
            "42",
            "--model-seed",
            "40",
        ]
    )
    plan = {
        "training": {"seed": 40, "fold_seed": 42, "model_seed": 40},
    }

    assert runner.resolve_seed_contract(args, plan) == (42, 40)
    assert args.fold_seed == 42
    assert args.model_seed == 40


def test_confirmation_seed_contract_rejects_fold_or_model_drift():
    plan = {"training": {"seed": 40, "fold_seed": 42, "model_seed": 40}}
    args = runner.build_parser().parse_args(
        [
            "--public-dir",
            ".",
            "--sparse-bundle",
            "sparse.npz",
            "--output-root",
            ".",
            "--run-id",
            "fixture",
            "--frozen-plan",
            "plan.json",
            "--hf-cache",
            ".",
            "--fold-seed",
            "41",
            "--model-seed",
            "40",
        ]
    )
    with pytest.raises(ValueError, match="fold_seed"):
        runner.resolve_seed_contract(args, plan)


def test_safe_unicode_ids_load_without_pickle(tmp_path):
    ids = runner.safe_unicode_ids(["row-1", "测试-2", 3])
    path = tmp_path / "ids.npz"
    np.savez_compressed(path, ids=ids)
    with np.load(path, allow_pickle=False) as archive:
        assert archive["ids"].dtype.kind == "U"
        assert archive["ids"].tolist() == ["row-1", "测试-2", "3"]


def test_confirmation_plan_can_bind_shared_token_cache(tmp_path):
    shared = tmp_path / "shared-token-cache"
    args = runner.build_parser().parse_args(
        [
            "--public-dir",
            ".",
            "--sparse-bundle",
            "sparse.npz",
            "--output-root",
            ".",
            "--run-id",
            "fixture",
            "--frozen-plan",
            "plan.json",
            "--hf-cache",
            ".",
            "--token-cache-dir",
            str(shared),
        ]
    )
    plan = {"execution": {"token_cache_dir": str(shared)}}
    assert runner.resolve_token_cache_dir(args, plan, tmp_path / "run") == shared.resolve()


def test_unbound_shared_token_cache_is_rejected(tmp_path):
    args = runner.build_parser().parse_args(
        [
            "--public-dir",
            ".",
            "--sparse-bundle",
            "sparse.npz",
            "--output-root",
            ".",
            "--run-id",
            "fixture",
            "--frozen-plan",
            "plan.json",
            "--hf-cache",
            ".",
            "--token-cache-dir",
            str(tmp_path / "shared"),
        ]
    )
    with pytest.raises(ValueError, match="not bound"):
        runner.resolve_token_cache_dir(args, {}, tmp_path / "run")
