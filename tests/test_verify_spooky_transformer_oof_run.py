"""Independent reconstruction tests for the Spooky neural OOF verifier."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts import verify_spooky_transformer_oof_run_v2 as verifier


def _fixture():
    rng = np.random.default_rng(19761007)
    folds = np.repeat(np.arange(5), 24)
    truth = (np.arange(len(folds)) + folds) % 3
    oof = {}
    test = {}
    for index, name in enumerate(("sparse", "byte", "transformer")):
        logits = rng.normal(0.0, 0.8, size=(len(truth), 3))
        logits[np.arange(len(truth)), truth] += 1.0 + 0.25 * index
        probability = np.exp(logits - logits.max(axis=1, keepdims=True))
        oof[name] = probability / probability.sum(axis=1, keepdims=True)
        raw = rng.uniform(0.01, 1.0, size=(5, 13, 3))
        test[name] = raw / raw.sum(axis=2, keepdims=True)
    return oof, test, truth, folds


def test_independent_reconstruction_is_exact_once_and_normalized():
    oof, test, truth, folds = _fixture()
    prediction, test_prediction, counts, records = verifier.reconstruct_exact_deployment(
        oof,
        test,
        truth,
        folds,
        steps=5,
        temperatures=(0.9, 1.0, 1.1),
    )
    assert prediction.shape == (len(truth), 3)
    assert test_prediction.shape == (13, 3)
    assert np.all(counts == 1)
    assert np.allclose(prediction.sum(axis=1), 1.0)
    assert np.allclose(test_prediction.sum(axis=1), 1.0)
    assert len(records) == 5


def test_held_fold_truth_does_not_change_its_independent_transform():
    oof, test, truth, folds = _fixture()
    baseline = verifier.reconstruct_exact_deployment(
        oof,
        test,
        truth,
        folds,
        steps=4,
        temperatures=(0.9, 1.0),
    )
    changed_truth = truth.copy()
    changed_truth[folds == 3] = (changed_truth[folds == 3] + 1) % 3
    changed = verifier.reconstruct_exact_deployment(
        oof,
        test,
        changed_truth,
        folds,
        steps=4,
        temperatures=(0.9, 1.0),
    )
    assert np.array_equal(baseline[0][folds == 3], changed[0][folds == 3])
    baseline_record = [item for item in baseline[3] if item["score_fold"] == 3]
    changed_record = [item for item in changed[3] if item["score_fold"] == 3]
    assert baseline_record == changed_record


def test_duplicate_verifier_detects_cross_fold_groups():
    texts = ["alpha", "beta", "alpha", "gamma"]
    labels = np.array([0, 1, 0, 2])
    report = verifier.verify_duplicate_fold_isolation(texts, labels, np.array([0, 1, 2, 3]))
    assert report["duplicate_rows_beyond_first"] == 1
    assert report["cross_fold_groups"] == 1
    assert report["passed"] is False


def test_legacy_object_id_bundle_writes_complete_failed_report(tmp_path):
    run_dir = tmp_path / "run"
    public_dir = tmp_path / "public"
    run_dir.mkdir()
    public_dir.mkdir()
    oof, test_by_fold, truth, folds = _fixture()
    candidate_oof, candidate_test, candidate_counts, _ = verifier.reconstruct_exact_deployment(
        oof,
        test_by_fold,
        truth,
        folds,
        steps=5,
        temperatures=(0.9, 1.0, 1.1),
    )
    train_ids = [f"train-{index}" for index in range(len(truth))]
    test_ids = [f"test-{index}" for index in range(candidate_test.shape[0])]
    train = pd.DataFrame(
        {
            "id": train_ids,
            "text": [f"unique public row {index}" for index in range(len(truth))],
            "author": [verifier.CLASS_COLUMNS[int(value)] for value in truth],
        }
    )
    test = pd.DataFrame(
        {
            "id": test_ids,
            "text": [f"unique public test row {index}" for index in range(len(test_ids))],
        }
    )
    sample = pd.DataFrame({"id": test_ids, **{name: 0.0 for name in verifier.CLASS_COLUMNS}})
    train_path = public_dir / "train.csv"
    test_path = public_dir / "test.csv"
    sample_path = public_dir / "sample_submission.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    sample.to_csv(sample_path, index=False)

    submission = sample.copy()
    submission.loc[:, list(verifier.CLASS_COLUMNS)] = candidate_test
    submission.to_csv(run_dir / "candidate_submission_withheld.csv", index=False)
    np.savez_compressed(
        run_dir / "spooky_transformer_oof_and_test.npz",
        truth=truth,
        fold=folds,
        sparse_oof=oof["sparse"],
        byte_oof=oof["byte"],
        transformer_oof=oof["transformer"],
        sparse_test_by_fold=test_by_fold["sparse"],
        byte_test_by_fold=test_by_fold["byte"],
        transformer_test_by_fold=test_by_fold["transformer"],
        candidate_oof=candidate_oof,
        candidate_test=candidate_test,
        component_write_counts=np.ones((len(truth), 3), dtype=np.uint8),
        candidate_write_counts=candidate_counts,
        train_id=np.asarray(train_ids, dtype=object),
        test_id=np.asarray(test_ids, dtype=object),
    )
    plan = {
        "inputs": {
            "public_train_sha256": verifier.sha256_file(train_path),
            "public_test_sha256": verifier.sha256_file(test_path),
            "public_sample_submission_sha256": verifier.sha256_file(sample_path),
        },
        "ensemble": {"weight_grid_steps": 5, "temperatures": [0.9, 1.0, 1.1]},
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    summary = {
        "run_id": "legacy-object-id-fixture",
        "status": "single_seed_gate_failed",
        "plan_sha256": verifier.sha256_file(plan_path),
        "candidate_oof_log_loss": verifier.independent_log_loss(truth, candidate_oof),
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    output = run_dir / "independent_verification.json"

    exit_code = verifier.main(
        [
            "--run-dir",
            str(run_dir),
            "--public-dir",
            str(public_dir),
            "--plan",
            str(plan_path),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 3
    assert report["status"] == "failed"
    assert report["bundle_load"]["allow_pickle"] is False
    assert report["bundle_load"]["bundle_id_arrays_safe"] is False
    assert report["bundle_load"]["id_arrays"]["train_id"]["status"] == (
        "legacy_object_dtype_rejected"
    )
    assert report["bundle_load"]["id_arrays"]["test_id"]["status"] == (
        "legacy_object_dtype_rejected"
    )
    assert report["contract_checks"]["bundle_id_arrays_safe"] is False
    assert report["identity_checks"]["train_id"] is False
    assert report["identity_checks"]["test_id"] is False
    assert all(report["probability_checks"].values())
    assert np.isclose(
        report["recomputed_candidate_oof_log_loss"],
        summary["candidate_oof_log_loss"],
    )
    assert report["official_grader_executed"] is False
    assert report["kaggle_submission_executed"] is False
