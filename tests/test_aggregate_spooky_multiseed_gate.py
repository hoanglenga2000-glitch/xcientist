"""Tests for the Spooky three-seed OOF promotion gate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import aggregate_spooky_multiseed_gate as aggregate


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _probability(truth: np.ndarray, confidence: float) -> np.ndarray:
    result = np.full((len(truth), 3), (1.0 - confidence) / 2.0, dtype=np.float64)
    result[np.arange(len(truth)), truth] = confidence
    return result


def _seed_run(root: Path, seed: int, *, confidence: float = 0.82) -> Path:
    run = root / f"seed_{seed}"
    run.mkdir()
    truth = np.tile(np.arange(3), 30)
    oof = _probability(truth, confidence)
    test = _probability(np.arange(12) % 3, confidence - 0.02)
    train_id = np.asarray([f"train-{index}" for index in range(len(truth))])
    test_id = np.asarray([f"test-{index}" for index in range(len(test))])
    bundle = run / "spooky_transformer_oof_and_test.npz"
    np.savez_compressed(
        bundle,
        truth=truth,
        candidate_oof=oof,
        candidate_test=test,
        train_id=train_id,
        test_id=test_id,
    )
    submission = pd.DataFrame({"id": test_id})
    for index, column in enumerate(aggregate.CLASS_COLUMNS):
        submission[column] = test[:, index]
    submission_path = run / "candidate_submission_withheld.csv"
    submission.to_csv(submission_path, index=False)
    score = aggregate.multiclass_log_loss(truth, oof)
    summary_path = run / "summary.json"
    _write_json(
        summary_path,
        {
            "run_id": f"spooky-seed-{seed}",
            "status": "single_seed_gate_passed_confirmation_pending",
            "candidate_oof_log_loss": score,
            "promotion_gate": {"seed": seed, "single_seed_passed": True},
            "prediction_bundle": {
                "path": str(bundle),
                "sha256": aggregate.sha256_file(bundle),
            },
            "submission_withheld": {
                "path": str(submission_path),
                "sha256": aggregate.sha256_file(submission_path),
            },
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        },
    )
    verification_path = run / "independent_verification.json"
    _write_json(
        verification_path,
        {
            "status": "passed",
            "recomputed_candidate_oof_log_loss": score,
            "contract_checks": {"one": True, "two": True},
            "summary": {
                "path": str(summary_path),
                "sha256": aggregate.sha256_file(summary_path),
            },
            "prediction_bundle": {
                "path": str(bundle),
                "sha256": aggregate.sha256_file(bundle),
            },
            "submission_withheld": {
                "path": str(submission_path),
                "sha256": aggregate.sha256_file(submission_path),
            },
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        },
    )
    return run


def test_complete_verified_three_seed_gate_passes_and_stays_withheld(tmp_path: Path):
    runs = {seed: _seed_run(tmp_path, seed) for seed in (40, 41, 42)}
    output = tmp_path / "aggregate"
    report = aggregate.aggregate_runs(runs, output)
    assert report["status"] == "promotion_gate_passed"
    assert report["promotion_allowed"] is True
    assert report["gate_checks"]["every_seed_independently_verified"] is True
    assert report["gate_checks"]["averaged_oof_test_ids_aligned"] is True
    assert report["official_grader_executed"] is False
    assert report["kaggle_submission_executed"] is False
    assert report["process_signals_sent"] == 0
    assert (output / "spooky_multiseed_gate.json").is_file()
    assert (output / "spooky_multiseed_oof_and_test.npz").is_file()
    withheld = pd.read_csv(output / "candidate_submission_multiseed_withheld.csv")
    assert list(withheld.columns) == ["id", *aggregate.CLASS_COLUMNS]
    assert np.allclose(withheld[list(aggregate.CLASS_COLUMNS)].sum(axis=1), 1.0)


def test_gate_fails_when_one_independent_verification_is_not_passed(tmp_path: Path):
    runs = {seed: _seed_run(tmp_path, seed) for seed in (40, 41, 42)}
    verification_path = runs[41] / "independent_verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    verification["status"] = "failed"
    _write_json(verification_path, verification)
    report = aggregate.aggregate_runs(runs, tmp_path / "aggregate")
    assert report["promotion_allowed"] is False
    assert report["gate_checks"]["every_seed_independently_verified"] is False


def test_gate_rejects_misaligned_test_ids(tmp_path: Path):
    runs = {seed: _seed_run(tmp_path, seed) for seed in (40, 41, 42)}
    run = runs[42]
    bundle = run / "spooky_transformer_oof_and_test.npz"
    with np.load(bundle, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["test_id"] = arrays["test_id"][::-1]
    np.savez_compressed(bundle, **arrays)
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["prediction_bundle"]["sha256"] = aggregate.sha256_file(bundle)
    _write_json(summary_path, summary)
    verification_path = run / "independent_verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    verification["summary"]["sha256"] = aggregate.sha256_file(summary_path)
    verification["prediction_bundle"]["sha256"] = aggregate.sha256_file(bundle)
    _write_json(verification_path, verification)
    report = aggregate.aggregate_runs(runs, tmp_path / "aggregate")
    assert report["promotion_allowed"] is False
    assert report["identity_checks"]["test_id_aligned"] is False
    assert report["gate_checks"]["averaged_oof_test_ids_aligned"] is False
    assert report["artifacts"] == {}


def test_gate_propagates_private_label_boundary_failure(tmp_path: Path):
    runs = {seed: _seed_run(tmp_path, seed) for seed in (40, 41, 42)}
    summary_path = runs[40] / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["private_labels_used"] = True
    _write_json(summary_path, summary)
    verification_path = runs[40] / "independent_verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    verification["summary"]["sha256"] = aggregate.sha256_file(summary_path)
    _write_json(verification_path, verification)
    report = aggregate.aggregate_runs(runs, tmp_path / "aggregate")
    assert report["promotion_allowed"] is False
    assert report["gate_checks"]["private_labels_unused"] is False
