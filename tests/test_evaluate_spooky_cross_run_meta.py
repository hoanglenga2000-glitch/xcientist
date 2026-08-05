"""Contract tests for the frozen cross-run Spooky meta evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import evaluate_spooky_cross_run_meta as evaluator


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _probability(
    rng: np.random.Generator,
    rows: int,
    *,
    truth: np.ndarray | None = None,
    signal: float = 0.0,
) -> np.ndarray:
    values = rng.uniform(0.05, 0.35, size=(rows, 3))
    if truth is not None:
        values[np.arange(rows), truth] += signal
    return values / values.sum(axis=1, keepdims=True)


def _numeric_payloads(
    *, seed: int = 42000, rows: int = 150, test_rows: int = 11
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    truth = np.tile(np.arange(3, dtype=np.int64), rows // 3)
    folds = np.tile(np.arange(5, dtype=np.int16), rows // 5)

    prior_sparse = _probability(rng, rows, truth=truth, signal=0.8)
    prior_byte = _probability(rng, rows, truth=truth, signal=0.6)
    prior_transformer = _probability(rng, rows, truth=truth, signal=1.0)
    prior_candidate = evaluator.validate_probability(
        "prior_candidate",
        0.25 * prior_sparse + 0.20 * prior_byte + 0.55 * prior_transformer,
        rows,
    )
    current_transformer = _probability(rng, rows, truth=truth, signal=1.2)
    current_sparse = _probability(rng, rows, truth=truth, signal=0.75)
    current_candidate = evaluator.validate_probability(
        "current_candidate",
        0.65 * current_transformer + 0.35 * current_sparse,
        rows,
    )

    def foldwise() -> np.ndarray:
        return np.stack([_probability(rng, test_rows) for _ in range(5)])

    prior_sparse_test = foldwise()
    prior_byte_test = foldwise()
    prior_transformer_test = foldwise()
    current_transformer_test = foldwise()
    current_sparse_test = foldwise()
    prior = {
        "truth": truth,
        "fold": folds,
        "sparse_oof": prior_sparse,
        "byte_oof": prior_byte,
        "transformer_oof": prior_transformer,
        "candidate_oof": prior_candidate,
        "sparse_test_by_fold": prior_sparse_test,
        "byte_test_by_fold": prior_byte_test,
        "transformer_test_by_fold": prior_transformer_test,
        "candidate_test": evaluator.validate_probability(
            "prior_candidate_test",
            (
                0.25 * prior_sparse_test
                + 0.20 * prior_byte_test
                + 0.55 * prior_transformer_test
            ).mean(axis=0),
            test_rows,
        ),
    }
    current = {
        "truth": truth.copy(),
        "fold": folds.copy(),
        "transformer_oof": current_transformer,
        "sparse_oof": current_sparse,
        "candidate_oof": current_candidate,
        "transformer_test_by_fold": current_transformer_test,
        "sparse_test_by_fold": current_sparse_test,
        "candidate_test": evaluator.validate_probability(
            "current_candidate_test",
            (0.65 * current_transformer_test + 0.35 * current_sparse_test).mean(
                axis=0
            ),
            test_rows,
        ),
    }
    return prior, current


def _sha_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": evaluator.sha256_file(path)}


def _build_end_to_end_fixture(tmp_path: Path) -> Path:
    prior, current = _numeric_payloads()
    truth = current["truth"]
    rows = len(truth)
    test_rows = current["candidate_test"].shape[0]

    prior_bundle = tmp_path / "prior.npz"
    # The legacy object array proves the evaluator accesses only its numeric allowlist.
    np.savez_compressed(
        prior_bundle,
        **prior,
        legacy_train_id=np.asarray([f"legacy-{i}" for i in range(rows)], dtype=object),
    )
    current_bundle = tmp_path / "current.npz"
    np.savez_compressed(current_bundle, **current)
    prior_summary = tmp_path / "prior_summary.json"
    _write_json(prior_summary, {"status": "single_seed_gate_failed"})
    current_plan = tmp_path / "current_plan.json"
    _write_json(current_plan, {"schema": "synthetic.current.plan.v1"})
    current_summary = tmp_path / "current_summary.json"
    _write_json(
        current_summary,
        {
            "status": "single_seed_gate_failed",
            "run_id": "current-run",
            "plan_sha256": evaluator.sha256_file(current_plan),
            "prediction_bundle": _sha_record(current_bundle),
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "process_signals_sent": 0,
        },
    )

    train = pd.DataFrame(
        {
            "id": [f"train-{index}" for index in range(rows)],
            "text": [f"public text {index}" for index in range(rows)],
            "author": [evaluator.CLASS_COLUMNS[value] for value in truth],
        }
    )
    test = pd.DataFrame(
        {
            "id": [f"test-{index}" for index in range(test_rows)],
            "text": [f"withheld public text {index}" for index in range(test_rows)],
        }
    )
    sample = pd.DataFrame(
        {
            "id": test["id"],
            **{name: np.full(test_rows, 1.0 / 3.0) for name in evaluator.CLASS_COLUMNS},
        }
    )
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    sample_path = tmp_path / "sample_submission.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    sample.to_csv(sample_path, index=False)

    output_dir = tmp_path / "output"
    plan = {
        "schema": evaluator.EXPECTED_SCHEMA,
        "status": "frozen_waiting_current_run",
        "competition_id": "spooky-author-identification",
        "public_data_only": True,
        "source": {
            "evaluator": _sha_record(Path(evaluator.__file__).resolve()),
            "meta_helper": _sha_record(
                evaluator.PROJECT_ROOT / "scripts" / "spooky_leakage_free_meta.py"
            ),
        },
        "prior": {
            "run_id": "prior-run",
            "summary": _sha_record(prior_summary),
            "bundle": _sha_record(prior_bundle),
        },
        "current": {
            "run_id": "current-run",
            "plan": _sha_record(current_plan),
            "summary_path": str(current_summary),
        },
        "inputs": {
            "train": _sha_record(train_path),
            "test": _sha_record(test_path),
            "sample": _sha_record(sample_path),
        },
        "meta": {"c_value": 0.1, "max_iter": 3000, "random_state": 42000},
        "promotion_gate": {"single_seed_log_loss_threshold": 0.292},
        "output": {"run_id": "synthetic-cross-run", "directory": str(output_dir)},
        "boundaries": {
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "process_signals_sent": 0,
        },
    }
    plan_path = tmp_path / "frozen_plan.json"
    _write_json(plan_path, plan)
    return plan_path


def test_synthetic_five_fold_evaluate_is_exact_once_and_human_gated(tmp_path: Path):
    result = evaluator.evaluate(_build_end_to_end_fixture(tmp_path))

    assert result["schema"] == "evomind.spooky.cross_run_meta_result.v1"
    assert result["checks"]["exact_once_oof"] is True
    assert result["checks"]["finite_normalized_oof"] is True
    assert result["checks"]["finite_normalized_test"] is True
    assert result["private_labels_used"] is False
    assert result["official_grader_executed"] is False
    assert result["kaggle_submission_executed"] is False
    assert result["process_signals_sent"] == 0
    assert result["promotion_allowed"] is False
    assert result["human_gate_preserved"] is True
    assert "kaggle_public_score" not in result
    assert "kaggle_private_score" not in result

    with np.load(result["bundle"]["path"], allow_pickle=False) as bundle:
        assert np.all(bundle["candidate_write_counts"] == 1)
        assert np.allclose(bundle["candidate_oof"].sum(axis=1), 1.0)
        assert np.allclose(bundle["candidate_test"].sum(axis=1), 1.0)
        assert bundle["train_id"].dtype.kind == "U"
        assert bundle["test_id"].dtype.kind == "U"


def test_build_channels_rejects_truth_and_fold_drift():
    prior, current = _numeric_payloads(rows=30, test_rows=7)
    prior["truth"] = prior["truth"].copy()
    prior["truth"][0] = (prior["truth"][0] + 1) % 3
    with pytest.raises(ValueError, match="truth arrays"):
        evaluator.build_channels(prior, current)

    prior, current = _numeric_payloads(rows=30, test_rows=7)
    prior["fold"] = prior["fold"].copy()
    prior["fold"][0], prior["fold"][1] = prior["fold"][1], prior["fold"][0]
    with pytest.raises(ValueError, match="fold assignments"):
        evaluator.build_channels(prior, current)


def test_probability_shape_and_normalization_contract():
    values = evaluator.validate_probability(
        "channel", np.asarray([[2.0, 1.0, 1.0], [0.2, 0.3, 0.5]]), 2
    )
    assert values.shape == (2, 3)
    assert np.all(np.isfinite(values))
    assert np.allclose(values.sum(axis=1), 1.0)
    with pytest.raises(ValueError, match="expected"):
        evaluator.validate_probability("bad-shape", np.ones((2, 2)), 2)
    with pytest.raises(ValueError, match="finite"):
        evaluator.validate_probability(
            "bad-finite", np.asarray([[np.nan, 0.5, 0.5]]), 1
        )


def test_numeric_loader_ignores_legacy_object_id_array(tmp_path: Path):
    path = tmp_path / "legacy_bundle.npz"
    truth = np.asarray([0, 1, 2], dtype=np.int64)
    folds = np.asarray([0, 1, 2], dtype=np.int16)
    np.savez_compressed(
        path,
        truth=truth,
        fold=folds,
        legacy_id=np.asarray(["a", "b", "c"], dtype=object),
    )

    loaded = evaluator.load_numeric_bundle(path, ("truth", "fold"))

    assert set(loaded) == {"truth", "fold"}
    assert np.array_equal(loaded["truth"], truth)
    assert np.array_equal(loaded["fold"], folds)
