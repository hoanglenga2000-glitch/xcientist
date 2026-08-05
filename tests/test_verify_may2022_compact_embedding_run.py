from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "verify_may2022_compact_embedding_run.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "verify_may2022_compact_embedding_run", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_fold_record(fold: int) -> dict:
    return {
        "fold": fold,
        "selected_epoch": 2,
        "selection_history": [
            {"epoch": 1, "inner_validation_auc": 0.8},
            {"epoch": 2, "inner_validation_auc": 0.9},
        ],
        "refit_history": [{"epoch": 1}, {"epoch": 2}],
        "outer_validation_used_for_epoch_selection": False,
        "outer_refit_fixed_budget": True,
    }


def test_fold_record_enforces_nested_selection_and_fixed_refit() -> None:
    module = load_module()
    module.validate_fold_record(valid_fold_record(0), 0)
    leaking = valid_fold_record(0)
    leaking["outer_validation_used_for_epoch_selection"] = True
    with pytest.raises(RuntimeError, match="outer validation"):
        module.validate_fold_record(leaking, 0)
    wrong_budget = valid_fold_record(0)
    wrong_budget["refit_history"] = [{"epoch": 1}]
    with pytest.raises(RuntimeError, match="epoch count"):
        module.validate_fold_record(wrong_budget, 0)


def test_assemble_fold_predictions_requires_exact_manifest_indices(tmp_path: Path) -> None:
    module = load_module()
    assignment = np.array([0, 1, 0, 1], dtype=np.int16)
    for fold in range(2):
        fold_dir = tmp_path / f"fold_{fold:02d}"
        fold_dir.mkdir()
        validation = np.flatnonzero(assignment == fold)
        result = fold_dir / "result.npz"
        np.savez_compressed(
            result,
            validation_indices=validation,
            validation_probability=np.array([0.1 + fold, 0.2 + fold]),
            test_probability=np.array([0.3 + fold, 0.4 + fold]),
        )
        checkpoint = fold_dir / "model.pt"
        checkpoint.write_bytes(f"checkpoint-{fold}".encode())
        record = valid_fold_record(fold)
        metadata = {
            "result_sha256": module.sha256_file(result),
            "checkpoint_sha256": module.sha256_file(checkpoint),
            "private_labels_used": False,
            "official_grader_executed": False,
            "process_signals_sent": 0,
            "fold_record": record,
        }
        (fold_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    oof, test, records = module.assemble_fold_predictions(
        tmp_path, assignment, fold_count=2, test_rows=2
    )
    assert np.isfinite(oof).all()
    assert test.tolist() == pytest.approx([0.8, 0.9])
    assert [record["fold"] for record in records] == [0, 1]
    with np.load(tmp_path / "fold_00" / "result.npz") as result:
        valid = result["validation_probability"]
        test_probability = result["test_probability"]
    np.savez_compressed(
        tmp_path / "fold_00" / "result.npz",
        validation_indices=np.array([2, 0]),
        validation_probability=valid,
        test_probability=test_probability,
    )
    metadata_path = tmp_path / "fold_00" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["result_sha256"] = module.sha256_file(tmp_path / "fold_00" / "result.npz")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="indices"):
        module.assemble_fold_predictions(tmp_path, assignment, fold_count=2, test_rows=2)


def test_gate_requires_aggregate_and_every_fold() -> None:
    module = load_module()
    target = np.array([0, 1, 0, 1, 0, 1], dtype=np.int8)
    folds = np.array([0, 0, 1, 1, 2, 2], dtype=np.int16)
    perfect = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    gate = module.evaluate_gate(
        target,
        folds,
        perfect,
        aggregate_threshold=0.99,
        every_fold_threshold=0.99,
    )
    assert gate["passed"] is True
    failing = module.evaluate_gate(
        target,
        folds,
        np.array([0.1, 0.9, 0.8, 0.2, 0.3, 0.7]),
        aggregate_threshold=0.5,
        every_fold_threshold=0.99,
    )
    assert failing["aggregate_passed"] is True
    assert failing["all_folds_passed"] is False
    assert failing["passed"] is False
