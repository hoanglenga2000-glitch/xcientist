"""Tests for the submission format checker (pre-Kaggle gate validation)."""
from __future__ import annotations

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from research_agent_workstation.server.pipelines.submission_checker import check_submission


def _write_csv(path: Path, frame) -> Path:
    frame.to_csv(path, index=False)
    return path


def test_valid_submission(tmp_path: Path):
    sample = pd.DataFrame({"id": [1, 2, 3], "target": [0, 0, 0]})
    sub = pd.DataFrame({"id": [1, 2, 3], "target": [0.1, 0.8, 0.5]})
    result = check_submission(
        _write_csv(tmp_path / "sub.csv", sub),
        _write_csv(tmp_path / "sample.csv", sample),
    )
    assert result["valid"] is True
    assert result["rows_match"] is True
    assert result["columns_match"] is True
    assert result["missing_predictions"] == 0


def test_row_count_mismatch_is_invalid(tmp_path: Path):
    sample = pd.DataFrame({"id": [1, 2, 3], "target": [0, 0, 0]})
    sub = pd.DataFrame({"id": [1, 2], "target": [0.1, 0.8]})
    result = check_submission(
        _write_csv(tmp_path / "sub.csv", sub),
        _write_csv(tmp_path / "sample.csv", sample),
    )
    assert result["rows_match"] is False
    assert result["valid"] is False


def test_column_mismatch_is_invalid(tmp_path: Path):
    sample = pd.DataFrame({"id": [1, 2], "target": [0, 0]})
    sub = pd.DataFrame({"id": [1, 2], "prediction": [0.1, 0.8]})
    result = check_submission(
        _write_csv(tmp_path / "sub.csv", sub),
        _write_csv(tmp_path / "sample.csv", sample),
    )
    assert result["columns_match"] is False
    assert result["valid"] is False


def test_missing_predictions_is_invalid(tmp_path: Path):
    sample = pd.DataFrame({"id": [1, 2, 3], "target": [0, 0, 0]})
    sub = pd.DataFrame({"id": [1, 2, 3], "target": [0.1, None, 0.5]})
    result = check_submission(
        _write_csv(tmp_path / "sub.csv", sub),
        _write_csv(tmp_path / "sample.csv", sample),
    )
    assert result["missing_predictions"] == 1
    assert result["valid"] is False


def test_return_types_are_json_safe(tmp_path: Path):
    import json

    sample = pd.DataFrame({"id": [1, 2], "target": [0, 0]})
    sub = pd.DataFrame({"id": [1, 2], "target": [0.3, 0.7]})
    result = check_submission(
        _write_csv(tmp_path / "sub.csv", sub),
        _write_csv(tmp_path / "sample.csv", sample),
    )
    # All values must be native (not numpy) so downstream JSON writes never crash.
    json.dumps(result)
    assert isinstance(result["valid"], bool)
    assert isinstance(result["missing_predictions"], int)
