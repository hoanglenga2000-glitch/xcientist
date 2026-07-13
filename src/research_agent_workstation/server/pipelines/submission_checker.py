from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def check_submission(submission_path: Path, sample_submission_path: Path) -> dict[str, Any]:
    submission = pd.read_csv(submission_path)
    sample = pd.read_csv(sample_submission_path)
    prediction_columns = sample.columns[1:].tolist()
    columns_match = submission.columns.tolist() == sample.columns.tolist()
    # Only inspect prediction columns that actually exist in the submission so a
    # column mismatch is reported as invalid rather than raising a KeyError.
    available_prediction_columns = [c for c in prediction_columns if c in submission.columns]
    result = {
        "submission_path": str(submission_path),
        "rows_match": len(submission) == len(sample),
        "columns_match": columns_match,
        "missing_predictions": (
            int(submission[available_prediction_columns].isna().sum().sum())
            if available_prediction_columns
            else 0
        ),
        "positive_predictions": True,
    }
    if available_prediction_columns:
        numeric = submission[available_prediction_columns].apply(pd.to_numeric, errors="coerce")
        result["positive_predictions"] = bool((numeric > 0).all().all())
    result["valid"] = bool(
        result["rows_match"] and result["columns_match"] and result["missing_predictions"] == 0
    )
    return result

