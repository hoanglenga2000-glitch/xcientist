from __future__ import annotations

import json

import pandas as pd
import pytest

from research_os.independent_holdout import evaluate_independent_holdout


def _fixture(tmp_path, *, mismatch: bool = False, threshold: bool = True):
    ids = [0, 1, 2, 3, 4, 5]
    pd.DataFrame({"row_id": ids, "is_fraud": [0.01, 0.2, 0.8, 0.9, 0.4, 0.7]}).to_csv(
        tmp_path / "submission.csv", index=False
    )
    if mismatch:
        ids = [0, 2, 1, 3, 4, 5]
    pd.DataFrame({"row_id": ids, "is_fraud": [0, 0, 1, 1, 0, 1]}).to_csv(
        tmp_path / "labels.csv", index=False
    )
    metrics = {"cv_score": 0.8}
    if threshold:
        metrics["oof_decision_threshold"] = 0.6
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return tmp_path / "submission.csv", tmp_path / "labels.csv", tmp_path / "metrics.json"


def test_independent_review_uses_oof_threshold_and_writes_evidence(tmp_path):
    submission, labels, metrics = _fixture(tmp_path)
    output = tmp_path / "review.json"
    review = evaluate_independent_holdout(submission, labels, metrics, output_path=output)

    assert review["status"] == "passed"
    assert review["threshold_provenance"]["value"] == pytest.approx(0.6)
    assert review["threshold_provenance"]["source"] == "oof_decision_threshold"
    assert review["metrics"]["pr_auc"] == pytest.approx(1.0)
    assert review["claim_audit"]["official_submission_performed"] is False
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "passed"


def test_independent_review_uses_fixed_threshold_when_oof_value_missing(tmp_path):
    submission, labels, metrics = _fixture(tmp_path, threshold=False)
    review = evaluate_independent_holdout(submission, labels, metrics)
    assert review["status"] == "passed"
    assert review["threshold_provenance"]["source"] == "fixed_default_0.5"
    assert review["threshold_provenance"]["holdout_optimization_performed"] is False


def test_independent_review_rejects_row_order_mismatch(tmp_path):
    submission, labels, metrics = _fixture(tmp_path, mismatch=True)
    review = evaluate_independent_holdout(submission, labels, metrics)
    assert review["status"] == "rejected"
    assert review["checks"]["id_order_matches"] is False
    assert review["metrics"]["pr_auc"] is None
