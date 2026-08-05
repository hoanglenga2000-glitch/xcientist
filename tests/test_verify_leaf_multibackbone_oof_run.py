from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import verify_leaf_multibackbone_oof_run as verifier


def test_apply_blend_is_normalized_and_temperature_sensitive():
    first = np.array([[0.9, 0.1], [0.2, 0.8]])
    second = np.array([[0.7, 0.3], [0.4, 0.6]])
    neutral = verifier.apply_blend([first, second], [0.5, 0.5], 1.0)
    sharpened = verifier.apply_blend([first, second], [0.5, 0.5], 0.5)

    assert np.allclose(neutral.sum(axis=1), 1.0)
    assert np.allclose(sharpened.sum(axis=1), 1.0)
    assert sharpened[0, 0] > neutral[0, 0]


def test_reconstruct_crossfit_uses_each_fold_record_once():
    folds = np.array([0, 0, 1, 1], dtype=np.int16)
    components = {
        "a": np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]]),
        "b": np.full((4, 2), 0.5),
    }
    records = [
        {"fold": 0, "weights": [1.0, 0.0], "temperature": 1.0},
        {"fold": 1, "weights": [0.0, 1.0], "temperature": 1.0},
    ]

    rebuilt = verifier.reconstruct_crossfit(components, folds, records)

    assert np.allclose(rebuilt[:2], components["a"][:2])
    assert np.allclose(rebuilt[2:], components["b"][2:])


def test_verify_artifact_manifest_rejects_hash_drift(tmp_path: Path):
    artifact = tmp_path / "candidate.csv"
    artifact.write_text("id,A\n1,1.0\n", encoding="utf-8")
    manifest = tmp_path / "artifact_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "evomind.leaf.multibackbone_artifacts.v1",
                "artifact_count": 1,
                "artifacts": [
                    {
                        "path": artifact.name,
                        "bytes": artifact.stat().st_size,
                        "sha256": verifier.sha256_file(artifact),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert verifier.verify_artifact_manifest(tmp_path, manifest)["passed"] is True

    artifact.write_text("changed", encoding="utf-8")
    try:
        verifier.verify_artifact_manifest(tmp_path, manifest)
    except RuntimeError as exc:
        assert "differs" in str(exc)
    else:
        raise AssertionError("hash drift was accepted")


def test_failure_report_contract_preserves_human_gate_fields(tmp_path: Path):
    output = tmp_path / "report.json"
    code = verifier.main(
        [
            "--run-dir",
            str(tmp_path / "missing-run"),
            "--plan",
            str(tmp_path / "missing-plan.json"),
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert code == 1
    assert report["status"] == "verification_failed"
    assert report["process_signals_sent"] == 0
    assert report["official_grader_executed"] is False
    assert report["kaggle_submission_executed"] is False
