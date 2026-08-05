from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts import verify_siim_final_candidate as verifier


def _nested_payload() -> dict:
    folds = []
    all_groups = [{"a", "b"}, {"c", "d"}]
    for outer, valid in enumerate(all_groups):
        fit = set().union(*(groups for index, groups in enumerate(all_groups) if index != outer))
        inner_folds = []
        ordered = sorted(fit)
        for inner, inner_valid in enumerate(({ordered[0]}, {ordered[1]})):
            inner_folds.append({
                "inner_fold": inner,
                "roles": {
                    "inner_fit": {"leakage_groups": sorted(fit - inner_valid)},
                    "inner_valid": {"leakage_groups": sorted(inner_valid)},
                },
            })
        folds.append({
            "outer_fold": outer,
            "all_inner_folds_aggregated": True,
            "inner_folds": inner_folds,
            "roles": {
                "outer_fit": {"leakage_groups": sorted(fit)},
                "outer_valid": {"leakage_groups": sorted(valid)},
                "refit": {"leakage_groups": sorted(fit)},
            },
        })
    return {
        "schema": "evomind.siim_nested_patient_folds.v2",
        "outer_fold_count": 2,
        "requested_inner_fold_count": 2,
        "all_inner_folds_aggregated": True,
        "outer_validation_role": "final_oof_only",
        "fixed_epoch_outer_refit": True,
        "channel_specific_epoch_selection": True,
        "fixed_iteration_metadata_outer_refit": True,
        "folds": folds,
    }


def test_nested_isolation_accepts_disjoint_fresh_refit_contract():
    report = verifier.verify_nested_isolation(_nested_payload(), 2, 2)
    assert report["passed"] is True


def test_nested_isolation_rejects_outer_group_leakage():
    payload = _nested_payload()
    payload["folds"][0]["roles"]["outer_valid"]["leakage_groups"].append("c")
    report = verifier.verify_nested_isolation(payload, 2, 2)
    assert report["passed"] is False
    assert report["checks"]["outer_0_fit_valid_disjoint"] is False


def test_artifact_manifest_rehashes_every_file(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"verified")
    manifest_path = tmp_path / "siim_artifact_manifest.json"
    manifest = {
        "schema": "evomind.siim_artifact_manifest.v1",
        "hash_algorithm": "sha256",
        "artifact_count": 1,
        "artifacts": [{
            "path": artifact.name,
            "role": "test",
            "bytes": artifact.stat().st_size,
            "sha256": verifier.sha256_file(artifact),
        }],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert verifier.verify_artifact_manifest(tmp_path, manifest_path)["passed"] is True
    artifact.write_bytes(b"changed")
    assert verifier.verify_artifact_manifest(tmp_path, manifest_path)["passed"] is False


def test_oof_auc_and_group_isolation_are_recomputed(tmp_path):
    frame = pd.DataFrame({
        "image_name": ["a", "b", "c", "d"],
        "patient_id": ["p1", "p1", "p2", "p3"],
        "leakage_group": ["g1", "g1", "g2", "g3"],
        "target": [0, 0, 1, 1],
        "fold": [0, 0, 1, 1],
        "blended_probability": [0.1, 0.2, 0.8, 0.9],
    })
    path = tmp_path / "siim_oof_predictions.csv"
    frame.to_csv(path, index=False)
    result = {"cv_score": 1.0, "budget": {"folds": 2}}
    report = verifier.verify_oof(tmp_path, result, expected_rows=4)
    assert report["passed"] is True
    frame.loc[1, "fold"] = 1
    frame.to_csv(path, index=False)
    assert verifier.verify_oof(tmp_path, result, expected_rows=4)["passed"] is False


def test_submission_is_rebuilt_from_fold_ensemble_arrays(tmp_path):
    ids = np.array(["x", "y"])
    probabilities = np.array([0.2, 0.8])
    sample = pd.DataFrame({"image_name": ids, "target": [0.0, 0.0]})
    sample_path = tmp_path / "sample_submission.csv"
    sample.to_csv(sample_path, index=False)
    pd.DataFrame({"image_name": ids, "target": probabilities}).to_csv(
        tmp_path / "submission.csv", index=False
    )
    pd.DataFrame({"image_name": ids, "blended_probability": probabilities}).to_csv(
        tmp_path / "siim_test_components.csv", index=False
    )
    fold = np.array([[0.1, 0.7], [0.3, 0.9]])
    np.savez_compressed(
        tmp_path / "siim_fold_ensemble.npz",
        test_id=ids,
        test_probability=probabilities,
        pure_image_test_by_fold=fold,
        lesion_focus_test_by_fold=fold,
        image_metadata_fusion_test_by_fold=fold,
        metadata_catboost_test_by_fold=fold,
    )
    report = verifier.verify_submission_rebuild(tmp_path, sample_path, expected_rows=2)
    assert report["passed"] is True
