from __future__ import annotations

import copy
import json
import tarfile

from scripts import verify_taxi_source_audit_readiness as verify


def test_taxi_source_audit_readiness_covers_all_acceptance_contracts() -> None:
    report = verify.verify()

    assert report["passed"] is True
    assert report["ready_for_candidate_execution"] is True
    assert report["candidate_metric_available"] is False
    assert report["external_seed_results_available"] is False
    assert report["official_grader_executed"] is False
    assert report["kaggle_submission_executed"] is False
    assert set(report["checks"]) == {
        "duplicate_safe_oof",
        "conflicting_targets_quarantined",
        "fold_local_route_statistics",
        "key_and_submission_alignment",
        "geographic_stress_split",
        "full_refit_and_stress_contract",
        "frozen_plan_and_bundle_binding",
        "verified_public_precomputed_cache",
        "verified_route_stat_sidecar",
    }
    assert all(check["passed"] for check in report["checks"].values())
    assert report["checks"]["verified_public_precomputed_cache"]["train_rows"] == 5_000_000
    route = report["checks"]["verified_route_stat_sidecar"]
    assert route["train_rows"] == 5_000_000
    assert route["cache_seed"] == 42
    assert route["folds"] == 3
    assert route["artifact_count"] == 10
    assert route["validation_targets_used"] == 0
    assert route["private_labels_used"] is False
    assert route["gpu_used"] is False
    frozen = report["checks"]["frozen_plan_and_bundle_binding"]
    assert frozen["source_identity_count"] >= 7
    assert frozen["bundle_bound_source_count"] >= 5
    assert frozen["source_identity_mismatches"] == []
    assert frozen["bundle_source_mismatches"] == []


def test_frozen_binding_rejects_non_runner_source_drift() -> None:
    plan = json.loads(verify.PLAN.read_text(encoding="utf-8"))
    with tarfile.open(verify.DEFAULT_BUNDLE, "r:gz") as archive:
        handle = archive.extractfile("bundle_manifest.json")
        assert handle is not None
        manifest = json.loads(handle.read().decode("utf-8"))
    drifted = copy.deepcopy(plan)
    adapter = next(
        record
        for record in drifted["source_identity"]
        if record["relative_path"] == "scripts/mlebench_medal_recovery_adapters.py"
    )
    adapter["sha256"] = "0" * 64

    result = verify.verify_frozen_source_bindings(drifted, manifest)

    assert result["passed"] is False
    assert result["source_identity_mismatches"] == [
        "scripts/mlebench_medal_recovery_adapters.py"
    ]


def test_frozen_binding_rejects_duplicate_or_missing_identity_records() -> None:
    plan = json.loads(verify.PLAN.read_text(encoding="utf-8"))
    duplicate = copy.deepcopy(plan)
    duplicate["source_identity"][-1] = copy.deepcopy(duplicate["source_identity"][0])

    result = verify.verify_frozen_source_bindings(duplicate, {"files": {}})

    assert result["passed"] is False
    assert result["source_identity_mismatches"]
