from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import resume_siim_campaign_with_harmonization_fix as campaign_recovery
from scripts import run_siim_harmonization_boundary_recovery as recovery

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_boundary_recovery_clips_only_valid_sigmoid_endpoints() -> None:
    oof = np.asarray([0.0, 0.2, 0.8, 1.0], dtype=np.float64)
    test = np.asarray([[0.0, 0.5, 1.0], [0.1, 0.5, 0.9]], dtype=np.float64)
    clipped_oof, clipped_test, evidence = recovery.sanitize_probabilities(oof, test)
    assert evidence["exact_zero_count"] == 2
    assert evidence["exact_one_count"] == 2
    assert evidence["clipped_value_count"] == 4
    assert evidence["nonfinite_count"] == 0
    assert evidence["below_zero_count"] == 0
    assert evidence["above_one_count"] == 0
    assert clipped_oof.tolist() == pytest.approx([1e-6, 0.2, 0.8, 1.0 - 1e-6])
    assert clipped_test[0].tolist() == pytest.approx([1e-6, 0.5, 1.0 - 1e-6])
    assert evidence["oof_input_sha256"] != evidence["oof_output_sha256"]
    assert evidence["rank_policy"] == "monotonic_epsilon_clip_preserves_strict_order_and_endpoint_ties"


@pytest.mark.parametrize("invalid", [np.nan, -1e-9, 1.0 + 1e-9])
def test_boundary_recovery_rejects_nonfinite_or_out_of_range(invalid: float) -> None:
    oof = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    test = np.asarray([[0.1, 0.5], [0.2, 0.7]], dtype=np.float64)
    oof[0] = invalid
    with pytest.raises(
        recovery.HarmonizationBoundaryRecoveryError,
        match="finite and closed-unit",
    ):
        recovery.sanitize_probabilities(oof, test)


def test_campaign_recovery_only_substitutes_formal_runner_and_never_signals() -> None:
    path = PROJECT_ROOT / "scripts" / "resume_siim_campaign_with_harmonization_fix.py"
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    assert 'Path(adjusted[1]).name == "run_mlebench_lite_full.py"' in source
    assert "adjusted[1] = str(runner)" in source
    assert "campaign.TERMINAL_CANDIDATE_STATES.add(PROMOTION_GATE_FAILED)" in source
    assert "campaign.KNOWN_TERMINAL_RESULT_STATES.add(PROMOTION_GATE_FAILED)" in source
    assert '"completed_below_promotion_gate"' in source
    assert 'record["trainer_exit_code"]' in source
    assert 'record["normalized_campaign_exit_code"]' in source
    assert '"signals_sent": 0' in source
    assert '"other_processes_modified": False' in source
    assert "os.kill" not in source
    assert "terminate(" not in source
    assert "kill(" not in source


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_below_gate_result_is_reused_only_with_complete_scientific_evidence(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "formal_s43"
    task_root = run_root / campaign_recovery.COMPETITION
    attempt = task_root / "attempts" / "attempt_001"
    attempt.mkdir(parents=True)
    bundle = tmp_path / "bundle"
    adapter = bundle / "scripts" / "mlebench_medal_recovery_adapters.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text("immutable adapter", encoding="utf-8")
    adapter_sha256 = _sha256(adapter)
    wave2 = bundle / "scripts" / "mlebench_wave2_adapters.py"
    wave2.write_text("immutable wave2", encoding="utf-8")

    oof = attempt / "siim_oof_predictions.csv"
    oof.write_text(
        "image_name,patient_id,leakage_group,target,fold,blended_probability\n"
        + "\n".join(
            f"image_{fold},patient_{fold},group_{fold},{fold % 2},{fold},{0.1 + fold * 0.1}" for fold in range(5)
        )
        + "\n",
        encoding="utf-8",
    )
    submission = attempt / "submission.csv"
    submission.write_text(
        "image_name,target\nimage_a,0.1\nimage_b,0.2\nimage_c,0.3\n",
        encoding="utf-8",
    )
    image_manifest = attempt / "siim_image_content_manifest.csv"
    image_manifest.write_text("image_name,sha256\nimage_0,fixture\n", encoding="utf-8")
    expected_manifest_sha256 = _sha256(image_manifest)
    contract = {
        "schema": campaign_recovery.FORMAL_RESUME_SCHEMA,
        "competition_id": campaign_recovery.COMPETITION,
        "model_seed": 43,
        "outer_folds": 5,
        "inner_folds": 3,
        "leakage_group_policy": campaign_recovery.FORMAL_GROUP_POLICY,
        "perceptual_edge_policy": campaign_recovery.FORMAL_PERCEPTUAL_POLICY,
        "image_content_manifest_sha256": expected_manifest_sha256,
        "adapter_source_sha256": adapter_sha256,
        "wave2_source_sha256": _sha256(wave2),
        "model": {"workers": 8, "fast_kernel_mode": True},
    }
    contract["contract_sha256"] = campaign_recovery.contract_sha256(contract)
    _write_json(
        task_root / "attempts" / "siim_resume_state" / "resume_contract.json",
        contract,
    )

    artifact_roles = {
        "siim_duplicate_connected_groups.json": "duplicate_connected_groups",
        "siim_fold_ensemble.npz": "fold_ensemble_arrays",
        "siim_image_content_manifest.csv": "image_content_manifest",
        "siim_nested_patient_folds.json": "nested_leakage_group_folds",
        "siim_oof_predictions.csv": "outer_oof_predictions",
        "siim_resume_contract.json": "formal_resume_contract",
        "siim_test_components.csv": "test_component_predictions",
        "siim_training_history.json": "training_history",
        "submission.csv": "candidate_submission",
        "siim_preprocessing_ablation_gate.json": "preprocessing_ablation_gate",
    }
    for fold in range(1, 6):
        artifact_roles[f"siim_fusion_fold_{fold:02d}.pt"] = "vision_channel_checkpoint"
        artifact_roles[f"siim_metadata_fold_{fold:02d}.cbm"] = "metadata_catboost_model"
    for name in artifact_roles:
        path = attempt / name
        if not path.exists():
            path.write_bytes(f"fixture:{name}".encode())
    entries = [
        {
            "path": name,
            "role": role,
            "bytes": (attempt / name).stat().st_size,
            "sha256": _sha256(attempt / name),
        }
        for name, role in artifact_roles.items()
    ]
    artifact_manifest = attempt / "siim_artifact_manifest.json"
    _write_json(
        artifact_manifest,
        {
            "schema": "evomind.siim_artifact_manifest.v1",
            "source_sha256": adapter_sha256,
            "artifact_count": len(entries),
            "artifacts": entries,
        },
    )
    _write_json(
        attempt / "submission_validation.json",
        {
            "schema": "evomind.mlebench_submission_validation.v1",
            "valid": True,
            "row_count": 3,
            "expected_row_count": 3,
            "columns_match": True,
            "rows_match": True,
            "id_multiset_match": True,
            "id_order_match": True,
            "id_set_match": True,
            "errors": [],
            "duplicate_id_count": 0,
            "missing_prediction_count": 0,
            "nonfinite_prediction_count": 0,
            "nonnumeric_prediction_count": 0,
            "probability_out_of_range_count": 0,
            "row_sum_violation_count": 0,
            "sample_duplicate_id_count": 0,
        },
    )
    hashes = {
        "oof_input_sha256": "1" * 64,
        "oof_output_sha256": "2" * 64,
        "test_input_sha256": "3" * 64,
        "test_output_sha256": "4" * 64,
    }
    _write_json(
        task_root / "siim_harmonization_boundary_hotfix.json",
        {
            "formal_seed": 43,
            "status": "failed",
            "exit_code": 3,
            "epsilon": 1e-6,
            "adapter_sha256": adapter_sha256,
            "signals_sent": 0,
            "other_processes_modified": False,
            "calls": [
                {
                    "channel": channel,
                    "exact_zero_count": 1,
                    "exact_one_count": 0,
                    "clipped_value_count": 1,
                    "nonfinite_count": 0,
                    "below_zero_count": 0,
                    "above_one_count": 0,
                    **hashes,
                }
                for channel in campaign_recovery.EXPECTED_HARMONIZATION_CHANNELS
            ],
        },
    )
    result = {
        "status": "promotion_gate_failed",
        "candidate_only": True,
        "valid_submission": True,
        "official_grader_executed": False,
        "cv_score": 0.91,
        "attempt": 1,
        "attempt_dir": str(attempt),
        "folds": [{"fold": fold, "valid_rows": 1, "inner_fold_count": 3} for fold in range(5)],
        "promotion_gate": {
            "passed": False,
            "internal_score": 0.91,
            "threshold": 0.942,
        },
        "submission_sha256": _sha256(submission),
        "artifact_manifest_sha256": _sha256(artifact_manifest),
        "competition_id": campaign_recovery.COMPETITION,
        "budget": {
            "seed": 43,
            "folds": 5,
            "image_content_manifest_sha256": expected_manifest_sha256,
            "resume_contract_sha256": contract["contract_sha256"],
            "duplicate_group_report": {
                "schema": campaign_recovery.FORMAL_GROUP_SCHEMA,
                "leakage_group_policy": campaign_recovery.FORMAL_GROUP_POLICY,
                "perceptual_edge_policy": campaign_recovery.FORMAL_PERCEPTUAL_POLICY,
                "perceptual_edges_applied_to_groups": 0,
            },
        },
    }

    checks = campaign_recovery.promotion_gate_failed_artifact_checks(
        run_root,
        result,
        expected_seed=43,
        expected_manifest_sha256=expected_manifest_sha256,
        bundle_root=bundle,
        expected_train_rows=5,
        expected_test_rows=3,
    )

    assert checks and all(checks.values()), [name for name, passed in checks.items() if not passed]
    legacy_checks = campaign_recovery.legacy_campaign_formal_result_checks(
        run_root,
        result,
        expected_seed=43,
        expected_manifest_sha256=expected_manifest_sha256,
        bundle_root=bundle,
        terminal_states={"promotion_gate_failed"},
    )
    assert legacy_checks and all(legacy_checks.values()), [name for name, passed in legacy_checks.items() if not passed]
    result["official_grader_executed"] = True
    tampered = campaign_recovery.promotion_gate_failed_artifact_checks(
        run_root,
        result,
        expected_seed=43,
        expected_manifest_sha256=expected_manifest_sha256,
        bundle_root=bundle,
        expected_train_rows=5,
        expected_test_rows=3,
    )
    assert tampered["below_gate_official_grader_excluded"] is False
