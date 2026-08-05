from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "generate_dog_gpt56_audit",
    ROOT / "scripts" / "generate_dog_gpt56_audit.py",
)
assert SPEC and SPEC.loader
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)


def valid_plan() -> dict:
    source_sha = audit_module.sha256_bytes(audit_module.SOURCE_PATH.read_bytes())
    return {
        "schema": "evomind.mlebench.dog_breed_imagenet_head_recovery_plan.v1",
        "competition_id": audit_module.DOG_ID,
        "promotion_threshold": 0.04,
        "active_parent": {
            "run_id": "local4060_test",
            "adapter_sha256_at_start": "old-source",
        },
        "frozen_teacher_evidence": {
            "rows": 9199,
            "classes": 120,
            "oof_like_train_log_loss": 0.18795286176572076,
            "top1_accuracy": 0.9616262637243178,
            "top5_accuracy": 0.9988042178497663,
            "private_labels_used": False,
        },
        "material_revision": {
            "adapter_sha256": source_sha,
            "pinned_weight_sha256": "pinned-weight",
            "changes": [
                "initialize the 120-class task head from ImageNet classifier rows",
                "use differential learning rates",
            ],
        },
        "launch_policy": "launch only after the parent exits",
    }


def valid_probe() -> dict:
    return {
        "schema": "evomind.dog_breed.imagenet_head_runtime_probe.v1",
        "cuda_available": True,
        "runtime_contract_passed": True,
        "private_labels_used": False,
        "pinned_weight_sha256": "pinned-weight",
        "production_classifier_execution": {
            "output_count": 120,
            "weights_copied_exactly": True,
            "bias_copied_exactly": True,
            "full_classifier_logit_max_abs_delta": 0.0,
            "full_classifier_probability_max_abs_delta": 0.0,
        },
    }


def valid_progress() -> dict:
    return {
        "lite_total_competitions": 22,
        "scored_competitions": 20,
        "any_medal_count": 13,
        "medal_deficit": 5,
        "medals_required_to_strictly_exceed_top": 18,
        "remaining_competition_ids": [audit_module.DOG_ID, "ranzcr"],
    }


def active_log() -> dict:
    return {
        "path": "full.log",
        "sha256": "log-sha",
        "epoch_records": 10,
        "latest": {"fold": 1, "epoch": 10, "cv_log_loss": 0.391431},
        "best_observed_epoch": {"fold": 1, "epoch": 10, "cv_log_loss": 0.391431},
        "latest_by_fold": {"1": {"fold": 1, "epoch": 10, "cv_log_loss": 0.391431}},
        "terminal_summary_present": False,
    }


def valid_audit() -> dict:
    return {
        "truth_boundary_acknowledgement": {
            "official_grader_only": True,
            "oof_not_official": True,
            "private_labels_unused": True,
            "dog_medal_not_campaign_completion": True,
        },
        "source_findings": [
            {
                "finding": "The revised head copies exact ImageNet classifier rows.",
                "evidence": ["initialize_classifier_from_imagenet_rows"],
                "confidence": "high",
            }
        ],
        "risk_assessment": {
            "target_feasibility": "medium",
            "largest_generalization_risks": ["overconfident wrong classes"],
            "why_teacher_0_18795_is_not_enough": "It exceeds 0.040 by 0.14795.",
        },
        "bounded_experiments": [
            {
                "id": "imagenet_head_probe",
                "hypothesis": "The exact classifier head starts below the old random head.",
                "exact_source_changes": ["No extra source change before the queued run."],
                "resource_budget": "One RTX 4060 fold checkpoint.",
                "acceptance_oof_log_loss": 0.12,
                "early_checkpoint_rule": "Fold 1 epoch 1 must beat the old 0.639101.",
                "stop_rule": "Stop the experiment after its configured first-fold checkpoint fails.",
                "rollback": "Retain the prior immutable source and artifacts.",
            }
        ],
        "recommended_next_run": {
            "implementation_verdict": "GO",
            "base_revision": "source-sha",
            "cli_arguments": ["--wave2-dog-breed-epochs=10"],
            "expected_oof_log_loss_low": 0.03,
            "expected_oof_log_loss_high": 0.20,
            "promotion_gate_log_loss": 0.04,
            "resource_fit": "RTX 4060 8GB",
            "stop_rules": ["Respect the existing bounded queue."],
        },
        "campaign_impact": {
            "official_medal_if_passed": 1,
            "medals_still_needed_after_dog": 4,
            "next_competitions_after_dog": ["siim-isic-melanoma-classification"],
        },
    }


def test_prompt_is_source_grounded_and_quantifies_gap() -> None:
    prompt = audit_module.build_prompt(valid_plan(), valid_probe(), valid_progress(), active_log())
    payload = json.loads(prompt)
    assert "initialize_classifier_from_imagenet_rows" in payload["current_source_definitions"]
    assert (
        "classifier_linear_1.0x_pretrained_and_classifier_norm_0.1x"
        in payload["current_run_vision_windows"]
    )
    target_math = payload["verified_state"]["target_math"]
    assert target_math["promotion_log_loss"] == 0.04
    assert target_math["absolute_improvement_needed_from_teacher"] == pytest.approx(
        0.14795286176572076
    )
    assert len(prompt) < 90_000


def test_parse_training_log_tracks_latest_and_best(tmp_path: Path) -> None:
    log = tmp_path / "full.log"
    log.write_text(
        "fold=1/5 epoch=1 cv=0.639101 images_per_second=46.7\n"
        "fold=1/5 epoch=2 cv=0.550991 images_per_second=59.3\n"
        "fold=2/5 epoch=1 cv=0.668282 images_per_second=52.5\n",
        encoding="utf-8",
    )
    parsed = audit_module.parse_training_log(log)
    assert parsed["epoch_records"] == 3
    assert parsed["latest"]["fold"] == 2
    assert parsed["latest_by_fold"]["1"]["epoch"] == 2
    assert parsed["best_observed_epoch"]["cv_log_loss"] == pytest.approx(0.550991)


def test_validate_audit_accepts_complete_contract() -> None:
    audit_module.validate_audit(valid_audit())


def test_validate_audit_rejects_truth_boundary_failure() -> None:
    payload = valid_audit()
    payload["truth_boundary_acknowledgement"]["oof_not_official"] = False
    with pytest.raises(ValueError, match="truth boundary"):
        audit_module.validate_audit(payload)


def test_validate_audit_rejects_unbounded_experiment_list() -> None:
    payload = valid_audit()
    payload["bounded_experiments"] *= 6
    with pytest.raises(ValueError, match="one and five"):
        audit_module.validate_audit(payload)


def test_validate_inputs_detects_source_drift() -> None:
    plan = valid_plan()
    plan["material_revision"]["adapter_sha256"] = "stale"
    with pytest.raises(ValueError, match="adapter hash"):
        audit_module.validate_inputs(plan, valid_probe(), valid_progress())


def test_validate_inputs_requires_full_120_class_runtime_execution() -> None:
    probe = valid_probe()
    probe["production_classifier_execution"]["output_count"] = 3
    with pytest.raises(ValueError, match="full 120-class"):
        audit_module.validate_inputs(valid_plan(), probe, valid_progress())


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:65068/v1",
        "http://127.0.0.1:65069/v1",
        "http://example.com:65068/v1",
    ],
)
def test_gateway_is_fail_closed_to_loopback(url: str) -> None:
    with pytest.raises(RuntimeError, match="loopback"):
        audit_module.assert_local_gateway(url)
