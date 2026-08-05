#!/usr/bin/env python3
"""Build the evidence-hashed MLE-Bench medal candidate ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "workspace" / "mlebench_plans" / "medal_candidate_gates_current.json"
)
RELATIVE_INPUTS = (
    "workspace/mlebench_progress/lite11_current.json",
    "workspace/mlebench_plans/text_normalization_public_oof_validation_current.json",
    "workspace/mlebench_plans/text_normalization_russian_public_oof_validation_current.json",
    "workspace/mlebench_plans/nomad_geometry_public_oof_validation_current.json",
    "workspace/mlebench_plans/spooky_nbsvm_public_oof_validation_current.json",
    "workspace/mlebench_plans/pizza_nbsvm_public_oof_validation_current.json",
    "workspace/mlebench_plans/pizza_full_public_oof_validation_current.json",
    "workspace/mlebench_plans/leaf_linear_public_oof_validation_current.json",
    "workspace/mlebench_plans/aerial_exact_duplicate_audit_current.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs(root: Path) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for relative in RELATIVE_INPUTS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Candidate-ledger evidence is missing: {relative}")
        values[relative] = {
            "data": json.loads(path.read_text(encoding="utf-8-sig")),
            "sha256": sha256_file(path),
        }
    return values


def build_payload(root: Path) -> dict[str, Any]:
    inputs = load_inputs(root)

    def data(relative: str) -> dict[str, Any]:
        return inputs[relative]["data"]

    def evidence(relative: str) -> dict[str, str]:
        return {"path": relative, "sha256": inputs[relative]["sha256"]}

    progress_path = RELATIVE_INPUTS[0]
    english_path, russian_path, nomad_path = RELATIVE_INPUTS[1:4]
    spooky_path, pizza_text_path, pizza_full_path = RELATIVE_INPUTS[4:7]
    leaf_path, aerial_path = RELATIVE_INPUTS[7:9]
    progress = data(progress_path)
    english, russian, nomad = data(english_path), data(russian_path), data(nomad_path)
    spooky, pizza_text, pizza_full = (
        data(spooky_path),
        data(pizza_text_path),
        data(pizza_full_path),
    )
    leaf, aerial = data(leaf_path), data(aerial_path)
    official = {row["competition_id"]: row for row in progress["results"]}
    spooky_public_oof = float(
        spooky["best"].get(
            "cross_fitted_multicomponent_log_loss",
            spooky["best"].get("cross_fitted_blend_log_loss"),
        )
    )

    if not (english["threshold_passed"] and english["target_passed"]):
        raise RuntimeError("English normalization no longer passes its internal gates")
    if not (
        russian["bronze_passed"]
        and russian["grouped_target_passed"]
        and russian["source_stress_target_passed"]
    ):
        raise RuntimeError("Russian normalization no longer passes its internal gates")
    if not (nomad["bronze_oriented_gate_passed"] and nomad["target_passed"]):
        raise RuntimeError("NOMAD no longer passes its internal gates")
    if not (pizza_full["bronze_oriented_gate_passed"] and pizza_full["target_passed"]):
        raise RuntimeError("Pizza full recovery no longer passes its internal gates")
    if spooky["bronze_oriented_gate_passed"] or leaf["bronze_oriented_gate_passed"]:
        raise RuntimeError("A non-promoted recovery now passes and must be explicitly promoted")
    if aerial["exact_label_transfer_available"]:
        raise RuntimeError("Aerial duplicate-transfer conclusion changed")

    pending_candidates = [
        {
            "competition_id": english["dataset"],
            "metric": "accuracy",
            "direction": "maximize",
            "public_grouped_oof": english["inferred_class_overall_accuracy"],
            "bronze_oriented_threshold": english["bronze_threshold"],
            "internal_target": english["target_accuracy"],
            "official_status": "not_yet_scored",
            "evidence": evidence(english_path),
        },
        {
            "competition_id": russian["dataset"],
            "metric": "accuracy",
            "direction": "maximize",
            "public_grouped_combined_oof": russian["combined_correct"] / russian["rows"],
            "source_ordered_stress_combined_oof": (
                russian["source_combined_correct"] / russian["source_stress_rows"]
            ),
            "bronze_oriented_threshold": russian["bronze_threshold"],
            "official_status": "not_yet_scored",
            "evidence": evidence(russian_path),
        },
        {
            "competition_id": nomad["competition_id"],
            "metric": "mean_columnwise_rmsle",
            "direction": "minimize",
            "public_oof": nomad["mean_columnwise_rmsle"],
            "bronze_oriented_threshold": nomad["bronze_threshold"],
            "internal_target": nomad["target_threshold"],
            "cross_fitted_meta_validation": True,
            "official_status": "not_yet_scored",
            "evidence": evidence(nomad_path),
        },
    ]
    unscored_candidates = [
        candidate
        for candidate in pending_candidates
        if candidate["competition_id"] not in official
    ]
    officially_scored_internal_candidates = [
        {
            "competition_id": candidate["competition_id"],
            "mle_private_grader_score": official[candidate["competition_id"]][
                "mle_private_grader_score"
            ],
            "any_medal": official[candidate["competition_id"]]["any_medal"],
            "gold_medal": official[candidate["competition_id"]]["gold_medal"],
            "silver_medal": official[candidate["competition_id"]]["silver_medal"],
            "bronze_medal": official[candidate["competition_id"]]["bronze_medal"],
            "official_status": "officially_scored",
            "internal_gate_evidence": candidate["evidence"],
            "official_progress_evidence": evidence(progress_path),
        }
        for candidate in pending_candidates
        if candidate["competition_id"] in official
    ]

    pizza_id = pizza_full["competition_id"]
    pizza_official = official.get(pizza_id)
    pizza_recovery_candidate = {
        "competition_id": pizza_id,
        "metric": "roc_auc",
        "direction": "maximize",
        "public_cross_fitted_oof_by_seed": pizza_full["cross_fitted_blend_scores"],
        "bronze_oriented_threshold": pizza_full["bronze_threshold"],
        "internal_target": pizza_full["target_threshold"],
        "official_score_before_recovery": (
            pizza_official["mle_private_grader_score"] if pizza_official else None
        ),
        "official_medal_before_recovery": (
            pizza_official["any_medal"] if pizza_official else None
        ),
        "cross_fitted_meta_validation": True,
        "official_status": "regrade_required_after_upgraded_runner",
        "evidence": evidence(pizza_full_path),
        "supersedes_text_only_evidence": evidence(pizza_text_path),
    }
    scored_non_medal_recovery_candidates = (
        [pizza_recovery_candidate]
        if pizza_official is not None and not pizza_official["any_medal"]
        else []
    )
    official_medal_recoveries = (
        [{
            **pizza_recovery_candidate,
            "mle_private_grader_score": pizza_official["mle_private_grader_score"],
            "gold_medal": pizza_official["gold_medal"],
            "silver_medal": pizza_official["silver_medal"],
            "bronze_medal": pizza_official["bronze_medal"],
            "official_status": "official_medal_confirmed",
        }]
        if pizza_official is not None and pizza_official["any_medal"]
        else []
    )

    return {
        "schema": "evomind.mlebench.medal_candidate_gates.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/build_mlebench_candidate_gates.py",
        "truth_boundary": {
            "internal_candidate_is_official_medal": False,
            "rule": "internal_candidate != official_medal",
            "medal_authority": "official MLE-Bench private grader",
            "kaggle_auto_submission": False,
            "private_score_used_as_training_signal": False,
        },
        "official_progress": {
            "scored": progress["scored_competitions"],
            "total": progress["lite_total_competitions"],
            "official_medals": progress["any_medal_count"],
            "official_gold": progress["gold_medal_count"],
            "medals_required_to_strictly_exceed_public_top": progress[
                "medals_required_to_strictly_exceed_top"
            ],
            "source": evidence(progress_path),
        },
        "unscored_internal_gate_pass_waiting_official_grader": unscored_candidates,
        "officially_scored_internal_candidates": officially_scored_internal_candidates,
        "scored_non_medal_internal_recovery_candidates": scored_non_medal_recovery_candidates,
        "official_medal_recoveries": official_medal_recoveries,
        "not_promoted": [
            {
                "competition_id": spooky["competition_id"],
                "metric": "multiclass_log_loss",
                "direction": "minimize",
                "public_oof": spooky_public_oof,
                "bronze_oriented_threshold": spooky["bronze_threshold"],
                "reason": "cross-fitted public OOF remains above bronze-oriented threshold",
                "channels": [
                    "word_nbsvm",
                    "character_word_boundary_nbsvm",
                    "character_raw_nbsvm",
                    "stylometry",
                ],
                "stop_rule_triggered": bool(spooky["stop_rule_triggered"]),
                "evidence": evidence(spooky_path),
            },
            {
                "competition_id": leaf["competition_id"],
                "metric": "multiclass_log_loss",
                "direction": "minimize",
                "public_confirmation_oof_by_seed": leaf["confirmation_scores"],
                "bronze_oriented_threshold": leaf["bronze_threshold"],
                "reason": "numeric-only calibrated recovery remains above bronze-oriented threshold",
                "required_next_channel": "pretrained_image_features_or_image_model",
                "evidence": evidence(leaf_path),
            },
        ],
        "audit_only_no_recovery_gain": [
            {
                "competition_id": aerial["competition_id"],
                "strategy": "exact_duplicate_label_transfer",
                "available": aerial["exact_label_transfer_available"],
                "train_duplicate_hashes": aerial["train_duplicate_hashes"],
                "train_test_exact_matches": aerial["train_test_exact_match_count"],
                "conclusion": "exact duplicate transfer unavailable",
                "evidence": evidence(aerial_path),
            }
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()
    payload = build_payload(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "sha256": sha256_file(args.output),
        "unscored_candidates": len(payload["unscored_internal_gate_pass_waiting_official_grader"]),
        "recovery_candidates": len(payload["scored_non_medal_internal_recovery_candidates"]),
        "not_promoted": len(payload["not_promoted"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
