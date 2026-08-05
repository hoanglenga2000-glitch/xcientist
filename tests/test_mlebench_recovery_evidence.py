from __future__ import annotations

import json

from scripts import build_mlebench_candidate_gates as candidate_gates
from scripts import generate_mlebench_recovery_deep_review as deep_review
from scripts import generate_mlebench_unscored_deep_review as unscored_review
from scripts import validate_leaf_linear_recovery_remote as leaf_linear
from scripts import validate_leaf_recovery_remote as leaf_grid
from scripts import validate_pizza_full_recovery_remote as pizza_full

RECOVERY_IDS = [
    "aerial-cactus-identification",
    "denoising-dirty-documents",
    "dogs-vs-cats-redux-kernels-edition",
    "leaf-classification",
    "new-york-city-taxi-fare-prediction",
    "random-acts-of-pizza",
    "ranzcr-clip-catheter-line-classification",
    "siim-isic-melanoma-classification",
    "spooky-author-identification",
    "tabular-playground-series-may-2022",
]

UNSCORED_IDS = sorted(unscored_review.RUNNER_BY_COMPETITION)


def test_deep_review_prompt_contains_all_live_recovery_runners_and_registries():
    progress = {
        "results": [
            {"competition_id": competition_id, "any_medal": False}
            for competition_id in RECOVERY_IDS
        ],
        "scored_competitions": 11,
        "lite_total_competitions": 22,
        "any_medal_count": 2,
        "medals_required_to_strictly_exceed_top": 18,
    }
    ledger = {
        "truth_boundary": {"rule": "internal_candidate != official_medal"},
        "unscored_internal_gate_pass_waiting_official_grader": [],
        "scored_non_medal_internal_recovery_candidates": [
            {"competition_id": "random-acts-of-pizza"}
        ],
        "not_promoted": [],
        "audit_only_no_recovery_gain": [],
    }
    prompt, allowed = deep_review.build_prompt(progress, {}, ledger)
    payload = json.loads(prompt)
    inventory = payload["implementation_inventory"]
    sources = payload["source_files"]
    assert inventory["implemented_medal_recovery_count"] == 10
    assert inventory["missing_medal_recovery_implementations"] == []
    assert set(allowed) == set(RECOVERY_IDS)
    assert "run_aerial_cactus_convnext" in sources["recovery_runner_definitions"]
    assert "run_spooky_nbsvm_oof" in sources["recovery_runner_definitions"]
    assert "run_ranzcr" in sources["vision_runtime_definitions"]
    assert "multilabel_stratified_group_kfold" in sources["vision_runtime_definitions"]
    assert "fold_values = np.asarray(folds)" in sources["legacy_runner_definitions"]
    assert "RUNNERS" in sources["production_runner_registry"]
    assert "RUNNERS" in sources["recovery_registry"]


def test_unscored_deep_review_prompt_is_current_source_grounded():
    progress = {
        "remaining_competition_ids": UNSCORED_IDS,
        "scored_competitions": 14,
        "lite_total_competitions": 22,
        "any_medal_count": 6,
        "medals_required_to_strictly_exceed_top": 18,
    }
    prompt, allowed, live_source = unscored_review.build_prompt(
        progress,
        {"planner": {"model": "gpt-5.6-sol"}},
        {"status": "waiting_for_gpu_idle", "release_deployed": False},
        {"sha256": "a" * 64, "passed": True},
        {"weight_cache": {"status": "VISION_BACKBONE_WEIGHTS_READY"}},
    )
    payload = json.loads(prompt)
    assert set(allowed) == set(UNSCORED_IDS)
    assert payload["allowed_competition_ids"] == UNSCORED_IDS
    assert live_source["file_sha256"] == unscored_review.sha256_file(
        unscored_review.PROJECT_ROOT / "scripts" / "mlebench_wave2_adapters.py"
    )
    source = live_source["source"]
    assert "require_pretrained=True" in source
    assert "cross_fit_ordinal_thresholds" in source
    assert 'group_values=train["PatientID"]' in source
    assert "make_multilabel_stratified_folds" in source
    assert "per_fold_calibrate_then_probability_average" in source
    assert payload["current_execution_evidence"]["pinned_bundle_sha256"] == "a" * 64


def test_unscored_deep_review_rejects_incomplete_task_coverage():
    rows = [
        {"competition_id": competition_id}
        for competition_id in UNSCORED_IDS[:-1]
    ]
    review = {
        "audit_summary": {},
        "per_competition": rows,
        "implementation_order": UNSCORED_IDS,
        "immediate_code_changes": [],
        "resource_schedule": [],
        "global_stop_rules": [],
    }
    import pytest

    with pytest.raises(ValueError, match="unscored eight"):
        unscored_review.validate_review(review, set(UNSCORED_IDS))


def test_remote_recovery_validation_sources_compile():
    sources = [
        leaf_grid._remote_source(leaf_grid.build_parser().parse_args([])),
        leaf_linear._remote_source(leaf_linear.build_parser().parse_args([])),
        pizza_full._remote_source(pizza_full.build_parser().parse_args([])),
    ]
    for index, source in enumerate(sources):
        compile(source, f"<remote-recovery-{index}>", "exec")
        assert "EVOMIND_RESULT=" in source
        assert "prepared/public" in source


def test_candidate_ledger_does_not_queue_already_scored_or_medaled_tasks():
    payload = candidate_gates.build_payload(candidate_gates.PROJECT_ROOT)
    progress = json.loads(
        (candidate_gates.PROJECT_ROOT / "workspace/mlebench_progress/lite11_current.json")
        .read_text(encoding="utf-8")
    )
    official = {row["competition_id"]: row for row in progress["results"]}

    pending = payload["unscored_internal_gate_pass_waiting_official_grader"]
    assert all(candidate["competition_id"] not in official for candidate in pending)

    recoveries = payload["scored_non_medal_internal_recovery_candidates"]
    assert all(not official[candidate["competition_id"]]["any_medal"] for candidate in recoveries)

    confirmed = payload["official_medal_recoveries"]
    assert all(official[candidate["competition_id"]]["any_medal"] for candidate in confirmed)
