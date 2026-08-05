from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from scripts import mlebench_campaign_supervisor as supervisor
from scripts.mlebench_campaign_supervisor import (
    CAMPAIGN_TASKS,
    CONFIRMATION_TASKS,
    a800_reserved_competitions,
    apply_primary_recovery_stops,
    build_multiseed_report_payload,
    campaign_reachability,
    completed_competitions,
    data_inventory_evidence,
    integrate_external_a800_completions,
    intentional_promotion_withholds,
    llm_runtime_evidence,
    next_gpu_successor,
    next_task,
    official_results_from_summary,
    performance_evidence_from_summary,
    record_outcome,
    remote_status_finished,
    retry_remote_call,
    seed42_confirmation_eligibility,
    validated_primary_recovery_stops,
)

EXPECTED_UNSCORED = {
    "aptos2019-blindness-detection",
    "dog-breed-identification",
    "ranzcr-clip-catheter-line-classification",
}


EXPECTED_RECOVERY = [
    "jigsaw-toxic-comment-classification-challenge",
    "spooky-author-identification",
    "new-york-city-taxi-fare-prediction",
    "siim-isic-melanoma-classification",
    "tabular-playground-series-may-2022",
    "aerial-cactus-identification",
    "leaf-classification",
]

EXPECTED_WAVE2_GPT56_ORDER = [
    "mlsp-2013-birds",
    "the-icml-2013-whale-challenge-right-whale-redux",
    "plant-pathology-2020-fgvc7",
    "jigsaw-toxic-comment-classification-challenge",
    "aptos2019-blindness-detection",
    "dog-breed-identification",
    "ranzcr-clip-catheter-line-classification",
    "histopathologic-cancer-detection",
]


def test_a800_completion_is_integrated_once_without_losing_active_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watch_dir = tmp_path / "watch"
    collection_root = tmp_path / "collected"
    integration_dir = tmp_path / "integrations"
    state_path = tmp_path / "campaign.json"
    log_path = tmp_path / "campaign.log"
    run_id = "a800_siim_s42"
    competition_id = "siim-isic-melanoma-classification"
    summary_dir = collection_root / run_id
    summary_dir.mkdir(parents=True)
    watch_dir.mkdir(parents=True)
    (watch_dir / f"{run_id}_completion.json").write_text(
        json.dumps(
            {
                "schema": "evomind.mlebench.a800_lane_watch.completion.v1",
                "created_at": "2026-07-26T13:45:37+08:00",
                "run_id": run_id,
                "process": "stopped",
                "summary_status": "passed",
                "official_grade_count": 1,
                "official_results": [{"competition_id": competition_id}],
                "collection": {"passed": True, "file_count": 13},
            }
        ),
        encoding="utf-8",
    )
    (summary_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "results": [
                    {
                        "competition_id": competition_id,
                        "status": "passed",
                        "cv_score": 0.918,
                        "mle_private_grader_score": 0.92165,
                        "metric": "roc_auc",
                        "direction": "maximize",
                        "valid_submission": True,
                        "official_grader_executed": True,
                        "private_grader": {
                            "upstream_report": {"any_medal": False}
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(supervisor, "A800_WATCH_DIR", watch_dir)
    monkeypatch.setattr(supervisor, "A800_COLLECTION_ROOT", collection_root)
    monkeypatch.setattr(supervisor, "A800_INTEGRATION_DIR", integration_dir)
    monkeypatch.setattr(supervisor, "STATE_PATH", state_path)
    monkeypatch.setattr(supervisor, "LOG_PATH", log_path)
    monkeypatch.setattr(
        supervisor,
        "refresh_official_progress",
        lambda path: {"updated": True, "summary": str(path)},
    )
    state = {
        "outcomes": [],
        "current_task": "tabular-playground-series-may-2022",
        "current_run_id": "a40_may_s42",
    }

    first = integrate_external_a800_completions(state)
    second = integrate_external_a800_completions(state)

    assert len(first) == 1
    assert second == []
    assert len(state["outcomes"]) == 1
    assert state["outcomes"][0]["run_id"] == run_id
    assert state["outcomes"][0]["official_results"][0]["score"] == 0.92165
    assert state["current_task"] == "tabular-playground-series-may-2022"
    assert state["current_run_id"] == "a40_may_s42"
    assert (integration_dir / f"{run_id}.json").is_file()


EXPECTED_LITE22 = (
    "aerial-cactus-identification",
    "aptos2019-blindness-detection",
    "denoising-dirty-documents",
    "detecting-insults-in-social-commentary",
    "dog-breed-identification",
    "dogs-vs-cats-redux-kernels-edition",
    "histopathologic-cancer-detection",
    "jigsaw-toxic-comment-classification-challenge",
    "leaf-classification",
    "mlsp-2013-birds",
    "new-york-city-taxi-fare-prediction",
    "nomad2018-predict-transparent-conductors",
    "plant-pathology-2020-fgvc7",
    "random-acts-of-pizza",
    "ranzcr-clip-catheter-line-classification",
    "siim-isic-melanoma-classification",
    "spooky-author-identification",
    "tabular-playground-series-dec-2021",
    "tabular-playground-series-may-2022",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
    "the-icml-2013-whale-challenge-right-whale-redux",
)


def test_campaign_queue_covers_every_unscored_and_recovery_target() -> None:
    ids = [task.competition_id for task in CAMPAIGN_TASKS]
    coverage_ids = {
        task.competition_id
        for task in CAMPAIGN_TASKS
        if task.purpose == "official_coverage"
    }
    recovery_ids = [
        task.competition_id
        for task in CAMPAIGN_TASKS
        if task.purpose == "official_medal_conversion"
    ]

    assert len(ids) == len(set(ids)) == 14
    assert EXPECTED_UNSCORED <= coverage_ids
    assert recovery_ids == EXPECTED_RECOVERY
    assert all(task.wave == "Wave2" for task in CAMPAIGN_TASKS[:8])
    assert CAMPAIGN_TASKS[3].purpose == "official_medal_conversion"
    assert all(
        task.purpose == "official_coverage"
        for index, task in enumerate(CAMPAIGN_TASKS[:8])
        if index != 3
    )
    assert all(task.purpose == "official_medal_conversion" for task in CAMPAIGN_TASKS[8:])
    assert ids[:8] == EXPECTED_WAVE2_GPT56_ORDER


def test_jigsaw_prefetches_aptos_as_the_next_gpu_successor() -> None:
    state = {"outcomes": []}
    task = next_gpu_successor(
        state,
        42,
        "jigsaw-toxic-comment-classification-challenge",
        {"jigsaw-toxic-comment-classification-challenge"},
    )

    assert task is not None
    assert task.competition_id == "aptos2019-blindness-detection"


def test_non_cpu_task_does_not_prefetch_a_successor() -> None:
    assert next_gpu_successor(state={"outcomes": []}, seed=42,
                              current_competition_id="plant-pathology-2020-fgvc7",
                              cpu_concurrent_competitions={"jigsaw-toxic-comment-classification-challenge"}) is None


def test_live_a800_decision_reserves_its_current_and_planned_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "marker.json"
    decision = tmp_path / "decision.json"
    marker.write_text(
        json.dumps(
            {
                "schema": "evomind.mlebench.a800_recovery_queue.v1",
                "pid": 1234,
                "status": "waiting_for_official_completion",
                "current_competition_id": "siim-isic-melanoma-classification",
                "remaining": ["jigsaw-toxic-comment-classification-challenge"],
            }
        ),
        encoding="utf-8",
    )
    decision.write_text(
        json.dumps(
            {
                "schema": "evomind.mlebench.gpt56_adaptive_loop.v1",
                "status": "passed",
                "ok": True,
                "validation_errors": [],
                "decision": {
                    "schema": "evomind.mlebench.adaptive_decision.v1",
                    "next_actions": [
                        {
                            "lane": "a800",
                            "action": "queue_experiment",
                            "competition_id": "aerial-cactus-identification",
                            "requires_active_run_completion": True,
                        },
                        {
                            "lane": "cpu",
                            "action": "collect_and_grade",
                            "competition_id": "new-york-city-taxi-fare-prediction",
                            "requires_active_run_completion": True,
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(supervisor, "process_exists", lambda pid: pid == 1234)

    assert a800_reserved_competitions(marker, decision) == {
        "siim-isic-melanoma-classification",
        "aerial-cactus-identification",
        "jigsaw-toxic-comment-classification-challenge",
    }


def test_process_exists_detects_the_current_process() -> None:
    assert supervisor.process_exists(os.getpid()) is True


def test_primary_queue_skips_live_a800_reservations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        supervisor,
        "a800_reserved_competitions",
        lambda: {CAMPAIGN_TASKS[0].competition_id},
    )
    state = {
        "outcomes": [],
        "a800_reservation_coordination_enabled": True,
    }

    assert next_task(state) == CAMPAIGN_TASKS[1]


def test_campaign_queue_preserves_gpt56_wave2_plan_order() -> None:
    plan = json.loads(supervisor.WAVE2_PLAN.read_text(encoding="utf-8"))

    assert plan["planner"]["model"] == "gpt-5.6-sol"
    assert plan["competition_order"] == EXPECTED_WAVE2_GPT56_ORDER
    assert [task.competition_id for task in CAMPAIGN_TASKS[:8]] == plan["competition_order"]


def test_campaign_freezes_live_gpt56_gateway_and_native_tool_evidence() -> None:
    evidence = llm_runtime_evidence()

    assert evidence["passed"] is True
    assert evidence["required_model"] == "gpt-5.6-sol"
    assert evidence["gateway"]["served_model"] == "gpt-5.6-sol"
    assert evidence["gateway"]["tool_call_count"] >= 1
    assert len(evidence["gateway"]["sha256"]) == 64
    assert evidence["native_tool_loop"]["provider"] == "openai"
    assert evidence["native_tool_loop"]["model"] == "gpt-5.6-sol"
    assert evidence["native_tool_loop"]["native_tool_calls"] >= 1
    assert evidence["native_tool_loop"]["tool_rounds"] >= 2
    assert len(evidence["native_tool_loop"]["sha256"]) == 64


def test_campaign_freezes_complete_live_hpc_lite22_inventory() -> None:
    evidence = data_inventory_evidence()

    assert evidence["passed"] is True
    assert evidence["competition_count"] == 22
    assert evidence["official_prepared"] == 22
    assert evidence["missing"] == 0
    assert evidence["partial_by_size"] == 0
    assert evidence["raw_present_unverified"] == 0
    assert evidence["remote_root"] == "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra"
    assert evidence["disk_free_bytes"] > 0
    assert len(evidence["sha256"]) == 64
    assert len(evidence["low_split_sha256"]) == 64


def test_confirmation_queue_exactly_covers_authoritative_lite22() -> None:
    ids = tuple(task.competition_id for task in CONFIRMATION_TASKS)

    assert ids == EXPECTED_LITE22
    assert len(ids) == len(set(ids)) == 22
    assert all(task.purpose == "seed_confirmation" for task in CONFIRMATION_TASKS)
    assert {
        task.optimization_plan_name for task in CONFIRMATION_TASKS if task.wave == "Wave1"
    } <= {"wave1_gpt56_current.json", "medal_recovery_gpt56_current.json"}


def test_campaign_reachability_proves_18_medals_and_gpt56_order() -> None:
    progress = {
        "any_medal_count": 12,
        "medals_required_to_strictly_exceed_top": 18,
        "remaining_competition_ids": sorted(EXPECTED_UNSCORED),
        "results": [
            {"competition_id": f"medal-{index}", "any_medal": True}
            for index in range(12)
        ],
    }
    result = campaign_reachability(progress, {"priority_order": EXPECTED_RECOVERY})
    assert result["reachable"] is True
    assert result["maximum_reachable_medals"] == 22
    assert result["target_medals"] == 18
    assert result["recovery_queue"] == EXPECTED_RECOVERY


def test_campaign_reachability_fails_when_recovery_plan_or_capacity_is_incomplete() -> None:
    progress = {
        "any_medal_count": 6,
        "medals_required_to_strictly_exceed_top": 18,
        "remaining_competition_ids": [],
        "results": [],
    }
    with pytest.raises(RuntimeError, match="priority order"):
        campaign_reachability(progress, {"priority_order": EXPECTED_RECOVERY[:-1]})

    already_medals = [
        {"competition_id": value, "any_medal": True}
        for value in EXPECTED_RECOVERY[:4]
    ]
    with pytest.raises(RuntimeError, match="mathematically unable"):
        campaign_reachability(
            {**progress, "results": already_medals},
            {"priority_order": EXPECTED_RECOVERY},
        )


def test_queue_resumes_after_finalized_outcomes() -> None:
    state = {
        "outcomes": [
            {
                "competition_id": CAMPAIGN_TASKS[0].competition_id,
                "finalized": True,
                "official_grade_count": 1,
            },
            {"competition_id": CAMPAIGN_TASKS[1].competition_id, "finalized": False},
        ]
    }

    assert completed_competitions(state) == {CAMPAIGN_TASKS[0].competition_id}
    assert next_task(state) == CAMPAIGN_TASKS[1]


def test_runtime_failure_is_retried_but_attempts_are_bounded() -> None:
    competition_id = CAMPAIGN_TASKS[0].competition_id
    failed = {
        "competition_id": competition_id,
        "finalized": True,
        "official_grade_count": 0,
        "promotion_gate_withheld": False,
        "run_id": "failed-attempt-1",
    }
    state = {"outcomes": [failed]}

    assert completed_competitions(state) == set()
    assert next_task(state) == CAMPAIGN_TASKS[0]

    state["outcomes"].extend([
        {**failed, "run_id": "failed-attempt-2"},
        {**failed, "run_id": "failed-attempt-3"},
    ])
    with pytest.raises(RuntimeError, match="exhausted 3 recorded attempts"):
        next_task(state)


def test_explicit_retry_reopens_a_promotion_withheld_task() -> None:
    competition_id = CAMPAIGN_TASKS[0].competition_id
    state = {
        "outcomes": [{
            "competition_id": competition_id,
            "finalized": True,
            "official_grade_count": 0,
            "promotion_gate_withheld": True,
            "retry_required": True,
            "run_id": "old-gated-attempt",
        }]
    }

    assert completed_competitions(state) == set()
    assert next_task(state) == CAMPAIGN_TASKS[0]


def test_evidence_stop_skips_only_primary_recovery_without_official_grade(
    tmp_path, monkeypatch
) -> None:
    evidence = tmp_path / "spooky.json"
    evidence.write_text('{"stop_rule_triggered": true}', encoding="utf-8")
    gates = tmp_path / "gates.json"
    gates.write_text(json.dumps({
        "not_promoted": [{
            "competition_id": "spooky-author-identification",
            "reason": "bounded stop",
            "stop_rule_triggered": True,
            "evidence": {
                "path": str(evidence),
                "sha256": supervisor.sha256_file(evidence),
            },
        }],
    }), encoding="utf-8")
    monkeypatch.setattr(supervisor, "append_log", lambda *_args, **_kwargs: None)

    stops = validated_primary_recovery_stops(gates)
    state = {
        "outcomes": [],
        "confirmation_outcomes": {"43": [], "44": []},
        "reachability": {
            "available_conversion_targets": 8,
            "maximum_reachable_medals": 22,
            "target_medals": 18,
            "reachable": True,
        },
    }
    applied = apply_primary_recovery_stops(state, stops)

    assert len(applied) == 1
    assert applied[0]["official_grade_count"] == 0
    assert applied[0]["official_grader_executed"] is False
    assert "spooky-author-identification" in completed_competitions(state, 42)
    assert "spooky-author-identification" not in completed_competitions(state, 43)
    assert next_task(state, 42) == CAMPAIGN_TASKS[0]
    assert state["reachability"]["available_conversion_targets"] == 7
    assert state["reachability"]["maximum_reachable_medals"] == 21
    assert state["reachability"]["reachable"] is True

    assert apply_primary_recovery_stops(state, stops) == []
    assert state["reachability"]["available_conversion_targets"] == 7
    assert state["reachability"]["maximum_reachable_medals"] == 21


def test_evidence_stop_fails_closed_on_hash_mismatch(tmp_path) -> None:
    evidence = tmp_path / "spooky.json"
    evidence.write_text("{}", encoding="utf-8")
    gates = tmp_path / "gates.json"
    gates.write_text(json.dumps({
        "not_promoted": [{
            "competition_id": "spooky-author-identification",
            "stop_rule_triggered": True,
            "evidence": {"path": str(evidence), "sha256": "0" * 64},
        }],
    }), encoding="utf-8")

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        validated_primary_recovery_stops(gates)


def test_confirmation_seed_outcomes_are_isolated() -> None:
    state = {
        "outcomes": [],
        "confirmation_outcomes": {"43": [], "44": []},
    }
    seed43 = {
        "competition_id": EXPECTED_LITE22[0],
        "finalized": True,
        "official_grade_count": 1,
    }
    seed44 = {
        "competition_id": EXPECTED_LITE22[1],
        "finalized": True,
        "official_grade_count": 1,
    }

    record_outcome(state, 43, seed43)
    record_outcome(state, 44, seed44)

    assert state["outcomes"] == []
    assert completed_competitions(state, 43) == {EXPECTED_LITE22[0]}
    assert completed_competitions(state, 44) == {EXPECTED_LITE22[1]}
    assert next_task(state, 43) == CONFIRMATION_TASKS[1]
    assert next_task(state, 44) == CONFIRMATION_TASKS[0]


@pytest.mark.parametrize(
    ("scored", "valid", "remaining", "medals"),
    [
        (21, 21, 1, 18),
        (22, 21, 0, 18),
        (22, 22, 0, 17),
    ],
)
def test_seed42_must_have_22_valid_grades_and_18_medals_before_confirmation(
    scored: int,
    valid: int,
    remaining: int,
    medals: int,
) -> None:
    result = seed42_confirmation_eligibility(
        {
            "scored_competitions": scored,
            "valid_official_grades": valid,
            "remaining_competitions": remaining,
            "any_medal_count": medals,
        }
    )

    assert result["eligible_for_confirmation_seeds"] is False


def test_seed42_authorizes_confirmation_only_at_official_target() -> None:
    result = seed42_confirmation_eligibility(
        {
            "scored_competitions": 22,
            "valid_official_grades": 22,
            "remaining_competitions": 0,
            "any_medal_count": 18,
        }
    )

    assert result["eligible_for_confirmation_seeds"] is True


def test_multiseed_report_requires_complete_official_evidence_for_all_seeds() -> None:
    complete = {
        "scored_competitions": 22,
        "valid_official_grades": 22,
        "remaining_competitions": 0,
        "any_medal_count": 18,
        "low_split_source": {"sha256": "a" * 64},
        "leaderboard_source": {"sha256": "b" * 64},
    }
    passed = build_multiseed_report_payload(
        {42: dict(complete), 43: dict(complete), 44: dict(complete)}
    )

    assert passed["leaderboard_comparable"] is True
    assert passed["stable_target_met"] is True
    assert passed["status"] == "passed"
    assert passed["complete_seed_count"] == 3
    assert passed["mean_complete_seed_percentage"] == pytest.approx(18 / 22 * 100)
    assert passed["sample_sd_complete_seed_percentage"] == pytest.approx(0.0)
    assert passed["sem_complete_seed_percentage"] == pytest.approx(0.0)

    incomplete_seed = {**complete, "valid_official_grades": 21}
    incomplete = build_multiseed_report_payload(
        {42: dict(complete), 43: incomplete_seed, 44: dict(complete)}
    )
    assert incomplete["leaderboard_comparable"] is False
    assert incomplete["stable_target_met"] is False
    assert incomplete["status"] == "incomplete"


def test_multiseed_report_calculates_sample_sd_and_sem_from_official_medal_rates() -> None:
    def complete_report(medals: int) -> dict[str, object]:
        return {
            "scored_competitions": 22,
            "valid_official_grades": 22,
            "remaining_competitions": 0,
            "any_medal_count": medals,
            "low_split_source": {"sha256": "a" * 64},
            "leaderboard_source": {"sha256": "b" * 64},
        }

    payload = build_multiseed_report_payload(
        {
            42: complete_report(18),
            43: complete_report(19),
            44: complete_report(20),
        }
    )
    rates = [value / 22 * 100 for value in (18, 19, 20)]
    expected_mean = sum(rates) / len(rates)
    expected_sample_sd = math.sqrt(
        sum((value - expected_mean) ** 2 for value in rates) / (len(rates) - 1)
    )

    assert payload["leaderboard_comparable"] is True
    assert payload["stable_target_met"] is True
    assert payload["complete_seed_count"] == 3
    assert payload["mean_complete_seed_percentage"] == pytest.approx(expected_mean)
    assert payload["sample_sd_complete_seed_percentage"] == pytest.approx(
        expected_sample_sd
    )
    assert payload["sem_complete_seed_percentage"] == pytest.approx(
        expected_sample_sd / math.sqrt(len(rates))
    )


def test_dry_run_exposes_primary_and_confirmation_queue_counts(capsys) -> None:
    assert supervisor.main(["--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["primary_seed"] == 42
    assert payload["primary_queue_count"] == 14
    assert payload["confirmation_seeds"] == [43, 44]
    assert payload["confirmation_queue_count"] == 22
    assert tuple(item["competition_id"] for item in payload["confirmation_queue"]) == EXPECTED_LITE22


def test_remote_run_is_final_only_after_process_stops() -> None:
    assert not remote_status_finished({"process": "running", "summary": {"status": "passed"}})
    assert remote_status_finished({"process": "stopped", "summary": None})


def test_remote_retry_recovers_transport_failure_and_clears_state(monkeypatch) -> None:
    calls = 0
    state: dict[str, object] = {}

    def operation():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise EOFError("transport closed")
        return {"passed": True}

    monkeypatch.setattr(supervisor, "write_state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(supervisor, "append_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(supervisor.time, "sleep", lambda *_args, **_kwargs: None)
    result = retry_remote_call(
        operation,
        state,
        status="retrying",
        detail="test",
        event="remote_test_failed",
        failure_key="test_failures",
        poll_seconds=1,
    )

    assert result == {"passed": True}
    assert calls == 2
    assert "test_failures" not in state
    assert "last_error_type" not in state


def test_remote_retry_does_not_mask_deterministic_contract_failure(monkeypatch) -> None:
    monkeypatch.setattr(supervisor.time, "sleep", lambda *_args, **_kwargs: None)
    with pytest.raises(RuntimeError, match="contract mismatch"):
        retry_remote_call(
            lambda: (_ for _ in ()).throw(RuntimeError("contract mismatch")),
            {},
            status="retrying",
            detail="test",
            event="remote_test_failed",
            failure_key="test_failures",
            poll_seconds=1,
        )


def test_performance_evidence_is_compact_and_preserves_peaks() -> None:
    evidence = performance_evidence_from_summary(
        {
            "results": [
                {
                    "competition_id": "example",
                    "runtime_seconds_total": 12.5,
                    "runtime_seconds_model": 10.0,
                    "model_family": "ConvNeXt",
                    "torch_peak_memory_allocated_mib": 4096,
                    "gpu_telemetry": {
                        "sample_count": 13,
                        "peak_memory_used_mib": 8192,
                        "peak_utilization_gpu_percent": 99,
                        "samples": [{"memory_used_mib": 1}] * 13,
                    },
                }
            ]
        }
    )

    assert evidence == [
        {
            "competition_id": "example",
            "runtime_seconds_total": 12.5,
            "runtime_seconds_model": 10.0,
            "runtime_seconds_feature_engineering": None,
            "model_family": "ConvNeXt",
            "gpu_sample_count": 13,
            "gpu_peak_memory_used_mib": 8192,
            "gpu_peak_utilization_percent": 99,
            "torch_peak_memory_allocated_mib": 4096,
        }
    ]


def test_official_result_filter_requires_complete_finite_private_grade() -> None:
    good = {
        "competition_id": "example",
        "mle_private_grader_score": 0.75,
        "valid_submission": True,
        "official_grader_executed": True,
    }
    payload = {
        "results": [
            good,
            {**good, "mle_private_grader_score": math.nan},
            {**good, "valid_submission": False},
            {**good, "official_grader_executed": False},
            {**good, "mle_private_grader_score": None},
        ]
    }

    assert official_results_from_summary(payload) == [good]


def test_intentional_promotion_withhold_is_not_regraded() -> None:
    payload = {
        "results": [{
            "competition_id": "aerial-cactus-identification",
            "official_grader_withheld": True,
            "promotion_gate": {"passed": False},
        }]
    }
    assert len(intentional_promotion_withholds(payload)) == 1


def test_monitor_finalizes_promotion_withhold_without_remote_regrade(
    tmp_path, monkeypatch
) -> None:
    summary = {
        "status": "partial_failure",
        "results": [{
            "competition_id": "aerial-cactus-identification",
            "status": "promotion_gate_failed",
            "official_grader_executed": False,
            "official_grader_withheld": True,
            "promotion_gate": {"passed": False, "name": "aerial_perfect_ranking"},
            "submission_path": "submission.csv",
            "submission_sha256": "a" * 64,
        }],
    }
    collection_root = tmp_path / "collection"
    collection_root.mkdir()
    (collection_root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    class Remote:
        DEFAULT_BUNDLE = Path("bundle.tar.gz")

        def __init__(self) -> None:
            self.regrade_calls = 0

        def read_remote_status(self, _run_id: str):
            return {"process": "stopped", "summary": summary, "checkpoint": {}}

        def regrade_run(self, *_args, **_kwargs):
            self.regrade_calls += 1
            raise AssertionError("promotion-gate withhold must not be regraded")

        def collect_run(self, _run_id: str, *, include_checkpoints: bool):
            assert include_checkpoints is False
            return {"local_root": str(collection_root), "file_count": 2, "passed": True}

    remote = Remote()
    monkeypatch.setattr(supervisor, "write_state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(supervisor, "append_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        supervisor,
        "refresh_official_progress",
        lambda _summary_path: {"updated": False, "reason": "promotion_gate_withheld"},
    )

    outcome = supervisor.monitor_and_finalize(
        remote,
        {},
        CAMPAIGN_TASKS[0],
        "aerial_s42_gate",
        run_poll_seconds=0,
    )

    assert remote.regrade_calls == 0
    assert outcome["promotion_gate_withheld"] is True
    assert outcome["official_grade_count"] == 0
