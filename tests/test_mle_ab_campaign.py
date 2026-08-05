from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_os.mle_ab_campaign import (
    ARMS,
    SCREEN_TASKS,
    CampaignContractError,
    aggregate,
    build_config,
    build_schedule,
    counterbalanced_order,
    freeze_candidate,
    invalidate_campaign,
    load_campaign,
    preregister,
    record_blind_local_grade,
    sha256_file,
    shadow_replay,
)


def config(namespace: str = "evomind-mle-screen-20260802"):
    frozen_source = Path(__file__).resolve().parents[1] / "src" / "research_os" / "experience_mcgs.py"
    return build_config(
        namespace=namespace,
        phase="screen",
        tasks=SCREEN_TASKS,
        provider="local-gateway",
        source_hashes={"src/research_os/experience_mcgs.py": sha256_file(frozen_source)},
        canonical_manifest_sha256="b" * 64,
        created_at="2026-08-02T00:00:00+00:00",
    )


def test_preregistration_is_36_runs_and_counterbalanced(tmp_path: Path):
    cfg = config()
    schedule = build_schedule(cfg)
    assert schedule["expected_task_runs"] == 36
    assert len(schedule["runs"]) == 36
    for task in SCREEN_TASKS:
        for seed in (42, 43, 44):
            pair = [item for item in schedule["runs"] if item["task_id"] == task and item["seed"] == seed]
            assert tuple(item["arm"] for item in pair) == counterbalanced_order(task, seed)
            assert {item["arm"] for item in pair} == set(ARMS)
    result = preregister(tmp_path, cfg)
    assert result["expected_task_runs"] == 36
    with pytest.raises(FileExistsError):
        preregister(tmp_path, cfg)


def test_campaign_lock_rejects_schedule_drift(tmp_path: Path):
    result = preregister(tmp_path, config("evomind-mle-screen-lock-test"))
    campaign = Path(result["campaign_dir"])
    schedule_path = campaign / "schedule.json"
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule["runs"][0]["seed"] = 999
    schedule_path.write_text(json.dumps(schedule, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(CampaignContractError, match="campaign lock schedule hash drifted"):
        aggregate(campaign)


def test_campaign_cannot_alias_frozen_siim_run():
    with pytest.raises(CampaignContractError, match="SIIM"):
        build_config(
            namespace="evomind_siim_isic_a800_job90353_20260730_095826",
            phase="screen",
            tasks=SCREEN_TASKS,
            provider="local-gateway",
            source_hashes={"source": "a" * 64},
            canonical_manifest_sha256="b" * 64,
        )


def test_append_only_invalidation_preserves_frozen_contract_and_blocks_use(tmp_path: Path):
    result = preregister(tmp_path, config("evomind-mle-screen-invalidation-test"))
    campaign = Path(result["campaign_dir"])
    frozen_paths = [campaign / "preregistration.json", campaign / "schedule.json", campaign / "campaign-lock.json"]
    before = {path.name: sha256_file(path) for path in frozen_paths}

    event = invalidate_campaign(
        campaign,
        reason_code="source_hash_drift_zero_runs",
        reason="A corrected selector implementation was frozen before any task-run launched.",
        replacement_namespace="evomind-mle-screen-invalidation-r2",
    )

    assert event["status"] == "invalidated_failed_closed"
    assert event["zero_task_runs_verified"] is True
    assert event["observed_grade_receipts"] == 0
    assert {path.name: sha256_file(path) for path in frozen_paths} == before
    lines = (campaign / "invalidation-events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["event_sha256"] == event["event_sha256"]

    with pytest.raises(CampaignContractError, match="campaign invalidated"):
        load_campaign(campaign)
    with pytest.raises(CampaignContractError, match="already invalidated"):
        invalidate_campaign(
            campaign,
            reason_code="duplicate",
            reason="A duplicate invalidation must never rewrite the existing evidence log.",
        )


def test_invalidation_can_record_source_drift_without_weakening_normal_load(tmp_path: Path):
    cfg = build_config(
        namespace="evomind-mle-screen-source-drift-test",
        phase="screen",
        tasks=SCREEN_TASKS,
        provider="local-gateway",
        source_hashes={"src/research_os/experience_mcgs.py": "a" * 64},
        canonical_manifest_sha256="b" * 64,
        created_at="2026-08-02T00:00:00+00:00",
    )
    campaign = Path(preregister(tmp_path, cfg)["campaign_dir"])
    with pytest.raises(CampaignContractError, match="source hash drifted"):
        load_campaign(campaign)
    event = invalidate_campaign(
        campaign,
        reason_code="source_hash_drift_zero_runs",
        reason="Frozen source changed before the first task-run, so this campaign is sealed.",
    )
    assert event["source_drift"]
    assert event["source_drift"][0]["expected_sha256"] == "a" * 64


def _freeze_first(tmp_path: Path):
    campaign = Path(preregister(tmp_path / "campaigns", config())["campaign_dir"])
    schedule = json.loads((campaign / "schedule.json").read_text(encoding="utf-8"))
    run = schedule["runs"][0]
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    candidate = artifact_root / "submission.csv"
    candidate.write_text("id,prediction\n1,0.5\n", encoding="utf-8")
    frozen = freeze_candidate(
        campaign,
        target_run_id=run["run_id"],
        candidate_files=[candidate],
        allowed_root=artifact_root,
        public_validation={"score": 0.5, "source": "public_validation"},
    )
    return campaign, run, frozen


def test_candidate_is_frozen_before_exactly_one_blind_grade(tmp_path: Path):
    campaign, run, frozen = _freeze_first(tmp_path)
    grade = {
        "status": "ready",
        "valid": True,
        "normalized_score": 0.6,
        "any_medal": False,
        "total_tokens": 1000,
        "new_best_count": 1,
        "prompt_tokens_p99": 300,
        "data_leakage": False,
        "private_feedback_used": False,
        "grader_reuse": False,
        "hash_drift": False,
        "unsupported_claim": False,
    }
    receipt = record_blind_local_grade(
        campaign,
        target_run_id=run["run_id"],
        freeze_sha256=frozen["freeze_sha256"],
        grade=grade,
    )
    assert receipt["execution_count"] == 1
    assert receipt["official_kaggle_result"] is False
    with pytest.raises(CampaignContractError, match="already claimed"):
        record_blind_local_grade(
            campaign,
            target_run_id=run["run_id"],
            freeze_sha256=frozen["freeze_sha256"],
            grade=grade,
        )


def test_aggregate_screen_gate_passes_synthetic_evidence(tmp_path: Path):
    campaign = Path(preregister(tmp_path / "campaigns", config())["campaign_dir"])
    schedule = json.loads((campaign / "schedule.json").read_text(encoding="utf-8"))
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    for index, run in enumerate(schedule["runs"]):
        candidate = artifact_root / f"{index}.csv"
        candidate.write_text(f"id,prediction\n1,{index / 100:.2f}\n", encoding="utf-8")
        frozen = freeze_candidate(
            campaign,
            target_run_id=run["run_id"],
            candidate_files=[candidate],
            allowed_root=artifact_root,
            public_validation={"score": 0.5},
        )
        treatment = run["arm"] == "experience_mcgs_v1"
        grade = {
            "status": "ready",
            "valid": True,
            "normalized_score": 0.65 if treatment else 0.50,
            "any_medal": treatment,
            "total_tokens": 400_000 if treatment else 700_000,
            "new_best_count": 3 if treatment else 2,
            "prompt_tokens_p99": 500 if treatment else 1000,
            "data_leakage": False,
            "private_feedback_used": False,
            "grader_reuse": False,
            "hash_drift": False,
            "unsupported_claim": False,
        }
        record_blind_local_grade(
            campaign,
            target_run_id=run["run_id"],
            freeze_sha256=frozen["freeze_sha256"],
            grade=grade,
        )
    result = aggregate(campaign, bootstrap_samples=500)
    assert result["status"] == "passed"
    assert all(result["checks"].values())
    assert result["observed_task_runs"] == 36
    assert (campaign / "aggregate.json").is_file()
    first_hash = sha256_file(campaign / "aggregate.json")
    assert len(first_hash) == 64
    assert aggregate(campaign, bootstrap_samples=500) == result
    assert sha256_file(campaign / "aggregate.json") == first_hash


def test_aggregate_uses_per_task_arm_medians_for_primary_metrics(tmp_path: Path):
    campaign = Path(preregister(tmp_path / "campaigns", config("evomind-mle-screen-median-contract"))["campaign_dir"])
    schedule = json.loads((campaign / "schedule.json").read_text(encoding="utf-8"))
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    baseline_scores = {42: 0.10, 43: 0.90, 44: 0.80}
    treatment_scores = {42: 0.70, 43: 0.20, 44: 0.95}
    baseline_valid = {42: True, 43: True, 44: False}
    treatment_valid = {42: True, 43: False, 44: False}
    baseline_medal = {42: False, 43: True, 44: True}
    treatment_medal = {42: True, 43: False, 44: False}

    for index, run in enumerate(schedule["runs"]):
        candidate = artifact_root / f"{index}.csv"
        candidate.write_text(f"id,prediction\n1,{index / 100:.2f}\n", encoding="utf-8")
        frozen = freeze_candidate(
            campaign,
            target_run_id=run["run_id"],
            candidate_files=[candidate],
            allowed_root=artifact_root,
            public_validation={"score": 0.5},
        )
        treatment = run["arm"] == "experience_mcgs_v1"
        seed = int(run["seed"])
        grade = {
            "status": "ready",
            "valid": treatment_valid[seed] if treatment else baseline_valid[seed],
            "normalized_score": treatment_scores[seed] if treatment else baseline_scores[seed],
            "any_medal": treatment_medal[seed] if treatment else baseline_medal[seed],
            "total_tokens": 1000,
            "new_best_count": 1,
            "prompt_tokens_p99": 100,
            "data_leakage": False,
            "private_feedback_used": False,
            "grader_reuse": False,
            "hash_drift": False,
            "unsupported_claim": False,
        }
        record_blind_local_grade(
            campaign,
            target_run_id=run["run_id"],
            freeze_sha256=frozen["freeze_sha256"],
            grade=grade,
        )

    result = aggregate(campaign, bootstrap_samples=100)
    # Seed-wise deltas are +0.60, -0.70, +0.15, whose median is +0.15.
    # The preregistered task estimand is instead median(treatment) -
    # median(baseline) = 0.70 - 0.80 = -0.10.
    assert result["metrics"]["median_paired_delta"] == pytest.approx(-0.10)
    assert all(row["median_paired_delta"] == pytest.approx(-0.10) for row in result["task_results"])
    assert result["metrics"]["baseline_valid_rate"] == 1.0
    assert result["metrics"]["treatment_valid_rate"] == 0.0
    assert result["metrics"]["baseline_medal_average"] == 1.0
    assert result["metrics"]["treatment_medal_average"] == 0.0


def test_shadow_replay_is_deterministic_and_private_feedback_free():
    trajectory = [{"node": "EXP000", "score": 0.5}, {"node": "EXP001", "score": 0.6}]
    result = shadow_replay(trajectory, replay=lambda rows: {"selected": [row["node"] for row in rows]})
    assert result["status"] == "passed"
    rejected = shadow_replay(trajectory, replay=lambda _: {"private_feedback": "x"})
    assert rejected["status"] == "failed_closed"
