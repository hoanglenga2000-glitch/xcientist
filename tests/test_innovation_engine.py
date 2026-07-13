from __future__ import annotations

import json

from xsci.innovation_engine import InnovationEngine
from xsci.scientist_state import _decision_for_ready_task


class _MemoryLibrary:
    def __init__(self, records):
        self.records = list(records)

    def retrieve(self, *, task_type, limit):
        matches = [record for record in self.records if record.get("task_type") == task_type]
        return matches[-limit:]


def _memory_record(
    memory_id: str,
    strategy: str,
    *,
    evidence_level: str = "validated",
    run_success: bool = True,
    promoted: bool = True,
    metric_delta: float | None = 0.01,
    no_training_started: bool = False,
) -> dict:
    return {
        "memory_id": memory_id,
        "task_type": "classification",
        "dataset_profile": {"evidence_level": evidence_level},
        "reusable_strategy": strategy,
        "metric_delta": metric_delta,
        "run_success": run_success,
        "promoted": promoted,
        "no_training_started": no_training_started,
    }


def test_blueprints_and_gate_outcomes_do_not_unlock_innovation(tmp_path):
    records = [
        _memory_record(
            f"blueprint-{index}",
            f"artifact_strategy_{index}",
            evidence_level="provisional",
            run_success=False,
            promoted=False,
            metric_delta=None,
            no_training_started=True,
        )
        for index in range(8)
    ]
    engine = InnovationEngine(_MemoryLibrary(records), workspace_root=tmp_path)

    assert engine.ready_for_innovation("classification") is False
    assert engine.propose_innovations("classification") == []


def test_only_validated_positive_promotions_feed_strategy_synthesis(tmp_path):
    validated = [
        _memory_record(f"run-{index}", strategy)
        for index, strategy in enumerate(
            ["target_encoding", "oof_stacking", "calibration", "pseudo_labels", "feature_crossing"]
        )
    ]
    held = _memory_record(
        "held-run",
        "artifact_only_strategy",
        evidence_level="failure",
        promoted=False,
        metric_delta=-0.02,
    )
    engine = InnovationEngine(_MemoryLibrary([*validated, held]), workspace_root=tmp_path)

    assert engine.ready_for_innovation("classification") is True
    proposals = engine.propose_innovations("classification", n=20)
    assert proposals
    assert all("artifact_only_strategy" not in proposal.components for proposal in proposals)
    assert all(proposal.source_tasks for proposal in proposals)


def test_record_attempt_keeps_gate_feedback_without_counting_it_as_tried(tmp_path):
    engine = InnovationEngine(workspace_root=tmp_path)

    gate_record = engine.record_attempt(
        "memory_guided_frontier_blend",
        False,
        run_success=False,
        promoted=False,
        evidence_level="provisional",
        no_training_started=True,
        attempt_id="gate-feedback-1",
    )

    assert gate_record["attempt_status"] == "gated_or_planning_evidence"
    assert engine.stats() == {
        "proposals_generated": 0,
        "innovations_tried": 0,
        "executed_attempts": 0,
        "successes": 0,
        "failures": 0,
        "hit_rate": "0.0%",
        "most_successful": [],
        "evidence_records": 1,
        "negative_evidence": 1,
    }


def test_record_attempt_counts_real_held_and_positive_promoted_runs(tmp_path):
    engine = InnovationEngine(workspace_root=tmp_path)
    held = engine.record_attempt(
        "held_combo",
        False,
        cv_score=0.79,
        metric_delta=-0.01,
        run_success=True,
        promoted=False,
        evidence_level="failure",
        attempt_id="held-1",
    )
    success = engine.record_attempt(
        "validated_combo",
        True,
        cv_score=0.82,
        metric_delta=0.02,
        run_success=True,
        promoted=True,
        evidence_level="validated",
        task_type="classification",
        attempt_id="success-1",
        source_memory_ids=["source-a", "source-b"],
    )

    assert held["attempt_status"] == "executed_held_or_failed"
    assert success["attempt_status"] == "validated_success"
    assert engine.stats()["innovations_tried"] == 1
    assert engine.stats()["executed_attempts"] == 2
    assert engine.stats()["successes"] == 1
    assert engine.stats()["failures"] == 1
    assert engine.stats()["most_successful"] == ["validated_combo"]

    reloaded = InnovationEngine(workspace_root=tmp_path)
    assert reloaded.stats()["innovations_tried"] == 1
    assert reloaded.stats()["executed_attempts"] == 2
    payload = json.loads((tmp_path / ".xsci" / "innovation_log.json").read_text(encoding="utf-8"))
    assert payload["tried"][1]["source_memory_ids"] == ["source-a", "source-b"]


def test_requested_success_without_positive_delta_is_negative_evidence(tmp_path):
    engine = InnovationEngine(workspace_root=tmp_path)
    record = engine.record_attempt(
        "unproven_combo",
        True,
        cv_score=0.8,
        metric_delta=0.0,
        run_success=True,
        promoted=True,
        evidence_level="validated",
    )

    assert record["requested_success"] is True
    assert record["success"] is False
    assert record["attempt_status"] == "executed_held_or_failed"
    assert engine.stats()["successes"] == 0
    assert engine.stats()["failures"] == 1


def _ready_checkpoint(innovation_stats: dict) -> dict:
    return {
        "recent_runs": [{
            "run_id": "run-1",
            "promotions": 1,
            "iterations": 1,
            "best_cv_score": 0.8,
        }],
        "memory": {
            "innovation": innovation_stats,
            "retrospective": {"success_records": 0},
            "scientist_turns": {},
            "scientist_upgrade_backlog": {},
        },
    }


def test_gate_feedback_does_not_trigger_cross_task_innovation(tmp_path):
    engine = InnovationEngine(workspace_root=tmp_path)
    engine.record_attempt(
        "blueprint_only",
        False,
        no_training_started=True,
        evidence_level="provisional",
        run_success=False,
        promoted=False,
    )

    decision = _decision_for_ready_task(
        _ready_checkpoint(engine.stats()), "classification", "tabular", "accuracy"
    )
    assert decision["selected_branch"] == "model_family_or_feature_engineering"


def test_persisted_blueprint_gate_feedback_is_not_a_real_attempt(tmp_path):
    log_path = tmp_path / ".xsci" / "innovation_log.json"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(json.dumps({
        "proposals": [{"strategy_name": "memory_guided_frontier_blend"}],
        "tried": [{
            "trial_id": "innovation_trial_gate_only",
            "task_id": "house-prices",
            "hypothesis_id": "hypothesis-1",
            "strategy_name": "memory_guided_frontier_blend",
            "gate_status": "blocked_by_gate",
            "outcome": "blocked_by_gate",
            "execution_contract": "no_go",
            "no_training_started": True,
        }],
        "successes": 7,
        "failures": 4,
    }), encoding="utf-8")

    engine = InnovationEngine(workspace_root=tmp_path)
    stats = engine.stats()
    assert stats["innovations_tried"] == 0
    assert stats["executed_attempts"] == 0
    assert stats["successes"] == 0
    assert stats["failures"] == 0
    assert stats["negative_evidence"] == 1
    decision = _decision_for_ready_task(
        _ready_checkpoint(stats), "regression", "tabular", "rmse"
    )
    assert decision["selected_branch"] == "model_family_or_feature_engineering"


def test_validated_positive_attempt_triggers_and_survives_restart(tmp_path):
    engine = InnovationEngine(workspace_root=tmp_path)
    strategies = [
        "target_encoding",
        "oof_stacking",
        "calibration",
        "pseudo_labels",
        "feature_crossing",
    ]
    for index, strategy in enumerate(strategies):
        engine.record_attempt(
            strategy,
            True,
            cv_score=0.80 + index / 100,
            metric_delta=0.01 + index / 1000,
            run_success=True,
            promoted=True,
            evidence_level="validated",
            task_type="classification",
            attempt_id=f"validated-{index}",
        )

    reloaded = InnovationEngine(workspace_root=tmp_path)
    assert reloaded.ready_for_innovation("classification") is True
    assert reloaded.propose_innovations("classification")
    decision = _decision_for_ready_task(
        _ready_checkpoint(reloaded.stats()), "classification", "tabular", "accuracy"
    )
    assert decision["selected_branch"] == "cross_task_innovation"
