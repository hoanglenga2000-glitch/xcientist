from __future__ import annotations

import json

import pytest

from scripts.generate_mlebench_medal_recovery_plan import (
    extract_ids,
    extract_json,
    safe_validation_diagnostics,
    validate_targets,
)


def test_medal_recovery_json_and_target_contract():
    candidates = {f"task-{index}" for index in range(7)}
    targets = [f"task-{index}" for index in range(5)]
    payload = extract_json("```json\n" + json.dumps({"conversion_targets": targets}) + "\n```")
    assert validate_targets(payload["conversion_targets"], candidates) == targets
    with pytest.raises(ValueError, match="at least five"):
        validate_targets(targets[:4], candidates)
    with pytest.raises(ValueError, match="at least five"):
        validate_targets([*targets[:4], "unknown"], candidates)
    assert extract_ids([{"competition_id": "task-1"}, {"id": "task-2"}, "task-3"]) == [
        "task-1", "task-2", "task-3"
    ]
    assert extract_ids({"likely": ["task-1", {"competition": "task-2"}]}) == [
        "likely", "task-1", "task-2"
    ]
    assert extract_ids({"task-1": {"actions": ["fit"]}, "task-2": {}})[:2] == [
        "task-1", "actions"
    ]


def test_medal_recovery_diagnostics_are_structural_and_bounded():
    sentinel = "MODEL_CONTROL_TEXT_MUST_NOT_PERSIST"
    diagnostics = safe_validation_diagnostics(
        stage="validate_conversion_targets",
        plan={
            "conversion_targets": [],
            "priority_order": [],
            "rationale_summary": "private prose",
            sentinel: [],
        },
        returned_targets=[{"not": "serialized"}, sentinel, "task-1"],
        filtered_targets=["task-1"],
        returned_priority=[sentinel, "task-2"],
        filtered_priority=[],
        raw_targets={sentinel: ["task-1"]},
        raw_priority=["task-2"],
    )
    assert diagnostics["validation_stage"] == "validate_conversion_targets"
    assert diagnostics["parsed_plan_key_count"] == 4
    assert diagnostics["known_plan_keys_present"] == [
        "conversion_targets", "priority_order", "rationale_summary"
    ]
    assert diagnostics["returned_conversion_target_count"] == 3
    assert diagnostics["returned_priority_count"] == 2
    assert diagnostics["filtered_valid_conversion_targets"] == ["task-1"]
    assert diagnostics["conversion_targets_shape"] == {
        "type": "object", "length": 1
    }
    assert diagnostics["priority_order_shape"]["type"] == "array"
    assert "private prose" not in json.dumps(diagnostics)
    assert sentinel not in json.dumps(diagnostics)
