from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from evomind_runtime.competition_goal import (
    COMPETITIONS,
    FIXED_ALLOCATION,
    FIXED_GOAL_ID,
    FIXED_GOAL_KEY,
    FIXED_RUN_ID,
    GOAL_RECORD_SCHEMA,
    compare_metric_to_human,
    canonical_json_bytes,
    sha256_json,
    load_goal_spec,
    validate_goal_board,
    validate_goal_record,
    validate_human_baseline,
    validate_run_preflight,
    validate_serial_action_sequence,
    validate_goal_spec,
)
from scripts import five_competition_goal_audit as goal_audit
from scripts.five_competition_goal_audit import human_baseline_gate_summary
from evomind_runtime.tools import build_default_registry


DEFAULT_HUMAN_EVIDENCE_SHA = "e" * 64
ARTIFACT_ROLES = (
    "command",
    "environment",
    "data_manifest",
    "source_code",
    "model",
    "metrics",
    "rollback",
    "holdout_ledger",
    "holdout_claim",
    "human_baseline_source",
)


def _artifact_bytes(competition: str, role: str) -> bytes:
    return f"{competition}:{role}:evidence\n".encode("utf-8")


def _artifact_id(competition: str, role: str) -> str:
    return f"artifact-{competition.replace('_', '-')}-{role.replace('_', '-')}"


def _baseline(*, competition: str = "cure_bench", metric: str, direction: str, mean: float = 0.5) -> dict:
    protocol = "Participants used the organizer's frozen task, scoring, aggregation, and holdout protocol."
    source_sha = hashlib.sha256(_artifact_bytes(competition, "human_baseline_source")).hexdigest()
    return {
        "status": "VERIFIED",
        "participant_type": "human_participants",
        "source_url": "https://doi.org/10.1234/example-study",
        "source_doi": "10.1234/example-study",
        "source_authority": "peer_reviewed",
        "source_sha256": source_sha,
        "evidence_artifact_id": _artifact_id(competition, "human_baseline_source"),
        "evidence_artifact_sha256": source_sha,
        "title": "Protocol-comparable human participant study",
        "sample_size": 24,
        "protocol_id": f"{competition}-official-human-protocol-v1",
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(protocol.encode("utf-8")).hexdigest(),
        "protocol_comparable": True,
        "metric": metric,
        "direction": direction,
        "mean": mean,
        "uncertainty": {
            "lower": mean - 0.02,
            "upper": mean + 0.02,
            "confidence_level": 0.95,
            "method": "participant bootstrap",
            "source_sha256": source_sha,
        },
    }


def _item(
    competition: str,
    *,
    direction: str = "higher_is_better",
    metric: str = "score",
    candidate: float = 0.8,
    ci_lower: float = 0.7,
    ci_upper: float = 0.9,
    mean: float = 0.5,
) -> dict:
    solution_id = f"{competition}-candidate-1"
    entries = []
    role_hashes = {}
    for role in ARTIFACT_ROLES:
        payload = _artifact_bytes(competition, role)
        digest = hashlib.sha256(payload).hexdigest()
        role_hashes[role] = digest
        entries.append(
            {
                "artifact_id": _artifact_id(competition, role),
                "path": f"{competition}/{role}.json",
                "role": role,
                "bytes": len(payload),
                "sha256": digest,
                "regular_file": True,
                "symlink": False,
            }
        )
    manifest = {
        "schema": "evomind.artifact-manifest.v1",
        "run_id": FIXED_RUN_ID,
        "competition": competition,
        "solution_id": solution_id,
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
    }
    review_payload = {
        "schema": "evomind.independent-review.v1",
        "run_id": FIXED_RUN_ID,
        "competition": competition,
        "solution_id": solution_id,
        "artifact_manifest_sha256": sha256_json(manifest),
        "verified": True,
        "test_labels_used": False,
        "manifest_closed": True,
    }
    review_raw = canonical_json_bytes(review_payload)
    result = {
        "competition": competition,
        "solution_id": solution_id,
        "goal_status": "VERIFIED_COMPLETE",
        "official_metric": metric,
        "metric_direction": direction,
        "human_baseline": _baseline(
            competition=competition,
            metric=metric,
            direction=direction,
            mean=mean,
        ),
        "human_comparison": {
            "candidate_metric": candidate,
            "ci95_lower": ci_lower,
            "ci95_upper": ci_upper,
        },
        "artifact_manifest": manifest,
        "artifact_manifest_artifact_id": _artifact_id(competition, "artifact_manifest"),
        "artifact_manifest_sha256": sha256_json(manifest),
        "command_sha256": role_hashes["command"],
        "environment_sha256": role_hashes["environment"],
        "data_manifest_sha256": role_hashes["data_manifest"],
        "source_code_sha256": role_hashes["source_code"],
        "model_sha256": role_hashes["model"],
        "metrics_sha256": role_hashes["metrics"],
        "rollback_sha256": role_hashes["rollback"],
        "holdout_ledger": {
            "ledger_id": f"ledger-{competition}-v1",
            "ledger_sha256": role_hashes["holdout_ledger"],
            "claim_id": f"claim-{competition}-candidate-1",
            "claim_sha256": role_hashes["holdout_claim"],
            "unconsumed_at_claim": True,
            "consumed_once": True,
            "selection_data_scope": "development_only",
        },
        "leakage_audit": {
            "test_labels_used": False,
            "hidden_test_labels_used": False,
            "holdout_overlap": 0,
            "id_overlap": 0,
            "architecture_overlap": 0,
            "point_overlap": 0,
            "manifest_closed": True,
            "loader_smoke": True,
        },
        "side_effects": {"signals_sent": 0, "other_processes_modified": False},
        "independent_review": {
            "schema": "evomind.independent-review.v1",
            "artifact_id": f"artifact-{competition.replace('_', '-')}-independent-review",
            "sha256": hashlib.sha256(review_raw).hexdigest(),
            "verified": True,
        },
    }
    if competition == "open_polymer":
        result["existing_candidate_adoption"] = {
            "schema": "evomind.existing-candidate-adoption.v1",
            "source_tool_call_id": "call-open-polymer-existing-candidate",
            "source_solution_id": solution_id,
            "source_artifact_manifest_sha256": result["artifact_manifest_sha256"],
            "retraining_performed": False,
        }
    else:
        result["managed_execution"] = {
            "schema": "evomind.managed-candidate-execution.v1",
            "training_route_tool_call_id": f"call-{competition}-route",
            "hpc_verify_tool_call_id": f"call-{competition}-verify",
            "hpc_execute_tool_call_id": f"call-{competition}-execute",
        }
    return result


def _board() -> dict:
    return {
        "schema": "evomind.goal-board.v2",
        "run_id": FIXED_RUN_ID,
        "allocation": FIXED_ALLOCATION,
        "target_count": len(COMPETITIONS),
        "weather_actions": 0,
        "human_baseline_evidence_artifact_id": "artifact-human-baseline-audit",
        "human_baseline_evidence_sha256": DEFAULT_HUMAN_EVIDENCE_SHA,
        "competitions": [
            _item("cure_bench"),
            _item("e2lmc", metric="mmluvar"),
            _item("mindgames", metric="TrueSkill"),
            _item("open_polymer", direction="lower_is_better", metric="normalized_rmse", candidate=0.3, ci_lower=0.2, ci_upper=0.4),
            _item("ariel_2025", direction="lower_is_better", metric="normalized_rmse", candidate=0.3, ci_lower=0.2, ci_upper=0.4),
        ],
    }


def _independent_review_bytes(item: dict) -> bytes:
    return canonical_json_bytes(
        {
            "schema": "evomind.independent-review.v1",
            "run_id": FIXED_RUN_ID,
            "competition": item["competition"],
            "solution_id": item["solution_id"],
            "artifact_manifest_sha256": item["artifact_manifest_sha256"],
            "verified": True,
            "test_labels_used": False,
            "manifest_closed": True,
        }
    )


def _insert_deliverable(
    connection: sqlite3.Connection,
    *,
    artifact_id: str,
    name: str,
    path: Path,
    raw: bytes,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    connection.execute(
        "INSERT INTO deliverables VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            artifact_id,
            FIXED_RUN_ID,
            FIXED_RUN_ID,
            name,
            str(path),
            "application/json",
            len(raw),
            hashlib.sha256(raw).hexdigest(),
            "independent_verify",
            "structured_text",
            "2026-08-30T00:00:00Z",
        ),
    )


def test_valid_board_requires_all_five_strict_human_gates() -> None:
    result = validate_goal_board(_board())
    assert result.valid is True
    assert result.completion_proven is True
    assert result.complete_count == 5


def test_declared_complete_without_human_baseline_is_not_complete() -> None:
    board = _board()
    board["competitions"][0].pop("human_baseline")
    result = validate_goal_board(board)
    assert result.valid is False
    assert result.complete_count == 4
    assert any(issue.code == "cure_bench:HUMAN_BASELINE_MISSING" for issue in result.issues)


@pytest.mark.parametrize(
    ("direction", "candidate", "lower", "upper", "expected_issue"),
    [
        ("higher_is_better", 0.6, 0.49, 0.7, "HUMAN_GATE_CI_LOWER_NOT_ABOVE_ZERO"),
        ("lower_is_better", 0.4, 0.2, 0.51, "HUMAN_GATE_CI_LOWER_NOT_ABOVE_ZERO"),
        ("higher_is_better", 0.49, 0.45, 0.5, "HUMAN_GATE_POINT_ESTIMATE_NOT_BETTER"),
    ],
)
def test_direction_normalized_human_gate_is_conservative(
    direction: str,
    candidate: float,
    lower: float,
    upper: float,
    expected_issue: str,
) -> None:
    result = compare_metric_to_human(
        candidate_metric=candidate,
        human_mean=0.5,
        ci95_lower=lower,
        ci95_upper=upper,
        direction=direction,
    )
    assert result.valid is False
    assert any(issue.code == expected_issue for issue in result.issues)


def test_human_baseline_rejects_model_agent_reference() -> None:
    value = _baseline(metric="score", direction="higher_is_better")
    value["participant_type"] = "model_agent"
    result = validate_human_baseline(value, expected_metric="score", expected_direction="higher_is_better")
    assert result.valid is False
    assert any(issue.code == "HUMAN_PARTICIPANTS_NOT_PROVEN" for issue in result.issues)


def test_human_baseline_rejects_local_source_path() -> None:
    value = _baseline(metric="score", direction="higher_is_better")
    value["source_url"] = "C:\\evidence\\human-study.pdf"
    value.pop("source_doi")
    result = validate_human_baseline(value, expected_metric="score", expected_direction="higher_is_better")
    assert result.valid is False
    assert any(issue.code == "HUMAN_BASELINE_SOURCE_MISSING" for issue in result.issues)


def test_complete_board_rejects_non_authoritative_source_unclosed_manifest_and_missing_overlap() -> None:
    board = _board()
    item = board["competitions"][0]
    baseline = item["human_baseline"]
    baseline["source_url"] = "not-a-url-or-doi"
    baseline.pop("source_doi")
    manifest = item["artifact_manifest"]
    manifest["files"] = [{"anything": "accepted-before-the-fix"}]
    manifest["file_count"] = 1
    manifest["total_bytes"] = 0
    item["artifact_manifest_sha256"] = sha256_json(manifest)
    for key in ("holdout_overlap", "id_overlap", "architecture_overlap", "point_overlap"):
        item["leakage_audit"].pop(key)

    result = validate_goal_board(board)

    assert result.completion_proven is False
    codes = {issue.code for issue in result.issues}
    assert "cure_bench:HUMAN_BASELINE_SOURCE_MISSING" in codes
    assert "cure_bench:ARTIFACT_MANIFEST_ENTRY_INVALID" in codes
    assert "cure_bench:HOLDOUT_OVERLAP_EVIDENCE_MISSING" in codes


def test_goal_record_rejects_human_evidence_drift_and_initial_negative_evidence_completion() -> None:
    spec = load_goal_spec("configs/g21_five_competition_goal_v1_frozen.json")
    board = _board()
    board["human_baseline_evidence_sha256"] = str(spec["human_baseline_evidence_sha256"])
    record = {
        "schema": GOAL_RECORD_SCHEMA,
        "goal_id": FIXED_GOAL_ID,
        "run_id": FIXED_RUN_ID,
        "allocation": FIXED_ALLOCATION,
        "goal_key": FIXED_GOAL_KEY,
        "status": "completed",
        "spec_sha256": sha256_json(spec),
        "spec": spec,
        "board_sha256": sha256_json(board),
        "human_baseline_sha256": str(spec["human_baseline_evidence_sha256"]),
        "board": board,
    }
    result = validate_goal_record(record)
    assert result.valid is False
    assert any(issue.code == "GOAL_RECORD_HUMAN_BASELINE_NEGATIVE_EVIDENCE" for issue in result.issues)

    drifted = deepcopy(record)
    drifted["human_baseline_sha256"] = "f" * 64
    result = validate_goal_record(drifted)
    assert result.valid is False
    assert any(issue.code == "GOAL_RECORD_HUMAN_BASELINE_BOARD_MISMATCH" for issue in result.issues)


def test_run_preflight_requires_fixed_id_and_quiescent_counts() -> None:
    ok, issues = validate_run_preflight(
        {"id": FIXED_RUN_ID, "effective_active_count": 0, "pending_approval_count": 0}
    )
    assert ok is True
    assert issues == ()

    ok, issues = validate_run_preflight(
        {"id": "run_other", "effective_active_count": 1, "pending_approval_count": 2}
    )
    assert ok is False
    assert {issue.code for issue in issues} == {
        "RUN_ID_MISMATCH",
        "EFFECTIVE_ACTIVE_NONZERO",
        "PENDING_APPROVAL_NONZERO",
    }


def test_run_preflight_accepts_numeric_zero_strings_but_rejects_missing_counts() -> None:
    ok, issues = validate_run_preflight(
        {"id": FIXED_RUN_ID, "effective_active_count": "0", "pending_approval_count": "0"}
    )
    assert ok is True
    assert issues == ()
    ok, issues = validate_run_preflight({"id": FIXED_RUN_ID})
    assert ok is False
    assert {issue.code for issue in issues} == {
        "EFFECTIVE_ACTIVE_NONZERO",
        "PENDING_APPROVAL_NONZERO",
    }


def test_run_preflight_rejects_malformed_count_lists() -> None:
    ok, issues = validate_run_preflight(
        {"id": FIXED_RUN_ID, "active_tool_calls": {"id": "x"}, "approvals": "pending"}
    )
    assert ok is False
    assert {issue.code for issue in issues} == {
        "EFFECTIVE_ACTIVE_NONZERO",
        "PENDING_APPROVAL_NONZERO",
    }


def test_serial_action_sequence_rejects_duplicate_idempotency_and_weather() -> None:
    calls = [
        {"run_id": FIXED_RUN_ID, "tool_name": "hpc_verify", "idempotency_key": "same"},
        {"run_id": FIXED_RUN_ID, "tool_name": "competition_data_accelerate", "idempotency_key": "same"},
    ]
    ok, issues = validate_serial_action_sequence(
        calls,
        competition="weather4cast",
        solution_id="candidate-1",
    )
    assert ok is False
    assert {issue.code for issue in issues} >= {
        "PROHIBITED_COMPETITION",
        "PROHIBITED_TOOL",
        "IDEMPOTENCY_KEY_MISSING_OR_DUPLICATE",
    }


def test_serial_action_sequence_accepts_one_managed_candidate() -> None:
    calls = [
        {"run_id": FIXED_RUN_ID, "tool_name": "training_route", "solution_id": "candidate-1", "idempotency_key": "route-1"},
        {"run_id": FIXED_RUN_ID, "tool_name": "hpc_verify", "solution_id": "candidate-1", "idempotency_key": "verify-1"},
        {"run_id": FIXED_RUN_ID, "tool_name": "hpc_execute_solution", "solution_id": "candidate-1", "idempotency_key": "execute-1"},
    ]
    ok, issues = validate_serial_action_sequence(
        calls,
        competition="cure_bench",
        solution_id="candidate-1",
    )
    assert ok is True
    assert issues == ()


def test_serial_action_sequence_rejects_incomplete_real_tool_sequence() -> None:
    calls = [
        {"run_id": FIXED_RUN_ID, "tool_name": "training_route", "solution_id": "candidate-1", "idempotency_key": "route-1"},
        {"run_id": FIXED_RUN_ID, "tool_name": "hpc_verify", "solution_id": "candidate-1", "idempotency_key": "verify-1"},
    ]
    ok, issues = validate_serial_action_sequence(calls, competition="cure_bench", solution_id="candidate-1")
    assert ok is False
    assert any(issue.code == "MANAGED_CALL_ORDER_INVALID" for issue in issues)


def test_serial_action_sequence_rejects_nonexistent_evidence_stage_tool_rows() -> None:
    calls = [
        {"run_id": FIXED_RUN_ID, "tool_name": "training_route", "idempotency_key": "route-1"},
        {"run_id": FIXED_RUN_ID, "tool_name": "hpc_verify", "idempotency_key": "verify-1"},
        {"run_id": FIXED_RUN_ID, "tool_name": "hpc_execute_solution", "solution_id": "candidate-1", "idempotency_key": "execute-1"},
        {"run_id": FIXED_RUN_ID, "tool_name": "artifact_collect", "idempotency_key": "collect-1"},
    ]
    ok, issues = validate_serial_action_sequence(calls, competition="cure_bench", solution_id="candidate-1")
    assert ok is False
    assert any(issue.code == "EVIDENCE_STAGE_IS_NOT_A_TOOL" for issue in issues)


def test_goal_audit_real_tool_stages_match_runtime_registry() -> None:
    specs = {spec.name: spec for spec in build_default_registry().specs()}
    assert goal_audit.MANAGED_SEQUENCE_TOOLS == {
        "training_route", "hpc_verify", "hpc_execute_solution"
    }
    assert goal_audit.NONEXISTENT_EVIDENCE_STAGE_TOOLS.isdisjoint(specs)
    assert goal_audit.MANAGED_SEQUENCE_TOOLS <= set(specs)
    assert "solution_id" not in specs["training_route"].input_schema["properties"]
    assert specs["hpc_verify"].input_schema["properties"] == {}
    assert {"competition", "solution_id"} <= set(
        specs["hpc_execute_solution"].input_schema["properties"]
    )


def test_serial_action_sequence_rejects_missing_or_duplicate_managed_steps() -> None:
    calls = [
        {"run_id": FIXED_RUN_ID, "tool_name": "hpc_verify", "solution_id": "candidate-1", "idempotency_key": "verify-1"},
        {"run_id": FIXED_RUN_ID, "tool_name": "hpc_execute_solution", "solution_id": "candidate-1", "idempotency_key": "execute-1"},
        {"run_id": FIXED_RUN_ID, "tool_name": "hpc_execute_solution", "solution_id": "candidate-1", "idempotency_key": "execute-2"},
    ]
    ok, issues = validate_serial_action_sequence(calls, competition="cure_bench", solution_id="candidate-1")
    assert ok is False
    assert any(issue.code == "MANAGED_CALL_ORDER_INVALID" for issue in issues)
    assert any(issue.code == "SOLUTION_REPEATED" for issue in issues)


def test_serial_action_sequence_rejects_unsettled_calls_and_parallel_solutions() -> None:
    calls = [
        {"run_id": FIXED_RUN_ID, "tool_name": "training_route", "solution_id": "candidate-1", "idempotency_key": "route-1", "status": "completed"},
        {"run_id": FIXED_RUN_ID, "tool_name": "hpc_verify", "solution_id": "candidate-2", "idempotency_key": "verify-2", "status": "running"},
        {"run_id": FIXED_RUN_ID, "tool_name": "hpc_execute_solution", "solution_id": "candidate-2", "idempotency_key": "execute-2", "status": "completed"},
    ]
    ok, issues = validate_serial_action_sequence(calls, competition="cure_bench", solution_id="candidate-2")
    assert ok is False
    assert any(issue.code == "UNSETTLED_CALL" for issue in issues)
    assert any(issue.code == "MULTIPLE_SOLUTIONS_IN_SEQUENCE" for issue in issues)


def test_goal_unsettled_projection_scopes_paused_rows_and_accepts_real_settlement() -> None:
    boundary = datetime(2026, 8, 30, tzinfo=timezone.utc)
    calls = [
        {
            "id": "pre-goal-paused",
            "tool_name": "training_route",
            "status": "paused",
            "created_at": "2026-08-29T23:00:00Z",
            "started_at": "2026-08-29T23:00:00Z",
        },
        {
            "id": "current-paused",
            "tool_name": "hpc_execute_solution",
            "status": "paused",
            "created_at": "2026-08-30T00:01:00Z",
            "started_at": "2026-08-30T00:01:00Z",
        },
        {
            "id": "settled-approval",
            "tool_name": "hpc_verify",
            "status": "waiting_approval",
            "approval_id": "approval-rejected",
            "created_at": "2026-08-30T00:02:00Z",
            "started_at": "2026-08-30T00:02:00Z",
        },
    ]
    unsettled = goal_audit.goal_unsettled_calls_since(
        calls,
        since=boundary,
        approvals_by_id={"approval-rejected": {"status": "rejected"}},
    )
    assert [item["id"] for item in unsettled] == ["current-paused"]

    calls.append(
        {
            "id": "successor",
            "tool_name": "hpc_execute_solution",
            "status": "completed",
            "created_at": "2026-08-30T00:03:00Z",
            "started_at": "2026-08-30T00:03:00Z",
            "completed_at": "2026-08-30T00:04:00Z",
        }
    )
    assert goal_audit.goal_unsettled_calls_since(
        calls,
        since=boundary,
        approvals_by_id={"approval-rejected": {"status": "rejected"}},
    ) == []


def test_board_requires_explicit_fixed_binding() -> None:
    board = _board()
    board.pop("run_id")
    board.pop("allocation")
    result = validate_goal_board(board)
    assert result.valid is False
    assert any(issue.code == "GOAL_BOARD_RUN_MISMATCH" for issue in result.issues)
    assert any(issue.code == "GOAL_BOARD_ALLOCATION_MISMATCH" for issue in result.issues)


def test_board_scope_and_weather_counter_are_fail_closed() -> None:
    board = _board()
    board["competitions"] = board["competitions"][:-1]
    board["weather_actions"] = 1
    result = validate_goal_board(board)
    assert result.valid is False
    assert any(issue.code == "GOAL_BOARD_SCOPE_MISMATCH" for issue in result.issues)
    assert any(issue.code == "WEATHER_ACTIONS_NONZERO" for issue in result.issues)


def test_artifact_hash_contract_is_required_for_declared_completion() -> None:
    board = _board()
    item = deepcopy(board["competitions"][0])
    item.pop("model_sha256")
    board["competitions"][0] = item
    result = validate_goal_board(board)
    assert result.valid is False
    assert any(issue.code == "cure_bench:MODEL_SHA256_MISSING" for issue in result.issues)


def test_artifact_manifest_canonical_hash_is_required() -> None:
    board = _board()
    item = board["competitions"][0]
    item["artifact_manifest_sha256"] = "f" * 64
    result = validate_goal_board(board)
    assert result.valid is False
    assert any(issue.code == "cure_bench:ARTIFACT_MANIFEST_HASH_MISMATCH" for issue in result.issues)


def test_open_polymer_completion_requires_existing_candidate_adoption_binding() -> None:
    board = _board()
    open_item = next(item for item in board["competitions"] if item["competition"] == "open_polymer")
    open_item.pop("existing_candidate_adoption")
    result = validate_goal_board(board)
    assert result.completion_proven is False
    assert any(
        issue.code == "open_polymer:OPEN_POLYMER_ADOPTION_SCHEMA_INVALID"
        for issue in result.issues
    )


def test_open_polymer_adoption_audit_binds_historical_source_and_review_results() -> None:
    item = next(entry for entry in _board()["competitions"] if entry["competition"] == "open_polymer")
    manifest_sha = item["artifact_manifest_sha256"]
    solution_id = item["solution_id"]
    source_id = item["existing_candidate_adoption"]["source_tool_call_id"]
    since = datetime(2026, 8, 30, tzinfo=timezone.utc)
    calls = [
        {
            "id": source_id,
            "tool_name": "hpc_execute_solution",
            "status": "completed",
            "created_at": "2026-08-29T00:00:00Z",
            "completed_at": "2026-08-29T00:10:00Z",
            "idempotency_key": "historic-execute",
            "arguments": {"competition": "open_polymer", "solution_id": solution_id},
            "result": {"ok": True, "content": {"artifact_manifest_sha256": manifest_sha}},
        },
    ]
    valid, issues = goal_audit.validate_open_polymer_adoption(
        board_item=item,
        all_calls=calls,
        since=since,
    )
    assert valid is True
    assert issues == []

    invalid_time = deepcopy(calls)
    invalid_time[0]["created_at"] = "not-a-timestamp"
    valid, issues = goal_audit.validate_open_polymer_adoption(
        board_item=item,
        all_calls=invalid_time,
        since=since,
    )
    assert valid is False
    assert any(issue["code"] == "OPEN_POLYMER_SOURCE_CALL_BINDING_INVALID" for issue in issues)

    empty_result = deepcopy(calls)
    empty_result[0]["result"] = {}
    valid, issues = goal_audit.validate_open_polymer_adoption(
        board_item=item,
        all_calls=empty_result,
        since=since,
    )
    assert valid is False
    assert any(issue["code"] == "OPEN_POLYMER_SOURCE_CALL_BINDING_INVALID" for issue in issues)

    retrained = deepcopy(calls)
    retrained.append(
        {
            "id": "call-open-retrain",
            "tool_name": "hpc_execute_solution",
            "status": "failed",
            "created_at": "2026-08-30T00:05:00Z",
            "completed_at": "2026-08-30T00:06:00Z",
            "idempotency_key": "execute-new",
            "arguments": {"competition": "open_polymer", "solution_id": solution_id},
            "result": {"ok": False},
        }
    )
    valid, issues = goal_audit.validate_open_polymer_adoption(
        board_item=item,
        all_calls=retrained,
        since=since,
    )
    assert valid is False
    assert any(issue["code"] == "OPEN_POLYMER_WINDOW_RETRAINING_DETECTED" for issue in issues)


def test_versioned_goal_spec_is_fixed_and_loadable() -> None:
    spec = load_goal_spec("configs/g21_five_competition_goal.json")
    ok, issues = validate_goal_spec(spec)
    assert ok is True
    assert issues == ()
    assert spec["run_id"] == FIXED_RUN_ID
    assert spec["allocation"] == FIXED_ALLOCATION
    assert spec["execution"]["max_concurrent_solutions"] == 1


def test_goal_spec_rejects_weather_or_parallelism_drift() -> None:
    spec = load_goal_spec("configs/g21_five_competition_goal.json")
    spec["competitions"] = [*spec["competitions"], "weather4cast"]
    spec["execution"]["max_concurrent_solutions"] = 2
    ok, issues = validate_goal_spec(spec)
    assert ok is False
    assert {issue.code for issue in issues} >= {
        "GOAL_SPEC_COMPETITION_SET_INVALID",
        "GOAL_SPEC_CONCURRENCY_INVALID",
    }


def test_goal_spec_rejects_goal_identity_drift() -> None:
    spec = load_goal_spec("configs/g21_five_competition_goal.json")
    changed = deepcopy(spec)
    changed["goal_id"] = "goal_other"
    changed["goal_key"] = "goal_other"
    ok, issues = validate_goal_spec(changed)
    assert ok is False
    assert {issue.code for issue in issues} >= {
        "GOAL_SPEC_ID_MISMATCH",
        "GOAL_SPEC_KEY_MISMATCH",
    }


def test_goal_spec_requires_complete_forbidden_and_artifact_contracts() -> None:
    spec = load_goal_spec("configs/g21_five_competition_goal.json")
    for forbidden in (
        "competition_data_prepare",
        "competition_data_accelerate",
        "kaggle_download",
        "kaggle_submit",
        "join_competition",
        "accept_terms",
    ):
        changed = deepcopy(spec)
        changed["forbidden"] = [item for item in changed["forbidden"] if item != forbidden]
        ok, issues = validate_goal_spec(changed)
        assert ok is False
        assert any(issue.code == "GOAL_SPEC_FORBIDDEN_SCOPE_INCOMPLETE" for issue in issues)
    changed = deepcopy(spec)
    changed["artifact_requirements"] = changed["artifact_requirements"][:-1]
    ok, issues = validate_goal_spec(changed)
    assert ok is False
    assert any(issue.code == "GOAL_SPEC_ARTIFACT_REQUIREMENTS_INCOMPLETE" for issue in issues)


def test_five_competition_audit_uses_strict_contract(tmp_path: Path) -> None:
    board_definition = _board()
    human_evidence_path = tmp_path / "human-baseline.json"
    human_evidence = {
        "schema": "evomind.five_competition_human_baseline_audit.v1",
        "run_id": FIXED_RUN_ID,
        "allocation": FIXED_ALLOCATION,
        "classification": "HUMAN_BASELINE_VERIFIED",
        "competitions": {
            name: {
                "status": "VERIFIED",
                "unique_gate": "source-backed",
                "metric": next(item["official_metric"] for item in board_definition["competitions"] if item["competition"] == name),
                "direction": next(item["metric_direction"] for item in board_definition["competitions"] if item["competition"] == name),
                    "baseline": _baseline(
                        competition=name,
                        metric=next(item["official_metric"] for item in board_definition["competitions"] if item["competition"] == name),
                    direction=next(item["metric_direction"] for item in board_definition["competitions"] if item["competition"] == name),
                ),
            }
            for name in COMPETITIONS
        },
        "completion": {"human_baseline_gate_passed": True, "all_five_verified": True},
    }
    human_evidence_path.write_text(json.dumps(human_evidence), encoding="utf-8")
    human_evidence_sha = hashlib.sha256(human_evidence_path.read_bytes()).hexdigest()
    board_definition["human_baseline_evidence_sha256"] = human_evidence_sha
    board_path = tmp_path / "goal-board-v2.json"
    board_bytes = (json.dumps(board_definition, sort_keys=True) + "\n").encode("utf-8")
    board_path.write_bytes(board_bytes)
    goal_spec_path = tmp_path / "goal-spec.json"
    goal_spec = load_goal_spec("configs/g21_five_competition_goal_v1_frozen.json")
    goal_spec["human_baseline_evidence_sha256"] = human_evidence_sha
    goal_spec_path.write_text(json.dumps(goal_spec), encoding="utf-8")
    database = tmp_path / "runtime.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        PRAGMA foreign_keys=ON;
        CREATE TABLE assistant_runs (id TEXT PRIMARY KEY, status TEXT);
        CREATE TABLE goal_records (
            id TEXT PRIMARY KEY, run_id TEXT NOT NULL, goal_key TEXT NOT NULL,
            schema TEXT NOT NULL, allocation TEXT NOT NULL,
            spec_sha256 TEXT NOT NULL, spec_json TEXT NOT NULL,
            board_sha256 TEXT NOT NULL, human_baseline_sha256 TEXT NOT NULL,
            board_json TEXT NOT NULL, status TEXT NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            UNIQUE(goal_key), FOREIGN KEY(run_id) REFERENCES assistant_runs(id)
        );
        CREATE TABLE approvals (session_id TEXT, status TEXT);
        CREATE TABLE events (session_id TEXT, event_type TEXT, payload_json TEXT, created_at TEXT);
        CREATE TABLE deliverables (
            id TEXT, run_id TEXT, session_id TEXT, name TEXT, path TEXT,
            media_type TEXT, bytes INTEGER, sha256 TEXT, source_tool_call TEXT,
            preview_kind TEXT, verified_at TEXT
        );
        CREATE TABLE tool_calls (
            id TEXT, session_id TEXT, tool_name TEXT, status TEXT, created_at TEXT,
            completed_at TEXT, arguments_json TEXT, result_json TEXT, idempotency_key TEXT
        );
        """
    )
    connection.execute(
        "INSERT INTO assistant_runs(id,status) VALUES (?,?)",
        (FIXED_RUN_ID, "completed"),
    )
    connection.execute(
        "INSERT INTO events VALUES (?,?,?,?)",
        (FIXED_RUN_ID, "goal.created", json.dumps({"goal_id": FIXED_GOAL_ID}), "2026-08-30T00:00:00Z"),
    )
    connection.execute(
        "INSERT INTO tool_calls VALUES (?,?,?,?,?,?,?,?,?)",
        (
            "call-historical-weather",
            FIXED_RUN_ID,
            "competition_data_accelerate",
            "completed",
            "2026-08-28T00:00:00Z",
            "2026-08-28T00:01:00Z",
            json.dumps({"run_id": FIXED_RUN_ID, "competition": "weather4cast"}),
            json.dumps({"ok": True}),
            "historical-weather-before-goal",
        ),
    )
    connection.execute(
        "INSERT INTO tool_calls VALUES (?,?,?,?,?,?,?,?,?)",
        (
            "call-historical-orphan",
            FIXED_RUN_ID,
            "training_route",
            "running",
            "2026-08-28T00:02:00Z",
            "",
            json.dumps({"run_id": FIXED_RUN_ID, "competition": "legacy", "solution_id": "legacy"}),
            "{}",
            "historical-orphan-before-goal",
        ),
    )
    spec = goal_spec
    spec_json = json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    board = board_definition
    board_json = json.dumps(board, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    connection.execute(
        "INSERT INTO goal_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            FIXED_GOAL_ID,
            FIXED_RUN_ID,
            FIXED_GOAL_KEY,
            GOAL_RECORD_SCHEMA,
            FIXED_ALLOCATION,
            sha256_json(spec),
            spec_json,
            sha256_json(board),
                human_evidence_sha,
            board_json,
            "completed",
            "2026-08-30T00:00:00Z",
            "2026-08-30T00:00:00Z",
        ),
    )
    _insert_deliverable(
        connection,
        artifact_id=board["human_baseline_evidence_artifact_id"],
        name=human_evidence_path.name,
        path=human_evidence_path,
        raw=human_evidence_path.read_bytes(),
    )
    for item in board["competitions"]:
        competition = item["competition"]
        manifest = item["artifact_manifest"]
        manifest_raw = canonical_json_bytes(manifest)
        _insert_deliverable(
            connection,
            artifact_id=item["artifact_manifest_artifact_id"],
            name=f"{competition}-artifact-manifest.json",
            path=tmp_path / f"{competition}-artifact-manifest.json",
            raw=manifest_raw,
        )
        for entry in manifest["files"]:
            role = entry["role"]
            _insert_deliverable(
                connection,
                artifact_id=entry["artifact_id"],
                name=Path(entry["path"]).name,
                path=tmp_path / entry["path"],
                raw=_artifact_bytes(competition, role),
            )
        review_raw = _independent_review_bytes(item)
        _insert_deliverable(
            connection,
            artifact_id=item["independent_review"]["artifact_id"],
            name="independent-review.json",
            path=tmp_path / competition / "independent-review.json",
            raw=review_raw,
        )
    open_item = next(item for item in board["competitions"] if item["competition"] == "open_polymer")
    connection.execute(
        "INSERT INTO tool_calls VALUES (?,?,?,?,?,?,?,?,?)",
        (
            open_item["existing_candidate_adoption"]["source_tool_call_id"],
            FIXED_RUN_ID,
            "hpc_execute_solution",
            "completed",
            "2026-08-28T01:00:00Z",
            "2026-08-28T01:10:00Z",
            json.dumps({
                "run_id": FIXED_RUN_ID,
                "competition": "open_polymer",
                "solution_id": open_item["solution_id"],
            }),
            json.dumps({
                "ok": True,
                "content": {
                    "status": "completed",
                    "published_artifacts": [
                        {"sha256": open_item["artifact_manifest_sha256"], "name": "artifact-manifest.json"}
                    ],
                },
            }),
            "open-polymer-historical-execute",
        ),
    )
    for competition_index, competition in enumerate(COMPETITIONS):
        if competition == "open_polymer":
            continue
        board_item = next(item for item in board["competitions"] if item["competition"] == competition)
        solution_id = f"{competition}-candidate-1"
        sequence = ("training_route", "hpc_verify", "hpc_execute_solution")
        for index, tool_name in enumerate(sequence, 1):
            second = competition_index * 10 + index
            arguments = (
                {"competition": competition, "task_description": f"{competition} route"}
                if tool_name == "training_route"
                else {}
                if tool_name == "hpc_verify"
                else {"competition": competition, "solution_id": solution_id}
            )
            call_id = {
                "training_route": board_item["managed_execution"]["training_route_tool_call_id"],
                "hpc_verify": board_item["managed_execution"]["hpc_verify_tool_call_id"],
                "hpc_execute_solution": board_item["managed_execution"]["hpc_execute_tool_call_id"],
            }[tool_name]
            connection.execute(
                "INSERT INTO tool_calls VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    call_id,
                    FIXED_RUN_ID,
                    tool_name,
                    "completed",
                    f"2026-08-30T00:00:{second:02d}Z",
                    f"2026-08-30T00:00:{second:02d}Z",
                    json.dumps(arguments),
                    json.dumps(
                        {
                            "ok": True,
                            "content": {
                                "samples_requested": 5,
                                "samples_passed": 5,
                                "samples": [{"complete": True} for _ in range(5)],
                            },
                        }
                        if tool_name == "hpc_verify"
                        else {
                            "ok": True,
                            "content": {
                                "artifact_manifest_sha256": board_item["artifact_manifest_sha256"],
                                "published_artifacts": [
                                    {"sha256": board_item["artifact_manifest_sha256"]}
                                ],
                            },
                        }
                    ),
                    f"{competition}-step-{index}",
                ),
            )
    connection.execute(
        "INSERT INTO deliverables VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            "artifact-board",
            FIXED_RUN_ID,
            FIXED_RUN_ID,
            "goal-board-v2.json",
            str(board_path),
            "application/json",
            len(board_bytes),
            hashlib.sha256(board_bytes).hexdigest(),
            "goal_update",
            "structured_text",
            "2026-08-30T00:00:00Z",
        ),
    )
    connection.execute(
        "INSERT INTO events VALUES (?,?,?,?)",
        (
            FIXED_RUN_ID,
            "goal.updated",
            json.dumps({"goal_id": FIXED_GOAL_ID, "board_sha256": sha256_json(board)}),
            "2026-08-30T00:10:00Z",
        ),
    )
    connection.commit()
    connection.close()
    output = tmp_path / "audit.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "five_competition_goal_audit.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--runtime-db",
            str(database),
            "--run-id",
            FIXED_RUN_ID,
            "--output",
            str(output),
            "--human-baseline-evidence",
            str(human_evidence_path),
            "--goal-spec",
            str(goal_spec_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(completed.stdout)
    assert receipt["completion_proven"] is True
    assert receipt["verified_complete_count"] == 5
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["goal_contract"]["completion_proven"] is True
    assert report["declared_verified_complete_count"] == 5
    assert report["raw_unsettled_tool_rows"]
    assert report["effective_active_tool_rows"] == []
    assert report["prohibited_calls_since_goal"] == []

    connection = sqlite3.connect(database)
    connection.execute("INSERT INTO approvals(session_id,status) VALUES (?,?)", (FIXED_RUN_ID, "pending"))
    connection.commit()
    connection.close()
    pending_run = subprocess.run(
        [
            sys.executable,
            str(script),
            "--runtime-db",
            str(database),
            "--run-id",
            FIXED_RUN_ID,
            "--output",
            str(output),
            "--human-baseline-evidence",
            str(human_evidence_path),
            "--goal-spec",
            str(goal_spec_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert pending_run.returncode == 0
    assert json.loads(pending_run.stdout)["completion_proven"] is False


def test_negative_human_baseline_evidence_never_proves_completion(tmp_path: Path) -> None:
    evidence_path = Path("configs/g21_five_competition_human_baseline_gate.json")
    database = tmp_path / "runtime.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE assistant_runs (id TEXT PRIMARY KEY, status TEXT);
        CREATE TABLE approvals (session_id TEXT, status TEXT);
        CREATE TABLE deliverables (
            id TEXT, run_id TEXT, session_id TEXT, name TEXT, path TEXT,
            media_type TEXT, bytes INTEGER, sha256 TEXT, source_tool_call TEXT,
            preview_kind TEXT, verified_at TEXT
        );
        CREATE TABLE tool_calls (
            session_id TEXT, tool_name TEXT, status TEXT, created_at TEXT,
            completed_at TEXT, arguments_json TEXT, result_json TEXT
        );
        """
    )
    connection.execute("INSERT INTO assistant_runs(id,status) VALUES (?,?)", (FIXED_RUN_ID, "recovering"))
    connection.commit()
    connection.close()
    output = tmp_path / "audit.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "five_competition_goal_audit.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--runtime-db",
            str(database),
            "--run-id",
            FIXED_RUN_ID,
            "--output",
            str(output),
            "--human-baseline-evidence",
            str(evidence_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    receipt = json.loads(completed.stdout)
    assert receipt["completion_proven"] is False


def test_human_baseline_gate_evidence_is_explicitly_non_complete() -> None:
    evidence_path = Path("configs/g21_five_competition_human_baseline_gate.json")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["schema"] == "evomind.five_competition_human_baseline_audit.v1"
    assert evidence["run_id"] == FIXED_RUN_ID
    assert evidence["allocation"] == FIXED_ALLOCATION
    assert evidence["completion"]["human_baseline_gate_passed"] is False
    assert evidence["completion"]["all_five_verified"] is False
    assert set(evidence["competitions"]) == set(COMPETITIONS)
    assert all(
        item["status"] == "HUMAN_BASELINE_UNDEFINED"
        and item["unique_gate"]
        for item in evidence["competitions"].values()
    )


def test_human_baseline_audit_completion_flags_follow_classification() -> None:
    verified = {
        "schema": "evomind.five_competition_human_baseline_audit.v1",
        "run_id": FIXED_RUN_ID,
        "allocation": FIXED_ALLOCATION,
        "classification": "HUMAN_BASELINE_VERIFIED",
        "competitions": {
            name: {
                "status": "VERIFIED",
                "unique_gate": "source-backed",
                "baseline": _baseline(metric="score", direction="higher_is_better"),
            }
            for name in COMPETITIONS
        },
        "completion": {"human_baseline_gate_passed": True, "all_five_verified": True},
    }
    result = human_baseline_gate_summary(verified, run_id=FIXED_RUN_ID)
    assert result["valid"] is True
    assert result["completion_eligible"] is True

    mismatch = json.loads(json.dumps(verified))
    mismatch["completion"]["all_five_verified"] = False
    result = human_baseline_gate_summary(mismatch, run_id=FIXED_RUN_ID)
    assert result["valid"] is False
    assert result["completion_eligible"] is False
    assert any(
        issue["code"] == "HUMAN_BASELINE_AUDIT_COMPLETION_FLAGS_INVALID"
        for issue in result["issues"]
    )


def test_human_baseline_audit_accepts_partial_monotonic_evidence_without_goal_completion() -> None:
    partial = {
        "schema": "evomind.five_competition_human_baseline_audit.v1",
        "run_id": FIXED_RUN_ID,
        "allocation": FIXED_ALLOCATION,
        "classification": "HUMAN_BASELINE_PARTIAL",
        "competitions": {
            name: (
                {
                    "status": "VERIFIED",
                    "metric": "score",
                    "direction": "higher_is_better",
                    "baseline": _baseline(
                        competition=name,
                        metric="score",
                        direction="higher_is_better",
                    ),
                }
                if name == "cure_bench"
                else {"status": "HUMAN_BASELINE_UNDEFINED", "unique_gate": f"{name} pending"}
            )
            for name in COMPETITIONS
        },
        "completion": {"human_baseline_gate_passed": False, "all_five_verified": False},
    }
    result = human_baseline_gate_summary(partial, run_id=FIXED_RUN_ID)
    assert result["valid"] is True
    assert result["completion_eligible"] is False


def test_verified_human_baseline_requires_source_bound_details() -> None:
    minimal = {
        "schema": "evomind.five_competition_human_baseline_audit.v1",
        "run_id": FIXED_RUN_ID,
        "allocation": FIXED_ALLOCATION,
        "classification": "HUMAN_BASELINE_VERIFIED",
        "competitions": {
            name: {"status": "VERIFIED", "unique_gate": "source-backed"}
            for name in COMPETITIONS
        },
        "completion": {"human_baseline_gate_passed": True, "all_five_verified": True},
    }
    result = human_baseline_gate_summary(minimal, run_id=FIXED_RUN_ID)
    assert result["valid"] is False
    assert result["completion_eligible"] is False
    assert all(
        issue["code"] == "HUMAN_BASELINE_AUDIT_DETAIL_MISSING"
        for issue in result["issues"]
    )


def test_goal_audit_detects_wrong_key_or_unbound_fixed_id(tmp_path: Path) -> None:
    database = sqlite3.connect(tmp_path / "goal-records.sqlite3")
    database.executescript(
        """
        CREATE TABLE goal_records (
            id TEXT, run_id TEXT, goal_key TEXT, schema TEXT, allocation TEXT,
            spec_sha256 TEXT, spec_json TEXT, board_sha256 TEXT, human_baseline_sha256 TEXT,
            board_json TEXT, status TEXT, created_at TEXT, updated_at TEXT
        );
        """
    )
    database.execute(
        "INSERT INTO goal_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            FIXED_GOAL_ID,
            "run_other",
            "wrong_goal",
            GOAL_RECORD_SCHEMA,
            FIXED_ALLOCATION,
            "0" * 64,
            "{}",
            "0" * 64,
            "0" * 64,
            "{}",
            "active",
            "now",
            "now",
        ),
    )
    database.commit()
    database.row_factory = sqlite3.Row
    result = goal_audit.inspect_goal_records(database, run_id=FIXED_RUN_ID)
    database.close()
    assert result["valid"] is False
    assert result["extra_goal_records"]
    assert result["unbound_goal_records"]


def test_goal_audit_blocks_terms_and_join_tools() -> None:
    assert {"join_competition", "accept_terms"} <= goal_audit.PROHIBITED_TOOLS
