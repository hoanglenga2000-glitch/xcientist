from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from evomind_runtime.competition_goal import (
    COMPETITIONS,
    FIXED_ALLOCATION,
    FIXED_GOAL_ID,
    FIXED_GOAL_KEY,
    FIXED_RUN_ID,
    GOAL_RECORD_SCHEMA,
    canonical_json_bytes,
    load_goal_spec,
    sha256_json,
    validate_goal_record,
)
from evomind_runtime.goal_board import GoalRecordError, ensure_fixed_goal, update_fixed_goal_board
from evomind_runtime.models import Session, ToolCall, ToolResult, utc_now
from evomind_runtime.store import RuntimeStore


def _board() -> dict:
    return {
        "schema": "evomind.goal-board.v2",
        "run_id": FIXED_RUN_ID,
        "allocation": FIXED_ALLOCATION,
        "target_count": len(COMPETITIONS),
        "weather_actions": 0,
        "competitions": [
            {"competition": name, "goal_status": "WAITING_EXACT_GATE", "exact_gate": "pending"}
            for name in COMPETITIONS
        ],
    }


def _seed_store(tmp_path: Path) -> RuntimeStore:
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    now = utc_now()
    store.create_session(
        Session(
            id=FIXED_RUN_ID,
            workspace_root=str(tmp_path),
            title="fixed goal fixture",
            objective="fixed G21 goal fixture",
            created_at=now,
            updated_at=now,
        )
    )
    store.create_assistant_run(
        {
            "id": FIXED_RUN_ID,
            "session_id": FIXED_RUN_ID,
            "conversation_id": "fixed_goal_fixture",
            "prompt": "fixed G21 goal fixture",
            "task_root": str(tmp_path),
            "status": "recovering",
            "plan": {},
            "attachment_ids": [],
            "retry_count": 0,
            "error_class": "",
            "error_message": "",
            "model_provider": "",
            "model": "",
            "created_at": now,
            "updated_at": now,
            "completed_at": "",
        }
    )
    return store


def _publish(store: RuntimeStore, tmp_path: Path, artifact_id: str, name: str, raw: bytes) -> dict:
    path = tmp_path / "published" / artifact_id / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return store.publish_deliverable(
        {
            "id": artifact_id,
            "run_id": FIXED_RUN_ID,
            "session_id": FIXED_RUN_ID,
            "name": name,
            "path": str(path),
            "media_type": "application/json",
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "source_tool_call": "independent_verify",
            "preview_kind": "structured_text",
            "verified_at": utc_now(),
        }
    )


def _partially_completed_board(store: RuntimeStore, tmp_path: Path) -> dict:
    competition = "cure_bench"
    solution_id = "cure-bench-candidate-evidence-1"
    protocol = "Participants used the official frozen CURE task, aggregation, scoring, and holdout protocol."
    role_payloads: dict[str, bytes] = {
        role: f"{competition}:{role}:evidence\n".encode("utf-8")
        for role in (
            "command", "environment", "data_manifest", "source_code", "model", "metrics", "rollback",
            "human_baseline_source",
        )
    }
    ledger_id = "ledger-cure-bench-v1"
    claim_id = "claim-cure-bench-candidate-1"
    role_payloads["holdout_ledger"] = canonical_json_bytes(
        {
            "schema": "evomind.holdout-ledger.v1",
            "run_id": FIXED_RUN_ID,
            "competition": competition,
            "ledger_id": ledger_id,
        }
    )
    role_payloads["holdout_claim"] = canonical_json_bytes(
        {
            "schema": "evomind.holdout-claim.v1",
            "run_id": FIXED_RUN_ID,
            "competition": competition,
            "solution_id": solution_id,
            "ledger_id": ledger_id,
            "claim_id": claim_id,
            "unconsumed_at_claim": True,
            "consumed_once": True,
            "selection_data_scope": "development_only",
        }
    )
    entries = []
    by_role: dict[str, dict] = {}
    for role, raw in role_payloads.items():
        artifact_id = f"artifact-{competition}-{role}".replace("_", "-")
        name = f"{role}.json"
        row = _publish(store, tmp_path, artifact_id, name, raw)
        entry = {
            "artifact_id": artifact_id,
            "path": f"{competition}/{name}",
            "role": role,
            "bytes": row["bytes"],
            "sha256": row["sha256"],
            "regular_file": True,
            "symlink": False,
        }
        entries.append(entry)
        by_role[role] = entry
    baseline = {
        "status": "VERIFIED",
        "participant_type": "human_participants",
        "source_url": "https://doi.org/10.1234/cure-human-study",
        "source_doi": "10.1234/cure-human-study",
        "source_authority": "peer_reviewed",
        "source_sha256": by_role["human_baseline_source"]["sha256"],
        "evidence_artifact_id": by_role["human_baseline_source"]["artifact_id"],
        "evidence_artifact_sha256": by_role["human_baseline_source"]["sha256"],
        "title": "CURE human participant study",
        "sample_size": 24,
        "protocol_id": "cure-bench-official-human-v1",
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(protocol.encode("utf-8")).hexdigest(),
        "protocol_comparable": True,
        "metric": "accuracy",
        "direction": "higher_is_better",
        "mean": 0.5,
        "uncertainty": {
            "lower": 0.45,
            "upper": 0.55,
            "confidence_level": 0.95,
            "method": "participant bootstrap",
            "source_sha256": by_role["human_baseline_source"]["sha256"],
        },
    }
    manifest = {
        "schema": "evomind.artifact-manifest.v1",
        "run_id": FIXED_RUN_ID,
        "competition": competition,
        "solution_id": solution_id,
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
    }
    manifest_raw = canonical_json_bytes(manifest)
    manifest_row = _publish(
        store,
        tmp_path,
        "artifact-cure-bench-artifact-manifest",
        "artifact-manifest.json",
        manifest_raw,
    )
    review_raw = canonical_json_bytes(
        {
            "schema": "evomind.independent-review.v1",
            "run_id": FIXED_RUN_ID,
            "competition": competition,
            "solution_id": solution_id,
            "artifact_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "verified": True,
            "test_labels_used": False,
            "manifest_closed": True,
        }
    )
    review_row = _publish(
        store,
        tmp_path,
        "artifact-cure-bench-independent-review",
        "independent-review.json",
        review_raw,
    )
    execution_ids = {
        "training_route_tool_call_id": "call-cure-bench-route",
        "hpc_verify_tool_call_id": "call-cure-bench-verify",
        "hpc_execute_tool_call_id": "call-cure-bench-execute",
    }
    goal_created = next(
        event for event in store.list_events(FIXED_RUN_ID)
        if event["event_type"] == "goal.created"
    )
    base_time = datetime.fromisoformat(str(goal_created["created_at"]).replace("Z", "+00:00"))
    calls = [
        ToolCall(
            execution_ids["training_route_tool_call_id"], FIXED_RUN_ID, "training_route",
            {"competition": competition, "task_description": "CURE managed route"}, "completed",
            (base_time + timedelta(seconds=1)).isoformat(), "", (base_time + timedelta(seconds=2)).isoformat(),
            "", "route-cure-1",
        ),
        ToolCall(
            execution_ids["hpc_verify_tool_call_id"], FIXED_RUN_ID, "hpc_verify", {}, "completed",
            (base_time + timedelta(seconds=3)).isoformat(), "", (base_time + timedelta(seconds=4)).isoformat(),
            "", "verify-cure-1",
        ),
        ToolCall(
            execution_ids["hpc_execute_tool_call_id"], FIXED_RUN_ID, "hpc_execute_solution",
            {"competition": competition, "solution_id": solution_id}, "completed",
            (base_time + timedelta(seconds=5)).isoformat(), "", (base_time + timedelta(seconds=6)).isoformat(),
            "", "execute-cure-1",
        ),
    ]
    results = [
        ToolResult(calls[0].id, True, {"route": "hpc"}, "routed"),
        ToolResult(
            calls[1].id,
            True,
            {
                "samples_requested": 5,
                "samples_passed": 5,
                "samples": [{"complete": True} for _ in range(5)],
            },
            "verified",
        ),
        ToolResult(
            calls[2].id,
            True,
            {
                "artifact_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
                "published_artifacts": [{"id": manifest_row["id"], "sha256": hashlib.sha256(manifest_raw).hexdigest()}],
            },
            "completed",
        ),
    ]
    for call, result in zip(calls, results):
        store.put_tool_call(call, result)
    complete_item = {
        "competition": competition,
        "solution_id": solution_id,
        "goal_status": "VERIFIED_COMPLETE",
        "official_metric": "accuracy",
        "metric_direction": "higher_is_better",
        "human_baseline": baseline,
        "human_comparison": {"candidate_metric": 0.8, "ci95_lower": 0.7, "ci95_upper": 0.9},
        "artifact_manifest": manifest,
        "artifact_manifest_artifact_id": manifest_row["id"],
        "artifact_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        **{f"{role}_sha256": by_role[role]["sha256"] for role in (
            "command", "environment", "data_manifest", "source_code", "model", "metrics", "rollback"
        )},
        "holdout_ledger": {
            "ledger_id": ledger_id,
            "ledger_sha256": by_role["holdout_ledger"]["sha256"],
            "claim_id": claim_id,
            "claim_sha256": by_role["holdout_claim"]["sha256"],
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
        "managed_execution": {"schema": "evomind.managed-candidate-execution.v1", **execution_ids},
        "independent_review": {
            "schema": "evomind.independent-review.v1",
            "artifact_id": review_row["id"],
            "sha256": review_row["sha256"],
            "verified": True,
        },
    }
    human_evidence = {
        "schema": "evomind.five_competition_human_baseline_audit.v1",
        "run_id": FIXED_RUN_ID,
        "allocation": FIXED_ALLOCATION,
        "classification": "HUMAN_BASELINE_PARTIAL",
        "competitions": {
            name: (
                {"status": "VERIFIED", "baseline": baseline}
                if name == competition
                else {"status": "HUMAN_BASELINE_UNDEFINED", "unique_gate": f"{name} pending"}
            )
            for name in COMPETITIONS
        },
        "completion": {"human_baseline_gate_passed": False, "all_five_verified": False},
    }
    human_raw = canonical_json_bytes(human_evidence)
    human_row = _publish(
        store,
        tmp_path,
        "artifact-human-baseline-partial",
        "human-baseline.json",
        human_raw,
    )
    board = _board()
    board["human_baseline_evidence_artifact_id"] = human_row["id"]
    board["human_baseline_evidence_sha256"] = human_row["sha256"]
    board["competitions"][0] = complete_item
    return board


def test_fixed_goal_create_or_get_is_idempotent(tmp_path: Path) -> None:
    store = _seed_store(tmp_path)
    try:
        spec = load_goal_spec("configs/g21_five_competition_goal.json")
        board = _board()
        first = ensure_fixed_goal(store, run_id=FIXED_RUN_ID, spec=spec, board=board)
        second = ensure_fixed_goal(store, run_id=FIXED_RUN_ID, spec=spec, board=board)

        assert first["created"] is True
        assert second["created"] is False
        assert first["record"]["id"] == FIXED_GOAL_ID
        assert second["record"] == first["record"]
        assert store.list_goal_records(FIXED_RUN_ID) == [first["record"]]
        store.close()
        store = RuntimeStore(tmp_path / "runtime.sqlite3")
        assert store.get_goal_record(FIXED_GOAL_ID) == first["record"]
    finally:
        store.close()


def test_partial_human_evidence_update_is_actual_atomic_idempotent_and_monotonic(tmp_path: Path) -> None:
    store = _seed_store(tmp_path)
    try:
        spec = load_goal_spec("configs/g21_five_competition_goal_v1_frozen.json")
        ensure_fixed_goal(store, run_id=FIXED_RUN_ID, spec=spec, board=_board())
        board = _partially_completed_board(store, tmp_path)

        updated = update_fixed_goal_board(
            store,
            goal_id=FIXED_GOAL_ID,
            board=board,
            status="blocked",
        )
        assert updated["human_baseline_sha256"] == board["human_baseline_evidence_sha256"]
        metadata = store.get_session(FIXED_RUN_ID)["metadata"]
        assert metadata["goal_human_baseline_sha256"] == board["human_baseline_evidence_sha256"]
        assert metadata["goal_board_sha256"] == sha256_json(board)
        assert [
            event["event_type"]
            for event in store.list_events(FIXED_RUN_ID)
            if event["event_type"] == "goal.updated"
        ] == ["goal.updated"]

        replayed = update_fixed_goal_board(
            store,
            goal_id=FIXED_GOAL_ID,
            board=board,
            status="blocked",
        )
        assert replayed == updated
        assert [
            event["event_type"]
            for event in store.list_events(FIXED_RUN_ID)
            if event["event_type"] == "goal.updated"
        ] == ["goal.updated"]

        with pytest.raises(GoalRecordError, match="immutable"):
            update_fixed_goal_board(
                store,
                goal_id=FIXED_GOAL_ID,
                board=_board(),
                status="blocked",
            )
        with pytest.raises(GoalRecordError, match="all five"):
            update_fixed_goal_board(
                store,
                goal_id=FIXED_GOAL_ID,
                board=board,
                status="completed",
            )
    finally:
        store.close()


def test_completed_item_rejects_changed_published_evidence_file(tmp_path: Path) -> None:
    store = _seed_store(tmp_path)
    try:
        spec = load_goal_spec("configs/g21_five_competition_goal_v1_frozen.json")
        ensure_fixed_goal(store, run_id=FIXED_RUN_ID, spec=spec, board=_board())
        board = _partially_completed_board(store, tmp_path)
        source_id = board["competitions"][0]["human_baseline"]["evidence_artifact_id"]
        source = store.get_deliverable(source_id)
        assert source is not None
        Path(source["path"]).write_bytes(b"tampered after publication")

        with pytest.raises(GoalRecordError, match="current file mismatch"):
            update_fixed_goal_board(
                store,
                goal_id=FIXED_GOAL_ID,
                board=board,
                status="blocked",
            )
    finally:
        store.close()


def test_fixed_goal_requires_existing_fixed_run(tmp_path: Path) -> None:
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    try:
        spec = load_goal_spec("configs/g21_five_competition_goal.json")
        with pytest.raises(GoalRecordError, match="fixed G21 Run"):
            ensure_fixed_goal(store, run_id="run_other", spec=spec, board=_board())
        with pytest.raises(KeyError):
            ensure_fixed_goal(store, run_id=FIXED_RUN_ID, spec=spec, board=_board())
    finally:
        store.close()


def test_fixed_goal_rejects_spec_and_board_drift(tmp_path: Path) -> None:
    store = _seed_store(tmp_path)
    try:
        spec = load_goal_spec("configs/g21_five_competition_goal.json")
        board = _board()
        ensure_fixed_goal(store, run_id=FIXED_RUN_ID, spec=spec, board=board)

        changed_spec = deepcopy(spec)
        changed_spec["metric_gate"] = {**changed_spec["metric_gate"], "human_baseline_required": False}
        with pytest.raises(GoalRecordError):
            ensure_fixed_goal(store, run_id=FIXED_RUN_ID, spec=changed_spec, board=board)

        changed_board = deepcopy(board)
        changed_board["run_id"] = "run_other"
        with pytest.raises(GoalRecordError):
            ensure_fixed_goal(store, run_id=FIXED_RUN_ID, spec=spec, board=changed_board)
    finally:
        store.close()


def test_goal_key_cannot_be_rebound_to_another_run(tmp_path: Path) -> None:
    store = _seed_store(tmp_path)
    now = utc_now()
    second_run = "run_second_fixture"
    store.create_session(Session(id=second_run, workspace_root=str(tmp_path), created_at=now, updated_at=now))
    store.create_assistant_run(
        {
            "id": second_run,
            "session_id": second_run,
            "conversation_id": "second_fixture",
            "prompt": "second",
            "task_root": str(tmp_path),
            "status": "recovering",
            "plan": {},
            "attachment_ids": [],
            "retry_count": 0,
            "error_class": "",
            "error_message": "",
            "model_provider": "",
            "model": "",
            "created_at": now,
            "updated_at": now,
            "completed_at": "",
        }
    )
    try:
        spec = load_goal_spec("configs/g21_five_competition_goal.json")
        board = _board()
        ensure_fixed_goal(store, run_id=FIXED_RUN_ID, spec=spec, board=board)
        payload = {
            "id": "goal_other",
            "run_id": second_run,
            "goal_key": FIXED_GOAL_KEY,
            "schema": GOAL_RECORD_SCHEMA,
            "allocation": FIXED_ALLOCATION,
            "spec_sha256": sha256_json(spec),
            "spec_json": json.dumps(spec, sort_keys=True, separators=(",", ":")),
            "board_sha256": sha256_json(board),
            "human_baseline_sha256": str(spec["human_baseline_evidence_sha256"]),
            "board_json": json.dumps(board, sort_keys=True, separators=(",", ":")),
            "status": "active",
            "created_at": now,
            "updated_at": now,
        }
        with pytest.raises(ValueError, match="goal identity conflict"):
            store.ensure_goal_record(payload)
    finally:
        store.close()


def test_default_goal_projection_ignores_non_fixed_goal_keys(tmp_path: Path) -> None:
    store = _seed_store(tmp_path)
    try:
        now = utc_now()
        store._connection.execute(
            "INSERT INTO goal_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "goal_extra",
                FIXED_RUN_ID,
                "other_goal",
                GOAL_RECORD_SCHEMA,
                FIXED_ALLOCATION,
                "0" * 64,
                "{}",
                "0" * 64,
                "0" * 64,
                "{}",
                "active",
                now,
                now,
            ),
        )
        store._connection.commit()
        assert store.get_goal_for_run(FIXED_RUN_ID) is None
        assert store.get_goal_for_run(FIXED_RUN_ID, "other_goal")["id"] == "goal_extra"
    finally:
        store.close()


def test_update_fixed_goal_board_rehashes_and_remains_valid(tmp_path: Path) -> None:
    store = _seed_store(tmp_path)
    try:
        spec = load_goal_spec("configs/g21_five_competition_goal.json")
        board = _board()
        created = ensure_fixed_goal(store, run_id=FIXED_RUN_ID, spec=spec, board=board)
        updated_board = deepcopy(board)
        updated_board["competitions"][0]["exact_gate"] = "PRETRAINED_ENCODER_MANIFEST_MISSING"
        updated = update_fixed_goal_board(store, goal_id=FIXED_GOAL_ID, board=updated_board, status="blocked")
        assert updated["status"] == "blocked"
        assert updated["board_sha256"] == sha256_json(updated_board)
        assert updated["board_sha256"] != created["record"]["board_sha256"]
        checked = validate_goal_record(updated)
        assert checked.valid is True
        assert [
            event["event_type"]
            for event in store.list_events(FIXED_RUN_ID)
            if event["event_type"] in {"goal.created", "goal.updated"}
        ] == ["goal.created", "goal.updated"]
    finally:
        store.close()


def test_store_rejects_direct_invalid_goal_update(tmp_path: Path) -> None:
    store = _seed_store(tmp_path)
    try:
        spec = load_goal_spec("configs/g21_five_competition_goal.json")
        ensure_fixed_goal(store, run_id=FIXED_RUN_ID, spec=spec, board=_board())
        with pytest.raises(ValueError, match="board hash mismatch"):
            store.update_goal_record(FIXED_GOAL_ID, board=_board(), board_sha256="0" * 64)
        with pytest.raises(ValueError, match="goal record invalid"):
            store.update_goal_record(FIXED_GOAL_ID, status="WAITING_EXTERNAL_GATE")
    finally:
        store.close()


def test_completed_goal_requires_strict_board_evidence(tmp_path: Path) -> None:
    store = _seed_store(tmp_path)
    try:
        spec = load_goal_spec("configs/g21_five_competition_goal.json")
        with pytest.raises(GoalRecordError, match="bootstrapped in blocked status"):
            ensure_fixed_goal(
                store,
                run_id=FIXED_RUN_ID,
                spec=spec,
                board=_board(),
                status="completed",
            )
        assert store.get_goal_record(FIXED_GOAL_ID) is None
    finally:
        store.close()


def test_legacy_goal_table_is_fail_closed(tmp_path: Path) -> None:
    database = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE goal_records (id TEXT, run_id TEXT, goal_key TEXT, schema TEXT, allocation TEXT, "
        "spec_sha256 TEXT, spec_json TEXT, board_sha256 TEXT, board_json TEXT, status TEXT, created_at TEXT, updated_at TEXT)"
    )
    connection.commit()
    connection.close()
    store = RuntimeStore(database)
    try:
        issues = store.goal_records_schema_issues()
        assert "GOAL_RECORD_ID_NOT_PRIMARY_KEY" in issues
        assert "GOAL_RECORD_REQUIRED_COLUMNS_NULLABLE" in issues
        assert "GOAL_RECORD_GOAL_KEY_NOT_UNIQUE" in issues
        assert "GOAL_RECORD_RUN_FOREIGN_KEY_MISSING" in issues
        payload = {
            "id": FIXED_GOAL_ID,
            "run_id": FIXED_RUN_ID,
            "goal_key": FIXED_GOAL_KEY,
            "schema": GOAL_RECORD_SCHEMA,
            "allocation": FIXED_ALLOCATION,
            "spec_sha256": "0" * 64,
            "human_baseline_sha256": "0" * 64,
            "spec_json": "{}",
            "board_sha256": "0" * 64,
            "board_json": "{}",
            "status": "active",
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        with pytest.raises(ValueError, match="schema invalid"):
            store.ensure_goal_record(payload)
    finally:
        store.close()
