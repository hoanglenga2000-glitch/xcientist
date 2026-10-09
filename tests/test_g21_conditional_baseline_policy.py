from __future__ import annotations

from copy import deepcopy
import hashlib
import http.client
import json
from pathlib import Path
import threading

import pytest
from http.server import ThreadingHTTPServer

from evomind_runtime.competition_goal import (
    CONDITIONAL_BASELINE_MIGRATION_ID,
    CONDITIONAL_BASELINE_POLICY_MODE,
    FIXED_ALLOCATION,
    FIXED_GOAL_ID,
    FIXED_RUN_ID,
    INITIAL_GOAL_SPEC_CANONICAL_SHA256,
    INITIAL_GOAL_SPEC_FILE_SHA256,
    load_goal_spec,
    sha256_json,
    validate_goal_board,
    validate_goal_item,
    validate_goal_spec,
    validate_policy_migration_evidence,
    validate_primary_baseline,
)
from evomind_runtime.goal_board import ensure_fixed_goal
from evomind_runtime.http_server import ensure_token, make_handler
from evomind_runtime.models import Session, utc_now
from evomind_runtime.runtime import AgentRuntime
from evomind_runtime.store import RuntimeStore


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _board() -> dict:
    competitions = ("cure_bench", "e2lmc", "mindgames", "open_polymer", "ariel_2025")
    return {
        "schema": "evomind.goal-board.v2",
        "run_id": FIXED_RUN_ID,
        "allocation": FIXED_ALLOCATION,
        "target_count": 5,
        "weather_actions": 0,
        "competitions": [
            {"competition": competition, "goal_status": "WAITING_EXACT_GATE", "exact_gate": "pending"}
            for competition in competitions
        ],
    }


def _seed_store(tmp_path: Path) -> RuntimeStore:
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    session = Session(
        FIXED_RUN_ID,
        str(tmp_path),
        "workspace_write",
        title="fixed goal",
        objective="fixed goal",
    )
    store.create_session(session)
    now = utc_now()
    store.create_assistant_run(
        {
            "id": FIXED_RUN_ID,
            "session_id": FIXED_RUN_ID,
            "conversation_id": "conditional-policy-fixture",
            "prompt": "fixed goal",
            "task_root": str(tmp_path),
            "status": "blocked",
            "plan": {},
            "attachment_ids": [],
            "retry_count": 0,
            "error_class": "",
            "error_message": "",
            "model_provider": "",
            "model": "",
            "created_at": now,
            "updated_at": now,
            "completed_at": now,
        }
    )
    return store


def _policy_inputs() -> tuple[dict, dict]:
    return (
        load_goal_spec("configs/g21_five_competition_goal.json"),
        _load_json("configs/g21_conditional_strong_baseline_policy_evidence.json"),
    )


def _migrate(store: RuntimeStore) -> tuple[dict, bool]:
    spec, evidence = _policy_inputs()
    return store.migrate_goal_policy(
        FIXED_GOAL_ID,
        run_id=FIXED_RUN_ID,
        expected_spec_sha256=INITIAL_GOAL_SPEC_CANONICAL_SHA256,
        new_spec=spec,
        policy_evidence=evidence,
        migration_id=CONDITIONAL_BASELINE_MIGRATION_ID,
    )


def _primary_baseline(*, baseline_type: str = "official_organizer") -> dict:
    source_sha = hashlib.sha256(b"source-bound primary baseline\n").hexdigest()
    protocol = "The organizer scorer, split, aggregation unit, and metric are identical for both candidates."
    value = {
        "status": "VERIFIED",
        "baseline_type": baseline_type,
        "source_url": "https://organizer.example.edu/baselines/frozen-v1",
        "source_authority": "official_organizer",
        "source_sha256": source_sha,
        "evidence_artifact_id": "artifact-primary-baseline-source",
        "evidence_artifact_sha256": source_sha,
        "title": "Frozen organizer reference baseline",
        "protocol_id": "official-frozen-protocol-v1",
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(protocol.encode("utf-8")).hexdigest(),
        "protocol_comparable": True,
        "metric": "score",
        "direction": "higher_is_better",
        "value": 0.5,
        "official_organizer_baseline": baseline_type == "official_organizer",
    }
    if baseline_type == "publicly_reproducible_strong":
        value.update(
            {
                "publicly_reproducible": True,
                "implementation_sha256": "1" * 64,
                "data_manifest_sha256": "2" * 64,
                "environment_sha256": "3" * 64,
                "reproduction_receipt_sha256": "4" * 64,
            }
        )
    return value


def _conditional_complete_item(competition: str) -> dict:
    solution_id = f"{competition}-conditional-candidate-1"
    primary = _primary_baseline()
    availability = {
        "status": "NOT_VERIFIED_AVAILABLE",
        "audit_complete": True,
        "protocol_search_complete": True,
        "human_baseline_verified": False,
        "evidence_artifact_id": "artifact-human-negative-audit",
        "evidence_artifact_sha256": "5" * 64,
    }
    role_hashes = {
        "command": "a" * 64,
        "environment": "b" * 64,
        "data_manifest": "c" * 64,
        "source_code": "d" * 64,
        "model": "e" * 64,
        "metrics": "f" * 64,
        "rollback": "1" * 64,
        "holdout_ledger": "2" * 64,
        "holdout_claim": "3" * 64,
        "primary_baseline_source": primary["evidence_artifact_sha256"],
        "human_baseline_availability": availability["evidence_artifact_sha256"],
    }
    entries = [
        {
            "artifact_id": (
                primary["evidence_artifact_id"]
                if role == "primary_baseline_source"
                else availability["evidence_artifact_id"]
                if role == "human_baseline_availability"
                else f"artifact-{competition}-{role}"
            ),
            "path": f"{competition}/{role}.json",
            "role": role,
            "bytes": 1,
            "sha256": digest,
            "regular_file": True,
            "symlink": False,
        }
        for role, digest in role_hashes.items()
    ]
    manifest = {
        "schema": "evomind.artifact-manifest.v1",
        "run_id": FIXED_RUN_ID,
        "competition": competition,
        "solution_id": solution_id,
        "file_count": len(entries),
        "total_bytes": len(entries),
        "files": entries,
    }
    item = {
        "competition": competition,
        "solution_id": solution_id,
        "goal_status": "VERIFIED_COMPLETE",
        "official_metric": "score",
        "metric_direction": "higher_is_better",
        "primary_baseline": primary,
        "primary_comparison": {
            "candidate_metric": 0.8,
            "ci95_lower": 0.7,
            "ci95_upper": 0.9,
        },
        "human_baseline_availability": availability,
        "artifact_manifest": manifest,
        "artifact_manifest_artifact_id": f"artifact-{competition}-manifest",
        "artifact_manifest_sha256": sha256_json(manifest),
        "command_sha256": role_hashes["command"],
        "environment_sha256": role_hashes["environment"],
        "data_manifest_sha256": role_hashes["data_manifest"],
        "source_code_sha256": role_hashes["source_code"],
        "model_sha256": role_hashes["model"],
        "metrics_sha256": role_hashes["metrics"],
        "rollback_sha256": role_hashes["rollback"],
        "holdout_ledger": {
            "ledger_id": f"ledger-{competition}",
            "ledger_sha256": role_hashes["holdout_ledger"],
            "claim_id": f"claim-{competition}",
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
            "artifact_id": f"artifact-{competition}-review",
            "sha256": "6" * 64,
            "verified": True,
        },
    }
    if competition == "open_polymer":
        item["existing_candidate_adoption"] = {
            "schema": "evomind.existing-candidate-adoption.v1",
            "source_tool_call_id": "call-open-polymer-existing",
            "source_solution_id": solution_id,
            "source_artifact_manifest_sha256": item["artifact_manifest_sha256"],
            "retraining_performed": False,
        }
    else:
        item["managed_execution"] = {
            "schema": "evomind.managed-candidate-execution.v1",
            "training_route_tool_call_id": f"call-{competition}-route",
            "hpc_verify_tool_call_id": f"call-{competition}-verify",
            "hpc_execute_tool_call_id": f"call-{competition}-execute",
        }
    return item


def test_policy_files_bind_exact_old_spec_and_explicit_authorization() -> None:
    frozen = Path("configs/g21_five_competition_goal_v1_frozen.json")
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == INITIAL_GOAL_SPEC_FILE_SHA256
    old_spec = load_goal_spec(str(frozen))
    assert sha256_json(old_spec) == INITIAL_GOAL_SPEC_CANONICAL_SHA256
    spec, evidence = _policy_inputs()
    ok, issues = validate_goal_spec(spec)
    assert ok is True, issues
    assert spec["metric_gate"]["mode"] == CONDITIONAL_BASELINE_POLICY_MODE
    assert spec["previous_goal_spec_canonical_sha256"] == INITIAL_GOAL_SPEC_CANONICAL_SHA256
    ok, issues = validate_policy_migration_evidence(evidence, new_spec=spec)
    assert ok is True, issues


def test_primary_baseline_contract_rejects_unreproducible_or_human_disguise() -> None:
    assert validate_primary_baseline(
        _primary_baseline(), expected_metric="score", expected_direction="higher_is_better"
    ).valid
    strong = _primary_baseline(baseline_type="publicly_reproducible_strong")
    strong.pop("reproduction_receipt_sha256")
    result = validate_primary_baseline(
        strong, expected_metric="score", expected_direction="higher_is_better"
    )
    assert result.valid is False
    assert any("REPRODUCTION_RECEIPT" in issue.code for issue in result.issues)
    disguised = _primary_baseline()
    disguised["participant_type"] = "human_participants"
    assert any(
        issue.code == "HUMAN_BASELINE_MUST_USE_HUMAN_CONTRACT"
        for issue in validate_primary_baseline(disguised).issues
    )


def test_conditional_human_gate_never_accepts_fake_human_or_test_leakage() -> None:
    item = {
        "competition": "cure_bench",
        "solution_id": "cure-candidate-conditional",
        "goal_status": "VERIFIED_COMPLETE",
        "official_metric": "score",
        "metric_direction": "higher_is_better",
        "primary_baseline": _primary_baseline(),
        "primary_comparison": {"candidate_metric": 0.8, "ci95_lower": 0.7, "ci95_upper": 0.9},
        "human_baseline_availability": {
            "status": "NOT_VERIFIED_AVAILABLE",
            "audit_complete": True,
            "protocol_search_complete": True,
            "human_baseline_verified": False,
            "evidence_artifact_id": "artifact-human-negative-audit",
            "evidence_artifact_sha256": "5" * 64,
        },
        "leakage_audit": {
            "test_labels_used": True,
            "hidden_test_labels_used": False,
            "holdout_overlap": 0,
            "id_overlap": 0,
            "architecture_overlap": 0,
            "point_overlap": 0,
            "manifest_closed": True,
            "loader_smoke": True,
        },
    }
    checked = validate_goal_item("cure_bench", item, conditional_policy=True)
    codes = {issue.code for issue in checked.issues}
    assert checked.human_baseline_required is False
    assert "HUMAN_BASELINE_MISSING" not in codes
    assert "TEST_LABEL_USAGE_NOT_FALSE" in codes
    fake = deepcopy(item)
    fake["human_baseline"] = {"status": "VERIFIED", "participant_type": "model"}
    checked = validate_goal_item("cure_bench", fake, conditional_policy=True)
    codes = {issue.code for issue in checked.issues}
    assert checked.human_baseline_required is True
    assert "HUMAN_BASELINE_AVAILABILITY_CONFLICT" in codes
    assert "HUMAN_PARTICIPANTS_NOT_PROVEN" in codes


def test_conditional_board_completes_without_human_only_with_source_bound_primary_gate() -> None:
    spec, _ = _policy_inputs()
    board = {
        "schema": "evomind.goal-board.v2",
        "run_id": FIXED_RUN_ID,
        "allocation": FIXED_ALLOCATION,
        "target_count": 5,
        "weather_actions": 0,
        "baseline_policy": CONDITIONAL_BASELINE_POLICY_MODE,
        "policy_evidence_sha256": spec["policy_evidence_sha256"],
        "human_baseline_evidence_artifact_id": "artifact-human-negative-audit",
        "human_baseline_evidence_sha256": "5" * 64,
        "competitions": [
            _conditional_complete_item(competition)
            for competition in ("cure_bench", "e2lmc", "mindgames", "open_polymer", "ariel_2025")
        ],
    }
    result = validate_goal_board(board, spec=spec)
    assert result.completion_proven is True, [issue.to_dict() for issue in result.issues]
    requires_human = deepcopy(board)
    requires_human["competitions"][0]["human_baseline_availability"]["status"] = "VERIFIED_AVAILABLE"
    result = validate_goal_board(requires_human, spec=spec)
    assert result.completion_proven is False
    assert any(
        issue.code == "cure_bench:HUMAN_BASELINE_REQUIRED_WHEN_VERIFIED_AVAILABLE"
        for issue in result.issues
    )


def test_atomic_policy_migration_and_exact_idempotent_replay(tmp_path: Path) -> None:
    store = _seed_store(tmp_path)
    try:
        old_spec = load_goal_spec("configs/g21_five_competition_goal_v1_frozen.json")
        created = ensure_fixed_goal(store, run_id=FIXED_RUN_ID, spec=old_spec, board=_board())
        assert created["record"]["spec_sha256"] == INITIAL_GOAL_SPEC_CANONICAL_SHA256
        migrated, changed = _migrate(store)
        spec, evidence = _policy_inputs()
        assert changed is True
        assert migrated["spec_sha256"] == sha256_json(spec)
        assert migrated["board"] == _board()
        metadata = store.get_session(FIXED_RUN_ID)["metadata"]
        assert metadata["goal_spec_sha256"] == sha256_json(spec)
        assert metadata["goal_previous_spec_sha256"] == INITIAL_GOAL_SPEC_CANONICAL_SHA256
        assert metadata["goal_policy_evidence_sha256"] == sha256_json(evidence)
        events = [
            event for event in store.list_events(FIXED_RUN_ID)
            if event["event_type"] == "goal.policy_migrated"
        ]
        assert len(events) == 1
        assert events[0]["payload"]["old_spec_sha256"] == INITIAL_GOAL_SPEC_CANONICAL_SHA256
        replayed, changed = _migrate(store)
        assert changed is False
        assert replayed == migrated
        assert len([
            event for event in store.list_events(FIXED_RUN_ID)
            if event["event_type"] == "goal.policy_migrated"
        ]) == 1
        drifted_metadata = dict(store.get_session(FIXED_RUN_ID)["metadata"])
        drifted_metadata["goal_policy_evidence_sha256"] = "0" * 64
        store.update_session(FIXED_RUN_ID, metadata_json=drifted_metadata)
        with pytest.raises(ValueError, match="replay metadata invalid"):
            _migrate(store)
    finally:
        store.close()


def test_policy_migration_wrong_precondition_cross_run_and_metadata_conflict_roll_back(tmp_path: Path) -> None:
    store = _seed_store(tmp_path)
    try:
        old_spec = load_goal_spec("configs/g21_five_competition_goal_v1_frozen.json")
        ensure_fixed_goal(store, run_id=FIXED_RUN_ID, spec=old_spec, board=_board())
        spec, evidence = _policy_inputs()
        with pytest.raises(ValueError, match="old spec precondition"):
            store.migrate_goal_policy(
                FIXED_GOAL_ID,
                run_id=FIXED_RUN_ID,
                expected_spec_sha256="0" * 64,
                new_spec=spec,
                policy_evidence=evidence,
                migration_id=CONDITIONAL_BASELINE_MIGRATION_ID,
            )
        with pytest.raises(ValueError, match="identity mismatch"):
            store.migrate_goal_policy(
                FIXED_GOAL_ID,
                run_id="run_other",
                expected_spec_sha256=INITIAL_GOAL_SPEC_CANONICAL_SHA256,
                new_spec=spec,
                policy_evidence=evidence,
                migration_id=CONDITIONAL_BASELINE_MIGRATION_ID,
            )
        session = store.get_session(FIXED_RUN_ID)
        metadata = dict(session["metadata"])
        metadata["goal_policy_evidence_sha256"] = "0" * 64
        store.update_session(FIXED_RUN_ID, metadata_json=metadata)
        with pytest.raises(ValueError, match="metadata conflicts"):
            _migrate(store)
        unchanged = store.get_goal_record(FIXED_GOAL_ID)
        assert unchanged["spec_sha256"] == INITIAL_GOAL_SPEC_CANONICAL_SHA256
        assert not any(
            event["event_type"] == "goal.policy_migrated"
            for event in store.list_events(FIXED_RUN_ID)
        )
    finally:
        store.close()


def _request(port: int, token: str, body: dict) -> tuple[int, dict]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    raw = json.dumps(body).encode("utf-8")
    connection.request(
        "PUT",
        f"/v1/goals/{FIXED_GOAL_ID}",
        body=raw,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Content-Length": str(len(raw)),
        },
    )
    response = connection.getresponse()
    payload = json.loads(response.read().decode("utf-8"))
    connection.close()
    return response.status, payload


def test_existing_goal_put_exposes_only_isolated_policy_migration(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    now = utc_now()
    runtime.create_session(session_id=FIXED_RUN_ID, workspace_root=str(tmp_path), objective="fixed goal")
    runtime.store.create_assistant_run(
        {
            "id": FIXED_RUN_ID,
            "session_id": FIXED_RUN_ID,
            "conversation_id": "conditional-policy-http",
            "prompt": "fixed goal",
            "task_root": str(tmp_path),
            "status": "blocked",
            "plan": {},
            "attachment_ids": [],
            "retry_count": 0,
            "error_class": "",
            "error_message": "",
            "model_provider": "",
            "model": "",
            "created_at": now,
            "updated_at": now,
            "completed_at": now,
        }
    )
    old_spec = load_goal_spec("configs/g21_five_competition_goal_v1_frozen.json")
    runtime.ensure_fixed_goal(run_id=FIXED_RUN_ID, spec=old_spec, board=_board())
    token = ensure_token(runtime.runtime_root)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runtime, token))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    spec, evidence = _policy_inputs()
    migration = {
        "policy_migration": {
            "run_id": FIXED_RUN_ID,
            "expected_spec_sha256": INITIAL_GOAL_SPEC_CANONICAL_SHA256,
            "new_spec": spec,
            "policy_evidence": evidence,
            "migration_id": CONDITIONAL_BASELINE_MIGRATION_ID,
        }
    }
    try:
        status, result = _request(int(server.server_address[1]), token, migration)
        assert status == 200
        assert result["migrated"] is True
        status, replay = _request(int(server.server_address[1]), token, migration)
        assert status == 200
        assert replay["migrated"] is False
        status, rejected = _request(
            int(server.server_address[1]), token, {**migration, "status": "active"}
        )
        assert status == 400
        assert rejected["error"] == "goal_policy_migration_must_be_isolated"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        runtime.close()
