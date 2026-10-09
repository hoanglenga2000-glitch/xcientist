from __future__ import annotations

import http.client
import json
from pathlib import Path
import threading

import pytest
from http.server import ThreadingHTTPServer

from evomind_runtime.competition_goal import (
    COMPETITIONS,
    FIXED_ALLOCATION,
    FIXED_RUN_ID,
    load_goal_spec,
)
from evomind_runtime.http_server import ensure_token, make_handler
from evomind_runtime.models import Session, utc_now
from evomind_runtime.runtime import AgentRuntime


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


def _seed_fixed_run(runtime: AgentRuntime, root: Path) -> None:
    now = utc_now()
    runtime.create_session(
        session_id=FIXED_RUN_ID,
        workspace_root=str(root),
        objective="fixed G21 goal fixture",
    )
    runtime.store.create_assistant_run(
        {
            "id": FIXED_RUN_ID,
            "session_id": FIXED_RUN_ID,
            "conversation_id": "fixed_goal_http",
            "prompt": "fixed G21 goal fixture",
            "task_root": str(root),
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


def _request(port: int, token: str, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    raw = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Authorization": f"Bearer {token}"}
    if raw is not None:
        headers.update({"Content-Type": "application/json", "Content-Length": str(len(raw))})
    connection.request(method, path, body=raw, headers=headers)
    response = connection.getresponse()
    payload = json.loads(response.read().decode("utf-8"))
    connection.close()
    return response.status, payload


def test_fixed_goal_http_create_get_replay_and_cross_run_guard(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    _seed_fixed_run(runtime, tmp_path)
    token = ensure_token(runtime.runtime_root)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runtime, token))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    payload = {
        "run_id": FIXED_RUN_ID,
        "spec": load_goal_spec("configs/g21_five_competition_goal.json"),
        "board": _board(),
    }
    try:
        status, created = _request(port, token, "POST", "/v1/goals", payload)
        assert status == 201
        assert created["created"] is True
        assert created["record"]["run_id"] == FIXED_RUN_ID
        assert created["record"]["status"] == "blocked"

        status, replayed = _request(port, token, "POST", "/v1/goals", payload)
        assert status == 200
        assert replayed["created"] is False
        assert replayed["record"] == created["record"]

        status, fetched = _request(port, token, "GET", "/v1/goals/goal_g21_five_competition")
        assert status == 200
        assert fetched == created["record"]
        status, listed = _request(port, token, "GET", "/v1/goals")
        assert status == 200
        assert listed["goals"][0]["id"] == created["record"]["id"]
        status, run_goal = _request(port, token, "GET", f"/v1/runs/{FIXED_RUN_ID}/goal")
        assert status == 200
        assert run_goal["id"] == created["record"]["id"]
        assert runtime.assistant.snapshot(FIXED_RUN_ID)["goal"]["id"] == created["record"]["id"]
        public_event_types = [item["event_type"] for item in runtime.assistant.events(FIXED_RUN_ID)]
        assert "goal_created" in public_event_types

        updated_board = _board()
        updated_board["competitions"][0]["exact_gate"] = "PRETRAINED_ENCODER_MANIFEST_MISSING"
        status, updated = _request(
            port,
            token,
            "PUT",
            "/v1/goals/goal_g21_five_competition",
            {"board": updated_board, "status": "blocked"},
        )
        assert status == 200
        assert updated["status"] == "blocked"
        assert updated["board"]["competitions"][0]["exact_gate"] == "PRETRAINED_ENCODER_MANIFEST_MISSING"
        assert "goal_updated" in [item["event_type"] for item in runtime.assistant.events(FIXED_RUN_ID)]

        status, hidden = _request(port, token, "GET", "/v1/runs/run_other/goal")
        assert status == 404
        assert hidden["error"] == "not_found"

        status, rejected = _request(
            port,
            token,
            "POST",
            "/v1/goals",
            {**payload, "run_id": "run_other"},
        )
        assert status == 400
        assert rejected["error"] == "invalid_request"
        assert len(runtime.store.list_goal_records()) == 1
        assert [item["event_type"] for item in runtime.store.list_events(FIXED_RUN_ID) if item["event_type"] == "goal.created"] == ["goal.created"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        runtime.close()


def test_fixed_goal_http_put_cannot_complete_with_self_reported_fake_evidence(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    _seed_fixed_run(runtime, tmp_path)
    token = ensure_token(runtime.runtime_root)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runtime, token))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    payload = {
        "run_id": FIXED_RUN_ID,
        "spec": load_goal_spec("configs/g21_five_competition_goal.json"),
        "board": _board(),
    }
    fake_hash = "a" * 64
    fake_board = {
        "schema": "evomind.goal-board.v2",
        "run_id": FIXED_RUN_ID,
        "allocation": FIXED_ALLOCATION,
        "target_count": len(COMPETITIONS),
        "weather_actions": 0,
        "human_baseline_evidence_artifact_id": "artifact-unpublished-human-evidence",
        "human_baseline_evidence_sha256": fake_hash,
        "competitions": [
            {
                "competition": name,
                "solution_id": f"{name}-candidate-fake",
                "goal_status": "VERIFIED_COMPLETE",
                "official_metric": "score",
                "metric_direction": "higher_is_better",
                "human_baseline": {
                    "status": "VERIFIED",
                    "participant_type": "human",
                    "source": "not-a-url-or-doi",
                    "source_sha256": fake_hash,
                    "title": "self reported",
                    "sample_size": 2,
                    "protocol": "arbitrary text longer than twenty characters",
                    "metric": "score",
                    "direction": "higher_is_better",
                    "mean": 0.5,
                    "uncertainty": {"lower": 0.4, "upper": 0.6},
                },
                "human_comparison": {"candidate_metric": 1.0, "ci95_lower": 0.8, "ci95_upper": 1.2},
                "artifact_manifest": {"files": [{"anything": "accepted-before-the-fix"}]},
                "artifact_manifest_artifact_id": "artifact-unpublished-manifest",
                "artifact_manifest_sha256": fake_hash,
                "command_sha256": fake_hash,
                "environment_sha256": fake_hash,
                "data_manifest_sha256": fake_hash,
                "source_code_sha256": fake_hash,
                "model_sha256": fake_hash,
                "metrics_sha256": fake_hash,
                "rollback_sha256": fake_hash,
                "leakage_audit": {
                    "test_labels_used": False,
                    "hidden_test_labels_used": False,
                    "manifest_closed": True,
                    "loader_smoke": True,
                },
                "side_effects": {"signals_sent": 0, "other_processes_modified": False},
            }
            for name in COMPETITIONS
        ],
    }
    try:
        status, _ = _request(port, token, "POST", "/v1/goals", payload)
        assert status == 201
        status, rejected = _request(
            port,
            token,
            "PUT",
            "/v1/goals/goal_g21_five_competition",
            {"board": fake_board, "status": "completed"},
        )
        assert status == 400
        assert rejected["error"] == "invalid_request"
        status, unchanged = _request(port, token, "GET", "/v1/goals/goal_g21_five_competition")
        assert status == 200
        assert unchanged["status"] == "blocked"
        assert not any(
            item["event_type"] == "goal.updated"
            for item in runtime.store.list_events(FIXED_RUN_ID)
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        runtime.close()


def test_fixed_goal_metadata_conflict_fails_before_persistence(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    _seed_fixed_run(runtime, tmp_path)
    runtime.store.update_session(
        FIXED_RUN_ID,
        metadata_json={"goal_id": "goal_wrong", "goal_key": "wrong_goal"},
    )
    try:
        with pytest.raises(ValueError, match="metadata conflicts"):
            runtime.ensure_fixed_goal(
                run_id=FIXED_RUN_ID,
                spec=load_goal_spec("configs/g21_five_competition_goal.json"),
                board=_board(),
            )
        assert runtime.store.get_goal_record("goal_g21_five_competition") is None
        assert not any(
            event["event_type"] == "goal.created"
            for event in runtime.store.list_events(FIXED_RUN_ID)
        )
    finally:
        runtime.close()


def test_fixed_goal_hash_metadata_conflict_fails_before_persistence(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    _seed_fixed_run(runtime, tmp_path)
    runtime.store.update_session(
        FIXED_RUN_ID,
        metadata_json={"goal_spec_sha256": "0" * 64},
    )
    try:
        with pytest.raises(ValueError, match="metadata conflicts"):
            runtime.ensure_fixed_goal(
                run_id=FIXED_RUN_ID,
                spec=load_goal_spec("configs/g21_five_competition_goal.json"),
                board=_board(),
            )
        assert runtime.store.get_goal_record("goal_g21_five_competition") is None
    finally:
        runtime.close()


def test_fixed_goal_corrupt_existing_record_is_not_replayed(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    _seed_fixed_run(runtime, tmp_path)
    spec = load_goal_spec("configs/g21_five_competition_goal.json")
    created = runtime.ensure_fixed_goal(run_id=FIXED_RUN_ID, spec=spec, board=_board())
    runtime.store._connection.execute(
        "UPDATE goal_records SET board_sha256=? WHERE id=?",
        ("0" * 64, created["record"]["id"]),
    )
    runtime.store._connection.commit()
    try:
        with pytest.raises(ValueError, match="goal identity conflict"):
            runtime.ensure_fixed_goal(run_id=FIXED_RUN_ID, spec=spec, board=_board())
    finally:
        runtime.close()
