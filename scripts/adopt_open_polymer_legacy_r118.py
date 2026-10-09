from __future__ import annotations

import argparse
import hashlib
import http.client
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


FIXED_RUN_ID = "run_7b1efb878afb40f396db431e91f093a5"
FIXED_GOAL_ID = "goal_g21_five_competition"
FIXED_ALLOCATION = "G21"
FIXED_SPEC_SHA256 = "fbb9939dff28c355c2798bf32fce104845337064f42f9c280f2b6906cccf4036"
INITIAL_BOARD_SHA256 = "85288de01557f7801804c4a5e0c86d3dbb07a4d93acba1ec9c3277835f9a4cc6"
POLICY_MODE = "organizer_or_public_strong_with_conditional_human"
POLICY_EVIDENCE_SHA256 = "f81701f13aeebef146659e091cacfc4e39ae18df345ee6b58be2ad5ef40829f0"
ADOPTION_SCHEMA = "evomind.existing-candidate-adoption.v2"
SOURCE_CALL_ID = "call_54e3a187cfc179efd9060a283489691a"
SOLUTION_ID = "open-polymer-v2-baseline-candidate-1"
COMPETITIONS = ("cure_bench", "e2lmc", "mindgames", "open_polymer", "ariel_2025")


class AdoptionGate(ValueError):
    pass


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise AdoptionGate(f"regular JSON file required: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise AdoptionGate(f"JSON object required: {path.name}")
    return value


def validate_item(item: dict[str, Any]) -> None:
    adoption = item.get("existing_candidate_adoption")
    if item.get("holdout_ledger") or item.get("holdout_claim"):
        raise AdoptionGate("posthoc holdout ledger or claim is forbidden for legacy adoption")
    if (
        item.get("competition") != "open_polymer"
        or item.get("goal_status") != "VERIFIED_COMPLETE"
        or item.get("solution_id") != SOLUTION_ID
        or not isinstance(adoption, dict)
        or adoption.get("schema") != ADOPTION_SCHEMA
        or adoption.get("mode") != "legacy_pre_goal_completed_candidate"
        or adoption.get("run_id") != FIXED_RUN_ID
        or adoption.get("allocation") != FIXED_ALLOCATION
        or adoption.get("source_tool_call_id") != SOURCE_CALL_ID
        or adoption.get("source_solution_id") != SOLUTION_ID
        or adoption.get("retraining_performed") is not False
        or adoption.get("historical_atomic_holdout_claim_present") is not False
        or adoption.get("historical_ledger_fabricated") is not False
    ):
        raise AdoptionGate("Open Polymer legacy adoption identity rejected")
    if item.get("official_metric_claimed") is not False:
        raise AdoptionGate("legacy adoption cannot claim the official metric")
    leakage = item.get("leakage_audit")
    if not isinstance(leakage, dict) or leakage.get("test_labels_used") is not False:
        raise AdoptionGate("test-label boundary rejected")
    side_effects = item.get("side_effects")
    if (
        not isinstance(side_effects, dict)
        or int(side_effects.get("training_tool_calls", -1)) != 0
        or int(side_effects.get("remote_writes", -1)) != 0
        or int(side_effects.get("signals_sent", -1)) != 0
        or side_effects.get("other_processes_modified") is not False
    ):
        raise AdoptionGate("legacy adoption side-effect boundary rejected")


def build_board(goal: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    if (
        goal.get("id") != FIXED_GOAL_ID
        or goal.get("run_id") != FIXED_RUN_ID
        or goal.get("allocation") != FIXED_ALLOCATION
        or goal.get("status") != "blocked"
        or goal.get("spec_sha256") != FIXED_SPEC_SHA256
        or goal.get("board_sha256") != INITIAL_BOARD_SHA256
    ):
        raise AdoptionGate("current Goal identity or initial board precondition rejected")
    current = goal.get("board")
    rows = current.get("competitions") if isinstance(current, dict) else None
    if not isinstance(rows, list) or len(rows) != 5:
        raise AdoptionGate("current Goal board is not the exact five-item initial board")
    by_name = {str(row.get("competition")): row for row in rows if isinstance(row, dict)}
    if set(by_name) != set(COMPETITIONS):
        raise AdoptionGate("current Goal board competition set rejected")
    validate_item(item)
    next_rows = [item if name == "open_polymer" else by_name[name] for name in COMPETITIONS]
    board = {
        "schema": "evomind.goal-board.v2",
        "run_id": FIXED_RUN_ID,
        "allocation": FIXED_ALLOCATION,
        "baseline_policy": POLICY_MODE,
        "policy_evidence_sha256": POLICY_EVIDENCE_SHA256,
        "target_count": 5,
        "weather_actions": 0,
        "competitions": next_rows,
    }
    for name in ("cure_bench", "e2lmc", "mindgames", "ariel_2025"):
        if next(row for row in next_rows if row["competition"] == name) != by_name[name]:
            raise AdoptionGate(f"non-target competition drifted: {name}")
    return board


class LoopbackClient:
    def __init__(self, base_url: str, token: str) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise AdoptionGate("loopback HTTP base URL required")
        self.host = parsed.hostname
        self.port = parsed.port or 80
        self.prefix = parsed.path.rstrip("/")
        self.token = token

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        raw = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {"Authorization": f"Bearer {self.token}"}
        if raw is not None:
            headers.update({"Content-Type": "application/json", "Content-Length": str(len(raw))})
        connection = http.client.HTTPConnection(self.host, self.port, timeout=20)
        connection.request(method, self.prefix + path, body=raw, headers=headers)
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        connection.close()
        return response.status, payload


def execute(client: LoopbackClient, item: dict[str, Any]) -> dict[str, Any]:
    status, run = client.request("GET", f"/v1/runs/{FIXED_RUN_ID}")
    pending = [row for row in run.get("approvals", []) if row.get("status") == "pending"]
    if (
        status != 200
        or run.get("id") != FIXED_RUN_ID
        or run.get("terminal") is not True
        or len(run.get("active_tool_calls", [])) != 0
        or pending
    ):
        raise AdoptionGate("Run is not terminal with active=0 and pending=0")
    after_seq = int(run.get("last_event_seq", 0))
    status, goal = client.request("GET", f"/v1/goals/{FIXED_GOAL_ID}")
    if status != 200:
        raise AdoptionGate("current Goal GET failed")
    board = build_board(goal, item)
    body = {"board": board, "status": "blocked"}
    status, updated = client.request("PUT", f"/v1/goals/{FIXED_GOAL_ID}", body)
    if status != 200 or updated.get("board_sha256") != canonical_sha256(board) or updated.get("status") != "blocked":
        raise AdoptionGate("managed Goal PUT failed")
    status, replay = client.request("PUT", f"/v1/goals/{FIXED_GOAL_ID}", body)
    if status != 200 or replay != updated:
        raise AdoptionGate("managed Goal PUT replay is not idempotent")
    status, final_run = client.request("GET", f"/v1/runs/{FIXED_RUN_ID}")
    upper = int(final_run.get("last_event_seq", 0))
    status, events = client.request("GET", f"/v1/sessions/{FIXED_RUN_ID}/events?after={after_seq}")
    updates = [event for event in events.get("events", []) if event.get("event_type") == "goal.updated" and int(event.get("seq", 0)) <= upper]
    if status != 200 or len(updates) != 1:
        raise AdoptionGate("goal.updated event count rejected")
    return {
        "schema": "evomind.open_polymer.legacy_adoption_result.r118.v1",
        "status": "adopted_and_verified",
        "run_id": FIXED_RUN_ID,
        "goal_id": FIXED_GOAL_ID,
        "goal_status": "blocked",
        "board_sha256": updated["board_sha256"],
        "goal_updated_event_count": 1,
        "idempotent_replay": True,
        "training_tool_calls": 0,
        "hpc_accessed": False,
        "gpu_touched": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Adopt the fixed Open Polymer legacy candidate through loopback Goal API only")
    parser.add_argument("--adoption-item", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--fixture-goal")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--token-file")
    args = parser.parse_args()
    item = load_object(Path(args.adoption_item))
    try:
        if args.validate_only:
            if not args.fixture_goal:
                raise AdoptionGate("--fixture-goal is required for ValidateOnly")
            board = build_board(load_object(Path(args.fixture_goal)), item)
            result = {
                "schema": "evomind.open_polymer.legacy_adoption_validate_only.r118.v1",
                "status": "validated_not_applied",
                "board_sha256": canonical_sha256(board),
                "production_action_performed": False,
                "goal_put_calls": 0,
                "hpc_accessed": False,
                "gpu_touched": False,
            }
        else:
            if not args.token_file:
                raise AdoptionGate("--token-file is required")
            token = Path(args.token_file).read_text(encoding="utf-8").strip()
            if not token:
                raise AdoptionGate("runtime token is empty")
            result = execute(LoopbackClient(args.base_url, token), item)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
