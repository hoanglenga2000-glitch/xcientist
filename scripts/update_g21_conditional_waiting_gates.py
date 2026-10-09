from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import http.client
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


FIXED_RUN_ID = "run_7b1efb878afb40f396db431e91f093a5"
FIXED_GOAL_ID = "goal_g21_five_competition"
FIXED_SPEC_SHA256 = "fbb9939dff28c355c2798bf32fce104845337064f42f9c280f2b6906cccf4036"
EXPECTED_CURRENT_BOARD_SHA256 = "0b7e948ebf52ebf63432bffb91f6dbff1763bed7861752525e1d201359d3b0b5"
POLICY_MODE = "organizer_or_public_strong_with_conditional_human"
POLICY_EVIDENCE_SHA256 = "f81701f13aeebef146659e091cacfc4e39ae18df345ee6b58be2ad5ef40829f0"

AVAILABILITY = {
    "cure_bench": (
        "artifact_cfadb7ced1f04d48b97174533bd8ca09",
        "4abbcdf1532c00291030a96417fce4ddd9d211e6a0e5204067fe0d22a8050b87",
    ),
    "e2lmc": (
        "artifact_64a60a65a01d435fbc1c31d4f312d06e",
        "2396774b7016c2d41af12e390fd91e928cd5d9a2d35768d062ccc2f1db8772c5",
    ),
    "mindgames": (
        "artifact_d37ebd45b9c94f0b8201903d93ad7083",
        "83aaddaa7f05ca05de942dc60e2fffa59af33b65d48282f4ad0f003ccc724d66",
    ),
    "ariel_2025": (
        "artifact_5e7dd897b0094cbc899660531e6daffe",
        "dd87a44502f04f4ca5c25fe24db024dd2876bc7d74715ea13e808446d491e356",
    ),
}

ADDITIONAL_GATES = {
    "cure_bench": [
        "UNCONSUMED_HOLDOUT_LEDGER_REQUIRED",
        "PRIMARY_BASELINE_MANIFEST_REQUIRED",
        "FROZEN_TRANSFORMER_SNAPSHOT_REQUIRED",
    ],
    "e2lmc": [],
    "mindgames": ["FORMAL_PROTOCOL_MANIFEST_REQUIRED"],
    "ariel_2025": [],
}


class GateUpdateError(ValueError):
    pass


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_board(goal: dict[str, Any]) -> dict[str, Any]:
    if (
        goal.get("id") != FIXED_GOAL_ID
        or goal.get("run_id") != FIXED_RUN_ID
        or goal.get("status") != "blocked"
        or goal.get("spec_sha256") != FIXED_SPEC_SHA256
        or goal.get("board_sha256") != EXPECTED_CURRENT_BOARD_SHA256
    ):
        raise GateUpdateError("current Goal precondition rejected")
    board = deepcopy(goal.get("board"))
    if not isinstance(board, dict):
        raise GateUpdateError("Goal board is missing")
    rows = board.get("competitions")
    if not isinstance(rows, list) or len(rows) != 5:
        raise GateUpdateError("five competition rows required")
    by_name = {str(row.get("competition")): row for row in rows if isinstance(row, dict)}
    if set(by_name) != {"cure_bench", "e2lmc", "mindgames", "open_polymer", "ariel_2025"}:
        raise GateUpdateError("competition set rejected")
    open_polymer = deepcopy(by_name["open_polymer"])
    if (
        open_polymer.get("goal_status") != "VERIFIED_COMPLETE"
        or open_polymer.get("solution_id") != "open-polymer-v2-baseline-candidate-1"
        or open_polymer.get("official_metric_claimed") is not False
        or (open_polymer.get("existing_candidate_adoption") or {}).get("schema")
        != "evomind.existing-candidate-adoption.v2"
    ):
        raise GateUpdateError("Open Polymer completion evidence rejected")
    for competition, (artifact_id, digest) in AVAILABILITY.items():
        row = by_name[competition]
        if row.get("goal_status") != "WAITING_EXACT_GATE":
            raise GateUpdateError(f"non-target status drift: {competition}")
        row["human_baseline_availability"] = {
            "status": "NOT_VERIFIED_AVAILABLE",
            "audit_complete": True,
            "protocol_search_complete": True,
            "human_baseline_verified": False,
            "evidence_artifact_id": artifact_id,
            "evidence_artifact_sha256": digest,
        }
        row["additional_gates"] = list(ADDITIONAL_GATES[competition])
    if by_name["open_polymer"] != open_polymer:
        raise GateUpdateError("Open Polymer evidence changed")
    board.update(
        baseline_policy=POLICY_MODE,
        policy_evidence_sha256=POLICY_EVIDENCE_SHA256,
        completion_count=1,
        goal_record_status="blocked",
    )
    return board


class LoopbackClient:
    def __init__(self, base_url: str, token: str) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise GateUpdateError("loopback HTTP required")
        self.host = parsed.hostname
        self.port = parsed.port or 80
        self.prefix = parsed.path.rstrip("/")
        self.token = token

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        raw = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {"Authorization": f"Bearer {self.token}"}
        if raw is not None:
            headers.update({"Content-Type": "application/json", "Content-Length": str(len(raw))})
        connection = http.client.HTTPConnection(self.host, self.port, timeout=60)
        connection.request(method, self.prefix + path, body=raw, headers=headers)
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        connection.close()
        return response.status, payload


def execute(client: LoopbackClient) -> dict[str, Any]:
    status, run = client.request("GET", f"/v1/runs/{FIXED_RUN_ID}")
    if (
        status != 200
        or run.get("terminal") is not True
        or run.get("active_tool_calls")
        or [item for item in run.get("approvals", []) if item.get("status") == "pending"]
    ):
        raise GateUpdateError("Run is not quiescent")
    after = int(run.get("last_event_seq") or 0)
    status, goal = client.request("GET", f"/v1/goals/{FIXED_GOAL_ID}")
    if status != 200:
        raise GateUpdateError("Goal GET failed")
    board = build_board(goal)
    body = {"board": board, "status": "blocked"}
    status, first = client.request("PUT", f"/v1/goals/{FIXED_GOAL_ID}", body)
    status2, replay = client.request("PUT", f"/v1/goals/{FIXED_GOAL_ID}", body)
    expected_sha = canonical_sha256(board)
    if (
        status != 200
        or status2 != 200
        or first.get("board_sha256") != expected_sha
        or replay != first
    ):
        raise GateUpdateError("Goal PUT or replay failed")
    status, events = client.request("GET", f"/v1/sessions/{FIXED_RUN_ID}/events?after={after}")
    updates = [event for event in events.get("events", []) if event.get("event_type") == "goal.updated"]
    if status != 200 or len(updates) != 1 or (updates[0].get("payload") or {}).get("board_sha256") != expected_sha:
        raise GateUpdateError("goal.updated evidence rejected")
    status, final_run = client.request("GET", f"/v1/runs/{FIXED_RUN_ID}")
    if status != 200 or final_run.get("terminal") is not True or final_run.get("active_tool_calls"):
        raise GateUpdateError("Run changed during Goal update")
    return {
        "schema": "evomind.g21_conditional_waiting_gate_update.v1",
        "status": "updated_and_replayed",
        "run_id": FIXED_RUN_ID,
        "goal_id": FIXED_GOAL_ID,
        "board_sha256": expected_sha,
        "completion_count": 1,
        "goal_updated_event_count": 1,
        "goal_updated_event_seq": int(updates[0]["seq"]),
        "training_tool_calls": 0,
        "hpc_accessed": False,
        "gpu_touched": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Update non-blocking human availability on the fixed G21 Goal")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--token-file")
    parser.add_argument("--validate-only-goal")
    args = parser.parse_args()
    try:
        if args.validate_only_goal:
            goal = json.loads(Path(args.validate_only_goal).read_text(encoding="utf-8-sig"))
            board = build_board(goal)
            result = {
                "schema": "evomind.g21_conditional_waiting_gate_update_validate_only.v1",
                "status": "validated_not_applied",
                "board_sha256": canonical_sha256(board),
                "production_action_performed": False,
            }
        else:
            if not args.token_file:
                raise GateUpdateError("--token-file is required")
            token = Path(args.token_file).read_text(encoding="utf-8").strip()
            if not token:
                raise GateUpdateError("runtime token is empty")
            result = execute(LoopbackClient(args.base_url, token))
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
