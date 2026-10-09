from __future__ import annotations

import hashlib
import json
from pathlib import Path

from xsci.parity_campaign_harness import (
    ORACLE_EVIDENCE_SCHEMA,
    REQUIRED_AGENTS,
    SUITE_SCHEMA,
    TRIAL_SCHEMA,
    load_suite_manifest,
    verify_campaign,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, candidate_failures: int = 0):
    domains = [f"domain-{index}" for index in range(8)]
    tasks = [
        {
            "task_id": f"hidden-{index:03d}",
            "domain": domains[index % len(domains)],
            "prompt_sha256": hashlib.sha256(f"prompt:{index}".encode()).hexdigest(),
        }
        for index in range(104)
    ]
    suite = {
        "schema": SUITE_SCHEMA,
        "suite_id": "held-out-suite-1",
        "version": "1.0.0",
        "held_out": True,
        "selection_locked_before_evaluation": True,
        "tasks_not_used_for_development": True,
        "tasks": tasks,
    }
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(suite, sort_keys=True), encoding="utf-8")
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    rows = []
    cell = 0
    for agent in REQUIRED_AGENTS:
        for task in tasks:
            for repetition in range(1, 4):
                candidate_failed = agent == "evomind" and cell < candidate_failures
                outcome = "failed" if candidate_failed else "passed"
                cell_id = f"{agent}-{task['task_id']}-{repetition}"
                oracle_path = evidence_root / f"oracle-{cell_id}.json"
                oracle_path.write_text(json.dumps({
                    "schema": ORACLE_EVIDENCE_SCHEMA,
                    "campaign_id": "campaign-1",
                    "suite_id": suite["suite_id"],
                    "task_id": task["task_id"],
                    "domain": task["domain"],
                    "repetition": repetition,
                    "agent": agent,
                    "oracle_id": "private-oracle-1",
                    "passed": not candidate_failed,
                    "fixture_sha256": hashlib.sha256(f"fixture:{cell_id}".encode()).hexdigest(),
                    "workspace_snapshot_sha256": hashlib.sha256(f"workspace:{cell_id}".encode()).hexdigest(),
                }, sort_keys=True), encoding="utf-8")
                artifact_path = evidence_root / f"workspace-{cell_id}.diff"
                artifact_path.write_text(f"verified workspace delta {cell_id}\n", encoding="utf-8")
                trace_path = evidence_root / f"tool-trace-{cell_id}.jsonl"
                trace_path.write_text(
                    json.dumps({"tool_call_id": f"{cell_id}-1", "tool_name": "read", "status": "succeeded"}) + "\n"
                    + json.dumps({"tool_call_id": f"{cell_id}-2", "tool_name": "write", "status": "succeeded"}) + "\n",
                    encoding="utf-8",
                )
                rows.append({
                    "schema": TRIAL_SCHEMA,
                    "campaign_id": "campaign-1",
                    "suite_id": suite["suite_id"],
                    "task_id": task["task_id"],
                    "domain": task["domain"],
                    "repetition": repetition,
                    "agent": agent,
                    "agent_version": "test-version",
                    "provider": "test-provider",
                    "model": "test-model",
                    "outcome": outcome,
                    "timed_out": False,
                    "duration_seconds": 1.25,
                    "scope_violation": False,
                    "unsupported_claims": 0,
                    "tool_calls": {"total": 2, "succeeded": 2, "failed": 0},
                    "oracle": {
                        "id": "private-oracle-1",
                        "passed": not candidate_failed,
                        "evidence_path": oracle_path.name,
                        "evidence_sha256": _sha(oracle_path),
                        "evidence_bytes": oracle_path.stat().st_size,
                    },
                    "artifacts": [{
                        "role": "workspace_result",
                        "path": artifact_path.name,
                        "sha256": _sha(artifact_path),
                        "bytes": artifact_path.stat().st_size,
                    }, {
                        "role": "tool_trace",
                        "path": trace_path.name,
                        "sha256": _sha(trace_path),
                        "bytes": trace_path.stat().st_size,
                    }],
                })
                if agent == "evomind":
                    cell += 1
    loaded, suite_sha = load_suite_manifest(suite_path)
    return loaded, suite_sha, rows, evidence_root


def test_complete_real_evidence_matrix_is_projection_ready_but_not_release_certified(tmp_path: Path):
    suite, suite_sha, rows, evidence_root = _fixture(tmp_path)
    result = verify_campaign(suite, suite_sha, rows, evidence_root=evidence_root)

    assert result["status"] == "evidence_projection_ready"
    assert result["release_gate"] == "NO-GO"
    assert result["hidden_task_count"] == 104
    assert len(result["domain_task_counts"]) == 8
    assert result["expected_trials"] == result["verified_trials"] == 104 * 3 * 3
    assert result["matrix_complete"] is True
    assert result["agent_summaries"]["evomind"]["tool_calls_total"] == 104 * 3 * 2
    assert all(item["noninferior"] for item in result["paired_comparisons"].values())
    assert result["next_gate"] == "xsci.capability_certification.verify_capability_certification"


def test_missing_cell_and_forged_artifact_hash_fail_closed(tmp_path: Path):
    suite, suite_sha, rows, evidence_root = _fixture(tmp_path)
    rows.pop()
    rows[0]["artifacts"][0]["sha256"] = "0" * 64

    result = verify_campaign(suite, suite_sha, rows, evidence_root=evidence_root)

    assert result["status"] == "no_go"
    assert result["release_gate"] == "NO-GO"
    assert result["matrix_complete"] is False
    assert any("SHA-256 does not match" in blocker for blocker in result["blockers"])
    assert any("matrix is missing" in blocker for blocker in result["blockers"])


def test_unsupported_claim_and_tool_trace_partition_are_release_blockers(tmp_path: Path):
    suite, suite_sha, rows, evidence_root = _fixture(tmp_path)
    rows[0]["unsupported_claims"] = 1
    rows[1]["tool_calls"] = {"total": 2, "succeeded": 2, "failed": 1}

    result = verify_campaign(suite, suite_sha, rows, evidence_root=evidence_root)

    assert result["status"] == "no_go"
    assert any("unsupported claims" in blocker for blocker in result["blockers"])
    assert any("tool-call counters" in blocker for blocker in result["blockers"])
