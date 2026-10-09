#!/usr/bin/env python3
"""Verify two independent evolution CLI runs as a cross-session memory loop."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_STRATEGIES = {
    "linear_baseline",
    "bagged_trees",
    "invalid_hyperparameter_probe",
    "gradient_boosting_repair",
    "shallow_tree_ablation",
    "probabilistic_baseline",
    "randomized_tree_ablation",
    "complementary_soft_voting",
}
EXPECTED_OPERATORS = {"Draft", "Improve", "Debug", "Crossover"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _resolve_repo_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT.resolve())
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing regular evidence file: {value}")
    return path


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "relative_path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("evolution event must be an object")
        events.append(payload)
    return events


def verify(round0_path: Path, round1_path: Path) -> dict[str, Any]:
    round0 = _load_json(round0_path)
    round1 = _load_json(round1_path)
    if round0.get("schema") != "evomind.evolution_cli_receipt.v1" or round1.get("schema") != round0.get("schema"):
        raise ValueError("CLI receipt schema mismatch")

    summary0_path = _resolve_repo_path(str(round0.get("summary_path") or ""))
    summary1_path = _resolve_repo_path(str(round1.get("summary_path") or ""))
    summary0 = _load_json(summary0_path)
    summary1 = _load_json(summary1_path)
    events1_path = _resolve_repo_path(str(round1["observability"]["events"]))
    events1 = _load_events(events1_path)
    propose_events1 = [item for item in events1 if item.get("type") == "propose"]

    order0 = list(round0.get("memory", {}).get("strategy_order") or [])
    order1 = list(round1.get("memory", {}).get("strategy_order") or [])
    refs1 = set(round1.get("memory", {}).get("memory_ref_ids_used") or [])
    round0_run_id = str(round0.get("run_id") or "")
    round1_run_id = str(round1.get("run_id") or "")
    expected_refs = {
        f"evomind_demo_customer_churn:{round0_run_id}:EXP{index:03d}"
        for index in range(8)
    }
    promoted0 = {
        str(strategy)
        for item in list(summary0.get("iterations") or [])
        if item.get("promoted") is True
        for strategy in list(item.get("applied_strategies") or [])[:1]
    }

    claim0_path = _resolve_repo_path(f"{round0['exp_dir']}/claim-audit.json")
    claim1_path = _resolve_repo_path(f"{round1['exp_dir']}/claim-audit.json")
    claim0 = _load_json(claim0_path)
    claim1 = _load_json(claim1_path)
    operators0 = set(claim0.get("evidence", {}).get("search", {}).get("operators") or [])
    operators1 = set(claim1.get("evidence", {}).get("search", {}).get("operators") or [])
    crossovers0 = claim0.get("evidence", {}).get("search", {}).get("crossover_lineages") or []
    crossovers1 = claim1.get("evidence", {}).get("search", {}).get("crossover_lineages") or []

    memory_path = _resolve_repo_path(str(round1["observability"]["retrospective_memory"]))
    memory_payload = json.loads(memory_path.read_text(encoding="utf-8"))
    if not isinstance(memory_payload, list):
        raise ValueError("retrospective memory must be a list")
    memory_by_run: dict[str, list[dict[str, Any]]] = {round0_run_id: [], round1_run_id: []}
    for item in memory_payload:
        if not isinstance(item, dict):
            continue
        profile = item.get("dataset_profile") if isinstance(item.get("dataset_profile"), dict) else {}
        run_id = str(profile.get("run_id") or "")
        if run_id in memory_by_run:
            memory_by_run[run_id].append(item)

    event_refs_match = (
        len(propose_events1) == 8
        and all(set(item.get("memory_ref_ids") or []) == expected_refs for item in propose_events1)
    )
    checks = {
        "both_cli_runs_passed": round0.get("ok") is True and round1.get("ok") is True,
        "round0_started_with_empty_memory": round0.get("memory", {}).get("records_before") == 0,
        "round0_persisted_eight_records": round0.get("memory", {}).get("records_after") == 8,
        "round1_loaded_round0_records": round1.get("memory", {}).get("records_before") == 8,
        "round1_preserved_both_sessions": round1.get("memory", {}).get("records_after") == 16,
        "round1_referenced_all_round0_memories": refs1 == expected_refs,
        "round1_events_expose_memory_refs": event_refs_match,
        "strategy_set_is_frozen": set(order0) == set(order1) == EXPECTED_STRATEGIES,
        "round1_strategy_order_changed": order0 != order1,
        "round1_frontloads_promoted_strategies": len(order1) >= 2 and set(order1[:2]).issubset(promoted0),
        "round1_retains_bounded_debug_probe": len(order1) >= 3 and order1[2] == "invalid_hyperparameter_probe",
        "round0_claim_audit_passed": claim0.get("status") == "passed",
        "round1_claim_audit_passed": claim1.get("status") == "passed",
        "round0_operator_set_complete": operators0 == EXPECTED_OPERATORS,
        "round1_operator_set_complete": operators1 == EXPECTED_OPERATORS,
        "round0_valid_crossover": any(item.get("valid") is True for item in crossovers0),
        "round1_valid_crossover": any(item.get("valid") is True for item in crossovers1),
        "round0_memory_survives_process_restart": len(memory_by_run.get(round0_run_id, [])) == 8,
        "round1_memory_persisted": len(memory_by_run.get(round1_run_id, [])) == 8,
        "success_and_failure_lessons_persisted": (
            any(item.get("what_worked") for item in memory_payload)
            and any(item.get("what_failed") or item.get("failure_pattern") for item in memory_payload)
        ),
        "independent_holdout_scores_match": (
            round0.get("demo_evidence", {}).get("independent_review", {}).get("score")
            == round1.get("demo_evidence", {}).get("independent_review", {}).get("score")
        ),
    }

    evidence_paths = [
        round0_path,
        round1_path,
        summary0_path,
        summary1_path,
        events1_path,
        memory_path,
        claim0_path,
        claim1_path,
        _resolve_repo_path(str(round0["observability"]["experience_board"])),
        _resolve_repo_path(str(round1["observability"]["experience_board"])),
        _resolve_repo_path(f"{round0['exp_dir']}/search_graph.json"),
        _resolve_repo_path(f"{round1['exp_dir']}/search_graph.json"),
        _resolve_repo_path(f"{round0['exp_dir']}/candidate-freeze.json"),
        _resolve_repo_path(f"{round1['exp_dir']}/candidate-freeze.json"),
        _resolve_repo_path(f"{round0['exp_dir']}/independent-review.json"),
        _resolve_repo_path(f"{round1['exp_dir']}/independent-review.json"),
    ]
    artifacts = [_artifact(path) for path in evidence_paths]
    memory_hash_first = _sha256(memory_path)
    memory_hash_second = _sha256(memory_path)
    checks["restart_read_is_byte_stable"] = memory_hash_first == memory_hash_second

    return {
        "schema": "evomind.cross_session_evolution_acceptance.v1",
        "status": "passed" if all(checks.values()) else "rejected",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "round0": {
            "run_id": round0_run_id,
            "strategy_order": order0,
            "memory_records_before": round0.get("memory", {}).get("records_before"),
            "memory_records_after": round0.get("memory", {}).get("records_after"),
            "best_cv_score": round0.get("best_cv_score"),
            "independent_holdout_score": round0.get("demo_evidence", {}).get("independent_review", {}).get("score"),
        },
        "round1": {
            "run_id": round1_run_id,
            "strategy_order": order1,
            "memory_ref_ids": sorted(refs1),
            "memory_records_before": round1.get("memory", {}).get("records_before"),
            "memory_records_after": round1.get("memory", {}).get("records_after"),
            "best_cv_score": round1.get("best_cv_score"),
            "independent_holdout_score": round1.get("demo_evidence", {}).get("independent_review", {}).get("score"),
        },
        "persistence": {
            "memory_sha256": memory_hash_first,
            "records": len(memory_payload),
            "round0_records": len(memory_by_run.get(round0_run_id, [])),
            "round1_records": len(memory_by_run.get(round1_run_id, [])),
        },
        "artifacts": artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round0", required=True)
    parser.add_argument("--round1", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    round0_path = _resolve_repo_path(args.round0)
    round1_path = _resolve_repo_path(args.round1)
    payload = verify(round0_path, round1_path)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({
        "status": payload["status"],
        "output": str(output),
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "passed_checks": sum(1 for value in payload["checks"].values() if value is True),
        "total_checks": len(payload["checks"]),
    }, ensure_ascii=False))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
