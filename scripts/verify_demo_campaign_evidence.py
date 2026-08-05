"""Fail-closed evidence verifier for the isolated EvoMind product-demo run.

The verifier is intentionally read-only with respect to the demo campaign.  It
does not execute candidates, reload models, grade predictions, contact Kaggle,
or invoke any private grader.  It verifies the already-produced evidence and
writes a compact recording-readiness report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_os import experience_mcgs  # noqa: E402

SCHEMA = "evomind.demo.evidence_verification.v1"
APPROVAL_SCHEMA = "evomind.evolution_approval_plan.v2"
CYCLE_RECEIPT_SCHEMA = "evomind.demo.cycle_receipt.v1"
CLI_RECEIPT_SCHEMA = "evomind.evolution_cli_receipt.v1"
INPUT_CONTRACT_SCHEMA = "evomind.demo.input_contract.v1"
CANONICAL_HASH_SCHEMA = experience_mcgs.CANONICAL_HASH_SCHEMA
EXPECTED_TASK_ID = "evomind_demo_customer_churn"
EXPECTED_OPERATORS = {"Draft", "Improve", "Debug", "Crossover"}
EXPECTED_EXPERIMENT_IDS = [f"EXP{index:03d}" for index in range(8)]
PLAN_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
FORBIDDEN_EVALUATION_RE = re.compile(
    r"(?:"
    r"(?<![A-Za-z0-9])private(?=$|[\s_-])|"
    r"private[\s_-]*(?:grader|score|feedback|label(?:s)?|metric|evaluation|result|data|split|holdout|test|submission)|"
    r"leaderboard|official[\s_-]*(?:rank|score|medal)"
    r")",
    re.IGNORECASE,
)
LOCAL_TIMEZONE = timezone(timedelta(hours=8))


def canonical_json(value: Any) -> str:
    return experience_mcgs.canonical_json(value)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name}:{line_number} must contain a JSON object")
        records.append(value)
    return records


def _safe_artifact_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    candidate.relative_to(root.resolve())
    return candidate


def _iso_timestamp(value: Any, *, assume_local: bool = False) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=LOCAL_TIMEZONE if assume_local else timezone.utc)
    return parsed.astimezone(timezone.utc)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _numbers_equal(left: Any, right: Any) -> bool:
    return _finite_number(left) and _finite_number(right) and math.isclose(
        float(left),
        float(right),
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def _regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def resolve_evidence_paths(
    run_dir: Path,
    *,
    cli_log: Path | None = None,
    input_contract: Path | None = None,
) -> dict[str, Path]:
    """Resolve modern run-local receipts while preserving legacy overrides."""

    root = run_dir.resolve()
    return {
        "cli_receipt": (cli_log if cli_log is not None else root / "cli-receipt.json").resolve(),
        "input_contract": (
            input_contract if input_contract is not None else root / "input-contract.json"
        ).resolve(),
        "approval_receipt": root / "approval-receipt.json",
        "cycle_receipt": root / "cycle-receipt.json",
    }


def _new_check(
    check_id: str,
    passed: bool,
    *,
    blocking: bool = True,
    detail: str,
    evidence: Any = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": check_id,
        "status": "passed" if passed else "failed",
        "blocking": blocking,
        "detail": detail,
    }
    if evidence is not None:
        result["evidence"] = evidence
    return result


def find_forbidden_evaluation_values(value: Any, path: str = "root") -> list[str]:
    """Return key/value paths that could leak non-public evaluation material."""

    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if FORBIDDEN_EVALUATION_RE.search(key_text):
                findings.append(child_path)
            findings.extend(find_forbidden_evaluation_values(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            findings.extend(find_forbidden_evaluation_values(child, f"{path}[{index}]"))
    elif isinstance(value, str) and FORBIDDEN_EVALUATION_RE.search(value):
        findings.append(path)
    return findings


def verify_board(board: Mapping[str, Any]) -> dict[str, Any]:
    """Independently recompute every card id, append chain and board hash."""

    errors: list[str] = []
    cards_raw = board.get("cards")
    append_order = board.get("append_order")
    cards = cards_raw if isinstance(cards_raw, list) else []
    order = append_order if isinstance(append_order, list) else []
    if not isinstance(cards_raw, list):
        errors.append("cards is not a list")
    if not isinstance(append_order, list):
        errors.append("append_order is not a list")
    card_ids: list[str] = []
    content_hashes: dict[str, str] = {}
    for index, raw_card in enumerate(cards):
        if not isinstance(raw_card, dict):
            errors.append(f"card[{index}] is not an object")
            continue
        card = dict(raw_card)
        card_id = card.get("card_id")
        if not isinstance(card_id, str):
            errors.append(f"card[{index}] has no card_id")
            continue
        without_id = dict(card)
        without_id.pop("card_id", None)
        content_hash = canonical_sha256(without_id)
        expected_card_id = f"exp_{content_hash[:24]}"
        if card_id != expected_card_id:
            errors.append(f"{card_id}: expected content-derived id {expected_card_id}")
        card_ids.append(card_id)
        content_hashes[card_id] = content_hash

    if card_ids != order:
        errors.append("append_order does not match serialized card order")
    if len(card_ids) != len(set(card_ids)):
        errors.append("duplicate card ids violate append-only deduplication")
    if board.get("append_only") is not True:
        errors.append("append_only marker is not true")
    if board.get("deduplication_key") != "canonical_card_hash":
        errors.append("unexpected deduplication key")
    if board.get("hash_canonicalization") != CANONICAL_HASH_SCHEMA:
        errors.append("unexpected canonical hash schema")

    chain_head = "0" * 64
    for card_id in order:
        card_hash = content_hashes.get(str(card_id))
        if not card_hash:
            errors.append(f"append_order references unknown card {card_id}")
            continue
        chain_head = canonical_sha256(
            {"previous": chain_head, "card_id": card_id, "card_hash": card_hash}
        )
    if board.get("append_chain_head") != chain_head:
        errors.append("append_chain_head mismatch")

    hash_payload = {
        "schema": board.get("schema"),
        "hash_canonicalization": board.get("hash_canonicalization"),
        "task_id": board.get("task_id"),
        "append_order": order,
        "append_chain_head": chain_head,
        "cards": cards,
        "task_global_aggregation": board.get("task_global_aggregation"),
    }
    expected_board_hash = canonical_sha256(hash_payload)
    if board.get("board_hash") != expected_board_hash:
        errors.append("board_hash mismatch")

    aggregation = board.get("task_global_aggregation")
    if not isinstance(aggregation, dict):
        errors.append("task_global_aggregation missing")
        aggregation = {}
    operator_counts = Counter(
        card.get("operator") for card in cards if isinstance(card, dict) and isinstance(card.get("operator"), str)
    )
    status_counts = Counter(
        card.get("status") for card in cards if isinstance(card, dict) and isinstance(card.get("status"), str)
    )
    expected_edges = sorted(
        {
            (str(parent), str(card.get("node_id")))
            for card in cards
            if isinstance(card, dict)
            for parent in (card.get("parent_ids") if isinstance(card.get("parent_ids"), list) else [])
        }
    )
    lineage = aggregation.get("lineage") if isinstance(aggregation.get("lineage"), dict) else {}
    observed_edges = sorted(
        (str(edge.get("parent_node_id")), str(edge.get("child_node_id")))
        for edge in (lineage.get("edges") if isinstance(lineage.get("edges"), list) else [])
        if isinstance(edge, dict)
    )
    if aggregation.get("card_count") != len(cards):
        errors.append("task-global card_count mismatch")
    if aggregation.get("operator_counts") != dict(sorted(operator_counts.items())):
        errors.append("task-global operator_counts mismatch")
    if aggregation.get("status_counts") != dict(sorted(status_counts.items())):
        errors.append("task-global status_counts mismatch")
    if observed_edges != expected_edges or lineage.get("edge_count") != len(expected_edges):
        errors.append("task-global lineage mismatch")

    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "card_count": len(cards),
        "card_ids": card_ids,
        "content_hashes": content_hashes,
        "append_chain_head": chain_head,
        "board_hash": expected_board_hash,
        "operator_counts": dict(sorted(operator_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "lineage_edge_count": len(expected_edges),
    }


def verify_dual_parent_crossover(
    cards: Iterable[Mapping[str, Any]],
    graph: Mapping[str, Any],
    traces: Iterable[Mapping[str, Any]],
    events: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    cards_by_node = {
        str(card.get("node_id")): card for card in cards if isinstance(card.get("node_id"), str)
    }
    crossover_cards = [card for card in cards_by_node.values() if card.get("operator") == "Crossover"]
    errors: list[str] = []
    evidence: list[dict[str, Any]] = []
    if not crossover_cards:
        errors.append("no Crossover card")
    graph_nodes = {
        str(node.get("exp_id")): node
        for node in (graph.get("nodes") if isinstance(graph.get("nodes"), list) else [])
        if isinstance(node, dict) and isinstance(node.get("exp_id"), str)
    }
    reference_edges = graph.get("reference_edges") if isinstance(graph.get("reference_edges"), list) else []
    for card in crossover_cards:
        node_id = str(card.get("node_id"))
        parents = card.get("parent_ids") if isinstance(card.get("parent_ids"), list) else []
        parents = [str(parent) for parent in parents]
        if len(parents) != 2 or len(set(parents)) != 2:
            errors.append(f"{node_id} does not have exactly two distinct parents")
            continue
        parent_cards = [cards_by_node.get(parent) for parent in parents]
        if any(parent is None for parent in parent_cards):
            errors.append(f"{node_id} references an unknown parent")
            continue
        if any(parent.get("status") != "success" for parent in parent_cards if parent):
            errors.append(f"{node_id} has a non-successful parent")
        families = {str(parent.get("method_family")) for parent in parent_cards if parent}
        if len(families) != 2:
            errors.append(f"{node_id} parents do not belong to distinct method families")
        graph_parents = graph_nodes.get(node_id, {}).get("reference_parent_ids")
        if not isinstance(graph_parents, list) or set(map(str, graph_parents)) != set(parents):
            errors.append(f"{node_id} graph parent set differs from card parent set")
        trace_match = any(
            trace.get("operator") == "Crossover"
            and set(map(str, trace.get("selected_parent_ids", []))) == set(parents)
            for trace in traces
        )
        if not trace_match:
            errors.append(f"{node_id} has no matching two-parent SelectionTrace")
        select_match = any(
            event.get("type") == "select"
            and event.get("exp_id") == node_id
            and event.get("operator") == "Crossover"
            and set(map(str, event.get("parent_exp_ids", []))) == set(parents)
            for event in events
        )
        propose_match = any(
            event.get("type") == "propose"
            and event.get("exp_id") == node_id
            and set(map(str, event.get("parent_exp_ids", []))) == set(parents)
            for event in events
        )
        if not select_match or not propose_match:
            errors.append(f"{node_id} event lineage is incomplete")
        secondary_edges = [
            edge
            for edge in reference_edges
            if isinstance(edge, dict)
            and edge.get("target") == node_id
            and edge.get("reference_type") == "crossover_parent"
            and str(edge.get("source")) in parents
        ]
        if not secondary_edges:
            errors.append(f"{node_id} has no secondary crossover reference edge")
        evidence.append({"node_id": node_id, "parent_ids": parents, "method_families": sorted(families)})
    return {"status": "passed" if not errors else "failed", "errors": errors, "crossovers": evidence}


def verify_cache_evidence(
    retrievals: Iterable[Mapping[str, Any]], stats: Mapping[str, Any]
) -> dict[str, Any]:
    records = list(retrievals)
    bundle_hits = sum(item.get("cache_hit") is True for item in records)
    bundle_misses = sum(item.get("cache_hit") is False for item in records)
    card_hits = sum(int(item.get("card_cache_hits") or 0) for item in records)
    card_misses = sum(int(item.get("card_cache_misses") or 0) for item in records)
    used_card_count = sum(len(item.get("card_ids", [])) for item in records if isinstance(item.get("card_ids"), list))
    nonempty_bundles = sum(bool(item.get("card_ids")) for item in records)
    stats_consistent = stats.get("hits") == card_hits and stats.get("misses") == card_misses
    return {
        "retrieval_bundle_hits": bundle_hits,
        "retrieval_bundle_misses": bundle_misses,
        "summary_card_hits": card_hits,
        "summary_card_misses": card_misses,
        "summary_stats_consistent": stats_consistent,
        "retrieved_card_references": used_card_count,
        "nonempty_retrieval_bundles": nonempty_bundles,
        # A bundle-level cache miss can still contain several per-card summary
        # hits plus one newly summarized card.  Recording readiness therefore
        # follows the authoritative per-card counters rather than requiring a
        # whole-bundle hit that may never occur during an append-only run.
        "hit_observed": card_hits > 0,
        "miss_observed": card_misses > 0,
        "memory_used": used_card_count > 0 and nonempty_bundles > 0,
    }


def verify_approval_receipt(
    receipt: Mapping[str, Any] | None,
    input_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Verify the consumed one-time plan and its exact request contract."""

    errors: list[str] = []
    payload = dict(receipt) if isinstance(receipt, Mapping) else {}
    request_contract = payload.get("request_contract")
    plan = payload.get("plan")
    if not payload:
        errors.append("approval receipt is missing")
    if payload.get("schema") != APPROVAL_SCHEMA:
        errors.append("approval schema mismatch")
    if payload.get("hash_canonicalization") != CANONICAL_HASH_SCHEMA:
        errors.append("approval canonical hash schema mismatch")
    if not isinstance(payload.get("plan_id"), str) or not PLAN_ID_RE.fullmatch(payload["plan_id"]):
        errors.append("approval plan_id is not a UUIDv4")
    if payload.get("task_id") != EXPECTED_TASK_ID:
        errors.append("approval task_id mismatch")
    if payload.get("status") != "approved":
        errors.append("approval receipt was not consumed exactly once")
    if not isinstance(request_contract, dict):
        errors.append("approval request_contract is missing")
        request_contract = {}
    if not isinstance(plan, dict):
        errors.append("approval plan is missing")
        plan = {}

    try:
        request_contract_sha256 = canonical_sha256(request_contract)
        plan_sha256 = canonical_sha256(plan)
        unsigned = dict(payload)
        unsigned.pop("receipt_sha256", None)
        receipt_sha256 = canonical_sha256(unsigned)
    except (TypeError, ValueError, OverflowError) as error:
        errors.append(f"approval canonical hashing failed: {error}")
        request_contract_sha256 = ""
        plan_sha256 = ""
        receipt_sha256 = ""
    if payload.get("request_fingerprint") != request_contract_sha256:
        errors.append("approval request_fingerprint mismatch")
    if payload.get("plan_sha256") != plan_sha256:
        errors.append("approval plan_sha256 mismatch")
    if payload.get("receipt_sha256") != receipt_sha256:
        errors.append("approval receipt_sha256 mismatch")

    created_at = _iso_timestamp(payload.get("created_at"))
    approved_at = _iso_timestamp(payload.get("approved_at"))
    expires_at = _iso_timestamp(payload.get("expires_at"))
    if not (
        created_at is not None
        and approved_at is not None
        and expires_at is not None
        and created_at <= approved_at <= expires_at
    ):
        errors.append("approval timestamps are missing, invalid, or out of order")

    expected_request_values: dict[str, Any] = {
        "task_id": EXPECTED_TASK_ID,
        "engine": "research_os",
        "runner": "local",
        "iterations": 8,
        "mcgs": True,
        "search_mode": "experience_mcgs_v1",
        "max_nodes": 8,
        "max_tokens": 100_000,
        "max_wall_seconds": 600,
        "max_cost": 0.01,
    }
    if any(request_contract.get(key) != value for key, value in expected_request_values.items()):
        errors.append("approval request_contract does not match the fixed demo request")
    if plan.get("task_id") != EXPECTED_TASK_ID or plan.get("official_submit_allowed") is not False:
        errors.append("approval plan identity or submission boundary mismatch")

    contract_payload = dict(input_contract) if isinstance(input_contract, Mapping) else {}
    request_to_input_keys = (
        "task_id",
        "runner",
        "iterations",
        "mcgs",
        "search_mode",
        "max_nodes",
        "max_tokens",
        "max_wall_seconds",
        "max_cost",
    )
    if not contract_payload or any(
        request_contract.get(key) != contract_payload.get(key) for key in request_to_input_keys
    ):
        errors.append("approval request_contract is not bound to input-contract.json")

    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "plan_id": payload.get("plan_id"),
        "plan_sha256": plan_sha256,
        "request_contract_sha256": request_contract_sha256,
        "receipt_sha256": receipt_sha256,
        "created_at": created_at.isoformat() if created_at else None,
        "approved_at": approved_at.isoformat() if approved_at else None,
        "expires_at": expires_at.isoformat() if expires_at else None,
    }


def verify_cycle_receipt(
    receipt: Mapping[str, Any] | None,
    approval_receipt: Mapping[str, Any] | None,
    approval_result: Mapping[str, Any],
    run_dir: Path,
    *,
    db_run_id: str | None,
) -> dict[str, Any]:
    """Verify plan/request/file hashes and the exact Prisma run binding."""

    errors: list[str] = []
    payload = dict(receipt) if isinstance(receipt, Mapping) else {}
    approval = dict(approval_receipt) if isinstance(approval_receipt, Mapping) else {}
    if not payload:
        errors.append("cycle receipt is missing")
    if payload.get("schema") != CYCLE_RECEIPT_SCHEMA:
        errors.append("cycle receipt schema mismatch")
    if payload.get("task_id") != EXPECTED_TASK_ID:
        errors.append("cycle receipt task_id mismatch")
    if payload.get("output_dir_name") != run_dir.name:
        errors.append("cycle receipt output directory mismatch")
    if payload.get("status") != "completed":
        errors.append("cycle receipt is not completed")
    if payload.get("official_submit_allowed") is not False:
        errors.append("cycle receipt submission boundary mismatch")

    linked_fields = ("plan_id", "plan_sha256", "request_fingerprint")
    if any(payload.get(key) != approval.get(key) for key in linked_fields):
        errors.append("cycle receipt plan/request fields do not match approval receipt")
    if payload.get("request_contract_sha256") != approval_result.get("request_contract_sha256"):
        errors.append("cycle receipt request contract hash mismatch")
    if payload.get("approval_receipt_sha256") != approval_result.get("receipt_sha256"):
        errors.append("cycle receipt approval canonical hash mismatch")
    if approval_result.get("status") != "passed":
        errors.append("cycle receipt is linked to an invalid approval receipt")

    required_names = {
        "input-contract.json",
        "cli-receipt.json",
        "candidate-freeze.json",
        "independent-review.json",
        "claim-audit.json",
        "approval-receipt.json",
    }
    observed_hashes = payload.get("artifact_hashes")
    observed_hashes = observed_hashes if isinstance(observed_hashes, dict) else {}
    actual_hashes: dict[str, str | None] = {}
    for name in sorted(required_names):
        path = run_dir / name
        actual_hashes[name] = sha256_path(path) if _regular_file(path) else None
    if set(observed_hashes) != required_names:
        errors.append("cycle receipt artifact hash set mismatch")
    if any(observed_hashes.get(name) != actual_hashes[name] for name in required_names):
        errors.append("cycle receipt artifact SHA-256 mismatch")

    if not isinstance(db_run_id, str) or not db_run_id:
        errors.append("exact Prisma experiment run is unavailable")
    elif payload.get("run_id") != db_run_id:
        errors.append("cycle receipt run_id does not match exact Prisma experiment run")

    completed_at = _iso_timestamp(payload.get("completed_at"))
    approved_at = _iso_timestamp(approval.get("approved_at"))
    input_path = run_dir / "input-contract.json"
    input_created_at = None
    if _regular_file(input_path):
        input_created_at = _iso_timestamp(read_json(input_path).get("created_at"))
    if not (
        completed_at is not None
        and approved_at is not None
        and input_created_at is not None
        and approved_at <= input_created_at <= completed_at
    ):
        errors.append("approval, input, and cycle timestamps are missing, invalid, or out of order")

    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "run_id": payload.get("run_id"),
        "db_run_id": db_run_id,
        "cycle_receipt_sha256": sha256_path(run_dir / "cycle-receipt.json")
        if _regular_file(run_dir / "cycle-receipt.json")
        else None,
        "artifact_hashes": actual_hashes,
        "completed_at": completed_at.isoformat() if completed_at else None,
    }


def verify_demo_campaign_evidence(
    run_dir: Path,
    campaign_dir: Path,
    *,
    cli_log: Path | None = None,
    input_contract: Path | None = None,
    workstation_db: Path | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    campaign_dir = campaign_dir.resolve()
    evidence_paths = resolve_evidence_paths(
        run_dir,
        cli_log=cli_log,
        input_contract=input_contract,
    )
    required_json = {
        name: run_dir / name
        for name in (
            "summary.json",
            "search-run-contract.json",
            "search_graph.json",
            "experience-board.json",
            "budget-ledger.json",
            "summary-cache-stats.json",
            "candidate-freeze.json",
            "independent-review.json",
            "claim-audit.json",
            "model_provenance.json",
        )
    }
    required_jsonl = {
        name: run_dir / name
        for name in ("evolution-events.jsonl", "selection-traces.jsonl", "retrieval-bundles.jsonl")
    }
    missing = [name for name, path in {**required_json, **required_jsonl}.items() if not path.is_file()]
    if missing:
        return {
            "schema": SCHEMA,
            "status": "failed_closed",
            "recording_ready": False,
            "run_id": run_dir.name,
            "task_id": EXPECTED_TASK_ID,
            "checks": [_new_check("required_artifacts", False, detail=f"Missing: {', '.join(missing)}")],
            "blocking_findings": [f"Missing required evidence: {', '.join(missing)}"],
        }

    payloads = {name: read_json(path) for name, path in required_json.items()}
    events = read_jsonl(required_jsonl["evolution-events.jsonl"])
    traces = read_jsonl(required_jsonl["selection-traces.jsonl"])
    retrievals = read_jsonl(required_jsonl["retrieval-bundles.jsonl"])
    summary = payloads["summary.json"]
    contract = payloads["search-run-contract.json"]
    graph = payloads["search_graph.json"]
    board = payloads["experience-board.json"]
    budget = payloads["budget-ledger.json"]
    cache_stats = payloads["summary-cache-stats.json"]
    freeze = payloads["candidate-freeze.json"]
    review = payloads["independent-review.json"]
    claim = payloads["claim-audit.json"]
    provenance = payloads["model_provenance.json"]
    dataset_manifest = read_json(campaign_dir / "dataset-manifest.json")
    board_result = verify_board(board)
    cards = board.get("cards") if isinstance(board.get("cards"), list) else []
    crossover_result = verify_dual_parent_crossover(cards, graph, traces, events)
    cache_result = verify_cache_evidence(retrievals, cache_stats)
    checks: list[dict[str, Any]] = []

    task_ids = {
        summary.get("task"), contract.get("task_id"), graph.get("task_id"), board.get("task_id"),
        freeze.get("task_id"), dataset_manifest.get("task_id"),
    }
    checks.append(_new_check(
        "task_and_run_identity",
        task_ids == {EXPECTED_TASK_ID} and freeze.get("run_id") == run_dir.name,
        detail="All task/run identity fields must bind to the isolated demo namespace.",
        evidence={"task_ids": sorted(str(item) for item in task_ids), "run_id": run_dir.name},
    ))

    summary_iterations = summary.get("iterations") if isinstance(summary.get("iterations"), list) else []
    summary_ids = [str(item.get("exp_id")) for item in summary_iterations if isinstance(item, dict)]
    graph_nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
    graph_ids = [str(item.get("exp_id")) for item in graph_nodes if isinstance(item, dict)]
    exec_begin = [item for item in events if item.get("type") == "exec_begin"]
    scores = [item for item in events if item.get("type") == "score"]
    solution_ids = sorted(path.parent.name for path in (run_dir / "runs").glob("EXP*/solution.py"))
    validation_ids = sorted(path.parent.name for path in run_dir.glob("EXP*/validation_contract.json"))
    cpu_execution_ok = (
        summary.get("n_iterations") == 8
        and summary_ids == EXPECTED_EXPERIMENT_IDS
        and graph_ids == EXPECTED_EXPERIMENT_IDS
        and sorted(str(item.get("exp_id")) for item in exec_begin) == EXPECTED_EXPERIMENT_IDS
        and sorted(str(item.get("exp_id")) for item in scores) == EXPECTED_EXPERIMENT_IDS
        and solution_ids == EXPECTED_EXPERIMENT_IDS
        and validation_ids == EXPECTED_EXPERIMENT_IDS
        and all(item.get("runner") == "LocalSubprocessRunner" for item in exec_begin)
        and contract.get("runner") == "local"
        and contract.get("evidence_class") == "real_local_cpu_execution"
        and contract.get("demo_campaign") is True
    )
    success_count = sum(item.get("success") is True for item in summary_iterations if isinstance(item, dict))
    failure_count = sum(item.get("success") is False for item in summary_iterations if isinstance(item, dict))
    checks.append(_new_check(
        "eight_real_local_cpu_candidates",
        cpu_execution_ok and success_count == 7 and failure_count == 1,
        detail="Eight local subprocess candidates must execute; the deliberate failure remains evidence for the Debug repair round.",
        evidence={"executed": len(exec_begin), "successful": success_count, "failed": failure_count},
    ))

    operators = {str(card.get("operator")) for card in cards if isinstance(card, dict)}
    checks.append(_new_check(
        "four_operator_timeline",
        len(cards) == 8 and operators == EXPECTED_OPERATORS,
        detail="The Experience Board must contain Draft, Improve, Debug and Crossover across eight cards.",
        evidence={"operators": sorted(operators), "operator_counts": board_result["operator_counts"]},
    ))
    checks.append(_new_check(
        "board_append_order_hash_chain",
        board_result["status"] == "passed",
        detail="Every content-derived card id, append position, chain link and board hash was recomputed independently.",
        evidence=board_result,
    ))
    private_feedback_findings = find_forbidden_evaluation_values(
        {"board": board, "selection_traces": traces, "retrieval_bundles": retrievals}
    )
    checks.append(_new_check(
        "public_validation_only_experience",
        not private_feedback_findings,
        detail="Experience Board, SelectionTrace and RetrievalBundle values must remain free of private/leaderboard/official-score feedback.",
        evidence={"forbidden_paths": private_feedback_findings[:20], "finding_count": len(private_feedback_findings)},
    ))
    checks.append(_new_check(
        "true_dual_parent_crossover",
        crossover_result["status"] == "passed",
        detail="Crossover must use two distinct successful method families across Card, graph, trace and event evidence.",
        evidence=crossover_result,
    ))

    costs = [card.get("cost") for card in cards if isinstance(card, dict) and isinstance(card.get("cost"), dict)]
    cost_totals = {
        key: sum(float(cost.get(key) or 0) for cost in costs)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "wall_seconds", "gpu_seconds", "estimated_cost_usd")
    }
    budget_matches = (
        budget.get("nodes") == 8
        and budget.get("max_nodes") == 8
        and budget.get("terminal_reason") == "node_budget_exhausted"
        and budget.get("can_continue") is False
        and summary.get("terminal_reason") == budget.get("terminal_reason") == graph.get("terminal_reason")
        and all(abs(float(budget.get(key) or 0) - value) < 1e-8 for key, value in cost_totals.items())
        and summary.get("budget") == budget
        and _finite_number(budget.get("wall_seconds"))
        and 0 < float(budget["wall_seconds"]) <= float(budget.get("max_wall_seconds") or 0)
        and float(budget.get("gpu_seconds") or 0) == 0.0
    )
    checks.append(_new_check(
        "budget_ledger",
        budget_matches,
        detail="Per-card costs must sum to the terminal BudgetLedger and end at the eight-node hard limit.",
        evidence={"ledger": budget, "recomputed_cost": cost_totals},
    ))

    checks.append(_new_check(
        "cache_telemetry_consistency",
        cache_result["summary_stats_consistent"] and len(retrievals) == 7,
        detail="Summary-cache counters must equal the per-retrieval card-cache counters.",
        evidence=cache_result,
    ))
    checks.append(_new_check(
        "cache_miss_observed",
        cache_result["miss_observed"],
        detail="At least one real summary-card cache miss must be observed before recording memory behavior.",
        evidence=cache_result,
    ))
    checks.append(_new_check(
        "cache_hit_observed",
        cache_result["hit_observed"],
        detail="At least one real summary-card cache hit must be observed before claiming cache reuse.",
        evidence=cache_result,
    ))
    checks.append(_new_check(
        "prompt_memory_retrieval_used",
        cache_result["memory_used"],
        detail="At least one retrieval bundle must contain a verified Experience Card reference.",
        evidence=cache_result,
    ))

    dataset_entries = dataset_manifest.get("files") if isinstance(dataset_manifest.get("files"), list) else []
    dataset_hashes: dict[str, str] = {}
    dataset_ok = True
    for entry in dataset_entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            dataset_ok = False
            continue
        path = _safe_artifact_path(campaign_dir, entry["path"])
        actual = sha256_path(path) if path.is_file() else ""
        dataset_hashes[entry["path"]] = actual
        dataset_ok &= actual == entry.get("sha256") and path.stat().st_size == entry.get("bytes")
    train_hash = dataset_hashes.get("data/train.csv")
    candidate_artifacts_ok = True
    candidate_hash_count = 0
    for card in cards:
        if not isinstance(card, dict):
            continue
        solution = run_dir / "runs" / str(card.get("node_id")) / "solution.py"
        # ExperienceCardBuilder hashes the canonical generated source string.
        # On Windows the persisted file can contain CRLF bytes, so compare the
        # decoded text (universal-newline normalized by Path.read_text) rather
        # than incorrectly treating code_hash as a raw file checksum.
        candidate_artifacts_ok &= (
            solution.is_file()
            and sha256_bytes(solution.read_text(encoding="utf-8").encode("utf-8")) == card.get("code_hash")
        )
        provenance_record = card.get("provenance") if isinstance(card.get("provenance"), dict) else {}
        artifacts = provenance_record.get("artifact_hashes") if isinstance(provenance_record.get("artifact_hashes"), list) else []
        for artifact in artifacts:
            if not isinstance(artifact, dict) or not isinstance(artifact.get("name"), str):
                candidate_artifacts_ok = False
                continue
            path = run_dir / str(card.get("node_id")) / "out" / artifact["name"]
            candidate_artifacts_ok &= (
                path.is_file()
                and sha256_path(path) == artifact.get("sha256")
                and path.stat().st_size == artifact.get("size")
            )
            candidate_hash_count += 1
        metrics_path = run_dir / str(card.get("node_id")) / "out" / "metrics.json"
        if metrics_path.is_file():
            candidate_artifacts_ok &= read_json(metrics_path).get("train_sha256") == train_hash
    checks.append(_new_check(
        "dataset_and_candidate_hash_binding",
        dataset_ok and candidate_artifacts_ok and candidate_hash_count == 21,
        detail="Dataset manifest, candidate source hashes, 21 successful output hashes and each metrics train hash must agree.",
        evidence={"dataset_hashes": dataset_hashes, "candidate_artifact_hashes_verified": candidate_hash_count},
    ))
    card_data_hashes_present = all(bool(card.get("data_hashes")) for card in cards if isinstance(card, dict))
    checks.append(_new_check(
        "card_level_data_hash_provenance",
        card_data_hashes_present and bool(contract.get("public_data_hashes")),
        blocking=False,
        detail="Card and run-contract public_data_hashes are expected for portable provenance; legacy demo evidence can still be bound through metrics files.",
        evidence={
            "run_contract_public_data_hashes": contract.get("public_data_hashes"),
            "cards_with_data_hashes": sum(bool(card.get("data_hashes")) for card in cards if isinstance(card, dict)),
        },
    ))

    best_id = summary.get("best_exp_id")
    best_score = summary.get("best_cv_score")
    freeze_file_hashes: dict[str, str] = {}
    freeze_ok = (
        freeze.get("status") == "frozen_before_independent_review"
        and freeze.get("best_exp_id") == best_id
        and freeze.get("public_validation_score") == best_score
        and freeze.get("official_submission_performed") is False
        and freeze.get("private_grader_used") is False
    )
    for entry in freeze.get("files", []) if isinstance(freeze.get("files"), list) else []:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            freeze_ok = False
            continue
        path = _safe_artifact_path(run_dir, entry["path"])
        actual = sha256_path(path) if path.is_file() else ""
        freeze_file_hashes[entry["path"]] = actual
        freeze_ok &= actual == entry.get("sha256") and path.stat().st_size == entry.get("bytes")
    best_solution = run_dir / "best_solution.py"
    source_solution = run_dir / "runs" / str(best_id) / "solution.py"
    freeze_ok &= best_solution.is_file() and source_solution.is_file() and sha256_path(best_solution) == sha256_path(source_solution)
    freeze_hash = sha256_path(run_dir / "candidate-freeze.json")
    checks.append(_new_check(
        "candidate_freeze",
        freeze_ok and len(freeze_file_hashes) == 3,
        detail="The best public-validation candidate must be frozen by immutable file hashes before review.",
        evidence={"best_exp_id": best_id, "public_validation_score": best_score, "freeze_sha256": freeze_hash, "files": freeze_file_hashes},
    ))

    labels_entry = next(
        (entry for entry in dataset_entries if isinstance(entry, dict) and entry.get("path") == "review/holdout_labels.csv"),
        {},
    )
    review_path = run_dir / "independent-review.json"
    review_hash = sha256_path(review_path)
    review_ok = (
        review.get("status") == "passed"
        and review.get("reviewer") == "IndependentDemoReviewer"
        and review.get("candidate_freeze_sha256") == freeze_hash
        and review.get("submission_sha256") == freeze_file_hashes.get(f"{best_id}/out/submission.csv")
        and review.get("sealed_labels_sha256") == labels_entry.get("sha256")
        and review.get("rows") == dataset_manifest.get("sealed_holdout_rows") == 600
        and review.get("model_reload_passed") is True
        and review.get("source") == "sealed_synthetic_holdout"
        and review.get("official_submission_performed") is False
        and _finite_number(review.get("score"))
        and 0.0 <= float(review["score"]) <= 1.0
    )
    checks.append(_new_check(
        "independent_review_linkage",
        review_ok,
        detail="The existing sealed-holdout review must link to the exact freeze, submission and labels hashes; no regrading is performed here.",
        evidence={"review_sha256": review_hash, "reviewer": review.get("reviewer"), "score": review.get("score"), "rows": review.get("rows")},
    ))

    claim_ok = (
        claim.get("status") == "passed"
        and claim.get("unsupported_claims") == []
        and isinstance(claim.get("supported_claims"), list)
        and len(claim["supported_claims"]) == 3
        and isinstance(claim.get("evidence"), dict)
        and claim["evidence"].get("candidate_freeze_sha256") == freeze_hash
        and claim["evidence"].get("independent_review_sha256") == review_hash
        and "not MLE-Bench" in str(claim.get("claim_boundary"))
        and "official medal" in str(claim.get("claim_boundary"))
    )
    checks.append(_new_check(
        "claim_audit",
        claim_ok,
        detail="Every public demo claim must be supported and bounded away from MLE-Bench, Kaggle, SIIM grading and official medals.",
        evidence={"supported_claims": claim.get("supported_claims"), "unsupported_claims": claim.get("unsupported_claims"), "claim_boundary": claim.get("claim_boundary")},
    ))

    event_times = [_iso_timestamp(item.get("ts"), assume_local=True) for item in events]
    event_times_valid = all(item is not None for item in event_times)
    normalized_event_times = [item for item in event_times if item is not None]
    events_monotonic = all(left <= right for left, right in zip(normalized_event_times, normalized_event_times[1:]))
    freeze_time = _iso_timestamp(freeze.get("frozen_at"))
    review_time = _iso_timestamp(review.get("reviewed_at"))
    run_end = next((item for item in reversed(events) if item.get("type") == "run_end"), {})
    run_end_time = _iso_timestamp(run_end.get("ts"), assume_local=True)
    timeline_ok = (
        event_times_valid
        and events_monotonic
        and run_end_time is not None
        and freeze_time is not None
        and review_time is not None
        and run_end_time <= freeze_time <= review_time
        and [item.get("seq") for item in events] == list(range(1, len(events) + 1))
        and run_end.get("best_exp_id") == best_id
        and run_end.get("n_iterations") == 8
    )
    checks.append(_new_check(
        "run_time_consistency",
        timeline_ok,
        detail="Event sequence, run end, freeze and review timestamps must be monotonic for the same run.",
        evidence={"event_count": len(events), "run_end_utc": run_end_time.isoformat() if run_end_time else None, "freeze_utc": freeze_time.isoformat() if freeze_time else None, "review_utc": review_time.isoformat() if review_time else None},
    ))

    observed_generators = provenance.get("observed_generators") if isinstance(provenance.get("observed_generators"), list) else []
    provenance_ok = (
        provenance.get("status") == "passed"
        and len(observed_generators) == 8
        and all(
            isinstance(item, dict)
            and item.get("provider") == "deterministic_demo_plan"
            and item.get("model") == "evomind.demo.verified_sklearn_plan.v1"
            for item in observed_generators
        )
    )
    checks.append(_new_check(
        "generator_provenance",
        provenance_ok,
        detail="All eight proposals must identify the deterministic, isolated demo generator.",
        evidence={"generator_count": len(observed_generators), "status": provenance.get("status")},
    ))
    environment_known = all(
        isinstance(card.get("provenance"), dict)
        and card["provenance"].get("environment_hash") not in {None, "", "not_recorded"}
        and card["provenance"].get("evaluator_version") not in {None, "", "unknown"}
        for card in cards
        if isinstance(card, dict)
    )
    checks.append(_new_check(
        "execution_environment_provenance",
        environment_known,
        blocking=False,
        detail="Environment hash and evaluator version should be captured before using the demo as portable benchmark evidence.",
        evidence={"cards": len(cards), "environment_provenance_complete": environment_known},
    ))

    cli_receipt_path = evidence_paths["cli_receipt"]
    cli_payload: dict[str, Any] | None = None
    if _regular_file(cli_receipt_path):
        cli_payload = read_json(cli_receipt_path)
        modern_cli_required = cli_log is None or "schema" in cli_payload
        modern_cli_ok = (
            cli_payload.get("schema") == CLI_RECEIPT_SCHEMA
            and Path(str(cli_payload.get("input_contract_path") or "")).name
            == "input-contract.json"
            and cli_payload.get("input_contract_sha256")
            == (
                sha256_path(evidence_paths["input_contract"])
                if _regular_file(evidence_paths["input_contract"])
                else None
            )
        )
        cli_ok = (
            cli_payload.get("ok") is True
            and cli_payload.get("task_id") == EXPECTED_TASK_ID
            and Path(str(cli_payload.get("exp_dir"))).name == run_dir.name
            and cli_payload.get("best_exp_id") == best_id
            and cli_payload.get("best_cv_score") == best_score
            and cli_payload.get("n_iterations") == 8
            and cli_payload.get("terminal_reason") == "node_budget_exhausted"
            and cli_payload.get("demo_campaign") is True
            and isinstance(cli_payload.get("demo_evidence"), dict)
            and cli_payload["demo_evidence"].get("candidate_freeze_sha256") == freeze_hash
            and (not modern_cli_required or modern_cli_ok)
        )
        checks.append(_new_check(
            "cli_receipt_consistency", cli_ok,
            detail="The CLI receipt must identify the same run, input contract, best candidate, terminal state and freeze hash.",
            evidence={
                "path": str(cli_receipt_path),
                "cli_receipt_sha256": sha256_path(cli_receipt_path),
                "auto_discovered": cli_log is None,
                "modern_contract_verified": modern_cli_ok,
            },
        ))
    else:
        checks.append(_new_check("cli_receipt_consistency", False, detail="CLI receipt is missing."))

    input_contract_path = evidence_paths["input_contract"]
    input_payload: dict[str, Any] | None = None
    if _regular_file(input_contract_path):
        input_payload = read_json(input_contract_path)
        modern_input_required = input_contract is None or "schema" in input_payload
        modern_input_ok = (
            input_payload.get("schema") == INPUT_CONTRACT_SCHEMA
            and input_payload.get("demo_campaign") is True
            and input_payload.get("official_submission") == "disabled"
            and _iso_timestamp(input_payload.get("created_at")) is not None
        )
        input_ok = (
            input_payload.get("task_id") == EXPECTED_TASK_ID
            and input_payload.get("iterations") == contract.get("effective_iterations") == 8
            and input_payload.get("max_nodes") == budget.get("max_nodes") == 8
            and input_payload.get("max_tokens") == budget.get("max_total_tokens")
            and _numbers_equal(input_payload.get("max_wall_seconds"), budget.get("max_wall_seconds"))
            and _numbers_equal(input_payload.get("max_cost"), budget.get("max_cost_usd"))
            and input_payload.get("runner") == contract.get("runner") == "local"
            and input_payload.get("search_mode") == contract.get("search_mode") == "experience_mcgs_v1"
            and (not modern_input_required or modern_input_ok)
        )
        checks.append(_new_check(
            "input_contract_consistency", input_ok,
            detail="The predeclared demo input must match the executed run contract and BudgetLedger.",
            evidence={
                "path": str(input_contract_path),
                "input_contract_sha256": sha256_path(input_contract_path),
                "auto_discovered": input_contract is None,
                "modern_contract_verified": modern_input_ok,
            },
        ))
    else:
        checks.append(_new_check("input_contract_consistency", False, detail="Demo input contract is missing."))

    approval_receipt_path = evidence_paths["approval_receipt"]
    approval_payload = (
        read_json(approval_receipt_path) if _regular_file(approval_receipt_path) else None
    )
    approval_result = verify_approval_receipt(approval_payload, input_payload)
    checks.append(_new_check(
        "approval_receipt_binding",
        approval_result["status"] == "passed",
        detail="The approved one-time plan must bind canonical plan, request and input-contract hashes.",
        evidence={
            **approval_result,
            "path": str(approval_receipt_path),
            "file_sha256": sha256_path(approval_receipt_path)
            if _regular_file(approval_receipt_path)
            else None,
        },
    ))

    cycle_receipt_path = evidence_paths["cycle_receipt"]
    cycle_payload = read_json(cycle_receipt_path) if _regular_file(cycle_receipt_path) else None

    bound_db_run_id: str | None = None
    if workstation_db and workstation_db.is_file():
        db_path = workstation_db.resolve()
        connection = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            rows = [
                dict(row)
                for row in connection.execute(
                    "SELECT id, output_dir, status, validation_status, created_at "
                    "FROM experiment_runs WHERE task_id = ? ORDER BY created_at DESC LIMIT 30",
                    (EXPECTED_TASK_ID,),
                )
            ]
        finally:
            connection.close()
        projected_rows = [
            {
                "id": row.get("id"),
                "output_dir_name": Path(str(row.get("output_dir") or "")).name,
                "status": row.get("status"),
                "validation_status": row.get("validation_status"),
                "created_at": row.get("created_at"),
            }
            for row in rows
        ]
        bound = next((row for row in projected_rows if row["output_dir_name"] == run_dir.name), None)
        latest = projected_rows[0] if projected_rows else None
        bound_db_run_id = str(bound.get("id")) if bound and bound.get("id") else None
        ui_binding_ok = (
            bound is not None
            and bound.get("status") == "passed"
            and bound.get("validation_status") == "passed"
            and latest is not None
            and latest.get("id") == bound.get("id")
        )
        checks.append(_new_check(
            "ui_run_binding",
            ui_binding_ok,
            detail="The exact demo must be the newest passed experiment_runs row so the Evolution UI cannot bind an older run.",
            evidence={"bound": bound, "latest": latest, "rows_considered": len(projected_rows)},
        ))
    else:
        checks.append(_new_check(
            "ui_run_binding",
            False,
            detail="The read-only workstation SQLite path is missing, so the Evolution UI binding cannot be verified.",
        ))

    cycle_result = verify_cycle_receipt(
        cycle_payload,
        approval_payload,
        approval_result,
        run_dir,
        db_run_id=bound_db_run_id,
    )
    checks.append(_new_check(
        "cycle_receipt_binding",
        cycle_result["status"] == "passed",
        detail="The completed cycle must bind the approved plan, request contract, six artifact hashes and exact Prisma run ID.",
        evidence={**cycle_result, "path": str(cycle_receipt_path)},
    ))

    official_or_grader_clean = (
        contract.get("official_submission") == "disabled"
        and freeze.get("official_submission_performed") is False
        and freeze.get("private_grader_used") is False
        and review.get("official_submission_performed") is False
    )
    checks.append(_new_check(
        "no_submission_no_private_grader",
        official_or_grader_clean,
        detail="The demo must remain isolated from Kaggle submission and private graders.",
        evidence={"official_submission": contract.get("official_submission"), "private_grader_used": freeze.get("private_grader_used")},
    ))

    blocking_failures = [check for check in checks if check["blocking"] and check["status"] != "passed"]
    warnings = [check for check in checks if not check["blocking"] and check["status"] != "passed"]
    status = "passed" if not blocking_failures else "failed_closed"
    report_artifacts = {**required_json, **required_jsonl}
    report_artifacts.update(
        {
            "cli-receipt.json": cli_receipt_path,
            "input-contract.json": input_contract_path,
            "approval-receipt.json": approval_receipt_path,
            "cycle-receipt.json": cycle_receipt_path,
        }
    )
    artifact_hashes = {
        name: sha256_path(path) for name, path in report_artifacts.items() if _regular_file(path)
    }
    return {
        "schema": SCHEMA,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "recording_ready": status == "passed",
        "run_id": run_dir.name,
        "task_id": EXPECTED_TASK_ID,
        "evidence_class": "real_local_cpu_execution",
        "candidate_execution": {"scheduled_and_executed": 8, "successful": success_count, "failed": failure_count},
        "best_candidate": {"exp_id": best_id, "public_validation_score": best_score, "independent_review_score": review.get("score")},
        "operator_counts": board_result["operator_counts"],
        "board": {
            "card_count": board_result["card_count"],
            "append_chain_head": board_result["append_chain_head"],
            "board_hash": board_result["board_hash"],
            "lineage_edge_count": board_result["lineage_edge_count"],
        },
        "cache": cache_result,
        "budget": budget,
        "freeze_sha256": freeze_hash,
        "independent_review_sha256": review_hash,
        "claim_audit_sha256": sha256_path(run_dir / "claim-audit.json"),
        "checks": checks,
        "blocking_findings": [check["detail"] for check in blocking_failures],
        "warnings": [check["detail"] for check in warnings],
        "artifact_hashes": dict(sorted(artifact_hashes.items())),
        "claim_boundary": claim.get("claim_boundary"),
        "verifier_actions": [
            "read_existing_evidence",
            "recompute_hashes_and_cross_artifact_links",
            "no_candidate_execution",
            "no_model_reload",
            "no_regrading",
            "no_private_grader",
            "no_kaggle_submission",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--cli-log", type=Path)
    parser.add_argument("--input-contract", type=Path)
    parser.add_argument("--workstation-db", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = verify_demo_campaign_evidence(
            args.run_dir,
            args.campaign_dir,
            cli_log=args.cli_log,
            input_contract=args.input_contract,
            workstation_db=args.workstation_db,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        report = {
            "schema": SCHEMA,
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "status": "failed_closed",
            "recording_ready": False,
            "run_id": args.run_dir.name,
            "task_id": EXPECTED_TASK_ID,
            "checks": [_new_check("verifier_execution", False, detail=str(error))],
            "blocking_findings": [str(error)],
        }
    output = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(output, encoding="utf-8")
        temporary.replace(args.output)
    sys.stdout.write(output)
    return 0 if report.get("recording_ready") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
