#!/usr/bin/env python3
"""Safe JSON-in / JSON-out adapter that exposes the evolution engine brain to the
Next.js workstation API.

Design constraints (see integration brief):
  * No shell string interpolation: the caller passes ``--input <file>`` and we
    read a JSON document. We never eval or exec caller input.
  * No secrets: this module never reads or prints tokens / keys / cookies. The
    light research_os modules it imports have no import-time side effects.
  * No bypass training: ``step`` only PLANS and records governance artifacts in
    dry_run mode. Real training must go through the workstation orchestrator; a
    non-dry-run step returns a ``blocked_use_workstation`` decision.
  * Always emits valid JSON on stdout, even for empty / missing state.

Modes:  state | plan | step | graph | memory

Every artifact is written under ``workspace/evolution/<task_id>/`` so the web
layer can read it back. Cross-task retrospective memory is the SHARED real store
at ``experiments/evolution/retrospective_memory.json`` (reused, not forked).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (SRC, ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

# Light research_os modules only — none touch the network or read secrets on import.
from research_os.search_graph import ExperimentNode, SearchGraph  # noqa: E402
from research_os.mcgs_selector import MCGSSelector  # noqa: E402
from research_os.strategy_selector import TaskProfile, recommend_strategies  # noqa: E402
from research_os.retrospective_memory import MemoryRecord, RetrospectiveMemoryStore  # noqa: E402
from research_os.validation_contract import (  # noqa: E402
    check_required_artifacts,
    create_contract,
    evaluate_acceptance,
)
from research_os.claim_audit import audit_claim  # noqa: E402

CLAIM_BOUNDARY = (
    "Local CV / proxy metric only. No official Kaggle rank, percentile or medal "
    "is claimed without a real Kaggle response artifact. Official submission is "
    "disabled; real training must run through the workstation orchestrator."
)
REQUIRED_ARTIFACTS = ["metrics.json", "submission.csv"]
SHARED_MEMORY = ROOT / "experiments" / "evolution" / "retrospective_memory.json"


def _safe_task_id(task_id: str) -> str:
    """Whitelist the task id so it can only ever be a single path segment."""
    tid = (task_id or "").strip()
    if not tid or not re.fullmatch(r"[A-Za-z0-9._\-]+", tid) or ".." in tid:
        raise ValueError(f"unsafe or empty task_id: {task_id!r}")
    return tid


def _task_dir(task_id: str) -> Path:
    return ROOT / "workspace" / "evolution" / task_id


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8").replace("﻿", ""))
    except Exception:
        return None


def _direction_lower(direction: str) -> bool:
    return (direction or "maximize").lower() in {"minimize", "lower", "lower_is_better"}
def _graph_path(task_id: str) -> Path:
    return _task_dir(task_id) / "search_graph.json"


def _load_graph(task_id: str, *, metric_direction: str = "maximize",
                metric_name: str = "cv_score") -> SearchGraph:
    """Load the persisted graph for a task, or build a fresh empty one."""
    graph = SearchGraph(
        task_id=task_id, root_exp_id="EXP000",
        metric_name=metric_name, metric_direction=metric_direction,
    )
    data = _read_json(_graph_path(task_id))
    if not isinstance(data, dict):
        return graph
    graph.metric_name = data.get("metric_name", metric_name)
    graph.metric_direction = data.get("metric_direction", metric_direction)
    graph.root_exp_id = data.get("root_exp_id", "EXP000")
    graph.best_exp_id = data.get("best_exp_id")
    graph.selected_next_branch = data.get("selected_next_branch")
    graph.exploration_stage = data.get("exploration_stage", "exploration")
    graph.promotion_history = list(data.get("promotion_history", []))
    node_fields = set(ExperimentNode.__dataclass_fields__.keys())
    for raw in data.get("nodes", []):
        if not isinstance(raw, dict) or "exp_id" not in raw:
            continue
        clean = {k: v for k, v in raw.items() if k in node_fields}
        clean.setdefault("parent_id", None)
        clean.setdefault("branch_type", "Base")
        clean.setdefault("task_name", task_id)
        clean.setdefault("hypothesis", "")
        clean.setdefault("implementation_summary", "")
        clean.setdefault("code_path", "")
        try:
            graph.add_node(ExperimentNode(**clean))
        except TypeError:
            continue
    for edge in data.get("edges", []):
        if isinstance(edge, dict) and edge.get("source") in graph.nodes and edge.get("target") in graph.nodes:
            graph.add_edge(edge["source"], edge["target"], edge.get("reason", ""))
    return graph


def _next_exp_id(graph: SearchGraph) -> str:
    n = len(graph.nodes)
    while f"EXP{n:03d}" in graph.nodes:
        n += 1
    return f"EXP{n:03d}"


def _profile_from_input(data: dict) -> TaskProfile:
    return TaskProfile(
        modality=data.get("modality", "tabular"),
        task_type=data.get("task_type", "classification"),
        train_size=int(data.get("n_train", data.get("train_size", 0)) or 0),
        test_size=int(data.get("n_test", data.get("test_size", 0)) or 0),
        metric=data.get("metric", "accuracy"),
        n_features=int(data.get("n_features", 0) or 0),
        n_high_cardinality_features=int(data.get("n_high_cardinality_features", 0) or 0),
        n_model_families=int(data.get("n_model_families", 3) or 3),
        has_time_column=bool(data.get("has_time_column", False)),
        target_is_positive=bool(data.get("target_is_positive", False)),
    )
def mode_graph(data: dict) -> dict:
    task_id = _safe_task_id(data.get("task_id", ""))
    graph = _load_graph(
        task_id,
        metric_direction=data.get("metric_direction", "maximize"),
        metric_name=data.get("metric_name", "cv_score"),
    )
    payload = graph.to_dict()
    payload["node_count"] = len(graph.nodes)
    payload["has_run"] = bool(graph.nodes)
    payload["claim_boundary"] = CLAIM_BOUNDARY
    return payload


def mode_state(data: dict) -> dict:
    task_id = _safe_task_id(data.get("task_id", ""))
    graph = _load_graph(
        task_id,
        metric_direction=data.get("metric_direction", "maximize"),
        metric_name=data.get("metric_name", "cv_score"),
    )
    tdir = _task_dir(task_id)
    plan = _read_json(tdir / "latest_plan.json")
    latest_step = _read_json(tdir / "latest_step.json")
    best_node = graph.nodes.get(graph.best_exp_id) if graph.best_exp_id else None
    # active branches: leaf frontier (nodes with no children)
    parents = {n.parent_id for n in graph.nodes.values() if n.parent_id}
    active_branches = [
        {
            "exp_id": n.exp_id,
            "branch_type": n.branch_type,
            "cv_score": n.cv_score,
            "promoted": n.promoted,
        }
        for n in graph.nodes.values() if n.exp_id not in parents
    ]
    memory_hits = _memory_records(task_type=data.get("task_type", ""))
    last_artifacts = []
    if tdir.exists():
        for name in ("search_graph.json", "latest_plan.json", "latest_step.json",
                     "validation_contract.json", "claim_audit.json"):
            if (tdir / name).exists():
                last_artifacts.append(f"workspace/evolution/{task_id}/{name}")
    stage = graph.exploration_stage if graph.nodes else "no_evolution_run_yet"
    risk_flags = sorted({flag for n in graph.nodes.values() for flag in (n.risk_flags or [])})
    if graph.detect_global_stagnation():
        risk_flags.append("global_stagnation")
    return {
        "task_id": task_id,
        "current_stage": stage,
        "has_run": bool(graph.nodes),
        "best_so_far": {
            "exp_id": graph.best_exp_id,
            "cv_score": best_node.cv_score if best_node else None,
            "metric": graph.metric_name,
            "metric_direction": graph.metric_direction,
            "promotion_reason": best_node.promotion_reason if best_node else "",
        },
        "search_graph_summary": {
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "top_candidates": graph.to_dict().get("top_candidates", []),
            "exploration_stage": graph.exploration_stage,
            "global_stagnation": graph.detect_global_stagnation(),
        },
        "active_branches": active_branches,
        "latest_decision": latest_step.get("decision") if isinstance(latest_step, dict) else (
            plan.get("search_controller_decision") if isinstance(plan, dict) else None
        ),
        "memory_hits": len(memory_hits),
        "risk_flags": sorted(set(risk_flags)),
        "gate_status": (latest_step or {}).get("gate_status") if isinstance(latest_step, dict) else "no_gate_yet",
        "last_artifacts": last_artifacts,
        "claim_boundary": CLAIM_BOUNDARY,
        "official_submit_allowed": False,
        "generated_at": _now(),
    }
def _memory_records(*, task_type: str = "") -> list[dict]:
    store = RetrospectiveMemoryStore(SHARED_MEMORY)
    records = store.retrieve_by_task_type(task_type) if task_type else store._load()  # noqa: SLF001
    out = []
    for r in records:
        out.append({
            "memory_id": r.memory_id,
            "task_type": r.task_type,
            "method": r.method,
            "what_worked": r.what_worked,
            "what_failed": r.what_failed,
            "metric_delta": r.metric_delta,
            "reusable_strategy": r.reusable_strategy,
            "failure_pattern": r.failure_pattern,
            "linked_exp_ids": list(r.linked_exp_ids or []),
        })
    return out


def mode_memory(data: dict) -> dict:
    task_type = data.get("task_type", "")
    records = _memory_records(task_type=task_type)
    return {
        "task_id": data.get("task_id", ""),
        "task_type": task_type,
        "record_count": len(records),
        "memory": records,
        "reusable_strategies": [r["reusable_strategy"] for r in records if r["reusable_strategy"]],
        "failure_patterns": [r["failure_pattern"] for r in records if r["failure_pattern"]],
        "claim_boundary": CLAIM_BOUNDARY,
        "memory_store": "experiments/evolution/retrospective_memory.json",
        "generated_at": _now(),
    }


def mode_plan(data: dict) -> dict:
    task_id = _safe_task_id(data.get("task_id", ""))
    tdir = _task_dir(task_id)
    tdir.mkdir(parents=True, exist_ok=True)
    metric_direction = data.get("metric_direction", "maximize")
    graph = _load_graph(task_id, metric_direction=metric_direction,
                        metric_name=data.get("metric_name", "cv_score"))
    profile = _profile_from_input(data)
    strategies = recommend_strategies(profile).to_dict()

    # Use the MCGS brain to pick the next expansion when a graph already exists;
    # otherwise seed a baseline plan. The selector never runs code here.
    total_steps = int(data.get("budget", 6) or 6)
    if graph.nodes:
        selector = MCGSSelector(total_steps=max(1, total_steps))
        # rebuild branch/visit side-tables from persisted node topology so the
        # plan reflects real structure (best-effort; selector degrades gracefully).
        for node in graph.nodes.values():
            selector.branch_of.setdefault(node.exp_id, node.branch_type or "branch_0")
        try:
            expansion = selector.select(graph, step=len(graph.nodes))
            decision = "expand_selected_node"
            selected_branch = expansion.node_exp_id
            coding_mode = expansion.coding_mode
            expansion_type = expansion.expansion_type
            reference_exp_ids = list(expansion.reference_exp_ids)
            phase = expansion.phase
        except Exception as exc:  # selector must never crash the API
            decision, selected_branch = "seed_baseline_fallback", graph.root_exp_id
            coding_mode, expansion_type, reference_exp_ids, phase = "Base", "primary", [], "exploration"
            data.setdefault("_selector_error", f"{type(exc).__name__}")
    else:
        decision, selected_branch = "seed_baseline", "EXP000"
        coding_mode, expansion_type, reference_exp_ids, phase = "Base", "primary", [], "exploration"

    best_node = graph.nodes.get(graph.best_exp_id) if graph.best_exp_id else None
    expected_delta = "+0.3-2% (proxy CV, strategy-dependent)" if graph.nodes else "baseline (no parent yet)"
    contract = create_contract(
        contract_id=f"{task_id}:plan:{_now()}",
        exp_id=_next_exp_id(graph),
        claim=f"{coding_mode}/{expansion_type} candidate for {task_id}",
        hypothesis=data.get("objective", f"Improve {task_id} via {expansion_type} expansion."),
        implementation_requirement="Runnable script emitting CV_SCORE, submission.csv, metrics.json.",
        metric=graph.metric_name,
        baseline_exp_id=selected_branch or "",
        acceptance_criteria={graph.metric_name: {"min" if not _direction_lower(metric_direction) else "max": None}},
        ablation_plan=list(strategies.get("strategies", [])),
        conclusion_boundary=CLAIM_BOUNDARY,
        required_artifacts=REQUIRED_ARTIFACTS,
    )
    contract_rel = f"workspace/evolution/{task_id}/validation_contract.json"
    (tdir / "validation_contract.json").write_text(
        json.dumps({
            "schema": "academic_research_os.validation_contract.v1",
            "contract_id": contract.contract_id, "exp_id": contract.exp_id,
            "task_id": task_id, "created_at": _now(),
            "hypothesis": contract.hypothesis, "metric": contract.metric,
            "required_artifacts": contract.required_artifacts,
            "conclusion_boundary": contract.conclusion_boundary,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plan = {
        "task_id": task_id,
        "objective": data.get("objective", ""),
        "budget": total_steps,
        "resource_mode": data.get("resource_mode", "workstation_gpu"),
        "rank_target_percentile": data.get("rank_target_percentile"),
        "official_submit_allowed": False,
        "search_controller_decision": decision,
        "selected_branch": selected_branch,
        "code_generation_mode": coding_mode,
        "expansion_type": expansion_type,
        "reference_exp_ids": reference_exp_ids,
        "phase": phase,
        "recommended_strategies": strategies.get("strategies", []),
        "strategy_rationale": strategies.get("rationale", {}),
        "expected_delta": expected_delta,
        "parent_best_score": best_node.cv_score if best_node else None,
        "rollback_condition": [
            "candidate does not improve best-so-far beyond min_delta",
            "run did not complete successfully (non-zero exit / OOM / timeout)",
            "validation contract acceptance fails",
        ],
        "validation_contract_path": contract_rel,
        "claim_boundary": CLAIM_BOUNDARY,
        "generated_at": _now(),
    }
    (tdir / "latest_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    plan["plan_path"] = f"workspace/evolution/{task_id}/latest_plan.json"
    return plan
def mode_step(data: dict) -> dict:
    """One evolution step. In dry_run (default) it PLANS and writes governance
    artifacts + a graph node WITHOUT training. A non-dry-run step is intentionally
    blocked here: real training must be launched via the workstation orchestrator
    (the Node layer wires that), so the brain never bypasses the gates."""
    task_id = _safe_task_id(data.get("task_id", ""))
    dry_run = bool(data.get("dry_run", True))
    tdir = _task_dir(task_id)
    tdir.mkdir(parents=True, exist_ok=True)
    metric_direction = data.get("metric_direction", "maximize")
    graph = _load_graph(task_id, metric_direction=metric_direction,
                        metric_name=data.get("metric_name", "cv_score"))
    plan = _read_json(tdir / "latest_plan.json") or mode_plan(dict(data))

    if not dry_run:
        result = {
            "task_id": task_id,
            "dry_run": False,
            "decision": "blocked_use_workstation",
            "gate_status": "blocked",
            "reason": (
                "Evolution engine only plans/schedules. Real training must run "
                "through the workstation orchestrator (run_local_experiment / "
                "run_mcgs_experiment / GPU gate). Launch it from the workstation."
            ),
            "next_action": "POST /api/tasks/<id>/run-mcgs-experiment or a workstation GPU job",
            "plan": plan,
            "claim_boundary": CLAIM_BOUNDARY,
            "generated_at": _now(),
        }
        (tdir / "latest_step.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    # --- dry_run: record a PLANNED node + full governance artifacts (no training) ---
    exp_id = _next_exp_id(graph)
    parent_id = plan.get("selected_branch") if isinstance(plan, dict) else graph.best_exp_id
    if parent_id not in graph.nodes:
        parent_id = graph.best_exp_id if graph.best_exp_id in graph.nodes else None
    coding_mode = plan.get("code_generation_mode", "Base") if isinstance(plan, dict) else "Base"
    expansion_type = plan.get("expansion_type", "primary") if isinstance(plan, dict) else "primary"
    node = ExperimentNode(
        exp_id=exp_id, parent_id=parent_id, branch_type=coding_mode,
        task_name=task_id,
        hypothesis=plan.get("objective") or f"{expansion_type} expansion (planned)",
        implementation_summary=f"{coding_mode}/{expansion_type} planned candidate (dry_run, not trained)",
        code_path=f"workspace/evolution/{task_id}/{exp_id}/solution.py",
        cv_score=None, metric_name=graph.metric_name, metric_direction=metric_direction,
        decision="planned_dry_run", risk_flags=["not_trained_dry_run"],
        created_at=_now(),
    )
    graph.add_node(node)
    if parent_id and parent_id in graph.nodes:
        graph.add_edge(parent_id, exp_id, coding_mode)
    graph.export_json(_graph_path(task_id))

    # validation contract + claim audit (same libraries the loop uses)
    contract = create_contract(
        contract_id=f"{task_id}:{exp_id}:contract", exp_id=exp_id,
        claim=f"{coding_mode}/{expansion_type} candidate for {task_id}",
        hypothesis=node.hypothesis,
        implementation_requirement="Runnable script emitting CV_SCORE, submission.csv, metrics.json.",
        metric=graph.metric_name, baseline_exp_id=parent_id or "",
        ablation_plan=list(plan.get("recommended_strategies", []) if isinstance(plan, dict) else []),
        conclusion_boundary=CLAIM_BOUNDARY, required_artifacts=REQUIRED_ARTIFACTS,
    )
    artifact_check = check_required_artifacts(contract, [])  # nothing produced in dry_run
    acceptance = evaluate_acceptance(contract, {})
    (tdir / "validation_contract.json").write_text(json.dumps({
        "schema": "academic_research_os.validation_contract.v1",
        "contract_id": contract.contract_id, "exp_id": exp_id, "task_id": task_id,
        "created_at": _now(), "hypothesis": contract.hypothesis, "metric": contract.metric,
        "required_artifacts": contract.required_artifacts, "artifact_check": artifact_check,
        "acceptance": acceptance, "conclusion_boundary": contract.conclusion_boundary,
        "run_success": False, "dry_run": True, "cv_score": None,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    audit = audit_claim(
        claim_id=f"{task_id}:{exp_id}:claim",
        claim_text=f"{exp_id} is a PLANNED candidate (dry_run); no score claimed.",
        related_exp_ids=[exp_id],
        contract={"hypothesis": node.hypothesis, "conclusion_boundary": CLAIM_BOUNDARY},
        supporting_metrics={}, required_ablations=list(plan.get("recommended_strategies", []) if isinstance(plan, dict) else []),
        completed_ablations=[], evidence={"has_required_experiments": False,
                                          "has_mechanistic_evidence": False, "missing_evidence": REQUIRED_ARTIFACTS},
    )
    from dataclasses import asdict as _asdict
    (tdir / "claim_audit.json").write_text(json.dumps({
        "schema": "academic_research_os.claim_audit.v1", "created_at": _now(),
        "task_id": task_id, "dry_run": True, **_asdict(audit),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    result = {
        "task_id": task_id, "dry_run": True, "exp_id": exp_id,
        "decision": "planned_node_recorded",
        "gate_status": "dry_run_no_gate",
        "code_generation_mode": coding_mode, "expansion_type": expansion_type,
        "parent_exp_id": parent_id, "node_count": len(graph.nodes),
        "artifacts": [
            f"workspace/evolution/{task_id}/search_graph.json",
            f"workspace/evolution/{task_id}/validation_contract.json",
            f"workspace/evolution/{task_id}/claim_audit.json",
        ],
        "reason": "Dry-run planning step: recorded a search-graph node + validation contract + claim audit. No training was executed.",
        "next_action": "Launch real training through the workstation orchestrator to score this node.",
        "claim_boundary": CLAIM_BOUNDARY, "official_submit_allowed": False,
        "generated_at": _now(),
    }
    (tdir / "latest_step.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _resolve_run_artifacts(task_id: str, run_id: str) -> tuple[list[dict], list[str]]:
    """Find REAL artifacts on disk for a completed run. Never fabricates: only
    files that actually exist under experiments/<task_id>/<run_id>/ are attached,
    so the promotion gate can only ever promote on real evidence."""
    artifacts: list[dict] = []
    available: list[str] = []
    if not run_id or ".." in run_id or not re.fullmatch(r"[A-Za-z0-9._\-]+", run_id):
        return artifacts, available
    run_dir = ROOT / "experiments" / task_id / run_id
    for name in REQUIRED_ARTIFACTS:  # metrics.json, submission.csv
        if (run_dir / name).exists():
            artifacts.append({"path": f"experiments/{task_id}/{run_id}/{name}", "artifact_type": name})
            available.append(name)
    return artifacts, available


def mode_ingest_result(data: dict) -> dict:
    """Backfill bridge: ingest a REAL training result (cv_score + on-disk
    artifacts) produced by the workstation orchestrator into the search graph and
    the shared retrospective memory, then apply the promotion gate. THIS is what
    makes best_so_far real. Scores/artifacts are never fabricated; official Kaggle
    rank fields stay None (claim boundary)."""
    task_id = _safe_task_id(data.get("task_id", ""))
    tdir = _task_dir(task_id)
    tdir.mkdir(parents=True, exist_ok=True)
    metric_direction = data.get("metric_direction", "maximize")
    metric_name = data.get("metric_name", "cv_score")
    graph = _load_graph(task_id, metric_direction=metric_direction, metric_name=metric_name)
    plan = _read_json(tdir / "latest_plan.json") or {}

    run_id = str(data.get("run_id", "") or "")
    cv_raw = data.get("cv_score", None)
    cv_score = float(cv_raw) if isinstance(cv_raw, (int, float)) else None
    run_success_provided = "run_success" in data
    run_success = bool(data.get("run_success", cv_score is not None))

    # Guard against fabricating spurious nodes: an ingest MUST carry a real signal
    # (a run_id, a cv_score, or an EXPLICIT run_success). A bare call like
    # {"task_id": ...} with none of these is rejected WITHOUT touching the graph or
    # shared memory, so accidental/empty POSTs can never pollute the loop with
    # phantom run_failed nodes or junk memory records. The legitimate failure path
    # (cycle training_failed) always passes run_success=false explicitly, so it is
    # unaffected and still records a real negative example.
    if not run_id and cv_score is None and not run_success_provided:
        _reason = ("ingest_result needs at least one of: run_id, cv_score, or an "
                   "explicit run_success. Refusing to fabricate a node/memory record.")
        return {
            "ok": False, "task_id": task_id, "decision": "rejected_empty_ingest",
            "gate_status": "rejected", "node_count": len(graph.nodes),
            "reason": _reason, "error": _reason,
            "claim_boundary": CLAIM_BOUNDARY, "official_submit_allowed": False,
            "generated_at": _now(),
        }
    method = str(data.get("method") or plan.get("code_generation_mode") or "Base")
    expansion_type = str(data.get("expansion_type") or plan.get("expansion_type") or "primary")

    # Resolve or create the node this result belongs to. Prefer the most recent
    # untrained PLANNED node so plan -> run -> score maps onto a single node.
    exp_id = str(data.get("exp_id", "") or "")
    if exp_id and exp_id in graph.nodes:
        node = graph.nodes[exp_id]
    else:
        planned = [n for n in graph.nodes.values()
                   if n.cv_score is None and "not_trained_dry_run" in (n.risk_flags or [])]
        if planned:
            node = sorted(planned, key=lambda n: n.created_at or "")[-1]
            exp_id = node.exp_id
        else:
            exp_id = _next_exp_id(graph)
            parent_id = plan.get("selected_branch") if isinstance(plan, dict) else graph.best_exp_id
            if parent_id not in graph.nodes:
                parent_id = graph.best_exp_id if graph.best_exp_id in graph.nodes else None
            node = ExperimentNode(
                exp_id=exp_id, parent_id=parent_id, branch_type=method, task_name=task_id,
                hypothesis=plan.get("objective") or f"{expansion_type} expansion",
                implementation_summary=f"{method}/{expansion_type} trained via workstation orchestrator",
                code_path=f"experiments/{task_id}/{run_id}/solution.py",
                metric_name=metric_name, metric_direction=metric_direction, created_at=_now(),
            )
            graph.add_node(node)
            if parent_id and parent_id in graph.nodes:
                graph.add_edge(parent_id, exp_id, method)
    # Attach REAL on-disk artifacts + real score/metrics to the node.
    artifacts, available = _resolve_run_artifacts(task_id, run_id)
    node.artifacts = artifacts
    node.cv_score = cv_score
    node.metric_name = metric_name
    node.metric_direction = metric_direction
    node.metrics = {**(data.get("metrics") or {}), metric_name: cv_score}
    node.decision = "needs_review"
    node.risk_flags = [f for f in (node.risk_flags or []) if f != "not_trained_dry_run"]
    if not run_success:
        node.risk_flags = sorted(set(node.risk_flags + ["run_failed"]))
    # Official Kaggle fields stay None: local CV / proxy only (claim boundary).
    node.official_rank = None
    node.rank_percentile = None
    node.official_submission_ref = None

    # Promotion gate: real run_success + real score + real artifacts required.
    decision = graph.decide_promotion(
        candidate_exp_id=exp_id, metric=metric_name, direction=metric_direction,
        min_delta=float(data.get("min_delta", 0.0) or 0.0),
        required_artifacts=REQUIRED_ARTIFACTS, run_success=run_success,
    )

    # Compute ALL fallible governance in-memory BEFORE touching disk, so a crash
    # here can never leave a half-written ingest (graph updated but files missing).
    contract = create_contract(
        contract_id=f"{task_id}:{exp_id}:ingest", exp_id=exp_id,
        claim=f"{method}/{expansion_type} trained candidate for {task_id}",
        hypothesis=node.hypothesis,
        implementation_requirement="Runnable script emitting CV_SCORE, submission.csv, metrics.json.",
        metric=metric_name, baseline_exp_id=node.parent_id or "",
        ablation_plan=list(plan.get("recommended_strategies", []) if isinstance(plan, dict) else []),
        conclusion_boundary=CLAIM_BOUNDARY, required_artifacts=REQUIRED_ARTIFACTS,
    )
    artifact_check = check_required_artifacts(contract, available)
    acceptance = evaluate_acceptance(contract, {metric_name: cv_score})
    audit = audit_claim(
        claim_id=f"{task_id}:{exp_id}:claim",
        claim_text=(f"{exp_id} scored {metric_name}={cv_score} (local CV / proxy). "
                    "No official Kaggle rank claimed."),
        related_exp_ids=[exp_id],
        contract={"hypothesis": node.hypothesis, "conclusion_boundary": CLAIM_BOUNDARY},
        supporting_metrics={metric_name: cv_score} if cv_score is not None else {},
        required_ablations=list(plan.get("recommended_strategies", []) if isinstance(plan, dict) else []),
        completed_ablations=[], evidence={
            "has_required_experiments": artifact_check["passed"],
            "has_mechanistic_evidence": bool(available),
            "missing_evidence": artifact_check["missing_artifacts"]},
    )
    from dataclasses import asdict as _asdict

    # All governance computed successfully -> now commit to disk (graph first).
    graph.export_json(_graph_path(task_id))
    (tdir / "validation_contract.json").write_text(json.dumps({
        "schema": "academic_research_os.validation_contract.v1",
        "contract_id": contract.contract_id, "exp_id": exp_id, "task_id": task_id,
        "created_at": _now(), "hypothesis": contract.hypothesis, "metric": metric_name,
        "required_artifacts": contract.required_artifacts, "artifact_check": artifact_check,
        "acceptance": acceptance, "conclusion_boundary": contract.conclusion_boundary,
        "run_success": run_success, "dry_run": False, "cv_score": cv_score, "run_id": run_id,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    (tdir / "claim_audit.json").write_text(json.dumps({
        "schema": "academic_research_os.claim_audit.v1", "created_at": _now(),
        "task_id": task_id, "dry_run": False, "run_id": run_id, **_asdict(audit),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # Backfill SHARED retrospective memory (real outcome only).
    try:
        store = RetrospectiveMemoryStore(SHARED_MEMORY)
        store.add_memory(MemoryRecord(
            memory_id=f"{task_id}:{exp_id}:{run_id or _now()}",
            task_type=str(data.get("task_type", "") or ""),
            dataset_profile={k: data.get(k) for k in ("modality", "metric", "n_train", "n_test") if data.get(k) is not None},
            method=f"{method}/{expansion_type}",
            what_worked=(f"cv={cv_score}" if decision.get("promoted") else ""),
            what_failed=("" if run_success else "run did not complete successfully"),
            metric_delta=decision.get("promotion_delta"),
            reusable_strategy=(method if decision.get("promoted") else ""),
            failure_pattern=("" if run_success else "orchestrator_run_failed"),
            linked_exp_ids=[exp_id],
        ))
        memory_written = True
    except Exception as exc:  # memory write must never crash the bridge
        memory_written = False
        data.setdefault("_memory_error", f"{type(exc).__name__}: {exc}")

    best_node = graph.nodes.get(graph.best_exp_id) if graph.best_exp_id else None
    result = {
        "task_id": task_id, "dry_run": False, "exp_id": exp_id, "run_id": run_id,
        "decision": "promoted" if decision.get("promoted") else "held",
        "gate_status": "promoted" if decision.get("promoted") else "held_not_promoted",
        "promotion": decision,
        "cv_score": cv_score, "run_success": run_success,
        "artifacts_found": available,
        "best_so_far": {"exp_id": graph.best_exp_id,
                        "cv_score": best_node.cv_score if best_node else None,
                        "metric": metric_name, "metric_direction": metric_direction},
        "node_count": len(graph.nodes),
        "memory_written": memory_written,
        "artifacts": [
            f"workspace/evolution/{task_id}/search_graph.json",
            f"workspace/evolution/{task_id}/validation_contract.json",
            f"workspace/evolution/{task_id}/claim_audit.json",
        ],
        "reason": ("Real result ingested; promotion gate applied on real artifacts. "
                   "best_so_far now reflects a real local-CV score."),
        "claim_boundary": CLAIM_BOUNDARY, "official_submit_allowed": False,
        "generated_at": _now(),
    }
    (tdir / "latest_step.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _safe_exp_dir(rel: str) -> Path:
    """Resolve an engine-A experiment dir under experiments/evolution/ safely.

    Only a path that stays inside experiments/evolution/ is accepted, so a
    caller can never point ingest at an arbitrary location on disk.
    """
    rel = (rel or "").strip().replace("\\", "/")
    if not rel or ".." in rel:
        raise ValueError(f"unsafe exp_dir: {rel!r}")
    base = (ROOT / "experiments" / "evolution").resolve()
    target = (ROOT / rel).resolve()
    if base not in target.parents and target != base:
        raise ValueError(f"exp_dir must live under experiments/evolution/: {rel!r}")
    return target


def mode_ingest_summary(data: dict) -> dict:
    """Ingest an ALREADY-GATED engine-A run (research_os EvolutionLoop) into the
    workstation search graph + shared memory, then re-apply the workstation
    promotion gate on top.

    Engine A runs its OWN promotion gate remotely (GPURunner verifies metrics.json
    / submission.csv exist on the box via ``ls`` and records that in the best-EXP
    ``validation_contract.json`` locally). This mode keys on that recorded local
    governance -- it never fabricates local artifact files. The best node's real
    cv_score + artifact references are imported; official Kaggle rank stays None
    (claim boundary preserved)."""
    task_id = _safe_task_id(data.get("task_id", ""))
    tdir = _task_dir(task_id)
    tdir.mkdir(parents=True, exist_ok=True)
    exp_dir = _safe_exp_dir(str(data.get("exp_dir", "") or ""))

    summary = _read_json(exp_dir / "summary.json")
    a_graph = _read_json(exp_dir / "search_graph.json")
    if not isinstance(summary, dict) or not isinstance(a_graph, dict):
        _reason = "engine-A summary.json / search_graph.json not found or invalid; refusing to ingest."
        return {"ok": False, "task_id": task_id, "decision": "rejected_missing_summary",
                "gate_status": "rejected", "reason": _reason, "error": _reason,
                "claim_boundary": CLAIM_BOUNDARY, "official_submit_allowed": False, "generated_at": _now()}

    best_id = str(summary.get("best_exp_id") or a_graph.get("best_exp_id") or "")
    metric_name = str(data.get("metric_name") or summary.get("metric") or "cv_score")
    metric_direction = str(data.get("metric_direction") or summary.get("metric_direction") or "maximize")
    # export_json writes nodes as a list; older callers may pass a dict. Handle both.
    raw_nodes = a_graph.get("nodes")
    if isinstance(raw_nodes, dict):
        a_best = raw_nodes.get(best_id) if best_id else None
    elif isinstance(raw_nodes, list):
        a_best = next((n for n in raw_nodes if isinstance(n, dict) and n.get("exp_id") == best_id), None)
    else:
        a_best = None
    if not best_id or not isinstance(a_best, dict):
        _reason = "engine-A run has no best_exp_id / best node; nothing scored to ingest."
        return {"ok": False, "task_id": task_id, "decision": "rejected_no_best",
                "gate_status": "rejected", "reason": _reason, "error": _reason,
                "claim_boundary": CLAIM_BOUNDARY, "official_submit_allowed": False, "generated_at": _now()}

    # Gate on engine A's OWN recorded local governance for the best EXP. This is
    # the real, on-disk evidence -- not a fabricated local file.
    vc = _read_json(exp_dir / best_id / "validation_contract.json") or {}
    run_success = bool(vc.get("run_success", False))
    artifact_check = vc.get("artifact_check") or {}
    artifacts_verified = bool(artifact_check.get("passed", False))
    cv_raw = a_best.get("cv_score", summary.get("best_cv_score"))
    cv_score = float(cv_raw) if isinstance(cv_raw, (int, float)) else None
    if cv_score is None or not run_success or not artifacts_verified:
        _reason = (f"engine-A best {best_id} not eligible: run_success={run_success}, "
                   f"artifacts_verified={artifacts_verified}, cv_score={cv_score}. Not promoting.")
        run_success = False  # force a hold below; still record a negative node

    graph = _load_graph(task_id, metric_direction=metric_direction, metric_name=metric_name)
    method = str(data.get("method") or a_best.get("branch_type") or "research_os")
    expansion_type = str(data.get("expansion_type") or "primary")

    exp_id = _next_exp_id(graph)
    parent_id = graph.best_exp_id if graph.best_exp_id in graph.nodes else None
    # Import engine A's REAL artifact references (remote-verified by GPURunner).
    imported_artifacts = [a for a in (a_best.get("artifacts") or []) if isinstance(a, dict)]
    node = ExperimentNode(
        exp_id=exp_id, parent_id=parent_id, branch_type=method, task_name=task_id,
        hypothesis=str(a_best.get("hypothesis") or f"engine_A best {best_id}"),
        implementation_summary=str(a_best.get("implementation_summary")
                                   or f"research_os EvolutionLoop best ({best_id}) imported via ingest_summary"),
        code_path=f"{str(data.get('exp_dir'))}/best_solution.py",
        artifacts=imported_artifacts,
        metrics={**(a_best.get("metrics") or {}), metric_name: cv_score},
        cv_score=cv_score, metric_name=metric_name, metric_direction=metric_direction,
        created_at=_now(), decision="needs_review",
        risk_flags=[] if run_success else ["engine_a_ineligible"],
    )
    graph.add_node(node)
    if parent_id and parent_id in graph.nodes:
        graph.add_edge(parent_id, exp_id, method)
    node.official_rank = None
    node.rank_percentile = None
    node.official_submission_ref = None

    decision = graph.decide_promotion(
        candidate_exp_id=exp_id, metric=metric_name, direction=metric_direction,
        min_delta=float(data.get("min_delta", 0.0) or 0.0),
        required_artifacts=REQUIRED_ARTIFACTS, run_success=run_success,
    )

    available = [str(a.get("path") or "") for a in imported_artifacts]
    contract = create_contract(
        contract_id=f"{task_id}:{exp_id}:ingest_summary", exp_id=exp_id,
        claim=f"engine_A {best_id} imported as {exp_id} for {task_id}",
        hypothesis=node.hypothesis,
        implementation_requirement="research_os EvolutionLoop run emitting CV_SCORE + submission.csv + metrics.json.",
        metric=metric_name, baseline_exp_id=node.parent_id or "",
        ablation_plan=[], conclusion_boundary=CLAIM_BOUNDARY, required_artifacts=REQUIRED_ARTIFACTS,
    )
    check2 = check_required_artifacts(contract, [Path(p).name for p in available])
    acceptance = evaluate_acceptance(contract, {metric_name: cv_score})
    audit = audit_claim(
        claim_id=f"{task_id}:{exp_id}:claim",
        claim_text=(f"{exp_id} (engine_A {best_id}) scored {metric_name}={cv_score} (local CV / proxy). "
                    "No official Kaggle rank claimed."),
        related_exp_ids=[exp_id],
        contract={"hypothesis": node.hypothesis, "conclusion_boundary": CLAIM_BOUNDARY},
        supporting_metrics={metric_name: cv_score} if cv_score is not None else {},
        required_ablations=[], completed_ablations=[],
        evidence={"has_required_experiments": artifacts_verified,
                  "has_mechanistic_evidence": bool(available),
                  "missing_evidence": check2.get("missing_artifacts", [])},
    )
    from dataclasses import asdict as _asdict

    graph.export_json(_graph_path(task_id))
    (tdir / "validation_contract.json").write_text(json.dumps({
        "schema": "academic_research_os.validation_contract.v1",
        "contract_id": contract.contract_id, "exp_id": exp_id, "task_id": task_id,
        "created_at": _now(), "hypothesis": contract.hypothesis, "metric": metric_name,
        "required_artifacts": contract.required_artifacts, "artifact_check": check2,
        "acceptance": acceptance, "conclusion_boundary": contract.conclusion_boundary,
        "run_success": run_success, "dry_run": False, "cv_score": cv_score,
        "source": "engine_a_ingest_summary", "engine_a_exp_id": best_id,
        "engine_a_exp_dir": str(data.get("exp_dir")),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (tdir / "claim_audit.json").write_text(json.dumps({
        "schema": "academic_research_os.claim_audit.v1", "created_at": _now(),
        "task_id": task_id, "dry_run": False, "engine_a_exp_id": best_id, **_asdict(audit),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        store = RetrospectiveMemoryStore(SHARED_MEMORY)
        store.add_memory(MemoryRecord(
            memory_id=f"{task_id}:{exp_id}:engine_a:{best_id}",
            task_type=str(data.get("task_type", "") or ""),
            dataset_profile={k: data.get(k) for k in ("modality", "metric", "n_train", "n_test") if data.get(k) is not None},
            method=f"{method}/{expansion_type}",
            what_worked=(f"cv={cv_score}" if decision.get("promoted") else ""),
            what_failed=("" if run_success else "engine_a best ineligible for promotion"),
            metric_delta=decision.get("promotion_delta"),
            reusable_strategy=(method if decision.get("promoted") else ""),
            failure_pattern=("" if run_success else "engine_a_ineligible"),
            linked_exp_ids=[exp_id],
        ))
        memory_written = True
    except Exception as exc:  # memory write must never crash the bridge
        memory_written = False
        data.setdefault("_memory_error", f"{type(exc).__name__}: {exc}")

    best_node = graph.nodes.get(graph.best_exp_id) if graph.best_exp_id else None
    return {
        "ok": True, "task_id": task_id, "exp_id": exp_id, "engine_a_exp_id": best_id,
        "decision": "promoted" if decision.get("promoted") else "held",
        "gate_status": "passed" if decision.get("promoted") else "held",
        "promotion": decision, "cv_score": cv_score, "run_success": run_success,
        "artifacts_found": available, "memory_written": memory_written,
        "best_so_far": {"exp_id": graph.best_exp_id, "metric": metric_name,
                        "cv_score": (best_node.cv_score if best_node else None)},
        "artifacts": [
            f"workspace/evolution/{task_id}/search_graph.json",
            f"workspace/evolution/{task_id}/validation_contract.json",
            f"workspace/evolution/{task_id}/claim_audit.json",
        ],
        "official_rank": None, "official_submit_allowed": False,
        "claim_boundary": CLAIM_BOUNDARY, "generated_at": _now(),
    }


_MODES = {"state": mode_state, "plan": mode_plan, "step": mode_step,
          "graph": mode_graph, "memory": mode_memory,
          "ingest_result": mode_ingest_result, "ingest_summary": mode_ingest_summary}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evolution engine JSON adapter.")
    parser.add_argument("--mode", required=True, choices=sorted(_MODES.keys()))
    parser.add_argument("--input", required=True, help="path to a JSON input file")
    args = parser.parse_args()
    try:
        raw = Path(args.input).read_text(encoding="utf-8")
        data = json.loads(raw) if raw.strip() else {}
        if not isinstance(data, dict):
            raise ValueError("input JSON must be an object")
        payload = _MODES[args.mode](data)
        payload.setdefault("ok", True)
        payload.setdefault("mode", args.mode)
        sys.stdout.write(json.dumps(payload, ensure_ascii=False))
        return 0
    except Exception as exc:  # always emit valid JSON, never a bare traceback
        sys.stdout.write(json.dumps({
            "ok": False, "mode": args.mode,
            "error": f"{type(exc).__name__}: {exc}",
            "claim_boundary": CLAIM_BOUNDARY,
        }, ensure_ascii=False))
        # Exit 0 on purpose: the managed runner rejects non-zero exits and drops
        # stdout, but we always emit a structured JSON error the Node layer reads
        # via the `ok` field. This keeps the "always valid JSON" contract intact.
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
