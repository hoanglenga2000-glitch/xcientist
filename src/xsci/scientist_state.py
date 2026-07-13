"""Structured scientist checkpoint for the EvoMind terminal agent.

This module is intentionally deterministic and side-effect free.  It gives the
terminal shell and the LLM tool loop a compact Observe -> Analyze -> Propose ->
Gate -> Act view of the current research state.  The checkpoint never reads or
prints secrets and never starts training.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .kaggle_session import SessionState

SENSITIVE_RE = re.compile(
    r"(api[_-]?key|token|cookie|password|passwd|secret|ssh[_-]?key)\s*[:=]\s*\S+",
    re.IGNORECASE,
)


def _safe_state_text(value: Any, *, limit: int = 180) -> str:
    text = "" if value is None else str(value)
    text = SENSITIVE_RE.sub(r"\1=[redacted]", text)
    return text[:limit]


def _short_list(values: Any, *, limit: int = 5, item_limit: int = 180) -> list[str]:
    if not isinstance(values, list):
        return []
    rows: list[str] = []
    for item in values:
        text = _safe_state_text(item, limit=item_limit).strip()
        if text:
            rows.append(text)
        if len(rows) >= limit:
            break
    return rows


def _critique_gap_rows(values: Any, *, limit: int = 5) -> list[dict[str, str]]:
    if not isinstance(values, list):
        return []
    rows: list[dict[str, str]] = []
    for item in values:
        if isinstance(item, dict):
            gap = _safe_state_text(item.get("gap") or item.get("name") or item.get("id") or "", limit=180).strip()
            severity = _safe_state_text(item.get("severity") or "unknown", limit=40).strip()
            suggested_tool = _safe_state_text(item.get("suggested_tool") or item.get("tool") or "", limit=80).strip()
        else:
            gap = _safe_state_text(item, limit=180).strip()
            severity = "unknown"
            suggested_tool = ""
        if not gap:
            continue
        rows.append({
            "gap": gap,
            "severity": severity,
            "suggested_tool": suggested_tool,
        })
        if len(rows) >= limit:
            break
    return rows


def _load_scientist_upgrade_backlog(root: Path, *, limit: int = 8) -> dict[str, Any]:
    """Read the latest self-audit upgrade backlog as a safe decision signal."""
    xsci = Path(root) / ".xsci"
    backlog_path = xsci / "scientist_upgrade_backlog.json"
    audit_path = xsci / "scientist_self_audit.json"
    payload: dict[str, Any] = {}
    try:
        if backlog_path.exists():
            loaded = json.loads(backlog_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
    except (json.JSONDecodeError, OSError):
        payload = {}

    audit: dict[str, Any] = {}
    try:
        if audit_path.exists():
            loaded_audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if isinstance(loaded_audit, dict):
                audit = loaded_audit
    except (json.JSONDecodeError, OSError):
        audit = {}

    items = payload.get("items")
    if not isinstance(items, list):
        items = audit.get("upgrade_backlog") if isinstance(audit.get("upgrade_backlog"), list) else []

    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        status = _safe_state_text(item.get("status") or "proposed", limit=40)
        priority = _safe_state_text(item.get("priority") or "", limit=20)
        rows.append({
            "id": _safe_state_text(item.get("id") or "", limit=100),
            "title": _safe_state_text(item.get("title") or "", limit=180),
            "priority": priority,
            "status": status,
            "why": _safe_state_text(item.get("why") or "", limit=220),
            "safe_next_command": _safe_state_text(item.get("safe_next_command") or "", limit=120),
            "expected_artifacts": _short_list(item.get("expected_artifacts"), limit=5, item_limit=140),
            "gate": _safe_state_text(item.get("gate") or "", limit=80),
        })
        if len(rows) >= limit:
            break

    open_items = [
        row for row in rows
        if str(row.get("status") or "").lower() not in {"done", "closed", "resolved", "complete", "completed"}
    ]
    p0_items = [
        row for row in open_items
        if str(row.get("priority") or "").upper() in {"P0", "CRITICAL", "HIGH"}
    ]
    overall_score = payload.get("overall_score", audit.get("overall_score"))
    try:
        overall_score = int(overall_score)
    except (TypeError, ValueError):
        overall_score = None
    return {
        "artifact": str(backlog_path),
        "self_audit_artifact": str(audit_path),
        "present": bool(payload or audit),
        "generated_at": str(payload.get("generated_at") or audit.get("generated_at") or "")[:80],
        "overall_score": overall_score,
        "launch_readiness": str(audit.get("launch_readiness") or "")[:80],
        "open_count": len(open_items),
        "p0_count": len(p0_items),
        "open_items": open_items[:limit],
        "p0_items": p0_items[:5],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }


def _load_task_config(session: SessionState, root: Path) -> tuple[dict[str, Any], str]:
    if not session.selected_task:
        return {}, "No task selected."
    try:
        from .tasks import resolve_task

        path = resolve_task(session.selected_task, project_root=root)
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data, str(path)
        return {}, f"Task config is not an object: {path}"
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        return {}, str(exc)


def _data_status(task: dict[str, Any]) -> dict[str, Any]:
    local_dir = str(task.get("local_data_dir") or "")
    explicit_remote_dir = str(task.get("gpu_data_dir") or "")
    remote_dir = str(explicit_remote_dir or task.get("remote_data_dirname") or "")
    result: dict[str, Any] = {
        "local_data_dir": local_dir,
        "remote_data_dir": remote_dir,
        "remote_data_explicit": bool(explicit_remote_dir),
        "train_csv": False,
        "test_csv": False,
        "sample_submission": False,
        "available_files": [],
    }
    if not local_dir:
        result["message"] = "No local_data_dir configured."
        return result
    base = Path(local_dir)
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        exists = (base / name).exists()
        result[name.replace(".", "_")] = exists
        if exists:
            result["available_files"].append(name)
    result["message"] = (
        "Data CSVs found: " + ", ".join(result["available_files"])
        if result["available_files"]
        else f"No expected CSVs found under {base}."
    )
    return result


def _same_task(candidate: str, selected_task: str) -> bool:
    if not selected_task:
        return True
    def norm(value: str) -> str:
        return (value or "").lower().replace("-", "").replace("_", "").replace(" ", "")
    c = norm(candidate)
    s = norm(selected_task)
    return c == s or c.startswith(s)


def _recent_experiment_summaries(root: Path, *, selected_task: str = "",
                                 limit: int = 5) -> list[dict[str, Any]]:
    base = root / "experiments" / "evolution"
    if not base.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    try:
        run_dirs = sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True)
    except OSError:
        return []
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        task_name = str(data.get("task") or data.get("task_id") or run_dir.name)
        if selected_task and not (_same_task(task_name, selected_task) or _same_task(run_dir.name, selected_task)):
            continue
        rows.append({
            "run_id": run_dir.name,
            "task": task_name,
            "best_exp_id": data.get("best_exp_id") or "",
            "best_cv_score": data.get("best_cv_score"),
            "promotions": data.get("n_promotions", 0),
            "iterations": data.get("n_iterations", 0),
            "path": str(run_dir),
        })
        if len(rows) >= limit:
            break
    return rows


def _ledger_lines(root: Path) -> list[str]:
    try:
        from .tool_ledger import ToolLedger

        return ToolLedger(root).summary_lines(limit=6)
    except Exception:
        return []


def _evolution_stats(root: Path) -> dict[str, Any]:
    try:
        from .evolution_tracker import EvolutionTracker

        snap = EvolutionTracker(root).current_snapshot()
        return {
            "skill_level": snap.skill_level,
            "total_runs": snap.total_runs,
            "total_promotions": snap.total_promotions,
            "repair_attempts": snap.repair_attempts,
            "repair_successes": snap.repair_successes,
            "innovations_tried": snap.innovations_tried,
            "innovation_successes": snap.innovation_successes,
            "lessons_recorded": snap.lessons_recorded,
            "reusable_lessons": snap.reusable_lessons,
            "failure_lessons": snap.failure_lessons,
            "cross_task_transfers": snap.cross_task_transfers,
        }
    except Exception:
        return {"skill_level": "unknown"}


def _retrospective_records(root: Path, *, task_type: str = "",
                           limit: int = 8) -> list[dict[str, Any]]:
    """Return sanitized lessons from shared retrospective memory.

    This is intentionally read-only and bounded.  It extracts only lesson text,
    strategy names, failure patterns, and deltas; it never inspects credentials
    or raw training data.
    """
    path = root / "experiments" / "evolution" / "retrospective_memory.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    raw_records = payload if isinstance(payload, list) else payload.get("records", [])
    if not isinstance(raw_records, list):
        return []

    rows: list[dict[str, Any]] = []
    wanted = task_type.strip().lower()
    if wanted in {"?", "unknown", "unset", "none"}:
        wanted = ""
    for item in reversed(raw_records):
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("task_type") or "").strip().lower()
        if wanted and item_type and item_type != wanted:
            continue
        rows.append({
            "memory_id": str(item.get("memory_id") or "")[:120],
            "task_type": str(item.get("task_type") or "")[:60],
            "method": str(item.get("method") or "")[:80],
            "what_worked": str(item.get("what_worked") or "")[:220],
            "what_failed": str(item.get("what_failed") or "")[:220],
            "metric_delta": item.get("metric_delta"),
            "reusable_strategy": str(item.get("reusable_strategy") or "")[:220],
            "failure_pattern": str(item.get("failure_pattern") or "")[:80],
        })
        if len(rows) >= limit:
            break
    return rows


def _retrospective_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [
        r for r in records
        if r.get("what_worked") or r.get("reusable_strategy")
    ]
    failures = [
        r for r in records
        if r.get("what_failed") or r.get("failure_pattern")
    ]
    strategies: list[str] = []
    failure_patterns: list[str] = []
    for r in records:
        strategy = str(r.get("reusable_strategy") or "").strip()
        if strategy and strategy not in strategies:
            strategies.append(strategy)
        pattern = str(r.get("failure_pattern") or "").strip()
        if pattern and pattern not in failure_patterns:
            failure_patterns.append(pattern)
    return {
        "records_considered": len(records),
        "success_records": len(successes),
        "failure_records": len(failures),
        "top_strategies": strategies[:5],
        "failure_patterns": failure_patterns[:5],
        "records": records[:5],
    }


def _scientist_turn_records(root: Path, *, selected_task: str = "",
                            limit: int = 8) -> list[dict[str, Any]]:
    """Return recent sanitized scientist turns relevant to the selected task.

    The turn ledger is a second memory channel: it captures what the terminal
    scientist observed, which tools were used, which blockers recurred, and
    which next actions were proposed.  It is read-only here and already
    sanitized by ``scientist_turns``.
    """
    try:
        from .scientist_turns import load_recent_scientist_turns

        raw_turns = load_recent_scientist_turns(root, limit=max(limit * 3, limit))
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for item in reversed(raw_turns):
        if not isinstance(item, dict):
            continue
        task_name = str(item.get("task") or "")
        if selected_task and task_name and not _same_task(task_name, selected_task):
            continue
        decision = item.get("decision") if isinstance(item.get("decision"), dict) else {}
        critique = decision.get("scientific_critique") if isinstance(decision.get("scientific_critique"), dict) else {}
        tool_budget = decision.get("tool_budget") if isinstance(decision.get("tool_budget"), dict) else {}
        deferred_tools = _short_list(decision.get("deferred_tools"), limit=8, item_limit=80)
        must_run_deferred_tools = _short_list(
            decision.get("must_run_deferred_tools"),
            limit=8,
            item_limit=80,
        )
        try:
            must_run_deferred_count = int(tool_budget.get("must_run_deferred_count") or 0)
        except (TypeError, ValueError):
            must_run_deferred_count = 0
        budget_exhausted = bool(
            decision.get("budget_exhausted")
            or must_run_deferred_tools
            or must_run_deferred_count > 0
        )
        executed_tools = item.get("executed_tools") if isinstance(item.get("executed_tools"), list) else []
        tool_names: list[str] = []
        for tool in executed_tools[:12]:
            if isinstance(tool, dict):
                name = str(tool.get("tool") or "").strip()
                if name:
                    tool_names.append(name[:80])
            elif isinstance(tool, str):
                tool_names.append(tool[:80])
        rows.append({
            "turn_id": str(item.get("turn_id") or "")[:120],
            "ts": str(item.get("ts") or "")[:80],
            "task": task_name[:120],
            "route": str(item.get("route") or "")[:80],
            "mode": str(item.get("mode") or "")[:80],
            "selected_action": str(decision.get("selected_action") or "")[:120],
            "selected_branch": str(decision.get("selected_branch") or "")[:120],
            "code_generation_mode": str(decision.get("code_generation_mode") or "")[:80],
            "evidence_gaps": _critique_gap_rows(critique.get("evidence_gaps"), limit=6),
            "uncertainty_drivers": _short_list(critique.get("uncertainty_drivers"), limit=6, item_limit=180),
            "claim_boundaries": _short_list(critique.get("claim_boundaries"), limit=6, item_limit=220),
            "tool_budget": {
                "recommended_min_tools": tool_budget.get("recommended_min_tools"),
                "requested_max_tools": tool_budget.get("requested_max_tools"),
                "effective_max_tools": tool_budget.get("effective_max_tools"),
                "executed_tool_count": tool_budget.get("executed_tool_count"),
                "must_run_deferred_count": must_run_deferred_count,
            },
            "deferred_tools": deferred_tools,
            "must_run_deferred_tools": must_run_deferred_tools,
            "budget_exhausted": budget_exhausted,
            "blockers": [str(x)[:180] for x in (item.get("blockers") or [])[:8]],
            "next_actions": [str(x)[:220] for x in (item.get("next_actions") or [])[:8]],
            "tools": tool_names,
            "answer_preview": str(item.get("answer_preview") or "")[:260],
        })
        if len(rows) >= limit:
            break
    return rows


def _scientist_turn_summary(turns: list[dict[str, Any]]) -> dict[str, Any]:
    blockers: dict[str, int] = {}
    evidence_gaps: dict[str, dict[str, Any]] = {}
    next_actions: list[str] = []
    uncertainty_drivers: list[str] = []
    claim_boundaries: list[str] = []
    routes: dict[str, int] = {}
    tools: dict[str, int] = {}
    decisions: list[dict[str, Any]] = []
    budget_exhausted_turns = 0
    must_run_deferred_tools: dict[str, int] = {}
    budget_risks: list[dict[str, Any]] = []
    for turn in turns:
        route = str(turn.get("route") or "").strip()
        if route:
            routes[route] = routes.get(route, 0) + 1
        for blocker in turn.get("blockers") or []:
            text = str(blocker).strip()
            if text:
                blockers[text] = blockers.get(text, 0) + 1
        for action in turn.get("next_actions") or []:
            text = str(action).strip()
            if text and text not in next_actions:
                next_actions.append(text)
        for gap in turn.get("evidence_gaps") or []:
            if not isinstance(gap, dict):
                continue
            text = str(gap.get("gap") or "").strip()
            if not text:
                continue
            existing = evidence_gaps.setdefault(text, {
                "gap": text,
                "severity": str(gap.get("severity") or "unknown")[:40],
                "suggested_tool": str(gap.get("suggested_tool") or "")[:80],
                "count": 0,
            })
            existing["count"] = int(existing.get("count") or 0) + 1
        for driver in turn.get("uncertainty_drivers") or []:
            text = str(driver).strip()
            if text and text not in uncertainty_drivers:
                uncertainty_drivers.append(text)
        for boundary in turn.get("claim_boundaries") or []:
            text = str(boundary).strip()
            if text and text not in claim_boundaries:
                claim_boundaries.append(text)
        if turn.get("budget_exhausted"):
            budget_exhausted_turns += 1
            budget = turn.get("tool_budget") if isinstance(turn.get("tool_budget"), dict) else {}
            budget_risks.append({
                "turn_id": turn.get("turn_id", ""),
                "route": turn.get("route", ""),
                "effective_max_tools": budget.get("effective_max_tools"),
                "recommended_min_tools": budget.get("recommended_min_tools"),
                "must_run_deferred_tools": (turn.get("must_run_deferred_tools") or [])[:5],
            })
        for tool in turn.get("must_run_deferred_tools") or []:
            text = str(tool).strip()
            if text:
                must_run_deferred_tools[text] = must_run_deferred_tools.get(text, 0) + 1
        for tool in turn.get("tools") or []:
            text = str(tool).strip()
            if text:
                tools[text] = tools.get(text, 0) + 1
        if turn.get("selected_action") or turn.get("selected_branch"):
            decisions.append({
                "turn_id": turn.get("turn_id", ""),
                "action": turn.get("selected_action", ""),
                "branch": turn.get("selected_branch", ""),
                "mode": turn.get("code_generation_mode", ""),
            })

    recurring_blockers = [
        {"blocker": key, "count": value}
        for key, value in sorted(blockers.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    ]
    recurring_evidence_gaps = [
        value
        for _, value in sorted(
            evidence_gaps.items(),
            key=lambda kv: (-int(kv[1].get("count") or 0), kv[0]),
        )[:5]
    ]
    return {
        "turns_considered": len(turns),
        "latest_turn_id": turns[0].get("turn_id", "") if turns else "",
        "routes": routes,
        "tool_coverage": tools,
        "recent_decisions": decisions[:5],
        "recurring_blockers": recurring_blockers,
        "recurring_evidence_gaps": recurring_evidence_gaps,
        "recent_next_actions": next_actions[:6],
        "uncertainty_drivers": uncertainty_drivers[:6],
        "claim_boundaries": claim_boundaries[:6],
        "budget_exhausted_turns": budget_exhausted_turns,
        "must_run_deferred_tools": [
            {"tool": key, "count": value}
            for key, value in sorted(must_run_deferred_tools.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        ],
        "budget_risks": budget_risks[:5],
        "records": turns[:5],
    }


def _turn_reuse_records(turn_summary: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(turn_summary, dict):
        return []
    records: list[dict[str, Any]] = []
    for record in turn_summary.get("records") or []:
        if not isinstance(record, dict):
            continue
        if not (
            record.get("next_actions")
            or record.get("blockers")
            or record.get("selected_action")
            or record.get("evidence_gaps")
            or record.get("budget_exhausted")
        ):
            continue
        records.append({
            "turn_id": record.get("turn_id", ""),
            "route": record.get("route", ""),
            "selected_action": record.get("selected_action", ""),
            "selected_branch": record.get("selected_branch", ""),
            "next_actions": record.get("next_actions", [])[:3],
            "blockers": record.get("blockers", [])[:3],
            "evidence_gaps": record.get("evidence_gaps", [])[:3],
            "budget_exhausted": bool(record.get("budget_exhausted")),
            "must_run_deferred_tools": record.get("must_run_deferred_tools", [])[:5],
            "claim_boundaries": record.get("claim_boundaries", [])[:3],
            "tools": record.get("tools", [])[:6],
        })
        if len(records) >= 3:
            break
    return records


def _innovation_stats(root: Path) -> dict[str, Any]:
    try:
        from .innovation_engine import InnovationEngine

        return InnovationEngine(workspace_root=root).stats()
    except Exception:
        return {
            "proposals_generated": 0,
            "innovations_tried": 0,
            "successes": 0,
            "failures": 0,
            "hit_rate": "0.0%",
            "most_successful": [],
        }


def _strategy_family(task_type: str, modality: str, metric: str) -> list[str]:
    task_type_l = task_type.lower()
    modality_l = modality.lower()
    metric_l = metric.lower()
    if "time" in task_type_l or "time" in modality_l or "forecast" in task_type_l:
        return [
            "Use time-aware validation before any leaderboard claim.",
            "Start with calendar/lag/rolling features and a leakage audit.",
            "Compare a simple seasonal baseline against GBDT or sequence models.",
        ]
    if "regression" in task_type_l or metric_l in {"rmse", "rmsle", "mae"}:
        return [
            "Build a K-fold tabular baseline with robust numeric/categorical preprocessing.",
            "Try GBDT families first, then blend with linear or neural residual models.",
            "Track public-proxy risk with outlier and target-transformation audits.",
        ]
    if "image" in modality_l:
        return [
            "Start from a small pretrained baseline with deterministic augmentations.",
            "Record image size, class balance, and GPU memory risk before scaling.",
            "Use ensemble or TTA only after a clean OOF baseline exists.",
        ]
    return [
        "Start with a stratified or grouped CV baseline if labels allow it.",
        "Compare LightGBM/XGBoost/CatBoost-style tabular families before deep models.",
        "Add feature interactions, calibrated probabilities, and OOF blending only after the baseline is stable.",
    ]


def _build_proposals(task: dict[str, Any], data: dict[str, Any], recent: list[dict[str, Any]],
                     blockers: list[str]) -> list[str]:
    if not task:
        return [
            "Register or select a Kaggle task before designing experiments.",
            "Use `competitions` to browse, then `task add <kaggle-url>`.",
        ]
    if blockers:
        return [
            "Clear blocking gates before execution; keep the current turn in planning or inspection mode.",
            "Use `evomind ready` and the control panel to confirm every gate is visible.",
        ]
    if not data.get("train_csv") and not data.get("remote_data_explicit"):
        return [
            "Download data or set local_data_dir in the task config.",
            "After data appears, run a schema audit before code generation.",
        ]
    modality = str(task.get("modality") or "tabular")
    task_type = str(task.get("task_type") or "classification")
    metric = str(task.get("metric") or "accuracy")
    proposals = _strategy_family(task_type, modality, metric)
    if recent:
        proposals.append("Use the latest run as best-so-far; the next branch should improve or hold without overwriting it.")
        proposals.append("If the last run stalled, choose a Diff or model-family branch and record the retrospective memory.")
    else:
        proposals.append("First run should be a reproducible baseline with metrics, OOF/submission artifacts, and claim audit.")
    return proposals


def _build_research_hypotheses(task: dict[str, Any], data: dict[str, Any],
                               recent: list[dict[str, Any]],
                               memory_records: list[dict[str, Any]],
                               blockers: list[str]) -> list[dict[str, Any]]:
    if not task:
        return [{
            "id": "H0-task-selection",
            "claim": "A research hypothesis requires a selected task.",
            "test": "Register a task and rebuild the checkpoint.",
            "evidence_required": ["task config", "metric", "data location"],
        }]
    if blockers:
        return [{
            "id": "H0-gate-readiness",
            "claim": "Execution quality is limited by unresolved setup gates.",
            "test": "Clear readiness blockers before model search.",
            "evidence_required": ["evomind ready", "system_status", "gate artifact"],
        }]
    if not data.get("train_csv") and not data.get("remote_data_explicit"):
        return [{
            "id": "H0-data-readiness",
            "claim": "No model comparison is trustworthy until training data is declared.",
            "test": "Register/download data and run schema audit.",
            "evidence_required": ["train.csv or explicit remote data path", "schema audit"],
        }]

    task_type = str(task.get("task_type") or "classification")
    metric = str(task.get("metric") or "task metric")
    hypotheses: list[dict[str, Any]] = []
    if not recent:
        hypotheses.append({
            "id": "H1-reproducible-baseline",
            "claim": f"A simple audited baseline can establish a valid {metric} reference.",
            "test": "Run Base-mode baseline with deterministic CV and required artifacts.",
            "evidence_required": ["metrics.json", "validation_contract", "claim_audit", "submission audit when applicable"],
        })
    else:
        latest = _latest_run_signal(recent)
        if latest.get("signal") == "stalled_or_failed":
            hypotheses.append({
                "id": "H1-repair-frontier",
                "claim": "The latest frontier is blocked by failure or stagnation, so a Diff-mode repair is higher value than another broad search.",
                "test": "Diagnose failure pattern, simplify the branch, and rerun with rollback protection.",
                "evidence_required": ["failure artifact", "repair note", "new metrics", "hold/promote gate"],
            })
        else:
            hypotheses.append({
                "id": "H1-best-so-far-improvement",
                "claim": "A bounded feature/model-family branch may improve the protected best-so-far.",
                "test": "Run one Stepwise branch and compare against current best with the promotion gate.",
                "evidence_required": ["OOF comparison", "score_promotion_gate", "artifact_manifest"],
            })

    transferable = [
        r for r in memory_records
        if r.get("reusable_strategy") or r.get("what_worked")
    ]
    if transferable:
        hypotheses.append({
            "id": "H2-retrospective-transfer",
            "claim": "Prior retrospective memory contains strategies worth reusing on this task family.",
            "test": "Reuse one compatible memory strategy as a controlled branch, not as an untracked manual tweak.",
            "evidence_required": ["memory_reuse_records", "branch rationale", "OOF delta"],
        })
    elif "classification" in task_type.lower():
        hypotheses.append({
            "id": "H2-model-family-check",
            "claim": "A model-family comparison is likely to expose a stronger classification baseline.",
            "test": "Compare calibrated GBDT-style baseline against the current candidate under the same split.",
            "evidence_required": ["same-split metrics", "fold stability", "claim audit"],
        })

    return hypotheses[:4]


def _failure_avoidance(memory_records: list[dict[str, Any]],
                       recent: list[dict[str, Any]],
                       turn_summary: dict[str, Any] | None = None) -> list[str]:
    patterns: list[str] = []
    for r in memory_records:
        pattern = str(r.get("failure_pattern") or "").strip()
        if pattern and pattern not in patterns:
            patterns.append(pattern)
    guidance = [f"Pre-check known failure pattern: {p}." for p in patterns[:4]]
    if isinstance(turn_summary, dict):
        for row in turn_summary.get("recurring_blockers") or []:
            if not isinstance(row, dict):
                continue
            blocker = str(row.get("blocker") or "").strip()
            count = int(row.get("count") or 0)
            if blocker and count >= 1:
                guidance.append(f"Reuse scientist-turn blocker memory: {blocker}.")
                break
    if recent and _latest_run_signal(recent).get("signal") == "stalled_or_failed":
        guidance.append("Latest run stalled or failed; use Diff mode and keep the previous best untouched.")
    if not guidance:
        guidance.append("No reusable failure pattern found; enforce validation_contract and claim_audit before promotion.")
    return guidance


def _experiment_plan(decision: dict[str, Any],
                     hypotheses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    branch = decision.get("selected_branch", "unknown")
    code_mode = decision.get("code_generation_mode", "none")
    return [
        {
            "step": "observe",
            "action": "Inspect task config, data schema, metric direction, and available prior runs.",
            "done_when": "checkpoint observe/analyze fields are current",
        },
        {
            "step": "generate",
            "action": f"Generate branch={branch} using code_generation_mode={code_mode}.",
            "done_when": "candidate code is tied to a hypothesis and artifact manifest",
        },
        {
            "step": "validate",
            "action": "Run deterministic validation and compare only against the protected best-so-far.",
            "done_when": "metrics, OOF/submission artifact, validation_contract, and score gate exist",
        },
        {
            "step": "learn",
            "action": "Write retrospective memory for what worked/failed and keep failed claims blocked.",
            "done_when": "claim_audit and memory/update artifacts are present",
        },
    ] + [
        {
            "step": f"hypothesis:{h.get('id', 'H?')}",
            "action": str(h.get("test") or ""),
            "done_when": ", ".join(h.get("evidence_required") or []),
        }
        for h in hypotheses[:2]
    ]


def _workplan_step(step_id: str, title: str, status: str, *,
                   owner: str = "evomind",
                   tool: str = "",
                   action: str = "",
                   evidence: list[str] | None = None,
                   gate: str = "",
                   blocked_reason: str = "") -> dict[str, Any]:
    return {
        "id": step_id,
        "title": title,
        "status": status,
        "owner": owner,
        "tool": tool,
        "action": action,
        "evidence": evidence or [],
        "gate": gate,
        "blocked_reason": blocked_reason,
    }


def _build_workplan_steps(checkpoint: dict[str, Any],
                          decision_payload: dict[str, Any]) -> list[dict[str, Any]]:
    gate = checkpoint.get("gate", {}) if isinstance(checkpoint.get("gate"), dict) else {}
    blockers = [str(x) for x in gate.get("blockers", [])]
    warnings = [str(x) for x in gate.get("warnings", [])]
    can_execute = bool(gate.get("can_execute"))
    decision = decision_payload.get("decision", {}) if isinstance(decision_payload.get("decision"), dict) else {}
    brief = decision_payload.get("research_brief", {}) if isinstance(decision_payload.get("research_brief"), dict) else {}
    hypotheses = brief.get("hypotheses", []) if isinstance(brief.get("hypotheses"), list) else []
    selected_branch = str(decision.get("selected_branch") or "none")
    code_mode = str(decision.get("code_generation_mode") or "none")
    selected_action = str(decision.get("selected_action") or "none")
    data_warning = next((w for w in warnings if "data" in w.lower() or "train.csv" in w.lower()), "")

    steps = [
        _workplan_step(
            "observe",
            "Observe current task, resource, data, memory, and recent runs",
            "completed" if checkpoint.get("ok") else "blocked",
            tool="scientist_checkpoint",
            action="Build the read-only Observe/Analyze/Propose/Gate/Act state.",
            evidence=["scientist_checkpoint", ".xsci/scientist_turns.jsonl"],
        ),
        _workplan_step(
            "decide",
            "Select next branch and code generation mode",
            "completed" if decision_payload.get("ok") else "blocked",
            tool="research_decision",
            action=f"selected_action={selected_action}; branch={selected_branch}; code_generation_mode={code_mode}",
            evidence=[".xsci/scientist_decision.json", "research_brief"],
        ),
        _workplan_step(
            "gate_preflight",
            "Clear setup, compute, and human-safety gates",
            "completed" if not blockers else "blocked",
            tool="system_status",
            action="Stop before execution if any blocking setup gate remains.",
            evidence=["evomind ready", "system_status"],
            gate="setup_gate",
            blocked_reason="; ".join(blockers[:5]),
        ),
        _workplan_step(
            "data_contract",
            "Confirm training data and schema contract",
            "completed" if can_execute else ("blocked" if data_warning else "pending"),
            tool="data_check",
            action="Verify train/test data, metric, id/target columns, schema, and leakage-sensitive split.",
            evidence=["data_check", "validation_contract"],
            gate="data_gate",
            blocked_reason=data_warning,
        ),
        _workplan_step(
            "hypothesis",
            "Bind the branch to explicit research hypotheses",
            "completed" if hypotheses else "pending",
            tool="scientist_checkpoint",
            action="Use checkpoint hypotheses and memory reuse records to avoid ad hoc training.",
            evidence=["hypotheses", "memory_reuse_records", "scientist_turn_reuse_records"],
        ),
        _workplan_step(
            "branch_design",
            "Prepare candidate branch implementation plan",
            "ready" if can_execute and code_mode != "none" else "blocked",
            tool="research_decision",
            action=f"Generate a {code_mode} branch for {selected_branch}; protect best-so-far.",
            evidence=["search_controller_decision", "artifact_manifest"],
            gate="code_review_gate",
            blocked_reason="" if can_execute and code_mode != "none" else "Execution/data gates are not clear enough for branch generation.",
        ),
        _workplan_step(
            "execute_candidate",
            "Run audited workstation candidate only after explicit run command",
            "ready" if can_execute and int(decision.get("timeout_budget_seconds") or 0) > 0 else "blocked",
            tool="AgentSession",
            action="Start `evomind run <task>` or workstation action; do not bypass AgentOrchestrator.",
            evidence=["agent_trace", "metrics.json", "OOF/submission when applicable"],
            gate="execution_gate",
            blocked_reason="" if can_execute else "Execution is blocked until setup/data gates pass.",
        ),
        _workplan_step(
            "validate_promote",
            "Validate, compare, and promote or hold",
            "pending" if can_execute else "blocked",
            tool="score_promotion_gate",
            action="Promote only if validation_contract, score gate, submission audit, and claim audit pass.",
            evidence=["validation_contract", "score_promotion_gate", "submission_audit", "claim_audit"],
            gate="promotion_gate",
            blocked_reason="" if can_execute else "No candidate run can be promoted before execution gates clear.",
        ),
        _workplan_step(
            "learn",
            "Write reusable lessons back to memory",
            "pending",
            tool="retrospective_memory",
            action="Record what worked, what failed, reusable strategy, and failure pattern.",
            evidence=["retrospective_memory.json", ".xsci/evolution_tracker.json"],
        ),
        _workplan_step(
            "official_submit",
            "Official Kaggle submission gate",
            "blocked",
            tool="submission_approval",
            action="Do not submit unless the user explicitly approves and Kaggle response artifact is captured.",
            evidence=["submission_approval", "kaggle_submission_response.json"],
            gate="human_submission_gate",
            blocked_reason="blocked_until_explicit_human_approval",
        ),
    ]
    return steps


def _workplan_focus(steps: list[dict[str, Any]]) -> dict[str, Any]:
    for status in ("blocked", "ready", "pending"):
        for step in steps:
            if step.get("status") == status:
                if step.get("id") == "official_submit" and any(
                    other.get("status") in {"ready", "pending"}
                    and other.get("id") != "official_submit"
                    for other in steps
                ):
                    continue
                return {
                    "step_id": step.get("id", ""),
                    "status": status,
                    "title": step.get("title", ""),
                    "action": step.get("action", ""),
                    "blocked_reason": step.get("blocked_reason", ""),
                }
    return {"step_id": "complete", "status": "completed", "title": "All workplan steps are complete."}


def _recent_step_trace_records(root: Path, *, limit: int = 12) -> list[dict[str, Any]]:
    try:
        from .scientist_trace import load_recent_scientist_step_events

        raw_events = load_recent_scientist_step_events(root, limit=limit)
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for event in raw_events:
        if not isinstance(event, dict):
            continue
        rows.append({
            "ts": str(event.get("ts") or "")[:80],
            "source": str(event.get("source") or "")[:80],
            "phase": str(event.get("phase") or "")[:80],
            "step_id": str(event.get("step_id") or "")[:80],
            "status": str(event.get("status") or "")[:60],
            "tool": str(event.get("tool") or "")[:80],
            "message": str(event.get("message") or "")[:240],
            "artifact_path": str(event.get("artifact_path") or "")[:220],
        })
    return rows


def _root_cause_from_text(text: str) -> str:
    low = (text or "").lower()
    if any(token in low for token in ("no task", "task required", "selected task", "select_task", "competition is selected")):
        return "no_task"
    if "kaggle" in low and any(token in low for token in ("submit", "submission", "human", "rank", "medal", "top30")):
        return "official_submit_blocked"
    if "kaggle" in low:
        return "kaggle_blocked"
    if any(token in low for token in (
        "train.csv",
        "local_data_dir",
        "training data is missing",
        "data gate",
        "data path",
        "schema contract",
        "sample_submission",
    )):
        return "data_missing"
    if any(token in low for token in ("gpu", "hpc", "ssh", "remote workspace", "compute")):
        return "gpu_blocked"
    if any(token in low for token in ("claim", "rank", "medal", "top30", "leaderboard")):
        return "claim_gate_blocked"
    if any(token in low for token in ("failed", "stalled", "no promotions", "repair", "traceback", "exception")):
        return "stale_best_so_far"
    if any(token in low for token in ("llm", "api", "setup", "gate", "ready", "credential", "provider")):
        return "setup_gate_blocked"
    if any(token in low for token in ("trace", "artifact", "manifest", "evidence")):
        return "observability_gap"
    return "quality_improvement"


def _repair_issue(severity: str, source: str, evidence: str, *, root_cause: str = "",
                  recommendation: str = "") -> dict[str, Any]:
    cause = root_cause or _root_cause_from_text(evidence)
    return {
        "severity": severity,
        "source": source,
        "root_cause": cause,
        "evidence": str(evidence)[:320],
        "recommendation": str(recommendation or _default_repair_recommendation(cause))[:360],
    }


def _default_repair_recommendation(root_cause: str) -> str:
    return {
        "no_task": "Register or select a task, then rebuild the Scientist checkpoint.",
        "setup_gate_blocked": "Run readiness checks and configure missing LLM/Kaggle/GPU resources through the safe setup helpers.",
        "data_missing": "Register local train/test data or an explicit GPU data path before any training run.",
        "gpu_blocked": "Run the GPU/HPC status smoke and keep training blocked until the compute gate is visible.",
        "kaggle_blocked": "Configure Kaggle API through DPAPI or keep official downloads/submits blocked.",
        "official_submit_blocked": "Keep official submit blocked until human approval and a Kaggle response artifact exist.",
        "claim_gate_blocked": "Run claim audit and keep rank/medal/top30 fields empty without official response evidence.",
        "stale_best_so_far": "Use Diff-mode repair, inspect failure artifacts, and keep best-so-far protected.",
        "observability_gap": "Generate trace/workplan artifacts before attempting another execution.",
        "quality_improvement": "Use a bounded branch with validation_contract, score gate, and retrospective memory.",
    }.get(root_cause, "Use EvoMind gates and artifacts to repair the blocker before execution.")


def _dedupe_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    rows: list[dict[str, Any]] = []
    for issue in issues:
        key = (
            str(issue.get("root_cause") or ""),
            str(issue.get("source") or ""),
            str(issue.get("evidence") or "")[:120],
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(issue)
    return rows


def _repair_step(step_id: str, title: str, *, status: str = "pending",
                 priority: str = "P1", owner: str = "evomind",
                 tool: str = "", action: str = "",
                 gate: str = "", evidence: list[str] | None = None,
                 done_when: str = "", command: str = "") -> dict[str, Any]:
    return {
        "id": step_id,
        "title": title,
        "status": status,
        "priority": priority,
        "owner": owner,
        "tool": tool,
        "action": action,
        "gate": gate,
        "evidence": evidence or [],
        "done_when": done_when,
        "command": command,
    }


def _build_repair_steps(root_causes: list[str], checkpoint: dict[str, Any],
                        workplan: dict[str, Any], task: str) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    if "no_task" in root_causes:
        steps.append(_repair_step(
            "select_task",
            "Select or register a Kaggle/MLE-Bench task",
            status="ready",
            priority="P0",
            tool="task_list",
            action="Use `evomind competitions` and `evomind task add <url>`, then rebuild the checkpoint.",
            gate="task_gate",
            evidence=[".xsci/tasks/<task>.json", "scientist_checkpoint"],
            done_when="selected_task is non-empty and task config resolves",
            command="evomind task list",
        ))
    if "setup_gate_blocked" in root_causes:
        steps.append(_repair_step(
            "repair_setup_gate",
            "Repair missing setup resources before execution",
            status="ready",
            priority="P0",
            tool="system_status",
            action="Run readiness and configure missing model/Kaggle/compute credentials through setup helpers without printing secrets.",
            gate="setup_gate",
            evidence=["evomind ready", "system_status", "verified launch audit"],
            done_when="blocking_setup is empty",
            command="evomind ready",
        ))
    if "data_missing" in root_causes:
        steps.append(_repair_step(
            "repair_data_contract",
            "Repair data readiness and schema contract",
            status="ready",
            priority="P0",
            tool="data_check",
            action="Set local_data_dir or GPU data path, verify train/test/sample submission, then generate validation_contract.",
            gate="data_gate",
            evidence=["data_check", "train.csv or remote GPU data path", "validation_contract"],
            done_when="data_check reports training data available for the selected resource mode",
            command=f"evomind download {task}" if task else "evomind workplan",
        ))
    if "gpu_blocked" in root_causes:
        steps.append(_repair_step(
            "repair_compute_gate",
            "Repair GPU/HPC compute gate",
            status="ready",
            priority="P1",
            tool="gpu_status",
            action="Run GPU smoke only; keep long training blocked until remote workspace and provenance are verified.",
            gate="compute_gate",
            evidence=["gpu_status", "remote provenance", "safe command manifest"],
            done_when="GPU/HPC status is verified or resource mode is explicitly local smoke",
            command="evomind ready",
        ))
    if "kaggle_blocked" in root_causes:
        steps.append(_repair_step(
            "repair_kaggle_gate",
            "Repair Kaggle API readiness",
            status="ready",
            priority="P1",
            tool="kaggle_status",
            action="Configure Kaggle token through DPAPI helpers; do not write tokens to repo files.",
            gate="kaggle_api_gate",
            evidence=["kaggle_status", "DPAPI readiness artifact"],
            done_when="Kaggle status is configured or task is marked proxy-only",
            command="evomind ready",
        ))
    if "stale_best_so_far" in root_causes:
        steps.append(_repair_step(
            "repair_failed_frontier",
            "Diagnose failed/stalled frontier before another branch",
            status="ready",
            priority="P1",
            tool="recent_run",
            action="Inspect failure artifacts, choose Diff-mode repair, and protect best-so-far from overwrite.",
            gate="best_so_far_gate",
            evidence=["failure artifact", "retrospective_memory", "hold/promote gate"],
            done_when="failure cause is recorded and the next run has rollback protection",
            command="evomind live",
        ))
    if "observability_gap" in root_causes:
        steps.append(_repair_step(
            "repair_observability",
            "Create missing trace and workplan artifacts",
            status="ready",
            priority="P1",
            tool="scientist_autopilot",
            action="Run read-only diagnosis so the next executor has current evidence.",
            gate="evidence_gate",
            evidence=[".xsci/scientist_step_trace.jsonl", ".xsci/scientist_workplan.json"],
            done_when="step trace, workplan, and decision artifacts are present",
            command="evomind autopilot",
        ))
    if "claim_gate_blocked" in root_causes or "official_submit_blocked" in root_causes:
        steps.append(_repair_step(
            "enforce_claim_boundary",
            "Keep official claims and submission behind human gates",
            status="blocked",
            priority="P0",
            tool="claim_audit",
            action="Do not display rank, medal, top30, or submit until official Kaggle response artifact exists.",
            gate="human_submission_gate",
            evidence=["claim_audit", "submission_approval", "kaggle_submission_response.json"],
            done_when="human approval and Kaggle response artifact exist",
            command="evomind report",
        ))

    gate = checkpoint.get("gate", {}) if isinstance(checkpoint.get("gate"), dict) else {}
    can_execute = bool(gate.get("can_execute"))
    if can_execute and task:
        focus = workplan.get("current_focus") if isinstance(workplan.get("current_focus"), dict) else {}
        steps.append(_repair_step(
            "prepare_guarded_execution",
            "Proceed only through guarded EvoMind execution",
            status="ready",
            priority="P2",
            tool="AgentSession",
            action=f"Current focus={focus.get('step_id', 'unknown')}; run through workstation/AgentOrchestrator only.",
            gate="execution_gate",
            evidence=["agent_trace", "metrics.json", "validation_contract", "score_promotion_gate", "claim_audit"],
            done_when="candidate run creates required artifacts and promotes or holds safely",
            command=f"evomind run {task}",
        ))

    steps.append(_repair_step(
        "write_memory_after_repair",
        "Write repair outcome to retrospective memory",
        status="pending",
        priority="P2",
        tool="retrospective_memory",
        action="Record what failed, what fixed it, and whether the strategy is reusable across tasks.",
        gate="memory_gate",
        evidence=["retrospective_memory.json", ".xsci/evolution_tracker.json"],
        done_when="repair outcome is durable and visible in evolution_status",
        command="evomind evolution",
    ))
    return steps


def build_scientist_repair_plan(session: SessionState, root: Path, *,
                                persist: bool = False) -> dict[str, Any]:
    """Build a read-only self-repair plan from current Scientist evidence.

    The repair plan is the missing "debug loop" between diagnosis and action:
    it looks at gates, data readiness, recent run signal, turn memory, and step
    trace blockers, then writes an ordered fix plan.  It never starts training
    and never submits to Kaggle.
    """
    root = Path(root)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    task = session.selected_task or ""
    checkpoint = build_scientist_checkpoint(session, root)
    decision_payload = build_research_decision(session, root, persist=False)
    workplan = build_scientist_workplan(session, root, persist=False)
    gate = checkpoint.get("gate", {}) if isinstance(checkpoint.get("gate"), dict) else {}
    blockers = [str(x) for x in gate.get("blockers", [])]
    warnings = [str(x) for x in gate.get("warnings", [])]
    can_execute = bool(gate.get("can_execute"))
    recent_runs = checkpoint.get("recent_runs", []) if isinstance(checkpoint.get("recent_runs"), list) else []
    latest_signal = _latest_run_signal(recent_runs)
    memory = checkpoint.get("memory", {}) if isinstance(checkpoint.get("memory"), dict) else {}
    turn_summary = memory.get("scientist_turns", {}) if isinstance(memory.get("scientist_turns"), dict) else {}
    step_trace = _recent_step_trace_records(root, limit=16)

    issues: list[dict[str, Any]] = []
    if not task:
        issues.append(_repair_issue(
            "blocker", "task_gate", "No task selected.",
            root_cause="no_task",
            recommendation="Select or register a task before planning, code generation, or training."
        ))
    for blocker in blockers:
        issues.append(_repair_issue("blocker", "setup_gate", blocker))
    for warning in warnings:
        severity = "warning"
        if _root_cause_from_text(warning) == "data_missing" and task:
            severity = "blocker"
        issues.append(_repair_issue(severity, "checkpoint_warning", warning))
    if task and not can_execute and not blockers and not any(_root_cause_from_text(w) == "data_missing" for w in warnings):
        issues.append(_repair_issue(
            "blocker", "execution_gate",
            "Execution gate is not clear, but no setup blocker was listed.",
            root_cause="observability_gap",
            recommendation="Regenerate workplan and step trace so the blocked gate has inspectable evidence."
        ))
    if latest_signal.get("signal") == "stalled_or_failed":
        issues.append(_repair_issue(
            "warning", "recent_run",
            f"Latest run {latest_signal.get('latest_run_id')} had iterations={latest_signal.get('iterations')} and promotions={latest_signal.get('promotions')}.",
            root_cause="stale_best_so_far",
            recommendation="Use a Diff repair branch and preserve the previous best-so-far."
        ))
    for row in turn_summary.get("recurring_blockers", []) if isinstance(turn_summary, dict) else []:
        if not isinstance(row, dict):
            continue
        blocker = str(row.get("blocker") or "")
        if blocker:
            issues.append(_repair_issue(
                "warning", "scientist_turn_memory",
                f"{blocker} (count={row.get('count')})",
            ))
    blocked_trace = [
        row for row in step_trace
        if str(row.get("status") or "").lower() in {"blocked", "failed"}
    ]
    for row in blocked_trace[-5:]:
        evidence = f"{row.get('phase')} / {row.get('tool')}: {row.get('message')}"
        if _root_cause_from_text(evidence) in {"official_submit_blocked", "claim_gate_blocked"}:
            continue
        issues.append(_repair_issue("warning", "scientist_step_trace", evidence))
    if not step_trace:
        issues.append(_repair_issue(
            "info", "scientist_step_trace",
            "No step trace events exist yet.",
            root_cause="observability_gap",
            recommendation="Run read-only Autopilot or Workplan first so later actions are inspectable."
        ))

    issues = _dedupe_issues(issues)
    root_causes = list(dict.fromkeys(str(issue.get("root_cause") or "quality_improvement") for issue in issues))
    if not root_causes and can_execute:
        root_causes = ["quality_improvement"]
    repair_steps = _build_repair_steps(root_causes, checkpoint, workplan, task)

    if not task:
        mode = "no_task"
    elif any(issue.get("severity") == "blocker" for issue in issues):
        mode = "blocked_repair"
    elif can_execute:
        mode = "ready_to_execute_guarded"
    else:
        mode = "quality_improvement"

    ready_commands = [
        str(step.get("command") or "")
        for step in repair_steps
        if step.get("status") == "ready" and step.get("command")
    ]
    if mode == "ready_to_execute_guarded" and task:
        safe_next = f"evomind run {task}"
    elif ready_commands:
        safe_next = ready_commands[0]
    elif task:
        safe_next = "evomind workplan"
    else:
        safe_next = "evomind task list"

    artifact_path = root / ".xsci" / "scientist_repair_plan.json"
    payload: dict[str, Any] = {
        "ok": True,
        "tool": "scientist_repair_plan",
        "generated_at": generated_at,
        "selected_task": task,
        "mode": mode,
        "diagnosis": issues,
        "root_causes": root_causes,
        "repair_steps": repair_steps,
        "safe_next_command": safe_next,
        "decision": decision_payload.get("decision", {}),
        "workplan_focus": workplan.get("current_focus", {}),
        "latest_run_signal": latest_signal,
        "step_trace_considered": step_trace[-8:],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
        "claim_boundary": (
            "Repair readiness is not a score claim. Official submit, rank, medal, and top30 "
            "remain blocked without explicit human approval and Kaggle response artifact."
        ),
        "artifact_path": str(artifact_path),
    }

    if persist:
        try:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = artifact_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(artifact_path)
        except OSError as exc:
            payload["ok"] = False
            payload["message"] = f"Could not write repair plan artifact: {exc}"
        try:
            from .scientist_trace import record_scientist_step_event

            trace_run_id = f"repair_{generated_at.replace(':', '').replace('+', 'Z')}"
            record_scientist_step_event(root, {
                "trace_run_id": trace_run_id,
                "source": "scientist_repair_plan",
                "task": task,
                "phase": "repair_plan_snapshot",
                "status": "completed" if payload.get("ok") else "failed",
                "tool": "scientist_repair_plan",
                "message": f"Repair plan generated: mode={mode}; root_causes={', '.join(root_causes[:5])}.",
                "artifact_path": str(artifact_path),
                "details": {
                    "mode": mode,
                    "root_causes": root_causes,
                    "safe_next_command": safe_next,
                },
                "no_training_started": True,
            })
            for step in repair_steps:
                record_scientist_step_event(root, {
                    "trace_run_id": trace_run_id,
                    "source": "scientist_repair_plan",
                    "task": task,
                    "phase": "repair_step",
                    "step_id": step.get("id", ""),
                    "status": step.get("status", "pending"),
                    "tool": step.get("tool", ""),
                    "message": step.get("title", ""),
                    "artifact_path": str(artifact_path),
                    "gate": step.get("gate", ""),
                    "evidence": step.get("evidence", []),
                    "details": {
                        "priority": step.get("priority", ""),
                        "action": step.get("action", ""),
                        "command": step.get("command", ""),
                        "done_when": step.get("done_when", ""),
                    },
                    "no_training_started": True,
                })
        except Exception:
            pass
        try:
            from .scientist_turns import record_scientist_turn

            record_scientist_turn(root, {
                "task": task,
                "route": "scientist_repair_plan",
                "user": "scientist_repair_plan",
                "forced_tools": ["scientist_checkpoint", "research_decision", "scientist_workplan", "scientist_step_trace"],
                "executed_tools": [
                    {"tool": "scientist_repair_plan", "ok": bool(payload.get("ok", True)), "status": mode}
                ],
                "mode": mode,
                "decision": decision_payload.get("decision", {}),
                "blockers": [issue.get("evidence", "") for issue in issues if issue.get("severity") == "blocker"],
                "next_actions": [step.get("action", "") for step in repair_steps[:5]],
                "artifacts": [
                    str(artifact_path),
                    str(root / ".xsci" / "scientist_step_trace.jsonl"),
                    str(root / ".xsci" / "scientist_workplan.json"),
                ],
                "answer_preview": f"Repair plan mode={mode}; safe_next_command={safe_next}",
                "no_training_started": True,
                "official_submit": "blocked_until_explicit_human_approval",
            })
        except Exception:
            pass
    return payload


def _contract_status_from_repair(repair_plan: dict[str, Any]) -> tuple[str, list[str]]:
    if not isinstance(repair_plan, dict):
        return "review_required", ["repair_plan_missing"]
    mode = str(repair_plan.get("mode") or "")
    root_causes = [str(item) for item in repair_plan.get("root_causes", []) if str(item)]
    hard_blockers = {
        "no_task",
        "setup_gate_blocked",
        "gpu_blocked",
    }
    if any(cause in hard_blockers for cause in root_causes) or mode in {"no_task", "blocked_repair"}:
        if "data_missing" in root_causes and not any(cause in hard_blockers for cause in root_causes):
            return "conditional_go_data_contract_first", root_causes
        return "no_go", root_causes
    if "data_missing" in root_causes:
        return "conditional_go_data_contract_first", root_causes
    return "go", root_causes or ["quality_improvement"]


def _execution_contract_goal(task: str, decision: dict[str, Any], *,
                             go_no_go: str, contract_path: Path,
                             repair_plan: dict[str, Any]) -> str:
    selected_action = decision.get("selected_action", "run_audited_baseline")
    selected_branch = decision.get("selected_branch", "baseline")
    code_mode = decision.get("code_generation_mode", "Base")
    validation_focus = decision.get("validation_focus", "standard_cv")
    rollback = decision.get("rollback_condition", "hold if gates fail")
    safe_next = repair_plan.get("safe_next_command", f"evomind run {task}") if isinstance(repair_plan, dict) else f"evomind run {task}"
    return (
        f"Scientist decision: action={selected_action}; branch={selected_branch}; "
        f"code_generation_mode={code_mode}; validation_focus={validation_focus}. "
        f"Execution contract: go_no_go={go_no_go}; safe_next={safe_next}; "
        f"contract_artifact={contract_path}. Protect best-so-far; rollback condition: {rollback}. "
        "Produce agent_trace, metrics, validation_contract, score_promotion_gate, "
        "claim_audit, and artifact_manifest. If data is not ready, complete data/schema "
        "contract first before model training. Do not submit to official Kaggle."
    )


def build_scientist_execution_contract(session: SessionState, root: Path, *,
                                       persist: bool = False) -> dict[str, Any]:
    """Build a read-only execution contract before AgentSession starts.

    This is the bridge from "thinking" to "doing".  It captures go/no-go,
    branch/code-mode, data-contract conditions, rollback, required artifacts,
    and claim boundaries.  It never starts training; the terminal run path may
    use it to enrich the AgentSession goal.
    """
    root = Path(root)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    task = session.selected_task or ""
    compute_mode = session.current_compute_override or session.compute_backend
    checkpoint = build_scientist_checkpoint(session, root)
    decision_payload = build_research_decision(session, root, persist=False)
    repair_plan = build_scientist_repair_plan(session, root, persist=False)
    decision = decision_payload.get("decision", {}) if isinstance(decision_payload.get("decision"), dict) else {}
    gate = checkpoint.get("gate", {}) if isinstance(checkpoint.get("gate"), dict) else {}
    setup_blockers = session.blocking_setup(compute_override=compute_mode)
    repair_status, root_causes = _contract_status_from_repair(repair_plan)

    if setup_blockers:
        go_no_go = "no_go"
    elif repair_status == "no_go":
        go_no_go = "no_go"
    elif repair_status == "conditional_go_data_contract_first":
        go_no_go = "conditional_go_data_contract_first"
    else:
        go_no_go = "go"

    model_training_ready = bool(gate.get("can_execute")) and go_no_go == "go"
    agent_session_ready = bool(task) and not setup_blockers and go_no_go in {"go", "conditional_go_data_contract_first"}
    artifact_path = root / ".xsci" / "scientist_execution_contract.json"
    enriched_goal = _execution_contract_goal(task, decision, go_no_go=go_no_go, contract_path=artifact_path, repair_plan=repair_plan)
    required_artifacts = [
        "agent_trace",
        "metrics.json",
        "OOF or validation predictions when applicable",
        "submission.csv when applicable",
        "artifact_manifest",
        "validation_contract",
        "score_promotion_gate",
        "claim_audit",
        "scientist_execution_contract",
    ]
    payload: dict[str, Any] = {
        "ok": True,
        "tool": "scientist_execution_contract",
        "generated_at": generated_at,
        "selected_task": task,
        "compute_mode": compute_mode,
        "go_no_go": go_no_go,
        "agent_session_ready": agent_session_ready,
        "model_training_ready": model_training_ready,
        "root_causes": root_causes,
        "setup_blockers": setup_blockers,
        "data_contract_status": "ready" if model_training_ready else (
            "planning_ready_hpc_execution_blocked" if gate.get("planning_can_proceed") else
            "must_be_completed_inside_agent_session" if go_no_go == "conditional_go_data_contract_first" else "blocked"
        ),
        "planning_can_proceed": bool(gate.get("planning_can_proceed")),
        "decision": decision,
        "execution_command": f"evomind run {task}" if task else "evomind task list",
        "enriched_goal": enriched_goal,
        "required_artifacts": required_artifacts,
        "rollback_condition": decision.get("rollback_condition", "hold if gates fail"),
        "risk_controls": [
            "Do not bypass AgentOrchestrator or workstation gates.",
            "Protect best-so-far; hold candidates when score/claim gates fail.",
            "Complete data/schema contract before model training when data is not ready.",
            "Official Kaggle submit remains blocked until explicit human approval and response artifact.",
        ],
        "linked_artifacts": {
            "decision": str(root / ".xsci" / "scientist_decision.json"),
            "workplan": str(root / ".xsci" / "scientist_workplan.json"),
            "repair_plan": str(root / ".xsci" / "scientist_repair_plan.json"),
            "step_trace": str(root / ".xsci" / "scientist_step_trace.jsonl"),
        },
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
        "claim_boundary": (
            "Execution-contract readiness is not a leaderboard claim. Rank, medal, top30, and official "
            "submission remain blocked without human approval and Kaggle response artifact."
        ),
        "artifact_path": str(artifact_path),
    }
    try:
        from .scientist_gate_decision import build_execution_gate_decision

        payload["execution_gate_decision"] = build_execution_gate_decision(payload)
    except Exception:
        payload["execution_gate_decision"] = {
            "blocked": go_no_go != "go",
            "status": "blocked" if go_no_go != "go" else "ready_for_gated_training",
            "no_training_started": True,
            "official_submit": "blocked_until_explicit_human_approval",
        }

    if persist:
        try:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = artifact_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(artifact_path)
        except OSError as exc:
            payload["ok"] = False
            payload["message"] = f"Could not write execution contract artifact: {exc}"
        try:
            from .scientist_trace import record_scientist_step_event

            trace_run_id = f"contract_{generated_at.replace(':', '').replace('+', 'Z')}"
            record_scientist_step_event(root, {
                "trace_run_id": trace_run_id,
                "source": "scientist_execution_contract",
                "task": task,
                "phase": "execution_contract_snapshot",
                "status": "completed" if payload.get("ok") else "failed",
                "tool": "scientist_execution_contract",
                "message": (
                    f"Execution contract generated: go_no_go={go_no_go}; "
                    f"agent_session_ready={agent_session_ready}; model_training_ready={model_training_ready}."
                ),
                "artifact_path": str(artifact_path),
                "gate": "execution_contract_gate",
                "evidence": required_artifacts,
                "details": {
                    "go_no_go": go_no_go,
                    "compute_mode": compute_mode,
                    "root_causes": root_causes,
                    "decision": decision,
                },
                "no_training_started": True,
            })
        except Exception:
            pass
        try:
            from .scientist_turns import record_scientist_turn

            record_scientist_turn(root, {
                "task": task,
                "route": "scientist_execution_contract",
                "user": "scientist_execution_contract",
                "forced_tools": ["scientist_checkpoint", "research_decision", "scientist_repair_plan"],
                "executed_tools": [
                    {"tool": "scientist_checkpoint", "ok": bool(checkpoint.get("ok", True)), "status": gate.get("can_execute")},
                    {"tool": "research_decision", "ok": bool(decision_payload.get("ok", True)), "status": decision.get("selected_action", "none")},
                    {"tool": "scientist_repair_plan", "ok": bool(repair_plan.get("ok", True)), "status": repair_plan.get("mode", "")},
                ],
                "mode": go_no_go,
                "decision": decision,
                "blockers": setup_blockers,
                "next_actions": [payload.get("execution_command", "")],
                "artifacts": [
                    str(artifact_path),
                    str(root / ".xsci" / "scientist_decision.json"),
                    str(root / ".xsci" / "scientist_repair_plan.json"),
                    str(root / ".xsci" / "scientist_step_trace.jsonl"),
                ],
                "answer_preview": (
                    f"Execution contract go_no_go={go_no_go}; "
                    f"agent_session_ready={agent_session_ready}; "
                    f"model_training_ready={model_training_ready}."
                ),
                "no_training_started": True,
                "official_submit": "blocked_until_explicit_human_approval",
            })
        except Exception:
            pass
    return payload


def build_scientist_checkpoint(session: SessionState, root: Path) -> dict[str, Any]:
    """Return a current-state research checkpoint for terminal and LLM use."""
    root = Path(root)
    task, task_source = _load_task_config(session, root)
    data = _data_status(task) if task else {
        "local_data_dir": "",
        "remote_data_dir": "",
        "train_csv": False,
        "test_csv": False,
        "sample_submission": False,
        "available_files": [],
        "message": "No task selected.",
    }
    effective_compute = session.current_compute_override or session.compute_backend
    recent = _recent_experiment_summaries(root, selected_task=session.selected_task or "")
    blockers = session.blocking_setup(compute_override=effective_compute)
    warnings = [gap for gap in session.missing_setup(compute_override=effective_compute) if gap not in blockers]
    remote_data_ready = bool(data.get("remote_data_explicit")) and effective_compute == "gpu"
    data_ready = bool(data.get("train_csv")) or remote_data_ready
    if task and not data_ready:
        warnings.append(
            "Data: local train.csv is missing or local_data_dir is unset; do not launch a local training run until data is registered."
        )
    can_execute = not blockers and bool(task) and data_ready
    planning_blockers = [
        blocker for blocker in blockers
        if _root_cause_from_text(blocker) != "gpu_blocked"
    ]
    planning_can_proceed = not planning_blockers and bool(task) and data_ready
    metric = str(task.get("metric") or "?") if task else "?"
    direction = str(task.get("metric_direction") or "?") if task else "?"
    modality = str(task.get("modality") or "?") if task else "?"
    task_type = str(task.get("task_type") or "?") if task else "?"
    memory_records = _retrospective_records(root, task_type=task_type if task else "", limit=8)
    turn_records = _scientist_turn_records(root, selected_task=session.selected_task or "", limit=8)
    turn_summary = _scientist_turn_summary(turn_records)
    upgrade_backlog = _load_scientist_upgrade_backlog(root)
    hypotheses = _build_research_hypotheses(task, data, recent, memory_records, planning_blockers)

    observe = [
        f"workspace={root}",
        f"selected_task={session.selected_task or '(none)'}",
        f"task_source={task_source}",
        f"task={modality}/{task_type}, metric={metric}({direction})",
        f"compute={effective_compute}",
        data.get("message", ""),
        f"recent_runs={len(recent)}",
        f"scientist_turns={turn_summary.get('turns_considered', 0)}",
        f"memory={session.memory_summary or 'empty'}",
    ]
    if can_execute:
        gate_line = "Execution and data gates are clear."
    elif blockers:
        gate_line = "Execution gate is blocked."
    else:
        gate_line = "Core setup gates are clear, but the data gate still needs attention."
    if data.get("train_csv"):
        data_line = "Local training data is available."
    elif remote_data_ready:
        data_line = "Remote GPU training data is declared."
    else:
        data_line = "Training data is missing or undeclared."
    analyze = [
        gate_line,
        data_line,
        "Use official Kaggle submission only after human gate and Kaggle response artifact."
    ]
    if recent:
        best = recent[0]
        analyze.append(
            f"Latest known run {best.get('run_id')} best={best.get('best_exp_id') or 'N/A'} "
            f"cv={best.get('best_cv_score')}"
        )
    recurring_turn_blockers = turn_summary.get("recurring_blockers", [])
    if recurring_turn_blockers:
        top_blocker = recurring_turn_blockers[0]
        analyze.append(
            "Recent scientist turns repeatedly flagged: "
            f"{top_blocker.get('blocker')} (count={top_blocker.get('count')})."
        )
    recurring_evidence_gaps = turn_summary.get("recurring_evidence_gaps", [])
    if recurring_evidence_gaps:
        top_gap = recurring_evidence_gaps[0]
        analyze.append(
            "Recent scientist critique requires evidence closure: "
            f"{top_gap.get('gap')} via {top_gap.get('suggested_tool') or 'next safe audit tool'}."
        )
    if int(turn_summary.get("budget_exhausted_turns") or 0) > 0:
        analyze.append(
            "A prior Scientist turn exhausted its critical tool budget; the next decision must close deferred must-run tools before training."
        )
    if int(upgrade_backlog.get("p0_count") or 0) > 0:
        first_upgrade = (upgrade_backlog.get("p0_items") or [{}])[0]
        analyze.append(
            "Self-audit upgrade backlog has P0 work before training: "
            f"{first_upgrade.get('id') or first_upgrade.get('title')}."
        )

    return {
        "ok": True,
        "tool": "scientist_checkpoint",
        "mode": "ready_to_execute" if can_execute else "needs_attention",
        "observe": observe,
        "analyze": analyze,
        "propose": _build_proposals(task, data, recent, planning_blockers),
        "hypotheses": hypotheses,
        "gate": {
            "can_execute": can_execute,
            "planning_can_proceed": planning_can_proceed,
            "blockers": blockers,
            "planning_blockers": planning_blockers,
            "warnings": warnings,
            "official_submit": "blocked_by_human_gate",
        },
        "act": [
            "plan: ask for a concrete research plan without launching training",
            "run: enter preflight and audited AgentSession only if gates pass",
            "report: inspect artifacts and claim audit after a run",
        ],
        "memory": {
            "terminal_ledger": _ledger_lines(root),
            "evolution": _evolution_stats(root),
            "innovation": _innovation_stats(root),
            "retrospective": _retrospective_summary(memory_records),
            "scientist_turns": turn_summary,
            "scientist_upgrade_backlog": upgrade_backlog,
        },
        "recent_runs": recent,
    }


def _latest_run_signal(recent: list[dict[str, Any]]) -> dict[str, Any]:
    if not recent:
        return {
            "has_runs": False,
            "latest_run_id": "",
            "promotions": 0,
            "iterations": 0,
            "best_cv_score": None,
            "signal": "no_prior_run",
        }
    latest = recent[0]
    promotions = int(latest.get("promotions") or 0)
    iterations = int(latest.get("iterations") or 0)
    score = latest.get("best_cv_score")
    if iterations and promotions == 0:
        signal = "stalled_or_failed"
    elif promotions > 0:
        signal = "best_so_far_available"
    else:
        signal = "inconclusive"
    return {
        "has_runs": True,
        "latest_run_id": latest.get("run_id") or "",
        "promotions": promotions,
        "iterations": iterations,
        "best_cv_score": score,
        "signal": signal,
    }


def _decision_for_ready_task(checkpoint: dict[str, Any], task_type: str,
                             modality: str, metric: str) -> dict[str, Any]:
    recent = checkpoint.get("recent_runs", [])
    latest = _latest_run_signal(recent if isinstance(recent, list) else [])
    memory = checkpoint.get("memory", {})
    innovation = memory.get("innovation", {}) if isinstance(memory, dict) else {}
    retrospective = memory.get("retrospective", {}) if isinstance(memory, dict) else {}
    scientist_turns = memory.get("scientist_turns", {}) if isinstance(memory, dict) else {}
    upgrade_backlog = memory.get("scientist_upgrade_backlog", {}) if isinstance(memory, dict) else {}
    innovations_tried = int(innovation.get("innovations_tried") or 0) if isinstance(innovation, dict) else 0
    success_records = int(retrospective.get("success_records") or 0) if isinstance(retrospective, dict) else 0
    turn_reuse = _turn_reuse_records(scientist_turns if isinstance(scientist_turns, dict) else {})
    budget_exhausted_turns = (
        int(scientist_turns.get("budget_exhausted_turns") or 0)
        if isinstance(scientist_turns, dict)
        else 0
    )
    must_run_deferred = (
        scientist_turns.get("must_run_deferred_tools", [])
        if isinstance(scientist_turns, dict)
        else []
    )

    p0_upgrade_items = (
        upgrade_backlog.get("p0_items", [])
        if isinstance(upgrade_backlog, dict)
        else []
    )

    if budget_exhausted_turns > 0:
        selected_branch = "scientist_turn_budget_repair"
        code_generation_mode = "none"
        action = "complete_scientist_turn_closure"
        deferred_names = [
            str(item.get("tool") if isinstance(item, dict) else item)
            for item in must_run_deferred[:3]
            if item
        ]
        rationale = (
            "A previous Scientist turn deferred critical audit tools; close "
            f"{', '.join(deferred_names) or 'the must-run tool list'} before launching a new experiment."
        )
    elif p0_upgrade_items:
        selected_branch = "scientist_capability_upgrade"
        code_generation_mode = "none"
        action = "close_agent_upgrade_backlog"
        first = p0_upgrade_items[0] if isinstance(p0_upgrade_items[0], dict) else {}
        rationale = (
            "The latest self-audit contains P0 agent capability work; close "
            f"{first.get('id') or first.get('title') or 'the upgrade backlog'} before spending training compute."
        )
    elif not latest["has_runs"] and success_records > 0:
        selected_branch = "baseline_with_retrospective_transfer"
        code_generation_mode = "Base"
        action = "run_memory_guided_baseline"
        rationale = "No current-task run exists, but compatible memory exists; start from a baseline that explicitly cites reusable lessons."
    elif not latest["has_runs"] and turn_reuse:
        selected_branch = "baseline_with_scientist_turn_reuse"
        code_generation_mode = "Base"
        action = "run_turn_memory_guided_baseline"
        rationale = "No current-task run exists, but recent scientist turns contain reusable next actions; start from a baseline that carries those checks forward."
    elif not latest["has_runs"]:
        selected_branch = "baseline"
        code_generation_mode = "Base"
        action = "run_audited_baseline"
        rationale = "No prior run exists; establish a reproducible baseline before optimizing."
    elif latest["signal"] == "stalled_or_failed":
        selected_branch = "repair"
        code_generation_mode = "Diff"
        action = "repair_or_simplify_failed_frontier"
        rationale = "The latest run has no promotions; diagnose failure/stagnation before spending a new branch."
    elif innovations_tried >= 1:
        selected_branch = "cross_task_innovation"
        code_generation_mode = "Stepwise"
        action = "reuse_memory_and_test_novel_combination"
        rationale = "Prior innovation attempts exist; use memory-guided combinations while protecting best-so-far."
    else:
        selected_branch = "model_family_or_feature_engineering"
        code_generation_mode = "Stepwise"
        action = "improve_best_so_far"
        rationale = "A promoted best-so-far exists; try a bounded improvement branch without overwriting it."

    timeout = 2400 if "image" in modality.lower() else 1800
    if "time" in task_type.lower() or "forecast" in task_type.lower():
        validation = "time_aware_split_required"
    elif "classification" in task_type.lower():
        validation = "stratified_or_grouped_cv_if_applicable"
    else:
        validation = "kfold_with_target_transform_audit_if_applicable"

    return {
        "selected_action": action,
        "selected_branch": selected_branch,
        "code_generation_mode": code_generation_mode,
        "expected_delta": "local CV/OOF improvement only; official rank remains unknown without Kaggle response",
        "timeout_budget_seconds": timeout,
        "validation_focus": validation,
        "rollback_condition": (
            "hold candidate and keep best-so-far if validation_contract, submission_audit, "
            "claim_audit, required artifacts, or CV delta gate fail"
        ),
        "rationale": rationale,
        "latest_run_signal": latest,
    }


def build_research_decision(session: SessionState, root: Path, *,
                            persist: bool = False) -> dict[str, Any]:
    """Build the next audited experiment decision.

    The decision is the bridge between "thinking" and "doing": it chooses the
    safest next branch and records why. It never starts training and never
    claims official Kaggle rank/medal.
    """
    root = Path(root)
    checkpoint = build_scientist_checkpoint(session, root)
    task, task_source = _load_task_config(session, root)
    decision_path = root / ".xsci" / "scientist_decision.json"

    if not task:
        decision = {
            "selected_action": "select_task",
            "selected_branch": "none",
            "code_generation_mode": "none",
            "expected_delta": "none",
            "timeout_budget_seconds": 0,
            "validation_focus": "task_required",
            "rollback_condition": "no execution without a selected task",
            "rationale": "No task is selected, so EvoMind must not generate or run training code.",
            "latest_run_signal": _latest_run_signal([]),
        }
    elif checkpoint.get("gate", {}).get("planning_blockers"):
        blockers = checkpoint.get("gate", {}).get("planning_blockers") or []
        decision = {
            "selected_action": "fix_blocking_setup",
            "selected_branch": "gate_repair",
            "code_generation_mode": "none",
            "expected_delta": "none until setup gates clear",
            "timeout_budget_seconds": 0,
            "validation_focus": "readiness_gate",
            "rollback_condition": "no execution while blocking setup gates remain",
            "rationale": "Blocking setup gates must be cleared before any experiment can run.",
            "latest_run_signal": _latest_run_signal(checkpoint.get("recent_runs", [])),
            "blockers": blockers,
        }
    elif not checkpoint.get("gate", {}).get("planning_can_proceed"):
        warnings = checkpoint.get("gate", {}).get("warnings") or []
        decision = {
            "selected_action": "prepare_data_or_schema",
            "selected_branch": "data_readiness",
            "code_generation_mode": "none",
            "expected_delta": "none until train/test data are registered",
            "timeout_budget_seconds": 0,
            "validation_focus": "data_schema_audit",
            "rollback_condition": "do not launch training without local train.csv or an explicit GPU data path",
            "rationale": "Core gates may be clear, but data is not ready enough for a trustworthy run.",
            "latest_run_signal": _latest_run_signal(checkpoint.get("recent_runs", [])),
            "warnings": warnings,
        }
    else:
        decision = _decision_for_ready_task(
            checkpoint,
            str(task.get("task_type") or ""),
            str(task.get("modality") or ""),
            str(task.get("metric") or ""),
        )

    payload = {
        "ok": True,
        "tool": "research_decision",
        "selected_task": session.selected_task or "",
        "task_source": task_source,
        "metric": task.get("metric") if task else "",
        "metric_direction": task.get("metric_direction") if task else "",
        "decision": decision,
        "gates": checkpoint.get("gate", {}),
        "required_artifacts": [
            "agent_trace",
            "metrics.json",
            "OOF or validation predictions when applicable",
            "submission.csv when applicable",
            "artifact_manifest",
            "validation_contract",
            "score_promotion_gate",
            "claim_audit",
        ],
        "human_gate": {
            "official_kaggle_submit": "blocked_until_explicit_user_approval",
            "rank_or_medal_claims": "blocked_without_kaggle_response_artifact",
        },
    }
    memory_records = checkpoint.get("memory", {}).get("retrospective", {}).get("records", [])
    turn_summary = checkpoint.get("memory", {}).get("scientist_turns", {})
    upgrade_backlog = checkpoint.get("memory", {}).get("scientist_upgrade_backlog", {})
    hypotheses = checkpoint.get("hypotheses", [])
    payload["research_brief"] = {
        "hypotheses": hypotheses,
        "memory_reuse_records": [
            r for r in memory_records
            if r.get("what_worked") or r.get("reusable_strategy")
        ][:3],
        "scientist_turn_reuse_records": _turn_reuse_records(
            turn_summary if isinstance(turn_summary, dict) else {}
        ),
        "turn_derived_failure_avoidance": [
            f"Recent scientist turn blocker: {row.get('blocker')}."
            for row in (turn_summary.get("recurring_blockers", []) if isinstance(turn_summary, dict) else [])[:3]
            if isinstance(row, dict) and row.get("blocker")
        ],
        "turn_derived_evidence_gaps": (
            turn_summary.get("recurring_evidence_gaps", [])[:5]
            if isinstance(turn_summary, dict)
            else []
        ),
        "turn_derived_budget_risks": (
            {
                "budget_exhausted_turns": turn_summary.get("budget_exhausted_turns", 0),
                "must_run_deferred_tools": turn_summary.get("must_run_deferred_tools", [])[:8],
                "budget_risks": turn_summary.get("budget_risks", [])[:5],
            }
            if isinstance(turn_summary, dict)
            else {
                "budget_exhausted_turns": 0,
                "must_run_deferred_tools": [],
                "budget_risks": [],
            }
        ),
        "turn_derived_claim_boundaries": (
            turn_summary.get("claim_boundaries", [])[:6]
            if isinstance(turn_summary, dict)
            else []
        ),
        "agent_capability_gate": (
            {
                "status": "upgrade_required_before_training"
                if int(upgrade_backlog.get("p0_count") or 0) > 0
                else "clear_or_no_p0_backlog",
                "overall_score": upgrade_backlog.get("overall_score"),
                "launch_readiness": upgrade_backlog.get("launch_readiness", ""),
                "p0_count": upgrade_backlog.get("p0_count", 0),
                "open_count": upgrade_backlog.get("open_count", 0),
                "artifact": upgrade_backlog.get("artifact", ""),
                "self_audit_artifact": upgrade_backlog.get("self_audit_artifact", ""),
            }
            if isinstance(upgrade_backlog, dict)
            else {
                "status": "not_run",
                "overall_score": None,
                "launch_readiness": "",
                "p0_count": 0,
                "open_count": 0,
                "artifact": "",
                "self_audit_artifact": "",
            }
        ),
        "agent_upgrade_backlog": (
            upgrade_backlog.get("open_items", [])[:8]
            if isinstance(upgrade_backlog, dict)
            else []
        ),
        "recent_decision_context": (
            turn_summary.get("recent_decisions", [])[:5]
            if isinstance(turn_summary, dict)
            else []
        ),
        "failure_avoidance": _failure_avoidance(
            memory_records,
            checkpoint.get("recent_runs", []),
            turn_summary if isinstance(turn_summary, dict) else {},
        ),
        "experiment_plan": _experiment_plan(decision, hypotheses if isinstance(hypotheses, list) else []),
        "claim_boundary": (
            "Only local CV/OOF/proxy claims are allowed until an official Kaggle response artifact exists."
        ),
    }

    if persist:
        try:
            decision_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = decision_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(decision_path)
            payload["artifact_path"] = str(decision_path)
        except OSError as exc:
            payload["ok"] = False
            payload["message"] = f"Could not write decision artifact: {exc}"
    return payload


def build_scientist_workplan(session: SessionState, root: Path, *,
                             persist: bool = False) -> dict[str, Any]:
    """Build a recoverable multi-step AI Scientist workplan.

    This is the durable "planner/executor boundary" for EvoMind.  It turns the
    current checkpoint and decision into explicit steps, gates, evidence
    requirements, and resume commands.  It never starts training; execution
    still requires the normal run/workstation path.
    """
    root = Path(root)
    checkpoint = build_scientist_checkpoint(session, root)
    decision_payload = build_research_decision(session, root, persist=persist)
    steps = _build_workplan_steps(checkpoint, decision_payload)
    focus = _workplan_focus(steps)
    decision = decision_payload.get("decision", {}) if isinstance(decision_payload.get("decision"), dict) else {}
    gate = checkpoint.get("gate", {}) if isinstance(checkpoint.get("gate"), dict) else {}
    can_execute = bool(gate.get("can_execute"))
    task = session.selected_task or ""
    ready_steps = [step for step in steps if step.get("status") == "ready"]
    blocked_steps = [step for step in steps if step.get("status") == "blocked"]
    pending_steps = [step for step in steps if step.get("status") == "pending"]
    artifact_path = root / ".xsci" / "scientist_workplan.json"

    if can_execute and ready_steps:
        mode = "ready_for_gated_execution"
        autonomy_level = "planner_plus_gated_executor"
    elif blocked_steps:
        mode = "blocked_by_gate"
        autonomy_level = "planner_only_until_gate_clears"
    else:
        mode = "planning"
        autonomy_level = "planner_only"

    payload: dict[str, Any] = {
        "ok": True,
        "tool": "scientist_workplan",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selected_task": task,
        "mode": mode,
        "autonomy_level": autonomy_level,
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
        "current_focus": focus,
        "summary": {
            "steps_total": len(steps),
            "completed": sum(1 for step in steps if step.get("status") == "completed"),
            "ready": len(ready_steps),
            "pending": len(pending_steps),
            "blocked": len(blocked_steps),
            "selected_action": decision.get("selected_action", ""),
            "selected_branch": decision.get("selected_branch", ""),
            "code_generation_mode": decision.get("code_generation_mode", ""),
        },
        "steps": steps,
        "resume_commands": [
            "evomind autopilot",
            "evomind workplan",
            f"evomind decide {task}".strip(),
        ] + ([f"evomind run {task}"] if can_execute and task else []),
        "required_artifacts": decision_payload.get("required_artifacts", []),
        "research_brief": decision_payload.get("research_brief", {}),
        "human_gate": decision_payload.get("human_gate", {}),
        "decision_artifact_path": decision_payload.get("artifact_path") or str(root / ".xsci" / "scientist_decision.json"),
        "artifact_path": str(artifact_path),
        "claim_boundary": (
            "Workplan readiness is not a score claim. Rank, medal, and top30 remain blocked "
            "without an official Kaggle response artifact."
        ),
    }

    if persist:
        try:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = artifact_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(artifact_path)
        except OSError as exc:
            payload["ok"] = False
            payload["message"] = f"Could not write workplan artifact: {exc}"
        try:
            from .scientist_trace import record_scientist_step_event

            trace_run_id = f"workplan_{payload['generated_at'].replace(':', '').replace('+', 'Z')}"
            record_scientist_step_event(root, {
                "trace_run_id": trace_run_id,
                "source": "scientist_workplan",
                "task": task,
                "phase": "workplan_snapshot",
                "status": "completed" if payload.get("ok") else "failed",
                "tool": "scientist_workplan",
                "message": f"Workplan generated with {len(steps)} steps; mode={mode}.",
                "artifact_path": str(artifact_path),
                "details": {
                    "mode": mode,
                    "autonomy_level": autonomy_level,
                    "current_focus": focus,
                    "summary": payload.get("summary", {}),
                },
                "no_training_started": True,
            })
            for step in steps:
                record_scientist_step_event(root, {
                    "trace_run_id": trace_run_id,
                    "source": "scientist_workplan",
                    "task": task,
                    "phase": "workplan_step",
                    "step_id": step.get("id", ""),
                    "status": step.get("status", "pending"),
                    "tool": step.get("tool", ""),
                    "message": step.get("title", ""),
                    "artifact_path": str(artifact_path),
                    "gate": step.get("gate", ""),
                    "evidence": step.get("evidence", []),
                    "details": {
                        "action": step.get("action", ""),
                        "blocked_reason": step.get("blocked_reason", ""),
                        "owner": step.get("owner", ""),
                    },
                    "no_training_started": True,
                })
        except Exception:
            pass
    return payload
