"""Evidence-grounded research synthesis for EvoMind Scientist turns.

Tool orchestration is necessary but not sufficient for an intelligent research
agent. This module turns the collected task, memory, gate, and artifact evidence
into a direct answer that satisfies the user's requested scientific deliverables.
It never starts training or submits to Kaggle.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .kaggle_session import SessionState

_SENSITIVE_KEY_RE = re.compile(
    r"(api[_-]?key|token|cookie|password|passwd|secret|private[_-]?key|ssh[_-]?key)",
    re.IGNORECASE,
)
_SENSITIVE_VALUE_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|cookie|password|passwd|secret)\s*[:=]\s*\S+"
)
_NO_EXECUTION_RE = re.compile(
    r"(不要|不需要|无需|不允许|禁止).{0,12}(训练|建模|运行|执行|提交)|"
    r"(do not|don't|dont|without).{0,20}(train|run|execute|submit)|"
    r"(只|仅).{0,8}(分析|规划|研究|比较|诊断)",
    re.IGNORECASE,
)
_UNSAFE_COMMAND_RE = re.compile(
    r"\b(evomind\s+run|kaggle(?:-official)?\s+competitions\s+submit|submit|train)\b",
    re.IGNORECASE,
)
_VOLATILE_CACHE_KEYS = {
    "generated_at",
    "created_at",
    "updated_at",
    "decided_at",
    "trace_run_id",
    "turn_id",
    "artifact_path",
    "markdown_artifact_path",
    "history_path",
    "path",
    "source_paths",
    "cache_key",
    "cache_hit",
}
_SEMANTIC_CACHE_KEYS = {
    "tool",
    "selected_task",
    "task_profile",
    "task_slug",
    "task_name",
    "modality",
    "task_type",
    "target_column",
    "id_column",
    "data_schema",
    "extra_notes",
    "intent",
    "kind",
    "payload",
    "autonomy_level",
    "readiness",
    "llm_ready",
    "kaggle_ready",
    "compute_backend",
    "gpu_ready",
    "gpu_blocked",
    "can_execute",
    "advisory_gaps",
    "active_strategy",
    "present",
    "scientific_critique",
    "memory_digest",
    "memory_summary",
    "requirement_context",
    "open_requirements",
    "blocked_requirements",
    "execution_partition",
    "decision",
    "selected_strategy",
    "selected_hypothesis",
    "hypothesis_id",
    "strategy_name",
    "branch_type",
    "code_generation_mode",
    "experiment_blueprint",
    "blueprint_id",
    "blueprint_status",
    "gate_summary",
    "go_no_go",
    "status",
    "mode",
    "next_safe_command",
    "next_safe_commands",
    "blocking_gates",
    "blockers",
    "root_causes",
    "metric",
    "metric_direction",
    "score",
    "cv_score",
    "best_score",
    "recent_run_id",
    "data_ready",
    "train_csv",
    "test_csv",
    "strategy_posture",
    "selected_action",
    "selected_command",
    "gate_status",
    "why",
    "evidence_strength",
    "risk",
    "cost",
    "expected_value",
    "hypotheses_reviewed",
    "reviews",
    "recommendation",
}
_CACHE_FACT_KEYS = {
    "selected_task",
    "task_slug",
    "task_type",
    "modality",
    "metric",
    "metric_direction",
    "target_column",
    "kind",
    "payload",
    "autonomy_level",
    "llm_ready",
    "kaggle_ready",
    "compute_backend",
    "gpu_ready",
    "gpu_blocked",
    "can_execute",
    "selected_action",
    "selected_command",
    "gate_status",
    "go_no_go",
    "status",
    "mode",
    "next_safe_command",
    "blocking_gates",
    "blockers",
    "root_causes",
    "open_requirements",
    "blocked_requirements",
    "hypothesis_id",
    "strategy_name",
    "branch_type",
    "code_generation_mode",
    "blueprint_id",
    "blueprint_status",
    "recommendation",
    "cv_score",
    "best_score",
    "recent_run_id",
    "data_ready",
    "train_csv",
    "test_csv",
    "retrospective_records",
    "task_relevant_records",
    "exact_task_records",
}

_CHINESE_NUMBERS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def _safe_text(value: Any, *, limit: int = 1600) -> str:
    text = " ".join(str(value or "").replace("\x00", " ").split())
    text = _SENSITIVE_VALUE_RE.sub(r"\1=[redacted]", text)
    return text[:limit]


def _safe_json(value: Any, *, depth: int = 0) -> Any:
    if depth > 6:
        return "[truncated]"
    if isinstance(value, str):
        return _safe_text(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_safe_json(item, depth=depth + 1) for item in value[:30]]
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in list(value.items())[:100]:
            key_text = str(key)
            output[key_text] = (
                "[redacted]"
                if _SENSITIVE_KEY_RE.search(key_text)
                else _safe_json(item, depth=depth + 1)
            )
        return output
    return _safe_text(value)


def _stable_cache_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 7:
        return "[truncated]"
    if isinstance(value, dict):
        return {
            str(key): _stable_cache_value(item, depth=depth + 1)
            for key, item in value.items()
            if str(key) not in _VOLATILE_CACHE_KEYS
        }
    if isinstance(value, list):
        return [_stable_cache_value(item, depth=depth + 1) for item in value]
    return value


def _semantic_cache_projection(evidence: dict[str, Any]) -> dict[str, Any]:
    facts: dict[str, Any] = {}

    def source(name: str) -> dict[str, Any]:
        value = evidence.get(name)
        return value if isinstance(value, dict) else {}

    def read_path(payload: dict[str, Any], *keys: str) -> Any:
        value: Any = payload
        for key in keys:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return value

    def add(label: str, value: Any) -> None:
        if value in (None, "", [], {}):
            return
        if isinstance(value, list):
            facts[label] = sorted(_safe_text(item, limit=500) for item in value)
        else:
            facts[label] = _safe_json(value)

    turn_plan = source("turn_plan")
    add("turn.intent.kind", read_path(turn_plan, "intent", "kind"))
    add("turn.intent.payload", read_path(turn_plan, "intent", "payload"))
    add("turn.autonomy_level", turn_plan.get("autonomy_level"))
    add("turn.next_safe_command", turn_plan.get("next_safe_command"))
    for key in ("llm_ready", "kaggle_ready", "compute_backend", "gpu_ready", "gpu_blocked", "can_execute", "blocking_gates"):
        add(f"turn.readiness.{key}", read_path(turn_plan, "readiness", key))

    context = source("context_packet")
    for key in ("task_slug", "task_type", "modality", "metric", "metric_direction", "target_column"):
        add(f"context.task.{key}", read_path(context, "task_profile", key))
    for key in ("llm_ready", "kaggle_ready", "compute_backend", "gpu_ready", "gpu_blocked", "can_execute", "blocking_gates"):
        add(f"context.readiness.{key}", read_path(context, "readiness", key))
    for key in ("selected_action", "selected_command", "gate_status"):
        add(f"context.strategy.{key}", read_path(context, "active_strategy", key))
    add("context.requirements.open", read_path(context, "requirement_context", "open_requirements"))
    add("context.requirements.blocked", read_path(context, "requirement_context", "blocked_requirements"))
    for key in ("retrospective_records", "task_relevant_records"):
        add(f"context.memory.{key}", read_path(context, "memory_digest", key))

    system = source("system_status")
    for key in ("llm_ready", "kaggle_ready", "compute_backend", "gpu_blocked", "selected_task", "recent_run_id", "blockers"):
        add(f"system.{key}", system.get(key))

    data = source("data_check")
    for key in ("train_csv", "test_csv", "data_ready", "status", "message"):
        add(f"data.{key}", data.get(key))

    strategy = source("scientist_strategy_optimizer")
    for key in ("strategy_posture", "next_safe_command"):
        add(f"strategy.{key}", strategy.get(key))
    for key in ("id", "gate_status", "command"):
        add(f"strategy.selected.{key}", read_path(strategy, "selected_strategy", key))

    review = source("scientist_hypothesis_review")
    add("review.recommendation", review.get("recommendation"))
    for key in ("hypothesis_id", "strategy_name", "status", "branch_type", "code_generation_mode"):
        add(f"review.selected.{key}", read_path(review, "selected_hypothesis", key))
    for key in ("data_ready", "execution_contract"):
        add(f"review.gate.{key}", read_path(review, "gate_summary", key))

    blueprint = source("scientist_experiment_blueprint")
    add("blueprint.status", blueprint.get("blueprint_status"))
    for key in ("hypothesis_id", "strategy_name", "status", "branch_type", "code_generation_mode"):
        add(f"blueprint.selected.{key}", read_path(blueprint, "selected_hypothesis", key))
    add("blueprint.id", read_path(blueprint, "experiment_blueprint", "blueprint_id"))

    contract = source("scientist_execution_contract")
    add("contract.go_no_go", contract.get("go_no_go"))
    add("contract.root_causes", contract.get("root_causes"))

    repair = source("scientist_repair_plan")
    add("repair.mode", repair.get("mode"))
    add("repair.root_causes", repair.get("root_causes"))
    add("repair.selected_action", read_path(repair, "decision", "selected_action"))

    adaptive = source("adaptive_tool_loop")
    for key in (
        "status", "stop_reason", "executed_tools", "dynamic_tool_selection",
        "failure_observed", "replanned_after_failure", "open_requirements",
    ):
        add(f"adaptive.{key}", adaptive.get(key))
    for item in adaptive.get("tool_calls") or []:
        if not isinstance(item, dict):
            continue
        tool_name = _safe_text(item.get("tool"), limit=120)
        if not tool_name:
            continue
        add(f"adaptive.tool.{tool_name}.ok", item.get("ok"))
        add(f"adaptive.tool.{tool_name}.summary", item.get("summary"))
        add(f"adaptive.tool.{tool_name}.artifact_path", item.get("artifact_path"))

    continuation_resume = source("continuation_resume")
    for key in (
        "status", "stop_reason", "executed_tools", "remaining_safe_tools",
    ):
        add(f"continuation_resume.{key}", continuation_resume.get(key))
    for item in continuation_resume.get("steps") or []:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        if index in (None, ""):
            continue
        for key in (
            "status", "executed_tool", "selected_action_id",
            "before_remaining_safe_tools", "after_remaining_safe_tools",
        ):
            add(f"continuation_resume.step.{index}.{key}", item.get(key))

    for source_name in (
        "scientist_self_audit",
        "scientist_causal_diagnosis",
        "scientist_engineering_loop",
        "scientist_memory_consolidation",
    ):
        payload = source(source_name)
        for key in ("status", "mode", "message", "artifact_path", "overall_score", "next_safe_command"):
            add(f"{source_name}.{key}", payload.get(key))

    recent = source("recent_run")
    for key in ("run_id", "recent_run_id", "cv_score", "best_score", "status"):
        add(f"recent.{key}", recent.get(key))

    return _stable_cache_value(facts)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _update_cache_stats(
    path: Path,
    *,
    hit: bool,
    cache_key: str,
    generated_at: str,
) -> dict[str, Any]:
    previous = _read_json(path) or {}
    cache_algorithm = "semantic_v5"
    if previous.get("cache_algorithm") != cache_algorithm:
        previous = {}
    requests = int(previous.get("requests") or 0) + 1
    hits = int(previous.get("hits") or 0) + (1 if hit else 0)
    misses = int(previous.get("misses") or 0) + (0 if hit else 1)
    payload = {
        "schema": "evomind.ai_scientist.reasoning_cache_stats.v1",
        "cache_algorithm": cache_algorithm,
        "updated_at": generated_at,
        "requests": requests,
        "hits": hits,
        "misses": misses,
        "hit_ratio": round(hits / requests, 4) if requests else 0.0,
        "last_cache_key": cache_key,
        "last_result": "hit" if hit else "miss",
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
    return payload


def _requested_hypothesis_count(goal: str) -> int:
    patterns = (
        r"([1-9])\s*(?:个|条|组)?(?:可证伪的?)?(?:提分|研究|实验)?假设",
        r"([一二两三四五六七八九十])\s*(?:个|条|组)?(?:可证伪的?)?(?:提分|研究|实验)?假设",
        r"(?:propose|give|generate)\s+([1-9])\s+(?:falsifiable\s+)?hypotheses",
    )
    for pattern in patterns:
        match = re.search(pattern, goal, flags=re.IGNORECASE)
        if not match:
            continue
        raw = match.group(1)
        if raw.isdigit():
            return max(1, min(6, int(raw)))
        if raw in _CHINESE_NUMBERS:
            return _CHINESE_NUMBERS[raw]
    return 3 if ("假设" in goal or "hypoth" in goal.lower()) else 0


def _request_contract(goal: str) -> dict[str, Any]:
    low = goal.lower()
    hypothesis_count = _requested_hypothesis_count(goal)
    analysis_only = bool(_NO_EXECUTION_RE.search(goal))
    return {
        "analysis_only": analysis_only,
        "requested_hypothesis_count": hypothesis_count,
        "requires_falsifiability": hypothesis_count > 0 or "可证伪" in goal or "falsifiable" in low,
        "requires_comparison": any(
            item in low
            for item in ("比较", "对比", "证据", "风险", "成本", "compare", "risk", "cost", "evidence")
        ),
        "requires_selection": any(
            item in low
            for item in ("选择", "选出", "优先", "下一步", "choose", "select", "next")
        ),
        "requires_context_recovery": any(
            item in low
            for item in ("恢复上下文", "上下文", "之前", "记忆", "context", "memory", "resume")
        ),
    }


def _compact_artifact(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    preferred = (
        "tool",
        "selected_task",
        "task_profile",
        "context_quality",
        "readiness",
        "active_strategy",
        "memory_digest",
        "memory_summary",
        "requirement_context",
        "scientific_critique",
        "decision",
        "selected_strategy",
        "intervention_ranking",
        "hypotheses_reviewed",
        "reviews",
        "selected_hypothesis",
        "experiment_blueprint",
        "gate_summary",
        "go_no_go",
        "blockers",
        "root_causes",
        "next_safe_command",
        "next_safe_commands",
        "message",
    )
    return _safe_json({key: payload.get(key) for key in preferred if key in payload})


def _load_evidence(root: Path, supplied: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(supplied, dict) and supplied:
        return _safe_json(supplied)
    xsci = root / ".xsci"
    names = (
        "scientist_context_packet.json",
        "scientist_situation_model.json",
        "scientist_strategy_optimizer.json",
        "scientist_innovation_backlog.json",
        "scientist_hypothesis_review.json",
        "scientist_experiment_blueprint.json",
        "scientist_execution_contract.json",
        "scientist_repair_plan.json",
    )
    return {
        name.removesuffix(".json"): _compact_artifact(_read_json(xsci / name))
        for name in names
    }


def _task_profile(evidence: dict[str, Any], session: SessionState) -> dict[str, Any]:
    candidates: list[Any] = [
        evidence.get("task_profile"),
        (evidence.get("context_packet") or {}).get("task_profile")
        if isinstance(evidence.get("context_packet"), dict)
        else None,
    ]
    for value in evidence.values():
        if isinstance(value, dict):
            candidates.append(value.get("task_profile"))
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            return _safe_json(candidate)
    return {
        "task_slug": session.selected_task or "",
        "task_name": session.selected_task or "",
        "modality": "unknown",
        "task_type": "unknown",
        "metric": "unknown",
        "metric_direction": "unknown",
    }


def _blocking_gates(evidence: dict[str, Any]) -> list[str]:
    gates: list[str] = []
    for value in evidence.values():
        if not isinstance(value, dict):
            continue
        readiness = value.get("readiness")
        if isinstance(readiness, dict):
            gates.extend(str(item) for item in readiness.get("blocking_gates") or [])
        gates.extend(str(item) for item in value.get("blocking_gates") or [])
        gates.extend(str(item) for item in value.get("blockers") or [])
    return list(dict.fromkeys(_safe_text(item, limit=500) for item in gates if item))[:8]


def _default_hypotheses(task: dict[str, Any], count: int) -> list[dict[str, Any]]:
    modality = str(task.get("modality") or "").lower()
    task_type = str(task.get("task_type") or "").lower()
    metric = str(task.get("metric") or "declared metric")
    target = str(task.get("target_column") or "target")

    if "time" in modality or "forecast" in task_type:
        templates = [
            (
                "Leakage-safe temporal validation",
                "A rolling-origin split plus lag-availability audit will reduce optimistic validation bias.",
                "The candidate should improve fold-to-fold stability without using future information.",
            ),
            (
                "Residual seasonality model",
                "Calendar and group-level residual features capture signal left by the current baseline.",
                "Residual autocorrelation and the declared error metric should both fall on held-out periods.",
            ),
            (
                "Horizon-aware ensemble",
                "Different model families dominate at different forecast horizons.",
                "An OOF-weighted horizon blend should beat each component on the same backtest folds.",
            ),
        ]
    elif "image" in modality or "vision" in modality:
        templates = [
            (
                "Augmentation ablation",
                "A task-compatible augmentation policy improves generalization without label distortion.",
                "The same-fold metric should improve and calibration should not deteriorate.",
            ),
            (
                "Backbone diversity",
                "A second pretrained backbone contributes complementary errors.",
                "OOF error overlap should be below the current model and a gated blend should improve.",
            ),
            (
                "Test-time calibration",
                "Calibrated test-time augmentation reduces variance more than it adds bias.",
                "OOF calibration error and the declared metric should improve together.",
            ),
        ]
    elif "regression" in task_type or metric.lower() in {"rmse", "rmsle", "mae"}:
        templates = [
            (
                "Target transformation and residual audit",
                f"A leakage-safe transform of {target} plus inverse-transform bias correction better matches the error geometry.",
                f"Same-split {metric} improves and residual bias stays bounded across target quantiles.",
            ),
            (
                "Categorical model-family diversity",
                "Native categorical boosting captures interactions missed by one-hot or ordinal baselines.",
                f"OOF {metric} improves on the identical folds and category-frequency slices show consistent gains.",
            ),
            (
                "OOF residual blend",
                "Linear, bagged-tree, and boosted-tree residuals are sufficiently complementary for a constrained blend.",
                f"A nested OOF weight search beats every component on {metric} without increasing fold variance.",
            ),
            (
                "Outlier-aware validation",
                "The current score is dominated by a small high-leverage regime that needs robust loss or stratified folds.",
                f"Robust training lowers worst-fold {metric} while preserving median-fold performance.",
            ),
        ]
    else:
        templates = [
            (
                "Leakage-safe feature family",
                "A task-specific feature family adds signal not represented in the baseline.",
                f"Same-split {metric} improves in an isolated ablation with stable folds.",
            ),
            (
                "Model-family diversity",
                "A second model family makes complementary OOF errors.",
                f"Error correlation falls and an OOF-gated blend improves {metric}.",
            ),
            (
                "Calibration and threshold audit",
                "The ranking model is stronger than the current decision rule.",
                f"OOF calibration or threshold search improves {metric} without leakage.",
            ),
        ]

    hypotheses: list[dict[str, Any]] = []
    for index, (title, mechanism, prediction) in enumerate(templates[: max(1, count)], start=1):
        hypotheses.append(
            {
                "id": f"H{index}",
                "title": title,
                "mechanism": mechanism,
                "falsifiable_prediction": prediction,
                "required_evidence": [
                    "same-split OOF or validation predictions",
                    f"declared metric: {metric}",
                    "fold stability and error-slice diagnostics",
                ],
                "experiment": "Run one controlled ablation against the current best-so-far using identical folds and seeds.",
                "success_threshold": f"Positive {metric} delta with no hard fold-stability or leakage failure.",
                "disconfirming_result": "No same-split gain, unstable fold behavior, or improvement that disappears after leakage controls.",
                "evidence_strength": "medium",
                "risk": "medium",
                "cost": "low_to_medium",
                "expected_value": "medium",
                "epistemic_status": "proposed_unvalidated",
            }
        )
    return hypotheses


def _fallback_payload(
    goal: str,
    contract: dict[str, Any],
    task: dict[str, Any],
    blockers: list[str],
) -> dict[str, Any]:
    count = max(3, int(contract.get("requested_hypothesis_count") or 0))
    hypotheses = _default_hypotheses(task, count)
    comparison = [
        {
            "hypothesis_id": item["id"],
            "evidence": item["evidence_strength"],
            "risk": item["risk"],
            "cost": item["cost"],
            "expected_value": item["expected_value"],
            "priority_score": max(40, 82 - index * 7),
        }
        for index, item in enumerate(hypotheses)
    ]
    next_command = "evomind repair" if blockers else "evomind blueprint"
    selected = hypotheses[0]
    return {
        "reasoning_mode": "analysis_only" if contract.get("analysis_only") else "research_planning",
        "direct_answer": (
            "Current artifacts prove orchestration and gates, but they do not yet prove that self-evolution improves the task metric under a controlled same-split experiment."
        ),
        "problem_frame": {
            "known": [
                f"Selected task: {task.get('task_slug') or task.get('task_name') or 'unknown'}",
                f"Metric: {task.get('metric') or 'unknown'}",
                "The workstation preserves best-so-far, claim, and submission gates.",
            ],
            "unknown": [
                "Whether remembered strategies transfer to this task under identical folds.",
                "Whether any candidate produces a reproducible positive metric delta.",
            ],
            "why_not_proven": [
                "Artifact existence is not behavioral evidence of better scientific decisions.",
                "Unvalidated proposals must not be recorded as successful self-evolution.",
            ],
        },
        "hypotheses": hypotheses,
        "comparison": comparison,
        "selected_hypothesis_id": selected["id"],
        "selected_rationale": (
            "It has the lowest initial compute cost, a clear disconfirming result, and can be tested without changing the validation split."
        ),
        "next_safe_action": {
            "command": next_command,
            "action": "Materialize the selected controlled experiment blueprint without starting training.",
            "gate": "analysis_only_or_execution_contract_gate",
            "expected_evidence": [
                "scientist_experiment_blueprint.json",
                "validation_contract.json",
                "rollback_condition",
            ],
        },
        "unresolved_questions": blockers[:4],
        "claim_boundaries": [
            "No claim of self-evolution improvement until a controlled same-split result passes the promotion gate.",
            "No official score, rank, medal, or top30 claim without a Kaggle response artifact.",
        ],
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_hypothesis(item: Any, index: int) -> dict[str, Any]:
    source = item if isinstance(item, dict) else {}
    return {
        "id": _safe_text(source.get("id") or f"H{index}", limit=40),
        "title": _safe_text(source.get("title") or source.get("name") or f"Hypothesis {index}", limit=180),
        "mechanism": _safe_text(source.get("mechanism") or source.get("rationale"), limit=700),
        "falsifiable_prediction": _safe_text(source.get("falsifiable_prediction") or source.get("prediction"), limit=700),
        "required_evidence": [
            _safe_text(value, limit=300)
            for value in (source.get("required_evidence") or source.get("evidence") or [])
            if _safe_text(value, limit=300)
        ][:8],
        "experiment": _safe_text(source.get("experiment") or source.get("test"), limit=700),
        "success_threshold": _safe_text(source.get("success_threshold") or source.get("success_criterion"), limit=500),
        "disconfirming_result": _safe_text(source.get("disconfirming_result") or source.get("falsifier"), limit=500),
        "evidence_strength": _safe_text(source.get("evidence_strength") or "unknown", limit=40),
        "risk": _safe_text(source.get("risk") or "unknown", limit=80),
        "cost": _safe_text(source.get("cost") or "unknown", limit=80),
        "expected_value": _safe_text(source.get("expected_value") or "unknown", limit=80),
        "epistemic_status": "proposed_unvalidated",
    }


def _normalize_payload(
    raw: dict[str, Any],
    fallback: dict[str, Any],
    contract: dict[str, Any],
    task: dict[str, Any],
    blockers: list[str],
) -> dict[str, Any]:
    required_count = int(contract.get("requested_hypothesis_count") or 0)
    source_hypotheses = raw.get("hypotheses") if isinstance(raw.get("hypotheses"), list) else []
    hypotheses = [
        _normalize_hypothesis(item, index)
        for index, item in enumerate(source_hypotheses, start=1)
    ]
    if required_count and len(hypotheses) < required_count:
        defaults = _default_hypotheses(task, required_count)
        existing_titles = {item["title"].lower() for item in hypotheses}
        for item in defaults:
            if item["title"].lower() not in existing_titles:
                hypotheses.append(item)
            if len(hypotheses) >= required_count:
                break
    if not hypotheses:
        hypotheses = fallback["hypotheses"]

    comparison = raw.get("comparison") if isinstance(raw.get("comparison"), list) else []
    if not comparison:
        comparison = fallback["comparison"]
    normalized_comparison = [
        _safe_json(item)
        for item in comparison
        if isinstance(item, dict)
    ][:8]

    next_action = raw.get("next_safe_action") if isinstance(raw.get("next_safe_action"), dict) else {}
    next_action = {
        **fallback["next_safe_action"],
        **_safe_json(next_action),
    }
    command = _safe_text(next_action.get("command"), limit=160)
    if contract.get("analysis_only") and _UNSAFE_COMMAND_RE.search(command):
        next_action["command"] = "evomind repair" if blockers else "evomind blueprint"
        next_action["gate"] = "analysis_only_gate"

    selected_id = _safe_text(
        raw.get("selected_hypothesis_id") or fallback["selected_hypothesis_id"],
        limit=80,
    )
    valid_ids = {item["id"] for item in hypotheses}
    if selected_id not in valid_ids:
        selected_id = hypotheses[0]["id"]

    return {
        "reasoning_mode": _safe_text(
            raw.get("reasoning_mode")
            or ("analysis_only" if contract.get("analysis_only") else "research_planning"),
            limit=80,
        ),
        "direct_answer": _safe_text(raw.get("direct_answer") or fallback["direct_answer"], limit=1600),
        "problem_frame": _safe_json(
            raw.get("problem_frame")
            if isinstance(raw.get("problem_frame"), dict)
            else fallback["problem_frame"]
        ),
        "hypotheses": hypotheses[: max(required_count, 6) if required_count else 6],
        "comparison": normalized_comparison,
        "selected_hypothesis_id": selected_id,
        "selected_rationale": _safe_text(
            raw.get("selected_rationale") or fallback["selected_rationale"],
            limit=1000,
        ),
        "next_safe_action": next_action,
        "unresolved_questions": [
            _safe_text(item, limit=500)
            for item in (
                raw.get("unresolved_questions")
                if isinstance(raw.get("unresolved_questions"), list)
                else fallback["unresolved_questions"]
            )
        ][:8],
        "claim_boundaries": [
            _safe_text(item, limit=500)
            for item in (
                raw.get("claim_boundaries")
                if isinstance(raw.get("claim_boundaries"), list)
                else fallback["claim_boundaries"]
            )
        ][:8],
    }


def _quality(payload: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    hypotheses = payload.get("hypotheses") if isinstance(payload.get("hypotheses"), list) else []
    required_count = int(contract.get("requested_hypothesis_count") or 0)
    complete_hypotheses = [
        item
        for item in hypotheses
        if isinstance(item, dict)
        and item.get("mechanism")
        and item.get("falsifiable_prediction")
        and item.get("experiment")
        and item.get("success_threshold")
        and item.get("disconfirming_result")
    ]
    comparison = payload.get("comparison") if isinstance(payload.get("comparison"), list) else []
    next_action = payload.get("next_safe_action") if isinstance(payload.get("next_safe_action"), dict) else {}
    checks = {
        "direct_answer": bool(payload.get("direct_answer")),
        "problem_frame": bool(payload.get("problem_frame")),
        "hypothesis_count": not required_count or len(hypotheses) >= required_count,
        "falsifiability": not required_count or len(complete_hypotheses) >= required_count,
        "comparison": not contract.get("requires_comparison") or len(comparison) >= max(1, required_count),
        "selection": not contract.get("requires_selection") or bool(payload.get("selected_hypothesis_id")),
        "next_safe_action": bool(next_action.get("command") and next_action.get("gate")),
        "claim_boundaries": bool(payload.get("claim_boundaries")),
    }
    weights = {
        "direct_answer": 15,
        "problem_frame": 10,
        "hypothesis_count": 15,
        "falsifiability": 20,
        "comparison": 15,
        "selection": 10,
        "next_safe_action": 10,
        "claim_boundaries": 5,
    }
    score = sum(weights[name] for name, passed in checks.items() if passed)
    return {
        "score": score,
        "status": "strong" if score >= 85 else "usable" if score >= 70 else "insufficient",
        "checks": checks,
        "missing_contract_items": [name for name, passed in checks.items() if not passed],
        "hypotheses_requested": required_count,
        "hypotheses_produced": len(hypotheses),
        "complete_falsifiable_hypotheses": len(complete_hypotheses),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "## Scientist answer",
        "",
        str(payload.get("direct_answer") or ""),
        "",
        "### Problem frame",
    ]
    frame = payload.get("problem_frame") if isinstance(payload.get("problem_frame"), dict) else {}
    for label, key in (("Known", "known"), ("Unknown", "unknown"), ("Why not proven", "why_not_proven")):
        values = frame.get(key) if isinstance(frame.get(key), list) else []
        if values:
            lines.append(f"**{label}**")
            lines.extend(f"- {value}" for value in values)
    lines.extend(["", "### Falsifiable hypotheses"])
    for item in payload.get("hypotheses") or []:
        if not isinstance(item, dict):
            continue
        lines.extend(
            [
                "",
                f"**{item.get('id')}: {item.get('title')}**",
                f"- Mechanism: {item.get('mechanism')}",
                f"- Prediction: {item.get('falsifiable_prediction')}",
                f"- Test: {item.get('experiment')}",
                f"- Success: {item.get('success_threshold')}",
                f"- Disconfirming result: {item.get('disconfirming_result')}",
                f"- Evidence/Risk/Cost: {item.get('evidence_strength')} / {item.get('risk')} / {item.get('cost')}",
            ]
        )
    lines.extend(
        [
            "",
            "### Decision",
            f"- Selected: {payload.get('selected_hypothesis_id')}",
            f"- Why: {payload.get('selected_rationale')}",
        ]
    )
    next_action = payload.get("next_safe_action") if isinstance(payload.get("next_safe_action"), dict) else {}
    lines.extend(
        [
            "",
            "### Next safe action",
            f"- Action: {next_action.get('action')}",
            f"- Command: `{next_action.get('command')}`",
            f"- Gate: {next_action.get('gate')}",
            "",
            "### Claim boundaries",
        ]
    )
    lines.extend(f"- {item}" for item in payload.get("claim_boundaries") or [])
    return "\n".join(lines).strip()


def build_scientist_reasoning_synthesis(
    session: SessionState,
    root: Path,
    *,
    goal: str | None = None,
    evidence: dict[str, Any] | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    root = Path(root)
    xsci = root / ".xsci"
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    user_goal = _safe_text(goal or session.last_goal or "Analyze the current research state.", limit=4000)
    request_contract = _request_contract(user_goal)
    evidence_packet = _load_evidence(root, evidence)
    task = _task_profile(evidence_packet, session)
    blockers = _blocking_gates(evidence_packet)
    fallback = _fallback_payload(user_goal, request_contract, task, blockers)
    allow_llm = bool(session.llm_ready) and os.environ.get("EVOMIND_REASONING_DISABLE_LLM") != "1" and (
        not os.environ.get("PYTEST_CURRENT_TEST")
        or os.environ.get("EVOMIND_TEST_LLM") == "1"
    )
    primary_provider = os.environ.get("EVOLUTION_PRIMARY_PROVIDER", "anthropic").lower()
    fallback_provider = os.environ.get("EVOLUTION_FALLBACK_PROVIDER", "deepseek").lower()
    engine_signature = {
        "mode": "llm" if allow_llm else "deterministic",
        "session_provider": _safe_text(session.llm_provider or "", limit=120),
        "primary_provider": primary_provider,
        "primary_model": (
            os.environ.get("CLAUDE_CODE_MODEL", "claude-opus-4-8")
            if primary_provider == "anthropic"
            else os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        ),
        "fallback_provider": fallback_provider,
        "fallback_model": (
            os.environ.get("CLAUDE_CODE_MODEL", "claude-opus-4-8")
            if fallback_provider == "anthropic"
            else os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        ),
    }
    stable_context = {
        "goal": user_goal,
        "task": task,
        "contract": request_contract,
        "blockers": blockers,
        "evidence": evidence_packet,
    }
    cache_basis = {
        "goal": user_goal,
        "task": task,
        "contract": request_contract,
        "blockers": blockers,
        "engine": engine_signature,
        "semantic_evidence": _semantic_cache_projection(evidence_packet),
    }
    cache_key = hashlib.sha256(
        json.dumps(_stable_cache_value(cache_basis), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    artifact_path = xsci / "scientist_reasoning_synthesis.json"
    markdown_path = xsci / "scientist_reasoning_synthesis.md"
    history_path = xsci / "scientist_reasoning_history.jsonl"
    cache_stats_path = xsci / (
        "scientist_reasoning_cache_stats_llm.json"
        if allow_llm
        else "scientist_reasoning_cache_stats_deterministic.json"
    )

    previous = _read_json(artifact_path)
    if isinstance(previous, dict) and previous.get("cache_key") == cache_key:
        cache_stats = _update_cache_stats(
            cache_stats_path,
            hit=True,
            cache_key=cache_key,
            generated_at=generated_at,
        )
        cached_payload = {
            **previous,
            "cache_hit": True,
            "cache_stats": cache_stats,
            "cache_stats_path": str(cache_stats_path),
        }
        if persist:
            try:
                artifact_path.write_text(
                    json.dumps(cached_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except OSError:
                pass
        return cached_payload

    llm_meta: dict[str, Any] = {
        "used": False,
        "provider": "",
        "model": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "error": "",
    }
    model_payload: dict[str, Any] | None = None
    if allow_llm:
        try:
            from research_os.llm_client import LLMClient

            system = (
                "You are EvoMind's evidence-grounded AI Scientist reasoning engine. "
                "Answer the user's actual research question after tools have already run. "
                "Resource gates constrain execution, but they never excuse omitting requested analysis, "
                "falsifiable hypotheses, comparisons, or a decision. Distinguish observed facts, "
                "inferences, and unvalidated proposals. Never claim training improvement, official score, "
                "rank, medal, or top30 without the required artifacts. Return one JSON object only."
            )
            schema = {
                "reasoning_mode": "analysis_only|research_planning|gated_execution_advice",
                "direct_answer": "answer the user's central question first",
                "problem_frame": {
                    "known": ["evidence-backed facts"],
                    "unknown": ["important unknowns"],
                    "why_not_proven": ["why the requested claim is not yet proven"],
                },
                "hypotheses": [
                    {
                        "id": "H1",
                        "title": "short title",
                        "mechanism": "why it may work",
                        "falsifiable_prediction": "observable prediction",
                        "required_evidence": ["specific evidence"],
                        "experiment": "controlled test",
                        "success_threshold": "promotion threshold",
                        "disconfirming_result": "result that rejects it",
                        "evidence_strength": "low|medium|high",
                        "risk": "low|medium|high",
                        "cost": "low|medium|high",
                        "expected_value": "low|medium|high",
                    }
                ],
                "comparison": [
                    {
                        "hypothesis_id": "H1",
                        "evidence": "low|medium|high",
                        "risk": "low|medium|high",
                        "cost": "low|medium|high",
                        "expected_value": "low|medium|high",
                        "priority_score": 0,
                    }
                ],
                "selected_hypothesis_id": "H1",
                "selected_rationale": "why this comes first",
                "next_safe_action": {
                    "action": "safe action description",
                    "command": "one safe EvoMind command",
                    "gate": "gate name",
                    "expected_evidence": ["artifacts"],
                },
                "unresolved_questions": ["remaining questions"],
                "claim_boundaries": ["forbidden overclaims"],
            }
            prompt = (
                "[USER GOAL]\n"
                f"{user_goal}\n\n"
                "[REQUEST CONTRACT]\n"
                f"{json.dumps(request_contract, ensure_ascii=False, indent=2)}\n\n"
                "[TASK AND EVIDENCE]\n"
                f"{json.dumps(stable_context, ensure_ascii=False, indent=2)[:26000]}\n\n"
                "[OUTPUT CONTRACT]\n"
                f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
                "Produce at least the requested number of distinct hypotheses. Every hypothesis must "
                "include a disconfirming result. Compare evidence, risk, cost, and expected value. "
                "If execution is blocked, still complete the scientific reasoning and choose a safe "
                "planning/repair command rather than a training or submission command."
            )
            response = LLMClient(max_retries=1, timeout=120, temperature=0.2).generate(
                prompt,
                system=system,
                max_tokens=2600,
                temperature=0.2,
            )
            model_payload = _extract_json_object(response.text)
            llm_meta.update(
                {
                    "used": True,
                    "provider": response.provider,
                    "model": response.model,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cache_read_tokens": response.cache_read_tokens,
                    "error": "" if model_payload else "response_not_valid_json",
                }
            )
        except Exception as exc:
            llm_meta["error"] = type(exc).__name__

    normalized = _normalize_payload(model_payload or {}, fallback, request_contract, task, blockers)
    quality = _quality(normalized, request_contract)
    answer_markdown = _markdown(normalized)
    payload = {
        "ok": True,
        "schema": "evomind.ai_scientist.reasoning_synthesis.v1",
        "tool": "scientist_reasoning_synthesis",
        "generated_at": generated_at,
        "selected_task": session.selected_task or "",
        "user_goal": user_goal,
        "request_contract": request_contract,
        "task_profile": task,
        "blocking_gates": blockers,
        **normalized,
        "answer_markdown": answer_markdown,
        "reasoning_quality": quality,
        "llm": llm_meta,
        "cache_key": cache_key,
        "engine_signature": engine_signature,
        "cache_basis_sources": sorted((cache_basis.get("semantic_evidence") or {}).keys()),
        "cache_facts": cache_basis.get("semantic_evidence") or {},
        "cache_hit": False,
        "epistemic_status": "proposed_unvalidated",
        "artifact_path": str(artifact_path),
        "markdown_artifact_path": str(markdown_path),
        "history_path": str(history_path),
        "cache_stats_path": str(cache_stats_path),
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }
    payload["cache_stats"] = _update_cache_stats(
        cache_stats_path,
        hit=False,
        cache_key=cache_key,
        generated_at=generated_at,
    )
    if persist:
        try:
            xsci.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            markdown_path.write_text(answer_markdown + "\n", encoding="utf-8")
            history_record = {
                "generated_at": generated_at,
                "selected_task": session.selected_task or "",
                "goal_hash": hashlib.sha256(user_goal.encode("utf-8")).hexdigest()[:16],
                "cache_key": cache_key,
                "reasoning_quality": quality,
                "selected_hypothesis_id": payload.get("selected_hypothesis_id"),
                "next_safe_action": payload.get("next_safe_action"),
                "epistemic_status": "proposed_unvalidated",
                "llm": llm_meta,
                "artifact_path": str(artifact_path),
                "no_training_started": True,
                "official_submit": "blocked_until_explicit_human_approval",
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(history_record, ensure_ascii=False) + "\n")
        except OSError as exc:
            payload["ok"] = False
            payload["artifact_error"] = _safe_text(exc, limit=300)
    return payload
