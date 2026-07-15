"""Bounded multi-agent hypothesis generation and independent criticism.

The panel deliberately separates proposal generation from review.  Three fresh
model clients generate role-specific proposals in parallel, then three fresh
clients independently review the complete anonymized proposal set.  Only
structured summaries are persisted; raw model transcripts never enter the
Scientist state directory.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from research_os.agent.messaging import AgentMessageClient

PANEL_SCHEMA = "evomind.ai_scientist.hypothesis_panel.v1"
ROLE_SPECS: dict[str, str] = {
    "methodologist": (
        "Design falsifiable hypotheses with a valid comparison, declared metric, "
        "confound controls, and an explicit rejection condition."
    ),
    "adversarial_validator": (
        "Search for leakage, hidden assumptions, distribution shift, unsupported "
        "causal claims, and ways an apparently positive result could be false."
    ),
    "resource_strategist": (
        "Find high-information experiments with realistic data, compute, time, and "
        "dependency requirements; prefer staged probes before expensive execution."
    ),
}

_SENSITIVE_KEY = re.compile(r"(?i)(api[_-]?key|token|cookie|password|passwd|secret|private[_-]?key)")
_SENSITIVE_VALUE = re.compile(
    r"(?i)\b(api[_-]?key|token|cookie|password|passwd|secret|private[_-]?key)\s*[:=]\s*\S+"
)


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return low
    if not math.isfinite(number):
        return low
    return max(low, min(high, number))


def _safe_text(value: Any, *, limit: int = 1600) -> str:
    text = str(value or "").replace("\x00", " ").strip()
    return _SENSITIVE_VALUE.sub(r"\1=[redacted]", text)[:limit]


def _safe_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 5:
        return "[nested]"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    if isinstance(value, str):
        return _safe_text(value, limit=2400)
    if isinstance(value, list):
        return [_safe_value(item, depth=depth + 1) for item in value[:30]]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in list(value.items())[:60]:
            name = str(key)[:120]
            result[name] = "[redacted]" if _SENSITIVE_KEY.search(name) else _safe_value(item, depth=depth + 1)
        return result
    return _safe_text(value, limit=1200)


def _canonical_digest(value: Any) -> str:
    data = json.dumps(_safe_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _strict_object(text: str) -> dict[str, Any] | None:
    candidate = str(text or "").strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            candidate,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
            object_pairs_hook=reject_duplicates,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_context(root: Path) -> dict[str, Any]:
    context: dict[str, Any] = {}
    xsci = root / ".xsci"
    for name in (
        "scientist_context_packet.json",
        "scientist_situation_model.json",
        "scientist_execution_contract.json",
        "scientist_innovation_backlog.json",
    ):
        path = xsci / name
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            context[name.removesuffix(".json")] = _safe_value(value)
    return context


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    temporary.replace(path)


def _fallback_proposal(role: str, goal: str, lineage: Mapping[str, str]) -> dict[str, Any]:
    goal_text = _safe_text(goal, limit=320) or "the selected research objective"
    templates = {
        "methodologist": {
            "hypothesis": f"A single controlled intervention can improve the declared metric for {goal_text}.",
            "mechanism": "Hold the split, seed, budget, and preprocessing constant while changing one causal factor.",
            "falsification_test": "Reject when the paired held-out delta is non-positive or unstable across declared repeats.",
            "evidence_required": ["paired baseline/candidate metrics", "split and seed manifest", "confidence interval"],
            "risks": ["confounding", "multiple-comparison bias"],
            "confidence": 0.58,
        },
        "adversarial_validator": {
            "hypothesis": f"The apparent gain for {goal_text} may be caused by leakage or distribution mismatch.",
            "mechanism": "Audit feature provenance, duplicate groups, time order, and train/test separability before promotion.",
            "falsification_test": "Reject the concern only when leakage probes are clean and the gain survives a leakage-safe split.",
            "evidence_required": ["leakage audit", "adversarial validation", "group/time-safe evaluation"],
            "risks": ["false reassurance", "unobserved shift"],
            "confidence": 0.62,
        },
        "resource_strategist": {
            "hypothesis": f"A low-cost staged probe can eliminate weak branches before full execution of {goal_text}.",
            "mechanism": "Run a small deterministic smoke, then a representative subset, and scale only after both gates pass.",
            "falsification_test": "Reject when the proxy ranking disagrees with repeated full-budget results.",
            "evidence_required": ["cost ledger", "proxy/full correlation", "resource readiness proof"],
            "risks": ["proxy mismatch", "underestimated setup cost"],
            "confidence": 0.55,
        },
    }
    body = dict(templates[role])
    body.update({
        "role": role,
        "resource_cost": {"level": "low", "needs_gpu": False, "estimated_rounds": 1},
        "generated_by": "deterministic_fallback",
        "provider": "",
        "model": "",
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "lineage": dict(lineage),
    })
    body["proposal_id"] = f"HP-{role[:3].upper()}-{_canonical_digest(body)[:10]}"
    return body


def _normalize_proposal(
    item: Any,
    *,
    role: str,
    index: int,
    provider: str,
    model: str,
    usage: Mapping[str, Any],
    lineage: Mapping[str, str],
) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    hypothesis = _safe_text(item.get("hypothesis"), limit=900)
    mechanism = _safe_text(item.get("mechanism"), limit=900)
    falsification = _safe_text(item.get("falsification_test"), limit=900)
    evidence = [_safe_text(value, limit=320) for value in item.get("evidence_required", []) if str(value).strip()][:8]
    risks = [_safe_text(value, limit=320) for value in item.get("risks", []) if str(value).strip()][:8]
    if not hypothesis or not mechanism or not falsification or not evidence:
        return None
    proposal = {
        "role": role,
        "hypothesis": hypothesis,
        "mechanism": mechanism,
        "falsification_test": falsification,
        "evidence_required": evidence,
        "risks": risks,
        "resource_cost": _safe_value(item.get("resource_cost") or {"level": "unknown"}),
        "confidence": _clamp(item.get("confidence", 0.5)),
        "generated_by": "model",
        "provider": _safe_text(provider, limit=100),
        "model": _safe_text(model, limit=160),
        "usage": {
            "input_tokens": max(0, int(usage.get("input_tokens", 0) or 0)),
            "output_tokens": max(0, int(usage.get("output_tokens", 0) or 0)),
        },
        "lineage": dict(lineage),
    }
    proposal["proposal_id"] = f"HP-{role[:3].upper()}-{_canonical_digest([proposal, index])[:10]}"
    return proposal


def _generation_system(role: str) -> str:
    return (
        "You are one isolated member of EvoMind's research hypothesis panel. "
        f"Your role is {role}: {ROLE_SPECS[role]} Return one JSON object only with key proposals. "
        "proposals must contain 1-3 objects with hypothesis, mechanism, falsification_test, "
        "evidence_required (array), risks (array), resource_cost (object), and confidence (0..1). "
        "Do not claim an experiment ran. Do not include credentials or prose outside JSON."
    )


def _generate_role(
    role: str,
    *,
    goal: str,
    context: Mapping[str, Any],
    lineage: Mapping[str, str],
    client_factory: Callable[[str, str], Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fallback = _fallback_proposal(role, goal, lineage)
    try:
        client = client_factory(role, "generation")
        if client is None or not bool(client.is_available()):
            return [fallback], {"role": role, "status": "fallback_provider_unavailable"}
        turn = client.send(
            [{"role": "user", "content": json.dumps({"goal": goal, "evidence_context": _safe_value(context)}, ensure_ascii=False)[:18000]}],
            system=_generation_system(role),
            tools=[],
            max_tokens=1800,
            temperature=0.35,
        )
        value = _strict_object(turn.text)
        rows = value.get("proposals") if isinstance(value, dict) else None
        usage = {"input_tokens": turn.input_tokens, "output_tokens": turn.output_tokens}
        proposals = [
            normalized
            for index, item in enumerate(rows if isinstance(rows, list) else [], start=1)
            if (normalized := _normalize_proposal(
                item,
                role=role,
                index=index,
                provider=turn.provider,
                model=turn.model,
                usage=usage,
                lineage=lineage,
            )) is not None
        ][:3]
        if not proposals:
            return [fallback], {"role": role, "status": "fallback_invalid_model_output"}
        return proposals, {
            "role": role,
            "status": "model",
            "provider": _safe_text(turn.provider, limit=100),
            "model": _safe_text(turn.model, limit=160),
            "usage": usage,
            "proposal_count": len(proposals),
        }
    except Exception as exc:  # one panel member must not collapse the whole panel
        return [fallback], {"role": role, "status": f"fallback_error:{type(exc).__name__}"}


def _fallback_reviews(role: str, proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []
    for item in proposals:
        complete = bool(item.get("falsification_test") and item.get("evidence_required") and item.get("mechanism"))
        risks = len(item.get("risks") or [])
        base = 0.72 if complete else 0.35
        if role == "adversarial_validator" and risks == 0:
            base -= 0.20
        reviews.append({
            "proposal_id": item["proposal_id"],
            "critic_role": role,
            "confidence_adjustment": -0.08 if not complete else 0.0,
            "methodological_score": base,
            "evidence_score": min(0.9, 0.45 + 0.08 * len(item.get("evidence_required") or [])),
            "feasibility_score": 0.68 if item.get("resource_cost") else 0.45,
            "critical_veto": not complete,
            "veto_reason": "missing falsification or evidence contract" if not complete else "",
            "critique": "Deterministic completeness review; provider critique unavailable.",
            "generated_by": "deterministic_fallback",
            "provider": "",
            "model": "",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        })
    return reviews


def _normalize_review(item: Any, *, role: str, proposal_ids: set[str], turn: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict) or item.get("proposal_id") not in proposal_ids:
        return None
    veto = item.get("critical_veto") is True
    return {
        "proposal_id": item["proposal_id"],
        "critic_role": role,
        "confidence_adjustment": _clamp(item.get("confidence_adjustment", 0.0), -0.3, 0.3),
        "methodological_score": _clamp(item.get("methodological_score", 0.0)),
        "evidence_score": _clamp(item.get("evidence_score", 0.0)),
        "feasibility_score": _clamp(item.get("feasibility_score", 0.0)),
        "critical_veto": veto,
        "veto_reason": _safe_text(item.get("veto_reason"), limit=500) if veto else "",
        "critique": _safe_text(item.get("critique"), limit=700),
        "generated_by": "model",
        "provider": _safe_text(turn.provider, limit=100),
        "model": _safe_text(turn.model, limit=160),
        "usage": {"input_tokens": max(0, int(turn.input_tokens or 0)), "output_tokens": max(0, int(turn.output_tokens or 0))},
    }


def _review_role(
    role: str,
    *,
    goal: str,
    proposals: list[dict[str, Any]],
    client_factory: Callable[[str, str], Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fallback = _fallback_reviews(role, proposals)
    anonymized = [
        {
            key: item.get(key)
            for key in (
                "proposal_id", "hypothesis", "mechanism", "falsification_test",
                "evidence_required", "risks", "resource_cost",
            )
        }
        for item in proposals
    ]
    try:
        client = client_factory(role, "review")
        if client is None or not bool(client.is_available()):
            return fallback, {"role": role, "status": "fallback_provider_unavailable"}
        system = (
            "You are an independent critic in EvoMind's hypothesis panel. "
            f"Your role is {role}: {ROLE_SPECS[role]} Review every proposal independently. "
            "Return one JSON object only with key reviews. Each review must contain proposal_id, "
            "confidence_adjustment (-0.3..0.3), methodological_score, evidence_score, "
            "feasibility_score (all 0..1), critical_veto (boolean), veto_reason, and critique. "
            "A critical veto is reserved for leakage, invalid evaluation, impossible dependencies, "
            "or a non-falsifiable claim. Do not reveal credentials or claim execution."
        )
        turn = client.send(
            [{"role": "user", "content": json.dumps({"goal": goal, "proposals": anonymized}, ensure_ascii=False)[:24000]}],
            system=system,
            tools=[],
            max_tokens=2600,
            temperature=0.15,
        )
        value = _strict_object(turn.text)
        rows = value.get("reviews") if isinstance(value, dict) else None
        proposal_ids = {item["proposal_id"] for item in proposals}
        reviews = [
            normalized
            for item in rows if isinstance(rows, list)
            if (normalized := _normalize_review(item, role=role, proposal_ids=proposal_ids, turn=turn)) is not None
        ]
        by_id = {item["proposal_id"]: item for item in reviews}
        if set(by_id) != proposal_ids:
            return fallback, {"role": role, "status": "fallback_incomplete_model_review"}
        return [by_id[item["proposal_id"]] for item in proposals], {
            "role": role,
            "status": "model",
            "provider": _safe_text(turn.provider, limit=100),
            "model": _safe_text(turn.model, limit=160),
            "usage": {"input_tokens": turn.input_tokens, "output_tokens": turn.output_tokens},
            "review_count": len(reviews),
        }
    except Exception as exc:
        return fallback, {"role": role, "status": f"fallback_error:{type(exc).__name__}"}


def _aggregate(proposals: list[dict[str, Any]], reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for proposal in proposals:
        relevant = [item for item in reviews if item.get("proposal_id") == proposal["proposal_id"]]
        adjustments = [float(item["confidence_adjustment"]) for item in relevant]
        review_scores = [
            statistics.median([
                float(item["methodological_score"]),
                float(item["evidence_score"]),
                float(item["feasibility_score"]),
            ])
            for item in relevant
        ]
        median_adjustment = statistics.median(adjustments) if adjustments else -0.2
        median_review = statistics.median(review_scores) if review_scores else 0.0
        disagreement = (max(review_scores) - min(review_scores)) if len(review_scores) > 1 else 0.0
        vetoes = sum(item.get("critical_veto") is True for item in relevant)
        adjusted_confidence = _clamp(float(proposal["confidence"]) + median_adjustment)
        score = _clamp(
            0.35 * float(proposal["confidence"])
            + 0.40 * median_review
            + 0.25 * adjusted_confidence
            - 0.20 * disagreement
            - 0.35 * min(vetoes, 2)
        )
        status = "rejected_critical_veto" if vetoes >= 2 else "hold_for_repair" if vetoes == 1 else "ranked"
        ranked.append({
            **proposal,
            "panel_score": round(score, 6),
            "adjusted_confidence": round(adjusted_confidence, 6),
            "median_review_score": round(median_review, 6),
            "review_disagreement": round(disagreement, 6),
            "critical_veto_count": vetoes,
            "status": status,
            "review_count": len(relevant),
            "veto_reasons": [item["veto_reason"] for item in relevant if item.get("critical_veto")][:6],
        })
    ranked.sort(key=lambda item: (
        item["status"] != "ranked",
        -float(item["panel_score"]),
        item["proposal_id"],
    ))
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index
    return ranked


def run_scientist_hypothesis_panel(
    session: Any,
    root: Path | str,
    *,
    goal: str = "",
    evidence_context: Mapping[str, Any] | None = None,
    client_factory: Callable[[str, str], Any] | None = None,
    max_workers: int = 3,
    persist: bool = True,
) -> dict[str, Any]:
    """Run two bounded parallel rounds and persist a structured panel verdict."""

    root_path = Path(root).resolve()
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    objective = _safe_text(goal or getattr(session, "last_goal", "") or getattr(session, "selected_task", ""), limit=1600)
    context = _safe_value(dict(evidence_context) if evidence_context is not None else _read_context(root_path))
    lineage = {
        "goal_sha256": _canonical_digest(objective),
        "context_sha256": _canonical_digest(context),
    }
    artifact_path = root_path / ".xsci" / "scientist_hypothesis_panel.json"
    history_path = root_path / ".xsci" / "scientist_hypothesis_panel_history.jsonl"
    factory = client_factory or (lambda _role, _phase: AgentMessageClient(max_retries=1, timeout=120))
    workers = max(1, min(3, int(max_workers)))
    started = time.monotonic()

    proposals: list[dict[str, Any]] = []
    generation_runs: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="evomind-hypothesis") as executor:
        futures = {
            executor.submit(
                _generate_role,
                role,
                goal=objective,
                context=context,
                lineage=lineage,
                client_factory=factory,
            ): role
            for role in ROLE_SPECS
        }
        for future in as_completed(futures):
            rows, run = future.result()
            proposals.extend(rows)
            generation_runs.append(run)
    proposals.sort(key=lambda item: (item["role"], item["proposal_id"]))
    generation_runs.sort(key=lambda item: item["role"])

    reviews: list[dict[str, Any]] = []
    review_runs: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="evomind-critic") as executor:
        futures = {
            executor.submit(
                _review_role,
                role,
                goal=objective,
                proposals=proposals,
                client_factory=factory,
            ): role
            for role in ROLE_SPECS
        }
        for future in as_completed(futures):
            rows, run = future.result()
            reviews.extend(rows)
            review_runs.append(run)
    reviews.sort(key=lambda item: (item["proposal_id"], item["critic_role"]))
    review_runs.sort(key=lambda item: item["role"])
    ranked = _aggregate(proposals, reviews)
    selectable = [item for item in ranked if item["status"] == "ranked"]
    model_calls = sum(item.get("status") == "model" for item in [*generation_runs, *review_runs])
    fallback_calls = 6 - model_calls
    payload: dict[str, Any] = {
        "ok": bool(ranked),
        "schema": PANEL_SCHEMA,
        "tool": "scientist_hypothesis_panel",
        "generated_at": generated_at,
        "selected_task": _safe_text(getattr(session, "selected_task", ""), limit=240),
        "goal": objective,
        "lineage": lineage,
        "roles": list(ROLE_SPECS),
        "parallel_generation": True,
        "independent_review_round": True,
        "generation_runs": generation_runs,
        "review_runs": review_runs,
        "model_call_count": model_calls,
        "fallback_call_count": fallback_calls,
        "mode": "model_parallel" if model_calls == 6 else "deterministic_fallback" if model_calls == 0 else "hybrid",
        "proposals": proposals,
        "reviews": reviews,
        "ranked_hypotheses": ranked,
        "selected_hypothesis": selectable[0] if selectable else None,
        "selection_status": "selected" if selectable else "blocked_by_critical_review",
        "aggregation": {
            "method": "median_adjustment_disagreement_penalty_critical_veto_v1",
            "minimum_independent_critics": 3,
            "critical_veto_rejection_count": 2,
        },
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "artifact_path": str(artifact_path),
        "history_path": str(history_path),
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
        "human_gate": "review_panel_selection_before_experiment_execution",
    }
    if persist:
        _atomic_json(artifact_path, payload)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    return payload


__all__ = ["PANEL_SCHEMA", "ROLE_SPECS", "run_scientist_hypothesis_panel"]
