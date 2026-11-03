"""Debug agent + pre-execution code review helpers (DeepEvo W5 / P0-AG-01/02).

Kept at package root so ``evolution_loop`` can import without cycling through
``research_os.agent`` (which imports tools → evolution_loop).
"""
from __future__ import annotations

import ast
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

# ── P0-AG-01 / P0-AG-02 helpers (flag-gated at call sites) ────────────────────

_OVERCLAIM_IN_CODE_COMMENTS = re.compile(
    r"\b(beat|beats|beating|surpass(?:es|ed)?|outperform(?:s|ed)?)\s+"
    r"(ai2|mlevolve|mle-?bench)\b",
    re.IGNORECASE,
)

_HARD_FAIL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("shell_injection", re.compile(r"\bos\.system\s*\(|\bsubprocess\.call\s*\(\s*['\"]", re.I)),
    ("credential_literal", re.compile(r"(api[_-]?key|secret[_-]?key|password)\s*=\s*['\"][^'\"]{8,}", re.I)),
)


def build_debug_feedback(
    error: str,
    *,
    exp_id: str = "",
    failure_pattern: str = "",
    stderr_trace: str = "",
) -> dict[str, Any]:
    """Structured debug-agent payload for the next Diff prompt (P0-AG-01).

    Call only when ``EVOMIND_FF_DEBUG_AGENT`` is on. Flag-off callers keep the
    prior short ``LAST RUN FAILED`` notes path in ``evolution_loop``.
    """
    trace = (stderr_trace or error or "").strip()
    cleaned = trace[-4000:] if len(trace) > 4000 else trace
    return {
        "schema": "academic_research_os.debug_feedback.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "feature_flag": "EVOMIND_FF_DEBUG_AGENT",
        "exp_id": exp_id,
        "failure_pattern": failure_pattern or "",
        "stderr_trace": cleaned,
        "prompt_hint": (
            "DEBUG_AGENT: diagnose the stderr/traceback below, then emit a Diff "
            "fix that addresses the root cause. Stay within the remaining budget."
        ),
    }


def format_debug_prompt_section(feedback: dict[str, Any]) -> str:
    """Render debug feedback for injection into the next Diff/user notes."""
    trace = str(feedback.get("stderr_trace") or "")
    pattern = str(feedback.get("failure_pattern") or "")
    exp_id = str(feedback.get("exp_id") or "")
    header = feedback.get("prompt_hint") or "DEBUG_AGENT TRACE"
    parts = [
        str(header),
        f"failed_exp={exp_id} failure_pattern={pattern or 'unknown'}",
        "stderr/traceback:",
        trace,
    ]
    return "\n".join(parts)


def write_debug_feedback(work_dir: str | Path, feedback: dict[str, Any]) -> Path:
    """Persist debug_feedback.json under the failed exp (or work_dir root)."""
    root = Path(work_dir)
    exp_id = str(feedback.get("exp_id") or "").strip()
    target_dir = (root / exp_id) if exp_id else root
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "debug_feedback.json"
    path.write_text(json.dumps(feedback, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def review_code_before_enqueue(
    code: str,
    *,
    exp_id: str = "",
    task_id: str = "",
) -> dict[str, Any]:
    """CPU-only pre-execution code review + quality gate (P0-AG-02).

    Hard fails (syntax errors, dangerous patterns) set ``quality_gate=block`` so
    callers can refuse enqueue. Soft issues warn but allow. Does not train or
    touch HPC.
    """
    issues: list[str] = []
    hard_fail = False
    syntax_status = "passed"
    stripped = (code or "").strip()
    if not stripped:
        hard_fail = True
        syntax_status = "failed"
        issues.append("empty_code")
    else:
        try:
            ast.parse(stripped)
        except SyntaxError as exc:
            hard_fail = True
            syntax_status = "failed"
            issues.append(f"syntax_error:{exc.msg}@line{exc.lineno}")
        for label, pattern in _HARD_FAIL_PATTERNS:
            if pattern.search(stripped):
                hard_fail = True
                issues.append(f"hard_fail:{label}")
        # Comments must not assert unverified benchmark wins (claim-gate alignment).
        if _OVERCLAIM_IN_CODE_COMMENTS.search(stripped):
            issues.append("overclaim_language_in_code")
            # Soft warn only — claim gate handles external claims; do not hard-block enqueue.
    overall = "failed" if hard_fail else "passed"
    gate = "block" if hard_fail else "allow"
    return {
        "schema": "academic_research_os.code_review.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "feature_flag": "EVOMIND_FF_CODE_REVIEW_AGENT",
        "exp_id": exp_id,
        "task_id": task_id,
        "overall_status": overall,
        "quality_gate": gate,
        "syntax_check": syntax_status,
        "issues": issues,
        "human_gate_required": True,
        "enqueue_allowed": not hard_fail,
    }


def write_code_review_artifacts(
    work_dir: str | Path,
    review: dict[str, Any],
) -> dict[str, str]:
    """Write code_review.json + code_quality_gate.json beside the candidate."""
    root = Path(work_dir)
    exp_id = str(review.get("exp_id") or "").strip()
    target_dir = (root / exp_id) if exp_id else root
    target_dir.mkdir(parents=True, exist_ok=True)
    review_path = target_dir / "code_review.json"
    gate_path = target_dir / "code_quality_gate.json"
    review_path.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    gate = {
        "schema": "academic_research_os.code_quality_gate.v1",
        "created_at": review.get("created_at"),
        "feature_flag": "EVOMIND_FF_CODE_REVIEW_AGENT",
        "overall_status": review.get("overall_status"),
        "quality_gate": review.get("quality_gate"),
        "enqueue_allowed": review.get("enqueue_allowed"),
        "issues": list(review.get("issues") or []),
        "review_artifact": str(review_path.name),
    }
    gate_path.write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"code_review": str(review_path), "code_quality_gate": str(gate_path)}

