"""Context compaction for long research runs.

A research run can span hundreds of tool calls; the message history will outgrow
any context window. This module compacts it WITHOUT losing the thread.

Key design choice for a *research* agent: the durable state is not a fuzzy LLM
summary — it is the AUDITABLE SEARCH GRAPH. So the "summary" we splice in is built
DETERMINISTICALLY from ``toolbox.graph`` + memory (current best, promotions, failed
hypotheses, pending plan) — no extra LLM call, and it can never hallucinate a
result the graph does not contain. This is faithful to the plan's thesis that the
search graph is the source of truth.

Compaction rules (mirroring the sound part of Hermes' compactor):
  * HEAD is protected: the seed message (task + goal + experience index) always
    stays, and the deterministic research-state block is appended to it.
  * TAIL is protected: the most recent turns stay verbatim, cut at a safe boundary
    so no assistant ``tool_use`` is left without its ``tool_result``.
  * The MIDDLE is dropped (its outcome already lives in the search-graph state).
  * A TIME ANCHOR states which experiments are already DONE, so the agent does not
    re-run completed work after compaction.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough token estimate (~chars/4) for triggering when the API didn't report
    a real prompt-token count (e.g. offline tests). Real runs use the API's count."""
    chars = 0
    for msg in messages:
        content = msg.get("content", "")
        chars += len(content if isinstance(content, str) else str(content))
    return chars // 4


def should_compact(*, prompt_tokens: int, messages: list[dict[str, Any]],
                   threshold_tokens: int, last_compact_tokens: int,
                   min_messages: int = 8) -> bool:
    """Compact when the prompt is large AND there is enough to gain from it.

    Uses the API-reported ``prompt_tokens`` when available, else the char estimate.
    Anti-thrash: don't recompact until the prompt has grown meaningfully past the
    size it was at the last compaction (so we don't compact every single turn)."""
    if len(messages) < min_messages:
        return False
    size = prompt_tokens if prompt_tokens > 0 else estimate_tokens(messages)
    if size < threshold_tokens:
        return False
    # grown at least 10% past the last post-compaction size (anti-thrash)
    return size >= last_compact_tokens * 1.1 if last_compact_tokens else True


def build_research_state_block(toolbox: Any) -> str:
    """Deterministic 'what we know so far' from the search graph + memory.

    Duck-typed on ResearchToolbox (graph, best_exp_id, library, _pending_plan) so
    it stays testable with a light fake. Never invents scores — reads the graph."""
    graph = toolbox.graph
    lines = ["[RESEARCH STATE SO FAR — reconstructed from the auditable search graph]"]
    # current best (the promotion gate's verdict, not a guess)
    best_id = getattr(toolbox, "best_exp_id", None)
    best = graph.nodes.get(best_id) if best_id else None
    if best is not None:
        lines.append(f"BEST: {best_id} cv={best.cv_score} "
                     f"({graph.metric_name} {graph.metric_direction})  hypothesis: {best.hypothesis}")
    else:
        lines.append("BEST: (none promoted yet)")
    # every experiment node: id, mode, score, success, decision (the audit trail)
    if graph.nodes:
        lines.append("EXPERIMENTS DONE (do NOT re-run these):")
        for node in graph.nodes.values():
            mark = "+" if node.promoted else " "
            lines.append(f"  [{mark}] {node.exp_id} {node.branch_type} cv={node.cv_score} "
                         f"success={node.run_success} decision={node.decision}")
    # failed hypotheses worth remembering (so the agent doesn't repeat them)
    failed = [n for n in graph.nodes.values() if not n.run_success]
    if failed:
        lines.append("FAILED (already tried — diagnose, don't repeat blindly):")
        for n in failed[-6:]:
            hyp = (n.hypothesis or "").strip().replace("\n", " ")[:100]
            lines.append(f"  - {n.exp_id}: {hyp}")
    # a pending plan, if one is awaiting a run
    plan = getattr(toolbox, "_pending_plan", None)
    if plan is not None:
        lines.append(f"PENDING PLAN: expand {plan.node_exp_id} via {plan.expansion_type} "
                     f"({plan.coding_mode}) — run_experiment is expected next.")
    # experience index (cross-run lessons)
    try:
        lines.append("EXPERIENCE INDEX: " + toolbox.library.index_digest(toolbox.context.task_type))
    except Exception:  # memory must never block compaction
        pass
    # time anchor + invariant reminder
    lines.append(f"(state as of {datetime.now().isoformat(timespec='seconds')}; "
                 "the runner decides success, the gate decides promotion — you cannot fake either.)")
    return "\n".join(lines)


def _first_text(content: Any) -> str:
    """Extract plain text from a message content (str or block list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return str(content)


def compact_messages(messages: list[dict[str, Any]], *, state_block: str,
                     tail_turns: int = 6) -> list[dict[str, Any]]:
    """Return a compacted copy: protected head (+state block) then a safe tail.

    Boundary safety: the tail is advanced to start on an assistant turn so a
    ``tool_result`` is never orphaned from its ``tool_use``. If the history is too
    short to gain anything, it is returned unchanged.
    """
    if len(messages) <= tail_turns + 1:
        return list(messages)
    head = messages[0]
    head_text = _first_text(head.get("content", ""))
    rebuilt_head = {"role": "user", "content": f"{head_text}\n\n{state_block}"}

    # take the last `tail_turns` messages, then advance the start to an assistant
    # turn so we don't begin the tail on an orphaned tool_result.
    tail = messages[-tail_turns:]
    start = 0
    while start < len(tail) and tail[start].get("role") != "assistant":
        start += 1
    tail = tail[start:] if start < len(tail) else []
    return [rebuilt_head, *tail]
