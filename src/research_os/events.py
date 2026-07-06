"""Typed research-event stream shared by the engine, the CLI, and the dashboard.

The evolution loop emits ONE event per meaningful step of the research cycle
(select -> propose -> exec -> score -> promote -> repair -> lesson) plus the
lifecycle markers (run_begin / iter_begin / iter_end / run_end). Every consumer
-- the streaming ``xsci run`` renderer, ``xsci watch``, and the web dashboard --
observes this same stream, so there is a single source of truth for "what the
agent did this run".

Events are plain JSON-serializable dicts (no secrets, no large code blobs) so
they append cleanly to an ``events.jsonl`` that can be tailed live. The engine
owns the monotonic ``seq`` and the timestamp; consumers only read.

This module is stdlib-only on purpose: the engine runs on the GPU box too.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

# ── event types: one constant per research-cycle joint, imported by name so a
# typo becomes an ImportError instead of a silently-dropped event ──────────────
RUN_BEGIN = "run_begin"    # loop about to start; carries task + run meta
ITER_BEGIN = "iter_begin"  # an iteration starts; carries index + exp_id
SELECT = "select"          # MCGS chose a node/expansion (absent in linear mode)
PROPOSE = "propose"        # a candidate script was generated (hypothesis, mode)
EXEC_BEGIN = "exec_begin"  # the runner started executing the candidate
SCORE = "score"            # a RunResult came back (success, cv_score, exit_code)
PROMOTE = "promote"        # promotion gate ruled (promoted?, reason, new best?)
REPAIR = "repair"          # a failure's real error was captured for feedback
LESSON = "lesson"          # a lesson was written to retrospective memory
ITER_END = "iter_end"      # the iteration's record is complete
RUN_END = "run_end"        # loop finished; carries the summary essentials

# ── agent-mode joints: when the LLM itself drives the loop (deep AI-Scientist
# mode) it narrates and calls tools instead of the engine running a fixed ladder.
# These share the SAME stream/JSONL/dashboard, so a run started by `xsci agent`
# and one started by `xsci run` render through one code path. The scored/promote/
# lesson joints above are reused verbatim (a tool handler emits them), so the
# dashboard's search-tree / score views need no agent-specific branch. ──────────
AGENT_MSG = "agent_msg"     # the researcher's free-text reasoning for this turn
TOOL_CALL = "tool_call"     # the agent requested a tool (name + brief args)
TOOL_RESULT = "tool_result" # a tool returned (name + one-line outcome, no blobs)
COMPACTION = "compaction"   # the message history was compacted (before/after sizes)

# Ordered for docs/tests; not used for validation (unknown types pass through).
ALL_TYPES = (
    RUN_BEGIN, ITER_BEGIN, SELECT, PROPOSE, EXEC_BEGIN, SCORE,
    PROMOTE, REPAIR, LESSON, ITER_END, RUN_END,
    AGENT_MSG, TOOL_CALL, TOOL_RESULT, COMPACTION,
)

EventSink = Callable[[dict], None]


class JsonlEventSink:
    """Append events as one JSON object per line. Safe to tail with `watch`.

    Opens in append mode and flushes each line so a reader following the file
    sees events as they happen. Never raises into the engine: a sink failure
    must not crash a research run, so callers wrap emission defensively and this
    class keeps writes best-effort.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event: dict) -> None:
        line = json.dumps(event, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()


def fan_out(*sinks: EventSink) -> EventSink:
    """Compose several sinks into one (e.g. write JSONL AND render to terminal).

    Each sink is isolated: if one raises, the others still receive the event.
    """
    real = [s for s in sinks if s is not None]

    def _emit(event: dict) -> None:
        for sink in real:
            try:
                sink(event)
            except Exception:  # noqa: BLE001 - one bad sink must not blind the rest
                pass

    return _emit


def read_events(path: str | Path) -> list[dict]:
    """Read a whole events.jsonl, tolerating a half-written trailing line.

    An in-progress run may be mid-write when we read, so a final truncated line
    is skipped rather than raising. Ordering follows file order (== seq order).
    """
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # partial trailing line during a live run
    return out


def _fmt_score(v: Any) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "  -   "


def format_event(event: dict) -> str:
    """One compact human line for an event. Shared by `run` streaming and `watch`
    so the two surfaces render identically. Unknown types get a generic line."""
    t = event.get("type", "?")
    exp = event.get("exp_id", "")
    if t == RUN_BEGIN:
        brain = "MCGS" if event.get("mcgs") else "linear"
        return (f"[run] {event.get('task','?')}  metric={event.get('metric','?')}"
                f"({event.get('metric_direction','')})  budget={event.get('max_iterations','?')}  brain={brain}")
    if t == ITER_BEGIN:
        return f"[iter] {exp}  iter {event.get('iteration','?')}"
    if t == SELECT:
        refs = event.get("reference_exp_ids") or []
        ref = f" refs={','.join(refs)}" if refs else ""
        return (f"[select] from={event.get('node_exp_id','?')}  "
                f"type={event.get('expansion_type','?')}  mode={event.get('coding_mode','?')}{ref}")
    if t == PROPOSE:
        hyp = (event.get("hypothesis") or "").strip().replace("\n", " ")
        if len(hyp) > 88:
            hyp = hyp[:87] + "..."
        return f"[propose] {event.get('mode','?')}/{event.get('expansion_type','?')}: {hyp or '(no hypothesis)'}"
    if t == EXEC_BEGIN:
        return f"[exec] running {exp} on {event.get('runner','?')}"
    if t == SCORE:
        ok = "ok" if event.get("success") else "FAIL"
        code = event.get("exit_code")
        tail = f"  exit={code}" if (not event.get("success") and code is not None) else ""
        return f"[score] {ok}  cv={_fmt_score(event.get('cv_score'))}{tail}"
    if t == PROMOTE:
        if event.get("promoted"):
            d = event.get("delta")
            dtxt = f"  delta={d:+.4f}" if isinstance(d, (int, float)) else ""
            return f"[promote] OK new best {event.get('best_exp_id', exp)}  cv={_fmt_score(event.get('best_cv_score'))}{dtxt}"
        return f"[promote] held  ({event.get('reason','') or 'no improvement'})"
    if t == REPAIR:
        return f"[repair] pattern={event.get('failure_pattern','?')}: {(event.get('error') or '').strip()[:100]}"
    if t == LESSON:
        fp = event.get("failure_pattern")
        return f"[lesson] {'avoid ' + fp if fp else 'reuse: ' + (event.get('reusable_strategy') or 'baseline')}"
    if t == ITER_END:
        mark = "+" if event.get("promoted") else " "
        return f"[done] {exp} {event.get('mode','')}  cv={_fmt_score(event.get('cv_score'))} {mark}"
    if t == RUN_END:
        return (f"[run-end] best={event.get('best_exp_id','?')}  cv={_fmt_score(event.get('best_cv_score'))}  "
                f"promoted={event.get('n_promotions','?')}/{event.get('n_iterations','?')}")
    if t == AGENT_MSG:
        text = (event.get("text") or "").strip().replace("\n", " ")
        if len(text) > 200:
            text = text[:199] + "..."
        return f"[agent] {text}" if text else "[agent] (thinking)"
    if t == TOOL_CALL:
        args = event.get("args_brief") or ""
        return f"[tool-call] {event.get('tool','?')}({args})"
    if t == TOOL_RESULT:
        ok = event.get("ok")
        mark = "" if ok is None else ("OK " if ok else "ERR ")
        summary = (event.get("summary") or "").strip().replace("\n", " ")
        if len(summary) > 160:
            summary = summary[:159] + "..."
        return f"[tool-result] {mark}{event.get('tool','?')}: {summary}"
    if t == COMPACTION:
        return (f"[compact] history {event.get('messages_before','?')} -> "
                f"{event.get('messages_after','?')} msgs (prompt~{event.get('prompt_tokens','?')} tok)")
    return f"- {t} {json.dumps({k: v for k, v in event.items() if k not in ('seq','ts','type')}, ensure_ascii=False)}"


def render_stream(events: Iterable[dict]) -> str:
    """Render a whole event list to text (used by tests and non-live replay)."""
    return "\n".join(format_event(e) for e in events)
