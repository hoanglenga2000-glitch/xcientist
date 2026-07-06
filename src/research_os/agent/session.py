"""AgentSession — the conversation loop that lets the model drive the research.

This is the deep-agent counterpart to ``EvolutionLoop``. Instead of a fixed
Python ladder that calls the LLM as a code generator, the model itself is the
researcher: it reasons, calls tools (``ResearchToolbox``), reads their results,
and decides the next move — turn after turn — until it calls ``finish`` or hits
the turn budget.

The loop's shape is the canonical tool-use loop:

    send(history, tools) -> assistant turn
        if the turn has tool_use blocks:
            run each tool, append a user turn of tool_result(s), loop
        else:
            (the model is talking, not acting) — end the turn

Every joint emits an event to the SAME stream the fixed loop uses, persisted to
``<exp_dir>/events.jsonl``, and the run's ``search_graph.json`` / ``summary.json``
are written on exit — so a run started by ``xsci agent`` shows up in the existing
:8088 dashboard with no frontend changes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from .. import events as ev
from ..variation_generator import TaskContext
from .context import build_research_state_block, compact_messages, should_compact
from .guardrails import ToolGuardrailController
from .ledger import MessageLedger
from .messaging import AgentMessageClient, ToolResult
from .report import write_report
from .tools import ResearchToolbox

# The researcher's charter. It states the loop AND the guardrails plainly so the
# model spends tool calls productively and never fights the gates it cannot win.
_SYSTEM = """You are XCIENTIST, an autonomous machine-learning research agent working in a
terminal. You are not a chatbot: you make measurable progress on a Kaggle-style
task by running real experiments through your tools.

CONTROL MODEL (分层共治 — layered co-governance):
  * The MCGS SELECTION BRAIN owns TOPOLOGY: which node to expand, the expansion
    type (primary/intra_branch/cross_branch/aggregation), and the coding mode. You
    consult it via plan_next_experiment and work on the node it picks.
  * YOU own the SCIENCE: the hypothesis, the actual code, and the interpretation —
    grounded in the data, the CV history, the memory lessons, and the plan.
  * The GATES own the verdicts (success, promotion, conclusion, submission) — you
    request them, you cannot fake them.

YOUR RESEARCH LOOP each round:
  1. Ground yourself: inspect_data, read_memory, read_search_tree, recommend_strategies.
  2. plan_next_experiment — the MCGS brain returns the node to expand, the expansion
     type, the coding mode, and (for cross_branch/aggregation) reference solutions.
  3. Form ONE clear hypothesis for THAT expansion, then run_experiment with a
     COMPLETE runnable script (honor the contract: read --data-dir, print exactly
     `CV_SCORE=<float>`, write submission.csv + metrics.json to --out-dir). Keep runs
     inside the compute budget. (The parent/branch/mode come from the plan — you
     supply only hypothesis + code.)
  4. Read the result. If it FAILED, the tool returns the REAL error (noise stripped);
     diagnose the root cause. plan_next_experiment will likely give you Diff mode to
     fix it — do not repeat the same mistake.
  5. If it succeeded, call evaluate_promotion to let the gate rule (this also feeds
     the outcome back to the MCGS brain, unlocking cross-branch/fusion later).
  6. record_lesson so the next round (and future tasks) learn from this.
  7. Repeat. plan_next_experiment again each round — the brain may DIVERSIFY into a
     new branch or FUSE branches when the search stalls. Trust its plan.
  8. Before you finish with a headline result, request_audit — an INDEPENDENT read-only
     sub-agent re-checks your evidence in its own context and reports back. Treat a
     SUPPORTED verdict as your green light and an OVERCLAIMED/INSUFFICIENT verdict as a
     signal to keep working, not to argue.

HARD RULES (enforced by the tools; do not fight them):
  * You MUST plan_next_experiment before every run_experiment. Topology is the
    brain's job; run_experiment refuses to run without a plan.
  * The RUNNER decides success. A crashed/timed-out run is a failure even if a
    score reached disk. Never claim a failed run worked.
  * The PROMOTION GATE is deterministic: failed runs are never promoted, and a
    success is promoted only if it beats the best-so-far with the required
    artifacts. Do not argue with a HELD ruling — improve and try again.
  * Before stating a finding, audit_conclusion; report only the allowed conclusion.
  * Official Kaggle submission is ALWAYS blocked behind a human gate. Never expect
    to auto-submit.

Be concise in your reasoning. Prefer acting (calling tools) over long explanations.
When you are done or genuinely blocked, call finish with a short honest summary."""


def _has_unanswered_tool_use(msg: dict[str, Any]) -> bool:
    """True if ``msg`` is an assistant turn containing at least one tool_use block.

    On the wire, every assistant tool_use MUST be answered by a tool_result in the
    next user turn. A ledger whose LAST message is such an assistant turn (the run
    died before the results were appended) is invalid to resend — so a resume must
    drop it."""
    if msg.get("role") != "assistant":
        return False
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content)


def _drop_dangling_tool_use(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return ``messages`` with a trailing, unanswered assistant tool_use turn
    removed (crash mid-turn). Also drops a now-trailing empty user turn if one is
    left behind, so the resumed history always ends ready for a new user turn."""
    msgs = list(messages)
    while msgs and _has_unanswered_tool_use(msgs[-1]):
        msgs.pop()
    return msgs


@dataclass
class AgentSessionConfig:
    max_turns: int = 40           # hard cap on assistant turns (safety budget)
    max_tokens: int = 8192
    temperature: float = 0.3
    tool_result_cap: int = 6000   # trim tool output fed back to the model
    compact_threshold_tokens: int = 120_000  # compact the history past this prompt size
    compact_tail_turns: int = 6              # recent turns kept verbatim on compaction


@dataclass
class AgentSession:
    context: TaskContext
    toolbox: ResearchToolbox
    exp_dir: Path
    client: Optional[AgentMessageClient] = None
    config: AgentSessionConfig = field(default_factory=AgentSessionConfig)
    on_event: Optional[Callable[[dict], None]] = None
    run_meta: dict[str, Any] = field(default_factory=dict)
    guardrails: Optional[ToolGuardrailController] = None
    resume: bool = False          # continue a prior run's conversation from the ledger

    def __post_init__(self) -> None:
        self.exp_dir = Path(self.exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.client = self.client or AgentMessageClient()
        # Persist the stream to events.jsonl AND fan out to the live renderer.
        self._sink = ev.fan_out(ev.JsonlEventSink(self.exp_dir / "events.jsonl"), self.on_event)
        self._seq = 0
        # Wire the toolbox's emitter to our stream so score/promote/lesson events
        # flow through the SAME jsonl the dashboard reads.
        self.toolbox.emit = self._emit
        # Give the TOP-LEVEL agent (unrestricted toolbox) a callback to spawn the
        # read-only audit sub-agent on demand via the request_audit tool. Restricted
        # toolboxes (the auditor itself) keep audit_spawner=None, so audits can't nest.
        if self.toolbox.allowed_tools is None and self.toolbox.audit_spawner is None:
            self.toolbox.audit_spawner = self._spawn_audit
        # Loop-breaker between the model and the toolbox (repeated-failure block,
        # truncated-code refusal, consecutive-failure halt). Never decides research
        # outcomes — the tools' deterministic gates own those.
        self.guardrails = self.guardrails or ToolGuardrailController()
        self.messages: list[dict[str, Any]] = []
        self._last_prompt_tokens = 0     # API-reported prompt size of the last send
        self._last_compact_tokens = 0    # size at the last compaction (anti-thrash)
        self._agent_summary = ""         # the agent's own finish summary, if any
        # Crash-survivable conversation log (resume + audit). Search-graph/summary
        # already hold research state; this preserves the raw dialogue.
        self._ledger = MessageLedger(self.exp_dir / "messages.jsonl")
        self._resumed_turns = 0          # how many prior messages a resume reloaded
        if self.resume:
            self._load_prior_conversation()

    def _load_prior_conversation(self) -> None:
        """Reload the prior conversation from the ledger so a killed/interrupted run
        can CONTINUE instead of restarting from zero.

        A run can die mid-turn (host exit, GPU blip, 1800s timeout). The last
        persisted message may then be an assistant turn whose tool_use blocks were
        never answered — an invalid state to send back to the API (every tool_use
        MUST be followed by its tool_result). We drop that dangling turn: the
        toolbox's restored search graph already captures whatever that run did, so
        nothing PROVEN is lost — only an unanswered request the model can re-issue.
        """
        prior = self._ledger.load()
        prior = _drop_dangling_tool_use(prior)
        self.messages = prior
        self._resumed_turns = len(prior)
        # rewrite the ledger to the sanitized history so it stays a faithful,
        # replayable record (and a second resume sees the same clean state).
        if prior:
            self._ledger.rewrite(prior)

    def _emit(self, event_type: str, **fields: Any) -> None:
        """Attach a monotonic seq + timestamp and push one event to the stream.
        Never raises — observability must not break a research run."""
        self._seq += 1
        event = {"seq": self._seq, "ts": datetime.now().isoformat(timespec="seconds"),
                 "type": event_type, **fields}
        try:
            self._sink(event)
        except Exception:  # noqa: BLE001
            pass

    def _spawn_audit(self, focus: str) -> str:
        """Run the read-only audit sub-agent over this session's graph and return its
        compact brief. Lazy import avoids the tools<->subagents import cycle. The
        child's events are tagged and forwarded to our renderer so the terminal shows
        the audit happening, but its transcript never enters the parent context."""
        from .subagents import spawn_audit_agent  # local: breaks the import cycle

        goal = (
            "You are an INDEPENDENT auditor. Do NOT trust the parent agent's claims. "
            "Inspect the search tree and evidence, then judge whether the reported best "
            "result is genuinely supported (real CV gain, run succeeded, artifacts present) "
            "or whether it is thin/leaky/over-claimed. "
            + (f"Focus: {focus}. " if focus else "")
            + "Use audit_conclusion on the best experiment's claim, then finish with a short, "
            "honest verdict (SUPPORTED / OVERCLAIMED / INSUFFICIENT + one-line reason)."
        )

        def _child_event(event: dict) -> None:
            try:
                event = {**event, "sub_agent": "audit"}
                if self.on_event:
                    self.on_event(event)
            except Exception:  # noqa: BLE001 - child rendering must not break the parent
                pass

        result = spawn_audit_agent(
            self.toolbox, goal=goal, client=self.client,
            exp_dir=self.exp_dir / "_audit", on_event=_child_event,
        )
        self._emit("audit_report", status=result.status, turns=result.turns_used,
                   summary=result.summary[:400])
        return result.to_brief()

    def _maybe_compact(self) -> None:
        """Compact the message history in place when it grows past the threshold.

        Uses the API-reported prompt size from the previous send (falls back to a
        char estimate offline). The compacted head carries a deterministic snapshot
        of the search graph, so nothing the run has PROVEN is lost — only the raw
        turn-by-turn transcript that produced it."""
        if not should_compact(
            prompt_tokens=self._last_prompt_tokens, messages=self.messages,
            threshold_tokens=self.config.compact_threshold_tokens,
            last_compact_tokens=self._last_compact_tokens,
        ):
            return
        before = len(self.messages)
        state_block = build_research_state_block(self.toolbox)
        self.messages = compact_messages(
            self.messages, state_block=state_block, tail_turns=self.config.compact_tail_turns)
        after = len(self.messages)
        # keep the ledger in sync with the compacted history (so a resume replays
        # the compacted conversation, not the pre-compaction one).
        self._ledger.rewrite(self.messages)
        # remember the size we compacted at, so we don't recompact every turn
        self._last_compact_tokens = self._last_prompt_tokens
        self._emit("compaction", messages_before=before, messages_after=after,
                   prompt_tokens=self._last_prompt_tokens)

    def run(self, goal: str) -> dict[str, Any]:
        """Drive the tool-use loop for one goal until finish / budget exhaustion."""
        specs = self.toolbox.specs()
        resuming = bool(self.resume and self._resumed_turns)
        self._emit(ev.RUN_BEGIN, task=self.context.task_name, metric=self.context.metric,
                   metric_direction=self.context.metric_direction,
                   max_iterations=self.config.max_turns, mcgs=False,
                   mode="agent", goal=goal, resumed=resuming,
                   resumed_turns=self._resumed_turns, **self.run_meta)
        # Seed the conversation. A resume continues the reloaded transcript with a
        # short continuation turn (grounded in the RESTORED search graph) rather
        # than re-seeding the full task briefing; a fresh run gets the full seed.
        if resuming:
            seed = {"role": "user", "content": self._resume_prompt(goal)}
        else:
            seed = {"role": "user", "content": self._initial_prompt(goal)}
        self.messages.append(seed)
        self._ledger.append(seed)

        finished = False
        turns = 0
        while turns < self.config.max_turns and not finished:
            turns += 1
            # Compact the history if the last prompt grew past the threshold. The
            # durable state is the search graph, so we splice a deterministic
            # research-state block into the head and keep only the recent tail —
            # no LLM summary call, no risk of hallucinating a result.
            self._maybe_compact()
            turn = self.client.send(
                self.messages, system=_SYSTEM, tools=specs,
                max_tokens=self.config.max_tokens, temperature=self.config.temperature,
            )
            self._last_prompt_tokens = turn.input_tokens
            if turn.text:
                self._emit(ev.AGENT_MSG, text=turn.text, model=turn.model,
                           input_tokens=turn.input_tokens, output_tokens=turn.output_tokens)
            # Echo the assistant turn verbatim into history (tool_use blocks and all).
            assistant_msg = {"role": "assistant", "content": turn.raw_content}
            self.messages.append(assistant_msg)
            self._ledger.append(assistant_msg)

            if not turn.wants_tool:
                # The model is talking, not acting. In non-interactive mode that
                # means it's done reasoning for now; end cleanly.
                break

            tool_results: list[dict[str, Any]] = []
            for call in turn.tool_calls:
                self._emit(ev.TOOL_CALL, tool=call.name, args_brief=self._brief(call.input))
                # Guardrail: may block this call BEFORE it runs (repeated failure /
                # truncated code). A block returns a corrective message, not a result.
                decision = self.guardrails.before_call(call.name, call.input)
                if decision.blocked:
                    self._emit(ev.TOOL_RESULT, tool=call.name, ok=False, summary=decision.summary)
                    tool_results.append(ToolResult(call.id, decision.content, is_error=True).to_wire())
                    continue
                outcome = self.toolbox.dispatch(call.name, call.input)
                # Record the outcome so the guardrail can track failure streaks and
                # idempotent-read spin, then append any spin nudge to the result.
                self.guardrails.after_call(call.name, call.input, ok=outcome.ok,
                                           result_content=outcome.content)
                nudge = self.guardrails.idempotent_spin_warning(call.name)
                self._emit(ev.TOOL_RESULT, tool=call.name, ok=outcome.ok, summary=outcome.summary)
                content = outcome.content[: self.config.tool_result_cap]
                if nudge:
                    content = f"{content}\n\n{nudge}"
                tool_results.append(ToolResult(call.id, content, is_error=not outcome.ok).to_wire())
                if outcome.finished:
                    finished = True
                    self._agent_summary = str(call.input.get("summary") or "")
            # Feed every tool_result back as ONE user turn (Anthropic requires all
            # tool_use blocks from a turn to be answered before the next send).
            if tool_results:
                results_msg = {"role": "user", "content": tool_results}
                self.messages.append(results_msg)
                self._ledger.append(results_msg)
            # Consecutive-failure halt: stop with an honest partial result rather
            # than flail through the whole budget.
            if self.guardrails.halted:
                self._emit(ev.AGENT_MSG, text=f"[guardrail halt] {self.guardrails.halt_reason}")
                break

        summary = self._finalize(turns=turns, finished=finished)
        self._emit(ev.RUN_END, task=self.context.task_name,
                   best_exp_id=summary.get("best_exp_id"),
                   best_cv_score=summary.get("best_cv_score"),
                   n_iterations=summary.get("n_iterations"),
                   n_promotions=summary.get("n_promotions"))
        return summary

    def _initial_prompt(self, goal: str) -> str:
        ctx = self.context
        # Inject the COMPACT experience index up front (layer 1) so the agent starts
        # grounded in what past runs learned — the plan's "experience-driven" thesis.
        # Detailed lessons stay behind read_memory (layer 2), keeping this cheap.
        try:
            digest = self.toolbox.library.index_digest(ctx.task_type)
        except Exception:  # memory must never block a run
            digest = "(experience library unavailable)"
        return (
            f"TASK: {ctx.task_name}\n"
            f"modality={ctx.modality} | task_type={ctx.task_type} | "
            f"metric={ctx.metric} ({ctx.metric_direction})\n"
            f"target_column={ctx.target_column or 'unknown'} | id_column={ctx.id_column or 'none'}\n"
            f"n_train={ctx.n_train} | n_test={ctx.n_test}\n"
            f"DATA SCHEMA:\n{ctx.data_schema or '(infer from the CSV files via inspect_data)'}\n"
            f"{('NOTES: ' + ctx.extra_notes) if ctx.extra_notes else ''}\n\n"
            f"EXPERIENCE LIBRARY (from past runs — reuse what worked, avoid past failures):\n"
            f"{digest}\n\n"
            f"GOAL: {goal}\n\n"
            "Begin by grounding yourself in the data and any past lessons, then plan your "
            "first experiment."
        )

    def _resume_prompt(self, goal: str) -> str:
        """Continuation turn for a resumed run: a deterministic snapshot of the
        RESTORED search graph (so the model re-grounds in what the prior run
        actually proved, never on a hallucinated memory of it) plus the new goal."""
        try:
            state_block = build_research_state_block(self.toolbox)
        except Exception:  # never let the snapshot break a resume
            state_block = "(research state unavailable)"
        return (
            "RESUMING a prior run of this task. The conversation above is the earlier "
            "transcript; the run was interrupted before it finished. Here is the "
            "current, authoritative research state reconstructed from the persisted "
            "search graph (trust THIS over anything above if they disagree):\n\n"
            f"{state_block}\n\n"
            f"GOAL (continue): {goal}\n\n"
            "Do NOT restart from scratch or re-run experiments that already succeeded. "
            "Read the restored tree, then plan the next experiment that builds on the "
            "best-so-far."
        )

    @staticmethod
    def _brief(args: dict[str, Any]) -> str:
        """A short, blob-free arg summary for the event stream (no full code)."""
        parts = []
        for k, v in args.items():
            if isinstance(v, str) and len(v) > 40:
                parts.append(f"{k}=<{len(v)} chars>")
            else:
                parts.append(f"{k}={v}")
        return ", ".join(parts)[:160]

    def _finalize(self, *, turns: int, finished: bool) -> dict[str, Any]:
        """Write search_graph.json + summary.json (dashboard-visible) and return
        the summary dict, mirroring EvolutionLoop.summary()'s shape."""
        graph = self.toolbox.graph
        best_node = graph.nodes.get(self.toolbox.best_exp_id) if self.toolbox.best_exp_id else None
        promoted_ids = [n.exp_id for n in graph.nodes.values() if n.promoted]
        summary = {
            "task": self.context.task_name,
            "mode": "agent",
            "best_exp_id": self.toolbox.best_exp_id,
            "best_cv_score": best_node.cv_score if best_node else None,
            "metric": self.context.metric,
            "metric_direction": self.context.metric_direction,
            "n_iterations": len(graph.nodes),
            "n_promotions": len(promoted_ids),
            "promotion_history": graph.promotion_history,
            "turns_used": turns,
            "finished_by_agent": finished,
            "agent_summary": self._agent_summary,
        }
        (self.exp_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        graph.export_json(self.exp_dir / "search_graph.json")
        if self.toolbox.best_code:
            (self.exp_dir / "best_solution.py").write_text(self.toolbox.best_code, encoding="utf-8")
        # Auto-generate the deterministic research report from the artifacts just
        # written (search graph + summary + memory) — every claim is graph-backed.
        try:
            write_report(self.exp_dir)
        except Exception:  # report generation must never fail a completed run
            pass
        return summary
