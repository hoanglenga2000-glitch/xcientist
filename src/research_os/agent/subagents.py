"""Sub-agent fan-out for the research agent.

Two uses, both bounded and context-isolated (the parent sees only a structured
summary, never the child's full transcript — the pattern Hermes' delegate_task
uses to keep child noise out of the parent context):

  * READ-ONLY AUDIT sub-agent: reviews the search graph / evidence and reports
    whether the conclusions are supported. It is given a HARD tool whitelist
    (inspect/read/audit only) so it structurally CANNOT mutate data, the graph, or
    promotion — the plan's "只读审计" requirement enforced by code.
  * WORKER sub-agents (future): parallel hypothesis branches. The scaffolding here
    (bounded budget, structured-summary return, depth cap) is shared.

Design rules mirrored from the reference agents:
  * a child returns a compact ``SubAgentResult`` (status + summary + metrics),
    never its raw message history;
  * the child has its OWN turn budget, independent of the parent's;
  * ``max_spawn_depth`` prevents runaway recursion (a child cannot itself fan out
    past the cap).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .session import AgentSession, AgentSessionConfig
from .tools import ResearchToolbox

# The read-only audit agent may ONLY inspect and audit — never run/promote/submit.
AUDIT_TOOLS = {"inspect_data", "read_memory", "read_search_tree",
               "recommend_strategies", "audit_conclusion"}


@dataclass
class SubAgentResult:
    """Compact result the parent sees — no raw transcript."""

    role: str                    # e.g. "audit" | "worker"
    status: str                  # "ok" | "error" | "budget_exhausted"
    summary: str                 # the child's own finish summary (or error)
    turns_used: int = 0
    detail: dict[str, Any] = field(default_factory=dict)

    def to_brief(self) -> str:
        lines = [f"[sub-agent {self.role}] status={self.status} turns={self.turns_used}",
                 self.summary]
        return "\n".join(lines)


def spawn_audit_agent(
    parent_toolbox: ResearchToolbox,
    *,
    goal: str,
    client: Any = None,
    exp_dir: Optional[Path] = None,
    max_turns: int = 8,
    depth: int = 0,
    max_spawn_depth: int = 1,
    on_event=None,
) -> SubAgentResult:
    """Spawn a read-only audit sub-agent over the parent's search graph.

    The child shares the parent's graph/memory (so it audits the REAL run) but
    through a restricted toolbox whose whitelist forbids any mutation. It returns
    only a structured summary.
    """
    if depth >= max_spawn_depth:
        return SubAgentResult(role="audit", status="error",
                              summary=f"spawn depth {depth} >= cap {max_spawn_depth}; not spawning")

    # A restricted toolbox that SHARES the parent's graph/memory but can only
    # inspect + audit. It reuses the same runner object but never calls it (no
    # run_experiment in the whitelist), so no code executes from the audit agent.
    audit_box = ResearchToolbox(
        parent_toolbox.context,
        data_dir=parent_toolbox.data_dir,
        work_dir=parent_toolbox.work_dir,
        runner=parent_toolbox.runner,
        memory=parent_toolbox.memory,
        selector=None,
        allowed_tools=AUDIT_TOOLS,
    )
    # Share the parent's live graph + code so the audit sees the actual results.
    audit_box.graph = parent_toolbox.graph
    audit_box.best_exp_id = parent_toolbox.best_exp_id
    audit_box.best_code = parent_toolbox.best_code
    audit_box.code_by_exp = parent_toolbox.code_by_exp

    child_dir = Path(exp_dir or (parent_toolbox.work_dir / "_audit"))
    session = AgentSession(
        context=parent_toolbox.context, toolbox=audit_box, exp_dir=child_dir,
        client=client, config=AgentSessionConfig(max_turns=max_turns), on_event=on_event,
    )
    try:
        summary = session.run(goal)
    except Exception as exc:  # a child failure must not crash the parent
        return SubAgentResult(role="audit", status="error",
                              summary=f"audit sub-agent errored: {type(exc).__name__}: {exc}")
    status = "ok" if summary.get("finished_by_agent") else "budget_exhausted"
    return SubAgentResult(
        role="audit", status=status,
        summary=str(summary.get("agent_summary") or "audit complete"),
        turns_used=int(summary.get("turns_used", 0)),
        detail={k: summary.get(k) for k in ("best_exp_id", "best_cv_score", "n_promotions")},
    )
