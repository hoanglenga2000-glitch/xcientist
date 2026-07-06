"""The research tool set the deep agent drives — and where the guardrails live.

Each tool wraps an existing ``research_os`` primitive. The agent may *request*
any of them; the outcome of the safety-critical ones is decided by deterministic
code here, NOT by anything the model says:

  * ``run_experiment`` — the Runner decides success/exit_code. The agent cannot
    declare a crashed run "successful".
  * ``evaluate_promotion`` — ``SearchGraph.decide_promotion`` decides, using the
    node's *recorded* ``run_success``. A crashed run is never promotable even if
    it flushed a score to disk (the classic remote-GPU-kill hole).
  * ``audit_conclusion`` — ``claim_audit.audit_claim`` decides; thin evidence is
    rejected with the fixed boundary text.
  * ``submit_to_kaggle`` — always returns BLOCKED behind a human gate; the agent
    can never auto-submit.

So the no-fabrication invariants are the same ones the fixed loop enforces; they
have simply moved from "hardcoded in the ladder" to "hardcoded in the guardrail".
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from .memory_library import MemoryLibrary
from .messaging import ToolSpec
from ..claim_audit import audit_claim
from ..evolution_loop import _clean_error_for_feedback, _classify_failure
from ..mcgs_selector import ExpansionPlan, MCGSSelector
from ..retrospective_memory import MemoryRecord, RetrospectiveMemoryStore
from ..search_graph import ExperimentNode, SearchGraph
from ..strategy_selector import TaskProfile, recommend_strategies
from ..validation_contract import check_required_artifacts, create_contract, evaluate_acceptance
from ..variation_generator import TaskContext

# What decide_promotion requires on disk before a candidate may be promoted.
_REQUIRED_ARTIFACTS = ["metrics.json", "submission.csv"]
# Bound how much of a data preview / stdout we hand back to the model per tool.
_MAX_PREVIEW_CHARS = 4000


@dataclass
class ToolOutcome:
    """A tool's return: text the model reads + a one-line summary for the event
    stream + an ``ok`` flag for the dashboard/terminal renderer."""

    content: str
    summary: str
    ok: bool = True
    finished: bool = False  # set by `finish` to end the session


class ResearchToolbox:
    """Holds the run's live state (search graph, memory, runner) and exposes the
    tool handlers. One instance per agent session."""

    def __init__(
        self,
        context: TaskContext,
        *,
        data_dir: str,
        work_dir: str | Path,
        runner: Any,
        memory: Optional[RetrospectiveMemoryStore] = None,
        required_artifacts: Optional[list[str]] = None,
        selector: Optional[MCGSSelector] = None,
        allowed_tools: Optional[set[str]] = None,
        audit_spawner: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self.context = context
        # Optional HARD tool whitelist. When set, specs() advertises only these and
        # dispatch() refuses anything else — this is how a read-only audit sub-agent
        # is made structurally unable to mutate data / graph / promotion (the plan's
        # "只读审计" requirement enforced by code, not by prompt).
        self.allowed_tools = allowed_tools
        self.data_dir = data_dir
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.runner = runner
        self.memory = memory or RetrospectiveMemoryStore(self.work_dir.parent / "retrospective_memory.json")
        # Layered experience access (index digest + on-demand detail) over the
        # same store — this is how the plan's "experience reuse" reaches the agent.
        self.library = MemoryLibrary(self.memory)
        self.required_artifacts = required_artifacts or list(_REQUIRED_ARTIFACTS)
        self.graph = SearchGraph(
            task_id=context.task_name, root_exp_id="EXP000",
            metric_name="cv_score", metric_direction=context.metric_direction,
        )
        # 分层共治: the MCGS brain owns TOPOLOGY (which node to expand, expansion
        # type, coding mode). When present, run_experiment refuses to run without a
        # pending ExpansionPlan the selector produced — so the auditable selection
        # step (plan innovation #1/#2) can never be bypassed by the model. When None
        # (--no-mcgs), the toolbox falls back to model-chosen parents (Phase A).
        self.selector = selector
        self._pending_plan: Optional[ExpansionPlan] = None
        self._last_run_exp_id: Optional[str] = None
        self.code_by_exp: dict[str, str] = {}
        self.best_exp_id: Optional[str] = None
        self.best_code: Optional[str] = None
        self._exp_counter = 0
        # Optional emitters wired by the session so tool handlers can push the
        # SAME select/score/promote/lesson events the fixed loop emits (dashboard parity).
        self.emit: Callable[..., None] = lambda *a, **k: None
        # Optional callback (wired by the session for the TOP-LEVEL agent only) that
        # spawns the read-only audit sub-agent over THIS toolbox's graph and returns a
        # compact brief. Left None for restricted/audit toolboxes so an audit agent can
        # never recursively fan out (depth is also capped in spawn_audit_agent).
        self.audit_spawner = audit_spawner

    # ── resume: rehydrate the audited research state from a prior run ──────────
    def restore_from(self, exp_dir: str | Path) -> dict[str, Any]:
        """Reload a prior run's persisted state so the agent can CONTINUE it.

        Reads ``<exp_dir>/search_graph.json`` and rebuilds the graph in place, so
        the resumed session sees every experiment, edge, promotion and best-so-far
        that already happened — and ``_finalize`` re-exports the SAME lineage
        instead of overwriting it with an empty graph (which would silently destroy
        the record of prior work — a no-fabrication violation).

        This method NEVER fabricates: it only reloads what was durably written.
        Whatever it cannot recover (e.g. non-best per-experiment source, or the
        selector's private visit table) is re-seeded conservatively, never invented.

        Returns a small dict describing what was restored (for the resume event).
        """
        exp_dir = Path(exp_dir)
        graph_path = exp_dir / "search_graph.json"
        if not graph_path.exists():
            raise FileNotFoundError(f"cannot resume: no search_graph.json in {exp_dir}")

        self.graph = SearchGraph.load_json(graph_path)
        # best-so-far comes straight from the persisted graph (the audited verdict).
        self.best_exp_id = self.graph.best_exp_id
        best_solution = exp_dir / "best_solution.py"
        if self.best_exp_id and best_solution.exists():
            self.best_code = best_solution.read_text(encoding="utf-8")
            # the best node's code is the only per-exp source guaranteed on disk.
            self.code_by_exp[self.best_exp_id] = self.best_code

        # Bump the experiment counter PAST every existing EXP### so new nodes never
        # collide with restored ones (ids are allocated as f"EXP{counter:03d}").
        max_idx = -1
        for exp_id in self.graph.nodes:
            m = re.fullmatch(r"EXP(\d+)", exp_id)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        self._exp_counter = max_idx + 1

        # Re-seed the selector's private side-table so topology stays coherent. The
        # visit/branch history isn't part of the audited schema, so we conservatively
        # register every restored node as its own visited node on branch_0. This never
        # affects the audited graph; at worst the MCGS UCT re-warms over a few steps.
        if self.selector is not None:
            try:
                for exp_id in self.graph.nodes:
                    self.selector.visits.setdefault(exp_id, 1)
                    self.selector.branch_of.setdefault(exp_id, "branch_0")
            except Exception:  # selector bookkeeping must never break a resume
                pass

        return {
            "restored_nodes": len(self.graph.nodes),
            "best_exp_id": self.best_exp_id,
            "next_exp_id": f"EXP{self._exp_counter:03d}",
            "promotions": len([n for n in self.graph.nodes.values() if getattr(n, "promoted", False)]),
        }

    # ── tool specs advertised to the model ────────────────────────────────────
    def specs(self) -> list[ToolSpec]:
        ctx = self.context
        all_specs = [
            ToolSpec(
                "inspect_data",
                "Read the dataset's schema and a small preview (columns, dtypes, head, "
                "basic stats) for train/test/sample_submission. Read-only; call this "
                "early to ground your hypotheses in what the data actually looks like.",
                {"type": "object", "properties": {
                    "rows": {"type": "integer", "description": "preview rows (default 5, max 20)"}}},
            ),
            ToolSpec(
                "recommend_strategies",
                "Get deterministic, task-profile-based strategy suggestions (e.g. "
                "target_encoding, oof_stacking, log1p_target) with rationale.",
                {"type": "object", "properties": {}},
            ),
            ToolSpec(
                "read_memory",
                f"Retrieve detailed retrospective lessons — what worked, what failed, and "
                f"failure patterns from past experiments. Defaults to task_type='{ctx.task_type}'; "
                "set all_tasks=true for transferable cross-task lessons, or failure_pattern to "
                "study a specific failure class (e.g. 'timeout', 'oom').",
                {"type": "object", "properties": {
                    "all_tasks": {"type": "boolean", "description": "include lessons from other task types"},
                    "failure_pattern": {"type": "string",
                                        "description": "filter to one pattern: timeout|oom|segfault|..."},
                }},
            ),
            ToolSpec(
                "search_memory",
                "SEMANTIC search over ALL memory records using TF-IDF + cosine similarity. "
                "Unlike read_memory (exact filters), this finds the MOST RELEVANT lessons "
                "by MEANING — critical with 100+ records where filtering by task_type='regression' "
                "would return too many. Query in natural language: 'timeout on text models with "
                "large vocab', 'successful feature engineering for tabular', 'GPU import errors'. "
                "Returns top results ranked by relevance score (0-1).",
                {"type": "object", "properties": {
                    "query": {"type": "string", "description": "natural-language search query"},
                    "k": {"type": "integer", "description": "number of results (default 8, max 20)"},
                    "task_type": {"type": "string", "description": "optional: narrow to one task_type"},
                }, "required": ["query"]},
            ),
            ToolSpec(
                "read_search_tree",
                "Read the current search tree: every experiment node with its cv_score, "
                "run_success, promotion decision, and the current best.",
                {"type": "object", "properties": {}},
            ),
            ToolSpec(
                "plan_next_experiment",
                "Ask the MCGS selection brain which node to expand next. It returns an "
                "auditable plan: the node to expand FROM, the expansion type (primary / "
                "intra_branch / cross_branch / aggregation), the coding mode (Base/Stepwise/"
                "Diff), and — for cross_branch/aggregation — reference solutions to borrow or "
                "fuse. You MUST call this before run_experiment; it decides the topology, you "
                "decide the science (the hypothesis and code) for the node it picks.",
                {"type": "object", "properties": {}},
            ),
            ToolSpec(
                "run_experiment",
                "Execute a candidate solution for the CURRENTLY PLANNED expansion (call "
                "plan_next_experiment first). Provide the FULL runnable Python script "
                "(honoring the solution contract: reads --data-dir, prints CV_SCORE=<float>, "
                "writes submission.csv + metrics.json to --out-dir) plus a 1-2 sentence "
                "hypothesis. The node's parent, branch, and coding mode come from the plan — "
                "you supply only the science. The RUNNER decides success — a crash/timeout is "
                "reported as failure regardless of any score on disk.",
                {"type": "object", "properties": {
                    "hypothesis": {"type": "string", "description": "what this change tests and why"},
                    "code": {"type": "string", "description": "the complete runnable Python script"},
                }, "required": ["hypothesis", "code"]},
            ),
            ToolSpec(
                "evaluate_promotion",
                "Ask the promotion gate to rule on an experiment. The gate is deterministic: "
                "a failed run is NEVER promoted; a success is promoted only if it beats the "
                "best-so-far under the metric direction AND required artifacts exist. You "
                "cannot override this ruling.",
                {"type": "object", "properties": {
                    "exp_id": {"type": "string", "description": "the experiment to evaluate"}},
                    "required": ["exp_id"]},
            ),
            ToolSpec(
                "record_lesson",
                "Write a reusable lesson to retrospective memory so future rounds (and "
                "future tasks of this type) learn from this experiment.",
                {"type": "object", "properties": {
                    "exp_id": {"type": "string"},
                    "what_worked": {"type": "string"},
                    "what_failed": {"type": "string"},
                    "reusable_strategy": {"type": "string"},
                    "failure_pattern": {"type": "string",
                                        "description": "one of: timeout, oom, segfault, or '' if the run succeeded"},
                }, "required": ["exp_id"]},
            ),
            ToolSpec(
                "audit_conclusion",
                "Before stating a result as a finding, submit the claim for a deterministic "
                "evidence audit. Thin evidence is REJECTED with a fixed boundary; you must "
                "respect the ruling and only report what is allowed.",
                {"type": "object", "properties": {
                    "exp_id": {"type": "string"},
                    "claim": {"type": "string", "description": "the improvement/finding you want to state"},
                }, "required": ["exp_id", "claim"]},
            ),
            ToolSpec(
                "request_audit",
                "Spawn an INDEPENDENT read-only audit sub-agent to review your search tree and "
                "evidence before you conclude. It runs in its own isolated context with a HARD "
                "whitelist (inspect/read/audit only) — it structurally cannot run, promote, or "
                "mutate anything — and returns a compact verdict on whether your conclusions are "
                "supported. Use it once you have a promoted best and want a second opinion, or "
                "when a result looks too good. You get only its summary, not its transcript.",
                {"type": "object", "properties": {
                    "focus": {"type": "string",
                              "description": "what the auditor should scrutinize (e.g. 'is EXP003's "
                                             "gain real or CV leakage?'); optional"}}},
            ),
            ToolSpec(
                "submit_to_kaggle",
                "Request an official Kaggle submission. This ALWAYS returns blocked: official "
                "submission requires explicit human approval (the Human Gate) and is never "
                "automated. Use this only to record the intent.",
                {"type": "object", "properties": {
                    "exp_id": {"type": "string"}}, "required": ["exp_id"]},
            ),
            ToolSpec(
                "finish",
                "End the research session. Provide a short summary of the outcome (best score, "
                "what was learned). Only call this when you are done or blocked.",
                {"type": "object", "properties": {
                    "summary": {"type": "string"}}, "required": ["summary"]},
            ),
        ]
        if self.allowed_tools is not None:
            # Advertise only whitelisted tools (finish is always allowed so a
            # restricted agent can still end its session cleanly).
            allowed = self.allowed_tools | {"finish"}
            all_specs = [s for s in all_specs if s.name in allowed]
        return all_specs

    def dispatch(self, name: str, args: dict[str, Any]) -> ToolOutcome:
        # HARD whitelist deny (read-only audit agent): a restricted toolbox refuses
        # any tool outside its whitelist, so it structurally cannot mutate/run/promote.
        if self.allowed_tools is not None and name not in (self.allowed_tools | {"finish"}):
            return ToolOutcome(
                f"tool `{name}` is not permitted for this (read-only) agent.",
                f"{name} denied (not in whitelist)", ok=False)
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return ToolOutcome(f"unknown tool: {name}", f"unknown tool {name}", ok=False)
        try:
            return handler(args)
        except Exception as exc:  # a tool error is data for the model, not a crash
            return ToolOutcome(f"tool {name} errored: {type(exc).__name__}: {exc}",
                               f"{name} errored: {type(exc).__name__}", ok=False)

    # ── read-only tools ───────────────────────────────────────────────────────
    def _tool_inspect_data(self, args: dict[str, Any]) -> ToolOutcome:
        rows = max(1, min(int(args.get("rows", 5) or 5), 20))
        data_path = Path(self.data_dir) if self.data_dir else None
        if not data_path or not data_path.exists():
            return ToolOutcome(
                f"data dir not available: {self.data_dir or '(unset)'}. Task declares "
                f"schema:\n{self.context.data_schema or '(none)'}",
                "data dir unavailable — used declared schema", ok=False)
        lines: list[str] = [f"data dir: {data_path}"]
        for fname in ("train.csv", "test.csv", "sample_submission.csv"):
            fpath = data_path / fname
            if not fpath.exists():
                lines.append(f"\n{fname}: (missing)")
                continue
            lines.append(f"\n=== {fname} ===")
            lines.append(self._preview_csv(fpath, rows))
        content = "\n".join(lines)[:_MAX_PREVIEW_CHARS]
        return ToolOutcome(content, f"inspected {data_path.name} ({rows}-row preview)")

    def _preview_csv(self, path: Path, rows: int) -> str:
        try:  # pandas if present (richer), else stdlib csv
            import pandas as pd  # noqa: PLC0415
            df = pd.read_csv(path, nrows=2000)
            head = df.head(rows).to_string(max_cols=40)
            dtypes = "; ".join(f"{c}:{t}" for c, t in list(df.dtypes.astype(str).items())[:40])
            return (f"shape(sampled<=2000)={df.shape}\ncolumns/dtypes: {dtypes}\n"
                    f"head({rows}):\n{head}")
        except ImportError:
            import csv  # noqa: PLC0415
            with path.open(encoding="utf-8", newline="") as fh:
                reader = csv.reader(fh)
                out = []
                for i, row in enumerate(reader):
                    if i > rows:
                        break
                    out.append(", ".join(row[:40]))
            return "\n".join(out)

    def _tool_recommend_strategies(self, _args: dict[str, Any]) -> ToolOutcome:
        ctx = self.context
        profile = TaskProfile(
            modality=ctx.modality, task_type=ctx.task_type,
            train_size=ctx.n_train, test_size=ctx.n_test, metric=ctx.metric,
        )
        rec = recommend_strategies(profile)
        if not rec.strategies:
            return ToolOutcome("no strategy triggered for this profile; use strong defaults "
                               "(GBM + clean K-fold CV).", "no strategy triggered")
        body = "\n".join(f"- {s}: {rec.rationale.get(s, '')}"
                         f"  [{rec.expected_gains.get(s, '')}]" for s in rec.strategies)
        return ToolOutcome(f"recommended strategies:\n{body}",
                           f"{len(rec.strategies)} strategies suggested")

    def _tool_read_memory(self, args: dict[str, Any]) -> ToolOutcome:
        task_type = None if args.get("all_tasks") else self.context.task_type
        failure_pattern = args.get("failure_pattern")
        records = self.library.retrieve(task_type, failure_pattern=failure_pattern, limit=12)
        if not records:
            scope = "any task" if task_type is None else f"task_type={task_type}"
            fp = f" pattern={failure_pattern}" if failure_pattern else ""
            return ToolOutcome(f"(no retrospective memory for {scope}{fp} yet)", "memory empty")
        lines = []
        for r in records:
            piece = f"- {r.get('memory_id')} [{r.get('task_type')}] method={r.get('method')}"
            if r.get("what_worked"):
                piece += f" | WORKED: {r['what_worked']}"
            if r.get("what_failed"):
                piece += f" | FAILED: {r['what_failed']}"
            if r.get("reusable_strategy"):
                piece += f" | STRATEGY: {r['reusable_strategy']}"
            if r.get("failure_pattern"):
                piece += f" | pattern={r['failure_pattern']}"
            lines.append(piece)
        return ToolOutcome("\n".join(lines)[:_MAX_PREVIEW_CHARS],
                           f"{len(records)} lessons retrieved")

    def _tool_search_memory(self, args: dict[str, Any]) -> ToolOutcome:
        """Semantic (TF-IDF + cosine) search over ALL memory records.

        Unlike read_memory (exact filters), this finds the MOST RELEVANT
        lessons by meaning — critical when the memory store has 500+ records
        and exact task_type matching returns too many results.

        Args:
            query: natural-language search (e.g. "timeout on large text models")
            k: number of results (default 8)
            task_type: optional task_type filter (still ranked by similarity)
        """
        query = args.get("query", "")
        if not query:
            return ToolOutcome("search_memory requires a 'query' argument", "no query")
        k = min(int(args.get("k", 8)), 20)
        task_type = args.get("task_type") or None
        records = self.library.semantic_search(query, k=k, task_type=task_type)
        if not records:
            return ToolOutcome(f"(no memory records similar to: {query})", "no results")
        lines = []
        for r in records:
            score = r.get("_score", 0)
            piece = f"- [{score:.3f}] {r.get('memory_id')} [{r.get('task_type')}] method={r.get('method')}"
            if r.get("what_worked"):
                worked = r["what_worked"][:120]
                piece += f" | WORKED: {worked}"
            if r.get("what_failed"):
                failed = r["what_failed"][:120]
                piece += f" | FAILED: {failed}"
            if r.get("reusable_strategy"):
                piece += f" | STRATEGY: {r['reusable_strategy']}"
            if r.get("failure_pattern"):
                piece += f" | pattern={r['failure_pattern']}"
            lines.append(piece)
        return ToolOutcome("\n".join(lines)[:_MAX_PREVIEW_CHARS],
                           f"{len(records)} semantically similar lessons (search query: {query[:80]})")

    def _tool_read_search_tree(self, _args: dict[str, Any]) -> ToolOutcome:
        if not self.graph.nodes:
            return ToolOutcome("(search tree is empty — no experiments run yet)",
                               "tree empty")
        lines = [f"best_exp_id={self.best_exp_id or '(none)'}  "
                 f"metric={self.graph.metric_name}({self.graph.metric_direction})"]
        for node in self.graph.nodes.values():
            lines.append(
                f"- {node.exp_id} parent={node.parent_id or '-'} mode={node.branch_type} "
                f"cv={node.cv_score} success={node.run_success} decision={node.decision}")
        return ToolOutcome("\n".join(lines)[:_MAX_PREVIEW_CHARS],
                           f"{len(self.graph.nodes)} nodes")

    # ── 分层共治: the MCGS brain plans topology, the model does the science ─────
    def _tool_plan_next_experiment(self, _args: dict[str, Any]) -> ToolOutcome:
        if self.selector is None:
            # --no-mcgs fallback: no selector, so the model chooses freely. Emit a
            # trivial "primary" plan anchored on the current best so run_experiment
            # still has a pending plan to consume (uniform code path).
            parent = self.best_exp_id or self.graph.root_exp_id
            self._pending_plan = ExpansionPlan(node_exp_id=parent, expansion_type="primary",
                                               coding_mode="Base", branch_id="branch_0")
            return ToolOutcome(
                "MCGS disabled (--no-mcgs): expand freely from the current best. "
                f"parent={parent}. Write your hypothesis + code, then run_experiment.",
                "planned (mcgs off)", ok=True)
        plan = self.selector.select(self.graph, step=self._exp_counter)
        self._pending_plan = plan
        self.emit("select", exp_id=f"EXP{self._exp_counter:03d}", node_exp_id=plan.node_exp_id,
                  expansion_type=plan.expansion_type, coding_mode=plan.coding_mode,
                  reference_exp_ids=list(plan.reference_exp_ids or []), phase=plan.phase)
        refs_block = self._format_reference_solutions(plan)
        guidance = {
            "primary": "Improve the selected node with one meaningful change.",
            "intra_branch": "Deepen THIS branch using its own recent history.",
            "cross_branch": "This branch stalled — borrow 1-2 ideas from the reference "
                            "solutions below into the selected node's approach.",
            "aggregation": "The search plateaued — FUSE the reference solutions below into "
                           "ONE stronger solution (OOF stacking or probability blending).",
        }.get(plan.expansion_type, "")
        body = (f"MCGS PLAN (topology is decided; you write the science):\n"
                f"  expand from : {plan.node_exp_id}\n"
                f"  expansion   : {plan.expansion_type}\n"
                f"  coding mode : {plan.coding_mode}\n"
                f"  branch      : {plan.branch_id}   phase: {plan.phase}\n"
                f"  guidance    : {guidance}\n"
                f"{refs_block}\n"
                f"Now write a hypothesis + complete code for THIS expansion, then run_experiment.")
        return ToolOutcome(body, f"plan {plan.expansion_type}/{plan.coding_mode} from {plan.node_exp_id}",
                           ok=True)

    def _format_reference_solutions(self, plan: ExpansionPlan) -> str:
        """Give the model the reference solutions' code so it can borrow/fuse. Only
        cross_branch/aggregation/intra_branch carry references; primary has none."""
        if not plan.reference_exp_ids:
            return "  references  : (none)"
        lines = ["  reference solutions to draw on:"]
        for eid in plan.reference_exp_ids[:4]:
            node = self.graph.nodes.get(eid)
            code = self.code_by_exp.get(eid, "")
            if node is None:
                continue
            excerpt = code.strip()
            if len(excerpt) > 1400:  # keep the prompt bounded
                excerpt = excerpt[:1400] + "\n# ... (truncated)"
            lines.append(f"── {eid} (cv={node.cv_score})")
            if excerpt:
                lines += ["```python", excerpt, "```"]
        return "\n".join(lines)

    # ── guardrailed tools: outcome decided by deterministic code, not the model ─
    def _tool_run_experiment(self, args: dict[str, Any]) -> ToolOutcome:
        code = args.get("code") or ""
        if not code.strip():
            return ToolOutcome("run_experiment needs a non-empty `code` script.",
                               "rejected: empty code", ok=False)
        # GUARDRAIL (分层共治): topology must come from the MCGS brain, not the model.
        # No pending plan -> refuse, forcing the auditable selection step first.
        plan = self._pending_plan
        if plan is None:
            return ToolOutcome(
                "no experiment is planned. Call plan_next_experiment first — the MCGS brain "
                "decides which node to expand and the coding mode; you then write the code.",
                "rejected: no plan", ok=False)
        hypothesis = (args.get("hypothesis") or "").strip()
        mode = plan.coding_mode                    # from the plan, NOT the model
        parent = plan.node_exp_id if plan.node_exp_id in self.graph.nodes else None
        exp_id = f"EXP{self._exp_counter:03d}"
        self._exp_counter += 1
        self.code_by_exp[exp_id] = code
        out_dir = str(self.work_dir / exp_id / "out")

        self.emit("propose", exp_id=exp_id, mode=mode, expansion_type=plan.expansion_type,
                  parent_exp_id=parent, hypothesis=hypothesis)
        self.emit("exec_begin", exp_id=exp_id, runner=type(self.runner).__name__)
        # The RUNNER is the source of truth for success/exit_code. Nothing the
        # model wrote can make a crashed run count as successful.
        result = self.runner.run(code, data_dir=self.data_dir, out_dir=out_dir, exp_id=exp_id)
        self.emit("score", exp_id=exp_id, success=result.success,
                  cv_score=result.cv_score, exit_code=result.exit_code)

        node = ExperimentNode(
            exp_id=exp_id, parent_id=parent,
            branch_type=mode, task_name=self.context.task_name, hypothesis=hypothesis,
            implementation_summary=f"{mode}/{plan.expansion_type} candidate (agent)",
            code_path=f"{exp_id}/solution.py",
            artifacts=[{"path": Path(a).name} for a in result.artifacts],
            cv_score=result.cv_score, metric_name="cv_score",
            metric_direction=self.context.metric_direction,
            run_success=result.success,  # recorded from the Runner, used by the gate
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        self.graph.add_node(node)
        if parent:
            self.graph.add_edge(parent, exp_id, mode)
        # Register the child's branch with the selector so cross/aggregation topology
        # is tracked (backpropagation happens after evaluate_promotion rules).
        if self.selector is not None:
            try:
                self.selector.register_child(plan, exp_id)
            except Exception:  # selector bookkeeping must never crash a run
                pass
        self._last_run_exp_id = exp_id  # so evaluate_promotion can backpropagate
        self._pending_plan = None       # consumed; must plan again before next run

        if result.success:
            body = (f"exp_id={exp_id} SUCCESS cv_score={result.cv_score} exit={result.exit_code}\n"
                    f"artifacts: {[Path(a).name for a in result.artifacts]}\n"
                    f"stdout tail:\n{result.stdout_tail[:1200]}\n"
                    f"Next: call evaluate_promotion('{exp_id}') to let the gate rule.")
            return ToolOutcome(body, f"{exp_id} ok cv={result.cv_score}", ok=True)
        # Observable repair (创新四): strip progress-bar noise, persist the REAL
        # traceback to run_error.txt, and feed the clean error back so the next
        # Diff-mode proposal fixes the actual root cause (not a truncated summary).
        raw = result.error or result.stdout_tail or ""
        clean_full = _clean_error_for_feedback(raw, max_chars=8000)
        clean_feed = _clean_error_for_feedback(raw, max_chars=1500)
        pattern = _classify_failure(raw)
        try:
            (self.work_dir / exp_id).mkdir(parents=True, exist_ok=True)
            (self.work_dir / exp_id / "run_error.txt").write_text(clean_full, encoding="utf-8")
        except OSError:
            pass
        self.emit("repair", exp_id=exp_id, failure_pattern=pattern, error=clean_feed[:300])
        body = (f"exp_id={exp_id} FAILED (run did not complete successfully) exit={result.exit_code}\n"
                f"failure_pattern={pattern or 'unknown'}\n"
                f"This run is NOT promotable regardless of any score on disk.\n"
                f"REAL error (progress-bar noise stripped):\n{clean_feed}\n"
                f"Diagnose the root cause, then plan_next_experiment (it will likely give Diff "
                f"mode to fix this) and submit a corrected script.")
        return ToolOutcome(body, f"{exp_id} FAILED exit={result.exit_code} ({pattern})", ok=False)

    def _tool_evaluate_promotion(self, args: dict[str, Any]) -> ToolOutcome:
        exp_id = args.get("exp_id") or ""
        if exp_id not in self.graph.nodes:
            return ToolOutcome(f"no such experiment: {exp_id}. Run it first.",
                               f"unknown exp {exp_id}", ok=False)
        node = self.graph.nodes[exp_id]
        # GUARDRAIL: pass the node's RECORDED run_success into the gate. A crashed
        # run (run_success False) is refused here even if the model claims a score.
        decision = self.graph.decide_promotion(
            exp_id, parent_exp_id=self.best_exp_id, metric="cv_score",
            direction=self.context.metric_direction, min_delta=1e-4,
            required_artifacts=self.required_artifacts,
            run_success=bool(node.run_success),
        )
        promoted = bool(decision.get("promoted"))
        if promoted:
            self.best_exp_id = exp_id
            self.best_code = self.code_by_exp.get(exp_id, self.best_code)
        # Feed the outcome back to the MCGS brain: increment visit counts up the
        # ancestor chain (what makes UCT live, plan innovation #1) and update
        # branch/global stagnation (what makes DIVERSIFY reachable, innovation #2).
        if self.selector is not None:
            try:
                self.selector.backpropagate(self.graph, exp_id, improved=promoted)
            except Exception:  # backprop must never crash a run
                pass
        self._write_audit(node, decision)
        best_node = self.graph.nodes.get(self.best_exp_id) if self.best_exp_id else None
        self.emit("promote", exp_id=exp_id, promoted=promoted,
                  delta=decision.get("promotion_delta"), reason=decision.get("reason"),
                  best_exp_id=self.best_exp_id,
                  best_cv_score=(best_node.cv_score if best_node else None))
        verdict = "PROMOTED (new best)" if promoted else "HELD (not promoted)"
        body = (f"gate ruling for {exp_id}: {verdict}\n"
                f"reason: {decision.get('reason')}\n"
                f"candidate_score={decision.get('candidate_score')} "
                f"parent_score={decision.get('parent_score')} "
                f"delta={decision.get('promotion_delta')}\n"
                f"missing_artifacts={decision.get('missing_artifacts')}\n"
                f"current best={self.best_exp_id}")
        return ToolOutcome(body, f"{exp_id} {'promoted' if promoted else 'held'}", ok=True)

    def _tool_record_lesson(self, args: dict[str, Any]) -> ToolOutcome:
        exp_id = args.get("exp_id") or "unknown"
        node = self.graph.nodes.get(exp_id)
        delta = node.promotion_delta if node else None
        record = MemoryRecord(
            memory_id=f"{self.context.task_name}:{exp_id}",
            task_type=self.context.task_type,
            dataset_profile={"modality": self.context.modality, "n_train": self.context.n_train},
            method=(node.branch_type if node else "agent"),
            what_worked=(args.get("what_worked") or "").strip(),
            what_failed=(args.get("what_failed") or "").strip(),
            metric_delta=delta,
            reusable_strategy=(args.get("reusable_strategy") or "").strip(),
            failure_pattern=(args.get("failure_pattern") or "").strip().lower(),
            linked_exp_ids=[exp_id],
        )
        self.library.add(record)
        self.emit("lesson", exp_id=exp_id, failure_pattern=record.failure_pattern,
                  reusable_strategy=record.reusable_strategy, metric_delta=delta)
        return ToolOutcome(f"lesson recorded for {exp_id}.", f"lesson for {exp_id}", ok=True)

    def _tool_audit_conclusion(self, args: dict[str, Any]) -> ToolOutcome:
        exp_id = args.get("exp_id") or ""
        claim = (args.get("claim") or "").strip()
        node = self.graph.nodes.get(exp_id)
        if node is None:
            return ToolOutcome(f"no such experiment: {exp_id}.", f"unknown exp {exp_id}", ok=False)
        # GUARDRAIL: a claim is only backed if the run succeeded AND was promoted.
        # audit_claim rejects thin evidence deterministically; the model must obey.
        boundary = "Local CV/proxy only; no official rank without a Kaggle response artifact."
        audit = audit_claim(
            claim_id=f"{self.context.task_name}:{exp_id}:claim",
            claim_text=claim, related_exp_ids=[exp_id],
            contract={"hypothesis": node.hypothesis, "conclusion_boundary": boundary},
            supporting_metrics={"cv_score": node.cv_score},
            required_ablations=[], completed_ablations=[],
            evidence={"has_required_experiments": bool(node.run_success),
                      "has_mechanistic_evidence": bool(node.run_success and node.promoted),
                      "missing_evidence": [] if node.run_success else ["run did not succeed"]},
        )
        body = (f"audit result: {audit.audit_result.upper()}\n"
                f"drift_type: {audit.drift_type}\n"
                f"allowed_conclusion: {audit.allowed_conclusion}\n"
                f"missing_evidence: {audit.missing_evidence}")
        return ToolOutcome(body, f"audit {exp_id}: {audit.audit_result}",
                           ok=(audit.audit_result == "allow"))

    def _tool_request_audit(self, args: dict[str, Any]) -> ToolOutcome:
        # Delegation to the read-only audit sub-agent. The spawner is wired by the
        # session for the top-level agent only; a restricted (audit) toolbox has none,
        # so an auditor can never recursively spawn another auditor.
        if self.audit_spawner is None:
            return ToolOutcome(
                "audit sub-agent is not available in this context (only the top-level agent "
                "may request an audit). Use audit_conclusion for a per-claim evidence check.",
                "request_audit unavailable", ok=False)
        if not self.graph.nodes:
            return ToolOutcome(
                "nothing to audit yet — run at least one experiment first.",
                "request_audit: empty tree", ok=False)
        focus = (args.get("focus") or "").strip()
        try:
            brief = self.audit_spawner(focus)
        except Exception as exc:  # a child failure must never crash the parent
            return ToolOutcome(f"audit sub-agent errored: {type(exc).__name__}: {exc}",
                               "request_audit errored", ok=False)
        return ToolOutcome(brief, "audit sub-agent reported", ok=True)

    def _tool_submit_to_kaggle(self, args: dict[str, Any]) -> ToolOutcome:
        exp_id = args.get("exp_id") or "(unspecified)"
        # GUARDRAIL: official submission is ALWAYS blocked behind the human gate.
        # There is deliberately no code path here that submits.
        body = ("BLOCKED: official Kaggle submission requires explicit human approval "
                "(Human Gate) and is never automated by the agent. Intent recorded for "
                f"{exp_id}. A human must review artifacts and submit manually.")
        return ToolOutcome(body, f"kaggle submit blocked (human gate) for {exp_id}", ok=False)

    def _tool_finish(self, args: dict[str, Any]) -> ToolOutcome:
        summary = (args.get("summary") or "").strip() or "(no summary provided)"
        return ToolOutcome(f"session finished: {summary}", "finished", ok=True, finished=True)

    def _write_audit(self, node: ExperimentNode, decision: dict[str, Any]) -> None:
        """Persist validation_contract.json + claim_audit.json for this node, using
        the SAME research_os helpers the fixed loop uses, so the dashboard's
        evidence/gate views read identical artifacts for agent and loop runs."""
        exp_dir = self.work_dir / node.exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        artifact_names = [str(a.get("path")) for a in node.artifacts]
        boundary = "Local CV/proxy only; no official rank without a Kaggle response artifact."
        contract = create_contract(
            contract_id=f"{self.context.task_name}:{node.exp_id}:contract",
            exp_id=node.exp_id, claim=f"{node.branch_type} candidate for {self.context.task_name}",
            hypothesis=node.hypothesis,
            implementation_requirement="Runnable script emitting CV_SCORE, submission.csv, metrics.json.",
            metric="cv_score", baseline_exp_id=node.parent_id or "",
            acceptance_criteria=({"cv_score": {"min" if self.context.metric_direction == "maximize" else "max": node.cv_score}}
                                 if node.cv_score is not None else {}),
            ablation_plan=[], conclusion_boundary=boundary,
            required_artifacts=list(self.required_artifacts),
        )
        artifact_check = check_required_artifacts(contract, artifact_names)
        acceptance = evaluate_acceptance(contract, {"cv_score": node.cv_score} if node.cv_score is not None else {})
        (exp_dir / "validation_contract.json").write_text(json.dumps({
            "schema": "academic_research_os.validation_contract.v1",
            "contract_id": contract.contract_id, "exp_id": node.exp_id,
            "task_id": self.context.task_name, "created_at": datetime.now().isoformat(timespec="seconds"),
            "hypothesis": contract.hypothesis, "metric": contract.metric,
            "required_artifacts": contract.required_artifacts,
            "artifact_check": artifact_check, "acceptance": acceptance,
            "conclusion_boundary": contract.conclusion_boundary,
            "run_success": node.run_success, "cv_score": node.cv_score,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
