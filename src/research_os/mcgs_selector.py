"""Correct, tested MCGS selection brain for the evolution loop.

This is the *selection* half of a Monte-Carlo-Graph-Search evolutionary loop:
given the current ``research_os.search_graph.SearchGraph`` it decides WHICH node
to expand next, WHAT expansion type to use (primary / intra-branch / cross-branch
/ aggregation) and WHICH coding mode (base / stepwise / diff). Execution and the
promotion gate stay entirely in ``evolution_loop`` / ``search_graph`` — this file
never runs code and never mutates the audited node schema.

Why this exists (design note): the workstation already had a MCGS engine
(``strategy/mlevolve_search.py``) with the right *ideas* — piecewise exploration
decay, four expansion types, branch/global stagnation. But its UCT was
non-functional: ``visit_count`` was declared, read, and never incremented, so
``_uct_value`` always hit the ``visits==0 -> inf`` path and selection collapsed to
"first child". It also divided a final CV score by visits (direction-unaware, wrong
for minimize) and had zero tests. Rather than import that liability across packages
(and inherit a second ``SearchGraph`` name clash), we port the *sound* ideas here,
operate on B's own graph, keep a private visit side-table, and use B's
direction-aware ``_is_better`` so UCT is correct for maximize AND minimize.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Optional

from .experience_mcgs import (
    BudgetLedger,
    ExperienceBoard,
    ExperienceCardBuilder,
    ExperienceCost,
    LazySummaryCache,
    RetrievalBundle,
    SearchMode,
    SearchOperator,
    SelectionTrace,
    render_retrieval_context,
    retrieve_experience,
    route_operator,
    select_experience_parent,
)


# ── plan the loop asks for each round ────────────────────────────────────────
@dataclass
class ExpansionPlan:
    node_exp_id: str                 # node to expand FROM (not necessarily global best)
    expansion_type: str              # primary | intra_branch | cross_branch | aggregation
    coding_mode: str                 # Base | Stepwise | Diff  (matches VariationGenerator)
    reference_exp_ids: list[str] = field(default_factory=list)
    branch_id: str = ""              # branch the *new* child will belong to
    phase: str = "exploration"       # exploration | balanced | exploitation (for observability)
    operator: str = ""               # Draft | Improve | Debug | Crossover
    retrieval_card_ids: list[str] = field(default_factory=list)
    # Ordered selected parents.  Crossover requires exactly two entries from
    # distinct successful method families; the first remains the tree parent.
    parent_exp_ids: list[str] = field(default_factory=list)


# ── progressive exploration schedule (ported from the sound part of A) ───────
def _phase_for(progress: float) -> str:
    """Map normalized search progress [0,1] to a coarse phase."""
    if progress < 0.3:
        return "exploration"
    if progress < 0.7:
        return "balanced"
    return "exploitation"


def _piecewise_decay(step: int, total_steps: int, *, initial_c: float = 1.414,
                     lower_bound: float = 0.5) -> float:
    """Exploration constant C decays from 1.414 -> 0.5 across the budget.

    Flat during early exploration, linear decay through the middle, floored late.
    Scales to whatever total budget the caller uses (unlike A's hard-coded 500).
    """
    if total_steps <= 1:
        return lower_bound
    t1 = max(1, round(total_steps * 0.3))
    t2 = max(t1 + 1, round(total_steps * 0.7))
    if step < t1:
        return initial_c
    if step <= t2:
        frac = (step - t1) / max(1, (t2 - t1))
        return max(initial_c - (initial_c - lower_bound) * frac, lower_bound)
    return lower_bound


class MCGSSelector:
    """Selection brain over a ``research_os.search_graph.SearchGraph``.

    Keeps its OWN state (visit counts, branch membership, stagnation) in private
    side-tables keyed by ``exp_id`` so B's audited node schema is never touched.
    """

    def __init__(self, *, total_steps: int = 6, exploration_c: float = 1.414,
                 branch_stagnation_patience: int = 2, global_stagnation_patience: int = 4,
                 min_delta: float = 1e-4, max_branches: int = 3,
                 search_mode: str | SearchMode | None = None,
                 max_total_tokens: int = 2_000_000,
                 max_wall_seconds: float = 12 * 60 * 60,
                 max_cost_usd: float | None = None,
                 summarizer_model: str = "deterministic-public-summary-v1") -> None:
        self.total_steps = max(1, total_steps)
        self.exploration_c = exploration_c
        self.branch_stagnation_patience = branch_stagnation_patience
        self.global_stagnation_patience = global_stagnation_patience
        self.min_delta = min_delta
        self.max_branches = max_branches
        self.search_mode = search_mode if isinstance(search_mode, SearchMode) else SearchMode.parse(search_mode)
        self.experience_board: ExperienceBoard | None = None
        self.last_selection_trace: SelectionTrace | None = None
        self.last_retrieval_bundle: RetrievalBundle | None = None
        self.selection_traces: list[SelectionTrace] = []
        self.retrieval_records: list[dict[str, Any]] = []
        self.summary_cache: LazySummaryCache | None = None
        self.summarizer_model = str(summarizer_model)
        self.summary_prompt_template_hash = hashlib.sha256(
            b"method_overview,parent_comparison_experience:v1"
        ).hexdigest()
        self.summary_cache_hits = 0
        self.summary_cache_misses = 0
        self.budget = BudgetLedger(
            max_nodes=self.total_steps,
            max_total_tokens=max(1, int(max_total_tokens)),
            max_wall_seconds=max(1.0, float(max_wall_seconds)),
            max_cost_usd=max_cost_usd,
        )
        # private side-tables (never written back to ExperimentNode)
        self.visits: dict[str, int] = {}
        self.branch_of: dict[str, str] = {}          # exp_id -> branch_id
        self.branch_stagnation: dict[str, int] = {}  # branch_id -> consecutive no-improve
        self.global_stagnation = 0
        self._branch_counter = 0

    # ── reward normalization: make "better" always mean higher, any direction ──
    def _reward(self, graph, node) -> float:
        """Map a node's score to [0,1] where 1 is best under the metric direction.

        This is the fix for A's bug: A used ``score/visits`` which is wrong for a
        final CV score and ignores minimize metrics. We normalize against the range
        of evaluated scores and flip for minimize, using the graph's own direction.
        """
        score = graph._node_score(node)
        if score is None:
            return 0.0
        scored = [graph._node_score(n) for n in graph.nodes.values()]
        scored = [s for s in scored if s is not None]
        if len(scored) < 2:
            return 0.5
        lo, hi = min(scored), max(scored)
        if hi <= lo:
            return 0.5
        direction = (graph.metric_direction or "maximize").lower()
        if direction in {"minimize", "lower", "lower_is_better"}:
            return (hi - score) / (hi - lo)
        return (score - lo) / (hi - lo)

    def _uct(self, graph, node, parent_visits: int, c: float) -> float:
        """UCT = exploitation(reward) + C * sqrt(ln(parent_visits)/visits).

        Unvisited nodes return +inf so every frontier gets tried once (the correct
        MCTS behavior A only *looked* like it had).
        """
        v = self.visits.get(node.exp_id, 0)
        if v == 0:
            return float("inf")
        exploitation = self._reward(graph, node)
        exploration = c * math.sqrt(math.log(max(parent_visits, 1)) / v)
        return exploitation + exploration

    def _children(self, graph, exp_id: str) -> list:
        return [n for n in graph.nodes.values() if n.parent_id == exp_id]

    def select(self, graph, *, step: int) -> ExpansionPlan:
        """Walk from root by UCT to a frontier node, then decide expansion + mode.

        Falls back gracefully: if the graph has no root yet, the caller should be
        seeding the baseline (this returns a primary/Base plan on the root id).
        """
        c = _piecewise_decay(step, self.total_steps, initial_c=self.exploration_c)
        phase = _phase_for(step / self.total_steps)
        root_id = graph.root_exp_id
        # traverse to a leaf by best UCT child
        node = graph.nodes.get(root_id)
        if node is None:
            return ExpansionPlan(node_exp_id=root_id, expansion_type="primary",
                                 coding_mode="Base", branch_id="branch_0", phase=phase,
                                 operator=SearchOperator.DRAFT.value)
        if self.search_mode is SearchMode.EXPERIENCE_MCGS_V1:
            return self._select_experience(graph, phase=phase)
        while True:
            children = self._children(graph, node.exp_id)
            if not children:
                break
            parent_visits = self.visits.get(node.exp_id, 0)
            node = max(children, key=lambda ch: self._uct(graph, ch, parent_visits, c))
        return self._plan_expansion(graph, node, phase=phase)

    def _ensure_experience_board(self, graph, *, skip_node_ids: set[str] | None = None) -> ExperienceBoard:
        if self.experience_board is None:
            self.experience_board = ExperienceBoard(str(graph.task_id))
        for node in sorted(graph.nodes.values(), key=lambda item: item.exp_id):
            if node.exp_id in (skip_node_ids or set()):
                continue
            if self.experience_board.by_node(node.exp_id) is not None:
                continue
            status = "success" if getattr(node, "run_success", True) else "failed"
            parent_ids = list(getattr(node, "reference_parent_ids", []) or [])
            if node.parent_id and node.parent_id not in parent_ids:
                parent_ids.insert(0, node.parent_id)
            operator = SearchOperator.DRAFT if not node.parent_id else (SearchOperator.DEBUG if status == "failed" else SearchOperator.IMPROVE)
            card = ExperienceCardBuilder.build(
                task_id=str(graph.task_id),
                node_id=node.exp_id,
                parent_ids=parent_ids,
                operator=operator,
                method_family=str(node.branch_type or "unknown"),
                code=str(node.implementation_summary or node.code_path or ""),
                prompt=str(node.hypothesis or ""),
                execution_id=node.exp_id,
                status=status,
                public_validation_score=graph._node_score(node),
                metric_direction=str(graph.metric_direction or "maximize"),
                error=str(node.promotion_reason if status == "failed" else ""),
                provenance={"source": "search_graph_replay", "changes_summary": node.implementation_summary},
            )
            self.experience_board.append(card)
        return self.experience_board

    def _select_experience(self, graph, *, phase: str) -> ExpansionPlan:
        board = self._ensure_experience_board(graph)
        trace = select_experience_parent(board, visits=self.visits, parent_visits=self.visits, exploration_c=1.0)
        operator = route_operator(board, no_new_best_expansions=self.global_stagnation)
        selected_id = trace.selected_node_id
        parent_exp_ids: list[str] = []
        if operator is SearchOperator.DEBUG:
            failed = sorted((card for card in board.cards.values() if card.status != "success" or card.error_signature), key=lambda card: (card.execution_id, card.card_id))
            if failed:
                selected_id = failed[-1].node_id
            parent_exp_ids = [selected_id]
        elif operator is SearchOperator.CROSSOVER:
            ranked_cards = [
                board.cards[item.card_id]
                for item in trace.candidates
                if item.card_id in board.cards
                and board.cards[item.card_id].status == "success"
                and board.cards[item.card_id].public_validation_score is not None
            ]
            distinct: list[Any] = []
            seen_families: set[str] = set()
            for card in ranked_cards:
                family = card.method_family.strip().lower()
                if family in seen_families:
                    continue
                distinct.append(card)
                seen_families.add(family)
                if len(distinct) == 2:
                    break
            if len(distinct) != 2:
                # Router and selector must agree.  A Crossover without two
                # successful, different-family parents is an invalid plan.
                raise RuntimeError("crossover requires two successful distinct-family parents")
            parent_exp_ids = [distinct[0].node_id, distinct[1].node_id]
            selected_id = parent_exp_ids[0]
        elif operator is SearchOperator.IMPROVE:
            parent_exp_ids = [selected_id]
        bundle = retrieve_experience(board, operator, selected_node_id=selected_id, max_cards=8, max_tokens=6000)
        selected_card = board.by_node(selected_id)
        trace_updates: dict[str, Any] = {
            "selected_node_id": selected_id,
            "selected_card_id": selected_card.card_id if selected_card is not None else trace.selected_card_id,
        }
        trace_field_names = {item.name for item in dataclass_fields(trace)}
        if "operator" in trace_field_names:
            trace_updates["operator"] = operator
        if "selection_reason" in trace_field_names:
            trace_updates["selection_reason"] = (
                "two_distinct_successful_method_families" if operator is SearchOperator.CROSSOVER
                else "latest_failed_node" if operator is SearchOperator.DEBUG
                else "maximum_utility"
            )
        if "selected_parent_ids" in trace_field_names:
            trace_updates["selected_parent_ids"] = tuple(parent_exp_ids)
        if "budget_snapshot" in trace_field_names:
            trace_updates["budget_snapshot"] = self.budget.to_dict()
        trace = replace(trace, **trace_updates)
        self.last_selection_trace = trace
        self.last_retrieval_bundle = bundle
        self.selection_traces.append(trace)
        self.retrieval_records.append({
            "selected_node_id": selected_id,
            "selected_parent_ids": list(parent_exp_ids),
            **bundle.to_dict(),
            "cache_hit": None,
            "cache_status": "lazy_summary_pending",
            "card_cache_hits": 0,
            "card_cache_misses": 0,
        })
        selected_node = graph.nodes.get(selected_id) or graph.nodes[trace.selected_node_id]
        refs: list[str] = []
        for node_id in parent_exp_ids[1:]:
            if node_id in graph.nodes and node_id not in refs:
                refs.append(node_id)
        for card_id in bundle.card_ids:
            node_id = board.cards[card_id].node_id
            if node_id != selected_node.exp_id and node_id in graph.nodes and node_id not in refs:
                refs.append(node_id)
        bid = self._branch(selected_node.exp_id)
        n_branches = len(set(self.branch_of.values())) or 1
        if operator is SearchOperator.DEBUG:
            expansion_type, coding_mode, branch_id = "intra_branch", "Diff", bid
        elif operator is SearchOperator.CROSSOVER:
            expansion_type, coding_mode = ("aggregation" if self.global_stagnation >= self.global_stagnation_patience else "cross_branch"), "Stepwise"
            branch_id = self._new_branch() if n_branches < self.max_branches else bid
        elif operator is SearchOperator.DRAFT:
            expansion_type, coding_mode, branch_id = "primary", "Base", self._new_branch() if graph.nodes else "branch_0"
        else:
            expansion_type, coding_mode, branch_id = "intra_branch", "Stepwise", bid
        return ExpansionPlan(
            node_exp_id=selected_node.exp_id,
            expansion_type=expansion_type,
            coding_mode=coding_mode,
            reference_exp_ids=refs[:8],
            branch_id=branch_id,
            phase=phase,
            operator=operator.value,
            retrieval_card_ids=list(bundle.card_ids),
            parent_exp_ids=parent_exp_ids,
        )

    def prompt_context(self) -> str:
        if self.experience_board is None or self.last_retrieval_bundle is None:
            return ""
        rendered = render_retrieval_context(self.experience_board, self.last_retrieval_bundle)
        summaries: list[str] = []
        call_hits = 0
        call_misses = 0
        if self.summary_cache is not None:
            for card_id in self.last_retrieval_bundle.card_ids:
                card = self.experience_board.cards[card_id]
                summary, hit = self.summary_cache.get_or_create(
                    card,
                    summarizer_model=self.summarizer_model,
                    prompt_template_hash=self.summary_prompt_template_hash,
                    summarize=self._deterministic_rich_summary,
                )
                if hit:
                    self.summary_cache_hits += 1
                    call_hits += 1
                else:
                    self.summary_cache_misses += 1
                    call_misses += 1
                summaries.append(
                    f"- {card_id}: "
                    + json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                )
        budget = self.budget.to_dict()
        remaining = {
            "nodes": max(0, int(budget["max_nodes"]) - int(budget["nodes"])),
            "tokens": max(0, int(budget["max_total_tokens"]) - int(budget["total_tokens"])),
            "wall_seconds": max(0.0, float(budget["max_wall_seconds"]) - float(budget["wall_seconds"])),
            "cost_usd": (
                None
                if budget.get("max_cost_usd") is None
                else max(0.0, float(budget["max_cost_usd"]) - float(budget["estimated_cost_usd"]))
            ),
        }
        if summaries:
            rendered += "\nRICH EXPERIENCE SUMMARIES (lazy cache, public only):\n" + "\n".join(summaries)
        rendered += "\nREMAINING HARD BUDGET: " + json.dumps(remaining, sort_keys=True)
        if self.retrieval_records:
            record = self.retrieval_records[-1]
            record["card_cache_hits"] = call_hits
            record["card_cache_misses"] = call_misses
            record["cache_hit"] = bool(call_hits and not call_misses)
            record["cache_status"] = "hit" if call_hits and not call_misses else "miss" if call_misses else "empty"
        return rendered

    @staticmethod
    def _deterministic_rich_summary(card) -> dict[str, str]:
        overview = str(card.provenance.get("changes_summary") or card.method_family or "unknown")
        comparison = (
            f"parents={','.join(card.parent_ids)}; progress={card.progress:.6g}; "
            f"status={card.status}; error_signature={card.error_signature or 'none'}"
        )
        return {
            "method_overview": overview,
            "parent_comparison_experience": comparison,
        }

    def record_execution(
        self,
        graph,
        *,
        exp_id: str,
        plan: ExpansionPlan,
        code: str,
        prompt: str,
        method_family: str,
        success: bool,
        error: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        wall_seconds: float = 0.0,
        gpu_seconds: float = 0.0,
        estimated_cost_usd: float = 0.0,
        data_hashes: tuple[str, ...] | list[str] = (),
        artifact_paths: tuple[str, ...] | list[str] = (),
        execution_output: str = "",
        evaluator_version: str = "",
        environment_hash: str = "",
    ) -> None:
        if self.search_mode is not SearchMode.EXPERIENCE_MCGS_V1:
            return
        board = self._ensure_experience_board(graph, skip_node_ids={exp_id})
        node = graph.nodes[exp_id]
        operator = SearchOperator(plan.operator or (SearchOperator.DEBUG.value if not success else SearchOperator.IMPROVE.value))
        cost = ExperienceCost(
            prompt_tokens=max(0, int(prompt_tokens)),
            completion_tokens=max(0, int(completion_tokens)),
            total_tokens=max(0, int(prompt_tokens)) + max(0, int(completion_tokens)),
            wall_seconds=max(0.0, float(wall_seconds)),
            gpu_seconds=max(0.0, float(gpu_seconds)),
            estimated_cost_usd=max(0.0, float(estimated_cost_usd)),
        )
        parent_ids = list(getattr(plan, "parent_exp_ids", []) or [])
        if not parent_ids and node.parent_id:
            parent_ids = [node.parent_id]
        provenance: dict[str, Any] = {
            "source": "verified_execution",
            "split": "public_validation",
            "changes_summary": node.implementation_summary,
            "promotion_decision": node.decision,
            "artifact_hashes": self._artifact_hashes(artifact_paths),
            "execution_output_hash": hashlib.sha256((execution_output or "").encode("utf-8")).hexdigest(),
            "evaluator_version": str(evaluator_version or "unknown"),
            "environment_hash": str(environment_hash or "not_recorded"),
        }
        if success:
            repaired = next(
                (
                    parent.error_signature
                    for parent_id in parent_ids
                    if (parent := board.by_node(parent_id)) is not None
                    and parent.error_signature
                ),
                "",
            )
            if repaired:
                provenance["resolved_error_signature"] = repaired
                provenance["fixes_error_signature"] = repaired
        card = ExperienceCardBuilder.build(
            task_id=str(graph.task_id),
            node_id=exp_id,
            parent_ids=parent_ids,
            operator=operator,
            method_family=method_family or str(node.branch_type or "unknown"),
            code=code,
            data_hashes=data_hashes,
            prompt=prompt,
            execution_id=exp_id,
            status="success" if success else "failed",
            public_validation_score=graph._node_score(node),
            metric_direction=str(graph.metric_direction or "maximize"),
            error=error,
            cost=cost,
            provenance=provenance,
        )
        if board.append(card):
            self.budget.record(cost)

    @staticmethod
    def _artifact_hashes(paths: tuple[str, ...] | list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for raw in sorted({str(item) for item in paths if item}):
            path = Path(raw)
            if not path.is_file():
                continue
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            rows.append({"name": path.name, "size": path.stat().st_size, "sha256": digest.hexdigest()})
        return rows

    def export_observability(self, work_dir: str | Path) -> None:
        if self.search_mode is not SearchMode.EXPERIENCE_MCGS_V1 or self.experience_board is None:
            return
        root = Path(work_dir)
        if self.summary_cache is None:
            self.summary_cache = LazySummaryCache(root / "experience-summary-cache.json")
        self.experience_board.write(root / "experience-board.json")
        traces = root / "selection-traces.jsonl"
        traces.write_text("".join(json.dumps(trace.to_dict(), ensure_ascii=False, sort_keys=True) + "\n" for trace in self.selection_traces), encoding="utf-8")
        retrievals = root / "retrieval-bundles.jsonl"
        retrievals.write_text(
            "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in self.retrieval_records),
            encoding="utf-8",
        )
        (root / "budget-ledger.json").write_text(json.dumps(self.budget.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (root / "summary-cache-stats.json").write_text(
            json.dumps(
                {
                    "schema": "evomind.experience_mcgs.summary_cache_stats.v1",
                    "hits": self.summary_cache_hits,
                    "misses": self.summary_cache_misses,
                    "model": self.summarizer_model,
                    "prompt_template_hash": self.summary_prompt_template_hash,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    # ── helpers on B's graph ────────────────────────────────────────────────
    def _depth(self, graph, node) -> int:
        d, cur = 0, node
        seen = set()
        while cur.parent_id and cur.parent_id in graph.nodes and cur.exp_id not in seen:
            seen.add(cur.exp_id)
            cur = graph.nodes[cur.parent_id]
            d += 1
        return d

    def _branch(self, exp_id: str) -> str:
        return self.branch_of.get(exp_id, "branch_0")

    def _ancestors_in_branch(self, graph, node, k: int = 3) -> list[str]:
        out, cur, bid = [], node, self._branch(node.exp_id)
        seen = {node.exp_id}   # cycle-guard, consistent with _depth: never loop forever
        while cur.parent_id and cur.parent_id in graph.nodes and len(out) < k:
            if cur.parent_id in seen:
                break
            seen.add(cur.parent_id)
            cur = graph.nodes[cur.parent_id]
            if self._branch(cur.exp_id) == bid:
                out.append(cur.exp_id)
        return out

    def _top_exp_ids(self, graph, *, n: int, exclude_branch: Optional[str] = None) -> list[str]:
        scored = [(self._reward(graph, nd), nd.exp_id) for nd in graph.nodes.values()
                  if graph._node_score(nd) is not None
                  and (exclude_branch is None or self._branch(nd.exp_id) != exclude_branch)]
        scored.sort(reverse=True)
        return [eid for _, eid in scored[:n]]

    def _coding_mode(self, graph, node, *, phase: str) -> str:
        depth = self._depth(graph, node)
        branch_age = sum(1 for e in self.branch_of.values() if e == self._branch(node.exp_id))
        if phase == "exploration" and depth <= 1:
            return "Base"
        if branch_age >= 4 and phase != "exploration":
            return "Diff"
        if depth <= 2:
            return "Stepwise"
        return "Diff"

    def _plan_expansion(self, graph, node, *, phase: str) -> ExpansionPlan:
        bid = self._branch(node.exp_id)
        n_branches = len(set(self.branch_of.values())) or 1
        # 1) aggregation: global stall + enough branches to fuse
        if self.global_stagnation >= self.global_stagnation_patience and n_branches >= 2:
            refs = self._top_exp_ids(graph, n=4)
            new_bid = self._new_branch() if n_branches < self.max_branches else bid
            return ExpansionPlan(node.exp_id, "aggregation", "Base", refs,
                                 branch_id=new_bid, phase=phase)
        # 2) this branch stalled. Two sub-cases:
        #    (a) other branches exist -> borrow their ideas (cross_branch).
        #    (b) single-branch world -> DIVERSIFY: open a fresh branch from the global
        #        best. Without (b) the search can never leave branch_0, so cross_branch
        #        AND aggregation are unreachable dead code (the multi-branch brain needs
        #        a second branch to bootstrap, but only these paths create one).
        if self.branch_stagnation.get(bid, 0) >= self.branch_stagnation_patience:
            refs = self._top_exp_ids(graph, n=3, exclude_branch=bid)
            if refs:
                new_bid = self._new_branch() if n_branches < self.max_branches else bid
                return ExpansionPlan(node.exp_id, "cross_branch", "Stepwise", refs,
                                     branch_id=new_bid, phase=phase)
            if n_branches < self.max_branches:
                seed = (self._top_exp_ids(graph, n=1) or [node.exp_id])[0]
                return ExpansionPlan(seed, "primary", "Base", [],
                                     branch_id=self._new_branch(), phase=phase)
        # 3) intra-branch: refine using this branch's own history (post-exploration)
        if phase != "exploration":
            hist = self._ancestors_in_branch(graph, node, k=3)
            if hist:
                mode = self._coding_mode(graph, node, phase=phase)
                return ExpansionPlan(node.exp_id, "intra_branch", mode, hist,
                                     branch_id=bid, phase=phase)
        # 4) primary: straightforward improvement of the selected node
        mode = self._coding_mode(graph, node, phase=phase)
        return ExpansionPlan(node.exp_id, "primary", mode, [], branch_id=bid, phase=phase)

    def _new_branch(self) -> str:
        self._branch_counter += 1
        return f"branch_{self._branch_counter}"

    # ── backpropagation: the increment A never did + stagnation bookkeeping ──
    def register_child(self, plan: ExpansionPlan, child_exp_id: str) -> None:
        """Record which branch the newly-created child belongs to."""
        self.branch_of[child_exp_id] = plan.branch_id or self._branch(plan.node_exp_id)

    def backpropagate(self, graph, exp_id: str, *, improved: bool) -> None:
        """Increment visit counts up the ancestor chain and update stagnation.

        ``improved`` is whether this node improved the global best (the caller
        already knows this from the promotion decision, so we don't recompute it).
        """
        # visit increment along the path to root (this is what makes UCT live)
        cur = graph.nodes.get(exp_id)
        seen = set()
        while cur is not None and cur.exp_id not in seen:
            seen.add(cur.exp_id)
            self.visits[cur.exp_id] = self.visits.get(cur.exp_id, 0) + 1
            cur = graph.nodes.get(cur.parent_id) if cur.parent_id else None
        # stagnation bookkeeping
        bid = self._branch(exp_id)
        if improved:
            self.global_stagnation = 0
            self.branch_stagnation[bid] = 0
        else:
            self.global_stagnation += 1
            self.branch_stagnation[bid] = self.branch_stagnation.get(bid, 0) + 1



