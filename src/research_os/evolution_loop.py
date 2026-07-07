"""The evolution loop: result-driven closed loop that ties the pieces together.

This replaces the old fixed ``if n_prev == 0/1/2`` ladder with a real loop:

    seed baseline -> run -> score -> promote/hold -> remember lesson ->
    propose next variation (informed by history + memory) -> repeat.

Design:
  * The *runner* is pluggable. A ``LocalSubprocessRunner`` executes generated
    code locally; a GPU runner (SSH) can implement the same ``Runner`` protocol,
    which is how the local and GPU tracks get unified behind one loop.
  * Promotion uses the existing ``search_graph.SearchGraph`` invariant.
  * Lessons are written to the existing ``RetrospectiveMemoryStore``.
  * Stagnation/failure switches the code-generation mode (Stepwise <-> Diff)
    via the existing ``mlevolve_controller.choose_code_generation_mode``.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from .mlevolve_controller import choose_code_generation_mode
from . import events as ev
from .claim_audit import audit_claim
from .retrospective_memory import MemoryRecord, RetrospectiveMemoryStore
from .search_graph import ExperimentNode, SearchGraph
from .validation_contract import check_required_artifacts, create_contract, evaluate_acceptance
from .variation_generator import TaskContext, VariationGenerator, VariationProposal


@dataclass
class RunResult:
    success: bool
    cv_score: Optional[float]
    stdout_tail: str = ""
    error: str = ""
    out_dir: str = ""
    artifacts: list[str] = field(default_factory=list)
    # Raw process exit status when known (remote kills carry no traceback, so the
    # code is the ONLY signal that a run was timed-out=124 / OOM-killed=137). None
    # means "not a subprocess outcome" (e.g. an infra exception before launch).
    exit_code: Optional[int] = None


class Runner(Protocol):
    """Executes a candidate script and returns its CV score / error."""

    def run(self, code: str, *, data_dir: str, out_dir: str, exp_id: str) -> RunResult: ...


def _parse_cv_score(text: str) -> Optional[float]:
    score = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("CV_SCORE="):
            try:
                score = float(line.split("=", 1)[1].strip())
            except ValueError:
                continue
    return score


class LocalSubprocessRunner:
    """Runs generated code as a local subprocess (fast, free, no GPU)."""

    def __init__(self, workdir: str | Path, *, timeout: int = 900, python_exe: Optional[str] = None) -> None:
        self.workdir = Path(workdir)
        self.timeout = timeout
        self.python_exe = python_exe or sys.executable

    def run(self, code: str, *, data_dir: str, out_dir: str, exp_id: str) -> RunResult:
        script_dir = self.workdir / exp_id
        script_dir.mkdir(parents=True, exist_ok=True)
        script_path = script_dir / "solution.py"
        script_path.write_text(code, encoding="utf-8")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        try:
            proc = subprocess.run(
                [self.python_exe, str(script_path), "--data-dir", data_dir, "--out-dir", out_dir],
                capture_output=True, text=True, timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return RunResult(False, None, error=f"timeout after {self.timeout}s", out_dir=out_dir)
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        score = _parse_cv_score(proc.stdout or "")
        metrics_path = Path(out_dir) / "metrics.json"
        if score is None and metrics_path.exists():
            try:
                score = float(json.loads(metrics_path.read_text(encoding="utf-8")).get("cv_score"))
            except (ValueError, TypeError, json.JSONDecodeError):
                score = None
        artifacts = [str(p) for p in Path(out_dir).glob("*") if p.is_file()]
        if proc.returncode != 0 or score is None:
            return RunResult(
                False, score, stdout_tail=combined[-1500:],
                error=(proc.stderr or "no CV_SCORE emitted")[-1500:],
                out_dir=out_dir, artifacts=artifacts,
            )
        return RunResult(True, score, stdout_tail=combined[-800:], out_dir=out_dir, artifacts=artifacts)


@dataclass
class EvolutionConfig:
    max_iterations: int = 6
    min_delta: float = 1e-4
    stagnation_patience: int = 2  # consecutive non-improving iters before Diff mode
    required_artifacts: list[str] = field(default_factory=lambda: ["metrics.json", "submission.csv"])
    transient_retries: int = 3  # re-run the SAME proposal on transient infra (SSH/SOCKS) errors; 3 suits long large-data runs where a flaky tunnel is likely to drop mid-training


@dataclass
class IterationRecord:
    exp_id: str
    mode: str
    success: bool
    cv_score: Optional[float]
    promoted: bool
    note: str
    provider: str = ""
    model: str = ""


class EvolutionLoop:
    """Result-driven evolutionary search over solutions for a single task."""

    def __init__(
        self,
        context: TaskContext,
        *,
        data_dir: str,
        work_dir: str | Path,
        runner: Runner,
        generator: Optional[VariationGenerator] = None,
        memory: Optional[RetrospectiveMemoryStore] = None,
        config: Optional[EvolutionConfig] = None,
        selector: Optional[Any] = None,
        on_event: Optional[Callable[[dict], None]] = None,
        run_meta: Optional[dict[str, Any]] = None,
    ) -> None:
        self.context = context
        self.data_dir = data_dir
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.runner = runner
        self.generator = generator or VariationGenerator()
        self.memory = memory or RetrospectiveMemoryStore(self.work_dir / "retrospective_memory.json")
        self.config = config or EvolutionConfig()
        # Optional MCGS selection brain. When None (default), the loop keeps its
        # simple linear best-so-far behavior, so all existing callers are unchanged.
        self.selector = selector
        # Optional research-event stream. When None (default), emission is a no-op,
        # so the engine's behavior is byte-for-byte unchanged for existing callers
        # and every test. When provided (by `xsci run`), each research-cycle joint
        # emits one JSON-safe event to drive live streaming / watch / dashboard.
        self._on_event = on_event
        self._run_meta = run_meta or {}
        self._event_seq = 0
        self.graph = SearchGraph(
            task_id=context.task_name, root_exp_id="EXP000",
            metric_name="cv_score", metric_direction=context.metric_direction,
        )
        self.cv_history: list[dict[str, Any]] = []
        self.iterations: list[IterationRecord] = []
        self.best_code: Optional[str] = None
        self.best_exp_id: Optional[str] = None
        self.code_by_exp: dict[str, str] = {}     # exp_id -> code (for MCGS node expansion)
        self.last_code: Optional[str] = None      # last attempted code (even if it failed)
        self.last_exp_id: Optional[str] = None
        self.last_error: str = ""                  # cleaned full error of last failed run (for Diff feedback)

    def _emit(self, event_type: str, **fields: Any) -> None:
        """Emit one research event to the optional stream. Never raises.

        A monotonic ``seq`` and an ISO timestamp are attached by the engine so
        the stream is totally ordered and every consumer (run/watch/dashboard)
        agrees on ordering. A failing sink must never crash a research run, so
        the whole call is defensive -- emission is observability, not logic.
        """
        if self._on_event is None:
            return
        self._event_seq += 1
        event = {
            "seq": self._event_seq,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "type": event_type,
            **fields,
        }
        try:
            self._on_event(event)
        except Exception:  # noqa: BLE001 - observability must not break the loop
            pass

    def _lessons(self) -> list[dict[str, Any]]:
        from dataclasses import asdict
        return [asdict(r) for r in self.memory.retrieve_by_task_type(self.context.task_type)]

    def _build_references(self, exp_ids: list[str]) -> list[Any]:
        """Turn selector-chosen node ids into RefSolution objects for the prompt."""
        from .variation_generator import RefSolution
        refs: list[Any] = []
        for eid in exp_ids:
            node = self.graph.nodes.get(eid)
            if node is None:
                continue
            branch = ""
            if self.selector is not None:
                branch = getattr(self.selector, "branch_of", {}).get(eid, "")
            refs.append(RefSolution(
                exp_id=eid, score=node.cv_score, branch_id=branch,
                code=self.code_by_exp.get(eid, ""),
                note=(node.promotion_reason or node.implementation_summary or ""),
            ))
        return refs

    def _record_memory(self, proposal: VariationProposal, result: RunResult, promoted: bool, delta: Optional[float]) -> None:
        self.memory.add_memory(MemoryRecord(
            memory_id=f"{self.context.task_name}:{proposal.exp_id}",
            task_type=self.context.task_type,
            dataset_profile={"modality": self.context.modality, "n_train": self.context.n_train},
            method=f"{proposal.code_generation_mode}:{','.join(proposal.applied_strategies) or 'baseline'}",
            what_worked=(proposal.hypothesis if promoted else ""),
            what_failed=("" if result.success else _salient_error(result.error, max_chars=300)),
            metric_delta=delta,
            reusable_strategy=(",".join(proposal.applied_strategies) if promoted else ""),
            failure_pattern=("" if result.success else _classify_failure(result.error)),
            linked_exp_ids=[proposal.exp_id],
        ))

    def _decide_mode(self, iteration: int, consecutive_no_improve: int, last_failed: bool) -> str:
        return choose_code_generation_mode(
            has_parent=self.best_code is not None,
            branch_stagnant=consecutive_no_improve >= self.config.stagnation_patience,
            global_stagnant=self.graph.detect_global_stagnation(min_delta=self.config.min_delta),
            failure_count=2 if last_failed else 0,
        )

    def run(self, *, strategies: Optional[list[str]] = None) -> dict[str, Any]:
        consecutive_no_improve = 0
        last_failed = False
        self._emit(
            ev.RUN_BEGIN, task=self.context.task_name, metric=self.context.metric,
            metric_direction=self.context.metric_direction,
            max_iterations=self.config.max_iterations,
            mcgs=self.selector is not None, strategies=list(strategies or []),
            **self._run_meta,
        )
        for iteration in range(self.config.max_iterations):
            exp_id = f"EXP{iteration:03d}"
            self._emit(ev.ITER_BEGIN, iteration=iteration, exp_id=exp_id)
            mode = self._decide_mode(iteration, consecutive_no_improve, last_failed)
            # Recovery case: no promoted best yet, but the last attempt failed. Fix that
            # failed code via Diff (debug our way to a first working solution) instead of
            # regenerating Base from scratch and repeating the same mistake.
            base_code = self.best_code
            base_parent = self.best_exp_id
            if self.best_code is None and last_failed and self.last_code:
                mode = "Diff"
                base_code = self.last_code
                base_parent = self.last_exp_id
            # MCGS selection brain (opt-in). It decides which node to expand, the
            # expansion type, the coding mode, and which sibling solutions to
            # reference. Any failure degrades gracefully to the linear logic above.
            expansion_type = "primary"
            reference_solutions: list[Any] = []
            tree_parent = base_parent
            if self.selector is not None and self.graph.nodes:
                try:
                    plan = self.selector.select(self.graph, step=iteration)
                    mode = plan.coding_mode
                    expansion_type = plan.expansion_type
                    tree_parent = plan.node_exp_id
                    if plan.node_exp_id in self.code_by_exp:
                        base_code = self.code_by_exp[plan.node_exp_id]
                    reference_solutions = self._build_references(plan.reference_exp_ids)
                    self._pending_plan = plan
                    self._emit(
                        ev.SELECT, exp_id=exp_id, node_exp_id=plan.node_exp_id,
                        expansion_type=expansion_type, coding_mode=mode,
                        reference_exp_ids=list(plan.reference_exp_ids or []),
                        phase=getattr(plan, "phase", ""),
                    )
                except Exception as exc:  # selector must never crash the loop
                    self._append_history(exp_id, mode, None, False,
                                         f"selector_fallback: {type(exc).__name__}")
                    expansion_type, reference_solutions, tree_parent = "primary", [], base_parent
                    self._pending_plan = None
            else:
                self._pending_plan = None
            # feed the last error back so Diff-mode proposals can fix it. Prefer the
            # cleaned full error (real traceback) over the 120-char summary note.
            notes = self.context.extra_notes
            if last_failed and (self.last_error or self.cv_history):
                err = self.last_error or str(self.cv_history[-1].get("note", ""))
                notes = (notes + "\nLAST RUN FAILED. Fix THIS error:\n" + err)[:2400]
            ctx = self.context
            if notes != self.context.extra_notes:
                ctx = _with_notes(self.context, notes)
            try:
                proposal = self.generator.propose(
                    ctx, exp_id=exp_id, mode=mode, cv_history=self.cv_history,
                    lessons=self._lessons(), strategies=strategies, best_code=base_code,
                    parent_exp_id=tree_parent, expansion_type=expansion_type,
                    reference_solutions=reference_solutions,
                )
            except Exception as exc:  # generation failure is itself a recorded outcome
                self._append_history(exp_id, mode, None, False, f"generation_failed: {type(exc).__name__}")
                self._emit(ev.SCORE, exp_id=exp_id, success=False, cv_score=None,
                           error=f"generation_failed: {type(exc).__name__}")
                self._emit(ev.ITER_END, exp_id=exp_id, mode=mode, success=False,
                           cv_score=None, promoted=False)
                last_failed = True
                consecutive_no_improve += 1
                continue

            self.last_code = proposal.code
            self.last_exp_id = exp_id
            self.code_by_exp[exp_id] = proposal.code   # so MCGS can expand this node later
            self._emit(
                ev.PROPOSE, exp_id=exp_id, mode=mode, expansion_type=expansion_type,
                parent_exp_id=tree_parent, hypothesis=proposal.hypothesis,
                changes_summary=proposal.changes_summary,
                strategies=list(proposal.applied_strategies or []),
                provider=proposal.provider, model=proposal.model,
            )
            out_dir = str(self.work_dir / exp_id / "out")
            self._emit(ev.EXEC_BEGIN, exp_id=exp_id, runner=type(self.runner).__name__)
            # Transient infra (SSH dropped, SOCKS/connect timeout) is NOT a bad
            # proposal: re-run the SAME code before giving up, so we don't waste an
            # LLM proposal on a network blip and needlessly flip into Diff mode.
            result = None
            for attempt in range(self.config.transient_retries + 1):
                try:
                    result = self.runner.run(proposal.code, data_dir=self.data_dir, out_dir=out_dir, exp_id=exp_id)
                    break
                except Exception as exc:  # noqa: BLE001 - runner backends vary
                    err = f"runner_exception: {type(exc).__name__}: {str(exc)[:200]}"
                    if _is_transient_infra(exc) and attempt < self.config.transient_retries:
                        continue
                    result = RunResult(False, None, error=err, out_dir=out_dir)
                    break
            self._emit(ev.SCORE, exp_id=exp_id, success=result.success,
                       cv_score=result.cv_score, exit_code=result.exit_code)
            promoted, delta = self._integrate(proposal, result, tree_parent=tree_parent)
            best_node = self.graph.nodes.get(self.best_exp_id) if self.best_exp_id else None
            self._emit(
                ev.PROMOTE, exp_id=exp_id, promoted=promoted, delta=delta,
                reason=(self.cv_history[-1].get("note", "") if self.cv_history else ""),
                best_exp_id=self.best_exp_id,
                best_cv_score=(best_node.cv_score if best_node else None),
            )
            # Feed the outcome back to the MCGS brain: increment visit counts up the
            # ancestor chain (what makes UCT live) and update stagnation counters.
            if self.selector is not None and getattr(self, "_pending_plan", None) is not None:
                try:
                    self.selector.register_child(self._pending_plan, exp_id)
                    self.selector.backpropagate(self.graph, exp_id, improved=promoted)
                except Exception:  # backprop must never crash the loop
                    pass
            last_failed = not result.success
            if not result.success:
                # Persist the full, noise-stripped error so failures are debuggable
                # after the fact (the 120-char note is not enough to diagnose a crash).
                err_text = _clean_error_for_feedback(result.error or result.stdout_tail, max_chars=8000)
                self.last_error = _clean_error_for_feedback(result.error or result.stdout_tail, max_chars=1500)
                try:
                    (self.work_dir / exp_id).mkdir(parents=True, exist_ok=True)
                    (self.work_dir / exp_id / "run_error.txt").write_text(err_text, encoding="utf-8")
                except OSError:
                    pass
                self._emit(ev.REPAIR, exp_id=exp_id,
                           failure_pattern=_classify_failure(result.error),
                           error=_salient_error(result.error, max_chars=300))
            consecutive_no_improve = 0 if promoted else consecutive_no_improve + 1
            self._record_memory(proposal, result, promoted, delta)
            self._emit(
                ev.LESSON, exp_id=exp_id,
                failure_pattern=("" if result.success else _classify_failure(result.error)),
                reusable_strategy=(",".join(proposal.applied_strategies) if promoted else ""),
                metric_delta=delta,
            )
            self.iterations.append(IterationRecord(
                exp_id=exp_id, mode=mode, success=result.success, cv_score=result.cv_score,
                promoted=promoted,
                note=(_clean_error_for_feedback(result.error, max_chars=120) if not result.success else "ok"),
                provider=proposal.provider, model=proposal.model,
            ))
            self._emit(ev.ITER_END, exp_id=exp_id, mode=mode, success=result.success,
                       cv_score=result.cv_score, promoted=promoted)
            # Linear mode: global stagnation means "stop, we've plateaued". But when
            # the MCGS brain is driving, stagnation is the SIGNAL to fuse branches
            # (cross_branch / aggregation), not a reason to quit — so let it use its
            # full budget. Breaking here would kill the brain exactly when it should
            # be doing its most valuable work.
            if (self.selector is None
                    and self.graph.detect_global_stagnation(min_delta=self.config.min_delta)
                    and iteration >= 2):
                break
        summary = self.summary()
        self._emit(
            ev.RUN_END, task=self.context.task_name,
            best_exp_id=summary.get("best_exp_id"),
            best_cv_score=summary.get("best_cv_score"),
            n_iterations=summary.get("n_iterations"),
            n_promotions=summary.get("n_promotions"),
        )
        return summary

    def _integrate(self, proposal: VariationProposal, result: RunResult,
                   *, tree_parent: Optional[str] = None) -> tuple[bool, Optional[float]]:
        # tree_parent = the node we expanded FROM (MCGS topology). Falls back to the
        # global best so non-selector callers keep the original linear ancestry.
        parent_for_tree = tree_parent if tree_parent is not None else self.best_exp_id
        node = ExperimentNode(
            exp_id=proposal.exp_id, parent_id=parent_for_tree, branch_type=proposal.code_generation_mode,
            task_name=self.context.task_name, hypothesis=proposal.hypothesis,
            implementation_summary=proposal.changes_summary, code_path=f"{proposal.exp_id}/solution.py",
            artifacts=[{"path": Path(a).name} for a in result.artifacts],
            cv_score=result.cv_score, metric_name="cv_score", metric_direction=self.context.metric_direction,
            run_success=result.success,
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        self.graph.add_node(node)
        if parent_for_tree and parent_for_tree in self.graph.nodes:
            self.graph.add_edge(parent_for_tree, proposal.exp_id, proposal.code_generation_mode)
        decision = self.graph.decide_promotion(
            proposal.exp_id, parent_exp_id=self.best_exp_id, metric="cv_score",
            direction=self.context.metric_direction, min_delta=self.config.min_delta,
            required_artifacts=self.config.required_artifacts,
            run_success=result.success,
        )
        promoted = bool(decision.get("promoted"))
        if promoted:
            self.best_code = proposal.code
            self.best_exp_id = proposal.exp_id
        self._emit_audit(proposal, result, decision)
        self._append_history(
            proposal.exp_id, proposal.code_generation_mode, result.cv_score, promoted,
            (_clean_error_for_feedback(result.error, max_chars=120) if not result.success else decision.get("reason", "")),
        )
        return promoted, decision.get("promotion_delta")

    def _emit_audit(self, proposal: VariationProposal, result: RunResult, decision: dict[str, Any]) -> None:
        """Write library-sourced validation_contract.json + claim_audit.json.

        Uses research_os.{validation_contract,claim_audit} so the new engine and
        any other caller share ONE audit implementation (no inline .v1 fork).
        """
        exp_dir = self.work_dir / proposal.exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        artifact_names = [Path(a).name for a in result.artifacts]

        contract = create_contract(
            contract_id=f"{self.context.task_name}:{proposal.exp_id}:contract",
            exp_id=proposal.exp_id,
            claim=f"{proposal.code_generation_mode} candidate for {self.context.task_name}",
            hypothesis=proposal.hypothesis,
            implementation_requirement="Runnable script emitting CV_SCORE, submission.csv, metrics.json.",
            metric="cv_score",
            baseline_exp_id=proposal.parent_exp_id or "",
            acceptance_criteria={"cv_score": {"min" if self.context.metric_direction == "maximize" else "max": result.cv_score}}
            if result.cv_score is not None else {},
            ablation_plan=list(proposal.applied_strategies),
            conclusion_boundary="Local CV/proxy only; no official rank without a Kaggle response artifact.",
            required_artifacts=list(self.config.required_artifacts),
        )
        artifact_check = check_required_artifacts(contract, artifact_names)
        acceptance = evaluate_acceptance(contract, {"cv_score": result.cv_score} if result.cv_score is not None else {})
        contract_payload = {
            "schema": "academic_research_os.validation_contract.v1",
            "contract_id": contract.contract_id, "exp_id": proposal.exp_id,
            "task_id": self.context.task_name, "created_at": datetime.now().isoformat(timespec="seconds"),
            "hypothesis": contract.hypothesis, "metric": contract.metric,
            "required_artifacts": contract.required_artifacts,
            "artifact_check": artifact_check, "acceptance": acceptance,
            "conclusion_boundary": contract.conclusion_boundary,
            "run_success": result.success, "cv_score": result.cv_score,
        }
        (exp_dir / "validation_contract.json").write_text(
            json.dumps(contract_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        claim_text = (f"{self.context.task_name} {proposal.exp_id} reached cv_score={result.cv_score}"
                      if result.success else f"{proposal.exp_id} failed to produce a valid score")
        audit = audit_claim(
            claim_id=f"{self.context.task_name}:{proposal.exp_id}:claim",
            claim_text=claim_text, related_exp_ids=[proposal.exp_id],
            contract={"hypothesis": proposal.hypothesis, "conclusion_boundary": contract.conclusion_boundary},
            supporting_metrics={"cv_score": result.cv_score},
            required_ablations=list(proposal.applied_strategies),
            completed_ablations=list(proposal.applied_strategies) if result.success else [],
            evidence={"has_required_experiments": result.success,
                      "has_mechanistic_evidence": result.success,
                      "missing_evidence": [] if artifact_check["passed"] else artifact_check["missing_artifacts"]},
        )
        from dataclasses import asdict
        audit_payload = {"schema": "academic_research_os.claim_audit.v1",
                         "created_at": datetime.now().isoformat(timespec="seconds"),
                         "task_id": self.context.task_name, **asdict(audit)}
        (exp_dir / "claim_audit.json").write_text(
            json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8")


    def _append_history(self, exp_id: str, branch: str, cv: Optional[float], promoted: bool, note: str) -> None:
        self.cv_history.append({"exp_id": exp_id, "branch": branch, "cv_score": cv, "promoted": promoted, "note": note})

    def summary(self) -> dict[str, Any]:
        best_node = self.graph.nodes.get(self.best_exp_id) if self.best_exp_id else None
        return {
            "task": self.context.task_name,
            "best_exp_id": self.best_exp_id,
            "best_cv_score": best_node.cv_score if best_node else None,
            "metric": self.context.metric,
            "metric_direction": self.context.metric_direction,
            "iterations": [vars(it) for it in self.iterations],
            "promotion_history": self.graph.promotion_history,
            "n_iterations": len(self.iterations),
            "n_promotions": sum(1 for it in self.iterations if it.promoted),
        }


def _clean_error_for_feedback(error: str, *, max_chars: int = 1200) -> str:
    """Strip progress-bar / spinner noise so the real traceback survives.

    Remote tools (e.g. torch.hub downloads, tqdm) emit thousands of carriage-
    return progress frames. Those crowd out the actual error in the captured
    tail, which is why the Diff-recovery prompt was effectively blindfolded.
    Keep the last real content lines and drop the progress frames.
    """
    if not error:
        return ""
    # collapse CR-updated progress lines to their last frame, then drop frames
    text = error.replace("\r", "\n")
    keep: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        # progress frames look like "  32%|###   | 14.1M/44.7M [04:08<15:01, 35kB/s]"
        if ("%|" in s) or ("it/s]" in s) or ("kB/s]" in s) or ("MB/s]" in s):
            continue
        keep.append(s)
    cleaned = "\n".join(keep) if keep else text.strip()
    return cleaned[-max_chars:]


def _is_transient_infra(exc: Exception) -> bool:
    """True for network/SSH blips that warrant re-running the same proposal.

    These are infrastructure faults, not bad candidate code: an SSH session that
    dropped, a SOCKS/connect timeout, a reset connection. We retry the identical
    script rather than spending a fresh LLM proposal on a transient error.
    """
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    needles = ("sshexception", "session not active", "socks", "timed out", "timeout",
               "connection reset", "connection closed", "broken pipe", "eof")
    return any(n in name or n in msg for n in needles)


def _classify_failure(error: str) -> str:
    """Bucket a failure into a *reusable* pattern name.

    Ordered specific -> generic: the first matching, most-actionable bucket
    wins. Buckets exist so that ``retrieve_failures`` surfaces a lesson the
    proposer can actually act on ("estimator_api_misuse" -> read the message
    and fix the argument), not a shapeless "runtime_error". Add a bucket only
    when it maps to a distinct corrective action.
    """
    low = (error or "").lower()
    # Infrastructure/connection faults win FIRST. A dropped SSH session, SOCKS/
    # connect timeout, EOF before any stdout, or an auth failure is NOT a code
    # bug — it means the backend never ran the script. It must never be mislabeled
    # (e.g. as "segfault"), which would poison memory with a useless lesson and
    # send the proposer chasing a phantom native crash. These mirror the faults
    # _is_transient_infra() retries on. Kept ahead of the timeout bucket so a
    # "connection timed out" reads as infra, not a training wall-clock timeout.
    _infra_needles = (
        "runner_exception", "eoferror", "sshexception", "paramiko",
        "session not active", "socks", "connection reset", "connection closed",
        "connection refused", "connection aborted", "broken pipe",
        "ssh protocol banner", "authentication failed",
        "authentication did not complete", "no ssh backend",
        "gpu backend unreachable", "backend unreachable",
    )
    if any(n in low for n in _infra_needles):
        return "infra"
    # Exit-code diagnostics win next: a killed remote process has no traceback,
    # so the RUN_EXIT marker (set by the GPU runner) is the ground truth. Without
    # this, a timeout/OOM kill ending in "...before emitting CV_SCORE" would be
    # mis-bucketed as a contract_violation and the wrong lesson would be learned.
    if "timeout" in low or "timed out" in low or "run_exit=124" in low:
        return "timeout"
    if ("oom_or_killed" in low or "run_exit=137" in low
            or "out of memory" in low or "memoryerror" in low
            or "cuda error: out of memory" in low or "cublas" in low):
        return "oom"
    if "segfault" in low or "run_exit=139" in low or "sigsegv" in low:
        return "segfault"
    if "no cv_score" in low or "contract" in low:
        return "contract_violation"
    # Estimator/API misuse: wrong arg/solver/loss for the chosen model. This is
    # the single most reusable lesson (e.g. liblinear can't do multiclass).
    if (("solver" in low and ("support" in low or "does not" in low))
            or "onevsrest" in low
            or ("loss" in low and "not" in low and "support" in low)
            or "invalid parameter" in low
            or "got an unexpected keyword" in low
            or "unexpected keyword argument" in low):
        return "estimator_api_misuse"
    # Array/tensor shape or class-count mismatches between fit and data.
    if ("shape" in low or "dimension" in low or "size mismatch" in low
            or "n_classes" in low or "number of classes" in low
            or "inconsistent numbers of samples" in low):
        return "shape_mismatch"
    if ("did not converge" in low or "convergencewarning" in low
            or "failed to converge" in low or "nan" in low and "loss" in low):
        return "convergence"
    if ("modulenotfounderror" in low or "importerror" in low
            or "no module named" in low):
        return "import_error"
    if ("filenotfounderror" in low or "no such file" in low
            or "cannot find" in low):
        return "file_not_found"
    if ("dtype" in low or "could not convert" in low
            or "must be" in low or "isnan" in low or "infinity" in low):
        return "dtype_encoding"
    if ("keyerror" in low or "no column" in low or "not in index" in low):
        return "schema_mismatch"
    return "runtime_error"


# Exception/warning lines look like "ValueError: ...", "torch.cuda.OutOfMemoryError: ...".
_EXC_LINE = re.compile(r"^[\w\.]*(?:Error|Exception|Warning)\b\s*:?.*$")


def _salient_error(error: str, *, max_chars: int = 300) -> str:
    """Extract the actionable exception line, not a blind character tail.

    A raw ``[-300:]`` slice tends to cut mid-token (``ib\\site-packages\\...``)
    and bury the real ``ValueError:`` under file-path noise, so the stored
    lesson opens with junk. Here we keep the LAST exception/warning line (the
    one that actually names the fault) and, when present, the ``File ...``
    frame just above it for context, then trim from the left.
    """
    if not error:
        return ""
    lines = [ln.strip() for ln in error.replace("\r", "\n").splitlines() if ln.strip()]
    exc_idx = None
    for i, ln in enumerate(lines):
        if _EXC_LINE.match(ln):
            exc_idx = i
    if exc_idx is None:
        return _clean_error_for_feedback(error, max_chars=max_chars)
    exc = lines[exc_idx]
    # find the nearest "File ..., line N, in ..." frame above for locality
    ctx = ""
    for j in range(exc_idx - 1, -1, -1):
        if lines[j].startswith("File ") and ", line " in lines[j]:
            ctx = lines[j]
            break
    msg = (ctx + " -> " + exc) if ctx else exc
    return msg[-max_chars:]


def _with_notes(context: TaskContext, notes: str) -> TaskContext:
    from dataclasses import replace
    return replace(context, extra_notes=notes)


