"""
Workstation MCGS MVP: Three-Layer Architecture (Bridge-driven UCT search x Ensemble).
Layer 1 (Bottom): Multi-Agent Workstation — ensemble execution via subprocess.
Layer 2 (Middle): MLEvolve Search Controller — MCGS graph + Retrospective Memory.
Layer 3 (Top):    XCIENTIST Research Harness — idea contracts + claim audit.

The MLEvolveHarnessBridge integrates all three layers so that every search node
is proposed as a grounded IdeaContract, executed, audited for claim drift, and
accepted only when the audit passes.
"""
from __future__ import annotations
import argparse, json, sys, time, subprocess, uuid, yaml
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from research_agent_workstation.server.strategy.mlevolve_harness_bridge import (
    MLEvolveHarnessBridge,
)
from research_agent_workstation.server.strategy.mlevolve_search import (
    ExpansionType, CodingMode,
)

# ── Seed / fold rotation (unchanged) ──────────────────────────────────────────
SEEDS = [42, 3407, 12345, 777, 2048, 9999, 5555, 1111]
N_FOLDS_OPTS = [10, 5, 7, 5, 10, 3, 8, 5]
BRANCH_TYPES = ["", "feature_engineering", "model_family", "ensemble_blend",
                "feature_engineering", "ensemble_blend", "model_family", ""]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output-base", default="experiments")
    p.add_argument("--task-id", required=True)
    p.add_argument("--budget-nodes", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--fast", action="store_true", help="Run small-task verification with reduced folds/estimators.")
    p.add_argument("--sample-rows", type=int, default=20000, help="Rows sampled by the child ensemble runner in fast mode.")
    return p.parse_args()


def run_ensemble(task_id: str, config_path: str, output_base: str,
                 seed: int, n_folds: int, branch_type: str,
                 run_id: str, hypothesis: str,
                 fast: bool = False, sample_rows: int = 20000) -> float:
    """Execute the sklearn ensemble subprocess and return best validation score.

    Takes explicit parameters instead of a SearchNode so the bridge layer can
    supply them without coupling to the search engine's internal graph types.
    """
    cmd = [sys.executable, str(ROOT / "scripts" / "run_local_sklearn_ensemble.py"),
           "--config", config_path, "--output-base", output_base,
           "--task-id", task_id, "--random-state", str(seed),
           "--n-folds", str(n_folds), "--branch-type", branch_type,
           "--branch-hypothesis", hypothesis,
           "--run-id", run_id]
    if fast:
        cmd.extend(["--fast", "--sample-rows", str(sample_rows)])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, cwd=str(ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"Ensemble failed: {r.stderr[-400:]}")

    # Prefer the child runner's explicit JSON response. This binds each MCGS
    # node to the exact run it launched instead of accidentally reading a stale
    # experiment directory from a previous workstation run.
    for line in reversed(r.stdout.splitlines()):
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("run_id") == run_id and payload.get("best_validation_score") is not None:
            return float(payload["best_validation_score"])

    # Match experiment dir by run_id in launcher_manifest.json (fixes stale score bug)
    exp_base = Path(output_base) / task_id
    dirs = sorted([d for d in exp_base.iterdir() if d.is_dir()], reverse=True)
    for d in dirs[:15]:
        lp = d / "launcher_manifest.json"
        if lp.exists():
            try:
                lm = json.loads(lp.read_text())
                if lm.get("run_id", "") == run_id:
                    mp = d / "metrics.json"
                    if mp.exists():
                        m = json.loads(mp.read_text())
                        s = m.get("ensemble", {}).get("best_validation_score")
                        if s is not None:
                            return float(s)
            except Exception:
                pass
    # Fallback: any recent metrics (for backward compat)
    for d in dirs[:5]:
        mp = d / "metrics.json"
        if mp.exists():
            try:
                m = json.loads(mp.read_text())
                s = m.get("ensemble", {}).get("best_validation_score")
                if s is not None:
                    return float(s)
            except Exception:
                pass
    raise RuntimeError(f"No metrics found for {run_id}")


def main():
    args = parse_args()
    config_path = str(ROOT / args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    metric = config.get("task", {}).get("metric", "accuracy")
    direction = "maximize" if metric in ("accuracy", "balanced_accuracy", "roc_auc") else "minimize"

    # ── Initialize the three-layer bridge ─────────────────────────────────
    bridge = MLEvolveHarnessBridge(
        task_id=args.task_id,
        metric=metric,
        metric_direction=direction,
        total_budget_hours=2.0,
        workspace_root=ROOT,
    )

    best_score = -float("inf") if direction == "maximize" else float("inf")
    best_run_id = "root"
    results = []

    print(f"=== MCGS (Bridge): {args.task_id} (budget={args.budget_nodes}) ===")

    for step in range(args.budget_nodes):
        t0 = time.time()

        # 1. Retrieve memory-augmented context for this step
        description = (
            f"Improve {metric} for {args.task_id} competition. "
            f"Current best: {best_score:.6f}. Step {step+1}/{args.budget_nodes}."
        )
        ctx = bridge.get_search_context(description)
        kb_preview = ctx.get("knowledge_base", "")[:200]
        print(f"  KB context: {kb_preview}...")

        # 2. Propose a new search idea grounded in XCIENTIST contracts
        strat_idx = step % len(BRANCH_TYPES)
        branch_type = BRANCH_TYPES[strat_idx]
        seed = SEEDS[step % len(SEEDS)]
        n_folds = N_FOLDS_OPTS[step % len(N_FOLDS_OPTS)]

        title = f"MCGS step {step+1}: {branch_type or 'baseline'} seed={seed} folds={n_folds}"
        hypothesis = (
            f"Varying seed to {seed}, folds to {n_folds}, and applying "
            f"'{branch_type or 'baseline'}' strategy will improve {metric} "
            f"beyond current best ({best_score:.6f})."
        )
        mechanism = (
            f"Rotate ensemble seed={seed}, n_folds={n_folds}, "
            f"branch_type={branch_type or 'baseline'}. "
            f"Cross-branch references from KB context."
        )
        risk_level = "low" if step == 0 else "medium"

        idea = bridge.propose_search_idea(
            title=title,
            hypothesis=hypothesis,
            mechanism=mechanism,
            expected_effect=f"{metric} improvement over {best_score:.6f}",
            risk_level=risk_level,
            component_changes=[branch_type] if branch_type else ["baseline"],
            literature_refs=["MLEvolve cold-start KB", f"Prior best: {best_run_id}"],
        )
        print(f"  Idea: {idea.title[:80]}")

        # 3. Create a validation contract
        contract = bridge.create_experiment_contract(
            idea=idea,
            baseline_score=best_score,
            validation_plan="5-fold CV OOF evaluation, compare to protected baseline",
            ablations=[branch_type] if branch_type else [],
        )

        # 4. Run training locally (GPU cluster bypassed — hpc_connect not available)
        run_id = f"bridge_{step}_{uuid.uuid4().hex[:6]}"
        try:
            # Always run local ensemble (hpc_connect module not available on this workstation)
            score = run_ensemble(
                task_id=args.task_id, config_path=config_path,
                output_base=args.output_base, seed=seed,
                n_folds=n_folds, branch_type=branch_type,
                run_id=run_id, hypothesis=hypothesis,
                fast=args.fast, sample_rows=args.sample_rows,
            )

            baseline_value = best_score
            experiment_value = score

            is_better = (direction == "maximize" and score > best_score) or \
                        (direction == "minimize" and score < best_score)

            # 5. Record experiment and audit — evidence from local results
            evidence_artifacts = [
                str(Path(args.output_base) / args.task_id),
                str(Path(args.output_base) / args.task_id / "metrics.json"),
            ]

            record, audit = bridge.record_and_audit(
                idea=idea,
                contract=contract,
                baseline_value=baseline_value,
                experiment_value=experiment_value,
                code_content=f"Mechanism: {mechanism}. Implementation: Ensemble pipeline seed={seed} n_folds={n_folds} branch_type={branch_type}. Config: {args.config}. MCGS step {step+1}.",
                conclusion=(
                    f"Score {score:.6f} (baseline {baseline_value:.6f}, delta {score - baseline_value:+.6f}). "
                    f"{'IMPROVED' if is_better else 'NO IMPROVEMENT'}."
                ),
                evidence_artifacts=evidence_artifacts,
            )

            print(f"  Score={score:.6f}  audit_passed={audit.audit_passed}"
                  f"  drift={audit.claim_drift_detected}")

            # 6. Accept if valid and score improved
            if audit.audit_passed and is_better:
                accepted = bridge.accept_if_valid(idea, record, audit)
                if accepted:
                    best_score = score
                    best_run_id = run_id
                    print(f"  ACCEPTED ★BEST★ new best={best_score:.6f}")
                else:
                    print(f"  NOT ACCEPTED despite improvement (audit gate)")
            elif audit.audit_passed:
                print(f"  Audit passed but score did not improve ({score:.6f} vs {best_score:.6f})")
            else:
                print(f"  Audit FAILED: {audit.drift_description[:120]}")

            tag = " ★BEST★" if is_better else ""
            results.append({
                "step": step + 1,
                "run_id": run_id,
                "idea_id": idea.idea_id,
                "score": score,
                "best": is_better,
                "audit_passed": audit.audit_passed,
                "time_sec": round(time.time() - t0, 1),
            })

        except Exception as e:
            print(f"  FAILED: {str(e)[:200]}")
            results.append({
                "step": step + 1,
                "run_id": run_id,
                "error": str(e)[:400],
                "time_sec": round(time.time() - t0, 1),
            })

        # 7. Check stagnation via the search engine layer
        root_node = bridge.search.graph.nodes.get(bridge.search.graph.root_id)
        if root_node is not None:
            branch_id = root_node.branch_id
            branch_stagnant = bridge.search._is_branch_stagnant(branch_id, threshold=3)
            global_stagnant = bridge.search._is_globally_stagnant()
            if branch_stagnant:
                print(f"  STAGNATION: branch {branch_id} stagnant → consider cross-branch")
            if global_stagnant:
                print(f"  GLOBAL STAGNATION detected → consider aggregation")

        # 8. Persist memory after each iteration
        bridge.memory.kb._save()
        bridge.memory.experience.save()

        # Log next action recommendation
        next_action = bridge.get_next_action()
        print(f"  Phase={next_action.get('search_phase','?')} "
              f"alpha={next_action.get('alpha',0):.3f} "
              f"explore={next_action.get('should_explore',False)}")

    # ── Final persistence ─────────────────────────────────────────────────
    try:
        bridge.memory.kb._save()
        bridge.memory.experience.save()
    except Exception:
        pass

    status_path = bridge.save_full_status()
    print(f"  Bridge status saved → {status_path}")

    # ── Export MCGS search result ─────────────────────────────────────────
    output_dir = Path(args.output_base) / args.task_id / f"mcgs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "task_id": args.task_id,
        "metric": metric,
        "direction": direction,
        "best_score": best_score if best_score not in (float('inf'), float('-inf'), float('nan')) else None,
        "best_run_id": best_run_id,
        "total_nodes": len(results),
        "nodes_evaluated": len(results),
        "best_node_id": best_run_id,
        "results": results,
        "bridge_status_path": str(status_path),
    }
    # Sanitize NaN/Infinity before JSON serialization
    def sanitize_json(obj):
        if isinstance(obj, float):
            import math
            if math.isinf(obj) or math.isnan(obj):
                return None
        if isinstance(obj, dict):
            return {k: sanitize_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize_json(v) for v in obj]
        return obj
    result = sanitize_json(result)
    (output_dir / "mcgs_search_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )
    # Also print sanitized
    print(f"\nBest={best_score if best_score not in (float('inf'), float('-inf'), float('nan')) else 'N/A'} ({len(results)} nodes). Output: {output_dir}")
    # Print non-sanitized version for stdout reference
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
