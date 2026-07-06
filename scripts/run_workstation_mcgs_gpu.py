"""
MCGS GPU Dispatcher — runs the 4-layer self-evolving architecture,
dispatching ALL training to GPU cluster via hpc_connect.
Designed to be called from the workstation API or directly.
"""
import sys, os, json, time, uuid, argparse
from pathlib import Path

ROOT = Path(r'D:\桌面\codex\科研港科技')
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from hpc_connect import hpc_exec, JOBS
from research_agent_workstation.server.strategy.mlevolve_harness_bridge import MLEvolveHarnessBridge

# Configuration
GPU_SERVER = "87571"  # S2: 4xA800 primary
TRAINING_SCRIPT = "/hpc2hdd/home/aimslab/gpu_train_v3.py"
RESULTS_DIR = "/hpc2hdd/home/aimslab/results"

# MCGS exploration strategies
BRANCH_TYPES = [
    "feature_engineering",   # Add/improve features
    "model_diversity",       # Try different model params
    "ensemble_blend",        # Blend multiple models
    "regularization",        # Increase regularization
    None,                    # Baseline run
]

# Task configuration — from existing configs
TASK_CONFIGS = {
    "titanic": {"metric": "accuracy", "direction": "maximize", "bronze": 0.794},
    "spaceship_titanic": {"metric": "accuracy", "direction": "maximize", "bronze": 0.795},
    "house_prices": {"metric": "rmsle", "direction": "minimize", "bronze": 0.140},
    "digit_recognizer": {"metric": "accuracy", "direction": "maximize", "bronze": 0.986},
    "bike_sharing_demand": {"metric": "rmsle", "direction": "minimize", "bronze": 0.480},
    "porto_seguro": {"metric": "normalized_gini", "direction": "maximize", "bronze": 0.285},
    "store_sales": {"metric": "rmsle", "direction": "minimize", "bronze": 0.500},
    "ps3e1": {"metric": "rmse", "direction": "minimize", "bronze": 0.600},
    "ps3e7": {"metric": "accuracy", "direction": "maximize", "bronze": 0.800},
    "ps4e1": {"metric": "accuracy", "direction": "maximize", "bronze": 0.750},
    "ps4e2": {"metric": "accuracy", "direction": "maximize", "bronze": 0.750},
    "ps4e3": {"metric": "accuracy", "direction": "maximize", "bronze": 0.700},
    "ps4e6": {"metric": "accuracy", "direction": "maximize", "bronze": 0.750},
    "ps4e7": {"metric": "accuracy", "direction": "maximize", "bronze": 0.600},
    "ps6e2": {"metric": "accuracy", "direction": "maximize", "bronze": 0.800},
    "ps6e3": {"metric": "accuracy", "direction": "maximize", "bronze": 0.800},
    "ps6e6": {"metric": "accuracy", "direction": "maximize", "bronze": 0.400},
    "tps_feb2022": {"metric": "accuracy", "direction": "maximize", "bronze": 0.800},
    "tps_may2022": {"metric": "accuracy", "direction": "maximize", "bronze": 0.750},
}


def run_gpu_training(task_id, gpu_device=0, n_folds=5, fast=False):
    """Dispatch a single training run to the GPU cluster."""
    cmd = f"cd /hpc2hdd/home/aimslab && python3 {TRAINING_SCRIPT} {task_id} --gpu-device {gpu_device} --n-folds {n_folds}"
    if fast:
        cmd += " --fast"

    out, err = hpc_exec(GPU_SERVER, cmd, timeout=600)

    # Parse result
    result_path = f"{RESULTS_DIR}/v3_result_{task_id}.json"
    out2, _ = hpc_exec(GPU_SERVER, f"cat {result_path} 2>/dev/null || echo NOT_FOUND", timeout=10)

    if "NOT_FOUND" in out2:
        return None

    try:
        import json
        data = json.loads(out2)
        return data.get("oof_score")
    except:
        return None


def run_mcgs_gpu(task_id, budget_nodes=8, gpu_device=0):
    """Run MCGS self-evolving loop with GPU dispatch."""
    config = TASK_CONFIGS.get(task_id)
    if not config:
        print(f"Unknown task: {task_id}")
        return

    metric = config["metric"]
    direction = config["direction"]
    bronze = config["bronze"]

    print(f"\n{'='*60}")
    print(f"MCGS-GPU: {task_id} | {metric} | {direction} | bronze={bronze}")
    print(f"{'='*60}")

    # Initialize the 4-layer bridge
    bridge = MLEvolveHarnessBridge(
        task_id=task_id,
        metric=metric,
        metric_direction=direction,
        total_budget_hours=2.0,
        workspace_root=ROOT,
    )

    is_max = direction == "maximize"
    best_score = -float("inf") if is_max else float("inf")
    results = []

    for step in range(budget_nodes):
        t0 = time.time()
        print(f"\n--- Step {step+1}/{budget_nodes} ---")

        # 1. Get search context from memory
        ctx = bridge.get_search_context(f"Improve {metric} for {task_id}")
        kb_preview = ctx.get("knowledge_base", "")[:150]
        if kb_preview:
            print(f"  KB: {kb_preview}")

        # 2. Propose search idea via XCIENTIST contract
        strat_idx = step % len(BRANCH_TYPES)
        branch_type = BRANCH_TYPES[strat_idx]

        idea = bridge.propose_search_idea(
            title=f"MCGS-GPU step {step+1}: {branch_type or 'baseline'}",
            hypothesis=f"Strategy '{branch_type or 'baseline'}' will improve {metric} beyond {best_score:.6f}",
            mechanism=f"GPU CatBoost training with {branch_type or 'baseline'} strategy",
            expected_effect=f"{metric} improvement",
            risk_level="low" if step == 0 else "medium",
            component_changes=[branch_type] if branch_type else ["baseline"],
        )

        # 3. Create validation contract
        contract = bridge.create_experiment_contract(
            idea=idea,
            baseline_score=best_score,
            validation_plan="5-fold CV OOF on GPU cluster",
        )

        # 4. DISPATCH TO GPU (instead of local subprocess)
        score = run_gpu_training(task_id, gpu_device=gpu_device, n_folds=5)
        print(f"  GPU result: OOF={score}")

        if score is None:
            print("  Training failed, skipping audit")
            continue

        # 5. Gate check
        if is_max:
            gate_pass = score >= bronze
            improved = score > best_score
        else:
            gate_pass = score <= bronze
            improved = score < best_score

        # 6. Record and audit
        record, audit = bridge.record_and_audit(
            idea=idea, contract=contract,
            baseline_value=best_score,
            experiment_value=score,
            evidence_artifacts=[f"{RESULTS_DIR}/v3_result_{task_id}.json"],
        )

        audit_status = "PASS" if audit.audit_passed else "DRIFT"
        gate_status = "GATE_PASS" if gate_pass else "GATE_FAIL"
        improvement = "BETTER" if improved else ("SAME" if score == best_score else "WORSE")

        print(f"  Audit: {audit_status} | Gate: {gate_status} | {improvement} | Score: {score:.4f}")

        # 7. Update best
        if improved:
            best_score = score
            # Accept if audit passes
            if bridge.accept_if_valid(idea, record, audit):
                print(f"  ✅ New best: {score:.4f}")

        # 8. Check stagnation → trigger Island Model
        next_action = bridge.get_next_action()
        if next_action.get("should_trigger_island_model"):
            print(f"  🌊 Stagnation detected! Triggering Island Model...")
            island = bridge.trigger_island_exploration()
            print(f"  Islands: {island.get('active_islands', 0)} active")

        results.append({
            "step": step,
            "score": score,
            "best": best_score,
            "improved": improved,
            "gate_pass": gate_pass,
            "elapsed": time.time() - t0,
        })

        elapsed = time.time() - t0
        print(f"  [{elapsed:.0f}s]")

    # Final report
    print(f"\n{'='*60}")
    print(f"MCGS-GPU COMPLETE: {task_id}")
    print(f"  Best OOF: {best_score:.4f} (bronze: {bronze})")
    gate_ok = (is_max and best_score >= bronze) or (not is_max and best_score <= bronze)
    print(f"  Bronze: {'YES' if gate_ok else 'NO'}")
    print(f"  Improvement: {best_score - results[0]['score']:.4f}" if len(results) > 1 else "")

    # Save manifest
    manifest_path = ROOT / "workspace" / f"mcgs_gpu_{task_id}_{int(time.time())}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "task_id": task_id, "metric": metric, "bronze": bronze,
        "best_score": best_score, "bronze_achieved": gate_ok,
        "steps": len(results), "results": results,
    }, open(manifest_path, 'w'), indent=2)

    return best_score, gate_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id")
    parser.add_argument("--budget-nodes", type=int, default=8)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--all-gate-passed", action="store_true",
                       help="Run MCGS for ALL gate-passed tasks to improve scores")
    args = parser.parse_args()

    if args.all_gate_passed:
        # Run for all tasks that are gate-passed but not yet bronze
        tasks = list(TASK_CONFIGS.keys())
        print(f"Running MCGS-GPU for {len(tasks)} tasks...")
        results = {}
        for i, task in enumerate(tasks):
            print(f"\n[{i+1}/{len(tasks)}] {task}")
            try:
                score, ok = run_mcgs_gpu(task, args.budget_nodes, args.gpu_device)
                results[task] = {"best": score, "bronze": ok}
            except Exception as e:
                print(f"  ERROR: {e}")
                results[task] = {"error": str(e)}

        # Summary
        bronze_count = sum(1 for r in results.values() if r.get("bronze"))
        print(f"\n{'='*60}")
        print(f"ALL DONE: {bronze_count}/{len(results)} bronze achieved")
        for t, r in sorted(results.items()):
            print(f"  {t}: best={r.get('best', 'ERR')} bronze={r.get('bronze', False)}")
    else:
        score, ok = run_mcgs_gpu(args.task_id, args.budget_nodes, args.gpu_device)
        print(f"\nFinal: {args.task_id} -> {score} (bronze={ok})")


if __name__ == "__main__":
    main()
