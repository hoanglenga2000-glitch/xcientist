from __future__ import annotations

"""Workstation ensemble run launcher.

Dispatches an ensemble training run through the workstation orchestrator
using a registered ensemble template. All training, validation, and
submission audits go through the workstation agent workflow.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_agent_workstation.server.services import AgentOrchestrator
from research_agent_workstation.server.training import EnsembleTemplateRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ensemble training through workstation orchestrator.",
    )
    parser.add_argument(
        "--template", default="sklearn_rf_hgb_et_ensemble",
        choices=EnsembleTemplateRegistry.template_ids(),
        help="Ensemble template ID to dispatch.",
    )
    parser.add_argument(
        "--config",
        required=not ("--list-templates" in sys.argv),
        help="Task config YAML, e.g. configs/generated/playground_series_s6e6.yaml",
    )
    parser.add_argument(
        "--output-base", default="experiments",
        help="Output base directory.",
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random state for reproducibility.",
    )
    parser.add_argument(
        "--list-templates", action="store_true",
        help="List all registered ensemble templates and exit.",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Use reduced estimators and sample data for quick verification.",
    )
    parser.add_argument(
        "--sample-rows", type=int, default=20000,
        help="Number of rows to sample in fast mode.",
    )
    parser.add_argument(
        "--n-folds", type=int, default=None,
        help="Override the local training fold count.",
    )
    parser.add_argument(
        "--seeds", default="",
        help="Comma-separated random seeds to pass to the local training runner.",
    )
    parser.add_argument(
        "--timeout-seconds", type=int, default=3600,
        help="Maximum seconds allowed for the local training child process before writing timeout artifacts.",
    )
    parser.add_argument("--branch-id", default="", help="Search-controller branch id.")
    parser.add_argument("--branch-type", default="", help="Search-controller branch type.")
    parser.add_argument("--code-generation-mode", default="", help="Base / Stepwise / Diff mode selected by the Search Controller.")
    parser.add_argument("--branch-hypothesis", default="", help="Branch hypothesis for governance artifacts.")
    parser.add_argument("--cross-branch-references", default="", help="JSON encoded cross-branch reference list.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_templates:
        print("Registered Ensemble Templates:")
        print("=" * 60)
        for tid in EnsembleTemplateRegistry.template_ids():
            t = EnsembleTemplateRegistry.get(tid)
            print(f"  {tid}")
            print(f"    Name: {t.name}")
            print(f"    Models: {', '.join(t.model_family)}")
            print(f"    HPC Required: {t.hpc_required}")
            print(f"    Risk Level: {t.risk_level}")
            print(f"    Approved: {t.approved}")
            print()
        return

    orchestrator = AgentOrchestrator(ROOT)
    print(f"Starting ensemble run: template={args.template}, config={args.config}")
    print(f"Connector status: GPU={orchestrator.gpu.provider}, Kaggle={orchestrator.kaggle.provider}")

    summary = orchestrator.run_ensemble_closed_loop(
        config_path=ROOT / args.config,
        template_id=args.template,
        output_base=ROOT / args.output_base,
        random_state=args.random_state,
        fast_mode=args.fast,
        sample_rows=args.sample_rows,
        n_folds=args.n_folds,
        seeds=[int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()] if args.seeds else None,
        training_timeout_seconds=args.timeout_seconds,
        branch_metadata={
            "branch_id": args.branch_id,
            "branch_type": args.branch_type,
            "code_generation_mode": args.code_generation_mode,
            "hypothesis": args.branch_hypothesis,
            "cross_branch_references": json.loads(args.cross_branch_references) if args.cross_branch_references else [],
        },
    )

    run_info = summary.get("run", {})
    ensemble_metrics = summary.get("ensemble_metrics", {})
    best_score = ensemble_metrics.get("ensemble", {}).get("best_validation_score")
    reflection = summary.get("reflection", {})
    run_dir = run_info.get("output_dir")
    if run_dir:
        manifest_path = ROOT / run_dir / "launcher_manifest.json"
        try:
            launcher_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        except json.JSONDecodeError:
            launcher_manifest = {}
        launcher_manifest.update({
            "schema": launcher_manifest.get("schema", "academic_research_os.launcher_manifest.v1"),
            "top_level_launcher": "run_workstation_ensemble.py",
            "top_level_argv": sys.argv,
            "top_level_config": args.config,
            "top_level_template": args.template,
            "top_level_output_base": args.output_base,
            "top_level_random_state": args.random_state,
            "top_level_fast": bool(args.fast),
            "top_level_sample_rows": args.sample_rows if args.fast else None,
            "top_level_n_folds": args.n_folds,
            "top_level_seeds": args.seeds,
            "top_level_timeout_seconds": args.timeout_seconds,
            "top_level_branch_id": args.branch_id,
            "top_level_branch_type": args.branch_type,
            "top_level_code_generation_mode": args.code_generation_mode,
            "top_level_branch_hypothesis": args.branch_hypothesis,
            "top_level_cross_branch_references": json.loads(args.cross_branch_references) if args.cross_branch_references else [],
            "summary_run_id": run_info.get("id"),
            "summary_output_dir": run_dir,
            "agent_orchestrator": "AgentOrchestrator",
        })
        # Backward-compatible aliases used by existing reports.
        launcher_manifest = {
            **launcher_manifest,
            "launcher": launcher_manifest.get("launcher", "run_workstation_ensemble.py"),
            "config": args.config,
            "template": args.template,
            "output_base": args.output_base,
            "random_state": args.random_state,
            "fast": bool(args.fast),
            "sample_rows": args.sample_rows if args.fast else None,
            "n_folds": args.n_folds,
            "seeds": args.seeds,
            "timeout_seconds": args.timeout_seconds,
            "branch_id": args.branch_id,
            "branch_type": args.branch_type,
            "code_generation_mode": args.code_generation_mode,
        }
        manifest_path.write_text(json.dumps(launcher_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 60)
    print("ENSEMBLE RUN COMPLETE")
    print("=" * 60)
    print(f"Template: {args.template}")
    print(f"Run dir: {run_info.get('output_dir', 'N/A')}")
    print(f"Best OOF score: {best_score}")
    print(f"Delta vs historical best: {reflection.get('delta_vs_historical_best', 'N/A')}")
    print(f"Stages: {len(summary.get('stages', []))}")
    print(f"Agents executed: {len(summary.get('agent_trace', []))}")
    print()
    print(json.dumps({
        "run_id": run_info.get("output_dir", "").split("/")[-1] if run_info.get("output_dir") else None,
        "template": args.template,
        "best_score": best_score,
        "stages_passed": [s["stage"] for s in summary.get("stages", []) if s.get("status") == "passed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
