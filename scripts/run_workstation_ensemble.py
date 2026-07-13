"""Prepare a governed HPC ensemble manifest without starting local training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_agent_workstation.server.services import AgentOrchestrator  # noqa: E402
from research_agent_workstation.server.training import EnsembleTemplateRegistry  # noqa: E402
from research_os.hpc_policy import HPCPolicyError  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a registered HPC ensemble manifest. This command does not execute training.",
    )
    parser.add_argument(
        "--template",
        default="exp007_style_lgb_xgb_cat_blend",
        choices=EnsembleTemplateRegistry.template_ids(),
        help="Approved HPC ensemble template ID.",
    )
    parser.add_argument(
        "--config",
        required="--list-templates" not in sys.argv,
        help="Task config YAML, e.g. configs/generated/playground_series_s6e6.yaml",
    )
    parser.add_argument("--output-base", default="experiments", help="Manifest output base directory.")
    parser.add_argument("--random-state", type=int, default=42, help="Compatibility metadata only.")
    parser.add_argument("--list-templates", action="store_true", help="List registered templates and exit.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Compatibility metadata only; this flag never enables local training.",
    )
    parser.add_argument("--sample-rows", type=int, default=20000, help="Compatibility metadata only.")
    parser.add_argument("--n-folds", type=int, default=None, help="Compatibility metadata only.")
    parser.add_argument("--seeds", default="", help="Compatibility metadata only.")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=7200,
        help="Timeout recorded in the HPC job manifest.",
    )
    parser.add_argument("--branch-id", default="", help="Search-controller branch id.")
    parser.add_argument("--branch-type", default="", help="Search-controller branch type.")
    parser.add_argument("--code-generation-mode", default="", help="Search-controller code generation mode.")
    parser.add_argument("--branch-hypothesis", default="", help="Governed branch hypothesis.")
    parser.add_argument("--cross-branch-references", default="", help="JSON encoded cross-branch references.")
    return parser.parse_args()


def _list_templates() -> None:
    for template_id in EnsembleTemplateRegistry.template_ids():
        template = EnsembleTemplateRegistry.get(template_id)
        if template is None:
            continue
        print(
            json.dumps(
                {
                    "template_id": template_id,
                    "name": template.name,
                    "hpc_required": template.hpc_required,
                    "approved": template.approved,
                    "risk_level": template.risk_level,
                },
                ensure_ascii=False,
            )
        )


def main() -> int:
    args = parse_args()
    if args.list_templates:
        _list_templates()
        return 0

    try:
        cross_branch_references = json.loads(args.cross_branch_references) if args.cross_branch_references else []
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--cross-branch-references must be valid JSON: {exc}") from exc
    if not isinstance(cross_branch_references, list):
        raise SystemExit("--cross-branch-references must decode to a JSON list")

    orchestrator = AgentOrchestrator(ROOT)
    try:
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
                "cross_branch_references": cross_branch_references,
            },
        )
    except HPCPolicyError as exc:
        print(
            json.dumps(
                {
                    "status": "blocked_hpc_configuration",
                    "reason": str(exc),
                    "training_started": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    run = summary.get("run", {})
    if (
        summary.get("status") != "manifest_prepared_awaiting_dispatch"
        or run.get("hpc_job_queued") is not False
        or run.get("training_started") is not False
        or run.get("accepted") is not False
    ):
        raise RuntimeError("HPC manifest preparation returned an invalid dispatch or training claim")
    payload = {
        "status": summary.get("status"),
        "template": summary.get("template"),
        "training_started": False,
        "accepted": False,
        "job_manifest": run.get("job_manifest"),
        "output_dir": run.get("output_dir"),
        "task_state": summary.get("task_state", {}).get("state"),
        "claim_boundary": summary.get("claim_boundary"),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
