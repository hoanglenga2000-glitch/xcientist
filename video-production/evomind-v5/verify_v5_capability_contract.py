from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(root: Path, relative: str) -> str:
    path = root / relative
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8-sig")


def _check(
    checks: list[dict[str, Any]],
    *,
    capability: str,
    source: str,
    text: str,
    markers: tuple[str, ...],
) -> None:
    missing = [marker for marker in markers if marker not in text]
    checks.append(
        {
            "capability": capability,
            "passed": not missing and bool(text),
            "source": source,
            "missing_markers": missing,
        }
    )


def verify(root: Path) -> dict[str, Any]:
    report_ui_path = "web/research-agent-workstation/src/components/workstation/screens/ReportStudioScreen.tsx"
    report_server_path = "web/research-agent-workstation/src/lib/server/scientific-report.ts"
    report_api_path = "web/research-agent-workstation/src/app/api/tasks/[taskId]/scientific-report/route.ts"
    refinement_api_path = "web/research-agent-workstation/src/app/api/tasks/[taskId]/refinement/route.ts"
    resume_api_path = "web/research-agent-workstation/src/app/api/multi-agent/runs/[runId]/[action]/route.ts"
    refinement_path = "src/research_os/agent/llm_refinement_workflow.py"
    evidence_gate_path = "video-production/evomind-v5/verify_v5_evidence.py"
    public_gate_path = "video-production/evomind-v5/verify_v5_public_text.py"
    build_path = "video-production/evomind-v5/build_video_v5.py"

    report_ui = _read(root, report_ui_path)
    report_server = _read(root, report_server_path)
    report_api = _read(root, report_api_path)
    refinement_api = _read(root, refinement_api_path)
    resume_api = _read(root, resume_api_path)
    refinement = _read(root, refinement_path)
    evidence_gate = _read(root, evidence_gate_path)
    public_gate = _read(root, public_gate_path)
    build = _read(root, build_path)

    checks: list[dict[str, Any]] = []
    _check(
        checks,
        capability="Nature Skills invocation is visible and durable",
        source=f"{report_ui_path}; {report_api_path}",
        text=report_ui + report_api,
        markers=(
            "Collecting verified evidence",
            "Loading training metrics",
            "Rendering scientific figures",
            "Building Nature Skills report",
            "Attaching audit appendix",
            "Nature Skills scientific report rendered from reviewed run evidence.",
        ),
    )
    _check(
        checks,
        capability="Final report uses HTML and PDF instead of Markdown presentation",
        source=report_server_path,
        text=report_server,
        markers=(
            "Final representation: scientific-report.html / scientific-report.pdf",
            'artifact("report-html"',
            'artifact("report-pdf"',
        ),
    )
    _check(
        checks,
        capability="Scientific report contains reviewed methods, results, limits and audit",
        source=report_server_path,
        text=report_server,
        markers=(
            "EvoMind Scientific Report",
            "方法与训练设置",
            "关键结论",
            "局限性",
            "Independent Reviewer",
            "Claim Audit",
        ),
    )
    _check(
        checks,
        capability="Figures are reconstructed from real metrics and telemetry",
        source=report_server_path,
        text=report_server,
        markers=(
            "Figure 1 | Model performance and training behaviour",
            "telemetry.jsonl",
            "no values are interpolated",
            "fixed test set",
        ),
    )
    _check(
        checks,
        capability="Report preview exposes Report, Figures, Methods, Audit and Files",
        source=report_ui_path,
        text=report_ui,
        markers=(
            'id: "report", label: "Report"',
            'id: "figures", label: "Figures"',
            'id: "methods", label: "Methods"',
            'id: "audit", label: "Audit"',
            'id: "files", label: "Files"',
            'title="Interactive scientific report"',
        ),
    )
    _check(
        checks,
        capability="Figure interaction opens a stable detail view",
        source=report_ui_path,
        text=report_ui,
        markers=("setSelectedFigure", 'role="dialog"', 'object-contain'),
    )
    _check(
        checks,
        capability="Artifact Center is grouped by report, model, reproducibility and audit",
        source=report_ui_path,
        text=report_ui,
        markers=(
            'label: "Research Report"',
            'label: "Model Artifacts"',
            'label: "Reproducibility"',
            'label: "Audit"',
        ),
    )
    _check(
        checks,
        capability="Downloads expose preparation, completion and hash verification",
        source=report_ui_path,
        text=report_ui,
        markers=(
            '"preparing"',
            '"downloaded"',
            "downloaded and SHA256 verified",
            "Download Final Bundle",
        ),
    )
    _check(
        checks,
        capability="Natural-language refinement continues the same reviewed task",
        source=f"{report_ui_path}; {refinement_path}",
        text=report_ui + refinement,
        markers=("parseRefinement", "parse_refinement_changes", "parent_run_id", "fixed_test_set"),
    )
    _check(
        checks,
        capability="Requested changes and affected steps are explicit",
        source=f"{report_ui_path}; {refinement_path}",
        text=report_ui + refinement,
        markers=("Changed parameters", "Affected workflow", "Only affected steps will rerun", "requested_changes"),
    )
    _check(
        checks,
        capability="Refinement requires a second Human Gate",
        source=f"{report_ui_path}; {refinement_api_path}",
        text=report_ui + refinement_api,
        markers=("Human Gate · Refinement approval", "Approve refinement", 'action === "approve"', "refine-decide"),
    )
    _check(
        checks,
        capability="V1 is immutable and V2 is a separately reserved run",
        source=refinement_path,
        text=refinement,
        markers=("parent_preserved", "proposed_version", "reserve_refinement_run", "never mutates the parent run directory"),
    )
    _check(
        checks,
        capability="Interrupted V2 resumes the same run",
        source=f"{report_ui_path}; {resume_api_path}",
        text=report_ui + resume_api,
        markers=("Resume", 'action === "resume"', '"resume", "--run-id", runId', "already_resuming"),
    )
    _check(
        checks,
        capability="Independent Reviewer and Claim Audit rerun before synthesis",
        source=refinement_path,
        text=refinement,
        markers=("independent_review", "claim_audit", "version_comparison_recomputed", "parent_run_preserved"),
    )
    _check(
        checks,
        capability="V1 and V2 comparison reports honest outcomes",
        source=f"{report_server_path}; {refinement_path}",
        text=report_server + refinement,
        markers=("version_comparison.json", '"improved"', '"no_material_change"', '"trade_off_detected"', "不会把无实质变化或 trade-off 描述为提升"),
    )
    _check(
        checks,
        capability="Nature report is regenerated automatically after reviewed V2",
        source=f"{refinement_api_path}; {resume_api_path}",
        text=refinement_api + resume_api,
        markers=("generateScientificReport", "automated: true"),
    )
    _check(
        checks,
        capability="Evidence gate blocks unreviewed V2 and missing real captures",
        source=evidence_gate_path,
        text=evidence_gate,
        markers=("EXPECTED_REPORT_RENDERER", "version_comparison_recomputed", "adapter_reload", "require-captures"),
    )
    _check(
        checks,
        capability="Public video hides private compute facts and uses reviewed A40 wording",
        source=f"{public_gate_path}; {build_path}",
        text=public_gate + build,
        markers=("private_gpu_model", "internal_run_id", "A40 recommended configuration", "require-captures"),
    )

    passed = sum(1 for item in checks if item["passed"])
    return {
        "schema": "evomind.video.v5_capability_contract.v1",
        "status": "passed" if passed == len(checks) else "blocked",
        "passed": passed,
        "total": len(checks),
        "checks": checks,
        "blockers": [
            f"{item['capability']}: {', '.join(item['missing_markers']) or 'source missing'}"
            for item in checks
            if not item["passed"]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the complete EvoMind V5 product capability contract.")
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    result = verify(args.workspace_root.resolve())
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
