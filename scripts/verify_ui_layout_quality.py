from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCREENS = ROOT / "web" / "research-agent-workstation" / "src" / "components" / "workstation" / "Screens.tsx"


def fail(message: str, evidence: dict | None = None) -> None:
    raise SystemExit(
        json.dumps(
            {
                "status": "failed",
                "message": message,
                "evidence": evidence or {},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def section(source: str, start: str, end: str) -> str:
    start_index = source.find(start)
    if start_index < 0:
        fail("section start not found", {"start": start})
    end_index = source.find(end, start_index + len(start))
    if end_index < 0:
        fail("section end not found", {"start": start, "end": end})
    return source[start_index:end_index]


def require(condition: bool, message: str, evidence: dict | None = None) -> None:
    if not condition:
        fail(message, evidence)


def main() -> None:
    source = SCREENS.read_text(encoding="utf-8")
    system_action_calls = source.count("<SystemActionPanel")
    compact_calls = source.count("<SystemActionPanel") - source.count("<SystemActionPanel(")
    non_compact_lines = [
        line.strip()
        for line in source.splitlines()
        if "<SystemActionPanel" in line and "compact" not in line
    ]
    require(system_action_calls >= 6, "expected multiple system action panels in workstation UI", {"count": system_action_calls})
    require(not non_compact_lines, "all SystemActionPanel usages must use compact mode", {"non_compact_lines": non_compact_lines})

    report = section(source, "export function ReportStudio", "export function IntegrityGates")
    require(
        'xl:grid-cols-[220px_minmax(0,1fr)]' in report and '2xl:grid-cols-[240px_minmax(0,1fr)]' in report,
        "report studio must keep a narrow two-column outline plus wide document workspace",
    )
    require(
        'data-testid="toggle-report-outline"' in report
        and "outlineOpen" in report
        and "report_outline_toggle" in report
        and "toggleReportOutline" in report,
        "report studio must support an audited focus mode that hides the outline and gives the document full width",
    )
    require(
        'xl:grid-cols-[260px_minmax(0,1fr)_360px]' not in report and 'xl:grid-cols-[250px_minmax(0,1fr)_370px]' not in report,
        "report studio must not reintroduce a fixed third action-log column",
    )
    require(
        report.find("<ReportInspector") < report.find("<SystemActionPanel"),
        "report studio audit log should appear after report tools and inspector",
    )
    require(
        'data-testid="report-preview-canvas"' in report and 'max-w-[1180px]' in report,
        "report studio preview must render as a wide centered document canvas",
    )
    require(
        'data-testid="toggle-report-figure-tray"' in report
        and "figureTrayOpen" in report
        and "report_figure_tray_toggle" in report
        and "toggleFigureTray" in report
        and 'h-[calc(100vh-210px)]' in report,
        "report figures must live in an audited collapsible tray so the document preview is not compressed",
    )
    require(
        "headingTargets.map" in report and "pendingHeadingId" in report and "scrollIntoView" in report,
        "report outline clicks must scroll the preview canvas to the selected section",
    )
    require(
        ("artifactPreviewUrl(rawPath)" in source or "artifactPreviewUrl(image.src)" in source)
        and "parseMarkdownImageLine" in source
        and 'data-testid="generate-insert-report-figures"' in report
        and "Professional Figure Workbench" in report
        and "figureManifestPath" in report,
        "report figures must render through the artifact preview route and support one-click insertion",
    )
    require(
        'xl:grid-cols-[minmax(0,1fr)_420px]' not in report,
        "report studio must not place the action log in a fixed right rail",
    )

    settings = section(source, "export function SettingsCenter", "export function OverviewBoard")
    require(
        'xl:grid-cols-[260px_minmax(0,1fr)]' in settings,
        "settings center must keep configuration navigation and form as the primary two-column layout",
    )
    require(
        'xl:grid-cols-[260px_minmax(0,1fr)_360px]' not in settings,
        "settings center must not reintroduce a fixed third action-log column",
    )

    workflow = section(source, "export function WorkflowGraph", "function ResearchWorkflowNode")
    require(
        workflow.find("<SelectedNodePanel") < workflow.find("<SystemActionPanel"),
        "workflow graph right rail should prioritize selected node details before action log",
    )

    code_runner = section(source, "export function CodeRunner", "export function AgentRuntime")
    code_agent_index = code_runner.find("<CodeAgentBridge")
    gpu_index = code_runner.find("<GpuGatewayPanel")
    patch_index = code_runner.find("<DeveloperPatch")
    action_index = code_runner.find("<SystemActionPanel")
    require(
        min(code_agent_index, gpu_index, patch_index, action_index) >= 0,
        "code runner must expose code agent, gpu gateway, patch review, and action log",
        {
            "code_agent_index": code_agent_index,
            "gpu_index": gpu_index,
            "patch_index": patch_index,
            "action_index": action_index,
        },
    )
    require(
        code_agent_index < action_index and gpu_index < action_index and patch_index < action_index,
        "code runner action log must stay below primary coding agent, GPU, and patch workflow controls",
    )
    require(
        'data-testid="toggle-code-assistant-rail"' in code_runner
        and "codeAssistantRailOpen" in code_runner
        and "code_workspace_rail_toggle" in code_runner,
        "code runner must provide an audited IDE focus mode that can hide the agent rail",
    )

    runtime = section(source, "export function AgentRuntime", "type ExperimentDisplayRun")
    require(
        runtime.find("<ManualOversight") < runtime.find("<SystemActionPanel"),
        "agent runtime right rail should prioritize manual oversight before action log",
    )

    print(
        json.dumps(
            {
                "status": "passed",
                "system_action_panels": system_action_calls,
                "protected_layouts": [
                    "report_studio_two_column_editor",
                    "report_studio_focus_mode",
                    "report_studio_wide_document_preview",
                    "report_studio_outline_anchor_scroll",
                    "report_studio_rendered_figure_workflow",
                    "report_studio_action_log_below_workspace",
                    "settings_two_column_form",
                    "workflow_selected_node_before_log",
                    "code_runner_agent_gpu_patch_before_log",
                    "code_runner_audited_focus_mode",
                    "agent_runtime_manual_gate_before_log",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
