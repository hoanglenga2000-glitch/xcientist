from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSTATION = ROOT / "web" / "research-agent-workstation" / "src"
APP_SHELL = WORKSTATION / "components" / "workstation" / "AppShell.tsx"
COMMON = WORKSTATION / "components" / "workstation" / "Common.tsx"
SIDEBAR = WORKSTATION / "components" / "workstation" / "Sidebar.tsx"
SCREENS = WORKSTATION / "components" / "workstation" / "Screens.tsx"
AI_CONTROL = WORKSTATION / "components" / "workstation" / "AiControlConsole.tsx"
CSS = WORKSTATION / "app" / "globals.css"
PAGE = WORKSTATION / "app" / "page.tsx"


def fail(message: str, evidence: dict | None = None) -> None:
    raise SystemExit(json.dumps({"status": "failed", "message": message, "evidence": evidence or {}}, ensure_ascii=False, indent=2))


def require(condition: bool, message: str, evidence: dict | None = None) -> None:
    if not condition:
        fail(message, evidence)


def missing_terms(source: str, terms: list[str]) -> list[str]:
    return [term for term in terms if term not in source]


def require_terms(source: str, terms: list[str], message: str) -> None:
    missing = missing_terms(source, terms)
    require(not missing, message, {"missing": missing})


def main() -> None:
    files = [APP_SHELL, COMMON, SIDEBAR, SCREENS, AI_CONTROL, CSS, PAGE]
    missing_files = [str(path.relative_to(ROOT)) for path in files if not path.exists()]
    require(not missing_files, "scientific UI source files are missing", {"missing_files": missing_files})

    app_shell = APP_SHELL.read_text(encoding="utf-8")
    common = COMMON.read_text(encoding="utf-8")
    sidebar = SIDEBAR.read_text(encoding="utf-8")
    screens = SCREENS.read_text(encoding="utf-8")
    ai_control = AI_CONTROL.read_text(encoding="utf-8")
    css = CSS.read_text(encoding="utf-8")
    page = PAGE.read_text(encoding="utf-8")
    combined = "\n".join([app_shell, common, sidebar, screens, ai_control, css, page])

    require_terms(
        app_shell,
        [
            "EvoMind Gateway",
            "EvoMind 工作站",
            "Research Overview",
            "Experiment Ledger",
            "Code Agent IDE",
            "GPU / HPC",
            "Evidence Ledger",
            "Integrity Gates",
            "Literature / RAG",
            "Workflow Graph",
            "Settings",
            "Design System",
            "max-w-[1672px]",
            "workstation-chrome",
            "Topbar",
        ],
        "AppShell must provide a polished EvoMind research-workstation shell and page map",
    )

    require_terms(
        sidebar,
        [
            "EvoMind",
            "XCIENTIST RESEARCH AGENT",
            "toggle_mobile_navigation",
            "setMobileOpen(false)",
            "lg:overflow-y-auto",
            "bg-[linear-gradient",
            "科研",
            "开发",
            "基础设施",
            "治理",
            "资源与连接",
            "人工 Gate",
        ],
        "Sidebar must preserve brand identity, accessible mobile navigation, grouped pages, and resource/gate status",
    )

    require_terms(
        page,
        [
            "parsePageId",
            "pageFromLocation",
            "changeActivePage",
            "searchParams.set(\"page\", page)",
            "hashchange",
            "AiControlConsole",
            "EvolutionConsole",
            "runWorkstationAction",
            "language_select",
            "create_task",
        ],
        "Workbench page must support URL deep links, EvoMind default routing, action logging, and locale updates",
    )

    require_terms(
        css,
        [
            ".thin-scrollbar",
            ".report-page",
            ".metric-num",
            ".accent-bar",
            ".workstation-chrome",
            "[data-ui-action]:focus-visible",
            ".dense-surface",
        ],
        "scientific visual tokens and focus states are missing",
    )

    require_terms(
        common,
        [
            "ResearchBrief",
            "grid min-w-0 grid-cols-1",
            "md:grid-flow-row",
            "xl:grid-cols-5",
            "MetricCurve",
            "ArtifactList",
            "ReproducibilityRecord",
        ],
        "shared scientific widgets must remain mobile-safe and evidence-oriented",
    )
    research_brief = common.split("export function ResearchBrief", 1)[1].split("function BriefExtra", 1)[0]
    require("overflow-x-auto" not in research_brief, "ResearchBrief must not rely on mobile horizontal scrolling")

    require_terms(
        screens,
        [
            "LiveRunEvidencePanel",
            "实时运行证据 / Live Run Evidence",
            "无 Kaggle response 时不显示排名或奖牌",
            "TerminalKaggleAgentPanel",
            "DataKagglePipeline",
            "GpuHpcConsole",
            "EvidenceLedger",
            "ReportStudio",
            "LiteratureKnowledge",
            "AgentRuntime",
            "IntegrityGates",
            "Experiments",
            "ResearchTasks",
            "WorkflowGraph",
            "SettingsCenter",
            "DesignSystem",
            "No - waiting for Submission Gate",
            "DPAPI ready",
            "HPC / GPU",
            "claim_audit.json",
            "暂无官方 response artifact",
            "Code Agent",
            "GPU Agent",
            "Submission Gate",
        ],
        "scientific workstation surfaces must expose live evidence, terminal agent, data, GPU, evidence, report, literature, runtime, gates, tasks, and settings",
    )

    require_terms(
        ai_control,
        [
            "Scientist Autopilot",
            "Scientist Action Queue",
            "Scientist Workplan",
            "Scientist Repair Plan",
            "Scientist Execution Contract",
            "Scientist Step Trace",
            "Command Input",
            "Quick Actions",
            "Page Shortcuts",
            "Message History",
            "Code Agents never bypass the workstation",
            "Official Kaggle submission requires human approval",
        ],
        "AI Control Console must look and behave like a serious research-agent gateway",
    )

    forbidden_brand_terms = ["Kaggle Agent", "kaggle agent", "AutoKaggle 工作站入口"]
    forbidden_hits = [term for term in forbidden_brand_terms if term in combined]
    require(not forbidden_hits, "UI must use EvoMind branding instead of the old Kaggle-agent product name", {"forbidden_hits": forbidden_hits})

    replacement_count = combined.count("\ufffd")
    private_use_count = sum(1 for char in combined if "\ue000" <= char <= "\uf8ff")
    require(replacement_count == 0, "UI source must not contain replacement characters", {"replacement_count": replacement_count})
    require(private_use_count == 0, "UI source must not contain private-use mojibake characters", {"private_use_count": private_use_count})

    print(json.dumps({
        "status": "passed",
        "protected_ui_polish": [
            "evomind_brand_shell",
            "accessible_mobile_sidebar",
            "url_deeplinks",
            "scientific_visual_tokens",
            "mobile_safe_research_brief",
            "live_run_evidence_surface",
            "terminal_agent_panel",
            "data_gpu_evidence_report_literature_runtime_pages",
            "ai_scientist_control_gateway",
            "human_gate_and_no_fake_rank_boundaries",
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
