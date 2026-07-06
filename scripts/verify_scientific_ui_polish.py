from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_SHELL = ROOT / "web" / "research-agent-workstation" / "src" / "components" / "workstation" / "AppShell.tsx"
COMMON = ROOT / "web" / "research-agent-workstation" / "src" / "components" / "workstation" / "Common.tsx"
SIDEBAR = ROOT / "web" / "research-agent-workstation" / "src" / "components" / "workstation" / "Sidebar.tsx"
SCREENS = ROOT / "web" / "research-agent-workstation" / "src" / "components" / "workstation" / "Screens.tsx"
CSS = ROOT / "web" / "research-agent-workstation" / "src" / "app" / "globals.css"
VISUAL_NOTE = ROOT / "docs" / "可视化验收记录-20260612-科研质感补充.md"
UI_DESIGNER_CONTRACT = ROOT / ".codex-ui-designer.md"
UI_AGENT_PROMPT = ROOT / "UI_AGENT_PROMPT.md"
PAGE = ROOT / "web" / "research-agent-workstation" / "src" / "app" / "page.tsx"


def fail(message: str, evidence: dict | None = None) -> None:
    raise SystemExit(json.dumps({"status": "failed", "message": message, "evidence": evidence or {}}, ensure_ascii=False, indent=2))


def require(condition: bool, message: str, evidence: dict | None = None) -> None:
    if not condition:
        fail(message, evidence)


def main() -> None:
    app_shell = APP_SHELL.read_text(encoding="utf-8")
    common = COMMON.read_text(encoding="utf-8")
    sidebar = SIDEBAR.read_text(encoding="utf-8")
    screens = SCREENS.read_text(encoding="utf-8")
    css = CSS.read_text(encoding="utf-8")
    page = PAGE.read_text(encoding="utf-8")

    required_strip_terms = [
        "function LabStatusStrip",
        "CV Gate",
        "Evidence Index",
        "Manual Gate",
        "Launch Boundary",
        "GPU/Kaggle",
        "aria-label",
    ]
    missing_strip_terms = [term for term in required_strip_terms if term not in app_shell]
    require(not missing_strip_terms, "scientific lab status strip is incomplete", {"missing": missing_strip_terms})
    require("<LabStatusStrip locale={locale} />" in app_shell, "lab status strip is not rendered in AppShell main area")
    required_research_brief_terms = [
        "grid-cols-1",
        "overflow-hidden",
        "md:grid-flow-row",
        "xl:grid-cols-5",
    ]
    missing_research_brief_terms = [term for term in required_research_brief_terms if term not in common]
    require(
        not missing_research_brief_terms,
        "research brief must use a mobile-safe single column before desktop grids",
        {"missing": missing_research_brief_terms},
    )
    require(
        "overflow-x-auto" not in common.split("export function ResearchBrief", 1)[1].split("export function MetricCurve", 1)[0],
        "research brief should not rely on mobile horizontal scrolling",
    )
    required_mobile_nav_terms = [
        "mobile-nav-toggle",
        "mobile-workstation-nav",
        "aria-expanded",
        "setMobileOpen(false)",
        "hidden lg:block",
        "sticky top-0",
    ]
    missing_mobile_nav_terms = [term for term in required_mobile_nav_terms if term not in sidebar]
    require(
        not missing_mobile_nav_terms,
        "mobile navigation must collapse by default and expose an accessible toggle",
        {"missing": missing_mobile_nav_terms},
    )
    required_deeplink_terms = [
        "parsePageId",
        "pageFromLocation",
        "changeActivePage",
        "searchParams.set(\"page\", page)",
        "hashchange",
    ]
    missing_deeplink_terms = [term for term in required_deeplink_terms if term not in page]
    require(
        not missing_deeplink_terms,
        "Academic OS pages must support URL deep links for browser acceptance",
        {"missing": missing_deeplink_terms},
    )

    required_scientific_surfaces = [
        "ResourceReadinessPanel",
        "Launch Control",
        "verified-launch-audit-card",
        "Verified Launch Audit",
        "verified_workstation_launch_audit.json",
        "launch-evidence-strip",
        "Next Evidence",
        "manage_claude_secret.ps1 install-key",
        "KAGGLE_USERNAME + KAGGLE_KEY",
        "kaggle-dpapi-readiness-card",
        "Kaggle DPAPI Readiness",
        "Windows DPAPI token",
        "docs/kaggle_dpapi_readiness.json",
        "readinessLabel",
        "Not Configured",
        "nvidia-smi",
        "kaggle-new-competition-readiness-card",
        "final-delivery-status-card",
        "report-preview-canvas",
        "generate-insert-report-figures",
        "toggle-code-assistant-rail",
        "academic-os-readiness-panel",
        "academic-quality-scorecard-panel",
        "Academic OS quality scorecard",
        "DataKagglePipeline",
        "GpuHpcConsole",
        "EvidenceLedger",
        "LiteratureKnowledge",
        "No citation without source",
        "GPU Compute",
        "SSH Credential",
        "Missing-proof policy",
    ]
    missing_surfaces = [term for term in required_scientific_surfaces if term not in screens]
    require(not missing_surfaces, "scientific workstation surfaces are missing", {"missing": missing_surfaces})

    required_css_terms = [
        "scientific work-surface",
        "metric-num",
        "report-page",
        "accent-bar",
    ]
    missing_css_terms = [term for term in required_css_terms if term not in css]
    require(not missing_css_terms, "scientific visual tokens are missing", {"missing": missing_css_terms})

    require(VISUAL_NOTE.is_file(), "scientific visual acceptance note is missing", {"path": str(VISUAL_NOTE.relative_to(ROOT))})
    require(UI_DESIGNER_CONTRACT.is_file(), "UI designer contract is missing", {"path": str(UI_DESIGNER_CONTRACT.relative_to(ROOT))})
    require(UI_AGENT_PROMPT.is_file(), "UI agent prompt is missing", {"path": str(UI_AGENT_PROMPT.relative_to(ROOT))})
    note = VISUAL_NOTE.read_text(encoding="utf-8")
    required_note_terms = [
        "scientific_ui_polish_20260612.png",
        "scientific_ui_mobile_fixed_20260612.png",
        "ResearchBrief",
        "GPU/Kaggle",
    ]
    missing_note_terms = [term for term in required_note_terms if term not in note]
    require(not missing_note_terms, "scientific visual acceptance note is incomplete", {"missing": missing_note_terms})
    designer_contract = UI_DESIGNER_CONTRACT.read_text(encoding="utf-8")
    agent_prompt = UI_AGENT_PROMPT.read_text(encoding="utf-8")
    required_design_resource_terms = [
        "Academic Research OS",
        "No fake citation",
        "SSH Gateway Ready",
        "Not Configured",
        "任务总控台",
    ]
    missing_design_resource_terms = [
        term for term in required_design_resource_terms
        if term not in designer_contract and term not in agent_prompt
    ]
    require(
        not missing_design_resource_terms,
        "Academic OS UI design resources are incomplete",
        {"missing": missing_design_resource_terms},
    )
    screenshot_paths = [
        ROOT / "docs" / "scientific_ui_polish_20260612.png",
        ROOT / "docs" / "scientific_ui_mobile_fixed_20260612.png",
    ]
    missing_screenshots = [str(path.relative_to(ROOT)) for path in screenshot_paths if not path.is_file() or path.stat().st_size < 10_000]
    require(not missing_screenshots, "scientific visual screenshots are missing or too small", {"missing": missing_screenshots})

    print(json.dumps({
        "status": "passed",
        "protected_ui_polish": [
            "lab_status_strip",
            "kaggle_new_competition_readiness_card",
            "final_delivery_status_card",
            "wide_report_canvas",
            "report_figure_workflow",
            "code_focus_mode",
            "scientific_grid_surface",
            "mobile_safe_research_brief",
            "launch_resource_readiness_panel",
            "launch_evidence_action_strip",
            "verified_launch_audit_card",
            "scientific_visual_acceptance_note",
            "academic_os_ui_designer_contract",
            "academic_os_ui_agent_prompt",
            "academic_os_page_deeplinks",
            "mobile_collapsible_navigation",
            "academic_os_capability_map",
            "academic_os_quality_scorecard",
            "data_kaggle_pipeline_page",
            "gpu_hpc_console_page",
            "evidence_ledger_page",
            "literature_knowledge_page",
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
