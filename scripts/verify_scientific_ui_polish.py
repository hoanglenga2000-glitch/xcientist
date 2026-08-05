from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSTATION = ROOT / "web" / "research-agent-workstation" / "src"
APP_SHELL = WORKSTATION / "components" / "workstation" / "AppShell.tsx"
SIDEBAR = WORKSTATION / "components" / "workstation" / "Sidebar.tsx"
NAV = WORKSTATION / "components" / "workstation" / "navigation.ts"
LAYOUT_DIR = WORKSTATION / "components" / "workstation" / "layout"
PRIMITIVES_DIR = WORKSTATION / "components" / "workstation" / "primitives"
SCREENS_DIR = WORKSTATION / "components" / "workstation" / "screens"
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


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> None:
    screens = sorted(SCREENS_DIR.glob("*.tsx"))
    primitives = sorted(PRIMITIVES_DIR.glob("*.tsx"))
    layout = sorted(LAYOUT_DIR.glob("*.tsx"))

    missing_files = []
    for path in [APP_SHELL, SIDEBAR, NAV, AI_CONTROL, CSS, PAGE]:
        if not path.exists():
            missing_files.append(str(path.relative_to(ROOT)))
    require(not missing_files, "scientific UI source files are missing", {"missing_files": missing_files})
    require(len(screens) >= 14, "expected at least 14 migrated V2 screens", {"screen_count": len(screens)})
    require(len(primitives) >= 3, "expected V2 primitive library", {"primitive_count": len(primitives)})

    app_shell = read(APP_SHELL)
    sidebar = read(SIDEBAR)
    nav = read(NAV)
    ai_control = read(AI_CONTROL)
    css = read(CSS)
    page = read(PAGE)
    screens_src = "\n".join(read(p) for p in screens)
    primitives_src = "\n".join(read(p) for p in primitives)
    layout_src = "\n".join(read(p) for p in layout)
    combined = "\n".join([app_shell, sidebar, nav, ai_control, css, page, screens_src, primitives_src, layout_src])

    # AppShell: polished EvoMind shell, deep link page map, evidence rail, click audit.
    require_terms(
        app_shell,
        [
            "EvoMind",
            "workstation-chrome",
            "max-w-[var(--content-max-width)]",
            "onClickCapture={handleUiClick}",
            "data-ui-page",
            "uiActionRoutes",
            "uiActionRoutePatterns",
            "RunContextBar",
            "EvidenceRail",
        ],
        "AppShell must provide the polished EvoMind shell, click audit delegate, and Evidence Rail",
    )

    # Sidebar: brand, accessible mobile nav, grouped IA, gate status.
    require_terms(
        sidebar,
        [
            "EvoMind",
            "toggle_mobile_navigation",
            "setMobileOpen(false)",
            "navSections",
        ],
        "Sidebar must preserve brand identity, accessible mobile navigation, and grouped IA",
    )

    # Navigation: the 6 IA groups and the 15 leaf pages.
    require_terms(
        nav,
        [
            "command_center",
            "research_loop",
            "workbench",
            "infrastructure",
            "governance",
            "overview",
            "tasks",
            "evidence",
            "experiments",
            "gpu",
            "literature",
            "report",
            "gates",
            "settings",
        ],
        "Navigation must expose the 6 IA groups and all leaf pages",
    )

    # Page: URL deep links, EvoMind default routing, action logging, locale updates.
    require_terms(
        page,
        [
            "parsePageId",
            "pageFromLocation",
            "changeActivePage",
            'searchParams.set("page", page)',
            "hashchange",
            "runWorkstationAction",
            "language_select",
            "create_task",
            "screens/",
        ],
        "Workbench page must support URL deep links, action logging, locale updates, and V2 screens",
    )
    require('from "@/components/workstation/Screens"' not in page, "page.tsx must not import the old Screens monolith")

    # CSS: scientific visual tokens and focus states.
    require_terms(
        css,
        [
            ".workstation-chrome",
            "[data-ui-action]:focus-visible",
            ".thin-scrollbar",
        ],
        "scientific visual tokens and focus states are missing",
    )

    # Primitives: the V2 design system building blocks.
    require_terms(
        primitives_src,
        [
            "PageHeader",
            "MetricTile",
            "Panel",
            "EmptyState",
            "CopyablePath",
            "StatusBadgeV2",
            "GateBadge",
            "ClaimBoundary",
        ],
        "V2 primitives must provide the scientific design-system building blocks",
    )

    # Screens: every page surfaces claim boundaries, status tones, and human gates.
    require_terms(
        screens_src,
        [
            "runWorkstationAction",
            "data-ui-action",
            "StatusBadgeV2",
            "MetricTile",
            "PageHeader",
            "blocked_start_training",
            "blocked_final_evidence_approval",
            "blocked_allow_official_submit",
            "blocked_kaggle_submit",
            "blocked_submit_gpu_job",
            "blocked_send_to_hpc",
            "approve_integrity_gate",
            "export_evidence_csv",
            "report_generate_scientific",
            "report_download_final_bundle",
            "report_analyze_refinement",
            "tasks_create_workstation_run",
            "literature_search",
            "rag_export_context_markdown",
            "settings_language_zh_cn",
            "experiments_export_ledger",
        ],
        "V2 screens must expose status tones, action contracts, exports, and human-gate boundaries",
    )

    # Human-gate integrity: every blocked_* gate must be present with a lock affordance.
    blocked_gates = [term for term in [
        "blocked_start_training",
        "blocked_final_evidence_approval",
        "blocked_allow_official_submit",
        "blocked_kaggle_submit",
        "blocked_submit_gpu_job",
        "blocked_send_to_hpc",
    ] if term in screens_src]
    require(len(blocked_gates) == 6, "all 6 irreversible human-gate boundaries must be present in V2 screens", {"found": blocked_gates})

    # No fabricated ranks/medals: screens must reference unknown/never-fabricate boundaries.
    require(
        "Unknown" in screens_src or "未知" in screens_src,
        "screens must surface an explicit Unknown/unverified state",
    )

    # AI Control Console: research-agent gateway behaviour.
    require_terms(
        ai_control,
        [
            "Scientist Autopilot",
            "Scientist Action Queue",
            "Scientist Workplan",
            "Scientist Execution Contract",
            "Command Input",
            "Code Agents never bypass the workstation",
            "Official Kaggle submission requires human approval",
        ],
        "AI Control Console must look and behave like a serious research-agent gateway",
    )

    forbidden_brand_terms = ["Kaggle Agent", "kaggle agent", "AutoKaggle 工作站入口"]
    forbidden_hits = [term for term in forbidden_brand_terms if term in combined]
    require(not forbidden_hits, "UI must use EvoMind branding instead of the old Kaggle-agent product name", {"forbidden_hits": forbidden_hits})

    replacement_count = combined.count("�")
    private_use_count = sum(1 for char in combined if 0xE000 <= ord(char) <= 0xF8FF)
    require(replacement_count == 0, "UI source must not contain replacement characters", {"replacement_count": replacement_count})
    require(private_use_count == 0, "UI source must not contain private-use mojibake characters", {"private_use_count": private_use_count})

    print(json.dumps({
        "status": "passed",
        "protected_ui_polish": [
            "evomind_brand_shell",
            "accessible_mobile_sidebar",
            "url_deeplinks",
            "scientific_visual_tokens",
            "v2_screens_present",
            "v2_primitives_present",
            "evidence_rail_surface",
            "click_audit_delegate",
            "human_gate_boundaries",
            "no_fake_rank_boundaries",
            "ai_scientist_control_gateway",
        ],
        "screen_count": len(screens),
        "blocked_gate_count": len(blocked_gates),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
