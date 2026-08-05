from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSTATION = ROOT / "web" / "research-agent-workstation" / "src" / "components" / "workstation"
SCREENS_DIR = WORKSTATION / "screens"
PRIMITIVES_DIR = WORKSTATION / "primitives"
LAYOUT_DIR = WORKSTATION / "layout"
APP_SHELL = WORKSTATION / "AppShell.tsx"
AI_CONTROL = WORKSTATION / "AiControlConsole.tsx"
PAGE = ROOT / "web" / "research-agent-workstation" / "src" / "app" / "page.tsx"


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


def require(condition: bool, message: str, evidence: dict | None = None) -> None:
    if not condition:
        fail(message, evidence)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def missing_terms(source: str, terms: list[str]) -> list[str]:
    return [term for term in terms if term not in source]


def require_terms(source: str, terms: list[str], message: str) -> None:
    missing = missing_terms(source, terms)
    require(not missing, message, {"missing": missing})


def screen_source(name: str) -> str:
    path = SCREENS_DIR / name
    require(path.exists(), "expected migrated V2 screen", {"screen": name})
    return read(path)


def main() -> None:
    require(APP_SHELL.exists(), "AppShell missing")
    require(AI_CONTROL.exists(), "AI Control missing")
    screens = sorted(SCREENS_DIR.glob("*.tsx"))
    require(len(screens) >= 14, "expected at least 14 migrated V2 screens", {"screen_count": len(screens)})

    app_shell = read(APP_SHELL)
    ai_control = read(AI_CONTROL)
    page = read(PAGE)
    primitives_src = "\n".join(read(p) for p in sorted(PRIMITIVES_DIR.glob("*.tsx")))
    layout_src = "\n".join(read(p) for p in sorted(LAYOUT_DIR.glob("*.tsx")))
    all_screens_src = "\n".join(read(p) for p in screens)
    combined = "\n".join([all_screens_src, app_shell, ai_control, primitives_src, layout_src])

    action_count = combined.count("data-ui-action")
    route_count = app_shell.count("route:") + app_shell.count("prefix:")
    run_action_count = combined.count("runWorkstationAction")

    require(action_count >= 70, "workstation UI must expose broad clickable action coverage", {"data_ui_action_count": action_count})
    require(route_count >= 40, "AppShell must route major UI actions to page/action contracts", {"route_count": route_count})
    require(run_action_count >= 20, "UI actions must be connected to workstation action logging/API calls", {"runWorkstationAction_count": run_action_count})

    # Global click router + audit + export fallback + blocked-action routing.
    require_terms(
        app_shell,
        [
            "interactiveSelector",
            "handleUiClick",
            "onClickCapture={handleUiClick}",
            "resolveUiActionRoute",
            "thin-scrollbar",
            "blocked_",
            "data-ui-skip-action",
        ],
        "AppShell must keep a global clickable action router and blocked-action routing",
    )

    # V2 primitives provide the dense scientific layout building blocks.
    require_terms(
        primitives_src,
        ["PageHeader", "MetricTile", "Panel", "StatusBadgeV2", "CopyablePath", "EmptyState"],
        "V2 primitives must provide dense scientific layout building blocks",
    )

    # Report Studio: reviewed Nature report, verified downloads, refinement and Human Gate.
    report = screen_source("ReportStudioScreen.tsx")
    require_terms(
        report,
        [
            "report_generate_scientific",
            "report_download_final_bundle",
            "report_analyze_refinement",
            "report_review_human_gate",
            "report_approve_refinement",
            "Nature Skills",
            "responseHash",
            "Human Gate approved · recorded",
        ],
        "Report Studio must support reviewed Nature rendering, verified delivery, refinement, and Human Gate",
    )

    # Evidence Ledger: real export/download/lineage actions with human-gate boundary.
    evidence = screen_source("EvidenceLedgerScreen.tsx")
    require_terms(
        evidence,
        [
            "export_evidence_csv",
            "export_draft_evidence_bundle",
            "open_evidence_lineage_graph",
            "apply_evidence_filters",
            "reset_evidence_filters",
            "blocked_final_evidence_approval",
            "toCsv",
            "人工闸门",
        ],
        "Evidence Ledger must provide real front-end export/lineage actions with human-gate boundaries",
    )

    # Literature/RAG: dynamic retrieval, export context/manifest, claim-audit boundary.
    literature = screen_source("LiteratureScreen.tsx")
    require_terms(
        literature,
        [
            "literature_search",
            "rag_send_research_agent",
            "rag_send_code_agent",
            "rag_bind_report_claim",
            "rag_request_citation_audit",
            "rag_export_context_markdown",
            "rag_export_manifest_json",
            "claim_audit",
        ],
        "Literature/RAG page must perform dynamic retrieval, export context/manifest, and preserve claim-audit boundaries",
    )

    # Code Agent IDE: quality gate, smoke test, human HPC gate.
    code = screen_source("CodeAgentScreen.tsx")
    require_terms(
        code,
        [
            "ask_code_agent",
            "run_code_smoke_test",
            "request_code_quality_gate",
            "blocked_send_to_hpc",
            "CopyablePath",
        ],
        "Code Agent IDE must keep auditable code review actions and human HPC gate",
    )

    # Settings: language/theme/connector controls and action logging.
    settings = screen_source("SettingsScreen.tsx")
    require_terms(
        settings,
        [
            "settings_language_zh_cn",
            "settings_language_en_us",
            "settings_theme_light",
            "settings_theme_dark",
            "recordSettingsAction",
            "test_all_connectors",
            "rotate_credentials_batch",
        ],
        "Settings must expose language/theme/connector controls and action logging",
    )

    # Tasks: run creation, dispatch, selection contracts.
    tasks = screen_source("TasksScreen.tsx")
    require_terms(
        tasks,
        ["tasks_select_", "tasks_refresh_queue", "tasks_create_workstation_run", "tasks_dispatch_agents"],
        "Tasks must expose selection, refresh, run creation, and dispatch contracts",
    )

    # GPU/HPC: compute selection + human training gate.
    gpu = screen_source("GpuHpcScreen.tsx")
    require_terms(
        gpu,
        ["compute_select_hpc_gpu", "compute_select_local", "blocked_start_training", "人工闸门"],
        "GPU/HPC must expose compute selection and the human training gate",
    )

    # Gates: integrity decision contracts + official submit human gate.
    gates = screen_source("GatesScreen.tsx")
    require_terms(
        gates,
        ["approve_integrity_gate", "reject_integrity_gate", "request_gate_revision", "blocked_allow_official_submit"],
        "Gates must expose integrity decision contracts and the official-submit human gate",
    )

    # AI Control Console: Scientist autopilot, safe-next, workplan, contract, trace, safety gates.
    require_terms(
        ai_control,
        [
            "Scientist Autopilot",
            "Scientist Action Queue",
            "Scientist Workplan",
            "Scientist Execution Contract",
            "Scientist Step Trace",
            "no_training_started",
            "blocked_until_explicit_human_approval",
        ],
        "AI Control Console must expose Scientist autopilot, workplan, contract, trace, and safety gates",
    )

    private_use_chars = sorted({char for char in combined if 0xE000 <= ord(char) <= 0xF8FF})
    replacement_count = combined.count("�")
    require(replacement_count == 0, "UI source must not contain replacement characters", {"replacement_count": replacement_count})
    require(
        len(private_use_chars) <= 30,
        "UI source has too many private-use characters; likely mojibake leaked into visible copy",
        {"private_use_char_count": len(private_use_chars), "sample": [hex(ord(char)) for char in private_use_chars[:12]]},
    )

    print(
        json.dumps(
            {
                "status": "passed",
                "action_coverage": {
                    "data_ui_action_count": action_count,
                    "route_count": route_count,
                    "runWorkstationAction_count": run_action_count,
                },
                "protected_layouts": [
                    "global_click_router",
                    "v2_primitives",
                    "report_nature_rendering_verified_delivery_and_refinement",
                    "evidence_export_download_lineage",
                    "literature_dynamic_rag_and_claim_audit",
                    "code_agent_auditable_actions",
                    "settings_language_theme_connector_actions",
                    "tasks_run_dispatch_contracts",
                    "gpu_human_training_gate",
                    "integrity_gate_contracts",
                    "ai_scientist_autopilot_contract_trace",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
