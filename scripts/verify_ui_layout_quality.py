from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSTATION = ROOT / "web" / "research-agent-workstation" / "src" / "components" / "workstation"
SCREENS = WORKSTATION / "Screens.tsx"
APP_SHELL = WORKSTATION / "AppShell.tsx"
AI_CONTROL = WORKSTATION / "AiControlConsole.tsx"


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


def section(source: str, start: str, end: str) -> str:
    start_index = source.find(start)
    if start_index < 0:
        fail("section start not found", {"start": start})
    end_index = source.find(end, start_index + len(start))
    if end_index < 0:
        fail("section end not found", {"start": start, "end": end})
    return source[start_index:end_index]


def missing_terms(source: str, terms: list[str]) -> list[str]:
    return [term for term in terms if term not in source]


def require_terms(source: str, terms: list[str], message: str) -> None:
    missing = missing_terms(source, terms)
    require(not missing, message, {"missing": missing})


def main() -> None:
    missing_files = [str(path.relative_to(ROOT)) for path in [SCREENS, APP_SHELL, AI_CONTROL] if not path.exists()]
    require(not missing_files, "required workstation UI source files are missing", {"missing_files": missing_files})

    screens = SCREENS.read_text(encoding="utf-8")
    app_shell = APP_SHELL.read_text(encoding="utf-8")
    ai_control = AI_CONTROL.read_text(encoding="utf-8")
    combined = "\n".join([screens, app_shell, ai_control])

    action_count = combined.count("data-ui-action")
    route_count = app_shell.count("route:")
    run_action_count = combined.count("runWorkstationAction")
    local_scroll_count = combined.count("overflow-auto") + combined.count("overflow-y-auto")

    require(action_count >= 150, "workstation UI must expose broad clickable action coverage", {"data_ui_action_count": action_count})
    require(route_count >= 70, "AppShell must route major UI actions to page/action contracts", {"route_count": route_count})
    require(run_action_count >= 20, "UI actions must be connected to workstation action logging/API calls", {"runWorkstationAction_count": run_action_count})
    require(local_scroll_count >= 12, "dense workstation panels must use local scroll containers", {"local_scroll_count": local_scroll_count})

    require_terms(
        app_shell,
        [
            "interactiveSelector",
            "handleUiClick",
            "onClickCapture={handleUiClick}",
            "resolveUiActionRoute",
            "exportFallbackForUiAction",
            "downloadTextFile",
            "lg:overflow-y-auto",
            "thin-scrollbar",
            "report_export_draft_pdf",
            "export_evidence_csv",
            "rag_send_research_agent",
            "settings_language_zh_cn",
            "settings_theme_dark",
            "blocked_",
        ],
        "AppShell must keep a global clickable action router, local scrolling, export fallback, and blocked-action routing",
    )

    report = section(screens, "export function ReportStudio", "export function LiteratureKnowledge")
    require_terms(
        report,
        [
            "api.getReport",
            "api.generateReportDraft",
            "selectTaskReport",
            "selectOutlineItem",
            "scrollIntoView",
            "exportMarkdown",
            "exportAuditJson",
            "exportDraftPdfGate",
            "regenerateDraft",
            "report_task_select_",
            "report_section_select_",
            "report_studio_complete_export",
            "report_studio_draft_pdf_gate",
            "blocked_final_report_export",
            "stopWheelPropagation",
            "overscroll-contain",
            "人工 Gate",
            "官方排名",
            "奖牌",
        ],
        "Report Studio must support task switching, section navigation, complete draft regeneration, export, local scroll, and claim boundaries",
    )
    require("xl:grid-cols-[292px_minmax(720px,1fr)_390px]" in report, "Report Studio must keep a wide editable report workspace")

    evidence = section(screens, "export function EvidenceLedger", "export function ReportStudio")
    require_terms(
        evidence,
        [
            "exportCsv",
            "exportDraftBundle",
            "downloadSelectedArtifact",
            "exportArtifactEvidence",
            "artifact_ledger.csv",
            "evidence_draft_bundle.json",
            "export_audit_bundle",
            "download_selected_artifact",
            "trace_artifact_claim",
            "claim_exp_038.json",
            "最终导出需人工 Gate",
        ],
        "Evidence Ledger must provide real front-end export/download/lineage actions with human-gate boundaries",
    )

    literature = section(screens, "export function LiteratureKnowledge", "export function AgentRuntime")
    require_terms(
        literature,
        [
            "api.searchLiterature",
            "runSearch",
            "exportRagMarkdown",
            "exportRagManifest",
            "rag_send_research_agent",
            "rag_send_code_agent",
            "rag_bind_report_claim",
            "rag_request_citation_audit",
            "文献检索",
            "claim_audit",
            "不得直接声明官方提分或奖牌",
        ],
        "Literature/RAG page must perform dynamic retrieval, export context/manifest, and preserve claim-audit boundaries",
    )

    code_runner = section(screens, "export function CodeRunner", "export function GpuHpcConsole")
    require_terms(
        code_runner,
        [
            "selectedFile",
            "openFile",
            "fileAction",
            "code_file_select",
            "Ask Code Agent",
            "Review Diff",
            "Run Smoke",
            "request_code_quality_gate",
            "open_code_folder_agents",
            "quality_gate.json",
        ],
        "Code Agent IDE must keep clickable file navigation and auditable code review actions",
    )

    settings = section(screens, "export function SettingsCenter", "export function DesignSystem")
    require_terms(
        settings,
        [
            "setLanguage",
            "props.setLocale?.(language)",
            "language_select",
            "settings_language_zh_cn",
            "settings_language_en_us",
            "settings_theme_light",
            "settings_theme_dark",
            "recordSettingsAction",
            "test_all_connectors",
            "rotate_credentials_batch",
            "每一项设置变更都写入 evidence ledger",
        ],
        "Settings Center must expose account/language/theme/connector controls and action logging",
    )

    ai_terms = [
        "runScientistAutopilot",
        "executeScientistNextAction",
        "Scientist Action Queue",
        "Execute Safe Next",
        "scientist_workplan",
        "scientist_repair_plan",
        "scientist_execution_contract",
        "scientist_step_trace",
        "no_training_started",
        "blocked_until_explicit_human_approval",
        "训练",
        "官方提交",
    ]
    require_terms(
        ai_control,
        ai_terms,
        "AI Control Console must expose Scientist autopilot, safe-next, workplan, repair, contract, trace, and safety gates",
    )

    private_use_chars = sorted({char for char in combined if "\ue000" <= char <= "\uf8ff"})
    replacement_count = combined.count("\ufffd")
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
                    "local_scroll_count": local_scroll_count,
                },
                "protected_layouts": [
                    "global_click_router",
                    "offline_export_fallback",
                    "report_task_switch_and_export",
                    "evidence_export_download_lineage",
                    "literature_dynamic_rag_and_claim_audit",
                    "code_agent_clickable_file_ide",
                    "settings_language_theme_connector_actions",
                    "ai_scientist_autopilot_safe_next_contract_trace",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
