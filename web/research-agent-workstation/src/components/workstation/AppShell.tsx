"use client";

import {
  Bell,
  Bot,
  ChevronDown,
  Code2,
  Database,
  FileText,
  Search,
  Server,
  ShieldCheck,
  Timer
} from "lucide-react";
import type { MouseEvent } from "react";
import { cn } from "@/lib/utils";
import type { UiComponentClickMetadata } from "@/lib/api/types";
import { Sidebar } from "./Sidebar";
import type { PageId } from "./navigation";

type Locale = "zh-CN" | "en-US";
const pageIds = [
  "overview",
  "control",
  "experiments",
  "data",
  "report",
  "code",
  "gpu",
  "evidence",
  "gates",
  "literature",
  "tasks",
  "runtime",
  "workflow",
  "settings",
  "design"
] as const satisfies readonly PageId[];

function copy(locale: Locale | undefined, en: string, zh: string) {
  return locale === "zh-CN" ? zh : en;
}

const interactiveSelector = [
  "button",
  "a",
  "input",
  "select",
  "textarea",
  "[role='button']",
  "[data-testid]",
  "[data-ui-action]",
  "[data-ui-component]",
  "tr"
].join(",");

function cleanText(value: string | null | undefined) {
  return (value ?? "").replace(/\s+/g, " ").trim().slice(0, 120);
}

function downloadTextFile(filename: string, content: string, mime = "text/plain;charset=utf-8") {
  if (typeof window === "undefined") return;
  const blob = new Blob([content], { type: mime });
  const url = window.URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.URL.revokeObjectURL(url);
}

function exportFallbackForUiAction({
  action,
  actionId,
  activePage,
  label,
  metadata
}: {
  action?: string;
  actionId?: string;
  activePage: PageId;
  label?: string;
  metadata?: Record<string, unknown>;
}) {
  const id = String(actionId ?? "");
  if (!action && !id.includes("export") && !id.includes("download")) return;
  if (action !== "export_audit_bundle" && action !== "export_report" && !id.includes("export") && !id.includes("download")) return;

  const exportedAt = new Date().toISOString();
  const basePayload = {
    exported_at: exportedAt,
    page: activePage,
    ui_action_id: id,
    label,
    route_action: action,
    metadata,
    note: "前端离线导出兜底文件；后端 API 接入后可替换为真实 artifact/export response。"
  };

  if (action === "export_report" || id.includes("report_export")) {
    downloadTextFile(
      `research_workstation_${activePage}_draft_report.md`,
      [
        "# AI 科研工作站导出草稿",
        "",
        `- 导出时间: ${exportedAt}`,
        `- 页面: ${activePage}`,
        `- 动作: ${id || action}`,
        `- 来源: ${String(metadata?.source ?? "ui")}`,
        "",
        "## 审计边界",
        "- 当前文件为前端草稿导出，用于验证页面导出链路。",
        "- 最终报告仍需要后端 artifacts、claim audit 与人工 Gate。"
      ].join("\n"),
      "text/markdown;charset=utf-8"
    );
    return;
  }

  if (id.includes("csv") || metadata?.format === "csv" || metadata?.format === "ledger") {
    downloadTextFile(
      `research_workstation_${activePage}_export.csv`,
      "field,value\n" +
        Object.entries(basePayload)
          .map(([key, value]) => `"${key}","${String(typeof value === "object" ? JSON.stringify(value) : value).replaceAll('"', '""')}"`)
          .join("\n"),
      "text/csv;charset=utf-8"
    );
    return;
  }

  downloadTextFile(
    `research_workstation_${activePage}_${id || action || "export"}.json`,
    JSON.stringify(basePayload, null, 2),
    "application/json;charset=utf-8"
  );
}

function normalizeActionId(label: string | undefined, componentType: string) {
  const normalized = (label ?? "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 80);
  return normalized ? `${componentType}_${normalized}` : `${componentType}_interaction`;
}

function getComponentType(element: Element) {
  if (element instanceof HTMLButtonElement) return "button";
  if (element instanceof HTMLAnchorElement) return "link";
  if (element instanceof HTMLInputElement) return "input";
  if (element instanceof HTMLTextAreaElement) return "textarea";
  if (element instanceof HTMLSelectElement) return "select";
  if (element instanceof HTMLTableRowElement) return "table_row";
  return element.getAttribute("data-ui-component") ?? element.getAttribute("role") ?? element.tagName.toLowerCase();
}

const uiActionRoutes: Record<string, { page?: PageId; action?: string; metadata?: Record<string, unknown> }> = {
  mission_create_workstation_run: { action: "create_workstation_run", metadata: { source: "mission_control" } },
  mission_open_evidence_ledger: { page: "evidence", action: "navigate_page", metadata: { page: "evidence", source: "mission_control" } },
  mission_prepare_hpc_job: { page: "gpu", action: "prepare_hpc_execution_gate", metadata: { source: "mission_control" } },
  mission_view_workflow_details: { page: "workflow", action: "navigate_page", metadata: { page: "workflow", source: "mission_control" } },
  mission_view_all_evidence: { page: "evidence", action: "navigate_page", metadata: { page: "evidence", source: "mission_control" } },
  mission_view_all_claim_audits: { page: "gates", action: "gate_check_open", metadata: { source: "mission_control" } },
  mission_export_evidence: { page: "evidence", action: "export_audit_bundle", metadata: { source: "mission_control" } },
  mission_build_report_package: { page: "report", action: "generate_teacher_evidence_bundle", metadata: { source: "mission_control" } },
  tasks_open_context: { page: "tasks", action: "view_reproducibility_record", metadata: { source: "task_queue" } },
  tasks_view_agent_logs: { page: "runtime", action: "view_full_log", metadata: { source: "task_queue" } },
  tasks_copy_validation_contract: { page: "tasks", action: "open_validation_review", metadata: { source: "task_queue" } },
  tasks_refresh_resources: { page: "settings", action: "test_all_connectors", metadata: { source: "task_queue" } },
  code_explorer_menu: { page: "code", action: "code_file_select", metadata: { source: "code_agent_ide" } },
  report_add_section: { page: "report", action: "report_section_select", metadata: { source: "report_studio" } },
  report_export_draft_pdf: { page: "report", action: "export_report", metadata: { source: "report_studio", format: "pdf_draft" } },
  design_export_spec: { page: "design", action: "design_sample_action", metadata: { source: "design_system", intent: "export_spec" } },
  design_generate_contract: { page: "design", action: "design_sample_action", metadata: { source: "design_system", intent: "generate_frontend_contract" } },
  design_run_ui_audit: { page: "design", action: "design_sample_action", metadata: { source: "design_system", intent: "run_ui_audit" } },
  design_sync_figma: { page: "design", action: "design_sample_action", metadata: { source: "design_system", intent: "sync_figma", requires: "figma_editor_access" } }
};

const uiActionRoutePatterns: Array<{ prefix: string; route: { page?: PageId; action?: string; metadata?: Record<string, unknown> } }> = [
  { prefix: "code_stage_", route: { page: "code", action: "code_editor_tab_select", metadata: { source: "code_agent_ide", intent: "stage_filter" } } },
  { prefix: "code_filter_", route: { page: "code", action: "code_file_select", metadata: { source: "code_agent_ide", intent: "file_scope_filter" } } },
  { prefix: "open_code_folder_", route: { page: "code", action: "code_file_select", metadata: { source: "code_agent_ide", intent: "open_folder" } } },
  { prefix: "open_code_file_", route: { page: "code", action: "code_file_select", metadata: { source: "code_agent_ide", intent: "open_file" } } },
  { prefix: "ask_code_agent", route: { page: "code", action: "review_agent_patch", metadata: { source: "code_agent_ide", intent: "ask_code_agent" } } },
  { prefix: "review_code_diff", route: { page: "code", action: "review_agent_patch", metadata: { source: "code_agent_ide", intent: "review_diff" } } },
  { prefix: "run_code_smoke_test", route: { page: "code", action: "review_agent_patch", metadata: { source: "code_agent_ide", intent: "run_smoke_test" } } },
  { prefix: "request_code_quality_gate", route: { page: "gates", action: "review_agent_patch", metadata: { source: "code_agent_ide", intent: "request_code_quality_gate" } } },
  { prefix: "view_all_code_agent_trace", route: { page: "runtime", action: "view_full_log", metadata: { source: "code_agent_ide" } } },
  { prefix: "experiments_kpi_", route: { page: "experiments", action: "experiment_select", metadata: { source: "experiment_ledger", intent: "filter_by_kpi" } } },
  { prefix: "experiments_select_node_", route: { page: "experiments", action: "experiment_select", metadata: { source: "experiment_graph", intent: "select_node" } } },
  { prefix: "experiments_layout_graph", route: { page: "experiments", action: "workflow_node_select", metadata: { source: "experiment_graph", intent: "layout" } } },
  { prefix: "experiments_filter_graph", route: { page: "experiments", action: "experiment_select", metadata: { source: "experiment_graph", intent: "filter" } } },
  { prefix: "experiments_zoom_", route: { page: "experiments", action: "experiment_select", metadata: { source: "experiment_graph", intent: "zoom" } } },
  { prefix: "experiments_open_artifact_", route: { page: "evidence", action: "open_artifact_folder", metadata: { source: "experiment_ledger" } } },
  { prefix: "experiments_review_gate", route: { page: "gates", action: "gate_check_open", metadata: { source: "experiment_ledger" } } },
  { prefix: "experiments_open_artifacts", route: { page: "evidence", action: "open_artifact_folder", metadata: { source: "experiment_ledger" } } },
  { prefix: "experiments_launch_branch", route: { page: "workflow", action: "workflow_dry_run", metadata: { source: "experiment_ledger", intent: "launch_branch_preview" } } },
  { prefix: "experiments_filter_ledger", route: { page: "experiments", action: "experiment_select", metadata: { source: "experiment_ledger", intent: "filter_ledger" } } },
  { prefix: "experiments_export_ledger", route: { page: "report", action: "export_report", metadata: { source: "experiment_ledger", format: "ledger" } } },
  { prefix: "experiments_view_", route: { page: "experiments", action: "experiment_select", metadata: { source: "experiment_ledger", intent: "view_experiment" } } },
  { prefix: "experiments_open_folder_", route: { page: "evidence", action: "open_artifact_folder", metadata: { source: "experiment_ledger", intent: "open_experiment_folder" } } },
  { prefix: "gate_row_", route: { page: "gates", action: "gate_check_open", metadata: { source: "integrity_gates", intent: "inspect_gate_row" } } },
  { prefix: "approve_integrity_gate", route: { page: "gates", action: "approve_gate", metadata: { source: "integrity_gates" } } },
  { prefix: "reject_integrity_gate", route: { page: "gates", action: "reject_gate", metadata: { source: "integrity_gates" } } },
  { prefix: "request_gate_revision", route: { page: "gates", action: "reject_gate", metadata: { source: "integrity_gates", decision_note: "revision_requested" } } },
  { prefix: "gpu_view_job_manifest_yaml", route: { page: "gpu", action: "view_full_log", metadata: { source: "gpu_hpc_console", intent: "view_job_manifest" } } },
  { prefix: "runtime_refresh_5s", route: { page: "runtime", action: "runtime_agent_select", metadata: { source: "agent_runtime", intent: "refresh" } } },
  { prefix: "runtime_close_trace_detail", route: { page: "runtime", action: "runtime_agent_select", metadata: { source: "agent_runtime", intent: "close_trace_detail" } } },
  { prefix: "runtime_trace_tab_", route: { page: "runtime", action: "runtime_agent_select", metadata: { source: "agent_runtime", intent: "trace_tab" } } },
  { prefix: "runtime_view_agent_context", route: { page: "runtime", action: "view_reproducibility_record", metadata: { source: "agent_runtime" } } },
  { prefix: "runtime_open_agent_artifact", route: { page: "evidence", action: "open_artifact_folder", metadata: { source: "agent_runtime" } } },
  { prefix: "runtime_request_code_gate", route: { page: "gates", action: "review_agent_patch", metadata: { source: "agent_runtime", intent: "request_code_gate" } } },
  { prefix: "evidence_filter_", route: { page: "evidence", action: "open_artifact_folder", metadata: { source: "evidence_ledger", intent: "filter" } } },
  { prefix: "evidence_date_range", route: { page: "evidence", action: "open_artifact_folder", metadata: { source: "evidence_ledger", intent: "date_range" } } },
  { prefix: "evidence_detail_filter_", route: { page: "evidence", action: "open_artifact_folder", metadata: { source: "evidence_ledger", intent: "detail_filter" } } },
  { prefix: "preview_selected_artifact", route: { page: "evidence", action: "open_artifact_folder", metadata: { source: "evidence_ledger", intent: "preview" } } },
  { prefix: "download_selected_artifact", route: { page: "evidence", action: "export_audit_bundle", metadata: { source: "evidence_ledger", intent: "download_artifact" } } },
  { prefix: "open_selected_artifact_lineage", route: { page: "evidence", action: "view_reproducibility_record", metadata: { source: "evidence_ledger", intent: "lineage" } } },
  { prefix: "reset_evidence_filters", route: { page: "evidence", action: "open_artifact_folder", metadata: { source: "evidence_ledger", intent: "reset_filters" } } },
  { prefix: "apply_evidence_filters", route: { page: "evidence", action: "open_artifact_folder", metadata: { source: "evidence_ledger", intent: "apply_filters" } } },
  { prefix: "open_evidence_lineage", route: { page: "evidence", action: "view_reproducibility_record", metadata: { source: "evidence_ledger", intent: "lineage" } } },
  { prefix: "configure_evidence_columns", route: { page: "evidence", action: "open_artifact_folder", metadata: { source: "evidence_ledger", intent: "configure_columns" } } },
  { prefix: "export_evidence_csv", route: { page: "evidence", action: "export_audit_bundle", metadata: { source: "evidence_ledger", format: "csv" } } },
  { prefix: "export_draft_evidence_bundle", route: { page: "evidence", action: "export_audit_bundle", metadata: { source: "evidence_ledger", format: "draft_bundle" } } },
  { prefix: "filter_evidence_detail", route: { page: "evidence", action: "open_artifact_folder", metadata: { source: "evidence_ledger", intent: "filter_detail" } } },
  { prefix: "save_evidence_view", route: { page: "evidence", action: "open_artifact_folder", metadata: { source: "evidence_ledger", intent: "save_view" } } },
  { prefix: "view_full_lineage", route: { page: "evidence", action: "view_reproducibility_record", metadata: { source: "evidence_ledger", intent: "full_lineage" } } },
  { prefix: "open_lineage_", route: { page: "evidence", action: "view_reproducibility_record", metadata: { source: "evidence_ledger", intent: "lineage_node" } } },
  { prefix: "add_evidence_annotation", route: { page: "evidence", action: "add_evidence", metadata: { source: "evidence_ledger", intent: "annotation" } } },
  { prefix: "expand_artifact_detail", route: { page: "evidence", action: "open_artifact_folder", metadata: { source: "evidence_ledger", intent: "expand_detail" } } },
  { prefix: "close_artifact_detail", route: { page: "evidence", action: "open_artifact_folder", metadata: { source: "evidence_ledger", intent: "close_detail" } } },
  { prefix: "preview_evidence_artifact", route: { page: "evidence", action: "open_artifact_folder", metadata: { source: "evidence_ledger", intent: "preview_artifact" } } },
  { prefix: "download_evidence_artifact", route: { page: "evidence", action: "export_audit_bundle", metadata: { source: "evidence_ledger", intent: "download_artifact" } } },
  { prefix: "open_evidence_menu", route: { page: "evidence", action: "open_artifact_folder", metadata: { source: "evidence_ledger", intent: "artifact_menu" } } },
  { prefix: "artifact_detail_tab_", route: { page: "evidence", action: "open_artifact_folder", metadata: { source: "evidence_ledger", intent: "detail_tab" } } },
  { prefix: "copy_artifact_preview", route: { page: "evidence", action: "open_artifact_folder", metadata: { source: "evidence_ledger", intent: "copy_preview" } } },
  { prefix: "open_artifact", route: { page: "evidence", action: "open_artifact_folder", metadata: { source: "evidence_ledger", intent: "open_artifact" } } },
  { prefix: "copy_artifact_path", route: { page: "evidence", action: "open_artifact_folder", metadata: { source: "evidence_ledger", intent: "copy_path" } } },
  { prefix: "trace_artifact_claim", route: { page: "evidence", action: "view_reproducibility_record", metadata: { source: "evidence_ledger", intent: "trace_claim" } } },
  { prefix: "export_artifact_evidence", route: { page: "evidence", action: "export_audit_bundle", metadata: { source: "evidence_ledger", intent: "export_artifact_evidence" } } },
  { prefix: "literature_refresh_library", route: { page: "literature", action: "view_reproducibility_record", metadata: { source: "literature_rag", intent: "refresh_library" } } },
  { prefix: "literature_filter_", route: { page: "literature", action: "view_reproducibility_record", metadata: { source: "literature_rag", intent: "filter" } } },
  { prefix: "literature_apply_filters", route: { page: "literature", action: "view_reproducibility_record", metadata: { source: "literature_rag", intent: "apply_filters" } } },
  { prefix: "rag_build_agent_context", route: { page: "literature", action: "view_reproducibility_record", metadata: { source: "literature_rag", intent: "build_agent_context" } } },
  { prefix: "rag_send_research_agent", route: { page: "control", action: "view_reproducibility_record", metadata: { source: "literature_rag", intent: "send_research_agent" } } },
  { prefix: "rag_send_code_agent", route: { page: "code", action: "view_reproducibility_record", metadata: { source: "literature_rag", intent: "send_code_agent" } } },
  { prefix: "rag_bind_report_claim", route: { page: "report", action: "report_section_select", metadata: { source: "literature_rag", intent: "bind_report_claim" } } },
  { prefix: "rag_request_citation_audit", route: { page: "gates", action: "gate_check_open", metadata: { source: "literature_rag", intent: "citation_audit" } } },
  { prefix: "rag_refresh_index", route: { page: "literature", action: "view_reproducibility_record", metadata: { source: "literature_rag", intent: "refresh_index" } } },
  { prefix: "rag_open_all_logs", route: { page: "runtime", action: "view_full_log", metadata: { source: "literature_rag", intent: "open_logs" } } },
  { prefix: "settings_language_zh_cn", route: { page: "settings", action: "language_select", metadata: { source: "settings", language: "zh-CN" } } },
  { prefix: "settings_language_en_us", route: { page: "settings", action: "language_select", metadata: { source: "settings", language: "en-US" } } },
  { prefix: "settings_theme_light", route: { page: "settings", action: "settings_theme_change", metadata: { source: "settings", theme: "light" } } },
  { prefix: "settings_theme_dark", route: { page: "settings", action: "settings_theme_change", metadata: { source: "settings", theme: "dark" } } },
  { prefix: "open_settings_section_", route: { page: "settings", action: "open_settings_section", metadata: { source: "settings" } } },
  { prefix: "design_component_", route: { page: "design", action: "design_sample_action", metadata: { source: "design_system", intent: "component_demo" } } },
  { prefix: "design_metric_", route: { page: "design", action: "design_sample_action", metadata: { source: "design_system", intent: "metric_demo" } } },
  { prefix: "open_connector_settings", route: { page: "settings", action: "open_settings_section", metadata: { source: "sidebar", section: "connectors" } } },
  { prefix: "open_research_mode_gates", route: { page: "gates", action: "gate_check_open", metadata: { source: "sidebar" } } },
  { prefix: "toggle_mobile_navigation", route: { page: "overview", action: "navigate_page", metadata: { source: "sidebar", intent: "toggle_mobile_navigation" } } },
  { prefix: "open_current_workspace", route: { page: "overview", action: "quick_open_workspace", metadata: { source: "topbar" } } },
  { prefix: "global_search_input", route: { page: "overview", action: "search_command", metadata: { source: "topbar" } } },
  { prefix: "topbar_open_", route: { action: "quick_open_workspace", metadata: { source: "topbar" } } },
  { prefix: "open_notifications", route: { page: "runtime", action: "notification_open", metadata: { source: "topbar" } } },
  { prefix: "open_user_menu", route: { page: "settings", action: "profile_open", metadata: { source: "topbar" } } },
  { prefix: "select_research_workspace", route: { page: "overview", action: "workspace_select", metadata: { source: "mission_control" } } },
  { prefix: "filter_", route: { page: "overview", action: "search_command", metadata: { source: "filter_panel" } } },
  { prefix: "apply_filters", route: { page: "overview", action: "search_command", metadata: { source: "filter_panel", intent: "apply_filters" } } },
  { prefix: "refresh_filtered_view", route: { page: "overview", action: "search_command", metadata: { source: "filter_panel", intent: "refresh" } } },
  { prefix: "open_table_row", route: { page: "evidence", action: "open_artifact_folder", metadata: { source: "table" } } },
  { prefix: "control_open_attachment_", route: { page: "control", action: "view_reproducibility_record", metadata: { source: "ai_control" } } },
  { prefix: "control_run_through_workstation", route: { page: "control", action: "create_workstation_run", metadata: { source: "ai_control" } } },
  { prefix: "legacy_control_", route: { page: "control", action: "view_reproducibility_record", metadata: { source: "legacy_control" } } },
  { prefix: "blocked_", route: { page: "gates", action: "gate_check_open", metadata: { source: "blocked_control", blocked: true } } },
  { prefix: "tasks_select_", route: { page: "tasks", action: "task_select", metadata: { source: "task_queue" } } },
  { prefix: "tasks_agent_", route: { page: "runtime", action: "runtime_agent_select", metadata: { source: "task_queue" } } },
  { prefix: "tasks_refresh_queue", route: { page: "tasks", action: "task_select", metadata: { source: "task_queue", intent: "refresh_queue" } } },
  { prefix: "tasks_prev_page", route: { page: "tasks", action: "task_select", metadata: { source: "task_queue", intent: "prev_page" } } },
  { prefix: "tasks_next_page", route: { page: "tasks", action: "task_select", metadata: { source: "task_queue", intent: "next_page" } } },
  { prefix: "tasks_create_workstation_run", route: { page: "tasks", action: "create_workstation_run", metadata: { source: "task_queue" } } },
  { prefix: "tasks_dispatch_agents", route: { page: "workflow", action: "workflow_dry_run", metadata: { source: "task_queue", intent: "dispatch_agents_preview" } } },
  { prefix: "tasks_contract_", route: { page: "tasks", action: "open_validation_review", metadata: { source: "task_queue", intent: "contract_detail" } } },
  { prefix: "tasks_resource_", route: { page: "settings", action: "test_all_connectors", metadata: { source: "task_queue", intent: "resource_detail" } } },
  { prefix: "tasks_route_validation_agent", route: { page: "runtime", action: "runtime_agent_select", metadata: { source: "task_queue", intent: "route_validation_agent" } } }
];

function resolveUiActionRoute(actionId: string | undefined) {
  if (!actionId) return undefined;
  if (actionId.startsWith("topbar_open_")) {
    const page = actionId.replace("topbar_open_", "") as PageId;
    if ((pageIds as readonly string[]).includes(page)) {
      return { page, action: "quick_open_workspace", metadata: { source: "topbar", page } };
    }
  }
  return uiActionRoutes[actionId] ?? uiActionRoutePatterns.find((item) => actionId.startsWith(item.prefix))?.route;
}

const pageTitles: Record<PageId, { zh: string; en: string; subZh: string; subEn: string }> = {
  overview: { zh: "科研总览", en: "Research Overview", subZh: "科研工作站运行态势与闭环总控", subEn: "Research workstation operating status and mission control" },
  control: { zh: "AI 控制台", en: "AI Control", subZh: "调度 Agent、资源与门禁", subEn: "Orchestrate agents, resources, and gates" },
  experiments: { zh: "实验中心", en: "Experiment Ledger", subZh: "实验台账、分支与分数门禁", subEn: "Experiment ledger, branches, and score gates" },
  evolution: { zh: "自进化引擎", en: "Evolution Engine", subZh: "搜索图、检索记忆与分支扩展规划", subEn: "Search graph, retrospective memory, and branch expansion planning" },
  data: { zh: "数据 / Kaggle", en: "Data / Kaggle", subZh: "数据审计、提交结构与排行榜证据", subEn: "Data audit, submission schema, and leaderboard evidence" },
  report: { zh: "报告工作室", en: "Report Studio", subZh: "AI 生成报告、证据链与风险审计", subEn: "AI-generated reports with evidence and risk audit" },
  code: { zh: "代码 Agent IDE", en: "Code Agent IDE", subZh: "可审计代码生成、Diff、终端与门禁", subEn: "Auditable code generation, diff, terminal, and gates" },
  gpu: { zh: "GPU / HPC", en: "GPU / HPC", subZh: "远程算力、作业 manifest 与产物回传", subEn: "Remote compute, job manifests, and artifact pullback" },
  evidence: { zh: "证据台账", en: "Evidence Ledger", subZh: "统一归档 artifact、日志、指标、报告和审计证据", subEn: "Artifacts, logs, metrics, reports, and audit evidence" },
  gates: { zh: "完整性 Gate", en: "Integrity Gates", subZh: "代码、算力、提交与报告门禁", subEn: "Code, compute, submission, and report gates" },
  literature: { zh: "文献 / RAG", en: "Literature / RAG", subZh: "文献检索、RAG 资料库与研究上下文", subEn: "Literature retrieval, RAG, and research context" },
  tasks: { zh: "任务队列", en: "Task Queue", subZh: "任务配置、上下文与运行入口", subEn: "Task specs, context, and run entry points" },
  runtime: { zh: "Agent 运行时", en: "Agent Runtime", subZh: "模型、缓存、成本与调度状态", subEn: "Models, cache, cost, and scheduler status" },
  workflow: { zh: "流程编排", en: "Workflow Graph", subZh: "多 Agent 工作流、回退与证据交接", subEn: "Multi-agent workflow, fallback, and evidence handoff" },
  settings: { zh: "系统设置", en: "Settings", subZh: "账号、语言、主题、凭据与资源偏好", subEn: "Account, language, theme, credentials, and resources" },
  design: { zh: "设计系统", en: "Design System", subZh: "Research OS 视觉 token、组件规则与页面基线", subEn: "Visual tokens, components, and page baselines" }
};

export function AppShell({
  activePage,
  onPageChange,
  onAction,
  locale = "zh-CN",
  children
}: {
  activePage: PageId;
  onPageChange: (page: PageId) => void;
  onAction?: (action: string, metadata?: Record<string, unknown>) => Promise<unknown>;
  locale?: Locale;
  children: React.ReactNode;
}) {
  function handleUiClick(event: MouseEvent<HTMLDivElement>) {
    if (!onAction) return;
    const target = event.target instanceof Element ? event.target.closest(interactiveSelector) : null;
    if (!target || target.closest("[data-ui-skip-action='true']")) return;
    const label = cleanText(target.getAttribute("aria-label") ?? target.textContent);
    const componentType = getComponentType(target);
    const metadata: UiComponentClickMetadata = {
      page: activePage,
      component_type: componentType,
      action_id: target.getAttribute("data-ui-action") ?? target.getAttribute("data-testid") ?? normalizeActionId(label, componentType),
      label: label || undefined,
      href: target instanceof HTMLAnchorElement ? target.href : undefined,
      disabled:
        target instanceof HTMLButtonElement ||
        target instanceof HTMLInputElement ||
        target instanceof HTMLSelectElement ||
        target instanceof HTMLTextAreaElement
          ? target.disabled || target.getAttribute("aria-disabled") === "true"
          : target.getAttribute("aria-disabled") === "true"
    };
    const route = resolveUiActionRoute(String(metadata.action_id ?? ""));
    if (route && !metadata.disabled) {
      if (route.page) onPageChange(route.page);
      if (route.action) {
        exportFallbackForUiAction({
          action: route.action,
          actionId: String(metadata.action_id ?? ""),
          activePage,
          label: metadata.label,
          metadata: route.metadata
        });
        void onAction(route.action, {
          ...route.metadata,
          ui_action_id: metadata.action_id,
          ui_label: metadata.label,
          from_page: activePage,
          target_page: route.page
        }).catch(() => undefined);
      }
      return;
    }
    void onAction("ui_component_click", metadata).catch(() => undefined);
  }

  return (
    <div className="workstation-chrome min-h-screen overflow-x-hidden bg-[#f6f8fb] text-slate-950">
      <Sidebar activePage={activePage} onPageChange={onPageChange} onAction={onAction} locale={locale} />
      <div className="min-w-0 lg:pl-[224px]">
        <Topbar activePage={activePage} onPageChange={onPageChange} onAction={onAction} locale={locale} />
        <main className="min-h-screen min-w-0 px-2 pb-5 pt-2 sm:px-3 lg:h-screen lg:overflow-hidden lg:px-3 lg:pb-0 lg:pt-[72px]">
          <div
            className="thin-scrollbar mx-auto w-full max-w-[1672px] lg:h-[calc(100vh-76px)] lg:overflow-y-auto lg:pr-1"
            data-ui-component="workstation-page"
            data-ui-page={activePage}
            onClickCapture={handleUiClick}
          >
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}

function Topbar({
  activePage,
  onPageChange,
  onAction,
  locale
}: {
  activePage: PageId;
  onPageChange: (page: PageId) => void;
  onAction?: (action: string, metadata?: Record<string, unknown>) => Promise<unknown>;
  locale?: Locale;
}) {
  const shortcuts = [
    { id: "control", icon: Bot, label: copy(locale, "AI Control", "AI 控制") },
    { id: "code", icon: Code2, label: copy(locale, "Code", "代码") },
    { id: "report", icon: FileText, label: copy(locale, "Report", "报告") },
    { id: "runtime", icon: Timer, label: "Agent" },
    { id: "gpu", icon: Server, label: "GPU" },
    { id: "gates", icon: ShieldCheck, label: "Gate" },
    { id: "evidence", icon: Database, label: copy(locale, "Evidence", "证据") }
  ] as const;
  const title = pageTitles[activePage] ?? pageTitles.overview;

  return (
    <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/96 px-3 py-2 shadow-[0_1px_0_rgba(15,23,42,0.04)] backdrop-blur lg:fixed lg:left-[224px] lg:right-0 lg:h-[64px] lg:px-3 lg:py-0">
      <div className="mx-auto flex h-full max-w-[1672px] flex-wrap items-center gap-2 xl:flex-nowrap">
        <button
          className="hidden min-w-[292px] shrink-0 text-left lg:block"
          data-ui-action="open_current_workspace"
          data-ui-skip-action="true"
          onClick={() => {
            onPageChange(activePage);
            void onAction?.("quick_open_workspace", { page: activePage });
          }}
        >
          <span className="block truncate text-[20px] font-black leading-6 text-slate-950">
            {copy(locale, title.en, title.zh)}
          </span>
          <span className="block truncate text-[11px] font-semibold leading-4 text-slate-500">
            {copy(locale, title.subEn, title.subZh)}
          </span>
        </button>

        <label className="relative min-w-[230px] flex-1 xl:max-w-[430px]">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
          <input
            className="h-9 w-full rounded-md border border-slate-200 bg-slate-50 pl-10 pr-16 text-sm text-slate-700 outline-none transition focus:border-blue-500 focus:bg-white focus:ring-2 focus:ring-blue-100"
            data-ui-action="global_search_input"
            placeholder={copy(locale, "Search runs, artifacts, datasets, agents...", "搜索 run、artifact、数据集、Agent...")}
            onKeyDown={(event) => {
              if (event.key === "Enter") void onAction?.("search_command", { query: event.currentTarget.value });
            }}
          />
          <span className="absolute right-3 top-1/2 -translate-y-1/2 rounded border border-slate-200 bg-white px-1.5 py-0.5 text-[10px] font-bold text-slate-500">
            Ctrl K
          </span>
        </label>

        <div className="flex w-full gap-1 overflow-x-auto pb-1 sm:w-auto sm:pb-0 xl:flex-1 xl:justify-end">
          {shortcuts.map((item) => {
            const Icon = item.icon;
            const active = activePage === item.id;
            return (
              <button
                key={item.id}
                className={cn(
                  "flex h-8 shrink-0 items-center gap-1.5 rounded-md border px-2 text-xs font-black transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500",
                  active ? "border-blue-200 bg-blue-50 text-blue-700" : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                )}
                data-ui-action={`topbar_open_${item.id}`}
                data-ui-skip-action="true"
                onClick={() => {
                  onPageChange(item.id as PageId);
                  void onAction?.("quick_open_workspace", { page: item.id });
                }}
              >
                <Icon className="h-4 w-4" />
                {item.label}
              </button>
            );
          })}
        </div>

        <div className="ml-auto hidden items-center gap-2 sm:flex">
          <button className="relative flex h-9 w-9 items-center justify-center rounded-md border border-slate-200 text-slate-700 hover:bg-slate-50" aria-label="Notifications" data-ui-action="open_notifications">
            <Bell className="h-4 w-4" />
            <span className="absolute -right-1 -top-1 flex h-4 w-4 items-center justify-center rounded-full bg-red-600 text-[10px] font-black text-white">3</span>
          </button>
          <button className="flex items-center gap-2 rounded-md px-1.5 py-1 text-left hover:bg-slate-50" data-ui-action="open_user_menu">
            <span className="flex h-8 w-8 items-center justify-center rounded-full bg-blue-600 text-sm font-black text-white">RA</span>
            <span className="hidden sm:block">
              <span className="block text-sm font-black text-slate-950">{copy(locale, "Research Admin", "科研管理员")}</span>
              <span className="block text-xs text-slate-500">{copy(locale, "Owner", "负责人")}</span>
            </span>
            <ChevronDown className="hidden h-4 w-4 text-slate-400 sm:block" />
          </button>
        </div>
      </div>
    </header>
  );
}
