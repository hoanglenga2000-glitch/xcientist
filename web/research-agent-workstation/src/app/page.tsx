"use client";

import { useEffect, useState } from "react";
import { AppShell } from "@/components/workstation/AppShell";
import { AiControlConsole } from "@/components/workstation/AiControlConsole";
import { EvolutionConsole } from "@/components/workstation/EvolutionConsole";
import * as api from "@/lib/api/client";
import type { WorkstationSummary } from "@/lib/api/types";
import {
  AgentRuntime,
  CodeRunner,
  DataKagglePipeline,
  DesignSystem,
  EvidenceLedger,
  Experiments,
  GpuHpcConsole,
  IntegrityGates,
  LiteratureKnowledge,
  OverviewBoardEnhanced,
  ReportStudio,
  ResearchTasks,
  SettingsCenter,
  WorkflowGraph
} from "@/components/workstation/Screens";
import type { PageId } from "@/components/workstation/navigation";

type Locale = "zh-CN" | "en-US";

const pageIds = [
  "tasks",
  "data",
  "gpu",
  "evidence",
  "literature",
  "workflow",
  "code",
  "runtime",
  "experiments",
  "evolution",
  "report",
  "gates",
  "settings",
  "design",
  "overview",
  "control"
] as const satisfies PageId[];

function parsePageId(value: string | null | undefined): PageId | null {
  if (!value) return null;
  const normalized = value.replace(/^#/, "").trim();
  if (normalized === "mission") return "overview";
  if (normalized === "evidence-detail") return "evidence";
  if (normalized === "design") return "settings";
  return pageIds.includes(normalized as PageId) ? normalized as PageId : null;
}

function pageFromLocation(): PageId {
  if (typeof window === "undefined") return "overview";
  const url = new URL(window.location.href);
  return parsePageId(url.searchParams.get("page")) ?? parsePageId(url.hash) ?? "overview";
}

function normalizeTaskId(taskId: string) {
  return taskId === "house-prices" ? "house_prices" : taskId;
}

function text(locale: Locale, zh: string, en: string) {
  return locale === "zh-CN" ? zh : en;
}

type HomeProps = {
  searchParams?: {
    page?: string;
  };
};

export default function Home({ searchParams }: HomeProps) {
  const [activePage, setActivePage] = useState<PageId>(() => parsePageId(searchParams?.page) ?? "overview");
  const [selectedTask, setSelectedTask] = useState("playground_series_s6e6");
  const [selectedStage, setSelectedStage] = useState("stage-7");
  const [selectedExperiment, setSelectedExperiment] = useState("exp_20250606_192030");
  const [gateStatus, setGateStatus] = useState<"Pending" | "Approved" | "Rejected">("Pending");
  const [patchApplied, setPatchApplied] = useState(false);
  const [reportSubmitted, setReportSubmitted] = useState(false);
  const [summary, setSummary] = useState<WorkstationSummary | null>(null);
  const [locale, setLocale] = useState<Locale>("zh-CN");
  const [lastActionTrace, setLastActionTrace] = useState<{
    action: string;
    taskId?: string;
    request?: Record<string, unknown>;
    response?: Record<string, unknown>;
    message: string;
    artifact?: string | null;
    at: string;
  } | null>(null);
  const [runState, setRunState] = useState<{
    status: "idle" | "running" | "passed" | "failed";
    message: string;
    experimentDir?: string;
  }>({ status: "idle", message: "工作站运行器已就绪。" });
  const [agentActionMessage, setAgentActionMessage] = useState("外部 Code Agent 网关已就绪。");
  const [systemActionMessage, setSystemActionMessage] = useState("系统动作已就绪。");

  async function refreshSummary() {
    const payload = await api.getWorkstationSummary();
    setSummary(payload);
    return payload;
  }

  useEffect(() => {
    refreshSummary().catch(() => {
      setRunState({ status: "failed", message: "无法加载工作站摘要。" });
    });
    api.getSettings()
      .then((payload) => {
        const uiLanguage = payload.settings?.language?.ui_language;
        if (uiLanguage === "en-US" || uiLanguage === "zh-CN") setLocale(uiLanguage);
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    const applyLocationPage = () => {
      const nextPage = pageFromLocation();
      setActivePage((current) => (current === nextPage ? current : nextPage));
    };
    applyLocationPage();
    window.addEventListener("popstate", applyLocationPage);
    window.addEventListener("hashchange", applyLocationPage);
    return () => {
      window.removeEventListener("popstate", applyLocationPage);
      window.removeEventListener("hashchange", applyLocationPage);
    };
  }, []);

  function changeActivePage(page: PageId) {
    setActivePage(page);
    if (typeof window === "undefined") return;
    const url = new URL(window.location.href);
    url.searchParams.set("page", page);
    url.hash = "";
    window.history.replaceState(null, "", url);
  }

  async function runLocalExperiment(taskId = selectedTask) {
    const normalized = normalizeTaskId(taskId);
    setRunState({ status: "running", message: `正在通过工作站运行任务：${normalized}...` });
    setLastActionTrace({
      action: "run_local_experiment",
      taskId: normalized,
      request: { task_id: normalized },
      message: "实验请求已发送到工作站后端。",
      at: new Date().toISOString()
    });
    try {
      const payload = await api.runLocalExperiment(normalized);
      setLastActionTrace({
        action: "run_local_experiment",
        taskId: normalized,
        request: { task_id: normalized },
        response: payload as unknown as Record<string, unknown>,
        message: `实验${payload.ok ? "已受理" : "已返回"}：${payload.experiment_dir ?? payload.run_id ?? normalized}`,
        artifact: payload.experiment_dir,
        at: new Date().toISOString()
      });
      setSummary(payload.summary ?? null);
      setRunState({
        status: "passed",
        message: `实验已记录到工作站证据链：${payload.experiment_dir ?? "已写入数据库"}`,
        experimentDir: payload.experiment_dir
      });
    } catch (error) {
      setRunState({
        status: "failed",
        message: error instanceof Error ? error.message : "实验运行失败。"
      });
    }
  }

  async function exportCodeAgentContext(taskId = "house_prices", targetAgent = "claude_code") {
    const payload = await api.exportCodeAgentContext(taskId, targetAgent);
    setAgentActionMessage(`已导出 ${payload.target_agent ?? targetAgent} 的上下文：${payload.context_dir}`);
    setLastActionTrace({
      action: "export_code_agent_context",
      taskId,
      request: { task_id: taskId, target_agent: targetAgent },
      response: payload as unknown as Record<string, unknown>,
      message: `Code Agent 上下文已导出：${payload.context_dir}`,
      artifact: payload.context_dir,
      at: new Date().toISOString()
    });
  }

  async function importDemoPatch(taskId = "house_prices") {
    const payload = await api.importAgentPatch(taskId, {
      source_agent: "codex",
      patch_diff: "diff --git a/workspace_note.md b/workspace_note.md\n+Imported from frontend patch queue demo."
    });
    setPatchApplied(true);
    setAgentActionMessage(`补丁已导入：${payload.patch_path}`);
    setLastActionTrace({
      action: "import_agent_patch",
      taskId,
      request: { task_id: taskId, source_agent: "codex" },
      response: payload as unknown as Record<string, unknown>,
      message: `补丁已导入：${payload.patch_path}`,
      artifact: payload.patch_path,
      at: new Date().toISOString()
    });
  }

  async function runWorkstationAction(action: string, metadata?: Record<string, unknown>) {
    setSystemActionMessage(`正在执行动作：${action}...`);
    try {
      const requestedTask = typeof metadata?.task_id === "string" ? metadata.task_id : selectedTask;
      const taskId = normalizeTaskId(requestedTask);
      setLastActionTrace({
        action,
        taskId,
        request: { action, task_id: taskId, metadata: metadata ?? {} },
        message: `动作已发送：${action}`,
        at: new Date().toISOString()
      });
      const payload = await api.runWorkstationAction(action, taskId, metadata);
      setSystemActionMessage(`${payload.message}${payload.artifact ? ` (${payload.artifact})` : ""}`);
      setLastActionTrace({
        action,
        taskId,
        request: { action, task_id: taskId, metadata: metadata ?? {} },
        response: payload as unknown as Record<string, unknown>,
        message: payload.message,
        artifact: payload.artifact,
        at: new Date().toISOString()
      });
      if (action.includes("approve")) setGateStatus("Approved");
      if (action.includes("reject")) setGateStatus("Rejected");
      if (action === "language_select" && (metadata?.language === "zh-CN" || metadata?.language === "en-US")) {
        setLocale(metadata.language);
      }
      if (action === "submit_report_review") setReportSubmitted(true);
      if (action === "create_task" && typeof payload.task_id === "string") {
        setSelectedTask(payload.task_id);
        setActivePage("overview");
        await refreshSummary();
        const configPath = typeof payload.config_path === "string" ? payload.config_path : "configs/generated";
        setSystemActionMessage(
          text(
            locale,
            `新任务已创建并可训练：${payload.task_id}，配置文件：${configPath}`,
            `Runnable task created: ${payload.task_id}; config: ${configPath}`
          )
        );
      } else {
        void refreshSummary();
      }
      return payload;
    } catch (error) {
      const message = error instanceof Error ? error.message : `动作执行失败：${action}`;
      setSystemActionMessage(message);
      throw error;
    }
  }

  const screenProps = {
    selectedTask,
    setSelectedTask,
    selectedStage,
    setSelectedStage,
    selectedExperiment,
    setSelectedExperiment,
    gateStatus,
    setGateStatus,
    patchApplied,
    setPatchApplied,
    reportSubmitted,
    setReportSubmitted,
    summary,
    refreshSummary,
    runLocalExperiment,
    runState,
    exportCodeAgentContext,
    importDemoPatch,
    agentActionMessage,
    runWorkstationAction,
    systemActionMessage,
    locale,
    setLocale,
    lastActionTrace
  };

  return (
    <AppShell activePage={activePage} onPageChange={changeActivePage} onAction={runWorkstationAction} locale={locale}>
      {activePage === "tasks" && <ResearchTasks {...screenProps} />}
      {activePage === "data" && <DataKagglePipeline {...screenProps} />}
      {activePage === "gpu" && <GpuHpcConsole {...screenProps} />}
      {activePage === "evidence" && <EvidenceLedger {...screenProps} />}
      {activePage === "literature" && <LiteratureKnowledge {...screenProps} />}
      {activePage === "workflow" && <WorkflowGraph {...screenProps} />}
      {activePage === "code" && <CodeRunner {...screenProps} />}
      {activePage === "runtime" && <AgentRuntime {...screenProps} />}
      {activePage === "experiments" && <Experiments {...screenProps} />}
      {activePage === "evolution" && (
        <EvolutionConsole selectedTask={screenProps.selectedTask} refreshSummary={screenProps.refreshSummary} />
      )}
      {activePage === "report" && <ReportStudio {...screenProps} />}
      {activePage === "gates" && <IntegrityGates {...screenProps} />}
      {activePage === "settings" && <SettingsCenter {...screenProps} />}
      {activePage === "design" && <DesignSystem {...screenProps} />}
      {activePage === "overview" && <OverviewBoardEnhanced {...screenProps} />}
      {activePage === "control" && <AiControlConsole {...screenProps} />}
    </AppShell>
  );
}
