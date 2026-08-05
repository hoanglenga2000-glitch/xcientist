"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { useSearchParams } from "next/navigation";
import { AppShell } from "@/components/workstation/AppShell";

const AiControlConsole = dynamic(() => import("@/components/workstation/AiControlConsole").then((module) => module.AiControlConsole), { ssr: false });
const EvolutionConsole = dynamic(() => import("@/components/workstation/EvolutionConsole").then((module) => module.EvolutionConsole), { ssr: false });
const CodeAgentScreen = dynamic(() => import("@/components/workstation/screens/CodeAgentScreen").then((module) => module.CodeAgentScreen), { ssr: false });
const DataKaggleScreen = dynamic(() => import("@/components/workstation/screens/DataKaggleScreen").then((module) => module.DataKaggleScreen), { ssr: false });
const EvidenceLedgerScreen = dynamic(() => import("@/components/workstation/screens/EvidenceLedgerScreen").then((module) => module.EvidenceLedgerScreen), { ssr: false });
const ExperimentsScreen = dynamic(() => import("@/components/workstation/screens/ExperimentsScreen").then((module) => module.ExperimentsScreen), { ssr: false });
const GatesScreen = dynamic(() => import("@/components/workstation/screens/GatesScreen").then((module) => module.GatesScreen), { ssr: false });
const GpuHpcScreen = dynamic(() => import("@/components/workstation/screens/GpuHpcScreen").then((module) => module.GpuHpcScreen), { ssr: false });
const LiteratureScreen = dynamic(() => import("@/components/workstation/screens/LiteratureScreen").then((module) => module.LiteratureScreen), { ssr: false });
const OverviewScreen = dynamic(() => import("@/components/workstation/screens/OverviewScreen").then((module) => module.OverviewScreen), { ssr: false });
const ReportStudioScreen = dynamic(() => import("@/components/workstation/screens/ReportStudioScreen").then((module) => module.ReportStudioScreen), { ssr: false });
const RuntimeScreen = dynamic(() => import("@/components/workstation/screens/RuntimeScreen").then((module) => module.RuntimeScreen), { ssr: false });
const SettingsScreen = dynamic(() => import("@/components/workstation/screens/SettingsScreen").then((module) => module.SettingsScreen), { ssr: false });
const TasksScreen = dynamic(() => import("@/components/workstation/screens/TasksScreen").then((module) => module.TasksScreen), { ssr: false });
const WorkflowScreen = dynamic(() => import("@/components/workstation/screens/WorkflowScreen").then((module) => module.WorkflowScreen), { ssr: false });
const AssistantScreen = dynamic(() => import("@/components/workstation/screens/AssistantScreen").then((module) => module.AssistantScreen), { ssr: false });
import * as api from "@/lib/api/client";
import type { WorkstationSummary } from "@/lib/api/types";
import { resolvePageId, type PageId } from "@/components/workstation/navigation";
import { latestTaskSignal, normalizeTaskId } from "@/lib/task-context";

type Locale = "zh-CN" | "en-US";

const pageIds = [
  "assistant",
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
  if (normalized === "design") return "settings";
  const resolved = resolvePageId(normalized);
  return pageIds.includes(resolved) ? resolved : null;
}

function pageFromLocation(): PageId {
  if (typeof window === "undefined") return "assistant";
  const url = new URL(window.location.href);
  return parsePageId(url.searchParams.get("page")) ?? parsePageId(url.hash) ?? "assistant";
}

function text(locale: Locale, zh: string, en: string) {
  return locale === "zh-CN" ? zh : en;
}

function HomeClient() {
  const searchParams = useSearchParams();
  const [activePage, setActivePage] = useState<PageId>(() => parsePageId(searchParams?.get("page")) ?? "assistant");
  const [selectedTask, setSelectedTask] = useState(() => normalizeTaskId(searchParams?.get("task")) || "playground_series_s6e6");
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
  const refreshInFlightRef = useRef<{ page: PageId; promise: Promise<WorkstationSummary> } | null>(null);
  const mountedRef = useRef(true);
  const latestAppliedTaskSignalRef = useRef<string | null>(null);
  const explicitTaskFromUrlRef = useRef(Boolean(searchParams?.get("task")));

  const writeTaskToLocation = useCallback((taskId: string) => {
    if (typeof window === "undefined") return;
    const url = new URL(window.location.href);
    url.searchParams.set("task", taskId);
    window.history.replaceState(null, "", url);
  }, []);

  const applySummaryPayload = useCallback((payload: WorkstationSummary) => {
    if (!mountedRef.current) return;

    const signal = latestTaskSignal(payload);
    if (signal && latestAppliedTaskSignalRef.current !== signal.key) {
      const isInitialSignal = latestAppliedTaskSignalRef.current === null;
      latestAppliedTaskSignalRef.current = signal.key;
      if (!(isInitialSignal && explicitTaskFromUrlRef.current)) {
        explicitTaskFromUrlRef.current = false;
        setSelectedTask((current) => (current === signal.taskId ? current : signal.taskId));
        writeTaskToLocation(signal.taskId);
      }
    }

    // React batches the task selection and summary update, so screens never render a
    // fresh terminal summary against the previous task while auto-sync is settling.
    const payloadMode = (payload as WorkstationSummary & { _meta?: { mode?: string } })._meta?.mode;
    if (payloadMode === "detail") {
      setSummary(payload);
    } else {
      // Action responses may carry only the lightweight projection; merge those
      // fields so the active page keeps its already loaded detail slice.
      setSummary((current) => ({ ...current, ...payload }));
    }
  }, [writeTaskToLocation]);

  const selectTask = useCallback((taskId: string) => {
    const normalized = normalizeTaskId(taskId);
    if (!normalized) return;
    explicitTaskFromUrlRef.current = true;
    setSelectedTask(normalized);
    writeTaskToLocation(normalized);
  }, [writeTaskToLocation]);

  const refreshSummary = useCallback(async (force = false) => {
    if (!force && refreshInFlightRef.current?.page === activePage) return refreshInFlightRef.current.promise;
    const request = api.getWorkstationSummary(activePage, force).then((payload) => {
      applySummaryPayload(payload);
      return payload;
    });
    refreshInFlightRef.current = { page: activePage, promise: request };
    try {
      return await request;
    } finally {
      if (refreshInFlightRef.current?.promise === request) refreshInFlightRef.current = null;
    }
  }, [activePage, applySummaryPayload]);

  useEffect(() => {
    mountedRef.current = true;
    refreshSummary().catch(() => {
      setRunState({ status: "failed", message: "无法加载工作站摘要。" });
    });
    api.getSettings()
      .then((payload) => {
        const uiLanguage = payload.settings?.language?.ui_language;
        if (uiLanguage === "en-US" || uiLanguage === "zh-CN") setLocale(uiLanguage);
      })
      .catch(() => undefined);
    const refreshWhenVisible = () => {
      if (document.visibilityState === "visible") void refreshSummary().catch(() => undefined);
    };
    // Detailed projections are cached server-side and refreshed at a human-scale
    // cadence. Visibility changes trigger an immediate catch-up without polling a
    // hidden tab or rebuilding a multi-megabyte object every 2.5 seconds.
    const timer = window.setInterval(refreshWhenVisible, 30_000);
    document.addEventListener("visibilitychange", refreshWhenVisible);
    window.addEventListener("focus", refreshWhenVisible);
    return () => {
      mountedRef.current = false;
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", refreshWhenVisible);
      window.removeEventListener("focus", refreshWhenVisible);
    };
  }, [refreshSummary]);

  useEffect(() => {
    document.documentElement.lang = locale;
  }, [locale]);

  useEffect(() => {
    const applyLocationPage = () => {
      const nextPage = pageFromLocation();
      setActivePage((current) => (current === nextPage ? current : nextPage));
      const requestedTask = normalizeTaskId(new URL(window.location.href).searchParams.get("task"));
      if (requestedTask) {
        explicitTaskFromUrlRef.current = true;
        setSelectedTask((current) => (current === requestedTask ? current : requestedTask));
      }
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

  function openUserResearchResults() {
    if (typeof window !== "undefined") {
      const url = new URL(window.location.href);
      url.searchParams.set("demo", "user");
      window.history.replaceState(null, "", url);
    }
    changeActivePage("control");
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
      if (payload.summary) applySummaryPayload(payload.summary);
      else await refreshSummary(true);
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
        selectTask(payload.task_id);
        setActivePage("overview");
        await refreshSummary(true);
        const configPath = typeof payload.config_path === "string" ? payload.config_path : "configs/generated";
        setSystemActionMessage(
          text(
            locale,
            `新任务已创建并可训练：${payload.task_id}，配置文件：${configPath}`,
            `Runnable task created: ${payload.task_id}; config: ${configPath}`
          )
        );
      } else {
        void refreshSummary(true);
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
    setSelectedTask: selectTask,
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
    <AppShell activePage={activePage} onPageChange={changeActivePage} onAction={runWorkstationAction} locale={locale} summary={summary} selectedTask={selectedTask} ready={summary !== null}>
      {activePage === "assistant" && <AssistantScreen locale={locale} selectedTask={selectedTask} onOpenAdvanced={openUserResearchResults} />}
      {activePage === "tasks" && <TasksScreen {...screenProps} />}
      {activePage === "data" && <DataKaggleScreen {...screenProps} />}
      {activePage === "gpu" && <GpuHpcScreen {...screenProps} />}
      {activePage === "evidence" && <EvidenceLedgerScreen {...screenProps} />}
      {activePage === "literature" && <LiteratureScreen {...screenProps} />}
      {activePage === "workflow" && <WorkflowScreen {...screenProps} />}
      {activePage === "code" && <CodeAgentScreen {...screenProps} />}
      {activePage === "runtime" && <RuntimeScreen {...screenProps} />}
      {activePage === "experiments" && <ExperimentsScreen {...screenProps} />}
      {activePage === "evolution" && (
        <EvolutionConsole selectedTask={screenProps.selectedTask} refreshSummary={screenProps.refreshSummary} />
      )}
      {activePage === "report" && <ReportStudioScreen {...screenProps} />}
      {activePage === "gates" && <GatesScreen {...screenProps} />}
      {activePage === "settings" && <SettingsScreen {...screenProps} />}
      {activePage === "design" && <SettingsScreen {...screenProps} />}
      {activePage === "overview" && <OverviewScreen {...screenProps} />}
      {activePage === "control" && <AiControlConsole {...screenProps} />}
    </AppShell>
  );
}

export default function Home() {
  return (
    <Suspense fallback={null}>
      <HomeClient />
    </Suspense>
  );
}
