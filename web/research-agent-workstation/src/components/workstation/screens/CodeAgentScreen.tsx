"use client";

import { useState } from "react";
import {
  Bot,
  Code2,
  FileCode,
  FilePlus2,
  Loader2,
  Lock,
  Merge,
  RotateCcw,
  ShieldCheck,
} from "lucide-react";
import type { WorkstationSummary } from "@/lib/api/types";
import { PageHeader, Panel, MetricTile, AiLabel, CopyablePath } from "../primitives/Layout";
import { StatusBadgeV2, StatusDot, type StatusTone } from "../primitives/StatusBadge";
import { GateBadge, ClaimBoundary } from "../primitives/GateBadge";
import { t } from "../localization";

type Locale = "zh-CN" | "en-US";

type ScreenProps = {
  selectedTask: string;
  setSelectedTask: (id: string) => void;
  selectedStage: string;
  setSelectedStage: (id: string) => void;
  selectedExperiment: string;
  setSelectedExperiment: (id: string) => void;
  gateStatus: "Pending" | "Approved" | "Rejected";
  setGateStatus: (status: "Pending" | "Approved" | "Rejected") => void;
  patchApplied: boolean;
  setPatchApplied: (value: boolean) => void;
  reportSubmitted: boolean;
  setReportSubmitted: (value: boolean) => void;
  summary?: WorkstationSummary | null;
  refreshSummary?: () => Promise<WorkstationSummary>;
  runLocalExperiment?: (taskId?: string) => Promise<void>;
  runState?: { status: "idle" | "running" | "passed" | "failed"; message: string; experimentDir?: string };
  exportCodeAgentContext?: (taskId?: string, targetAgent?: string) => Promise<void>;
  importDemoPatch?: (taskId?: string) => Promise<void>;
  agentActionMessage?: string;
  runWorkstationAction?: (action: string, metadata?: Record<string, unknown>) => Promise<unknown>;
  systemActionMessage?: string;
  lastActionTrace?: {
    action: string; taskId?: string; request?: Record<string, unknown>;
    response?: Record<string, unknown>; message: string; artifact?: string | null; at: string;
  } | null;
  locale?: Locale;
  setLocale?: (locale: Locale) => void;
};

function agentTone(status: string | undefined): StatusTone {
  const s = String(status ?? "").toLowerCase();
  if (s.includes("complete") || s.includes("success") || s.includes("ready")) return "verified";
  if (s.includes("running") || s.includes("active")) return "running";
  if (s.includes("failed") || s.includes("error")) return "failed";
  if (s.includes("pending") || s.includes("waiting")) return "pending";
  return "unknown";
}

export function CodeAgentScreen(props: ScreenProps) {
  const { summary, locale = "zh-CN", runWorkstationAction, patchApplied, exportCodeAgentContext } = props;
  const engLoop = summary?.scientist_engineering_loop;
  const agentStatus = String(engLoop?.status ?? "idle");
  const changedFiles: string[] = (engLoop?.changed_files as string[] | undefined) ?? [];
  const humanGate = String(engLoop?.human_gate ?? "");
  const mergeReady = Boolean(engLoop?.merge_ready);
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [actionMessage, setActionMessage] = useState<string>("");

  const executeCodeAction = async (action: string, metadata: Record<string, unknown> = {}) => {
    if (!runWorkstationAction || !props.selectedTask) return null;
    setBusyAction(action);
    setActionMessage("");
    try {
      const result = await runWorkstationAction(action, {
        task_id: props.selectedTask,
        source: "code_agent_screen",
        ...metadata,
      }) as Record<string, unknown>;
      setActionMessage(String(result?.message ?? `${action} completed`));
      await props.refreshSummary?.();
      return result;
    } catch (error) {
      setActionMessage(error instanceof Error ? error.message : `${action} failed`);
      return null;
    } finally {
      setBusyAction(null);
    }
  };

  const askCodeAgent = async () => {
    if (!props.selectedTask) return;
    setBusyAction("ask_code_agent");
    setActionMessage("");
    try {
      const response = await fetch(`/api/tasks/${encodeURIComponent(props.selectedTask)}/code-agent-draft`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source_agent: "local_template",
          prompt: `Generate a reviewable local patch for ${props.selectedTask} using current task evidence.`
        })
      });
      const payload = await response.json() as Record<string, unknown>;
      if (!response.ok || payload.ok === false) throw new Error(String(payload.error ?? `Code Agent request failed: ${response.status}`));
      const patchPath = typeof payload.patch_path === "string" ? payload.patch_path : undefined;
      if (!patchPath) throw new Error("Code Agent response did not include patch_path.");
      const review = await runWorkstationAction?.("review_agent_patch", {
        task_id: props.selectedTask,
        source: "code_agent_screen",
        source_agent: "local_template",
        patch_path: patchPath
      }) as Record<string, unknown> | undefined;
      setActionMessage(String(review?.message ?? `Draft generated: ${patchPath}`));
      await props.refreshSummary?.();
    } catch (error) {
      setActionMessage(error instanceof Error ? error.message : "Code Agent request failed");
    } finally {
      setBusyAction(null);
    }
  };

  return (
    <div className="space-y-4">
      <PageHeader
        title={t(locale, "Code Agent IDE", "代码 Agent IDE")}
        subtitle={t(locale, "AI-assisted code generation and patch management", "AI 辅助代码生成与补丁管理")}
        breadcrumb={`${t(locale, "Workbench", "工坊")} > ${t(locale, "Code Agent IDE", "代码 Agent IDE")}`}
      />

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricTile label={t(locale, "Agent Status", "Agent 状态")} value={agentStatus} icon={Bot} tone={agentTone(agentStatus) === "running" ? "blue" : "neutral"} />
        <MetricTile label={t(locale, "Changed Files", "变更文件")} value={changedFiles.length} icon={FileCode} tone={changedFiles.length > 0 ? "blue" : "neutral"} />
        <MetricTile label={t(locale, "Quality Gate", "质量门")} value={engLoop?.epistemic_status ?? "—"} icon={ShieldCheck} tone={engLoop?.epistemic_status === "clean" ? "green" : "amber"} />
        <MetricTile label={t(locale, "Merge Ready", "合并就绪")} value={mergeReady ? t(locale, "Yes", "是") : t(locale, "No", "否")} icon={Merge} tone={mergeReady ? "green" : "neutral"} />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="lg:col-span-2 space-y-4">
          <Panel title={t(locale, "Agent Session", "Agent 会话")} accent={agentTone(agentStatus) === "running" ? "blue" : undefined}>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <StatusDot tone={agentTone(agentStatus)} />
                  <span className="text-sm font-semibold text-ink">{engLoop?.work_order?.title ?? t(locale, "No active session", "无活跃会话")}</span>
                </div>
                <StatusBadgeV2 tone={agentTone(agentStatus)} size="sm">{agentStatus}</StatusBadgeV2>
              </div>
              {engLoop?.patch_path && <CopyablePath path={engLoop.patch_path} />}
              <AiLabel />
            </div>
          </Panel>

          <Panel
            title={t(locale, "File Tree", "文件树")}
            action={
              <button
                type="button"
                data-ui-action="code_add_new_file"
                data-ui-skip-action="true"
                onClick={() => void executeCodeAction("code_add_new_file")}
                disabled={Boolean(busyAction) || !props.selectedTask}
                className="flex items-center gap-1 rounded border border-edge px-2 py-1 text-2xs font-medium text-ink-secondary hover:bg-surface-sunken"
              >
                <FilePlus2 className="h-3 w-3" /> {t(locale, "New File", "新文件")}
              </button>
            }
          >
            <div className="space-y-1.5">
              {changedFiles.length === 0 ? (
                <span className="text-xs text-ink-muted">{t(locale, "No changed files", "无变更文件")}</span>
              ) : (
                changedFiles.map((f, i) => (
                  <div key={i} className="flex items-center gap-2 rounded-sm border border-edge px-2.5 py-1.5 text-xs">
                    <FileCode className="h-3 w-3 text-ink-muted" />
                    <span className="truncate font-mono text-ink-secondary">{f}</span>
                  </div>
                ))
              )}
            </div>
          </Panel>
        </div>

        <div className="space-y-4">
          <Panel title={t(locale, "Quality Gate", "质量门")}>
            <div className="space-y-2">
              {(engLoop?.acceptance_checks as Array<Record<string, unknown>> | undefined)?.map((check, i) => (
                <div key={i} className="flex items-center justify-between text-xs">
                  <span className="text-ink-secondary">{String(check.command ?? `Check ${i + 1}`)}</span>
                  <StatusDot tone={check.passed ? "verified" : "failed"} />
                </div>
              )) ?? <span className="text-xs text-ink-muted">{t(locale, "No checks", "无检查项")}</span>}
            </div>
          </Panel>

          <Panel title={t(locale, "Human Merge Gate", "人工合并门")}>
            <div className="space-y-2">
              <GateBadge status={humanGate.toLowerCase().includes("required") ? "pending" : mergeReady ? "approved" : "blocked"} />
              {humanGate && <span className="text-xs text-ink-secondary">{humanGate}</span>}
              <ClaimBoundary />
            </div>
          </Panel>

          <Panel title={t(locale, "Actions", "操作")} compact>
            <div className="space-y-2">
              <button
                type="button"
                data-ui-action="ask_code_agent"
                data-ui-skip-action="true"
                onClick={() => void askCodeAgent()}
                disabled={Boolean(busyAction) || !props.selectedTask}
                className="flex w-full items-center gap-2 rounded-md border border-edge px-3 py-2 text-xs font-medium text-ink-secondary hover:bg-surface-sunken"
              >
                <Bot className="h-3.5 w-3.5" /> {t(locale, "Ask Code Agent", "询问代码 Agent")}
              </button>
              <button
                type="button"
                data-ui-action="run_code_smoke_test"
                data-ui-skip-action="true"
                onClick={() => void executeCodeAction("run_code_smoke_test")}
                disabled={Boolean(busyAction) || !props.selectedTask}
                className="flex w-full items-center gap-2 rounded-md border border-edge px-3 py-2 text-xs font-medium text-ink-secondary hover:bg-surface-sunken"
              >
                <ShieldCheck className="h-3.5 w-3.5" /> {t(locale, "Run Smoke Test", "运行冒烟测试")}
              </button>
              <button
                type="button"
                data-ui-action="request_code_quality_gate"
                data-ui-skip-action="true"
                onClick={() => void executeCodeAction("request_code_quality_gate")}
                disabled={Boolean(busyAction) || !props.selectedTask}
                className="flex w-full items-center gap-2 rounded-md border border-edge px-3 py-2 text-xs font-medium text-ink-secondary hover:bg-surface-sunken"
              >
                <ShieldCheck className="h-3.5 w-3.5" /> {t(locale, "Request Quality Gate", "请求质量门")}
              </button>
              <button
                type="button"
                data-ui-action="apply_agent_patch"
                data-ui-skip-action="true"
                onClick={() => void executeCodeAction("apply_agent_patch").then((result) => result && props.setPatchApplied(true))}
                disabled={Boolean(busyAction) || !props.selectedTask || patchApplied}
                className="flex w-full items-center gap-2 rounded-md border border-edge px-3 py-2 text-xs font-medium text-ink-secondary hover:bg-surface-sunken disabled:opacity-40"
              >
                <Merge className="h-3.5 w-3.5" /> {t(locale, "Apply verified patch", "应用已验证补丁")}
              </button>
              <button
                type="button"
                data-ui-action="rollback_agent_patch"
                data-ui-skip-action="true"
                onClick={() => void executeCodeAction("rollback_agent_patch").then((result) => result && props.setPatchApplied(false))}
                disabled={Boolean(busyAction) || !props.selectedTask || !patchApplied}
                className="flex w-full items-center gap-2 rounded-md border border-edge px-3 py-2 text-xs font-medium text-ink-secondary hover:bg-surface-sunken disabled:opacity-40"
              >
                <RotateCcw className="h-3.5 w-3.5" /> {t(locale, "Rollback patch", "回滚补丁")}
              </button>
              {busyAction && (
                <div className="flex items-center gap-2 text-2xs text-ink-muted"><Loader2 className="h-3 w-3 animate-spin" /> {busyAction}</div>
              )}
              {actionMessage && <p className="rounded border border-edge bg-surface-sunken px-2 py-1.5 text-2xs text-ink-secondary">{actionMessage}</p>}
              <button
                type="button"
                data-ui-action="code_export_context"
                data-ui-skip-action="true"
                onClick={() => void exportCodeAgentContext?.()}
                className="flex w-full items-center gap-2 rounded-md border border-edge px-3 py-2 text-xs font-medium text-ink-secondary hover:bg-surface-sunken"
              >
                <Code2 className="h-3.5 w-3.5" /> {t(locale, "Export Context", "导出上下文")}
              </button>
              <button
                type="button"
                data-ui-action="blocked_send_to_hpc"
                className="flex w-full items-center gap-2 rounded-md border border-danger/30 bg-danger/5 px-3 py-2 text-xs font-medium text-danger-text"
              >
                <Lock className="h-3.5 w-3.5" /> {t(locale, "Send to HPC (Human Gate)", "发送到 HPC(人工闸门)")}
              </button>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
