"use client";

import {
  Activity,
  ArrowRight,
  Bot,
  Database,
  FlaskConical,
  Loader2,
  Play,
  RefreshCw,
  Server,
  ShieldCheck,
  Zap,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { WorkstationSummary } from "@/lib/api/types";
import { PageHeader, Panel, MetricTile, AiLabel, EmptyState, CopyablePath } from "../primitives/Layout";
import { StatusBadgeV2, StatusDot, type StatusTone } from "../primitives/StatusBadge";
import { GateBadge } from "../primitives/GateBadge";
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

/* ── Helpers ── */
function normalizeTone(status: string | undefined): StatusTone {
  const s = String(status ?? "").toLowerCase();
  if (s.includes("complete") || s.includes("passed") || s.includes("verified") || s.includes("ready")) return "verified";
  if (s.includes("running") || s.includes("submitted")) return "running";
  if (s.includes("blocked") || s.includes("regression")) return "blocked";
  if (s.includes("failed")) return "failed";
  if (s.includes("pending") || s.includes("waiting")) return "pending";
  return "unknown";
}

function connectorEntry(summary: WorkstationSummary | null | undefined, key: string) {
  return (summary?.connector_status as Record<string, Record<string, unknown>> | undefined)?.[key];
}

function isConfigured(entry: Record<string, unknown> | undefined) {
  return Boolean(entry?.configured);
}

function connectorTone(entry: Record<string, unknown> | undefined): StatusTone {
  if (!entry) return "unknown";
  const configured = isConfigured(entry);
  const state = String(entry.state ?? entry.status ?? "").toLowerCase();
  if (configured && (state.includes("verified") || state.includes("ready") || state.includes("passed"))) return "verified";
  if (configured) return "ready";
  if (state.includes("not_configured")) return "unknown";
  if (state.includes("blocked") || state.includes("failed")) return "blocked";
  return "pending";
}

function connectorLabel(locale: Locale | undefined, entry: Record<string, unknown> | undefined): string {
  if (!entry) return t(locale, "Unknown", "未知");
  const configured = isConfigured(entry);
  const state = String(entry.state ?? "").toLowerCase();
  if (configured && state.includes("verified")) return t(locale, "Verified", "已验证");
  if (configured && (state.includes("ready") || state.includes("passed"))) return t(locale, "Ready", "就绪");
  if (configured) return t(locale, "Configured", "已配置");
  if (state.includes("not_configured")) return t(locale, "Not configured", "未配置");
  if (state.includes("blocked")) return t(locale, "Blocked", "阻断");
  return t(locale, "Unknown", "未知");
}

function formatScore(value: unknown): string {
  if (typeof value === "number" && Number.isFinite(value)) return value.toFixed(5);
  return "—";
}

/* ── Research Loop stages ── */
const LOOP_STAGES = [
  { id: "data", label: "Data", labelZh: "数据", icon: Database },
  { id: "plan", label: "Plan", labelZh: "规划", icon: Bot },
  { id: "experiment", label: "Experiment", labelZh: "实验", icon: FlaskConical },
  { id: "train", label: "Train", labelZh: "训练", icon: Zap },
  { id: "validate", label: "Validate", labelZh: "验证", icon: ShieldCheck },
  { id: "submit", label: "Submit", labelZh: "提交", icon: Play },
  { id: "report", label: "Report", labelZh: "报告", icon: Activity },
  { id: "evolve", label: "Evolve", labelZh: "进化", icon: RefreshCw },
] as const;

function currentLoopStage(summary: WorkstationSummary | null | undefined): number {
  const autopilot = summary?.scientist_autopilot;
  const nextAction = summary?.scientist_next_action;
  const status = String(autopilot?.mode ?? nextAction?.status ?? "").toLowerCase();
  if (status.includes("data") || status.includes("audit")) return 0;
  if (status.includes("plan") || status.includes("workplan") || status.includes("situation")) return 1;
  if (status.includes("experiment") || status.includes("blueprint") || status.includes("hypothesis")) return 2;
  if (status.includes("train") || status.includes("execution")) return 3;
  if (status.includes("valid") || status.includes("continuation")) return 4;
  if (status.includes("submit") || status.includes("official")) return 5;
  if (status.includes("report")) return 6;
  if (status.includes("evolv") || status.includes("memory") || status.includes("innovation")) return 7;
  // Default: check runs
  const runCount = summary?.runs?.length ?? 0;
  if (runCount > 0) return 2;
  return 0;
}

/* ── Component ── */
export function OverviewScreen(props: ScreenProps) {
  const { summary, locale = "zh-CN", refreshSummary, runWorkstationAction, runLocalExperiment, runState } = props;
  const isLoading = !summary;

  const latestRun = summary?.runs?.[0];
  const tasks = summary?.tasks ?? [];
  const runs = summary?.runs ?? [];
  const actions = summary?.actions ?? [];
  const evidence = summary?.evidence ?? [];
  const gates = summary?.gates ?? [];
  const pendingGates = gates.filter((g) => {
    const d = String((g as Record<string, unknown>).decision ?? "").toLowerCase();
    return d === "pending" || d === "blocked";
  });
  const nextAction = summary?.scientist_next_action;
  const bestScore = latestRun?.best_metrics ? Object.values(latestRun.best_metrics)[0] : null;
  const loopStage = currentLoopStage(summary);

  /* Connector data */
  const connectors = [
    { key: "local_hpc", label: t(locale, "Local + HPC", "本地 + HPC") },
    { key: "kaggle_api", label: "Kaggle API" },
    { key: "deepseek_api", label: "DeepSeek API" },
  ];

  if (isLoading) {
    return (
      <div>
        <PageHeader title={t(locale, "Research Overview", "科研总览")} subtitle={t(locale, "Loading...", "加载中...")} />
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="h-24 rounded-md bg-surface-sunken animate-pulse" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <PageHeader
        title={t(locale, "Research Overview", "科研总览")}
        subtitle={t(locale, "Research workstation operating status and mission control", "科研工作站运行态势与闭环总控")}
        breadcrumb={`${t(locale, "Command Center", "指挥中心")} / ${t(locale, "Research Overview", "科研总览")}`}
        primaryAction={
          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              size="sm"
              data-ui-action="mission_refresh_overview"
              data-ui-skip-action="true"
              onClick={() => void refreshSummary?.()}
              className="text-xs"
            >
              <RefreshCw className="mr-1 h-3 w-3" />
              {t(locale, "Refresh", "刷新")}
            </Button>
            {runLocalExperiment && (
              <Button
                size="sm"
                data-ui-action="mission_create_workstation_run"
                data-ui-skip-action="true"
                onClick={() => void runLocalExperiment()}
                disabled={runState?.status === "running"}
                className="text-xs"
              >
                {runState?.status === "running" ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : <Play className="mr-1 h-3 w-3" />}
                {t(locale, "Run Experiment", "运行实验")}
              </Button>
            )}
            <button
              type="button"
              data-ui-action="mission_open_evidence_ledger"
              className="inline-flex items-center gap-1 rounded-md border border-edge px-2.5 py-1 text-xs font-medium text-ink-secondary hover:bg-surface-sunken"
            >
              <ShieldCheck className="h-3 w-3" /> {t(locale, "Evidence", "证据")}
            </button>
            <button
              type="button"
              data-ui-action="mission_prepare_hpc_job"
              className="inline-flex items-center gap-1 rounded-md border border-edge px-2.5 py-1 text-xs font-medium text-ink-secondary hover:bg-surface-sunken"
            >
              <Server className="h-3 w-3" /> {t(locale, "HPC", "HPC")}
            </button>
          </div>
        }
      />

      {/* ── KPI Grid ── */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <MetricTile
          label={t(locale, "Active Tasks", "活跃任务")}
          value={tasks.length}
          icon={Activity}
          tone="blue"
        />
        <MetricTile
          label={t(locale, "Running", "运行中")}
          value={runs.filter((r) => String(r.status ?? "").toLowerCase().includes("running")).length}
          icon={Loader2}
          tone={runs.some((r) => String(r.status ?? "").toLowerCase().includes("running")) ? "blue" : "neutral"}
        />
        <MetricTile
          label={t(locale, "Evidence", "证据")}
          value={evidence.length}
          icon={Database}
          tone="neutral"
        />
        <MetricTile
          label={t(locale, "Pending Gates", "待审 Gate")}
          value={pendingGates.length}
          icon={ShieldCheck}
          tone={pendingGates.length > 0 ? "amber" : "green"}
        />
        <MetricTile
          label={t(locale, "Best Score", "最佳分数")}
          value={formatScore(bestScore)}
          icon={Zap}
          tone={bestScore != null ? "green" : "neutral"}
        />
        <MetricTile
          label={t(locale, "System Health", "系统健康")}
          value={connectors.every((c) => connectorTone(connectorEntry(summary, c.key)) === "verified") ? t(locale, "OK", "正常") : t(locale, "Check", "需检查")}
          icon={Server}
          tone={connectors.every((c) => connectorTone(connectorEntry(summary, c.key)) === "verified") ? "green" : "amber"}
        />
      </div>

      {/* ── Main content: 2 columns on desktop ── */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        {/* Left column (2/3) */}
        <div className="lg:col-span-2 space-y-4">

          {/* Research Loop Stage */}
          <Panel title={t(locale, "Research Loop Stage", "研究循环阶段")} description={t(locale, "Current position in the research lifecycle", "当前处于研究生命周期的哪个阶段")}>
            <div className="flex items-center gap-0 overflow-x-auto pb-1">
              {LOOP_STAGES.map((stage, idx) => {
                const Icon = stage.icon;
                const isCurrent = idx === loopStage;
                const isPast = idx < loopStage;
                return (
                  <div key={stage.id} className="flex items-center">
                    <div className={cn(
                      "flex flex-col items-center gap-1 rounded-md px-2.5 py-2 text-center min-w-[64px]",
                      isCurrent && "bg-accent-light border border-accent/30",
                      isPast && "opacity-60"
                    )}>
                      <Icon className={cn("h-4 w-4", isCurrent ? "text-accent" : isPast ? "text-success" : "text-ink-muted")} />
                      <span className={cn("text-2xs font-semibold", isCurrent ? "text-accent-dark" : "text-ink-secondary")}>
                        {locale === "zh-CN" ? stage.labelZh : stage.label}
                      </span>
                      {isCurrent && <StatusDot tone="running" />}
                      {isPast && <StatusDot tone="verified" />}
                    </div>
                    {idx < LOOP_STAGES.length - 1 && (
                      <ArrowRight className="h-3 w-3 shrink-0 text-ink-faint mx-0.5" />
                    )}
                  </div>
                );
              })}
            </div>
          </Panel>

          {/* Current Run */}
          <Panel title={t(locale, "Current Run", "当前运行")} accent={latestRun ? (normalizeTone(latestRun.status) === "verified" ? "green" : normalizeTone(latestRun.status) === "running" ? "blue" : normalizeTone(latestRun.status) === "blocked" ? "red" : undefined) : undefined}>
            {latestRun ? (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <StatusDot tone={normalizeTone(latestRun.status)} />
                    <span className="text-sm font-semibold text-ink">{latestRun.task_id}</span>
                  </div>
                  <StatusBadgeV2 tone={normalizeTone(latestRun.status)} size="sm">
                    {String(latestRun.status ?? "unknown")}
                  </StatusBadgeV2>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-3">
                  {latestRun.best_model && (
                    <div><span className="text-ink-muted">{t(locale, "Best Model", "最佳模型")}:</span> <span className="ml-1 font-mono text-ink-secondary">{latestRun.best_model}</span></div>
                  )}
                  {latestRun.best_metrics && Object.keys(latestRun.best_metrics).length > 0 && (
                    <div><span className="text-ink-muted">{t(locale, "Best Metrics", "最佳指标")}:</span> <span className="ml-1 font-mono text-ink-secondary tabular-nums">{formatScore(Object.values(latestRun.best_metrics)[0])}</span></div>
                  )}
                  {latestRun.started_at && (
                    <div><span className="text-ink-muted">{t(locale, "Started", "开始")}:</span> <span className="ml-1 text-ink-secondary">{new Date(latestRun.started_at).toLocaleString()}</span></div>
                  )}
                </div>
                {latestRun.output_dir && <CopyablePath path={latestRun.output_dir} />}
              </div>
            ) : (
              <EmptyState message={t(locale, "No runs yet. Start an experiment to begin.", "尚无运行记录。启动实验以开始。")} icon={Activity} />
            )}
          </Panel>

          {/* Recent Actions */}
          <Panel title={t(locale, "Recent Actions", "最近动作")} action={
            actions.length > 5 ? <button data-ui-action="mission_view_all_actions" className="text-xs font-semibold text-accent hover:underline">{t(locale, "View all", "查看全部")}</button> : undefined
          }>
            {actions.length === 0 ? (
              <EmptyState message={t(locale, "No actions recorded", "无动作记录")} />
            ) : (
              <div className="space-y-1.5">
                {actions.slice(0, 5).map((action, idx) => {
                  const a = action as Record<string, unknown>;
                  return (
                    <div key={idx} className="flex items-center justify-between gap-2 rounded-sm border border-edge px-2.5 py-1.5 text-xs">
                      <div className="flex items-center gap-2 min-w-0">
                        <StatusDot tone={normalizeTone(String(a.action))} />
                        <span className="truncate text-ink-secondary">{String(a.message ?? a.action)}</span>
                      </div>
                      <span className="shrink-0 text-ink-muted">{String(a.at ?? "").slice(11, 19)}</span>
                    </div>
                  );
                })}
              </div>
            )}
          </Panel>
        </div>

        {/* Right column (1/3) */}
        <div className="space-y-4">
          {/* Next Safe Action */}
          {nextAction?.selected_action && (
            <Panel title={t(locale, "Next Safe Action", "下一步安全动作")} accent="blue">
              <div className="space-y-2">
                <div className="rounded-md border border-accent/20 bg-accent-light p-3">
                  <div className="text-sm font-semibold text-accent-dark">{nextAction.selected_action.title}</div>
                  {nextAction.selected_action.why && (
                    <div className="mt-1 text-xs text-ink-secondary">{nextAction.selected_action.why}</div>
                  )}
                  {nextAction.selected_action.gate && (
                    <div className="mt-2"><GateBadge status={nextAction.selected_action.gate.toLowerCase().includes("human") ? "pending" : "approved"} /></div>
                  )}
                  {nextAction.selected_action.risk && (
                    <div className="mt-1 text-2xs text-warning-text">⚠ {nextAction.selected_action.risk}</div>
                  )}
                </div>
                {nextAction.selected_action.command && (
                  <div className="rounded-sm bg-surface-sunken border border-edge px-2 py-1 font-mono text-2xs text-ink-secondary">
                    {nextAction.selected_action.command}
                  </div>
                )}
                <AiLabel />
              </div>
            </Panel>
          )}

          {/* Connector Health */}
          <Panel title={t(locale, "Connector Health", "连接器状态")} action={
            <button data-ui-action="mission_test_all_connectors" data-ui-skip-action="true" onClick={() => void runWorkstationAction?.("test_all_connectors")} className="text-xs font-semibold text-accent hover:underline">
              {t(locale, "Test All", "测试全部")}
            </button>
          }>
            <div className="space-y-2">
              {connectors.map((conn) => {
                const entry = connectorEntry(summary, conn.key);
                const tone = connectorTone(entry);
                return (
                  <div key={conn.key} className="flex items-center justify-between gap-2 text-xs">
                    <span className="text-ink-secondary">{conn.label}</span>
                    <span className="flex items-center gap-1.5">
                      <StatusDot tone={tone} />
                      <span className={cn(
                        "font-semibold",
                        tone === "verified" ? "text-success-text" : tone === "blocked" ? "text-danger-text" : tone === "unknown" ? "text-ink-muted" : "text-warning-text"
                      )}>
                        {connectorLabel(locale, entry)}
                      </span>
                    </span>
                  </div>
                );
              })}
            </div>
          </Panel>

          {/* System Status Summary */}
          <Panel title={t(locale, "System Status", "系统状态")} compact>
            <div className="space-y-1.5 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-ink-muted">{t(locale, "Autopilot", "自动驾驶")}</span>
                <StatusBadgeV2 tone={summary?.scientist_autopilot_status?.running ? "running" : summary?.scientist_autopilot?.present ? "ready" : "unknown"} size="xs">
                  {summary?.scientist_autopilot_status?.running ? t(locale, "Running", "运行中") : summary?.scientist_autopilot?.present ? t(locale, "Ready", "就绪") : t(locale, "Idle", "空闲")}
                </StatusBadgeV2>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-ink-muted">{t(locale, "Terminal Agent", "终端 Agent")}</span>
                <StatusBadgeV2 tone={summary?.terminal_agent?.status === "live_events" ? "running" : summary?.terminal_agent ? "ready" : "unknown"} size="xs">
                  {summary?.terminal_agent?.status === "live_events" ? t(locale, "Live", "活跃") : summary?.terminal_agent ? t(locale, "Ready", "就绪") : t(locale, "Unknown", "未知")}
                </StatusBadgeV2>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-ink-muted">{t(locale, "Memory Records", "记忆记录")}</span>
                <span className="font-mono tabular-nums text-ink-secondary">{summary?.terminal_agent?.memory_count ?? 0}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-ink-muted">{t(locale, "Evolution Iterations", "进化迭代")}</span>
                <span className="font-mono tabular-nums text-ink-secondary">{summary?.terminal_agent?.n_iterations ?? 0}</span>
              </div>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
