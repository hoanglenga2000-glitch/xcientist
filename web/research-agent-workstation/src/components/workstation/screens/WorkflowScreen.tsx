"use client";

import {
  ArrowRight,
  GitBranch,
  Loader2,
  Network,
  ShieldCheck,
  Zap,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { WorkstationSummary } from "@/lib/api/types";
import { PageHeader, Panel, MetricTile } from "../primitives/Layout";
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

export function WorkflowScreen(props: ScreenProps) {
  const { summary, locale = "zh-CN" } = props;
  const workflows = summary?.workflows ?? [];
  const gates = summary?.gates ?? [];
  const pendingGates = gates.filter((g) => {
    const d = String((g as Record<string, unknown>).decision ?? "").toLowerCase();
    return d === "pending" || d === "blocked";
  });
  const latestWorkflow = workflows[0] as Record<string, unknown> | undefined;
  const stages = (latestWorkflow?.stages ?? summary?.stages ?? []) as Array<Record<string, unknown>>;
  const actions = summary?.actions ?? [];

  return (
    <div className="space-y-4">
      <PageHeader
        title={t(locale, "Workflow Graph", "工作流图")}
        subtitle={t(locale, "Agent handoffs, gates, and fallback paths", "Agent 交接、门控与回退路径")}
        breadcrumb={`${t(locale, "Research Loop", "研究循环")} > ${t(locale, "Workflow Graph", "工作流图")}`}
      />

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricTile label={t(locale, "Workflows", "工作流")} value={workflows.length} icon={Network} tone="blue" />
        <MetricTile label={t(locale, "Stages", "阶段")} value={stages.length} icon={GitBranch} tone="neutral" />
        <MetricTile label={t(locale, "Pending Gates", "待审门")} value={pendingGates.length} icon={ShieldCheck} tone={pendingGates.length > 0 ? "amber" : "green"} />
        <MetricTile label={t(locale, "Actions", "动作")} value={actions.length} icon={Zap} tone="neutral" />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="lg:col-span-2 space-y-4">
          <Panel
            title={t(locale, "Agent Handoff Chain", "Agent 交接链")}
            action={
              <div className="flex items-center gap-1">
                <button type="button" data-ui-action="experiments_zoom_out" className="rounded border border-edge px-1.5 py-0.5 text-2xs text-ink-secondary hover:bg-surface-sunken" aria-label={t(locale, "Zoom out", "缩小")}>−</button>
                <button type="button" data-ui-action="experiments_zoom_reset" className="rounded border border-edge px-1.5 py-0.5 text-2xs text-ink-secondary hover:bg-surface-sunken" aria-label={t(locale, "Reset zoom", "重置缩放")}>100%</button>
                <button type="button" data-ui-action="experiments_zoom_in" className="rounded border border-edge px-1.5 py-0.5 text-2xs text-ink-secondary hover:bg-surface-sunken" aria-label={t(locale, "Zoom in", "放大")}>+</button>
                <button type="button" data-ui-action="experiments_filter_graph" className="rounded border border-edge px-1.5 py-0.5 text-2xs text-ink-secondary hover:bg-surface-sunken" aria-label={t(locale, "Filter graph", "过滤图")}>{t(locale, "Filter", "过滤")}</button>
              </div>
            }
          >
            <div className="flex items-center gap-1 overflow-x-auto pb-1">
              {["Scientist", "Planner", "Code Agent", "Trainer", "Validator"].map((agent, idx) => {
                const isCurrent = idx === 1;
                return (
                  <div key={agent} className="flex items-center">
                    <div className={cn(
                      "flex flex-col items-center gap-1 rounded-md px-3 py-2 text-center min-w-[72px]",
                      isCurrent && "bg-accent-light border border-accent/30",
                    )}>
                      <span className={cn("text-2xs font-semibold", isCurrent ? "text-accent-dark" : "text-ink-secondary")}>{agent}</span>
                      {isCurrent && <StatusDot tone="running" />}
                    </div>
                    {idx < 4 && <ArrowRight className="h-3 w-3 shrink-0 text-ink-faint mx-0.5" />}
                  </div>
                );
              })}
            </div>
          </Panel>

          <Panel title={t(locale, "Stages", "阶段")}>
            <div className="space-y-1.5">
              {stages.length === 0 ? (
                <span className="text-xs text-ink-muted">{t(locale, "No stages", "无阶段")}</span>
              ) : (
                stages.slice(0, 8).map((stage, i) => {
                  const status = String(stage.status ?? stage.id ?? "unknown").toLowerCase();
                  const tone: StatusTone = status.includes("complete") || status.includes("done") ? "verified" : status.includes("running") ? "running" : status.includes("fail") ? "failed" : "pending";
                  return (
                    <div key={i} className="flex items-center justify-between rounded-sm border border-edge px-2.5 py-1.5 text-xs">
                      <div className="flex items-center gap-2">
                        <StatusDot tone={tone} />
                        <span className="text-ink-secondary">{String(stage.label ?? stage.id ?? `Stage ${i + 1}`)}</span>
                      </div>
                      <StatusBadgeV2 tone={tone} size="xs">{status}</StatusBadgeV2>
                    </div>
                  );
                })
              )}
            </div>
          </Panel>
        </div>

        <div className="space-y-4">
          <Panel title={t(locale, "Gates & Fallbacks", "门控与回退")}>
            <div className="space-y-2">
              {gates.slice(0, 5).map((g, i) => {
                const gate = g as Record<string, unknown>;
                const decision = String(gate.decision ?? "unknown").toLowerCase();
                return (
                  <div key={i} className="flex items-center justify-between text-xs">
                    <span className="text-ink-secondary">{String(gate.name ?? gate.gate_id ?? `Gate ${i + 1}`)}</span>
                    <GateBadge status={decision === "approved" ? "approved" : decision === "rejected" ? "rejected" : "pending"} />
                  </div>
                );
              })}
              {gates.length === 0 && <span className="text-xs text-ink-muted">{t(locale, "No gates", "无门控")}</span>}
            </div>
          </Panel>

          <Panel title={t(locale, "Deep Link", "深层链接")} compact>
            <div className="space-y-2">
              <button
                type="button"
                data-ui-action="workflow_open_in_runtime"
                className="flex w-full items-center gap-2 rounded-md border border-edge px-3 py-2 text-xs font-medium text-ink-secondary hover:bg-surface-sunken"
              >
                <Loader2 className="h-3.5 w-3.5" /> {t(locale, "Open in Runtime", "在运行时中打开")}
              </button>
              <ClaimBoundary />
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
