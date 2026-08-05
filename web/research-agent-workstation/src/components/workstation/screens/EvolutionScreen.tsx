"use client";

import {
  GitBranch,
  Network,
  RefreshCw,
  Zap,
} from "lucide-react";
import type { WorkstationSummary } from "@/lib/api/types";
import { PageHeader, Panel, MetricTile, AiLabel, CopyablePath } from "../primitives/Layout";
import { StatusBadgeV2, StatusDot } from "../primitives/StatusBadge";
import { ClaimBoundary } from "../primitives/GateBadge";
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

function formatScore(value: unknown): string {
  if (typeof value === "number" && Number.isFinite(value)) return value.toFixed(5);
  return "—";
}

export function EvolutionScreen(props: ScreenProps) {
  const { summary, locale = "zh-CN", refreshSummary } = props;
  const ta = summary?.terminal_agent;
  const iterations = ta?.iterations ?? [];
  const bestExpId = ta?.best_exp_id;
  const bestCvScore = ta?.best_cv_score;
  const nIterations = ta?.n_iterations ?? 0;
  const nPromotions = ta?.n_promotions ?? 0;
  const memoryCount = ta?.memory_count ?? 0;
  const recentMemory = ta?.recent_memory ?? [];
  const recentIter = iterations.slice(0, 6);
  const stagnation = nIterations > 3 && nPromotions === 0;

  return (
    <div className="space-y-4">
      <PageHeader
        title={t(locale, "Evolution Engine", "进化引擎")}
        subtitle={t(locale, "Search graph, best scores, memory and stagnation", "搜索图、最佳分数、记忆与停滞检测")}
        breadcrumb={`${t(locale, "Research Loop", "研究循环")} > ${t(locale, "Evolution Engine", "进化引擎")}`}
        primaryAction={
          <button
            type="button"
            data-ui-action="evolution_refresh"
            data-ui-skip-action="true"
            onClick={() => void refreshSummary?.()}
            className="flex items-center gap-1 rounded-md border border-edge px-2.5 py-1 text-xs font-medium text-ink-secondary hover:bg-surface-sunken"
          >
            <RefreshCw className="h-3 w-3" /> {t(locale, "Refresh", "刷新")}
          </button>
        }
      />

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricTile label={t(locale, "Iterations", "迭代")} value={nIterations} icon={RefreshCw} tone={nIterations > 0 ? "blue" : "neutral"} />
        <MetricTile label={t(locale, "Best Score", "最佳分数")} value={formatScore(bestCvScore)} icon={Zap} tone={bestCvScore != null ? "green" : "neutral"} />
        <MetricTile label={t(locale, "Promotions", "晋升")} value={nPromotions} icon={GitBranch} tone={nPromotions > 0 ? "green" : "neutral"} />
        <MetricTile label={t(locale, "Memory Hits", "记忆命中")} value={memoryCount} icon={Network} tone="neutral" />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="lg:col-span-2 space-y-4">
          <Panel title={t(locale, "Search Graph Summary", "搜索图摘要")} accent={stagnation ? "red" : undefined}>
            <div className="space-y-2">
              <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
                <div>
                  <span className="text-ink-muted">{t(locale, "Best Exp", "最佳实验")}</span>
                  <div className="font-mono text-ink-secondary">{bestExpId ?? "—"}</div>
                </div>
                <div>
                  <span className="text-ink-muted">{t(locale, "Best CV", "最佳 CV")}</span>
                  <div className="font-mono tabular-nums text-ink-secondary">{formatScore(bestCvScore)}</div>
                </div>
                <div>
                  <span className="text-ink-muted">{t(locale, "Metric", "指标")}</span>
                  <div className="font-mono text-ink-secondary">{ta?.metric ?? "—"}</div>
                </div>
                <div>
                  <span className="text-ink-muted">{t(locale, "Direction", "方向")}</span>
                  <div className="font-mono text-ink-secondary">{ta?.metric_direction ?? "—"}</div>
                </div>
              </div>
              {ta?.evolution_root && <CopyablePath path={ta.evolution_root} />}
              <AiLabel />
            </div>
          </Panel>

          <Panel title={t(locale, "Recent Iterations", "最近迭代")}>
            <div className="space-y-1.5">
              {recentIter.length === 0 ? (
                <span className="text-xs text-ink-muted">{t(locale, "No iterations yet", "尚无迭代")}</span>
              ) : (
                recentIter.map((iter, i) => {
                  const it = iter as Record<string, unknown>;
                  const promoted = Boolean(it.promoted);
                  return (
                    <div key={i} data-ui-action={`evolution_select_iteration_${String(it.exp_id ?? i)}`} className="flex items-center justify-between rounded-sm border border-edge px-2.5 py-1.5 text-xs">
                      <div className="flex items-center gap-2 min-w-0">
                        <StatusDot tone={promoted ? "verified" : "unknown"} />
                        <span className="truncate font-mono text-ink-secondary">{String(it.exp_id ?? `Iter ${i + 1}`)}</span>
                      </div>
                      <div className="flex items-center gap-3 shrink-0">
                        <span className="font-mono tabular-nums text-ink-muted">{formatScore(it.cv_score)}</span>
                        {promoted && <StatusBadgeV2 tone="verified" size="xs">{t(locale, "Promoted", "晋升")}</StatusBadgeV2>}
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </Panel>
        </div>

        <div className="space-y-4">
          <Panel title={t(locale, "Branch Info", "分支信息")}>
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="text-ink-muted">{t(locale, "Task", "任务")}</span>
                <span className="font-mono text-ink-secondary">{ta?.task_id ?? "—"}</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-ink-muted">{t(locale, "Completed Runs", "已完成")}</span>
                <span className="font-mono tabular-nums text-ink-secondary">{ta?.completed_run_count ?? 0}</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-ink-muted">{t(locale, "Total Runs", "总运行")}</span>
                <span className="font-mono tabular-nums text-ink-secondary">{ta?.run_count ?? 0}</span>
              </div>
            </div>
          </Panel>

          <Panel title={t(locale, "Stagnation", "停滞检测")} accent={stagnation ? "red" : undefined}>
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-ink-muted">{t(locale, "Status", "状态")}</span>
                <StatusBadgeV2 tone={stagnation ? "blocked" : "verified"} size="xs">
                  {stagnation ? t(locale, "Stagnant", "停滞") : t(locale, "Healthy", "健康")}
                </StatusBadgeV2>
              </div>
              {stagnation && <div className="rounded-sm border border-danger/20 bg-danger/5 px-2 py-1.5 text-2xs text-danger-text">{t(locale, "No promotions after iterations. Consider strategy change.", "多次迭代无晋升。建议调整策略。")}</div>}
            </div>
          </Panel>

          <Panel title={t(locale, "Memory", "记忆")} compact>
            <div className="space-y-1.5 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-ink-muted">{t(locale, "Records", "记录")}</span>
                <span className="font-mono tabular-nums text-ink-secondary">{memoryCount}</span>
              </div>
              {recentMemory.slice(0, 3).map((m, i) => (
                <div key={i} className="truncate text-ink-muted">{String((m as Record<string, unknown>).reusable_strategy ?? (m as Record<string, unknown>).method ?? `Record ${i + 1}`)}</div>
              ))}
              <ClaimBoundary />
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
