"use client";

import {
  ArrowUpRight,
  Download,
  FlaskConical,
  Minus,
  RefreshCw,
  RotateCcw,
  Trophy,
} from "lucide-react";
import type { WorkstationRun } from "@/lib/api/types";
import type { WorkstationSummary } from "@/lib/api/types";
import { PageHeader, Panel, MetricTile } from "../primitives/Layout";
import { StatusBadgeV2, StatusDot, type StatusTone } from "../primitives/StatusBadge";
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

/* ── Helpers ── */
function runTone(status: string | undefined): StatusTone {
  const s = String(status ?? "").toLowerCase();
  if (s.includes("complete") || s.includes("passed") || s.includes("promoted")) return "verified";
  if (s.includes("running") || s.includes("submitted")) return "running";
  if (s.includes("blocked") || s.includes("regression") || s.includes("rollback")) return "blocked";
  if (s.includes("failed")) return "failed";
  if (s.includes("pending") || s.includes("held")) return "pending";
  return "unknown";
}

function firstMetric(run: WorkstationRun): number | null {
  if (!run.best_metrics) return null;
  const v = Object.values(run.best_metrics)[0];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function decisionCategory(run: WorkstationRun): "promoted" | "held" | "rollback" | null {
  const d = String((run.validation_gate?.status ?? "")).toLowerCase();
  const s = String(run.status ?? "").toLowerCase();
  if (d.includes("promot") || s.includes("promoted")) return "promoted";
  if (d.includes("held") || s.includes("held")) return "held";
  if (d.includes("rollback") || s.includes("regression") || s.includes("rollback")) return "rollback";
  if (s.includes("complete") || s.includes("passed")) return "promoted";
  return null;
}

/* ── Tiny SVG spark-line for score trend ── */
function ScoreTrendChart({ runs, locale }: { runs: WorkstationRun[]; locale: Locale }) {
  const scores = runs
    .slice(0, 30)
    .map((r) => firstMetric(r))
    .filter((v): v is number => v !== null);
  if (scores.length < 2) {
    return <div className="py-6 text-center text-xs text-ink-muted">{t(locale, "Need at least 2 scored runs", "至少需要 2 条得分运行")}</div>;
  }
  const W = 480, H = 160, PX = 40, PY = 20;
  const mn = Math.min(...scores), mx = Math.max(...scores);
  const range = mx - mn || 1;
  const pts = scores.map((v, i) => {
    const x = PX + (i / (scores.length - 1)) * (W - PX * 2);
    const y = PY + (1 - (v - mn) / range) * (H - PY * 2);
    return { x, y, v };
  });
  const pathD = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
  const bestIdx = scores.indexOf(mx);
  const bestPt = pts[bestIdx];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img" aria-label={t(locale, "Score trend chart", "分数趋势图")}>
      {/* Title & unit */}
      <text x={W / 2} y={14} textAnchor="middle" className="fill-ink-secondary text-[10px] font-semibold">
        {t(locale, "Score Trend (last 30 runs)", "分数趋势（近 30 次运行）")}
      </text>
      {/* Y axis labels */}
      <text x={PX - 4} y={PY + 4} textAnchor="end" className="fill-ink-muted text-[9px]">{mx.toFixed(3)}</text>
      <text x={PX - 4} y={H - PY + 4} textAnchor="end" className="fill-ink-muted text-[9px]">{mn.toFixed(3)}</text>
      {/* Grid line */}
      <line x1={PX} y1={H - PY} x2={W - PX} y2={H - PY} stroke="currentColor" className="text-ink-faint" strokeWidth={0.5} />
      {/* Area fill */}
      <path d={`${pathD} L${pts[pts.length - 1].x.toFixed(1)},${H - PY} L${pts[0].x.toFixed(1)},${H - PY} Z`}
        className="fill-accent/10" />
      {/* Line */}
      <path d={pathD} fill="none" className="stroke-accent" strokeWidth={1.5} strokeLinejoin="round" />
      {/* Best dot + label */}
      <circle cx={bestPt.x} cy={bestPt.y} r={3.5} className="fill-success-text" />
      <text x={bestPt.x} y={bestPt.y - 8} textAnchor="middle" className="fill-success-text text-[9px] font-semibold">
        {t(locale, "Best", "最佳")} {mx.toFixed(4)}
      </text>
      {/* Dots */}
      {pts.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r={1.5} className="fill-accent" opacity={0.6} />
      ))}
    </svg>
  );
}

/* ── Component ── */
export function ExperimentsScreen(props: ScreenProps) {
  const { summary, locale = "zh-CN", refreshSummary } = props;
  const runs = summary?.runs ?? [];
  const inv = summary?.kaggle_experiment_inventory;

  // KPIs
  const scoredRuns = runs.filter((r) => firstMetric(r) !== null);
  const bestScore = scoredRuns.length > 0 ? Math.max(...scoredRuns.map(firstMetric) as number[]) : null;
  const promoted = runs.filter((r) => decisionCategory(r) === "promoted").length;
  const held = runs.filter((r) => decisionCategory(r) === "held").length;
  const rollback = runs.filter((r) => decisionCategory(r) === "rollback").length;

  // Evolution state
  const evoState = summary?.learning_loop_readiness;
  const evoProgress = evoState?.training_progress;

  const recordExperimentsAction = (action: string, metadata: Record<string, unknown> = {}) => {
    void props.runWorkstationAction?.(action, {
      task_id: props.selectedTask,
      source: "experiments_screen",
      ...metadata,
    });
  };

  const exportLedger = () => {
    const rows = runs.map((r) => ({
      task_id: r.task_id,
      run_id: r.id,
      status: r.status,
      best_metric: firstMetric(r),
      decision: decisionCategory(r),
      started_at: r.started_at,
    }));
    const header = "task_id,run_id,status,best_metric,decision,started_at";
    const esc = (v: unknown) => `"${String(v ?? "").replace(/"/g, '""')}"`;
    const csv = [header, ...rows.map((r) => [r.task_id, r.run_id, r.status, r.best_metric, r.decision, r.started_at].map(esc).join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `experiment_ledger_${props.selectedTask || "all"}.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
    recordExperimentsAction("experiments_export_ledger", { rows: rows.length });
  };

  return (
    <div className="space-y-4">
      <PageHeader
        title={t(locale, "Experiment Ledger", "实验台账")}
        subtitle={t(locale, "Run history, score trends, and promotion governance", "运行历史、分数趋势与晋升治理")}
        breadcrumb={`${t(locale, "Research Loop", "研究循环")} / ${t(locale, "Experiment Ledger", "实验台账")}`}
        primaryAction={
          <div className="flex items-center gap-1.5">
            <button
              type="button"
              data-ui-action="experiments_refresh"
              data-ui-skip-action="true"
              onClick={() => void refreshSummary?.()}
              className="inline-flex items-center gap-1 rounded-md border border-edge px-2.5 py-1 text-xs font-medium text-ink-secondary hover:bg-surface-sunken"
            >
              <RefreshCw className="h-3 w-3" />
              {t(locale, "Refresh", "刷新")}
            </button>
            <button
              type="button"
              data-ui-action="experiments_export_ledger"
              data-ui-skip-action="true"
              onClick={exportLedger}
              disabled={runs.length === 0}
              className="inline-flex items-center gap-1 rounded-md border border-edge px-2.5 py-1 text-xs font-medium text-ink-secondary hover:bg-surface-sunken disabled:opacity-40"
            >
              <Download className="h-3 w-3" />
              {t(locale, "Export", "导出")}
            </button>
          </div>
        }
      />

      {/* ── KPI Row ── */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
        <MetricTile label={t(locale, "Best Score", "最佳分数")} value={bestScore !== null ? bestScore.toFixed(5) : "—"} icon={Trophy} tone={bestScore !== null ? "green" : "neutral"} />
        <MetricTile label={t(locale, "Total Runs", "总运行")} value={runs.length} icon={FlaskConical} tone="blue" />
        <MetricTile label={t(locale, "Promoted", "已晋升")} value={promoted} icon={ArrowUpRight} tone="green" />
        <MetricTile label={t(locale, "Held", "保留待审")} value={held} icon={Minus} tone="amber" />
        <MetricTile label={t(locale, "Rollback", "回退")} value={rollback} icon={RotateCcw} tone={rollback > 0 ? "red" : "neutral"} />
      </div>

      {/* ── Score Trend + Search Graph Summary ── */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <Panel title={t(locale, "Score Trend", "分数趋势")} description={t(locale, "Best metric across recent runs", "近期运行的最佳指标走势")}>
            <ScoreTrendChart runs={[...runs].reverse()} locale={locale} />
            <div className="mt-2 flex items-center gap-4 text-[10px] text-ink-muted">
              <span className="flex items-center gap-1"><span className="inline-block h-2 w-4 rounded-sm bg-accent" /> {t(locale, "Run score", "运行分数")}</span>
              <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-full bg-success-text" /> {t(locale, "Best so far", "历史最佳")}</span>
            </div>
          </Panel>
        </div>
        <Panel title={t(locale, "Search Graph Summary", "搜索图摘要")} description={t(locale, "Evolution state and exploration progress", "进化状态与探索进展")}>
          <div className="space-y-2.5 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-ink-muted">{t(locale, "Exploration Stage", "探索阶段")}</span>
              <StatusBadgeV2 tone={evoProgress ? "running" : "unknown"} size="sm">
                {evoProgress ? t(locale, "Active", "活跃") : t(locale, "Idle", "空闲")}
              </StatusBadgeV2>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-ink-muted">{t(locale, "Scored Runs", "已评分")}</span>
              <span className="font-mono tabular-nums text-ink-secondary">{evoProgress?.scored_runs ?? inv?.total_scored_runs ?? scoredRuns.length}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-ink-muted">{t(locale, "Promoted", "已晋升")}</span>
              <span className="font-mono tabular-nums text-success-text">{evoProgress?.promoted_runs ?? inv?.total_promoted_runs ?? promoted}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-ink-muted">{t(locale, "Held / Timeout", "保留/超时")}</span>
              <span className="font-mono tabular-nums text-warning-text">{evoProgress?.held_runs ?? inv?.total_held_runs ?? held}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-ink-muted">{t(locale, "Official Top30", "官方 Top30")}</span>
              <span className="font-mono tabular-nums text-ink-secondary">{evoProgress?.official_top30_tasks ?? inv?.official_top30_count ?? "—"}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-ink-muted">{t(locale, "Medal Count", "奖牌数")}</span>
              <span className="font-mono tabular-nums text-ink-secondary">{evoProgress?.medal_count ?? "—"}</span>
            </div>
            {evoState?.claim_boundary && <ClaimBoundary boundary={evoState.claim_boundary} />}
          </div>
        </Panel>
      </div>

      {/* ── Run Ledger ── */}
      <Panel title={t(locale, "Run Ledger", "运行台账")} description={t(locale, "Recent experiment runs with promotion decisions", "近期实验运行及晋升决策")}>
        {/* Desktop table */}
        <div className="hidden sm:block overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-edge text-left text-ink-muted">
                <th className="pb-1.5 pr-3 font-medium">{t(locale, "Task / Run", "任务/运行")}</th>
                <th className="pb-1.5 pr-3 font-medium">{t(locale, "Status", "状态")}</th>
                <th className="pb-1.5 pr-3 font-medium">{t(locale, "Best Metric", "最佳指标")}</th>
                <th className="pb-1.5 pr-3 font-medium">{t(locale, "Decision", "决策")}</th>
                <th className="pb-1.5 font-medium">{t(locale, "Started", "开始时间")}</th>
              </tr>
            </thead>
            <tbody>
              {runs.slice(0, 20).map((run, idx) => {
                const score = firstMetric(run);
                const cat = decisionCategory(run);
                return (
                  <tr key={run.id ?? idx} data-ui-action={`experiments_view_${run.id ?? idx}`} className="border-b border-edge/50 hover:bg-surface-sunken/50 cursor-pointer">
                    <td className="py-1.5 pr-3 font-mono text-ink-secondary">{run.task_id}{run.id ? ` / ${run.id.slice(0, 8)}` : ""}</td>
                    <td className="py-1.5 pr-3"><StatusDot tone={runTone(run.status)} /> <span className="ml-1 text-ink-secondary">{run.status ?? "—"}</span></td>
                    <td className="py-1.5 pr-3 font-mono tabular-nums text-ink-secondary">{score !== null ? score.toFixed(5) : "—"}</td>
                    <td className="py-1.5 pr-3">
                      {cat === "promoted" && <span className="inline-flex items-center gap-0.5 text-success-text"><ArrowUpRight className="h-3 w-3" />{t(locale, "Promoted", "晋升")}</span>}
                      {cat === "held" && <span className="inline-flex items-center gap-0.5 text-warning-text"><Minus className="h-3 w-3" />{t(locale, "Held", "保留")}</span>}
                      {cat === "rollback" && <span className="inline-flex items-center gap-0.5 text-danger-text"><RotateCcw className="h-3 w-3" />{t(locale, "Rollback", "回退")}</span>}
                      {!cat && <span className="text-ink-muted">—</span>}
                    </td>
                    <td className="py-1.5 text-ink-muted">{run.started_at ? new Date(run.started_at).toLocaleString() : "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {/* Mobile cards */}
        <div className="space-y-2 sm:hidden">
          {runs.slice(0, 10).map((run, idx) => {
            const score = firstMetric(run);
            const cat = decisionCategory(run);
            return (
              <div key={run.id ?? idx} className="rounded-md border border-edge px-3 py-2 space-y-1">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs text-ink-secondary">{run.task_id}</span>
                  <StatusBadgeV2 tone={runTone(run.status)} size="xs">{run.status ?? "—"}</StatusBadgeV2>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-ink-muted">{t(locale, "Best Metric", "最佳指标")}:</span>
                  <span className="font-mono tabular-nums text-ink-secondary">{score !== null ? score.toFixed(5) : "—"}</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-ink-muted">{t(locale, "Decision", "决策")}:</span>
                  {cat === "promoted" && <span className="text-success-text">{t(locale, "Promoted", "晋升")}</span>}
                  {cat === "held" && <span className="text-warning-text">{t(locale, "Held", "保留")}</span>}
                  {cat === "rollback" && <span className="text-danger-text">{t(locale, "Rollback", "回退")}</span>}
                  {!cat && <span className="text-ink-muted">—</span>}
                </div>
                <div className="text-[10px] text-ink-muted">{run.started_at ? new Date(run.started_at).toLocaleString() : "—"}</div>
              </div>
            );
          })}
        </div>
        {runs.length === 0 && (
          <div className="py-8 text-center text-xs text-ink-muted">{t(locale, "No runs recorded yet", "尚无运行记录")}</div>
        )}
      </Panel>
    </div>
  );
}
