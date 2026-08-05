"use client";

import {
  Database,
  FileSearch,
  Lock,
  RefreshCw,
  Send,
  ShieldCheck,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { WorkstationSummary } from "@/lib/api/types";
import { PageHeader, Panel, MetricTile } from "../primitives/Layout";
import { StatusBadgeV2, StatusDot, type StatusTone } from "../primitives/StatusBadge";
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

export function DataKaggleScreen(props: ScreenProps) {
  const { summary, locale = "zh-CN" } = props;
  const dpapi = summary?.kaggle_dpapi_readiness;
  const inv = summary?.kaggle_experiment_inventory;
  const newComp = summary?.kaggle_new_competition_readiness;

  const dpapiTone: StatusTone = dpapi?.configured && dpapi?.toolchain_ready ? "verified"
    : dpapi?.configured ? "ready" : dpapi?.present ? "pending" : "unknown";

  const recordDataAction = (action: string, metadata: Record<string, unknown> = {}) => {
    void props.runWorkstationAction?.(action, {
      task_id: props.selectedTask,
      source: "data_kaggle_screen",
      ...metadata,
    });
  };

  return (
    <div className="space-y-4">
      <PageHeader
        title={t(locale, "Data / Kaggle", "数据 / Kaggle")}
        subtitle={t(locale, "Dataset audit, schema, lineage, and Kaggle integration", "数据集审计、模式、血缘与 Kaggle 集成")}
        breadcrumb={`${t(locale, "Workbench", "工作台")} > ${t(locale, "Data / Kaggle", "数据 / Kaggle")}`}
        primaryAction={
          <button
            type="button"
            data-ui-action="data_refresh_inventory"
            data-ui-skip-action="true"
            onClick={() => {
              recordDataAction("data_refresh_inventory");
              void props.refreshSummary?.();
            }}
            className="flex items-center gap-1 rounded-md border border-edge px-2.5 py-1 text-xs font-medium text-ink-secondary hover:bg-surface-sunken"
          >
            <RefreshCw className="h-3 w-3" /> {t(locale, "Refresh", "刷新")}
          </button>
        }
      />

      {/* KPI row */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricTile
          label={t(locale, "DPAPI Configured", "DPAPI 已配置")}
          value={dpapi?.configured ? t(locale, "Yes", "是") : t(locale, "No", "否")}
          icon={ShieldCheck}
          tone={dpapiTone}
        />
        <MetricTile
          label={t(locale, "Toolchain Ready", "工具链就绪")}
          value={dpapi?.toolchain_ready ? t(locale, "Ready", "就绪") : t(locale, "Not Ready", "未就绪")}
          icon={Database}
          tone={dpapi?.toolchain_ready ? "verified" : "blocked"}
        />
        <MetricTile
          label={t(locale, "Total Runs", "总运行")}
          value={inv?.total_runs_observed ?? 0}
          icon={FileSearch}
          tone="neutral"
        />
        <MetricTile
          label={t(locale, "Top-30 Rate", "Top-30 率")}
          value={inv?.official_top30_rate != null ? `${(inv.official_top30_rate * 100).toFixed(1)}%` : "—"}
          icon={Send}
          tone={(inv?.official_top30_rate ?? 0) > 0 ? "green" : "neutral"}
        />
      </div>

      {/* DPAPI Readiness */}
      <Panel title={t(locale, "Kaggle DPAPI Readiness", "Kaggle DPAPI 就绪状态")}>
        <div className="space-y-2 text-xs">
          {[
            { label: t(locale, "Credential Status", "凭据状态"), value: dpapi?.credential_status ?? "—", ok: dpapi?.configured },
            { label: t(locale, "Token Type", "Token 类型"), value: dpapi?.token_type ?? "—", ok: dpapi?.token_loaded_in_env },
            { label: t(locale, "Token in Env", "环境变量 Token"), value: dpapi?.token_loaded_in_env ? t(locale, "Loaded", "已加载") : t(locale, "Missing", "缺失"), ok: dpapi?.token_loaded_in_env },
            { label: t(locale, "Credential File", "凭据文件"), value: dpapi?.credential_file_present ? t(locale, "Present", "存在") : t(locale, "Missing", "缺失"), ok: dpapi?.credential_file_present },
            { label: t(locale, "Python Package", "Python 包"), value: dpapi?.python_package_version ?? "—", ok: dpapi?.toolchain_ready },
            { label: t(locale, "Human Gate Required", "需人工 Gate"), value: dpapi?.human_gate_required_for_submission ? t(locale, "Yes", "是") : t(locale, "No", "否"), ok: !dpapi?.human_gate_required_for_submission },
          ].map((row) => (
            <div key={row.label} className="flex items-center justify-between gap-2">
              <span className="text-ink-muted">{row.label}</span>
              <span className="flex items-center gap-1.5">
                <StatusDot tone={row.ok ? "verified" : "unknown"} />
                <span className={cn("font-medium", row.ok ? "text-success-text" : "text-ink-muted")}>{row.value}</span>
              </span>
            </div>
          ))}
        </div>
      </Panel>

      {/* Experiment Inventory */}
      <Panel title={t(locale, "Kaggle Experiment Inventory", "Kaggle 实验清单")}>
        <div className="grid grid-cols-2 gap-3 text-xs sm:grid-cols-3">
          {[
            { label: t(locale, "Tasks w/ Experiments", "有实验的任务"), value: inv?.task_count_with_experiments ?? 0 },
            { label: t(locale, "Scored Runs", "有分数运行"), value: inv?.total_scored_runs ?? 0 },
            { label: t(locale, "Promoted", "晋升"), value: inv?.total_promoted_runs ?? 0 },
            { label: t(locale, "Held", "保留"), value: inv?.total_held_runs ?? 0 },
            { label: t(locale, "Timeout/Failed", "超时/失败"), value: inv?.total_timeout_or_failed_runs ?? 0 },
            { label: t(locale, "Top-30 Count", "Top-30 数"), value: inv?.official_top30_count ?? 0 },
          ].map((row) => (
            <div key={row.label} className="rounded-md border border-edge px-2.5 py-2">
              <div className="text-ink-muted">{row.label}</div>
              <div className="mt-0.5 text-lg font-semibold tabular-nums text-ink">{row.value}</div>
            </div>
          ))}
        </div>
      </Panel>

      {/* New Competition Readiness (if present) */}
      {newComp && (
        <Panel title={t(locale, "New Competition Readiness", "新竞赛就绪")} accent={newComp.local_baseline_ready ? "green" : undefined}>
          <div className="space-y-1.5 text-xs">
            <div className="flex justify-between"><span className="text-ink-muted">{t(locale, "Competition", "竞赛")}</span><span className="font-mono text-ink-secondary">{newComp.competition_slug ?? newComp.task_id ?? "—"}</span></div>
            <div className="flex justify-between"><span className="text-ink-muted">{t(locale, "Train Rows", "训练行数")}</span><span className="tabular-nums text-ink-secondary">{newComp.train_rows ?? "—"}</span></div>
            <div className="flex justify-between"><span className="text-ink-muted">{t(locale, "Baseline", "基线")}</span><StatusBadgeV2 tone={newComp.local_baseline_ready ? "verified" : "pending"} size="xs">{newComp.local_baseline_ready ? t(locale, "Ready", "就绪") : t(locale, "Pending", "待定")}</StatusBadgeV2></div>
          </div>
        </Panel>
      )}

      {/* Kaggle submission — Human Gate */}
      <Panel title={t(locale, "Kaggle Submission (Human Gate)", "Kaggle 提交(人工闸门)")} accent="red" compact>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="text-2xs text-ink-secondary">
            {t(
              locale,
              "Official Kaggle submission requires explicit human approval. This UI never auto-submits, never fabricates ranks or medals, and shows Unknown when official data is unavailable.",
              "官方 Kaggle 提交需明确人工批准。本界面不会自动提交,不伪造排名或奖牌,官方数据缺失时显示未知。",
            )}
          </div>
          <button
            type="button"
            data-ui-action="blocked_kaggle_submit"
            className="flex items-center gap-1 rounded border border-danger/30 bg-danger/5 px-2 py-1 text-2xs font-medium text-danger-text"
          >
            <Lock className="h-3 w-3" /> {t(locale, "Blocked: Human Gate", "受阻:人工闸门")}
          </button>
        </div>
      </Panel>
    </div>
  );
}
