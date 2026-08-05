"use client";

import {
  Cpu,
  FileText,
  HardDrive,
  Lock,
  MonitorSmartphone,
  Server,
  Signal,
} from "lucide-react";
import type { WorkstationSummary } from "@/lib/api/types";
import { PageHeader, Panel, MetricTile } from "../primitives/Layout";
import { StatusBadgeV2, type StatusTone } from "../primitives/StatusBadge";
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

function connectorEntry(summary: WorkstationSummary | null | undefined, key: string) {
  return (summary?.connector_status as Record<string, Record<string, unknown>> | undefined)?.[key];
}

function connectorTone(entry: Record<string, unknown> | undefined): StatusTone {
  if (!entry) return "unknown";
  const configured = Boolean(entry.configured);
  const state = String(entry.state ?? entry.status ?? "").toLowerCase();
  if (configured && (state.includes("verified") || state.includes("ready") || state.includes("passed"))) return "verified";
  if (configured) return "ready";
  if (state.includes("not_configured")) return "unknown";
  if (state.includes("blocked") || state.includes("failed")) return "blocked";
  return "pending";
}

function connectorLabel(locale: Locale | undefined, entry: Record<string, unknown> | undefined): string {
  if (!entry) return t(locale, "Unknown/Blocked", "未知/阻断");
  const configured = Boolean(entry.configured);
  const state = String(entry.state ?? "").toLowerCase();
  if (configured && state.includes("verified")) return t(locale, "Verified", "已验证");
  if (configured && (state.includes("ready") || state.includes("passed"))) return t(locale, "Ready", "就绪");
  if (configured) return t(locale, "Configured", "已配置");
  if (state.includes("not_configured")) return t(locale, "Not configured", "未配置");
  if (state.includes("blocked")) return t(locale, "Blocked", "阻断");
  return t(locale, "Unknown/Blocked", "未知/阻断");
}

export function GpuHpcScreen(props: ScreenProps) {
  const { summary, locale = "zh-CN" } = props;
  const hpcEntry = connectorEntry(summary, "local_hpc");
  const runs = summary?.runs ?? [];
  const runningJobs = runs.filter((r) => String(r.status ?? "").toLowerCase().includes("running"));

  return (
    <div className="space-y-4">
      <PageHeader
        title={t(locale, "GPU / HPC", "GPU / HPC")}
        subtitle={t(locale, "Compute infrastructure and job monitoring", "计算基础设施与作业监控")}
        breadcrumb={`${t(locale, "Infrastructure", "基础设施")} > ${t(locale, "GPU / HPC", "GPU / HPC")}`}
        primaryAction={
          <div className="flex items-center gap-1.5">
            <button
              type="button"
              data-ui-action="compute_select_hpc_gpu"
              className="flex items-center gap-1 rounded-md border border-edge px-2.5 py-1 text-xs font-medium text-ink-secondary hover:bg-surface-sunken"
            >
              <Server className="h-3 w-3" /> {t(locale, "HPC GPU", "HPC GPU")}
            </button>
            <button
              type="button"
              data-ui-action="compute_select_local"
              className="flex items-center gap-1 rounded-md border border-edge px-2.5 py-1 text-xs font-medium text-ink-secondary hover:bg-surface-sunken"
            >
              <MonitorSmartphone className="h-3 w-3" /> {t(locale, "Local", "本地")}
            </button>
            <button
              type="button"
              data-ui-action="gpu_view_job_manifest_yaml"
              className="flex items-center gap-1 rounded-md border border-edge px-2.5 py-1 text-xs font-medium text-ink-secondary hover:bg-surface-sunken"
            >
              <FileText className="h-3 w-3" /> YAML
            </button>
          </div>
        }
      />

      {/* Connector status */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricTile
          label={t(locale, "HPC Connector", "HPC 连接器")}
          value={connectorLabel(locale, hpcEntry)}
          icon={Server}
          tone={connectorTone(hpcEntry)}
        />
        <MetricTile
          label={t(locale, "Total Runs", "总运行")}
          value={runs.length}
          icon={HardDrive}
          tone="neutral"
        />
        <MetricTile
          label={t(locale, "Running Jobs", "运行中作业")}
          value={runningJobs.length}
          icon={Cpu}
          tone={runningJobs.length > 0 ? "blue" : "neutral"}
        />
        <MetricTile
          label={t(locale, "Signal", "信号")}
          value={hpcEntry ? t(locale, "Connected", "已连接") : t(locale, "Unknown/Blocked", "未知/阻断")}
          icon={Signal}
          tone={hpcEntry ? "verified" : "unknown"}
        />
      </div>

      {/* HPC detail */}
      <Panel title={t(locale, "HPC Connector Detail", "HPC 连接器详情")}>
        {hpcEntry ? (
          <div className="space-y-2 text-xs">
            {Object.entries(hpcEntry).map(([key, val]) => (
              <div key={key} className="flex items-center justify-between gap-2">
                <span className="text-ink-muted">{key}</span>
                <span className="font-mono text-ink-secondary">
                  {typeof val === "boolean" ? (val ? t(locale, "Yes", "是") : t(locale, "No", "否")) : String(val ?? "—")}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div className="py-4 text-center text-sm text-ink-muted">
            {t(locale, "No HPC connector data. Resource may be unverified or blocked.", "无 HPC 连接器数据。资源可能未验证或被阻断。")}
          </div>
        )}
      </Panel>

      {/* Job table */}
      <Panel title={t(locale, "Job History", "作业历史")}>
        {runs.length === 0 ? (
          <div className="py-4 text-center text-sm text-ink-muted">{t(locale, "No runs recorded", "无运行记录")}</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-edge text-left text-ink-muted">
                  <th className="pb-2 pr-3 font-medium">{t(locale, "Task", "任务")}</th>
                  <th className="pb-2 pr-3 font-medium">{t(locale, "Status", "状态")}</th>
                  <th className="pb-2 pr-3 font-medium">{t(locale, "Best Model", "最佳模型")}</th>
                  <th className="pb-2 font-medium">{t(locale, "Started", "开始")}</th>
                </tr>
              </thead>
              <tbody>
                {runs.slice(0, 20).map((run) => (
                  <tr key={run.id ?? run.task_id} className="border-b border-edge/50">
                    <td className="py-2 pr-3 font-medium text-ink">{run.task_id}</td>
                    <td className="py-2 pr-3">
                      <StatusBadgeV2
                        tone={String(run.status ?? "").toLowerCase().includes("running") ? "running"
                          : String(run.status ?? "").toLowerCase().includes("complete") ? "verified"
                          : String(run.status ?? "").toLowerCase().includes("fail") ? "failed" : "unknown"}
                        size="xs"
                      >
                        {run.status ?? "unknown"}
                      </StatusBadgeV2>
                    </td>
                    <td className="py-2 pr-3 font-mono text-ink-secondary">{run.best_model ?? "—"}</td>
                    <td className="py-2 text-ink-secondary">{run.started_at ? new Date(run.started_at).toLocaleString() : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      {/* Start Training — Human Gate */}
      <Panel title={t(locale, "Start Training (Human Gate)", "启动训练(人工闸门)")} accent="red" compact>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="text-2xs text-ink-secondary">
            {t(
              locale,
              "GPU training jobs are only launched after explicit human approval. This UI never auto-starts training and never runs a local GPU.",
              "GPU 训练作业仅在明确人工批准后启动。本界面不会自动启动训练,也不在本机使用 GPU。",
            )}
          </div>
          <button
            type="button"
            data-ui-action="blocked_start_training"
            className="flex items-center gap-1 rounded border border-danger/30 bg-danger/5 px-2 py-1 text-2xs font-medium text-danger-text"
          >
            <Lock className="h-3 w-3" /> {t(locale, "Blocked: Human Gate", "受阻:人工闸门")}
          </button>
        </div>
      </Panel>
    </div>
  );
}
