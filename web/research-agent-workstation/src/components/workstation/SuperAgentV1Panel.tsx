"use client";

import { useCallback, useEffect, useState } from "react";
import {
  AlertCircle,
  Bot,
  Boxes,
  Database,
  GitBranch,
  HardDrive,
  RefreshCw,
  RotateCcw,
  ShieldAlert,
  ShieldCheck,
  Wrench,
} from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusBadgeV2, type StatusTone } from "@/components/workstation/primitives/StatusBadge";
import {
  isSuperAgentV1Status,
  type SuperAgentKernelStatus,
  type SuperAgentMode,
  type SuperAgentMountHealth,
  type SuperAgentNodeStatus,
  type SuperAgentRunView,
  type SuperAgentV1Status,
} from "@/lib/super-agent-v1";
import { cn } from "@/lib/utils";

type Locale = "zh-CN" | "en-US";

export type SuperAgentV1PanelProps = {
  initialStatus?: SuperAgentV1Status | null;
  locale?: Locale;
  className?: string;
};

function copy(locale: Locale, english: string, chinese: string) {
  return locale === "zh-CN" ? chinese : english;
}

function modeTone(mode: SuperAgentMode): StatusTone {
  if (mode === "enabled") return "ready";
  if (mode === "shadow") return "pending";
  return "unknown";
}

function stateTone(state: SuperAgentKernelStatus | SuperAgentNodeStatus): StatusTone {
  if (state === "completed") return "verified";
  if (["running", "verifying", "canary_verifying"].includes(state)) return "running";
  if (state === "waiting_exact_gate") return "blocked";
  if (state === "repairing") return "pending";
  if (state === "planned" || state === "pending") return "draft";
  return "unknown";
}

function healthTone(health: SuperAgentMountHealth): StatusTone {
  if (health === "ready") return "ready";
  if (health === "degraded") return "blocked";
  return "unknown";
}

function statusLabel(locale: Locale, status: SuperAgentKernelStatus | SuperAgentNodeStatus) {
  const labels: Record<string, [string, string]> = {
    planned: ["Planned", "已规划"],
    pending: ["Pending", "待执行"],
    running: ["Running", "运行中"],
    verifying: ["Verifying", "验证中"],
    repairing: ["Repairing", "修复中"],
    canary_verifying: ["Canary", "灰度验证"],
    waiting_exact_gate: ["Exact Gate", "精确 Gate"],
    completed: ["Verified", "已验证"],
    unknown: ["Unknown", "未知"],
  };
  const [english, chinese] = labels[status] ?? labels.unknown!;
  return copy(locale, english, chinese);
}

function formatCount(value: number, locale: Locale) {
  return new Intl.NumberFormat(locale).format(value);
}

function formatBytes(value: number, locale: Locale) {
  if (value < 1024) return `${formatCount(value, locale)} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let current = value;
  let unit = -1;
  while (current >= 1024 && unit < units.length - 1) {
    current /= 1024;
    unit += 1;
  }
  return `${new Intl.NumberFormat(locale, { maximumFractionDigits: 1 }).format(current)} ${units[unit]}`;
}

function Metric({
  label,
  value,
  icon: Icon,
}: {
  label: string;
  value: string;
  icon: typeof Boxes;
}) {
  return (
    <div className="min-w-0 rounded-md border border-edge bg-surface-sunken px-3 py-2.5">
      <div className="flex items-center gap-2 text-2xs font-semibold uppercase tracking-wide text-ink-muted">
        <Icon className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
        <span className="truncate">{label}</span>
      </div>
      <div className="mt-1.5 font-mono text-lg font-semibold tabular-nums text-ink">{value}</div>
    </div>
  );
}

function Mounts({ status, locale }: { status: SuperAgentV1Status; locale: Locale }) {
  return (
    <section aria-labelledby="super-agent-mounts-heading">
      <div className="mb-2 flex flex-wrap items-end justify-between gap-2">
        <div>
          <h3 id="super-agent-mounts-heading" className="flex items-center gap-2 text-sm font-semibold text-ink">
            <HardDrive className="h-4 w-4 text-primary" aria-hidden="true" />
            {copy(locale, "Authorized directory mounts", "已授权目录挂载")}
          </h3>
          <p className="mt-0.5 text-2xs leading-4 text-ink-muted">
            {copy(locale, "Opaque roots; only directory IDs and granted operations are visible.", "真实根路径保持不可见，仅展示目录 ID 与已授权操作。")}
          </p>
        </div>
        {status.mount_list_truncated ? (
          <StatusBadgeV2 tone="stale" size="xs">{copy(locale, "List truncated", "列表已截断")}</StatusBadgeV2>
        ) : null}
      </div>

      {status.mounts.length === 0 ? (
        <div className="rounded-md border border-dashed border-edge bg-surface-sunken px-4 py-6 text-center text-xs text-ink-muted">
          {copy(locale, "No authorized directory mount is visible.", "当前没有可见的已授权目录挂载。")}
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-2 lg:grid-cols-2">
          {status.mounts.map((mount) => (
            <article key={mount.directory_id} className="min-w-0 rounded-md border border-edge bg-surface-raised p-3">
              <div className="flex min-w-0 items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="truncate font-mono text-xs font-semibold text-ink" title={mount.directory_id}>{mount.directory_id}</div>
                  <div className="mt-1 truncate font-mono text-2xs text-ink-muted" title={mount.connector_id}>{mount.connector_id}</div>
                </div>
                <StatusBadgeV2 tone={healthTone(mount.health)} size="xs">
                  {mount.health === "ready"
                    ? copy(locale, "Healthy", "健康")
                    : mount.health === "degraded"
                      ? copy(locale, "Degraded", "降级")
                      : copy(locale, "Mounted", "已挂载")}
                </StatusBadgeV2>
              </div>
              <div className="mt-3 flex flex-wrap gap-1" aria-label={copy(locale, "Granted operations", "已授权操作")}>
                {mount.operations.map((operation) => (
                  <span key={operation} className="rounded-sm border border-edge bg-surface-sunken px-1.5 py-0.5 font-mono text-2xs text-ink-secondary">
                    {operation}
                  </span>
                ))}
              </div>
              <dl className="mt-3 grid grid-cols-2 gap-x-3 gap-y-1 border-t border-edge pt-2 text-2xs">
                <dt className="text-ink-muted">{copy(locale, "File quota", "文件配额")}</dt>
                <dd className="text-right font-mono tabular-nums text-ink-secondary">{formatCount(mount.max_files, locale)}</dd>
                <dt className="text-ink-muted">{copy(locale, "Byte quota", "容量配额")}</dt>
                <dd className="text-right font-mono tabular-nums text-ink-secondary">{formatBytes(mount.max_bytes, locale)}</dd>
                <dt className="text-ink-muted">{copy(locale, "Link policy", "链接策略")}</dt>
                <dd className="text-right font-mono text-ink-secondary">deny</dd>
              </dl>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}

function ExactGate({ run, locale }: { run: SuperAgentRunView; locale: Locale }) {
  if (!run.exact_gate) return null;
  return (
    <div className="mt-3 rounded-md border border-danger/30 bg-danger-light/40 p-3" role="status">
      <div className="flex items-center gap-2 text-xs font-semibold text-danger-text">
        <ShieldAlert className="h-4 w-4 shrink-0" aria-hidden="true" />
        {copy(locale, "Exact human Gate", "精确人工 Gate")}
      </div>
      <dl className="mt-2 grid gap-2 text-xs sm:grid-cols-[110px_minmax(0,1fr)]">
        <dt className="font-semibold text-ink-muted">{copy(locale, "Reason", "原因")}</dt>
        <dd className="break-words text-ink-secondary">{run.exact_gate.reason}</dd>
        <dt className="font-semibold text-ink-muted">{copy(locale, "One action", "唯一操作")}</dt>
        <dd className="break-words text-ink-secondary">{run.exact_gate.required_action}</dd>
        <dt className="font-semibold text-ink-muted">{copy(locale, "Resume point", "恢复点")}</dt>
        <dd className="break-words font-mono text-ink-secondary">{run.exact_gate.resume_point}</dd>
      </dl>
    </div>
  );
}

function RunGraph({ run, locale }: { run: SuperAgentRunView; locale: Locale }) {
  return (
    <article className="rounded-md border border-edge bg-surface-raised p-3">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <div className="truncate font-mono text-xs font-semibold text-ink" title={run.run_id}>{run.run_id}</div>
          <p className="mt-1 break-words text-xs leading-5 text-ink-secondary">{run.objective || copy(locale, "Objective not recorded", "未记录任务目标")}</p>
        </div>
        <StatusBadgeV2 tone={stateTone(run.status)} size="xs" pulse={run.status === "running"}>
          {statusLabel(locale, run.status)}
        </StatusBadgeV2>
      </div>

      <ol className="mt-3 space-y-1.5" aria-label={copy(locale, "Task graph", "任务图")}>
        {run.nodes.map((node, index) => (
          <li key={node.id} className="grid min-w-0 grid-cols-[24px_minmax(0,1fr)_auto] items-start gap-2 rounded-sm border border-edge bg-surface-sunken px-2.5 py-2">
            <span className="flex h-6 w-6 items-center justify-center rounded-full border border-edge bg-surface-raised font-mono text-2xs text-ink-muted" aria-hidden="true">
              {index + 1}
            </span>
            <div className="min-w-0">
              <div className="break-words text-xs font-medium text-ink">{node.action}</div>
              <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-2xs text-ink-muted">
                <span className="font-mono">{node.id}</span>
                {node.dependencies.length > 0 ? <span>{copy(locale, "Depends on", "依赖")}: {node.dependencies.join(", ")}</span> : null}
                {node.capability_ids.length > 0 ? <span>{copy(locale, "Capabilities", "能力")}: {node.capability_ids.join(", ")}</span> : null}
              </div>
            </div>
            <StatusBadgeV2 tone={stateTone(node.status)} size="xs">{statusLabel(locale, node.status)}</StatusBadgeV2>
          </li>
        ))}
      </ol>

      {run.node_list_truncated ? (
        <p className="mt-2 text-2xs text-warning-text">{copy(locale, "Additional nodes are omitted from this read-only view.", "其余节点已从只读视图中省略。")}</p>
      ) : null}

      <div className="mt-3 grid grid-cols-2 gap-2" aria-label={copy(locale, "Recovery history", "恢复轨迹")}>
        <div className="rounded-sm border border-edge bg-surface-sunken p-2">
          <div className="flex items-center gap-1.5 text-2xs text-ink-muted">
            <AlertCircle className="h-3.5 w-3.5" aria-hidden="true" />
            {copy(locale, "Observed failures", "已观察失败")}
          </div>
          <div className="mt-1 font-mono text-sm font-semibold tabular-nums text-ink">{formatCount(run.failure_count, locale)}</div>
        </div>
        <div className="rounded-sm border border-edge bg-surface-sunken p-2">
          <div className="flex items-center gap-1.5 text-2xs text-ink-muted">
            <Wrench className="h-3.5 w-3.5" aria-hidden="true" />
            {copy(locale, "Repair actions", "修复动作")}
          </div>
          <div className="mt-1 font-mono text-sm font-semibold tabular-nums text-ink">{formatCount(run.repair_count, locale)}</div>
        </div>
      </div>
      <ExactGate run={run} locale={locale} />
    </article>
  );
}

export function SuperAgentV1Panel({ initialStatus = null, locale = "zh-CN", className }: SuperAgentV1PanelProps) {
  const [status, setStatus] = useState<SuperAgentV1Status | null>(
    isSuperAgentV1Status(initialStatus) ? initialStatus : null,
  );
  const [loading, setLoading] = useState(!status);
  const [error, setError] = useState(false);
  const [scopeRestricted, setScopeRestricted] = useState(false);

  const refresh = useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    setError(false);
    setScopeRestricted(false);
    try {
      const response = await fetch("/api/super-agent/status", {
        method: "GET",
        headers: { Accept: "application/json" },
        cache: "no-store",
        signal,
      });
      const payload = await response.json().catch(() => null) as unknown;
      if (response.status === 403) { setScopeRestricted(true); setStatus(null); return; }
      if (!response.ok || !isSuperAgentV1Status(payload)) throw new Error("status_unavailable");
      setStatus(payload);
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === "AbortError") return;
      setStatus(null);
      setError(true);
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void refresh(controller.signal);
    return () => controller.abort();
  }, [refresh]);

  return (
    <Card className={cn("overflow-hidden", className)} data-ui-component="super-agent-v1-panel" aria-busy={loading}>
      <CardHeader className="border-b border-edge bg-surface-sunken/60 pb-3.5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <Bot className="h-4 w-4 shrink-0 text-primary" aria-hidden="true" />
              <CardTitle>{copy(locale, "Super Agent V1", "通用超级 Agent V1")}</CardTitle>
              {status ? (
                <StatusBadgeV2 tone={modeTone(status.mode)} size="xs">
                  {status.mode === "enabled"
                    ? copy(locale, "Enabled", "已启用")
                    : status.mode === "shadow"
                      ? copy(locale, "Shadow", "影子模式")
                      : copy(locale, "Off", "未启用")}
                </StatusBadgeV2>
              ) : null}
            </div>
            <CardDescription>
              {copy(locale, "Read-only capability, task graph, recovery and Exact Gate projection.", "只读展示能力挂载、任务图、恢复轨迹与精确 Gate；本面板不会执行任何工具或写入操作。")}
            </CardDescription>
          </div>
          <button
            type="button"
            onClick={() => void refresh()}
            disabled={loading}
            className="inline-flex h-11 items-center gap-2 rounded-md border border-edge bg-surface-raised px-3 text-xs font-semibold text-ink-secondary transition-colors hover:border-primary/40 hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary disabled:cursor-not-allowed disabled:opacity-50"
            aria-label={copy(locale, "Refresh Super Agent status", "刷新超级 Agent 状态")}
          >
            <RefreshCw className={cn("h-3.5 w-3.5", loading && "animate-spin")} aria-hidden="true" />
            {copy(locale, "Refresh", "刷新")}
          </button>
        </div>
      </CardHeader>

      <CardContent className="space-y-5">
        <div className="min-h-5" aria-live="polite">
          {scopeRestricted ? (
            <div role="status" className="rounded-md border border-edge p-3 text-xs text-ink-secondary">
              {copy(locale, "The global runtime view is not available in this account scope. This is an access boundary, not a model outage. Open a task's execution and audit views in Assistant.", "全局运行时视图尚未向当前账户范围开放。这是访问边界，不代表模型故障；请在助手中打开具体任务的执行与审计记录。")}
              <a href="/?page=projects&demo=user" className="ml-2 underline">{copy(locale, "Open my tasks", "查看我的任务")}</a>
            </div>
          ) : error ? (
            <div className="flex items-start gap-2 rounded-md border border-danger/30 bg-danger-light/40 px-3 py-2 text-xs text-danger-text" role="alert">
              <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
              <span>{copy(locale, "Managed runtime status is unavailable or failed contract validation. No cached state is presented as current.", "受管运行时状态不可用或未通过契约校验；不会把缓存状态标记为当前状态。")}</span>
            </div>
          ) : loading && !status ? (
            <div className="text-xs text-ink-muted">{copy(locale, "Loading verified status…", "正在加载已验证状态…")}</div>
          ) : null}
        </div>

        {!status ? (
          <div className="rounded-md border border-dashed border-edge bg-surface-sunken px-4 py-8 text-center">
            <RotateCcw className="mx-auto h-5 w-5 text-ink-muted" aria-hidden="true" />
            <p className="mt-2 text-xs text-ink-muted">{copy(locale, "No verified Super Agent snapshot is available.", "暂无已验证的超级 Agent 快照。")}</p>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
              <Metric label={copy(locale, "Capabilities", "能力总数")} value={formatCount(status.capability_count, locale)} icon={Boxes} />
              <Metric label={copy(locale, "Directory mounts", "目录挂载")} value={formatCount(status.directory_count, locale)} icon={Database} />
              <Metric label={copy(locale, "Active runs", "活动任务")} value={formatCount(status.active_run_ids.length, locale)} icon={GitBranch} />
              <Metric
                label={copy(locale, "Schema migration", "结构迁移")}
                value={status.migration_applied ? copy(locale, "Applied", "已应用") : copy(locale, "Blocked", "未应用")}
                icon={status.migration_applied ? ShieldCheck : ShieldAlert}
              />
            </div>

            <Mounts status={status} locale={locale} />

            <section aria-labelledby="super-agent-runs-heading">
              <div className="mb-2 flex flex-wrap items-end justify-between gap-2">
                <div>
                  <h3 id="super-agent-runs-heading" className="flex items-center gap-2 text-sm font-semibold text-ink">
                    <GitBranch className="h-4 w-4 text-primary" aria-hidden="true" />
                    {copy(locale, "Task graphs and recovery", "任务图与恢复轨迹")}
                  </h3>
                  <p className="mt-0.5 text-2xs leading-4 text-ink-muted">
                    {copy(locale, "Only active durable runs are listed; counts are evidence summaries, not success claims.", "仅列出活动的持久化任务；计数是证据摘要，不代表任务已成功。")}
                  </p>
                </div>
                {status.run_list_truncated ? (
                  <StatusBadgeV2 tone="stale" size="xs">{copy(locale, "List truncated", "列表已截断")}</StatusBadgeV2>
                ) : null}
              </div>

              {status.runs.length === 0 ? (
                <div className="rounded-md border border-dashed border-edge bg-surface-sunken px-4 py-6 text-center text-xs text-ink-muted">
                  {copy(locale, "No active durable task graph is recorded.", "当前没有活动的持久化任务图。")}
                </div>
              ) : (
                <div className="space-y-2">
                  {status.runs.map((run) => <RunGraph key={run.run_id} run={run} locale={locale} />)}
                </div>
              )}
            </section>
          </>
        )}
      </CardContent>
    </Card>
  );
}
