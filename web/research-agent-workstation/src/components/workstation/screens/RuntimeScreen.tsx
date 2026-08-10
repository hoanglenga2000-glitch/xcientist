"use client";

import { useEffect, useMemo, useState } from "react";

import {
  Activity,
  CheckCircle2,
  Cpu,
  Eye,
  FileCheck2,
  FileText,
  GitBranch,
  Lock,
  PackageOpen,
  RefreshCw,
  Server,
  Workflow,
  X,
} from "lucide-react";
import type { WorkstationSummary } from "@/lib/api/types";
import { PageHeader, Panel, MetricTile, CopyablePath } from "../primitives/Layout";
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

function runtimeTone(status: string | undefined): StatusTone {
  const normalized = String(status ?? "").toLowerCase();
  if (normalized.includes("running") || normalized.includes("live")) return "running";
  if (normalized.includes("completed") || normalized.includes("ready") || normalized.includes("passed")) return "verified";
  if (normalized.includes("failed") || normalized.includes("error") || normalized.includes("rejected")) return "failed";
  if (normalized.includes("blocked") || normalized.includes("forbidden")) return "blocked";
  if (normalized.includes("continuation") || normalized.includes("retry")) return "stale";
  if (normalized.includes("idle") || normalized.includes("pending")) return "pending";
  return "unknown";
}

function formatScore(value: number | undefined): string {
  return typeof value === "number" ? value.toFixed(6) : "-";
}

function formatMetric(value: unknown, digits = 2): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "-";
}

function formatBytes(value: number | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(2)} GiB`;
  if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(2)} MiB`;
  if (value >= 1024) return `${(value / 1024).toFixed(1)} KiB`;
  return `${value} B`;
}

type ReplayState = {
  seq: number;
  telemetryIndex: number;
  finished: boolean;
};

function eventSeq(event: Record<string, unknown>): number {
  return typeof event.seq === "number" ? event.seq : Number(event.seq ?? 0);
}

function eventStatus(event: Record<string, unknown>): string {
  return typeof event.status === "string" ? event.status : "event";
}

export function RuntimeScreen(props: ScreenProps) {
  const { summary, locale = "zh-CN" } = props;
  const runtime = summary?.runtime;
  const terminalAgent = summary?.terminal_agent;
  const currentRun = runtime?.current_run;
  const taskGraph = runtime?.task_graph;
  const sourceTaskNodes = useMemo(() => taskGraph?.nodes ?? [], [taskGraph?.nodes]);
  const handoffs = runtime?.handoffs ?? [];
  const runtimeSnapshot = runtime?.runtime_snapshot;
  const review = runtime?.review;
  const claimAudit = runtimeSnapshot?.llm?.claim_audit ?? review?.claim_audit;
  const claimAuditStatus = typeof claimAudit?.status === "string"
    ? claimAudit.status
    : runtimeSnapshot?.gates?.claim_audit ?? "pending";
  const hpcProbe = runtime?.hpc_probe;
  const gpu = hpcProbe?.gpu_inventory?.[0];
  const metrics = runtimeSnapshot?.metrics;
  const llm = runtimeSnapshot?.llm;
  const isLlmRun = llm?.task_type === "llm_finetune";
  const candidates = metrics?.candidates ?? review?.accepted_candidates ?? [];
  const artifactManifest = runtime?.artifact_manifest;
  const artifacts = artifactManifest?.artifacts ?? [];
  const publicArtifactPreviews = runtime?.public_artifact_previews ?? [];
  const hasCurrentRun = Boolean(currentRun?.run_id && currentRun?.task_id);
  const telemetry = useMemo(
    () => (llm?.telemetry ?? []).filter((sample) => (
      typeof sample.step === "number" && typeof sample.loss === "number"
    )),
    [llm?.telemetry],
  );
  const [publicPresentation, setPublicPresentation] = useState(false);
  const [selectedPreviewId, setSelectedPreviewId] = useState<string | null>(null);
  const [replayEnabled, setReplayEnabled] = useState(false);
  const [replayState, setReplayState] = useState<ReplayState>({ seq: 0, telemetryIndex: 0, finished: false });

  useEffect(() => {
    const params = new URL(window.location.href).searchParams;
    const isPublic = params.get("presentation") === "public";
    const shouldReplay = params.get("replay") === "1";
    setPublicPresentation(isPublic);
    setReplayEnabled(shouldReplay);
    setReplayState({
      seq: shouldReplay ? 0 : Number(currentRun?.last_seq ?? 0),
      telemetryIndex: shouldReplay ? 0 : Math.max(0, telemetry.length - 1),
      finished: !shouldReplay,
    });
  }, [currentRun?.run_id, currentRun?.last_seq, telemetry.length]);

  useEffect(() => {
    if (!replayEnabled || replayState.finished) return;
    const maxSeq = Number(currentRun?.last_seq ?? 0);
    const timer = window.setTimeout(() => {
      setReplayState((state) => {
        // The real training occupied most of the original run. Hold the event
        // ledger at hpc_train/running while replaying every recorded sample.
        if (state.seq === 29 && state.telemetryIndex < Math.max(0, telemetry.length - 1)) {
          return { ...state, telemetryIndex: state.telemetryIndex + 1 };
        }
        if (state.seq >= maxSeq) return { ...state, finished: true };
        return { ...state, seq: state.seq + 1 };
      });
    }, replayState.seq === 29 ? 360 : 460);
    return () => window.clearTimeout(timer);
  }, [currentRun?.last_seq, replayEnabled, replayState, telemetry.length]);

  // The pointer-backed run is authoritative. Legacy Scientist/Terminal events
  // remain available only when the workspace has no current multi-agent run.
  const sourceEventLog = hasCurrentRun
    ? runtime?.event_log ?? []
    : terminalAgent?.recent_terminal_events?.length
      ? terminalAgent.recent_terminal_events
      : terminalAgent?.recent_events?.length
        ? terminalAgent.recent_events
        : runtime?.event_log ?? [];
  const eventLog = replayEnabled
    ? sourceEventLog.filter((event) => eventSeq(event) <= replayState.seq)
    : sourceEventLog;
  const eventCount = replayEnabled
    ? replayState.seq
    : hasCurrentRun
      ? currentRun?.last_seq ?? eventLog.length
      : terminalAgent?.terminal_event_count ?? terminalAgent?.event_count ?? eventLog.length;
  const visibleEvents = eventLog.slice(-10).reverse();
  const taskNodes = useMemo(() => {
    if (!replayEnabled) return sourceTaskNodes;
    return sourceTaskNodes.map((node) => {
      const stateEvent = eventLog
        .filter((event) => event.schema === "evomind.multi_agent.task.state.v1" && event.task_id === node.task_id)
        .at(-1);
      return { ...node, status: stateEvent ? eventStatus(stateEvent) : "pending" };
    });
  }, [eventLog, replayEnabled, sourceTaskNodes]);
  const completedTasks = taskNodes.filter((node) => node.status === "completed").length;
  const activeTasks = taskNodes.filter((node) => node.status === "running");
  const publicRunId = "EVOMIND-DEMO-7B-20260722";
  const publicTaskId = "evomind-qwen7b-finetune";
  const reportPath = currentRun?.run_dir ? `${currentRun.run_dir}/research_report.md` : "research_report.md";
  const submissionPath = currentRun?.run_dir ? `${currentRun.run_dir}/submission.csv` : "submission.csv";
  const adapterPath = currentRun?.run_dir ? `${currentRun.run_dir}/llm_output/adapter/` : "llm_output/adapter/";
  const modelCardPath = currentRun?.run_dir ? `${currentRun.run_dir}/model_card.md` : "model_card.md";
  const datasetCounts = llm?.dataset?.counts;
  const beforeDomain = llm?.before_after_eval?.before?.domain_composite;
  const afterDomain = llm?.before_after_eval?.after?.domain_composite;
  const adapterReloadPassed = llm?.adapter_reload?.passed === true;
  const canonicalRunStatus = String(currentRun?.status ?? runtime?.task_state?.status ?? "unknown");
  const displayedRunStatus = replayEnabled ? (replayState.finished ? "completed" : "running") : canonicalRunStatus;
  const runCompleted = displayedRunStatus.toLowerCase() === "completed";
  const replayTelemetry = telemetry[Math.min(replayState.telemetryIndex, Math.max(0, telemetry.length - 1))] as Record<string, unknown> | undefined;
  const replayGpu = Array.isArray(replayTelemetry?.gpu)
    ? replayTelemetry?.gpu?.[0] as Record<string, unknown> | undefined
    : undefined;
  const displayedTraining: {
    step?: number | null;
    loss?: number | null;
    gpu_memory_mb?: number | null;
    utilization?: number | string | null;
  } = replayEnabled
    ? replayState.seq < 29
      ? {}
      : replayState.seq < 33
        ? {
            step: typeof replayTelemetry?.step === "number" ? replayTelemetry.step : 0,
            loss: typeof replayTelemetry?.loss === "number" ? replayTelemetry.loss : null,
            gpu_memory_mb: typeof replayTelemetry?.cuda_max_allocated_mb === "number" ? replayTelemetry.cuda_max_allocated_mb : null,
            utilization: typeof replayGpu?.utilization_percent === "number" || typeof replayGpu?.utilization_percent === "string"
              ? replayGpu.utilization_percent
              : null,
          }
        : llm?.training ?? {}
    : llm?.training ?? {};
  const replayReviewStatus = replayEnabled && replayState.seq < 45 ? "pending" : review?.status ?? "pending";
  const replayClaimStatus = replayEnabled && replayState.seq < 51 ? "pending" : claimAuditStatus;
  const replayArtifactCount = replayEnabled && replayState.seq < 57 ? 0 : artifacts.length;
  const replayEvaluationReady = !replayEnabled || replayState.seq >= 39;
  const replayAdapterReady = !replayEnabled || replayState.seq >= 39;
  const replayDeliverablesReady = !replayEnabled || replayState.seq >= 57;
  const replayHandoffCount = replayEnabled
    ? eventLog.filter((event) => String(event.schema ?? "").includes("handoff.created")).length
    : handoffs.length;
  const selectedPreview = publicArtifactPreviews.find((item) => item.id === selectedPreviewId) ?? null;

  const recordRuntimeAction = (action: string, metadata: Record<string, unknown> = {}) => {
    void props.runWorkstationAction?.(action, {
      task_id: currentRun?.task_id ?? props.selectedTask,
      run_id: currentRun?.run_id,
      source: "runtime_screen",
      ...metadata,
    });
  };

  return (
    <div className="space-y-4" data-presentation-mode={publicPresentation ? "public" : "internal"} data-replay-seq={eventCount}>
      <PageHeader
        title={t(locale, "Agent Runtime", "Agent 运行时")}
        subtitle={publicPresentation
          ? t(locale, "Public evidence-ledger replay from a completed real run", "真实运行证据账本公开回放")
          : t(locale, "Authoritative multi-agent execution, review, and evidence", "当前 Multi-Agent 运行、审核与证据的权威视图")}
        breadcrumb={`${t(locale, "Research Loop", "研究循环")} > ${t(locale, "Agent Runtime", "Agent 运行时")}`}
        primaryAction={
          <button
            type="button"
            data-ui-action="runtime_refresh_5s"
            data-ui-skip-action="true"
            onClick={() => {
              recordRuntimeAction("runtime_refresh_5s");
              void props.refreshSummary?.();
            }}
            className="flex items-center gap-1 rounded-md border border-edge px-2.5 py-1 text-xs font-medium text-ink-secondary hover:bg-surface-sunken"
          >
            <RefreshCw className="h-3 w-3" /> {t(locale, "Refresh", "刷新")}
          </button>
        }
      />

      {publicPresentation && (
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-accent/30 bg-accent/5 px-3 py-2 text-xs">
          <div className="flex items-center gap-2 text-ink-secondary">
            <Activity className="h-3.5 w-3.5 text-accent" />
            <span className="font-semibold">真实运行证据账本回放</span>
            <span className="text-ink-muted">原始事件与训练遥测 · 加速展示</span>
          </div>
          <StatusBadgeV2 tone={runCompleted ? "verified" : "running"} size="xs">
            {runCompleted ? "回放完成" : `seq ${eventCount}/${currentRun?.last_seq ?? sourceEventLog.length}`}
          </StatusBadgeV2>
        </div>
      )}

      <Panel
        title={t(locale, "Current Run", "当前运行")}
        description={publicPresentation
          ? t(locale, "Anonymized public alias; the private evidence ledger remains unchanged.", "公开别名展示；原始证据账本保持不变。")
          : t(locale, "Bound to the Run Ledger; historical tasks cannot replace this view.", "绑定 Run Ledger，历史任务不会覆盖此视图。")}
        accent={hasCurrentRun ? "green" : "amber"}
        icon={Workflow}
      >
        {hasCurrentRun ? (
          <div className="grid min-w-0 gap-3 lg:grid-cols-[minmax(0,1.35fr)_minmax(260px,0.65fr)]">
            <div className="min-w-0 space-y-2">
              <div className="flex flex-wrap items-center gap-2">
                <StatusBadgeV2 tone={runtimeTone(displayedRunStatus)}>{displayedRunStatus}</StatusBadgeV2>
                <span className="text-xs font-semibold text-ink-secondary">{publicPresentation ? publicTaskId : currentRun?.task_id}</span>
                <span className="text-2xs text-ink-muted">seq {eventCount}</span>
              </div>
              <CopyablePath path={publicPresentation ? publicRunId : currentRun?.run_id ?? "unknown-run"} className="max-w-full" />
              {publicPresentation
                ? <div className="rounded-md border border-edge bg-surface-sunken px-2.5 py-2 font-mono text-xs text-ink-secondary">本地交付目录 / 已脱敏</div>
                : currentRun?.run_dir && <CopyablePath path={currentRun.run_dir} className="max-w-full" />}
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="rounded-md border border-edge bg-surface-sunken p-2">
                <div className="text-2xs text-ink-muted">{t(locale, "Reviewer", "独立审核")}</div>
                <div className="mt-1 font-semibold text-ink">{replayReviewStatus}</div>
              </div>
              <div className="rounded-md border border-edge bg-surface-sunken p-2">
                <div className="text-2xs text-ink-muted">Claim Audit</div>
                <div className="mt-1 font-semibold text-ink">{replayClaimStatus}</div>
              </div>
              {isLlmRun ? (
                <>
                  <div className="rounded-md border border-edge bg-surface-sunken p-2">
                    <div className="text-2xs text-ink-muted">Base Model</div>
                    <div className="mt-1 truncate font-mono font-semibold text-ink" title={llm?.base_model ?? undefined}>{llm?.base_model ?? "-"}</div>
                  </div>
                  <div className="rounded-md border border-edge bg-surface-sunken p-2">
                    <div className="text-2xs text-ink-muted">Before / After</div>
                    <div className="mt-1 font-mono font-semibold tabular-nums text-ink">{replayEvaluationReady ? `${formatMetric(beforeDomain)} → ${formatMetric(afterDomain)}` : "pending"}</div>
                  </div>
                </>
              ) : (
                <>
                  <div className="rounded-md border border-edge bg-surface-sunken p-2">
                    <div className="text-2xs text-ink-muted">{t(locale, "Selected", "最佳方案")}</div>
                    <div className="mt-1 font-mono font-semibold text-ink">{metrics?.selected_solution ?? artifactManifest?.selected_solution ?? "-"}</div>
                  </div>
                  <div className="rounded-md border border-edge bg-surface-sunken p-2">
                    <div className="text-2xs text-ink-muted">CV {metrics?.metric ?? "score"}</div>
                    <div className="mt-1 font-mono font-semibold tabular-nums text-ink">{formatScore(metrics?.cv_score)}</div>
                  </div>
                </>
              )}
            </div>
          </div>
        ) : (
          <p className="text-xs text-ink-muted">{t(locale, "No pointer-backed multi-agent run is active.", "当前没有指针绑定的 Multi-Agent 运行。")}</p>
        )}
      </Panel>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <MetricTile label={t(locale, "DAG Tasks", "DAG 任务")} value={`${completedTasks}/${taskNodes.length}`} icon={GitBranch} tone={completedTasks === taskNodes.length && taskNodes.length > 0 ? "green" : "blue"} />
        <MetricTile label={t(locale, "Handoffs", "任务交接")} value={replayHandoffCount} icon={Workflow} tone="neutral" />
        <MetricTile label={t(locale, "Events", "连续事件")} value={eventCount} icon={Activity} tone="neutral" detail={eventLog.length ? `1-${eventCount}` : undefined} />
        <MetricTile label={t(locale, "Artifacts", "校验产物")} value={replayArtifactCount} icon={FileCheck2} tone="green" />
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1.45fr)_minmax(300px,0.55fr)]">
        <div className="min-w-0 space-y-4">
          <Panel
            title={t(locale, "Dynamic Task Graph", "动态任务图")}
            description={`${taskNodes.length} ${t(locale, "nodes", "节点")} / ${taskGraph?.edges?.length ?? 0} ${t(locale, "dependencies", "依赖")}`}
            action={activeTasks.length ? <StatusBadgeV2 tone="running">{activeTasks.length} active</StatusBadgeV2> : <StatusBadgeV2 tone={completedTasks === taskNodes.length && taskNodes.length > 0 ? "verified" : "pending"}>{completedTasks === taskNodes.length && taskNodes.length > 0 ? "complete" : "pending"}</StatusBadgeV2>}
          >
            <div className="grid max-h-72 gap-1.5 overflow-y-auto pr-1 sm:grid-cols-2">
              {taskNodes.map((node) => (
                <div key={node.task_id} className="flex min-w-0 items-center justify-between gap-2 rounded-md border border-edge px-2.5 py-2 text-xs">
                  <div className="flex min-w-0 items-center gap-2">
                    <StatusDot tone={runtimeTone(node.status)} />
                    <div className="min-w-0">
                      <div className="truncate font-mono font-semibold text-ink" title={node.task_id}>{node.task_id}</div>
                      <div className="truncate text-2xs text-ink-muted">{node.role ?? node.resource_type ?? "agent"}</div>
                    </div>
                  </div>
                  <StatusBadgeV2 tone={runtimeTone(node.status)} size="xs">{node.status ?? "unknown"}</StatusBadgeV2>
                </div>
              ))}
              {taskNodes.length === 0 && <span className="text-xs text-ink-muted">{t(locale, "No DAG loaded", "尚未加载 DAG")}</span>}
            </div>
          </Panel>

          <Panel title={t(locale, "Current Run Event Stream", "当前运行事件流")} description={t(locale, "Monotonic sequence from the selected run only", "仅展示当前运行的单调事件序列")}>
            <div className="space-y-1.5">
              {visibleEvents.map((event, index) => {
                const item = event as Record<string, unknown>;
                const seq = String(item.seq ?? eventCount - index);
                const label = String(item.agent ?? item.task_id ?? item.schema ?? `Event ${seq}`);
                return (
                  <div key={`${seq}-${String(item.schema ?? index)}`} className="grid min-w-0 grid-cols-[40px_minmax(0,1fr)_auto] items-center gap-2 rounded-sm border border-edge px-2.5 py-1.5 text-xs">
                    <span className="font-mono tabular-nums text-ink-muted">#{seq}</span>
                    <div className="min-w-0">
                      <div className="truncate text-ink-secondary">{label}</div>
                      <div className="truncate text-2xs text-ink-muted">{String(item.task_id ?? item.schema ?? "")}</div>
                    </div>
                    <StatusBadgeV2 tone={runtimeTone(String(item.status ?? ""))} size="xs">{String(item.status ?? "event")}</StatusBadgeV2>
                  </div>
                );
              })}
              {eventLog.length === 0 && <span className="text-xs text-ink-muted">{t(locale, "No current-run events", "当前运行暂无事件")}</span>}
            </div>
          </Panel>
        </div>

        <div className="min-w-0 space-y-4">
          <Panel title={t(locale, "HPC Runtime", "HPC 运行环境")} accent={hpcProbe?.status === "passed" ? "green" : "amber"} icon={Server} compact>
            <div className="space-y-2 text-xs">
              <div className="flex items-center justify-between gap-2">
                <span className="text-ink-muted">{t(locale, "Probe", "资源探针")}</span>
                <StatusBadgeV2 tone={runtimeTone(hpcProbe?.status)} size="xs">{hpcProbe?.status ?? "unknown"}</StatusBadgeV2>
              </div>
              <div className="flex items-center justify-between gap-2">
                <span className="text-ink-muted">GPU</span>
                <span className="truncate font-mono font-semibold text-ink-secondary" title={publicPresentation ? "Remote GPU" : gpu?.name}>{publicPresentation ? t(locale, "Remote GPU", "远程 GPU") : gpu?.name ?? "-"}</span>
              </div>
              {publicPresentation && (
                <div className="flex items-center justify-between gap-2">
                  <span className="text-ink-muted">{t(locale, "Recommended", "推荐配置")}</span>
                  <span className="font-mono font-semibold text-ink-secondary">NVIDIA A40 48GB</span>
                </div>
              )}
              <div className="flex items-center justify-between gap-2">
                <span className="text-ink-muted">CUDA</span>
                <span className="font-mono text-ink-secondary">{hpcProbe?.torch?.cuda_available ? `${hpcProbe.torch.device_count ?? 0} device` : "unavailable"}</span>
              </div>
              <div className="flex items-center justify-between gap-2">
                <span className="text-ink-muted">{t(locale, "Local GPU", "本地 GPU")}</span>
                <span className="font-semibold text-success-text">{t(locale, "not used", "未使用")}</span>
              </div>
              <button
                type="button"
                disabled
                data-ui-action="blocked_submit_gpu_job"
                className="flex w-full cursor-not-allowed items-center gap-2 rounded-md border border-danger/30 bg-danger/5 px-3 py-2 text-xs font-medium text-danger-text"
              >
                <Lock className="h-3.5 w-3.5" />
                {t(locale, "Submit GPU Job (Human Gate)", "提交 GPU Job（人工闸门）")}
              </button>
            </div>
          </Panel>

          {isLlmRun ? (
            <Panel title={t(locale, "QLoRA Training Facts", "QLoRA 训练事实")} icon={CheckCircle2} compact>
              <div className="space-y-2 text-xs">
                <div className="grid grid-cols-2 gap-2">
                  <div className="rounded-md border border-edge p-2"><div className="text-2xs text-ink-muted">Dataset</div><div className="mt-1 font-mono font-semibold">{datasetCounts?.train ?? "-"}/{datasetCounts?.validation ?? "-"}/{datasetCounts?.test ?? "-"}</div></div>
                  <div className="rounded-md border border-edge p-2"><div className="text-2xs text-ink-muted">Method</div><div className="mt-1 font-semibold">{llm?.qlora_config?.method ?? "NF4 4-bit QLoRA"}</div></div>
                  <div className="rounded-md border border-edge p-2"><div className="text-2xs text-ink-muted">Step / Loss</div><div className="mt-1 font-mono font-semibold">{displayedTraining?.step ?? "-"} / {formatMetric(displayedTraining?.loss, 4)}</div></div>
                  <div className="rounded-md border border-edge p-2"><div className="text-2xs text-ink-muted">GPU Memory</div><div className="mt-1 font-mono font-semibold">{formatMetric(displayedTraining?.gpu_memory_mb)} MiB</div></div>
                </div>
                {replayEnabled && replayState.seq >= 29 && replayState.seq < 33 && (
                  <div className="space-y-1 rounded-md border border-accent/30 bg-accent/5 p-2">
                    <div className="flex items-center justify-between font-mono text-2xs text-ink-secondary">
                      <span>训练遥测 {replayState.telemetryIndex + 1}/{telemetry.length}</span>
                      <span>util {String(displayedTraining?.utilization ?? "-")}%</span>
                    </div>
                    <div className="h-1.5 overflow-hidden rounded-full bg-surface-sunken">
                      <div className="h-full bg-accent transition-all duration-300" style={{ width: `${Math.min(100, Number(displayedTraining?.step ?? 0))}%` }} />
                    </div>
                  </div>
                )}
                <div className="flex items-center justify-between gap-2"><span className="text-ink-muted">Adapter Reload</span><StatusBadgeV2 tone={adapterReloadPassed && replayAdapterReady ? "verified" : "pending"} size="xs">{adapterReloadPassed && replayAdapterReady ? "passed" : "pending"}</StatusBadgeV2></div>
                <div className="flex items-center justify-between gap-2"><span className="text-ink-muted">Improvement</span><span className="font-mono font-semibold">{replayEvaluationReady ? `${formatMetric(llm?.before_after_eval?.improvement_pp)} pp` : "pending"}</span></div>
              </div>
            </Panel>
          ) : (
            <Panel title={t(locale, "Reviewed Candidates", "已审核候选")} icon={CheckCircle2} compact>
              <div className="space-y-1.5">
                {candidates.map((candidate) => (
                  <div key={candidate.solution_id} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-2 rounded-md border border-edge px-2.5 py-2 text-xs">
                    <div className="min-w-0"><div className="truncate font-mono font-semibold text-ink">{candidate.solution_id ?? "candidate"}</div><div className="text-2xs text-ink-muted">{candidate.uses_cuda ? "CUDA telemetry" : "HPC CPU"}</div></div>
                    <span className="font-mono font-semibold tabular-nums text-ink-secondary">{formatScore(candidate.cv_score)}</span>
                  </div>
                ))}
                {candidates.length === 0 && <span className="text-xs text-ink-muted">{t(locale, "No reviewed candidates", "暂无已审核候选")}</span>}
              </div>
            </Panel>
          )}

          <Panel title={t(locale, "Report & Deliverables", "报告与交付物")} icon={FileText} compact>
            <div className="space-y-2">
              {!replayDeliverablesReady ? (
                <div className="rounded-md border border-dashed border-edge px-3 py-3 text-xs text-ink-muted">等待独立审核与 Claim Audit 完成后生成交付物</div>
              ) : publicPresentation ? (
                <div className="grid gap-2">
                  {publicArtifactPreviews.map((preview) => (
                    <button
                      key={preview.id}
                      type="button"
                      aria-label={`打开${preview.title}预览`}
                      data-testid={`public-artifact-${preview.id}`}
                      onClick={() => setSelectedPreviewId(preview.id)}
                      className="flex min-w-0 items-center gap-2 rounded-md border border-edge bg-surface px-3 py-2 text-left text-xs text-ink-secondary transition-colors hover:border-accent/45 hover:bg-accent/5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/35"
                    >
                      {preview.kind === "file_list" ? <PackageOpen className="h-4 w-4 shrink-0 text-accent" /> : <FileText className="h-4 w-4 shrink-0 text-accent" />}
                      <span className="min-w-0 flex-1 truncate font-mono font-semibold">{preview.name}</span>
                      <Eye className="h-3.5 w-3.5 shrink-0 text-ink-muted" />
                    </button>
                  ))}
                  {publicArtifactPreviews.length === 0 && (
                    <div className="rounded-md border border-dashed border-edge px-3 py-3 text-xs text-ink-muted">公开交付预览正在加载</div>
                  )}
                </div>
              ) : (
                <>
                  <CopyablePath path={reportPath} className="max-w-full" />
                  <CopyablePath path={isLlmRun ? modelCardPath : submissionPath} className="max-w-full" />
                  {isLlmRun && <CopyablePath path={adapterPath} className="max-w-full" />}
                </>
              )}
              <div className="text-2xs leading-4 text-ink-muted">
                {isLlmRun
                  ? t(locale, "This is a domain adapter for a mature 7B base model, not foundation-model pretraining.", "这是对成熟 7B 基座的领域适配器微调，不是从零预训练基础大模型。")
                  : t(locale, "CV metrics are internal validation, not an official Kaggle score.", "CV 指标是内部验证结果，不是 Kaggle 官方成绩。")}
              </div>
            </div>
          </Panel>

          <Panel title={isLlmRun ? t(locale, "Model Publication Gate", "模型发布 Gate") : t(locale, "Official Kaggle Submission", "Kaggle 正式提交")} accent="red" icon={Cpu} compact>
            <div className="space-y-2">
              <div className="text-2xs leading-4 text-ink-secondary">
                {t(
                  locale,
                  isLlmRun
                    ? (runCompleted ? "The reviewed adapter is complete; publication remains blocked by the Human Gate." : "Training or review is still incomplete. Publication remains blocked and no success claim is shown.")
                    : (runCompleted ? "HPC training completed. Only the irreversible official submission remains blocked by the Human Gate." : "Training or review is still incomplete. Official submission remains blocked."),
                  isLlmRun
                    ? (runCompleted ? "适配器已通过审核；模型发布继续由 Human Gate 阻断。" : "训练或审核尚未完成；模型发布保持阻断，当前不显示成功结论。")
                    : (runCompleted ? "HPC 训练已完成；只有不可逆的 Kaggle 正式提交继续由 Human Gate 阻断。" : "训练或审核尚未完成；Kaggle 正式提交保持阻断。"),
                )}
              </div>
              <button
                type="button"
                disabled
                data-ui-action={isLlmRun ? "blocked_model_publication" : "blocked_kaggle_official_submission"}
                className="flex w-full cursor-not-allowed items-center gap-2 rounded-md border border-danger/30 bg-danger/5 px-3 py-2 text-xs font-medium text-danger-text"
              >
                <Lock className="h-3.5 w-3.5" /> {t(locale, "Blocked: Human approval required", "已阻断：需要人工批准")}
              </button>
            </div>
          </Panel>
        </div>
      </div>

      {publicPresentation && selectedPreview && (
        <div className="fixed inset-0 z-[80] flex items-center justify-center bg-frame/55 p-6 backdrop-blur-[2px]" role="presentation">
          <section
            aria-label={`${selectedPreview.title}预览`}
            aria-modal="true"
            role="dialog"
            className="flex max-h-[82vh] w-full max-w-4xl flex-col overflow-hidden rounded-md border border-edge-strong bg-surface shadow-2xl"
          >
            <header className="flex items-center justify-between gap-4 border-b border-edge px-5 py-3.5">
              <div className="min-w-0">
                <div className="text-sm font-semibold text-ink">{selectedPreview.title}</div>
                <div className="mt-0.5 truncate font-mono text-2xs text-ink-muted">{selectedPreview.name}</div>
              </div>
              <button
                type="button"
                aria-label="关闭产物预览"
                onClick={() => setSelectedPreviewId(null)}
                className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-edge text-ink-muted transition-colors hover:bg-surface-sunken hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/35"
              >
                <X className="h-4 w-4" />
              </button>
            </header>
            <div className="min-h-0 flex-1 overflow-auto p-5">
              {selectedPreview.kind === "markdown" ? (
                <pre className="whitespace-pre-wrap break-words font-sans text-sm leading-7 text-ink-secondary">{selectedPreview.content}</pre>
              ) : (
                <div className="overflow-hidden rounded-md border border-edge">
                  <div className="grid grid-cols-[minmax(0,1fr)_110px_130px] gap-3 border-b border-edge bg-surface-sunken px-4 py-2 text-2xs font-semibold uppercase text-ink-muted">
                    <span>文件</span><span>大小</span><span>SHA256</span>
                  </div>
                  {(selectedPreview.files ?? []).map((file) => (
                    <div key={file.name} className="grid grid-cols-[minmax(0,1fr)_110px_130px] gap-3 border-b border-edge px-4 py-3 text-xs last:border-b-0">
                      <span className="truncate font-mono font-semibold text-ink" title={file.name}>{file.name}</span>
                      <span className="font-mono tabular-nums text-ink-secondary">{formatBytes(file.bytes)}</span>
                      <span className="truncate font-mono text-ink-muted" title={file.sha256}>{file.sha256?.slice(0, 12) ?? "-"}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
            <footer className="flex items-center justify-between gap-3 border-t border-edge bg-surface-sunken px-5 py-2.5 text-2xs text-ink-muted">
              <span>来源：当前已审核运行的公开 allowlist</span>
              {selectedPreview.sha256 && <span className="font-mono">SHA256 {selectedPreview.sha256.slice(0, 16)}...</span>}
            </footer>
          </section>
        </div>
      )}
    </div>
  );
}
