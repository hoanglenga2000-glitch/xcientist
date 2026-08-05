"use client";

import { useEffect, useRef } from "react";
import {
  ChevronUp,
  ChevronRight,
  Database,
  ShieldCheck,
  ArrowRight,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { StatusBadgeV2, StatusDot, type StatusTone } from "../primitives/StatusBadge";
import { GateBadge, ClaimBoundary } from "../primitives/GateBadge";
import { CopyablePath, SectionLabel } from "../primitives/Layout";
import type { WorkstationSummary } from "@/lib/api/types";
import { normalizeTaskId, recordMatchesTask } from "@/lib/task-context";
import { t } from "../localization";

type Locale = "zh-CN" | "en-US";

/* ── Helpers ── */
function normalizeStatus(status: string | undefined): StatusTone {
  const s = String(status ?? "").toLowerCase();
  if (s.includes("complete") || s.includes("passed") || s.includes("verified") || s.includes("ready")) return "verified";
  if (s.includes("running") || s.includes("submitted")) return "running";
  if (s.includes("needs_continuation") || s.includes("paused")) return "pending";
  if (s.includes("blocked") || s.includes("regression")) return "blocked";
  if (s.includes("failed")) return "failed";
  if (s.includes("pending") || s.includes("waiting")) return "pending";
  return "unknown";
}

function runStatusText(locale: Locale | undefined, status: string | undefined): string {
  const s = String(status ?? "").toLowerCase();
  if (s.includes("complete")) return t(locale, "Completed", "已完成");
  if (s.includes("running")) return t(locale, "Running", "运行中");
  if (s.includes("needs_continuation")) return t(locale, "Needs continuation", "待续跑");
  if (s.includes("paused")) return t(locale, "Paused", "已暂停");
  if (s.includes("blocked")) return t(locale, "Blocked", "阻断");
  if (s.includes("failed")) return t(locale, "Failed", "失败");
  if (s.includes("pending")) return t(locale, "Pending", "待确认");
  if (s.includes("passed")) return t(locale, "Passed", "通过");
  return t(locale, "Unknown", "未知");
}

/* ── RunContextBar: compact bar ── */
export function RunContextBar({
  summary,
  locale = "zh-CN",
  selectedTask,
  onExpand,
  compact = false,
}: {
  summary?: WorkstationSummary | null;
  locale?: Locale;
  selectedTask?: string;
  onExpand?: () => void;
  compact?: boolean;
}) {
  const normalizedTask = normalizeTaskId(selectedTask);
  const currentRun = summary?.runtime?.current_run;
  const currentRunMatchesTask = recordMatchesTask(currentRun, normalizedTask);
  const latestRun = summary?.runs?.find((run) => recordMatchesTask(run, normalizedTask));
  const currentTask = summary?.tasks?.find((task) => recordMatchesTask(task, normalizedTask));
  const taskName = currentTask?.name ?? (normalizedTask ? normalizedTask.replaceAll("_", " ") : t(locale, "No active task", "无活跃任务"));
  const runStatus = currentRunMatchesTask ? currentRun?.status ?? latestRun?.status : latestRun?.status;
  const tone = normalizeStatus(runStatus);
  const evidenceCount = summary?.evidence?.filter((item) => recordMatchesTask(item, normalizedTask)).length ?? 0;
  const gateCount = summary?.gates?.filter((g) => recordMatchesTask(g, normalizedTask)).filter((g) => {
    const decision = String(g.decision ?? "").toLowerCase();
    return decision === "pending" || decision === "blocked";
  }).length ?? 0;
  const nextActionSummary = summary?.scientist_next_action;
  const nextActionMatchesTask = !nextActionSummary?.selected_task || normalizeTaskId(nextActionSummary.selected_task) === normalizedTask;
  const nextAction = nextActionMatchesTask ? nextActionSummary?.message ?? nextActionSummary?.selected_action?.title : undefined;

  if (compact) {
    return (
      <div className="flex items-center gap-2 rounded-md border border-edge bg-surface-raised px-2.5 py-1.5 text-xs">
        <StatusDot tone={tone} />
        <span className="font-semibold text-ink truncate max-w-[200px]">{taskName}</span>
        <span className="text-ink-muted">·</span>
        <span className={cn("font-medium", tone === "running" && "text-info-text", tone === "blocked" && "text-danger-text", tone === "verified" && "text-success-text")}>
          {runStatusText(locale, runStatus)}
        </span>
        <span className="text-ink-muted">·</span>
        <span className="text-ink-secondary"><Database className="inline h-3 w-3 mr-0.5" />{evidenceCount}</span>
        {gateCount > 0 && (
          <>
            <span className="text-ink-muted">·</span>
            <span className="text-warning-text"><ShieldCheck className="inline h-3 w-3 mr-0.5" />{gateCount} {t(locale, "gates", "门禁")}</span>
          </>
        )}
        {onExpand && (
          <button onClick={onExpand} className="ml-auto text-ink-muted hover:text-ink" aria-label="Expand evidence rail">
            <ChevronUp className="h-3.5 w-3.5" />
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="flex min-w-0 flex-wrap items-center gap-2 rounded-md border border-edge bg-surface-raised px-3 py-2 text-xs shadow-hairline sm:gap-3">
      <div className="flex min-w-0 flex-[1_1_140px] items-center gap-2">
        <StatusDot tone={tone} />
        <span className="max-w-[280px] truncate font-semibold text-ink">{taskName}</span>
      </div>
      <div className="hidden h-4 w-px bg-edge sm:block" />
      <StatusBadgeV2 tone={tone} size="xs">
        {runStatusText(locale, runStatus)}
      </StatusBadgeV2>
      <div className="hidden h-4 w-px bg-edge sm:block" />
      <span className="flex shrink-0 items-center gap-1 text-ink-secondary">
        <Database className="h-3.5 w-3.5" />
        {evidenceCount} {t(locale, "evidence", "证据")}
      </span>
      {gateCount > 0 && (
        <>
          <div className="hidden h-4 w-px bg-edge sm:block" />
          <span className="flex shrink-0 items-center gap-1 text-warning-text">
            <ShieldCheck className="h-3.5 w-3.5" />
            {gateCount} {t(locale, "gates pending", "门禁待审")}
          </span>
        </>
      )}
      {nextAction && (
        <>
          <div className="hidden h-4 w-px bg-edge lg:block" />
          <span className="hidden min-w-0 flex-[1_1_220px] items-center gap-1 truncate text-accent lg:flex">
            <ArrowRight className="h-3 w-3 shrink-0" />
            {nextAction}
          </span>
        </>
      )}
      {onExpand && (
        <button type="button" onClick={onExpand} className="ml-auto shrink-0 rounded-sm border border-edge p-1 text-ink-muted hover:bg-surface-sunken hover:text-ink" aria-label={t(locale, "Open evidence rail", "打开证据轨")} data-ui-action="open_evidence_overlay" data-ui-skip-action="true">
          <ChevronRight className="h-3.5 w-3.5" />
        </button>
      )}
    </div>
  );
}

/* ── EvidenceRail: expandable right-side panel ── */
export function EvidenceRail({
  summary,
  locale = "zh-CN",
  selectedTask,
  onClose,
  onNavigate,
  embedded = false,
  titleId = "evidence-rail-title",
}: {
  summary?: WorkstationSummary | null;
  locale?: Locale;
  selectedTask?: string;
  onClose: () => void;
  onNavigate?: (page: string) => void;
  embedded?: boolean;
  titleId?: string;
}) {
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const restoreRef = useRef<HTMLElement | null>(null);
  const normalizedTask = normalizeTaskId(selectedTask);
  const currentRun = summary?.runtime?.current_run;
  const currentRunMatchesTask = recordMatchesTask(currentRun, normalizedTask);
  const latestRun = summary?.runs?.find((run) => recordMatchesTask(run, normalizedTask));
  const currentTask = summary?.tasks?.find((task) => recordMatchesTask(task, normalizedTask));
  const runStatus = currentRunMatchesTask ? currentRun?.status ?? latestRun?.status : latestRun?.status;
  const runId = currentRunMatchesTask ? currentRun?.run_id ?? latestRun?.id : latestRun?.id;
  const evidence = summary?.evidence?.filter((item) => recordMatchesTask(item, normalizedTask)) ?? [];
  const gates = summary?.gates?.filter((gate) => recordMatchesTask(gate, normalizedTask)) ?? [];
  const claimBoundary = normalizeTaskId(summary?.terminal_agent?.task_id) === normalizedTask ? summary?.terminal_agent?.claim_boundary : undefined;
  const nextActionSummary = summary?.scientist_next_action;
  const nextAction = !nextActionSummary?.selected_task || normalizeTaskId(nextActionSummary.selected_task) === normalizedTask
    ? nextActionSummary
    : undefined;

  useEffect(() => {
    if (embedded) return;
    restoreRef.current = document.activeElement as HTMLElement | null;
    closeButtonRef.current?.focus();
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      onClose();
    };
    document.addEventListener("keydown", handleKeyDown, true);
    return () => {
      document.removeEventListener("keydown", handleKeyDown, true);
      restoreRef.current?.focus?.();
    };
  }, [embedded, onClose]);

  return (
    <div
      role={embedded ? undefined : "complementary"}
      aria-label={embedded ? undefined : t(locale, "Evidence rail", "证据轨")}
      data-ui-component={embedded ? "evidence-content" : "evidence-rail"}
      className={cn(
        "flex flex-col bg-surface-raised",
        embedded
          ? "max-h-[70vh] w-full"
          : "fixed inset-y-0 right-0 z-evidence-rail w-[360px] max-w-[90vw] animate-slide-in-right border-l border-edge shadow-overlay"
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between border-b border-edge px-4 py-3">
        <div className="flex items-center gap-2">
          <Database className="h-4 w-4 text-accent" />
          <h2 id={titleId} className="text-sm font-bold text-ink">{t(locale, "Evidence Rail", "证据轨")}</h2>
        </div>
        <button ref={closeButtonRef} type="button" onClick={onClose} className="rounded-sm p-1 text-ink-muted hover:bg-surface-sunken hover:text-ink" aria-label={t(locale, "Close evidence rail", "关闭证据轨")}>
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Current context */}
      <div className="border-b border-edge px-4 py-3 space-y-2">
        <SectionLabel>{t(locale, "Current Context", "当前上下文")}</SectionLabel>
        <div className="flex items-center gap-2">
          <StatusDot tone={normalizeStatus(runStatus)} />
          <span className="text-sm font-semibold text-ink truncate">{currentTask?.name ?? "—"}</span>
        </div>
        {runId && (
          <div className="flex items-center gap-2 text-xs text-ink-secondary">
            <span>{t(locale, "Run", "运行")}:</span>
            <CopyablePath path={runId} />
          </div>
        )}
        {claimBoundary && <ClaimBoundary boundary={claimBoundary} />}
      </div>

      {/* Next safe action */}
      {nextAction?.selected_action && (
        <div className="border-b border-edge px-4 py-3 space-y-1">
          <SectionLabel>{t(locale, "Next Safe Action", "下一步安全动作")}</SectionLabel>
          <div className="rounded-md border border-accent/20 bg-accent-light p-2.5">
            <div className="text-xs font-semibold text-accent-dark">{nextAction.selected_action.title}</div>
            {nextAction.selected_action.why && <div className="mt-1 text-2xs text-ink-secondary">{nextAction.selected_action.why}</div>}
            {nextAction.selected_action.gate && (
              <div className="mt-1.5">
                <GateBadge status={nextAction.selected_action.gate.toLowerCase().includes("human") ? "pending" : "approved"} />
              </div>
            )}
          </div>
        </div>
      )}

      {/* Evidence summary */}
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
        <SectionLabel>{t(locale, "Evidence", "证据")} ({evidence.length})</SectionLabel>
        {evidence.length === 0 ? (
          <div className="text-xs text-ink-muted py-4 text-center">{t(locale, "No evidence available", "暂无证据")}</div>
        ) : (
          <div className="space-y-1.5">
            {evidence.slice(0, 10).map((item, idx) => {
              const record = item as Record<string, unknown>;
              return (
                <div key={idx} className="flex items-center justify-between gap-2 rounded-sm border border-edge px-2 py-1.5 text-xs">
                  <span className="truncate text-ink-secondary">{String(record.name ?? record.artifact_type ?? `Evidence ${idx + 1}`)}</span>
                  <StatusBadgeV2 tone={normalizeStatus(String(record.verification_status ?? record.status))} size="xs">
                    {String(record.verification_status ?? record.status ?? "unknown")}
                  </StatusBadgeV2>
                </div>
              );
            })}
            {evidence.length > 10 && (
              <button onClick={() => onNavigate?.("evidence")} className="text-xs text-accent font-semibold hover:underline">
                {t(locale, `View all ${evidence.length} evidence`, `查看全部 ${evidence.length} 证据`)}
              </button>
            )}
          </div>
        )}

        {/* Gates summary */}
        {gates.length > 0 && (
          <>
            <SectionLabel>{t(locale, "Integrity Gates", "完整性 Gate")} ({gates.length})</SectionLabel>
            <div className="space-y-1.5">
              {gates.slice(0, 6).map((gate, idx) => {
                const g = gate as Record<string, unknown>;
                const decision = String(g.decision ?? "pending").toLowerCase();
                return (
                  <div key={idx} className="flex items-center justify-between gap-2 rounded-sm border border-edge px-2 py-1.5 text-xs">
                    <span className="truncate text-ink-secondary">{String(g.gate_type ?? `Gate ${idx + 1}`)}</span>
                    <GateBadge status={decision === "approved" ? "approved" : decision === "rejected" ? "rejected" : "pending"} />
                  </div>
                );
              })}
              {gates.length > 6 && (
                <button onClick={() => onNavigate?.("gates")} className="text-xs text-accent font-semibold hover:underline">
                  {t(locale, `View all gates`, `查看全部门禁`)}
                </button>
              )}
            </div>
          </>
        )}
      </div>

      {/* Footer: quick nav */}
      <div className="border-t border-edge px-4 py-2.5 flex gap-2">
        <button onClick={() => onNavigate?.("evidence")} className="flex-1 rounded-md border border-edge px-2 py-1.5 text-xs font-medium text-ink-secondary hover:bg-surface-sunken hover:text-ink text-center">
          <Database className="inline h-3 w-3 mr-1" />{t(locale, "Evidence Ledger", "证据台账")}
        </button>
        <button onClick={() => onNavigate?.("gates")} className="flex-1 rounded-md border border-edge px-2 py-1.5 text-xs font-medium text-ink-secondary hover:bg-surface-sunken hover:text-ink text-center">
          <ShieldCheck className="inline h-3 w-3 mr-1" />{t(locale, "Integrity Gates", "完整性 Gate")}
        </button>
      </div>
    </div>
  );
}

/* ── MobileEvidenceDrawer: bottom sheet ── */
export function MobileEvidenceDrawer({
  summary,
  locale = "zh-CN",
  selectedTask,
  onClose,
  onNavigate,
}: {
  summary?: WorkstationSummary | null;
  locale?: Locale;
  selectedTask?: string;
  onClose: () => void;
  onNavigate?: (page: string) => void;
}) {
  const panelRef = useRef<HTMLDivElement>(null);
  const restoreRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    restoreRef.current = document.activeElement as HTMLElement | null;
    document.body.setAttribute("data-evidence-lock", "true");
    const panel = panelRef.current;
    const focusables = () => Array.from(panel?.querySelectorAll<HTMLElement>("button, a, input, select, textarea, [tabindex]:not([tabindex='-1'])") ?? []).filter((element) => !element.hasAttribute("disabled"));
    focusables()[0]?.focus();
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== "Tab") return;
      const items = focusables();
      if (items.length === 0) return;
      const first = items[0];
      const last = items[items.length - 1];
      const active = document.activeElement as HTMLElement | null;
      if (event.shiftKey && (active === first || !panel?.contains(active))) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && (active === last || !panel?.contains(active))) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", handleKeyDown, true);
    return () => {
      document.removeEventListener("keydown", handleKeyDown, true);
      document.body.removeAttribute("data-evidence-lock");
      restoreRef.current?.focus?.();
    };
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-drawer lg:hidden">
      <div className="mobile-evidence-backdrop absolute inset-0 bg-ink/40" onClick={onClose} aria-hidden="true" />
      <div ref={panelRef} role="dialog" aria-modal="true" aria-labelledby="mobile-evidence-title" className="mobile-evidence-drawer absolute inset-x-0 bottom-0 max-h-[70vh] overflow-hidden rounded-t-lg border-t border-edge bg-surface-raised shadow-overlay animate-slide-in-bottom">
        {/* Drag handle */}
        <div className="flex justify-center pt-2 pb-1">
          <div className="h-1 w-8 rounded-full bg-edge-strong" />
        </div>
        <EvidenceRail summary={summary} locale={locale} selectedTask={selectedTask} onClose={onClose} onNavigate={onNavigate} embedded titleId="mobile-evidence-title" />
      </div>
    </div>
  );
}
