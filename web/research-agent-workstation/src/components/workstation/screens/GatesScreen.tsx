"use client";

import {
  ArrowRight,
  CheckCircle2,
  Code2,
  Cpu,
  FileText,
  Send,
  ShieldCheck,
  XCircle,
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

const GATE_PIPELINE = [
  { id: "code", label: "Code", labelZh: "代码", icon: Code2 },
  { id: "compute", label: "Compute", labelZh: "计算", icon: Cpu },
  { id: "submission", label: "Submission", labelZh: "提交", icon: Send },
  { id: "evidence", label: "Evidence", labelZh: "证据", icon: FileText },
  { id: "report", label: "Report", labelZh: "报告", icon: ShieldCheck },
] as const;

export type GateDecisionState = "approved" | "rejected" | "pending" | "unknown";

export type GateDecisionView = {
  canonical: GateDecisionState;
  raw: string;
  display: string;
  tone: StatusTone;
  isPending: boolean;
};

const GATE_DECISION_ALIASES: Record<string, GateDecisionState> = {
  approved: "approved",
  rejected: "rejected",
  pending: "pending",
  promote: "approved",
  promoted: "approved",
  hold: "rejected",
  held: "rejected",
  passed: "approved",
  failed: "rejected",
  blocked: "pending",
  waiting: "pending",
};

export function normalizeGateDecision(value: unknown): GateDecisionView {
  const raw = typeof value === "string" ? value.trim().toLowerCase() : "";
  const canonical = GATE_DECISION_ALIASES[raw] ?? "unknown";
  const normalizedRaw = raw || "unknown";
  const display = canonical === "unknown" || canonical === normalizedRaw
    ? normalizedRaw
    : `${canonical} / ${normalizedRaw}`;
  const tone: StatusTone = canonical === "approved"
    ? "verified"
    : canonical === "rejected"
      ? "failed"
      : canonical === "pending"
        ? "pending"
        : "unknown";
  return { canonical, raw: normalizedRaw, display, tone, isPending: canonical === "pending" };
}

export function gateBelongsToSelectedTask(gate: unknown, selectedTask: string): boolean {
  if (!gate || typeof gate !== "object" || Array.isArray(gate)) return false;
  const taskId = selectedTask.trim();
  if (!taskId) return false;
  const record = gate as Record<string, unknown>;
  const snakeCaseId = typeof record.task_id === "string" ? record.task_id.trim() : "";
  const camelCaseId = typeof record.taskId === "string" ? record.taskId.trim() : "";
  return (snakeCaseId || camelCaseId) === taskId;
}

export function GatesScreen(props: ScreenProps) {
  const { summary, locale = "zh-CN", runWorkstationAction } = props;
  const gates = (summary?.gates ?? []).filter((gate) => gateBelongsToSelectedTask(gate, props.selectedTask));

  const recordGateAction = (action: string, metadata: Record<string, unknown> = {}) => {
    void runWorkstationAction?.(action, {
      task_id: props.selectedTask,
      source: "gates_screen",
      ...metadata,
    });
  };
  const pendingGates = gates.filter((gate) => normalizeGateDecision(gate.decision).canonical === "pending");
  const approvedGates = gates.filter((gate) => normalizeGateDecision(gate.decision).canonical === "approved");

  return (
    <div className="space-y-4">
      <PageHeader
        title={t(locale, "Integrity Gates", "完整性闸门")}
        subtitle={t(locale, "Gate pipeline and approval status", "闸门流水线与审批状态")}
        breadcrumb={`${t(locale, "Governance", "治理")} > ${t(locale, "Integrity Gates", "完整性闸门")}`}
      />

      {/* KPIs */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        <MetricTile label={t(locale, "Total Gates", "总闸门")} value={gates.length} icon={ShieldCheck} tone="blue" />
        <MetricTile label={t(locale, "Pending", "待审")} value={pendingGates.length} icon={ShieldCheck} tone={pendingGates.length > 0 ? "amber" : "green"} />
        <MetricTile label={t(locale, "Approved", "已通过")} value={approvedGates.length} icon={CheckCircle2} tone="green" />
      </div>

      {/* Gate pipeline visualization */}
      <Panel title={t(locale, "Gate Pipeline", "闸门流水线")} compact>
        <div className="flex items-center gap-0 overflow-x-auto pb-1">
          {GATE_PIPELINE.map((stage, idx) => {
            const Icon = stage.icon;
            return (
              <div key={stage.id} className="flex items-center">
                <div className="flex flex-col items-center gap-1 rounded-md px-3 py-2 min-w-[60px] border border-edge">
                  <Icon className="h-4 w-4 text-ink-secondary" />
                  <span className="text-2xs font-semibold text-ink-secondary">
                    {locale === "zh-CN" ? stage.labelZh : stage.label}
                  </span>
                </div>
                {idx < GATE_PIPELINE.length - 1 && (
                  <ArrowRight className="h-3 w-3 shrink-0 text-ink-faint mx-0.5" />
                )}
              </div>
            );
          })}
        </div>
      </Panel>

      {/* Gate table */}
      <Panel title={t(locale, "Gate Decisions", "闸门决议")}>
        {gates.length === 0 ? (
          <div className="py-4 text-center text-sm text-ink-muted">{t(locale, "No gates recorded", "无闸门记录")}</div>
        ) : (
          <div className="space-y-1.5">
            {gates.map((gate, idx) => {
              const g = gate as Record<string, unknown>;
              const decision = normalizeGateDecision(g.decision);
              const gateId = String(g.gate_id ?? g.id ?? `${props.selectedTask}:${idx}`);
              return (
                <div
                  key={gateId}
                  className={cn(
                    "flex items-center justify-between gap-2 rounded-sm border px-2.5 py-1.5 text-xs",
                    decision.isPending ? "border-warning/30 bg-warning-light/40" : "border-edge"
                  )}
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <StatusDot tone={decision.tone} />
                    <span className="truncate font-medium text-ink">{String(g.name ?? g.gate_type ?? g.gate_id ?? g.id ?? `Gate #${idx + 1}`)}</span>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <span title={decision.raw === decision.canonical ? undefined : `decision=${decision.canonical}; source=${decision.raw}`}>
                      <StatusBadgeV2 tone={decision.tone} size="xs">{decision.display}</StatusBadgeV2>
                    </span>
                    {decision.isPending && (
                      <div className="flex gap-1">
                        <button
                          type="button"
                          data-ui-action="approve_integrity_gate"
                          data-ui-skip-action="true"
                          onClick={() => recordGateAction("approve_integrity_gate", { gate_id: gateId })}
                          className="rounded p-0.5 text-success-text hover:bg-success-light"
                          title={t(locale, "Approve", "通过")}
                        >
                          <CheckCircle2 className="h-3.5 w-3.5" />
                        </button>
                        <button
                          type="button"
                          data-ui-action="request_gate_revision"
                          data-ui-skip-action="true"
                          onClick={() => recordGateAction("request_gate_revision", { gate_id: gateId })}
                          className="rounded p-0.5 text-warning-text hover:bg-warning-light"
                          title={t(locale, "Request Revision", "请求修订")}
                        >
                          <ShieldCheck className="h-3.5 w-3.5" />
                        </button>
                        <button
                          type="button"
                          data-ui-action="reject_integrity_gate"
                          data-ui-skip-action="true"
                          onClick={() => recordGateAction("reject_integrity_gate", { gate_id: gateId })}
                          className="rounded p-0.5 text-danger-text hover:bg-danger-light"
                          title={t(locale, "Reject", "拒绝")}
                        >
                          <XCircle className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </Panel>

      {/* Official submission — Human Gate */}
      <Panel title={t(locale, "Official Submission (Human Gate)", "官方提交(人工闸门)")} accent="red" compact>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="text-2xs text-ink-secondary">
            {t(
              locale,
              "Official Kaggle submission is a human decision. This UI never auto-submits to Kaggle.",
              "官方 Kaggle 提交属于人工决策。本界面不会自动提交到 Kaggle。",
            )}
          </div>
          <button
            type="button"
            data-ui-action="blocked_allow_official_submit"
            className="flex items-center gap-1 rounded border border-danger/30 bg-danger/5 px-2 py-1 text-2xs font-medium text-danger-text"
          >
            <Send className="h-3 w-3" /> {t(locale, "Blocked: Human Gate", "受阻:人工闸门")}
          </button>
        </div>
      </Panel>
    </div>
  );
}
