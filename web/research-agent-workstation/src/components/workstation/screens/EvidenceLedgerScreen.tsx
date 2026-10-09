"use client";

import { useMemo, useState } from "react";
import {
  Ban,
  CalendarDays,
  Columns3,
  Database,
  FileJson,
  FileSearch,
  Filter,
  GitBranch,
  Network,
  ShieldCheck,
} from "lucide-react";
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

function statusToneOf(r: Record<string, unknown>): StatusTone {
  const status = String(r.verification_status ?? r.status ?? r.decision ?? "unknown").toLowerCase();
  if (status.includes("verified") || status.includes("approved") || status.includes("complete")) return "verified";
  if (status.includes("pending")) return "pending";
  if (status.includes("failed") || status.includes("blocked") || status.includes("reject")) return "blocked";
  return "unknown";
}

function evidenceStatus(r: Record<string, unknown>, locale: Locale): string {
  return String(r.verification_status ?? r.status ?? r.decision ?? t(locale, "Unknown", "未知"));
}

function evidenceName(r: Record<string, unknown>, index: number): string {
  return String(r.name ?? r.artifact_type ?? r.message ?? r.type ?? r.id ?? `Evidence #${index + 1}`);
}

function recordDate(r: Record<string, unknown>): string | null {
  for (const key of ["timestamp", "created_at", "at", "date", "ts"]) {
    const v = r[key];
    if (typeof v === "string" && v && !Number.isNaN(Date.parse(v))) return v.slice(0, 10);
  }
  return null;
}

function toCsv(rows: Record<string, unknown>[]): string {
  const keys = Array.from(new Set(rows.flatMap((r) => Object.keys(r))));
  const esc = (v: unknown) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  return [keys.map(esc).join(","), ...rows.map((r) => keys.map((k) => esc(r[k])).join(","))].join("\n");
}

function downloadTextFile(name: string, content: string, mime: string): void {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function EvidenceLedgerScreen(props: ScreenProps) {
  const { summary, locale = "zh-CN" } = props;
  const evidence = useMemo(() => summary?.evidence ?? [], [summary?.evidence]);

  const [draftText, setDraftText] = useState("");
  const [draftStatus, setDraftStatus] = useState("all");
  const [draftFrom, setDraftFrom] = useState("");
  const [draftTo, setDraftTo] = useState("");
  const [applied, setApplied] = useState({ text: "", status: "all", from: "", to: "" });
  const [showColumnConfig, setShowColumnConfig] = useState(false);
  const [showTypeColumn, setShowTypeColumn] = useState(true);
  const [showLineageColumn, setShowLineageColumn] = useState(true);
  const [showLineageGraph, setShowLineageGraph] = useState(false);

  const recordEvidenceAction = (action: string, metadata: Record<string, unknown> = {}) => {
    void props.runWorkstationAction?.(action, {
      task_id: props.selectedTask,
      source: "evidence_ledger",
      ...metadata,
    });
  };

  const filtered = useMemo(() => {
    return evidence.filter((e) => {
      const r = e as Record<string, unknown>;
      if (applied.text) {
        const haystack = Object.values(r).map((v) => String(v).toLowerCase()).join(" ");
        if (!haystack.includes(applied.text.toLowerCase())) return false;
      }
      if (applied.status !== "all" && statusToneOf(r) !== applied.status) return false;
      if (applied.from || applied.to) {
        const d = recordDate(r);
        if (d) {
          if (applied.from && d < applied.from) return false;
          if (applied.to && d > applied.to) return false;
        }
      }
      return true;
    });
  }, [evidence, applied]);

  const verifiedCount = evidence.filter((e) => statusToneOf(e as Record<string, unknown>) === "verified").length;

  const applyFilters = () => {
    setApplied({ text: draftText, status: draftStatus, from: draftFrom, to: draftTo });
    recordEvidenceAction("apply_evidence_filters", {
      text: draftText, status: draftStatus, from: draftFrom, to: draftTo,
    });
  };

  const resetFilters = () => {
    setDraftText("");
    setDraftStatus("all");
    setDraftFrom("");
    setDraftTo("");
    setApplied({ text: "", status: "all", from: "", to: "" });
    recordEvidenceAction("reset_evidence_filters");
  };

  const exportCsv = () => {
    const rows = filtered.map((e) => e as Record<string, unknown>);
    downloadTextFile(
      `evidence_ledger_${props.selectedTask || "all"}.csv`,
      toCsv(rows),
      "text/csv;charset=utf-8",
    );
    recordEvidenceAction("export_evidence_csv", { rows: rows.length });
  };

  const exportDraftBundle = () => {
    const bundle = {
      schema: "research_os.evidence_draft_bundle.v1",
      task_id: props.selectedTask || null,
      exported_at: new Date().toISOString(),
      claim_boundary: summary?.terminal_agent?.claim_boundary ?? null,
      record_count: filtered.length,
      records: filtered,
      note: "Draft bundle for human review. Final evidence approval remains a Human Gate and is not performed by this UI.",
    };
    downloadTextFile(
      `evidence_draft_bundle_${props.selectedTask || "all"}.json`,
      JSON.stringify(bundle, null, 2),
      "application/json;charset=utf-8",
    );
    recordEvidenceAction("export_draft_evidence_bundle", { rows: filtered.length });
  };

  return (
    <div className="space-y-4">
      <PageHeader
        title={t(locale, "Evidence Ledger", "证据账本")}
        subtitle={t(locale, "Audit trail, claim boundaries, and lineage", "审计轨迹、声明边界与血缘")}
        breadcrumb={`${t(locale, "Governance", "治理")} > ${t(locale, "Evidence Ledger", "证据账本")}`}
      />

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        <MetricTile label={t(locale, "Total Evidence", "总证据")} value={evidence.length} icon={Database} tone="blue" />
        <MetricTile label={t(locale, "Verified", "已验证")} value={verifiedCount} icon={ShieldCheck} tone="green" />
        <MetricTile label={t(locale, "Pending", "待验证")} value={evidence.length - verifiedCount} icon={FileSearch} tone="amber" />
      </div>

      <Panel title={t(locale, "Claim Boundary", "声明边界")} compact>
        <div className="break-words text-xs leading-5 text-ink-secondary [overflow-wrap:anywhere]">
          {summary?.terminal_agent?.claim_boundary
            ? String(summary.terminal_agent.claim_boundary)
            : t(locale, "No claim boundary recorded", "无声明边界记录")}
        </div>
      </Panel>

      <Panel
        title={t(locale, "Evidence Records", "证据记录")}
        action={
          <div className="flex flex-wrap items-center gap-1.5">
            <button
              type="button"
              data-ui-action="open_evidence_lineage_graph"
              data-ui-skip-action="true"
              onClick={() => setShowLineageGraph(true)}
              className="flex items-center gap-1 rounded border border-edge px-2 py-1 text-2xs font-medium text-ink-secondary hover:bg-surface-sunken"
            >
              <Network className="h-3 w-3" /> {t(locale, "Lineage Graph", "血缘图")}
            </button>
            <button
              type="button"
              data-ui-action="configure_evidence_columns"
              data-ui-skip-action="true"
              onClick={() => {
                setShowColumnConfig((v) => !v);
                recordEvidenceAction("configure_evidence_columns", { open: !showColumnConfig });
              }}
              className="flex items-center gap-1 rounded border border-edge px-2 py-1 text-2xs font-medium text-ink-secondary hover:bg-surface-sunken"
            >
              <Columns3 className="h-3 w-3" /> {t(locale, "Columns", "列")}
            </button>
            <button
              type="button"
              data-ui-action="export_evidence_csv"
              data-ui-skip-action="true"
              onClick={exportCsv}
              disabled={filtered.length === 0}
              className="flex items-center gap-1 rounded border border-edge px-2 py-1 text-2xs font-medium text-ink-secondary hover:bg-surface-sunken disabled:opacity-40"
            >
              <FileSearch className="h-3 w-3" /> CSV
            </button>
            <button
              type="button"
              data-ui-action="export_draft_evidence_bundle"
              data-ui-skip-action="true"
              onClick={exportDraftBundle}
              disabled={filtered.length === 0}
              className="flex items-center gap-1 rounded border border-edge px-2 py-1 text-2xs font-medium text-ink-secondary hover:bg-surface-sunken disabled:opacity-40"
            >
              <FileJson className="h-3 w-3" /> {t(locale, "Draft Bundle", "草稿包")}
            </button>
          </div>
        }
      >
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-1.5 rounded-sm border border-edge bg-surface-sunken px-2 py-1.5">
            <Filter className="h-3 w-3 text-ink-muted" />
            <input
              type="text"
              data-ui-action="evidence_filter_text"
              value={draftText}
              onChange={(e) => setDraftText(e.target.value)}
              placeholder={t(locale, "Filter text...", "过滤文本...")}
              className="h-6 min-w-[8rem] flex-1 rounded border border-edge bg-surface px-2 text-xs text-ink outline-none focus:border-accent"
            />
            <select
              data-ui-action="evidence_filter_status"
              value={draftStatus}
              onChange={(e) => setDraftStatus(e.target.value)}
              className="h-6 rounded border border-edge bg-surface px-1.5 text-xs text-ink outline-none focus:border-accent"
            >
              <option value="all">{t(locale, "All status", "全部状态")}</option>
              <option value="verified">{t(locale, "Verified", "已验证")}</option>
              <option value="pending">{t(locale, "Pending", "待验证")}</option>
              <option value="blocked">{t(locale, "Blocked", "受阻")}</option>
              <option value="unknown">{t(locale, "Unknown", "未知")}</option>
            </select>
            <span className="flex items-center gap-1 text-2xs text-ink-muted">
              <CalendarDays className="h-3 w-3" />
              <input
                type="date"
                data-ui-action="evidence_date_range"
                value={draftFrom}
                onChange={(e) => setDraftFrom(e.target.value)}
                className="h-6 rounded border border-edge bg-surface px-1 text-2xs text-ink outline-none focus:border-accent"
              />
              <span>→</span>
              <input
                type="date"
                data-ui-action="evidence_date_range"
                value={draftTo}
                onChange={(e) => setDraftTo(e.target.value)}
                className="h-6 rounded border border-edge bg-surface px-1 text-2xs text-ink outline-none focus:border-accent"
              />
            </span>
            <button
              type="button"
              data-ui-action="apply_evidence_filters"
              data-ui-skip-action="true"
              onClick={applyFilters}
              className="rounded bg-accent px-2 py-1 text-2xs font-medium text-accent-fg hover:opacity-90"
            >
              {t(locale, "Apply", "应用")}
            </button>
            <button
              type="button"
              data-ui-action="reset_evidence_filters"
              data-ui-skip-action="true"
              onClick={resetFilters}
              className="rounded border border-edge px-2 py-1 text-2xs font-medium text-ink-secondary hover:bg-surface"
            >
              {t(locale, "Reset", "重置")}
            </button>
          </div>

          {showColumnConfig && (
            <div className="flex flex-wrap items-center gap-3 rounded-sm border border-edge px-2 py-1.5 text-2xs text-ink-secondary">
              <label className="flex items-center gap-1">
                <input type="checkbox" checked={showTypeColumn} onChange={(e) => setShowTypeColumn(e.target.checked)} />
                {t(locale, "Type column", "类型列")}
              </label>
              <label className="flex items-center gap-1">
                <input type="checkbox" checked={showLineageColumn} onChange={(e) => setShowLineageColumn(e.target.checked)} />
                {t(locale, "Lineage column", "血缘列")}
              </label>
            </div>
          )}

          {filtered.length === 0 ? (
            <div className="py-4 text-center text-sm text-ink-muted">
              {evidence.length === 0 ? t(locale, "No evidence recorded", "无证据记录") : t(locale, "No matches", "无匹配")}
            </div>
          ) : (
            <div className="space-y-1.5">
              {filtered.slice(0, 30).map((item, idx) => {
                const r = item as Record<string, unknown>;
                const tone = statusToneOf(r);
                const lineageId = String(r.id ?? r.artifact_id ?? r.sha256 ?? idx);
                const artifactType = r.artifact_type ?? r.type;
                const sha256 = typeof r.sha256 === "string" ? r.sha256 : null;
                return (
                  <div key={idx} className="flex items-center justify-between gap-2 rounded-sm border border-edge px-2.5 py-1.5 text-xs">
                    <div className="flex min-w-0 items-center gap-2">
                      <StatusDot tone={tone} />
                      <span className="truncate text-ink-secondary">{evidenceName(r, idx)}</span>
                      {showTypeColumn && artifactType != null && (
                        <span className="shrink-0 rounded bg-surface-sunken px-1 py-0.5 text-2xs text-ink-muted">{String(artifactType)}</span>
                      )}
                      {sha256 && <code className="hidden shrink-0 text-2xs text-ink-muted xl:inline">{sha256.slice(0, 10)}...</code>}
                    </div>
                    <div className="flex shrink-0 items-center gap-2">
                      {showLineageColumn && (
                        <button
                          type="button"
                          data-ui-action={`open_evidence_lineage_${lineageId}`}
                          data-ui-skip-action="true"
                          title={t(locale, "Open lineage", "打开血缘")}
                          onClick={() => setShowLineageGraph(true)}
                          className="text-ink-muted hover:text-accent"
                        >
                          <GitBranch className="h-3 w-3" />
                        </button>
                      )}
                      <StatusBadgeV2 tone={tone} size="xs">{evidenceStatus(r, locale)}</StatusBadgeV2>
                    </div>
                  </div>
                );
              })}
              {filtered.length > 30 && (
                <div className="py-1 text-center text-2xs text-ink-muted">
                  {t(locale, `Showing 30 of ${filtered.length}`, `显示 ${filtered.length} 条中的 30 条`)}
                </div>
              )}
            </div>
          )}
        </div>
      </Panel>

      {showLineageGraph && (
        <div
          className="fixed inset-0 z-[70] flex items-center justify-center bg-ink/45 p-4"
          role="dialog"
          aria-modal="true"
          aria-labelledby="evidence-lineage-title"
          onClick={(event) => { if (event.target === event.currentTarget) setShowLineageGraph(false); }}
        >
          <section className="flex max-h-[min(720px,90vh)] w-full max-w-5xl flex-col overflow-hidden rounded-md border border-edge bg-surface-raised shadow-overlay">
            <header className="flex items-center justify-between border-b border-edge px-4 py-3">
              <div>
                <h2 id="evidence-lineage-title" className="text-sm font-bold text-ink">{t(locale, "Evidence Lineage Graph", "证据血缘图")}</h2>
                <p className="mt-1 text-xs text-ink-muted">{t(locale, "Task to artifact relationships for the selected evidence set", "当前证据集的任务到产物关系")}</p>
              </div>
              <button type="button" aria-label={t(locale, "Close lineage graph", "关闭血缘图")} onClick={() => setShowLineageGraph(false)} className="rounded-sm p-2 text-ink-muted hover:bg-surface-sunken hover:text-ink">
                <Ban className="h-4 w-4" />
              </button>
            </header>
            <div className="min-h-0 overflow-auto p-4">
              {filtered.length === 0 ? (
                <div className="rounded-md border border-dashed border-edge px-4 py-10 text-center text-sm text-ink-muted">{t(locale, "No evidence matches the current filters.", "当前筛选条件没有匹配证据。")}</div>
              ) : (
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  {filtered.slice(0, 60).map((item, index) => {
                    const record = item as Record<string, unknown>;
                    const artifact = evidenceName(record, index);
                    const task = String(record.task_id ?? props.selectedTask ?? "—");
                    const run = String(record.run_id ?? record.experiment_id ?? "—");
                    return (
                      <article key={`${artifact}-${index}`} className="rounded-md border border-edge bg-surface-sunken p-3">
                        <div className="break-words text-xs font-semibold text-ink">{artifact}</div>
                        <div className="mt-2 space-y-1 text-2xs text-ink-secondary">
                          <div><span className="text-ink-muted">Task:</span> {task}</div>
                          <div><span className="text-ink-muted">Run:</span> {run}</div>
                          <div><span className="text-ink-muted">Status:</span> {evidenceStatus(record, locale)}</div>
                        </div>
                      </article>
                    );
                  })}
                </div>
              )}
            </div>
          </section>
        </div>
      )}

      <Panel
        title={t(locale, "Final Evidence Approval (Human Gate)", "最终证据批准(人工闸门)")}
        accent="red"
        compact
      >
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="text-2xs text-ink-secondary">
            {t(
              locale,
              "Final approval of evidence for claims is a human decision. This UI never auto-approves; use the draft bundle export for review.",
              "证据最终批准属于人工决策。本界面不会自动批准;请导出草稿包供人工审阅。",
            )}
          </div>
          <button
            type="button"
            data-ui-action="blocked_final_evidence_approval"
            className="flex items-center gap-1 rounded border border-danger/30 bg-danger/5 px-2 py-1 text-2xs font-medium text-danger-text"
          >
            <Ban className="h-3 w-3" /> {t(locale, "Blocked: Human Gate", "受阻:人工闸门")}
          </button>
        </div>
      </Panel>
    </div>
  );
}
