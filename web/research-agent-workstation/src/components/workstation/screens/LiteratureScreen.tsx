"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  BookOpen,
  CheckCircle2,
  Database,
  Download,
  ExternalLink,
  FileCheck2,
  Filter,
  Link2,
  LoaderCircle,
  RefreshCw,
  Search,
  Send,
  ShieldCheck,
  Upload,
} from "lucide-react";
import type {
  LiteraturePaper,
  LiteratureSearchResponse,
  LiteratureTaskState,
  WorkstationActionResponse,
  WorkstationSummary,
} from "@/lib/api/types";
import * as api from "@/lib/api/client";
import { PageHeader, Panel, MetricTile, AiLabel } from "../primitives/Layout";
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
    action: string;
    taskId?: string;
    request?: Record<string, unknown>;
    response?: Record<string, unknown>;
    message: string;
    artifact?: string | null;
    at: string;
  } | null;
  locale?: Locale;
  setLocale?: (locale: Locale) => void;
};

type ActionState = {
  status: "idle" | "running" | "passed" | "failed";
  message: string;
  artifact?: string;
};

const actionLabels: Record<string, { zh: string; en: string }> = {
  rag_build_agent_context: { zh: "构建 Agent 上下文", en: "Build Agent Context" },
  rag_send_research_agent: { zh: "发送研究 Agent", en: "Send to Research Agent" },
  rag_send_code_agent: { zh: "发送代码 Agent", en: "Send to Code Agent" },
  rag_bind_report_claim: { zh: "绑定报告声明", en: "Bind Report Claim" },
  rag_request_citation_audit: { zh: "独立引文审计", en: "Independent Citation Audit" },
};

function sourceHref(paper: LiteraturePaper) {
  if (paper.url) return paper.url;
  if (paper.source_url) return paper.source_url;
  if (paper.doi) return `https://doi.org/${paper.doi.replace(/^https?:\/\/doi.org\//i, "")}`;
  return null;
}

function auditTone(status: string) {
  const normalized = status.toLowerCase();
  if (normalized === "passed" || normalized.includes("supported") || normalized.includes("verified")) return "verified" as const;
  if (normalized === "blocked" || normalized === "needs_evidence" || normalized.includes("flagged") || normalized.includes("overclaim")) return "blocked" as const;
  return "pending" as const;
}

export function LiteratureScreen(props: ScreenProps) {
  const { summary, locale = "zh-CN", runWorkstationAction, refreshSummary, selectedTask } = props;
  const [liveByTask, setLiveByTask] = useState<Record<string, LiteratureSearchResponse>>({});
  const [searchQuery, setSearchQuery] = useState("");
  const [searchStatus, setSearchStatus] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [isImporting, setIsImporting] = useState(false);
  const [selectedPaperId, setSelectedPaperId] = useState("");
  const [claim, setClaim] = useState("");
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [actionStates, setActionStates] = useState<Record<string, ActionState>>({});
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const storedLiterature = summary?.literature_by_task?.[selectedTask]
    ?? (summary?.literature_context?.task_id === selectedTask ? summary.literature_context as LiteratureTaskState : null);
  const literature = (liveByTask[selectedTask] ?? storedLiterature ?? null) as LiteratureTaskState | null;
  const papers = useMemo(() => literature?.papers ?? [], [literature]);
  const latestCitationAudit = literature?.latest_citation_audit ?? literature?.citation_audits?.at(-1) ?? null;
  const legacyClaimAudit = literature?.claim_audit ?? [];
  const noFakePapers = papers.length > 0 && (literature?.integrity?.fabricated ?? 0) === 0;
  const selectedPaper = papers.find((paper) => paper.id === selectedPaperId) ?? null;
  const hasManifest = Boolean(literature?.manifest_path && literature?.context_path && papers.length);

  useEffect(() => {
    setSearchQuery("");
    setSelectedPaperId("");
    setSearchStatus("");
    setActionStates({});
  }, [selectedTask]);

  useEffect(() => {
    if (!selectedPaperId && papers[0]?.id) setSelectedPaperId(papers[0].id);
  }, [papers, selectedPaperId]);

  const setActionState = (action: string, state: ActionState) => {
    setActionStates((current) => ({ ...current, [action]: state }));
  };

  const actionMetadata = (extra: Record<string, unknown> = {}) => ({
    task_id: selectedTask,
    source: "literature_screen",
    context_path: literature?.context_path ?? null,
    manifest_path: literature?.manifest_path ?? null,
    ...extra,
  });

  const recordReceipt = async (action: string, metadata: Record<string, unknown> = {}) => {
    if (!runWorkstationAction) return;
    await runWorkstationAction(action, actionMetadata(metadata));
  };

  const runSearch = async (queryInput?: string) => {
    const trimmed = (queryInput ?? searchQuery).trim();
    if (!trimmed || isSearching) return;
    setIsSearching(true);
    setSearchStatus(t(locale, "Searching arXiv, OpenAlex and Crossref...", "正在检索 arXiv、OpenAlex 与 Crossref..."));
    try {
      const result = await api.searchLiterature({
        task_id: selectedTask,
        query: trimmed,
        max_results: 18,
        include_arxiv: true,
        include_openalex: true,
        include_crossref: true,
        include_internal: false,
      });
      setLiveByTask((current) => ({ ...current, [selectedTask]: result }));
      setSearchQuery(trimmed);
      setSelectedPaperId(result.papers[0]?.id ?? "");
      const sourceErrors = result.source_errors ?? [];
      setSearchStatus(sourceErrors.length
        ? t(locale, `Completed with ${sourceErrors.length} source error(s). ${result.papers.length} papers are available.`, `检索完成，${sourceErrors.length} 个来源异常；当前可用 ${result.papers.length} 篇论文。`)
        : t(locale, `Verified retrieval completed: ${result.papers.length} papers.`, `真实检索完成：${result.papers.length} 篇论文。`));
      await recordReceipt("literature_search", {
        query: trimmed,
        context_path: result.context_path,
        manifest_path: result.manifest_path,
        source_counts: result.source_counts,
        source_errors: sourceErrors,
      });
      await refreshSummary?.();
    } catch (error) {
      setSearchStatus(error instanceof Error ? error.message : t(locale, "Literature search failed.", "文献检索失败。"));
    } finally {
      setIsSearching(false);
    }
  };

  const importFile = async (file: File) => {
    if (isImporting) return;
    setIsImporting(true);
    setSearchStatus(t(locale, `Importing ${file.name} and verifying provenance...`, `正在导入 ${file.name} 并校验来源...`));
    try {
      const result = await api.importLiterature(selectedTask, file);
      await recordReceipt("literature_import", {
        manifest_path: result.manifest_path,
        source_artifact: result.source_artifact,
        text_artifact: result.text_artifact,
        sha256: result.paper.provenance?.checksum,
        paper_id: result.paper.id,
      });
      setSearchQuery(result.paper.title);
      setSearchStatus(t(locale, `Imported ${result.paper.title}; rebuilding the task index...`, `已导入 ${result.paper.title}，正在重建当前任务索引...`));
      await runSearch(result.paper.title);
    } catch (error) {
      setSearchStatus(error instanceof Error ? error.message : t(locale, "Import failed.", "文献导入失败。"));
    } finally {
      setIsImporting(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const executeAgentAction = async (action: keyof typeof actionLabels) => {
    if (!runWorkstationAction || busyAction) return;
    if (!hasManifest) {
      setActionState(action, { status: "failed", message: t(locale, "Run a real search for this task first.", "请先为当前任务执行一次真实检索。") });
      return;
    }
    if ((action === "rag_bind_report_claim" || action === "rag_request_citation_audit") && (!selectedPaperId || claim.trim().length < 8)) {
      setActionState(action, { status: "failed", message: t(locale, "Select a paper and enter a claim of at least 8 characters.", "请选择论文，并输入至少 8 个字符的报告声明。") });
      return;
    }
    setBusyAction(action);
    setActionState(action, { status: "running", message: t(locale, "Running...", "执行中...") });
    try {
      const response = await runWorkstationAction(action, actionMetadata({
        selected_paper_id: selectedPaperId || null,
        claim: claim.trim() || null,
      })) as WorkstationActionResponse;
      const artifact = response.artifact_path ?? response.artifact ?? undefined;
      setActionState(action, { status: "passed", message: response.message, artifact });
      await refreshSummary?.();
      setLiveByTask((current) => {
        const next = { ...current };
        delete next[selectedTask];
        return next;
      });
    } catch (error) {
      setActionState(action, { status: "failed", message: error instanceof Error ? error.message : t(locale, "Action failed.", "动作执行失败。") });
    } finally {
      setBusyAction(null);
    }
  };

  const downloadTextFile = (name: string, content: string, mime: string) => {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = name;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const exportRagMarkdown = async () => {
    const content = literature?.context_markdown || [
      "# RAG Context Export",
      "",
      `Task: ${selectedTask}`,
      `Papers: ${papers.length}`,
      "",
      ...papers.slice(0, 20).map((paper) => `- ${paper.title} (${paper.year || "n/a"})`),
    ].join("\n");
    downloadTextFile(`rag_context_${selectedTask}.md`, content, "text/markdown;charset=utf-8");
    await recordReceipt("rag_export_context_markdown", { papers: papers.length });
  };

  const exportRagManifest = async () => {
    downloadTextFile(`rag_manifest_${selectedTask}.json`, JSON.stringify(literature ?? {}, null, 2), "application/json;charset=utf-8");
    await recordReceipt("rag_export_manifest_json", { papers: papers.length });
  };

  const sourceCount = new Set(
    papers.map((paper) => paper.source).filter((source) =>
      ["arxiv", "openalex", "crossref", "imported"].includes(source)
    )
  ).size;
  const auditCount = literature?.citation_audits?.length
    ?? (latestCitationAudit ? 1 : 0);

  return (
    <div className="space-y-4">
      <PageHeader
        title={t(locale, "Literature / RAG", "文献 / RAG")}
        subtitle={t(locale, "Verified retrieval, user imports, agent handoffs and independent citation review", "真实检索、用户导入、Agent 交接与独立引文审核")}
        breadcrumb={`${t(locale, "Workbench", "工坊")} > ${t(locale, "Literature / RAG", "文献 / RAG")} > ${selectedTask}`}
        primaryAction={
          <button
            type="button"
            data-ui-action="literature_refresh_library"
            data-ui-skip-action="true"
            onClick={() => void runSearch()}
            disabled={isSearching || !searchQuery.trim()}
            className="flex min-h-9 items-center gap-1.5 rounded-md border border-edge px-3 py-1.5 text-xs font-medium text-ink-secondary hover:bg-surface-sunken disabled:cursor-not-allowed disabled:opacity-50"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${isSearching ? "animate-spin" : ""}`} />
            {t(locale, "Refresh index", "刷新索引")}
          </button>
        }
      />

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricTile label={t(locale, "Papers", "论文")} value={papers.length} icon={BookOpen} tone="blue" />
        <MetricTile label={t(locale, "Reviews", "审核")} value={auditCount} icon={ShieldCheck} tone="neutral" />
        <MetricTile label={t(locale, "No Fake Papers", "无伪造论文")} value={papers.length ? (noFakePapers ? t(locale, "Verified", "已验证") : t(locale, "Check", "需检查")) : t(locale, "No papers", "待检索")} icon={FileCheck2} tone={papers.length ? (noFakePapers ? "green" : "red") : "neutral"} />
        <MetricTile label={t(locale, "Active Sources", "有效来源")} value={sourceCount} icon={Filter} tone="neutral" />
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1.7fr)_minmax(280px,0.8fr)]">
        <div className="min-w-0 space-y-4">
          <Panel title={t(locale, "Verified Search & Import", "真实检索与文献导入")}>
            <label htmlFor="literature-search" className="mb-1.5 block text-xs font-medium text-ink-secondary">
              {t(locale, "Keywords, method, paper title or author", "关键词、方法、论文标题或作者")}
            </label>
            <div className="flex flex-col gap-2 sm:flex-row">
              <div className="relative min-w-0 flex-1">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-muted" />
                <input
                  id="literature-search"
                  type="search"
                  value={searchQuery}
                  onChange={(event) => setSearchQuery(event.target.value)}
                  onKeyDown={(event) => { if (event.key === "Enter") void runSearch(); }}
                  placeholder={t(locale, "e.g. tabular deep learning calibration", "例如：表格深度学习与概率校准")}
                  className="h-10 w-full rounded-md border border-edge bg-surface pl-10 pr-3 text-sm text-ink placeholder:text-ink-faint focus:outline-none focus:ring-2 focus:ring-accent/40"
                />
              </div>
              <button
                type="button"
                data-ui-action="literature_search_manual"
                data-ui-skip-action="true"
                onClick={() => void runSearch()}
                disabled={isSearching || !searchQuery.trim()}
                className="flex min-h-10 items-center justify-center gap-2 rounded-md bg-accent px-4 text-xs font-semibold text-accent-fg hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {isSearching ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
                {isSearching ? t(locale, "Searching", "检索中") : t(locale, "Search 3 sources", "检索三大来源")}
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.md,.markdown,.txt,.json,application/pdf,text/plain,text/markdown,application/json"
                className="sr-only"
                onChange={(event) => { const file = event.target.files?.[0]; if (file) void importFile(file); }}
              />
              <button
                type="button"
                data-ui-action="literature_import_file"
                data-ui-skip-action="true"
                onClick={() => fileInputRef.current?.click()}
                disabled={isImporting}
                className="flex min-h-10 items-center justify-center gap-2 rounded-md border border-edge px-4 text-xs font-semibold text-ink-secondary hover:bg-surface-sunken disabled:cursor-not-allowed disabled:opacity-50"
              >
                {isImporting ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
                {isImporting ? t(locale, "Importing", "导入中") : t(locale, "Import paper", "导入论文")}
              </button>
            </div>
            <div className="mt-2 min-h-5 text-[11px] text-ink-muted" aria-live="polite" aria-atomic="true">
              {searchStatus || t(locale, "PDF, Markdown, TXT and JSON are preserved with SHA-256 provenance.", "支持 PDF、Markdown、TXT 与 JSON；原件和 SHA-256 来源证据会完整保留。")}
            </div>
          </Panel>

          <Panel title={t(locale, "Task Papers", "当前任务论文")}>
            {papers.length === 0 ? (
              <div className="py-8 text-center text-xs text-ink-muted">
                {t(locale, "No literature exists for this task. Run a search or import a paper.", "当前任务尚无文献。请执行检索或导入论文。")}
              </div>
            ) : (
              <div className="max-h-[520px] space-y-2 overflow-y-auto pr-1">
                {papers.slice(0, 18).map((paper, index) => {
                  const href = sourceHref(paper);
                  const checked = selectedPaperId === paper.id;
                  return (
                    <label key={paper.id || index} className={`block cursor-pointer rounded-md border p-3 transition-colors ${checked ? "border-accent bg-accent/5" : "border-edge hover:bg-surface-sunken"}`}>
                      <div className="flex items-start gap-3">
                        <input
                          type="radio"
                          name="selected-literature-paper"
                          value={paper.id}
                          checked={checked}
                          onChange={() => setSelectedPaperId(paper.id)}
                          className="mt-1 h-4 w-4 accent-current"
                        />
                        <div className="min-w-0 flex-1">
                          <div className="flex flex-wrap items-start gap-2">
                            <span className="min-w-0 flex-1 text-sm font-medium leading-5 text-ink">{paper.title}</span>
                            <StatusBadgeV2 tone={paper.provenance?.verified ? "verified" : "pending"} size="xs">
                              {paper.source}
                            </StatusBadgeV2>
                          </div>
                          <p className="mt-1 line-clamp-2 text-[11px] leading-4 text-ink-muted">
                            {[paper.authors?.slice(0, 4).join(", "), paper.venue, paper.year].filter(Boolean).join(" · ")}
                          </p>
                          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-ink-muted">
                            {paper.doi ? <span className="font-mono">DOI {paper.doi}</span> : null}
                            {paper.provenance?.checksum ? <span className="font-mono">SHA-256 {paper.provenance.checksum.slice(0, 12)}...</span> : null}
                            {href ? (
                              <a href={href} target="_blank" rel="noreferrer" onClick={(event) => event.stopPropagation()} className="inline-flex items-center gap-1 font-medium text-accent hover:underline">
                                {t(locale, "Open source", "打开来源")} <ExternalLink className="h-3 w-3" />
                              </a>
                            ) : null}
                          </div>
                        </div>
                      </div>
                    </label>
                  );
                })}
              </div>
            )}
            <AiLabel />
          </Panel>

          <Panel title={t(locale, "Agent Evidence Workflow", "Agent 证据工作流")}>
            <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(260px,0.8fr)]">
              <div>
                <label htmlFor="literature-claim" className="mb-1.5 block text-xs font-medium text-ink-secondary">
                  {t(locale, "Report claim to verify", "需要验证的报告声明")}
                </label>
                <textarea
                  id="literature-claim"
                  value={claim}
                  onChange={(event) => setClaim(event.target.value)}
                  rows={4}
                  placeholder={t(locale, "State one precise claim supported by the selected paper...", "输入一条需要由所选论文支持的精确声明...")}
                  className="w-full resize-y rounded-md border border-edge bg-surface p-3 text-xs leading-5 text-ink placeholder:text-ink-faint focus:outline-none focus:ring-2 focus:ring-accent/40"
                />
                <p className="mt-1 text-[10px] text-ink-muted">
                  {selectedPaper ? t(locale, `Selected: ${selectedPaper.title}`, `已选择：${selectedPaper.title}`) : t(locale, "Select one paper above.", "请先在上方选择一篇论文。")}
                </p>
              </div>
              <ol className="space-y-2 text-[11px] text-ink-secondary">
                <li className="flex gap-2"><span className="font-semibold text-accent">1</span><span>{t(locale, "Build a versioned context from the task manifest.", "从当前任务 manifest 构建版本化上下文。")}</span></li>
                <li className="flex gap-2"><span className="font-semibold text-accent">2</span><span>{t(locale, "Send evidence-only handoffs to Research or Code Agent.", "向 Research 或 Code Agent 发送仅含证据的 handoff。")}</span></li>
                <li className="flex gap-2"><span className="font-semibold text-accent">3</span><span>{t(locale, "Bind a claim to one paper and run an independent read-only review.", "将声明绑定到单篇论文，再由独立只读 Reviewer 审核。")}</span></li>
              </ol>
            </div>
            <div className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
              {([
                ["rag_build_agent_context", Database],
                ["rag_send_research_agent", Send],
                ["rag_send_code_agent", Send],
                ["rag_bind_report_claim", Link2],
                ["rag_request_citation_audit", ShieldCheck],
              ] as const).map(([action, Icon]) => {
                const state = actionStates[action];
                const running = busyAction === action;
                return (
                  <button
                    key={action}
                    type="button"
                    data-ui-action={action}
                    data-ui-skip-action="true"
                    onClick={() => void executeAgentAction(action)}
                    disabled={Boolean(busyAction) || !hasManifest}
                    className="flex min-h-10 items-center justify-center gap-2 rounded-md border border-edge px-3 py-2 text-xs font-medium text-ink-secondary hover:bg-surface-sunken disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {running ? <LoaderCircle className="h-4 w-4 animate-spin" /> : state?.status === "passed" ? <CheckCircle2 className="h-4 w-4 text-success" /> : <Icon className="h-4 w-4" />}
                    {t(locale, actionLabels[action].en, actionLabels[action].zh)}
                  </button>
                );
              })}
            </div>
            <div className="mt-3 space-y-1" aria-live="polite" aria-atomic="false">
              {Object.entries(actionStates).map(([action, state]) => (
                <div key={action} className={`rounded border px-2.5 py-2 text-[11px] ${state.status === "failed" ? "border-danger/45 bg-danger-light text-danger-text" : state.status === "passed" ? "border-success/45 bg-success-light text-success-text" : "border-edge bg-surface-sunken text-ink-muted"}`}>
                  <b>{t(locale, actionLabels[action]?.en ?? action, actionLabels[action]?.zh ?? action)}:</b> {state.message}
                  {state.artifact ? <div className="mt-1 break-all font-mono text-[10px]">{state.artifact}</div> : null}
                </div>
              ))}
            </div>
          </Panel>
        </div>

        <aside className="min-w-0 space-y-4">
          <Panel title={t(locale, "Independent Reviewer", "独立 Reviewer")}>
            {latestCitationAudit ? (
              <div className="space-y-3 text-xs">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-ink-muted">{t(locale, "Citation gate", "引文 Gate")}</span>
                  <StatusBadgeV2 tone={auditTone(latestCitationAudit.status ?? "pending")} size="xs">
                    {latestCitationAudit.status ?? "pending"}
                  </StatusBadgeV2>
                </div>
                <p className="leading-5 text-ink-secondary">{latestCitationAudit.conclusion}</p>
                {(latestCitationAudit.blockers?.length ?? 0) > 0 ? (
                  <div className="rounded border border-danger/45 bg-danger-light p-2 text-[11px] text-danger-text">
                    {latestCitationAudit.blockers?.join(" · ")}
                  </div>
                ) : null}
                {(latestCitationAudit.open_requirements?.length ?? 0) > 0 ? (
                  <div className="rounded border border-warning/45 bg-warning-light p-2 text-[11px] text-warning-text">
                    {latestCitationAudit.open_requirements?.join(" · ")}
                  </div>
                ) : null}
                {literature?.citation_audit_path ? <div className="break-all rounded border border-edge bg-surface-sunken p-2 font-mono text-[10px]">{literature.citation_audit_path}</div> : null}
              </div>
            ) : (
              <div className="space-y-2 text-xs text-ink-muted">
                <p>{t(locale, "No independent citation audit has been completed for this task.", "当前任务尚未完成独立引文审计。")}</p>
                {legacyClaimAudit.slice(0, 4).map((audit, index) => (
                  <div key={`${audit.paper}-${index}`} className="flex items-center justify-between gap-2">
                    <span className="truncate">{audit.claim}</span>
                    <StatusDot tone={auditTone(audit.status)} />
                  </div>
                ))}
              </div>
            )}
          </Panel>

          <Panel title={t(locale, "Task Provenance", "当前任务来源证据")} compact>
            <div className="space-y-2 text-[11px]">
              <div className="flex justify-between gap-3"><span className="text-ink-muted">arXiv</span><b>{literature?.source_counts?.arxiv ?? 0}</b></div>
              <div className="flex justify-between gap-3"><span className="text-ink-muted">OpenAlex</span><b>{literature?.source_counts?.openalex ?? 0}</b></div>
              <div className="flex justify-between gap-3"><span className="text-ink-muted">Crossref</span><b>{literature?.source_counts?.crossref ?? 0}</b></div>
              <div className="flex justify-between gap-3"><span className="text-ink-muted">{t(locale, "User imports", "用户导入")}</span><b>{literature?.source_counts?.imported ?? 0}</b></div>
              {literature?.context_path ? <div className="break-all rounded border border-edge bg-surface-sunken p-2 font-mono text-[10px]">context: {literature.context_path}</div> : null}
              {literature?.manifest_path ? <div className="break-all rounded border border-edge bg-surface-sunken p-2 font-mono text-[10px]">manifest: {literature.manifest_path}</div> : null}
              {literature?.agent_context_path ? <div className="break-all rounded border border-edge bg-surface-sunken p-2 font-mono text-[10px]">agent: {literature.agent_context_path}</div> : null}
              {literature?.claim_binding_path ? <div className="break-all rounded border border-edge bg-surface-sunken p-2 font-mono text-[10px]">binding: {literature.claim_binding_path}</div> : null}
              {(literature?.source_errors?.length ?? 0) > 0 ? (
                <div className="rounded border border-danger/45 bg-danger-light p-2 text-danger-text">
                  {literature?.source_errors?.map((item) => `${item.source}: ${item.error}`).join("; ")}
                </div>
              ) : null}
            </div>
          </Panel>

          <Panel title={t(locale, "Handoffs", "Agent 交接")} compact>
            <div className="space-y-2 text-[11px]">
              {(literature?.handoffs?.length ?? 0) === 0 ? <span className="text-ink-muted">{t(locale, "No handoffs yet", "尚无 handoff")}</span> : null}
              {literature?.handoffs?.slice(-5).reverse().map((handoff, index) => (
                <div key={handoff.handoff_id ?? index} className="rounded border border-edge p-2">
                  <div className="flex items-center justify-between gap-2">
                    <b className="text-ink-secondary">{handoff.recipient}</b>
                    <StatusBadgeV2 tone="pending" size="xs">{handoff.status ?? "queued"}</StatusBadgeV2>
                  </div>
                  <div className="mt-1 break-all font-mono text-[10px] text-ink-muted">{handoff.handoff_id}</div>
                </div>
              ))}
            </div>
          </Panel>

          <Panel title={t(locale, "Export", "导出")} compact>
            <div className="grid grid-cols-2 gap-2">
              <button type="button" onClick={() => void exportRagMarkdown()} disabled={!papers.length} className="flex min-h-10 items-center justify-center gap-1.5 rounded-md border border-edge px-2 text-xs font-medium text-ink-secondary hover:bg-surface-sunken disabled:opacity-50">
                <Download className="h-4 w-4" /> MD
              </button>
              <button type="button" onClick={() => void exportRagManifest()} disabled={!papers.length} className="flex min-h-10 items-center justify-center gap-1.5 rounded-md border border-edge px-2 text-xs font-medium text-ink-secondary hover:bg-surface-sunken disabled:opacity-50">
                <Database className="h-4 w-4" /> JSON
              </button>
            </div>
          </Panel>

          <ClaimBoundary />
        </aside>
      </div>
    </div>
  );
}
