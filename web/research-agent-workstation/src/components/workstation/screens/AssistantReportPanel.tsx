"use client";

import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import {
  BarChart3,
  Check,
  ClipboardCheck,
  Download,
  FileArchive,
  FileText,
  FlaskConical,
  Loader2,
  RefreshCw,
  ShieldCheck,
} from "lucide-react";
import {
  assistantReportNarrative,
  assistantReportPresentation,
  assistantReportTitle,
  assistantReportLead,
  reportJobStatusLabel,
  checkedAssistantReportDetail,
  reportExportDownloadUrl,
  reportableAssistantRuns,
  parseReportMarkdown,
  parseReportCsv,
  preferredAssistantReportArtifact,
  resolveAssistantReportRunId,
  type AssistantReportArtifact,
  type AssistantReportRun,
} from "@/lib/assistant-report";
import { cn } from "@/lib/utils";
import { PageHeader } from "../primitives/Layout";
import { t } from "../localization";

type Locale = "zh-CN" | "en-US";
type AssistantReportTab = "report" | "figures" | "methods" | "audit" | "files";
type ReportJob = { id: string; run_id: string; status: string; stage: string; report_status: string;
  evidence_status: string; error_code?: string; artifacts?: AssistantReportArtifact[] };

const tabs: Array<{ id: AssistantReportTab; en: string; zh: string; icon: typeof FileText }> = [
  { id: "report", en: "Report", zh: "报告", icon: FileText },
  { id: "figures", en: "Figures", zh: "图表", icon: BarChart3 },
  { id: "methods", en: "Execution", zh: "执行", icon: FlaskConical },
  { id: "audit", en: "Audit", zh: "审计", icon: ShieldCheck },
  { id: "files", en: "Files", zh: "文件", icon: FileArchive },
];

function formatBytes(value: number) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(2)} KB`;
  return `${(value / 1024 ** 2).toFixed(2)} MB`;
}

function formatDate(value?: string) {
  if (!value) return "—";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString("zh-CN", { hour12: false });
}

function inline(value: string): ReactNode[] {
  return value.split(/(`[^`]+`|\*\*[^*]+\*\*)/g).filter(Boolean).map((part, index) => {
    if (part.startsWith("`") && part.endsWith("`")) {
      return <code key={index} className="rounded bg-surface-sunken px-1.5 py-0.5 font-mono text-[0.92em] text-accent-dark">{part.slice(1, -1)}</code>;
    }
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={index} className="font-semibold text-ink">{part.slice(2, -2)}</strong>;
    }
    return part;
  });
}

export function ProfessionalMarkdown({ value }: { value: string }) {
  const blocks = useMemo(() => parseReportMarkdown(value), [value]);
  return (
    <article className="report-paper mx-auto w-full max-w-5xl rounded-lg border border-edge bg-surface-paper px-6 py-8 text-ink shadow-raised sm:px-10 sm:py-10" data-ui-professional-report>
      {blocks.map((block, index) => {
        if (block.kind === "heading") {
          if (block.level === 1) return <h1 key={index} className="mb-6 mt-2 border-b border-edge pb-4 text-3xl font-black tracking-tight text-ink">{inline(block.text ?? "")}</h1>;
          if (block.level === 2) return <h2 key={index} className="mb-3 mt-8 text-xl font-bold text-ink">{inline(block.text ?? "")}</h2>;
          return <h3 key={index} className="mb-2 mt-6 text-base font-bold text-ink">{inline(block.text ?? "")}</h3>;
        }
        if (block.kind === "list") {
          return <ul key={index} className="my-4 space-y-2 pl-1">{(block.items ?? []).map((item, itemIndex) => <li key={itemIndex} className="flex gap-3 text-sm leading-7 text-ink-secondary"><span className="mt-3 h-1.5 w-1.5 shrink-0 rounded-full bg-accent" /> <span>{inline(item)}</span></li>)}</ul>;
        }
        if (block.kind === "code") {
          return <pre key={index} className="my-5 overflow-x-auto rounded-lg border border-edge bg-frame p-4 text-xs leading-6 text-white">{block.text}</pre>;
        }
        if (block.kind === "table") {
          return <div key={index} className="my-5 overflow-x-auto rounded-lg border border-edge"><table className="min-w-full text-left text-xs"><thead className="bg-surface-sunken text-ink"><tr>{(block.headers ?? []).map((header, cellIndex) => <th key={cellIndex} className="border-b border-edge px-3 py-2.5 font-bold">{inline(header)}</th>)}</tr></thead><tbody>{(block.rows ?? []).map((row, rowIndex) => <tr key={rowIndex} className="border-b border-edge-light last:border-0">{row.map((cell, cellIndex) => <td key={cellIndex} className="px-3 py-2.5 text-ink-secondary">{inline(cell)}</td>)}</tr>)}</tbody></table></div>;
        }
        return <p key={index} className="my-3 text-sm leading-7 text-ink-secondary">{inline(block.text ?? "")}</p>;
      })}
    </article>
  );
}

function CsvReport({ value }: { value: string }) {
  const parsed = parseReportCsv(value);
  const rows = parsed.rows;
  const headers = rows[0] ?? [];
  if (parsed.error) return <p role="alert" className="rounded border border-danger/30 bg-danger-light p-4 text-sm text-danger-text">CSV 格式无法解析：{parsed.error}</p>;
  return (
    <div className="overflow-hidden rounded-lg border border-edge bg-surface-paper shadow-raised" data-ui-professional-report>
      <div className="border-b border-edge bg-surface-sunken px-4 py-3 text-xs font-semibold text-ink-secondary">Verified CSV preview · {Math.max(0, rows.length - 1)} displayed rows</div>
      <div className="overflow-auto"><table className="min-w-full text-left text-xs"><thead className="bg-surface-sunken text-ink-secondary"><tr>{headers.map((header, index) => <th key={index} className="border-b border-edge px-3 py-2 font-semibold">{header}</th>)}</tr></thead><tbody>{rows.slice(1).map((row, rowIndex) => <tr key={rowIndex} className="border-b border-edge-light last:border-0">{headers.map((_, columnIndex) => <td key={columnIndex} className="px-3 py-2 font-mono text-ink-secondary">{row[columnIndex] ?? ""}</td>)}</tr>)}</tbody></table></div>
    </div>
  );
}

export function VerifiedReportDocument({
  name,
  url,
  mediaType,
  previewKind,
  expectedBytes,
  expectedSha256,
}: {
  name: string;
  url: string;
  mediaType: string;
  previewKind: string;
  expectedBytes: number | null;
  expectedSha256: string | null;
}) {
  const [text, setText] = useState("");
  const [verifiedPdfUrl, setVerifiedPdfUrl] = useState("");
  const [verifiedVisualUrl, setVerifiedVisualUrl] = useState("");
  const [archiveEntries, setArchiveEntries] = useState<Array<{ name: string; bytes: number }>>([]);
  const [state, setState] = useState<"loading" | "ready" | "failed">("loading");
  const [renderedIdentity, setRenderedIdentity] = useState("");
  const [error, setError] = useState("");
  const identity = JSON.stringify([url, expectedSha256, expectedBytes, previewKind, mediaType]);
  const visual = ["image", "svg", "html"].includes(previewKind);
  const metadataOnly = ["binary", "office"].includes(previewKind)
    || (expectedBytes !== null && expectedBytes > (previewKind === "pdf" ? 64 : 8) * 1024 * 1024);

  useEffect(() => {
    if (!url) {
      setState("failed");
      setError("Artifact preview URL is missing.");
      setRenderedIdentity(identity);
      return;
    }
    if (metadataOnly) { setState("ready"); return; }
    const controller = new AbortController();
    let objectUrl = "";
    let disposed = false;
    setState("loading");
    setError("");
    setVerifiedPdfUrl("");
    setVerifiedVisualUrl("");
    setArchiveEntries([]);
    const previewUrl = previewKind === "archive" ? `${url}?preview=1` : url;
    void fetch(previewUrl, { cache: "no-store", signal: controller.signal }).then(async (response) => {
      if (!response.ok) throw new Error(`Artifact preview failed (${response.status}).`);
      if (previewKind === "archive") {
        const payload = await response.json() as { preview?: Array<{ name?: unknown; bytes?: unknown }> };
        if (controller.signal.aborted || disposed) return;
        const entries = (Array.isArray(payload.preview) ? payload.preview : []).flatMap((entry) => {
          const entryName = typeof entry.name === "string" ? entry.name : "";
          const entryBytes = typeof entry.bytes === "number" && Number.isFinite(entry.bytes) && entry.bytes >= 0 ? entry.bytes : -1;
          return entryName && entryBytes >= 0 ? [{ name: entryName, bytes: entryBytes }] : [];
        }).slice(0, 500);
        setArchiveEntries(entries);
        setRenderedIdentity(identity);
        setState("ready");
        return;
      }
      const body = await response.arrayBuffer();
      const digest = await crypto.subtle.digest("SHA-256", body);
      if (controller.signal.aborted || disposed) return;
      const actualSha256 = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
      const headerSha256 = response.headers.get("X-Artifact-SHA256")?.toLowerCase() ?? "";
      const headerBytes = Number(response.headers.get("X-Artifact-Bytes"));
      if (
        (expectedBytes !== null && (body.byteLength !== expectedBytes || headerBytes !== expectedBytes))
        || (expectedSha256 && (actualSha256 !== expectedSha256.toLowerCase() || headerSha256 !== expectedSha256.toLowerCase()))
      ) {
        throw new Error("Artifact preview bytes failed SHA-256 verification.");
      }
      if (visual) {
        if (!expectedSha256 || expectedBytes === null) throw new Error("Artifact manifest identity is missing.");
        const nextObjectUrl = URL.createObjectURL(new Blob([body], { type: mediaType }));
        if (controller.signal.aborted || disposed) { URL.revokeObjectURL(nextObjectUrl); return; }
        objectUrl = nextObjectUrl;
        setVerifiedVisualUrl(objectUrl);
        setRenderedIdentity(identity);
        setState("ready");
        return;
      }
      if (previewKind === "pdf") {
        const nextObjectUrl = URL.createObjectURL(new Blob([body], { type: "application/pdf" }));
        if (controller.signal.aborted || disposed) {
          URL.revokeObjectURL(nextObjectUrl);
          return;
        }
        objectUrl = nextObjectUrl;
        setVerifiedPdfUrl(objectUrl);
        setRenderedIdentity(identity);
        setState("ready");
        return;
      }
      let decoded = new TextDecoder().decode(body);
      if (mediaType.includes("json")) {
        try { decoded = JSON.stringify(JSON.parse(decoded), null, 2); } catch { /* Preserve exact source text. */ }
      }
      setText(decoded);
      setRenderedIdentity(identity);
      setState("ready");
    }).catch((reason) => {
      if (controller.signal.aborted) return;
      setError(reason instanceof Error ? reason.message : "Artifact preview failed.");
      setRenderedIdentity(identity);
      setState("failed");
    });
    return () => {
      disposed = true;
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [expectedBytes, expectedSha256, mediaType, previewKind, url, visual, metadataOnly, identity]);

  if (state === "loading" || (!metadataOnly && renderedIdentity !== identity)) return <div className="flex min-h-[520px] items-center justify-center gap-2 text-sm text-ink-muted"><Loader2 className="h-4 w-4 animate-spin" /> Verifying report bytes</div>;
  if (state === "failed") return <div className="rounded-lg border border-danger/30 bg-danger-light p-4 text-sm text-danger-text">{error}</div>;
  if (metadataOnly) return <section className="rounded-lg border border-edge bg-surface-raised p-5 text-sm text-ink-secondary"><h3 className="font-semibold">{name}</h3><p className="mt-2">此文件仅展示登记信息，未在浏览器重新读取全部字节。请下载查看；Word 可编辑内容可通过同版本 PDF 预览。</p><a className="mt-3 inline-block text-accent underline" href={url} download={name} data-ui-skip-action="true">下载原始文件</a></section>;
  if (previewKind === "image" || previewKind === "svg") return <div className="report-paper rounded-lg border border-edge bg-surface-paper p-4"><img src={verifiedVisualUrl} alt={name} className="mx-auto max-h-[720px] max-w-full object-contain" /></div>;
  if (previewKind === "pdf") return <iframe src={verifiedPdfUrl} title={name} referrerPolicy="no-referrer" className="h-[760px] w-full rounded-lg border border-edge bg-white shadow-raised" data-ui-verified-pdf-preview />;
  if (previewKind === "html") return <iframe src={verifiedVisualUrl} title={name} sandbox="" referrerPolicy="no-referrer" className="h-[760px] w-full rounded-lg border border-edge bg-white shadow-raised" />;
  if (previewKind === "archive") return <div className="overflow-hidden rounded-lg border border-edge bg-surface-paper shadow-raised" data-ui-report-archive-preview><div className="border-b border-edge bg-surface-sunken px-4 py-3 text-xs font-semibold text-ink-secondary">Safe ZIP file list · {archiveEntries.length} entries</div>{archiveEntries.length ? archiveEntries.map((entry) => <div key={entry.name} className="flex justify-between gap-4 border-b border-edge-light px-4 py-3 text-xs last:border-0"><code className="break-all text-ink-secondary">{entry.name}</code><span className="shrink-0 text-ink-muted">{formatBytes(entry.bytes)}</span></div>) : <div className="px-4 py-10 text-center text-sm text-ink-muted">Archive contains no safe regular file entries.</div>}</div>;
  if (previewKind === "table" || mediaType.includes("csv")) return <CsvReport value={text} />;
  if (previewKind === "markdown" || mediaType.includes("markdown")) return <ProfessionalMarkdown value={text} />;
  return <pre className="max-h-[760px] overflow-auto whitespace-pre-wrap break-words rounded-lg border border-edge bg-surface-paper p-6 text-xs leading-6 text-ink shadow-raised" data-ui-professional-report>{text}</pre>;
}

function ArtifactCard({ artifact, selected, onSelect }: { artifact: AssistantReportArtifact; selected: boolean; onSelect: () => void }) {
  return (
    <div className={cn("rounded-lg border p-3", selected ? "border-accent/60 bg-accent-light/35" : "border-edge bg-surface-raised")} data-ui-report-artifact-card>
      <div className="flex items-start gap-3"><FileText className="mt-0.5 h-4 w-4 shrink-0 text-accent" /><div className="min-w-0 flex-1"><div className="truncate text-sm font-semibold text-ink">{artifact.name}</div><div className="mt-1 text-[11px] text-ink-muted">{formatBytes(artifact.bytes)} · SHA-256 {artifact.sha256.slice(0, 16)}…</div></div><span className="rounded bg-success-light px-1.5 py-0.5 text-[10px] font-semibold text-success-text">SHA-256 已登记</span></div>
      <div className="mt-3 flex flex-wrap gap-2"><button type="button" data-ui-skip-action="true" onClick={onSelect} className="rounded border border-edge px-2 py-1 text-xs text-ink-secondary hover:border-accent hover:text-accent">Preview</button><a href={artifact.download_url} data-ui-skip-action="true" download={artifact.name} className="inline-flex items-center gap-1 rounded border border-edge px-2 py-1 text-xs text-ink-secondary hover:border-accent hover:text-accent"><Download className="h-3 w-3" /> Download</a><button type="button" data-ui-skip-action="true" onClick={() => void navigator.clipboard.writeText(artifact.sha256)} className="rounded border border-edge px-2 py-1 text-xs text-ink-secondary hover:border-accent hover:text-accent">Copy hash</button></div>
    </div>
  );
}

export function AssistantReportPanel({ locale = "zh-CN", onOpenScientific }: { locale?: Locale; onOpenScientific: () => void }) {
  const [runs, setRuns] = useState<AssistantReportRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [selectedArtifactId, setSelectedArtifactId] = useState("");
  const [activeTab, setActiveTab] = useState<AssistantReportTab>("report");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [reportJobs, setReportJobs] = useState<ReportJob[]>([]);
  const [generationError, setGenerationError] = useState("");
  const [generationBusy, setGenerationBusy] = useState(false);
  const [runDetail, setRunDetail] = useState<AssistantReportRun | null>(null);
  const [detailError, setDetailError] = useState("");
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailRevision, setDetailRevision] = useState(0);

  const loadRuns = useCallback(async (signal?: AbortSignal) => {
    const response = await fetch("/api/assistant/runs?limit=100", { cache: "no-store", signal });
    if (!response.ok) throw new Error(`Assistant run list failed (${response.status}).`);
    const payload = await response.json() as { runs?: AssistantReportRun[] };
    const available = reportableAssistantRuns((Array.isArray(payload.runs) ? payload.runs : []).map((run) => ({
      ...run, artifacts: (run.artifacts ?? []).filter((artifact) => artifact.run_id === run.id),
    })));
    setRuns(available);
    const query = new URL(window.location.href).searchParams;
    const requested = query.get("report_run") || query.get("assistant_run") || "";
    setSelectedRunId(resolveAssistantReportRunId(available, requested));
    setError(requested && !available.some((run) => run.id === requested) ? "请求的任务不可用或不属于当前账户；未替换为其他任务。" : "");
    setLoading(false);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void loadRuns(controller.signal).catch((reason) => {
      if (controller.signal.aborted) return;
      setError(reason instanceof Error ? reason.message : "Assistant run list failed.");
      setLoading(false);
    });
    const refresh = () => { if (document.visibilityState === "visible") void loadRuns().catch(() => undefined); };
    const timer = window.setInterval(refresh, 15_000);
    window.addEventListener("focus", refresh);
    return () => { controller.abort(); window.clearInterval(timer); window.removeEventListener("focus", refresh); };
  }, [loadRuns]);

  useEffect(() => {
    const controller = new AbortController();
    setRunDetail(null);
    setDetailError("");
    setDetailLoading(Boolean(selectedRunId));
    if (!selectedRunId) return () => controller.abort();
    const refresh = async () => {
      try {
        const response = await fetch(`/api/assistant/runs/${encodeURIComponent(selectedRunId)}`, { cache: "no-store", signal: controller.signal });
        if (!response.ok) throw new Error(`任务详情读取失败（${response.status}），未将缺失内容视为成功。`);
        const detail = checkedAssistantReportDetail(await response.json(), selectedRunId);
        if (!controller.signal.aborted) { setRunDetail(detail); setDetailError(""); }
      } catch (reason) {
        if (!controller.signal.aborted) setDetailError(reason instanceof Error ? reason.message : "任务详情暂不可用。");
      } finally { if (!controller.signal.aborted) setDetailLoading(false); }
    };
    void refresh();
    const timer = window.setInterval(() => { if (document.visibilityState === "visible") void refresh(); }, 15000);
    return () => { controller.abort(); window.clearInterval(timer); };
  }, [selectedRunId, detailRevision]);

  const selectedRun = runDetail?.id === selectedRunId ? runDetail : runs.find((run) => run.id === selectedRunId) ?? null;
  const preferredArtifact = useMemo(() => preferredAssistantReportArtifact(selectedRun), [selectedRun]);
  const selectedArtifact = selectedRun?.artifacts?.find((artifact) => artifact.id === selectedArtifactId) ?? preferredArtifact;
  const reportNarrative = detailLoading ? t(locale, "Loading task details…", "正在读取任务详情…") : assistantReportNarrative(selectedRun, selectedArtifact, locale);
  const presentation = assistantReportPresentation(selectedRun, locale);
  const completedSteps = selectedRun?.plan?.steps?.filter((step) => step.status === "completed").length ?? 0;
  const totalSteps = selectedRun?.plan?.steps?.length ?? 0;
  const totalBytes = selectedRun?.artifacts?.reduce((sum, artifact) => sum + artifact.bytes, 0) ?? 0;
  const latestReportJob = reportJobs[0];

  useEffect(() => {
    setReportJobs([]);
    setGenerationError("");
    if (!selectedRunId) return;
    const controller = new AbortController();
    const refresh = async () => {
      try {
        const response = await fetch(`/api/assistant/runs/${encodeURIComponent(selectedRunId)}/reports`, { cache: "no-store", signal: controller.signal });
        if (!response.ok) return;
        const payload = await response.json() as { reports?: ReportJob[] };
        if (!controller.signal.aborted) setReportJobs((payload.reports ?? []).filter((job) => job.run_id === selectedRunId));
      } catch { /* The existing task record remains usable while report status is unavailable. */ }
    };
    void refresh();
    const interval = window.setInterval(() => void refresh(), 3000);
    return () => { controller.abort(); window.clearInterval(interval); };
  }, [selectedRunId]);

  const generateReport = async (reportId?: string) => {
    if (!selectedRun || generationBusy) return;
    setGenerationBusy(true);
    setGenerationError("");
    try {
      const response = await fetch(`/api/assistant/runs/${encodeURIComponent(selectedRun.id)}/reports`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(reportId ? { report_id: reportId, idempotency_key: `resume-${reportId}` } : {
          title: assistantReportTitle(selectedRun), language: locale,
          summary: selectedRun.answer_is_current !== false ? selectedRun.answer : undefined,
          report_kind: presentation.kind === "diagnostic" ? "diagnostic" : undefined,
          formats: ["markdown", "html", "docx", "pdf"], idempotency_key: `report-${selectedRun.id}-${crypto.randomUUID()}`,
        }),
      });
      const payload = await response.json() as { status?: string; result?: { ok?: boolean; content?: { report_job?: ReportJob }; error?: string } };
      if (!response.ok || !payload.result?.ok) throw new Error(payload.status === "waiting_approval" ? "报告生成需要审批，请返回助手处理。" : payload.result?.error || "报告生成服务暂不可用。");
      const job = payload.result.content?.report_job;
      if (!job || job.run_id !== selectedRun.id) throw new Error("报告与任务身份不匹配。");
      setReportJobs((current) => [job, ...current.filter((item) => item.id !== job.id)]);
    } catch (reason) {
      setGenerationError(reason instanceof Error ? reason.message : "报告生成失败。");
    } finally { setGenerationBusy(false); }
  };

  useEffect(() => { setSelectedArtifactId(preferredArtifact?.id ?? ""); setActiveTab("report"); }, [preferredArtifact?.id, selectedRunId]);
  useEffect(() => {
    const openAudit = () => setActiveTab("audit");
    window.addEventListener("evomind:open-report-evidence", openAudit);
    return () => window.removeEventListener("evomind:open-report-evidence", openAudit);
  }, []);

  const chooseRun = (runId: string) => {
    setSelectedRunId(runId);
    const url = new URL(window.location.href);
    url.searchParams.set("page", "report");
    url.searchParams.set("report_source", "assistant");
    url.searchParams.set("report_run", runId);
    url.searchParams.set("assistant_run", runId);
    const chosen = runs.find((run) => run.id === runId);
    if (chosen?.project_id) url.searchParams.set("project", chosen.project_id); else url.searchParams.delete("project");
    if (chosen?.conversation_id) url.searchParams.set("conversation", chosen.conversation_id); else url.searchParams.delete("conversation");
    window.history.replaceState(null, "", url);
  };

  const switchToScientific = () => {
    const url = new URL(window.location.href);
    url.searchParams.set("report_source", "scientific");
    url.searchParams.delete("report_run");
    window.history.replaceState(null, "", url);
    onOpenScientific();
  };

  return (
    <div className="space-y-4" data-ui-assistant-report-studio>
      <PageHeader
        title={t(locale, "Task Report Studio", "任务报告工作室")}
        subtitle={t(locale, "Execution state, file identity and research conclusions are shown separately", "任务执行、文件身份与科研结论分别展示")}
        breadcrumb={`${t(locale, "Workbench", "工作台")} > ${t(locale, "Reports", "报告")}`}
        secondaryActions={[
          <button key="refresh" type="button" onClick={() => void loadRuns()} title={t(locale, "Refresh task reports", "刷新任务报告")} className="flex h-8 w-8 items-center justify-center rounded-md border border-edge text-ink-secondary hover:bg-surface-sunken"><RefreshCw className="h-3.5 w-3.5" /></button>,
          <button key="scientific" type="button" onClick={switchToScientific} className="h-8 rounded-md border border-edge px-3 text-xs font-semibold text-ink-secondary hover:border-accent hover:text-accent">{t(locale, "Scientific runs", "科研运行报告")}</button>,
        ]}
      />

      {detailLoading ? <p role="status" className="text-sm text-ink-secondary">正在读取所选任务的完整记录…</p> : null}
      {detailError ? <div role="alert" className="rounded-lg border border-danger/30 bg-danger-light p-3 text-sm text-danger-text">{detailError}<button type="button" className="ml-3 underline" onClick={() => setDetailRevision((value) => value + 1)}>重试详情</button></div> : null}

      <section className="rounded-lg border border-edge bg-surface-raised p-4" data-ui-report-run-selector>
        <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-end">
          <label className="min-w-0"><span className="mb-1.5 block text-xs font-semibold text-ink-secondary">{t(locale, "Select a task record or report", "选择任务记录或报告")}</span><select value={selectedRunId} onChange={(event) => chooseRun(event.target.value)} disabled={loading || runs.length === 0} className="h-11 w-full rounded-md border border-edge bg-surface-sunken px-3 text-sm font-medium text-ink outline-none focus:border-accent"><option value="">{loading ? t(locale, "Loading tasks…", "正在加载任务…") : t(locale, "No task available", "暂无可用任务")}</option>{runs.map((run) => <option key={run.id} value={run.id}>{assistantReportTitle(run)} · {formatDate(run.completed_at || run.updated_at || run.created_at)}</option>)}</select></label>
          <div className="flex flex-wrap gap-2 text-xs"><span className="rounded border border-edge bg-surface-sunken px-2.5 py-2 text-ink-secondary">{runs.length} {t(locale, "tasks", "条任务")}</span>{selectedRun ? <span className={cn("rounded border px-2.5 py-2 font-semibold", presentation.statusTone === "success" ? "border-success/25 bg-success-light text-success-text" : presentation.statusTone === "danger" ? "border-danger/30 bg-danger-light text-danger-text" : presentation.statusTone === "warning" ? "border-warning/30 bg-warning-light text-warning-text" : "border-edge bg-surface-sunken text-ink-secondary")}>{presentation.statusLabel}</span> : null}</div>
        </div>
      </section>

      {selectedRun ? <section className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-edge bg-surface-raised px-4 py-3" data-ui-report-generation>
        <div className="min-w-0 text-xs text-ink-secondary">
          <strong>{t(locale, "Unified report package", "统一报告包")}</strong>
          <p className="mt-1 text-ink-muted">{latestReportJob ? reportJobStatusLabel(latestReportJob.status, locale) : t(locale, "Web, editable Word, PDF and source-bound figures", "网页、可编辑 Word、PDF 与来源绑定图表")}</p>
          {generationError || latestReportJob?.error_code ? <p role="alert" className="mt-1 text-danger-text">{generationError || latestReportJob?.error_code}</p> : null}
        </div>
        <div className="flex flex-wrap gap-2">
          {(latestReportJob?.artifacts ?? []).filter((artifact) => /report\.(docx|pdf)$|report-bundle\.zip$/.test(artifact.name) && reportExportDownloadUrl(selectedRun.id, latestReportJob?.run_id ?? "", artifact)).map((artifact) => <a key={artifact.id} href={reportExportDownloadUrl(selectedRun.id, latestReportJob?.run_id ?? "", artifact)} download={artifact.name} data-ui-skip-action="true" className="rounded border border-edge px-3 py-2 text-xs font-semibold text-ink-secondary">{artifact.name}</a>)}
          {latestReportJob && ["failed", "paused"].includes(latestReportJob.status) ? <button type="button" disabled={generationBusy} onClick={() => void generateReport(latestReportJob.id)} className="rounded border border-edge px-3 py-2 text-xs text-ink-secondary">{t(locale, "Resume this export", "恢复本次导出")}</button> : null}
          <button type="button" disabled={detailLoading || Boolean(detailError) || generationBusy || ["running", "queued"].includes(latestReportJob?.status ?? "")} onClick={() => void generateReport()} className="rounded bg-accent px-3 py-2 text-xs font-semibold text-accent-foreground disabled:opacity-50">{generationBusy ? t(locale, "Submitting…", "正在提交…") : t(locale, "Generate report package", "生成报告包")}</button>
        </div>
      </section> : null}

      {loading ? <div className="flex min-h-[560px] items-center justify-center gap-2 text-sm text-ink-muted"><Loader2 className="h-4 w-4 animate-spin" /> {t(locale, "Loading durable runs", "正在加载持久任务")}</div> : error ? <div className="rounded-lg border border-danger/30 bg-danger-light p-4 text-sm text-danger-text">{error}</div> : !selectedRun ? <div className="flex min-h-[560px] flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-edge text-center text-sm text-ink-muted"><FileText className="h-8 w-8" /><span>{t(locale, "Complete a task in the assistant and its report will appear here automatically.", "在智能助手中完成任务后，报告会自动出现在这里。")}</span></div> : <>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-lg border border-edge bg-surface-sunken px-4 py-3 text-xs text-ink-secondary" data-ui-report-context>
          <span>{t(locale, "Project", "项目")}：{selectedRun.project_id || t(locale, "Not associated", "未关联项目")}</span>
          <code className="break-all">Run：{selectedRun.id}</code>
          <span>{presentation.reportLabel}</span>
          <span className="text-ink-muted">{presentation.evidenceLabel}</span>
        </div>
        <section className="rounded-lg border border-edge bg-gradient-to-r from-surface-raised to-surface-sunken p-5">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between"><div className="max-w-3xl"><div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-success-text"><ClipboardCheck className="h-4 w-4" /> {presentation.title}</div><h2 className="mt-2 text-2xl font-black tracking-tight text-ink">{assistantReportTitle(selectedRun)}</h2><p className="mt-2 text-sm leading-6 text-ink-secondary">{assistantReportLead(selectedRun, locale)}</p></div><div className="grid w-full min-w-0 grid-cols-2 gap-3 lg:w-auto lg:min-w-[300px]"><div className="rounded border border-edge bg-surface-raised p-3"><div className="text-2xs font-semibold uppercase text-ink-muted">Model</div><div className="mt-1 truncate font-mono text-xs font-semibold text-ink">{[selectedRun.model_provider, selectedRun.model].filter(Boolean).join(" / ") || "not recorded"}</div></div><div className="rounded border border-edge bg-surface-raised p-3"><div className="text-2xs font-semibold uppercase text-ink-muted">{t(locale, "Published artifacts", "已发布产物")}</div><div className="mt-1 text-lg font-bold text-ink">{selectedRun.artifacts?.length ?? 0}</div></div><div className="rounded border border-edge bg-surface-raised p-3"><div className="text-2xs font-semibold uppercase text-ink-muted">{t(locale, "Execution steps", "执行步骤")}</div><div className="mt-1 text-lg font-bold text-ink">{completedSteps}/{totalSteps}</div></div><div className="rounded border border-edge bg-surface-raised p-3"><div className="text-2xs font-semibold uppercase text-ink-muted">{t(locale, "Output size", "产物大小")}</div><div className="mt-1 text-lg font-bold text-ink">{formatBytes(totalBytes)}</div></div></div></div>
        </section>

        <div className="grid grid-cols-5 border-b border-edge" role="tablist" aria-label={t(locale, "Task report views", "任务报告视图")}>{tabs.map((tab) => { const Icon = tab.icon; return <button key={tab.id} type="button" role="tab" data-ui-skip-action="true" aria-selected={activeTab === tab.id} onClick={() => setActiveTab(tab.id)} className={cn("flex h-10 items-center justify-center gap-1.5 border-b-2 px-2 text-xs font-semibold", activeTab === tab.id ? "border-accent text-accent" : "border-transparent text-ink-muted hover:text-ink-secondary")}><Icon className="h-3.5 w-3.5" /> {locale === "zh-CN" ? tab.zh : tab.en}</button>; })}</div>

        <div className="min-h-[320px]">
          {activeTab === "report" ? <div className="space-y-5 rounded-b-lg bg-frame/35 p-3 sm:p-6">{selectedArtifact ? <VerifiedReportDocument name={selectedArtifact.name} url={selectedArtifact.preview_url} mediaType={selectedArtifact.media_type} previewKind={selectedArtifact.preview_kind} expectedBytes={selectedArtifact.bytes} expectedSha256={selectedArtifact.sha256} /> : <section className="mx-auto max-w-5xl rounded-lg border border-edge bg-surface-raised p-5"><div className="text-xs font-semibold uppercase tracking-wide text-accent">{t(locale, "Executive summary", "执行摘要")}</div><div className="mt-2"><ProfessionalMarkdown value={reportNarrative || t(locale, "The task completed without a textual summary.", "当前没有可用的报告正文，请查看执行与审计记录。")}/></div></section>}</div> : activeTab === "figures" ? <div className="grid gap-4 p-4 lg:grid-cols-2">{(selectedRun.artifacts ?? []).filter((artifact) => ["image", "svg"].includes(artifact.preview_kind)).map((artifact) => <VerifiedReportDocument key={artifact.id} name={artifact.name} url={artifact.preview_url} mediaType={artifact.media_type} previewKind={artifact.preview_kind} expectedBytes={artifact.bytes} expectedSha256={artifact.sha256} />)}{!(selectedRun.artifacts ?? []).some((artifact) => ["image", "svg"].includes(artifact.preview_kind)) ? <div className="col-span-full flex min-h-[420px] items-center justify-center text-sm text-ink-muted">{t(locale, "This task did not publish figure artifacts.", "本任务没有发布图表产物。")}</div> : null}</div> : activeTab === "methods" ? <div className="grid gap-5 p-4 lg:grid-cols-2"><section className="rounded-lg border border-edge bg-surface-raised p-4"><h3 className="text-sm font-semibold text-ink">{t(locale, "Execution plan", "执行计划")}</h3><ol className="mt-3 space-y-2">{(selectedRun.plan?.steps ?? []).map((step, index) => <li key={step.id ?? index} className="flex gap-3 rounded border border-edge bg-surface-sunken p-3 text-xs"><span className={cn("mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full", step.status === "completed" ? "bg-success-light text-success" : "bg-warning-light text-warning")}>{step.status === "completed" ? <Check className="h-3 w-3" /> : index + 1}</span><span><strong className="block text-ink">{step.label || `Step ${index + 1}`}</strong>{step.detail ? <span className="mt-1 block text-ink-muted">{step.detail}</span> : null}</span></li>)}</ol></section><section className="rounded-lg border border-edge bg-surface-raised p-4"><h3 className="text-sm font-semibold text-ink">{t(locale, "Runtime identity", "运行身份")}</h3><dl className="mt-3 divide-y divide-edge text-xs"><div className="flex justify-between gap-4 py-2"><dt className="text-ink-muted">Run ID</dt><dd className="break-all font-mono text-ink-secondary">{selectedRun.id}</dd></div><div className="flex justify-between gap-4 py-2"><dt className="text-ink-muted">Model</dt><dd className="font-mono text-ink-secondary">{[selectedRun.model_provider, selectedRun.model].filter(Boolean).join(" / ") || "—"}</dd></div><div className="flex justify-between gap-4 py-2"><dt className="text-ink-muted">Created</dt><dd className="text-ink-secondary">{formatDate(selectedRun.created_at)}</dd></div><div className="flex justify-between gap-4 py-2"><dt className="text-ink-muted">Completed</dt><dd className="text-ink-secondary">{formatDate(selectedRun.completed_at)}</dd></div><div className="flex justify-between gap-4 py-2"><dt className="text-ink-muted">Attachments</dt><dd className="text-ink-secondary">{selectedRun.attachments?.length ?? 0}</dd></div></dl></section></div> : activeTab === "audit" ? <div className="grid gap-5 p-4 lg:grid-cols-[1.2fr_0.8fr]"><section className="rounded-lg border border-edge bg-surface-raised p-4"><h3 className="flex items-center gap-2 text-sm font-semibold text-ink"><ShieldCheck className="h-4 w-4 text-success" /> {t(locale, "Artifact verification", "产物校验")}</h3><div className="mt-3 space-y-2">{(selectedRun.artifacts ?? []).map((artifact) => <div key={artifact.id} className="rounded border border-edge bg-surface-sunken p-3"><div className="flex items-center justify-between gap-3"><span className="text-xs font-semibold text-ink">{artifact.name}</span><span className="text-[10px] font-semibold text-success-text">{t(locale, "SHA-256 recorded", "SHA-256 已登记")}</span></div><code className="mt-2 block break-all text-[10px] leading-5 text-ink-muted">{artifact.sha256}</code><div className="mt-1 text-[10px] text-ink-muted">{formatBytes(artifact.bytes)} · {formatDate(artifact.verified_at)}</div></div>)}</div></section><section className="rounded-lg border border-edge bg-surface-raised p-4"><h3 className="text-sm font-semibold text-ink">{t(locale, "Governance", "治理状态")}</h3><div className="mt-3 space-y-3"><div className="flex items-center justify-between rounded border border-edge bg-surface-sunken p-3 text-xs"><span className="text-ink-muted">Run state</span><span className="font-semibold text-ink">{presentation.statusLabel}</span></div><div className="flex items-center justify-between rounded border border-edge bg-surface-sunken p-3 text-xs"><span className="text-ink-muted">Pending approvals</span><span className="font-semibold text-ink">{(selectedRun.approvals ?? []).filter((approval) => approval.status === "pending").length}</span></div><div className="flex items-center justify-between rounded border border-edge bg-surface-sunken p-3 text-xs"><span className="text-ink-muted">Published artifacts</span><span className="font-semibold text-ink">{selectedRun.artifacts?.length ?? 0}</span></div></div></section></div> : <div className="grid gap-3 p-4 lg:grid-cols-2">{!selectedRun.artifacts?.length ? <p className="col-span-full py-8 text-sm text-ink-secondary">本任务未发布文件。只读诊断可查看报告摘要与执行记录，无需生成模型或图表。</p> : null}{(selectedRun.artifacts ?? []).map((artifact) => <ArtifactCard key={artifact.id} artifact={artifact} selected={artifact.id === selectedArtifact?.id} onSelect={() => { setSelectedArtifactId(artifact.id); setActiveTab("report"); }} />)}</div>}
        </div>
      </>}
    </div>
  );
}
