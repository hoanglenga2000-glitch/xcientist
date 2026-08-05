"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Archive,
  ArrowRight,
  BarChart3,
  Check,
  ChevronRight,
  Download,
  ExternalLink,
  FileArchive,
  FileCheck2,
  FileText,
  FlaskConical,
  Gauge,
  History,
  Loader2,
  LockKeyhole,
  Play,
  RefreshCw,
  Search,
  ShieldCheck,
  Sparkles,
  X,
} from "lucide-react";
import {
  controlMultiAgentRun,
  decideRefinement,
  generateScientificReport,
  getLatestRefinement,
  getScientificReport,
  getScientificReportGeneration,
  parseRefinement,
} from "@/lib/api/client";
import { ScientificReportPendingTimeoutError } from "@/lib/api/report-polling";
import type { RefinementPlan, ScientificArtifact, ScientificFigure, ScientificReportGenerationStage, ScientificReportGenerationStatus, ScientificReportPackage, WorkstationSummary } from "@/lib/api/types";
import { normalizeTaskId, runtimeForTask } from "@/lib/task-context";
import { cn } from "@/lib/utils";
import { PageHeader } from "../primitives/Layout";
import { t } from "../localization";

type Locale = "zh-CN" | "en-US";
type ReportTab = "report" | "figures" | "methods" | "audit" | "files";
type DownloadState = "idle" | "preparing" | "downloaded" | "failed";

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
  lastActionTrace?: { action: string; taskId?: string; request?: Record<string, unknown>; response?: Record<string, unknown>; message: string; artifact?: string | null; at: string } | null;
  locale?: Locale;
  setLocale?: (locale: Locale) => void;
};

const tabs: Array<{ id: ReportTab; label: string; icon: typeof FileText }> = [
  { id: "report", label: "Report", icon: FileText },
  { id: "figures", label: "Figures", icon: BarChart3 },
  { id: "methods", label: "Methods", icon: FlaskConical },
  { id: "audit", label: "Audit", icon: ShieldCheck },
  { id: "files", label: "Files", icon: Archive },
];

const categoryMeta: Record<ScientificArtifact["category"], { label: string; description: string; icon: typeof FileText }> = {
  research_report: { label: "Research Report", description: "Scientific report, figures and publication-ready views", icon: FileText },
  model_artifacts: { label: "Model Artifacts", description: "Adapter, configuration, tokenizer and model documentation", icon: Gauge },
  reproducibility: { label: "Reproducibility", description: "Run contract, environment, telemetry and checksums", icon: FileArchive },
  audit: { label: "Audit", description: "Independent Review, Claim Audit and Human Gate records", icon: ShieldCheck },
};

const generationStages: Array<{ id: ScientificReportGenerationStage; label: string }> = [
  { id: "collecting_evidence", label: "Collecting verified evidence" },
  { id: "loading_metrics", label: "Loading training metrics" },
  { id: "rendering_figures", label: "Rendering scientific figures" },
  { id: "building_report", label: "Building Nature Skills report" },
  { id: "rendering_pdf", label: "Rendering report PDF" },
  { id: "bundling", label: "Bundling reproducibility artifacts" },
  { id: "attaching_audit", label: "Attaching audit appendix" },
];

function artifactUrl(relativePath: string | null, taskId: string, runId: string, publicPresentation: boolean, download = false) {
  if (!relativePath) return "";
  if (relativePath.startsWith("public:")) {
    const artifactId = relativePath.slice("public:".length);
    const query = new URLSearchParams({ task_id: taskId, run_id: runId, artifact_id: artifactId });
    if (download) query.set("download", "1");
    return `/api/public-scientific-report/artifact?${query.toString()}`;
  }
  if (publicPresentation) return "";
  const query = new URLSearchParams({ path: relativePath, task_id: taskId, run_id: runId });
  if (download) query.set("download", "1");
  return `/api/artifacts?${query.toString()}`;
}

function filenameFromDisposition(disposition: string | null) {
  if (!disposition) return null;
  const encoded = disposition.match(/filename\*=UTF-8''([^;]+)/i)?.[1];
  if (encoded) {
    try { return decodeURIComponent(encoded.replace(/^"|"$/g, "")); } catch { return encoded; }
  }
  return disposition.match(/filename="?([^";]+)"?/i)?.[1] ?? null;
}

function artifactFilename(artifact: ScientificArtifact, disposition: string | null) {
  const headerName = filenameFromDisposition(disposition);
  if (headerName) return headerName;
  const extensions: Record<string, string> = { HTML: ".html", PDF: ".pdf", SVG: ".svg", ZIP: ".zip", JSON: ".json", JSONL: ".jsonl", Markdown: ".md", SafeTensors: ".safetensors" };
  const base = artifact.name.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "").toLowerCase() || "artifact";
  const extension = extensions[artifact.type] ?? "";
  return extension && !base.toLowerCase().endsWith(extension) ? `${base}${extension}` : base;
}

function formatBytes(value: number | null) {
  if (value === null) return "Not available";
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 ** 2).toFixed(value > 100 * 1024 ** 2 ? 0 : 1)} MB`;
}

function formatMetric(value: number | null, suffix = "") {
  return value === null ? "Not recorded" : `${value.toFixed(2)}${suffix}`;
}

function statusTone(value: string) {
  const status = value.toLowerCase();
  if (["ready", "passed", "completed", "approved", "downloaded"].includes(status)) return "border-success/25 bg-success-light text-success-text";
  if (["failed", "rejected"].includes(status)) return "border-danger/25 bg-danger-light text-danger-text";
  if (["running", "preparing"].includes(status)) return "border-info/25 bg-info-light text-info-text";
  return "border-warning/25 bg-warning-light text-warning-text";
}

function StatusPill({ value }: { value: string }) {
  return <span className={cn("inline-flex items-center rounded-sm border px-1.5 py-0.5 text-2xs font-semibold", statusTone(value))}>{value.replaceAll("_", " ")}</span>;
}

function MetricBlock({ label, value, detail }: { label: string; value: string; detail?: string }) {
  return (
    <div className="min-w-0 border-l-2 border-edge px-3 py-1 first:border-l-0 first:pl-0">
      <div className="text-2xs font-semibold uppercase text-ink-muted">{label}</div>
      <div className="mt-0.5 text-lg font-semibold tabular-nums text-ink">{value}</div>
      {detail && <div className="mt-0.5 truncate text-2xs text-ink-muted">{detail}</div>}
    </div>
  );
}

function EvidenceCheck({ label, passed }: { label: string; passed: boolean }) {
  return (
    <div className="flex min-h-9 items-center justify-between gap-3 border-b border-edge-light py-2 last:border-b-0">
      <span className="text-xs text-ink-secondary">{label.replaceAll("_", " ")}</span>
      <span className={cn("flex items-center gap-1 text-xs font-semibold", passed ? "text-success" : "text-danger")}>
        {passed ? <Check className="h-3.5 w-3.5" /> : <X className="h-3.5 w-3.5" />}
        {passed ? "Passed" : "Not passed"}
      </span>
    </div>
  );
}

export function ReportStudioScreen(props: ScreenProps) {
  const { locale = "zh-CN", refreshSummary, runWorkstationAction } = props;
  const selectedRuntime = runtimeForTask(props.summary, props.selectedTask);
  const currentRun = selectedRuntime?.current_run;
  const currentRunMatchesSelection = Boolean(
    currentRun?.task_id
    && normalizeTaskId(currentRun.task_id) === normalizeTaskId(props.selectedTask)
  );
  // The UI accepts legacy underscore aliases, but all run-bound APIs use the
  // canonical task ID persisted by the Supervisor in current_run.json.
  const taskId = currentRunMatchesSelection ? currentRun!.task_id! : props.selectedTask;
  const runId = currentRunMatchesSelection ? currentRun?.run_id ?? null : null;
  const bindingError = currentRun && !currentRunMatchesSelection
    ? `Selected task ${props.selectedTask} does not match current run task ${currentRun.task_id}.`
    : !currentRun ? "No current run is available for the selected task." : "";
  const [publicPresentation, setPublicPresentation] = useState<boolean | null>(null);
  const [activeTab, setActiveTab] = useState<ReportTab>("report");
  const [report, setReport] = useState<ScientificReportPackage | null>(null);
  const [reportState, setReportState] = useState<"loading" | "generating" | "ready" | "failed">("loading");
  const [generation, setGeneration] = useState<ScientificReportGenerationStatus | null>(null);
  const [reportError, setReportError] = useState("");
  const [selectedFigure, setSelectedFigure] = useState<ScientificFigure | null>(null);
  const [downloads, setDownloads] = useState<Record<string, DownloadState>>({});
  const [refinementPrompt, setRefinementPrompt] = useState("结果整体达到目标，但验证曲线后期仍有波动。请降低学习率，继续训练一个短周期，并与上一轮结果进行对比，然后更新报告。");
  const [refinement, setRefinement] = useState<RefinementPlan | null>(null);
  const [refinementBusy, setRefinementBusy] = useState(false);
  const [gateOpen, setGateOpen] = useState(false);
  const [notice, setNotice] = useState("");
  const closeGate = useCallback(() => setGateOpen(false), []);

  const loadReport = useCallback(async (generateReport = false, requestedRunId = runId, signal?: AbortSignal) => {
    if (signal?.aborted) return;
    if (!requestedRunId || publicPresentation === null) {
      setReport(null);
      setReportError(bindingError || "A task-bound run is required before loading a report.");
      setReportState("failed");
      return;
    }
    if (generateReport && publicPresentation) {
      setReportError("Public presentation mode is read-only.");
      setReportState("failed");
      return;
    }
    setReportError("");
    setReportState(generateReport ? "generating" : "loading");
    let generationTimer: number | null = null;
    const refreshGeneration = async () => {
      const response = await getScientificReportGeneration(taskId, requestedRunId);
      setGeneration(response.generation);
    };
    if (generateReport) {
      setGeneration(null);
      generationTimer = window.setInterval(() => void refreshGeneration().catch(() => undefined), 350);
    }
    try {
      const response = generateReport
        ? await generateScientificReport(taskId, requestedRunId)
        : await getScientificReport(taskId, requestedRunId, publicPresentation, {
          signal,
          ...(publicPresentation === false && requestedRunId === runId && currentRun?.status === "needs_continuation"
            ? { maxAttempts: 1 }
            : {}),
        });
      if (signal?.aborted) return;
      if (response.report.task_id !== taskId || response.report.run_id !== requestedRunId) throw new Error("Scientific report binding mismatch.");
      setReport(response.report);
      setReportState("ready");
      if (generateReport) await refreshGeneration().catch(() => undefined);
    } catch (error) {
      if (signal?.aborted) return;
      if (
        error instanceof ScientificReportPendingTimeoutError
        && !generateReport
        && publicPresentation === false
        && requestedRunId === runId
        && currentRun?.status === "needs_continuation"
      ) {
        try {
          const refinementResponse = await getLatestRefinement(taskId, requestedRunId);
          const parentRunId = refinementResponse.refinement?.parent_run_id;
          if (parentRunId && refinementResponse.refinement?.child_run_id === requestedRunId) {
            const parentResponse = await getScientificReport(taskId, parentRunId, false, { signal });
            if (signal?.aborted) return;
            if (parentResponse.report.task_id !== taskId || parentResponse.report.run_id !== parentRunId) throw new Error("Preserved report binding mismatch.");
            setRefinement(refinementResponse.refinement);
            setReport(parentResponse.report);
            setReportState("ready");
            setReportError("");
            setNotice(`${refinementResponse.refinement.proposed_version} needs continuation. Showing the immutable ${refinementResponse.refinement.parent_version} report.`);
            return;
          }
        } catch {
          // The original report error below remains the source of truth.
        }
      }
      setReport(null);
      setReportError(error instanceof Error ? error.message : generateReport ? "Scientific report generation failed" : "Scientific report is not available. Use Generate Scientific Report to create it.");
      setReportState("failed");
      if (generateReport) await refreshGeneration().catch(() => undefined);
    } finally {
      if (generationTimer !== null) window.clearInterval(generationTimer);
    }
  }, [bindingError, currentRun?.status, publicPresentation, runId, taskId]);

  const loadRefinement = useCallback(async () => {
    if (!runId || publicPresentation !== false) {
      setRefinement(null);
      return;
    }
    try {
      const response = await getLatestRefinement(taskId, runId);
      setRefinement(response.refinement);
    } catch {
      // No prior refinement is a valid initial state.
    }
  }, [publicPresentation, runId, taskId]);

  useEffect(() => {
    const params = new URL(window.location.href).searchParams;
    setPublicPresentation(params.get("presentation") === "public");
  }, []);

  useEffect(() => {
    if (publicPresentation === null) return;
    const controller = new AbortController();
    void loadReport(false, runId, controller.signal);
    void loadRefinement();
    return () => controller.abort();
  }, [loadReport, loadRefinement, publicPresentation, runId]);

  useEffect(() => {
    if (refinement?.status !== "running" && refinement?.status !== "approved") return;
    const timer = window.setInterval(async () => {
      try {
        if (!runId) return;
        const response = await getLatestRefinement(taskId, runId);
        setRefinement(response.refinement);
        if (response.refinement?.status === "completed") {
          window.clearInterval(timer);
          const completedRunId = response.refinement.child_run_id;
          if (!completedRunId) throw new Error("Completed refinement is missing its child run ID.");
          await loadReport(true, completedRunId);
          await refreshSummary?.();
          setNotice(`Scientific Report ${response.refinement.proposed_version} updated from the independently reviewed refinement run.`);
        }
      } catch {
        // The durable run remains queryable after a temporary UI disconnect.
      }
    }, 4000);
    return () => window.clearInterval(timer);
  }, [loadReport, refinement?.status, refreshSummary, runId, taskId]);

  const activeRefinementPending = Boolean(
    runId
    && refinement?.child_run_id === runId
    && currentRun?.status !== "completed"
  );

  const resumeRefinement = async () => {
    if (!runId || !activeRefinementPending) return;
    setRefinementBusy(true);
    setNotice("");
    try {
      await controlMultiAgentRun(runId, "resume");
      setRefinement((current) => current && current.child_run_id === runId
        ? { ...current, status: "running" }
        : current);
      setNotice(`${refinement?.proposed_version ?? "Refinement"} resume requested. The preserved ${refinement?.parent_version ?? "parent"} report remains available while remote execution is retried.`);
      await refreshSummary?.();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Refinement resume failed");
    } finally {
      setRefinementBusy(false);
    }
  };

  const artifactsByCategory = useMemo(() => {
    const groups: Record<ScientificArtifact["category"], ScientificArtifact[]> = { research_report: [], model_artifacts: [], reproducibility: [], audit: [] };
    for (const artifact of report?.artifacts ?? []) groups[artifact.category].push(artifact);
    return groups;
  }, [report]);

  const reviewerChecks = useMemo(() => {
    const checks = report?.reviewer?.checks;
    return checks && typeof checks === "object" && !Array.isArray(checks) ? Object.entries(checks as Record<string, unknown>) : [];
  }, [report]);

  const privateGrader = useMemo(() => {
    const value = report?.method?.private_grader;
    return value && typeof value === "object" && !Array.isArray(value)
      ? value as Record<string, unknown>
      : null;
  }, [report]);

  const versionComparison = report?.version_comparison ?? null;
  const comparisonOutcome = versionComparison?.outcome === "improved"
    ? "Improved"
    : versionComparison?.outcome === "trade_off_detected"
      ? "Trade-off detected"
      : versionComparison?.outcome === "no_material_change"
        ? "No material change"
        : "Review required";

  const downloadArtifact = async (artifact: ScientificArtifact) => {
    const artifactRunId = report?.run_id ?? runId;
    if (!artifact.path || artifact.status !== "ready" || !artifactRunId || publicPresentation === null) return;
    setDownloads((current) => ({ ...current, [artifact.id]: "preparing" }));
    try {
      const response = await fetch(artifactUrl(artifact.path, taskId, artifactRunId, publicPresentation, true));
      if (!response.ok) throw new Error(`Download failed (${response.status})`);
      const body = await response.blob();
      const responseHash = response.headers.get("X-Artifact-SHA256")?.toLowerCase() ?? "";
      const responseBytes = Number(response.headers.get("X-Artifact-Bytes"));
      if (!artifact.sha256 || responseHash !== artifact.sha256.toLowerCase() || responseBytes !== artifact.bytes || body.size !== artifact.bytes) {
        throw new Error("Downloaded bytes did not pass the report manifest checksum.");
      }
      const link = document.createElement("a");
      const url = URL.createObjectURL(body);
      link.href = url;
      link.download = artifactFilename(artifact, response.headers.get("Content-Disposition"));
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      setDownloads((current) => ({ ...current, [artifact.id]: "downloaded" }));
      setNotice(`${artifact.name} downloaded and SHA256 verified.`);
      props.setReportSubmitted(true);
      void runWorkstationAction?.("scientific_artifact_downloaded", { task_id: taskId, run_id: artifactRunId, artifact_id: artifact.id, sha256: artifact.sha256 });
    } catch (error) {
      setDownloads((current) => ({ ...current, [artifact.id]: "failed" }));
      setNotice(error instanceof Error ? error.message : "Download failed");
    }
  };

  const parseRequest = async () => {
    if (!runId || publicPresentation !== false || activeRefinementPending) {
      if (activeRefinementPending) {
        setNotice(`${refinement?.proposed_version ?? "The active refinement"} must be completed or cancelled before another refinement can be created.`);
        return;
      }
      setNotice(publicPresentation ? "Public presentation mode is read-only." : bindingError || "A task-bound run is required.");
      return;
    }
    setRefinementBusy(true);
    setNotice("");
    try {
      const response = await parseRefinement(taskId, runId, refinementPrompt);
      setRefinement(response.refinement);
      setGateOpen(true);
      void runWorkstationAction?.("llm_refinement_parsed", { task_id: taskId, refinement_id: response.refinement.refinement_id, affected_steps: response.refinement.affected_steps });
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Refinement parsing failed");
    } finally {
      setRefinementBusy(false);
    }
  };

  const decideGate = async (decision: "approve" | "reject") => {
    if (!refinement) return;
    setRefinementBusy(true);
    try {
      const response = await decideRefinement(taskId, refinement.refinement_id, decision);
      setRefinement(response.refinement);
      setGateOpen(false);
      setNotice(decision === "approve" ? `${response.refinement.proposed_version} created. Only affected steps will rerun.` : `Refinement rejected. ${response.refinement.parent_version} remains unchanged.`);
      void runWorkstationAction?.("llm_refinement_gate_decision", { task_id: taskId, refinement_id: refinement.refinement_id, decision });
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Human Gate decision failed");
    } finally {
      setRefinementBusy(false);
    }
  };

  const reportArtifact = report?.artifacts.find((item) => item.id === "report-html" && item.status === "ready");
  const finalBundle = report?.artifacts.find((item) => item.id === "final-bundle" && item.status === "ready");
  const generationStageIndex = generation?.stage === "ready"
    ? generationStages.length
    : generationStages.findIndex((stage) => stage.id === generation?.stage);
  const readyFigures = report?.figures.filter((item) => item.status === "ready") ?? [];
  const reportRunId = report?.run_id ?? runId;
  const showingPreservedParent = Boolean(report?.run_id && runId && report.run_id !== runId);
  const artifactHref = useCallback((artifactPath: string | null, download = false) => (
    reportRunId && publicPresentation !== null ? artifactUrl(artifactPath, taskId, reportRunId, publicPresentation, download) : ""
  ), [publicPresentation, reportRunId, taskId]);

  return (
    <div className="space-y-4">
      <PageHeader
        title={t(locale, "Scientific Report Studio", "科研报告工坊")}
        subtitle={t(locale, "Reviewed evidence, Nature Skills rendering and versioned delivery", "经审核证据、Nature Skills 渲染与版本化交付")}
        breadcrumb={`${t(locale, "Workbench", "工坊")} > ${t(locale, "Scientific Report", "科研报告")}`}
        secondaryActions={[
          <button key="refresh" type="button" title="Refresh scientific report" data-ui-action="report_refresh_scientific" data-ui-skip-action="true" onClick={() => void loadReport(false)} className="flex h-8 w-8 items-center justify-center rounded-md border border-edge text-ink-secondary hover:bg-surface-sunken">
            <RefreshCw className={cn("h-3.5 w-3.5", reportState === "loading" && "animate-spin")} />
          </button>,
        ]}
        primaryAction={publicPresentation !== false || reportState === "ready" ? undefined : (
          <button type="button" data-ui-action="report_generate_scientific" data-ui-skip-action="true" onClick={() => void loadReport(true)} disabled={reportState === "generating" || showingPreservedParent} title={showingPreservedParent ? "Resume the refinement before generating its report" : "Generate Scientific Report"} className="flex h-8 items-center gap-1.5 rounded-md bg-accent px-3 text-xs font-semibold text-accent-fg hover:opacity-90 disabled:opacity-60">
            {reportState === "generating" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
            {reportState === "generating" ? "Rendering" : "Generate Scientific Report"}
          </button>
        )}
      />

      <section className="border-y border-edge bg-surface-raised px-4 py-3">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-md border border-success/25 bg-success-light text-success"><ShieldCheck className="h-4.5 w-4.5" /></div>
            <div>
              <div className="flex items-center gap-2 text-sm font-semibold text-ink">Independent Review <StatusPill value={String(report?.reviewer?.status ?? reportState)} /></div>
              <div className="mt-0.5 text-xs text-ink-muted">Nature Skills uses only reviewed metrics, telemetry, data and audit evidence.</div>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <MetricBlock label="Version" value={report?.version ?? "—"} detail={report?.parent_run_id ? `${refinement?.parent_version ?? "Parent"} preserved` : "Reviewed baseline"} />
            <MetricBlock label="Before" value={formatMetric(report?.metrics.before ?? null)} />
            <MetricBlock label="After" value={formatMetric(report?.metrics.after ?? null)} />
            <MetricBlock label="Change" value={report?.metrics.improvement_pp === null || report?.metrics.improvement_pp === undefined ? "Not recorded" : `${report.metrics.improvement_pp >= 0 ? "+" : ""}${report.metrics.improvement_pp.toFixed(2)} pp`} />
          </div>
        </div>
      </section>

      {versionComparison && (
        <section aria-label="Reviewed version comparison" className="border-y border-edge bg-surface-raised px-4 py-4">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
            <div className="max-w-xl">
              <div className="flex items-center gap-2">
                <BarChart3 className="h-4 w-4 text-accent" />
                <h2 className="text-sm font-semibold text-ink">Reviewed version comparison</h2>
                <StatusPill value={comparisonOutcome} />
              </div>
              <p className="mt-1.5 text-xs leading-5 text-ink-muted">Fixed test metrics are supplied by VersionComparatorAgent and independently recomputed by the Reviewer. The parent run remains immutable.</p>
              <div className="mt-3 flex flex-wrap gap-x-5 gap-y-2 text-xs text-ink-secondary">
                {versionComparison.requested_changes.map((change) => (
                  <span key={`${change.field}-${change.source}`}>
                    <strong className="font-semibold text-ink">{change.field.replaceAll("_", " ")}</strong>
                    {" "}{String(change.old_value ?? "recorded")} <ArrowRight className="mx-1 inline h-3 w-3" /> {String(change.value ?? "recorded")}
                  </span>
                ))}
              </div>
            </div>
            <div className="grid min-w-0 grid-cols-3 gap-2 sm:min-w-[390px]">
              <MetricBlock label={versionComparison.parent_version} value={formatMetric(versionComparison.v1)} detail="Preserved" />
              <MetricBlock label={versionComparison.child_version} value={formatMetric(versionComparison.v2)} detail="Reviewed" />
              <MetricBlock label="Delta" value={versionComparison.delta_pp === null ? "Not recorded" : `${versionComparison.delta_pp >= 0 ? "+" : ""}${versionComparison.delta_pp.toFixed(2)} pp`} detail="Fixed test set" />
            </div>
          </div>
        </section>
      )}

      {notice && <div role="status" className="flex items-center justify-between gap-3 border border-info/20 bg-info-light px-3 py-2 text-xs text-info-text"><span>{notice}</span><button type="button" title="Dismiss" data-ui-action="report_dismiss_notice" data-ui-skip-action="true" onClick={() => setNotice("")}><X className="h-3.5 w-3.5" /></button></div>}
      {showingPreservedParent && <div className="flex flex-col gap-3 border border-warning/25 bg-warning-light px-3 py-2 text-xs leading-5 text-warning-text sm:flex-row sm:items-center sm:justify-between"><div className="flex min-w-0 items-start gap-2"><History className="mt-0.5 h-3.5 w-3.5 shrink-0" /><span>{refinement?.proposed_version ?? "Refinement"} {currentRun?.status?.replaceAll("_", " ") ?? "is paused"} and has no reviewed report. The immutable {refinement?.parent_version ?? report?.version ?? "parent"} report remains available while the failed HPC step is repaired.</span></div><button type="button" data-ui-action="report_resume_refinement" data-ui-skip-action="true" onClick={() => void resumeRefinement()} disabled={refinementBusy || !activeRefinementPending} className="flex h-8 shrink-0 items-center justify-center gap-1.5 rounded-md border border-warning/40 bg-surface-raised px-3 text-xs font-semibold text-warning-text disabled:opacity-50">{refinementBusy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />} Resume {refinement?.proposed_version ?? "refinement"}</button></div>}
      {reportError && <div className="border border-danger/20 bg-danger-light p-3 text-xs text-danger-text">{reportError}</div>}

      <div className="border-b border-edge">
        <div className="grid w-full min-w-0 grid-cols-5 gap-0 sm:flex sm:gap-1" role="tablist" aria-label="Scientific report views">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button key={tab.id} type="button" role="tab" aria-selected={activeTab === tab.id} data-ui-action={`report_view_${tab.id}`} data-ui-skip-action="true" onClick={() => setActiveTab(tab.id)} className={cn("flex h-9 min-w-0 items-center justify-center gap-1 border-b-2 px-1 text-2xs font-semibold sm:min-w-24 sm:gap-1.5 sm:px-3 sm:text-xs", activeTab === tab.id ? "border-accent text-accent" : "border-transparent text-ink-muted hover:text-ink-secondary")}>
                <Icon className="h-3.5 w-3.5" /> {tab.label}
              </button>
            );
          })}
        </div>
      </div>

      <div className="min-h-[620px]">
        {reportState === "loading" ? (
          <div className="flex min-h-[620px] items-center justify-center bg-surface-sunken text-xs text-ink-secondary"><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Loading reviewed report</div>
        ) : reportState === "generating" ? (
          <div className="flex min-h-[620px] items-center justify-center bg-surface-sunken">
            <div className="w-full max-w-sm space-y-3">
              {generationStages.map((stage, index) => {
                const completed = generation?.stage === "ready" || generationStageIndex > index;
                const active = generation?.stage === stage.id || (!generation && index === 0);
                return <div key={stage.id} className="flex items-center gap-3 text-xs text-ink-secondary"><span className={cn("flex h-6 w-6 items-center justify-center rounded-full border", completed ? "border-success/30 bg-success-light text-success" : active ? "border-info/30 bg-info-light text-info" : "border-edge text-ink-muted")}>{completed ? <Check className="h-3 w-3" /> : active ? <Loader2 className="h-3 w-3 animate-spin" /> : index + 1}</span>{stage.label}</div>;
              })}
              {generation?.status === "failed" && <div className="border-l-2 border-danger pl-3 text-xs text-danger">{generation.error ?? "Scientific report generation failed"}</div>}
            </div>
          </div>
        ) : activeTab === "report" ? (
          reportArtifact?.path ? <div className="bg-frame/55 p-2 sm:p-5"><iframe title="Interactive scientific report" src={artifactHref(reportArtifact.path)} sandbox="" referrerPolicy="no-referrer" className="mx-auto h-[760px] w-full border border-edge bg-surface-paper shadow-overlay" /></div> : <div className="flex min-h-[620px] items-center justify-center text-sm text-ink-muted">Scientific report is not ready. Generate it explicitly after the run passes review.</div>
        ) : activeTab === "figures" ? (
          <div className="grid gap-4 p-1 lg:grid-cols-2">
            {readyFigures.map((figure) => (
              <button key={figure.id} type="button" data-ui-action={`report_open_figure_${figure.id}`} data-ui-skip-action="true" onClick={() => setSelectedFigure(figure)} className="group text-left">
                <div className="aspect-[16/9] overflow-hidden border border-edge bg-surface-raised"><img src={figure.preview_data_url ?? artifactHref(figure.path)} alt={figure.title} referrerPolicy="no-referrer" className="h-full w-full object-contain" /></div>
                <div className="mt-2 flex items-start justify-between gap-3"><div><h3 className="text-sm font-semibold text-ink">{figure.title}</h3><p className="mt-0.5 text-xs text-ink-muted">{figure.caption}</p></div><ExternalLink className="mt-0.5 h-3.5 w-3.5 shrink-0 text-ink-muted group-hover:text-accent" /></div>
              </button>
            ))}
          </div>
        ) : activeTab === "methods" ? (
          <div className="grid gap-6 lg:grid-cols-2">
            <section><h2 className="border-b border-edge pb-2 text-sm font-semibold text-ink">Training method</h2><dl className="mt-2 divide-y divide-edge-light">{Object.entries(report?.method ?? {}).filter(([key]) => !["schema", "target_modules", "acceptance"].includes(key)).map(([key, value]) => <div key={key} className="grid grid-cols-[minmax(120px,0.7fr)_1fr] gap-4 py-2 text-xs"><dt className="text-ink-muted">{key.replaceAll("_", " ")}</dt><dd className="break-words font-mono text-ink-secondary">{typeof value === "object" ? JSON.stringify(value) : String(value)}</dd></div>)}</dl></section>
            <section><h2 className="border-b border-edge pb-2 text-sm font-semibold text-ink">Dataset contract</h2><dl className="mt-2 divide-y divide-edge-light">{Object.entries(report?.dataset ?? {}).filter(([key]) => ["method", "counts", "source_split_policy", "duplicate_record_ids", "duplicate_content_records", "grounded_context_contract", "secrets_removed"].includes(key)).map(([key, value]) => <div key={key} className="grid grid-cols-[minmax(120px,0.7fr)_1fr] gap-4 py-2 text-xs"><dt className="text-ink-muted">{key.replaceAll("_", " ")}</dt><dd className="break-words font-mono text-ink-secondary">{typeof value === "object" ? JSON.stringify(value) : String(value)}</dd></div>)}</dl></section>
          </div>
        ) : activeTab === "audit" ? (
          <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
            <section>
              <div className="flex items-center justify-between border-b border-edge pb-2"><h2 className="text-sm font-semibold text-ink">Independent Reviewer</h2><StatusPill value={String(report?.reviewer?.status ?? "not recorded")} /></div>
              <div className="mt-1">{reviewerChecks.map(([key, value]) => <EvidenceCheck key={key} label={key === "private_grader_not_executed" ? "private grader not executed at Independent Review time" : key} passed={value === true} />)}</div>
            </section>
            <section className="space-y-5">
              <div>
                <div className="flex items-center justify-between border-b border-edge pb-2"><h2 className="text-sm font-semibold text-ink">Claim Audit</h2><StatusPill value={String(report?.claim_audit?.status ?? "not recorded")} /></div>
                <p className="mt-3 text-xs leading-5 text-ink-secondary">{taskId === "siim-isic-melanoma-classification" ? "Claims are limited to independently reviewed, patient/content-grouped offline validation. No official rank, medal or clinical-use claim is inferred." : "Claims are limited to domain fine-tuning on a mature 7B base model. No publication or external benchmark claim is inferred."}</p>
              </div>
              {taskId === "siim-isic-melanoma-classification" && privateGrader ? (
                <div>
                  <div className="flex items-center justify-between border-b border-edge pb-2"><h2 className="text-sm font-semibold text-ink">Private grader terminal record</h2><StatusPill value={String(privateGrader.status ?? "not recorded")} /></div>
                  <dl className="mt-2 grid grid-cols-2 gap-2 text-xs text-ink-secondary">
                    <div><dt className="text-ink-muted">Execution count</dt><dd className="mt-0.5 font-mono font-semibold text-ink">{String(privateGrader.execution_count ?? "not recorded")}</dd></div>
                    <div><dt className="text-ink-muted">Score</dt><dd className="mt-0.5 font-mono font-semibold text-ink">{privateGrader.score == null ? "null" : String(privateGrader.score)}</dd></div>
                  </dl>
                  <p className="mt-2 text-xs leading-5 text-ink-muted">Recorded once after candidate freeze; failed closed and not reused for tuning.</p>
                </div>
              ) : null}
              <div><h2 className="border-b border-edge pb-2 text-sm font-semibold text-ink">Evidence scope</h2><div className="mt-3 space-y-2 text-xs text-ink-secondary">{(taskId === "siim-isic-melanoma-classification" ? ["Grouped OOF metrics and bootstrap interval", "Candidate freeze and once-only grader ledger", "Independent Review and Claim Audit", "Artifact hashes"] : ["Raw training telemetry", "Fixed-test metrics", "Adapter reload verification", "Artifact hashes"]).map((item) => <div key={item} className="flex items-center gap-2"><FileCheck2 className="h-3.5 w-3.5 text-success" /> {item}</div>)}</div></div>
            </section>
          </div>
        ) : (
          <div className="space-y-7">
            {(Object.keys(categoryMeta) as ScientificArtifact["category"][]).map((category) => {
              const meta = categoryMeta[category];
              const Icon = meta.icon;
              return (
                <section key={category}>
                  <div className="mb-2 flex items-end justify-between gap-3 border-b border-edge pb-2"><div><h2 className="flex items-center gap-2 text-sm font-semibold text-ink"><Icon className="h-4 w-4 text-ink-muted" />{meta.label}</h2><p className="mt-0.5 text-xs text-ink-muted">{meta.description}</p></div><span className="text-2xs text-ink-muted">{artifactsByCategory[category].filter((item) => item.status === "ready").length} ready</span></div>
                  <div className="grid gap-2 lg:grid-cols-2">{artifactsByCategory[category].map((artifact) => <ArtifactRow key={artifact.id} artifact={artifact} previewUrl={artifact.path ? artifactHref(artifact.path) : ""} downloadState={downloads[artifact.id] ?? "idle"} onDownload={() => void downloadArtifact(artifact)} />)}</div>
                </section>
              );
            })}
          </div>
        )}
      </div>

      {publicPresentation === false && <section className="border-t border-edge pt-5">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="max-w-2xl"><div className="flex items-center gap-2 text-sm font-semibold text-ink"><History className="h-4 w-4 text-accent" /> Continue with natural language</div><p className="mt-1 text-xs leading-5 text-ink-muted">Keep the model, dataset and reviewed evidence. EvoMind resolves only explicit changes and reruns affected steps after Human Gate approval.</p></div>
          {refinement && <div className="flex items-center gap-2"><StatusPill value={refinement.status} /><span className="text-xs font-semibold text-ink-secondary">{refinement.parent_version} preserved <ArrowRight className="inline h-3.5 w-3.5" /> {refinement.proposed_version}</span></div>}
        </div>
        <div className="mt-3 flex flex-col gap-2 sm:flex-row"><textarea value={refinementPrompt} onChange={(event) => setRefinementPrompt(event.target.value)} rows={3} disabled={activeRefinementPending} className="min-h-20 flex-1 resize-none rounded-md border border-edge bg-surface-raised px-3 py-2 text-sm leading-6 text-ink outline-none focus:border-accent disabled:cursor-not-allowed disabled:opacity-55" aria-label="Natural-language refinement request" /><button type="button" data-ui-action="report_analyze_refinement" data-ui-skip-action="true" onClick={() => void parseRequest()} disabled={refinementBusy || activeRefinementPending || !refinementPrompt.trim()} className="flex min-h-10 items-center justify-center gap-1.5 rounded-md bg-accent px-4 text-xs font-semibold text-accent-fg disabled:opacity-50 sm:self-end">{refinementBusy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Search className="h-3.5 w-3.5" />} {activeRefinementPending ? `${refinement?.proposed_version ?? "Refinement"} in progress` : "Analyze changes"}</button></div>
        {refinement && <RefinementSummary refinement={refinement} onReviewGate={() => setGateOpen(true)} />}
      </section>}

      {finalBundle && publicPresentation === false && <div className="sticky bottom-3 z-10 ml-auto flex w-fit items-center gap-3 rounded-md border border-edge bg-surface-raised px-3 py-2 shadow-lg"><div><div className="text-xs font-semibold text-ink">{showingPreservedParent ? `Preserved ${refinement?.parent_version ?? report?.version ?? "parent"} delivery` : "Final delivery ready"}</div><div className="text-2xs text-ink-muted">Report, model, reproducibility and audit bundle</div></div><button type="button" data-ui-action="report_download_final_bundle" data-ui-skip-action="true" onClick={() => void downloadArtifact(finalBundle)} className="flex h-8 items-center gap-1.5 rounded-md bg-accent px-3 text-xs font-semibold text-accent-fg">{downloads[finalBundle.id] === "preparing" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : downloads[finalBundle.id] === "downloaded" ? <Check className="h-3.5 w-3.5" /> : <Download className="h-3.5 w-3.5" />} {downloads[finalBundle.id] === "downloaded" ? "Downloaded" : showingPreservedParent ? `Download Preserved ${refinement?.parent_version ?? report?.version ?? "Parent"} Bundle` : "Download Final Bundle"}</button></div>}

      {selectedFigure && <div className="fixed inset-0 z-50 flex items-center justify-center bg-frame/80 p-5" role="dialog" aria-modal="true" aria-label={selectedFigure.title}><div className="flex h-[92vh] w-full max-w-6xl flex-col overflow-hidden border border-edge bg-surface-raised p-4 shadow-overlay"><div className="mb-3 flex items-start justify-between gap-4"><div><h2 className="text-base font-semibold text-ink">{selectedFigure.title}</h2><p className="mt-1 text-xs text-ink-muted">{selectedFigure.caption}</p></div><button type="button" title="Close figure" data-ui-action="report_close_figure" data-ui-skip-action="true" onClick={() => setSelectedFigure(null)} className="flex h-11 w-11 items-center justify-center rounded-md border border-edge text-ink-secondary transition hover:border-accent hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"><X className="h-4 w-4" /></button></div><img src={selectedFigure.preview_data_url ?? artifactHref(selectedFigure.path)} alt={selectedFigure.title} referrerPolicy="no-referrer" className="min-h-0 flex-1 object-contain" /></div></div>}
      {gateOpen && refinement && <HumanGate refinement={refinement} busy={refinementBusy} onCancel={closeGate} onDecision={decideGate} />}
    </div>
  );
}

function ArtifactRow({ artifact, previewUrl, downloadState, onDownload }: { artifact: ScientificArtifact; previewUrl: string; downloadState: DownloadState; onDownload: () => void }) {
  return (
    <div className="flex min-h-16 items-center gap-3 border border-edge px-3 py-2">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-surface-sunken text-ink-muted"><FileText className="h-4 w-4" /></div>
      <div className="min-w-0 flex-1"><div className="flex items-center gap-2"><span className="truncate text-xs font-semibold text-ink">{artifact.name}</span><StatusPill value={artifact.status} /></div><div className="mt-1 flex items-center gap-2 text-2xs text-ink-muted"><span>{artifact.type}</span><span>·</span><span>{artifact.version}</span><span>·</span><span>{formatBytes(artifact.bytes)}</span>{artifact.sha256 && <><span>·</span><span title={artifact.sha256}>{downloadState === "downloaded" ? "SHA256 verified" : "Manifest hash available"}</span></>}</div></div>
      <div className="flex shrink-0 items-center gap-1">{artifact.previewable && previewUrl && <a href={previewUrl} target="_blank" rel="noreferrer" title={`Preview ${artifact.name}`} data-ui-action={`report_preview_${artifact.id}`} data-ui-skip-action="true" className="flex h-8 w-8 items-center justify-center rounded-md border border-edge text-ink-muted hover:text-accent"><ExternalLink className="h-3.5 w-3.5" /></a>}<button type="button" data-ui-action={`report_download_${artifact.id}`} data-ui-skip-action="true" onClick={onDownload} disabled={!artifact.downloadable || downloadState === "preparing"} title={`Download ${artifact.name}`} className="flex h-8 w-8 items-center justify-center rounded-md border border-edge text-ink-muted hover:text-accent disabled:opacity-40">{downloadState === "preparing" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : downloadState === "downloaded" ? <Check className="h-3.5 w-3.5 text-success" /> : <Download className="h-3.5 w-3.5" />}</button></div>
    </div>
  );
}

function RefinementSummary({ refinement, onReviewGate }: { refinement: RefinementPlan; onReviewGate: () => void }) {
  return (
    <div className="mt-4 border-l-2 border-accent bg-surface-sunken px-4 py-3">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between"><div><div className="text-xs font-semibold uppercase text-accent">Requested refinement</div><div className="mt-1 text-sm font-semibold text-ink">Only affected steps will rerun</div></div>{refinement.status === "awaiting_human_gate" ? <button type="button" data-ui-action="report_review_human_gate" data-ui-skip-action="true" onClick={onReviewGate} className="flex h-8 items-center gap-1.5 rounded-md border border-accent px-3 text-xs font-semibold text-accent"><LockKeyhole className="h-3.5 w-3.5" /> Review Human Gate</button> : refinement.gate.decision === "approved" ? <div className="flex h-8 items-center gap-1.5 border border-success/30 bg-success-light px-3 text-xs font-semibold text-success"><Check className="h-3.5 w-3.5" /> Human Gate approved · recorded</div> : null}</div>
      <div className="mt-3 grid gap-3 md:grid-cols-2"><div><div className="text-2xs font-semibold uppercase text-ink-muted">Changed parameters</div><div className="mt-1 space-y-1">{refinement.requested_changes.map((change) => <div key={change.field} className="flex items-center gap-2 text-xs text-ink-secondary"><span className="font-mono">{change.field}</span><ChevronRight className="h-3 w-3 text-ink-faint" /><span className="font-mono font-semibold text-ink">{String(change.old_value ?? "recorded")}</span><ArrowRight className="h-3 w-3 text-ink-faint" /><span className="font-mono font-semibold text-accent">{String(change.value)}</span></div>)}</div></div><div><div className="text-2xs font-semibold uppercase text-ink-muted">Affected workflow</div><div className="mt-1 flex flex-wrap gap-1">{refinement.affected_steps.map((step) => <span key={step} className="rounded-sm border border-edge bg-surface-raised px-1.5 py-0.5 text-2xs text-ink-secondary">{step.replaceAll("_", " ")}</span>)}</div></div></div>
    </div>
  );
}

function HumanGate({ refinement, busy, onCancel, onDecision }: { refinement: RefinementPlan; busy: boolean; onCancel: () => void; onDecision: (decision: "approve" | "reject") => Promise<void> }) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const approveRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const previouslyFocused = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    approveRef.current?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onCancel();
        return;
      }
      if (event.key !== "Tab" || !dialogRef.current) return;
      const focusable = Array.from(dialogRef.current.querySelectorAll<HTMLElement>('button:not([disabled]), [href], input:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'));
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.body.style.overflow = previousOverflow;
      previouslyFocused?.focus();
    };
  }, [onCancel]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-frame/65 p-4 backdrop-blur-md" role="dialog" aria-modal="true" aria-labelledby="human-gate-title" aria-describedby="human-gate-description">
      <div ref={dialogRef} tabIndex={-1} className="w-full max-w-xl rounded-md border border-edge bg-surface-raised shadow-2xl">
        <div className="flex items-start justify-between gap-4 border-b border-edge px-5 py-4"><div className="flex gap-3"><div className="flex h-9 w-9 items-center justify-center rounded-md bg-warning-light text-warning"><LockKeyhole className="h-4.5 w-4.5" /></div><div><h2 id="human-gate-title" className="text-base font-semibold text-ink">Human Gate · Refinement approval</h2><p id="human-gate-description" className="mt-0.5 text-xs text-ink-muted">Run {refinement.parent_version} is preserved. A new {refinement.proposed_version} will be created.</p></div></div><button type="button" title="Close Human Gate" onClick={onCancel} className="flex h-8 w-8 items-center justify-center rounded-md border border-edge text-ink-muted"><X className="h-4 w-4" /></button></div>
        <div className="space-y-4 px-5 py-4"><div className="grid grid-cols-2 gap-x-5 gap-y-3">{refinement.requested_changes.map((change) => <div key={change.field}><div className="text-2xs font-semibold uppercase text-ink-muted">{change.field.replaceAll("_", " ")}</div><div className="mt-1 text-xs font-mono text-ink-secondary">{String(change.old_value ?? "recorded")} <ArrowRight className="mx-1 inline h-3 w-3" /> <strong className="text-accent">{String(change.value)}</strong></div></div>)}</div><div className="border-y border-edge py-3"><div className="flex items-center justify-between text-xs"><span className="text-ink-muted">Execution policy</span><span className="font-semibold text-ink">Only affected steps rerun</span></div><div className="mt-2 flex items-center justify-between text-xs"><span className="text-ink-muted">Compute</span><span className="font-semibold text-ink">Remote GPU only · local GPU disabled</span></div><div className="mt-2 flex items-center justify-between text-xs"><span className="text-ink-muted">Post-run controls</span><span className="font-semibold text-ink">Reviewer + Claim Audit + report update</span></div></div><div className="flex items-start gap-2 text-xs leading-5 text-ink-secondary"><ShieldCheck className="mt-0.5 h-3.5 w-3.5 shrink-0 text-success" />The original model, data split, audit and {refinement.parent_version} Adapter remain immutable. No publication action is included.</div></div>
        <div className="flex items-center justify-end gap-2 border-t border-edge px-5 py-3"><button type="button" data-ui-action="report_reject_refinement" data-ui-skip-action="true" disabled={busy} onClick={() => void onDecision("reject")} className="h-8 rounded-md border border-edge px-3 text-xs font-semibold text-ink-secondary">Reject</button><button ref={approveRef} type="button" data-ui-action="report_approve_refinement" data-ui-skip-action="true" disabled={busy} onClick={() => void onDecision("approve")} className="flex h-8 items-center gap-1.5 rounded-md bg-accent px-3 text-xs font-semibold text-accent-fg">{busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />} Approve refinement</button></div>
      </div>
    </div>
  );
}
