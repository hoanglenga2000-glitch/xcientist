import { execFile } from "node:child_process";
import crypto from "node:crypto";
import { createReadStream } from "node:fs";
import { promises as fs } from "node:fs";
import path from "node:path";
import { promisify } from "node:util";
import { resolveWorkspacePath, toRelativePath, workspaceRoot } from "@/lib/server/paths";
import { reviewedExistingReportEligible } from "@/lib/server/reviewed-existing-report";
import { classifyMissingScientificReport } from "@/lib/server/scientific-report-state";

const execFileAsync = promisify(execFile);
const RUN_ID_PATTERN = /^[A-Za-z0-9_-]{8,180}$/;
const TASK_ID_PATTERN = /^[A-Za-z0-9_-]{1,160}$/;

type JsonRecord = Record<string, unknown>;
type TelemetryRow = JsonRecord & {
  step?: number;
  loss?: number;
  eval_loss?: number;
  learning_rate?: number;
  cuda_max_allocated_mb?: number;
};

export type ScientificFigure = {
  id: string;
  title: string;
  caption: string;
  path: string;
  status: "ready" | "missing_data";
  sources: string[];
  preview_data_url?: string;
};

export type ScientificArtifact = {
  id: string;
  category: "research_report" | "model_artifacts" | "reproducibility" | "audit";
  name: string;
  type: string;
  version: string;
  status: "ready" | "unavailable";
  bytes: number | null;
  sha256: string | null;
  path: string | null;
  previewable: boolean;
  downloadable: boolean;
};

export type ScientificVersionComparison = {
  schema: "evomind.llm_version_comparison.v1";
  status: string;
  outcome: "improved" | "no_material_change" | "trade_off_detected" | "not_available";
  metric: "fixed_test_domain_composite";
  parent_version: string;
  child_version: string;
  v1: number | null;
  v2: number | null;
  delta_pp: number | null;
  requested_changes: Array<{
    field: string;
    operation: string;
    old_value: number | string | boolean | null;
    value: number | string | boolean | null;
    factor?: number;
    source: string;
  }>;
  parent_preserved: boolean;
  generated_by: "VersionComparatorAgent";
  generated_at: string;
};

export type ScientificReportPackage = {
  schema: "evomind.scientific_report_package.v1";
  task_id: string;
  run_id: string;
  version: string;
  parent_run_id: string | null;
  status: "ready" | "partial";
  renderer: "EvoMind Nature Skills";
  generated_at: string;
  source_evidence: string[];
  metrics: {
    before: number | null;
    after: number | null;
    improvement_pp: number | null;
    train_steps: number | null;
    train_loss: number | null;
    max_gpu_memory_mb: number | null;
  };
  dataset: JsonRecord;
  method: JsonRecord;
  reviewer: JsonRecord;
  claim_audit: JsonRecord;
  version_comparison: ScientificVersionComparison | null;
  figures: ScientificFigure[];
  artifacts: ScientificArtifact[];
  report_html_path: string;
  report_pdf_path: string | null;
  bundle_path: string | null;
  manifest_path: string;
};

export type ScientificReportGenerationStage =
  | "collecting_evidence"
  | "loading_metrics"
  | "rendering_figures"
  | "building_report"
  | "rendering_pdf"
  | "bundling"
  | "attaching_audit"
  | "ready"
  | "failed";

export type ScientificReportGenerationStatus = {
  schema: "evomind.scientific_report_generation.v1";
  task_id: string;
  run_id: string;
  seq: number;
  status: "running" | "ready" | "failed";
  stage: ScientificReportGenerationStage;
  automated: boolean;
  error: string | null;
  updated_at: string;
  history: Array<{
    seq: number;
    stage: ScientificReportGenerationStage;
    status: "running" | "ready" | "failed";
    at: string;
  }>;
};

export type ScientificReportLookupResult =
  | { status: "ready"; report: ScientificReportPackage }
  | { status: "pending_report"; runStatus: string; generation: ScientificReportGenerationStatus | null }
  | { status: "generation_failed"; runStatus: string; error: string; generation: ScientificReportGenerationStatus | null }
  | { status: "not_found" }
  | { status: "integrity_failed"; error: string };

type RunEvidence = {
  runId: string;
  runDir: string;
  runRelative: string;
  parentRunId: string | null;
  version: string;
  run: JsonRecord;
  metrics: JsonRecord;
  telemetry: TelemetryRow[];
  dataset: JsonRecord;
  config: JsonRecord;
  review: JsonRecord;
  claimAudit: JsonRecord;
  adapterReload: JsonRecord;
  evaluationSummary: JsonRecord;
  environment: JsonRecord;
  sourceArtifactManifest: JsonRecord;
  request: JsonRecord;
  versionComparison: JsonRecord;
  parentMetrics: JsonRecord;
  modelCard: string;
};

const reportRuntime = globalThis as unknown as {
  scientificReportJobs?: Map<string, Promise<ScientificReportPackage>>;
};
const scientificReportJobs = reportRuntime.scientificReportJobs ?? new Map<string, Promise<ScientificReportPackage>>();
reportRuntime.scientificReportJobs = scientificReportJobs;

function isRecord(value: unknown): value is JsonRecord {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function record(value: unknown): JsonRecord {
  return isRecord(value) ? value : {};
}

function number(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function sameNumber(left: unknown, right: unknown) {
  const a = number(left);
  const b = number(right);
  return a !== null && b !== null && Math.abs(a - b) <= 1e-9;
}

function reportVersionComparison(evidence: RunEvidence): ScientificVersionComparison | null {
  if (!evidence.parentRunId) return null;
  const comparison = evidence.versionComparison;
  const requestedChanges = Array.isArray(comparison.requested_changes)
    ? comparison.requested_changes.map(record).map((change) => ({
      field: text(change.field),
      operation: text(change.operation),
      old_value: typeof change.old_value === "number" || typeof change.old_value === "string" || typeof change.old_value === "boolean" || change.old_value === null ? change.old_value : null,
      value: typeof change.value === "number" || typeof change.value === "string" || typeof change.value === "boolean" || change.value === null ? change.value : null,
      ...(number(change.factor) === null ? {} : { factor: number(change.factor)! }),
      source: text(change.source),
    }))
    : [];
  const outcome = text(comparison.outcome);
  if (!["improved", "no_material_change", "trade_off_detected", "not_available"].includes(outcome)) {
    throw new Error(`run ${evidence.runId} has an invalid reviewed comparison outcome`);
  }
  return {
    schema: "evomind.llm_version_comparison.v1",
    status: text(comparison.status),
    outcome: outcome as ScientificVersionComparison["outcome"],
    metric: "fixed_test_domain_composite",
    parent_version: text(comparison.parent_version),
    child_version: text(comparison.child_version),
    v1: number(comparison.v1),
    v2: number(comparison.v2),
    delta_pp: number(comparison.delta_pp),
    requested_changes: requestedChanges,
    parent_preserved: comparison.parent_preserved === true,
    generated_by: "VersionComparatorAgent",
    generated_at: text(comparison.generated_at),
  };
}

function text(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function escapeHtml(value: unknown) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function escapeXml(value: unknown) {
  return escapeHtml(value).replaceAll("'", "&apos;");
}

async function readJson(filePath: string): Promise<JsonRecord> {
  try {
    const payload = JSON.parse((await fs.readFile(filePath, "utf-8")).replace(/^\uFEFF/, ""));
    return record(payload);
  } catch {
    return {};
  }
}

async function readJsonl(filePath: string): Promise<TelemetryRow[]> {
  const content = await fs.readFile(filePath, "utf-8").catch(() => "");
  return content
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .flatMap((line) => {
      try {
        const value = JSON.parse(line);
        return isRecord(value) ? [value as TelemetryRow] : [];
      } catch {
        return [];
      }
    });
}

async function sha256(filePath: string) {
  return new Promise<string>((resolve, reject) => {
    const hash = crypto.createHash("sha256");
    const input = createReadStream(filePath);
    input.on("data", (chunk) => hash.update(chunk));
    input.on("error", reject);
    input.on("end", () => resolve(hash.digest("hex")));
  });
}

async function writeJsonAtomic(filePath: string, payload: unknown) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const tempPath = `${filePath}.${process.pid}.${crypto.randomUUID()}.tmp`;
  await fs.writeFile(tempPath, JSON.stringify(payload, null, 2) + "\n", "utf-8");
  await fs.rename(tempPath, filePath).catch(async () => {
    await fs.rm(filePath, { force: true });
    await fs.rename(tempPath, filePath);
  });
}

function assertInsideWorkspace(target: string) {
  const resolved = path.resolve(target);
  const root = path.resolve(workspaceRoot);
  if (resolved !== root && !resolved.startsWith(root + path.sep)) {
    throw new Error("scientific report path escapes workspace");
  }
  return resolved;
}

function assertSafeId(value: string, label: "task_id" | "run_id") {
  const pattern = label === "task_id" ? TASK_ID_PATTERN : RUN_ID_PATTERN;
  if (!pattern.test(value)) throw new Error(`invalid ${label}`);
  return value;
}

function requireRecord(value: JsonRecord, label: string) {
  if (!Object.keys(value).length) throw new Error(`${label} is missing or invalid`);
}

function requirePassedChecks(payload: JsonRecord, label: string) {
  const checks = record(payload.checks);
  const values = Object.values(checks);
  if (!values.length || values.some((value) => value !== true)) {
    throw new Error(`${label} checks are incomplete or failed`);
  }
}

function matchesTask(taskId: string, request: JsonRecord, run: JsonRecord) {
  const explicitTask = text(request.task_id) || text(run.task_id);
  if (explicitTask) return explicitTask === taskId;
  if (taskId === "siim-isic-melanoma-classification" && text(request.dataset) === taskId) return true;
  return taskId === "evomind-qwen7b-finetune" && text(request.task_type) === "llm_finetune";
}

function isSiimIsicTask(taskId: string) {
  return taskId === "siim-isic-melanoma-classification";
}

async function artifactFromWorkspacePath(
  id: string,
  category: ScientificArtifact["category"],
  name: string,
  type: string,
  version: string,
  workspacePath: string,
  options: { previewable?: boolean; downloadable?: boolean } = {},
): Promise<ScientificArtifact> {
  const target = assertInsideWorkspace(resolveWorkspacePath(workspacePath));
  const stat = await fs.stat(target).catch(() => null);
  if (!stat?.isFile()) {
    return {
      id,
      category,
      name,
      type,
      version,
      status: "unavailable",
      bytes: null,
      sha256: null,
      path: null,
      previewable: false,
      downloadable: false,
    };
  }
  return {
    id,
    category,
    name,
    type,
    version,
    status: "ready",
    bytes: stat.size,
    sha256: await sha256(target),
    path: workspacePath.replaceAll("\\", "/"),
    previewable: options.previewable === true,
    downloadable: options.downloadable !== false,
  };
}

async function buildSiimExistingReport(taskId: string, runId: string): Promise<ScientificReportPackage | null> {
  if (!isSiimIsicTask(taskId)) return null;
  assertSafeId(taskId, "task_id");
  assertSafeId(runId, "run_id");
  const runDir = assertInsideWorkspace(resolveWorkspacePath(path.join("workspace", "evomind_runs", runId)));
  const stat = await fs.stat(runDir).catch(() => null);
  if (!stat?.isDirectory()) return null;
  const [
    run,
    request,
    metrics,
    datasetProfile,
    researchDesign,
    review,
    claimAudit,
    artifactManifest,
    deliverables,
    experimentComparison,
    candidateFreeze,
    privateGrader,
  ] = await Promise.all([
    readJson(path.join(runDir, "run.json")),
    readJson(path.join(runDir, "request.json")),
    readJson(path.join(runDir, "metrics.json")),
    readJson(path.join(runDir, "dataset_profile.json")),
    readJson(path.join(runDir, "research_design.json")),
    readJson(path.join(runDir, "review.json")),
    readJson(path.join(runDir, "claim_audit.json")),
    readJson(path.join(runDir, "artifact_manifest.json")),
    readJson(path.join(runDir, "deliverables.json")),
    readJson(path.join(runDir, "experiment_comparison.json")),
    readJson(path.join(runDir, "candidate_freeze.json")),
    readJson(path.join(runDir, "private_grader.json")),
  ]);
  if (text(run.run_id) !== runId || !matchesTask(taskId, request, run)) return null;
  if (text(run.status) !== "completed" || text(artifactManifest.status) !== "verified") return null;
  const reviewStatus = text(review.status);
  const claimStatus = text(claimAudit.status);
  if (reviewStatus !== "review_passed" && reviewStatus !== "passed") return null;
  if (claimStatus !== "passed") return null;

  const version = "SIIM-A800";
  const base = "workspace/evomind_runs/" + runId;
  const artifacts = await Promise.all([
    artifactFromWorkspacePath("report-html", "research_report", "Interactive Research Report", "HTML", version, base + "/delivery/research_report.html", { previewable: true, downloadable: false }),
    artifactFromWorkspacePath("report-pdf", "research_report", "Scientific Report PDF", "PDF", version, base + "/delivery/evomind-siim-isic-report.pdf", { previewable: true }),
    artifactFromWorkspacePath("results-csv", "research_report", "Results CSV", "CSV", version, base + "/delivery/evomind-siim-isic-results.csv"),
    artifactFromWorkspacePath("code-zip", "reproducibility", "Code Package", "ZIP", version, base + "/delivery/evomind-siim-isic-code.zip"),
    artifactFromWorkspacePath("final-bundle", "reproducibility", "Evidence Package", "ZIP", version, base + "/delivery/evomind-siim-isic-evidence.zip"),
    artifactFromWorkspacePath("artifact-manifest", "audit", "Artifact Manifest", "JSON", version, base + "/artifact_manifest.json", { previewable: true }),
    artifactFromWorkspacePath("deliverables", "audit", "Deliverables Manifest", "JSON", version, base + "/deliverables.json", { previewable: true }),
    artifactFromWorkspacePath("review", "audit", "Independent Review", "JSON", version, base + "/review.json", { previewable: true }),
    artifactFromWorkspacePath("claim-audit", "audit", "Claim Audit", "JSON", version, base + "/claim_audit.json", { previewable: true }),
    artifactFromWorkspacePath("candidate-freeze", "audit", "Candidate Freeze", "JSON", version, base + "/candidate_freeze.json", { previewable: true }),
    artifactFromWorkspacePath("private-grader", "audit", "Private Grader Once-Only Ledger", "JSON", version, base + "/private_grader.json", { previewable: true }),
  ]);
  const figureCandidates = await Promise.all(Array.from({ length: 9 }, (_, index) => {
    const page = String(index + 1).padStart(2, "0");
    const figurePath = base + "/delivery/qa/report-page-" + page + ".png";
    return fs.stat(resolveWorkspacePath(figurePath)).then((pageStat) => pageStat.isFile() ? {
      id: "report-page-" + page,
      title: "Report page " + page,
      caption: "Rendered PDF QA page from the existing SIIM-ISIC delivery package.",
      path: figurePath,
      status: "ready" as const,
      sources: [base + "/delivery/evomind-siim-isic-report.pdf"],
    } : null).catch(() => null);
  }));
  const figures: ScientificFigure[] = figureCandidates.flatMap((item) => item ? [item] : []);
  const rocAuc = number(metrics.roc_auc);
  const after = rocAuc === null ? null : rocAuc * 100;
  const reviewer = {
    ...review,
    raw_status: reviewStatus,
    status: "passed",
  };
  return {
    schema: "evomind.scientific_report_package.v1",
    task_id: taskId,
    run_id: runId,
    version,
    parent_run_id: null,
    status: "ready",
    renderer: "EvoMind Nature Skills",
    generated_at: new Date().toISOString(),
    source_evidence: [
      base + "/metrics.json",
      base + "/dataset_profile.json",
      base + "/research_design.json",
      base + "/review.json",
      base + "/claim_audit.json",
      base + "/artifact_manifest.json",
      base + "/deliverables.json",
    ],
    metrics: {
      before: null,
      after,
      improvement_pp: null,
      train_steps: null,
      train_loss: null,
      max_gpu_memory_mb: number(metrics.max_cuda_memory_mb),
    },
    dataset: {
      ...datasetProfile,
      task_id: taskId,
      dataset: text(datasetProfile.dataset) || taskId,
      metrics: {
        roc_auc: rocAuc,
        pr_auc: number(metrics.pr_auc),
        brier: number(metrics.brier),
        fold_roc_auc_mean: number(metrics.fold_roc_auc_mean),
        fold_roc_auc_std: number(metrics.fold_roc_auc_std),
        bootstrap_95ci: record(metrics.patient_grouped_bootstrap_roc_auc_95ci),
      },
    },
    method: {
      ...researchDesign,
      selected_profile: text(experimentComparison.selected_profile),
      candidate_freeze: {
        status: text(candidateFreeze.status),
        sha256: text(claimAudit.candidate_freeze_sha256),
      },
      private_grader: {
        status: text(privateGrader.status),
        score: privateGrader.score ?? null,
        execution_count: 1,
        failed_closed: text(privateGrader.status) === "failed_closed",
      },
      deliverables,
    },
    reviewer,
    claim_audit: claimAudit,
    version_comparison: null,
    figures,
    artifacts,
    report_html_path: artifacts.find((item) => item.id === "report-html")?.path ?? "",
    report_pdf_path: artifacts.find((item) => item.id === "report-pdf")?.path ?? null,
    bundle_path: artifacts.find((item) => item.id === "final-bundle")?.path ?? null,
    manifest_path: base + "/artifact_manifest.json",
  };
}

async function buildReviewedExistingReport(taskId: string, runId: string): Promise<ScientificReportPackage | null> {
  assertSafeId(taskId, "task_id");
  assertSafeId(runId, "run_id");
  const runDir = assertInsideWorkspace(resolveWorkspacePath(path.join("workspace", "evomind_runs", runId)));
  const [run, metrics, review, artifactManifest, pointer, dataAudit, researchContext, hpcReceipt] = await Promise.all([
    readJson(path.join(runDir, "run.json")),
    readJson(path.join(runDir, "metrics.json")),
    readJson(path.join(runDir, "review.json")),
    readJson(path.join(runDir, "artifact_manifest.json")),
    readJson(resolveWorkspacePath("workspace/current_run.json")),
    readJson(path.join(runDir, "data_audit.json")),
    readJson(path.join(runDir, "research_context.json")),
    readJson(path.join(runDir, "hpc_job_receipt.json")),
  ]);
  if (!reviewedExistingReportEligible({ taskId, runId, run, review, artifactManifest, pointer })) return null;
  try {
    await verifyArtifactManifest(runDir, artifactManifest, [
      "research_report.md",
      "metrics.json",
      "review.json",
      "submission.csv",
      "hpc_job_receipt.json",
    ]);
  } catch {
    return null;
  }

  const selectedSolution = text(artifactManifest.selected_solution) || text(metrics.selected_solution);
  const version = `Reviewed-${selectedSolution || "run"}`;
  const base = `workspace/evomind_runs/${runId}`;
  const artifacts = await Promise.all([
    artifactFromWorkspacePath("report-html", "research_report", "Reviewed Research Report", "Markdown", version, `${base}/research_report.md`, { previewable: true, downloadable: false }),
    artifactFromWorkspacePath("metrics", "research_report", "Aggregate Metrics", "JSON", version, `${base}/metrics.json`, { previewable: true }),
    artifactFromWorkspacePath("submission", "research_report", "Candidate Submission", "CSV", version, `${base}/submission.csv`),
    artifactFromWorkspacePath("artifact-manifest", "audit", "Artifact Manifest", "JSON", version, `${base}/artifact_manifest.json`, { previewable: true }),
    artifactFromWorkspacePath("review", "audit", "Independent Review and Claim Audit", "JSON", version, `${base}/review.json`, { previewable: true }),
    artifactFromWorkspacePath("hpc-receipt", "reproducibility", "HPC Job Receipt", "JSON", version, `${base}/hpc_job_receipt.json`, { previewable: true }),
  ]);
  const reportArtifact = artifacts.find((item) => item.id === "report-html");
  if (reportArtifact?.status !== "ready" || !reportArtifact.path) return null;

  const candidates = Array.isArray(metrics.candidates) ? metrics.candidates.map(record) : [];
  const selectedScore = number(metrics.cv_score);
  const baselineScore = number(candidates[0]?.cv_score);
  const before = baselineScore === null ? null : baselineScore * 100;
  const after = selectedScore === null ? null : selectedScore * 100;
  const claimAudit = record(review.claim_audit);
  return {
    schema: "evomind.scientific_report_package.v1",
    task_id: taskId,
    run_id: runId,
    version,
    parent_run_id: null,
    status: "ready",
    renderer: "EvoMind Nature Skills",
    generated_at: text(review.generated_at) || new Date().toISOString(),
    source_evidence: [
      `${base}/research_report.md`,
      `${base}/metrics.json`,
      `${base}/review.json`,
      `${base}/artifact_manifest.json`,
      `${base}/hpc_job_receipt.json`,
    ],
    metrics: {
      before,
      after,
      improvement_pp: before === null || after === null ? null : after - before,
      train_steps: null,
      train_loss: null,
      max_gpu_memory_mb: null,
    },
    dataset: { ...dataAudit, task_id: taskId },
    method: {
      ...researchContext,
      metric: text(metrics.metric),
      selected_solution: selectedSolution,
      candidates,
      hpc_job: hpcReceipt,
      official_kaggle_score: metrics.official_kaggle_score ?? null,
    },
    reviewer: { ...review, status: "passed" },
    claim_audit: claimAudit,
    version_comparison: null,
    figures: [],
    artifacts,
    report_html_path: reportArtifact.path,
    report_pdf_path: null,
    bundle_path: null,
    manifest_path: `${base}/artifact_manifest.json`,
  };
}

function resolveRunArtifact(runDir: string, relativePath: string) {
  if (!relativePath || path.isAbsolute(relativePath) || relativePath.split(/[\\/]+/).includes("..")) {
    throw new Error(`artifact manifest contains an unsafe path: ${relativePath || "<empty>"}`);
  }
  const target = path.resolve(runDir, ...relativePath.split("/"));
  const root = path.resolve(runDir);
  if (target === root || !target.startsWith(root + path.sep)) {
    throw new Error(`artifact manifest path escapes run directory: ${relativePath}`);
  }
  return target;
}

async function verifyArtifactManifest(runDir: string, manifest: JsonRecord, requiredPaths: string[]) {
  const rawArtifacts = manifest.artifacts;
  if (!Array.isArray(rawArtifacts) || !rawArtifacts.length) {
    throw new Error("source artifact manifest has no artifacts");
  }
  const entries = rawArtifacts.map((value, index) => {
    const item = record(value);
    const relativePath = text(item.path).replaceAll("\\", "/");
    const expectedHash = text(item.sha256).toLowerCase();
    const expectedBytes = number(item.bytes);
    if (!relativePath || !/^[a-f0-9]{64}$/.test(expectedHash) || expectedBytes === null || expectedBytes < 0) {
      throw new Error(`source artifact manifest entry ${index} is incomplete`);
    }
    return { relativePath, expectedHash, expectedBytes };
  });
  const uniquePaths = new Set(entries.map((entry) => entry.relativePath));
  if (uniquePaths.size !== entries.length) throw new Error("source artifact manifest contains duplicate paths");

  for (const required of requiredPaths) {
    if (!uniquePaths.has(required)) throw new Error(`source artifact manifest is missing ${required}`);
  }

  const rootRealPath = await fs.realpath(runDir);
  for (const entry of entries) {
    const target = resolveRunArtifact(runDir, entry.relativePath);
    const stat = await fs.stat(target).catch(() => null);
    if (!stat?.isFile() || stat.size !== entry.expectedBytes) {
      throw new Error(`source artifact size verification failed: ${entry.relativePath}`);
    }
    const targetRealPath = await fs.realpath(target);
    if (!targetRealPath.startsWith(rootRealPath + path.sep)) {
      throw new Error(`source artifact resolves outside run directory: ${entry.relativePath}`);
    }
    if (await sha256(target) !== entry.expectedHash) {
      throw new Error(`source artifact hash verification failed: ${entry.relativePath}`);
    }
  }
  return entries;
}

async function verifySourceArtifactManifest(runDir: string, manifest: JsonRecord) {
  return verifyArtifactManifest(runDir, manifest, [
    "claim_audit.json",
    "data/dataset_manifest.json",
    "evaluation_summary.json",
    "llm_output/adapter/adapter_config.json",
    "llm_output/adapter/adapter_model.safetensors",
    "llm_output/adapter_reload.json",
    "llm_output/environment.json",
    "llm_output/metrics.json",
    "llm_output/telemetry.jsonl",
    "model_card.md",
    "qlora_config.json",
    "review.json"
  ]);
  // The source manifest cannot hash itself without becoming recursive; its
  // presence was checked before verifying every artifact it declares.
}

async function validateEvidence(runId: string, runDir: string, taskId: string): Promise<RunEvidence> {
  const [run, request, metrics, dataset, config, review, claimAudit, adapterReload, evaluationSummary, environment, sourceArtifactManifest, versionRecord, versionComparison] = await Promise.all([
    readJson(path.join(runDir, "run.json")),
    readJson(path.join(runDir, "request.json")),
    readJson(path.join(runDir, "llm_output", "metrics.json")),
    readJson(path.join(runDir, "data", "dataset_manifest.json")),
    readJson(path.join(runDir, "qlora_config.json")),
    readJson(path.join(runDir, "review.json")),
    readJson(path.join(runDir, "claim_audit.json")),
    readJson(path.join(runDir, "llm_output", "adapter_reload.json")),
    readJson(path.join(runDir, "evaluation_summary.json")),
    readJson(path.join(runDir, "llm_output", "environment.json")),
    readJson(path.join(runDir, "artifact_manifest.json")),
    readJson(path.join(runDir, "version.json")),
    readJson(path.join(runDir, "version_comparison.json"))
  ]);
  for (const [label, payload] of Object.entries({ run, request, metrics, dataset, config, review, claimAudit, adapterReload, evaluationSummary, environment, sourceArtifactManifest })) {
    requireRecord(payload, label);
  }
  if (!matchesTask(taskId, request, run)) throw new Error(`run ${runId} does not belong to task ${taskId}`);
  if (text(run.run_id) !== runId || text(metrics.run_id) !== runId || text(environment.run_id) !== runId || text(sourceArtifactManifest.run_id) !== runId) {
    throw new Error(`run identity evidence does not match ${runId}`);
  }
  if (run.status !== "completed") throw new Error(`run ${runId} is not completed`);

  const gates = record(run.gates);
  if (gates.hpc_execution !== "passed" || gates.reviewer !== "passed" || gates.claim_audit !== "passed" || gates.adapter_reload !== "passed" || gates.model_publication !== "blocked") {
    throw new Error(`run ${runId} has incomplete release gates`);
  }
  if (review.status !== "passed" || review.parent_subjective_summary_received !== false) {
    throw new Error(`run ${runId} did not pass an independent review`);
  }
  requirePassedChecks(review, "Independent Reviewer");
  if (claimAudit.status !== "passed") throw new Error(`run ${runId} did not pass Claim Audit`);
  requirePassedChecks(claimAudit, "Claim Audit");
  if (adapterReload.passed !== true || evaluationSummary.status !== "passed") {
    throw new Error(`run ${runId} did not pass adapter reload and evaluation checks`);
  }
  if (metrics.local_gpu_used !== false || environment.local_gpu_used !== false || record(review.checks).local_gpu_unused !== true) {
    throw new Error(`run ${runId} does not prove local GPU remained unused`);
  }

  const computePolicy = record(request.compute_policy);
  const submissionPolicy = record(request.submission_policy);
  const negativeConstraints = Array.isArray(request.negative_constraints) ? request.negative_constraints : [];
  if (computePolicy.local_gpu_allowed !== false || computePolicy.remote_gpu_required !== true || computePolicy.backend !== "hpc" || !negativeConstraints.includes("no_local_gpu")) {
    throw new Error(`run ${runId} violates the remote-only compute contract`);
  }
  if (metrics.model_published !== false || sourceArtifactManifest.model_publication !== "blocked" || submissionPolicy.model_publication !== "forbidden" || !negativeConstraints.includes("no_model_publication")) {
    throw new Error(`run ${runId} does not prove model publication remained blocked`);
  }
  if (sourceArtifactManifest.review_status !== "passed" || sourceArtifactManifest.claim_audit_status !== "passed") {
    throw new Error(`run ${runId} source artifact manifest is not approved`);
  }

  const adapterConfig = path.join(runDir, "llm_output", "adapter", "adapter_config.json");
  const adapterModel = path.join(runDir, "llm_output", "adapter", "adapter_model.safetensors");
  const [adapterConfigHash, adapterModelHash] = await Promise.all([sha256(adapterConfig), sha256(adapterModel)]);
  if (text(adapterReload.adapter_config_sha256) !== adapterConfigHash || text(adapterReload.adapter_model_sha256) !== adapterModelHash) {
    throw new Error(`run ${runId} adapter reload hashes do not match the delivered Adapter`);
  }
  const parentRunId = text(versionRecord.parent_run_id) || text(run.parent_run_id) || null;
  const sourceManifestEntries = await verifySourceArtifactManifest(runDir, sourceArtifactManifest);
  const version = text(versionRecord.version) || (parentRunId ? "V2" : "V1");
  let refinement: JsonRecord = {};
  if (parentRunId) {
    assertSafeId(parentRunId, "run_id");
    if (parentRunId === runId) throw new Error("refinement parent run cannot equal child run");
    const refinementEvidence = await Promise.all([
      readJson(path.join(runDir, "refinement.json")),
      readJson(path.join(runDir, "human_gate.json"))
    ]);
    [refinement] = refinementEvidence;
    const humanGate = refinementEvidence[1];
    for (const [label, payload] of Object.entries({ versionRecord, versionComparison, refinement, humanGate })) requireRecord(payload, label);
    if (versionRecord.parent_preserved !== true || text(versionComparison.parent_run_id) !== parentRunId || text(versionComparison.child_run_id) !== runId || humanGate.decision !== "approved") {
      throw new Error(`run ${runId} has an incomplete refinement version contract`);
    }
    if (!sourceManifestEntries.some((entry) => entry.relativePath === "version_comparison.json")) {
      throw new Error(`run ${runId} source artifact manifest is missing version_comparison.json`);
    }
  }
  const parentMetrics = parentRunId
    ? await readJson(path.join(assertInsideWorkspace(path.join(path.dirname(runDir), parentRunId)), "llm_output", "metrics.json"))
    : {};
  if (parentRunId) requireRecord(parentMetrics, "parentMetrics");
  if (parentRunId) {
    const comparisonPath = path.join(runDir, "version_comparison.json");
    const parentMetricsPath = path.join(path.dirname(runDir), parentRunId, "llm_output", "metrics.json");
    const childMetricsPath = path.join(runDir, "llm_output", "metrics.json");
    const parentScore = number(record(parentMetrics.after).domain_composite);
    const childScore = number(record(metrics.after).domain_composite);
    if (parentScore === null || childScore === null) throw new Error(`run ${runId} has incomplete version comparison metrics`);
    const delta = childScore - parentScore;
    const outcome = delta > 0.25 ? "improved" : delta < -1.0 ? "trade_off_detected" : "no_material_change";
    const comparisonHash = await sha256(comparisonPath);
    const comparisonStat = await fs.stat(comparisonPath);
    const sourceEntry = sourceManifestEntries.find((entry) => entry.relativePath === "version_comparison.json");
    const reviewedArtifacts = Array.isArray(review.reviewed_artifacts) ? review.reviewed_artifacts.map(record) : [];
    const reviewedComparison = reviewedArtifacts.find((item) => text(item.path).replaceAll("\\", "/") === "version_comparison.json");
    const comparisonChecks = record(review.checks);
    const comparisonMatches =
      versionComparison.schema === "evomind.llm_version_comparison.v1"
      && text(versionComparison.parent_run_id) === parentRunId
      && text(versionComparison.child_run_id) === runId
      && text(versionComparison.parent_version) === text(refinement.parent_version)
      && text(versionComparison.child_version) === version
      && text(versionComparison.metric) === "fixed_test_domain_composite"
      && sameNumber(versionComparison.v1, parentScore)
      && sameNumber(versionComparison.v2, childScore)
      && sameNumber(versionComparison.delta_pp, delta)
      && text(versionComparison.outcome) === outcome
      && text(versionComparison.parent_metrics_sha256) === await sha256(parentMetricsPath)
      && text(versionComparison.child_metrics_sha256) === await sha256(childMetricsPath)
      && sourceEntry?.expectedHash === comparisonHash
      && sourceEntry?.expectedBytes === comparisonStat.size
      && text(reviewedComparison?.sha256) === comparisonHash
      && number(reviewedComparison?.bytes) === comparisonStat.size
      && comparisonChecks.version_comparison_recomputed === true
      && comparisonChecks.parent_run_preserved === true;
    if (!comparisonMatches) {
      throw new Error(`run ${runId} version comparison is not bound to metrics, Reviewer and source manifest evidence`);
    }
  }
  const modelCard = await fs.readFile(path.join(runDir, "model_card.md"), "utf-8").catch(() => "");
  if (!modelCard.trim()) throw new Error(`run ${runId} model card is missing`);
  const telemetry = await readJsonl(path.join(runDir, "llm_output", "telemetry.jsonl"));
  if (!telemetry.length) throw new Error(`run ${runId} training telemetry is missing`);
  return {
    runId,
    runDir,
    runRelative: toRelativePath(runDir)!.replaceAll("\\", "/"),
    parentRunId,
    version,
    run,
    metrics,
    telemetry,
    dataset,
    config,
    review,
    claimAudit,
    adapterReload,
    evaluationSummary,
    environment,
    sourceArtifactManifest,
    request,
    versionComparison,
    parentMetrics,
    modelCard
  };
}

async function findRun(taskId: string, requestedRunId?: string | null): Promise<RunEvidence> {
  assertSafeId(taskId, "task_id");
  const runsRoot = assertInsideWorkspace(resolveWorkspacePath("workspace/evomind_runs"));
  const pointer = await readJson(resolveWorkspacePath("workspace/current_run.json"));
  const pointerRun = text(pointer.run_id);
  const pointerTask = text(pointer.task_id);
  let runId = requestedRunId?.trim() || "";
  if (!runId) {
    if (pointerTask !== taskId || !pointerRun) {
      throw new Error(`current run pointer does not identify task ${taskId}`);
    }
    runId = pointerRun;
  }
  assertSafeId(runId, "run_id");
  const runDir = assertInsideWorkspace(path.join(runsRoot, runId));
  const stat = await fs.stat(runDir).catch(() => null);
  if (!stat?.isDirectory()) throw new Error(`run ${runId} was not found`);
  return validateEvidence(runId, runDir, taskId);
}

function svgDocument(width: number, height: number, title: string, body: string) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeXml(title)}">
  <rect width="${width}" height="${height}" fill="#fffefb"/>
  <style>
    .title{font:700 22px Arial,sans-serif;fill:#111827}.label{font:600 13px Arial,sans-serif;fill:#374151}.tick{font:11px Arial,sans-serif;fill:#6b7280}.value{font:700 14px Arial,sans-serif;fill:#111827}.panel{font:700 15px Arial,sans-serif;fill:#111827}.note{font:12px Arial,sans-serif;fill:#6b7280}.axis{stroke:#9ca3af;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.line{fill:none;stroke:#0f766e;stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round}.line2{fill:none;stroke:#2563eb;stroke-width:2.2;stroke-linecap:round;stroke-linejoin:round}.point{fill:#0f766e;stroke:#fff;stroke-width:1.5}.point2{fill:#2563eb;stroke:#fff;stroke-width:1.5}
  </style>
  <text class="title" x="38" y="38">${escapeXml(title)}</text>
  ${body}
  </svg>`;
}

function performanceFigure(before: number | null, after: number | null) {
  if (before === null || after === null) return null;
  const max = Math.max(before, after, 1) * 1.12;
  const bars = [
    { label: "Baseline", value: before, color: "#64748b" },
    { label: "Domain QLoRA", value: after, color: "#0f766e" }
  ].map((item, index) => {
    const height = (item.value / max) * 280;
    const x = 150 + index * 300;
    const y = 390 - height;
    return `<rect x="${x}" y="${y.toFixed(1)}" width="160" height="${height.toFixed(1)}" fill="${item.color}"/>
      <text class="value" x="${x + 80}" y="${(y - 12).toFixed(1)}" text-anchor="middle">${item.value.toFixed(2)}</text>
      <text class="label" x="${x + 80}" y="418" text-anchor="middle">${item.label}</text>`;
  }).join("\n");
  return svgDocument(760, 470, "Figure 1a | Fixed-test domain composite", `
    <line class="axis" x1="90" y1="390" x2="690" y2="390"/><line class="axis" x1="90" y1="90" x2="90" y2="390"/>
    <text class="tick" x="54" y="394">0</text><text class="tick" x="38" y="98">${max.toFixed(0)}</text>
    ${bars}<text class="note" x="90" y="450">Values are read directly from llm_output/metrics.json; higher is better.</text>`);
}

type PlotPoint = { x: number; y: number };
type PlotBox = { x: number; y: number; w: number; h: number };
type PlotDomain = { minX: number; maxX: number; minY: number; maxY: number };

function plotDomain(series: PlotPoint[][]): PlotDomain {
  const points = series.flat();
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const observedMaxY = Math.max(...ys, 0);
  return {
    minX,
    maxX,
    minY: 0,
    maxY: Math.max(observedMaxY * 1.08, 1e-9),
  };
}

function plotCoordinates(rows: PlotPoint[], domain: PlotDomain, box: PlotBox) {
  const xRange = Math.max(domain.maxX - domain.minX, 1);
  const yRange = Math.max(domain.maxY - domain.minY, 1e-9);
  return rows.map((row) => ({
    x: box.x + ((row.x - domain.minX) / xRange) * box.w,
    y: box.y + box.h - ((row.y - domain.minY) / yRange) * box.h,
  }));
}

function formatAxisValue(value: number) {
  const absolute = Math.abs(value);
  if (absolute > 0 && absolute < 0.001) return value.toExponential(1);
  if (absolute >= 1000) return Math.round(value).toLocaleString("en-US");
  if (absolute >= 10) return value.toFixed(0);
  if (absolute >= 1) return value.toFixed(1);
  return value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}

function chartAxes(domain: PlotDomain, box: PlotBox) {
  const yFractions = [0, 0.5, 1];
  const yTicks = yFractions.map((fraction) => {
    const y = box.y + box.h - fraction * box.h;
    const value = domain.minY + fraction * (domain.maxY - domain.minY);
    return `<line class="grid" x1="${box.x}" y1="${y.toFixed(1)}" x2="${box.x + box.w}" y2="${y.toFixed(1)}"/><text class="tick" x="${box.x - 8}" y="${(y + 4).toFixed(1)}" text-anchor="end">${formatAxisValue(value)}</text>`;
  }).join("");
  return `${yTicks}<line class="axis" x1="${box.x}" y1="${box.y + box.h}" x2="${box.x + box.w}" y2="${box.y + box.h}"/><line class="axis" x1="${box.x}" y1="${box.y}" x2="${box.x}" y2="${box.y + box.h}"/><text class="tick" x="${box.x}" y="${box.y + box.h + 18}">${formatAxisValue(domain.minX)}</text><text class="tick" x="${box.x + box.w}" y="${box.y + box.h + 18}" text-anchor="end">${formatAxisValue(domain.maxX)}</text><text class="tick" x="${box.x + box.w / 2}" y="${box.y + box.h + 18}" text-anchor="middle">training step</text>`;
}

function plotSeries(rows: PlotPoint[], domain: PlotDomain, box: PlotBox, className: "line" | "line2", pointsOnly = false) {
  if (!rows.length) return "";
  const coordinates = plotCoordinates(rows, domain, box);
  const polyline = pointsOnly ? "" : `<polyline class="${className}" points="${coordinates.map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" ")}"/>`;
  const pointClass = className === "line" ? "point" : "point2";
  const points = coordinates.map((point) => `<circle class="${pointClass}" cx="${point.x.toFixed(1)}" cy="${point.y.toFixed(1)}" r="${pointsOnly ? 4.2 : 2.4}"/>`).join("");
  return `${polyline}${points}`;
}

function trainingBehaviourFigure(telemetry: TelemetryRow[]) {
  const stepRows = telemetry.filter((row) => number(row.step) !== null);
  const trainLoss = stepRows.flatMap((row) => number(row.loss) === null ? [] : [{ x: number(row.step)!, y: number(row.loss)! }]);
  const validationLoss = stepRows.flatMap((row) => number(row.eval_loss) === null ? [] : [{ x: number(row.step)!, y: number(row.eval_loss)! }]);
  const learningRate = stepRows.flatMap((row) => number(row.learning_rate) === null ? [] : [{ x: number(row.step)!, y: number(row.learning_rate)! }]);
  const gpuMemory = stepRows.flatMap((row) => number(row.cuda_max_allocated_mb) === null ? [] : [{ x: number(row.step)!, y: number(row.cuda_max_allocated_mb)! }]);
  if (!trainLoss.length && !validationLoss.length && !learningRate.length && !gpuMemory.length) return null;

  const panels = [
    { key: "a", title: "Loss", primary: trainLoss, secondary: validationLoss, primaryLabel: "training", secondaryLabel: "validation observations", primaryPointsOnly: false, secondaryPointsOnly: true },
    { key: "b", title: "Learning rate", primary: learningRate, secondary: [], primaryLabel: "schedule", secondaryLabel: "", primaryPointsOnly: false, secondaryPointsOnly: false },
    { key: "c", title: "CUDA memory (MB)", primary: gpuMemory, secondary: [], primaryLabel: "max allocated", secondaryLabel: "", primaryPointsOnly: false, secondaryPointsOnly: false },
    { key: "d", title: "Validation loss observations", primary: validationLoss, secondary: [], primaryLabel: "observed eval loss", secondaryLabel: "", primaryPointsOnly: true, secondaryPointsOnly: false }
  ];
  const body = panels.map((panel, index) => {
    const col = index % 2;
    const row = Math.floor(index / 2);
    const x = 45 + col * 575;
    const y = 70 + row * 310;
    const box = { x: x + 72, y: y + 54, w: 445, h: 180 };
    const missing = !panel.primary.length && !panel.secondary.length;
    const domain = missing ? null : plotDomain([panel.primary, panel.secondary]);
    const primary = domain ? plotSeries(panel.primary, domain, box, "line", panel.primaryPointsOnly) : "";
    const secondary = domain ? plotSeries(panel.secondary, domain, box, "line2", panel.secondaryPointsOnly) : "";
    const legend = missing ? "" : `<line class="line" x1="${x + 300}" y1="${y + 23}" x2="${x + 322}" y2="${y + 23}"/><text class="tick" x="${x + 328}" y="${y + 27}">${escapeXml(panel.primaryLabel)}</text>${panel.secondary.length ? `<circle class="point2" cx="${x + 430}" cy="${y + 23}" r="4"/><text class="tick" x="${x + 440}" y="${y + 27}">${escapeXml(panel.secondaryLabel)}</text>` : ""}`;
    return `<rect x="${x}" y="${y}" width="535" height="275" fill="#fff" stroke="#d1d5db"/>
      <text class="panel" x="${x + 16}" y="${y + 27}">${panel.key}</text><text class="label" x="${x + 42}" y="${y + 27}">${escapeXml(panel.title)}</text>
      ${legend}${domain ? chartAxes(domain, box) : ""}${primary}${secondary}
      ${missing ? `<text class="note" x="${x + 267}" y="${y + 150}" text-anchor="middle">Not recorded in this run</text>` : ""}
      `;
  }).join("\n");
  return svgDocument(1200, 710, "Figure 1 | Model performance and training behaviour", `${body}<text class="note" x="45" y="690">Traces and discrete observations are reconstructed from telemetry.jsonl on shared per-panel axes; no values are interpolated.</text>`);
}

function datasetFigure(dataset: JsonRecord) {
  const counts = record(dataset.counts);
  const rows = ["train", "validation", "test"].flatMap((key) => {
    const value = number(counts[key]);
    return value === null ? [] : [{ key, value }];
  });
  if (!rows.length) return null;
  const max = Math.max(...rows.map((row) => row.value), 1);
  const body = rows.map((row, index) => {
    const y = 118 + index * 90;
    const width = (row.value / max) * 600;
    return `<text class="label" x="70" y="${y + 25}">${row.key}</text><rect x="180" y="${y}" width="${width.toFixed(1)}" height="36" fill="#0f766e"/><text class="value" x="${Math.min(840, 195 + width)}" y="${y + 25}">${row.value}</text>`;
  }).join("\n");
  return svgDocument(920, 440, "Figure 2 | Source-isolated dataset composition", `${body}<text class="note" x="70" y="405">Counts and split policy are read from data/dataset_manifest.json.</text>`);
}

function versionComparisonFigure(comparison: JsonRecord) {
  const v1 = number(comparison.v1);
  const v2 = number(comparison.v2);
  if (v1 === null || v2 === null) return null;
  const max = Math.max(v1, v2, 1);
  const bars = [
    { label: text(comparison.parent_version) || "V1", value: v1, color: "#64748b" },
    { label: text(comparison.child_version) || "V2", value: v2, color: "#0f766e" },
  ].map((item, index) => {
    const x = 180 + index * 330;
    const height = (item.value / max) * 230;
    const y = 335 - height;
    return `<rect x="${x}" y="${y.toFixed(1)}" width="180" height="${height.toFixed(1)}" fill="${item.color}"/><text class="value" x="${x + 90}" y="${Math.max(80, y - 12).toFixed(1)}" text-anchor="middle">${item.value.toFixed(2)}</text><text class="label" x="${x + 90}" y="370" text-anchor="middle">${escapeXml(item.label)}</text>`;
  }).join("\n");
  const delta = number(comparison.delta_pp);
  return svgDocument(900, 460, "Figure 3 | Version comparison", `${bars}<line class="axis" x1="110" y1="336" x2="790" y2="336"/><text class="note" x="110" y="420">Outcome: ${escapeXml(comparison.outcome || "not recorded")} · Delta: ${delta === null ? "not recorded" : `${delta >= 0 ? "+" : ""}${delta.toFixed(2)} pp`}</text>`);
}

async function writeFigure(targetDir: string, id: string, svg: string | null, title: string, caption: string, sources: string[]): Promise<ScientificFigure> {
  const target = path.join(targetDir, `${id}.svg`);
  if (!svg) {
    return { id, title, caption, path: "", status: "missing_data", sources };
  }
  await fs.writeFile(target, svg, "utf-8");
  return { id, title, caption, path: toRelativePath(target)!.replaceAll("\\", "/"), status: "ready", sources };
}

function metricRows(metrics: JsonRecord) {
  const before = record(metrics.before);
  const after = record(metrics.after);
  return [
    ["Domain composite", number(before.domain_composite), number(after.domain_composite)],
    ["Fixed-test loss", number(before.loss), number(after.loss)],
    ["Keyword recall", number(before.keyword_recall), number(after.keyword_recall)],
    ["Format compliance", number(before.format_compliance), number(after.format_compliance)],
    ["Sensitive leak rate", number(before.sensitive_leak_rate), number(after.sensitive_leak_rate)]
  ].map(([label, beforeValue, afterValue]) => ({ label: String(label), before: beforeValue as number | null, after: afterValue as number | null }));
}

function formatNumber(value: number | null, digits = 4) {
  return value === null ? "Not recorded" : value.toFixed(digits);
}

async function reportHtml(evidence: RunEvidence, figures: ScientificFigure[]) {
  const before = number(record(evidence.metrics.before).domain_composite);
  const after = number(record(evidence.metrics.after).domain_composite);
  const improvement = number(evidence.metrics.improvement_pp);
  const training = record(evidence.metrics.training);
  const counts = record(evidence.dataset.counts);
  const reviewChecks = record(evidence.review.checks);
  const comparison = evidence.versionComparison;
  const comparisonDelta = number(comparison.delta_pp);
  const reviewRows = Object.entries(reviewChecks).map(([key, value]) => `<tr><td>${escapeHtml(key.replaceAll("_", " "))}</td><td class="${value === true ? "pass" : "fail"}">${value === true ? "Passed" : "Not passed"}</td></tr>`).join("");
  const metricsRows = metricRows(evidence.metrics).map((row) => `<tr><td>${escapeHtml(row.label)}</td><td>${formatNumber(row.before)}</td><td>${formatNumber(row.after)}</td></tr>`).join("");
  const figureHtml = await Promise.all(figures.filter((figure) => figure.status === "ready").map(async (figure, index) => {
    const absolute = resolveWorkspacePath(figure.path);
    const svg = await fs.readFile(absolute, "utf-8");
    return `<figure id="figure-${index + 1}">${svg.replace(/^<\?xml[^>]*>/, "")}<figcaption><strong>${escapeHtml(figure.title)}</strong> ${escapeHtml(figure.caption)}</figcaption></figure>`;
  }));
  return `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EvoMind 7B Domain Fine-tuning Scientific Report</title>
<style>
:root{--ink:#151719;--muted:#60656b;--rule:#cfd4d8;--teal:#0f766e;--paper:#fffefb;--wash:#f4f5f2;--pass:#166534;--fail:#9f1239}*{box-sizing:border-box}body{margin:0;background:#e9ecea;color:var(--ink);font-family:Georgia,"Noto Serif SC","Songti SC",serif;line-height:1.62}.report{width:min(1180px,calc(100% - 40px));margin:32px auto;background:var(--paper);box-shadow:0 12px 40px rgba(20,30,24,.12)}header{padding:68px 72px 46px;border-top:8px solid var(--teal);border-bottom:1px solid var(--rule)}.kicker{font:700 12px/1.2 Arial,sans-serif;letter-spacing:.12em;text-transform:uppercase;color:var(--teal)}h1{font-size:42px;line-height:1.12;margin:18px 0 20px;max-width:900px;letter-spacing:0}.dek{font-size:19px;color:#3f454a;max-width:820px}.meta{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:18px;margin-top:32px;padding-top:20px;border-top:1px solid var(--rule);font:13px/1.45 Arial,sans-serif}.meta b{display:block;color:#111;margin-bottom:3px}.report-nav{position:sticky;top:0;z-index:2;display:flex;gap:4px;padding:10px 72px;background:rgba(255,254,251,.97);border-bottom:1px solid var(--rule);font:600 12px Arial,sans-serif}.report-nav a{color:#4b5563;text-decoration:none;padding:7px 10px}.report-nav a:hover{color:var(--teal)}main{padding:42px 72px 72px}.grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:28px 48px}.wide{grid-column:1/-1}section{break-inside:avoid}h2{font-size:25px;line-height:1.2;margin:10px 0 16px;padding-bottom:8px;border-bottom:2px solid #202426}h3{font-size:17px;margin:22px 0 8px}.lead{font-size:18px}.result-band{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--rule);border:1px solid var(--rule);margin:24px 0}.result-band div{background:#fff;padding:18px}.result-band span{display:block;font:12px Arial,sans-serif;color:var(--muted);margin-bottom:6px}.result-band strong{font:700 27px Arial,sans-serif}table{width:100%;border-collapse:collapse;font:13px/1.4 Arial,sans-serif;margin:12px 0 20px}th{border-top:2px solid #1f2937;border-bottom:1px solid #1f2937;text-align:left;padding:8px 7px}td{border-bottom:1px solid #d8dcdf;padding:8px 7px;vertical-align:top}.pass{color:var(--pass);font-weight:700}.fail{color:var(--fail);font-weight:700}figure{margin:30px 0 40px;border-top:1px solid var(--rule);padding-top:18px}figure svg{display:block;width:100%;height:auto;background:#fff}figcaption{font-size:13px;color:#3f454a;margin-top:12px}.evidence{font:12px/1.55 "Cascadia Mono",Consolas,monospace;background:var(--wash);padding:14px 16px;border-left:3px solid var(--teal);overflow-wrap:anywhere}.notice{border:1px solid var(--rule);background:#f8faf8;padding:16px 18px;font-size:14px}.footer{padding:24px 72px;border-top:1px solid var(--rule);font:12px Arial,sans-serif;color:var(--muted);display:flex;justify-content:space-between}@media(max-width:820px){header{padding:42px 24px 30px}main{padding:30px 24px 48px}.report-nav{padding:8px 16px;flex-wrap:wrap;overflow:visible;white-space:normal}.report-nav a{flex:1 1 auto;min-width:56px;text-align:center}.grid,.meta,.result-band{grid-template-columns:1fr}.report{width:100%;margin:0;box-shadow:none}h1{font-size:30px;overflow-wrap:anywhere}.dek{font-size:17px}.meta{gap:12px}th,td{overflow-wrap:anywhere}.footer{padding:20px 24px;flex-direction:column;gap:4px}}@media print{body{background:#fff}.report{width:100%;margin:0;box-shadow:none}.report-nav{display:none}main{padding-top:28px}figure{break-inside:avoid}.footer{position:relative}}
</style></head><body><article class="report"><header><div class="kicker">EvoMind Scientific Report · Nature Skills renderer</div><h1>面向科研工作流的成熟 7B 模型领域微调</h1><p class="dek">基于来源隔离的 EvoMind 文档数据、4-bit QLoRA 训练、固定测试集评估、Independent Reviewer 与 Claim Audit 的可审计报告。</p><div class="meta"><div><b>Base model</b>${escapeHtml(evidence.metrics.base_model)}</div><div><b>Run version</b>${escapeHtml(evidence.version)}</div><div><b>Review</b>${escapeHtml(evidence.review.status || "not recorded")}</div><div><b>Publication</b>Blocked by policy</div></div></header>
<nav class="report-nav"><a href="#summary">Report</a><a href="#figures">Figures</a><a href="#methods">Methods</a><a href="#audit">Audit</a><a href="#files">Files</a></nav>
<main><div class="grid"><section id="summary" class="wide"><h2>研究摘要</h2><p class="lead">本研究不是从零训练基础模型，而是在成熟的 7B 指令模型上进行领域适配。数据、训练、评估和审核均绑定到同一运行账本。</p><div class="result-band"><div><span>Baseline domain composite</span><strong>${formatNumber(before, 2)}</strong></div><div><span>After domain QLoRA</span><strong>${formatNumber(after, 2)}</strong></div><div><span>Absolute improvement</span><strong>${improvement === null ? "Not recorded" : `+${improvement.toFixed(2)} pp`}</strong></div></div></section>
<section><h2>研究目标</h2><p>使用本地 EvoMind 文档构造带来源哈希的中文科研工作流指令数据，在远程 GPU 上完成领域微调，并交付可重载 Adapter、模型卡、评估、审核和复现材料。</p></section>
<section><h2>关键结论</h2><p>固定测试集记录的领域综合分数从 ${formatNumber(before, 2)} 变为 ${formatNumber(after, 2)}。该结果只描述当前内部固定测试集，不代表外部排行榜或官方基准成绩。</p><div class="notice">Independent Reviewer: <strong>${escapeHtml(evidence.review.status || "not recorded")}</strong> · Claim Audit: <strong>${escapeHtml(evidence.claimAudit.status || "not recorded")}</strong></div></section>
${evidence.parentRunId ? `<section class="wide"><h2>版本比较</h2><p>${escapeHtml(evidence.version)} 在保留上一版本、数据集和固定测试集的前提下完成增量 refinement。比较结论为 <strong>${escapeHtml(comparison.outcome || "not recorded")}</strong>${comparisonDelta === null ? "" : `，固定测试集变化为 <strong>${comparisonDelta >= 0 ? "+" : ""}${comparisonDelta.toFixed(2)} pp</strong>`}。系统不会把无实质变化或 trade-off 描述为提升。</p></section>` : ""}
<section id="figures" class="wide"><h2>实验结果与图表</h2>${figureHtml.join("\n") || "<p>当前运行没有足够的真实数值生成图表。</p>"}</section>
<section id="methods"><h2>方法与训练设置</h2><table><thead><tr><th>Field</th><th>Recorded value</th></tr></thead><tbody><tr><td>Method</td><td>${escapeHtml(evidence.config.method || evidence.metrics.training_method)}</td></tr><tr><td>LoRA rank / alpha</td><td>${escapeHtml(evidence.config.lora_r)} / ${escapeHtml(evidence.config.lora_alpha)}</td></tr><tr><td>Dropout</td><td>${escapeHtml(evidence.config.lora_dropout)}</td></tr><tr><td>Effective batch</td><td>${escapeHtml(evidence.config.effective_batch_size)}</td></tr><tr><td>Epochs</td><td>${escapeHtml(evidence.config.epochs)}</td></tr><tr><td>Learning rate</td><td>${escapeHtml(evidence.config.learning_rate)}</td></tr><tr><td>Train steps</td><td>${escapeHtml(training.steps)}</td></tr></tbody></table></section>
<section><h2>数据与许可说明</h2><table><thead><tr><th>Split</th><th>Records</th></tr></thead><tbody><tr><td>Train</td><td>${escapeHtml(counts.train)}</td></tr><tr><td>Validation</td><td>${escapeHtml(counts.validation)}</td></tr><tr><td>Fixed test</td><td>${escapeHtml(counts.test)}</td></tr></tbody></table><p>Split policy: ${escapeHtml(evidence.dataset.source_split_policy)}. Source overlap and duplicate checks are retained in the dataset manifest. This report does not infer a license that was not recorded by the source documents.</p></section>
<section class="wide"><h2>结果表</h2><table><thead><tr><th>Metric</th><th>Before</th><th>After</th></tr></thead><tbody>${metricsRows}</tbody></table></section>
<section><h2>局限性</h2><ul><li>结果来自单次固定测试集运行，未记录重复实验，因此不展示误差棒、置信区间或显著性检验。</li><li>内部领域综合分数不等同于外部官方基准。</li><li>模型发布保持关闭；产物只在当前工作区交付。</li></ul></section>
<section><h2>可复现性</h2><p>报告绑定数据、代码、配置、训练遥测、Adapter、评估和审核文件的 SHA256。Adapter reload 检查与本地 GPU 禁用状态由 Reviewer 直接核验。</p><div class="evidence">version=${escapeHtml(evidence.version)}<br>data_hash=${escapeHtml(evidence.metrics.data_hash)}<br>code_hash=${escapeHtml(evidence.metrics.code_hash)}<br>config_hash=${escapeHtml(evidence.metrics.config_hash)}</div></section>
<section id="audit" class="wide"><h2>Independent Reviewer 与审计附录</h2><table><thead><tr><th>Check</th><th>Status</th></tr></thead><tbody>${reviewRows || "<tr><td colspan=\"2\">Review checks not recorded</td></tr>"}</tbody></table><p>Reviewer input is restricted to raw dataset hashes, training logs, metrics, telemetry, Adapter artifacts and environment evidence; the parent Agent's subjective summary is excluded.</p></section>
<section id="files" class="wide"><h2>产物与版本信息</h2><p>正式交付物由 Artifact Center 分类呈现，包括 Scientific Report、Figures、Adapter、配置、运行 Manifest、环境快照、Independent Review、Claim Audit 和 Human Gate 记录。文件哈希保存在报告包 manifest。</p></section>
</div></main><footer class="footer"><span>EvoMind Scientific Report</span><span>Generated from reviewed run evidence</span></footer></article></body></html>`;
}

async function browserExecutable() {
  const candidates = [
    process.env.EVOMIND_BROWSER_PATH,
    process.env.BROWSER_PATH,
    process.platform === "win32" ? "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe" : null,
    process.platform === "win32" ? "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe" : null,
    process.platform === "win32" ? "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" : null,
    process.platform === "win32" ? path.join(process.env.LOCALAPPDATA ?? "", "Google", "Chrome", "Application", "chrome.exe") : null,
    process.platform !== "win32" ? "/usr/bin/google-chrome" : null,
    process.platform !== "win32" ? "/usr/bin/chromium" : null
  ].filter((candidate): candidate is string => Boolean(candidate));
  for (const candidate of candidates) {
    if (await fs.stat(candidate).then((stat) => stat.isFile()).catch(() => false)) return candidate;
  }
  return null;
}

async function renderPdf(htmlPath: string, pdfPath: string) {
  const executable = await browserExecutable();
  if (!executable) return false;
  const tempProfile = path.join(
    process.env.TEMP ?? process.env.TMP ?? path.dirname(pdfPath),
    `evomind-report-${process.pid}-${crypto.randomUUID()}`
  );
  await fs.mkdir(tempProfile, { recursive: true });
  try {
    await execFileAsync(executable, [
      "--headless=new",
      "--disable-gpu",
      "--no-pdf-header-footer",
      `--user-data-dir=${tempProfile}`,
      `--print-to-pdf=${pdfPath}`,
      new URL(`file:///${htmlPath.replaceAll("\\", "/")}`).href
    ], { timeout: 120000, windowsHide: true, maxBuffer: 4 * 1024 * 1024 }).catch(() => null);
    return await fs.stat(pdfPath).then((stat) => stat.size > 1024).catch(() => false);
  } finally {
    await fs.rm(tempProfile, { recursive: true, force: true }).catch(() => undefined);
  }
}

async function linkOrCopy(source: string, target: string) {
  await fs.mkdir(path.dirname(target), { recursive: true });
  await fs.link(source, target).catch(async () => {
    await fs.copyFile(source, target);
  });
}

async function bundleFiles(root: string) {
  const entries: string[] = [];
  async function visit(current: string) {
    const children = await fs.readdir(current, { withFileTypes: true });
    for (const child of children) {
      const target = path.join(current, child.name);
      if (child.isDirectory()) await visit(target);
      else if (child.isFile()) entries.push(target);
    }
  }
  await visit(root);
  return entries.sort((left, right) => left.localeCompare(right));
}

async function copyBundleSource(runDir: string, staging: string, sourceRelative: string, targetRelative: string, required = true) {
  const source = resolveRunArtifact(runDir, sourceRelative);
  const stat = await fs.stat(source).catch(() => null);
  if (!stat?.isFile()) {
    if (required) throw new Error(`required bundle artifact is missing: ${sourceRelative}`);
    return;
  }
  await linkOrCopy(source, path.join(staging, ...targetRelative.split("/")));
}

async function buildBundle(evidence: RunEvidence, reportRoot: string, htmlPath: string, pdfPath: string | null, figures: ScientificFigure[]) {
  const staging = path.join(reportRoot, "bundle-staging");
  await fs.rm(staging, { recursive: true, force: true });
  await fs.mkdir(staging, { recursive: true });
  const bundlePath = path.join(reportRoot, "final-delivery-bundle.zip");
  try {
    await linkOrCopy(htmlPath, path.join(staging, "research-report", "scientific-report.html"));
    if (!pdfPath) throw new Error("scientific report PDF is required for the final delivery bundle");
    await linkOrCopy(pdfPath, path.join(staging, "research-report", "scientific-report.pdf"));
    for (const figure of figures.filter((item) => item.status === "ready")) {
      await linkOrCopy(resolveWorkspacePath(figure.path), path.join(staging, "research-report", "figures", path.basename(figure.path)));
    }

    const sourceFiles = [
      ["llm_output/adapter/adapter_model.safetensors", "model/adapter/adapter_model.safetensors", true],
      ["llm_output/adapter/adapter_config.json", "model/adapter/adapter_config.json", true],
      ["llm_output/adapter/tokenizer.json", "model/adapter/tokenizer.json", true],
      ["llm_output/adapter/tokenizer_config.json", "model/adapter/tokenizer_config.json", true],
      ["llm_output/adapter/special_tokens_map.json", "model/adapter/special_tokens_map.json", true],
      ["llm_output/adapter/added_tokens.json", "model/adapter/added_tokens.json", true],
      ["llm_output/adapter/merges.txt", "model/adapter/merges.txt", true],
      ["llm_output/adapter/vocab.json", "model/adapter/vocab.json", true],
      ["llm_output/adapter/README.md", "model/adapter/README.md", true],
      ["model_card.md", "model/model_card.md", true],
      ["run.json", "reproducibility/run.json", true],
      ["request.json", "reproducibility/request.json", true],
      ["task_graph.json", "reproducibility/task_graph.json", true],
      ["events.jsonl", "reproducibility/events.jsonl", true],
      ["messages.jsonl", "reproducibility/messages.jsonl", true],
      ["handoffs.jsonl", "reproducibility/handoffs.jsonl", true],
      ["qlora_config.json", "reproducibility/qlora_config.json", true],
      ["code_manifest.json", "reproducibility/code_manifest.json", true],
      ["code/train_qlora.py", "reproducibility/code/train_qlora.py", true],
      ["code/requirements.lock", "reproducibility/code/requirements.lock", false],
      ["data/dataset_manifest.json", "reproducibility/data/dataset_manifest.json", true],
      ["evaluation_summary.json", "reproducibility/evaluation_summary.json", true],
      ["llm_output/adapter_reload.json", "reproducibility/adapter_reload.json", true],
      ["llm_output/metrics.json", "reproducibility/metrics.json", true],
      ["llm_output/telemetry.jsonl", "reproducibility/telemetry.jsonl", true],
      ["llm_output/training.log", "reproducibility/training.log", true],
      ["llm_output/environment.json", "reproducibility/environment.json", true],
      ["artifact_manifest.json", "reproducibility/source-artifact-manifest.json", true],
      ["review.json", "audit/independent-review.json", true],
      ["claim_audit.json", "audit/claim-audit.json", true],
      ["refinement.json", "versions/refinement.json", Boolean(evidence.parentRunId)],
      ["human_gate.json", "versions/human-gate.json", Boolean(evidence.parentRunId)],
      ["version.json", "versions/version.json", Boolean(evidence.parentRunId)],
      ["version_comparison.json", "versions/version-comparison.json", Boolean(evidence.parentRunId)]
    ] as const;
    for (const [sourceRelative, targetRelative, required] of sourceFiles) {
      await copyBundleSource(evidence.runDir, staging, sourceRelative, targetRelative, required);
    }

    const payloadFiles = await bundleFiles(staging);
    const payloadEntries = await Promise.all(payloadFiles.map(async (filePath) => ({
      path: path.relative(staging, filePath).replaceAll("\\", "/"),
      bytes: (await fs.stat(filePath)).size,
      sha256: await sha256(filePath)
    })));
    const manifestDir = path.join(staging, "manifest");
    await fs.mkdir(manifestDir, { recursive: true });
    const bundleManifestPath = path.join(manifestDir, "bundle-manifest.json");
    await fs.writeFile(bundleManifestPath, JSON.stringify({
      schema: "evomind.final_delivery_bundle.v1",
      run_id: evidence.runId,
      version: evidence.version,
      parent_run_id: evidence.parentRunId,
      generated_at: new Date().toISOString(),
      hash_scope: "payload files only; bundle-manifest.json and SHA256SUMS.txt are excluded to prevent recursive self-hashing",
      files: payloadEntries
    }, null, 2) + "\n", "utf-8");
    const manifestHash = await sha256(bundleManifestPath);
    const checksumLines = [
      ...payloadEntries.map((entry) => `${entry.sha256}  ${entry.path}`),
      `${manifestHash}  manifest/bundle-manifest.json`
    ].sort();
    await fs.writeFile(path.join(manifestDir, "SHA256SUMS.txt"), checksumLines.join("\n") + "\n", "utf-8");

    await fs.rm(bundlePath, { force: true });
    const tar = process.platform === "win32" ? "tar.exe" : "tar";
    await execFileAsync(tar, ["-a", "-c", "-f", bundlePath, "-C", staging, "."], { timeout: 240000, windowsHide: true, maxBuffer: 4 * 1024 * 1024 });
    const valid = await fs.stat(bundlePath).then((stat) => stat.size > 1024).catch(() => false);
    return valid ? bundlePath : null;
  } finally {
    await fs.rm(staging, { recursive: true, force: true });
  }
}

async function artifact(id: string, category: ScientificArtifact["category"], name: string, type: string, version: string, filePath: string | null, previewable: boolean): Promise<ScientificArtifact> {
  if (!filePath || !await fs.stat(filePath).then((stat) => stat.isFile()).catch(() => false)) {
    return { id, category, name, type, version, status: "unavailable", bytes: null, sha256: null, path: null, previewable: false, downloadable: false };
  }
  const stat = await fs.stat(filePath);
  return {
    id,
    category,
    name,
    type,
    version,
    status: "ready",
    bytes: stat.size,
    sha256: await sha256(filePath),
    path: toRelativePath(filePath)!.replaceAll("\\", "/"),
    previewable,
    downloadable: true
  };
}

function generationStatusPath(taskId: string, runId: string) {
  assertSafeId(taskId, "task_id");
  assertSafeId(runId, "run_id");
  return assertInsideWorkspace(resolveWorkspacePath(path.join(
    "workspace",
    "tasks",
    taskId,
    "reports",
    "scientific",
    runId,
    "generation-status.json",
  )));
}

export async function readScientificReportGenerationStatus(taskId: string, runId: string): Promise<ScientificReportGenerationStatus | null> {
  const payload = await readJson(generationStatusPath(taskId, runId));
  return payload.schema === "evomind.scientific_report_generation.v1"
    && payload.task_id === taskId
    && payload.run_id === runId
    ? payload as ScientificReportGenerationStatus
    : null;
}

async function updateGenerationStatus(
  taskId: string,
  runId: string,
  stage: ScientificReportGenerationStage,
  status: "running" | "ready" | "failed",
  automated: boolean,
  error: string | null = null,
) {
  const filePath = generationStatusPath(taskId, runId);
  const previous = await readScientificReportGenerationStatus(taskId, runId);
  const seq = (previous?.seq ?? 0) + 1;
  const now = new Date().toISOString();
  const history = [...(previous?.history ?? []), { seq, stage, status, at: now }].slice(-64);
  const payload: ScientificReportGenerationStatus = {
    schema: "evomind.scientific_report_generation.v1",
    task_id: taskId,
    run_id: runId,
    seq,
    status,
    stage,
    automated,
    error,
    updated_at: now,
    history,
  };
  await writeJsonAtomic(filePath, payload);
  return payload;
}

async function generateScientificReportUnlocked(taskId: string, requestedRunId: string, automated: boolean): Promise<ScientificReportPackage> {
  const evidence = await findRun(taskId, requestedRunId);
  await updateGenerationStatus(taskId, evidence.runId, "loading_metrics", "running", automated);
  const reportRoot = assertInsideWorkspace(resolveWorkspacePath(path.join("workspace", "tasks", taskId, "reports", "scientific", evidence.runId)));
  const figuresDir = path.join(reportRoot, "figures");
  await fs.mkdir(figuresDir, { recursive: true });

  const before = number(record(evidence.metrics.before).domain_composite);
  const after = number(record(evidence.metrics.after).domain_composite);
  await updateGenerationStatus(taskId, evidence.runId, "rendering_figures", "running", automated);
  const figures: ScientificFigure[] = [
    await writeFigure(figuresDir, "figure-1a-performance", performanceFigure(before, after), "Figure 1a | Fixed-test model performance", "Baseline and domain-adapted scores from the same fixed test set.", [`${evidence.runRelative}/llm_output/metrics.json`]),
    await writeFigure(figuresDir, "figure-1-training", trainingBehaviourFigure(evidence.telemetry), "Figure 1 | Model performance and training behaviour", "Training loss, validation observations, learning-rate schedule and CUDA memory reconstructed from the run ledger.", [`${evidence.runRelative}/llm_output/telemetry.jsonl`]),
    await writeFigure(figuresDir, "figure-2-dataset", datasetFigure(evidence.dataset), "Figure 2 | Dataset composition", "Source-isolated train, validation and fixed-test record counts.", [`${evidence.runRelative}/data/dataset_manifest.json`])
  ];
  if (evidence.parentRunId) {
    figures.push(await writeFigure(
      figuresDir,
      "figure-3-version-comparison",
      versionComparisonFigure(evidence.versionComparison),
      `Figure 3 | Run ${text(evidence.versionComparison.parent_version) || "parent"} and Run ${text(evidence.versionComparison.child_version) || evidence.version} comparison`,
      "Fixed-test comparison from the preserved parent and independently reviewed child run.",
      [`${evidence.runRelative}/version_comparison.json`],
    ));
  }

  const htmlPath = path.join(reportRoot, "scientific-report.html");
  const sourcePath = path.join(reportRoot, "report-source.md");
  await updateGenerationStatus(taskId, evidence.runId, "building_report", "running", automated);
  await fs.writeFile(htmlPath, await reportHtml(evidence, figures), "utf-8");
  await fs.writeFile(sourcePath, `# EvoMind Scientific Report Source\n\n- Run: ${evidence.runId}\n- Version: ${evidence.version}\n- Renderer: EvoMind Nature Skills\n- Final representation: scientific-report.html / scientific-report.pdf\n`, "utf-8");
  const pdfCandidate = path.join(reportRoot, "scientific-report.pdf");
  await updateGenerationStatus(taskId, evidence.runId, "rendering_pdf", "running", automated);
  const pdfPath = await renderPdf(htmlPath, pdfCandidate) ? pdfCandidate : null;
  await updateGenerationStatus(taskId, evidence.runId, "bundling", "running", automated);
  const bundlePath = await buildBundle(evidence, reportRoot, htmlPath, pdfPath, figures);

  await updateGenerationStatus(taskId, evidence.runId, "attaching_audit", "running", automated);
  const runArtifacts = {
    adapter: path.join(evidence.runDir, "llm_output", "adapter", "adapter_model.safetensors"),
    adapterConfig: path.join(evidence.runDir, "llm_output", "adapter", "adapter_config.json"),
    tokenizer: path.join(evidence.runDir, "llm_output", "adapter", "tokenizer.json"),
    adapterReadme: path.join(evidence.runDir, "llm_output", "adapter", "README.md"),
    modelCard: path.join(evidence.runDir, "model_card.md"),
    run: path.join(evidence.runDir, "run.json"),
    config: path.join(evidence.runDir, "qlora_config.json"),
    environment: path.join(evidence.runDir, "llm_output", "environment.json"),
    metrics: path.join(evidence.runDir, "llm_output", "metrics.json"),
    telemetry: path.join(evidence.runDir, "llm_output", "telemetry.jsonl"),
    adapterReload: path.join(evidence.runDir, "llm_output", "adapter_reload.json"),
    evaluationSummary: path.join(evidence.runDir, "evaluation_summary.json"),
    sourceArtifactManifest: path.join(evidence.runDir, "artifact_manifest.json"),
    review: path.join(evidence.runDir, "review.json"),
    claimAudit: path.join(evidence.runDir, "claim_audit.json"),
    refinement: path.join(evidence.runDir, "refinement.json"),
    humanGate: path.join(evidence.runDir, "human_gate.json"),
    version: path.join(evidence.runDir, "version.json"),
    versionComparison: path.join(evidence.runDir, "version_comparison.json")
  };
  const figureArtifacts = await Promise.all(figures
    .filter((item) => item.status === "ready")
    .map((item) => artifact(`figure-${item.id}`, "research_report", item.title, "SVG", evidence.version, resolveWorkspacePath(item.path), true)));
  const artifacts = [
    artifact("report-html", "research_report", "Interactive Scientific Report", "HTML", evidence.version, htmlPath, true),
    artifact("report-pdf", "research_report", "Scientific Report PDF", "PDF", evidence.version, pdfPath, true),
    artifact("adapter", "model_artifacts", "Adapter Weights", "SafeTensors", evidence.version, runArtifacts.adapter, false),
    artifact("adapter-config", "model_artifacts", "Adapter Config", "JSON", evidence.version, runArtifacts.adapterConfig, true),
    artifact("tokenizer", "model_artifacts", "Tokenizer", "JSON", evidence.version, runArtifacts.tokenizer, false),
    artifact("adapter-readme", "model_artifacts", "Adapter Documentation", "Markdown", evidence.version, runArtifacts.adapterReadme, true),
    artifact("model-card", "model_artifacts", "Model Card", "Markdown", evidence.version, runArtifacts.modelCard, true),
    artifact("run-manifest", "reproducibility", "Run Manifest", "JSON", evidence.version, runArtifacts.run, true),
    artifact("training-config", "reproducibility", "Training Config", "JSON", evidence.version, runArtifacts.config, true),
    artifact("environment", "reproducibility", "Environment Snapshot", "JSON", evidence.version, runArtifacts.environment, true),
    artifact("metrics", "reproducibility", "Metrics", "JSON", evidence.version, runArtifacts.metrics, true),
    artifact("telemetry", "reproducibility", "Training Telemetry", "JSONL", evidence.version, runArtifacts.telemetry, true),
    artifact("adapter-reload", "reproducibility", "Adapter Reload Verification", "JSON", evidence.version, runArtifacts.adapterReload, true),
    artifact("evaluation-summary", "reproducibility", "Evaluation Summary", "JSON", evidence.version, runArtifacts.evaluationSummary, true),
    artifact("source-artifact-manifest", "reproducibility", "Source Artifact Manifest", "JSON", evidence.version, runArtifacts.sourceArtifactManifest, true),
    artifact("review", "audit", "Independent Review Report", "JSON", evidence.version, runArtifacts.review, true),
    artifact("claim-audit", "audit", "Claim Audit", "JSON", evidence.version, runArtifacts.claimAudit, true),
    artifact("refinement", "audit", "Refinement Contract", "JSON", evidence.version, runArtifacts.refinement, true),
    artifact("human-gate", "audit", "Human Gate Record", "JSON", evidence.version, runArtifacts.humanGate, true),
    artifact("version", "audit", "Run Version Record", "JSON", evidence.version, runArtifacts.version, true),
    artifact("version-comparison", "audit", "Version Comparison", "JSON", evidence.version, runArtifacts.versionComparison, true),
    artifact("final-bundle", "reproducibility", "Final Delivery Bundle", "ZIP", evidence.version, bundlePath, false)
  ];
  const resolvedArtifacts = [...await Promise.all(artifacts), ...figureArtifacts];
  const manifestPath = path.join(reportRoot, "artifact-manifest.json");
  const training = record(evidence.metrics.training);
  const payload: ScientificReportPackage = {
    schema: "evomind.scientific_report_package.v1",
    task_id: taskId,
    run_id: evidence.runId,
    version: evidence.version,
    parent_run_id: evidence.parentRunId,
    status: pdfPath && bundlePath ? "ready" : "partial",
    renderer: "EvoMind Nature Skills",
    generated_at: new Date().toISOString(),
    source_evidence: [
      `${evidence.runRelative}/llm_output/metrics.json`,
      `${evidence.runRelative}/llm_output/telemetry.jsonl`,
      `${evidence.runRelative}/data/dataset_manifest.json`,
      `${evidence.runRelative}/review.json`,
      `${evidence.runRelative}/claim_audit.json`
    ],
    metrics: {
      before,
      after,
      improvement_pp: number(evidence.metrics.improvement_pp),
      train_steps: number(training.steps),
      train_loss: number(training.train_loss),
      max_gpu_memory_mb: number(training.max_cuda_memory_mb)
    },
    dataset: evidence.dataset,
    method: evidence.config,
    reviewer: evidence.review,
    claim_audit: evidence.claimAudit,
    version_comparison: reportVersionComparison(evidence),
    figures,
    artifacts: resolvedArtifacts,
    report_html_path: toRelativePath(htmlPath)!.replaceAll("\\", "/"),
    report_pdf_path: pdfPath ? toRelativePath(pdfPath)!.replaceAll("\\", "/") : null,
    bundle_path: bundlePath ? toRelativePath(bundlePath)!.replaceAll("\\", "/") : null,
    manifest_path: toRelativePath(manifestPath)!.replaceAll("\\", "/")
  };
  await writeJsonAtomic(manifestPath, payload);
  return payload;
}

export async function generateScientificReport(
  taskId: string,
  requestedRunId?: string | null,
  options: { automated?: boolean } = {},
): Promise<ScientificReportPackage> {
  const safeTaskId = assertSafeId(taskId, "task_id");
  const safeRunId = assertSafeId(requestedRunId?.trim() ?? "", "run_id");
  const key = `${safeTaskId}:${safeRunId}`;
  const active = scientificReportJobs.get(key);
  if (active) return active;
  const automated = options.automated === true;
  const job = (async () => {
    await updateGenerationStatus(safeTaskId, safeRunId, "collecting_evidence", "running", automated);
    try {
      const payload = await generateScientificReportUnlocked(safeTaskId, safeRunId, automated);
      await updateGenerationStatus(safeTaskId, safeRunId, "ready", "ready", automated);
      return payload;
    } catch (error) {
      await updateGenerationStatus(
        safeTaskId,
        safeRunId,
        "failed",
        "failed",
        automated,
        error instanceof Error ? error.message : "scientific report generation failed",
      ).catch(() => undefined);
      throw error;
    }
  })();
  scientificReportJobs.set(key, job);
  try {
    return await job;
  } finally {
    if (scientificReportJobs.get(key) === job) scientificReportJobs.delete(key);
  }
}

function reportManifestPath(taskId: string, runId: string) {
  return assertInsideWorkspace(resolveWorkspacePath(path.join(
    "workspace",
    "tasks",
    taskId,
    "reports",
    "scientific",
    runId,
    "artifact-manifest.json",
  )));
}

function integrityMessage(error: unknown) {
  const message = error instanceof Error ? error.message : "scientific report integrity verification failed";
  return message.replaceAll(path.resolve(workspaceRoot), "[workspace]").slice(0, 800);
}

async function rawRunState(taskId: string, runId: string): Promise<
  | { status: "not_found" }
  | { status: "integrity_failed"; error: string }
  | { status: "known"; runStatus: string }
> {
  const runDir = assertInsideWorkspace(resolveWorkspacePath(path.join("workspace", "evomind_runs", runId)));
  const stat = await fs.stat(runDir).catch(() => null);
  if (!stat?.isDirectory()) return { status: "not_found" };
  const [run, request, refinement, pointer] = await Promise.all([
    readJson(path.join(runDir, "run.json")),
    readJson(path.join(runDir, "request.json")),
    readJson(path.join(runDir, "refinement.json")),
    readJson(resolveWorkspacePath("workspace/current_run.json")),
  ]);

  const storedRunId = text(run.run_id);
  if (storedRunId && storedRunId !== runId) {
    return { status: "integrity_failed", error: "run identity does not match the requested report binding" };
  }
  const explicitTasks = [text(run.task_id), text(request.task_id)].filter(Boolean);
  if (explicitTasks.some((value) => value !== taskId)) {
    return { status: "integrity_failed", error: "run task identity does not match the requested report binding" };
  }
  const refinementTask = text(refinement.task_id);
  const refinementChild = text(refinement.child_run_id);
  if ((refinementTask || refinementChild) && (refinementTask !== taskId || refinementChild !== runId)) {
    return { status: "integrity_failed", error: "refinement binding drift was detected" };
  }

  const boundByRun = matchesTask(taskId, request, run);
  const boundByRefinement = refinementTask === taskId && refinementChild === runId;
  const boundByPointer = text(pointer.task_id) === taskId && text(pointer.run_id) === runId;
  if (!boundByRun && !boundByRefinement && !boundByPointer) return { status: "not_found" };
  return { status: "known", runStatus: text(run.status) || "starting" };
}

async function verifyReportManifest(
  taskId: string,
  runId: string,
  evidence: RunEvidence,
  manifestPath: string,
): Promise<ScientificReportPackage> {
  const payload = await readJson(manifestPath);
  if (
    payload.schema !== "evomind.scientific_report_package.v1"
    || payload.task_id !== taskId
    || payload.run_id !== runId
    || payload.status !== "ready"
  ) {
    throw new Error("scientific report manifest schema, binding or ready status is invalid");
  }
  if (record(payload.reviewer).status !== "passed" || record(payload.claim_audit).status !== "passed") {
    throw new Error("scientific report manifest is not bound to passed review and Claim Audit evidence");
  }
  if (text(evidence.review.status) !== "passed" || text(evidence.claimAudit.status) !== "passed") {
    throw new Error("scientific report source review binding is no longer valid");
  }
  const expectedManifestRelative = toRelativePath(manifestPath)?.replaceAll("\\", "/") ?? "";
  if (!expectedManifestRelative || text(payload.manifest_path).replaceAll("\\", "/") !== expectedManifestRelative) {
    throw new Error("scientific report manifest path binding drift was detected");
  }

  if (!Array.isArray(payload.artifacts) || !Array.isArray(payload.figures)) {
    throw new Error("scientific report manifest artifact collections are malformed");
  }
  const rawArtifacts = payload.artifacts.map(record);
  const ids = new Set<string>();
  const paths = new Set<string>();
  const workspaceRealPath = await fs.realpath(path.resolve(workspaceRoot));
  for (const [index, item] of rawArtifacts.entries()) {
    const id = text(item.id);
    if (!id || ids.has(id)) throw new Error(`scientific report artifact ${index} has a duplicate or missing ID`);
    ids.add(id);
    const status = text(item.status);
    if (status === "unavailable") continue;
    const relativePath = text(item.path).replaceAll("\\", "/");
    const expectedHash = text(item.sha256).toLowerCase();
    const expectedBytes = number(item.bytes);
    if (
      status !== "ready"
      || !relativePath
      || path.isAbsolute(relativePath)
      || relativePath.split("/").includes("..")
      || !/^[a-f0-9]{64}$/.test(expectedHash)
      || expectedBytes === null
      || expectedBytes < 0
      || !Number.isInteger(expectedBytes)
    ) {
      throw new Error(`scientific report artifact ${id} has an invalid manifest entry`);
    }
    if (paths.has(relativePath)) throw new Error(`scientific report artifact path is duplicated: ${relativePath}`);
    paths.add(relativePath);
    const target = assertInsideWorkspace(resolveWorkspacePath(relativePath));
    const artifactStat = await fs.stat(target).catch(() => null);
    if (!artifactStat?.isFile() || artifactStat.size !== expectedBytes) {
      throw new Error(`scientific report artifact size verification failed: ${id}`);
    }
    const targetRealPath = await fs.realpath(target);
    if (!targetRealPath.startsWith(workspaceRealPath + path.sep)) {
      throw new Error(`scientific report artifact resolves outside workspace: ${id}`);
    }
    if (await sha256(target) !== expectedHash) {
      throw new Error(`scientific report artifact hash verification failed: ${id}`);
    }
  }

  const readyArtifact = (id: string) => rawArtifacts.find((item) => item.id === id && item.status === "ready");
  const htmlArtifact = readyArtifact("report-html");
  const pdfArtifact = readyArtifact("report-pdf");
  const bundleArtifact = readyArtifact("final-bundle");
  if (
    !htmlArtifact
    || !pdfArtifact
    || !bundleArtifact
    || text(payload.report_html_path) !== text(htmlArtifact.path)
    || text(payload.report_pdf_path) !== text(pdfArtifact.path)
    || text(payload.bundle_path) !== text(bundleArtifact.path)
  ) {
    throw new Error("scientific report primary artifact bindings are incomplete");
  }

  const report = {
    ...payload,
    version_comparison: isRecord(payload.version_comparison)
      ? payload.version_comparison as ScientificVersionComparison
      : null,
  } as ScientificReportPackage;
  const unsafeSvg = /<(?:script|foreignObject|iframe|object|embed)\b|\son[a-z]+\s*=|(?:href|src)\s*=\s*["']\s*(?:javascript:|https?:|\/\/|file:|data:(?!image\/))/i;
  const figures = await Promise.all(report.figures.map(async (figure) => {
    if (figure.status !== "ready") return figure;
    const figurePath = text(figure.path).replaceAll("\\", "/");
    const boundArtifact = rawArtifacts.find((item) => item.id === `figure-${figure.id}` && item.status === "ready");
    if (!figurePath.toLowerCase().endsWith(".svg") || text(boundArtifact?.path).replaceAll("\\", "/") !== figurePath) {
      throw new Error(`scientific report figure binding verification failed: ${figure.id}`);
    }
    const source = await fs.readFile(assertInsideWorkspace(resolveWorkspacePath(figurePath)), "utf-8");
    if (!source || unsafeSvg.test(source)) throw new Error(`scientific report SVG integrity verification failed: ${figure.id}`);
    return { ...figure, preview_data_url: `data:image/svg+xml;base64,${Buffer.from(source, "utf-8").toString("base64")}` };
  }));
  return { ...report, figures };
}

export async function lookupScientificReport(taskId: string, requestedRunId?: string | null): Promise<ScientificReportLookupResult> {
  const safeTaskId = assertSafeId(taskId, "task_id");
  const safeRunId = assertSafeId(requestedRunId?.trim() ?? "", "run_id");
  const raw = await rawRunState(safeTaskId, safeRunId);
  if (raw.status !== "known") return raw;

  const generationPath = generationStatusPath(safeTaskId, safeRunId);
  const manifestPath = reportManifestPath(safeTaskId, safeRunId);
  const [generation, generationStat, manifestStat] = await Promise.all([
    readScientificReportGenerationStatus(safeTaskId, safeRunId),
    fs.stat(generationPath).catch(() => null),
    fs.stat(manifestPath).catch(() => null),
  ]);
  if (!manifestStat?.isFile()) {
    const existingSiimReport = await buildSiimExistingReport(safeTaskId, safeRunId);
    if (existingSiimReport) return { status: "ready", report: existingSiimReport };
    const existingReviewedReport = await buildReviewedExistingReport(safeTaskId, safeRunId);
    if (existingReviewedReport) return { status: "ready", report: existingReviewedReport };
  }
  if (generationStat?.isFile() && !generation) {
    return { status: "integrity_failed", error: "scientific report generation ledger is malformed" };
  }
  if (generation?.status === "failed") {
    return {
      status: "generation_failed",
      runStatus: raw.runStatus,
      error: generation.error || "scientific report generation failed",
      generation,
    };
  }
  if (!manifestStat?.isFile()) {
    const state = classifyMissingScientificReport(raw.runStatus, generation?.status ?? null);
    if (state === "pending") return { status: "pending_report", runStatus: raw.runStatus, generation };
    if (state === "generation_failed") {
      return {
        status: "generation_failed",
        runStatus: raw.runStatus,
        error: generation?.error || "scientific report generation failed",
        generation,
      };
    }
    if (state === "not_found") return { status: "not_found" };
    return { status: "integrity_failed", error: "ready generation ledger is missing its bound report manifest" };
  }

  try {
    // Once a manifest exists, every source-readiness, Reviewer, Claim Audit,
    // binding and hash error is terminal integrity drift.  Do not downgrade a
    // broken reviewed artifact to an indefinitely retryable 202 merely because
    // the run still carries an active-looking status.
    const evidence = await findRun(safeTaskId, safeRunId);
    return { status: "ready", report: await verifyReportManifest(safeTaskId, safeRunId, evidence, manifestPath) };
  } catch (error) {
    return { status: "integrity_failed", error: integrityMessage(error) };
  }
}

export async function readScientificReport(taskId: string, requestedRunId?: string | null): Promise<ScientificReportPackage | null> {
  const lookup = await lookupScientificReport(taskId, requestedRunId);
  if (lookup.status === "ready") return lookup.report;
  if (lookup.status === "integrity_failed") throw new Error(lookup.error);
  return null;
}
