import { promises as fs } from "node:fs";
import { createHash } from "node:crypto";
import path from "node:path";
import type { ScientificArtifact, ScientificFigure, ScientificReportPackage } from "@/lib/api/types";
import { resolveWorkspacePath } from "@/lib/server/paths";
import { readScientificReport } from "@/lib/server/scientific-report";

export const PUBLIC_REPORT_TASK_ID = "evomind-qwen7b-finetune";
export const PUBLIC_REPORT_RUN_ID = "EVOMIND-DEMO-7B-20260722";

export const DEFAULT_PUBLIC_REPORT_SOURCE_RUN_ID = "qwen7b_qlora_20260722T135300Z_a800_grounded";
const allowedArtifactTypes = new Set(["HTML", "PDF", "SVG"]);

type PublicArtifactSource = {
  absolutePath: string;
  filename: string;
  contentType: string;
  transformText: boolean;
};

export function publicReportSourceRunId() {
  return process.env.EVOMIND_PUBLIC_DEMO_SOURCE_RUN?.trim() || DEFAULT_PUBLIC_REPORT_SOURCE_RUN_ID;
}

function publicPath(artifactId: string) {
  return `public:${artifactId}`;
}

function publicDataset(dataset: Record<string, unknown>) {
  return {
    method: dataset.method,
    counts: dataset.counts,
    source_split_policy: dataset.source_split_policy,
    duplicate_record_ids: dataset.duplicate_record_ids,
    duplicate_content_records: dataset.duplicate_content_records,
    grounded_context_contract: dataset.grounded_context_contract,
    secrets_removed: dataset.secrets_removed,
  };
}

function publicMethod(method: Record<string, unknown>) {
  return {
    base_model: method.base_model,
    method: method.method,
    compute_dtype: method.compute_dtype,
    double_quant: method.double_quant,
    gradient_checkpointing: method.gradient_checkpointing,
    lora_r: method.lora_r,
    lora_alpha: method.lora_alpha,
    lora_dropout: method.lora_dropout,
    max_sequence_length: method.max_sequence_length,
    effective_batch_size: method.effective_batch_size,
    epochs: method.epochs,
    learning_rate: method.learning_rate,
    local_gpu_allowed: method.local_gpu_allowed,
    model_publication: method.model_publication,
  };
}

function publicReviewer(reviewer: Record<string, unknown>) {
  const checks = reviewer.checks;
  const safeChecks = checks && typeof checks === "object" && !Array.isArray(checks)
    ? Object.fromEntries(Object.entries(checks as Record<string, unknown>).filter(([, value]) => typeof value === "boolean"))
    : {};
  return { status: reviewer.status, checks: safeChecks };
}

function publicClaimAudit(claimAudit: Record<string, unknown>) {
  return { status: claimAudit.status };
}

function publicFigure(figure: ScientificFigure): ScientificFigure {
  return {
    ...figure,
    path: publicPath(figure.id),
    sources: ["Verified report evidence"],
  };
}

function publicArtifact(artifact: ScientificArtifact): ScientificArtifact | null {
  if (artifact.status !== "ready" || !artifact.path || !allowedArtifactTypes.has(artifact.type)) return null;
  return { ...artifact, path: publicPath(artifact.id) };
}

export async function readPublicArtifactBody(source: PublicArtifactSource) {
  const raw = await fs.readFile(source.absolutePath);
  const body = source.transformText
    ? Buffer.from(sanitizePublicReportText(raw.toString("utf-8")), "utf-8")
    : raw;
  return {
    body,
    bytes: body.byteLength,
    sha256: createHash("sha256").update(body).digest("hex"),
  };
}

function assertPublicRequest(taskId: string, runId: string) {
  if (taskId !== PUBLIC_REPORT_TASK_ID || runId !== PUBLIC_REPORT_RUN_ID) {
    throw new Error("The requested public report is not available.");
  }
}

async function sourceReport() {
  const report = await readScientificReport(PUBLIC_REPORT_TASK_ID, publicReportSourceRunId());
  if (!report || report.status !== "ready" || report.reviewer?.status !== "passed" || report.claim_audit?.status !== "passed") {
    throw new Error("The reviewed public report is not available.");
  }
  return report;
}

export async function readPublicScientificReport(taskId: string, runId: string): Promise<ScientificReportPackage> {
  assertPublicRequest(taskId, runId);
  const report = await sourceReport();
  const artifactResults = await Promise.all(report.artifacts.map(async (artifact): Promise<ScientificArtifact | null> => {
    const safe = publicArtifact(artifact);
    const source = safe ? sourceFromArtifact(artifact) : null;
    if (!safe || !source) return null;
    const digest = await readPublicArtifactBody(source);
    return { ...safe, bytes: digest.bytes, sha256: digest.sha256 };
  }));
  const artifacts = artifactResults.filter((artifact): artifact is ScientificArtifact => artifact !== null);
  return {
    ...report,
    task_id: PUBLIC_REPORT_TASK_ID,
    run_id: PUBLIC_REPORT_RUN_ID,
    parent_run_id: report.parent_run_id ? "EVOMIND-DEMO-7B-PREVIOUS" : null,
    source_evidence: ["Reviewed metrics", "Training telemetry", "Dataset contract", "Independent Review", "Claim Audit"],
    dataset: publicDataset(report.dataset),
    method: publicMethod(report.method),
    reviewer: publicReviewer(report.reviewer),
    claim_audit: publicClaimAudit(report.claim_audit),
    figures: report.figures.map(publicFigure),
    artifacts,
    report_html_path: publicPath("report-html"),
    report_pdf_path: artifacts.some((artifact) => artifact.id === "report-pdf") ? publicPath("report-pdf") : null,
    bundle_path: null,
    manifest_path: "public:report-manifest",
  };
}

function safeTarget(relativePath: string) {
  const target = path.resolve(resolveWorkspacePath(relativePath));
  const workspace = path.resolve(resolveWorkspacePath("."));
  if (!target.startsWith(`${workspace}${path.sep}`)) throw new Error("Public artifact path escapes workspace.");
  return target;
}

function sourceFromArtifact(artifact: ScientificArtifact): PublicArtifactSource | null {
  if (!artifact.path || !allowedArtifactTypes.has(artifact.type)) return null;
  const extension = path.extname(artifact.path).toLowerCase();
  const contentTypes: Record<string, string> = {
    ".html": "text/html; charset=utf-8",
    ".pdf": "application/pdf",
    ".svg": "image/svg+xml; charset=utf-8",
  };
  const contentType = contentTypes[extension];
  if (!contentType) return null;
  return {
    absolutePath: safeTarget(artifact.path),
    filename: path.basename(artifact.path),
    contentType,
    transformText: extension === ".html" || extension === ".svg",
  };
}

function sourceFromFigure(figure: ScientificFigure): PublicArtifactSource | null {
  const extension = path.extname(figure.path).toLowerCase();
  if (extension !== ".svg") return null;
  return {
    absolutePath: safeTarget(figure.path),
    filename: path.basename(figure.path),
    contentType: "image/svg+xml; charset=utf-8",
    transformText: true,
  };
}

export async function resolvePublicScientificArtifact(taskId: string, runId: string, artifactId: string) {
  assertPublicRequest(taskId, runId);
  if (!/^[A-Za-z0-9_-]{1,120}$/.test(artifactId)) throw new Error("Invalid public artifact ID.");
  const report = await sourceReport();
  const artifact = report.artifacts.find((item) => item.id === artifactId);
  const figure = report.figures.find((item) => item.id === artifactId);
  const source = artifact ? sourceFromArtifact(artifact) : figure ? sourceFromFigure(figure) : null;
  if (!source) throw new Error("The requested public artifact is not available.");
  await fs.access(source.absolutePath);
  return source;
}

export function sanitizePublicReportText(value: string) {
  return value
    .replaceAll(publicReportSourceRunId(), PUBLIC_REPORT_RUN_ID)
    .replace(/NVIDIA\s+A800(?:-SXM4)?-80GB/gi, "Remote GPU")
    .replace(/\bA800\b/gi, "Remote GPU")
    .replace(/(?:[A-Za-z]:\\|\/hpc2hdd\/)[^\s<>'"]+/g, "private-artifact");
}
