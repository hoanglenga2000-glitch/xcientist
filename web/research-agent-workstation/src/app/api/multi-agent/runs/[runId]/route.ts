import crypto from "node:crypto";
import { promises as fs } from "node:fs";
import { NextResponse } from "next/server";
import { sanitizeClientJson } from "@/lib/server/json";
import { readJsonFile, resolveWorkspacePath } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

function safeRunId(value: string) {
  return /^[A-Za-z0-9_-]{8,160}$/.test(value) ? value : null;
}

async function readJsonLines(filePath: string) {
  const text = await fs.readFile(filePath, "utf-8").catch(() => "");
  return text.split(/\r?\n/).filter(Boolean).flatMap((line) => {
    try { return [JSON.parse(line) as Record<string, unknown>]; } catch { return []; }
  });
}

const siimDeliverableNames = [
  "evomind-siim-isic-report.pdf",
  "evomind-siim-isic-results.csv",
  "evomind-siim-isic-code.zip",
  "evomind-siim-isic-evidence.zip",
] as const;

async function verifiedSiimDeliverables(
  runDir: string,
  runId: string,
  manifest: Record<string, unknown> | null,
) {
  const records = new Map(
    (Array.isArray(manifest?.artifacts) ? manifest.artifacts : [])
      .filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object")
      .map((item) => [String(item.path ?? ""), item]),
  );
  const files = await Promise.all(siimDeliverableNames.map(async (name) => {
    const record = records.get(name);
    const body = await fs.readFile(`${runDir}/${name}`).catch(() => null);
    const sha256 = body ? crypto.createHash("sha256").update(body).digest("hex") : null;
    const ready = Boolean(
      manifest?.run_id === runId
      && manifest?.status === "verified"
      && body
      && record
      && record.sha256 === sha256
      && record.bytes === body.byteLength
    );
    return {
      name,
      status: ready ? "ready" : "unavailable",
      bytes: ready && body ? body.byteLength : null,
      sha256: ready ? sha256 : null,
      download_url: ready ? `/api/multi-agent/runs/${runId}/download/${name}` : null,
    };
  }));
  return {
    schema: "evomind.siim.deliverables.snapshot.v1",
    run_id: runId,
    status: files.every((file) => file.status === "ready") ? "ready" : "unavailable",
    files,
  };
}

export async function GET(_request: Request, { params }: { params: Promise<{ runId: string }> }) {
  const { runId: rawRunId } = await params;
  const runId = safeRunId(rawRunId);
  if (!runId) return NextResponse.json({ ok: false, error: "invalid run id" }, { status: 400 });
  const runDir = resolveWorkspacePath(`workspace/evomind_runs/${runId}`);
  const requestPath = resolveWorkspacePath(`workspace/evomind_requests/${runId}.txt`);
  const [runDirectoryExists, requestExists] = await Promise.all([
    fs.stat(runDir).then((value) => value.isDirectory()).catch(() => false),
    fs.stat(requestPath).then((value) => value.isFile()).catch(() => false)
  ]);
  if (!runDirectoryExists && !requestExists) {
    return NextResponse.json({ ok: false, error: "run not found", run_id: runId }, { status: 404 });
  }
  const [
    run, graph, requestPayload, review, rootMetrics, llmMetrics, manifest, report,
    qloraConfig, datasetManifest, evaluation, environment, telemetry, adapterReload, claimAudit, modelCard,
    datasetProfile, experimentComparison, hpcRuntime, historicalThresholds, deliverables,
    privateGrader, privateGraderLedger, candidateFreeze,
  ] = await Promise.all([
    readJsonFile(`${runDir}/run.json`),
    readJsonFile(`${runDir}/task_graph.json`),
    readJsonFile(`${runDir}/request.json`),
    readJsonFile(`${runDir}/review.json`),
    readJsonFile(`${runDir}/metrics.json`),
    readJsonFile(`${runDir}/llm_output/metrics.json`),
    readJsonFile(`${runDir}/artifact_manifest.json`),
    fs.readFile(`${runDir}/research_report.md`, "utf-8").catch(() => ""),
    readJsonFile(`${runDir}/qlora_config.json`),
    readJsonFile(`${runDir}/data/dataset_manifest.json`),
    readJsonFile(`${runDir}/llm_output/evaluation.json`),
    readJsonFile(`${runDir}/llm_output/environment.json`),
    readJsonLines(`${runDir}/llm_output/telemetry.jsonl`),
    readJsonFile(`${runDir}/llm_output/adapter_reload.json`),
    readJsonFile(`${runDir}/claim_audit.json`),
    fs.readFile(`${runDir}/model_card.md`, "utf-8").catch(() => ""),
    readJsonFile(`${runDir}/dataset_profile.json`),
    readJsonFile(`${runDir}/experiment_comparison.json`),
    readJsonFile(`${runDir}/hpc_runtime.json`),
    readJsonFile(`${runDir}/historical_thresholds.json`),
    readJsonFile(`${runDir}/deliverables.json`),
    readJsonFile(`${runDir}/private_grader.json`),
    readJsonFile(`${runDir}/private_grader_ledger.json`),
    readJsonFile(`${runDir}/candidate_freeze.json`),
  ]);
  if (!run) return NextResponse.json({ ok: false, run_id: runId, status: "starting" }, { status: 202 });
  const metrics = llmMetrics ?? rootMetrics;
  const metricsRecord = metrics as Record<string, unknown> | null;
  const training = metricsRecord?.training as Record<string, unknown> | undefined;
  const requestRecord = requestPayload as Record<string, unknown> | null;
  const runRecord = run as Record<string, unknown>;
  const isSiim = requestRecord?.task_type === "image_classification"
    && requestRecord?.dataset === "siim-isic-melanoma-classification";
  const siimDeliverables = isSiim
    ? await verifiedSiimDeliverables(runDir, runId, manifest as Record<string, unknown> | null)
    : null;
  return NextResponse.json(sanitizeClientJson({
    ok: true,
    run,
    task_graph: graph,
    request: requestPayload,
    review,
    metrics,
    artifact_manifest: manifest,
    report_markdown: report,
    model_card_markdown: modelCard,
    dataset_profile: datasetProfile ?? runRecord.dataset_profile ?? null,
    experiment_comparison: experimentComparison ?? runRecord.experiment_comparison ?? null,
    hpc_runtime: hpcRuntime ?? runRecord.hpc_runtime ?? null,
    historical_thresholds: historicalThresholds ?? runRecord.historical_thresholds ?? null,
    deliverables: siimDeliverables ?? deliverables ?? runRecord.deliverables ?? null,
    siim: isSiim ? {
      run_id: runId,
      dataset_profile: datasetProfile ?? runRecord.dataset_profile ?? null,
      experiment_comparison: experimentComparison ?? runRecord.experiment_comparison ?? null,
      hpc_runtime: hpcRuntime ?? runRecord.hpc_runtime ?? null,
      review,
      claim_audit: claimAudit,
      historical_thresholds: historicalThresholds ?? runRecord.historical_thresholds ?? null,
      candidate_freeze: candidateFreeze,
      private_grader: privateGrader,
      private_grader_ledger: privateGraderLedger,
      deliverables: siimDeliverables,
    } : null,
    llm: {
      task_type: requestRecord?.task_type ?? null,
      base_model: metricsRecord?.base_model ?? (qloraConfig as Record<string, unknown> | null)?.base_model ?? null,
      dataset: datasetManifest,
      qlora_config: qloraConfig,
      training: training ? {
        step: training.steps ?? null,
        loss: training.train_loss ?? null,
        gpu_memory_mb: training.max_cuda_memory_mb ?? null,
      } : null,
      before_after_eval: metricsRecord ? {
        before: metricsRecord.before ?? null,
        after: metricsRecord.after ?? null,
        improvement_pp: metricsRecord.improvement_pp ?? null,
      } : null,
      evaluation,
      environment,
      telemetry,
      adapter_reload: adapterReload,
      claim_audit: claimAudit,
    },
  }));
}
