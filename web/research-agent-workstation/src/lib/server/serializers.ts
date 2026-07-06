import { decodeJson } from "@/lib/server/json";

type AnyRecord = Record<string, any>;

export function serializeTask(task: AnyRecord) {
  return {
    id: task.id,
    name: task.name,
    task_type: task.taskType,
    target: task.target,
    metric: task.metric,
    status: task.status,
    priority: task.priority,
    owner: task.owner,
    config_path: task.configPath,
    task_dir: task.taskDir,
    created_at: task.createdAt?.toISOString?.() ?? task.createdAt,
    updated_at: task.updatedAt?.toISOString?.() ?? task.updatedAt
  };
}

export function serializeRun(run: AnyRecord) {
  return {
    id: run.id,
    task_id: run.taskId,
    output_dir: run.outputDir,
    status: run.status,
    best_model: run.bestModel,
    best_metrics: decodeJson(run.metricsJson),
    validation_status: run.validationStatus,
    validation_gate: run.validationStatus ? { status: run.validationStatus } : undefined,
    accepted: run.validationStatus === "passed" || run.status === "passed",
    process_id: run.processId,
    started_at: run.startedAt?.toISOString?.() ?? run.startedAt ?? null,
    finished_at: run.finishedAt?.toISOString?.() ?? run.finishedAt ?? null,
    created_at: run.createdAt?.toISOString?.() ?? run.createdAt,
    updated_at: run.updatedAt?.toISOString?.() ?? run.updatedAt
  };
}

export function serializeGate(gate: AnyRecord) {
  return {
    id: gate.id,
    task_id: gate.taskId,
    run_id: gate.runId,
    gate_type: gate.gateType,
    decision: gate.decision,
    reviewer: gate.reviewer,
    evidence: decodeJson(gate.evidenceJson),
    created_at: gate.createdAt?.toISOString?.() ?? gate.createdAt,
    decided_at: gate.decidedAt?.toISOString?.() ?? gate.decidedAt ?? null
  };
}

export function serializeEvidence(evidence: AnyRecord) {
  return {
    id: evidence.id,
    task_id: evidence.taskId,
    run_id: evidence.runId,
    label: evidence.label,
    artifact_path: evidence.artifactPath,
    hash: evidence.hash,
    source: evidence.source,
    claim_binding: evidence.claimBinding,
    created_at: evidence.createdAt?.toISOString?.() ?? evidence.createdAt
  };
}

export function serializeReport(report: AnyRecord | null) {
  if (!report) return null;
  return {
    id: report.id,
    task_id: report.taskId,
    run_id: report.runId,
    title: report.title,
    status: report.status,
    markdown_content: report.markdownContent,
    content: decodeJson(report.contentJson),
    markdown_path: report.markdownPath,
    docx_path: report.docxPath,
    selected_section: report.selectedSection,
    submitted_at: report.submittedAt?.toISOString?.() ?? report.submittedAt ?? null,
    created_at: report.createdAt?.toISOString?.() ?? report.createdAt,
    updated_at: report.updatedAt?.toISOString?.() ?? report.updatedAt
  };
}

export function serializeWorkflow(workflow: AnyRecord | null) {
  if (!workflow) return null;
  return {
    id: workflow.id,
    task_id: workflow.taskId,
    name: workflow.name,
    status: workflow.status,
    version: workflow.version,
    nodes: decodeJson(workflow.nodesJson),
    edges: decodeJson(workflow.edgesJson),
    published_at: workflow.publishedAt?.toISOString?.() ?? workflow.publishedAt ?? null,
    created_at: workflow.createdAt?.toISOString?.() ?? workflow.createdAt,
    updated_at: workflow.updatedAt?.toISOString?.() ?? workflow.updatedAt
  };
}

export function serializeSetting(setting: AnyRecord) {
  return {
    key: setting.key,
    value: decodeJson(setting.valueJson),
    updated_at: setting.updatedAt?.toISOString?.() ?? setting.updatedAt
  };
}
