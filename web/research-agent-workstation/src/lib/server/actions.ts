import { promises as fs } from "node:fs";
import { randomUUID } from "node:crypto";
import path from "node:path";
import { prisma } from "@/lib/db";
import {
  normalizeAuditAction,
  sanitizeAuditArtifactPath,
  sanitizeAuditMetadata,
  sanitizeAuditText
} from "@/lib/security/audit-log";
import { encodeJson } from "@/lib/server/json";
import { normalizeTaskId, runtimeRoot, stamp } from "@/lib/server/paths";

export type LogActionInput = {
  action: string;
  message: string;
  taskId?: string;
  runId?: string;
  artifactPath?: string | null;
  metadata?: Record<string, unknown>;
};

export async function logAction(input: LogActionInput) {
  const id = `action_${stamp()}_${randomUUID()}`;
  const createdAt = new Date();
  const action = normalizeAuditAction(input.action);
  const message = sanitizeAuditText(input.message, 4_000);
  const artifactPath = sanitizeAuditArtifactPath(input.artifactPath);
  const metadata = sanitizeAuditMetadata(input.metadata);
  const taskId = input.taskId ? normalizeTaskId(input.taskId) : undefined;
  let runId = input.runId?.trim();
  if (runId && !/^[A-Za-z0-9_.:-]{1,160}$/.test(runId)) runId = undefined;
  if (taskId) {
    await prisma.task.upsert({
      where: { id: taskId },
      update: {},
      create: {
        id: taskId,
        name: taskId.replaceAll("_", " "),
        taskType: "tabular_runtime",
        status: "runtime_ready",
        priority: "Runtime",
        owner: "Research Agent Runtime",
        configPath: `configs/${taskId}.yaml`,
        taskDir: `tasks/${taskId}`
      }
    });
  }
  if (runId) {
    const run = await prisma.experimentRun.findUnique({ where: { id: runId }, select: { id: true } });
    if (!run) runId = undefined;
  }
  const record = {
    action_id: id,
    action,
    task_id: taskId,
    run_id: runId,
    message,
    artifact: artifactPath ?? undefined,
    metadata,
    at: createdAt.toISOString()
  };

  await prisma.actionLog.create({
    data: {
      id,
      action,
      taskId,
      runId,
      message,
      artifactPath,
      metadataJson: encodeJson(metadata),
      createdAt
    }
  });

  await fs.mkdir(runtimeRoot, { recursive: true });
  const logHandle = await fs.open(path.join(runtimeRoot, "action_log.jsonl"), "a", 0o600);
  try {
    // lgtm[js/http-to-file-access] All request-derived fields are bounded and credential-redacted above.
    await logHandle.writeFile(`${JSON.stringify(record)}\n`, "utf-8");
    await logHandle.sync();
  } finally {
    await logHandle.close();
  }

  return record;
}
