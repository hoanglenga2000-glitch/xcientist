import { promises as fs } from "node:fs";
import path from "node:path";
import { prisma } from "@/lib/db";
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
  const id = `action_${stamp()}_${Math.random().toString(36).slice(2, 8)}`;
  const createdAt = new Date();
  const taskId = input.taskId ? normalizeTaskId(input.taskId) : undefined;
  let runId = input.runId;
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
    action: input.action,
    task_id: taskId,
    run_id: runId,
    message: input.message,
    artifact: input.artifactPath ?? undefined,
    metadata: input.metadata ?? {},
    at: createdAt.toISOString()
  };

  await prisma.actionLog.create({
    data: {
      id,
      action: input.action,
      taskId,
      runId,
      message: input.message,
      artifactPath: input.artifactPath ?? null,
      metadataJson: encodeJson(input.metadata ?? null),
      createdAt
    }
  });

  await fs.mkdir(runtimeRoot, { recursive: true });
  await fs.appendFile(path.join(runtimeRoot, "action_log.jsonl"), `${JSON.stringify(record)}\n`, "utf-8");

  return record;
}
