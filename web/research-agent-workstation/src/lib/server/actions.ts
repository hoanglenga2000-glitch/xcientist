import { promises as fs } from "node:fs";
import { createHash } from "node:crypto";
import path from "node:path";
import { prisma } from "@/lib/db";
import { decodeJson, encodeJson } from "@/lib/server/json";
import { readJsonFile, runtimeRoot, stamp } from "@/lib/server/paths";

export type LogActionInput = {
  action: string;
  message: string;
  taskId?: string;
  runId?: string;
  artifactPath?: string | null;
  metadata?: Record<string, unknown>;
};

type ActionMirrorRecord = {
  action_id: string;
  action: string;
  task_id?: string;
  run_id?: string;
  message: string;
  artifact?: string;
  metadata: Record<string, unknown>;
  at: string;
};

type MirrorState = {
  actionCount: number;
  lastActionId: string | null;
  bytes: number;
};

const actionLogPath = path.join(runtimeRoot, "action_log.jsonl");
const checkpointPath = path.join(runtimeRoot, "action_log.checkpoint.json");
const reconcileMarkerPath = path.join(runtimeRoot, "action_log.reconcile-required.json");
let mirrorState: MirrorState | null = null;
let mirrorVerification: Promise<void> | null = null;
let mirrorQueue: Promise<void> = Promise.resolve();

function rowToMirrorRecord(row: {
  id: string;
  action: string;
  taskId: string | null;
  runId: string | null;
  message: string;
  artifactPath: string | null;
  metadataJson: string | null;
  createdAt: Date;
}): ActionMirrorRecord {
  return {
    action_id: row.id,
    action: row.action,
    task_id: row.taskId ?? undefined,
    run_id: row.runId ?? undefined,
    message: row.message,
    artifact: row.artifactPath ?? undefined,
    metadata: decodeJson<Record<string, unknown>>(row.metadataJson) ?? {},
    at: row.createdAt.toISOString()
  };
}

async function writeFileDurably(target: string, payload: string) {
  await fs.mkdir(path.dirname(target), { recursive: true });
  const temporary = `${target}.${process.pid}.${Math.random().toString(36).slice(2)}.tmp`;
  const handle = await fs.open(temporary, "w");
  try {
    await handle.writeFile(payload, "utf-8");
    await handle.sync();
  } finally {
    await handle.close();
  }
  await fs.rename(temporary, target).catch(async (error: NodeJS.ErrnoException) => {
    if (error.code !== "EEXIST" && error.code !== "EPERM") throw error;
    await fs.rm(target, { force: true });
    await fs.rename(temporary, target);
  });
}

async function writeCheckpoint(state: MirrorState) {
  const payload = {
    schema: "research_workstation.action_log_checkpoint.v1",
    sqlite_is_canonical: true,
    action_count: state.actionCount,
    last_action_id: state.lastActionId,
    mirror_bytes: state.bytes,
    updated_at: new Date().toISOString()
  };
  await writeFileDurably(checkpointPath, `${JSON.stringify(payload, null, 2)}\n`);
}

async function appendMirrorRecord(record: ActionMirrorRecord) {
  await fs.mkdir(runtimeRoot, { recursive: true });
  const line = `${JSON.stringify(record)}\n`;
  const handle = await fs.open(actionLogPath, "a");
  try {
    await handle.writeFile(line, "utf-8");
    await handle.sync();
  } finally {
    await handle.close();
  }
  const previous = mirrorState ?? { actionCount: 0, lastActionId: null, bytes: 0 };
  mirrorState = {
    actionCount: previous.actionCount + 1,
    lastActionId: record.action_id,
    bytes: (await fs.stat(actionLogPath)).size
  };
  await writeCheckpoint(mirrorState);
  await fs.rm(reconcileMarkerPath, { force: true });
}

export async function reconcileActionLogMirror(options: { archivePrevious?: boolean } = {}) {
  const rows = await prisma.actionLog.findMany({
    orderBy: [{ createdAt: "asc" }, { id: "asc" }],
    select: {
      id: true,
      action: true,
      taskId: true,
      runId: true,
      message: true,
      artifactPath: true,
      metadataJson: true,
      createdAt: true
    }
  });
  const records = rows.map(rowToMirrorRecord);
  const canonical = records.length ? `${records.map((record: ActionMirrorRecord) => JSON.stringify(record)).join("\n")}\n` : "";
  const previous = await fs.readFile(actionLogPath, "utf-8").catch(() => "");
  let archivePath: string | null = null;
  if (previous !== canonical) {
    await fs.mkdir(runtimeRoot, { recursive: true });
    if (previous && options.archivePrevious !== false) {
      archivePath = path.join(runtimeRoot, `action_log.pre-reconcile.${stamp()}.${Math.random().toString(36).slice(2, 8)}.jsonl`);
      await fs.copyFile(actionLogPath, archivePath);
    }
    await writeFileDurably(actionLogPath, canonical);
  }
  mirrorState = {
    actionCount: records.length,
    lastActionId: records.at(-1)?.action_id ?? null,
    bytes: Buffer.byteLength(canonical)
  };
  await writeCheckpoint(mirrorState);
  await fs.rm(reconcileMarkerPath, { force: true });
  return {
    ok: true,
    sqlite_is_canonical: true,
    action_count: records.length,
    mirror_sha256: createHash("sha256").update(canonical).digest("hex"),
    rewritten: previous !== canonical,
    archive_path: archivePath
  };
}

async function ensureActionMirrorConsistency() {
  if (mirrorVerification) return mirrorVerification;
  mirrorVerification = (async () => {
    const checkpoint = await readJsonFile(checkpointPath) as Record<string, unknown> | null;
    const actionCount = await prisma.actionLog.count();
    const latest = await prisma.actionLog.findFirst({ orderBy: [{ createdAt: "desc" }, { id: "desc" }], select: { id: true } });
    const mirrorBytes = await fs.stat(actionLogPath).then((item) => item.size).catch(() => -1);
    const valid = checkpoint?.sqlite_is_canonical === true
      && Number(checkpoint.action_count) === actionCount
      && String(checkpoint.last_action_id ?? "") === String(latest?.id ?? "")
      && Number(checkpoint.mirror_bytes) === mirrorBytes;
    if (valid) {
      mirrorState = { actionCount, lastActionId: latest?.id ?? null, bytes: mirrorBytes };
      return;
    }
    await reconcileActionLogMirror();
  })().catch(async (error) => {
    const message = error instanceof Error ? error.message : String(error);
    await writeFileDurably(reconcileMarkerPath, `${JSON.stringify({
      schema: "research_workstation.action_log_reconcile_required.v1",
      sqlite_is_canonical: true,
      error: message,
      detected_at: new Date().toISOString()
    }, null, 2)}\n`).catch(() => undefined);
  });
  return mirrorVerification;
}

export async function logAction(input: LogActionInput) {
  await ensureActionMirrorConsistency();
  const id = `action_${stamp()}_${Math.random().toString(36).slice(2, 8)}`;
  const createdAt = new Date();
  let taskId = input.taskId;
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

  mirrorQueue = mirrorQueue.then(async () => {
    try {
      await appendMirrorRecord(record);
    } catch {
      await reconcileActionLogMirror().catch(async (error) => {
        const message = error instanceof Error ? error.message : String(error);
        await writeFileDurably(reconcileMarkerPath, `${JSON.stringify({
          schema: "research_workstation.action_log_reconcile_required.v1",
          sqlite_is_canonical: true,
          action_id: record.action_id,
          error: message,
          detected_at: new Date().toISOString()
        }, null, 2)}\n`).catch(() => undefined);
      });
    }
  });
  await mirrorQueue;

  return record;
}
