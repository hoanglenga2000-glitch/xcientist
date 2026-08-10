import { promises as fs } from "node:fs";
import path from "node:path";
import { randomUUID } from "node:crypto";

const SAFE_ID = /^[A-Za-z0-9_-]{1,160}$/;
const workspaceRoot = path.resolve(process.env.WORKSTATION_ROOT ?? path.resolve(process.cwd(), "..", ".."));

function resolveWorkspacePath(relativePath: string) {
  const target = path.resolve(workspaceRoot, relativePath);
  const boundary = path.relative(workspaceRoot, target);
  if (boundary.startsWith("..") || path.isAbsolute(boundary)) {
    throw new Error(`Path escapes WORKSTATION_ROOT: ${relativePath}`);
  }
  return target;
}

export type RunLedgerProjection = {
  status: string;
  updated_at: string;
  last_seq?: number | null;
  process_id?: number | null;
};

export type RunLedgerEntry = {
  schema: "evomind.run_ledger_entry.v1";
  task_id: string;
  run_id: string;
  run_dir: string;
  output_dir: string | null;
  status: string;
  lifecycle_state: string | null;
  last_seq: number | null;
  revision: number;
  source: string;
  updated_at: string;
  projections: Record<string, RunLedgerProjection>;
  metadata?: Record<string, unknown>;
};

export type RunLedgerWrite = {
  taskId: string;
  runId: string;
  status: string;
  source: string;
  lifecycleState?: string | null;
  runDir?: string;
  outputDir?: string | null;
  lastSeq?: number | null;
  processId?: number | null;
  metadata?: Record<string, unknown>;
};

function validateId(value: string, label: string) {
  if (!SAFE_ID.test(value)) throw new Error(`Invalid ${label} for Run Ledger.`);
  return value;
}

async function readJson(filePath: string): Promise<Record<string, unknown> | null> {
  try {
    return JSON.parse(await fs.readFile(filePath, "utf-8")) as Record<string, unknown>;
  } catch {
    return null;
  }
}

async function atomicJson(filePath: string, payload: unknown) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const temporary = `${filePath}.${process.pid}.${randomUUID()}.tmp`;
  await fs.writeFile(temporary, `${JSON.stringify(payload, null, 2)}\n`, "utf-8");
  await fs.rename(temporary, filePath);
}

function projectionKey(source: string) {
  return source.replace(/[^A-Za-z0-9_-]+/g, "_").slice(0, 80) || "unknown";
}

export async function readCurrentRunLedger(): Promise<RunLedgerEntry | null> {
  const payload = await readJson(resolveWorkspacePath("workspace/run_ledger/current.json"));
  if (payload?.schema !== "evomind.run_ledger_entry.v1") return null;
  const taskId = typeof payload.task_id === "string" ? payload.task_id : "";
  const runId = typeof payload.run_id === "string" ? payload.run_id : "";
  if (!SAFE_ID.test(taskId) || !SAFE_ID.test(runId)) return null;
  return payload as RunLedgerEntry;
}

export async function readRunLedger(runIdInput: string): Promise<RunLedgerEntry | null> {
  const runId = validateId(runIdInput, "run_id");
  const payload = await readJson(resolveWorkspacePath(`workspace/run_ledger/runs/${runId}.json`));
  if (payload?.schema !== "evomind.run_ledger_entry.v1" || payload.run_id !== runId) return null;
  return payload as RunLedgerEntry;
}

export async function writeRunLedger(input: RunLedgerWrite): Promise<RunLedgerEntry> {
  const taskId = validateId(input.taskId, "task_id");
  const runId = validateId(input.runId, "run_id");
  if (!input.status.trim()) throw new Error("Run Ledger status is required.");
  if (!input.source.trim()) throw new Error("Run Ledger source is required.");

  const runPath = resolveWorkspacePath(`workspace/run_ledger/runs/${runId}.json`);
  const previous = await readJson(runPath);
  if (previous && (previous.run_id !== runId || previous.task_id !== taskId)) {
    throw new Error("Run Ledger identity mismatch.");
  }

  const updatedAt = new Date().toISOString();
  const lifecycleState = input.lifecycleState === undefined
    ? (typeof previous?.lifecycle_state === "string" ? previous.lifecycle_state : null)
    : input.lifecycleState;
  const canonicalStatus = lifecycleState ?? input.status;
  const previousProjections = previous?.projections && typeof previous.projections === "object"
    ? previous.projections as Record<string, RunLedgerProjection>
    : {};
  const projection: RunLedgerProjection = {
    status: input.status,
    updated_at: updatedAt,
    last_seq: input.lastSeq ?? null,
    ...(input.processId === undefined ? {} : { process_id: input.processId }),
  };
  const entry: RunLedgerEntry = {
    schema: "evomind.run_ledger_entry.v1",
    task_id: taskId,
    run_id: runId,
    run_dir: input.runDir ?? (typeof previous?.run_dir === "string" ? previous.run_dir : `workspace/evomind_runs/${runId}`),
    output_dir: input.outputDir === undefined
      ? (typeof previous?.output_dir === "string" ? previous.output_dir : null)
      : input.outputDir,
    status: canonicalStatus,
    lifecycle_state: lifecycleState,
    last_seq: input.lastSeq ?? (typeof previous?.last_seq === "number" ? previous.last_seq : null),
    revision: (typeof previous?.revision === "number" ? previous.revision : 0) + 1,
    source: input.source,
    updated_at: updatedAt,
    projections: {
      ...previousProjections,
      [projectionKey(input.source)]: projection,
    },
    ...(input.metadata ? { metadata: { ...(previous?.metadata as Record<string, unknown> | undefined), ...input.metadata } } : {}),
  };

  const ledgerRoot = resolveWorkspacePath("workspace/run_ledger");
  await fs.mkdir(path.join(ledgerRoot, "runs"), { recursive: true });
  await atomicJson(runPath, entry);
  await atomicJson(path.join(ledgerRoot, "current.json"), entry);
  await fs.appendFile(path.join(ledgerRoot, "events.jsonl"), `${JSON.stringify({
    schema: "evomind.run_ledger_event.v1",
    event_id: randomUUID(),
    revision: entry.revision,
    task_id: taskId,
    run_id: runId,
    status: entry.status,
    lifecycle_state: entry.lifecycle_state,
    projection_status: input.status,
    source: input.source,
    created_at: updatedAt,
    metadata: input.metadata ?? null,
  })}\n`, "utf-8");

  // Compatibility-only projection for older CLI/report consumers. The Run Ledger
  // remains the source of truth and this file is replaced atomically after it.
  await atomicJson(resolveWorkspacePath("workspace/current_run.json"), {
    schema: "evomind.current_run.v1",
    task_id: taskId,
    run_id: runId,
    run_dir: entry.run_dir,
    status: entry.status,
    last_seq: entry.last_seq,
    updated_at: entry.updated_at,
    ledger_revision: entry.revision,
    ledger_path: `workspace/run_ledger/runs/${runId}.json`,
  });
  return entry;
}
