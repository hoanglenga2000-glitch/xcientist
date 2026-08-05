import { promises as fs } from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { logAction } from "@/lib/server/actions";
import { decodeJson } from "@/lib/server/json";
import { normalizeTaskId, stamp, workspaceRoot, writeJsonArtifact } from "@/lib/server/paths";
import { serializeTask } from "@/lib/server/serializers";

export const dynamic = "force-dynamic";

function safeTaskId(raw: string) {
  const taskId = normalizeTaskId(raw);
  return /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(taskId) && taskId !== "." && taskId !== ".." ? taskId : null;
}

async function taskParam(params: Promise<{ taskId: string }>) {
  const { taskId } = await params;
  return safeTaskId(taskId);
}

export async function GET(_request: Request, { params }: { params: Promise<{ taskId: string }> }) {
  const taskId = await taskParam(params);
  if (!taskId) return NextResponse.json({ ok: false, error: "Invalid task_id." }, { status: 400 });
  const task = await prisma.task.findUnique({ where: { id: taskId } });
  if (!task) return NextResponse.json({ ok: false, error: "Task not found." }, { status: 404 });
  return NextResponse.json({ ok: true, task: serializeTask(task), archived: task.status === "archived" });
}

export async function PATCH(request: Request, { params }: { params: Promise<{ taskId: string }> }) {
  const taskId = await taskParam(params);
  if (!taskId) return NextResponse.json({ ok: false, error: "Invalid task_id." }, { status: 400 });
  const body = await request.json().catch(() => ({})) as Record<string, unknown>;
  const action = String(body.action ?? "").toLowerCase();
  const task = await prisma.task.findUnique({ where: { id: taskId } });
  if (!task) return NextResponse.json({ ok: false, error: "Task not found." }, { status: 404 });

  if (action === "archive") {
    if (task.status === "archived") return NextResponse.json({ ok: true, task: serializeTask(task), archived: true, idempotent: true });
    const updated = await prisma.task.update({ where: { id: taskId }, data: { status: "archived" } });
    const record = await logAction({
      action: "archive_task",
      taskId,
      message: `Task archived: ${taskId}`,
      metadata: { previous_status: task.status, archived_at: new Date().toISOString() }
    });
    return NextResponse.json({ ok: true, ...record, task: serializeTask(updated), archived: true });
  }

  if (action === "restore") {
    if (task.status !== "archived") return NextResponse.json({ ok: true, task: serializeTask(task), archived: false, idempotent: true });
    const archiveRecord = await prisma.actionLog.findFirst({
      where: { taskId, action: "archive_task" },
      orderBy: { createdAt: "desc" }
    });
    const metadata = decodeJson(archiveRecord?.metadataJson) as Record<string, unknown> | null;
    const previousStatus = typeof metadata?.previous_status === "string" && metadata.previous_status !== "archived"
      ? metadata.previous_status
      : "ready_to_train";
    const updated = await prisma.task.update({ where: { id: taskId }, data: { status: previousStatus } });
    const record = await logAction({
      action: "restore_task",
      taskId,
      message: `Task restored: ${taskId}`,
      metadata: { restored_status: previousStatus, archive_action_id: archiveRecord?.id ?? null, restored_at: new Date().toISOString() }
    });
    return NextResponse.json({ ok: true, ...record, task: serializeTask(updated), archived: false });
  }

  return NextResponse.json({ ok: false, error: "action must be archive or restore" }, { status: 400 });
}

export async function DELETE(request: Request, { params }: { params: Promise<{ taskId: string }> }) {
  const taskId = await taskParam(params);
  if (!taskId) return NextResponse.json({ ok: false, error: "Invalid task_id." }, { status: 400 });
  const body = await request.json().catch(() => ({})) as Record<string, unknown>;
  if (body.confirm_task_id !== taskId) {
    return NextResponse.json({ ok: false, error: "confirm_task_id must exactly match the task being purged." }, { status: 409 });
  }
  const task = await prisma.task.findUnique({
    where: { id: taskId },
    include: { _count: { select: { runs: true, workflows: true, gates: true, evidence: true, reports: true, actions: true } } }
  });
  if (!task) return NextResponse.json({ ok: false, error: "Task not found." }, { status: 404 });
  if (task.status !== "archived" && body.force !== true) {
    return NextResponse.json({ ok: false, error: "Task must be archived before purge." }, { status: 409 });
  }

  const purgeId = `purge_${stamp()}`;
  const stagingRoot = path.join(workspaceRoot, "workspace", ".purge-staging", `${taskId}_${purgeId}`);
  const generatedRoots = [
    path.join(workspaceRoot, "workspace", "tasks", taskId),
    path.join(workspaceRoot, "workspace", "workflows", taskId),
    path.join(workspaceRoot, "workspace", "workstation_runs", taskId),
    path.join(workspaceRoot, "experiments", taskId)
  ];
  const staged: Array<{ source: string; target: string }> = [];
  let dbDeleted = false;
  let purgeActionId: string | null = null;
  try {
    for (const source of generatedRoots) {
      const exists = await fs.stat(source).then((stat) => stat.isDirectory()).catch(() => false);
      if (!exists) continue;
      const relative = path.relative(workspaceRoot, source);
      if (relative.startsWith("..") || path.isAbsolute(relative)) throw new Error(`Purge path escaped workspace: ${source}`);
      const target = path.join(stagingRoot, relative);
      await fs.mkdir(path.dirname(target), { recursive: true });
      await fs.rename(source, target);
      staged.push({ source, target });
    }

    const tombstone = await writeJsonArtifact(`workspace/runtime/task_tombstones/${taskId}_${purgeId}.json`, {
      schema: "evomind.task_purge_tombstone.v1",
      task_id: taskId,
      task_snapshot: serializeTask(task),
      related_counts: task._count,
      staged_generated_paths: staged.map((item) => path.relative(workspaceRoot, item.source).replaceAll("\\", "/")),
      purged_at: new Date().toISOString()
    });
    const record = await logAction({
      action: "purge_task",
      taskId,
      message: `Task purge transaction accepted: ${taskId}`,
      artifactPath: tombstone,
      metadata: { purged_task_id: taskId, related_counts: task._count, purge_id: purgeId }
    });
    purgeActionId = record.action_id;
    await prisma.task.delete({ where: { id: taskId } });
    dbDeleted = true;
    await prisma.actionLog.update({ where: { id: record.action_id }, data: { message: `Task permanently purged: ${taskId}` } }).catch(() => undefined);
    const cleanupError = await fs.rm(stagingRoot, { recursive: true, force: true }).then(() => null).catch((error) => error instanceof Error ? error.message : String(error));
    const remaining = await Promise.all(generatedRoots.map(async (target) => ({
      path: path.relative(workspaceRoot, target).replaceAll("\\", "/"),
      exists: await fs.stat(target).then(() => true).catch(() => false)
    })));
    return NextResponse.json({
      ok: true,
      ...record,
      task_id: taskId,
      purged: true,
      tombstone,
      related_counts: task._count,
      generated_paths: remaining,
      zero_generated_residual: remaining.every((item) => !item.exists),
      cleanup_pending: Boolean(cleanupError),
      cleanup_error: cleanupError,
      cleanup_staging_path: cleanupError ? path.relative(workspaceRoot, stagingRoot).replaceAll("\\", "/") : null
    });
  } catch (error) {
    if (!dbDeleted) {
      for (const item of [...staged].reverse()) {
        const stagedExists = await fs.stat(item.target).then(() => true).catch(() => false);
        if (stagedExists) {
          await fs.mkdir(path.dirname(item.source), { recursive: true });
          await fs.rename(item.target, item.source).catch(() => undefined);
        }
      }
    }
    if (purgeActionId) {
      await prisma.actionLog.update({ where: { id: purgeActionId }, data: { message: dbDeleted ? `Task purged; generated-path cleanup requires recovery: ${taskId}` : `Task purge failed and staged paths were restored: ${taskId}` } }).catch(() => undefined);
    }
    return NextResponse.json({ ok: false, error: error instanceof Error ? error.message : "Task purge failed; staged paths were restored." }, { status: 500 });
  }
}
