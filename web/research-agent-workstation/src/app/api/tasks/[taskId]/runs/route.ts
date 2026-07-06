import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { ensureWorkstationSeeded } from "@/lib/server/bootstrap";
import { normalizeTaskId } from "@/lib/server/paths";
import { serializeRun } from "@/lib/server/serializers";

export const dynamic = "force-dynamic";

export async function GET(_request: Request, { params }: { params: { taskId: string } }) {
  await ensureWorkstationSeeded();
  const taskId = normalizeTaskId(params.taskId);
  const runs = await prisma.experimentRun.findMany({ where: { taskId }, orderBy: { createdAt: "desc" } });
  return NextResponse.json({ ok: true, task_id: taskId, runs: runs.map(serializeRun) });
}
