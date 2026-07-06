import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { ensureWorkstationSeeded } from "@/lib/server/bootstrap";
import { normalizeTaskId } from "@/lib/server/paths";
import { serializeWorkflow } from "@/lib/server/serializers";

export const dynamic = "force-dynamic";

export async function GET(_request: Request, { params }: { params: { taskId: string } }) {
  await ensureWorkstationSeeded();
  const taskId = normalizeTaskId(params.taskId);
  const workflow = await prisma.workflow.findFirst({ where: { taskId }, orderBy: { updatedAt: "desc" } });
  return NextResponse.json({ ok: true, task_id: taskId, workflow: serializeWorkflow(workflow) });
}
