import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { ensureWorkstationSeeded } from "@/lib/server/bootstrap";
import { normalizeTaskId } from "@/lib/server/paths";
import { serializeGate } from "@/lib/server/serializers";

export const dynamic = "force-dynamic";

export async function GET(_request: Request, { params }: { params: Promise<{ taskId: string }> }) {
  await ensureWorkstationSeeded();
  const { taskId: rawTaskId } = await params;
  const taskId = normalizeTaskId(rawTaskId);
  const gates = await prisma.gate.findMany({ where: { taskId }, orderBy: { createdAt: "desc" } });
  return NextResponse.json({ ok: true, task_id: taskId, gates: gates.map(serializeGate) });
}
