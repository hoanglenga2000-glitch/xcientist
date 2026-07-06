import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { ensureWorkstationSeeded } from "@/lib/server/bootstrap";
import { serializeTask } from "@/lib/server/serializers";

export const dynamic = "force-dynamic";

export async function GET() {
  await ensureWorkstationSeeded();
  const tasks = await prisma.task.findMany({ orderBy: { updatedAt: "desc" } });
  return NextResponse.json({ ok: true, tasks: tasks.map(serializeTask) });
}
