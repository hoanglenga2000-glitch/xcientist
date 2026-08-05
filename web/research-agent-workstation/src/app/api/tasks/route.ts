import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { ensureWorkstationSeeded } from "@/lib/server/bootstrap";
import { serializeTask } from "@/lib/server/serializers";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  await ensureWorkstationSeeded();
  const includeArchived = new URL(request.url).searchParams.get("include_archived") === "1";
  const tasks = await prisma.task.findMany({
    where: includeArchived ? undefined : { status: { not: "archived" } },
    orderBy: { updatedAt: "desc" }
  });
  return NextResponse.json({ ok: true, tasks: tasks.map(serializeTask) });
}
