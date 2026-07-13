import { NextResponse } from "next/server";
import { planEvolution } from "@/lib/server/evolution";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  const body = (await request.json().catch(() => ({}))) as Record<string, unknown>;
  const taskId = typeof body.task_id === "string" ? body.task_id : "";
  try {
    // official_submit_allowed is forced false regardless of the request body.
    const payload = await planEvolution({ ...body, official_submit_allowed: false });
    return NextResponse.json({ ok: true, ...payload });
  } catch (error) {
    const message = error instanceof Error ? error.message : "evolution plan failed";
    return NextResponse.json({ ok: false, task_id: taskId, error: message }, { status: 400 });
  }
}
