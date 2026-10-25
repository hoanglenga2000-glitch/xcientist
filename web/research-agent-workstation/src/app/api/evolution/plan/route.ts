import { NextResponse } from "next/server";
import { evolutionErrorBody } from "@/lib/evolution-cli-path";
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
    const failure = evolutionErrorBody(error, { task_id: taskId }, "evolution plan failed");
    return NextResponse.json(failure.body, { status: failure.status });
  }
}
