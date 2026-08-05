import { NextResponse } from "next/server";
import { readPublicScientificReport } from "@/lib/server/public-scientific-report";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const taskId = url.searchParams.get("task_id") ?? "";
  const runId = url.searchParams.get("run_id") ?? "";
  try {
    const report = await readPublicScientificReport(taskId, runId);
    return NextResponse.json(
      { ok: true, task_id: report.task_id, report },
      { headers: { "Cache-Control": "no-store, max-age=0", "X-Content-Type-Options": "nosniff" } },
    );
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Public scientific report unavailable." },
      { status: 404 },
    );
  }
}
