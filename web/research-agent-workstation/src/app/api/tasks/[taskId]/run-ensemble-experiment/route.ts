import { NextResponse } from "next/server";
import { runEnsembleExperiment } from "@/lib/server/runs";

export const dynamic = "force-dynamic";

type RouteContext = { params: Promise<{ taskId: string }> };

async function parseOptions(request: Request) {
  const body = await request.json().catch(() => ({}));
  const sampleRowsRaw = Number(body?.sampleRows ?? body?.sample_rows ?? 0);
  const nFoldsRaw = Number(body?.nFolds ?? body?.n_folds ?? 0);
  const timeoutMsRaw = Number(body?.timeoutMs ?? body?.timeout_ms ?? 0);
  const timeoutSecondsRaw = Number(body?.timeoutSeconds ?? body?.timeout_seconds ?? 0);
  const timeoutMs =
    Number.isFinite(timeoutMsRaw) && timeoutMsRaw > 0
      ? Math.floor(timeoutMsRaw)
      : Number.isFinite(timeoutSecondsRaw) && timeoutSecondsRaw > 0
        ? Math.floor(timeoutSecondsRaw * 1000)
        : undefined;
  const seedsRaw = typeof body?.seeds === "string" ? body.seeds.trim() : "";
  const seeds = /^[0-9,\s]+$/.test(seedsRaw) ? seedsRaw.replace(/\s+/g, "") : "";
  return {
    fast: body?.fast === true,
    sampleRows: Number.isFinite(sampleRowsRaw) && sampleRowsRaw > 0 ? Math.floor(sampleRowsRaw) : undefined,
    nFolds: Number.isFinite(nFoldsRaw) && nFoldsRaw > 0 ? Math.floor(nFoldsRaw) : undefined,
    seeds: seeds || undefined,
    timeoutMs
  };
}

export async function POST(request: Request, { params }: RouteContext) {
  const { taskId } = await params;
  try {
    const options = await parseOptions(request);
    return NextResponse.json(await runEnsembleExperiment(taskId, options));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown ensemble run error";
    return NextResponse.json({ ok: false, task_id: taskId, error: message }, { status: 500 });
  }
}
