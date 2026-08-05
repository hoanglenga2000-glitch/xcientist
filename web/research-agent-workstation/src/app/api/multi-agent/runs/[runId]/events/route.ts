import { promises as fs } from "node:fs";
import { NextResponse } from "next/server";
import { sanitizeClientJson } from "@/lib/server/json";
import { resolveWorkspacePath } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

export async function GET(request: Request, { params }: { params: Promise<{ runId: string }> }) {
  const { runId } = await params;
  if (!/^[A-Za-z0-9_-]{8,160}$/.test(runId)) return NextResponse.json({ ok: false, error: "invalid run id" }, { status: 400 });
  const runDir = resolveWorkspacePath(`workspace/evomind_runs/${runId}`);
  if (!await fs.stat(runDir).then((value) => value.isDirectory()).catch(() => false)) {
    return NextResponse.json({ ok: false, error: "run not found", run_id: runId }, { status: 404 });
  }
  const afterSeq = Math.max(0, Number.parseInt(new URL(request.url).searchParams.get("after_seq") ?? "0", 10) || 0);
  const text = await fs.readFile(`${runDir}/events.jsonl`, "utf-8").catch(() => "");
  const events = text.split(/\r?\n/).filter(Boolean).map((line) => {
    try { return JSON.parse(line) as Record<string, unknown>; } catch { return null; }
  }).filter((event): event is Record<string, unknown> => Boolean(event))
    .filter((event) => event.run_id === runId && Number(event.seq ?? 0) > afterSeq)
    .sort((left, right) => Number(left.seq ?? 0) - Number(right.seq ?? 0));
  return NextResponse.json(sanitizeClientJson({ ok: true, run_id: runId, after_seq: afterSeq, last_seq: events.at(-1)?.seq ?? afterSeq, events }));
}
