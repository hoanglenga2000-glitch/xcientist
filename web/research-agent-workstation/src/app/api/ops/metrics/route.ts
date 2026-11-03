import { NextResponse } from "next/server";
import { verifiedOperatorAccess } from "@/lib/server/legacy-gpu-access";
import { assistantRuntimeFetch } from "@/lib/server/assistant-runtime";

export const dynamic = "force-dynamic";

// t238 C7e O3: Prometheus text from the runtime, operator-only (same boundary as /api/healthz?deep=1).
const METRICS_TIMEOUT_MS = 900;

export async function GET(request: Request) {
  const access = verifiedOperatorAccess(request);
  if (!access.ok) {
    return NextResponse.json({ ok: false, code: access.status === 401 ? "session_required" : "operator_required" },
      { status: access.status, headers: { "Cache-Control": "no-store" } });
  }
  try {
    const response = await assistantRuntimeFetch("/v1/metrics", { headers: { Accept: "text/plain" }, signal: AbortSignal.timeout(METRICS_TIMEOUT_MS) });
    if (!response.ok) throw new Error("runtime_metrics_failed");
    return new NextResponse(await response.text(), {
      status: 200,
      headers: { "Content-Type": "text/plain; version=0.0.4; charset=utf-8", "Cache-Control": "no-store" },
    });
  } catch {
    return new NextResponse("# runtime unreachable\n", { status: 503, headers: { "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-store" } });
  }
}
