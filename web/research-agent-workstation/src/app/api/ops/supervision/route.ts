import { NextResponse } from "next/server";
import { verifiedOperatorAccess } from "@/lib/server/legacy-gpu-access";
import { assistantRuntimeFetch } from "@/lib/server/assistant-runtime";

export const dynamic = "force-dynamic";

// Run supervision v1: runs that wait for a person, model-slot use and failure categories. Operator only.
const SUPERVISION_TIMEOUT_MS = 5000;

export async function GET(request: Request) {
  const access = verifiedOperatorAccess(request);
  if (!access.ok) {
    return NextResponse.json({ ok: false, code: access.status === 401 ? "session_required" : "operator_required" },
      { status: access.status, headers: { "Cache-Control": "no-store" } });
  }
  try {
    const response = await assistantRuntimeFetch("/v1/ops/supervision", { signal: AbortSignal.timeout(SUPERVISION_TIMEOUT_MS) });
    if (!response.ok) throw new Error("runtime_supervision_failed");
    return NextResponse.json(await response.json(), { headers: { "Cache-Control": "no-store" } });
  } catch {
    return NextResponse.json({ ok: false, code: "runtime_unreachable" }, { status: 503, headers: { "Cache-Control": "no-store" } });
  }
}
