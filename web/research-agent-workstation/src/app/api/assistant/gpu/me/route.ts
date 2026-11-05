import { proxyAssistantRuntime } from "@/lib/server/assistant-runtime";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// t238 C7c-6: the caller's own GPU training state (queue position, ETA estimate, budget). Runtime scopes by principal.
export function GET(request: Request) {
  const runId = new URL(request.url).searchParams.get("run_id") ?? "";
  if (runId && !/^[A-Za-z0-9_-]{1,120}$/.test(runId)) return Response.json({ error: "invalid_run_id" }, { status: 400 });
  return proxyAssistantRuntime(request, runId ? `/v1/gpu/me?run_id=${encodeURIComponent(runId)}` : "/v1/gpu/me");
}
