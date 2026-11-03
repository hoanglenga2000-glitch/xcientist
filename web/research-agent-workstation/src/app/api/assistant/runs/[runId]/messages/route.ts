import { assistantRuntimeFetch } from "@/lib/server/assistant-runtime";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** C6b L10–L11: queue / immediate-send / cancel_queue while a run is live. */
export async function POST(request: Request, context: { params: Promise<{ runId: string }> }) {
  const { runId } = await context.params;
  const body = await request.json().catch(() => ({})) as Record<string, unknown>;
  const mode = typeof body.mode === "string" ? body.mode : "";
  const text = typeof body.text === "string" ? body.text : "";
  const idempotency_key = typeof body.idempotency_key === "string" ? body.idempotency_key : "";
  const response = await assistantRuntimeFetch(`/v1/runs/${encodeURIComponent(runId)}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode, text, idempotency_key }),
    signal: request.signal,
  });
  return new Response(response.body, {
    status: response.status,
    headers: {
      "Content-Type": response.headers.get("content-type") ?? "application/json; charset=utf-8",
      "Cache-Control": "private, no-store",
    },
  });
}
