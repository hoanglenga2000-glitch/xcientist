import { cookies } from "next/headers";
import { assistantRuntimeFetch, proxyAssistantRuntime } from "@/lib/server/assistant-runtime";
import { currentAssistantManagedHpcIdentity } from "@/lib/server/assistant-managed-hpc";
import { validateRunSecretRefs } from "@/lib/server/assistant-run-secrets";
import { SESSION_COOKIE, sessionPrincipal } from "@/lib/server/local-session";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const incoming = new URL(request.url);
  const query = new URLSearchParams();
  const conversationId = incoming.searchParams.get("conversation_id");
  const limit = incoming.searchParams.get("limit");
  if (conversationId) query.set("conversation_id", conversationId);
  if (limit) query.set("limit", limit);
  return proxyAssistantRuntime(request, `/v1/runs${query.size ? `?${query}` : ""}`);
}

export async function POST(request: Request) {
  let body: Record<string, unknown>;
  try {
    const parsed = await request.json() as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("json_object_required");
    body = parsed as Record<string, unknown>;
  } catch {
    return Response.json({ ok: false, error: "invalid_json" }, { status: 400 });
  }
  const bodyKey = body.idempotency_key;
  const headerKey = request.headers.get("Idempotency-Key") ?? "";
  if ((bodyKey !== undefined && typeof bodyKey !== "string") || (bodyKey && headerKey && bodyKey !== headerKey)) {
    return Response.json({ ok: false, error: "invalid_idempotency_key" }, { status: 400 });
  }
  const key = String(bodyKey || headerKey);
  if (key && !/^[A-Za-z0-9_-]{8,128}$/.test(key)) {
    return Response.json({ ok: false, error: "invalid_idempotency_key" }, { status: 400 });
  }
  body.idempotency_key = key;
  try {
    if (key) {
      // Replay before inspecting expiring/consumed one-shot references. This
      // endpoint compares the original client payload and cannot start a Run.
      const replay = await assistantRuntimeFetch("/v1/run-requests/replay", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body), signal: request.signal,
      });
      if (replay.status !== 404) return new Response(replay.body, {
        status: replay.status, headers: { "Content-Type": "application/json", "Cache-Control": "private, no-store" },
      });
    }
    const serverBoundIdentity = { managed_hpc_identity: await currentAssistantManagedHpcIdentity() };
    const managedHpcIdentity = serverBoundIdentity.managed_hpc_identity;
    const principal = sessionPrincipal((await cookies()).get(SESSION_COOKIE)?.value);
    const secretRefs = await validateRunSecretRefs(principal, managedHpcIdentity, body.secret_refs);
    const response = await assistantRuntimeFetch("/v1/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...body,
        secret_refs: secretRefs,
        ...serverBoundIdentity,
      }),
      signal: request.signal,
    });
    return new Response(response.body, {
      status: response.status,
      headers: {
        "Content-Type": response.headers.get("content-type") ?? "application/json; charset=utf-8",
        "Cache-Control": "private, no-store",
      },
    });
  } catch {
    return Response.json({ ok: false, error: "runtime_not_ready" }, { status: 503 });
  }
}
