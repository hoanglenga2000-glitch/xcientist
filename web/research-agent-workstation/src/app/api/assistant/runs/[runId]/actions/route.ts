import { cookies } from "next/headers";

import { currentAssistantManagedHpcIdentity } from "@/lib/server/assistant-managed-hpc";
import { validateRunSecretRefs } from "@/lib/server/assistant-run-secrets";
import { assistantRuntimeFetch } from "@/lib/server/assistant-runtime";
import { SESSION_COOKIE, sessionPrincipal } from "@/lib/server/local-session";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: Request, context: { params: Promise<{ runId: string }> }) {
  const { runId } = await context.params;
  try {
    const body = await request.json() as Record<string, unknown>;
    const serverBody = { ...body };
    delete serverBody.managed_hpc_identity;
    const requiresHpcIdentity = body.action === "rebind_managed_hpc_identity" || body.action === "attach_secret_refs"
      || (Array.isArray(body.secret_refs) && body.secret_refs.length > 0);
    const managedHpcIdentity = requiresHpcIdentity ? await currentAssistantManagedHpcIdentity() : null;
    const principal = sessionPrincipal((await cookies()).get(SESSION_COOKIE)?.value);
    const secretRefs = await validateRunSecretRefs(principal, managedHpcIdentity, body.secret_refs);
    const response = await assistantRuntimeFetch(`/v1/runs/${encodeURIComponent(runId)}/actions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...serverBody,
        idempotency_key: typeof body.idempotency_key === "string" ? body.idempotency_key : "",
        secret_refs: secretRefs,
        managed_hpc_identity: managedHpcIdentity,
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
    return Response.json({ ok: false, error: "run_secret_ref_rejected" }, { status: 400 });
  }
}
