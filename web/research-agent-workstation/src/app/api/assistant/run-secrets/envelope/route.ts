import { cookies } from "next/headers";

import { currentAssistantManagedHpcIdentity } from "@/lib/server/assistant-managed-hpc";
import { storeRunSecretEnvelope } from "@/lib/server/assistant-run-secrets";
import { SESSION_COOKIE, sessionPrincipal } from "@/lib/server/local-session";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  try {
    const body = await request.json() as Record<string, unknown>;
    const principal = sessionPrincipal((await cookies()).get(SESSION_COOKIE)?.value);
    const result = await storeRunSecretEnvelope(principal, await currentAssistantManagedHpcIdentity(), body);
    return Response.json({ ok: true, ...result }, { status: 201, headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    return Response.json({ ok: false, code: error instanceof Error ? error.message : "run_secret_envelope_rejected" }, { status: 400, headers: { "Cache-Control": "no-store" } });
  }
}
