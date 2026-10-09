import { cookies } from "next/headers";

import { currentAssistantManagedHpcIdentity } from "@/lib/server/assistant-managed-hpc";
import { issueRunSecretIntent } from "@/lib/server/assistant-run-secrets";
import { SESSION_COOKIE, sessionPrincipal } from "@/lib/server/local-session";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  try {
    const body = await request.json() as Record<string, unknown>;
    if (Object.keys(body).length !== 1 || !("purpose" in body)) throw new Error("run_secret_intent_fields_invalid");
    const principal = sessionPrincipal((await cookies()).get(SESSION_COOKIE)?.value);
    const intent = await issueRunSecretIntent(principal, await currentAssistantManagedHpcIdentity(), body.purpose);
    return Response.json({ ok: true, ...intent, secrets_echoed: false }, { status: 201, headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    return Response.json({ ok: false, code: error instanceof Error ? error.message : "run_secret_intent_rejected" }, { status: 400, headers: { "Cache-Control": "no-store" } });
  }
}
