import { promises as fs } from "node:fs";
import { cookies } from "next/headers";
import { NextResponse } from "next/server";
import { SESSION_COOKIE, sessionPrincipal } from "@/lib/server/local-session";
import { issueTenantEnrollmentIdentity, tenantPrincipal } from "@/lib/server/tenant-byoa";
import { managedUtcFromEnvironment, type ClockAnchor } from "@/lib/server/service-clock";

export const dynamic = "force-dynamic";

async function managedUtcNowMs() {
  return managedUtcFromEnvironment({nodeEnv:process.env.NODE_ENV,platform:process.platform,anchorPath:process.env.EVOMIND_SERVICE_CLOCK_ANCHOR_PATH,instance:process.env.EVOMIND_SERVICE_INSTANCE_ID,boot:process.env.EVOMIND_BOOT_ID,qpc:process.env.EVOMIND_CLOCK_QPC_TIMESTAMP,frequency:process.env.EVOMIND_CLOCK_QPC_FREQUENCY,epoch:process.env.EVOMIND_CLOCK_EPOCH_MS},async(anchorPath)=>JSON.parse(await fs.readFile(anchorPath,"utf8")) as ClockAnchor,process.hrtime.bigint(),Date.now);
}

export async function POST(request: Request) {
  try {
    const identity = tenantPrincipal(sessionPrincipal((await cookies()).get(SESSION_COOKIE)?.value));
    const body = await request.json() as Record<string, unknown>;
    if ("allocation_generation" in body || "profile_instance_id" in body || "nonce" in body) {
      throw new Error("server_issued_identity_required");
    }
    const jobId = Number(body.job_id);
    const allocationBindingId = String(body.allocation_binding_id ?? "");
    if (!Number.isSafeInteger(jobId) || jobId < 1 || !/^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$/.test(allocationBindingId)) {
      throw new Error("allocation_identity_invalid");
    }
    const issued = await issueTenantEnrollmentIdentity(identity, {
      job_id: jobId,
      allocation_binding_id: allocationBindingId,
      authoritative_now_ms: (await managedUtcNowMs()).utc_ms,
    });
    return NextResponse.json({
      ok: true,
      reservation_id: issued.reservation_id,
      nonce: issued.nonce,
      job_id: issued.job_id,
      allocation_binding_id: issued.allocation_binding_id,
      allocation_generation: issued.allocation_generation,
      credential_profile: issued.credential_profile,
      profile_instance_id: issued.profile_instance_id,
      expires_at_utc: issued.expires_at_utc,
      server_issued: true,
      secrets_echoed: false,
    }, { status: 201, headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    return NextResponse.json({
      ok: false,
      code: error instanceof Error ? error.message : "enrollment_identity_rejected",
    }, { status: 400, headers: { "Cache-Control": "no-store" } });
  }
}

