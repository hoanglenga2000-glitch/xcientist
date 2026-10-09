import { promises as fs } from "node:fs";
import { createHash } from "node:crypto";
import path from "node:path";
import { cookies } from "next/headers";
import { NextResponse } from "next/server";
import { SESSION_COOKIE, sessionPrincipal } from "@/lib/server/local-session";
import { consumeTenantEnrollmentIdentity, tenantByoaRoot, tenantPrincipal } from "@/lib/server/tenant-byoa";
import { managedUtcFromEnvironment, type ClockAnchor } from "@/lib/server/service-clock";
export const dynamic = "force-dynamic";
async function managedUtcNowMs() {
  return managedUtcFromEnvironment({nodeEnv:process.env.NODE_ENV,platform:process.platform,anchorPath:process.env.EVOMIND_SERVICE_CLOCK_ANCHOR_PATH,instance:process.env.EVOMIND_SERVICE_INSTANCE_ID,boot:process.env.EVOMIND_BOOT_ID,qpc:process.env.EVOMIND_CLOCK_QPC_TIMESTAMP,frequency:process.env.EVOMIND_CLOCK_QPC_FREQUENCY,epoch:process.env.EVOMIND_CLOCK_EPOCH_MS},async(anchorPath)=>JSON.parse(await fs.readFile(anchorPath,"utf8")) as ClockAnchor,process.hrtime.bigint(),Date.now);
}
export async function GET(){try{const clock=await managedUtcNowMs();return NextResponse.json({ok:true,time_source:clock.source,utc_epoch_ms:clock.utc_ms,helper_spawned:false,secrets_echoed:false})}catch(error){return NextResponse.json({ok:false,code:error instanceof Error?error.message:"clock_probe_rejected"},{status:503})}}
export async function POST(request: Request) {
  try {
    const identity = tenantPrincipal(sessionPrincipal((await cookies()).get(SESSION_COOKIE)?.value)); const body = await request.json() as Record<string, unknown>;
    if (["password", "proxy_password", "hpc_password", "secret"].some((key) => key in body)) throw new Error("plaintext_secret_rejected");
    if (["nonce", "expires_at_utc", "allocation_generation", "profile_instance_id"].some((key) => key in body)) throw new Error("server_issued_identity_required");
    const envelope = body.encrypted_envelope && typeof body.encrypted_envelope === "object" ? body.encrypted_envelope as Record<string, unknown> : {}; const reservationId=String(body.reservation_id??"").toLowerCase();const authoritativeClock = await managedUtcNowMs(); const authoritativeNow=authoritativeClock.utc_ms;
    if (envelope.schema !== "evomind.hybrid_tenant_envelope.v1" || envelope.algorithm !== "RSA-OAEP-256+A256GCM" || typeof envelope.key_id !== "string" || !/^[a-f0-9]{64}$/.test(envelope.key_id) || !["wrapped_key","iv","ciphertext","tag"].every((key)=>typeof envelope[key] === "string" && /^[A-Za-z0-9+/=]{16,65536}$/.test(String(envelope[key]))) || !/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(reservationId)) throw new Error("encrypted_envelope_invalid");
    const keyRecord = JSON.parse(await fs.readFile(path.join(tenantByoaRoot(), "enrollment-keys", "current-public-key.json"), "utf8")) as Record<string, unknown>;
    if (keyRecord.key_id !== envelope.key_id || keyRecord.algorithm !== envelope.algorithm || Date.parse(String(keyRecord.expires_at_utc)) <= authoritativeNow) throw new Error("enrollment_key_invalid");
    const jobId = Number(body.job_id); const allocationBindingId=String(body.allocation_binding_id??"");
    if (!Number.isSafeInteger(jobId) || jobId < 1 || !/^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$/.test(allocationBindingId)) throw new Error("allocation_identity_invalid");
    const cipher = Buffer.from(`${JSON.stringify(envelope)}\n`, "utf8");
    const dir = path.join(tenantByoaRoot(), "inbox"); await fs.mkdir(dir, { recursive: true, mode: 0o700 });
    let target="",receipt="",stagedReceipt="",queuedBinding="",trigger="",serverExpiresAt=0;
    try {
      const consumed = await consumeTenantEnrollmentIdentity(identity, reservationId, { key_id: String(envelope.key_id), job_id: jobId, allocation_binding_id: allocationBindingId,authoritative_now_ms:authoritativeNow }, async(reservation,prospectiveBinding)=>{
        const nonce=reservation.nonce;const generation=reservation.allocation_generation;const profileInstanceId=reservation.profile_instance_id;
        target=path.join(dir,`${nonce}.envelope`);receipt=path.join(dir,`${nonce}.receipt.json`);stagedReceipt=path.join(dir,`${nonce}.receipt.staged.json`);queuedBinding=path.join(dir,`${nonce}.binding.json`);
        const triggerDir=path.join(tenantByoaRoot(),"queue-triggers");trigger=path.join(triggerDir,`${nonce}.trigger`);
        if(await fs.stat(receipt).then(()=>true).catch(()=>false)||await fs.stat(trigger).then(()=>true).catch(()=>false))throw new Error("uncommitted_enrollment_publish_marker_rejected");
        for(const orphan of [target,stagedReceipt,queuedBinding])await fs.rm(orphan,{force:true});
        await fs.writeFile(target,cipher,{flag:"wx",mode:0o600});
        const envelopeStat=await fs.stat(target);const issuedAt=(await managedUtcNowMs()).utc_ms;
        if(!Number.isFinite(envelopeStat.birthtimeMs)||!Number.isFinite(envelopeStat.mtimeMs)||Math.abs(envelopeStat.mtimeMs-envelopeStat.birthtimeMs)>60_000||Math.abs(issuedAt-authoritativeNow)>60_000)throw new Error("server_clock_evidence_invalid");
        serverExpiresAt = issuedAt + 10 * 60_000;
        await fs.writeFile(stagedReceipt,`${JSON.stringify({schema:"evomind.tenant_enrollment_envelope.v1",reservation_id:reservationId,tenant_id:identity.tenantId,owner_principal_id:identity.username,nonce,issued_at_utc:new Date(issuedAt).toISOString(),expires_at_utc:new Date(serverExpiresAt).toISOString(),issued_at_unix_ms:issuedAt,expires_at_unix_ms:serverExpiresAt,ciphertext_sha256:createHash("sha256").update(cipher).digest("hex"),key_id:envelope.key_id,algorithm:envelope.algorithm,job_id:jobId,allocation_generation:generation,profile_instance_id:profileInstanceId})}\n`,{flag:"wx",mode:0o600});
        await fs.writeFile(queuedBinding,`${JSON.stringify(prospectiveBinding)}\n`,{flag:"wx",mode:0o600});
      });
      const {reservation,binding}=consumed;const nonce=reservation.nonce;const generation=reservation.allocation_generation;const profileInstanceId=reservation.profile_instance_id;
      await fs.link(stagedReceipt,receipt);await fs.rm(stagedReceipt,{force:true});
      const triggerDir=path.join(tenantByoaRoot(),"queue-triggers");await fs.mkdir(triggerDir,{recursive:true,mode:0o700});await fs.writeFile(trigger,`${JSON.stringify({schema:"evomind.tenant_enrollment_queue_trigger.v1",opaque_id:nonce,created_at_utc:new Date().toISOString()})}\n`,{flag:"wx",mode:0o600});
      return NextResponse.json({ ok: true, tenant_id: identity.tenantId, state: binding.state, allocation_generation:generation,profile_instance_id:profileInstanceId,expires_at_utc: new Date(serverExpiresAt).toISOString(), clock_source: authoritativeClock.source, secrets_echoed: false }, { status: 202 });
    } catch (error) {
      const consumedExists=await fs.stat(path.join(tenantByoaRoot(),"tenants",identity.tenantId,"enrollment-reservations",`${reservationId}.consumed.json`)).then(()=>true).catch(()=>false);
      if(!consumedExists){for(const candidate of [target,receipt,stagedReceipt,queuedBinding,trigger])if(candidate)await fs.rm(candidate,{force:true});}
      else{
        const recoveryDir=path.join(tenantByoaRoot(),"enrollment-recovery");await fs.mkdir(recoveryDir,{recursive:true,mode:0o700});
        await fs.writeFile(path.join(recoveryDir,`${reservationId}.json`),`${JSON.stringify({schema:"evomind.tenant_enrollment_publish_recovery.v1",reservation_id:reservationId,tenant_id:identity.tenantId,status:"committed_pending_watcher_recovery",recorded_at_utc:new Date(authoritativeNow).toISOString(),secrets_echoed:false})}\n`,{flag:"wx",mode:0o600}).catch((writeError:NodeJS.ErrnoException)=>{if(writeError.code!=="EEXIST")throw writeError});
      }
      throw error;
    }
  } catch (error) { return NextResponse.json({ ok: false, code: error instanceof Error ? error.message : "enrollment_rejected" }, { status: 400 }); }
}

