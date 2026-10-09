import { cookies } from "next/headers";
import { NextResponse } from "next/server";
import { SESSION_COOKIE, sessionPrincipal } from "@/lib/server/local-session";
import { promises as fs } from "node:fs";
import path from "node:path";
import { activateTenantBinding, freezeOrRetireBinding, readTenantBinding, tenantByoaRoot, tenantPrincipal } from "@/lib/server/tenant-byoa";
import { transitionTenantHpcProfile } from "@/lib/server/tenant-hpc-profile-lifecycle";
export const dynamic = "force-dynamic";
function publicBinding(binding: Awaited<ReturnType<typeof readTenantBinding>>) {
  return binding ? {
    state: binding.state,
    reason: binding.reason,
    job_id: binding.job_id,
    allocation_binding_id: binding.allocation_binding_id,
    credential_profile: binding.credential_profile,
    allocation_generation: binding.allocation_generation,
    updated_at_utc: binding.updated_at_utc,
  } : null;
}
type PublicActivationStatus = {
  state: string;
  attempts: number;
  next_attempt_at_utc: string | null;
  updated_at_utc: string | null;
  error_code: string | null;
  receipt_exists: boolean;
  evidence_exists: boolean;
};
const PUBLIC_ACTIVATION_STATES = new Set([
  "pending_activation",
  "retryable_activation_failure",
  "completed",
  "terminal_consumption_failure",
  "consumption_state_unknown",
  "terminal_generation_tombstoned",
]);
function safeTimestamp(value: unknown) {
  if (typeof value !== "string" || value.length > 64 || Number.isNaN(Date.parse(value))) return null;
  return value;
}
function safeActivationError(value: unknown) {
  if (typeof value !== "string" || !/^[A-Za-z0-9_.:-]{1,160}$/.test(value)) return null;
  return value;
}
async function publicActivationStatus(
  identity: ReturnType<typeof tenantPrincipal>,
  binding: NonNullable<Awaited<ReturnType<typeof readTenantBinding>>>,
): Promise<PublicActivationStatus | null> {
  const tenantRoot = path.join(tenantByoaRoot(), "tenants", identity.tenantId);
  const requestRoot = path.join(tenantRoot, "activation-requests");
  const prefix = `g${binding.allocation_generation}.`;
  const names = await fs.readdir(requestRoot).catch(() => [] as string[]);
  const matches = names.filter((name) => name.startsWith(prefix) && name.endsWith(".json"));
  if (matches.length !== 1) return null;
  const request = JSON.parse(await fs.readFile(path.join(requestRoot, matches[0]), "utf8")) as Record<string, unknown>;
  const state = typeof request.state === "string" && PUBLIC_ACTIVATION_STATES.has(request.state) ? request.state : null;
  const matchesBinding = (
    request.tenant_id === identity.tenantId
    && request.owner_principal_id === identity.username
    && request.job_id === binding.job_id
    && request.credential_profile === binding.credential_profile
    && request.allocation_binding_id === binding.allocation_binding_id
    && request.allocation_generation === binding.allocation_generation
    && request.profile_instance_id === binding.profile_instance_id
  );
  if (!state || !matchesBinding) return null;
  const attempts = Number.isSafeInteger(request.attempts) && Number(request.attempts) >= 0 ? Number(request.attempts) : 0;
  const receipt = path.join(tenantRoot, "activation-receipts", `g${binding.allocation_generation}.json`);
  const evidence = path.join(tenantRoot, "activation-evidence", `g${binding.allocation_generation}.json`);
  return {
    state,
    attempts,
    next_attempt_at_utc: safeTimestamp(request.next_attempt_at_utc),
    updated_at_utc: safeTimestamp(request.updated_at_utc),
    error_code: safeActivationError(request.error_code),
    receipt_exists: await fs.stat(receipt).then(() => true).catch(() => false),
    evidence_exists: await fs.stat(evidence).then(() => true).catch(() => false),
  };
}
async function assertReplaceableFailedProvisioning(identity: ReturnType<typeof tenantPrincipal>, current: NonNullable<Awaited<ReturnType<typeof readTenantBinding>>>) {
  if (current.state !== "provisioning") throw new Error("tenant_binding_not_failed_provisioning");
  const tenantRoot = path.join(tenantByoaRoot(), "tenants", identity.tenantId);
  const requestRoot = path.join(tenantRoot, "activation-requests");
  const prefix = `g${current.allocation_generation}.`;
  const names = (await fs.readdir(requestRoot)).filter((name) => name.startsWith(prefix) && name.endsWith(".json"));
  if (names.length !== 1) throw new Error("failed_provisioning_request_not_unique");
  const request = JSON.parse(await fs.readFile(path.join(requestRoot, names[0]), "utf8")) as Record<string, unknown>;
  if (
    request.state !== "retryable_activation_failure"
    || request.tenant_id !== identity.tenantId
    || request.owner_principal_id !== identity.username
    || request.job_id !== current.job_id
    || request.credential_profile !== current.credential_profile
    || request.allocation_binding_id !== current.allocation_binding_id
    || request.allocation_generation !== current.allocation_generation
    || request.profile_instance_id !== current.profile_instance_id
    || !Number.isSafeInteger(request.attempts) || Number(request.attempts) < 1
    || request.secrets_present === true
  ) throw new Error("failed_provisioning_request_not_replaceable");
  const receipt = path.join(tenantRoot, "activation-receipts", `g${current.allocation_generation}.json`);
  const evidence = path.join(tenantRoot, "activation-evidence", `g${current.allocation_generation}.json`);
  if (await fs.stat(receipt).then(() => true).catch(() => false)) throw new Error("failed_provisioning_receipt_exists");
  if (await fs.stat(evidence).then(() => true).catch(() => false)) throw new Error("failed_provisioning_identity_evidence_exists");
}
export async function GET() {
  try { const identity = tenantPrincipal(sessionPrincipal((await cookies()).get(SESSION_COOKIE)?.value)); const binding = await readTenantBinding(identity); const activation = binding ? await publicActivationStatus(identity, binding) : null; return NextResponse.json({ ok: true, tenant_id: identity.tenantId, binding: publicBinding(binding), activation }, { headers: { "Cache-Control": "no-store" } }); }
  catch { return NextResponse.json({ ok: false, code: "tenant_session_required" }, { status: 401 }); }
}
export async function POST(request: Request) {
  try {
    const identity=tenantPrincipal(sessionPrincipal((await cookies()).get(SESSION_COOKIE)?.value));const body=await request.json() as Record<string,unknown>;const action=String(body.action??"");
    if(action==="activate"){
      const current=await readTenantBinding(identity);if(!current)throw new Error("tenant_binding_missing");const receiptPath=path.join(tenantByoaRoot(),"tenants",identity.tenantId,"activation-receipts",`g${current.allocation_generation}.json`);const receipt=JSON.parse(await fs.readFile(receiptPath,"utf8")) as Record<string,unknown>;
      return NextResponse.json({ok:true,binding:await activateTenantBinding(identity,receipt)},{headers:{"Cache-Control":"no-store"}});
    }
    if(action==="replace_failed_provisioning"){
      const current=await readTenantBinding(identity);if(!current)throw new Error("tenant_binding_missing");
      await assertReplaceableFailedProvisioning(identity,current);
      const reason="failed_provisioning_replaced";const state="retired";
      const profileTransition=await transitionTenantHpcProfile(current,state,reason);const observed=await readTenantBinding(identity);
      if(!observed)throw new Error("tenant_binding_missing");
      const same=observed.credential_profile===current.credential_profile&&observed.allocation_generation===current.allocation_generation&&observed.profile_instance_id===current.profile_instance_id&&observed.allocation_binding_id===current.allocation_binding_id;
      if(!same)throw new Error("tenant_binding_changed_during_lifecycle_transition");
      const next=observed.state===state?observed:await freezeOrRetireBinding(identity,state,reason,current);
      return NextResponse.json({ok:true,binding:next,replacement_allowed:true,profile_transition:{profile_state:profileTransition.profile_state,idempotent_replay:profileTransition.idempotent_replay,credential_decrypted:false,network_accessed:false}},{headers:{"Cache-Control":"no-store"}});
    }
    if(action==="freeze"||action==="retire"){
      if(body.administrator_reclaim_confirmed!==true) throw new Error("administrator_reclaim_confirmation_required");
      const reason=String(body.reason??"allocation_reclaimed_by_administrator");const state=action==="freeze"?"frozen":"retired";const current=await readTenantBinding(identity);if(!current)throw new Error("tenant_binding_missing");
      const profileTransition=await transitionTenantHpcProfile(current,state,reason);const observed=await readTenantBinding(identity);
      if(!observed)throw new Error("tenant_binding_missing");
      const same=observed.credential_profile===current.credential_profile&&observed.allocation_generation===current.allocation_generation&&observed.profile_instance_id===current.profile_instance_id&&observed.allocation_binding_id===current.allocation_binding_id;
      if(!same)throw new Error("tenant_binding_changed_during_lifecycle_transition");
      const next=observed.state===state?observed:await freezeOrRetireBinding(identity,state,reason,current);
      return NextResponse.json({ok:true,binding:next,profile_transition:{profile_state:profileTransition.profile_state,idempotent_replay:profileTransition.idempotent_replay,credential_decrypted:false,network_accessed:false}},{headers:{"Cache-Control":"no-store"}});
    }
    throw new Error("binding_action_invalid");
  } catch(error){return NextResponse.json({ok:false,code:error instanceof Error?error.message:"binding_action_rejected"},{status:400,headers:{"Cache-Control":"no-store"}});}
}
