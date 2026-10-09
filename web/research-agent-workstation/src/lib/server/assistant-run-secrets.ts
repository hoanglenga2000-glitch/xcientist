import { createHash, randomBytes, randomUUID } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";

import type { AssistantManagedHpcIdentity } from "./assistant-managed-hpc";
import type { SessionPrincipal } from "./local-session";
import { runtimeRoot } from "./paths";
import { tenantByoaRoot } from "./tenant-byoa";

const INTENT_SCHEMA = "evomind.run_secret_intent.v1";
const PENDING_SCHEMA = "evomind.run_secret_pending.v1";
const ENVELOPE_SCHEMA = "evomind.hybrid_run_secret_envelope.v1";
const ALGORITHM = "RSA-OAEP-256+A256GCM";
// Allow the mandatory read-only HPC identity gate to finish before the
// one-shot download reference expires. The key lifetime remains the upper
// bound, so this does not outlive the tenant enrollment key.
const INTENT_TTL_MS = 2 * 60 * 60 * 1000;
const UUID_V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const SECRET_REF = /^rs_[A-Za-z0-9_-]{43}$/;
const KEY_ID = /^[a-f0-9]{64}$/;
const PROFILE_INSTANCE = UUID_V4;
const PURPOSES = new Set(["weather4cast_sftp", "kaggle_api"]);

type PrincipalIdentity = { tenantId: string; username: string };

function principalIdentity(principal: SessionPrincipal | null): PrincipalIdentity {
  if (!principal || principal.authentication !== "public_password" || !principal.tenantId) throw new Error("tenant_session_required");
  if (!/^tenant_[a-f0-9]{24}$/.test(principal.tenantId) || !/^[A-Za-z0-9_.-]{1,64}$/.test(principal.username)) throw new Error("tenant_session_invalid");
  return { tenantId: principal.tenantId, username: principal.username };
}

function assertManagedIdentity(identity: PrincipalIdentity, hpc: AssistantManagedHpcIdentity | null): AssistantManagedHpcIdentity {
  if (!hpc || hpc.tenant_id !== identity.tenantId || hpc.owner_principal_id !== identity.username) throw new Error("tenant_hpc_not_ready");
  if (!Number.isSafeInteger(hpc.job_id) || hpc.job_id < 1 || !Number.isSafeInteger(hpc.allocation_generation) || hpc.allocation_generation < 1 || !PROFILE_INSTANCE.test(hpc.profile_instance_id) || hpc.credential_profile !== `${identity.tenantId}_job${hpc.job_id}_g${hpc.allocation_generation}` || !/^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$/.test(hpc.allocation_binding_id)) throw new Error("managed_hpc_identity_invalid");
  return hpc;
}

function root() { return path.join(runtimeRoot, "run-secrets"); }
function intentPath(secretId: string) { return path.join(root(), "intents", `${secretId}.json`); }
function consumedIntentPath(secretId: string) { return path.join(root(), "intents", `${secretId}.consumed.json`); }
function refHash(secretRef: string) { return createHash("sha256").update(secretRef, "utf8").digest("hex"); }
function pendingMetadataPath(secretRef: string) { return path.join(root(), "pending", `${refHash(secretRef)}.json`); }
function pendingEnvelopePath(secretRef: string) { return path.join(root(), "pending", `${refHash(secretRef)}.envelope`); }

async function writeExclusive(target: string, value: unknown) {
  await fs.mkdir(path.dirname(target), { recursive: true, mode: 0o700 });
  await fs.writeFile(target, `${JSON.stringify(value)}\n`, { flag: "wx", mode: 0o600 });
}

async function writeExclusiveBytes(target: string, value: Uint8Array) {
  await fs.mkdir(path.dirname(target), { recursive: true, mode: 0o700 });
  await fs.writeFile(target, value, { flag: "wx", mode: 0o600 });
}

async function currentKey(nowMs: number) {
  const key = JSON.parse(await fs.readFile(path.join(tenantByoaRoot(), "enrollment-keys", "current-public-key.json"), "utf8")) as Record<string, unknown>;
  if (key.schema !== "evomind.tenant_enrollment_key.v1" || key.algorithm !== ALGORITHM || typeof key.key_id !== "string" || !KEY_ID.test(key.key_id) || !key.rsa_jwk || Date.parse(String(key.not_before_utc)) > nowMs || Date.parse(String(key.expires_at_utc)) <= nowMs) throw new Error("run_secret_key_unavailable");
  return key as { key_id: string; rsa_jwk: JsonWebKey; expires_at_utc: string; algorithm: string };
}

export async function issueRunSecretIntent(
  principal: SessionPrincipal | null,
  hpcIdentity: AssistantManagedHpcIdentity | null,
  purposeInput: unknown,
  nowMs = Date.now(),
) {
  const identity = principalIdentity(principal);
  const hpc = assertManagedIdentity(identity, hpcIdentity);
  const purpose = String(purposeInput ?? "");
  if (!PURPOSES.has(purpose)) throw new Error("run_secret_purpose_invalid");
  const key = await currentKey(nowMs);
  const secretId = randomUUID();
  const expiresAt = Math.min(nowMs + INTENT_TTL_MS, Date.parse(key.expires_at_utc));
  if (expiresAt <= nowMs + 30_000) throw new Error("run_secret_key_expiring");
  const record = {
    schema: INTENT_SCHEMA,
    secret_id: secretId,
    purpose,
    tenant_id: identity.tenantId,
    owner_principal_id: identity.username,
    job_id: hpc.job_id,
    allocation_generation: hpc.allocation_generation,
    profile_instance_id: hpc.profile_instance_id,
    allocation_binding_id: hpc.allocation_binding_id,
    credential_profile: hpc.credential_profile,
    key_id: key.key_id,
    issued_at_utc: new Date(nowMs).toISOString(),
    expires_at_utc: new Date(expiresAt).toISOString(),
  };
  await writeExclusive(intentPath(secretId), record);
  return { ...record, algorithm: ALGORITHM, rsa_jwk: key.rsa_jwk };
}

function validEnvelope(value: unknown): value is Record<string, string> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const envelope = value as Record<string, unknown>;
  const expected = ["schema", "algorithm", "key_id", "wrapped_key", "iv", "ciphertext", "tag"];
  return Object.keys(envelope).length === expected.length && expected.every((key) => key in envelope)
    && envelope.schema === ENVELOPE_SCHEMA
    && envelope.algorithm === ALGORITHM
    && typeof envelope.key_id === "string" && KEY_ID.test(envelope.key_id)
    && ["wrapped_key", "iv", "ciphertext", "tag"].every((key) => typeof envelope[key] === "string" && /^[A-Za-z0-9+/=]{16,131072}$/.test(String(envelope[key])));
}

export async function storeRunSecretEnvelope(
  principal: SessionPrincipal | null,
  hpcIdentity: AssistantManagedHpcIdentity | null,
  body: Record<string, unknown>,
  nowMs = Date.now(),
) {
  const identity = principalIdentity(principal);
  const hpc = assertManagedIdentity(identity, hpcIdentity);
  if (Object.keys(body).length !== 2 || !("secret_id" in body) || !("encrypted_envelope" in body)) throw new Error("run_secret_envelope_fields_invalid");
  for (const forbidden of ["password", "token", "credential", "secret", "username", "host", "remote_path"]) {
    if (forbidden in body) throw new Error("plaintext_secret_rejected");
  }
  const secretId = String(body.secret_id ?? "").toLowerCase();
  if (!UUID_V4.test(secretId) || !validEnvelope(body.encrypted_envelope)) throw new Error("run_secret_envelope_invalid");
  if (await fs.stat(consumedIntentPath(secretId)).then(() => true).catch(() => false)) throw new Error("run_secret_intent_consumed");
  const intent = JSON.parse(await fs.readFile(intentPath(secretId), "utf8")) as Record<string, unknown>;
  if (intent.schema !== INTENT_SCHEMA || intent.secret_id !== secretId || intent.tenant_id !== identity.tenantId || intent.owner_principal_id !== identity.username || intent.job_id !== hpc.job_id || intent.allocation_generation !== hpc.allocation_generation || intent.profile_instance_id !== hpc.profile_instance_id || intent.allocation_binding_id !== hpc.allocation_binding_id || intent.credential_profile !== hpc.credential_profile || Date.parse(String(intent.expires_at_utc)) <= nowMs) throw new Error("run_secret_intent_mismatch_or_expired");
  const envelope = body.encrypted_envelope as Record<string, string>;
  const key = await currentKey(nowMs);
  if (envelope.key_id !== intent.key_id || envelope.key_id !== key.key_id) throw new Error("run_secret_key_mismatch");
  const secretRef = `rs_${randomBytes(32).toString("base64url")}`;
  const envelopeBytes = Buffer.from(`${JSON.stringify(envelope)}\n`, "utf8");
  const metadata = {
    schema: PENDING_SCHEMA,
    secret_ref_sha256: refHash(secretRef),
    secret_id: secretId,
    purpose: intent.purpose,
    tenant_id: identity.tenantId,
    owner_principal_id: identity.username,
    job_id: hpc.job_id,
    allocation_generation: hpc.allocation_generation,
    profile_instance_id: hpc.profile_instance_id,
    allocation_binding_id: hpc.allocation_binding_id,
    credential_profile: hpc.credential_profile,
    key_id: envelope.key_id,
    issued_at_utc: new Date(nowMs).toISOString(),
    expires_at_utc: intent.expires_at_utc,
    envelope_sha256: createHash("sha256").update(envelopeBytes).digest("hex"),
    state: "pending",
    secret_values_echoed: false,
  };
  await writeExclusiveBytes(pendingEnvelopePath(secretRef), envelopeBytes);
  try {
    await writeExclusive(pendingMetadataPath(secretRef), metadata);
    await writeExclusive(consumedIntentPath(secretId), { schema: "evomind.run_secret_intent_consumed.v1", secret_id: secretId, consumed_at_utc: new Date(nowMs).toISOString() });
  } catch (error) {
    await fs.rm(pendingEnvelopePath(secretRef), { force: true }).catch(() => undefined);
    await fs.rm(pendingMetadataPath(secretRef), { force: true }).catch(() => undefined);
    throw error;
  }
  await fs.rm(intentPath(secretId), { force: true });
  return { secret_ref: secretRef, purpose: intent.purpose, expires_at_utc: intent.expires_at_utc, secrets_echoed: false };
}

export async function validateRunSecretRefs(
  principal: SessionPrincipal | null,
  hpcIdentity: AssistantManagedHpcIdentity | null,
  values: unknown,
  nowMs = Date.now(),
) {
  const refs = Array.isArray(values) ? [...new Set(values.map((value) => String(value)))] : [];
  if (refs.length > 4 || refs.some((value) => !SECRET_REF.test(value))) throw new Error("run_secret_refs_invalid");
  if (!refs.length) return [] as string[];
  const identity = principalIdentity(principal);
  const hpc = assertManagedIdentity(identity, hpcIdentity);
  for (const secretRef of refs) {
    const metadata = JSON.parse(await fs.readFile(pendingMetadataPath(secretRef), "utf8")) as Record<string, unknown>;
    if (metadata.schema !== PENDING_SCHEMA || metadata.secret_ref_sha256 !== refHash(secretRef) || metadata.state !== "pending" || metadata.tenant_id !== identity.tenantId || metadata.owner_principal_id !== identity.username || metadata.job_id !== hpc.job_id || metadata.allocation_generation !== hpc.allocation_generation || metadata.profile_instance_id !== hpc.profile_instance_id || metadata.allocation_binding_id !== hpc.allocation_binding_id || metadata.credential_profile !== hpc.credential_profile || Date.parse(String(metadata.expires_at_utc)) <= nowMs) throw new Error("run_secret_ref_mismatch_or_expired");
    const envelope = await fs.readFile(pendingEnvelopePath(secretRef));
    if (createHash("sha256").update(envelope).digest("hex") !== metadata.envelope_sha256) throw new Error("run_secret_envelope_hash_mismatch");
  }
  return refs;
}

export const RUN_SECRET_ENVELOPE_SCHEMA = ENVELOPE_SCHEMA;
