import { promises as fs } from "node:fs";
import path from "node:path";
import { randomBytes, randomUUID } from "node:crypto";
export { assertTenantOwner, tenantIdForUsername, tenantPrincipal } from "./tenant-identity.mjs";
export type { TenantIdentity } from "./tenant-identity.mjs";
import type { TenantIdentity } from "./tenant-identity.mjs";

const TENANT = /^tenant_[a-f0-9]{24}$/;
const PROFILE = /^tenant_[a-f0-9]{24}_job[1-9][0-9]*_g[1-9][0-9]*$/;
const UUID_V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const ALLOCATION_BINDING = /^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$/;
const RESERVATION_SCHEMA = "evomind.tenant_enrollment_identity_reservation.v1";
const RESERVATION_CONSUMED_SCHEMA = "evomind.tenant_enrollment_identity_consumed.v1";
const RESERVATION_TTL_MS = 10 * 60_000;
export type TenantBinding = {
  schema: "evomind.tenant_hpc_binding.v1"; tenant_id: string; username: string;
  state: "provisioning" | "active" | "frozen" | "retired"; reason: string;
  job_id: number; credential_profile: string; allocation_binding_id: string;
  allocation_generation: number; profile_instance_id: string;
  expected_host_uuid: string; expected_gpu_uuid: string; container_binding_sha256: string;
  remote_root: string; updated_at_utc: string;
};

export function tenantByoaRoot() { return path.resolve(process.env.WORKSTATION_BYOA_STATE_ROOT ?? path.join(process.env.WORKSTATION_DATA_DIR ?? process.cwd(), "byoa")); }
function root() { return path.join(tenantByoaRoot(), "tenants"); }
function tenantDir(tenantId: string) {
  if (!TENANT.test(tenantId)) throw new Error("invalid_tenant_id");
  const resolved = path.resolve(root(), tenantId); if (path.dirname(resolved) !== path.resolve(root())) throw new Error("tenant_path_escape"); return resolved;
}
export function tenantBindingPath(tenantId: string) { return path.join(tenantDir(tenantId), "hpc-binding.json"); }
const historyPath = (tenantId: string, generation: number) => path.join(tenantDir(tenantId), "bindings", `g${generation}.json`);
const lockPath = (tenantId: string) => path.join(tenantDir(tenantId), ".binding.lock");
const tombstonePath = (tenantId: string, state: "frozen" | "retired", generation: number) => path.join(tenantDir(tenantId), `hpc-binding.g${generation}.${state}.tombstone.json`);
const reservationDir = (tenantId: string) => path.join(tenantDir(tenantId), "enrollment-reservations");
const reservationPath = (tenantId: string, reservationId: string) => path.join(reservationDir(tenantId), `${reservationId}.json`);
const reservationConsumedPath = (tenantId: string, reservationId: string) => path.join(reservationDir(tenantId), `${reservationId}.consumed.json`);
async function withLock<T>(tenantId: string, action: () => Promise<T>) {
  await fs.mkdir(tenantDir(tenantId), { recursive: true, mode: 0o700 });
  let acquired = false;
  for (let attempt = 0; attempt < 40; attempt += 1) {
    try { await fs.mkdir(lockPath(tenantId)); acquired = true; break; }
    catch (error) { if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error; await new Promise((resolve) => setTimeout(resolve, 25)); }
  }
  if (!acquired) throw new Error("tenant_binding_lock_timeout");
  try { return await action(); } finally { await fs.rm(lockPath(tenantId), { recursive: true, force: true }); }
}
function assertNoSecrets(value: unknown) {
  const text = JSON.stringify(value).toLowerCase();
  if (["password", "private_key", "secret", "credential_xml", "protected_password"].some((key) => text.includes(`"${key}"`))) throw new Error("binding_metadata_contains_secret");
}
async function writeAtomic(target: string, value: unknown, exclusive = false) {
  assertNoSecrets(value); const temporary = `${target}.${process.pid}.${randomBytes(8).toString("hex")}.tmp`;
  await fs.writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, { encoding: "utf8", mode: 0o600, flag: "wx" });
  try { if (exclusive) await fs.link(temporary, target); else await fs.rename(temporary, target); }
  finally { await fs.rm(temporary, { force: true }); }
}

type ProvisioningBindingInput = Omit<TenantBinding, "schema" | "tenant_id" | "username" | "state" | "reason" | "updated_at_utc">;
export type TenantEnrollmentIdentityReservation = {
  schema: typeof RESERVATION_SCHEMA;
  reservation_id: string;
  tenant_id: string;
  owner_principal_id: string;
  nonce: string;
  job_id: number;
  allocation_binding_id: string;
  allocation_generation: number;
  credential_profile: string;
  profile_instance_id: string;
  issued_at_utc: string;
  expires_at_utc: string;
};
type HistoricalIdentity = { maxGeneration: number; profileInstanceIds: Set<string>; profileInstanceCounts: Map<string, number> };

function positiveGeneration(value: unknown): number | null {
  const generation = Number(value);
  return Number.isSafeInteger(generation) && generation > 0 ? generation : null;
}
async function readJsonObject(target: string, label: string): Promise<Record<string, unknown>> {
  try {
    const value = JSON.parse(await fs.readFile(target, "utf8")) as unknown;
    if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(label);
    return value as Record<string, unknown>;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") throw error;
    throw new Error(`${label}_invalid`);
  }
}
async function historicalIdentity(identity: TenantIdentity): Promise<HistoricalIdentity> {
  const generations = new Set<number>();
  const profileInstanceIds = new Set<string>();
  const profileInstanceCounts = new Map<string, number>();
  const collect = (value: Record<string, unknown>, fileGeneration?: number) => {
    if (fileGeneration) generations.add(fileGeneration);
    const generation = positiveGeneration(value.allocation_generation);
    if (generation) generations.add(generation);
    const instance = String(value.profile_instance_id ?? "").toLowerCase();
    if (instance) {
      if (!UUID_V4.test(instance)) throw new Error("tenant_profile_instance_history_invalid");
      profileInstanceIds.add(instance);
      profileInstanceCounts.set(instance, (profileInstanceCounts.get(instance) ?? 0) + 1);
    }
  };
  const collectDirectory = async (directory: string, matcher: RegExp, label: string) => {
    for (const entry of await fs.readdir(directory, { withFileTypes: true }).catch(() => [] as import("node:fs").Dirent[])) {
      if (!entry.isFile()) continue;
      const match = matcher.exec(entry.name); if (!match) continue;
      collect(await readJsonObject(path.join(directory, entry.name), label), match[1] ? Number(match[1]) : undefined);
    }
  };
  const directory = tenantDir(identity.tenantId);
  try { collect(await readJsonObject(tenantBindingPath(identity.tenantId), "tenant_current_binding")); }
  catch (error) { if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error; }
  await collectDirectory(path.join(directory, "bindings"), /^g([1-9][0-9]*)\.json$/, "tenant_binding_history");
  await collectDirectory(directory, /^hpc-binding\.g([1-9][0-9]*)\.(?:frozen|retired)\.tombstone\.json$/, "tenant_binding_tombstone");
  await collectDirectory(path.join(directory, "activation-receipts"), /^g([1-9][0-9]*)(?:\.prepared)?\.json$/, "tenant_activation_receipt");
  await collectDirectory(path.join(directory, "activation-requests"), /^g([1-9][0-9]*)\.[0-9a-f-]{36}\.json$/i, "tenant_activation_request");
  await collectDirectory(reservationDir(identity.tenantId), /^[0-9a-f-]{36}(?:\.consumed)?\.json$/i, "tenant_enrollment_reservation");
  const inbox = path.join(tenantByoaRoot(), "inbox");
  for (const entry of await fs.readdir(inbox, { withFileTypes: true }).catch(() => [] as import("node:fs").Dirent[])) {
    if (!entry.isFile() || !/^[0-9a-f-]{36}\.(?:binding|receipt)\.json$/i.test(entry.name)) continue;
    const value = await readJsonObject(path.join(inbox, entry.name), "tenant_queue_binding");
    if (value.tenant_id === identity.tenantId) collect(value);
  }
  return { maxGeneration: generations.size ? Math.max(...generations) : 0, profileInstanceIds, profileInstanceCounts };
}
async function currentBindingForProvisioning(identity: TenantIdentity): Promise<TenantBinding | null> {
  let previous: TenantBinding | null = null;
  try { previous = await readTenantBinding(identity); }
  catch (error) {
    if (!String((error as Error).message).includes("tenant_binding_tombstoned")) throw error;
    throw new Error("tenant_binding_crash_window_fail_closed");
  }
  if (previous && previous.state !== "retired") throw new Error("tenant_binding_already_exists");
  if (previous && !(await fs.stat(tombstonePath(identity.tenantId, "retired", previous.allocation_generation)).then((item) => item.isFile()).catch(() => false))) {
    throw new Error("tenant_binding_retirement_tombstone_missing");
  }
  return previous;
}
function validateProvisioningInput(identity: TenantIdentity, input: ProvisioningBindingInput) {
  if (!Number.isSafeInteger(input.job_id) || input.job_id < 1 || !Number.isSafeInteger(input.allocation_generation) || input.allocation_generation < 1) throw new Error("allocation_identity_invalid");
  if (!ALLOCATION_BINDING.test(input.allocation_binding_id) || !UUID_V4.test(input.profile_instance_id.toLowerCase())) throw new Error("allocation_identity_invalid");
  if (input.credential_profile !== `${identity.tenantId}_job${input.job_id}_g${input.allocation_generation}`) throw new Error("tenant_profile_binding_mismatch");
}
function provisioningBindingValue(identity: TenantIdentity, input: ProvisioningBindingInput, updatedAtMs?: number): TenantBinding {
  return { schema: "evomind.tenant_hpc_binding.v1", tenant_id: identity.tenantId, username: identity.username, state: "provisioning", reason: "awaiting_encrypted_enrollment", updated_at_utc: new Date(updatedAtMs ?? Date.now()).toISOString(), ...input, profile_instance_id: input.profile_instance_id.toLowerCase() };
}
async function createProvisioningBindingLocked(identity: TenantIdentity, input: ProvisioningBindingInput, history: HistoricalIdentity, reserved = false, updatedAtMs?: number) {
  validateProvisioningInput(identity, input);
  const previous = await currentBindingForProvisioning(identity);
  const expectedGeneration = history.maxGeneration + (reserved ? 0 : 1);
  if (input.allocation_generation !== expectedGeneration) throw new Error("tenant_binding_generation_not_next");
  if (!reserved && history.profileInstanceIds.has(input.profile_instance_id.toLowerCase())) throw new Error("tenant_profile_instance_reused");
  const value = provisioningBindingValue(identity, input, updatedAtMs);
  await fs.mkdir(path.dirname(historyPath(identity.tenantId, value.allocation_generation)), { recursive: true, mode: 0o700 });
  await writeAtomic(historyPath(identity.tenantId, value.allocation_generation), value, true);
  await writeAtomic(tenantBindingPath(identity.tenantId), value, !previous);
  return value;
}
export async function createProvisioningBinding(identity: TenantIdentity, input: ProvisioningBindingInput) {
  return withLock(identity.tenantId, async () => createProvisioningBindingLocked(identity, input, await historicalIdentity(identity)));
}
function authoritativeNow(value: number): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new Error("tenant_enrollment_authoritative_time_invalid");
  return value;
}
export async function issueTenantEnrollmentIdentity(identity: TenantIdentity, input: { job_id: number; allocation_binding_id: string; authoritative_now_ms: number }): Promise<TenantEnrollmentIdentityReservation> {
  return withLock(identity.tenantId, async () => {
    if (!Number.isSafeInteger(input.job_id) || input.job_id < 1 || !ALLOCATION_BINDING.test(input.allocation_binding_id)) throw new Error("allocation_identity_invalid");
    const now = authoritativeNow(input.authoritative_now_ms);
    await currentBindingForProvisioning(identity);
    const history = await historicalIdentity(identity);
    let instance = randomUUID().toLowerCase();
    while (history.profileInstanceIds.has(instance)) instance = randomUUID().toLowerCase();
    const reservationId = randomUUID().toLowerCase(); const nonce = randomUUID().toLowerCase(); const generation = history.maxGeneration + 1;
    const reservation: TenantEnrollmentIdentityReservation = { schema: RESERVATION_SCHEMA, reservation_id: reservationId, tenant_id: identity.tenantId, owner_principal_id: identity.username, nonce, job_id: input.job_id, allocation_binding_id: input.allocation_binding_id, allocation_generation: generation, credential_profile: `${identity.tenantId}_job${input.job_id}_g${generation}`, profile_instance_id: instance, issued_at_utc: new Date(now).toISOString(), expires_at_utc: new Date(now + RESERVATION_TTL_MS).toISOString() };
    await fs.mkdir(reservationDir(identity.tenantId), { recursive: true, mode: 0o700 });
    await writeAtomic(reservationPath(identity.tenantId, reservationId), reservation, true);
    return reservation;
  });
}
type TenantEnrollmentIdentityExpectation = { key_id: string; job_id: number; allocation_binding_id: string; authoritative_now_ms: number };
type ValidatedReservation = { reservation: TenantEnrollmentIdentityReservation; history: HistoricalIdentity; now: number; reservationId: string; prospectiveBinding: TenantBinding };
async function validateTenantEnrollmentIdentityLocked(identity: TenantIdentity, reservationIdInput: string, expected: TenantEnrollmentIdentityExpectation): Promise<ValidatedReservation> {
    const now = authoritativeNow(expected.authoritative_now_ms); const reservationId = reservationIdInput.toLowerCase();
    if (!UUID_V4.test(reservationId) || !/^[a-f0-9]{64}$/.test(expected.key_id) || !Number.isSafeInteger(expected.job_id) || expected.job_id < 1 || !ALLOCATION_BINDING.test(expected.allocation_binding_id)) throw new Error("tenant_enrollment_reservation_invalid");
    if (await fs.stat(reservationConsumedPath(identity.tenantId, reservationId)).then(() => true).catch(() => false)) throw new Error("tenant_enrollment_reservation_consumed");
    const reservation = await readJsonObject(reservationPath(identity.tenantId, reservationId), "tenant_enrollment_reservation") as unknown as TenantEnrollmentIdentityReservation;
    if (reservation.schema !== RESERVATION_SCHEMA || reservation.tenant_id !== identity.tenantId || reservation.owner_principal_id !== identity.username || reservation.reservation_id !== reservationId || reservation.job_id !== expected.job_id || reservation.allocation_binding_id !== expected.allocation_binding_id || !UUID_V4.test(reservation.nonce) || !Number.isSafeInteger(reservation.job_id) || reservation.job_id < 1 || !ALLOCATION_BINDING.test(reservation.allocation_binding_id) || !Number.isSafeInteger(reservation.allocation_generation) || reservation.allocation_generation < 1 || reservation.credential_profile !== `${identity.tenantId}_job${reservation.job_id}_g${reservation.allocation_generation}` || !UUID_V4.test(reservation.profile_instance_id) || !Number.isFinite(Date.parse(reservation.issued_at_utc)) || Date.parse(reservation.issued_at_utc) > now + 60_000 || Date.parse(reservation.expires_at_utc) <= now || Date.parse(reservation.expires_at_utc) - Date.parse(reservation.issued_at_utc) !== RESERVATION_TTL_MS) throw new Error("tenant_enrollment_reservation_mismatch_or_expired");
    const history = await historicalIdentity(identity);
    if (reservation.allocation_generation !== history.maxGeneration) throw new Error("tenant_enrollment_reservation_superseded");
    const occurrences = history.profileInstanceCounts.get(reservation.profile_instance_id) ?? 0;
    if (occurrences !== 1) throw new Error("tenant_profile_instance_reused");
    const input: ProvisioningBindingInput = { job_id: reservation.job_id, allocation_generation: reservation.allocation_generation, credential_profile: reservation.credential_profile, allocation_binding_id: reservation.allocation_binding_id, profile_instance_id: reservation.profile_instance_id, expected_host_uuid: "", expected_gpu_uuid: "", container_binding_sha256: "", remote_root: "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra" };
    validateProvisioningInput(identity, input); await currentBindingForProvisioning(identity);
    return { reservation, history, now, reservationId, prospectiveBinding: provisioningBindingValue(identity, input, now) };
}
export async function validateTenantEnrollmentIdentity(identity: TenantIdentity, reservationIdInput: string, expected: TenantEnrollmentIdentityExpectation): Promise<TenantEnrollmentIdentityReservation> {
  return withLock(identity.tenantId, async () => (await validateTenantEnrollmentIdentityLocked(identity, reservationIdInput, expected)).reservation);
}
export async function consumeTenantEnrollmentIdentity(identity: TenantIdentity, reservationIdInput: string, expected: TenantEnrollmentIdentityExpectation, beforeCommit?: (reservation: TenantEnrollmentIdentityReservation, prospectiveBinding: TenantBinding) => Promise<void>) {
  return withLock(identity.tenantId, async () => {
    const validated = await validateTenantEnrollmentIdentityLocked(identity, reservationIdInput, expected);
    const { reservation, history, now, reservationId, prospectiveBinding } = validated;
    if (beforeCommit) await beforeCommit(reservation, prospectiveBinding);
    await writeAtomic(reservationConsumedPath(identity.tenantId, reservationId), { schema: RESERVATION_CONSUMED_SCHEMA, reservation_id: reservationId, tenant_id: identity.tenantId, owner_principal_id: identity.username, nonce: reservation.nonce, job_id: reservation.job_id, allocation_binding_id: reservation.allocation_binding_id, allocation_generation: reservation.allocation_generation, credential_profile: reservation.credential_profile, profile_instance_id: reservation.profile_instance_id, key_id: expected.key_id, consumed_at_utc: new Date(now).toISOString() }, true);
    const binding = await createProvisioningBindingLocked(identity, { job_id: reservation.job_id, allocation_generation: reservation.allocation_generation, credential_profile: reservation.credential_profile, allocation_binding_id: reservation.allocation_binding_id, profile_instance_id: reservation.profile_instance_id, expected_host_uuid: "", expected_gpu_uuid: "", container_binding_sha256: "", remote_root: "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra" }, history, true, now);
    return { reservation, binding };
  });
}
function sameBindingIdentity(left: TenantBinding, right: TenantBinding) {
  return left.tenant_id === right.tenant_id
    && left.username === right.username
    && left.job_id === right.job_id
    && left.credential_profile === right.credential_profile
    && left.allocation_binding_id === right.allocation_binding_id
    && left.allocation_generation === right.allocation_generation
    && left.profile_instance_id === right.profile_instance_id;
}
export async function freezeOrRetireBinding(identity: TenantIdentity, state: "frozen" | "retired", reason: string, expected?: TenantBinding) {
  if (!/^[a-z0-9_:-]{3,128}$/.test(reason)) throw new Error("invalid_lifecycle_reason");
  return withLock(identity.tenantId, async () => {
    const current = await readTenantBinding(identity); if (!current) throw new Error("tenant_binding_missing");
    if (expected && !sameBindingIdentity(current, expected)) throw new Error("tenant_binding_changed_during_lifecycle_transition");
    if (current.state === "retired" || (current.state === "frozen" && state === "frozen")) throw new Error("tenant_binding_transition_rejected");
    const tombstone = { schema: "evomind.tenant_hpc_binding_tombstone.v1", tenant_id: identity.tenantId, job_id: current.job_id, credential_profile: current.credential_profile, allocation_generation: current.allocation_generation, state, reason, created_at_utc: new Date().toISOString() };
    await writeAtomic(tombstonePath(identity.tenantId, state, current.allocation_generation), tombstone, true);
    const next = { ...current, state, reason, updated_at_utc: new Date().toISOString() }; await writeAtomic(tenantBindingPath(identity.tenantId), next); return next;
  });
}
export async function activateTenantBinding(identity: TenantIdentity, receipt: Record<string, unknown>) {
  return withLock(identity.tenantId, async () => {
    const current = await readTenantBinding(identity); if (!current || current.state !== "provisioning") throw new Error("tenant_binding_not_provisioning");
    const gpuUuids = Array.isArray(receipt.gpu_uuids) ? receipt.gpu_uuids : [];
    if (receipt.schema !== "evomind.tenant_hpc_activation_receipt.v1" || receipt.tenant_id !== identity.tenantId || receipt.owner_principal_id !== identity.username || receipt.job_id !== current.job_id || receipt.credential_profile !== current.credential_profile || receipt.allocation_binding_id !== current.allocation_binding_id || receipt.allocation_generation !== current.allocation_generation || receipt.profile_instance_id !== current.profile_instance_id || typeof receipt.host_uuid !== "string" || !receipt.host_uuid || gpuUuids.length !== 1 || typeof gpuUuids[0] !== "string" || !gpuUuids[0] || receipt.remote_root !== current.remote_root || !/^[a-f0-9]{64}$/.test(String(receipt.container_binding_sha256 ?? "")) || receipt.designated_proxy_path_verified !== true || receipt.job_container_verified !== true || receipt.five_of_five_identity_samples !== true || receipt.connection_ready !== true) throw new Error("tenant_activation_receipt_invalid");
    const next: TenantBinding = { ...current, expected_host_uuid: receipt.host_uuid, expected_gpu_uuid: gpuUuids[0], container_binding_sha256: String(receipt.container_binding_sha256), state: "active", reason: "current_five_of_five_identity_activation_verified", updated_at_utc: new Date().toISOString() };
    await writeAtomic(tenantBindingPath(identity.tenantId), next); return next;
  });
}
export async function readTenantBinding(identity: TenantIdentity): Promise<TenantBinding | null> {
  try {
    const value = JSON.parse(await fs.readFile(tenantBindingPath(identity.tenantId), "utf8")) as TenantBinding;
    if (value.schema !== "evomind.tenant_hpc_binding.v1" || value.tenant_id !== identity.tenantId || value.username.toLowerCase() !== identity.username.toLowerCase() || !PROFILE.test(value.credential_profile)) throw new Error("tenant_binding_invalid");
    const entries = await fs.readdir(tenantDir(identity.tenantId)).catch(() => [] as string[]);
    if (entries.some((name) => name === `hpc-binding.g${value.allocation_generation}.frozen.tombstone.json` || name === `hpc-binding.g${value.allocation_generation}.retired.tombstone.json`)) {
      if (value.state === "active") throw new Error("tenant_binding_tombstoned");
    }
    return value;
  } catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return null; throw error; }
}
export async function requireActiveTenantBinding(identity: TenantIdentity) {
  const binding = await readTenantBinding(identity);
  if (!binding || binding.state !== "active") throw new Error("tenant_hpc_not_ready");
  if (binding.credential_profile !== `${identity.tenantId}_job${binding.job_id}_g${binding.allocation_generation}`) throw new Error("tenant_profile_binding_mismatch");
  if (!binding.profile_instance_id || !binding.expected_host_uuid || !binding.expected_gpu_uuid || !/^[a-f0-9]{64}$/.test(binding.container_binding_sha256)) throw new Error("tenant_identity_binding_incomplete");
  if (binding.remote_root !== "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra") throw new Error("tenant_remote_root_mismatch");
  return binding;
}

