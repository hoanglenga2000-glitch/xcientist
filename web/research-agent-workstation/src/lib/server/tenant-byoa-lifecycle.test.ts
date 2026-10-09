import assert from "node:assert/strict";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

const root = await fs.mkdtemp(path.join(os.tmpdir(), "evomind-binding-r3-"));
process.env.WORKSTATION_BYOA_STATE_ROOT = root;
// @ts-ignore explicit suffix required by strip-types runner
const api = await import("./tenant-byoa.ts");
type Identity = { username: string; tenantId: string };
const identity = (suffix: string): Identity => ({ username: `user-${suffix}`, tenantId: `tenant_${suffix.repeat(24).slice(0, 24)}` });
const uuid = (number: number) => `00000000-0000-4000-8000-${String(number).padStart(12, "0")}`;
const input = (owner: Identity, generation: number, instance: string, jobId = 91493) => ({
  job_id: jobId,
  credential_profile: `${owner.tenantId}_job${jobId}_g${generation}`,
  allocation_binding_id: `aimslab-job${jobId}-generation${generation}`,
  allocation_generation: generation,
  profile_instance_id: instance,
  expected_host_uuid: "",
  expected_gpu_uuid: "",
  container_binding_sha256: "",
  remote_root: "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra",
});
const tenantRoot = (owner: Identity) => path.dirname(api.tenantBindingPath(owner.tenantId));
const writeJson = async (target: string, value: unknown) => {
  await fs.mkdir(path.dirname(target), { recursive: true });
  await fs.writeFile(target, `${JSON.stringify(value)}\n`, { flag: "wx" });
};
const now = Date.UTC(2026, 7, 14, 0, 0, 0);
const keyId = "a".repeat(64);

test("direct provisioning requires exact historical max plus one and rejects skipped duplicate low generations and UUID reuse", async () => {
  const owner = identity("a");
  const first = await api.createProvisioningBinding(owner, input(owner, 1, uuid(1)));
  assert.equal(first.allocation_generation, 1);
  await api.freezeOrRetireBinding(owner, "retired", "allocation_reclaimed");
  for (const generation of [1, 3]) {
    await assert.rejects(api.createProvisioningBinding(owner, input(owner, generation, uuid(10 + generation))), /tenant_binding_generation_not_next/);
  }
  await assert.rejects(api.createProvisioningBinding(owner, input(owner, 2, uuid(1))), /tenant_profile_instance_reused/);
  const second = await api.createProvisioningBinding(owner, input(owner, 2, uuid(2)));
  assert.equal(second.allocation_generation, 2);
  assert.deepEqual((await fs.readdir(path.join(tenantRoot(owner), "bindings"))).sort(), ["g1.json", "g2.json"]);
});

test("trusted current history tombstone activation request receipt queue and reservation state determine max plus one", async () => {
  const owner = identity("b"); const dir = tenantRoot(owner);
  const current = { schema: "evomind.tenant_hpc_binding.v1", tenant_id: owner.tenantId, username: owner.username, state: "retired", reason: "allocation_reclaimed", ...input(owner, 3, uuid(3)), updated_at_utc: new Date().toISOString() };
  await writeJson(api.tenantBindingPath(owner.tenantId), current);
  await writeJson(path.join(dir, "hpc-binding.g3.retired.tombstone.json"), { allocation_generation: 3, profile_instance_id: uuid(3) });
  await writeJson(path.join(dir, "bindings", "g5.json"), { allocation_generation: 5, profile_instance_id: uuid(5) });
  await writeJson(path.join(dir, "activation-receipts", "g7.json"), { allocation_generation: 7, profile_instance_id: uuid(7) });
  await writeJson(path.join(dir, "activation-requests", `g8.${uuid(80)}.json`), { allocation_generation: 8, profile_instance_id: uuid(8) });
  await writeJson(path.join(root, "inbox", `${uuid(90)}.binding.json`), { tenant_id: owner.tenantId, allocation_generation: 9, profile_instance_id: uuid(9) });
  await assert.rejects(api.createProvisioningBinding(owner, input(owner, 11, uuid(11))), /tenant_binding_generation_not_next/);
  const next = await api.createProvisioningBinding(owner, input(owner, 10, uuid(10)));
  assert.equal(next.allocation_generation, 10);
});

test("active binding plus terminal tombstone crash window remains fail closed and cannot provision", async () => {
  const owner = identity("c");
  const currentPath = api.tenantBindingPath(owner.tenantId);
  const value = await api.createProvisioningBinding(owner, input(owner, 1, uuid(21)));
  await fs.writeFile(currentPath, `${JSON.stringify({ ...value, state: "active" })}\n`);
  await writeJson(path.join(tenantRoot(owner), "hpc-binding.g1.retired.tombstone.json"), { allocation_generation: 1, profile_instance_id: uuid(21) });
  await assert.rejects(api.requireActiveTenantBinding(owner), /tenant_binding_tombstoned/);
  await assert.rejects(api.createProvisioningBinding(owner, input(owner, 2, uuid(22))), /tenant_binding_crash_window_fail_closed/);
  assert.equal(JSON.parse(await fs.readFile(currentPath, "utf8")).state, "active");
});

test("server issued reservations are lock serialized unique persistent and consumed once into exact provisioning identity", async () => {
  const owner = identity("d");
  const [first, second] = await Promise.all([
    api.issueTenantEnrollmentIdentity(owner, { job_id: 91493, allocation_binding_id: "aimslab-job91493-generation-auto-a", authoritative_now_ms: now }),
    api.issueTenantEnrollmentIdentity(owner, { job_id: 91493, allocation_binding_id: "aimslab-job91493-generation-auto-b", authoritative_now_ms: now }),
  ]);
  assert.deepEqual([first.allocation_generation, second.allocation_generation].sort((a, b) => a - b), [1, 2]);
  assert.notEqual(first.profile_instance_id, second.profile_instance_id);
  assert.notEqual(first.nonce, second.nonce);
  const latest = first.allocation_generation > second.allocation_generation ? first : second;
  // @ts-ignore query suffix intentionally simulates a fresh server module after restart
  const restarted = await import(`./tenant-byoa.ts?restart=${Date.now()}`);
  const consumed = await restarted.consumeTenantEnrollmentIdentity(owner, latest.reservation_id, { key_id: keyId, job_id: latest.job_id, allocation_binding_id: latest.allocation_binding_id, authoritative_now_ms: now + 1_000 });
  assert.equal(consumed.binding.state, "provisioning");
  assert.equal(consumed.binding.allocation_generation, 2);
  assert.equal(consumed.binding.profile_instance_id, latest.profile_instance_id);
  assert.equal(JSON.parse(await fs.readFile(api.tenantBindingPath(owner.tenantId), "utf8")).credential_profile, latest.credential_profile);
  const consumedPath = path.join(tenantRoot(owner), "enrollment-reservations", `${latest.reservation_id}.consumed.json`);
  const before = await fs.readFile(consumedPath);
  await assert.rejects(restarted.consumeTenantEnrollmentIdentity(owner, latest.reservation_id, { key_id: keyId, job_id: latest.job_id, allocation_binding_id: latest.allocation_binding_id, authoritative_now_ms: now + 2_000 }), /tenant_enrollment_reservation_consumed/);
  assert.deepEqual(await fs.readFile(consumedPath), before);
});

test("reservation consumption rejects drift expiry superseded tickets and historical UUID reuse", async () => {
  const owner = identity("e");
  const first = await api.issueTenantEnrollmentIdentity(owner, { job_id: 91493, allocation_binding_id: "aimslab-job91493-generation-one", authoritative_now_ms: now });
  const second = await api.issueTenantEnrollmentIdentity(owner, { job_id: 91493, allocation_binding_id: "aimslab-job91493-generation-two", authoritative_now_ms: now });
  await assert.rejects(api.consumeTenantEnrollmentIdentity(owner, first.reservation_id, { key_id: keyId, job_id: first.job_id, allocation_binding_id: first.allocation_binding_id, authoritative_now_ms: now + 1_000 }), /tenant_enrollment_reservation_superseded/);
  const reservationFile = path.join(tenantRoot(owner), "enrollment-reservations", `${second.reservation_id}.json`);
  const expired = JSON.parse(await fs.readFile(reservationFile, "utf8")); expired.expires_at_utc = new Date(0).toISOString();
  await fs.writeFile(reservationFile, `${JSON.stringify(expired)}\n`);
  await assert.rejects(api.consumeTenantEnrollmentIdentity(owner, second.reservation_id, { key_id: keyId, job_id: second.job_id, allocation_binding_id: second.allocation_binding_id, authoritative_now_ms: now + 1_000 }), /tenant_enrollment_reservation_mismatch_or_expired/);
  await assert.rejects(api.issueTenantEnrollmentIdentity(owner, { job_id: 91493, allocation_binding_id: "aimslab-job91493-generation-time", authoritative_now_ms: Number.NaN }), /tenant_enrollment_authoritative_time_invalid/);
  await fs.mkdir(path.join(tenantRoot(owner), "bindings"), { recursive: true });
  await writeJson(path.join(tenantRoot(owner), "bindings", "g3.json"), { allocation_generation: 3, profile_instance_id: uuid(33) });
  const current = { schema: "evomind.tenant_hpc_binding.v1", tenant_id: owner.tenantId, username: owner.username, state: "retired", reason: "allocation_reclaimed", ...input(owner, 3, uuid(33)), updated_at_utc: new Date().toISOString() };
  await fs.writeFile(api.tenantBindingPath(owner.tenantId), `${JSON.stringify(current)}\n`);
  await writeJson(path.join(tenantRoot(owner), "hpc-binding.g3.retired.tombstone.json"), { allocation_generation: 3, profile_instance_id: uuid(33) });
  await assert.rejects(api.createProvisioningBinding(owner, input(owner, 4, uuid(33))), /tenant_profile_instance_reused/);
});

test("reservation validation is read only and beforeCommit failure leaves no tombstone or binding", async () => {
  const owner = identity("2");
  const reservation = await api.issueTenantEnrollmentIdentity(owner, { job_id: 91493, allocation_binding_id: "aimslab-job91493-staging-contract", authoritative_now_ms: now });
  const expected = { key_id: keyId, job_id: reservation.job_id, allocation_binding_id: reservation.allocation_binding_id, authoritative_now_ms: now + 1_000 };
  const validated = await api.validateTenantEnrollmentIdentity(owner, reservation.reservation_id, expected);
  assert.equal(validated.profile_instance_id, reservation.profile_instance_id);
  assert.equal(await fs.stat(api.tenantBindingPath(owner.tenantId)).then(() => true).catch(() => false), false);
  await assert.rejects(api.consumeTenantEnrollmentIdentity(owner, reservation.reservation_id, expected, async (_ticket: unknown, preview: { state: string }) => {
    assert.equal(preview.state, "provisioning"); throw new Error("staging_failed");
  }), /staging_failed/);
  assert.equal(await fs.stat(api.tenantBindingPath(owner.tenantId)).then(() => true).catch(() => false), false);
  assert.equal(await fs.stat(path.join(tenantRoot(owner), "enrollment-reservations", `${reservation.reservation_id}.consumed.json`)).then(() => true).catch(() => false), false);
  const consumed = await api.consumeTenantEnrollmentIdentity(owner, reservation.reservation_id, expected, async () => undefined);
  assert.equal(consumed.binding.profile_instance_id, reservation.profile_instance_id);
});

test("activation receipt propagates authoritative host and GPU identity and rejects drift", async () => {
  const owner = identity("f"); const current = await api.createProvisioningBinding(owner, input(owner, 1, uuid(41)));
  const receipt = { schema: "evomind.tenant_hpc_activation_receipt.v1", tenant_id: owner.tenantId, owner_principal_id: owner.username, job_id: current.job_id, credential_profile: current.credential_profile, allocation_binding_id: current.allocation_binding_id, allocation_generation: current.allocation_generation, profile_instance_id: current.profile_instance_id, host_uuid: "host-authoritative", gpu_uuids: ["gpu-authoritative"], remote_root: current.remote_root, container_binding_sha256: "c".repeat(64), designated_proxy_path_verified: true, job_container_verified: true, five_of_five_identity_samples: true, connection_ready: true };
  const active = await api.activateTenantBinding(owner, receipt);
  assert.equal(active.expected_host_uuid, "host-authoritative"); assert.equal(active.expected_gpu_uuid, "gpu-authoritative");
  const driftOwner = identity("1"); const driftCurrent = await api.createProvisioningBinding(driftOwner, input(driftOwner, 1, uuid(42)));
  await assert.rejects(api.activateTenantBinding(driftOwner, { ...receipt, tenant_id: driftOwner.tenantId, owner_principal_id: driftOwner.username, credential_profile: driftCurrent.credential_profile, allocation_binding_id: "drifted-allocation-binding", profile_instance_id: driftCurrent.profile_instance_id }), /tenant_activation_receipt_invalid/);
});

