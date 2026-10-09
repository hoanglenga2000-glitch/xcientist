import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error Node's native TypeScript runner requires the explicit suffix.
import { assertExactTenantProfileTransition } from "./tenant-hpc-profile-lifecycle.ts";
import type { TenantBinding } from "./tenant-byoa.ts";

const binding: TenantBinding = {
  schema: "evomind.tenant_hpc_binding.v1",
  tenant_id: "tenant_0123456789abcdef01234567",
  username: "admin",
  state: "active",
  reason: "current_five_of_five_identity_activation_verified",
  job_id: 91493,
  credential_profile: "tenant_0123456789abcdef01234567_job91493_g10",
  allocation_binding_id: "aimslab-job91493-20260811T221758",
  allocation_generation: 10,
  profile_instance_id: "73872371-f0ee-402d-aae2-e9563280deeb",
  expected_host_uuid: "host",
  expected_gpu_uuid: "gpu",
  container_binding_sha256: "a".repeat(64),
  remote_root: "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra",
  updated_at_utc: "2026-08-15T00:00:00Z",
};

const reason = "allocation_reclaimed_by_administrator";
const exact = {
  schema: "evomind.tenant_hpc_profile_transition.v1",
  status: "completed",
  tenant_id: binding.tenant_id,
  owner_principal_id: binding.username,
  credential_profile: binding.credential_profile,
  job_id: binding.job_id,
  allocation_binding_id: binding.allocation_binding_id,
  allocation_generation: binding.allocation_generation,
  profile_instance_id: binding.profile_instance_id,
  profile_state: "retired",
  reason,
  lifecycle_revision: 8,
  tombstone: "hpc_profile_retired.tombstone.json",
  tombstone_sha256: "b".repeat(64),
  idempotent_replay: false,
  credential_decrypted: false,
  network_accessed: false,
};

test("accepts only an exact local-only terminal profile result", () => {
  assert.equal(assertExactTenantProfileTransition(binding, "retired", reason, exact).profile_state, "retired");
  for (const drift of [
    { job_id: 91555 },
    { allocation_generation: 11 },
    { profile_instance_id: "00000000-0000-4000-8000-000000000011" },
    { network_accessed: true },
    { credential_decrypted: true },
    { tombstone: "other.json" },
  ]) {
    assert.throws(
      () => assertExactTenantProfileTransition(binding, "retired", reason, { ...exact, ...drift }),
      /tenant_profile_transition_identity_mismatch/,
    );
  }
});

