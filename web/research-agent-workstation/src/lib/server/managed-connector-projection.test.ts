import assert from "node:assert/strict";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
// @ts-expect-error Node's native TypeScript test runner requires the explicit extension.
import { projectManagedKaggleConnector, projectManagedTenantHpcConnector } from "./managed-connector-projection.ts";

async function writeJson(target: string, payload: unknown) {
  await fs.mkdir(path.dirname(target), { recursive: true });
  await fs.writeFile(target, JSON.stringify(payload), "utf-8");
}

test("managed tenant projection prefers active matching 5-of-5 evidence", async (t) => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "evomind-managed-hpc-"));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const tenant = "tenant_0123456789abcdef01234567";
  const tenantRoot = path.join(root, "tenants", tenant);
  const binding = {
    schema: "evomind.tenant_hpc_binding.v1",
    state: "active",
    job_id: 91555,
    allocation_generation: 13,
    credential_profile: `${tenant}_job91555_g13`,
    allocation_binding_id: "aimslab-job91555-20260814T180205",
  };
  await writeJson(path.join(tenantRoot, "hpc-binding.json"), binding);
  await writeJson(path.join(tenantRoot, "runtime", "gpu-health.json"), {
    schema: "evomind.tenant_gpu_runtime_health.v1",
    status: "ready",
    code: "five_of_five_identity_verified",
    job_id: 91555,
    allocation_generation: 13,
    allocation_binding_id: binding.allocation_binding_id,
    designated_proxy_path_verified: true,
    job_container_verified: true,
    identity_samples_passed: 5,
    probe_remote_write_performed: false,
    checked_at_utc: "2026-08-16T01:00:00.000Z",
    expires_at_utc: "2026-08-16T01:05:00.000Z",
  });

  const projected = await projectManagedTenantHpcConnector(root, Date.parse("2026-08-16T01:03:00.000Z"));
  assert.equal(projected?.configured, true);
  assert.equal(projected?.state, "Tenant HPC Ready");
  assert.equal(projected?.job_id, 91555);
  assert.equal(projected?.allocation_generation, 13);
  assert.equal(projected?.identity_samples_passed, 5);
  assert.equal(projected?.current_gate_ready, true);
});

test("managed tenant projection preserves verified binding but marks an expired live proof", async (t) => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "evomind-managed-hpc-stale-"));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const tenant = "tenant_89abcdef0123456789abcdef";
  const tenantRoot = path.join(root, "tenants", tenant);
  const allocationBindingId = "aimslab-job91555-20260814T180205";
  await writeJson(path.join(tenantRoot, "hpc-binding.json"), {
    schema: "evomind.tenant_hpc_binding.v1",
    state: "active",
    job_id: 91555,
    allocation_generation: 13,
    credential_profile: `${tenant}_job91555_g13`,
    allocation_binding_id: allocationBindingId,
  });
  await writeJson(path.join(tenantRoot, "runtime", "gpu-health.json"), {
    schema: "evomind.tenant_gpu_runtime_health.v1",
    status: "ready",
    code: "five_of_five_identity_verified",
    job_id: 91555,
    allocation_generation: 13,
    allocation_binding_id: allocationBindingId,
    designated_proxy_path_verified: true,
    job_container_verified: true,
    identity_samples_passed: 5,
    probe_remote_write_performed: false,
    checked_at_utc: "2026-08-15T01:00:00.000Z",
    expires_at_utc: "2026-08-15T01:05:00.000Z",
  });

  const projected = await projectManagedTenantHpcConnector(root, Date.parse("2026-08-16T01:00:00.000Z"));
  assert.equal(projected?.state, "Tenant HPC Active Binding Verified");
  assert.equal(projected?.live_probe_fresh, false);
  assert.equal(projected?.current_gate_ready, false);
  assert.equal(projected?.job_container_verified, true);
});

test("managed tenant projection rejects mismatched credential profiles", async (t) => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "evomind-managed-hpc-reject-"));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const tenant = "tenant_fedcba9876543210fedcba98";
  await writeJson(path.join(root, "tenants", tenant, "hpc-binding.json"), {
    schema: "evomind.tenant_hpc_binding.v1",
    state: "active",
    job_id: 91555,
    allocation_generation: 13,
    credential_profile: "job91555",
    allocation_binding_id: "aimslab-job91555-20260814T180205",
  });
  assert.equal(await projectManagedTenantHpcConnector(root), null);
});

test("managed Kaggle projection accepts v2 real API proof only when the DPAPI file exists", async (t) => {
  const workspace = await fs.mkdtemp(path.join(os.tmpdir(), "evomind-managed-kaggle-workspace-"));
  const appData = await fs.mkdtemp(path.join(os.tmpdir(), "evomind-managed-kaggle-appdata-"));
  t.after(() => Promise.all([
    fs.rm(workspace, { recursive: true, force: true }),
    fs.rm(appData, { recursive: true, force: true }),
  ]));
  await writeJson(path.join(workspace, "docs", "kaggle_dpapi_readiness.json"), {
    schema: "evomind.kaggle_dpapi_readiness.v2",
    generated_at: "2026-08-16T01:00:00.000Z",
    status: "passed",
    verification_method: "dpapi_status_and_real_api_smoke",
    credential_status: "authenticated_real_api",
    credential_installed: true,
    authenticated: true,
    human_gate_required_for_submission: true,
    real_api_smoke: { status: "passed", real_external_called: true },
    tool_status: { python_package_installed: true, python_package_version: "2.2.3" },
  });
  const credential = path.join(appData, "ResearchAgentWorkstation", "kaggle_api_token.xml");
  await fs.mkdir(path.dirname(credential), { recursive: true });
  await fs.writeFile(credential, "dpapi-fixture", "utf-8");

  const projected = await projectManagedKaggleConnector(
    workspace,
    appData,
    Date.parse("2026-08-16T01:30:00.000Z"),
    false,
  );
  assert.equal(projected?.configured, true);
  assert.equal(projected?.authenticated, true);
  assert.equal(projected?.state, "Kaggle DPAPI Real API Verified");
  assert.equal(projected?.credential_storage, "windows_dpapi_current_user");
  assert.equal(projected?.human_gate_required_for_submission, true);
});

test("managed Kaggle projection never promotes report-only evidence after credential removal", async (t) => {
  const workspace = await fs.mkdtemp(path.join(os.tmpdir(), "evomind-managed-kaggle-empty-workspace-"));
  const appData = await fs.mkdtemp(path.join(os.tmpdir(), "evomind-managed-kaggle-empty-appdata-"));
  t.after(() => Promise.all([
    fs.rm(workspace, { recursive: true, force: true }),
    fs.rm(appData, { recursive: true, force: true }),
  ]));
  await writeJson(path.join(workspace, "docs", "kaggle_dpapi_readiness.json"), {
    schema: "evomind.kaggle_dpapi_readiness.v2",
    generated_at: "2026-08-16T01:00:00.000Z",
    status: "passed",
    verification_method: "dpapi_status_and_real_api_smoke",
    credential_status: "authenticated_real_api",
    credential_installed: true,
    authenticated: true,
    human_gate_required_for_submission: true,
    real_api_smoke: { status: "passed", real_external_called: true },
  });
  const projected = await projectManagedKaggleConnector(workspace, appData, Date.now(), false);
  assert.equal(projected?.configured, false);
  assert.equal(projected?.authenticated, false);
  assert.equal(projected?.state, "Not Configured");
});
