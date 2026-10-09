import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

test("HPC Evidence Identity enforces one run per job and one job per run", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "evomind-hpc-lineage-"));
  const previousRoot = process.env.WORKSTATION_ROOT;
  process.env.WORKSTATION_ROOT = root;
  try {
    const { bindHpcJobIdentity, listHpcJobLineage } = await import(`./hpc-job-lineage.ts?root=${Date.now()}`);
    const base = {
      taskId: "siim-isic-melanoma-classification",
      cluster: "aimslab",
      owner: "workstation_orchestrator",
      status: "RESERVED",
      credentialProfile: "job90001",
      resourceProfile: "aimslab_a800_80gb",
      executionBackend: "hpc",
    };
    await bindHpcJobIdentity({ ...base, runId: "wr_hpc_lineage_001", jobId: 90001 });
    await assert.rejects(
      bindHpcJobIdentity({ ...base, runId: "wr_hpc_lineage_002", jobId: 90001 }),
      /already bound to run/,
    );
    await assert.rejects(
      bindHpcJobIdentity({ ...base, runId: "wr_hpc_lineage_001", jobId: 90002 }),
      /already bound to HPC job/,
    );
    const updated = await bindHpcJobIdentity({ ...base, runId: "wr_hpc_lineage_001", jobId: 90001, status: "FAILED" });
    assert.equal(updated.status, "FAILED");
    assert.equal((await listHpcJobLineage()).length, 1);
  } finally {
    if (previousRoot === undefined) delete process.env.WORKSTATION_ROOT;
    else process.env.WORKSTATION_ROOT = previousRoot;
    await rm(root, { recursive: true, force: true });
  }
});

test("execution identity supersedes the profile-only projection for the same job", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "evomind-hpc-lineage-projection-"));
  const previousRoot = process.env.WORKSTATION_ROOT;
  process.env.WORKSTATION_ROOT = root;
  try {
    const { bindHpcJobIdentity, listHpcJobLineage, recordHpcProfileIdentity } = await import(`./hpc-job-lineage.ts?projection=${Date.now()}`);
    await recordHpcProfileIdentity({
      profile: "job91051",
      jobId: 91051,
      cluster: "aimslab",
      owner: "workstation_orchestrator",
      status: "PROFILE_ONLY_NOT_EXECUTION_JOB",
      evidencePath: "workspace/hpc/job91051_onboarding_current.json",
    });
    assert.equal((await listHpcJobLineage())[0]?.identity_kind, "allocation_profile");

    await bindHpcJobIdentity({
      runId: "wr_job91051_execution",
      taskId: "titanic",
      jobId: 91051,
      cluster: "aimslab",
      owner: "workstation_orchestrator",
      status: "COMPLETED",
      credentialProfile: "job91051",
      resourceProfile: "aimslab_a800_80gb",
      executionBackend: "hpc",
    });
    const lineage = await listHpcJobLineage();
    assert.equal(lineage.length, 1);
    assert.equal(lineage[0]?.identity_kind, "execution_job");
    assert.equal(lineage[0]?.run_id, "wr_job91051_execution");
  } finally {
    if (previousRoot === undefined) delete process.env.WORKSTATION_ROOT;
    else process.env.WORKSTATION_ROOT = previousRoot;
    await rm(root, { recursive: true, force: true });
  }
});
