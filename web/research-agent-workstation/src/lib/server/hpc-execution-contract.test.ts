import assert from "node:assert/strict";
import test from "node:test";

import {
  buildHpcSubprocessEnvironment,
  HpcExecutionContractError,
  parseHpcExecutionContract,
  taskRequiresHpcExecutionContract,
// @ts-expect-error Node's native TypeScript test runner requires the explicit extension.
} from "./hpc-execution-contract.ts";

test("SIIM dispatch requires the complete HPC execution contract", () => {
  assert.equal(taskRequiresHpcExecutionContract("siim-isic-melanoma-classification", {}), true);
  assert.throws(
    () => parseHpcExecutionContract({}, { required: true }),
    (error: unknown) => {
      assert.equal(error instanceof HpcExecutionContractError, true);
      assert.equal((error as Error).message, "Missing HPC execution contract");
      assert.deepEqual((error as HpcExecutionContractError).missingFields, [
        "job_id",
        "credential_profile",
        "resource_profile",
        "execution_backend",
      ]);
      return true;
    },
  );
});

test("HPC execution contract rejects partial or cross-job bindings", () => {
  assert.throws(
    () => parseHpcExecutionContract({
      job_id: 90353,
      credential_profile: "job90948",
      resource_profile: "aimslab_a800_80gb",
      execution_backend: "hpc",
    }, { required: true }),
    /Invalid HPC execution contract/,
  );
  assert.throws(
    () => parseHpcExecutionContract({
      job_id: 90353,
      credential_profile: "job90353",
      execution_backend: "hpc",
    }, { required: true }),
    /Missing HPC execution contract/,
  );
});

test("complete HPC execution contract is normalized without ambient fallback", () => {
  assert.deepEqual(parseHpcExecutionContract({
    job_id: "90353",
    credential_profile: "job90353",
    resource_profile: "aimslab_a800_80gb",
    execution_backend: "hpc",
  }, { required: true }), {
    job_id: 90353,
    credential_profile: "job90353",
    resource_profile: "aimslab_a800_80gb",
    execution_backend: "hpc",
  });
});

test("generic research objectives that request HPC require the execution contract", () => {
  assert.equal(taskRequiresHpcExecutionContract("titanic", {
    objective: "请在 HPC 上训练，并且不要使用本地 GPU。",
  }), true);
  assert.equal(taskRequiresHpcExecutionContract("titanic", {
    objective: "Train the candidate on a remote GPU and return evidence.",
  }), true);
  assert.equal(taskRequiresHpcExecutionContract("titanic", {
    requires_hpc: true,
  }), true);
});

test("explicit local-only research objectives do not become HPC tasks", () => {
  assert.equal(taskRequiresHpcExecutionContract("titanic", {
    objective: "Do not use HPC; run the bounded CPU evaluation locally.",
  }), false);
  assert.equal(taskRequiresHpcExecutionContract("titanic", {
    objective: "只运行本地 CPU 评估，不使用 HPC。",
  }), false);
});

test("HPC subprocess environment removes ambient SSH overrides", () => {
  const environment = buildHpcSubprocessEnvironment({
    PATH: "C:/tools",
    GPU_SSH_HOST: "legacy-host",
    GPU_SSH_PASSWORD_FILE: "legacy-secret-file",
    GPU_SSH_KNOWN_HOSTS_PATH: "legacy-known-hosts",
    EVOMIND_HPC_EXPECTED_GPU_UUID: "legacy-gpu",
  }, {
    job_id: 91051,
    credential_profile: "job91051",
    resource_profile: "aimslab_a800_80gb",
    execution_backend: "hpc",
  });

  assert.equal(environment.PATH, "C:/tools");
  assert.equal(environment.GPU_SSH_HOST, undefined);
  assert.equal(environment.GPU_SSH_PASSWORD_FILE, undefined);
  assert.equal(environment.GPU_SSH_KNOWN_HOSTS_PATH, undefined);
  assert.equal(environment.EVOMIND_HPC_EXPECTED_GPU_UUID, undefined);
  assert.equal(environment.EVOMIND_SIIM_HPC_JOB_ID, "91051");
  assert.equal(environment.EVOMIND_HPC_CREDENTIAL_PROFILE, "job91051");
  assert.equal(environment.EVOMIND_HPC_RESOURCE_PROFILE, "aimslab_a800_80gb");
  assert.equal(environment.EVOMIND_EXECUTION_BACKEND, "hpc");
});
