import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

// @ts-expect-error Node's strip-types test runner requires the explicit .ts suffix.
import { consumeEvolutionApprovalPlan, createEvolutionApprovalPlan } from "./evolution-approval.ts";
// @ts-expect-error Node's strip-types test runner requires the explicit .ts suffix.
import { CANONICAL_HASH_SCHEMA, sha256Canonical } from "./evolution-integrity.ts";

const request = {
  task_id: "fixture",
  engine: "research_os",
  runner: "local",
  search_mode: "experience_mcgs_v1",
  iterations: 8,
  max_nodes: 8,
  max_tokens: 100000,
  max_wall_seconds: 600,
  max_cost: 0.01,
  mcgs: true,
};

test("approval receipt binds one exact plan and is single use", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "evomind-approval-"));
  const created = await createEvolutionApprovalPlan("fixture", { ...request, approve: false }, { selected_branch: "EXP001" }, {
    storageRoot: root,
    now: new Date("2026-08-02T12:00:00Z"),
  });
  const approvedInput = {
    ...request,
    approve: true,
    plan_id: created.plan_id,
    plan_sha256: created.plan_sha256,
    request_fingerprint: created.request_fingerprint,
  };
  const approved = await consumeEvolutionApprovalPlan("fixture", approvedInput, {
    storageRoot: root,
    now: new Date("2026-08-02T12:01:00Z"),
  });
  assert.equal(approved.status, "approved");
  assert.deepEqual(approved.plan, { selected_branch: "EXP001" });
  assert.deepEqual(approved.request_contract, request);
  assert.equal(approved.hash_canonicalization, CANONICAL_HASH_SCHEMA);
  assert.equal(approved.request_fingerprint, sha256Canonical(approved.request_contract));
  assert.equal(approved.plan_sha256, sha256Canonical(approved.plan));
  await assert.rejects(() => consumeEvolutionApprovalPlan("fixture", approvedInput, { storageRoot: root }), /binding verification failed/);
});

test("approval receipt has an atomic one-consumer boundary", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "evomind-approval-"));
  const created = await createEvolutionApprovalPlan("fixture", request, { selected_branch: "EXP001" }, {
    storageRoot: root,
    now: new Date("2026-08-02T12:00:00Z"),
  });
  const approvedInput = {
    ...request,
    approve: true,
    plan_id: created.plan_id,
    plan_sha256: created.plan_sha256,
    request_fingerprint: created.request_fingerprint,
  };
  const results = await Promise.allSettled([
    consumeEvolutionApprovalPlan("fixture", approvedInput, { storageRoot: root, now: new Date("2026-08-02T12:01:00Z") }),
    consumeEvolutionApprovalPlan("fixture", approvedInput, { storageRoot: root, now: new Date("2026-08-02T12:01:00Z") }),
  ]);
  assert.equal(results.filter((result) => result.status === "fulfilled").length, 1);
  assert.equal(results.filter((result) => result.status === "rejected").length, 1);
});

test("approval receipt rejects values outside the canonical JSON contract", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "evomind-approval-"));
  const invalidRequests: Record<string, unknown>[] = [
    { ...request, max_cost: Number.NaN },
    { ...request, max_tokens: 2 ** 53 },
    { ...request, extra: new Date("2026-08-02T12:00:00Z") },
    { ...request, extra: new Set(["aliased"]) },
    { ...request, extra: undefined },
  ];
  for (const invalid of invalidRequests) {
    await assert.rejects(
      () => createEvolutionApprovalPlan("fixture", invalid, { selected_branch: "EXP001" }, { storageRoot: root }),
      TypeError,
    );
  }
  await assert.rejects(
    () => createEvolutionApprovalPlan("../escaped", request, { selected_branch: "EXP001" }, { storageRoot: root }),
    /Invalid evolution approval task_id/,
  );
  await assert.rejects(
    () => createEvolutionApprovalPlan("fixture", request, { selected_branch: undefined }, { storageRoot: root }),
    TypeError,
  );
});

test("approval receipt rejects request drift, tampering, and expiry", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "evomind-approval-"));
  const created = await createEvolutionApprovalPlan("fixture", { ...request, approve: false }, { selected_branch: "EXP002" }, {
    storageRoot: root,
    now: new Date("2026-08-02T12:00:00Z"),
    ttlSeconds: 60,
  });
  const bound = {
    ...request,
    approve: true,
    plan_id: created.plan_id,
    plan_sha256: created.plan_sha256,
    request_fingerprint: created.request_fingerprint,
  };
  await assert.rejects(
    () => consumeEvolutionApprovalPlan("fixture", { ...bound, iterations: 7 }, { storageRoot: root, now: new Date("2026-08-02T12:00:30Z") }),
    /binding verification failed/,
  );
  await assert.rejects(
    () => consumeEvolutionApprovalPlan("fixture", bound, { storageRoot: root, now: new Date("2026-08-02T12:01:00Z") }),
    /expired/,
  );

  const file = path.join(root, "workspace", "evolution", "approvals", "fixture", `${created.plan_id}.json`);
  const tampered = JSON.parse(await readFile(file, "utf8"));
  tampered.plan.selected_branch = "EXP999";
  await writeFile(file, `${JSON.stringify(tampered)}\n`, "utf8");
  await assert.rejects(
    () => consumeEvolutionApprovalPlan("fixture", bound, { storageRoot: root, now: new Date("2026-08-02T12:00:30Z") }),
    /binding verification failed/,
  );
});
