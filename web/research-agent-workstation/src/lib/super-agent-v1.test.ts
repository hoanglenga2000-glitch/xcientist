import assert from "node:assert/strict";
import test from "node:test";

// @ts-expect-error Node's native TypeScript test runner requires the explicit extension.
import { SUPER_AGENT_STATUS_SCHEMA, isSuperAgentV1Status, sanitizeSuperAgentV1Status } from "./super-agent-v1.ts";

const validStatus = () => ({
  schema: "evomind.super_agent_runtime.v1",
  mode: "shadow",
  enabled: false,
  shadow: true,
  migration_applied: true,
  capability_count: 1,
  directory_count: 1,
  directory_ids_sha256: "a".repeat(64),
  active_run_ids: ["run_demo"],
  directories: [
    {
      directory_id: "workspace",
      connector_id: "local-workspace",
      opaque_root_ref: "D:\\private\\workspace",
      operations: ["list", "read", "hash", "write"],
      tenant_id: "tenant_private",
      project_id: "project_private",
      max_files: 100,
      max_bytes: 1024,
      symlink_or_reparse_policy: "deny",
      expires_at: "",
      credential_lease_ref: "lease_private",
      approval_fingerprint: "fingerprint_private",
      health_receipt: {
        ok: true,
        secret: "must-never-cross-the-api-boundary",
      },
    },
  ],
  runs: [
    {
      run_id: "run_demo",
      status: "waiting_exact_gate",
      objective: "Inspect D:\\private\\dataset and use api_key=do-not-render",
      nodes: [
        {
          id: "discover",
          action: "Discover authorized capabilities",
          status: "completed",
          dependencies: [],
          capability_ids: ["workspace"],
        },
        {
          id: "execute",
          action: "Resume at /home/private/run without exposing the path",
          status: "waiting_exact_gate",
          dependencies: ["discover"],
          capability_ids: ["workspace"],
        },
      ],
      failure_count: 1,
      repair_count: 1,
      exact_gate: {
        id: "gate_demo",
        run_id: "run_demo",
        node_id: "execute",
        reason: "Endpoint 10.0.0.8:22 is unavailable",
        required_action: "Provide changed precondition evidence",
        resume_point: "execute",
        failure_signature: "b".repeat(64),
        precondition_fingerprint: "c".repeat(64),
        created_at: "2026-08-26T00:00:00.000Z",
      },
    },
  ],
  unexpected_secret: "never projected",
});

test("projects only the browser-safe Super Agent status contract", () => {
  const projected = sanitizeSuperAgentV1Status(validStatus());

  assert.equal(projected.schema, SUPER_AGENT_STATUS_SCHEMA);
  assert.equal(projected.mode, "shadow");
  assert.equal(projected.mounts[0]?.health, "ready");
  assert.deepEqual(projected.mounts[0]?.operations, ["list", "read", "hash", "write"]);
  assert.match(projected.runs[0]?.objective ?? "", /\[path redacted\]/);
  assert.match(projected.runs[0]?.objective ?? "", /\[credential redacted\]/);
  assert.match(projected.runs[0]?.exact_gate?.reason ?? "", /\[infrastructure endpoint redacted\]/);
  assert.equal(isSuperAgentV1Status(projected), true);

  const serialized = JSON.stringify(projected);
  for (const forbidden of [
    "opaque_root_ref",
    "tenant_private",
    "project_private",
    "lease_private",
    "fingerprint_private",
    "must-never-cross-the-api-boundary",
    "do-not-render",
    "10.0.0.8",
    "unexpected_secret",
    "failure_signature",
    "precondition_fingerprint",
  ]) {
    assert.doesNotMatch(serialized, new RegExp(forbidden));
  }
});

test("fails closed on schema, mode, count, digest and mount contract drift", () => {
  const mutations = [
    { schema: "evomind.super_agent_runtime.v2" },
    { mode: "enabled", enabled: false, shadow: false },
    { capability_count: 0, directory_count: 1 },
    { directory_ids_sha256: "not-a-digest" },
    { directories: [{ ...validStatus().directories[0], symlink_or_reparse_policy: "allow" }] },
  ];

  for (const mutation of mutations) {
    assert.throws(
      () => sanitizeSuperAgentV1Status({ ...validStatus(), ...mutation }),
      /managed Super Agent status did not satisfy the public contract/,
    );
  }
});

test("rejects malformed runs and cross-run Exact Gates", () => {
  const malformedNode = validStatus();
  malformedNode.runs[0]!.nodes[0]!.status = "invented_state";
  assert.throws(() => sanitizeSuperAgentV1Status(malformedNode));

  const crossRunGate = validStatus();
  crossRunGate.runs[0]!.exact_gate!.run_id = "run_other";
  assert.throws(() => sanitizeSuperAgentV1Status(crossRunGate));
});

test("status type guard rejects error and partial responses", () => {
  assert.equal(isSuperAgentV1Status({ ok: false, error: "runtime_not_ready" }), false);
  assert.equal(isSuperAgentV1Status({ ok: true, schema: SUPER_AGENT_STATUS_SCHEMA }), false);
});
