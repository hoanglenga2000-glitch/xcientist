import assert from "node:assert/strict";
import test from "node:test";

import {
  evaluateRuntimeIdentity,
  parseRuntimeBuildManifest,
  type RuntimeIdentityObservation,
// @ts-expect-error Node's native TypeScript test runner requires the explicit extension.
} from "./runtime-version-contract.ts";

const manifest = {
  schema: "evomind.runtime_build.v1",
  commit_hash: "b".repeat(40),
  source_dirty: true,
  source_tree_sha256: "a".repeat(64),
  build_id: "build-123",
  build_time: "2026-08-09T00:00:00Z",
  backend_version: "0.3.0",
  frontend_version: "0.3.0",
  database_schema_version: "20260728171000_performance_indexes",
  database_schema_sha256: "c".repeat(64),
};

const observation: RuntimeIdentityObservation = {
  launched_source_tree_sha256: "a".repeat(64),
  frontend_version: "0.3.0",
  database_schema_sha256: "c".repeat(64),
  runtime: {
    reachable: true,
    backend_version: "0.3.0",
    commit_hash: "b".repeat(40),
    source_tree_sha256: "a".repeat(64),
  },
};

test("runtime build manifest rejects missing or malformed identity fields", () => {
  assert.equal(parseRuntimeBuildManifest(null), null);
  assert.equal(parseRuntimeBuildManifest({ ...manifest, commit_hash: "short" }), null);
  assert.equal(parseRuntimeBuildManifest({ ...manifest, database_schema_sha256: "nope" }), null);
  assert.deepEqual(parseRuntimeBuildManifest(manifest), manifest);
});

test("runtime identity is ready only when every build boundary matches", () => {
  const result = evaluateRuntimeIdentity(manifest, observation);
  assert.equal(result.status, "ready");
  assert.equal(result.ready, true);
  assert.deepEqual(result.failures, []);
});

test("runtime identity fails closed for source backend frontend and database drift", () => {
  const result = evaluateRuntimeIdentity(manifest, {
    ...observation,
    launched_source_tree_sha256: "d".repeat(64),
    frontend_version: "0.3.1",
    database_schema_sha256: "e".repeat(64),
    runtime: {
      reachable: true,
      backend_version: "0.2.0",
      commit_hash: "f".repeat(40),
      source_tree_sha256: "9".repeat(64),
    },
  });
  assert.equal(result.status, "error");
  assert.equal(result.ready, false);
  assert.deepEqual(result.failures, [
    "source_tree_mismatch",
    "frontend_version_mismatch",
    "database_schema_mismatch",
    "backend_version_mismatch",
    "runtime_commit_mismatch",
    "runtime_source_tree_mismatch",
  ]);
});

test("runtime identity fails closed when the Python runtime is unavailable", () => {
  const result = evaluateRuntimeIdentity(manifest, {
    ...observation,
    runtime: { reachable: false },
  });
  assert.equal(result.ready, false);
  assert.deepEqual(result.failures, ["runtime_unavailable"]);
});
