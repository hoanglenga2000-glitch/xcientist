import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error Node's native TypeScript test runner requires the explicit extension.
import { normalizeConnectorHealth, normalizeConnectorRegistry } from "./connector-health.ts";

test("Connector Health Service emits only canonical states", () => {
  assert.equal(normalizeConnectorHealth("gpu", { configured: true, state: "GPU SSH Gateway Ready" }).state, "READY");
  assert.equal(normalizeConnectorHealth("gpu", { configured: true, state: "ready", current_allocation_blocked: true }).state, "OFFLINE");
  assert.equal(normalizeConnectorHealth("llm", { configured: true, state: "rule_based" }).state, "READY");
  assert.equal(normalizeConnectorHealth("api", { configured: true, state: "warning" }).state, "DEGRADED");
  assert.equal(normalizeConnectorHealth("api", { configured: false, state: "Not Configured" }).state, "NOT_CONFIGURED");
});

test("Connector Registry projects one canonical GPU state to the local_hpc alias", () => {
  const registry = normalizeConnectorRegistry({ gpu: { configured: true, state: "warning", name: "GPU" } });
  assert.equal(registry.gpu.state, "DEGRADED");
  assert.equal(registry.local_hpc.state, "DEGRADED");
  assert.equal(registry.local_hpc.raw_state, registry.gpu.raw_state);
});
