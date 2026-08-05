import assert from "node:assert/strict";
import test from "node:test";

// @ts-expect-error Node's strip-types test runner requires the explicit .ts suffix.
import { classifyMissingScientificReport } from "./scientific-report-state.ts";

test("missing reports remain pending only for active review or generation", () => {
  for (const status of ["awaiting_human_gate", "approved", "starting", "running", "needs_continuation"]) {
    assert.equal(classifyMissingScientificReport(status, null), "pending");
  }
  assert.equal(classifyMissingScientificReport("completed", "running"), "pending");
});

test("generation failures and terminal run states are terminal", () => {
  assert.equal(classifyMissingScientificReport("running", "failed"), "generation_failed");
  assert.equal(classifyMissingScientificReport("failed", null), "generation_failed");
  assert.equal(classifyMissingScientificReport("cancelled", null), "generation_failed");
});

test("a ready ledger without a manifest and unknown states fail integrity", () => {
  assert.equal(classifyMissingScientificReport("completed", "ready"), "integrity_failed");
  assert.equal(classifyMissingScientificReport("corrupt-state", null), "integrity_failed");
  assert.equal(classifyMissingScientificReport("completed", null), "not_found");
});
