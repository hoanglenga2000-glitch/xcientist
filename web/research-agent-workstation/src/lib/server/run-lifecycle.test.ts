import assert from "node:assert/strict";
import test from "node:test";

import {
  assertRunTransition,
  canExecuteRun,
// @ts-expect-error Node's native TypeScript test runner requires the explicit extension.
} from "./run-lifecycle.ts";

test("run lifecycle cannot execute before the plan gate", () => {
  assert.doesNotThrow(() => assertRunTransition("CREATED", "PLANNING"));
  assert.doesNotThrow(() => assertRunTransition("PLANNING", "WAIT_PLAN_GATE"));
  assert.throws(() => assertRunTransition("WAIT_PLAN_GATE", "EXECUTING"), /Illegal run transition/);
  assert.equal(canExecuteRun("WAIT_PLAN_GATE", "pending"), false);
  assert.equal(canExecuteRun("WAIT_PLAN_GATE", "approved"), false);
  assert.equal(canExecuteRun("APPROVED", "pending"), false);
  assert.equal(canExecuteRun("APPROVED", "approved"), true);
});

test("run lifecycle preserves result and reporting gates", () => {
  assert.doesNotThrow(() => assertRunTransition("APPROVED", "EXECUTING"));
  assert.doesNotThrow(() => assertRunTransition("EXECUTING", "WAIT_RESULT_GATE"));
  assert.throws(() => assertRunTransition("WAIT_RESULT_GATE", "COMPLETED"), /Illegal run transition/);
  assert.doesNotThrow(() => assertRunTransition("WAIT_RESULT_GATE", "REPORTING"));
  assert.doesNotThrow(() => assertRunTransition("REPORTING", "COMPLETED"));
});
