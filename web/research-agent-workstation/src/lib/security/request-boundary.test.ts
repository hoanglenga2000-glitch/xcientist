import assert from "node:assert/strict";
import test from "node:test";

// @ts-ignore Node's strip-types test runner requires the explicit .ts suffix.
import { normalizeTaskId } from "./request-boundary.ts";


test("normalizeTaskId accepts registry-backed ISO-style task ids", () => {
  assert.equal(
    normalizeTaskId("task_2026-06-12T06-28-14-863Z"),
    "task_2026-06-12T06-28-14-863Z",
  );
});

test("normalizeTaskId keeps path traversal and reserved names blocked", () => {
  for (const candidate of ["../task", "task%2Fchild", "task\\child", "NUL", "task."]) {
    assert.throws(() => normalizeTaskId(candidate), /Invalid task ID/);
  }
});
