import assert from "node:assert/strict";
import test from "node:test";

// @ts-expect-error Node's strip-types test runner requires the explicit .ts suffix.
import { reviewedExistingReportEligible } from "./reviewed-existing-report.ts";

function fixture() {
  const taskId = "titanic";
  const runId = "wr_2026-08-09T12-27-37-867Z_w9i07";
  return {
    taskId,
    runId,
    run: { run_id: runId, status: "completed" },
    review: { status: "passed", claim_audit: { status: "passed" } },
    artifactManifest: {
      schema: "evomind.artifact_manifest.v1",
      run_id: runId,
      selected_solution: "solution_02",
      artifacts: [{ path: "research_report.md", sha256: "a".repeat(64), bytes: 10 }],
    },
    pointer: { task_id: taskId, run_id: runId },
  };
}

test("reviewed existing report accepts one fully bound completed run", () => {
  assert.equal(reviewedExistingReportEligible(fixture()), true);
});

test("reviewed existing report fails closed on task, review, claim, or manifest drift", () => {
  const base = fixture();
  assert.equal(reviewedExistingReportEligible({ ...base, pointer: { task_id: "other", run_id: base.runId } }), false);
  assert.equal(reviewedExistingReportEligible({ ...base, review: { status: "failed", claim_audit: { status: "passed" } } }), false);
  assert.equal(reviewedExistingReportEligible({ ...base, review: { status: "passed", claim_audit: { status: "failed" } } }), false);
  assert.equal(reviewedExistingReportEligible({ ...base, artifactManifest: { ...base.artifactManifest, run_id: "other" } }), false);
});
