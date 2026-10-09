import assert from "node:assert/strict";
import test from "node:test";
import { runProgressView } from "./run-progress-presentation.ts";

const now = 100_000;
const progress = { source: "managed_adapter", work_kind: "training", heartbeat_at: 99, completed_units: 2, total_units: 8, percent: 99 };

test("percent is calculated from measured work rather than a claimed percentage", () => {
  assert.equal(runProgressView(progress, "running", now).percent, 25);
  for (const patch of [{ total_units: 0 }, { completed_units: 9 }, { completed_units: NaN }, { total_units: Infinity }, { source: "model" }]) assert.equal(runProgressView({ ...progress, ...patch }, "running", now).percent, null);
});

test("approval and terminal states do not retain a training percentage", () => {
  for (const status of ["waiting_approval", "completed", "blocked", "failed", "cancelled"]) {
    const view = runProgressView(progress, status, now);
    assert.equal(view.percent, null);
    assert.equal(view.detailKey, status);
  }
  assert.equal(runProgressView(undefined, "waiting_approval", now).stale, false);
});

test("stale or future-dated heartbeats cannot establish live training", () => {
  for (const heartbeat_at of [undefined, 1, 200, NaN]) {
    const view = runProgressView({ ...progress, heartbeat_at }, "running", now);
    assert.equal(view.stale, true);
    assert.equal(view.percent, null);
  }
});

test("model service waiting is distinct from fitting and retains raw diagnostics", () => {
  const detail = "Research role is awaiting the configured model";
  const view = runProgressView({ source: "executor", phase: "research_model_call", heartbeat_at: 99, detail, stalled: true }, "running", now);
  assert.equal(view.kind, "model_wait");
  assert.equal(view.detailKey, "model_wait");
  assert.equal(view.rawDiagnostic, detail);
  assert.equal(view.stalled, true);
  assert.equal(view.percent, null);
});
