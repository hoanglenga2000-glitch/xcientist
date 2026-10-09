import assert from "node:assert/strict";
import test from "node:test";
import { assistantActivityIdentity, upsertAssistantActivity, type AssistantActivityRecord } from "./assistant-activity.mjs";

const item = (identity: string, seq: number, status: string): AssistantActivityRecord => ({
  identity,
  seq,
  type: "tool_completed",
  status,
  label: status,
});

test("tool start and completion share one activity row", () => {
  const identity = assistantActivityIdentity({ type: "tool_started", seq: 1, toolCallId: "call-1" });
  const completedIdentity = assistantActivityIdentity({ type: "tool_completed", seq: 2, toolCallId: "call-1" });
  assert.equal(identity, completedIdentity);
  const rows = upsertAssistantActivity(upsertAssistantActivity([], item(identity, 1, "running")), item(completedIdentity, 2, "completed"));
  assert.equal(rows.length, 1);
  assert.equal(rows[0].status, "completed");
});

test("model observations collapse to one technical activity", () => {
  const a = assistantActivityIdentity({ type: "model_observed", seq: 4 });
  const b = assistantActivityIdentity({ type: "model_observed", seq: 5 });
  assert.equal(a, "model");
  assert.equal(a, b);
});

test("terminal blocker replaces stale terminal activity", () => {
  const first = item(assistantActivityIdentity({ type: "run_blocked", seq: 8 }), 8, "blocked");
  const second = item(assistantActivityIdentity({ type: "run_failed", seq: 9 }), 9, "error");
  const rows = upsertAssistantActivity([first], second);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].status, "error");
});
