import assert from "node:assert/strict";
import test from "node:test";
import { latestAssistantMessageId, restoreRunMessages, updateRunMessage, updateStreamingMessage } from "./assistant-history.ts";
import { isAssistantExecutionRecord } from "./assistant-presentation.ts";

const old = { id: "old", role: "assistant", runId: "run_a", content: "Previous failure", status: "blocked", artifacts: [{ id: "old-receipt" }] };
const live = { id: "live", role: "assistant", runId: "run_a", content: "", status: "streaming" };

test("cumulative streaming replaces replayed text and protects frozen replies", () => {
  const first = updateStreamingMessage([old, live], "run_a", "live", "first");
  const replay = updateStreamingMessage(first, "run_a", "live", "first");
  assert.equal(replay[1].content, "first");
  assert.equal(replay[0], old);
  assert.equal(updateStreamingMessage(replay, "run_a", "old", "stale"), replay);
  assert.equal(updateStreamingMessage(replay, "run_a", "live", "")[1].content, "");
});

test("refresh restores durable partial text only into the active message", () => {
  const run = { id: "run_a", status: "running", streaming_text: "fresh partial", message_history: {
    schema: "evomind.assistant_message_history.v1", active_message_id: "live",
    messages: [{ ...old }, { ...live }],
  } };
  const restored = restoreRunMessages(run, "Continue");
  assert.equal(restored[0].content, old.content);
  assert.equal(restored[1].content, "fresh partial");
  assert.equal(updateRunMessage([old, live], run, "live")[1].content, "fresh partial");
  assert.equal(updateStreamingMessage([{ ...live, status: "complete" }], "run_a", "live", "late")[0].content, "");
});

test("snapshots update only the current reply and never inherit Run-wide artifacts", () => {
  const updated = updateRunMessage([old, live], { id: "run_a", status: "completed", answer: "New verified result", artifacts: [{ id: "other-attempt" }] }, "live");
  assert.equal(updated[0], old);
  assert.equal(updated[1].content, "New verified result");
  assert.equal(updated[1].artifacts, undefined);
  assert.equal(latestAssistantMessageId(updated, "run_a"), "live");
});

test("resuming cannot display an old answer as a new response", () => {
  const updated = updateRunMessage([old, live], { id: "run_a", status: "running", answer: old.content }, "live");
  assert.equal(updated[0], old);
  assert.equal(updated[1].content, "");
  assert.equal(updated[1].status, "streaming");
  assert.equal(updateRunMessage([old, live], { id: "run_a", status: "waiting_approval", answer: old.content, answer_is_current: false }, "live")[1].content, "");
});

test("late old-stream updates cannot overwrite an earlier reply", () => {
  const messages = [old, live];
  assert.equal(updateRunMessage(messages, { id: "run_a", status: "failed", answer: "stale failure" }, "old"), messages);
  assert.equal(updateRunMessage(messages, { id: "run_a", status: "completed", answer: "settled" }, "missing-default-id")[1].content, "settled");
});

test("durable history restores each response without running or associating old artifacts", () => {
  const run = { id: "run_a", status: "waiting_approval", prompt: "Fit one model", answer_is_current: false, artifacts: [{ id: "old-receipt" }], message_history: {
    schema: "evomind.assistant_message_history.v1", active_message_id: "active",
    messages: [{ id: "user", role: "user", content: "Fit one model", kind: "original_request" }, { id: "old", role: "assistant", content: "Previous failure" }, { id: "resume", role: "user", content: "", kind: "continued_execution" }, { id: "active", role: "assistant", content: "" }],
  } };
  const restored = restoreRunMessages(run, "Continue this Run (recovery record)");
  assert.equal(restored.length, 4);
  assert.equal(restored[1].content, "Previous failure");
  assert.equal(restored[2].content, "Continue this Run (recovery record)");
  assert.equal(restored[3].status, "blocked");
  assert.ok(restored.every((row) => row.artifacts === undefined));
});

test("orchestration and audit files are execution records, not delivered models", () => {
  for (const name of ["task_graph.json", "claim-audit.json", "capability-contract.json", "search-selection.json"]) assert.equal(isAssistantExecutionRecord(name), true);
  assert.equal(isAssistantExecutionRecord("model.safetensors"), false);
});
