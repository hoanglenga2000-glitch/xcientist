import assert from "node:assert/strict";
import test from "node:test";

// @ts-expect-error Node's strip-types test runner requires the explicit .ts suffix.
import { ScientificReportPendingTimeoutError, ScientificReportRequestError, pollScientificReport } from "../api/report-polling.ts";

function response(status: number, payload: unknown) {
  return { status, async json() { return payload; } };
}

const expected = { taskId: "task-one", runId: "run-00000001" };
const pending = {
  ok: false,
  task_id: expected.taskId,
  run_id: expected.runId,
  status: "pending_report",
  run_status: "reviewing",
};
const ready = {
  ok: true,
  task_id: expected.taskId,
  report: { task_id: expected.taskId, run_id: expected.runId, status: "ready" },
};

test("internal report polling resolves a 202 to 200 transition with capped backoff", async () => {
  const queue = [response(202, pending), response(202, pending), response(200, ready)];
  const delays: number[] = [];
  let clock = 100;
  const result = await pollScientificReport(
    async () => queue.shift()!,
    expected,
    { initialDelayMs: 10, maxDelayMs: 15, maxAttempts: 4, maxElapsedMs: 100 },
    {
      now: () => clock,
      sleep: async (milliseconds) => { delays.push(milliseconds); clock += milliseconds; },
    },
  );
  assert.deepEqual(result, ready);
  assert.deepEqual(delays, [10, 15]);
});

test("permanent 202 responses stop at the attempt budget", async () => {
  let attempts = 0;
  await assert.rejects(
    pollScientificReport(
      async () => { attempts += 1; return response(202, pending); },
      expected,
      { initialDelayMs: 1, maxDelayMs: 2, maxAttempts: 3, maxElapsedMs: 100 },
      { now: () => attempts, sleep: async () => undefined },
    ),
    (error: unknown) => error instanceof ScientificReportPendingTimeoutError
      && error.code === "pending_report_timeout"
      && error.attempts === 3,
  );
  assert.equal(attempts, 3);
});

test("the elapsed-time budget is a hard polling limit", async () => {
  let clock = 0;
  let attempts = 0;
  await assert.rejects(
    pollScientificReport(
      async () => { attempts += 1; return response(202, pending); },
      expected,
      { initialDelayMs: 7, maxDelayMs: 20, maxAttempts: 20, maxElapsedMs: 10 },
      { now: () => clock, sleep: async (milliseconds) => { clock += milliseconds; } },
    ),
    (error: unknown) => error instanceof ScientificReportPendingTimeoutError && error.elapsedMs === 10,
  );
  assert.equal(attempts, 2);
});

test("409 integrity failures are terminal and are not retried", async () => {
  let attempts = 0;
  await assert.rejects(
    pollScientificReport(
      async () => { attempts += 1; return response(409, { ok: false, code: "report_integrity_failed", error: "hash drift" }); },
      expected,
    ),
    (error: unknown) => error instanceof ScientificReportRequestError
      && error.status === 409
      && error.code === "report_integrity_failed",
  );
  assert.equal(attempts, 1);
});

test("malformed 202 responses fail immediately", async () => {
  let attempts = 0;
  await assert.rejects(
    pollScientificReport(
      async () => { attempts += 1; return response(202, { ...pending, ok: true }); },
      expected,
    ),
    (error: unknown) => error instanceof ScientificReportRequestError && error.code === "malformed_pending_report",
  );
  assert.equal(attempts, 1);
});

test("pending and ready binding drift fail closed", async (t) => {
  await t.test("pending binding", async () => {
    await assert.rejects(
      pollScientificReport(async () => response(202, { ...pending, run_id: "run-00000002" }), expected),
      (error: unknown) => error instanceof ScientificReportRequestError && error.code === "pending_report_binding_drift",
    );
  });
  await t.test("ready binding", async () => {
    await assert.rejects(
      pollScientificReport(
        async () => response(200, { ...ready, report: { ...ready.report, task_id: "other-task" } }),
        expected,
      ),
      (error: unknown) => error instanceof ScientificReportRequestError && error.code === "report_binding_drift",
    );
  });
});

test("AbortSignal cancels polling before another request is sent", async () => {
  const controller = new AbortController();
  let attempts = 0;
  await assert.rejects(
    pollScientificReport(
      async (signal) => {
        attempts += 1;
        assert.equal(signal, controller.signal);
        controller.abort();
        return response(202, pending);
      },
      expected,
      { signal: controller.signal },
    ),
    (error: unknown) => error instanceof DOMException && error.name === "AbortError",
  );
  assert.equal(attempts, 1);
});
