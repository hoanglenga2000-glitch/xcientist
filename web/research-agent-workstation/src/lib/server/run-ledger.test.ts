import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

test("Run Ledger keeps one canonical state while preserving component projections", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "evomind-run-ledger-"));
  const previousRoot = process.env.WORKSTATION_ROOT;
  process.env.WORKSTATION_ROOT = root;
  try {
    const modulePath = `./run-ledger.ts?root=${Date.now()}`;
    const { writeRunLedger, readCurrentRunLedger } = await import(modulePath);
    await writeRunLedger({
      taskId: "siim-isic-melanoma-classification",
      runId: "wr_test_ledger_001",
      status: "EXECUTING",
      lifecycleState: "EXECUTING",
      source: "workstation_database",
    });
    const pythonProjection = await writeRunLedger({
      taskId: "siim-isic-melanoma-classification",
      runId: "wr_test_ledger_001",
      status: "needs_continuation",
      source: "python_supervisor",
      lastSeq: 20,
    });
    assert.equal(pythonProjection.status, "EXECUTING");
    assert.equal(pythonProjection.projections.python_supervisor.status, "needs_continuation");

    const failed = await writeRunLedger({
      taskId: "siim-isic-melanoma-classification",
      runId: "wr_test_ledger_001",
      status: "FAILED",
      lifecycleState: "FAILED",
      source: "workstation_database",
    });
    assert.equal(failed.status, "FAILED");
    assert.equal((await readCurrentRunLedger())?.status, "FAILED");
    const compatibility = JSON.parse(await readFile(path.join(root, "workspace", "current_run.json"), "utf-8"));
    assert.equal(compatibility.status, "FAILED");
    assert.equal(compatibility.ledger_revision, 3);
    const events = (await readFile(path.join(root, "workspace", "run_ledger", "events.jsonl"), "utf-8")).trim().split("\n");
    assert.equal(events.length, 3);
  } finally {
    if (previousRoot === undefined) delete process.env.WORKSTATION_ROOT;
    else process.env.WORKSTATION_ROOT = previousRoot;
    await rm(root, { recursive: true, force: true });
  }
});
