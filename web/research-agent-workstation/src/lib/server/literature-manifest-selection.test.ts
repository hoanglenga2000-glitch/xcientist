import assert from "node:assert/strict";
import test from "node:test";
import {
  selectLatestValidLiteratureManifest,
  type LiteratureManifestCandidate
// @ts-expect-error Node's native TypeScript test runner requires the explicit extension.
} from "./literature-manifest-selection.ts";

const taskId = "siim-isic-melanoma-classification";

function candidate(
  absolutePath: string,
  mtimeMs: number,
  papers: unknown[],
  externalVerified: number,
  expectedTaskId = taskId,
  manifestTaskId = taskId
): LiteratureManifestCandidate {
  return {
    absolutePath,
    expectedTaskId,
    mtimeMs,
    payload: {
      task_id: manifestTaskId,
      papers,
      integrity: { external_verified: externalVerified }
    }
  };
}

test("latest empty manifest does not hide the latest externally verified manifest", () => {
  const selected = selectLatestValidLiteratureManifest([
    candidate("context_latest_empty.json", 400, [], 0),
    candidate("context_latest_valid.json", 300, [{ id: "paper-1" }], 7),
    candidate("context_older_valid.json", 200, [{ id: "paper-2" }], 10)
  ]);

  assert.equal(selected?.absolutePath, "context_latest_valid.json");
});

test("externally verified manifests are preferred over newer local-only manifests", () => {
  const selected = selectLatestValidLiteratureManifest([
    candidate("context_newer_local.json", 400, [{ id: "local" }], 0),
    candidate("context_verified.json", 300, [{ id: "external" }], 1)
  ]);

  assert.equal(selected?.absolutePath, "context_verified.json");
});

test("task mismatches and manifests without papers are ignored", () => {
  const selected = selectLatestValidLiteratureManifest([
    candidate("context_wrong_task.json", 500, [{ id: "wrong" }], 5, taskId, "other-task"),
    candidate("context_empty.json", 400, [], 5)
  ]);

  assert.equal(selected, null);
});
