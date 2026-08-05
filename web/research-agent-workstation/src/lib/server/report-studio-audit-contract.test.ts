import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = process.cwd();

test("SIIM audit view distinguishes the pre-grader review check from the terminal grader record", async () => {
  const source = await readFile(
    path.join(root, "src", "components", "workstation", "screens", "ReportStudioScreen.tsx"),
    "utf8"
  );

  assert.match(source, /private grader not executed at Independent Review time/);
  assert.match(source, /Private grader terminal record/);
  assert.match(source, /privateGrader\.execution_count/);
  assert.match(source, /privateGrader\.score/);
});
