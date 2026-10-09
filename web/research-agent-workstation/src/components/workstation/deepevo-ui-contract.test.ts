import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const ROOT = process.cwd();

async function text(relativePath: string) {
  return readFile(path.join(ROOT, relativePath), "utf8");
}

test("public workstation branding is DeepEvo while runtime compatibility identifiers remain unchanged", async () => {
  const [layout, sidebar, assistant] = await Promise.all([
    text("src/app/layout.tsx"),
    text("src/components/workstation/Sidebar.tsx"),
    text("src/components/workstation/screens/AssistantScreen.tsx"),
  ]);

  assert.match(layout, /DeepEvo Research Workstation/);
  assert.match(sidebar, /DeepEvoLockup/);
  assert.match(assistant, /DeepEvo Assistant/);
  assert.match(assistant, /X-EvoMind-CSRF/);
  assert.doesNotMatch(assistant, /X-DeepEvo-CSRF/);
});

test("assistant composer uses the concise professional placeholder", async () => {
  const assistant = await text("src/components/workstation/screens/AssistantScreen.tsx");
  assert.match(assistant, /输入任务内容…/);
  assert.match(assistant, /Enter a task…/);
  assert.doesNotMatch(assistant, /一句话交代任务/);
});

test("teacher-provided DeepEvo lockup is byte-identical and wired to a reduced-motion-safe orbit", async () => {
  const [asset, brand, styles] = await Promise.all([
    readFile(path.join(ROOT, "public/brand/deepevo-lockup.png")),
    text("src/components/workstation/DeepEvoBrand.tsx"),
    text("src/app/globals.css"),
  ]);

  assert.equal(createHash("sha256").update(asset).digest("hex"), "2a180c85832fd34953f94921b1d7bc8a95460a83ad4253a77d6b7ed6532f16ae");
  assert.match(brand, /\/brand\/deepevo-lockup\.png/);
  assert.match(styles, /@keyframes deepevo-orbit-spin/);
  assert.match(styles, /prefers-reduced-motion: reduce/);
});
