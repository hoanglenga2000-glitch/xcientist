import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("invitation mode retains navigation and five always-visible core entries", async () => {
  const shell = await readFile("src/components/workstation/AppShell.tsx", "utf8");
  const nav = await readFile("src/components/workstation/navigation.ts", "utf8");
  const sidebar = await readFile("src/components/workstation/Sidebar.tsx", "utf8");
  assert.doesNotMatch(shell, /if \(userDemoMode\)\s*\{\s*return/);
  assert.match(shell, /<Sidebar/);
  assert.match(nav, /ids: \["assistant", "projects", "data", "report", "settings"\]/);
  assert.match(sidebar, /navSectionCollapsed\(sectionId, activeSectionId, collapsed, collapsedSections\[sectionId\]\)/);
  assert.match(sidebar, /if \(section\.id !== "core"\)/);
  assert.match(nav, /labelZh: "高级工具"/);
});

test("assistant restores context without repeating execution and shows stale evidence", async () => {
  const ui = await readFile("src/components/workstation/screens/AssistantScreen.tsx", "utf8");
  const progress = await readFile("src/components/workstation/screens/run-progress-presentation.ts", "utf8");
  const page = await readFile("src/app/page.tsx", "utf8");
  assert.doesNotMatch(page, /searchParams\?\.get\("task"\)\) \|\| "playground_series_s6e6"/);
  assert.match(ui, /latestSnapshot\.last_event_seq \?\? 0/);
  assert.match(ui, /sessionStorage\.setItem\(draftOwner, draft\)/);
  assert.match(ui, /setLoginRequired\(true\)/);
  assert.match(ui, /event === "run_progress"/);
  assert.match(progress, /age > 45/);
  assert.match(progress, /source === "managed_adapter"/);
  assert.match(ui, /h-full max-h-full min-h-0/);
});
