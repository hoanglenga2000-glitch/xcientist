import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { assistantArtifactLabel, assistantMessagePresentation, LEGACY_EMPTY_SUMMARY } from "./assistant-presentation.ts";
import { navSectionCollapsed } from "../navigation.ts";
import { workspaceLocation } from '../task-workspace/workspace-location.ts';

test("legacy executor fallback becomes an explanation, never a training success", () => {
  const view = assistantMessagePresentation(LEGACY_EMPTY_SUMMARY, "assistant", "zh-CN");
  assert.match(view.text, /不代表已经交付模型/);
  assert.equal(view.rawDiagnostic, LEGACY_EMPTY_SUMMARY);
  assert.deepEqual(assistantMessagePresentation(LEGACY_EMPTY_SUMMARY, "user", "zh-CN"), { text: LEGACY_EMPTY_SUMMARY, rawDiagnostic: null });
  assert.equal(assistantMessagePresentation("Actual model response", "assistant", "zh-CN").text, "Actual model response");
});

test("file naming cannot establish a verified model or score", () => {
  assert.match(assistantArtifactLabel("training.log", "zh-CN"), /非模型产物/);
  assert.match(assistantArtifactLabel("status-receipt.json", "zh-CN"), /非模型产物/);
  assert.match(assistantArtifactLabel("model.safetensors", "zh-CN"), /待独立验证/);
  assert.match(assistantArtifactLabel("predictions.csv", "zh-CN"), /待评分核验/);
});

test("core navigation remains visible and advanced tools respect manual collapse", () => {
  assert.equal(navSectionCollapsed("core", "core", false, true), false);
  assert.equal(navSectionCollapsed("advanced", "core", false), true);
  assert.equal(navSectionCollapsed("advanced", "advanced", false), false);
  assert.equal(navSectionCollapsed("advanced", "advanced", false, true), true);
  assert.equal(navSectionCollapsed("advanced", "core", false, false), false);
});

test("global summary cannot choose an assistant task and new experiments reset context", async () => {
  const page = await readFile(new URL("../../../app/page.tsx", import.meta.url), "utf8");
  const screen = await readFile(new URL("./AssistantScreen.tsx", import.meta.url), "utf8");
  assert.doesNotMatch(page, /latestTaskSignal|latestAppliedTaskSignalRef/);
  assert.equal(workspaceLocation(new URLSearchParams('page=assistant')), '/workspace?new=1');
  assert.equal(workspaceLocation(new URLSearchParams('page=overview')), '/workspace');
  assert.match(screen, /onNewExperiment\?\.\(\)/);
  assert.match(screen, /setContextState\(null\)/);
});
