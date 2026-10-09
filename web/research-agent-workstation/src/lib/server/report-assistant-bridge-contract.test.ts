import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
// @ts-expect-error Native Node strip-types tests resolve explicit .ts extensions.
import { workspaceLocation } from '../../components/workstation/task-workspace/workspace-location.ts';

const root = process.cwd();

async function source(...segments: string[]) {
  return readFile(path.join(root, "src", ...segments), "utf-8");
}

test("report detail loading is Run-bound and cancelled on selection changes", async () => {
  const panel = await source("components", "workstation", "screens", "AssistantReportPanel.tsx");
  assert.match(panel, /fetch\(`\/api\/assistant\/runs\/\$\{encodeURIComponent\(selectedRunId\)\}`/);
  assert.match(panel, /checkedAssistantReportDetail\(await response\.json\(\), selectedRunId\)/);
  assert.match(panel, /if \(!controller\.signal\.aborted\) \{ setRunDetail\(detail\)/);
  assert.match(panel, /runDetail\?\.id === selectedRunId \? runDetail/);
  assert.match(panel, /重试详情/);
  assert.match(panel, /reportExportDownloadUrl\(selectedRun.id/);
  assert.match(panel, /summary: selectedRun\.answer_is_current !== false \? selectedRun\.answer : undefined/);
  assert.doesNotMatch(panel, /href=\{artifact.download_url\} download=\{artifact.name\} data-ui-skip-action="true" className="rounded border/);
});

test("Report Studio keeps partial task records scoped and uses opaque artifact URLs", async () => {
  const studio = await source("components", "workstation", "screens", "ReportStudioScreen.tsx");
  const assistantReport = await source("components", "workstation", "screens", "AssistantReportPanel.tsx");
  assert.match(studio, /AssistantReportPanel/);
  assert.match(assistantReport, /\/api\/assistant\/runs\?limit=100/);
  assert.match(assistantReport, /reportableAssistantRuns/);
  assert.match(assistantReport, /artifact\.run_id === run\.id/);
  assert.doesNotMatch(assistantReport, /已验证 Agent 报告/);
  assert.match(assistantReport, /report_run/);
  assert.match(assistantReport, /resolveAssistantReportRunId\(available, requested\)/);
  assert.match(assistantReport, /runs\.map\(\(run\) => <option/);
  assert.match(assistantReport, /artifact\.preview_url/);
  assert.match(assistantReport, /artifact\.download_url/);
  assert.doesNotMatch(assistantReport, /\/api\/artifacts\?[^\n]*artifact\.path/);
});

test("report export uses the owner-scoped asynchronous route, never the generic tool lifecycle", async () => {
  const route = await source("app", "api", "assistant", "runs", "[runId]", "reports", "route.ts");
  assert.match(route, /assistantRuntimeFetch\(`\/v1\/runs\/\$\{encodeURIComponent\(runId\)\}\/reports`,/);
  assert.doesNotMatch(route, /\/v1\/sessions\//);
  assert.doesNotMatch(route, /tool_name:/);
  assert.match(route, /status: response\.status/);
  assert.match(route, /summary: body\.summary/);
  assert.doesNotMatch(route, /body\.(?:path|output_dir|destination)/);
});

test("legacy result links retain the exact run for the authenticated workspace lookup", async () => {
  const assistant = await source("components", "workstation", "screens", "AssistantScreen.tsx");
  const target = workspaceLocation(new URLSearchParams('page=report&report_run=run_exact_original'));
  assert.match(target, /legacy_run=run_exact_original/);
  assert.match(target, /section=results/);
  assert.doesNotMatch(target, /[?&]task=/);
  assert.match(assistant, /onOpenAdvanced\?\.\(currentRun\?\.id\)/);
});

test("Report Studio archive preview uses the bounded safe file-list endpoint", async () => {
  const assistantReport = await source("components", "workstation", "screens", "AssistantReportPanel.tsx");
  assert.match(assistantReport, /previewKind === "archive"/);
  assert.match(assistantReport, /\$\{url\}\?preview=1/);
  assert.match(assistantReport, /payload\.preview/);
  assert.match(assistantReport, /Safe ZIP file list/);
});

test("Report Studio verifies PDF bytes before rendering a blob preview", async () => {
  const assistantReport = await source("components", "workstation", "screens", "AssistantReportPanel.tsx");
  const assistant = await source("components", "workstation", "screens", "AssistantScreen.tsx");
  const proxy = await source("proxy.ts");
  assert.match(assistantReport, /previewKind === "pdf"/);
  assert.match(assistantReport, /crypto\.subtle\.digest\("SHA-256", body\)/);
  assert.match(assistantReport, /const nextObjectUrl = URL\.createObjectURL\(new Blob\(\[body\], \{ type: "application\/pdf" \}\)\)/);
  assert.match(assistantReport, /controller\.signal\.aborted \|\| disposed/);
  assert.match(assistantReport, /URL\.revokeObjectURL\(nextObjectUrl\)/);
  assert.match(assistantReport, /URL\.revokeObjectURL\(objectUrl\)/);
  assert.match(assistantReport, /data-ui-verified-pdf-preview/);
  assert.doesNotMatch(assistantReport, /previewKind === "pdf" \|\| previewKind === "html"/);
  assert.match(assistant, /import \{ VerifiedReportDocument \} from "\.\/AssistantReportPanel"/);
  assert.match(assistant, /preview\.artifact\.preview_kind === "pdf" \? <VerifiedReportDocument/);
  assert.doesNotMatch(assistant, /preview\.artifact\.preview_kind === "pdf" \? <iframe/);
  assert.match(proxy, /"frame-src 'self' blob:"/);
});

test("Report Studio interactive tabs and artifact actions bypass global click auditing", async () => {
  const assistantReport = await source("components", "workstation", "screens", "AssistantReportPanel.tsx");
  assert.match(assistantReport, /role="tab"[^>]*data-ui-skip-action="true"[^>]*onClick=/);
  assert.match(assistantReport, /data-ui-skip-action="true"[^>]*>Preview<\/button>/);
  assert.match(assistantReport, /data-ui-skip-action="true"[^>]*download=\{artifact\.name\}/);
  assert.match(assistantReport, /data-ui-skip-action="true"[^\n]*Copy hash<\/button>/);
});

test("resource footer derives the active model connector instead of pinning DeepSeek", async () => {
  const sidebar = await source("components", "workstation", "Sidebar.tsx");
  assert.match(sidebar, /preferredModelConnector/);
  assert.match(sidebar, /@\/lib\/connector-presentation/);
  const presentation = await source("lib", "connector-presentation.ts");
  assert.match(presentation, /entry\.model/);
  assert.doesNotMatch(sidebar, /\{ key: "deepseek", label: "DeepSeek API" \}/);
});

test("legacy evomind report artifacts are admitted only with exact report-manifest binding", async () => {
  const route = await source("app", "api", "artifacts", "route.ts");
  assert.match(route, /allowedReportBoundArtifactPath/);
  assert.match(route, /report-bound artifacts require task and run binding/);
  assert.match(route, /artifact is not bound to the requested report manifest/);
  assert.match(route, /artifact bytes do not match the report manifest/);
});
