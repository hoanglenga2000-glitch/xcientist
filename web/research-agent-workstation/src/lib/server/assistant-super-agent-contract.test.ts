import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = process.cwd();

async function source(...segments: string[]) {
  return readFile(path.join(root, "src", ...segments), "utf-8");
}

test("assistant UI uses durable runs and keeps only the conversation pointer in localStorage", async () => {
  const ui = await source("components", "workstation", "screens", "AssistantScreen.tsx");
  assert.match(ui, /loadAssistantRestoration<AssistantRun>/);
  assert.match(ui, /restoreRunMessages\(run/);
  assert.match(ui, /fetch\("\/api\/assistant\/runs"/);
  assert.match(ui, /conversation_id: sessionId/);
  assert.match(ui, /window\.localStorage\.setItem\(STORAGE_KEY, JSON\.stringify\(\{ sessionId \}\)\)/);
  assert.doesNotMatch(ui, /localStorage\.setItem\([^\n]+messages/);
  assert.doesNotMatch(ui, /fetch\("\/api\/assistant\/stream"/);
  assert.match(ui, /assistantRunTaskLabel\(run, selectedTask\)/);
  assert.match(ui, /`\u5f53\u524d\u4efb\u52a1 \u00b7 \$\{contextState\.taskLabel/);
});

test("assistant UI wires chunked attachments, reconnectable SSE, approvals, served model, and artifact cards", async () => {
  const ui = await source("components", "workstation", "screens", "AssistantScreen.tsx");
  assert.match(ui, /UPLOAD_CHUNK_BYTES = 16 \* 1024 \* 1024/);
  assert.match(ui, /\/api\/assistant\/uploads\/\$\{encodeURIComponent\(upload\.id\)\}\/chunks\/\$\{index\}/);
  assert.match(ui, /request\.setRequestHeader\("X-Chunk-SHA256", digest\)/);
  assert.match(ui, /Last-Event-ID/);
  assert.match(ui, /plan_updated/);
  assert.match(ui, /approval_required/);
  assert.match(ui, /artifact_published/);
  assert.match(ui, /model_observed/);
  assert.match(ui, /data-ui-assistant-approval-card/);
  assert.match(ui, /data-ui-assistant-artifact-card/);
  assert.match(ui, /data-ui-assistant-model/);
  assert.match(ui, /artifact\.sha256/);
  assert.doesNotMatch(ui, /gpt-5\.6-sol|deepseek/i);
});

test("assistant UI reconciles stale blocked events against newer managed tool state", async () => {
  const ui = await source("components", "workstation", "screens", "AssistantScreen.tsx");
  const start = ui.indexOf('} else if (event === "run_blocked" || event === "run_failed") {');
  const end = ui.indexOf("appendActivity(event, data);", start);
  assert.ok(start >= 0 && end > start);
  const branch = ui.slice(start, end);
  assert.doesNotMatch(branch, /stopStream = true/);
  assert.match(branch, /reconcile against the durable snapshot/);
  assert.match(ui, /const snapshotResponse = await fetch\(`\/api\/assistant\/runs\/\$\{encodeURIComponent\(runId\)\}`/);
});

test("assistant history hydrates the latest effective snapshot before exposing recovery actions", async () => {
  const ui = await source("components", "workstation", "screens", "AssistantScreen.tsx");
  assert.match(ui, /const snapshotResponse = await fetch\(`\/api\/assistant\/runs\/\$\{encodeURIComponent\(latest\.id\)\}`/);
  assert.match(ui, /latestSnapshot = await snapshotResponse\.json\(\) as AssistantRun/);
  assert.match(ui, /latestAssistantMessageId\(restored, latest\.id\)/);
  assert.match(ui, /applySnapshot\(latestSnapshot, assistantId\)/);
  assert.match(ui, /isActiveStatus\(latestSnapshot\.status\)/);
});

test("assistant approval decisions resume from the durable event cursor and never replay an expired gate", async () => {
  const ui = await source("components", "workstation", "screens", "AssistantScreen.tsx");
  assert.match(ui, /last_event_seq\?: number/);
  assert.match(ui, /payload\.approval\?\.status === "approved"/);
  assert.match(ui, /connectRun\(payload\.run\.id, assistantId, controller, payload\.run\.last_event_seq \?\? 0\)/);
  assert.match(ui, /approval\.status !== "pending"/);
  assert.match(ui, /approval_expired/);
  assert.doesNotMatch(ui, /if \(approved\) \{\s*setRunning\(true\)/);
});

test("assistant attachments start immediately and expose byte-level accessible progress through verification", async () => {
  const ui = await source("components", "workstation", "screens", "AssistantScreen.tsx");
  assert.match(ui, /if \(item\.status === "queued"\) void uploadOne\(item\)/);
  assert.match(ui, /new XMLHttpRequest\(\)/);
  assert.match(ui, /request\.upload\.onprogress/);
  assert.match(ui, /start \+ loadedBytes/);
  assert.match(ui, /phase: "verifying", progress: 96/);
  assert.match(ui, /phase: "complete", progress: 100/);
  assert.match(ui, /data-ui-assistant-upload-progress/);
  assert.match(ui, /role=\{item\.status === "error" \? undefined : "progressbar"\}/);
  assert.match(ui, /aria-valuenow=\{item\.status === "error" \? undefined : item\.progress\}/);
  assert.match(ui, /uploadPromisesRef\.current\.get\(draftItem\.id\)/);
  assert.match(ui, /uploadControllersRef\.current\.get\(id\)\?\.abort\(\)/);
});

test("assistant reuses prior attachments only by explicit opt-in and never creates a replacement Run for an active retry", async () => {
  const ui = await source("components", "workstation", "screens", "AssistantScreen.tsx");
  assert.match(ui, /const retainedAttachments = uploadDrafts\.length \|\| !reuseRetainedAttachments \? \[\] : \(currentRun\?\.attachments \?\? \[\]\)/);
  assert.match(ui, /attachmentIds\.push\(\.\.\.retainedAttachments\.map\(\(item\) => item\.id\)\)/);
  assert.match(ui, /setReuseRetainedAttachments\(false\)/);
  assert.match(ui, /data-ui-assistant-retained-attachments/);
  assert.match(ui, /data-attachment-reuse=\{reuseRetainedAttachments \? "enabled" : "disabled"\}/);
  assert.match(ui, /上一轮附件（默认不再发送）/);
  assert.match(ui, /下条消息继续使用/);
  assert.match(ui, /清除上一轮附件上下文/);
  assert.match(ui, /currentRun\.status !== "completed"/);
  assert.match(ui, /runAction\(currentRun\.status === "waiting_approval" \? "resume" : "retry"\)/);
  assert.match(ui, /setCurrentRun\(null\)/);
  assert.match(ui, /attachment_context\?\.mode === "inherited"/);
});

test("assistant auto-approves only reversible same-Run non-sensitive operations", async () => {
  const ui = await source("components", "workstation", "screens", "AssistantScreen.tsx");
  assert.match(ui, /SAFE_AUTO_APPROVAL_STORAGE_KEY/);
  assert.match(ui, /isSafeSameRunApproval/);
  assert.match(ui, /approval\.reversible/);
  assert.match(ui, /scope\.outside_workspace !== false/);
  assert.match(ui, /scope\.cross_run_workspace !== false/);
  assert.match(ui, /raw\.githubusercontent\.com/);
  assert.match(ui, /SENSITIVE_APPROVAL_TERMS/);
  assert.match(ui, /data-ui-assistant-safe-auto-approval/);
  assert.match(ui, /void decideApproval\(approval, true\)/);
});

test("blocked assistant runs accept a user correction before resuming the same Run", async () => {
  const ui = await source("components", "workstation", "screens", "AssistantScreen.tsx");
  assert.match(ui, /\{ action: "resume", instruction: prompt \}/);
  assert.match(ui, /const resumeRequested = continueRunRef\.current/);
  assert.match(ui, /continueRunRef\.current = false/);
  assert.match(ui, /resumeRequested && currentRun && \["blocked", "failed", "cancelled"\]/);
  assert.match(ui, /continueRunRef\.current = true; textareaRef\.current\?\.focus\(\)/);
  assert.match(ui, /\["blocked", "failed", "cancelled"\]\.includes\(currentRun\.status\)/);
  assert.match(ui, /connectRun\(run\.id, assistantId, controller, resumableRun \? \(run\.last_event_seq \?\? 0\) : 0\)/);
  assert.match(ui, /继续任务/);
  assert.match(ui, /currentRun\.error_class === "approval_expired"/);
});

test("assistant keeps operational internals and credential tools out of the default conversation surface", async () => {
  const ui = await source("components", "workstation", "screens", "AssistantScreen.tsx");
  const credentials = await source("components", "workstation", "screens", "AssistantControlledCredentials.tsx");
  assert.match(ui, /data-ui-assistant-workbench-grid/);
  assert.match(ui, /data-ui-assistant-conversation/);
  assert.match(ui, /data-ui-assistant-inspector/);
  assert.match(ui, /data-ui-assistant-recovery-callout/);
  assert.match(ui, /h-full max-h-full min-h-0/);
  assert.match(ui, /xl:grid-cols-\[minmax\(0,1fr\)_20rem\]/);
  assert.match(ui, /xl:sticky xl:top-6/);
  assert.match(ui, /开始新实验/);
  assert.match(ui, /data-ui-assistant-run-summary/);
  assert.match(ui, /data-ui-assistant-run-technical-details/);
  assert.match(ui, /runDetailsOpen/);
  assert.match(ui, /运行详情与错误/);
  assert.match(ui, /任务进度已保留/);
  assert.match(credentials, /连接与下载设置/);
  const openPanel = credentials.indexOf("{open ? (");
  const quickEnrollment = credentials.indexOf('data-ui-action="assistant_open_hpc_quick_enrollment"');
  assert.ok(openPanel >= 0 && quickEnrollment > openPanel, "HPC quick enrollment must stay inside the expanded panel");
});

test("assistant API surface is a managed runtime proxy and supports safe archive preview", async () => {
  const runtimeProxy = await source("lib", "server", "assistant-runtime.ts");
  const uploadRoute = await source("app", "api", "assistant", "uploads", "[uploadId]", "chunks", "[index]", "route.ts");
  const eventRoute = await source("app", "api", "assistant", "runs", "[runId]", "events", "route.ts");
  const artifactRoute = await source("app", "api", "assistant", "artifacts", "[artifactId]", "route.ts");
  assert.match(runtimeProxy, /ensureEvoMindRuntime/);
  assert.match(runtimeProxy, /Authorization/);
  assert.match(runtimeProxy, /x-artifact-sha256/);
  assert.match(uploadRoute, /proxyAssistantRuntime/);
  assert.match(eventRoute, /last-event-id/);
  assert.match(eventRoute, /X-Accel-Buffering/);
  assert.match(artifactRoute, /searchParams\.get\("preview"\) === "1"/);
  assert.match(artifactRoute, /\/preview/);
});

test("assistant runs bind the current authenticated tenant HPC allocation on the server", async () => {
  const binding = await source("lib", "server", "assistant-managed-hpc.ts");
  const runRoute = await source("app", "api", "assistant", "runs", "route.ts");
  const compatibilityRoute = await source("app", "api", "assistant", "stream", "route.ts");

  assert.match(binding, /sessionPrincipal/);
  assert.match(binding, /tenantPrincipal/);
  assert.match(binding, /requireActiveTenantBinding/);
  assert.match(binding, /credential_profile: binding\.credential_profile/);
  assert.match(runRoute, /managed_hpc_identity: await currentAssistantManagedHpcIdentity\(\)/);
  assert.match(compatibilityRoute, /managed_hpc_identity: await currentAssistantManagedHpcIdentity\(\)/);
  assert.doesNotMatch(runRoute, /body\.managed_hpc_identity/);
});

test("assistant compatibility stream binds selected task context and completes with observed execution metadata", async () => {
  const compatibilityRoute = await source("app", "api", "assistant", "stream", "route.ts");
  assert.match(compatibilityRoute, /selected_task: selectedTask/);
  assert.match(compatibilityRoute, /native_tool_calls: nativeToolCalls/);
  assert.match(compatibilityRoute, /tool_names: \[\.\.\.toolNames\]/);
  assert.match(compatibilityRoute, /provider: observedProvider/);
  assert.match(compatibilityRoute, /model: observedModel/);
  assert.match(compatibilityRoute, /input_tokens: inputTokens/);
  assert.match(compatibilityRoute, /output_tokens: outputTokens/);
});
