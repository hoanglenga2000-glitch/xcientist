import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import { createLoopbackSession, LoopbackSessionError } from "./lib/loopback-session.mjs";

const baseUrl = process.env.REPORT_TEST_BASE_URL ?? "http://127.0.0.1:8088";
const workspaceRoot = path.resolve(process.env.WORKSTATION_ROOT ?? path.resolve(process.cwd(), "..", ".."));
const taskId = "evomind-qwen7b-finetune";
const scientificRoot = path.join(workspaceRoot, "workspace", "tasks", taskId, "reports", "scientific");
const reportCandidates = await Promise.all((await readdir(scientificRoot, { withFileTypes: true }))
  .filter((entry) => entry.isDirectory())
  .map(async (entry) => JSON.parse(await readFile(path.join(scientificRoot, entry.name, "artifact-manifest.json"), "utf-8").catch(() => "{}"))));
const selectedReport = reportCandidates
  .filter((item) => item?.schema === "evomind.scientific_report_package.v1" && item?.task_id === taskId && item?.run_id)
  .sort((left, right) => String(right.generated_at ?? "").localeCompare(String(left.generated_at ?? "")))[0];
const runId = String(selectedReport?.run_id ?? "");
const publicRunId = "EVOMIND-DEMO-7B-20260722";
const forbiddenPublicText = /A800|qwen7b_qlora_\d|\/hpc2hdd\/|workspace\/evomind_runs|jinghw|景浩伟/i;
const reportStudioSource = await readFile(path.join(process.cwd(), "src", "components", "workstation", "screens", "ReportStudioScreen.tsx"), "utf-8");
const taskContextSource = await readFile(path.join(process.cwd(), "src", "lib", "task-context.ts"), "utf-8");
const reportServerSource = await readFile(path.join(process.cwd(), "src", "lib", "server", "scientific-report.ts"), "utf-8");
const reportRouteSource = await readFile(path.join(process.cwd(), "src", "app", "api", "tasks", "[taskId]", "scientific-report", "route.ts"), "utf-8");
const artifactRouteSource = await readFile(path.join(process.cwd(), "src", "app", "api", "artifacts", "route.ts"), "utf-8");
const resumeRouteSource = await readFile(path.join(process.cwd(), "src", "app", "api", "multi-agent", "runs", "[runId]", "[action]", "route.ts"), "utf-8");
const publicSummaryRouteSource = await readFile(path.join(process.cwd(), "src", "app", "api", "public-demo-summary", "route.ts"), "utf-8");
const reportPollingSource = await readFile(path.join(process.cwd(), "src", "lib", "api", "report-polling.ts"), "utf-8");
const bootstrapRouteSource = await readFile(path.join(process.cwd(), "src", "app", "api", "session", "bootstrap", "route.ts"), "utf-8");
const loopbackSessionSource = await readFile(path.join(process.cwd(), "scripts", "lib", "loopback-session.mjs"), "utf-8");

assert.match(taskId, /^[A-Za-z0-9_.-]+$/, "current task ID is required");
assert.match(runId, /^[A-Za-z0-9_.-]+$/, "current run ID is required");
assert.match(taskContextSource, /\["evomind_qwen7b_finetune",\s*"evomind-qwen7b-finetune"\]/, "the legacy UI alias must resolve to the canonical Supervisor task ID");
assert.match(taskContextSource, /function taskComparisonKey[\s\S]*replace\(\/-\/g,\s*"_"\)/, "task matching must remain tolerant without rewriting API task IDs");
assert.match(taskContextSource, /runtime_by_task[\s\S]*current_run\?\.task_id[\s\S]*taskComparisonKey/, "task-bound views must prefer the selected task runtime and its current-run pointer");
assert.match(taskContextSource, /currentRuntime[\s\S]*taskComparisonKey\(currentTaskId\)\s*===\s*selectedKey/, "top-level runtime fallback must be rejected when it belongs to another task");
assert.match(reportStudioSource, /runtimeForTask\(props\.summary,\s*props\.selectedTask\)/, "Report Studio must resolve its run from the selected task runtime");
assert.match(reportStudioSource, /normalizeTaskId\(currentRun\.task_id\)\s*===\s*normalizeTaskId\(props\.selectedTask\)/, "Report Studio must accept legacy task aliases");
assert.match(reportStudioSource, /const taskId = currentRunMatchesSelection \? currentRun!\.task_id! : props\.selectedTask/, "run-bound APIs must use the Supervisor canonical task ID");
assert.match(reportStudioSource, /getScientificReport\(taskId, parentRunId, false, \{ signal \}\)/, "an incomplete child run must preserve the reviewed parent report with cancellation support");
assert.match(reportStudioSource, /controlMultiAgentRun\(runId, "resume"\)/, "Report Studio must expose a durable resume action for the current child run");
assert.match(reportStudioSource, /activeRefinementPending/, "Report Studio must prevent overlapping refinement runs");
assert.match(reportStudioSource, /grid-cols-5[\s\S]*sm:flex/, "Report Studio views must use a stable five-column mobile tab layout");
assert.doesNotMatch(reportStudioSource, /Scientific report views[\s\S]{0,240}overflow-x-auto/, "Report Studio views must not require horizontal scrolling on mobile");
assert.match(reportStudioSource, /responseHash\s*!==\s*artifact\.sha256\.toLowerCase\(\)/, "download success must compare the response SHA256 with the report manifest");
assert.match(reportStudioSource, /figure\.preview_data_url \?\? artifactHref\(figure\.path\)/, "reviewed SVG figures must use the sanitized data preview with a bound artifact fallback");
assert.match(reportStudioSource, /Human Gate approved · recorded/, "completed refinement approvals must remain visible as durable Human Gate evidence");
assert.match(reportServerSource, /scientificReportJobs/, "Nature report generation must be mutually exclusive per task and run");
assert.match(reportServerSource, /version_comparison_recomputed/, "V2 report generation must bind comparison evidence to the Reviewer");
assert.match(reportServerSource, /version_comparison:\s*reportVersionComparison\(evidence\)/, "the report package must expose the reviewed version comparison without frontend recomputation");
assert.match(reportRouteSource, /status:\s*"pending_report"[\s\S]*status:\s*202/, "known runs without a reviewed report must return 202 pending_report");
assert.match(reportRouteSource, /code:\s*"report_integrity_failed"[\s\S]*status:\s*409/, "report binding, reviewer and hash drift must return a terminal 409");
assert.match(reportServerSource, /verifyReportManifest[\s\S]*lookupScientificReport/, "report lookup must explicitly verify a reviewed manifest");
assert.match(reportServerSource, /artifact hash verification failed/, "ready report artifacts must pass SHA256 verification before a 200 response");
assert.doesNotMatch(reportServerSource, /findRun\(taskId, requestedRunId\)\.catch\(\(\) => null\)/, "report lookup must not swallow evidence failures");
assert.match(reportPollingSource, /maxAttempts[\s\S]*maxElapsedMs[\s\S]*maxDelayMs/, "internal report polling must have attempt, time and delay caps");
assert.match(reportPollingSource, /response\.status !== 202/, "only a structured 202 report state may be retried");
assert.match(reportStudioSource, /new AbortController\(\)[\s\S]*controller\.abort\(\)/, "Report Studio must abort polling during React cleanup");
assert.match(reportStudioSource, /error instanceof ScientificReportPendingTimeoutError[\s\S]*getScientificReport\(taskId, parentRunId, false, \{ signal \}\)/, "only a genuine pending child may fall back to its immutable parent report");
assert.match(bootstrapRouteSource, /request\.headers\.get\("origin"\)[\s\S]*request\.headers\.get\("host"\)/, "bootstrap must verify exact Origin and Host");
assert.match(bootstrapRouteSource, /MAX_BOOTSTRAP_BODY_BYTES[\s\S]*unsupported_content_type/, "bootstrap must enforce a small JSON-only body before token validation");
assert.match(loopbackSessionSource, /httponly[\s\S]*samesite[\s\S]*path/, "the loopback helper must verify strict cookie attributes");
assert.match(loopbackSessionSource, /mutation && !csrfToken/, "the loopback helper must reject mutations before fetch when CSRF state is absent");
assert.match(loopbackSessionSource, /body === undefined[\s\S]*Content-Type", "application\/json"/, "the loopback helper must send bounded JSON mutation envelopes");
assert.match(reportStudioSource, /Reviewed version comparison/, "the report studio must render the reviewed V1/V2 comparison");
assert.match(reportStudioSource, /versionComparison\.delta_pp/, "the report studio must use the backend-reviewed comparison delta");
assert.doesNotMatch(reportStudioSource, /versionComparison\.v2\s*-\s*versionComparison\.v1/, "the frontend must not recompute the reviewed version delta");
assert.match(reportServerSource, /@media\(max-width:820px\)[\s\S]*?\.report-nav\{[^}]*flex-wrap:wrap[^}]*overflow:visible/, "Nature report navigation must wrap without a mobile horizontal scrollbar");
assert.match(reportServerSource, /@media\(max-width:820px\)[\s\S]*?th,td\{overflow-wrap:anywhere\}/, "Nature report tables must remain readable on narrow screens");
assert.match(artifactRouteSource, /X-Artifact-SHA256/, "artifact downloads must return the server-recomputed SHA256");
assert.match(resumeRouteSource, /already_resuming/, "duplicate resume requests must return an explicit conflict");
assert.ok(resumeRouteSource.indexOf("already_resuming") < resumeRouteSource.lastIndexOf("runManagedCommand("), "duplicate resume must be rejected before a managed process can launch");
assert.match(reportStudioSource, /setRefinement\(\(current\)[\s\S]*status:\s*"running"/, "accepted resume must re-enable durable refinement polling");
assert.match(publicSummaryRouteSource, /loadMultiAgentRuntimeByRunId\(PUBLIC_TASK_ID,\s*sourceRunId\)/, "public summary must load the reviewed source run directly");
assert.doesNotMatch(publicSummaryRouteSource, /getWorkstationSummary/, "public summary must not depend on workspace current_run");

async function shellSecuritySnapshot() {
  const response = await fetch(new URL("/?page=assistant", baseUrl), { cache: "no-store" });
  assert.equal(response.status, 200, "the production shell must load");
  const contentSecurityPolicy = response.headers.get("content-security-policy") ?? "";
  const scriptDirective = contentSecurityPolicy.split(";").map((item) => item.trim()).find((item) => item.startsWith("script-src ")) ?? "";
  const nonce = scriptDirective.match(/'nonce-([^']+)'/)?.[1] ?? "";
  assert.ok(nonce, "the production shell CSP must carry a request-scoped script nonce");
  assert.doesNotMatch(scriptDirective, /'unsafe-inline'/, "the script policy must not permit arbitrary inline JavaScript");
  const html = await response.text();
  const scriptNonces = [...html.matchAll(/<script\b[^>]*\bnonce="([^"]+)"/g)].map((match) => match[1]);
  assert.ok(scriptNonces.length > 0, "Next.js bootstrap scripts must carry the CSP nonce");
  assert.ok(scriptNonces.every((value) => value === nonce), "every bootstrap script nonce must match the response CSP");
  const staticScript = html.match(/\bsrc="(\/_next\/static\/[^"]+\.js)"/)?.[1] ?? "";
  assert.ok(staticScript, "the production shell must reference a content-addressed JavaScript asset");
  return { nonce, staticScript };
}

const firstShell = await shellSecuritySnapshot();
const secondShell = await shellSecuritySnapshot();
assert.notEqual(firstShell.nonce, secondShell.nonce, "the CSP nonce must rotate for every document response");
const staticAsset = await fetch(new URL(firstShell.staticScript, baseUrl), { cache: "no-store" });
assert.equal(staticAsset.status, 200, "the production shell JavaScript asset must load");
assert.match(staticAsset.headers.get("cache-control") ?? "", /immutable/, "content-addressed JavaScript must use immutable caching");

let bootstrapToken = process.env.REPORT_TEST_BOOTSTRAP_TOKEN ?? "";
delete process.env.REPORT_TEST_BOOTSTRAP_TOKEN;
assert.match(bootstrapToken, /^\S{24,256}$/, "an isolated one-time bootstrap token is required");
const internalReportPath = `/api/tasks/${encodeURIComponent(taskId)}/scientific-report?run_id=${encodeURIComponent(runId)}`;
const unauthenticated = await fetch(new URL(internalReportPath, baseUrl), { cache: "no-store", redirect: "error" });
assert.equal(unauthenticated.status, 401, "internal report APIs must reject a pre-bootstrap request");
assert.equal((await unauthenticated.json()).code, "session_required", "pre-bootstrap rejection must be explicit");

const rejectedOrigin = await fetch(new URL("/api/session/bootstrap", baseUrl), {
  method: "POST",
  headers: { "Content-Type": "application/json", Origin: "http://localhost.invalid" },
  body: JSON.stringify({ token: bootstrapToken }),
  cache: "no-store",
  redirect: "error",
});
assert.equal(rejectedOrigin.status, 403, "an incorrect bootstrap Origin must fail before consuming the token");
const session = await createLoopbackSession({ baseUrl, bootstrapToken });
bootstrapToken = "";
const request = session.request;
const authenticatedStatus = await request("/api/session/status");
assert.equal(authenticatedStatus.status, 200, "the closure-held session must authenticate subsequent requests");
assert.equal((await authenticatedStatus.json()).authenticated, true, "session status must confirm authentication");

const internal = await request(internalReportPath);
assert.equal(internal.status, 200, "current reviewed report binding must load");
const internalPayload = await internal.json();
assert.equal(internalPayload.report.task_id, taskId, "report task binding must be exact");
assert.equal(internalPayload.report.run_id, runId, "report run binding must be exact");
if (internalPayload.report.parent_run_id) {
  assert.equal(internalPayload.report.version_comparison?.schema, "evomind.llm_version_comparison.v1", "V2 reports must expose a structured reviewed comparison");
  assert.equal(internalPayload.report.version_comparison?.generated_by, "VersionComparatorAgent", "version comparison provenance must be explicit");
  assert.equal(typeof internalPayload.report.version_comparison?.delta_pp, "number", "V2 comparison delta must come from reviewed evidence");
} else {
  assert.equal(internalPayload.report.version_comparison, null, "V1 reports must not fabricate a version comparison");
}
assert.ok(internalPayload.report.figures.filter((figure) => figure.status === "ready").every((figure) => String(figure.preview_data_url).startsWith("data:image/svg+xml;base64,")), "ready SVG figures must expose sanitized in-memory previews");
const generation = await request(`/api/tasks/${encodeURIComponent(taskId)}/scientific-report?run_id=${encodeURIComponent(runId)}&status_only=1`);
assert.equal(generation.status, 200, "report generation status ledger must be queryable");

const internalPdf = internalPayload.report.artifacts.find((artifact) => artifact.id === "report-pdf" && artifact.status === "ready");
assert.ok(internalPdf?.path && internalPdf.sha256 && Number.isFinite(internalPdf.bytes), "internal report PDF must have manifest bytes and SHA256");
const internalPdfQuery = new URLSearchParams({ path: internalPdf.path, task_id: taskId, run_id: runId, download: "1" });
const internalPdfDownload = await request(`/api/artifacts?${internalPdfQuery.toString()}`);
assert.equal(internalPdfDownload.status, 200, "manifest-bound report PDF download must succeed");
assert.equal(internalPdfDownload.headers.get("x-artifact-sha256"), internalPdf.sha256, "download SHA256 must match the report manifest");
assert.equal(Number(internalPdfDownload.headers.get("x-artifact-bytes")), internalPdf.bytes, "download byte count must match the report manifest");
assert.equal((await internalPdfDownload.arrayBuffer()).byteLength, internalPdf.bytes, "downloaded response bytes must match the report manifest");
assert.equal((await request(`/api/artifacts?path=${encodeURIComponent(internalPdf.path)}`)).status, 400, "scientific report downloads must reject missing task/run binding");

const internalFigure = internalPayload.report.artifacts.find((artifact) => artifact.type === "SVG" && artifact.status === "ready");
assert.ok(internalFigure?.path, "an internal reviewed SVG figure is required");
const internalFigureQuery = new URLSearchParams({ path: internalFigure.path, task_id: taskId, run_id: runId });
const internalFigureResponse = await request(`/api/artifacts?${internalFigureQuery.toString()}`);
assert.equal(internalFigureResponse.status, 200, "manifest-bound SVG preview must succeed");
assert.match(internalFigureResponse.headers.get("content-type") ?? "", /^image\/svg\+xml/i);
assert.match(internalFigureResponse.headers.get("content-disposition") ?? "", /^inline;/, "sanitized SVG previews must render inline inside the script-disabled frame");
assert.match(artifactRouteSource, /SVG artifact contains active content/, "SVG previews must reject active content before rendering");

assert.equal((await request(`/api/tasks/${encodeURIComponent(taskId)}/scientific-report`)).status, 400, "missing run ID must fail");
assert.equal((await request(`/api/tasks/${encodeURIComponent(taskId)}/scientific-report?run_id=unknown-report-run`)).status, 404, "unknown run must not fall back");

const publicResponse = await request(`/api/public-scientific-report?task_id=${encodeURIComponent(taskId)}&run_id=${publicRunId}`);
assert.equal(publicResponse.status, 200, "public report must load through its alias");
const publicText = await publicResponse.text();
assert.equal(forbiddenPublicText.test(publicText), false, "public report JSON must not expose internal run, GPU or paths");
const publicPayload = JSON.parse(publicText);
assert.equal(publicPayload.report.run_id, publicRunId);
assert.equal(publicPayload.report.bundle_path, null, "raw final bundle is not public");
assert.ok(publicPayload.report.artifacts.length > 0, "public report must expose reviewed visual artifacts");
assert.ok(publicPayload.report.artifacts.every((artifact) => String(artifact.path).startsWith("public:")), "public artifacts must use opaque IDs");
assert.ok(publicPayload.report.artifacts.every((artifact) => ["HTML", "PDF", "SVG"].includes(artifact.type)), "public artifacts must stay inside the visual allowlist");

const artifactQuery = `task_id=${encodeURIComponent(taskId)}&run_id=${publicRunId}`;
const html = await request(`/api/public-scientific-report/artifact?${artifactQuery}&artifact_id=report-html`);
assert.equal(html.status, 200);
assert.match(html.headers.get("content-disposition") ?? "", /scientific-report\.html/i);
assert.match(html.headers.get("content-security-policy") ?? "", /default-src 'none'/);
assert.match(html.headers.get("content-security-policy") ?? "", /script-src 'none'/);
const htmlBody = await html.arrayBuffer();
const publicHtmlArtifact = publicPayload.report.artifacts.find((artifact) => artifact.id === "report-html");
assert.equal(forbiddenPublicText.test(new TextDecoder().decode(htmlBody)), false, "public report HTML must be sanitized");
assert.equal(html.headers.get("x-artifact-sha256"), publicHtmlArtifact.sha256, "public HTML download hash must match the sanitized manifest");
assert.equal(Number(html.headers.get("x-artifact-bytes")), publicHtmlArtifact.bytes, "public HTML byte count must match the sanitized manifest");
assert.equal(htmlBody.byteLength, publicHtmlArtifact.bytes, "public HTML response bytes must match the sanitized manifest");

const pdf = await request(`/api/public-scientific-report/artifact?${artifactQuery}&artifact_id=report-pdf&download=1`);
assert.equal(pdf.status, 200);
assert.match(pdf.headers.get("content-disposition") ?? "", /attachment; filename="scientific-report\.pdf"/i);
const publicPdfArtifact = publicPayload.report.artifacts.find((artifact) => artifact.id === "report-pdf");
const pdfBody = await pdf.arrayBuffer();
assert.equal(pdf.headers.get("x-artifact-sha256"), publicPdfArtifact.sha256, "public PDF download hash must match the public manifest");
assert.equal(pdfBody.byteLength, publicPdfArtifact.bytes, "public PDF response bytes must match the public manifest");

const figureId = String(publicPayload.report.figures.find((figure) => figure.status === "ready")?.id ?? "");
assert.ok(figureId, "at least one reviewed figure is required");
const figure = await request(`/api/public-scientific-report/artifact?${artifactQuery}&artifact_id=${encodeURIComponent(figureId)}&download=1`);
assert.equal(figure.status, 200);
assert.match(figure.headers.get("content-disposition") ?? "", /\.svg"$/i);
assert.match(figure.headers.get("content-security-policy") ?? "", /script-src 'none'/);
const figureBody = Buffer.from(await figure.arrayBuffer());
assert.equal(figure.headers.get("x-artifact-sha256"), createHash("sha256").update(figureBody).digest("hex"), "public SVG hash must describe transformed response bytes");
assert.equal(Number(figure.headers.get("x-artifact-bytes")), figureBody.byteLength, "public SVG byte count must describe transformed response bytes");
assert.equal(forbiddenPublicText.test(figureBody.toString("utf-8")), false, "public figure must be sanitized");

assert.equal((await request(`/api/public-scientific-report/artifact?${artifactQuery}&artifact_id=final-bundle`)).status, 404, "raw bundle must remain blocked");
assert.equal((await request(`/api/public-scientific-report?task_id=${encodeURIComponent(taskId)}&run_id=unknown-public-run`)).status, 404, "unknown public alias must fail");

const publicSummary = await request("/api/public-demo-summary");
assert.equal(publicSummary.status, 200, "the pinned public summary must remain available while current_run points at a refinement child");
const publicSummaryText = await publicSummary.text();
assert.equal(forbiddenPublicText.test(publicSummaryText), false, "public summary must not expose internal run, GPU or paths");
const publicSummaryPayload = JSON.parse(publicSummaryText);
assert.equal(publicSummaryPayload.runtime?.current_run?.run_id, publicRunId, "public summary must expose only the stable public run alias");
assert.equal(publicSummaryPayload.runtime?.review?.status, "passed", "public summary must remain bound to the reviewed V1 source");

const refinement = await request(`/api/tasks/${encodeURIComponent(taskId)}/refinement?parent_run_id=${encodeURIComponent(runId)}`);
assert.equal(refinement.status, 200, "refinement lookup must bind to the current run");
const refinementPayload = await refinement.json();
assert.equal(refinementPayload.ok, true, "refinement lookup must return a structured response");
assert.equal((await request(`/api/tasks/${encodeURIComponent(taskId)}/refinement`)).status, 400, "unbound refinement lookup must fail");

session.close();
await assert.rejects(
  () => request(`/api/tasks/${encodeURIComponent(taskId)}/scientific-report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_id: runId }),
  }),
  (error) => error instanceof LoopbackSessionError && error.code === "csrf_unavailable",
  "a closed helper must reject mutation locally before any request can be sent",
);

console.log(JSON.stringify({
  ok: true,
  base_url: baseUrl,
  task_id: taskId,
  report_run_id: runId,
  public_run_id: publicRunId,
  public_artifacts: publicPayload.report.artifacts.length,
  checks: 76,
}, null, 2));
