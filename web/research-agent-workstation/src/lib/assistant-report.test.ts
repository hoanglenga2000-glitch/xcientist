import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error Node's native TypeScript test runner requires the explicit extension.
import { assistantReportLead, reportJobStatusLabel, checkedAssistantReportDetail, reportExportDownloadUrl, assistantReportNarrative, assistantReportPresentation, completedAssistantReportRuns, reportableAssistantRuns, parseReportMarkdown, parseReportCsv, preferredAssistantArtifact, preferredAssistantReportRun, resolveAssistantReportRunId } from "./assistant-report.ts";

test("header leads with the current conclusion, never a stale answer", () => {
  const run = { id: "run_1", prompt: "GPU状态", status: "completed", answer: "**GPU 正常。**\n\n细节", answer_is_current: true };
  assert.equal(assistantReportLead(run), "GPU 正常。");
  assert.doesNotMatch(assistantReportLead({ ...run, answer_is_current: false }), /GPU 正常/);
  assert.equal(reportJobStatusLabel("ready"), "报告已生成，可下载");
  assert.equal(reportJobStatusLabel("failed"), "报告生成失败");
});

test("GPU status questions are diagnostics, not verified research reports", () => {
  for (const prompt of ["gpu状态呢", "GPU status", "显卡现在的状态"]) {
    const presentation = assistantReportPresentation({ id: "run_1", prompt, status: "completed", artifacts: [] });
    assert.equal(presentation.kind, "diagnostic");
    assert.equal(presentation.title, "连接与执行诊断");
    assert.equal(presentation.reportLabel, "尚无已发布产物");
  }
  assert.equal(assistantReportPresentation({ id: "x", prompt: "GPU状态", status: "failed" }).statusTone, "danger");
});

test("legacy report tables with blank line separators render structurally without changing values", () => {
  const blocks = parseReportMarkdown("| Tool | Result |\n\n| --- | --- |\n\n| hpc_verify | passed |\n\n| runtime_health | passed |\n\n## Next\nEvidence remains separate.");
  assert.equal(blocks[0].kind, "table");
  assert.deepEqual(blocks[0].headers, ["Tool", "Result"]);
  assert.deepEqual(blocks[0].rows, [["hpc_verify", "passed"], ["runtime_health", "passed"]]);
  assert.equal(blocks[1].text, "Next");
  assert.deepEqual(parseReportMarkdown("| Name | Value |\n| --- | --- |\n| a\\|b | -1.25 |")[0].rows, [["a|b", "-1.25"]]);
});

test("complete Run detail supplies the current assistant narrative missing from the list", () => {
  const detail = checkedAssistantReportDetail({ id: "run_1", status: "completed", prompt: "GPU状态", answer: "本次身份核验5/5通过。", answer_is_current: true }, "run_1");
  assert.equal(assistantReportNarrative(detail, null), "本次身份核验5/5通过。");
  assert.throws(() => checkedAssistantReportDetail({ ...detail, id: "foreign" }, "run_1"));
  assert.throws(() => checkedAssistantReportDetail(null, "run_1"));
});

test("report export download is derived from an opaque owned artifact ID", () => {
  const id = "artifact_" + "a".repeat(32);
  assert.equal(reportExportDownloadUrl("run_1", "run_1", { id }), `/api/assistant/artifacts/${id}?download=1`);
  assert.equal(reportExportDownloadUrl("run_1", "foreign", { id }), "");
  assert.equal(reportExportDownloadUrl("run_1", "run_1", { id, run_id: "foreign" }), "");
  assert.equal(reportExportDownloadUrl("run_1", "run_1", { id: "../../etc" }), "");
});

const artifact = (id: string, mediaType: string, previewKind: string) => ({
  id,
  run_id: "run_1",
  name: `${id}.txt`,
  media_type: mediaType,
  bytes: 12,
  sha256: "a".repeat(64),
  preview_kind: previewKind,
  preview_url: `/api/assistant/artifacts/${id}`,
  download_url: `/api/assistant/artifacts/${id}?download=1`,
});

test("completed report runs exclude approvals and sort newest first", () => {
  const runs = completedAssistantReportRuns([
    { id: "older", status: "completed", prompt: "old", artifacts: [artifact("old", "text/plain", "text")], completed_at: "2026-08-01T00:00:00Z" },
    { id: "approval", status: "waiting_approval", prompt: "gate", artifacts: [artifact("gate", "text/plain", "text")], completed_at: "" },
    { id: "repair", status: "blocked", prompt: "missing final artifacts", artifacts: [artifact("partial", "text/plain", "text")], completed_at: "" },
    { id: "newer", status: "completed", prompt: "new", artifacts: [artifact("new", "text/plain", "text")], completed_at: "2026-08-02T00:00:00Z" },
  ]);
  assert.deepEqual(runs.map((run) => run.id), ["newer", "older"]);
});

test("Report Studio defaults to the latest renderable training report instead of newer read-only checks", () => {
  const professionalReport = { ...artifact("professional", "text/markdown", "markdown"), name: "professional-training-report.md" };
  const resultOnly = { ...artifact("result", "text/markdown", "markdown"), name: "result.md" };
  const runs = [
    { id: "latest-read-only-gpu-check", status: "completed", prompt: "verify GPU identity", artifacts: [], completed_at: "2026-08-18T13:39:35Z" },
    { id: "newer-result-only", status: "completed", prompt: "check training prerequisites", artifacts: [resultOnly], completed_at: "2026-08-18T12:48:16Z" },
    { id: "completed-training", status: "completed", prompt: "train and compare models", artifacts: [professionalReport], completed_at: "2026-08-18T12:47:11Z" },
  ];

  assert.equal(preferredAssistantReportRun(runs)?.id, "completed-training");
  assert.deepEqual(completedAssistantReportRuns(runs).map((run) => run.id), [
    "latest-read-only-gpu-check",
    "newer-result-only",
    "completed-training",
  ]);
});

test("Report Studio chooses the newest report candidate and recognizes a verified training bundle", () => {
  const olderReport = { ...artifact("older-report", "application/pdf", "pdf"), name: "professional-training-report.pdf" };
  const newerReport = { ...artifact("newer-report", "text/markdown", "markdown"), name: "analysis-report.md" };
  assert.equal(preferredAssistantReportRun([
    { id: "older", status: "completed", prompt: "older", artifacts: [olderReport], completed_at: "2026-08-01T00:00:00Z" },
    { id: "newer", status: "completed", prompt: "newer", artifacts: [newerReport], completed_at: "2026-08-02T00:00:00Z" },
  ])?.id, "newer");

  const metrics = { ...artifact("metrics", "application/json", "structured_text"), name: "metrics.json" };
  const predictions = { ...artifact("predictions", "text/csv", "table"), name: "oof_predictions.csv" };
  assert.equal(preferredAssistantReportRun([
    { id: "read-only", status: "completed", prompt: "verify", artifacts: [], completed_at: "2026-08-03T00:00:00Z" },
    { id: "training-bundle", status: "completed", prompt: "fit", artifacts: [metrics, predictions], completed_at: "2026-08-02T00:00:00Z" },
  ])?.id, "training-bundle");
});

test("explicit report_run wins and invalid deep links never substitute another Run", () => {
  const report = { ...artifact("report", "text/markdown", "markdown"), name: "professional-training-report.md" };
  const runs = [
    { id: "explicit-read-only", status: "completed", prompt: "verify", artifacts: [], completed_at: "2026-08-03T00:00:00Z" },
    { id: "preferred-training", status: "completed", prompt: "train", artifacts: [report], completed_at: "2026-08-02T00:00:00Z" },
  ];

  assert.equal(resolveAssistantReportRunId(runs, "explicit-read-only"), "explicit-read-only");
  assert.equal(resolveAssistantReportRunId(runs, "missing-run"), "");
  assert.equal(resolveAssistantReportRunId(runs), "preferred-training");
});

test("failed and paused task records stay accessible without claiming verified reports", () => {
  const failed = { id: "failed", status: "failed", prompt: "检查 GPU 连接", artifacts: [], plan: { steps: [{ label: "身份检查", status: "completed" }] } };
  const paused = { ...failed, id: "paused", status: "paused" };
  assert.equal(reportableAssistantRuns([failed, paused]).length, 2);
  assert.equal(resolveAssistantReportRunId([failed, paused], "paused"), "paused");
  assert.equal(assistantReportPresentation(failed).kind, "diagnostic");
  assert.equal(assistantReportPresentation(failed).statusLabel, "执行失败");
  assert.match(assistantReportNarrative(failed, null), /身份检查/);
  assert.doesNotMatch(assistantReportNarrative(failed, null), /任务已完成/);
  assert.doesNotMatch(assistantReportNarrative({ ...failed, answer: "STALE_SUCCESS", answer_is_current: false }, null), /STALE_SUCCESS/);
});

test("Report Studio fallback prefers any delivered artifact before a completed run without outputs", () => {
  const result = { ...artifact("result", "text/markdown", "markdown"), name: "result.md" };
  const runs = [
    { id: "newest-empty", status: "completed", prompt: "verify", artifacts: [], completed_at: "2026-08-03T00:00:00Z" },
    { id: "older-delivery", status: "completed", prompt: "answer", artifacts: [result], completed_at: "2026-08-02T00:00:00Z" },
  ];

  assert.equal(preferredAssistantReportRun(runs)?.id, "older-delivery");
  assert.equal(preferredAssistantReportRun([{ ...runs[0], id: "only-empty" }])?.id, "only-empty");
});

test("professional report prefers Markdown before other deliverables", () => {
  const markdown = artifact("report", "text/markdown", "markdown");
  const json = artifact("data", "application/json", "structured_text");
  assert.equal(preferredAssistantArtifact({ id: "run", status: "completed", prompt: "x", artifacts: [json, markdown] })?.id, "report");
});

test("recovered runs prefer the professional report and do not render a stale answer as the primary narrative", () => {
  const staleResult = { ...artifact("stale", "text/markdown", "markdown"), name: "result.md" };
  const professionalReport = { ...artifact("professional", "text/markdown", "markdown"), name: "professional-training-report.md" };
  const run = {
    id: "run_recovered",
    status: "completed",
    prompt: "uploaded",
    answer: "Training is blocked by FileNotFoundError.",
    artifacts: [staleResult, professionalReport],
  };

  assert.equal(preferredAssistantArtifact(run)?.id, "professional");
  assert.equal(assistantReportNarrative(run, professionalReport), "");
  assert.equal(assistantReportNarrative({ ...run, artifacts: [] }, null), run.answer);
});

test("Markdown parser emits safe structural blocks without interpreting HTML", () => {
  const blocks = parseReportMarkdown("# Title\n\n- one\n- two\n\n| Metric | Value |\n|---|---:|\n| CV | 0.82 |\n\n<script>alert(1)</script>");
  assert.deepEqual(blocks.map((block) => block.kind), ["heading", "list", "table", "paragraph"]);
  assert.deepEqual(blocks[2].headers, ["Metric", "Value"]);
  assert.deepEqual(blocks[2].rows, [["CV", "0.82"]]);
  assert.match(String(blocks[3].text), /<script>/);
});

test("CSV preserves quoted commas, escaped quotes and multiline fields", () => {
  const parsed = parseReportCsv('name,value\r\n"a,b","line1\nline2"\r\n"a""b",0\r\n');
  assert.equal(parsed.error, null);
  assert.deepEqual(parsed.rows, [["name", "value"], ["a,b", "line1\nline2"], ['a"b', "0"]]);
  assert.equal(parseReportCsv('a,b\n"unfinished').error, "CSV_QUOTE_UNCLOSED");
  assert.equal(parseReportCsv('a,b\n"quoted"wrong,1').error, "CSV_QUOTE_BOUNDARY_INVALID");
  assert.equal(parseReportCsv('a\n1\n2\n3', 2).truncated, true);
});
