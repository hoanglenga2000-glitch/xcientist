export type AssistantReportArtifact = {
  id: string;
  run_id: string;
  name: string;
  media_type: string;
  bytes: number;
  sha256: string;
  preview_kind: string;
  preview_url: string;
  download_url: string;
  verified_at?: string;
};

export type AssistantReportRun = {
  id: string;
  status: string;
  prompt: string;
  answer?: string;
  answer_is_current?: boolean;
  project_id?: string;
  conversation_id?: string;
  selected_task?: string;
  user_pause_requested?: boolean;
  error_class?: string;
  error_message?: string;
  model_provider?: string;
  model?: string;
  plan?: { steps?: Array<{ id?: string; label?: string; status?: string; detail?: string }> };
  artifacts?: AssistantReportArtifact[];
  approvals?: Array<{ id: string; tool_name: string; status: string; risk_level?: string; reversible?: boolean }>;
  attachments?: Array<{ id: string; name: string; bytes: number; sha256: string }>;
  created_at?: string;
  updated_at?: string;
  completed_at?: string;
};

export type ReportMarkdownBlock = {
  kind: "heading" | "paragraph" | "list" | "code" | "table";
  level?: number;
  text?: string;
  items?: string[];
  headers?: string[];
  rows?: string[][];
};

function timestamp(run: AssistantReportRun) {
  const value = run.completed_at || run.updated_at || run.created_at || "";
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

export function completedAssistantReportRuns(runs: AssistantReportRun[]) {
  return runs
    .filter((run) => run.status === "completed")
    .sort((left, right) => timestamp(right) - timestamp(left));
}

export function reportableAssistantRuns(runs: AssistantReportRun[]) {
  return runs.filter((run) => Boolean(run.id)).sort((left, right) => timestamp(right) - timestamp(left));
}

export function assistantReportPresentation(run: AssistantReportRun | null | undefined, locale = "zh-CN") {
  const zh = locale === "zh-CN";
  const status = run?.status || "unknown";
  const labels: Record<string, [string, string]> = {
    completed: ["执行已完成", "Execution completed"], running: ["执行中", "Executing"],
    planning: ["规划中", "Planning"], verifying: ["核验中", "Verifying"],
    recovering: ["等待恢复核对", "Awaiting recovery checks"], paused: ["已暂停", "Paused"],
    pausing: ["正在安全暂停", "Pausing at a safe boundary"], waiting_approval: ["等待审批", "Awaiting approval"],
    failed: ["执行失败", "Execution failed"], blocked: ["需要处理", "Needs attention"],
    cancelled: ["已取消", "Cancelled"], queued: ["排队中", "Queued"], created: ["已创建", "Created"],
  };
  const prompt = run?.prompt || "";
  const kind = /训练|微调|\btrain|fine.?tun/i.test(prompt) ? "training"
    : /推理|预测|inference|predict/i.test(prompt) ? "inference"
    : /连接|连通|健康|检查|diagnostic|health|connect|(?:gpu|显卡|算力).{0,20}(?:状态|情况|可用|正常|status|availability)/i.test(prompt) ? "diagnostic" : "task";
  const titles = {
    training: ["训练任务记录", "Training task record"], inference: ["推理任务记录", "Inference task record"],
    diagnostic: ["连接与执行诊断", "Connection and execution diagnostics"], task: ["任务记录与成果", "Task record and outputs"],
  };
  const hasReport = Boolean(run && hasRenderableReportArtifact(run));
  const artifactCount = run?.artifacts?.length ?? 0;
  return {
    kind, title: titles[kind][zh ? 0 : 1],
    statusTone: status === "completed" ? "success" : status === "failed" ? "danger"
      : ["blocked", "recovering", "waiting_approval", "paused", "pausing"].includes(status) ? "warning" : "neutral",
    statusLabel: labels[status]?.[zh ? 0 : 1] || (zh ? "状态未确认" : "State unconfirmed"),
    reportLabel: hasReport ? (zh ? "报告文件已发布" : "Report file published")
      : artifactCount ? (zh ? "已有产物，尚无正式报告" : "Artifacts available; no formal report")
      : (zh ? "尚无已发布产物" : "No published artifacts"),
    evidenceLabel: artifactCount ? (zh ? "文件身份已登记；科研结论须单独核验" : "File identities recorded; research claims require separate review")
      : (zh ? "执行步骤不等于科研验证" : "Execution steps are not scientific verification"),
  };
}

export function checkedAssistantReportDetail(value: unknown, runId: string): AssistantReportRun {
  if (!value || typeof value !== "object") throw new Error("任务详情格式无效。");
  const run = value as AssistantReportRun;
  if (run.id !== runId || typeof run.status !== "string" || typeof run.prompt !== "string") {
    throw new Error("任务详情与所选 Run 不匹配。");
  }
  return { ...run, artifacts: (Array.isArray(run.artifacts) ? run.artifacts : []).filter((artifact) => artifact.run_id === runId) };
}

export function assistantReportLead(run: AssistantReportRun, locale = "zh-CN") {
  if (run.answer?.trim() && run.answer_is_current !== false) {
    const lead = run.answer.trim().split(/\n\s*\n/)[0].replace(/^#{1,6}\s*/, "").replace(/\*\*|`/g, "").replace(/\s+/g, " ");
    return lead.length > 180 ? lead.slice(0, 180) + "…" : lead;
  }
  const presentation = assistantReportPresentation(run, locale);
  return presentation.reportLabel + "。" + presentation.evidenceLabel + "。";
}

export function reportJobStatusLabel(status: string, locale = "zh-CN") {
  const labels: Record<string, [string, string]> = {
    queued: ["报告排队生成中", "Report queued"], running: ["正在生成报告", "Generating report"],
    ready: ["报告已生成，可下载", "Report ready to download"], partial: ["报告部分生成，请查看缺失项", "Partial report; review missing evidence"],
    failed: ["报告生成失败", "Report generation failed"], paused: ["报告已暂停", "Report paused"],
  };
  return labels[status]?.[locale === "zh-CN" ? 0 : 1] ?? (locale === "zh-CN" ? "报告状态待确认" : "Report state unconfirmed");
}

export function reportExportDownloadUrl(runId: string, jobRunId: string, artifact: { id: string; run_id?: string }) {
  if (runId !== jobRunId || (artifact.run_id && artifact.run_id !== runId) || !/^artifact_[a-f0-9]{32}$/.test(artifact.id)) return "";
  return `/api/assistant/artifacts/${encodeURIComponent(artifact.id)}?download=1`;
}

function reportArtifactPriority(artifact: AssistantReportArtifact) {
  const name = artifact.name.trim().toLowerCase();
  const reportLike = /(^|[-_.])(report|analysis)([-_.]|$)/.test(name);
  if (/^professional(?:[-_.].*)?report\.(?:md|markdown|html?|pdf)$/i.test(name)
    || /^professional[-_.]training[-_.]report\.(?:md|markdown|html?|pdf)$/i.test(name)) return 0;
  if (reportLike && (artifact.preview_kind === "markdown" || artifact.media_type.includes("markdown"))) return 1;
  if (reportLike && ["html", "pdf", "structured_text", "text"].includes(artifact.preview_kind)) return 2;
  return 3;
}

function hasRenderableReportArtifact(run: AssistantReportRun) {
  return (run.artifacts ?? []).some((artifact) => reportArtifactPriority(artifact) < 3);
}

function hasTrainingResultBundle(run: AssistantReportRun) {
  const names = (run.artifacts ?? []).map((artifact) => artifact.name.trim().toLowerCase());
  const hasMetrics = names.some((name) => /(^|[-_.])(metrics?|fold[-_.]?metrics|evaluation)([-_.]|$)/.test(name));
  const hasTrainingOutput = names.some((name) => /(^|[-_.])(predictions?|oof[-_.]?predictions?|submission|model|checkpoint|training[-_.]?log)([-_.]|$)/.test(name));
  return hasMetrics && hasTrainingOutput;
}

export function preferredAssistantReportRun(runs: AssistantReportRun[]) {
  const completed = completedAssistantReportRuns(runs);
  return completed.find((run) => hasRenderableReportArtifact(run) || hasTrainingResultBundle(run))
    ?? completed.find((run) => (run.artifacts?.length ?? 0) > 0)
    ?? completed[0]
    ?? null;
}

export function resolveAssistantReportRunId(runs: AssistantReportRun[], requestedRunId = "") {
  const available = reportableAssistantRuns(runs);
  if (requestedRunId) return available.find((run) => run.id === requestedRunId)?.id ?? "";
  return preferredAssistantReportRun(available)?.id ?? available[0]?.id ?? "";
}

export function preferredAssistantArtifact(run: AssistantReportRun | null | undefined) {
  const artifacts = run?.artifacts ?? [];
  const priority = ["markdown", "html", "pdf", "structured_text", "table", "text", "image", "svg", "archive"];
  return [...artifacts].sort((left, right) => {
    const reportPriority = reportArtifactPriority(left) - reportArtifactPriority(right);
    if (reportPriority !== 0) return reportPriority;
    const leftIndex = priority.indexOf(left.preview_kind);
    const rightIndex = priority.indexOf(right.preview_kind);
    const previewPriority = (leftIndex < 0 ? priority.length : leftIndex) - (rightIndex < 0 ? priority.length : rightIndex);
    if (previewPriority !== 0) return previewPriority;
    return left.name.localeCompare(right.name);
  })[0] ?? null;
}

export function preferredAssistantReportArtifact(run: AssistantReportRun | null | undefined) {
  const candidate = preferredAssistantArtifact(run);
  return candidate && reportArtifactPriority(candidate) < 3 ? candidate : null;
}

export function assistantReportNarrative(
  run: AssistantReportRun | null | undefined,
  selectedArtifact: AssistantReportArtifact | null | undefined,
  locale = "zh-CN",
) {
  if (selectedArtifact) return "";
  if (run?.answer?.trim() && run.answer_is_current !== false) return run.answer.trim();
  if (!run) return "";
  const presentation = assistantReportPresentation(run, locale);
  const steps = run.plan?.steps ?? [];
  const zh = locale === "zh-CN";
  return [
    `## ${zh ? "执行记录摘要" : "Execution record summary"}`,
    `${presentation.title} · ${presentation.statusLabel}`,
    presentation.reportLabel + "。" + presentation.evidenceLabel + "。",
    zh ? "以下来自任务记录，不是补造的模型结论。只读诊断不一定产生模型或科研图表。" : "The following comes from task records, not invented model conclusions. Read-only diagnostics may not produce models or research figures.",
    ...(steps.length ? [`### ${zh ? "已记录的执行步骤" : "Recorded execution steps"}`,
      ...steps.map((step) => `- ${step.label || step.id || "Step"} · ${step.status || "unknown"}${step.detail ? `：${step.detail}` : ""}`)] : []),
    zh ? "### 证据边界\n尚无当前有效的文字总结。请查看执行、审计与文件；不能仅根据步骤计数判定研究成功。"
      : "### Evidence boundary\nNo current textual conclusion is available. Inspect execution, audit and files; step counts alone do not establish research success.",
  ].join("\n\n");
}

export function assistantReportTitle(run: AssistantReportRun) {
  const prompt = run.prompt
    .replace(/^\[Selected task:[^\]]+\]\s*/i, "")
    .replace(/\s+/g, " ")
    .trim();
  if (!prompt) return `DeepEvo Task ${run.id}`;
  const sentence = prompt.split(/[。！？!?]\s*/, 1)[0]?.trim() || prompt;
  return sentence.length > 72 ? `${sentence.slice(0, 72)}…` : sentence;
}

export function parseReportCsv(value: string, maxRows = 101) {
  const rows: string[][] = [];
  let row: string[] = [], cell = "", quoted = false, closed = false;
  const input = value.replace(/^\uFEFF/, "");
  for (let index = 0; index < input.length; index += 1) {
    const character = input[index];
    if (quoted) {
      if (character === '"' && input[index + 1] === '"') { cell += '"'; index += 1; }
      else if (character === '"') { quoted = false; closed = true; }
      else cell += character;
      continue;
    }
    if (character === '"' && cell === "" && !closed) { quoted = true; continue; }
    if (character === ",") { row.push(cell); cell = ""; closed = false; continue; }
    if (character === "\r" || character === "\n") {
      if (character === "\r" && input[index + 1] === "\n") index += 1;
      row.push(cell); rows.push(row); row = []; cell = ""; closed = false;
      if (rows.length >= maxRows) return { rows, truncated: index + 1 < input.length, error: null };
      continue;
    }
    if (closed) {
      if (/\s/.test(character)) continue;
      return { rows: [], truncated: false, error: "CSV_QUOTE_BOUNDARY_INVALID" };
    }
    cell += character;
  }
  if (quoted) return { rows: [], truncated: false, error: "CSV_QUOTE_UNCLOSED" };
  if (cell || row.length || closed) { row.push(cell); rows.push(row); }
  return { rows, truncated: false, error: null };
}

export function parseReportMarkdown(value: string): ReportMarkdownBlock[] {
  const lines = value.replace(/\r\n?/g, "\n").split("\n");
  const blocks: ReportMarkdownBlock[] = [];
  let paragraph: string[] = [];
  let list: string[] = [];
  let code: string[] | null = null;

  const flushParagraph = () => {
    const text = paragraph.join(" ").trim();
    if (text) blocks.push({ kind: "paragraph", text });
    paragraph = [];
  };
  const flushList = () => {
    if (list.length) blocks.push({ kind: "list", items: list });
    list = [];
  };

  const tableCells = (line: string) => line.trim().replace(/^\||\|$/g, "").split(/(?<!\\)\|/).map((cell) => cell.replace(/\\\|/g, "|").trim());
  const tableDivider = (line: string) => {
    const cells = tableCells(line);
    return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell.replace(/\s+/g, "")));
  };

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    if (line.trim().startsWith("```")) {
      flushParagraph();
      flushList();
      if (code) {
        blocks.push({ kind: "code", text: code.join("\n") });
        code = null;
      } else {
        code = [];
      }
      continue;
    }
    if (code) {
      code.push(line);
      continue;
    }
    // Read older report packages without rewriting their immutable Markdown:
    // that renderer separated table lines with a single empty line.
    let dividerIndex = index + 1;
    if (line.trim().startsWith("|") && line.trim().endsWith("|") && lines[dividerIndex]?.trim() === "") dividerIndex += 1;
    if (line.includes("|") && dividerIndex < lines.length && tableDivider(lines[dividerIndex])) {
      flushParagraph();
      flushList();
      const headers = tableCells(line);
      const rows: string[][] = [];
      index = dividerIndex + 1;
      while (index < lines.length) {
        if (!lines[index].trim() && lines[index + 1]?.trim().startsWith("|")) { index += 1; continue; }
        if (!lines[index].trim() || !lines[index].includes("|")) break;
        const cells = tableCells(lines[index]);
        rows.push(headers.map((_, cellIndex) => cells[cellIndex] ?? ""));
        index += 1;
      }
      index -= 1;
      blocks.push({ kind: "table", headers, rows });
      continue;
    }
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      flushList();
      blocks.push({ kind: "heading", level: heading[1].length, text: heading[2].trim() });
      continue;
    }
    const item = line.match(/^\s*[-*]\s+(.+)$/);
    if (item) {
      flushParagraph();
      list.push(item[1].trim());
      continue;
    }
    if (!line.trim()) {
      flushParagraph();
      flushList();
      continue;
    }
    flushList();
    paragraph.push(line.trim());
  }
  if (code) blocks.push({ kind: "code", text: code.join("\n") });
  flushParagraph();
  flushList();
  return blocks;
}
