import { promises as fs } from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { logAction } from "@/lib/server/actions";
import { ensureWorkstationSeeded } from "@/lib/server/bootstrap";
import { encodeJson } from "@/lib/server/json";
import { latestExperimentPath, normalizeTaskId, readJsonFile, resolveWorkspacePath, toRelativePath } from "@/lib/server/paths";
import { serializeReport } from "@/lib/server/serializers";

export const dynamic = "force-dynamic";

function htmlEscape(value: string) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function artifactHtmlSrc(rawPath: string) {
  if (/^(https?:|data:|blob:|\/api\/)/i.test(rawPath)) return rawPath;
  return `/api/artifacts?path=${encodeURIComponent(rawPath)}`;
}

function markdownToHtml(markdown: string, title: string) {
  const body = markdown
    .split(/\r?\n/)
    .map((line) => {
      if (line.startsWith("# ")) return `<h1>${htmlEscape(line.slice(2))}</h1>`;
      if (line.startsWith("## ")) return `<h2>${htmlEscape(line.slice(3))}</h2>`;
      if (line.startsWith("### ")) return `<h3>${htmlEscape(line.slice(4))}</h3>`;
      const image = line.match(/^\s*!\[([^\]]*)\]\(([^)\s]+)(?:\s+["'][^"']+["'])?\)\s*$/);
      if (image) {
        const [, alt, src] = image;
        return `<figure><img src="${htmlEscape(artifactHtmlSrc(src))}" alt="${htmlEscape(alt)}"/><figcaption>${htmlEscape(alt || src)}</figcaption></figure>`;
      }
      if (line.startsWith("- ")) return `<li>${htmlEscape(line.slice(2))}</li>`;
      if (line.includes("|")) return `<p class="table-line">${htmlEscape(line)}</p>`;
      if (!line.trim()) return "<br/>";
      return `<p>${htmlEscape(line)}</p>`;
    })
    .join("\n");

  return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>${htmlEscape(title)}</title>
  <style>
    body { font-family: Inter, "Microsoft YaHei", Arial, sans-serif; max-width: 1180px; margin: 40px auto; color: #111827; line-height: 1.75; }
    h1 { font-size: 34px; margin-bottom: 8px; }
    h2 { margin-top: 34px; border-bottom: 1px solid #e5e7eb; padding-bottom: 8px; }
    h3 { margin-top: 24px; }
    p, li { font-size: 15px; }
    figure { margin: 28px 0; padding: 14px; border: 1px solid #e5e7eb; border-radius: 8px; background: #f9fafb; }
    img { display: block; max-width: 100%; margin: 0 auto; background: #fff; border-radius: 6px; }
    figcaption { margin-top: 8px; text-align: center; color: #4b5563; font-size: 12px; font-weight: 600; }
    .table-line { font-family: "Cascadia Mono", Consolas, monospace; white-space: pre-wrap; background: #f9fafb; margin: 0; padding: 2px 8px; }
  </style>
</head>
<body>
${body}
</body>
</html>`;
}

function toPairs(value: unknown) {
  return value && typeof value === "object" ? Object.entries(value as Record<string, unknown>) : [];
}

function formatMetricValue(value: unknown) {
  if (typeof value === "number") return Number.isFinite(value) ? value.toFixed(6) : String(value);
  if (typeof value === "boolean") return value ? "true" : "false";
  if (value === null || value === undefined) return "";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function metricTable(metrics: Record<string, unknown>) {
  const rows = Object.entries(metrics);
  if (!rows.length) {
    return "| Metric | Value |\n| --- | --- |\n| Status | Pending latest run |";
  }
  return [
    "| Metric | Value |",
    "| --- | --- |",
    ...rows.map(([key, value]) => `| ${key} | ${formatMetricValue(value)} |`)
  ].join("\n");
}

function findMetrics(validation: Record<string, unknown> | null, experiment: Record<string, unknown> | null, latestRun: Record<string, unknown> | null) {
  const candidates = [
    validation?.metrics,
    validation?.best_metrics,
    experiment?.best_metrics,
    latestRun?.best_metrics,
    latestRun?.metrics
  ];
  for (const candidate of candidates) {
    if (candidate && typeof candidate === "object" && Object.keys(candidate as Record<string, unknown>).length) {
      return candidate as Record<string, unknown>;
    }
  }
  const evaluation = experiment?.evaluation as Record<string, unknown> | undefined;
  const results = evaluation?.model_results as Record<string, Record<string, unknown>> | undefined;
  const first = results ? Object.values(results)[0] : undefined;
  return first ?? {};
}

function buildMarkdown({
  taskId,
  language,
  latest,
  validation,
  dataQuality,
  experiment,
  figures,
  latestRun
}: {
  taskId: string;
  language: string;
  latest: string | null;
  validation: Record<string, unknown> | null;
  dataQuality: Record<string, unknown> | null;
  experiment: Record<string, unknown> | null;
  figures: Array<{ name: string; path: string }>;
  latestRun: Record<string, unknown> | null;
}) {
  const zh = language !== "en-US";
  const metrics = findMetrics(validation, experiment, latestRun);
  const gateStatus = String(validation?.status ?? latestRun?.validation_status ?? latestRun?.status ?? "pending");
  const taskTitle = taskId.replaceAll("_", " ");
  const qualityRows = toPairs(dataQuality).filter(([, value]) => typeof value === "number" || typeof value === "string").slice(0, 8);
  const figureLines = figures.length
    ? figures.slice(0, 4).map((figure) => `![${figure.name}](${figure.path})`).join("\n\n")
    : zh
      ? "图表尚未生成。可在工作站 Professional Figures 面板生成后插入报告。"
      : "Figures are not generated yet. Use Professional Figures to create and insert them.";

  if (!zh) {
    return [
      `# ${taskTitle} Experiment Report`,
      "",
      "## Executive Summary",
      `This report is generated from the latest workstation-visible run for **${taskId}**. The current validation gate is **${gateStatus}**, and the latest experiment directory is \`${latest ?? "not available"}\`. The draft is saved to SQLite and workspace artifacts so it can be reviewed, edited, audited, and exported.`,
      "",
      "## Research Question and Task Definition",
      "The task is represented as an auditable machine-learning workflow. The workstation keeps task configuration, data checks, modeling records, validation gates, evidence artifacts, submission boundaries, and final reporting in one traceable loop.",
      "",
      "## Data Quality and Provenance",
      qualityRows.length ? "| Check | Value |\n| --- | --- |\n" + qualityRows.map(([key, value]) => `| ${key} | ${formatMetricValue(value)} |`).join("\n") : "Data quality artifacts are pending for this task.",
      "",
      "## Modeling Method and Validation",
      metricTable(metrics),
      "",
      "## Professional Figures",
      figureLines,
      "",
      "## Evidence Binding",
      `- Validation gate: ${gateStatus}`,
      `- Latest run: ${latest ?? "not recorded"}`,
      "- Report source: workstation database plus local artifact ledger",
      "- Human review: required before official submission or external publishing",
      "",
      "## Limitations and Controlled Boundaries",
      "- This draft does not prove a new official Kaggle score, rank, medal, or MLE-Bench parity.",
      "- Official submission still requires submission audit, claim audit, and human approval gate.",
      "- GPU/HPC training must be launched through workstation resource mode.",
      "",
      "## Reproducibility Command",
      "```powershell",
      `python scripts\\run_workstation_orchestrator.py --config configs\\${taskId}.yaml --output-base experiments --random-state 42`,
      "```"
    ].join("\n");
  }

  return [
    `# ${taskTitle} 实验报告`,
    "",
    "## 摘要",
    `本报告由 AI 科研工作站根据 **${taskId}** 的最新可见运行记录自动生成。当前验证 Gate 状态为 **${gateStatus}**，最新实验目录为 \`${latest ?? "暂无"}\`。报告已写入 SQLite 和 workspace artifact，可继续在工作站中编辑、审核和导出。`,
    "",
    "## 研究问题与任务定义",
    "该任务被组织为可审计的机器学习科研流程。工作站统一管理任务配置、数据检查、建模记录、验证门禁、证据 artifact、提交边界和最终报告，目标是形成可复现、可追踪、可人工审核的闭环，而不是单独运行一个训练脚本。",
    "",
    "## 数据质量与来源记录",
    qualityRows.length ? "| 检查项 | 数值 |\n| --- | --- |\n" + qualityRows.map(([key, value]) => `| ${key} | ${formatMetricValue(value)} |`).join("\n") : "当前任务的数据质量 artifact 尚未生成或需要补齐。",
    "",
    "## 建模方法与验证结果",
    metricTable(metrics),
    "",
    "## 专业图表",
    figureLines,
    "",
    "## 证据绑定与人工审核",
    `- 验证 Gate：${gateStatus}`,
    `- 最新运行：${latest ?? "未记录"}`,
    "- 报告来源：工作站数据库记录与本地 artifact ledger",
    "- 人工审核：正式提交、对外发布或教师汇报前需要人工 Gate 审核",
    "",
    "## 局限性与受控边界",
    "- 本报告草稿不证明新的 Kaggle 官方分数、排名、奖牌或 MLE-Bench 75 对齐结果。",
    "- 官方提交仍需要 submission audit、claim audit 和 human approval gate。",
    "- GPU/HPC 训练必须通过工作站 resource mode 发起，不能绕过工作站直接训练。",
    "",
    "## 复现实验命令",
    "```powershell",
    `python scripts\\run_workstation_orchestrator.py --config configs\\${taskId}.yaml --output-base experiments --random-state 42`,
    "```"
  ].join("\n");
}

export async function POST(request: Request, { params }: { params: Promise<{ taskId: string }> }) {
  await ensureWorkstationSeeded();
  const { taskId: rawTaskId } = await params;
  const taskId = normalizeTaskId(rawTaskId);
  const body = await request.json().catch(() => ({}));
  const language = String(body.language ?? "zh-CN");
  const latest = await latestExperimentPath(taskId);
  const [validation, dataQuality, experiment, latestRun, figuresManifest] = await Promise.all([
    latest ? readJsonFile(resolveWorkspacePath(path.join(latest, "validation_gate.json"))) as Promise<Record<string, unknown> | null> : Promise.resolve(null),
    latest ? readJsonFile(resolveWorkspacePath(path.join(latest, "data_quality.json"))) as Promise<Record<string, unknown> | null> : Promise.resolve(null),
    latest ? readJsonFile(resolveWorkspacePath(path.join(latest, "experiment_log.json"))) as Promise<Record<string, unknown> | null> : Promise.resolve(null),
    prisma.experimentRun.findFirst({ where: { taskId }, orderBy: { createdAt: "desc" } }).then((run) => run ? { ...run, best_metrics: run.metricsJson ? JSON.parse(run.metricsJson) as Record<string, unknown> : null } : null),
    readJsonFile(resolveWorkspacePath(path.join("workspace", "tasks", taskId, "reports", "figures", "figures_manifest.json"))) as Promise<{ figures?: Array<{ name: string; path: string }> } | null>
  ]);
  const figures = figuresManifest?.figures ?? [];
  const title = language === "en-US" ? `${taskId.replaceAll("_", " ")} Experiment Report` : `${taskId.replaceAll("_", " ")} 实验报告`;
  const markdown = buildMarkdown({ taskId, language, latest, validation, dataQuality, experiment, figures, latestRun });

  const draftDir = resolveWorkspacePath(path.join("workspace", "tasks", taskId, "reports", "draft"));
  await fs.mkdir(draftDir, { recursive: true });
  const markdownPath = path.join(draftDir, "report.md");
  const htmlPath = path.join(draftDir, "report.html");
  const metaPath = path.join(draftDir, "report.json");
  const content = {
    html_path: toRelativePath(htmlPath),
    markdown_path: toRelativePath(markdownPath),
    generated_by: "workstation_report_agent",
    language,
    source_run: latest,
    generated_at: new Date().toISOString()
  };

  await fs.writeFile(markdownPath, markdown, "utf-8");
  await fs.writeFile(htmlPath, markdownToHtml(markdown, title), "utf-8");
  await fs.writeFile(metaPath, JSON.stringify({ title, ...content }, null, 2), "utf-8");

  const report = await prisma.report.upsert({
    where: { id: `${taskId}_latest_report` },
    update: {
      runId: typeof latestRun?.id === "string" ? latestRun.id : undefined,
      title,
      status: "draft",
      markdownContent: markdown,
      contentJson: encodeJson(content),
      markdownPath: toRelativePath(markdownPath),
      selectedSection: language === "en-US" ? "Executive Summary" : "摘要"
    },
    create: {
      id: `${taskId}_latest_report`,
      taskId,
      runId: typeof latestRun?.id === "string" ? latestRun.id : null,
      title,
      status: "draft",
      markdownContent: markdown,
      contentJson: encodeJson(content),
      markdownPath: toRelativePath(markdownPath),
      selectedSection: language === "en-US" ? "Executive Summary" : "摘要"
    }
  });

  await logAction({
    action: "generate_report_draft",
    taskId,
    runId: typeof latestRun?.id === "string" ? latestRun.id : undefined,
    message: "Publishing-style report draft generated and saved.",
    artifactPath: toRelativePath(markdownPath),
    metadata: { html_path: toRelativePath(htmlPath), language, source_run: latest, figure_count: figures.length }
  });

  return NextResponse.json({
    ok: true,
    task_id: taskId,
    report: serializeReport(report),
    markdown_content: markdown,
    markdown_path: toRelativePath(markdownPath),
    html_path: toRelativePath(htmlPath)
  });
}
