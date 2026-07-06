import { NextResponse } from "next/server";
import { promises as fs } from "node:fs";
import path from "node:path";
import { prisma } from "@/lib/db";
import { logAction } from "@/lib/server/actions";
import { ensureWorkstationSeeded } from "@/lib/server/bootstrap";
import { encodeJson } from "@/lib/server/json";
import { latestExperimentPathWithAnyArtifacts, normalizeTaskId, resolveWorkspacePath, toRelativePath } from "@/lib/server/paths";
import { serializeReport } from "@/lib/server/serializers";

export const dynamic = "force-dynamic";

const FALLBACK_REPORTS = [
  "report.md",
  "workstation_report.md",
  "research_report.md",
  "local_report.md",
  "titanic_local_report.md"
] as const;

export async function GET(_request: Request, { params }: { params: { taskId: string } }) {
  await ensureWorkstationSeeded();
  const taskId = normalizeTaskId(params.taskId);
  const report = await prisma.report.findFirst({ where: { taskId }, orderBy: { updatedAt: "desc" } });
  return NextResponse.json({ ok: true, task_id: taskId, report: serializeReport(report ?? await filesystemReport(taskId)) });
}

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

function markdownToBasicHtml(markdown: string, title: string) {
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
    body { font-family: "Microsoft YaHei", Arial, sans-serif; max-width: 920px; margin: 40px auto; color: #0f172a; line-height: 1.75; }
    h1, h2, h3 { color: #0f172a; }
    li { margin: 4px 0; }
    figure { margin: 28px 0; padding: 14px; border: 1px solid #e2e8f0; border-radius: 8px; background: #f8fafc; }
    img { display: block; max-width: 100%; margin: 0 auto; background: #fff; border-radius: 6px; }
    figcaption { margin-top: 8px; text-align: center; color: #64748b; font-size: 12px; font-weight: 600; }
  </style>
</head>
<body>
${body}
</body>
</html>`;
}

function extractOutline(markdown: string) {
  return markdown
    .split(/\r?\n/)
    .filter((line) => /^#{1,3}\s+/.test(line))
    .map((line, index) => ({
      id: `section_${index + 1}`,
      level: line.match(/^#+/)?.[0].length ?? 1,
      title: line.replace(/^#+\s+/, "").trim()
    }));
}

async function filesystemReport(taskId: string) {
  const latest = await latestExperimentPathWithAnyArtifacts(taskId, [...FALLBACK_REPORTS]);
  if (!latest) return null;

  for (const name of FALLBACK_REPORTS) {
    const markdownPath = path.join(latest, name).replaceAll("\\", "/");
    const absolutePath = resolveWorkspacePath(markdownPath);
    const stat = await fs.stat(absolutePath).catch(() => null);
    if (!stat?.isFile()) continue;

    const markdown = await fs.readFile(absolutePath, "utf-8");
    const firstHeading = markdown.match(/^#\s+(.+)$/m)?.[1]?.trim();
    return {
      id: `${taskId}_filesystem_report`,
      taskId,
      runId: null,
      title: firstHeading || `${taskId} Research Report`,
      status: "filesystem_fallback",
      markdownContent: markdown,
      contentJson: encodeJson({
        outline: extractOutline(markdown),
        markdown_path: markdownPath,
        source: "filesystem_fallback",
        latest_experiment_path: latest
      }),
      markdownPath,
      docxPath: null,
      selectedSection: null,
      submittedAt: null,
      createdAt: stat.birthtime,
      updatedAt: stat.mtime
    };
  }

  return null;
}

export async function PATCH(request: Request, { params }: { params: { taskId: string } }) {
  await ensureWorkstationSeeded();
  const taskId = normalizeTaskId(params.taskId);
  const body = await request.json().catch(() => ({}));
  const title = String(body.title ?? `${taskId} Research Report`);
  const markdown = String(body.markdown_content ?? body.markdown ?? "");
  const selectedSection = typeof body.selected_section === "string" ? body.selected_section : null;
  const status = String(body.status ?? "draft");

  if (!markdown.trim()) {
    return NextResponse.json({ ok: false, error: "markdown_content is required" }, { status: 400 });
  }

  const draftDir = resolveWorkspacePath(path.join("workspace", "tasks", taskId, "reports", "draft"));
  await fs.mkdir(draftDir, { recursive: true });
  const markdownPath = path.join(draftDir, "report.md");
  const htmlPath = path.join(draftDir, "report.html");
  const metaPath = path.join(draftDir, "report.json");
  const content = {
    outline: extractOutline(markdown),
    html_path: toRelativePath(htmlPath),
    markdown_path: toRelativePath(markdownPath),
    saved_at: new Date().toISOString()
  };

  await fs.writeFile(markdownPath, markdown, "utf-8");
  await fs.writeFile(htmlPath, markdownToBasicHtml(markdown, title), "utf-8");
  await fs.writeFile(metaPath, JSON.stringify({ title, status, selected_section: selectedSection, ...content }, null, 2), "utf-8");

  const latestRun = await prisma.experimentRun.findFirst({
    where: { taskId },
    orderBy: { createdAt: "desc" }
  });

  const report = await prisma.report.upsert({
    where: { id: `${taskId}_latest_report` },
    update: {
      runId: latestRun?.id ?? undefined,
      title,
      status,
      markdownContent: markdown,
      contentJson: encodeJson(content),
      markdownPath: toRelativePath(markdownPath),
      selectedSection
    },
    create: {
      id: `${taskId}_latest_report`,
      taskId,
      runId: latestRun?.id ?? null,
      title,
      status,
      markdownContent: markdown,
      contentJson: encodeJson(content),
      markdownPath: toRelativePath(markdownPath),
      selectedSection
    }
  });

  await logAction({
    action: "save_report_draft",
    taskId,
    runId: latestRun?.id,
    message: "Report draft saved to SQLite and workspace files.",
    artifactPath: toRelativePath(markdownPath),
    metadata: {
      html_path: toRelativePath(htmlPath),
      selected_section: selectedSection,
      outline_count: content.outline.length
    }
  });

  return NextResponse.json({ ok: true, task_id: taskId, report: serializeReport(report), markdown_path: toRelativePath(markdownPath), html_path: toRelativePath(htmlPath) });
}
