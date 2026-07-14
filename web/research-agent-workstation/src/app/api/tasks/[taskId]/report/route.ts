import { createHash, randomUUID } from "node:crypto";
import { NextResponse } from "next/server";
import path from "node:path";
import { prisma } from "@/lib/db";
import { logAction } from "@/lib/server/actions";
import { ensureWorkstationSeeded } from "@/lib/server/bootstrap";
import { encodeJson } from "@/lib/server/json";
import { latestExperimentPathWithAnyArtifacts, normalizeTaskId, resolveWorkspacePath, toRelativePath, workspaceRoot } from "@/lib/server/paths";
import { serializeReport } from "@/lib/server/serializers";
import { ensurePrivateDirectory, readStableRegularTextFile, writeAtomicPrivateTextFile } from "@/lib/server/stable-file";

export const dynamic = "force-dynamic";

const FALLBACK_REPORTS = [
  "report.md",
  "workstation_report.md",
  "research_report.md",
  "local_report.md",
  "titanic_local_report.md"
] as const;
const MAX_REPORT_BYTES = 2_000_000;
const MAX_REPORT_HTML_BYTES = 4_000_000;
const MAX_REPORT_METADATA_BYTES = 128_000;
const MAX_TITLE_CHARACTERS = 300;
const MAX_TITLE_BYTES = 1_200;
const MAX_STATUS_CHARACTERS = 80;
const MAX_STATUS_BYTES = 320;
const MAX_SECTION_CHARACTERS = 300;
const MAX_SECTION_BYTES = 1_200;

export async function GET(_request: Request, { params }: { params: Promise<{ taskId: string }> }) {
  await ensureWorkstationSeeded();
  const { taskId: rawTaskId } = await params;
  const taskId = normalizeTaskId(rawTaskId);
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

function byteLength(value: string) {
  return Buffer.byteLength(value, "utf8");
}

function sha256(value: string) {
  return createHash("sha256").update(Buffer.from(value, "utf8")).digest("hex");
}

function hasUnpairedSurrogate(value: string) {
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code >= 0xd800 && code <= 0xdbff) {
      if (index + 1 >= value.length) return true;
      const next = value.charCodeAt(index + 1);
      if (next < 0xdc00 || next > 0xdfff) return true;
      index += 1;
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      return true;
    }
  }
  return false;
}

function validSingleLine(value: string, maxCharacters: number, maxBytes: number) {
  return Boolean(value.trim())
    && value.length <= maxCharacters
    && byteLength(value) <= maxBytes
    && !/[\u0000-\u001f\u007f]/.test(value)
    && !hasUnpairedSurrogate(value);
}

function workspaceRelativePath(absolutePath: string) {
  const relativePath = toRelativePath(absolutePath);
  if (!relativePath) throw new Error("Report artifact path is outside the workspace");
  return relativePath;
}

async function filesystemReport(taskId: string) {
  const latest = await latestExperimentPathWithAnyArtifacts(taskId, [...FALLBACK_REPORTS]);
  if (!latest) return null;

  for (const name of FALLBACK_REPORTS) {
    const markdownPath = path.join(latest, name).replaceAll("\\", "/");
    const absolutePath = resolveWorkspacePath(markdownPath);
    const stable = await readStableRegularTextFile(absolutePath, {
      allowedRoot: resolveWorkspacePath(latest),
      maxBytes: 2_000_000
    }).catch(() => null);
    if (!stable) continue;
    const markdown = stable.text;
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
      createdAt: stable.birthtime,
      updatedAt: stable.mtime
    };
  }

  return null;
}

export async function PATCH(request: Request, { params }: { params: Promise<{ taskId: string }> }) {
  await ensureWorkstationSeeded();
  const { taskId: rawTaskId } = await params;
  const taskId = normalizeTaskId(rawTaskId);
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    return NextResponse.json({ ok: false, error: "invalid_report_payload" }, { status: 400 });
  }
  const payload = body as Record<string, unknown>;
  const rawMarkdown = payload.markdown_content ?? payload.markdown;
  const rawTitle = payload.title ?? `${taskId} Research Report`;
  const rawStatus = payload.status ?? "draft";
  const rawSelectedSection = payload.selected_section ?? null;
  if (
    typeof rawMarkdown !== "string"
    || typeof rawTitle !== "string"
    || typeof rawStatus !== "string"
    || (rawSelectedSection !== null && typeof rawSelectedSection !== "string")
  ) {
    return NextResponse.json({ ok: false, error: "invalid_report_payload" }, { status: 400 });
  }
  const title = rawTitle;
  const markdown = rawMarkdown;
  const selectedSection = rawSelectedSection;
  const status = rawStatus;

  if (
    !markdown.trim()
    || byteLength(markdown) > MAX_REPORT_BYTES
    || hasUnpairedSurrogate(markdown)
    || !validSingleLine(title, MAX_TITLE_CHARACTERS, MAX_TITLE_BYTES)
    || !validSingleLine(status, MAX_STATUS_CHARACTERS, MAX_STATUS_BYTES)
    || (selectedSection !== null && !validSingleLine(selectedSection, MAX_SECTION_CHARACTERS, MAX_SECTION_BYTES))
  ) {
    return NextResponse.json({ ok: false, error: "invalid_report_payload" }, { status: 400 });
  }

  const revisionId = randomUUID();
  const draftDir = resolveWorkspacePath(path.join("workspace", "tasks", taskId, "reports", "draft", revisionId));
  const markdownPath = path.join(draftDir, "report.md");
  const htmlPath = path.join(draftDir, "report.html");
  const metaPath = path.join(draftDir, "report.json");
  const html = markdownToBasicHtml(markdown, title);
  if (byteLength(html) > MAX_REPORT_HTML_BYTES || hasUnpairedSurrogate(html)) {
    return NextResponse.json({ ok: false, error: "report_html_too_large" }, { status: 413 });
  }
  const markdownRelativePath = workspaceRelativePath(markdownPath);
  const htmlRelativePath = workspaceRelativePath(htmlPath);
  const metadataRelativePath = workspaceRelativePath(metaPath);
  const savedAt = new Date().toISOString();
  const markdownSha256 = sha256(markdown);
  const htmlSha256 = sha256(html);
  const content = {
    outline: extractOutline(markdown),
    html_path: htmlRelativePath,
    markdown_path: markdownRelativePath,
    metadata_path: metadataRelativePath,
    revision_id: revisionId,
    markdown_sha256: markdownSha256,
    html_sha256: htmlSha256,
    saved_at: savedAt
  };
  const metadata = `${JSON.stringify({ title, status, selected_section: selectedSection, ...content }, null, 2)}\n`;
  if (byteLength(metadata) > MAX_REPORT_METADATA_BYTES || hasUnpairedSurrogate(metadata)) {
    return NextResponse.json({ ok: false, error: "report_metadata_too_large" }, { status: 413 });
  }

  try {
    await ensurePrivateDirectory(draftDir, workspaceRoot);
    const markdownFile = await writeAtomicPrivateTextFile(markdownPath, markdown, {
      allowedRoot: workspaceRoot,
      maxBytes: MAX_REPORT_BYTES
    });
    const htmlFile = await writeAtomicPrivateTextFile(htmlPath, html, {
      allowedRoot: workspaceRoot,
      maxBytes: MAX_REPORT_HTML_BYTES
    });
    if (markdownFile.sha256 !== markdownSha256 || htmlFile.sha256 !== htmlSha256) {
      throw new Error("Report artifact hash mismatch after write");
    }
    await writeAtomicPrivateTextFile(metaPath, metadata, {
      allowedRoot: workspaceRoot,
      maxBytes: MAX_REPORT_METADATA_BYTES
    });

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
        markdownPath: markdownRelativePath,
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
        markdownPath: markdownRelativePath,
        selectedSection
      }
    });

    await logAction({
      action: "save_report_draft",
      taskId,
      runId: latestRun?.id,
      message: "Report draft saved to SQLite and workspace files.",
      artifactPath: markdownRelativePath,
      metadata: {
        html_path: htmlRelativePath,
        metadata_path: metadataRelativePath,
        selected_section: selectedSection,
        outline_count: content.outline.length,
        revision_id: revisionId,
        markdown_sha256: markdownSha256,
        html_sha256: htmlSha256
      }
    });

    return NextResponse.json({
      ok: true,
      task_id: taskId,
      report: serializeReport(report),
      revision_id: revisionId,
      markdown_path: markdownRelativePath,
      html_path: htmlRelativePath
    });
  } catch {
    return NextResponse.json({ ok: false, error: "report_artifact_write_failed" }, { status: 500 });
  }
}
