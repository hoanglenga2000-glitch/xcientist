import { createHash, randomUUID } from "node:crypto";
import path from "node:path";
import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { logAction } from "@/lib/server/actions";
import { ensureWorkstationSeeded } from "@/lib/server/bootstrap";
import { encodeJson } from "@/lib/server/json";
import { normalizeTaskId, resolveWorkspacePath, toRelativePath, workspaceRoot } from "@/lib/server/paths";
import { serializeReport } from "@/lib/server/serializers";
import { ensurePrivateDirectory, writeAtomicPrivateTextFile } from "@/lib/server/stable-file";

export const dynamic = "force-dynamic";

const MAX_REPORT_BYTES = 2_000_000;
const MAX_REPORT_HTML_BYTES = 4_000_000;
const MAX_REPORT_METADATA_BYTES = 128_000;
const MAX_TITLE_CHARACTERS = 300;
const MAX_TITLE_BYTES = 1_200;
const MAX_CAPTION_CHARACTERS = 500;
const MAX_CAPTION_BYTES = 2_000;
const MAX_FIGURE_PATH_BYTES = 2_048;
const MAX_REPORT_ID_BYTES = 512;

function htmlEscape(value: string) {
  return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
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
      if (!line.trim()) return "<br/>";
      return `<p>${htmlEscape(line)}</p>`;
    })
    .join("\n");
  return `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"/><title>${htmlEscape(title)}</title><style>body{font-family:"Microsoft YaHei",Arial,sans-serif;max-width:920px;margin:40px auto;color:#0f172a;line-height:1.75}figure{margin:28px 0;padding:14px;border:1px solid #e2e8f0;border-radius:8px;background:#f8fafc}img{display:block;max-width:100%;margin:0 auto;background:#fff;border-radius:6px}figcaption{margin-top:8px;text-align:center;color:#64748b;font-size:12px;font-weight:600}</style></head><body>${body}</body></html>`;
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

function normalizeFigurePath(value: string, taskId: string) {
  if (!value.trim() || byteLength(value) > MAX_FIGURE_PATH_BYTES || /[\u0000-\u001f\u007f]/.test(value) || hasUnpairedSurrogate(value)) {
    return null;
  }
  try {
    const relativePath = workspaceRelativePath(resolveWorkspacePath(value)).replaceAll("\\", "/");
    const prefix = `workspace/tasks/${taskId}/reports/figures/`;
    const filename = relativePath.startsWith(prefix) ? relativePath.slice(prefix.length) : "";
    if (!filename || filename.includes("/") || /[\s()[\]]/.test(filename) || !/\.(svg|png|jpe?g|webp)$/i.test(filename)) return null;
    return relativePath;
  } catch {
    return null;
  }
}

export async function POST(request: Request, { params }: { params: Promise<{ reportId: string }> }) {
  const { reportId } = await params;
  await ensureWorkstationSeeded();
  if (!validSingleLine(reportId, 256, MAX_REPORT_ID_BYTES)) {
    return NextResponse.json({ ok: false, error: "invalid_report_id" }, { status: 400 });
  }
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    return NextResponse.json({ ok: false, error: "invalid_figure_payload" }, { status: 400 });
  }
  const payload = body as Record<string, unknown>;
  const rawFigurePath = payload.figure_path ?? payload.path;
  const rawCaption = payload.caption ?? "Generated figure";
  if (
    typeof rawFigurePath !== "string"
    || typeof rawCaption !== "string"
    || !validSingleLine(rawCaption, MAX_CAPTION_CHARACTERS, MAX_CAPTION_BYTES)
    || /[\[\]]/.test(rawCaption)
  ) {
    return NextResponse.json({ ok: false, error: "invalid_figure_payload" }, { status: 400 });
  }

  try {
    const report = await prisma.report.findUnique({ where: { id: reportId } });
    if (!report) {
      return NextResponse.json({ ok: false, error: "report_not_found" }, { status: 404 });
    }
    let taskId: string;
    try {
      taskId = normalizeTaskId(report.taskId);
    } catch {
      return NextResponse.json({ ok: false, error: "invalid_report_state" }, { status: 409 });
    }
    const figurePath = normalizeFigurePath(rawFigurePath, taskId);
    if (!figurePath) {
      return NextResponse.json({ ok: false, error: "invalid_figure_path" }, { status: 400 });
    }
    if (!validSingleLine(report.title, MAX_TITLE_CHARACTERS, MAX_TITLE_BYTES)) {
      return NextResponse.json({ ok: false, error: "invalid_report_state" }, { status: 409 });
    }

    const currentMarkdown = report.markdownContent ?? "";
    const nextMarkdown = currentMarkdown.includes(figurePath) ? currentMarkdown : `${currentMarkdown}\n\n![${rawCaption}](${figurePath})\n`;
    const html = markdownToHtml(nextMarkdown, report.title);
    if (
      byteLength(nextMarkdown) > MAX_REPORT_BYTES
      || byteLength(html) > MAX_REPORT_HTML_BYTES
      || hasUnpairedSurrogate(nextMarkdown)
      || hasUnpairedSurrogate(html)
    ) {
      return NextResponse.json({ ok: false, error: "report_with_figure_exceeds_limits" }, { status: 413 });
    }

    const revisionId = randomUUID();
    const draftDir = resolveWorkspacePath(path.join("workspace", "tasks", taskId, "reports", "draft", revisionId));
    const markdownPath = path.join(draftDir, "report.md");
    const htmlPath = path.join(draftDir, "report.html");
    const metaPath = path.join(draftDir, "report.json");
    const markdownRelativePath = workspaceRelativePath(markdownPath);
    const htmlRelativePath = workspaceRelativePath(htmlPath);
    const metadataRelativePath = workspaceRelativePath(metaPath);
    const markdownSha256 = sha256(nextMarkdown);
    const htmlSha256 = sha256(html);
    const content = {
      inserted_figure: figurePath,
      caption: rawCaption,
      html_path: htmlRelativePath,
      markdown_path: markdownRelativePath,
      metadata_path: metadataRelativePath,
      revision_id: revisionId,
      markdown_sha256: markdownSha256,
      html_sha256: htmlSha256,
      saved_at: new Date().toISOString()
    };
    const metadata = `${JSON.stringify({ title: report.title, ...content }, null, 2)}\n`;
    if (byteLength(metadata) > MAX_REPORT_METADATA_BYTES || hasUnpairedSurrogate(metadata)) {
      return NextResponse.json({ ok: false, error: "report_metadata_exceeds_limits" }, { status: 413 });
    }

    await ensurePrivateDirectory(draftDir, workspaceRoot);
    const markdownFile = await writeAtomicPrivateTextFile(markdownPath, nextMarkdown, {
      allowedRoot: workspaceRoot,
      maxBytes: MAX_REPORT_BYTES
    });
    const htmlFile = await writeAtomicPrivateTextFile(htmlPath, html, {
      allowedRoot: workspaceRoot,
      maxBytes: MAX_REPORT_HTML_BYTES
    });
    if (markdownFile.sha256 !== markdownSha256 || htmlFile.sha256 !== htmlSha256) {
      throw new Error("Report figure artifact hash mismatch after write");
    }
    await writeAtomicPrivateTextFile(metaPath, metadata, {
      allowedRoot: workspaceRoot,
      maxBytes: MAX_REPORT_METADATA_BYTES
    });

    const update = await prisma.report.updateMany({
      where: { id: reportId, updatedAt: report.updatedAt },
      data: {
        markdownContent: nextMarkdown,
        contentJson: encodeJson(content),
        markdownPath: markdownRelativePath
      }
    });
    if (update.count !== 1) {
      return NextResponse.json({ ok: false, error: "report_revision_conflict" }, { status: 409 });
    }
    const updated = await prisma.report.findUnique({ where: { id: reportId } });
    if (!updated) {
      return NextResponse.json({ ok: false, error: "report_not_found" }, { status: 404 });
    }

    await logAction({
      action: "insert_report_figure",
      taskId,
      runId: report.runId ?? undefined,
      message: "A workspace figure was inserted into a new report revision.",
      artifactPath: markdownRelativePath,
      metadata: {
        figure_path: figurePath,
        caption: rawCaption,
        html_path: htmlRelativePath,
        metadata_path: metadataRelativePath,
        revision_id: revisionId,
        markdown_sha256: markdownSha256,
        html_sha256: htmlSha256
      }
    });

    return NextResponse.json({ ok: true, report: serializeReport(updated), revision_id: revisionId });
  } catch {
    return NextResponse.json({ ok: false, error: "report_figure_insert_failed" }, { status: 500 });
  }
}
