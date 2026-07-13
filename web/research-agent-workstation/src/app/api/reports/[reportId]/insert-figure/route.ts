import { promises as fs } from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { logAction } from "@/lib/server/actions";
import { ensureWorkstationSeeded } from "@/lib/server/bootstrap";
import { encodeJson } from "@/lib/server/json";
import { resolveWorkspacePath, toRelativePath } from "@/lib/server/paths";
import { serializeReport } from "@/lib/server/serializers";

export const dynamic = "force-dynamic";

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

export async function POST(request: Request, { params }: { params: Promise<{ reportId: string }> }) {
  const { reportId } = await params;
  await ensureWorkstationSeeded();
  const body = await request.json().catch(() => ({}));
  const figurePath = String(body.figure_path ?? body.path ?? "");
  const caption = String(body.caption ?? "Generated figure");
  if (!figurePath) {
    return NextResponse.json({ ok: false, error: "figure_path is required" }, { status: 400 });
  }

  const report = await prisma.report.findUnique({ where: { id: reportId } });
  if (!report) {
    return NextResponse.json({ ok: false, error: "report not found" }, { status: 404 });
  }

  const nextMarkdown = report.markdownContent?.includes(figurePath) ? (report.markdownContent ?? "") : `${report.markdownContent ?? ""}\n\n![${caption}](${figurePath})\n`;
  const markdownPath = report.markdownPath ? resolveWorkspacePath(report.markdownPath) : resolveWorkspacePath(path.join("workspace", "tasks", report.taskId, "reports", "draft", "report.md"));
  const htmlPath = resolveWorkspacePath(path.join("workspace", "tasks", report.taskId, "reports", "draft", "report.html"));
  await fs.mkdir(path.dirname(markdownPath), { recursive: true });
  await fs.writeFile(markdownPath, nextMarkdown, "utf-8");
  await fs.writeFile(htmlPath, markdownToHtml(nextMarkdown, report.title), "utf-8");

  const content = {
    inserted_figure: figurePath,
    html_path: toRelativePath(htmlPath),
    markdown_path: toRelativePath(markdownPath),
    saved_at: new Date().toISOString()
  };
  const updated = await prisma.report.update({
    where: { id: reportId },
    data: {
      markdownContent: nextMarkdown,
      contentJson: encodeJson(content),
      markdownPath: toRelativePath(markdownPath)
    }
  });

  await logAction({
    action: "insert_report_figure",
    taskId: report.taskId,
    runId: report.runId ?? undefined,
    message: `Figure inserted into report: ${figurePath}`,
    artifactPath: toRelativePath(markdownPath),
    metadata: { figure_path: figurePath, caption }
  });

  return NextResponse.json({ ok: true, report: serializeReport(updated) });
}
