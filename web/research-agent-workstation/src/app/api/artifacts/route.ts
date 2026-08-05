import crypto from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server";
import { resolveWorkspacePath, workspaceRoot } from "@/lib/server/paths";
import { readScientificReport } from "@/lib/server/scientific-report";

export const dynamic = "force-dynamic";

const allowedArtifactPath = /^(?:workspace[\\/]tasks[\\/]+[A-Za-z0-9_-]+[\\/]reports[\\/]+(?:figures|scientific)[\\/]+[A-Za-z0-9_.-]+(?:[\\/][A-Za-z0-9_.-]+)*\.(?:svg|png|jpe?g|webp|html|pdf|json|md|zip)|workspace[\\/]evomind_runs[\\/]+[A-Za-z0-9_-]+[\\/]+(?:(?:delivery[\\/](?:research_report\.html|evomind-siim-isic-(?:report\.pdf|results\.csv|code\.zip|evidence\.zip)|qa[\\/]report-page-\d{2}\.png))|(?:run\.json|task_graph\.json|events\.jsonl|messages\.jsonl|handoffs\.jsonl|qlora_config\.json|model_card\.md|review\.json|claim_audit\.json|candidate_freeze\.json|private_grader\.json|artifact_manifest\.json|deliverables\.json|human_gate\.json|version_comparison\.json|data[\\/]dataset_manifest\.json|llm_output[\\/](?:metrics\.json|telemetry\.jsonl|environment\.json|adapter[\\/](?:adapter_model\.safetensors|adapter_config\.json)))))$/i;

const contentTypes: Record<string, string> = {
  ".svg": "image/svg+xml; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".html": "text/html; charset=utf-8",
  ".pdf": "application/pdf",
  ".csv": "text/csv; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".md": "text/markdown; charset=utf-8",
  ".jsonl": "application/x-ndjson; charset=utf-8",
  ".zip": "application/zip",
  ".safetensors": "application/octet-stream"
};

export async function GET(request: Request) {
  const url = new URL(request.url);
  const relativePath = url.searchParams.get("path") ?? "";
  const normalizedPath = relativePath.replaceAll("/", path.sep).replaceAll("\\", path.sep);

  if (!allowedArtifactPath.test(normalizedPath)) {
    return NextResponse.json({ ok: false, error: "artifact path is not allowed" }, { status: 403 });
  }

  const target = path.resolve(resolveWorkspacePath(normalizedPath));
  const root = path.resolve(workspaceRoot);
  if (!target.startsWith(root + path.sep)) {
    return NextResponse.json({ ok: false, error: "artifact path escapes workspace" }, { status: 403 });
  }
  // path.resolve is lexical and does not stop a junction/symlink inside the
  // workspace from redirecting reads elsewhere.  Bind both paths to their
  // filesystem-canonical locations before opening the artifact.
  const [realRoot, realTarget] = await Promise.all([
    fs.realpath(root).catch(() => null),
    fs.realpath(target).catch(() => null),
  ]);
  if (!realRoot || !realTarget) {
    return NextResponse.json({ ok: false, error: "artifact not found" }, { status: 404 });
  }
  const canonicalRelative = path.relative(realRoot, realTarget);
  if (!canonicalRelative || canonicalRelative.startsWith(`..${path.sep}`) || canonicalRelative === ".." || path.isAbsolute(canonicalRelative)) {
    return NextResponse.json({ ok: false, error: "artifact canonical path escapes workspace" }, { status: 403 });
  }

  const taskId = url.searchParams.get("task_id") ?? "";
  const runId = url.searchParams.get("run_id") ?? "";
  const hasReportBinding = Boolean(taskId || runId);
  let expectedHash: string | null = null;
  let expectedBytes: number | null = null;
  if (hasReportBinding) {
    if (!/^[A-Za-z0-9_.-]{1,160}$/.test(taskId) || !/^[A-Za-z0-9_-]{8,180}$/.test(runId)) {
      return NextResponse.json({ ok: false, error: "invalid artifact report binding" }, { status: 400 });
    }
    const report = await readScientificReport(taskId, runId);
    const requestedPath = relativePath.replaceAll("\\", "/");
    const reportArtifact = report?.artifacts.find((artifact) => artifact.path?.replaceAll("\\", "/") === requestedPath);
    if (!report || report.task_id !== taskId || report.run_id !== runId || !reportArtifact || reportArtifact.status !== "ready") {
      return NextResponse.json({ ok: false, error: "artifact is not bound to the requested report manifest" }, { status: 403 });
    }
    expectedHash = reportArtifact.sha256;
    expectedBytes = reportArtifact.bytes;
  } else if (/^workspace[\\/]tasks[\\/].+[\\/]reports[\\/]scientific[\\/]/i.test(normalizedPath)) {
    return NextResponse.json({ ok: false, error: "scientific report artifacts require task and run binding" }, { status: 400 });
  }

  const extension = path.extname(realTarget).toLowerCase();
  const body = await fs.readFile(realTarget).catch(() => null);
  if (!body) {
    return NextResponse.json({ ok: false, error: "artifact not found" }, { status: 404 });
  }
  if (extension === ".svg") {
    const source = body.toString("utf-8");
    const unsafeSvg = /<(?:script|foreignObject|iframe|object|embed)\b|\son[a-z]+\s*=|(?:href|src)\s*=\s*["']\s*(?:javascript:|https?:|\/\/|file:|data:(?!image\/))/i;
    if (unsafeSvg.test(source)) {
      return NextResponse.json({ ok: false, error: "SVG artifact contains active content" }, { status: 422 });
    }
  }
  const actualHash = crypto.createHash("sha256").update(body).digest("hex");
  if (hasReportBinding && (expectedHash !== actualHash || expectedBytes !== body.byteLength)) {
    return NextResponse.json({ ok: false, error: "artifact bytes do not match the report manifest" }, { status: 409 });
  }

  const headers = {
    "Content-Type": contentTypes[extension] ?? "application/octet-stream",
    "Cache-Control": "no-store",
    "X-Artifact-Bytes": String(body.byteLength),
    "X-Artifact-SHA256": actualHash,
    "X-Content-Type-Options": "nosniff"
  } as Record<string, string>;
  if (extension === ".html") {
    headers["Content-Security-Policy"] = "default-src 'none'; script-src 'none'; object-src 'none'; style-src 'unsafe-inline'; img-src 'self' data:; font-src data:; base-uri 'none'; form-action 'none'; frame-ancestors 'self'";
    headers["X-Frame-Options"] = "SAMEORIGIN";
  }
  if (extension === ".svg" && url.searchParams.get("download") !== "1") {
    headers["Content-Disposition"] = `inline; filename="${path.basename(target).replaceAll('"', "")}"`;
  }
  if (url.searchParams.get("download") === "1") {
    headers["Content-Disposition"] = `attachment; filename="${path.basename(target).replaceAll('"', "")}"`;
  }

  return new NextResponse(body, {
    headers
  });
}
