import crypto from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server";
import { readJsonFile, resolveWorkspacePath } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

const allowedFiles = new Map([
  ["evomind-siim-isic-report.pdf", "application/pdf"],
  ["evomind-siim-isic-results.csv", "text/csv; charset=utf-8"],
  ["evomind-siim-isic-code.zip", "application/zip"],
  ["evomind-siim-isic-evidence.zip", "application/zip"],
]);

function validRunId(value: string) {
  return /^[A-Za-z0-9_-]{8,160}$/.test(value);
}

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ runId: string; fileName: string }> },
) {
  const { runId, fileName } = await params;
  const contentType = allowedFiles.get(fileName);
  if (!validRunId(runId) || !contentType) {
    return NextResponse.json({ ok: false, error: "invalid SIIM deliverable request" }, { status: 400 });
  }

  const runDir = resolveWorkspacePath(`workspace/evomind_runs/${runId}`);
  const manifest = await readJsonFile(`${runDir}/artifact_manifest.json`) as Record<string, unknown> | null;
  if (!manifest || manifest.run_id !== runId || manifest.status !== "verified") {
    return NextResponse.json({ ok: false, error: "verified artifact manifest is unavailable" }, { status: 404 });
  }
  const artifacts = Array.isArray(manifest.artifacts) ? manifest.artifacts : [];
  const record = artifacts.find((item): item is Record<string, unknown> => (
    Boolean(item)
    && typeof item === "object"
    && (item as Record<string, unknown>).path === fileName
  ));
  if (!record || typeof record.sha256 !== "string" || typeof record.bytes !== "number") {
    return NextResponse.json({ ok: false, error: "deliverable is not hash-bound to this run" }, { status: 404 });
  }

  const target = path.resolve(runDir, fileName);
  if (path.dirname(target) !== path.resolve(runDir)) {
    return NextResponse.json({ ok: false, error: "deliverable path escaped the run" }, { status: 403 });
  }
  const body = await fs.readFile(target).catch(() => null);
  if (!body) {
    return NextResponse.json({ ok: false, error: "deliverable file is unavailable" }, { status: 404 });
  }
  const sha256 = crypto.createHash("sha256").update(body).digest("hex");
  if (body.byteLength !== record.bytes || sha256 !== record.sha256.toLowerCase()) {
    return NextResponse.json({ ok: false, error: "deliverable bytes do not match the run manifest" }, { status: 409 });
  }

  return new NextResponse(body, {
    headers: {
      "Cache-Control": "no-store, max-age=0",
      "Content-Disposition": `attachment; filename="${fileName}"`,
      "Content-Type": contentType,
      "X-Artifact-Bytes": String(body.byteLength),
      "X-Artifact-SHA256": sha256,
      "X-Content-Type-Options": "nosniff",
    },
  });
}
