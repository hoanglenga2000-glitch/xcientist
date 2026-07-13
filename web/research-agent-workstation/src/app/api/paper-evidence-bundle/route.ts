import { NextResponse } from "next/server";
import { generatePaperEvidenceBundle } from "@/lib/server/runs";
import { readJsonFile, resolveWorkspacePath } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

export async function GET() {
  const bundlePath = "workspace/paper_evidence_bundle_20260623.json";
  const payload = await readJsonFile(resolveWorkspacePath(bundlePath));
  if (!payload) {
    return NextResponse.json({ ok: false, error: "Paper evidence bundle not generated yet.", bundle_path: bundlePath }, { status: 404 });
  }
  return NextResponse.json({ ok: true, bundle_path: bundlePath, bundle: payload });
}

export async function POST() {
  try {
    return NextResponse.json(await generatePaperEvidenceBundle());
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown paper evidence bundle error";
    return NextResponse.json({ ok: false, error: message }, { status: 500 });
  }
}
