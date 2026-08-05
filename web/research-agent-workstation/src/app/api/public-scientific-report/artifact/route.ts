import { NextResponse } from "next/server";
import { readPublicArtifactBody, resolvePublicScientificArtifact } from "@/lib/server/public-scientific-report";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const taskId = url.searchParams.get("task_id") ?? "";
  const runId = url.searchParams.get("run_id") ?? "";
  const artifactId = url.searchParams.get("artifact_id") ?? "";
  try {
    const artifact = await resolvePublicScientificArtifact(taskId, runId, artifactId);
    const { body, bytes, sha256 } = await readPublicArtifactBody(artifact);
    const disposition = url.searchParams.get("download") === "1" ? "attachment" : "inline";
    return new NextResponse(body, {
      headers: {
        "Cache-Control": "no-store, max-age=0",
        "Content-Disposition": `${disposition}; filename="${artifact.filename.replaceAll('"', "")}"`,
        "Content-Security-Policy": "default-src 'none'; script-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'; frame-ancestors 'self'",
        "X-Frame-Options": "SAMEORIGIN",
        "Content-Type": artifact.contentType,
        "X-Artifact-Bytes": String(bytes),
        "X-Artifact-SHA256": sha256,
        "X-Content-Type-Options": "nosniff",
      },
    });
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Public artifact unavailable." },
      { status: 404 },
    );
  }
}
