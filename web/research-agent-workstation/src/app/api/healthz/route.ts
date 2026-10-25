import { NextResponse } from "next/server";
import { runtimeVersionIdentity } from "@/lib/server/runtime-version";

export const dynamic = "force-dynamic";

export async function GET() {
  const identity = await runtimeVersionIdentity();
  return NextResponse.json(
    {
      ok: identity.ready,
      status: identity.status,
      service: "evomind-workstation",
      version: identity.frontend_version ?? "unknown",
      // `commit_hash` stays the deployed baseline. `source_revision` is the
      // commit recorded on this build. A dirty tree is not reported as clean.
      commit_hash: identity.commit_hash,
      base_commit: identity.commit_hash,
      source_revision: identity.source_revision,
      source_commit: identity.source_commit,
      source_tree_sha256: identity.source_tree_sha256,
      source_provenance: identity.source_provenance,
      build_id: identity.build_id,
      failures: identity.failures,
    },
    {
      status: identity.ready ? 200 : 503,
      headers: { "Cache-Control": "no-store" },
    },
  );
}
