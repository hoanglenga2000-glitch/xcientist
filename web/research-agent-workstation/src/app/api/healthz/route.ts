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
      // `commit_hash` is the base commit the deployed baseline was built from
      // (kept for deploy-script compatibility). The fields below say honestly
      // whether the running tree differs from it and where the changes came from.
      commit_hash: identity.commit_hash,
      base_commit: identity.commit_hash,
      source_dirty: identity.source_dirty,
      source_commit: identity.source_commit,
      source_commit_dirty: identity.source_commit_dirty,
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
