import { NextResponse } from "next/server";
import { runtimeVersionIdentity } from "@/lib/server/runtime-version";
import { verifiedOperatorAccess } from "@/lib/server/legacy-gpu-access";
import { assistantRuntimeFetch } from "@/lib/server/assistant-runtime";

export const dynamic = "force-dynamic";

// t238 C7b O2: `?deep=1` is operator-only (same boundary as the legacy operator routes). A tenant or
// anonymous caller gets 401/403 with no deep data; the plain response below is unchanged.
// H1: a forged `x-evomind-local-automation-verified` alone is never enough (see verifiedOperatorAccess).
const DEEP_TIMEOUT_MS = 900;

async function deepHealth(request: Request) {
  const access = verifiedOperatorAccess(request);
  if (!access.ok) {
    return NextResponse.json({ ok: false, code: access.status === 401 ? "session_required" : "operator_required" },
      { status: access.status, headers: { "Cache-Control": "no-store" } });
  }
  try {
    const response = await assistantRuntimeFetch("/v1/health/deep", { signal: AbortSignal.timeout(DEEP_TIMEOUT_MS) });
    const body = await response.json();
    return NextResponse.json({ ok: body.status === "ok", ...body }, { status: response.ok ? 200 : 503, headers: { "Cache-Control": "no-store" } });
  } catch {
    return NextResponse.json({ ok: false, status: "degraded", failed: ["runtime"], components: { runtime: { ok: false, error: "unreachable" } } },
      { status: 503, headers: { "Cache-Control": "no-store" } });
  }
}

export async function GET(request: Request) {
  if (request && new URL(request.url).searchParams.get("deep") === "1") return deepHealth(request);
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
