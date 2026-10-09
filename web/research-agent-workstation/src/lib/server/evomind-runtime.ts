import fs from "node:fs";
import path from "node:path";
import { runtimeRoot } from "@/lib/server/paths";

const port = Number(process.env.EVOMIND_RUNTIME_PORT ?? "8765");
const baseUrl = `http://127.0.0.1:${port}`;
const tokenPath = path.join(runtimeRoot, "runtime.token");

export type EvoMindRuntimeHealth = {
  reachable: boolean;
  status?: string;
  backend_version?: string;
  commit_hash?: string;
  source_tree_sha256?: string;
};

function token() {
  return fs.readFileSync(tokenPath, "ascii").trim();
}

export async function runtimeHealth(): Promise<EvoMindRuntimeHealth> {
  if (!fs.existsSync(tokenPath)) return { reachable: false };
  try {
    const response = await fetch(`${baseUrl}/v1/health`, {
      headers: { Authorization: `Bearer ${token()}` },
      cache: "no-store",
      signal: AbortSignal.timeout(2_000),
    });
    if (!response.ok) return { reachable: false };
    const payload = await response.json() as Record<string, unknown>;
    return {
      reachable: payload.status === "ready",
      status: typeof payload.status === "string" ? payload.status : undefined,
      backend_version: typeof payload.backend_version === "string" ? payload.backend_version : undefined,
      commit_hash: typeof payload.commit_hash === "string" ? payload.commit_hash : undefined,
      source_tree_sha256: typeof payload.source_tree_sha256 === "string" ? payload.source_tree_sha256 : undefined,
    };
  } catch {
    return { reachable: false };
  }
}

export async function ensureEvoMindRuntime() {
  if ((await runtimeHealth()).reachable) return { baseUrl, token: token() };
  // Runtime process creation and recovery belong exclusively to the signed
  // lifecycle manager, which records PID, creation time, executable,
  // command-line, cwd, listener and release nonce.  An API request must never
  // create an untracked replacement after a crash.
  throw new Error("EvoMind Runtime is not managed and ready");
}
