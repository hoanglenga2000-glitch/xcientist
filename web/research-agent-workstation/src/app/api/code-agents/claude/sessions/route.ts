import { NextResponse } from "next/server";
import { promises as fs } from "node:fs";
import path from "node:path";
import { createClaudeSession } from "@/lib/server/claude-agent-sessions";
import { readJsonFile, resolveWorkspacePath } from "@/lib/server/paths";

export const dynamic = "force-dynamic";

function summarizeSession(payload: unknown, relativePath: string) {
  const record = payload && typeof payload === "object" && !Array.isArray(payload)
    ? payload as Record<string, unknown>
    : {};
  const deepseekCache = record.deepseek_cache && typeof record.deepseek_cache === "object" && !Array.isArray(record.deepseek_cache)
    ? record.deepseek_cache as Record<string, unknown>
    : null;
  const deepseekUsage = deepseekCache?.usage && typeof deepseekCache.usage === "object" && !Array.isArray(deepseekCache.usage)
    ? deepseekCache.usage as Record<string, unknown>
    : null;

  return {
    session_id: typeof record.session_id === "string" ? record.session_id : path.basename(path.dirname(relativePath)),
    task_id: typeof record.task_id === "string" ? record.task_id : null,
    status: typeof record.status === "string" ? record.status : "unknown",
    configured: typeof record.configured === "boolean" ? record.configured : null,
    provider: typeof record.provider === "string" ? record.provider : null,
    model: typeof record.model === "string" ? record.model : null,
    manifest_path: relativePath,
    transcript_path: typeof record.transcript_path === "string" ? record.transcript_path : null,
    patch_path: typeof record.patch_path === "string" ? record.patch_path : null,
    prompt_cache_hit_tokens: typeof record.prompt_cache_hit_tokens === "number" ? record.prompt_cache_hit_tokens : null,
    deepseek_cache_hit_ratio: typeof deepseekUsage?.cache_hit_ratio === "number" ? deepseekUsage.cache_hit_ratio : null,
    created_at: typeof record.created_at === "string" ? record.created_at : null,
    updated_at: typeof record.updated_at === "string" ? record.updated_at : null,
    error: typeof record.error === "string" ? record.error : null
  };
}

export async function GET() {
  const root = resolveWorkspacePath(path.join("workspace", "code_agent_sessions"));
  const entries = await fs.readdir(root, { withFileTypes: true }).catch(() => []);
  const candidates = await Promise.all(entries
    .filter((entry) => entry.isDirectory())
    .map(async (entry) => {
      const relativePath = path.join("workspace", "code_agent_sessions", entry.name, "session_manifest.json").replaceAll("\\", "/");
      const absolutePath = path.join(root, entry.name, "session_manifest.json");
      const [stat, payload] = await Promise.all([
        fs.stat(absolutePath).catch(() => null),
        readJsonFile(absolutePath)
      ]);
      return { ...summarizeSession(payload, relativePath), mtime_ms: stat?.mtimeMs ?? 0 };
    }));
  const sessions = candidates
    .filter((session) => session.mtime_ms > 0)
    .sort((a, b) => b.mtime_ms - a.mtime_ms)
    .slice(0, 50)
    .map(({ mtime_ms: _mtimeMs, ...session }) => session);
  return NextResponse.json({ ok: true, sessions });
}

export async function POST(request: Request) {
  const body = await request.json().catch(() => ({}));
  const taskId = String(body.task_id ?? "house_prices");
  const prompt = typeof body.prompt === "string" ? body.prompt : undefined;
  const model = typeof body.model === "string" ? body.model : undefined;
  const maxTurns = Number.isFinite(Number(body.max_turns)) ? Number(body.max_turns) : undefined;
  const timeoutSeconds = Number.isFinite(Number(body.timeout_seconds)) ? Number(body.timeout_seconds) : undefined;
  const result = await createClaudeSession({ taskId, prompt, model, maxTurns, timeoutSeconds });
  return NextResponse.json(result);
}
