import { randomUUID } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";
import { ProviderBoundaryError, providerHttpFailure, readBoundedProviderJson, safeProviderFailure } from "@/lib/security/provider-boundary";
import { logAction } from "@/lib/server/actions";
import { claudeApiKeyStatus, claudeApiKeyValue, deepSeekApiKeyStatus, deepSeekConfig, hasClaudeApiKey, hasDeepSeekApiKey } from "@/lib/server/capabilities";
import { attachDeepSeekCacheUsage, createDeepSeekCacheMessages, localResponseCacheUsage, readDeepSeekCachedResponse, recordDeepSeekCacheSession, writeDeepSeekCachedResponse, type DeepSeekCacheMetadata } from "@/lib/server/deepseek-cache";
import { latestExperimentPath, latestScoreGatedWorkstationRunPath, normalizeTaskId, readJsonFile, resolveWorkspacePath, workspaceRoot, writeJsonArtifact, writeTextArtifact } from "@/lib/server/paths";

type ClaudeSessionStatus = "not_configured" | "running" | "completed" | "failed" | "cancelled";

type CreateClaudeSessionInput = {
  taskId: string;
  prompt?: string;
  model?: string;
  maxTurns?: number;
  timeoutSeconds?: number;
  cacheOnly?: boolean;
  provider?: ClaudeSessionRecord["provider"];
};

export type ClaudeSessionRecord = {
  ok: true;
  configured: boolean;
  session_id: string;
  task_id: string;
  status: ClaudeSessionStatus;
  provider: "claude_agent_sdk" | "deepseek_code_agent";
  model: string;
  prompt_summary: string;
  transcript_path: string;
  manifest_path: string;
  patch_path: string | null;
  generated_code: string;
  patch_diff: string;
  usage?: Record<string, unknown>;
  prompt_cache_hit_tokens?: number | null;
  deepseek_cache?: DeepSeekCacheMetadata;
  missing_env?: string[];
  error?: string;
  created_at: string;
  updated_at: string;
};

const abortControllers = new Map<string, AbortController>();

const SESSION_ID_RE = /^(?:claude|deepseek_code)_(?:[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}|\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{3}Z_[a-z0-9]{6})$/;

export function normalizeClaudeSessionId(value: string) {
  const sessionId = String(value ?? "").trim();
  if (!SESSION_ID_RE.test(sessionId)) throw new Error("Invalid code-agent session ID");
  return sessionId;
}

function sessionDir(sessionId: string) {
  return resolveWorkspacePath(path.join("workspace", "code_agent_sessions", normalizeClaudeSessionId(sessionId)));
}

function sessionManifestRelative(sessionId: string) {
  return path.join("workspace", "code_agent_sessions", normalizeClaudeSessionId(sessionId), "session_manifest.json").replaceAll("\\", "/");
}

function transcriptRelative(sessionId: string) {
  return path.join("workspace", "code_agent_sessions", normalizeClaudeSessionId(sessionId), "transcript.jsonl").replaceAll("\\", "/");
}

function patchRelative(taskId: string, sessionId: string) {
  return path.join("workspace", "tasks", taskId, "code", "patches", `claude_agent_${normalizeClaudeSessionId(sessionId)}.diff`).replaceAll("\\", "/");
}

function extractDiff(text: string) {
  const fenced = text.match(/```(?:diff|patch)\s*([\s\S]*?)```/i);
  if (fenced?.[1]?.trim()) return fenced[1].trim();
  const lines = text.split(/\r?\n/);
  const start = lines.findIndex((line) => line.startsWith("diff --git ") || line.startsWith("--- "));
  if (start >= 0) return lines.slice(start).join("\n").trim();
  return "";
}

function normalizePatchText(text: string) {
  return text
    .replaceAll("\u2014", " - ")
    .replaceAll("\u2013", "-")
    .replaceAll("\u2018", "'")
    .replaceAll("\u2019", "'")
    .replaceAll("\u201c", '"')
    .replaceAll("\u201d", '"')
    .replaceAll("\u2026", "...");
}

async function readContext(taskId: string) {
  const latest = taskId === "playground_series_s6e6"
    ? (await latestScoreGatedWorkstationRunPath(taskId)) ?? await latestExperimentPath(taskId)
    : await latestExperimentPath(taskId);
  const validation = latest ? await readJsonFile(resolveWorkspacePath(path.join(latest, "validation_gate.json"))) : null;
  const experiment = latest ? await readJsonFile(resolveWorkspacePath(path.join(latest, "experiment_log.json"))) : null;
  const metrics = latest ? await readJsonFile(resolveWorkspacePath(path.join(latest, "hpc_gpu_training", "metrics.json"))) : null;
  const scoreGate = latest ? await readJsonFile(resolveWorkspacePath(path.join(latest, "score_improvement_gate.json"))) : null;
  const submissionAudit = latest ? await readJsonFile(resolveWorkspacePath(path.join(latest, "submission_audit.json"))) : null;
  const strategyRoot = resolveWorkspacePath("workspace/strategy");
  const strategyEntries = await fs.readdir(strategyRoot, { withFileTypes: true }).catch(() => []);
  const diagnosisCandidates = await Promise.all(strategyEntries
    .filter((entry) => entry.isFile() && entry.name.startsWith("s6e6_score_regression_diagnosis_") && entry.name.endsWith(".json"))
    .map(async (entry) => {
      const absolutePath = path.join(strategyRoot, entry.name);
      const stat = await fs.stat(absolutePath).catch(() => null);
      return { absolutePath, relativePath: path.join("workspace", "strategy", entry.name).replaceAll("\\", "/"), mtimeMs: stat?.mtimeMs ?? 0 };
    }));
  const latestDiagnosis = diagnosisCandidates.sort((a, b) => b.mtimeMs - a.mtimeMs)[0];
  const scoreRegressionDiagnosis = latestDiagnosis ? await readJsonFile(latestDiagnosis.absolutePath) : null;
  const patchRoot = resolveWorkspacePath(path.join("workspace", "tasks", taskId, "code", "patches"));
  const patchEntries = await fs.readdir(patchRoot, { withFileTypes: true }).catch(() => []);
  async function latestPatchJson(prefix: string) {
    const candidates = await Promise.all(patchEntries
      .filter((entry) => entry.isFile() && entry.name.startsWith(prefix) && entry.name.endsWith(".json"))
      .map(async (entry) => {
        const absolutePath = path.join(patchRoot, entry.name);
        const stat = await fs.stat(absolutePath).catch(() => null);
        return { absolutePath, relativePath: path.join("workspace", "tasks", taskId, "code", "patches", entry.name).replaceAll("\\", "/"), mtimeMs: stat?.mtimeMs ?? 0 };
      }));
    const latestPatch = candidates.sort((a, b) => b.mtimeMs - a.mtimeMs)[0];
    const payload = latestPatch ? await readJsonFile(latestPatch.absolutePath) : null;
    return payload ? { ...(payload as Record<string, unknown>), artifact_path: latestPatch?.relativePath } : null;
  }
  const [latestPatchReview, latestCodeQualityCheck] = await Promise.all([
    latestPatchJson("patch_review_"),
    latestPatchJson("code_quality_check_")
  ]);
  const reportCandidates = latest
    ? [
        resolveWorkspacePath(path.join(latest, "research_report.md")),
        resolveWorkspacePath(path.join(latest, taskId === "titanic" ? "titanic_local_report.md" : "local_report.md"))
      ]
    : [];
  let report = "";
  for (const reportPath of reportCandidates) {
    report = await fs.readFile(reportPath, "utf-8").catch(() => "");
    if (report) break;
  }
  return {
    latest,
    validation,
    experiment,
    metrics,
    score_gate: scoreGate,
    submission_audit: submissionAudit,
    score_regression_diagnosis: scoreRegressionDiagnosis ? { ...(scoreRegressionDiagnosis as Record<string, unknown>), artifact_path: latestDiagnosis?.relativePath } : null,
    latest_patch_review: latestPatchReview,
    latest_code_quality_check: latestCodeQualityCheck,
    report_excerpt: report.slice(0, 6000)
  };
}

function buildPrompt(taskId: string, context: Awaited<ReturnType<typeof readContext>>, userPrompt?: string) {
  const s6e6AllowedTargets = [
    "web/research-agent-workstation/src/lib/server/workstation-closed-loop.ts",
    "web/research-agent-workstation/src/lib/server/workstation-actions.ts",
    "web/research-agent-workstation/src/lib/server/strategy-registry.ts",
    "web/research-agent-workstation/src/lib/server/summary.ts",
    "web/research-agent-workstation/src/components/workstation/Screens.tsx",
    "scripts/verify_s6e6_strategy_gate.py",
    "configs/playground_series_s6e6*.yaml",
    "workspace/tasks/playground_series_s6e6/code/**/*.py",
    "workspace/tasks/playground_series_s6e6/code/**/*.json",
    "workspace/tasks/playground_series_s6e6/code/**/*.md"
  ];
  return [
    "You are a code-writing agent inside the Research Agent Workstation.",
    "Return a reviewable implementation suggestion only. Do not claim that files were applied.",
    "The workstation will import your diff through a code quality gate and a human manual gate.",
    "Prefer concise, practical changes to the tabular research pipeline, report generation, or validation logic.",
    "Your final answer must include a unified diff inside a ```diff fenced block if a code change is useful.",
    "Use plain ASCII punctuation inside generated diffs so patch artifacts remain readable in Windows, WSL, and CI logs.",
    taskId === "playground_series_s6e6"
      ? "For S6E6, do not propose a direct Kaggle submit or direct Codex training. Produce only workstation-controlled code/manifest improvements that must pass Code Quality Gate, HPC Execution Gate, score improvement gate, and Human Submission Gate."
      : "",
    taskId === "playground_series_s6e6"
      ? [
          "S6E6 path discipline:",
          "- Only patch existing workstation-controlled files or task code artifacts.",
          "- Do not invent paths such as workstation/gates/score_gate.py.",
          "- If you need score-gate logic, patch the existing TypeScript gate in web/research-agent-workstation/src/lib/server/workstation-closed-loop.ts.",
          "- If you need action/review logic, patch web/research-agent-workstation/src/lib/server/workstation-actions.ts.",
          "- If you need visible Code Agent evidence, patch web/research-agent-workstation/src/lib/server/summary.ts or web/research-agent-workstation/src/components/workstation/Screens.tsx.",
          "- If you need training-template changes, put them under workspace/tasks/playground_series_s6e6/code/ and keep them gated; do not run them.",
          `Allowed target examples: ${s6e6AllowedTargets.join("; ")}`
        ].join("\n")
      : "",
    "",
    `Task ID: ${taskId}`,
    `Latest experiment: ${context.latest ?? "none"}`,
    `User request: ${userPrompt ?? "Improve this task using the latest experiment evidence."}`,
    "",
    "Validation evidence:",
    stableJsonForPrompt(context.validation, 5000),
    "",
    "Experiment evidence:",
    stableJsonForPrompt(context.experiment, 5000),
    "",
    "HPC/workstation metrics evidence:",
    stableJsonForPrompt(context.metrics, 5000),
    "",
    "Score improvement gate evidence:",
    stableJsonForPrompt(context.score_gate, 5000),
    "",
    "Submission audit evidence:",
    stableJsonForPrompt(context.submission_audit, 3000),
    "",
    "Score regression diagnosis and next-agent work order:",
    stableJsonForPrompt(context.score_regression_diagnosis, 6000),
    "",
    "Latest failed/passed patch review evidence to learn from:",
    stableJsonForPrompt(context.latest_patch_review, 3000),
    "",
    "Latest code quality gate evidence:",
    stableJsonForPrompt(context.latest_code_quality_check, 3000),
    "",
    "Report excerpt:",
    context.report_excerpt || "(no report excerpt available)"
  ].filter(Boolean).join("\n");
}

function hasAnyCodeAgent() {
  return hasClaudeApiKey() || hasDeepSeekApiKey();
}

function selectedProvider(preferredProvider?: ClaudeSessionRecord["provider"]): ClaudeSessionRecord["provider"] {
  if (preferredProvider === "deepseek_code_agent" && hasDeepSeekApiKey()) return "deepseek_code_agent";
  if (preferredProvider === "claude_agent_sdk" && hasClaudeApiKey()) return "claude_agent_sdk";
  return hasClaudeApiKey() ? "claude_agent_sdk" : "deepseek_code_agent";
}

function selectedModel(provider: ClaudeSessionRecord["provider"], inputModel?: string) {
  if (provider === "claude_agent_sdk") return inputModel || process.env.CLAUDE_CODE_MODEL || "sonnet";
  return inputModel || deepSeekConfig().model;
}

function promptCacheHitTokens(usage: unknown) {
  if (!usage || typeof usage !== "object" || Array.isArray(usage)) return null;
  const record = usage as Record<string, unknown>;
  if (typeof record.prompt_cache_hit_tokens === "number") return record.prompt_cache_hit_tokens;
  const details = record.prompt_tokens_details;
  if (details && typeof details === "object" && !Array.isArray(details)) {
    const cached = (details as Record<string, unknown>).cached_tokens;
    if (typeof cached === "number") return cached;
  }
  return null;
}

const VOLATILE_PROMPT_KEYS = new Set([
  "at",
  "created_at",
  "updated_at",
  "generated_at",
  "started_at",
  "finished_at",
  "completed_at",
  "decided_at",
  "submitted_at",
  "saved_at",
  "timestamp",
  "mtime",
  "mtimeMs",
  "session_id",
  "transcript_path",
  "manifest_path",
  "cache_entry_path"
]);

function normalizePromptString(value: string) {
  return value
    .replaceAll(workspaceRoot, "$WORKSPACE_ROOT")
    .replaceAll("\\", "/")
    .replace(/\b\d{4}-\d{2}-\d{2}T\d{2}[:.]\d{2}[:.]\d{2}(?:[.\-]\d+)?Z?\b/g, "$TIMESTAMP");
}

function stablePromptValue(value: unknown): unknown {
  if (value === null || typeof value === "number" || typeof value === "boolean") return value;
  if (typeof value === "string") return normalizePromptString(value);
  if (Array.isArray(value)) return value.map(stablePromptValue);
  if (typeof value !== "object") return value;

  const record = value as Record<string, unknown>;
  const sorted: Record<string, unknown> = {};
  for (const key of Object.keys(record).sort()) {
    if (VOLATILE_PROMPT_KEYS.has(key)) continue;
    sorted[key] = stablePromptValue(record[key]);
  }
  return sorted;
}

function stableJsonForPrompt(value: unknown, limit: number) {
  return JSON.stringify(stablePromptValue(value ?? {}), null, 2).slice(0, limit);
}

async function writeRecord(record: ClaudeSessionRecord) {
  await writeJsonArtifact(sessionManifestRelative(record.session_id), record);
  return record;
}

export async function readClaudeSession(sessionId: string) {
  return readJsonFile(resolveWorkspacePath(sessionManifestRelative(sessionId))) as Promise<ClaudeSessionRecord | null>;
}

export async function cancelClaudeSession(sessionId: string) {
  const normalizedSessionId = normalizeClaudeSessionId(sessionId);
  const controller = abortControllers.get(normalizedSessionId);
  if (controller) controller.abort();
  const current = await readClaudeSession(normalizedSessionId);
  if (!current) return null;
  const updated: ClaudeSessionRecord = { ...current, status: "cancelled", updated_at: new Date().toISOString() };
  await writeRecord(updated);
  await logAction({
    action: "cancel_claude_agent_session",
    taskId: updated.task_id,
    message: `Claude Code session cancelled: ${sessionId}`,
    artifactPath: updated.manifest_path,
    metadata: { session_id: sessionId }
  });
  return updated;
}

export async function probeDeepSeekCodeCache(input: { taskId: string; prompt?: string; model?: string }) {
  const taskId = normalizeTaskId(input.taskId);
  const context = await readContext(taskId);
  const prompt = buildPrompt(taskId, context, input.prompt);
  const cachedPrompt = createDeepSeekCacheMessages(prompt);
  const config = deepSeekConfig();
  const requestedModel = input.model || config.model;
  const localCachedResponse = await readDeepSeekCachedResponse({
    promptFingerprint: cachedPrompt.metadata.prompt_fingerprint,
    cacheKey: cachedPrompt.metadata.cache_key,
    model: requestedModel
  });

  return {
    ok: true,
    configured: hasDeepSeekApiKey(),
    task_id: taskId,
    source_agent: "deepseek_code_agent",
    cli_status: "cache_probe",
    model: requestedModel,
    prompt_fingerprint: cachedPrompt.metadata.prompt_fingerprint,
    cache_key: cachedPrompt.metadata.cache_key,
    local_response_cache_hit: Boolean(localCachedResponse),
    cache_entry_path: localCachedResponse?.relativePath ?? null,
    external_model_calls_allowed: false,
    prompt_chars: prompt.length,
    stable_prefix_chars: cachedPrompt.metadata.stable_prefix_chars,
    dynamic_suffix_chars: cachedPrompt.metadata.dynamic_suffix_chars,
    missing_env: hasDeepSeekApiKey() ? undefined : ["DEEPSEEK_API_KEY"]
  };
}

export async function createClaudeSession(input: CreateClaudeSessionInput) {
  const taskId = normalizeTaskId(input.taskId);
  const provider = selectedProvider(input.provider);
  const sessionId = `${provider === "claude_agent_sdk" ? "claude" : "deepseek_code"}_${randomUUID()}`;
  const now = new Date().toISOString();
  const model = selectedModel(provider, input.model);
  await fs.mkdir(sessionDir(sessionId), { recursive: true });

  const baseRecord: ClaudeSessionRecord = {
    ok: true,
    configured: hasAnyCodeAgent(),
    session_id: sessionId,
    task_id: taskId,
    status: hasAnyCodeAgent() ? "running" : "not_configured",
    provider,
    model,
    prompt_summary: input.prompt ?? "Generate a gated coding-agent patch from current task evidence.",
    transcript_path: transcriptRelative(sessionId),
    manifest_path: sessionManifestRelative(sessionId),
    patch_path: null,
    generated_code: "",
    patch_diff: "",
    missing_env: hasAnyCodeAgent() ? undefined : ["ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY"],
    created_at: now,
    updated_at: now
  };

  if (!hasAnyCodeAgent()) {
    await writeRecord(baseRecord);
    await logAction({
      action: "claude_agent_session_not_configured",
      taskId,
      message: "Code Agent is not configured. Set ANTHROPIC_API_KEY or DEEPSEEK_API_KEY before starting a real Coding Agent session.",
      artifactPath: baseRecord.manifest_path,
      metadata: { provider, claude_api_key_status: claudeApiKeyStatus(), deepseek_api_key_status: deepSeekApiKeyStatus(), missing_env: baseRecord.missing_env }
    });
    return baseRecord;
  }

  if (provider === "deepseek_code_agent") {
    return createDeepSeekCodeSession(baseRecord, input);
  }

  await writeRecord(baseRecord);
  const controller = new AbortController();
  abortControllers.set(sessionId, controller);
  const timeout = setTimeout(() => controller.abort(), Math.max(10, input.timeoutSeconds ?? 120) * 1000);

  try {
    const context = await readContext(taskId);
    const prompt = buildPrompt(taskId, context, input.prompt);
    const apiKey = claudeApiKeyValue();
    if (apiKey && !process.env.ANTHROPIC_API_KEY) process.env.ANTHROPIC_API_KEY = apiKey;
    const { query } = await import("@anthropic-ai/claude-agent-sdk");
    const transcriptLines: string[] = [];
    const textParts: string[] = [];
    const iterable = query({
      prompt,
      options: {
        abortController: controller,
        cwd: workspaceRoot,
        model,
        maxTurns: Math.max(1, Math.min(input.maxTurns ?? Number(process.env.CLAUDE_CODE_MAX_TURNS ?? 5), 8)),
        permissionMode: "dontAsk",
        tools: ["Read", "Grep", "Glob"],
        disallowedTools: ["Edit", "MultiEdit", "Write", "Bash", "NotebookEdit"],
        persistSession: false,
        env: { ...process.env, ANTHROPIC_API_KEY: process.env.ANTHROPIC_API_KEY || apiKey, CLAUDE_AGENT_SDK_CLIENT_APP: "research-agent-workstation/0.1.0" }
      }
    });

    for await (const message of iterable) {
      transcriptLines.push(JSON.stringify(message));
      const maybeMessage = (message as { message?: { content?: Array<{ type?: string; text?: string }> } }).message;
      for (const block of maybeMessage?.content ?? []) {
        if (block.type === "text" && block.text) textParts.push(block.text);
      }
    }

    const generatedText = textParts.join("\n\n").trim();
    const patchDiff = normalizePatchText(extractDiff(generatedText));
    const transcriptPath = await writeTextArtifact(transcriptRelative(sessionId), `${transcriptLines.join("\n")}\n`);
    let patchPath: string | null = null;
    if (patchDiff) {
      patchPath = patchRelative(taskId, sessionId);
      await writeTextArtifact(patchPath, patchDiff);
    }

    const completed: ClaudeSessionRecord = {
      ...baseRecord,
      configured: true,
      status: "completed",
      transcript_path: transcriptPath,
      patch_path: patchPath,
      generated_code: generatedText,
      patch_diff: patchDiff,
      missing_env: undefined,
      updated_at: new Date().toISOString()
    };
    await writeRecord(completed);
    await logAction({
      action: "claude_agent_session_completed",
      taskId,
      message: patchPath ? "Claude Code session completed and produced a reviewable patch." : "Claude Code session completed; no patch was produced.",
      artifactPath: completed.manifest_path,
      metadata: { session_id: sessionId, model, patch_path: patchPath, transcript_path: transcriptPath }
    });
    return completed;
  } catch (error) {
    const failed: ClaudeSessionRecord = {
      ...baseRecord,
      configured: true,
      status: controller.signal.aborted ? "cancelled" : "failed",
      error: error instanceof Error ? error.message : "Claude Agent SDK session failed.",
      updated_at: new Date().toISOString()
    };
    await writeRecord(failed);
    await logAction({
      action: failed.status === "cancelled" ? "claude_agent_session_cancelled" : "claude_agent_session_failed",
      taskId,
      message: failed.error ?? "Claude Agent SDK session failed.",
      artifactPath: failed.manifest_path,
      metadata: { session_id: sessionId, model, status: failed.status }
    });
    return failed;
  } finally {
    clearTimeout(timeout);
    abortControllers.delete(sessionId);
  }
}

async function createDeepSeekCodeSession(baseRecord: ClaudeSessionRecord, input: CreateClaudeSessionInput) {
  const taskId = baseRecord.task_id;
  await writeRecord(baseRecord);
  const attemptLog: Array<Record<string, unknown>> = [];
  let cacheMetadata: DeepSeekCacheMetadata | undefined;
  try {
    const context = await readContext(taskId);
    const prompt = buildPrompt(taskId, context, input.prompt);
    const cachedPrompt = createDeepSeekCacheMessages(prompt);
    const messages = cachedPrompt.messages;
    cacheMetadata = cachedPrompt.metadata;
    const config = deepSeekConfig();
    const requestedModel = baseRecord.model || config.model;
    const localCachedResponse = await readDeepSeekCachedResponse({
      promptFingerprint: cacheMetadata.prompt_fingerprint,
      cacheKey: cacheMetadata.cache_key,
      model: requestedModel
    });
    if (localCachedResponse) {
      const generatedText = localCachedResponse.record.generated_text.trim();
      const patchDiff = normalizePatchText(localCachedResponse.record.patch_diff);
      const localUsage = localResponseCacheUsage(prompt.length, generatedText.length);
      const deepseekCache: DeepSeekCacheMetadata = {
        ...cacheMetadata,
        usage: localUsage
      };
      const transcriptPath = await writeTextArtifact(transcriptRelative(baseRecord.session_id), `${JSON.stringify({
        provider: "deepseek_code_agent",
        model: requestedModel,
        created_at: new Date().toISOString(),
        prompt_chars: prompt.length,
        response_chars: generatedText.length,
        deepseek_cache: deepseekCache,
        local_response_cache: {
          hit: true,
          cache_entry_path: localCachedResponse.relativePath
        },
        usage: deepseekCache.usage,
        attempts: [{ attempt: 0, status: "local_cache_hit", created_at: new Date().toISOString() }]
      })}\n`);
      let patchPath: string | null = null;
      if (patchDiff) {
        patchPath = patchRelative(taskId, baseRecord.session_id);
        await writeTextArtifact(patchPath, patchDiff);
      }
      const completed: ClaudeSessionRecord = {
        ...baseRecord,
        configured: true,
        status: "completed",
        model: requestedModel,
        transcript_path: transcriptPath,
        patch_path: patchPath,
        generated_code: generatedText,
        patch_diff: patchDiff,
        usage: localUsage as unknown as Record<string, unknown>,
        prompt_cache_hit_tokens: localUsage.cached_tokens,
        deepseek_cache: deepseekCache,
        missing_env: undefined,
        updated_at: new Date().toISOString()
      };
      await writeRecord(completed);
      await recordDeepSeekCacheSession({
        sessionId: baseRecord.session_id,
        taskId,
        model: completed.model,
        status: "completed_local_cache_hit",
        transcriptPath,
        manifestPath: completed.manifest_path,
        metadata: deepseekCache
      });
      await logAction({
        action: "deepseek_code_agent_session_local_cache_hit",
        taskId,
        message: "DeepSeek Code Agent reused a local exact-response cache entry; no external model call was needed.",
        artifactPath: completed.manifest_path,
        metadata: {
          session_id: baseRecord.session_id,
          model: completed.model,
          patch_path: patchPath,
          transcript_path: transcriptPath,
          cache_entry_path: localCachedResponse.relativePath,
          cache_hit_ratio: localUsage.cache_hit_ratio
        }
      });
      return completed;
    }
    if (input.cacheOnly) {
      const failureCacheMetadata = attachDeepSeekCacheUsage(cacheMetadata, undefined);
      const transcriptPath = await writeTextArtifact(transcriptRelative(baseRecord.session_id), `${JSON.stringify({
        provider: "deepseek_code_agent",
        model: requestedModel,
        created_at: new Date().toISOString(),
        prompt_chars: prompt.length,
        deepseek_cache: failureCacheMetadata,
        local_response_cache: {
          hit: false,
          external_model_call_blocked: true
        },
        final_status: "cache_only_miss_blocked",
        error: "DeepSeek cache-only session missed local cache; external model call was blocked."
      })}\n`);
      const failed: ClaudeSessionRecord = {
        ...baseRecord,
        configured: true,
        status: "failed",
        model: requestedModel,
        transcript_path: transcriptPath,
        deepseek_cache: failureCacheMetadata,
        error: "DeepSeek cache-only session missed local cache; external model call was blocked.",
        updated_at: new Date().toISOString()
      };
      await writeRecord(failed);
      await recordDeepSeekCacheSession({
        sessionId: baseRecord.session_id,
        taskId,
        model: requestedModel,
        status: "cache_only_miss_blocked",
        transcriptPath,
        manifestPath: failed.manifest_path,
        metadata: failureCacheMetadata
      });
      await logAction({
        action: "deepseek_code_agent_cache_only_miss",
        taskId,
        message: "DeepSeek Code Agent cache-only mode missed local response cache; external model call was blocked.",
        artifactPath: failed.manifest_path,
        metadata: {
          session_id: baseRecord.session_id,
          model: requestedModel,
          transcript_path: transcriptPath,
          cache_key: failureCacheMetadata.cache_key,
          prompt_fingerprint: failureCacheMetadata.prompt_fingerprint
        }
      });
      return failed;
    }
    if (config.boundaryError || !config.chatCompletionsUrl) {
      throw new ProviderBoundaryError(config.boundaryError ?? "provider_endpoint_invalid");
    }
    const maxAttempts = Math.max(1, Math.min(Number(input.maxTurns ?? 2), 3));
    let payload: Record<string, unknown> = {};
    let lastError: ProviderBoundaryError | null = null;
    for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
      try {
        const response = await fetch(config.chatCompletionsUrl, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${config.apiKey}`,
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            model: requestedModel,
            messages,
            stream: false
          }),
          redirect: "error",
          signal: AbortSignal.timeout(Math.max(10, input.timeoutSeconds ?? 120) * 1000)
        });
        const responsePayload = await readBoundedProviderJson(response);
        if (!response.ok) {
          throw new ProviderBoundaryError(providerHttpFailure("deepseek_code_agent", response));
        }
        payload = responsePayload;
        attemptLog.push({ attempt, status: "passed", created_at: new Date().toISOString() });
        lastError = null;
        break;
      } catch (error) {
        const failureCode = error instanceof ProviderBoundaryError
          ? error.code
          : safeProviderFailure(error, "deepseek_code_agent_request_failed");
        lastError = new ProviderBoundaryError(failureCode);
        attemptLog.push({ attempt, status: "failed", error: failureCode, created_at: new Date().toISOString() });
        if (attempt < maxAttempts) {
          await new Promise((resolve) => setTimeout(resolve, 750 * attempt));
        }
      }
    }
    if (lastError) {
      throw lastError;
    }
    const choices = payload.choices as Array<{ message?: { content?: string } }> | undefined;
    const generatedText = choices?.[0]?.message?.content?.trim() ?? "";
    const patchDiff = normalizePatchText(extractDiff(generatedText));
    const deepseekCache = attachDeepSeekCacheUsage(cacheMetadata, payload.usage);
    const transcriptPath = await writeTextArtifact(transcriptRelative(baseRecord.session_id), `${JSON.stringify({
      provider: "deepseek_code_agent",
      model: String(payload.model ?? requestedModel),
      created_at: new Date().toISOString(),
      prompt_chars: prompt.length,
      response_chars: generatedText.length,
      deepseek_cache: deepseekCache,
      usage: payload.usage,
      attempts: attemptLog
    })}\n`);
    let patchPath: string | null = null;
    if (patchDiff) {
      patchPath = patchRelative(taskId, baseRecord.session_id);
      await writeTextArtifact(patchPath, patchDiff);
    }
    const cacheEntryPath = await writeDeepSeekCachedResponse({
      promptFingerprint: deepseekCache.prompt_fingerprint,
      cacheKey: deepseekCache.cache_key,
      model: String(payload.model ?? requestedModel),
      generatedText,
      patchDiff,
      usage: (payload.usage && typeof payload.usage === "object" && !Array.isArray(payload.usage)) ? payload.usage as Record<string, unknown> : undefined
    });
    const completed: ClaudeSessionRecord = {
      ...baseRecord,
      configured: true,
      status: "completed",
      model: String(payload.model ?? requestedModel),
      transcript_path: transcriptPath,
      patch_path: patchPath,
      generated_code: generatedText,
      patch_diff: patchDiff,
      usage: (payload.usage && typeof payload.usage === "object" && !Array.isArray(payload.usage)) ? payload.usage as Record<string, unknown> : undefined,
      prompt_cache_hit_tokens: deepseekCache.usage?.cached_tokens ?? promptCacheHitTokens(payload.usage),
      deepseek_cache: deepseekCache,
      missing_env: undefined,
      updated_at: new Date().toISOString()
    };
    await writeRecord(completed);
    await recordDeepSeekCacheSession({
      sessionId: baseRecord.session_id,
      taskId,
      model: completed.model,
      status: completed.status,
      transcriptPath,
      manifestPath: completed.manifest_path,
      metadata: deepseekCache
    });
    await logAction({
      action: "deepseek_code_agent_session_completed",
      taskId,
      message: patchPath ? "DeepSeek Code Agent completed and produced a reviewable patch." : "DeepSeek Code Agent completed; no patch was produced.",
      artifactPath: completed.manifest_path,
      metadata: {
        session_id: baseRecord.session_id,
        model: completed.model,
        patch_path: patchPath,
        transcript_path: transcriptPath,
        deepseek_cache: {
          cache_key: deepseekCache.cache_key,
          prompt_fingerprint: deepseekCache.prompt_fingerprint,
          cache_entry_path: cacheEntryPath,
          cached_tokens: deepseekCache.usage?.cached_tokens ?? null,
          cache_hit_ratio: deepseekCache.usage?.cache_hit_ratio ?? null
        }
      }
    });
    return completed;
  } catch (error) {
    const failureCode = safeProviderFailure(error, "deepseek_code_agent_session_failed");
    const failureCacheMetadata = typeof cacheMetadata !== "undefined"
      ? attachDeepSeekCacheUsage(cacheMetadata, undefined)
      : undefined;
    const transcriptPath = await writeTextArtifact(transcriptRelative(baseRecord.session_id), `${JSON.stringify({
      provider: "deepseek_code_agent",
      model: baseRecord.model,
      created_at: new Date().toISOString(),
      attempts: attemptLog,
      deepseek_cache: failureCacheMetadata,
      final_status: "failed",
      error: failureCode
    })}\n`);
    const failed: ClaudeSessionRecord = {
      ...baseRecord,
      configured: true,
      status: "failed",
      transcript_path: transcriptPath,
      deepseek_cache: failureCacheMetadata,
      error: failureCode,
      updated_at: new Date().toISOString()
    };
    await writeRecord(failed);
    if (failureCacheMetadata) {
      await recordDeepSeekCacheSession({
        sessionId: baseRecord.session_id,
        taskId,
        model: baseRecord.model,
        status: failed.status,
        transcriptPath,
        manifestPath: failed.manifest_path,
        metadata: failureCacheMetadata
      });
    }
    await logAction({
      action: "deepseek_code_agent_session_failed",
      taskId,
      message: failed.error ?? "DeepSeek Code Agent session failed.",
      artifactPath: failed.manifest_path,
      metadata: { session_id: baseRecord.session_id, model: baseRecord.model, status: failed.status }
    });
    return failed;
  }
}

export function sessionToDraftPayload(record: ClaudeSessionRecord) {
  const draftPath = record.manifest_path;
  return {
    ok: true,
    task_id: record.task_id,
    source_agent: record.provider === "deepseek_code_agent" ? "deepseek_code_agent" : "claude_code",
    draft_path: draftPath,
    patch_path: record.patch_path,
    manifest_path: record.manifest_path,
    generated_code: record.generated_code || (record.configured ? "Code Agent session completed without generated text." : "Code Agent is not configured. Set ANTHROPIC_API_KEY or DEEPSEEK_API_KEY to enable real execution."),
    patch_diff: record.patch_diff,
    cli_status: record.status,
    configured: record.configured,
    session_id: record.session_id,
    transcript_path: record.transcript_path,
    usage: record.usage,
    prompt_cache_hit_tokens: record.prompt_cache_hit_tokens,
    deepseek_cache: record.deepseek_cache,
    missing_env: record.missing_env,
    error: record.error
  };
}
