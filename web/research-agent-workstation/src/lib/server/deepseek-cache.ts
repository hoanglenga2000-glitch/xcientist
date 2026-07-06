import { createHash } from "node:crypto";
import path from "node:path";
import { readJsonFile, resolveWorkspacePath, writeJsonArtifact } from "@/lib/server/paths";

type DeepSeekMessage = {
  role: "system" | "user";
  content: string;
};

export type DeepSeekCacheUsage = {
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
  cached_tokens: number | null;
  cache_hit_ratio: number | null;
  cache_observed: boolean;
  local_response_cache_hit?: boolean;
};

export type DeepSeekCacheMetadata = {
  strategy_version: "deepseek_prompt_cache_v2";
  cache_agent: "DeepSeekCacheOptimizationAgent";
  cache_policy: string;
  provider_cache_mode: "local_exact_response_cache_plus_automatic_stable_prefix";
  target_cache_hit_ratio: number;
  stable_system_hash: string;
  stable_user_prefix_hash: string;
  dynamic_suffix_hash: string;
  prompt_fingerprint: string;
  stable_prefix_chars: number;
  dynamic_suffix_chars: number;
  cache_key: string;
  usage?: DeepSeekCacheUsage;
};

export type DeepSeekCachedResponse = {
  ok: true;
  artifact_type: "deepseek_code_agent_response_cache_entry";
  strategy_version: typeof STRATEGY_VERSION;
  prompt_fingerprint: string;
  cache_key: string;
  model: string;
  generated_text: string;
  patch_diff: string;
  usage?: Record<string, unknown>;
  created_at: string;
  updated_at: string;
};

const STRATEGY_VERSION = "deepseek_prompt_cache_v2" as const;
const TARGET_CACHE_HIT_RATIO = 0.8;

const STABLE_SYSTEM_PROMPT = [
  "Research Agent Workstation DeepSeek Code Agent stable system prefix.",
  "Role: disciplined code-writing agent for an auditable AI research workstation.",
  "Output contract: produce concise, reviewable implementation guidance and a unified diff when useful.",
  "Never claim files were applied. The workstation imports diffs through Code Quality Gate and Human Gate.",
  "Respect artifact-based workflow: every proposed change must be traceable to task evidence, gate state, and report requirements.",
  "Do not request or expose secrets. Do not print API keys, Kaggle tokens, SSH passwords, cookies, or environment secrets.",
  "Do not directly train, submit to Kaggle, or bypass workstation gates. Generate code proposals only.",
  "Prefer minimal patches, existing paths, deterministic outputs, and Windows-safe plain ASCII punctuation inside diffs.",
  "Cache discipline: this system prompt is intentionally stable across sessions to maximize DeepSeek automatic prefix cache reuse.",
  "",
  "Stable implementation rubric:",
  "- Use existing workstation APIs and artifacts rather than inventing new entry points.",
  "- Bind every proposed code change to a gate, artifact, metric, or report requirement.",
  "- Keep changes small, reviewable, reversible, and compatible with Windows paths.",
  "- Prefer deterministic helpers, explicit manifests, and schema-compatible JSON outputs.",
  "- Include rollback conditions when proposing training or orchestration changes.",
  "- Do not convert weak evidence into confirmed conclusions.",
  "- If a branch does not improve, preserve parent best and write memory rather than claiming success.",
  "- Treat generated code as draft evidence until Code Quality Gate and Human Gate approve it.",
  "- For tabular MLE tasks, preserve OOF, metrics, submission schema, and claim audit artifacts.",
  "- For UI/report tasks, preserve real statuses and never display blocked resources as ready.",
  "- For DeepSeek cost control, keep reusable instructions byte-stable and deterministic.",
  "- Exact prompt fingerprints may be served from local response cache without an API call."
].join("\n");

const STABLE_USER_PREFIX = [
  "Research Agent Workstation cacheable request prefix v2.",
  "The dynamic task evidence begins after the marker below. Treat all following task evidence as bounded context.",
  "Workflow gates: plan approval, code quality approval, HPC execution approval, submission approval, final report approval.",
  "Agent responsibilities: read evidence, propose bounded code changes, include rollback notes, avoid unrelated refactors.",
  "Cache objective: keep this prefix byte-stable so repeated coding-agent sessions can reuse provider prompt cache.",
  "Cost objective: target >=80% cache hit ratio through local exact-response cache for identical prompt fingerprints plus provider prompt cache for stable prefixes.",
  "Local exact-response cache rule: if prompt_fingerprint and model match a completed cached response, reuse it and do not call the external model.",
  "Provider cache rule: if local exact-response cache misses, send a byte-stable system prompt and user prefix so the provider can cache the shared prefix.",
  "Do not store prompt text in cache ledgers; store hashes, response artifacts, usage, and hit-rate metadata only.",
  "Do not store secrets in cache entries.",
  "Dynamic evidence marker:"
].join("\n");

function sha256(text: string) {
  return createHash("sha256").update(text, "utf-8").digest("hex");
}

function numberFrom(record: Record<string, unknown>, key: string) {
  const value = record[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function extractDeepSeekCacheUsage(usage: unknown): DeepSeekCacheUsage {
  if (!usage || typeof usage !== "object" || Array.isArray(usage)) {
    return {
      prompt_tokens: null,
      completion_tokens: null,
      total_tokens: null,
      cached_tokens: null,
      cache_hit_ratio: null,
      cache_observed: false,
      local_response_cache_hit: false
    };
  }

  const record = usage as Record<string, unknown>;
  const promptTokens = numberFrom(record, "prompt_tokens");
  const completionTokens = numberFrom(record, "completion_tokens");
  const totalTokens = numberFrom(record, "total_tokens");
  let cachedTokens = numberFrom(record, "prompt_cache_hit_tokens");

  const details = record.prompt_tokens_details;
  if (cachedTokens === null && details && typeof details === "object" && !Array.isArray(details)) {
    cachedTokens = numberFrom(details as Record<string, unknown>, "cached_tokens");
  }

  const cacheHitRatio = promptTokens && cachedTokens !== null
    ? Number((cachedTokens / Math.max(promptTokens, 1)).toFixed(6))
    : null;

  return {
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    total_tokens: totalTokens,
    cached_tokens: cachedTokens,
    cache_hit_ratio: cacheHitRatio,
    cache_observed: cachedTokens !== null,
    local_response_cache_hit: false
  };
}

export function localResponseCacheUsage(promptChars: number, completionChars: number): DeepSeekCacheUsage {
  const estimatedPromptTokens = Math.max(1, Math.ceil(promptChars / 4));
  const estimatedCompletionTokens = Math.max(1, Math.ceil(completionChars / 4));
  return {
    prompt_tokens: estimatedPromptTokens,
    completion_tokens: estimatedCompletionTokens,
    total_tokens: estimatedPromptTokens + estimatedCompletionTokens,
    cached_tokens: estimatedPromptTokens,
    cache_hit_ratio: 1,
    cache_observed: true,
    local_response_cache_hit: true
  };
}

export function createDeepSeekCacheMessages(dynamicPrompt: string) {
  const userContent = `${STABLE_USER_PREFIX}\n---BEGIN_DYNAMIC_TASK_EVIDENCE---\n${dynamicPrompt}`;
  const stableUserPrefixHash = sha256(STABLE_USER_PREFIX);
  const stableSystemHash = sha256(STABLE_SYSTEM_PROMPT);
  const dynamicSuffixHash = sha256(dynamicPrompt);
  const cacheKey = sha256(`${STRATEGY_VERSION}:${stableSystemHash}:${stableUserPrefixHash}`);
  const metadata: DeepSeekCacheMetadata = {
    strategy_version: STRATEGY_VERSION,
    cache_agent: "DeepSeekCacheOptimizationAgent",
    cache_policy: "local exact-response cache for identical prompt fingerprints plus automatic provider prompt cache via byte-stable system prompt and user prefix",
    provider_cache_mode: "local_exact_response_cache_plus_automatic_stable_prefix",
    target_cache_hit_ratio: TARGET_CACHE_HIT_RATIO,
    stable_system_hash: stableSystemHash,
    stable_user_prefix_hash: stableUserPrefixHash,
    dynamic_suffix_hash: dynamicSuffixHash,
    prompt_fingerprint: sha256(`${cacheKey}:${dynamicSuffixHash}`),
    stable_prefix_chars: STABLE_SYSTEM_PROMPT.length + STABLE_USER_PREFIX.length,
    dynamic_suffix_chars: dynamicPrompt.length,
    cache_key: cacheKey
  };

  const messages: DeepSeekMessage[] = [
    { role: "system", content: STABLE_SYSTEM_PROMPT },
    { role: "user", content: userContent }
  ];

  return { messages, metadata };
}

export function attachDeepSeekCacheUsage(metadata: DeepSeekCacheMetadata, usage: unknown): DeepSeekCacheMetadata {
  return {
    ...metadata,
    usage: extractDeepSeekCacheUsage(usage)
  };
}

function responseCacheRelative(promptFingerprint: string) {
  return path.join("workspace", "code_agent_cache", "responses", `${promptFingerprint}.json`).replaceAll("\\", "/");
}

export async function readDeepSeekCachedResponse(input: {
  promptFingerprint: string;
  cacheKey: string;
  model: string;
}) {
  const relativePath = responseCacheRelative(input.promptFingerprint);
  const cached = await readJsonFile(resolveWorkspacePath(relativePath));
  if (!cached || typeof cached !== "object" || Array.isArray(cached)) return null;
  const record = cached as Partial<DeepSeekCachedResponse>;
  if (
    record.ok === true &&
    record.prompt_fingerprint === input.promptFingerprint &&
    record.cache_key === input.cacheKey &&
    record.model === input.model &&
    typeof record.generated_text === "string" &&
    typeof record.patch_diff === "string"
  ) {
    return { relativePath, record: record as DeepSeekCachedResponse };
  }
  return null;
}

export async function writeDeepSeekCachedResponse(input: {
  promptFingerprint: string;
  cacheKey: string;
  model: string;
  generatedText: string;
  patchDiff: string;
  usage?: Record<string, unknown>;
}) {
  const relativePath = responseCacheRelative(input.promptFingerprint);
  const now = new Date().toISOString();
  const existing = await readJsonFile(resolveWorkspacePath(relativePath));
  const createdAt = existing && typeof existing === "object" && !Array.isArray(existing) && typeof (existing as Record<string, unknown>).created_at === "string"
    ? String((existing as Record<string, unknown>).created_at)
    : now;
  const record: DeepSeekCachedResponse = {
    ok: true,
    artifact_type: "deepseek_code_agent_response_cache_entry",
    strategy_version: STRATEGY_VERSION,
    prompt_fingerprint: input.promptFingerprint,
    cache_key: input.cacheKey,
    model: input.model,
    generated_text: input.generatedText,
    patch_diff: input.patchDiff,
    usage: input.usage,
    created_at: createdAt,
    updated_at: now
  };
  await writeJsonArtifact(relativePath, record);
  return relativePath;
}

export async function recordDeepSeekCacheSession(input: {
  sessionId: string;
  taskId: string;
  model: string;
  status: string;
  transcriptPath?: string | null;
  manifestPath?: string | null;
  metadata: DeepSeekCacheMetadata;
}) {
  const relativePath = path.join("workspace", "code_agent_cache", "deepseek_cache_manifest.json").replaceAll("\\", "/");
  const existing = await readJsonFile(resolveWorkspacePath(relativePath));
  const sessions = Array.isArray((existing as Record<string, unknown> | null)?.sessions)
    ? ((existing as Record<string, unknown>).sessions as Array<Record<string, unknown>>)
    : [];
  const nextSessions = [
    ...sessions.filter((session) => session.session_id !== input.sessionId),
    {
      session_id: input.sessionId,
      task_id: input.taskId,
      model: input.model,
      status: input.status,
      transcript_path: input.transcriptPath ?? null,
      manifest_path: input.manifestPath ?? null,
      cache_key: input.metadata.cache_key,
      target_cache_hit_ratio: input.metadata.target_cache_hit_ratio,
      prompt_fingerprint: input.metadata.prompt_fingerprint,
      stable_system_hash: input.metadata.stable_system_hash,
      stable_user_prefix_hash: input.metadata.stable_user_prefix_hash,
      dynamic_suffix_hash: input.metadata.dynamic_suffix_hash,
      cached_tokens: input.metadata.usage?.cached_tokens ?? null,
      cache_hit_ratio: input.metadata.usage?.cache_hit_ratio ?? null,
      local_response_cache_hit: input.metadata.usage?.local_response_cache_hit ?? false,
      cache_observed: input.metadata.usage?.cache_observed ?? false,
      updated_at: new Date().toISOString()
    }
  ].slice(-200);

  await writeJsonArtifact(relativePath, {
    ok: true,
    artifact_type: "deepseek_code_agent_cache_manifest",
    created_by_agent: "DeepSeekCacheOptimizationAgent",
    strategy_version: STRATEGY_VERSION,
    cache_policy: "local exact-response cache plus stable-prefix prompt construction; metadata ledger stores no prompt text or secrets",
    target_cache_hit_ratio: TARGET_CACHE_HIT_RATIO,
    session_count: nextSessions.length,
    sessions: nextSessions,
    updated_at: new Date().toISOString()
  });

  return relativePath;
}
