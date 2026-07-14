import { logAction } from "@/lib/server/actions";
import { deepSeekApiKeyStatus, deepSeekConfig, hasDeepSeekApiKey } from "@/lib/server/capabilities";
import { stamp, writeJsonArtifact } from "@/lib/server/paths";
import { providerHttpFailure, readBoundedProviderJson, safeProviderFailure } from "@/lib/security/provider-boundary";

type DeepSeekSmokeStatus = "not_configured" | "passed" | "failed";

const SUPPORTED_DEEPSEEK_MODELS = ["deepseek-v4-flash", "deepseek-v4-pro"] as const;

export type DeepSeekSmokeResult = {
  ok: boolean;
  configured: boolean;
  provider: "deepseek";
  status: DeepSeekSmokeStatus;
  model: string;
  base_url: string;
  content?: string;
  usage?: Record<string, unknown>;
  artifact_path?: string;
  missing_env?: string[];
  error?: string;
};

function artifactPath() {
  return `workspace/llm/deepseek_smoke_${stamp()}.json`;
}

async function writeSmokeArtifact(result: DeepSeekSmokeResult) {
  const path = artifactPath();
  await writeJsonArtifact(path, {
    ...result,
    api_key_status: deepSeekApiKeyStatus(),
    checked_at: new Date().toISOString()
  });
  return path;
}

export async function runDeepSeekSmoke(prompt = "Return exactly: deepseek-ok"): Promise<DeepSeekSmokeResult> {
  const config = deepSeekConfig();
  if (config.boundaryError || !config.chatCompletionsUrl) {
    const result: DeepSeekSmokeResult = {
      ok: false,
      configured: hasDeepSeekApiKey(),
      provider: "deepseek",
      status: "failed",
      model: config.model,
      base_url: "invalid",
      error: `deepseek_endpoint_blocked_${config.boundaryError ?? "invalid"}`
    };
    const artifact = await writeSmokeArtifact(result);
    await logAction({
      action: "deepseek_smoke_failed",
      message: result.error ?? "DeepSeek endpoint blocked by provider policy.",
      artifactPath: artifact,
      metadata: { provider: "deepseek", model: result.model, endpoint_status: "blocked" }
    });
    return { ...result, artifact_path: artifact };
  }
  if (!SUPPORTED_DEEPSEEK_MODELS.includes(config.model as (typeof SUPPORTED_DEEPSEEK_MODELS)[number])) {
    const message = `Unsupported DeepSeek model: ${config.model}. Use deepseek-v4-flash or deepseek-v4-pro.`;
    const result: DeepSeekSmokeResult = {
      ok: false,
      configured: hasDeepSeekApiKey(),
      provider: "deepseek",
      status: "failed",
      model: config.model,
      base_url: config.baseUrl,
      error: message,
      missing_env: hasDeepSeekApiKey() ? undefined : ["DEEPSEEK_API_KEY"]
    };
    const artifact = await writeSmokeArtifact(result);
    await logAction({
      action: "deepseek_smoke_failed",
      message,
      artifactPath: artifact,
      metadata: { provider: "deepseek", model: result.model, base_url: result.base_url }
    });
    return { ...result, artifact_path: artifact };
  }
  const base: DeepSeekSmokeResult = {
    ok: hasDeepSeekApiKey(),
    configured: hasDeepSeekApiKey(),
    provider: "deepseek",
    status: hasDeepSeekApiKey() ? "failed" : "not_configured",
    model: config.model,
    base_url: config.baseUrl,
    missing_env: hasDeepSeekApiKey() ? undefined : ["DEEPSEEK_API_KEY"]
  };

  if (!hasDeepSeekApiKey()) {
    const artifact = await writeSmokeArtifact(base);
    await logAction({
      action: "deepseek_smoke_not_configured",
      message: "DeepSeek is not configured. Set DEEPSEEK_API_KEY before running real model smoke tests.",
      artifactPath: artifact,
      metadata: { provider: "deepseek", api_key_status: deepSeekApiKeyStatus(), missing_env: base.missing_env }
    });
    return { ...base, artifact_path: artifact };
  }

  try {
    const response = await fetch(config.chatCompletionsUrl, {
      method: "POST",
      headers: {
        // lgtm[js/file-access-to-http] The destination is an explicit allowlisted origin from provider-boundary.
        Authorization: `Bearer ${config.apiKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: config.model,
        messages: [
          { role: "system", content: "You are a concise research workstation smoke-test assistant." },
          { role: "user", content: prompt }
        ],
        stream: false
      }),
      redirect: "error",
      signal: AbortSignal.timeout(120_000)
    });
    const payload = await readBoundedProviderJson(response);
    if (!response.ok) {
      throw new Error(providerHttpFailure("deepseek", response));
    }
    const choices = payload.choices as Array<{ message?: { content?: string } }> | undefined;
    const result: DeepSeekSmokeResult = {
      ...base,
      ok: true,
      configured: true,
      status: "passed",
      model: String(payload.model ?? config.model),
      content: choices?.[0]?.message?.content ?? "",
      usage: payload.usage as Record<string, unknown> | undefined,
      missing_env: undefined
    };
    const artifact = await writeSmokeArtifact(result);
    await logAction({
      action: "deepseek_smoke_passed",
      message: "DeepSeek model smoke test passed.",
      artifactPath: artifact,
      metadata: { provider: "deepseek", model: result.model, base_url: result.base_url, usage: result.usage }
    });
    return { ...result, artifact_path: artifact };
  } catch (error) {
    const result: DeepSeekSmokeResult = {
      ...base,
      ok: false,
      configured: true,
      status: "failed",
      error: error instanceof Error && /^deepseek_http_\d{3}_request_[A-Za-z0-9._:-]+$/.test(error.message)
        ? error.message
        : safeProviderFailure(error, "deepseek_smoke_failed"),
      missing_env: undefined
    };
    const artifact = await writeSmokeArtifact(result);
    await logAction({
      action: "deepseek_smoke_failed",
      message: result.error ?? "DeepSeek smoke test failed.",
      artifactPath: artifact,
      metadata: { provider: "deepseek", model: result.model, base_url: result.base_url }
    });
    return { ...result, artifact_path: artifact };
  }
}
