const OFFICIAL_DEEPSEEK_ORIGINS = new Set(["https://api.deepseek.com"]);
const MAX_PROVIDER_RESPONSE_BYTES = 1_000_000;

export class ProviderBoundaryError extends Error {
  constructor(public readonly code: string) {
    super(code);
    this.name = "ProviderBoundaryError";
  }
}

function isLoopback(hostname: string) {
  const normalized = hostname.toLowerCase().replace(/^\[|\]$/g, "");
  return normalized === "localhost" || normalized === "127.0.0.1" || normalized === "::1";
}

function parseOrigin(value: string) {
  const url = new URL(value);
  if (url.username || url.password || url.search || url.hash) {
    throw new ProviderBoundaryError("provider_url_contains_forbidden_components");
  }
  if (url.protocol !== "https:" && !(url.protocol === "http:" && isLoopback(url.hostname))) {
    throw new ProviderBoundaryError("provider_url_requires_https_or_loopback");
  }
  return url;
}

function configuredOrigins() {
  const origins = new Set(OFFICIAL_DEEPSEEK_ORIGINS);
  for (const raw of (process.env.DEEPSEEK_ALLOWED_ORIGINS ?? "").split(",")) {
    const value = raw.trim();
    if (!value) continue;
    const url = parseOrigin(value);
    if (url.pathname !== "/") {
      throw new ProviderBoundaryError("provider_allowlist_entry_must_be_an_origin");
    }
    origins.add(url.origin);
  }
  return origins;
}

export function resolveDeepSeekEndpoint(rawBaseUrl: string) {
  const parsed = parseOrigin(rawBaseUrl);
  if (!configuredOrigins().has(parsed.origin)) {
    throw new ProviderBoundaryError("provider_origin_not_allowlisted");
  }
  const basePath = parsed.pathname.replace(/\/+$/, "");
  const baseUrl = `${parsed.origin}${basePath}`;
  return {
    baseUrl,
    chatCompletionsUrl: `${baseUrl}/chat/completions`
  };
}

export async function readBoundedProviderJson(
  response: Response,
  maxBytes = MAX_PROVIDER_RESPONSE_BYTES
): Promise<Record<string, unknown>> {
  const declaredLength = Number(response.headers.get("content-length") ?? "0");
  if (Number.isFinite(declaredLength) && declaredLength > maxBytes) {
    throw new ProviderBoundaryError("provider_response_too_large");
  }
  if (!response.body) {
    throw new ProviderBoundaryError("provider_response_missing_body");
  }
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  while (true) {
    const item = await reader.read();
    if (item.done) break;
    total += item.value.byteLength;
    if (total > maxBytes) {
      await reader.cancel();
      throw new ProviderBoundaryError("provider_response_too_large");
    }
    chunks.push(item.value);
  }
  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    throw new ProviderBoundaryError("provider_response_invalid_utf8");
  }
  try {
    const payload = JSON.parse(text);
    if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
      throw new ProviderBoundaryError("provider_response_invalid_json_shape");
    }
    return payload as Record<string, unknown>;
  } catch (error) {
    if (error instanceof ProviderBoundaryError) throw error;
    throw new ProviderBoundaryError("provider_response_invalid_json");
  }
}

export function providerHttpFailure(provider: string, response: Response) {
  const hasRequestId = Boolean(response.headers.get("x-request-id") ?? response.headers.get("request-id"));
  return `${provider}_http_${response.status}_request_${hasRequestId ? "present" : "unavailable"}`;
}

export function safeProviderFailure(error: unknown, fallback: string) {
  if (error instanceof ProviderBoundaryError) return `${fallback}_${error.code}`;
  if (error instanceof DOMException && error.name === "TimeoutError") return `${fallback}_timeout`;
  return `${fallback}_network_error`;
}
