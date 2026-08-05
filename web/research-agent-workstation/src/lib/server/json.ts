export function encodeJson(value: unknown) {
  if (value == null) return null;
  // Sanitize Infinity/NaN before serialization
  const sanitized = sanitizeJsonValue(value);
  return JSON.stringify(sanitized);
}

function sanitizeJsonValue(value: unknown): unknown {
  if (typeof value === "number" && !isFinite(value)) return null;
  if (Array.isArray(value)) return value.map(sanitizeJsonValue);
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const result: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      result[k] = sanitizeJsonValue(v);
    }
    return result;
  }
  return value;
}

export function decodeJson<T = unknown>(value: string | null | undefined): T | null {
  if (!value) return null;
  try {
    // Pre-sanitize Infinity/NaN tokens that Python may have written
    const cleaned = value.replace(/: -?Infinity/g, ": null").replace(/: NaN/g, ": null");
    return JSON.parse(cleaned) as T;
  } catch {
    return null;
  }
}

export function sanitizeClientJson(value: unknown): unknown {
  if (typeof value === "string") return sanitizeClientText(value);
  if (Array.isArray(value)) return value.map(sanitizeClientJson);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([key, item]) => [key, sanitizeClientJson(item)])
    );
  }
  return value;
}

const OMIT_FROM_CLIENT = Symbol("omit-from-client");

/**
 * Produce the deliberately narrow projection used by workstation-summary.
 *
 * The summary aggregates historical artifacts and connector diagnostics, so a
 * single nested record can otherwise expose a local path, credential store, or
 * remote allocation identity.  Relative workspace references remain useful to
 * the UI and are retained.  This is kept separate from sanitizeClientJson so
 * internal server serializers do not silently lose fields.
 */
export function sanitizeWorkstationSummary(value: unknown): unknown {
  const sanitized = sanitizeSummaryValue(value, []);
  return sanitized === OMIT_FROM_CLIENT ? null : sanitized;
}

function sanitizeSummaryValue(value: unknown, parents: string[]): unknown | typeof OMIT_FROM_CLIENT {
  if (typeof value === "string") return sanitizeSummaryText(value);
  if (Array.isArray(value)) {
    return value
      .map((item) => sanitizeSummaryValue(item, parents))
      .filter((item) => item !== OMIT_FROM_CLIENT);
  }
  if (value && typeof value === "object") {
    const result: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(value as Record<string, unknown>)) {
      if (isSensitiveSummaryKey(key, parents)) continue;
      const child = sanitizeSummaryValue(item, [...parents, key]);
      if (child !== OMIT_FROM_CLIENT) result[key] = child;
    }
    return result;
  }
  return value;
}

function isSensitiveSummaryKey(key: string, parents: string[]) {
  const normalized = key.toLowerCase().replaceAll("-", "_");
  const context = [...parents, normalized].join(".").toLowerCase();

  if (/^(?:workspace_root|root_path|absolute_path|local_path|remote_root|base_url|api_base|endpoint_url)$/.test(normalized)) return true;
  if (/(?:password|passwd|secret|private_key|api_key|authorization)/.test(normalized)) return true;
  if (/^(?:token|access_token|refresh_token|session_token|bootstrap_token|api_token|auth_token)$/.test(normalized)) return true;
  if (/^(?:cookie|cookies|session_cookie|set_cookie)$/.test(normalized)) return true;
  if (/(?:credential|dpapi)/.test(normalized)) return true;

  // Infrastructure identity is never needed for rendering the local UI.  Keep
  // generic readiness booleans/status text, but remove the addressable route.
  if (/^(?:host|hostname|host_uuid|gpu_uuid|port|username|profile|profile_id|profile_name|account|role_account)$/.test(normalized)) return true;
  if (/^(?:latest_ssh_connection|ssh_route|gateway_route|proxy_route|remote_endpoint)$/.test(normalized)) return true;
  if (/(?:ssh|gateway|proxy|allocation|connection).*(?:host|port|user|account|uuid|key|path|route|address|endpoint|command|url)/.test(normalized)) return true;
  if (/(?:host|port|user|account|uuid|key|path|route|address|endpoint|command|url).*(?:ssh|gateway|proxy|allocation|connection)/.test(normalized)) return true;
  if (normalized === "id" && /(?:ssh|gateway|proxy|allocation|profile|connection)[^.]*$/.test(context)) return true;
  return false;
}

function sanitizeSummaryText(value: string) {
  const isAbsolutePath = /^(?:[A-Za-z]:[\\/]|\\\\|\/\/|\/(?!(?:api|_next|healthz)(?:\/|$)))/.test(value);
  if (isAbsolutePath) return "[local path redacted]";

  let sanitized = sanitizeClientText(value);
  sanitized = sanitized
    .replace(/(["'])[A-Za-z]:[\\/][^"'`\r\n]+\1/g, "[local path redacted]")
    .replace(/(["'])\/(?:home|Users|var|tmp|opt|etc|mnt|srv|root|data|workspace|usr|run|bin|sbin|lib|lib64)\/[^"'`\r\n]+\1/g, "[local path redacted]")
    .replace(/\b(?:Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+/gi, "[credential redacted]")
    .replace(/\b(api[_-]?key|token|access[_-]?token|refresh[_-]?token|session[_-]?token|password|passwd|secret|cookie)\s*[:=]\s*(?:"[^"]*"|'[^']*'|[^\s,;]+)/gi, "$1=[redacted]")
    .replace(/([a-z][a-z0-9+.-]*:\/\/)[^\s/@:]+:[^\s/@]+@/gi, "$1[credential redacted]@")
    .replace(/\b(?:ssh|sftp):\/\/[^\s,;]+/gi, "[infrastructure endpoint redacted]")
    .replace(/\b(host|hostname|port|gateway|proxy|endpoint|profile)\s*[:=]\s*[^\s,;]+/gi, "$1=[infrastructure identity redacted]")
    .replace(/\b(?:\d{1,3}\.){3}\d{1,3}:\d{1,5}\b/g, "[infrastructure endpoint redacted]")
    .replace(/\b[A-Za-z]:[\\/][^\s"'`<>|]+/g, "[local path redacted]")
    .replace(/(?:\\\\|(?<!:)\/\/)[^\s"'`<>|]+[\\/][^\s"'`<>|]+/g, "[local path redacted]")
    .replace(/\/(?:home|Users|var|tmp|opt|etc|mnt|srv|root|data|workspace|usr|run|bin|sbin|lib|lib64)\/[^\s"'`<>]*/g, "[local path redacted]");

  return sanitized;
}

function sanitizeClientText(value: string) {
  if (!value.includes("\uFFFD")) return value;
  return "[unreadable historical text redacted]";
}
