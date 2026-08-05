const MAX_AUDIT_TEXT_CHARS = 16_000;
const MAX_AUDIT_METADATA_BYTES = 128 * 1024;
const MAX_AUDIT_NODES = 2_000;
const MAX_COLLECTION_ITEMS = 128;

const PRIVATE_KEY_RE = /-----BEGIN(?: [A-Z0-9]+)* PRIVATE KEY-----[\s\S]*?-----END(?: [A-Z0-9]+)* PRIVATE KEY-----/gi;
const AUTHORIZATION_RE = /\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{3,}/gi;
const URL_USERINFO_RE = /\b(https?:\/\/)[^/@\s:]+:[^/@\s]+@/gi;
const QUERY_SECRET_RE = /([?&](?:api[_-]?key|password|passwd|passphrase|secret|access[_-]?token|refresh[_-]?token|token|aws[_-]?(?:access[_-]?key[_-]?id|secret[_-]?access[_-]?key)|github[_-]?pat|gitlab[_-]?pat|kaggle[_-]?key)=)[^&#\s]+/gi;
const KEY_VALUE_SECRET_RE = /(["']?(?:api[_-]?key|authorization|cookie|credential|credentials|password|passwd|passphrase|private[_-]?key|client[_-]?secret|access[_-]?token|refresh[_-]?token|auth[_-]?token|secret|token|aws[_-]?(?:access[_-]?key[_-]?id|secret[_-]?access[_-]?key)|github[_-]?pat|gitlab[_-]?pat|kaggle[_-]?key)["']?\s*[:=]\s*["']?)([^"'\s,;}\]]{3,})/gi;
const CONTROL_CHARACTER_RE = /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/g;

const SAFE_TOKEN_KEYS = new Set(["input_tokens", "max_tokens", "output_tokens", "token_count", "tokens_used"]);
const SENSITIVE_KEYS = new Set([
  "access_token",
  "api_key",
  "apikey",
  "authorization",
  "auth_token",
  "aws_access_key_id",
  "aws_secret_access_key",
  "client_secret",
  "cookie",
  "credential",
  "credentials",
  "github_pat",
  "gitlab_pat",
  "kaggle_key",
  "passphrase",
  "password",
  "passwd",
  "private_key",
  "refresh_token",
  "secret",
  "token"
]);
const SAFE_SENSITIVE_SUFFIXES = [
  "_configured",
  "_count",
  "_digest",
  "_enabled",
  "_length",
  "_len",
  "_path",
  "_persisted",
  "_present",
  "_sha256",
  "_source",
  "_status"
];

function normalizedKey(value: string) {
  return value
    .trim()
    .replace(/([A-Z]+)([A-Z][a-z])/g, "$1_$2")
    .replace(/([a-z0-9])([A-Z])/g, "$1_$2")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function isSensitiveKey(value: string) {
  const key = normalizedKey(value);
  if (!key || SAFE_TOKEN_KEYS.has(key) || SAFE_SENSITIVE_SUFFIXES.some((suffix) => key.endsWith(suffix))) {
    return false;
  }
  return SENSITIVE_KEYS.has(key)
    || /_(?:api_key|auth_token|password|passphrase|private_key|secret|token)$/.test(key)
    || /_(?:github|gitlab)_pat$/.test(key);
}

export function sanitizeAuditText(value: unknown, maxChars = MAX_AUDIT_TEXT_CHARS) {
  const boundedLimit = Math.max(0, Math.min(Number.isSafeInteger(maxChars) ? maxChars : 0, MAX_AUDIT_TEXT_CHARS));
  return String(value ?? "")
    .replace(CONTROL_CHARACTER_RE, " ")
    .replace(PRIVATE_KEY_RE, "[redacted private key]")
    .replace(URL_USERINFO_RE, "$1[redacted]@")
    .replace(QUERY_SECRET_RE, "$1[redacted]")
    .replace(AUTHORIZATION_RE, "$1 [redacted]")
    .replace(KEY_VALUE_SECRET_RE, "$1[redacted]")
    .slice(0, boundedLimit);
}

function sanitizeValue(value: unknown, key: string, depth: number, budget: { nodes: number }): unknown {
  budget.nodes += 1;
  if (budget.nodes > MAX_AUDIT_NODES || depth > 10) return "[truncated]";
  if (isSensitiveKey(key)) return "[redacted]";
  if (value === null || typeof value === "boolean") return value;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value === "string") return sanitizeAuditText(value, 4_000);
  if (Array.isArray(value)) {
    return value.slice(0, MAX_COLLECTION_ITEMS).map((item) => sanitizeValue(item, key, depth + 1, budget));
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .slice(0, MAX_COLLECTION_ITEMS)
        .map(([itemKey, item]) => [
          sanitizeAuditText(itemKey, 120),
          sanitizeValue(item, itemKey, depth + 1, budget)
        ])
    );
  }
  return sanitizeAuditText(value, 1_000);
}

export function sanitizeAuditMetadata(value: Record<string, unknown> | undefined) {
  const sanitized = sanitizeValue(value ?? {}, "", 0, { nodes: 0 });
  const record = sanitized && typeof sanitized === "object" && !Array.isArray(sanitized)
    ? sanitized as Record<string, unknown>
    : { value: sanitized };
  if (Buffer.byteLength(JSON.stringify(record), "utf8") > MAX_AUDIT_METADATA_BYTES) {
    return { audit_metadata_truncated: true };
  }
  return record;
}

export function normalizeAuditAction(value: unknown) {
  const action = String(value ?? "").trim();
  if (!/^[a-z][a-z0-9_.:-]{0,119}$/.test(action)) {
    throw new Error("Invalid audit action");
  }
  return action;
}

export function sanitizeAuditArtifactPath(value: unknown) {
  if (value === null || value === undefined || value === "") return null;
  const artifactPath = sanitizeAuditText(value, 600).replaceAll("\\", "/");
  const segments = artifactPath.split("/");
  if (
    artifactPath.startsWith("/")
    || /^[A-Za-z]:/.test(artifactPath)
    || segments.includes("..")
    || segments.some((segment) => segment.includes(":"))
  ) {
    return null;
  }
  return artifactPath;
}
