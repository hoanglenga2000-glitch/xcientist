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

function sanitizeClientText(value: string) {
  if (!value.includes("\uFFFD")) return value;
  return "[unreadable historical text redacted]";
}
