const TASK_ID_PATTERN = /^[a-z0-9][a-z0-9._-]{0,127}$/;
const WINDOWS_RESERVED_NAME = /^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i;
const LOOPBACK_HOST_HEADER = /^(localhost|127(?:\.\d{1,3}){3}|\[::1\])(?::(\d{1,5}))?$/i;

export function normalizeTaskId(taskId: string) {
  let decoded: string;
  try {
    decoded = decodeURIComponent(taskId);
  } catch {
    throw new Error("Invalid task ID encoding");
  }
  const normalized = decoded === "house-prices" ? "house_prices" : decoded;
  if (
    !TASK_ID_PATTERN.test(normalized)
    || normalized.includes("..")
    || normalized.endsWith(".")
    || WINDOWS_RESERVED_NAME.test(normalized)
  ) {
    throw new Error("Invalid task ID");
  }
  return normalized;
}

export function isLoopbackHostname(hostname: string) {
  const normalized = hostname.toLowerCase().replace(/^\[|\]$/g, "").replace(/\.$/, "");
  if (normalized === "localhost" || normalized === "::1") return true;
  const octets = normalized.split(".");
  if (octets.length !== 4 || octets[0] !== "127") return false;
  return octets.every((octet) => /^\d{1,3}$/.test(octet) && Number(octet) <= 255);
}

export function isLoopbackHostHeader(hostHeader: string | null) {
  if (!hostHeader) return false;
  if (hostHeader !== hostHeader.trim() || /[\s\\/@?#]/.test(hostHeader)) return false;
  const match = LOOPBACK_HOST_HEADER.exec(hostHeader);
  if (!match) return false;
  const port = match[2] ? Number(match[2]) : null;
  if (port !== null && (!Number.isInteger(port) || port < 1 || port > 65535)) return false;
  return isLoopbackHostname(match[1]);
}

export function isAllowedBrowserOrigin(origin: string | null, hostHeader: string | null) {
  if (!origin) return false;
  if (!isLoopbackHostHeader(hostHeader)) return false;
  try {
    const parsed = new URL(origin);
    return (
      (parsed.protocol === "http:" || parsed.protocol === "https:")
      && !parsed.username
      && !parsed.password
      && parsed.pathname === "/"
      && !parsed.search
      && !parsed.hash
      && isLoopbackHostname(parsed.hostname)
      && parsed.host.toLowerCase() === hostHeader?.toLowerCase()
    );
  } catch {
    return false;
  }
}

export function isAllowedMutationSource(
  origin: string | null,
  hostHeader: string | null,
  secFetchSite: string | null
) {
  if (origin) return isAllowedBrowserOrigin(origin, hostHeader);
  return isLoopbackHostHeader(hostHeader) && secFetchSite?.toLowerCase() === "same-origin";
}
