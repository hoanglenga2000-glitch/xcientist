const SAFE_METHODS = new Set(["GET", "HEAD", "OPTIONS"]);
const SESSION_COOKIE_NAME = "evomind_local_session";

export class LoopbackSessionError extends Error {
  constructor(message, code) {
    super(message);
    this.name = "LoopbackSessionError";
    this.code = code;
  }
}

function loopbackOrigin(value) {
  const url = new URL(value);
  if (!new Set(["http:", "https:"]).has(url.protocol)) {
    throw new LoopbackSessionError("The contract base URL must use HTTP or HTTPS.", "invalid_protocol");
  }
  if (!new Set(["127.0.0.1", "localhost", "[::1]", "::1"]).has(url.hostname)) {
    throw new LoopbackSessionError("The contract base URL must resolve to loopback.", "non_loopback_origin");
  }
  if (url.username || url.password || url.pathname !== "/" || url.search || url.hash) {
    throw new LoopbackSessionError("The contract base URL must be a bare loopback origin.", "invalid_base_url");
  }
  return url.origin;
}

function cookieFromHeader(headerValue) {
  const parts = String(headerValue ?? "").split(";").map((value) => value.trim()).filter(Boolean);
  const pair = parts.shift() ?? "";
  const separator = pair.indexOf("=");
  if (separator <= 0) throw new LoopbackSessionError("Session bootstrap did not set a cookie.", "missing_session_cookie");
  const name = pair.slice(0, separator);
  const value = pair.slice(separator + 1);
  const attributes = new Map(parts.map((attribute) => {
    const index = attribute.indexOf("=");
    return index < 0
      ? [attribute.toLowerCase(), ""]
      : [attribute.slice(0, index).toLowerCase(), attribute.slice(index + 1)];
  }));
  if (name !== SESSION_COOKIE_NAME || !value) {
    throw new LoopbackSessionError("Session bootstrap returned an unexpected cookie.", "invalid_session_cookie");
  }
  if (!attributes.has("httponly")) {
    throw new LoopbackSessionError("Session cookie is missing HttpOnly.", "cookie_missing_httponly");
  }
  if (String(attributes.get("samesite") ?? "").toLowerCase() !== "strict") {
    throw new LoopbackSessionError("Session cookie is missing SameSite=Strict.", "cookie_missing_samesite_strict");
  }
  if (attributes.get("path") !== "/") {
    throw new LoopbackSessionError("Session cookie is missing Path=/.", "cookie_missing_root_path");
  }
  if (attributes.has("domain")) {
    throw new LoopbackSessionError("Loopback session cookies must remain host-only.", "cookie_has_domain");
  }
  return pair;
}

async function json(response, code) {
  try {
    const payload = await response.json();
    if (payload && typeof payload === "object" && !Array.isArray(payload)) return payload;
  } catch {
    // The caller receives a stable contract error below.
  }
  throw new LoopbackSessionError("The local session endpoint returned malformed JSON.", code);
}

export async function createLoopbackSession({ baseUrl, bootstrapToken, fetchImpl = globalThis.fetch }) {
  const origin = loopbackOrigin(baseUrl);
  if (typeof fetchImpl !== "function") throw new LoopbackSessionError("Fetch is unavailable.", "fetch_unavailable");
  let bootstrapSecret = typeof bootstrapToken === "string" ? bootstrapToken : "";
  if (bootstrapSecret.length < 24 || bootstrapSecret.length > 256) {
    throw new LoopbackSessionError("Bootstrap token is missing or malformed.", "invalid_bootstrap_token");
  }

  let sessionCookie = "";
  let csrfToken = "";
  let closed = false;

  const request = async (relativeUrl, options = {}) => {
    const target = new URL(relativeUrl, origin);
    if (target.origin !== origin) {
      throw new LoopbackSessionError("Authenticated requests must remain on the bootstrap origin.", "cross_origin_request");
    }
    const method = String(options.method ?? "GET").toUpperCase();
    const mutation = !SAFE_METHODS.has(method);
    if (mutation && !csrfToken) {
      throw new LoopbackSessionError("Mutation refused because CSRF state is unavailable.", "csrf_unavailable");
    }
    if (closed || !sessionCookie) {
      throw new LoopbackSessionError("The local session is closed.", "session_closed");
    }
    const headers = new Headers(options.headers);
    if (headers.has("cookie") || headers.has("origin") || headers.has("x-evomind-csrf")) {
      throw new LoopbackSessionError("Authentication headers are owned by the session helper.", "reserved_auth_header");
    }
    headers.set("Cookie", sessionCookie);
    let body = options.body;
    if (mutation) {
      headers.set("Origin", origin);
      headers.set("x-evomind-csrf", csrfToken);
      if (body === undefined) body = "{}";
      if (!headers.has("content-type") && typeof body === "string") {
        headers.set("Content-Type", "application/json");
      }
    }
    return fetchImpl(target, { ...options, body, cache: "no-store", redirect: "error", method, headers });
  };

  try {
    const bootstrapBody = JSON.stringify({ token: bootstrapSecret });
    bootstrapSecret = "";
    const bootstrap = await fetchImpl(new URL("/api/session/bootstrap", origin), {
      method: "POST",
      headers: { "Content-Type": "application/json", Origin: origin },
      body: bootstrapBody,
      cache: "no-store",
      redirect: "error",
    });
    const bootstrapPayload = await json(bootstrap, "malformed_bootstrap_response");
    if (bootstrap.status !== 200 || bootstrapPayload.ok !== true || typeof bootstrapPayload.csrf_token !== "string") {
      throw new LoopbackSessionError("Local session bootstrap was rejected.", "bootstrap_rejected");
    }
    sessionCookie = cookieFromHeader(bootstrap.headers.get("set-cookie"));
    csrfToken = bootstrapPayload.csrf_token;
    if (csrfToken.length < 24 || csrfToken.length > 256) {
      throw new LoopbackSessionError("Session bootstrap returned malformed CSRF state.", "invalid_csrf_token");
    }

    const status = await request("/api/session/status");
    const statusPayload = await json(status, "malformed_session_status");
    if (
      status.status !== 200
      || statusPayload.ok !== true
      || statusPayload.authenticated !== true
      || statusPayload.csrf_token !== csrfToken
    ) {
      throw new LoopbackSessionError("Authenticated session status verification failed.", "session_status_rejected");
    }
  } catch (error) {
    sessionCookie = "";
    csrfToken = "";
    closed = true;
    throw error;
  } finally {
    bootstrapSecret = "";
  }

  return Object.freeze({
    origin,
    request,
    close() {
      csrfToken = "";
      sessionCookie = "";
      closed = true;
    },
  });
}
