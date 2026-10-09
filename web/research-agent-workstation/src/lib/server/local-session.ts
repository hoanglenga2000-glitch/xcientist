import { createHash, createHmac, randomUUID, timingSafeEqual } from "node:crypto";
import { accountRegistryEnabled, findAccount } from "./account-registry.mjs";

export const SESSION_COOKIE = "evomind_local_session";
export const CSRF_HEADER = "x-evomind-csrf";
export const LOCAL_AUTOMATION_VERIFIED_HEADER = "x-evomind-local-automation-verified";
export const SESSION_MAX_AGE_SECONDS = 12 * 60 * 60;

export type SessionPrincipal = {
  username: string;
  tenantId: string | null;
  authentication: "local_bootstrap" | "public_password";
  issuedAt: number | null;
  expiresAt: number | null;
  role?: "admin" | "user";
  sessionVersion?: number;
};

const SECRET_KEY = Symbol.for("evomind.local.session.secret.v1");
const BOOTSTRAP_HASH_KEY = Symbol.for("evomind.local.bootstrap.hash.v1");
const AUTOMATION_HASH_KEY = Symbol.for("evomind.local.automation.hash.v1");
type LocalSecretGlobal = typeof globalThis & {
  [SECRET_KEY]?: string;
  [BOOTSTRAP_HASH_KEY]?: string;
  [AUTOMATION_HASH_KEY]?: string;
};

function secret() {
  const state = globalThis as LocalSecretGlobal;
  const value = state[SECRET_KEY] ?? process.env.WORKSTATION_SESSION_SECRET?.trim();
  if (!value || value.length < 32) {
    throw new Error("WORKSTATION_SESSION_SECRET is missing or too short");
  }
  state[SECRET_KEY] = value;
  // Keep authentication material in this Node process only. Any Python/Node
  // tool process spawned later inherits process.env, not this private symbol.
  delete process.env.WORKSTATION_SESSION_SECRET;
  return value;
}

function safeEqual(left: string, right: string) {
  const a = Buffer.from(left, "utf8");
  const b = Buffer.from(right, "utf8");
  return a.length === b.length && timingSafeEqual(a, b);
}

export function sha256Hex(value: string) {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

export function expectedSessionCookie() {
  return createHmac("sha256", secret()).update("evomind.local.session.v1").digest("base64url");
}

function publicSessionSignature(payload: string) {
  return createHmac("sha256", secret()).update(`evomind.public.session.v2:${payload}`).digest("base64url");
}

export function issuePublicSessionCookie(username: string, tenantId: string, nowSeconds = Math.floor(Date.now() / 1000), sessionVersion = 0) {
  if (!/^[A-Za-z0-9._-]{1,64}$/.test(username)) throw new Error("invalid username");
  if (!/^tenant_[a-f0-9]{24}$/.test(tenantId)) throw new Error("invalid tenant id");
  const payload = Buffer.from(JSON.stringify({
    typ: "public",
    sub: username,
    tid: tenantId,
    iat: nowSeconds,
    exp: nowSeconds + SESSION_MAX_AGE_SECONDS,
    jti: randomUUID(),
    av: sessionVersion,
  }), "utf8").toString("base64url");
  return `v2.${payload}.${publicSessionSignature(payload)}`;
}

export function sessionPrincipal(value: string | undefined, nowSeconds = Math.floor(Date.now() / 1000)): SessionPrincipal | null {
  if (typeof value !== "string" || value.length < 20 || value.length > 2048) return null;
  if (safeEqual(value, expectedSessionCookie())) {
    if (accountRegistryEnabled()) return null;
    return { username: "local", tenantId: null, authentication: "local_bootstrap", issuedAt: null, expiresAt: null };
  }
  const match = /^v2\.([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]+)$/.exec(value);
  if (!match || !safeEqual(match[2], publicSessionSignature(match[1]))) return null;
  try {
    const payload = JSON.parse(Buffer.from(match[1], "base64url").toString("utf8")) as Record<string, unknown>;
    const username = typeof payload.sub === "string" ? payload.sub : "";
    const tenantId = typeof payload.tid === "string" ? payload.tid : "";
    const issuedAt = Number(payload.iat);
    const expiresAt = Number(payload.exp);
    const jti = typeof payload.jti === "string" ? payload.jti : "";
    if (
      payload.typ !== "public"
      || !/^[A-Za-z0-9._-]{1,64}$/.test(username)
      || !/^tenant_[a-f0-9]{24}$/.test(tenantId)
      || !Number.isSafeInteger(issuedAt)
      || !Number.isSafeInteger(expiresAt)
      || issuedAt > nowSeconds + 60
      || expiresAt <= nowSeconds
      || expiresAt - issuedAt !== SESSION_MAX_AGE_SECONDS
      || !/^[0-9a-f-]{36}$/i.test(jti)
    ) return null;
    if (accountRegistryEnabled()) {
      const account = findAccount(username);
      if (!account || account.enabled !== 1 || account.username !== username || account.tenant_id !== tenantId
        || account.session_version !== payload.av) return null;
      return { username, tenantId, authentication: "public_password", issuedAt, expiresAt,
        role: account.role, sessionVersion: account.session_version };
    }
    return { username, tenantId, authentication: "public_password", issuedAt, expiresAt };
  } catch {
    return null;
  }
}

export function expectedCsrfToken(sessionCookie = expectedSessionCookie()) {
  return createHmac("sha256", secret()).update(`evomind.local.csrf.v1:${sessionCookie}`).digest("base64url");
}

export function validSessionCookie(value: string | undefined) {
  return sessionPrincipal(value) !== null;
}

export function validCsrfToken(value: string | null, sessionCookie: string) {
  return typeof value === "string" && safeEqual(value, expectedCsrfToken(sessionCookie));
}

export function validBootstrapToken(value: string) {
  const state = globalThis as LocalSecretGlobal;
  const expectedHash = state[BOOTSTRAP_HASH_KEY]
    ?? process.env.WORKSTATION_BOOTSTRAP_TOKEN_HASH?.trim().toLowerCase();
  if (expectedHash) state[BOOTSTRAP_HASH_KEY] = expectedHash;
  delete process.env.WORKSTATION_BOOTSTRAP_TOKEN_HASH;
  return Boolean(expectedHash && /^[a-f0-9]{64}$/.test(expectedHash) && safeEqual(sha256Hex(value), expectedHash));
}

export function validLocalAutomationToken(value: string | null) {
  const token = typeof value === "string" ? value.trim() : "";
  if (!token) return false;
  const state = globalThis as LocalSecretGlobal;
  const expectedHash = state[AUTOMATION_HASH_KEY]
    ?? process.env.WORKSTATION_LOCAL_AUTOMATION_TOKEN_HASH?.trim().toLowerCase();
  if (expectedHash) state[AUTOMATION_HASH_KEY] = expectedHash;
  delete process.env.WORKSTATION_LOCAL_AUTOMATION_TOKEN_HASH;
  return Boolean(expectedHash && /^[a-f0-9]{64}$/.test(expectedHash) && safeEqual(sha256Hex(token), expectedHash));
}

export function localOrigin() {
  const host = process.env.HOSTNAME === "localhost" ? "localhost" : "127.0.0.1";
  const port = process.env.PORT?.trim() || "8088";
  return `http://${host}:${port}`;
}

export function cookieOptions() {
  return {
    httpOnly: true,
    sameSite: "strict" as const,
    secure: process.env.WORKSTATION_LOCAL_HTTPS === "1",
    path: "/",
    maxAge: SESSION_MAX_AGE_SECONDS,
  };
}
