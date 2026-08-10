import { createHash, createHmac, timingSafeEqual } from "node:crypto";

export const SESSION_COOKIE = "evomind_local_session";
export const CSRF_HEADER = "x-evomind-csrf";
export const LOCAL_AUTOMATION_VERIFIED_HEADER = "x-evomind-local-automation-verified";
export const SESSION_MAX_AGE_SECONDS = 12 * 60 * 60;

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

export function expectedCsrfToken(sessionCookie = expectedSessionCookie()) {
  return createHmac("sha256", secret()).update(`evomind.local.csrf.v1:${sessionCookie}`).digest("base64url");
}

export function validSessionCookie(value: string | undefined) {
  return typeof value === "string" && safeEqual(value, expectedSessionCookie());
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
