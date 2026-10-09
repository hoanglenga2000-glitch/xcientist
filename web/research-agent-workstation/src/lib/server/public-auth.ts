import { randomBytes, scrypt, scryptSync, timingSafeEqual } from "node:crypto";

import { tenantIdForUsername } from "./tenant-identity.mjs";
import { accountRegistryEnabled, findAccount } from "./account-registry.mjs";

const USERNAME_PATTERN = /^[A-Za-z0-9._-]{1,64}$/;
const SCRYPT_PATTERN = /^scrypt\$(\d+)\$(\d+)\$(\d+)\$([a-f0-9]{32})\$([a-f0-9]{64})$/;
const RATE_LIMIT_KEY = Symbol.for("evomind.public.auth.rate-limit.v2");
const PASSWORD_HASH_KEY = Symbol.for("evomind.public.auth.password-hash.v2");
const USERNAME_KEY = Symbol.for("evomind.public.auth.username.v2");
const WINDOW_MS = 15 * 60 * 1000;
const MAX_FAILURES = 5;
const MAX_ATTEMPT_KEYS = 4096;
const MAX_CONCURRENT_DERIVATIONS = 4;
const DUMMY_HASH = `scrypt$16384$8$1$${"0".repeat(32)}$${"0".repeat(64)}`;
let activeDerivations = 0;

type Attempt = { failures: number; blockedUntil: number; lastFailureAt: number };
type AuthGlobal = typeof globalThis & {
  [RATE_LIMIT_KEY]?: Map<string, Attempt>;
  [PASSWORD_HASH_KEY]?: string;
  [USERNAME_KEY]?: string;
};

function safeEqual(left: string, right: string) {
  const a = Buffer.from(left, "utf8");
  const b = Buffer.from(right, "utf8");
  return a.length === b.length && timingSafeEqual(a, b);
}

function configuredUsername() {
  const state = globalThis as AuthGlobal;
  const value = state[USERNAME_KEY] ?? process.env.WORKSTATION_ADMIN_USERNAME?.trim() ?? "admin";
  if (!USERNAME_PATTERN.test(value)) throw new Error("WORKSTATION_ADMIN_USERNAME is invalid");
  state[USERNAME_KEY] = value;
  delete process.env.WORKSTATION_ADMIN_USERNAME;
  return value;
}

function configuredHash() {
  const state = globalThis as AuthGlobal;
  const value = state[PASSWORD_HASH_KEY] ?? process.env.WORKSTATION_ADMIN_PASSWORD_SCRYPT?.trim() ?? "";
  if (!SCRYPT_PATTERN.test(value)) throw new Error("WORKSTATION_ADMIN_PASSWORD_SCRYPT is missing or invalid");
  state[PASSWORD_HASH_KEY] = value;
  delete process.env.WORKSTATION_ADMIN_PASSWORD_SCRYPT;
  return value;
}

async function derive(password: string, encoded: string) {
  const match = SCRYPT_PATTERN.exec(encoded);
  if (!match) throw new Error("Invalid password hash");
  const N = Number(match[1]);
  const r = Number(match[2]);
  const p = Number(match[3]);
  if (N !== 16384 || r !== 8 || p !== 1) throw new Error("Unsupported scrypt parameters");
  if (activeDerivations >= MAX_CONCURRENT_DERIVATIONS) throw new Error("AUTH_DERIVATION_BUSY");
  activeDerivations += 1;
  try {
    return await new Promise<string>((resolve, reject) => {
      scrypt(password, Buffer.from(match[4], "hex"), 32, { N, r, p, maxmem: 64 * 1024 * 1024 }, (error, key) => {
        if (error) reject(error);
        else resolve(key.toString("hex"));
      });
    });
  } finally {
    activeDerivations -= 1;
  }
}

function attempts(now = Date.now()) {
  const state = globalThis as AuthGlobal;
  state[RATE_LIMIT_KEY] ??= new Map<string, Attempt>();
  for (const [key, entry] of state[RATE_LIMIT_KEY]) {
    if (entry.blockedUntil <= now && now - entry.lastFailureAt >= WINDOW_MS) state[RATE_LIMIT_KEY].delete(key);
  }
  while (state[RATE_LIMIT_KEY].size >= MAX_ATTEMPT_KEYS) {
    const oldest = [...state[RATE_LIMIT_KEY].entries()].sort((left, right) => left[1].lastFailureAt - right[1].lastFailureAt)[0]?.[0];
    if (!oldest) break;
    state[RATE_LIMIT_KEY].delete(oldest);
  }
  return state[RATE_LIMIT_KEY];
}

export function createPasswordHash(password: string, salt = randomBytes(16)) {
  if (!password || password.length > 256) throw new Error("Password length is invalid");
  const N = 16384;
  const r = 8;
  const p = 1;
  const digest = scryptSync(password, salt, 32, { N, r, p, maxmem: 64 * 1024 * 1024 }).toString("hex");
  return `scrypt$${N}$${r}$${p}$${salt.toString("hex")}$${digest}`;
}

export function publicAuthStatus(rateKey: string, now = Date.now()) {
  const entry = attempts(now).get(rateKey);
  if (!entry || entry.blockedUntil <= now) return { blocked: false, retryAfterSeconds: 0 };
  return { blocked: true, retryAfterSeconds: Math.max(1, Math.ceil((entry.blockedUntil - now) / 1000)) };
}

export async function verifyPublicCredentials(username: string, password: string, rateKey: string, now = Date.now()) {
  const rate = publicAuthStatus(rateKey, now);
  if (rate.blocked) return { ok: false as const, code: "rate_limited", ...rate };
  const managed = accountRegistryEnabled();
  const account = managed ? findAccount(username.trim()) : null;
  const expectedUsername = managed ? account?.username ?? "" : configuredUsername();
  const encoded = managed ? account?.password_hash ?? DUMMY_HASH : configuredHash();
  let actualDigest: string;
  try {
    actualDigest = await derive(password.slice(0, 256), encoded);
  } catch (error) {
    if (error instanceof Error && error.message === "AUTH_DERIVATION_BUSY") {
      return { ok: false as const, code: "rate_limited", blocked: true, retryAfterSeconds: 1 };
    }
    throw error;
  }
  const expectedDigest = SCRYPT_PATTERN.exec(encoded)![5];
  const valid = password.length <= 256
    && (!managed || account?.enabled === 1)
    && safeEqual(username.trim().toLowerCase(), expectedUsername.toLowerCase())
    && safeEqual(actualDigest, expectedDigest);
  if (valid) {
    attempts(now).delete(rateKey);
    return { ok: true as const, username: expectedUsername, tenantId: account?.tenant_id ?? tenantIdForUsername(expectedUsername),
      sessionVersion: account?.session_version ?? 0 };
  }
  const previous = attempts(now).get(rateKey);
  const failures = previous && now - previous.lastFailureAt < WINDOW_MS ? previous.failures + 1 : 1;
  attempts(now).set(rateKey, {
    failures,
    lastFailureAt: now,
    blockedUntil: failures >= MAX_FAILURES ? now + WINDOW_MS : 0,
  });
  return {
    ok: false as const,
    code: failures >= MAX_FAILURES ? "rate_limited" : "invalid_credentials",
    blocked: failures >= MAX_FAILURES,
    retryAfterSeconds: failures >= MAX_FAILURES ? WINDOW_MS / 1000 : 0,
  };
}

export function safePostLoginPath(value: unknown) {
  const candidate = typeof value === "string" ? value.trim() : "";
  const hasControlCharacter = [...candidate].some((character) => character.charCodeAt(0) < 32);
  if (!candidate || !candidate.startsWith("/") || candidate.startsWith("//") || candidate.includes("\\") || hasControlCharacter) {
    return "/?page=assistant";
  }
  try {
    const parsed = new URL(candidate, "https://evomind.invalid");
    return parsed.origin === "https://evomind.invalid" ? `${parsed.pathname}${parsed.search}${parsed.hash}` : "/?page=assistant";
  } catch {
    return "/?page=assistant";
  }
}
