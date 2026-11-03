import { createHash, randomBytes, randomInt, timingSafeEqual } from "node:crypto";

/**
 * Login graphic captcha (preprod contract, `scripts/captcha_login_probe.py`): `GET /api/auth/captcha` issues
 * `{ captcha_id, image }` where `image` is an SVG data URL whose `<text>` nodes spell the answer; the login POST
 * sends `captcha_id` + `captcha_answer`. Challenges are single-use (consumed on success and failure) and expire
 * after 5 minutes. Process-local store, same as the public-auth rate limiter.
 */
export const CAPTCHA_TTL_MS = 5 * 60 * 1000;
export const CAPTCHA_LENGTH = 5;
export const MAX_PENDING_CAPTCHAS = 4096;
export const CAPTCHA_ERROR_MESSAGE = "验证码错误或已过期，请刷新后重试。";

const STORE_KEY = Symbol.for("evomind.public.auth.login-captcha.v1");
const ID_PATTERN = /^[A-Za-z0-9_-]{22,64}$/;
const WIDTH = 124;
const HEIGHT = 46;
const INK = ["#2F855A", "#18211C", "#3E7C5C", "#22302A"];

type Challenge = { digest: Buffer; expiresAt: number };
type CaptchaGlobal = typeof globalThis & { [STORE_KEY]?: Map<string, Challenge> };

function store(now: number) {
  const state = globalThis as CaptchaGlobal;
  state[STORE_KEY] ??= new Map<string, Challenge>();
  const pending = state[STORE_KEY];
  for (const [id, challenge] of pending) if (challenge.expiresAt <= now) pending.delete(id);
  return pending;
}

function digestOf(answer: string) {
  return createHash("sha256").update(answer.trim().toUpperCase(), "utf8").digest();
}

const between = (min: number, max: number) => min + randomInt(0, 1000) / 1000 * (max - min);
const n = (value: number) => value.toFixed(1);

/** Noise is paths/circles only: every `<text>` node is one answer character, in order. */
export function renderCaptchaSvg(answer: string) {
  const parts = [`<rect width="${WIDTH}" height="${HEIGHT}" fill="#F4F7F5"/>`];
  for (let i = 0; i < 4; i += 1) {
    const points = Array.from({ length: 4 }, () => `${n(between(0, WIDTH))} ${n(between(0, HEIGHT))}`);
    parts.push(`<path d="M${points[0]} C${points.slice(1).join(" ")}" fill="none" stroke="rgba(62,124,92,.45)" stroke-width="${n(between(1, 2))}"/>`);
  }
  for (let i = 0; i < 30; i += 1) {
    parts.push(`<circle cx="${n(between(0, WIDTH))}" cy="${n(between(0, HEIGHT))}" r="1" fill="rgba(40,70,55,${between(0.1, 0.35).toFixed(2)})"/>`);
  }
  const step = (WIDTH - 22) / answer.length;
  [...answer].forEach((character, index) => {
    const x = n(9 + step * index + between(0, 4));
    const y = n(between(31, 37));
    parts.push(`<text x="${x}" y="${y}" transform="rotate(${n(between(-18, 18))} ${x} ${y})" fill="${INK[index % INK.length]}" font-family="Arial, sans-serif" font-size="${randomInt(22, 27)}" font-weight="700">${character}</text>`);
  });
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${WIDTH}" height="${HEIGHT}" viewBox="0 0 ${WIDTH} ${HEIGHT}">${parts.join("")}</svg>`;
}

export function issueLoginCaptcha(now = Date.now()) {
  const pending = store(now);
  while (pending.size >= MAX_PENDING_CAPTCHAS) {
    const oldest = pending.keys().next().value;
    if (oldest === undefined) break;
    pending.delete(oldest);
  }
  const answer = Array.from({ length: CAPTCHA_LENGTH }, () => String(randomInt(0, 10))).join("");
  const captchaId = randomBytes(18).toString("base64url");
  pending.set(captchaId, { digest: digestOf(answer), expiresAt: now + CAPTCHA_TTL_MS });
  return { captcha_id: captchaId, image: `data:image/svg+xml,${encodeURIComponent(renderCaptchaSvg(answer))}` };
}

/** Consumes the challenge whatever the outcome; fails closed on unknown, expired or malformed input. */
export function verifyLoginCaptcha(captchaId: unknown, answer: unknown, now = Date.now()) {
  if (typeof captchaId !== "string" || !ID_PATTERN.test(captchaId)) return false;
  const pending = store(now);
  const challenge = pending.get(captchaId);
  pending.delete(captchaId);
  if (!challenge || challenge.expiresAt <= now) return false;
  if (typeof answer !== "string" || !answer.trim() || answer.length > 16) return false;
  return timingSafeEqual(challenge.digest, digestOf(answer));
}

export function pendingCaptchaCount(now = Date.now()) {
  return store(now).size;
}
