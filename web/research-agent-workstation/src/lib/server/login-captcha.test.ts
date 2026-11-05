import assert from "node:assert/strict";
import test from "node:test";

// @ts-ignore Node's strip-types test runner requires explicit TypeScript suffixes.
import { CAPTCHA_TTL_MS, MAX_PENDING_CAPTCHAS, issueLoginCaptcha, pendingCaptchaCount, verifyLoginCaptcha } from "./login-captcha.ts";

/** Same extraction as /workspace/evomind-preprod-20261101/scripts/captcha_login_probe.py. */
function answerOf(image: string) {
  assert.match(image, /^data:image\/svg\+xml,/);
  const svg = decodeURIComponent(image.split(",", 2)[1]);
  assert.match(svg, /^<svg[^>]*xmlns="http:\/\/www\.w3\.org\/2000\/svg"/);
  return [...svg.matchAll(/<text\b[^>]*>([^<]+)<\/text>/g)].map((match) => match[1]).join("");
}

test("captcha issue → verify ok once; the SVG text nodes are the whole answer", () => {
  const challenge = issueLoginCaptcha(1_000);
  assert.match(challenge.captcha_id, /^[A-Za-z0-9_-]{22,}$/);
  const answer = answerOf(challenge.image);
  assert.match(answer, /^[0-9]{5}$/);
  assert.equal(verifyLoginCaptcha(challenge.captcha_id, ` ${answer} `, 2_000), true);
  assert.equal(verifyLoginCaptcha(challenge.captcha_id, answer, 2_000), false, "single use: reuse fails");
});

test("wrong answer fails and consumes the challenge", () => {
  const challenge = issueLoginCaptcha(1_000);
  const answer = answerOf(challenge.image);
  const wrong = answer.replace(/^./, (digit) => String((Number(digit) + 1) % 10));
  assert.equal(verifyLoginCaptcha(challenge.captcha_id, wrong, 1_500), false);
  assert.equal(verifyLoginCaptcha(challenge.captcha_id, answer, 1_600), false, "consumed on failure");
});

test("expired, unknown and malformed challenges fail closed", () => {
  const challenge = issueLoginCaptcha(1_000);
  assert.equal(CAPTCHA_TTL_MS, 5 * 60 * 1000);
  assert.equal(verifyLoginCaptcha(challenge.captcha_id, answerOf(challenge.image), 1_000 + CAPTCHA_TTL_MS), false);
  const fresh = issueLoginCaptcha(1_000);
  assert.equal(verifyLoginCaptcha("nope", answerOf(fresh.image), 1_001), false);
  for (const [id, answer] of [[undefined, "12345"], [fresh.captcha_id, undefined], [fresh.captcha_id, ""], [42, 42], [fresh.captcha_id, "1".repeat(64)]]) {
    assert.equal(verifyLoginCaptcha(id, answer, 1_002), false, JSON.stringify([id, answer]));
  }
});

test("each refresh issues a new id and the pending store stays bounded", () => {
  const ids = new Set(Array.from({ length: 50 }, () => issueLoginCaptcha(5_000).captcha_id));
  assert.equal(ids.size, 50);
  for (let i = 0; i < MAX_PENDING_CAPTCHAS + 25; i += 1) issueLoginCaptcha(10_000);
  assert.ok(pendingCaptchaCount() <= MAX_PENDING_CAPTCHAS);
  const latest = issueLoginCaptcha(10_000);
  assert.equal(verifyLoginCaptcha(latest.captcha_id, answerOf(latest.image), 10_001), true, "newest challenge survives eviction");
});
