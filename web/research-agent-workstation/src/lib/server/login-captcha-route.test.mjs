// t237 C6e login-yeti T2: GET /api/auth/captcha + captcha-gated POST /api/auth/login
// (contract: /workspace/evomind-preprod-20261101/scripts/captcha_login_probe.py).
import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire, Module } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';

const require = createRequire(import.meta.url);
const resolve = Module._resolveFilename;
Module._resolveFilename = function(name, parent, ...args) {
  if (name.startsWith('@/')) name = fileURLToPath(new URL('../../' + name.slice(2), import.meta.url));
  return resolve.call(this, name, parent, ...args);
};
Module._extensions['.ts'] = (module, filename) => module._compile(ts.transpileModule(readFileSync(filename, 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true }, fileName: filename,
}).outputText, filename);

const { createPasswordHash } = require('./public-auth.ts');
process.env.WORKSTATION_SESSION_SECRET = 'isolated-login-captcha-session-key-not-a-real-secret';
process.env.WORKSTATION_ADMIN_USERNAME = 'captcha-admin';
process.env.WORKSTATION_ADMIN_PASSWORD_SCRYPT = createPasswordHash('captcha-fixture-password');
delete process.env.WORKSTATION_ACCOUNTS_DB;
const captchaRoute = require('../../app/api/auth/captcha/route.ts');
const { POST } = require('../../app/api/auth/login/route.ts');
const CAPTCHA_ERROR = '验证码错误或已过期，请刷新后重试。';

async function challenge() {
  const response = await captchaRoute.GET();
  assert.equal(response.status, 200);
  assert.match(response.headers.get('cache-control'), /no-store/);
  const body = await response.json();
  assert.deepEqual(Object.keys(body).sort(), ['captcha_id', 'image']);
  const svg = decodeURIComponent(body.image.split(',', 2)[1]);
  return { id: body.captcha_id, answer: [...svg.matchAll(/<text\b[^>]*>([^<]+)<\/text>/g)].map((m) => m[1]).join('') };
}

function login(body, client) {
  return POST(new Request('http://127.0.0.1/api/auth/login', { method: 'POST',
    headers: { 'Content-Type': 'application/json', 'x-forwarded-for': client }, body: JSON.stringify(body) }));
}
const creds = { username: 'captcha-admin', password: 'captcha-fixture-password', next: '/workspace?task=utask_1' };

test('GET /api/auth/captcha is dynamic and returns { captcha_id, image } with an SVG data URL', async () => {
  assert.equal(captchaRoute.dynamic, 'force-dynamic');
  const { id, answer } = await challenge();
  assert.match(id, /^[A-Za-z0-9_-]{22,}$/);
  assert.match(answer, /^\d{5}$/);
});

test('login with the probe-parsed captcha answer succeeds and keeps the CSRF / next contract', async () => {
  const { id, answer } = await challenge();
  const response = await login({ ...creds, captcha_id: id, captcha_answer: answer }, '10.0.0.1');
  assert.equal(response.status, 200);
  const body = await response.json();
  assert.equal(body.ok, true);
  assert.match(body.csrf_token, /\S{16,}/);
  assert.equal(body.redirect_to, '/workspace?task=utask_1');
  assert.match(response.headers.get('set-cookie'), /^evomind/);
});

test('missing / wrong / reused captcha → 400 Chinese message before any credential check', async () => {
  for (const extra of [{}, { captcha_id: 'missing' }, { captcha_answer: '12345' }]) {
    const response = await login({ ...creds, ...extra }, '10.0.0.2');
    assert.equal(response.status, 400);
    assert.deepEqual(await response.json(), { ok: false, code: 'captcha_invalid', message: CAPTCHA_ERROR });
  }
  const { id, answer } = await challenge();
  const wrong = answer.replace(/^./, (digit) => String((Number(digit) + 1) % 10));
  assert.equal((await login({ ...creds, captcha_id: id, captcha_answer: wrong }, '10.0.0.2')).status, 400);
  const retried = await login({ ...creds, captcha_id: id, captcha_answer: answer }, '10.0.0.2');
  assert.equal(retried.status, 400, 'a failed challenge is consumed');
  assert.equal((await retried.json()).message, CAPTCHA_ERROR);
});

test('wrong password with a valid captcha → 401 non-leaking copy; the captcha cannot be replayed', async () => {
  const { id, answer } = await challenge();
  const response = await login({ ...creds, password: 'wrong-password', captcha_id: id, captcha_answer: answer }, '10.0.0.3');
  assert.equal(response.status, 401);
  assert.equal((await response.json()).message, '账号或密码不正确');
  assert.equal((await login({ ...creds, captcha_id: id, captcha_answer: answer }, '10.0.0.3')).status, 400);
});

test('existing rate limit is unchanged: five wrong passwords (each with a fresh captcha) → 429', async () => {
  const statuses = [];
  for (let i = 0; i < 6; i += 1) {
    const { id, answer } = await challenge();
    statuses.push((await login({ ...creds, password: `wrong-${i}`, captcha_id: id, captcha_answer: answer }, '10.0.0.4')).status);
  }
  assert.deepEqual(statuses, [401, 401, 401, 401, 429, 429]);
  const { id, answer } = await challenge();
  assert.equal((await login({ ...creds, captcha_id: id, captcha_answer: answer }, '10.0.0.4')).status, 429, 'blocked even with the right password');
});
