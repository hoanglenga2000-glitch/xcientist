// t237 C6e login-yeti: Proto-1 light login shell + Yeti mascot (/workspace/login-mocks/DESIGN-proto1-yeti.md).
import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire, Module } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

const require = createRequire(import.meta.url);
const originalResolve = Module._resolveFilename;
Module._resolveFilename = function (name, parent, ...args) {
  if (name.startsWith('@/')) name = fileURLToPath(new URL('../../' + name.slice(2), import.meta.url));
  return originalResolve.call(this, name, parent, ...args);
};
for (const ext of ['.ts', '.tsx']) Module._extensions[ext] = (module, filename) => {
  module._compile(ts.transpileModule(readFileSync(filename, 'utf8'), { compilerOptions: {
    module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022, esModuleInterop: true,
  }, fileName: filename }).outputText, filename);
};

const yetiSource = readFileSync(new URL('./LoginYeti.tsx', import.meta.url), 'utf8');
const pageSource = readFileSync(new URL('../../app/login/page.tsx', import.meta.url), 'utf8');
const yeti = () => require('./LoginYeti.tsx');
const loginHtml = () => renderToStaticMarkup(React.createElement(require('../../app/login/page.tsx').default));

function settle(input, frames = 120) {
  const { RESTING_YETI, stepYeti } = yeti();
  let motion = RESTING_YETI;
  for (let i = 0; i < frames; i += 1) motion = stepYeti(motion, input);
  return motion;
}
const translateY = (transform) => Number(/translate\([^,]+,\s*(-?[\d.]+)\)/.exec(transform)[1]);

test('login-yeti A1: decorative Yeti SVG from proto-1 renders static on the server with hands down', () => {
  const { LoginYeti } = yeti();
  const html = renderToStaticMarkup(React.createElement(LoginYeti, { passwordFocused: true, passwordVisible: false }));
  assert.match(html, /<svg[^>]*viewBox="0 0 200 220"[^>]*aria-hidden="true"/);
  assert.equal((html.match(/data-yeti-pupil/g) || []).length, 2);
  assert.match(html, /data-yeti-hand="left"[^>]*transform="translate\(0, 0\)/);
  assert.match(html, /data-yeti-hand="right"[^>]*transform="translate\(0, 0\)/);
  assert.match(html, /rgb\(var\(--color-success/);
  assert.doesNotMatch(yetiSource, /#[0-9A-Fa-f]{3,8}/, 'LoginYeti must use semantic tokens, not raw hex');
  assert.doesNotMatch(html, /<text/);
});

test('login-yeti A2: password focus covers both eyes; show-password peeks with the right hand lower', () => {
  const { handTransforms, pupilOffset } = yeti();
  const covered = settle({ covering: true, peeking: false, dx: 300, dy: 0 });
  assert.ok(covered.cover > 0.99 && covered.peek < 0.01);
  const coverHands = handTransforms(covered);
  assert.ok(translateY(coverHands.left) < -120 && translateY(coverHands.right) < -120, JSON.stringify(coverHands));
  assert.ok(Math.abs(pupilOffset(covered)[0]) < 0.1, 'pupils are hidden/dampened while covered');

  const peeking = settle({ covering: false, peeking: true, dx: 300, dy: 0 });
  const peekHands = handTransforms(peeking);
  assert.ok(translateY(peekHands.left) < translateY(peekHands.right), 'left stays high, right hand drops to peek');
  assert.ok(pupilOffset(peeking)[0] > 3, 'pupils track again while peeking');

  const idle = settle({ covering: false, peeking: false, dx: -400, dy: 0 });
  assert.equal(translateY(handTransforms(idle).left), 0);
  assert.ok(pupilOffset(idle)[0] < -5, 'pupils follow the pointer when idle');
});

test('login-yeti A3: prefers-reduced-motion keeps a static Yeti and every loop/listener is torn down', () => {
  assert.match(yetiSource, /^"use client";/);
  assert.match(yetiSource, /matchMedia\("\(prefers-reduced-motion: reduce\)"\)/);
  assert.match(yetiSource, /requestAnimationFrame/);
  assert.match(yetiSource, /cancelAnimationFrame/);
  assert.match(yetiSource, /removeEventListener\("mousemove"/);
  assert.match(yetiSource, /clearTimeout/);
  assert.doesNotMatch(yetiSource, /from "(?!react|@\/)[^"]+"/, 'no new packages');
});

test('login-yeti B1: /login is the light Proto-1 card — no dark glass, empty username, DeepEvo brand top-left', () => {
  const html = loginHtml();
  assert.match(html, /bg-\[#F5F8F6\]/);
  assert.match(html, /rounded-\[20px\]/);
  assert.match(html, /data-brand-mark="(small|master)"/);
  assert.match(html, />DeepEvo</);
  assert.match(html, /深度进化/);
  assert.match(html, /viewBox="0 0 200 220"/);
  assert.match(html, /<input[^>]*id="username"[^>]*value=""/);
  assert.doesNotMatch(html, /value="admin"/);
  assert.match(html, /placeholder="请输入账号"/);
  assert.match(html, /placeholder="请输入密码"/);
  assert.match(html, /aria-label="显示密码"/);
  assert.match(html, /<h1[^>]*>登录<\/h1>/);
  assert.match(html, /<button[^>]*type="submit"[^>]*>登录<\/button>/);
  assert.doesNotMatch(html, /进入工作台|正在验证/);
  for (const banned of ['#070b14', 'indigo', 'cyan', 'backdrop-blur', 'bg-gradient', 'LockKeyhole']) assert.ok(!pageSource.includes(banned), banned);
  assert.doesNotMatch(html, /deepevo-mark text-white/);
  assert.match(pageSource, /evomind\.local\.csrf\.v1/);
  assert.match(pageSource, /credentials: "same-origin"/);
  assert.match(pageSource, /cache: "no-store"/);
  assert.match(pageSource, /get\("next"\)/);
  assert.match(pageSource, /登录中…/);
});

test('login-yeti C1: 验证码 row is wired to the real BE challenge — no client-side generator', () => {
  const html = loginHtml();
  assert.match(html, /<label[^>]*for="captcha"[^>]*>验证码<\/label>/);
  assert.match(html, /<input[^>]*id="captcha"[^>]*placeholder="请输入验证码"/);
  assert.match(html, /<button[^>]*type="button"[^>]*aria-label="刷新验证码"/);
  assert.doesNotMatch(html, /<img[^>]*src=""/, 'no empty image before the challenge loads');
  assert.match(pageSource, /fetch\("\/api\/auth\/captcha"/);
  assert.match(pageSource, /alt="验证码"/);
  assert.match(pageSource, /captcha_id: captchaId/);
  assert.match(pageSource, /captcha_answer: captchaAnswer/);
  assert.match(pageSource, /captcha_invalid/);
  assert.doesNotMatch(pageSource, /canvas|getContext|Math\.random/, 'no forgeable client captcha');
  assert.match(pageSource, /请输入验证码/);
});
