// t238 C7c-6 (U2/U3): tenant GPU state copy, routes, proxy scope and the task view mount.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire, Module } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
const require = createRequire(import.meta.url);
const app = (path) => fileURLToPath(new URL('../../../' + path, import.meta.url));
const originalResolve = Module._resolveFilename;
Module._resolveFilename = function (name, parent, ...args) {
  if (name.startsWith('@/')) name = app(name.slice(2));
  return originalResolve.call(this, name, parent, ...args);
};
Module._extensions['.ts'] = (module, filename) => {
  module._compile(ts.transpileModule(readFileSync(filename, 'utf8'), { compilerOptions: {
    module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true }, fileName: filename }).outputText, filename);
};
const scoped = new Module(app('lib/server/scoped-runtime-fetch.ts'));
scoped.filename = scoped.id; scoped.loaded = true;
scoped.exports = { scopedRuntimeFetch: (input, init) => globalThis.fetch(input, init) };
require.cache[scoped.id] = scoped;
const launcher = new Module(app('lib/server/evomind-runtime.ts'));
launcher.filename = launcher.id; launcher.loaded = true;
launcher.exports = { ensureEvoMindRuntime: async () => ({ baseUrl: 'http://runtime.test', token: 'fixture-token' }) };
require.cache[launcher.id] = launcher;
const read = (name) => readFileSync(new URL('./' + name, import.meta.url), 'utf8');
const NO_TECH = /98270|job\d|gjob_|glease_|GPU-|hpc2hdd|jinghw|aimslab|node|[A-Za-z_]{6,}|您/;

test('t238 C7c-6: each state renders one plain Chinese line, ETA labelled as an estimate', () => {
  const { gpuJobLine, GPU_COPY } = require('./gpu-status.ts');
  const lines = {
    queued: gpuJobLine({ run_id: 'r', state: 'queued', position: 2, eta_minutes: 14 }),
    running: gpuJobLine({ run_id: 'r', state: 'running', elapsed_minutes: 3, max_minutes: 30 }),
    succeeded: gpuJobLine({ run_id: 'r', state: 'succeeded' }), cancelled: gpuJobLine({ run_id: 'r', state: 'cancelled' }),
    timeout: gpuJobLine({ run_id: 'r', state: 'timeout' }), unavailable: gpuJobLine({ run_id: 'r', state: 'unavailable' }),
  };
  assert.deepEqual(lines, { queued: '排队中（第 2 位，预计等待约 14 分钟）', running: '训练中（已运行 3 分，最长 30 分）',
    succeeded: '已完成', cancelled: '已取消', timeout: '到时间了，已停止', unavailable: '暂不可用' });
  // never "0 位" / "0 分钟" / negative
  assert.equal(gpuJobLine({ run_id: 'r', state: 'queued', position: 0, eta_minutes: -3 }), '排队中（第 1 位，预计等待约 1 分钟）');
  assert.equal(gpuJobLine({ run_id: 'r', state: 'queued', label: '排队中（第 3 位，预计等待约 9 分钟）' }), '排队中（第 3 位，预计等待约 9 分钟）');
  assert.match(GPU_COPY.estimate, /估计/);
  assert.equal(GPU_COPY.budget(45, 120), '今天还可以训练 45 分钟（每天 120 分钟）');
  for (const line of [...Object.values(lines), GPU_COPY.estimate, GPU_COPY.cancelQueued, GPU_COPY.cancelRunning, GPU_COPY.cancelFailed]) assert.doesNotMatch(line, NO_TECH);
});

test('t238 C7c-6 U3: breaker / quota / queue full / VRAM / timeout map to a sentence with a next action', () => {
  const { gpuErrorLine, GPU_ERROR_COPY } = require('./gpu-status.ts');
  assert.deepEqual(Object.keys(GPU_ERROR_COPY).sort(), ['breaker_open', 'insufficient_gpu_memory', 'pool_unavailable', 'queue_full', 'quota_exhausted', 'quota_exhausted_rolling', 'timeout']);
  assert.equal(gpuErrorLine('breaker_open'), '训练暂时用不了，你的排队位置还在。恢复后在这个任务里重新发起就行。');
  assert.equal(gpuErrorLine('quota_exhausted'), '你今天的训练时长已用完，明天可以继续训练。');
  assert.equal(gpuErrorLine('queue_full'), '现在排队的训练太多，请等前面的训练结束后再提交。');
  assert.equal(gpuErrorLine('insufficient_gpu_memory'), '现在训练排满了，这次训练还没有开始。请稍后在这个任务里重新发起。');
  assert.equal(gpuErrorLine('timeout'), '这次训练超过了最长时间，已自动停止。可以缩小数据后重新开始。');
  assert.equal(gpuErrorLine('anything_else'), '训练暂时不可用，请稍后再试。');
  for (const { line } of Object.values(GPU_ERROR_COPY)) assert.doesNotMatch(line, NO_TECH);
});

test('t238 C7c-6: picks only this run\'s job; scheduler off ⇒ nothing rendered', () => {
  const { gpuJobForRun, isLiveGpuJob } = require('./gpu-status.ts');
  const payload = { enabled: true, jobs: [{ run_id: 'run_b', state: 'running' }, { run_id: 'run_a', state: 'queued' }], budget: null };
  assert.equal(gpuJobForRun(payload, 'run_a').state, 'queued');
  assert.equal(gpuJobForRun(payload, 'run_c'), null);
  assert.equal(gpuJobForRun({ enabled: false, jobs: [], budget: null }, 'run_a'), null);
  assert.equal(gpuJobForRun(null, 'run_a'), null);
  assert.equal(isLiveGpuJob({ run_id: 'x', state: 'running' }), true);
  assert.equal(isLiveGpuJob({ run_id: 'x', state: 'timeout' }), false);
});

test('t238 C7c-6: routes proxy to the runtime; cancel is a JSON POST; bad run ids never reach the runtime', async (context) => {
  const me = require('../../../app/api/assistant/gpu/me/route.ts');
  const cancel = require('../../../app/api/assistant/gpu/runs/[runId]/cancel/route.ts');
  const seen = [];
  context.mock.method(globalThis, 'fetch', async (url, init) => {
    seen.push(`${init?.method ?? 'GET'} ${String(url)} ${init?.body ? Buffer.from(init.body).toString() : ''}`.trim());
    if (String(url).includes('run_other')) return Response.json({ error: 'not_found' }, { status: 404 });
    return Response.json({ enabled: true, jobs: [], budget: null });
  });
  assert.equal((await me.GET(new Request('http://localhost/api/assistant/gpu/me?run_id=run_a'))).status, 200);
  assert.equal((await me.GET(new Request('http://localhost/api/assistant/gpu/me'))).status, 200);
  assert.equal((await me.GET(new Request('http://localhost/api/assistant/gpu/me?run_id=../x'))).status, 400);
  const post = () => new Request('http://localhost/x', { method: 'POST', body: '{}', headers: { 'content-type': 'application/json' } });
  assert.equal((await cancel.POST(post(), { params: Promise.resolve({ runId: 'run_a' }) })).status, 200);
  assert.equal((await cancel.POST(post(), { params: Promise.resolve({ runId: 'run_other' }) })).status, 404);
  assert.equal((await cancel.POST(post(), { params: Promise.resolve({ runId: 'a/b' }) })).status, 400);
  assert.deepEqual(seen, ['GET http://runtime.test/v1/gpu/me?run_id=run_a', 'GET http://runtime.test/v1/gpu/me',
    'POST http://runtime.test/v1/gpu/runs/run_a/cancel {}', 'POST http://runtime.test/v1/gpu/runs/run_other/cancel {}']);
});

test('t238 C7c-6: the proxy scopes /api/assistant/gpu/* as a personal route (CSRF + JSON body like other mutations)', () => {
  const proxy = readFileSync(new URL('../../../proxy.ts', import.meta.url), 'utf8');
  assert.match(proxy, /experience\|tenant\|gpu\)\(\\\/\|\$\)/);
  assert.match(proxy, /MUTATING_METHODS\.has\(request\.method\)/);
});

test('t238 C7c-6: the status card is mounted in the conversation and progress views, with a 44px cancel', () => {
  const view = read('TaskRunView.tsx');
  assert.equal((view.match(/<GpuJobStatus runId=\{run\.id\}/g) ?? []).length, 2);
  assert.equal((view.match(/hideSettled=\{confirmPending\(run\)\} onRetry=\{onRetryTraining\} onJob=\{onGpuJob\}/g) ?? []).length, 2);
  const card = read('GpuJobStatus.tsx');
  assert.match(card, /role="status"/);
  assert.match(card, /jsonPost\(\{\}\)/);
  assert.match(card, /job\.can_cancel && !atCap && <button/);
  assert.match(card, /data-testid="tw-gpu-cancel"/);
  assert.match(card, /role="alertdialog"/);
  assert.match(card, /data-testid="tw-gpu-cancel-confirm"/);
  assert.match(card, /onClick=\{\(\) => setConfirming\(true\)\}/);
  assert.doesNotMatch(card, /显存|显卡|VRAM|A800|vLLM|训练资源/);
  assert.match(card, /if \(!job \|\| /);
  const css = read('workspace.css');
  assert.match(css, /\.tw-gpu-cancel \{[^}]*min-height: 44px/);
});

test('t238 C7c-6: shots fixtures cover every state and stay free of technical details', () => {
  const { GPU_SHOT_FIXTURES } = require('./gpu-status.fixtures.ts');
  const { gpuJobForRun, gpuJobLine } = require('./gpu-status.ts');
  const states = new Set();
  for (const make of Object.values(GPU_SHOT_FIXTURES)) {
    const payload = make('run_shot');
    const job = gpuJobForRun(payload, 'run_shot');
    states.add(job.state);
    assert.doesNotMatch(gpuJobLine(job) + (job.notice ?? ''), NO_TECH);
  }
  assert.deepEqual([...states].sort(), ['cancelled', 'failed', 'queued', 'running', 'succeeded', 'timeout', 'unavailable']);
});

test('t238 C7c-FIX3: every tenant-visible GPU string avoids card/memory/GPU/serving words', () => {
  const { GPU_COPY, GPU_ERROR_COPY } = require('./gpu-status.ts');
  const { GPU_SHOT_FIXTURES } = require('./gpu-status.fixtures.ts');
  const strings = [];
  const walk = (value) => {
    if (typeof value === 'string') strings.push(value);
    else if (typeof value === 'function') { try { strings.push(String(value(45, 120))); } catch { /* not a copy fn */ } }
    else if (value && typeof value === 'object') Object.values(value).forEach(walk);
  };
  walk(GPU_COPY);
  walk(GPU_ERROR_COPY);
  walk(GPU_SHOT_FIXTURES);
  assert.ok(strings.length > 10);
  for (const text of strings) assert.doesNotMatch(text, /显存|显卡|VRAM|GPU|vLLM|A800|训练资源|腾出/i, text);
});

test('t238 C7c-FIX3b: unknown ETA shows no number; budget line names the per-run limit; near-cap warns; stop vs cancel wording', () => {
  const { gpuJobLine, GPU_COPY, isNearCap } = require('./gpu-status.ts');
  assert.equal(gpuJobLine({ run_id: 'r', state: 'queued', position: 2 }), '排队中（第 2 位）');
  assert.equal(gpuJobLine({ run_id: 'r', state: 'queued', position: 2, eta_minutes: 6 }), '排队中（第 2 位，预计等待约 6 分钟）');
  assert.equal(GPU_COPY.budget(75, 120, 30), '今天还可以训练 75 分钟（每天 120 分钟），每次最长 30 分钟');
  assert.equal(GPU_COPY.budget(75, 120), '今天还可以训练 75 分钟（每天 120 分钟）');
  assert.equal(isNearCap({ run_id: 'r', state: 'running', elapsed_minutes: 28, max_minutes: 30 }), true);
  assert.equal(isNearCap({ run_id: 'r', state: 'running', elapsed_minutes: 7, max_minutes: 30 }), false);
  assert.equal(isNearCap({ run_id: 'r', state: 'queued', elapsed_minutes: 28, max_minutes: 30 }), false);
  assert.equal(GPU_COPY.nearCap(30), '快到 30 分钟了，到点会自动停止。');
  assert.notEqual(GPU_COPY.cancelQueued, GPU_COPY.cancelRunning);
  const card = readFileSync(app('components/workstation/task-workspace/GpuJobStatus.tsx'), 'utf8');
  assert.match(card, /budget\.remaining_minutes > 0/);
  assert.match(card, /isNearCap\(job\)/);
});

// t238 C7c-FIX4: status words must follow what happened, and no refusal code reaches a tenant.
test('C7c-FIX4: a completed run whose training failed is not 已完成', () => {
  const p = require('./task-presentation.ts');
  const label = (status) => ({ completed: '执行已完成', blocked: '暂时无法继续', cancelled: '已取消', running: '正在处理' }[status] ?? status);
  const run = { id: 'r', status: 'completed', training_outcome: 'failed', started_at: '2026-11-02T01:00:00Z', completed_at: '2026-11-02T01:15:00Z', activity: [] };
  assert.equal(p.runStatusWord(run, label), '没有完成');
  assert.equal(p.runStatusWord(run, label, 'undelivered'), '没有完成');
  assert.equal(p.statusPillKind(p.runStatusWord(run, label)), 'error');
  assert.equal(p.thinkingRowState(run).phase, 'failed');
  assert.match(p.thinkingRowState(run).text, /^没有完成 · 用时 /);
  const stuck = p.streamPhaseState(run);
  assert.equal(stuck.stuck, true); assert.equal(stuck.done, false);
  assert.equal(p.stuckSurface(run), null, 'no second "我先检查再接着跑" line under the agent reply');
  // success and plain completions are unchanged
  assert.equal(p.runStatusWord({ ...run, training_outcome: 'succeeded' }, label), '已完成');
  assert.equal(p.runStatusWord({ ...run, training_outcome: '' }, label), '已完成');
  assert.equal(p.runStatusWord({ ...run, training_outcome: undefined }, label), '已完成');
  assert.equal(p.streamPhaseState({ ...run, training_outcome: 'succeeded' }).done, true);
});

test('C7c-FIX4: a rejected confirmation reads 已取消, not a red 这一步停住了', () => {
  const p = require('./task-presentation.ts');
  const label = (status) => ({ blocked: '暂时无法继续' }[status] ?? status);
  const rejected = { id: 'r', status: 'blocked', error_class: 'approval_rejected', started_at: '2026-11-02T01:00:00Z', updated_at: '2026-11-02T01:01:00Z', activity: [] };
  assert.equal(p.runStatusWord(rejected, label), '已取消');
  assert.equal(p.statusPillKind(p.runStatusWord(rejected, label)), 'idle');
  assert.equal(p.thinkingRowState(rejected).phase, 'stopped');
  assert.equal(p.blockedProgressCopy(rejected), null);
  assert.equal(p.stuckSurface(rejected), null);
  assert.equal(p.streamPhaseState(rejected).stuck, false);
  // an expired confirmation and any other blocked run keep the stuck copy
  assert.equal(p.runStatusWord({ ...rejected, error_class: 'approval_expired' }, label), '这一步停住了');
  assert.equal(p.runStatusWord({ ...rejected, error_class: '' }, label), '这一步停住了');
});

test('C7c-FIX4: a leaked refusal code becomes plain copy in the visible text', () => {
  const { visibleMessageText } = require('./task-presentation.ts');
  const text = visibleMessageText('训练没有开始：`tenant_training_identity_gate_not_ready`。tenant_training_environment_not_ready 也一样');
  assert.doesNotMatch(text, /tenant_training|identity_gate|_not_ready/);
  assert.match(text, /训练暂时不可用，请稍后再试。/);
  assert.equal(visibleMessageText('普通回答'), '普通回答');
});

test('t238 C7c-FIX4: a job that only holds its place while preparing is never "训练中" and never near the cap', () => {
  const { gpuJobLine, isNearCap } = require('./gpu-status.ts');
  assert.equal(gpuJobLine({ run_id: 'r', state: 'running', preparing: true, elapsed_minutes: 0, max_minutes: 30 }), '正在准备训练环境');
  assert.equal(gpuJobLine({ run_id: 'r', state: 'running', preparing: true, label: '正在准备训练环境' }), '正在准备训练环境');
  assert.equal(isNearCap({ run_id: 'r', state: 'running', preparing: true, elapsed_minutes: 29, max_minutes: 30 }), false);
  assert.equal(isNearCap({ run_id: 'r', state: 'running', elapsed_minutes: 29, max_minutes: 30 }), true);
});

// t238 C7c-FIX4d: Critiquito final review — status words, at-cap copy, prep/held/failed-prep cards, no raw codes.
test('C7c-FIX4d: gpuBadge only says 训练中 once compute really runs', () => {
  const { gpuBadge, isAtCap, isNearCap, gpuJobLine } = require('./gpu-status.ts');
  assert.equal(gpuBadge({ state: 'queued' }), '排队中');
  assert.equal(gpuBadge({ state: 'running', preparing: true }), '准备中');
  assert.equal(gpuBadge({ state: 'running', elapsed_minutes: 3, max_minutes: 30 }), '训练中');
  assert.equal(gpuBadge({ state: 'succeeded' }), null);
  assert.equal(gpuBadge(null), null);
  assert.equal(isNearCap({ state: 'running', elapsed_minutes: 28, max_minutes: 30 }), true);
  assert.equal(isNearCap({ state: 'running', elapsed_minutes: 30, max_minutes: 30 }), false);
  assert.equal(isAtCap({ state: 'running', elapsed_minutes: 30, max_minutes: 30 }), true);
  assert.equal(isAtCap({ state: 'running', elapsed_minutes: 29, max_minutes: 30 }), false);
  assert.equal(isAtCap({ state: 'running', preparing: true, elapsed_minutes: 0, max_minutes: 30 }), false);
  assert.equal(gpuJobLine({ state: 'running', preparing: true }), '正在准备训练环境');
});

test('C7c-FIX4d: the card words at-cap, near-cap, retry and prep-cancel as the review asks', () => {
  const card = read('GpuJobStatus.tsx');
  const copy = read('gpu-status.ts');
  assert.match(copy, /atCap: '已到时间上限，正在停止…'/);
  assert.match(copy, /nearCap: \(max: number\) => `快到 \$\{max\} 分钟了，到点会自动停止。`/);
  assert.match(card, /data-testid="tw-gpu-at-cap"/);
  assert.match(card, /data-testid="tw-gpu-retry"/);
  assert.match(card, /job\.retry_label && onRetry/);
  assert.match(copy, /confirmPreparingTitle: '取消准备？'/);
  assert.doesNotMatch(copy, /放弃排队/); // FIX5: a held place offers only 重新发起
});

test('C7c-FIX4d: the title chip follows the live card, the retry is a new request, and the old continuation prompt is hidden', () => {
  const session = read('TaskSession.tsx');
  assert.match(session, /gpuBadge\(gpuJob\)/);
  assert.match(session, /liveBadge \?\? baseStatusLabel/);
  assert.match(session, /async function retryTraining\(\)/);
  assert.match(session, /onRetryTraining=\{retryTraining\} onGpuJob=\{setGpuJob\}/);
  const types = read('types.ts');
  assert.doesNotMatch(types, /\$\{explanation\}\$\{code/);
  const history = readFileSync(new URL('../screens/assistant-history.ts', import.meta.url), 'utf8');
  assert.match(history, /kind === "continued_execution"/);
});

test('C7c-FIX4d: fixtures for the new states carry the review copy and no technical words', () => {
  const { GPU_SHOT_FIXTURES } = require('./gpu-status.fixtures.ts');
  const { gpuJobForRun } = require('./gpu-status.ts');
  const pick = key => gpuJobForRun(GPU_SHOT_FIXTURES[key]('r'), 'r');
  assert.equal(pick('breaker').label, '训练暂时用不了');
  assert.equal(pick('breaker').eta_minutes, undefined);
  assert.equal(pick('held').retry_label, '重新发起');
  assert.equal(pick('prepFailed').label, '环境没有准备好');
  assert.equal(pick('prepFailed').retry_label, '重新准备');
  for (const key of ['preparing', 'preparingSlow', 'prepFailed', 'held', 'breaker']) {
    const job = pick(key);
    assert.doesNotMatch(`${job.label}${job.notice ?? ''}`, /torch|GB|MB|超时|timeout|900|下载|缓存|依赖|哈希|identity|显存|GPU|AIMSLAB|Workbench/i);
  }
});
