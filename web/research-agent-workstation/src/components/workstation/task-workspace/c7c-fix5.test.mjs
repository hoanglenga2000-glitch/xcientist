// t238 C7c-FIX5: prep row, first-use notice, tenant approval copy, chip after send, bar-safe reveal, declined-follow-up consistency.
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
  if (name.startsWith('@/')) name = fileURLToPath(new URL('../../../' + name.slice(2), import.meta.url));
  return originalResolve.call(this, name, parent, ...args);
};
for (const ext of ['.ts', '.tsx']) Module._extensions[ext] = (module, filename) => {
  module._compile(ts.transpileModule(readFileSync(filename, 'utf8'), { compilerOptions: {
    module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022, esModuleInterop: true,
  }, fileName: filename }).outputText, filename);
};
const read = name => readFileSync(new URL('./' + name, import.meta.url), 'utf8');
const FORBIDDEN = /受控操作|managed_tensor_train|操作标记为可恢复|GPU|显存|lease|租约|ProgramData|sha256|哈希|C:\\/;

test('the activity row says 正在准备训练环境 while the card prepares, then goes back to the tool step', () => {
  const { thinkingRowState } = require('./task-presentation.ts');
  const run = { id: 'run_1', status: 'running', started_at: new Date().toISOString(), current_activity: { kind: 'tool', label: '训练模型', status: 'running' }, activity: [] };
  const prep = { run_id: 'run_1', state: 'running', preparing: true, label: '正在准备训练环境' };
  assert.equal(thinkingRowState(run, Date.now(), prep).text, '正在准备训练环境');
  assert.match(thinkingRowState(run, Date.now(), null).text, /训练模型/);
  assert.match(thinkingRowState(run, Date.now(), { ...prep, preparing: false }).text, /训练模型/);
  assert.equal(thinkingRowState(run, Date.now(), { ...prep, label: '正在取消…' }).text, '正在取消…'); // C1: same word as card
  assert.match(thinkingRowState(run, Date.now(), { ...prep, run_id: 'run_other' }).text, /训练模型/);
});

test('the training confirmation is plain: what it does, no tool id, no path, no 受控操作', () => {
  const { TaskControls } = require('./TaskControls.tsx');
  const card = approval => renderToStaticMarkup(React.createElement(TaskControls, { onChanged() {}, run: {
    id: 'run_fixture', status: 'waiting_approval', approvals: [{ id: 'approval_fixture', status: 'pending', reversible: true, risk_level: 'high', impact_scope: { workspace_root: 'C:\\ProgramData\\EvoMind\\x' }, ...approval }] } }));
  const html = card({ tool_name: 'managed_tensor_train', normalized_arguments: { configuration_path: 'C:\\ProgramData\\EvoMind\\x\\work\\c.json', configuration_sha256: 'ab'.repeat(32) } });
  assert.match(html, /待你确认：开始训练这个模型/); assert.match(html, /开始训练/); assert.match(html, /先不训练/); assert.match(html, /拒绝后不会开始/);
  assert.doesNotMatch(html, FORBIDDEN);
  const other = card({ tool_name: 'some_new_tool', normalized_arguments: { path: 'work/a.txt' } });
  assert.match(other, /待你确认：继续这一步/); assert.doesNotMatch(other, /受控操作|操作标记为可恢复/); assert.match(other, /确认执行/);
  const del = card({ tool_name: 'file_delete', reversible: false, normalized_arguments: { path: 'outputs/old.txt' } });
  assert.match(del, /删除文件/); assert.match(del, /拒绝后不会执行/);
});

test('a declined step offers no 检查并继续; an ordinary blocked run still does', () => {
  const { TaskControls } = require('./TaskControls.tsx');
  const view = run => renderToStaticMarkup(React.createElement(TaskControls, { onChanged() {}, run: { id: 'run_x', approvals: [], ...run } }));
  assert.doesNotMatch(view({ status: 'blocked', error_class: 'approval_rejected', training_outcome: 'succeeded' }), /检查并继续/);
  assert.doesNotMatch(view({ status: 'blocked', error_class: 'approval_rejected', training_outcome: '' }), /检查并继续/);
  assert.match(view({ status: 'blocked', error_class: 'planning_idle' }), /检查并继续/);
});

test('files already sent leave the chip row but stay for the next request', () => {
  const { pendingAttachments } = require('./task-presentation.ts');
  const files = [{ id: 'a' }, { id: 'b' }];
  assert.deepEqual(pendingAttachments(files, []).map(f => f.id), ['a', 'b']);
  assert.deepEqual(pendingAttachments(files, ['a']).map(f => f.id), ['b']);
  const source = read('TaskSession.tsx');
  assert.match(source, /setSentIds\(files\.map/); assert.match(source, /attachment_ids: files\.map\(file => file\.id\)/);
  assert.match(source, /attachment_ids: \[\.\.\.new Set\(\[\.\.\.files\.map\(file => file\.id\)/);
});

test('waiting cards stop below the sticky bar', () => {
  const { revealBelowBar } = require('./chat-scroll.ts');
  const calls = [];
  revealBelowBar({ scrollIntoView: options => calls.push(options) }); revealBelowBar(null);
  assert.equal(calls.length, 1); assert.equal(calls[0].block, 'start');
  assert.match(read('workspace.css'), /\.tw-gpu,\.task-workspace \.tw-confirm\{scroll-margin-top:72px\}/);
  assert.match(read('GpuJobStatus.tsx'), /revealBelowBar\(card\.current\)/);
  assert.match(read('TaskControls.tsx'), /revealBelowBar\(confirmCard\.current\)/);
});

test('cancelling a preparation says the step in flight ends first', () => {
  const { GPU_COPY } = require('./gpu-status.ts');
  assert.equal(GPU_COPY.confirmPreparingBody, '准备会停止，不占用你的训练时长，之后可以重新发起。'); assert.equal(GPU_COPY.confirmPreparingTitle, '取消准备？');
});

test('removing a code leaves no punctuation fragments, and 「（code）」 never shows', () => {
  const { visibleMessageText } = require('./task-presentation.ts');
  const out = visibleMessageText('（训练暂时不可用，请稍后再试。，`tenant_training_queue_full`，队列中没有位置）');
  assert.doesNotMatch(out, /[，,]`|`[，,]|。，|tenant_training/); assert.match(out, /队列中没有位置/);
  assert.equal(visibleMessageText('没有完成（environment_not_ready）。'), '没有完成。');
  assert.equal(visibleMessageText('正常的回答，没有变化。'), '正常的回答，没有变化。');
});

test('the title chip says 正在取消 while a cancel is in flight, never 训练中', () => {
  const { gpuBadge } = require('./gpu-status.ts');
  assert.equal(gpuBadge({ state: 'running', preparing: false, label: '正在取消…' }), '正在取消');
  assert.equal(gpuBadge({ state: 'running', preparing: true, label: '正在取消…' }), '正在取消');
  assert.equal(gpuBadge({ state: 'running', preparing: true, label: '正在准备训练环境' }), '准备中');
  assert.equal(gpuBadge({ state: 'running', preparing: false, label: '正在训练' }), '训练中');
  assert.equal(gpuBadge({ state: 'queued', label: '排队中' }), '排队中');
});

test('a held place offers 重新发起 and 不训练了 (no 放弃排队 / 取消排队)', () => {
  const card = read('GpuJobStatus.tsx');
  assert.match(card, /job\.can_cancel && !atCap && <button/);
  const { GPU_COPY } = require('./gpu-status.ts');
  assert.deepEqual([GPU_COPY.giveUpHeld, GPU_COPY.confirmHeldTitle, GPU_COPY.confirmHeldBody, GPU_COPY.confirmHeldNo, GPU_COPY.confirmHeldYes],
    ['不训练了', '不再训练？', '这次训练不会再自动开始。你之后随时可以重新发起。', '先留着', '不训练了']);
  assert.match(card, /data-testid="tw-gpu-retry"/);
  assert.doesNotMatch(read('gpu-status.ts') + card, /放弃排队/);
});

test('检查并继续 is hidden on 已完成, 已取消 and a declined step', () => {
  const { TaskControls } = require('./TaskControls.tsx');
  const view = run => renderToStaticMarkup(React.createElement(TaskControls, { onChanged() {}, run: { id: 'run_x', approvals: [], ...run } }));
  for (const run of [{ status: 'blocked', error_class: 'approval_rejected' }, { status: 'blocked', error_class: 'approval_rejected', training_outcome: 'succeeded' },
    { status: 'cancelled' }, { status: 'completed' }]) assert.doesNotMatch(view(run), /检查并继续/, JSON.stringify(run));
  assert.match(view({ status: 'paused' }), /检查并继续/);
  // ACCEPT11 M-1: a task paused or blocked after a finished training can still be resumed.
  assert.match(view({ status: 'paused', training_outcome: 'succeeded' }), /检查并继续/);
  // Critiquito F5-7 / product decision 8: CTA follows training_outcome — failed+succeeded is terminal (no 检查并继续).
  assert.doesNotMatch(view({ status: 'failed', error_class: 'incomplete_deliverables', training_outcome: 'succeeded' }), /检查并继续/);
  assert.doesNotMatch(view({ status: 'failed', training_outcome: 'failed' }), /检查并继续/);
});

test('FIX5g: digest lines, the managed channel and the operator console never show in a chat body', () => {
  const { visibleMessageText } = require('./task-presentation.ts');
  assert.equal(visibleMessageText('训练完成。训练证据的哈希已逐项核对。准确率 0.91。'), '训练完成。准确率 0.91。');
  assert.ok(!/GPU|CUDA|shell|显存|lease/i.test(visibleMessageText('I checked the CUDA driver in the shell. 结果在文件里。GPU lease ready. 准确率 0.91。')));
  assert.equal(visibleMessageText('Configuration written (sha256 b56d…fcdb). I will go through the managed adapter. 结果在文件里。'), '结果在文件里。');
  const json = visibleMessageText('结果：\n```json\n{"data_sha256": "e97a1ce7a0c2f8b1d0c5e3a49b7f6c2d1e0f9a8b7c6d5e4f3a2b1c0d9e8f7a6b"}\n```\n完成。');
  assert.ok(!/sha256|e97a1ce7/.test(json) && json.includes('完成。'), json);
  assert.ok(!/AIMSLAB|Workbench|受管/.test(visibleMessageText('AIMSLAB Workbench 没有绑定。请稍后再试。')));
  assert.equal(visibleMessageText('正常的回答，没有变化。'), '正常的回答，没有变化。');
});

test('FIX5g: the title chip agrees with the card for a held place, a paused card and a cancelled card beside a new confirmation', () => {
  const { gpuBadge, gpuJobLine } = require('./gpu-status.ts');
  assert.equal(gpuBadge({ state: 'queued', label: '需要你重新发起', retry_label: '重新发起' }), '需要你重新发起');
  assert.equal(gpuBadge({ state: 'queued', label: '训练暂时用不了', paused: true }), '训练暂时用不了');
  assert.equal(gpuBadge({ state: 'queued', label: '排队中（第 1 位）' }), '排队中');
  assert.match(read('TaskRunView.tsx'), /hideSettled=\{confirmPending\(run\)\}/);
  assert.match(read('GpuJobStatus.tsx'), /hideSettled && !isLiveGpuJob\(job\)/);
  assert.equal(gpuJobLine({ state: 'timeout', run_id: 'r' }), '到时间了，已停止');
  const { GPU_SHOT_FIXTURES } = require('./gpu-status.fixtures.ts');
  const banned = /超时|timeout|显存|显卡|GPU|VRAM|哈希|AIMSLAB|Workbench|受管|租约|lease/i;
  for (const [name, make] of Object.entries(GPU_SHOT_FIXTURES)) {
    const payload = make('run_x');
    for (const job of payload.jobs ?? []) assert.doesNotMatch(`${job.label ?? ''}${job.notice ?? ''}${job.retry_label ?? ''}`, banned, name);
  }
});

test('FIX5h: ordinary code, calls and markdown are shown exactly as written (ACCEPT11 H-2)', () => {
  const { visibleMessageText } = require('./task-presentation.ts');
  const code = '代码：\n```python\ndef main():\n    df = pd.read_csv(file_path)\n    model.fit()\n    print(total_count)\n```\n运行后会打印行数。';
  assert.equal(visibleMessageText(code), code);
  assert.equal(visibleMessageText('用 `load_data(input_dir)` 读取，再 `model.fit()`。'), '用 `load_data(input_dir)` 读取，再 `model.fit()`。');
  assert.equal(visibleMessageText('1. 读取\n   - 检查\n2. 训练'), '1. 读取\n   - 检查\n2. 训练');
  assert.equal(visibleMessageText('We trained it. The accuracy is 0.91.'), 'We trained it. The accuracy is 0.91.');
  assert.equal(visibleMessageText('没有完成(environment_not_ready)。'), '没有完成。');
});

test('FIX5h: a dropped sentence keeps its neighbour\'s period and a dropped list item leaves no orphan number', () => {
  const { visibleMessageText } = require('./task-presentation.ts');
  assert.equal(visibleMessageText('We trained it. The sha256 was checked. Done here.'), 'We trained it. Done here.');
  assert.equal(visibleMessageText('1. 读取\n2. 哈希已核对\n3. 保存'), '1. 读取\n3. 保存');
  assert.equal(visibleMessageText('```python\nx = 1\n    y = 2\n```\n哈希已核对。结果在文件里。'), '```python\nx = 1\n    y = 2\n```\n结果在文件里。');
});

test('FIX5h: the fence clean-up is linear on long whitespace runs', () => {
  const { visibleMessageText } = require('./task-presentation.ts');
  const start = Date.now();
  visibleMessageText('哈希 ```json' + ' '.repeat(100000) + 'x');
  assert.ok(Date.now() - start < 1500);
});

test('FIX5i: the title chip is the server badge, as is; the section title stays 训练状态', () => {
  const { gpuBadge, GPU_COPY } = require('./gpu-status.ts');
  assert.equal(GPU_COPY.heading, '训练状态');
  const card = read('GpuJobStatus.tsx');
  assert.match(card, /<h2 id="tw-gpu-heading" className="tw-gpu-heading">\{GPU_COPY\.heading\}<\/h2>/);
  assert.doesNotMatch(card, /sr-only|visually-hidden|aria-label=\{GPU_COPY\.heading\}/);
  assert.match(read('workspace.css'), /\.tw-gpu-heading\s*\{/);
  const machinery = read('task-presentation.ts');
  assert.match(machinery, /\\bGPU\[s\]\?\\b/);
  assert.match(machinery, /\\bCUDA\\b/);
  assert.match(machinery, /\\bshell\\b/);
  assert.match(machinery, /显存/);
  assert.match(machinery, /\\blease\\b/);
  for (const [job, word] of [[{ state: 'queued', badge: '排队中' }, '排队中'], [{ state: 'queued', badge: '训练暂时用不了', label: '训练暂时用不了' }, '训练暂时用不了'],
    [{ state: 'queued', badge: '需要你重新发起' }, '需要你重新发起'], [{ state: 'running', badge: '准备中', preparing: true }, '准备中'], [{ state: 'running', badge: '训练中' }, '训练中'],
    [{ state: 'running', badge: '正在取消', label: '正在取消…' }, '正在取消']]) assert.equal(gpuBadge(job), word);
  assert.equal(gpuBadge({ state: 'cancelled', badge: '已取消' }), '已取消'); // C7c-FIX5c C6: terminal chip reads server badge
  const fixtures = read('gpu-status.fixtures.ts');
  assert.match(fixtures, /badge: '排队中'/);
});

test('FIX5i: the card follows a fast job within two seconds, cancel and approvals keep their wording', () => {
  const card = read('GpuJobStatus.tsx');
  assert.match(card, /POLL_FAST_MS = 1500/);
  assert.match(card, /fast \? POLL_FAST_MS : POLL_MS/);
  assert.match(card, /waitingKey = job && \(job\.state === 'queued' \|\| job\.state === 'running'\)/);
});

test('FIX5i: a settled page lists no 尚未执行 step and never ticks the checking step when nothing finished', () => {
  const { settledStepStatus } = require('./task-presentation.ts');
  const done = { status: 'completed', training_outcome: 'succeeded' };
  assert.equal(settledStepStatus(done, { status: 'pending', label: '验证结果与文件' }), 'skipped');
  assert.equal(settledStepStatus(done, { status: 'completed', label: '验证结果与文件' }), 'completed');
  const failed = { status: 'completed', training_outcome: 'failed' };
  assert.equal(settledStepStatus(failed, { status: 'completed', label: '验证结果与文件' }), 'skipped');
  assert.equal(settledStepStatus(failed, { status: 'completed', label: '训练模型' }), 'completed');
  assert.equal(settledStepStatus({ status: 'cancelled' }, { status: 'pending', label: '训练模型' }), 'skipped');
  assert.equal(settledStepStatus({ status: 'running' }, { status: 'pending', label: '训练模型' }), 'pending');
});

test('FIX5i: JSON previews in the chat are folded under 技术详情; a cancelled run shows no files or 已通过校验; 您 becomes 你', () => {
  assert.match(read('AttachmentChip.tsx'), /<summary>技术详情<\/summary><pre className="tw-filechip-small"/);
  const view = read('TaskRunView.tsx');
  assert.match(view, /cancelledTrainingDelivery\(run, gpuJob\)/);
  assert.match(view, /resultArtifacts\(run, gpuJob\)/);
  assert.match(view, /CANCELLED_RESULTS_EMPTY|没有可用的结果/);
  const { visibleMessageText } = require('./task-presentation.ts');
  assert.equal(visibleMessageText('您的训练已经开始（training_started）。'), '你的训练已经开始。');
});

test('FIX5m (M-1): the browser tidy keeps metrics, file names and column names verbatim', () => {
  const { visibleMessageText } = require('./task-presentation.ts');
  for (const line of ['AUC 0.91', 'F1 0.87', 'loss 0.123', 'accuracy 92.4%', '验证集 AUC 0.91', 'Test Accuracy: 0.91', 'model_final.pt', 'customer_churn.csv']) {
    assert.equal(visibleMessageText(line), line);
  }
  assert.equal(visibleMessageText('训练完成。验证集 AUC 0.91。'), '训练完成。验证集 AUC 0.91。');
});

test('FIX5m (Critiquito 06:53): user file and column names in Chinese sentences stay verbatim; no empty 「」 is left', () => {
  const { visibleMessageText } = require('./task-presentation.ts');
  for (const line of ['已读取 churn_2026.csv，共 1200 行', '按 customer_id 去重', '目标列是「churned」', '结果保存在「model_final.pt」。']) {
    assert.equal(visibleMessageText(line), line);
  }
  assert.equal(visibleMessageText('已读取 churn_2026.csv。哈希已核对。共 1200 行。'), '已读取 churn_2026.csv。共 1200 行。');
  assert.equal(visibleMessageText('结果见「」。'), '结果见。');
  assert.equal(visibleMessageText('目标列是「churned」。哈希见「sha256 b56d…fcdb」。'), '目标列是「churned」。');
});

test('FIX5m (ACCEPT15 H-1): multi-word English names inside 「」 stay verbatim in the browser tidy', () => {
  const { visibleMessageText } = require('./task-presentation.ts');
  for (const line of ['目标列是「Churn Label」', '取值有「Fiber optic」和「Electronic check」', '合同类型「Month-to-month」占多数']) {
    assert.equal(visibleMessageText(line), line);
  }
});

test('FIX5c: name/value lines such as Best model: Random Forest stay; an internal value in a name/value line never shows', () => {
  const { visibleMessageText } = require('./task-presentation.ts');
  for (const line of ['Best model: Random Forest', 'Model: XGBoost', 'Best model: Random Forest\nAUC 0.91\nF1 0.87\nloss 0.123\naccuracy 92.4%', 'Dataset: churn_2026.csv',
    'Model file: model_final.pt', 'Target column: 「Churn Label」', '目标列是「Churn Label」，数据 churn_2026.csv，列 customer_id，取值「churned」。', '模型：Random Forest']) {
    assert.equal(visibleMessageText(line), line);
  }
  for (const line of ['Device: A800', 'Model: vLLM', 'Lease: ready', 'GPU: A800', 'Model: gpt-4o', 'Model: claude-opus-5.5', 'Model: deepseek-ai/DeepSeek-R1', 'Model: qwen2.5-72b-instruct',
    '- Model: gpt-4o', '**Model**: claude-opus-5.5', 'Model: Claude', 'Provider model: anthropic/claude-3', 'Model id: tenant_model_7f3a9c', 'Token: sk-abc123def456ghi789',
    'Best model: 9f2c1a7b3d5e8f4a6b0c2d4e6f8a1b3c', 'Path: /home/aimslab/x.csv', 'Worker: node-17', 'Best model: Random_Forest_v3_internal', '模型：gpt-4o']) {
    const out = visibleMessageText('训练完成。\n' + line + '\nAUC 0.91').replace(/\n{2,}/g, '\n');
    assert.equal(out, '训练完成。\nAUC 0.91', line);
    const inline = visibleMessageText('AUC 0.91. ' + line.replace(/^[-*\d. ]+/, '').replace(/\*\*/g, ''));
    assert.ok(inline.startsWith('AUC 0.91') && !/A800|vLLM|gpt-4o|claude|Lease|tenant_model/i.test(inline), `${line} -> ${inline}`);
  }
});

test('FIX5c: a cancelling card is polled fast, and its worst-case wait for the terminal state is well under 10 seconds', () => {
  const card = read('GpuJobStatus.tsx');
  const fast = Number(/POLL_FAST_MS = (\d+)/.exec(card)[1]);
  assert.ok(fast <= 2000 && fast * 2 <= 10000);
  assert.match(card, /job\.label === GPU_COPY\.cancelling/); // the cancelling card is on the fast cadence
  assert.match(card, /const polling = live \|\| isLiveGpuJob\(job\)/);
  assert.match(read('gpu-status.ts'), /isLiveGpuJob = .*job\.state === 'running'/);
});

test('FIX5c C1: ThinkingRow / title / card share 「正在取消…」 while cancel is in flight', () => {
  const { thinkingRowState } = require('./task-presentation.ts');
  const { gpuBadge } = require('./gpu-status.ts');
  const run = { id: 'r1', status: 'running', started_at: new Date().toISOString() };
  const cancelling = { run_id: 'r1', state: 'running', preparing: false, label: '正在取消…', badge: '正在取消', can_cancel: false };
  const row = thinkingRowState(run, Date.now(), cancelling);
  assert.equal(row.text, '正在取消…');
  assert.equal(row.live, true);
  assert.equal(gpuBadge(cancelling), '正在取消');
  const prepCancel = { ...cancelling, preparing: true };
  assert.equal(thinkingRowState(run, Date.now(), prepCancel).text, '正在取消…');
  assert.equal(gpuBadge(prepCancel), '正在取消');
});

test('FIX5c C6: terminal badges come from server badge (failed/lost/timeout/prep-failed/rejects)', () => {
  const { gpuBadge } = require('./gpu-status.ts');
  for (const [job, word] of [
    [{ state: 'failed', badge: '没有完成', label: '没有完成' }, '没有完成'],
    [{ state: 'failed', badge: '环境没有准备好', label: '环境没有准备好' }, '环境没有准备好'],
    [{ state: 'timeout', badge: '到时间了，已停止', label: '到时间了，已停止' }, '到时间了，已停止'],
    [{ state: 'unavailable', badge: '今天的训练时长已用完', label: '今天的训练时长已用完' }, '今天的训练时长已用完'],
    [{ state: 'unavailable', badge: '这次训练太大', label: '这次训练太大' }, '这次训练太大'],
    [{ state: 'unavailable', badge: '现在训练排满了', label: '现在训练排满了' }, '现在训练排满了'],
    [{ state: 'unavailable', badge: '现在排队的训练太多', label: '现在排队的训练太多' }, '现在排队的训练太多'],
    [{ state: 'succeeded', badge: '已完成', label: '已完成' }, '已完成'],
    [{ state: 'cancelled', badge: '已取消', label: '已取消' }, '已取消'],
  ]) assert.equal(gpuBadge(job), word);
  // older payload without badge still matches the card label
  assert.equal(gpuBadge({ state: 'failed', label: '没有完成' }), '没有完成');
});


test('FIX5c C2 CLEARING: cancelled / GPU-cancelled runs expose no result files and no 生成文件 count', () => {
  const { cancelledTrainingDelivery, resultArtifacts, CANCELLED_RESULTS_EMPTY } = require('./task-presentation.ts');
  const { libraryLabel } = require('./TaskSession.tsx');
  const arts = [{ id: 'a1', name: 'model.pkl' }, { id: 'a2', name: 'metrics.json' }];
  const run = { id: 'run_1', status: 'completed', artifacts: arts };
  assert.equal(cancelledTrainingDelivery(run, null), false);
  assert.equal(resultArtifacts(run, null).length, 2);
  assert.equal(cancelledTrainingDelivery({ ...run, status: 'cancelled' }, null), true);
  assert.deepEqual(resultArtifacts({ ...run, status: 'cancelled' }, null), []);
  const gpuCancel = { run_id: 'run_1', state: 'cancelled' };
  assert.equal(cancelledTrainingDelivery(run, gpuCancel), true);
  assert.deepEqual(resultArtifacts(run, gpuCancel), []);
  assert.equal(CANCELLED_RESULTS_EMPTY, '没有可用的结果');
  assert.equal(libraryLabel(0, 0), '从资料库选择');
  assert.ok(!/生成文件/.test(libraryLabel(0, 0)));
  assert.equal(libraryLabel(0, 7), '从资料库选择 · 生成文件（7）');
  const view = read('TaskRunView.tsx');
  assert.match(view, /CANCELLED_RESULTS_EMPTY/);
  assert.match(view, /cancelledTrainingDelivery\(run, gpuJob\)/);
  assert.match(read('task-presentation.ts'), /没有可用的结果/);
  assert.match(read('TaskSession.tsx'), /cancelledTrainingDelivery\(run, gpuJob\) \? 0/);
});

test('FIX5c C2 CLEARING: results markup renders empty state without 已通过校验 or file chips when cancelled', () => {
  const React = require('react');
  const { renderToStaticMarkup } = require('react-dom/server');
  const { TaskRunView } = require('./TaskRunView.tsx');
  const arts = Array.from({ length: 7 }, (_, i) => ({
    id: `f${i}`, name: `out_${i}.pkl`, bytes: 10, sha256: 'ab'.repeat(32), preview_kind: 'other',
    download_url: `/api/assistant/artifacts/f${i}?download=1`, preview_url: `/api/assistant/artifacts/f${i}`,
    source_tool_call: 'train_model',
  }));
  const run = {
    id: 'run_l5', status: 'completed', artifacts: arts, artifact_evidence: arts.slice(0, 2),
    answer: '已取消', prompt: '训练', created_at: '2026-11-02T00:00:00Z',
    message_history: { schema: 'evomind.assistant_message_history.v1', active_message_id: 'a', messages: [
      { id: 'u', role: 'user', content: '训练' }, { id: 'a', role: 'assistant', content: '已取消' },
    ]},
  };
  const html = renderToStaticMarkup(React.createElement(TaskRunView, {
    view: 'results', run, gpuJob: { run_id: 'run_l5', state: 'cancelled', label: '已取消', badge: '已取消' },
  }));
  assert.match(html, /没有可用的结果/);
  assert.doesNotMatch(html, /已通过校验/);
  assert.doesNotMatch(html, /out_0\.pkl/);
  assert.doesNotMatch(html, /中间文件与执行记录/);
  // Without CLEARING, same run would list files — control: completed + no gpu cancel still shows files
  const kept = renderToStaticMarkup(React.createElement(TaskRunView, { view: 'results', run, gpuJob: null }));
  assert.match(kept, /out_0\.pkl/);
  assert.match(kept, /已通过校验/);
  // Chat path: cleared chips (no attachments / 已通过校验)
  const chat = renderToStaticMarkup(React.createElement(TaskRunView, {
    view: 'conversation', run, gpuJob: { run_id: 'run_l5', state: 'cancelled', label: '已取消', badge: '已取消' },
  }));
  assert.doesNotMatch(chat, /已通过校验/);
  assert.doesNotMatch(chat, /data-testid="tw-attachments"/);
  assert.doesNotMatch(chat, /out_0\.pkl/);
});
