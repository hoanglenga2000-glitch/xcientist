import test from 'node:test';
import assert from 'node:assert/strict';
const base = 'http://127.0.0.1:8098';
async function session() {
  const response = await fetch(`${base}/api/preview/session`);
  assert.equal(response.status, 200, 'isolated session must exist');
  return response.headers.get('set-cookie').split(';')[0];
}
const get = (path, cookie) => fetch(`${base}/api/preview/${path}`, { headers: cookie ? { cookie } : {} });
const post = (path, cookie, body, origin = base) => fetch(`${base}/api/preview/${path}`, {
  method: 'POST', headers: { cookie, origin, 'content-type': 'application/json' }, body: JSON.stringify(body),
});

test('signed-in preview lists only its sample tasks and preserves selected identity', async () => {
  const cookie = await session();
  assert.equal((await get('tasks')).status, 401);
  const listing = await (await get('tasks', cookie)).json();
  assert.equal(listing.sample, true);
  assert.deepEqual(listing.tasks.map(task => task.id), ['sample-review', 'sample-report']);
  const task = await (await get('tasks/sample-report', cookie)).json();
  assert.equal(task.title, '整理实验结果');
  assert.equal(task.id, 'sample-report');
  assert.equal((await get('tasks/sample-private', cookie)).status, 404);
  assert.equal((await get('tasks/not-a-task', cookie)).status, 404);
});

test('draft creation executes nothing; explicit sample reply yields task-bound downloadable artifact without replay', async () => {
  const cookie = await session();
  const body = { title: '工业缺陷检测阅读提纲', requestId: crypto.randomUUID() };
  const created = await post('tasks', cookie, body);
  assert.equal(created.status, 201);
  const draft = await created.json();
  assert.equal(draft.status, 'draft');
  assert.equal(draft.result, null);
  assert.equal(draft.messages.length, 0);
  assert.equal((await (await post('tasks', cookie, body)).json()).id, draft.id);
  const action = { text: '请关注工业表面缺陷，不需要训练', version: draft.version, requestId: crypto.randomUUID() };
  const reply = await post(`tasks/${draft.id}/messages`, cookie, action);
  assert.equal(reply.status, 200);
  const completed = await reply.json();
  assert.equal(completed.status, 'completed');
  assert.match(completed.messages.at(-1).text, /样例/);
  assert.equal((await (await post(`tasks/${draft.id}/messages`, cookie, action)).json()).version, completed.version);
  const readback = await (await get(`tasks/${draft.id}`, cookie)).json();
  assert.equal(readback.messages.length, 2);
  const artifact = await get(`tasks/${draft.id}/artifact`, cookie);
  assert.equal(artifact.status, 200);
  assert.match(artifact.headers.get('content-disposition'), /attachment/);
  const content = await artifact.text();
  assert.match(content, /工业缺陷检测阅读提纲/);
  assert.match(content, /未调用模型/);
  assert.ok(content.includes(draft.id));
  const other = await session();
  assert.equal((await get(`tasks/${draft.id}`, other)).status, 404);
  assert.equal((await get(`tasks/${draft.id}/artifact`, other)).status, 404);
  assert.equal((await post(`tasks/${draft.id}/messages`, other, action)).status, 404);
  assert.equal((await post(`tasks/${draft.id}/messages`, cookie, { ...action, requestId: crypto.randomUUID() })).status, 409);
  assert.equal((await post('tasks', cookie, body, 'https://foreign.invalid')).status, 403);
});

test('preview service failure leaves messages and artifact unchanged for a safe retry', async () => {
  const cookie = await session();
  const before = await (await get('tasks/sample-review', cookie)).json();
  const failed = await fetch(`${base}/api/preview/tasks/sample-review/messages`, {
    method: 'POST', headers: { cookie, origin: base, 'content-type': 'application/json', 'x-preview-failure': '503' },
    body: JSON.stringify({ text: '失败时也应保留的需求', version: before.version, requestId: crypto.randomUUID() }),
  });
  assert.equal(failed.status, 503);
  const after = await (await get('tasks/sample-review', cookie)).json();
  assert.deepEqual(after, before);
});
