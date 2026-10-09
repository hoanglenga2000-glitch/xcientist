import assert from "node:assert/strict";
import test from "node:test";
import { assistantRestorationTarget, loadAssistantRestoration } from "./assistant-restoration.ts";

const oldRun = "run_" + "a".repeat(32);

test("explicit Run wins over both the URL conversation and local cache", () => {
  const target = assistantRestorationTarget(`https://fixture.invalid/?assistant_run=${oldRun}&conversation=chat_recent`, "chat_cached", "chat_new");
  assert.deepEqual(target, { kind: "run", id: oldRun });
  assert.deepEqual(assistantRestorationTarget("https://fixture.invalid/?conversation=chat_url", "chat_cached", "chat_new"), { kind: "conversation", id: "chat_url" });
});

test("invalid explicit links never fall back to a different conversation", () => {
  for (const query of ["assistant_run=bad", "assistant_run=", "conversation=../escape"]) {
    assert.throws(() => assistantRestorationTarget(`https://fixture.invalid/?${query}`, "chat_cached", "chat_new"));
  }
});

test("opening an old Run performs only its exact read", async () => {
  const calls = [];
  const fetcher = async (url, options) => { calls.push({ url, method: options.method ?? "GET" }); return Response.json({ id: oldRun, status: "completed" }); };
  assert.deepEqual(await loadAssistantRestoration({ kind: "run", id: oldRun }, fetcher), [{ id: oldRun, status: "completed" }]);
  assert.deepEqual(calls, [{ url: `/api/assistant/runs/${oldRun}`, method: "GET" }]);
});

test("forbidden and missing Run links do not query recent history", async () => {
  for (const status of [401, 403, 404]) {
    const calls = [];
    await assert.rejects(loadAssistantRestoration({ kind: "run", id: oldRun }, async (url) => { calls.push(url); return Response.json({ error: "denied" }, { status }); }));
    assert.deepEqual(calls, [`/api/assistant/runs/${oldRun}`]);
  }
  await assert.rejects(loadAssistantRestoration({ kind: "run", id: oldRun }, async () => Response.json({ id: "run_" + "b".repeat(32) })), /identity_mismatch/);
});

test('refresh restores earlier conversation Runs but keeps the exact Run current', async () => {
  const previous = { id: 'run_' + 'b'.repeat(32), conversation_id: 'chat_server', answer: 'previous' };
  const exact = { id: oldRun, conversation_id: 'chat_server', answer: 'fresh', status: 'running' };
  const newer = { id: 'run_' + 'c'.repeat(32), conversation_id: 'chat_server' };
  const calls = [];
  const fetcher = async (url) => {
    calls.push(url);
    if (url === '/api/assistant/runs/' + previous.id) return Response.json(previous);
    return Response.json(calls.length === 1 ? exact : { runs: [newer, { ...exact, answer: 'stale' }, { id: previous.id, conversation_id: previous.conversation_id }] });
  };
  assert.deepEqual(await loadAssistantRestoration({ kind: 'run', id: oldRun }, fetcher), [previous, exact]);
  assert.deepEqual(calls, [`/api/assistant/runs/${oldRun}`, '/api/assistant/runs?conversation_id=chat_server&limit=40', '/api/assistant/runs/' + previous.id]);
});

test('conversation history rejects mixed ownership and never substitutes an older deep link', async () => {
  const exact = { id: oldRun, conversation_id: 'chat_server' };
  let count = 0;
  await assert.rejects(loadAssistantRestoration({ kind: 'run', id: oldRun }, async () => Response.json(++count === 1 ? exact : { runs: [{ id: 'other', conversation_id: 'chat_foreign' }] })), /identity_mismatch/);
  count = 0;
  assert.deepEqual(await loadAssistantRestoration({ kind: 'run', id: oldRun }, async () => Response.json(++count === 1 ? exact : { runs: [{ id: 'newer', conversation_id: 'chat_server' }] })), [exact]);
});

test('conversation summaries are hydrated in order with at most four reads at once', async () => {
  const details = Array.from({ length: 9 }, (_, i) => ({ id: 'run_' + i.toString(16).padStart(32, '0'), conversation_id: 'chat_complete', messages: [{ role: 'assistant', content: 'reply ' + i }] }));
  let active = 0, peak = 0;
  const controller = new AbortController();
  const restored = await loadAssistantRestoration({ kind: 'conversation', id: 'chat_complete' }, async (url, options) => {
    assert.equal(options.signal, controller.signal);
    if (url.includes('?')) return Response.json({ runs: [...details].reverse().map(({ id, conversation_id }) => ({ id, conversation_id })) });
    active++; peak = Math.max(peak, active);
    await new Promise(resolve => setTimeout(resolve, 5));
    active--;
    return Response.json(details.find(run => url.endsWith(run.id)));
  }, controller.signal);
  assert.deepEqual(restored, details);
  assert.equal(peak, 4);
});

test('history detail access failures and changed ownership never render summary placeholders', async () => {
  const row = { id: oldRun, conversation_id: 'chat_complete' };
  for (const detail of [Response.json({ error: 'denied' }, { status: 403 }), Response.json({ ...row, conversation_id: 'chat_foreign' })]) {
    await assert.rejects(loadAssistantRestoration({ kind: 'conversation', id: 'chat_complete' }, async url => url.includes('?') ? Response.json({ runs: [row] }) : detail));
  }
});
