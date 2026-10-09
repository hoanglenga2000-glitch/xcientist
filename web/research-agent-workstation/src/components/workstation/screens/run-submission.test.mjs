import assert from "node:assert/strict";
import { afterEach, beforeEach, test } from "node:test";
import { ensureCreationKey, finishCreation, recoverCreation } from "./run-submission.ts";

const originalWindow = globalThis.window;
const originalFetch = globalThis.fetch;
let entries;
beforeEach(() => {
  entries = new Map();
  const location = { href: "https://fixture.invalid/?page=assistant" };
  globalThis.window = {
    location,
    history: { replaceState(_state, _title, url) { location.href = String(url); } },
    sessionStorage: { getItem: (key) => entries.get(key) ?? null, setItem: (key, value) => entries.set(key, value), removeItem: (key) => entries.delete(key) },
  };
});
afterEach(() => { globalThis.window = originalWindow; globalThis.fetch = originalFetch; });

test("a pending creation keeps its key across repeated sends and refresh", () => {
  const key = ensureCreationKey("chat_a", "owner_a");
  assert.equal(ensureCreationKey("chat_a", "owner_a"), key);
  entries.clear();
  assert.equal(ensureCreationKey("chat_a", "owner_a"), key);
  assert.notEqual(ensureCreationKey("chat_b", "owner_b"), key);
});

test("only opaque request identity is persisted", () => {
  ensureCreationKey("chat_a", "owner_a");
  const record = JSON.parse(entries.get("owner_a:creation"));
  assert.deepEqual(Object.keys(record).sort(), ["conversation", "key"]);
  finishCreation("chat_a", "owner_a");
  assert.equal(new URL(window.location.href).searchParams.has("creation_request"), false);
  assert.equal(entries.size, 0);
});

test("recovery reads the original result without posting another creation", async () => {
  const key = ensureCreationKey("chat_a", "owner_a");
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, method: options.method ?? "GET" });
    return Response.json({ id: "run_original" });
  };
  assert.deepEqual(await recoverCreation("chat_a", "owner_a"), { id: "run_original" });
  assert.deepEqual(calls, [{ url: `/api/assistant/run-requests/${key}`, method: "GET" }]);
});

test("unreceived or in-progress requests retain the same key", async () => {
  const key = ensureCreationKey("chat_a", "owner_a");
  globalThis.fetch = async () => Response.json({ error: "run_request_not_found" }, { status: 404 });
  assert.equal(await recoverCreation("chat_a", "owner_a"), null);
  globalThis.fetch = async () => Response.json({ error: "run_creation_in_progress" }, { status: 409 });
  await assert.rejects(recoverCreation("chat_a", "owner_a"), /run_creation_in_progress/);
  assert.equal(ensureCreationKey("chat_a", "owner_a"), key);
});
