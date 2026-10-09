export type RestorationTarget = { kind: "run"; id: string } | { kind: "conversation"; id: string };
const conversationId = /^[A-Za-z0-9_.-]{1,160}$/;

export function assistantRestorationTarget(href: string, stored: string | undefined, fallback: string): RestorationTarget {
  const url = new URL(href);
  if (url.searchParams.has("assistant_run")) {
    const id = url.searchParams.get("assistant_run") ?? "";
    if (!/^run_[a-f0-9]{32}$/.test(id)) throw new Error("invalid_run_link");
    return { kind: "run", id };
  }
  if (url.searchParams.has("conversation")) {
    const id = url.searchParams.get("conversation") ?? "";
    if (!conversationId.test(id)) throw new Error("invalid_conversation_link");
    return { kind: "conversation", id };
  }
  return { kind: "conversation", id: stored && conversationId.test(stored) ? stored : fallback };
}

export async function loadAssistantRestoration<T extends { id: string; conversation_id?: string }>(target: RestorationTarget, fetcher: typeof fetch, signal?: AbortSignal): Promise<T[]> {
  async function hydrate(rows: T[], expectedConversation: string): Promise<T[]> {
    if (rows.some(row => row.conversation_id !== expectedConversation || !/^run_[a-f0-9]{32}$/.test(row.id))) throw new Error("restored_conversation_identity_mismatch");
    const results = new Array<T>(rows.length);
    let next = 0;
    // List responses omit message bodies; load owned details before rendering.
    await Promise.all(Array.from({ length: Math.min(4, rows.length) }, async () => {
      while (next < rows.length) {
        const index = next++;
        const detail = await fetcher('/api/assistant/runs/' + encodeURIComponent(rows[index].id), { cache: "no-store", signal });
        if (!detail.ok) throw Object.assign(new Error("requested_history_unavailable"), { status: detail.status });
        const run = await detail.json() as T;
        if (run.id !== rows[index].id || run.conversation_id !== expectedConversation) throw new Error("restored_run_identity_mismatch");
        results[index] = run;
      }
    }));
    return results;
  }
  const path = target.kind === "run"
    ? `/api/assistant/runs/${encodeURIComponent(target.id)}`
    : `/api/assistant/runs?conversation_id=${encodeURIComponent(target.id)}&limit=40`;
  const response = await fetcher(path, { cache: "no-store", signal });
  if (!response.ok) throw Object.assign(new Error("requested_history_unavailable"), { status: response.status });
  if (target.kind === "run") {
    const run = await response.json() as T;
    if (run.id !== target.id) throw new Error("restored_run_identity_mismatch");
    // Resolve ownership through the exact Run first. Never trust a URL/cache
    // conversation in preference to the server's authenticated Run binding.
    if (!run.conversation_id || !conversationId.test(run.conversation_id)) return [run];
    const history = await fetcher(`/api/assistant/runs?conversation_id=${encodeURIComponent(run.conversation_id)}&limit=40`, { cache: "no-store", signal });
    if (!history.ok) throw Object.assign(new Error("requested_history_unavailable"), { status: history.status });
    const payload = await history.json() as { runs?: T[] };
    const ordered = [...(payload.runs ?? [])].reverse();
    if (ordered.some((row) => row.conversation_id !== run.conversation_id)) throw new Error("restored_conversation_identity_mismatch");
    const index = ordered.findIndex((row) => row.id === run.id);
    // An older deep link outside the recent page remains that exact Run.
    // More recent Runs must never replace or resume the requested one.
    return index < 0 ? [run] : [...await hydrate(ordered.slice(0, index), run.conversation_id), run];
  }
  const payload = await response.json() as { runs?: T[] };
  return hydrate([...(payload.runs ?? [])].reverse(), target.id);
}
