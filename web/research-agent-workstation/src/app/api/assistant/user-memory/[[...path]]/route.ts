import { proxyAssistantRuntime } from '@/lib/server/assistant-runtime';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
type Context = { params: Promise<{ path?: string[] }> };

// t237 C3.7. Per method: which shapes of /v1/user-memory pass through ('' = the list, '*' = one memory).
// No create (memory is recorded by the agent) and no restore after clear-all (no undo).
const ALLOWED: Record<string, string[]> = {
  GET: [''],
  PATCH: ['*'],
  DELETE: ['', '*'],
  POST: ['*/restore'],
};

async function forward(request: Request, context: Context) {
  const { path = [] } = await context.params;
  const id = path[0] ?? '';
  const shape = path.length === 0 ? '' : !/^memory_[a-f0-9]{32}$/.test(id) ? null
    : path.length === 1 ? '*' : path.length === 2 && path[1] === 'restore' ? '*/restore' : null;
  if (shape === null || !(ALLOWED[request.method] ?? []).includes(shape)) {
    return Response.json({ error: 'memory_not_found' }, { status: 404 });
  }
  // The runtime derives the owner from the session; no query reaches memory routes.
  return proxyAssistantRuntime(request, `/v1/user-memory${path.length ? '/' + path.join('/') : ''}`);
}
export const GET = forward;
export const POST = forward;
export const PATCH = forward;
export const DELETE = forward;
