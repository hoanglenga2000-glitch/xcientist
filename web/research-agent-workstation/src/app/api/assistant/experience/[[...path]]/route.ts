import { proxyAssistantRuntime } from '@/lib/server/assistant-runtime';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
type Context = { params: Promise<{ path?: string[] }> };

// Phase 2 我的经验. Per method: which shapes of /v1/experience pass through ('' = the list, 'settings', '*' = one experience).
// No create: the platform records experience after each training. /summary keeps its own route.
const ALLOWED: Record<string, string[]> = {
  GET: ['', 'settings'],
  PATCH: ['settings'],
  DELETE: ['*'],
};

async function forward(request: Request, context: Context) {
  const { path = [] } = await context.params;
  const id = path[0] ?? '';
  const shape = path.length === 0 ? '' : path.length !== 1 ? null : id === 'settings' ? 'settings' : /^exp_[a-f0-9]{32}$/.test(id) ? '*' : null;
  if (shape === null || !(ALLOWED[request.method] ?? []).includes(shape)) {
    return Response.json({ error: 'experience_not_found' }, { status: 404 });
  }
  // The runtime derives the owner from the session; no query reaches experience routes.
  return proxyAssistantRuntime(request, `/v1/experience${path.length ? '/' + path.join('/') : ''}`);
}
export const GET = forward;
export const PATCH = forward;
export const DELETE = forward;
