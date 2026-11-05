import { proxyAssistantRuntime } from '@/lib/server/assistant-runtime';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
type Context = { params: Promise<{ path?: string[] }> };

// Per method: which sub-resources of /v1/user-tasks/{id} pass through ('' = the task itself).
const ALLOWED: Record<string, string[]> = {
  GET: ['', 'export'],
  POST: ['draft', 'links', 'files', 'restore'],
  PATCH: [''],
  DELETE: [''],
};

async function forward(request: Request, context: Context) {
  const parts = (await context.params).path ?? [];
  const valid = parts.length === 0 ? ['GET', 'POST'].includes(request.method) : /^utask_[a-f0-9]{32}$/.test(parts[0]) &&
    parts.length <= 2 && (ALLOWED[request.method] ?? []).includes(parts[1] ?? '');
  if (!valid) return Response.json({ error: 'not_found' }, { status: 404 });
  // Only reads carry a query (list filters, search, export format).
  const search = request.method === 'GET' ? new URL(request.url).search : '';
  return proxyAssistantRuntime(request, `/v1/user-tasks${parts.length ? '/' + parts.join('/') : ''}${search}`);
}
export const GET = forward;
export const POST = forward;
export const PATCH = forward;
export const DELETE = forward;
