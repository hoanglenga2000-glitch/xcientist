import { proxyAssistantRuntime } from '@/lib/server/assistant-runtime';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
type Context = { params: Promise<{ path?: string[] }> };

async function forward(request: Request, context: Context) {
  const parts = (await context.params).path ?? [];
  const valid = parts.length === 0 || (/^utask_[a-f0-9]{32}$/.test(parts[0]) &&
    (parts.length === 1 || (parts.length === 2 && ['draft', 'links', 'files'].includes(parts[1]))));
  if (!valid) return Response.json({ error: 'not_found' }, { status: 404 });
  return proxyAssistantRuntime(request, `/v1/user-tasks${parts.length ? '/' + parts.join('/') : ''}`);
}
export const GET = forward;
export const POST = forward;
