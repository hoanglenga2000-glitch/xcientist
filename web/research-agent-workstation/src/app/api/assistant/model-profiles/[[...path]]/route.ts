import { proxyAssistantRuntime } from '@/lib/server/assistant-runtime';
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
type Context = { params: Promise<{ path?: string[] }> };
async function forward(request: Request, context: Context) {
  const parts = (await context.params).path ?? [];
  const valid = parts.length === 0 || (/^model_[a-f0-9]{32}$/.test(parts[0]) &&
    (parts.length === 1 || (parts.length === 2 && ['default', 'disable', 'enable', 'test'].includes(parts[1]))));
  // DELETE is a soft delete of one profile only (body: { version }).
  if (!valid || (request.method === 'DELETE' && parts.length !== 1)) return Response.json({ error: 'not_found' }, { status: 404 });
  return proxyAssistantRuntime(request, `/v1/model-profiles${parts.length ? '/' + parts.join('/') : ''}`);
}
export const GET = forward;
export const POST = forward;
export const DELETE = forward;
