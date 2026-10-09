import { proxyAssistantRuntime } from '@/lib/server/assistant-runtime';
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export async function GET(request: Request, context: { params: Promise<{ path?: string[] }> }) {
  const parts = (await context.params).path ?? [];
  if (parts.length > 1 || (parts[0] && !/^attachment_[a-f0-9]{32}$/.test(parts[0]))) return Response.json({ error: 'not_found' }, { status: 404 });
  return proxyAssistantRuntime(request, '/v1/user-files' + (parts.length ? '/' + parts[0] : ''));
}
