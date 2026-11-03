import { proxyAssistantRuntime } from '@/lib/server/assistant-runtime';
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export async function GET(request: Request, context: { params: Promise<{ path?: string[] }> }) {
  const parts = (await context.params).path ?? [];
  // t238 C7b D6: `/files/<id>/usage` lists the caller's tasks that reference this file.
  if (parts.length === 2 && parts[1] === 'usage' && /^attachment_[a-f0-9]{32}$/.test(parts[0])) return proxyAssistantRuntime(request, `/v1/user-files/${parts[0]}/usage`);
  if (parts.length > 1 || (parts[0] && !/^attachment_[a-f0-9]{32}$/.test(parts[0]))) return Response.json({ error: 'not_found' }, { status: 404 });
  // The list alone may be narrowed to one project's bindings; every other query is dropped.
  const project = parts.length ? null : new URL(request.url).searchParams.get('project_id');
  if (project !== null && !/^project_[a-f0-9]{32}$/.test(project)) return Response.json({ error: 'project_not_found' }, { status: 404 });
  return proxyAssistantRuntime(request, '/v1/user-files' + (parts.length ? '/' + parts[0] : '') + (project ? `?project_id=${project}` : ''));
}
// t238 C7b D6: delete one owned 资料 file by its id (non-owner → 404; active run → 409 attachment_in_use).
export async function DELETE(request: Request, context: { params: Promise<{ path?: string[] }> }) {
  const parts = (await context.params).path ?? [];
  if (parts.length !== 1 || !/^attachment_[a-f0-9]{32}$/.test(parts[0])) return Response.json({ error: 'not_found' }, { status: 404 });
  return proxyAssistantRuntime(request, `/v1/attachments/${parts[0]}`);
}
