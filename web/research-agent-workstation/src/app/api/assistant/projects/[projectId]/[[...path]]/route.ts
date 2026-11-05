import { proxyAssistantRuntime } from '@/lib/server/assistant-runtime';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
type Context = { params: Promise<{ projectId: string; path?: string[] }> };

// Per method: which sub-resources of /v1/projects/{id} pass through ('' = the project itself, 'files/*' = one binding).
const ALLOWED: Record<string, string[]> = {
  GET: ['', 'files', 'memory-settings'],
  POST: ['restore', 'files'],
  PATCH: ['', 'memory-settings'],
  DELETE: ['', 'files/*'],
};

async function forward(request: Request, context: Context) {
  const { projectId, path = [] } = await context.params;
  const shape = path.length === 2 && path[0] === 'files' && /^attachment_[a-f0-9]{32}$/.test(path[1]) ? 'files/*' : path.length <= 1 ? path[0] ?? '' : null;
  if (!/^project_[a-f0-9]{32}$/.test(projectId) || shape === null || !(ALLOWED[request.method] ?? []).includes(shape)) {
    return Response.json({ error: 'project_not_found' }, { status: 404 });
  }
  // The runtime derives the owner from the session; no query reaches project routes.
  return proxyAssistantRuntime(request, `/v1/projects/${projectId}${path.length ? '/' + path.join('/') : ''}`);
}
export const GET = forward;
export const POST = forward;
export const PATCH = forward;
export const DELETE = forward;
