import { randomUUID } from 'node:crypto';
import { sampleTasks, type PreviewTask } from '../../../../model';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
type Session = { expires: number; tasks: PreviewTask[]; requests: Map<string, { fingerprint: string; task: PreviewTask }> };
const sessions = new Map<string, Session>();
const cookieName = 'deepevo_preview';
function json(value: unknown, status = 200, headers: HeadersInit = {}) {
  return Response.json(value, { status, headers: { 'Cache-Control': 'private, no-store', ...headers } });
}
function boundary(request: Request) {
  return process.env.EVOMIND_USABILITY_PREVIEW === '1'
    && ['127.0.0.1:8098', 'localhost:8098'].includes(request.headers.get('host') ?? '');
}
function sessionFor(request: Request) {
  const tokens = (request.headers.get('cookie') ?? '').split(';').map(value => value.trim()).filter(value => value.startsWith(`${cookieName}=`));
  if (tokens.length !== 1) return null;
  const session = sessions.get(tokens[0].slice(cookieName.length + 1));
  return session && session.expires > Date.now() ? session : null;
}
export async function GET(request: Request, context: { params: Promise<{ path: string[] }> }) {
  if (!boundary(request)) return json({ error: 'preview_disabled' }, 404);
  const { path } = await context.params;
  if (path.join('/') === 'session') {
    if (sessionFor(request)) return json({ sample: true });
    for (const [key, value] of sessions) if (value.expires <= Date.now()) sessions.delete(key);
    if (sessions.size >= 64) return json({ error: 'preview_sessions_full' }, 503);
    const token = randomUUID();
    sessions.set(token, { expires: Date.now() + 4 * 60 * 60 * 1000, tasks: sampleTasks(), requests: new Map() });
    return json({ sample: true }, 200, { 'Set-Cookie': `${cookieName}=${token}; HttpOnly; SameSite=Strict; Path=/; Max-Age=14400` });
  }
  const session = sessionFor(request);
  if (!session) return json({ error: 'session_required', message: '预览会话已结束，请重新打开预览。' }, 401);
  if (path.join('/') === 'tasks') return json({ sample: true, tasks: session.tasks });
  const task = path[0] === 'tasks' ? session.tasks.find(item => item.id === path[1]) : null;
  if (!task) return json({ error: 'not_found', message: '找不到该任务，或你没有访问权限。不会跳转到其他任务。' }, 404);
  if (path.length === 2) return json(task);
  if (path.length === 3 && path[2] === 'artifact' && task.result) {
    return new Response(task.result.content, { headers: {
      'Content-Type': 'text/markdown; charset=utf-8',
      'Content-Disposition': `attachment; filename="${task.result.name}"`,
      'Cache-Control': 'private, no-store',
    } });
  }
  return json({ error: 'not_found' }, 404);
}

export async function POST(request: Request, context: { params: Promise<{ path: string[] }> }) {
  if (!boundary(request)) return json({ error: 'preview_disabled' }, 404);
  if (request.headers.get('origin') !== `http://${request.headers.get('host')}`) {
    return json({ error: 'origin_rejected', message: '此操作只允许从本地预览页面发起。' }, 403);
  }
  const session = sessionFor(request);
  if (!session) return json({ error: 'session_required', message: '预览会话已结束。请重新打开，输入草稿仍保留在本页。' }, 401);
  // Explicit, opt-in fixture fault. Never imported by the production application.
  const failure = request.headers.get('x-preview-failure');
  const failures: Record<string, string> = {
    '401': '样例：预览会话已过期。请重新加载任务，输入已保留。',
    '403': '样例：你没有修改此任务的权限。请联系任务负责人；输入已保留。',
    '409': '样例：任务内容已更新。请重新加载后再发送，输入已保留。',
    '503': '样例：服务暂时不可用。请稍后重试，输入已保留。',
  };
  if (failure && failures[failure]) return json({ error: 'preview_injected_failure', message: failures[failure] }, Number(failure));
  if (failure === 'timeout') {
    await new Promise(resolve => setTimeout(resolve, 1200));
    return json({ error: 'preview_timeout', message: '样例请求超时，未执行任何操作。' }, 504);
  }
  const { path } = await context.params;
  if (Number(request.headers.get('content-length')) > 16000) return json({ error: 'too_large' }, 413);
  let body: Record<string, unknown>;
  try { body = await request.json(); } catch { return json({ error: 'invalid_json' }, 400); }
  if (!body || typeof body.requestId !== 'string' || !/^[a-f0-9-]{36}$/.test(body.requestId)) return json({ error: 'invalid_request' }, 400);
  const fingerprint = JSON.stringify({ path, body });
  const previous = session.requests.get(body.requestId);
  if (previous) return previous.fingerprint === fingerprint ? json(previous.task) : json({ error: 'request_conflict', message: '这次请求已用于其他内容，请重新发送。' }, 409);
  if (session.requests.size >= 200) return json({ error: 'preview_limit', message: '本次预览已达到操作上限，请重新启动预览。' }, 429);
  if (path.join('/') === 'tasks') {
    if (typeof body.title !== 'string' || !body.title.trim() || body.title.length > 120) return json({ error: 'invalid_title' }, 400);
    const task: PreviewTask = {
      id: `sample-${randomUUID()}`, title: body.title.trim(), sample: true,
      version: 1, status: 'draft', updated: new Date().toISOString(), messages: [], steps: [], result: null,
    };
    session.tasks.unshift(task);
    session.requests.set(body.requestId, { fingerprint, task: structuredClone(task) });
    return json(task, 201);
  }
  const task = path[0] === 'tasks' ? session.tasks.find(item => item.id === path[1]) : null;
  if (!task || path.length !== 3 || path[2] !== 'messages') return json({ error: 'not_found', message: '找不到该任务，或你没有访问权限。' }, 404);
  if (body.version !== task.version) return json({ error: 'stale_version', message: '任务内容已更新。请重新加载后再发送，你的输入不会丢失。' }, 409);
  if (typeof body.text !== 'string' || !body.text.trim() || body.text.length > 4000) return json({ error: 'invalid_text', message: '请输入 1–4000 字的需求。' }, 400);
  task.messages.push({ role: 'user', text: body.text.trim() }, {
    role: 'assistant', text: '已保存你的输入，并准备了可下载的需求记录样例。你可以打开“进度”查看发生了什么，或打开“文件与结果”预览文件。\n\n这是一条固定的样例回复，不是模型对需求的分析；没有检索论文、修改代码或启动训练。',
  });
  task.version += 1;
  task.updated = new Date().toISOString();
  task.status = 'completed';
  task.steps = [
    { title: '保存这次需求', state: 'done', detail: '仅保存到当前隔离预览会话，未创建真实运行。' },
    { title: '准备样例文件', state: 'done', detail: '将需求写入 Markdown 样例，不含研究结论；没有调用模型或算力。' },
  ];
  task.result = { name: 'request-sample.md', content: `# 需求记录（样例）\n\n所属任务：${task.title}\n样例标识：${task.id}\n版本：${task.version}\n\n## 本次需求\n${body.text.trim()}\n\n## 验证范围\n本文件仅记录输入，未调用模型、未检索文献、未执行实验。\n文件可下载，不代表模型质量或科研结论已验证。\n` };
  session.requests.set(body.requestId, { fingerprint, task: structuredClone(task) });
  return json(task);
}
