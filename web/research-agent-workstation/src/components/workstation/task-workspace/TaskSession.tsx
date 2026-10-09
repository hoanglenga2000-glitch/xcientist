'use client';
import { useEffect, useRef, useState } from 'react';
import { ensureCreationKey, finishCreation, recoverCreation } from '../screens/run-submission';
import { TaskRunView } from './TaskRunView';
import { TaskFiles } from './TaskFiles';
import { TaskControls } from './TaskControls';
import type { ModelList } from './ModelSettings';
import { jsonPost, taskRequest, taskStatus, type TaskDetail, type TaskRun, type UserTask } from './types';

const active = (status: string) => ['queued', 'planning', 'running', 'verifying', 'recovering', 'pausing', 'waiting_approval'].includes(status);
const views = [['conversation', '对话'], ['progress', '进度'], ['results', '文件与结果'], ['more', '更多']] as const;

export function TaskSession({ initial, initialDraft }: { initial: TaskDetail; initialDraft?: string }) {
  const [task, setTask] = useState(initial.task);
  const [files, setFiles] = useState(initial.files ?? []);
  const [runs, setRuns] = useState(initial.runs);
  const [run, setRun] = useState<TaskRun | null>(null);
  const [runId, setRunId] = useState('');
  const [view, setView] = useState('conversation');
  const [draft, setDraft] = useState(initialDraft ?? task.draft);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [busy, setBusy] = useState(false);
  const [autosavePaused, setAutosavePaused] = useState(false);
  const [streamError, setStreamError] = useState('');
  const [reconnect, setReconnect] = useState(0);
  const [history, setHistory] = useState<Array<TaskRun & { title: string }>>([]);
  const [linkId, setLinkId] = useState('');
  const [models, setModels] = useState<ModelList>({ profiles: [], default_profile_id: '' });
  const [model, setModel] = useState('');
  const sending = useRef(false);
  const saving = useRef<Promise<UserTask> | null>(null);
  const saved = useRef(task);
  const alive = useRef(true);
  const currentRuns = useRef(runs);
  currentRuns.current = runs;

  useEffect(() => {
    const controller = new AbortController();
    void taskRequest<ModelList>('/api/assistant/model-profiles', { signal: controller.signal }).then(value => {
      if (!controller.signal.aborted) { setModels(value); setModel(value.default_profile_id); }
    }).catch(cause => { if (!controller.signal.aborted) setError(cause.message); });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    alive.current = true;
    function restore() {
      const params = new URLSearchParams(window.location.search);
      setView(views.some(([id]) => id === params.get('view')) ? params.get('view')! : 'conversation');
      const requested = params.get('assistant_run');
      if (requested && !currentRuns.current.some(item => item.id === requested)) {
        setRunId(''); setError('该运行没有关联到当前任务。请先在“更多”中明确关联。');
      } else setRunId(requested || currentRuns.current[0]?.id || '');
    }
    restore(); window.addEventListener('popstate', restore);
    return () => { alive.current = false; window.removeEventListener('popstate', restore); };
  }, [initial]);

  function navigate(nextView: string, nextRun = runId) {
    const url = new URL(window.location.href);
    url.searchParams.set('view', nextView);
    if (nextRun) url.searchParams.set('assistant_run', nextRun); else url.searchParams.delete('assistant_run');
    window.history.pushState(null, '', url); setView(nextView); setRunId(nextRun);
  }

  async function saveDraft(value = draft): Promise<UserTask> {
    if (saving.current) await saving.current;
    if (saved.current.draft === value) return saved.current;
    const promise = taskRequest<{ task: UserTask }>(`/api/assistant/tasks/${task.id}/draft`, jsonPost({ title: saved.current.title, draft: value, version: saved.current.version }))
      .then(result => { saved.current = result.task; if (alive.current) setTask(result.task); return result.task; });
    saving.current = promise;
    try { return await promise; } finally { if (saving.current === promise) saving.current = null; }
  }

  useEffect(() => {
    if (draft === saved.current.draft || error || autosavePaused) return;
    const timer = window.setTimeout(() => {
      void saveDraft(draft).then(() => { if (alive.current) setNotice('草稿已保存'); })
        .catch(cause => { if (alive.current) { setAutosavePaused(true); setError(cause instanceof Error ? cause.message : '草稿保存失败，当前输入已保留。'); } });
    }, 800);
    return () => window.clearTimeout(timer);
  }, [draft, error, autosavePaused]); // saveDraft serializes writes; it never sends to a model.

  async function reloadDraftVersion() {
    try {
      const detail = await taskRequest<TaskDetail>(`/api/assistant/tasks/${task.id}`);
      if (!alive.current) return;
      saved.current = detail.task; setTask(detail.task); setAutosavePaused(true);
      setError(''); setNotice('已读取最新版本，本页输入仍保留。检查后点击“保存草稿”才会替换服务器草稿。');
    } catch (cause) { setError(cause instanceof Error ? cause.message : '读取失败，输入保留。'); }
  }

  useEffect(() => {
    const warn = (event: BeforeUnloadEvent) => { if (draft !== saved.current.draft) { event.preventDefault(); event.returnValue = ''; } };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [draft]);

  useEffect(() => {
    setRun(null); setStreamError('');
    if (!runId) return;
    const controller = new AbortController();
    let stream: EventSource | undefined;
    let reading = false;
    let queued = false;
    let settled = false;
    let sequence = 0;
    async function read() {
      if (controller.signal.aborted) return;
      if (reading) { queued = true; return; }
      reading = true;
      try {
        const snapshot = await taskRequest<TaskRun>(`/api/assistant/runs/${encodeURIComponent(runId)}`, { signal: controller.signal });
        if (!controller.signal.aborted) {
          setRun(snapshot); setRuns(current => current.map(item => item.id === snapshot.id ? { ...item, status: snapshot.status } : item));
          setStreamError(''); settled = !active(snapshot.status); sequence = snapshot.last_event_seq ?? 0; if (settled) stream?.close();
        }
      } catch (cause) { if (!controller.signal.aborted) setStreamError(cause instanceof Error ? cause.message : '连接中断，运行状态待确认。'); }
      finally { reading = false; if (queued) { queued = false; void read(); } }
    }
    void read().then(() => {
      if (controller.signal.aborted || settled) return;
      stream = new EventSource(`/api/assistant/runs/${encodeURIComponent(runId)}/events?after=${sequence}`);
      for (const event of ['run_created', 'plan_updated', 'run_progress', 'run_completed', 'run_failed', 'run_blocked', 'approval_required', 'approval.requested', 'run_pause_requested', 'run_paused', 'artifact_published', 'tool_completed']) stream.addEventListener(event, () => void read());
      stream.addEventListener('assistant_text', event => {
        try { const payload = JSON.parse((event as MessageEvent).data) as { text?: string }; if (!controller.signal.aborted) setRun(current => current?.id === runId ? { ...current, streaming_text: payload.text ?? '' } : current); } catch { /* The next authenticated snapshot remains authoritative. */ }
      });
      stream.onerror = () => { if (!controller.signal.aborted) { stream?.close(); setStreamError('实时连接已断开，已显示的内容保留。重新连接只读取状态，不重复执行。'); } };
      stream.addEventListener('stream_error', () => { stream?.close(); if (!controller.signal.aborted) setStreamError('暂时无法读取实时状态，不能据此判断任务失败。请重新登录或连接后查看。'); });
    });
    return () => { controller.abort(); stream?.close(); };
  }, [runId, reconnect]);

  async function send() {
    if (sending.current || !draft.trim() || !model || (run && active(run.status))) return;
    sending.current = true; setBusy(true); setError('');
    try {
      await saveDraft();
      const recovered = await recoverCreation<TaskRun>(task.conversation_id);
      const profile = models.profiles.find(item => item.id === model && item.enabled);
      if (model !== 'platform' && !profile) throw new Error('该模型配置不可用，请重新选择或前往设置。');
      const result = recovered ?? await taskRequest<TaskRun>('/api/assistant/runs', jsonPost({
        prompt: draft, user_task_id: task.id, conversation_id: task.conversation_id,
        attachment_ids: files.map(file => file.id),
        ...(profile ? { model_profile_id: profile.id, model_profile_version: profile.version } : {}),
        idempotency_key: ensureCreationKey(task.conversation_id),
      }));
      if (!alive.current) return;
      if (result.user_task_id !== task.id) throw new Error('返回的运行未绑定当前任务，已停止切换。');
      finishCreation(task.conversation_id);
      setRuns(current => [result, ...current.filter(item => item.id !== result.id)]); setRun(result);
      navigate('conversation', result.id);
      if (!recovered) { await saveDraft(''); setDraft(''); } else setNotice('已找回此前请求，没有重复发送。当前草稿保留。');
    } catch (cause) { if (alive.current) setError(cause instanceof Error ? cause.message : '发送失败，输入已保留。'); }
    finally { sending.current = false; if (alive.current) setBusy(false); }
  }

  async function showHistory() {
    setError('');
    try { const result = await taskRequest<{ runs: Array<TaskRun & { title: string }> }>('/api/assistant/projects'); setHistory(result.runs.filter(item => !runs.some(linked => linked.id === item.id))); }
    catch (cause) { setError(cause instanceof Error ? cause.message : '历史记录读取失败'); }
  }
  async function link() {
    if (!linkId || busy) return;
    setBusy(true); setError('');
    try {
      const detail = await taskRequest<TaskDetail>(`/api/assistant/tasks/${task.id}/links`, jsonPost({ run_id: linkId }));
      setRuns(detail.runs); setHistory(current => current.filter(item => item.id !== linkId)); navigate('conversation', linkId); setLinkId('');
    } catch (cause) { setError(cause instanceof Error ? cause.message : '关联失败'); }
    finally { setBusy(false); }
  }

  return <section><a className="tw-back" href="/workspace">← 我的任务</a><p className="tw-eyebrow">{taskStatus(run?.status ?? task.status)}</p><h1>{task.title}</h1>
    <nav className="tw-tabs" aria-label="任务内容">{views.map(([id, label]) => <button key={id} aria-current={view === id ? 'page' : undefined} onClick={() => navigate(id)}>{label}</button>)}</nav>
    {runs.length > 1 && <label className="tw-run-select">查看哪次运行 <select aria-label="选择运行" value={runId} onChange={event => navigate(view, event.target.value)}>{runs.map(item => <option key={item.id} value={item.id}>{new Date(item.created_at ?? '').toLocaleString('zh-CN')} · {taskStatus(item.status)}</option>)}</select></label>}
    {streamError && <p role="alert" className="tw-error">{streamError}<button onClick={() => setReconnect(value => value + 1)}>重新连接</button></p>}
    {run && ['conversation', 'progress'].includes(view) && <TaskControls key={run.id} run={run} onChanged={() => setReconnect(value => value + 1)} />}
    {view !== 'more' ? (runId && !run ? <p role="status">正在读取所选运行；尚不能判断其状态。</p> : <TaskRunView run={run} view={view} />) : <section><h2>关联已有记录</h2><p className="tw-muted">只列出当前账户有权访问的运行。关联不会改写历史、合并对话或重新执行。</p><button onClick={() => void showHistory()}>查看可关联的历史运行</button>
      {history.length > 0 ? <div className="tw-link-form"><label htmlFor="link-history">选择要关联的记录</label><select id="link-history" value={linkId} onChange={event => setLinkId(event.target.value)}><option value="">请选择</option>{history.map(item => <option key={item.id} value={item.id}>{item.title} · {taskStatus(item.status)}</option>)}</select><button disabled={!linkId || busy} onClick={() => void link()}>关联到“{task.title}”</button></div> : <p className="tw-muted">未列出可关联记录。无明确所有权的旧记录需要管理员核实，不能自动认领。</p>}
    </section>}
    {view === 'conversation' && <form className="tw-composer" onSubmit={event => { event.preventDefault(); void send(); }}><label htmlFor="task-message">输入需求或追问</label><textarea id="task-message" value={draft} maxLength={20000} onChange={event => setDraft(event.target.value)} placeholder="描述希望完成的工作与输出结果…" />
      <label htmlFor="task-model">本次使用的模型</label><select id="task-model" value={model} disabled={busy} onChange={event => setModel(event.target.value)}><option value="">请选择模型</option><option value="platform">平台模型（沿用现有权限与配置）</option>{models.profiles.filter(item => item.enabled).map(item => <option key={item.id} value={item.id}>{item.name} · {item.model} · v{item.version}{item.verified_at ? '' : ' · 尚未测试'}</option>)}</select><a href={`/workspace?settings=models&return_task=${task.id}`}>配置我的模型</a>
      <div className="tw-composer-bottom"><span>{draft === task.draft ? '草稿已保存' : '草稿尚未保存'} · 发送才会开始执行<br />真实模型调用可能产生费用，执行仍受权限与预算控制。</span><button type="submit" className="tw-primary" disabled={busy || !draft.trim() || !model || Boolean(run && active(run.status))}>{busy ? '正在发送…' : runs.length ? '发送需求' : '开始任务'}</button></div>
      <button type="button" disabled={busy} onClick={() => { setError(''); void saveDraft().then(() => { setNotice('草稿已保存'); setAutosavePaused(false); }).catch(cause => { setAutosavePaused(true); setError(cause.message); }); }}>保存草稿</button>
      {autosavePaused && <button type="button" disabled={busy} onClick={() => void reloadDraftVersion()}>重新读取版本（保留本页输入）</button>}
    </form>}
    {view === 'conversation' && <details className="tw-details"><summary>添加或选择资料（{files.length}）</summary><TaskFiles selected={files} onSelect={async ids => {
      await saveDraft();
      const updated = await taskRequest<TaskDetail>(`/api/assistant/tasks/${task.id}/files`, jsonPost({ attachment_ids: ids, version: saved.current.version }));
      if (!alive.current) return;
      saved.current = updated.task; setTask(updated.task); setFiles(updated.files ?? []);
    }} /></details>}
    {error && <p role="alert" className="tw-error">{error}</p>}<p role="status" className="tw-muted">{error || (notice === '草稿已保存' && draft !== task.draft) ? '' : notice}</p>
  </section>;
}
