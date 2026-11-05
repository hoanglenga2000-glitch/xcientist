'use client';
import { useEffect, useRef, useState } from 'react';
import { TaskRunView } from './TaskRunView';
import { jsonPost, taskRequest, taskStatus, type TaskDetail, type TaskRun, type UserTask } from './types';
import { ErrorNotice } from './ErrorNotice';

type RecordRun = TaskRun & { title: string; project_id?: string };
type Project = { id: string; name: string };
export function WorkspaceRecords({ projects = false }: { projects?: boolean }) {
  const [runs, setRuns] = useState<RecordRun[]>([]);
  const [groups, setGroups] = useState<Project[]>([]);
  const [group, setGroup] = useState('');
  const [name, setName] = useState('');
  const [error, setError] = useState<unknown>(null);
  const [hint, setHint] = useState('');
  const [failed, setFailed] = useState<'load' | 'create'>('load');
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [revision, setRevision] = useState(0);
  const creation = useRef<{ name: string; key: string } | null>(null);
  useEffect(() => {
    const abort = new AbortController(); setLoading(true); setError(null); setFailed('load');
    void taskRequest<{ runs: RecordRun[]; projects: Project[] }>('/api/assistant/projects', { signal: abort.signal }).then(data => {
      if (!abort.signal.aborted) { setRuns(data.runs); setGroups(data.projects); }
    }).catch(cause => { if (!abort.signal.aborted) setError(cause); }).finally(() => { if (!abort.signal.aborted) setLoading(false); });
    return () => abort.abort();
  }, [revision]);
  async function create() {
    if (busy || !name.trim()) return;
    if (creation.current && creation.current.name !== name.trim()) { setHint('上次创建结果尚待确认，请保留原名称重试。'); return; }
    creation.current ??= { name: name.trim(), key: crypto.randomUUID() };
    setBusy(true); setError(null); setHint(''); setFailed('create');
    try {
      await taskRequest('/api/assistant/projects', jsonPost({ name: creation.current.name, idempotency_key: creation.current.key }));
      creation.current = null; setName(''); setRevision(value => value + 1);
    } catch (cause) { setError(cause); }
    finally { setBusy(false); }
  }
  const visible = group ? runs.filter(run => run.project_id === group) : runs;
  return <section><h1>{projects ? '项目' : '成果'}</h1>{!projects && <p className="tw-muted">打开自己的运行记录，预览和下载实际生成的文件。没有文件的运行仍可查看原因。</p>}
    {projects && <form className="tw-settings-form" onSubmit={event => { event.preventDefault(); void create(); }}><label htmlFor="new-project">项目名称</label><input id="new-project" value={name} maxLength={120} onChange={event => setName(event.target.value)} /><button disabled={busy || !name.trim()}>创建项目</button></form>}
    <div className="tw-run-select"><label htmlFor="records-project">项目</label><select id="records-project" value={group} onChange={event => setGroup(event.target.value)}><option value="">全部</option>{groups.map(project => <option key={project.id} value={project.id}>{project.name}</option>)}</select></div>
    {loading && <p role="status">正在读取你的记录…</p>}{hint && <p role="alert" className="tw-error">{hint}</p>}<ErrorNotice error={error} onAction={action => failed === 'create' && action === '重试' ? void create() : setRevision(value => value + 1)} />
    {!loading && !visible.length && <p>还没有运行记录。<a className="tw-button" href="/workspace?new=1">新对话</a></p>}
    {visible.map(run => <article className="tw-task-row" key={run.id}><div className="tw-task-summary"><h2>{run.title || run.prompt?.slice(0, 80) || '历史运行'}</h2><p>{taskStatus(run.status)}</p></div><a className="tw-button" href={`/workspace?legacy_run=${encodeURIComponent(run.id)}&view=results`}>查看文件与来源</a></article>)}
    <details className="tw-details"><summary>实验对比说明</summary>当前候选尚未完成同数据、同指标及同验证口径的对比迁移，不展示混合排名。</details>
  </section>;
}

export function LegacyRecord({ id, tasks, view }: { id: string; tasks: UserTask[]; view: string }) {
  const [run, setRun] = useState<TaskRun | null>(null);
  const [selected, setSelected] = useState('');
  const [error, setError] = useState<unknown>(null);
  const [revision, setRevision] = useState(0);
  const [failed, setFailed] = useState<'load' | 'link'>('load');
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    const abort = new AbortController(); setRun(null); setError(null); setFailed('load');
    void taskRequest<TaskRun>(`/api/assistant/runs/${encodeURIComponent(id)}`, { signal: abort.signal }).then(data => {
      if (!abort.signal.aborted) setRun(data);
    }).catch(cause => { if (!abort.signal.aborted) setError(cause); });
    return () => abort.abort();
  }, [id, revision]);
  async function link() {
    if (!selected || busy || !run) return;
    setBusy(true); setError(null); setFailed('link');
    try {
      await taskRequest<TaskDetail>(`/api/assistant/tasks/${selected}/links`, jsonPost({ run_id: run.id }));
      window.location.assign(`/workspace?task=${selected}&assistant_run=${encodeURIComponent(run.id)}&view=${encodeURIComponent(view)}`);
    } catch (cause) { setError(cause); setBusy(false); }
  }
  return <section><a className="tw-back" href="/workspace">← 对话</a><h1>历史记录</h1>
    <ErrorNotice error={error} onAction={action => failed === 'link' && action === '重试' ? void link() : setRevision(value => value + 1)} />
    {run && <><TaskRunView run={run} view={view} onResume={async () => { await taskRequest(`/api/assistant/runs/${encodeURIComponent(run.id)}/actions`, jsonPost({ action: 'resume', idempotency_key: crypto.randomUUID() })); setRevision(value => value + 1); }} />
      {run.user_task_id ? <a className="tw-button" href={`/workspace?task=${encodeURIComponent(run.user_task_id)}&assistant_run=${encodeURIComponent(run.id)}&view=${encodeURIComponent(view)}`}>返回所属任务</a> : <div className="tw-link-form"><p>这条记录已通过当前账户所有权校验。要继续工作，请明确关联到自己的任务，不会改写旧对话或自动执行。</p><label htmlFor="legacy-target">关联到哪个任务</label><select id="legacy-target" value={selected} onChange={event => setSelected(event.target.value)}><option value="">请选择自己的任务</option>{tasks.map(task => <option key={task.id} value={task.id}>{task.title}</option>)}</select><button disabled={!selected || busy} onClick={() => void link()}>确认关联</button><a href="/workspace?new=1">先新建任务</a></div>}
    </>}{!run && !error && <p role="status">正在核验这条记录的归属…</p>}
  </section>;
}
