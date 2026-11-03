'use client';
import { useEffect, useState } from 'react';
import { COPY, deleteConversation, listConversations, patchConversation, restoreConversation } from './conversation-menu';
import { showToast } from './Toast';
import type { UserTask } from './types';

/** Presentational list (also rendered statically in tests). */
export function ArchivedList({ tasks, busy = '', onUnarchive, onDelete }: { tasks: UserTask[]; busy?: string; onUnarchive: (task: UserTask) => void; onDelete: (task: UserTask) => void }) {
  if (!tasks.length) return <div className="tw-empty"><p>没有归档的对话。</p></div>;
  return <div className="tw-task-list">{tasks.map(task => <article className="tw-task-row" key={task.id}>
    <div className="tw-task-summary"><h2><a href={`/workspace?task=${encodeURIComponent(task.id)}`}>{task.title}</a></h2>
      <p className="tw-task-line"><time dateTime={task.updated_at}>{new Date(task.updated_at).toLocaleDateString('zh-CN')}</time></p></div>
    <div className="tw-row-actions"><button type="button" disabled={busy === task.id} onClick={() => onUnarchive(task)}>恢复</button>
      <button type="button" disabled={busy === task.id} aria-label={COPY.remove(task.title)} onClick={() => onDelete(task)}>删除</button></div>
  </article>)}</div>;
}

/** 我的 → 已归档. Changes are reported up so the sidebar list and its 已归档（n） stay in step. */
export function ArchivedConversations({ onChanged }: { onChanged?: () => void }) {
  const [tasks, setTasks] = useState<UserTask[] | null>(null);
  const [busy, setBusy] = useState('');
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    void listConversations({ archived: '1' }, controller.signal).then(found => { if (!controller.signal.aborted) setTasks(found); })
      .catch(() => { if (!controller.signal.aborted) setTasks(current => current ?? []); });
    return () => controller.abort();
  }, [revision]);
  const reload = () => { setRevision(value => value + 1); onChanged?.(); };
  async function run(task: UserTask, action: () => Promise<unknown>, toast?: Parameters<typeof showToast>[0]) {
    setBusy(task.id);
    try { await action(); setTasks(current => current?.filter(item => item.id !== task.id) ?? null); onChanged?.(); if (toast) showToast(toast); }
    catch { showToast({ message: COPY.saveFailed }); }
    finally { setBusy(''); }
  }
  return <section><h2>已归档</h2>
    {tasks !== null && <ArchivedList tasks={tasks} busy={busy}
      onUnarchive={task => void run(task, () => patchConversation(task, { archived: false }))}
      onDelete={task => void run(task, () => deleteConversation(task.id), { message: COPY.deleted, action: { label: COPY.undo,
        run: () => void restoreConversation(task.id).then(reload).catch(() => showToast({ message: COPY.saveFailed })) } })} />}
  </section>;
}
