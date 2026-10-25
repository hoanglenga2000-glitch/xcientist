/** Labels for the model the next call will use, and for a model.response already served. */
export function platformOptionLabel(
  platform: { provider?: string; model?: string } | null | undefined,
  served?: { provider?: string; model?: string } | null,
) {
  const provider = served?.model ? served.provider : platform?.provider;
  const model = served?.model || platform?.model || '';
  if (!model) return '平台模型（沿用现有权限与配置）';
  const identity = [provider, model].filter(Boolean).join(' / ');
  return `平台模型 · ${identity}`;
}

export function servedModelLabel(provider?: string, model?: string) {
  return [provider, model].filter(Boolean).join(' / ');
}

/** Follow the run's owning task instead of leaving another task's card in place. */
export function owningTaskHref(taskId: string, run: { id?: string; user_task_id?: string }, view = 'conversation') {
  const owner = run.user_task_id || '';
  if (!owner || !run.id || owner === taskId) return '';
  const target = new URLSearchParams({ task: owner, assistant_run: run.id, view: view || 'conversation' });
  return `/workspace?${target.toString()}`;
}

const SETTLED_WORK = new Set(['delivered', 'completed', 'cancelled', 'draft']);

/** Pending filter and the status chip both read this field. */
export function needsHandling(status: string) {
  return !SETTLED_WORK.has(status);
}

/** One newest-first ordering for the work list. */
export function sortTasksByUpdatedAt<T extends { id: string; updated_at?: string }>(tasks: T[]): T[] {
  return [...tasks].sort((left, right) => {
    const byTime = String(right.updated_at || '').localeCompare(String(left.updated_at || ''));
    return byTime || String(right.id).localeCompare(String(left.id));
  });
}

export function visibleTasks<T extends { id: string; title: string; status: string; updated_at?: string }>(
  tasks: T[],
  query: string,
  pendingOnly: boolean,
) {
  const needle = query.trim().toLocaleLowerCase();
  return sortTasksByUpdatedAt(tasks).filter(task => {
    const titled = task.title.toLocaleLowerCase().includes(needle);
    return titled && (!pendingOnly || needsHandling(task.status));
  });
}

const RAW_TOOL_ID = /^[a-z][a-z0-9_]{1,80}$/;

/** Short progress label. A tool id or tool_completed event is never shown. */
export function toolStepLabel(label: string, status = '') {
  if (status === 'failed' || label === 'tool_failed') return '这一步没有完成';
  if (!label || label === 'tool_completed' || RAW_TOOL_ID.test(label)) return '已完成一步';
  return label;
}

export function toolStepDetail(detail?: string) {
  if (!detail || detail === 'tool_completed' || RAW_TOOL_ID.test(detail)) return '';
  return detail;
}

/** A composer draft belongs to one task. Another task keeps its own saved draft. */
export function sessionDraft(taskDraft: string, composerDraft: string | undefined, composerTaskId: string, taskId: string) {
  if (composerDraft === undefined || composerTaskId !== taskId) return taskDraft;
  return composerDraft;
}
