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

/** A composer draft belongs to one task. Another task keeps its own saved draft. */
export function sessionDraft(taskDraft: string, composerDraft: string | undefined, composerTaskId: string, taskId: string) {
  if (composerDraft === undefined || composerTaskId !== taskId) return taskDraft;
  return composerDraft;
}
