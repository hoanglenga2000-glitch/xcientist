/** Short product name for a stored model id. The id itself stays in the data. */
const PRODUCT_MODELS: Array<[RegExp, string]> = [
  [/^claude-opus(?:$|[-_.])/, 'Claude Opus'],
  [/^claude-sonnet(?:$|[-_.])/, 'Claude Sonnet'],
  [/^claude-haiku(?:$|[-_.])/, 'Claude Haiku'],
  [/^gpt(?:$|[-_.])/, 'GPT'],
  [/^deepseek(?:$|[-_.])/, 'DeepSeek'],
];

export function productModelLabel(model?: string) {
  const name = String(model || '').trim().toLowerCase();
  if (!name || name === 'none') return '';
  for (const [pattern, label] of PRODUCT_MODELS) {
    if (pattern.test(name)) return label;
  }
  return '';
}

/** Labels for the model the next call will use, and for a model.response already served. */
export function platformOptionLabel(
  platform: { provider?: string; model?: string } | null | undefined,
  served?: { provider?: string; model?: string } | null,
) {
  const model = served?.model || platform?.model || '';
  const product = productModelLabel(model);
  return product ? `平台模型 · ${product}` : '平台模型';
}

export function servedModelLabel(_provider?: string, model?: string) {
  return productModelLabel(model);
}

/** Stored answer prefix is a raw provider path. The chat line already names the model. */
export function visibleMessageText(content: string) {
  return content.replace(/\*\*本轮实际模型\*\*：`[^`]*`（来自受管 model_observed 事件）\s*/g, '').trim();
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
  if (/^verified \d+ artifact(?:\(s\)|s)?$/i.test(detail.trim())) return '已核对';
  return detail;
}

const ACTION_NAMES: Record<string, string> = {
  model_publish: '发布文件',
  artifact_publish: '发布文件',
  auto_finalize: '写入文件',
  verified_answer_fallback: '写入文件',
  artifact_bundle: '打包文件',
  report_generate: '生成报告',
  hpc_execute_solution: '生成文件',
  independent_model_verifier: '生成文件',
  research_coordinator: '生成文件',
};

/** Plain name for a stored operation id. An unknown id is never shown. */
export function actionName(source?: string) {
  const raw = String(source || '').trim();
  if (ACTION_NAMES[raw]) return ACTION_NAMES[raw];
  if (raw.startsWith('artifact_import:')) return '导入文件';
  if (raw.startsWith('competition_data_')) return '整理资料';
  return '已生成';
}

/** A composer draft belongs to one task. Another task keeps its own saved draft. */
export function sessionDraft(taskDraft: string, composerDraft: string | undefined, composerTaskId: string, taskId: string) {
  if (composerDraft === undefined || composerTaskId !== taskId) return taskDraft;
  return composerDraft;
}
