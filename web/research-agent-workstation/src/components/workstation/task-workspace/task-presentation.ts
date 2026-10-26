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

const CJK = /[\u3400-\u9fff]/g;
const PROCESS_OPENER = /^\s*(?:I(?:'ll| will|'m going to| am going to| need to|'ve| have)\b|Let me\b|Let's\b|Now,? (?:I|let)\b|First,? I\b|Next,? I\b|Okay\b|OK,|Alright\b|(?:Checking|Reading|Looking at|Writing|Creating|Verifying|Running)\b)/im;
const PATH_OR_ARTIFACT = /(?:^|[\s`'"(])(?:\.{0,2}\/)?(?:inputs|outputs|work|workspace|artifacts?|runs)\/\S*|[\w.-]+\.(?:md|json|csv|txt|py|ipynb|ya?ml|log)\b|\b(?:artifacts?|tool[_ ]calls?|manifest|required result|deliverables?|verifier)\b/i;

/** English process chatter or a path/artifact dump. It belongs under 查看过程, not in the chat. */
export function isProcessLeak(text: string) {
  const body = text.trim();
  if (!body) return false;
  const cjk = (body.match(CJK) ?? []).length;
  const latin = (body.match(/[A-Za-z]/g) ?? []).length;
  if (cjk * 4 > latin) return false;
  return PROCESS_OPENER.test(body) || PATH_OR_ARTIFACT.test(body);
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

const INTERNAL_PROGRESS = /required result artifacts|missing final artifacts|\brecoverable\b|技术错误|(?:^|[\s`'"(])(?:inputs|outputs|workspace)\/\S*|[A-Za-z0-9_.-]+\.(?:md|json|csv|txt|pdf|png|html|py)\b/i;

export function toolStepDetail(detail?: string) {
  if (!detail || detail === 'tool_completed' || RAW_TOOL_ID.test(detail)) return '';
  if (/^verified \d+ artifact(?:\(s\)|s)?$/i.test(detail.trim())) return '已核对';
  if (INTERNAL_PROGRESS.test(detail)) return '';
  return detail;
}

export const CHAT_TITLE = '今天想推进哪一项？';

export const CHAT_SUGGESTIONS = [
  '复现一篇论文的基线',
  '整理比赛数据，给出切分方案',
  '估算一次训练要花多少',
] as const;

export const STREAM_PHASES = ['理解目标', '执行', '核对', '交付'] as const;

const LIVE_TASK = new Set(['queued', 'planning', 'running', 'verifying', 'recovering', 'pausing', 'waiting_approval']);

export function isLiveTask(status: string) {
  return LIVE_TASK.has(status);
}

/** Sidebar groups. Unfinished steps use the same gray, distinguished by a hollow dot. */
export function conversationGroup(updatedAt?: string, now = Date.now()): '今天' | '昨天' | '更早' {
  const time = new Date(updatedAt || '').getTime();
  if (!Number.isFinite(time)) return '更早';
  const start = new Date(now);
  start.setHours(0, 0, 0, 0);
  if (time >= start.getTime()) return '今天';
  if (time >= start.getTime() - 86_400_000) return '昨天';
  return '更早';
}

export function formatFileSize(bytes: number) {
  if (!Number.isFinite(bytes) || bytes < 0) return '';
  if (bytes < 1024) return `${Math.round(bytes)} B`;
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  const mb = bytes / (1024 * 1024);
  return `${mb >= 10 ? Math.round(mb) : mb.toFixed(1)} MB`;
}

export type PhaseState = { current: number; stuck: boolean; done: boolean; copy: { status: string; action: string } | null };

/** Four visible phases. A stuck phase is the red step; earlier phases are done. */
export function streamPhaseState(run: { status?: string; error_class?: string; error_message?: string; answer?: string }): PhaseState {
  const status = String(run.status || '');
  const copy = blockedProgressCopy(run);
  if (status === 'completed' || status === 'delivered') return { current: -1, stuck: false, done: true, copy: null };
  if (copy) {
    const current = copy.status === '方案停住了' ? 0
      : copy.status === '有一步还没核对完' ? 2
      : copy.status === '结果还没齐' || copy.status === '尚未交付' ? 3
      : 1;
    return { current, stuck: true, done: false, copy };
  }
  const current = status === 'verifying' ? 2 : ['running', 'recovering', 'pausing'].includes(status) ? 1 : 0;
  return { current, stuck: false, done: false, copy: null };
}

/** One sentence before a stuck card. No round counts, filenames, or internal codes. */
export function blockedAssistantLine(status: string) {
  if (status === '结果还没齐') return '结果还没齐，我先检查再接着跑。';
  if (status === '方案停住了') return '方案停住了，我先检查这一步再继续。';
  if (status === '这次没有完成') return '这次没有完成，我先检查再接着跑。';
  if (status === '尚未交付') return '还没有交付，我先检查缺了什么。';
  if (status === '有一步还没核对完') return '有一步还没核对完，我先核对再继续。';
  return '这一步停住了，我先检查再接着跑。';
}

const DANGER_CONFIRM = /delete|submit|kaggle|external|shell|hpc|publish/i;
const OUTBOUND_OR_OVER_FLAG = /over_budget|exceeds|external_submit|kaggle/i;

/** Spend checks that do not send work out. Every other tool keeps the separate confirmation. */
const BUDGET_INLINE_TOOLS = new Set(['cost_estimate']);

type ConfirmApproval = {
  tool_name?: string;
  risk_level?: string;
  normalized_arguments?: Record<string, unknown>;
  impact_scope?: Record<string, unknown>;
};

function finiteNumber(value: unknown) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function spendPair(approval: ConfirmApproval) {
  const args = approval.normalized_arguments || {};
  const amount = finiteNumber(args.estimated_cost ?? args.cost ?? args.amount);
  const cap = finiteNumber(args.budget ?? args.limit ?? args.cap);
  return amount === null || cap === null ? null : { amount, cap };
}

function riskIsLow(approval: ConfirmApproval) {
  const risk = String(approval.risk_level ?? 'low').trim().toLowerCase();
  return risk === '' || risk === 'low';
}

function knownBudgetTool(approval: ConfirmApproval) {
  const name = String(approval.tool_name || '');
  return BUDGET_INLINE_TOOLS.has(name) && !DANGER_CONFIRM.test(name);
}

function outboundOrOverFlag(approval: ConfirmApproval) {
  const name = String(approval.tool_name || '');
  const blob = `${JSON.stringify(approval.normalized_arguments || {})}${JSON.stringify(approval.impact_scope || {})}`;
  return DANGER_CONFIRM.test(name) || OUTBOUND_OR_OVER_FLAG.test(blob);
}

/** In-chat card only when risk is low, the tool is a known budget check, and the numeric spend is within the cap. */
export function usesSeparateConfirm(approval: ConfirmApproval) {
  if (!riskIsLow(approval) || !knownBudgetTool(approval) || outboundOrOverFlag(approval)) return true;
  const spend = spendPair(approval);
  return !spend || spend.amount > spend.cap;
}

/** The no-outbound sentence is shown only for a budget tool that is not an external send. */
export function confirmNoOutboundCopy(approval: ConfirmApproval) {
  if (!knownBudgetTool(approval) || outboundOrOverFlag(approval)) return '';
  return '不外发。提交到比赛平台前会再问你。';
}

export function confirmActionLabel(tool?: string) {
  const name = String(tool || '');
  if (name === 'file_write' || name === 'write_file') return '写入这次任务的文件';
  if (!name || RAW_TOOL_ID.test(name)) return '继续推进当前方案';
  return name;
}

export function confirmSpendLabel(approval: { normalized_arguments?: Record<string, unknown> }) {
  const args = approval.normalized_arguments || {};
  const amount = args.estimated_cost ?? args.cost ?? args.amount;
  const cap = args.budget ?? args.limit ?? args.cap;
  if (typeof amount === 'number' && typeof cap === 'number') return `约 ¥${amount}，上限 ¥${cap}`;
  if (typeof amount === 'number') return `约 ¥${amount}`;
  return '在当前额度内';
}

/** One short status for a blocked or undelivered progress card. No filenames or internal codes. */
export function blockedProgressCopy(run: { status?: string; error_class?: string; error_message?: string; answer?: string }) {
  const status = String(run.status || '');
  const blob = `${run.error_class || ''}\n${run.error_message || ''}\n${run.answer || ''}`;
  if (run.error_class === 'model_unavailable' || blob.includes('Model execution is unavailable:')) return null;
  const relevant = status === 'blocked' || status === 'failed' || status === 'undelivered'
    || /required result artifacts unavailable|missing final artifacts|\brecoverable\b/i.test(blob);
  if (!relevant || status === 'waiting_approval') return null;
  if (run.error_class === 'execution_reconciliation_required' || run.error_class === 'approval_execution_unconfirmed') {
    return { status: '有一步还没核对完', action: '先核对这一步' };
  }
  let label = '这一步停住了';
  if (/required result artifacts unavailable|missing final artifacts/i.test(blob)) label = '结果还没齐';
  else if (run.error_class === 'planning_idle') label = '方案停住了';
  else if (status === 'failed') label = '这次没有完成';
  else if (status === 'undelivered') label = '尚未交付';
  return { status: label, action: '检查并继续' };
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
