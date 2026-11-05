import type { TaskRun } from './types';
// @ts-expect-error Native Node tests resolve the explicit .ts extension.
import { isTrainingInputLimit, TRAINING_INPUT_LIMIT_LINE, TRAINING_INPUT_LIMIT_STATUS } from './error-copy.ts';
// @ts-expect-error Native Node tests resolve the explicit .ts extension.
import { MODEL_DISPLAY_NAMES } from './model-display-names.ts';

/**
 * Product name of a model id, from the explicit registry in model-display-names.ts ONLY (t238 C7e-FE-BTN F-2).
 * The id is normalised (trim, lower-case, provider prefix `deepseek/…` dropped, `_`/spaces → `-`, trailing -YYYYMMDD dropped) and looked up.
 * An id with no registered name returns '' — never the raw id and never a derived or prettified name — so callers hide the line.
 * The Python `model_display_name` is a separate mapping, kept in step through tests/fixtures/model_display_names.json.
 */
export function productModelLabel(model?: string) {
  const id = String(model ?? '').trim().toLowerCase().split('/').pop()!.trim().replace(/[_\s]+/g, '-').replace(/-\d{8}$/, '');
  return Object.prototype.hasOwnProperty.call(MODEL_DISPLAY_NAMES, id) ? MODEL_DISPLAY_NAMES[id] : '';
}

export function servedModelLabel(_provider?: string, model?: string) {
  return productModelLabel(model);
}

/**
 * t238 C7c-FIX5h: what is left on ONE edited line after a code is removed — 「，`，」, 「。，」, empty brackets, doubled commas.
 * Only lines the filter actually changed are tidied, and never inside a fenced code block (ACCEPT11 H-2: code stays as written).
 */
function tidyLine(line: string) {
  let out = line.replace(/[，,、]\s*`+\s*(?=[，,。；;])/g, '');
  if ((out.match(/`/g) ?? []).length % 2) out = out.replace(/(?<![`\w])`+(?![`\w])/g, '');
  return out.replace(/[（(]\s*[）)]|「\s*」/g, '').replace(/([。！？；])\s*[，,、]+/g, '$1').replace(/[，,]{2,}/g, '，').replace(/^([ \t]*)[，,、；;]+\s*/, '$1').trimEnd();
}

const FENCE_LINE = /^\s*(?:```|~~~)/;
function editProseLines(text: string, edit: (line: string) => string) {
  let fenced = false;
  return text.split('\n').map(line => {
    if (FENCE_LINE.test(line)) { fenced = !fenced; return line; }
    if (fenced) return line;
    const next = edit(line).replace(/「\s*」/g, ''); // never leave an empty 「」 behind
    return next === line ? line : tidyLine(next);
  }).join('\n');
}

/** A leaked 「（environment_not_ready）」 code outside inline code; a call such as `load_data(path)` in code spans is left alone. */
function withoutCodeResidue(line: string) {
  const swapped = line.replace(/您/g, '你').replace(/`?tenant_training_[a-z0-9_]+`?/g, '训练暂时不可用，请稍后再试。').replace(/（\s*`?[a-z]+(?:_[a-z0-9]+)+`?\s*）/g, '');
  return swapped.split(/(`[^`\n]*`)/).map((part, index) => index % 2 ? part : part.replace(/\(\s*[a-z]+(?:_[a-z0-9]+)+\s*\)/g, '')).join('');
}

/** t238 C7c-FIX5g: the machinery's own vocabulary (digests, the managed channel, the operator's console). The server filters the same in tenant-training chats; this guards pages and streams that bypass it. */
const MACHINERY = /哈希|校验和|sha-?256|_sha256|受管|适配器|managed[ _-](?:adapter|channel|output)|AIMSLAB|Workbench|protocol[\s_-]*(?:id|hash)|\bGPU[s]?\b|\bCUDA\b|\bVRAM\b|vLLM|A800|\bshell\b|显存|显卡|\blease\b|租约|\b[0-9a-f]{40,64}\b/i;
const SENTENCE_END = /(?<=[。！？；!?;])|(?<=[a-z0-9)\]"”]\.)(?=\s+[A-Z“"(\u3400-\u9fff])/;

/** t238 C7c-FIX5c: same screen as the server's `_leaky_name_value`. A `Name: value` line is data (`Best model: Random Forest`), but an internal value in it (model/provider id, host, device, path, secret, digest) never shows. */
const NV_LINE = /^\s*(?:(?:\d{1,3}[.)]|[-*+•])\s+)?[*_`]*\s*([A-Za-z\u3400-\u9fff][A-Za-z \u3400-\u9fff]{0,38}?)\s*[*_`]*\s*[:：]\s*[*_`]*\s*(\S.*?)\s*$/;
const NV_KEY_SENSITIVE = new Set('api token tokens secret secrets password passwd credential credentials host hostname node worker workers device devices path paths url uri endpoint provider backend cluster queue job jobs partition container server machine account tenant user email ip engine driver runtime id gpu'.split(' '));
const NV_KEY_MODELISH = new Set('model models algorithm algorithms classifier classifiers regressor estimator engine provider backend llm'.split(' '));
const FAMILY = /\b(?:gpt(?:[-_ ]?\d\w*)?|chatgpt|claude|anthropic|openai|opus|sonnet|haiku|gemini|grok|llama(?:[-_ ]?\d\w*)?|qwen\d*(?:\.\d+)?|deepseek|mistral|mixtral|kimi|moonshot|glm[-_ ]?\d\w*|antigravity|lt4net|xai)\b/i;
const FILE_EXT = /\.(?:pt|pth|pkl|pickle|joblib|onnx|h5|ckpt|bin|safetensors|csv|tsv|json|jsonl|txt|md|png|jpg|xlsx|parquet|yaml|yml)\b/i;
const CLEAN_TOKEN = /^(?:[A-Za-z][A-Za-z0-9+]{0,24}(?:-[A-Za-z][A-Za-z0-9+]{0,24})*|\d{1,6}(?:\.\d+)?%?)$/;
const RAW_CODE = /^(?:[A-Za-z0-9.]+(?:[_-][A-Za-z0-9.]+)+|\S*\/\S*)$/;

export function leakyNameValue(line: string) {
  const m = NV_LINE.exec(line);
  if (!m) return false;
  const key = m[1].trim(), value = m[2].trim();
  if (/[\u3400-\u9fff]/.test(key) && !key.includes('模型')) return false; // other Chinese keys: the sentence filters above already decide
  const words = (key.match(/[A-Za-z]+/g) ?? []).map(w => w.toLowerCase());
  if (words.some(w => NV_KEY_SENSITIVE.has(w))) return true;
  if (FAMILY.test(value) || FAMILY.test(key)) return true;
  if (/^(?:[/~\\]|[A-Za-z]:[\\/])/.test(value) || value.includes('://') || value.includes('\\\\')) return true;
  const plain = value.replace(new RegExp(FILE_EXT.source, 'ig'), '');
  if (/\b(?=[0-9a-f]*\d)[0-9a-f]{16,}\b/i.test(plain) || /\b(?:sk|pk|ghp|gho|xox[a-z]|AKIA)[-_]?[A-Za-z0-9]{8,}/.test(plain)) return true;
  if ((plain.match(/[A-Za-z0-9_-]+/g) ?? []).some(t => t.length >= 24)) return true;
  if (words.some(w => NV_KEY_MODELISH.has(w)) || key.includes('模型')) {
    for (const raw of value.split(/\s+/)) {
      const tok = raw.replace(/^[,;()]+|[,;()]+$/g, '');
      if (!tok || FILE_EXT.test(tok)) continue;
      if (RAW_CODE.test(tok) && !CLEAN_TOKEN.test(tok) && !/^\d+(?:\.\d+)?%?$/.test(tok)) return true;
    }
  }
  return false;
}

export function withoutMachinery(text: string) {
  if (!MACHINERY.test(text) && !text.includes(':') && !text.includes('：')) return text;
  let fenced = false;
  const kept: string[] = [];
  for (const line of text.split('\n')) {
    if (FENCE_LINE.test(line)) { fenced = !fenced; kept.push(line); continue; }
    if (!fenced && leakyNameValue(line)) continue;
    if (!MACHINERY.test(line)) {
      const sentences = fenced || !/[:：]/.test(line) ? [] : line.split(SENTENCE_END).filter(part => part.trim());
      if (sentences.length > 1 && sentences.some(leakyNameValue)) {
        const left = sentences.filter(part => !leakyNameValue(part)).join('');
        if (left.trim()) kept.push(tidyLine(left)); // the leaky sentence goes, the rest of the line stays
        continue;
      }
      kept.push(line);
      continue;
    }
    if (fenced || line.trimStart().startsWith('|')) continue;
    const item = /^(\s*(?:\d{1,3}[.)]|[-*+•])\s+)(.*)$/.exec(line);
    const prefix = item ? item[1] : '';
    const body = item ? item[2] : line;
    const parts = body.split(SENTENCE_END).filter(part => part.trim());
    const left = parts.length > 1 ? parts.filter(part => !MACHINERY.test(part)).join('') : '';
    if (!left.trim()) { if (!item) kept.push(''); continue; }
    kept.push(tidyLine(prefix + left.trimStart()));
  }
  return kept.join('\n').replace(/```[a-z]{0,12}[{}[\],\s]{0,200}```/g, '').replace(/\n{3,}/g, '\n\n').replace(/^(?:[ \t]*\n)+/, '').trimEnd();
}

/** Stored answer prefix is a raw provider path. The chat line already names the model. */
export function visibleMessageText(content: string) {
  return editProseLines(withoutMachinery(content), withoutCodeResidue).replace(/\*\*本轮实际模型\*\*：`[^`]*`（来自(?:受管 model_observed 事件|模型输出)）\s*/g, '').replace(/^(?:[ \t]*\n)+/, '').trimEnd();
}

const CJK = /[\u3400-\u9fff]/g;
const PROCESS_OPENER = /^\s*(?:I(?:'ll| will|'m going to| am going to| need to|'ve| have)\b|Let me\b|Let's\b|Now,? (?:I|let)\b|First,? I\b|Next,? I\b|Okay\b|OK,|Alright\b|(?:Checking|Reading|Looking at|Writing|Creating|Verifying|Running)\b)/im;
// C6-answer H_a8: a bare file name (secret_code.txt) is not process chatter; only workspace paths and runtime jargon are.
const PATH_OR_ARTIFACT = /(?:^|[\s`'"(])(?:\.{0,2}\/)?(?:inputs|outputs|work|workspace|artifacts?|runs)\/\S*|\b(?:artifacts?|tool[_ ]calls?|manifest|required result|deliverables?|verifier)\b/i;

/**
 * English process chatter or a path/artifact dump. It belongs under the process row, not in the chat.
 * C6-answer H_a8: the last assistant message of a turn is the answer and is never hidden (`last`).
 */
export function isProcessLeak(text: string, { last = false }: { last?: boolean } = {}) {
  const body = text.trim();
  if (!body || last) return false;
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

/** L5 empty task page — one line only (lock). */
export const EMPTY_TASK_LINE = '从这里开始。';

export const CHAT_SUGGESTIONS = [
  '复现一篇论文的基线',
  '整理比赛数据，给出切分方案',
  '估算一次训练要花多少',
] as const;

/**
 * C6a L5: 进行中 is running work only. A task waiting on the user is 待确认, never 进行中. t237 C6d: the one live set —
 * the process row, the thinking state, the task list and the session all read it, so `pausing` is live everywhere.
 */
export const LIVE_RUN_STATUSES: readonly string[] = ['queued', 'planning', 'running', 'verifying', 'recovering', 'pausing'];
/** Live work plus a pending approval: the session stays active and 停止 is offered. */
export const STOPPABLE_RUN_STATUSES: readonly string[] = [...LIVE_RUN_STATUSES, 'waiting_approval'];
const LIVE_TASK = new Set(LIVE_RUN_STATUSES);
const AWAITING_CONFIRM = new Set(['waiting_approval', 'plan_pending']);

export function isLiveTask(status: string) {
  return LIVE_TASK.has(status);
}

export function isAwaitingConfirm(item: string | { status?: string; plan_pending?: boolean }) {
  if (typeof item === 'string') return AWAITING_CONFIRM.has(item);
  return AWAITING_CONFIRM.has(String(item.status || '')) || item.plan_pending === true;
}

/** CP3 H3-1: 「2 项进行中 · 1 项待确认」; a zero segment is omitted, both zero ⇒ no row. */
export function liveCountSegments(running: number, waiting: number) {
  return [
    ...(running > 0 ? [{ kind: 'live' as const, text: `${running} 项进行中` }] : []),
    ...(waiting > 0 ? [{ kind: 'confirm' as const, text: `${waiting} 项待确认` }] : []),
  ];
}

/** m:ss without a leading zero minute (0:42, 1:12); h:mm:ss past an hour. */
export function formatElapsed(ms: number) {
  const total = Math.max(0, Math.floor((Number(ms) || 0) / 1000));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = String(total % 60).padStart(2, '0');
  return hours ? `${hours}:${String(minutes).padStart(2, '0')}:${seconds}` : `${minutes}:${seconds}`;
}

export type ActivityRow = { kind: string; label: string; status: string; started_at?: string; duration_ms?: number | null; error?: string; id?: string };
export type CurrentActivity = { kind: string; label: string; status: string; started_at?: string; index?: number; total?: number };

/**
 * C6-answer H_a21: the clock starts at run.started_at (execution start, never the queue). Snapshots from before
 * started_at existed fall back to created_at.
 */
export function runStartedAt(run: Pick<TaskRun, 'created_at' | 'started_at'>) {
  const started = new Date(run.started_at || '').getTime();
  if (Number.isFinite(started)) return started;
  const created = new Date(run.created_at || '').getTime();
  return Number.isFinite(created) ? created : null;
}

/** C6-brand H_b15 / C6-answer H_a21: a run's working time, started_at (else created_at) → completed_at; unknown ⇒ null. */
export function workedMs(run: Pick<TaskRun, 'created_at' | 'completed_at' | 'started_at'>): number | null {
  const started = runStartedAt(run);
  const ended = new Date(run.completed_at || '').getTime();
  return started !== null && Number.isFinite(ended) && ended >= started ? ended - started : null;
}

/** C6-answer H_b15′: whole seconds; under a second reads 「不到 1s」 — never 0s, decimals, or a rounded-up 1s. */
export function secondsLabel(ms: number) {
  return ms < 1000 ? '不到 1s' : `${Math.floor(ms / 1000)}s`;
}

/** C6-answer H_a13: the live clock — hidden under 1s, `Ns` under a minute, `1m 05s` from there. */
export function liveClock(ms: number | null) {
  if (ms === null || !Number.isFinite(ms) || ms < 1000) return null;
  const total = Math.floor(ms / 1000);
  return total < 60 ? `${total}s` : `${Math.floor(total / 60)}m ${String(total % 60).padStart(2, '0')}s`;
}

export type ThinkingPhase = 'queued' | 'thinking' | 'acting' | 'answering' | 'done' | 'stopped' | 'failed' | 'waiting';
export type ThinkingRowState = { phase: ThinkingPhase; text: string; clock: string | null; steps: number; live: boolean };

const THINKING_LIVE = new Set(LIVE_RUN_STATUSES.filter(status => status !== 'queued'));
const THINKING_FAILED = new Set(['failed', 'blocked', 'undelivered']);
const ACTION_MAX = 14;

/** H_a13: 「正在{动作}…」 stays within 14 characters; a longer label is cut, never wrapped. */
function actionText(label: string) {
  const body = activityLabel(label).replace(/^正在/, '').replace(/[…。.]+$/, '');
  const room = ACTION_MAX - 3;
  return `正在${body.length > room ? body.slice(0, room) : body}…`;
}

/**
 * C6-answer ThinkingRow: one line for the whole process (CP-A3). Live: 排队中 / 思考中… / 正在{动作}… / 正在整理答案…
 * with a clock from started_at. Done: 「工作了 Ns · N 步」 (tools) or 「思考了 Ns」 (pure reasoning) — N 步 is the
 * expanded timeline's row count, and there is no m:ss. Stopped / failed carry 用时 and never 「工作了」.
 */
/**
 * t237 C6d: has this turn's answer begun? A live run counts only its own stream — `run.answer` on a live continuation
 * is the previous turn's reply. A settled run counts its answer too.
 */
export function answerStarted(run: TaskRun) {
  const streamed = Boolean(String(run.streaming_text || '').trim());
  return LIVE_TASK.has(String(run.status || '')) ? streamed : streamed || Boolean(run.answer);
}

/** t238 C7c-FIX4: the run asked for a training and none succeeded, so a finished reply is an explanation, not a result. */
export const TRAINING_FAILED_STATUS = '没有完成';
export function trainingFailed(run: { status?: string; training_outcome?: string }) {
  return (run.status === 'completed' || run.status === 'delivered') && run.training_outcome === 'failed';
}

/**
 * C7c-FIX5c C2 (Critiquito CLEARING): after a cancel (run cancelled OR this run's GPU job cancelled),
 * do not present artifacts as a successful delivery — no file cards, no 「已通过校验」, no 「生成文件（N）」.
 * Prefer clearing over keeping mid-train files; billing math is unchanged (C5 is copy only).
 */
export const CANCELLED_RESULTS_EMPTY = '没有可用的结果';
export function cancelledTrainingDelivery(
  run: { id?: string; status?: string } | null | undefined,
  gpuJob?: { run_id?: string; state?: string } | null,
): boolean {
  if (!run) return false;
  if (run.status === 'cancelled') return true;
  return Boolean(gpuJob && gpuJob.run_id === run.id && gpuJob.state === 'cancelled');
}
export function resultArtifacts<T>(
  run: { id?: string; status?: string; artifacts?: T[] | null } | null | undefined,
  gpuJob?: { run_id?: string; state?: string } | null,
): T[] {
  if (!run || cancelledTrainingDelivery(run, gpuJob)) return [];
  return run.artifacts ?? [];
}
/** t238 C7c-FIX4: the user declined the confirmation (rejected), so the run stopped by choice. An expired one stays blocked. */
export function stoppedByUser(run: { status?: string; error_class?: string }) {
  return run.status === 'blocked' && run.error_class === 'approval_rejected';
}

/** t238 C7c-FIX5i (critique F5-7): a settled page never lists a step as 尚未执行, and a page that did not finish never ticks the checking step. */
const CHECK_STEP = /验证|校验|核对|检查结果|交付/;
export function settledStepStatus(run: { status?: string; training_outcome?: string }, step: { status?: string; label?: string }): string {
  const status = String(step.status || '');
  const settled = ['completed', 'delivered', 'cancelled'].includes(String(run.status || '')) || trainingFailed(run) || stoppedByUserLike(run);
  if (!settled) return status;
  if (status === 'pending' || status === 'running') return 'skipped';
  if (status === 'completed' && (run.status === 'cancelled' || trainingFailed(run)) && CHECK_STEP.test(String(step.label || ''))) return 'skipped';
  return status;
}
const stoppedByUserLike = (run: { status?: string; error_class?: string }) => stoppedByUser(run);

/** t238 C7c-FIX5: while the run's card is still preparing the training environment the row says so, not 「正在训练模型」. */
export const PREPARING_ROW_TEXT = '正在准备训练环境';
export function thinkingRowState(run: TaskRun, now = Date.now(), gpuJob?: { run_id?: string; state?: string; preparing?: boolean; label?: string } | null): ThinkingRowState {
  const status = String(run.status || '');
  const steps = timelineRows(run.activity ?? [], true).length;
  const started = runStartedAt(run);
  if (status === 'queued') return { phase: 'queued', text: '排队中', clock: null, steps, live: true };
  if (THINKING_LIVE.has(status)) {
    const clock = liveClock(started === null ? null : Math.max(0, now - started));
    if (String(run.streaming_text || '').trim()) return { phase: 'answering', text: '正在整理答案…', clock, steps, live: true };
    // C7c-FIX5c C1: while the GPU card is 「正在取消…」 the process line shares that word (title/badge/card), not 思考中/工具名 ahead of the card.
    if (gpuJob && gpuJob.run_id === run.id && gpuJob.state === 'running' && gpuJob.label === '正在取消…') {
      return { phase: 'acting', text: '正在取消…', clock, steps, live: true };
    }
    const current = run.current_activity;
    if (gpuJob && gpuJob.run_id === run.id && gpuJob.state === 'running' && gpuJob.preparing) return { phase: 'acting', text: PREPARING_ROW_TEXT, clock, steps, live: true };
    if (current?.kind === 'tool' && current.label) return { phase: 'acting', text: actionText(current.label), clock, steps, live: true };
    return { phase: 'thinking', text: '思考中…', clock, steps, live: true };
  }
  const ended = new Date(run.completed_at || run.updated_at || '').getTime();
  const spent = started !== null && Number.isFinite(ended) && ended >= started ? ended - started : null;
  if (trainingFailed(run)) {
    return { phase: 'failed', text: `${TRAINING_FAILED_STATUS}${spent === null ? '' : ` · 用时 ${secondsLabel(spent)}`}`, clock: null, steps, live: false };
  }
  if (status === 'completed' || status === 'delivered') {
    const time = spent === null ? null : secondsLabel(spent);
    const text = steps
      ? time === null ? `已完成 · ${steps} 步` : `工作了${time === '不到 1s' ? '' : ' '}${time} · ${steps} 步`
      : time === null ? '已完成' : `思考了${time === '不到 1s' ? '' : ' '}${time}`;
    return { phase: 'done', text, clock: null, steps, live: false };
  }
  const used = spent === null ? '' : ` · 用时 ${secondsLabel(spent)}`;
  if (stoppedByUser(run) && run.training_outcome === 'succeeded') return { phase: 'done', text: `${RUN_DONE_STATUS}${used}`, clock: null, steps, live: false };
  if (status === 'cancelled' || stoppedByUser(run)) return { phase: 'stopped', text: `已取消${used}`, clock: null, steps, live: false };
  if (THINKING_FAILED.has(status)) return { phase: 'failed', text: `这一步没有完成${used}`, clock: null, steps, live: false };
  return { phase: 'waiting', text: status === 'waiting_approval' ? '等你确认' : status === 'paused' ? '已暂停' : '', clock: null, steps, live: false };
}

/** H_a18: aria-live speaks phase changes only — never the clock. */
export function thinkingAnnouncement(phase: ThinkingPhase) {
  return ({ queued: '排队中', thinking: '开始思考', acting: '开始思考', answering: '正在整理答案', done: '已完成', stopped: '已取消', failed: '这一步没有完成', waiting: '' } as Record<ThinkingPhase, string>)[phase];
}

/** H_a17: first answer character ⇒ the live row folds after this delay (hard cap 800ms). */
export const AUTO_COLLAPSE_MS = 600;

/** H_a17: a user toggle wins for good; otherwise only a live run before its first answer character is open. */
export function thinkingRowOpen(manual: boolean | null, autoOpen: boolean) {
  return manual ?? autoOpen;
}

export type DeliveryMode = 'inline' | 'inline+file' | 'file_only';

/** SPEC §4.1 / H_a10: snapshots without delivery_mode infer it — files ⇒ inline+file (file_only when no answer), else inline. */
export function deliveryMode(run: Pick<TaskRun, 'delivery_mode' | 'artifacts' | 'answer'>): DeliveryMode {
  const files = (run.artifacts ?? []).length > 0;
  if (run.delivery_mode === 'inline' || run.delivery_mode === 'inline+file' || run.delivery_mode === 'file_only') {
    return run.delivery_mode === 'inline' && files ? 'inline+file' : run.delivery_mode;
  }
  if (!files) return 'inline';
  return String(run.answer || '').trim() ? 'inline+file' : 'file_only';
}

/** H_a10/H_a11b: attachment chips only in a file mode; at most 3, the rest fold into 「还有 N 个文件」. */
export const CHIP_LIMIT = 3;
export function attachmentChips<T>(mode: DeliveryMode, files: T[]) {
  if (mode === 'inline' && !files.length) return { shown: [] as T[], more: [] as T[] };
  return { shown: files.slice(0, CHIP_LIMIT), more: files.slice(CHIP_LIMIT) };
}

/** H_a12 / SPEC §4.5: files with no answer text still get one inline line above them. */
export function filesSummaryLine(names: string[]) {
  if (!names.length) return '';
  const shown = names.slice(0, 3).map(name => `\`${name}\``).join('、');
  return `已生成 ${shown}${names.length > 3 ? ` 等 ${names.length} 个文件` : ''}。`;
}

/** S_a1: a small text deliverable (≤2 KB, ≤40 lines; md/txt/csv/json) previews inline under its chip. */
export const SMALL_PREVIEW_BYTES = 2048;
export const SMALL_PREVIEW_LINES = 40;
export function smallPreviewable(file: { name: string; bytes: number }) {
  return file.bytes > 0 && file.bytes <= SMALL_PREVIEW_BYTES && /\.(?:md|txt|csv|json)$/i.test(file.name);
}
export function smallPreviewText(text: string) {
  const lines = text.split(/\r?\n/);
  return lines.length > SMALL_PREVIEW_LINES ? null : text.replace(/\s+$/, '');
}

/**
 * C6c H_c6c_s4: one stuck surface per run — the model fail card, else the stale-heartbeat banner, else the blocked
 * progress card (with its one assistant line). Never two of them for the same stall.
 */
export function stuckSurface(run: Pick<TaskRun, 'status' | 'error_class' | 'error_message' | 'answer' | 'progress'>): 'model' | 'banner' | 'card' | null {
  if (isModelFailure(run)) return 'model';
  if (isStuckBanner(run.progress, run.status)) return 'banner';
  return blockedProgressCopy(run) ? 'card' : null;
}

/** C6b L13 / H5-1: stuck banner only when freshness is stale (≥45s or no heartbeat). Fresh heartbeat ⇒ never. */
export function isStuckBanner(progress?: { freshness?: string; heartbeat_age_seconds?: number | null } | null, status?: string) {
  if (!status || ['completed', 'delivered', 'cancelled', 'failed', 'blocked', 'paused', 'waiting_approval', 'draft'].includes(status)) return false;
  return progress?.freshness === 'stale';
}

/** C6b H5-1: 「{N} 秒无进展」 — whole seconds; ≥60 keeps seconds for product consistency (S5-1). */
export function stuckSecondsLabel(ageSeconds: number | null | undefined) {
  const n = Math.max(0, Math.floor(Number(ageSeconds) || 0));
  return `${n || 45} 秒无进展`;
}

const THINKING_LEAK = /(?:^|[\s`'"(])\/?(?:inputs|outputs|workspace|home|var|tmp)\/\S*|[A-Za-z0-9_-]{8,}=[A-Za-z0-9_\-+/=]{12,}|sk-[A-Za-z0-9_-]{4,}|-----BEGIN |api[_ -]?key|tool_call|call_[a-z0-9]+/i;

/** C6b L14 / H5-6: thinking summary never carries paths, keys, tool ids, or raw stacks. */
export function sanitizeThinkingSummary(text?: string | null) {
  const raw = String(text || '').trim();
  if (!raw) return '';
  const cleaned = raw.split(/\n+/).map(line => THINKING_LEAK.test(line) ? '' : line.trim()).filter(Boolean).join(' ').slice(0, 400);
  return cleaned;
}

/** A label from the runtime is already Chinese; anything shaped like a tool id never reaches the screen. */
export function activityLabel(label?: string) {
  const text = String(label || '').trim();
  return !text || RAW_TOOL_ID.test(text) || /[/{}]/.test(text) ? '处理一步' : text;
}

/** CP1 H1-3 / S1-2: consecutive finished repeats merge (「搜索资料 ×3」); collapsed view keeps the last 3 rows. */
export function timelineRows(rows: ActivityRow[], expanded: boolean) {
  const merged: Array<ActivityRow & { count: number }> = [];
  for (const raw of rows) {
    const row = { ...raw, label: activityLabel(raw.label), count: 1 };
    const last = merged.at(-1);
    // C6c H_c6c_s4: a step that keeps failing the same way folds into one row too (「写入文件 ×3 · 这一步没有完成」).
    const same = last && last.status === row.status && (row.status === 'completed' || (row.status === 'failed' && (last.error || '') === (row.error || '')));
    if (last && same && last.label.replace(/ ×\d+$/, '') === row.label) {
      last.count += 1;
      last.label = `${row.label} ×${last.count}`;
      last.duration_ms = (last.duration_ms ?? 0) + (row.duration_ms ?? 0);
      continue;
    }
    merged.push(row);
  }
  return expanded ? merged : merged.slice(-3);
}

const elapsedSince = (stamp?: string) => {
  const time = new Date(stamp || '').getTime();
  return Number.isFinite(time) ? Math.max(0, Date.now() - time) : null;
};
export { elapsedSince };

/** Run selector option text; a missing or unparseable timestamp is omitted rather than printed as "Invalid Date". */
export function runOptionLabel(createdAt: string | null | undefined, statusLabel: string) {
  const time = new Date(createdAt || '').getTime();
  return Number.isFinite(time) ? `${new Date(time).toLocaleString('zh-CN')} · ${statusLabel}` : statusLabel;
}

type ActivityRun = { activity?: ActivityRow[]; activity_total?: number; current_activity?: CurrentActivity | null; current_step?: CurrentActivity | null };
type ActivityPayload = { tool_call_id?: string; label?: string; started_at?: string; status?: string; duration_ms?: number | null; error_label?: string; step_id?: string; index?: number; total?: number };

/** C6a: one SSE tool/step event updates the timeline in place; the next snapshot stays authoritative. */
export function applyActivityEvent<T extends ActivityRun>(run: T, type: string, payload: ActivityPayload): T {
  const rows = [...(run.activity ?? [])];
  const step = run.current_step ?? (run.current_activity?.kind === 'step' ? run.current_activity : null);
  if (type === 'step_started') {
    const current: CurrentActivity = { kind: 'step', label: activityLabel(payload.label), status: 'running', started_at: payload.started_at || new Date().toISOString(),
      ...(payload.index ? { index: payload.index } : {}), ...(payload.total ? { total: payload.total } : {}) };
    return { ...run, current_step: current, current_activity: run.current_activity?.kind === 'tool' ? { ...run.current_activity, index: current.index, total: current.total } : current };
  }
  if (type === 'tool_started') {
    if (payload.tool_call_id && rows.some(row => row.id === payload.tool_call_id)) return run;
    const row: ActivityRow = { kind: 'tool', id: payload.tool_call_id, label: activityLabel(payload.label), status: 'running', started_at: payload.started_at || new Date().toISOString(), duration_ms: null };
    return { ...run, activity: [...rows, row], activity_total: (run.activity_total ?? rows.length) + 1, current_step: step,
      current_activity: { kind: 'tool', label: row.label, status: 'running', started_at: row.started_at, ...(step?.index ? { index: step.index, total: step.total } : {}) } };
  }
  if (type === 'tool_completed') {
    let at = rows.findIndex(row => payload.tool_call_id && row.id === payload.tool_call_id);
    if (at < 0) at = rows.findLastIndex(row => row.status === 'running' && row.label === activityLabel(payload.label));
    if (at < 0) return run;
    const failed = payload.status === 'failed';
    rows[at] = { ...rows[at], status: failed ? 'failed' : 'completed', duration_ms: payload.duration_ms ?? null, ...(failed ? { error: payload.error_label || '这一步没有完成' } : {}) };
    const stillRunning = rows.findLast(row => row.status === 'running');
    return { ...run, activity: rows, current_step: step,
      current_activity: stillRunning ? run.current_activity : step };
  }
  if (type === 'user_steered') {
    const summary = activityLabel((payload as { summary?: string }).summary || '你补充了…').slice(0, 40);
    const row: ActivityRow = { kind: 'steer', label: summary, status: 'completed', started_at: new Date().toISOString(), duration_ms: null };
    return { ...run, activity: [...rows, row], activity_total: (run.activity_total ?? rows.length) + 1 };
  }
  return run;
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
export function streamPhaseState(run: { status?: string; error_class?: string; error_message?: string; answer?: string; training_outcome?: string }): PhaseState {
  const status = String(run.status || '');
  const copy = blockedProgressCopy(run);
  if (trainingFailed(run)) return { current: 1, stuck: true, done: false, copy: { status: TRAINING_FAILED_STATUS, action: '可以稍后再试一次' } };
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
  if (status === '这次没有完成' || status === TRAINING_FAILED_STATUS) return '这次没有完成，我先检查再接着跑。';
  if (status === '尚未交付') return '还没有交付，我先检查缺了什么。';
  if (status === '有一步还没核对完') return '有一步还没核对完，我先核对再继续。';
  if (status === TRAINING_INPUT_LIMIT_STATUS) return TRAINING_INPUT_LIMIT_LINE;
  return '这一步停住了，我先检查再接着跑。';
}

const DANGER_CONFIRM = /delete|submit|kaggle|external|shell|hpc|publish/i;
const OUTBOUND_OR_OVER_FLAG = /over_budget|exceeds|external_submit|kaggle/i;

/** Spend checks that do not send work out. Every other tool keeps the separate confirmation. */
// experience_compute (C4b): the confirmed-plan compute ask; low risk, zero spend, so it reads as the in-chat 等你确认 card.
const BUDGET_INLINE_TOOLS = new Set(['cost_estimate', 'experience_compute']);

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

/** C4b: an experience plan card still waiting for 确认计划 / 取消 — the run is completed, but the user owes a decision. */
export const PLAN_PENDING_STATUS = '等你确认';
export function planPending(run: { plan_proposed?: unknown; plan_decision?: string }) {
  return Boolean(run.plan_proposed) && !run.plan_decision;
}

/** C4c: neutral status while a run snapshot is still being read — never a terminal word like 已交付. */
export const STATUS_LOADING = '正在读取';
type EyebrowRun = { id?: string; status?: string; plan_proposed?: unknown; plan_decision?: string; plan_pending?: boolean; approvals?: Array<{ status?: string }> };
/** A run the user still owes a decision on: an undecided plan card, or an approval waiting. */
export function confirmPending(run: EyebrowRun) {
  return planPending(run) || run.plan_pending === true || run.status === 'waiting_approval' || Boolean(run.approvals?.some(item => item.status === 'pending'));
}
/**
 * C4c: the title status line, decided before the first paint. While the snapshot is in flight it uses what the run list
 * already says (pending ⇒ 等你确认); if that cannot say, it is neutral 正在读取 — the task's own status (已交付 for a
 * pending plan) is never shown as a stand-in.
 */

/** Five lock pill kinds mapped onto existing Chinese status strings (visual only; copy unchanged). */
export type StatusPillKind = 'delivered' | 'confirm' | 'progress' | 'idle' | 'error';

const PILL_DELIVERED = new Set(['已交付', '执行已完成', '已完成']);
const PILL_CONFIRM = new Set(['等你确认']);
const PILL_IDLE = new Set(['尚未开始', '已暂停', '已取消', '已停止']);
const PILL_ERROR = new Set([
  '执行失败', '暂时无法继续', '未交付', '状态待确认', '不可用',
  // blockedProgressCopy headings (progress-card first line when phase.stuck)
  '这一步停住了', '结果还没齐', '方案停住了', '有一步还没核对完', '尚未交付', '这次没有完成',
]);

export function statusPillKind(label: string): StatusPillKind {
  const text = String(label || '').trim();
  if (PILL_DELIVERED.has(text)) return 'delivered';
  if (PILL_CONFIRM.has(text)) return 'confirm';
  if (PILL_IDLE.has(text)) return 'idle';
  if (PILL_ERROR.has(text) || /失败|无法继续|没处理完|不可用|用不了|停住了|还没齐|没有完成|还没核对/.test(text)) return 'error';
  return 'progress';
}

export function statusPillClass(label: string) {
  return `tw-status-pill tw-status-${statusPillKind(label)}`;
}

/** C6c W3 (H_c6c_s2): the one status word for a run — the title chip and the progress card both read it. */
export const RUN_DONE_STATUS = '已完成';
export const RUN_STOPPED_STATUS = '已取消';
export function runStatusWord(run: EyebrowRun & { error_class?: string; error_message?: string; answer?: string; training_outcome?: string }, taskStatus: (status: string) => string, taskState = '') {
  const status = String(run.status || '');
  if (confirmPending(run)) return PLAN_PENDING_STATUS;
  if (trainingFailed(run)) return TRAINING_FAILED_STATUS;
  // A finished training followed by a declined later step is still a finished task: the declined step is only a note.
  if (stoppedByUser(run)) return run.training_outcome === 'succeeded' ? RUN_DONE_STATUS : RUN_STOPPED_STATUS;
  // The task's delivery check outranks a finished run: 未交付 on both the title chip and the card.
  if ((status === 'completed' || status === 'delivered') && taskState === 'undelivered') return taskStatus(taskState);
  if (status === 'completed' || status === 'delivered') return RUN_DONE_STATUS;
  if (status === 'cancelled') return RUN_STOPPED_STATUS;
  if (LIVE_TASK.has(status)) return taskStatus(status);
  return blockedProgressCopy(run)?.status || taskStatus(status);
}

export function chatEyebrow({ run, runs, runId, restoring = false, task, taskStatus }: {
  run: EyebrowRun | null; runs: EyebrowRun[]; runId: string; restoring?: boolean;
  task: { status: string }; taskStatus: (status: string) => string;
}) {
  const settled = (item: EyebrowRun) => runStatusWord(item, taskStatus, task.status);
  if (run) return settled(run);
  const known = runId ? runs.find(item => item.id === runId) : restoring ? runs[0] : undefined;
  if (!known) return taskStatus(task.status);
  if (confirmPending(known)) return PLAN_PENDING_STATUS;
  // The task-detail run summary carries plan_pending (C4c); without it a completed run cannot be told apart from a pending plan.
  return known.plan_pending === false ? settled(known) : STATUS_LOADING;
}

/** t238 C7c-FIX5: the card for a confirmation that is not a budget step. A training says what it will do; nothing shows a tool id or 「受控操作」. */
const TRAINING_TOOLS = new Set(['managed_tensor_train', 'tenant_tensor_train']);
export function separateConfirmCopy(approval: { tool_name?: string }) {
  const name = String(approval.tool_name || '');
  if (name === 'file_delete') return { title: '删除文件', target: undefined, reversible: undefined, refusal: '拒绝后不会执行这项操作，任务会保留并等待调整。',
    scope: '仅确认当前目标和参数；涉及资源的操作仍须符合已批准上限，不会因本次确认扩大权限。', showDetails: true,
    consent: '我已核对目标与影响', approve: '确认执行', reject: '拒绝这项操作' };
  if (TRAINING_TOOLS.has(name)) return { title: '开始训练这个模型', target: '用这次任务里的数据训练一个模型', reversible: '确认后才会开始训练。',
    refusal: '拒绝后不会开始，任务会保留，可以调整后再试。', scope: '只确认这一次训练，不会因此提高你的训练时长上限。', showDetails: false,
    consent: '我确认开始这次训练', approve: '开始训练', reject: '先不训练' };
  return { title: '继续这一步', target: undefined, reversible: undefined, refusal: '拒绝后不会执行这项操作，任务会保留并等待调整。',
    scope: '仅确认当前目标和参数；涉及资源的操作仍须符合已批准上限，不会因本次确认扩大权限。', showDetails: true,
    consent: '我已核对目标与影响', approve: '确认执行', reject: '拒绝这项操作' };
}


/** 2026-11-03: the plan the runtime attaches to a training approval (impact_scope.training_plan), as card rows. */
export type TrainingPlanRow = { label: string; value: string };
export function trainingPlanRows(approval: { impact_scope?: Record<string, unknown> }): TrainingPlanRow[] {
  const plan = approval.impact_scope?.training_plan;
  if (!plan || typeof plan !== 'object') return [];
  const p = plan as Record<string, unknown>;
  const num = (value: unknown) => (typeof value === 'number' && Number.isFinite(value) ? value : null);
  const rows: TrainingPlanRow[] = [];
  const rowsCount = num(p.rows); const features = num(p.features);
  if (typeof p.data_name === 'string') rows.push({ label: '数据', value: `${p.data_name}${rowsCount !== null && features !== null ? `（${rowsCount} 行 × ${features} 个特征）` : ''}` });
  if (typeof p.task_label === 'string') rows.push({ label: '训练场景', value: `${p.task_label}，预测目标：${String(p.target_column ?? '最后一列')}` });
  const hidden = Array.isArray(p.hidden_sizes) ? p.hidden_sizes.filter(item => typeof item === 'number') : [];
  if (hidden.length) rows.push({ label: '推荐算法', value: `${String(p.network ?? '全连接神经网络')}，隐藏层 ${hidden.join(' → ')}` });
  const epochs = num(p.epochs); const batch = num(p.batch_size); const lr = num(p.learning_rate);
  if (epochs !== null && batch !== null) rows.push({ label: '训练参数', value: `${epochs} 轮，批大小 ${batch}${lr !== null ? `，学习率 ${lr}` : ''}` });
  const train = num(p.train_rows); const holdout = num(p.holdout_rows);
  if (train !== null && holdout !== null) rows.push({ label: '数据划分', value: `${train} 行训练，${holdout} 行留作评估` });
  const eta = p.estimated_gpu_seconds as Record<string, unknown> | undefined;
  const low = num(eta?.low); const high = num(eta?.high);
  if (low !== null && high !== null) rows.push({ label: '预计训练时间', value: high < 120 ? `约 ${low}–${high} 秒（显卡上）` : `约 ${Math.max(1, Math.round(low / 60))}–${Math.ceil(high / 60)} 分钟（显卡上）` });
  if (Array.isArray(p.metrics) && p.metrics.length) rows.push({ label: '评估指标', value: p.metrics.map(String).join(' / ') });
  if (typeof p.billing_note === 'string') rows.push({ label: '计费', value: p.billing_note });
  return rows;
}

export function confirmActionLabel(tool?: string) {
  const name = String(tool || '');
  if (name === 'file_write' || name === 'write_file') return '写入这次任务的文件';
  // C4: experience compute approval — Chinese product label; never surface HPC/tool ids in default view
  if (name === 'experience_compute') return '使用算力继续';
  if (!name || RAW_TOOL_ID.test(name)) return '继续推进当前方案';
  return name;
}

/** C4c: a real positive amount only. 0, negative, NaN or unknown reads as "not known yet" — the card then omits the line. */
const money = (value: unknown) => (typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : null);
export function confirmSpendLabel(approval: { normalized_arguments?: Record<string, unknown> }) {
  const args = approval.normalized_arguments || {};
  const amount = money(args.estimated_cost ?? args.cost ?? args.amount);
  const cap = money(args.budget ?? args.limit ?? args.cap);
  if (amount !== null && cap !== null) return `约 ¥${amount}，上限 ¥${cap}`;
  if (amount !== null) return `约 ¥${amount}`;
  if (cap !== null) return `上限 ¥${cap}`;
  return '';
}

/** One short status for a blocked or undelivered progress card. No filenames or internal codes. */
export function blockedProgressCopy(run: { status?: string; error_class?: string; error_message?: string; answer?: string }) {
  const status = String(run.status || '');
  const blob = `${run.error_class || ''}\n${run.error_message || ''}\n${run.answer || ''}`;
  if (run.error_class === 'model_unavailable' || blob.includes('Model execution is unavailable:')) return null;
  const relevant = status === 'blocked' || status === 'failed' || status === 'undelivered'
    || /required result artifacts unavailable|missing final artifacts|\brecoverable\b/i.test(blob);
  if (!relevant || status === 'waiting_approval' || stoppedByUser(run)) return null;
  if (run.error_class === 'execution_reconciliation_required' || run.error_class === 'approval_execution_unconfirmed') {
    return { status: '有一步还没核对完', action: '先核对这一步' };
  }
  if (isTrainingInputLimit(blob)) return { status: TRAINING_INPUT_LIMIT_STATUS, action: '检查并继续' };
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

/** The run stopped on its model (unavailable, not the requested one, or an internal fault); the card then decides which single action to offer. */
export function isModelFailure(run: Pick<TaskRun, 'status' | 'error_class' | 'answer'>) {
  return run.status === 'blocked' && (run.error_class === 'model_unavailable' || run.error_class === 'model_identity_unconfirmed' || Boolean(run.answer?.includes('Model execution is unavailable:')));
}

/** The one line for a run that stopped on an internal fault (encoding/transport/exception). */
export const INTERNAL_RUN_ERROR_LINE = '这次没处理完，可以再试一次';

/** The safe `LLMError/<code>` the runtime names in a failed model turn; '' when the text carries none. */
export function modelFailureCode(run: Pick<TaskRun, 'error_message' | 'answer'>) {
  return /LLMError\/([A-Za-z0-9_]{1,80})/.exec(`${run.answer || ''}\n${run.error_message || ''}`)?.[1] ?? '';
}

// Genuine configuration/auth faults: the key or address is wrong, the profile is off, or the model is missing.
const CONFIG_FAULT = /^(?:model_)?http_(?:401|403|404)$|^model_(?:profile_disabled|address_invalid|identity_unconfirmed|contract_drift|contract_invalid|not_found)$|^(?:legacy_model_contract_requires_review|no_model_transport)$/;

/** 「检查模型配置」 is only for these. Encoding, transport and exception failures are internal and get a retry instead. */
export function isConfigModelFailure(run: Pick<TaskRun, 'status' | 'error_class' | 'error_message' | 'answer'>) {
  if (!isModelFailure(run)) return false;
  if (run.error_class === 'model_unavailable' || run.error_class === 'model_identity_unconfirmed') return true;
  return CONFIG_FAULT.test(modelFailureCode(run));
}

/** t238 C7c-FIX5 H4: the composer's chip row holds only files not yet sent; sent files stay bound to the task. */
export function pendingAttachments<T extends { id: string }>(files: T[], sentIds: string[]): T[] {
  return files.filter(file => !sentIds.includes(file.id));
}
