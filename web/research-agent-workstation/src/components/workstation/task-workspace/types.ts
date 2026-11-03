import type { ActivityRow, CurrentActivity } from './task-presentation';
// @ts-expect-error Native Node tests resolve the explicit .ts extension.
import { timeoutSignal } from './abort-signal.ts';
// @ts-expect-error Native Node tests resolve the explicit .ts extension.
import { isUploadLimitCode, uploadLimitLine } from './error-copy.ts';

export type UserTask = {
  id: string; title: string; draft: string; version: number; conversation_id: string;
  status: string; created_at: string; updated_at: string;
  pinned_at?: string | null; archived_at?: string | null; project_id?: string | null; archived?: boolean;
  /** Search results only: one plain line around the match. */
  snippet?: string;
  /** t237 C6e G14: newest run's status, for the bell's reconcile (task list only). */
  latest_run?: { id: string; status: string; plan_pending?: boolean; updated_at?: string } | null;
  /** Client only: soft-deleted while its undo toast is open. */
  deleted?: boolean;
};
export type Project = { id: string; name: string; instructions?: string; version?: number; conversation_count?: number; created_at?: string; updated_at?: string };
export type TaskArtifact = { id: string; run_id: string; name: string; bytes: number; sha256: string; verified_at?: string; preview_kind: string; source_tool_call?: string };
export type RunProgress = {
  schema?: string; work_kind?: string; phase?: string; freshness?: string; stalled?: boolean;
  heartbeat_at?: number | null; heartbeat_age_seconds?: number | null; last_progress_at?: number | null;
  completed_units?: number | null; total_units?: number | null; unit?: string | null;
  percent?: number | null; detail?: string; source?: string; objective_outcome?: string; execution_terminal?: boolean;
};

export type TaskRun = {
  id: string; user_task_id?: string; conversation_id?: string; status: string; prompt?: string; answer?: string; completed_at?: string;
  /** t237 C6b L9: produced text kept after Stop (cancel). */
  partial_answer?: string | null;
  /** t237 C6b L13: executor progress (freshness/stalled/heartbeat). */
  progress?: RunProgress | null;
  /** t237 C6b L14: summarized thinking only; never raw thinking / paths / keys. */
  reasoning_summary?: string | null;
  /** t237 C6b L10: pending queue rows (preview only; not sent bubbles). */
  queued_messages?: Array<{ text: string; idempotency_key?: string; status?: string; created_at?: string }>;
  /** t237 C6b L11: pending immediate-send items awaiting the next tool boundary. */
  pending_steer?: Array<{ text: string; summary?: string; idempotency_key?: string; status?: string }>;
  /** t237 C6a: live process timeline rebuilt from the snapshot, then updated by SSE tool/step events. */
  activity?: ActivityRow[]; activity_total?: number; current_activity?: CurrentActivity | null; current_step?: CurrentActivity | null;
  streaming_text?: string | null; answer_is_current?: boolean; terminal?: boolean; model?: string; model_provider?: string;
  error_class?: string; error_message?: string; /** t238 C7c-FIX4: 'failed' | 'succeeded' | '' for a managed training asked in this run. */ training_outcome?: string; created_at?: string; updated_at?: string; last_event_seq?: number;
  /** t237 C6-answer H_a21: execution start (never queue/creation); '' while queued. */
  started_at?: string;
  /** t237 C6-answer SPEC §4.1: how the answer is delivered. Older snapshots omit it (see deliveryMode). */
  delivery_mode?: 'inline' | 'inline+file' | 'file_only';
  model_profile?: { id: string; version: number; provider: string; model: string } | null;
  message_history?: { schema: string; messages: Array<{ id: string; role: 'user' | 'assistant'; content: string; kind?: string }>; active_message_id: string };
  plan?: { steps?: Array<{ id: string; label: string; status: string; detail?: string }> };
  artifacts?: TaskArtifact[]; artifact_evidence?: TaskArtifact[];
  /** t237 C3: long-term memories this run recorded ({id, text}). */
  memory_saved?: Array<{ id: string; text: string }>;
  /** t237 C6c: memories injected into this run's prompts, and the run's memory mode (default on). */
  memory_used?: Array<{ id: string; text: string }>;
  memory_mode?: 'on' | 'off' | 'incognito';
  /** t237 C3.10: project files this run could not read (project_files_unreadable event); total / failed mark a full miss. */
  project_files_unreadable?: Array<{ name: string; error_class: string }>; project_files_total?: number; project_files_failed?: boolean;
  /** t237 C3.11: experience entry (intent=experience) and its plan card. */
  intent?: string; plan_proposed?: { references: Array<{ title: string; summary: string }>; steps: string[]; needs_compute: boolean } | null; plan_decision?: string; plan_pending?: boolean;
  approvals?: Array<{ id: string; status: string; tool_name: string; risk_level: string; reversible: boolean; normalized_arguments?: Record<string, unknown>; impact_scope?: Record<string, unknown> }>;
  /** t238 C7e A1/A2: visible backend switch, the user's pre-selected alternate, and the classified model failure. */
  model_failover?: { notice: string; label: string; explicit: boolean } | null;
  fallback_model_id?: string | null;
  attachment_ids?: string[];
  model_error?: { kind: string; code: string; explicit?: boolean; platform_model_id?: string | null } | null;
};
/** t238 C7a D2: advisory precheck against the training entry's input bound; uploads stay usable for chat and analysis. */
export type Trainability = { trainable: boolean; reasons: string[]; limits: { max_files: number; max_bytes: number }; bytes: number; files: number };
export type UserFile = { id: string; name: string; bytes: number; sha256: string; media_type: string; created_at: string; trainability?: Trainability };
export type GeneratedFile = TaskArtifact & { media_type?: string; download_url?: string; preview_url?: string };
export type TaskDetail = { task: UserTask; runs: TaskRun[]; files?: UserFile[]; generated_files?: GeneratedFile[] };

export const taskStatus = (status: string) => ({
  draft: '尚未开始', queued: '等待开始', planning: '正在整理方案', running: '正在处理', verifying: '正在检查结果',
  completed: '执行已完成', delivered: '已交付', undelivered: '未交付', failed: '执行失败', blocked: '暂时无法继续', waiting_approval: '等你确认',
  paused: '已暂停', pausing: '正在暂停', cancelled: '已取消', recovering: '正在恢复',
}[status] ?? '状态待确认');

/** Keeps the HTTP status and BE code so the UI can map them to product copy (error-copy.ts). */
export class TaskRequestError extends Error {
  readonly status: number; readonly code: string;
  constructor(message: string, status: number, code: string) { super(message); this.name = 'TaskRequestError'; this.status = status; this.code = code; }
}

export async function taskRequest<T>(path: string, init: RequestInit = {}): Promise<T> {
  const signal = timeoutSignal(15000, init.signal ?? undefined);
  let response: Response;
  try {
    response = await fetch(path, { ...init, cache: 'no-store', signal });
  } catch (cause) {
    if (init.signal?.aborted) throw cause;
    if (signal.aborted || (cause instanceof Error && cause.name === 'TimeoutError')) {
      throw new TaskRequestError('网络有点慢，输入仍保留。请先重新读取状态，核对是否已保存或开始，再重试。', 0, 'network');
    }
    throw new TaskRequestError('网络连接中断，输入仍保留。恢复连接后请先核对状态，再重试。', 0, 'network');
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const known = ({
      model_address_invalid: '服务地址只接受公网 HTTPS 域名与 443 端口，请移除账号、查询参数或内网地址。',
      invalid_model_profile: '请检查配置名称、服务类型、模型名称和密钥格式；密钥不能含空白字符。',
      model_key_required: '首次保存需要填写 API 密钥。',
      model_profile_disabled: '该模型已停用，请选择其他配置；系统不会自动换用平台模型。',
      model_profile_deleted: '该模型已删除，请选择其他配置；系统不会自动换用平台模型。',
      model_profile_version_conflict: '模型配置已在其他页面修改。请重新打开配置，核对后再保存或发送。',
      model_test_hourly_limit: '此账户一小时内已测试 3 次，请稍后再试。',
      model_credential_storage_unavailable: '服务器的加密存储或目录权限未通过检查，配置未保存。请联系管理员；本页输入仍保留。',
      task_has_active_run: '此任务还有运行或待确认事项，请先处理当前运行，不能重复启动。',
      explicit_tool_task_link_required: '此任务尚未关联旧工具记录，不能通过输入内部任务编号自动关联。',
    } as Record<string, string>)[String(body.error)];
    const code = String(body.error ?? '').slice(0, 120);
    if (isUploadLimitCode(code)) throw new TaskRequestError(uploadLimitLine(code, body.message), response.status, code);
    if (known) throw new TaskRequestError(known, response.status, code);
    const explanation = ({ 401: '登录已过期，请重新登录。', 403: '没有执行此操作的权限。',
      404: '记录不存在，或不属于当前账户。', 409: '记录已更新或操作冲突。请重新读取后再试，当前输入已保留。',
    } as Record<number, string>)[response.status] ?? '暂时无法完成操作，请稍后重试。';
    throw new TaskRequestError(explanation, response.status, code);
  }
  return body as T;
}

export const jsonPost = (body: unknown): RequestInit => jsonRequest('POST', body);
// The web proxy requires a JSON body on every mutation, DELETE included.
export const jsonRequest = (method: 'POST' | 'PATCH' | 'DELETE', body: unknown = {}): RequestInit => ({ method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
