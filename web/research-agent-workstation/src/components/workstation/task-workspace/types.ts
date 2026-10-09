export type UserTask = {
  id: string; title: string; draft: string; version: number; conversation_id: string;
  status: string; created_at: string; updated_at: string;
};
export type TaskArtifact = { id: string; run_id: string; name: string; bytes: number; sha256: string; verified_at?: string; preview_kind: string; source_tool_call?: string };
export type TaskRun = {
  id: string; user_task_id?: string; conversation_id?: string; status: string; prompt?: string; answer?: string;
  streaming_text?: string | null; answer_is_current?: boolean; terminal?: boolean; model?: string; model_provider?: string;
  error_class?: string; error_message?: string; created_at?: string; last_event_seq?: number;
  model_profile?: { id: string; version: number; provider: string; model: string } | null;
  message_history?: { schema: string; messages: Array<{ id: string; role: 'user' | 'assistant'; content: string; kind?: string }>; active_message_id: string };
  plan?: { steps?: Array<{ id: string; label: string; status: string; detail?: string }> };
  artifacts?: TaskArtifact[]; artifact_evidence?: TaskArtifact[];
  approvals?: Array<{ id: string; status: string; tool_name: string; risk_level: string; reversible: boolean; normalized_arguments?: Record<string, unknown>; impact_scope?: Record<string, unknown> }>;
};
export type UserFile = { id: string; name: string; bytes: number; sha256: string; media_type: string; created_at: string };
export type TaskDetail = { task: UserTask; runs: TaskRun[]; files?: UserFile[] };

export const taskStatus = (status: string) => ({
  draft: '尚未开始', queued: '等待开始', planning: '正在整理方案', running: '正在处理', verifying: '正在检查结果',
  completed: '执行已完成', failed: '执行失败', blocked: '暂时无法继续', waiting_approval: '等待你确认',
  paused: '已暂停', pausing: '正在暂停', cancelled: '已取消', recovering: '正在恢复',
}[status] ?? '状态待确认');

export async function taskRequest<T>(path: string, init: RequestInit = {}): Promise<T> {
  const timeout = AbortSignal.timeout(15000);
  let response: Response;
  try {
    response = await fetch(path, { ...init, cache: 'no-store', signal: init.signal ? AbortSignal.any([init.signal, timeout]) : timeout });
  } catch (cause) {
    if (init.signal?.aborted) throw cause;
    if (timeout.aborted || (cause instanceof Error && cause.name === 'TimeoutError')) {
      throw new Error('请求超时，输入仍保留。请先重新读取状态，核对是否已保存或开始，再重试。');
    }
    throw new Error('网络连接中断，输入仍保留。恢复连接后请先核对状态，再重试。');
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const known = ({
      model_address_invalid: '服务地址只接受公网 HTTPS 域名与 443 端口，请移除账号、查询参数或内网地址。',
      invalid_model_profile: '请检查配置名称、服务类型、模型名称和密钥格式；密钥不能含空白字符。',
      model_key_required: '首次保存需要填写 API 密钥。',
      model_profile_disabled: '该模型已停用，请选择其他配置；系统不会自动换用平台模型。',
      model_profile_version_conflict: '模型配置已在其他页面修改。请重新打开配置，核对后再保存或发送。',
      model_test_hourly_limit: '此账户一小时内已测试 3 次，请稍后再试。',
      model_credential_storage_unavailable: '服务器的加密存储或目录权限未通过检查，配置未保存。请联系管理员；本页输入仍保留。',
      task_has_active_run: '此任务还有运行或待确认事项，请先处理当前运行，不能重复启动。',
      explicit_tool_task_link_required: '此任务尚未关联旧工具记录，不能通过输入内部任务编号自动关联。',
    } as Record<string, string>)[String(body.error)];
    if (known) throw new Error(known);
    const explanation = ({ 401: '登录已过期，请重新登录。', 403: '没有执行此操作的权限。',
      404: '记录不存在，或不属于当前账户。', 409: '记录已更新或操作冲突。请重新读取后再试，当前输入已保留。',
    } as Record<number, string>)[response.status] ?? '暂时无法完成操作，请稍后重试。';
    throw new Error(`${explanation}${body.error ? `（${String(body.error).slice(0, 120)}）` : ''}`);
  }
  return body as T;
}

export const jsonPost = (body: unknown): RequestInit => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
