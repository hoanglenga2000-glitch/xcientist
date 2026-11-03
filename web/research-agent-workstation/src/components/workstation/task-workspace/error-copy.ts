// t237 C2a: one mapper from a failed request to Chinese product copy (Critiquito-final, ERROR_COPY_DRAFT.md md5 652453b0).
// The main line never carries HTTP codes, English or error_class; those go only into the closed 技术详情.
export const ERROR_COPY = {
  network: { line: '网络断了，连上后再试。', action: '重试' },
  401: { line: '登录已过期，请重新登录。', action: '重新登录' },
  403: { line: '你没有权限打开这项内容。', action: '返回' },
  404: { line: '找不到这项内容，可能已被删除。', action: '返回' },
  409: { line: '内容刚被更新，请刷新后再改。', action: '刷新' },
  413: { line: '文件太大，请换一个小一点的。', action: '重新选择文件' },
  '5xx': { line: '服务暂时不可用，请稍后再试。', action: '重试' },
  unknown: { line: '没能完成这次操作，请再试一次。', action: '重试' },
  // Model failures share this line; a page without a same-model retry shows 「检查模型配置」 instead. Never auto-switch models.
  model_identity_unconfirmed: { line: '这次没有拿到模型回复，对话已保留。', action: '重试' },
} as const;
export type ErrorCopyKey = keyof typeof ERROR_COPY;

// t238 C7a: upload rejections keep the server's Chinese line (it names the quota and current use); these are the fallbacks.
// upload_completing (409): another request is still merging and verifying this upload.
export const UPLOAD_LIMIT_COPY = {
  quota_exceeded: { line: '存储空间已用完。删掉不用的文件后再上传。', action: '管理文件' },
  disk_low_water: { line: '暂时无法上传。请稍后再试，或联系管理员。', action: '稍后再试' },
  upload_completing: { line: '文件还在处理，稍后点「刷新」。', action: '刷新' },
  // t238 C7b D6: delete / export refusals keep the server's Chinese line (it names the task).
  attachment_in_use: { line: '有任务正在运行，结束后才能删除这个文件。', action: '刷新' },
  export_busy: { line: '现在导出的人较多，请稍后再试。', action: '稍后再试' },
} as const;
export const isUploadLimitCode = (code: string): code is keyof typeof UPLOAD_LIMIT_COPY => Object.hasOwn(UPLOAD_LIMIT_COPY, code);
// Kept as the string keys C7a returned ('413' / '409'); errorKey() itself yields numbers.
const UPLOAD_LIMIT_STATUS = {
  quota_exceeded: '413', disk_low_water: '413', upload_completing: '409', attachment_in_use: '409', export_busy: '5xx',
} as unknown as Record<keyof typeof UPLOAD_LIMIT_COPY, ErrorCopyKey>;
const HAS_CHINESE = /[\u4e00-\u9fff]/;
/** The server line when it is Chinese product copy, else the fixed fallback for that code. */
export const uploadLimitLine = (code: keyof typeof UPLOAD_LIMIT_COPY, message: unknown) =>
  typeof message === 'string' && HAS_CHINESE.test(message) ? message.slice(0, 300) : UPLOAD_LIMIT_COPY[code].line;

// t238 C7a D2/U3: the training entry's input bound (hpc_runtime 256 MiB / 256 files) in user words.
export const TRAINING_INPUT_LIMIT_STATUS = '超过训练上限';
// t238 C7c-A3 #5: one phrase everywhere (「超过训练上限」); number and unit joined by a no-break space.
export const TRAINING_INPUT_LIMIT_LINE = '训练数据超过训练上限（256\u00a0MiB / 256\u00a0个文件）。先缩小或预处理，再继续。';
export const isTrainingInputLimit = (text: string) => /input directory exceeds the bounded (?:byte limit|file count)/.test(text);
export type ErrorCopy = { key: ErrorCopyKey; line: string; action: string; details: string[] };

function errorKey(status: number, code: string, cause: unknown): ErrorCopyKey {
  if (code === 'model_identity_unconfirmed') return code;
  if (code === 'network' || cause instanceof TypeError) return 'network';
  if (status === 401 || status === 403 || status === 404 || status === 409 || status === 413) return String(status) as ErrorCopyKey;
  if (status >= 500 && status <= 599) return '5xx';
  return 'unknown';
}

/** `(error | status | code | message) → { line, action, details }`. `details` is 技术详情: `HTTP {status} · {code}`, then the raw message. */
export function errorCopy(cause: unknown): ErrorCopy {
  const fields = (cause && typeof cause === 'object' ? cause : {}) as { status?: unknown; code?: unknown; message?: unknown };
  const status = Number(fields.status) || 0;
  const code = String(fields.code ?? '');
  const message = String(fields.message ?? (typeof cause === 'string' ? cause : ''));
  const head = [status ? `HTTP ${status}` : '', code].filter(Boolean).join(' · ');
  if (isUploadLimitCode(code)) return { key: UPLOAD_LIMIT_STATUS[code], line: uploadLimitLine(code, message), action: UPLOAD_LIMIT_COPY[code].action, details: [head] };
  const key = errorKey(status, code, cause);
  return { key, ...ERROR_COPY[key], details: [head, message].filter(Boolean) };
}
