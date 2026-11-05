// t238 C7b D6: per-file delete on the 资料 page. Pure copy + decision helpers (node-tested); the dialog renders them.
export type FileReference = { task_id: string; task_title: string; run_id: string; active: boolean };

export const FILE_DELETE_COPY = {
  action: '删除',
  // t238 C7c-7b: bold title alone; the file name and the consequence sit in the body (normal weight).
  title: '删除这个文件？',
  // t238 C7c-A3 #3: the running-task refusal never says 「删除…不能恢复」; its title names the file.
  refusedTitle: (name: string) => `现在不能删除「${name}」`,
  irreversible: '删除后不能恢复。',
  inUse: (task: string) => `这个文件正在被任务「${task}」使用。删除后，这个任务就读不到它了。`,
  activeRefusal: (task: string) => `任务「${task}」正在运行，结束后才能删除这个文件。`,
  cancel: '取消',
  confirm: '删除',
  confirmAnyway: '仍然删除',
  close: '知道了',
  deleted: (name: string) => `已删除「${name}」。`,
  export: '导出我的数据',
  exporting: '正在导出…',
} as const;

/** The line after a delete: when a training copy is still being cleaned up the server's sentence wins over 「已删除」. */
export function deleteResultLine(name: string, result?: { status?: string; message?: string } | null): string {
  return result?.status === 'partial' && result.message ? result.message : FILE_DELETE_COPY.deleted(name);
}

export type DeleteDialogState =
  | { kind: 'plain'; title: string; name: string; line: string; confirm: string }
  | { kind: 'referenced'; title: string; name: string; line: string; confirm: string }
  | { kind: 'refused'; title: string; name: string; line: string };

/** What the confirm dialog shows for a file with these references (active run → refusal, no delete button). */
export function deleteDialogState(name: string, references: FileReference[]): DeleteDialogState {
  const title = FILE_DELETE_COPY.title;
  const active = references.find(item => item.active);
  if (active) return { kind: 'refused', title: FILE_DELETE_COPY.refusedTitle(name), name, line: FILE_DELETE_COPY.activeRefusal(active.task_title) };
  if (references.length) {
    const names = [...new Set(references.map(item => item.task_title))].join('」「');
    return { kind: 'referenced', title, name, line: `${FILE_DELETE_COPY.inUse(names)}${FILE_DELETE_COPY.irreversible}`, confirm: FILE_DELETE_COPY.confirmAnyway };
  }
  return { kind: 'plain', title, name, line: FILE_DELETE_COPY.irreversible, confirm: FILE_DELETE_COPY.confirm };
}

export const fileUsageUrl = (id: string) => `/api/assistant/files/${encodeURIComponent(id)}/usage`;
export const fileDeleteUrl = (id: string) => `/api/assistant/files/${encodeURIComponent(id)}`;
export const TENANT_EXPORT_URL = '/api/assistant/tenant/export';

// t238 C7c-A3 #4 (VERIFY 1 / Critiquito A): export is a fetch → blob → save action, so busy / auth / server errors
// show an in-app Chinese line instead of a browser error page.
export const EXPORT_COPY = {
  pending: '导出中…',
  busy: '现在导出的人较多，请稍后再试。',
  auth: '登录已过期，请重新登录后再导出。',
  forbidden: '你没有权限导出这些资料。',
  failed: '导出没有完成，请稍后再试。',
} as const;

/** The in-app line for a failed export response (server Chinese copy wins for 429). */
export function exportErrorLine(status: number, serverMessage = ''): string {
  if (status === 429) return /[\u4e00-\u9fff]/.test(serverMessage) ? serverMessage.slice(0, 120) : EXPORT_COPY.busy;
  if (status === 401) return EXPORT_COPY.auth;
  if (status === 403) return EXPORT_COPY.forbidden;
  return EXPORT_COPY.failed;
}

/** File name from Content-Disposition (RFC 5987 filename* first), else a dated fallback. */
export function exportFileName(disposition: string | null, today = new Date()): string {
  const header = disposition ?? '';
  const star = header.match(/filename\*\s*=\s*UTF-8''([^;]+)/i);
  if (star) { try { return safeName(decodeURIComponent(star[1].trim())); } catch { /* fall through */ } }
  const plain = header.match(/filename\s*=\s*"([^"]+)"|filename\s*=\s*([^;]+)/i);
  if (plain) return safeName((plain[1] ?? plain[2]).trim());
  const stamp = today.toISOString().slice(0, 10).replaceAll('-', '');
  return `evomind-export-${stamp}.zip`;
}

function safeName(name: string): string {
  const base = [...name.replace(/^.*[\\/]/, '')].filter(char => char.charCodeAt(0) >= 0x20).join('').trim();
  return base || 'evomind-export.zip';
}
