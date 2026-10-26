import { jsonPost, taskRequest, type UserFile } from './types';

export type UploadJob = { file: File; id: string; chunkSize: number };

async function chunkDigest(chunk: ArrayBuffer) {
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', chunk)), value => value.toString(16).padStart(2, '0')).join('');
}

/** Chunked upload to the user's library. Pass `job` to resume the same file after a failed chunk. */
export async function uploadUserFile(file: File, signal: AbortSignal, onProgress?: (percent: number) => void, job?: { current: UploadJob | null }): Promise<UserFile> {
  const pending = job ?? { current: null };
  if (pending.current?.file !== file) {
    const created = await taskRequest<{ id: string; chunk_size: number }>('/api/assistant/uploads', { ...jsonPost({ name: file.name, total_bytes: file.size, media_type: file.type }), signal });
    pending.current = { file, id: created.id, chunkSize: created.chunk_size };
  }
  const current = pending.current!;
  for (let offset = 0, index = 0; offset < file.size; offset += current.chunkSize, index++) {
    const chunk = await file.slice(offset, offset + current.chunkSize).arrayBuffer();
    const digest = await chunkDigest(chunk);
    onProgress?.(Math.round(offset / file.size * 100));
    await taskRequest(`/api/assistant/uploads/${current.id}/chunks/${index}`, { method: 'PUT', headers: { 'Content-Type': 'application/octet-stream', 'X-Chunk-SHA256': digest }, body: chunk, signal });
  }
  const result = await taskRequest<{ attachment: UserFile }>(`/api/assistant/uploads/${current.id}/complete`, { ...jsonPost({}), signal });
  return result.attachment;
}

/** Attachments chosen on the home composer, handed to the task created from it. */
export const NEXT_ATTACHMENTS_KEY = 'deepevo.next-attachments';

export function carryAttachments(taskId: string, ids: string[]) {
  if (ids.length) window.sessionStorage.setItem(NEXT_ATTACHMENTS_KEY, JSON.stringify({ task_id: taskId, attachment_ids: ids }));
}

/** Read once. Ids saved for another task are dropped, never bound here. */
export function takeCarriedAttachments(taskId: string): string[] {
  const raw = window.sessionStorage.getItem(NEXT_ATTACHMENTS_KEY);
  if (!raw) return [];
  try {
    const value = JSON.parse(raw) as { task_id?: string; attachment_ids?: unknown };
    if (value.task_id !== taskId) return [];
    window.sessionStorage.removeItem(NEXT_ATTACHMENTS_KEY);
    return Array.isArray(value.attachment_ids) ? value.attachment_ids.filter((id): id is string => typeof id === 'string' && Boolean(id)) : [];
  } catch { window.sessionStorage.removeItem(NEXT_ATTACHMENTS_KEY); return []; }
}
