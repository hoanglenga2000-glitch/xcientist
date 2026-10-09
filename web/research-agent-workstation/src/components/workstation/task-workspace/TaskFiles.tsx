'use client';
import { useEffect, useRef, useState } from 'react';
import { jsonPost, taskRequest, type UserFile } from './types';
import { DownloadFile } from './DownloadFile';

const EMPTY_FILES: UserFile[] = [];
export function TaskFiles({ selected = EMPTY_FILES, onSelect }: { selected?: UserFile[]; onSelect?: (ids: string[]) => Promise<void> }) {
  const [files, setFiles] = useState<UserFile[]>([]);
  const [chosen, setChosen] = useState<string[]>(selected.map(item => item.id));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [retry, setRetry] = useState<File | null>(null);
  const pending = useRef<{ file: File; id: string; chunkSize: number } | null>(null);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => {
    const abort = new AbortController(); controller.current = abort;
    void taskRequest<{ files: UserFile[] }>('/api/assistant/files', { signal: abort.signal }).then(data => { if (!abort.signal.aborted) setFiles(data.files); }).catch(cause => { if (!abort.signal.aborted) setError(cause.message); });
    return () => abort.abort();
  }, []);
  useEffect(() => { setChosen(selected.map(item => item.id)); }, [selected]);

  async function upload(file: File) {
    if (busy) return;
    setBusy(true); setError(''); setRetry(file);
    const signal = controller.current!.signal;
    try {
      if (pending.current?.file !== file) {
        const created = await taskRequest<{ id: string; chunk_size: number }>('/api/assistant/uploads', { ...jsonPost({ name: file.name, total_bytes: file.size, media_type: file.type }), signal });
        pending.current = { file, id: created.id, chunkSize: created.chunk_size };
      }
      const job = pending.current!;
      for (let offset = 0, index = 0; offset < file.size; offset += job.chunkSize, index++) {
        const chunk = await file.slice(offset, offset + job.chunkSize).arrayBuffer();
        const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', chunk)), value => value.toString(16).padStart(2, '0')).join('');
        setNotice(`正在上传 ${file.name} · ${Math.round(offset / file.size * 100)}%`);
        await taskRequest(`/api/assistant/uploads/${job.id}/chunks/${index}`, { method: 'PUT', headers: { 'Content-Type': 'application/octet-stream', 'X-Chunk-SHA256': digest }, body: chunk, signal });
      }
      const result = await taskRequest<{ attachment: UserFile }>(`/api/assistant/uploads/${job.id}/complete`, { ...jsonPost({}), signal });
      const list = await taskRequest<{ files: UserFile[] }>('/api/assistant/files', { signal });
      if (signal.aborted) return;
      setFiles(list.files);
      if (onSelect) {
        const ids = [...new Set([...chosen, result.attachment.id])];
        await onSelect(ids); setChosen(ids);
      }
      pending.current = null; setRetry(null);
      setNotice(onSelect ? '资料已校验并保存到当前任务，没有开始执行。' : '已保存到你的资料库，可在任务中选用。');
    } catch (cause) { if (!signal.aborted) setError(cause instanceof Error ? cause.message : '上传未完成，请重试。已传分片不会重复写入。'); }
    finally { if (!signal.aborted) setBusy(false); }
  }
  async function saveSelection() {
    if (!onSelect || busy) return;
    setBusy(true); setError('');
    try { await onSelect(chosen); setNotice('本任务的资料选择已保存；已有运行的输入不变。'); }
    catch (cause) { setError(cause instanceof Error ? cause.message : '保存失败，选择已保留。'); }
    finally { setBusy(false); }
  }
  return <section aria-label={onSelect ? '本任务资料' : '我的资料'}><h2>{onSelect ? '本任务资料' : '我的资料'}</h2>
    <p className="tw-muted">{onSelect ? '选择自己的资料用于本任务。其他任务不会自动使用这些资料。' : '仅显示当前账户上传的资料。上传不会调用模型或启动任务。'}</p>
    <label className="tw-upload">上传资料<input type="file" aria-label="上传资料" disabled={busy} onChange={event => { const file = event.target.files?.[0]; if (file) void upload(file); event.target.value = ''; }} /></label>
    {!files.length && <p className="tw-muted">还没有资料，可以先上传文件。</p>}
    {files.map(file => <article className="tw-file" key={file.id}><div className="tw-file-heading">
      {onSelect && <input type="checkbox" aria-label={`用于本任务：${file.name}`} checked={chosen.includes(file.id)} disabled={busy} onChange={event => setChosen(current => event.target.checked ? [...current, file.id] : current.filter(id => id !== file.id))} />}
      <div><h3>{file.name}</h3><small>{file.bytes.toLocaleString()} 字节</small></div><DownloadFile url={`/api/assistant/files/${file.id}`} file={file} label="下载资料" /></div>
      <details className="tw-details"><summary>上传校验</summary><p>SHA-256：{file.sha256}</p><p>校验只证明文件内容一致，不代表科研结论成立。</p></details>
    </article>)}
    {onSelect && files.length > 0 && <button disabled={busy} onClick={() => void saveSelection()}>保存资料选择</button>}
    {error && <p role="alert" className="tw-error">{error}{retry && <button disabled={busy} onClick={() => void upload(retry)}>重试上传</button>}</p>}
    <p role="status">{notice}</p>
  </section>;
}
