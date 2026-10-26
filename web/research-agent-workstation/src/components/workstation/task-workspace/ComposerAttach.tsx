'use client';
import { useEffect, useRef, useState } from 'react';
import { Paperclip, X } from 'lucide-react';
import { formatFileSize } from './task-presentation';
import { uploadUserFile } from './upload-attachment';
import type { UserFile } from './types';

/** Uploads composer files one by one; each finished file is handed to `onUploaded`. */
export function useComposerUpload(onUploaded: (file: UserFile) => Promise<void> | void) {
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => controller.current?.abort(), []);
  async function upload(files: File[]) {
    if (busy || !files.length) return;
    const abort = new AbortController(); controller.current = abort;
    setBusy(true); setError('');
    try {
      for (const file of files) {
        setStatus(`正在上传 ${file.name}`);
        const uploaded = await uploadUserFile(file, abort.signal, percent => setStatus(`正在上传 ${file.name} · ${percent}%`));
        if (abort.signal.aborted) return;
        await onUploaded(uploaded);
      }
      setStatus('');
    } catch (cause) { if (!abort.signal.aborted) { setStatus(''); setError(cause instanceof Error ? cause.message : '上传未完成，请重试。'); } }
    finally { if (!abort.signal.aborted) setBusy(false); }
  }
  return { upload, status, error, busy };
}

export function AttachButton({ disabled, onFiles }: { disabled?: boolean; onFiles: (files: File[]) => void }) {
  const input = useRef<HTMLInputElement>(null);
  return <>
    <button type="button" className="tw-attach" aria-label="添加资料" title="添加资料" disabled={disabled} onClick={() => input.current?.click()}><Paperclip size={18} aria-hidden="true" /></button>
    <input ref={input} type="file" multiple hidden tabIndex={-1} aria-hidden="true" onChange={event => { const files = Array.from(event.target.files ?? []); event.target.value = ''; onFiles(files); }} />
  </>;
}

export function AttachmentChips({ files, status, error, disabled, onRemove }: { files: UserFile[]; status?: string; error?: string; disabled?: boolean; onRemove: (id: string) => void }) {
  if (!files.length && !status && !error) return null;
  return <div className="tw-chips" aria-label="已添加的资料">
    {files.map(file => <span className="tw-chip" key={file.id}><Paperclip size={14} aria-hidden="true" /><span className="tw-chip-name">{file.name}</span><small>{formatFileSize(file.bytes)}</small>
      <button type="button" aria-label={`移除 ${file.name}`} disabled={disabled} onClick={() => onRemove(file.id)}><X size={14} aria-hidden="true" /></button></span>)}
    {status && <span className="tw-chip tw-chip-status" role="status">{status}</span>}
    {error && <span className="tw-chip tw-chip-error" role="alert">{error}</span>}
  </div>;
}
