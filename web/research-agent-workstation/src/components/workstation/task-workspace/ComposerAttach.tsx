'use client';
import { useEffect, useRef, useState } from 'react';
import { Paperclip, X } from 'lucide-react';
import { formatFileSize } from './task-presentation';
import { trainabilityNotice, untrainable, uploadUserFile } from './upload-attachment';
import { TaskRequestError, type UserFile } from './types';

/** t238 C7c-A3 #2: the composer quota-full error links to the data page in a new tab, so the draft and attachments stay. */
export const MANAGE_FILES_FOCUS = 'files';
export const MANAGE_FILES = { label: '管理文件', href: `/workspace?section=data&focus=${MANAGE_FILES_FOCUS}` } as const;

/** Uploads composer files one by one; each finished file is handed to `onUploaded`. */
export function useComposerUpload(onUploaded: (file: UserFile) => Promise<void> | void) {
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const [errorCode, setErrorCode] = useState('');
  const [busy, setBusy] = useState(false);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => controller.current?.abort(), []);
  async function upload(files: File[]) {
    if (busy || !files.length) return;
    const abort = new AbortController(); controller.current = abort;
    setBusy(true); setError(''); setErrorCode('');
    try {
      let advisory = '';
      for (const file of files) {
        setStatus(`正在上传 ${file.name}`);
        const uploaded = await uploadUserFile(file, abort.signal, percent => setStatus(`正在上传 ${file.name} · ${percent}%`));
        if (abort.signal.aborted) return;
        await onUploaded(uploaded);
        // t238 C7c-A3 #6: the chip above already names a single file; repeat it only when several were added.
        if (untrainable(uploaded)) advisory = files.length > 1 ? `${uploaded.name}：${trainabilityNotice(uploaded)}` : trainabilityNotice(uploaded);
      }
      setStatus(advisory);
    } catch (cause) {
      if (!abort.signal.aborted) {
        setStatus(''); setError(cause instanceof Error ? cause.message : '上传未完成，请重试。');
        setErrorCode(cause instanceof TaskRequestError ? cause.code : '');
      }
    }
    finally { if (!abort.signal.aborted) setBusy(false); }
  }
  return { upload, status, error, errorCode, busy };
}

export function AttachButton({ disabled, onFiles }: { disabled?: boolean; onFiles: (files: File[]) => void }) {
  const input = useRef<HTMLInputElement>(null);
  return <>
    <button type="button" className="tw-attach" aria-label="添加资料" title="添加资料" disabled={disabled} onClick={() => input.current?.click()}><Paperclip size={18} aria-hidden="true" /></button>
    <input ref={input} type="file" multiple hidden tabIndex={-1} aria-hidden="true" onChange={event => { const files = Array.from(event.target.files ?? []); event.target.value = ''; onFiles(files); }} />
  </>;
}

export function AttachmentChips({ files, status, error, errorCode, disabled, onRemove }: { files: UserFile[]; status?: string; error?: string; errorCode?: string; disabled?: boolean; onRemove: (id: string) => void }) {
  if (!files.length && !status && !error) return null;
  const quotaFull = errorCode === 'quota_exceeded';
  return <div className="tw-chips" aria-label="已添加的资料">
    {files.map(file => <span className="tw-chip" key={file.id}><Paperclip size={14} aria-hidden="true" /><span className="tw-chip-name">{file.name}</span><small>{formatFileSize(file.bytes)}</small>
      <button type="button" aria-label={`移除 ${file.name}`} disabled={disabled} onClick={() => onRemove(file.id)}><X size={14} aria-hidden="true" /></button></span>)}
    {status && <span className="tw-chip tw-chip-status" role="status">{status}</span>}
    {error && <span className="tw-chip tw-chip-error" role="alert">{error}{quotaFull && <a className="tw-chip-action tw-nowrap" href={MANAGE_FILES.href} target="_blank" rel="noopener">{MANAGE_FILES.label}</a>}</span>}
  </div>;
}
