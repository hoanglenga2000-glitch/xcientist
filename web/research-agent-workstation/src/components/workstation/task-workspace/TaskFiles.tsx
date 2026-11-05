'use client';
import { useEffect, useRef, useState } from 'react';
import { Upload } from 'lucide-react';
import { jsonRequest, taskRequest, type GeneratedFile, type UserFile } from './types';
import { trainabilityNotice, untrainable, uploadUserFile, type UploadJob } from './upload-attachment';
import { DownloadFile } from './DownloadFile';
import { ErrorNotice } from './ErrorNotice';
import { DeleteFileDialog } from './DeleteFileDialog';
import { MANAGE_FILES_FOCUS } from './ComposerAttach';
import { ExportData } from './ExportData';
import { FILE_DELETE_COPY, deleteDialogState, deleteResultLine, fileDeleteUrl, fileUsageUrl, type DeleteDialogState, type FileReference } from './file-delete';

const EMPTY_FILES: UserFile[] = [];
export function TaskFiles({ selected = EMPTY_FILES, onSelect }: { selected?: UserFile[]; onSelect?: (ids: string[]) => Promise<void> }) {
  const [files, setFiles] = useState<UserFile[]>([]);
  const [generated, setGenerated] = useState<GeneratedFile[]>([]);
  const [chosen, setChosen] = useState<string[]>(selected.map(item => item.id));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [notice, setNotice] = useState('');
  const pending = useRef<UploadJob | null>(null);
  const controller = useRef<AbortController | null>(null);
  const picker = useRef<HTMLInputElement>(null);
  const [removing, setRemoving] = useState<{ file: UserFile; state: DeleteDialogState } | null>(null);
  // t238 C7b COPY/S3: trainability hints live on the file row they describe, not above the next heading.
  const [hints, setHints] = useState<Record<string, string>>({});
  const errorBox = useRef<HTMLDivElement>(null);
  const fileList = useRef<HTMLDivElement>(null);
  useEffect(() => { if (error) errorBox.current?.scrollIntoView?.({ block: 'nearest' }); }, [error]);
  // The single action under an error repeats the operation that failed.
  const again = useRef<() => void>(() => refresh());
  function refresh() {
    const signal = controller.current!.signal; setError(null); again.current = refresh;
    void taskRequest<{ files: UserFile[]; generated?: GeneratedFile[] }>('/api/assistant/files', { signal }).then(data => { if (!signal.aborted) { setFiles(data.files); setGenerated(data.generated ?? []); } }).catch(cause => { if (!signal.aborted) setError(cause); });
  }
  useEffect(() => {
    const abort = new AbortController(); controller.current = abort;
    refresh();
    return () => abort.abort();
  }, []);
  useEffect(() => { setChosen(selected.map(item => item.id)); }, [selected]);

  async function upload(file: File) {
    if (busy) return;
    setBusy(true); setError(null); again.current = () => void upload(file);
    const signal = controller.current!.signal;
    try {
      const attachment = await uploadUserFile(file, signal, percent => setNotice(`正在上传 ${file.name} · ${percent}%`), pending);
      const list = await taskRequest<{ files: UserFile[]; generated?: GeneratedFile[] }>('/api/assistant/files', { signal });
      if (signal.aborted) return;
      setFiles(list.files); setGenerated(list.generated ?? []);
      if (onSelect) {
        const ids = [...new Set([...chosen, attachment.id])];
        await onSelect(ids); setChosen(ids);
      }
      pending.current = null;
      const saved = onSelect ? '资料已校验并保存到当前任务，没有开始执行。' : '已保存到你的资料库，可在任务中选用。';
      if (untrainable(attachment)) setHints(current => ({ ...current, [attachment.id]: trainabilityNotice(attachment) }));
      setNotice(saved);
    } catch (cause) { if (!signal.aborted) { setNotice(''); setError(cause); } }  // S3: the stale 「正在上传…%」 line goes
    finally { if (!signal.aborted) setBusy(false); }
  }
  // t238 C7b D6: ask the runtime which tasks use the file first; an active run is refused before any delete.
  async function askDelete(file: UserFile) {
    if (busy) return;
    setBusy(true); setError(null); again.current = () => void askDelete(file);
    try {
      const usage = await taskRequest<{ referenced_by: FileReference[] }>(fileUsageUrl(file.id), { signal: controller.current!.signal });
      setRemoving({ file, state: deleteDialogState(file.name, usage.referenced_by ?? []) });
    } catch (cause) { if (!controller.current!.signal.aborted) setError(cause); }
    finally { setBusy(false); }
  }
  async function confirmDelete() {
    if (!removing || busy) return;
    const { file } = removing;
    setBusy(true); setError(null); again.current = refresh;
    try {
      const result = await taskRequest<{ status?: string; message?: string }>(fileDeleteUrl(file.id), jsonRequest('DELETE'));
      setRemoving(null); setFiles(current => current.filter(item => item.id !== file.id));
      setChosen(current => current.filter(id => id !== file.id));
      setNotice(deleteResultLine(file.name, result)); refresh();
    } catch (cause) { setRemoving(null); setError(cause); }
    finally { setBusy(false); }
  }
  // 「管理文件」 (quota 413): bring the file list into view so a file can be deleted.
  function focusFileList() {
    // t238 C7c-A3 #2: no animation under prefers-reduced-motion.
    const reduce = typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    fileList.current?.scrollIntoView?.({ block: 'start', behavior: reduce ? 'auto' : 'smooth' });
    fileList.current?.focus({ preventScroll: true });
  }
  // 「管理文件」 links land here with ?focus=files: same as the page's own button, once the list is rendered.
  const landed = useRef(false);
  useEffect(() => {
    if (onSelect || landed.current || typeof window === 'undefined') return;
    if (new URLSearchParams(window.location.search).get('focus') !== MANAGE_FILES_FOCUS) return;
    landed.current = true;
    focusFileList();
  }, [files.length]); // eslint-disable-line react-hooks/exhaustive-deps
  async function saveSelection() {
    if (!onSelect || busy) return;
    setBusy(true); setError(null); again.current = () => void saveSelection();
    try { await onSelect(chosen); setNotice('本任务的资料选择已保存；已有运行的输入不变。'); }
    catch (cause) { setError(cause); }
    finally { setBusy(false); }
  }
  return <section aria-label={onSelect ? '本任务资料' : '我的资料'}><h2>{onSelect ? '本任务资料' : '我的资料'}</h2>
    {/* untrainable(attachment) → trainabilityNotice(attachment): shown on that file's row below */}
    <label className={`tw-upload${busy ? ' is-busy' : ''}`}><Upload size={16} aria-hidden="true" />上传资料<input ref={picker} className="tw-sr-only" type="file" aria-label="上传资料" disabled={busy} onChange={event => { const file = event.target.files?.[0]; if (file) void upload(file); event.target.value = ''; }} /></label>
    <div ref={errorBox} className="tw-upload-feedback"><ErrorNotice error={error} onAction={action => action === '重新选择文件' ? picker.current?.click()
      : action === '管理文件' ? focusFileList() : action === '稍后再试' ? setError(null) : action === '刷新' ? refresh() : again.current()} />
      <p role="status">{notice}</p></div>
    <div ref={fileList} tabIndex={-1} className="tw-file-list" aria-label={onSelect ? '本任务资料列表' : '资料列表'}>
    {!onSelect && files.length > 0 && <ExportData />}
    {!files.length && <p className="tw-muted tw-empty-files">还没有资料。</p>}
    {files.map(file => <article className="tw-file" key={file.id}><div className="tw-file-heading">
      {onSelect && <input type="checkbox" aria-label={`用于本任务：${file.name}`} checked={chosen.includes(file.id)} disabled={busy} onChange={event => setChosen(current => event.target.checked ? [...current, file.id] : current.filter(id => id !== file.id))} />}
      <div><h3>{file.name}</h3><small>{file.bytes.toLocaleString()} 字节</small>{hints[file.id] && <p className="tw-file-hint">{hints[file.id]}</p>}</div><div className="tw-file-actions"><DownloadFile url={`/api/assistant/files/${file.id}`} file={file} label="下载资料" className="tw-button tw-nowrap tw-row-action" />
        {!onSelect && <button type="button" className="tw-button tw-text-danger tw-nowrap tw-row-action" disabled={busy} aria-label={`${FILE_DELETE_COPY.action}：${file.name}`} onClick={() => void askDelete(file)}>{FILE_DELETE_COPY.action}</button>}</div></div>
      <details className="tw-details"><summary>上传校验</summary><p>SHA-256：{file.sha256}</p><p>校验只证明文件内容一致，不代表科研结论成立。</p></details>
    </article>)}
    </div>
    {onSelect && files.length > 0 && <button disabled={busy} onClick={() => void saveSelection()}>保存资料选择</button>}
    {!onSelect && generated.length > 0 && <h2>生成文件</h2>}
    {!onSelect && generated.map(file => <article className="tw-file" key={file.id}><div className="tw-file-heading"><div><h3>{file.name}</h3><small>{file.bytes.toLocaleString()} 字节</small></div><DownloadFile url={file.download_url || `/api/assistant/artifacts/${encodeURIComponent(file.id)}?download=1`} file={file} label="下载生成文件" /></div></article>)}
    {removing && <DeleteFileDialog state={removing.state} busy={busy} onCancel={() => setRemoving(null)} onConfirm={() => void confirmDelete()} />}
  </section>;
}
