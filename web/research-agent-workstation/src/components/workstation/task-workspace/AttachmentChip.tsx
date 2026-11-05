'use client';
import { useEffect, useState } from 'react';
import { FileText } from 'lucide-react';
import { actionName, formatFileSize, smallPreviewable, smallPreviewText } from './task-presentation';
import type { TaskArtifact } from './types';
import { DownloadFile } from './DownloadFile';
import { timeoutSignal } from './abort-signal';

const KIND: Record<string, string> = { md: 'Markdown', txt: '文本', csv: 'CSV', json: 'JSON', pdf: 'PDF', docx: 'Word', xlsx: 'Excel', pptx: 'PPT', zip: '压缩包', py: 'Python', html: '网页', png: '图片', jpg: '图片', jpeg: '图片', svg: '图片' };

/** 「Markdown · 3 KB」: the file type in Chinese short copy, then its size. */
export function fileKindLabel(file: { name: string; bytes: number }) {
  const ext = /\.([A-Za-z0-9]+)$/.exec(file.name)?.[1]?.toLowerCase() ?? '';
  return [KIND[ext] || (ext ? ext.toUpperCase() : '文件'), formatFileSize(file.bytes)].filter(Boolean).join(' · ');
}

/**
 * C6-answer SPEC §5.6 (H_a11): a file is an attachment under the answer, not the answer itself. Wash background, no border;
 * one-line name; 预览 / 下载 are real controls (≥44px on touch). S_a1: a small text file previews inline below it.
 */
export function AttachmentChip({ file, inlinePreview = false }: { file: TaskArtifact; inlinePreview?: boolean }) {
  const [preview, setPreview] = useState<string | null>(null);
  const [small, setSmall] = useState<string | null>(null);
  const [opened, setOpened] = useState(false);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const url = `/api/assistant/artifacts/${encodeURIComponent(file.id)}`;
  const smallFile = inlinePreview && smallPreviewable(file);
  useEffect(() => {
    if (!smallFile) return;
    const abort = new AbortController();
    fetch(url, { cache: 'no-store', signal: timeoutSignal(12000, abort.signal) })
      .then(response => response.ok ? response.text() : null)
      .then(text => { if (text !== null && !abort.signal.aborted) setSmall(smallPreviewText(text)); })
      .catch(() => undefined);
    return () => abort.abort();
  }, [smallFile, url]);
  async function open() {
    setOpened(!opened); setError('');
    if (opened || preview !== null || ['pdf', 'image', 'svg', 'html'].includes(file.preview_kind) || file.bytes > 131072) return;
    setBusy(true);
    try {
      const response = await fetch(url, { cache: 'no-store', signal: timeoutSignal(12000) });
      if (!response.ok) throw new Error(`预览失败（${response.status}），可重新打开重试。`);
      setPreview((await response.text()).slice(0, 131072));
    } catch (cause) { setError(cause instanceof Error ? cause.message : '预览失败，请重试。'); }
    finally { setBusy(false); }
  }
  return <article className="tw-filechip" data-testid="tw-filechip">
    <div className="tw-filechip-row">
      <span className="tw-filechip-icon" aria-hidden="true"><FileText size={18} /></span>
      <div className="tw-filechip-text"><span className="tw-filechip-name" title={file.name}>{file.name}</span><small>{fileKindLabel(file)}</small></div>
      <div className="tw-filechip-actions">
        <button type="button" className="tw-filechip-action is-preview" onClick={() => void open()} aria-expanded={opened}>{opened ? '收起' : '预览'}</button>
        <DownloadFile url={`${url}?download=1`} file={file} className="tw-filechip-action" />
      </div>
    </div>
    {/* C7c-FIX5i (F5-2): a structured (JSON) preview is technical detail: folded by default, never shown open in the chat */}
    {!opened && small && (/\.jsonl?$/i.test(file.name) || file.preview_kind === 'json'
      ? <details className="tw-details tw-filechip-tech" data-testid="tw-filechip-tech"><summary>技术详情</summary><pre className="tw-filechip-small" data-testid="tw-filechip-small">{small}</pre></details>
      : <pre className="tw-filechip-small" data-testid="tw-filechip-small">{small}</pre>)}
    {opened && <div className="tw-filechip-preview">{busy && <p role="status">正在读取文件…</p>}{error && <p role="alert" className="tw-error">{error}</p>}
      {['image', 'svg'].includes(file.preview_kind) ? <img className="tw-image" src={url} alt={file.name} /> : ['html', 'pdf'].includes(file.preview_kind) ? <iframe className="tw-document" title={file.name} src={url} sandbox="" /> : preview !== null ? <pre className="tw-file-preview">{preview}</pre> : !busy && !error ? <p className="tw-muted">此文件较大或为二进制格式，请下载后查看。</p> : null}</div>}
  </article>;
}

/** SPEC §5.6: one folded 「已通过校验 · 查看来源」 under the chips, default closed; the ThinkingRow's › chevron, not ▶. */
export function FileSources({ files }: { files: TaskArtifact[] }) {
  if (!files.length) return null;
  return <details className="tw-details tw-file-sources"><summary>已通过校验 · 查看来源<span className="tw-thinking-chevron" aria-hidden="true">›</span></summary>
    <dl>{files.map(file => <div key={file.id}><dt>{file.name}</dt><dd>{actionName(file.source_tool_call)}</dd></div>)}</dl>
  </details>;
}
