'use client';
import { useEffect, useRef, useState } from 'react';
import { verifiedDownloadBytes } from './verified-download';

export function DownloadFile({ url, file, label = '下载' }: { url: string; file: { name: string; bytes: number; sha256: string }; label?: string }) {
  const [busy, setBusy] = useState(false); const [error, setError] = useState('');
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => { controller.current?.abort(); }, [url]);
  async function download() {
    if (controller.current) return;
    const abort = new AbortController(); controller.current = abort; setBusy(true); setError('');
    try {
      const response = await fetch(url, { cache: 'no-store', signal: AbortSignal.any([abort.signal, AbortSignal.timeout(60000)]) });
      const bytes = await verifiedDownloadBytes(response, file);
      if (abort.signal.aborted) return;
      const objectUrl = URL.createObjectURL(new Blob([bytes], { type: 'application/octet-stream' }));
      const link = document.createElement('a'); link.href = objectUrl; link.download = file.name; link.click();
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 30000);
    } catch (cause) { if (!abort.signal.aborted) setError(cause instanceof Error ? cause.message : '下载未完成，请重试。'); }
    finally { controller.current = null; if (!abort.signal.aborted) setBusy(false); }
  }
  return <div><a className="tw-button" href={url} aria-disabled={busy} onClick={event => { event.preventDefault(); void download(); }}>{busy ? '正在校验下载…' : label}</a>{error && <p role="alert" className="tw-error">{error}</p>}</div>;
}
