'use client';
// t238 C7c-A3 #4: 「导出我的数据」 as an in-app action: 导出中… while pending, the busy / error line inline (role="alert").
import { useRef, useState } from 'react';
import { EXPORT_COPY, FILE_DELETE_COPY, TENANT_EXPORT_URL, exportErrorLine, exportFileName } from './file-delete';
import { timeoutSignal } from './abort-signal';

export function ExportData() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const running = useRef(false);
  async function run() {
    if (running.current) return;
    running.current = true; setBusy(true); setError('');
    try {
      const response = await fetch(TENANT_EXPORT_URL, { cache: 'no-store', signal: timeoutSignal(600000) });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({})) as { message?: string };
        setError(exportErrorLine(response.status, String(payload.message ?? '')));
        return;
      }
      const blob = await response.blob();
      const objectUrl = URL.createObjectURL(blob);
      const link = document.createElement('a'); link.href = objectUrl; link.download = exportFileName(response.headers.get('content-disposition')); link.click();
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 30000);
    } catch { setError(EXPORT_COPY.failed); }
    finally { running.current = false; setBusy(false); }
  }
  return <div className="tw-export">
    <button type="button" className="tw-button tw-nowrap tw-row-action" data-testid="tw-export" disabled={busy} aria-busy={busy} onClick={() => void run()}>
      {busy && <span className="tw-spinner" aria-hidden="true" />}{busy ? EXPORT_COPY.pending : FILE_DELETE_COPY.export}</button>
    {error && <p role="alert" className="tw-error tw-export-error">{error}</p>}
  </div>;
}
