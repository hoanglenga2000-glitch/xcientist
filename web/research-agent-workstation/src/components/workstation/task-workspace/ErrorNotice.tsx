'use client';
import { errorCopy } from './error-copy';

/** One Chinese line, one action, raw status/code/message only inside a closed 技术详情 (t237 C2a). */
export function ErrorNotice({ error, onAction }: { error: unknown; onAction?: (action: string) => void }) {
  if (!error) return null;
  const copy = errorCopy(error);
  const action = copy.action === '重新登录' ? <a className="tw-button" href="/login">{copy.action}</a>
    : copy.action === '返回' ? <a className="tw-button" href="/workspace">{copy.action}</a>
    : <button type="button" onClick={() => onAction ? onAction(copy.action) : window.location.reload()}>{copy.action}</button>;
  return <div role="alert" className="tw-error"><p>{copy.line}</p>{action}
    {copy.details.length > 0 && <details className="tw-details"><summary>技术详情</summary>{copy.details.map((line, index) => <p key={index}>{line}</p>)}</details>}
  </div>;
}
