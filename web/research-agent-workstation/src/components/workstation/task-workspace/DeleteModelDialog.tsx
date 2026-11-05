'use client';
import { useEffect, useRef, type KeyboardEvent } from 'react';
import { useReturnFocus } from './focus-return';

/** CP2 H2-4: exact strings. No wording about irreversibility. */
export const MODEL_DELETE_COPY = {
  title: (name: string) => `删除${name}？`, line: '删除后不能再选用，历史任务仍保留名称。', cancel: '取消', confirm: '删除',
} as const;

/** 取消 is focused first, Tab stays inside, Esc cancels; 删除 is the one danger action. */
export function DeleteModelDialog({ name, busy, onCancel, onConfirm }: { name: string; busy: boolean; onCancel: () => void; onConfirm: () => void }) {
  useReturnFocus();
  const cancel = useRef<HTMLButtonElement>(null);
  const confirm = useRef<HTMLButtonElement>(null);
  useEffect(() => { cancel.current?.focus(); }, []);
  function keys(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); onCancel(); }
    else if (event.key === 'Tab') {
      event.preventDefault();
      (document.activeElement === cancel.current ? confirm.current : cancel.current)?.focus();
    }
  }
  return <div className="tw-dialog-backdrop" onMouseDown={event => { if (event.target === event.currentTarget) { event.preventDefault(); onCancel(); } }}>
    <div className="tw-dialog" role="alertdialog" aria-modal="true" aria-labelledby="tw-delete-model-title" aria-describedby="tw-delete-model-line" data-testid="tw-model-delete-dialog" onKeyDown={keys}>
      <h2 id="tw-delete-model-title">{MODEL_DELETE_COPY.title(name)}</h2><p id="tw-delete-model-line">{MODEL_DELETE_COPY.line}</p>
      <div className="tw-dialog-actions"><button ref={cancel} type="button" className="tw-button" data-testid="tw-model-delete-cancel" onClick={onCancel}>{MODEL_DELETE_COPY.cancel}</button>
        <button ref={confirm} type="button" className="tw-button is-danger" data-testid="tw-model-delete-confirm" disabled={busy || undefined} onClick={onConfirm}>{MODEL_DELETE_COPY.confirm}</button></div>
    </div></div>;
}
