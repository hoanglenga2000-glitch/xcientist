'use client';
import { useEffect, useRef, type KeyboardEvent } from 'react';
import { FILE_DELETE_COPY, type DeleteDialogState } from './file-delete';
import { useReturnFocus } from './focus-return';

/** t238 C7b D6: 取消 is focused first, Tab stays inside, Esc cancels; the destructive button uses the danger token. */
export function DeleteFileDialog({ state, busy, onCancel, onConfirm }: { state: DeleteDialogState; busy: boolean; onCancel: () => void; onConfirm: () => void }) {
  useReturnFocus();
  const cancel = useRef<HTMLButtonElement>(null);
  const confirm = useRef<HTMLButtonElement>(null);
  useEffect(() => { cancel.current?.focus(); }, []);
  function keys(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); onCancel(); }
    else if (event.key === 'Tab') {
      event.preventDefault();
      (document.activeElement === cancel.current && confirm.current ? confirm.current : cancel.current)?.focus();
    }
  }
  return <div className="tw-dialog-backdrop" onMouseDown={event => { if (event.target === event.currentTarget) { event.preventDefault(); onCancel(); } }}>
    <div className={`tw-dialog${state.kind === 'refused' ? ' is-refusal' : ''}`} role="alertdialog" aria-modal="true" aria-labelledby="tw-delete-file-title" aria-describedby="tw-delete-file-body" onKeyDown={keys}>
      {/* t238 C7c-A3 #3: refusal title reads exactly 「现在不能删除「xxx」」; the quoted name sits on its own centred line. */}
      <h2 id="tw-delete-file-title">{state.kind === 'refused' ? <>现在不能删除<span className="tw-dialog-title-name">「{state.name}」</span></> : state.title}</h2>
      <div id="tw-delete-file-body" className="tw-dialog-body">{state.kind !== 'refused' && <p className="tw-dialog-name">{state.name}</p>}{state.line && <p id="tw-delete-file-line">{state.line}</p>}</div>
      <div className="tw-dialog-actions">
        <button ref={cancel} type="button" className="tw-button tw-nowrap" onClick={onCancel}>{state.kind === 'refused' ? FILE_DELETE_COPY.close : FILE_DELETE_COPY.cancel}</button>
        {state.kind !== 'refused' && <button ref={confirm} type="button" className="tw-button tw-danger-solid tw-nowrap" disabled={busy} onClick={onConfirm}>{state.confirm}</button>}
      </div>
    </div></div>;
}
