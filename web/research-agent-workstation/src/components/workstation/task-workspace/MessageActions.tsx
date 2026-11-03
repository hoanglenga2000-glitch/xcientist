'use client';
import { useEffect, useRef, useState } from 'react';
import { Check, Copy } from 'lucide-react';

/** L2: copy-only action bar under a completed assistant reply. Copy alone; no regenerate, feedback, or more menu (no backend). */
export function MessageActions({ text, latest = false }: { text: string; latest?: boolean }) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(timer.current), []);
  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.clearTimeout(timer.current);
      timer.current = window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  }
  return (
    <div className={`tw-msg-actions${latest ? ' is-latest' : ''}`}>
      <button
        type="button"
        className="tw-msg-action"
        aria-label={copied ? '已复制' : '复制'}
        title={copied ? '已复制' : '复制'}
        onClick={() => void copy()}
      >
        {copied ? <Check size={16} aria-hidden="true" /> : <Copy size={16} aria-hidden="true" />}
      </button>
    </div>
  );
}
