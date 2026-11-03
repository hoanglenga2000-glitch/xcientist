'use client';
import { memo, useEffect, useRef, useState, type ReactNode } from 'react';
import Markdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeSanitize from 'rehype-sanitize';
import { Check, Copy } from 'lucide-react';

type HastNode = { type: string; value?: string; properties?: { className?: unknown }; children?: HastNode[] };

function hastText(node?: HastNode): string {
  if (!node) return '';
  if (node.type === 'text') return node.value ?? '';
  return (node.children ?? []).map(hastText).join('');
}

function codeLanguage(pre?: HastNode) {
  const classes = pre?.children?.find(child => child.type === 'element')?.properties?.className;
  const match = (Array.isArray(classes) ? classes : []).map(String).find(name => name.startsWith('language-'));
  return match ? match.slice('language-'.length) : '';
}

function CodeBlock({ code, language, children }: { code: string; language: string; children: ReactNode }) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(timer.current), []);
  async function copy() {
    try {
      await navigator.clipboard.writeText(code.replace(/\n$/, ''));
      setCopied(true);
      window.clearTimeout(timer.current);
      timer.current = window.setTimeout(() => setCopied(false), 1500);
    } catch { setCopied(false); }
  }
  return <div className="tw-md-code">
    <div className="tw-md-code-head"><span>{language || '代码'}</span><button type="button" className="tw-md-copy" aria-label="复制代码" title={copied ? '已复制' : '复制代码'} onClick={() => void copy()}>
      {copied ? <Check size={14} aria-hidden="true" /> : <Copy size={14} aria-hidden="true" />}<span>{copied ? '已复制' : '复制'}</span>
    </button></div>
    <pre>{children}</pre>
  </div>;
}

// Raw HTML is never parsed (no rehype-raw); rehype-sanitize also strips unsafe URLs and attributes.
const COMPONENTS: Components = {
  a: ({ node: _node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer" />,
  pre: ({ node, children }) => <CodeBlock code={hastText(node as HastNode)} language={codeLanguage(node as HastNode)}>{children}</CodeBlock>,
  table: ({ node: _node, ...props }) => <div className="tw-md-table"><table {...props} /></div>,
};

/** Assistant reply as safe GitHub-flavoured Markdown. Memoised so streaming only re-parses on new text.
 * C6e tip2: when `answer`, put tw-msg-answer + data-testid=tw-answer on this .tw-md node only (not the article),
 * so VERIFY JS_ANSWER_TEXT does not scoop AttachmentChip 「下载」 from .tw-attachments. */
export const MarkdownMessage = memo(function MarkdownMessage({ text, answer }: { text: string; answer?: boolean }) {
  return <div className={answer ? 'tw-md tw-msg-answer' : 'tw-md'} data-testid={answer ? 'tw-answer' : undefined}><Markdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeSanitize]} components={COMPONENTS}>{text}</Markdown></div>;
});
