"use client";

import Image from "next/image";
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  ArrowUp,
  ChevronDown,
  Clipboard,
  Plus,
  RotateCcw,
  Square,
  UserRound,
} from "lucide-react";
import { cn } from "@/lib/utils";

type Locale = "zh-CN" | "en-US";
type MessageStatus = "complete" | "streaming" | "stopped" | "error";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  status: MessageStatus;
};

type ActivityItem = {
  seq: number;
  type: string;
  status: string;
  label: string;
  detail?: string;
};

type ContextState = {
  currentTask: boolean;
  taskLabel: string;
  runStatus: string;
  memoryAvailable: boolean;
  toolsAvailable: boolean;
};

type StreamEvent = {
  type?: string;
  seq?: number;
  route?: string;
  status?: string;
  label?: string;
  detail?: string;
  delta?: string;
  answer?: string;
  message?: string;
  current_task?: boolean;
  task_label?: string;
  run_status?: string;
  memory_available?: boolean;
  tools_available?: boolean;
};

const STORAGE_KEY = "evomind_assistant_session_v1";
const LOCAL_CSRF_STORAGE_KEY = "evomind.local.csrf.v1";

function tx(locale: Locale, zh: string, en: string) {
  return locale === "zh-CN" ? zh : en;
}

function makeId(prefix: string) {
  const random = typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID().replaceAll("-", "").slice(0, 14)
    : `${Date.now()}_${Math.random().toString(16).slice(2, 8)}`;
  return `${prefix}_${random}`;
}

async function assistantStreamErrorMessage(response: Response, locale: Locale) {
  const fallback = tx(locale, "助手服务未返回流式响应。", "The assistant did not return a stream.");
  let code = "";
  try {
    const payload = (await response.clone().json()) as { code?: unknown; error?: unknown; message?: unknown };
    code = String(payload.code ?? payload.error ?? payload.message ?? "");
  } catch {
    code = "";
  }
  if (response.status === 401 || code === "session_required") {
    window.sessionStorage.removeItem(LOCAL_CSRF_STORAGE_KEY);
    return tx(
      locale,
      "本地会话已失效：通常是 EvoMind 重启后浏览器还停留在旧页面。请用新的启动链接重新打开工作站，然后再发同一句话；不会启动训练或提交 Kaggle。",
      "The local session expired, usually because EvoMind restarted while this tab stayed open. Reopen the workstation with the latest launch URL, then send the same message again.",
    );
  }
  if (response.status === 403 || code === "csrf_rejected" || code === "origin_rejected") {
    window.sessionStorage.removeItem(LOCAL_CSRF_STORAGE_KEY);
    return tx(
      locale,
      "本地安全令牌已过期：请刷新或用新的启动链接重新打开工作站，再重试这一句。",
      "The local security token expired. Refresh or reopen the workstation with the latest launch URL, then retry.",
    );
  }
  if (response.status === 411 || response.status === 413 || response.status === 415) {
    return tx(
      locale,
      "助手请求格式被安全门拒绝（" + (code || String(response.status)) + "）。",
      "Assistant request rejected by the safety gate (" + (code || String(response.status)) + ").",
    );
  }
  return code ? fallback + " (" + code + ")" : fallback;
}

function assistantLinkTarget(rawTarget: string) {
  const target = rawTarget.trim().replace(/^<|>$/g, "");
  if (target.startsWith("/api/")) return { href: target, external: false };
  if (/^https:\/\//i.test(target)) return { href: target, external: true };
  return null;
}

function renderInlineMarkdown(text: string, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  const token = /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\((?:<[^>]+>|[^)]+)\))/g;
  let cursor = 0;
  let match: RegExpExecArray | null;
  while ((match = token.exec(text)) !== null) {
    if (match.index > cursor) nodes.push(text.slice(cursor, match.index));
    const value = match[0];
    const key = `${keyPrefix}-${match.index}`;
    if (value.startsWith("**")) {
      nodes.push(<strong key={key} className="font-semibold text-ink">{value.slice(2, -2)}</strong>);
    } else if (value.startsWith("`")) {
      nodes.push(<code key={key} className="rounded bg-surface-sunken px-1 py-0.5 font-mono text-[0.9em] text-accent">{value.slice(1, -1)}</code>);
    } else {
      const link = value.match(/^\[([^\]]+)\]\((.+)\)$/);
      const label = link?.[1] ?? value;
      const target = link?.[2] ?? "";
      const safeTarget = assistantLinkTarget(target);
      nodes.push(safeTarget ? (
        <a
          key={key}
          href={safeTarget.href}
          target={safeTarget.external ? "_blank" : undefined}
          rel={safeTarget.external ? "noreferrer noopener" : undefined}
          className="font-medium text-accent underline decoration-accent/35 underline-offset-4 hover:decoration-accent"
        >{label}</a>
      ) : (
        <span key={key}>{label}{target ? <code className="ml-1 break-all rounded bg-surface-sunken px-1 py-0.5 font-mono text-[0.82em] text-ink-secondary">{target.replace(/^<|>$/g, "")}</code> : null}</span>
      ));
    }
    cursor = match.index + value.length;
  }
  if (cursor < text.length) nodes.push(text.slice(cursor));
  return nodes;
}

function markdownTableCells(line: string) {
  return line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
}

function isMarkdownTableDivider(line: string) {
  const cells = markdownTableCells(line);
  return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function MarkdownText({ text, keyPrefix }: { text: string; keyPrefix: string }) {
  const lines = text.split(/\r?\n/);
  const nodes: ReactNode[] = [];
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const key = `${keyPrefix}-${index}`;
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      const body = renderInlineMarkdown(heading[2], `${key}-heading`);
      if (heading[1].length === 1) nodes.push(<h2 key={key} className="mt-5 text-xl font-semibold tracking-tight text-ink first:mt-0">{body}</h2>);
      else if (heading[1].length === 2) nodes.push(<h3 key={key} className="mt-5 border-b border-edge pb-2 text-base font-semibold text-ink first:mt-0">{body}</h3>);
      else nodes.push(<h4 key={key} className="mt-4 text-sm font-semibold text-ink first:mt-0">{body}</h4>);
      continue;
    }
    if (line.includes("|") && index + 1 < lines.length && isMarkdownTableDivider(lines[index + 1])) {
      const headers = markdownTableCells(line);
      const rows: string[][] = [];
      index += 2;
      while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
        rows.push(markdownTableCells(lines[index]));
        index += 1;
      }
      index -= 1;
      nodes.push(
        <div key={key} className="thin-scrollbar my-3 overflow-x-auto rounded-md border border-edge">
          <table className="w-full min-w-[520px] border-collapse text-left text-xs">
            <thead className="bg-surface-sunken text-ink-secondary"><tr>{headers.map((cell, cellIndex) => <th key={`${key}-h-${cellIndex}`} className="border-b border-edge px-3 py-2 font-semibold">{renderInlineMarkdown(cell, `${key}-h-${cellIndex}`)}</th>)}</tr></thead>
            <tbody>{rows.map((row, rowIndex) => <tr key={`${key}-r-${rowIndex}`} className="border-b border-edge/70 last:border-b-0">{headers.map((_, cellIndex) => <td key={`${key}-r-${rowIndex}-${cellIndex}`} className="px-3 py-2 align-top text-ink-secondary">{renderInlineMarkdown(row[cellIndex] ?? "", `${key}-r-${rowIndex}-${cellIndex}`)}</td>)}</tr>)}</tbody>
          </table>
        </div>
      );
      continue;
    }
    if (!line.trim()) {
      continue;
    }
    if (/^---+$/.test(line.trim())) {
      nodes.push(<hr key={key} className="my-4 border-edge" />);
      continue;
    }
    const bullet = line.match(/^\s*[-*]\s+(.+)$/);
    if (bullet) {
      nodes.push(<div key={key} className="flex gap-2 pl-1 text-ink"><span className="mt-[0.68em] h-1.5 w-1.5 shrink-0 rounded-full bg-accent" /><p>{renderInlineMarkdown(bullet[1], `${key}-bullet`)}</p></div>);
      continue;
    }
    const numbered = line.match(/^\s*(\d+)\.\s+(.+)$/);
    if (numbered) {
      nodes.push(<div key={key} className="grid grid-cols-[1.5rem_1fr] gap-1 text-ink"><span className="font-medium text-accent">{numbered[1]}.</span><p>{renderInlineMarkdown(numbered[2], `${key}-number`)}</p></div>);
      continue;
    }
    const quote = line.match(/^>\s?(.*)$/);
    if (quote) {
      nodes.push(<blockquote key={key} className="border-l-2 border-accent/60 pl-3 text-ink-secondary">{renderInlineMarkdown(quote[1], `${key}-quote`)}</blockquote>);
      continue;
    }
    nodes.push(<p key={key} className="break-words text-ink">{renderInlineMarkdown(line, `${key}-text`)}</p>);
  }
  return <>{nodes}</>;
}

function MessageBody({ content }: { content: string }) {
  const blocks = content.split("```");
  return (
    <div className="space-y-3 text-[15px] leading-7 text-ink">
      {blocks.map((block, index) => {
        if (index % 2 === 0) {
          return block ? <MarkdownText key={index} text={block} keyPrefix={`message-${index}`} /> : null;
        }
        const newline = block.indexOf("\n");
        const language = newline > -1 ? block.slice(0, newline).trim() : "";
        const code = newline > -1 ? block.slice(newline + 1) : block;
        return (
          <div key={index} className="overflow-hidden rounded-md border border-edge-strong bg-frame text-ink">
            {language ? <div className="border-b border-edge px-3 py-1.5 text-[11px] text-ink-muted">{language}</div> : null}
            <pre className="thin-scrollbar overflow-x-auto p-3 text-[13px] leading-6"><code>{code}</code></pre>
          </div>
        );
      })}
    </div>
  );
}

function decodeSseBlock(block: string): { event: string; data: StreamEvent } | null {
  const lines = block.split("\n");
  const event = lines.find((line) => line.startsWith("event:"))?.slice(6).trim() || "message";
  const raw = lines.filter((line) => line.startsWith("data:")).map((line) => line.slice(5).trimStart()).join("\n");
  if (!raw) return null;
  try {
    return { event, data: JSON.parse(raw) as StreamEvent };
  } catch {
    return null;
  }
}

export function AssistantScreen({
  locale = "zh-CN",
  selectedTask,
  onOpenAdvanced,
}: {
  locale?: Locale;
  selectedTask?: string;
  onOpenAdvanced?: () => void;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessionId, setSessionId] = useState(() => makeId("chat"));
  const [draft, setDraft] = useState("");
  const [running, setRunning] = useState(false);
  const [route, setRoute] = useState("chat");
  const [routeLabel, setRouteLabel] = useState(tx(locale, "直接对话", "Direct chat"));
  const [contextState, setContextState] = useState<ContextState | null>(null);
  const [activities, setActivities] = useState<ActivityItem[]>([]);
  const [activityOpen, setActivityOpen] = useState(true);
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const lastPromptRef = useRef("");

  useEffect(() => {
    try {
      const stored = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || "null") as {
        sessionId?: string;
        messages?: ChatMessage[];
      } | null;
      if (stored?.sessionId && Array.isArray(stored.messages)) {
        setSessionId(stored.sessionId);
        setMessages(stored.messages.slice(-40).filter((item) =>
          item && (item.role === "user" || item.role === "assistant") && typeof item.content === "string"
        ));
      }
    } catch {
      // A damaged local draft should never block the assistant.
    }
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ sessionId, messages: messages.slice(-40) }));
    } catch {
      // Private browsing or storage quotas do not affect the live session.
    }
  }, [messages, sessionId]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: running ? "auto" : "smooth", block: "end" });
  }, [activities, messages, running]);

  const history = useMemo(() => messages
    .filter((item) => item.status === "complete" && item.content.trim())
    .map((item) => ({ role: item.role, content: item.content })), [messages]);

  const appendActivity = useCallback((event: string, data: StreamEvent) => {
    const item: ActivityItem = {
      seq: Number(data.seq ?? Date.now()),
      type: event,
      status: String(data.status ?? (event.includes("completed") ? "completed" : "running")),
      label: String(data.label ?? data.message ?? event),
      detail: data.detail ? String(data.detail) : undefined,
    };
    setActivities((current) => [...current.filter((entry) => entry.seq !== item.seq), item].slice(-12));
  }, []);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setRunning(false);
    setMessages((current) => current.map((item) => item.status === "streaming" ? { ...item, status: "stopped" } : item));
    setActivities((current) => [...current, {
      seq: Date.now(), type: "stopped", status: "stopped",
      label: tx(locale, "已停止生成", "Generation stopped"),
    }].slice(-12));
  }, [locale]);

  const send = useCallback(async (rawPrompt?: string, options?: { retry?: boolean }) => {
    const prompt = String(rawPrompt ?? draft).trim();
    if (!prompt || running) return;
    lastPromptRef.current = prompt;
    setDraft("");
    setActivities([]);
    setActivityOpen(true);
    setRoute("chat");
    setRouteLabel(tx(locale, "正在判断请求", "Routing request"));

    const assistantId = makeId("assistant");
    if (options?.retry) {
      setMessages((current) => [...current.filter((item, index) => !(index === current.length - 1 && item.role === "assistant")), {
        id: assistantId, role: "assistant", content: "", status: "streaming",
      }]);
    } else {
      setMessages((current) => [...current,
        { id: makeId("user"), role: "user", content: prompt, status: "complete" },
        { id: assistantId, role: "assistant", content: "", status: "streaming" },
      ]);
    }
    setRunning(true);
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch("/api/assistant/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({
          prompt,
          session_id: sessionId,
          selected_task: selectedTask || "",
          history: history.slice(-20),
        }),
        cache: "no-store",
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(await assistantStreamErrorMessage(response, locale));
      if (!response.body) throw new Error(tx(locale, "助手服务未返回流式响应。", "The assistant did not return a stream."));

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let completed = false;

      const handle = (event: string, data: StreamEvent) => {
        if (event === "context") {
          setContextState({
            currentTask: Boolean(data.current_task),
            taskLabel: String(data.task_label ?? ""),
            runStatus: String(data.run_status ?? "none"),
            memoryAvailable: Boolean(data.memory_available),
            toolsAvailable: Boolean(data.tools_available),
          });
          return;
        }
        if (event === "route") {
          const nextRoute = String(data.route ?? "chat");
          setRoute(nextRoute);
          setRouteLabel(String(data.label ?? nextRoute));
          appendActivity(event, data);
          return;
        }
        if (event === "answer_delta" && data.delta) {
          setMessages((current) => current.map((item) => item.id === assistantId
            ? { ...item, content: item.content + String(data.delta) }
            : item));
          return;
        }
        if (event === "answer_completed") {
          completed = true;
          setMessages((current) => current.map((item) => item.id === assistantId
            ? { ...item, content: item.content || String(data.answer ?? ""), status: "complete" }
            : item));
          return;
        }
        if (event === "error") throw new Error(String(data.message || tx(locale, "回复中断，可以重试。", "The response was interrupted. Retry it.")));
        if (["thinking_status", "tool_started", "tool_completed", "research_run", "model"].includes(event)) appendActivity(event, data);
      };

      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value, { stream: !done }).replaceAll("\r\n", "\n");
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() ?? "";
        for (const block of blocks) {
          const parsed = decodeSseBlock(block);
          if (parsed) handle(parsed.event, parsed.data);
        }
        if (done) break;
      }
      if (buffer.trim()) {
        const parsed = decodeSseBlock(buffer);
        if (parsed) handle(parsed.event, parsed.data);
      }
      if (!completed) throw new Error(tx(locale, "回复流提前结束，可以重试。", "The response stream ended early. Retry it."));
    } catch (error) {
      if (controller.signal.aborted) return;
      const message = error instanceof Error ? error.message : tx(locale, "回复失败，可以重试。", "The response failed. Retry it.");
      setMessages((current) => current.map((item) => item.id === assistantId
        ? { ...item, content: item.content || message, status: "error" }
        : item));
      setActivities((current) => [...current, {
        seq: Date.now(), type: "error", status: "error", label: message,
      }].slice(-12));
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
      setRunning(false);
    }
  }, [appendActivity, draft, history, locale, running, selectedTask, sessionId]);

  const newSession = useCallback(() => {
    stop();
    setMessages([]);
    setActivities([]);
    setDraft("");
    setSessionId(makeId("chat"));
    setRoute("chat");
    setRouteLabel(tx(locale, "直接对话", "Direct chat"));
    window.setTimeout(() => textareaRef.current?.focus(), 0);
  }, [locale, stop]);

  const retry = useCallback(() => {
    const prompt = lastPromptRef.current || [...messages].reverse().find((item) => item.role === "user")?.content || "";
    if (prompt) void send(prompt, { retry: true });
  }, [messages, send]);

  return (
    <section className="flex h-full min-h-0 flex-col bg-surface/74" aria-label={tx(locale, "EvoMind 智能助手", "EvoMind Assistant")} data-ui-assistant-screen>
      <div className="flex h-12 shrink-0 items-center border-b border-edge bg-surface-raised/78 px-3 sm:px-5">
        <div className="flex min-w-0 items-center gap-2">
          <Image src="/brand/evomind-mark.png" alt="" width={22} height={22} className="h-[22px] w-[22px] shrink-0 object-contain" priority />
          <h1 className="truncate text-sm font-semibold text-ink">EvoMind</h1>
          <span className={cn(
            "rounded border px-1.5 py-0.5 text-xs font-medium",
            route === "chat" ? "border-success/45 bg-success-light text-success-text" : "border-accent-muted bg-accent-light text-accent-dark"
          )}>{routeLabel}</span>
          <span className="hidden truncate text-xs text-ink-muted md:inline">
            {contextState?.currentTask
              ? tx(locale, `已接入 ${contextState.taskLabel || "当前任务"}`, `Attached to ${contextState.taskLabel || "current task"}`)
              : tx(locale, "项目上下文自动接入", "Project context ready")}
          </span>
          {contextState?.memoryAvailable && contextState.toolsAvailable ? (
            <span className="hidden items-center gap-1 text-xs text-success-text xl:flex">
              <span className="h-1.5 w-1.5 rounded-full bg-success" />
              {tx(locale, "记忆与工具就绪", "Memory and tools ready")}
            </span>
          ) : null}
        </div>
        <div className="ml-auto flex items-center gap-1">
          {onOpenAdvanced ? (
            <button type="button" data-ui-action="assistant_open_advanced_control" data-ui-skip-action="true" onClick={onOpenAdvanced} className="hidden h-8 rounded-md border border-success/35 bg-success-light px-2.5 text-xs font-semibold text-success-text hover:border-success/55 sm:block">
              {tx(locale, "查看研究结果", "View research results")}
            </button>
          ) : null}
          <button type="button" data-ui-action="assistant_new_session" data-ui-skip-action="true" onClick={newSession} className="flex h-8 w-8 items-center justify-center rounded-md text-ink-secondary hover:bg-surface-sunken" title={tx(locale, "新会话", "New chat")} aria-label={tx(locale, "新会话", "New chat")}>
            <Plus className="h-4 w-4" />
          </button>
        </div>
      </div>

      <div className="thin-scrollbar min-h-0 flex-1 overflow-y-auto" aria-live="polite" data-ui-assistant-messages>
        <div className="mx-auto flex min-h-full w-full max-w-3xl flex-col px-4 py-6 sm:px-8 sm:py-10">
          {messages.length === 0 ? (
            <div className="my-auto py-10">
              <Image src="/brand/evomind-mark.png" alt="EvoMind" width={48} height={48} className="mb-5 h-12 w-12 object-contain" priority />
              <h2 className="text-2xl font-semibold text-ink">{tx(locale, "今天需要处理什么？", "What should we work on?")}</h2>
              <p className="mt-2 text-sm text-ink-muted">{tx(locale, "当前任务、经验记忆和受控工具会按需自动接入。", "Current tasks, experience memory, and governed tools attach automatically.")}</p>
              <div className="mt-6 grid gap-2 sm:grid-cols-2">
                {[
                  tx(locale, "解释当前项目的核心架构", "Explain the current project architecture"),
                  tx(locale, "上次模型微调完成到哪了", "Show the previous fine-tuning status"),
                  tx(locale, "检查本地开发环境", "Check the local development environment"),
                  tx(locale, "继续优化上次模型", "Continue improving the previous model"),
                ].map((prompt) => (
                  <button key={prompt} type="button" data-ui-action="assistant_suggestion" data-ui-skip-action="true" onClick={() => void send(prompt)} className="min-h-12 rounded-md border border-edge bg-surface-raised px-3 py-2 text-left text-sm text-ink-secondary hover:border-accent/50 hover:bg-accent-light/40">
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-7">
              {messages.map((message) => (
                <article key={message.id} className={cn("group flex gap-3", message.role === "user" && "justify-end")}>
                  {message.role === "assistant" ? (
                    <span className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-frame">
                      <Image src="/brand/evomind-mark.png" alt="EvoMind" width={22} height={22} className="h-[22px] w-[22px] object-contain" />
                    </span>
                  ) : null}
                  <div className={cn(
                    "min-w-0",
                    message.role === "user" ? "max-w-[88%] rounded-md bg-surface-sunken px-4 py-2.5" : "flex-1"
                  )}>
                    {message.content ? <MessageBody content={message.content} /> : message.status === "streaming" ? (
                      <div className="flex h-7 items-center gap-1.5 text-sm text-ink-muted">
                        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />
                        {tx(locale, "正在思考", "Thinking")}
                      </div>
                    ) : message.status === "complete" ? (
                      <div className="text-sm text-ink-muted">
                        {tx(locale, "本轮检查已完成，没有新增可显示内容。", "This check completed with no new displayable content.")}
                      </div>
                    ) : (
                      <div className="text-sm text-ink-muted">
                        {message.status === "stopped"
                          ? tx(locale, "已停止生成。", "Generation stopped.")
                          : message.status === "error"
                            ? tx(locale, "本轮回复中断，可以重试。", "This response was interrupted. Retry it.")
                            : tx(locale, "本轮未生成内容。", "No content was generated.")}
                      </div>
                    )}
                    {message.role === "assistant" && message.status !== "streaming" ? (
                      <div className="mt-2 flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
                        <button type="button" data-ui-action="assistant_copy_message" data-ui-skip-action="true" onClick={() => void navigator.clipboard.writeText(message.content)} className="flex h-7 w-7 items-center justify-center rounded text-ink-muted hover:bg-surface-sunken hover:text-ink" title={tx(locale, "复制", "Copy")} aria-label={tx(locale, "复制回复", "Copy response")}><Clipboard className="h-3.5 w-3.5" /></button>
                        <button type="button" data-ui-action="assistant_retry_message" data-ui-skip-action="true" onClick={retry} disabled={running} className="flex h-7 w-7 items-center justify-center rounded text-ink-muted hover:bg-surface-sunken hover:text-ink disabled:opacity-40" title={tx(locale, "重试", "Retry")} aria-label={tx(locale, "重试回复", "Retry response")}><RotateCcw className="h-3.5 w-3.5" /></button>
                        {message.status === "stopped" ? <span className="ml-1 text-xs text-ink-muted">{tx(locale, "已停止", "Stopped")}</span> : null}
                      </div>
                    ) : null}
                  </div>
                  {message.role === "user" ? (
                    <span className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-md border border-edge bg-surface-raised text-ink-secondary"><UserRound className="h-4 w-4" /></span>
                  ) : null}
                </article>
              ))}
            </div>
          )}

          {activities.length > 0 ? (
            <div className="mt-6 border-t border-edge pt-3">
              <button type="button" data-ui-action="assistant_toggle_activity" data-ui-skip-action="true" className="flex w-full items-center gap-2 text-left text-xs font-medium text-ink-muted hover:text-ink" onClick={() => setActivityOpen((value) => !value)} aria-expanded={activityOpen}>
                <ChevronDown className={cn("h-3.5 w-3.5 transition-transform", !activityOpen && "-rotate-90")} />
                {running ? tx(locale, "正在处理", "Working") : tx(locale, "处理记录", "Activity")}
              </button>
              {activityOpen ? (
                <ol className="mt-2 space-y-1.5 pl-5">
                  {activities.map((item) => (
                    <li key={`${item.seq}-${item.type}`} className="flex gap-2 text-xs leading-5 text-ink-muted">
                      <span className={cn("mt-2 h-1.5 w-1.5 shrink-0 rounded-full", item.status === "completed" ? "bg-success" : item.status === "error" || item.status === "blocked" ? "bg-danger" : "bg-accent")} />
                      <span><span className="font-medium text-ink-secondary">{item.label}</span>{item.detail ? ` · ${item.detail}` : ""}</span>
                    </li>
                  ))}
                </ol>
              ) : null}
            </div>
          ) : null}
          <div ref={endRef} />
        </div>
      </div>

      <div className="shrink-0 border-t border-edge bg-surface-raised/92 px-3 pb-[max(12px,env(safe-area-inset-bottom))] pt-3 sm:px-6">
        <div className="mx-auto max-w-3xl">
          <div className="flex items-end gap-2 rounded-lg border border-edge-strong bg-surface-raised p-2 shadow-raised focus-within:border-accent focus-within:ring-2 focus-within:ring-accent/20">
            <textarea
              ref={textareaRef}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void send();
                }
              }}
              rows={1}
              data-ui-assistant-input
              className="max-h-40 min-h-10 flex-1 resize-none bg-transparent px-2 py-2 text-[15px] leading-6 text-ink outline-none placeholder:text-ink-muted"
              placeholder={tx(locale, "向 EvoMind 提问", "Ask EvoMind")}
              aria-label={tx(locale, "消息输入", "Message input")}
              disabled={running}
            />
            {running ? (
              <button type="button" data-ui-action="assistant_stop" data-ui-skip-action="true" onClick={stop} className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-frame-light text-white hover:bg-frame-lighter" title={tx(locale, "停止生成", "Stop")} aria-label={tx(locale, "停止生成", "Stop generation")}><Square className="h-3.5 w-3.5 fill-current" /></button>
            ) : (
              <button type="button" data-ui-action="assistant_send" data-ui-skip-action="true" onClick={() => void send()} disabled={!draft.trim()} className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-accent text-accent-fg hover:bg-accent-dark disabled:cursor-not-allowed disabled:bg-edge-strong" title={tx(locale, "发送", "Send")} aria-label={tx(locale, "发送消息", "Send message")}><ArrowUp className="h-4 w-4" /></button>
            )}
          </div>
          <div className="mt-1.5 text-center text-xs text-ink-muted">{tx(locale, "Enter 发送 · Shift+Enter 换行", "Enter to send · Shift+Enter for a new line")}</div>
        </div>
      </div>
    </section>
  );
}
