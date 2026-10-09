"use client";
import { ensureCreationKey, finishCreation, recoverCreation } from "./run-submission";

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  ArrowUp,
  ChevronDown,
  Clipboard,
  Download,
  FileText,
  Loader2,
  Paperclip,
  Plus,
  RefreshCw,
  RotateCcw,
  ShieldCheck,
  Square,
  UserRound,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { assistantRunTaskLabel } from "@/lib/task-context";
import { VerifiedReportDocument } from "./AssistantReportPanel";
import { AssistantControlledCredentials, type AssistantControlledCredentialsHandle } from "./AssistantControlledCredentials";
import { isConfirmedControlledSecretBinding, isExactPendingWeatherAcceleration } from "./controlled-secret-binding";
import { DeepEvoGlyph, DeepEvoLockup } from "../DeepEvoBrand";
import { assistantActivityIdentity, upsertAssistantActivity } from "./assistant-activity.mjs";
import { RunProgressPanel, type RunProgress } from "./RunProgressPanel";
import { assistantArtifactLabel, assistantMessagePresentation, isAssistantExecutionRecord } from "./assistant-presentation";
import { assistantRestorationTarget, loadAssistantRestoration } from "./assistant-restoration";
import { latestAssistantMessageId, restoreRunMessages, updateRunMessage, updateStreamingMessage, type MessageHistory } from "./assistant-history";

type Locale = "zh-CN" | "en-US";
type MessageStatus = "complete" | "streaming" | "stopped" | "blocked" | "error";

type AssistantArtifact = {
  id: string;
  run_id: string;
  name: string;
  path?: string;
  media_type: string;
  bytes: number;
  sha256: string;
  source_tool_call?: string;
  preview_kind: string;
  preview_url: string;
  download_url: string;
  verified_at?: string;
};

type AssistantApproval = {
  id: string;
  tool_name: string;
  risk_level: string;
  reversible: boolean;
  status: string;
  normalized_arguments?: Record<string, unknown>;
  impact_scope?: Record<string, unknown>;
};

type AssistantRun = {
  project_id?: string;
  conversation_id?: string;
  progress?: RunProgress;
  id: string;
  status: string;
  prompt: string;
  selected_task?: string;
  task_id?: string;
  answer?: string;
  answer_is_current?: boolean;
  streaming_text?: string | null;
  message_history?: MessageHistory;
  model_provider?: string;
  model?: string;
  plan?: { steps?: Array<{ id?: string; label?: string; status?: string; detail?: string }> };
  artifacts?: AssistantArtifact[];
  approvals?: AssistantApproval[];
  artifact_evidence?: AssistantArtifact[];
  attachments?: Array<{ id: string; name: string; bytes: number; sha256: string }>;
  attachment_context?: { mode?: "none" | "explicit" | "inherited" | "repaired"; source_run_id?: string };
  terminal?: boolean;
  error_class?: string;
  error_message?: string;
  last_event_seq?: number;
  created_at?: string;
  controlled_secret_availability?: Record<string, number>;
};

type UploadDraft = {
  id: string;
  file: File;
  progress: number;
  uploadedBytes: number;
  phase: "queued" | "preparing" | "hashing" | "uploading" | "verifying" | "complete" | "error";
  status: "queued" | "uploading" | "complete" | "error";
  attachmentId?: string;
  attachmentSha256?: string;
  error?: string;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  status: MessageStatus;
  runId?: string;
  artifacts?: AssistantArtifact[];
  attachmentNames?: string[];
};

type ActivityItem = {
  identity?: string;
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
  text?: string;
  phase?: string;
  progress?: RunProgress;
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
  run_id?: string;
  provider?: string;
  model?: string;
  tool?: string;
  tool_call_id?: string;
  error?: string;
  step_id?: string;
  plan?: AssistantRun["plan"];
  approval?: AssistantApproval;
  artifact?: AssistantArtifact;
  artifacts?: AssistantArtifact[];
  recoverable?: boolean;
};

const STORAGE_KEY = "evomind_assistant_session_v2";
const LOCAL_CSRF_STORAGE_KEY = "evomind.local.csrf.v1";
const SAFE_AUTO_APPROVAL_STORAGE_KEY = "evomind.assistant.safe_auto_approval.v1";
const CREDENTIAL_RUN_ID = /^run_[0-9a-f]{32}$/;
const UPLOAD_CHUNK_BYTES = 16 * 1024 * 1024;

const SAFE_AUTO_APPROVAL_TOOLS = new Set([
  "artifact_list", "artifact_preview", "artifact_publish", "capability_discover",
  "directory_hash", "directory_list", "directory_mkdir", "directory_read", "directory_stat", "directory_write_atomic",
  "hpc_execute_solution", "hpc_verify", "research_invoke", "training_route", "transfer_fetch",
]);
const SAFE_RESEARCH_ACTIONS = new Set([
  "ask", "briefing", "causal-diagnosis", "engineer", "readiness-report", "scientist", "self-audit", "strategy", "upgrade-plan", "workspace",
]);
const SENSITIVE_APPROVAL_TERMS = /(?:credential|secret|password|passwd|token|cookie|authorization|join|submit|submission|terms|delete|remove|permission|account|private[_ -]?label|hidden[_ -]?label|密码|口令|凭据|令牌|提交|条款|删除|权限|账号|隐藏标签)/i;

function isSafeSameRunApproval(approval: AssistantApproval, runId: string) {
  if (!runId || approval.status !== "pending" || !approval.reversible) return false;
  if (!new Set(["low", "medium"]).has(String(approval.risk_level).toLowerCase())) return false;
  if (!SAFE_AUTO_APPROVAL_TOOLS.has(approval.tool_name)) return false;
  const scope = approval.impact_scope ?? {};
  if (scope.outside_workspace !== false || scope.cross_run_workspace !== false) return false;
  const scopedRun = String(scope.run_id ?? "");
  const workspaceRoot = String(scope.workspace_root ?? "");
  if (scopedRun ? scopedRun !== runId : !workspaceRoot.toLowerCase().includes(runId.toLowerCase())) return false;
  const args = approval.normalized_arguments ?? {};
  if (SENSITIVE_APPROVAL_TERMS.test(JSON.stringify({ args, scope }))) return false;
  if (approval.tool_name === "research_invoke" && !SAFE_RESEARCH_ACTIONS.has(String(args.action ?? ""))) return false;
  if (approval.tool_name === "transfer_fetch") {
    try {
      const source = new URL(String(args.source_url ?? ""));
      if (source.protocol !== "https:" || source.hostname !== "raw.githubusercontent.com") return false;
    } catch {
      return false;
    }
  }
  return true;
}

function tx(locale: Locale, zh: string, en: string) {
  return locale === "zh-CN" ? zh : en;
}

function runStatusLabel(status: string, locale: Locale) {
  if (status === "completed") return tx(locale, "执行结束", "Execution ended");
  if (status === "waiting_approval") return tx(locale, "待确认", "Needs approval");
  if (["blocked", "failed", "cancelled"].includes(status)) return tx(locale, "需处理", "Needs attention");
  return tx(locale, "进行中", "In progress");
}

function runStatusSummary(run: AssistantRun, locale: Locale) {
  if (run.status === "completed") return tx(locale, "本次执行已结束，文件已保留；研究目标是否通过请查看验收结果。", "Execution ended and files are retained; check acceptance for the research outcome.");
  if (run.status === "waiting_approval") return tx(locale, "需要你的确认后才能继续。", "Your approval is needed before this task can continue.");
  if (run.error_class === "controlled_secret_required") return tx(locale, "需要更新连接凭据后继续，任务进度已保留。", "Update the connection credentials to continue; progress is saved.");
  if (["blocked", "failed", "cancelled"].includes(run.status)) return tx(locale, "任务进度已保留，可补充要求后继续。", "Progress is saved. Add an instruction to continue.");
  return tx(locale, "任务正在后台处理，可以继续浏览当前对话。", "The task is running in the background; you can keep using this conversation.");
}

function ArtifactCards({ artifacts, locale, onPreview }: { artifacts: AssistantArtifact[]; locale: Locale; onPreview: (artifact: AssistantArtifact) => void }) {
  return <div className="mt-3 grid gap-2 sm:grid-cols-2" data-ui-assistant-artifacts>{artifacts.map((artifact) => <div key={artifact.id} data-ui-assistant-artifact-card className="rounded-md border border-edge bg-surface-raised p-3 shadow-sm">
    <div className="flex items-start gap-2"><FileText className="mt-0.5 h-4 w-4 shrink-0 text-accent" /><div className="min-w-0"><div className="truncate text-sm font-semibold text-ink">{artifact.name}</div><div className="mt-1 text-[11px] text-ink-secondary">{assistantArtifactLabel(artifact.name, locale)}</div><div className="mt-0.5 text-[11px] text-ink-muted">{formatBytes(artifact.bytes)} · SHA-256 {artifact.sha256.slice(0, 12)}…</div></div></div>
    <div className="mt-2 flex flex-wrap gap-1.5">
      <button type="button" data-ui-action="assistant_artifact_preview" onClick={() => onPreview(artifact)} className="rounded border border-edge px-2 py-1 text-xs text-ink-secondary hover:border-accent hover:text-accent">{tx(locale, "预览", "Preview")}</button>
      <a href={artifact.download_url} download={artifact.name} data-ui-action="assistant_artifact_download" className="inline-flex items-center gap-1 rounded border border-edge px-2 py-1 text-xs text-ink-secondary hover:border-accent hover:text-accent"><Download className="h-3 w-3" />{tx(locale, "下载", "Download")}</a>
      <button type="button" onClick={() => void navigator.clipboard.writeText(artifact.path || artifact.name)} className="rounded border border-edge px-2 py-1 text-xs text-ink-secondary hover:border-accent hover:text-accent">{tx(locale, "复制路径", "Copy path")}</button>
      <button type="button" onClick={() => void navigator.clipboard.writeText(artifact.sha256)} className="rounded border border-edge px-2 py-1 text-xs text-ink-secondary hover:border-accent hover:text-accent">{tx(locale, "复制哈希", "Copy hash")}</button>
    </div>
  </div>)}</div>;
}

function makeId(prefix: string) {
  const random = typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID().replaceAll("-", "").slice(0, 14)
    : `${Date.now()}_${Math.random().toString(16).slice(2, 8)}`;
  return `${prefix}_${random}`;
}

function formatBytes(value: number) {
  if (!Number.isFinite(value) || value < 0) return "—";
  if (value < 1024) return `${value} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let amount = value / 1024;
  let unit = 0;
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024;
    unit += 1;
  }
  return `${amount >= 10 ? amount.toFixed(1) : amount.toFixed(2)} ${units[unit]}`;
}

async function sha256Hex(blob: Blob) {
  const bytes = await blob.arrayBuffer();
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest)).map((value) => value.toString(16).padStart(2, "0")).join("");
}

function uploadFailureMessage(request: XMLHttpRequest) {
  try {
    const payload = JSON.parse(request.responseText) as { message?: unknown; error?: unknown; code?: unknown };
    return String(payload.message ?? payload.error ?? payload.code ?? `upload_${request.status}`);
  } catch {
    return `upload_${request.status || "network_error"}`;
  }
}

function uploadChunkWithProgress(
  url: string,
  chunk: Blob,
  digest: string,
  signal: AbortSignal,
  onProgress: (loadedBytes: number) => void,
) {
  return new Promise<void>((resolve, reject) => {
    const csrf = window.sessionStorage.getItem(LOCAL_CSRF_STORAGE_KEY) ?? "";
    if (!csrf) {
      reject(new Error("upload_csrf_missing"));
      return;
    }
    const request = new XMLHttpRequest();
    let settled = false;
    const cleanup = () => signal.removeEventListener("abort", abort);
    const finish = (callback: () => void) => {
      if (settled) return;
      settled = true;
      cleanup();
      callback();
    };
    const abort = () => request.abort();
    request.open("PUT", url, true);
    request.withCredentials = true;
    request.setRequestHeader("Content-Type", "application/octet-stream");
    request.setRequestHeader("X-Chunk-SHA256", digest);
    request.setRequestHeader("X-EvoMind-CSRF", csrf);
    request.upload.onprogress = (event) => onProgress(Math.min(chunk.size, event.loaded));
    request.onload = () => finish(() => {
      if (request.status >= 200 && request.status < 300) resolve();
      else reject(new Error(uploadFailureMessage(request)));
    });
    request.onerror = () => finish(() => reject(new Error("upload_network_error")));
    request.onabort = () => finish(() => {
      const error = new Error("upload_aborted");
      error.name = "AbortError";
      reject(error);
    });
    if (signal.aborted) {
      const error = new Error("upload_aborted");
      error.name = "AbortError";
      reject(error);
      return;
    }
    signal.addEventListener("abort", abort, { once: true });
    request.send(chunk);
  });
}

function uploadPhaseText(item: UploadDraft, locale: Locale) {
  if (item.status === "error") return item.error ?? tx(locale, "上传失败", "Upload failed");
  if (item.status === "complete") return tx(locale, "已校验 · 100%", "Verified · 100%");
  if (item.phase === "preparing") return tx(locale, `建立上传会话 · ${item.progress}%`, `Preparing upload · ${item.progress}%`);
  if (item.phase === "hashing") return tx(locale, `计算分块哈希 · ${item.progress}%`, `Hashing chunk · ${item.progress}%`);
  if (item.phase === "verifying") return tx(locale, `服务器校验 · ${item.progress}%`, `Server verification · ${item.progress}%`);
  if (item.phase === "uploading") return `${formatBytes(item.uploadedBytes)} / ${formatBytes(item.file.size)} · ${item.progress}%`;
  return tx(locale, "等待上传 · 0%", "Queued · 0%");
}

function mergeArtifacts(current: AssistantArtifact[] = [], incoming: AssistantArtifact[] = []) {
  const byId = new Map(current.map((item) => [item.id, item]));
  for (const item of incoming) if (item?.id) byId.set(item.id, item);
  return [...byId.values()];
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
      "本地会话已失效：通常是 DeepEvo 重启后浏览器还停留在旧页面。请用新的启动链接重新打开工作站，然后再发同一句话；不会启动训练或提交 Kaggle。",
      "The local session expired, usually because DeepEvo restarted while this tab stayed open. Reopen the workstation with the latest launch URL, then send the same message again.",
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

function decodeSseBlock(block: string): { event: string; id?: number; data: StreamEvent } | null {
  const lines = block.split("\n");
  const event = lines.find((line) => line.startsWith("event:"))?.slice(6).trim() || "message";
  const rawId = lines.find((line) => line.startsWith("id:"))?.slice(3).trim();
  const id = rawId && Number.isSafeInteger(Number(rawId)) ? Number(rawId) : undefined;
  const raw = lines.filter((line) => line.startsWith("data:")).map((line) => line.slice(5).trimStart()).join("\n");
  if (!raw) return null;
  try {
    return { event, id, data: JSON.parse(raw) as StreamEvent };
  } catch {
    return null;
  }
}

export function AssistantScreen({
  locale = "zh-CN",
  selectedTask,
  selectedProjectId = "",
  onOpenAdvanced,
  onNewExperiment,
}: {
  locale?: Locale;
  selectedTask?: string;
  selectedProjectId?: string;
  onOpenAdvanced?: (runId?: string) => void;
  onNewExperiment?: () => void;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessionId, setSessionId] = useState(() => makeId("chat"));
  const [sessionReady, setSessionReady] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [loginRequired, setLoginRequired] = useState(false);
  const [draftOwner, setDraftOwner] = useState("");
  const [draft, setDraft] = useState("");
  const [running, setRunning] = useState(false);
  const [route, setRoute] = useState("agent");
  const [routeLabel, setRouteLabel] = useState(tx(locale, "持久任务 Agent", "Durable task agent"));
  const [contextState, setContextState] = useState<ContextState | null>(null);
  const [activities, setActivities] = useState<ActivityItem[]>([]);
  const [activityOpen, setActivityOpen] = useState(false);
  const [uploadDrafts, setUploadDrafts] = useState<UploadDraft[]>([]);
  const [reuseRetainedAttachments, setReuseRetainedAttachments] = useState(false);
  const [dismissedAttachmentRunId, setDismissedAttachmentRunId] = useState("");
  const [autoApproveSafeActions, setAutoApproveSafeActions] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [currentRun, setCurrentRun] = useState<AssistantRun | null>(null);
  const [runDetailsOpen, setRunDetailsOpen] = useState(false);
  const [approvalBusyId, setApprovalBusyId] = useState("");
  const [credentialRunId, setCredentialRunId] = useState("");
  const [credentialAttachBusy, setCredentialAttachBusy] = useState(false);
  const [preview, setPreview] = useState<{ artifact: AssistantArtifact; content?: string; entries?: Array<{ name: string; bytes: number }>; loading: boolean; error?: string } | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const uploadControllersRef = useRef(new Map<string, AbortController>());
  const uploadPromisesRef = useRef(new Map<string, Promise<string>>());
  const controlledCredentialsRef = useRef<AssistantControlledCredentialsHandle>(null);
  const approvalBusyRef = useRef("");
  const endRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const lastPromptRef = useRef("");
  const sendBusyRef = useRef(false);
  const continueRunRef = useRef(false);
  const requestedRunRef = useRef<string | null>(null);
  const activeRunIdRef = useRef<string | null>(null);
  const runEventCursorRef = useRef(new Map<string, number>());
  const [restorationError, setRestorationError] = useState("");
  const [projectName, setProjectName] = useState("");

  const isActiveStatus = useCallback((status: string) => ["queued", "planning", "running", "verifying", "recovering", "pausing"].includes(status), []);

  useEffect(() => {
    setReuseRetainedAttachments(false);
    setDismissedAttachmentRunId(currentRun?.attachment_context?.mode === "inherited" ? (currentRun?.id ?? "") : "");
  }, [currentRun?.attachment_context?.mode, currentRun?.id]);

  useEffect(() => {
    setAutoApproveSafeActions(window.localStorage.getItem(SAFE_AUTO_APPROVAL_STORAGE_KEY) === "enabled");
  }, []);

  useEffect(() => {
    setRunDetailsOpen(false);
  }, [currentRun?.id]);

  useEffect(() => {
    const value = new URL(window.location.href).searchParams.get("credential_run") ?? "";
    setCredentialRunId(CREDENTIAL_RUN_ID.test(value) ? value : "");
  }, []);

  const apiError = useCallback(async (response: Response) => {
    if (response.status === 401) setLoginRequired(true);
    const friendly = await assistantStreamErrorMessage(response, locale);
    try {
      const payload = await response.clone().json() as { message?: unknown; error?: unknown; code?: unknown };
      return String(payload.message ?? payload.error ?? payload.code ?? friendly);
    } catch {
      return friendly;
    }
  }, [locale]);

  const appendActivity = useCallback((event: string, data: StreamEvent) => {
    // The model is already shown in the header and technical details. Keeping
    // it out of the activity feed prevents repeated "served model" rows from
    // competing with the current task state.
    if (event === "model_observed") return;
    const defaults: Record<string, string> = {
      run_created: tx(locale, "任务已持久化", "Run persisted"),
      plan_updated: tx(locale, "计划已更新", "Plan updated"),
      step_started: tx(locale, "开始执行步骤", "Step started"),
      tool_started: tx(locale, `开始 ${String(data.tool ?? "工具")}`, `Started ${String(data.tool ?? "tool")}`),
      tool_completed: tx(locale, data.status === "failed" ? `工具失败 ${String(data.tool ?? "")}` : `完成 ${String(data.tool ?? "工具")}`, data.status === "failed" ? `Failed ${String(data.tool ?? "tool")}` : `Completed ${String(data.tool ?? "tool")}`),
      verification_completed: tx(locale, "结果验证完成", "Verification completed"),
      retry_scheduled: tx(locale, "已安排自动恢复重试", "Recovery retry scheduled"),
      approval_required: tx(locale, "等待精确审批", "Exact approval required"),
      artifact_published: tx(locale, "产物已验证发布", "Artifact verified and published"),
      attachments_inherited: tx(locale, "已继承本对话中校验通过的附件", "Verified conversation attachments retained"),
      model_observed: tx(locale, "已记录实际模型", "Served model observed"),
      run_blocked: tx(locale, "任务已保留现场并阻塞", "Run blocked with state preserved"),
    };
    const item: ActivityItem = {
      identity: assistantActivityIdentity({
        type: event,
        seq: Number(data.seq ?? Date.now()),
        toolCallId: data.tool_call_id,
        stepId: data.step_id,
        approvalId: data.approval?.id,
        artifactId: data.artifact?.id,
      }),
      seq: Number(data.seq ?? Date.now()),
      type: event,
      status: String(data.status ?? (event.includes("completed") || event === "artifact_published" ? "completed" : event.includes("blocked") ? "blocked" : "running")),
      label: String(data.label ?? data.message ?? defaults[event] ?? event),
      detail: data.detail ? String(data.detail) : data.error ? String(data.error) : undefined,
    };
    setActivities((current) => upsertAssistantActivity(current, item));
  }, [locale]);

  const applySnapshot = useCallback((run: AssistantRun, assistantId?: string) => {
    const cursor = Number(run.last_event_seq ?? 0);
    if (cursor < (runEventCursorRef.current.get(run.id) ?? 0)) return;
    runEventCursorRef.current.set(run.id, cursor);
    activeRunIdRef.current = run.id;
    const runTaskLabel = assistantRunTaskLabel(run, selectedTask);
    setCurrentRun(run);
    const locationUrl = new URL(window.location.href);
    locationUrl.searchParams.set("assistant_run", run.id);
    if (run.project_id) locationUrl.searchParams.set("project", run.project_id);
    else locationUrl.searchParams.delete("project");
    const acceptedConversation = (run as AssistantRun & { conversation_id?: string }).conversation_id || sessionId;
    if (acceptedConversation) locationUrl.searchParams.set("conversation", acceptedConversation);
    window.history.replaceState(null, "", locationUrl);
    setRoute("agent");
    setRouteLabel(tx(locale, "持久任务 Agent", "Durable task agent"));
    setContextState({
      currentTask: true,
      taskLabel: runTaskLabel,
      runStatus: run.status,
      memoryAvailable: true,
      toolsAvailable: true,
    });
    setRunning(isActiveStatus(run.status));
    // Keep technical activity opt-in; run progress has its own concise panel.
    setMessages((current) => updateRunMessage(current, run, assistantId));
  }, [isActiveStatus, locale, selectedTask, sessionId]);

  const connectRun = useCallback(async (runId: string, assistantId: string, controller: AbortController, initialAfter = 0) => {
    let after = initialAfter;
    try {
      while (!controller.signal.aborted) {
        const response = await fetch(`/api/assistant/runs/${encodeURIComponent(runId)}/events?after=${after}`, {
          headers: { Accept: "text/event-stream", ...(after ? { "Last-Event-ID": String(after) } : {}) },
          cache: "no-store",
          signal: controller.signal,
        });
        if (!response.ok) throw new Error(await apiError(response));
        if (!response.body) throw new Error(tx(locale, "任务事件流为空。", "The run event stream was empty."));
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let stopStream = false;

        const handle = (event: string, data: StreamEvent) => {
          if (controller.signal.aborted || abortRef.current !== controller || (data.run_id && data.run_id !== runId)) return;
          if (typeof data.seq === "number") {
            if (data.seq <= Math.max(after, runEventCursorRef.current.get(runId) ?? 0)) return;
            after = data.seq;
          }
          runEventCursorRef.current.set(runId, Math.max(after, runEventCursorRef.current.get(runId) ?? 0));
          if (event === "assistant_text" && typeof data.text === "string") {
            setMessages((current) => updateStreamingMessage(current, runId, assistantId, data.text!));
            setCurrentRun((current) => current?.id === runId ? { ...current, streaming_text: data.text, last_event_seq: after } : current);
            return;
          }
          if (event === "run_progress" && data.progress) {
            setCurrentRun((current) => current?.id === runId ? { ...current, progress: data.progress, last_event_seq: after } : current);
            return;
          } else if (event === "plan_updated" && data.plan) {
            setCurrentRun((current) => current?.id === runId ? { ...current, plan: data.plan } : current);
          } else if (event === "step_started") {
            const nextStatus = data.step_id === "understand" ? "planning" : data.step_id === "verify" || data.step_id === "deliver" ? "verifying" : "running";
            setCurrentRun((current) => current?.id === runId ? { ...current, status: nextStatus } : current);
            setRunning(true);
          } else if (event === "tool_started" || event === "tool_completed") {
            setCurrentRun((current) => current?.id === runId ? { ...current, status: "running" } : current);
            setRunning(true);
          } else if (event === "retry_scheduled") {
            setCurrentRun((current) => current?.id === runId ? { ...current, status: "recovering" } : current);
            setRunning(true);
          } else if (event === "model_observed") {
            setCurrentRun((current) => current?.id === runId ? { ...current, model_provider: String(data.provider ?? ""), model: String(data.model ?? "") } : current);
          } else if (event === "artifact_published" && data.artifact) {
            setCurrentRun((current) => current?.id === runId ? { ...current, artifacts: mergeArtifacts(current.artifacts, [data.artifact!]) } : current);
            setMessages((current) => current.map((item) => item.id === assistantId ? { ...item, artifacts: mergeArtifacts(item.artifacts, [data.artifact!]) } : item));
          } else if (event === "approval_required" && data.approval) {
            setCurrentRun((current) => current?.id === runId ? {
              ...current,
              status: "waiting_approval",
              approvals: [...(current.approvals ?? []).filter((item) => item.id !== data.approval!.id), data.approval!],
            } : current);
            setMessages((current) => current.map((item) => item.id === assistantId ? { ...item, status: "blocked" } : item));
            setRunning(false);
            stopStream = true;
          } else if (event === "run_paused") {
            setCurrentRun((current) => current?.id === runId ? { ...current, status: "paused", terminal: false } : current);
            setRunning(false);
            stopStream = true;
          } else if (event === "run_completed") {
            const artifacts = mergeArtifacts([], data.artifacts);
            setCurrentRun((current) => current?.id === runId ? { ...current, status: "completed", answer: String(data.answer ?? current.answer ?? ""), artifacts: mergeArtifacts(current.artifacts, artifacts), terminal: true } : current);
            setMessages((current) => current.map((item) => item.id === assistantId ? {
              ...item,
              content: String(data.answer ?? item.content),
              status: "complete",
            } : item));
            setRunning(false);
            stopStream = true;
          } else if (event === "run_blocked" || event === "run_failed") {
            const blocked = event === "run_blocked" || Boolean(data.recoverable);
            const message = String(data.message ?? tx(locale, "任务已保留现场，可继续。", "The run state was preserved and can be resumed."));
            setCurrentRun((current) => current?.id === runId ? { ...current, status: blocked ? "blocked" : "failed", error_message: message, terminal: true } : current);
            setMessages((current) => current.map((item) => item.id === assistantId ? { ...item, content: item.content || message, status: blocked ? "blocked" : "error" } : item));
            setRunning(false);
            // A newer direct managed tool may already be running after this event.
            // Consume the remaining event batch, then reconcile against the durable snapshot below.
          }
          appendActivity(event, data);
        };

        while (!controller.signal.aborted && !stopStream) {
          const { value, done } = await reader.read();
          buffer += decoder.decode(value, { stream: !done }).replaceAll("\r\n", "\n");
          const blocks = buffer.split("\n\n");
          buffer = blocks.pop() ?? "";
          for (const block of blocks) {
            const parsed = decodeSseBlock(block);
            if (parsed) handle(parsed.event, { ...parsed.data, seq: parsed.data.seq ?? parsed.id });
            if (stopStream) break;
          }
          if (done) break;
        }
        if (stopStream || controller.signal.aborted) {
          await reader.cancel().catch(() => undefined);
          return;
        }

        const snapshotResponse = await fetch(`/api/assistant/runs/${encodeURIComponent(runId)}`, { cache: "no-store", signal: controller.signal });
        if (!snapshotResponse.ok) throw new Error(await apiError(snapshotResponse));
        const snapshot = await snapshotResponse.json() as AssistantRun;
        if (controller.signal.aborted || abortRef.current !== controller) return;
        applySnapshot(snapshot, assistantId);
        if (snapshot.terminal || snapshot.status === "waiting_approval" || snapshot.status === "cancelled" || snapshot.status === "paused") return;
        await new Promise((resolve) => window.setTimeout(resolve, 350));
      }
    } catch (error) {
      if (controller.signal.aborted || abortRef.current !== controller) return;
      const message = error instanceof Error ? error.message : tx(locale, "任务事件流中断。", "The run event stream was interrupted.");
      setActivities((current) => [...current, { seq: Date.now(), type: "stream_recovering", status: "blocked", label: message }].slice(-30));
      try {
        const snapshotResponse = await fetch(`/api/assistant/runs/${encodeURIComponent(runId)}`, { cache: "no-store" });
        if (snapshotResponse.ok) {
          const snapshot = await snapshotResponse.json() as AssistantRun;
          if (controller.signal.aborted || abortRef.current !== controller) return;
          applySnapshot(snapshot, assistantId);
          if (isActiveStatus(snapshot.status)) {
            const replacement = new AbortController();
            abortRef.current = replacement;
            window.setTimeout(() => void connectRun(runId, assistantId, replacement, after), 1000);
          }
          return;
        }
      } catch {
        // The durable run remains server-side even if this tab cannot reconnect yet.
      }
      setRunning(false);
      setMessages((current) => current.map((item) => item.id === assistantId ? { ...item, content: item.content || message, status: "blocked" } : item));
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
    }
  }, [apiError, appendActivity, applySnapshot, isActiveStatus, locale]);

  useEffect(() => {
    try {
      let stored: { sessionId?: string } | null = null;
      try { stored = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || "null"); } catch { /* URL intent still wins over a damaged cache. */ }
      const target = assistantRestorationTarget(window.location.href, stored?.sessionId, sessionId);
      if (target.kind === "run") requestedRunRef.current = target.id;
      else setSessionId(target.id);
    } catch {
      requestedRunRef.current = "invalid";
      setRestorationError(tx(locale, "任务链接无效，请从项目与实验重新选择。", "Invalid task link. Select an experiment from Projects."));
    } finally {
      setSessionReady(true);
    }
  }, []);

  useEffect(() => {
    if (!sessionReady) return;
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ sessionId }));
    } catch {
      // The server remains authoritative when local storage is unavailable.
    }
    let disposed = false;
    const controller = new AbortController();
    setHistoryLoading(true);
    void (async () => {
      try {
        if (requestedRunRef.current === "invalid") throw new Error("invalid_run_link");
        const explicit = requestedRunRef.current;
        const runs = await loadAssistantRestoration<AssistantRun>(explicit ? { kind: "run", id: explicit } : { kind: "conversation", id: sessionId }, fetch, controller.signal);
        if (disposed) return;
        setRestorationError("");
        const restored: ChatMessage[] = [];
        for (const run of runs) {
          restored.push(...restoreRunMessages(run, tx(locale, "继续执行此任务（恢复记录）", "Continue this task (recovery record)")));
        }
        setMessages(restored);
        const latest = runs.at(-1);
        if (latest) {
          let latestSnapshot = latest;
          try {
            if (explicit) {
              if (latest.conversation_id && latest.conversation_id !== sessionId) setSessionId(latest.conversation_id);
            } else {
            const snapshotResponse = await fetch(`/api/assistant/runs/${encodeURIComponent(latest.id)}`, { cache: "no-store", signal: controller.signal });
            if (snapshotResponse.ok) latestSnapshot = await snapshotResponse.json() as AssistantRun;
            }
          } catch {
            // Keep the history response as a safe fallback when the snapshot
            // request is unavailable; the next reconnect can reconcile it.
          }
          if (disposed) return;
          const assistantId = latestAssistantMessageId(restored, latest.id) ?? `assistant_${latest.id}`;
          applySnapshot(latestSnapshot, assistantId);
          if (isActiveStatus(latestSnapshot.status)) {
            abortRef.current?.abort();
            const live = new AbortController();
            abortRef.current = live;
            void connectRun(latest.id, assistantId, live, latestSnapshot.last_event_seq ?? 0);
          }
        } else {
          setCurrentRun(null);
          setContextState({ currentTask: Boolean(selectedTask), taskLabel: selectedTask || "", runStatus: "none", memoryAvailable: true, toolsAvailable: true });
        }
      } catch (error) {
        if (!controller.signal.aborted) {
          if ((error as { status?: number }).status === 401) setLoginRequired(true);
          setRestorationError(tx(locale, "无法读取指定任务，可能已失效或无权访问。不会改为其他任务。", "The requested task is unavailable or not accessible. No other task was substituted."));
          setCurrentRun(null); setMessages([]); setRunning(false);
          setActivities([{ seq: Date.now(), type: "history", status: "blocked", label: error instanceof Error ? error.message : "history_load_failed" }]);
        }
      } finally {
        if (!disposed) setHistoryLoading(false);
      }
    })();
    return () => {
      disposed = true;
      controller.abort();
    };
  }, [apiError, applySnapshot, connectRun, isActiveStatus, selectedTask, sessionId, sessionReady]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: running ? "auto" : "smooth", block: "end" });
  }, [activities, messages, running, uploadDrafts]);

  const queueFiles = useCallback((files: FileList | File[]) => {
    const incoming = Array.from(files).slice(0, 50);
    setUploadDrafts((current) => {
      const known = new Set(current.map((item) => `${item.file.name}:${item.file.size}:${item.file.lastModified}`));
      const additions = incoming
        .filter((file) => !known.has(`${file.name}:${file.size}:${file.lastModified}`))
        .map((file): UploadDraft => ({ id: makeId("upload_draft"), file, progress: 0, uploadedBytes: 0, phase: "queued", status: "queued" }));
      return [...current, ...additions].slice(0, 50);
    });
  }, []);

  const uploadOne = useCallback((draftItem: UploadDraft, outerSignal?: AbortSignal) => {
    if (draftItem.attachmentId) return Promise.resolve(draftItem.attachmentId);
    const existing = uploadPromisesRef.current.get(draftItem.id);
    if (existing) return existing;
    const update = (fields: Partial<UploadDraft>) => setUploadDrafts((current) => current.map((item) => item.id === draftItem.id ? { ...item, ...fields } : item));
    const controller = new AbortController();
    const forwardAbort = () => controller.abort();
    if (outerSignal?.aborted) controller.abort();
    else outerSignal?.addEventListener("abort", forwardAbort, { once: true });
    uploadControllersRef.current.set(draftItem.id, controller);
    update({ status: "uploading", phase: "preparing", progress: 1, uploadedBytes: 0, error: undefined });
    const promise = (async () => {
      const createResponse = await fetch("/api/assistant/uploads", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: draftItem.file.name, total_bytes: draftItem.file.size, media_type: draftItem.file.type }),
        cache: "no-store",
        signal: controller.signal,
      });
      if (!createResponse.ok) throw new Error(await apiError(createResponse));
      const upload = await createResponse.json() as { id: string; chunk_size?: number };
      const chunkSize = Number(upload.chunk_size) || UPLOAD_CHUNK_BYTES;
      const chunkCount = Math.ceil(draftItem.file.size / chunkSize);
      for (let index = 0; index < chunkCount; index += 1) {
        const start = index * chunkSize;
        const end = Math.min(draftItem.file.size, (index + 1) * chunkSize);
        const chunk = draftItem.file.slice(start, end);
        const baseProgress = draftItem.file.size ? Math.max(1, Math.round((start / draftItem.file.size) * 94)) : 1;
        update({ phase: "hashing", progress: baseProgress, uploadedBytes: start });
        const digest = await sha256Hex(chunk);
        update({ phase: "uploading", progress: baseProgress, uploadedBytes: start });
        await uploadChunkWithProgress(
          `/api/assistant/uploads/${encodeURIComponent(upload.id)}/chunks/${index}`,
          chunk,
          digest,
          controller.signal,
          (loadedBytes) => {
            const transferred = Math.min(draftItem.file.size, start + loadedBytes);
            const progress = draftItem.file.size ? Math.min(94, Math.max(1, Math.round((transferred / draftItem.file.size) * 94))) : 94;
            update({ phase: "uploading", progress, uploadedBytes: transferred });
          },
        );
        update({
          phase: "uploading",
          progress: draftItem.file.size ? Math.min(94, Math.max(1, Math.round((end / draftItem.file.size) * 94))) : 94,
          uploadedBytes: end,
        });
      }
      update({ phase: "verifying", progress: 96, uploadedBytes: draftItem.file.size });
      const completeResponse = await fetch(`/api/assistant/uploads/${encodeURIComponent(upload.id)}/complete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
        cache: "no-store",
        signal: controller.signal,
      });
      if (!completeResponse.ok) throw new Error(await apiError(completeResponse));
      const completed = await completeResponse.json() as { attachment: { id: string; sha256: string } };
      update({ status: "complete", phase: "complete", progress: 100, uploadedBytes: draftItem.file.size, attachmentId: completed.attachment.id, attachmentSha256: completed.attachment.sha256 });
      return completed.attachment.id;
    })().catch((error) => {
      const message = error instanceof Error ? error.message : "upload_failed";
      update({ status: "error", phase: "error", error: message });
      uploadPromisesRef.current.delete(draftItem.id);
      throw error;
    }).finally(() => {
      outerSignal?.removeEventListener("abort", forwardAbort);
      uploadControllersRef.current.delete(draftItem.id);
    });
    uploadPromisesRef.current.set(draftItem.id, promise);
    return promise;
  }, [apiError]);

  useEffect(() => {
    if (!sessionReady) return;
    for (const item of uploadDrafts) {
      if (item.status === "queued") void uploadOne(item).catch(() => undefined);
    }
  }, [sessionReady, uploadDrafts, uploadOne]);

  const removeUploadDraft = useCallback((id: string) => {
    uploadControllersRef.current.get(id)?.abort();
    uploadControllersRef.current.delete(id);
    uploadPromisesRef.current.delete(id);
    setUploadDrafts((current) => current.filter((item) => item.id !== id));
  }, []);

  useEffect(() => () => {
    abortRef.current?.abort();
    for (const controller of uploadControllersRef.current.values()) controller.abort();
    uploadControllersRef.current.clear();
    uploadPromisesRef.current.clear();
  }, []);

  const adoptRecoveredCreation = useCallback((run: AssistantRun) => {
    const conversation = (run as AssistantRun & { conversation_id?: string }).conversation_id;
    if (conversation) setSessionId(conversation);
    continueRunRef.current = false;
    const restored = restoreRunMessages(run, tx(locale, "继续执行此任务（恢复记录）", "Continue this task (recovery record)"));
    const assistantId = latestAssistantMessageId(restored, run.id) ?? `assistant_${run.id}`;
    setMessages((current) => current.some((item) => item.runId === run.id) ? current : [...current, ...restored]);
    applySnapshot(run, assistantId);
    finishCreation(sessionId, draftOwner);
    setActivities((current) => [...current, { seq: Date.now(), type: "request_recovered", status: "completed", label: tx(locale, "已恢复此前请求，没有重复创建。当前草稿未重新发送。", "Previous request recovered without creating another Run. The current draft was not resent.") }]);
    if (isActiveStatus(run.status)) {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      void connectRun(run.id, assistantId, controller, run.last_event_seq ?? 0);
    }
  }, [applySnapshot, connectRun, draftOwner, isActiveStatus, locale, sessionId]);

  useEffect(() => {
    if (!sessionReady) return;
    const controller = new AbortController();
    void recoverCreation<AssistantRun>(sessionId, draftOwner, controller.signal).then((run) => {
      if (run && !controller.signal.aborted) adoptRecoveredCreation(run);
    }).catch((error) => {
      if (!controller.signal.aborted) setActivities((current) => [...current, { seq: Date.now(), type: "request_recovery_pending", status: "blocked", label: error instanceof Error ? error.message : "run_request_recovery_unavailable" }]);
    });
    return () => controller.abort();
  }, [adoptRecoveredCreation, draftOwner, sessionId, sessionReady]);

  const send = useCallback(async (rawPrompt?: string) => {
    const raw = String(rawPrompt ?? draft).trim();
    const prompt = raw || (uploadDrafts.length ? tx(locale, "请读取附件，完成分析并交付验证后的结果文件。", "Read the attachments, complete the analysis, and deliver verified result files.") : "");
    if (!prompt || running || historyLoading || restorationError || sendBusyRef.current) return;
    sendBusyRef.current = true;
    try {
      const previous = await recoverCreation<AssistantRun>(sessionId, draftOwner);
      if (previous && previous.id === currentRun?.id) finishCreation(sessionId, draftOwner);
      else if (previous) {
        if (requestedRunRef.current) throw new Error(tx(locale, "另一个请求尚待恢复，请先开始新实验或恢复该请求。", "Another request needs reconciliation. Start a new experiment or recover it first."));
        adoptRecoveredCreation(previous); sendBusyRef.current = false; return;
      }
    } catch (error) {
      setActivities((current) => [...current, { seq: Date.now(), type: "request_recovery_pending", status: "blocked", label: error instanceof Error ? error.message : "run_request_recovery_unavailable" }]);
      sendBusyRef.current = false;
      return;
    }
    let secretRefs: string[] = [];
    try {
      secretRefs = await controlledCredentialsRef.current?.sealPending() ?? [];
    } catch (error) {
      controlledCredentialsRef.current?.open();
      setActivities((current) => [...current, {
        seq: Date.now(), type: "controlled_secret_failed", status: "blocked",
        label: error instanceof Error ? error.message : tx(locale, "受控凭据加密失败。", "Controlled credential encryption failed."),
      }].slice(-30));
      sendBusyRef.current = false;
      return;
    }
    if (
      secretRefs.length === 1
      && controlledCredentialsRef.current?.sealedPurpose() === "weather4cast_sftp"
      && isExactPendingWeatherAcceleration(currentRun)
    ) {
      setCredentialAttachBusy(true);
      try {
        const response = await fetch(`/api/assistant/runs/${encodeURIComponent(currentRun!.id)}/actions`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action: "attach_secret_refs", secret_refs: secretRefs }),
          cache: "no-store",
        });
        if (!response.ok) throw new Error(await apiError(response));
        const run = await response.json() as AssistantRun;
        if (!isConfirmedControlledSecretBinding(run, currentRun!.id, "weather4cast_sftp", 1)) {
          throw new Error("controlled_secret_binding_not_confirmed");
        }
        controlledCredentialsRef.current?.commitSealed(secretRefs);
        setDraft("");
        applySnapshot(run, `assistant_${run.id}`);
        setActivities((current) => [...current, {
          seq: Date.now(), type: "controlled_secret_attached", status: "completed",
          label: tx(locale, "凭据已加密绑定到当前待审批 Run；审批尚未执行。", "Credential bound to the current approval Run; the approval remains unexecuted."),
        }].slice(-30));
      } catch (error) {
        controlledCredentialsRef.current?.open();
        setActivities((current) => [...current, {
          seq: Date.now(), type: "controlled_secret_attach_failed", status: "blocked",
          label: error instanceof Error ? error.message : "controlled_secret_attach_failed",
        }].slice(-30));
      } finally {
        setCredentialAttachBusy(false);
        sendBusyRef.current = false;
      }
      return;
    }
    lastPromptRef.current = prompt;
    setActivities([]);
    setActivityOpen(false);
    setRoute("agent");
    setRouteLabel(tx(locale, "正在创建持久任务", "Creating durable run"));
    setRunning(true);
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    const assistantId = makeId("assistant");
    const resumeRequested = continueRunRef.current;
    continueRunRef.current = false;
    const resumableRun = resumeRequested && currentRun && ["blocked", "failed", "cancelled"].includes(currentRun.status) && uploadDrafts.length === 0 ? currentRun : null;
    const retainedAttachments = uploadDrafts.length || !reuseRetainedAttachments ? [] : (currentRun?.attachments ?? []);
    const attachmentNames = uploadDrafts.length
      ? uploadDrafts.map((item) => item.file.name)
      : retainedAttachments.map((item) => item.name);
    setMessages((current) => [...current,
      { id: makeId("user"), role: "user", content: prompt, status: "complete", attachmentNames, runId: resumableRun?.id },
      { id: assistantId, role: "assistant", content: "", status: "streaming", runId: resumableRun?.id },
    ]);
    try {
      const attachmentIds: string[] = [];
      for (const item of uploadDrafts) attachmentIds.push(await uploadOne(item, controller.signal));
      if (!uploadDrafts.length) attachmentIds.push(...retainedAttachments.map((item) => item.id));
      const resumeBody = { action: "resume", instruction: prompt };
      const creationKey = resumableRun ? "" : ensureCreationKey(sessionId, draftOwner);
      const requestOptions: RequestInit = {
        method: "POST",
        headers: { "Content-Type": "application/json", ...(creationKey ? { "Idempotency-Key": creationKey } : {}) },
        body: JSON.stringify(resumableRun
          ? { ...resumeBody, secret_refs: secretRefs }
          : { prompt, conversation_id: sessionId, attachment_ids: attachmentIds, secret_refs: secretRefs, idempotency_key: creationKey, ...((currentRun ? currentRun.project_id : selectedProjectId) ? { project_id: currentRun ? currentRun.project_id : selectedProjectId } : {}) }),
        cache: "no-store",
        signal: controller.signal,
      };
      const response = resumableRun
        ? await fetch(`/api/assistant/runs/${encodeURIComponent(resumableRun.id)}/actions`, requestOptions)
        : await fetch("/api/assistant/runs", requestOptions);
      if (!response.ok) throw new Error(await apiError(response));
      const run = await response.json() as AssistantRun;
      requestedRunRef.current = run.id;
      setDraft("");
      if (secretRefs.length) controlledCredentialsRef.current?.commitSealed(secretRefs);
      setUploadDrafts([]);
      setReuseRetainedAttachments(false);
      uploadPromisesRef.current.clear();
      uploadControllersRef.current.clear();
      const runAttachmentNames = run.attachments?.map((item) => item.name) ?? attachmentNames;
      setMessages((current) => current.map((item) => item.id === assistantId || (item.role === "user" && !item.runId && item.content === prompt) ? {
        ...item,
        runId: run.id,
        ...(item.role === "user" ? { attachmentNames: runAttachmentNames } : {}),
      } : item));
      applySnapshot(run, assistantId);
      if (!resumableRun) finishCreation(sessionId, draftOwner);
      appendActivity("run_created", { run_id: run.id, status: run.status });
      void connectRun(run.id, assistantId, controller, resumableRun ? (run.last_event_seq ?? 0) : 0);
    } catch (error) {
      if (controller.signal.aborted) {
        setRunning(false);
        setMessages((current) => current.map((item) => item.id === assistantId ? { ...item, status: "blocked", content: tx(locale, "连接已中断，请求结果仍待恢复；不会自动创建替代任务。", "Connection interrupted; recover the existing request before creating another Run.") } : item));
        return;
      }
      const message = error instanceof Error ? error.message : tx(locale, "任务创建失败。", "Run creation failed.");
      setRunning(false);
      setMessages((current) => current.map((item) => item.id === assistantId ? { ...item, content: message, status: "blocked" } : item));
      setActivities((current) => [...current, { seq: Date.now(), type: "run_create_failed", status: "blocked", label: message }]);
    } finally {
      sendBusyRef.current = false;
    }
  }, [adoptRecoveredCreation, apiError, appendActivity, applySnapshot, connectRun, currentRun, draft, draftOwner, historyLoading, restorationError, locale, reuseRetainedAttachments, running, selectedTask, selectedProjectId, sessionId, uploadDrafts, uploadOne]);

  const runAction = useCallback(async (action: "cancel" | "retry" | "resume" | "pause") => {
    if (!currentRun) return;
    if (action === "cancel") abortRef.current?.abort();
    // A retry/resume may follow a one-shot credential failure. Ask the
   // browser-memory panel for a newly sealed opaque reference when the user
   // has re-entered credentials; never reuse a consumed reference.
    let secretRefs: string[] = [];
    if (action === "retry" || action === "resume") {
      try {
        secretRefs = await controlledCredentialsRef.current?.sealPending() ?? [];
      } catch (error) {
        setActivities((current) => [...current, { seq: Date.now(), type: "controlled_secret_failed", status: "blocked", label: error instanceof Error ? error.message : tx(locale, "受控凭据加密失败。", "Controlled credential encryption failed.") }].slice(-30));
        controlledCredentialsRef.current?.open();
        return;
      }
    }
   if ((action === "retry" || action === "resume") && currentRun.error_class === "controlled_secret_required" && !secretRefs.length) {
      controlledCredentialsRef.current?.open();
      setActivities((current) => [...current, { seq: Date.now(), type: "controlled_secret_required", status: "blocked", label: tx(locale, "请在受控下载凭据面板中重新加密后再继续。", "Re-encrypt a fresh controlled credential before resuming this Run.") }].slice(-30));
      return;
    }
    const response = await fetch(`/api/assistant/runs/${encodeURIComponent(currentRun.id)}/actions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, secret_refs: secretRefs }),
      cache: "no-store",
    });
    if (!response.ok) {
      const label = await apiError(response);
      setActivities((current) => [...current, { seq: Date.now(), type: "action_failed", status: "error", label }].slice(-30));
      return;
    }
    const run = await response.json() as AssistantRun;
    if (activeRunIdRef.current !== run.id) return;
    if (secretRefs.length) controlledCredentialsRef.current?.commitSealed(secretRefs);
    const assistantId = action === "cancel" || action === "pause" ? latestAssistantMessageId(messages, run.id) : makeId("assistant");
    if (action !== "cancel" && action !== "pause") setMessages((current) => [...current, { id: assistantId!, role: "assistant", content: "", status: "streaming", runId: run.id }]);
    applySnapshot(run, assistantId);
    if (action === "cancel" || run.status === "paused") {
      setRunning(false);
      return;
    }
    setRunning(true);
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    void connectRun(run.id, assistantId!, controller, run.last_event_seq ?? 0);
  }, [apiError, applySnapshot, connectRun, currentRun, locale, messages]);

  const attachCredentialsToPendingRun = useCallback(async () => {
    if (!credentialRunId || credentialAttachBusy) return;
    setCredentialAttachBusy(true);
    try {
      const secretRefs = await controlledCredentialsRef.current?.sealPending() ?? [];
      if (secretRefs.length !== 1) throw new Error("controlled_credentials_not_ready");
      const response = await fetch(`/api/assistant/runs/${encodeURIComponent(credentialRunId)}/actions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "attach_secret_refs", secret_refs: secretRefs }),
        cache: "no-store",
      });
      if (!response.ok) throw new Error(await apiError(response));
      const run = await response.json() as AssistantRun;
      if (!isConfirmedControlledSecretBinding(run, credentialRunId, "weather4cast_sftp", 1)) {
        throw new Error("credential_run_approval_state_changed");
      }
      controlledCredentialsRef.current?.commitSealed(secretRefs);
      applySnapshot(run, `assistant_${run.id}`);
      setCredentialRunId("");
      const url = new URL(window.location.href);
      url.searchParams.delete("credential_run");
      window.history.replaceState(null, "", url);
      setActivities((current) => [...current, {
        seq: Date.now(), type: "controlled_secret_attached", status: "completed",
        label: tx(locale, "凭据已加密附加到待审批 Run。", "Credential sealed and attached to the approval Run."),
      }].slice(-30));
    } catch (error) {
      controlledCredentialsRef.current?.open();
      setActivities((current) => [...current, {
        seq: Date.now(), type: "controlled_secret_attach_failed", status: "blocked",
        label: error instanceof Error ? error.message : "controlled_secret_attach_failed",
      }].slice(-30));
    } finally {
      setCredentialAttachBusy(false);
    }
  }, [apiError, applySnapshot, credentialAttachBusy, credentialRunId, locale]);

  const decideApproval = useCallback(async (approval: AssistantApproval, approved: boolean) => {
    if (approval.status !== "pending" || approvalBusyRef.current) return;
    approvalBusyRef.current = approval.id;
    setApprovalBusyId(approval.id);
    try {
      const response = await fetch(`/api/assistant/approvals/${encodeURIComponent(approval.id)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ approved, note: approved ? "Approved in DeepEvo assistant UI" : "Rejected in DeepEvo assistant UI" }),
        cache: "no-store",
      });
      if (!response.ok) {
        const label = await apiError(response);
        setActivities((current) => [...current, { seq: Date.now(), type: "approval_failed", status: "error", label }].slice(-30));
        return;
      }
      const payload = await response.json() as { run?: AssistantRun; approval?: AssistantApproval };
      if (payload.run) {
        if (activeRunIdRef.current !== payload.run.id) return;
        const assistantId = latestAssistantMessageId(messages, payload.run.id) ?? `assistant_${payload.run.id}`;
        applySnapshot(payload.run, assistantId);
        if (payload.approval?.status === "approved" && isActiveStatus(payload.run.status)) {
          setActivities((current) => [...current, { seq: Date.now(), type: "approval_decided", status: "completed", label: tx(locale, "审批已确认，任务继续执行", "Approval confirmed; the Run is continuing") }].slice(-30));
          setRunning(true);
          abortRef.current?.abort();
          const controller = new AbortController();
          abortRef.current = controller;
          void connectRun(payload.run.id, assistantId, controller, payload.run.last_event_seq ?? 0);
        } else if (payload.approval?.status === "approved") {
          setRunning(false);
          setActivities((current) => [...current, { seq: Date.now(), type: "approval_decided", status: "completed", label: tx(locale, "审批已记录，操作状态已更新；研究结论仍以独立验收为准。", "Approval is recorded and the operation state is updated; research claims still require independent acceptance.") }].slice(-30));
        } else {
          setRunning(false);
          const label = payload.run.error_message || tx(locale, "审批未生效，请按当前状态继续。", "The approval was not applied; continue from the current Run state.");
          setActivities((current) => [...current, { seq: Date.now(), type: "approval_not_applied", status: "blocked", label }].slice(-30));
        }
      }
    } catch (error) {
      const label = error instanceof Error ? error.message : tx(locale, "审批请求失败。", "Approval request failed.");
      setActivities((current) => [...current, { seq: Date.now(), type: "approval_failed", status: "error", label }].slice(-30));
    } finally {
      approvalBusyRef.current = "";
      setApprovalBusyId("");
    }
  }, [apiError, applySnapshot, connectRun, isActiveStatus, locale, messages]);

  useEffect(() => {
    if (!autoApproveSafeActions || !currentRun || approvalBusyRef.current) return;
    const approval = (currentRun.approvals ?? []).find((item) => isSafeSameRunApproval(item, currentRun.id));
    if (!approval) return;
    setActivities((current) => [...current, {
      seq: Date.now(), type: "approval_auto_deciding", status: "running",
      label: tx(locale, "正在自动批准同 Run 可逆操作", "Auto-approving a reversible same-Run action"),
    }].slice(-30));
    void decideApproval(approval, true);
  }, [autoApproveSafeActions, currentRun, decideApproval, locale]);

  const stop = useCallback(() => {
    for (const controller of uploadControllersRef.current.values()) controller.abort();
    uploadControllersRef.current.clear();
    uploadPromisesRef.current.clear();
    if (currentRun && isActiveStatus(currentRun.status)) void runAction("cancel");
    else {
      abortRef.current?.abort();
      abortRef.current = null;
      setRunning(false);
      setMessages((current) => current.map((item) => item.status === "streaming" ? { ...item, status: "stopped" } : item));
      setActivities((current) => [...current, { seq: Date.now(), type: "stopped", status: "stopped", label: tx(locale, "已停止当前上传或任务连接", "Stopped the current upload or run connection") }].slice(-30));
    }
  }, [currentRun, isActiveStatus, locale, runAction]);

  const newSession = useCallback(() => {
    activeRunIdRef.current = null;
    runEventCursorRef.current.clear();
    requestedRunRef.current = null;
    setRestorationError("");
    abortRef.current?.abort();
    abortRef.current = null;
    for (const controller of uploadControllersRef.current.values()) controller.abort();
    uploadControllersRef.current.clear();
    uploadPromisesRef.current.clear();
    setMessages([]);
    setActivities([]);
    setUploadDrafts([]);
    setReuseRetainedAttachments(false);
    setDismissedAttachmentRunId("");
    setDraft("");
    setCurrentRun(null);
    setContextState(null);
    lastPromptRef.current = "";
    onNewExperiment?.();
    continueRunRef.current = false;
    setRunDetailsOpen(false);
    setRunning(false);
    const next = makeId("chat");
    setSessionId(next);
    const nextUrl = new URL(window.location.href);
    nextUrl.searchParams.set("conversation", next);
    nextUrl.searchParams.delete("assistant_run");
    nextUrl.searchParams.delete("report_run");
    window.history.replaceState(null, "", nextUrl);
    setRoute("agent");
    setRouteLabel(tx(locale, "持久任务 Agent", "Durable task agent"));
    window.setTimeout(() => textareaRef.current?.focus(), 0);
  }, [locale, onNewExperiment]);

  const retry = useCallback(() => {
    if (currentRun && currentRun.status !== "completed") {
      void runAction(currentRun.status === "waiting_approval" ? "resume" : "retry");
      return;
    }
    const prompt = lastPromptRef.current || [...messages].reverse().find((item) => item.role === "user")?.content || "";
    if (prompt) void send(prompt);
  }, [currentRun, messages, runAction, send]);

  const openPreview = useCallback(async (artifact: AssistantArtifact) => {
    setPreview({ artifact, loading: true });
    if (["image", "pdf", "html", "svg"].includes(artifact.preview_kind)) {
      setPreview({ artifact, loading: false });
      return;
    }
    try {
      if (artifact.preview_kind === "archive") {
        const response = await fetch(`${artifact.preview_url}?preview=1`, { cache: "no-store" });
        if (!response.ok) throw new Error(await apiError(response));
        const payload = await response.json() as { preview?: Array<{ name: string; bytes: number }> };
        setPreview({ artifact, entries: payload.preview ?? [], loading: false });
      } else if (["text", "markdown", "structured_text", "table"].includes(artifact.preview_kind)) {
        const response = await fetch(artifact.preview_url, { cache: "no-store" });
        if (!response.ok) throw new Error(await apiError(response));
        setPreview({ artifact, content: (await response.text()).slice(0, 131_072), loading: false });
      } else {
        setPreview({ artifact, loading: false });
      }
    } catch (error) {
      setPreview({ artifact, loading: false, error: error instanceof Error ? error.message : "preview_failed" });
    }
  }, [apiError]);

  const pendingApprovals = useMemo(() => (currentRun?.approvals ?? []).filter((item) => item.status === "pending"), [currentRun]);
  const actualModel = [currentRun?.model_provider, currentRun?.model].filter(Boolean).join(" / ");

  useEffect(() => {
    const controller = new AbortController();
    void fetch("/api/session/status", { cache: "no-store", signal: controller.signal }).then(async (response) => {
      if (response.status === 401) { setLoginRequired(true); return; }
      if (!response.ok) return;
      const identity = await response.json() as { tenant_id?: string; username?: string };
      if (!identity.tenant_id) return;
      const key = `evomind.draft.v1:${identity.tenant_id}:${identity.username ?? ""}:${sessionId}`;
      setDraftOwner(key);
      try { const saved = window.sessionStorage.getItem(key); if (saved) setDraft((current) => current || saved); } catch { /* memory-only draft remains available */ }
    }).catch(() => undefined);
    return () => controller.abort();
  }, [sessionId]);

  useEffect(() => { if (draftOwner) { try { if (draft) window.sessionStorage.setItem(draftOwner, draft); else window.sessionStorage.removeItem(draftOwner); } catch { /* restricted browser storage */ } } }, [draft, draftOwner]);

  useEffect(() => {
    if (!currentRun || !isActiveStatus(currentRun.status) || loginRequired) return;
    const id = currentRun.id;
    const controller = new AbortController();
    const sample = async () => {
      if (document.visibilityState !== "visible") return;
      try {
        const response = await fetch(`/api/assistant/runs/${encodeURIComponent(id)}`, { cache: "no-store", signal: controller.signal });
        if (response.status === 401) { setLoginRequired(true); return; }
        if (!response.ok) return;
        const latest = await response.json() as AssistantRun;
        if (controller.signal.aborted) return;
        setCurrentRun((old) => old?.id === id && (latest.last_event_seq ?? 0) >= (old.last_event_seq ?? 0) ? { ...old, status: latest.status, terminal: latest.terminal, progress: latest.progress, last_event_seq: latest.last_event_seq } : old);
      } catch { /* freshness panel marks stale evidence; never trigger a retry */ }
    };
    const timer = window.setInterval(() => void sample(), 15_000);
    document.addEventListener("visibilitychange", sample);
    return () => { controller.abort(); window.clearInterval(timer); document.removeEventListener("visibilitychange", sample); };
  }, [currentRun?.id, currentRun?.status, isActiveStatus, loginRequired]);

  useEffect(() => { if (currentRun && !isActiveStatus(currentRun.status)) setRunning(false); }, [currentRun?.status, isActiveStatus]);

  useEffect(() => {
    const id = currentRun ? currentRun.project_id : selectedProjectId;
    setProjectName("");
    if (!id) return;
    const controller = new AbortController();
    void fetch(`/api/assistant/projects/${encodeURIComponent(id)}`, { cache: "no-store", signal: controller.signal }).then(async (response) => {
      if (!response.ok) throw new Error("project_unavailable");
      const payload = await response.json() as { project: { name: string } };
      if (!controller.signal.aborted) setProjectName(payload.project.name);
    }).catch(() => { if (!controller.signal.aborted) setRestorationError(tx(locale, "所选项目不可用，请从项目与实验重新选择。", "The selected project is unavailable. Select it again from Projects.")); });
    return () => controller.abort();
  }, [currentRun?.id, currentRun?.project_id, selectedProjectId, locale]);

  return (
    <section className="relative flex h-full max-h-full min-h-0 flex-col overflow-hidden bg-gradient-to-b from-surface-raised/70 via-surface/80 to-surface" aria-label={tx(locale, "DeepEvo 智能助手", "DeepEvo Assistant")} data-ui-assistant-screen>
      {restorationError ? <div role="alert" className="shrink-0 border-b border-warning/40 bg-warning-light p-3 text-sm text-warning-text">{restorationError}<a className="ml-2 underline" href="/?page=projects">{tx(locale, "项目与实验", "Projects")}</a></div> : null}
      {loginRequired ? <div role="alert" className="flex shrink-0 items-center justify-between gap-2 border-b border-warning/40 bg-warning-light p-3 text-xs text-warning-text"><span>{tx(locale, "登录已过期，后台任务不会因此重新启动。草稿已保留。", "Session expired. Background work is not restarted; your draft is retained.")}</span><a className="shrink-0 underline" href={`/login?next=${encodeURIComponent(typeof window === "undefined" ? "/?page=assistant" : window.location.pathname + window.location.search)}`}>{tx(locale, "重新登录", "Sign in")}</a></div> : null}
      <div className="flex h-14 shrink-0 items-center border-b border-edge/80 bg-surface-raised/82 px-3 shadow-sm backdrop-blur-xl sm:px-5">
        <div className="flex min-w-0 items-center gap-2">
          <DeepEvoGlyph size={22} className="ring-1 ring-edge/80" />
          <div className="min-w-0">
            <h1 className="truncate text-sm font-semibold leading-4 text-ink">DeepEvo</h1>
            <p className="hidden truncate text-[10px] leading-4 text-ink-muted sm:block">{tx(locale, "可复现研究工作台", "Reproducible research workspace")}</p>
          </div>
          <span className={cn("rounded-full border px-2 py-0.5 text-[11px] font-medium", route === "agent" ? "border-success/45 bg-success-light text-success-text" : "border-accent-muted bg-accent-light text-accent-dark")}>{routeLabel}</span>
          {actualModel ? <span data-ui-assistant-model className="hidden rounded border border-edge bg-surface-sunken px-1.5 py-0.5 font-mono text-[11px] text-ink-secondary lg:inline">{actualModel}</span> : null}
          <span className="hidden truncate text-xs text-ink-muted md:inline">
            {contextState?.currentTask
              ? tx(locale, `当前任务 · ${contextState.taskLabel || "Run"}`, `Current task · ${contextState.taskLabel || "Run"}`)
              : projectName ? tx(locale, `项目 · ${projectName}`, `Project · ${projectName}`) : tx(locale, "新实验 · 尚未选择项目", "New experiment, no project selected")}
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
            <button type="button" data-ui-action="assistant_open_advanced_control" data-ui-skip-action="true" onClick={() => onOpenAdvanced?.(currentRun?.id)} className="hidden h-8 rounded-md border border-success/35 bg-success-light px-2.5 text-xs font-semibold text-success-text hover:border-success/55 sm:block">
              {tx(locale, "查看研究结果", "View research results")}
            </button>
          ) : null}
          <button type="button" data-ui-action="assistant_new_session" data-ui-skip-action="true" onClick={newSession} className="inline-flex h-9 items-center justify-center gap-1.5 rounded-lg border border-edge bg-surface-raised px-2.5 text-xs font-semibold text-ink-secondary shadow-sm transition-colors hover:border-accent/45 hover:bg-accent-light/35 hover:text-ink" title={tx(locale, "开始新实验", "Start a new experiment")} aria-label={tx(locale, "开始新实验", "Start a new experiment")}>
            <Plus className="h-4 w-4" />
            <span className="hidden sm:inline">{tx(locale, "新实验", "New experiment")}</span>
          </button>
        </div>
      </div>

      <div className="thin-scrollbar min-h-0 flex-1 overflow-y-auto" aria-live="polite" data-ui-assistant-messages>
        <div className="mx-auto grid min-h-full w-full max-w-6xl grid-cols-1 gap-6 px-4 py-6 sm:px-8 sm:py-8 xl:grid-cols-[minmax(0,1fr)_20rem] xl:gap-8" data-ui-assistant-workbench-grid>
          <main className="min-w-0" data-ui-assistant-conversation>
          {historyLoading ? (
            <div className="my-auto flex items-center justify-center gap-2 py-10 text-sm text-ink-muted"><Loader2 className="h-4 w-4 animate-spin" />{tx(locale, "正在恢复服务端会话", "Restoring server-side conversation")}</div>
          ) : messages.length === 0 ? (
            <div className="my-auto rounded-2xl border border-edge/70 bg-surface-raised/72 p-6 shadow-sm backdrop-blur sm:p-8">
              <DeepEvoLockup height={48} className="mb-5 rounded-lg shadow-sm" />
              <h2 className="text-2xl font-semibold text-ink">{tx(locale, "今天需要处理什么？", "What should we work on?")}</h2>
              <p className="mt-2 text-sm text-ink-muted">{tx(locale, "描述任务或添加数据，系统会先确认执行方案与资源，再开始工作。", "Describe a task or attach data. The system will establish the plan and resources before execution.")}</p>
              <div className="mt-6 grid gap-2 sm:grid-cols-2">
                {[
                  tx(locale, "帮我制定模型训练方案", "Help me plan model training"),
                  tx(locale, "分析我上传的数据", "Analyze the data I attached"),
                  tx(locale, "解释系统能完成哪些实验", "Explain the experiments this system supports"),
                  tx(locale, "查看我已有的实验与结果", "Show my existing experiments and results"),
                ].map((prompt) => (
                  <button key={prompt} type="button" data-ui-action="assistant_suggestion" data-ui-skip-action="true" onClick={() => void send(prompt)} className="min-h-12 rounded-md border border-edge bg-surface-raised px-3 py-2 text-left text-sm text-ink-secondary hover:border-accent/50 hover:bg-accent-light/40">
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-7">
              {messages.map((message) => {
                const presentation = assistantMessagePresentation(message.content, message.role, locale);
                const products = (message.artifacts ?? []).filter((item) => !isAssistantExecutionRecord(item.name));
                const records = (message.artifacts ?? []).filter((item) => isAssistantExecutionRecord(item.name));
                return (
                <article key={message.id} className={cn("group flex gap-3", message.role === "user" && "justify-end")}>
                  {message.role === "assistant" ? (
                    <span className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-frame">
                      <DeepEvoGlyph size={22} animated={false} />
                    </span>
                  ) : null}
                  <div className={cn(
                    "min-w-0",
                    message.role === "user" ? "max-w-[88%] rounded-md bg-surface-sunken px-4 py-2.5" : "flex-1"
                  )}>
                    {message.attachmentNames?.length ? <div className="mb-2 flex flex-wrap gap-1.5">{message.attachmentNames.map((name) => <span key={name} className="inline-flex items-center gap-1 rounded border border-edge bg-surface-raised px-2 py-1 text-xs text-ink-secondary"><Paperclip className="h-3 w-3" />{name}</span>)}</div> : null}
                    {message.content ? <MessageBody content={presentation.text} /> : message.status === "streaming" ? (
                      <div className="flex h-7 items-center gap-1.5 text-sm text-ink-muted">
                        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />
                        {tx(locale, "任务处理中，执行进度见右侧", "Task in progress; see execution status")}
                      </div>
                    ) : message.status === "complete" ? (
                      <div className="text-sm text-ink-muted">
                        {tx(locale, "本次执行没有新增可显示的回复，请查看任务状态与产物。", "This execution has no new readable reply. Inspect the task status and artifacts.")}
                      </div>
                    ) : (
                      <div className="text-sm text-ink-muted">
                        {message.status === "stopped"
                          ? tx(locale, "已停止生成。", "Generation stopped.")
                          : message.status === "error"
                            ? tx(locale, "本轮回复中断，可以重试。", "This response was interrupted. Retry it.")
                            : message.status === "blocked"
                              ? tx(locale, "任务现场已保留，可从任务卡继续。", "Run state preserved; resume from the task card.")
                              : tx(locale, "本轮未生成内容。", "No content was generated.")}
                      </div>
                    )}
                    {presentation.rawDiagnostic ? <details className="mt-3 rounded border border-edge p-2 text-xs text-ink-muted"><summary className="cursor-pointer">{tx(locale, "原始技术记录", "Original diagnostic")}</summary><div className="mt-2"><MessageBody content={presentation.rawDiagnostic} /></div></details> : null}
                    {products.length ? <ArtifactCards artifacts={products} locale={locale} onPreview={openPreview} /> : null}
                    {records.length ? <details className="mt-3 rounded border border-edge p-2 text-xs text-ink-muted"><summary className="cursor-pointer">{tx(locale, "本次回复的运行记录", "Execution records for this reply")} ({records.length})</summary><ArtifactCards artifacts={records} locale={locale} onPreview={openPreview} /></details> : null}
                    {message.role === "assistant" && message.status !== "streaming" ? (
                      <div className="mt-2 flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
                        <button type="button" data-ui-action="assistant_copy_message" data-ui-skip-action="true" onClick={() => void navigator.clipboard.writeText(message.content)} className="flex h-7 w-7 items-center justify-center rounded text-ink-muted hover:bg-surface-sunken hover:text-ink" title={tx(locale, "复制", "Copy")} aria-label={tx(locale, "复制回复", "Copy response")}><Clipboard className="h-3.5 w-3.5" /></button>
                        {currentRun && message.id === latestAssistantMessageId(messages, currentRun.id) ? <button type="button" data-ui-action="assistant_retry_message" data-ui-skip-action="true" onClick={retry} disabled={running} className="flex h-7 w-7 items-center justify-center rounded text-ink-muted hover:bg-surface-sunken hover:text-ink disabled:opacity-40" title={tx(locale, "重试", "Retry")} aria-label={tx(locale, "重试回复", "Retry response")}><RotateCcw className="h-3.5 w-3.5" /></button> : null}
                        {message.status === "stopped" ? <span className="ml-1 text-xs text-ink-muted">{tx(locale, "已停止", "Stopped")}</span> : null}
                      </div>
                    ) : null}
                  </div>
                  {message.role === "user" ? (
                    <span className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-md border border-edge bg-surface-raised text-ink-secondary"><UserRound className="h-4 w-4" /></span>
                  ) : null}
                </article>
                );
              })}
            </div>
          )}

          {currentRun && ((currentRun.artifacts?.length ?? 0) + (currentRun.artifact_evidence?.length ?? 0) > 0) ? <details className="mt-6 rounded-lg border border-edge p-3 text-sm text-ink-secondary" data-ui-run-artifacts><summary className="cursor-pointer font-medium">{tx(locale, "本任务全部产物与记录", "All artifacts and records for this Run")}</summary><p className="mt-2 text-xs text-ink-muted">{tx(locale, "这里包含不同轮次的历史产物。日志、文件名或编排图不代表模型已通过验收。", "This includes artifacts from different attempts. Logs, filenames and orchestration graphs do not establish model acceptance.")}</p><ArtifactCards artifacts={mergeArtifacts(currentRun.artifact_evidence, currentRun.artifacts)} locale={locale} onPreview={openPreview} /></details> : null}

          {activities.length > 0 ? (
            <div className="mt-6 border-t border-edge pt-3">
              <button type="button" data-ui-action="assistant_toggle_activity" data-ui-skip-action="true" className="flex w-full items-center gap-2 text-left text-xs font-medium text-ink-muted hover:text-ink" onClick={() => setActivityOpen((value) => !value)} aria-expanded={activityOpen}>
                <ChevronDown className={cn("h-3.5 w-3.5 transition-transform", !activityOpen && "-rotate-90")} />
                {running ? tx(locale, "正在处理", "Working") : tx(locale, "处理记录", "Activity")}
              </button>
              {activityOpen ? (
                <ol className="mt-2 space-y-1.5 pl-5">
                  {activities.map((item) => (
                    <li key={item.identity ?? `${item.seq}-${item.type}`} className="flex gap-2 text-xs leading-5 text-ink-muted">
                      <span className={cn("mt-2 h-1.5 w-1.5 shrink-0 rounded-full", item.status === "completed" ? "bg-success" : item.status === "error" || item.status === "blocked" ? "bg-danger" : "bg-accent")} />
                      <span><span className="font-medium text-ink-secondary">{item.label}</span>{item.detail ? ` · ${item.detail}` : ""}</span>
                    </li>
                  ))}
                </ol>
              ) : null}
            </div>
          ) : null}
          <div ref={endRef} />
          </main>

          <aside className="self-start xl:sticky xl:top-6" aria-label={tx(locale, "实验运行检查器", "Experiment run inspector")} data-ui-assistant-inspector>
          {currentRun ? (
            <div className="rounded-xl border border-edge/80 bg-surface-raised/95 p-4 shadow-raised backdrop-blur" data-ui-assistant-run-drawer>
              <div data-ui-assistant-run-summary>
                <RunProgressPanel progress={currentRun.progress} status={currentRun.status} locale={locale} />
                <div className="mb-3 flex items-center justify-between gap-3 border-b border-edge/70 pb-3">
                  <div className="min-w-0">
                    <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-ink-muted">{tx(locale, "实验运行", "Experiment run")}</p>
                    <div className="mt-1 flex items-center gap-2">
                      <ShieldCheck className="h-4 w-4 text-accent" />
                      <span className="truncate text-sm font-semibold text-ink">{tx(locale, "当前任务", "Current task")}</span>
                    </div>
                  </div>
                  <span className={cn("rounded px-2 py-0.5 text-xs font-medium", currentRun.status === "completed" ? "bg-success-light text-success-text" : currentRun.status === "blocked" || currentRun.status === "waiting_approval" ? "bg-warning-light text-warning-text" : currentRun.status === "failed" ? "bg-danger-light text-danger-text" : "bg-accent-light text-accent-dark")}>{runStatusLabel(currentRun.status, locale)}</span>
                </div>
                <p className="text-xs leading-5 text-ink-muted">{runStatusSummary(currentRun, locale)}</p>
                {currentRun.status === "failed" || currentRun.status === "blocked" ? (
                  <div className="mt-3 rounded-lg border border-warning/35 bg-warning-light/45 p-3" data-ui-assistant-recovery-callout>
                    <p className="text-xs font-semibold text-warning-text">{tx(locale, "实验现场已保存", "Experiment state is saved")}</p>
                    <p className="mt-1 text-[11px] leading-4 text-ink-muted">{tx(locale, "失败不会覆盖已完成产物。补充指令后可在同一 Run 恢复，或从顶部开始新实验。", "Failure does not overwrite completed artifacts. Resume this Run with a correction, or start a new experiment from the header.")}</p>
                  </div>
                ) : null}
                <div className="mt-3 grid gap-2">
                    <label className="flex items-center justify-between gap-2 rounded-lg border border-edge bg-surface-sunken/55 px-2.5 py-2 text-[11px] text-ink-secondary" data-ui-assistant-safe-auto-approval>
                      <span>{tx(locale, "自动批准同 Run 可逆操作", "Auto-approve reversible same-Run actions")}</span>
                      <input
                        type="checkbox"
                        checked={autoApproveSafeActions}
                        onChange={(event) => {
                          const enabled = event.target.checked;
                          setAutoApproveSafeActions(enabled);
                          window.localStorage.setItem(SAFE_AUTO_APPROVAL_STORAGE_KEY, enabled ? "enabled" : "disabled");
                        }}
                      />
                    </label>
                    {isActiveStatus(currentRun.status) && currentRun.status !== "pausing" ? <button type="button" onClick={() => void runAction("pause")} className="rounded-lg border border-edge px-3 py-2 text-xs font-semibold text-ink-secondary hover:border-accent">{tx(locale, "安全暂停", "Pause safely")}</button> : null}
                    {currentRun.status === "paused" ? <button type="button" onClick={() => void runAction("resume")} className="rounded-lg border border-accent px-3 py-2 text-xs font-semibold text-accent">{tx(locale, "恢复当前任务", "Resume this task")}</button> : null}
                    {isActiveStatus(currentRun.status) ? <button type="button" onClick={() => void runAction("cancel")} className="rounded-lg border border-edge px-3 py-2 text-xs font-semibold text-ink-secondary hover:border-danger hover:text-danger">{tx(locale, "取消当前运行", "Cancel current run")}</button> : null}
                    {currentRun.error_class === "approval_expired" ? <button type="button" onClick={() => void runAction("resume")} className="inline-flex items-center justify-center gap-1 rounded-lg border border-accent/50 bg-accent-light px-3 py-2 text-xs font-semibold text-accent-dark"><RefreshCw className="h-3 w-3" />{tx(locale, "重新确认", "Confirm again")}</button> : null}
                    {["blocked", "failed", "cancelled"].includes(currentRun.status) ? <button type="button" onClick={() => { continueRunRef.current = true; textareaRef.current?.focus(); }} className="inline-flex items-center justify-center gap-1 rounded-lg bg-accent px-3 py-2 text-xs font-semibold text-accent-fg shadow-sm hover:bg-accent-dark" title={tx(locale, "在输入框补充要求后发送，系统会在同一 Run 中继续", "Add an instruction in the composer and send it to resume the same Run")}><RefreshCw className="h-3 w-3" />{tx(locale, "继续任务", "Continue Run")}</button> : null}
                </div>
                <div className="mt-3 border-t border-edge/70 pt-3">
                  <button type="button" data-ui-action="assistant_toggle_run_details" data-ui-skip-action="true" onClick={() => setRunDetailsOpen((value) => !value)} aria-expanded={runDetailsOpen} className="inline-flex w-full items-center justify-between gap-2 text-left text-[11px] font-medium text-ink-muted hover:text-ink-secondary">
                    <span>{runDetailsOpen ? tx(locale, "收起运行详情", "Hide run details") : tx(locale, "运行详情与错误", "Run details and errors")}</span>
                    <ChevronDown className={cn("h-3 w-3 transition-transform", runDetailsOpen && "rotate-180")} />
                  </button>
                </div>
              </div>
              {runDetailsOpen ? (
                <div className="mt-3 border-t border-edge pt-3" data-ui-assistant-run-technical-details>
                  <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-ink-muted">
                    <span>Run <code className="text-ink-secondary">{currentRun.id}</code></span>
                    {actualModel ? <span>{tx(locale, "模型", "Model")}: <code className="text-ink-secondary">{actualModel}</code></span> : null}
                  </div>
                  {currentRun.plan?.steps?.length ? <ol className="mt-3 grid gap-2 sm:grid-cols-2">{currentRun.plan.steps.map((step, index) => <li key={step.id ?? index} className="flex gap-2 rounded border border-edge/80 bg-surface-sunken/55 px-2.5 py-2 text-xs"><span className={cn("mt-1 h-2 w-2 shrink-0 rounded-full", step.status === "completed" ? "bg-success" : step.status === "waiting_approval" ? "bg-warning" : step.status === "running" ? "bg-accent" : "bg-edge-strong")} /><span><span className="font-medium text-ink-secondary">{step.label}</span>{step.detail ? <span className="block text-ink-muted">{step.detail}</span> : null}</span></li>)}</ol> : null}
                  {currentRun.error_message ? <div className="mt-3 rounded border border-edge bg-surface-sunken px-3 py-2 text-[11px] text-ink-muted"><span className="font-medium text-ink-secondary">{tx(locale, "技术信息", "Technical information")}:</span> {currentRun.error_message}</div> : null}
                </div>
              ) : null}
              {pendingApprovals.map((approval) => (
                <div key={approval.id} className="mt-3 rounded-md border border-warning/45 bg-warning-light/60 p-3" data-ui-assistant-approval-card>
                  <div className="flex items-center gap-2 text-sm font-semibold text-warning-text"><ShieldCheck className="h-4 w-4" />{tx(locale, "需要精确审批", "Exact approval required")}</div>
                  <dl className="mt-2 grid gap-1 text-xs text-ink-secondary sm:grid-cols-[7rem_1fr]"><dt>{tx(locale, "动作", "Action")}</dt><dd className="font-mono">{approval.tool_name}</dd><dt>{tx(locale, "风险", "Risk")}</dt><dd>{approval.risk_level}</dd><dt>{tx(locale, "可回滚", "Reversible")}</dt><dd>{approval.reversible ? tx(locale, "是", "Yes") : tx(locale, "否", "No")}</dd><dt>{tx(locale, "目标与参数", "Target and arguments")}</dt><dd className="break-all font-mono">{JSON.stringify(approval.normalized_arguments ?? {})}</dd><dt>{tx(locale, "影响范围", "Impact scope")}</dt><dd className="break-all font-mono">{JSON.stringify(approval.impact_scope ?? {})}</dd></dl>
                  <div className="mt-3 flex gap-2"><button type="button" onClick={() => void decideApproval(approval, true)} disabled={Boolean(approvalBusyId)} className="rounded bg-accent px-3 py-1.5 text-xs font-semibold text-accent-fg disabled:cursor-wait disabled:opacity-50">{approvalBusyId === approval.id ? tx(locale, "处理中…", "Processing…") : tx(locale, "批准一次", "Approve once")}</button><button type="button" onClick={() => void decideApproval(approval, false)} disabled={Boolean(approvalBusyId)} className="rounded border border-edge px-3 py-1.5 text-xs font-semibold text-ink-secondary disabled:cursor-wait disabled:opacity-50">{tx(locale, "拒绝", "Reject")}</button></div>
                </div>
              ))}
            </div>
          ) : (
            <div className="rounded-xl border border-edge/80 bg-surface-raised/88 p-4 shadow-sm">
              <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-ink-muted">{tx(locale, "实验上下文", "Experiment context")}</p>
              <p className="mt-2 text-sm font-semibold text-ink">{contextState?.taskLabel || tx(locale, "新研究任务", "New research task")}</p>
              <p className="mt-2 text-xs leading-5 text-ink-muted">{tx(locale, "发送任务后，这里会显示计划、审批、模型、错误和产物状态。", "Plans, approvals, models, errors, and artifacts appear here after you send a task.")}</p>
            </div>
          )}
          </aside>
        </div>
      </div>

      <div className="shrink-0 border-t border-edge/80 bg-surface-raised/88 px-3 pb-[max(12px,env(safe-area-inset-bottom))] pt-3 shadow-[0_-8px_28px_rgba(0,0,0,0.05)] backdrop-blur-xl sm:px-6">
        <div className="mx-auto max-w-3xl xl:max-w-6xl xl:pr-[22rem]">
          <AssistantControlledCredentials ref={controlledCredentialsRef} locale={locale} />
          {credentialRunId ? (
            <div className="mb-2 rounded-lg border border-accent/35 bg-accent-light/35 px-3 py-2" data-ui-assistant-credential-run-target>
              <button
                type="button"
                data-ui-action="assistant_attach_credentials_to_run"
                onClick={() => void attachCredentialsToPendingRun()}
                disabled={credentialAttachBusy}
                className="rounded bg-accent px-3 py-1.5 text-xs font-semibold text-accent-fg disabled:cursor-wait disabled:opacity-50"
              >
                {credentialAttachBusy ? tx(locale, "正在加密附加…", "Sealing and attaching…") : tx(locale, "加密并附加到待审批 Run", "Seal and attach to approval Run")}
              </button>
              <p className="mt-1 text-[10px] leading-4 text-ink-muted">{tx(locale, "只附加一次性加密引用；不会发送消息、创建新 Run、自动批准或启动训练。", "Attaches one encrypted reference only; it does not send a message, create a Run, approve, or start training.")}</p>
            </div>
          ) : null}
          {uploadDrafts.length ? (
            <div className="mb-2 flex flex-wrap gap-2" data-ui-assistant-upload-list aria-live="polite">
              {uploadDrafts.map((item) => {
                const statusText = uploadPhaseText(item, locale);
                return (
                  <div
                    key={item.id}
                    data-ui-assistant-upload-item
                    data-upload-status={item.status}
                    data-upload-progress={item.progress}
                    className={cn("min-w-[220px] max-w-full rounded-md border px-2.5 py-2 text-xs", item.status === "error" ? "border-danger/45 bg-danger-light" : item.status === "complete" ? "border-success/35 bg-success-light/45" : "border-edge bg-surface-sunken")}
                  >
                    <div className="flex items-center gap-2">
                      <FileText className={cn("h-3.5 w-3.5 shrink-0", item.status === "complete" ? "text-success" : "text-accent")} />
                      <span className="min-w-0 flex-1 truncate text-ink-secondary" title={item.file.name}>{item.file.name}</span>
                      {item.status === "error" ? (
                        <button type="button" onClick={() => void uploadOne(item).catch(() => undefined)} className="rounded px-1.5 py-1 text-[10px] font-medium text-accent hover:bg-accent-light" aria-label={tx(locale, `重试上传 ${item.file.name}`, `Retry upload ${item.file.name}`)}>{tx(locale, "重试", "Retry")}</button>
                      ) : null}
                      <button type="button" onClick={() => removeUploadDraft(item.id)} className="flex h-7 w-7 items-center justify-center rounded text-ink-muted hover:bg-surface-raised hover:text-danger" aria-label={item.status === "uploading" ? tx(locale, `取消上传 ${item.file.name}`, `Cancel upload ${item.file.name}`) : tx(locale, `移除附件 ${item.file.name}`, `Remove attachment ${item.file.name}`)}><X className="h-3.5 w-3.5" /></button>
                    </div>
                    <div className="mt-1 flex items-center justify-between gap-3 text-[10px] text-ink-muted">
                      <span className="shrink-0">{formatBytes(item.file.size)}{item.attachmentSha256 ? ` · ${item.attachmentSha256.slice(0, 8)}…` : ""}</span>
                      <span className="min-w-0 truncate text-right tabular-nums" title={statusText}>{statusText}</span>
                    </div>
                    <div
                      className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-edge"
                      role={item.status === "error" ? undefined : "progressbar"}
                      aria-label={tx(locale, `${item.file.name} 上传进度`, `${item.file.name} upload progress`)}
                      aria-valuemin={item.status === "error" ? undefined : 0}
                      aria-valuemax={item.status === "error" ? undefined : 100}
                      aria-valuenow={item.status === "error" ? undefined : item.progress}
                      aria-valuetext={item.status === "error" ? undefined : statusText}
                      data-ui-assistant-upload-progress
                    >
                      <div className={cn("h-full rounded-full transition-[width] duration-200 motion-reduce:transition-none", item.status === "complete" ? "bg-success" : item.status === "error" ? "bg-danger" : "bg-accent")} style={{ width: `${item.status === "error" ? Math.max(item.progress, 4) : item.progress}%` }} />
                    </div>
                  </div>
                );
              })}
            </div>
          ) : null}
          {!uploadDrafts.length && currentRun?.attachments?.length && dismissedAttachmentRunId !== currentRun.id ? (
            <div className="mb-2 flex flex-wrap items-center gap-2 rounded-md border border-edge bg-surface-sunken/70 px-2.5 py-2 text-xs text-ink-secondary" data-ui-assistant-retained-attachments data-attachment-reuse={reuseRetainedAttachments ? "enabled" : "disabled"}>
              <span className={cn("font-medium", reuseRetainedAttachments ? "text-success-text" : "text-ink-secondary")}>
                {reuseRetainedAttachments
                  ? tx(locale, "下一条消息将继续使用", "Will be reused in the next message")
                  : tx(locale, "上一轮附件（默认不再发送）", "Previous-run attachments (not sent by default)")}
              </span>
              {currentRun.attachments.map((item) => (
                <span key={item.id} className="inline-flex min-w-0 max-w-full items-center gap-1 rounded border border-edge bg-surface-raised px-2 py-1" title={`${item.name} · SHA-256 ${item.sha256}`}>
                  <FileText className="h-3 w-3 shrink-0 text-success" />
                  <span className="max-w-[280px] truncate">{item.name}</span>
                  <span className="text-[10px] text-ink-muted">{formatBytes(item.bytes)}</span>
                </span>
              ))}
              <div className="ml-auto flex items-center gap-1.5">
                <button
                  type="button"
                  disabled={running}
                  onClick={() => setReuseRetainedAttachments((value) => !value)}
                  className="rounded border border-edge bg-surface-raised px-2 py-1 text-[10px] font-medium text-ink-secondary hover:border-accent hover:text-accent disabled:opacity-40"
                >
                  {reuseRetainedAttachments ? tx(locale, "取消复用", "Do not reuse") : tx(locale, "下条消息继续使用", "Reuse next message")}
                </button>
                <button
                  type="button"
                  disabled={running}
                  onClick={() => {
                    setReuseRetainedAttachments(false);
                    setDismissedAttachmentRunId(currentRun.id);
                  }}
                  className="inline-flex h-7 items-center gap-1 rounded px-2 text-[10px] font-medium text-ink-muted hover:bg-surface-raised hover:text-danger disabled:opacity-40"
                  aria-label={tx(locale, "清除上一轮附件上下文", "Clear previous-run attachment context")}
                >
                  <X className="h-3 w-3" />{tx(locale, "清除", "Clear")}
                </button>
              </div>
            </div>
          ) : null}
          <div
            className={cn("flex items-end gap-2 rounded-xl border border-edge bg-surface-raised/95 p-2 shadow-sm transition-[border-color,box-shadow,background-color] duration-200 hover:border-edge-strong focus-within:border-accent focus-within:shadow-raised focus-within:ring-2 focus-within:ring-accent/15", dragActive && "border-accent bg-accent-light/30")}
            onDragEnter={(event) => { event.preventDefault(); setDragActive(true); }}
            onDragOver={(event) => { event.preventDefault(); setDragActive(true); }}
            onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragActive(false); }}
            onDrop={(event) => { event.preventDefault(); setDragActive(false); queueFiles(event.dataTransfer.files); }}
            onPaste={(event) => { if (event.clipboardData.files.length) queueFiles(event.clipboardData.files); }}
            data-ui-assistant-dropzone
          >
            <input ref={fileInputRef} type="file" multiple className="hidden" onChange={(event) => { if (event.target.files) queueFiles(event.target.files); event.currentTarget.value = ""; }} />
            <button type="button" data-ui-action="assistant_attach" onClick={() => fileInputRef.current?.click()} disabled={running} className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-transparent text-ink-secondary transition-colors hover:border-edge hover:bg-surface-sunken hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/30 disabled:opacity-40" title={tx(locale, "添加附件", "Add attachments")} aria-label={tx(locale, "添加附件", "Add attachments")}><Paperclip className="h-4 w-4" /></button>
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
              className="max-h-40 min-h-10 flex-1 resize-none bg-transparent px-1 py-2 text-[15px] leading-6 text-ink outline-none placeholder:text-ink-muted/80"
              placeholder={dragActive ? tx(locale, "松开即可添加文件", "Drop files to attach") : tx(locale, "输入任务内容…", "Enter a task…")}
              aria-label={tx(locale, "消息输入", "Message input")}
              disabled={running}
            />
            {running ? (
              <button type="button" data-ui-action="assistant_stop" data-ui-skip-action="true" onClick={stop} className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-frame-light text-white transition-colors hover:bg-frame-lighter focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/30" title={tx(locale, "停止生成", "Stop")} aria-label={tx(locale, "停止生成", "Stop generation")}><Square className="h-3.5 w-3.5 fill-current" /></button>
            ) : (
              <button type="button" data-ui-action="assistant_send" data-ui-skip-action="true" onClick={() => void send()} disabled={!draft.trim() && uploadDrafts.length === 0} className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-accent text-accent-fg shadow-sm transition-[background-color,box-shadow,transform] hover:bg-accent-dark hover:shadow-raised active:translate-y-px focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/30 disabled:cursor-not-allowed disabled:bg-edge-strong disabled:shadow-none disabled:transform-none" title={tx(locale, "发送", "Send")} aria-label={tx(locale, "发送消息", "Send message")}><ArrowUp className="h-4 w-4" /></button>
            )}
          </div>
          <div className="mt-1.5 text-center text-xs text-ink-muted">{tx(locale, "Enter 发送 · Shift+Enter 换行", "Enter to send · Shift+Enter for a new line")}</div>
        </div>
      </div>
      {preview ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-frame/70 p-4" role="dialog" aria-modal="true" aria-label={tx(locale, "产物预览", "Artifact preview")} data-ui-assistant-artifact-preview>
          <div className="flex max-h-[92vh] w-full max-w-5xl flex-col overflow-hidden rounded-xl border border-edge bg-surface-raised shadow-2xl">
            <div className="flex items-center gap-2 border-b border-edge px-4 py-3"><FileText className="h-4 w-4 text-accent" /><div className="min-w-0 flex-1"><div className="truncate text-sm font-semibold text-ink">{preview.artifact.name}</div><div className="text-[11px] text-ink-muted">{formatBytes(preview.artifact.bytes)} · SHA-256 {preview.artifact.sha256}</div></div><a href={preview.artifact.download_url} download={preview.artifact.name} className="inline-flex items-center gap-1 rounded border border-edge px-2 py-1 text-xs text-ink-secondary"><Download className="h-3 w-3" />{tx(locale, "下载", "Download")}</a><button type="button" onClick={() => setPreview(null)} className="flex h-8 w-8 items-center justify-center rounded hover:bg-surface-sunken" aria-label={tx(locale, "关闭预览", "Close preview")}><X className="h-4 w-4" /></button></div>
            <div className="thin-scrollbar min-h-[320px] flex-1 overflow-auto bg-surface-sunken p-4">
              {preview.loading ? <div className="flex h-64 items-center justify-center gap-2 text-sm text-ink-muted"><Loader2 className="h-4 w-4 animate-spin" />{tx(locale, "正在验证并加载预览", "Loading verified preview")}</div> : preview.error ? <div className="rounded border border-danger/30 bg-danger-light p-3 text-sm text-danger-text">{preview.error}</div> : preview.artifact.preview_kind === "image" || preview.artifact.preview_kind === "svg" ? <img src={preview.artifact.preview_url} alt={preview.artifact.name} className="mx-auto max-h-[70vh] max-w-full object-contain" /> : preview.artifact.preview_kind === "pdf" ? <VerifiedReportDocument name={preview.artifact.name} url={preview.artifact.preview_url} mediaType={preview.artifact.media_type} previewKind="pdf" expectedBytes={preview.artifact.bytes} expectedSha256={preview.artifact.sha256} /> : preview.artifact.preview_kind === "html" ? <iframe src={preview.artifact.preview_url} title={preview.artifact.name} sandbox="" className="h-[72vh] w-full rounded border border-edge bg-white" /> : preview.entries ? <div className="overflow-hidden rounded border border-edge bg-surface-raised"><div className="border-b border-edge px-3 py-2 text-xs font-semibold text-ink-secondary">{tx(locale, "ZIP 安全文件清单", "Safe ZIP file list")}</div>{preview.entries.map((entry) => <div key={entry.name} className="flex justify-between gap-4 border-b border-edge/60 px-3 py-2 text-xs last:border-b-0"><code className="break-all text-ink-secondary">{entry.name}</code><span className="shrink-0 text-ink-muted">{formatBytes(entry.bytes)}</span></div>)}</div> : preview.content !== undefined ? <pre className="whitespace-pre-wrap break-words rounded border border-edge bg-surface-raised p-4 text-xs leading-6 text-ink">{preview.content}</pre> : <div className="flex h-64 flex-col items-center justify-center gap-3 text-sm text-ink-muted"><FileText className="h-9 w-9" /><span>{tx(locale, "此二进制产物提供元数据与原文件下载。", "This binary artifact provides verified metadata and original download.")}</span></div>}
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}
