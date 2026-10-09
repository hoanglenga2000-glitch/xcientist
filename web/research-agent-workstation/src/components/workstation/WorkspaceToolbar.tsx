"use client";

import { useEffect, useRef, useState } from "react";
import { Bell, FileText, Search, UserRound, X } from "lucide-react";

type Run = { id: string; title?: string; status?: string; updated_at?: string };
type Panel = "search" | "activity" | "account";

/** Header controls use owner-scoped read APIs, never generic execution actions. */
export function WorkspaceToolbar({ locale = "zh-CN", onOpenEvidenceRail }: {
  locale?: "zh-CN" | "en-US"; onOpenEvidenceRail: () => void;
}) {
  const text = (zh: string, en: string) => locale === "zh-CN" ? zh : en;
  const [panel, setPanel] = useState<Panel | null>(null);
  const [query, setQuery] = useState("");
  const [runs, setRuns] = useState<Run[]>([]);
  const [username, setUsername] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const dialog = useRef<HTMLDialogElement>(null);
  const search = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const keydown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault(); setPanel("search");
      }
    };
    window.addEventListener("keydown", keydown);
    return () => window.removeEventListener("keydown", keydown);
  }, []);
  useEffect(() => {
    if (!panel) { dialog.current?.close(); return; }
    if (!dialog.current?.open) dialog.current?.showModal();
    if (panel === "search") search.current?.focus();
    const controller = new AbortController();
    setLoading(true); setError(""); setRuns([]); setUsername("");
    const endpoint = panel === "account" ? "/api/session/status" : "/api/assistant/projects";
    void fetch(endpoint, { cache: "no-store", signal: controller.signal }).then(async (response) => {
      if (!response.ok) throw new Error(response.status === 401 ? "session_expired" : "unavailable");
      const payload = await response.json();
      if (controller.signal.aborted) return;
      if (panel === "account") {
        if (typeof payload.username !== "string" || !/^[A-Za-z0-9._-]{1,64}$/.test(payload.username)) throw new Error("unavailable");
        setUsername(payload.username);
      } else {
        if (!Array.isArray(payload.runs)) throw new Error("unavailable");
        setRuns(payload.runs.filter((run: Run) => typeof run.id === "string" && /^run_[A-Za-z0-9_-]+$/.test(run.id)).slice(0,100));
      }
    }).catch((failure) => { if (!controller.signal.aborted) setError(failure.message === "session_expired" ? "session_expired" : "unavailable"); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [panel, revision]);

  const title = panel === "search" ? text("搜索任务", "Search tasks") : panel === "activity" ? text("最近任务动态", "Recent task activity") : text("当前账户", "Current account");
  const filtered = runs.filter((run) => `${run.id} ${run.title ?? ""}`.toLowerCase().includes(query.trim().toLowerCase()));
  const labels: Record<string, string> = { completed:text("执行结束", "Execution ended"), failed:text("失败", "Failed"), blocked:text("需要处理", "Needs attention"), cancelled:text("已取消", "Cancelled"), recovering:text("等待恢复核对", "Recovery reconciliation required"), running:text("记录状态：运行中", "Recorded: running"), waiting_approval:text("等待审批", "Awaiting approval") };
  return <div className="ml-auto flex min-w-0 flex-1 items-center justify-end gap-2" data-ui-skip-action="true">
    <button type="button" onClick={() => { setQuery(""); setPanel("search"); }} aria-label={text("搜索任务", "Search tasks")} className="mr-auto flex h-8 min-w-0 items-center gap-2 rounded-md border border-edge bg-surface-sunken px-3 text-sm text-ink-muted sm:w-64">
      <Search className="h-4 w-4 shrink-0"/><span className="hidden truncate sm:block">{text("搜索任务名称或 Run ID", "Task name or Run ID")}</span><kbd className="ml-auto hidden shrink-0 whitespace-nowrap text-xs sm:block">Ctrl/⌘ K</kbd>
    </button>
    <button type="button" onClick={onOpenEvidenceRail} className="hidden h-8 items-center gap-1 rounded-md border border-edge px-2 text-xs text-ink-secondary sm:flex" aria-label={text("打开证据轨", "Open evidence rail")}><FileText className="h-4 w-4"/>{text("证据", "Evidence")}</button>
    <button type="button" onClick={() => setPanel("activity")} aria-label={text("最近任务动态", "Recent task activity")} className="h-8 rounded-md border border-edge px-2 text-ink-secondary"><Bell className="h-4 w-4"/></button>
    <button type="button" onClick={() => setPanel("account")} aria-label={text("用户菜单", "User menu")} className="flex h-8 items-center gap-2 rounded-md border border-edge px-2 text-xs text-ink-secondary"><UserRound className="h-4 w-4"/><span className="hidden sm:inline">{text("账户", "Account")}</span></button>
    <dialog ref={dialog} onClose={() => setPanel(null)} aria-labelledby="workspace-toolbar-dialog-title" className="fixed inset-0 m-auto max-h-[80dvh] w-[min(40rem,calc(100vw-2rem))] overflow-y-auto rounded-xl border border-edge bg-surface-raised p-5 text-ink shadow-2xl backdrop:bg-black/60">
      <div className="mb-4 flex items-center justify-between gap-3"><h2 id="workspace-toolbar-dialog-title" className="text-lg font-semibold">{title}</h2><button type="button" onClick={() => dialog.current?.close()} aria-label={text("关闭", "Close")} className="rounded-md p-2 hover:bg-surface-sunken"><X className="h-4 w-4"/></button></div>
      {panel === "search" ? <input ref={search} value={query} onChange={(event) => setQuery(event.target.value)} aria-label={text("任务关键词", "Task keywords")} className="mb-3 h-10 w-full rounded-md border border-edge bg-surface px-3 text-sm outline-none focus:ring-2 focus:ring-accent"/> : null}
      {loading ? <p role="status">{text("正在加载…", "Loading…")}</p> : error ? <div role="alert"><p>{error === "session_expired" ? text("登录已过期，请重新登录。", "Your session expired. Sign in again.") : text("暂时无法加载，请重试。", "Could not load. Retry.")}</p><button type="button" className="mt-3 rounded border border-edge px-3 py-2" onClick={() => setRevision((value) => value+1)}>{text("重试", "Retry")}</button></div> : panel === "account" ? <div className="space-y-3"><p>{text("已登录账户", "Signed-in account")}：<strong>{username}</strong></p><p className="text-sm text-ink-muted">{text("权限由服务器核对，本界面不推断管理员身份。", "Permissions are checked by the server; this view does not infer an administrator role.")}</p><a href="/?page=settings&demo=user" className="inline-block rounded border border-edge px-3 py-2">{text("打开个人设置", "Open personal settings")}</a></div> : <div className="space-y-2">
        <p className="text-xs text-ink-muted">{text("仅展示当前账户最近 100 条任务记录；不表示 worker 正在运行。", "Up to 100 recent tasks owned by this account; recorded status does not prove a live worker.")}</p>
        {(panel === "search" ? filtered : runs.slice(0,20)).map((run) => <a key={run.id} href={`/?page=assistant&demo=user&assistant_run=${encodeURIComponent(run.id)}`} className="block rounded-md border border-edge p-3 hover:border-accent"><span className="block truncate text-sm font-medium">{run.title || run.id}</span><span className="mt-1 block text-xs text-ink-muted">{labels[run.status ?? ""] ?? text("状态待核对", "Status unconfirmed")} · {run.id}</span></a>)}
        {!(panel === "search" ? filtered : runs).length ? <p className="py-5 text-sm text-ink-muted">{text("没有匹配的任务。", "No matching tasks.")}</p> : null}
      </div>}
    </dialog>
  </div>;
}
