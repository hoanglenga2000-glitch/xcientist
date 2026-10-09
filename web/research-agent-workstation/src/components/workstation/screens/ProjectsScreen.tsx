"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowUpRight, FolderPlus, FlaskConical, RefreshCw } from "lucide-react";

type Project = { id: string; name: string; created_at: string };
type Experiment = { id: string; title: string; status: string; project_id: string; created_at: string; updated_at: string; model?: string };
type Payload = { projects?: Project[]; project?: Project; runs?: Experiment[] };
type Locale = "zh-CN" | "en-US";

export function ProjectsScreen({ locale = "zh-CN" }: { locale?: Locale }) {
  const text = (zh: string, en: string) => locale === "zh-CN" ? zh : en;
  const [projects, setProjects] = useState<Project[]>([]);
  const [runs, setRuns] = useState<Experiment[]>([]);
  const [selected, setSelected] = useState(() => typeof window === "undefined" ? "" : new URL(window.location.href).searchParams.get("project") ?? "");
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const creation = useRef<{ key: string; name: string } | null>(null);
  const generation = useRef(0);

  const load = useCallback(async (projectId = "") => {
    const current = ++generation.current;
    setLoading(true); setError(""); setRuns([]);
    try {
      const response = await fetch(`/api/assistant/projects${projectId ? `/${encodeURIComponent(projectId)}` : ""}`, { cache: "no-store" });
      if (!response.ok) throw new Error(response.status === 401 ? "session_expired" : "projects_unavailable");
      const payload = await response.json() as Payload;
      if (current !== generation.current) return;
      if (projectId && payload.project?.id !== projectId) throw new Error("project_identity_mismatch");
      if (projectId) {
        const indexResponse = await fetch("/api/assistant/projects", { cache: "no-store" });
        if (!indexResponse.ok) throw new Error(indexResponse.status === 401 ? "session_expired" : "projects_unavailable");
        const index = await indexResponse.json() as Payload;
        if (current !== generation.current) return;
        setProjects(index.projects ?? []);
      }
      if (payload.projects) setProjects(payload.projects);
      if (payload.project) setProjects((old) => [payload.project!, ...old.filter((item) => item.id !== payload.project!.id)]);
      setRuns(payload.runs ?? []);
    } catch (failure) {
      if (current === generation.current) setError(failure instanceof Error ? failure.message : "projects_unavailable");
    } finally { if (current === generation.current) setLoading(false); }
  }, []);

  useEffect(() => { void load(selected); return () => { generation.current += 1; }; }, [load, selected]);
  useEffect(() => {
    const url = new URL(window.location.href);
    if (selected) url.searchParams.set("project", selected); else url.searchParams.delete("project");
    window.history.replaceState(null, "", url);
  }, [selected]);

  async function createProject() {
    const value = name.trim();
    if (!value || saving) return;
    if (creation.current && creation.current.name !== value) {
      setError("project_request_pending"); return;
    }
    creation.current ??= { name: value, key: `project_request_${crypto.randomUUID().replaceAll("-", "")}` };
    setSaving(true); setError("");
    try {
      const response = await fetch("/api/assistant/projects", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: value, idempotency_key: creation.current.key }),
      });
      if (!response.ok) throw new Error(response.status === 401 ? "session_expired" : "project_create_failed");
      const payload = await response.json() as { project: Project };
      creation.current = null; setName("");
      setProjects((old) => [payload.project, ...old.filter((item) => item.id !== payload.project.id)]);
      setSelected(payload.project.id);
    } catch (failure) { setError(failure instanceof Error ? failure.message : "project_create_failed"); }
    finally { setSaving(false); }
  }

  function startExperiment() {
    if (loading || error) return;
    const query = new URLSearchParams({ page: "assistant", demo: "user", conversation: `chat_${crypto.randomUUID().replaceAll("-", "")}` });
    if (selected) query.set("project", selected);
    window.location.assign(`/?${query}`);
  }

  const selectedProject = projects.find((project) => project.id === selected);
  const statuses: Record<string, string> = {
    completed: text("执行结束", "Execution ended"), blocked: text("需要处理", "Needs attention"),
    failed: text("执行失败", "Failed"), cancelled: text("已取消", "Cancelled"),
    running: text("执行中", "Running"), queued: text("排队中", "Queued"),
    planning: text("准备方案", "Planning"), verifying: text("核验中", "Verifying"),
    waiting_approval: text("等待确认", "Awaiting approval"), recovering: text("等待恢复核对", "Recovery reconciliation required"),
  };

  return (
    <section className="mx-auto max-w-6xl space-y-6 py-4 sm:py-7" aria-label={text("项目与实验", "Projects and experiments")}>
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div><p className="text-xs font-semibold uppercase tracking-[0.18em] text-accent">Research workspace</p><h1 className="mt-2 text-2xl font-semibold tracking-tight text-ink">{text("项目与实验", "Projects and experiments")}</h1><p className="mt-2 text-sm text-ink-muted">{text("这里展示你的真实项目和最近实验，不使用示例任务或演示分数。", "Your real projects and recent experiments, without demo tasks or scores.")}</p></div>
        <div className="flex gap-2"><button type="button" onClick={() => void load(selected)} className="inline-flex min-h-10 items-center gap-2 rounded-lg border border-edge px-3 text-sm text-ink-secondary"><RefreshCw className="h-4 w-4" />{text("刷新", "Refresh")}</button><button type="button" onClick={startExperiment} className="inline-flex min-h-10 items-center gap-2 rounded-lg bg-accent px-4 text-sm font-semibold text-frame"><FlaskConical className="h-4 w-4" />{text("开始实验", "Start experiment")}</button></div>
      </header>
      {error ? <div role="alert" className="rounded-lg border border-warning/40 bg-warning-light p-3 text-sm text-warning-text">{error === "session_expired" ? text("登录已过期，请重新登录。项目和任务不会因此重新创建。", "Sign in again. Existing projects and Runs are retained.") : error === "project_request_pending" ? text("上次创建结果尚待确认，请保留原名称重试，避免重复创建。", "Retry the original name to reconcile the pending creation.") : text("暂时无法完成请求，请重试。已有项目与实验已保留。", "The request could not be completed. Existing projects and experiments are retained.")}{error === "session_expired" ? <a className="ml-2 underline" href="/login?next=%2F%3Fpage%3Dprojects">{text("重新登录", "Sign in")}</a> : null}</div> : null}
      <div className="grid gap-5 lg:grid-cols-[18rem_minmax(0,1fr)]">
        <aside className="space-y-4">
          <form className="rounded-xl border border-edge bg-surface-raised/80 p-4" onSubmit={(event) => { event.preventDefault(); void createProject(); }}>
            <label htmlFor="new-project-name" className="text-sm font-semibold text-ink">{text("新建项目", "New project")}</label>
            <input id="new-project-name" value={name} onChange={(event) => setName(event.target.value)} maxLength={120} disabled={saving} placeholder={text("例如：实验室模型评测", "For example: Lab model evaluation")} className="mt-3 h-10 w-full rounded-lg border border-edge bg-surface px-3 text-sm text-ink outline-none focus:border-accent" />
            <button type="submit" disabled={saving || !name.trim()} className="mt-3 inline-flex min-h-10 w-full items-center justify-center gap-2 rounded-lg border border-accent/40 text-sm font-medium text-accent disabled:opacity-50"><FolderPlus className="h-4 w-4" />{saving ? text("正在保存", "Saving") : text("创建项目", "Create project")}</button>
            <p className="mt-2 text-xs leading-5 text-ink-muted">{text("创建项目不会启动模型或使用 GPU。", "Creating a project does not start a model or use the GPU.")}</p>
          </form>
          <nav aria-label={text("我的项目", "My projects")} className="space-y-1 rounded-xl border border-edge bg-surface-raised/60 p-2">
            <button type="button" onClick={() => setSelected("")} aria-current={!selected ? "page" : undefined} className={`min-h-10 w-full rounded-lg px-3 text-left text-sm ${!selected ? "bg-accent-light text-accent" : "text-ink-secondary"}`}>{text("最近实验", "Recent experiments")}</button>
            {projects.map((project) => <button type="button" key={project.id} onClick={() => setSelected(project.id)} aria-current={selected === project.id ? "page" : undefined} className={`min-h-10 w-full truncate rounded-lg px-3 text-left text-sm ${selected === project.id ? "bg-accent-light text-accent" : "text-ink-secondary"}`}>{project.name}</button>)}
            {!projects.length && !loading ? <p className="px-3 py-4 text-xs text-ink-muted">{text("还没有项目。历史实验仍可在右侧查看。", "No projects yet. Your earlier experiments remain available.")}</p> : null}
          </nav>
        </aside>
        <div className="min-w-0 rounded-xl border border-edge bg-surface-raised/70 p-4 sm:p-5">
          <div className="flex items-center justify-between gap-3"><h2 className="truncate text-base font-semibold text-ink">{selectedProject?.name ?? text("最近实验", "Recent experiments")}</h2><span className="text-xs text-ink-muted">{text("最多显示最近 100 条", "Up to 100 recent entries")}</span></div>
          <p className="mt-2 text-xs leading-5 text-ink-muted">{text("执行结束不等于模型验收通过。训练状态、文件与独立评测证据请进入具体实验查看。", "Execution completion is not model acceptance. Open an experiment for execution, artifacts, and evaluation evidence.")}</p>
          {loading ? <p role="status" className="py-12 text-center text-sm text-ink-muted">{text("正在加载你的实验", "Loading your experiments")}</p> : !runs.length ? <div className="py-14 text-center"><FlaskConical className="mx-auto h-8 w-8 text-accent/60" /><p className="mt-3 text-sm text-ink-secondary">{text("还没有实验，开始一个新任务吧。", "No experiments yet. Start a new task.")}</p></div> : <div className="mt-5 space-y-2">{runs.map((run) => <a key={run.id} href={`/?page=assistant&demo=user&assistant_run=${encodeURIComponent(run.id)}`} className="group flex items-center justify-between gap-3 rounded-lg border border-edge bg-surface/70 p-3 transition-colors hover:border-accent/50"><div className="min-w-0"><h3 className="truncate text-sm font-medium text-ink">{run.title}</h3><div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs text-ink-muted"><span>{statuses[run.status] ?? text("状态待确认", "Status unconfirmed")}</span><time dateTime={run.updated_at || run.created_at}>{new Date(run.updated_at || run.created_at).toLocaleString(locale)}</time>{run.model ? <span>{run.model}</span> : null}</div></div><ArrowUpRight className="h-4 w-4 shrink-0 text-ink-muted group-hover:text-accent" /></a>)}</div>}
        </div>
      </div>
    </section>
  );
}
