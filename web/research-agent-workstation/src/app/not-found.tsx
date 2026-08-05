export default function NotFound() {
  return (
    <main className="min-h-screen bg-slate-950 px-6 py-16 text-slate-100">
      <div className="mx-auto max-w-2xl rounded-2xl border border-slate-800 bg-slate-900/80 p-8 shadow-2xl">
        <p className="text-sm font-semibold uppercase tracking-[0.3em] text-cyan-300">EvoMind</p>
        <h1 className="mt-4 text-3xl font-bold">页面未找到</h1>
        <p className="mt-3 text-sm leading-6 text-slate-300">
          请求的本地工作站页面不存在。请返回 EvoMind 控制台或重新打开当前任务链接。
        </p>
        <a
          className="mt-6 inline-flex rounded-full bg-cyan-400 px-5 py-2 text-sm font-semibold text-slate-950"
          href="/?page=assistant"
        >
          返回 EvoMind 控制台
        </a>
      </div>
    </main>
  );
}
