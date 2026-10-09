"use client";

import { ArrowRight, Eye, EyeOff, LockKeyhole, ShieldCheck } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { DeepEvoGlyph, DeepEvoLockup } from "@/components/workstation/DeepEvoBrand";

const CSRF_STORAGE_KEY = "evomind.local.csrf.v1";

type LoginResponse = {
  ok?: boolean;
  message?: string;
  csrf_token?: string;
  redirect_to?: string;
};

export default function LoginPage() {
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [ready, setReady] = useState(false);
  const passwordRef = useRef<HTMLInputElement>(null);
  useEffect(() => setReady(true), []);

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!username.trim() || !password) {
      setError("请输入账号和密码");
      passwordRef.current?.focus();
      return;
    }
    setBusy(true);
    setError("");
    try {
      const next = new URLSearchParams(window.location.search).get("next") ?? "/?page=assistant";
      const response = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: username.trim(), password, next }),
        credentials: "same-origin",
        cache: "no-store",
      });
      const payload = await response.json().catch(() => ({})) as LoginResponse;
      if (!response.ok || !payload.ok || !payload.csrf_token) throw new Error(payload.message || "登录失败，请检查账号和密码");
      window.sessionStorage.setItem(CSRF_STORAGE_KEY, payload.csrf_token);
      window.location.assign(payload.redirect_to || "/?page=assistant");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "登录失败，请稍后重试");
      passwordRef.current?.focus();
      passwordRef.current?.select();
      setBusy(false);
    }
  }

  return (
    <main className="relative flex min-h-screen items-center justify-center overflow-hidden bg-[#070b14] px-5 py-12 text-white">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_20%_15%,rgba(99,102,241,.22),transparent_38%),radial-gradient(circle_at_82%_72%,rgba(6,182,212,.15),transparent_34%)]" />
      <section className="relative grid w-full max-w-5xl overflow-hidden rounded-[30px] border border-white/10 bg-white/[0.045] shadow-[0_30px_100px_rgba(0,0,0,.55)] backdrop-blur-xl md:grid-cols-[1.1fr_.9fr]">
        <div className="hidden min-h-[610px] flex-col justify-between border-r border-white/10 p-12 md:flex">
          <DeepEvoLockup height={48} className="w-fit rounded-lg shadow-lg shadow-black/30" />
          <div>
            <p className="mb-5 text-xs font-medium uppercase tracking-[0.28em] text-indigo-200/80">Scientific Research OS</p>
            <h1 className="max-w-lg text-5xl font-semibold leading-[1.08] tracking-[-0.045em]">让研究任务从一句话走到可验证的真实产物。</h1>
            <p className="mt-6 max-w-md text-base leading-7 text-slate-300">统一调用数据、文献、代码、HPC、Kaggle、报告与自进化工具，并保留完整运行证据。</p>
          </div>
          <div className="flex items-center gap-2 text-sm text-slate-400"><ShieldCheck className="h-4 w-4 text-emerald-300" /> 12 小时加密会话 · 精确 CSRF 与来源校验</div>
        </div>

        <div className="flex min-h-[610px] flex-col justify-center p-8 sm:p-12">
          <div className="mb-9">
            <DeepEvoGlyph size={48} className="mb-5 ring-1 ring-white/10 md:hidden" />
            <p className="text-xs font-medium uppercase tracking-[0.24em] text-indigo-200/80">Private workspace</p>
            <h2 className="mt-3 text-3xl font-semibold tracking-[-0.035em]">登录 DeepEvo</h2>
            <p className="mt-3 text-sm leading-6 text-slate-400">继续你的 Agent 任务、训练记录和专业报告。</p>
          </div>

          <form className="space-y-5" method="post" action="/api/auth/login" onSubmit={submit} noValidate>
            <label className="block text-sm text-slate-300" htmlFor="username">
              账号
              <input id="username" name="username" type="text" autoComplete="username" value={username} onChange={(event) => { setUsername(event.target.value); setError(""); }} disabled={busy} className="mt-2 h-12 w-full rounded-xl border border-white/10 bg-black/20 px-4 text-white outline-none transition focus:border-indigo-300/60 focus:ring-4 focus:ring-indigo-400/10" />
            </label>
            <label className="block text-sm text-slate-300" htmlFor="password">
              密码
              <span className="relative mt-2 block">
                <LockKeyhole className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
                <input ref={passwordRef} id="password" name="password" type={showPassword ? "text" : "password"} autoComplete="current-password" value={password} onChange={(event) => { setPassword(event.target.value); setError(""); }} disabled={busy} className="h-12 w-full rounded-xl border border-white/10 bg-black/20 pl-11 pr-12 text-white outline-none transition focus:border-indigo-300/60 focus:ring-4 focus:ring-indigo-400/10" />
                <button type="button" onClick={() => setShowPassword((value) => !value)} disabled={busy} aria-label={showPassword ? "隐藏密码" : "显示密码"} className="absolute right-3 top-1/2 grid h-8 w-8 -translate-y-1/2 place-items-center rounded-lg text-slate-400 hover:bg-white/5 hover:text-white">{showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}</button>
              </span>
            </label>
            <div className="min-h-6 text-sm text-rose-300" aria-live="polite">{error ? <p role="alert">{error}</p> : null}</div>
            <button type="submit" disabled={busy || !ready} className="flex h-12 w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-indigo-500 to-cyan-500 font-medium text-white shadow-lg shadow-indigo-950/40 transition hover:brightness-110 disabled:cursor-wait disabled:opacity-60">
              {!ready ? "正在准备登录…" : busy ? "正在验证…" : "进入工作台"}
              {!busy ? <ArrowRight className="h-4 w-4" /> : null}
            </button>
            <noscript>登录需要启用 JavaScript；账号和密码不会通过网址提交。</noscript>
          </form>
          <p className="mt-8 text-center text-xs text-slate-500">DeepEvo v0.3.0 · Production Agent Workspace</p>
        </div>
      </section>
    </main>
  );
}
