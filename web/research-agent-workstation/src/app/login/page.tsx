"use client";

import { Eye, EyeOff } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { DeepEvoMark } from "@/components/workstation/DeepEvoBrand";
import { LoginYeti, type YetiStatus } from "@/components/workstation/LoginYeti";
import { cn } from "@/lib/utils";

const CSRF_STORAGE_KEY = "evomind.local.csrf.v1";

type LoginResponse = {
  ok?: boolean;
  code?: string;
  message?: string;
  csrf_token?: string;
  redirect_to?: string;
};
type Field = "username" | "password" | "captcha";
type CaptchaResponse = { captcha_id?: unknown; image?: unknown };

const inputClass = "h-[46px] w-full rounded-[10px] border border-[#DDE3DF] bg-white px-3.5 text-[15px] text-[#18211C] outline-none transition placeholder:text-[#A8B2AC] hover:border-[#C6CFCA] focus:border-[#4FB37E] focus:shadow-[0_0_0_3px_rgba(79,179,126,.22)] disabled:bg-[#F7F9F8]";
const invalidClass = "border-[#D9534F] shadow-[0_0_0_3px_rgba(217,83,79,.12)] hover:border-[#D9534F]";
const FONT_STACK = '-apple-system, BlinkMacSystemFont, Inter, "PingFang SC", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif';
const labelClass = "mb-[7px] block text-[13px] text-[#6B7770]";

function prefersReducedMotion() {
  return typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export default function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [invalidField, setInvalidField] = useState<Field | null>(null);
  const [ready, setReady] = useState(false);
  const [focusedField, setFocusedField] = useState<Field | null>(null);
  const [yetiStatus, setYetiStatus] = useState<YetiStatus>("idle");
  const [statusKey, setStatusKey] = useState(0);
  const [captchaId, setCaptchaId] = useState("");
  const [captchaImage, setCaptchaImage] = useState("");
  const [captchaAnswer, setCaptchaAnswer] = useState("");
  const [captchaLoading, setCaptchaLoading] = useState(true);
  const usernameRef = useRef<HTMLInputElement>(null);
  const passwordRef = useRef<HTMLInputElement>(null);
  const captchaRef = useRef<HTMLInputElement>(null);
  const formRef = useRef<HTMLFormElement>(null);
  const captchaRequest = useRef(0);
  useEffect(() => setReady(true), []);

  /** Single-use server challenge: every refresh (and every failed login, which consumes it) needs a new one. */
  const refreshCaptcha = useCallback(async () => {
    const request = ++captchaRequest.current;
    setCaptchaLoading(true);
    setCaptchaAnswer("");
    let id = "";
    let image = "";
    try {
      const response = await fetch("/api/auth/captcha", { credentials: "same-origin", cache: "no-store" });
      const payload = await response.json().catch(() => ({})) as CaptchaResponse;
      if (response.ok && typeof payload.captcha_id === "string" && typeof payload.image === "string" && payload.image.startsWith("data:image/")) {
        id = payload.captcha_id;
        image = payload.image;
      }
    } catch {
      // Leave the image empty; the button shows 「点击刷新」.
    }
    if (request !== captchaRequest.current) return;
    setCaptchaId(id);
    setCaptchaImage(image);
    setCaptchaLoading(false);
  }, []);
  useEffect(() => { void refreshCaptcha(); }, [refreshCaptcha]);

  function fail(message: string, field: Field | null, focus?: HTMLInputElement | null, select = false) {
    setError(message);
    setInvalidField(field);
    setYetiStatus("error");
    setStatusKey((key) => key + 1);
    if (!prefersReducedMotion()) {
      formRef.current?.animate?.(["0", "-7px", "7px", "-4px", "4px", "0"].map((x) => ({ transform: `translateX(${x})` })), { duration: 450 });
    }
    // Next frame: after a server failure the inputs are still `disabled` (busy) until React re-renders.
    if (focus) requestAnimationFrame(() => { focus.focus(); if (select) focus.select(); });
  }

  function edited(field: Field) {
    setError("");
    setYetiStatus("idle");
    if (invalidField === field) setInvalidField(null);
  }

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!username.trim()) return fail("请输入账号", "username", usernameRef.current);
    if (!password) return fail("请输入密码", "password", passwordRef.current);
    if (!captchaAnswer.trim()) return fail("请输入验证码", "captcha", captchaRef.current);
    if (!captchaId) {
      void refreshCaptcha();
      return fail("验证码加载失败，请点击图片刷新", "captcha", captchaRef.current);
    }
    setBusy(true);
    setError("");
    setInvalidField(null);
    try {
      const next = new URLSearchParams(window.location.search).get("next") ?? "/workspace";
      const response = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: username.trim(), password, next, captcha_id: captchaId, captcha_answer: captchaAnswer.trim() }),
        credentials: "same-origin",
        cache: "no-store",
      });
      const payload = await response.json().catch(() => ({})) as LoginResponse;
      if (!response.ok || !payload.ok || !payload.csrf_token) {
        setBusy(false);
        void refreshCaptcha();
        if (payload.code === "captcha_invalid") return fail(payload.message || "验证码错误或已过期，请刷新后重试。", "captcha", captchaRef.current);
        return fail(payload.message || "登录失败，请检查账号和密码", null, passwordRef.current, true);
      }
      window.sessionStorage.setItem(CSRF_STORAGE_KEY, payload.csrf_token);
      setYetiStatus("success");
      setStatusKey((key) => key + 1);
      window.location.assign(payload.redirect_to || "/workspace");
    } catch {
      setBusy(false);
      void refreshCaptcha();
      fail("登录失败，请稍后重试", null, passwordRef.current);
    }
  }

  const focusHandlers = (field: Field) => ({
    onFocus: () => setFocusedField(field),
    onBlur: () => setFocusedField((current) => (current === field ? null : current)),
  });

  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-[#F5F8F6] px-5 pb-[60px] pt-10 text-[#18211C] antialiased" style={{ fontFamily: FONT_STACK }}>
      <div className="fixed left-5 top-5 z-[5] flex items-center gap-2.5 text-[17px] font-semibold tracking-[.2px] sm:left-10 sm:top-[34px]">
        <DeepEvoMark size={28} className="text-[#2F855A]" />
        <span>DeepEvo</span>
        <small className="ml-0.5 text-[15px] font-medium text-[#6B7770]">深度进化</small>
      </div>

      <div className="relative w-full max-w-[420px]">
        <LoginYeti
          passwordFocused={focusedField === "password"}
          passwordVisible={showPassword}
          status={yetiStatus}
          statusKey={statusKey}
          lookAtFieldId={focusedField && focusedField !== "password" ? focusedField : null}
        />
        <section className="relative z-[3] -mt-1.5 rounded-[20px] border border-[#E3E9E5] bg-white px-6 pb-8 pt-9 shadow-[0_1px_2px_rgba(20,40,30,.04)] sm:px-11">
          <h1 className="mb-[26px] text-center text-[26px] font-semibold tracking-[.5px]">登录</h1>

          <form ref={formRef} method="post" action="/api/auth/login" onSubmit={submit} noValidate>
            <div className="mb-[18px]">
              <label className={labelClass} htmlFor="username">账号</label>
              <input ref={usernameRef} id="username" name="username" type="text" autoComplete="username" placeholder="请输入账号" value={username}
                onChange={(event) => { setUsername(event.target.value); edited("username"); }} {...focusHandlers("username")} disabled={busy}
                aria-invalid={invalidField === "username" || undefined} className={cn(inputClass, invalidField === "username" && invalidClass)} />
            </div>
            <div className="mb-[18px]">
              <label className={labelClass} htmlFor="password">密码</label>
              <span className="relative flex items-center">
                <input ref={passwordRef} id="password" name="password" type={showPassword ? "text" : "password"} autoComplete="current-password" placeholder="请输入密码" value={password}
                  onChange={(event) => { setPassword(event.target.value); edited("password"); }} {...focusHandlers("password")} disabled={busy}
                  aria-invalid={invalidField === "password" || undefined} className={cn(inputClass, "pr-11", invalidField === "password" && invalidClass)} />
                <button type="button" onMouseDown={(event) => event.preventDefault()} onClick={() => { setShowPassword((value) => !value); passwordRef.current?.focus(); }} disabled={busy}
                  aria-label={showPassword ? "隐藏密码" : "显示密码"} className="absolute right-1.5 grid h-[34px] w-[34px] place-items-center rounded-lg text-[#8A958F] transition hover:bg-[#F0F3F1] hover:text-[#18211C]">
                  {showPassword ? <EyeOff className="h-[19px] w-[19px]" strokeWidth={1.8} /> : <Eye className="h-[19px] w-[19px]" strokeWidth={1.8} />}
                </button>
              </span>
            </div>
            <div className="mb-[18px]">
              <label className={labelClass} htmlFor="captcha">验证码</label>
              <div className="flex gap-2.5">
                <input ref={captchaRef} id="captcha" name="captcha" type="text" inputMode="numeric" autoComplete="off" maxLength={6} placeholder="请输入验证码" value={captchaAnswer}
                  onChange={(event) => { setCaptchaAnswer(event.target.value); edited("captcha"); }} {...focusHandlers("captcha")} disabled={busy}
                  aria-invalid={invalidField === "captcha" || undefined} className={cn(inputClass, "min-w-0 flex-1", invalidField === "captcha" && invalidClass)} />
                <button type="button" onClick={() => { void refreshCaptcha(); captchaRef.current?.focus(); }} disabled={busy} title="看不清？点击刷新" aria-label="刷新验证码"
                  className="grid h-[46px] w-[124px] flex-none place-items-center overflow-hidden rounded-[10px] border border-[#DDE3DF] bg-[#F4F7F5] p-0 text-[12px] text-[#6B7770] transition hover:border-[#C6CFCA]">
                  {captchaImage
                    ? <img src={captchaImage} alt="验证码" width={124} height={46} className={cn("block h-full w-full", captchaLoading && "opacity-50")} />
                    : <span>{captchaLoading ? "加载中…" : "点击刷新"}</span>}
                </button>
              </div>
            </div>
            <div className="-mt-1 mb-2.5 min-h-5 text-[13px] text-[#C0443F]" aria-live="polite">{error ? <p role="alert">{error}</p> : null}</div>
            <button type="submit" disabled={busy || !ready} className="h-12 w-full rounded-[10px] bg-[#2F855A] pl-[6px] text-base font-semibold tracking-[6px] text-white transition hover:bg-[#276B49] active:scale-[.99] disabled:cursor-wait disabled:opacity-75">
              {busy ? "登录中…" : "登录"}
            </button>
            <noscript>请启用浏览器脚本后再登录。</noscript>
          </form>
          <p className="mt-[22px] text-right text-[13px] text-[#6B7770]">© DeepEvo</p>
        </section>
      </div>
    </main>
  );
}
