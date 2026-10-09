"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, CheckCircle2, RefreshCw, ShieldCheck } from "lucide-react";

type Binding = {
  state: "provisioning" | "active" | "frozen" | "retired";
  reason?: string;
  job_id: number;
  allocation_binding_id?: string;
  allocation_generation: number;
  credential_profile: string;
  updated_at_utc: string;
};

type ActivationStatus = {
  state: "pending_activation" | "retryable_activation_failure" | "completed" | "terminal_consumption_failure" | "consumption_state_unknown" | "terminal_generation_tombstoned";
  attempts: number;
  next_attempt_at_utc: string | null;
  updated_at_utc: string | null;
  error_code: string | null;
  receipt_exists: boolean;
  evidence_exists: boolean;
};

type EnrollmentIdentity = {
  reservation_id: string;
  nonce: string;
  job_id: number;
  allocation_binding_id: string;
  allocation_generation: number;
  credential_profile: string;
  profile_instance_id: string;
  expires_at_utc: string;
};

type PublicKeyResponse = {
  key_id: string;
  rsa_jwk: JsonWebKey;
  code?: string;
};

type PersistedQuickState = {
  schema: "evomind.hpc.quick_replace.v1";
  job: string;
  allocation: string;
  last_generation?: number;
  last_state?: string;
  updated_at: string;
};

const CSRF_STORAGE_KEY = "evomind.local.csrf.v1";
const QUICK_STATE_KEY = "evomind.hpc.quick_replace.v1";
const QUICK_SESSION_KEY = "evomind.hpc.quick_replace.session.v1";
const ACTIVATION_POLL_MS = 2_000;
const QUICK_TARGET_MS = 60_000;

function b64(bytes: ArrayBuffer | ArrayBufferView) {
  const view = bytes instanceof ArrayBuffer
    ? new Uint8Array(bytes)
    : new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return btoa(String.fromCharCode(...view));
}

function hex(bytes: ArrayBuffer) {
  return [...new Uint8Array(bytes)].map((value) => value.toString(16).padStart(2, "0")).join("");
}

function csrfToken() {
  const value = window.sessionStorage.getItem(CSRF_STORAGE_KEY);
  if (!value) throw new Error("csrf_missing");
  return value;
}

function writePersistentState(value: PersistedQuickState) {
  window.localStorage.setItem(QUICK_STATE_KEY, JSON.stringify(value));
}

function readPersistentState(): PersistedQuickState | null {
  try {
    const raw = window.localStorage.getItem(QUICK_STATE_KEY);
    if (!raw) return null;
    const value = JSON.parse(raw) as PersistedQuickState;
    return value?.schema === "evomind.hpc.quick_replace.v1" ? value : null;
  } catch {
    return null;
  }
}

function writeSessionIdentifiers(role: string, proxyUser: string) {
  window.sessionStorage.setItem(QUICK_SESSION_KEY, JSON.stringify({ role, proxyUser }));
}

function readSessionIdentifiers() {
  try {
    const raw = window.sessionStorage.getItem(QUICK_SESSION_KEY);
    if (!raw) return { role: "", proxyUser: "" };
    const value = JSON.parse(raw) as Record<string, unknown>;
    return {
      role: typeof value.role === "string" ? value.role : "",
      proxyUser: typeof value.proxyUser === "string" ? value.proxyUser : "",
    };
  } catch {
    return { role: "", proxyUser: "" };
  }
}

export function TenantHpcQuickEnrollment() {
  const [binding, setBinding] = useState<Binding | null>(null);
  const [activation, setActivation] = useState<ActivationStatus | null>(null);
  const [message, setMessage] = useState("");
  const [messageTone, setMessageTone] = useState<"neutral" | "success" | "warning">("neutral");
  const [busy, setBusy] = useState(false);
  const [watching, setWatching] = useState(false);
  const [replaceConfirmed, setReplaceConfirmed] = useState(false);
  const [job, setJob] = useState("");
  const [allocation, setAllocation] = useState("");
  const [role, setRole] = useState("");
  const [proxyUser, setProxyUser] = useState("");
  const [proxyPassword, setProxyPassword] = useState("");
  const [hpcPassword, setHpcPassword] = useState("");
  const mounted = useRef(true);
  const autoWatchKey = useRef("");

  const refresh = useCallback(async () => {
    const response = await fetch("/api/hpc/byoa/binding", { cache: "no-store" });
    const data = await response.json() as { ok?: boolean; code?: string; binding?: Binding | null; activation?: ActivationStatus | null };
    if (!response.ok || !data.ok) throw new Error(data.code ?? "tenant_binding_read_failed");
    const next = data.binding ?? null;
    if (mounted.current) {
      setBinding(next);
      setActivation(data.activation ?? null);
    }
    return next;
  }, []);

  useEffect(() => {
    mounted.current = true;
    const persisted = readPersistentState();
    const session = readSessionIdentifiers();
    if (persisted) {
      setJob(persisted.job);
      setAllocation(persisted.allocation);
    }
    setRole(session.role);
    setProxyUser(session.proxyUser);
    void refresh().catch((error) => {
      setMessage(error instanceof Error ? error.message : "tenant_binding_read_failed");
      setMessageTone("warning");
    });
    return () => { mounted.current = false; };
  }, [refresh]);

  useEffect(() => {
    writePersistentState({
      schema: "evomind.hpc.quick_replace.v1",
      job,
      allocation,
      last_generation: binding?.allocation_generation,
      last_state: binding?.state,
      updated_at: new Date().toISOString(),
    });
  }, [allocation, binding?.allocation_generation, binding?.state, job]);

  useEffect(() => {
    writeSessionIdentifiers(role, proxyUser);
  }, [proxyUser, role]);

  const needsRetirement = Boolean(binding && binding.state !== "retired");
  const retryableActivation = binding?.state === "provisioning" && activation?.state === "retryable_activation_failure";
  const activationMessage = useMemo(() => {
    if (!retryableActivation || !activation) return "";
    const nextAttempt = activation.next_attempt_at_utc
      ? new Date(activation.next_attempt_at_utc).toLocaleString(undefined, { hour12: false })
      : "等待后台调度";
    const detail = activation.error_code === "SOCKS5_proxy_could_not_connect_to_target_code_1"
      ? "指定代理暂时无法连接 HPC SSH 网关"
      : activation.error_code ? `激活前置检查未通过（${activation.error_code}）` : "激活前置检查未通过";
    return `${detail}；已安全重试 ${activation.attempts} 次，下次：${nextAttempt}。系统仍保持 provisioning，未启动 GPU 或训练。`;
  }, [activation, retryableActivation]);
  // The binding id is pasted from the allocation page and can carry invisible
  // leading/trailing whitespace. Validate the trimmed value (and submit the
  // trimmed value) so a stray space can never silently lock the submit button.
  const missingRequirements = useMemo(() => {
    const missing: string[] = [];
    if (!/^\d+$/.test(job) || !(Number(job) > 0)) missing.push("Job ID");
    if (!/^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$/.test(allocation.trim())) missing.push("Allocation binding ID");
    if (!role.trim()) missing.push("HPC role account");
    if (!proxyUser.trim()) missing.push("Proxy user");
    if (!proxyPassword) missing.push("Proxy password");
    if (!hpcPassword) missing.push("HPC password");
    if (needsRetirement && !replaceConfirmed) missing.push("确认勾选");
    return missing;
  }, [allocation, hpcPassword, job, needsRetirement, proxyPassword, proxyUser, replaceConfirmed, role]);
  const canSubmit = missingRequirements.length === 0;

  const replaceFailedProvisioningBinding = async () => {
    const response = await fetch("/api/hpc/byoa/binding", {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-evomind-csrf": csrfToken() },
      body: JSON.stringify({ action: "replace_failed_provisioning" }),
    });
    const result = await response.json() as { ok?: boolean; code?: string; binding?: Binding };
    if (!response.ok || !result.ok || result.binding?.state !== "retired") {
      throw new Error(result.code ?? "failed_provisioning_replacement_rejected");
    }
    return result.binding;
  };

  const retireCurrent = async () => {
    if (!binding || binding.state === "retired") return;
    if (binding.state === "provisioning") {
      // A retryable activation failure is still represented as provisioning in
      // the public binding. Use the same guarded server transition as the
      // explicit recovery button, then continue the new enrollment in this
      // click. The server rejects this path when no retryable request exists
      // or any identity receipt/evidence has already been written.
      const replaced = await replaceFailedProvisioningBinding();
      setBinding(replaced);
      return;
    }
    const response = await fetch("/api/hpc/byoa/binding", {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-evomind-csrf": csrfToken() },
      body: JSON.stringify({
        action: "retire",
        administrator_reclaim_confirmed: true,
        reason: "allocation_reclaimed_by_administrator",
      }),
    });
    const result = await response.json() as { ok?: boolean; code?: string; binding?: Binding };
    if (!response.ok || !result.ok || result.binding?.state !== "retired") {
      throw new Error(result.code ?? "tenant_profile_retirement_rejected");
    }
    setBinding(result.binding);
  };

  const submitEncryptedEnvelope = async () => {
    const sessionResponse = await fetch("/api/session/status", { cache: "no-store" });
    const sessionData = await sessionResponse.json() as { tenant_id?: string; username?: string };
    if (!sessionResponse.ok || !sessionData.tenant_id || !sessionData.username) throw new Error("tenant_session_required");

    const keyResponse = await fetch("/api/hpc/byoa/public-key", { cache: "no-store" });
    const key = await keyResponse.json() as PublicKeyResponse;
    if (!keyResponse.ok || !key.key_id || !key.rsa_jwk) throw new Error(key.code ?? "enrollment_key_invalid");

    const intentResponse = await fetch("/api/hpc/byoa/enrollment-identity", {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-evomind-csrf": csrfToken() },
      body: JSON.stringify({ job_id: Number(job), allocation_binding_id: allocation.trim() }),
    });
    const intent = await intentResponse.json() as EnrollmentIdentity & { code?: string };
    if (!intentResponse.ok) throw new Error(intent.code ?? "enrollment_identity_rejected");

    const source = `${sessionData.tenant_id}|${sessionData.username}|${intent.nonce}|${intent.job_id}|${intent.allocation_generation}|${intent.profile_instance_id}`;
    const bindingHash = hex(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(source)));
    const payload = {
      schema: "evomind.tenant_enrollment_secret.v1",
      binding_sha256: bindingHash,
      hpc_role_account: role.trim(),
      proxy_user: proxyUser.trim(),
      proxy_password: proxyPassword,
      hpc_password: hpcPassword,
    };
    const publicKey = await crypto.subtle.importKey("jwk", key.rsa_jwk, { name: "RSA-OAEP", hash: "SHA-256" }, false, ["encrypt"]);
    const aesKey = await crypto.subtle.generateKey({ name: "AES-GCM", length: 256 }, true, ["encrypt"]);
    const rawKey = await crypto.subtle.exportKey("raw", aesKey);
    const wrappedKey = await crypto.subtle.encrypt({ name: "RSA-OAEP" }, publicKey, rawKey);
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const aad = new TextEncoder().encode(`${sessionData.tenant_id}|${sessionData.username}|${intent.nonce}|${intent.job_id}|${intent.allocation_generation}|${intent.profile_instance_id}|${key.key_id}`);
    const encrypted = await crypto.subtle.encrypt({ name: "AES-GCM", iv, additionalData: aad, tagLength: 128 }, aesKey, new TextEncoder().encode(JSON.stringify(payload)));
    const encryptedBytes = new Uint8Array(encrypted);
    const ciphertext = encryptedBytes.slice(0, -16);
    const tag = encryptedBytes.slice(-16);

    const response = await fetch("/api/hpc/byoa/enrollment-envelope", {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-evomind-csrf": csrfToken() },
      body: JSON.stringify({
        reservation_id: intent.reservation_id,
        job_id: intent.job_id,
        allocation_binding_id: intent.allocation_binding_id,
        encrypted_envelope: {
          schema: "evomind.hybrid_tenant_envelope.v1",
          algorithm: "RSA-OAEP-256+A256GCM",
          key_id: key.key_id,
          wrapped_key: b64(wrappedKey),
          iv: b64(iv),
          ciphertext: b64(ciphertext),
          tag: b64(tag),
        },
      }),
    });
    const result = await response.json() as { ok?: boolean; code?: string };
    if (!response.ok || !result.ok) throw new Error(result.code ?? "enrollment_rejected");
    return intent;
  };

  const watchActivation = useCallback(async (expectedJob: number, expectedGeneration: number, startedAt: number) => {
    setWatching(true);
    try {
      while (mounted.current && Date.now() - startedAt < QUICK_TARGET_MS) {
        const current = await refresh();
        const elapsed = Math.max(0, Math.round((Date.now() - startedAt) / 1_000));
        if (current?.job_id === expectedJob && current.allocation_generation === expectedGeneration && current.state === "active") {
          setMessage(`Generation ${expectedGeneration} 已完成 5/5 身份绑定，用时 ${elapsed} 秒。`);
          setMessageTone("success");
          return true;
        }
        if (current?.job_id === expectedJob && current.allocation_generation === expectedGeneration && ["frozen", "retired"].includes(current.state)) {
          throw new Error(`enrollment_${current.state}`);
        }
        setMessage(`Generation ${expectedGeneration} 正在服务账户中完成代理、Host/GPU UUID 与 5/5 容器身份门禁（${elapsed}s）…`);
        await new Promise((resolve) => window.setTimeout(resolve, ACTIVATION_POLL_MS));
      }
      setMessage(`Generation ${expectedGeneration} 已安全受理，60 秒窗口内仍处于 provisioning；后台会继续处理，刷新页面可恢复状态。`);
      setMessageTone("warning");
      return false;
    } finally {
      setWatching(false);
    }
  }, [refresh]);

  useEffect(() => {
    if (!binding || binding.state !== "provisioning" || watching || busy) return;
    const key = `${binding.job_id}:${binding.allocation_generation}:${binding.updated_at_utc}`;
    if (autoWatchKey.current === key) return;
    autoWatchKey.current = key;
    void watchActivation(binding.job_id, binding.allocation_generation, Date.now()).catch((error) => {
      setMessage(error instanceof Error ? error.message : "enrollment_watch_failed");
      setMessageTone("warning");
    });
  }, [binding, busy, watchActivation, watching]);

  const quickReplace = async () => {
    if (!canSubmit || busy) return;
    setBusy(true);
    setMessageTone("neutral");
    setMessage(needsRetirement ? "正在为旧 allocation 写入退役 tombstone…" : "正在签发新的 enrollment identity…");
    const startedAt = Date.now();
    try {
      if (needsRetirement) await retireCurrent();
      setMessage("正在浏览器内存中加密新凭据并提交 envelope…");
      const intent = await submitEncryptedEnvelope();
      setProxyPassword("");
      setHpcPassword("");
      setReplaceConfirmed(false);
      setMessage(`Generation ${intent.allocation_generation} 已加密受理，正在等待 5/5 身份绑定…`);
      await watchActivation(intent.job_id, intent.allocation_generation, startedAt);
    } catch (error) {
      setProxyPassword("");
      setHpcPassword("");
      setMessage(error instanceof Error ? error.message : "hpc_quick_replace_failed");
      setMessageTone("warning");
      await refresh().catch(() => null);
    } finally {
      setBusy(false);
    }
  };

  const replaceFailedProvisioning = async () => {
    if (!binding || binding.state !== "provisioning" || busy) return;
    setBusy(true);
    try {
      const replaced = await replaceFailedProvisioningBinding();
      setBinding(replaced);
      setMessage(`Generation ${binding.allocation_generation} 的失败登记已安全退役，可以立即提交新 allocation。`);
      setMessageTone("success");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "failed_provisioning_replacement_failed");
      setMessageTone("warning");
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="rounded-xl border border-edge bg-surface p-4" data-testid="tenant-hpc-quick-enrollment" data-ui-skip-action="true">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-ink">AIMSLAB 60 秒快速换绑</h3>
          <p className="mt-1 max-w-3xl text-xs leading-5 text-ink-muted">
            新 allocation 使用独立 Generation 与 DPAPI profile。密码只在当前浏览器内存中参与 Web Crypto 加密，提交后立即清空，不写入 localStorage、日志或源码。
          </p>
        </div>
        <button type="button" onClick={() => void refresh()} disabled={busy || watching} className="inline-flex items-center gap-1 rounded-md border border-edge px-2.5 py-1 text-xs font-medium text-ink-secondary disabled:opacity-50">
          <RefreshCw className="h-3 w-3" />刷新状态
        </button>
      </div>

      <div className="mt-3 rounded-lg border border-edge bg-surface-sunken/55 px-3 py-2 text-xs text-ink-secondary">
        当前：<strong>{binding?.state ?? "未绑定"}</strong>
        {binding ? <span className="ml-2 font-mono">job{binding.job_id} · G{binding.allocation_generation} · {binding.credential_profile}</span> : null}
      </div>

      {activationMessage ? (
        <p data-testid="tenant-hpc-activation-retry-status" className="mt-2 rounded-lg border border-warning/35 bg-warning/5 px-3 py-2 text-xs leading-5 text-warning-text">
          {activationMessage}
        </p>
      ) : null}

      <div className="mt-3 grid gap-2 md:grid-cols-2 xl:grid-cols-3">
        <input value={job} onChange={(event) => setJob(event.target.value.replace(/\D/g, ""))} inputMode="numeric" placeholder="Job ID" aria-label="Job ID" className="h-10 rounded border border-edge bg-surface px-3 text-xs" />
        <input value={allocation} onChange={(event) => setAllocation(event.target.value)} placeholder="Allocation binding ID" aria-label="Allocation binding ID" className="h-10 rounded border border-edge bg-surface px-3 text-xs" />
        <input value={role} onChange={(event) => setRole(event.target.value)} placeholder="HPC role account" aria-label="HPC role account" autoComplete="off" className="h-10 rounded border border-edge bg-surface px-3 text-xs" />
        <input value={proxyUser} onChange={(event) => setProxyUser(event.target.value)} placeholder="Proxy user" aria-label="Proxy user" autoComplete="off" className="h-10 rounded border border-edge bg-surface px-3 text-xs" />
        <input type="password" value={proxyPassword} onChange={(event) => setProxyPassword(event.target.value)} placeholder="Proxy password" aria-label="Proxy password" autoComplete="new-password" className="h-10 rounded border border-edge bg-surface px-3 text-xs" />
        <input type="password" value={hpcPassword} onChange={(event) => setHpcPassword(event.target.value)} placeholder="HPC password" aria-label="HPC password" autoComplete="new-password" className="h-10 rounded border border-edge bg-surface px-3 text-xs" />
      </div>

      {needsRetirement ? (
        <label className="mt-3 flex items-start gap-2 text-xs text-warning-text">
          <input type="checkbox" checked={replaceConfirmed} onChange={(event) => setReplaceConfirmed(event.target.checked)} aria-label="确认退役旧 allocation 并创建新 Generation" className="mt-0.5" />
          <span>我确认当前 allocation 已被管理员删除、回收或替换；系统将先退役旧 Generation，再提交上述新 allocation。</span>
        </label>
      ) : null}

      <div className="mt-3 flex flex-wrap items-center gap-3">
        <button type="button" disabled={!canSubmit || busy || watching} onClick={() => void quickReplace()} className="inline-flex min-h-10 items-center gap-2 rounded-lg bg-accent px-4 py-2 text-xs font-semibold text-accent-foreground disabled:cursor-not-allowed disabled:opacity-45">
          <ShieldCheck className="h-4 w-4" />{busy || watching ? "正在换绑…" : needsRetirement ? "退役旧绑定并快速换绑" : "加密登记新 allocation"}
        </button>
        {retryableActivation ? (
          <button type="button" disabled={busy || watching} onClick={() => void replaceFailedProvisioning()} className="rounded-lg border border-warning-border px-3 py-2 text-xs font-semibold text-warning-text disabled:opacity-45">
            废弃已确认失败的 provisioning
          </button>
        ) : null}
        {missingRequirements.length ? (
          <span data-testid="tenant-hpc-quick-missing" className="text-[11px] text-warning-text">
            还差：{missingRequirements.join("、")}
          </span>
        ) : null}
        <span className="text-[11px] text-ink-muted">目标窗口：正常代理与账号可用时 60 秒内 active；异常时保持 fail-closed。</span>
      </div>

      {message ? (
        <p role="status" className={`mt-3 flex items-start gap-2 rounded-lg border px-3 py-2 text-xs leading-5 ${messageTone === "success" ? "border-success/35 bg-success/5 text-success-text" : messageTone === "warning" ? "border-warning/35 bg-warning/5 text-warning-text" : "border-edge bg-surface-sunken/55 text-ink-secondary"}`}>
          {messageTone === "success" ? <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" /> : messageTone === "warning" ? <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" /> : <RefreshCw className="mt-0.5 h-4 w-4 shrink-0 animate-spin" />}
          <span>{message}</span>
        </p>
      ) : null}
    </section>
  );
}
