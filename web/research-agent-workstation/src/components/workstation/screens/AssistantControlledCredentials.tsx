"use client";

import { forwardRef, useImperativeHandle, useMemo, useRef, useState } from "react";
import { ChevronDown, ShieldCheck } from "lucide-react";

type Locale = "zh-CN" | "en-US";
type Purpose = "weather4cast_sftp" | "kaggle_api";
type PublicKeyResponse = { ok?: boolean; code?: string; key_id?: string; rsa_jwk?: JsonWebKey };
type SecretIntent = {
  ok?: boolean; code?: string; secret_id: string; purpose: string; tenant_id: string;
  owner_principal_id: string; job_id: number; allocation_generation: number;
  profile_instance_id: string; key_id: string; expires_at_utc: string;
};

export type AssistantControlledCredentialsHandle = {
  sealPending: () => Promise<string[]>;
  commitSealed: (refs: string[]) => void;
  sealedPurpose: () => Purpose | null;
  open: () => void;
};
const CSRF_STORAGE_KEY = "evomind.local.csrf.v1";

function b64(value: ArrayBuffer | Uint8Array) {
  const bytes = value instanceof Uint8Array ? value : new Uint8Array(value);
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

function hex(value: ArrayBuffer) {
  return Array.from(new Uint8Array(value)).map((item) => item.toString(16).padStart(2, "0")).join("");
}

function csrfToken() {
  return window.sessionStorage.getItem(CSRF_STORAGE_KEY) || window.localStorage.getItem(CSRF_STORAGE_KEY) || "";
}

export const AssistantControlledCredentials = forwardRef<AssistantControlledCredentialsHandle, { locale: Locale }>(
  function AssistantControlledCredentials({ locale }, ref) {
    const [open, setOpen] = useState(false);
    const [purpose, setPurpose] = useState<Purpose>("weather4cast_sftp");
    const [host, setHost] = useState("");
    const [port, setPort] = useState("22");
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const [remotePath, setRemotePath] = useState("/");
    const [targetSubdir, setTargetSubdir] = useState("official");
    const [kaggleUsername, setKaggleUsername] = useState("");
    const [kaggleToken, setKaggleToken] = useState("");
    const [status, setStatus] = useState("");
    const [busy, setBusy] = useState(false);
    const [sealedCount, setSealedCount] = useState(0);
    const sealedBatchRef = useRef<{ purpose: Purpose; refs: string[] } | null>(null);
    const ready = useMemo(() => purpose === "weather4cast_sftp"
      ? Boolean(host.trim() && username.trim() && password && remotePath.startsWith("/"))
      : Boolean(kaggleToken), [host, kaggleToken, password, purpose, remotePath, username]);
    const tx = (zh: string, en: string) => locale === "zh-CN" ? zh : en;

    const openHpcQuickEnrollment = () => {
      const url = new URL(window.location.href);
      url.searchParams.set("page", "data");
      window.location.assign(url.toString());
    };

    useImperativeHandle(ref, () => ({
      sealPending: async () => {
        if (sealedBatchRef.current) return [...sealedBatchRef.current.refs];
        if (!ready) return [];
        if (busy) throw new Error("controlled_download_credentials_busy");
        setBusy(true);
        setStatus(tx("正在浏览器内存中加密…", "Encrypting in browser memory…"));
        try {
          const keyResponse = await fetch("/api/hpc/byoa/public-key", { cache: "no-store" });
          const key = await keyResponse.json() as PublicKeyResponse;
          if (!keyResponse.ok || !key.key_id || !key.rsa_jwk) throw new Error(key.code ?? "run_secret_key_unavailable");
          const count = purpose === "kaggle_api" ? 3 : 1;
          const refs: string[] = [];
          for (let index = 0; index < count; index += 1) {
            const intentResponse = await fetch("/api/assistant/run-secrets/intent", {
              method: "POST",
              headers: { "Content-Type": "application/json", "x-evomind-csrf": csrfToken() },
              body: JSON.stringify({ purpose }),
            });
            const intent = await intentResponse.json() as SecretIntent;
            if (!intentResponse.ok || !intent.ok) throw new Error(intent.code ?? "run_secret_intent_rejected");
            const aadText = `${intent.tenant_id}|${intent.owner_principal_id}|${intent.secret_id}|${intent.purpose}|${intent.job_id}|${intent.allocation_generation}|${intent.profile_instance_id}|${intent.key_id}|${intent.expires_at_utc}`;
            const aad = new TextEncoder().encode(aadText);
            const payload = purpose === "weather4cast_sftp"
              ? { schema: "evomind.run_secret_payload.v1", purpose, binding_sha256: hex(await crypto.subtle.digest("SHA-256", aad)), host: host.trim(), port: Number(port), username: username.trim(), password, remote_path: remotePath.trim(), target_subdir: targetSubdir.trim() || "official" }
              : { schema: "evomind.run_secret_payload.v1", purpose, binding_sha256: hex(await crypto.subtle.digest("SHA-256", aad)), ...(kaggleUsername.trim() ? { username: kaggleUsername.trim() } : {}), token: kaggleToken };
            const publicKey = await crypto.subtle.importKey("jwk", key.rsa_jwk, { name: "RSA-OAEP", hash: "SHA-256" }, false, ["encrypt"]);
            const aesKey = await crypto.subtle.generateKey({ name: "AES-GCM", length: 256 }, true, ["encrypt"]);
            const rawKey = await crypto.subtle.exportKey("raw", aesKey);
            const wrappedKey = await crypto.subtle.encrypt({ name: "RSA-OAEP" }, publicKey, rawKey);
            const iv = crypto.getRandomValues(new Uint8Array(12));
            const encrypted = await crypto.subtle.encrypt({ name: "AES-GCM", iv, additionalData: aad, tagLength: 128 }, aesKey, new TextEncoder().encode(JSON.stringify(payload)));
            const encryptedBytes = new Uint8Array(encrypted);
            const envelopeResponse = await fetch("/api/assistant/run-secrets/envelope", {
              method: "POST",
              headers: { "Content-Type": "application/json", "x-evomind-csrf": csrfToken() },
              body: JSON.stringify({ secret_id: intent.secret_id, encrypted_envelope: { schema: "evomind.hybrid_run_secret_envelope.v1", algorithm: "RSA-OAEP-256+A256GCM", key_id: intent.key_id, wrapped_key: b64(wrappedKey), iv: b64(iv), ciphertext: b64(encryptedBytes.slice(0, -16)), tag: b64(encryptedBytes.slice(-16)) } }),
            });
            const envelope = await envelopeResponse.json() as { ok?: boolean; code?: string; secret_ref?: string };
            if (!envelopeResponse.ok || !envelope.ok || !envelope.secret_ref) throw new Error(envelope.code ?? "run_secret_envelope_rejected");
            refs.push(envelope.secret_ref);
          }
          sealedBatchRef.current = { purpose, refs: [...refs] };
          setSealedCount(refs.length);
          setStatus(tx("已在浏览器内加密，正在等待后端绑定到当前 Run；绑定失败时不会清空表单或生成第二个引用。", "Encrypted in browser memory and awaiting backend binding to the current Run. A failed bind keeps the form and reuses this reference."));
          return refs;
        } finally {
          setBusy(false);
        }
      },
      commitSealed: (refs: string[]) => {
        const batch = sealedBatchRef.current;
        if (!batch || refs.length !== batch.refs.length || refs.some((value, index) => value !== batch.refs[index])) {
          throw new Error("controlled_secret_commit_mismatch");
        }
        sealedBatchRef.current = null;
        setSealedCount(0);
        setHost(""); setPort("22"); setUsername(""); setPassword(""); setRemotePath("/"); setTargetSubdir("official"); setKaggleUsername(""); setKaggleToken("");
        setStatus(tx(batch.purpose === "kaggle_api" ? "3 张独立一次性 Kaggle 引用已绑定到当前 Run。" : "Weather4cast 一次性引用已绑定到当前 Run。", batch.purpose === "kaggle_api" ? "Three independent one-time Kaggle references are bound to the current Run." : "The one-time Weather4cast reference is bound to the current Run."));
        setOpen(false);
      },
      sealedPurpose: () => sealedBatchRef.current?.purpose ?? null,
      open: () => setOpen(true),
   }), [busy, host, kaggleToken, kaggleUsername, locale, password, port, purpose, ready, remotePath, targetSubdir, username]);

    return (
      <div className="mb-2 rounded-lg border border-edge bg-surface-sunken/45" data-ui-assistant-controlled-credentials data-ui-skip-action="true">
        <button type="button" onClick={() => setOpen((value) => !value)} className="flex min-h-9 w-full items-center gap-2 px-3 py-2 text-left text-xs text-ink-secondary" aria-expanded={open}>
          <ShieldCheck className="h-3.5 w-3.5 text-accent" />
          <span className="flex-1 font-medium">{tx("连接与下载设置", "Connection and download settings")}</span>
          {sealedCount ? <span className="rounded bg-warning-light px-2 py-0.5 text-[10px] text-warning-text">{tx("待绑定", "Awaiting bind")}</span> : ready ? <span className="rounded bg-success-light px-2 py-0.5 text-[10px] text-success-text">{tx("待加密", "Ready")}</span> : null}
          <ChevronDown className={`h-3.5 w-3.5 transition-transform ${open ? "rotate-180" : ""}`} />
        </button>
        {open ? (
          <>
            <div className="grid gap-2 border-t border-edge p-3 sm:grid-cols-2 lg:grid-cols-3" data-ui-assistant-connection-details>
              <select value={purpose} onChange={(event) => setPurpose(event.target.value as Purpose)} aria-label="Controlled download purpose" className="h-9 rounded border border-edge bg-surface px-2.5 text-xs sm:col-span-2 lg:col-span-3">
                <option value="weather4cast_sftp">Weather4cast SFTP</option>
                <option value="kaggle_api">Kaggle 官方比赛数据（CURE / Ariel / Open Polymer）</option>
              </select>
              {purpose === "weather4cast_sftp" ? <>
                <input value={host} onChange={(event) => setHost(event.target.value)} placeholder="SFTP host" aria-label="Weather4cast SFTP host" autoComplete="off" className="h-9 rounded border border-edge bg-surface px-2.5 text-xs" />
                <input value={port} onChange={(event) => setPort(event.target.value.replace(/\D/g, ""))} placeholder="22" aria-label="Weather4cast SFTP port" inputMode="numeric" className="h-9 rounded border border-edge bg-surface px-2.5 text-xs" />
                <input value={username} onChange={(event) => setUsername(event.target.value)} placeholder={tx("SFTP 用户名", "SFTP username")} aria-label="Weather4cast SFTP username" autoComplete="off" className="h-9 rounded border border-edge bg-surface px-2.5 text-xs" />
                <input type="password" value={password} onChange={(event) => setPassword(event.target.value)} placeholder={tx("SFTP 访问口令", "SFTP access password")} aria-label="Weather4cast SFTP password" autoComplete="new-password" className="h-9 rounded border border-edge bg-surface px-2.5 text-xs" />
                <input value={remotePath} onChange={(event) => setRemotePath(event.target.value)} placeholder="/official/path" aria-label="Weather4cast remote path" autoComplete="off" className="h-9 rounded border border-edge bg-surface px-2.5 text-xs" />
                <input value={targetSubdir} onChange={(event) => setTargetSubdir(event.target.value)} placeholder="official" aria-label="Weather4cast target subdirectory" autoComplete="off" className="h-9 rounded border border-edge bg-surface px-2.5 text-xs" />
              </> : <>
                <input value={kaggleUsername} onChange={(event) => setKaggleUsername(event.target.value)} placeholder={tx("Kaggle 用户名（可选）", "Kaggle username (optional)")} aria-label="Kaggle username" autoComplete="off" className="h-9 rounded border border-edge bg-surface px-2.5 text-xs" />
                <input type="password" value={kaggleToken} onChange={(event) => setKaggleToken(event.target.value)} placeholder={tx("Kaggle API Token", "Kaggle API token")} aria-label="Kaggle API token" autoComplete="new-password" className="h-9 rounded border border-edge bg-surface px-2.5 text-xs sm:col-span-2" />
              </>}
              <p className="sm:col-span-2 lg:col-span-3 text-[10px] leading-4 text-ink-muted">{tx("字段仅保存在当前组件内存；发送时用 RSA-OAEP-256 + AES-256-GCM 加密，Run 只收到一次性引用。Kaggle 会自动生成 3 张引用，避免一个引用被第一个比赛消费后其余比赛失败。", "Fields remain only in component memory; send encrypts with RSA-OAEP-256 + AES-256-GCM. Kaggle creates three one-time references so one competition cannot consume the credential needed by the others.")}</p>
            </div>
            <div className="flex flex-wrap items-center justify-between gap-2 border-t border-edge px-3 py-2 text-[10px] text-ink-muted">
              <span>{tx("仅在更换连接或恢复 GPU allocation 时使用。", "Use only when changing a connection or restoring a GPU allocation.")}</span>
              <button
                type="button"
                data-ui-action="assistant_open_hpc_quick_enrollment"
                data-ui-skip-action="true"
                onClick={openHpcQuickEnrollment}
                className="rounded border border-accent-muted bg-accent-light px-2 py-1 font-medium text-accent-dark"
              >
                {tx("高级：AIMSLAB 快速换绑", "Advanced: AIMSLAB quick replacement")}
              </button>
            </div>
          </>
        ) : null}
        {status ? <div role="status" className="border-t border-edge px-3 py-1.5 text-[10px] text-ink-muted">{status}</div> : null}
      </div>
    );
  },
);
