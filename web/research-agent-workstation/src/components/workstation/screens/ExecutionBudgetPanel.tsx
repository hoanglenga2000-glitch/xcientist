"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Sliders } from "lucide-react";
import { Panel } from "../primitives/Layout";
import { t } from "../localization";
import { hoursToSeconds, validateBudget, type BudgetResponse } from "./execution-budget";

const endpoint = "/api/assistant/execution-budget";

export function ExecutionBudgetPanel({ locale }: { locale: "zh-CN" | "en-US" }) {
  const [data, setData] = useState<BudgetResponse | null>(null);
  const [gpu, setGpu] = useState("");
  const [engineering, setEngineering] = useState("");
  const [reason, setReason] = useState("");
  const [confirmed, setConfirmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const requestRef = useRef<{ payload: string; id: string } | null>(null);
  const locked = useRef(false);
  const zh = locale === "zh-CN";
  const hours = (seconds: number) => (seconds / 3600).toLocaleString(locale, { maximumFractionDigits: 3 });
  const errorText = (code: string) => ({
    budget_policy_owner_required: zh ? "此账户未被登记为预算管理者，请联系管理员。" : "This account is not a registered budget owner.",
    budget_revision_conflict: zh ? "额度已被其他页面修改，请刷新后重新确认。" : "Budget changed elsewhere. Refresh and confirm again.",
    budget_below_committed: zh ? "额度不能低于已消耗与预留之和。" : "Limits cannot be lower than charged plus reserved time.",
    budget_limits_invalid: zh ? "请输入有效小时数；工程额度不得超过总额度（上限 8760 小时）。" : "Enter valid hours. Engineering cannot exceed total (maximum 8760 hours).",
    budget_unchanged: zh ? "额度未改变，无需保存。" : "Limits are unchanged.",
    session_required: zh ? "请登录后重试。" : "Please sign in and retry.",
  }[code] ?? (zh ? `操作未完成：${code}。请刷新核实当前额度。` : `Not completed: ${code}. Refresh to verify the current budget.`));
  const adopt = useCallback((payload: BudgetResponse) => {
    setData(payload);
    setGpu(String(Number((payload.budget.gpu_limit_seconds / 3600).toFixed(4))));
    setEngineering(String(Number((payload.budget.engineering_limit_seconds / 3600).toFixed(4))));
    setConfirmed(false);
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    fetch(endpoint, { cache: "no-store", signal: controller.signal }).then(async (response) => {
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error ?? `HTTP_${response.status}`);
      if (!controller.signal.aborted) adopt(payload);
    }).catch((error: Error) => { if (!controller.signal.aborted) setMessage(error.message); });
    return () => controller.abort();
  }, [adopt]);
  const refresh = async () => {
    if (locked.current) return;
    locked.current = true; setBusy(true); setMessage("");
    try {
      const response = await fetch(endpoint, { cache: "no-store" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error ?? `HTTP_${response.status}`);
      adopt(payload); setReason(""); requestRef.current = null;
    } catch (error) { setMessage(error instanceof Error ? error.message : "request_failed"); }
    finally { locked.current = false; setBusy(false); }
  };
  const save = async () => {
    if (!data || !confirmed || reason.trim().length < 3 || locked.current) return;
    locked.current = true; setBusy(true); setMessage("");
    try {
      const total = hoursToSeconds(gpu), portion = hoursToSeconds(engineering);
      const invalid = validateBudget(total, portion, data.budget, data.max_limit_seconds);
      if (invalid) throw new Error(invalid);
      const body = { gpu_limit_seconds: total, engineering_limit_seconds: portion, expected_revision: data.budget.revision, reason: reason.trim(), confirmed: true };
      const serialized = JSON.stringify(body);
      if (requestRef.current?.payload !== serialized) requestRef.current = { payload: serialized, id: crypto.randomUUID() };
      const response = await fetch(endpoint, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...body, request_id: requestRef.current.id }) });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error ?? `HTTP_${response.status}`);
      adopt(payload); setReason(""); requestRef.current = null; setMessage("saved");
    } catch (error) { setMessage(error instanceof Error ? error.message : "request_failed"); }
    finally { locked.current = false; setBusy(false); }
  };
  return <Panel title={t(locale, "GPU execution budget", "GPU 执行预算")} icon={Sliders} className="lg:col-span-2">
    <div className="space-y-4 text-xs" data-testid="execution-budget-panel">
      <p className="text-ink-muted">{t(locale, "Engineering is part of the total, not an extra allowance. Saving preserves all charges and does not start training or bypass approvals. SIIM and official study limits remain separate.", "工程额度包含在总额度内，不额外叠加。保存只调整后续准入，保留全部历史记账，不启动训练、不绕过审批；SIIM 与正式研究预算独立。")}</p>
      {!data && !message && <p role="status">{t(locale, "Loading budget…", "正在读取预算…")}</p>}
      {data && <>
        <p className="break-all text-ink-secondary">{t(locale, "Study", "研究")}: {data.budget.study} · {t(locale, "Revision", "版本")} {data.budget.revision} · {data.budget.operations} {t(locale, "operations", "条记账")}</p>
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          {[[t(locale, "Total remaining (h)", "总剩余额度（小时）"), data.budget.gpu_remaining_seconds], [t(locale, "Engineering remaining (h)", "工程剩余额度（小时）"), data.budget.engineering_remaining_seconds], [t(locale, "Charged (h)", "累计消耗（小时）"), data.budget.charged_wall_seconds], [t(locale, "Reserved (h)", "当前预留（小时）"), data.budget.pending_reserved_seconds]].map(([label, value]) => <div key={String(label)} className="rounded border border-edge p-3"><div className="text-ink-muted">{label}</div><div className="mt-1 text-lg font-semibold tabular-nums">{hours(Number(value))}</div></div>)}
        </div>
        {data.budget.reconciliation_required && <p role="alert" className="text-amber-600">{t(locale, "Reconciliation required. Raising the budget cannot clear this execution block.", "存在待对账操作；提高额度也不会清除此执行阻断。")}</p>}
        <fieldset disabled={busy || !data.can_manage} className="space-y-3">
          <div className="grid gap-3 sm:grid-cols-2">
            <label className="space-y-1"><span>{t(locale, "Total limit (hours)", "总额度上限（小时）")}</span><input aria-label={t(locale, "Total limit (hours)", "总额度上限（小时）")} type="number" min="0" max="8760" step="0.0001" value={gpu} onChange={(e) => { setGpu(e.target.value); setConfirmed(false); }} className="w-full rounded border border-edge bg-surface px-3 py-2" /></label>
            <label className="space-y-1"><span>{t(locale, "Engineering limit (hours)", "工程额度上限（小时）")}</span><input aria-label={t(locale, "Engineering limit (hours)", "工程额度上限（小时）")} type="number" min="0" max="8760" step="0.0001" value={engineering} onChange={(e) => { setEngineering(e.target.value); setConfirmed(false); }} className="w-full rounded border border-edge bg-surface px-3 py-2" /></label>
          </div>
          <p className="text-ink-muted">{t(locale, "Committed total / engineering", "已消耗及预留：总计 / 工程")}: {hours(data.budget.committed_seconds)} / {hours(data.budget.engineering_committed_seconds)} h</p>
          <label className="block space-y-1"><span>{t(locale, "Adjustment reason (required)", "调整原因（必填）")}</span><textarea aria-label={t(locale, "Adjustment reason (required)", "调整原因（必填）")} maxLength={500} value={reason} onChange={(e) => { setReason(e.target.value); setConfirmed(false); }} placeholder={t(locale, "Purpose and authorized scope; do not enter credentials", "填写用途与授权范围，请勿填写密码")} className="w-full rounded border border-edge bg-surface px-3 py-2" /></label>
          <label className="flex items-start gap-2"><input type="checkbox" checked={confirmed} onChange={(e) => setConfirmed(e.target.checked)} /><span>{t(locale, "I authorize these limits and understand that historical charges are preserved.", "我确认授权以上额度，且了解历史消耗不会被清零。")}</span></label>
          <button type="button" data-ui-skip-action="true" disabled={!confirmed || reason.trim().length < 3} onClick={() => void save()} className="rounded bg-accent px-3 py-2 font-medium text-accent-fg disabled:opacity-50">{t(locale, "Save authorized budget", "保存授权预算")}</button>
        </fieldset>
        <details><summary className="cursor-pointer text-ink-secondary">{t(locale, "Recent budget adjustments", "最近预算调整记录")} ({data.history.length})</summary><ul className="mt-2 space-y-2">{data.history.map((row) => <li key={row.revision} className="break-words rounded border border-edge p-2">#{row.revision} · {new Date(row.created * 1000).toLocaleString(locale)} · {row.owner_id}<br />{hours(row.previous_gpu_limit)} → {hours(row.gpu_limit)} h / {hours(row.previous_engineering_limit)} → {hours(row.engineering_limit)} h<br />{row.reason}</li>)}</ul></details>
      </>}
      <button type="button" data-ui-skip-action="true" disabled={busy} onClick={() => void refresh()} className="rounded border border-edge px-3 py-2 disabled:opacity-50">{t(locale, "Refresh / discard edits", "刷新预算 / 撤销未保存修改")}</button>
      {message && <p role="status" aria-live="polite">{message === "saved" ? t(locale, "Budget saved. No execution started.", "预算已保存，未启动任何执行。") : errorText(message)}</p>}
    </div>
  </Panel>;
}
