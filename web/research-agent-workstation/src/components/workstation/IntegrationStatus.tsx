"use client";

import { useEffect, useState } from "react";
import { PlugZap, Settings2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusBadge, type StatusTone } from "@/components/ui/status-badge";

type ProviderStatus = {
  name: string;
  state: string;
  configured: boolean;
  notes?: string;
};

type Summary = {
  connector_status?: {
    code_agent: ProviderStatus;
    deepseek?: ProviderStatus;
    python_runner: ProviderStatus;
    gpu: ProviderStatus;
    kaggle: ProviderStatus;
    llm: ProviderStatus;
    storage: ProviderStatus;
    env_keys: Record<string, string>;
  };
  runs?: Array<{
    task_id: string;
    best_model?: string;
    best_metrics?: Record<string, number>;
    accepted?: boolean;
  }>;
};

const fallback: Summary = {
  connector_status: {
    code_agent: { name: "Code Agent", state: "Local Template", configured: true },
    deepseek: { name: "DeepSeek", state: "Not Configured", configured: false },
    python_runner: { name: "Python Runner", state: "Local", configured: true },
    gpu: { name: "GPU", state: "Not Connected", configured: false },
    kaggle: { name: "Kaggle", state: "Not Configured", configured: false },
    llm: { name: "LLM", state: "Rule-based", configured: true },
    storage: { name: "Storage", state: "Local Workspace", configured: true },
    env_keys: {
      CODE_AGENT_PROVIDER: "local_template",
      PYTHON_RUNNER: "local",
      GPU_PROVIDER: "mock",
      KAGGLE_ENABLED: "false",
      LLM_PROVIDER: "rule_based"
    }
  },
  runs: []
};

function ui(locale: "zh-CN" | "en-US" | undefined, en: string, zh: string) {
  return locale === "zh-CN" ? zh : en;
}

function providerName(locale: "zh-CN" | "en-US" | undefined, name: string) {
  if (locale !== "zh-CN") return name;
  const names: Record<string, string> = {
    "Code Agent": "代码 Agent",
    "Claude Code": "Claude Code",
    DeepSeek: "DeepSeek",
    "Python Runner": "Python 运行器",
    GPU: "GPU",
    "GPU SSH Gateway": "GPU SSH 网关",
    Kaggle: "Kaggle",
    LLM: "大模型",
    Storage: "存储"
  };
  return names[name] ?? name;
}

function providerState(locale: "zh-CN" | "en-US" | undefined, state: string) {
  if (locale !== "zh-CN") return state;
  const normalized = state.toLowerCase();
  if (normalized.includes("gpu ssh gateway ready") || normalized.includes("gpu verified")) {
    const liveSummary = state.match(/(\d+\s*x\s*NVIDIA\s*[^/]+)/i)?.[1]?.trim();
    return liveSummary ? `GPU 已验证：${liveSummary}` : state;
  }
  if (normalized.includes("not configured")) return "未配置";
  if (normalized.includes("not connected")) return "未连接";
  if (normalized.includes("web terminal ready")) return "GPU 环境已创建 / Web 终端可用 / 外部 SSH 待确认";
  if (normalized.includes("external ssh pending")) return "外部 SSH 待确认";
  if (normalized.includes("ready")) return "就绪";
  if (normalized.includes("local workspace")) return "本地工作区";
  if (normalized.includes("local template")) return "本地模板";
  if (normalized.includes("local")) return "本地";
  if (normalized.includes("rule")) return "规则引擎";
  return state;
}

function providerBadge(locale: "zh-CN" | "en-US" | undefined, provider: ProviderStatus) {
  const normalized = `${provider.state} ${provider.notes ?? ""}`.toLowerCase();
  if (normalized.includes("public key") || normalized.includes("authorization") || normalized.includes("pending")) {
    if (normalized.includes("verified")) return ui(locale, "Verified", "已验证");
    return ui(locale, "Pending", "待确认");
  }
  if (provider.configured) return ui(locale, "Ready", "就绪");
  return ui(locale, "Not Configured", "待配置");
}

function providerTone(provider: ProviderStatus): StatusTone {
  const normalized = `${provider.state} ${provider.notes ?? ""}`.toLowerCase();
  if (normalized.includes("verified") && (normalized.includes("public key") || normalized.includes("authorization") || normalized.includes("pending"))) return "blue";
  if (provider.configured) return "green";
  if (normalized.includes("not configured")) return "slate";
  return "amber";
}

export function IntegrationStatus({ compact = false, locale }: { compact?: boolean; locale?: "zh-CN" | "en-US" }) {
  const [summary, setSummary] = useState<Summary>(fallback);

  useEffect(() => {
    let mounted = true;
    fetch("/api/workstation-summary")
      .then((response) => response.json())
      .then((payload: Summary) => {
        if (mounted) setSummary(payload);
      })
      .catch(() => {
        if (mounted) setSummary(fallback);
      });
    return () => {
      mounted = false;
    };
  }, []);

  const status = { ...fallback.connector_status!, ...(summary.connector_status ?? {}) };
  const providers = [
    status.code_agent,
    status.deepseek ?? fallback.connector_status!.deepseek!,
    status.python_runner,
    status.gpu,
    status.kaggle,
    status.llm,
    status.storage
  ];

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <div className="flex items-center gap-2">
          <PlugZap className="h-4 w-4 text-primary" />
          <CardTitle>{ui(locale, "Integration Status", "接入状态")}</CardTitle>
        </div>
        <StatusBadge tone="blue">{ui(locale, "Adapter Ready", "适配器就绪")}</StatusBadge>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className={compact ? "grid grid-cols-3 gap-2" : "grid grid-cols-2 gap-2 xl:grid-cols-3"}>
          {providers.map((provider) => (
            <div key={provider.name} className="rounded-md border border-border bg-white px-3 py-2 shadow-[0_1px_0_rgba(15,23,42,0.03)]">
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-bold text-slate-700">{providerName(locale, provider.name)}</span>
                <StatusBadge tone={providerTone(provider)}>
                  {providerBadge(locale, provider)}
                </StatusBadge>
              </div>
              <div className="mt-2 min-h-8 text-xs leading-4 text-slate-600">{providerState(locale, provider.state)}</div>
              {!compact && provider.notes ? (
                <div className="mt-2 border-t border-slate-100 pt-2 text-[11px] leading-4 text-slate-500">{provider.notes}</div>
              ) : null}
            </div>
          ))}
        </div>
        {!compact ? (
          <div className="rounded-md border border-border bg-slate-50 p-3">
            <div className="mb-2 flex items-center gap-2 text-xs font-bold text-slate-700">
              <Settings2 className="h-3.5 w-3.5" />
              {ui(locale, "Environment Slots", "环境配置槽位")}
            </div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-slate-600">
              {Object.entries(status.env_keys ?? fallback.connector_status!.env_keys).map(([key, value]) => (
                <div key={key} className="flex justify-between gap-3">
                  <span className="font-semibold">{key}</span>
                  <span>{value}</span>
                </div>
              ))}
            </div>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
