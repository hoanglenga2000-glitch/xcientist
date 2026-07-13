export type ConnectorTone = "green" | "amber" | "red" | "slate";
export type ConnectorLocale = "zh-CN" | "en-US";

export type ConnectorDisplay = {
  name: string;
  state: string;
  detail: string;
  tone: ConnectorTone;
};

export type ConnectorDisplays = {
  gpu: ConnectorDisplay;
  kaggle: ConnectorDisplay;
  deepseek: ConnectorDisplay;
  codeAgent: ConnectorDisplay;
  humanGate: ConnectorDisplay;
};

function record(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function entry(status: Record<string, unknown> | undefined, key: string) {
  return record(status?.[key]);
}

function rawState(value: Record<string, unknown> | null) {
  return String(value?.state ?? value?.status ?? "").trim();
}

function localized(locale: ConnectorLocale, en: string, zh: string) {
  return locale === "zh-CN" ? zh : en;
}

function noEvidence(name: string, locale: ConnectorLocale): ConnectorDisplay {
  return {
    name,
    state: localized(locale, "Not Verified", "未验证"),
    detail: localized(locale, "No connector status evidence is available", "暂无连接状态证据"),
    tone: "slate"
  };
}

function rawDetail(value: Record<string, unknown> | null, fallback: string) {
  return String(value?.notes ?? value?.detail ?? value?.evidence_path ?? fallback).trim();
}

function hasFailureState(state: string) {
  return /(blocked|failed|error|invalid|unreadable)/i.test(state);
}

function blocked(name: string, locale: ConnectorLocale, value: Record<string, unknown>, state: string): ConnectorDisplay {
  return {
    name,
    state: localized(locale, "Blocked", "阻断"),
    detail: rawDetail(value, state || "failed"),
    tone: "red"
  };
}

export function deriveConnectorDisplays(
  connectorStatus: Record<string, unknown> | undefined,
  locale: ConnectorLocale = "zh-CN"
): ConnectorDisplays {
  const names = {
    gpu: "GPU / HPC",
    kaggle: "Kaggle API",
    deepseek: "DeepSeek API",
    codeAgent: localized(locale, "Code Agent", "代码 Agent"),
    humanGate: localized(locale, "Human Gate", "人工 Gate")
  };

  if (!connectorStatus) {
    return {
      gpu: noEvidence(names.gpu, locale),
      kaggle: noEvidence(names.kaggle, locale),
      deepseek: noEvidence(names.deepseek, locale),
      codeAgent: noEvidence(names.codeAgent, locale),
      humanGate: noEvidence(names.humanGate, locale)
    };
  }

  const gpu = entry(connectorStatus, "gpu");
  const gpuRaw = rawState(gpu);
  const gpuDisplay: ConnectorDisplay = !gpu
    ? noEvidence(names.gpu, locale)
    : hasFailureState(gpuRaw)
      ? blocked(names.gpu, locale, gpu, gpuRaw)
      : /pending/i.test(gpuRaw)
      ? { name: names.gpu, state: localized(locale, "SSH Pending", "SSH 待验证"), detail: rawDetail(gpu, gpuRaw), tone: "amber" }
      : gpu.configured === true && gpu.current_gate_ready === true
        ? { name: names.gpu, state: localized(locale, "Verified Ready", "已验证就绪"), detail: rawDetail(gpu, gpuRaw || "current_gate_ready"), tone: "green" }
        : gpu?.configured === true
          ? { name: names.gpu, state: localized(locale, "Runtime Unverified", "运行态未验证"), detail: rawDetail(gpu, gpuRaw || "configured"), tone: "amber" }
          : { name: names.gpu, state: localized(locale, "Not Configured", "未配置"), detail: rawDetail(gpu, gpuRaw || "not_configured"), tone: "red" };

  const kaggle = entry(connectorStatus, "kaggle");
  const kaggleRaw = rawState(kaggle);
  const kaggleDisplay: ConnectorDisplay = !kaggle
    ? noEvidence(names.kaggle, locale)
    : hasFailureState(kaggleRaw)
      ? blocked(names.kaggle, locale, kaggle, kaggleRaw)
    : kaggle.configured === true && kaggle.authenticated === true && kaggle.ready === true
      ? { name: names.kaggle, state: localized(locale, "Authenticated", "已认证"), detail: rawDetail(kaggle, kaggleRaw || "authenticated"), tone: "green" }
    : kaggle?.configured === true
      ? { name: names.kaggle, state: localized(locale, "Auth Pending", "待认证"), detail: rawDetail(kaggle, kaggleRaw || "configured_unverified"), tone: "amber" }
      : { name: names.kaggle, state: localized(locale, "Not Configured", "未配置"), detail: rawDetail(kaggle, kaggleRaw || "not_configured"), tone: "red" };

  const deepseek = entry(connectorStatus, "deepseek");
  const deepseekRaw = rawState(deepseek);
  const deepseekVerified = deepseek?.runtime_verified === true || deepseek?.smoke_passed === true || deepseek?.authenticated === true;
  const deepseekDisplay: ConnectorDisplay = !deepseek
    ? noEvidence(names.deepseek, locale)
    : hasFailureState(deepseekRaw)
      ? blocked(names.deepseek, locale, deepseek, deepseekRaw)
    : deepseek.configured === true && deepseekVerified
      ? { name: names.deepseek, state: localized(locale, "Runtime Verified", "运行态已验证"), detail: rawDetail(deepseek, deepseekRaw || "runtime_verified"), tone: "green" }
    : deepseek?.configured === true
      ? { name: names.deepseek, state: localized(locale, "Runtime Unverified", "运行态未验证"), detail: rawDetail(deepseek, deepseekRaw || "configured"), tone: "amber" }
      : { name: names.deepseek, state: localized(locale, "Not Configured", "未配置"), detail: rawDetail(deepseek, deepseekRaw || "not_configured"), tone: "red" };

  const codeAgent = entry(connectorStatus, "code_agent");
  const codeAgentRaw = rawState(codeAgent);
  const codeAgentVerified = codeAgent?.runtime_verified === true || codeAgent?.smoke_passed === true || codeAgent?.authenticated === true;
  const codeAgentDisplay: ConnectorDisplay = !codeAgent
    ? noEvidence(names.codeAgent, locale)
    : hasFailureState(codeAgentRaw)
      ? blocked(names.codeAgent, locale, codeAgent, codeAgentRaw)
    : codeAgent.configured === true && codeAgentVerified
      ? { name: names.codeAgent, state: localized(locale, "Runtime Verified", "运行态已验证"), detail: rawDetail(codeAgent, codeAgentRaw || "runtime_verified"), tone: "green" }
      : codeAgent.configured === true
        ? { name: names.codeAgent, state: localized(locale, "Runtime Unverified", "运行态未验证"), detail: rawDetail(codeAgent, codeAgentRaw || "configured"), tone: "amber" }
        : { name: names.codeAgent, state: localized(locale, "Not Configured", "未配置"), detail: rawDetail(codeAgent, codeAgentRaw || "not_configured"), tone: "red" };

  return {
    gpu: gpuDisplay,
    kaggle: kaggleDisplay,
    deepseek: deepseekDisplay,
    codeAgent: codeAgentDisplay,
    humanGate: kaggle?.human_gate_required_for_submission === true ? {
      name: names.humanGate,
      state: localized(locale, "Controlled", "受控"),
      detail: localized(locale, "Human approval required", "需要人工审批"),
      tone: "amber"
    } : noEvidence(names.humanGate, locale)
  };
}
