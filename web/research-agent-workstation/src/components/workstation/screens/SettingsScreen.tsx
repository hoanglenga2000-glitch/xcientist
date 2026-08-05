"use client";

import { useEffect, useRef, useState } from "react";
import {
  Eye,
  Globe,
  KeyRound,
  Moon,
  Palette,
  ShieldCheck,
  Sliders,
  Sun,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { WorkstationSummary } from "@/lib/api/types";
import { PageHeader, Panel } from "../primitives/Layout";
import { StatusBadgeV2, StatusDot } from "../primitives/StatusBadge";
import { t } from "../localization";
import { useTheme, type ThemeMode } from "../theme/ThemeProvider";
import * as api from "@/lib/api/client";

type Locale = "zh-CN" | "en-US";

type ScreenProps = {
  selectedTask: string;
  setSelectedTask: (id: string) => void;
  selectedStage: string;
  setSelectedStage: (id: string) => void;
  selectedExperiment: string;
  setSelectedExperiment: (id: string) => void;
  gateStatus: "Pending" | "Approved" | "Rejected";
  setGateStatus: (status: "Pending" | "Approved" | "Rejected") => void;
  patchApplied: boolean;
  setPatchApplied: (value: boolean) => void;
  reportSubmitted: boolean;
  setReportSubmitted: (value: boolean) => void;
  summary?: WorkstationSummary | null;
  refreshSummary?: () => Promise<WorkstationSummary>;
  runLocalExperiment?: (taskId?: string) => Promise<void>;
  runState?: { status: "idle" | "running" | "passed" | "failed"; message: string; experimentDir?: string };
  exportCodeAgentContext?: (taskId?: string, targetAgent?: string) => Promise<void>;
  importDemoPatch?: (taskId?: string) => Promise<void>;
  agentActionMessage?: string;
  runWorkstationAction?: (action: string, metadata?: Record<string, unknown>) => Promise<unknown>;
  systemActionMessage?: string;
  lastActionTrace?: {
    action: string; taskId?: string; request?: Record<string, unknown>;
    response?: Record<string, unknown>; message: string; artifact?: string | null; at: string;
  } | null;
  locale?: Locale;
  setLocale?: (locale: Locale) => void;
};

function connectorConfigured(summary: WorkstationSummary | null | undefined, key: string): boolean {
  const cs = summary?.connector_status as Record<string, Record<string, unknown>> | undefined;
  return Boolean(cs?.[key]?.configured);
}

export function SettingsScreen(props: ScreenProps) {
  const { summary, locale = "zh-CN", setLocale } = props;
  const { mode, resolvedTheme, persistedMode, previewMode, saveMode, resetMode, ready, saving } = useTheme();
  const persistedLocaleRef = useRef<Locale>(locale);
  const [localeSettingsReady, setLocaleSettingsReady] = useState(false);
  const [settingsFeedback, setSettingsFeedback] = useState("");
  const preferencesReady = ready && localeSettingsReady;

  useEffect(() => {
    let active = true;
    api.getSettings()
      .then((payload) => {
        const saved = payload.settings?.language?.ui_language;
        if (active && (saved === "en-US" || saved === "zh-CN")) persistedLocaleRef.current = saved;
      })
      .catch(() => undefined)
      .finally(() => { if (active) setLocaleSettingsReady(true); });
    return () => { active = false; };
  }, []);

  const secretEntries = [
    { key: "kaggle_api", label: "Kaggle API" },
    { key: "deepseek_api", label: "DeepSeek API" },
    { key: "gpu_hpc", label: "GPU / HPC SSH" },
  ];

  const recordSettingsAction = (action: string, metadata: Record<string, unknown> = {}) => {
    void props.runWorkstationAction?.(action, {
      source: "settings_screen",
      ...metadata,
    });
  };

  const selectTheme = (nextMode: ThemeMode) => {
    if (!preferencesReady) return;
    previewMode(nextMode);
    recordSettingsAction("settings_theme_change", { theme: nextMode, preview: true });
    setSettingsFeedback(t(locale, "Theme preview active. Save to persist it.", "主题预览已生效，保存后持久化。"));
  };

  const savePreferences = async () => {
    if (!preferencesReady) return;
    setSettingsFeedback(t(locale, "Saving preferences…", "正在保存偏好…"));
    try {
      const current = await api.getSettings();
      await saveMode({
        language: {
          ...current.settings?.language,
          ui_language: locale,
        },
      });
      persistedLocaleRef.current = locale;
      recordSettingsAction("save_settings_changes", { theme: mode, language: locale, persisted: true });
      setSettingsFeedback(t(locale, "Theme and language saved.", "主题和语言已保存。"));
    } catch (error) {
      setSettingsFeedback(error instanceof Error ? error.message : t(locale, "Save failed.", "保存失败。"));
    }
  };

  const cancelPreferences = () => {
    if (!preferencesReady) return;
    resetMode();
    setLocale?.(persistedLocaleRef.current);
    recordSettingsAction("cancel_settings_changes", { theme: persistedMode, language: persistedLocaleRef.current });
    setSettingsFeedback(t(locale, "Unsaved changes reverted.", "未保存的更改已撤销。"));
  };

  return (
    <div className="space-y-4">
      <PageHeader
        title={t(locale, "Settings", "设置")}
        subtitle={t(locale, "General, appearance, connectors, and governance", "通用、外观、连接器与治理")}
        breadcrumb={`${t(locale, "Admin", "管理")} > ${t(locale, "Settings", "设置")}`}
      />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* General */}
        <Panel title={t(locale, "General", "通用")} icon={Sliders}>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs text-ink-secondary">{t(locale, "Language", "语言")}</span>
              <div className="flex items-center gap-1.5">
                <button
                  type="button"
                  data-ui-action="settings_language_en_us"
                  data-ui-skip-action="true"
                  aria-pressed={locale === "en-US"}
                  disabled={!preferencesReady}
                  onClick={() => {
                    setLocale?.("en-US");
                    recordSettingsAction("language_select", { language: "en-US", preview: true });
                  }}
                  className={cn("rounded px-2 py-0.5 text-2xs font-medium", locale === "en-US" ? "bg-accent-light text-accent-dark" : "text-ink-muted hover:bg-surface-sunken")}
                >EN</button>
                <button
                  type="button"
                  data-ui-action="settings_language_zh_cn"
                  data-ui-skip-action="true"
                  aria-pressed={locale === "zh-CN"}
                  disabled={!preferencesReady}
                  onClick={() => {
                    setLocale?.("zh-CN");
                    recordSettingsAction("language_select", { language: "zh-CN", preview: true });
                  }}
                  className={cn("rounded px-2 py-0.5 text-2xs font-medium", locale === "zh-CN" ? "bg-accent-light text-accent-dark" : "text-ink-muted hover:bg-surface-sunken")}
                >中文</button>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-ink-secondary">{t(locale, "Workspace Root", "工作区根目录")}</span>
              <span className="font-mono text-2xs text-ink-muted">{summary?.workspace_root ?? "—"}</span>
            </div>
          </div>
        </Panel>

        {/* Appearance */}
        <Panel title={t(locale, "Appearance", "外观")} icon={Palette}>
          <div className="flex items-center justify-between">
            <span className="text-xs text-ink-secondary">{t(locale, "Theme", "主题")}</span>
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                data-ui-action="settings_theme_light"
                data-ui-skip-action="true"
                aria-pressed={mode === "light"}
                disabled={!preferencesReady}
                onClick={() => selectTheme("light")}
                className={cn("flex min-h-8 items-center gap-1 rounded px-2 py-1 text-xs font-medium transition-colors", mode === "light" ? "bg-accent text-accent-fg" : "text-ink-muted hover:bg-surface-sunken hover:text-ink")}
              >
                <Sun className="h-3 w-3" /> {t(locale, "Light", "浅色")}
              </button>
              <button
                type="button"
                data-ui-action="settings_theme_dark"
                data-ui-skip-action="true"
                aria-pressed={mode === "dark"}
                disabled={!preferencesReady}
                onClick={() => selectTheme("dark")}
                className={cn("flex min-h-8 items-center gap-1 rounded px-2 py-1 text-xs font-medium transition-colors", mode === "dark" ? "bg-accent text-accent-fg" : "text-ink-muted hover:bg-surface-sunken hover:text-ink")}
              >
                <Moon className="h-3 w-3" /> {t(locale, "Dark", "深色")}
              </button>
              <button
                type="button"
                data-ui-action="settings_theme_system"
                data-ui-skip-action="true"
                aria-pressed={mode === "system"}
                disabled={!preferencesReady}
                onClick={() => selectTheme("system")}
                className={cn("flex min-h-8 items-center gap-1 rounded px-2 py-1 text-xs font-medium transition-colors", mode === "system" ? "bg-accent text-accent-fg" : "text-ink-muted hover:bg-surface-sunken hover:text-ink")}
              >
                <Eye className="h-3 w-3" /> {t(locale, "System", "系统")}
              </button>
            </div>
          </div>
          <div className="mt-2 flex items-center justify-between">
            <span className="text-2xs text-ink-muted">{t(locale, "Current", "当前")}</span>
            <StatusBadgeV2 tone="ready" size="xs">
              {mode === "system"
                ? `${t(locale, "System", "跟随系统")} · ${resolvedTheme}`
                : mode === "dark" ? t(locale, "Dark", "深色") : t(locale, "Light", "浅色")}
            </StatusBadgeV2>
          </div>
        </Panel>

        {/* Connectors */}
        <Panel title={t(locale, "Connectors", "连接器")} icon={Globe}>
          <div className="space-y-2">
            {secretEntries.map((entry) => {
              const configured = connectorConfigured(summary, entry.key);
              return (
                <div key={entry.key} className="flex items-center justify-between text-xs">
                  <span className="text-ink-secondary">{entry.label}</span>
                  <StatusBadgeV2 tone={configured ? "verified" : "unknown"} size="xs">
                    {configured ? t(locale, "Configured", "已配置") : t(locale, "Not configured", "未配置")}
                  </StatusBadgeV2>
                </div>
              );
            })}
          </div>
        </Panel>

        {/* Credentials */}
        <Panel title={t(locale, "Credentials", "凭证")} icon={KeyRound}>
          <div className="space-y-2">
            {secretEntries.map((entry) => {
              const configured = connectorConfigured(summary, entry.key);
              return (
                <div key={entry.key} className="flex items-center justify-between text-xs">
                  <span className="text-ink-secondary">{entry.label}</span>
                  <div className="flex items-center gap-1.5">
                    <StatusDot tone={configured ? "verified" : "unknown"} />
                    <span className="text-ink-muted">{configured ? "••••••••" : t(locale, "Not set", "未设置")}</span>
                  </div>
                </div>
              );
            })}
            <div className="mt-2 rounded-sm border border-edge bg-surface-sunken px-2 py-1.5 text-2xs text-ink-muted">
              {t(locale, "Secrets are never displayed. Only configured/not configured status is shown.", "密钥从不显示。仅展示已配置/未配置状态。")}
            </div>
            <div className="mt-2 flex flex-wrap gap-1.5">
              <button
                type="button"
                data-ui-action="test_all_connectors"
                data-ui-skip-action="true"
                onClick={() => recordSettingsAction("test_all_connectors")}
                className="rounded border border-edge px-2 py-1 text-2xs font-medium text-ink-secondary hover:bg-surface-sunken"
              >
                {t(locale, "Test All Connectors", "测试所有连接器")}
              </button>
              <button
                type="button"
                data-ui-action="rotate_credentials_batch"
                data-ui-skip-action="true"
                onClick={() => recordSettingsAction("rotate_credentials_batch")}
                className="rounded border border-edge px-2 py-1 text-2xs font-medium text-ink-secondary hover:bg-surface-sunken"
              >
                {t(locale, "Rotate Credentials", "轮换凭据")}
              </button>
              <button
                type="button"
                data-ui-action="save_settings_changes"
                data-ui-skip-action="true"
                onClick={() => void savePreferences()}
                disabled={!preferencesReady || saving}
                className="rounded bg-accent px-2 py-1 text-2xs font-medium text-accent-fg hover:opacity-90"
              >
                {t(locale, "Save Changes", "保存更改")}
              </button>
              <button
                type="button"
                data-ui-action="cancel_settings_changes"
                data-ui-skip-action="true"
                onClick={cancelPreferences}
                disabled={!preferencesReady}
                className="rounded border border-edge px-2 py-1 text-2xs font-medium text-ink-secondary hover:bg-surface-sunken"
              >
                {t(locale, "Cancel", "取消")}
              </button>
            </div>
            {settingsFeedback ? <p className="mt-2 text-xs text-ink-muted" role="status" aria-live="polite">{settingsFeedback}</p> : null}
          </div>
        </Panel>

        {/* Design Governance */}
        <Panel title={t(locale, "Design Governance", "设计治理")} icon={ShieldCheck} className="lg:col-span-2">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <div className="flex items-center justify-between rounded-sm border border-edge px-2.5 py-2 text-xs">
              <span className="text-ink-secondary">{t(locale, "Claim Boundary", "声明边界")}</span>
              <StatusBadgeV2 tone="verified" size="xs">{t(locale, "Active", "活跃")}</StatusBadgeV2>
            </div>
            <div className="flex items-center justify-between rounded-sm border border-edge px-2.5 py-2 text-xs">
              <span className="text-ink-secondary">{t(locale, "Gate Engine", "门控引擎")}</span>
              <StatusBadgeV2 tone="verified" size="xs">{t(locale, "Active", "活跃")}</StatusBadgeV2>
            </div>
            <div className="flex items-center justify-between rounded-sm border border-edge px-2.5 py-2 text-xs">
              <span className="text-ink-secondary">{t(locale, "Audit Trail", "审计轨迹")}</span>
              <StatusBadgeV2 tone="verified" size="xs">{t(locale, "Active", "活跃")}</StatusBadgeV2>
            </div>
            <div className="flex items-center justify-between rounded-sm border border-edge px-2.5 py-2 text-xs">
              <span className="text-ink-secondary">{t(locale, "Rollback", "回滚")}</span>
              <StatusBadgeV2 tone="verified" size="xs">{t(locale, "Active", "活跃")}</StatusBadgeV2>
            </div>
          </div>
        </Panel>
      </div>
    </div>
  );
}
