"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Image from "next/image";
import {
  ChevronRight,
  Menu,
  X,
  Cloud,
  ShieldCheck,
  PanelLeftClose,
  PanelLeftOpen,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { navSections, navItems, type PageId } from "./navigation";
import { StatusDot, type StatusTone } from "./primitives/StatusBadge";
import type { WorkstationSummary } from "@/lib/api/types";
import { t } from "./localization";

type Locale = "zh-CN" | "en-US";

/* ── Unified page names (single source: navigation.ts labelZh) ── */
const navLabelZh: Record<PageId, string> = {
  assistant: "智能助手",
  overview: "科研总览",
  control: "EvoMind 工作站",
  experiments: "实验中心",
  evolution: "自进化引擎",
  data: "数据 / Kaggle",
  report: "报告工作室",
  code: "代码 Agent",
  gpu: "GPU / HPC",
  evidence: "证据台账",
  gates: "完整性门禁",
  literature: "文献 / RAG",
  tasks: "任务队列",
  runtime: "Agent 运行时",
  workflow: "流程编排",
  settings: "系统设置",
  design: "设计治理",
};

function connectorStatus(summary: WorkstationSummary | null | undefined, key: string): { configured: boolean; state: string; tone: StatusTone } {
  const connectors = summary?.connector_status as Record<string, Record<string, unknown>> | undefined;
  const entry = connectors?.[key];
  if (!entry) return { configured: false, state: "unknown", tone: "unknown" };
  const configured = Boolean(entry.configured);
  const state = String(entry.state ?? entry.status ?? "").toLowerCase();
  const tone: StatusTone = state.includes("blocked") || state.includes("failed")
    ? "blocked"
    : configured && (state.includes("verified") || state.includes("ready") || state.includes("passed"))
    ? "verified"
    : configured
    ? "ready"
    : state.includes("not_configured")
    ? "unknown"
    : "pending";
  return { configured, state, tone };
}

function connectorLabel(locale: Locale | undefined, cfg: { configured: boolean; state: string }): string {
  if (cfg.state.includes("blocked") || cfg.state.includes("failed")) return t(locale, "Blocked", "阻断");
  if (cfg.configured && cfg.state.includes("verified")) return t(locale, "Verified", "已验证");
  if (cfg.configured && (cfg.state.includes("ready") || cfg.state.includes("passed"))) return t(locale, "Ready", "就绪");
  if (cfg.configured) return t(locale, "Configured", "已配置");
  if (cfg.state.includes("not_configured")) return t(locale, "Not configured", "未配置");
  return t(locale, "Unknown", "未知");
}

/* ── Shared nav tree (used by desktop sidebar + mobile drawer) ── */
function NavTree({
  activePage,
  locale,
  collapsed,
  idPrefix,
  onNavigate,
}: {
  activePage: PageId;
  locale: Locale;
  collapsed: boolean;
  idPrefix: "desktop" | "mobile";
  onNavigate: (page: PageId) => void;
}) {
  const [collapsedSections, setCollapsedSections] = useState<Record<string, boolean>>({});
  const activeSectionId = navSections.find((s) => s.ids.includes(activePage as never))?.id;
  const isSectionCollapsed = (sectionId: string) => {
    if (collapsed) return false;
    if (sectionId === activeSectionId) return false;
    return collapsedSections[sectionId] ?? true;
  };

  return (
    <nav aria-label={t(locale, "Primary navigation", "主导航")} className="dark-scrollbar mt-3 min-h-0 flex-1 space-y-1 overflow-y-auto pr-1">
      {navSections.map((section) => {
        const isCollapsed = isSectionCollapsed(section.id);
        return (
          <div key={section.id}>
            {!collapsed && (
              <button
                type="button"
                className="mb-1 flex min-h-8 w-full items-center gap-1 rounded px-2 py-1 text-xs font-medium text-ink-muted/80 hover:bg-surface-raised/5 hover:text-ink-secondary"
                onClick={() => setCollapsedSections((prev) => ({ ...prev, [section.id]: !isCollapsed }))}
                aria-expanded={!isCollapsed}
                aria-controls={`${idPrefix}-nav-section-${section.id}`}
              >
                <ChevronRight className={cn("h-3 w-3 transition-transform duration-150", !isCollapsed && "rotate-90")} />
                {locale === "zh-CN" ? section.labelZh : section.label}
              </button>
            )}
            <div id={`${idPrefix}-nav-section-${section.id}`} className={cn("space-y-0.5", isCollapsed && "hidden")}>
              {section.ids.map((id) => {
                const item = navItems.find((c) => c.id === id);
                if (!item) return null;
                const Icon = item.icon;
                const active = item.id === activePage;
                const label = locale === "zh-CN" ? navLabelZh[item.id] : item.label;
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => onNavigate(item.id)}
                    data-testid={`nav-${item.id}`}
                    data-ui-skip-action="true"
                    aria-current={active ? "page" : undefined}
                    aria-label={collapsed ? label : undefined}
                    className={cn(
                      "ui-tooltip group relative flex h-10 w-full items-center gap-2 rounded-md border border-transparent px-2.5 text-[13px] font-medium text-ink-secondary transition-colors duration-150 hover:border-edge hover:bg-surface-raised/8 hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/70",
                      collapsed && "justify-center px-0",
                      active && "border-accent/35 bg-accent-light text-accent-dark hover:bg-accent-light"
                    )}
                  >
                    <Icon className={cn("h-4 w-4 shrink-0", active ? "text-accent-dark" : "text-ink-muted group-hover:text-ink")} />
                    {!collapsed && <span className="min-w-0 truncate text-left">{label}</span>}
                    {collapsed && <span className="ui-tooltip-text" role="tooltip">{label}</span>}
                    {active && collapsed && <span className="absolute right-1 top-1 h-1.5 w-1.5 rounded-full bg-surface-raised" aria-hidden="true" />}
                  </button>
                );
              })}
            </div>
          </div>
        );
      })}
    </nav>
  );
}

function BrandMark({ collapsed }: { collapsed: boolean }) {
  return (
    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-edge/80 bg-surface-sunken shadow-raised ring-1 ring-inset ring-accent/10">
      <Image
        src="/brand/evomind-mark-symbol.png"
        alt={collapsed ? "EvoMind" : ""}
        width={40}
        height={40}
        className="h-[34px] w-[34px] object-contain drop-shadow-sm"
        priority
      />
    </div>
  );
}

function BrandText() {
  return (
    <div className="min-w-0">
      <div className="truncate text-sm font-bold">EvoMind</div>
      <div className="text-[11px] font-medium tracking-wide text-ink-muted">Scientific Research OS</div>
    </div>
  );
}

function SidebarFooter({ locale, summary, onNavigate }: { locale: Locale; summary?: WorkstationSummary | null; onNavigate: (p: PageId) => void }) {
  const connectors = [
    { key: "gpu", label: t(locale, "Remote GPU", "远程 GPU") },
    { key: "kaggle", label: "Kaggle API" },
    { key: "deepseek", label: "DeepSeek API" },
  ];
  return (
    <div className="mt-3 shrink-0 space-y-2">
      <div className="rounded-md border border-edge bg-surface-raised/[0.05] p-2">
        <div className="flex items-center gap-2">
          <Cloud className="h-3.5 w-3.5 shrink-0 text-ink-muted" />
          <span className="truncate text-2xs font-bold text-ink">{t(locale, "Resource Status", "资源与连接")}</span>
        </div>
        <div className="mt-1.5 space-y-1">
          {connectors.map((conn) => {
            const status = connectorStatus(summary, conn.key);
            return (
              <div key={conn.key} className="flex items-center justify-between gap-2 text-2xs">
                <span className="truncate text-ink-secondary/80">{conn.label}</span>
                <span className="flex items-center gap-1">
                  <StatusDot tone={status.tone} />
                  <span className={cn("font-semibold", status.tone === "verified" ? "text-success" : status.tone === "blocked" ? "text-danger" : status.tone === "unknown" ? "text-ink-muted/50" : "text-warning")}>
                    {connectorLabel(locale, status)}
                  </span>
                </span>
              </div>
            );
          })}
        </div>
        <button
          type="button"
          className="mt-1.5 flex min-h-8 w-full items-center justify-between rounded-md border border-edge bg-surface-raised/[0.05] px-2 py-1 text-left text-xs font-medium text-ink hover:bg-surface-raised/10"
          onClick={() => onNavigate("settings")}
          data-ui-action="open_connector_settings"
          data-ui-skip-action="true"
        >
          {t(locale, "Connector Settings", "资源与连接管理")}
          <ChevronRight className="h-3.5 w-3.5" />
        </button>
      </div>

      <button
        type="button"
        className="flex w-full items-center gap-2 rounded-md border border-success/20 bg-success/10 p-2 text-left hover:bg-success/15"
        onClick={() => onNavigate("gates")}
        data-ui-action="open_research_mode_gates"
        data-ui-skip-action="true"
      >
        <ShieldCheck className="h-4 w-4 shrink-0 text-success-text" />
        <div className="min-w-0">
          <div className="text-xs font-semibold">{t(locale, "Research Mode", "科研模式")}</div>
          <div className="text-xs text-success">{t(locale, "Human-gated execution", "人工 Gate 受控")}</div>
        </div>
      </button>

      <div className="grid grid-cols-[1fr_auto] gap-2 text-xs font-medium text-ink-muted/70">
        <span>v{process.env.NEXT_PUBLIC_EVOMIND_VERSION ?? "0.0.0"}</span>
        <span>Asia/Shanghai</span>
      </div>
    </div>
  );
}

/* ── Mobile drawer with focus trap, Esc, backdrop, scroll lock, focus restore ── */
function MobileDrawer({
  open,
  onClose,
  returnFocusRef,
  activePage,
  locale,
  summary,
  onNavigate,
}: {
  open: boolean;
  onClose: () => void;
  returnFocusRef: { current: HTMLElement | null };
  activePage: PageId;
  locale: Locale;
  summary?: WorkstationSummary | null;
  onNavigate: (p: PageId) => void;
}) {
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const restoreTarget = returnFocusRef.current ?? (document.activeElement as HTMLElement | null);
    document.body.setAttribute("data-nav-lock", "true");
    const panel = panelRef.current;
    const focusables = () => Array.from(panel?.querySelectorAll<HTMLElement>("button, a, input, select, textarea, [tabindex]:not([tabindex='-1'])") ?? []).filter((el) => !el.hasAttribute("disabled"));
    focusables()[0]?.focus();

    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
        return;
      }
      if (e.key !== "Tab") return;
      const items = focusables();
      if (items.length === 0) return;
      const first = items[0];
      const last = items[items.length - 1];
      const active = document.activeElement as HTMLElement | null;
      if (e.shiftKey) {
        if (active === first || !panel?.contains(active)) {
          e.preventDefault();
          last.focus();
        }
      } else if (active === last || !panel?.contains(active)) {
        e.preventDefault();
        first.focus();
      }
    }
    document.addEventListener("keydown", onKeyDown, true);
    return () => {
      document.removeEventListener("keydown", onKeyDown, true);
      document.body.removeAttribute("data-nav-lock");
      restoreTarget?.focus?.();
    };
  }, [open, onClose, returnFocusRef]);

  if (!open) return null;
  return (
    <>
      <div className="shell-drawer-backdrop lg:hidden" onClick={onClose} aria-hidden="true" />
      <div
        id="mobile-navigation-dialog"
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-label={t(locale, "Navigation menu", "导航菜单")}
        className="shell-drawer flex flex-col bg-frame p-3 text-white lg:hidden"
      >
        <div className="flex items-center gap-2.5">
          <BrandMark collapsed={false} />
          <BrandText />
          <button
            type="button"
            className="ml-auto flex h-11 w-11 items-center justify-center rounded-md border border-white/10 bg-surface-raised/10 text-white hover:bg-surface-raised/15"
            data-ui-action="toggle_mobile_navigation"
            data-ui-skip-action="true"
            onClick={onClose}
            aria-label={t(locale, "Close navigation", "关闭导航")}
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <NavTree activePage={activePage} locale={locale} collapsed={false} idPrefix="mobile" onNavigate={onNavigate} />
        <SidebarFooter locale={locale} summary={summary} onNavigate={onNavigate} />
      </div>
    </>
  );
}

/* ── Sidebar (desktop grid track + mobile top bar + drawer) ── */
export function Sidebar({
  activePage,
  onPageChange,
  onAction,
  locale = "zh-CN",
  summary,
  desktopCollapsed = false,
  onToggleDesktop,
}: {
  activePage: PageId;
  onPageChange: (page: PageId) => void;
  onAction?: (action: string, metadata?: Record<string, unknown>) => Promise<unknown>;
  locale?: Locale;
  summary?: WorkstationSummary | null;
  desktopCollapsed?: boolean;
  onToggleDesktop?: () => void;
}) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const mobileTriggerRef = useRef<HTMLButtonElement>(null);
  const activeItem = navItems.find((item) => item.id === activePage);
  const activeLabel = activeItem ? (locale === "zh-CN" ? navLabelZh[activeItem.id] : activeItem.label) : "Menu";

  const closeMobile = useCallback(() => setMobileOpen(false), []);

  function go(page: PageId) {
    onPageChange(page);
    setMobileOpen(false);
    void onAction?.("navigate_page", { page });
  }

  return (
    <>
      {/* Mobile top bar (<1024px) */}
      <div className="sticky top-0 z-sidebar flex items-center gap-2.5 border-b border-frame-border bg-frame p-3 text-white lg:hidden">
        <BrandMark collapsed={false} />
        <BrandText />
        <button
          ref={mobileTriggerRef}
          type="button"
          className="ml-auto flex h-11 w-11 items-center justify-center rounded-md border border-white/10 bg-surface-raised/10 text-white hover:bg-surface-raised/15"
          data-ui-action="toggle_mobile_navigation"
          data-ui-skip-action="true"
          onClick={() => setMobileOpen(true)}
          aria-label={t(locale, "Open navigation", "打开导航") + `, ${t(locale, "current page", "当前页")} ${activeLabel}`}
          aria-haspopup="dialog"
          aria-expanded={mobileOpen}
          aria-controls="mobile-navigation-dialog"
        >
          <Menu className="h-4 w-4" />
        </button>
      </div>

      <MobileDrawer
        open={mobileOpen}
        onClose={closeMobile}
        returnFocusRef={mobileTriggerRef}
        activePage={activePage}
        locale={locale}
        summary={summary}
        onNavigate={go}
      />

      {/* Desktop sidebar (real grid track, ≥1024px) */}
      <aside className="shell-sidebar hidden flex-col bg-frame p-3 text-white lg:flex">
        <div className={cn("flex items-center gap-2.5", desktopCollapsed && "flex-col gap-2")}>
          <BrandMark collapsed={desktopCollapsed} />
          {!desktopCollapsed && <BrandText />}
          <button
            type="button"
            className={cn("ui-tooltip flex h-9 w-9 items-center justify-center rounded-md border border-white/10 bg-surface-raised/10 text-white hover:bg-surface-raised/15", !desktopCollapsed && "ml-auto")}
            data-ui-action="toggle_desktop_sidebar"
            data-ui-skip-action="true"
            onClick={onToggleDesktop}
            aria-label={desktopCollapsed ? t(locale, "Expand sidebar", "展开侧栏") : t(locale, "Collapse sidebar", "收起侧栏")}
          >
            {desktopCollapsed ? <PanelLeftOpen className="h-4 w-4" /> : <PanelLeftClose className="h-4 w-4" />}
            <span className="ui-tooltip-text" role="tooltip">{desktopCollapsed ? t(locale, "Expand sidebar", "展开侧栏") : t(locale, "Collapse sidebar", "收起侧栏")}</span>
          </button>
        </div>

        <NavTree activePage={activePage} locale={locale} collapsed={desktopCollapsed} idPrefix="desktop" onNavigate={go} />

        {!desktopCollapsed && <SidebarFooter locale={locale} summary={summary} onNavigate={go} />}
      </aside>
    </>
  );
}
