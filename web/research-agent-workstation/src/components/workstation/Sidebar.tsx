"use client";

import { useState } from "react";
import { Beaker, CheckCircle2, ChevronRight, Cloud, Menu, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { navItems, type PageId } from "./navigation";

type Locale = "zh-CN" | "en-US";

const navLabelZh: Record<PageId, string> = {
  overview: "科研总览",
  control: "EvoMind 工作站",
  experiments: "实验中心",
  evolution: "自进化引擎",
  data: "数据 / Kaggle",
  report: "报告工作室",
  code: "代码 Agent",
  gpu: "GPU / HPC",
  evidence: "证据台账",
  gates: "完整性 Gate",
  literature: "文献 / RAG",
  tasks: "任务队列",
  runtime: "Agent 运行时",
  workflow: "流程编排",
  settings: "系统设置",
  design: "设计治理"
};
const navSections = [
  { label: "总览", en: "Overview", ids: ["overview"] },
  { label: "科研", en: "Research", ids: ["control", "tasks", "experiments", "evolution", "runtime", "workflow"] },
  { label: "开发", en: "Build", ids: ["code", "report", "literature"] },
  { label: "基础设施", en: "Infrastructure", ids: ["gpu", "data"] },
  { label: "治理", en: "Governance", ids: ["evidence", "gates"] },
  { label: "管理", en: "Admin", ids: ["settings"] }
] as const satisfies Array<{ label: string; en: string; ids: readonly PageId[] }>;

const connectorRows = [
  ["本地 + HPC", "在线", "green"],
  ["Kaggle API", "在线", "green"],
  ["DeepSeek API", "在线", "green"],
  ["人工 Gate", "受控", "amber"]
] as const;

function text(locale: Locale | undefined, en: string, zh: string) {
  return locale === "zh-CN" ? zh : en;
}

export function Sidebar({
  activePage,
  onPageChange,
  onAction,
  locale = "zh-CN"
}: {
  activePage: PageId;
  onPageChange: (page: PageId) => void;
  onAction?: (action: string, metadata?: Record<string, unknown>) => Promise<unknown>;
  locale?: Locale;
}) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const activeItem = navItems.find((item) => item.id === activePage);
  const activeLabel = activeItem ? (locale === "zh-CN" ? navLabelZh[activeItem.id] : activeItem.label) : "Menu";

  function go(page: PageId) {
    onPageChange(page);
    setMobileOpen(false);
    void onAction?.("navigate_page", { page });
  }

  return (
    <aside className="sticky top-0 z-40 w-full bg-[#061326] text-white lg:fixed lg:left-0 lg:top-0 lg:h-screen lg:w-[224px] lg:border-r lg:border-white/10">
      <div className="flex h-full flex-col bg-[linear-gradient(180deg,#061326_0%,#071a32_58%,#061326_100%)] p-3">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-blue-300/30 bg-blue-600 text-xs font-black text-white shadow-[0_14px_32px_-20px_rgba(37,99,235,0.9)]">
            EM
          </div>
          <div className="min-w-0">
            <div className="truncate text-sm font-black">EvoMind</div>
            <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-blue-100/80">XCIENTIST RESEARCH AGENT</div>
          </div>
          <button
            className="ml-auto flex h-9 w-9 items-center justify-center rounded-md border border-white/10 bg-white/10 text-white hover:bg-white/15 lg:hidden"
            data-ui-action="toggle_mobile_navigation"
            data-ui-skip-action="true"
            onClick={() => setMobileOpen((value) => !value)}
            aria-label={mobileOpen ? "Close navigation" : `Open navigation, current page ${activeLabel}`}
          >
            {mobileOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
          </button>
        </div>

        <nav className={cn("dark-scrollbar mt-3 grid grid-cols-2 gap-1 lg:block lg:min-h-0 lg:flex-1 lg:overflow-y-auto lg:pr-1", !mobileOpen && "hidden lg:block")}>
          {navSections.map((section) => (
            <div key={section.label} className="lg:mb-2">
              <div className="mb-0.5 hidden px-2 text-[9px] font-black uppercase tracking-[0.12em] text-blue-100/55 lg:block">
                {locale === "zh-CN" ? section.label : section.en}
              </div>
              <div className="space-y-1">
                {section.ids.map((id) => {
                  const item = navItems.find((candidate) => candidate.id === id);
                  if (!item) return null;
                  const Icon = item.icon;
                  const active = item.id === activePage;
                  const label = locale === "zh-CN" ? navLabelZh[item.id] : item.label;
                  return (
                    <Button
                      key={item.id}
                      variant="ghost"
                      className={cn(
                        "group h-8 w-full justify-start gap-2 rounded-md border border-transparent px-2.5 text-[13px] font-bold text-blue-50/86 shadow-none transition hover:border-white/10 hover:bg-white/10 hover:text-white focus-visible:ring-white/70",
                        active && "border-blue-400/30 bg-blue-600 text-white shadow-[0_12px_26px_-20px_rgba(37,99,235,0.9)] hover:bg-blue-600"
                      )}
                      onClick={() => go(item.id)}
                      data-testid={`nav-${item.id}`}
                      data-ui-skip-action="true"
                    >
                      <Icon className={cn("h-4 w-4 shrink-0", active ? "text-white" : "text-blue-100/75 group-hover:text-white")} />
                      <span className="min-w-0 truncate text-left">{label}</span>
                    </Button>
                  );
                })}
              </div>
            </div>
          ))}
        </nav>

        <div className={cn("mt-3 shrink-0 space-y-2 lg:block", !mobileOpen && "hidden lg:block")}>
          <div className="rounded-md border border-white/12 bg-white/[0.07] p-2.5 shadow-[inset_0_1px_0_rgba(255,255,255,0.08)]">
            <div className="flex items-center justify-between gap-2">
              <div className="flex min-w-0 items-center gap-2">
                <Cloud className="h-4 w-4 shrink-0 text-blue-100" />
                <span className="truncate text-xs font-black text-blue-50">{text(locale, "RESOURCE STATUS", "资源与连接")}</span>
              </div>
              <span className="h-2 w-2 rounded-full bg-emerald-400" />
            </div>
            <div className="mt-2 space-y-1">
              {connectorRows.map(([name, state, tone]) => (
                <div key={name} className="flex items-center justify-between gap-2 text-xs">
                  <span className="truncate text-blue-50/86">{name}</span>
                  <span className={cn("font-black", tone === "amber" ? "text-amber-300" : "text-emerald-300")}>{state}</span>
                </div>
              ))}
            </div>
            <button
              className="mt-2 flex w-full items-center justify-between rounded-md border border-white/12 bg-white/[0.06] px-2 py-1.5 text-left text-xs font-black text-blue-50 hover:bg-white/10"
              onClick={() => go("settings")}
              data-ui-action="open_connector_settings"
              data-ui-skip-action="true"
            >
              {text(locale, "Connector Settings", "资源与连接管理")}
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>

          <button
            className="flex w-full items-center gap-2 rounded-md border border-emerald-300/20 bg-emerald-300/10 p-2 text-left hover:bg-emerald-300/15"
            onClick={() => go("gates")}
            data-ui-action="open_research_mode_gates"
            data-ui-skip-action="true"
          >
            <Beaker className="h-4 w-4 shrink-0 text-emerald-200" />
            <div className="min-w-0">
              <div className="text-xs font-black">{text(locale, "Research Mode", "科研模式")}</div>
              <div className="text-[11px] font-bold text-emerald-300">{text(locale, "Human-gated execution", "人工 Gate 受控")}</div>
            </div>
            <CheckCircle2 className="ml-auto h-4 w-4 text-emerald-300" />
          </button>
          <div className="grid grid-cols-[1fr_auto] gap-2 text-[11px] font-semibold text-blue-100/70">
            <span>v2.3.1</span>
            <span>Asia/Shanghai</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
