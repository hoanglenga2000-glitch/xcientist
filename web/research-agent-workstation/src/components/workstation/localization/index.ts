/**
 * EvoMind Research OS — Centralized Localization
 *
 * All user-facing text goes through this module.
 * API field names, run IDs, metric names stay in English.
 * Business terms use consistent translations per locale.
 */

export type Locale = "zh-CN" | "en-US";

/* ── Navigation & Sections ── */
const navLabels: Record<Locale, Record<string, string>> = {
  "zh-CN": {
    "command_center": "指挥中心",
    "research_loop": "研究循环",
    "workbench": "工作台",
    "infrastructure": "基础设施",
    "governance": "治理",
    "admin": "管理",
    "assistant": "智能助手",
    "overview": "科研总览",
    "control": "EvoMind 工作站",
    "tasks": "任务队列",
    "experiments": "实验中心",
    "evolution": "自进化引擎",
    "workflow": "流程编排",
    "runtime": "Agent 运行时",
    "data": "数据 / Kaggle",
    "code": "代码 Agent IDE",
    "literature": "文献 / RAG",
    "report": "报告工作室",
    "gpu": "GPU / HPC",
    "evidence": "证据台账",
    "gates": "完整性 Gate",
    "settings": "系统设��",
    "design": "设计治理",
  },
  "en-US": {
    "command_center": "Command Center",
    "research_loop": "Research Loop",
    "workbench": "Workbench",
    "infrastructure": "Infrastructure",
    "governance": "Governance",
    "admin": "Admin",
    "assistant": "Assistant",
    "overview": "Research Overview",
    "control": "EvoMind Gateway",
    "tasks": "Task Queue",
    "experiments": "Experiment Ledger",
    "evolution": "Evolution Engine",
    "workflow": "Workflow Graph",
    "runtime": "Agent Runtime",
    "data": "Data / Kaggle",
    "code": "Code Agent IDE",
    "literature": "Literature / RAG",
    "report": "Report Studio",
    "gpu": "GPU / HPC",
    "evidence": "Evidence Ledger",
    "gates": "Integrity Gates",
    "settings": "Settings",
    "design": "Design System",
  },
};

/* ── Status labels ── */
const statusLabels: Record<Locale, Record<string, string>> = {
  "zh-CN": {
    "assistant": "自然语言对话、研究规划与受控执行入口",
    "verified": "已验证",
    "ready": "就绪",
    "running": "运行中",
    "pending": "待确认",
    "blocked": "阻断",
    "failed": "失败",
    "unknown": "未知",
    "stale": "过期",
    "draft": "草稿",
    "configured": "已配置",
    "not_configured": "未配置",
    "passed": "通过",
    "approved": "已批准",
    "rejected": "已拒绝",
    "waiting": "等待中",
    "completed": "已完成",
    "queued": "排队中",
    "live": "活跃",
  },
  "en-US": {
    "assistant": "Natural-language chat, research planning, and controlled execution",
    "verified": "Verified",
    "ready": "Ready",
    "running": "Running",
    "pending": "Pending",
    "blocked": "Blocked",
    "failed": "Failed",
    "unknown": "Unknown",
    "stale": "Stale",
    "draft": "Draft",
    "configured": "Configured",
    "not_configured": "Not Configured",
    "passed": "Passed",
    "approved": "Approved",
    "rejected": "Rejected",
    "waiting": "Waiting",
    "completed": "Completed",
    "queued": "Queued",
    "live": "Live",
  },
};

/* ── Common actions ── */
const actionLabels: Record<Locale, Record<string, string>> = {
  "zh-CN": {
    "search": "搜索",
    "refresh": "刷新",
    "export": "导出",
    "download": "下载",
    "copy": "复制",
    "view_all": "查看全部",
    "review": "审核",
    "approve": "批准",
    "reject": "拒绝",
    "apply": "应用",
    "cancel": "取消",
    "close": "关闭",
    "expand": "展开",
    "collapse": "收起",
    "filter": "筛选",
    "reset": "重置",
    "save": "保存",
    "run": "运行",
    "submit": "提交",
    "no_data": "暂无数据",
    "loading": "加载中...",
    "error": "加载失败",
    "ai_generated": "AI 生成 / 草稿",
    "evidence_rail": "证据轨",
    "resource_status": "资源与连接",
    "connector_settings": "资源与连接管理",
    "research_mode": "科研模式",
    "human_gated": "人工 Gate 受控",
    "review_now": "立���审核",
  },
  "en-US": {
    "search": "Search",
    "refresh": "Refresh",
    "export": "Export",
    "download": "Download",
    "copy": "Copy",
    "view_all": "View All",
    "review": "Review",
    "approve": "Approve",
    "reject": "Reject",
    "apply": "Apply",
    "cancel": "Cancel",
    "close": "Close",
    "expand": "Expand",
    "collapse": "Collapse",
    "filter": "Filter",
    "reset": "Reset",
    "save": "Save",
    "run": "Run",
    "submit": "Submit",
    "no_data": "No data available",
    "loading": "Loading...",
    "error": "Failed to load",
    "ai_generated": "AI-generated / Draft",
    "evidence_rail": "Evidence Rail",
    "resource_status": "Resource Status",
    "connector_settings": "Connector Settings",
    "research_mode": "Research Mode",
    "human_gated": "Human-gated execution",
    "review_now": "Review Now",
  },
};

/* ── Page subtitles ── */
const pageSubtitles: Record<Locale, Record<string, string>> = {
  "zh-CN": {
    "overview": "科研工作站运行态势与闭环总控",
    "control": "调度 Agent、资源与门禁",
    "experiments": "实验台账、分支与分数门禁",
    "evolution": "搜索图、检索记忆与分支扩展规划",
    "data": "数据审计、提交结构与排行榜证据",
    "report": "AI 生成报告、证据链与风险审计",
    "code": "可审计代码生成、Diff、终端与门禁",
    "gpu": "远程算力、作业 manifest 与产物回传",
    "evidence": "统一归档 artifact、日志、指标、报告和审计证据",
    "gates": "代码、算力、提交与报告门禁",
    "literature": "文献检索、RAG 资料库与研究上下文",
    "tasks": "任务配置、上下文与运行入口",
    "runtime": "模型、缓存、成本与调度状态",
    "workflow": "多 Agent 工作流、回退与证据交接",
    "settings": "账号、语言、主题、凭据与资源偏好",
    "design": "Research OS 视觉 token、组件规则与页面基线",
  },
  "en-US": {
    "overview": "Research workstation operating status and mission control",
    "control": "Orchestrate agents, resources, and gates",
    "experiments": "Experiment ledger, branches, and score gates",
    "evolution": "Search graph, retrospective memory, and branch expansion planning",
    "data": "Data audit, submission schema, and leaderboard evidence",
    "report": "AI-generated reports with evidence and risk audit",
    "code": "Auditable code generation, diff, terminal, and gates",
    "gpu": "Remote compute, job manifests, and artifact pullback",
    "evidence": "Artifacts, logs, metrics, reports, and audit evidence",
    "gates": "Code, compute, submission, and report gates",
    "literature": "Literature retrieval, RAG, and research context",
    "tasks": "Task specs, context, and run entry points",
    "runtime": "Models, cache, cost, and scheduler status",
    "workflow": "Multi-agent workflow, fallback, and evidence handoff",
    "settings": "Account, language, theme, credentials, and resources",
    "design": "Visual tokens, components, and page baselines",
  },
};

/* ── Lookup helpers ── */
export function nav(locale: Locale | undefined, key: string): string {
  return navLabels[locale ?? "zh-CN"]?.[key] ?? navLabels["en-US"]?.[key] ?? key;
}

export function status(locale: Locale | undefined, key: string): string {
  return statusLabels[locale ?? "zh-CN"]?.[key] ?? statusLabels["en-US"]?.[key] ?? key;
}

export function action(locale: Locale | undefined, key: string): string {
  return actionLabels[locale ?? "zh-CN"]?.[key] ?? actionLabels["en-US"]?.[key] ?? key;
}

export function subtitle(locale: Locale | undefined, pageId: string): string {
  return pageSubtitles[locale ?? "zh-CN"]?.[pageId] ?? pageSubtitles["en-US"]?.[pageId] ?? "";
}

/** Shorthand: pick zh or en for a simple string pair */
export function t(locale: Locale | undefined, en: string, zh: string): string {
  return locale === "en-US" ? en : zh;
}
