import {
  Activity,
  Beaker,
  BookOpen,
  Bot,
  Boxes,
  CheckCircle2,
  Code2,
  Database,
  FileText,
  FlaskConical,
  GitBranch,
  ListChecks,
  ScrollText,
  Server,
  Settings,
  ShieldCheck,
  MessageSquareText
} from "lucide-react";

export type PageId =
  | "assistant"
  | "projects"
  | "tasks"
  | "data"
  | "gpu"
  | "evidence"
  | "literature"
  | "workflow"
  | "code"
  | "runtime"
  | "experiments"
  | "report"
  | "gates"
  | "settings"
  | "design"
  | "overview"
  | "evolution"
  | "control";

/** Nav items with new IA grouping per the V2 spec */
export const navSections = [
  {
    id: "core",
    label: "Workspace",
    labelZh: "工作台",
    ids: ["assistant", "projects", "data", "report", "settings"] as const,
  },
  {
    id: "advanced",
    label: "Advanced tools",
    labelZh: "高级工具",
    ids: ["overview", "control", "tasks", "experiments", "literature", "code", "gpu", "workflow", "evolution", "runtime", "evidence", "gates"] as const,
  },
] as const;

export function navSectionCollapsed(sectionId: string, activeSectionId: string | undefined, sidebarCollapsed: boolean, preference?: boolean): boolean {
  if (sectionId === "core" || sidebarCollapsed) return false;
  return preference ?? sectionId !== activeSectionId;
}

/** Flat nav items for backward compat */
export const navItems = [
  { id: "assistant", label: "Assistant", icon: MessageSquareText },
  { id: "projects", label: "Projects and experiments", icon: FlaskConical },
  { id: "overview", label: "Research Overview", icon: Boxes },
  { id: "control", label: "AI Control", icon: Bot },
  { id: "tasks", label: "Task Queue", icon: ListChecks },
  { id: "experiments", label: "Experiments", icon: FlaskConical },
  { id: "evolution", label: "Evolution Engine", icon: Beaker },
  { id: "workflow", label: "Workflow Graph", icon: GitBranch },
  { id: "runtime", label: "Agent Runtime", icon: Activity },
  { id: "data", label: "Data & Kaggle", icon: Database },
  { id: "code", label: "Code Agent", icon: Code2 },
  { id: "literature", label: "Literature", icon: ScrollText },
  { id: "report", label: "Report Studio", icon: BookOpen },
  { id: "gpu", label: "GPU / HPC", icon: Server },
  { id: "evidence", label: "Evidence Ledger", icon: FileText },
  { id: "gates", label: "Integrity Gates", icon: ShieldCheck },
  { id: "settings", label: "Settings", icon: Settings },
] as const satisfies Array<{
  id: PageId;
  label: string;
  icon: typeof CheckCircle2;
}>;

export const pageTitles: Record<PageId, string> = {
  assistant: "DeepEvo Assistant",
  projects: "Projects and experiments",
  overview: "Research Overview",
  control: "AI Control Console",
  tasks: "Task Research Workspace",
  data: "Data & Kaggle Pipeline",
  gpu: "GPU / HPC Console",
  evidence: "Evidence Ledger",
  literature: "Literature & Knowledge Layer",
  workflow: "Research Workflow Graph",
  code: "Code Agent IDE",
  runtime: "Agent Runtime",
  experiments: "Experiments",
  evolution: "Evolution Engine",
  report: "Report Studio",
  gates: "Integrity Gates",
  settings: "Settings",
  design: "Design System"
};

/** Map a route alias to canonical PageId (backward compat) */
export const routeAliases: Record<string, PageId> = {
  chat: "assistant",
  ask: "assistant",
  mission: "overview",
  "evidence-detail": "evidence",
  "ai-control": "control",
  "experiment-ledger": "experiments",
  "gpu-hpc": "gpu",
  "data-kaggle": "data",
  "literature-rag": "literature",
  "code-agent": "code",
  "agent-runtime": "runtime",
  "integrity-gates": "gates",
  "report-studio": "report",
  "workflow-graph": "workflow",
  "evolution-engine": "evolution",
  "task-queue": "tasks",
};

export function resolvePageId(raw: string): PageId {
  return routeAliases[raw] ?? (raw as PageId);
}
