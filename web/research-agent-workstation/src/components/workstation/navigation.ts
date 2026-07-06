import {
  Activity,
  Beaker,
  BookOpen,
  Bot,
  Boxes,
  CheckCircle2,
  Database,
  FileText,
  FlaskConical,
  GitBranch,
  ListChecks,
  ScrollText,
  Server,
  Settings,
  ShieldCheck,
  TerminalSquare
} from "lucide-react";

export type PageId =
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

export const navItems = [
  { id: "overview", label: "Research Overview", icon: Boxes },
  { id: "control", label: "AI Control", icon: Bot },
  { id: "experiments", label: "Experiments", icon: FlaskConical },
  { id: "evolution", label: "Evolution Engine", icon: Beaker },
  { id: "data", label: "Data & Kaggle", icon: Database },
  { id: "report", label: "Report Studio", icon: BookOpen },
  { id: "code", label: "Code Agent", icon: TerminalSquare },
  { id: "gpu", label: "GPU / HPC", icon: Server },
  { id: "evidence", label: "Evidence Ledger", icon: FileText },
  { id: "gates", label: "Integrity Gates", icon: ShieldCheck },
  { id: "literature", label: "Literature", icon: ScrollText },
  { id: "tasks", label: "Task Queue", icon: ListChecks },
  { id: "runtime", label: "Agent Runtime", icon: Activity },
  { id: "workflow", label: "Workflow Graph", icon: GitBranch },
  { id: "settings", label: "Settings", icon: Settings }
] as const satisfies Array<{
  id: PageId;
  label: string;
  icon: typeof CheckCircle2;
}>;

export const pageTitles: Record<PageId, string> = {
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
