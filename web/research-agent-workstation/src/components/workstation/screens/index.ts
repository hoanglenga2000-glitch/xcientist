/**
 * Screen wrappers — V2 design shell around the original page components.
 * Each wrapper adds PageHeader, proper breadcrumbs, and responsive layout.
 */

export { OverviewScreen } from "./OverviewScreen";

// Re-export original screens (will be wrapped incrementally)
export {
  MissionControl,
  OverviewBoardEnhanced,
  DataKagglePipeline,
  CodeRunner,
  GpuHpcConsole,
  EvidenceLedger,
  EvidenceDetail,
  ReportStudio,
  LiteratureKnowledge,
  AgentRuntime,
  IntegrityGates,
  Experiments,
  ResearchTasks,
  WorkflowGraph,
  SettingsCenter,
  DesignSystem,
} from "../Screens";
