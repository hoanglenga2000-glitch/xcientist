export type ScientificReportGenerationState = "running" | "ready" | "failed" | null;
export type MissingScientificReportState = "pending" | "generation_failed" | "not_found" | "integrity_failed";

const ACTIVE_REVIEW_STATES = new Set([
  "awaiting_human_gate",
  "approved",
  "starting",
  "running",
  "needs_continuation",
]);
const TERMINAL_FAILURE_STATES = new Set(["cancelled", "failed"]);

/** Classify only the absence of a report manifest; evidence errors are always integrity failures. */
export function classifyMissingScientificReport(
  runStatus: string,
  generationStatus: ScientificReportGenerationState,
): MissingScientificReportState {
  if (generationStatus === "failed") return "generation_failed";
  if (generationStatus === "running") return "pending";
  if (generationStatus === "ready") return "integrity_failed";
  if (ACTIVE_REVIEW_STATES.has(runStatus)) return "pending";
  if (TERMINAL_FAILURE_STATES.has(runStatus)) return "generation_failed";
  if (runStatus === "completed") return "not_found";
  return "integrity_failed";
}
