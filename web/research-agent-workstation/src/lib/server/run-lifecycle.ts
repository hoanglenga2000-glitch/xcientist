export type RunLifecycleState =
  | "CREATED"
  | "PLANNING"
  | "WAIT_PLAN_GATE"
  | "APPROVED"
  | "EXECUTING"
  | "WAIT_RESULT_GATE"
  | "REPORTING"
  | "COMPLETED"
  | "FAILED";

const ALLOWED_TRANSITIONS: Record<RunLifecycleState, ReadonlySet<RunLifecycleState>> = {
  CREATED: new Set(["PLANNING", "FAILED"]),
  PLANNING: new Set(["WAIT_PLAN_GATE", "FAILED"]),
  WAIT_PLAN_GATE: new Set(["APPROVED", "FAILED"]),
  APPROVED: new Set(["EXECUTING", "FAILED"]),
  EXECUTING: new Set(["WAIT_RESULT_GATE", "FAILED"]),
  WAIT_RESULT_GATE: new Set(["REPORTING", "FAILED"]),
  REPORTING: new Set(["COMPLETED", "FAILED"]),
  COMPLETED: new Set(),
  FAILED: new Set(),
};

export function assertRunTransition(from: RunLifecycleState, to: RunLifecycleState) {
  if (!ALLOWED_TRANSITIONS[from].has(to)) {
    throw new Error(`Illegal run transition: ${from} -> ${to}`);
  }
}

export function canExecuteRun(state: string, planGateDecision: string) {
  return state === "APPROVED" && planGateDecision === "approved";
}
