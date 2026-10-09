export type ExecutionBudget = {
  study: string; revision: number; gpu_limit_seconds: number; engineering_limit_seconds: number;
  gpu_remaining_seconds: number; engineering_remaining_seconds: number;
  committed_seconds: number; engineering_committed_seconds: number; charged_wall_seconds: number;
  pending_reserved_seconds: number; operations: number; reconciliation_required: boolean;
};
export type BudgetResponse = {
  ok: boolean; can_manage: boolean; budget: ExecutionBudget; max_limit_seconds: number;
  history: Array<{ revision: number; gpu_limit: number; engineering_limit: number;
    previous_gpu_limit: number; previous_engineering_limit: number; owner_id: string; reason: string; created: number }>;
};

export function hoursToSeconds(value: string): number {
  if (!/^\d+(?:\.\d{1,4})?$/.test(value.trim())) throw new Error("budget_limits_invalid");
  const seconds = Math.round(Number(value) * 3600);
  if (!Number.isSafeInteger(seconds) || seconds < 0) throw new Error("budget_limits_invalid");
  return seconds;
}

export function validateBudget(gpu: number, engineering: number, current: ExecutionBudget, maximum: number) {
  if (engineering > gpu || gpu > maximum) return "budget_limits_invalid";
  if (gpu < current.committed_seconds || engineering < current.engineering_committed_seconds) return "budget_below_committed";
  if (gpu === current.gpu_limit_seconds && engineering === current.engineering_limit_seconds) return "budget_unchanged";
  return "";
}
