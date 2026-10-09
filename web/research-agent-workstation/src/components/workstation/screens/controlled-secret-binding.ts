export type RunBindingPurpose = "weather4cast_sftp" | "kaggle_api";

type ApprovalProjection = {
  status?: string;
  tool_name?: string;
  normalized_arguments?: Record<string, unknown>;
};

export type RunBindingProjection = {
  id?: string;
  status?: string;
  approvals?: ApprovalProjection[];
  controlled_secret_availability?: Record<string, number>;
};

export function isExactPendingWeatherAcceleration(run: RunBindingProjection | null | undefined) {
  if (!run || run.status !== "waiting_approval") return false;
  const pending = (run.approvals ?? []).filter((item) => item.status === "pending");
  if (pending.length !== 1 || pending[0].tool_name !== "competition_data_accelerate") return false;
  const args = pending[0].normalized_arguments;
  if (!args || args.competition !== "weather4cast") return false;
  const keys = Object.keys(args).sort();
  if (keys.some((key) => !["competition", "timeout_seconds"].includes(key))) return false;
  if ("timeout_seconds" in args && (!Number.isInteger(args.timeout_seconds) || Number(args.timeout_seconds) < 30 || Number(args.timeout_seconds) > 86400)) return false;
  return true;
}

export function isConfirmedControlledSecretBinding(
  run: RunBindingProjection | null | undefined,
  expectedRunId: string,
  purpose: RunBindingPurpose,
  expectedCount: number,
) {
  if (!run || run.id !== expectedRunId || !isExactPendingWeatherAcceleration(run)) return false;
  const observed = run.controlled_secret_availability?.[purpose];
  return Number.isInteger(observed) && observed === expectedCount;
}
