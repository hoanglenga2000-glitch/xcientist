export function assistantActivityIdentity(input) {
  if (input.type === "tool_started" || input.type === "tool_completed") {
    return input.toolCallId ? `tool:${input.toolCallId}` : `${input.type}:${input.seq}`;
  }
  if (input.type === "model_observed") return "model";
  if (input.type === "step_started") return input.stepId ? `step:${input.stepId}` : `step:${input.seq}`;
  if (input.type === "approval_required") return input.approvalId ? `approval:${input.approvalId}` : `approval:${input.seq}`;
  if (input.type === "artifact_published") return input.artifactId ? `artifact:${input.artifactId}` : `artifact:${input.seq}`;
  if (input.type === "run_blocked" || input.type === "run_failed") return "run-terminal";
  return `${input.type}:${input.seq}`;
}

export function upsertAssistantActivity(current, item, limit = 30) {
  const identity = item.identity ?? `${item.type}:${item.seq}`;
  const existingIndex = current.findIndex((entry) => (entry.identity ?? `${entry.type}:${entry.seq}`) === identity);
  if (existingIndex >= 0) {
    const next = current.slice();
    next[existingIndex] = item;
    return next;
  }
  return [...current, item].slice(-limit);
}
