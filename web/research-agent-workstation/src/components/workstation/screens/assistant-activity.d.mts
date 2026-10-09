export type AssistantActivityIdentityInput = {
  type: string;
  seq: number;
  toolCallId?: string;
  stepId?: string;
  approvalId?: string;
  artifactId?: string;
};

export type AssistantActivityRecord = {
  identity?: string;
  seq: number;
  type: string;
  status: string;
  label: string;
  detail?: string;
};

export function assistantActivityIdentity(input: AssistantActivityIdentityInput): string;
export function upsertAssistantActivity<T extends AssistantActivityRecord>(current: T[], item: T, limit?: number): T[];
