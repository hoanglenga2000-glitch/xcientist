export type MessageStatus = "complete" | "streaming" | "stopped" | "blocked" | "error";
export type MessageRecord = { id: string; role: "user" | "assistant"; content: string; status: MessageStatus; runId?: string; attachmentNames?: string[] };
export type MessageHistory = {
  schema: string;
  messages: Array<{ id: string; role: "user" | "assistant"; content: string; kind?: string }>;
  active_message_id: string;
};
type Snapshot = { id: string; status: string; prompt?: string; answer?: string; streaming_text?: string | null; answer_is_current?: boolean; message_history?: MessageHistory; attachments?: Array<{ name: string }> };

export function messageStatus(status: string): MessageStatus {
  if (status === "completed") return "complete";
  if (status === "blocked" || status === "waiting_approval") return "blocked";
  if (status === "failed") return "error";
  if (status === "cancelled") return "stopped";
  return "streaming";
}

export function latestAssistantMessageId(messages: MessageRecord[], runId: string): string | undefined {
  return [...messages].reverse().find((row) => row.role === "assistant" && row.runId === runId)?.id;
}

export function updateStreamingMessage<T extends MessageRecord>(messages: T[], runId: string, assistantId: string, text: string): T[] {
  if (latestAssistantMessageId(messages, runId) !== assistantId) return messages;
  // Each event is a cumulative snapshot, so replay cannot duplicate text.
  return messages.map((row) => row.id === assistantId && row.status === "streaming" ? { ...row, content: text } : row);
}

export function updateRunMessage<T extends MessageRecord>(messages: T[], run: Snapshot, preferredId?: string): T[] {
  const latestId = latestAssistantMessageId(messages, run.id);
  const preferred = messages.find((row) => row.role === "assistant" && row.id === preferredId && (!row.runId || row.runId === run.id));
  // A response from an older stream must not rewrite a frozen earlier reply.
  if (preferred?.runId && latestId && preferred.id !== latestId) return messages;
  const target = preferred?.id ?? latestId;
  if (!target) return messages;
  const status = messageStatus(run.status);
  return messages.map((row) => row.id !== target ? row : {
    ...row,
    runId: run.id,
    status,
    content: status === "streaming" ? (run.streaming_text ?? (row.status === "streaming" ? row.content : ""))
      : run.answer_is_current === false ? "" : run.answer || row.content,
  });
}

export function restoreRunMessages(run: Snapshot, continuedLabel: string): MessageRecord[] {
  if (run.message_history?.schema === "evomind.assistant_message_history.v1") {
    return run.message_history.messages.map((row) => ({
      id: row.id, role: row.role, runId: run.id,
      content: row.id === run.message_history!.active_message_id && messageStatus(run.status) === "streaming"
        ? run.streaming_text ?? row.content : row.kind === "continued_execution" ? continuedLabel : row.content,
      status: row.id === run.message_history!.active_message_id ? messageStatus(run.status) : "complete",
      ...(row.kind === "original_request" ? { attachmentNames: run.attachments?.map((item) => item.name) } : {}),
    }));
  }
  const status = messageStatus(run.status);
  return [
    { id: `user_${run.id}`, role: "user", content: (run.prompt ?? "").replace(/^\[Selected task: [^\]]+\]\r?\n/, ""), status: "complete", runId: run.id, attachmentNames: run.attachments?.map((row) => row.name) },
    { id: `assistant_${run.id}`, role: "assistant", content: status === "streaming" || run.status === "waiting_approval" ? "" : run.answer ?? "", status, runId: run.id },
  ];
}
