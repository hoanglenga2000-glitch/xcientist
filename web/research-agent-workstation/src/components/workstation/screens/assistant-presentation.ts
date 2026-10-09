type Locale = "zh-CN" | "en-US";

export const LEGACY_EMPTY_SUMMARY = "The model returned no final prose after verified tool execution. Use the published artifacts and run evidence as the authoritative result.";

export function assistantMessagePresentation(content: string, role: "user" | "assistant", locale: Locale) {
  if (role !== "assistant" || content.trim() !== LEGACY_EMPTY_SUMMARY) return { text: content, rawDiagnostic: null };
  return {
    text: locale === "zh-CN"
      ? "本次执行没有生成可读总结。请查看右侧运行详情与错误及下方记录；日志不代表已经交付模型或完成研究目标。"
      : "This execution returned without a readable summary. Open run details and inspect the attached records; logs do not prove model delivery or research completion.",
    rawDiagnostic: content,
  };
}

export function isAssistantExecutionRecord(name: string): boolean {
  return /\.(?:log|jsonl)$/i.test(name) || /(?:^|[-_.])(?:receipt|manifest|log)(?:[-_.]|$)/i.test(name)
    || /^(?:task[-_]graph|search[-_](?:graph|selection)|claim[-_]audit|capability[-_]contract)\.json$/i.test(name);
}

export function assistantArtifactLabel(name: string, locale: Locale): string {
  const record = isAssistantExecutionRecord(name);
  const model = /\.(?:safetensors|pt|pth|ckpt|npz|onnx)$/i.test(name);
  const prediction = /(?:prediction|predictions|preds|oof)/i.test(name) && /\.(?:csv|parquet|npy)$/i.test(name);
  if (record) return locale === "zh-CN" ? "运行记录 · 非模型产物" : "Execution record, not a model";
  if (model) return locale === "zh-CN" ? "模型候选文件 · 待独立验证" : "Model candidate, independent verification required";
  if (prediction) return locale === "zh-CN" ? "预测文件 · 待评分核验" : "Predictions, scorer verification required";
  return locale === "zh-CN" ? "结果文件" : "Result file";
}
