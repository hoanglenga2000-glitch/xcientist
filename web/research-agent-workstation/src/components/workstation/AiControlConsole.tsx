"use client";

import { useRef, useState } from "react";
import {
  Bot,
  BrainCircuit,
  Cpu,
  FileCheck2,
  GitBranch,
  Play,
  RefreshCcw,
  Send,
  ShieldCheck,
  TerminalSquare,
  Upload,
  XCircle
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusBadge, type StatusTone } from "@/components/ui/status-badge";
import * as api from "@/lib/api/client";
import { JsonInspector } from "./Common";

type Locale = "zh-CN" | "en-US";

type ControlIntent =
  | "create_workstation_run"
  | "onboard_playground_s6e6"
  | "prepare_hpc_execution_gate"
  | "export_code_agent_context"
  | "deepseek_code_draft"
  | "claude_code_draft"
  | "deepseek_smoke"
  | "gpu_smoke"
  | "gpu_probe_job"
  | "run_local_experiment"
  | "generate_report_draft"
  | "generate_teacher_evidence_bundle"
  | "kaggle_submit"
  | "unknown";

type RiskLevel = "safe" | "gated" | "blocked";

type ParsedControlCommand = {
  intent: ControlIntent;
  taskId: string;
  metadata: Record<string, unknown>;
  risk: RiskLevel;
  blockedReason?: string;
  description: string;
};

type ControlMessage = {
  role: "user" | "system" | "error";
  content: string;
  timestamp: number;
};

type ScreenProps = {
  selectedTask: string;
  locale?: Locale;
  runWorkstationAction?: (action: string, metadata?: Record<string, unknown>) => Promise<unknown>;
  exportCodeAgentContext?: (taskId?: string, targetAgent?: string) => Promise<void>;
  runLocalExperiment?: (taskId?: string) => Promise<void>;
  lastActionTrace?: {
    action: string;
    taskId?: string;
    request?: Record<string, unknown>;
    response?: Record<string, unknown>;
    message: string;
    artifact?: string | null;
    at: string;
  } | null;
};

function tx(locale: Locale | undefined, en: string, zh: string) {
  return locale === "zh-CN" ? zh : en;
}

function parseControlCommand(input: string, taskId: string): ParsedControlCommand {
  const lower = input.toLowerCase();
  const map: Array<{
    keywords: string[];
    intent: ControlIntent;
    risk: RiskLevel;
    blockedReason?: string;
    description: string;
  }> = [
    { keywords: ["onboard", "s6e6", "playground", "接入"], intent: "onboard_playground_s6e6", risk: "gated", description: "通过工作站门禁接入 Playground S6E6。" },
    { keywords: ["create", "workstation", "run", "创建", "新建"], intent: "create_workstation_run", risk: "gated", description: "创建一个新的可审计工作站 run。" },
    { keywords: ["hpc", "gate", "prepare", "算力", "门禁"], intent: "prepare_hpc_execution_gate", risk: "gated", description: "准备 HPC/GPU 执行门禁，不直接启动长训练。" },
    { keywords: ["export", "context", "agent", "导出", "上下文"], intent: "export_code_agent_context", risk: "safe", description: "导出 Code Agent 上下文包，供 Claude Code / Codex 审查。" },
    { keywords: ["deepseek", "draft", "草稿"], intent: "deepseek_code_draft", risk: "gated", description: "生成 DeepSeek Code Agent 草稿，仍需 code review gate。" },
    { keywords: ["claude", "draft", "草稿"], intent: "claude_code_draft", risk: "gated", description: "生成 Claude Code 草稿，仍需 code review gate。" },
    { keywords: ["deepseek", "smoke", "测试"], intent: "deepseek_smoke", risk: "safe", description: "运行 DeepSeek 连接烟测，不打印密钥。" },
    { keywords: ["gpu", "smoke", "连接"], intent: "gpu_smoke", risk: "safe", description: "运行 GPU SSH 网关烟测。" },
    { keywords: ["gpu", "probe"], intent: "gpu_probe_job", risk: "gated", description: "提交 GPU probe job，仅允许 smoke template。" },
    {
      keywords: ["local", "experiment", "本地小任务"],
      intent: "run_local_experiment",
      risk: "gated",
      blockedReason: "本地实验只允许作为工作站资源策略下的受控 smoke；默认训练仍应走 HPC/GPU gate。",
      description: "请求受控小任务实验；若资源策略禁用本地 fallback，后端会返回 blocked artifact。"
    },
    { keywords: ["report", "draft", "报告"], intent: "generate_report_draft", risk: "gated", description: "根据真实 evidence 生成报告草稿。" },
    { keywords: ["evidence", "bundle", "teacher", "证据包"], intent: "generate_teacher_evidence_bundle", risk: "gated", description: "生成教师汇报证据包。" },
    {
      keywords: ["kaggle", "submit", "提交"],
      intent: "kaggle_submit",
      risk: "blocked",
      blockedReason: "官方 Kaggle 提交必须经过 human submission_approval gate，本控制台不会自动提交。",
      description: "官方 Kaggle 提交（被门禁阻断）。"
    }
  ];

  for (const entry of map) {
    if (entry.keywords.some((kw) => lower.includes(kw))) {
      return {
        intent: entry.intent,
        taskId,
        metadata: { trigger: "ai_control_console", raw_input: input },
        risk: entry.risk,
        blockedReason: entry.blockedReason,
        description: entry.description
      };
    }
  }
  return {
    intent: "unknown",
    taskId,
    metadata: { trigger: "ai_control_console", raw_input: input },
    risk: "blocked",
    blockedReason: "无法识别该命令。请使用快捷动作或输入更明确的工作站动作。",
    description: "未知命令。"
  };
}

function riskTone(risk: RiskLevel): StatusTone {
  return risk === "safe" ? "green" : risk === "gated" ? "amber" : "red";
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex min-w-0 items-start justify-between gap-3 border-b border-slate-100 py-1.5 text-xs last:border-b-0">
      <span className="shrink-0 font-semibold text-slate-500">{label}</span>
      <span className="min-w-0 break-all text-right font-medium text-slate-800">{value}</span>
    </div>
  );
}

export function AiControlConsole({
  selectedTask,
  locale,
  runWorkstationAction,
  exportCodeAgentContext,
  runLocalExperiment,
  lastActionTrace
}: ScreenProps) {
  const [input, setInput] = useState("");
  const [parsed, setParsed] = useState<ParsedControlCommand | null>(null);
  const [busy, setBusy] = useState(false);
  const [messages, setMessages] = useState<ControlMessage[]>([]);
  const [lastResult, setLastResult] = useState<{
    action: string;
    request: string;
    ok: boolean;
    status: number;
    artifact?: string;
    sessionId?: string;
    error?: string;
    rawResponse?: unknown;
  } | null>(null);
  const previewRef = useRef<HTMLDivElement>(null);

  function pushMessage(role: ControlMessage["role"], content: string) {
    setMessages((prev) => [...prev, { role, content, timestamp: Date.now() }]);
  }

  function parseInput(value = input) {
    if (!value.trim()) return;
    const result = parseControlCommand(value.trim(), selectedTask);
    setParsed(result);
    pushMessage("user", value.trim());
    setTimeout(() => previewRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 30);
  }

  async function executeAction() {
    if (!parsed || parsed.risk === "blocked") return;
    setBusy(true);
    let rawResponse: unknown = null;
    try {
      const taskId = parsed.taskId;
      const meta = parsed.metadata;
      switch (parsed.intent) {
        case "create_workstation_run":
          rawResponse = await runWorkstationAction?.("create_workstation_run", { task_id: taskId, ...meta });
          pushMessage("system", `已为 ${taskId} 创建工作站 run。`);
          break;
        case "onboard_playground_s6e6":
          rawResponse = await runWorkstationAction?.("onboard_playground_s6e6", { task_id: taskId, ...meta });
          pushMessage("system", "S6E6 接入流程已提交到工作站门禁。");
          break;
        case "prepare_hpc_execution_gate":
          rawResponse = await runWorkstationAction?.("prepare_hpc_execution_gate", { task_id: taskId, template: "connection_smoke", ...meta });
          pushMessage("system", "HPC/GPU 执行门禁已准备。");
          break;
        case "export_code_agent_context":
          await exportCodeAgentContext?.(taskId, "claude_code");
          rawResponse = { ok: true, task_id: taskId, target_agent: "claude_code" };
          pushMessage("system", "Code Agent 上下文已导出。");
          break;
        case "deepseek_code_draft":
          rawResponse = await api.generateCodeAgentDraft(taskId, { source_agent: "deepseek_code_agent" });
          pushMessage("system", "DeepSeek Code Agent 草稿已生成。");
          break;
        case "claude_code_draft":
          rawResponse = await api.generateCodeAgentDraft(taskId, { source_agent: "claude_code" });
          pushMessage("system", "Claude Code 草稿已生成。");
          break;
        case "deepseek_smoke":
          rawResponse = await api.testDeepSeek("Hello from AI Control Console");
          pushMessage("system", "DeepSeek 连接烟测完成。");
          break;
        case "gpu_smoke":
          rawResponse = await api.testGpuConnection();
          pushMessage("system", "GPU SSH 网关烟测完成。");
          break;
        case "gpu_probe_job":
          rawResponse = await api.submitGpuJob(taskId, "connection_smoke", meta);
          pushMessage("system", "GPU probe job 已提交，仅使用 smoke template。");
          break;
        case "run_local_experiment":
          await runLocalExperiment?.(taskId);
          rawResponse = { ok: true, task_id: taskId, mode: "workstation_resource_policy_smoke" };
          pushMessage("system", "受控小任务实验请求已交给工作站资源策略处理。");
          break;
        case "generate_report_draft":
          rawResponse = await api.generateReportDraft(taskId, { language: locale ?? "zh-CN", style: "teacher_evidence_bundle" });
          pushMessage("system", "报告草稿已生成。");
          break;
        case "generate_teacher_evidence_bundle":
          rawResponse = await api.generatePaperEvidenceBundle();
          pushMessage("system", "教师汇报证据包已生成。");
          break;
        case "kaggle_submit":
        case "unknown":
          break;
      }

      const data = rawResponse as Record<string, unknown> | null;
      setLastResult({
        action: parsed.intent,
        request: JSON.stringify(parsed.metadata),
        ok: Boolean(data?.ok ?? true),
        status: 200,
        artifact: (data?.artifact_dir ?? data?.artifact ?? data?.context_dir ?? data?.markdown_path) as string | undefined,
        sessionId: (data?.session_id ?? data?.job_id ?? data?.run_id ?? data?.action_id) as string | undefined,
        rawResponse
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      pushMessage("error", `动作失败：${msg}`);
      setLastResult({ action: parsed.intent, request: JSON.stringify(parsed.metadata), ok: false, status: 500, error: msg });
    } finally {
      setBusy(false);
    }
  }

  function quick(label: string, command: string) {
    pushMessage("user", `[Quick Action] ${label}`);
    setInput(command);
    parseInput(command);
  }

  function navigateTo(page: string) {
    if (typeof window === "undefined") return;
    const url = new URL(window.location.href);
    url.searchParams.set("page", page);
    window.history.pushState(null, "", url);
    window.dispatchEvent(new PopStateEvent("popstate"));
  }

  const quickActions = [
    { label: tx(locale, "Create Run", "创建 Run"), icon: GitBranch, command: "create workstation run" },
    { label: tx(locale, "Export Context", "导出上下文"), icon: TerminalSquare, command: "export code agent context" },
    { label: tx(locale, "DeepSeek Draft", "DeepSeek 草稿"), icon: BrainCircuit, command: "deepseek draft" },
    { label: tx(locale, "Claude Draft", "Claude 草稿"), icon: Bot, command: "claude code draft" },
    { label: tx(locale, "DeepSeek Smoke", "DeepSeek 烟测"), icon: BrainCircuit, command: "deepseek smoke test" },
    { label: tx(locale, "GPU Smoke", "GPU 烟测"), icon: Cpu, command: "gpu smoke test" },
    { label: tx(locale, "HPC Gate", "HPC 门禁"), icon: ShieldCheck, command: "prepare hpc execution gate" },
    { label: tx(locale, "Report Draft", "报告草稿"), icon: FileCheck2, command: "generate report draft" },
    { label: tx(locale, "Evidence Bundle", "证据包"), icon: Upload, command: "generate teacher evidence bundle" }
  ];

  return (
    <main className="space-y-4">
      <div>
        <h2 className="text-xl font-bold tracking-normal text-slate-950">{tx(locale, "AI Control Console", "AI 控制台")}</h2>
        <p className="mt-1 text-sm leading-6 text-slate-500">
          {tx(locale, "Command the workstation with natural language. All actions are gated and logged.", "用自然语言调度工作站；所有动作都经过门禁并写入审计日志。")}
        </p>
      </div>

      <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs leading-5 text-amber-800">
        <strong>{tx(locale, "Safety Rules:", "安全规则：")}</strong>{" "}
        {tx(
          locale,
          "Code Agents never bypass the workstation. Training requires a workstation action or GPU job manifest. Official Kaggle submission requires human approval. Secrets are never displayed.",
          "Code Agent 不绕过工作站；训练必须由 workstation action 或 GPU job manifest 发起；官方 Kaggle 提交必须有人类审批；页面不展示任何密钥。"
        )}
      </div>

      <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>{tx(locale, "Command Input", "命令输入")}</CardTitle>
              <CardDescription>{tx(locale, "Type a command or use a quick action.", "输入自然语言命令，或使用下方快捷动作。")}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <textarea
                className="w-full rounded-md border border-slate-200 bg-white p-3 text-sm text-slate-800 shadow-sm focus:border-blue-300 focus:outline-none focus:ring-2 focus:ring-blue-200"
                rows={3}
                placeholder={tx(locale, "e.g. Create a workstation run for playground_series_s6e6", "例如：为 playground_series_s6e6 创建工作站 run")}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    parseInput();
                  }
                }}
              />
              <Button onClick={() => parseInput()} disabled={!input.trim()}>
                <Send className="h-4 w-4" />
                {tx(locale, "Parse Command", "解析命令")}
              </Button>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>{tx(locale, "Quick Actions", "快捷动作")}</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-wrap gap-2">
              {quickActions.map((item) => (
                <Button key={item.label} size="sm" variant="secondary" onClick={() => quick(item.label, item.command)} disabled={busy}>
                  <item.icon className="h-3.5 w-3.5" />
                  {item.label}
                </Button>
              ))}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>{tx(locale, "Page Shortcuts", "页面跳转")}</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-wrap gap-2">
              {[
                ["gates", tx(locale, "Open Gates", "进入 Gate")],
                ["code", tx(locale, "Open Code Studio", "进入代码工作台")],
                ["report", tx(locale, "Open Report Studio", "进入报告工作台")]
              ].map(([page, label]) => (
                <Button key={page} size="sm" variant="ghost" onClick={() => navigateTo(page)}>
                  {label}
                </Button>
              ))}
            </CardContent>
          </Card>
        </div>

        <div className="space-y-4">
          <div ref={previewRef}>
            <Card>
              <CardHeader>
                <CardTitle>{tx(locale, "Command Preview", "命令预览")}</CardTitle>
                <CardDescription>{tx(locale, "Review intent and risk before execution.", "执行前确认意图、风险和后端动作。")}</CardDescription>
              </CardHeader>
              <CardContent>
                {!parsed ? (
                  <p className="text-xs text-slate-400">{tx(locale, "No command parsed yet.", "尚未解析命令。")}</p>
                ) : (
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <StatusBadge tone={riskTone(parsed.risk)}>{parsed.risk}</StatusBadge>
                      <span className="text-xs font-bold text-slate-700">{parsed.intent}</span>
                    </div>
                    <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
                      <Row label={tx(locale, "Intent", "意图")} value={parsed.intent} />
                      <Row label={tx(locale, "Task ID", "任务 ID")} value={parsed.taskId} />
                      <Row label={tx(locale, "Risk", "风险")} value={parsed.risk} />
                      <Row label={tx(locale, "Description", "说明")} value={parsed.description} />
                      {parsed.blockedReason && <Row label={tx(locale, "Blocked", "阻断原因")} value={<span className="text-red-600">{parsed.blockedReason}</span>} />}
                    </div>
                    {parsed.risk !== "blocked" ? (
                      <Button variant={parsed.risk === "gated" ? "secondary" : "primary"} onClick={executeAction} disabled={busy}>
                        {busy ? <RefreshCcw className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                        {parsed.risk === "gated" ? tx(locale, "Submit to Gate", "提交到 Gate") : tx(locale, "Execute", "执行")}
                      </Button>
                    ) : (
                      <Button disabled variant="danger">
                        <XCircle className="h-4 w-4" />
                        {tx(locale, "Blocked", "已阻断")}
                      </Button>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {lastResult && (
            <Card>
              <CardHeader>
                <CardTitle>{tx(locale, "Execution Result", "执行结果")}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
                  <Row label={tx(locale, "Action", "动作")} value={lastResult.action} />
                  <Row label={tx(locale, "Status", "状态")} value={lastResult.ok ? <span className="text-emerald-600">OK</span> : <span className="text-red-600">{lastResult.error ?? "failed"}</span>} />
                  {lastResult.artifact && <Row label={tx(locale, "Artifact", "产物")} value={<span>{lastResult.artifact}</span>} />}
                  {lastResult.sessionId && <Row label="Session / Run / Gate ID" value={lastResult.sessionId} />}
                  {!!lastResult.rawResponse && (
                    <div className="mt-3">
                      <div className="mb-2 text-xs font-semibold text-slate-500">Raw JSON Response</div>
                      <JsonInspector data={lastResult.rawResponse} />
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader>
              <CardTitle>{tx(locale, "Message History", "消息记录")}</CardTitle>
            </CardHeader>
            <CardContent>
              {messages.length === 0 ? (
                <p className="text-xs text-slate-400">{tx(locale, "No messages yet.", "暂无消息。")}</p>
              ) : (
                <div className="max-h-48 space-y-2 overflow-y-auto">
                  {messages.map((msg, i) => (
                    <div
                      key={`${msg.timestamp}-${i}`}
                      className={`rounded-md px-3 py-2 text-xs ${
                        msg.role === "user" ? "bg-blue-50 text-blue-800" : msg.role === "error" ? "bg-red-50 text-red-700" : "bg-emerald-50 text-emerald-800"
                      }`}
                    >
                      <span className="mr-2 font-bold uppercase">{msg.role}</span>
                      {msg.content}
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {lastActionTrace && (
            <Card>
              <CardHeader>
                <CardTitle>{tx(locale, "Latest Action Trace", "最新 Action Trace")}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
                  <Row label={tx(locale, "Action", "动作")} value={lastActionTrace.action} />
                  <Row label={tx(locale, "Message", "消息")} value={lastActionTrace.message} />
                  {lastActionTrace.artifact && <Row label={tx(locale, "Artifact", "产物")} value={<span>{lastActionTrace.artifact}</span>} />}
                  <Row label={tx(locale, "At", "时间")} value={lastActionTrace.at} />
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </main>
  );
}
