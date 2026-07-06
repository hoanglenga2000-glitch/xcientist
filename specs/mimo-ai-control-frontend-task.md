# MiMo Claude Code 实现任务：Academic Research OS 前端与 AI 对话控制页

你是本项目的执行工程师，负责写实际代码。Codex 只负责审查、验收和阻止破坏主线。你必须基于当前仓库真实代码实现，不要重写一套假类型，不要输出方案稿。

项目根目录：

`D:\桌面\codex\科研港科技`

前端目录：

`D:\桌面\codex\科研港科技\web\research-agent-workstation`

启动快捷方式：

`D:\桌面\Lean Claude\Claude Code MiMo V2.5 Pro.lnk`

## 一、核心目标

把当前 AI 科研工作站前端继续优化为真正的 Academic Research OS 操作台，并新增一个“AI 对话控制智能体”页面，让用户可以在工作站内用对话方式发起和监督 Kaggle/HPC/Code Agent/Report/Gate 流程。

注意：这个页面不是聊天玩具。它必须是工作站控制台的一部分，所有动作必须调用现有 API/action，并写入 action log。不能绕过工作站直接训练、直接提交 Kaggle、直接调用 GPU 长任务。

## 二、当前真实入口，必须保持

当前 `page.tsx` 已使用以下命名导出，不能删，不能改名：

- `MissionControl`
- `ResearchTasks`
- `DataKagglePipeline`
- `GpuHpcConsole`
- `EvidenceLedger`
- `LiteratureKnowledge`
- `WorkflowGraph`
- `CodeRunner`
- `AgentRuntime`
- `Experiments`
- `ReportStudio`
- `IntegrityGates`
- `SettingsCenter`
- `DesignSystem`
- `OverviewBoardEnhanced`

当前 `PageId` 在：

`web/research-agent-workstation/src/components/workstation/navigation.ts`

当前页面分发在：

`web/research-agent-workstation/src/app/page.tsx`

当前 API 客户端在：

`web/research-agent-workstation/src/lib/api/client.ts`

当前动作 API 在：

`POST /api/workstation-actions`

对应封装：

```ts
api.runWorkstationAction(action, taskId, metadata)
```

已有可复用 API：

```ts
api.getWorkstationSummary()
api.runLocalExperiment(taskId)
api.exportCodeAgentContext(taskId, targetAgent)
api.importAgentPatch(taskId, payload)
api.generateCodeAgentDraft(taskId, { source_agent })
api.createClaudeAgentSession(taskId, payload)
api.getClaudeAgentSession(sessionId)
api.cancelClaudeAgentSession(sessionId)
api.testGpuConnection()
api.submitGpuJob(taskId, template, metadata)
api.testDeepSeek(prompt)
api.generateReportDraft(taskId, payload)
api.generatePaperEvidenceBundle()
```

## 三、必须新增页面：AI Control / 对话控制智能体

新增一个页面入口，建议 PageId 为：

```ts
"control"
```

导航显示：

- 英文：`AI Control`
- 中文：`AI 控制台`
- 图标：优先使用 `Bot`、`MessageSquareText`、`Sparkles` 或 `BrainCircuit`

页面组件名称建议：

```ts
AiControlConsole
```

放置位置二选一：

- 直接加到 `web/research-agent-workstation/src/components/workstation/Screens.tsx`
- 或新建 `web/research-agent-workstation/src/components/workstation/AiControlConsole.tsx` 并在 `page.tsx` 导入

## 四、AI Control 页面必须具备的功能

### 1. 对话输入区

提供一个 textarea，用户可以输入自然语言命令，例如：

- “为 playground_series_s6e6 创建工作站 run”
- “导出 Claude Code 上下文”
- “让 DeepSeek Code Agent 生成下一轮代码草稿”
- “测试 GPU 连接”
- “准备 HPC execution gate”
- “生成教师证据包”
- “生成报告草稿”
- “查看最新 action log”

### 2. 命令解析

先实现本地轻量 intent parser，不要引入新依赖，不需要真实 LLM API。

根据关键词把用户命令映射到受控动作：

- 创建 run：`runWorkstationAction("create_workstation_run", metadata)`
- S6E6 onboard：`runWorkstationAction("onboard_playground_s6e6", metadata)`
- 准备 HPC Gate：`runWorkstationAction("prepare_hpc_execution_gate", metadata)`
- 导出 Code Agent 上下文：调用 `exportCodeAgentContext(selectedTask, "claude_code")`
- DeepSeek 代码草稿：调用 `api.generateCodeAgentDraft(taskId, { source_agent: "deepseek_code_agent" })`
- Claude Code 代码草稿：调用 `api.generateCodeAgentDraft(taskId, { source_agent: "claude_code" })`
- DeepSeek smoke：调用 `api.testDeepSeek(prompt)`
- GPU smoke：调用 `api.testGpuConnection()`
- GPU job：只允许 smoke/probe 模板，默认 `connection_smoke` 或已有白名单 template；非 smoke 训练必须显示 blocked，需要 HPC execution gate
- 本地实验：调用 `runLocalExperiment(taskId)`
- 报告草稿：调用 `api.generateReportDraft(taskId, { language: locale, style: "teacher_evidence_bundle" })`
- 教师证据包：调用 `api.generatePaperEvidenceBundle()` 或 `runWorkstationAction("generate_teacher_evidence_bundle", metadata)`
- Kaggle 官方提交：默认不执行。只显示需要 `submission_approval` Gate

### 3. 安全约束必须体现在 UI

页面必须明确显示并强制执行：

- Codex / MiMo / Code Agent 不直接训练
- 所有训练必须由工作站 action 或 GPU job manifest 发起
- 非 smoke GPU job 必须需要 HPC execution Gate
- Kaggle 官方提交必须需要 human submission Gate
- API key / 密码 / token 不显示、不输入、不写入前端状态

### 4. 命令预览与确认

用户输入后，先展示：

- 解析出的 intent
- 将调用的 API/action
- task_id
- metadata
- 风险等级：safe / gated / blocked
- blocked reason

对于 safe 动作可以直接执行。

对于 gated 动作，按钮文案显示：

`Submit to Workstation Gate`

对于 blocked 动作，按钮 disabled，并显示原因。

### 5. 结果面板

执行后展示：

- action name
- request summary
- response ok/status
- artifact path
- session id / run id / gate id（如果有）
- 错误信息
- latest action trace

所有长路径必须换行或截断，不能撑破布局。

### 6. 快捷动作区

提供按钮：

- Create Workstation Run
- Export Code Agent Context
- DeepSeek Draft
- Claude Code Draft
- Test DeepSeek
- Test GPU
- Prepare HPC Gate
- Generate Report Draft
- Generate Evidence Bundle
- Open Gates
- Open Code Studio
- Open Report Studio

如果某个按钮只是跳页面，必须真实调用页面切换或 URL query，不要死链接。

## 五、前端 UI 优化要求

顺手修补当前 UI 的明显问题，但不要推翻功能：

1. `Sidebar`
   - 加入 AI Control 导航
   - 保留所有旧入口
   - 不要出现死链接

2. `AppShell`
   - 顶部快捷入口增加 AI Control
   - 不要硬编码伪状态
   - 中文乱码如果改动到相关文案，必须写成正常中文

3. `OverviewBoardEnhanced`
   - 加一个 AI Control 工作区卡片
   - 卡片按钮真实跳转到 `?page=control`
   - 不伪造 DeepSeek/GPU/Kaggle 状态

4. `Screens.tsx`
   - 保留所有已有导出
   - 新增 `AiControlConsole` 或从新文件 re-export
   - 不要删除 CodeRunner / ReportStudio / GpuHpcConsole / IntegrityGates

## 六、实现文件范围

优先只改这些文件：

- `web/research-agent-workstation/src/components/workstation/navigation.ts`
- `web/research-agent-workstation/src/app/page.tsx`
- `web/research-agent-workstation/src/components/workstation/Sidebar.tsx`
- `web/research-agent-workstation/src/components/workstation/AppShell.tsx`
- `web/research-agent-workstation/src/components/workstation/OverviewBoardEnhanced.tsx`
- `web/research-agent-workstation/src/components/workstation/Screens.tsx`
- 可新增：`web/research-agent-workstation/src/components/workstation/AiControlConsole.tsx`

如果必须新增类型，只能增量修改：

- `web/research-agent-workstation/src/lib/api/types.ts`

不要改训练脚本，不要改 Kaggle 提交逻辑，不要改 GPU 网关核心逻辑。

## 七、强制禁止

- 禁止引入新依赖
- 禁止重写全项目
- 禁止把真实 blocked 显示为 ready
- 禁止伪造 Kaggle 分数/medal/rank
- 禁止伪造 GPU ready
- 禁止在代码中写入 API key、token、密码
- 禁止直接执行 Kaggle submit
- 禁止直接启动长训练
- 禁止删除现有功能入口
- 禁止使用不存在的类型，例如 `MissionRunBrief`、`summary.activeRun`、`summary.codeSessions`，除非你先在真实 `types.ts` 中增量定义并由真实 API 返回

## 八、输出要求

你必须直接修改代码，完成后输出：

1. 修改文件列表
2. 新增 AI Control 页面如何工作
3. 它调用了哪些真实 API/action
4. 哪些动作被 Gate/blocked
5. 运行结果

并且必须运行：

```powershell
cd D:\桌面\codex\科研港科技\web\research-agent-workstation
npm run typecheck
npm run build
```

如果失败，继续修到通过。不要把失败代码留给用户。

## 九、验收标准

Codex 审查时会检查：

- `npm run typecheck` 通过
- `npm run build` 通过
- `?page=control` 可以打开
- Sidebar 有 AI Control
- Topbar 或 Overview 有 AI Control 入口
- Code Agent、Report Studio、GPU/HPC、Kaggle、Evidence、Gate 入口仍存在
- AI Control 页面没有死按钮
- blocked/gated/safe 状态真实
- 没有写入密钥
- 没有直接训练/直接提交 Kaggle
- 所有动作有 response/action trace 可看

## 十、建议实现方式

不要复杂化。推荐先实现一个纯前端受控命令控制台：

```ts
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
  | "submission_blocked"
  | "unknown";
```

核心函数：

```ts
function parseControlCommand(input: string, taskId: string): ParsedControlCommand
```

然后根据 intent 调用已有 props/API。

页面内部维护：

```ts
const [input, setInput] = useState("")
const [parsed, setParsed] = useState<ParsedControlCommand | null>(null)
const [busy, setBusy] = useState(false)
const [messages, setMessages] = useState<ControlMessage[]>([])
const [lastResult, setLastResult] = useState<Record<string, unknown> | null>(null)
```

布局建议：

- 左侧：对话输入 + 快捷动作
- 中间：解析预览 + 执行结果
- 右侧：资源状态 + Gate 规则 + 最新 action trace

所有卡片圆角不超过 `rounded-lg`，按钮使用现有 `Button`，图标用 lucide-react。

