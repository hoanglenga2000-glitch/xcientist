# Claude Code 最终执行 Prompt：EvoMind Research OS GUI V2 全面升级

你现在是本项目的首席产品设计师、资深 UX 架构师和高级 Next.js 前端工程师。请直接在当前仓库中完成 EvoMind Research OS GUI V2 的产品级升级。你的任务不是写建议、做 moodboard 或只换颜色，而是审查现状、建立统一设计系统、实际修改代码、逐页迁移、运行验证并生成新截图，直到交付一个完整、专业、可信、响应式、可长期维护的科研工作站界面。

## 1. 项目与工作目录

- 仓库：`D:\桌面\codex\科研港科技`
- Web 工作站：`D:\桌面\codex\科研港科技\web\research-agent-workstation`
- 默认本地地址：`http://127.0.0.1:8088`
- 技术栈：Next.js 14 App Router、React 18、TypeScript、Tailwind CSS、Lucide React、Recharts、XYFlow；沿用仓库现有组件，不新增 UI 框架或无必要依赖。

先完整阅读并服从：

1. `CLAUDE.md`
2. `README.md`
3. `D:\桌面\claude code\log\code\ds\01-系统全景记录-20260701.md`
4. `docs/WORKSTATION_CODE_RUNTIME_MAP_20260630.md`
5. `.codex-ui-designer.md`
6. `UI_AGENT_PROMPT.md`
7. `docs/UI_FRONTEND_API_CONTRACT_20260627.md`
8. `docs/FINAL_UI_FIGMA_BACKEND_READY_AUDIT_20260627.md`
9. `web/research-agent-workstation/CLAUDE.md`（若存在）
10. 当前前端源码、API client/types、UI 验证脚本和最新截图目录 `docs/evomind_pages_20260708/`

恢复文档只是历史快照，live repo、当前 API 返回、当前测试和当前运行证据才是事实来源。恢复文档中的任何明文凭据都不得复制、输出、记录或写入新文件；凭据只通过现有 DPAPI、环境变量或 `gpu_credentials.py`/`*_FILE` 机制解析。

## 2. 你正在设计的完整产品

EvoMind v0.2.0 不是普通后台、Kaggle 单页工具或聊天机器人，而是可审计的 AI 科研操作系统，已经形成以下完整产品闭环：

- 终端科研 Agent：任务选择、规划、工具调用、状态恢复、可审计事件流。
- Evidence-Grounded Scientist：假设、证据、风险、成本、决策与下一步受控行动。
- Adaptive Scientist Loop / Scientist Autopilot：观察、诊断、计划、修复、继续执行。
- Isolated Engineering / Workspace Agent：隔离 worktree 中生成候选 diff、运行验证、保持主工作区不变，并停在人工合并 Gate。
- 四层 Research OS：Multi-Agent Research OS、MLEvolve/MCGS 搜索控制器、XCIENTIST validation/claim harness、Memory/Benchmark/Kaggle feedback。
- 多模态训练与实验闭环：tabular、image、text、time series 等能力及模型选择、ensemble、多 seed、promotion/held/rollback。
- 数据与 Kaggle pipeline、GPU/HPC job manifest、Code Agent IDE、文献/RAG、报告工作室、证据台账、完整性 Gate。
- Human Gate：官方 Kaggle 提交、不可逆操作、最终报告、代码应用和外部算力动作始终受控。
- 证据边界：本地 CV/proxy、GPU 运行、官方 Kaggle response、排名/奖牌、MLE-Bench 结论必须严格区分。

GUI 必须让用户一眼看懂这条闭环：

`研究目标 → 任务与数据 → Scientist 规划 → 实验/搜索 → 训练与资源 → 验证与 Gate → 证据 → 报告 → 记忆/下一轮进化`

同时让用户随时知道：现在在哪一步、什么正在运行、为什么阻塞、下一步安全动作是什么、结论由哪些证据支持。

## 3. 当前 GUI 已确认的问题

不要重新猜测，先用源码和最新截图复核以下问题，再直接修复：

1. `Live Run Evidence` 以大面板重复出现在多页首屏，抢占当前页面核心任务，造成严重重复和视觉噪声。
2. 约 15 个路由入口全部长期暴露，导航虽有分组但仍过载；页面之间缺少明确的流程关系、父子层级和上下文导航。
3. 多个页面是大量白色边框卡片的平铺，卡片层级、标题层级、主次动作和留白缺少节奏，像粗糙 Admin 模板而不是科研操作系统。
4. 中英文、业务术语、状态词、路径、ID 与长文本混排不统一；字号过小，粗体过多，mono 使用无节制。
5. `Screens.tsx` 与 `AiControlConsole.tsx` 已成为超大单体文件；重复 Panel/Row/Metric/Badge 写法导致视觉漂移和维护困难。
6. 移动端存在横向溢出、长路径/长状态越界、桌面表格直接缩小或纵向硬堆叠、顶部快捷入口拥挤等问题。
7. 部分大区域留白与极密小字同时存在，信息密度没有按任务优先级分层。
8. 非交互 Card 也呈现 pointer/hover/active 反馈，交互 signifier 不准确。
9. 侧栏 connector 状态含硬编码展示；任何资源状态都必须来自真实 summary/API，未知就显示 Unknown/未验证，不能默认“在线”。
10. loading、empty、error、stale、blocked、pending、verified 等状态缺少统一表达。

## 4. 唯一视觉方向：Scientific Instrument

只采用一个方向：`Scientific Instrument / 科研仪器工作台`。气质应像顶级实验室的研究控制台与审计台账，而不是营销站、普通 SaaS 后台、游戏 HUD 或廉价赛博朋克。

设计论文：用清晰的科研流程层级、精确的状态语义、克制的工业标注和可追溯证据，让高信息密度仍然安静、可信、可操作。

视觉规则：

- 框架：深墨蓝/近黑导航框架，纸白/冷灰工作面；大面积内容保持明亮、低疲劳。
- 强调：只用一个品牌强调色（克制钴蓝）；teal/amber/red 只承担真实状态语义，不能拿状态色做装饰。
- 层级：主要依赖字号、间距、线条、表面层级和布局建立，不依赖到处加阴影/渐变。
- 字体：优先沿用现有系统字体栈；中文正文清晰，指标、run id、hash、路径才使用 mono/tabular numerals。不要为了“设计感”引入远程字体或造成构建时下载依赖。
- 形状：小到中等圆角、1px hairline、极轻阴影；不要满屏玻璃拟态、霓虹 glow、巨型渐变、悬浮大卡片。
- 图标：只使用现有 Lucide，同一层级保持统一尺寸与 stroke；不得用 emoji 作为结构图标。
- 动效：150–220ms，主要用于折叠、drawer、状态更新和因果反馈；只动画 transform/opacity，支持 `prefers-reduced-motion`。
- 明暗主题：以语义 token 实现 light/dark，不在组件中散落硬编码 hex；两套主题分别满足对比度。

唯一标志性设计是 `Evidence Rail`：把当前 task/run、证据数量、Gate 状态、claim boundary、下一步安全动作集中为可折叠的运行上下文条/右侧证据轨。它取代每页重复的巨大 `Live Run Evidence`，在 Overview、Experiments、Report、Evidence 等需要深度上下文的页面可展开完整内容，在其他页面只保留紧凑状态条。移动端使用 bottom sheet/drawer。

## 5. 信息架构升级

保留现有所有 route id、URL deep link、历史别名和后端契约，不做破坏性路由合并；重新组织导航呈现：

1. Command Center：Overview、AI Control。
2. Research Loop：Tasks、Experiments、Evolution、Workflow、Runtime。
3. Workbench：Data/Kaggle、Code Agent、Literature/RAG、Report Studio。
4. Infrastructure：GPU/HPC。
5. Governance：Evidence Ledger、Integrity Gates。
6. Settings：系统、外观、连接、安全；Design System 作为设置内的设计治理页或保留隐藏 deep link，不占主导航。

桌面侧栏支持展开/收起，默认只展开当前分组；当前 route、父分组、未完成/阻塞数量有清晰状态。移动端使用真正的 off-canvas 导航，关闭后不占布局，不允许横向页面滚动。顶部栏只保留：页面标题/面包屑、全局搜索/Command Palette、关键系统状态摘要、一个当前页面主动作和用户菜单；当前 7 个并列快捷按钮移入 Command Palette 或 overflow。

## 6. 全局布局与核心组件

先建立或重构以下基础能力，再迁移页面：

- 语义 design tokens：background/surface/ink/muted/border/accent/info/success/warning/danger、type ramp、spacing、radius、shadow、z-index、motion。
- `AppShell`、折叠导航、响应式 topbar、breadcrumb、Command Palette 入口。
- `PageHeader`：标题、说明、breadcrumb、主动作、次动作，保证每页只有一个 primary CTA。
- `RunContextBar` + `EvidenceRail`：当前 task/run/gate/claim/next action，真实数据驱动，可折叠。
- `Panel`、`MetricTile`、`StatusBadge`、`GateBadge`、`ClaimBoundary`、`CopyablePath`、`EmptyState`、`ErrorState`、`Skeleton`、`StaleIndicator`。
- 数据表：桌面支持 sticky header、排序/筛选、合理列宽；移动端变成摘要卡 + detail drawer，不能把 10 列表格硬塞进 390px。
- Inspector/drawer：长 JSON、artifact、run detail、evidence detail、agent trace 统一在可关闭的 inspector 中显示。
- AI 内容必须标注 `AI-generated/Draft`，提供 Apply/Accept/Reject/Undo 或明确的人工确认；引用与证据就近展示。
- 非交互卡片不得有 pointer、hover lift 或 click action；真正可点击的卡片要有键盘语义与可见 focus。
- 长路径/run id/hash 采用中段省略或换行，并提供 tooltip/copy；不得破坏布局。
- 状态词统一：Verified、Ready、Running、Pending、Blocked、Failed、Unknown、Stale；颜色之外必须有文本/图标。
- 中英文 copy 集中管理；同一 locale 下不再无规则混搭，保留 API 字段名、run id、metric 等必要英文。

不要引入新的全局状态库、图标库、动画库或完整 UI kit。优先复用现有 Tailwind、Lucide、Recharts、XYFlow 和 API client。

## 7. 逐页产品目标

### Overview

把首页做成真正的 Mission Control：第一屏只回答“系统是否健康、当前在做什么、下一步安全动作、哪里阻塞”。显示紧凑 KPI、当前 run、Research Loop stage、真实 connector 健康度、最近证据/告警和下一步动作。复杂 ledger 与 trace 下沉或进入 Evidence Rail；不得再重复整页大表。

### AI Control

形成科研指令工作台：Command Composer、当前 situation/context、Scientist plan/action queue、阻塞与不确定性、工具/agent trace、human-gated action。桌面可采用主工作区 + 右侧上下文轨；移动端用 tabs/segmented views。安全规则应清晰但不以巨型警告框长期占据首屏。

### Tasks

任务台账支持搜索、过滤、状态、modality、metric、resource mode、last run 和 next action；任务详情进入 drawer。明确区分配置完成、数据缺失、资源阻塞、可运行和已完成。

### Experiments

突出 search graph、score trend、best-so-far、promotion/held/rollback、run ledger 和实验对比。图表必须有标题、单位、legend、tooltip 与表格替代；不依赖红绿颜色表达结果。

### Evolution

清晰展示 MCGS/search tree、branch、expansion type、memory hit、stagnation、diversify/aggregation、promotion gate 与 best protection。真实图谱优先使用 XYFlow；节点详情进入 inspector，避免大面积空白 canvas。

### Workflow / Runtime

Workflow 展示 agent handoff、gate、fallback、artifact flow；Runtime 展示 agent、provider、cache、cost、event stream、continuation/recovery。两页视觉语法一致但职责明确，提供互相 deep link。

### Data / Kaggle

按 Dataset Audit、Schema/Split、Lineage、Readiness、Submission Structure、Official Response 分层。未配置、未验证和官方证据缺失必须真实显示；不得把 proxy/CV 当官方结果。

### Code Agent

做成可用的审计型 IDE：file tree、editor/diff、agent suggestions、terminal/check output、Code Quality Gate。桌面三栏，移动端 tabs；保留现有 export/import/review/apply action 和隔离 worktree/human merge gate 语义。

### Literature / RAG

包括检索输入、结果、来源/引用、RAG bundle、claim binding 和 citation audit。来源信息形成可扫描 citation rail；无真实来源时不生成假论文或假引用。

### Report Studio

形成 outline + document + evidence/claim rail 的写作环境。清晰标记 Draft/AI-generated，支持任务切换、章节导航、证据绑定、claim audit、Markdown/audit export 和最终人工 Gate。移动端按 Outline/Document/Evidence 三个视图切换。

### GPU / HPC

显示真实 connector、资源、job manifest、队列、日志、产物回传和失败恢复。任何未实时验证的资源均显示 Unknown/Blocked；不得硬编码 Ready，不得在本地 GPU 启动训练。

### Evidence Ledger

以可筛选 ledger + lineage/claim relation + detail inspector 为核心，支持复制路径、下载/导出草稿、trace claim。证据类型、验证级别、来源 run、Gate 和 claim 绑定关系必须可读。

### Integrity Gates

把 Code、Compute、Submission、Evidence、Report 等 Gate 组织为有顺序的 pipeline；展示 required artifact、当前决定、阻塞原因、审批历史和下一步。危险/不可逆操作与普通导航明确分离。

### Settings

分为 General、Appearance、Connectors、Credentials/Security、Design Governance。主题与语言使用真实设置 action；secret 只显示已配置/未配置/需轮换，不显示值。

## 8. 代码架构要求

当前 `Screens.tsx` 和 `AiControlConsole.tsx` 体积过大。采用渐进式拆分，不做高风险业务重写：

- 新建 `components/workstation/layout/`、`components/workstation/primitives/`、`components/workstation/features/`、`components/workstation/screens/` 等清晰目录。
- 每个正式页面独立文件；共享组件只保留一份；使用 typed props 和数据映射，避免复制大量 Tailwind 字符串。
- 可以用 barrel export 保持 `page.tsx` 的导入稳定。
- 客户端组件只覆盖需要 state/effect/browser API 的交互岛；不要为了重构改写现有后端调用语义。
- 如果现有验收脚本硬编码旧文件位置，允许同步更新为新文件结构，但只能替换为等价或更强的语义检查，不得删除、跳过或弱化测试。

## 9. 不可破坏的系统契约

以下全部必须保留：

- `PageId`、现有 query route、`mission → overview`、`evidence-detail → evidence` 等兼容行为。
- `data-ui-component`、`data-ui-page`、关键 `data-ui-action`、`data-testid`、全局 click audit、`uiActionRoutes`/pattern routing、`data-ui-skip-action` 防重复逻辑。
- `/api/workstation-actions`、现有 API client/types 与所有任务/实验/Scientist/Evolution/GPU/Evidence/Report/Literature 接口。
- loading、error、blocked action 仍然写审计；不能绕过 Gate 直接执行。
- Human Gate、Code Quality Gate、HPC execution gate、official submission gate、claim audit。
- 不自动训练、不自动提交 Kaggle、不操作本地 GPU、不创建官方结果、不虚构排名/奖牌/文献/connector readiness。
- GPU 远程文件策略、凭据安全与失败记录规则。
- 未经用户明确授权，不创建 commit、不 push、不开 PR。

本次主要修改前端和必要的 UI 测试/文档；不要改训练算法、研究结论、GPU 连接逻辑、Kaggle 提交流程或后端业务语义。

## 10. 响应式、可访问性与性能硬标准

- 验证视口至少：`375×812`、`390×844`、`768×1024`、`1024×768`、`1440×900`、`1920×1080`。
- 页面级 `scrollWidth <= clientWidth`；不允许移动端横向溢出。极宽数据表只能在受控表格容器内滚动，优先移动端摘要卡/详情 drawer。
- 正文移动端不小于 16px；桌面密集辅助信息可 12–14px，但不能整页 10–11px。
- 正常文本对比度至少 4.5:1，大字 3:1；focus 清晰；完整键盘导航；icon-only 按钮有 accessible name。
- 主要交互触控区域至少 44×44px；不可依赖 hover；modal/drawer 可 Escape 关闭并正确管理焦点。
- 支持 reduced motion；loading 超过 300ms 使用 skeleton/progress；异步操作有 disabled/loading/success/error 反馈。
- 避免 layout shift；图表和异步面板预留尺寸；长列表按现有能力做分页/局部渲染，不新增重型库。

## 11. 必须执行的工作流程

1. 先运行 `git status --short`，识别并保留用户已有修改；不得覆盖无关 dirty files。
2. 修改前为将触碰的 UI 文件创建带时间戳的备份或可恢复 patch，放在清晰的备份目录；不得使用 `git reset --hard`、`git checkout --` 等破坏命令。
3. 记录改造前基线：运行当前静态 UI 检查、typecheck，并浏览最新桌面/移动截图；发现已有失败要记录为 pre-existing。
4. 先输出一段简短设计 thesis、IA 和 token 方案，然后直接实现，不等待我再次确认。
5. Phase A：tokens、AppShell、导航、topbar、RunContextBar/EvidenceRail、响应式基础。
6. Phase B：共享组件、状态系统、table/drawer/empty/loading/error、localization。
7. Phase C：优先重做 Overview、AI Control、Experiments/Evolution、Report；随后完成其余全部正式页面，不能只交付示范页。
8. Phase D：拆分单体组件、清理重复样式和硬编码 connector/demo readiness；确保 API 仍提供同样业务能力。
9. Phase E：逐页 desktop/mobile 视觉验收、自动测试、构建、截图和最终报告。

遇到不确定的展示数据时，优先调用现有真实 API；没有证据就显示 `Unknown/未验证/暂无数据`。不要用漂亮的假数据填满界面。不要因 Figma OAuth 不可用而停工；本地源码与截图足以完成本轮升级。若 Figma 已真实可用，只可作为辅助对照，不得用整页图片覆盖 React UI。

## 12. 验证命令

在仓库现有命令基础上执行，构建与 typecheck 串行运行，避免并行改写 `.next`：

```powershell
cd D:\桌面\codex\科研港科技\web\research-agent-workstation
npm.cmd run typecheck
npm.cmd run build

cd D:\桌面\codex\科研港科技
python scripts\verify_scientific_ui_polish.py
python scripts\verify_ui_layout_quality.py
python scripts\verify_ui_localization_contract.py
python scripts\verify_workstation_ui_action_contract.py
python scripts\verify_workstation_frontend_api_contract.py --write-report
python scripts\verify_workstation_ui_component_wiring.py
python scripts\verify_no_plaintext_secrets.py
python scripts\run_ci_checks.py
```

启动工作站后继续运行适用的在线/浏览器验收：

```powershell
cd D:\桌面\codex\科研港科技\web\research-agent-workstation
npm.cmd run dev

cd D:\桌面\codex\科研港科技
node scripts\verify_workstation_responsive_smoke.mjs
python scripts\verify_workstation_browser_render_smoke.py
python scripts\run_full_acceptance.py --dashboard-url http://127.0.0.1:8088
powershell -ExecutionPolicy Bypass -File scripts\audit_ui_backend_ready.ps1 -BaseUrl http://127.0.0.1:8088
```

不要为了“全绿”修改业务数据、伪造外部资源状态、关闭测试或降低断言。若某命令因既有环境/外部资源阻塞，给出命令、原始错误、已完成的替代验证和明确边界。

## 13. 最终交付

完成后一次性给出：

1. 最终设计 thesis、信息架构与 `Evidence Rail` 的实现说明。
2. 修改/新增文件清单及每个文件职责。
3. 所有正式页面的升级结果，不只列四个示范页。
4. desktop/mobile 截图绝对路径和视口尺寸；截图必须来自真实 React 页面。
5. typecheck、build、UI contract、responsive、browser smoke、CI 的真实结果。
6. 未解决项、pre-existing 问题和外部阻塞；不得用“已完成”掩盖失败。
7. 确认没有泄露 secret、没有启动本地 GPU、没有自动 Kaggle submit、没有创建 commit。

立即开始。先审查 live repo 和建立备份，然后持续实施到所有页面、响应式和验收闭环完成；不要只返回计划或询问我选择哪种视觉风格。
