# Roadmap to MLE-Bench 75

## Phase 0：完成三层架构骨架

- 目标：形成 Multi-Agent Research OS + MLEvolve-style Search Controller + XCIENTIST-style Research Harness 的最小工程骨架。
- 输入：现有工作站、实验台账、论文思想、schema 和 Agent prompt。
- 输出：三层架构文档、schema、prompt、research_os 轻量代码和 demo。
- 验收标准：本地 demo 可运行，claim audit 能在证据不足时阻断结论。
- 风险：只停留在文档和样例，没有接入真实工作站 run。
- 下一步：把历史 EXP 记录转成 search graph 和 retrospective memory。

## Phase 1：单任务闭环稳定化

- 目标：以一个 tabular 任务打通工作站发起、Agent 分工、HPC/GPU 执行、Gate、报告和 claim audit。
- 输入：任务 spec、数据路径、metric、GPU/HPC manifest、Kaggle Gate。
- 输出：baseline、search graph、metrics、OOF、submission audit、final reproducibility report。
- 验收标准：至少一次完整链路由工作站发起并留下 artifact，而不是旁路训练。
- 风险：资源凭据变化、HPC 连接不稳定、Code Agent 未接入缓存。
- 下一步：固化 job manifest 和失败回退。

## Phase 2：3-5 个 tabular 任务自动化

- 目标：扩展到 3-5 个 tabular Kaggle 任务，优先提高 valid submission rate。
- 输入：tasks_template、benchmark task schema、历史策略记忆。
- 输出：每个任务的 valid submission、experiment records、contract、audit 和 gap report。
- 验收标准：所有任务无论成功失败都被记录；失败能归因并进入 Retrospective Memory。
- 风险：不同任务的数据解析和 metric 细节导致 baseline 失败。
- 下一步：抽象 robust baseline first 模板。

## Phase 3：10-15 个任务批量评测

- 目标：形成批量 benchmark manager，支持多任务结果汇总和 medal/proxy 指标统计。
- 输入：10-15 个任务注册表、统一运行预算、Search Controller 策略。
- 输出：benchmark summary、valid submission rate、medal rate、失败分类和下一轮优化计划。
- 验收标准：可复现实验目录、指标统计和 gap report 自动生成。
- 风险：任务异质性增大，依赖环境和训练时间差异显著。
- 下一步：引入任务画像和模型路线选择器。

## Phase 4：加入多模态/图像/文本任务支持

- 目标：从 tabular 扩展到图像、文本、多模态或科学数据任务。
- 输入：多模态数据解析器、任务画像、模型族模板和资源预算。
- 输出：图像/文本任务 baseline、验证 contract、artifact manifest 和报告。
- 验收标准：非 tabular 任务也能产生 valid submission 或明确 failure artifact。
- 风险：GPU 依赖更强，训练时间更长，模型选择复杂度上升。
- 下一步：建立 modality-specific agent prompts 和 job templates。

## Phase 5：覆盖完整 75 个任务

- 目标：建立完整 MLE-Bench 75 任务注册表和批量评测台账。
- 输入：完整任务列表、统一 budget、Kaggle/MLE-Bench 评测规则。
- 输出：75 任务 benchmark result、每任务报告、总 benchmark gap report。
- 验收标准：不能只报告成功任务；失败、超时、schema 错误、claim drift 都必须入账。
- 风险：时间和算力成本高，部分任务可能因为数据权限或比赛状态不可复现。
- 下一步：统一对齐 MLEvolve 评测条件。

## Phase 6：与 MLEvolve 指标对齐比较

- 目标：在可说明预算和评测规则下，与 MLEvolve 报告指标进行对齐比较。
- 输入：75 任务结果、MLEvolve 报告指标、medal 判定规则、预算说明。
- 输出：对齐比较报告、差距矩阵、失败原因统计。
- 验收标准：只在证据充分时报告 achieved；否则输出 partial 或 no。
- 风险：排行榜时间变化、private score 不可得、任务集不完全一致。
- 下一步：针对最大差距模块优化。

## Phase 7：优化 medal rate，目标达到或超过 MLEvolve 报告水平

- 目标：持续提高 medal rate、valid submission rate 和 best score trajectory。
- 输入：gap report、Retrospective Memory、失败分类和 top candidate 分析。
- 输出：优化后的 Search Controller、模型策略、代码生成策略和资源调度策略。
- 验收标准：在可对齐 benchmark 下 medal rate 达到或超过目标，且 claim audit 允许对外声明。
- 风险：过拟合 public leaderboard、牺牲 CV 可靠性、不可复现。
- 下一步：长期维护 benchmark dashboard 和回归评测。
