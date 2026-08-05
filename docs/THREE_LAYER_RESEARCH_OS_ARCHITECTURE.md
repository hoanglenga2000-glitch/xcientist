# AI 科研工作站三层融合架构

## 1. 项目定位

本项目不是单纯的 Kaggle 自动训练脚本，也不是只负责展示结果的后台页面，而是面向 Kaggle、HPC/GPU、MLE-Bench 风格任务的可审计自进化 AI 科研工作站。

它的核心目标是让科研任务从“人工临时调脚本”升级为“工作站发起、Agent 分工、artifact 记录、Gate 审批、GPU/HPC 执行、指标回传、报告审计”的长期运行系统。每一次实验都必须能回答三个问题：

- 这个实验为什么做；
- 它实际做了什么、产出了哪些 artifact；
- 它的结论是否被 CV、OOF、submission audit、日志和报告证据支持。

因此，工作站的成功标准不是单次 public score，而是可复现、可追踪、可回退、可审查，并能在多个任务上持续积累经验。

## 2. 三层架构

### 2.1 底层：Multi-Agent Research OS

底层保留当前项目已有的多 Agent 上下文分治和 artifact-based workflow，负责把科研任务变成可执行、可验证、可审计的工程流程。

底层职责包括：

- 任务拆解：把 Kaggle/HPC/MLE-Bench 任务拆成数据审计、实验设计、代码实现、执行、验证、报告等阶段；
- Agent 分工：由总控 Agent 协调研究背景、数据审计、特征工程、模型选择、代码实现、HPC/GPU 执行、验证分析、提交门禁、报告总结、反思审查等角色；
- artifact workflow：每个阶段输出结构化 artifact，例如 task spec、data audit、code diff、job manifest、metrics、OOF、submission、report；
- HPC/GPU 执行：通过工作站生成的 job manifest 和受控模板调度远程算力，回传 stdout、stderr、环境信息和实验产物；
- CV/OOF 追踪：强制记录交叉验证、OOF 预测、训练配置、随机种子、数据版本和指标；
- submission 门禁：官方提交前必须经过 schema 检查、分数风险检查、人工 Gate 和 submission audit；
- 报告生成：把实验台账、指标、失败回退、证据链和风险边界汇总为可交付报告。

底层解决的是 orchestration + execution，即“谁来做、按什么顺序做、怎么执行、怎么留下证据”。

### 2.2 中层：MLEvolve-style Search Controller

中层引入 MLEvolve-style Search Controller，负责让工作站不只是顺序执行固定实验，而是具备自进化机器学习搜索能力。

它负责：

- 自动 baseline 生成：根据任务类型创建第一个可复现 baseline，并绑定数据版本、metric 和 submission schema；
- 多分支实验搜索图：把 Logistic Regression、ExtraTrees、LightGBM、XGBoost、CatBoost、NN、Ensemble 等路线组织成 search graph；
- 多路线探索：在不同模型族、特征处理、调参、融合、校准、后处理之间分支探索；
- Retrospective Memory 历史经验检索：从历史任务中检索成功策略、失败模式、数据画像和 metric delta；
- Base / Stepwise / Diff 三种代码生成模式：
  - Base：从任务 spec 生成完整 baseline；
  - Stepwise：在明确计划下逐步扩展特征、模型或验证逻辑；
  - Diff：基于已有稳定代码做最小变更；
- 前期 exploration：优先覆盖模型族和数据处理路线，避免过早陷入单一路线；
- 后期 exploitation：在稳定候选附近做调参、融合、校准和风险受控优化；
- 下一轮决策：综合 CV、OOF、public score、稳定性、风险 flags、失败日志和 claim boundary 决定下一轮实验。

中层解决的是 optimization + self-evolution，即“下一步该试什么、为什么值得试、失败后怎么回退、历史经验如何复用”。

### 2.3 上层：XCIENTIST-style Research Harness

上层引入 XCIENTIST-style Research Harness，负责把每一次实验从“提分尝试”提升为“可验证科研声明”。

它负责：

- hypothesis：每次实验必须有明确假设，例如“OOF blend 能降低单模型方差并改善 CV 稳定性”；
- implementation contract：定义实现必须满足的约束，例如模型、数据切分、特征范围、禁止泄漏、日志和 artifact 要求；
- metric 和 acceptance criteria：明确主指标、辅助指标、相对 baseline 的接受标准；
- ablation plan：定义必要消融，例如去掉某类特征、去掉某个模型、只用单模型对比；
- risk check：检查 data leakage、CV-public gap、submission schema、过拟合、重复提交、指标漂移；
- conclusion boundary：限制结论范围，例如只能说“在当前 CV 设置下改善”，不能直接声称泛化到所有任务；
- claim drift audit：检查最终报告中的 claim 是否偏离实验原始假设、实现、指标或机制解释；
- evidence binding：最终报告中的每个 claim 必须绑定 exp_id、artifact、metrics、audit 或日志证据。

上层解决的是 validation + auditability，即“结论能不能说、证据够不够、风险在哪里、报告是否过度声称”。

## 3. 三篇/三类系统关系

三层系统不是互相替代，而是分工互补：

- 原项目解决 orchestration + execution：让科研工作站可以发起任务、调度 Agent、连接 Kaggle/HPC/GPU、记录 artifact、执行 Gate 和生成报告；
- MLEvolve 解决 optimization + self-evolution：让工作站具备 Progressive MCGS、多分支搜索、Retrospective Memory、自适应代码生成和探索/利用切换能力；
- XCIENTIST 解决 validation + auditability：让每次实验都有 hypothesis、contract、ablation、risk check、claim boundary 和 claim drift audit。

融合后的目标形态是 Self-Evolving and Auditable MLE Research OS：既能持续优化机器学习方案，又不会把 public score 当成唯一目标；既能自动探索，又能把每个结论绑定到证据链。

## 4. 最小可行版本 MVP

MVP 以 Kaggle tabular 任务为第一类验证对象。

输入：

- Kaggle tabular 任务配置；
- 数据路径、metric、submission schema；
- 已有 baseline 或空白任务；
- 可选历史实验台账和 Retrospective Memory。

输出：

- baseline solution.py；
- 多轮 experiment records；
- search graph；
- metrics.json；
- OOF prediction；
- submission.csv；
- validation contract；
- claim audit report；
- final reproducibility report。

MVP 的关键验收标准：

- 每个 experiment node 有 exp_id、parent_id、hypothesis、implementation、metrics、risk flags 和 decision；
- 每次进入新分支前由 Search Controller 给出计划；
- 每次实验前由 Research Harness 生成 validation contract；
- 每个报告 claim 经过 claim audit；
- 官方提交默认阻断，只有通过 submission Gate 后才能调用 Kaggle API。

## 5. 后续实现路线

### Phase 1：文档和 schema

完成三层架构文档、experiment node schema、search graph schema、retrospective memory schema、validation contract schema、claim audit schema，以及对应 Agent prompt 模板和 Python 轻量骨架。

### Phase 2：实验搜索图和记忆检索

把现有 EXP000-EXP034 历史实验转化为 search graph 和 retrospective memory，支持按任务类型、数据画像、模型族、失败模式和 metric delta 检索。

### Phase 3：自适应代码生成

把 DeepSeek Code Agent、Claude Code 或其他代码 Agent 接入 Base / Stepwise / Diff 三种模式。代码生成必须先产出 plan、diff、transcript 和 artifact manifest，再进入 Code Quality Gate。

### Phase 4：validation contract 和 claim audit

把 validation contract 嵌入每个新实验：实验执行前生成 contract，执行后检查 required artifacts 和 acceptance criteria，报告生成前运行 claim drift audit。

### Phase 5：多任务 MLE-Bench 风格验证

从单个 S6E6 任务扩展到多个 Kaggle tabular、NLP、CV 或科学数据任务，以 MLE-Bench 风格评估工作站跨任务自动化、复现性、提分效率和审计质量。
