# 科研 Agent 工作站建设计划

## 1. 任务理解

老师给我的方向不是简单做一个聊天式 AI 助手，而是对标已有 Kaggle Agent 和 AI Scientist 系统，建设一个能真实执行数据科学任务的科研 Agent 工作站。

我理解这个任务的核心是：利用我现有的 AI 工作站、Codex/Claude Code、GPU 服务器和自动化实验能力，做一个可运行、可评测、可复盘的 Kaggle/科研数据任务 Agent。

这个系统要解决的问题不是“AI 会不会写代码”，而是“AI 能不能稳定完成一个真实数据竞赛流程”：理解任务、分析数据、写代码、调试错误、训练模型、生成提交文件、记录实验、形成报告。

## 2. 老师给的材料说明了什么

### AutoKaggle

AutoKaggle 是一个面向 Kaggle 数据竞赛的多 Agent 框架。它把比赛流程拆成六个阶段：背景理解、初步 EDA、数据清洗、深度 EDA、特征工程、建模验证和预测。

它的 Agent 分工包括 Reader、Planner、Developer、Reviewer 和 Summarizer。这个设计说明，一个可用的 Kaggle Agent 不能只靠单个模型直接写代码，而要有明确分工、阶段控制、代码执行、调试、单元测试和报告生成。

AutoKaggle 的结果也说明了现实问题：它在 8 个 Kaggle 任务上验证提交率约为 0.85，综合分约为 0.82，效果有用但不是完全可靠。因此我的目标不能写成“一步全自动完成比赛”，而要写成“先做稳定可控的科研 Agent 工作流”。

### Agent K

Agent K 是更强的 Kaggle Agent 方向。它强调 scaffold 阶段和 post-scaffold 阶段：先根据数据和任务生成结构化解题脚手架，再用 ReAct 式流程持续改进方案。

它的文档还强调 Kaggle API、原始数据目录、任务 ID、时间预算、提交文件、leaderboard 记录和 GPU 设备控制。这说明真正的 Kaggle Agent 不是只生成一段代码，而是一个包含环境、数据、模型、提交和评测的完整系统。

但是 Agent K 的部署复杂度高，需要专门环境、Kaggle 登录、浏览器驱动、数据路径、多个第三方组件和较长运行时间。我的方案第一阶段不应该直接复刻 Agent K，而应该先做轻量版工作站原型，再逐步吸收它的 scaffold、ReAct、时间预算和 leaderboard 机制。

### AutoResearch AI 综述

AutoResearch AI 综述把 AI 科研自动化分成文献 grounding、假设生成、实验执行、验证反馈、报告传播等环节。它特别指出，现在很多系统的问题在于证据保存、可复现性、来源追踪、可靠性和人工监督不足。

这对我的方案很重要。我的科研 Agent 不能只追求自动化，而要把证据、日志、数据版本、实验结果和失败原因保存下来。否则即使模型跑出结果，也很难作为科研成果展示。

### Kaggle 任务

老师提到 Kaggle 正在进行或已经结束的比赛都可以用来评测系统。我的理解是，Kaggle 的价值不只是排行榜，而是它提供了明确的数据、指标、提交格式和外部评价机制。

因此我的第一版系统要优先选择表格类比赛，因为 AutoKaggle 和 Agent K 的已有经验主要也集中在 tabular classification/regression。等表格任务跑通后，再考虑图像、文本或多模态比赛。

## 3. 现有方案为什么还不够

我前一版计划的问题是太通用，只写了“数据分析、baseline、报告”，没有真正回应老师给的文献和项目。

需要改进的地方有四点：

1. 要对标 AutoKaggle 的阶段化流程，而不是泛泛写 EDA 和建模。
2. 要吸收 Agent K 的 scaffold + post-scaffold 思路，而不是只做一次性脚本。
3. 要加入可复现、证据保存、实验日志和失败复盘，这是 AutoResearch AI 综述强调的重点。
4. 要明确评测标准，用 Kaggle 的 valid submission、leaderboard score、运行时间和复现率判断系统效果。

## 4. 我的建设方案

我计划把系统命名为“科研 Agent 工作站”，第一版聚焦 Kaggle/科研表格数据任务。

系统采用“人机协同 + 阶段化 Agent”的方式建设。人负责确认研究目标、选择任务和判断结果是否有意义；Agent 负责执行重复性流程，包括任务解析、数据分析、代码生成、模型训练、错误调试、提交检查和报告生成。

第一版不追求完全自动化，而是先做到稳定完成一个真实比赛闭环。核心标准是：能跑、能提交、能记录、能复现、能说明为什么成功或失败。

## 5. 系统模块

### 任务理解模块

读取 Kaggle overview、data description、sample submission 和评价指标，生成结构化任务说明。

输出内容包括：任务类型、目标列、输入文件、提交格式、评价指标、可能的模型路线和风险点。

### 数据分析模块

检查数据规模、字段类型、缺失值、异常值、类别分布、训练集和测试集差异。

这个模块对应 AutoKaggle 的初步 EDA 和深度 EDA，不只是生成描述统计，而是要为数据清洗和特征工程提供依据。

### 数据清洗模块

处理缺失值、重复值、异常值、日期字段、类别字段和无效字段。

每个清洗动作都要记录原因，避免 Agent 随意删除字段或制造数据泄漏。

### 特征工程模块

实现基础特征工程，包括 one-hot、frequency encoding、数值缩放、时间特征、交叉特征和重要性筛选。

这里要谨慎，因为 AutoKaggle 的实验显示，特征工程工具太复杂反而会增加调试难度。因此第一版只做稳定、可解释、可回滚的特征。

### 建模与验证模块

先训练 baseline，再逐步加入 LightGBM、XGBoost、CatBoost、RandomForest、神经网络或 GPU 加速模型。

验证方式要和比赛指标一致，不能只看训练集分数。每次实验都要保存模型、参数、指标和输出文件。

### 调试与单元测试模块

这是必须加强的模块。AutoKaggle 的结果说明，没有单元测试时任务完成率会明显下降。

第一版至少要检查：数据文件是否存在、目标列是否正确、训练集和测试集字段是否对齐、submission 行数是否正确、列名是否符合 sample submission、预测值类型是否符合要求。

### 复盘与报告模块

每次实验结束后生成报告，包括任务背景、数据情况、方法、指标、提交文件、错误记录、失败原因和下一步计划。

这个模块对老师最有价值，因为它能展示系统不是“黑箱自动跑”，而是有清楚的科研过程。

## 6. 实施步骤

第一步，复现 AutoKaggle 的输入格式。准备 `overview.txt`、`train.csv`、`test.csv`、`sample_submission.csv` 四类文件，让系统先能读懂比赛。

第二步，选择一个简单表格比赛作为第一任务，例如 Titanic、House Prices 或一个老师指定的真实数据案例。

第三步，完成任务理解、EDA、baseline、submission 检查和实验日志。这个阶段先不追求高分，只追求完整闭环。

第四步，加入 AutoKaggle 式阶段控制，把流程拆成背景理解、EDA、清洗、特征工程、建模、报告六个阶段。

第五步，加入 Agent K 式 scaffold。让系统先生成解题脚手架，包括数据路径、指标、验证方案、模型路线和风险点，再进入代码执行。

第六步，加入 post-scaffold 改进。根据验证分数、错误日志和 submission 检查结果，自动提出下一轮修改方向。

第七步，接入 GPU 服务器。GPU 只用于真正需要算力的部分，如深度模型、超参数搜索、大规模特征处理或批量实验。

第八步，用 Kaggle 分数和实验报告评估系统。如果 valid submission、复现率、运行时间和报告质量都稳定，再扩展到更复杂比赛。

## 7. 评测标准

我会用以下指标判断系统是否有效：

- 是否生成 submission 文件。
- submission 是否符合 sample submission 格式。
- 是否能成功提交或通过本地提交检查。
- 验证集指标是否与比赛指标一致。
- 实验是否能复现。
- 每次失败是否有明确错误日志。
- 是否能生成老师可阅读的报告。
- 是否能在第二个任务中复用第一轮经验。

这比单纯看一次模型分数更重要，因为我的目标是建设科研 Agent 工作站，而不是只调出一个分数。

## 8. 预期成果

短期成果：完成一个 Kaggle 表格比赛闭环，包括任务说明、EDA、baseline、submission、实验日志和报告。

中期成果：形成一个轻量版科研 Agent 原型，支持阶段化流程、代码执行、调试、实验记录和报告生成。

长期成果：扩展为可服务 Kaggle 比赛和老师科研案例的 AI Data Scientist 工作站，具备任务迁移、经验复用和持续改进能力。

## 9. 我需要重点改进的地方

1. 不能只写通用计划，要把 AutoKaggle 和 Agent K 的流程吸收到自己的系统里。
2. 不能只做脚本，要做实验记录和可复现工作流。
3. 不能只追求自动化，要保留人工审核和科研判断。
4. 不能一开始做太大，要先从表格比赛跑通闭环。
5. 不能把 GPU 当噱头，要明确 GPU 用在训练、搜索和批量实验。
6. 不能忽略失败案例，要把失败变成系统下一轮改进的经验。

## 10. 给老师的简洁表述

我的计划是把现有 AI 工作站建设成一个面向 Kaggle 和科研数据任务的 Agent 系统。这个系统参考 AutoKaggle 的多 Agent 分工、Agent K 的 scaffold + ReAct 改进流程，以及 AutoResearch AI 对可复现和证据追踪的要求。

第一阶段我会先选择一个表格类 Kaggle 比赛或老师提供的数据案例，跑通任务理解、数据分析、baseline 建模、提交检查、实验记录和报告生成。这个阶段不追求完全自动化，而是先保证系统能真实完成任务、能复现结果、能记录失败原因。

后续我会逐步加入特征工程、模型对比、GPU 加速、自动调试和多轮改进，让它从一个可控原型发展成可以服务科研数据任务的 AI Data Scientist 工作站。

## 11. 参考材料

- AutoKaggle: A Multi-Agent Framework for Autonomous Data Science Competitions, arXiv:2410.20424.
- AutoKaggle GitHub: https://github.com/multimodal-art-projection/AutoKaggle
- Kolb-Based Experiential Learning for Generalist Agents with Human-Level Kaggle Data Science Performance, arXiv:2411.03562.
- Agent K GitHub/docs: https://github.com/huawei-noah/HEBO/tree/dev-agent/Agent_K
- AutoResearch AI: Towards AI-Powered Research Automation for Scientific Discovery, arXiv:2605.23204.
- Kaggle competitions: https://www.kaggle.com/competitions
