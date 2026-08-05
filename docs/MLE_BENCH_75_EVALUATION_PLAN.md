# MLE-Bench 75 长期评测计划

## 1. 为什么选择 MLE-Bench 75 个 Kaggle 任务

MLE-Bench 以 Kaggle 风格竞赛任务作为机器学习工程能力评测对象，覆盖数据解析、特征工程、模型选择、训练调参、提交格式、排行榜反馈和复现报告等完整流程。它比单一 demo 更接近真实 MLE 工作：每个任务都有不同数据形态、metric、baseline 难度、依赖环境和 leaderboard 风险。

本项目的目标是构建面向 Kaggle/MLE-Bench 的可审计自进化 AI 科研工作站，因此需要一个长期、系统、可复现的 benchmark。75 个任务可以检验：

- 工作站是否能稳定产生 valid submission；
- Search Controller 是否能跨任务沉淀策略并提高 medal rate；
- Research Harness 是否能防止过度声称、leaderboard 过拟合和 claim drift；
- HPC/GPU、artifact、Gate、report 是否能在多任务上持续闭环。

## 2. 本项目如何对齐 MLEvolve

本项目对齐 MLEvolve 的方向不是简单复制论文结果，而是把 MLEvolve 的核心机制工程化到工作站：

- Progressive MCGS 风格多分支搜索：用 search graph 表达 baseline、model family、feature、tuning、ensemble、ablation 等路线；
- Retrospective Memory：把成功策略、失败模式、数据画像和 metric delta 沉淀为可复用记忆；
- Adaptive Code Generation：把 Base / Stepwise / Diff 三种代码生成模式交给受控 Code Agent；
- Exploration 到 exploitation：前期优先 robust baseline 和覆盖模型族，后期围绕 top candidates 做调参、融合、校准；
- 跨分支参考：允许 LightGBM 分支学习 XGBoost 的特征处理，允许 Ensemble 聚合多个 top nodes；
- 基于 CV、OOF、public/private score、稳定性、运行时间和风险 flags 决定下一轮实验。

同时，本项目增加 XCIENTIST-style Research Harness，要求每次实验都有 validation contract、claim boundary 和 claim audit，避免只追求 public score。

## 3. 不能直接声称超过 MLEvolve

本项目不能直接声称“已经超过 MLEvolve”。只有在以下条件满足后，才允许进行强对比声明：

- 使用相同或可对齐的 75 个 MLE-Bench/Kaggle 任务；
- 任务预算、时间预算、提交限制、硬件条件和模型调用条件可说明；
- medal 判定方式与 MLEvolve 报告一致或明确可换算；
- 所有任务都纳入统计，不能只报告成功任务；
- 每个任务都有 valid submission、artifact、reproducibility report 和 claim audit；
- benchmark result 可复现，并能追溯到具体 exp_id、metrics 和提交记录。

如果当前系统无法达到 MLEvolve 报告的 medal rate，必须输出 benchmark gap report，说明差距、失败原因、瓶颈模块和下一轮优化计划。

## 4. 核心指标

- `valid_submission_rate`：产生至少一个合法提交的任务比例；
- `medal_rate`：获得任意 medal 的任务比例；
- `bronze_rate`：获得 bronze 的任务比例；
- `silver_rate`：获得 silver 的任务比例；
- `gold_rate`：获得 gold 的任务比例；
- `average_best_score_rank`：最佳提交在可对齐排行榜中的平均排名或百分位；
- `time_to_valid_submission`：从任务启动到第一份合法提交的耗时；
- `time_to_best_submission`：从任务启动到最佳提交的耗时；
- `number_of_experiments_per_task`：每个任务产生的实验节点数量；
- `reproducibility_score`：代码、配置、数据版本、随机种子、日志和 artifact 完整度；
- `auditability_score`：validation contract、risk checklist、claim audit、Gate 记录完整度；
- `claim_drift_rate`：报告 claim 发生 drift 的比例；
- `human_intervention_count`：人工介入次数，包括 Gate、修复、资源处理和策略调整。

## 5. 目标

### 短期目标

完成 3-5 个 tabular Kaggle 任务闭环，至少包括 task spec、baseline、search graph、validation contract、metrics、submission audit、claim audit 和 reproducibility report。

### 中期目标

完成 10-15 个任务的自动 baseline + search graph + validation contract，验证 Search Controller 能够跨任务复用 Retrospective Memory。

### 长期目标

覆盖 75 个 MLE-Bench 任务，所有任务均记录成功、失败、回退、artifact、报告和 claim audit。

### 对标目标

medal rate 尽量达到或超过 MLEvolve 报告水平。若未达到，必须生成 benchmark gap report，不能虚假宣称已经超过。

## 6. 评测原则

- 不允许泄露测试标签；
- 不允许手动改 leaderboard 结果；
- 不允许只报告成功任务；
- 失败任务也必须记录；
- 每个任务必须保存 artifacts；
- 每个任务必须有 reproducibility report；
- 每个任务必须有 claim audit；
- 每个任务必须记录运行预算、提交限制和人工介入；
- 没有 private leaderboard 或 medal 判定时，只能标记为 proxy evaluation；
- 只测试少数任务时，只能称为 preliminary result；
- 任何“达到或超过 MLEvolve”的说法必须通过 benchmark gate 和 claim audit。
