# G21 五赛人类参与者基线研究协议

## 文档边界

- 固定 Run：`run_7b1efb878afb40f396db431e91f093a5`
- 固定 allocation：`G21`
- Goal spec SHA-256：`380a3b2c037c8067a4618717cb7aaf8847d323d4be401a88d64009468132848f`
- 已验证负证据 SHA-256：`5b6ed3d912ad1cd4655c96f832629cf9603a4b0ad58ab6e27c01861fadd69c5f`
- 本文只定义如何生成合格证据，不包含参与者数据、实验结果或人类均值，也不授权训练、HPC、提交、Join 或条款接受。
- 任何仅由脚本生成的数字、模型 leaderboard、组织者模型 baseline、专家评审意见或单个匿名玩家记录，都不能填写为 `human_baseline`。

## 可行性结论

| 比赛 | 人类能否执行目标任务 | 当前公开协议能否直接产出可比人类均值 | 当前用户/系统能否独立完成 | 精确缺口 |
|---|---|---|---|---|
| CURE-Bench | 可以；医学多项选择题是直接人类任务 | 条件可行；必须有全新、未消费、官方标注的盲测题集 | 否 | 合格医学参与者、伦理/同意、未消费官方盲测题及权威发布物 |
| MindGames | 可以；官方 starter kit 提供人工交互入口 | 条件可行；同一冻结四游戏、对手池、seed/座位和 TrueSkill 协议可复算 | 否 | 人类 cohort、托管 `textarena==0.7.4`/`trueskill`、冻结对手池和正式研究发布物 |
| E2LMC | 当前规则下不是直接的人类作答任务；提交物是早训评估方法/指标 | 不可直接比较 | 否 | 组织者必须新增盲化的“人类专家选择评估指标”私有 evaluator |
| Open Polymer | 专家可给数值估计，但公开比赛没有人类预测协议 | 当前不可比较 | 否 | 组织者必须提供同一私有分子集合上的人类预测入口和官方 wMAE 评分；本地 normalized RMSE 不能代替 |
| Ariel 2025 | 原始 283 维光谱回归不构成现实的徒手人类任务；若允许分析工具则测量的是专家工作流 | 当前不可比较 | 否 | 组织者必须定义允许工具/时间预算的人类专家盲评 endpoint，并以官方指标私有评分 |

“条件可行”只表示协议在方法上可以定义，不表示已经获得基线。五项当前仍保持 `HUMAN_BASELINE_UNDEFINED`。

## 通用预注册与执行字段

### 1. 研究身份

- `schema`: `evomind.human-participant-baseline-study.v1`
- `study_id`: 稳定、安全的研究 ID
- `competition`: 五赛之一
- `run_id`: 固定 Run
- `allocation`: `G21`
- `protocol_id`、`protocol_version`
- `protocol_text`、`protocol_sha256`
- `preregistration_url` 或 DOI
- `title`
- `source_authority`: 只能是 `official_organizer`、`peer_reviewed` 或 `official_dataset`
- `organizer_protocol_url`、`organizer_protocol_sha256`
- `created_at_utc`、`frozen_at_utc`

### 2. 参与者与伦理

- `participant_type`: 固定为 `human`
- `target_population`
- `inclusion_criteria`、`exclusion_criteria`
- `qualification_test_id`、`qualification_test_sha256`
- `experience_bands` 与预先声明的分层方案
- `age_policy`、`language_policy`
- `consent_form_sha256`、`ethics_review_reference`
- `compensation_policy`
- `planned_sample_size`、`minimum_sample_size`
- `sample_size_rationale`: 预注册精度或 power 计算，不得事后按结果停止
- `enrolled_count`、`excluded_count`、`included_count` 及逐原因计数
- `participant_id_scheme`: 仅保存不可逆的研究 ID，不进入姓名、邮箱等 PII

所有研究至少先达到 `n >= 30` 个独立参与者，再继续按预注册的 CI 精度停止规则收集；如果 power/precision 计算要求更大样本，以更大者为准。该下限是研究设计门槛，不是官方给出的样本量，也不能在没有真实 cohort 时写入证据结果。

### 3. 任务、数据与盲法

- `task_definition`
- `official_metric`、`metric_direction`
- `task_manifest_id`、`task_manifest_sha256`
- `item_or_environment_count`
- `item_id_sha256` 或 `seed_schedule_sha256`
- `answer_key_or_private_evaluator_sha256`
- `official_scoring_code_sha256`
- `time_budget_seconds`
- `allowed_tools`、`forbidden_tools`
- `training_or_familiarization_protocol`
- `randomization_seed`、`assignment_manifest_sha256`
- `seat_balance` 或 `item_order_balance`
- `participant_blind_to_labels`: 必须为 `true`
- `participant_blind_to_candidate_outputs`: 必须为 `true`
- `analyst_blind_to_condition_until_freeze`: 必须为 `true`
- `candidate_blind_to_human_responses`: 必须为 `true`
- `holdout_ledger_id`、`holdout_ledger_sha256`
- `holdout_claim_id`、`holdout_claim_sha256`
- `prior_holdout_overlap`: 必须为 `0`
- `test_labels_exposed`: 必须为 `false`

### 4. 原始观测与评分

- `response_schema_version`
- 每条记录的 `participant_id`、`task_id`/`environment_id`、`attempt_id`、`seat`、`started_at`、`completed_at`、`response`/`actions`、`validity_code`
- `raw_response_manifest_sha256`
- `scored_response_manifest_sha256`
- `invalid_response_policy`: 预注册，不得按输赢后验删除
- `primary_aggregation_unit`: 人类参与者
- `human_participant_scores`: 每人按官方指标计算的分数
- `mean`
- `uncertainty.lower`、`uncertainty.upper`
- `uncertainty.confidence_level`: `0.95`
- `uncertainty.method`
- `uncertainty.source_sha256`: 必须与公开/权威人类基线源 artifact SHA 相同
- `sensitivity_analyses`: 至少含未排除集、合格集和分层结果

默认 CI 使用参与者为一级 cluster 的 bootstrap。存在题目/环境重复测量时使用参与者 × 题目（或参与者 × seed）的双层 bootstrap；游戏的 TrueSkill 同时报告后验不确定性，但不能用单局 bootstrap 冒充独立参与者 CI。

### 5. 可交付 artifact

合格证据最少包含以下 regular files，并由一个 closure manifest 绑定路径、bytes 和 SHA-256：

1. `protocol.json`
2. `preregistration-receipt.json`
3. `participant-flow.json`
4. `qualification-manifest.json`
5. `task-manifest.json`
6. `assignment-manifest.json`
7. `holdout-ledger-receipt.json`
8. `raw-responses.redacted.jsonl` 或等价的脱敏不可变格式
9. `scored-responses.jsonl`
10. `participant-metrics.json`
11. `human-baseline.json`
12. `bootstrap-replicates-or-seed-receipt.json`
13. `leakage-audit.json`
14. `independent-review.json`
15. `artifact-manifest.json`

`human-baseline.json` 必须满足运行时 `validate_human_baseline` 的字段：

- `status=VERIFIED`
- `participant_type=human`
- `source_url` 或 `source_doi`
- `source_authority`
- `source_sha256`
- `evidence_artifact_id`
- `evidence_artifact_sha256 == source_sha256`
- `title`
- `sample_size >= 2`，但本协议另要求实际研究 `n >= 30` 或 power/precision 计算的更大值
- `protocol_id`
- `protocol`
- `protocol_sha256 == SHA256(UTF-8 protocol)`
- `protocol_comparable=true`
- `metric`
- `direction`
- `mean`
- `uncertainty={lower, upper, confidence_level:0.95, method, source_sha256}`

## 分赛最小协议

### CURE-Bench

**资格与样本**

- 目标人群：持证临床医生，或预注册后单独分层报告的高年级医学生/住院医师；不得混成一个未分层均值。
- `n >= 30`，并持续到参与者级 accuracy 的 bootstrap 95% CI 半宽达到预注册阈值。
- 每名参与者回答相同的全新官方盲测题，建议至少 100 题；题型、医学领域和选项数量按官方分布分层。

**任务和指标**

- 只展示官方 `question`、`question_type` 和 `options`；不得展示答案、模型输出或 leaderboard。
- primary metric：逐参与者 accuracy，再对参与者取均值，`higher_is_better`。
- 模型和人类必须在同一 item manifest 上用同一 answer-key/scorer 评估。
- CI：参与者 × 题目的双层 bootstrap；另报告题型/资历分层，但不得以最佳分层替代总体 primary metric。

**当前可执行性**

- 当前系统可实现本地界面、随机化、日志和 scorer。
- 当前系统不能独立生成合格证据：缺少真实医学 cohort、伦理/同意及全新未消费的官方盲测题。组织者至少需提供一次性私有题集/evaluator，或发布已完成、含均值与 CI 的权威人类研究。

### MindGames

**资格与样本**

- 目标人群：年满 18 岁、能理解比赛语言并通过每个游戏规则测验的人类；按棋盘/社交推理游戏经验分层。
- `n >= 30`；每人每个环境至少 8 局，总计至少 32 局，座位与先后手平衡。
- 冻结环境：`Codenames-v0`、`ColonelBlotto-v0`、`ThreePlayerIPD-v0`、`SecretMafia-v0`。

**任务和指标**

- 使用同一 `textarena==0.7.4`、`trueskill`、starter-kit commit、offline evaluator、对手池、随机 seed 和座位表。
- primary metric：同一冻结对手池下的官方 TrueSkill（以官方 leaderboard 使用的参数化定义为准），`higher_is_better`。
- secondary：平均 reward difference、win/draw/loss、invalid move rate。
- CI：参与者级 cluster bootstrap；TrueSkill 后验 `mu/sigma` 独立报告。任何 `Test_Human_Player` 单行或 3 局记录都不构成 cohort。

**当前可执行性**

- 协议和 harness 可以独立实现，不必改变比赛定义。
- 当前用户/系统仍不能独立完成证据：缺少真实 cohort、当前托管依赖/冻结参考池 Gate 尚未满足，也没有 peer-reviewed/organizer 发布的人类 cohort artifact。

### E2LMC

**为什么当前不可直接比较**

- 公开规则评估的是早期训练语言模型的 evaluation suite/metric，通过 SQ、RC、CS 和隐藏 checkpoints 评分；“人类答题分数”与该提交对象不是同一 estimand。
- 把人类在 MMLU 等题目上的 accuracy 作为基线会比较不同任务，应拒绝。

**组织者必须新增的协议/endpoint**

- 向至少 30 名合格的语言模型评估研究者展示盲化的 early-checkpoint curves 与允许的候选 evaluator 列表。
- 让每名专家在固定预算内选择/排序 evaluator 或预测最终 checkpoint ranking。
- 用未公开的全新训练 experiment 在同一个 SQ/RC/CS scorer 上评分人类选择；同一 experiment 也评估候选系统。
- endpoint 返回 participant-level SQ/RC/CS、composite、私有 experiment manifest SHA、无重叠证明和组织者签名 receipt。
- CI 以参与者为 cluster、experiment 为第二层 bootstrap。

当前用户/系统只有公开曲线和已消费 experiment，不能独立创建新的 organizer-private experiment，也不能自证 hidden checkpoint 未泄漏。

### Open Polymer

**为什么当前不可直接比较**

- 官方 primary metric 是 lower-is-better `wMAE`；当前本地 normalized RMSE 在误差形式、权重、缺失标签聚合和 test scope 上均不等价。
- 参赛者/团队数量和模型 submission 数不是人类逐样本预测均值。

**组织者必须新增的协议/endpoint**

- 参与者：至少 30 名聚合物/材料信息学专家；资历分层，预注册允许的软件、数据库和时间预算。若允许计算工具，结果必须标记为 `human_expert_workflow`，不能标成徒手人类认知成绩。
- 任务：同一未公开 molecule/target manifest，要求对 `Tg`、`FFV`、`Tc`、`Density`、`Rg` 输出数值；所有人和模型用完全相同的有效 pair。
- primary metric：官方 wMAE，`lower_is_better`；同一官方 scorer 计算 participant-level wMAE。
- CI：参与者 × polymer 的双层 bootstrap；缺失预测按预注册且与官方提交规则一致的方式处理。
- endpoint 仅返回评分与 SHA 绑定 receipt，不公开 test labels；同时提供本地候选用官方 wMAE 重评分的结果，从而关闭 RMSE→wMAE mismatch。

没有该 endpoint 或组织者发布的人类 wMAE cohort，当前用户/系统不能独立补齐证据。

### Ariel Data Challenge 2025

**为什么当前不可直接比较**

- 目标是每颗行星 283 维 transmission spectrum 回归。没有固定工具策略时，徒手人类预测不是可解释或可复现的比较对象。
- 当前所有内部行星 holdout 已消费；本地 normalized RMSE 也不能替代未核实的官方私有 scorer。

**组织者必须新增的协议/endpoint**

- 参与者：至少 30 名系外行星光谱/仪器系统学专家；若现实招募量不足，必须由 power/precision 计算给出更大不确定性并如实报告，不能降低运行时的人类 cohort 要求来制造通过。
- 冻结一套允许的分析软件、参考资料、计算预算和时间；记录每个版本 SHA。结果名称必须是 `human_expert_workflow`。
- 向人类和候选提供同一盲化 planet manifest、相同输入信号和 star metadata；输出同一 283 维预测 schema。
- 使用同一官方私有 scorer，`metric` 和 `direction` 由组织者签名的 scorer manifest 定义；不得把内部 normalized RMSE 事后宣称为官方指标。
- CI：参与者 × planet 双层 bootstrap；组织者 receipt 证明 test label 不出 endpoint、内部 holdout 无重用、candidate 与人类输入一致。

没有该 organizer-private evaluator 或一份已发表的人类专家工作流研究，当前用户/系统不能独立补齐证据。

## 训练解锁判定

以下全部为真时，单项才可从 `WAITING_EXACT_GATE` 进入 `READY_FOR_GPU`：

1. 实际人类研究完成，不是只有 protocol 或招募计划。
2. 权威来源公开或由组织者签名，且 source bytes 与 artifact SHA 闭合。
3. `human-baseline.json` 通过运行时 `validate_human_baseline`。
4. 人类与候选在同一任务、输入、scorer、metric、direction 和聚合协议上比较。
5. 未消费 holdout/private evaluator 已原子 claim，所有 overlap 为 0。
6. 参与者和候选均未见 test labels 或对方输出。
7. 真实 cohort 均值及 95% CI 已独立复算。
8. 对应的非人类 Gate（冻结模型、依赖、private evaluator 等）同时满足。

仅完成本文、编写采集界面、取得招募意向或生成空 JSON，都不能解锁 GPU 训练。
