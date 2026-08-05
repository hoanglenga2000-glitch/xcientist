# MLEvolve 深度移植映射

## 迁移边界

本项目已将 `https://github.com/InternScience/MLEvolve` 克隆到 `external-projects/MLEvolve`，用途是工程逆向和协议移植。它不能绕过本项目的 AI 科研工作站主流程：训练、提交、报告仍必须由 `AgentOrchestrator` 或工作站 API 发起，并产生 run id、agent trace、artifact manifest、Gate 和 claim audit。

## 已落地的映射

| MLEvolve 机制 | 工作站落点 | 说明 |
|---|---|---|
| Progressive MCGS | `src/research_os/search_graph.py` | 增加 branch-diverse top candidates、global stagnation、reference edges。 |
| Cross-branch fusion/reference | `reference_edges` 与 `cross_branch_references` | 记录跨分支信息流，不改变父子血缘，便于审计。 |
| Retrospective Memory | `memory_reuse_records` 与 `RetrospectiveMemoryStore` | 失败和成功策略进入可复用记忆，避免重复失败路线。 |
| Base / Stepwise / Diff | `src/research_os/mlevolve_controller.py` | 根据 parent、stagnation 和 failure_count 选择代码生成模式。 |
| Submission quality gate | `submission_audit.json` + `rank_promotion_gate.json` | schema/local 候选与官方 top-30 排名分开判断。 |
| Benchmark claim control | `benchmark_claim_gate.json` | 少量任务只能输出 preliminary result，禁止宣称达到 75-task MLEvolve 水平。 |

## 当前真实基线

- 官方 Kaggle 提交记录：`spaceship-titanic` 一次。
- Public score：`0.80523`。
- Rank：`767 / 2084`。
- Rank percentile：`0.368042`。
- Top-30 Gate：未达到，下一轮必须进入自进化提升。

## 下一步接入方式

1. 工作站 Search Controller 读取 `search_controller_decision.json`、`research_os_search_graph.json`、`task_benchmark_state.json`。
2. Code Agent 只根据 bounded context 生成 Base / Stepwise / Diff 代码草稿。
3. Trainer/HPC Agent 通过工作站 run manifest 执行训练。
4. Validation Agent 生成 metrics、OOF、submission audit。
5. Human Gate 批准后才允许官方 Kaggle submit。
6. Kaggle response 回写 `kaggle_official_submission.json`，再由 `rank_promotion_gate.json` 判断是否达到前 30%。

## 禁止事项

- 不允许 Codex 直接训练或直接提交来绕过工作站。
- 不允许把本地 CV/OOF 写成官方排名。
- 不允许只用 3 个任务结果宣称达到或超过 MLEvolve 75-task medal rate。
- 不允许失败任务缺席 benchmark gap report。
