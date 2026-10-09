# EvoMind 五赛训练 Goal v2 — 官方/公开强基线

本文件是 `FIVE_COMPETITION_HUMAN_AVERAGE_GOAL_20260829.md` 的用户确认覆盖层。原始 Phase A 人类基准证据保留，不删除、不改写。

## 用户确认后的目标定义

对 `cure_bench`、`e2lmc`、`mindgames`、`open_polymer`、`ariel_2025` 五项，使用固定 G21、单 GPU 串行训练。每项必须在独立、无泄漏评估中超过下列最先可用且可验证的门槛：

1. 官方组织者发布并可执行复算的 baseline；
2. 若官方没有可执行 baseline，则采用有公开代码、固定版本、同一数据/指标/协议且结果可复算的强基线；
3. 若存在同协议、可验证的人类参与者平均，则候选还必须同时超过该人类门槛。

禁止把排行榜均值、队伍中位数、Prompt 数字、历史宣传、训练内 CV 或不可复现的论文点估计作为目标门槛。每个 baseline 必须记录来源、版本、代码/配置 SHA-256、数据 manifest SHA-256、指标公式与方向、复算命令和实际复算值。

## 自主权限

无需重复询问即可持续执行：任务/数据审计、workspace 文件创建与补丁、单 GPU canary、baseline 复现、受控超参搜索、checkpoint、失败诊断、可逆修复、独立 holdout 评估、artifact 发布、memory search/writeback 和报告生成。

仍须停在系统强制 Gate：删除数据、修改或终止非本 Run 进程、提权、付费、接受条款、Join、外部提交、创建/更换账号、改变 HPC profile/binding、替代代理或访问已移除的 Weather 数据。

## 每赛推进条件

1. 当前 `FULL_DATA_READY` 回执、manifest 与 loader 重新核对。
2. 从数据内 README/notebook/evaluator 和第一方来源冻结官方任务、指标与 baseline。
3. 按实体建立 group-aware split；实体交集必须为 0。官方隐藏测试标签不得用于选择。
4. `training_route` 后执行不超过 10 分钟的真实 canary。
5. 在相同 split/metric 上复现 baseline；复现偏差超过预声明容差时先修协议，不得训练更大模型掩盖问题。
6. 一次只改变一个主要因素；相同失败签名且前置条件未变化时禁止重试。
7. 训练内选择后冻结 pipeline，只在未触碰 holdout 上评估一次；用适用的 bootstrap/重复种子得到 95% CI。
8. higher-is-better：候选 CI 下界高于 baseline；lower-is-better：候选 CI 上界低于 baseline。若 baseline 只有确定性复算值，最小胜出裕量取 1% 相对尺度或候选一个标准误中的更严格者。
9. 未通过则保留 champion、记录失败分支并继续新的有根据候选；通过后执行独立复算与 `memory_writeback`。

## 顺序与资源

初始顺序按最小真实成本发现问题：`cure_bench → e2lmc → mindgames → open_polymer → ariel_2025`。允许在任务契约证据表明另一顺序显著降低风险时调整，但同一时刻只能有一个 GPU solution。Ariel 必须最后运行，先做分块读取与极小 canary，禁止一次性载入全量数据导致 OOM。

每个 GPU 动作前必须重新完成指定代理、固定主机键、allocation role、Host/GPU UUID、GPU 型号/显存、允许根和 5/5 采样。所有输出仅在 Run 隔离目录和允许根内；不得修改其他 GPU 进程。

## 强制产物

每项至少发布：`task-contract-v2.json`、`baseline-evidence.json`、`baseline-reproduction.json`、`dataset-audit.json`、`split-manifest.json`、`solution.py`、`training-config.json`、`training.log`、`candidate-history.jsonl`、`metrics.json`、`independent-verification.json`、`candidate-vs-baseline.json`、`retrospective-memory.json`、`artifact-manifest.json`。

全局持续更新：`five-competition-goal-board-v2.json`、`resource-schedule.json`、`cross-competition-evidence-index-v2.json`、`five-competition-final-report.md`、`five-competition-final-manifest.json`。

## 完成状态

只有五项均满足当前数据、baseline 复现、独立评估胜出、完整 SHA 证据和独立复算时，才允许 `VERIFIED_COMPLETE`。若遇到物理资源、官方协议缺失或外部服务限制，继续处理其他项；仅当所有剩余项目共享同一个不可自动化事实时，返回一个 `WAITING_EXACT_GATE`。
