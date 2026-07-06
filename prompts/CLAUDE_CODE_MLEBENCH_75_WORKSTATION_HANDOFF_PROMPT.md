# Claude Code 接手 Prompt：AI 科研工作站 + MLE-Bench 75

你现在接手我的 AI 科研工作站长期任务。你的角色是高级工程实现与监督 Agent，不是绕过系统手工训练模型的人。

工作目录：

```powershell
D:\桌面\codex\科研港科技
```

## 第一原则

所有 Kaggle / MLE-Bench 训练、代码生成、HPC/GPU 调度、实验记录、提交候选、报告生成，都必须通过我的 AI 科研工作站系统发起和留痕。

禁止：

- 禁止绕过工作站直接训练。
- 禁止使用本地 GPU。
- 禁止默认 Kaggle 官方提交。
- 禁止读取、打印、写入 token、API key、SSH 密码。
- 禁止把 CV/proxy 分数伪装成官方排名或奖牌。
- 禁止虚构 MLE-Bench 75 已完成。

## 先恢复系统记忆

请先阅读：

- `docs/WORKSTATION_CODE_RUNTIME_MAP_20260630.md`
- `reports/WORKSTATION_LAUNCH_READINESS_20260630.md`
- `reports/WORKSTATION_TASK_API_MATRIX_20260630.md`
- `reports/WORKSTATION_TRAINING_PROGRESS_20260630.md`
- `docs/THREE_LAYER_RESEARCH_OS_ARCHITECTURE.md`
- `docs/MLE_BENCH_75_EVALUATION_PLAN.md`
- `docs/ROADMAP_TO_MLE_BENCH_75.md`
- `D:\桌面\claude code\log\today\06-Day2-系统稳定与铜牌攻克.md`
- `D:\桌面\claude code\log\today\02-GPU集群连接方案.md`

然后全局搜索：

- `AgentOrchestrator`
- `workstation-actions`
- `workstation-run-contract`
- `workstation-closed-loop`
- `mlevolve_controller`
- `retrospective_memory`
- `validation_contract`
- `claim_audit`
- `deepseek-cache`
- `gpu-ssh-gateway`
- `rank_promotion_gate`
- `benchmark_claim_gate`

## 先运行只读验证

```powershell
cd D:\桌面\codex\科研港科技
python scripts\verify_workstation_launch_readiness.py --include-frontend --write-report
python scripts\verify_workstation_task_api_matrix.py --write-report
python scripts\build_workstation_training_progress_report.py --write-report
python scripts\build_kaggle_experiment_inventory.py
python scripts\build_mlebench_style_leaderboard_report.py
python scripts\verify_no_plaintext_secrets.py
```

前端：

```powershell
cd D:\桌面\codex\科研港科技\web\research-agent-workstation
npm run typecheck
npm run build
```

如果 build 后页面报 chunk 相关错误，重启 Next 服务：

```powershell
npm run dev
```

默认页面：

```text
http://127.0.0.1:8088/?page=overview
```

## 当前已知状态

截至 2026-06-30 最近报告：

- 已有实验任务：11
- 观测 run：740
- 有分数 run：358
- promoted：34
- held：62
- timeout/failed：2
- 官方提交任务：2
- 官方 top30 任务：1
- medal count/rate：0 / 0.0
- benchmark claim：`not_comparable_not_reached`
- launch state：`demo_ready_training_blocked_by_gpu`
- blockers：`figma_auth_blocked`, `gpu_resource_blocked`

这些数字必须重新从 JSON/报告读取确认，不要只相信 prompt。

## 系统四层架构

第一层：Multi-Agent Research OS

- 任务解析
- 数据审计
- baseline
- 代码实现
- HPC/GPU 执行
- CV/OOF
- submission schema
- Kaggle gate
- 实验台账
- 可复现报告

第二层：MLEvolve-style Search Controller

- 多分支搜索图
- Progressive MCGS
- exploration/exploitation 切换
- Retrospective Memory
- Base / Stepwise / Diff 代码生成
- LightGBM / XGBoost / CatBoost / NN / Ensemble / Stacking
- best-so-far 保护
- stagnation 检测

第三层：XCIENTIST-style Research Harness

- hypothesis
- validation contract
- metric
- ablation
- risk check
- claim boundary
- claim drift audit

第四层：Memory / Benchmark Evolution Layer

- 跨任务经验复用
- 失败归因
- 成功策略沉淀
- MLE-Bench 75 长期评测
- medal/top30/valid submission 统计

## 下一步任务

先不要训练，先输出：

1. 系统理解报告。
2. 当前运行状态报告。
3. 任务记忆恢复报告。
4. 当前 Kaggle/MLE-Bench 任务矩阵。
5. 哪些任务 full reportable，哪些只 minimum loop，哪些缺 evidence/report。
6. 当前 GPU/HPC gate 状态。
7. DeepSeek cache 状态。
8. 下一轮 3-5 个任务工作站自进化计划。

然后再由工作站发起任务。

## 训练必须满足的 artifact

每个实验必须生成：

- `workstation_run_id`
- `agent_trace.json`
- `search_controller_decision.json`
- `validation_contract.json`
- `code_generation_record.json`
- `gpu_job_manifest.json`
- `metrics.json`
- OOF prediction
- `submission.csv`
- `submission_audit.json`
- `rank_promotion_gate.json`
- `claim_audit.json`
- `retrospective_memory_update.json`
- `final_report.md`

失败时必须生成：

- `failure_review.json`
- `retry_or_rollback_decision.json`
- `memory_failure_pattern.json`

## MLE-Bench 75 策略

不要一次性盲跑 75 个任务。

Phase 1：

- 恢复已有任务。
- 补齐 evidence、report、claim audit。

Phase 2：

- 选择 3-5 个 tabular 任务完整闭环。
- 优先 valid submission rate。

Phase 3：

- 扩展到 10-15 个任务。
- 统计 valid submission rate、top30 rate、proxy medal gap。

Phase 4：

- 接入完整 MLE-Bench 75 registry。

Phase 5：

- 对齐 MLEvolve 指标，生成 benchmark gap report。

## 优先任务

根据当前训练进度报告，优先：

1. `spaceship_titanic`
   - 已有官方提交但未达 top30。
   - 适合下一轮自进化校准。

2. `bike_sharing_demand`
   - 有最小闭环和 promoted run。
   - 需要补齐 report。

3. `playground_series_s6e6`
   - 已有官方 top30。
   - 可作为完整样板任务。

4. `titanic`
   - full reportable。
   - 可作为稳定样板任务。

5. `porto_seguro_safe_driver_prediction`
   - 有 promoted/held。
   - 需要补齐 evidence/report。

## 关键文件位置

系统地图：

- `docs/WORKSTATION_CODE_RUNTIME_MAP_20260630.md`

前端入口：

- `web/research-agent-workstation/src/app/page.tsx`
- `web/research-agent-workstation/src/components/workstation/Screens.tsx`
- `web/research-agent-workstation/src/components/workstation/OverviewBoardEnhanced.tsx`
- `web/research-agent-workstation/src/components/workstation/AppShell.tsx`
- `web/research-agent-workstation/src/components/workstation/Sidebar.tsx`
- `web/research-agent-workstation/src/components/workstation/navigation.ts`

后端入口：

- `web/research-agent-workstation/src/app/api/workstation-actions/route.ts`
- `web/research-agent-workstation/src/lib/server/workstation-actions.ts`
- `web/research-agent-workstation/src/lib/server/workstation-run-contract.ts`
- `web/research-agent-workstation/src/lib/server/workstation-closed-loop.ts`
- `web/research-agent-workstation/src/lib/server/gpu-ssh-gateway.ts`
- `web/research-agent-workstation/src/lib/server/deepseek-provider.ts`
- `web/research-agent-workstation/src/lib/server/deepseek-cache.ts`

Research OS：

- `src/research_os/search_graph.py`
- `src/research_os/retrospective_memory.py`
- `src/research_os/validation_contract.py`
- `src/research_os/claim_audit.py`
- `src/research_os/benchmark_manager.py`
- `src/research_os/mlevolve_controller.py`
- `src/research_os/mlevolve_adapter.py`

验证脚本：

- `scripts/verify_workstation_launch_readiness.py`
- `scripts/verify_workstation_task_api_matrix.py`
- `scripts/build_workstation_training_progress_report.py`
- `scripts/build_kaggle_experiment_inventory.py`
- `scripts/build_mlebench_style_leaderboard_report.py`
- `scripts/verify_no_plaintext_secrets.py`

## 最终交付要求

你完成每一阶段后必须输出：

1. 实际运行了哪些检查。
2. 哪些文件被新增或修改。
3. 哪些任务通过工作站完成闭环。
4. 哪些任务只是 proxy/local，不能宣称官方结果。
5. 官方提交次数、public score、rank、top30、medal evidence。
6. 当前 medal rate 的真实证据。
7. 下一轮自进化计划。

记住：你要训练和完善的是我的 AI 科研工作站系统能力，不是你自己在旁边手工参加比赛。
