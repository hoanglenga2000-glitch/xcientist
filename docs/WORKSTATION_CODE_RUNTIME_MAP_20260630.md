# AI 科研工作站代码系统运行地图

生成时间：2026-06-30

本文档用于在系统记忆丢失时快速恢复上下文，让后续工程 Agent、Claude Code 或老师审查者理解：当前 AI 科研工作站从前端、API、Agent、HPC/GPU、Kaggle、MLE-Bench 评测到报告审计的关键代码位置和运行方式。

## 1. 当前系统定位

本项目不是单个 Kaggle 训练脚本，而是一个面向 Kaggle / MLE-Bench 的可审计自进化 AI 科研工作站。训练、代码生成、HPC/GPU 调度、Kaggle 提交候选、实验报告和 claim audit 都必须通过工作站流程发起并留下 artifact。

当前系统按四层理解：

1. Multi-Agent Research OS：任务解析、数据审计、代码实现、训练执行、OOF/CV、submission gate、报告生成。
2. MLEvolve-style Search Controller：多分支搜索、Progressive MCGS、Base/Stepwise/Diff 代码生成、retrospective memory、best-so-far 保护。现有两条可切换引擎：`legacy`（原 `mlevolve_search` / MCGS，默认）与 `research_os`（已修正的 `src/research_os` EvolutionLoop，经 `evolution_engine_cli.py --mode ingest_summary` 桥接回工作站图谱并重放 promotion gate）。二者由 `runEvolutionCycle` 的 `input.engine` 选择，共用同一 GPU/local runner 协议与 Human Submission Gate（本地 CV 不声称 Kaggle 官方名次）。
3. XCIENTIST-style Research Harness：hypothesis、validation contract、risk check、ablation、claim boundary、claim drift audit。
4. Memory / Benchmark Evolution Layer：跨任务经验沉淀、失败归因、MLE-Bench 75 长期评测、medal/top30/valid submission 统计。

## 2. 启动与验证入口

### 2.1 前端工作站

目录：

- `web/research-agent-workstation`

关键命令：

```powershell
cd D:\桌面\codex\科研港科技\web\research-agent-workstation
npm run dev
npm run typecheck
npm run build
```

默认地址：

- `http://127.0.0.1:8088/?page=overview`

注意：如果在 Next dev server 运行时执行 `npm run build`，`.next` chunk 可能被重写，需要重启 dev/start 后再做在线 smoke。

### 2.2 一键上线总闸

```powershell
cd D:\桌面\codex\科研港科技
python scripts\verify_workstation_launch_readiness.py --include-frontend --write-report
```

输出：

- `workspace/workstation_launch_readiness_20260630.json`
- `reports/WORKSTATION_LAUNCH_READINESS_20260630.md`

当前最近结论：

- `status`: `passed`
- `launch_state`: `demo_ready_training_blocked_by_gpu`
- blockers: `figma_auth_blocked`, `gpu_resource_blocked`

### 2.3 训练进度统计

```powershell
python scripts\build_workstation_training_progress_report.py --write-report
```

输出：

- `workspace/workstation_training_progress_20260630.json`
- `reports/WORKSTATION_TRAINING_PROGRESS_20260630.md`

当前统计：

- 已有实验任务：11
- 观测 run：740
- 有分数 run：358
- promoted：34
- held：62
- timeout/failed：2
- 官方提交任务：2
- 官方 top30 任务：1
- medal count/rate：0 / 0.0

## 3. 前端页面与组件

### 3.1 页面入口

- `web/research-agent-workstation/src/app/page.tsx`

职责：

- 读取 URL 中的 `?page=...`
- 维护当前页面、任务、stage、experiment、gate 状态
- 调用 API client
- 将 props 分发给所有页面组件

### 3.2 App Shell 与导航

- `web/research-agent-workstation/src/components/workstation/AppShell.tsx`
- `web/research-agent-workstation/src/components/workstation/Sidebar.tsx`
- `web/research-agent-workstation/src/components/workstation/navigation.ts`

当前页面：

- `overview`: Research Overview
- `control`: AI Control Console
- `experiments`: Experiments
- `data`: Data & Kaggle Pipeline
- `report`: Report Studio
- `code`: Code Agent IDE
- `gpu`: GPU / HPC Console
- `evidence`: Evidence Ledger
- `gates`: Integrity Gates
- `literature`: Literature
- `tasks`: Task Queue
- `runtime`: Agent Runtime
- `workflow`: Workflow Graph
- `settings`: Settings

### 3.3 页面实现集合

- `web/research-agent-workstation/src/components/workstation/Screens.tsx`
- `web/research-agent-workstation/src/components/workstation/OverviewBoardEnhanced.tsx`
- `web/research-agent-workstation/src/components/workstation/AiControlConsole.tsx`
- `web/research-agent-workstation/src/components/workstation/PaperEvidenceBundle.tsx`
- `web/research-agent-workstation/src/components/workstation/Common.tsx`

### 3.4 前端 API client

- `web/research-agent-workstation/src/lib/api/client.ts`
- `web/research-agent-workstation/src/lib/api/types.ts`

用途：

- 前端页面所有按钮和 action 不应直接读写文件，而应通过 API client 调用后端路由。

## 4. 后端 API 路由

核心路由目录：

- `web/research-agent-workstation/src/app/api`

重要路由：

- `/api/workstation-summary`
  - `web/research-agent-workstation/src/app/api/workstation-summary/route.ts`
  - 汇总工作站状态、connector status、Kaggle/DeepSeek/GPU readiness。

- `/api/workstation-actions`
  - `web/research-agent-workstation/src/app/api/workstation-actions/route.ts`
  - 前端按钮、Gate、工作站动作的统一入口。

- `/api/tasks`
  - `web/research-agent-workstation/src/app/api/tasks/route.ts`
  - 任务列表。

- `/api/tasks/[taskId]/runs`
- `/api/tasks/[taskId]/gates`
- `/api/tasks/[taskId]/evidence`
- `/api/tasks/[taskId]/workflow`
- `/api/tasks/[taskId]/report`
- `/api/tasks/[taskId]/figures`
  - 任务级只读闭环接口。

- `/api/tasks/[taskId]/code-agent-draft`
- `/api/tasks/[taskId]/export-code-agent-context`
- `/api/tasks/[taskId]/import-agent-patch`
  - Code Agent 上下文导出、草稿生成、patch 导入。

- `/api/tasks/[taskId]/run-local-experiment`
- `/api/tasks/[taskId]/run-mcgs-experiment`
- `/api/tasks/[taskId]/run-ensemble-experiment`
  - 工作站发起的实验入口。注意：训练必须通过工作站 gate，不允许 Codex 手工旁路。

- `/api/gpu/connections/test`
- `/api/gpu/jobs`
- `/api/gpu/jobs/[jobId]`
- `/api/gpu/jobs/[jobId]/cancel`
  - GPU/HPC resource mode 与 job lifecycle。

- `/api/llm/deepseek/smoke`
  - DeepSeek 连接 smoke。

- `/api/settings`
  - 语言、主题、系统设置。

## 5. 后端服务核心文件

### 5.1 工作站 Action 总入口

- `web/research-agent-workstation/src/lib/server/workstation-actions.ts`

职责：

- 处理 `create_task`
- 处理 `onboard_playground_s6e6`
- 处理 `review_agent_patch`
- 处理 `apply_agent_patch`
- 处理 `run_s6e6_*`
- 处理 `approve_gate`
- 处理 `settings_*`
- 处理 UI state action log

注意：

- 该文件是前后端打通的关键，不要大改业务语义。
- Patch apply 必须先过 Code Quality Gate。
- Kaggle 官方提交必须 Human Gate。

### 5.2 工作站 run contract

- `web/research-agent-workstation/src/lib/server/workstation-run-contract.ts`

职责：

- 创建 workstation run
- 创建 HPC execution gate
- 生成 teacher evidence bundle
- 固化 run/gate/artifact 契约

### 5.3 闭环与 S6E6 专项

- `web/research-agent-workstation/src/lib/server/workstation-closed-loop.ts`

职责：

- S6E6 工作站闭环
- score recovery frontier
- submission gate
- Kaggle submit gate wrapper

### 5.4 GPU/HPC 网关

- `web/research-agent-workstation/src/lib/server/gpu-ssh-gateway.ts`
- `web/research-agent-workstation/src/lib/server/job-registry.ts`

职责：

- GPU SSH 连接测试
- 白名单 job manifest
- job 提交、状态、取消、日志回收

约束：

- 不允许把脚本和数据散放到远程主目录。
- 远程推荐目录：`~/research_agent_workstation/`。
- 不使用本地 GPU。

### 5.5 DeepSeek / Code Agent

- `web/research-agent-workstation/src/lib/server/deepseek-provider.ts`
- `web/research-agent-workstation/src/lib/server/deepseek-cache.ts`
- `web/research-agent-workstation/src/app/api/llm/deepseek/smoke/route.ts`

职责：

- DeepSeek 模型调用
- 缓存 key / stable prompt prefix
- 重复 prompt 缓存命中率目标：>=80%

### 5.6 Summary 与序列化

- `web/research-agent-workstation/src/lib/server/summary.ts`
- `web/research-agent-workstation/src/lib/server/serializers.ts`
- `web/research-agent-workstation/src/lib/server/paths.ts`
- `web/research-agent-workstation/src/lib/server/json.ts`

职责：

- 工作站状态汇总
- artifact path 安全解析
- JSON 读写
- Prisma 数据转前端结构

## 6. 四层 Research OS Python 代码

目录：

- `src/research_os`

关键文件：

- `src/research_os/search_graph.py`
  - ExperimentNode
  - SearchGraph
  - add_node / add_edge / top candidates / stagnation

- `src/research_os/retrospective_memory.py`
  - MemoryRecord
  - 成功策略与失败模式检索

- `src/research_os/validation_contract.py`
  - ValidationContract
  - required artifacts
  - acceptance criteria

- `src/research_os/claim_audit.py`
  - ClaimAudit
  - claim drift 分类
  - allow / revise / reject

- `src/research_os/benchmark_manager.py`
  - BenchmarkTask
  - BenchmarkResult
  - valid submission rate
  - medal rate
  - gap report

- `src/research_os/mlevolve_controller.py`
  - rank gate
  - top30 target
  - SearchControllerDecision
  - Base / Stepwise / Diff mode
  - benchmark claim gate

- `src/research_os/mlevolve_adapter.py`
  - MLEvolve 思想与工作站 contract 的适配层。

## 7. 传统/服务端 Research Agent Workstation 代码

目录：

- `src/research_agent_workstation`

关键文件：

- `src/research_agent_workstation/cli.py`
- `src/research_agent_workstation/dashboard.py`
- `src/research_agent_workstation/tabular_pipeline.py`
- `src/research_agent_workstation/titanic_pipeline.py`
- `src/research_agent_workstation/porto_seguro_safe_driver_prediction_pipeline.py`
- `src/research_agent_workstation/store_sales_time_series_forecasting_pipeline.py`

服务端模块：

- `src/research_agent_workstation/server/core/agent_runtime.py`
- `src/research_agent_workstation/server/core/artifact_registry.py`
- `src/research_agent_workstation/server/core/evidence_graph.py`
- `src/research_agent_workstation/server/core/experiment_graph.py`
- `src/research_agent_workstation/server/core/gate_engine.py`
- `src/research_agent_workstation/server/core/memory_store.py`
- `src/research_agent_workstation/server/core/run_context.py`
- `src/research_agent_workstation/server/core/task_state_machine.py`

策略层：

- `src/research_agent_workstation/server/strategy/mlevolve_search.py`
- `src/research_agent_workstation/server/strategy/mlevolve_harness_bridge.py`
- `src/research_agent_workstation/server/strategy/retrospective_memory.py`
- `src/research_agent_workstation/server/strategy/harness_optimizer.py`
- `src/research_agent_workstation/server/strategy/strategy_registry.py`

训练与 pipeline：

- `src/research_agent_workstation/server/training/job_manifest.py`
- `src/research_agent_workstation/server/training/ensemble_templates.py`
- `src/research_agent_workstation/server/pipelines/tabular_baseline.py`
- `src/research_agent_workstation/server/pipelines/submission_checker.py`
- `src/research_agent_workstation/server/pipelines/eda_generator.py`

## 8. Schema / Prompt / Benchmark 配置

Schema：

- `configs/schemas/experiment_node.schema.json`
- `configs/schemas/search_graph.schema.json`
- `configs/schemas/retrospective_memory.schema.json`
- `configs/schemas/validation_contract.schema.json`
- `configs/schemas/claim_audit.schema.json`
- `configs/schemas/benchmark_task.schema.json`
- `configs/schemas/benchmark_result.schema.json`
- `configs/schemas/rank_promotion_gate.schema.json`
- `configs/schemas/benchmark_claim_gate.schema.json`

Agent prompt：

- `prompts/agents/search_controller_mlevolve_style.md`
- `prompts/agents/retrospective_memory_agent.md`
- `prompts/agents/validation_contract_agent_xcientist_style.md`
- `prompts/agents/claim_audit_agent.md`
- `prompts/agents/report_agent_research_harness.md`

Benchmark：

- `benchmark/mle_bench_75/tasks_template.json`
- `benchmark/kaggle_10_self_evolution/tasks_20260623.json`

## 9. 关键脚本索引

### 9.1 上线与只读验证

- `scripts/verify_workstation_launch_readiness.py`
- `scripts/verify_workstation_task_api_matrix.py`
- `scripts/verify_workstation_frontend_api_contract.py`
- `scripts/verify_workstation_ui_component_wiring.py`
- `scripts/verify_no_plaintext_secrets.py`
- `scripts/verify_verified_workstation_launch_audit.py`

### 9.2 训练进度与榜单统计

- `scripts/build_workstation_training_progress_report.py`
- `scripts/build_kaggle_experiment_inventory.py`
- `scripts/build_mlebench_style_leaderboard_report.py`
- `scripts/record_kaggle_submission_score.py`

### 9.3 DeepSeek / Kaggle / GPU readiness

- `scripts/verify_deepseek_provider.py`
- `scripts/verify_deepseek_cache_policy.py`
- `scripts/verify_deepseek_cache_hit_rate_target.py`
- `scripts/verify_kaggle_dpapi_readiness.py`
- `scripts/verify_kaggle_new_competition_readiness.py`
- `scripts/verify_gpu_job_templates.py`
- `scripts/start_verified_workstation.ps1`

### 9.4 Kaggle / MLE-Bench 执行相关

- `scripts/run_workstation_orchestrator.py`
- `scripts/run_workstation_mcgs.py`
- `scripts/run_workstation_mcgs_gpu.py`
- `scripts/run_workstation_ensemble.py`
- `scripts/execute_top30_workstation_orders.py`
- `scripts/create_mlevolve_next_orders_from_inventory.py`
- `scripts/create_kaggle_10_next_search_orders.py`
- `scripts/create_kaggle_10_workstation_supervision_contracts.py`
- `scripts/mlebench_autonomous_orchestrator.py`
- `scripts/mlebench_catboost_trainer.py`
- `scripts/mlebench_catboost_trainer_v2.py`
- `scripts/mlebench_catboost_trainer_v3.py`

注意：这些执行脚本必须通过工作站 gate 使用；不要让外部 Agent 绕过工作站直接训练或提交。

## 10. 当前权威报告

- `reports/WORKSTATION_LAUNCH_READINESS_20260630.md`
- `reports/WORKSTATION_TASK_API_MATRIX_20260630.md`
- `reports/WORKSTATION_FRONTEND_API_CONTRACT_20260630.md`
- `reports/WORKSTATION_TRAINING_PROGRESS_20260630.md`
- `workspace/workstation_launch_readiness_20260630.json`
- `workspace/workstation_task_api_matrix_20260630.json`
- `workspace/workstation_frontend_api_contract_20260630.json`
- `workspace/workstation_training_progress_20260630.json`
- `workspace/kaggle_experiment_inventory_20260624.json`
- `workspace/mlebench_style_current_leaderboard_20260625.json`

## 11. 正确工作方式

后续 Agent 接手时必须遵守：

1. 先跑 readiness 和 task API matrix，确认系统状态。
2. 先读训练进度报告，恢复任务记忆。
3. 不直接训练，不直接提交 Kaggle。
4. 训练只能通过工作站 AgentOrchestrator / API / gate 发起。
5. GPU/HPC 只能作为工作站 resource mode 使用。
6. 官方 Kaggle submit 必须 Human Gate。
7. 所有实验必须生成 run id、agent trace、metrics、OOF/submission、artifact manifest、validation contract、claim audit。
8. 没有官方 Kaggle response，不能声明官方排名、奖牌或 MLE-Bench 达标。
9. Figma 只有在 OAuth 恢复并读到真实 node metadata/screenshot 后，才能声明高保真核验完成。
