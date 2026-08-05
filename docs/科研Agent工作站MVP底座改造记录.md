# 科研 Agent 工作站 MVP 底座改造记录

日期：2026-06-08

## 本轮目标

根据老师补充要求，系统从静态 UI 原型推进到可运行 MVP 底座。核心原则是：本系统不替代 Codex / Claude Code，而是作为科研任务、实验证据、人工审核和外部计算资源的统一工作站底座。

## 已完成

- 新增 `src/research_agent_workstation/server/` 后端底座目录。
- 新增 adapter interface：
  - `CodeAgentAdapter`
  - `PythonRunnerAdapter`
  - `LLMAdapter`
  - `GPUAdapter`
  - `KaggleAdapter`
  - `StorageAdapter`
- 新增默认本地实现：
  - `LocalTemplateCodeAgentAdapter`
  - `LocalPythonRunnerAdapter`
  - `RuleBasedLLMAdapter`
  - `LocalStorageAdapter`
  - `LocalMockGPUAdapter`
  - `DisabledKaggleAdapter`
- 新增 service 层：
  - `TaskService`
  - `ExperimentService`
  - `EvidenceService`
  - `GateService`
  - `ReportService`
  - `AgentOrchestrator`
- 新增 pipeline 包装：
  - `tabular_baseline.py`
  - `eda_generator.py`
  - `submission_checker.py`
- 新增外部 Code Agent 上下文导出：
  - Python 脚本：`scripts/export_code_agent_context.py`
  - Next API：`POST /api/tasks/{task_id}/export-code-agent-context`
- 新增 patch 导入：
  - Python 脚本：`scripts/import_agent_patch.py`
  - Next API：`POST /api/tasks/{task_id}/import-agent-patch`
- 新增前端真实 summary API：
  - `GET /api/workstation-summary`
- 前端新增 `Integration Status`，展示：
  - Code Agent: Local Template / Codex Ready / Claude Code Ready
  - Python Runner: Local
  - GPU: Not Connected
  - Kaggle: Not Configured
  - LLM: Rule-based
  - Storage: Local Workspace

## 本地闭环验证

已通过 orchestrator 真实运行 House Prices 本地闭环：

- run：`experiments/house_prices/20260608_235247`
- source_type：`local_template`
- code_agent_provider：`local_template`
- runner_provider：`local_pipeline`
- gpu_provider：`mock`
- llm_provider：`rule_based`
- best_model：`gradient_boosting_log_target`
- CV RMSLE：`0.12899`
- holdout RMSLE：`0.129797`
- validation gate：`passed`

已生成：

- `experiment_record.json`
- `evidence_manifest.json`
- `human_submission_gate.json`
- `orchestrator_run.json`
- `workstation_report.md`
- `workspace/workstation_summary.json`

## 外部 Agent 机制验证

已导出：

- `workspace/tasks/house_prices/agent_context/task_profile.json`
- `workspace/tasks/house_prices/agent_context/eda_summary.json`
- `workspace/tasks/house_prices/agent_context/metrics.json`
- `workspace/tasks/house_prices/agent_context/instructions_for_codex.md`
- `workspace/tasks/house_prices/agent_context/instructions_for_claude_code.md`

已导入测试 patch：

- `workspace/tasks/house_prices/code/patches/patch_0001.diff`
- `workspace/tasks/house_prices/code/patches/patch_0002.diff`

## 验证命令

```powershell
python -m compileall src scripts
python scripts\run_workstation_orchestrator.py --config configs\house_prices.yaml --output-base experiments --random-state 42
python scripts\validate_tabular_experiment.py --experiment-dir experiments\house_prices\20260608_235247 --config configs\house_prices.yaml
python scripts\export_code_agent_context.py --config configs\house_prices.yaml --latest-output-dir experiments\house_prices\20260608_235247
cd web\research-agent-workstation
npm run typecheck
npm run build
```

## 仍需后续完成

- 将 `Screens.tsx` 拆成 10 个页面组件，进一步降低前端维护成本。
- 将当前 Next API 的本地文件逻辑逐步迁移为正式 Python/FastAPI 或统一 backend service。
- Patch review / apply / rollback / rerun comparison 目前已保留接口和 patch 队列，下一轮应补完整 apply 流程。
- Kaggle、GPU、OpenAI/Claude API 仍保持 adapter 预留状态，未配置时不报错、不伪装已接入。
