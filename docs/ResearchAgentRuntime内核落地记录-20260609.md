# Research Agent Runtime 内核落地记录

日期：2026-06-09

## 本轮已完成

本轮不新增页面，重点补系统内核，把原有“有 UI 的原型”向“有运行逻辑的科研 Agent 工作站”推进。

### 1. 统一 Agent 协议

已在 `src/research_agent_workstation/server/schemas/agent.py` 增加：

- `AgentInput`
- `AgentOutput`
- `AgentTrace`

每个 Agent 输出不再是随意字符串，而是包含 status、summary、decisions、artifacts、evidence、next_actions、risk_flags、suggested_gate。

### 2. Runtime Core

新增目录：

- `src/research_agent_workstation/server/core/agent_runtime.py`
- `src/research_agent_workstation/server/core/task_state_machine.py`
- `src/research_agent_workstation/server/core/event_bus.py`
- `src/research_agent_workstation/server/core/artifact_registry.py`
- `src/research_agent_workstation/server/core/gate_engine.py`
- `src/research_agent_workstation/server/core/experiment_graph.py`
- `src/research_agent_workstation/server/core/evidence_graph.py`
- `src/research_agent_workstation/server/core/memory_store.py`
- `src/research_agent_workstation/server/core/run_context.py`

### 3. 可运行 Agent

新增目录：

- `src/research_agent_workstation/server/agents/`

已实现最小可运行 Agent：

- `TaskReaderAgent`
- `DataAgent`
- `PlannerAgent`
- `TrainerAgent`
- `ReviewerAgent`
- `WriterAgent`
- `ReflectionAgent`

### 4. GateEngine 阻断逻辑

当前闭环中：

- `PLAN_APPROVAL` 必须 approved 后才进入代码生成和训练。
- `SUBMISSION_APPROVAL` 在 submission check 后进入 pending。
- `FINAL_CLAIM_APPROVAL` 在 report 完成后进入 pending。

Gate 决策写入：

- `gate_engine.json`
- `gate_audit_log.jsonl`

### 5. Scaffold 机制

新增 `TaskService.generate_scaffold()`。

每次任务进入计划阶段时生成：

- `workspace/tasks/{task_id}/scaffold/scaffold.json`
- `workspace/tasks/{task_id}/scaffold/scaffold.md`

### 6. 实验图谱、证据图、复盘记忆

每次 House Prices 闭环现在会生成：

- `experiment_graph.json`
- `evidence_index.json`
- `artifact_manifest.json`
- `reflection.json`
- `reflection.md`
- `memory_records.json`
- `agent_trace.jsonl`
- `event_log.jsonl`
- `runtime_snapshot.json`
- `task_state_machine.json`

### 7. 前端读取真实 Runtime

`/api/workstation-summary` 已扩展读取最新运行目录中的 runtime 文件，并返回：

- `runtime.task_state`
- `runtime.agent_trace`
- `runtime.event_log`
- `runtime.artifact_manifest`
- `runtime.evidence_graph`
- `runtime.experiment_graph`
- `runtime.reflection`
- `runtime.memory`
- `runtime.gate_engine`
- `runtime.runtime_snapshot`

前端已接入：

- Mission Control：显示最新 runtime run、metric、task state。
- Agent Runtime：显示真实 `agent_trace.jsonl`。
- Experiments：显示 `experiment_graph` 和 ReflectionAgent 下一轮建议。
- Integrity Gates：优先显示 `GateEngine` gates。

## 最新验证运行

最新闭环目录：

`experiments/house_prices/20260609_225829`

验证结果：

- `agent_trace.jsonl`：7 条
- `event_log.jsonl`：14 条
- `artifact_manifest.json`：15 个 artifact
- `evidence_index.json`：15 个 evidence item
- claims：1 条
- needs evidence：0 条
- experiment graph nodes：1
- gates：PLAN_APPROVAL approved，SUBMISSION_APPROVAL pending，FINAL_CLAIM_APPROVAL pending
- House Prices validation gate：passed
- CV RMSLE：0.12899
- submission rows：1459

## 已执行命令

```powershell
python -m compileall src scripts
python scripts\run_workstation_orchestrator.py --config configs\house_prices.yaml --output-base experiments --random-state 42
python scripts\validate_tabular_experiment.py --experiment-dir experiments\house_prices\20260609_225829 --config configs\house_prices.yaml
npm run typecheck
npm run build
powershell -ExecutionPolicy Bypass -File scripts\restart_workstation_frontend.ps1
```

## 打开与健康检查

新增脚本：

`scripts/restart_workstation_frontend.ps1`

用途：

- 停止 3090 旧进程
- 清理 `.next`
- 重启 Next dev server
- 验证首页 200
- 验证 CSS 200
- 验证 `/api/workstation-summary`
- 验证 runtime trace/event 数量

当前可打开：

`http://127.0.0.1:3090`

截图：

`docs/visual-acceptance-20260609/runtime-open-waited.png`

## 仍需继续推进

本轮已经完成核心 runtime MVP，但完整目标还未全部完成：

- Settings/provider config 还没有独立配置页。
- Workflow Graph 节点还需要完全由 TaskStateMachine 动态驱动。
- Code Runner 真实代码预览还需要从 ArtifactRegistry 读取文件内容。
- Report Studio 需要进一步读取真实 `local_report.md` 并逐段显示 claim-evidence binding。
- Titanic 也需要跑一次新版 runtime 闭环，和 House Prices 一起作为双任务验收。
- CodeAgentAdapter 还需要补完整 apply/rollback/compare before-after 能力。
