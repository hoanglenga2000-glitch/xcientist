# 科研 Agent 工作站真实前端闭环修复记录

日期：2026-06-09

## 修复目标

根据任务规划书要求，前端不能只展示静态 UI，需要能够实际触发本地数据训练任务、保存实验结果、导出外部 Code Agent 上下文，并导入 patch 队列。

## 已修复

- 新增真实训练 API：
  - `POST /api/tasks/{task_id}/run-local-experiment`
  - 调用 `scripts/run_workstation_orchestrator.py`
  - 运行后自动调用 validation gate
  - 自动刷新 `workspace/workstation_summary.json`
- Mission Control 新增 `Local Experiment Runner` 区域。
- Code Runner 顶部 `Run Local Experiment` 按钮已接入真实本地训练闭环。
- Code Runner 新增 `Code Agent Bridge`：
  - `Export Context` 调用 `POST /api/tasks/house_prices/export-code-agent-context`
  - 生成 `workspace/tasks/house_prices/agent_context`
- `Developer Agent Patch` 的按钮从假状态改为真实导入 patch：
  - 调用 `POST /api/tasks/house_prices/import-agent-patch`
  - patch 保存到 `workspace/tasks/house_prices/code/patches`

## 浏览器验收

- 页面打开：`http://127.0.0.1:3090`
- 从 Code Runner 点击 `Run Local Experiment` 后，真实生成新实验：
  - `experiments/house_prices/20260609_002058`
  - validation gate: `passed`
- 点击 `Export Context` 后，页面显示：
  - `Context exported: workspace\tasks\house_prices\agent_context`
- 点击 `Import Patch` 后，页面显示 patch 已导入，并将状态更新为 `Applied`。
- 浏览器 console 未发现 error。

## 当前仍需继续推进

- Mission Control 的运行入口已接入同一组件，但本轮主要验收了 Code Runner 路径。
- 后续应继续把更多 UI 控件接入真实接口：
  - New Task
  - Stop Run
  - Human Plan Gate
  - Human Submission Gate
  - Report generation refresh
  - Run comparison
- 继续拆分 `Screens.tsx`，避免前端业务逻辑过度集中。
