# Claude Code 接手科研 Agent 工作站任务书

日期：2026-06-09

## 一、接手目标

当前系统已经不是完全静态页面，部分按钮已经能调用本地 Next API，并且可以从前端触发 House Prices 本地训练。但是它仍然没有达到“完整科研 Agent 工作站”的上线标准。

这次交给 Claude Code 的任务，不是继续零散修按钮，而是从完整系统框架上推进：

- 做清楚前后端隔离。
- 建立数据库状态层。
- 把静态 UI 数据逐步换成 API 数据。
- 把所有可操作按钮接到真实接口。
- 保留实验产物、审计记录、人工 Gate、报告导出。
- 让系统能本地稳定运行，并为后续 Docker/服务器上线做准备。

Claude Code 第一步必须先阅读：

- `D:\桌面\codex\科研港科技\docs\claude-code-research-agent-workstation-skill\SKILL.md`

## 二、当前项目位置

项目根目录：

- `D:\桌面\codex\科研港科技`

前端 Next.js 项目：

- `D:\桌面\codex\科研港科技\web\research-agent-workstation`

前端核心文件：

- `web\research-agent-workstation\src\app\page.tsx`
- `web\research-agent-workstation\src\components\workstation\Screens.tsx`
- `web\research-agent-workstation\src\components\workstation\AppShell.tsx`
- `web\research-agent-workstation\src\components\workstation\Sidebar.tsx`
- `web\research-agent-workstation\src\components\workstation\Common.tsx`
- `web\research-agent-workstation\src\data\*.ts`

当前 Next API：

- `web\research-agent-workstation\src\app\api\workstation-summary\route.ts`
- `web\research-agent-workstation\src\app\api\workstation-actions\route.ts`
- `web\research-agent-workstation\src\app\api\tasks\[taskId]\run-local-experiment\route.ts`
- `web\research-agent-workstation\src\app\api\tasks\[taskId]\export-code-agent-context\route.ts`
- `web\research-agent-workstation\src\app\api\tasks\[taskId]\import-agent-patch\route.ts`

Python 后端能力雏形：

- `src\research_agent_workstation\server\services\*.py`
- `src\research_agent_workstation\server\schemas\*.py`
- `src\research_agent_workstation\server\adapters\*.py`
- `src\research_agent_workstation\tabular_pipeline.py`

配置和脚本：

- `configs\house_prices.yaml`
- `configs\titanic.yaml`
- `scripts\run_workstation_orchestrator.py`
- `scripts\validate_tabular_experiment.py`
- `scripts\validate_titanic_experiment.py`
- `scripts\run_full_acceptance.py`

运行和审计产物：

- `workspace\workstation_summary.json`
- `workspace\runtime\action_log.jsonl`
- `workspace\tasks\house_prices\agent_context`
- `workspace\tasks\house_prices\code\patches`
- `workspace\gates`
- `workspace\exports`
- `workspace\evidence`
- `workspace\workflows`

最新已验证实验：

- `experiments\house_prices\20260609_111512`
- `experiments\house_prices\20260609_111512\validation_gate.json`

## 三、当前已经完成的内容

1. 前端已搭建为 Research Agent Workstation 结构，包括：

- Mission Control
- Research Tasks
- Workflow Graph
- Code Runner
- Agent Runtime
- Experiments
- Report Studio
- Integrity Gates
- Design System
- Overview Board

2. 已经接入部分真实按钮：

- `Run House Prices`
- `Run Local Experiment`
- `Export Context`
- `Import Patch`
- `Dry Run`
- `Save`
- `Publish`
- `Approve`
- `Reject`
- `Export Report`
- `Submit for Review`
- `Export All`
- `Open Folder`
- `View full log`
- `Edit Claim`
- `Add Evidence`
- `View Reproducibility`
- 顶栏、侧栏、搜索框、Design System 样例按钮

3. 已从前端实际触发过 House Prices 训练。

最新结果：

- `status`: `passed`
- `best_model`: `gradient_boosting_log_target`
- `cv_rmsle_mean`: `0.12899`
- `holdout_rmsle`: `0.129797`
- `submission_rows`: `1459`
- `submission_columns`: `Id, SalePrice`
- `all_stages_passed`: `true`

4. 当前验收记录：

- `docs\科研Agent工作站前端后端闭环验收记录-20260609.md`
- `docs\browser-acceptance-20260609-workstation.png`

## 四、当前主要问题

请 Claude Code 注意：当前系统只是“局部接通”，还不是完整工程架构。

主要问题如下：

1. 前后端没有真正隔离。

当前很多操作是 `page.tsx` 或组件里直接 `fetch` 到 Next API，Next API 再写本地 JSON 文件。缺少统一 API client，缺少稳定业务服务层。

2. 没有数据库。

任务、实验、Gate、Evidence、Report、Workflow、ActionLog 都主要靠文件记录。文件可以保留作为审计产物，但系统状态应该进入数据库。

3. `Screens.tsx` 过大。

这个文件承载了太多页面、状态和业务动作，后续维护困难。应该逐步拆分，但不要一开始大规模重写 UI。

4. 静态数据仍然很多。

`src\data\*.ts` 里还有大量任务、报告、实验、Agent、Gate、Workflow 的假数据。下一步应该逐页迁移成 API + DB 数据。

5. “Stop Run” 还不是真取消。

当前只是写入停止请求 JSON，还没有真实取消运行中的子进程或任务。

6. Workflow Graph 仍偏展示。

节点和边还没有真正持久化为工作流定义，Dry Run/Save/Publish 只是写 JSON，不是完整工作流引擎。

7. 新建任务没有动态进入任务列表。

`New Task` 已生成 scaffold，但左侧任务队列没有从后端刷新。

8. Kaggle/GPU 仍是受控未完成。

本机没有 Kaggle token，也没有接入 GPU。不能伪装成已经完成，只能在 UI 中显示 Reserved / Not Configured。

## 五、Claude Code 下一步建议路线

### 阶段 1：建立稳定全栈底座

优先做这个，不要先重画 UI。

建议方案：

- 本地 MVP 使用 Prisma + SQLite。
- 未来上线可迁移 PostgreSQL。
- Next API 暂时作为后端边界。
- Python 训练脚本继续作为本地 runner 被调用。

必须新增：

- 数据库 schema
- 初始化/迁移命令
- API client 层
- 服务函数层
- 数据库-backed action log

推荐新增文件：

- `web\research-agent-workstation\prisma\schema.prisma`
- `web\research-agent-workstation\src\lib\db.ts`
- `web\research-agent-workstation\src\lib\api\client.ts`
- `web\research-agent-workstation\src\lib\api\types.ts`
- `web\research-agent-workstation\src\lib\server\actions.ts`
- `web\research-agent-workstation\src\lib\server\tasks.ts`
- `web\research-agent-workstation\src\lib\server\runs.ts`
- `web\research-agent-workstation\src\lib\server\gates.ts`
- `web\research-agent-workstation\src\lib\server\evidence.ts`
- `web\research-agent-workstation\src\lib\server\reports.ts`
- `web\research-agent-workstation\src\lib\server\workflows.ts`

### 阶段 2：先做一个完整垂直切片

不要一口气改完整系统。先把一个任务完整跑通：

任务：`house_prices`

垂直切片包括：

- 任务列表从 DB 读取。
- 点击任务后读取最新 run。
- 点击 Run Local Experiment 创建 DB run 记录。
- 训练完成后更新 run 状态和 metrics。
- validation gate 写入 DB。
- 前端刷新 run 状态。
- action log 同时写 DB 和 `workspace\runtime\action_log.jsonl`。
- 报告导出和 Gate 审批写 DB。

这个切片完成后，再迁移 Titanic 和其他页面。

### 阶段 3：逐页迁移 UI 数据

迁移顺序建议：

1. Mission Control
2. Code Runner
3. Integrity Gates
4. Report Studio
5. Experiments
6. Workflow Graph
7. Agent Runtime
8. Overview Board

每迁移一页，都要保证：

- loading 状态
- error 状态
- empty 状态
- success 状态
- `data-testid`
- 浏览器点击验收

### 阶段 4：补全运行控制

重点补：

- Job 状态模型：queued / running / passed / failed / cancelled
- 进程 ID 或 job ID
- Stop Run 真实取消
- 训练日志轮询或流式展示
- 重跑、失败重试、取消后恢复

### 阶段 5：部署和 Docker 验收

完成 DB 和 API 后，再做：

- Dockerfile 更新
- docker-compose 加数据库
- healthcheck
- 容器内 acceptance 命令
- 前端端口和后端端口明确
- 数据卷只读/可写边界明确

## 六、建议数据库模型

至少需要：

- `Task`
- `ExperimentRun`
- `ActionLog`
- `Workflow`
- `Gate`
- `Evidence`
- `Report`
- `ConnectorStatus`

字段建议见：

- `docs\claude-code-research-agent-workstation-skill\SKILL.md`

## 七、验收标准

Claude Code 每完成一个阶段，必须运行：

```powershell
cd D:\桌面\codex\科研港科技\web\research-agent-workstation
npm run typecheck
npm run build
```

Python 验证：

```powershell
cd D:\桌面\codex\科研港科技
python -m compileall src scripts
python scripts\validate_tabular_experiment.py --experiment-dir experiments\house_prices\20260609_111512 --config configs\house_prices.yaml
```

浏览器验收：

- 打开 `http://127.0.0.1:8088`
- 逐页点击主要按钮。
- 确认数据库有记录。
- 确认 `workspace\runtime\action_log.jsonl` 有记录。
- 确认实验产物没有丢失。

## 八、给 Claude Code 的直接启动提示词

可以直接复制下面这段给 Claude Code：

```text
你现在接手 D:\桌面\codex\科研港科技 的 Research Agent Workstation。

请先读取：
docs\claude-code-research-agent-workstation-skill\SKILL.md
docs\ClaudeCode接手科研Agent工作站任务书.md
docs\科研Agent工作站前端后端闭环验收记录-20260609.md

当前系统已有部分 Next API 和前端按钮接线，但还不是完整工程。你的任务是从系统框架上继续推进：前后端隔离、数据库接入、API 契约、真实任务状态、真实 Gate/Evidence/Report/Workflow 持久化、全按钮浏览器验收、Docker/上线准备。

不要先重画 UI。保持当前视觉方向。先做一个完整垂直切片：house_prices 任务从 DB 读取、点击运行创建 ExperimentRun、训练完成更新 metrics 和 validation gate、ActionLog 写 DB 和 JSONL、Report/Gate/Evidence 可查询、前端自动刷新。

建议优先使用 Prisma + SQLite 作为本地 MVP 数据库，未来可迁移 PostgreSQL。如果你认为 FastAPI + SQLAlchemy 更适合，请先说明取舍并保持边界清晰。

完成后必须运行：
npm run typecheck
npm run build
python -m compileall src scripts
并用浏览器打开 http://127.0.0.1:8088 点击验收主要按钮。

注意：Kaggle token 和 GPU 当前未配置，不能伪装为已完成；只能显示 Reserved / Not Configured。
```
