# Claude Code 后续修正与上线风险收口记录

日期：2026-06-09

## 修复背景

Claude Code 已经把系统从纯 JSON action 层推进到 Prisma + SQLite 的状态层，但仍存在一些工程风险：审计时间被 seed 逻辑反复改写、部分 API 直接返回 Prisma 原始对象、非紧凑版集成状态可能因为缺少 `env_keys` 崩溃、项目根目录依赖隐式 `cwd` 推断。

本轮目标是先恢复系统稳定性和上线安全边界，再继续推进更深层的 UI 数据迁移与任务运行控制。

## 本轮已修复

- 修复 `ensureWorkstationSeeded()` 反复污染历史 run/gate 时间的问题。
  - 现在 seed 只补缺失记录，不再每次请求改写已有 `finishedAt` / `decidedAt`。
- 修复 `connector_status.env_keys` 缺失风险。
  - `GET /api/workstation-summary` 现在返回 `DATABASE_PROVIDER=sqlite` 等环境槽位。
  - `IntegrationStatus` 在 summary 不完整时会回退到 fallback。
- 修复 action id 潜在主键冲突风险。
  - `logAction()` 的 id 增加随机尾缀，避免快速连点或并发写入时撞库。
- 增加显式项目根路径配置。
  - `src/lib/server/paths.ts` 支持 `WORKSTATION_ROOT`。
  - 新增 `web/research-agent-workstation/.env.example`。
  - 当前 `.env` 已配置本地 `WORKSTATION_ROOT`。
- 统一子资源 API 输出。
  - `/api/tasks`
  - `/api/tasks/{taskId}/runs`
  - `/api/tasks/{taskId}/gates`
  - `/api/tasks/{taskId}/evidence`
  - `/api/tasks/{taskId}/report`
  - `/api/tasks/{taskId}/workflow`
  - 这些接口不再直接暴露 Prisma camelCase 原始对象，而是输出前端和外部调用更稳定的 snake_case 契约。
- `System Action Log` 面板现在显示数据库中的最新 action 记录。
  - 页面刷新后仍能看到后端动作历史，不再只依赖前端临时 state。
- 修复 Python orchestrator 被 Claude Code 改坏后的真实训练失败。
  - `CodePlan` 是 `slots=True` dataclass，不能使用 `plan.__dict__`，已改为 `dataclasses.asdict(plan)`。
  - `CodeArtifact` 字段名是 `generated_files`，orchestrator 中错误引用 `files` 的位置已修正。
- 增加本地 job registry。
  - 训练进程启动后会记录 `processId`。
  - `Stop Run` 会尝试取消当前运行中的本地 job，并将取消请求写入 DB 和 `workspace/runtime`。

## 验证结果

已通过：

- `npx prisma validate`
- `npm run typecheck`
- `npm run build`
- `python -m compileall src scripts`
- `python scripts\validate_tabular_experiment.py --experiment-dir experiments\house_prices\20260609_123026 --config configs\house_prices.yaml`
- `python scripts\run_workstation_orchestrator.py --config configs\house_prices.yaml --output-base experiments --random-state 42`
- `POST /api/tasks/house_prices/run-local-experiment`

API smoke 已通过：

- `GET /api/workstation-summary`
- `GET /api/tasks`
- `GET /api/tasks/house_prices/runs`
- `GET /api/tasks/house_prices/gates`
- `POST /api/workstation-actions`

关键检查：

- summary 连续读取不再改写最新 run 的 `finished_at`。
- summary 已返回 `connector_status.env_keys.DATABASE_PROVIDER = sqlite`。
- runs/tasks/gates API 已返回规范化字段。
- action 同时写入数据库和 `workspace/runtime/action_log.jsonl`。
- House Prices 最新验证 gate 状态为 `passed`。
- Python orchestrator 恢复后生成并通过：
  - `experiments/house_prices/20260609_212705`
- 前端 API 训练恢复后生成并通过：
  - `experiments/house_prices/20260609_212822`

## 当前仍需继续推进

- 仍有部分页面使用 `src/data/*.ts` 静态数据，特别是：
  - Workflow Graph 节点/边
  - Experiments 对比矩阵
  - Report Studio 正文与 evidence binding
  - Integrity Gate checklist
  - Agent Runtime trace
- `Stop Run` 已具备本地进程取消能力，但仍受 Next 单进程内存 registry 限制；服务重启后无法取消重启前的进程。后续上线应改为持久化队列/独立 worker。
- 新建任务已进入 DB 和 scaffold，但任务详情页/任务工作区还需要完整动态化。
- Docker 还没有接入 Prisma 数据库初始化和健康检查。
- Kaggle/GPU 仍保持受控未完成，不能显示为已上线能力。

## 下一步建议

1. 先实现真实 job runner：
   - 记录 process id。
   - 支持取消运行。
   - 支持训练日志轮询。
2. 再按页面迁移静态数据：
   - Mission Control 已部分接 DB，可继续增强。
   - Code Runner 接 run/job/log。
   - Integrity Gates 接 DB gates/evidence。
   - Report Studio 接 DB report/evidence。
   - Workflow Graph 接 DB workflow。
3. 最后补 Docker/部署：
   - `prisma db push`
   - 数据卷
   - healthcheck
   - acceptance script
