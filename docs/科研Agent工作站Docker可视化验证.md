# 科研 Agent 工作站 Docker 可视化验证

## 目标

在本地 Docker 中部署一个只读可视化仪表盘，让当前科研 Agent 工作站的实施效果可以直接在浏览器中检查。

这个仪表盘不是营销页，而是面向学术研究 Agent 的验收面板，重点展示：

- Titanic 与 House Prices 双任务 validation gate。
- 阶段化工作流审计。
- scaffold 与 post-scaffold 改进记录。
- 服务器公开 Agent 模板到本地科研角色的映射。
- 实验日志、报告、submission 和最终审计证据。
- Kaggle token、GPU 服务器和私有 Agent 的边界状态。

## 启动方式

```powershell
docker compose -p research-agent-workstation up -d --build
```

浏览器打开：

```text
http://127.0.0.1:8088
```

健康检查：

```powershell
python scripts\verify_dashboard.py --url http://127.0.0.1:8088
python scripts\run_full_acceptance.py --dashboard-url http://127.0.0.1:8088
docker compose -p research-agent-workstation run --rm research-agent-workstation python scripts/run_full_acceptance.py
```

停止服务：

```powershell
docker compose -p research-agent-workstation down
```

## 本地直接运行

如果不想使用 Docker，也可以直接运行：

```powershell
python -m src.research_agent_workstation.dashboard --host 127.0.0.1 --port 8088
python scripts\verify_dashboard.py --url http://127.0.0.1:8088
```

## 可视化内容

仪表盘第一屏显示：

- 任务 Gate 通过数量。
- 阶段审计通过数量。
- 当前证据文件数量。
- 服务器私有区是否被改动。

任务区显示：

- Titanic 分类任务最新通过结果。
- House Prices 回归任务最新通过结果。
- 最佳模型、核心指标、submission 和阶段状态。

计划完成度区显示：

- 任务理解。
- EDA 与数据质量。
- 特征工程。
- 建模与验证。
- submission 检查。
- 实验记录与报告。
- scaffold 与 post-scaffold。
- 第二任务迁移。

Agent 模板映射区显示：

- `多 Agent 调度主管` -> `Orchestrator/Planner`
- `高级数据分析 Agent` -> `Analyst`
- `代码开发 Agent` -> `Developer`
- `知识库 RAG Agent` -> `Evidence/Summarizer`
- `流程自动化 Agent` -> `Reviewer/Gate`

证据区可以打开本地 Markdown、JSON、CSV 文本证据文件。

## 安全边界

- Docker 服务只读取本地项目目录。
- `docker-compose.yml` 使用只读 volume：`.:/app:ro`。
- 仪表盘不提供写入接口。
- 仪表盘不读取服务器私有 `/api/agents` 或 `/api/workflows`。
- 仪表盘不连接 GPU 服务器。
- Kaggle 官方提交仍然需要后续配置 Kaggle token。

## 验收标准

`scripts\verify_dashboard.py` 会检查：

- Titanic 和 House Prices 都在仪表盘 summary 中。
- 两个任务的 gate 都是 `passed`。
- 两个任务的阶段审计都全部通过。
- 计划完成度全部为 `passed`。
- 五个服务器公开 Agent 模板映射都存在。
- 服务器私有 Agent 未修改。
- GPU 服务器未连接。

`scripts\run_full_acceptance.py` 会额外串联：

- Titanic validation gate。
- House Prices validation gate。
- dashboard 离线 summary 验证。
- `python -m compileall src scripts`。
- Docker dashboard URL 验证。
- `/health` 可达性检查。
- 联网一手来源可达性检查。
- 科研完整性 gate 检查。

容器内完整复测命令会在 Docker 镜像中运行同一套 acceptance，确认依赖安装、只读挂载和脚本入口都可用。

## 浏览器验收记录

本项目可使用 browser-use MCP、Chrome headless 或人工浏览器检查页面。

验收项：

- 首屏非空，显示 `科研 Agent 工作站`。
- 双任务 gate 可见，包含 `titanic` 和 `house_prices`。
- Agent 模板映射可见，包含 `多 Agent 调度主管`。
- 新增 `研究依据` 和 `完整性检查` 区域可见。
- `长期路线图` 区域可见，且 Kaggle/GPU 显示为受控未启用扩展。

截图保存位置：

- `docs/visual_acceptance_desktop.png`
- `docs/visual_acceptance_mobile.png`

## 注意

当前项目路径包含中文字符时，Docker Compose 可能无法自动推导项目名。命令中显式使用 `-p research-agent-workstation` 可以避免这个问题。
