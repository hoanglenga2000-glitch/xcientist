# 科研 Agent 工作站最终完成审计

## 审计范围

本次收口以最开始的“科研 Agent 工作站详细建设计划”为验收主线，并结合服务器公开模板 `https://ai.zhjjq.tech/agents` 的 Agent/工作流结构做本地模板对齐。

实施边界：

- 本轮全部在本地 `D:\桌面\codex\科研港科技` 完成。
- 只读取服务器公开页面和公开 Agent 市场模板信息。
- 没有读取、同步或修改服务器私有 `/api/agents`、`/api/workflows`。
- 没有接入 GPU 服务器，避免影响服务器现有 AI 工作站运行。

## 服务器模板对齐

已新增本地模板映射：`configs/agent_templates.yaml`。

映射关系：

- `多 Agent 调度主管` -> `Orchestrator/Planner`：拆任务、定验收、管风险。
- `高级数据分析 Agent` -> `Analyst`：EDA、异常、趋势、验证计划。
- `代码开发 Agent` -> `Developer`：建模代码、测试点、实现风险。
- `知识库 RAG Agent` -> `Evidence/Summarizer`：依据引用、报告可追溯。
- `流程自动化 Agent` -> `Reviewer/Gate`：阶段检查、异常处理、审批点。

这个映射已进入 House Prices 的 `task_scaffold.json` 和实验日志，用于证明本地科研工作站不是随意命名的脚本，而是按实际 AI 工作站模板拆分角色和阶段。

## 联网一手参考依据

本地实现已经把老师最初材料中的关键一手来源写入 `configs/research_sources.yaml`，并通过 `scripts/verify_research_sources.py` 做可达性和本地映射检查。

- AutoKaggle：对应阶段化数据科学流程、代码执行、调试、单元测试和 validation gate。
- Agent K：对应 scaffold、post-scaffold、任务迁移和长期经验沉淀。
- AutoResearch AI：对应 evidence preservation、reproducibility、provenance、human oversight 和科研完整性 gate。
- AutoSOTA：对应论文复现、环境记录、指标有效性检查、反思改进和受控模型扩展。
- NanoResearch：对应文献调研、实验方案、代码执行、结果分析、图表/报告生成和 manifest 式产物追踪。
- Kaggle CLI：对应官方数据下载/提交能力，以及当前因缺少 token 而保持关闭的受控边界。

## 原计划逐项验收

| 计划项 | 当前状态 | 证据 |
| --- | --- | --- |
| v1 表格 Kaggle 闭环 | 已完成 | Titanic 与 House Prices 均生成 submission、日志、报告和 gate |
| AutoKaggle 式阶段化流程 | 已完成 | `workflow_stage_audit.json` / `workflow_stage_audit.md` |
| Agent K 式 scaffold | 已完成 | `task_scaffold.json` / `task_scaffold.md` |
| post-scaffold 改进记录 | 已完成 | House Prices 已生成 `post_scaffold_improvement.json` / `.md` |
| AutoSOTA 式论文复现路线 | 受控规划完成 | 当前只纳入路线图和界面呈现，不伪装成已完成复现实验 |
| NanoResearch 式科研流程 | 受控规划完成 | 当前已在 dashboard 呈现文献调研、方案、执行、结果、报告和人工审核阶段 |
| submission 格式检查 | 已完成 | Titanic 与 House Prices validation gate 均 passed |
| 实验日志与证据保存 | 已完成 | `experiment_log.json`、`data_quality.json`、`model_results.json` |
| Word/Markdown 报告 | 已完成 | Titanic 与 House Prices 均生成报告 |
| 第二任务迁移验证 | 已完成 | House Prices 使用通用 `tabular_pipeline` 跑通 |
| 服务器 AI 工作站保护 | 已完成 | 本轮未读取或修改服务器私有 Agent/工作流 |
| Kaggle 官方提交 | 未做，边界明确 | 本机无 `kaggle.json` 且 Kaggle CLI 未配置 |
| GPU 接入 | 未做，边界明确 | 本阶段故意不接入，待本地双任务闭环稳定后再规划 |

## 最新通过实验

### Titanic

实验目录：`experiments\titanic\20260606_192118`

验收结果：

- validation gate：`passed`
- 最佳模型：`gradient_boosting`
- 5 折 CV accuracy：`0.840619`
- Holdout accuracy：`0.815642`
- submission 行数：`418`
- submission 列名：`PassengerId, Survived`
- scaffold：已生成
- 阶段审计：已生成并通过
- 报告：Markdown 和 Word 均已生成

### House Prices

实验目录：`experiments\house_prices\20260606_192030`

验收结果：

- validation gate：`passed`
- 工作流版本：`v3_generic_tabular`
- 最佳模型：`gradient_boosting_log_target`
- 5 折 CV RMSLE：`0.12899`
- Holdout RMSLE：`0.129797`
- submission 行数：`1459`
- submission 列名：`Id, SalePrice`
- scaffold：已生成
- post-scaffold 改进记录：已生成
- 阶段审计：已生成并通过
- 报告：Markdown 和 Word 均已生成

House Prices 的本地阈值为：

- CV RMSLE `<= 0.18`
- Holdout RMSLE `<= 0.20`
- submission 行数 `1459`
- submission 列名严格匹配 `Id, SalePrice`
- 预测无缺失且全部为正数

当前结果已满足全部阈值。

## 新增核心文件

- `configs/agent_templates.yaml`
- `configs/house_prices.yaml`
- `scripts/prepare_house_prices_data.py`
- `scripts/validate_tabular_experiment.py`
- `src/research_agent_workstation/tabular_pipeline.py`
- `src/research_agent_workstation/dashboard.py`
- `web/dashboard/index.html`
- `Dockerfile`
- `docker-compose.yml`
- `scripts/verify_dashboard.py`
- `scripts/run_full_acceptance.py`
- `scripts/verify_research_sources.py`
- `scripts/verify_research_integrity.py`
- `configs/research_sources.yaml`
- `configs/long_term_roadmap.yaml`
- `docs/research_integrity_gate.json`
- `docs/科研Agent工作站最终完成审计.md`
- `docs/科研Agent工作站Docker可视化验证.md`
- `docs/可视化验收记录.md`
- `docs/visual_acceptance_desktop.png`
- `docs/visual_acceptance_mobile.png`

## 最新验收命令

```powershell
python scripts\prepare_house_prices_data.py
python -m src.research_agent_workstation.tabular_pipeline --config configs\house_prices.yaml --output-dir experiments
python scripts\validate_tabular_experiment.py --experiment-dir experiments\house_prices\20260606_192030 --config configs\house_prices.yaml

python -m src.research_agent_workstation.titanic_pipeline --config configs\titanic.yaml --output-dir experiments\titanic
python scripts\validate_titanic_experiment.py --experiment-dir experiments\titanic\20260606_192118 --config configs\titanic.yaml

python -m compileall src scripts
python scripts\verify_research_sources.py
python scripts\verify_research_integrity.py
python scripts\verify_dashboard.py
docker compose -p research-agent-workstation up -d --build
python scripts\verify_dashboard.py --url http://127.0.0.1:8088
python scripts\run_full_acceptance.py --dashboard-url http://127.0.0.1:8088
docker compose -p research-agent-workstation run --rm research-agent-workstation python scripts/run_full_acceptance.py
```

## 当前限制

1. 本机没有 Kaggle API token：
   - 未发现 `C:\Users\景浩伟\.kaggle\kaggle.json`
   - 未发现 `C:\Users\景浩伟\AppData\Roaming\kaggle\kaggle.json`
   - 未发现项目内 `kaggle.json`

2. 本机没有可用 Kaggle CLI：
   - `kaggle --version` 返回未安装状态。

3. 因此本轮没有伪造官方 leaderboard 结果：
   - Titanic 与 House Prices 都是本地 Kaggle 风格闭环。
   - 后续配置 Kaggle token 后，可把数据下载和提交切到官方 API。

4. GPU 服务器未接入：
   - 当前两个任务使用本地 sklearn baseline 已满足验收。
   - GPU 只建议用于后续深度模型、超参数搜索或批量实验。

## 人工监督与审核边界

当前系统定位为学术研究 Agent 工作站原型，不把科研判断完全交给自动化流程。

- Agent 负责任务理解、EDA、建模、submission 检查、日志和报告生成。
- 人工审核负责确认研究目标、评价指标、结果价值、是否需要官方 Kaggle 提交、是否允许服务器/GPU 接入。
- Kaggle token、服务器私有 Agent、GPU 训练和老师真实数据案例都需要用户明确授权后再进入下一阶段。
- Dashboard 中显示的 `controlled_pending` 项目不是失败项，而是需要人工确认和外部凭据的受控扩展。

## 收口结论

当前项目已经从“Titanic 专用脚本”推进到“可迁移的本地科研 Agent 工作站原型”。

它现在具备：

- 与实际 AI 工作站公开 Agent 模板对齐的角色映射。
- Kaggle 风格输入结构。
- 阶段化工作流审计。
- scaffold 与 post-scaffold 改进记录。
- 独立 validation gate。
- 双任务验证：Titanic 分类任务 + House Prices 回归任务。
- 报告、日志、指标、submission 和数据来源证据保存。
- 本地 Docker 可视化仪表盘，可直接查看任务 gate、阶段审计、模板映射和证据文件。
- 联网一手来源证据库、科研完整性 gate、长期路线图和浏览器验收记录。

下一阶段如果继续推进，优先级应是：

1. 配置 Kaggle token，增加官方下载和真实 leaderboard 记录。
2. 用老师真实数据案例复用同一套 workflow。
3. 在本地双任务闭环稳定后，再规划 GPU 服务器训练或模型搜索。
