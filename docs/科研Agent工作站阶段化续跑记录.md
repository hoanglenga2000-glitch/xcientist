# 科研 Agent 工作站阶段化续跑记录

## 本次续跑目标

在不连接、不改动服务器 AI 工作站的前提下，继续推进本地科研 Agent 工作站的实施工作。上一轮已经完成 Titanic 本地 Kaggle 闭环和基础 validation gate，本次重点是把一次性实验进一步固化为可复用、可审计的阶段化工作流。

## 本次新增内容

1. 将 Titanic pipeline 升级为阶段化工作流输出。
   - 每次运行都会生成 `workflow_stage_audit.json`。
   - 每次运行都会生成 `workflow_stage_audit.md`。
   - 审计覆盖任务理解、初步 EDA、数据质量检查、特征工程、模型验证、submission 生成、报告复盘七个阶段。

2. 将阶段化审计纳入独立 validation gate。
   - `scripts/validate_titanic_experiment.py` 现在会检查阶段审计文件是否存在。
   - gate 会确认配置中的 workflow 阶段都已经被审计覆盖。
   - 如果任一阶段状态不是 `passed`，gate 会失败。

3. 补齐可复现依赖。
   - `requirements.txt` 已加入 `python-docx>=1.1`。
   - 这样新环境按依赖安装后，也能稳定生成 Word 报告并通过 docx 报告检查。

## 最新通过验收的实验目录

`experiments/titanic/20260606_185148`

核心结果：

- 本地 validation gate：passed
- 最佳模型：gradient_boosting
- 5 折 CV accuracy：0.840619
- Holdout accuracy：0.815642
- submission 行数：418
- submission 列名：PassengerId, Survived
- scaffold：已生成
- 阶段化审计：已生成并通过
- Markdown/Word 报告：已生成

## 最新验收命令

```powershell
python -m src.research_agent_workstation.titanic_pipeline --config configs/titanic.yaml --output-dir experiments/titanic
python scripts\validate_titanic_experiment.py --experiment-dir experiments\titanic\20260606_185148 --config configs\titanic.yaml
python -m compileall src scripts
```

## 验收结论

当前本地工作站已经从 v1 的“跑通 Titanic Kaggle 闭环”，推进到 v2 的“阶段化 Agent 工作流审计”。这一步的价值是：后续换第二个 Kaggle 表格任务或老师真实数据案例时，不再只是复制一段训练脚本，而是可以复用同一套阶段划分、证据文件、submission 检查和 gate 验收逻辑。

## 仍未做的部分

1. 尚未配置 Kaggle API token，因此还没有官方 Kaggle 下载和 leaderboard 提交记录。
2. 尚未接入服务器 GPU，且本阶段故意不接入，避免影响服务器现有 AI 工作站运行。
3. 尚未用第二个表格任务验证迁移能力。下一步建议选择 House Prices、Bank Churn、Obesity Risk 或老师提供的真实数据案例。
4. 尚未加入 LightGBM/XGBoost/CatBoost 等更强模型。本阶段先保持 sklearn 基线稳定。

## 建议下一步

优先做第二个表格任务迁移验证。只有当同一套流程能在第二个任务上复用，工作站才算从 Titanic 专用脚本升级为真正可迁移的科研 Agent 工作流。
