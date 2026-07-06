# Research OS Demo

这个目录提供 AI 科研工作站三层融合框架的最小示例，不会调用 Kaggle、HPC、GPU 或外部 LLM API。

示例任务是一个 tabular Kaggle classification demo，包含：

- EXP000 PyTorch MLP baseline；
- EXP001 Logistic Regression；
- EXP002 ExtraTrees；
- EXP003 LightGBM；
- EXP004 XGBoost；
- EXP005 LightGBM + XGBoost OOF blend。

示例文件：

- `sample_experiment_nodes.json`：实验节点；
- `sample_search_graph.json`：搜索图摘要；
- `sample_validation_contract.json`：XCIENTIST-style validation contract；
- `sample_claim_audit.json`：claim audit 示例。

运行 demo：

```powershell
python scripts/demo_research_os.py
```

预期行为：

- 读取 sample experiment nodes；
- 构建 SearchGraph；
- 输出 top candidates；
- 创建 validation contract；
- 检查 required artifacts；
- 执行 claim audit；
- 打印审计结果。
