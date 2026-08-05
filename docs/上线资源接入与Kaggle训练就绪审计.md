# 科研 Agent 工作站上线资源就绪审计

- 生成时间：2026-07-24T15:19:59
- 总体状态：ready_for_external_resources
- 本地 Kaggle 风格训练闭环：ready

## 结论

本地 Kaggle 风格数据训练、指标阈值、submission 和审计产物已经就绪；GPU 容器硬件已由 Web Terminal 证据证明；正式进入外部增强训练/自动代码优化前，还缺 Claude API Key 与 GPU 自动作业凭据。

## 外部资源状态

- code_agent: 已配置
- gpu_ssh_gateway: 未配置
  缺少：GPU_SSH_HOST, GPU_SSH_USER, GPU_REMOTE_WORKSPACE, GPU_SSH_PASSWORD or GPU_SSH_KEY_PATH
- hpc_gpu_verified_container: 已配置
- kaggle_official_api_optional: 未配置
  缺少：KAGGLE_USERNAME, KAGGLE_KEY

## Kaggle 数据训练任务

### house_prices
- 最新实验：experiments\house_prices\20260623_160809
- 数据就绪：True
- Validation Gate：True
- 本地训练就绪：True
- 指标 cv_rmsle_mean <= 0.18： 当前 0.12899，通过
- 指标 holdout_rmsle <= 0.2： 当前 0.129797，通过
- 指标 submission_rows == 1459： 当前 1459，通过

### titanic
- 最新实验：experiments\titanic\20260701_144612
- 数据就绪：True
- Validation Gate：True
- 本地训练就绪：True
- 指标 cv_accuracy_mean >= 0.78： 当前 0.8305，通过

### telco_churn
- 最新实验：experiments\telco_churn\20260623_160853
- 数据就绪：True
- Validation Gate：True
- 本地训练就绪：True
- 指标 cv_accuracy_mean >= 0.78： 当前 0.807773，通过

## 配置后直接执行

1. 配置 Claude：`ANTHROPIC_API_KEY`。
2. 配置 GPU SSH：`GPU_SSH_HOST`、`GPU_SSH_USER`、`GPU_SSH_PASSWORD` 或 `GPU_SSH_KEY_PATH`、`GPU_REMOTE_WORKSPACE`。
3. 可选配置 Kaggle 官方下载/提交：`KAGGLE_USERNAME`、`KAGGLE_KEY`。
4. 重新运行：`python scripts\verify_launch_resource_readiness.py --write-report`。
5. 浏览器打开 `http://127.0.0.1:8088`，在 Code Runner 中启动 Claude Session 或 GPU Job。

说明：Kaggle token 只影响官方 API 下载/leaderboard 提交；当前本地训练使用已验证的 Kaggle 风格输入文件，因此不把 Kaggle token 计入本轮最后两项阻塞资源。
