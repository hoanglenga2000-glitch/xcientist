# Kaggle 新比赛接入就绪报告

- task_id: `kaggle_new_competition_smoke`
- competition_slug: `kaggle-new-competition-smoke`
- generated_config: `configs/generated/kaggle_new_competition_smoke.yaml`
- task_type: `regression`
- target: `SalePrice`
- metric: `rmsle`
- official_download_status: `local_files`
- local_baseline_ready: `True`

## 能力结论

系统已能把新 Kaggle 风格表格比赛转成可运行科研任务；本地 baseline 可立即训练，官方下载/提交等待 Kaggle 凭证和人工 Gate。

## 上线说明

- 没有 Kaggle token 时，系统仍可用本地上传或镜像数据完成 baseline、submission 和审计链路。
- 有 Kaggle token 后，可切换到官方下载和人工 Gate 后提交。
- 有 GPU 后，可把同一配置提交到 SSH GPU 网关做 seed sweep、超参搜索或更重模型。