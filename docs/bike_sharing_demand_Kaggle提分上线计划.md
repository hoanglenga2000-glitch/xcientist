# bike_sharing_demand Kaggle 提分上线计划

- 最新实验：experiments\bike_sharing_demand\wr_2026-06-24T21-11-31.153454_19104660
- Claude Code：deepseek_code_agent
- GPU SSH：configured
- Kaggle API：not_configured

## 分阶段策略

### 1. local_baseline_lock

- 动作：Run the generated tabular baseline, validation gate and report once.
- 验收：validation_gate.json status passed；submission.csv schema matches sample_submission.csv；action log records run request and response

### 2. data_quality_and_leakage_review

- 动作：Inspect train/test feature drift, target leakage candidates, missingness and duplicate patterns.
- 验收：data_quality.json reviewed；leakage risk listed in Gate；manual gate blocks official submit until reviewed

### 3. model_ladder

- 动作：Compare linear/tree/boosting baselines, then promote the best stable model.
- 验收：cross-validation mean and std recorded；holdout score not worse than threshold；feature importance exported

### 4. seed_and_hyperparameter_sweep

- 动作：Use local CPU for small sweeps; use GPU SSH gateway for heavier search after credentials are configured.
- 验收：all sweep jobs use whitelist commands；artifacts downloaded；provenance records remote host and command template

### 5. leaderboard_gate

- 动作：Only submit through Kaggle API after KAGGLE_USERNAME/KAGGLE_KEY and human approval exist.
- 验收：token status configured；human_submission_gate approved；official submit response archived

## 上线边界

- 未配置 Kaggle 凭证时，只允许本地跑分和 submission 格式验证。
- 未配置 GPU SSH 时，只允许本地 CPU baseline 和小规模搜索。
- Claude Code 只产出建议、diff 和草稿，必须经过 Code Quality Gate 与人工 Gate。