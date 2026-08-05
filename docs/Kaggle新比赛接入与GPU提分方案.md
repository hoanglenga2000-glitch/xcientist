# Kaggle 新比赛接入与 GPU 提分方案

## 当前结论

系统已经具备“新 Kaggle 风格表格比赛”的本地接入能力：给定 `train.csv`、`test.csv`、`sample_submission.csv` 后，可以自动生成科研任务配置，推断目标列、任务类型、基础指标，运行本地 baseline，生成 submission、validation gate、实验报告和证据文件。

本轮用 House Prices 数据模拟了一个新比赛：

- 任务 ID：`kaggle_new_competition_smoke`
- 配置文件：`configs/generated/kaggle_new_competition_smoke.yaml`
- 实验目录：`experiments/kaggle_new_competition_smoke/20260612_151019`
- 本地 CV RMSLE：`0.129645`
- Holdout RMSLE：`0.129119`
- Validation Gate：`passed`
- Submission：已生成，行数和列名与 `sample_submission.csv` 一致

## 新比赛来了以后怎么用

### 方式一：本地文件接入

把比赛文件放到一个目录，至少包含：

- `train.csv`
- `test.csv`
- `sample_submission.csv`

然后运行：

```bash
python scripts/onboard_kaggle_competition.py ^
  --competition-slug <kaggle-competition-slug> ^
  --task-id <local_task_id> ^
  --data-dir <data_dir> ^
  --target <target_column> ^
  --task-type regression ^
  --metric rmse
```

系统会生成：

- `configs/generated/<local_task_id>.yaml`
- `tasks/<local_task_id>/overview.txt`
- `tasks/<local_task_id>/data/DATA_SOURCE.md`
- `docs/kaggle_new_competition_readiness.json`
- `docs/Kaggle新比赛接入就绪报告.md`

### 方式二：Kaggle 官方 API 接入

配置 Kaggle 凭证和 CLI 后，可以加上 `--use-kaggle-api`：

```bash
python scripts/onboard_kaggle_competition.py ^
  --competition-slug <kaggle-competition-slug> ^
  --task-id <local_task_id> ^
  --use-kaggle-api
```

当前本机没有 Kaggle CLI 和 Kaggle token，因此系统会明确显示 `not_configured`，不会伪造官方下载或 leaderboard 提交。

## 跑分流程

生成配置后运行：

```bash
python scripts/run_workstation_orchestrator.py ^
  --config configs/generated/<local_task_id>.yaml ^
  --output-base experiments ^
  --random-state 42
```

验证：

```bash
python scripts/validate_tabular_experiment.py ^
  --experiment-dir experiments/<local_task_id>/<latest> ^
  --config configs/generated/<local_task_id>.yaml
```

## GPU 服务器接入后能提升什么

GPU 服务器不会自动保证 leaderboard 高分，但可以把提分流程系统化：

- 运行多 seed sweep，降低偶然性。
- 跑更重的模型或深度模型。
- 做超参数搜索。
- 并行训练多个候选方案。
- 保存远程日志、模型文件、指标、submission 和 provenance。

系统当前已经预留 GPU SSH 网关，后续只需要配置：

- `GPU_SSH_HOST`
- `GPU_SSH_USER`
- `GPU_SSH_KEY_PATH`
- `GPU_REMOTE_WORKSPACE`

## Claude Code 接入后能提升什么

Claude Code 可作为 Coding Agent：

- 根据失败 Gate 或低分原因生成代码 patch。
- 优化特征工程。
- 增加模型候选。
- 修复数据处理错误。
- 生成实验报告解释。

安全边界：Claude 只能生成建议、diff 和文件草稿，不能绕过人工 Gate 直接修改上线代码。

## 对老师问题的准确回答

如果 Kaggle 新出一个表格数据比赛，当前系统已经可以快速接入并跑出可复现 baseline 分数；它能保证的是“流程完整、submission 合法、本地验证可信、证据可审计、后续可持续提分”，不能承诺没有分析就直接拿 leaderboard 高分。

拿到 GPU 服务器和 Kaggle/Claude 凭证后，可以进一步把 baseline 扩展成自动迭代提分流程。
