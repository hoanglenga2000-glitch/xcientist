# Kaggle + HPC GPU 闭环验证 - 2026-06-14

## 结论

- Kaggle API：已通过 Windows DPAPI access token 完成真实认证。
- 官方比赛：`playground-series-s6e6` / Predicting Stellar Class。
- 数据下载：已通过 Kaggle API 下载并解压 `train.csv`、`test.csv`、`sample_submission.csv`。
- GPU 算力：已通过 HKUST(GZ) HPC SSH 网关在 `NVIDIA A800-SXM4-80GB` 上运行 PyTorch 训练。
- 提交文件：已生成并通过本地 schema 验证。
- 官方提交：已按用户授权完成一次真实 Kaggle 提交。
- 官方 public score：`0.95272`。
- 官方 public leaderboard 排名：`1280 / 1583`。
- 当前榜首 public score：`0.97216`，差距 `0.01944`。

## 数据证据

- `tasks/playground_series_s6e6/data/train.csv`
  - rows: 577347
  - sha256: `da2c5118c96d1958ed2cb402651665b3d622403807a8deea3b7fb0a70fe70e4a`
- `tasks/playground_series_s6e6/data/test.csv`
  - rows: 247435
  - sha256: `4689e052a2ee70d0baa0c76501e8309ccf52b260e7ad117308e4dba712e4fae0`
- `tasks/playground_series_s6e6/data/sample_submission.csv`
  - rows: 247435
  - sha256: `0a24c52ac50b686fe5d5df5381db494538edac98919c27d7ea7c17b152639e8f`

## GPU 运行结果

- 远程目录：`/hpc2ssd/JH_DATA/spooler/aimslab/research_agent_workstation/playground_series_s6e6/20260614_183531`
- 本地证据目录：`workspace/gpu/playground_series_s6e6/20260614_183531`
- Runner：`hpc_pytorch_mlp`
- Device：`cuda`
- GPU：`NVIDIA A800-SXM4-80GB`
- CUDA device count：`4`
- 训练耗时：约 `69.442` 秒
- 最佳验证 accuracy：`0.9487697622278033`
- 最佳 epoch：`16`
- 验证 log loss：`0.13884464276510716`

## Submission 验证

- 文件：`workspace/gpu/playground_series_s6e6/20260614_183531/submission.csv`
- 压缩提交文件：`workspace/gpu/playground_series_s6e6/20260614_183531/submission.zip`
- rows：`247435 / 247435`
- columns：`id,class`
- missing predictions：`0`
- invalid predictions：`0`
- id 顺序：与 sample submission 一致
- 预测分布：
  - `GALAXY`: 155209
  - `QSO`: 51589
  - `STAR`: 40637

## 官方提交与排名

- 提交记录：`workspace/kaggle_submissions/20260614_210303_hpc_submit.json`
- Kaggle submission ref：`53674602`
- 提交文件名：`submission.zip`
- 提交状态：`SubmissionStatus.COMPLETE`
- teamName：`Eizha Robinson`
- public score：`0.95272`
- public rank：`1280 / 1583`
- 榜首 team：`yuki #2`
- 榜首 public score：`0.97216`
- 与榜首差距：`0.01944`
- 完整 public leaderboard 快照：`workspace/leaderboards/playground_series_s6e6/expanded_py/playground-series-s6e6-publicleaderboard-2026-06-14T13_08_52.csv`

## 质量判断

这次结果证明了 Kaggle 数据接入、HPC GPU 训练、产物回传、submission 校验、远程官方提交、官方计分和 leaderboard 查询全链路已经打通。

当前模型是稳定 baseline，不是高分竞争模型。`0.95272` 能验证系统闭环，但距离榜首 `0.97216` 仍有明显差距，下一轮提分应优先做交叉验证、特征工程、树模型/神经网络集成、概率校准与多 seed ensemble。后续如需再次官方提交，需要单独授权。

## 安全边界

- Kaggle token 未写入仓库、报告或前端。
- SSH/GPU 密码和 Kaggle token 通过 Windows DPAPI 或临时远程环境加载。
- 远程提交完成后临时 token 文件已删除。
- Chrome 访问 Kaggle 被企业网络策略拦截，未尝试绕过；官方提交改用 HPC 远程 Kaggle API 完成。
