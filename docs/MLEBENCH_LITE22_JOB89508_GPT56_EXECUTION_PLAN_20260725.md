# MLE-Bench Lite 22 — Job 89508 + GPT-5.6 Sol 执行计划

生成日期：2026-07-25（Asia/Shanghai）  
状态：**PREPARED / WAITING FOR USER APPROVAL / TRAINING NOT STARTED**

## 1. 本轮已落地的配置

- Agent 默认 LLM：`openai / gpt-5.6-sol`
- 本地兼容网关：`http://127.0.0.1:65068/v1`
- Windows 密钥存储：DPAPI；密钥值不进入源码、日志、报告或 XSCI 非敏感配置。
- XSCI 默认计算后端：`gpu`
- Job：`89508`
- 远端专属根目录：`/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra`
- 数据根目录：`/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_official_data`
- Kaggle 正式提交：保留 Human Gate，本计划不自动提交。
- 本地 GPU：不使用。

## 2. 当前真实证据

### Agent LLM

| 检查 | 结果 |
|---|---|
| 模型发现 | PASS，10 个模型，`gpt-5.6-sol` 存在 |
| 非流式 | PASS |
| 流式 | PASS，实际 Agent SSE 逐段输出 |
| Agent 实际 provider/model | `openai / gpt-5.6-sol` |
| Tool Calling | PASS，实际解析 `resource_probe(target=JOB89508)` |
| 工作站状态卡 | `OpenAI Agent Ready (gpt-5.6-sol)` |

### Job 89508

| 项目 | 实测 |
|---|---|
| SSH / CUDA smoke | PASS |
| GPU | 1 × NVIDIA A800-SXM4-80GB |
| CPU | 64 logical cores |
| PyTorch | 2.7.1+cu118，CUDA 可用 |
| 专属目录 | 存在且可写 |
| 共享盘可用空间 | 约 160 TB（总盘占用率 97%，绝对余量仍充足） |
| 当前 GPU 占用快照 | 约 4.5 GiB、48% utilization，GPU PID 在当前容器命名空间不可见 |

结论：连接与 CUDA 能力 **GO**；正式训练 **WAIT**。批准后仍要连续做 3 次 GPU 空闲探针，只有 `compute processes=0`、显存占用回落到驱动基线且利用率稳定低位才启动，避免影响其他任务。

### Lite 22 数据

- 上一份完整文件级审计：22/22 `official_prepared`，missing=0，partial=0。
- 该完整审计中官方数据树合计：prepared 362,159 文件 + raw 439,152 文件 = 801,311 文件；约 281.46 GB。
- 本轮实时快速结构审计：22/22 的 `prepared/public` 与 `prepared/private` 都存在且非空。
- 本轮重新发起的全文件遍历受共享文件系统 I/O 影响，900 秒未完成，已终止我方审计进程；没有把超时解释为数据缺失。

## 3. 为什么现在不直接运行旧训练器

`gpu_batch_trainer_v1.py` 当前注册 35 个任务，但 Lite 22 只直接注册 4 个：

1. `leaf-classification`
2. `new-york-city-taxi-fare-prediction`
3. `tabular-playground-series-dec-2021`
4. `tabular-playground-series-may-2022`

仍存在三个硬阻塞：

1. 旧路径只解析 `data/` 或 `mlebench_prepared/`，不直接解析官方 `mlebench_official_data/<id>/prepared/public/`。
2. `leaf-classification` 配置成 accuracy + class index，但官方要求 multiclass log loss + 99 列概率。
3. 其余 18 项涉及图像、文本、音频、多标签、多输出、图像复原和文本规范化，旧表格训练循环不能正确评分或生成提交。

因此：**旧脚本直接全量运行 = NO-GO**；先完成统一任务适配层后再跑分。

## 4. 批准后的实施波次

### Phase A — 适配层与评分合同（不做长训练）

1. 新建统一 `CompetitionSpec`：数据路径、模态、target、metric、direction、submission schema、split strategy。
2. 数据解析只读官方 `prepared/public`；结果、cache、checkpoint 只写专属根目录。
3. 实现 metric registry：AUC、mean-column AUC、log loss、multiclass log loss、accuracy、QWK、RMSE、mean RMSLE、token exact accuracy。
4. 实现 submission validator：行数、ID、列顺序、概率范围、每行概率和、NaN/Inf、重复 ID。
5. 接入官方 MLE private grader；评分产物必须记录 grader 版本、数据 commit、代码 hash、seed 与运行预算。

验收：22/22 spec 可解析；22/22 submission schema 有静态验证；不启动长训练。

### Wave 0 — 四个代表任务冒烟

| 代表任务 | 模态 | 目的 |
|---|---|---|
| `tabular-playground-series-may-2022` | 表格 | 路径、AUC、GBDT、submission、grader |
| `spooky-author-identification` | 文本 | TF-IDF/线性 baseline、多类概率、log loss |
| `aerial-cactus-identification` | 图像 | 图像解包、DataLoader、GPU、AUC |
| `siim-isic-melanoma-classification` | 超大图像+表格 | 只做 I/O/metadata/小批推理预检，不先做全量训练 |

每项上限：1 seed、2 folds 或固定 holdout、短 epoch/小样本；验证 GPU、日志、断点、submission 和 grader。任何一项失败即停在该适配族修复，不扩散到 22 项。

### Wave 1 — 低成本单 seed baseline

优先覆盖表格、轻量文本、小图像、图像复原：每项至少产生合法 submission、CV/holdout、离线 grader 结果或明确失败原因。

### Wave 2 — 重任务与专用模态

Aptos、Dog Breed、Histopathologic、Jigsaw、MLSP Birds、NOMAD、Plant Pathology、RANZCR、文本规范化、Whale 等按适配族逐项执行；SIIM 只有 I/O 预检通过并确认预算后进入长训练。

### Wave 3 — 稳定项多 seed / 增强

仅对 Wave 1/2 稳定通过的任务运行 seeds `42 / 43 / 44`；再尝试融合、校准、伪标签或多模态增强。失败项不隐藏，保留 error taxonomy。

## 5. 评分与排行榜输出

每场比赛一行，严格分离四类值：

| 字段 | 含义 |
|---|---|
| `cv_score` | 本地 OOF/holdout；含 mean/std、fold、seed |
| `proxy_score` | 统一归一化代理分，只用于内部横向比较 |
| `mle_private_grader_score` | 官方 prepared/private 离线 grader 输出 |
| `kaggle_public/private_score` | 只有人工批准后真实提交返回才填写 |
| `medal_or_percentile` | 只有相同 split、预算和官方阈值证据时填写 |

总榜同时输出：valid submission rate、22 项成功率、各模态成功率、平均归一化分、runtime、peak VRAM、失败类型、复现率。CV/proxy 不冒充官方 Kaggle 排名或奖牌。

## 6. 训练启动门禁

批准后仍必须全部为绿：

1. Agent 实际 SSE 返回 `provider=openai, model=gpt-5.6-sol`。
2. Tool Calling 实测通过。
3. SSH、CUDA、PyTorch、专属目录通过。
4. GPU 连续三次空闲探针通过；发现不可见 PID/显存占用则等待。
5. 22/22 数据结构通过；目标任务输入 schema 通过。
6. 对应 adapter + metric + submission validator 单测通过。
7. 远端所有输出限定在专属根目录。
8. Kaggle submission gate 保持关闭，除非单独人工批准。

## 7. 用户批准后的第一条实际动作

先执行 **Phase A + Wave 0**，不直接启动 22 项长批处理。Wave 0 完成后回报四项的 CV、离线 grader、运行时、峰值显存、submission 验证和失败明细，再由用户决定是否进入 Wave 1。

## 8. 证据文件

- `D:\桌面\codex\科研港科技\workspace\llm\openai_gateway_smoke_current.json`
- `D:\桌面\codex\科研港科技\workspace\llm\assistant_stream_probe_current.sse`
- `D:\桌面\codex\科研港科技\workspace\llm\agent_tool_call_probe_current.json`
- `D:\桌面\codex\科研港科技\workspace\hpc\job89508_resource_probe_current.json`
- `D:\桌面\codex\科研港科技\workspace\hpc\job89508_api_connection_test_current.json`
- `D:\桌面\codex\科研港科技\workspace\mlebench_lite_inventory_job89508_previous.json`
- `D:\桌面\codex\科研港科技\workspace\mlebench_lite_fast_structure_job89508.json`
- `D:\桌面\codex\科研港科技\workspace\mlebench_lite22_trainer_coverage_job89508.json`
- `D:\桌面\codex\科研港科技\docs\verified_workstation_launch_audit.json`
