# 自我进化引擎 — 系统全面理解 → 全功能回归 → GPU 大规模数据自进化 接手 Prompt

> 生成时间: 2026-07-03 | 项目: D:\桌面\codex\科研港科技 (AI 科研工作站 / MLE-Bench 75)
> 交接目标: 让接手会话 (1) 全面理解自我进化引擎; (2) 再次跑通系统全部功能; (3) 用引擎的自我进化能力做 GPU 大规模数据测试
> 前置状态: 引擎核心已稳定, 全套 pytest 225 passed (其中进化引擎 38), tabular/text/image 三模态已在真实数据跑通

---

## 0. 绝对规则 (始终遵守, 优先级最高)

```
1.  所有 GPU 文件只放 ~/jinghw/scripts/gpu_tra/ (含 evolution 子目录), 不散落主目录
2.  不使用本地 GPU; 重训练一律走远程 A40/A800
3.  不直接提交 Kaggle — 必须经过 Human Gate (人工确认)
4.  不把 CV/proxy 分数当官方排名; 无 Kaggle 响应 artifact 不做 rank 断言
5.  不虚构奖牌 / 不虚构 75 任务完成
6.  失败任务必须记录 (引擎会自动写 run_error.txt + 记忆库), 不许只报成功
7.  凭据只从 gpu_credentials (env / *_FILE) 解析, 绝不硬编码进任何新代码/日志/文档
8.  创建 git commit 前必须获得用户显式授权
9.  只用授权的两台算力机 (见 §4), 不碰其它服务器以免影响他人进程
10. 数据/结果是唯一事实来源: 文档是快照, 落盘的 JSON/CSV/日志 优先于任何记忆
```

## 任务三阶段总览 (本次交接的核心)

| 阶段 | 目标 | 交付判据 |
|------|------|---------|
| **A. 全面理解系统** | 读透进化引擎四层架构 + 五个核心模块, 能画出闭环数据流 | 能准确说出 loop→graph→selector→memory→generator 各自职责与接口 |
| **B. 全功能回归** | 本地跑通 CI + 全套测试, 再本地跑一次进化循环冒烟 | `run_ci_checks.py` 绿; pytest 225 passed; 本地进化 run 产出 summary.json |
| **C. GPU 大规模自进化** | 用 GPU runner 在真实大数据集上跑多轮自进化, 观察自学习/记录失败/自我改进 | 多任务批量 run 产出 batch_leaderboard + 每任务 search_graph.json, 失败被记录并反哺 |

---

## 阶段 A — 全面理解系统 (先读代码, 再动手)

### A.1 引擎是什么 (一句话)
一个**结果驱动的闭环进化引擎**: 播种基线 → 运行 → 打分 → 晋级/保留 → 记住教训 → (用历史+记忆) 提出下一个变异 → 循环。
它取代了旧的 `if n_prev==0/1/2` 硬编码阶梯, 变成真正读分数、读教训来决策的循环。runner 可插拔, 本地子进程与 GPU SSH 走同一个 `Runner` 协议, 所以三模态、本地/GPU 统一在一个循环后面。

### A.2 四层架构
```
Layer 4  Memory / Benchmark Evolution   benchmark_manager.py, retrospective_memory.py
Layer 3  Research Harness (审计)          validation_contract.py, claim_audit.py
Layer 2  MLEvolve Search Controller      mlevolve_controller.py, search_graph.py, mcgs_selector.py
Layer 1  Multi-Agent Research OS         Web 前端 + API + GPU Gateway + Agent Runtime
```

### A.3 五个核心模块 (src/research_os/, 必须读懂)
| 文件 | 职责 | 关键接口/不变量 |
|------|------|----------------|
| `evolution_loop.py` | 闭环主体 + 本地 runner + 失败分类 + 记忆写入 | `EvolutionLoop.run()`; `_classify_failure` (具体→通用桶); `_salient_error` (取异常行不取盲尾); 瞬时网络错误重跑同一份代码 |
| `search_graph.py` | 实验图 + 晋级门禁 | `decide_promotion()`: 晋级需 **run_success + 超过 min_delta + 产物齐全** 三条件; **崩溃父节点的 score 视为 absent** (防幽灵分数锚定基线) |
| `mcgs_selector.py` | MCGS 选择大脑 (opt-in) | UCT 遍历选节点; 四种扩展 primary/intra_branch/cross_branch/aggregation; C 从 1.414 衰减到 0.5; `backpropagate` 自增 visit (旧引擎这里坏了) |
| `variation_generator.py` | 变异提案 (调 LLM) | `propose()` 把 CV 历史+记忆教训+上次错误塞进提示; 按 modality 切库 (tabular/text 走 CPU 树/线性, image/multimodal/audio 走 GPU DL) |
| `retrospective_memory.py` | JSON 记忆库 | `add_memory / retrieve_by_task_type / retrieve_failures / retrieve_successes`; 记录 what_worked / what_failed / failure_pattern / metric_delta |

### A.4 三项能力落在哪 (阶段 C 要观察的就是这三项)
- **自学习/优化**: MCGS 大脑 + 晋级门禁 (能分辨"真提升"和"CV 噪声", min_delta 门禁)
- **主动记录失败**: 每次失败落 `run_error.txt` + `validation_contract.json` + `claim_audit.json`, 并写入全局 `retrospective_memory.json`
- **自我改进**: 失败的完整 traceback 通过 `LAST RUN FAILED. Fix THIS error:` 反馈给下一轮, 切 Diff 模式定向修复; 停滞切 cross_branch/aggregation 融合

### A.5 已知的一个存量问题 (值得先处理)
全局 `experiments/evolution/retrospective_memory.json` 里有 **`_salient_error` 修复之前**写入的旧脏记录 (what_failed 以文件路径噪声截断)。修复只对新记录生效, 存量不回填。可选: 写一次性迁移用新 `_salient_error` 重清洗存量 `what_failed` (纯本地、可逆, 先备份再写)。

---

## 入口脚本地图 (真实存在, 已核对 argparse)

| 脚本 | 用途 | 关键参数 |
|------|------|---------|
| `scripts/run_ci_checks.py` | CI 总闸: 编译 + 导入 + pytest | `--skip-tests`, `--quiet` |
| `scripts/run_evolution.py` | **单任务进化** (本地或 GPU) | `--task-config <json>` `--runner local\|gpu` `--iterations N` `--data-dir` `--remote-data-dirname` `--timeout` `--mcgs` |
| `scripts/run_evolution_batch.py` | **批量进化** + 汇总榜单 | `--config-dir configs/evolution` `--only <names...>` `--runner local\|gpu` `--iterations N` `--timeout` |
| `scripts/evolution_engine_cli.py` | JSON-in/JSON-out 适配器 (给前端/编排调用) | `--mode <mode>` `--input <json>` |

现有 6 个任务配置 `configs/evolution/` (已核对: 每个 `remote_data_dirname` 都命中真实数据目录):

| config | task / remote_data_dirname | modality | metric | 远程数据量 |
|--------|---------------------------|----------|--------|-----------|
| `nyc_taxi.json` | new-york-city-taxi-fare-prediction | tabular | rmse | **5.4GB (最大, 大数据首选)** |
| `tps_may2022.json` | tabular-playground-series-may-2022 | tabular | roc_auc | 804MB |
| `tps_dec2021.json` | tabular-playground-series-dec-2021 | tabular | accuracy | 786MB |
| `nomad2018.json` | nomad2018-predict-transparent-conductors | tabular | rmsle_columnwise_mean | 小 |
| `spooky_author.json` | spooky-author-identification | text | multiclass_log_loss | 4MB |
| `aerial_cactus.json` | aerial-cactus-identification | image | roc_auc | 3MB |

配置字段: task_name, remote_data_dirname, modality, task_type, metric, metric_direction, target_column, id_column, n_train, n_test, data_schema, extra_notes。
> `spooky_author` 在数据清单里被误归到"表格类", 但它是 **text/NLP** 任务 — 以 config 的 `modality=text` 为准。

---

## 阶段 B — 全功能回归 (本地, 无需 GPU)

按顺序执行, 每步绿了再下一步:

```powershell
cd D:\桌面\codex\科研港科技

# B1. CI 总闸: 编译所有一方 Python + 导入所有模块 + 跑 pytest
python scripts/run_ci_checks.py

# B2. 单独确认全套测试 (预期 225 passed, 进化引擎占 38)
python -m pytest tests/ -q

# B3. 只看进化引擎回归 (锁定 run_success / 晋级门禁 / 失败分类 / _salient_error)
python -m pytest tests/test_evolution_engine.py -v

# B4. 本地进化冒烟 (不碰 GPU, 用 tabular 任务; 需要本地 sklearn/lightgbm + .env 里的 LLM key)
python scripts/run_evolution.py --task-config configs/evolution/nomad2018.json --runner local --iterations 3
#   产物: experiments/evolution/<task>_local_<ts>/{summary.json, search_graph.json, best_solution.py, EXP00x/...}
#   验收: summary.json 有 best_exp_id + n_promotions; 失败的 EXP 目录里有 run_error.txt
```

回归判据: B1/B2/B3 全绿; B4 产出 summary.json 且 promotion_history 里能看到 "no parent score...promoted as first scored node" 之类的真实决策。

---

## §4. GPU 连接 (阶段 C 前置; 凭据走安全通道, 本文不含明文)

授权算力机 (只用授权机, 见规则 9): AI-X86_NVIDIA 集群。作业号会轮换 (记忆里出现过 87384 A800 / 87318 A40 / 87907 A40) — **不要写死作业号**, 以 `load_gpu_env.ps1` 解出的当前凭据 + 现场 SSH 探测为准。
代理链路: `本地 Clash(127.0.0.1:7890) → SOCKS5(8.163.52.223:1080) → SSHPiper(100.85.169.63:1235)`。
数据根 (绝对路径): `/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/` (即 `~/jinghw/scripts/gpu_tra/mlebench_raw_data`, 与 gpu_runner 默认 data_root 一致)。

**凭据加载 (不硬编码, 规则 7)**:
```powershell
# 从 Windows 凭据库解出 .env (含 GPU_SSH_* + LLM key); 之后引擎/脚本自动读 .env
powershell -ExecutionPolicy Bypass -File "D:\桌面\codex\科研港科技\scripts\load_gpu_env.ps1"
```
`src/research_os/gpu_runner.py` 通过 `research_agent_workstation.server.core.gpu_credentials.connect_ssh()` 建连, 凭据来自 env/`*_FILE`; 远程根目录 `$GPU_REMOTE_WORKSPACE` 默认 `~/jinghw/scripts/gpu_tra`, 数据在其下 `mlebench_raw_data/<task>`。GPU runner 每次 run 建一条 SSH, SFTP 上传 solution.py 到 `evolution/<task>/<exp_id>/`, 远程 `timeout` 保护执行, 回收 CV_SCORE + metrics.json + 产物清单。

**代理不通时重启桥**:
```powershell
powershell -ExecutionPolicy Bypass -File "D:\桌面\codex\科研港科技\scripts\manage_hpc_proxy_bridge.ps1" stop; sleep 3
powershell -ExecutionPolicy Bypass -File "D:\桌面\codex\科研港科技\scripts\manage_hpc_proxy_bridge.ps1" start; sleep 8
```

> 注意: 隧道偶发掉线。引擎已内建 `_is_transient_infra` 重试 (SSH/SOCKS 抖动重跑同一份代码, 不浪费 LLM 提案)。若持续掉线才需硬化重连, 属独立任务。

---

## 阶段 C — GPU 大规模数据自进化 (核心目标)

前提: 阶段 B 全绿 + §4 凭据已加载 + 代理连通。

### C.0 先验数据在位 (关键! 清单标注 43 可用 / 33 空)
数据清单 `prompts/DATA_INVENTORY_FOR_CLAUDE.md` 有空目录、未解压 zip (siim-covid19 9.1GB)、下载中 (herbarium)。**跑任何任务前先 SSH 确认该任务目录非空且有 train/test 文件**:
```bash
# 在 GPU 上列出各数据集大小, 确认目标任务真的有数据 (非空)
cd ~/jinghw/scripts/gpu_tra/mlebench_raw_data && du -sh */ | sort -h
# 单个任务确认结构 (示例 nyc_taxi)
ls -la ~/jinghw/scripts/gpu_tra/mlebench_raw_data/new-york-city-taxi-fare-prediction/
```
空/未解压的任务先跳过或先解压, 不要直接进进化循环 (会全 EXP 失败, 浪费 LLM 提案)。

### C.1 先单任务 GPU 打通 (小步验证链路)
```powershell
# 图像任务真实 GPU 训练 (数据已在远程 mlebench_raw_data/aerial-cactus-identification)
python scripts/run_evolution.py --task-config configs/evolution/aerial_cactus.json --runner gpu --iterations 3 --mcgs
#   验收: experiments/evolution/aerial-cactus-identification_gpu_<ts>/summary.json 有 best_cv_score;
#         search_graph.json 的 promotion_history 出现真实晋级/hold 决策; 失败的 EXP 有 run_error.txt
```

### C.2 大数据单任务 (真正的"大规模数据"验证)
现有 config 里 **`nyc_taxi.json` (5.4GB) 是最大的、可直接跑的大数据任务**, 优先用它验证大规模行为:
```powershell
python scripts/run_evolution.py --task-config configs/evolution/nyc_taxi.json --runner gpu --iterations 6 --mcgs --timeout 1800
#   大数据: 提案 budget 会对 >20 万行子采样做 CV, 但仍对全量 test 出预测。--timeout 视数据量放大。
```

### C.3 再批量大规模自进化 (多任务 + 更多轮次 + MCGS 大脑)
```powershell
# 批量: 跨多任务, 每任务多轮自进化, 走 GPU。--mcgs 必须显式加, 否则各任务不走大脑!
python scripts/run_evolution_batch.py --only champs leaf_classification nomad2018 nyc_taxi tps_dec2021 tps_may2022 ventilator --runner gpu --iterations 4 --timeout 1200 --mcgs
#   产物: experiments/evolution/_batch_gpu_<ts>/{batch_leaderboard.json, batch_leaderboard.md}
#         + 每任务独立目录 (summary.json / search_graph.json / best_solution.py)
```
> **`--mcgs` 已于 20260703 加入 batch 脚本** (原先 `_run_one` 不透传, 会导致批量跑不走 MCGS 大脑 — 已修)。批量必带 `--mcgs`。
> **表格任务 GPU 利用率=0% 是正常的**: lightgbm/xgboost 走 CPU/RAM, 不吃显存。GPU 显存只有图像/NLP 的 DL 执行器才吃。别把 0% 当成"没在算" — 看远程 `ps` 里的 solution.py 进程 + 本地 EXP 目录落盘。
> **batch 顺序按文件名字母序**: champs 最先 (466 万行, 首轮慢), ventilator 最后 (604 万行)。大数据首轮可能几分钟无本地产物 (远程训练中, GPU runner 训完才下载建本地 EXP 目录) —— 用 `ls -tR ~/jinghw/scripts/gpu_tra/evolution/<task>/` 看远程进度。

### C.4 大规模测试要观察/验收的三件事 (对应 A.4)
1. **自学习**: 多轮里 best_cv_score 是否随晋级单调改善; MCGS 是否在停滞时切到 cross_branch/aggregation (看 search_graph.json 的 branch_type 与 promotion reason)
2. **主动记录失败**: 崩溃/OOM/超时是否都落 run_error.txt + 写进 retrospective_memory.json, failure_pattern 是否归类正确 (oom / shape_mismatch / estimator_api_misuse ...)
3. **自我改进**: 失败后下一轮是否切 Diff 且提案确实针对该错误 (对比 EXPk 的 run_error.txt 与 EXPk+1 的 solution.py / hypothesis)

### C.5 继续扩规模: 给更多大数据集建 config (数据已在远程, 只缺 config)
数据清单里这些大数据集**已下载但还没有 evolution config**, 想扩规模就照 §入口 里的字段新建 JSON (`remote_data_dirname` 必须与目录名逐字一致):

**已建 (20260703, 表格, 已验证 schema 对齐真实表头):**
- `champs.json` — champs-scalar-coupling, 回归, mae, 466 万行 (GroupKFold by molecule_name)
- `ventilator.json` — ventilator-pressure-prediction, 回归, mae, 604 万行 (GroupKFold by breath_id, 官方只算 u_out==0)
- `leaf_classification.json` — leaf-classification, 99 类多分类, multiclass_log_loss, 990 行×192 特征 (prepared 有 test_private.csv)
→ 现共 **9 个 config** (原 6 + 这 3)。表格执行器已在 nyc_taxi + 上述验证。

**数据真相 (20260703 SSH 全量探测, 覆盖 raw_data 81 目录):** 三件套 (train+test+sample_submission) 齐全约 **32 个**; 残缺/未解压约 **11 个** (statoil=全.7z / siim-covid19=zip未解压 / tgs-salt缺test / alaska2缺train / cdiscount只train_example / multi-modal-gesture=tar.gz未解压 / tensorflow-speech无train等); **完全空目录 32 个** (rsna-* / iwildcam-* / herbarium-2021,2022 / vesuvius 等 — 目录在但无数据)。清单里"44 可用"偏乐观, 以 C.0 实探为准。

**仍缺 config 的完整数据集 (想扩规模照建):** 图像 (dog-breed 717MB / dogs-vs-cats 814MB / kuzushiji 4.3GB / cassava / aptos2019 — 需先验图像 DL 执行器 + 远程 timm); NLP (tweet-sentiment 跨度抽取 / us-patent / google-quest 多标签回归 / jigsaw-toxic 多标签 / essay-scoring / lmsys — 需先验 NLP 执行器); billion-word 4GB 但无 sample_submission。

放大节奏: 表格 batch (上 C.3 命令) 先跑通 → 再**单独逐个**验证图像执行器 (先 aerial_cactus 小图, 再 dog-breed) 和 NLP 执行器 (先 spooky_author, 再 us-patent), 每个确认能出 submission 再纳入 batch。**非表格执行器接大脑这条链尚未独立验证** (远程可能缺 timm)。每加一个先 C.0 验数据在位。

---

## 常见坑 (实测踩过)

- **中文路径 + shell cwd 重置**: 部分终端每条命令后把 cwd 重置回别处, 且 python 处理含中文的路径易崩。命令内先 `cd "D:\桌面\codex\科研港科技"`, 读写产物用**绝对路径**, 用 Read 工具而非 `cat` 管道去读结果文件。
- **pytest 汇总行被吞**: 若 `grep passed` 拿不到, 把输出重定向到项目内文件再用 Read 读, 别依赖管道尾部。
- **GPU 产物在远程**: `--runner gpu` 时 submission.csv 等在 A40 上, 本地 `experiments/evolution/<task>_gpu_*/` 只有 summary/search_graph/审计 JSON。要核对 CSV 行数/大小需 SSH 探远程 `~/jinghw/scripts/gpu_tra/evolution/<task>/<exp>/out/`。
- **LLM key**: variation_generator 调 LLM 生成候选; 本地/GPU 都需要 `.env` 里的 key (由 load_gpu_env.ps1 写入)。无 key 时 propose 会失败并被记为 generation_failed。
- **min_delta 门禁**: 分数只高一点点 (< 1e-4) 会被正确 hold, 这是**特性不是 bug** (防 CV 噪声误晋级)。

## 接手会话启动顺序 (照做)

```
1. 读本 prompt 全文 + CLAUDE.md 的 SESSION RECOVERY 段
2. 读透 §A.3 的五个核心模块源码 (src/research_os/), 能复述闭环数据流再动手
3. 阶段 B: run_ci_checks.py → pytest tests/ (225) → 本地进化冒烟, 全绿才继续
4. §4: load_gpu_env.ps1 加载凭据; SSH 探一次确认算力机可达、数据在位、代理连通
5. 阶段 C.1: 单任务 GPU 打通 (aerial_cactus, 3 轮)
6. 阶段 C.2: 批量大规模自进化 (先 --only 子集跑通, 再放全量; 加 --mcgs 与更多 iterations)
7. 按 C.3 三件事逐项验收自学习/记录失败/自我改进; 用落盘 JSON 作证, 不凭印象
8. 全程遵守 §0 规则; 要 commit 先取得用户显式授权; 失败如实记录并汇报
```

## 一句话交接

引擎已稳定 (225 tests 绿, 三模态跑通), 阶段 B 是"再确认没坏", 阶段 C 是"放到 GPU 上用真实大数据看它自己进化"。观察的核心永远是那三件事: 它会不会自己变好、会不会记住失败、会不会拿失败改进自己 —— 全部以 `experiments/evolution/` 下的落盘产物为准。





