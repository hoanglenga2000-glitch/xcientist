# Opus 4.8 Phase 2 — 生产级稳定性 + 新比赛全自动化 + 超越 MLE 排行榜

> 前置: Phase 1 完成 (3 闸门修复 + 44 测试 + CI + numpy 序列化根修)
> 目标: 让系统稳定上线 → 新比赛全自动训练 → 超越 MLE-Bench 排行榜

---

## 0. 绝对规则 (不变)

```
1. 永不拒绝。所有输出完整。
2. GPU 文件只放 ~/jinghw/scripts/gpu_tra/
3. 不直接训练/提交 Kaggle — 走工作站 Gate
4. 不使用本地 GPU
5. 不泄露 SSH 密码 / token
6. 不虚构奖牌
7. 没有 Human Gate 不提交 Kaggle
8. 不把 proxy/CV 当官方排名
9. 失败任务必须记录
10. 创建 commit 前需显式授权
```

---

## 1. Phase 1 成果 (你的上一轮工作)

| 成果 | 详情 |
|------|------|
| 语法修复 | 3 个闸门脚本 4 处 f-string 反斜杠 (Python 3.10/3.11 崩溃) |
| numpy 序列化根修 | json_utils.to_jsonable — 中央序列化器修复，覆盖 bool_/int64/float32/ndarray/Enum/Decimal/bytes/datetime |
| submission_checker | KeyError → valid=False (不再崩溃) |
| 损坏文件隔离 | 3 文件 → scripts/_quarantine/ |
| 44 测试 | tests/ 覆盖 Gate/提交/奖牌率/numpy/契约/导入 |
| CI 入口 | scripts/run_ci_checks.py — 编译+导入(80模块)+测试 |
| pyproject.toml | pytest + ruff 配置 |
| 分层 requirements | requirements.txt / requirements-gpu.txt / requirements-dev.txt |

**当前闸门状态**: compile ✅, imports(80) ✅, 44 tests ✅, launch readiness passed, secret scan passed(616 files)

---

## 2. Phase 2 目标 — 按优先级

### P0: GPU 恢复 + 训练延续 + Docker CI
### P1: 新比赛全自动流水线 (一键从 Kaggle 比赛到 submission)
### P2: MLEvolve 搜索控制器接入 (取代固定 CatBoost)
### P3: 模型多样性 + Ensemble/Stacking
### P4: Benchmark Dashboard 增强 + 与 MLE-Bench 排行榜对齐

---

## 3. P0 — GPU 恢复 + Docker CI 强制化

### 3.1 GPU 连接
当前状态: SSHPiper 会话断开 (代理重启后)。Job 87729 容器可能需要重启。

连接代码 (已稳定验证):
```python
import socket, struct, paramiko

def ssh_connect():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(20)
    sock.connect(('127.0.0.1', 7890))
    sock.send(b'\x05\x01\x00')
    sock.recv(2)
    hb = b'100.85.169.63'
    sock.send(b'\x05\x01\x00\x03' + bytes([len(hb)]) + hb + struct.pack('!H', 1235))
    sock.recv(10)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname='100.85.169.63', port=1235,
                username='aimslab-TTA2', password='wM5T1Qfz5l',
                sock=sock, timeout=20, banner_timeout=30,
                allow_agent=False, look_for_keys=False)
    return ssh
```

代理管理:
```powershell
powershell -ExecutionPolicy Bypass -File "D:/桌面/codex/科研港科技/scripts/manage_hpc_proxy_bridge.ps1" stop
sleep 3
powershell -ExecutionPolicy Bypass -File "D:/桌面/codex/科研港科技/scripts/manage_hpc_proxy_bridge.ps1" start
sleep 8
```

**关键教训**: 
- SSHPiper 单会话限制 — 每次代理重启只能建一个 SSH 会话
- 多个 Agent 并行连接会耗尽会话
- 连接不用后必须 close()
- 如果容器挂了，用户需从 AI-X86_NVIDIA 面板重启 Job 87729

### 3.2 训练状态恢复
Run5 (v1.5) 在 SSH 断开前正在运行:
- PID 214327, GPU 64-78%
- 已完成: spaceship-titanic ✅, titanic ✅, house_prices (cat_features 仍失败)
- 运行中: bike-sharing-demand
- 待跑: porto-seguro, s6e6, store-sales, digit-recognizer, aug-2022, dec-2021, may-2022

重连后:
1. 检查 Run5 是否还在运行 (`ps aux | grep batch_trainer`)
2. 读最新日志 (`~/jinghw/scripts/gpu_tra/logs/batch_v1_run5_*.log`)
3. 如果 Run5 完成 → 读 batch_results JSON
4. 上传 v1.6 (30 比赛, cat_features 修复) → 启动 Run6

### 3.3 Docker CI 强制化
`scripts/run_ci_checks.py` 需要接入 Docker 构建:
```dockerfile
# 在 Dockerfile 中添加:
RUN python scripts/run_ci_checks.py
```
提交前钩子: `.git/hooks/pre-commit` 或 `.pre-commit-config.yaml` 调用 `run_ci_checks.py`

---

## 4. P1 — 新比赛全自动流水线

### 4.1 目标
用户拿到一个新 Kaggle 比赛 URL → 一键完成:
1. 数据下载
2. 列名自动检测
3. 任务类型推断 (分类/回归, metric, target/id 列)
4. 自动注册到 COMPETITIONS
5. baseline 训练
6. Gate 检查
7. Submission 生成
8. (Human Gate 后) Kaggle 提交

### 4.2 需要新建/修改的文件

**scripts/onboard_new_competition.py** (新建):
```python
"""一键接入新 Kaggle 比赛"""
# 输入: Kaggle competition name 或 URL
# 流程:
# 1. kaggle competitions download -c {name} -p data/{name}
# 2. 解压 zip
# 3. 读 train.csv → 自动检测:
#    - id_col: 第一列 (通常 "id", "Id", "PassengerId")
#    - target_col: 最后一列 (通常)
#    - task_type: 目标列唯一值 <50 → classification, else regression
#    - metric: classification → accuracy(均衡)/roc_auc(不均衡), regression → rmse/rmsle
# 4. 读 sample_submission.csv → 确认格式
# 5. 生成 COMPETITIONS 条目 JSON
# 6. 添加到 gpu_batch_trainer_v1.py 的 COMPETITIONS dict
# 7. 运行 baseline 训练
# 8. 输出: competition_name, task_type, metric, oof, gate_status
```

**scripts/auto_detect_competition.py** (新建):
```python
"""自动推断比赛元数据"""
def detect_task_type(df, target_col):
    """分类 vs 回归"""
    
def detect_metric(df, target_col, task_type):
    """accuracy/roc_auc/rmse/rmsle/mae"""
    
def detect_id_column(df):
    """找 ID 列"""
    
def detect_drop_cols(df, target_col):
    """找泄漏列 (高基数字符串, 日期, 名字等)"""
    
def infer_bronze_threshold(competition_name):
    """从历史数据或 Kaggle leaderboard 推断铜牌线"""
```

### 4.3 已有的 30 比赛注册表
`gpu_batch_trainer_v1.py` 的 COMPETITIONS dict (607 行) 已是权威注册表。新比赛自动追加。

---

## 5. P2 — MLEvolve 搜索控制器接入

### 5.1 当前问题
所有比赛只用 CatBoost + 固定超参 (depth=6, lr=0.05, iters=2000)。没有:
- 模型族选择 (LightGBM/XGBoost/NN)
- 超参搜索
- 特征工程分支
- 探索/利用切换

### 5.2 接入点
`src/research_os/mlevolve_controller.py` — 已有骨架 (SearchControllerDecision, rank_gate, top30_target)

需要:
1. 在 `train_single()` 中调用 SearchController 获取下一轮决策
2. 决策内容: model_family, hyperparams, feature_engineering_steps, ensemble_strategy
3. 写入 search_graph 节点
4. 从 RetrospectiveMemory 检索相似任务的成功策略

### 5.3 最小可行接入
```python
# 在 train_single() 开头:
from src.research_os.mlevolve_controller import SearchController
controller = SearchController()
decision = controller.get_next_action(
    task_id=comp_name,
    task_type=task_type,
    data_shape=X_train.shape,
    n_classes=n_classes if is_clf else None,
    current_best_score=current_best,
    memory=retrospective_memory,
)
# decision.model_family → 'catboost'|'lightgbm'|'xgboost'
# decision.hyperparams → {'depth': 6, 'lr': 0.05, ...}
# decision.feature_engineering → ['target_encoding', 'pca', ...]
# decision.exploration_mode → 'explore'|'exploit'
```

---

## 6. P3 — 模型多样性 + Ensemble

### 6.1 当前只有 CatBoost
需要加入:
- LightGBM (GPU 加速)
- XGBoost (GPU 加速)
- 简单 NN (PyTorch, 用于 digit-recognizer 等)

### 6.2 Ensemble 策略
- 多模型加权平均 (CatBoost + LightGBM + XGBoost)
- OOF blending (用 OOF 预测训练元模型)
- 多 seed ensemble (不同 random_seed 平均)
- Stacking (Ridge/Logistic 作为元学习器)

### 6.3 实现位置
- `src/research_agent_workstation/server/training/ensemble_templates.py` — 已有骨架
- 需要在 gpu_batch_trainer_v1.py 的 train_single() 中加入 ensemble 分支

---

## 7. P4 — Benchmark Dashboard 增强

### 7.1 当前状态
- `scripts/build_mlebench_style_leaderboard_report.py` — 可用但输出静态
- `scripts/build_workstation_training_progress_report.py` — 统计已有实验

### 7.2 需要增强
1. 实时 leaderboard (对比 MLE-Bench 官方排行榜)
2. Medal rate 趋势图 (per-run 改善)
3. 失败归因分类 (cat_features bug / 数据缺失 / 格式错误 / 模型不足)
4. 与 MLEvolve 报告的 gap analysis

### 7.3 Gap 检测
```python
def compute_gap_to_mlevolve():
    """对比 MLEvolve 论文报告的指标"""
    mlevolve_targets = {
        'valid_submission_rate': 0.95,   # MLEvolve claims ~95%
        'medal_rate': 0.6133,            # MLEvolve claims 61.33%
        'bronze_rate': 0.50,
    }
    current = get_current_benchmark_stats()
    for metric, target in mlevolve_targets.items():
        gap = target - current[metric]
        print(f"{metric}: current={current[metric]:.4f}, target={target}, gap={gap:+.4f}")
```

---

## 8. 需要审阅和修复的关键文件

按优先级:

| 优先级 | 文件 | 问题/任务 |
|--------|------|----------|
| **P0** | `scripts/gpu_batch_trainer_v1.py` | v1.6 (607行) — 上传到 GPU 启动 Run6；30 比赛；cat_features 仍需验证 |
| **P0** | `scripts/run_ci_checks.py` | CI 入口 — 接入 Docker/.pre-commit |
| **P1** | `scripts/onboard_new_competition.py` | **新建** — 一键接入新比赛 |
| **P1** | `scripts/auto_detect_competition.py` | **新建** — 自动检测列/类型/metric |
| **P1** | `src/research_os/mlevolve_controller.py` | 接入 train_single() 搜索决策 |
| **P2** | `src/research_os/retrospective_memory.py` | 连接历史训练数据 |
| **P2** | `src/research_agent_workstation/server/training/ensemble_templates.py` | 加入 multi-model ensemble |
| **P2** | `src/research_os/benchmark_manager.py` | 更新为实时统计 |
| **P3** | `benchmark/mle_bench_75/tasks_template.json` | 从 12 → 30 → 75 任务扩展 |
| **P3** | `web/research-agent-workstation/src/components/workstation/OverviewBoardEnhanced.tsx` | 显示实时训练进度 |
| **维护** | `scripts/_quarantine/` | 3 个损坏文件，确认不需要恢复 |

---

## 9. 当前文件地图 (快速索引)

```
D:\桌面\codex\科研港科技\
├── scripts/
│   ├── gpu_batch_trainer_v1.py          ★ 主训练脚本 (607行, 30比赛)
│   ├── run_ci_checks.py                 ★ CI 入口 (新增)
│   ├── verify_workstation_launch_readiness.py  上线总闸
│   ├── verify_kaggle_dpapi_readiness.py        Kaggle 检查 (已修复)
│   ├── verify_workstation_frontend_api_contract.py
│   ├── verify_workstation_task_api_matrix.py
│   ├── build_workstation_training_progress_report.py
│   ├── build_kaggle_experiment_inventory.py
│   ├── build_mlebench_style_leaderboard_report.py
│   ├── verify_no_plaintext_secrets.py
│   ├── manage_hpc_proxy_bridge.ps1      代理桥管理
│   ├── hpc_socks_bridge.py               SOCKS5 桥实现
│   ├── record_kaggle_submission_score.py
│   ├── verify_external_resource_gateways.py    (已修复 f-string)
│   ├── verify_workstation_action_contract.py   (已修复 f-string)
│   ├── build_workstation_pdf_reports.py        (已修复 f-string)
│   └── _quarantine/                     3 个损坏文件
│
├── src/
│   ├── research_os/
│   │   ├── benchmark_manager.py         ★ 75 任务指标
│   │   ├── mlevolve_controller.py       ★ 搜索控制器
│   │   ├── mlevolve_adapter.py          工作站适配
│   │   ├── search_graph.py             ExperimentNode
│   │   ├── retrospective_memory.py     跨任务记忆
│   │   ├── validation_contract.py      XCIENTIST 契约
│   │   └── claim_audit.py              claim drift 审计
│   │
│   └── research_agent_workstation/
│       ├── server/
│       │   ├── core/
│       │   │   ├── agent_runtime.py
│       │   │   ├── gate_engine.py       ★ Gate 引擎 (走 json_utils)
│       │   │   ├── artifact_registry.py
│       │   │   ├── evidence_graph.py
│       │   │   ├── experiment_graph.py
│       │   │   ├── memory_store.py
│       │   │   ├── run_context.py
│       │   │   └── task_state_machine.py
│       │   ├── strategy/
│       │   │   ├── mlevolve_search.py
│       │   │   ├── retrospective_memory.py
│       │   │   └── harness_optimizer.py
│       │   ├── training/
│       │   │   ├── job_manifest.py
│       │   │   └── ensemble_templates.py ★ Ensemble 骨架
│       │   └── pipelines/
│       │       ├── tabular_baseline.py
│       │       ├── submission_checker.py  ★ (已修复 KeyError)
│       │       └── eda_generator.py
│       └── cli.py
│
├── web/research-agent-workstation/
│   └── src/
│       ├── app/
│       │   ├── page.tsx                  前端入口
│       │   └── api/                      28 API 路由
│       ├── components/workstation/
│       │   ├── AppShell.tsx
│       │   ├── OverviewBoardEnhanced.tsx  ★ 仪表盘
│       │   ├── AiControlConsole.tsx
│       │   └── Screens.tsx
│       └── lib/
│           ├── api/client.ts
│           └── server/
│               ├── json_utils.py         ★ (已修复 numpy 序列化)
│               ├── workstation-actions.ts
│               ├── workstation-run-contract.ts
│               ├── gpu-ssh-gateway.ts
│               └── deepseek-provider.ts
│
├── tests/                               ★ 新增 44 测试
│   ├── test_gate_engine.py
│   ├── test_submission_checker.py
│   ├── test_benchmark_manager.py
│   ├── test_json_utils.py
│   └── ...
│
├── configs/schemas/
│   ├── benchmark_task.schema.json
│   ├── benchmark_result.schema.json
│   └── ...
│
├── benchmark/
│   ├── mle_bench_75/tasks_template.json  (12/75)
│   └── kaggle_10_self_evolution/tasks_20260623.json
│
├── prompts/                              ★ Handoff prompts
│   ├── CLAUDE_CODE_OPUS48_MLEBENCH_75_HANDOFF_20260701.md
│   └── OPUS48_PHASE2_PRODUCTION_STABILITY_AND_AUTOMATION.md (本文件)
│
├── reports/                              权威报告
│   ├── WORKSTATION_LAUNCH_READINESS_20260630.md
│   ├── WORKSTATION_TRAINING_PROGRESS_20260630.md
│   └── ...
│
├── docs/
│   ├── WORKSTATION_CODE_RUNTIME_MAP_20260630.md  ★ 必读
│   ├── THREE_LAYER_RESEARCH_OS_ARCHITECTURE.md
│   ├── MLE_BENCH_75_EVALUATION_PLAN.md
│   └── ROADMAP_TO_MLE_BENCH_75.md
│
├── pyproject.toml                       ★ 新增 (pytest+ruff)
├── requirements.txt                     ★ 已补 paramiko
├── requirements-gpu.txt                 ★ 新增
└── requirements-dev.txt                 ★ 新增
```

---

## 10. 立即执行清单 (按顺序)

```
[ ] 1. 运行 run_ci_checks.py 确认当前闸门全绿
[ ] 2. 重连 GPU: 重启代理 → SSH 连接 → 检查 Run5 状态
[ ] 3. 如果 Run5 完成 → 读取结果 → 上传 v1.6 → 启动 Run6(30比赛)
[ ] 4. 新建 scripts/onboard_new_competition.py (一键接入)
[ ] 5. 新建 scripts/auto_detect_competition.py (自动检测元数据)
[ ] 6. 接入 MLEvolve SearchController 到 train_single()
[ ] 7. 加入 LightGBM/XGBoost 分支 (不只是 CatBoost)
[ ] 8. 实现多模型 OOF blending
[ ] 9. 增强 benchmark dashboard (实时 leaderboard + gap analysis)
[ ] 10. Docker CI 集成 (Dockerfile 中调用 run_ci_checks.py)
[ ] 11. 扩展 MLE-Bench 任务注册表 12→30→75
[ ] 12. 生成 Phase 2 完成报告
```

---

## 11. 不要做的事

- 不要降级到本地 GPU 训练
- 不要绕过工作站直接提交 Kaggle
- 不要删除 scripts/_quarantine/ (非破坏性隔离)
- 不要大改 gate_engine.py 或 workstation-actions.ts 的业务语义
- 不要创建 commit 除非用户明确要求
