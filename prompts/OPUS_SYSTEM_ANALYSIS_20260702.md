# Opus 4.8 — AI科研工作站全系统分析 + 自进化引擎深度解读

> 生成时间: 2026-07-02 20:50 CST
> 目标: 让 Opus 完整理解系统架构、训练流程、自进化引擎，并给出优化建议

---

## 系统全景速览

这是一个 **自进化 AI 科研工作站**（Self-Evolving MLE Research OS），目标是用四层 Agent 架构自动化完成 MLE-Bench 75 全部 Kaggle 比赛，并通过渐进式搜索控制器持续改进模型方案，最终超越 MLEvolve 的 61.33% medal rate。

**核心思想**: 不是写一个普通的 Kaggle 训练脚本，而是构建一个可持续运行、可审计、可自进化的 AI 科研操作系统。

---

## 一、四层架构详解

### Layer 1: Multi-Agent Research OS (Orchestration + Execution)

**代码位置**: `src/research_agent_workstation/server/`

```
AgentOrchestrator → 14阶段闭环工作流:
  1. task_understanding     ← TaskReader Agent
  2. data_loading           ← Data Agent
  3. preprocessing          ← Data Agent + Planner Agent
  4. baseline_training      ← Trainer Agent
  5. multi_model_explore    ← SearchController Agent (Layer 2 桥接)
  6. ensemble_exploit       ← SearchController Agent
  7. oof_cv_validation      ← Reviewer Agent
  8. submission_generation   ← Writer Agent
  9. schema_check           ← Reviewer Agent
  10. validation_contract    ← Research Harness (Layer 3)
  11. claim_audit            ← Research Harness
  12. artifact_manifest      ← Reflection Agent
  13. retrospective_memory   ← Memory System (Layer 4)
  14. task_report            ← Writer Agent
```

**每个实验产生 9 个标准化 artifact**:
1. `agent_trace.json` — Agent 决策链
2. `metrics.json` — 完整指标
3. `oof_predictions.csv` — OOF/CV 预测
4. `submission.csv` — 可提交文件
5. `submission_audit.json` — schema 校验
6. `validation_contract.json` — 验证合约
7. `claim_audit.json` — 声明审计
8. `artifact_manifest.json` — artifact 清单
9. `search_controller_decision.json` — 搜索决策记录
10. `retrospective_memory.json` — 记忆更新
11. `task_report.md` — 任务报告

### Layer 2: MLEvolve Search Controller (Optimization + Self-Evolution)

**代码位置**: `src/research_os/mlevolve_controller.py`, `search_graph.py`, `strategy_selector.py`

这是系统的 **自进化引擎核心**。它不是一个固定的模型调参脚本，而是一个带记忆的探索系统：

**三种代码生成模式**:
- **Base**: 从任务 spec 生成完整 baseline（无父节点时）
- **Stepwise**: 在明确计划下逐步扩展特征/模型/验证逻辑（正常进化）
- **Diff**: 基于已有稳定代码做最小变更（停滞或失败时回退）

**两阶段探索**:
- **Exploration**: 前期覆盖多模型族（RF/GB/XGB/CatBoost/NN/Ensemble），不同特征处理路线
- **Exploitation**: 后期在最佳候选附近调参、融合、校准

**SearchGraph 数据结构**:
- `ExperimentNode` — 每个实验节点，含 exp_id, parent_id, hypothesis, artifacts, metrics, promotion status
- `SearchGraph` — 完整搜索图，含节点、边、引用边、promotion 历史
- `decide_promotion()` — **核心 invariant**: candidate 只有在 score 提升 + 所有 required artifacts 存在时才可 promote
- `detect_stagnation()` — 分支级停滞检测
- `detect_global_stagnation()` — 全局停滞检测
- `get_branch_diverse_top_candidates()` — 防止单一分支垄断融合输入

**Strategy Selector** (`strategy_selector.py`):
- 根据任务画像自动推荐金策略：target_encoding, multi_seed_ensemble, pseudo_labeling, oof_stacking, TTA, log1p_target, feature_crossing, time_series_cv
- 每种策略都有预期增益范围（+0.2% 到 -70% error）

### Layer 3: XCIENTIST Research Harness (Validation + Auditability)

**代码位置**: `src/research_os/validation_contract.py`, `claim_audit.py`

这是系统的 **科研严谨性保证层**。每次实验不只是"提分"，而是可验证的科研声明：

**ValidationContract**:
- `hypothesis` — 明确假设（如 "OOF blend 降低单模型方差"）
- `implementation_requirement` — 实现约束
- `acceptance_criteria` — 接受标准（min/max/equals）
- `ablation_plan` — 消融计划
- `risk_checklist` — 风险检查（data leakage, CV-public gap, schema mismatch）
- `conclusion_boundary` — 结论边界限制
- `required_artifacts` — 必需工件清单

**核心原则**: 
- 无证据 = 不声称
- 无官方 response = 不写排名/奖牌
- 无 private leaderboard = 只写 proxy evaluation
- 失败也是有效产出，必须记录

### Layer 4: Retrospective Memory (Learning + Reuse)

**代码位置**: `src/research_os/retrospective_memory.py`

**MemoryRecord 结构**:
```
task_type → dataset_profile → method
  → what_worked (成功策略)
  → what_failed (失败模式)
  → metric_delta (改善幅度)
  → reusable_strategy (可复用策略)
  → failure_pattern (规避模式)
  → linked_exp_ids (关联实验)
```

**检索能力**:
- `retrieve_by_task_type()` — 按任务类型检索
- `retrieve_failures()` — 检索失败记录（避免重复错误）
- `retrieve_successes()` — 检索成功策略（复用有效方法）

---

## 二、训练流程（端到端）

```
1. 输入: Kaggle URL 或任务配置 YAML
2. 数据审计: load_task_data() → train/test/sample → 数据类型/缺失值/特征分析
3. 预处理: ColumnTransformer → num(impute+scale) + cat(impute+encode)
4. 策略推荐: strategy_selector.recommend_strategies() → 金策略列表
5. EXP000 baseline: build_models() → evaluate_model() → CV score
6. Search Controller 决策: 选择下一分支/模型/代码生成模式
7. EXP001 multi-model branch: 多模型探索
8. EXP002 ensemble/stacking: 融合/堆叠
9. 每个实验产出 9+ artifacts
10. promotion gate: decide_promotion() → promote/hold
11. 停滞检测 → exploration/exploitation 切换
12. 人工 Gate → Kaggle 提交
```

**GPU 训练引擎** (`gpu_batch_trainer_v1.py`):
- 支持 tabular 比赛批量训练
- 11 个已注册比赛，含 bronze 分数线
- 多模型: RF, GB, XGB, CatBoost, HistGB, LogisticRegression, Ridge
- 多 fold CV，OOF 预测
- 自动 submission 格式匹配
- 训练记录写入 ~/jinghw/scripts/gpu_tra/results/

**闭环管道** (`mlebench_closed_loop_pipeline.py`, 932 行):
- 直接在 GPU 上运行的完整 3-exp 闭环
- 支持 --fast 模式（采样 + 少 estimators）
- 产出完整 MLE-Bench 风格结果

---

## 三、连接拓扑

```
本地 Claude Code / Opus
  → Clash Verge SOCKS5 (127.0.0.1:7890)
    → HKUST SOCKS5 代理 (8.163.52.223:1080)
      → SSHPiper (100.85.169.63:1235)
        → GPU 容器 (A40 48GB / A800 80GB, 1TB RAM)
          → Python 3.10.12, CUDA 12.8
```

**GPU 服务器文件规则**: 所有文件必须在 `~/jinghw/scripts/gpu_tra/` 下，不污染主目录。

---

## 四、当前状态

| 指标 | 值 |
|---|---|
| MLE-Bench 75 数据集 | 17 完整 + 3 部分 + 56 下载中 |
| 已验证闭环流水线 | 3 个 (tps-may-2022, nyc-taxi, tps-dec-2021) |
| 已注册比赛 (GPU trainer) | 11 个 |
| 历史实验记录 | 740+ runs |
| GPU 训练结果 | 91+ JSON + submission CSVs |
| Research OS 代码 | 骨架完成，benchmark_manager 可用 |
| 前端仪表盘 | Next.js 14, 17 页, 28 API, ✅ 通过 |
| 下载脚本 | 正在后台运行 (bash, kaggle CLI) |

---

## 五、自进化引擎的关键创新点

1. **不是固定流水线**: Search Controller 根据 CV/停滞/失败信号动态切换 exploration→exploitation 和 Base→Stepwise→Diff 代码生成模式
2. **跨任务记忆复用**: retrospective_memory 让不同比赛之间的成功/失败经验互相传递
3. **分支多样性**: BranchDiverseTopCandidates 防止单一模型族垄断，确保融合输入多元
4. **promotion gate**: 只有真正改善 best-so-far + 证据齐全的候选才能 promote，防止退化
5. **策略推荐确定性**: strategy_selector 是纯逻辑函数，可单测、可追溯
6. **声明审计**: claim_audit 防止实验报告出现过声称（claim drift）

---

## 六、需要 Opus 重点分析的问题

### 1. 训练流程
- 从 Kaggle URL → 数据下载 → baseline → 多分支探索 → ensemble → submission 的完整链路中，哪些环节最脆弱？
- GPU trainer 和闭环管道之间如何协同？是否应该合并？
- 图像/NLP 任务目前缺乏专用管道，如何扩展？

### 2. 自进化引擎
- SearchGraph 的 promotion 逻辑是否足够鲁棒？当多个分支产生相近 score 时如何决策？
- 停滞检测的阈值（min_delta=0.0001, window=2/4）是否合理？
- retrospective_memory 目前是 JSON 文件存储，是否应该升级为向量数据库以支持语义检索？
- 当前 explore/exploit 切换是手动的，是否应该自动化？

### 3. 系统架构
- AgentOrchestrator 的 14 阶段是否过于僵化？是否应该变成动态 DAG？
- 9+ artifacts 是否过多？哪些可以合并或省略？
- Layer 2 (Search Controller) 和 Layer 1 (AgentOrchestrator) 之间的接口是否清晰？

### 4. 扩展性
- 从 3 个验证任务扩展到 75 个全任务，最大瓶颈在哪？
- GPU 资源（单卡 A40）是否足够？是否需要多 GPU 并行？
- 磁盘（580GB 剩余）能否容纳 75 个完整数据集？

### 5. 与 MLEvolve 的差距
- MLEvolve 的 Progressive MCGS 和本系统的 SearchGraph 有何异同？
- MLEvolve 使用 LLM 做代码生成决策，本系统目前用规则引擎，哪种更适合？
- 要达到 61.33% medal rate，本系统还需要哪些关键能力？

---

## 七、回答格式要求

请按以下结构回答：

### A. 系统理解确认
- 用你自己的话描述本系统的核心思想（3-5 句话）
- 确认你理解了四层架构的分工

### B. 训练流程走查
- 逐阶段走查训练流程，标注每个阶段的输入/输出/风险点
- 指出你认为最需要改进的 3 个环节

### C. 自进化引擎深度分析
- 分析 SearchGraph + MLEvolveController + StrategySelector + RetrospectiveMemory 的协同机制
- 给出 3 个具体的改进建议（含伪代码或修改方案）

### D. 差距分析
- 当前（3/75 验证, 0 官方提交）→ 目标（75/75, >61.33% medal rate）的差距
- 估算时间线和资源需求

### E. 立即行动项
- 列出 5 个你现在就可以执行的改进（按优先级排序）
