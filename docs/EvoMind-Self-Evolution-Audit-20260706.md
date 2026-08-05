# EvoMind 自进化学习能力真实性审计报告

**审计日期**: 2026-07-06 | **审计范围**: 4 Layers, 30+ 源文件, 52 实验目录, 150+ 测试 | **结论: PARTIAL**

---

## 1. 总结结论

**PARTIAL** — 系统具备部分自进化闭环，关键的搜索图、记忆库、promotion gate、validation contract 和 claim audit 均为真实实现且有代码和 artifact 证明。但 **events.jsonl 审计流缺失**（代码已写但未在真实运行中产生证据），且部分新近添加的监控组件（recovery_guard, tool_ledger, evolution_tracker）尚无持久化 artifact 证据。

### 诚实说明（可以直接讲给老师听）

**系统目前已经证明了什么：**
- ✅ 52 个真实实验运行，覆盖 15+ Kaggle 比赛，产生了 search_graph.json、summary.json、best_solution.py
- ✅ Retrospective Memory（55KB）在实验间读写——LLM 在生成新代码时会读取过去的成功/失败经验
- ✅ Search Graph 正确实现了 best-so-far promotion gate（失败的运行即使刷出了分数也不会被 promote）
- ✅ MCGS Selector 实现了 UCT 树搜索、branch 分叉、DIVERSIFY 和 AGGREGATION 拓扑
- ✅ 每次实验产生 validation_contract.json + claim_audit.json（XCIENTIST 审计）
- ✅ 163 个测试全部通过，覆盖了 promotion gate、MCGS、memory 反馈循环

**系统还没有证明什么：**
- ❌ 事件审计流（events.jsonl）从未在真实运行中产生——代码存在但无 artifact
- ❌ 所有历史运行都是 GPU 远程执行，无 `_local_` 运行的证据
- ❌ 没有 MLE-Bench 75 的完整 benchmark 结果——benchmark 目录仅有模板和部分 round 数据
- ❌ 前端仪表盘是否真正展示实时训练事件——代码存在但无运行时证据
- ❌ 自进化 tracker（EvolutionTracker）的持久化能力——tracker.json 从未产生

**下一步如何证明真正自动进化能力：**
1. 在 `_local_` 模式下运行一次完整的 EvolutionLoop，验证 events.jsonl 是否产生
2. 跑 1-2 个完整任务的独立实验，对比第 1 轮和第 10 轮的 CV 提升和 memory 使用
3. 启动前端仪表盘（8088），验证 events.jsonl 实时可读

---

## 2. 能力矩阵

| 能力项 | 是否真实实现 | 证据代码 | 证据 Artifact | 测试覆盖 | 风险 | 下一步修复 |
|--------|------------|---------|-------------|---------|------|-----------|
| 多轮实验生成 | ✅ REAL | `evolution_loop.py:244-413 EvolutionLoop.run()` | 52 search_graph.json | `test_evolution_engine.py:353-385` | 低 | 无需 |
| Search Graph | ✅ REAL | `search_graph.py:43-54 SearchGraph` | 52 search_graph.json | `test_agent_mcgs.py:67-76` | 低 | 无需 |
| MCGS/Selector | ✅ REAL | `mcgs_selector.py:68-89 MCGSSelector` | N/A（纯内存） | `test_mcgs_selector.py:9 tests` | 低 | 无需——已修复原始 bug（visit count 不递增） |
| Best-so-far Protection | ✅ REAL | `search_graph.py:75-167 decide_promotion()` | search_graph.json | `test_agent_tools.py:55-74` | 低 | 已通过 3 个独立测试 |
| Retrospective Memory 写入 | ✅ REAL | `evolution_loop.py:222-233 _record_memory()` | 55KB retrospective_memory.json | `test_agent_memory.py:67-94` | 低 | 跨 session 持久化已验证 |
| Retrospective Memory 读取 | ✅ REAL | `evolution_loop.py:200-202 _lessons()` → line 308 `lessons=self._lessons()` | N/A | `test_evolution_engine.py:401-429` | 低 | 跨任务课程传输已验证 |
| Failure Attribution | ✅ REAL | `evolution_loop.py:568-620 _classify_failure()` 18 种模式 | run_error.txt + memory json | `test_evolution_engine.py:131-155 (16 cases)` | 低 | 暂态错误重试已验证 |
| Reusable Strategy | ✅ REAL | `retrospective_memory.py:11-22 MemoryRecord` | retrospective_memory.json | `test_agent_memory.py:42-66` | 低 | 已注入 LLM prompt |
| Feature Engineering Branch | ✅ REAL | `strategy_selector.py:68-125 recommend_strategies()` | N/A | N/A（纯逻辑） | 中 | 需确认策略是否真的传给 VariationGenerator |
| Model Family Search | ✅ REAL | `model_selection.py:189-269 recommend_model_strategy()` | N/A | `test_evolution_engine.py` 间接覆盖 | 中 | 需验证 LLM 是否真的使用推荐 |
| Ensemble/Stacking | ⚠️ PARTIAL | `mlevolve_controller.py` rank gate | N/A | 无直接测试 | 中 | 仅有 rank gate 框架，未验证实际 ensemble 执行 |
| Validation Contract | ✅ REAL | `validation_contract.py:25-78` | validation_contract.json per experiment | `test_validation_contract.py:7 tests` | 低 | 已通过 artifact check + acceptance |
| Claim Audit | ✅ REAL | `claim_audit.py:58-95 audit_claim()` | claim_audit.json per experiment | `test_agent_tools.py:103-113` | 低 | 确实能 reject 过度结论 |
| Benchmark Manager | ⚠️ PARTIAL | `benchmark_manager.py` 完整实现 | benchmark_results_round3/4.json | `test_benchmark_manager.py:7 tests` | 高 | MLE-Bench 75 仅有模板，无完整对比结果 |
| Front-end Event Visibility | ❌ GAP | `engine.py:162 JsonlEventSink` | **0 个 events.jsonl** | `test_kaggle_stream.py:11 tests` | **P0** | 需验证 events.jsonl 是否在真实运行中产生 |
| Terminal Streaming | ✅ REAL | `terminal_events.py:TerminalEventEmitter` | `test_kaggle_stream.py:137-160 preflight test` | `test_autokaggle_cli.py:test_preflight_stages_appear_in_output` | 低 | 终端正常工作，但 dashboard 联动未验证 |
| Local/GPU Compute Routing | ✅ REAL | `agent.py:27-34 _make_runner()` | 52 GPU runs, 0 local runs | `test_autokaggle_cli.py:test_local_compute_override_bypasses_gpu_manifest_blocker` | 中 | 缺少 `_local_` 运行证据 |

---

## 3. 调用链图

```
用户输入 "开始训练"
  └── kaggle.py:_dispatch_intent()
        ├── classify() → EXECUTION intent
        ├── _infer_compute_override("用本地算力") → compute=local
        ├── _print_preflight_stream() → 6 阶段预检
        │     ├── TerminalTools.dispatch("inspect_task")    → 读取任务 JSON
        │     ├── TerminalTools.dispatch("data_check")      → 检查 CSV 文件
        │     ├── TerminalTools.dispatch("model_status")    → 读取 LLM 配置
        │     └── blocking_setup(compute_override=local)    → gate check
        ├── _execution_blocker_reply() → 若阻塞则返回修复建议
        └── _run_agent(task, root, goal, compute=local)
              └── agent.py:run_agent()
                    ├── build_plan() → RunPlan(task_name, compute=local, ...)
                    ├── _build_session()
                    │     ├── ResearchToolbox(ctx, data_dir, runner, memory, selector)
                    │     │     └── MCGSSelector(total_steps=40)  [MCGS 大脑]
                    │     └── AgentSession(context, toolbox, exp_dir)
                    └── session.run(goal)
                          └── [工具循环] send → LLM → tool_use → execute → feed → repeat
                                ├── plan_next_experiment → MCGSSelector.select(graph, step)
                                │     ├── UCT walk → _plan_expansion()
                                │     └── ExpansionPlan(expansion_type, coding_mode)
                                ├── run_experiment(hypothesis, code)
                                │     ├── runner.run() → subprocess / GPU
                                │     └── ExperimentNode → SearchGraph.add_node()
                                ├── evaluate_promotion(exp_id)
                                │     ├── search_graph.decide_promotion()
                                │     │     ├── run_success=False → NEVER promote (hard gate)
                                │     │     ├── _is_better(candidate, parent) → promote/hold
                                │     │     └── → 写入 validation_contract.json + claim_audit.json
                                │     └── selector.backpropagate() → UCT visits 递增
                                ├── record_lesson → RetrospectiveMemoryStore.add_memory()
                                │     └── → 写入 retrospective_memory.json
                                └── finish() → _finalize()
                                      ├── → 写入 summary.json, search_graph.json, best_solution.py
                                      └── → 写入 research_report.md
                    
                    [OR: execute_plan() → EvolutionLoop.run()]
                          ├── _lessons() → memory.retrieve_by_task_type()  [读取记忆]
                          ├── generator.propose(lessons=lessons, ...)  [LLM 含过去经验]
                          ├── runner.run() → 子进程执行
                          ├── _integrate() → decide_promotion() + _emit_audit()
                          └── _record_memory() → 写入记忆
```

---

## 4. 当前真实证据

### 4.1 experiments/evolution/ 目录

- **52 个实验目录**，覆盖 15+ Kaggle 比赛
- 每个目录包含：`search_graph.json`, `summary.json`, `best_solution.py`
- 部分目录包含：`validation_contract.json`, `claim_audit.json`, `run_error.txt`
- **1 个全局 memory**: `retrospective_memory.json` (55,710 bytes)
- **0 个 events.jsonl** — 关键缺口

### 4.2 benchmark/ 目录

- `local_proxy_three_task/` — round 3 & 4 的 benchmark 结果
- `kaggle_10_self_evolution/` — 10 个 Kaggle 任务的模板
- `mle_bench_75/` — 仅有模板 `tasks_template.json`，无实际运行结果

### 4.3 scripts/ 目录

- 309 个 Python 脚本：37 个 verify_*.py, 18 个 run_*.py, 5 个 build_*.py
- 包含完整的 GPU 训练、HPC 执行、ensemble、paper 生成、dashboard 管理基础设施

### 4.4 测试

- **150+ 测试** 全部通过
- 覆盖：promotion gate, MCGS, memory 读写, validation contract, claim audit, benchmark metrics, CLI 命令, intent classification, preflight stages

---

## 5. 缺口清单（按严重程度排序）

### P0 — 没有它就不能说自进化

| # | 缺口 | 影响 | 修复方向 |
|---|------|------|---------|
| P0-1 | **events.jsonl 审计流缺失** | 前端仪表盘无法看到实时训练进度；自进化能力缺乏可观测证据 | 运行一次完整 local EvolutionLoop，验证 `events.jsonl` 是否产生。若代码路径正确但未触发，检查 `JsonlEventSink` 的 directory 创建逻辑 |
| P0-2 | **缺少 `_local_` 运行证据** | 所有 52 个历史运行都是 GPU (`_gpu_`)。无法证明 local compute routing 端到端工作 | 在本地运行 `evomind auto <task>`（有 LLM key 的情况下），产生至少 1 个 `*_local_*` 运行目录 |
| P0-3 | **MLE-Bench 75 仅有模板** | `benchmark/mle_bench_75/tasks_template.json` 无实际分数。声称"可比较 MLEvolve"需要真实运行数据 | 在至少 1 个 MLE-Bench 任务上运行并记录结果 |

### P1 — 影响自进化质量

| # | 缺口 | 修复方向 |
|---|------|---------|
| P1-1 | **recovery_guard.md 无持久化证据** | 代码存在但无运行时 artifact。在新运行中验证是否写入 `.xsci/recovery_guard.md` |
| P1-2 | **tool_ledger.jsonl 无持久化证据** | 同上。`TerminalAgent` 中调用了 `ToolLedger` 但无 artifact |
| P1-3 | **evolution_tracker.json 无持久化证据** | 同上。`EvolutionTracker` 有 save 代码但无文件 |
| P1-4 | **innovation_log.json 无持久化证据** | `_show_innovations()` 中调用 `InnovationEngine` 但无持久化日志 |
| P1-5 | **memory 影响决策的端到端不充分验证** | `test_agent_memory.py` 验证了跨 session 持久化，但未验证 "LLM 读过 lesson X 后选择了不同的 hypothesis Y"。这需要 LLM 集成测试 |

### P2 — 影响展示和用户体验

| # | 缺口 | 修复方向 |
|---|------|---------|
| P2-1 | **docs/ 目录是 Chrome dump，不是文档** | 缺失 MLE_BENCH_75_*, ROADMAP_TO_*, EVOLUTION_ENGINE_* 规划文档 |
| P2-2 | **会话持久化 (session.json) 无可持久化证据** | 代码存在（`SessionState.persist()`），测试通过，但生产环境的 `.xsci/` 中无此文件 |
| P2-3 | **前端仪表盘 ↔ 后端 events 联动未验证** | `web/research-agent-workstation` 代码存在但未运行过完整的端到端 dashboard 展示 |

---

## 6. 修复计划

### 第 1 步：产生 events.jsonl 证据（P0-1, P0-2）

**文件**: `src/research_os/evolution_loop.py:162`

**操作**: 在本地运行一次完整的 evolution loop（需要 LLM API key），验证 `events.jsonl` 是否产生并包含合法内容。如果未产生，检查 `JsonlEventSink` 的 mkdir 和 write 路径。

**完成标准**: `experiments/evolution/<task>_local_<timestamp>/events.jsonl` 存在，至少包含 RUN_BEGIN, AGENT_MSG, TOOL_CALL, SCORE, PROMOTE, RUN_END 事件。

### 第 2 步：验证 memory 闭环（P1-5）

**操作**: 按以下步骤运行并验证：
1. 运行第 1 轮实验 → 产生至少 1 条 memory record
2. 检查 `retrospective_memory.json` 中的 `what_worked` 字段非空
3. 运行第 2 轮实验 → 检查 LLM prompt（通过日志或截获）中是否包含第 1 轮的 lesson
4. 检查第 2 轮的 hypothesis 是否与第 1 轮不同（证明了 memory 影响了决策）

### 第 3 步：补全持久化 artifact（P1-1 ~ P1-4）

**操作**: 在每个运行结束时调用 `RecoveryGuard.emit()` + `ToolLedger.record()` + `EvolutionTracker.record_run()`。确保：
- `recovery_guard.md` 在 `.xsci/` 下存在
- `tool_ledger.jsonl` 在 workspace 下存在并至少包含 1 条记录
- `evolution_tracker.json` 在 `.xsci/` 下存在并 skill_level > novice

### 第 4 步：前端仪表盘联动验证（P2-3）

**操作**: 启动前端（`http://127.0.0.1:8088`），运行一次训练，观察：
- 训练事件是否在前端实时可见
- `search_graph.json` 是否被前端读取并渲染
- 任务列表、状态、报告等视图是否正常

---

## 7. 不允许的结论

- ❌ 不能说已经超过 MLEvolve —— 当前 0 个 MLE-Bench 75 的完整 benchmark 对比结果
- ❌ 不能说已经完成 MLE-Bench 75 —— benchmark 目录仅有模板，无实际运行
- ❌ 不能说已经稳定拿奖牌 —— 无 official Kaggle submission 记录
- ❌ 不能把 proxy CV 写成 official rank —— claim_audit 硬性阻止此行为
- ❌ 不能把 UI 展示当成训练证据 —— 只以 disk artifact 为准

---

## 8. 结论

**EvoMind 自进化学习能力：真实存在，部分闭环，尚需验证。**

系统最强的证据是：
- 52 个真实实验运行产生的 search_graph.json + retrospective_memory.json
- 163 个测试通过，覆盖了所有关键 gate：promotion refusal, MCGS UCT, failure attribution, claim audit
- EvolutionLoop 的 memory 读写反馈循环（`_lessons()` → `propose()` → `_record_memory()`）已代码实现并通过测试

系统最关键的缺口是：
- events.jsonl 审计流从未在真实运行中产生（0 个 .jsonl 文件）
- recovery_guard, tool_ledger, evolution_tracker 的持久化 artifact 缺失
- MLE-Bench 75 仅停留在模板阶段
