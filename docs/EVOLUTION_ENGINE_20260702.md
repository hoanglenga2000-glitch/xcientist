# 自动进化引擎 — 实现说明 (2026-07-02)

## 这是什么

把工作站从"固定 3 步流水线"升级为**结果驱动的闭环进化引擎**。核心闭环:

```
seed baseline -> 运行 -> 打分 -> 晋升/保留 -> 记教训 -> 用历史+记忆提出下一个变体 -> 重复
```

之前系统缺的就是"提出变体"这一步(靠 `if n_prev==0/1/2` 硬编码)。现在由 LLM 读 CV 历史和记忆真实生成解题代码。

## 新增模块

| 文件 | 职责 |
|---|---|
| `src/research_os/llm_client.py` | 双后端 LLM:Opus 4.8 主 + DeepSeek 兜底。stdlib-only(GPU 上也能跑),故障自动切换,不泄密,带 User-Agent 绕过 Cloudflare 1010。 |
| `src/research_os/variation_generator.py` | 变体生成器。把任务上下文+CV历史+记忆教训+策略,交给 LLM 生成满足契约的可运行脚本。含算力预算约束。 |
| `src/research_os/evolution_loop.py` | 循环编排。晋升门禁(复用 `search_graph`)、停滞转 Diff、写记忆。Runner 可插拔。 |
| `src/research_os/gpu_runner.py` | GPU 运行器。SFTP 推代码到 A40,远程训练,取回 CV。文件只放 `gpu_tra/`。 |
| `scripts/run_evolution.py` | 统一入口。一条命令,本地或 GPU 同一套循环。 |
| `tests/test_evolution_engine.py` | 31 个离线确定性测试(含门禁/观测/瞬时重试/选点驱动/停滞不早退)。 |
| `src/research_os/mcgs_selector.py` | MCGS 选点大脑(UCT + 多分支 + 跨分支/融合 + 自举多样化)。 |
| `tests/test_mcgs_selector.py` | 9 个选点级离线测试(含冷启动到达多分支、自举桥、封顶)。 |

## 解题脚本契约

生成的脚本必须:接受 `--data-dir`/`--out-dir`;K 折 CV;打印 `CV_SCORE=<float>`;写 `submission.csv` + `metrics.json`;只用 pandas/numpy/sklearn/lightgbm/xgboost/catboost;4 分钟内跑完;>20万行则采样到约 15 万行。

## Runner 抽象 = 统一双轨的关键

`LocalSubprocessRunner` 和 `GPURunner` 实现同一个 `Runner` 协议,所以本地和 GPU 用**完全相同**的循环逻辑。这是把历史上"本地 mock 轨"和"真 GPU 轨"合一的机制。

## 已验证(真实 GPU,Opus 4.8 全程)

**nomad2018(回归,RMSLE 越低越好)**
```
EXP000 Base      0.06032  晋升
EXP001 Stepwise  0.05892  晋升   (真实改善 -2.3%)
EXP002 Stepwise  0.05892  保留   (Δ<1e-6,门禁精确拒绝打平)
```

**tps-may-2022(分类,ROC AUC 越高越好,900k 行)**
```
EXP000 Base  0.9934  干净运行(-u 无缓冲+门禁修复后复验),晋升
```
(早前批量里记录的 0.9832/0.9946 均出自非零退出的失败运行,已被修复后的门禁正确拒绝;详见下方多模态表脚注。)

## 多模态验证(4 个真实任务,真 GPU)

引擎已在 4 种问题类型上验证,证明它不只做表格:

| 任务 | 模态 | 类型 | 指标 | 最佳 CV |
|---|---|---|---|---|
| nomad2018 | 表格 | 回归(双目标) | RMSLE↓ | 0.0597 |
| tps-may-2022 | 表格 | 二分类 | ROC AUC↑ | 0.9934 |
| tps-dec-2021 | 表格 | 7 类分类 | accuracy↑ | 0.9569 |
| spooky-author | **文本** | 多分类 | log loss↓ | 0.3479 |
| aerial-cactus | **图像** | 二分类(CNN) | ROC AUC↑ | 0.9999 |

\*tps-may-2022 之前记录的 0.9946 出自一次**失败的运行**(远程进程非零退出/疑似 900k 行 OOM,
但在死亡前已把 metrics.json 刷到盘)。旧晋升门禁只看"分数+产物",漏判了这种"失败但产物已落盘"
的情况而误晋升。修复后已在 GPU 干净复验:EXP000 clean cv=**0.9934**(晋升),EXP001 真失败(held),
EXP002 Diff 恢复跑通 cv=0.9933 但未超最佳(held)。0.9934 比旧值低但**真实可复现**。

文本任务用 TF-IDF + 线性模型(无需 torch);图像任务用 torch CNN 跑 A40 CUDA。
库允许是**模态感知**的(`_modality_guidance`):表格禁 torch 保速度,图像放开 torch/torchvision/PIL。
批量运行由 `scripts/run_evolution_batch.py` 编排,产出 `batch_leaderboard.{json,md}`。

每个实验产出 `validation_contract.json` + `claim_audit.json`(源自 `research_os` 库,
非内联复刻)。claim_audit 会真实检测 drift(不是橡皮图章)。

**跨任务记忆复用**:测试证明,先前分类任务学到的策略会被自动注入到新分类任务的 prompt(`test_lessons_from_prior_task_reach_a_new_task_prompt`)。

**容错**:runner/SSH 瞬时故障转为失败 RunResult(循环转 Diff 重试),不崩任务;GPU 连接 3 次退避重试。

**晋升门禁修复(2026-07-02)**:`decide_promotion` 新增 `run_success` 硬前置条件——
运行非零退出(崩溃/OOM/超时)一律**不可晋升**,即便它在死亡前已刷出合法分数与产物。
这堵上了"失败但产物已落盘"的漏洞(远程 GPU 被 kill 时会触发)。同时 GPU 远程 Python 改为
`-u` 无缓冲执行,确保 `CV_SCORE=` 行在进程被杀前已逐行刷出、不再随块缓冲丢失。
新增两条回归测试锁定此行为(门禁级 + 循环级,复刻 tps-may-2022 场景)。

**变体存活率修复(2026-07-02)**:实测发现失败提议的自愈很弱——真正的根因是
**Diff 恢复时"蒙着眼"**。失败运行的完整报错(≤1500 字符)从不落盘,循环只把 `note[:120]`
喂回 Diff prompt;而 aerial-cactus EXP001 那个"报错"其实是一条 torch.hub 下载进度条,
真 traceback 被挤掉了。修复:
- 新增 `_clean_error_for_feedback`:剥离 `%|`/`it/s]`/`kB/s]` 等进度帧,保留真实 traceback。
- 失败时把清洗后的完整报错落盘到 `EXP###/run_error.txt`(事后可诊断)。
- 把清洗后的**完整报错**(而非 120 字符摘要)喂回 Diff prompt。
- 堵住三处噪声泄漏源(NOTES 注入、cv_history note、记忆 `what_failed`),
  进度条噪声不再进入任何 prompt 或记忆库。新增回归测试锁定 A+B(落盘+真报错进 prompt)。

**MCGS 选点大脑落地(2026-07-02)**:此前中层有"两套搜索"——工作站的
`strategy/mlevolve_search.py` 有 UCT 树的"大脑"却不接真实执行,新引擎 `evolution_loop`
有真实执行的"手"却只有线性 best-so-far。核查发现旧 UCT **不可用**:`visit_count` 声明并读取
但从不自增,`_uct_value` 永远命中 `visits==0→inf`,选点退化为"选第一个孩子";exploitation 用
`score/visits` 对 minimize 方向是错的;且零测试。
因此**不桥接旧负债**,而是把它的好设计移植成 research_os 里一个写对且有测试的
`mcgs_selector.py`(249 行):
- **方向正确的 UCT**:exploitation 用归一化 reward(复用 `search_graph._is_better`/`_node_score`),
  maximize/minimize 都对;未访问节点返回 +inf 先各试一次。
- **backpropagate 真正自增 visit_count**(沿父链),UCT 才第一次真正活。
- **4 种 expansion**:primary / intra_branch / cross_branch(分支停滞→借其它分支高分方案) /
  aggregation(全局停滞→融合 top 节点)。
- **只读 B 的 search_graph**,visit/branch/停滞全在私有 side-table,B 的审计节点 schema 零改动。
- **接入 `evolution_loop`(opt-in,默认关)**:`select→propose→run→gate→backpropagate`;
  `VariationGenerator` 新增 `expansion_type`/`reference_solutions`,cross/aggregation 把参考代码注入 prompt;
  selector 任何异常都优雅降级为线性(不崩)。入口新增 `--mcgs` 开关。
- **测试**:新增 6 条选点级 + 3 条循环级(共驱动/回退/参考进 prompt),全离线确定性。
- **状态更新(2026-07-03,真实 GPU 已验证)**:`--mcgs` 已在真实 A40(job 87907,Opus 4.8 全程)
  跑通两轮 nomad2018。**多分支在 GPU 上真实触发并胜出**——详见下方"真实 GPU 验证"。

**MCGS 落地后的三处深审修复(2026-07-02,均离线验证+回归测试锁定)**:落地后又做了一轮
"作为大脑不能出 bug"的深度审计,用离线 trace 逐步复现,发现并修复三个测试当时漏掉的真实缺陷:
1. **全局停滞早退杀死 MCGS(循环级)**:`evolution_loop` 的全局停滞 `break` 无条件触发。但对
   MCGS 而言,全局停滞恰是**触发跨分支/融合的信号**,不是停止理由。旧逻辑会在大脑最该融合时
   把循环杀掉。修复:早退**仅在线性模式(selector 为 None)生效**;MCGS 开启时跑满预算去融合。
   配对回归测试:线性模式该早退、MCGS 模式跑满预算(去齿验证:还原后 MCGS 在第 5 步被杀 `5≠6`)。
2. **单分支自举死锁(选点级,更严重)**:旧 `_plan_expansion` 里 aggregation 要 `≥2 分支`、
   cross_branch 要"别的分支有更优解",而**只有这两条路会新建分支**——于是要有第二分支才能
   建第二分支。真实单任务永远塌成一条 `intra_branch` 深链,cross/aggregation 是**够不到的死代码**
   (旧单测靠手工注入两个分支才"通过",从没测过能否自发到达多分支)。修复:分支停滞但无处可借时
   **主动 DIVERSIFY**——从全局最优另开新分支自举,之后 cross/aggregation 才可达;受 `max_branches` 封顶
   不失控。新增 3 条测试(diversify 桥、封顶、冷启动端到端到达多分支+触发 aggregation)。
3. **`_ancestors_in_branch` 缺环路保护(硬化)**:`_depth` 有 `seen` 环路守卫、它没有。当前图结构
   不可能成环,但 `export_json` 已在、未来加载图可能引入环,而无守卫会让大脑**死循环挂起**。补齐
   与 `_depth` 一致的守卫(离线用对抗性成环图验证:3 秒内返回、不挂起)。

## 真实 GPU 验证 `--mcgs`(2026-07-03,job 87907,A40,Opus 4.8 全程)

经"代理进容器"通路(Clash 7890 → SSHPiper 网关 100.85.169.63:1235 → 容器)接入真实 A40,
在 nomad2018(双目标回归,RMSLE↓)上跑通 `--mcgs`。这是 MCGS 大脑**首次在真实硬件上运行**
(此前仅离线)。连通性排障要点:本次 job 的 `10.120.18.240:6988` 是校内直连地址、代理不可达;
真正可达的是 SSHPiper 网关(同一 username 透传到当前容器),`.env` 只改网关口令即可,专属目录不变。

**8 轮结果(全部成功,无失败提议)**:
```
EXP000 Base/primary          cv=0.059770  晋升(基线)
EXP001 Base/primary          cv=0.059516  晋升
EXP002 Base/primary          cv=0.059708  保留    ┐ branch A
EXP003 Stepwise/intra_branch cv=0.059711  保留    ┘ (停滞)
EXP004 Base/primary(DIVERSIFY) cv=0.058864 晋升   ┐ branch B(从 EXP001 另开)
EXP005 Stepwise/primary      cv=0.058816  保留(Δ<min_delta) │
EXP006 Diff/intra_branch     cv=0.058732  晋升(最佳)        │
EXP007 Diff/intra_branch     cv=0.058728  保留(Δ=4e-6<1e-4) ┘
最佳=EXP006 cv=0.058732  晋升 4/8;基线→最佳 −1.74%
```

**证明了什么(诚实分级)**:
- ✅ `--mcgs` 在真实 GPU 端到端跑通,选点大脑真实驱动(`intra_branch` 只可能来自 selector,
  线性模式永不产生;模式序列 Base→Stepwise→Diff 也吻合 phase 逻辑而非线性 `_decide_mode`)。
- ✅ **多分支在 GPU 上真实触发并胜出**:EXP001 有两个孩子(EXP002 与 EXP004)= 真实分叉。
  EXP004 在 balanced 阶段却是 `Base`——只有本次修的 **DIVERSIFY 路径**会在此产出 `Base`
  (fallback primary 此时会给 Stepwise)。即 branch A 停滞两次触发自举、从最优 EXP001 另开 branch B,
  而 **branch B 产出了全局最佳解**。本会话修的"单分支自举死锁"不仅在 GPU 上生效,其开出的分支还赢了。
- ✅ 晋升门禁在 GPU 上精确:拒 EXP002(更差)、EXP007(Δ=4e-6<1e-4)、EXP005(<阈值)。
- ✅ 真实领域思考:EXP006 推理"spacegroup 是晶体学类别量(仅~6 个值)当前被当浮点喂入,
  独热能给树干净的对称群信号"——真实材料学洞见,非样板。最佳解含 LGBM/CatBoost/XGB 三族 +
  3-单纯形 OOF 融合权重搜索。
- ✅ 容错:另一轮 5 迭代里 EXP005 的 LLM 生成失败(无代码块),循环优雅 `continue` 跳过、不崩。
- ✅ **四种 expansion 全部在 GPU 触发(12 迭代轮)**:`primary×4 / cross_branch×4 / aggregation×3 /
  intra_branch×1`。拓扑在 EXP000 处分叉(→EXP001、EXP003)。而 **`aggregation`(融合)产出了该轮全局最佳解**:
  EXP006(cv=0.05885,晋升)的 hypothesis 明确是"融合共享的 lattice 几何 + spacegroup 独热特征工程 +
  三模型(LGBM/CatBoost/XGB)per-target OOF 融合权重 + 多种子平均"——即真的把多分支顶尖方案融为一体。
  至此**两大能力都在 GPU 上各自产出过冠军解**:diversify 赢 8 迭代轮、aggregation 赢 12 迭代轮。
- ✅ **本会话四处修复全部在真实 GPU 复验**:门禁(EXP004/EXP011 `run_success=False` 被正确挡下、
  不晋升)、观测(两者 `run_error.txt` 落盘的是真错——一条 `EOFError` 瞬时断连、一条真 traceback
  `from scipy_stub import ...` 的坏 import,均非进度条噪声)、自举 diversify(8 迭代轮触发并胜出)、
  停滞不早退(12 迭代轮全局停滞下仍跑满 12 轮、旧线性 bug 会中途早退)。
- ⚠️ 诚实说明:这是单任务小数据(nomad2018,2400 行)的**机制正确性**验证,证明大脑各路径在真实
  GPU 上可用且产出真实解;**跨多任务的稳定增益**仍需更大规模批量实证。CV 是本地 proxy,官方分数仍需人工提交确认。

## 运行方式

```bash
# 单任务(GPU)
python scripts/run_evolution.py --task-config configs/evolution/nomad2018.json --runner gpu --iterations 3

# 启用 MCGS 选点大脑(UCT + 多分支 + 跨分支/融合)
python scripts/run_evolution.py --task-config configs/evolution/nomad2018.json --runner gpu --iterations 6 --mcgs

# 批量(GPU),产出 leaderboard
python scripts/run_evolution_batch.py --config-dir configs/evolution --runner gpu --iterations 2

# 本地(需本地装 sklearn/lightgbm)
python scripts/run_evolution.py --task-config configs/evolution/<task>.json --runner local --data-dir <dir>
```

任务配置是小 JSON(见 `configs/evolution/`),描述 TaskContext + 远程数据目录名。

## 凭据与安全

- 全部凭据走 `.env`(gitignored)+ `gpu_credentials.py` 的 env/`*_FILE` 机制,不硬编码。
- LLM key 通过 `ANTHROPIC_API_KEY`/`DEEPSEEK_API_KEY` 解析,`repr` 不泄露。
- GPU 文件只放 `~/jinghw/scripts/gpu_tra/`。

## 已知限制 / 下一步

1. **缓存**:LT4Net 中转不透传 Anthropic 原生缓存(`cache_read` 恒 0)。省钱可切 DeepSeek(缓存生效)。
2. **旧轨道漂移**:`scripts/mlebench_closed_loop_pipeline.py` 仍内联 `.v1` 契约,应改为 import `research_os` 消除双份。
3. **规模化**:目前逐任务;需批量循环 15 个就绪数据集。
4. **图像/文本**:生成器可尝试,但未在 GPU 验证 torch 类方案。
5. **Human Gate → Kaggle**:官方分数仍需人工确认后提交(遵守规则)。

