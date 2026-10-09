# 生产溯源对账记录（P0-1），2026-10-09

分支：`release/v0.4-p0-fixes`（基于 master `b4967e5e`）。
本文件只记录事实和决定。上海节点（evomind-shanghai）全程只读，未做任何部署或重启。

## 1. 结论

1. 线上版本标识不准确：`/api/healthz` 返回的 `commit_hash` 固定为基线提交
   `664a636ddd419a66f73cc10820c0a16429784866`。实际运行的 overlay 是在它上面叠加
   工作区（未提交）文件构建出来的，`source_dirty=true`，但 healthz 没有暴露这一点，
   也没有记录这些文件来自哪个提交。
2. 2026-10-08 的重新部署回退了 2026-09-20 的热修。热修 D1/D2B/D4/D6/D8/D9/D10 当时是直接
   在服务器上原地打补丁（`operations\patch_d*.py`），从未进入仓库。10-08 用笔记本工作区
   重新构建 overlay 后，线上 `bundle\runtime\evomind_runtime` 中已没有任何热修标记：
   - `assistant_runs.py` 中仍是 `max_steps=24`；
   - `runtime.py` 中 flash 输出上限仍是 `16384`；
   - 不存在 `_retry_transient_file_op`。

## 2. 服务器与笔记本运行时文件比对（只读快照，2026-10-09 09:33，UTC+8）

服务器路径：`bundle\runtime\evomind_runtime`（共 71 个 .py）。
服务器副本的原始 mtime 为 2026-10-08 12:39。

| 文件 | 服务器 sha256（前 16 位） | 结论 / 决定 |
|---|---|---|
| `__init__.py` | 2cff1fcae7398e45 | 仅换行符不同（CRLF），内容相同 → 采用仓库版本 |
| `aibuild_engine.py` | de89658bb0ff133f | 笔记本版本是超集（+5 行） → 采用笔记本版本 |
| `assistant_runs.py` | 7c86f3ca27b584a5 | 笔记本版本更新（+42/-19），再叠加 D8/D9/D10 → 采用笔记本版本 + 热修 |
| `competition_data.py` | 1e45df525064c16b | 笔记本版本是超集（+148/-1，新增 histopathologic_cancer 本地数据源、MLEBENCH_PREPARED_ROOT） → 采用笔记本版本 |
| `http_server.py` | 72abef1a8c1705ed | 笔记本版本是超集（+46） → 采用笔记本版本 |
| `run_requests.py` | 7840265e6ba382b9 | 笔记本版本是超集（+40） → 采用笔记本版本 |
| `runtime.py` | b1d0f5b28bde6c4e | 笔记本版本更新（+19/-6），再叠加 D9 → 采用笔记本版本 + 热修 |
| `tenant_access.py` | e6ac54496baae000 | 笔记本版本是超集（+12） → 采用笔记本版本 |
| `super_agent_runtime.py` | 91e101158bcc9577 | 与笔记本完全相同；D1/D2B 已被取代，不修改 |

服务器上没有一个版本比笔记本更新，所以没有把服务器文件“搬回”仓库。
这些服务器文件的完整哈希保存在
`D:\AI-Outputs\EvoMind-backups\20261009-093104\` 的快照中，原件位于
`E:\tmp\evomind-server-snapshot-20261009\`。

另有 12 个文件只在笔记本上存在：

- `ev_calibration_runner.py`、`goal_release_scope.py`、`model_profile_secrets.py`、`model_profiles.py`
- `personal_model_client.py`、`personal_model_http.py`、`personal_tool_boundary.py`
- `siim_calibration_runner.py`、`siim_dataloader_runtime.py`、`siim_worker_isolation.py`
- `user_files.py`、`user_tasks.py`

它们不在 overlay 的 `RUNTIME_PATCHES` 列表中，因此当前不会被部署。

## 3. 热修移植状态

| 热修 | 目标 | 处理方式 | 位置 |
|---|---|---|---|
| D1：运行时自有目录能力对所有 Run 可用 | `super_agent_runtime.py` | **不移植（已被取代）**。脚本可以干净地应用，但应用后 `tests/test_super_agent_runtime_integration.py` 新增 2 个失败：租户会话可以访问全局 `workspace` 能力，削弱了隔离。当前代码（线上 10-08 版本与笔记本 sha256 相同，均为 91e101158bcc9577）已经用“会话级目录能力”（run_id 等于该会话）解决了同一个问题 | — |
| D2B：为单个 Run 签发作用域工作区能力 | `super_agent_runtime.py` | **不移植（已被取代）**，原因同上 | — |
| D8：训练类运行 `max_steps` 24→48，并加两条执行规则 | `assistant_runs.py` | 用原脚本先 dry-run，再对临时副本 `--apply`，结果拷回仓库 | 标记 `Hotfix D8` |
| D9：flash 输出上限 16384→32768，并要求并行数据管线 | `runtime.py` 与 `assistant_runs.py` | 同上；`tests/test_model_transport_reliability.py` 中的期望值同步改为 32768 | 标记 `Hotfix D9` |
| D10：对临时性文件写失败重试 | `assistant_runs.py` | 附件拷贝处的锚点缩进已从 16 变为 12，原脚本拒绝执行（fail closed）。改为同样的替换内容、按新缩进手工移植，并通过编译检查 | `_retry_transient_file_op` |
| D4：DeepSeek 环境变量 | 服务器 `Start-Node.ps1`（部署包，不在仓库） | 仅记录，未移植 | — |
| D6：角色端口预检 | 服务器 `Start-Node.ps1`（同上） | 仅记录，未移植 | — |

`super_agent_runtime.py` 不在 overlay 的 `RUNTIME_PATCHES` 列表中，保持不变：D1/D2B 已不需要。

需要用户确认的点：D9 把 flash 上限提高到 32768。线上从 09-20 到 10-08 一直以此值运行，
但仓库里原测试锁定的是 16384。

回归测试 `tests/test_production_hotfix_port.py` 会在 CI 中检查上述热修是否存在。
如果热修再次丢失，CI 会直接变红。

## 4. 溯源修复（代码）

- `runtime-build-manifest.json` 新增可选字段 `source_commit` 和 `source_commit_dirty`
  （schema 仍为 `evomind.runtime_build.v1`，向后兼容）。
- 新增 `sourceProvenance()`，返回以下四种之一：
  - `clean_commit`：整棵树就是某个已提交的树；
  - `clean_commit_overlay`：在基线上叠加的文件全部来自一个干净提交；
  - `dirty_worktree`：叠加文件与记录的提交不一致；
  - `unrecorded`：dirty overlay，但没有记录来源提交（**当前线上就是这种状态**）。
- `/api/healthz` 中 `commit_hash` 保持不变（`Start-Node.ps1` 依赖它），新增以下字段：
  - `base_commit`
  - `source_dirty`
  - `source_commit`
  - `source_commit_dirty`
  - `source_tree_sha256`
  - `source_provenance`
- `scripts/build_invitation_release.py stage` 新增 `--source-git-ref <ref>`：
  - 叠加文件和构建输入全部通过 `git cat-file --filters <commit>:<path>` 从已提交的树中导出，
    不读取工作区；
  - 记录 `source_commit=<sha>`、`source_commit_dirty=false`；
  - ref 无法解析或缺少文件时直接失败（fail closed）。
- 不带该参数时仍走旧的工作区路径，但会如实记录 `source_commit=HEAD`，并对实际用到的路径
  执行 `git status`，得出 `source_commit_dirty`。
- `validate_release_identity` 会校验 `runtime-build-manifest` 和 `operational-overlay-manifest`
  中的来源字段与 source receipt 一致，被篡改则拒绝。

## 5. 下一次上线必须满足

1. 所有改动先提交到分支，然后用 `stage --source-git-ref <分支提交>` 构建，
   使 `source_provenance=clean_commit_overlay`。
2. 预发（staging）验收中，`/api/healthz` 的 `source_commit` 必须等于该提交，
   且 `source_provenance` 不能是 `unrecorded` 或 `dirty_worktree`。
3. 构建产物中必须包含 D8/D9/D10 的标记（`tests/test_production_hotfix_port.py` 为绿）。
4. D4/D6 需要合入部署包的 `Start-Node.ps1` 源头。它不在本仓库，由用户确认后另行处理。
