# EvoMind Codex 级能力认证：最小可执行闭环

## 当前结论

当前代码具备强制失败关闭的发布认证 verifier，但尚无真实外部 900 次试验矩阵，因此只能声明“认证基础设施已实现”，不能声明“EvoMind 已达到 Codex/Claude Code 非劣水平”。

三套本机 CLI 已可发现：

- `evomind`：提供 `workspace` 与 12 项内部 hidden-oracle `benchmark-agent`；
- `codex`：提供非交互 `codex exec --json --ephemeral -C <workspace>`；
- `claude`：提供非交互 `claude --print --output-format stream-json`。

`src/evomind_runtime/benchmark.py` 仍是 60 个占位任务、5 个域的运行时状态页，且只比较均值；它不是正式认证源。正式门禁位于 `src/xsci/capability_certification.py`，已要求外部 suite/evaluator 信任锚、至少 100 hidden tasks、至少 8 domains、每域至少 3 项、每项至少 3 次、Wilson 下界、配对非劣、原始结果重算、产物哈希和干净源代码绑定。

## 新增的本地证据 harness

`src/xsci/parity_campaign_harness.py` 与 `scripts/verify_parity_campaign.py` 负责发布认证之前的证据收口：

1. 只接受 `evomind`、`codex`、`claude-code` 的完整 `100+ tasks × 3 repeats × 3 agents` 矩阵；
2. suite 只暴露 task/domain/prompt hash，不包含 oracle 答案；
3. 每个 trial 的 private-oracle JSON 必须绑定 campaign、suite、task、domain、repeat、agent、oracle id 和 pass/fail；
4. 每个 trial 必须包含真实 `workspace_result` 与 `tool_trace` 文件；逐文件重算 bytes/SHA-256；
5. 从严格 JSONL `tool_trace` 重算 total/succeeded/failed，拒绝自报计数；
6. 统计 scope violations、unsupported claims、timeouts、成功率、Wilson 下界与 candidate-vs-baseline 配对非劣下界；
7. 本地通过仅输出 `evidence_projection_ready`，`release_gate` 仍保持 `NO-GO`；必须再经过外部信任锚和 source/artifact binding 的 `verify_capability_certification` 才能正式开门。

验证命令：

```powershell
python scripts/verify_parity_campaign.py `
  --suite-manifest EXTERNAL_EVIDENCE/suite-public.json `
  --trials EXTERNAL_EVIDENCE/trials.jsonl `
  --evidence-root EXTERNAL_EVIDENCE `
  --output EXTERNAL_EVIDENCE/parity-campaign-verification.json
```

## 正式 campaign 执行协议

1. 在独立、干净、短 ASCII 路径冻结候选 source archive、wheel、sdist 和源哈希。
2. 由独立评估方冻结 suite/evaluator SHA-256；任务选择必须在测评前锁定且未用于开发。
3. 每个 agent/task/repeat 使用全新隔离 Git workspace；相同 prompt、fixture、timeout、网络策略、CPU/RAM 与最大工具调用预算。
4. 三个 agent 的执行顺序按 task/repeat 平衡随机化，禁止跨 trial 记忆和缓存污染。
5. 执行器只授予当前 trial workspace-write；生产、HPC、Kaggle 正式提交、凭据变更和其他外部副作用均不在认证权限内。
6. 独立 evaluator 执行 private oracle，写入 trial-bound oracle evidence、workspace result、tool trace 和严格 JSONL ledger。
7. 先运行 `verify_parity_campaign.py`，再构造 `capability_certification` 外部报告并用 out-of-band digest 验证。
8. 只有完整认证结果 `certified=true` 且绑定发布 source bytes 时，产品 UI/文档才可显示 Codex 级或非劣声明。

## 成本、权限和外部依赖

- 固定执行量：至少 `100 × 3 × 3 = 900` 次 agent trial；超过 100 项时按实际数量同比增加。
- 时间：若单 trial 平均 5 分钟，串行约 75 小时；3 路并行约 25 小时，9 路并行理论约 8.3 小时，实际还受 provider rate limit、冷启动与长尾 timeout 影响。
- Token：若每 trial 总输入输出约 20k–100k tokens，campaign 总量约 18M–90M tokens；最终费用必须按三家实际 served model、缓存命中和当期账单核算，不能预先写死。
- 存储：若每 trial 的 diff/trace/oracle/日志为 1–50 MiB，原始证据约 0.9–45 GiB；归档前还需保留 source artifacts 和 verifier 输出。
- 必需权限：独立 evaluator 对临时 workspace 的读写/进程权限、三套 provider 的调用凭据和足够配额、只读读取冻结产物；不需要生产部署权限。
- 外部依赖：独立 held-out suite、private oracle、suite/evaluator out-of-band SHA-256、三套 CLI 的稳定非交互 adapter、provider 认证/配额，以及用于最终 source binding 的干净 Git release candidate。

## 仍需外部完成的事项

- 生成至少 100 项、8 个域的独立 held-out suite 与 private oracles；现有 12 项内部 suite 只能做开发回归。
- 为三套 CLI 实现由独立 evaluator 持有的统一 runner adapter，并完成一次小规模无计分 preflight 后冻结 adapter 哈希。
- 批准并执行 900 次付费 campaign；本次代码变更没有启动任何模型调用。
- 由外部评估方签发 report/suite/evaluator digest，并在干净 release candidate 上运行最终认证和发布门禁。
