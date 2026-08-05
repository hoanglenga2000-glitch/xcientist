# GPU + DeepSeek Code Agent 闭环上线验收 - 2026-06-14

## 当前结论

系统已经从“本地可演示、外部资源待配置”推进到“DeepSeek Code Agent 可用、GPU/HPC 算力已被 Web Terminal 证实”的状态。当前不能把自动 SSH GPU job 标记为完全可用，因为登录节点认证仍停在 password partial success 后继续要求 public key。

- Dashboard: `http://127.0.0.1:8088`
- GPU/HPC: 登录节点网络与 Web Terminal 算力已通；自动 SSH job 仍需远端授权 public key 或平台签发 SSH 凭据
- GPU 资源: `4 x NVIDIA A800-SXM4-80GB`
- CUDA: `torch 2.9.1+cu128`，CUDA 可用
- Code Agent: 无需官方 Anthropic API，当前通过 DeepSeek fallback 提供 Claude-Code-like patch drafting
- DeepSeek model: `deepseek-v4-flash`
- 安全策略: Code Agent 只产出 transcript、manifest、patch diff 和审计记录，实际代码应用仍需 Manual Gate
- 密钥策略: DeepSeek key 与 HPC SSH password 均通过 Windows DPAPI 加载，不写入仓库

## 已补齐的闭环

1. `scripts/start_verified_workstation.ps1` 启动时加载 DeepSeek 与 HPC SSH DPAPI 凭据。
2. 网关配置支持 `GPU_SSH_PASSWORD` 或 `GPU_SSH_KEY_PATH` 二选一；当前平台实际认证仍要求 public key 授权后才会达到 `is_authenticated=true`。
3. `run_hpc_ssh_command.py` 通过 Paramiko + SOCKS5 代理尝试远端白名单命令，不打印密码，并显式检查 `is_authenticated=true`。
4. GPU job 模板改为远端 PyTorch CUDA smoke，不再假设远端已有本仓库脚本；但当前会被 SSH 认证边界阻止。
5. Code Agent 在缺少 `ANTHROPIC_API_KEY` 时自动使用 `deepseek_code_agent` provider。
6. UI 与 summary 显示 `DeepSeek Code Agent Ready` 和 `SSH Gateway Ready`，不再把 Anthropic key 当成唯一入口。

## 验收命令

```powershell
npm run typecheck
npm run build
python -m compileall scripts
python scripts\verify_no_plaintext_secrets.py
python scripts\verify_deepseek_provider.py --url http://127.0.0.1:8088 --require-configured
python scripts\verify_external_resource_gateways.py --url http://127.0.0.1:8088 --allow-real-external
python scripts\run_full_acceptance.py --dashboard-url http://127.0.0.1:8088
```

## 最新验收结果

- TypeScript typecheck: passed
- Production build: passed
- Python compileall: passed
- Plaintext secret scan: passed
- DeepSeek provider smoke: passed
- External resource gateways: DeepSeek Code Agent passed; GPU automated SSH smoke currently blocked by public-key authorization
- Full acceptance: 可在不要求真实 GPU SSH job 的模式下通过；要求真实 GPU job 时应失败并保留阻塞证据

## 关键证据

- Code Agent smoke provider: `deepseek_code_agent`
- Code Agent status: `configured_smoke_tested`
- GPU Web Terminal evidence: `workspace/hpc/web_terminal_probe.txt`
- GPU automated SSH evidence: latest job artifacts show `is_authenticated=false` / public-key authorization pending
- Example failed GPU job artifact: `workspace/gpu/jobs/gpu_2026-06-14T07-59-15-475Z_eexjax.json`
- Example Code Agent patch artifact: `workspace/tasks/house_prices/code/patches/claude_agent_deepseek_code_2026-06-14T07-39-59-008Z_4zw2zh.diff`

## 仍需保持的上线边界

- Kaggle 官方 API token 尚未配置，因此官方下载和 leaderboard 提交仍保持 Human Gate + Not Configured。
- Anthropic 官方 Claude API 尚未配置，但这不阻塞当前 DeepSeek-backed Code Agent 闭环。
- GPU 自动 SSH job 需要把本机 public key 绑定到该环境 SSH 账号，或让平台提供可用于该环境的私钥/证书；在完成前不能声称“工作站已能自动调用 GPU 算力”。
- GPU 远端执行仅允许白名单训练 / smoke 模板，不开放任意 shell。
- 用户提供过的所有 API key、HPC password、代理 password 都不得写入仓库、日志、Markdown 或前端状态。
