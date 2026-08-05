# GPU/HPC 真实 SSH 闭环验收 - 2026-06-14

## 结论

当前本机科研 AI 工作站已经可以通过项目脚本真实调用 HKUST(GZ) GPU 环境算力。

- 连接方式：`127.0.0.1:7890` SOCKS5/ncat 代理到 `100.85.169.63:1235`
- SSH 用户：环境卡分配的 GPU SSH 用户，存储在 Windows DPAPI
- 密码策略：只保存在用户本机 DPAPI，不写入仓库、日志、Markdown 或前端状态
- 远程账号：`aimslab`
- 远程主机：`9439b3552598`
- Python：`3.13.2`
- CUDA/PyTorch：`torch 2.9.1+cu128`
- GPU：`4 x NVIDIA A800-SXM4-80GB`
- 单卡显存：`81920 MiB`
- 验收标记：`GPU_JOB_COMPLETED`

## 实测证据

项目脚本通过 Paramiko + SOCKS5 代理完成远程命令执行：

```powershell
python scripts\run_hpc_ssh_command.py `
  --host 100.85.169.63 `
  --port 1235 `
  --user <DPAPI_USER> `
  --proxy-host 127.0.0.1 `
  --proxy-port 7890 `
  --command-file <temp_remote_smoke_script>
```

关键输出：

```text
aimslab
9439b3552598
Python 3.13.2
{"cuda_available": true, "cuda_device_count": 4, "cuda_matmul_sum": 134217728.0, "device0": "NVIDIA A800-SXM4-80GB", "torch_version": "2.9.1+cu128"}
0, NVIDIA A800-SXM4-80GB, 81920 MiB, 81154 MiB
1, NVIDIA A800-SXM4-80GB, 81920 MiB, 81154 MiB
2, NVIDIA A800-SXM4-80GB, 81920 MiB, 81154 MiB
3, NVIDIA A800-SXM4-80GB, 81920 MiB, 81154 MiB
GPU_JOB_COMPLETED
```

## 安全边界

- 不开放任意 shell。
- GPU 入口继续保持白名单模板策略。
- Kaggle 官方下载、leaderboard 提交仍需要 Kaggle token 与 Human Gate。
- 环境更新导致 SSH 密码变化时，需要重新安装 DPAPI 凭据后再验收。
- 不在报告里记录任何密码、API key、SSH 私钥或 Kaggle token。

## 当前状态

- GPU/HPC：`Ready`
- DeepSeek：`Ready`
- Claude Code：`Ready via DeepSeek fallback`
- Kaggle 官方 API：`Not Configured / Awaiting Token`
- Leaderboard 提交：`Human Gate Required`
