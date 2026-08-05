# DeepSeek 与 HPC/GPU 资源接入验收记录 - 2026-06-13

## 结论

科研 Agent 工作站当前已经完成 DeepSeek 真实模型接入、HPC SOCKS5 网关探测、GPU SSH 作业网关的可配置能力补强，以及全量本地验收。默认运行状态保持安全：DeepSeek 已配置并可真实调用；Claude Code 与 GPU SSH 在未提供最终授权前仍显示未配置或连接失败，不伪造成已上线。

GPU/HPC 的真实边界已经定位清楚：PDF 官方 `ncat + 127.0.0.1:7890` 路径已经补齐并验证到 SSH 认证层；学校 SOCKS 代理账号已用于本地 `127.0.0.1:7890` 桥，HPC 登录账号已做 password 与 keyboard-interactive 两种认证尝试，但登录节点仍返回 `Permission denied (password,publickey)`。平台登录节点可达，GPU 开发环境卡片里的 `10.120.x:6988` 内网地址仍不能从校外 SOCKS 路径直连。当前不能把 GPU 标为 Fully Ready，只能标记为 `GPU Environment Created / Web Terminal Ready / External SSH Pending`，并等待 Web 终端内 `nvidia-smi` 输出证明 4 张 A800 真实可见。

## DeepSeek 接入状态

- 官方文档入口：[DeepSeek API 文档](https://api-docs.deepseek.com/zh-cn/)
- API base URL：`https://api.deepseek.com`
- 当前工作站默认模型：`deepseek-v4-flash`
- 备用模型：`deepseek-v4-pro`
- 后端接口：`POST /api/llm/deepseek/smoke`
- 密钥策略：只读取 `DEEPSEEK_API_KEY`、`DEEPSEEK_API_KEY_FILE` 或 `WORKSTATION_SECRET_DIR`，不写入 SQLite、源码、文档或 env 模板。
- 当前重启后的 8088 运行进程没有拿到 `DEEPSEEK_API_KEY`，因此 live API/UI 正确显示 `Not Configured`。这不抹掉既有实现和历史 smoke 证据，但不能把当前运行态伪装成已配置。

实测结果：

- `scripts/verify_deepseek_provider.py --url http://127.0.0.1:8088 --require-configured`：通过。
- smoke route 返回：`status=passed`，`content=deepseek-ok`。
- 审计产物示例：`workspace/llm/deepseek_smoke_2026-06-13T10-36-32-110Z.json`。
- 总验收脚本已纳入 DeepSeek provider 检查。

## HPC / GPU 网关状态

已确认的资源入口：

- HPC Web：`http://100.85.169.63:1234/appform/login`
- HPC SSH 网关：`100.85.169.63:1235`
- SOCKS5 代理：`8.163.52.223:1080`
- SSH banner：`SSH-2.0-SSHPiper`

新增能力：

- `scripts/hpc_socks_proxy.py` 支持 SOCKS5 username/password 认证。
- 新增 `scripts/verify_hpc_socks_gateway.py`，用于复测 SOCKS5 到 SSH 网关的 banner。
- GPU SSH 后端支持 `GPU_SSH_SOCKS_HOST`、`GPU_SSH_SOCKS_PORT`、`GPU_SSH_SOCKS_USER`、`GPU_SSH_SOCKS_PASSWORD`。
- Windows 下自动把 ProxyCommand 脚本复制到 ASCII 临时目录，避免项目中文路径导致 OpenSSH 启动 Python 失败。
- GPU 作业仍只允许白名单模板，不开放任意 shell。

已验证：

- 远端 SOCKS5 代理需要账号密码认证。
- 认证通过后可读到 HPC SSH 网关 banner。
- Ncat 已通过 Nmap portable 官方包安装到用户目录：`C:/Users/景浩伟/AppData/Local/Programs/NmapPortable/nmap-7.92/ncat.exe`。
- `ncat --version` 返回 `Ncat: Version 7.92`。
- PDF 原版命令路径 `ssh -o ProxyCommand="ncat --proxy 127.0.0.1:7890 --proxy-type socks5 %h %p" aimslab@100.85.169.63 -p 1235` 已验证到认证层。
- 使用用户提供的 HPC 账号做 password 与 keyboard-interactive 登录测试，均返回 `Permission denied (password,publickey)`；未把密码写入项目文件或文档。
- `C:/Users/景浩伟/.ssh/config` 已备份为 `config.bak-20260613-200518`，并将 `Host hpc-hkust-gz` 收口到 ncat ProxyCommand。
- 工作站后端在临时配置 GPU/HPC env 后，能真实调用 `/api/gpu/connections/test`。
- GPU/HPC endpoint 已经能走到真实 SSH 授权层，失败原因是登录节点账号认证仍返回 `Permission denied (password,publickey)`，不是本地代码、ncat 或代理链路失败。
- 平台 GPU 环境内网 SSH `10.120.18.240:6988` 通过 `127.0.0.1:7890` 测试返回 SOCKS general failure，不能标记为外部 SSH ready。

待管理员/老师确认：

- `aimslab` 是否已经开通 SSH shell 权限。
- 是否必须在 HPC Web 平台绑定本机 SSH 公钥。
- 当前账号是否要求组合认证、首次登录激活或二次认证。
- 远程工作目录应使用哪个路径，例如 `$HOME`、项目目录或平台分配的数据目录。
- 可用 GPU 队列、镜像、CUDA/Python 环境和作业提交方式。

## 新增/修改的关键文件

- `web/research-agent-workstation/src/lib/server/deepseek-provider.ts`
- `web/research-agent-workstation/src/app/api/llm/deepseek/smoke/route.ts`
- `web/research-agent-workstation/src/lib/server/capabilities.ts`
- `web/research-agent-workstation/src/lib/server/gpu-ssh-gateway.ts`
- `web/research-agent-workstation/src/lib/server/summary.ts`
- `scripts/verify_deepseek_provider.py`
- `scripts/verify_hpc_socks_gateway.py`
- `scripts/hpc_socks_proxy.py`
- `scripts/run_full_acceptance.py`
- `.env.example`
- `web/research-agent-workstation/.env.example`
- `docker-compose.yml`

## 视觉与浏览器验收

浏览器验收必须以实际页面状态为准，不把插件异常或登录后的平台页面猜测伪装成成功。已用本机浏览器生成页面截图：

- 桌面截图：`docs/chrome_deepseek_hpc_desktop_20260613.png`
- 移动截图：`docs/chrome_deepseek_hpc_mobile_20260613.png`

截图显示 dashboard 可正常打开，DeepSeek 接入状态、任务队列、Gate、证据和行动日志区域可见。最新前端状态已调整为：GPU 未显示 `SSH Gateway Ready`，而显示 `GPU Environment Created / Web Terminal Ready / External SSH Pending`；只有拿到 Web 终端 `nvidia-smi` 结果后才能改成 Fully Ready。

## 最终验收命令

已通过：

- `npm run build`
- `npm run typecheck`
- `python -m compileall src scripts`
- `python scripts/verify_launch_integration_hardening.py`
- `python scripts/verify_deepseek_provider.py --url http://127.0.0.1:8088 --require-configured`
- `python scripts/verify_backend_resource_status.py --url http://127.0.0.1:8088`
- `python scripts/verify_hpc_socks_gateway.py --require-auth`
- `python scripts/run_full_acceptance.py --dashboard-url http://127.0.0.1:8088`
- `python scripts/verify_no_plaintext_secrets.py`

全量验收结果：`passed`，`checks_run=34`。

## 上线判断

当前可以向老师说明：

1. 本地科研 Agent 工作站不是空壳，已经能跑本地 tabular 实验、报告/图表、Gate、Action Log、Kaggle-style 新任务 smoke 和完整验收。
2. DeepSeek 模型已经真实接入，后续可以作为科研问答、报告初稿、证据摘要和轻量推理能力的模型提供方。
3. GPU/HPC 已完成网络层与后端网关准备；只差账号侧 SSH shell 授权与可用 GPU 队列信息。
4. 当前最强资源只能表述为“平台环境显示 4 x NVIDIA A800-SXM4-80GB、32 CPU、约 1TB RAM”；拿到 `nvidia-smi` 输出前不能表述为“4 x A800 已证实可用”。
5. 老师给出 GPU SSH 最终授权或 Web 终端输出后，下一步应先运行连接 smoke：`whoami && hostname && pwd && python --version && nvidia-smi && df -hT && free -h`，通过后再提交白名单训练任务。
6. 所有外部密钥都必须轮换后再正式上线；聊天中出现过的 API Key 和密码都应视为已暴露，不应作为长期生产凭证。
