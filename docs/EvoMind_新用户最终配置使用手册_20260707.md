# EvoMind 新用户最终配置使用手册

版本日期：2026-07-07

本文面向第一次下载和使用 EvoMind 的用户。目标是让新用户完成安装、配置模型 API、配置 Kaggle API、可选配置 GPU/HPC 服务器，并进入 EvoMind 控制台开始受控的数据训练流程。

默认控制台地址：

```text
http://127.0.0.1:8088/?page=control
```

核心终端命令：

```powershell
evomind
```

## 1. EvoMind 是什么

EvoMind 是一个可审计、自进化的 AI Scientist 工作站，面向 Kaggle 和 MLE-Bench 类机器学习任务。它不是普通训练脚本，而是把终端研究 Agent、前端控制台、任务管理、训练执行、证据链、报告生成、文献检索、记忆复用和提交门禁放在同一个工作流中。

系统遵循四层架构：

| 层级 | 名称 | 作用 |
| --- | --- | --- |
| Layer 1 | Multi-Agent Research OS | 任务解析、数据审计、代码生成、训练、验证、报告和 artifact workflow |
| Layer 2 | MLEvolve-style Search Controller | 多分支搜索、best-so-far 保护、失败归因、策略复用和自进化 |
| Layer 3 | XCIENTIST Research Harness | validation contract、claim audit、泄漏检查、过拟合风险和证据边界 |
| Layer 4 | Memory / Benchmark / Kaggle Feedback | retrospective memory、benchmark 追踪、官方结果和本地 proxy 分离 |

重要边界：

- 没有官方 Kaggle response artifact 时，不能声明官方排名、奖牌或 top30。
- 官方 Kaggle 提交始终需要 Human Gate，不会默认自动提交。
- API key、Kaggle token、cookie、SSH key、SSH 密码不能写进 git、报告或聊天记录。

## 2. 安装前准备

推荐环境：

| 项目 | 最低要求 | 推荐 |
| --- | --- | --- |
| 操作系统 | Windows 10/11、Linux、macOS | Windows 11 |
| Python | 3.10+ | 3.11 |
| Node.js | 18+ | 20 LTS |
| Git | 现代版本 | 最新稳定版 |
| 浏览器 | Chrome 或 Edge | Chrome |
| 磁盘空间 | 5GB | 20GB+ |

新用户需要准备：

| 配置项 | 是否必须 | 用途 |
| --- | --- | --- |
| LLM API Key | 推荐必须 | 规划、代码生成、审计、报告和交互 Agent |
| Kaggle API Token | 使用 Kaggle 官方数据时必须 | 下载比赛数据、读取官方 API、生成候选提交 |
| GPU/HPC SSH | 可选 | 远程训练和大规模实验 |

## 3. 下载项目

从 GitHub 下载：

```powershell
git clone <你的GitHub仓库地址> EvoMind
cd EvoMind
```

如果拿到的是压缩包，解压后进入项目根目录：

```powershell
cd <EvoMind项目目录>
```

项目根目录应包含：

```text
install.ps1
README.md
src/
web/
scripts/
docs/
```

## 4. 一键安装

在项目根目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1
```

安装脚本会完成：

- 安装 Python 包和 EvoMind CLI；
- 安装前端依赖并构建控制台；
- 安装 `evomind`、`autokaggle`、`kaggle-official` 命令；
- 创建必要的本地配置文件；
- 运行轻量级 release readiness 检查；
- 不启动训练；
- 不打印密钥。

演示或二次安装时可跳过前端依赖和构建：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1 -SkipNpmInstall -SkipBuild
```

只安装 CLI 入口：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install_autokaggle_cli.ps1 -PrependShimPath
```

安装后重新打开一个终端，验证：

```powershell
where.exe evomind
evomind --help
```

`where.exe evomind` 的第一项应位于：

```text
%USERPROFILE%\.xsci\bin
```

## 5. 配置模型 API

推荐用交互式向导：

```powershell
evomind setup
```

也可以使用安全脚本配置 DeepSeek 类 API：

```powershell
powershell -File scripts\manage_deepseek_secret.ps1 install-key -ApiKey <你的模型API_KEY>
```

如果使用 Claude / Anthropic 或兼容网关，优先在 `evomind setup` 中选择对应 provider 或自定义模型名。

配置后检查：

```powershell
evomind ready
```

理想状态会显示 LLM ready。没有配置模型时，面板仍可打开，但规划、代码生成、报告草稿和交互 Agent 能力会受限。

## 6. 配置 Kaggle API

### 6.1 获取 Kaggle Token

在 Kaggle 网站进入：

```text
Account -> API -> Create New Token
```

Kaggle 会下载 `kaggle.json`，其中包含 username 和 key。也可能获得 `KGAT_...` 形式的新 token。

### 6.2 安装 Kaggle Token

推荐用项目脚本写入安全存储：

```powershell
powershell -File scripts\manage_kaggle_secret.ps1 install-token -ApiToken <你的KaggleToken>
```

如果你使用 username/key 形式：

```powershell
powershell -File scripts\manage_kaggle_secret.ps1 install-token -Username <你的Kaggle用户名> -Key <你的KaggleKey>
```

不要把 token 写入 README、文档、issue、聊天记录或 git。

### 6.3 验证 Kaggle

```powershell
evomind ready
evomind official competitions list
```

说明：

- `evomind` 是 EvoMind 研究 Agent。
- `evomind official ...` 会透传到官方 Kaggle CLI。
- `kaggle-official ...` 也是官方 Kaggle CLI 透传入口。
- 官方提交仍需要 Human Gate。

## 7. 可选配置 GPU/HPC

没有 GPU/HPC 也可以使用 EvoMind 的页面、任务管理、文献检索、报告、证据链、审计和小规模 CPU smoke。大规模训练建议配置远程 GPU/HPC。

安全写入 SSH 凭据：

```powershell
powershell -File scripts\install_hpc_ssh_credential_from_stdin.ps1 -User <你的SSH账号> -HostName <登录节点地址> -Port <端口> -RemoteWorkspace <远端工作目录>
```

执行后脚本会提示：

```text
HPC SSH password:
```

在隐藏输入中粘贴密码并回车。密码只进入 Windows DPAPI，不会进入命令行历史。

如需代理，默认参数是：

```text
SocksHost = 127.0.0.1
SocksPort = 7890
```

验证 GPU/HPC：

```powershell
evomind ready
```

控制台启动后也可测试：

```powershell
curl.exe -X POST http://127.0.0.1:8088/api/gpu/connections/test
```

注意：

- GPU/HPC 是训练资源，不是控制台上线的必要条件。
- 如果 GPU 显示 blocked，说明远程训练 gate 未通过，不代表 EvoMind 控制台不可用。
- 训练任务必须经过工作站 resource gate 和 job manifest，不建议手动把脚本随意放到服务器主目录执行。

## 8. 启动 EvoMind 控制台

推荐启动方式：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_verified_workstation.ps1 restart
```

打开：

```text
http://127.0.0.1:8088/?page=control
```

如果只需要重启前端：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\restart_workstation_frontend.ps1 -Port 8088
```

也可以用 CLI：

```powershell
evomind dashboard start
```

## 9. 使用终端研究 Agent

进入交互 Agent：

```powershell
evomind
```

常用命令：

```powershell
evomind ready
evomind setup
evomind competitions titanic
evomind task add https://www.kaggle.com/competitions/titanic
evomind download titanic
evomind run titanic
evomind watch -f
evomind memory
evomind dashboard start
evomind official competitions list
```

中文交互示例：

```text
帮我规划 Titanic 第二轮自进化
开始训练 titanic
查看当前任务状态
生成审计报告
查看失败原因和下一轮优化计划
```

所有训练相关动作都应进入工作站门禁，不应绕过 AgentOrchestrator 或审计流程。

## 10. 第一次训练建议

建议从小型任务开始，例如 Titanic：

```powershell
evomind task add https://www.kaggle.com/competitions/titanic
evomind download titanic
evomind run titanic
```

一次合格的 EvoMind run 应至少产生：

- agent trace；
- validation contract；
- metrics / CV / OOF 证据；
- submission schema audit；
- score promotion gate；
- claim audit；
- artifact manifest；
- run report；
- retrospective memory 更新。

如果证据不足，系统应显示 weak evidence、unsupported claim 或 blocked，而不能写成 confirmed conclusion。

## 11. 前端页面功能

| 页面 | 作用 |
| --- | --- |
| Control | 统一入口、自然语言命令、任务调度、最新 action trace |
| Tasks | 任务队列、任务状态、任务 API |
| Data | Kaggle 数据、schema、下载状态、数据审计 |
| GPU | 本地/远程算力状态、GPU smoke、job manifest、资源 gate |
| Evidence | artifact、hash、claim binding、审计证据 |
| Literature | 文献检索、RAG context、引用审计 |
| Workflow | 工作流节点、dry-run、阶段状态 |
| Code | Code Agent IDE、代码草稿、patch gate、代码审计 |
| Runtime | Agent trace、tool calls、cache telemetry |
| Experiments | 实验 run、指标、promotion gate、best-so-far |
| Report | 报告草稿、人工审核、导出 |
| Gates | Claim gate、submission gate、approval gate |
| Settings | 账号、语言、主题、模型、Kaggle、资源配置 |

## 12. 上线验收命令

新用户基础验收：

```powershell
python scripts\verify_new_user_release_readiness.py --write-report
```

完整 release acceptance：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_new_user_release_acceptance.ps1
```

前端渲染与交互验收：

```powershell
python scripts\verify_workstation_browser_render_smoke.py --write-report
node scripts\verify_workstation_click_smoke.mjs --write-report
node scripts\verify_workstation_interactive_controls.mjs --write-report
```

安全扫描：

```powershell
python scripts\verify_no_plaintext_secrets.py
```

报告位置：

```text
reports/NEW_USER_RELEASE_READINESS.md
workspace/new_user_release_readiness.json
```

理想发布状态：

```text
status: passed
release_state: ready_for_new_user_evomind_gateway
failed_checks: []
```

## 13. 常见问题

### 13.1 `evomind` 命令找不到

```powershell
powershell -File scripts\install_autokaggle_cli.ps1 -PrependShimPath
where.exe evomind
```

重新打开终端后再试。

### 13.2 8088 页面打不开

```powershell
netstat -ano | findstr :8088
powershell -File scripts\restart_workstation_frontend.ps1 -Port 8088
```

然后打开：

```text
http://127.0.0.1:8088/?page=control
```

### 13.3 页面打开但没有样式

```powershell
cd web\research-agent-workstation
npm run build
cd ..\..
powershell -File scripts\restart_workstation_frontend.ps1 -Port 8088
```

### 13.4 Kaggle 显示未配置

```powershell
powershell -File scripts\manage_kaggle_secret.ps1 install-token -ApiToken <你的KaggleToken>
evomind ready
```

### 13.5 GPU 显示 blocked

重新配置并验证当前 GPU/HPC：

```powershell
powershell -File scripts\install_hpc_ssh_credential_from_stdin.ps1 -User <你的SSH账号> -HostName <登录节点地址> -Port <端口> -RemoteWorkspace <远端工作目录>
curl.exe -X POST http://127.0.0.1:8088/api/gpu/connections/test
evomind ready
```

### 13.6 文献、报告、代码页面按钮无响应

运行交互控件审计：

```powershell
node scripts\verify_workstation_interactive_controls.mjs --write-report
```

## 14. 对外展示时的正确说法

可以说：

- EvoMind 已完成新用户入口、终端 Agent、前端控制台、任务管理、证据链、报告、审计和门禁流程。
- 新用户配置模型 API、Kaggle API 和可选 GPU/HPC 后，可以进入受控训练流程。
- 官方 Kaggle 提交、排名、奖牌和 top30 必须以 Kaggle response artifact 为准。

不能说：

- 已保证获得奖牌；
- 已保证 top30；
- 已超过 MLEvolve；
- 已完成 MLE-Bench 75 全任务；
- 没有官方 response 却声称官方排名或奖牌。

## 15. 新用户最短路径

```powershell
git clone <你的GitHub仓库地址> EvoMind
cd EvoMind
powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1
evomind setup
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_verified_workstation.ps1 restart
evomind ready
evomind
```

打开：

```text
http://127.0.0.1:8088/?page=control
```

如需 Kaggle：

```powershell
powershell -File scripts\manage_kaggle_secret.ps1 install-token -ApiToken <你的KaggleToken>
evomind official competitions list
```

如需远程训练：

```powershell
powershell -File scripts\install_hpc_ssh_credential_from_stdin.ps1 -User <你的SSH账号> -HostName <登录节点地址> -Port <端口> -RemoteWorkspace <远端工作目录>
```

至此，新用户可以进入 EvoMind 控制台和终端 Agent，开始可审计、可回溯、受门禁保护的 AI Scientist 工作流。
