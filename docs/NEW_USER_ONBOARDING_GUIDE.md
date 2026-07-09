# EvoMind 新用户一键配置指南

本文面向第一次下载和使用 EvoMind 的新用户。目标是让用户在本机完成安装、配置模型 API、配置 Kaggle API、可选配置 GPU/HPC 服务器，然后进入 EvoMind 工作站页面，开始受控的数据训练流程。

默认入口：

```text
http://127.0.0.1:8088/?page=control
```

核心命令：

```powershell
evomind
```

> 注意：EvoMind 是本项目的研究 Agent 命令。官方 Kaggle CLI 使用 `kaggle-official` 或 `evomind official ...`，不要把两者混淆。

## 1. 准备环境

推荐系统：

| 项目 | 最低要求 | 推荐 |
| --- | --- | --- |
| 操作系统 | Windows 10/11、Linux、macOS | Windows 11 |
| Python | 3.10+ | 3.11 |
| Node.js | 18+ | 20 LTS |
| Git | 任意现代版本 | 最新稳定版 |
| 浏览器 | Chrome / Edge | Chrome |
| 磁盘空间 | 5GB | 20GB+ |

新用户需要准备：

| 配置项 | 是否必须 | 用途 |
| --- | --- | --- |
| LLM API Key | 必须推荐 | 用于规划、代码生成、审计、报告和交互 Agent |
| Kaggle API Token | 做 Kaggle 官方数据时必须 | 下载比赛数据、生成候选提交、读取官方响应 |
| GPU/HPC SSH | 可选 | 远程训练和大规模实验 |

安全原则：

- 不要把 API key、Kaggle token、cookie、SSH key、SSH 密码写入 git。
- Windows 下推荐使用项目提供的 Windows DPAPI 脚本保存密钥。
- 官方 Kaggle 提交必须经过 Human Gate，不会默认自动提交。
- 没有官方 Kaggle response artifact 时，系统不能声明官方排名、奖牌或 top30。

## 2. 下载项目

在 Windows 上打开 **PowerShell**，先进入你希望存放项目的位置，例如桌面：

```powershell
cd $env:USERPROFILE\Desktop
```

然后输入下面命令下载仓库并进入项目目录：

```powershell
git clone https://github.com/hoanglenga2000-glitch/xcientist.git EvoMind
cd EvoMind
```

如果电脑没有安装 Git，就在浏览器打开：

```text
https://github.com/hoanglenga2000-glitch/xcientist
```

点击 **Code -> Download ZIP**，解压后进入项目根目录即可：

```powershell
cd <EvoMind项目目录>
```

项目根目录下应能看到：

```text
install.ps1
README.md
src/
web/
scripts/
docs/
```

## 3. 一键安装

在项目根目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1
```

安装脚本会完成：

- Python 包 editable install；
- Next.js 前端依赖安装与构建；
- 安装 `evomind`、`autokaggle`、`kaggle-official` 命令；
- 创建必要的本地配置文件；
- 执行轻量级上线 smoke check；
- 不启动训练；
- 不打印任何密钥。

如果只是演示或已经安装过依赖，可以使用快速模式：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1 -SkipNpmInstall -SkipBuild
```

如果只想安装命令行入口：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install_autokaggle_cli.ps1 -PrependShimPath
```

安装后请重新打开一个终端，再验证：

```powershell
where.exe evomind
evomind --help
```

`where.exe evomind` 的第一项应位于：

```text
%USERPROFILE%\.xsci\bin
```

## 4. 配置模型 API

推荐使用交互式向导：

```powershell
evomind setup
```

向导会引导用户选择模型提供商、模型名和密钥保存方式。

也可以使用 DPAPI 脚本单独配置：

```powershell
powershell -File scripts\manage_deepseek_secret.ps1 install-key -ApiKey <你的模型API_KEY>
```

如果使用兼容 OpenAI/Claude 的网关，请在 `evomind setup` 中选择对应 provider 或自定义模型名。

配置完成后检查：

```powershell
evomind ready
```

理想状态会显示 LLM ready。若 LLM 没有 ready，页面仍可打开，但代码生成、规划、报告草拟等能力会受限。

## 5. 配置 Kaggle API

### 5.1 获取 Kaggle Token

在 Kaggle 网站中进入：

```text
Account -> API -> Create New Token
```

会下载一个 `kaggle.json`。其中包含 username 和 key。

### 5.2 安装 Kaggle Token

推荐使用项目脚本安装到安全存储：

```powershell
powershell -File scripts\manage_kaggle_secret.ps1 install-token -ApiToken <你的KaggleToken>
```

如果脚本要求 username/key 分开输入，请按提示输入。不要把 token 内容写入 README、文档、issue、聊天记录或 git。

### 5.3 验证 Kaggle 状态

```powershell
evomind ready
evomind official competitions list
```

说明：

- `evomind official ...` 会透传到官方 Kaggle CLI。
- `kaggle-official ...` 也会调用官方 Kaggle CLI。
- `evomind` 本身是 EvoMind 研究 Agent，不是官方 Kaggle CLI。

## 6. 可选配置 GPU/HPC 服务器

没有 GPU/HPC 也可以使用 EvoMind 的页面、任务管理、报告、证据、文献、审计和小规模 CPU smoke。

如需远程训练，运行（口令为隐藏录入，只存 Windows DPAPI，绝不落明文）：

```powershell
powershell -File scripts\install_hpc_ssh_credential_from_stdin.ps1 -User <你的SSH账号> -HostName <登录节点地址> -Port <端口> -RemoteWorkspace <远端工作目录>
```

运行说明：

- SSH 账号、登录节点地址、端口、远端工作目录通过命令行参数提供；
- 脚本随后弹出 `HPC SSH password` 隐藏输入，把口令粘入回车即可；
- 口令经 Windows DPAPI 加密保存，绝不写入命令行历史、日志或文件；
- 需要代理时脚本默认 `-SocksHost 127.0.0.1 -SocksPort 7890`，可按需覆盖。

> 进阶：查看已存凭据状态用 `powershell -File scripts\manage_hpc_ssh_secret.ps1 status`；仅更新连接元数据（host/port/workspace）用 `... set-metadata`。该脚本的凭据写入子命令是 `install-credential`（需 `-User`/`-Password` 参数），新用户请优先用上面的隐藏录入封装，避免口令进入命令行历史。

配置后验证：

```powershell
evomind ready
```

启动前端后也可以测试：

```powershell
curl.exe -X POST http://127.0.0.1:8088/api/gpu/connections/test
```

注意：

- GPU/HPC 是可选训练资源，不是系统上线的必要条件。
- 如果 GPU/SSH 显示 `blocked`，说明远程训练 gate 未通过，不代表 EvoMind 控制台不可用。
- 训练任务必须通过工作站 resource gate 和 job manifest，不允许绕过工作站直接在服务器乱放脚本。

## 7. 启动 EvoMind 工作站

推荐启动命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_verified_workstation.ps1 restart
```

然后打开：

```text
http://127.0.0.1:8088/?page=control
```

如果只想重启前端：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\restart_workstation_frontend.ps1 -Port 8088
```

也可以使用 CLI：

```powershell
evomind dashboard start
```

## 8. 进入终端交互 Agent

```powershell
evomind
```

常用命令：

```powershell
evomind ready
evomind setup
evomind competitions titanic
evomind task add https://www.kaggle.com/competitions/titanic
evomind task list
evomind run <task_id>
evomind watch -f
evomind memory
evomind dashboard start
evomind official competitions list
```

典型中文交互：

```text
帮我规划 Titanic 第二轮自进化
开始训练 titanic
查看当前任务状态
生成审计报告
查看失败原因和下一轮优化计划
```

训练相关动作会进入门禁，不会绕过工作站直接执行。

## 9. 新用户第一次训练流程

建议从小任务开始：

```powershell
evomind task add https://www.kaggle.com/competitions/titanic
evomind run titanic
```

推荐流程：

1. 任务解析；
2. 数据下载或数据路径登记；
3. 数据审计；
4. baseline 生成；
5. validation contract；
6. 训练执行；
7. OOF / CV 指标记录；
8. submission schema 检查；
9. score promotion gate；
10. claim audit；
11. 报告生成；
12. 人工确认是否允许官方提交。

前端页面会显示：

- latest action trace；
- run state；
- artifact path；
- gate status；
- report draft；
- claim audit；
- memory / failure reason。

## 10. 前端页面说明

| 页面 | 作用 |
| --- | --- |
| Control | EvoMind 统一入口、自然语言命令、任务调度、最新 action trace |
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

## 11. 上线验收命令

新用户基础验收：

```powershell
python scripts\verify_new_user_release_readiness.py --write-report
```

完整新用户 release acceptance：

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
reports/NEW_USER_RELEASE_ACCEPTANCE.md
reports/WORKSTATION_INTERACTIVE_CONTROLS_20260701.md
```

## 12. 常见问题

### 12.1 `evomind` 命令找不到

重新安装命令 shim，并打开新终端：

```powershell
powershell -File scripts\install_autokaggle_cli.ps1 -PrependShimPath
where.exe evomind
```

### 12.2 8088 页面打不开

```powershell
netstat -ano | findstr :8088
powershell -File scripts\restart_workstation_frontend.ps1 -Port 8088
```

### 12.3 页面打开但没有样式

```powershell
cd web\research-agent-workstation
npm run build
cd ..\..
powershell -File scripts\restart_workstation_frontend.ps1 -Port 8088
```

### 12.4 Kaggle 显示未配置

```powershell
powershell -File scripts\manage_kaggle_secret.ps1 install-token -ApiToken <你的KaggleToken>
evomind ready
```

### 12.5 GPU 显示 blocked

这是远程训练 gate 未通过。需要配置并验证服务器（口令隐藏录入，只存 DPAPI）：

```powershell
powershell -File scripts\install_hpc_ssh_credential_from_stdin.ps1 -User <你的SSH账号> -HostName <登录节点地址> -Port <端口> -RemoteWorkspace <远端工作目录>
curl.exe -X POST http://127.0.0.1:8088/api/gpu/connections/test
evomind ready
```

### 12.6 报告、文献、代码页面按钮无反应

运行交互控件审计：

```powershell
node scripts\verify_workstation_interactive_controls.mjs --write-report
```

## 13. 对外说明边界

Training and official Kaggle submission remain gate-controlled.

可以说明：

- EvoMind 已完成新用户入口、前端网关、终端 Agent、任务管理、证据链、报告、审计和门禁流程。
- 新用户配置模型 API、Kaggle API 和可选 GPU/HPC 后，可以进入受控训练流程。
- 官方 Kaggle 提交、排名、奖牌和 top30 必须以 Kaggle response artifact 为准。

不能说明：

- 已保证拿到奖牌；
- 已保证 top30；
- 已超过 MLEvolve；
- 已完成 MLE-Bench 75 全任务；
- 没有官方 response 却声明官方排名或奖牌。

发布状态建议使用：

```text
ready_for_new_user_evomind_gateway
```

这表示系统入口和受控工作流已可供新用户配置和使用，但不等于已经证明真实比赛成绩。
