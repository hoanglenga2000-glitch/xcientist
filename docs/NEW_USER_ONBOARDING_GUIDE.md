# AI 科研工作站 — 新用户完整上手指南

> 从零开始，每一步都有命令，每一步都有验证。

---

## 目录

1. [前置要求：你需要什么](#1-前置要求你需要什么)
2. [硬件需求说明](#2-硬件需求说明)
3. [步骤一：克隆项目并安装](#3-步骤一克隆项目并安装)
4. [步骤二：配置 API 密钥](#4-步骤二配置-api-密钥)
5. [步骤三：配置 GPU 服务器（可选）](#5-步骤三配置-gpu-服务器可选)
6. [步骤四：启动前端仪表盘](#6-步骤四启动前端仪表盘)
7. [步骤五：运行第一个训练任务](#7-步骤五运行第一个训练任务)
8. [步骤六：日常使用流程](#8-步骤六日常使用流程)
9. [故障排查](#9-故障排查)
10. [常用命令速查](#10-常用命令速查)

---

## 1. 前置要求：你需要什么

### 本地机器

| 组件 | 最低要求 | 推荐 |
|------|---------|------|
| 操作系统 | Windows 10/11 或 Linux/macOS | Windows 11 |
| Python | 3.10+ | 3.11 |
| Node.js | 18+ | 20 LTS |
| Git | 任意版本 | 最新 |
| 终端 | PowerShell 或 Git Bash | 两者都装 |
| 浏览器 | Chrome / Edge | Chrome |
| 磁盘空间 | 5GB（项目+依赖） | 20GB（含数据集） |

### API 账号（本地训练最低配置）

| 服务 | 用途 | 注册地址 |
|------|------|----------|
| DeepSeek API | LLM 推理、代码生成、报告撰写 | https://platform.deepseek.com |
| Kaggle | 数据集下载（可选） | https://kaggle.com |

### GPU 服务器（可选，用于加速训练）

如果你有自己的 GPU 服务器或 HPC 集群，系统支持 SSH 远程训练。详见[步骤三](#5-步骤三配置-gpu-服务器可选)。

---

## 2. 硬件需求说明

### 仅本地训练（无 GPU）

- 任何现代笔记本/台式机即可
- 表格类任务（Titanic, House Prices 等）在本地 CPU 上完全能跑
- DeepSeek API 在云端推理，不消耗本地资源

### 接入 GPU 服务器后的能力

系统当前对接的 GPU 服务器配置：

| 项目 | 规格 |
|------|------|
| GPU 型号 | **NVIDIA A800-SXM4-80GB** |
| 显存 | 80GB |
| CUDA | 12.8 |
| Driver | 570.195.03 |
| Python | 3.10.13 |
| 集群 | AI-X86_NVIDIA (HKUST-GZ) |
| 连接方式 | SSH via SOCKS5 proxy |

**GPU 上预装框架**: LightGBM 4.6.0, XGBoost 3.2.0, CatBoost 1.2.10, PyTorch 2.x

---

## 3. 步骤一：克隆项目并安装

### 3.1 克隆仓库

打开 **PowerShell**（推荐）或 **Git Bash**：

```powershell
# 克隆项目到你想要的位置
git clone <你的仓库地址> ai-research-workstation
cd ai-research-workstation
```

> 如果项目路径包含中文，不影响运行。系统已在中文路径上充分测试。

### 3.2 安装 Python 依赖

```powershell
# 安装核心依赖
pip install -e .

# 验证安装
python -c "import xsci; print('XCIENTIST installed OK')"
```

如果 `pip install -e .` 报错，可以先装 requirements：

```powershell
pip install -r requirements.txt
pip install -e . --no-deps
```

### 3.3 安装 Kaggle CLI 命令

```powershell
# 运行安装脚本（会自动创建 kaggle / autokaggle 命令）
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install_autokaggle_cli.ps1
```

安装成功后，会输出：

```
Created command shims in C:\Users\你的用户名\.xsci\bin
Prepended .xsci\bin to the user PATH
OK  autokaggle -> ...
OK  kaggle -> ...
```

**关闭并重新打开终端**（PATH 变更需新终端生效），然后验证：

```powershell
kaggle --help
```

应该看到：

```
  __ __                 __
 / //_/_ ____ ____ ____/ /__
/ ,< / _ `/ _ `/ _ `/ _  / -_)
/_/|_|\_,_/\_, /\_,_/\_,_/\__/
          /___/
Kaggle Research Agent  XSCI self-evolving MLE workstation
```

### 3.4 安装前端依赖

```powershell
cd web\research-agent-workstation
npm install
cd ..\..
```

### 3.5 创建 .env 配置文件

```powershell
copy .env.example .env
```

`.env` 文件已创建，但目前所有值为空。下一步填入 API 密钥。

---

## 4. 步骤二：配置 API 密钥

### 4.1 配置 DeepSeek API Key（必须）

> DeepSeek 是系统的核心 LLM 引擎。不配这个，系统无法运行。

**方法一：使用管理脚本（推荐 — 密钥存储在 Windows DPAPI，不会写入文件）**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\manage_deepseek_secret.ps1 install -ApiToken sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**方法二：写入 .env 文件（简单但不推荐 — 密钥在明文中）**

编辑 `.env` 文件：
```
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 4.2 配置 Kaggle API Key（可选）

> 需要从 Kaggle 下载数据集或提交时配置。

**获取 Kaggle API Token：**
1. 登录 https://kaggle.com
2. 点击右上角头像 → Settings → API → Create New Token
3. 下载 `kaggle.json`

**安装：**
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\manage_kaggle_secret.ps1 install-token -ApiToken "你的Kaggle API Token"
```

### 4.3 配置 Anthropic/Claude API Key（可选）

> 系统默认使用 DeepSeek 做 LLM 推理。如果你有 Anthropic API Key 想用 Claude，可以配置：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\manage_claude_secret.ps1 install -ApiToken sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 4.4 验证配置

```powershell
# 验证 DeepSeek
kaggle ready

# 输出应包含:
#   llm         Claude Opus 4.8 (ready)  ← 或 DeepSeek (ready)
#   kaggle      ready
```

---

## 5. 步骤三：配置 GPU 服务器（可选）

> **没有 GPU 也能用！** 系统设计为本地 CPU 训练 + 云端 GPU 训练双模式。
> 本地训练直接跳过此步骤即可（如 Titanic、House Prices 等表格任务在本地完全能跑）。

### 5.1 你需要什么

- 一台可通过 SSH 连接的 GPU 服务器/HPC 集群
- 服务器的 IP/域名、端口、用户名、密码或 SSH 密钥
- 如果需要代理跳转：SOCKS5 代理地址

### 5.2 配置 GPU SSH 凭据

```powershell
# 交互式安装（会提示你输入）
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\manage_hpc_ssh_secret.ps1 install

# 或通过管道输入
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install_hpc_ssh_credential_from_stdin.ps1
```

安装过程会依次询问：
1. SSH Host（如 `100.85.169.63`）
2. SSH Port（如 `1235`）
3. SSH Username
4. SSH Password
5. Remote Workspace（GPU 上的工作目录，如 `/home/user/research_agent_workstation`）
6. SOCKS5 Host（可选，代理跳转用）
7. SOCKS5 Port（可选）
8. SOCKS5 User/Password（可选）

所有凭据存储在 Windows DPAPI 加密文件中（`%APPDATA%\ResearchAgentWorkstation\`），**不会写入项目仓库**。

### 5.3 测试 GPU 连接

```powershell
# 通过 API 测试
curl -X POST http://127.0.0.1:8088/api/gpu/connections/test

# 通过 CLI 检查状态
kaggle ready
```

看到 `gpu/ssh: ready` 表示连接成功。看到 `blocked` 表示还需要检查配置。

### 5.4 GPU 代理桥（如需要）

```powershell
# 启动本地 SOCKS5 桥（如通过跳板机访问 GPU）
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\manage_hpc_proxy_bridge.ps1 start

# 验证桥接正常
python scripts\verify_hpc_socks_gateway.py --require-auth
```

---

## 6. 步骤四：启动前端仪表盘

### 6.1 一键启动（推荐）

```powershell
# 启动前端 + 加载 DPAPI 密钥 + 运行 smoke 测试
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_verified_workstation.ps1 restart
```

等待约 30 秒，输出 `status: passed` 即表示一切就绪。

### 6.2 手动启动

如果一键脚本有问题，可以手动分步启动：

```powershell
# 第一步：启动前端开发服务器
cd web\research-agent-workstation
npm run dev
```

打开浏览器访问：**http://127.0.0.1:8088/?page=control**

### 6.3 验证仪表盘

在浏览器中依次检查：
1. **Control 页** — 主控制台是否正常显示
2. **Tasks 页** — 是否能加载任务列表
3. **GPU 页** — 是否正确显示 GPU 状态（未配置则显示 blocked）
4. **Settings 页** — 是否能切换语言和主题

### 6.4 运行完整验证

```powershell
python scripts\verify_workstation_launch_readiness.py --write-report
```

期望输出：
```json
{
  "status": "passed",
  "launch_state": "demo_ready_training_blocked_by_gpu",
  "critical_failures": [],
  ...
}
```

---

## 7. 步骤五：运行第一个训练任务

### 7.1 选择任务

系统内置 3 个预配置的 Kaggle 表格任务：

| 任务 | 类型 | 数据量 | 难度 | 预计时间(本地) |
|------|------|--------|------|---------------|
| `titanic` | 二分类 | 891行 | ★☆☆☆☆ | ~2分钟 |
| `house_prices` | 回归 | 1460行 | ★★☆☆☆ | ~5分钟 |
| `telco_churn` | 二分类 | 3333行 | ★★☆☆☆ | ~3分钟 |

### 7.2 下载数据集

```powershell
# 方式一：通过 Kaggle CLI（需要 Kaggle API Key）
kaggle official competitions download -c titanic -p datasets/kaggle/titanic

# 方式二：手动下载
# 从 https://kaggle.com/c/titanic/data 下载 train.csv 放到 datasets/kaggle/titanic/
```

### 7.3 运行训练

```powershell
# 使用 CLI Agent 运行 Titanic
kaggle run titanic
```

系统会自动：
1. ✅ 加载并审计数据
2. ✅ 运行 baseline 模型
3. ✅ 搜索最优模型组合（ML-Evolve style）
4. ✅ 生成实验记录和报告
5. ✅ 验证 claim 边界

### 7.4 查看结果

**在仪表盘中查看：**
- 打开 http://127.0.0.1:8088/?page=experiments
- 点击 Titanic → 查看实验报告

**在终端中查看：**
```powershell
kaggle memory          # 查看学习记忆
kaggle ready           # 查看最新运行状态
```

**导出报告：**
- 在仪表盘 Report 页面点击"生成报告"
- 或通过 API: `POST /api/tasks/titanic/generate-report-draft`
- 报告自动保存为 Markdown + HTML + DOCX

---

## 8. 步骤六：日常使用流程

### 8.1 启动工作

```powershell
# 1. 打开终端，进入项目目录
cd ai-research-workstation

# 2. 启动仪表盘
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_verified_workstation.ps1 restart

# 3. 检查状态
kaggle ready
```

### 8.2 添加新任务

```powershell
# 方式一：从 Kaggle URL
kaggle task add https://www.kaggle.com/c/your-competition

# 方式二：手动创建配置文件
# 复制 configs/titanic.yaml → configs/your_task.yaml，修改数据路径和目标列
```

### 8.3 与系统交互

```powershell
kaggle                         # 进入对话式 Agent（类似 ChatGPT 但专注科研）
kaggle agent titanic           # 对特定任务打开 Agent
kaggle watch -f                # 实时监控训练进度
kaggle dashboard start         # 打开仪表盘
```

### 8.4 查看文献

在仪表盘 **Literature** 页面：
- 输入关键词如 "gradient boosting tabular data"
- 系统会从 arXiv + 本地论文库 + seed papers 中检索
- 结果包含摘要、方法标签、可信度分数

### 8.5 代码 Agent（生成代码补丁）

在仪表盘 **Code** 页面：
- 选择一个任务
- 输入需求描述（如 "添加 XGBoost 模型，使用 Optuna 调参"）
- 系统调用 LLM 生成代码补丁
- 生成的补丁会经过 Human Gate 审核后才能应用

---

## 9. 故障排查

### 9.1 前端无法访问 (http://127.0.0.1:8088)

```powershell
# 检查端口是否被占用
netstat -ano | findstr :8088

# 强制重启
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\restart_workstation_frontend.ps1 -Port 8088
```

### 9.2 `kaggle` 命令不识别

```powershell
# 确认 PATH
echo $env:Path  # PowerShell
echo $PATH      # Git Bash

# 重新安装 CLI
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install_autokaggle_cli.ps1

# 关闭并重新打开终端
```

### 9.3 Python 模块找不到

```powershell
pip install -e .
python -c "import xsci; print(xsci.__file__)"
```

### 9.4 DeepSeek API 报错

```powershell
# 检查余额
curl https://api.deepseek.com/user/balance -H "Authorization: Bearer sk-你的key"

# 重新安装密钥
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\manage_deepseek_secret.ps1 install -ApiToken sk-你的key
```

### 9.5 仪表盘页面 500 错误

```powershell
# build 之后 dev server 有 stale 模块，必须重启
cd web\research-agent-workstation
npm run build
cd ..\..
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_and_restart.ps1
```

### 9.6 GPU 连接失败

```powershell
# 检查 GPU 状态
curl -X POST http://127.0.0.1:8088/api/gpu/connections/test

# 重新配置 GPU SSH
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\manage_hpc_ssh_secret.ps1 install

# 查看详细 GPU 状态
kaggle ready
```

---

## 10. 常用命令速查

### 系统管理

```powershell
# 启动工作站（完整模式：加载密钥 + 启动前端 + smoke）
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_verified_workstation.ps1 restart

# 仅启动前端
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\restart_workstation_frontend.ps1 -Port 8088

# Build 后安全重启
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_and_restart.ps1

# 运行全部验证
python scripts\verify_workstation_launch_readiness.py --write-report
```

### Agent CLI

```powershell
kaggle                      # 进入交互式 Agent
kaggle ready                # 检查系统状态
kaggle status               # 同上
kaggle run <任务名>          # 运行任务训练
kaggle agent <任务名>        # 对任务打开 Agent
kaggle task add <KaggleURL> # 添加新任务
kaggle watch -f             # 实时监控
kaggle memory               # 查看学习记忆
kaggle official ...         # 调用官方 Kaggle CLI
kaggle-official ...         # 直接调用官方 Kaggle CLI
autokaggle                  # 同上（不会覆盖 kaggle 命令）
```

### API 接口

```powershell
# 工作站概览
curl http://127.0.0.1:8088/api/workstation-summary

# 任务列表
curl http://127.0.0.1:8088/api/tasks

# 任务报告
curl http://127.0.0.1:8088/api/tasks/titanic/report

# 文献搜索
curl -X POST http://127.0.0.1:8088/api/literature/search -H "Content-Type: application/json" -d '{"query":"gradient boosting"}'

# 演化状态
curl "http://127.0.0.1:8088/api/evolution/state?task_id=titanic"

# CPU 连接测试
curl -X POST http://127.0.0.1:8088/api/gpu/connections/test

# 系统设置
curl http://127.0.0.1:8088/api/settings
```

### 秘密管理

```powershell
# 安装/更新 DeepSeek API Key
powershell -File scripts\manage_deepseek_secret.ps1 install -ApiToken sk-xxx

# 安装/更新 Kaggle Token
powershell -File scripts\manage_kaggle_secret.ps1 install-token -ApiToken xxx

# 安装/更新 Claude API Key
powershell -File scripts\manage_claude_secret.ps1 install -ApiToken sk-ant-xxx

# 安装/更新 GPU SSH 凭据
powershell -File scripts\manage_hpc_ssh_secret.ps1 install
```

---

## 下一步

完成上述步骤后，你的 AI 科研工作站已经可以：

- 🖥️ 通过仪表盘可视化管理和监控实验
- 🤖 通过 CLI Agent 对话式运行机器学习任务
- 📊 自动生成含图表的实验报告
- 📚 检索 arXiv 文献并关联到任务
- 💻 生成和审核代码补丁（Human-in-the-loop）
- 🔬 在本地 CPU 上运行表格数据任务
- 🚀 接入 GPU 服务器后运行大规模训练

**现在打开 http://127.0.0.1:8088/?page=control 开始使用吧！**
