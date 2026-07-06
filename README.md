# AI 科研工作站 — XCIENTIST Research Agent

> 面向 Kaggle/MLE-Bench 的 AI 科研工作站，支持本地训练 + GPU 集群 +
> 文献检索 + 代码 Agent + 实验报告生成。四层架构（Agent OS → MLEvolve Search →
> XCIENTIST Claim Audit → Benchmark Feedback）。

## 🚀 新用户快速上手

**三条命令起步：**

```powershell
# 1. 克隆 + 一键安装
git clone <仓库地址> ai-research-workstation
cd ai-research-workstation
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\quick_setup.ps1

# 2. 配置 DeepSeek API Key
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\manage_deepseek_secret.ps1 install -ApiToken sk-你的key

# 3. 启动工作站
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_verified_workstation.ps1 restart
```

浏览器打开 **http://127.0.0.1:8088/?page=control**

```powershell
# 运行第一个训练任务
kaggle run titanic
```

📖 **完整新手指南：** [docs/NEW_USER_ONBOARDING_GUIDE.md](docs/NEW_USER_ONBOARDING_GUIDE.md)

---

## 系统架构

| Layer | 名称 | 功能 |
|-------|------|------|
| Layer 1 | Multi-Agent Research OS | AgentOrchestrator，任务解析→数据审计→baseline→建模→报告 |
| Layer 2 | MLEvolve Search Controller | search graph, best-so-far, progressive search, ensemble |
| Layer 3 | XCIENTIST Research Harness | validation contract, claim audit, data leakage 检查 |
| Layer 4 | Memory / Benchmark / Kaggle | retrospective memory, MLE-Bench 统计, submission gate |

## 仪表盘页面

14 个功能页面：overview · control · tasks · data · gpu · evidence · literature · workflow · code · runtime · experiments · report · gates · settings

## CLI 命令

```powershell
kaggle                         # 进入交互式 Agent
kaggle setup                   # 首次运行引导（LLM→Kaggle→Compute）
kaggle run titanic             # 运行训练
kaggle ready                   # 查看系统状态
kaggle watch -f                # 实时监控
kaggle official ...            # Kaggle 官方 CLI 透传
```

## 项目结构

```text
configs/           工作站与任务配置
src/xsci/          XCIENTIST CLI Agent 源码
web/               Next.js 前端仪表盘
scripts/           安装、管理、验证脚本
experiments/       自动生成的实验记录
workspace/         运行时产物（报告、证据、GPU job）
docs/              文档与新手引导
tests/             测试套件（69 个测试）
```

## 常用命令

```powershell
# 一键启动
powershell -File scripts\start_verified_workstation.ps1 restart

# 运行全部验证
python scripts\verify_workstation_launch_readiness.py --write-report

# 秘密管理（密钥存储在 Windows DPAPI，不写入仓库）
powershell -File scripts\manage_deepseek_secret.ps1 install -ApiToken sk-xxx
powershell -File scripts\manage_kaggle_secret.ps1 install-token -ApiToken xxx
powershell -File scripts\manage_hpc_ssh_secret.ps1 install

# 前端开发
cd web\research-agent-workstation
npm run dev            # 开发服务器 (8088)
npm run build          # 生产构建
npm run typecheck      # TypeScript 检查
```
