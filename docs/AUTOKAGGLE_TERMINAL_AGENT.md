# Kaggle Research Agent 全局终端

本项目提供一个像 Claude Code 一样的全局终端入口：安装后，用户可以在任意目录输入 `kaggle`，直接进入“科研港 Kaggle Research Agent”对话终端。

`autokaggle` 仍然保留为备用命令，适合不想覆盖官方 Kaggle CLI 的环境。官方 Kaggle CLI 通过 `kaggle official ...` 或 `kaggle-official ...` 透传。

## 产品定位

`kaggle` 不是普通训练脚本，也不是只包了一层 Kaggle API。它是本项目四层科研工作站的终端产品壳：

1. Multi-Agent Research OS：任务解析、数据审计、代码生成、训练、提交校验、实验台账、报告。
2. MLEvolve-style Search：多分支搜索、Base / Stepwise / Diff 代码模式、best-so-far 保护、Retrospective Memory。
3. XCIENTIST Audit Harness：validation contract、risk checklist、claim audit、claim drift 防护。
4. Memory / Benchmark Layer：跨任务记忆、MLE-Bench 任务追踪、valid submission rate、top30 / medal 统计。

系统可以生成候选 submission 和审计报告，但官方 Kaggle 提交保持人工门禁；没有 Kaggle response artifact 时，不显示官方排名、奖牌或 top30 达成。

## 安装

在项目根目录执行：

```powershell
.\scripts\install_autokaggle_cli.ps1
```

脚本会执行：

- `pip install -e .`
- 创建 `%USERPROFILE%\.xsci\bin\kaggle.cmd`
- 创建 `%USERPROFILE%\.xsci\bin\autokaggle.cmd`
- 创建 `%USERPROFILE%\.xsci\bin\kaggle-official.cmd`
- 把 `%USERPROFILE%\.xsci\bin` 放到用户 PATH 前面

重新打开一个终端后，直接输入：

```powershell
kaggle
```

即可进入科研工作站 Agent 对话页面。

如果不想覆盖官方 Kaggle CLI：

```powershell
.\scripts\install_autokaggle_cli.ps1 -NoKaggleAlias
```

如果只想创建 `kaggle.cmd`，但不改 PATH：

```powershell
.\scripts\install_autokaggle_cli.ps1 -NoPathPrepend
```

## 首次配置

第一次运行：

```powershell
kaggle
```

会进入类似 Claude Code 的首次配置向导，分三步：

1. LLM brain：配置 Anthropic 或 DeepSeek API，也可以跳过。
2. Kaggle account：配置 Kaggle API token、导入 `kaggle.json`，或使用 username/key，也可以跳过。
3. Compute backend：默认本地算力；也可以配置 GPU/SSH/HPC 服务器。

所有密钥只写入用户全局目录：

```text
~/.xsci/secrets.toml
```

项目仓库不会保存 API key、cookie、SSH 密码或 Kaggle token。

## 对话终端

进入后会看到：

```text
kaggle>
```

常用命令：

```text
task add <url|name|json>      注册并选择比赛
use <task>                    切换任务
agent [goal]                  进入或执行当前任务的深度科研 Agent
run <task>                    运行可审计自进化训练闭环
resume [task]                 续跑最近一次运行（重载搜索树 + 会话，不从零开始）
watch -f                      跟踪事件流
memory                        查看 retrospective memory
report                        查看实验报告
dashboard start               启动 8088 前端工作站
official <kaggle args...>     透传官方 Kaggle CLI
setup                         重新配置
exit                          退出
```

也可以直接输入自然语言目标：

```text
kaggle> 注册 spaceship-titanic，做第二轮自进化，优先提高 valid submission rate，不自动提交官方 Kaggle
```

## 续跑 / 崩溃恢复

一次深度运行可能因主机退出、GPU 抖动或 1800s 超时中途被杀。每一轮对话都会增量落盘到
`<exp_dir>/messages.jsonl`，搜索树落盘到 `search_graph.json`，所以运行可以从上次中断处继续，
而不是从零重来：

```text
kaggle> resume                 # 续跑当前任务最近一次运行
kaggle> 继续                    # 同义（也接受 continue）
```

命令行等价形式：

```powershell
xsci agent <task> --resume
```

`--resume` 会定位该任务最近一个带 `search_graph.json` 的运行目录，重载其搜索树（节点、边、
best-so-far、晋升记录）与对话历史，然后在**同一个目录**里继续。恢复是忠实的：只重载已持久化的
内容，绝不伪造；被杀在半途、未应答的工具调用回合会被安全丢弃（搜索树已记录该回合真正做过的事），
`_finalize` 会重新导出同一条谱系而不是覆盖成空图。

## 官方 Kaggle CLI

如果需要原始 Kaggle CLI：

```powershell
kaggle official competitions list
kaggle-official competitions list
```

## 全局工作区

当用户不在 `.xsci` 项目目录内运行 `kaggle` 时，系统会自动使用：

```text
~/.xsci/workspace
```

其中包含：

```text
~/.xsci/workspace/.xsci/tasks
~/.xsci/workspace/experiments
```

这样 `kaggle` 可以从任意目录启动，同时仍复用项目里的 `xsci`、`research_os`、工作站后端和前端能力。

## 验证命令

```powershell
kaggle --help
kaggle setup
kaggle task add https://www.kaggle.com/competitions/spaceship-titanic
kaggle agent spaceship-titanic "做第二轮自进化优化"
kaggle official competitions list
```
