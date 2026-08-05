# Claude Code 项目规则

## 会话连续性

- 默认在当前项目目录启动，不要把工作目录切到用户目录。
- 不要默认续接 `--print` / `sdk-cli` 创建的会话；这些会话可能缺少标准文件和 Bash 工具。
- 需要找回旧上下文时，使用 `/resume` 或读取 `.claude/projects` 中当前项目的历史记录。
- 开始任务前先读取当前目录、`package.json`、最近修改文件和 `.runtime-logs`。
- 如果任务被中断，重开后先检查未完成任务、最近日志、`git status` 和上次命令输出，再继续。

## 进程安全

- 禁止使用 `taskkill /F /IM node.exe`。
- 禁止使用 `Stop-Process -Name node`。
- 禁止任何全局杀 Node 的命令。
- Claude Code 的本地 API router/bridge 也依赖 Node 运行，全局杀 Node 会导致 `API Error: Unable to connect to API (ConnectionRefused)`。
- 需要重启本项目 dev server 时，只能按命令行或端口精确停止本项目进程。

安全停止 8088 端口示例：

```powershell
$conns = Get-NetTCPConnection -LocalPort 8088 -State Listen -ErrorAction SilentlyContinue
foreach ($conn in $conns) {
  $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$($conn.OwningProcess)" -ErrorAction SilentlyContinue
  if ($proc -and $proc.CommandLine -match [regex]::Escape((Get-Location).ProviderPath)) {
    Stop-Process -Id $conn.OwningProcess -Force
  }
}
```

启动开发服务：

```powershell
npm run dev -- --hostname 127.0.0.1 --port 8088
```

## 验证

- 修改后优先运行最相关验证：`npm run typecheck`、`npm run build` 或页面手动检查。
- 如果验证失败，先保留错误日志，再定位修复。
