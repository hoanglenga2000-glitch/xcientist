# EvoMind 终极重建 Prompt

> 这份 prompt 可以直接发给 Claude Code（Opus 4.8 终端）。
> 它基于对 Claude Code 完整运行体系的逆向分析（launcher → router → context guard → hooks），
> 结合现有 EvoMind 代码库，给出一份系统性的重建方案。

---

## 发给 Claude Code 的 Prompt

```text
你现在接手 EvoMind / XCIENTIST AI Scientist 工作站项目，工作目录是：

D:\桌面\codex\科研港科技

你的任务是把 EvoMind 从"命令行工具"升级为真正的 Claude Code 级别的科研终端超级 Agent。

═══════════════════════════════════════════════════════════════
第一阶段：理解 Claude Code 的核心架构（我已经为你分析好）
═══════════════════════════════════════════════════════════════

Claude Code 的架构分为四层，EvoMind 必须对标每一层：

**Layer 0: 终端交互层（对标 Claude Code CLI）**
- Claude Code: 交互式 REPL，流式 SSE，thinking/工具调用可见，状态栏，持续会话
- EvoMind 当前: evomind 命令 + 基础的 _agent_reply() 输出
- 差距: 没有流式 thinking 可见，没有工具调用实时展示，会话恢复不完整

**Layer 1: 中间件层（对标 MIMO Router）**
- Claude Code: Node.js router 拦截 Anthropic API 调用，管理 prompt cache，流预冲刷，
  自动上下文救援，持久化记忆写入，重试+回退
- EvoMind 当前: 直接调用 LLMClient，没有中间件
- 差距: 没有中间件层做缓存、救援、流控

**Layer 2: 生命周期钩子层（对标 Context Guard Hook）**
- Claude Code: SessionStart/UserPromptSubmit/PreCompact/PostCompact/Stop/StopFailure
  每个事件写入状态文件（git status + 会话快照 + 最近对话信号）
- EvoMind 当前: RecoveryGuard 存在但未被终端 Agent 的每个 turn 调用
- 差距: 钩子没有完全集成

**Layer 3: 工具执行层（对标 Claude Code 的 Tool-Use Loop）**
- Claude Code: Anthropic-native tool_use，模型决定调用哪个工具，
  工具结果返回模型，模型根据结果决定下一步
- EvoMind 当前: 有 ResearchToolbox（13个研究工具）和 TerminalTools（11个终端工具），
  但 ConversationAgent 是文本生成+解析，不是真正的 tool-use loop
- 差距: 终端层的工具调用不是真正的 Anthropic tool_use 格式

═══════════════════════════════════════════════════════════════
第二阶段：具体要做的事情
═══════════════════════════════════════════════════════════════

### A. 修复当前已知的 5 个产品级缺陷

这些是用户实际遇到的 bug，必须先修：

1. "house_prices 训练结果怎么样" → 不应该启动训练
   修复: kaggle_intent.py 的 _PROGRESS 匹配改用 _HARD_NOW 而非 _EXECUTION 做排除条件

2. "切换到 house_prices 开始训练" → 任务没切换
   修复: 在 ConversationAgent 的工具执行器中加入 switch_task 逻辑

3. "error: not inside an xsci project"
   修复: tasks.py 的 tasks_dir() 找不到项目时回退到全局 workspace

4. TerminalAgent 每个 turn 没有调用 RecoveryGuard.emit()
   修复: 在 handle() 的开始和结尾加入 guard 调用

5. SessionState 的 last_action 和 last_artifact 没有在对话中被更新
   修复: 在 _dispatch_intent 的每个分支更新 session 字段

### B. 实现真正的 Tool-Use Loop（而不是文本解析）

当前 ConversationAgent._scientist_loop() 的做法:
  1. 发送文本 prompt 给 LLM
  2. 用正则 _TOOL_HINT_RE 解析 LLM 回复中的 [tool: xxx] 标记
  3. 执行工具，把结果追加到 prompt
  4. 再发一次给 LLM 做综合

问题: 这不是标准的 Anthropic tool_use 格式。LLM 不知道工具的真实 schema。
      工具结果被当做文本拼接，而不是结构化的 tool_result 块。

正确做法（对标 Claude Code 的 AgentMessageClient）:
  1. 构造 tools 参数（Anthropic tool_use 格式），包含 TerminalTools 的完整 schema
  2. 发送给 LLM，LLM 返回 tool_use blocks
  3. 解析 tool_use，执行对应工具
  4. 构造 tool_result blocks，追加到消息历史
  5. 继续发送，LLM 可以再次调用工具或给出最终回答
  6. 最多 3 轮工具调用后强制 LLM 给出最终回答

代码位置: src/xsci/kaggle_conversation.py 的 ConversationAgent._scientist_loop()

参考实现: src/research_os/agent/session.py 的 AgentSession.run() 方法（这是已有的
正确 tool-use loop 实现，针对研究工具。终端对话需要类似的 loop 但用于终端工具）。

修改方案:
  - 不要重新发明轮子，直接复用 AgentMessageClient（src/research_os/agent/messaging.py）
  - 构造一个"终端工具 spec 列表"（model_status, task_list, data_check, recent_run 等）
  - 用一个轻量的循环: send → 如果 tool_use → execute → feed tool_result → 继续
  - 最多 3 轮工具调用，之后必须给出文本回答

### C. 中间件层：Context Rescue + Prompt Cache（对标 MIMO Router）

目前 EvoMind 没有中间件层，LLM 调用直接发到 API。

对标 Claude Code router 的两个关键模式：

**C1. 自动上下文救援（autoRescueContext）**
- 在发请求前检查 body 是否超过硬上限
- 如果超过，丢弃最旧的消息对（保留系统提示 + 最近 N 条）
- 已有 src/xsci/context_rescue.py 的实现，但没有被 ConversationAgent 调用

修改方案:
  - 在 ConversationAgent._scientist_loop() 的 send 之前调用 auto_rescue_context()
  - 如果发生了修剪，在系统提示中追加 rescue notice

**C2. 流预冲刷（stream preflush）**
- Claude Code 在收到上游响应前就打开 SSE 连接，发送 keepalive
- EvoMind 已有 StageRenderer，但只在训练时使用，对话时没有

修改方案:
  - 在 ConversationAgent 的 LLM 调用前，先输出 "Thinking..."
  - 收到第一个 token 后立即开始流式输出

### D. 生命周期钩子（对标 Context Guard Hook）

Claude Code 有 6 个钩子事件，每个都写入持久化状态文件:
  SessionStart / UserPromptSubmit / PreCompact / PostCompact / Stop / StopFailure

EvoMind 已有 RecoveryGuard，需要在以下时机调用:
  - ConversationAgent.reply() 开始时 → RecoveryGuard.emit("UserPromptSubmit")
  - ConversationAgent.reply() 结束时 → RecoveryGuard.emit("PostReply")
  - TerminalAgent.handle() 开始时 → 已部分实现

修改方案:
  - 在 ConversationAgent._scientist_loop() 调用 LLM 之前，调用 guard.emit()
  - 在 TerminalAgent.handle() 返回之前，调用 guard.emit()

### E. 开源参考：可以参考但不复制的项目

这些开源项目实现了类似 Claude Code 的 agent 能力，可以参考它们的架构:

1. **Aider** (https://github.com/paul-gauthier/aider)
   - 核心模式: LLM ↔ 工具 ↔ 代码编辑循环
   - 借鉴: 如何让 LLM 调用 Bash/Read/Write/Edit 工具并处理结果

2. **OpenHands** (https://github.com/All-Hands-AI/OpenHands)
   - 核心模式: Agent 在 Docker 容器中执行代码
   - 借鉴: 如何隔离 agent 的执行环境

3. **SWE-Agent** (https://github.com/princeton-nlp/SWE-agent)
   - 核心模式: Agent 用自定义 DSL 与终端交互
   - 借鉴: 如何设计 agent 的"语言"来调用工具

4. **Anthropic Cookbook** (https://github.com/anthropics/anthropic-cookbook)
   - 核心模式: 标准的 Anthropic tool_use 示例
   - 借鉴: 如何正确构造 tools 参数和解析 tool_use / tool_result

但不要直接复制这些项目的代码。EvoMind 是 Kaggle/ML 研究专用，
工具集不同（数据审计、训练、CV、门禁、claim audit）。

### F. 最终交付标准

做完后，以下场景必须无 bug:

```
C:\Users\景浩伟> evomind

evomind> 你好
→ 显示当前任务+简要建议，不启动训练

evomind> house_prices 训练结果怎么样
→ 显示所有已知任务的训练结果（含真实的 CV=0.1238, promotions=2/2）
→ 不走 preflight，不启动训练

evomind> 切换到 house_prices
→ selected_task 变为 house_prices

evomind> 这个任务数据准备好了吗
→ 检查数据并报告

evomind> 开始训练，用本地算力
→ 6阶段预检 → 门禁 → 训练启动
→ 不报 "xsci init" 错误

evomind> evolution
→ 显示自进化报告

evomind> auto house_prices
→ 全自动研究循环
```

═══════════════════════════════════════════════════════════════
第三阶段：实现参考（关键代码路径）
═══════════════════════════════════════════════════════════════

### 终端 tool-use loop 的实现骨架:

```python
# 在 ConversationAgent 中
def _real_tool_loop(self, session, user_text):
    from research_os.agent.messaging import AgentMessageClient, ToolSpec, ToolResult
    
    client = AgentMessageClient()
    
    # 构造终端工具 spec
    terminal_tool_specs = [
        ToolSpec("model_status", "Get current LLM provider/model/readiness", {
            "type": "object", "properties": {}, "required": []
        }),
        ToolSpec("task_list", "List all registered competitions", {
            "type": "object", "properties": {}, "required": []
        }),
        ToolSpec("data_check", "Check if train/test data exists", {
            "type": "object", "properties": {}, "required": []
        }),
        ToolSpec("recent_run", "Show latest training results", {
            "type": "object", "properties": {}, "required": []
        }),
        # ... 其他工具
    ]
    
    messages = [
        {"role": "user", "content": _SCIENTIST_SYSTEM + "\n\n" + _rich_context(session) + "\n\n" + user_text}
    ]
    
    max_rounds = 3
    for _ in range(max_rounds):
        turn = client.send(messages, system="", tools=terminal_tool_specs)
        messages.append({"role": "assistant", "content": turn.raw_content})
        
        if not turn.wants_tool:
            return turn.text  # LLM 给出了最终回答
        
        # 执行工具
        tool_results = []
        for call in turn.tool_calls:
            result = self._execute_terminal_tool(call.name, session)
            tool_results.append(ToolResult(call.id, result, is_error=not result.get("ok")).to_wire())
        
        messages.append({"role": "user", "content": tool_results})
    
    # 超过最大轮数，强制 LLM 总结
    turn = client.send(messages + [
        {"role": "user", "content": "Summarize concisely based on the tool results above."}
    ], system="", tools=[])
    return turn.text
```

### 恢复钩子集成骨架:

```python
# 在 TerminalAgent.handle() 中
def handle(self, text, session, root):
    # 1. 开始钩子
    self._guard.emit(session, event="UserPromptSubmit")
    
    # 2. 处理用户输入
    result = self._dispatch(text, session, root)
    
    # 3. 记录工具账本
    self._ledger.record(result.action, result.summary, ok=(result.rc == 0))
    
    # 4. 结束钩子
    self._guard.emit(session, event="PostReply")
    
    return result
```

═══════════════════════════════════════════════════════════════
具体实施顺序
═══════════════════════════════════════════════════════════════

1. 先修 5 个 P0 bug（意图分类、任务切换、项目检测、恢复钩子、会话状态）
2. 再实现真正的 tool-use loop（用 AgentMessageClient 替代正则解析）
3. 然后集成中间件层（上下文救援 + 流预冲刷）
4. 最后跑完整测试套件，确保 56+ 测试全部通过
5. 用 evomind 终端手动测试所有关键场景
```

## 参考开源资源（已调研）

| 项目 | 用途 | URL |
|------|------|-----|
| Anthropic Cookbook | Tool-use 标准实现 | github.com/anthropics/anthropic-cookbook |
| Aider | LLM ↔ 工具循环模式 | github.com/paul-gauthier/aider |
| Claude Code 逆向分析 | Router/Hooks 架构 | 本项目的 docs/EvoMind-ClaudeCode-Distillation-20260706.md |
| OpenHands | Agent 执行环境隔离 | github.com/All-Hands-AI/OpenHands |

## 当前系统已有、可直接复用的基础设施

- AgentMessageClient (src/research_os/agent/messaging.py) — 真正的 tool-use 客户端
- AgentSession (src/research_os/agent/session.py) — 研究级的 tool-use loop
- ResearchToolbox (src/research_os/agent/tools.py) — 13 个研究工具
- TerminalTools (src/xsci/terminal_tools.py) — 11 个终端工具
- RecoveryGuard (src/xsci/recovery_guard.py) — 持久化恢复
- ToolLedger (src/xsci/tool_ledger.py) — 工具调用账本
- ContextRescue (src/xsci/context_rescue.py) — 上下文修剪
- EvolutionLoop (src/research_os/evolution_loop.py) — 完整训练循环
- MCGSSelector (src/research_os/mcgs_selector.py) — 搜索决策

**不需要从头写任何东西——需要的是正确的集成。**
