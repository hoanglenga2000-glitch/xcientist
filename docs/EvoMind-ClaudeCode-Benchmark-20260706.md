# EvoMind vs Claude Code — Capability Comparison

## Phase 1 Results (2026-07-06)

### Architecture Layer Comparison

| Capability | Claude Code Opus 4.8 | EvoMind (after this upgrade) |
|-----------|----------------------|------------------------------|
| **Tool-loop interaction** | ✅ Full Anthropic tool-use with AgentSession | ✅ TerminalAgent + ResearchToolbox + 11 terminal tools |
| **Streaming visibility** | ✅ SSE preflush, thinking display, keepalive | ✅ 6-stage preflight + 9-stage research narrative + StageRenderer |
| **Recovery after compaction** | ✅ Context guard hook writes on 6 lifecycle events | ✅ RecoveryGuard writes on each turn + PostToolCall |
| **Persistent memory** | ✅ Router writes context guard to .md file | ✅ RecoveryGuard writes to .xsci/recovery_guard.md |
| **Tool call ledger** | ✅ messages.jsonl for crash recovery | ✅ tool_ledger.jsonl in workspace |
| **Auto context rescue** | ✅ Drops oldest messages at byte threshold | ✅ context_rescue.py — deterministic trimmer |
| **Model fallback** | ✅ Opus → Sonnet → Haiku chain | ⬜ Not yet (single model) |
| **Prompt cache** | ✅ Stable/volatile split, anchor, single breakpoint | ⬜ Not applicable (no API proxy layer) |
| **Git-aware recovery** | ✅ git status in every guard section | ✅ RecoveryGuard includes git status |
| **Deterministic gate** | ✅ buildFableAuthGateNotice fast-fail | ✅ blocking_setup() → exact blocker message |
| **Multi-profile** | ✅ core/browser/figma/reverse/full | ✅ ToolProfile in launcher (core/full via MCP) |
| **Dashboard sync** | ✅ events.jsonl fan-out | ✅ events.jsonl + tool_ledger.jsonl |

### Interaction Pattern Comparison

| User Action | Claude Code | EvoMind |
|------------|-------------|---------|
| "What model are you using?" | Reads ANTHROPIC_MODEL env → deterministic answer | TerminalAgent → model_status tool → structured dict |
| "What tools do you have?" | Lists via system prompt + tool definitions | tool_status tool → lists 11 terminal tools |
| "Check the data" | Calls inspect_data (research_os tool) | data_check tool → checks for train.csv/test.csv |
| "Start training with local compute" | Enters AgentSession with compute=local | Preflight (6 stages) → _run_agent(compute=local) |
| "Use GPU" (blocked) | Fast-fail with auth gate notice | Preflight → blocking_setup() → repair suggestion |
| "Resume last experiment" | Restores search_graph.json + message ledger | _run_agent(resume=True) + AgentSession message replay |
| After crash/compaction | Context guard recovery → no goal loss | RecoveryGuard → rebuilds state from recovery_guard.md |

### Quantitative Metrics

| Metric | Claude Code | EvoMind Target |
|--------|------------|----------------|
| Tool categories | 10+ (Bash, Read, Write, Edit, Glob, Grep, etc.) | 11 terminal tools + 13 research_os tools |
| Recovery events tracked | 6 lifecycle hooks | 2 hooks (UserPromptSubmit, PostToolCall) |
| Context rescue target | 680KB auto target | 680KB auto target (same algorithm) |
| Stream preflush latency | < 50ms to first SSE comment | Immediate stdout on preflight |
| Tool ledger retention | 100+ messages | Unlimited (append-only JSONL) |

### Key Differentiators (Where EvoMind is Still Not Claude Code)

1. **No browser/figma tools**: Claude Code has MCP servers for browser automation, Figma. EvoMind is ML-research focused.
2. **No prompt cache**: Claude Code's router splits system blocks for Anthropic's cache API. EvoMind doesn't need this (uses local LLM or general API).
3. **No multi-tenant gateway**: Claude Code's router handles multiple models + fallback chains. EvoMind is single-model.
4. **No subagent spawning**: Claude Code can spawn isolated subagents. EvoMind has audit subagent but not general subagent orchestration.
5. **Human gate is stronger**: Claude Code can auto-approve Bash commands in bypassPermissions mode. EvoMind's Kaggle submit is ALWAYS human-gated.

### What Makes EvoMind Comparable Now

1. ✅ Streaming preflight stages (6 phases before training)
2. ✅ Deterministic tool queries (model status, task list, data check)
3. ✅ Persistent recovery guard (survives compaction/restart)
4. ✅ Tool call ledger (crash recovery + dashboard timeline)
5. ✅ Auto context rescue (deterministic trimming)
6. ✅ Git-aware session state
7. ✅ Events.jsonl dashboard sync
8. ✅ LLM-driven tool-loop (when LLM available) + deterministic fallback
9. ✅ Compute routing (local vs GPU) with blocking gate
10. ✅ Chinese + English intent classification
