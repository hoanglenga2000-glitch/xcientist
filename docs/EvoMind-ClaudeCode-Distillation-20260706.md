# Claude Code Opus 4.8 Architecture — Distillation for EvoMind

## Patterns Identified

### P1: Middleware Stack
```
PowerShell Launcher → Node.js Router → Anthropic API
     (env,ports)      (cache,rewrite,persist)   (ai.lt4net.org)
```
EvoMind equivalent: `evomind → TerminalAgent → Router/API`

### P2: Persistent Recovery (Context Guard)
Every hook event writes session state to a `.md` file. After compaction: recover goal, files, 
constraints, git status from this file before asking user to restate.

### P3: Stream Preflush
Open SSE before upstream responds. Show keepalive comments. User sees "in progress" immediately.

### P4: Automatic Context Rescue
When request body exceeds byte threshold: drop oldest messages, keep N tail messages.
Deterministic algorithm — no LLM needed.

### P5: Prompt Cache Optimization
Split system blocks into stable (cached) and volatile (appended to latest user message).
Single stable breakpoint. Always-anchored.

### P6: Upstream Resilience
Retry with jittered backoff. Model fallback chain. Client disconnect handling.
Timeout → graceful text response.

### P7: Tool Ledger
Every tool call + result persisted to messages.jsonl. Enables resume-after-crash.
The search_graph is the audited source of truth.

### P8: Lifecycle Hooks
SessionStart, UserPromptSubmit, PreCompact, PostCompact, Stop, StopFailure.
Each writes recovery context and git status.

## What EvoMind Already Has
- Session persistence (session.json) → partial match for P2/P8
- Tool system (terminal_tools + ResearchToolbox) → P7
- StageRenderer streaming → P3
- Gate system → P6

## What EvoMind Needs to Implement
1. Recovery hooks (P2/P8) — write recovery context on each user turn
2. Auto context rescue (P4) — trim conversation when it grows too large  
3. Tool ledger (P7) — persist tool calls for learning
4. Stream preflush during agent runs (P3) — visual feedback immediately
5. Compaction recovery block (P2) — survive Claude Code-like compaction events
