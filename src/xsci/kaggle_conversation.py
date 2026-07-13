r"""Conversational brain for the EvoMind research terminal — SCIENTIST MODE.

Multi-turn conversation with tool-use reasoning, proactive data analysis,
and experiment suggestion — giving the terminal agent real ML research
scientist behavior, not just command-response.
"""
from __future__ import annotations

import json
import os
import re
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from .recovery_guard import RecoveryGuard

if TYPE_CHECKING:
    from .kaggle_session import SessionState

MAX_HISTORY = 20


def _load_history() -> list[dict[str, Any]]:
    from .config import GLOBAL_DIR
    path = GLOBAL_DIR / "conversation_history.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data[-MAX_HISTORY:]
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return []


def _save_history(messages: list[dict[str, Any]]) -> None:
    from .config import GLOBAL_DIR
    path = GLOBAL_DIR / "conversation_history.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(messages[-MAX_HISTORY:], ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


# ═══════════════════════════════════════════════════════════════════════
# SCIENTIST PERSONA — the system prompt that makes EvoMind think like a
# real ML researcher, not a chatbot.
# ═══════════════════════════════════════════════════════════════════════

_SCIENTIST_SYSTEM = textwrap.dedent("""\
你是一位资深的机器学习研究科学家，工作在一个名为 EvoMind 的 AI 科研终端中。
你不是聊天机器人——你的职责是深入理解数据、形成可验证的假说、设计实验、
并在配置就绪时启动可审计的训练。

你的思维方式：
1. 先观察（Observe）—— 查看数据长什么样、有什么特征、缺失值情况、分布特点
2. 再分析（Analyze）—— 这个任务是什么类型？metric 的方向是什么？数据的挑战在哪？
3. 然后提议（Propose）—— 基于分析提出 2-3 个可操作的改进方向
4. 最后执行（Execute）—— 如果用户同意且门禁通过，启动训练

你可以调用的终端工具（结果已注入到你的上下文中）：
- 模型状态（当前使用的 LLM provider/model）
- 任务列表（已注册的比赛）
- 任务详情（modality, metric, schema, 数据目录）
- 数据可用性（train.csv/test.csv 是否存在）
- 最近训练结果（best CV, promotions）
- GPU/HPC 状态
- Kaggle 配置状态
- 下一步建议

RULES（硬性规则）:
- 用用户的语言回复（中文用户用中文，英文用户用英文）
- 绝对不要虚构 Kaggle 分数、排名、奖牌 —— 除非有真实的 Kaggle response artifact
- 绝对不要读取或打印 API key, Kaggle token, SSH 密码
- 官方 Kaggle 提交必须 human gate —— 永远不能自动提交
- 如果被问到模型/数据/任务状态，基于已注入的工具结果回答，不要瞎编
- 训练前的门禁必须全部通过（LLM key, task selected, data available）
- 不要过度承诺 —— 对不确定的事情说"让我检查一下"而不是猜测
- 当数据可用时，主动建议可行的下一步实验方向
- 当训练完成后，客观分析结果并提出改进建议
- 像个真正的科学家：诚实、严谨、好奇、有洞察力
""")

# ── Tool suggestion patterns: the LLM can indicate which tool it wants
# results for by writing a special marker. We parse this, execute the tool,
# and feed the result back for a second round of reasoning. ──

_TOOL_HINT_RE = re.compile(
    r'\[(?:tool|check|检查|查看):\s*(model_status|system_status|task_list|inspect_task|'
    r'data_check|recent_run|gpu_status|kaggle_status|dashboard|next_steps|'
    r'evolution_status|scientist_checkpoint|research_decision|scientist_workplan|scientist_turn_plan|scientist_repair_plan|scientist_execution_contract|scientist_step_trace|scientist_recovery|scientist_action_queue|scientist_next_action|scientist_autopilot|scientist_loop|scientist_self_audit|scientist_upgrade_plan|scientist_self_upgrade_loop|scientist_memory_consolidation|scientist_innovation_backlog|scientist_hypothesis_review|scientist_experiment_blueprint|scientist_situation_model|switch_task)]',
    re.IGNORECASE,
)


def _forced_tool_hints(user: str) -> list[str]:
    """Return read-only tools that should run before broad AI Scientist answers."""
    text = (user or "").lower()
    turn_plan_tokens = (
        "turn plan",
        "tool plan",
        "per-turn plan",
        "plan this turn",
        "plan your tools",
        "what tools will you use",
        "before you answer, plan",
        "本轮计划",
        "工具计划",
        "行动计划",
        "你准备调用什么工具",
        "先规划本轮",
        "本次回合",
    )
    if any(token in text for token in turn_plan_tokens):
        return ["scientist_turn_plan"]
    innovation_tokens = (
        "innovation backlog",
        "innovate plan",
        "innovation plan",
        "innovation hypothesis",
        "innovation hypotheses",
        "research hypotheses",
        "memory guided innovation",
        "memory-guided innovation",
        "novel branch",
        "novel combination",
        "propose innovation",
        "generate innovation",
        "generate hypotheses",
        "创新假设",
        "创新计划",
        "创新分支",
        "生成创新",
        "生成假设",
        "根据记忆创新",
        "复用记忆",
        "记忆复用",
        "跨任务创新",
    )
    if any(token in text for token in innovation_tokens):
        return ["scientist_innovation_backlog"]
    review_tokens = (
        "review hypotheses",
        "review hypothesis",
        "hypothesis review",
        "rank hypotheses",
        "rank hypothesis",
        "critique hypotheses",
        "critique hypothesis",
        "score hypotheses",
        "proposal review",
        "review proposals",
        "rank proposals",
        "评审假设",
        "假设评审",
        "假设排序",
        "排序假设",
        "评估假设",
        "最佳假设",
        "评审方案",
        "排序方案",
    )
    if any(token in text for token in review_tokens):
        return ["scientist_hypothesis_review"]
    blueprint_tokens = (
        "experiment blueprint",
        "candidate blueprint",
        "execution blueprint",
        "plan experiment",
        "gated experiment plan",
        "实验蓝图",
        "执行蓝图",
        "实验方案",
        "执行方案",
        "生成实验蓝图",
        "生成实验计划",
        "把假设落地",
        "可执行实验",
        "实验设计",
    )
    if any(token in text for token in blueprint_tokens):
        return ["scientist_experiment_blueprint"]
    situation_tokens = (
        "situation model",
        "scientist situation",
        "state model",
        "current situation",
        "research situation",
        "orient",
        "why are we blocked",
        "what should the scientist do next",
        "analyze the current situation",
        "scientist state",
        "局势",
        "情境",
        "态势",
        "当前状态模型",
        "科学家状态",
        "现在局面",
        "现在卡在哪里",
        "为什么卡住",
        "下一步判断",
        "综合证据",
    )
    if any(token in text for token in situation_tokens):
        return ["scientist_situation_model"]
    self_upgrade_tokens = (
        "self-upgrade loop",
        "self upgrade loop",
        "upgrade loop",
        "capability work order",
        "self-upgrade work order",
        "execute self-upgrade",
        "run self-upgrade",
        "自升级闭环",
        "自我升级闭环",
        "能力自升级",
        "生成自升级工单",
        "创建自升级工单",
        "能力缺口转成工单",
        "把 p0 能力缺口转成工单",
        "自进化工程工单",
    )
    if any(token in text for token in self_upgrade_tokens):
        return ["scientist_self_upgrade_loop"]
    upgrade_tokens = (
        "upgrade plan",
        "upgrade backlog",
        "self upgrade",
        "agent upgrade",
        "capability upgrade",
        "close upgrade backlog",
        "engineering plan",
        "升级计划",
        "能力升级计划",
        "系统升级计划",
        "自我升级",
        "修复升级项",
        "升级 backlog",
        "修复 backlog",
        "工程升级计划",
    )
    if any(token in text for token in upgrade_tokens):
        return ["scientist_upgrade_plan"]
    self_audit_tokens = (
        "self audit",
        "self-audit",
        "capability audit",
        "agent audit",
        "agent capability",
        "intelligence audit",
        "how close to claude code",
        "what is missing from claude code",
        "自我审计",
        "能力审计",
        "能力评估",
        "智能度评估",
        "系统能力差距",
        "agent 能力",
        "和 claude code 差距",
        "像 claude code 还差什么",
        "像 codex 还差什么",
    )
    if any(token in text for token in self_audit_tokens):
        return ["scientist_self_audit"]
    loop_tokens = (
        "scientist loop",
        "agent loop",
        "autonomous loop",
        "像claude code一样",
        "像 claude code 一样",
        "像codex一样",
        "像 codex 一样",
        "持续优化",
        "继续优化",
        "自动推进",
        "自主循环",
        "自主回合",
        "多步回合",
        "连续诊断",
    )
    if any(token in text for token in loop_tokens):
        return ["scientist_loop"]
    recovery_tokens = (
        "recovery snapshot",
        "recovery guard",
        "recover context",
        "resume context",
        "compaction recovery",
        "restart recovery",
        "恢复现场",
        "恢复状态",
        "恢复上下文",
        "上下文恢复",
        "上下文丢了",
        "断点恢复",
        "重启后恢复",
        "从哪里继续",
    )
    if any(token in text for token in recovery_tokens):
        return ["scientist_recovery"]
    next_action_tokens = (
        "next action",
        "safe next",
        "act next",
        "安全下一步",
        "执行安全下一步",
        "推进下一步",
        "继续行动",
        "下一步行动",
    )
    if any(token in text for token in next_action_tokens):
        return ["scientist_next_action"]
    action_queue_tokens = (
        "action queue",
        "queue",
        "行动队列",
        "动作队列",
        "下一步队列",
        "下一步命令",
    )
    if any(token in text for token in action_queue_tokens):
        return ["scientist_action_queue"]
    workplan_tokens = (
        "workplan",
        "roadmap",
        "agenda",
        "multi-step plan",
        "工作计划",
        "执行计划",
        "路线图",
        "多步计划",
        "拆解步骤",
        "持续推进",
    )
    if any(token in text for token in workplan_tokens):
        return ["scientist_workplan"]
    repair_tokens = (
        "repair plan",
        "fix plan",
        "self repair",
        "root cause",
        "why blocked",
        "修复计划",
        "自我修复",
        "自修复",
        "怎么修",
        "如何修复",
        "哪里卡住",
        "卡在哪里",
        "阻塞原因",
        "失败归因",
        "修复路线",
    )
    if any(token in text for token in repair_tokens):
        return ["scientist_repair_plan"]
    contract_tokens = (
        "execution contract",
        "run contract",
        "pre-execution",
        "preflight contract",
        "执行合同",
        "执行契约",
        "执行前检查",
        "运行前检查",
        "开跑前检查",
        "能不能跑",
        "可以训练吗",
        "可以开跑吗",
        "训练合同",
    )
    if any(token in text for token in contract_tokens):
        return ["scientist_execution_contract"]
    trace_tokens = (
        "step trace",
        "steptrace",
        "trace",
        "tool trace",
        "步骤轨迹",
        "运行轨迹",
        "工具轨迹",
        "工具调用过程",
        "执行证据流",
    )
    if any(token in text for token in trace_tokens):
        return ["scientist_step_trace"]
    autopilot_tokens = (
        "autopilot",
        "diagnose",
        "diagnosis",
        "not smart",
        "complex problem",
        "ai scientist",
        "what should we do next",
        "全面诊断",
        "自动诊断",
        "主动分析",
        "自主分析",
        "不够智能",
        "下一步",
        "科学家",
        "复杂问题",
    )
    if any(token in text for token in autopilot_tokens):
        return ["scientist_autopilot"]
    return []


# ── Helper: build a rich context block ───────────────────────────────

def _rich_context(session: "SessionState") -> str:
    """Build a research-state context block for the LLM.

    Includes: task info, data status, model status, recent results,
    GPU/Kaggle readiness, and gate status — everything the scientist
    needs to reason about the current state.
    """
    lines = ["[CURRENT RESEARCH STATE]"]

    # Task
    if session.selected_task:
        lines.append(f"Selected task: {session.selected_task}")
        if session.task_brief:
            lines.append(f"  {session.task_brief}")
        lines.append(f"  Tasks registered: {session.n_tasks}")
    else:
        lines.append("Selected task: (none)")
        lines.append(f"  Tasks registered: {session.n_tasks}")

    # Scan ALL experiment results (not just selected task)
    all_runs = _scan_all_experiment_results(session)
    if all_runs:
        lines.append("ALL KNOWN EXPERIMENT RESULTS (across all tasks):")
        for run_info in all_runs[:12]:
            lines.append(f"  {run_info}")
    elif session.recent_run_id:
        lines.append(f"Recent run: {session.recent_run_id}"
                     + (f", best CV={session.recent_best_cv:.4f}" if session.recent_best_cv is not None else ""))

    # Data
    lines.append(f"Data status: {'kaggle ready' if session.kaggle_ready else 'kaggle not configured'}")

    # LLM
    lines.append(f"LLM: {session.llm_provider} — {'ready' if session.llm_ready else 'setup needed'}")

    # Compute
    lines.append(f"Compute: default={session.compute_backend}"
                 + (f", override={session.current_compute_override}" if getattr(session, 'current_compute_override', '') else ""))

    # GPU
    if session.gpu_ready:
        if session.gpu_blocked:
            lines.append(f"GPU: configured but BLOCKED — {session.gpu_blocker or session.gpu_status}")
        else:
            lines.append("GPU: configured and available")
    else:
        lines.append("GPU: not configured")

    if session.memory_summary:
        lines.append(f"Memory: {session.memory_summary}")

    # Gaps
    gaps = session.missing_setup()
    if gaps:
        heads = [g.split(":", 1)[0] for g in gaps]
        lines.append(f"Setup gaps: {', '.join(heads)}")

    return "\n".join(lines)


def _format_turn_plan_context(plan: dict[str, Any] | None) -> str:
    """Compact prompt block for the latest per-turn Scientist plan."""
    if not isinstance(plan, dict):
        return ""
    intent = plan.get("intent") if isinstance(plan.get("intent"), dict) else {}
    readiness = plan.get("readiness") if isinstance(plan.get("readiness"), dict) else {}
    lines = [
        "",
        "[AI SCIENTIST TURN PLAN]",
        f"Intent: {intent.get('kind') or 'unknown'}"
        + (f" payload={intent.get('payload')}" if intent.get("payload") else ""),
        f"Autonomy: {plan.get('autonomy_level') or 'unknown'}",
        f"Can execute: {readiness.get('can_execute')}",
    ]
    blockers = readiness.get("blocking_gates") if isinstance(readiness, dict) else []
    if blockers:
        lines.append("Blocking gates:")
        lines.extend(f"  - {item}" for item in blockers[:4])
    tools = plan.get("selected_tools") or []
    if tools:
        lines.append("Selected tools:")
        for item in tools[:6]:
            if isinstance(item, dict):
                lines.append(
                    f"  - {item.get('tool')} "
                    f"(confidence={item.get('confidence')}, gate={item.get('gate')}): "
                    f"{item.get('why')}"
                )
    stops = plan.get("stop_conditions") or []
    if stops:
        lines.append("Stop conditions:")
        lines.extend(f"  - {item}" for item in stops[:4])
    if plan.get("next_safe_command"):
        lines.append(f"Next safe command: {plan.get('next_safe_command')}")
    if plan.get("artifact_path"):
        lines.append(f"Turn-plan artifact: {plan.get('artifact_path')}")
    lines.append("No training or official Kaggle submit may start from this plan alone.")
    return "\n".join(lines)


def _scan_all_experiment_results(session: "SessionState") -> list[str]:
    """Scan ALL experiment directories for results, not just the selected task."""
    import json
    from pathlib import Path
    results = []
    exp_base = Path(session.workspace_root) / "experiments" / "evolution"
    if not exp_base.is_dir():
        return results
    try:
        for run_dir in sorted(exp_base.iterdir(), key=lambda d: d.name, reverse=True):
            if not run_dir.is_dir():
                continue
            summary = run_dir / "summary.json"
            if not summary.exists():
                continue
            try:
                data = json.loads(summary.read_text(encoding="utf-8"))
                task = data.get("task", "?")
                best = data.get("best_exp_id", "")
                cv = data.get("best_cv_score")
                promos = data.get("n_promotions", 0)
                iters = data.get("n_iterations", 0)
                cv_str = f"{cv:.4f}" if isinstance(cv, (int, float)) and cv is not None else "N/A"
                results.append(f"{task}: best={best} CV={cv_str} promotions={promos}/{iters}")
            except (json.JSONDecodeError, OSError):
                continue
    except OSError:
        pass
    return results


def _execute_terminal_tool(name: str, session: "SessionState") -> str:
    """Execute a terminal tool and return a formatted result string."""
    from .tasks import list_tasks, resolve_task
    from .terminal_tools import TerminalTools

    root = Path(session.workspace_root) if session.workspace_root else Path.cwd()

    # ── Special tool: switch_task ──────────────────────────────────
    if name == "switch_task":
        # Find a task by fuzzy match from registered tasks
        tasks = list_tasks(root)
        # Try to match from the user's request / context
        best_match = None
        user_mention = str(session.last_goal or "").lower()
        for slug, _ in tasks:
            if slug in user_mention or slug.replace("-", " ").replace("_", " ") in user_mention:
                best_match = slug
                break
        if best_match:
            try:
                resolve_task(best_match, project_root=root)
                session.selected_task = best_match
                session.refresh_task_brief(root)
                session.persist()
                return f"[TOOL RESULT: switch_task]\nstatus: OK\nswitched to: {best_match}\nbrief: {session.task_brief}"
            except FileNotFoundError:
                return f"[TOOL RESULT: switch_task]\nstatus: FAILED\nmessage: task '{best_match}' cannot be resolved"
        return f"[TOOL RESULT: switch_task]\nstatus: FAILED\nmessage: no matching task found in: {[s for s,_ in tasks]}"

    result = TerminalTools.dispatch(name, session, root)
    try:
        from .tool_ledger import ToolLedger
        summary = str(result.get("message") or result.get("mode") or "")[:240]
        ToolLedger(root).record(name, result, ok=bool(result.get("ok", True)), summary=summary)
    except Exception:
        pass

    lines = [f"[TOOL RESULT: {name}]"]
    ok = result.get("ok", True)
    lines.append(f"status: {'OK' if ok else 'BLOCKED/FAILED'}")
    for key, value in result.items():
        if key in ("ok", "tool"):
            continue
        if isinstance(value, list):
            if not value:
                lines.append(f"{key}: (empty)")
            else:
                lines.append(f"{key}:")
                for item in value[:10]:
                    if isinstance(item, dict):
                        parts = [f"{k}={v}" for k, v in item.items()
                                if k != "path" and not isinstance(v, (dict, list))]
                        lines.append(f"  - {', '.join(parts)}")
                    else:
                        lines.append(f"  - {item}")
        elif isinstance(value, dict):
            lines.append(f"{key}:")
            for k, v in value.items():
                if not isinstance(v, (dict, list)):
                    lines.append(f"  {k}: {v}")
        elif value not in (None, ""):
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _terminal_tool_specs():
    """Anthropic-native tool specs for the terminal tool-use loop (Plan B)."""
    from research_os.agent.messaging import ToolSpec
    no_args = {"type": "object", "properties": {}, "required": []}
    return [
        ToolSpec("model_status", "Current LLM provider/model/readiness (never the key).", no_args),
        ToolSpec("system_status", "Full readiness: LLM, Kaggle, GPU, tasks, recent run.", no_args),
        ToolSpec("task_list", "List all registered competitions/tasks.", no_args),
        ToolSpec("inspect_task", "Details of the selected task (modality/metric/schema).", no_args),
        ToolSpec("data_check", "Whether train/test/sample_submission CSVs exist.", no_args),
        ToolSpec("recent_run", "Latest training run id + best CV.", no_args),
        ToolSpec("gpu_status", "GPU/HPC config + manifest blocker status.", no_args),
        ToolSpec("kaggle_status", "Kaggle API configuration status.", no_args),
        ToolSpec("next_steps", "Blocking gates + the suggested next action.", no_args),
        ToolSpec("evolution_status",
                 "Durable self-evolution evidence: tracker, memory, innovation logs.",
                 no_args),
        ToolSpec("scientist_checkpoint",
                 "Structured Observe/Analyze/Propose/Gate/Act research checkpoint.",
                 no_args),
        ToolSpec("research_decision",
                 "Persisted next experiment decision: branch, code mode, gates, rollback.",
                 no_args),
        ToolSpec("scientist_workplan",
                 "Recoverable multi-step AI Scientist workplan: steps, gates, evidence, focus, and resume commands. Read-only.",
                 no_args),
        ToolSpec("scientist_turn_plan",
                 "Per-turn AI Scientist control plan: intent, selected tools, rationale, gates, expected artifacts, and stop conditions. Read-only.",
                 no_args),
        ToolSpec("scientist_repair_plan",
                 "Read-only self-repair plan: diagnoses blockers, root causes, repair steps, and safe next command. Never trains.",
                 no_args),
        ToolSpec("scientist_execution_contract",
                 "Read-only pre-execution contract: go/no-go, branch, rollback, required artifacts, and claim boundary. Never trains.",
                 no_args),
        ToolSpec("scientist_step_trace",
                 "Recent step-level AI Scientist event stream: tool calls, gates, artifacts, blockers, and no-training boundary. Read-only.",
                 no_args),
        ToolSpec("scientist_recovery",
                 "Long-horizon recovery snapshot: recovery guard, turn ledger, step trace, latest plans, blockers, and resume commands. Read-only.",
                 no_args),
        ToolSpec("scientist_action_queue",
                 "Read-only action queue: next command, gate, autonomy, risk, expected artifacts, and rollback.",
                 no_args),
        ToolSpec("scientist_next_action",
                 "Executes only the next safe read-only action from the queue; blocks at training/download/submit/user gates.",
                 no_args),
        ToolSpec("scientist_autopilot",
                 "Bounded multi-tool AI Scientist diagnosis chain: status, task, data, recent run, memory, gates, and next decision. Read-only.",
                 no_args),
        ToolSpec("scientist_loop",
                 "Bounded autonomous safe loop: run diagnosis, execute only read-only next actions, stop at gates, and write reusable lessons. Never trains or submits.",
                 no_args),
        ToolSpec("scientist_self_audit",
                 "Read-only capability audit of EvoMind itself: scores, gaps, evidence sources, and system-upgrade backlog. Never trains or submits.",
                 no_args),
        ToolSpec("scientist_upgrade_plan",
                 "Read-only engineering planner for the self-audit upgrade backlog: files to inspect, acceptance checks, closure gates, and safe next commands. Never edits, trains, or submits.",
                 no_args),
        ToolSpec("scientist_self_upgrade_loop",
                 "Safe self-upgrade bridge: selects the highest-priority capability backlog item and writes a code-agent work order, action queue, trace, and lesson. Never edits source code, trains, downloads, or submits.",
                 no_args),
        ToolSpec("scientist_memory_consolidation",
                 "Read-only memory writeback: consolidates Scientist loop, trace, contracts, and lessons into retrospective memory. Never trains or submits.",
                 no_args),
        ToolSpec("scientist_innovation_backlog",
                 "Read-only memory-guided innovation planner: proposes auditable branches, risk controls, artifacts, and gates before training. Never trains or submits.",
                 no_args),
        ToolSpec("scientist_hypothesis_review",
                 "Read-only Scientist review board: scores and ranks innovation hypotheses by evidence, readiness, impact, risk, and gates. Never trains or submits.",
                 no_args),
        ToolSpec("scientist_experiment_blueprint",
                 "Read-only experiment blueprint builder: turns the reviewed hypothesis into branch/code/resource/artifact/rollback/memory-writeback gates. Never trains or submits.",
                 no_args),
        ToolSpec("scientist_situation_model",
                 "Read-only situation model: synthesizes evidence, uncertainty, blockers, strategy, memory, and the next safe tool sequence. Never trains or submits.",
                 no_args),
        ToolSpec("switch_task", "Switch the selected task to a registered slug.",
                 {"type": "object",
                  "properties": {"task": {"type": "string", "description": "slug to switch to"}},
                  "required": ["task"]}),
    ]


def _format_tool_result(name: str, result: dict[str, Any]) -> tuple[str, bool]:
    """Render a terminal-tool dict as compact text + an ok flag (for tool_result)."""
    ok = bool(result.get("ok", True))
    lines = [f"[{name}] status={'OK' if ok else 'FAILED'}"]
    for key, value in result.items():
        if key in ("ok", "tool"):
            continue
        if isinstance(value, list):
            lines.append(f"{key}: " + (", ".join(str(v) for v in value[:10]) if value else "(empty)"))
        elif isinstance(value, dict):
            lines.append(f"{key}: " + ", ".join(f"{k}={v}" for k, v in value.items()
                                                 if not isinstance(v, (dict, list))))
        elif value not in (None, ""):
            lines.append(f"{key}: {value}")
    return "\n".join(lines), ok


def _execute_agent_tool_call(name: str, tool_input: dict[str, Any],
                             session: "SessionState") -> tuple[str, bool]:
    """Execute one Anthropic tool_use call from the real loop → (result_text, ok).

    Unlike ``_execute_terminal_tool`` (which fuzzy-matches switch targets from the
    last goal), this honours an EXPLICIT ``task`` argument the model supplied.
    """
    from .tasks import list_tasks, resolve_task
    from .terminal_tools import TerminalTools

    root = Path(session.workspace_root) if session.workspace_root else Path.cwd()

    if name == "switch_task":
        want = str(tool_input.get("task", "")).strip()
        norm = want.lower().replace(" ", "").replace("-", "").replace("_", "")
        target = None
        for slug, _ in list_tasks(root):
            s = slug.lower()
            if s == want.lower() or s.replace("-", "").replace("_", "") == norm or (norm and norm in s.replace("-", "").replace("_", "")):
                target = slug
                break
        if not target:
            names = [s for s, _ in list_tasks(root)]
            return (f"[switch_task] status=FAILED — no task matching '{want}'. Registered: {names}", False)
        try:
            resolve_task(target, project_root=root)
        except FileNotFoundError:
            return (f"[switch_task] status=FAILED — cannot resolve '{target}'", False)
        session.selected_task = target
        session.refresh_task_brief(root)
        session.refresh_recent_run(root)
        session.persist(root)
        return (f"[switch_task] status=OK switched_to={target} brief={session.task_brief}", True)

    result = TerminalTools.dispatch(name, session, root)
    return _format_tool_result(name, result)


# ═══════════════════════════════════════════════════════════════════════
# ConversationAgent — the research scientist brain
# ═══════════════════════════════════════════════════════════════════════

class ConversationAgent:
    """LLM-driven research scientist with tool-use reasoning.

    Architecture:
      1. If LLM available: send scientist system prompt + rich context → LLM reasons
      2. Parse LLM response for tool hints → execute tools → feed results back
      3. LLM synthesizes final natural-language response
      4. If LLM unavailable: deterministic scientist-style analysis using task brief,
         data status, and recent results.
    """

    def __init__(self, *, client=None) -> None:
        self._client = client
        self._resolved = client is not None
        self._max_tool_rounds = 2  # max tool-execution rounds per turn

    def _get_client(self):
        if not self._resolved:
            try:
                from research_os.llm_client import LLMClient
                self._client = LLMClient()
            except Exception:
                self._client = None
            self._resolved = True
        return self._client

    def _llm_available(self, session: "SessionState") -> bool:
        if not session.llm_ready:
            return False
        client = self._get_client()
        return client is not None

    # ── Main entry ───────────────────────────────────────────────────

    def _make_guard(self, session: "SessionState"):
        """Bind a RecoveryGuard to the workspace recovery file, or None on failure."""
        try:
            ws = session.workspace_root or str(Path.cwd())
            guard = RecoveryGuard()
            guard.set_state_file(Path(ws) / ".xsci" / "recovery_guard.md")
            return guard
        except Exception:
            return None

    def reply(self, text: str, session: "SessionState") -> str:
        """Reply to user input with scientist-quality analysis.

        Plan D: emit a recovery-guard section before and after the turn so the
        LLM conversation path (not just deterministic tool queries) leaves a
        durable anchor for compaction/restart recovery.
        """
        guard = self._make_guard(session)
        answer = ""
        route = "rule"
        forced_tools = _forced_tool_hints(text)
        turn_plan: dict[str, Any] | None = None
        try:
            from .scientist_turn_planner import build_scientist_turn_plan

            root = Path(session.workspace_root) if session.workspace_root else Path.cwd()
            turn_plan = build_scientist_turn_plan(session, root, text, persist=True)
        except Exception:
            turn_plan = None
        if guard is not None:
            guard.emit(session, event="UserPromptSubmit")
        try:
            if self._llm_available(session):
                history = _load_history()
                history.append({"role": "user", "content": text})

                answer = self._scientist_loop(session, text, history, turn_plan=turn_plan)
                if answer:
                    route = "llm_tool_loop"
                    history.append({"role": "assistant", "content": answer[:2000]})
                    _save_history(history)
                    return answer
            answer = self._rule_reply(text, session)
            return answer
        finally:
            self._record_turn(text, answer, session, route=route, forced_tools=forced_tools, turn_plan=turn_plan)
            if guard is not None:
                guard.record_tool(f"reply: task={session.selected_task or '(none)'}")
                guard.emit(session, event="PostReply")

    def _record_turn(self, user: str, answer: str, session: "SessionState", *,
                     route: str, forced_tools: list[str],
                     turn_plan: dict[str, Any] | None = None) -> None:
        """Persist a sanitized scientist turn for dashboard/recovery visibility."""
        try:
            from .scientist_turns import record_scientist_turn
            from .tool_ledger import ToolLedger

            root = Path(session.workspace_root) if session.workspace_root else Path.cwd()
            recent_tools = ToolLedger(root).recent(limit=8)
            latest_autopilot = root / ".xsci" / "scientist_autopilot.json"
            decision: dict[str, Any] = {}
            blockers: list[str] = []
            next_actions: list[str] = []
            mode = ""
            artifacts: list[str] = []
            if latest_autopilot.exists():
                try:
                    autopilot = json.loads(latest_autopilot.read_text(encoding="utf-8"))
                    if isinstance(autopilot, dict):
                        decision = autopilot.get("decision", {}) if isinstance(autopilot.get("decision"), dict) else {}
                        blockers = [str(x) for x in autopilot.get("blockers", [])] if isinstance(autopilot.get("blockers"), list) else []
                        next_actions = [str(x) for x in autopilot.get("next_actions", [])] if isinstance(autopilot.get("next_actions"), list) else []
                        mode = str(autopilot.get("mode") or "")
                        artifacts.append(str(latest_autopilot))
                except (json.JSONDecodeError, OSError):
                    pass
            if isinstance(turn_plan, dict):
                plan_artifact = str(turn_plan.get("artifact_path") or "")
                if plan_artifact:
                    artifacts.append(plan_artifact)
                plan_tools = [str(x) for x in (turn_plan.get("tool_sequence") or [])]
                forced_tools = list(dict.fromkeys([*forced_tools, "scientist_turn_plan", *plan_tools]))
                plan_readiness = turn_plan.get("readiness") if isinstance(turn_plan.get("readiness"), dict) else {}
                if plan_readiness and not blockers:
                    blockers = [str(x) for x in (plan_readiness.get("blocking_gates") or [])]
                if not next_actions and turn_plan.get("next_safe_command"):
                    next_actions = [str(turn_plan.get("next_safe_command"))]
                if not mode and turn_plan.get("autonomy_level"):
                    mode = str(turn_plan.get("autonomy_level"))
                if not decision:
                    decision = {
                        "intent": (turn_plan.get("intent") or {}).get("kind") if isinstance(turn_plan.get("intent"), dict) else "",
                        "next_safe_command": turn_plan.get("next_safe_command"),
                        "tool_sequence": plan_tools,
                    }
            record_scientist_turn(root, {
                "task": session.selected_task or "",
                "route": route,
                "user": user,
                "forced_tools": forced_tools,
                "executed_tools": recent_tools,
                "mode": mode,
                "decision": decision,
                "blockers": blockers,
                "next_actions": next_actions,
                "artifacts": artifacts,
                "answer_preview": answer,
                "no_training_started": True,
            })
        except Exception:
            pass

    # ── Scientist tool-use loop ──────────────────────────────────────

    def _real_tool_loop(self, session: "SessionState", user: str,
                        turn_plan: dict[str, Any] | None = None) -> str:
        """Plan B: a real Anthropic tool-use loop (send → tool_use → tool_result).

        Returns the model's final text, or "" to signal the caller to fall back
        to the two-pass text protocol (no Anthropic key, or a transport error).
        Only runs when Anthropic is the primary transport, because the loop feeds
        back Anthropic-native tool_result blocks. Plan C (auto context rescue) is
        applied before every send so an over-long history never hits the API.
        """
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return ""
        try:
            from research_os.agent.messaging import AgentMessageClient, ToolResult

            from .context_rescue import auto_rescue_context, build_context_rescue_system_block
            from .recovery_guard import build_compaction_recovery_block
            from .tool_ledger import ToolLedger
        except Exception:
            return ""

        client = AgentMessageClient()
        if not client.is_available():
            return ""

        specs = _terminal_tool_specs()
        root = Path(session.workspace_root) if session.workspace_root else Path.cwd()
        ledger = ToolLedger(root)

        system = _SCIENTIST_SYSTEM
        recovery = build_compaction_recovery_block(root / ".xsci" / "recovery_guard.md")
        if recovery:
            system += "\n\n" + recovery

        forced_results = ""
        for name in _forced_tool_hints(user):
            forced_results += _execute_terminal_tool(name, session) + "\n\n"

        messages = [{
            "role": "user",
            "content": (
                _rich_context(session)
                + _format_turn_plan_context(turn_plan)
                + (f"\n\n[REQUIRED PRECHECK]\n{forced_results}" if forced_results else "")
                + "\n\n[USER]\n"
                + user
            )
        }]

        max_rounds = 3
        last_text = ""
        for _ in range(max_rounds):
            # Plan C: trim oldest turns before the send so an over-long history
            # never hits the API; if we trimmed, tell the model via the system.
            messages, report = auto_rescue_context(messages)
            sys_for_send = system
            notice = build_context_rescue_system_block(report)
            if notice:
                sys_for_send = system + "\n\n" + notice
            try:
                turn = client.send(messages, system=sys_for_send, tools=specs,
                                   max_tokens=1200, temperature=0.3)
            except Exception:
                return last_text
            messages.append({"role": "assistant", "content": turn.raw_content})
            if turn.text:
                last_text = turn.text
            if not turn.wants_tool:
                return turn.text
            results = []
            for call in turn.tool_calls:
                out, ok = _execute_agent_tool_call(call.name, call.input, session)
                ledger.record(call.name, {"ok": ok}, ok=ok, summary=out[:200])
                results.append(ToolResult(tool_use_id=call.id, content=out,
                                          is_error=not ok).to_wire())
            messages.append({"role": "user", "content": results})

        # Budget exhausted — one final turn to synthesize (wrap-up instruction).
        try:
            wrap = system + ("\n\n[WRAP UP] Give a concise, scientist-quality answer "
                             "now from the tool results above; do not request tools.")
            final = client.send(messages, system=wrap, tools=specs,
                               max_tokens=1200, temperature=0.3)
            return final.text or last_text
        except Exception:
            return last_text

    def _scientist_loop(self, session: "SessionState", user: str,
                        history: list[dict[str, Any]],
                        turn_plan: dict[str, Any] | None = None) -> str:
        """Prefer a real Anthropic tool-use loop (Plan B); on any miss, fall back
        to the two-pass text protocol (reason → parse [tool:] → synthesize)."""
        real = self._real_tool_loop(session, user, turn_plan=turn_plan)
        if real:
            return real
        client = self._get_client()
        if client is None:
            return ""

        ctx = _rich_context(session) + _format_turn_plan_context(turn_plan)

        # Pass 1: LLM reasons about the user's question, may request tools
        pass1_prompt = (
            f"{ctx}\n\n"
            f"[USER QUESTION]\n{user}\n\n"
            "As a research scientist, analyze the situation. If you need to check "
            "something (model status, data, recent results, GPU, etc.), write "
            "[tool: <name>] on its own line. Available tools: model_status, "
            "system_status, task_list, inspect_task, data_check, recent_run, "
            "gpu_status, kaggle_status, dashboard, next_steps, evolution_status, "
            "scientist_checkpoint, research_decision, scientist_workplan, "
            "scientist_turn_plan, scientist_repair_plan, scientist_execution_contract, scientist_step_trace, "
            "scientist_autopilot, scientist_self_audit, scientist_innovation_backlog, "
            "scientist_hypothesis_review, scientist_experiment_blueprint, "
            "scientist_situation_model.\n\n"
            "Be concise. Think like a scientist — what do we know, what do we "
            "need to check, what should we do next?"
        )
        try:
            resp1 = client.generate(
                pass1_prompt,
                system=_SCIENTIST_SYSTEM,
                max_tokens=900,
            )
            pass1_text = (resp1.text or "").strip()
        except Exception:
            return ""

        if not pass1_text:
            return ""

        # Extract tool hints from pass 1
        hints = _forced_tool_hints(user) + _TOOL_HINT_RE.findall(pass1_text)
        hints = list(dict.fromkeys(hints))  # dedup, preserve order

        # Execute tools and collect results
        tool_results_text = ""
        executed = set()
        for hint in hints[:3]:  # max 3 tools per turn
            name = hint.lower()
            if name in executed:
                continue
            executed.add(name)
            tool_results_text += _execute_terminal_tool(name, session) + "\n\n"

        if not tool_results_text:
            # No tools requested — return pass 1 text directly
            return pass1_text

        # Pass 2: Feed tool results back to LLM for synthesis
        pass2_prompt = (
            f"{ctx}\n\n"
            f"[USER QUESTION]\n{user}\n\n"
            f"[YOUR INITIAL ANALYSIS]\n{pass1_text}\n\n"
            f"[TOOL RESULTS]\n{tool_results_text}\n\n"
            "Now synthesize these results into a clear, scientist-quality response "
            "for the user. Be specific — reference actual data, suggest concrete "
            "next steps, and flag any blockers. Keep it concise."
        )
        try:
            resp2 = client.generate(
                pass2_prompt,
                system=_SCIENTIST_SYSTEM,
                max_tokens=1200,
            )
            return (resp2.text or "").strip() or pass1_text
        except Exception:
            return pass1_text

    # ── Planning ─────────────────────────────────────────────────────

    def plan(self, goal: str, session: "SessionState") -> str:
        if self._llm_available(session):
            prompt = (
                "You are a research scientist. The user wants a research PLAN "
                "(NOT execution) for this goal:\n\n"
                f"{goal}\n\n"
                f"{_rich_context(session)}\n\n"
                "Design a concrete, actionable 6-9 step research plan. Each step "
                "should be specific to the task (reference the task brief). Include: "
                "data preparation, feature engineering approaches, model families to try, "
                "validation strategy, and how to evaluate success. Do NOT suggest "
                "starting training — this is PLANNING only."
            )
            answer = self._ask_raw(prompt, max_tokens=1200)
            if answer:
                return answer
        return self._rule_plan(goal, session)

    def capability(self, session: "SessionState") -> str:
        task_line = (
            f"当前任务：{session.selected_task}"
            if session.selected_task
            else "尚未选择比赛"
        )
        return (
            f"{task_line}\n\n"
            "我是 EvoMind，你的 AI 科研科学家终端。我可以：\n\n"
            "  🔍 数据探索：自动检查数据结构、缺失值、分布特征\n"
            "  📊 策略推荐：基于任务类型（表格/图像/时序）推荐合适的模型和特征工程\n"
            "  🧪 实验设计：形成可验证假说，设计对照实验\n"
            "  🏋️ 训练执行：通过工作站门禁启动可审计的自动训练\n"
            "  📈 结果分析：解读 CV 分数、分析提升原因、建议下一步\n"
            "  📝 报告导出：自动生成 Markdown/HTML/DOCX 实验报告\n"
            "  🔐 安全边界：Kaggle 提交永远需要人工确认，分数/排名必须有真实 artifact\n\n"
            "告诉我你想研究哪个比赛，或者描述你的研究目标。"
        )

    # ── Internal helpers ─────────────────────────────────────────────

    def _ask_raw(self, prompt: str, *, max_tokens: int = 900) -> Optional[str]:
        client = self._get_client()
        if client is None:
            return None
        try:
            resp = client.generate(prompt, system=_SCIENTIST_SYSTEM, max_tokens=max_tokens)
            return (resp.text or "").strip() or None
        except Exception:
            return None

    # ── Deterministic fallback (no LLM) ──────────────────────────────

    def _rule_reply(self, text: str, session: "SessionState") -> str:
        """Deterministic scientist-style reply when LLM is unavailable.

        Uses task brief, data status, recent results, and gate info to give
        meaningful responses — not just template text.
        """
        task = session.selected_task
        gaps = session.missing_setup()
        normalized = (text or "").strip().lower()

        if _forced_tool_hints(text):
            from .terminal_tools import TerminalTools
            root = Path(session.workspace_root) if session.workspace_root else Path.cwd()
            tool_name = _forced_tool_hints(text)[0]
            if tool_name == "scientist_turn_plan":
                from .scientist_turn_planner import build_scientist_turn_plan

                result = build_scientist_turn_plan(session, root, text, persist=True)
            else:
                result = TerminalTools.dispatch(tool_name, session, root)
            display_name = {
                "scientist_autopilot": "Scientist Autopilot",
                "scientist_repair_plan": "Scientist Repair Plan",
                "scientist_execution_contract": "Scientist Execution Contract",
                "scientist_workplan": "Scientist Workplan",
                "scientist_step_trace": "Scientist Step Trace",
                "scientist_recovery": "Scientist Recovery Snapshot",
                "scientist_action_queue": "Scientist Action Queue",
                "scientist_next_action": "Scientist Next Action",
                "scientist_self_audit": "Scientist Self Audit",
                "scientist_upgrade_plan": "Scientist Upgrade Plan",
                "scientist_self_upgrade_loop": "Scientist Self-Upgrade Loop",
                "scientist_innovation_backlog": "Scientist Innovation Backlog",
                "scientist_hypothesis_review": "Scientist Hypothesis Review",
                "scientist_experiment_blueprint": "Scientist Experiment Blueprint",
                "scientist_situation_model": "Scientist Situation Model",
                "scientist_turn_plan": "Scientist Turn Plan",
            }.get(tool_name, tool_name)
            lines = [f"EvoMind {display_name} completed a read-only analysis."]
            lines.extend(str(item) for item in result.get("summary_lines", []))
            situation_model = result.get("situation_model")
            if isinstance(situation_model, dict):
                lines.append(
                    "Situation: "
                    f"status={result.get('situation_status')}; "
                    f"readiness_score={result.get('readiness_score')}; "
                    f"posture={situation_model.get('posture')}."
                )
                research_question = situation_model.get("research_question")
                if research_question:
                    lines.append(f"Research question: {research_question}")
                readiness_checks = situation_model.get("readiness_checks")
                if isinstance(readiness_checks, dict):
                    passed = [key for key, value in readiness_checks.items() if value]
                    missing = [key for key, value in readiness_checks.items() if not value]
                    lines.append(f"Readiness: passed={len(passed)}, missing={len(missing)}")
                    if missing:
                        lines.append("Missing readiness:")
                        lines.extend(f"- {item}" for item in missing[:6])
                blocker_model = situation_model.get("blocker_model")
                if isinstance(blocker_model, list) and blocker_model:
                    lines.append("Blocker model:")
                    for item in blocker_model[:5]:
                        if isinstance(item, dict):
                            lines.append(
                                "- "
                                f"{item.get('category', 'unknown')}: "
                                f"{item.get('blocker', '')} "
                                f"(repair={item.get('repair_command', '')})"
                            )
                uncertainties = situation_model.get("uncertainties")
                if isinstance(uncertainties, list) and uncertainties:
                    lines.append("Uncertainties:")
                    lines.extend(f"- {item}" for item in uncertainties[:5])
                recommended = situation_model.get("recommended_tool_sequence")
                if isinstance(recommended, list) and recommended:
                    lines.append("Recommended safe tool sequence:")
                    lines.extend(f"- {item}" for item in recommended[:5])
            blockers = result.get("blockers", [])
            if blockers:
                lines.append("Blockers:")
                lines.extend(f"- {item}" for item in blockers[:6])
            root_causes = result.get("root_causes", [])
            if root_causes:
                lines.append("Root causes:")
                lines.extend(f"- {item}" for item in root_causes[:6])
            repair_steps = result.get("repair_steps", [])
            if repair_steps:
                lines.append("Repair steps:")
                for step in repair_steps[:5]:
                    if isinstance(step, dict):
                        lines.append(f"- {step.get('id')}: {step.get('title')} ({step.get('status')})")
            next_actions = result.get("next_actions", [])
            if next_actions:
                lines.append("Next actions:")
                lines.extend(f"- {item}" for item in next_actions[:6])
            hypotheses = result.get("innovation_hypotheses", [])
            if hypotheses:
                lines.append("Innovation hypotheses:")
                for item in hypotheses[:4]:
                    if isinstance(item, dict):
                        lines.append(
                            "- "
                            f"{item.get('strategy_name', item.get('id', 'hypothesis'))}: "
                            f"branch={item.get('proposed_branch_type', '')}; "
                            f"gate={item.get('gate', '')}"
                        )
            reviews = result.get("reviews", [])
            if reviews:
                lines.append("Hypothesis review:")
                for item in reviews[:4]:
                    if isinstance(item, dict):
                        lines.append(
                            "- "
                            f"#{item.get('rank')} {item.get('strategy_name', item.get('hypothesis_id', 'hypothesis'))}: "
                            f"score={item.get('score')}; status={item.get('status')}; risk={item.get('risk_level')}"
                        )
            selected_hypothesis = result.get("selected_hypothesis")
            if isinstance(selected_hypothesis, dict):
                lines.append(
                    "Selected hypothesis: "
                    f"{selected_hypothesis.get('strategy_name', selected_hypothesis.get('hypothesis_id', 'hypothesis'))}; "
                    f"score={selected_hypothesis.get('score')}; "
                    f"next_gate={selected_hypothesis.get('next_gate')}."
                )
            blueprint = result.get("experiment_blueprint")
            if isinstance(blueprint, dict):
                lines.append(
                    "Experiment blueprint: "
                    f"id={blueprint.get('blueprint_id')}; "
                    f"branch={blueprint.get('branch_type')}; "
                    f"mode={blueprint.get('code_generation_mode')}; "
                    f"run_command={blueprint.get('run_command')}."
                )
            actions = result.get("actions", []) or result.get("action_queue", [])
            if actions:
                lines.append("Action queue:")
                for action in actions[:5]:
                    if isinstance(action, dict):
                        lines.append(
                            "- "
                            f"{action.get('title', action.get('id', 'action'))}: "
                            f"command={action.get('command', '')}; "
                            f"gate={action.get('gate', '')}; "
                            f"status={action.get('status', '')}"
                        )
            selected_action = result.get("selected_action")
            if isinstance(selected_action, dict):
                lines.append(
                    "Selected next action: "
                    f"{selected_action.get('title', selected_action.get('id', 'action'))}; "
                    f"command={selected_action.get('command', '')}; "
                    f"gate={selected_action.get('gate', '')}."
                )
            safe_next = result.get("safe_next_command")
            if safe_next:
                lines.append(f"Safe next command: {safe_next}")
            if result.get("message"):
                lines.append(str(result.get("message")))
            go_no_go = result.get("go_no_go")
            if go_no_go:
                lines.append(f"Go/No-Go: {go_no_go}")
            decision = result.get("decision", {})
            if isinstance(decision, dict) and decision:
                lines.append(
                    "Decision: "
                    f"action={decision.get('selected_action')}, "
                    f"branch={decision.get('selected_branch')}, "
                    f"mode={decision.get('code_generation_mode')}"
                )
            lines.append(f"Artifact: {result.get('artifact_path', f'.xsci/{tool_name}.json')}")
            lines.append("No training or official Kaggle submission was started.")
            return "\n".join(lines)

        # ── Status ──
        if normalized in {"status", "/status", "ready", "就绪", "状态"}:
            return self._build_status_reply(session, gaps)

        # ── Greeting ──
        if normalized in {"你好", "hello", "hi", "hey"}:
            return self._build_greeting(session)

        # ── Task list ──
        if any(w in normalized for w in ("任务列表", "有哪些任务", "注册的任务", "我有哪些")):
            return self._build_task_list_reply(session)

        # ── No task ──
        if not task:
            return (
                "我还没有看到你选择比赛。\n\n"
                "你可以这样开始：\n"
                "1. 告诉我比赛名称（比如 Titanic、House Prices），我来搜索\n"
                "2. 粘贴 Kaggle URL：task add https://www.kaggle.com/c/titanic\n"
                "3. 运行 `evomind setup` 先配置环境\n\n"
                "你也可以直接说'浏览比赛'，我来帮你查看 Kaggle 上有什么有趣的任务。"
            )

        # ── Has task — build a context-rich response ──
        return self._build_task_aware_reply(session, task, gaps)

    def _build_status_reply(self, session: "SessionState", gaps) -> str:
        """Build a comprehensive status report."""
        lines = ["📊 EvoMind 系统状态\n"]
        lines.append(f"  工作区：{session.workspace_root}")
        lines.append(f"  当前任务：{session.selected_task or '(未选择)'}")

        # Task info
        if session.task_brief:
            lines.append(f"\n  📋 任务信息：{session.task_brief}")

        # LLM
        llm_status = "✅ 就绪" if session.llm_ready else "❌ 需要配置"
        lines.append(f"\n  🧠 LLM：{session.llm_provider} — {llm_status}")

        # Kaggle
        kg_status = "✅ 就绪" if session.kaggle_ready else "⚠️ 未配置"
        lines.append(f"  📦 Kaggle API：{kg_status}")

        # GPU
        if not session.gpu_ready:
            lines.append("  🖥️ GPU/HPC：未配置（仅本地算力可用）")
        elif session.gpu_blocked:
            lines.append(f"  🖥️ GPU/HPC：已配置但被阻塞 — {session.gpu_blocker or session.gpu_status}")
        else:
            lines.append("  🖥️ GPU/HPC：已配置且可用")

        # Recent results
        if session.recent_run_id:
            cv_str = f"{session.recent_best_cv:.4f}" if session.recent_best_cv is not None else "N/A"
            lines.append(f"\n  📈 最近训练：{session.recent_run_id} | Best CV: {cv_str}")
        else:
            lines.append("\n  📈 最近训练：尚无")

        if session.memory_summary:
            lines.append(f"  🧠 经验记忆：{session.memory_summary}")

        # Gaps
        if gaps:
            lines.append("\n  ⚠️ 需要配置：")
            for gap in gaps:
                lines.append(f"    - {gap.split(':', 1)[0]}")

        if not gaps and session.selected_task:
            lines.append("\n  ✅ 所有门禁就绪！输入你的研究目标开始训练。")

        return "\n".join(lines)

    def _build_greeting(self, session: "SessionState") -> str:
        """Build a warm, scientist-like greeting."""
        if session.selected_task:
            return (
                f"你好！我看到你在研究 **{session.selected_task}**。\n\n"
                + (f"任务概况：{session.task_brief}\n\n" if session.task_brief else "")
                + "我可以帮你：\n"
                "• 检查数据和配置状态\n"
                "• 分析任务特点，推荐实验方向\n"
                "• 制定研究计划\n"
                "• 在门禁通过后启动可审计的训练\n\n"
                "你想从哪个步骤开始？"
            )
        if session.n_tasks > 0:
            return (
                f"你好！我看到你有 {session.n_tasks} 个已注册的任务。"
                f"用 `use <任务名>` 选中一个，然后告诉我你想怎么研究它。"
            )
        return (
            "你好！我是 EvoMind，你的 AI 科研科学家。\n\n"
            "我目前还没有看到你注册比赛。你可以：\n"
            "• 说 `浏览比赛` 来搜索 Kaggle\n"
            "• 粘贴 Kaggle URL 来注册新任务\n"
            "• 说 `setup` 来配置 LLM 和 Kaggle API\n\n"
            "准备好了就开始吧！"
        )

    def _build_task_list_reply(self, session: "SessionState") -> str:
        from .terminal_tools import TerminalTools
        root = Path(session.workspace_root) if session.workspace_root else Path.cwd()
        result = TerminalTools.dispatch("task_list", session, root)
        tasks = result.get("tasks", [])
        if not tasks:
            return "还没有注册任何比赛。你可以说 `浏览比赛` 来搜索 Kaggle，或者直接粘贴比赛 URL。"
        lines = [f"已注册 {len(tasks)} 个任务："]
        for t in tasks:
            mark = "→" if t["slug"] == session.selected_task else " "
            lines.append(f"  {mark} {t['slug']}" + (f"  — {t['brief']}" if t.get("brief") else ""))
        if session.selected_task:
            lines.append(f"\n当前选中：**{session.selected_task}**。你打算怎么研究它？")
        else:
            lines.append("\n用 `use <任务名>` 选择一个任务开始研究。")
        return "\n".join(lines)

    def _build_task_aware_reply(self, session: "SessionState", task: str, gaps) -> str:
        """Build a context-aware response when a task is selected."""
        parts = [f"当前在研究 **{task}**。"]

        if session.task_brief:
            brief = session.task_brief
            # Parse the brief for useful info
            metric_match = re.search(r'metric=(\w+)', brief)
            modality_match = re.search(r'modality=(\w+)', brief)
            if metric_match and modality_match:
                parts.append(
                    f"这是一个 **{modality_match.group(1)}** 类型的任务，"
                    f"评估指标是 **{metric_match.group(1)}**。"
                )
            parts.append(f"详情：{brief}")

        # Data
        from .terminal_tools import TerminalTools
        root = Path(session.workspace_root) if session.workspace_root else Path.cwd()
        data = TerminalTools.dispatch("data_check", session, root)
        if data.get("train_csv"):
            parts.append("\n✅ 训练数据已就绪")
            # Check recent run
            if session.recent_run_id:
                cv_str = f"{session.recent_best_cv:.4f}" if session.recent_best_cv is not None else "N/A"
                parts.append(f"📈 最近训练：{session.recent_run_id}（Best CV: {cv_str}）")
                parts.append("\n💡 建议：分析上次训练结果，针对薄弱点进行下一轮改进。说'继续上次实验'来恢复训练。")
            else:
                parts.append("\n💡 建议：数据已就绪，你可以说'开始训练'来建立基线模型。")
        else:
            parts.append(f"\n⚠️ 训练数据尚未下载。运行 `evomind download {task}` 获取数据。")

        # Gaps
        if gaps:
            parts.append("\n⚠️ 以下配置缺失：")
            for gap in gaps:
                parts.append(f"  - {gap.split(':', 1)[0]}")

        if not gaps and data.get("train_csv"):
            parts.append("\n✅ 所有门禁通过，随时可以开始训练。告诉我你的研究目标，或者直接说'开始训练，建立基线'。")

        return "\n".join(parts)

    def _rule_plan(self, goal: str, session: "SessionState") -> str:
        task = session.selected_task or "(no task selected)"
        lines = [f"📋 {task} 研究计划（仅规划，不训练）\n"]
        if session.task_brief:
            lines.append(f"  任务概要：{session.task_brief}\n")
        lines.append(f"  目标：{goal.strip() or '建立强基线，然后通过2-3轮有证据的改进提升 CV'}\n")
        lines.append("  建议的实验步骤：")
        lines.extend([
            "  1. 数据审计 — 缺失值、异常值、分布、泄露风险",
            "  2. 特征工程 — 编码、交叉特征、目标编码（如适用）",
            "  3. 基线模型 — GBM（LightGBM/XGBoost/CatBoost）+ 标准 K-fold CV",
            "  4. 改进方向 — 超参调优、stacking、伪标签（如适用）",
            "  5. 验证策略 — 时间序列 split（时序任务）或 Stratified K-fold（分类任务）",
            "  6. 报告 — 自动生成含图表的实验报告",
        ])
        if session.can_execute():
            lines.append("\n✅ 门禁通过。说'开始训练'或 `/run` 进入执行阶段。")
        else:
            gaps = session.missing_setup()
            if gaps:
                lines.append("\n⚠️ 以下需要先配置：")
                for gap in gaps:
                    lines.append(f"  - {gap.split(':', 1)[0]}")
        return "\n".join(lines)
