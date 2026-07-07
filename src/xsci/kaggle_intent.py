"""Intent routing for the EvoMind research terminal.

This module is intentionally deterministic: a free-text line is mapped to chat,
planning, execution, or a command intent before the shell does anything
expensive. Planning never trains. Execution only means "enter the execution
gate"; the caller still checks task, LLM, resource, and human-gate readiness.
"""
from __future__ import annotations

from dataclasses import dataclass, field

GREETING = "greeting"
STATUS = "status"
CAPABILITY = "capability"
TASK_ADD = "task_add"
TASK_USE = "task_use"
PLANNING = "planning"
EXECUTION = "execution"
REPORT = "report"
MEMORY = "memory"
OFFICIAL = "official"
TOOL_QUERY = "tool_query"
CHAT = "chat"


@dataclass
class Intent:
    kind: str
    payload: str = ""
    args: list[str] = field(default_factory=list)


_GREETINGS = {
    "hi",
    "hello",
    "hey",
    "yo",
    "你好",
    "您好",
    "在吗",
    "在么",
    "哈喽",
    "嗨",
}

_HARD_NOW = (
    "开始",
    "立刻",
    "立即",
    "现在就",
    "马上",
    "启动",
    "开跑",
    "直接跑",
    "直接开始",
    "run now",
    "start now",
    "go ahead",
    "just run",
    "just do",
    "kick off",
    "let's go",
    "let's run",
    "lets run",
)

_PLANNING = (
    "规划",
    "计划",
    "方案",
    "思路",
    "设计一个",
    "策略",
    "打算",
    "怎么做",
    "如何",
    "制定",
    "分析一下",
    "评估一下",
    "先想",
    "想一个",
    "构思",
    "拟一个",
    "plan",
    "strategy",
    "outline",
    "approach",
    "how should",
    "how would",
    "propose a plan",
    "design a",
    "sketch",
)

_EXECUTION = (
    "训练",
    "运行",
    "执行",
    "自进化",
    "自我进化",
    "进化",
    "跑一轮",
    "跑一遍",
    "跑起来",
    "提交候选",
    "出个基线",
    "做基线",
    "建模",
    "开一轮",
    "再来一轮",
    "下一轮",
    "train",
    "run",
    "execute",
    "self-evolve",
    "self evolve",
    "evolve",
    "baseline it",
    "开始训练",
    "启动训练",
    "切到这个",
    "换到这个",
    "切换到这个",
    "切换到",
    "切到",
    "换成",
    "换到",
)

_STATUS = ("status", "doctor", "配置", "状态", "缺什么", "还缺", "检查", "就绪", "ready?")
_CAPABILITY = (
    "你能",
    "功能",
    "能力",
    "介绍一下",
    "架构",
    "system",
    "capability",
    "what can you",
    "who are you",
    "你是谁",
    "帮我做什么",
    "能做什么",
)
_REPORT = ("report", "报告", "结果", "台账", "results", "怎么样了", "进展")
_MEMORY = ("memory", "记忆", "经验", "教训", "lesson", "复盘", "retrospective")

# ── TOOL_QUERY keyword sets ────────────────────────────────────────────
_MODEL_STATUS = (
    "什么模型", "当前模型", "使用什么模型", "用的什么模型",
    "现在用的什么", "什么 llm", "当前 llm", "模型是什么",
    "哪个模型", "用的是什么", "model",
)
_TOOL_STATUS = (
    "什么工具", "哪些工具", "能调用什么", "有什么工具",
    "工具列表", "能用什么", "可以调用什么", "可以做什么",
    "有什么功能", "支持哪些",
)
_DATA_CHECK = (
    "数据准备", "数据好了", "数据在哪", "有没有数据",
    "检查数据", "数据就绪", "数据可用", "下载数据",
    "数据在哪下", "数据目录",
)
_RESUME = (
    "继续上次", "继续上一", "接着上次", "接着做", "继续跑",
    "resume", "continue", "继续训练", "接着训练",
    "继续做", "再做一轮", "再跑一轮",
)
_PROGRESS = (
    "进度", "怎么样了", "结果怎么样", "跑完了吗",
    "训练好了吗", "进展", "最近结果", "查看结果",
    "查看进度", "看进度", "上次结果", "训练结果",
)
_GPU_STATUS = (
    "gpu状态", "gpu 状态", "服务器状态", "gpu 就绪",
    "gpu 可用", "集群状态", "hpc 状态",
)
_KAGGLE_STATUS = (
    "kaggle状态", "kaggle 状态", "kaggle 配置",
    "kaggle 就绪",
)


def _contains(text: str, needles) -> bool:
    return any(n in text for n in needles)


def _first_token(text: str) -> str:
    return text.strip().split(maxsplit=1)[0].lower() if text.strip() else ""


def classify(text: str) -> Intent:
    raw = (text or "").strip()
    low = raw.lower()
    if not raw:
        return Intent(CHAT)

    tok = _first_token(raw)

    # ── Fast path: explicit commands ──────────────────────────────
    if tok == "official":
        return Intent(OFFICIAL, args=raw.split()[1:])
    if tok == "task" and low.split()[1:2] == ["add"]:
        parts = raw.split()
        return Intent(TASK_ADD, payload=parts[2] if len(parts) > 2 else "", args=parts[3:])
    if tok == "use" and len(raw.split()) >= 2:
        return Intent(TASK_USE, payload=raw.split()[1])

    # ── Greetings (short exact-match wins over longer keyword scans) ──
    if low in _GREETINGS or any(low.startswith(g) and len(low) <= len(g) + 3 for g in _GREETINGS):
        return Intent(GREETING)

    # ── TOOL_QUERY: lightweight tool calls that are NOT training ──
    if _contains(low, _MODEL_STATUS) and not _contains(low, _EXECUTION):
        return Intent(TOOL_QUERY, payload="model_status")
    # "我有哪些任务" / "有哪些任务" / "注册了哪些"
    if (any(w in low for w in ("有哪些任务", "有哪些比赛", "哪些任务", "注册了哪些", "任务列表"))
            and not _contains(low, _EXECUTION)):
        return Intent(TOOL_QUERY, payload="task_list")
    # "训练怎么样"/"训练完了吗" — progress query, not execution
    if (any(w in low for w in ("训练怎么样", "训练好了吗", "训练完了吗", "跑完了吗"))
            and not _contains(low, _HARD_NOW)):
        return Intent(TOOL_QUERY, payload="progress")
    if _contains(low, _TOOL_STATUS) and not _contains(low, _EXECUTION):
        return Intent(TOOL_QUERY, payload="tool_status")
    if _contains(low, _GPU_STATUS) and not _contains(low, _EXECUTION):
        return Intent(TOOL_QUERY, payload="gpu_status")
    if _contains(low, _KAGGLE_STATUS) and not _contains(low, _EXECUTION):
        return Intent(TOOL_QUERY, payload="kaggle_status")
    if _contains(low, _DATA_CHECK) and not _contains(low, _EXECUTION):
        return Intent(TOOL_QUERY, payload="data_check")
    # Progress/report queries take priority over execution when user is
    # asking about results, not requesting action.  "训练结果怎么样"
    # should be a query, not start training.
    if _contains(low, _PROGRESS) and not _contains(low, _HARD_NOW):
        return Intent(TOOL_QUERY, payload="progress")
    if _contains(low, _RESUME):
        # "resume" / "continue" — keep as EXECUTION with a resume flag
        return Intent(EXECUTION, payload="resume")

    # ── Traditional intents ───────────────────────────────────────
    if _contains(low, _STATUS) and not _contains(low, _HARD_NOW):
        return Intent(STATUS)
    if _contains(low, _CAPABILITY):
        return Intent(CAPABILITY)
    if _contains(low, _REPORT) and not _contains(low, _HARD_NOW):
        return Intent(REPORT)
    if _contains(low, _MEMORY) and not _contains(low, _EXECUTION):
        return Intent(MEMORY)

    hard_now = _contains(low, _HARD_NOW)
    wants_plan = _contains(low, _PLANNING)
    wants_exec = _contains(low, _EXECUTION)

    if wants_plan and not hard_now:
        return Intent(PLANNING)
    if hard_now or wants_exec:
        return Intent(EXECUTION)
    return Intent(CHAT)


def is_execution(text: str) -> bool:
    return classify(text).kind == EXECUTION


def is_planning(text: str) -> bool:
    return classify(text).kind == PLANNING
