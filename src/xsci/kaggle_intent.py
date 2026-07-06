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

    if tok == "official":
        return Intent(OFFICIAL, args=raw.split()[1:])
    if tok == "task" and low.split()[1:2] == ["add"]:
        parts = raw.split()
        return Intent(TASK_ADD, payload=parts[2] if len(parts) > 2 else "", args=parts[3:])
    if tok == "use" and len(raw.split()) >= 2:
        return Intent(TASK_USE, payload=raw.split()[1])

    if low in _GREETINGS or any(low.startswith(g) and len(low) <= len(g) + 3 for g in _GREETINGS):
        return Intent(GREETING)

    if _contains(low, _STATUS):
        return Intent(STATUS)
    if _contains(low, _CAPABILITY):
        return Intent(CAPABILITY)
    if _contains(low, _REPORT) and not _contains(low, _EXECUTION):
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
