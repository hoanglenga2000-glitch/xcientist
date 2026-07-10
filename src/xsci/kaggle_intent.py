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


_REAL_GREETINGS = {"你好", "您好", "在吗", "在么", "嗨", "哈喽"}
_REAL_HARD_NOW = (
    "开始", "立刻", "立即", "现在就", "马上", "启动", "直接跑", "直接开始",
)
_REAL_PLANNING = (
    "规划", "计划", "方案", "思路", "设计一个", "策略", "打算", "怎么做",
    "如何", "制定", "先想", "想一下", "拟一个", "帮我规划",
)
_REAL_EXECUTION = (
    "训练", "运行", "执行", "自进化", "自我进化", "进化", "跑一轮",
    "跑一遍", "跑起来", "出个基线", "做基线", "建模", "开一轮",
    "再来一轮", "开始训练", "启动训练",
)
_REAL_STATUS = ("配置", "状态", "缺什么", "还缺", "检查", "就绪")
_REAL_CAPABILITY = (
    "你能", "功能", "能力", "介绍一下", "架构", "你是谁", "帮我做什么", "能做什么",
)
_REAL_REPORT = ("报告", "结果", "台账", "怎么样了", "进展")
_REAL_MEMORY = ("记忆", "经验", "教训", "复盘")
_REAL_MODEL_STATUS = (
    "什么模型", "当前模型", "使用什么模型", "用的什么模型", "现在用的什么",
    "当前 llm", "模型是什么", "哪个模型",
)
_REAL_TOOL_STATUS = (
    "什么工具", "哪些工具", "能调用什么", "有什么工具", "工具列表",
    "能用什么", "可以调用什么", "支持哪些",
)
_REAL_DATA_CHECK = (
    "数据准备", "数据好了", "数据在哪", "有没有数据", "检查数据",
    "数据就绪", "数据可用", "下载数据", "数据目录",
)
_REAL_RESUME = (
    "继续上次", "继续上一", "接着上次", "接着做", "继续跑",
    "继续训练", "接着训练", "再做一轮", "再跑一轮",
)
_REAL_PROGRESS = (
    "进度", "结果怎么样", "跑完了吗", "训练好了吗", "最近结果",
    "查看结果", "查看进度", "看进度", "上次结果", "训练结果",
)
_REAL_GPU_STATUS = (
    "gpu状态", "gpu 状态", "服务器状态", "gpu 就绪", "gpu 可用",
    "集群状态", "hpc 状态",
)
_REAL_KAGGLE_STATUS = ("kaggle状态", "kaggle 状态", "kaggle 配置", "kaggle 就绪")
_REAL_EVOLUTION_STATUS = (
    "有没有学到", "学到经验", "学习经验", "自进化统计", "自动进化状态",
    "进化状态", "自进化证据", "经验沉淀", "能力成长", "进化报告",
)
_REAL_SCIENTIST_CHECKPOINT = (
    "科学家", "研究状态", "下一步", "怎么提升", "如何提升", "提升方案",
    "自进化能力", "自主学习", "智能分析", "checkpoint", "scientist",
)
_REAL_RESEARCH_DECISION = (
    "下一轮", "实验决策", "做什么实验", "选择分支", "选分支",
    "branch", "code mode", "决策", "怎么跑下一轮",
)
_REAL_SCIENTIST_WORKPLAN = (
    "工作计划", "执行计划", "路线图", "roadmap", "workplan", "agenda",
    "拆解步骤", "多步计划", "持续推进", "恢复计划", "下一步怎么执行",
)
_REAL_SCIENTIST_REPAIR = (
    "修复计划", "自我修复", "自修复", "怎么修", "如何修复", "哪里卡住",
    "卡在哪里", "阻塞原因", "失败归因", "修复路线", "修复建议",
    "repair plan", "fix plan", "self repair", "root cause", "why blocked",
)
_REAL_SCIENTIST_CONTRACT = (
    "执行合同", "执行契约", "执行前检查", "运行前检查", "开跑前检查",
    "能不能跑", "可以训练吗", "可以开跑吗", "执行计划合同", "训练合同",
    "execution contract", "run contract", "pre-execution", "preflight contract",
)
_REAL_SCIENTIST_TRACE = (
    "步骤轨迹", "运行轨迹", "工具轨迹", "工具调用过程", "执行证据流",
    "step trace", "steptrace", "tool trace", "trace",
    "live trace", "live stream", "scientist live", "scientist stream",
    "evidence stream", "real-time trace", "realtime trace", "streaming trace",
)
_REAL_SCIENTIST_RECOVERY = (
    "恢复现场", "恢复状态", "恢复上下文", "上下文恢复", "上下文丢了",
    "断点恢复", "断点状态", "重启后恢复", "从哪里继续", "当前恢复点",
    "recovery snapshot", "recovery guard", "recover context", "resume context",
    "compaction recovery", "restart recovery",
)
_REAL_SCIENTIST_ACTION_QUEUE = (
    "行动队列", "动作队列", "下一步队列", "action queue", "queue",
    "要做什么动作", "下一步命令", "计划队列",
)
_REAL_SCIENTIST_TURN_PLAN = (
    "turn plan", "tool plan", "per-turn plan", "plan this turn",
    "plan your tools", "what tools will you use", "tool rationale",
    "本轮计划", "工具计划", "行动计划", "本次回合", "先规划本轮",
    "你准备调用什么工具", "你会用哪些工具", "每轮计划",
)
_REAL_SCIENTIST_NEXT_ACTION = (
    "安全下一步", "执行安全下一步", "推进下一步", "执行下一步",
    "继续行动", "下一步行动", "next action", "safe next", "act next",
)
_REAL_SCIENTIST_CONTINUATION_RESUME = (
    "resume continuation", "resume safe", "finish continuation",
    "finish remaining safe tools", "finish remaining tools",
    "run remaining safe tools", "complete remaining safe tools",
    "continue remaining tools", "auto continue tools",
    "自动续跑", "自动继续工具", "自动完成剩余工具",
    "自动跑完剩余工具", "剩余安全工具自动跑完", "剩余工具自动跑完",
    "剩余只读工具自动跑完", "跑完剩余安全工具", "跑完剩余只读工具",
    "把剩余安全工具跑完", "把剩余只读工具跑完",
    "把没跑完的工具跑完", "完成上轮剩余工具",
    "继续完成上轮安全工具", "续跑剩余工具", "续跑安全工具",
)
_REAL_SCIENTIST_CONTINUATION_STATUS = (
    "continuation", "continuation status", "continue status", "turn status",
    "remaining tools", "deferred tools", "unfinished tools", "incomplete turn",
    "what is left to run", "what tools remain", "previous turn status",
    "续跑状态", "续跑进度", "续跑到哪", "续跑到哪了", "续跑还剩",
    "还有哪些工具没跑", "还剩哪些工具", "哪些工具没跑完", "没跑完的工具",
    "复杂任务进度", "上轮没跑完", "上一轮没跑完", "上次没跑完",
    "回合闭环了吗", "当前回合闭环", "复杂回合进度",
)
_REAL_SCIENTIST_LOOP = (
    "科学家循环", "自主循环", "自主回合", "多步回合", "自动推进",
    "持续推进", "持续优化", "继续优化", "连续诊断", "持续诊断",
    "像claude code一样", "像 claude code 一样", "像codex一样",
    "像 codex 一样", "scientist loop", "agent loop", "autonomous loop",
)
_REAL_SCIENTIST_SELF_AUDIT = (
    "自我审计", "能力审计", "能力评估", "智能度评估", "系统能力差距",
    "agent 能力", "agent能力", "像 claude code 还差什么", "像 codex 还差什么",
    "和 claude code 差距", "和 codex 差距", "够不够像 claude code",
    "够不够像 codex", "self audit", "self-audit", "capability audit",
    "agent audit", "agent capability", "intelligence audit",
    "how close to claude code", "what is missing from claude code",
)
_REAL_SCIENTIST_READINESS_REPORT = (
    "readiness report", "launch readiness", "scientist readiness",
    "agent readiness", "go no-go report", "go/no-go report",
    "上线报告", "上线就绪报告", "上线检查报告", "最终就绪报告",
    "能力报告", "智能体能力报告", "训练就绪报告", "能不能上线",
    "能否上线", "能不能训练", "能否训练", "系统是否稳定上线",
    "上线前检查", "上线前审计", "安全上线检查",
)
_REAL_SCIENTIST_CAUSAL_DIAGNOSIS = (
    "causal diagnosis", "causal graph", "cause map", "root cause map",
    "root-cause map", "why blocked", "why is it blocked",
    "why not training", "why cannot train", "diagnose causes",
    "因果诊断", "因果图", "因果分析", "根因图", "根因链路",
    "根因分析图", "为什么不能训练", "为什么不能上线",
    "问题归因", "阻塞归因", "症状根因",
)
_REAL_SCIENTIST_STRATEGY_OPTIMIZER = (
    "strategy optimizer", "priority plan", "intervention plan",
    "decision matrix", "intervention ranking", "action ranking",
    "rank interventions", "prioritize interventions", "choose next action",
    "which action first", "what should we do first", "best next strategy",
    "下一步策略", "策略优化", "策略排序", "优先级计划", "优先级排序",
    "干预排序", "干预优先级", "行动排序", "下一步优先级",
    "先做哪个", "应该先做什么", "哪个动作最重要", "哪个最划算",
    "怎么排优先级", "决策矩阵", "下一步决策矩阵",
)
_REAL_SCIENTIST_CONTEXT_PACKET = (
    "context packet", "scientist context", "scientist briefing",
    "context briefing", "state briefing", "research briefing",
    "working context", "turn context", "build context packet",
    "生成上下文包", "上下文包", "科学家上下文", "科学家简报",
    "科研简报", "状态简报", "回合上下文", "工作上下文",
    "认知上下文", "把当前上下文整理出来", "整理当前上下文",
)
_REAL_SCIENTIST_UPGRADE_PLAN = (
    "upgrade plan", "upgrade backlog", "self upgrade", "agent upgrade",
    "capability upgrade", "close upgrade backlog", "fix upgrade backlog",
    "engineering plan", "system upgrade plan",
    "升级计划", "能力升级计划", "系统升级计划", "自我升级",
    "修复升级项", "关闭升级项", "升级 backlog", "修复 backlog",
    "把backlog转成计划", "把 backlog 转成计划", "工程升级计划",
)
_REAL_SCIENTIST_SELF_UPGRADE_LOOP = (
    "self-upgrade loop", "self upgrade loop", "upgrade loop",
    "capability loop", "capability work order", "self-upgrade work order",
    "create self-upgrade work order", "execute self-upgrade",
    "execute self upgrade", "run self-upgrade",
    "自升级闭环", "自我升级闭环", "能力自升级", "能力升级闭环",
    "执行自升级", "运行自升级", "开启自升级", "开始自升级",
    "生成自升级工单", "创建自升级工单", "能力缺口转成工单",
    "把p0能力缺口转成工单", "把 p0 能力缺口转成工单",
    "把能力缺口转成工程工单", "自进化工程工单",
)
_REAL_SCIENTIST_PATCH_WORK_ORDER = (
    "patch work order", "patch-order", "code patch order", "repair work order",
    "code-agent patch", "code agent patch", "create patch work order",
    "generate patch work order", "turn failure into patch", "failure to patch",
    "补丁工单", "代码补丁工单", "修复工单", "生成补丁工单", "创建补丁工单",
    "生成代码修复工单", "创建代码修复工单", "把失败转成补丁", "把问题转成补丁",
    "把问题转成工程修复", "代码agent修复工单", "代码 agent 修复工单",
)
_REAL_SCIENTIST_MEMORY_CONSOLIDATION = (
    "巩固记忆", "沉淀经验", "沉淀记忆", "写入记忆", "写进记忆", "长期记忆",
    "经验入库", "记忆入库", "复盘入库", "学习经验", "把经验存起来",
    "consolidate memory", "memory consolidation", "write memory",
    "writeback memory", "memory writeback", "learn from trace",
    "persist lessons", "save lessons", "retrospective memory",
)
_REAL_SCIENTIST_INNOVATION_BACKLOG = (
    "innovation backlog", "innovate plan", "innovation plan",
    "innovation hypothesis", "innovation hypotheses", "research hypotheses",
    "memory guided innovation", "memory-guided innovation",
    "novel branch", "novel combination", "propose innovation",
    "generate innovation", "generate hypotheses",
    "创新假设", "创新计划", "创新分支", "生成创新", "生成假设",
    "根据记忆创新", "复用记忆", "记忆复用", "跨任务创新",
)
_REAL_SCIENTIST_HYPOTHESIS_REVIEW = (
    "review hypotheses", "review hypothesis", "hypothesis review",
    "rank hypotheses", "rank hypothesis", "critique hypotheses",
    "critique hypothesis", "score hypotheses", "score hypothesis",
    "which hypothesis", "best hypothesis", "proposal review",
    "review proposals", "rank proposals", "critique proposals",
    "评审假设", "假设评审", "假设排序", "排序假设", "评估假设",
    "评价假设", "哪一个假设", "哪个假设", "最佳假设", "评审创新",
    "创新评审", "方案评审", "评审方案", "排序方案",
)
_REAL_SCIENTIST_EXPERIMENT_BLUEPRINT = (
    "experiment blueprint", "candidate blueprint", "execution blueprint",
    "plan experiment", "blueprint", "gated experiment plan",
    "实验蓝图", "执行蓝图", "候选蓝图", "实验方案", "执行方案",
    "生成实验计划", "生成实验蓝图", "转成实验", "转成执行计划",
    "把假设落地", "把方案落地", "可执行实验", "实验设计",
)
_REAL_SCIENTIST_INNOVATION_FEEDBACK = (
    "innovation feedback", "trial feedback", "innovation trial feedback",
    "feedback innovation", "scientist feedback", "proposal feedback",
    "write innovation feedback", "write trial feedback",
    "write hypothesis result", "write hypothesis outcome",
    "record gate outcome", "record gate feedback",
    "innovation log feedback", "update innovation log",
    "创新反馈", "试验反馈", "实验反馈", "创新试验反馈", "创新实验反馈",
    "假设反馈", "方案反馈", "门禁反馈", "写回创新日志", "写入创新日志",
    "把假设结果写回创新日志", "把门禁结果写回创新日志", "把蓝图结果写回记忆",
    "记录创新反馈", "记录试验反馈", "记录门禁结果", "沉淀创新经验",
)
_REAL_SCIENTIST_SITUATION_MODEL = (
    "situation model", "scientist situation", "state model",
    "current situation", "research situation", "orient",
    "synthesize evidence", "synthesize blockers", "what is the situation",
    "why are we blocked", "what should the scientist do next",
    "analyze the current situation", "scientist state",
    "局势", "情境", "态势", "当前状态模型", "科学家状态", "现在局面",
    "现在卡在哪里", "为什么卡住", "下一步判断", "综合证据", "综合分析当前",
)
_REAL_SCIENTIST_AUTOPILOT = (
    "全面诊断", "自动诊断", "主动分析", "自主分析", "完整诊断",
    "系统诊断", "诊断当前", "不够智能", "像ai scientist", "像 ai scientist",
    "真正的ai scientist", "真正 ai scientist", "复杂问题", "超级终端",
    "完整检查", "全面检查", "自动分析下一步",
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
    if (low in _GREETINGS or low in _REAL_GREETINGS
            or any(low.startswith(g) and len(low) <= len(g) + 3 for g in tuple(_GREETINGS) + tuple(_REAL_GREETINGS))):
        return Intent(GREETING)

    # ── TOOL_QUERY: lightweight tool calls that are NOT training ──
    if (_contains(low, _MODEL_STATUS) or _contains(low, _REAL_MODEL_STATUS)) and not (_contains(low, _EXECUTION) or _contains(low, _REAL_EXECUTION)):
        return Intent(TOOL_QUERY, payload="model_status")
    if (_contains(low, _REAL_SCIENTIST_MEMORY_CONSOLIDATION)
            and not (_contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW))):
        return Intent(TOOL_QUERY, payload="scientist_memory_consolidation")
    if (_contains(low, _REAL_EVOLUTION_STATUS)
            and not (_contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW))):
        return Intent(TOOL_QUERY, payload="evolution_status")
    if _contains(low, _REAL_SCIENTIST_SELF_UPGRADE_LOOP):
        return Intent(TOOL_QUERY, payload="scientist_self_upgrade_loop")
    if _contains(low, _REAL_SCIENTIST_PATCH_WORK_ORDER):
        return Intent(TOOL_QUERY, payload="scientist_patch_work_order")
    if (_contains(low, _REAL_SCIENTIST_UPGRADE_PLAN)
            and not (_contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW))):
        return Intent(TOOL_QUERY, payload="scientist_upgrade_plan")
    if (_contains(low, _REAL_SCIENTIST_SELF_AUDIT)
            and not (_contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW))):
        return Intent(TOOL_QUERY, payload="scientist_self_audit")
    if (_contains(low, _REAL_SCIENTIST_READINESS_REPORT)
            and not (_contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW))):
        return Intent(TOOL_QUERY, payload="scientist_readiness_report")
    if (_contains(low, _REAL_SCIENTIST_CAUSAL_DIAGNOSIS)
            and not (_contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW))):
        return Intent(TOOL_QUERY, payload="scientist_causal_diagnosis")
    if (_contains(low, _REAL_SCIENTIST_CONTEXT_PACKET)
            and not (_contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW))):
        return Intent(TOOL_QUERY, payload="scientist_context_packet")
    if (_contains(low, _REAL_SCIENTIST_STRATEGY_OPTIMIZER)
            and not (_contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW))):
        return Intent(TOOL_QUERY, payload="scientist_strategy_optimizer")
    if (_contains(low, _REAL_SCIENTIST_HYPOTHESIS_REVIEW)
            and not (_contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW))):
        return Intent(TOOL_QUERY, payload="scientist_hypothesis_review")
    if (_contains(low, _REAL_SCIENTIST_EXPERIMENT_BLUEPRINT)
            and not (_contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW))):
        return Intent(TOOL_QUERY, payload="scientist_experiment_blueprint")
    if (_contains(low, _REAL_SCIENTIST_INNOVATION_FEEDBACK)
            and not (_contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW))):
        return Intent(TOOL_QUERY, payload="scientist_innovation_trial_feedback")
    if (_contains(low, _REAL_SCIENTIST_SITUATION_MODEL)
            and not (_contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW))):
        return Intent(TOOL_QUERY, payload="scientist_situation_model")
    if (_contains(low, _REAL_SCIENTIST_INNOVATION_BACKLOG)
            and not (_contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW))):
        return Intent(TOOL_QUERY, payload="scientist_innovation_backlog")
    if (_contains(low, _REAL_SCIENTIST_LOOP)
            and not (_contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW))):
        return Intent(TOOL_QUERY, payload="scientist_loop")
    if (_contains(low, _REAL_SCIENTIST_RECOVERY)
            and not (_contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW))
            and not (_contains(low, _EXECUTION) or _contains(low, _REAL_EXECUTION))):
        return Intent(TOOL_QUERY, payload="scientist_recovery")
    if _contains(low, _REAL_SCIENTIST_CONTINUATION_RESUME):
        return Intent(TOOL_QUERY, payload="scientist_continuation_resume")
    if _contains(low, _REAL_SCIENTIST_CONTINUATION_STATUS):
        return Intent(TOOL_QUERY, payload="scientist_continuation_status")
    if _contains(low, _REAL_SCIENTIST_ACTION_QUEUE):
        return Intent(TOOL_QUERY, payload="scientist_action_queue")
    if _contains(low, _REAL_SCIENTIST_TURN_PLAN):
        return Intent(TOOL_QUERY, payload="scientist_turn_plan")
    if _contains(low, _REAL_SCIENTIST_NEXT_ACTION):
        return Intent(TOOL_QUERY, payload="scientist_next_action")
    if (_contains(low, _REAL_SCIENTIST_AUTOPILOT)
            and not (_contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW))):
        return Intent(TOOL_QUERY, payload="scientist_autopilot")
    if (_contains(low, _REAL_SCIENTIST_REPAIR)
            and not (_contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW))):
        return Intent(TOOL_QUERY, payload="scientist_repair_plan")
    if (_contains(low, _REAL_SCIENTIST_CONTRACT)
            and not (_contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW))):
        return Intent(TOOL_QUERY, payload="scientist_execution_contract")
    if (_contains(low, _REAL_SCIENTIST_WORKPLAN)
            and not (_contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW))):
        return Intent(TOOL_QUERY, payload="scientist_workplan")
    if (_contains(low, _REAL_SCIENTIST_TRACE)
            and not (_contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW))):
        return Intent(TOOL_QUERY, payload="scientist_step_trace")
    if (_contains(low, _REAL_RESEARCH_DECISION)
            and not (_contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW))
            and not (_contains(low, _EXECUTION) or _contains(low, _REAL_EXECUTION))):
        return Intent(TOOL_QUERY, payload="research_decision")
    if (_contains(low, _REAL_SCIENTIST_CHECKPOINT)
            and not (_contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW))):
        return Intent(TOOL_QUERY, payload="scientist_checkpoint")
    # "我有哪些任务" / "有哪些任务" / "注册了哪些"
    if (any(w in low for w in ("有哪些任务", "有哪些比赛", "哪些任务", "注册了哪些", "任务列表"))
            and not (_contains(low, _EXECUTION) or _contains(low, _REAL_EXECUTION))):
        return Intent(TOOL_QUERY, payload="task_list")
    # "训练怎么样"/"训练完了吗" — progress query, not execution
    if (any(w in low for w in ("训练怎么样", "训练好了吗", "训练完了吗", "跑完了吗"))
            and not (_contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW))):
        return Intent(TOOL_QUERY, payload="progress")
    if (_contains(low, _TOOL_STATUS) or _contains(low, _REAL_TOOL_STATUS)) and not (_contains(low, _EXECUTION) or _contains(low, _REAL_EXECUTION)):
        return Intent(TOOL_QUERY, payload="tool_status")
    if (_contains(low, _GPU_STATUS) or _contains(low, _REAL_GPU_STATUS)) and not (_contains(low, _EXECUTION) or _contains(low, _REAL_EXECUTION)):
        return Intent(TOOL_QUERY, payload="gpu_status")
    if (_contains(low, _KAGGLE_STATUS) or _contains(low, _REAL_KAGGLE_STATUS)) and not (_contains(low, _EXECUTION) or _contains(low, _REAL_EXECUTION)):
        return Intent(TOOL_QUERY, payload="kaggle_status")
    if (_contains(low, _DATA_CHECK) or _contains(low, _REAL_DATA_CHECK)) and not (_contains(low, _EXECUTION) or _contains(low, _REAL_EXECUTION)):
        return Intent(TOOL_QUERY, payload="data_check")
    # Progress/report queries take priority over execution when user is
    # asking about results, not requesting action.  "训练结果怎么样"
    # should be a query, not start training.
    if (_contains(low, _PROGRESS) or _contains(low, _REAL_PROGRESS)) and not (_contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW)):
        return Intent(TOOL_QUERY, payload="progress")
    if _contains(low, _RESUME) or _contains(low, _REAL_RESUME):
        # "resume" / "continue" — keep as EXECUTION with a resume flag
        return Intent(EXECUTION, payload="resume")

    # ── Traditional intents ───────────────────────────────────────
    if (_contains(low, _STATUS) or _contains(low, _REAL_STATUS)) and not (_contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW)):
        return Intent(STATUS)
    if _contains(low, _CAPABILITY) or _contains(low, _REAL_CAPABILITY):
        return Intent(CAPABILITY)
    if (_contains(low, _REPORT) or _contains(low, _REAL_REPORT)) and not (_contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW)):
        return Intent(REPORT)
    if (_contains(low, _MEMORY) or _contains(low, _REAL_MEMORY)) and not (_contains(low, _EXECUTION) or _contains(low, _REAL_EXECUTION)):
        return Intent(MEMORY)

    hard_now = _contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW)
    wants_plan = _contains(low, _PLANNING) or _contains(low, _REAL_PLANNING)
    wants_exec = _contains(low, _EXECUTION) or _contains(low, _REAL_EXECUTION)

    if wants_plan and not hard_now:
        return Intent(PLANNING)
    if hard_now or wants_exec:
        return Intent(EXECUTION)
    return Intent(CHAT)


def is_execution(text: str) -> bool:
    return classify(text).kind == EXECUTION


def is_planning(text: str) -> bool:
    return classify(text).kind == PLANNING
