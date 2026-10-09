"""Intent routing for the EvoMind research terminal.

This module is intentionally deterministic: a free-text line is mapped to chat,
planning, execution, or a command intent before the shell does anything
expensive. Planning never trains. Execution only means "enter the execution
gate"; the caller still checks task, LLM, resource, and human-gate readiness.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .user_request import UserRequest, parse_user_request

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
    request: UserRequest | None = None


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
_REAL_LITERATURE_QUERY = (
    "检索论文", "搜索论文", "找论文", "查论文", "论文检索", "文献检索",
    "搜索文献", "找文献", "查文献", "参考文献", "学术论文", "论文资料",
    "literature search", "search papers", "find papers", "paper search",
    "arxiv", "openalex", "crossref", "rag 文献", "rag论文", "rag paper",
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
_REAL_CURRENT_RUN_STATUS = (
    "上次模型微调完成到哪", "上次模型完成到哪", "模型微调完成到哪",
    "当前模型微调状态", "上次微调状态", "当前运行状态", "当前 run 状态",
    "当前run状态", "目前任务完成情况", "当前任务完成情况", "上次任务完成情况",
    "previous fine-tune status", "last fine-tune status", "current run status",
)
_REAL_LOCAL_ENVIRONMENT = (
    "检查本地开发环境", "检查开发环境", "本地工具环境", "本地开发工具",
    "开发环境检查", "环境探针", "environment check", "check local environment",
    "inspect local environment", "development environment",
)
_REAL_REFINEMENT_APPROVE = (
    "批准微调", "确认微调", "批准这个微调", "确认这个微调", "批准优化方案",
    "确认优化方案", "approve refinement", "approve the refinement",
)
_REAL_REFINEMENT_REJECT = (
    "拒绝微调", "取消微调", "拒绝这个微调", "驳回优化方案",
    "reject refinement", "reject the refinement",
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
_REAL_SCIENTIST_ENGINEERING_LOOP = (
    "engineering loop", "execute engineering loop", "validate patch",
    "test patch", "apply patch in worktree", "isolated worktree",
    "execute generated patch", "run code-agent patch", "verify code-agent patch",
    "工程闭环", "执行工程闭环", "验证补丁", "测试补丁",
    "隔离验证补丁", "隔离工作树", "执行已生成补丁", "验证代码agent补丁",
    "验证 code agent 补丁", "运行补丁测试", "执行自升级补丁",
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
_REAL_SCIENTIST_HYPOTHESIS_PANEL = (
    "hypothesis panel", "research panel", "parallel hypotheses",
    "multi agent hypotheses", "multi-agent hypotheses", "independent critics",
    "parallel hypothesis generation", "adversarial hypothesis panel",
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


def _is_literature_query(text: str) -> bool:
    reviewed_reference = (
        "刚检索", "已检索", "已经检索", "检索到", "检索的", "上述文献",
        "当前文献", "已有文献", "所选论文", "这篇论文", "这些论文",
        "previously retrieved", "retrieved literature", "selected paper",
    )
    synthesis_actions = (
        "结合", "根据", "基于", "解释", "分析", "说明", "总结", "归纳", "比较",
        "为什么", "意味着", "explain", "analyze", "summarize", "based on",
    )
    if _contains(text, reviewed_reference) and _contains(text, synthesis_actions):
        return False
    if _contains(text, _REAL_LITERATURE_QUERY):
        return True
    search_actions = ("检索", "搜索", "查找", "查询", "找", "search", "find", "retrieve")
    literature_objects = ("论文", "文献", "参考资料", "paper", "papers", "literature", "arxiv", "openalex", "crossref")
    return _contains(text, search_actions) and _contains(text, literature_objects)


def _is_validation_explanation(text: str) -> bool:
    validation = ("交叉验证", "cross validation", "cross-validation", "分组验证", "grouped validation")
    grouping = ("患者", "重复", "病灶", "patient", "duplicate", "content group")
    explanation = ("解释", "分析", "说明", "为什么", "偏差", "依据", "explain", "analyze", "bias")
    return _contains(text, validation) and _contains(text, grouping) and _contains(text, explanation)


def _is_metric_explanation(text: str) -> bool:
    metrics = ("roc-auc", "roc_auc", "roc auc", "pr-auc", "pr_auc", "pr auc", "brier")
    explanation = (
        "为什么", "解释", "说明", "区别", "怎么看", "列出", "给出", "数值", "多少", "是多少",
        "mean", "interpret", "explain", "list", "show", "report", "value",
    )
    return _contains(text, metrics) and _contains(text, explanation)


def is_artifact_location_query(text: str) -> bool:
    """Recognize a question about already-produced files, not a report action."""
    normalized = (text or "").strip().lower()
    location = (
        "在哪", "哪里", "路径", "位置", "储存", "存储", "保存到", "本地电脑",
        "下载链接", "download", "path", "location", "stored", "saved",
    )
    artifact = (
        "证据包", "交付物", "报告", "结果", "代码包", "文件", "pdf", "csv", "zip",
        "artifact", "deliverable", "evidence", "result", "output",
    )
    return _contains(normalized, location) and _contains(normalized, artifact)


def _metric_query_requests_execution(text: str) -> bool:
    """Treat a named existing Run as context, not as the English `run` verb."""
    execution_terms = tuple(
        term for term in _EXECUTION + _REAL_EXECUTION
        if term not in {"run", "运行"}
    )
    if _contains(text, execution_terms):
        return True
    return _contains(text, (
        "run training", "run the training", "run experiment", "run an experiment",
        "run model", "run the model", "运行训练", "运行实验", "运行模型",
    ))


_NEGATED_EXECUTION_PATTERNS = (
    r"不要(?:再|立刻|立即|现在|马上)?(?:开始|启动|执行|运行|进行)?(?:任何)?(?:训练|建模|运行|执行|自进化|提交)",
    r"不(?:要|需要|用|必|必需|必須)?(?:开始|启动|执行|运行|进行)?(?:任何)?(?:训练|建模|运行|执行|自进化|提交)",
    r"无需(?:开始|启动|执行|运行|进行)?(?:训练|建模|运行|执行|自进化|提交)",
    r"只(?:做|要|需|需要)?(?:分析|规划|研究|检查|比较|诊断)",
    r"仅(?:做|要|需|需要)?(?:分析|规划|研究|检查|比较|诊断)",
    r"(?:do not|don't|dont|without)\s+(?:start(?:ing)?\s+)?(?:train(?:ing)?|run(?:ning)?|execute|submit)",
    r"(?:analysis|planning|research)\s+only",
)


def _mask_negated_execution(text: str) -> str:
    """Remove explicit no-execution clauses before intent scoring.

    Research prompts often contain words such as "training" only to forbid the
    action. Treating those words as an execution request makes the Scientist
    ignore the actual analysis task and collapse into resource-gate reporting.
    """
    import re

    masked = text
    for pattern in _NEGATED_EXECUTION_PATTERNS:
        masked = re.sub(pattern, " ", masked, flags=re.IGNORECASE)
    return masked


def _first_token(text: str) -> str:
    return text.strip().split(maxsplit=1)[0].lower() if text.strip() else ""


def _is_progress_only_query(text: str) -> bool:
    """Distinguish "训练好了吗" from a compound request that asks to train."""
    if not (_contains(text, _PROGRESS) or _contains(text, _REAL_PROGRESS)):
        return False
    action_cues = (
        "请", "帮我", "需要", "完成", "开始", "启动", "执行", "在hpc", "在 hpc",
        "并训练", "然后训练", "训练并", "please", "need you", "start", "run and",
    )
    return not _contains(text, action_cues)


def _is_llm_refinement_request(text: str) -> bool:
    lineage = (
        "上次模型", "上一轮", "当前模型", "已有模型", "原模型", "父版本",
        "继续训练", "接着训练", "增量训练", "current model", "previous run",
        "last run", "continue training", "incremental training", "parent adapter",
    )
    changes = (
        "继续优化", "降低学习率", "调低学习率", "调整学习率", "修改学习率",
        "短周期", "再训练一轮", "更新报告", "重新审核", "重新评测",
        "refine", "lower the learning rate", "short cycle", "update the report",
    )
    return _contains(text, changes) and (
        _contains(text, lineage)
        or _contains(text, ("降低学习率", "调低学习率", "短周期", "lower the learning rate"))
    )


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
    request = parse_user_request(raw)
    execution_scoring_text = _mask_negated_execution(low)
    hard_now = (
        _contains(execution_scoring_text, _HARD_NOW)
        or _contains(execution_scoring_text, _REAL_HARD_NOW)
    )
    # Asking where an existing result lives is a question for the LLM Agent.
    # It must not be swallowed by the generic REPORT/status adapters.
    if is_artifact_location_query(low):
        return Intent(CHAT)
    if (
        _is_metric_explanation(low)
        and not hard_now
        and not _metric_query_requests_execution(execution_scoring_text)
    ):
        return Intent(CHAT)
    if _contains(low, _REAL_REFINEMENT_APPROVE):
        return Intent(TOOL_QUERY, payload="llm_refinement_approve", args=[raw])
    if _contains(low, _REAL_REFINEMENT_REJECT):
        return Intent(TOOL_QUERY, payload="llm_refinement_reject", args=[raw])
    if _contains(low, _REAL_CURRENT_RUN_STATUS) and not hard_now:
        return Intent(TOOL_QUERY, payload="current_run", args=[raw])
    if _contains(low, _REAL_LOCAL_ENVIRONMENT) and not hard_now:
        return Intent(TOOL_QUERY, payload="local_environment", args=[raw])
    if _is_llm_refinement_request(low) and not _contains(low, _REAL_REFINEMENT_APPROVE + _REAL_REFINEMENT_REJECT):
        return Intent(TOOL_QUERY, payload="llm_refinement", args=[raw])
    if (_contains(low, _MODEL_STATUS) or _contains(low, _REAL_MODEL_STATUS)) and not (_contains(low, _EXECUTION) or _contains(low, _REAL_EXECUTION)):
        return Intent(TOOL_QUERY, payload="model_status")
    if (_is_literature_query(low)
            and not (_contains(execution_scoring_text, _HARD_NOW) or _contains(execution_scoring_text, _REAL_HARD_NOW))
            and not (_contains(execution_scoring_text, _EXECUTION) or _contains(execution_scoring_text, _REAL_EXECUTION))):
        # Preserve the complete user turn for the shared live literature API.
        return Intent(TOOL_QUERY, payload="literature_search", args=[raw])
    if any(token in low for token in ("external capability certification", "external certification status")):
        return Intent(TOOL_QUERY, payload="scientist_capability_certification")
    if "upgrade campaign status" in low:
        return Intent(TOOL_QUERY, payload="scientist_upgrade_campaign")
    if "research parity gate" in low:
        return Intent(TOOL_QUERY, payload="scientist_research_parity_gate")
    if (_contains(low, _REAL_SCIENTIST_MEMORY_CONSOLIDATION)
            and not (_contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW))):
        return Intent(TOOL_QUERY, payload="scientist_memory_consolidation")
    if (_contains(low, _REAL_EVOLUTION_STATUS)
            and not (_contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW))):
        return Intent(TOOL_QUERY, payload="evolution_status")
    if _contains(low, _REAL_SCIENTIST_ENGINEERING_LOOP):
        return Intent(TOOL_QUERY, payload="scientist_engineering_loop")
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
    if (_contains(low, _REAL_SCIENTIST_HYPOTHESIS_PANEL)
            and not (_contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW))):
        return Intent(TOOL_QUERY, payload="scientist_hypothesis_panel")
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
    # Atomic action extraction runs before generic status/report routing, but
    # after exact read-only research tools. Thus "检查数据、训练、审核并生成报告"
    # executes as one request while "为什么不能训练，做因果诊断" remains a
    # diagnosis rather than accidentally starting a job.
    if request.requests_execution and not _is_progress_only_query(low):
        return Intent(EXECUTION, request=request)
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
    if (
        _is_metric_explanation(low)
        and not hard_now
        and not _metric_query_requests_execution(execution_scoring_text)
    ):
        return Intent(CHAT)
    if (
        _is_validation_explanation(low)
        and not hard_now
        and not _contains(execution_scoring_text, _EXECUTION)
        and not _contains(execution_scoring_text, _REAL_EXECUTION)
    ):
        return Intent(CHAT)
    if request.requests_research and not hard_now:
        return Intent(PLANNING, request=request)
    if (_contains(low, _STATUS) or _contains(low, _REAL_STATUS)) and not (_contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW)):
        return Intent(STATUS)
    if _contains(low, _CAPABILITY) or _contains(low, _REAL_CAPABILITY):
        return Intent(CAPABILITY)
    if (_contains(low, _REPORT) or _contains(low, _REAL_REPORT)) and not (_contains(low, _HARD_NOW) or _contains(low, _REAL_HARD_NOW)):
        return Intent(REPORT)
    if (_contains(low, _MEMORY) or _contains(low, _REAL_MEMORY)) and not (_contains(low, _EXECUTION) or _contains(low, _REAL_EXECUTION)):
        return Intent(MEMORY)

    wants_plan = (
        _contains(low, _PLANNING)
        or _contains(low, _REAL_PLANNING)
        or (
            execution_scoring_text != low
            and any(term in low for term in ("分析", "假设", "比较", "诊断", "研究", "analyze", "hypoth", "compare", "diagnos"))
        )
    )
    wants_exec = (
        _contains(execution_scoring_text, _EXECUTION)
        or _contains(execution_scoring_text, _REAL_EXECUTION)
    )

    if hard_now or wants_exec:
        return Intent(EXECUTION, request=request)
    # Explicit no-training constraints only close the execution stage. Data
    # analysis, hypothesis work, and experiment design remain a research plan.
    if request.requests_research or wants_plan:
        return Intent(PLANNING, request=request)
    return Intent(CHAT)


_LLM_INTENT_CACHE: dict[str, tuple[float, Intent]] = {}
_LLM_INTENT_TTL = 600.0


def classify_with_llm_fallback(text: str, *, generate_fn=None) -> Intent:
    """classify() with optional LLM fallback for ambiguous inputs."""
    result = classify(text)
    if result.kind != CHAT or generate_fn is None or len((text or "").strip()) < 8:
        return result
    import time
    key = (text or "").strip()[:200]
    now = time.monotonic()
    cached = _LLM_INTENT_CACHE.get(key)
    if cached and now - cached[0] < _LLM_INTENT_TTL:
        return cached[1]
    try:
        prompt = (
            "Classify user intent into ONE category: STATUS, PLANNING, EXECUTION, "
            "REPORT, MEMORY, TOOL_QUERY, or CHAT.\n"
            f"User: {key}\nCategory:"
        )
        reply = (generate_fn(prompt, max_tokens=20) or "").strip().upper()
        kind_map = {
            "STATUS": STATUS, "PLANNING": PLANNING, "EXECUTION": EXECUTION,
            "REPORT": REPORT, "MEMORY": MEMORY, "TOOL_QUERY": TOOL_QUERY,
        }
        for k, v in kind_map.items():
            if k in reply:
                intent = Intent(v)
                _LLM_INTENT_CACHE[key] = (now, intent)
                return intent
    except Exception:
        pass
    return result


def is_execution(text: str) -> bool:
    return classify(text).kind == EXECUTION


def is_planning(text: str) -> bool:
    return classify(text).kind == PLANNING
