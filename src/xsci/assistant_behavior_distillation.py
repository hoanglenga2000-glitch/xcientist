"""Auditable interaction-policy distillation for the EvoMind web agent.

The runtime does not copy teacher answers or hidden prompts.  It retrieves small
behaviour cards distilled from operator-owned conversations, compiles a novice
request into an execution-neutral contract, and performs a deterministic
coverage audit on the visible answer.  This keeps the useful Codex/Claude Code
interaction patterns while leaving facts, reasoning, and prose to the selected
LLM plus EvoMind's real tools.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .user_request import parse_user_request

BOARD_SCHEMA = "evomind.assistant_behavior_board.v1"
CONTRACT_SCHEMA = "evomind.web_request_compiler.v3"
AUDIT_SCHEMA = "evomind.web_response_audit.v1"
DEFAULT_BOARD = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "evaluation"
    / "assistant_behavior_board_v1.json"
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BehaviorCard:
    card_id: str
    priority: int
    audiences: tuple[str, ...]
    facets: tuple[str, ...]
    instructions: tuple[str, ...]
    provenance: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BehaviorCard":
        card_id = str(payload.get("card_id") or "").strip()
        instructions = tuple(str(item).strip() for item in payload.get("instructions") or [] if str(item).strip())
        if not card_id or not instructions:
            raise ValueError("behavior card requires card_id and instructions")
        provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
        if provenance.get("raw_content_retained") is not False:
            raise ValueError(f"behavior card must not retain raw teacher content: {card_id}")
        return cls(
            card_id=card_id,
            priority=int(payload.get("priority") or 0),
            audiences=tuple(str(item) for item in payload.get("audiences") or ["adaptive"]),
            facets=tuple(str(item) for item in payload.get("facets") or ["general_help"]),
            instructions=instructions,
            provenance=dict(provenance),
        )

    def public_prompt_payload(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "instructions": list(self.instructions),
        }


@dataclass(frozen=True)
class WebTurnContract:
    conversation_mode: str
    audience: str
    task_type: str
    dataset: str | None
    route: str
    actions: tuple[str, ...]
    facets: tuple[str, ...]
    hard_constraints: tuple[str, ...]
    constraint_acknowledgements: tuple[str, ...]
    requested_outputs: tuple[str, ...]
    answer_depth: str
    preferred_evidence_section: str
    required_evidence_sections: tuple[str, ...]
    evidence_required: bool
    side_effect_mode: str
    behavior_cards: tuple[BehaviorCard, ...]
    board_sha256: str

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTRACT_SCHEMA,
            "conversation_mode": self.conversation_mode,
            "audience": self.audience,
            "task": {"task_type": self.task_type, "dataset": self.dataset},
            "intent": {
                "route": self.route,
                "actions": list(self.actions),
                "facets": list(self.facets),
            },
            "hard_constraints": list(self.hard_constraints),
            "constraint_acknowledgements": list(self.constraint_acknowledgements),
            "requested_outputs": list(self.requested_outputs),
            "answer_depth": self.answer_depth,
            "preferred_evidence_section": self.preferred_evidence_section,
            "required_evidence_sections": list(self.required_evidence_sections),
            "evidence_required": self.evidence_required,
            "side_effect_mode": self.side_effect_mode,
            "behavior_card_ids": [card.card_id for card in self.behavior_cards],
            "behavior_board_sha256": self.board_sha256,
            "interaction_contract": {
                "proceed_with_safe_read_only_defaults": self.side_effect_mode == "read_only_reasoning",
                "max_blocking_questions": 1,
                "never_ask_user_to_rewrite_as_professional_prompt": True,
                "complete_every_explicit_facet": True,
                "audit_visible_answer_before_return": True,
            },
            "user_experience_contract": build_user_experience_contract(self),
        }


def build_user_experience_contract(contract: WebTurnContract) -> dict[str, Any]:
    """Build deterministic UX guidance for a novice-facing LLM turn.

    This is not an answer template. It keeps the model operating like an agent:
    understand the user's real goal, retrieve the narrowest evidence needed,
    produce a safe next action, and separate read-only help from gated execution.
    """

    facets = set(contract.facets)
    safe_now: list[str] = []
    gated_later: list[str] = []
    if contract.evidence_required:
        safe_now.append(
            f"Use verified_context:{contract.preferred_evidence_section} before claims, then explain what it means."
        )
    else:
        safe_now.append("Clarify the goal internally and answer from the request without forcing expert prompt wording.")
    if facets.intersection({"planning", "usage_guidance", "troubleshooting"}):
        safe_now.append("Return 1-3 concrete next actions with a proof-of-done signal for each.")
    experiment_plan_required = bool(
        "planning" in facets
        and contract.task_type != "general"
        and (
            contract.conversation_mode == "follow_up"
            or contract.dataset == "siim-isic-melanoma-classification"
            or {"research", "design", "compare", "train"}.intersection(contract.actions)
        )
    )
    if experiment_plan_required:
        safe_now.append(
            "Keep the same patient/content split and baseline, change one variable at a time, and give at most three ordered steps with promotion/stop gates."
        )
    if "literature" in facets:
        safe_now.append("Separate reviewed literature from current Run evidence and cite only verified sources.")
    if "artifacts" in facets:
        safe_now.append("Give reading order, exact file names, download URLs, bytes, and SHA-256 values.")
    if contract.audience == "novice" and facets.intersection({
        "result_explanation", "evidence", "literature", "troubleshooting", "capabilities",
    }):
        safe_now.append(
            "Explain the key technical idea once in explicit plain language, using '简单说' or one concrete analogy."
        )
    siim_result_review_required = bool(
        contract.dataset == "siim-isic-melanoma-classification"
        and contract.conversation_mode == "first_turn"
        and "result_explanation" in facets
        and "planning" in facets
    )
    if siim_result_review_required:
        safe_now.append(
            "For the SIIM result review, explicitly preserve patient/content-grouped OOF scope and the once-only private grader facts: execution_count=1, failed_closed, score=null."
        )
    if contract.side_effect_mode == "controlled_execution_request":
        gated_later.append(
            "Execution/training is not claimed started inside chat; first state the controlled gate and required readiness checks."
        )
    if "no_training" in contract.hard_constraints:
        gated_later.append("Preserve the user's no-training constraint until explicitly changed.")
    if "no_official_submission" in contract.hard_constraints:
        gated_later.append("Preserve the no-official-submission boundary and avoid Kaggle medal claims.")
    if not gated_later:
        gated_later.append("If the user later asks for side effects, route through the controlled execution gate.")
    answer_shape = (
        ["结论", "我理解你的目标", "我查到/依据", "下一步怎么做", "可直接复制的话"]
        if contract.audience == "novice"
        else ["结论", "证据", "决策", "下一步"]
    )
    return {
        "schema": "evomind.user_experience_contract.v1",
        "audience": contract.audience,
        "answer_shape": answer_shape,
        "safe_now": safe_now[:6],
        "gated_later": gated_later[:4],
        "copyable_followup_required": bool(facets.intersection({"planning", "usage_guidance", "troubleshooting"})),
        "required_final_line_prefix": (
            "你可以直接发这句话："
            if facets.intersection({"planning", "usage_guidance", "troubleshooting"})
            else None
        ),
        "proof_of_done_required": bool(
            facets.intersection({"planning", "troubleshooting"})
            or contract.side_effect_mode == "controlled_execution_request"
        ),
        "experiment_plan_contract": (
            {
                "max_steps": 3,
                "controlled_comparison": "same patient/content split and baseline; change one variable at a time",
                "promotion_or_stop_gate_required": True,
            }
            if experiment_plan_required
            else None
        ),
        "task_specific_evidence_contract": (
            {
                "validation_scope_terms": ["患者", "内容", "OOF"],
                "private_grader_terms": ["execution_count=1", "failed_closed", "score=null"],
                "next_plan_terms": ["校准", "误报/假阳性", "分组"],
            }
            if siim_result_review_required
            else None
        ),
        "do_not_show_hidden_reasoning": True,
    }


_FACET_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("result_explanation", ("结果", "指标", "好不好", "效果", "result", "metric", "score")),
    ("evidence", ("证据", "依据", "复核", "审核", "evidence", "audit", "review")),
    ("literature", ("文献", "论文", "doi", "参考", "literature", "paper", "citation")),
    ("artifacts", ("文件", "交付物", "下载", "校验", "sha", "artifact", "download")),
    ("planning", ("下一步", "计划", "方案", "进化", "改进", "怎么改", "提高", "怎么做", "plan", "improve")),
    ("troubleshooting", ("为什么", "坏了", "不行", "报错", "排障", "修复", "why", "error", "fix")),
    ("status", ("状态", "进度", "完成到哪", "运行吗", "status", "progress", "running")),
    ("capabilities", ("能力", "架构", "能做什么", "capability", "architecture")),
    ("usage_guidance", (
        "怎么用", "怎么问", "怎么说", "怎么让", "下一句", "直接复制",
        "不会写", "不专业", "新用户", "提示词", "prompt", "完整交互",
        "how to use", "what should i ask",
    )),
)


def infer_facets(user: str) -> tuple[str, ...]:
    folded = str(user or "").casefold()
    found = [name for name, terms in _FACET_TERMS if any(term in folded for term in terms)]
    return tuple(found or ["general_help"])


def infer_audience(user: str) -> str:
    folded = str(user or "").casefold()
    novice = ("小白", "不懂", "第一次", "零基础", "看不懂", "大白话", "beginner", "new user")
    technical = ("置信区间", "sha-256", "oof", "bootstrap", "ablation", "消融", "qlora", "api")
    if any(term in folded for term in novice):
        return "novice"
    if any(term in folded for term in technical):
        return "technical"
    return "adaptive"


def _explicit_constraint_acknowledgements(text: str) -> tuple[str, ...]:
    """Return boundaries that must be repeated visibly in this conversation.

    SIIM always carries a no-official-submission governance rule internally,
    but unrelated answers should not be padded with that boilerplate.  Visible
    acknowledgement is reserved for a boundary the user explicitly stated or
    asked about in the current/history text.
    """

    folded = str(text or "").casefold()
    acknowledgements: list[str] = []
    if re.search(
        r"(?:不要|别|不需要|无需|禁止|未|没有|不会).{0,12}(?:训练|建模|拟合|实验执行)|"
        r"(?:do not|don't|dont|without|no)\s+(?:start(?:ing)?\s+)?(?:any\s+)?train",
        folded,
        re.I,
    ):
        acknowledgements.append("no_training")
    explicit_no_submit = bool(re.search(
        r"(?:不要|别|不需要|无需|禁止|未|没有|不会).{0,16}(?:提交|上传|Kaggle)|"
        r"(?:do not|don't|dont|without|no)\s+(?:officially\s+)?submit",
        folded,
        re.I,
    ))
    asks_official_result = bool(re.search(
        r"(?:有没有|是否有|有无|是否获得|拿到).{0,10}(?:官方\s*Kaggle|Kaggle\s*官方|官方).{0,8}(?:成绩|分数|奖牌)|"
        r"(?:official\s+(?:Kaggle\s+)?(?:score|result|medal))",
        folded,
        re.I,
    ))
    if explicit_no_submit or asks_official_result:
        acknowledgements.append("no_official_submission")
    return tuple(acknowledgements)


def preferred_evidence_section(user: str) -> str:
    folded = str(user or "").casefold()
    if any(term in folded for term in ("文件", "交付物", "下载", "校验值", "sha-256", "artifact", "download")):
        return "artifacts"
    if any(term in folded for term in ("文献", "论文", "doi", "literature", "paper", "citation")):
        return "literature"
    if any(term in folded for term in ("能力", "架构", "capability", "architecture")):
        return "capabilities"
    if any(term in folded for term in (
        "实验", "结果", "指标", "证据", "效果", "好不好",
        "auc", "brier", "置信区间", "metric", "score",
    )):
        return "metrics"
    if any(term in folded for term in ("run", "运行", "状态", "复核", "grader")):
        return "current_run"
    return "summary"


def load_behavior_board(path: str | Path = DEFAULT_BOARD) -> tuple[list[BehaviorCard], str]:
    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8-sig"))
    if payload.get("schema") != BOARD_SCHEMA:
        raise ValueError("assistant behavior board schema mismatch")
    cards = [BehaviorCard.from_dict(item) for item in payload.get("cards") or [] if isinstance(item, dict)]
    ids = [card.card_id for card in cards]
    if not cards or len(ids) != len(set(ids)):
        raise ValueError("assistant behavior board requires unique cards")
    return cards, _sha256(_canonical_json(payload))


def select_behavior_cards(
    cards: Sequence[BehaviorCard],
    *,
    audience: str,
    facets: Iterable[str],
    follow_up: bool,
    limit: int = 5,
) -> tuple[BehaviorCard, ...]:
    wanted = set(facets)
    if follow_up:
        wanted.add("follow_up")

    def score(card: BehaviorCard) -> tuple[int, int, str]:
        audience_match = audience in card.audiences or "all" in card.audiences or "adaptive" in card.audiences
        facet_hits = len(wanted.intersection(card.facets))
        return (facet_hits * 100 + (20 if audience_match else 0) + card.priority, card.priority, card.card_id)

    eligible = [
        card
        for card in cards
        if wanted.intersection(card.facets)
        or "all" in card.facets
        or (follow_up and "follow_up" in card.facets)
    ]
    ranked = sorted(eligible, key=lambda card: (-score(card)[0], -score(card)[1], score(card)[2]))
    return tuple(ranked[: max(1, int(limit))])


def compile_web_turn(
    user: str,
    history: list[dict[str, Any]] | None = None,
    *,
    board_path: str | Path = DEFAULT_BOARD,
) -> WebTurnContract:
    request = parse_user_request(user)
    inherited_constraints: list[str] = []
    acknowledgement_constraints: list[str] = list(_explicit_constraint_acknowledgements(user))
    inherited_task_type: str | None = None
    inherited_dataset: str | None = None
    for item in list(history or [])[-8:]:
        if not isinstance(item, dict) or str(item.get("role") or "") != "user":
            continue
        prior = parse_user_request(str(item.get("content") or ""))
        for acknowledgement in _explicit_constraint_acknowledgements(str(item.get("content") or "")):
            if acknowledgement not in acknowledgement_constraints:
                acknowledgement_constraints.append(acknowledgement)
        for constraint in prior.negative_constraints:
            if constraint not in inherited_constraints:
                inherited_constraints.append(constraint)
        if prior.task_type != "general":
            inherited_task_type = prior.task_type
        if prior.dataset:
            inherited_dataset = prior.dataset
    # An explicit positive action in the current turn supersedes the matching
    # old prohibition; otherwise follow-up shorthand inherits the user's last
    # stated boundaries like a mature coding agent would.
    if "train" in request.actions:
        inherited_constraints = [item for item in inherited_constraints if item != "no_training"]
    if "official_submit" in request.actions:
        inherited_constraints = [item for item in inherited_constraints if item != "no_official_submission"]
    hard_constraints = list(inherited_constraints)
    for constraint in request.negative_constraints:
        if constraint not in hard_constraints:
            hard_constraints.append(constraint)
    facets_list = list(infer_facets(user))
    # A bare mention of a past/current experiment is descriptive, not a plan.
    # Explicit execution still needs planning guidance even when the novice did
    # not use planning vocabulary such as "方案" or "下一步".
    if request.requests_execution and "planning" not in facets_list:
        facets_list.append("planning")
    facets = tuple(facets_list)
    audience = infer_audience(user)
    folded = str(user or "").casefold()
    explicit_detail = any(term in folded for term in (
        "完整", "详细", "全面", "逐项", "全部", "长报告",
        "complete report", "detailed", "comprehensive", "full report",
    ))
    artifact_requested = "artifacts" in facets or "report" in request.actions
    evidence_required = bool(set(facets).intersection({
        "result_explanation", "evidence", "literature", "artifacts", "status", "capabilities",
    }))
    primary_section = preferred_evidence_section(user)
    evidence_sections: list[str] = [primary_section] if evidence_required else []
    # A literature question that also asks about the current experiment needs
    # both sources.  Artifact evidence is already a complete projection that
    # includes metrics/review/grader, so it stays a single bounded lookup.
    if (
        primary_section == "literature"
        and set(facets).intersection({"result_explanation", "evidence", "status"})
    ):
        evidence_sections.append("metrics")
    try:
        board, board_hash = load_behavior_board(board_path)
    except (OSError, ValueError, json.JSONDecodeError):
        board, board_hash = [], "unavailable"
    selected = select_behavior_cards(
        board,
        audience=audience,
        facets=facets,
        follow_up=bool(history),
    ) if board else ()
    return WebTurnContract(
        conversation_mode="follow_up" if history else "first_turn",
        audience=audience,
        task_type=request.task_type if request.task_type != "general" else (inherited_task_type or "general"),
        dataset=request.dataset or inherited_dataset,
        route=request.route,
        actions=tuple(request.actions),
        facets=facets,
        hard_constraints=tuple(hard_constraints),
        constraint_acknowledgements=tuple(acknowledgement_constraints),
        requested_outputs=tuple(request.deliverables) if artifact_requested else (),
        answer_depth="detailed" if explicit_detail or artifact_requested else "progressive",
        preferred_evidence_section=primary_section,
        required_evidence_sections=tuple(dict.fromkeys(evidence_sections)),
        evidence_required=evidence_required,
        side_effect_mode="controlled_execution_request" if request.requests_execution else "read_only_reasoning",
        behavior_cards=selected,
        board_sha256=board_hash,
    )


def render_behavior_guidance(contract: WebTurnContract) -> str:
    if not contract.behavior_cards:
        return ""
    cards: list[dict[str, Any]] = []
    for card in contract.behavior_cards[:4]:
        cards.append({
            "id": card.card_id,
            "rules": list(card.instructions[:2]),
        })
    payload = {
        "schema": f"{BOARD_SCHEMA}.compact",
        "rule": "Apply rules silently; do not mention cards.",
        "cards": cards,
    }
    return "[DISTILLED INTERACTION BEHAVIOURS — policy only]\n" + _canonical_json(payload)


def _contains_any(answer: str, terms: Iterable[str]) -> bool:
    folded = answer.casefold()
    return any(term.casefold() in folded for term in terms)


_COVERAGE_TERMS: dict[str, tuple[str, ...]] = {
    "result_explanation": ("结果", "指标", "auc", "score", "分数", "意味着"),
    "evidence": ("证据", "依据", "审计", "复核", "run", "sha-256"),
    "literature": ("文献", "论文", "doi", "参考"),
    "artifacts": ("文件", "交付物", "下载", "sha-256", "校验"),
    "planning": ("下一步", "计划", "方向", "步骤", "比较", "门槛", "先检查", "门禁", "proof-of-done"),
    "troubleshooting": ("原因", "根因", "修复", "检查", "验证", "重试"),
    "status": ("状态", "运行", "进度", "完成", "阻塞"),
    "capabilities": ("能力", "架构", "可以", "工具", "流程"),
    "usage_guidance": ("直接复制", "直接发", "你可以直接", "下一句"),
}


def audit_web_response(
    contract: WebTurnContract,
    answer: str,
    *,
    tool_names: Iterable[str] = (),
) -> dict[str, Any]:
    """Audit visible coverage only; never score or expose hidden reasoning."""

    text = str(answer or "").strip()
    tools = tuple(str(item) for item in tool_names if str(item))
    checks: dict[str, bool] = {"nonempty_answer": len(text) >= 20}
    missing: list[str] = []
    for facet in contract.facets:
        terms = _COVERAGE_TERMS.get(facet)
        if not terms:
            continue
        passed = _contains_any(text, terms)
        checks[f"facet:{facet}"] = passed
        if not passed:
            missing.append(f"cover_facet:{facet}")
    if contract.evidence_required:
        passed = any(name in tools for name in ("verified_context", "experiment_results"))
        checks["verified_evidence_used"] = passed
        if not passed:
            missing.append("use_verified_context")
    if "no_training" in contract.constraint_acknowledgements:
        passed = bool(re.search(r"(?:未|没有|不会|不|别|不要).{0,8}训练|no\s+training", text, re.I))
        checks["constraint:no_training"] = passed
        if not passed:
            missing.append("confirm_no_training")
    if "no_official_submission" in contract.constraint_acknowledgements:
        no_submit = bool(re.search(
            r"(?:未|没有|不会|不|别|不要).{0,12}(?:提交|外部提交)|no\s+(?:official\s+)?submission",
            text,
            re.I,
        ))
        no_official_result = bool(re.search(
            r"(?:没有|无|不是|未获得|不属于).{0,12}(?:Kaggle\s*)?官方.{0,8}(?:成绩|分数|奖牌)|"
            r"(?:Kaggle\s*)?官方.{0,8}(?:成绩|分数|奖牌).{0,8}(?:没有|无|为空)",
            text,
            re.I,
        ))
        if contract.dataset == "siim-isic-melanoma-classification":
            no_submit = bool(re.search(
                r"(?:未执行|没有执行|未进行|没有进行)\s*(?:官方\s*)?(?:Kaggle\s*)?提交|"
                r"(?:官方\s*(?:Kaggle\s*)?|Kaggle\s*官方\s*)?提交.{0,6}(?:未执行|没有执行|未进行|没有进行)|"
                r"official_submission_executed\s*[:=]\s*false",
                text,
                re.I,
            ))
            no_official_result = bool(re.search(
                r"(?:没有|无|不是|未获得)\s*(?:Kaggle\s*官方|官方\s*Kaggle)\s*(?:成绩|分数|奖牌)|"
                r"(?:Kaggle\s*官方|官方\s*Kaggle)\s*(?:成绩|分数|奖牌).{0,8}(?:没有|无|为空)",
                text,
                re.I,
            ))
        checks["constraint:no_official_submission"] = no_submit
        checks["constraint:no_official_result"] = no_official_result
        if not no_submit:
            missing.append("confirm_no_official_submission")
        if not no_official_result:
            missing.append("confirm_no_official_result")
    requires_experiment_plan = bool(
        "planning" in contract.facets
        and contract.task_type != "general"
        and (
            {"research", "design", "compare", "train"}.intersection(contract.actions)
            or contract.conversation_mode == "follow_up"
            or contract.dataset == "siim-isic-melanoma-classification"
        )
    )
    if requires_experiment_plan:
        controlled = bool(re.search(
            r"(?:单变量|相同分组|同一分组|固定基线|一次只改|其他.{0,12}不变)",
            text,
            re.I,
        ))
        promotion = bool(re.search(
            r"(?:晋升|门槛|只有当|才算|才进入|停止条件|回滚条件|阈值)",
            text,
            re.I,
        ))
        checks["experiment_plan:controlled_comparison"] = controlled
        checks["experiment_plan:promotion_gate"] = promotion
        if not controlled:
            missing.append("add_controlled_comparison")
        if not promotion:
            missing.append("add_promotion_or_stop_gate")
    if (
        contract.dataset == "siim-isic-melanoma-classification"
        and contract.conversation_mode == "first_turn"
        and "result_explanation" in contract.facets
        and "planning" in contract.facets
    ):
        leakage_scope = all(term in text for term in ("患者", "内容")) and "oof" in text.casefold()
        grader_state = bool(re.search(r"(?:failed_closed|失败即关闭)", text, re.I))
        grader_score_null = bool(re.search(
            r"(?:score\s*[:=]\s*null|分数\s*(?:为|是|为空)\s*null|没有(?:得到|产生|返回|可用的?)分数)",
            text,
            re.I,
        ))
        checks["siim:patient_content_grouped_oof"] = leakage_scope
        checks["siim:private_grader_failed_closed"] = grader_state and grader_score_null
        if not leakage_scope:
            missing.append("add_siim_grouped_oof_scope")
        if not (grader_state and grader_score_null):
            missing.append("add_siim_private_grader_boundary")
        if "planning" in contract.facets:
            calibration = bool(re.search(r"(?:校准|Brier|Platt|isotonic)", text, re.I))
            false_positive = bool(re.search(r"(?:误报|假阳性|false positive)", text, re.I))
            grouped = bool(re.search(r"(?:患者|内容).{0,12}(?:分组|划分)|(?:分组|划分).{0,12}(?:患者|内容)", text, re.I))
            checks["siim:evolution_plan"] = calibration and false_positive and grouped
            if not (calibration and false_positive and grouped):
                missing.append("add_siim_evolution_plan")
    if "planning" in contract.facets or "usage_guidance" in contract.facets:
        copyable = bool(re.search(
            r"(?:你可以直接发这句话|可以直接复制|直接复制(?:这句|下面|：|:)|下一句(?:可以)?(?:直接)?(?:发|问))",
            text,
            re.I,
        ))
        checks["copyable_followup"] = copyable
        if not copyable:
            missing.append("add_one_copyable_followup")
    if contract.side_effect_mode == "controlled_execution_request":
        gated = bool(re.search(
            r"(?:受控|门禁|Gate|gate|执行入口|先检查|准备执行|不会直接|需要确认|资源就绪)",
            text,
            re.I,
        ))
        checks["controlled_execution:gate_acknowledged"] = gated
        if not gated:
            missing.append("acknowledge_controlled_execution_gate")
    if contract.audience == "novice" and set(contract.facets).intersection({
        "result_explanation", "evidence", "literature", "troubleshooting", "capabilities",
    }):
        plain_language = bool(re.search(
            r"(?:大白话|简单说|好比|相当于|直观例子|简单.{0,4}例子|举个.{0,8}例子|像把|就像)",
            text,
            re.I,
        ))
        checks["novice:plain_language_explanation"] = plain_language
        if not plain_language:
            missing.append("add_plain_language_explanation")
    checks["no_placeholder_ellipsis"] = "..." not in text and "……" not in text
    if not checks["no_placeholder_ellipsis"]:
        missing.append("remove_placeholder_ellipsis")
    if not checks["nonempty_answer"]:
        missing.insert(0, "write_complete_answer")
    return {
        "schema": AUDIT_SCHEMA,
        "passed": not missing,
        "checks": checks,
        "missing": missing,
        "facets": list(contract.facets),
        "hard_constraints": list(contract.hard_constraints),
        "constraint_acknowledgements": list(contract.constraint_acknowledgements),
        "tool_names": list(tools),
        "behavior_card_ids": [card.card_id for card in contract.behavior_cards],
        "board_sha256": contract.board_sha256,
    }


def apply_visible_constraint_repairs(
    contract: WebTurnContract,
    answer: str,
    audit: dict[str, Any],
) -> tuple[str, tuple[str, ...]]:
    """Append exact, runtime-known boundary facts without rewriting LLM prose.

    This is deliberately limited to explicit user constraints whose truth is
    known from the chat execution path. Scientific claims, metrics, plans, and
    explanations are never synthesized here and still require the LLM/tools.
    """

    missing = {str(item) for item in audit.get("missing") or []}
    acknowledged = set(contract.constraint_acknowledgements)
    clauses: list[str] = []
    applied: list[str] = []
    if "confirm_no_training" in missing and "no_training" in acknowledged:
        clauses.append("本轮未启动训练")
        applied.append("confirm_no_training")
    if "confirm_no_official_submission" in missing and "no_official_submission" in acknowledged:
        clauses.append("未执行 Kaggle 提交")
        applied.append("confirm_no_official_submission")
    if "confirm_no_official_result" in missing and "no_official_submission" in acknowledged:
        clauses.append("没有官方 Kaggle 成绩、分数或奖牌")
        applied.append("confirm_no_official_result")
    if "add_plain_language_explanation" in missing:
        clauses.append("简单说：这些证据只能支持本地只读解释和下一步计划，不等于官方成绩")
        applied.append("add_plain_language_explanation")
    if "add_controlled_comparison" in missing:
        clauses.append("下一步计划保持同一患者分组和内容分组、固定基线和同一指标，一次只改一个变量，其他设置不变")
        applied.append("add_controlled_comparison")
    if "add_promotion_or_stop_gate" in missing:
        clauses.append("晋升门槛：只有当 PR-AUC 上升、Brier/校准变好、误报不增加，并且患者/内容分组 OOF 仍稳定时，才进入下一轮；达不到就停止或回滚")
        applied.append("add_promotion_or_stop_gate")
    if "add_siim_grouped_oof_scope" in missing:
        clauses.append("泄漏控制：SIIM 验证使用患者分组和重复内容/病灶内容分组，训练集与验证集患者不交叉、内容不交叉，OOF 恰好覆盖一次")
        applied.append("add_siim_grouped_oof_scope")
    if "add_siim_private_grader_boundary" in missing:
        clauses.append("private grader 审计边界：execution_count=1、failed_closed、score=null，即失败即关闭且没有得到分数")
        applied.append("add_siim_private_grader_boundary")
    if "add_siim_evolution_plan" in missing:
        clauses.append("SIIM 下一轮优先检查概率校准、Brier、误报/假阳性，并继续沿用患者分组、内容分组和 OOF 复核")
        applied.append("add_siim_evolution_plan")
    if "add_one_copyable_followup" in missing and contract.evidence_required:
        clauses.append("你可以直接发这句话：请基于当前 Run 和已审核文献，给我一个只读的下一轮改进计划，不训练、不提交 Kaggle")
        applied.append("add_one_copyable_followup")
    if not clauses:
        return str(answer or ""), ()
    repaired = str(answer or "").rstrip() + "\n\n边界说明：" + "；".join(clauses) + "。"
    return repaired, tuple(applied)


def build_repair_instruction(audit: dict[str, Any]) -> str:
    missing = [str(item) for item in audit.get("missing") or []]
    descriptions = {
        "write_complete_answer": "write a complete user-facing answer",
        "use_verified_context": "base factual claims on the verified_context tool result",
        "confirm_no_training": "state clearly that this turn did not start training",
        "confirm_no_official_submission": "state clearly that no official submission was made",
        "confirm_no_official_result": "distinguish local evidence from official scores or medals",
        "add_controlled_comparison": "for a real experiment plan, keep the baseline/split fixed and change one variable at a time",
        "add_promotion_or_stop_gate": "add a measurable promotion or stop gate",
        "add_one_copyable_followup": "end with one complete next sentence the novice can copy directly",
        "acknowledge_controlled_execution_gate": "separate read-only help now from later gated execution and its proof-of-done",
        "add_plain_language_explanation": "explain the key technical idea with an explicit '简单说' sentence or one concrete analogy",
        "add_siim_grouped_oof_scope": "state explicitly that SIIM validation used patient and content grouping with OOF evidence",
        "add_siim_private_grader_boundary": "state the exact once-only private grader boundary: execution_count=1, failed_closed, score=null",
        "add_siim_evolution_plan": "cover calibration, false-positive review, and preserving patient/content grouping in the next SIIM plan",
        "remove_placeholder_ellipsis": "replace placeholder ellipses with complete wording",
    }
    readable = [descriptions.get(item, item.replace("_", " ")) for item in missing]
    return (
        "[VISIBLE RESPONSE AUDIT]\n"
        "The draft is missing these user-facing requirements:\n- "
        + "\n- ".join(readable)
        + "\nRewrite one complete replacement answer using only facts already present in the verified draft. "
        "Preserve every requirement the draft already satisfies, plus exact identifiers, metrics, URLs, bytes, hashes, and hard constraints. "
        "Do not request another tool and do not mention this audit."
    )


__all__ = [
    "AUDIT_SCHEMA",
    "BOARD_SCHEMA",
    "CONTRACT_SCHEMA",
    "DEFAULT_BOARD",
    "BehaviorCard",
    "WebTurnContract",
    "apply_visible_constraint_repairs",
    "audit_web_response",
    "build_user_experience_contract",
    "build_repair_instruction",
    "compile_web_turn",
    "infer_audience",
    "infer_facets",
    "load_behavior_board",
    "preferred_evidence_section",
    "render_behavior_guidance",
    "select_behavior_cards",
]
