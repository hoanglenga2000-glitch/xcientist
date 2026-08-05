from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from xsci.assistant_behavior_distillation import (
    BOARD_SCHEMA,
    apply_visible_constraint_repairs,
    audit_web_response,
    build_repair_instruction,
    build_user_experience_contract,
    compile_web_turn,
    load_behavior_board,
)


def _load_distiller():
    path = Path(__file__).resolve().parents[1] / "scripts" / "distill_assistant_interaction_patterns.py"
    spec = importlib.util.spec_from_file_location("distill_assistant_interaction_patterns", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_behavior_board_is_versioned_and_never_retains_raw_teacher_content():
    cards, digest = load_behavior_board()

    assert len(cards) >= 6
    assert len(digest) == 64
    assert all(card.provenance["raw_content_retained"] is False for card in cards)
    assert any(card.card_id == "goal_compile_for_novice_v1" for card in cards)
    assert any(card.card_id == "evidence_before_claim_v1" for card in cards)
    assert any(card.card_id == "novice_mission_control_v1" for card in cards)


def test_compiler_retrieves_behavior_cards_and_preserves_hard_constraints():
    contract = compile_web_turn(
        "我是小白，先不要训练也不要提交 Kaggle，帮我看上次结果、证据和下一步怎么改。",
        history=[{"role": "user", "content": "上次实验怎么样"}],
    )
    payload = contract.to_prompt_dict()

    assert payload["schema"] == "evomind.web_request_compiler.v3"
    assert payload["audience"] == "novice"
    assert payload["conversation_mode"] == "follow_up"
    assert payload["evidence_required"] is True
    assert payload["preferred_evidence_section"] == "metrics"
    assert payload["required_evidence_sections"] == ["metrics"]
    assert set(payload["hard_constraints"]) >= {"no_training", "no_official_submission"}
    assert set(payload["constraint_acknowledgements"]) >= {"no_training", "no_official_submission"}
    assert "follow_up_delta_v1" in payload["behavior_card_ids"]
    assert "evidence_before_claim_v1" in payload["behavior_card_ids"]
    assert payload["user_experience_contract"]["schema"] == "evomind.user_experience_contract.v1"
    assert payload["user_experience_contract"]["copyable_followup_required"] is True
    assert payload["user_experience_contract"]["required_final_line_prefix"] == "你可以直接发这句话："


def test_followup_inherits_prior_constraints_and_task_without_repeating_prompt():
    contract = compile_web_turn(
        "那下一步呢？",
        history=[
            {
                "role": "user",
                "content": "帮我看 SIIM 结果并计划改进，先不要训练，也不要提交 Kaggle。",
            },
            {"role": "assistant", "content": "已经解释了基线结果。"},
        ],
    )

    assert contract.dataset == "siim-isic-melanoma-classification"
    assert set(contract.hard_constraints) >= {"no_training", "no_official_submission"}
    assert contract.conversation_mode == "follow_up"
    assert any(card.card_id == "follow_up_delta_v1" for card in contract.behavior_cards)
    ux = contract.to_prompt_dict()["user_experience_contract"]
    assert ux["experiment_plan_contract"] == {
        "max_steps": 3,
        "controlled_comparison": "same patient/content split and baseline; change one variable at a time",
        "promotion_or_stop_gate_required": True,
    }


def test_followup_shorthand_requires_controlled_experiment_plan_repair():
    contract = compile_web_turn(
        "那就根据刚才的结果告诉我下一步怎么改吧，简单一点。",
        history=[
            {
                "role": "user",
                "content": "我是小白，帮我看上次 SIIM 结果并计划怎么提高，先不要训练，也不要提交 Kaggle。",
            },
            {"role": "assistant", "content": "已说明基线结果。"},
        ],
    )
    incomplete = audit_web_response(
        contract,
        "下一步先检查误报，再设一个晋升门槛。你可以直接发这句话：继续。",
        tool_names=["verified_context"],
    )
    complete = audit_web_response(
        contract,
        (
            "依据上次已验证结果指标，下一步最多做三个改进步骤：保持同一患者和内容分组、固定基线，"
            "一次只改一个变量；只有通过晋升门槛才进入下一轮。"
            "本轮未启动训练，未执行 Kaggle 提交，也没有官方 Kaggle 成绩、分数或奖牌。"
            "你可以直接发这句话：请按同一分组做单变量对照计划。"
        ),
        tool_names=["verified_context"],
    )

    assert "planning" in contract.facets
    assert "add_controlled_comparison" in incomplete["missing"]
    assert complete["passed"] is True


def test_siim_comprehensive_result_review_requires_grouped_oof_grader_and_plan():
    contract = compile_web_turn(
        "我是小白，帮我理解上次 SIIM 实验结果、证据、官方成绩、交付物和下一步进化计划。"
    )
    ux = contract.to_prompt_dict()["user_experience_contract"]
    incomplete = audit_web_response(
        contract,
        (
            "结果指标已经复核，下一步保持固定基线做单变量比较并设晋升门槛。"
            "没有官方 Kaggle 成绩。"
        ),
        tool_names=["verified_context"],
    )
    complete = audit_web_response(
        contract,
        (
            "简单说，结果依据患者分组、内容分组的 OOF 复核。private grader 审计为 "
            "execution_count=1、failed_closed、score=null。"
            "交付物包含报告、结果表、代码包和证据包，可从已验证下载链接获取。"
            "下一步最多三项：保持患者与内容分组，先做概率校准，再复核误报和假阳性；"
            "固定基线一次只改一个变量，只有通过晋升门槛才继续。"
            "没有官方 Kaggle 成绩、分数或奖牌。"
            "你可以直接发这句话：请按同一分组制定校准和误报复核计划。"
        ),
        tool_names=["verified_context"],
    )

    assert ux["task_specific_evidence_contract"] == {
        "validation_scope_terms": ["患者", "内容", "OOF"],
        "private_grader_terms": ["execution_count=1", "failed_closed", "score=null"],
        "next_plan_terms": ["校准", "误报/假阳性", "分组"],
    }
    assert "add_siim_grouped_oof_scope" in incomplete["missing"]
    assert "add_siim_private_grader_boundary" in incomplete["missing"]
    assert "add_siim_evolution_plan" in incomplete["missing"]
    assert complete["passed"] is True


def test_multisource_question_prefetches_literature_and_current_run_metrics():
    contract = compile_web_turn(
        "请结合当前实验的真实证据和参考文献解释患者分组验证。"
    )

    assert contract.preferred_evidence_section == "literature"
    assert contract.required_evidence_sections == ("literature", "metrics")


def test_implicit_siim_governance_does_not_force_unrelated_boundary_boilerplate():
    contract = compile_web_turn("我是小白，为什么 SIIM 要按患者分组？请结合证据解释。")
    audit = audit_web_response(
        contract,
        "结论：简单说，同一患者不能跨训练和验证两边，否则会造成泄漏和指标虚高。证据来自当前 Run。",
        tool_names=["verified_context"],
    )

    assert "no_official_submission" in contract.hard_constraints
    assert "no_official_submission" not in contract.constraint_acknowledgements
    assert "confirm_no_official_submission" not in audit["missing"]
    assert "confirm_no_official_result" not in audit["missing"]
    assert audit["passed"] is True


def test_visible_response_audit_finds_missing_facets_then_accepts_complete_answer():
    contract = compile_web_turn(
        "我是小白，不要训练，不要提交 Kaggle。先解释结果和证据，再给下一步。"
    )
    incomplete = audit_web_response(contract, "结果还可以。", tool_names=[])
    complete = audit_web_response(
        contract,
        (
            "结论：简单说，结果指标已有提升，证据来自已验证 Run。"
            "下一步保持同一分组做单变量比较并设晋升门槛。"
            "本轮未启动训练，也未提交 Kaggle，并且没有官方 Kaggle 成绩。"
            "你可以直接复制下一句继续。"
        ),
        tool_names=["verified_context"],
    )

    assert incomplete["passed"] is False
    assert "use_verified_context" in incomplete["missing"]
    assert "confirm_no_training" in incomplete["missing"]
    assert complete["passed"] is True


def test_novice_execution_request_is_gated_and_not_treated_like_template_chat():
    contract = compile_web_turn("我是小白，帮我做一次实验，先检查能不能跑。")
    payload = contract.to_prompt_dict()
    ux = build_user_experience_contract(contract)

    assert payload["schema"] == "evomind.web_request_compiler.v3"
    assert payload["side_effect_mode"] == "controlled_execution_request"
    assert ux["proof_of_done_required"] is True
    assert any("Execution/training" in item for item in ux["gated_later"])
    assert any(card.card_id == "novice_mission_control_v1" for card in contract.behavior_cards)

    weak = audit_web_response(contract, "结论：可以开始。", tool_names=[])
    strong = audit_web_response(
        contract,
        "结论：这是受控执行请求，不会直接假装训练已经开始。先检查数据、资源就绪和门禁；proof-of-done 是生成可复核的运行记录。你可以直接发这句话：先做只读资源检查。",
        tool_names=[],
    )

    assert "acknowledge_controlled_execution_gate" in weak["missing"]
    assert strong["passed"] is True


def test_past_experiment_artifact_request_stays_read_only_and_preserves_negation():
    contract = compile_web_turn(
        "我是小白，上次实验最后留下了哪些文件？给下载链接，不要启动任何训练。"
    )

    assert contract.side_effect_mode == "read_only_reasoning"
    assert "planning" not in contract.facets
    assert "no_training" in contract.hard_constraints
    assert "train" not in contract.actions


def test_novice_explanation_does_not_invent_an_experiment_plan_requirement():
    contract = compile_web_turn(
        "我是小白，为什么 SIIM 实验要按患者分组？结合证据和文献说明，最后告诉我下一句怎么问。"
    )
    audit = audit_web_response(
        contract,
        (
            "结论：简单说，同一患者不能同时出现在训练和验证两边，否则会泄漏并让指标虚高。"
            "证据来自当前 Run，参考文献也支持按患者分组。未执行 Kaggle 提交，也没有官方 Kaggle 成绩。"
            "下一句可以直接问：请继续解释内容组去重。"
        ),
        tool_names=["verified_context"],
    )

    assert "add_controlled_comparison" not in audit["missing"]
    assert "add_promotion_or_stop_gate" not in audit["missing"]
    assert audit["passed"] is True


def test_siim_official_boundary_requires_submission_and_score_facts_separately():
    contract = compile_web_turn("我是小白，解释 SIIM 结果和证据，有没有官方 Kaggle 成绩？")
    weak = audit_web_response(
        contract,
        "简单说，结果有证据。没有提交，也没有官方成绩。",
        tool_names=["verified_context"],
    )
    strong = audit_web_response(
        contract,
        "简单说，结果有证据。未执行 Kaggle 提交，也没有官方 Kaggle 成绩。",
        tool_names=["verified_context"],
    )
    equivalent_word_order = audit_web_response(
        contract,
        "简单说，结果有证据。未执行官方 Kaggle 提交，也没有官方 Kaggle 成绩。",
        tool_names=["verified_context"],
    )
    predicate_last = audit_web_response(
        contract,
        "简单说，结果有证据。官方提交没有执行，也没有官方 Kaggle 成绩。",
        tool_names=["verified_context"],
    )

    assert "confirm_no_official_submission" in weak["missing"]
    assert "confirm_no_official_result" in weak["missing"]
    assert strong["passed"] is True
    assert equivalent_word_order["passed"] is True
    assert predicate_last["passed"] is True


def test_visible_constraint_repair_only_appends_explicit_runtime_boundaries():
    contract = compile_web_turn(
        "我是小白，解释 SIIM 结果；不要启动训练，也不要提交 Kaggle，有没有官方成绩？"
    )
    draft = "结论：简单说，结果指标和证据已经核对。"
    before = audit_web_response(contract, draft, tool_names=["verified_context"])
    repaired, applied = apply_visible_constraint_repairs(contract, draft, before)
    after = audit_web_response(contract, repaired, tool_names=["verified_context"])

    assert set(applied) == {
        "confirm_no_training",
        "confirm_no_official_submission",
        "confirm_no_official_result",
    }
    assert draft in repaired
    assert "本轮未启动训练" in repaired
    assert "未执行 Kaggle 提交" in repaired
    assert "没有官方 Kaggle 成绩" in repaired
    assert all(item not in after["missing"] for item in applied)


def test_repair_instruction_explains_audit_ids_in_user_language():
    instruction = build_repair_instruction({
        "missing": ["add_plain_language_explanation", "add_one_copyable_followup"],
    })

    assert "简单说" in instruction
    assert "copy directly" in instruction
    assert "Preserve every requirement" in instruction


def test_distiller_outputs_only_aggregate_signals_and_is_deterministic(tmp_path):
    module = _load_distiller()
    # Keep the fixture effective at runtime without embedding a scanner-shaped
    # credential literal in release source.
    secret = "sk-" + "secret-must-never-appear"
    export = tmp_path / "owned-chat.json"
    export.write_text(json.dumps({
        "messages": [
            {"role": "user", "content": "我是小白，下一步怎么做？"},
            {
                "role": "assistant",
                "content": f"结论：已验证。证据见 SHA-256。下一步按门槛验证。{secret}",
            },
        ],
        "private_grader": {"feedback": "must not be read"},
    }, ensure_ascii=False), encoding="utf-8")

    first = module.distill([export])
    second = module.distill([export])
    wire = json.dumps(first, ensure_ascii=False, sort_keys=True)

    assert first == second
    assert first["source_policy"]["raw_content_retained"] is False
    assert first["source_policy"]["private_grader_feedback_used"] is False
    assert first["unique_user_assistant_pairs"] == 1
    assert first["behavior_signal_counts"]["conclusion_first"] == 1
    assert secret not in wire
    assert "must not be read" not in wire
