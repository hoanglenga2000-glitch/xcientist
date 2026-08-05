from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from evomind_runtime import AgentRuntime
from xsci.assistant_quality_evaluation import (
    build_recorded_pair_report,
    governance_snapshot,
    implementation_hashes,
    load_suite,
    score_response,
)
from scripts.verify_evomind_assistant_quality import verify
from xsci import assistant_quality_evaluation


ROOT = Path(__file__).resolve().parents[1]


def _case(suite: dict, case_id: str) -> dict:
    case = next(dict(item) for item in suite["cases"] if item["case_id"] == case_id)
    case["pass_threshold"] = suite["pass_threshold"]
    case["allowed_tool_names"] = suite["allowed_tool_names"]
    return case


def _good_answer() -> str:
    return (
        "结论：Run evomind_siim_isic_a800_job90353_20260730_095826 已完成。"
        "ROC-AUC 0.9225，PR-AUC 0.2397，Brier 0.2952，患者分组 95% 置信区间已记录。"
        "患者与重复内容分组无交叉，OOF 覆盖一次。私有 grader 失败即关闭，没有得到分数，也没有用于调参。"
        "没有官方 Kaggle 成绩，未执行 Kaggle 提交。"
        "交付物：evomind-siim-isic-report.pdf、evomind-siim-isic-results.csv、"
        "evomind-siim-isic-code.zip、evomind-siim-isic-evidence.zip。"
        "下载 /api/multi-agent/runs/run/download/file。下一轮先做概率校准、误报分析并沿用患者内容分组。"
        "你可以直接发这句话：先做只读计划，不训练不提交。"
        + "证据说明。" * 90
    )


def test_versioned_suite_is_valid_and_weights_are_complete() -> None:
    suite = load_suite(ROOT / "configs" / "evaluation" / "assistant_novice_v1.json")
    assert suite["suite_id"] == "evomind_novice_research_agent_v1"
    assert suite["version"] == 2
    assert len(suite["cases"]) == 5
    assert suite["required_provider"] == "openai"
    assert suite["required_model"] == "gpt-5.6-sol"
    assert all(sum(float(check["weight"]) for check in case["checks"]) == 100 for case in suite["cases"])
    assert len(suite["suite_sha256"]) == 64


def test_live_quality_evaluation_pins_suite_provider_and_model_strictly() -> None:
    suite = load_suite(ROOT / "configs" / "evaluation" / "assistant_novice_v1.json")
    original = {
        "EVOLUTION_PRIMARY_PROVIDER": "deepseek",
        "DEEPSEEK_MODEL": "deepseek-v4-flash",
        "OPENAI_MODEL": "stale-model",
    }

    bound = assistant_quality_evaluation._bind_quality_provider_env(original, suite)

    assert bound["EVOLUTION_PRIMARY_PROVIDER"] == "openai"
    assert bound["EVOLUTION_PROVIDER_STRICT"] == "1"
    assert bound["OPENAI_MODEL"] == "gpt-5.6-sol"
    assert original["EVOLUTION_PRIMARY_PROVIDER"] == "deepseek"


def test_live_quality_evaluation_binds_loopback_gateway_credential(tmp_path) -> None:
    suite = load_suite(ROOT / "configs" / "evaluation" / "assistant_novice_v1.json")
    config = tmp_path / "gateway.json"
    config.write_text(json.dumps({"api-keys": ["test-loopback-token"]}), encoding="utf-8")

    bound = assistant_quality_evaluation._bind_quality_provider_env(
        {
            "OPENAI_BASE_URL": "http://127.0.0.1:65068/v1",
            "OPENAI_API_KEY": "unrelated-cloud-key",
            "EVOMIND_LOCAL_GATEWAY_CONFIG": str(config),
        },
        suite,
    )

    assert bound["OPENAI_API_KEY"] == "test-loopback-token"
    assert bound["OPENAI_BASE_URL"] == "http://127.0.0.1:65068/v1"


def test_scorer_passes_grounded_answer_and_fails_stale_metadata() -> None:
    suite = load_suite(ROOT / "configs" / "evaluation" / "assistant_novice_v1.json")
    case = _case(suite, "siim_results_for_novice")
    good = score_response(case, answer=_good_answer(), tool_names=["verified_context"])
    stale = score_response(
        case,
        answer=_good_answer() + " house_prices SalePrice",
        tool_names=["verified_context"],
    )
    assert good["passed"] is True
    assert good["score"] == 1.0
    assert stale["passed"] is False
    assert stale["fatal_gate"]["forbidden_hits"] == ["house_prices", "SalePrice"]
    assert "answer" not in json.dumps(good["answer"])


def test_scorer_accepts_equivalent_audit_wording_and_rejects_link_drift() -> None:
    suite = load_suite(ROOT / "configs" / "evaluation" / "assistant_novice_v1.json")
    case = _case(suite, "siim_results_for_novice")
    equivalent = _good_answer().replace(
        "失败即关闭，没有得到分数",
        "failed_closed，分数为 null",
    ).replace(
        "没有官方 Kaggle 成绩，未执行 Kaggle 提交",
        "official_submission_executed=false",
    )
    drifted = equivalent.replace(
        "/api/multi-agent/runs/run/download/file",
        "http://localhost/api/multi-agent/runs/.../download/file",
    )

    accepted = score_response(case, answer=equivalent, tool_names=["verified_context"])
    rejected = score_response(case, answer=drifted, tool_names=["verified_context"])

    assert accepted["passed"] is True
    assert rejected["passed"] is False
    assert rejected["fatal_gate"]["forbidden_hits"] == ["http://localhost"]
    assert rejected["fatal_gate"]["forbidden_regex_hits"] == [r"\.\.\."]


def test_scorer_fails_closed_when_live_provider_or_model_is_not_required_one() -> None:
    suite = load_suite(ROOT / "configs" / "evaluation" / "assistant_novice_v1.json")
    case = _case(suite, "siim_results_for_novice")
    case["required_provider"] = suite["required_provider"]
    case["required_model"] = suite["required_model"]

    wrong = score_response(
        case,
        answer=_good_answer(),
        tool_names=["verified_context"],
        provider="deepseek",
        model="deepseek-v4-flash",
    )

    assert wrong["passed"] is False
    assert wrong["fatal_gate"]["provider_ok"] is False
    assert wrong["fatal_gate"]["model_ok"] is False
    assert wrong["fatal_gate"]["required_provider"] == "openai"
    assert wrong["fatal_gate"]["required_model"] == "gpt-5.6-sol"


def test_recorded_pair_uses_runtime_ledger_and_proves_regression_fixed(tmp_path) -> None:
    suite = load_suite(ROOT / "configs" / "evaluation" / "assistant_novice_v1.json")
    prompt = next(item["prompt"] for item in suite["cases"] if item["case_id"] == "siim_results_for_novice")
    runtime = AgentRuntime(tmp_path)
    try:
        for session_id, answer in (
            ("baseline", "house_prices SalePrice，没有具体 AUC。" + "旧回答。" * 200),
            ("treatment", _good_answer()),
        ):
            runtime.create_session(objective=prompt, title="Web assistant", session_id=session_id)
            runtime.store.add_turn(session_id, "user", prompt)
            runtime.store.add_turn(session_id, "assistant", answer)
            runtime.store.append_event(session_id, "web.tool_started", {"tool": "verified_context"})
            runtime.store.append_event(session_id, "web.usage", {
                "provider": "openai", "model": "gpt-5.6-sol", "input_tokens": 10, "output_tokens": 20,
            })
            runtime.store.append_event(session_id, "web.answer_completed", {
                "provider": "openai", "model": "gpt-5.6-sol", "llm_status": "completed",
            })
    finally:
        runtime.close()

    report = build_recorded_pair_report(
        tmp_path,
        suite,
        case_id="siim_results_for_novice",
        baseline_session_id="baseline",
        treatment_session_id="treatment",
    )

    assert report["status"] == "passed"
    assert report["baseline"]["passed"] is False
    assert report["treatment"]["passed"] is True
    assert report["comparison"]["score_delta"] > 0
    assert report["comparison"]["regression_fixed"] is True
    assert report["comparison"]["claim_scope"].endswith("not_general_quality_proof")


def test_governance_snapshot_tracks_exactly_once_grader_without_private_data(tmp_path) -> None:
    run_id = "evomind_siim_isic_a800_job90353_20260730_095826"
    run_dir = tmp_path / "workspace" / "evomind_runs" / run_id
    run_dir.mkdir(parents=True)
    (tmp_path / "workspace" / "current_run.json").write_text("{}\n", encoding="utf-8")
    for name in ("private_grader.json", "claim_audit.json", "candidate_freeze.json"):
        (run_dir / name).write_text("{}\n", encoding="utf-8")
    (run_dir / "private_grader_ledger.json").write_text(json.dumps({
        "execution_count": 1,
        "outcome": "failed_closed",
        "score": None,
        "official_submission_executed": False,
    }), encoding="utf-8")

    snapshot = governance_snapshot(tmp_path)

    assert snapshot["siim_run_directories"] == [run_id]
    assert snapshot["grader_execution_count"] == 1
    assert snapshot["grader_outcome"] == "failed_closed"
    assert snapshot["grader_score"] is None
    assert snapshot["official_submission_executed"] is False
    assert "credential" not in json.dumps(snapshot).lower()


def test_quality_verifier_binds_suite_source_provider_governance_and_latency(tmp_path) -> None:
    suite_path = ROOT / "configs" / "evaluation" / "assistant_novice_v1.json"
    suite = load_suite(suite_path)
    governance = {
        "grader_execution_count": 1,
        "grader_outcome": "failed_closed",
        "grader_score": None,
        "official_submission_executed": False,
    }
    results = [
        {
            "case_id": item["case_id"],
            "passed": True,
            "prompt_matches_case": True,
            "execution": {"provider": "openai", "model": "gpt-5.6-sol"},
            "fatal_gate": {
                "provider_ok": True,
                "model_ok": True,
                "missing_tools": [],
                "unexpected_tools": [],
                "forbidden_hits": [],
                "forbidden_regex_hits": [],
            },
        }
        for item in suite["cases"]
    ]
    report = {
        "schema": "evomind.assistant_quality_evaluation.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "live_production_tool_loop",
        "status": "passed",
        "suite_id": suite["suite_id"],
        "suite_version": suite["version"],
        "suite_sha256": suite["suite_sha256"],
        "implementation_hashes": implementation_hashes(ROOT),
        "selected_cases": [item["case_id"] for item in suite["cases"]],
        "results": results,
        "aggregate": {
            "case_count": 5,
            "passed_cases": 5,
            "pass_rate": 1.0,
            "mean_score": 1.0,
            "latency_p95_seconds": 70.0,
        },
        "governance_invariants": {"passed": True, "before": governance, "after": governance},
    }
    report_path = tmp_path / "quality.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    accepted = verify(report_path, suite_path, max_age_hours=24, max_p95_seconds=90)
    report["results"][0]["execution"]["model"] = "deepseek-v4-flash"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    rejected = verify(report_path, suite_path, max_age_hours=24, max_p95_seconds=90)

    assert accepted["status"] == "passed"
    assert accepted["failed_checks"] == []
    assert rejected["status"] == "failed"
    assert "provider_model" in rejected["failed_checks"]
