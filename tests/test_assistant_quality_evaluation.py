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
from xsci import assistant_quality_fixture
from xsci.assistant_context import build_assistant_context


ROOT = Path(__file__).resolve().parents[1]


def _case(suite: dict, case_id: str) -> dict:
    case = next(dict(item) for item in suite["cases"] if item["case_id"] == case_id)
    case["pass_threshold"] = suite["pass_threshold"]
    case["allowed_tool_names"] = suite["allowed_tool_names"]
    return case


def _good_answer() -> str:
    return (
        "结论：Run evomind_siim_isic_quality_fixture_v1 已完成。"
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
    assert suite["version"] == 5
    assert suite["evidence_fixture"]["fixture_id"] == "siim_quality_fixture_v1"
    assert len(suite["cases"]) == 5
    assert suite["required_provider"] == "openai"
    # The gate measures the model the production web node actually serves.
    assert suite["required_model"] == "deepseek-flash"
    assert suite["required_route"]["base_url"] == "https://api.pezayo.com/v1"
    assert suite["required_route"]["wire_protocol"] == "chat_completions"
    assert suite["pass_threshold"] == 0.85
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
    assert bound["OPENAI_MODEL"] == "deepseek-flash"
    assert bound["OPENAI_BASE_URL"] == "https://api.pezayo.com/v1"
    assert bound["EVOMIND_MODEL_WIRE_PROTOCOL"] == "chat_completions"
    assert original["EVOLUTION_PRIMARY_PROVIDER"] == "deepseek"


def test_live_quality_evaluation_overrides_ambient_base_url_with_suite_route() -> None:
    suite = load_suite(ROOT / "configs" / "evaluation" / "assistant_novice_v1.json")
    bound = assistant_quality_evaluation._bind_quality_provider_env(
        {"OPENAI_BASE_URL": "https://example.invalid/v1", "OPENAI_API_KEY": "test-route-key"},
        suite,
    )
    assert bound["OPENAI_BASE_URL"] == "https://api.pezayo.com/v1"
    assert bound["OPENAI_API_KEY"] == "test-route-key"


def test_quality_hashes_ignore_checkout_line_endings(tmp_path) -> None:
    for relative in assistant_quality_evaluation.QUALITY_IMPLEMENTATION_PATHS:
        source = ROOT / relative
        lf = source.read_bytes().replace(b"\r\n", b"\n")
        for name, payload in (("lf", lf), ("crlf", lf.replace(b"\n", b"\r\n"))):
            target = tmp_path / name / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
    assert implementation_hashes(tmp_path / "lf") == implementation_hashes(tmp_path / "crlf")
    assert all(implementation_hashes(tmp_path / "lf").values())

    suite_source = (ROOT / "configs" / "evaluation" / "assistant_novice_v1.json").read_bytes().replace(b"\r\n", b"\n")
    (tmp_path / "suite_lf.json").write_bytes(suite_source)
    (tmp_path / "suite_crlf.json").write_bytes(suite_source.replace(b"\n", b"\r\n"))
    assert load_suite(tmp_path / "suite_lf.json")["suite_sha256"] == load_suite(tmp_path / "suite_crlf.json")["suite_sha256"]


def test_quality_hashes_still_detect_content_changes(tmp_path) -> None:
    for relative in assistant_quality_evaluation.QUALITY_IMPLEMENTATION_PATHS:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    before = implementation_hashes(tmp_path)
    changed = tmp_path / "src" / "xsci" / "assistant_stream.py"
    changed.write_bytes(changed.read_bytes() + b"\n# changed\n")
    after = implementation_hashes(tmp_path)
    assert before["src/xsci/assistant_stream.py"] != after["src/xsci/assistant_stream.py"]


def test_live_quality_evaluation_binds_loopback_gateway_credential(tmp_path) -> None:
    # A suite without a pinned remote route keeps using the loopback gateway.
    suite = dict(load_suite(ROOT / "configs" / "evaluation" / "assistant_novice_v1.json"))
    suite.pop("required_route")
    config = tmp_path / "gateway.json"
    config.write_text(json.dumps({"api-keys": ["test-loopback-token"]}), encoding="utf-8")

    bound = assistant_quality_evaluation._bind_quality_provider_env(
        {
            "OPENAI_BASE_URL": "http://127.0.0.1:65068/v1",
            "OPENAI_API_KEY": "test-unrelated-cloud-key",
            "EVOMIND_LOCAL_GATEWAY_CONFIG": str(config),
        },
        suite,
    )

    assert bound["OPENAI_API_KEY"] == "test-loopback-token"
    assert bound["OPENAI_BASE_URL"] == "http://127.0.0.1:65068/v1"


def test_scorer_passes_grounded_answer_and_fails_stale_metadata() -> None:
    suite = load_suite(ROOT / "configs" / "evaluation" / "assistant_novice_v1.json")
    case = _case(suite, "siim_results_for_novice")
    good = score_response(case, answer=_good_answer(), tool_names=["experiment_results"])
    stale = score_response(
        case,
        answer=_good_answer() + " house_prices SalePrice",
        tool_names=["experiment_results"],
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

    accepted = score_response(case, answer=equivalent, tool_names=["experiment_results"])
    rejected = score_response(case, answer=drifted, tool_names=["experiment_results"])

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
        tool_names=["experiment_results"],
        provider="deepseek",
        model="deepseek-v4-flash",
    )

    assert wrong["passed"] is False
    assert wrong["fatal_gate"]["provider_ok"] is False
    assert wrong["fatal_gate"]["model_ok"] is False
    assert wrong["fatal_gate"]["required_provider"] == "openai"
    assert wrong["fatal_gate"]["required_model"] == "deepseek-flash"


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
            runtime.store.append_event(session_id, "web.tool_started", {"tool": "experiment_results"})
            runtime.store.append_event(session_id, "web.usage", {
                "provider": "openai", "model": "deepseek-flash", "input_tokens": 10, "output_tokens": 20,
            })
            runtime.store.append_event(session_id, "web.answer_completed", {
                "provider": "openai", "model": "deepseek-flash", "llm_status": "completed",
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
            "execution": {"provider": "openai", "model": "deepseek-flash"},
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
        "evidence_fixture": {
            key: value
            for key, value in assistant_quality_fixture.materialize(tmp_path / "fixture").items()
            if key != "root"
        },
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


def test_quality_verifier_rejects_report_from_a_different_fixture(tmp_path) -> None:
    suite_path = ROOT / "configs" / "evaluation" / "assistant_novice_v1.json"
    suite = load_suite(suite_path)
    from scripts.verify_evomind_assistant_quality import _fixture_matches

    summary = assistant_quality_fixture.materialize(tmp_path / "fixture")
    good = {"evidence_fixture": {k: v for k, v in summary.items() if k != "root"}}
    stale = {"evidence_fixture": {**good["evidence_fixture"], "content_sha256": "0" * 64}}
    assert _fixture_matches(suite, good) is True
    assert _fixture_matches(suite, stale) is False
    assert _fixture_matches(suite, {}) is False


def test_evidence_fixture_is_deterministic_and_pinned_by_the_suite(tmp_path) -> None:
    first = assistant_quality_fixture.materialize(tmp_path / "a")
    second = assistant_quality_fixture.materialize(tmp_path / "b")
    assert first["content_sha256"] == second["content_sha256"]

    suite = load_suite(ROOT / "configs" / "evaluation" / "assistant_novice_v1.json")
    deliverables = _case(suite, "deliverables_for_novice")
    checks = {item["check_id"]: item for item in deliverables["checks"]}
    index = {row["name"]: row for row in assistant_quality_fixture.deliverable_index()}
    assert set(checks["all_files"]["all_of"]) == set(index)
    assert sorted(checks["hashes"]["all_of"]) == sorted(row["sha256"][:12] for row in index.values())
    sizes = {f"{row['bytes']:,}" for row in index.values()} | {str(row["bytes"]) for row in index.values()}
    assert set(checks["sizes"]["any_of"]) <= sizes
    results = _case(suite, "siim_results_for_novice")
    identity = next(item for item in results["checks"] if item["check_id"] == "run_identity")
    assert identity["all_of"] == [assistant_quality_fixture.RUN_ID]


def test_evidence_fixture_exposes_verified_evidence_to_the_assistant(tmp_path) -> None:
    root = tmp_path / "workstation"
    assistant_quality_fixture.materialize(root)
    packet = build_assistant_context(root, selected_task=assistant_quality_fixture.TASK_ID)
    run = packet.current_run

    assert run["available"] is True
    assert run["run_id"] == assistant_quality_fixture.RUN_ID
    assert round(run["metrics"]["roc_auc"], 4) == 0.9225
    assert round(run["metrics"]["pr_auc"], 4) == 0.2397
    assert round(run["metrics"]["brier"], 4) == 0.2952
    assert run["review"]["checks"]["patient_group_overlap_zero"] is True
    assert run["review"]["checks"]["content_group_overlap_zero"] is True
    assert run["private_grader"] == {
        "status": "failed_closed",
        "score": None,
        "execution_count": 1,
        "feedback_used_for_tuning": False,
        "official_submission_executed": False,
    }
    deliverables = run["artifacts"]["deliverables"]
    assert len(deliverables) == 4 and all(item["verified"] for item in deliverables)
    assert all(item["download_url"].startswith("/api/multi-agent/runs/") for item in deliverables)
    dois = {paper["doi"] for paper in packet.evidence["literature"]["papers"]}
    assert {"10.1111/jdv.20479", "10.1016/j.media.2021.102305"} <= dois
    assert packet.evidence["citation_audits"][0]["status"] == "passed"

    snapshot = governance_snapshot(root, run_id=assistant_quality_fixture.RUN_ID)
    assert snapshot["grader_execution_count"] == 1
    assert snapshot["grader_outcome"] == "failed_closed"
    assert snapshot["official_submission_executed"] is False


def test_evidence_fixture_refuses_a_non_empty_target(tmp_path) -> None:
    (tmp_path / "keep.txt").write_text("operator data", encoding="utf-8")
    try:
        assistant_quality_fixture.materialize(tmp_path)
    except ValueError:
        pass
    else:
        raise AssertionError("materialize must not write into a non-empty directory")
    assert (tmp_path / "keep.txt").read_text(encoding="utf-8") == "operator data"


def test_live_case_timeout_is_scored_as_failed_not_crashed(monkeypatch, tmp_path) -> None:
    import subprocess

    suite = load_suite(ROOT / "configs" / "evaluation" / "assistant_novice_v1.json")

    def fake_run(cmd, **kwargs):
        assert kwargs["env"]["EVOMIND_WORKSTATION_ROOT"] == str(tmp_path.resolve())
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"], output=b"", stderr=b"")

    monkeypatch.setattr(assistant_quality_evaluation.subprocess, "run", fake_run)
    result = assistant_quality_evaluation.run_live_case(
        tmp_path,
        suite,
        case_id="grouped_validation_for_novice",
        timeout_seconds=5,
        source_root=ROOT,
    )
    assert result["passed"] is False
    assert result["process"]["timed_out"] is True
    assert result["execution"]["llm_status"] == "timeout"
