from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pytest

from xsci.agentic_capability_benchmark import _materialize_fixture, build_benchmark_cases, run_benchmark


def _task_id(prompt: str) -> str:
    return prompt.splitlines()[0].split(":", 1)[1].strip()


def _write(workspace: Path, relative_path: str, content: str) -> None:
    path = workspace / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _solve(prompt: str, workspace: Path):
    case_id = _task_id(prompt)

    if case_id == "retrieval_exact_release_token":
        notes = (workspace / "docs/release_notes.txt").read_text(encoding="utf-8")
        token = re.search(r"^current_release_token=(.+)$", notes, re.MULTILINE).group(1)
        _write(workspace, "answer.txt", token + "\n")

    elif case_id == "retrieval_rank_valid_candidate":
        candidates = [json.loads(path.read_text(encoding="utf-8")) for path in (workspace / "candidates").glob("*.json")]
        selected = max((item for item in candidates if item["validated"]), key=lambda item: item["score"])
        _write(workspace, "selection.txt", selected["candidate_id"] + "\n")

    elif case_id == "cross_file_release_profile":
        settings_path = workspace / "app/settings.json"
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        settings["active_profile"] = "production"
        settings_path.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write(
            workspace,
            "docs/release.txt",
            f"{settings['project_id']}@{settings['version']}:production\n",
        )

    elif case_id == "cross_file_catalog_rename":
        request = json.loads((workspace / "requests/rename.json").read_text(encoding="utf-8"))
        catalog_path = workspace / "data/catalog.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        for item in catalog:
            if item["id"] == request["target_id"]:
                item["label"] = request["new_label"]
        catalog_path.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write(workspace, "docs/catalog.index", "".join(f"{item['id']}={item['label']}\n" for item in catalog))

    elif case_id == "recovery_repair_invalid_pipeline_json":
        pipeline_path = workspace / "config/pipeline.json"
        repaired = re.sub(r",\s*([}\]])", r"\1", pipeline_path.read_text(encoding="utf-8"))
        pipeline = json.loads(repaired)
        failure = (workspace / "logs/last_failure.txt").read_text(encoding="utf-8")
        pipeline["retries"] = int(re.search(r"required_retries=(\d+)", failure).group(1))
        pipeline_path.write_text(json.dumps(pipeline, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    elif case_id == "recovery_follow_failed_threshold":
        failure = (workspace / "tests/last_failure.log").read_text(encoding="utf-8")
        expected = re.search(r"expected threshold=(\d+)", failure).group(1)
        rules_path = workspace / "src/rules.conf"
        rules = re.sub(
            r"^threshold=.*$",
            f"threshold={expected}",
            rules_path.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
        rules_path.write_text(rules, encoding="utf-8")

    elif case_id == "constraint_modify_only_named_config":
        config_path = workspace / "config/app.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["mode"] = "safe"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    elif case_id == "constraint_single_sorted_output":
        identifiers = (workspace / "input/identifiers.txt").read_text(encoding="utf-8").splitlines()
        _write(workspace, "output/final.txt", "\n".join(sorted(identifiers)) + "\n")

    elif case_id == "evidence_validated_promotion_verdict":
        run = json.loads((workspace / "evidence/run.json").read_text(encoding="utf-8"))
        supported = (
            run["evidence_level"] == "validated"
            and run["run_success"] is True
            and run["promoted"] is True
            and run["metric_delta"] > 0
            and run["no_training_started"] is not True
        )
        _write(workspace, "verdict.txt", ("CLAIM_SUPPORTED" if supported else "CLAIM_UNSUPPORTED") + "\n")

    elif case_id == "evidence_cross_source_corroboration":
        source_a = json.loads((workspace / "evidence/source_a.json").read_text(encoding="utf-8"))
        source_b = json.loads((workspace / "evidence/source_b.json").read_text(encoding="utf-8"))
        agreed = (
            source_a["status"] == source_b["status"] == "completed"
            and source_a["run_id"] == source_b["run_id"]
            and source_a["artifact_hash"] == source_b["artifact_hash"]
        )
        _write(workspace, "verified.txt", (source_a["run_id"] if agreed else "INSUFFICIENT_EVIDENCE") + "\n")

    elif case_id == "memory_choose_trusted_positive_record":
        records = json.loads((workspace / "memory/index.json").read_text(encoding="utf-8"))
        trusted = [
            item for item in records
            if item["evidence_level"] == "validated"
            and item["run_success"] is True
            and item["promoted"] is True
            and item["metric_delta"] > 0
        ]
        selected = max(trusted, key=lambda item: item["metric_delta"])
        _write(workspace, "chosen_memory.txt", selected["memory_id"] + "\n")

    elif case_id == "memory_latest_reusable_validated_strategy":
        records = json.loads((workspace / "memory/history.json").read_text(encoding="utf-8"))
        trusted = [
            item for item in records
            if item["evidence_level"] == "validated"
            and item["run_success"] is True
            and item["promoted"] is True
            and isinstance(item["metric_delta"], (int, float))
            and item["metric_delta"] > 0
        ]
        selected = max(trusted, key=lambda item: item["recorded_at"])
        _write(workspace, "strategy.txt", selected["reusable_strategy"] + "\n")

    else:  # pragma: no cover - catches additions that need a real solver in this test.
        raise AssertionError(f"unhandled benchmark case: {case_id}")

    return {"success": True, "case_id": case_id}


def test_seeded_case_catalog_is_repeatable_and_covers_core_capabilities():
    first = build_benchmark_cases(seed=20260711)
    second = build_benchmark_cases(seed=20260711)

    assert first == second
    assert len(first) >= 12
    assert len({case.case_id for case in first}) == len(first)
    assert {case.category for case in first} >= {
        "retrieval",
        "cross_file_edit",
        "failure_recovery",
        "constraint_following",
        "evidence_judgment",
        "memory_trust",
    }


def test_fixture_materialization_is_byte_stable_on_windows(tmp_path):
    _materialize_fixture(tmp_path, {"config/value.txt": "alpha\nbeta\n"})
    assert (tmp_path / "config" / "value.txt").read_bytes() == b"alpha\nbeta\n"


def test_all_cases_are_scored_by_hidden_workspace_oracles_and_report_is_written(tmp_path):
    report = run_benchmark(_solve, seed=20260711, workspace_root=tmp_path)

    assert report["cases_run"] == report["total_cases"] >= 12
    assert report["passed_cases"] == report["cases_run"]
    assert report["task_success_rate"] == 1.0
    assert report["scope_violations"] == 0
    assert report["unsupported_claims"] == 0
    assert all(result["oracle_passed"] for result in report["case_results"])

    report_path = tmp_path / ".xsci" / "agentic_capability_benchmark.json"
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_runner_success_text_cannot_fool_oracle_and_is_counted_as_unsupported_claim():
    case_id = "retrieval_exact_release_token"
    report = run_benchmark(
        lambda prompt, workspace: "All tests passed",
        case_ids=[case_id],
    )

    assert report["cases_run"] == 1
    assert report["passed_cases"] == 0
    assert report["task_success_rate"] == 0.0
    assert report["unsupported_claims"] == 1
    assert report["case_results"][0]["oracle_passed"] is False
    assert "missing required file" in report["case_results"][0]["failure_reason"]


def test_correct_answer_with_out_of_scope_edit_fails_scope_contract():
    def violating_runner(prompt: str, workspace: Path):
        result = _solve(prompt, workspace)
        (workspace / "README.md").write_text("rewritten\n", encoding="utf-8")
        return result

    report = run_benchmark(
        violating_runner,
        case_ids=["constraint_modify_only_named_config"],
    )

    result = report["case_results"][0]
    assert result["oracle_passed"] is True
    assert result["passed"] is False
    assert result["scope_violation"] is True
    assert result["scope_violation_paths"] == ["README.md"]
    assert report["scope_violations"] == 1
    assert report["unsupported_claims"] == 1


def test_timeout_is_a_bounded_failure_with_reason():
    def slow_runner(prompt: str, workspace: Path):
        time.sleep(0.2)
        return {"success": True}

    report = run_benchmark(
        slow_runner,
        case_ids=["retrieval_exact_release_token"],
        timeout_seconds=0.01,
    )

    result = report["case_results"][0]
    assert result["passed"] is False
    assert result["timed_out"] is True
    assert "timed out" in result["failure_reason"]
    assert report["timed_out_cases"] == 1


def test_each_case_receives_a_distinct_isolated_workspace():
    workspaces: list[Path] = []

    def recording_runner(prompt: str, workspace: Path):
        workspaces.append(workspace)
        assert not (workspace / "sentinel-from-previous-case.txt").exists()
        (workspace / "sentinel-from-previous-case.txt").write_text("isolated\n", encoding="utf-8")

    report = run_benchmark(
        recording_runner,
        case_ids=["retrieval_exact_release_token", "retrieval_rank_valid_candidate"],
    )

    assert report["cases_run"] == 2
    assert len(workspaces) == 2
    assert workspaces[0] != workspaces[1]
    assert all(result["scope_violation"] for result in report["case_results"])


def test_seed_controls_hidden_fixture_data_reproducibly():
    def fingerprint_runner(prompt: str, workspace: Path):
        files = sorted(
            (path.relative_to(workspace).as_posix(), path.read_text(encoding="utf-8"))
            for path in workspace.rglob("*")
            if path.is_file()
        )
        return {"fixture": files}

    selected = ["retrieval_exact_release_token"]
    first = run_benchmark(fingerprint_runner, seed=77, case_ids=selected)
    repeated = run_benchmark(fingerprint_runner, seed=77, case_ids=selected)
    different = run_benchmark(fingerprint_runner, seed=78, case_ids=selected)

    first_output = first["case_results"][0]["runner_output"]
    assert first_output == repeated["case_results"][0]["runner_output"]
    assert first_output != different["case_results"][0]["runner_output"]


def test_runner_exception_and_unknown_case_have_explicit_failures():
    def broken_runner(prompt: str, workspace: Path):
        raise RuntimeError("runner exploded")

    report = run_benchmark(
        broken_runner,
        case_ids=["retrieval_exact_release_token"],
    )
    assert "runner raised RuntimeError: runner exploded" in report["case_results"][0]["failure_reason"]

    with pytest.raises(ValueError, match="unknown benchmark case ids"):
        run_benchmark(broken_runner, case_ids=["not-a-real-case"])
