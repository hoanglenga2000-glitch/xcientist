"""Tests for the benchmark manager (medal-rate / valid-submission accounting).

These calculations back the anti-overclaim rules: medal_rate and
valid_submission_rate must reflect only real, recorded results.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_os.benchmark_manager import (
    BenchmarkRegistryError,
    BenchmarkResult,
    compute_gap_to_target,
    compute_medal_rate,
    compute_valid_submission_rate,
    export_benchmark_report,
    load_tasks,
    save_result,
    summarize_results,
    validate_result_task_ids,
)

ROOT = Path(__file__).resolve().parents[1]
SPLIT75 = ROOT / "benchmark" / "mle_bench_75" / "openai_split75_507f92e.txt"


def _write_registry(tmp_path: Path, name: str, payload: object) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / SPLIT75.name).write_bytes(SPLIT75.read_bytes())
    return path


def _result(task_id: str, *, medal: str = "none", valid: bool = True, failed: int = 0,
            top30: bool = False) -> BenchmarkResult:
    return BenchmarkResult(
        task_id=task_id,
        best_exp_id=f"exp_{task_id}",
        best_cv_score=0.5,
        best_public_score=0.5,
        best_private_score=None,
        valid_submission=valid,
        medal=medal,
        rank_percentile=None,
        runtime_hours=1.0,
        num_experiments=3,
        num_failed_runs=failed,
        num_recoveries=0,
        reproducibility_score=1.0,
        auditability_score=1.0,
        claim_drift_detected=False,
        final_report_path=None,
        artifacts_path=None,
        gap_to_mlevolve=None,
        next_improvement_plan="",
        top30_reached=top30,
    )


def test_empty_rates_are_zero():
    assert compute_medal_rate([]) == 0.0
    assert compute_valid_submission_rate([]) == 0.0


def test_medal_rate_counts_only_medals():
    results = [
        _result("a", medal="bronze"),
        _result("b", medal="silver"),
        _result("c", medal="gold"),
        _result("d", medal="none"),
    ]
    assert compute_medal_rate(results) == 0.75


def test_unknown_medal_does_not_count():
    results = [_result("a", medal="unknown"), _result("b", medal="bronze")]
    assert compute_medal_rate(results) == 0.5


def test_valid_submission_rate():
    results = [_result("a", valid=True), _result("b", valid=False), _result("c", valid=True)]
    assert abs(compute_valid_submission_rate(results) - 2 / 3) < 1e-9


def test_gap_to_target():
    assert compute_gap_to_target(0.2, 0.5) == pytest.approx(0.3)
    assert compute_gap_to_target(0.6, 0.5) == pytest.approx(-0.1)


def test_summarize_results_structure():
    results = [
        _result("a", medal="bronze", top30=True),
        _result("b", medal="none", valid=False, failed=1),
    ]
    summary = summarize_results(results)
    assert summary["evaluated_tasks"] == 2
    assert summary["medal_rate"] == 0.5
    assert summary["bronze_rate"] == 0.5
    assert summary["top30_rate"] == 0.5
    assert "b" in summary["failed_tasks"]
    assert summary["medal_counts"]["bronze"] == 1


def test_save_result_dedupes_by_task_id(tmp_path: Path):
    target = tmp_path / "results.json"
    save_result(target, _result("titanic", medal="none"))
    save_result(target, _result("titanic", medal="bronze"))  # overwrite same task
    save_result(target, _result("house_prices", medal="none"))

    data = json.loads(target.read_text(encoding="utf-8"))
    by_id = {row["task_id"]: row for row in data}
    assert len(data) == 2
    assert by_id["titanic"]["medal"] == "bronze"


def test_load_tasks_accepts_real_registry_object():
    registry = ROOT / "benchmark" / "mle_bench_75" / "tasks_template.json"

    tasks = load_tasks(registry)

    assert len(tasks) == 12
    assert len({task.task_id for task in tasks}) == 12
    assert {"spaceship_titanic", "house_prices", "tps_may2022"} <= {
        task.task_id for task in tasks
    }


def test_load_tasks_keeps_legacy_list_compatibility(tmp_path: Path):
    registry = json.loads(
        (ROOT / "benchmark" / "mle_bench_75" / "tasks_template.json").read_text(encoding="utf-8")
    )
    legacy = tmp_path / "legacy-tasks.json"
    legacy.write_text(json.dumps([registry["tasks"][0]]), encoding="utf-8")

    tasks = load_tasks(legacy)

    assert [task.task_id for task in tasks] == ["spaceship_titanic"]


@pytest.mark.parametrize("failure", ["total", "planned", "duplicate", "note", "reference"])
def test_load_tasks_rejects_inconsistent_registry(tmp_path: Path, failure: str):
    registry = json.loads(
        (ROOT / "benchmark" / "mle_bench_75" / "tasks_template.json").read_text(encoding="utf-8")
    )
    if failure == "total":
        registry["total_tasks"] = 13
    elif failure == "planned":
        registry["total_tasks"] = 13
        registry["remaining_tasks_planned"] = 1
        registry["planned_tasks_summary"] = [
            {"modality": "unallocated", "count": 2, "status": "planned"}
        ]
    elif failure == "duplicate":
        registry["tasks"].append(dict(registry["tasks"][0]))
        registry["total_tasks"] += 1
    elif failure == "note":
        registry["total_tasks"] = 13
        registry["remaining_tasks_planned"] = 1
        registry["planned_tasks_summary"] = [
            {"modality": "unallocated", "count": 1, "status": "planned"}
        ]
        registry["normalization_note"] = "Tasks 14-75 are planned but not yet registered."
    elif failure == "reference":
        registry["mle_bench_reference"]["commit"] = "0" * 40
    path = _write_registry(tmp_path, f"invalid-{failure}.json", registry)

    with pytest.raises(BenchmarkRegistryError):
        load_tasks(path)


def test_result_task_ids_must_exist_and_be_unique():
    tasks = load_tasks(ROOT / "benchmark" / "mle_bench_75" / "tasks_template.json")

    validate_result_task_ids(tasks, [_result("titanic"), _result("house_prices")])
    with pytest.raises(BenchmarkRegistryError, match="unknown task_id"):
        validate_result_task_ids(tasks, [_result("not_in_registry")])
    with pytest.raises(BenchmarkRegistryError, match="duplicate benchmark result"):
        validate_result_task_ids(tasks, [_result("titanic"), _result("titanic")])


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("modality", "spreadsheet"),
        ("evaluation_mode", "official_verified"),
        ("status", "done"),
    ],
)
def test_load_tasks_rejects_values_outside_schema_enums(
    tmp_path: Path,
    field_name: str,
    invalid_value: str,
):
    registry = json.loads(
        (ROOT / "benchmark" / "mle_bench_75" / "tasks_template.json").read_text(
            encoding="utf-8"
        )
    )
    registry["tasks"][0][field_name] = invalid_value
    path = _write_registry(tmp_path, f"invalid-{field_name}.json", registry)

    with pytest.raises(BenchmarkRegistryError, match=field_name):
        load_tasks(path)


@pytest.mark.parametrize(
    ("field_path", "invalid_value"),
    [
        (("time_budget_hours",), float("inf")),
        (("time_budget_hours",), 10**400),
        (("time_budget_hours",), 10**308),
        (("baseline_score",), float("nan")),
        (("target_score",), float("-inf")),
        (("medal_thresholds", "bronze"), float("inf")),
    ],
)
def test_load_tasks_rejects_non_finite_or_unbounded_numbers(
    tmp_path: Path,
    field_path: tuple[str, ...],
    invalid_value: float | int,
):
    registry = json.loads(
        (ROOT / "benchmark" / "mle_bench_75" / "tasks_template.json").read_text(
            encoding="utf-8"
        )
    )
    target = registry["tasks"][0]
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = invalid_value
    path = _write_registry(tmp_path, "invalid-number.json", registry)

    with pytest.raises(BenchmarkRegistryError):
        load_tasks(path)


def test_load_tasks_rejects_unbounded_submission_limit(tmp_path: Path):
    registry = json.loads(
        (ROOT / "benchmark" / "mle_bench_75" / "tasks_template.json").read_text(
            encoding="utf-8"
        )
    )
    registry["tasks"][0]["submission_limit"] = 10**400
    path = _write_registry(tmp_path, "invalid-submission-limit.json", registry)

    with pytest.raises(BenchmarkRegistryError, match="submission_limit"):
        load_tasks(path)


def test_load_tasks_rejects_unbounded_registry_counts(tmp_path: Path):
    registry = json.loads(
        (ROOT / "benchmark" / "mle_bench_75" / "tasks_template.json").read_text(
            encoding="utf-8"
        )
    )
    huge = 10**400
    registry["total_tasks"] = len(registry["tasks"]) + huge
    registry["remaining_tasks_planned"] = huge
    registry["planned_tasks_summary"] = [
        {"modality": "unallocated", "count": huge, "status": "planned"}
    ]
    registry["normalization_note"] = f"Tasks 13-{registry['total_tasks']} are planned."
    path = _write_registry(tmp_path, "invalid-registry-counts.json", registry)

    with pytest.raises(BenchmarkRegistryError, match="total_tasks"):
        load_tasks(path)


def test_load_tasks_requires_exact_official_overlap(tmp_path: Path):
    registry = json.loads(
        (ROOT / "benchmark" / "mle_bench_75" / "tasks_template.json").read_text(
            encoding="utf-8"
        )
    )
    registry["mle_bench_reference"]["locally_registered_official_competitions"] = []
    path = _write_registry(tmp_path, "invalid-overlap.json", registry)

    with pytest.raises(BenchmarkRegistryError, match="overlap does not match"):
        load_tasks(path)


def test_result_validation_rejects_duplicate_programmatic_tasks():
    tasks = load_tasks(ROOT / "benchmark" / "mle_bench_75" / "tasks_template.json")

    with pytest.raises(BenchmarkRegistryError, match="duplicate benchmark task"):
        validate_result_task_ids([tasks[0], tasks[0]], [_result(tasks[0].task_id)])


@pytest.mark.parametrize("failure", ["unknown_result", "duplicate_result", "duplicate_task"])
def test_export_rejects_invalid_ids_without_creating_report(tmp_path: Path, failure: str):
    tasks = load_tasks(ROOT / "benchmark" / "mle_bench_75" / "tasks_template.json")
    results = [_result(tasks[0].task_id)]
    if failure == "unknown_result":
        results = [_result("not_in_registry")]
    elif failure == "duplicate_result":
        results = [*results, _result(tasks[0].task_id)]
    elif failure == "duplicate_task":
        tasks = [tasks[0], tasks[0]]
    report = tmp_path / "benchmark-report.md"

    with pytest.raises(BenchmarkRegistryError):
        export_benchmark_report(report, tasks, results)

    assert not report.exists()
