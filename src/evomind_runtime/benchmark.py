from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Any

PARITY_TOOLS = ("evomind", "claude-code", "codex")
REQUIRED_REPETITIONS = 3


@dataclass(frozen=True)
class BenchmarkTask:
    id: str
    category: str
    objective: str
    repetitions: int = REQUIRED_REPETITIONS
    timeout_seconds: int = 300
    max_tool_calls: int = 30


def parity_suite() -> list[dict[str, Any]]:
    groups = [("coding", 20), ("shell", 10), ("browser_desktop", 10), ("research", 10), ("recovery_multi_agent", 10)]
    tasks = []
    for category, count in groups:
        for index in range(1, count + 1):
            tasks.append(asdict(BenchmarkTask(f"{category}-{index:02d}", category, f"Hidden {category} capability task {index}")))
    return tasks


def _validated_measurement(run: dict[str, Any], task_ids: set[str]) -> tuple[tuple[str, str, int], dict[str, Any]] | None:
    """Return a comparable benchmark cell only when its evidence is complete.

    A generic ``status=completed`` row is not proof of parity.  Each row must
    identify the exact hidden task and repetition, carry an oracle score, prove
    artifact completion, and report claim/tool-call integrity metrics.
    """

    if run.get("status") != "completed" or run.get("tool_name") not in PARITY_TOOLS:
        return None
    metrics = run.get("metrics")
    if not isinstance(metrics, dict):
        return None
    task_id = metrics.get("task_id")
    repetition = metrics.get("repetition")
    score = metrics.get("score")
    success = metrics.get("success")
    artifact_complete = metrics.get("artifact_complete")
    unsupported_claims = metrics.get("unsupported_claims")
    duration_seconds = metrics.get("duration_seconds")
    tool_calls_total = metrics.get("tool_calls_total")
    tool_calls_succeeded = metrics.get("tool_calls_succeeded")
    if task_id not in task_ids or not isinstance(repetition, int) or not 1 <= repetition <= REQUIRED_REPETITIONS:
        return None
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0.0 <= float(score) <= 1.0:
        return None
    if not isinstance(success, bool) or artifact_complete is not True:
        return None
    if not isinstance(unsupported_claims, int) or isinstance(unsupported_claims, bool) or unsupported_claims != 0:
        return None
    if isinstance(duration_seconds, bool) or not isinstance(duration_seconds, (int, float)) or duration_seconds < 0:
        return None
    if not isinstance(tool_calls_total, int) or isinstance(tool_calls_total, bool) or tool_calls_total < 0:
        return None
    if not isinstance(tool_calls_succeeded, int) or isinstance(tool_calls_succeeded, bool) or not 0 <= tool_calls_succeeded <= tool_calls_total:
        return None
    key = (str(run["tool_name"]), str(task_id), repetition)
    return key, {
        "score": float(score),
        "success": success,
        "duration_seconds": float(duration_seconds),
        "tool_calls_total": tool_calls_total,
        "tool_calls_succeeded": tool_calls_succeeded,
    }


def parity_status(runs: list[dict[str, Any]]) -> dict[str, Any]:
    suite = parity_suite()
    task_ids = {item["id"] for item in suite}
    expected_cells = {
        (tool_name, task_id, repetition)
        for tool_name in PARITY_TOOLS
        for task_id in task_ids
        for repetition in range(1, REQUIRED_REPETITIONS + 1)
    }
    measurements: dict[tuple[str, str, int], dict[str, Any]] = {}
    invalid_completed_records = 0
    completed_tools: set[str] = set()
    for run in runs:
        if run.get("status") == "completed" and run.get("tool_name") in PARITY_TOOLS:
            completed_tools.add(str(run["tool_name"]))
        validated = _validated_measurement(run, task_ids)
        if validated is None:
            if run.get("status") == "completed":
                invalid_completed_records += 1
            continue
        key, measurement = validated
        measurements.setdefault(key, measurement)

    covered_cells = expected_cells & measurements.keys()
    comparison_complete = covered_cells == expected_cells
    summaries: dict[str, dict[str, Any]] = {}
    for tool_name in PARITY_TOOLS:
        items = [value for (tool, _task, _rep), value in measurements.items() if tool == tool_name]
        calls_total = sum(item["tool_calls_total"] for item in items)
        calls_succeeded = sum(item["tool_calls_succeeded"] for item in items)
        summaries[tool_name] = {
            "records": len(items),
            "mean_score": round(fmean(item["score"] for item in items), 6) if items else None,
            "success_rate": round(sum(bool(item["success"]) for item in items) / len(items), 6) if items else None,
            "mean_duration_seconds": round(fmean(item["duration_seconds"] for item in items), 6) if items else None,
            "tool_call_success_rate": round(calls_succeeded / calls_total, 6) if calls_total else None,
        }

    parity_verified = False
    advantage_verified = False
    if comparison_complete:
        evomind = summaries["evomind"]
        competitors = [summaries["codex"], summaries["claude-code"]]
        parity_verified = all(
            float(evomind["mean_score"]) >= float(item["mean_score"])
            and float(evomind["success_rate"]) >= float(item["success_rate"])
            for item in competitors
        )
        advantage_verified = all(
            float(evomind["mean_score"]) > float(item["mean_score"])
            and float(evomind["success_rate"]) >= float(item["success_rate"])
            for item in competitors
        )

    status = (
        "parity_verified" if parity_verified else
        "comparison_complete_below_parity" if comparison_complete else
        "implemented_not_parity_verified"
    )
    return {
        "status": status,
        "task_count": len(suite),
        "required_repetitions": REQUIRED_REPETITIONS,
        "required_tools": list(PARITY_TOOLS),
        "expected_records": len(expected_cells),
        "qualifying_records": len(covered_cells),
        "missing_records": len(expected_cells - covered_cells),
        "invalid_completed_records": invalid_completed_records,
        "comparison_complete": comparison_complete,
        "compared_tools": sorted(completed_tools),
        "tool_summaries": summaries,
        "advantage_status": "verified" if advantage_verified else "not_verified",
        "release_gate": "GO" if parity_verified else "NO-GO",
    }
