"""Evidence-first quality evaluation for the EvoMind novice web assistant.

The suite scores production assistant responses against versioned, deterministic
contracts.  It deliberately keeps the full response in the runtime ledger and
stores only hashes, dimensions, and failed check identifiers in the report.
This makes prompt/agent regressions auditable without training on private grader
feedback or treating an LLM's self-rating as proof of quality.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.parse
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "evomind.assistant_quality_evaluation.v1"
SUITE_SCHEMA = "evomind.assistant_quality_suite.v1"
DEFAULT_SUITE = Path(__file__).resolve().parents[2] / "configs" / "evaluation" / "assistant_novice_v1.json"
QUALITY_IMPLEMENTATION_PATHS = (
    "src/xsci/assistant_context.py",
    "src/xsci/assistant_stream.py",
    "src/xsci/assistant_behavior_distillation.py",
    "src/xsci/kaggle_conversation.py",
    "src/xsci/user_request.py",
    "src/xsci/assistant_quality_evaluation.py",
    "src/research_os/agent/messaging.py",
    "configs/evaluation/assistant_behavior_board_v1.json",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_hashes(root: str | Path) -> dict[str, str | None]:
    workspace = Path(root).resolve()
    return {
        relative: _sha256_file(workspace / relative) if (workspace / relative).is_file() else None
        for relative in QUALITY_IMPLEMENTATION_PATHS
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def load_suite(path: str | Path = DEFAULT_SUITE) -> dict[str, Any]:
    target = Path(path).resolve()
    suite = _read_json(target)
    if suite.get("schema") != SUITE_SCHEMA:
        raise ValueError("assistant quality suite schema mismatch")
    if not isinstance(suite.get("cases"), list) or not suite["cases"]:
        raise ValueError("assistant quality suite has no cases")
    if not 0 < float(suite.get("pass_threshold") or 0) <= 1:
        raise ValueError("assistant quality suite pass_threshold must be in (0,1]")
    case_ids: set[str] = set()
    for case in suite["cases"]:
        if not isinstance(case, dict):
            raise ValueError("assistant quality case must be an object")
        case_id = str(case.get("case_id") or "")
        if not re.fullmatch(r"[a-z0-9_]{3,80}", case_id) or case_id in case_ids:
            raise ValueError(f"invalid or duplicate assistant quality case_id: {case_id}")
        case_ids.add(case_id)
        checks = case.get("checks")
        if not isinstance(checks, list) or not checks:
            raise ValueError(f"assistant quality case has no checks: {case_id}")
        weight = sum(float(item.get("weight") or 0) for item in checks if isinstance(item, dict))
        if not math.isclose(weight, 100.0, abs_tol=1e-9):
            raise ValueError(f"assistant quality case weights must sum to 100: {case_id}={weight}")
    suite["suite_path"] = str(target)
    suite["suite_sha256"] = _sha256_file(target)
    return suite


def _contains_all(answer: str, values: Iterable[str]) -> bool:
    folded = answer.casefold()
    return all(str(value).casefold() in folded for value in values)


def _contains_any(answer: str, values: Iterable[str]) -> bool:
    folded = answer.casefold()
    candidates = [str(value).casefold() for value in values]
    return not candidates or any(value in folded for value in candidates)


def _regex_all(answer: str, patterns: Iterable[str]) -> bool:
    return all(re.search(str(pattern), answer, flags=re.IGNORECASE | re.DOTALL) is not None for pattern in patterns)


def _regex_any(answer: str, patterns: Iterable[str]) -> bool:
    candidates = [str(pattern) for pattern in patterns]
    return not candidates or any(re.search(pattern, answer, flags=re.IGNORECASE | re.DOTALL) is not None for pattern in candidates)


def _check_passed(answer: str, spec: dict[str, Any]) -> bool:
    return (
        _contains_all(answer, spec.get("all_of") or [])
        and _contains_any(answer, spec.get("any_of") or [])
        and _regex_all(answer, spec.get("regex_all") or [])
        and _regex_any(answer, spec.get("regex_any") or [])
    )


def _matches_required_identity(actual: str, required: str) -> bool:
    return not required or actual.casefold() == required.casefold()


def score_response(
    case: dict[str, Any],
    *,
    answer: str,
    tool_names: Iterable[str],
    llm_status: str = "completed",
    provider: str = "",
    model: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
    duration_seconds: float = 0.0,
) -> dict[str, Any]:
    tools = [str(name) for name in tool_names if str(name)]
    required_tools = [str(name) for name in case.get("required_tools") or []]
    missing_tools = [name for name in required_tools if name not in tools]
    allowed_tools = {str(name) for name in case.get("allowed_tool_names") or []}
    unexpected_tools = sorted({name for name in tools if allowed_tools and name not in allowed_tools})

    check_rows: list[dict[str, Any]] = []
    dimension_totals: dict[str, float] = {}
    dimension_earned: dict[str, float] = {}
    total_weight = 0.0
    earned_weight = 0.0
    for spec in case.get("checks") or []:
        passed = _check_passed(answer, spec)
        weight = float(spec.get("weight") or 0)
        dimension = str(spec.get("dimension") or "quality")
        total_weight += weight
        dimension_totals[dimension] = dimension_totals.get(dimension, 0.0) + weight
        if passed:
            earned_weight += weight
            dimension_earned[dimension] = dimension_earned.get(dimension, 0.0) + weight
        check_rows.append({
            "check_id": str(spec.get("check_id") or "unknown"),
            "dimension": dimension,
            "weight": weight,
            "passed": passed,
        })

    forbidden_hits = [
        value for value in (str(item) for item in case.get("forbidden_any") or [])
        if value.casefold() in answer.casefold()
    ]
    forbidden_regex_hits = [
        pattern for pattern in (str(item) for item in case.get("forbidden_regex") or [])
        if re.search(pattern, answer, flags=re.IGNORECASE | re.DOTALL) is not None
    ]
    minimum_chars = int(case.get("minimum_answer_chars") or 1)
    maximum_chars = int(case.get("maximum_answer_chars") or 100_000)
    length_ok = minimum_chars <= len(answer) <= maximum_chars
    transport_ok = llm_status in {"completed", "completed_after_wrap"}
    required_provider = str(case.get("required_provider") or "")
    required_model = str(case.get("required_model") or "")
    provider_ok = _matches_required_identity(str(provider or ""), required_provider)
    model_ok = _matches_required_identity(str(model or ""), required_model)
    score = earned_weight / total_weight if total_weight else 0.0
    threshold = float(case.get("pass_threshold") or 0.85)
    fatal_gate_passed = bool(
        answer.strip()
        and transport_ok
        and provider_ok
        and model_ok
        and length_ok
        and not missing_tools
        and not unexpected_tools
        and not forbidden_hits
        and not forbidden_regex_hits
    )
    dimensions = {
        name: round(dimension_earned.get(name, 0.0) / total, 4) if total else 0.0
        for name, total in dimension_totals.items()
    }
    return {
        "case_id": str(case.get("case_id") or "unknown"),
        "category": str(case.get("category") or "unknown"),
        "passed": fatal_gate_passed and score >= threshold,
        "score": round(score, 4),
        "threshold": threshold,
        "dimensions": dimensions,
        "checks": check_rows,
        "failed_checks": [item["check_id"] for item in check_rows if not item["passed"]],
        "fatal_gate": {
            "passed": fatal_gate_passed,
            "transport_ok": transport_ok,
            "provider_ok": provider_ok,
            "model_ok": model_ok,
            "required_provider": required_provider,
            "required_model": required_model,
            "length_ok": length_ok,
            "missing_tools": missing_tools,
            "unexpected_tools": unexpected_tools,
            "forbidden_hits": forbidden_hits,
            "forbidden_regex_hits": forbidden_regex_hits,
        },
        "answer": {
            "sha256": _sha256_text(answer),
            "characters": len(answer),
        },
        "execution": {
            "llm_status": llm_status,
            "provider": provider,
            "model": model,
            "tool_names": tools,
            "native_tool_calls": len(tools),
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "duration_seconds": round(float(duration_seconds or 0), 3),
        },
    }


def _case_by_id(suite: dict[str, Any], case_id: str) -> dict[str, Any]:
    for raw in suite.get("cases") or []:
        if isinstance(raw, dict) and raw.get("case_id") == case_id:
            case = dict(raw)
            case["pass_threshold"] = float(suite.get("pass_threshold") or 0.85)
            case["allowed_tool_names"] = list(suite.get("allowed_tool_names") or [])
            case["required_provider"] = str(suite.get("required_provider") or "")
            case["required_model"] = str(suite.get("required_model") or "")
            return case
    raise KeyError(case_id)


def _parse_time(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def load_recorded_session(root: str | Path, session_id: str) -> dict[str, Any]:
    from evomind_runtime import AgentRuntime

    runtime = AgentRuntime(Path(root))
    try:
        if runtime.store.get_session(session_id) is None:
            raise KeyError(session_id)
        turns = runtime.store.list_turns(session_id)
        events = runtime.store.list_events(session_id, limit=5000)
    finally:
        runtime.close()
    user_turns = [str(item.get("content") or "") for item in turns if item.get("role") == "user"]
    assistant_turns = [str(item.get("content") or "") for item in turns if item.get("role") == "assistant"]
    completed = [item for item in events if item.get("event_type") == "web.answer_completed"]
    usage = [item for item in events if item.get("event_type") == "web.usage"]
    tool_names = [
        str((item.get("payload") or {}).get("tool") or "")
        for item in events if item.get("event_type") == "web.tool_started"
    ]
    first_time = _parse_time(str(events[0].get("created_at") or "")) if events else None
    last_time = _parse_time(str(events[-1].get("created_at") or "")) if events else None
    duration = (last_time - first_time).total_seconds() if first_time and last_time else 0.0
    final_payload = completed[-1].get("payload") if completed else {}
    usage_payload = usage[-1].get("payload") if usage else {}
    return {
        "session_id": session_id,
        "prompt": user_turns[-1] if user_turns else "",
        "answer": assistant_turns[-1] if assistant_turns else "",
        "tool_names": tool_names,
        "llm_status": str(final_payload.get("llm_status") or "unknown"),
        "provider": str(final_payload.get("provider") or usage_payload.get("provider") or ""),
        "model": str(final_payload.get("model") or usage_payload.get("model") or ""),
        "input_tokens": int(usage_payload.get("input_tokens") or 0),
        "output_tokens": int(usage_payload.get("output_tokens") or 0),
        "duration_seconds": duration,
    }


def evaluate_recorded_session(
    root: str | Path,
    suite: dict[str, Any],
    *,
    session_id: str,
    case_id: str,
) -> dict[str, Any]:
    case = _case_by_id(suite, case_id)
    record = load_recorded_session(root, session_id)
    result = score_response(
        case,
        answer=record["answer"],
        tool_names=record["tool_names"],
        llm_status=record["llm_status"],
        provider=record["provider"],
        model=record["model"],
        input_tokens=record["input_tokens"],
        output_tokens=record["output_tokens"],
        duration_seconds=record["duration_seconds"],
    )
    result["session_id"] = session_id
    result["prompt_sha256"] = _sha256_text(record["prompt"])
    result["prompt_matches_case"] = record["prompt"].strip() == str(case.get("prompt") or "").strip()
    if not result["prompt_matches_case"]:
        result["passed"] = False
        result["fatal_gate"]["passed"] = False
        result["fatal_gate"]["prompt_mismatch"] = True
    return result


def governance_snapshot(root: str | Path) -> dict[str, Any]:
    workspace = Path(root).resolve()
    run_root = workspace / "workspace" / "evomind_runs"
    target = run_root / "evomind_siim_isic_a800_job90353_20260730_095826"
    tracked = {
        "current_run": workspace / "workspace" / "current_run.json",
        "private_grader": target / "private_grader.json",
        "private_grader_ledger": target / "private_grader_ledger.json",
        "claim_audit": target / "claim_audit.json",
        "candidate_freeze": target / "candidate_freeze.json",
    }
    hashes = {name: _sha256_file(path) if path.is_file() else None for name, path in tracked.items()}
    grader = _read_json(tracked["private_grader_ledger"]) if tracked["private_grader_ledger"].is_file() else {}
    run_dirs = sorted(
        item.name for item in run_root.glob("evomind_siim_isic*")
        if item.is_dir()
    ) if run_root.is_dir() else []
    return {
        "siim_run_directories": run_dirs,
        "tracked_hashes": hashes,
        "grader_execution_count": int(grader.get("execution_count") or 0),
        "grader_outcome": grader.get("outcome"),
        "grader_score": grader.get("score"),
        "official_submission_executed": bool(grader.get("official_submission_executed", False)),
    }


def _parse_ndjson(raw: bytes) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _bind_quality_provider_env(env: dict[str, str], suite: dict[str, Any]) -> dict[str, str]:
    """Pin a live quality case to the provider/model declared by its suite.

    Production dashboard launches already pin the same values.  The evaluator
    runs the Python bridge directly, so inheriting a developer shell's unrelated
    primary provider would test a different product configuration and make the
    provider/model gate fail for every case.  Strict mode also prevents a silent
    fallback from turning a gateway outage into a misleading passing report.
    """

    provider = str(suite.get("required_provider") or "").strip().lower()
    model = str(suite.get("required_model") or "").strip()
    model_env = {
        "anthropic": "CLAUDE_CODE_MODEL",
        "deepseek": "DEEPSEEK_MODEL",
        "openai": "OPENAI_MODEL",
    }.get(provider)
    if not provider or not model or model_env is None:
        raise ValueError("assistant quality suite must declare a supported required provider/model")
    bound = dict(env)
    bound["EVOLUTION_PRIMARY_PROVIDER"] = provider
    bound["EVOLUTION_PROVIDER_STRICT"] = "1"
    bound[model_env] = model
    if provider == "openai":
        bound.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:65068/v1")
        try:
            parsed = urllib.parse.urlsplit(str(bound.get("OPENAI_BASE_URL") or ""))
            loopback_gateway = (
                parsed.scheme == "http"
                and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
                and parsed.port == 65068
                and parsed.path.rstrip("/") == "/v1"
            )
        except ValueError:
            loopback_gateway = False
        if loopback_gateway:
            configured = str(bound.get("EVOMIND_LOCAL_GATEWAY_CONFIG") or "").strip()
            gateway_config = (
                Path(configured).expanduser()
                if configured
                else Path.home() / ".antigravity_cockpit" / "codex_local_access_sidecar" / "config.json"
            )
            gateway_key = ""
            try:
                if (
                    gateway_config.is_file()
                    and not gateway_config.is_symlink()
                    and gateway_config.stat().st_size <= 1024 * 1024
                ):
                    config_payload = _read_json(gateway_config)
                    keys = config_payload.get("api-keys")
                    if isinstance(keys, list):
                        gateway_key = next(
                            (str(item).strip() for item in keys if str(item).strip()),
                            "",
                        )
            except (OSError, ValueError, json.JSONDecodeError):
                gateway_key = ""
            # Never send an unrelated ambient cloud key to the loopback gateway.
            if gateway_key:
                bound["OPENAI_API_KEY"] = gateway_key
            else:
                bound.pop("OPENAI_API_KEY", None)
    return bound


def run_live_case(
    root: str | Path,
    suite: dict[str, Any],
    *,
    case_id: str,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    workspace = Path(root).resolve()
    case = _case_by_id(suite, case_id)
    session_id = f"assistant_eval_{case_id}_{uuid.uuid4().hex[:12]}"
    payload = {
        "prompt": str(case.get("prompt") or ""),
        "session_id": session_id,
        "selected_task": str(suite.get("selected_task") or ""),
        "history": list(case.get("history") or []),
    }
    env = _bind_quality_provider_env(os.environ.copy(), suite)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(workspace / "src"), env.get("PYTHONPATH", "")) if item
    )
    started = time.monotonic()
    completed = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "xsci.assistant_stream"],
        cwd=workspace,
        input=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=max(5.0, float(timeout_seconds)),
        check=False,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    duration = time.monotonic() - started
    events = _parse_ndjson(completed.stdout)
    answers = [str(item.get("answer") or "") for item in events if item.get("type") == "answer_completed"]
    tools = [str(item.get("tool") or "") for item in events if item.get("type") == "tool_started"]
    usage_rows = [item for item in events if item.get("type") == "usage"]
    final_rows = [item for item in events if item.get("type") == "answer_completed"]
    usage = usage_rows[-1] if usage_rows else {}
    final = final_rows[-1] if final_rows else {}
    result = score_response(
        case,
        answer=answers[-1] if answers else "",
        tool_names=tools,
        llm_status=str(final.get("llm_status") or ("process_failed" if completed.returncode else "unknown")),
        provider=str(final.get("provider") or usage.get("provider") or ""),
        model=str(final.get("model") or usage.get("model") or ""),
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        duration_seconds=duration,
    )
    result.update({
        "session_id": session_id,
        "prompt_sha256": _sha256_text(payload["prompt"]),
        "prompt_matches_case": True,
        "process": {
            "returncode": completed.returncode,
            "stderr_present": bool(completed.stderr),
            "event_count": len(events),
        },
    })
    result["execution"].update({
        "native_tool_calls": int(final.get("native_tool_calls") or 0),
        "orchestrated_tool_calls": int(final.get("orchestrated_tool_calls") or 0),
        "tool_calls_total": int(
            final.get("tool_calls_total")
            or len(tools)
        ),
        "repair_rounds": int(final.get("repair_rounds") or 0),
        "response_audit_passed": bool(
            ((final.get("response_audit") or {}).get("after") or {}).get("passed")
        ),
    })
    if completed.returncode != 0 or not answers:
        result["passed"] = False
        result["fatal_gate"]["passed"] = False
        result["fatal_gate"]["assistant_process_failed"] = True
    return result


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(item.get("score") or 0) for item in results]
    durations = sorted(float((item.get("execution") or {}).get("duration_seconds") or 0) for item in results)
    p95_index = max(0, math.ceil(len(durations) * 0.95) - 1) if durations else 0
    return {
        "case_count": len(results),
        "passed_cases": sum(bool(item.get("passed")) for item in results),
        "pass_rate": round(sum(bool(item.get("passed")) for item in results) / len(results), 4) if results else 0.0,
        "mean_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "latency_p95_seconds": round(durations[p95_index], 3) if durations else 0.0,
        "input_tokens": sum(int((item.get("execution") or {}).get("input_tokens") or 0) for item in results),
        "output_tokens": sum(int((item.get("execution") or {}).get("output_tokens") or 0) for item in results),
    }


def build_recorded_pair_report(
    root: str | Path,
    suite: dict[str, Any],
    *,
    case_id: str,
    baseline_session_id: str,
    treatment_session_id: str,
) -> dict[str, Any]:
    baseline = evaluate_recorded_session(root, suite, session_id=baseline_session_id, case_id=case_id)
    treatment = evaluate_recorded_session(root, suite, session_id=treatment_session_id, case_id=case_id)
    delta = round(float(treatment["score"]) - float(baseline["score"]), 4)
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "recorded_pair",
        "suite_id": suite["suite_id"],
        "suite_version": suite["version"],
        "suite_sha256": suite["suite_sha256"],
        "case_id": case_id,
        "baseline": baseline,
        "treatment": treatment,
        "comparison": {
            "score_delta": delta,
            "treatment_passed": treatment["passed"],
            "baseline_passed": baseline["passed"],
            "regression_fixed": bool(treatment["passed"] and not baseline["passed"] and delta > 0),
            "claim_scope": "single_identical_prompt_regression_evidence_not_general_quality_proof",
        },
        "status": "passed" if treatment["passed"] and delta > 0 else "failed",
    }


def build_live_report(
    root: str | Path,
    suite: dict[str, Any],
    *,
    case_ids: Iterable[str] | None = None,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    selected = list(case_ids or [str(item["case_id"]) for item in suite["cases"]])
    before = governance_snapshot(root)
    results = [
        run_live_case(root, suite, case_id=case_id, timeout_seconds=timeout_seconds)
        for case_id in selected
    ]
    after = governance_snapshot(root)
    invariant_passed = before == after
    aggregate = _aggregate(results)
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "live_production_tool_loop",
        "suite_id": suite["suite_id"],
        "suite_version": suite["version"],
        "suite_sha256": suite["suite_sha256"],
        "implementation_hashes": implementation_hashes(root),
        "selected_cases": selected,
        "results": results,
        "aggregate": aggregate,
        "governance_invariants": {
            "passed": invariant_passed,
            "before": before,
            "after": after,
        },
        "claim_scope": "representative_local_novice_suite_not_mle_bench_or_official_kaggle_result",
        "status": "passed" if aggregate["passed_cases"] == len(results) and invariant_passed else "failed",
    }


def write_report(path: str | Path, report: dict[str, Any]) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# EvoMind 小白科研 Agent 质量评测",
        "",
        f"- 状态：`{report.get('status')}`",
        f"- 模式：`{report.get('mode')}`",
        f"- 套件：`{report.get('suite_id')} v{report.get('suite_version')}`",
        f"- 套件 SHA-256：`{report.get('suite_sha256')}`",
        f"- 生成时间：`{report.get('generated_at')}`",
        "",
    ]
    if report.get("mode") == "recorded_pair":
        baseline = report.get("baseline") or {}
        treatment = report.get("treatment") or {}
        comparison = report.get("comparison") or {}
        lines.extend([
            "## 同 Prompt 修复前后对比",
            "",
            "| 版本 | 分数 | 通过 | 失败检查 |",
            "|---|---:|---|---|",
            f"| baseline | {baseline.get('score')} | {baseline.get('passed')} | {', '.join(baseline.get('failed_checks') or []) or '-'} |",
            f"| treatment | {treatment.get('score')} | {treatment.get('passed')} | {', '.join(treatment.get('failed_checks') or []) or '-'} |",
            "",
            f"- 分数增量：`{comparison.get('score_delta')}`",
            f"- 回归已修复：`{comparison.get('regression_fixed')}`",
            f"- 声明范围：`{comparison.get('claim_scope')}`",
        ])
    else:
        aggregate = report.get("aggregate") or {}
        lines.extend([
            "## 汇总",
            "",
            f"- 用例：`{aggregate.get('passed_cases')}/{aggregate.get('case_count')}`",
            f"- 平均分：`{aggregate.get('mean_score')}`",
            f"- P95 时延：`{aggregate.get('latency_p95_seconds')}s`",
            f"- 工具/Run/grader/Kaggle 不变量：`{(report.get('governance_invariants') or {}).get('passed')}`",
            "",
            "| 用例 | 分数 | 通过 | 工具 | 失败检查 |",
            "|---|---:|---|---|---|",
        ])
        for item in report.get("results") or []:
            execution = item.get("execution") or {}
            lines.append(
                f"| {item.get('case_id')} | {item.get('score')} | {item.get('passed')} | "
                f"{', '.join(execution.get('tool_names') or []) or '-'} | "
                f"{', '.join(item.get('failed_checks') or []) or '-'} |"
            )
        lines.extend(["", f"- 声明范围：`{report.get('claim_scope')}`"])
    return "\n".join(lines) + "\n"


__all__ = [
    "DEFAULT_SUITE",
    "SCHEMA",
    "build_live_report",
    "build_recorded_pair_report",
    "evaluate_recorded_session",
    "governance_snapshot",
    "load_recorded_session",
    "load_suite",
    "render_markdown",
    "run_live_case",
    "score_response",
    "write_report",
]
