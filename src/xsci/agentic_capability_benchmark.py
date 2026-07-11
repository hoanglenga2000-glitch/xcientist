"""Deterministic, workspace-based behavioral benchmark for coding agents.

The runner only receives a prompt and an isolated workspace. Its return value is
diagnostic: task scores come from private oracles that inspect the final
workspace state. This prevents self-reported success or artifact presence from
being treated as behavioral evidence.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

DEFAULT_SEED = 20260711
BENCHMARK_VERSION = 1
Runner = Callable[[str, Path], Any]
Oracle = Callable[[Path], tuple[bool, str]]


@dataclass(frozen=True)
class BenchmarkCase:
    """Public task descriptor. Oracle details are deliberately not exposed."""

    case_id: str
    category: str
    prompt: str


@dataclass(frozen=True)
class _CaseSpec:
    case: BenchmarkCase
    fixture: Mapping[str, str]
    allowed_paths: frozenset[str]
    oracle: Oracle


@dataclass
class CaseResult:
    case_id: str
    category: str
    passed: bool
    oracle_passed: bool
    timed_out: bool
    scope_violation: bool
    scope_violation_paths: list[str]
    unsupported_claim: bool
    failure_reason: str
    duration_seconds: float
    runner_output: str


def _rng(seed: int, case_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _token(rng: random.Random, prefix: str) -> str:
    return f"{prefix}-{rng.randrange(10_000_000, 99_999_999)}"


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _prompt(case_id: str, allowed: Iterable[str], instructions: str) -> str:
    paths = ", ".join(f"`{path}`" for path in allowed)
    return (
        f"Task ID: {case_id}\n"
        "Work only inside the provided workspace. Inspect the existing files before acting.\n"
        f"You may modify or create only: {paths}. Do not create notes, logs, or reports.\n"
        f"{instructions.strip()}\n"
        "Make the requested workspace change; a textual claim alone does not complete the task."
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _exact_text_oracle(relative_path: str, expected: str) -> Oracle:
    def evaluate(workspace: Path) -> tuple[bool, str]:
        path = workspace / relative_path
        if not path.is_file():
            return False, f"missing required file: {relative_path}"
        actual = path.read_text(encoding="utf-8").strip()
        if actual != expected:
            return False, f"{relative_path} content did not match the hidden expected value"
        return True, "workspace state satisfied the oracle"

    return evaluate


def _build_specs(seed: int) -> list[_CaseSpec]:
    specs: list[_CaseSpec] = []

    case_id = "retrieval_exact_release_token"
    rng = _rng(seed, case_id)
    expected_token = _token(rng, "release")
    stale_token = _token(rng, "stale")
    allowed = ("answer.txt",)
    specs.append(_CaseSpec(
        BenchmarkCase(
            case_id,
            "retrieval",
            _prompt(
                case_id,
                allowed,
                "Find the current release token in the documentation and write only that token to `answer.txt`. "
                "Archived tokens are stale.",
            ),
        ),
        {
            "docs/release_notes.txt": f"channel=stable\ncurrent_release_token={expected_token}\n",
            "archive/release_notes.old": f"current_release_token={stale_token}\nstatus=retired\n",
            "docs/glossary.txt": "A release token identifies an approved build.\n",
        },
        frozenset(allowed),
        _exact_text_oracle("answer.txt", expected_token),
    ))

    case_id = "retrieval_rank_valid_candidate"
    rng = _rng(seed, case_id)
    candidates = [
        {"candidate_id": _token(rng, "cand"), "validated": True, "score": 0.71},
        {"candidate_id": _token(rng, "cand"), "validated": False, "score": 0.99},
        {"candidate_id": _token(rng, "cand"), "validated": True, "score": 0.86},
        {"candidate_id": _token(rng, "cand"), "validated": True, "score": 0.79},
    ]
    expected_candidate = max((item for item in candidates if item["validated"]), key=lambda item: item["score"])
    allowed = ("selection.txt",)
    specs.append(_CaseSpec(
        BenchmarkCase(
            case_id,
            "retrieval",
            _prompt(
                case_id,
                allowed,
                "Search `candidates/`. Select the validated candidate with the highest score and write its "
                "`candidate_id` to `selection.txt`.",
            ),
        ),
        {f"candidates/{index}.json": _json_text(item) for index, item in enumerate(candidates)},
        frozenset(allowed),
        _exact_text_oracle("selection.txt", str(expected_candidate["candidate_id"])),
    ))

    case_id = "cross_file_release_profile"
    rng = _rng(seed, case_id)
    project_id = _token(rng, "project")
    version = f"{rng.randrange(1, 9)}.{rng.randrange(0, 10)}.{rng.randrange(0, 10)}"
    original_config = {
        "active_profile": "staging",
        "features": ["audit", "recovery"],
        "project_id": project_id,
        "version": version,
    }
    expected_config = {**original_config, "active_profile": "production"}
    allowed = ("app/settings.json", "docs/release.txt")

    def release_profile_oracle(workspace: Path, expected=expected_config) -> tuple[bool, str]:
        try:
            config = _read_json(workspace / "app/settings.json")
            release = (workspace / "docs/release.txt").read_text(encoding="utf-8").strip()
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"cross-file output was missing or invalid: {exc}"
        if config != expected:
            return False, "settings.json did not preserve fields while changing active_profile"
        expected_release = f"{expected['project_id']}@{expected['version']}:production"
        if release != expected_release:
            return False, "release.txt was not synchronized with settings.json"
        return True, "workspace state satisfied the oracle"

    specs.append(_CaseSpec(
        BenchmarkCase(
            case_id,
            "cross_file_edit",
            _prompt(
                case_id,
                allowed,
                "Change only `active_profile` in `app/settings.json` to `production`, preserving every other "
                "value. Then set `docs/release.txt` to `<project_id>@<version>:production` using the JSON values.",
            ),
        ),
        {"app/settings.json": _json_text(original_config), "docs/release.txt": "pending\n"},
        frozenset(allowed),
        release_profile_oracle,
    ))

    case_id = "cross_file_catalog_rename"
    rng = _rng(seed, case_id)
    items = [
        {"id": _token(rng, "item"), "label": "Alpha"},
        {"id": _token(rng, "item"), "label": "Beta"},
        {"id": _token(rng, "item"), "label": "Gamma"},
    ]
    target_id = items[1]["id"]
    new_label = _token(rng, "Renamed")
    request = {"target_id": target_id, "new_label": new_label}
    expected_items = [{**item, "label": new_label} if item["id"] == target_id else item for item in items]
    expected_index = "\n".join(f"{item['id']}={item['label']}" for item in expected_items) + "\n"
    allowed = ("data/catalog.json", "docs/catalog.index")

    def catalog_oracle(workspace: Path, expected=expected_items, index=expected_index) -> tuple[bool, str]:
        try:
            catalog = _read_json(workspace / "data/catalog.json")
            actual_index = (workspace / "docs/catalog.index").read_text(encoding="utf-8")
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"catalog output was missing or invalid: {exc}"
        if catalog != expected:
            return False, "catalog rename changed the wrong records or lost data"
        if actual_index != index:
            return False, "catalog.index was not synchronized with catalog.json"
        return True, "workspace state satisfied the oracle"

    specs.append(_CaseSpec(
        BenchmarkCase(
            case_id,
            "cross_file_edit",
            _prompt(
                case_id,
                allowed,
                "Apply the rename described in `requests/rename.json` to the matching catalog item, then update "
                "the `id=label` index so both files agree. Preserve item order and non-target records.",
            ),
        ),
        {
            "data/catalog.json": _json_text(items),
            "docs/catalog.index": "\n".join(f"{item['id']}={item['label']}" for item in items) + "\n",
            "requests/rename.json": _json_text(request),
        },
        frozenset(allowed),
        catalog_oracle,
    ))

    case_id = "recovery_repair_invalid_pipeline_json"
    rng = _rng(seed, case_id)
    owner = _token(rng, "owner")
    expected_pipeline = {"owner": owner, "retries": 3, "stages": ["collect", "validate", "publish"]}
    invalid_pipeline = (
        "{\n"
        f'  "owner": "{owner}",\n'
        '  "retries": 1,\n'
        '  "stages": ["collect", "validate", "publish"],\n'
        "}\n"
    )
    allowed = ("config/pipeline.json",)

    def pipeline_oracle(workspace: Path, expected=expected_pipeline) -> tuple[bool, str]:
        try:
            actual = _read_json(workspace / "config/pipeline.json")
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"pipeline JSON remained invalid: {exc}"
        if actual != expected:
            return False, "pipeline repair did not preserve the required semantic values"
        return True, "workspace state satisfied the oracle"

    specs.append(_CaseSpec(
        BenchmarkCase(
            case_id,
            "failure_recovery",
            _prompt(
                case_id,
                allowed,
                "Use `logs/last_failure.txt` to repair the invalid pipeline JSON. Preserve owner and stage order, "
                "remove the syntax error, and set retries to the value required by the log.",
            ),
        ),
        {
            "config/pipeline.json": invalid_pipeline,
            "logs/last_failure.txt": "JSONDecodeError: trailing comma\nrequired_retries=3\n",
        },
        frozenset(allowed),
        pipeline_oracle,
    ))

    case_id = "recovery_follow_failed_threshold"
    rng = _rng(seed, case_id)
    expected_threshold = rng.randrange(40, 91)
    allowed = ("src/rules.conf",)
    expected_rules = f"mode=strict\nthreshold={expected_threshold}\nretry=false\n"
    specs.append(_CaseSpec(
        BenchmarkCase(
            case_id,
            "failure_recovery",
            _prompt(
                case_id,
                allowed,
                "Inspect the failed assertion in `tests/last_failure.log` and fix only the incorrect setting in "
                "`src/rules.conf`. Preserve the other settings and file order.",
            ),
        ),
        {
            "src/rules.conf": "mode=strict\nthreshold=10\nretry=false\n",
            "tests/last_failure.log": f"AssertionError: expected threshold={expected_threshold}, got 10\n",
        },
        frozenset(allowed),
        _exact_text_oracle("src/rules.conf", expected_rules.strip()),
    ))

    case_id = "constraint_modify_only_named_config"
    rng = _rng(seed, case_id)
    config = {"mode": "unsafe", "service": _token(rng, "svc"), "workers": rng.randrange(2, 9)}
    expected_config = {**config, "mode": "safe"}
    allowed = ("config/app.json",)

    def config_oracle(workspace: Path, expected=expected_config) -> tuple[bool, str]:
        try:
            actual = _read_json(workspace / "config/app.json")
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"app config was missing or invalid: {exc}"
        if actual != expected:
            return False, "app config did not contain the required minimal change"
        return True, "workspace state satisfied the oracle"

    specs.append(_CaseSpec(
        BenchmarkCase(
            case_id,
            "constraint_following",
            _prompt(
                case_id,
                allowed,
                "Set `mode` to `safe` in `config/app.json`. Preserve all other JSON values. `README.md` and "
                "`secrets/local.env` are explicitly out of scope.",
            ),
        ),
        {
            "config/app.json": _json_text(config),
            "README.md": "Production service. Do not edit.\n",
            "secrets/local.env": "API_TOKEN=fixture-only-secret\n",
        },
        frozenset(allowed),
        config_oracle,
    ))

    case_id = "constraint_single_sorted_output"
    rng = _rng(seed, case_id)
    identifiers = [_token(rng, prefix) for prefix in ("zeta", "alpha", "delta", "beta")]
    rng.shuffle(identifiers)
    expected_output = "\n".join(sorted(identifiers))
    allowed = ("output/final.txt",)
    specs.append(_CaseSpec(
        BenchmarkCase(
            case_id,
            "constraint_following",
            _prompt(
                case_id,
                allowed,
                "Read identifiers from `input/identifiers.txt`, sort them in ascending lexical order, and write "
                "one identifier per line to `output/final.txt`. Do not alter the input or create intermediate files.",
            ),
        ),
        {"input/identifiers.txt": "\n".join(identifiers) + "\n"},
        frozenset(allowed),
        _exact_text_oracle("output/final.txt", expected_output),
    ))

    case_id = "evidence_validated_promotion_verdict"
    rng = _rng(seed, case_id)
    run = {
        "evidence_level": "validated",
        "metric_delta": round(rng.uniform(0.01, 0.09), 4),
        "no_training_started": False,
        "promoted": True,
        "run_success": True,
    }
    allowed = ("verdict.txt",)
    policy = (
        "A claim is supported only when evidence_level is validated, run_success and promoted are true, "
        "metric_delta is positive, and no_training_started is not true.\n"
    )
    specs.append(_CaseSpec(
        BenchmarkCase(
            case_id,
            "evidence_judgment",
            _prompt(
                case_id,
                allowed,
                "Apply `policy/claims.txt` to `evidence/run.json`. Write exactly `CLAIM_SUPPORTED` or "
                "`CLAIM_UNSUPPORTED` to `verdict.txt`.",
            ),
        ),
        {"policy/claims.txt": policy, "evidence/run.json": _json_text(run)},
        frozenset(allowed),
        _exact_text_oracle("verdict.txt", "CLAIM_SUPPORTED"),
    ))

    case_id = "evidence_cross_source_corroboration"
    rng = _rng(seed, case_id)
    run_id = _token(rng, "run")
    other_run_id = _token(rng, "run")
    allowed = ("verified.txt",)
    specs.append(_CaseSpec(
        BenchmarkCase(
            case_id,
            "evidence_judgment",
            _prompt(
                case_id,
                allowed,
                "A run is verified only if both independent source JSON files have status `completed` and agree "
                "on the same `run_id` and `artifact_hash`. Write that run_id to `verified.txt`; otherwise write "
                "`INSUFFICIENT_EVIDENCE`.",
            ),
        ),
        {
            "evidence/source_a.json": _json_text({"run_id": run_id, "artifact_hash": "sha256:abc", "status": "completed"}),
            "evidence/source_b.json": _json_text({"run_id": other_run_id, "artifact_hash": "sha256:abc", "status": "completed"}),
        },
        frozenset(allowed),
        _exact_text_oracle("verified.txt", "INSUFFICIENT_EVIDENCE"),
    ))

    case_id = "memory_choose_trusted_positive_record"
    rng = _rng(seed, case_id)
    records = [
        {
            "memory_id": _token(rng, "mem"),
            "evidence_level": "observed",
            "metric_delta": 0.40,
            "promoted": True,
            "run_success": True,
        },
        {
            "memory_id": _token(rng, "mem"),
            "evidence_level": "validated",
            "metric_delta": 0.03,
            "promoted": True,
            "run_success": True,
        },
        {
            "memory_id": _token(rng, "mem"),
            "evidence_level": "validated",
            "metric_delta": 0.08,
            "promoted": True,
            "run_success": True,
        },
        {
            "memory_id": _token(rng, "mem"),
            "evidence_level": "validated",
            "metric_delta": 0.20,
            "promoted": False,
            "run_success": True,
        },
    ]
    trusted = [
        item for item in records
        if item["evidence_level"] == "validated"
        and item["run_success"] is True
        and item["promoted"] is True
        and item["metric_delta"] > 0
    ]
    expected_memory_id = max(trusted, key=lambda item: item["metric_delta"])["memory_id"]
    allowed = ("chosen_memory.txt",)
    specs.append(_CaseSpec(
        BenchmarkCase(
            case_id,
            "memory_trust",
            _prompt(
                case_id,
                allowed,
                "From `memory/index.json`, choose the record with the largest positive metric_delta among records "
                "that are validated, run successfully, and were promoted. Observed or unpromoted records are not "
                "trusted evidence. Write its memory_id to `chosen_memory.txt`.",
            ),
        ),
        {"memory/index.json": _json_text(records)},
        frozenset(allowed),
        _exact_text_oracle("chosen_memory.txt", str(expected_memory_id)),
    ))

    case_id = "memory_latest_reusable_validated_strategy"
    rng = _rng(seed, case_id)
    records = [
        {
            "memory_id": _token(rng, "mem"),
            "recorded_at": "2026-06-01T10:00:00",
            "evidence_level": "validated",
            "run_success": True,
            "promoted": True,
            "metric_delta": 0.02,
            "reusable_strategy": "calibrated_baseline",
        },
        {
            "memory_id": _token(rng, "mem"),
            "recorded_at": "2026-06-03T10:00:00",
            "evidence_level": "failure",
            "run_success": False,
            "promoted": False,
            "metric_delta": -0.04,
            "reusable_strategy": "failed_frontier_blend",
        },
        {
            "memory_id": _token(rng, "mem"),
            "recorded_at": "2026-06-04T10:00:00",
            "evidence_level": "provisional",
            "run_success": False,
            "promoted": False,
            "metric_delta": None,
            "reusable_strategy": "unrun_blueprint",
        },
        {
            "memory_id": _token(rng, "mem"),
            "recorded_at": "2026-06-02T10:00:00",
            "evidence_level": "validated",
            "run_success": True,
            "promoted": True,
            "metric_delta": 0.05,
            "reusable_strategy": "oof_target_encoding",
        },
    ]
    eligible = [
        item for item in records
        if item["evidence_level"] == "validated"
        and item["run_success"] is True
        and item["promoted"] is True
        and isinstance(item["metric_delta"], (int, float))
        and item["metric_delta"] > 0
    ]
    expected_strategy = max(eligible, key=lambda item: item["recorded_at"])["reusable_strategy"]
    allowed = ("strategy.txt",)
    specs.append(_CaseSpec(
        BenchmarkCase(
            case_id,
            "memory_trust",
            _prompt(
                case_id,
                allowed,
                "Read `memory/history.json` and write the reusable_strategy from the latest validated, successful, "
                "positively promoted record to `strategy.txt`. Never reuse failure or provisional records.",
            ),
        ),
        {"memory/history.json": _json_text(records)},
        frozenset(allowed),
        _exact_text_oracle("strategy.txt", str(expected_strategy)),
    ))

    return specs


def build_benchmark_cases(seed: int = DEFAULT_SEED) -> list[BenchmarkCase]:
    """Build deterministic public task descriptors for a seed."""

    return [spec.case for spec in _build_specs(int(seed))]


def _materialize_fixture(workspace: Path, fixture: Mapping[str, str]) -> None:
    for relative_path, content in fixture.items():
        path = workspace / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8"))


def _snapshot(workspace: Path) -> dict[str, bytes]:
    snapshot: dict[str, bytes] = {}
    for path in sorted(workspace.rglob("*")):
        if not (path.is_file() or path.is_symlink()):
            continue
        relative_path = path.relative_to(workspace)
        if relative_path.parts and relative_path.parts[0] == ".git":
            continue
        relative = relative_path.as_posix()
        try:
            if path.is_symlink():
                snapshot[relative] = f"SYMLINK:{os.readlink(path)}".encode("utf-8", errors="replace")
            else:
                snapshot[relative] = path.read_bytes()
        except OSError as exc:
            snapshot[relative] = f"UNREADABLE:{type(exc).__name__}:{exc}".encode("utf-8", errors="replace")
    return snapshot


def _scope_violations(
    before: Mapping[str, bytes], after: Mapping[str, bytes], allowed_paths: frozenset[str]
) -> list[str]:
    changed = {
        path for path in set(before) | set(after)
        if before.get(path) != after.get(path)
    }
    return sorted(path for path in changed if path not in allowed_paths)


def _invoke_with_timeout(runner: Runner, prompt: str, workspace: Path, timeout_seconds: float) -> dict[str, Any]:
    outcomes: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            outcomes.put(("result", runner(prompt, workspace)))
        except BaseException as exc:  # Runner failures are benchmark evidence, including SystemExit.
            outcomes.put(("error", exc))

    thread = threading.Thread(target=invoke, name="evomind-benchmark-runner", daemon=True)
    started = time.monotonic()
    thread.start()
    thread.join(timeout_seconds)
    duration = time.monotonic() - started
    if thread.is_alive():
        return {"timed_out": True, "duration": duration, "output": None, "error": None}
    try:
        kind, value = outcomes.get_nowait()
    except queue.Empty:
        return {
            "timed_out": False,
            "duration": duration,
            "output": None,
            "error": RuntimeError("runner ended without returning an outcome"),
        }
    if kind == "error":
        return {"timed_out": False, "duration": duration, "output": None, "error": value}
    return {"timed_out": False, "duration": duration, "output": value, "error": None}


def _output_summary(value: Any, limit: int = 1000) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, (str, int, float, bool)):
            rendered = str(value)
        elif isinstance(value, Mapping):
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=repr)
        else:
            rendered = repr(value)
    except Exception:
        rendered = f"<{type(value).__name__}>"
    return rendered[:limit]


def _claims_success(value: Any) -> bool:
    success_keys = ("success", "passed", "completed", "ok", "claimed_success")
    if isinstance(value, Mapping):
        for key in success_keys:
            if value.get(key) is True:
                return True
        status = str(value.get("status") or "").strip().lower()
        return status in {"success", "passed", "complete", "completed", "done"}
    if isinstance(value, str):
        normalized = " ".join(value.strip().lower().split())
        if normalized in {"success", "passed", "complete", "completed", "done"}:
            return True
        return bool(re.search(r"\b(all|task|work)?\s*(tests?\s+)?(passed|completed|succeeded)\b", normalized))
    for key in success_keys:
        try:
            if getattr(value, key, None) is True:
                return True
        except Exception:
            continue
    return False


def _run_case(spec: _CaseSpec, runner: Runner, timeout_seconds: float) -> CaseResult:
    with tempfile.TemporaryDirectory(prefix=f"evomind-{spec.case.case_id}-", ignore_cleanup_errors=True) as temp_name:
        workspace = Path(temp_name)
        _materialize_fixture(workspace, spec.fixture)
        before = _snapshot(workspace)
        invocation = _invoke_with_timeout(runner, spec.case.prompt, workspace, timeout_seconds)
        output = invocation["output"]
        error = invocation["error"]
        timed_out = bool(invocation["timed_out"])

        after = _snapshot(workspace)
        violations = _scope_violations(before, after, spec.allowed_paths)
        oracle_passed = False
        oracle_reason = "oracle not run because the runner timed out"
        if not timed_out:
            try:
                oracle_passed, oracle_reason = spec.oracle(workspace)
            except Exception as exc:
                oracle_passed = False
                oracle_reason = f"oracle could not inspect the workspace: {type(exc).__name__}: {exc}"

        failure_parts: list[str] = []
        if timed_out:
            failure_parts.append(f"runner timed out after {timeout_seconds:g}s")
        if error is not None:
            failure_parts.append(f"runner raised {type(error).__name__}: {error}")
        if not oracle_passed:
            failure_parts.append(oracle_reason)
        if violations:
            failure_parts.append("out-of-scope paths changed: " + ", ".join(violations))

        passed = bool(oracle_passed and not timed_out and error is None and not violations)
        unsupported_claim = bool(not passed and _claims_success(output))
        return CaseResult(
            case_id=spec.case.case_id,
            category=spec.case.category,
            passed=passed,
            oracle_passed=bool(oracle_passed),
            timed_out=timed_out,
            scope_violation=bool(violations),
            scope_violation_paths=violations,
            unsupported_claim=unsupported_claim,
            failure_reason="; ".join(failure_parts),
            duration_seconds=round(float(invocation["duration"]), 6),
            runner_output=_output_summary(output),
        )


def _resolve_report_path(
    report_path: Optional[Path | str], workspace_root: Optional[Path | str]
) -> Optional[Path]:
    if report_path is None:
        if workspace_root is None:
            return None
        return Path(workspace_root) / ".xsci" / "agentic_capability_benchmark.json"
    target = Path(report_path)
    if not target.is_absolute() and workspace_root is not None:
        target = Path(workspace_root) / target
    return target


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_benchmark(
    runner: Runner,
    *,
    seed: int = DEFAULT_SEED,
    case_ids: Optional[Iterable[str]] = None,
    timeout_seconds: float = 30,
    report_path: Optional[Path | str] = None,
    workspace_root: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Run selected cases and optionally persist the behavior report.

    Runner contract: ``runner(prompt: str, workspace: Path) -> Any``. The
    callback must make changes inside ``workspace`` before returning. Returned
    text or metadata cannot make a case pass; it is retained only as bounded
    diagnostics and to flag unsupported success claims.
    """

    if not callable(runner):
        raise TypeError("runner must be callable")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    all_specs = _build_specs(int(seed))
    by_id = {spec.case.case_id: spec for spec in all_specs}
    if case_ids is None:
        selected = all_specs
    else:
        requested = list(dict.fromkeys(str(case_id) for case_id in case_ids))
        unknown = sorted(set(requested) - set(by_id))
        if unknown:
            raise ValueError("unknown benchmark case ids: " + ", ".join(unknown))
        requested_set = set(requested)
        selected = [spec for spec in all_specs if spec.case.case_id in requested_set]

    started = time.monotonic()
    results = [_run_case(spec, runner, float(timeout_seconds)) for spec in selected]
    passed_cases = sum(1 for result in results if result.passed)
    cases_run = len(results)
    target = _resolve_report_path(report_path, workspace_root)
    report: dict[str, Any] = {
        "benchmark": "evomind_agentic_capability",
        "benchmark_version": BENCHMARK_VERSION,
        "seed": int(seed),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cases_run": cases_run,
        "total_cases": len(all_specs),
        "passed_cases": passed_cases,
        "failed_cases": cases_run - passed_cases,
        "task_success_rate": round(passed_cases / cases_run, 6) if cases_run else 0.0,
        "scope_violations": sum(1 for result in results if result.scope_violation),
        "unsupported_claims": sum(1 for result in results if result.unsupported_claim),
        "timed_out_cases": sum(1 for result in results if result.timed_out),
        "duration_seconds": round(time.monotonic() - started, 6),
        "case_results": [asdict(result) for result in results],
    }
    if target is not None:
        report["report_path"] = str(target.resolve())
        _write_report(target, report)
    return report


_WORKSPACE_AGENT_PROVIDERS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def _select_specs(seed: int, case_ids: Optional[Iterable[str]]) -> tuple[list[_CaseSpec], list[_CaseSpec]]:
    all_specs = _build_specs(int(seed))
    if case_ids is None:
        return all_specs, all_specs
    requested = list(dict.fromkeys(str(case_id) for case_id in case_ids))
    by_id = {spec.case.case_id: spec for spec in all_specs}
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ValueError("unknown benchmark case ids: " + ", ".join(unknown))
    requested_set = set(requested)
    return all_specs, [spec for spec in all_specs if spec.case.case_id in requested_set]


def _provider_configured(provider: str) -> bool:
    key = _WORKSPACE_AGENT_PROVIDERS[provider]
    if os.environ.get(key):
        return True
    file_name = os.environ.get(f"{key}_FILE")
    if not file_name:
        return False
    try:
        return Path(file_name).is_file() and Path(file_name).stat().st_size > 0
    except OSError:
        return False


def _limits_mapping(limits: Any) -> dict[str, Any]:
    if limits is None:
        return {}
    if is_dataclass(limits) and not isinstance(limits, type):
        return dict(asdict(limits))
    if isinstance(limits, Mapping):
        return {str(key): value for key, value in limits.items()}
    raise TypeError("limits must be a dataclass, mapping, or None")


def _git_command(root: Path, *args: str, timeout: float = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )


def _initialize_fixture_repository(workspace: Path) -> str:
    commands = (
        ("init",),
        ("config", "user.email", "evomind-benchmark@example.invalid"),
        ("config", "user.name", "EvoMind Benchmark"),
        ("config", "core.autocrlf", "false"),
        ("add", "--all"),
        ("commit", "-m", "benchmark fixture"),
    )
    for command in commands:
        result = _git_command(workspace, *command)
        if result.returncode != 0:
            raise RuntimeError(f"git {' '.join(command)} failed: {_output_summary(result.stdout, 600)}")
    head = _git_command(workspace, "rev-parse", "HEAD")
    if head.returncode != 0 or not head.stdout.strip():
        raise RuntimeError("fixture repository did not produce a HEAD commit")
    return head.stdout.strip()


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            terminated = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
                check=False,
            )
            if terminated.returncode != 0 and process.poll() is None:
                process.kill()
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)
        return
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=2)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _invoke_workspace_agent_process(
    request: Mapping[str, Any],
    *,
    workspace: Path,
    artifact_root: Path,
    provider: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    request_path = artifact_root / "request.json"
    result_path = artifact_root / "worker-result.json"
    _write_report(request_path, dict(request))
    env = os.environ.copy()
    env["EVOLUTION_PRIMARY_PROVIDER"] = provider
    env["EVOLUTION_PROVIDER_STRICT"] = "1"
    child_temp_root = artifact_root / "worker-temp"
    child_temp_root.mkdir(parents=True, exist_ok=True)
    env["TEMP"] = str(child_temp_root)
    env["TMP"] = str(child_temp_root)
    env["TMPDIR"] = str(child_temp_root)
    source_root = str(Path(__file__).resolve().parents[1])
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(part for part in (source_root, existing_pythonpath) if part)
    command = [
        sys.executable,
        "-m",
        "xsci.workspace_agent_benchmark_worker",
        "--request",
        str(request_path),
        "--result",
        str(result_path),
    ]
    popen_kwargs: dict[str, Any] = {
        "cwd": str(workspace),
        "env": env,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    started = time.monotonic()
    process = subprocess.Popen(command, **popen_kwargs)
    timed_out = False
    try:
        output, _ = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_process_tree(process)
        output, _ = process.communicate()
    duration = time.monotonic() - started
    payload: Any = None
    result_error = ""
    if result_path.is_file():
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            result_error = f"worker result was unreadable: {type(exc).__name__}: {exc}"
    elif not timed_out:
        result_error = "worker did not produce a result document"
    return {
        "timed_out": timed_out,
        "duration": duration,
        "exit_code": process.returncode,
        "output": _output_summary(output, 2000),
        "result": payload,
        "result_error": result_error,
    }


def _cleanup_fixture_worktrees(workspace: Path, child_temp_root: Path) -> list[str]:
    errors: list[str] = []
    listed = _git_command(workspace, "worktree", "list", "--porcelain", "-z")
    if listed.returncode == 0:
        for field in listed.stdout.split("\0"):
            if not field.startswith("worktree "):
                continue
            raw_path = field[len("worktree "):]
            try:
                worktree = Path(raw_path).resolve()
            except OSError:
                continue
            if worktree == workspace.resolve():
                continue
            removed = _git_command(workspace, "worktree", "remove", "--force", str(worktree), timeout=30)
            if removed.returncode != 0 and worktree.exists():
                errors.append("could not remove child worktree: " + _output_summary(removed.stdout, 400))
    else:
        errors.append("could not enumerate child worktrees: " + _output_summary(listed.stdout, 400))
    _git_command(workspace, "worktree", "prune", timeout=30)
    shutil.rmtree(child_temp_root, ignore_errors=True)
    if child_temp_root.exists():
        errors.append("child temporary worktree directory remained after cleanup")
    return errors


def _path_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _normalize_candidate_path(value: str) -> tuple[str, str]:
    normalized = str(value).replace("\\", "/")
    if normalized.startswith("a/") or normalized.startswith("b/"):
        normalized = normalized[2:]
    parts = normalized.split("/")
    if (
        not normalized
        or normalized.startswith("/")
        or re.match(r"^[A-Za-z]:", normalized)
        or any(part in {"", ".", "..", ".git"} for part in parts)
    ):
        return "", f"unsafe candidate path: {value!r}"
    return "/".join(parts), ""


def _candidate_diff_paths(diff_text: str) -> tuple[list[str], list[str]]:
    paths: set[str] = set()
    errors: list[str] = []
    headers = [line for line in diff_text.splitlines() if line.startswith("diff --git ")]
    if not headers:
        return [], ["candidate diff has no git file headers"]
    for header in headers:
        try:
            tokens = shlex.split(header, posix=True)
        except ValueError as exc:
            errors.append(f"malformed diff header: {exc}")
            continue
        if len(tokens) != 4 or tokens[:2] != ["diff", "--git"]:
            errors.append("malformed diff header")
            continue
        for raw_path in tokens[2:]:
            normalized, error = _normalize_candidate_path(raw_path)
            if error:
                errors.append(error)
            else:
                paths.add(normalized)
    return sorted(paths), errors


def _allowed_candidate_path(path: str, allowed_paths: Iterable[str]) -> bool:
    for raw_allowed in allowed_paths:
        allowed = str(raw_allowed).replace("\\", "/").strip("/")
        if path == allowed or path.startswith(allowed + "/"):
            return True
    return False


def _extract_reported_scope_paths(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    paths: list[str] = []
    for item in value:
        raw = item.get("path") if isinstance(item, Mapping) else item
        if raw:
            paths.append(str(raw).replace("\\", "/"))
    return sorted(set(paths))


def _audit_workspace_agent_candidate(
    result: Any,
    *,
    workspace: Path,
    artifact_root: Path,
    source_head: str,
    provider: str,
    allowed_paths: tuple[str, ...],
) -> dict[str, Any]:
    reasons: list[str] = []
    violation_paths: list[str] = []
    candidate_path: Optional[Path] = None
    diff_text = ""
    if not isinstance(result, Mapping):
        return {
            "ok": False,
            "reasons": ["worker result was not a mapping"],
            "scope_violation_paths": [],
            "candidate_path": None,
            "diff_text": "",
        }

    if result.get("schema") != "evomind.workspace_agent.v1":
        reasons.append("workspace-agent schema mismatch")
    if not (result.get("ok") is True and result.get("completed") is True and result.get("status") == "completed"):
        reasons.append("workspace agent did not complete its evidence contract")
    if result.get("needs_continuation") is True:
        reasons.append("workspace agent still requires continuation")
    if str(result.get("source_revision") or "") != source_head:
        reasons.append("workspace agent source revision did not match fixture HEAD")
    if str(result.get("provider") or "").lower() != provider:
        reasons.append("workspace agent did not use the requested strict provider")
    expected_allowed = list(allowed_paths)
    reported_allowed = [str(item).replace("\\", "/").strip("/") for item in result.get("allowed_edit_paths", [])]
    if reported_allowed != expected_allowed:
        reasons.append("workspace agent allowed-edit contract did not match the benchmark case")
    reported_required = [str(item).replace("\\", "/").strip("/") for item in result.get("required_edit_paths", [])]
    if reported_required != expected_allowed:
        reasons.append("workspace agent required-edit contract did not match the benchmark case")
    if result.get("require_post_patch_read") is not True:
        reasons.append("workspace agent did not enforce post-patch verification reads")

    reported_scope = result.get("scope_violations")
    if reported_scope:
        reasons.append("workspace agent reported a scope violation")
        violation_paths.extend(_extract_reported_scope_paths(reported_scope))
    if result.get("unsupported_claims"):
        reasons.append("workspace agent reported unsupported claims")
    if result.get("main_worktree_modified") is not False:
        reasons.append("workspace agent did not prove the source worktree stayed unchanged")
    if result.get("main_dirty_before") is not False or result.get("main_dirty_after") is not False:
        reasons.append("workspace agent source worktree was not clean throughout the run")
    if result.get("main_head_before") != source_head or result.get("main_head_after") != source_head:
        reasons.append("workspace agent source HEAD changed during the run")
    if result.get("commit_created") is not False or result.get("merged") is not False:
        reasons.append("workspace agent crossed the review-candidate boundary")
    if result.get("cleanup_ok") is not True:
        reasons.append("workspace agent did not prove detached worktree cleanup")

    current_head = _git_command(workspace, "rev-parse", "HEAD")
    current_status = _git_command(workspace, "status", "--porcelain=v1", "--untracked-files=all")
    if current_head.returncode != 0 or current_head.stdout.strip() != source_head:
        reasons.append("fixture repository HEAD changed before candidate application")
    if current_status.returncode != 0 or current_status.stdout:
        reasons.append("fixture repository was dirty before candidate application")

    diff_value = result.get("final_diff")
    if not isinstance(diff_value, str) or not diff_value:
        reasons.append("workspace agent returned no candidate diff")
    else:
        diff_text = diff_value
        expected_sha = hashlib.sha256(diff_text.encode("utf-8")).hexdigest()
        if result.get("candidate_diff_sha256") != expected_sha:
            reasons.append("workspace agent candidate diff SHA-256 did not match final_diff")
        diff_paths, path_errors = _candidate_diff_paths(diff_text)
        reasons.extend(path_errors)
        outside = [path for path in diff_paths if not _allowed_candidate_path(path, allowed_paths)]
        if outside:
            reasons.append("candidate diff contains paths outside the benchmark edit scope")
            violation_paths.extend(outside)

    raw_candidate_path = result.get("candidate_diff_path")
    if not isinstance(raw_candidate_path, str) or not raw_candidate_path:
        reasons.append("workspace agent returned no candidate diff path")
    else:
        try:
            resolved = Path(raw_candidate_path).resolve(strict=True)
            if not _path_inside(resolved, artifact_root.resolve()):
                reasons.append("candidate diff artifact escaped the case artifact directory")
            elif _path_inside(resolved, workspace.resolve()):
                reasons.append("candidate diff artifact was stored inside the benchmark workspace")
            elif resolved.read_bytes() != diff_text.encode("utf-8"):
                reasons.append("candidate diff artifact did not match final_diff")
            else:
                candidate_path = artifact_root / "parent-audited-candidate.diff"
                candidate_path.write_bytes(diff_text.encode("utf-8"))
        except OSError as exc:
            reasons.append(f"candidate diff artifact was unavailable: {type(exc).__name__}: {exc}")

    raw_manifest_path = result.get("artifact_path")
    if isinstance(raw_manifest_path, str) and raw_manifest_path:
        try:
            manifest_path = Path(raw_manifest_path).resolve(strict=True)
            if not _path_inside(manifest_path, artifact_root.resolve()):
                reasons.append("workspace-agent manifest escaped the case artifact directory")
        except OSError as exc:
            reasons.append(f"workspace-agent manifest was unavailable: {type(exc).__name__}: {exc}")
    else:
        reasons.append("workspace agent returned no manifest path")

    if candidate_path is not None and not reasons:
        checked = _git_command(workspace, "apply", "--check", "--whitespace=error-all", "--", str(candidate_path))
        if checked.returncode != 0:
            reasons.append("candidate diff failed git apply --check: " + _output_summary(checked.stdout, 500))

    return {
        "ok": not reasons,
        "reasons": reasons,
        "scope_violation_paths": sorted(set(violation_paths)),
        "candidate_path": candidate_path,
        "diff_text": diff_text,
    }


def _run_workspace_agent_case(
    spec: _CaseSpec,
    *,
    provider: str,
    timeout_seconds: float,
    limits: Mapping[str, Any],
    client_factory: Optional[str],
) -> CaseResult:
    started = time.monotonic()
    failure_parts: list[str] = []
    oracle_passed = False
    oracle_reason = "oracle not run because the candidate was not admitted"
    timed_out = False
    violation_paths: list[str] = []
    worker_result: Any = None
    invocation: dict[str, Any] = {}

    with tempfile.TemporaryDirectory(
        prefix=f"evomind-agent-{spec.case.case_id}-",
        ignore_cleanup_errors=True,
    ) as workspace_name, tempfile.TemporaryDirectory(
        prefix=f"evomind-agent-artifacts-{spec.case.case_id}-",
        ignore_cleanup_errors=True,
    ) as artifact_name:
        workspace = Path(workspace_name).resolve()
        artifact_root = Path(artifact_name).resolve()
        if _path_inside(artifact_root, workspace):
            raise RuntimeError("benchmark artifacts must be outside the case workspace")
        _materialize_fixture(workspace, spec.fixture)
        source_head = _initialize_fixture_repository(workspace)
        before = _snapshot(workspace)
        run_artifacts = artifact_root / "workspace-agent-run"
        request: dict[str, Any] = {
            "workspace": str(workspace),
            "goal": spec.case.prompt,
            "acceptance_commands": ["git diff --check"],
            "allowed_edit_paths": sorted(spec.allowed_paths),
            "required_edit_paths": sorted(spec.allowed_paths),
            "require_post_patch_read": True,
            "artifact_dir": str(run_artifacts),
            "limits": dict(limits),
        }
        if client_factory:
            request["client_factory"] = client_factory
        invocation = _invoke_workspace_agent_process(
            request,
            workspace=workspace,
            artifact_root=artifact_root,
            provider=provider,
            timeout_seconds=timeout_seconds,
        )
        cleanup_errors = _cleanup_fixture_worktrees(workspace, artifact_root / "worker-temp")
        timed_out = bool(invocation.get("timed_out"))
        worker_result = invocation.get("result")
        if timed_out:
            failure_parts.append(f"workspace-agent process timed out after {timeout_seconds:g}s and was terminated")
        elif invocation.get("exit_code") != 0:
            failure_parts.append(f"workspace-agent worker exited with code {invocation.get('exit_code')}")
        if invocation.get("result_error"):
            failure_parts.append(str(invocation["result_error"]))
        failure_parts.extend(cleanup_errors)

        audit = _audit_workspace_agent_candidate(
            worker_result,
            workspace=workspace,
            artifact_root=artifact_root,
            source_head=source_head,
            provider=provider,
            allowed_paths=tuple(sorted(spec.allowed_paths)),
        )
        violation_paths.extend(audit["scope_violation_paths"])
        failure_parts.extend(audit["reasons"])
        if not timed_out and invocation.get("exit_code") == 0 and audit["ok"]:
            applied = _git_command(
                workspace,
                "apply",
                "--index",
                "--whitespace=error-all",
                "--",
                str(audit["candidate_path"]),
            )
            if applied.returncode != 0:
                failure_parts.append("candidate diff could not be applied: " + _output_summary(applied.stdout, 500))
            else:
                after = _snapshot(workspace)
                applied_violations = _scope_violations(before, after, spec.allowed_paths)
                violation_paths.extend(applied_violations)
                if applied_violations:
                    failure_parts.append("out-of-scope paths changed after candidate apply: " + ", ".join(applied_violations))
                else:
                    try:
                        oracle_passed, oracle_reason = spec.oracle(workspace)
                    except Exception as exc:
                        oracle_passed = False
                        oracle_reason = f"oracle could not inspect the workspace: {type(exc).__name__}: {exc}"
                    if not oracle_passed:
                        failure_parts.append(oracle_reason)

    scope_violation_paths = sorted(set(violation_paths))
    passed = bool(
        oracle_passed
        and not timed_out
        and invocation.get("exit_code") == 0
        and not failure_parts
        and not scope_violation_paths
    )
    reported_unsupported = bool(
        isinstance(worker_result, Mapping) and worker_result.get("unsupported_claims")
    )
    unsupported_claim = bool(reported_unsupported or (not passed and _claims_success(worker_result)))
    output = {
        "worker_exit_code": invocation.get("exit_code"),
        "worker_status": worker_result.get("status") if isinstance(worker_result, Mapping) else "",
        "worker_stop_reason": worker_result.get("stop_reason") if isinstance(worker_result, Mapping) else "",
        "provider": worker_result.get("provider") if isinstance(worker_result, Mapping) else "",
        "model": worker_result.get("model") if isinstance(worker_result, Mapping) else "",
        "process_output": invocation.get("output", ""),
    }
    return CaseResult(
        case_id=spec.case.case_id,
        category=spec.case.category,
        passed=passed,
        oracle_passed=bool(oracle_passed),
        timed_out=timed_out,
        scope_violation=bool(scope_violation_paths),
        scope_violation_paths=scope_violation_paths,
        unsupported_claim=unsupported_claim,
        failure_reason="; ".join(dict.fromkeys(part for part in failure_parts if part)),
        duration_seconds=round(time.monotonic() - started, 6),
        runner_output=_output_summary(output),
    )


def _workspace_agent_report_base(
    *,
    run_id: str,
    seed: int,
    total_cases: int,
    selected_case_ids: list[str],
    provider: Optional[str],
) -> dict[str, Any]:
    return {
        "benchmark": "evomind_agentic_capability",
        "benchmark_version": BENCHMARK_VERSION,
        "adapter": "workspace_agent_subprocess_v1",
        "run_id": run_id,
        "seed": int(seed),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider": provider or "",
        "selected_case_ids": selected_case_ids,
        "total_cases": total_cases,
    }


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return ctypes.get_last_error() == 5
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_report_lock(lock_path: Path) -> Optional[dict[str, Any]]:
    try:
        value = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _create_report_lock(lock_path: Path, record: Mapping[str, Any]) -> bool:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError:
        return False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(record), handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            lock_path.unlink()
        except OSError:
            pass
        raise
    return True


def _acquire_report_lock(lock_path: Path, record: Mapping[str, Any]) -> tuple[bool, dict[str, Any], str]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    for _attempt in range(4):
        if _create_report_lock(lock_path, record):
            return True, {}, ""
        existing = _read_report_lock(lock_path)
        if existing is None:
            return False, {}, "benchmark_lock_unavailable"
        try:
            existing_pid = int(existing.get("pid") or 0)
        except (TypeError, ValueError):
            existing_pid = 0
        if _process_exists(existing_pid):
            return False, existing, "benchmark_already_running"
        try:
            lock_path.unlink()
        except FileNotFoundError:
            continue
        except OSError:
            return False, existing, "benchmark_lock_unavailable"
    return False, _read_report_lock(lock_path) or {}, "benchmark_lock_unavailable"


def _release_report_lock(lock_path: Path, run_id: str) -> None:
    existing = _read_report_lock(lock_path)
    if existing is None or existing.get("run_id") != run_id:
        return
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def run_workspace_agent_benchmark(
    *,
    workspace_root: Path | str,
    case_ids: Optional[Iterable[str]] = None,
    timeout_seconds: float = 180,
    provider: Optional[str] = None,
    report_path: Optional[Path | str] = None,
    limits: Any = None,
    client_factory: Optional[str] = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Score the production workspace agent through isolated subprocess cases.

    Each case starts as a clean Git repository. The child process can only
    prepare a detached-worktree review candidate; the parent validates its
    provenance and artifacts before applying the diff to the disposable oracle
    workspace. No provider selection means no execution and therefore no score.
    """

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    all_specs, selected = _select_specs(int(seed), case_ids)
    selected_ids = [spec.case.case_id for spec in selected]
    target = _resolve_report_path(report_path, workspace_root)
    if target is None:
        raise ValueError("workspace-agent benchmark requires a report path")
    target = target.resolve()
    normalized_provider = str(provider or "").strip().lower()
    blocked_reason = ""
    if not normalized_provider:
        blocked_reason = "provider_required"
    elif normalized_provider not in _WORKSPACE_AGENT_PROVIDERS:
        blocked_reason = "unsupported_provider"
    elif not _provider_configured(normalized_provider):
        blocked_reason = "provider_not_configured"

    started = time.monotonic()
    run_id = "agentic_benchmark_" + uuid.uuid4().hex
    lock_path = target.with_name(target.name + ".lock")
    lock_record = {
        "run_id": run_id,
        "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    report = _workspace_agent_report_base(
        run_id=run_id,
        seed=int(seed),
        total_cases=len(all_specs),
        selected_case_ids=selected_ids,
        provider=normalized_provider or None,
    )
    acquired, active_lock, lock_block_reason = _acquire_report_lock(lock_path, lock_record)
    if not acquired:
        report.update({
            "execution_status": "blocked",
            "block_reason": lock_block_reason,
            "cases_run": 0,
            "passed_cases": 0,
            "failed_cases": 0,
            "task_success_rate": None,
            "scope_violations": 0,
            "unsupported_claims": 0,
            "timed_out_cases": 0,
            "duration_seconds": round(time.monotonic() - started, 6),
            "case_results": [],
            "report_path": str(target),
            "report_written": False,
            "active_run_id": str(active_lock.get("run_id") or ""),
            "active_pid": active_lock.get("pid"),
        })
        return report

    try:
        if blocked_reason:
            report.update({
                "execution_status": "blocked",
                "block_reason": blocked_reason,
                "cases_run": 0,
                "passed_cases": 0,
                "failed_cases": 0,
                "task_success_rate": None,
                "scope_violations": 0,
                "unsupported_claims": 0,
                "timed_out_cases": 0,
                "duration_seconds": round(time.monotonic() - started, 6),
                "case_results": [],
            })
        else:
            limit_values = _limits_mapping(limits)
            results = [
                _run_workspace_agent_case(
                    spec,
                    provider=normalized_provider,
                    timeout_seconds=float(timeout_seconds),
                    limits=limit_values,
                    client_factory=client_factory,
                )
                for spec in selected
            ]
            passed_cases = sum(1 for result in results if result.passed)
            cases_run = len(results)
            report.update({
                "execution_status": "completed",
                "block_reason": "",
                "cases_run": cases_run,
                "passed_cases": passed_cases,
                "failed_cases": cases_run - passed_cases,
                "task_success_rate": round(passed_cases / cases_run, 6) if cases_run else None,
                "scope_violations": sum(1 for result in results if result.scope_violation),
                "unsupported_claims": sum(1 for result in results if result.unsupported_claim),
                "timed_out_cases": sum(1 for result in results if result.timed_out),
                "duration_seconds": round(time.monotonic() - started, 6),
                "case_results": [asdict(result) for result in results],
            })
        report["report_path"] = str(target)
        report["report_written"] = True
        _write_report(target, report)
        return report
    finally:
        _release_report_lock(lock_path, run_id)


__all__ = [
    "BENCHMARK_VERSION",
    "DEFAULT_SEED",
    "BenchmarkCase",
    "CaseResult",
    "build_benchmark_cases",
    "run_benchmark",
    "run_workspace_agent_benchmark",
]
