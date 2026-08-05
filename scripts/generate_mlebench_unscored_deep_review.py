#!/usr/bin/env python3
"""Run a source-grounded GPT-5.6 audit of the eight unscored Lite tasks."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import generate_mlebench_recovery_deep_review as review_common
except ModuleNotFoundError:
    from scripts import generate_mlebench_recovery_deep_review as review_common


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROGRESS = PROJECT_ROOT / "workspace" / "mlebench_progress" / "lite11_current.json"
DEFAULT_PLAN = PROJECT_ROOT / "workspace" / "mlebench_plans" / "wave2_gpt56_current.json"
DEFAULT_CAMPAIGN = (
    PROJECT_ROOT
    / "workspace"
    / "hpc"
    / "mlebench_remote_ops"
    / "job89441_full_campaign_current.json"
)
DEFAULT_BUNDLE_EVIDENCE = (
    PROJECT_ROOT
    / "workspace"
    / "hpc"
    / "mlebench_remote_ops"
    / "bundle_verification_current.json"
)
DEFAULT_WEIGHT_EVIDENCE = (
    PROJECT_ROOT
    / "workspace"
    / "hpc"
    / "mlebench_remote_ops"
    / "vision_weight_cache_current.json"
)
DEFAULT_PROMPT = (
    PROJECT_ROOT / "workspace" / "mlebench_plans" / "unscored8_deep_gpt56_prompt_current.txt"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "workspace" / "mlebench_plans" / "unscored8_deep_gpt56_review_current.json"
)
RUNNER_BY_COMPETITION = {
    "aptos2019-blindness-detection": "run_aptos",
    "dog-breed-identification": "run_dog_breed",
    "histopathologic-cancer-detection": "run_histopath",
    "jigsaw-toxic-comment-classification-challenge": "run_jigsaw",
    "mlsp-2013-birds": "run_birds",
    "plant-pathology-2020-fgvc7": "run_plant",
    "ranzcr-clip-catheter-line-classification": "run_ranzcr",
    "the-icml-2013-whale-challenge-right-whale-redux": "run_whale",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def _top_level_nodes(tree: ast.Module) -> tuple[dict[str, ast.AST], dict[str, ast.AST]]:
    functions: dict[str, ast.AST] = {}
    assignments: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = node
            continue
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                assignments[target.id] = node
    return functions, assignments


def extract_live_call_graph(path: Path, roots: Iterable[str]) -> dict[str, Any]:
    """Extract reachable module functions and referenced top-level constants."""

    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    functions, assignments = _top_level_nodes(tree)
    missing = set(roots) - set(functions)
    if missing:
        raise RuntimeError(f"Missing Wave2 runner definitions: {sorted(missing)}")

    pending = list(roots)
    included_functions: set[str] = set()
    included_assignments: set[str] = set()
    call_graph: dict[str, list[str]] = {}
    while pending:
        name = pending.pop()
        if name in included_functions:
            continue
        included_functions.add(name)
        node = functions[name]
        calls = sorted({
            item.func.id
            for item in ast.walk(node)
            if isinstance(item, ast.Call)
            and isinstance(item.func, ast.Name)
            and item.func.id in functions
        })
        call_graph[name] = calls
        pending.extend(calls)
        for item in ast.walk(node):
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load):
                if item.id in assignments:
                    included_assignments.add(item.id)

    assignment_pending = list(included_assignments)
    while assignment_pending:
        name = assignment_pending.pop()
        for item in ast.walk(assignments[name]):
            if (
                isinstance(item, ast.Name)
                and isinstance(item.ctx, ast.Load)
                and item.id in assignments
                and item.id not in included_assignments
            ):
                included_assignments.add(item.id)
                assignment_pending.append(item.id)

    selected_nodes = {
        *(functions[name] for name in included_functions),
        *(assignments[name] for name in included_assignments),
    }
    ordered = sorted(selected_nodes, key=lambda node: int(getattr(node, "lineno", 0)))
    chunks = ["\n".join(lines[node.lineno - 1 : node.end_lineno]) for node in ordered]
    extracted = "\n\n".join(chunks)
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "file_sha256": sha256_file(path),
        "extracted_sha256": review_common.sha256_text(extracted),
        "root_functions": sorted(roots),
        "reachable_function_count": len(included_functions),
        "referenced_assignment_count": len(included_assignments),
        "call_graph": call_graph,
        "source": extracted,
    }


def build_prompt(
    progress: dict[str, Any],
    plan: dict[str, Any],
    campaign: dict[str, Any],
    bundle: dict[str, Any],
    weights: dict[str, Any],
) -> tuple[str, set[str], dict[str, Any]]:
    remaining = {str(value) for value in progress["remaining_competition_ids"]}
    expected = set(RUNNER_BY_COMPETITION)
    if remaining != expected:
        raise RuntimeError(
            "Official remaining-task ledger differs from the frozen unscored-eight contract"
        )
    wave2_path = PROJECT_ROOT / "scripts" / "mlebench_wave2_adapters.py"
    full_path = PROJECT_ROOT / "scripts" / "run_mlebench_lite_full.py"
    live_source = extract_live_call_graph(wave2_path, RUNNER_BY_COMPETITION.values())
    parser_source = review_common.extract_definitions(full_path, {"parse_args"})
    registry_source = review_common.extract_assignments(wave2_path, {"RUNNERS"})
    prompt_payload = {
        "objective": (
            "Maximize official medal conversion per GPU hour for the eight currently unscored "
            "MLE-Bench Lite tasks on one NVIDIA A40 48GB, then support a frozen 18/22 campaign."
        ),
        "truth_boundary": [
            "Only official MLE-Bench private-grader output counts as a medal.",
            "OOF, CV, promotion gates, and valid submissions are internal evidence only.",
            "Private scores are not model-selection or training signals.",
            "No Kaggle auto-submission is enabled.",
        ],
        "current_official_progress": {
            "scored": progress["scored_competitions"],
            "total": progress["lite_total_competitions"],
            "official_medals": progress["any_medal_count"],
            "medals_required": progress["medals_required_to_strictly_exceed_top"],
            "remaining_competition_ids": sorted(remaining),
            "progress_sha256": review_common.sha256_text(
                json.dumps(progress, ensure_ascii=False, sort_keys=True)
            ),
        },
        "current_execution_evidence": {
            "campaign_status": campaign.get("status"),
            "release_deployed": campaign.get("release_deployed"),
            "current_seed": campaign.get("current_seed"),
            "current_task": campaign.get("current_task"),
            "latest_gpu_gate": campaign.get("latest_gpu_gate"),
            "pinned_bundle_sha256": bundle.get("sha256"),
            "pinned_bundle_passed": bundle.get("passed"),
            "vision_weight_cache": weights.get("weight_cache"),
        },
        "existing_wave2_plan": {
            "planner": plan.get("planner"),
            "priority_order": plan.get("priority_order"),
            "strategies": plan.get("strategies"),
            "stop_rules": plan.get("stop_rules"),
        },
        "authoritative_current_source": {
            "runner_by_competition": RUNNER_BY_COMPETITION,
            "wave2_call_graph": live_source,
            "production_registry": registry_source,
            "runner_parser_defaults": parser_source,
        },
        "required_output": {
            "audit_summary": {
                "current_coverage": 8,
                "required_coverage": 8,
                "confirmed_current_findings": ["function-grounded finding"],
                "stale_findings_rejected": ["old finding contradicted by current source"],
            },
            "per_competition": [{
                "competition_id": "exact allowed id",
                "current_runner": "exact function",
                "deployment_verdict": "GO or NO-GO",
                "confirmed_source_defects": ["current function-grounded defect only"],
                "already_implemented_strengths": ["current source strength"],
                "highest_value_upgrades": ["bounded implementation action"],
                "expected_incremental_medal_probability": "0 to 1",
                "estimated_gpu_hours": "positive number",
                "medal_probability_gain_per_gpu_hour": "non-negative number",
                "validation_contract": ["machine-checkable test"],
                "stop_rule": "bounded rule",
            }],
            "implementation_order": ["all eight exact ids ordered by expected gain per GPU hour"],
            "immediate_code_changes": [{
                "competition_id": "exact id",
                "file": "repo-relative path",
                "functions_to_add_or_change": ["exact current function"],
                "acceptance_tests": ["test"],
            }],
            "resource_schedule": ["A40-aware sequential stage"],
            "global_stop_rules": ["rule"],
        },
        "allowed_competition_ids": sorted(remaining),
        "review_instructions": [
            "Audit every allowed competition exactly once.",
            "Treat the supplied AST-extracted current source and registry as authoritative.",
            "Report only defects that still exist and name the exact affected function.",
            "Explicitly reject stale claims contradicted by current source, including silent random initialization, single holdout vision, non-grouped RANZCR, and non-OOF Jigsaw when those claims are false.",
            "Rank by expected incremental official-medal probability divided by A40 GPU hours, not by ease or novelty.",
            "Do not recommend a larger backbone without accounting for the A40 48GB memory limit and sequential runtime.",
            "Preserve the three-sample GPU idle gate, dedicated remote root, Human Gate, and frozen-seed contract.",
            "Return one JSON object with all required keys.",
        ],
    }
    return (
        json.dumps(prompt_payload, ensure_ascii=False, separators=(",", ":")),
        remaining,
        live_source,
    )


def validate_review(review: dict[str, Any], competition_ids: set[str]) -> None:
    rows = review.get("per_competition")
    if not isinstance(rows, list) or len(rows) != len(competition_ids):
        raise ValueError("per_competition must cover the unscored eight")
    returned = [str(row.get("competition_id")) for row in rows if isinstance(row, dict)]
    if len(returned) != len(set(returned)) or set(returned) != competition_ids:
        raise ValueError("per_competition ids must exactly match the unscored eight")
    order = review.get("implementation_order")
    if not isinstance(order, list) or len(order) != len(competition_ids):
        raise ValueError("implementation_order must contain all unscored tasks")
    if set(map(str, order)) != competition_ids:
        raise ValueError("implementation_order ids differ from the unscored eight")
    required = {
        "audit_summary",
        "per_competition",
        "implementation_order",
        "immediate_code_changes",
        "resource_schedule",
        "global_stop_rules",
    }
    if not required <= set(review):
        raise ValueError(f"Review is missing keys: {sorted(required - set(review))}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--bundle-evidence", type=Path, default=DEFAULT_BUNDLE_EVIDENCE)
    parser.add_argument("--weight-evidence", type=Path, default=DEFAULT_WEIGHT_EVIDENCE)
    parser.add_argument("--prompt-output", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:65068/v1"),
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.6-sol"))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-tokens", type=int, default=30_000)
    args = parser.parse_args(argv)
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY must be supplied from the secure runtime environment")
    prompt, competition_ids, live_source = build_prompt(
        read_json(args.progress),
        read_json(args.plan),
        read_json(args.campaign),
        read_json(args.bundle_evidence),
        read_json(args.weight_evidence),
    )
    review_common.atomic_write(args.prompt_output, prompt + "\n")
    request_payload = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are EvoMind's senior MLE-Bench performance and medal engineer. "
                    "Audit only the supplied current source. Optimize expected official medal "
                    "gain per A40 GPU hour and reject stale source claims."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": args.max_tokens,
        "reasoning_effort": "high",
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    request = urllib.request.Request(
        review_common.endpoint(args.base_url),
        data=json.dumps(request_payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    body: dict[str, Any] | None = None
    last_error = ""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
        except (OSError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = type(exc).__name__
        if attempt < 2:
            time.sleep(2.0 * (attempt + 1))
    if body is None:
        raise RuntimeError(f"GPT-5.6 unscored audit failed after three attempts: {last_error}")
    choice = (body.get("choices") or [{}])[0]
    content = str((choice.get("message") or {}).get("content") or "")
    review = review_common.extract_json(content)
    validate_review(review, competition_ids)
    served_model = str(body.get("model") or "")
    if served_model != args.model or served_model != "gpt-5.6-sol":
        raise RuntimeError(f"Gateway served unexpected model: {served_model}")
    report = {
        "schema": "evomind.mlebench_lite.unscored8_deep_gpt56_review.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "planner": {
            "provider": "openai",
            "model": args.model,
            "served_model": served_model,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "usage": body.get("usage") or {},
            "reasoning_effort_requested": "high",
            "json_mode_requested": True,
        },
        "prompt_path": str(args.prompt_output.resolve()),
        "prompt_sha256": review_common.sha256_text(prompt),
        "source_path": live_source["path"],
        "source_sha256": live_source["file_sha256"],
        "extracted_source_sha256": live_source["extracted_sha256"],
        "response_sha256": review_common.sha256_text(content),
        "audited_competition_ids": sorted(competition_ids),
        "parse_ok": True,
        "review": review,
        "secret_policy": "API key was loaded from the secure runtime environment and is absent from artifacts.",
        "ok": True,
    }
    review_common.atomic_write(
        args.output,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "served_model": served_model,
        "usage": report["planner"]["usage"],
        "latency_ms": report["planner"]["latency_ms"],
        "parse_ok": True,
        "source_sha256": report["source_sha256"],
        "implementation_order": review["implementation_order"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
