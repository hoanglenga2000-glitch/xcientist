#!/usr/bin/env python3
"""Run a compact, source-grounded GPT-5.6 audit of the Jigsaw runner."""
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "workspace"
    / "hpc"
    / "mlebench_remote_ops"
    / "collected"
    / "job89441_jigsaw_toxic_s42_parallel_20260726_035500"
    / "summary.json"
)
DEFAULT_PROMPT = (
    PROJECT_ROOT / "workspace" / "mlebench_plans" / "jigsaw_gpt56_audit_prompt_current.txt"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "workspace" / "mlebench_plans" / "jigsaw_gpt56_audit_current.json"
)
SOURCE_PATH = PROJECT_ROOT / "scripts" / "mlebench_wave2_adapters.py"
JIGSAW_ID = "jigsaw-toxic-comment-classification-challenge"
REQUIRED_FUNCTIONS = {
    "_fit_nbsvm",
    "fit_jigsaw_nbsvm_channels",
    "cross_fit_multilabel_rank_blend",
    "build_jigsaw_promotion_gate",
    "run_jigsaw",
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Audit response did not contain a JSON object")
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Audit response must be a JSON object")
    return value


def extract_definitions(path: Path, names: Iterable[str]) -> str:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    wanted = set(names)
    chunks: list[str] = []
    found: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
            chunks.append("\n".join(lines[node.lineno - 1 : node.end_lineno]))
            found.add(node.name)
    if found != wanted:
        raise RuntimeError(f"Missing Jigsaw definitions: {sorted(wanted - found)}")
    return "\n\n".join(chunks)


def find_jigsaw_result(summary: dict[str, Any]) -> dict[str, Any]:
    results = summary.get("results")
    if not isinstance(results, list):
        raise ValueError("Collected summary does not contain a result list")
    matches = [
        row
        for row in results
        if isinstance(row, dict) and row.get("competition_id") == JIGSAW_ID
    ]
    if len(matches) != 1:
        raise ValueError("Collected summary must contain exactly one Jigsaw result")
    result = matches[0]
    gate = result.get("promotion_gate")
    if not isinstance(gate, dict) or gate.get("schema") != "evomind.mlebench_lite.metric_promotion_gate.v1":
        raise ValueError("Jigsaw result is missing its metric promotion gate")
    if gate.get("official_grader_executed") is not False:
        raise ValueError("Expected the current Jigsaw official grader to be withheld")
    return result


def build_prompt(summary: dict[str, Any]) -> str:
    result = find_jigsaw_result(summary)
    gate = result["promotion_gate"]
    budget = result.get("budget") or {}
    observed = {
        "internal_oof_mean_auc": gate.get("internal_score"),
        "promotion_threshold": gate.get("threshold"),
        "required_absolute_gain": float(gate["threshold"]) - float(gate["internal_score"]),
        "checks": gate.get("checks"),
        "evidence": gate.get("evidence"),
        "per_label_oof": result.get("per_label_oof"),
        "training_budget": {
            "seed": budget.get("seed"),
            "train_rows": budget.get("train_rows"),
            "test_rows": budget.get("test_rows"),
            "folds": budget.get("folds"),
            "feature_counts": budget.get("feature_counts"),
            "nbsvm_c": budget.get("nbsvm_c"),
            "fit_workers": budget.get("fit_workers"),
            "split_strategy": budget.get("split_strategy"),
        },
        "official_private_grader": "withheld because the aggregate OOF gate failed",
    }
    payload = {
        "objective": (
            "Increase this exact Jigsaw pipeline from 0.9855174293 to at least 0.9870000000 "
            "on complete cross-fitted public OOF, then permit the official private grader. "
            "Prefer a credible +0.0015 or larger gain, not cosmetic refactoring."
        ),
        "truth_boundary": [
            "Only official MLE-Bench private-grader output counts as an official score or medal.",
            "OOF is an internal promotion signal and must not be called an official result.",
            "Private labels and private-score feedback are unavailable and must not be used.",
            "All model selection and stacking must be nested or cross-fitted without scoring fit rows.",
        ],
        "observed_run": observed,
        "required_output": {
            "diagnosis": ["specific source-grounded limitation"],
            "ranked_upgrades": [
                {
                    "id": "short stable id",
                    "expected_oof_auc_gain_low": "number",
                    "expected_oof_auc_gain_high": "number",
                    "compute_cost": "low, medium, or high",
                    "source_changes": ["function-level implementation action"],
                    "validation_contract": ["leakage-safe acceptance check"],
                    "stop_rule": "bounded stop rule",
                }
            ],
            "recommended_minimal_patch": {
                "functions_to_add_or_change": ["exact function names"],
                "algorithm": ["ordered implementation steps"],
                "new_cli_arguments": ["name, default, and purpose"],
                "artifact_changes": ["immutable OOF/test data to persist"],
                "acceptance_tests": ["specific deterministic test"],
            },
            "experiment_sequence": ["bounded experiments in execution order"],
            "go_no_go": {
                "implementation_verdict": "GO or NO-GO",
                "reason": "direct reason",
                "minimum_required_observed_gain": 0.0014825708,
            },
        },
        "review_instructions": [
            "Audit only this competition and the supplied current source.",
            "Separate confirmed source limitations from proposals.",
            "Prioritize new independent signal: word analyzer variants, character variants, label-correlation stacking, and strong pretrained text embeddings only when justified.",
            "Evaluate whether the current two-channel rank blend has already exhausted its likely upper bound.",
            "Any stacker must train on out-of-fold base predictions and predict a fold it never scored.",
            "Give exact implementation and tests suitable for immediate coding.",
            "Return one JSON object only with every required key.",
        ],
        "current_source": extract_definitions(SOURCE_PATH, REQUIRED_FUNCTIONS),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def validate_audit(audit: dict[str, Any]) -> None:
    required = {
        "diagnosis",
        "ranked_upgrades",
        "recommended_minimal_patch",
        "experiment_sequence",
        "go_no_go",
    }
    if not required <= set(audit):
        raise ValueError(f"Audit is missing required keys: {sorted(required - set(audit))}")
    if not isinstance(audit["diagnosis"], list) or not audit["diagnosis"]:
        raise ValueError("Audit diagnosis must be non-empty")
    upgrades = audit["ranked_upgrades"]
    if not isinstance(upgrades, list) or not upgrades:
        raise ValueError("Audit ranked_upgrades must be non-empty")
    for upgrade in upgrades:
        if not isinstance(upgrade, dict) or not {
            "id",
            "expected_oof_auc_gain_low",
            "expected_oof_auc_gain_high",
            "compute_cost",
            "source_changes",
            "validation_contract",
            "stop_rule",
        } <= set(upgrade):
            raise ValueError("Every ranked upgrade must satisfy the output contract")
    patch = audit["recommended_minimal_patch"]
    if not isinstance(patch, dict) or not {
        "functions_to_add_or_change",
        "algorithm",
        "new_cli_arguments",
        "artifact_changes",
        "acceptance_tests",
    } <= set(patch):
        raise ValueError("recommended_minimal_patch is incomplete")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--prompt-output", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:65068/v1"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.6-sol"))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-tokens", type=int, default=12_000)
    args = parser.parse_args(argv)
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY must be supplied from the secure runtime environment")
    summary = json.loads(args.summary.read_text(encoding="utf-8-sig"))
    prompt = build_prompt(summary)
    atomic_write(args.prompt_output, prompt + "\n")
    request_payload = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are EvoMind's senior competition NLP engineer. Audit the exact source and "
                    "return a bounded implementation plan that maximizes real private-grader medal probability."
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
        endpoint(args.base_url),
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
        raise RuntimeError(f"GPT-5.6 Jigsaw audit failed after three attempts: {last_error}")
    choice = (body.get("choices") or [{}])[0]
    content = str((choice.get("message") or {}).get("content") or "")
    audit = extract_json(content)
    validate_audit(audit)
    served_model = str(body.get("model") or "")
    if served_model != args.model:
        raise RuntimeError(f"Gateway served unexpected model: {served_model}")
    report = {
        "schema": "evomind.mlebench_lite.jigsaw_gpt56_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "planner": {
            "provider": "openai",
            "requested_model": args.model,
            "served_model": served_model,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "usage": body.get("usage") or {},
            "reasoning_effort_requested": "high",
            "json_mode_requested": True,
        },
        "summary_path": str(args.summary.resolve()),
        "summary_sha256": hashlib.sha256(args.summary.read_bytes()).hexdigest(),
        "source_path": str(SOURCE_PATH.resolve()),
        "source_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
        "prompt_path": str(args.prompt_output.resolve()),
        "prompt_sha256": sha256_text(prompt),
        "response_sha256": sha256_text(content),
        "parse_ok": True,
        "audit": audit,
        "secret_policy": "API key came from the secure child environment and is absent from artifacts.",
        "ok": True,
    }
    atomic_write(args.output, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "served_model": served_model,
                "usage": report["planner"]["usage"],
                "latency_ms": report["planner"]["latency_ms"],
                "parse_ok": True,
                "upgrade_ids": [row["id"] for row in audit["ranked_upgrades"]],
                "go_no_go": audit["go_no_go"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
