#!/usr/bin/env python3
"""Run a source-grounded GPT-5.6 review of scored non-medal recovery runners."""
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
DEFAULT_PROGRESS = PROJECT_ROOT / "workspace" / "mlebench_progress" / "lite11_current.json"
DEFAULT_PLAN = PROJECT_ROOT / "workspace" / "mlebench_plans" / "medal_recovery_gpt56_current.json"
DEFAULT_CANDIDATE_LEDGER = (
    PROJECT_ROOT / "workspace" / "mlebench_plans" / "medal_candidate_gates_current.json"
)
DEFAULT_PROMPT = (
    PROJECT_ROOT / "workspace" / "mlebench_plans" / "medal_recovery_deep_gpt56_prompt_current.txt"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "workspace" / "mlebench_plans" / "medal_recovery_deep_gpt56_review_current.json"
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Review response did not contain a JSON object")
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Review response must be a JSON object")
    return value


def extract_definitions(path: Path, names: Iterable[str]) -> str:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    wanted = set(names)
    tree = ast.parse(source)
    chunks: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
            chunks.append("\n".join(lines[node.lineno - 1 : node.end_lineno]))
    found = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
    }
    if found != wanted:
        raise RuntimeError(f"Missing requested source definitions in {path.name}: {sorted(wanted - found)}")
    return "\n\n".join(chunks)


def extract_assignments(path: Path, names: Iterable[str]) -> str:
    """Extract named top-level assignments without brittle line-number slices."""

    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    wanted = set(names)
    tree = ast.parse(source)
    chunks: list[str] = []
    found: set[str] = set()
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        matched = {
            target.id
            for target in targets
            if isinstance(target, ast.Name) and target.id in wanted
        }
        if matched:
            chunks.append("\n".join(lines[node.lineno - 1 : node.end_lineno]))
            found.update(matched)
    if found != wanted:
        raise RuntimeError(f"Missing requested source assignments in {path.name}: {sorted(wanted - found)}")
    return "\n\n".join(chunks)


def build_prompt(
    progress: dict[str, Any],
    plan: dict[str, Any],
    candidate_ledger: dict[str, Any],
) -> tuple[str, set[str]]:
    non_medals = [row for row in progress["results"] if not row["any_medal"]]
    candidate_ids = {str(row["competition_id"]) for row in non_medals}
    recovery_runner_ids = [
        "denoising-dirty-documents",
        "dogs-vs-cats-redux-kernels-edition",
        "leaf-classification",
        "siim-isic-melanoma-classification",
        "tabular-playground-series-may-2022",
        "aerial-cactus-identification",
        "spooky-author-identification",
    ]
    legacy_full_runner_ids = [
        "new-york-city-taxi-fare-prediction",
        "random-acts-of-pizza",
    ]
    wave2_recovery_runner_ids = [
        "ranzcr-clip-catheter-line-classification",
    ]
    implemented_ids = set(
        recovery_runner_ids + legacy_full_runner_ids + wave2_recovery_runner_ids
    )
    recovery_path = PROJECT_ROOT / "scripts" / "mlebench_medal_recovery_adapters.py"
    full_path = PROJECT_ROOT / "scripts" / "run_mlebench_lite_full.py"
    wave2_path = PROJECT_ROOT / "scripts" / "mlebench_wave2_adapters.py"
    siim_ablation_path = PROJECT_ROOT / "scripts" / "run_siim_preprocessing_ablation.py"
    sources = {
        "recovery_runner_definitions": extract_definitions(
            recovery_path,
            {
                "run_may2022_gpu_ensemble",
                "run_denoising_unet",
                "run_siim_image_metadata",
                "run_dogs_cats_convnext",
                "run_leaf_oof_ensemble",
                "run_aerial_cactus_convnext",
                "run_spooky_nbsvm_oof",
                "build_siim_multiview_fusion_model",
                "build_siim_image_content_manifest",
                "build_siim_content_connected_groups",
                "siim_crop_dark_border",
                "siim_shades_of_gray_color_constancy",
                "siim_suppress_dark_hairs",
                "siim_lesion_focus_crop",
                "prepare_siim_dermoscopy_views",
                "build_siim_patient_folds",
                "build_siim_nested_patient_folds",
                "aggregate_siim_inner_histories",
                "select_siim_channel_epochs",
                "_siim_center_focus_crop",
                "select_siim_preprocessing_ablation",
                "validate_siim_preprocessing_ablation_report",
                "build_siim_fold_harmonization_variants",
                "select_siim_crossfit_harmonization",
                "cross_fit_siim_harmonized_multichannel_blend",
                "write_siim_artifact_manifest",
                "verify_image_decode_manifest",
                "build_spooky_stylometric_features",
                "fit_spooky_stylometric_channel",
                "apply_spooky_multicomponent_blend",
                "select_spooky_multicomponent_blend",
                "cross_fit_spooky_multicomponent_blend",
            },
        ),
        "legacy_runner_definitions": extract_definitions(
            full_path,
            {
                "run_taxi",
                "taxi_features",
                "build_taxi_oof_splits",
                "run_pizza",
                "fit_pizza_nbsvm_fold",
                "pizza_structured_features",
                "cross_fit_binary_auc_blend",
            },
        ),
        "vision_runtime_definitions": extract_definitions(
            wave2_path,
            {
                "_vision_model",
                "_run_vision",
                "make_vision_splits",
                "vision_splits_are_scoreable",
                "evaluate_vision_promotion_gate",
                "run_ranzcr",
                "fit_binary_temperature_intercept",
                "apply_binary_temperature_intercept",
                "cross_fit_binary_logloss_calibration",
            },
        ),
        "siim_preprocessing_ablation_definitions": extract_definitions(
            siim_ablation_path,
            {
                "build_embedding_cache_contract",
                "load_public_dataset",
                "build_manifest_and_folds",
                "load_frozen_backbone",
                "verified_embedding_cache",
                "extract_view_embeddings",
                "evaluate_frozen_linear_head",
                "main",
            },
        ),
        "siim_preprocessing_profiles": extract_assignments(
            recovery_path,
            {"SIIM_PREPROCESSING_PROFILES", "SIIM_PREPROCESSING_PROFILE_STEPS"},
        ),
        "production_runner_registry": extract_assignments(full_path, {"RUNNERS"}),
        "recovery_registry": extract_assignments(recovery_path, {"RUNNERS"}),
    }
    payload = {
        "objective": (
            "Reach at least 18/22 official MLE-Bench Lite medals for each benchmark seed, "
            "then report at least three external seeds as mean plus SEM and strictly exceed 80.30%."
        ),
        "truth_boundary": [
            "Only official MLE-Bench private-grader output counts as a medal.",
            "CV, OOF, a valid submission, or grader execution alone is not a medal.",
            "No Kaggle auto-submission and no private-score feedback as a training signal.",
            "Do not invent completed scores or medals.",
        ],
        "current_progress": {
            "officially_scored": progress["scored_competitions"],
            "lite_total": progress["lite_total_competitions"],
            "current_medals": progress["any_medal_count"],
            "medals_required": progress["medals_required_to_strictly_exceed_top"],
            "non_medals": non_medals,
        },
        "existing_plan_summary": {
            "priority_order": plan.get("priority_order"),
            "strategies": plan.get("strategies"),
            "stop_rules": plan.get("stop_rules"),
            "rationale_summary": plan.get("rationale_summary"),
        },
        "current_internal_validation_ledger": {
            "truth_boundary": candidate_ledger.get("truth_boundary"),
            "unscored_candidates": candidate_ledger.get(
                "unscored_internal_gate_pass_waiting_official_grader"
            ),
            "scored_non_medal_recovery_candidates": candidate_ledger.get(
                "scored_non_medal_internal_recovery_candidates"
            ),
            "not_promoted": candidate_ledger.get("not_promoted"),
            "audit_only": candidate_ledger.get("audit_only_no_recovery_gain"),
        },
        "implementation_inventory": {
            "recovery_runner_ids": recovery_runner_ids,
            "legacy_full_runner_ids": legacy_full_runner_ids,
            "wave2_recovery_runner_ids": wave2_recovery_runner_ids,
            "implemented_medal_recovery_count": len(implemented_ids),
            "missing_medal_recovery_implementations": sorted(
                candidate_ids - implemented_ids
            ),
        },
        "required_output": {
            "audit_summary": {
                "current_coverage": "integer",
                "required_coverage": len(candidate_ids),
                "critical_findings": ["source-grounded finding"],
            },
            "per_competition": [
                {
                    "competition_id": "exact id from allowed ids",
                    "current_runner": "function name or missing",
                    "deployment_verdict": "GO or NO-GO",
                    "confirmed_source_defects": ["specific source-grounded defect"],
                    "highest_value_upgrades": ["implementation-level action"],
                    "validation_contract": ["fixed-fold and artifact requirement"],
                    "estimated_medal_probability_after_upgrade": "number from 0 to 1",
                    "compute_cost": "low, medium, or high",
                    "stop_rule": "bounded rule",
                }
            ],
            "implementation_order": ["all exact allowed ids in recommended order"],
            "immediate_code_changes": [
                {
                    "competition_id": "exact id",
                    "file": "repo-relative path",
                    "functions_to_add_or_change": ["name"],
                    "acceptance_tests": ["test"],
                }
            ],
            "exact_medal_paths": {
                "base_case": "integer converted medals expected across Wave2 plus recovery",
                "plausible_range": ["low integer", "high integer"],
                "minimum_path_to_18": ["competition ids"],
                "contingency_path": ["competition ids"],
            },
            "resource_schedule": ["ordered bounded waves that fit one A800 80GB"],
            "global_stop_rules": ["rule"],
        },
        "allowed_competition_ids": sorted(candidate_ids),
        "review_instructions": [
            "Audit every allowed competition exactly once.",
            "Treat the supplied implementation inventory and exact registry assignments as authoritative current source.",
            "Do not report a function as missing when its definition and registry entry are supplied.",
            "Use the internal validation ledger as evidence, while preserving internal_candidate != official_medal.",
            "Ground defects in the supplied source; distinguish confirmed defect from proposal.",
            "Prefer OOF fold ensembles, full-data refit where appropriate, immutable artifacts, and deterministic seeds.",
            "Flag leakage, holdout-only test training, missing pretrained weights, row-order risk, class-order risk, and weak features.",
            "Prioritize actual medal conversion probability, not ease of implementation.",
            "Return one JSON object only and include all required keys.",
        ],
        "source_files": sources,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")), candidate_ids


def validate_review(review: dict[str, Any], candidate_ids: set[str]) -> None:
    rows = review.get("per_competition")
    if not isinstance(rows, list) or len(rows) != len(candidate_ids):
        raise ValueError(
            "per_competition must audit every currently scored non-medal"
        )
    returned = [str(row.get("competition_id")) for row in rows if isinstance(row, dict)]
    if len(returned) != len(set(returned)) or set(returned) != candidate_ids:
        raise ValueError("per_competition ids must exactly match scored non-medals")
    order = review.get("implementation_order")
    if not isinstance(order, list) or set(map(str, order)) != candidate_ids:
        raise ValueError("implementation_order must contain all scored non-medals exactly once")
    required = {
        "audit_summary",
        "per_competition",
        "implementation_order",
        "immediate_code_changes",
        "exact_medal_paths",
        "resource_schedule",
        "global_stop_rules",
    }
    if not required <= set(review):
        raise ValueError(f"Review is missing required keys: {sorted(required - set(review))}")


def atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--candidate-ledger", type=Path, default=DEFAULT_CANDIDATE_LEDGER)
    parser.add_argument("--prompt-output", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:65068/v1"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.6-sol"))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-tokens", type=int, default=30_000)
    args = parser.parse_args(argv)
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY must be supplied from the secure runtime environment")
    progress = json.loads(args.progress.read_text(encoding="utf-8-sig"))
    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    candidate_ledger = json.loads(args.candidate_ledger.read_text(encoding="utf-8-sig"))
    prompt, candidate_ids = build_prompt(progress, plan, candidate_ledger)
    atomic_write(args.prompt_output, prompt + "\n")
    request_payload = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are EvoMind's senior MLE-Bench medal engineer. Perform a strict source-code audit. "
                    "Optimize for real official private-grader medal conversion, not reassuring prose."
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
        raise RuntimeError(f"GPT-5.6 review request failed after three attempts: {last_error}")
    choice = (body.get("choices") or [{}])[0]
    content = str((choice.get("message") or {}).get("content") or "")
    review = extract_json(content)
    validate_review(review, candidate_ids)
    served_model = str(body.get("model") or "")
    if served_model != args.model:
        raise RuntimeError(f"Gateway served unexpected model: {served_model}")
    report = {
        "schema": "evomind.mlebench_lite.medal_recovery_deep_gpt56_review.v1",
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
        "prompt_sha256": sha256_text(prompt),
        "response_sha256": sha256_text(content),
        "parse_ok": True,
        "review": review,
        "secret_policy": "API key was loaded from the secure runtime environment and is absent from artifacts.",
        "ok": True,
    }
    atomic_write(args.output, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "served_model": served_model,
        "usage": report["planner"]["usage"],
        "latency_ms": report["planner"]["latency_ms"],
        "parse_ok": True,
        "implementation_order": review["implementation_order"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
