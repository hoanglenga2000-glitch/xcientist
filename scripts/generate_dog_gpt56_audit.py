#!/usr/bin/env python3
"""Run a source-grounded GPT-5.6 audit for Dog Breed recovery training."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "scripts" / "mlebench_wave2_adapters.py"
DEFAULT_PLAN = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "dog_breed_imagenet_head_recovery_current.json"
)
DEFAULT_PROBE = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "dog_breed_imagenet_head_runtime_probe_current.json"
)
DEFAULT_PROGRESS = (
    PROJECT_ROOT / "workspace" / "mlebench_progress" / "lite11_current.json"
)
DEFAULT_PROMPT = (
    PROJECT_ROOT / "workspace" / "mlebench_plans" / "dog_breed_gpt56_audit_prompt_current.txt"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "workspace" / "mlebench_plans" / "dog_breed_gpt56_audit_current.json"
)
REQUIRED_MODEL = "gpt-5.6-sol"
DOG_ID = "dog-breed-identification"
PROMOTION_LOG_LOSS = 0.040
REQUIRED_FUNCTIONS = {
    "initialize_classifier_from_imagenet_rows",
    "_vision_model",
    "build_dog_breed_imagenet_mapping",
    "apply_dog_breed_probability_blend",
    "select_dog_breed_probability_blend",
    "cross_fit_dog_breed_probability_blend",
    "load_or_compute_dog_breed_imagenet_teacher",
    "run_dog_breed",
}
RUN_VISION_MARKERS = (
    "dog_breed_classifier_source_indices = [",
    "classifier_source_indices=dog_breed_classifier_source_indices",
    '"mode": "classifier_linear_1.0x_pretrained_and_classifier_norm_0.1x",',
    "cross_fitted_blend, cross_fitted_records =",
    'promotion_extra_checks["imagenet_classifier_rows_initialized_all_folds"]',
)
EPOCH_PATTERN = re.compile(
    r"fold=(?P<fold>\d+)/(?P<folds>\d+)\s+epoch=(?P<epoch>\d+)\s+"
    r"cv=(?P<cv>[-+0-9.eE]+)\s+images_per_second=(?P<ips>[-+0-9.eE]+)"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def assert_local_gateway(base_url: str) -> None:
    parsed = urlparse(base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.port != 65068
        or parsed.path.rstrip("/") not in {"/v1", "/v1/chat/completions"}
    ):
        raise RuntimeError("Dog audit requires the configured loopback gateway on port 65068")


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
            body = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            chunks.append(f"# source lines {node.lineno}-{node.end_lineno}\n{body}")
            found.add(node.name)
    if found != wanted:
        raise RuntimeError(f"Missing Dog Breed definitions: {sorted(wanted - found)}")
    return "\n\n".join(chunks)


def extract_marker_windows(
    path: Path,
    markers: Sequence[str],
    *,
    before: int = 12,
    after: int = 32,
) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    windows: list[tuple[int, int]] = []
    for marker in markers:
        matches = [index for index, line in enumerate(lines) if marker in line]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one source marker occurrence: {marker!r}; found={len(matches)}")
        index = matches[0]
        windows.append((max(0, index - before), min(len(lines), index + after + 1)))
    merged: list[tuple[int, int]] = []
    for start, end in sorted(windows):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    chunks = []
    for start, end in merged:
        body = "\n".join(f"{number + 1}: {lines[number]}" for number in range(start, end))
        chunks.append(f"# _run_vision source window {start + 1}-{end}\n{body}")
    return "\n\n".join(chunks)


def resolve_active_log(plan: dict[str, Any], explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    run_id = str((plan.get("active_parent") or {}).get("run_id") or "").strip()
    if not run_id:
        raise ValueError("Recovery plan is missing active_parent.run_id")
    return (
        PROJECT_ROOT
        / "workspace"
        / "local_gpu"
        / "mlebench_lite_runs"
        / run_id
        / "full.log"
    )


def parse_training_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = EPOCH_PATTERN.search(line)
        if not match:
            continue
        rows.append(
            {
                "fold": int(match.group("fold")),
                "folds": int(match.group("folds")),
                "epoch": int(match.group("epoch")),
                "cv_log_loss": float(match.group("cv")),
                "images_per_second": float(match.group("ips")),
            }
        )
    latest_by_fold: dict[str, dict[str, Any]] = {}
    for row in rows:
        latest_by_fold[str(row["fold"])] = row
    return {
        "path": str(path.resolve()),
        "sha256": sha256_text(text),
        "epoch_records": len(rows),
        "latest": rows[-1] if rows else None,
        "best_observed_epoch": min(rows, key=lambda row: row["cv_log_loss"]) if rows else None,
        "latest_by_fold": latest_by_fold,
        "terminal_summary_present": "summary.json" in text or "official grader" in text.lower(),
    }


def validate_inputs(
    plan: dict[str, Any],
    probe: dict[str, Any],
    progress: dict[str, Any],
) -> None:
    if plan.get("schema") != "evomind.mlebench.dog_breed_imagenet_head_recovery_plan.v1":
        raise ValueError("Unexpected Dog Breed recovery-plan schema")
    if plan.get("competition_id") != DOG_ID:
        raise ValueError("Recovery plan targets the wrong competition")
    teacher = plan.get("frozen_teacher_evidence") or {}
    revision = plan.get("material_revision") or {}
    if int(teacher.get("rows", 0)) <= 0 or int(teacher.get("classes", 0)) != 120:
        raise ValueError("Frozen teacher evidence must cover all 120 Dog Breed classes")
    if teacher.get("private_labels_used") is not False:
        raise ValueError("Frozen teacher evidence must exclude private labels")
    if float(plan.get("promotion_threshold", -1.0)) != PROMOTION_LOG_LOSS:
        raise ValueError("Dog Breed promotion threshold changed unexpectedly")
    if revision.get("adapter_sha256") != sha256_bytes(SOURCE_PATH.read_bytes()):
        raise ValueError("Recovery-plan adapter hash does not match the current source")
    if probe.get("schema") != "evomind.dog_breed.imagenet_head_runtime_probe.v1":
        raise ValueError("Unexpected Dog Breed runtime-probe schema")
    if probe.get("runtime_contract_passed") is not True or probe.get("cuda_available") is not True:
        raise ValueError("Dog Breed ImageNet-head runtime contract has not passed on CUDA")
    if probe.get("private_labels_used") is not False:
        raise ValueError("Runtime probe must exclude private labels")
    if probe.get("pinned_weight_sha256") != revision.get("pinned_weight_sha256"):
        raise ValueError("Pinned ConvNeXt weight identities disagree")
    production_execution = probe.get("production_classifier_execution") or {}
    if (
        int(production_execution.get("output_count", 0)) != 120
        or production_execution.get("weights_copied_exactly") is not True
        or production_execution.get("bias_copied_exactly") is not True
        or float(production_execution.get("full_classifier_logit_max_abs_delta", -1.0)) != 0.0
        or float(production_execution.get("full_classifier_probability_max_abs_delta", -1.0))
        != 0.0
    ):
        raise ValueError("Runtime probe lacks exact full 120-class production execution proof")
    if int(progress.get("lite_total_competitions", 0)) != 22:
        raise ValueError("Official progress is not the MLE-Bench Lite 22 campaign")
    if int(progress.get("medals_required_to_strictly_exceed_top", 0)) != 18:
        raise ValueError("Official progress no longer encodes the 18/22 target")
    if DOG_ID not in set(progress.get("remaining_competition_ids") or []):
        raise ValueError("Dog Breed is no longer an unscored official campaign item")


def build_prompt(
    plan: dict[str, Any],
    probe: dict[str, Any],
    progress: dict[str, Any],
    active_log: dict[str, Any],
) -> str:
    validate_inputs(plan, probe, progress)
    teacher = plan["frozen_teacher_evidence"]
    teacher_log_loss = float(teacher["oof_like_train_log_loss"])
    payload = {
        "objective": (
            "Audit the exact revised Dog Breed pipeline and decide the smallest bounded experiment "
            "sequence most likely to reach complete leakage-safe OOF multiclass log loss <=0.040, "
            "then permit the official MLE-Bench private grader. This medal is one of five still "
            "needed to reach 18/22 and strictly exceed the public 80.30% Lite leader."
        ),
        "truth_boundary": {
            "official_result_rule": "Only an official MLE-Bench private-grader result counts as a score or medal.",
            "oof_rule": "OOF, teacher train log loss, and fold CV are internal evidence, never official results.",
            "private_labels_rule": "Private answers and private-score feedback are excluded from training and selection.",
            "campaign_success": "At least 18 of 22 official medals; Dog Breed alone does not prove campaign success.",
        },
        "verified_state": {
            "official_campaign": {
                "scored": progress.get("scored_competitions"),
                "medals": progress.get("any_medal_count"),
                "medal_deficit": progress.get("medal_deficit"),
                "target_medals": progress.get("medals_required_to_strictly_exceed_top"),
            },
            "active_old_revision_run": active_log,
            "active_old_revision_source_sha256": (plan.get("active_parent") or {}).get(
                "adapter_sha256_at_start"
            ),
            "revised_source_sha256": (plan.get("material_revision") or {}).get("adapter_sha256"),
            "frozen_teacher": teacher,
            "runtime_probe": probe,
            "target_math": {
                "teacher_train_log_loss": teacher_log_loss,
                "promotion_log_loss": PROMOTION_LOG_LOSS,
                "absolute_improvement_needed_from_teacher": teacher_log_loss - PROMOTION_LOG_LOSS,
                "relative_reduction_needed_from_teacher": (
                    teacher_log_loss - PROMOTION_LOG_LOSS
                )
                / teacher_log_loss,
            },
            "queued_revision": plan.get("material_revision"),
            "launch_policy": plan.get("launch_policy"),
        },
        "required_output": {
            "truth_boundary_acknowledgement": {
                "official_grader_only": True,
                "oof_not_official": True,
                "private_labels_unused": True,
                "dog_medal_not_campaign_completion": True,
            },
            "source_findings": [
                {
                    "finding": "specific confirmed behavior or limitation",
                    "evidence": ["exact supplied function, contract, hash, or measured value"],
                    "confidence": "high, medium, or low",
                }
            ],
            "risk_assessment": {
                "target_feasibility": "high, medium, or low",
                "largest_generalization_risks": ["specific risk"],
                "why_teacher_0_18795_is_not_enough": "quantitative explanation",
            },
            "bounded_experiments": [
                {
                    "id": "short stable id",
                    "hypothesis": "one falsifiable hypothesis",
                    "exact_source_changes": ["function-level implementation action"],
                    "resource_budget": "one bounded RTX 4060 or A800 budget",
                    "acceptance_oof_log_loss": 0.04,
                    "early_checkpoint_rule": "measurable first-fold or early-epoch rule",
                    "stop_rule": "hard bounded stop rule",
                    "rollback": "exact rollback condition",
                }
            ],
            "recommended_next_run": {
                "implementation_verdict": "GO or NO-GO",
                "base_revision": "source SHA256",
                "cli_arguments": ["exact argument=value"],
                "expected_oof_log_loss_low": 0.0,
                "expected_oof_log_loss_high": 1.0,
                "promotion_gate_log_loss": 0.04,
                "resource_fit": "RTX 4060 8GB or A800 with reason",
                "stop_rules": ["hard stop rule"],
            },
            "campaign_impact": {
                "official_medal_if_passed": 1,
                "medals_still_needed_after_dog": 4,
                "next_competitions_after_dog": ["ranked competition id"],
            },
        },
        "review_instructions": [
            "Audit only the supplied current source and verified artifacts; separate confirmed behavior from proposals.",
            "Judge the exact ImageNet 120-row classifier initialization and the backbone 0.1x/classifier 1.0x optimizer groups.",
            "Quantify why 0.1879528618 remains far from 0.040 and do not treat high top-1 accuracy as sufficient for log loss.",
            "Prefer calibration-aware, leakage-safe changes that improve probability quality across 120 classes.",
            "Limit bounded_experiments to at most five and specify early termination so GPU time is not wasted.",
            "The currently running process uses the old source revision; never recommend mutating it in place.",
            "Return one JSON object only and populate every required key.",
        ],
        "current_source_definitions": extract_definitions(SOURCE_PATH, REQUIRED_FUNCTIONS),
        "current_run_vision_windows": extract_marker_windows(SOURCE_PATH, RUN_VISION_MARKERS),
    }
    prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(prompt) > 90_000:
        raise RuntimeError(f"Dog audit prompt is unexpectedly large: {len(prompt)}")
    return prompt


def validate_audit(audit: dict[str, Any]) -> None:
    required = {
        "truth_boundary_acknowledgement",
        "source_findings",
        "risk_assessment",
        "bounded_experiments",
        "recommended_next_run",
        "campaign_impact",
    }
    if not required <= set(audit):
        raise ValueError(f"Audit is missing required keys: {sorted(required - set(audit))}")
    truth = audit["truth_boundary_acknowledgement"]
    truth_keys = {
        "official_grader_only",
        "oof_not_official",
        "private_labels_unused",
        "dog_medal_not_campaign_completion",
    }
    if not isinstance(truth, dict) or any(truth.get(key) is not True for key in truth_keys):
        raise ValueError("Audit did not preserve the official-result truth boundary")
    findings = audit["source_findings"]
    if not isinstance(findings, list) or not findings:
        raise ValueError("Audit source_findings must be non-empty")
    for finding in findings:
        if not isinstance(finding, dict) or not {"finding", "evidence", "confidence"} <= set(finding):
            raise ValueError("Every source finding must satisfy the output contract")
    risk = audit["risk_assessment"]
    if not isinstance(risk, dict) or not {
        "target_feasibility",
        "largest_generalization_risks",
        "why_teacher_0_18795_is_not_enough",
    } <= set(risk):
        raise ValueError("risk_assessment is incomplete")
    experiments = audit["bounded_experiments"]
    if not isinstance(experiments, list) or not 1 <= len(experiments) <= 5:
        raise ValueError("bounded_experiments must contain between one and five experiments")
    experiment_keys = {
        "id",
        "hypothesis",
        "exact_source_changes",
        "resource_budget",
        "acceptance_oof_log_loss",
        "early_checkpoint_rule",
        "stop_rule",
        "rollback",
    }
    identifiers: list[str] = []
    for experiment in experiments:
        if not isinstance(experiment, dict) or not experiment_keys <= set(experiment):
            raise ValueError("Every bounded experiment must satisfy the output contract")
        identifiers.append(str(experiment["id"]))
        threshold = float(experiment["acceptance_oof_log_loss"])
        if not 0.0 < threshold <= 1.0:
            raise ValueError("Experiment acceptance_oof_log_loss is outside (0, 1]")
        if not str(experiment["stop_rule"]).strip():
            raise ValueError("Every bounded experiment requires a stop rule")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Bounded experiment ids must be unique")
    run = audit["recommended_next_run"]
    run_keys = {
        "implementation_verdict",
        "base_revision",
        "cli_arguments",
        "expected_oof_log_loss_low",
        "expected_oof_log_loss_high",
        "promotion_gate_log_loss",
        "resource_fit",
        "stop_rules",
    }
    if not isinstance(run, dict) or not run_keys <= set(run):
        raise ValueError("recommended_next_run is incomplete")
    if run["implementation_verdict"] not in {"GO", "NO-GO"}:
        raise ValueError("recommended_next_run implementation_verdict must be GO or NO-GO")
    if float(run["promotion_gate_log_loss"]) != PROMOTION_LOG_LOSS:
        raise ValueError("recommended_next_run changed the promotion gate")
    if not isinstance(run["stop_rules"], list) or not run["stop_rules"]:
        raise ValueError("recommended_next_run requires at least one stop rule")
    campaign = audit["campaign_impact"]
    if not isinstance(campaign, dict) or not {
        "official_medal_if_passed",
        "medals_still_needed_after_dog",
        "next_competitions_after_dog",
    } <= set(campaign):
        raise ValueError("campaign_impact is incomplete")
    if int(campaign["official_medal_if_passed"]) != 1:
        raise ValueError("Dog Breed can contribute exactly one official medal")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--active-log", type=Path)
    parser.add_argument("--prompt-output", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:65068/v1"),
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", REQUIRED_MODEL))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-tokens", type=int, default=14_000)
    args = parser.parse_args(argv)
    if args.model != REQUIRED_MODEL:
        raise RuntimeError(f"Dog audit requires {REQUIRED_MODEL}")
    assert_local_gateway(args.base_url)
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY must be supplied from the secure runtime environment")

    plan = read_json(args.plan)
    probe = read_json(args.probe)
    progress = read_json(args.progress)
    active_log_path = resolve_active_log(plan, args.active_log)
    active_log = parse_training_log(active_log_path)
    prompt = build_prompt(plan, probe, progress, active_log)
    atomic_write(args.prompt_output, prompt + "\n")
    request_payload = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are EvoMind's senior computer-vision competition engineer. Audit exact "
                    "source and evidence, then return a bounded probability-quality optimization "
                    "decision that maximizes official private-grader medal probability."
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
                decoded = json.loads(response.read().decode("utf-8"))
            if not isinstance(decoded, dict):
                raise ValueError("Gateway response was not a JSON object")
            body = decoded
            break
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
        except (OSError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_error = type(exc).__name__
        if attempt < 2:
            time.sleep(2.0 * (attempt + 1))
    if body is None:
        raise RuntimeError(f"GPT-5.6 Dog Breed audit failed after three attempts: {last_error}")

    choice = (body.get("choices") or [{}])[0]
    content = str((choice.get("message") or {}).get("content") or "")
    audit = extract_json(content)
    validate_audit(audit)
    served_model = str(body.get("model") or "")
    if served_model != REQUIRED_MODEL:
        raise RuntimeError(f"Gateway served unexpected model: {served_model}")
    report = {
        "schema": "evomind.mlebench_lite.dog_breed_gpt56_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "planner": {
            "provider": "openai",
            "requested_model": REQUIRED_MODEL,
            "served_model": served_model,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "usage": body.get("usage") or {},
            "reasoning_effort_requested": "high",
            "json_mode_requested": True,
        },
        "inputs": {
            "plan": {"path": str(args.plan.resolve()), "sha256": sha256_bytes(args.plan.read_bytes())},
            "probe": {"path": str(args.probe.resolve()), "sha256": sha256_bytes(args.probe.read_bytes())},
            "progress": {
                "path": str(args.progress.resolve()),
                "sha256": sha256_bytes(args.progress.read_bytes()),
            },
            "active_log": active_log,
            "source": {"path": str(SOURCE_PATH.resolve()), "sha256": sha256_bytes(SOURCE_PATH.read_bytes())},
        },
        "prompt": {"path": str(args.prompt_output.resolve()), "sha256": sha256_text(prompt)},
        "response_sha256": sha256_text(content),
        "parse_ok": True,
        "audit": audit,
        "secret_policy": "API key came from the DPAPI-backed child environment and is absent from artifacts.",
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
                "prompt_characters": len(prompt),
                "parse_ok": True,
                "experiment_ids": [row["id"] for row in audit["bounded_experiments"]],
                "recommended_next_run": audit["recommended_next_run"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
