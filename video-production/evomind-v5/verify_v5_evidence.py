from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

TASK_ID = "evomind-qwen7b-finetune"
EXPECTED_REPORT_RENDERER = "EvoMind Nature Skills"
REQUIRED_RUN_FILES = (
    "run.json",
    "refinement.json",
    "version.json",
    "version_comparison.json",
    "human_gate.json",
    "review.json",
    "claim_audit.json",
    "artifact_manifest.json",
    "evaluation_summary.json",
    "llm_output/metrics.json",
    "llm_output/telemetry.jsonl",
    "llm_output/adapter_reload.json",
    "llm_output/adapter/adapter_config.json",
    "llm_output/adapter/adapter_model.safetensors",
)
REQUIRED_REPORT_ARTIFACTS = (
    "report-html",
    "report-pdf",
    "version-comparison",
    "final-bundle",
)
REQUIRED_CAPTURES = {
    "v5-report-delivery-v1.mp4": 40.0,
    "v5-refinement-request.mp4": 19.0,
    "v5-refinement-completed.mp4": 7.0,
    "v5-comparison-final-delivery.mp4": 7.0,
}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric(payload: dict[str, Any]) -> float | None:
    after = payload.get("after")
    value = after.get("domain_composite") if isinstance(after, dict) else None
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def media_duration(ffprobe: Path, source: Path) -> float:
    result = subprocess.run(
        [str(ffprobe), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(source)],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def inside(root: Path, path_value: str) -> Path:
    target = (root / path_value).resolve()
    target.relative_to(root.resolve())
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed evidence gate for the EvoMind V5 success cut.")
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--ffprobe", type=Path, default=Path("ffprobe"))
    parser.add_argument("--require-captures", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    root = args.workspace_root.resolve()
    video_root = args.video_root.resolve()
    blockers: list[str] = []
    checks: list[str] = []

    pointer_path = root / "workspace" / "current_run.json"
    try:
        pointer = read_json(pointer_path)
    except Exception as exc:
        pointer = {}
        blockers.append(f"current run pointer unavailable: {exc}")
    run_id = str(pointer.get("run_id") or "")
    if pointer.get("task_id") != TASK_ID or not run_id:
        blockers.append("workspace/current_run.json is not bound to the EvoMind 7B refinement task")
    run_dir = root / "workspace" / "evomind_runs" / run_id

    missing_run_files = [item for item in REQUIRED_RUN_FILES if not (run_dir / item).is_file()]
    if missing_run_files:
        blockers.append("missing V2 evidence: " + ", ".join(missing_run_files))

    payloads: dict[str, dict[str, Any]] = {}
    for name in ("run", "refinement", "version", "version_comparison", "human_gate", "review", "claim_audit", "artifact_manifest", "metrics", "adapter_reload"):
        relative = "llm_output/metrics.json" if name == "metrics" else "llm_output/adapter_reload.json" if name == "adapter_reload" else f"{name}.json"
        path = run_dir / relative
        if path.is_file():
            try:
                payloads[name] = read_json(path)
            except Exception as exc:
                blockers.append(f"invalid {relative}: {exc}")

    run = payloads.get("run", {})
    gates = run.get("gates") if isinstance(run.get("gates"), dict) else {}
    if run.get("status") != "completed":
        blockers.append(f"V2 run status is {run.get('status') or 'missing'}, expected completed")
    for gate in ("hpc_execution", "reviewer", "claim_audit", "adapter_reload"):
        if gates.get(gate) != "passed":
            blockers.append(f"V2 gate {gate} is not passed")
    if gates.get("model_publication") != "blocked":
        blockers.append("model publication gate is not blocked")

    refinement = payloads.get("refinement", {})
    version = payloads.get("version", {})
    comparison = payloads.get("version_comparison", {})
    review = payloads.get("review", {})
    claim_audit = payloads.get("claim_audit", {})
    adapter_reload = payloads.get("adapter_reload", {})
    parent_run_id = str(refinement.get("parent_run_id") or version.get("parent_run_id") or "")
    parent_dir = root / "workspace" / "evomind_runs" / parent_run_id
    parent_metrics_path = parent_dir / "llm_output" / "metrics.json"
    child_metrics_path = run_dir / "llm_output" / "metrics.json"
    if not parent_metrics_path.is_file():
        blockers.append("preserved V1 metrics are unavailable")
    if version.get("parent_preserved") is not True or not parent_dir.is_dir():
        blockers.append("V1 immutable preservation evidence is missing")
    if payloads.get("human_gate", {}).get("decision") != "approved":
        blockers.append("V2 Human Gate is not approved")

    if parent_metrics_path.is_file() and child_metrics_path.is_file():
        parent_score = metric(read_json(parent_metrics_path))
        child_score = metric(payloads.get("metrics", {}))
        if parent_score is None or child_score is None:
            blockers.append("fixed-test V1/V2 scores are missing")
        else:
            delta = child_score - parent_score
            expected_outcome = "improved" if delta > 0.25 else "trade_off_detected" if delta < -1.0 else "no_material_change"
            comparisons = (
                comparison.get("schema") == "evomind.llm_version_comparison.v1",
                comparison.get("parent_run_id") == parent_run_id,
                comparison.get("child_run_id") == run_id,
                comparison.get("metric") == "fixed_test_domain_composite",
                math.isclose(float(comparison.get("v1", math.nan)), parent_score, abs_tol=1e-9),
                math.isclose(float(comparison.get("v2", math.nan)), child_score, abs_tol=1e-9),
                math.isclose(float(comparison.get("delta_pp", math.nan)), delta, abs_tol=1e-9),
                comparison.get("outcome") == expected_outcome,
                comparison.get("parent_metrics_sha256") == sha256(parent_metrics_path),
                comparison.get("child_metrics_sha256") == sha256(child_metrics_path),
                comparison.get("generated_by") == "VersionComparatorAgent",
            )
            if not all(comparisons):
                blockers.append("version_comparison.json is not bound to the V1/V2 metric ledgers")
            else:
                checks.append(f"reviewed version comparison: {expected_outcome}")

    review_checks = review.get("checks") if isinstance(review.get("checks"), dict) else {}
    if review.get("status") != "passed" or review.get("parent_subjective_summary_received") is not False:
        blockers.append("Independent Reviewer did not pass on evidence-only input")
    for name in ("version_comparison_recomputed", "parent_run_preserved"):
        if review_checks.get(name) is not True:
            blockers.append(f"Reviewer check {name} is not true")
    if claim_audit.get("status") != "passed":
        blockers.append("Claim Audit did not pass")
    if adapter_reload.get("passed") is not True:
        blockers.append("V2 Adapter reload validation did not pass")

    source_manifest = payloads.get("artifact_manifest", {})
    entries = source_manifest.get("artifacts") if isinstance(source_manifest.get("artifacts"), list) else []
    entry_by_path = {str(item.get("path")): item for item in entries if isinstance(item, dict)}
    comparison_entry = entry_by_path.get("version_comparison.json", {})
    comparison_path = run_dir / "version_comparison.json"
    if comparison_path.is_file() and (
        comparison_entry.get("sha256") != sha256(comparison_path)
        or comparison_entry.get("bytes") != comparison_path.stat().st_size
    ):
        blockers.append("source artifact manifest does not bind version_comparison.json")

    report_manifest_path = root / "workspace" / "tasks" / TASK_ID / "reports" / "scientific" / run_id / "artifact-manifest.json"
    report: dict[str, Any] = {}
    if report_manifest_path.is_file():
        try:
            report = read_json(report_manifest_path)
        except Exception as exc:
            blockers.append(f"invalid V2 scientific report manifest: {exc}")
    else:
        blockers.append("Scientific Report V2 manifest is missing")
    if report:
        if report.get("status") != "ready" or report.get("renderer") != EXPECTED_REPORT_RENDERER or report.get("version") != "V2":
            blockers.append("Scientific Report V2 is not a ready Nature Skills package")
        package_comparison = report.get("version_comparison")
        if not isinstance(package_comparison, dict) or package_comparison.get("schema") != "evomind.llm_version_comparison.v1":
            blockers.append("Scientific Report V2 does not expose the reviewed structured comparison")
        artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), list) else []
        artifacts_by_id = {str(item.get("id")): item for item in artifacts if isinstance(item, dict)}
        for artifact_id in REQUIRED_REPORT_ARTIFACTS:
            item = artifacts_by_id.get(artifact_id, {})
            artifact_path = item.get("path")
            if item.get("status") != "ready" or not isinstance(artifact_path, str):
                blockers.append(f"report artifact {artifact_id} is not ready")
                continue
            try:
                target = inside(root, artifact_path)
            except Exception:
                blockers.append(f"report artifact {artifact_id} escapes the workspace")
                continue
            if not target.is_file() or item.get("bytes") != target.stat().st_size or item.get("sha256") != sha256(target):
                blockers.append(f"report artifact {artifact_id} failed path, size or SHA256 verification")
        figures = report.get("figures") if isinstance(report.get("figures"), list) else []
        if not any(isinstance(item, dict) and item.get("status") == "ready" for item in figures):
            blockers.append("Scientific Report V2 has no reviewed figure")

    capture_results: dict[str, Any] = {}
    if args.require_captures:
        for filename, minimum in REQUIRED_CAPTURES.items():
            source = video_root / "captures" / filename
            if not source.is_file():
                blockers.append(f"missing real capture: {filename}")
                continue
            try:
                duration = media_duration(args.ffprobe, source)
            except Exception as exc:
                blockers.append(f"capture probe failed for {filename}: {exc}")
                continue
            if duration < minimum:
                blockers.append(f"capture {filename} is {duration:.2f}s, expected at least {minimum:.2f}s")
            capture_results[filename] = {"duration": duration, "bytes": source.stat().st_size, "sha256": sha256(source)}

    result = {
        "schema": "evomind.video.v5_evidence_gate.v1",
        "status": "passed" if not blockers else "blocked",
        "task_id": TASK_ID,
        "run_id": run_id or None,
        "parent_run_id": parent_run_id or None,
        "checks": checks,
        "blockers": blockers,
        "captures": capture_results,
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
