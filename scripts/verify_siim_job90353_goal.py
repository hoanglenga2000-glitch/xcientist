#!/usr/bin/env python3
"""Rebuild the local, read-only completion audit for the sole SIIM job90353 Run.

The verifier deliberately separates completed operational requirements from the
unverified medal outcome.  It never connects to HPC, reads private labels,
invokes a grader, submits to Kaggle, or mutates the frozen Run/deliverables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.cookiejar
import json
import math
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUN_ID = "evomind_siim_isic_a800_job90353_20260730_095826"
R2_ID = "evomind_siim_isic_a800_job90353_medal_r2_20260802_023416"
CURRENT_GOAL_THREAD_ID = "019fc04a-cabf-7b00-9d89-1c11836f9e34"
SOURCE_TASK_ID = "019fa4da-c818-7642-8541-76e31d6b5886"
RESCISSION_SHA256 = "47c3815cc1179355df2a99bb59ffd0d2a56fe241ccccb6f9f849583613fc90c3"
EXPECTED_DELIVERABLES = {
    "evomind-siim-isic-report.pdf": "477761766f81f2b9709fd9a1cfd39d3b0308f346f9764ae3ba66d420fd35785d",
    "evomind-siim-isic-results.csv": "429faa05f2e9438958d6ba6d87f30b7a7071480c529b849d3391d9ea2b26a6cb",
    "evomind-siim-isic-code.zip": "bfc531b9de0e7a31c60f4c3db2d203da2144ebb2772823a58f7f130cd8fc592e",
    "evomind-siim-isic-evidence.zip": "96f89980c4ee407310273e25544ac288f9fb0e422b923c82a14b92743ba89a39",
}
EXPECTED_METRICS = {
    "roc_auc": 0.9225357247684676,
    "pr_auc": 0.23970933285011597,
    "brier": 0.29524735217259324,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def close(a: float, b: float, *, atol: float = 1e-12) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= atol


def rank_roc_auc(labels: list[int], scores: list[float]) -> float:
    """Compute tie-aware binary ROC-AUC via the Mann-Whitney statistic."""
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must be non-empty and equal length")
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("both classes are required")
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(label for _, label in pairs[index:end])
        index = end
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


class Audit:
    def __init__(self) -> None:
        self.checks: dict[str, dict[str, Any]] = {}

    def add(self, name: str, passed: bool, evidence: Any = None) -> None:
        item: dict[str, Any] = {"passed": bool(passed)}
        if evidence is not None:
            item["evidence"] = evidence
        self.checks[name] = item

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(item["passed"] for item in self.checks.values())


def _task_values(tasks: Any) -> list[dict[str, Any]]:
    if isinstance(tasks, dict):
        return [value for value in tasks.values() if isinstance(value, dict)]
    if isinstance(tasks, list):
        return [value for value in tasks if isinstance(value, dict)]
    return []


def _check_oof(audit: Audit, run_dir: Path, expected_auc: float) -> dict[str, Any]:
    path = run_dir / "oof_predictions.csv"
    rows = 0
    images: set[str] = set()
    patients: dict[str, str] = {}
    leakage_groups: dict[str, str] = {}
    patient_conflicts = 0
    leakage_conflicts = 0
    labels: list[int] = []
    scores: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            image = row["image_name"]
            patient = row["patient_id"]
            group = row["leakage_group"]
            fold = row["fold"]
            if image in images:
                continue
            images.add(image)
            if patient in patients and patients[patient] != fold:
                patient_conflicts += 1
            patients.setdefault(patient, fold)
            if group in leakage_groups and leakage_groups[group] != fold:
                leakage_conflicts += 1
            leakage_groups.setdefault(group, fold)
            labels.append(int(row["target"]))
            scores.append(float(row["probability"]))
    recomputed_auc = rank_roc_auc(labels, scores)
    evidence = {
        "rows": rows,
        "unique_images": len(images),
        "patients": len(patients),
        "leakage_groups": len(leakage_groups),
        "cross_fold_patient_conflicts": patient_conflicts,
        "cross_fold_leakage_group_conflicts": leakage_conflicts,
        "recomputed_roc_auc": recomputed_auc,
    }
    audit.add(
        "oof_complete_and_group_leakage_zero",
        rows == 28_984
        and len(images) == 28_984
        and len(patients) == 2_056
        and len(leakage_groups) == 2_056
        and patient_conflicts == 0
        and leakage_conflicts == 0
        and close(recomputed_auc, expected_auc),
        evidence,
    )
    return evidence


def _authenticated_ui_opener(base_url: str, bootstrap_url_file: Path | None) -> tuple[Any, str]:
    if bootstrap_url_file is None:
        return urllib.request.build_opener(), "none"
    target = bootstrap_url_file.resolve()
    if target.is_symlink() or not target.is_file() or not 32 <= target.stat().st_size <= 4096:
        raise ValueError("bootstrap URL file is missing or unsafe")
    raw_url = target.read_text(encoding="utf-8").strip()
    parsed = urllib.parse.urlsplit(raw_url)
    expected = urllib.parse.urlsplit(base_url.rstrip("/"))
    fragment = urllib.parse.parse_qs(parsed.fragment, strict_parsing=True)
    token = fragment.get("bootstrap", [""])[0]
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.hostname != expected.hostname
        or parsed.port != expected.port
        or parsed.path != "/"
        or set(fragment) != {"bootstrap"}
        or not 24 <= len(token) <= 256
    ):
        raise ValueError("bootstrap URL does not match the loopback UI contract")

    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/session/bootstrap",
        data=json.dumps({"token": token}).encode("utf-8"),
        headers={"Content-Type": "application/json", "Origin": base_url.rstrip("/")},
        method="POST",
    )
    token = ""
    try:
        with opener.open(request, timeout=4.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
            if response.status != 200 or payload.get("ok") is not True or not payload.get("csrf_token"):
                raise ValueError("bootstrap exchange did not establish a ready local session")
    finally:
        target.unlink(missing_ok=True)
    return opener, "one_time_fragment_exchange"


def _check_ui(run_id: str, base_url: str, *, bootstrap_url_file: Path | None = None) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/multi-agent/runs/{run_id}"
    try:
        opener, authentication = _authenticated_ui_opener(base_url, bootstrap_url_file)
        with opener.open(url, timeout=4.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
            run = payload.get("run", {}) if isinstance(payload, dict) else {}
            return {
                "passed": response.status == 200
                and payload.get("ok") is True
                and run.get("run_id") == run_id
                and run.get("status") == "completed",
                "url": url,
                "http_status": response.status,
                "authentication": authentication,
                "run_id": run.get("run_id"),
                "run_status": run.get("status"),
            }
    except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return {"passed": False, "url": url, "error": f"{type(exc).__name__}: {exc}"}


def evaluate(
    root: Path,
    *,
    require_ui: bool = False,
    ui_base_url: str = "http://127.0.0.1:8088",
    bootstrap_url_file: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    audit = Audit()
    run_dir = root / "workspace" / "evomind_runs" / RUN_ID
    delivery_dir = run_dir / "delivery"
    video_root = root / "video-production" / "siim-isic-melanoma-commercial-v2-user-journey"
    final_video = video_root / "final" / "evomind-siim-isic-medical-research-user-journey-zh-92s.mp4"

    run = read_json(run_dir / "run.json")
    tasks = _task_values(run.get("tasks"))
    audit.add(
        "single_target_run_completed_9_of_9",
        run.get("run_id") == RUN_ID
        and run.get("status") == "completed"
        and len(tasks) == 9
        and all(task.get("status") == "completed" for task in tasks)
        and run.get("open_requirements") == [],
        {"run_id": run.get("run_id"), "status": run.get("status"), "completed_tasks": sum(t.get("status") == "completed" for t in tasks), "total_tasks": len(tasks)},
    )

    r2_formal_dir = root / "workspace" / "evomind_runs" / R2_ID
    audit.add("no_second_formal_r2_run", not r2_formal_dir.exists(), {"path": str(r2_formal_dir), "exists": r2_formal_dir.exists()})

    budget = read_json(run_dir / "budget_supersession.json")
    effective = budget.get("effective_budget_hours", {})
    runtime_budget = budget.get("runtime_evidence", {})
    audit.add(
        "effective_78_hour_budget_and_runtime_within_budget",
        budget.get("status") == "effective"
        and effective == {"ablation": 4, "delivery": 2, "formal_training": 72, "total": 78}
        and runtime_budget.get("within_budget") is True
        and float(runtime_budget.get("formal_training_elapsed_seconds", math.inf)) <= float(runtime_budget.get("formal_training_budget_seconds", -1)),
        {"effective_budget_hours": effective, "runtime_evidence": runtime_budget},
    )

    hpc = read_json(run_dir / "hpc_runtime.json")
    progress = hpc.get("training_progress", {})
    telemetry = hpc.get("telemetry_summary", {})
    audit.add(
        "a800_three_seed_fifteen_fold_training_complete",
        hpc.get("training_status") == "completed"
        and hpc.get("formal_seeds") == [43, 44, 45]
        and progress.get("completed_formal_seeds") == progress.get("total_formal_seeds") == 3
        and progress.get("completed_outer_folds") == progress.get("total_outer_folds") == 15
        and "A800" in str(hpc.get("gpu", {}).get("name", ""))
        and hpc.get("gpu", {}).get("memory_total_mb") == 81_920
        and hpc.get("signals_sent") == 0
        and hpc.get("other_processes_modified") is False,
        {"gpu": hpc.get("gpu"), "formal_seeds": hpc.get("formal_seeds"), "training_progress": progress, "telemetry_summary": telemetry},
    )

    metrics = read_json(run_dir / "metrics.json")
    metric_values = {name: metrics.get(name) for name in EXPECTED_METRICS}
    audit.add(
        "final_metrics_match_frozen_candidate",
        all(close(float(metric_values[name]), expected) for name, expected in EXPECTED_METRICS.items())
        and metrics.get("kaggle_submission_executed") is False
        and metrics.get("mle_private_grader_score") is None,
        metric_values,
    )
    oof_evidence = _check_oof(audit, run_dir, EXPECTED_METRICS["roc_auc"])

    review = read_json(run_dir / "review.json")
    audit.add(
        "independent_review_passed",
        review.get("status") == "review_passed"
        and review.get("checks", {}).get("patient_group_overlap_zero") is True
        and review.get("checks", {}).get("content_group_overlap_zero") is True
        and close(float(review.get("recomputed_metrics", {}).get("roc_auc")), EXPECTED_METRICS["roc_auc"]),
        {"status": review.get("status"), "checks": review.get("checks"), "recomputed_metrics": review.get("recomputed_metrics")},
    )

    freeze = read_json(run_dir / "candidate_freeze.json")
    freeze_drift: list[dict[str, Any]] = []
    for item in freeze.get("artifacts", []):
        path = run_dir / item["path"]
        actual_hash = sha256(path) if path.is_file() else None
        actual_bytes = path.stat().st_size if path.is_file() else None
        if actual_hash != item.get("sha256") or actual_bytes != item.get("bytes"):
            freeze_drift.append({"path": item.get("path"), "expected_sha256": item.get("sha256"), "actual_sha256": actual_hash, "expected_bytes": item.get("bytes"), "actual_bytes": actual_bytes})
    audit.add(
        "candidate_frozen_before_grader_without_drift",
        freeze.get("status") == "frozen_before_private_grader"
        and freeze.get("private_grader_execution_count_before_freeze") == 0
        and freeze.get("tuning_closed") is True
        and len(freeze.get("artifacts", [])) == 10
        and not freeze_drift,
        {"status": freeze.get("status"), "artifact_count": len(freeze.get("artifacts", [])), "drift": freeze_drift},
    )

    grader = read_json(run_dir / "private_grader.json")
    ledger = read_json(run_dir / "private_grader_ledger.json")
    audit.add(
        "terminal_private_grader_recorded_exactly_once_failed_closed",
        ledger.get("execution_count") == 1
        and ledger.get("status") == "terminal_execution_recorded"
        and ledger.get("outcome") == "failed_closed"
        and ledger.get("score") is None
        and grader.get("status") == "failed_closed"
        and grader.get("score") is None
        and grader.get("mle_private_grader_score") is None
        and grader.get("official_mlebench_grader_executed") is False
        and grader.get("executed_after_freeze") is True
        and grader.get("feedback_used_for_tuning") is False,
        {"execution_count": ledger.get("execution_count"), "ledger_status": ledger.get("status"), "outcome": ledger.get("outcome"), "score": ledger.get("score"), "failure_reason": grader.get("failure_reason")},
    )

    claim = read_json(run_dir / "claim_audit.json")
    bound_files = {
        "candidate_freeze_sha256": run_dir / "candidate_freeze.json",
        "metrics_sha256": run_dir / "metrics.json",
        "private_grader_ledger_sha256": run_dir / "private_grader_ledger.json",
        "private_grader_sha256": run_dir / "private_grader.json",
        "review_sha256": run_dir / "review.json",
    }
    claim_drift = {
        key: {"expected": claim.get("bindings", {}).get(key), "actual": sha256(path)}
        for key, path in bound_files.items()
        if claim.get("bindings", {}).get(key) != sha256(path)
    }
    audit.add(
        "claim_audit_passed_and_hash_bound",
        claim.get("status") == "passed"
        and claim.get("checks", {}).get("terminal_private_grader_recorded_once") is True
        and claim.get("checks", {}).get("no_official_medal_claim") is True
        and claim.get("official_rank_or_medal") == "not_claimed"
        and not claim_drift,
        {"status": claim.get("status"), "checks": claim.get("checks"), "drift": claim_drift},
    )

    deliverables = read_json(run_dir / "deliverables.json")
    delivery_drift: list[dict[str, Any]] = []
    for name, expected_hash in EXPECTED_DELIVERABLES.items():
        path = delivery_dir / name
        actual = sha256(path) if path.is_file() else None
        if actual != expected_hash:
            delivery_drift.append({"name": name, "expected_sha256": expected_hash, "actual_sha256": actual})
    declared = {item.get("name"): item.get("sha256") for item in deliverables.get("files", [])}
    audit.add(
        "four_frozen_deliverables_present_and_hash_bound",
        deliverables.get("status") == "ready"
        and declared == EXPECTED_DELIVERABLES
        and not delivery_drift,
        {"declared": declared, "drift": delivery_drift},
    )

    r2_control = root / "workspace" / "siim_evolution_control" / R2_ID
    rescission_path = r2_control / "constraint_rescission.json"
    amendment_path = r2_control / "constraint_rescission_binding_amendment.json"
    amendment = read_json(amendment_path)
    r2_execution = amendment.get("r2_reservation_execution", {})
    audit.add(
        "r2_rescission_and_goal_binding_amendment_valid",
        sha256(rescission_path) == RESCISSION_SHA256
        and amendment.get("status") == "effective"
        and amendment.get("binding", {}).get("current_goal_thread_id") == CURRENT_GOAL_THREAD_ID
        and amendment.get("binding", {}).get("source_task_id") == SOURCE_TASK_ID
        and amendment.get("formal_run_scope", {}).get("unique_formal_run_id") == RUN_ID
        and amendment.get("formal_run_scope", {}).get("second_formal_run_created") is False
        and r2_execution.get("training_started") is False
        and r2_execution.get("remote_write_executed") is False
        and r2_execution.get("grader_reexecuted") is False
        and r2_execution.get("official_submission_executed") is False,
        {"rescission_sha256": sha256(rescission_path), "amendment_sha256": sha256(amendment_path), "binding": amendment.get("binding"), "r2_reservation_execution": r2_execution},
    )

    render_manifest = read_json(video_root / "final" / "render-manifest.json")
    qa = read_json(video_root / "final" / "qa" / "final-qa-report.json")
    visual = read_json(video_root / "final" / "qa" / "visual-review.json")
    filter_graph = video_root / "preproduction" / "render-filter-complex.txt"
    final_hash = sha256(final_video)
    qa_gates = qa.get("gates", {})
    visual_checks = visual.get("checks", {})
    visual_check_aliases = [
        ("real_chrome_operation",),
        ("ordinary_user_action_path",),
        ("same_run_visible",),
        ("input_send_system_response_and_open_result", "input_send_response_result_loop_visible"),
        ("all_requested_modules_visible", "r1_r4_multiround_evolution_readable"),
        ("professional_report_charts_readable",),
        ("independent_review_visible", "independent_review_and_claim_audit_readable"),
        ("claim_audit_visible", "independent_review_and_claim_audit_readable"),
        ("four_real_downloads_visible", "four_downloads_visible"),
    ]
    visual_checks_passed = all(any(visual_checks.get(name) is True for name in aliases) for aliases in visual_check_aliases)
    multiround_evidence = any(item.get("time") == "00:34-00:43" for item in visual.get("journey_evidence", []))
    freeze_interval_count = qa.get("freezedetect", {}).get("detections", {}).get("interval_count")
    audit.add(
        "real_chrome_92_second_video_and_full_qa_passed_without_synthetic_freeze_guard",
        render_manifest.get("run_id") == RUN_ID
        and render_manifest.get("status") in {"qa_passed", "release_qa_passed"}
        and render_manifest.get("sha256") == final_hash
        and render_manifest.get("duration_seconds") == 92.0
        and render_manifest.get("frames") == 2_760
        and render_manifest.get("width") == 1_920
        and render_manifest.get("height") == 1_080
        and render_manifest.get("fps") == "30/1"
        and render_manifest.get("real_chrome_source_ratio") == 1.0
        and render_manifest.get("freeze_guard") == "none"
        and "eq=brightness" not in filter_graph.read_text(encoding="utf-8")
        and qa.get("status") == "pass"
        and bool(qa_gates)
        and all(value is True for value in qa_gates.values())
        and freeze_interval_count == 0
        and visual.get("status") == "pass"
        and visual_checks_passed
        and multiround_evidence,
        {
            "path": str(final_video),
            "sha256": final_hash,
            "render_manifest_status": render_manifest.get("status"),
            "freeze_guard": render_manifest.get("freeze_guard"),
            "qa_status": qa.get("status"),
            "failed_qa_gates": sorted(name for name, value in qa_gates.items() if value is not True),
            "freeze_interval_count": freeze_interval_count,
            "visual_review": visual.get("status"),
            "visual_checks_passed": visual_checks_passed,
            "multiround_evidence_00_34_to_00_43": multiround_evidence,
        },
    )

    capture_results: list[dict[str, Any]] = []
    for stem in [
        "C10-clean-assistant-input-and-attach",
        "C11-clean-system-modules",
        "C12-clean-professional-report",
        "C02-clean-four-downloads",
    ]:
        source = video_root / "sources" / "raw" / f"{stem}.mkv"
        capture = video_root / "sources" / "raw" / f"{stem}.capture.json"
        manifest = read_json(capture)
        capture_results.append(
            {
                "source": source.name,
                "sha256": sha256(source),
                "manifest_sha256": manifest.get("sha256"),
                "run_id": manifest.get("run_id"),
                "status": manifest.get("status"),
                "research_run_mutated": manifest.get("research_run_mutated"),
                "second_run_created": manifest.get("second_run_created"),
                "training_started": manifest.get("training_started"),
                "grader_executed": manifest.get("grader_executed"),
                "official_submission": manifest.get("official_submission"),
            }
        )
    audit.add(
        "four_real_chrome_sources_bound_to_same_completed_run",
        all(
            item["sha256"] == item["manifest_sha256"]
            and item["run_id"] == RUN_ID
            and item["status"] == "captured"
            and item["research_run_mutated"] is False
            and item["second_run_created"] is False
            and item["training_started"] is False
            and item["grader_executed"] is False
            and item["official_submission"] is False
            for item in capture_results
        ),
        capture_results,
    )

    if require_ui:
        ui = _check_ui(RUN_ID, ui_base_url, bootstrap_url_file=bootstrap_url_file)
        audit.add("evomind_ui_same_run_api_live", ui.pop("passed"), ui)

    bronze_reference = float(metrics.get("historical_thresholds", {}).get("bronze", 0.937))
    roc_auc = float(metrics["roc_auc"])
    medal = {
        "status": "not_established",
        "official_submission_executed": False,
        "terminal_private_score": None,
        "official_rank_or_medal_claimed": False,
        "offline_oof_roc_auc": roc_auc,
        "historical_bronze_reference": bronze_reference,
        "offline_gap_to_historical_bronze_reference": bronze_reference - roc_auc,
        "reason": "No official submission or recoverable private score exists; the frozen offline OOF score is below the reference bronze threshold.",
    }
    return {
        "schema": "evomind.siim.job90353_goal_verification.v1",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "current_goal_thread_id": CURRENT_GOAL_THREAD_ID,
        "source_task_id": SOURCE_TASK_ID,
        "run_id": RUN_ID,
        "verification_mode": "local_read_only_no_hpc_no_grader_no_submission",
        "achievable_requirements_status": "passed" if audit.passed else "failed",
        "checks_passed": sum(item["passed"] for item in audit.checks.values()),
        "checks_total": len(audit.checks),
        "checks": audit.checks,
        "oof_recomputation": oof_evidence,
        "medal_outcome": medal,
        "goal_complete": False,
        "goal_status_recommendation": "blocked" if audit.passed else "active",
        "overall_status": "all_achievable_requirements_passed_medal_not_established" if audit.passed else "achievable_requirement_verification_failed",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="Optional JSON output path. Parent directories are created.")
    parser.add_argument("--require-ui", action="store_true", help="Require the local EvoMind same-Run API on port 8088.")
    parser.add_argument("--ui-base-url", default="http://127.0.0.1:8088")
    parser.add_argument(
        "--bootstrap-url-file",
        type=Path,
        help="One-time fragment URL from the lifecycle manager; exchanged without printing its token.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = evaluate(
        args.root,
        require_ui=args.require_ui,
        ui_base_url=args.ui_base_url,
        bootstrap_url_file=args.bootstrap_url_file,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, output)
    sys.stdout.write(payload)
    return 0 if report["achievable_requirements_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
