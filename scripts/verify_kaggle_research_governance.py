from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_EXPERIMENTS = [f"EXP{i:03d}" for i in range(18)] + ["EXP021"]
REQUIRED_DIRS = [
    "experiments",
    "reports",
    "submissions",
    "notebooks_or_scripts",
    "references",
]
REQUIRED_FILES = [
    "experiments/EXPERIMENT_LOG.md",
    "reports/CV_AND_ERROR_ANALYSIS.md",
    "reports/RESEARCH_WORKSTATION_FINAL_REPORT.md",
    "reports/RESEARCH_GOAL_COMPLETION_AUDIT_AND_NEXT_MATRIX.md",
    "references/REFERENCE_REVIEW.md",
    "reports/SUBMISSION_GATEWAY_DIAGNOSTIC_20260615.md",
    "reports/SUBMISSION_OPERATION_RUNBOOK.md",
    "reports/SUBMISSION_TEST_RESULT_EXP010_20260615.md",
    "reports/SUBMISSION_TEST_RESULT_EXP015_20260615.md",
    "reports/MANUAL_SUBMISSION_EXP015_RUNBOOK.md",
    "reports/EXPERIMENT_EXP014.md",
    "reports/EXPERIMENT_EXP015.md",
    "reports/EXPERIMENT_EXP016.md",
    "reports/EXPERIMENT_EXP017.md",
    "reports/EXPERIMENT_EXP021.md",
    "reports/HPC_EXEC_READINESS_EXP016_20260615.md",
    "reports/SUBMISSION_EXP015_CANDIDATE.md",
    "reports/SUBMISSION_EXP017_CANDIDATE.md",
    "reports/SUBMISSION_EXP021_CANDIDATE.md",
    "reports/REPRODUCIBLE_SUBMISSION_PACKAGE.md",
    "workspace/kaggle_submissions/20260615_093855_exp010_submit_attempt_timeout.json",
    "workspace/kaggle_submissions/20260615_103322_exp015_submit_attempt_timeout.json",
    "workspace/kaggle_submissions/20260615_110927_exp015_submit_attempt_timeout_600s.json",
    "workspace/kaggle_submissions/gateway_preflight_20260615_113225.json",
    "workspace/hpc_experiments/playground_series_s6e6/EXP016_lgbm_optuna_dryrun_20260615_1050/metrics.json",
    "workspace/hpc_experiments/playground_series_s6e6/EXP016_lgbm_optuna_hpc_readiness_20260615_1103.json",
    "workspace/hpc_experiments/playground_series_s6e6/EXP017_exp015_bias_calibration_20260615_1138/metrics.json",
    "workspace/hpc_experiments/playground_series_s6e6/EXP021_rank_decision_blend_20260615_1235/metrics.json",
    "workspace/reproducibility/latest_manifest.json",
]
SUBMISSION_SCHEMA_CHECKS = {
    "EXP010": "submissions/submission_EXP010_stacker_lower_error_10fold_not_submitted.csv",
    "EXP011": "submissions/submission_EXP011_risk_constrained_stacker_not_submitted.csv",
    "EXP012": "submissions/submission_EXP012_binary_chain_not_recommended.csv",
    "EXP015": "submissions/submission_EXP015_constrained_oof_blend_not_submitted.csv",
    "EXP017": "submissions/submission_EXP017_exp015_bias_calibration_not_submitted.csv",
    "EXP021": "submissions/submission_EXP021_rank_decision_blend_not_submitted.csv",
}
SAMPLE_SUBMISSION = "tasks/playground_series_s6e6/data/sample_submission.csv"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def check_exists(path: str, kind: str) -> dict[str, Any]:
    full = ROOT / path
    exists = full.is_dir() if kind == "dir" else full.is_file()
    return {
        "name": f"{kind}_exists:{path}",
        "passed": exists,
        "path": path,
    }


def parse_experiment_ids(log_text: str) -> list[str]:
    ids: list[str] = []
    for line in log_text.splitlines():
        match = re.match(r"\|\s*(EXP\d{3})\s*\|", line)
        if match:
            ids.append(match.group(1))
    return ids


def check_experiment_log() -> list[dict[str, Any]]:
    log_path = ROOT / "experiments/EXPERIMENT_LOG.md"
    text = log_path.read_text(encoding="utf-8")
    ids = parse_experiment_ids(text)
    checks: list[dict[str, Any]] = []
    missing = [exp for exp in EXPECTED_EXPERIMENTS if exp not in ids]
    duplicate_ids = sorted({exp for exp in ids if ids.count(exp) > 1})
    checks.append(
        {
            "name": "experiment_log_has_required_experiment_ids",
            "passed": not missing,
            "missing": missing,
            "found_count": len(ids),
        }
    )
    checks.append(
        {
            "name": "experiment_log_has_no_duplicate_experiment_ids",
            "passed": not duplicate_ids,
            "duplicates": duplicate_ids,
        }
    )
    required_headers = [
        "experiment_id",
        "date",
        "model",
        "features",
        "CV scheme",
        "CV score",
        "public score if submitted",
        "seed",
        "notes",
        "artifact path",
        "decision",
    ]
    header_line = next((line for line in text.splitlines() if line.startswith("| experiment_id |")), "")
    missing_headers = [header for header in required_headers if header not in header_line]
    checks.append(
        {
            "name": "experiment_log_has_required_headers",
            "passed": not missing_headers,
            "missing_headers": missing_headers,
        }
    )
    return checks


def read_csv_ids(path: Path) -> tuple[list[str], int, list[str]]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        for row in reader:
            ids.append(str(row.get("id", "")))
    return fieldnames, len(ids), ids


def check_submission_schema() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    sample_path = ROOT / SAMPLE_SUBMISSION
    sample_fields, sample_rows, sample_ids = read_csv_ids(sample_path)
    checks.append(
        {
            "name": "sample_submission_schema_available",
            "passed": sample_fields == ["id", "class"] and sample_rows == 247435,
            "sample_fields": sample_fields,
            "sample_rows": sample_rows,
        }
    )
    valid_labels = {"GALAXY", "QSO", "STAR"}
    for exp_id, sub_rel in SUBMISSION_SCHEMA_CHECKS.items():
        sub_path = ROOT / sub_rel
        if not sub_path.is_file():
            checks.append({"name": f"{exp_id}_submission_file_exists", "passed": False, "path": sub_rel})
            continue
        labels: set[str] = set()
        missing_labels = 0
        ids: list[str] = []
        with sub_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            for row in reader:
                ids.append(str(row.get("id", "")))
                label = str(row.get("class", ""))
                if not label:
                    missing_labels += 1
                labels.add(label)
        invalid_labels = sorted(labels - valid_labels)
        checks.append(
            {
                "name": f"{exp_id}_submission_schema_gate",
                "passed": (
                    fields == ["id", "class"]
                    and len(ids) == sample_rows
                    and ids == sample_ids
                    and missing_labels == 0
                    and not invalid_labels
                ),
                "path": sub_rel,
                "fields": fields,
                "rows": len(ids),
                "id_order_match": ids == sample_ids,
                "missing_labels": missing_labels,
                "invalid_labels": invalid_labels,
            }
        )
    return checks


def check_report_currency() -> list[dict[str, Any]]:
    final_path = ROOT / "reports/RESEARCH_WORKSTATION_FINAL_REPORT.md"
    text = final_path.read_text(encoding="utf-8")
    required_markers = [
        "through `EXP021`",
        "EXP011",
        "EXP012",
        "EXP013",
        "EXP014",
        "EXP015",
        "EXP016",
        "EXP017",
        "EXP021",
        "EXPERIMENT_EXP014.md",
        "EXPERIMENT_EXP015.md",
        "EXPERIMENT_EXP016.md",
        "EXPERIMENT_EXP017.md",
        "EXPERIMENT_EXP021.md",
        "HPC_EXEC_READINESS_EXP016_20260615.md",
        "SUBMISSION_EXP015_CANDIDATE.md",
        "SUBMISSION_EXP017_CANDIDATE.md",
        "SUBMISSION_EXP021_CANDIDATE.md",
        "SUBMISSION_OPERATION_RUNBOOK.md",
        "preflight_kaggle_submission_gateway.py",
        "record_kaggle_submission_score.py",
        "RESEARCH_GOAL_COMPLETION_AUDIT_AND_NEXT_MATRIX.md",
        "SUBMISSION_TEST_RESULT_EXP010_20260615.md",
        "SUBMISSION_TEST_RESULT_EXP015_20260615.md",
        "MANUAL_SUBMISSION_EXP015_RUNBOOK.md",
        "verify_kaggle_research_governance.py",
        "upload timed out before entering Kaggle's submissions queue",
    ]
    stale_markers = [
        "through `EXP009`",
        "through `EXP014`",
        "through `EXP015`",
        "through `EXP016`",
        "through `EXP017`",
        "Optuna tuning and binary-chain decomposition are still pending.",
    ]
    return [
        {
            "name": "final_report_contains_current_markers",
            "passed": all(marker in text for marker in required_markers),
            "missing_markers": [marker for marker in required_markers if marker not in text],
        },
        {
            "name": "final_report_has_no_known_stale_markers",
            "passed": not any(marker in text for marker in stale_markers),
            "stale_markers": [marker for marker in stale_markers if marker in text],
        },
    ]


def check_exp010_retry_audit() -> list[dict[str, Any]]:
    audit_path = ROOT / "workspace/kaggle_submissions/20260615_093855_exp010_submit_attempt_timeout.json"
    data = json.loads(audit_path.read_text(encoding="utf-8"))
    return [
        {
            "name": "exp010_retry_audit_records_timeout_not_score",
            "passed": (
                data.get("experiment_id") == "EXP010"
                and data.get("attempt_result") == "upload_timed_out_before_submission_entered_kaggle_list"
                and data.get("post_attempt_kaggle_list_state", {}).get("exp010_present") is False
            ),
            "attempt_result": data.get("attempt_result"),
            "exp010_present": data.get("post_attempt_kaggle_list_state", {}).get("exp010_present"),
        }
    ]


def check_exp015_retry_audit() -> list[dict[str, Any]]:
    audit_path = ROOT / "workspace/kaggle_submissions/20260615_103322_exp015_submit_attempt_timeout.json"
    data = json.loads(audit_path.read_text(encoding="utf-8"))
    latest_audit_path = ROOT / "workspace/kaggle_submissions/20260615_110927_exp015_submit_attempt_timeout_600s.json"
    latest_data = json.loads(latest_audit_path.read_text(encoding="utf-8"))
    return [
        {
            "name": "exp015_retry_audit_records_timeout_not_score",
            "passed": (
                data.get("experiment_id") == "EXP015"
                and data.get("attempt_result") == "upload_timed_out_before_submission_entered_kaggle_list"
                and data.get("post_attempt_kaggle_list_state", {}).get("exp015_present") is False
            ),
            "attempt_result": data.get("attempt_result"),
            "exp015_present": data.get("post_attempt_kaggle_list_state", {}).get("exp015_present"),
        },
        {
            "name": "exp015_latest_retry_audit_records_timeout_and_browser_block",
            "passed": (
                latest_data.get("experiment_id") == "EXP015"
                and latest_data.get("result") == "upload_timed_out_before_submission_entered_kaggle_list"
                and latest_data.get("timeout_seconds") == 600
                and latest_data.get("exp015_present_in_queue_after_attempt") is False
                and latest_data.get("browser_fallback", {}).get("result")
                == "blocked_by_chrome_enterprise_policy_for_kaggle"
            ),
            "attempt_result": latest_data.get("result"),
            "exp015_present": latest_data.get("exp015_present_in_queue_after_attempt"),
            "browser_fallback": latest_data.get("browser_fallback", {}).get("result"),
        }
    ]


def check_gateway_preflight() -> list[dict[str, Any]]:
    preflight_path = ROOT / "workspace/kaggle_submissions/gateway_preflight_20260615_113225.json"
    data = json.loads(preflight_path.read_text(encoding="utf-8"))
    checks_by_name = {check.get("name"): check for check in data.get("checks", [])}
    return [
        {
            "name": "gateway_preflight_records_blocked_upload_route",
            "passed": (
                data.get("status") == "blocked"
                and checks_by_name.get("kaggle_submissions_list", {}).get("passed") is True
                and checks_by_name.get("http_get:https://www.kaggle.com/", {}).get("passed") is True
                and checks_by_name.get("http_get:https://www.googleapis.com/generate_204", {}).get("passed") is False
                and checks_by_name.get("tcp_connect:www.googleapis.com:443", {}).get("passed") is False
            ),
            "status": data.get("status"),
            "kaggle_list": checks_by_name.get("kaggle_submissions_list", {}).get("passed"),
            "kaggle_web": checks_by_name.get("http_get:https://www.kaggle.com/", {}).get("passed"),
            "googleapis_http": checks_by_name.get("http_get:https://www.googleapis.com/generate_204", {}).get("passed"),
            "googleapis_tcp": checks_by_name.get("tcp_connect:www.googleapis.com:443", {}).get("passed"),
        }
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_reproducible_package() -> list[dict[str, Any]]:
    manifest_path = ROOT / "workspace/reproducibility/latest_manifest.json"
    report_path = ROOT / "reports/REPRODUCIBLE_SUBMISSION_PACKAGE.md"
    checks: list[dict[str, Any]] = []
    if not manifest_path.is_file():
        return [{"name": "reproducible_package_latest_manifest_exists", "passed": False}]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gates = manifest.get("verification", [])
    failed_gates = [gate for gate in gates if gate.get("status") != "passed"]
    zip_info = manifest.get("package_zip", {})
    zip_path = ROOT / str(zip_info.get("path", ""))
    zip_exists = zip_path.is_file()
    zip_sha_match = zip_exists and sha256_file(zip_path) == zip_info.get("sha256")
    report_text = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""

    checks.append(
        {
            "name": "reproducible_package_manifest_passed",
            "passed": manifest.get("status") == "passed" and not failed_gates,
            "manifest_status": manifest.get("status"),
            "failed_gates": failed_gates,
        }
    )
    checks.append(
        {
            "name": "reproducible_package_zip_hash_matches",
            "passed": zip_sha_match,
            "zip_path": zip_info.get("path"),
            "zip_exists": zip_exists,
            "expected_sha256": zip_info.get("sha256"),
        }
    )
    checks.append(
        {
            "name": "reproducible_package_report_is_current",
            "passed": (
                "Reproducible Submission Package" in report_text
                and str(zip_info.get("sha256", "")) in report_text
                and "EXP010" in report_text
                and "EXP011" in report_text
                and "EXP015" in report_text
                and "manual upload" in report_text.lower()
            ),
            "report_path": "reports/REPRODUCIBLE_SUBMISSION_PACKAGE.md",
        }
    )
    return checks


def main() -> int:
    checks: list[dict[str, Any]] = []
    checks.extend(check_exists(path, "dir") for path in REQUIRED_DIRS)
    checks.extend(check_exists(path, "file") for path in REQUIRED_FILES)
    checks.extend(check_experiment_log())
    checks.extend(check_submission_schema())
    checks.extend(check_report_currency())
    checks.extend(check_exp010_retry_audit())
    checks.extend(check_exp015_retry_audit())
    checks.extend(check_gateway_preflight())
    checks.extend(check_reproducible_package())

    failed = [check for check in checks if not check.get("passed")]
    report = {
        "status": "passed" if not failed else "failed",
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "workspace": str(ROOT),
        "check_count": len(checks),
        "failed_count": len(failed),
        "checks": checks,
    }

    out_dir = ROOT / "workspace/governance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"kaggle_research_governance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["artifact_path"] = rel(out_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
