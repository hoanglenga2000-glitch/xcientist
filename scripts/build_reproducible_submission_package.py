from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "workspace" / "reproducibility"
LATEST_MANIFEST = PACKAGE_ROOT / "latest_manifest.json"
STABLE_REPORT = ROOT / "reports" / "REPRODUCIBLE_SUBMISSION_PACKAGE.md"

SAMPLE_SUBMISSION = ROOT / "tasks" / "playground_series_s6e6" / "data" / "sample_submission.csv"
DATA_FILES = [
    ROOT / "tasks" / "playground_series_s6e6" / "data" / "train.csv",
    ROOT / "tasks" / "playground_series_s6e6" / "data" / "test.csv",
    SAMPLE_SUBMISSION,
]

PORTABLE_FILES = [
    "experiments/EXPERIMENT_LOG.md",
    "reports/BASELINE_AUDIT_EXP000.md",
    "reports/CV_AND_ERROR_ANALYSIS.md",
    "reports/RESEARCH_WORKSTATION_FINAL_REPORT.md",
    "reports/RESEARCH_GOAL_COMPLETION_AUDIT_AND_NEXT_MATRIX.md",
    "reports/SUBMISSION_EXP007.md",
    "reports/SUBMISSION_EXP010_CANDIDATE.md",
    "reports/SUBMISSION_EXP011_CANDIDATE.md",
    "reports/SUBMISSION_TEST_RESULT_EXP010_20260615.md",
    "reports/SUBMISSION_TEST_RESULT_EXP015_20260615.md",
    "reports/MANUAL_SUBMISSION_EXP015_RUNBOOK.md",
    "reports/SUBMISSION_GATEWAY_DIAGNOSTIC_20260615.md",
    "reports/SUBMISSION_OPERATION_RUNBOOK.md",
    "reports/EXPERIMENT_EXP012.md",
    "reports/EXPERIMENT_EXP013.md",
    "reports/EXPERIMENT_EXP014.md",
    "reports/EXPERIMENT_EXP015.md",
    "reports/EXPERIMENT_EXP016.md",
    "reports/EXPERIMENT_EXP017.md",
    "reports/EXPERIMENT_EXP021.md",
    "reports/HPC_EXEC_READINESS_EXP016_20260615.md",
    "reports/SUBMISSION_EXP015_CANDIDATE.md",
    "reports/SUBMISSION_EXP017_CANDIDATE.md",
    "reports/SUBMISSION_EXP021_CANDIDATE.md",
    "references/REFERENCE_REVIEW.md",
    "submissions/submission_EXP007_blend_lgb052_xgb043_cat005_not_submitted.csv",
    "submissions/submission_EXP010_stacker_lower_error_10fold_not_submitted.csv",
    "submissions/submission_EXP011_risk_constrained_stacker_not_submitted.csv",
    "submissions/submission_EXP015_constrained_oof_blend_not_submitted.csv",
    "submissions/submission_EXP015_constrained_oof_blend_not_submitted.zip",
    "submissions/submission_EXP017_exp015_bias_calibration_not_submitted.csv",
    "submissions/submission_EXP021_rank_decision_blend_not_submitted.csv",
    "notebooks_or_scripts/exp007_three_model_blend.py",
    "notebooks_or_scripts/exp010_stacker_confirmation.py",
    "notebooks_or_scripts/exp011_risk_constrained_stacker.py",
    "notebooks_or_scripts/exp012_binary_chain_stacker.py",
    "notebooks_or_scripts/exp013_calibration_diagnostics.py",
    "notebooks_or_scripts/exp014_cv_stability_diagnostics.py",
    "notebooks_or_scripts/exp015_constrained_oof_blend.py",
    "notebooks_or_scripts/exp016_lgbm_optuna_search.py",
    "notebooks_or_scripts/exp017_exp015_bias_calibration.py",
    "notebooks_or_scripts/exp021_rank_decision_blending.py",
    "scripts/run_hpc_exp016_lgbm_optuna_search.py",
    "scripts/preflight_kaggle_submission_gateway.py",
    "scripts/record_kaggle_submission_score.py",
    "scripts/verify_kaggle_research_governance.py",
    "scripts/verify_no_plaintext_secrets.py",
    "requirements.txt",
]

ARTIFACT_REFERENCES = [
    "workspace/hpc_experiments/playground_series_s6e6/EXP007_three_blend_refined_20260614_2348/metrics.json",
    "workspace/hpc_experiments/playground_series_s6e6/EXP007_three_blend_refined_20260614_2348/oof_and_test_probabilities.npz",
    "workspace/hpc_experiments/playground_series_s6e6/EXP010_stacker_confirmation_20260615_004108/metrics.json",
    "workspace/hpc_experiments/playground_series_s6e6/EXP010_stacker_confirmation_20260615_004108/lower_error/oof_and_test_probabilities.npz",
    "workspace/hpc_experiments/playground_series_s6e6/EXP011_risk_constrained_stacker_20260615_0917/metrics.json",
    "workspace/hpc_experiments/playground_series_s6e6/EXP011_risk_constrained_stacker_20260615_0917/oof_and_test_probabilities.npz",
    "workspace/hpc_experiments/playground_series_s6e6/EXP012_binary_chain_stacker_20260615_0930/metrics.json",
    "workspace/hpc_experiments/playground_series_s6e6/EXP013_calibration_diagnostics_20260615_0950/metrics.json",
    "workspace/hpc_experiments/playground_series_s6e6/EXP014_cv_stability_diagnostics_20260615_1000/metrics.json",
    "workspace/hpc_experiments/playground_series_s6e6/EXP015_constrained_oof_blend_20260615_1020/metrics.json",
    "workspace/hpc_experiments/playground_series_s6e6/EXP015_constrained_oof_blend_20260615_1020/oof_and_test_probabilities.npz",
    "workspace/hpc_experiments/playground_series_s6e6/EXP016_lgbm_optuna_dryrun_20260615_1050/metrics.json",
    "workspace/hpc_experiments/playground_series_s6e6/EXP016_lgbm_optuna_dryrun_20260615_1050/optuna_trials_dataframe.csv",
    "workspace/hpc_experiments/playground_series_s6e6/EXP016_lgbm_optuna_dryrun_20260615_1050/trial_results_compact.csv",
    "workspace/hpc_experiments/playground_series_s6e6/EXP016_lgbm_optuna_hpc_readiness_20260615_1103.json",
    "workspace/hpc_experiments/playground_series_s6e6/EXP017_exp015_bias_calibration_20260615_1138/metrics.json",
    "workspace/hpc_experiments/playground_series_s6e6/EXP017_exp015_bias_calibration_20260615_1138/oof_and_test_probabilities.npz",
    "workspace/hpc_experiments/playground_series_s6e6/EXP021_rank_decision_blend_20260615_1235/metrics.json",
    "workspace/hpc_experiments/playground_series_s6e6/EXP021_rank_decision_blend_20260615_1235/oof_and_test_probabilities.npz",
    "workspace/kaggle_submissions/20260615_093855_exp010_submit_attempt_timeout.json",
    "workspace/kaggle_submissions/20260615_103322_exp015_submit_attempt_timeout.json",
    "workspace/kaggle_submissions/20260615_110927_exp015_submit_attempt_timeout_600s.json",
    "workspace/kaggle_submissions/gateway_preflight_20260615_113225.json",
]

SECRET_REGEXES = {
    "kaggle_access_token": re.compile(r"KGAT_[A-Za-z0-9_-]{16,}"),
    "generic_api_key": re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{20,}\b"),
    "hpc_or_proxy_password_literal": re.compile(r"aimslab@[A-Za-z0-9!@#$%^&*()_+\-=]{4,}"),
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, copy_role: str) -> dict[str, Any]:
    return {
        "path": rel(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "copy_role": copy_role,
    }


def read_csv_header_and_ids(path: Path) -> tuple[list[str], list[str], list[str]]:
    ids: list[str] = []
    labels: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        for row in reader:
            ids.append(str(row.get("id", "")))
            labels.append(str(row.get("class", "")))
    return fields, ids, labels


def validate_submission(path: Path, sample_ids: list[str]) -> dict[str, Any]:
    fields, ids, labels = read_csv_header_and_ids(path)
    invalid = sorted(set(labels) - {"GALAXY", "QSO", "STAR"})
    return {
        "path": rel(path),
        "fields": fields,
        "rows": len(ids),
        "id_order_match": ids == sample_ids,
        "missing_labels": sum(1 for label in labels if not label),
        "invalid_labels": invalid,
        "passed": fields == ["id", "class"] and ids == sample_ids and not invalid and all(labels),
        "prediction_distribution": {label: labels.count(label) for label in sorted(set(labels))},
    }


def run_command(command: list[str], cwd: Path) -> dict[str, Any]:
    import subprocess

    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=180)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "passed": completed.returncode == 0,
    }


def copy_portable_files(package_dir: Path, files: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for src in files:
        dst = package_dir / "files" / rel(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        records.append(file_record(src, copy_role="copied_into_package"))
    return records


def make_zip(package_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(package_dir.rglob("*")):
            if path == zip_path or path.is_dir():
                continue
            zf.write(path, path.relative_to(package_dir).as_posix())


def write_checksums(package_dir: Path, records: list[dict[str, Any]]) -> Path:
    checksum_path = package_dir / "CHECKSUMS.sha256"
    lines = [f"{record['sha256']}  {record['path']}" for record in sorted(records, key=lambda item: item["path"])]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksum_path


def write_reproduce_script(package_dir: Path) -> Path:
    script_path = package_dir / "REPRODUCE.ps1"
    content = """# Reproduce / verify the current Kaggle research package
$ErrorActionPreference = "Stop"
Write-Output "1. Verify governance gates"
python scripts\\verify_kaggle_research_governance.py
Write-Output "2. Verify plaintext secret scan"
python scripts\\verify_no_plaintext_secrets.py
Write-Output "3. Rebuild this reproducibility package"
python scripts\\build_reproducible_submission_package.py
Write-Output "Done. Official Kaggle submission still requires explicit user approval."
"""
    script_path.write_text(content, encoding="utf-8")
    return script_path


def text_has_secret_markers(path: Path) -> list[str]:
    if path.suffix.lower() not in {".md", ".txt", ".json", ".csv", ".py", ".ps1", ".yml", ".yaml"}:
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    return [name for name, pattern in SECRET_REGEXES.items() if pattern.search(text)]


def build_markdown(manifest: dict[str, Any]) -> str:
    official = manifest["models"]["current_official_best"]
    high = manifest["models"]["high_upside_candidate"]
    conservative = manifest["models"]["conservative_candidate"]
    middle = manifest["models"]["middle_ground_candidate"]
    metric_candidate = manifest["models"]["metric_prioritized_candidate"]
    negative_ablation = manifest["models"]["latest_negative_ablation"]
    zip_record = manifest["package_zip"]
    lines = [
        "# Reproducible Submission Package",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Scope",
        "",
        "This package captures the current Kaggle/HPC research state for `playground-series-s6e6`. It is for audit and reproduction. It does not authorize or perform an official Kaggle submission.",
        "",
        "## Current Model State",
        "",
        f"- Official best: `{official['experiment_id']}` public `{official['public_score']}`, ref `{official['submission_ref']}`.",
        f"- High-upside candidate: `{high['experiment_id']}`, local BA `{high['balanced_accuracy']}`, risk: `{high['risk']}`.",
        f"- Conservative candidate: `{conservative['experiment_id']}`, local BA `{conservative['balanced_accuracy']}`, risk: `{conservative['risk']}`.",
        f"- Middle-ground candidate: `{middle['experiment_id']}`, local BA `{middle['balanced_accuracy']}`, risk: `{middle['risk']}`.",
        f"- Metric-prioritized candidate: `{metric_candidate['experiment_id']}`, nested calibration BA `{metric_candidate['balanced_accuracy']}`, risk: `{metric_candidate['risk']}`.",
        f"- Latest negative ablation: `{negative_ablation['experiment_id']}`, result: `{negative_ablation['result']}`.",
        "",
        "## Package Outputs",
        "",
        f"- Package directory: `{manifest['package_dir']}`",
        f"- Zip archive: `{zip_record['path']}`",
        f"- Zip sha256: `{zip_record['sha256']}`",
        f"- Manifest: `{manifest['manifest_path']}`",
        f"- Checksums: `{manifest['checksums_path']}`",
        "",
        "## Gates",
        "",
    ]
    for gate in manifest["verification"]:
        lines.append(f"- `{gate['name']}`: `{gate['status']}`")
    lines.extend(
        [
            "",
            "## Reproduction Commands",
            "",
            "```powershell",
            "python scripts\\verify_kaggle_research_governance.py",
            "python scripts\\preflight_kaggle_submission_gateway.py --competition playground-series-s6e6",
            "python scripts\\verify_no_plaintext_secrets.py",
            "python scripts\\build_reproducible_submission_package.py",
            "```",
            "",
            "## Submission Policy",
            "",
            "EXP010, EXP011, EXP015, and EXP017 remain candidates only. Official Kaggle submission requires explicit user approval and a repaired upload route.",
            "",
            "EXP016 is a tuning scaffold dry-run, not a submission candidate.",
            "",
            "EXP017 is a higher-risk metric-prioritized calibration candidate over EXP015 probabilities.",
            "",
            "EXP021 is not a submission candidate; it is negative ablation evidence that current rank/decision/probability blending does not improve the risk-adjusted frontier.",
            "",
            "## Manual Upload Package",
            "",
            "The current Windows route can list Kaggle submissions but cannot complete object-storage uploads. A browser/manual upload package is included for EXP015:",
            "",
            "- `submissions/submission_EXP015_constrained_oof_blend_not_submitted.zip`",
            "- `reports/MANUAL_SUBMISSION_EXP015_RUNBOOK.md`",
            "- `reports/SUBMISSION_OPERATION_RUNBOOK.md`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a reproducible package for the current Kaggle research state.")
    parser.add_argument("--package-id", default=datetime.now().strftime("repro_package_%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    PACKAGE_ROOT.mkdir(parents=True, exist_ok=True)
    package_dir = PACKAGE_ROOT / args.package_id
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)

    missing = [path for path in [*(ROOT / item for item in PORTABLE_FILES), *DATA_FILES, *(ROOT / item for item in ARTIFACT_REFERENCES)] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required package inputs: " + ", ".join(str(path) for path in missing))

    sample_fields, sample_ids, _ = read_csv_header_and_ids(SAMPLE_SUBMISSION)
    portable_paths = [ROOT / item for item in PORTABLE_FILES]
    copied_records = copy_portable_files(package_dir, portable_paths)
    data_records = [file_record(path, copy_role="referenced_by_hash_only") for path in DATA_FILES]
    artifact_records = [file_record(ROOT / item, copy_role="referenced_by_hash_only") for item in ARTIFACT_REFERENCES]
    all_records = copied_records + data_records + artifact_records

    reproduce_script = write_reproduce_script(package_dir)
    all_records.append(
        {
            "path": reproduce_script.relative_to(package_dir).as_posix(),
            "bytes": reproduce_script.stat().st_size,
            "sha256": sha256_file(reproduce_script),
            "copy_role": "package_instruction",
        }
    )

    submission_checks = [
        validate_submission(ROOT / "submissions/submission_EXP007_blend_lgb052_xgb043_cat005_not_submitted.csv", sample_ids),
        validate_submission(ROOT / "submissions/submission_EXP010_stacker_lower_error_10fold_not_submitted.csv", sample_ids),
        validate_submission(ROOT / "submissions/submission_EXP011_risk_constrained_stacker_not_submitted.csv", sample_ids),
        validate_submission(ROOT / "submissions/submission_EXP015_constrained_oof_blend_not_submitted.csv", sample_ids),
        validate_submission(ROOT / "submissions/submission_EXP017_exp015_bias_calibration_not_submitted.csv", sample_ids),
        validate_submission(ROOT / "submissions/submission_EXP021_rank_decision_blend_not_submitted.csv", sample_ids),
    ]

    governance_checker = run_command([sys.executable, "-m", "py_compile", "scripts/verify_kaggle_research_governance.py"], ROOT)
    secret_scan = run_command([sys.executable, "scripts/verify_no_plaintext_secrets.py"], ROOT)
    secret_marker_hits = []
    for copied in (package_dir / "files").rglob("*"):
        if copied.is_file():
            hits = text_has_secret_markers(copied)
            if hits:
                secret_marker_hits.append({"path": copied.relative_to(package_dir).as_posix(), "markers": hits})

    checksums_path = write_checksums(package_dir, all_records)
    manifest = {
        "status": "passed",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "competition": "playground-series-s6e6",
        "package_dir": package_dir.relative_to(ROOT).as_posix(),
        "manifest_path": (package_dir / "MANIFEST.json").relative_to(ROOT).as_posix(),
        "checksums_path": checksums_path.relative_to(ROOT).as_posix(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "models": {
            "current_official_best": {
                "experiment_id": "EXP007",
                "public_score": 0.96659,
                "submission_ref": "53680150",
                "submission_file": "submissions/submission_EXP007_blend_lgb052_xgb043_cat005_not_submitted.csv",
            },
            "high_upside_candidate": {
                "experiment_id": "EXP010",
                "balanced_accuracy": 0.9667720565732942,
                "delta_vs_exp007": 0.0010291749866148,
                "submission_file": "submissions/submission_EXP010_stacker_lower_error_10fold_not_submitted.csv",
                "risk": "higher logloss and more OOF errors than EXP007/EXP011",
            },
            "conservative_candidate": {
                "experiment_id": "EXP011",
                "balanced_accuracy": 0.9661628813827866,
                "delta_vs_exp007": 0.0004199997961071844,
                "submission_file": "submissions/submission_EXP011_risk_constrained_stacker_not_submitted.csv",
                "risk": "lower upside than EXP010 but closer probability quality and fewer OOF errors",
            },
            "middle_ground_candidate": {
                "experiment_id": "EXP015",
                "balanced_accuracy": 0.9663672312518644,
                "delta_vs_exp007": 0.0006243496651849867,
                "submission_file": "submissions/submission_EXP015_constrained_oof_blend_not_submitted.csv",
                "risk": "more BA than EXP011 with modestly higher logloss and OOF errors; much less aggressive than EXP010",
            },
            "metric_prioritized_candidate": {
                "experiment_id": "EXP017",
                "balanced_accuracy": 0.9664821547636583,
                "delta_vs_exp015": 0.00011492351179387406,
                "submission_file": "submissions/submission_EXP017_exp015_bias_calibration_not_submitted.csv",
                "risk": "small BA lift over EXP015 but worse logloss and more OOF errors",
            },
            "latest_negative_ablation": {
                "experiment_id": "EXP021",
                "balanced_accuracy": 0.9667720565732942,
                "result": "best row reverts to EXP010; zero rows satisfy the EXP017 logloss/error risk guard",
                "submission_file": "submissions/submission_EXP021_rank_decision_blend_not_submitted.csv",
                "risk": "not a submission candidate",
            },
        },
        "data_files": data_records,
        "artifact_references": artifact_records,
        "copied_files": copied_records,
        "submission_schema_checks": submission_checks,
        "verification": [
            {
                "name": "governance_checker_compile",
                "status": "passed" if governance_checker["passed"] else "failed",
                "details": governance_checker,
            },
            {"name": "plaintext_secret_scan", "status": "passed" if secret_scan["passed"] else "failed", "details": secret_scan},
            {"name": "package_secret_marker_scan", "status": "passed" if not secret_marker_hits else "failed", "hits": secret_marker_hits},
            {
                "name": "candidate_submission_schema",
                "status": "passed" if all(item["passed"] for item in submission_checks) else "failed",
                "details": submission_checks,
            },
        ],
        "submission_policy": "No official Kaggle submission is performed. Submission requires explicit user approval and a working upload route.",
    }
    if any(gate["status"] != "passed" for gate in manifest["verification"]):
        manifest["status"] = "failed"

    manifest_path = package_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    all_records.append(
        {
            "path": manifest_path.relative_to(package_dir).as_posix(),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
            "copy_role": "package_manifest",
        }
    )
    write_checksums(package_dir, all_records)

    zip_path = PACKAGE_ROOT / f"{args.package_id}.zip"
    make_zip(package_dir, zip_path)
    zip_record = {
        "path": zip_path.relative_to(ROOT).as_posix(),
        "bytes": zip_path.stat().st_size,
        "sha256": sha256_file(zip_path),
    }
    manifest["package_zip"] = zip_record
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    LATEST_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    STABLE_REPORT.write_text(build_markdown(manifest), encoding="utf-8")

    print(json.dumps({"status": manifest["status"], "package_dir": manifest["package_dir"], "zip": zip_record, "stable_report": rel(STABLE_REPORT)}, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
