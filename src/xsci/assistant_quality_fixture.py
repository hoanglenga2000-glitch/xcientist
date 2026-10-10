"""Deterministic SIIM evidence workspace for the novice assistant quality gate.

Why this exists
---------------
The live novice gate used to read the operator's own ``workspace/`` and was
hard-wired to one historical Run
(``evomind_siim_isic_a800_job90353_20260730_095826``).  That Run directory was
lost from the operator machine and the owner decided not to recover it, so
the gate could no longer pass anywhere, and in clean checkouts it never could.

The gate now runs each live case against a hermetic workstation root that is
materialised from this module.  The fixture keeps the behaviours under test
intact: the assistant must answer from ``verified_context`` evidence (metrics,
grouped-validation review checks, reviewed literature, hash-verified
deliverables, a failed-closed private grader that ran exactly once), not by
searching the file system.

Provenance of the values
------------------------
* Metrics, confidence interval and review checks are the values recorded for
  the historical Run in ``scripts/verify_siim_job90353_goal.py`` and
  ``tests/test_assistant_context.py``.  They are reproduced, not re-measured.
* The literature entries are real, Crossref-verified papers.
* The four deliverables are small deterministic stand-ins generated here.
  They are NOT the historical report/results/code/evidence files; their sizes
  and hashes are specific to this fixture and are pinned in the suite.

Everything is generated from constants with LF line endings and fixed zip
metadata, so the bytes (and therefore the hashes) are identical on every OS.
See docs/provenance/assistant_quality_fixture.md.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

FIXTURE_ID = "siim_quality_fixture_v1"
TASK_ID = "siim-isic-melanoma-classification"
RUN_ID = "evomind_siim_isic_quality_fixture_v1"
SOURCE_RUN_ID = "evomind_siim_isic_a800_job90353_20260730_095826"
RUN_RELATIVE = f"workspace/evomind_runs/{RUN_ID}"
DOWNLOAD_PREFIX = f"/api/multi-agent/runs/{RUN_ID}/download"
FIXED_ZIP_TIME = (2026, 7, 30, 9, 58, 26)
FIXTURE_UPDATED_AT = "2026-07-31T03:07:07+00:00"

METRICS: dict[str, Any] = {
    "roc_auc": 0.9225357247684676,
    "pr_auc": 0.23970933285011597,
    "brier": 0.29524735217259324,
    "metric_scope": "independent_offline_patient_content_grouped_oof",
    "fold_roc_auc_mean": 0.922480488386402,
    "fold_roc_auc_std": 0.008603472136512058,
    "patient_grouped_bootstrap_roc_auc_95ci": {
        "lower": 0.9110109632922689,
        "upper": 0.9327846750376818,
        "valid_samples": 2000,
        "method": "patient_cluster_bootstrap",
    },
    "official_external_score": None,
    "local_gpu_used": False,
}

REVIEW: dict[str, Any] = {
    "status": "review_passed",
    "review_scope": "public_train_grouped_oof_and_withheld_test_predictions",
    "checks": {
        "patient_group_overlap_zero": True,
        "content_group_overlap_zero": True,
        "oof_coverage_exactly_once": True,
        "private_labels_unavailable_during_training": True,
        "submission_schema_and_order": True,
        "official_submission_not_executed": True,
    },
    "next_action": "calibration_and_false_positive_analysis_under_same_groups",
    "unresolved": [],
}

CLAIM_AUDIT: dict[str, Any] = {
    "status": "passed",
    "approved_claim": (
        "Independent offline patient/content-grouped OOF ROC-AUC 0.9225; "
        "not an official Kaggle score."
    ),
}

PRIVATE_GRADER: dict[str, Any] = {
    "status": "failed_closed",
    "score": None,
    "feedback_used_for_tuning": False,
    "official_submission_executed": False,
}

PRIVATE_GRADER_LEDGER: dict[str, Any] = {
    "execution_count": 1,
    "outcome": "failed_closed",
    "score": None,
    "feedback_used_for_tuning": False,
    "official_submission_executed": False,
}

CANDIDATE_FREEZE: dict[str, Any] = {
    "status": "frozen",
    "run_id": RUN_ID,
    "frozen_before_private_grader": True,
}

PAPERS: list[dict[str, Any]] = [
    {
        "title": (
            "Effect of patient-contextual skin images in human- and artificial "
            "intelligence-based diagnosis of melanoma: Results from the 2020 "
            "SIIM-ISIC melanoma classification challenge"
        ),
        "year": "2024",
        "source": "crossref",
        "doi": "10.1111/jdv.20479",
        "url": "https://doi.org/10.1111/jdv.20479",
        "methods": ["patient-grouped dermoscopy dataset", "AUROC private leaderboard"],
    },
    {
        "title": "Analysis of the ISIC image datasets: Usage, benchmarks and recommendations",
        "year": "2022",
        "source": "crossref",
        "doi": "10.1016/j.media.2021.102305",
        "url": "https://doi.org/10.1016/j.media.2021.102305",
        "methods": ["duplicate image analysis", "dataset recommendations"],
    },
]

CITATION_AUDIT: dict[str, Any] = {
    "task_id": TASK_ID,
    "status": "passed",
    "gate": "citation_gate_passed",
    "claim": (
        "The SIIM-ISIC 2020 challenge provided dermoscopy images grouped by patient "
        "and evaluated patient-contextual images for human and AI melanoma diagnosis."
    ),
    "paper_id": "crossref-10.1111_jdv.20479",
    "conclusion": "The claim is eligible for report use with this citation and its current wording.",
}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _zip_bytes(entries: list[tuple[str, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in sorted(entries):
            info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIME)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, payload)
    return buffer.getvalue()


def _pdf_bytes(lines: list[str]) -> bytes:
    text_ops = ["BT", "/F1 11 Tf", "72 760 Td", "14 TL"]
    for line in lines:
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        text_ops.append(f"({escaped}) Tj T*")
    text_ops.append("ET")
    stream = "\n".join(text_ops).encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("ascii")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode("ascii")
    return bytes(out)


def deliverable_payloads() -> dict[str, bytes]:
    """Return the four deterministic stand-in deliverables, by file name."""

    report = _pdf_bytes([
        "EvoMind SIIM-ISIC melanoma - quality-gate evidence fixture",
        f"Fixture run: {RUN_ID} (reproduces recorded evidence of {SOURCE_RUN_ID})",
        "Patient/content grouped OOF ROC-AUC 0.9225, PR-AUC 0.2397, Brier 0.2952",
        "Patient cluster bootstrap 95% CI [0.9110, 0.9328], 2000 samples",
        "Private grader: failed_closed, score null, executed once, not used for tuning",
        "No official Kaggle submission was executed.",
    ])
    results = (
        "metric,value,scope\n"
        f"roc_auc,{METRICS['roc_auc']},{METRICS['metric_scope']}\n"
        f"pr_auc,{METRICS['pr_auc']},{METRICS['metric_scope']}\n"
        f"brier,{METRICS['brier']},{METRICS['metric_scope']}\n"
        f"fold_roc_auc_mean,{METRICS['fold_roc_auc_mean']},{METRICS['metric_scope']}\n"
        f"fold_roc_auc_std,{METRICS['fold_roc_auc_std']},{METRICS['metric_scope']}\n"
    ).encode("utf-8")
    code = _zip_bytes([
        ("README.md", (
            "# SIIM-ISIC fixture code bundle\n\n"
            "Deterministic stand-in for the quality gate. It documents the grouped\n"
            "validation recipe; it is not the historical training code.\n"
        ).encode("utf-8")),
        ("grouped_split.py", (
            "\"\"\"Patient- and content-grouped fold assignment (fixture).\"\"\"\n\n"
            "def assign_folds(rows, n_folds=5):\n"
            "    groups = sorted({(row['patient_id'], row['leakage_group']) for row in rows})\n"
            "    return {group: index % n_folds for index, group in enumerate(groups)}\n"
        ).encode("utf-8")),
    ])
    evidence = _zip_bytes([
        ("metrics.json", _json_bytes(METRICS)),
        ("review.json", _json_bytes(REVIEW)),
        ("claim_audit.json", _json_bytes(CLAIM_AUDIT)),
        ("private_grader_ledger.json", _json_bytes(PRIVATE_GRADER_LEDGER)),
    ])
    return {
        "evomind-siim-isic-report.pdf": report,
        "evomind-siim-isic-results.csv": results,
        "evomind-siim-isic-code.zip": code,
        "evomind-siim-isic-evidence.zip": evidence,
    }


def deliverable_index() -> list[dict[str, Any]]:
    rows = []
    for name, payload in deliverable_payloads().items():
        rows.append({
            "name": name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "download_url": f"{DOWNLOAD_PREFIX}/{name}",
        })
    return rows


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def materialize(target: str | Path) -> dict[str, Any]:
    """Create a hermetic workstation root holding only the fixture evidence.

    ``target`` must be an empty or non-existent directory.  Returns a summary
    including a digest over every generated file, recorded in the report.
    """

    root = Path(target).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("quality fixture target must be empty")
    # Markers that make this a valid EvoMind workstation root for active_root().
    (root / ".xsci").mkdir()
    (root / "src" / "xsci").mkdir(parents=True)

    run_dir = root / RUN_RELATIVE
    deliverables = deliverable_index()
    files: dict[str, bytes] = {
        "workspace/current_run.json": _json_bytes({
            "schema": "evomind.current_run.v1",
            "task_id": TASK_ID,
            "run_id": RUN_ID,
            "run_dir": RUN_RELATIVE,
            "status": "verified",
            "updated_at": FIXTURE_UPDATED_AT,
        }),
        f"{RUN_RELATIVE}/run.json": _json_bytes({
            "run_id": RUN_ID,
            "task_id": TASK_ID,
            "status": "verified",
            "objective": (
                "SIIM-ISIC melanoma classification with patient/content grouped OOF "
                f"validation (quality-gate fixture reproducing {SOURCE_RUN_ID})."
            ),
            "tasks": {
                "data_audit": {"status": "completed"},
                "grouped_oof_training": {"status": "completed"},
                "independent_review": {"status": "completed"},
                "claim_audit": {"status": "completed"},
                "delivery": {"status": "completed"},
            },
            "gates": {"official_kaggle_submission": "not_executed", "model_publication": "blocked_by_request"},
            "open_requirements": [],
            "next_action": "calibration and false-positive analysis under the same patient/content groups",
            "updated_at": FIXTURE_UPDATED_AT,
        }),
        f"{RUN_RELATIVE}/request.json": _json_bytes({
            "run_id": RUN_ID,
            "task_id": TASK_ID,
            "task_type": "image_classification",
            "objective": "SIIM-ISIC melanoma classification, grouped validation, no official submission",
            "compute_policy": {"backend": "hpc"},
        }),
        f"{RUN_RELATIVE}/metrics.json": _json_bytes(METRICS),
        f"{RUN_RELATIVE}/review.json": _json_bytes(REVIEW),
        f"{RUN_RELATIVE}/claim_audit.json": _json_bytes(CLAIM_AUDIT),
        f"{RUN_RELATIVE}/private_grader.json": _json_bytes(PRIVATE_GRADER),
        f"{RUN_RELATIVE}/private_grader_ledger.json": _json_bytes(PRIVATE_GRADER_LEDGER),
        f"{RUN_RELATIVE}/candidate_freeze.json": _json_bytes(CANDIDATE_FREEZE),
        f"{RUN_RELATIVE}/deliverables.json": _json_bytes({"files": deliverables}),
        f"{RUN_RELATIVE}/artifact_manifest.json": _json_bytes({
            "run_id": RUN_ID,
            "task_id": TASK_ID,
            "status": "verified",
            "review_status": "review_passed",
            "claim_audit_status": "passed",
            "model_publication": "blocked_by_request",
            "artifacts": [
                {"path": f"delivery/{row['name']}", "sha256": row["sha256"], "bytes": row["bytes"]}
                for row in deliverables
            ],
        }),
        f"workspace/tasks/{TASK_ID}/rag/context_fixture_v1.json": _json_bytes({
            "task_id": TASK_ID,
            "query": "SIIM ISIC melanoma patient level grouped cross validation duplicates",
            "context_path": f"workspace/tasks/{TASK_ID}/rag/context_fixture_v1.md",
            "manifest_path": f"workspace/tasks/{TASK_ID}/rag/context_fixture_v1.json",
            "integrity": {"external_verified": len(PAPERS), "imported": 0, "fabricated": 0},
            "papers": PAPERS,
        }),
        f"workspace/tasks/{TASK_ID}/rag/citation_audits/citation_audit_fixture_v1.json": _json_bytes(CITATION_AUDIT),
    }
    for name, payload in deliverable_payloads().items():
        files[f"{RUN_RELATIVE}/delivery/{name}"] = payload

    digest = hashlib.sha256()
    for relative in sorted(files):
        payload = files[relative]
        _write(root / relative, payload)
        digest.update(relative.encode("utf-8") + b"\0" + hashlib.sha256(payload).digest())
    return {
        "fixture_id": FIXTURE_ID,
        "run_id": RUN_ID,
        "task_id": TASK_ID,
        "source_run_id": SOURCE_RUN_ID,
        "root": str(root),
        "file_count": len(files),
        "content_sha256": digest.hexdigest(),
        "deliverables": deliverables,
    }


__all__ = [
    "FIXTURE_ID",
    "RUN_ID",
    "SOURCE_RUN_ID",
    "TASK_ID",
    "deliverable_index",
    "deliverable_payloads",
    "materialize",
]
