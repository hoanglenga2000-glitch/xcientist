"""Governed SIIM-ISIC HPC research workflow.

The HTTP request that creates a Multi-Agent run is intentionally short lived,
while the governed experiment can span a superseded campaign budget.  This
workflow therefore uses a durable, resumable evidence-ingress contract.  A stage advances only
after the matching real HPC artifact has been collected into either
``<run_dir>/ingress`` or ``workspace/siim_hpc_ingress/<run_id>``.  Missing,
stale, cross-run, or internally inconsistent evidence fails closed and leaves
the same run resumable.

The module never manufactures model metrics and never submits to Kaggle.  It
also binds the terminal private-grader result to an immutable candidate freeze
and records exactly one execution in an append-proof ledger before delivery.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from xsci.user_request import UserRequest

from ..siim_hpc_binding import SiimHpcBinding, binding_from_compute_policy
from .aibuild_v1 import run_directory, write_current_run_pointer
from .multi_agent import (
    AgentResult,
    AgentRoleSpec,
    AgentTask,
    HandoffEnvelope,
    MultiAgentStore,
    MultiAgentSupervisor,
    SupervisorRun,
    create_run,
)

TASK_ID = "siim-isic-melanoma-classification"
REMOTE_ROOT = "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra"
DATASET_MANIFEST_SHA256 = "8fef392368fcc433f4532129312f3b2ef7118d76cad04dc70f694246c34eff13"
EXPECTED_DATASET = {
    "files": 33_129,
    "bytes": 25_765_345_055,
    "train_images": 28_984,
    "test_images": 4_142,
    "positive_rows": 513,
    "patients": 2_056,
}
ABLATION_PROFILES = (
    "raw_multiview",
    "border_removal",
    "color_constancy",
    "hair_suppression",
    "robust_combined_pipeline",
)
ABLATION_SEEDS = (40, 41, 42)
FORMAL_SEEDS = (43, 44, 45)
HISTORICAL_THRESHOLDS = {
    "historical_private_score": 0.92165,
    "bronze": 0.9370,
    "silver": 0.9401,
    "gold": 0.9455,
}
ORIGINAL_BUDGET_HOURS = {
    "ablation": 4,
    "formal_training": 18,
    "delivery": 2,
    "total": 24,
}
OPTION_A_BUDGET_HOURS = {
    "ablation": 4,
    "formal_training": 72,
    "delivery": 2,
    "total": 78,
}
OPTION_A_BUDGET_POLICY = "user_selected_A_corrected_full_closure"
DELIVERABLE_NAMES = (
    "evomind-siim-isic-report.pdf",
    "evomind-siim-isic-results.csv",
    "evomind-siim-isic-code.zip",
    "evomind-siim-isic-evidence.zip",
)
CORE_TRAINING_FILES = (
    "metrics.json",
    "fold_metrics.csv",
    "oof_predictions.csv",
    "submission.csv",
    "training_history.json",
    "hpc_telemetry.jsonl",
)
FROZEN_FILES = (
    "dataset_profile.json",
    "data_audit.json",
    "research_design.json",
    "preprocessing_ablation.json",
    *CORE_TRAINING_FILES,
)
MUTABLE_FILES = {
    "run.json",
    "task_graph.json",
    "events.jsonl",
    "messages.jsonl",
    "handoffs.jsonl",
    "control.json",
    "private_grader.lock",
}


class SiimEvidenceError(RuntimeError):
    failure_type = "evidence"


class SiimResourceGateError(RuntimeError):
    failure_type = "resource"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _safe_run_id(value: str) -> str:
    if (
        not value
        or len(value) > 160
        or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in value)
    ):
        raise ValueError("invalid SIIM run_id")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SiimEvidenceError(f"required SIIM artifact is missing: {path.name}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise SiimEvidenceError(f"invalid SIIM JSON artifact: {path.name}") from exc
    if not isinstance(payload, dict):
        raise SiimEvidenceError(f"SIIM artifact must contain an object: {path.name}")
    return payload


def _manifest_records(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = payload.get("files")
    if not isinstance(records, list) or not records:
        raise SiimEvidenceError("parent immutable manifest has no files")
    indexed: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise SiimEvidenceError("parent immutable manifest contains an invalid record")
        relative = str(record.get("path") or "")
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise SiimEvidenceError("parent immutable manifest contains an unsafe path")
        if relative in indexed:
            raise SiimEvidenceError("parent immutable manifest repeats a path")
        indexed[relative] = record
    return indexed


def _verify_parent_manifest(
    root: Path,
    *,
    parent_run_id: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    _safe_run_id(parent_run_id)
    _require(
        str(manifest.get("parent_run_id") or "") == parent_run_id,
        "parent immutable manifest belongs to a different Run",
    )
    parent_dir = run_directory(root, parent_run_id).resolve()
    _require(parent_dir.is_dir(), "parent Run directory is missing")
    expected = _manifest_records(manifest)
    observed_paths = {
        path.relative_to(parent_dir).as_posix(): path
        for path in parent_dir.rglob("*")
        if path.is_file()
    }
    changed: list[str] = []
    for relative in sorted(set(expected) | set(observed_paths)):
        record = expected.get(relative)
        path = observed_paths.get(relative)
        if record is None or path is None:
            changed.append(relative)
            continue
        if int(record.get("bytes") or -1) != path.stat().st_size:
            changed.append(relative)
            continue
        if str(record.get("sha256") or "").lower() != _sha256(path):
            changed.append(relative)
    _require(not changed, f"parent Run changed after child reservation: {changed}")
    return {
        "passed": True,
        "parent_run_id": parent_run_id,
        "file_count": len(expected),
        "changed_paths": [],
    }


def _prepare_child_lineage(
    root: Path,
    *,
    child_run_id: str,
    parent_run_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate an append-only reservation before any child Run file is created."""

    child_id = _safe_run_id(child_run_id)
    parent_id = _safe_run_id(parent_run_id)
    _require(child_id != parent_id, "child Run must differ from its parent")
    control = root / "workspace" / "siim_evolution_control" / child_id
    constraint_path = control / "constraint_supersession.json"
    manifest_path = control / "parent_immutable_manifest.json"
    requested_change_path = control / "requested_change.json"
    constraint = _read_json(constraint_path)
    manifest = _read_json(manifest_path)
    requested_change = _read_json(requested_change_path)
    _require(constraint.get("status") == "effective", "child constraint supersession is not effective")
    authorized = constraint.get("authorized_change")
    child_contract = constraint.get("child_run_contract")
    parent_contract = constraint.get("parent_run")
    _require(isinstance(authorized, dict), "child authorization is missing")
    _require(isinstance(child_contract, dict), "child execution contract is missing")
    _require(isinstance(parent_contract, dict), "parent execution contract is missing")
    _require(authorized.get("new_candidate_run_allowed") is True, "new child Run is not authorized")
    _require(authorized.get("new_run_id_reserved") == child_id, "reservation belongs to another child Run")
    _require(child_contract.get("run_id") == child_id, "child contract Run ID changed")
    _require(child_contract.get("parent_run_id") == parent_id, "child contract parent changed")
    _require(parent_contract.get("run_id") == parent_id, "parent contract Run ID changed")
    _require(child_contract.get("official_submission") == "forbidden", "official submission boundary changed")
    _require(
        child_contract.get("private_grader") == "once_after_candidate_freeze",
        "child grader boundary changed",
    )
    signals_sent = child_contract.get("signals_sent")
    _require(
        type(signals_sent) is int and signals_sent == 0,
        "child signal boundary changed",
    )
    _require(
        child_contract.get("other_processes_modified") is False,
        "child process-isolation boundary changed",
    )
    _require(
        str(parent_contract.get("immutable_manifest_sha256") or "").lower()
        == _sha256(manifest_path),
        "constraint does not bind the parent manifest",
    )
    _require(requested_change.get("run_id") == child_id, "requested change belongs to another child Run")
    _require(requested_change.get("parent_run_id") == parent_id, "requested change parent changed")
    _require(
        requested_change.get("status") in {"frozen", "approved"},
        "requested change is not frozen",
    )
    changes = requested_change.get("changes")
    _require(isinstance(changes, dict) and bool(changes), "child requested change is empty")
    preservation = _verify_parent_manifest(
        root,
        parent_run_id=parent_id,
        manifest=manifest,
    )
    manifest_index = _manifest_records(manifest)
    required_parent_files = (
        "request.json",
        "candidate_freeze.json",
        "submission.csv",
        "artifact_manifest.json",
        "private_grader_ledger.json",
        "post_completion_reconciliation.json",
    )
    for name in required_parent_files:
        _require(name in manifest_index, f"parent lineage evidence is missing: {name}")
    lineage = {
        "schema": "evomind.siim.run_lineage.v1",
        "created_at": _now(),
        "run_id": child_id,
        "child_run_id": child_id,
        "parent_run_id": parent_id,
        "relation": "scientific_evolution",
        "purpose": authorized.get("purpose"),
        "append_only": True,
        "parent_mutation_allowed": False,
        "parent_evidence": {
            name: {
                "bytes": int(manifest_index[name]["bytes"]),
                "sha256": str(manifest_index[name]["sha256"]),
            }
            for name in required_parent_files
        },
        "parent_immutable_manifest_sha256": _sha256(manifest_path),
        "constraint_supersession_sha256": _sha256(constraint_path),
        "requested_change_sha256": _sha256(requested_change_path),
        "official_submission": "forbidden",
        "private_grader_policy": "at_most_once_after_child_freeze",
        "post_grader_tuning": "forbidden",
        "signals_sent": 0,
        "other_processes_modified": False,
    }
    return lineage, manifest, requested_change, preservation


def _verify_child_lineage(root: Path, run_dir: Path, run_id: str) -> dict[str, Any] | None:
    lineage_path = run_dir / "lineage.json"
    if not lineage_path.is_file():
        return None
    lineage = _read_json(lineage_path)
    _same_run(lineage, run_id, "SIIM lineage")
    _require(lineage.get("append_only") is True, "SIIM lineage is not append-only")
    _require(lineage.get("official_submission") == "forbidden", "lineage submission boundary changed")
    parent_id = _safe_run_id(str(lineage.get("parent_run_id") or ""))
    manifest_path = run_dir / "parent_immutable_manifest.json"
    manifest = _read_json(manifest_path)
    _require(
        lineage.get("parent_immutable_manifest_sha256") == _sha256(manifest_path),
        "lineage parent-manifest hash changed",
    )
    return _verify_parent_manifest(root, parent_run_id=parent_id, manifest=manifest)


def _require(condition: bool, message: str, *, resource: bool = False) -> None:
    if not condition:
        if resource:
            raise SiimResourceGateError(message)
        raise SiimEvidenceError(message)


def _finite(value: Any, *, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SiimEvidenceError(f"expected finite metric, received {value!r}") from exc
    _require(math.isfinite(number), "metric is not finite")
    if minimum is not None:
        _require(number >= minimum, f"metric {number} is below {minimum}")
    if maximum is not None:
        _require(number <= maximum, f"metric {number} is above {maximum}")
    return number


def _artifact(path: Path, run_dir: Path, *, kind: str) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(run_dir)).replace("\\", "/"),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "kind": kind,
    }


def _manifest_entries(run_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in MUTABLE_FILES or path.name == "artifact_manifest.json":
            continue
        if ".tmp" in path.name or "__pycache__" in path.parts or "results" in path.parts:
            continue
        entries.append(_artifact(path, run_dir, kind=path.name))
    return entries


def _copy_file(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f".{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    shutil.copy2(source, temp)
    os.replace(temp, destination)
    return destination


def _same_run(payload: Mapping[str, Any], run_id: str, label: str) -> None:
    _require(str(payload.get("run_id") or "") == run_id, f"{label} belongs to a different run")


def _status_passed(payload: Mapping[str, Any]) -> bool:
    return str(payload.get("status") or "").lower() in {
        "passed",
        "completed",
        "verified",
        "go",
        "ready",
        "candidate_complete",
        "review_passed",
    }


def _slug(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _roles(binding: SiimHpcBinding) -> list[AgentRoleSpec]:
    return [
        AgentRoleSpec(
            "RequestAgent", ("request_contract", "submission_boundary"), output_contract=("workflow_contract.json",)
        ),
        AgentRoleSpec(
            "HpcPreflightAgent",
            ("named_credential_profile", "gpu_coexistence_gate", "dataset_manifest"),
            resource_permissions=("read_collected_hpc_evidence", "write_run_directory"),
            output_contract=("hpc_runtime.json", "dataset_profile.json"),
        ),
        AgentRoleSpec(
            "DataAuditor",
            ("patient_grouping", "content_deduplication", "leakage_audit"),
            output_contract=("data_audit.json",),
        ),
        AgentRoleSpec(
            "ResearchDesigner",
            ("nested_validation", "metric_contract", "budget_contract"),
            output_contract=("research_design.json",),
        ),
        AgentRoleSpec(
            "AblationAgent",
            ("preprocessing_ablation", "three_seed_decision"),
            resource_permissions=("read_collected_hpc_evidence", "write_run_directory"),
            output_contract=("preprocessing_ablation.json", "experiment_comparison.json"),
        ),
        AgentRoleSpec(
            "HpcTrainingAgent",
            ("a800_training", "nested_oof", "resource_telemetry"),
            resource_permissions=(
                binding.job_tag,
                "read_collected_hpc_evidence",
                "write_run_directory",
            ),
            wall_time_seconds=86_400,
            output_contract=CORE_TRAINING_FILES,
        ),
        AgentRoleSpec(
            "IndependentReviewer",
            ("raw_evidence_review", "hash_verification", "candidate_freeze"),
            resource_permissions=("read_run_evidence", "write_run_directory"),
            output_contract=("review.json", "candidate_freeze.json"),
        ),
        AgentRoleSpec(
            "TerminalGrader",
            ("single_private_grader", "post_freeze_binding"),
            resource_permissions=("read_frozen_candidate", "write_grader_ledger"),
            output_contract=("private_grader.json", "private_grader_ledger.json"),
        ),
        AgentRoleSpec(
            "DeliveryAgent",
            ("claim_audit", "manifest", "download_delivery"),
            resource_permissions=("read_reviewed_evidence", "write_run_directory"),
            output_contract=("claim_audit.json", "artifact_manifest.json", *DELIVERABLE_NAMES),
        ),
    ]


def build_siim_hpc_run(request: UserRequest, *, run_id: str) -> SupervisorRun:
    """Build the fixed nine-node, serial, review-gated SIIM task graph."""
    _safe_run_id(run_id)
    binding = binding_from_compute_policy(request.compute_policy)
    tasks = [
        AgentTask(
            "request_setup",
            "Bind the user's objective, compute boundary, evaluation policy, and deliverables.",
            "RequestAgent",
            priority=100,
            max_retries=0,
        ),
        AgentTask(
            "hpc_data_preflight",
            f"Verify {binding.job_tag}, A800 coexistence gate, isolated runtime, complete data manifest, and weights.",
            "HpcPreflightAgent",
            dependencies=("request_setup",),
            priority=90,
            max_retries=0,
        ),
        AgentTask(
            "data_audit",
            "Audit image integrity, class imbalance, patient groups, duplicate-content groups, and leakage.",
            "DataAuditor",
            dependencies=("hpc_data_preflight",),
            priority=80,
            max_retries=0,
        ),
        AgentTask(
            "research_design",
            "Freeze nested patient/content-group validation, metrics, seeds, and the 24-hour budget.",
            "ResearchDesigner",
            dependencies=("data_audit",),
            priority=70,
            max_retries=0,
        ),
        AgentTask(
            "preprocessing_ablation",
            "Compare the five preprocessing profiles on seeds 40, 41, and 42 under fixed groups.",
            "AblationAgent",
            dependencies=("research_design",),
            priority=60,
            resource_type="hpc_gpu",
            timeout_seconds=86_400,
            max_retries=0,
        ),
        AgentTask(
            "full_training",
            "Run the frozen multimodal candidate with disjoint formal seeds and collect complete OOF evidence.",
            "HpcTrainingAgent",
            dependencies=("preprocessing_ablation",),
            priority=50,
            resource_type="hpc_gpu",
            timeout_seconds=86_400,
            max_retries=0,
        ),
        AgentTask(
            "independent_review_freeze",
            "Independently recompute evidence checks and freeze the candidate hashes before grading.",
            "IndependentReviewer",
            dependencies=("full_training",),
            priority=40,
            max_retries=0,
        ),
        AgentTask(
            "terminal_private_grader",
            "Bind exactly one terminal private-grader execution to the immutable candidate.",
            "TerminalGrader",
            dependencies=("independent_review_freeze",),
            priority=30,
            max_retries=0,
        ),
        AgentTask(
            "claim_audit_delivery",
            "Audit claim boundaries and release the hash-bound report, CSV, code, and evidence packages.",
            "DeliveryAgent",
            dependencies=("terminal_private_grader",),
            priority=20,
            max_retries=0,
        ),
    ]
    run = create_run(
        objective=request.objective,
        tasks=tasks,
        roles=_roles(binding),
        run_id=run_id,
        max_concurrency=1,
    )
    run.gates = {
        "job_id": binding.job_id,
        "credential_profile": binding.credential_profile,
        "hpc_resource_gate": "five_samples_required",
        "patient_and_content_group_isolation": "required",
        "candidate_freeze": "required_before_private_grader",
        "private_grader": "terminal_once",
        "post_grader_tuning": "forbidden",
        "official_submission": "forbidden",
        "clinical_diagnosis_claim": "forbidden",
        "independent_review": "required",
        "claim_audit": "required",
    }
    run.resource_limits = {"cpu": 1, "gpu": 0, "hpc_gpu": 1}
    return run


def _pending_section(run_id: str, name: str) -> dict[str, Any]:
    return {"schema": f"evomind.siim.{name}.snapshot.v1", "run_id": run_id, "status": "awaiting_verified_evidence"}


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _deliverable_snapshot(run_dir: Path, run_id: str) -> dict[str, Any]:
    manifest = _read_optional_json(run_dir / "artifact_manifest.json") or {}
    records = {str(item.get("path") or ""): item for item in manifest.get("artifacts") or [] if isinstance(item, dict)}
    files = []
    for name in DELIVERABLE_NAMES:
        path = run_dir / name
        record = records.get(name) or {}
        actual_hash = _sha256(path) if path.is_file() else None
        ready = bool(
            path.is_file()
            and path.stat().st_size > 0
            and str(record.get("sha256") or "").lower() == actual_hash
            and int(record.get("bytes") or -1) == path.stat().st_size
        )
        files.append(
            {
                "name": name,
                "status": "ready" if ready else "unavailable",
                "bytes": path.stat().st_size if ready else None,
                "sha256": actual_hash if ready else None,
                "download_url": f"/api/multi-agent/runs/{run_id}/download/{name}" if ready else None,
            }
        )
    return {
        "schema": "evomind.siim.deliverables.snapshot.v1",
        "run_id": run_id,
        "status": "ready" if all(item["status"] == "ready" for item in files) else "unavailable",
        "files": files,
    }


def _snapshot_payload(run_dir: Path, run: SupervisorRun) -> dict[str, Any]:
    def section(filename: str, name: str) -> dict[str, Any]:
        payload = _read_optional_json(run_dir / filename)
        if payload is None:
            return _pending_section(run.run_id, name)
        return {**payload, "run_id": run.run_id}

    review = section("review.json", "review")
    claim = _read_optional_json(run_dir / "claim_audit.json")
    if claim is not None:
        review = {**review, "claim_audit": {**claim, "run_id": run.run_id}}
    return {
        "task_type": "image_classification",
        "dataset": TASK_ID,
        "dataset_profile": section("dataset_profile.json", "dataset_profile"),
        "experiment_comparison": section("experiment_comparison.json", "experiment_comparison"),
        "hpc_runtime": section("hpc_runtime.json", "hpc_runtime"),
        "review": review,
        "historical_thresholds": {
            "schema": "evomind.siim.historical_thresholds.v1",
            "run_id": run.run_id,
            "status": "reference_only",
            **HISTORICAL_THRESHOLDS,
            "claim_boundary": "Historical MLE-Bench thresholds are context, not this run's rank or medal.",
        },
        "deliverables": _deliverable_snapshot(run_dir, run.run_id),
    }


def _snapshot_store(run_dir: Path) -> MultiAgentStore:
    def on_save(run: SupervisorRun) -> None:
        _atomic_json(run_dir / "run.json", {**run.to_dict(), **_snapshot_payload(run_dir, run)})

    return MultiAgentStore(run_dir, on_save=on_save)


class SiimHpcExecutors:
    """Deterministic validators for externally collected, same-run evidence."""

    def __init__(self, *, workspace_root: Path, run_dir: Path, request: UserRequest) -> None:
        self.root = workspace_root
        self.run_dir = run_dir
        self.request = request
        self.binding = binding_from_compute_policy(request.compute_policy)
        self.external_ingress = self.root / "workspace" / "siim_hpc_ingress" / self.run_dir.name

    def _budget_contract(self, run_id: str) -> dict[str, Any]:
        """Resolve one immutable budget at request setup.

        The original workflow hard-coded the 24-hour template even when the
        campaign supervisor had already frozen the user-selected 78-hour
        option-A plan.  Prefer that same-run plan when it exists, validate its
        provenance fail-closed, and otherwise retain the documented original
        default for ordinary runs.
        """

        campaign_plan = (
            self.root
            / "workspace"
            / "hpc"
            / f"{self.binding.job_tag}_siim_campaign"
            / run_id
            / "campaign_plan.json"
        )
        if not campaign_plan.is_file():
            return {
                "budget_hours": dict(ORIGINAL_BUDGET_HOURS),
                "original_budget_hours": dict(ORIGINAL_BUDGET_HOURS),
                "budget_policy": "original_24_hour_template",
                "budget_source": "request_default",
                "budget_source_sha256": None,
            }

        payload = _read_json(campaign_plan)
        _same_run(payload, run_id, "campaign budget plan")
        budget = payload.get("budget_hours")
        original = payload.get("original_budget_hours")
        _require(isinstance(budget, dict), "campaign budget plan has no budget_hours")
        _require(isinstance(original, dict), "campaign budget plan has no original budget provenance")
        normalized = {str(key): int(value) for key, value in budget.items()}
        normalized_original = {str(key): int(value) for key, value in original.items()}
        _require(normalized == OPTION_A_BUDGET_HOURS, "campaign budget is not the corrected option-A plan")
        _require(normalized_original == ORIGINAL_BUDGET_HOURS, "campaign original budget provenance changed")
        _require(
            payload.get("budget_policy") == OPTION_A_BUDGET_POLICY,
            "campaign budget policy is not the corrected option-A policy",
        )
        return {
            "budget_hours": normalized,
            "original_budget_hours": normalized_original,
            "budget_policy": OPTION_A_BUDGET_POLICY,
            "budget_source": str(campaign_plan.relative_to(self.root)),
            "budget_source_sha256": _sha256(campaign_plan),
        }

    def source(self, relative: str) -> Path:
        normalized = Path(relative)
        _require(not normalized.is_absolute() and ".." not in normalized.parts, "unsafe SIIM ingress path")
        local = self.run_dir / "ingress" / normalized
        external = self.external_ingress / normalized
        if local.is_file():
            return local
        if external.is_file():
            return external
        raise SiimEvidenceError(f"required SIIM artifact is missing: {relative}")

    def request_setup(self, task: AgentTask, _handoff: HandoffEnvelope, run: SupervisorRun) -> AgentResult:
        _require(self.request.task_type == "image_classification", "SIIM workflow requires image_classification")
        _require(self.request.dataset == TASK_ID, "SIIM workflow received a different dataset")
        _require(self.request.compute_policy.backend == "hpc", "SIIM workflow requires the hpc backend", resource=True)
        _require(
            self.request.compute_policy.remote_gpu_required, "SIIM workflow requires remote GPU compute", resource=True
        )
        _require(
            not self.request.compute_policy.local_gpu_allowed, "SIIM workflow excludes local GPU compute", resource=True
        )
        _require(
            self.request.submission_policy.official_submission == "forbidden", "official submission must be forbidden"
        )
        budget = self._budget_contract(run.run_id)
        output = self.run_dir / "workflow_contract.json"
        _atomic_json(
            output,
            {
                "schema": "evomind.siim.workflow_contract.v1",
                "run_id": run.run_id,
                "task_type": "image_classification",
                "dataset": TASK_ID,
                "compute_policy": {
                    "backend": "hpc",
                    "job_id": self.binding.job_id,
                    "credential_profile": self.binding.credential_profile,
                    "remote_root": REMOTE_ROOT,
                },
                "submission_policy": {"official_submission": "forbidden", "private_grader": "once_after_freeze"},
                "ablation_seeds": list(ABLATION_SEEDS),
                "formal_seeds": list(FORMAL_SEEDS),
                "budget_hours": budget["budget_hours"],
                "original_budget_hours": budget["original_budget_hours"],
                "budget_policy": budget["budget_policy"],
                "budget_source": budget["budget_source"],
                "budget_source_sha256": budget["budget_source_sha256"],
                "ingress_roots": ["ingress", f"workspace/siim_hpc_ingress/{run.run_id}"],
                "required_deliverables": list(DELIVERABLE_NAMES),
                "created_at": _now(),
            },
        )
        artifact = _artifact(output, self.run_dir, kind="workflow_contract")
        return AgentResult(
            task.task_id,
            "SIIM request and immutable execution boundaries bound",
            [artifact["path"]],
            artifacts=[artifact],
            confidence=1.0,
        )

    def preflight(self, task: AgentTask, _handoff: HandoffEnvelope, run: SupervisorRun) -> AgentResult:
        runtime_source = self.source("hpc_preflight.json")
        dataset_source = self.source("dataset_profile.json")
        runtime = _read_json(runtime_source)
        dataset = _read_json(dataset_source)
        _same_run(runtime, run.run_id, "HPC preflight")
        _same_run(dataset, run.run_id, "dataset profile")
        _require(_status_passed(runtime), "HPC preflight is not passed", resource=True)
        _require(
            str(runtime.get("job_id") or "") == str(self.binding.job_id),
            f"HPC preflight is not bound to {self.binding.job_tag}",
            resource=True,
        )
        _require(
            str(runtime.get("credential_profile") or "") == self.binding.credential_profile,
            "wrong HPC credential profile",
            resource=True,
        )
        _require(str(runtime.get("remote_root") or "") == REMOTE_ROOT, "HPC remote root is not confined", resource=True)
        gpu = runtime.get("gpu") if isinstance(runtime.get("gpu"), dict) else {}
        _require("A800" in str(gpu.get("name") or runtime.get("gpu_name") or ""), "expected an A800 GPU", resource=True)
        _require(
            float(gpu.get("memory_total_mb") or runtime.get("memory_total_mb") or 0) >= 80_000,
            "A800 memory inventory is incomplete",
            resource=True,
        )
        samples = runtime.get("samples") if isinstance(runtime.get("samples"), list) else []
        _require(len(samples) >= 5, "HPC preflight requires five resource samples", resource=True)
        for sample in samples[:5]:
            _require(isinstance(sample, dict), "invalid HPC sample", resource=True)
            free_mb = float(sample.get("free_memory_mb") or sample.get("memory_free_mb") or 0)
            other_mb = float(sample.get("other_process_memory_mb") or 0)
            _require(free_mb >= 60 * 1024, "A800 free memory is below the coexistence gate", resource=True)
            _require(other_mb <= 8 * 1024, "other GPU processes exceed the coexistence gate", resource=True)
        _require(
            runtime.get("other_processes_modified") is False,
            "preflight did not attest other_processes_modified=false",
            resource=True,
        )
        _require(int(runtime.get("signals_sent") or 0) == 0, "preflight sent a process signal", resource=True)
        _require(
            str(runtime.get("launch_decision") or "").lower() == "go", "HPC launch decision is not GO", resource=True
        )
        _require(_status_passed(dataset), "dataset profile is not passed")
        _require(str(dataset.get("dataset") or dataset.get("competition_id") or "") == TASK_ID, "wrong dataset profile")
        observed = dataset.get("counts") if isinstance(dataset.get("counts"), dict) else dataset
        for key, expected in EXPECTED_DATASET.items():
            _require(int(observed.get(key) or -1) == expected, f"dataset {key} mismatch")
        _require(
            str(dataset.get("manifest_sha256") or "").lower() == DATASET_MANIFEST_SHA256,
            "dataset manifest hash mismatch",
        )
        _require(dataset.get("complete") is True, "dataset profile is not complete")
        runtime_path = _copy_file(runtime_source, self.run_dir / "hpc_runtime.json")
        dataset_path = _copy_file(dataset_source, self.run_dir / "dataset_profile.json")
        artifacts = [
            _artifact(runtime_path, self.run_dir, kind="hpc_runtime"),
            _artifact(dataset_path, self.run_dir, kind="dataset_profile"),
        ]
        return AgentResult(
            task.task_id,
            f"{self.binding.job_tag} A800 and complete SIIM data passed the coexistence gate",
            [item["path"] for item in artifacts],
            artifacts=artifacts,
            confidence=1.0,
        )

    def data_audit(self, task: AgentTask, _handoff: HandoffEnvelope, run: SupervisorRun) -> AgentResult:
        source = self.source("data_audit.json")
        audit = _read_json(source)
        _same_run(audit, run.run_id, "data audit")
        _require(_status_passed(audit), "SIIM data audit is not passed")
        _require(
            int(audit.get("train_rows") or -1) == EXPECTED_DATASET["train_images"], "data audit train row mismatch"
        )
        _require(int(audit.get("test_rows") or -1) == EXPECTED_DATASET["test_images"], "data audit test row mismatch")
        _require(
            int(audit.get("positive_rows") or -1) == EXPECTED_DATASET["positive_rows"],
            "data audit positive row mismatch",
        )
        _require(int(audit.get("patients") or -1) == EXPECTED_DATASET["patients"], "data audit patient count mismatch")
        _require(
            int(audit["patient_group_overlap"]) == 0 if "patient_group_overlap" in audit else False,
            "patient groups overlap across folds",
        )
        _require(
            int(audit["content_group_overlap"]) == 0 if "content_group_overlap" in audit else False,
            "duplicate-content groups overlap across folds",
        )
        _require(audit.get("target_in_test_features") is False, "target leaked into test features")
        _require(
            int(audit.get("private_label_access_count") or 0) == 0, "private labels were accessed during data audit"
        )
        _require(
            str(audit.get("split_policy") or "") in {"patient_and_content_grouped", "patient_content_grouped"},
            "wrong split policy",
        )
        output = _copy_file(source, self.run_dir / "data_audit.json")
        artifact = _artifact(output, self.run_dir, kind="data_audit")
        return AgentResult(
            task.task_id,
            "Patient/content grouping and leakage audit passed",
            [artifact["path"]],
            artifacts=[artifact],
            confidence=1.0,
        )

    def design(self, task: AgentTask, _handoff: HandoffEnvelope, run: SupervisorRun) -> AgentResult:
        _require((self.run_dir / "data_audit.json").is_file(), "data audit evidence is missing")
        workflow_contract = _read_json(self.run_dir / "workflow_contract.json")
        _same_run(workflow_contract, run.run_id, "workflow contract")
        workflow_budget = workflow_contract.get("budget_hours")
        _require(isinstance(workflow_budget, dict), "workflow contract has no structured budget")
        budget_hours = {str(key): int(value) for key, value in workflow_budget.items()}
        _require(
            budget_hours in (ORIGINAL_BUDGET_HOURS, OPTION_A_BUDGET_HOURS),
            "workflow contract contains an unsupported budget",
        )
        design_budget = {
            "ablation": budget_hours["ablation"],
            "formal_training": budget_hours["formal_training"],
            "review_grade_delivery": budget_hours["delivery"],
            "total": budget_hours["total"],
        }
        output = self.run_dir / "research_design.json"
        _atomic_json(
            output,
            {
                "schema": "evomind.siim.research_design.v1",
                "run_id": run.run_id,
                "status": "frozen_before_experiment",
                "primary_metric": "roc_auc",
                "secondary_metrics": [
                    "pr_auc",
                    "brier",
                    "calibration",
                    "patient_group_bootstrap_95ci",
                    "sensitivity",
                    "specificity",
                    "precision",
                    "negative_predictive_value",
                ],
                "validation": {
                    "outer_folds": 5,
                    "inner_folds": 3,
                    "grouping": ["patient_id", "duplicate_content_group"],
                    "threshold_source": "OOF_only",
                },
                "preprocessing_profiles": list(ABLATION_PROFILES),
                "ablation_seeds": list(ABLATION_SEEDS),
                "formal_seeds": list(FORMAL_SEEDS),
                "seed_sets_disjoint": set(ABLATION_SEEDS).isdisjoint(FORMAL_SEEDS),
                "adoption_gate": {
                    "minimum_mean_gain": 0.0005,
                    "maximum_worst_fold_regression": 0.002,
                    "minimum_seed_passes": 2,
                },
                "models": [
                    "ConvNeXt-Small full image",
                    "EfficientNetV2-S lesion focus",
                    "patient metadata fusion",
                    "CatBoost metadata",
                ],
                "training": {
                    "input_size": 384,
                    "precision": "BF16",
                    "memory_format": "channels_last",
                    "tta": True,
                    "batch_probe": [128, 96, 64, 48, 32],
                    "max_workers": 8,
                },
                "budget_hours": design_budget,
                "original_budget_hours": workflow_contract.get("original_budget_hours"),
                "budget_policy": workflow_contract.get("budget_policy"),
                "budget_source": workflow_contract.get("budget_source"),
                "budget_source_sha256": workflow_contract.get("budget_source_sha256"),
                "official_submission": "forbidden",
                "clinical_claim": "medical_imaging_research_benchmark_only",
                "created_at": _now(),
            },
        )
        artifact = _artifact(output, self.run_dir, kind="research_design")
        return AgentResult(
            task.task_id,
            f"Nested grouped validation and {budget_hours['total']}-hour research design frozen",
            [artifact["path"]],
            artifacts=[artifact],
            confidence=1.0,
        )

    def ablation(self, task: AgentTask, _handoff: HandoffEnvelope, run: SupervisorRun) -> AgentResult:
        source = self.source("preprocessing_ablation.json")
        payload = _read_json(source)
        _same_run(payload, run.run_id, "preprocessing ablation")
        _require(_status_passed(payload), "preprocessing ablation is not complete")
        _require(
            tuple(int(seed) for seed in payload.get("seeds") or []) == ABLATION_SEEDS,
            "ablation seeds differ from 40/41/42",
        )
        records = payload.get("profiles") if isinstance(payload.get("profiles"), list) else payload.get("candidates")
        _require(isinstance(records, list), "preprocessing ablation profile records are missing")
        names = {_slug(record.get("profile") or record.get("name")) for record in records if isinstance(record, dict)}
        _require(names == set(ABLATION_PROFILES), "preprocessing ablation did not compare all five profiles")
        _require(
            int(payload["patient_group_overlap"]) == 0 if "patient_group_overlap" in payload else False,
            "ablation patient groups overlap",
        )
        _require(
            int(payload["content_group_overlap"]) == 0 if "content_group_overlap" in payload else False,
            "ablation content groups overlap",
        )
        selected = _slug(payload.get("selected_profile"))
        _require(selected in ABLATION_PROFILES, "selected preprocessing profile is invalid")
        decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else payload
        mean_gain = _finite(decision.get("mean_gain"), minimum=-1.0, maximum=1.0)
        worst_fold_delta = _finite(decision.get("worst_fold_delta"), minimum=-1.0, maximum=1.0)
        seed_passes = int(decision.get("seed_passes") or 0)
        if selected != "raw_multiview":
            _require(mean_gain >= 0.0005, "selected preprocessing profile misses the mean-gain gate")
            _require(worst_fold_delta >= -0.002, "selected preprocessing profile exceeds worst-fold regression")
            _require(seed_passes >= 2, "selected preprocessing profile passes fewer than two seeds")
        output = _copy_file(source, self.run_dir / "preprocessing_ablation.json")
        comparison_path = self.run_dir / "experiment_comparison.json"
        _atomic_json(
            comparison_path,
            {
                "schema": "evomind.siim.experiment_comparison.v1",
                "run_id": run.run_id,
                "status": "passed",
                "profiles": records,
                "selected_profile": selected,
                "decision": {
                    "mean_gain": mean_gain,
                    "worst_fold_delta": worst_fold_delta,
                    "seed_passes": seed_passes,
                    "adopted": selected != "raw_multiview",
                },
                "invalid_changes_rejected": list(payload.get("rejected_profiles") or []),
                "source_sha256": _sha256(output),
            },
        )
        artifacts = [
            _artifact(output, self.run_dir, kind="preprocessing_ablation"),
            _artifact(comparison_path, self.run_dir, kind="experiment_comparison"),
        ]
        return AgentResult(
            task.task_id,
            f"Five-profile ablation retained {selected}",
            [item["path"] for item in artifacts],
            artifacts=artifacts,
            confidence=1.0,
        )

    @staticmethod
    def _csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = list(reader.fieldnames or [])
                rows = list(reader)
        except (OSError, csv.Error) as exc:
            raise SiimEvidenceError(f"invalid CSV artifact: {path.name}") from exc
        _require(bool(fieldnames), f"CSV header is missing: {path.name}")
        return fieldnames, rows

    def _validate_training_files(self, run_id: str) -> dict[str, Any]:
        sources = {name: self.source(f"training/{name}") for name in CORE_TRAINING_FILES}
        result_source = self.source("training/training_result.json")
        result = _read_json(result_source)
        _same_run(result, run_id, "training result")
        _require(_status_passed(result), "formal SIIM training is not complete")
        formal_seeds = tuple(int(seed) for seed in result.get("formal_seeds") or [])
        _require(bool(formal_seeds), "formal training seeds are missing")
        _require(set(formal_seeds).isdisjoint(ABLATION_SEEDS), "formal training reused an ablation seed")
        _require(set(formal_seeds).issubset(FORMAL_SEEDS), "formal training used an unapproved seed")
        _require(result.get("official_submission_executed") is False, "formal training executed an official submission")
        _require(
            int(result.get("private_grader_execution_count") or 0) == 0, "private grader ran before candidate freeze"
        )
        _require(int(result.get("private_label_access_count") or 0) == 0, "private labels were used during training")
        metrics = _read_json(sources["metrics.json"])
        _same_run(metrics, run_id, "metrics")
        for key in ("roc_auc", "pr_auc", "brier"):
            _finite(metrics.get(key), minimum=0.0, maximum=1.0)
        _require(metrics.get("mle_private_grader_score") is None, "training metrics contain a pre-freeze private score")
        oof_fields, oof_rows = self._csv_rows(sources["oof_predictions.csv"])
        _require(len(oof_rows) == EXPECTED_DATASET["train_images"], "OOF row count mismatch")
        id_key = next((key for key in ("image_name", "image_id") if key in oof_fields), "")
        prediction_key = next(
            (
                key
                for key in ("probability", "prediction", "oof_probability", "target_probability")
                if key in oof_fields
            ),
            "",
        )
        _require(bool(id_key and prediction_key and "fold" in oof_fields), "OOF schema is incomplete")
        identifiers = [row[id_key] for row in oof_rows]
        _require(len(set(identifiers)) == len(identifiers), "OOF identifiers are duplicated")
        for row in oof_rows:
            _finite(row[prediction_key], minimum=0.0, maximum=1.0)
            _require(str(row.get("fold") or "").strip() != "", "OOF fold is missing")
            if "coverage" in oof_fields:
                _require(int(float(row["coverage"])) == 1, "OOF coverage must equal one")
        submission_fields, submission_rows = self._csv_rows(sources["submission.csv"])
        _require(
            submission_fields == ["image_name", "target"], "submission schema must exactly match image_name,target"
        )
        _require(len(submission_rows) == EXPECTED_DATASET["test_images"], "submission row count mismatch")
        _require(
            len({row["image_name"] for row in submission_rows}) == len(submission_rows),
            "submission identifiers are duplicated",
        )
        for row in submission_rows:
            _finite(row["target"], minimum=0.0, maximum=1.0)
        sample = self.source("training/sample_submission.csv")
        sample_fields, sample_rows = self._csv_rows(sample)
        _require(sample_fields == submission_fields, "sample submission schema mismatch")
        _require(
            [row["image_name"] for row in sample_rows] == [row["image_name"] for row in submission_rows],
            "submission order differs from sample_submission.csv",
        )
        fold_fields, fold_rows = self._csv_rows(sources["fold_metrics.csv"])
        _require("fold" in fold_fields and len(fold_rows) >= 5, "fold metrics are incomplete")
        _require({int(row["fold"]) for row in fold_rows} == set(range(5)), "outer-fold metrics must cover folds 0..4")
        telemetry_lines = [
            line for line in sources["hpc_telemetry.jsonl"].read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        _require(bool(telemetry_lines), "HPC telemetry is empty")
        telemetry_events: list[dict[str, Any]] = []
        for line in telemetry_lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SiimEvidenceError("HPC telemetry contains invalid JSON") from exc
            _require(str(event.get("run_id") or "") == run_id, "HPC telemetry contains a different run")
            _require(
                int(event.get("job_id") or 0) == self.binding.job_id,
                "HPC telemetry contains a different job binding",
            )
            _require(
                event.get("credential_profile") == self.binding.credential_profile,
                "HPC telemetry contains a different credential profile",
            )
            telemetry_events.append(event)
        history = _read_json(sources["training_history.json"])
        _same_run(history, run_id, "training history")
        return {
            "result": result,
            "result_source": result_source,
            "sources": sources,
            "sample_source": sample,
            "metrics": metrics,
            "history": history,
            "telemetry_events": telemetry_events,
        }

    def training(self, task: AgentTask, _handoff: HandoffEnvelope, run: SupervisorRun) -> AgentResult:
        validated = self._validate_training_files(run.run_id)
        copied = []
        for name, source in validated["sources"].items():
            copied.append(_copy_file(source, self.run_dir / name))
        copied.append(_copy_file(validated["sample_source"], self.run_dir / "sample_submission.csv"))
        result_path = _copy_file(validated["result_source"], self.run_dir / "training_result.json")
        copied.append(result_path)
        runtime_path = self.run_dir / "hpc_runtime.json"
        runtime = _read_json(runtime_path)
        formal_seeds = [int(seed) for seed in validated["result"].get("formal_seeds") or []]
        source_runs = validated["history"].get("source_runs")
        source_runs = source_runs if isinstance(source_runs, list) else []
        fold_records: list[dict[str, Any]] = []
        for source_run in source_runs:
            if not isinstance(source_run, dict):
                continue
            history = source_run.get("history")
            folds = history.get("folds") if isinstance(history, dict) else None
            if isinstance(folds, list):
                fold_records.extend(record for record in folds if isinstance(record, dict))
        selected_epochs = [
            int(record["selected_epoch"])
            for record in fold_records
            if isinstance(record.get("selected_epoch"), (int, float))
        ]
        allocated_memory = [
            int(record[key])
            for record in fold_records
            for key in ("selection_peak_memory_allocated_mib", "peak_memory_allocated_mib")
            if isinstance(record.get(key), (int, float))
        ]
        expected_outer_folds = len(formal_seeds) * 5
        recorded_outer_folds = len(fold_records)
        if not recorded_outer_folds and isinstance(validated["history"].get("outer_folds"), (int, float)):
            recorded_outer_folds = int(validated["history"]["outer_folds"])
        gpu_samples = [
            event["gpu"]
            for event in validated["telemetry_events"]
            if isinstance(event.get("gpu"), dict)
        ]
        memory_used = [
            int(sample["memory_used_mib"])
            for sample in gpu_samples
            if isinstance(sample.get("memory_used_mib"), (int, float))
        ]
        utilizations = [
            int(sample["utilization_percent"])
            for sample in gpu_samples
            if isinstance(sample.get("utilization_percent"), (int, float))
        ]
        other_process_memory = [
            int(event["other_process_memory_mib"])
            for event in validated["telemetry_events"]
            if isinstance(event.get("other_process_memory_mib"), (int, float))
        ]
        runtime.update(
            {
                "run_id": run.run_id,
                "training_status": "completed",
                "formal_seeds": formal_seeds,
                "selected_batch_size": validated["result"].get("selected_batch_size"),
                "other_processes_modified": False,
                "signals_sent": 0,
                "telemetry_path": "hpc_telemetry.jsonl",
                "training_progress": {
                    "status": "completed",
                    "completed_formal_seeds": len(formal_seeds),
                    "total_formal_seeds": len(FORMAL_SEEDS),
                    "completed_outer_folds": recorded_outer_folds,
                    "total_outer_folds": expected_outer_folds,
                    "selected_epoch_min": min(selected_epochs) if selected_epochs else None,
                    "selected_epoch_max": max(selected_epochs) if selected_epochs else None,
                    "peak_model_memory_allocated_mib": max(allocated_memory) if allocated_memory else None,
                },
                "telemetry_summary": {
                    "sample_count": len(validated["telemetry_events"]),
                    "peak_gpu_memory_used_mib": max(memory_used) if memory_used else None,
                    "max_gpu_utilization_percent": max(utilizations) if utilizations else None,
                    "max_other_process_memory_mib": max(other_process_memory) if other_process_memory else 0,
                    "hold_sample_count": sum(
                        1 for event in validated["telemetry_events"] if bool(event.get("hold_reasons"))
                    ),
                },
            }
        )
        _atomic_json(runtime_path, runtime)
        artifacts = [_artifact(path, self.run_dir, kind=path.name) for path in copied]
        return AgentResult(
            task.task_id,
            "Formal A800 training and complete grouped OOF evidence verified",
            [item["path"] for item in artifacts],
            metrics={key: validated["metrics"].get(key) for key in ("roc_auc", "pr_auc", "brier")},
            artifacts=artifacts,
            confidence=1.0,
        )

    def review_freeze(self, task: AgentTask, _handoff: HandoffEnvelope, run: SupervisorRun) -> AgentResult:
        source = self.source("review.json")
        review = _read_json(source)
        _same_run(review, run.run_id, "independent review")
        _require(_status_passed(review), "independent review is not passed")
        checks = review.get("checks") if isinstance(review.get("checks"), dict) else {}
        for name in (
            "patient_group_overlap_zero",
            "content_group_overlap_zero",
            "oof_coverage_exactly_once",
            "submission_schema_and_order",
            "private_labels_unavailable_during_training",
            "private_grader_not_executed",
            "official_submission_not_executed",
        ):
            _require(checks.get(name) is True, f"independent review check failed: {name}")
        claimed_hashes = review.get("artifact_hashes") if isinstance(review.get("artifact_hashes"), dict) else {}
        frozen: list[dict[str, Any]] = []
        for name in FROZEN_FILES:
            path = self.run_dir / name
            _require(path.is_file(), f"freeze input is missing: {name}")
            actual = _sha256(path)
            _require(str(claimed_hashes.get(name) or "").lower() == actual, f"review hash mismatch: {name}")
            frozen.append(_artifact(path, self.run_dir, kind=name))
        lineage_path = self.run_dir / "lineage.json"
        lineage: dict[str, Any] | None = None
        if lineage_path.is_file():
            lineage = _read_json(lineage_path)
            _same_run(lineage, run.run_id, "SIIM lineage")
            _verify_child_lineage(self.root, self.run_dir, run.run_id)
            frozen.append(_artifact(lineage_path, self.run_dir, kind="run_lineage"))
        review_path = _copy_file(source, self.run_dir / "review.json")
        freeze_path = self.run_dir / "candidate_freeze.json"
        config_hash = hashlib.sha256("".join(item["sha256"] for item in frozen).encode("ascii")).hexdigest()
        _atomic_json(
            freeze_path,
            {
                "schema": "evomind.siim.candidate_freeze.v1",
                "run_id": run.run_id,
                "status": "frozen_before_private_grader",
                "frozen_at": _now(),
                "configuration_sha256": config_hash,
                "artifacts": frozen,
                "private_grader_execution_count_before_freeze": 0,
                "tuning_closed": True,
                "official_submission": "forbidden",
                **(
                    {
                        "parent_run_id": lineage["parent_run_id"],
                        "lineage_sha256": _sha256(lineage_path),
                    }
                    if lineage is not None
                    else {}
                ),
            },
        )
        artifacts = [
            _artifact(review_path, self.run_dir, kind="independent_review"),
            _artifact(freeze_path, self.run_dir, kind="candidate_freeze"),
        ]
        return AgentResult(
            task.task_id,
            "Independent review passed and candidate hashes frozen",
            [item["path"] for item in artifacts],
            artifacts=artifacts,
            confidence=1.0,
        )

    def _verify_freeze(self, run_id: str) -> dict[str, Any]:
        freeze_path = self.run_dir / "candidate_freeze.json"
        freeze = _read_json(freeze_path)
        _same_run(freeze, run_id, "candidate freeze")
        _require(freeze.get("status") == "frozen_before_private_grader", "candidate is not frozen")
        _require(freeze.get("tuning_closed") is True, "tuning is not closed")
        lineage_path = self.run_dir / "lineage.json"
        if lineage_path.is_file():
            lineage = _read_json(lineage_path)
            _same_run(lineage, run_id, "SIIM lineage")
            _require(
                freeze.get("parent_run_id") == lineage.get("parent_run_id"),
                "candidate freeze parent lineage changed",
            )
            _require(
                freeze.get("lineage_sha256") == _sha256(lineage_path),
                "candidate freeze lineage hash changed",
            )
            _verify_child_lineage(self.root, self.run_dir, run_id)
        records = freeze.get("artifacts") if isinstance(freeze.get("artifacts"), list) else []
        _require(bool(records), "candidate freeze artifact list is empty")
        for record in records:
            _require(isinstance(record, dict), "invalid candidate freeze record")
            relative = str(record.get("path") or "")
            path = (self.run_dir / relative).resolve()
            try:
                path.relative_to(self.run_dir.resolve())
            except ValueError as exc:
                raise SiimEvidenceError("candidate freeze path escapes the run") from exc
            _require(path.is_file(), f"frozen artifact is missing: {relative}")
            _require(_sha256(path) == str(record.get("sha256") or "").lower(), f"frozen artifact changed: {relative}")
        return freeze

    def terminal_grader(self, task: AgentTask, _handoff: HandoffEnvelope, run: SupervisorRun) -> AgentResult:
        freeze = self._verify_freeze(run.run_id)
        source = self.source("private_grader.json")
        result_hash = _sha256(source)
        freeze_hash = _sha256(self.run_dir / "candidate_freeze.json")
        ledger_path = self.run_dir / "private_grader_ledger.json"
        output = self.run_dir / "private_grader.json"
        lock_path = self.run_dir / "private_grader.lock"

        def validated_payload() -> tuple[dict[str, Any], float | None]:
            payload = _read_json(source)
            _same_run(payload, run.run_id, "private grader result")
            grader_status = str(payload.get("status") or "").lower()
            _require(_status_passed(payload) or grader_status == "failed_closed", "terminal private grader is not complete")
            _require(
                int(payload.get("execution_index") or payload.get("execution_count") or 0) == 1,
                "private grader execution index must be one",
            )
            _require(
                str(payload.get("candidate_freeze_sha256") or "").lower() == freeze_hash,
                "private grader is not bound to the candidate freeze",
            )
            _require(
                payload.get("executed_after_freeze") is True,
                "private grader did not attest post-freeze execution",
            )
            _require(
                payload.get("feedback_used_for_tuning") is False,
                "private grader feedback was used for tuning",
            )
            _require(
                payload.get("official_submission_executed") is False,
                "private grader path executed an official submission",
            )
            if grader_status == "failed_closed":
                _require(
                    payload.get("mle_private_grader_score", payload.get("score")) in (None, ""),
                    "failed-closed grader must not contain a score",
                )
                _require(
                    bool(str(payload.get("error") or payload.get("failure_reason") or "").strip()),
                    "failed-closed grader lacks failure evidence",
                )
                score = None
            else:
                score = _finite(
                    payload.get("mle_private_grader_score", payload.get("score")),
                    minimum=0.0,
                    maximum=1.0,
                )
            return payload, score

        def write_ledger(payload: Mapping[str, Any], score: float | None) -> None:
            _require(
                output.is_file() and not output.is_symlink() and _sha256(output) == result_hash,
                "registered private grader artifact changed",
            )
            _atomic_json(
                ledger_path,
                {
                    "schema": "evomind.siim.private_grader_ledger.v1",
                    "run_id": run.run_id,
                    "job_id": self.binding.job_id,
                    "credential_profile": self.binding.credential_profile,
                    "status": "terminal_execution_recorded",
                    "execution_count": 1,
                    "execution_id": payload.get("execution_id"),
                    "candidate_freeze_sha256": freeze_hash,
                    "configuration_sha256": freeze.get("configuration_sha256"),
                    "result_sha256": result_hash,
                    "score": score,
                    "outcome": (
                        "failed_closed"
                        if str(payload.get("status") or "").lower() == "failed_closed"
                        else "scored"
                    ),
                    "feedback_used_for_tuning": False,
                    "official_submission_executed": False,
                    "recorded_at": _now(),
                },
            )

        def remove_registration_lock() -> None:
            if not lock_path.exists():
                return
            _require(
                lock_path.is_file() and not lock_path.is_symlink(),
                "private grader registration lock is unsafe",
            )
            lock_path.unlink()

        existing = _read_optional_json(ledger_path)
        if existing is not None:
            _require(int(existing.get("execution_count") or 0) == 1, "private grader ledger count is not one")
            _require(
                int(existing.get("job_id") or 0) == self.binding.job_id,
                "private grader ledger job binding changed",
            )
            _require(
                existing.get("credential_profile") == self.binding.credential_profile,
                "private grader ledger credential profile changed",
            )
            _require(
                str(existing.get("result_sha256") or "").lower() == result_hash,
                "a different private grader result was supplied",
            )
            _require(
                str(existing.get("candidate_freeze_sha256") or "").lower() == freeze_hash,
                "private grader ledger freeze mismatch",
            )
            _require(
                output.is_file() and not output.is_symlink() and _sha256(output) == result_hash,
                "registered private grader artifact changed",
            )
            remove_registration_lock()
            artifact = _artifact(output, self.run_dir, kind="terminal_private_grader")
            return AgentResult(
                task.task_id,
                "Existing terminal private-grader result reused without another execution",
                [artifact["path"]],
                metrics={
                    "mle_private_grader_score": existing.get("score"),
                    "private_grader_outcome": existing.get("outcome"),
                },
                artifacts=[artifact],
                confidence=1.0,
            )
        payload, score = validated_payload()
        if output.exists():
            _require(
                output.is_file() and not output.is_symlink() and _sha256(output) == result_hash,
                "incomplete private grader registration has conflicting output",
            )
            write_ledger(payload, score)
            remove_registration_lock()
        else:
            _require(
                not lock_path.exists(),
                "private grader registration is already in progress",
            )
            try:
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError as exc:
                raise SiimEvidenceError("private grader registration is already in progress") from exc
            try:
                os.write(
                    descriptor,
                    json.dumps(
                        {
                            "pid": os.getpid(),
                            "run_id": run.run_id,
                            "result_sha256": result_hash,
                        },
                        separators=(",", ":"),
                    ).encode("utf-8"),
                )
                os.close(descriptor)
                descriptor = -1
                _copy_file(source, output)
                write_ledger(payload, score)
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
                try:
                    remove_registration_lock()
                except OSError:
                    pass
        artifacts = [
            _artifact(self.run_dir / "private_grader.json", self.run_dir, kind="terminal_private_grader"),
            _artifact(ledger_path, self.run_dir, kind="private_grader_ledger"),
        ]
        return AgentResult(
            task.task_id,
            "Exactly one post-freeze private-grader result recorded",
            [item["path"] for item in artifacts],
            metrics={
                "mle_private_grader_score": score,
                "private_grader_outcome": (
                    "failed_closed"
                    if str(payload.get("status") or "").lower() == "failed_closed"
                    else "scored"
                ),
            },
            artifacts=artifacts,
            confidence=1.0,
        )

    def _validate_deliverable(self, name: str, path: Path) -> None:
        _require(path.is_file() and path.stat().st_size > 0, f"deliverable is missing or empty: {name}")
        if name.endswith(".pdf"):
            _require(path.read_bytes()[:5] == b"%PDF-", "report deliverable is not a PDF")
        elif name.endswith("results.csv"):
            fields, rows = self._csv_rows(path)
            _require(fields[:2] == ["image_name", "target"], "results CSV schema is invalid")
            _require(len(rows) == EXPECTED_DATASET["test_images"], "results CSV row count mismatch")
            _, submission_rows = self._csv_rows(self.run_dir / "submission.csv")
            _require(
                [row["image_name"] for row in rows] == [row["image_name"] for row in submission_rows],
                "results CSV order mismatch",
            )
        elif name.endswith(".zip"):
            _require(zipfile.is_zipfile(path), f"invalid ZIP deliverable: {name}")
            with zipfile.ZipFile(path) as archive:
                members = [member for member in archive.namelist() if not member.endswith("/")]
                _require(bool(members), f"empty ZIP deliverable: {name}")
                if name.endswith("code.zip"):
                    _require(
                        any(member.lower().endswith(".py") for member in members), "code ZIP contains no Python source"
                    )
                if name.endswith("evidence.zip"):
                    required = {"metrics.json", "review.json", "claim_audit.json"}
                    basenames = {Path(member).name for member in members}
                    _require(required.issubset(basenames), "evidence ZIP is missing reviewed evidence")

    def delivery(self, task: AgentTask, _handoff: HandoffEnvelope, run: SupervisorRun) -> AgentResult:
        self._verify_freeze(run.run_id)
        ledger = _read_json(self.run_dir / "private_grader_ledger.json")
        _same_run(ledger, run.run_id, "private grader ledger")
        _require(int(ledger.get("execution_count") or 0) == 1, "delivery requires exactly one private grader execution")
        _require(int(ledger.get("job_id") or 0) == self.binding.job_id, "private grader ledger job binding changed")
        _require(
            ledger.get("credential_profile") == self.binding.credential_profile,
            "private grader ledger credential profile changed",
        )
        claim_source = self.source("claim_audit.json")
        claim = _read_json(claim_source)
        _same_run(claim, run.run_id, "claim audit")
        _require(_status_passed(claim), "Claim Audit is not passed")
        checks = claim.get("checks") if isinstance(claim.get("checks"), dict) else {}
        for name in (
            "no_public_leaderboard_claim",
            "no_official_medal_claim",
            "no_clinical_diagnosis_claim",
            "private_grader_not_used_for_tuning",
            "candidate_hashes_unchanged",
            "official_submission_not_executed",
        ):
            _require(checks.get(name) is True, f"Claim Audit check failed: {name}")
        html_source = self.source("deliverables/research_report.html")
        _require(html_source.stat().st_size > 0, "HTML report is empty")
        html_path = _copy_file(html_source, self.run_dir / "research_report.html")
        claim_path = _copy_file(claim_source, self.run_dir / "claim_audit.json")
        copied = [html_path, claim_path]
        for name in DELIVERABLE_NAMES:
            source = self.source(f"deliverables/{name}")
            self._validate_deliverable(name, source)
            copied.append(_copy_file(source, self.run_dir / name))
        delivery_path = self.run_dir / "deliverables.json"
        _atomic_json(
            delivery_path,
            {
                "schema": "evomind.siim.deliverables.v1",
                "run_id": run.run_id,
                "status": "ready",
                "files": [
                    {
                        **_artifact(self.run_dir / name, self.run_dir, kind="user_download"),
                        "name": name,
                        "download_url": f"/api/multi-agent/runs/{run.run_id}/download/{name}",
                    }
                    for name in DELIVERABLE_NAMES
                ],
                "official_submission": "forbidden",
                "clinical_use": "not_claimed",
                "created_at": _now(),
            },
        )
        copied.append(delivery_path)
        manifest_path = self.run_dir / "artifact_manifest.json"
        _atomic_json(
            manifest_path,
            {
                "schema": "evomind.siim.artifact_manifest.v1",
                "run_id": run.run_id,
                "task_id": TASK_ID,
                "status": "verified",
                "artifacts": _manifest_entries(self.run_dir),
                "mutable_ledgers_excluded_from_hash_seal": sorted(MUTABLE_FILES),
                "private_grader_execution_count": 1,
                "official_submission": "forbidden",
                "generated_at": _now(),
            },
        )
        artifacts = [_artifact(path, self.run_dir, kind=path.name) for path in copied]
        artifacts.append(_artifact(manifest_path, self.run_dir, kind="artifact_manifest"))
        return AgentResult(
            task.task_id,
            "Claim Audit passed and four hash-bound downloads are ready",
            [item["path"] for item in artifacts],
            metrics={"mle_private_grader_score": ledger.get("score")},
            artifacts=artifacts,
            confidence=1.0,
        )

    def mapping(self) -> dict[str, Any]:
        return {
            "RequestAgent": self.request_setup,
            "HpcPreflightAgent": self.preflight,
            "DataAuditor": self.data_audit,
            "ResearchDesigner": self.design,
            "AblationAgent": self.ablation,
            "HpcTrainingAgent": self.training,
            "IndependentReviewer": self.review_freeze,
            "TerminalGrader": self.terminal_grader,
            "DeliveryAgent": self.delivery,
        }


def _validate_request(request: UserRequest) -> None:
    if request.task_type != "image_classification" or request.dataset != TASK_ID:
        raise ValueError("SIIM workflow received a different task or dataset")
    if request.compute_policy.backend != "hpc" or not request.compute_policy.remote_gpu_required:
        raise ValueError("SIIM workflow requires the hpc compute policy")
    if request.compute_policy.local_gpu_allowed:
        raise ValueError("SIIM workflow excludes local GPU compute")
    if request.submission_policy.official_submission != "forbidden":
        raise ValueError("SIIM workflow requires official_submission=forbidden")


def run_siim_hpc_research(
    workspace_root: str | Path,
    request: UserRequest,
    *,
    run_id: str,
    parent_run_id: str | None = None,
) -> SupervisorRun:
    root = Path(workspace_root).resolve()
    _validate_request(request)
    run = build_siim_hpc_run(request, run_id=run_id)
    local_run_dir = run_directory(root, run.run_id)
    lineage: dict[str, Any] | None = None
    preservation: dict[str, Any] | None = None
    if parent_run_id is not None:
        lineage, _parent_manifest, _requested_change, preservation = _prepare_child_lineage(
            root,
            child_run_id=run.run_id,
            parent_run_id=parent_run_id,
        )
        # Both payloads are validated above; their exact source bytes are copied
        # below so the lineage file hash remains stable across platforms.
        del _parent_manifest, _requested_change
    local_run_dir.parent.mkdir(parents=True, exist_ok=True)
    local_run_dir.mkdir(exist_ok=False)
    if lineage is not None:
        control_dir = root / "workspace" / "siim_evolution_control" / run.run_id
        _copy_file(
            control_dir / "constraint_supersession.json",
            local_run_dir / "constraint_supersession.json",
        )
        _copy_file(
            control_dir / "parent_immutable_manifest.json",
            local_run_dir / "parent_immutable_manifest.json",
        )
        _copy_file(
            control_dir / "requested_change.json",
            local_run_dir / "requested_change.json",
        )
        _atomic_json(local_run_dir / "lineage.json", lineage)
        _atomic_json(
            local_run_dir / "parent_preservation.json",
            {
                "schema": "evomind.siim.parent_preservation.v1",
                "created_at": _now(),
                "run_id": run.run_id,
                "parent_run_id": parent_run_id,
                "initial": preservation,
                "current": preservation,
                "unchanged": True,
            },
        )
        run.gates.update(
            {
                "parent_run_id": parent_run_id,
                "lineage_sha256": _sha256(local_run_dir / "lineage.json"),
                "parent_preservation": "required",
            }
        )
    _atomic_json(local_run_dir / "request.json", request.to_dict())
    store = _snapshot_store(local_run_dir)
    store.append_message(run, sender="user", receiver="ExecutiveSupervisor", content=request.objective)
    write_current_run_pointer(root, task_id=TASK_ID, run=run, run_dir=local_run_dir)
    executors = SiimHpcExecutors(workspace_root=root, run_dir=local_run_dir, request=request)
    supervisor = MultiAgentSupervisor(run, store, executors.mapping())
    try:
        result = supervisor.run_until_blocked()
    finally:
        if lineage is not None:
            current = _verify_child_lineage(root, local_run_dir, run.run_id)
            preservation_payload = _read_json(local_run_dir / "parent_preservation.json")
            preservation_payload["current"] = current
            preservation_payload["unchanged"] = bool(current and current.get("passed"))
            preservation_payload["verified_at"] = _now()
            _atomic_json(local_run_dir / "parent_preservation.json", preservation_payload)
        write_current_run_pointer(root, task_id=TASK_ID, run=run, run_dir=local_run_dir)
    return result


def resume_siim_hpc_research(workspace_root: str | Path, run_id: str) -> SupervisorRun:
    root = Path(workspace_root).resolve()
    _safe_run_id(run_id)
    local_run_dir = run_directory(root, run_id)
    _verify_child_lineage(root, local_run_dir, run_id)
    store = _snapshot_store(local_run_dir)
    run = store.load()
    request = UserRequest.from_dict(_read_json(local_run_dir / "request.json"))
    _validate_request(request)
    if run.status == "completed":
        return run
    executors = SiimHpcExecutors(workspace_root=root, run_dir=local_run_dir, request=request)
    supervisor = MultiAgentSupervisor(run, store, executors.mapping())
    supervisor.resume(retry_failed=run.status == "needs_continuation")
    try:
        result = supervisor.run_until_blocked()
    finally:
        write_current_run_pointer(root, task_id=TASK_ID, run=run, run_dir=local_run_dir)
    return result


def stage_siim_ingress(
    workspace_root: str | Path,
    run_id: str,
    files: Mapping[str, str | Path],
) -> list[dict[str, Any]]:
    """Copy collected evidence into the run-scoped ingress without interpreting it.

    This helper is for the HPC watcher/collector.  Validation remains the
    responsibility of the corresponding DAG node on the next resume.
    """
    root = Path(workspace_root).resolve()
    _safe_run_id(run_id)
    run_dir = run_directory(root, run_id)
    records: list[dict[str, Any]] = []
    for relative, raw_source in files.items():
        target_rel = Path(relative)
        if target_rel.is_absolute() or ".." in target_rel.parts:
            raise ValueError("invalid SIIM ingress target")
        source = Path(raw_source).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = _copy_file(source, run_dir / "ingress" / target_rel)
        records.append(_artifact(destination, run_dir, kind="ingress"))
    return records


__all__ = [
    "ABLATION_PROFILES",
    "ABLATION_SEEDS",
    "DELIVERABLE_NAMES",
    "FORMAL_SEEDS",
    "FROZEN_FILES",
    "SiimEvidenceError",
    "SiimHpcExecutors",
    "build_siim_hpc_run",
    "resume_siim_hpc_research",
    "run_siim_hpc_research",
    "stage_siim_ingress",
]
