from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from research_os.hpc_policy import validate_remote_workspace


@dataclass(slots=True)
class RetryPolicy:
    max_retries: int = 2
    retry_delay_seconds: int = 60
    backoff_multiplier: float = 2.0
    pause_on_final_failure: bool = True
    generate_failure_review: bool = True

    def should_retry(self, attempt: int) -> bool:
        return attempt <= self.max_retries

    def is_final_failure(self, attempt: int) -> bool:
        return attempt > self.max_retries


@dataclass(slots=True)
class SubmissionGate:
    gate_id: str
    status: str = "pending"
    required_checks: list[str] = field(default_factory=lambda: [
        "submission_schema_valid",
        "no_missing_predictions",
        "train_test_features_match",
        "submission_audit_passed",
        "human_approval",
    ])
    audit_results: dict[str, Any] = field(default_factory=dict)
    approved_by: str | None = None
    approved_at: str | None = None

    def audit_passed(self) -> bool:
        return all(
            self.audit_results.get(check, False)
            for check in self.required_checks
            if check != "human_approval"
        )

    def approve(self, reviewer: str) -> None:
        self.status = "approved"
        self.approved_by = reviewer
        self.approved_at = datetime.now().isoformat(timespec="seconds")


@dataclass(slots=True)
class FailureReview:
    review_id: str
    task_id: str
    run_id: str
    attempt_count: int
    failure_reason: str
    gap_analysis: str
    next_strategy: str
    evidence_artifacts: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass(slots=True)
class JobManifest:
    """Standardized GPU/HPC job manifest.

    Every job dispatched through the workstation must include all required fields.
    """
    task_id: str
    run_id: str
    agent_id: str
    gate_id: str
    template_id: str
    resource_request: dict[str, Any]
    remote_workspace: str
    command_template: str
    log_path: str
    artifact_pullback: str
    timeout: int
    retry_policy: RetryPolicy
    submission_gate: SubmissionGate
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    status: str = "manifest_prepared"
    attempt: int = 0
    job_id: str | None = None
    dispatch_receipt: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class JobManifestBuilder:
    """Builder for standardized GPU/HPC job manifests."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root

    def build(
        self,
        task_id: str,
        run_id: str,
        agent_id: str,
        template_id: str,
        *,
        remote_workspace: str,
        command_template: str = "",
        log_path: str = "",
        artifact_pullback: str = "",
        timeout: int = 7200,
        resource_request: dict[str, Any] | None = None,
        max_retries: int = 2,
    ) -> JobManifest:
        remote_workspace = validate_remote_workspace(remote_workspace)
        if not log_path:
            log_path = f"workspace/gpu/{task_id}/{run_id}/remote_stdout.log"
        if not artifact_pullback:
            artifact_pullback = f"workspace/gpu/{task_id}/{run_id}"

        gate_id = f"gate_{uuid4().hex[:10]}"

        return JobManifest(
            task_id=task_id,
            run_id=run_id,
            agent_id=agent_id,
            gate_id=gate_id,
            template_id=template_id,
            resource_request=resource_request or {
                "gpu_count": 1,
                "gpu_type": "A800",
                "cpu_cores": 8,
                "memory_gb": 32,
                "disk_gb": 100,
            },
            remote_workspace=remote_workspace,
            command_template=command_template,
            log_path=log_path,
            artifact_pullback=artifact_pullback,
            timeout=timeout,
            retry_policy=RetryPolicy(max_retries=max_retries),
            submission_gate=SubmissionGate(gate_id=gate_id),
            metadata={
                "schema": "academic_research_os.job_manifest.v1",
                "builder": "JobManifestBuilder",
            },
        )

    def write(self, manifest: JobManifest, output_dir: Path) -> Path:
        self._validate_dispatch_state(manifest)
        data = {
            "task_id": manifest.task_id,
            "run_id": manifest.run_id,
            "agent_id": manifest.agent_id,
            "gate_id": manifest.gate_id,
            "template_id": manifest.template_id,
            "resource_request": manifest.resource_request,
            "remote_workspace": manifest.remote_workspace,
            "command_template": manifest.command_template,
            "log_path": manifest.log_path,
            "artifact_pullback": manifest.artifact_pullback,
            "timeout": manifest.timeout,
            "retry_policy": {
                "max_retries": manifest.retry_policy.max_retries,
                "retry_delay_seconds": manifest.retry_policy.retry_delay_seconds,
                "backoff_multiplier": manifest.retry_policy.backoff_multiplier,
                "pause_on_final_failure": manifest.retry_policy.pause_on_final_failure,
                "generate_failure_review": manifest.retry_policy.generate_failure_review,
            },
            "submission_gate": {
                "gate_id": manifest.submission_gate.gate_id,
                "status": manifest.submission_gate.status,
                "required_checks": manifest.submission_gate.required_checks,
            },
            "created_at": manifest.created_at,
            "status": manifest.status,
            "attempt": manifest.attempt,
            "job_id": manifest.job_id,
            "dispatch_receipt": manifest.dispatch_receipt,
            "metadata": manifest.metadata,
        }
        path = output_dir / "job_manifest.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def mark_queued(
        manifest: JobManifest,
        *,
        remote_job_id: str,
        dispatch_receipt: dict[str, Any],
    ) -> JobManifest:
        job_id = str(remote_job_id or "").strip()
        if not job_id:
            raise ValueError("remote_job_id is required before a manifest can be queued")
        if not isinstance(dispatch_receipt, dict) or not dispatch_receipt:
            raise ValueError("a non-empty dispatch_receipt is required before a manifest can be queued")
        receipt_job_id = str(
            dispatch_receipt.get("remote_job_id") or dispatch_receipt.get("job_id") or ""
        ).strip()
        if receipt_job_id != job_id:
            raise ValueError("dispatch_receipt job id does not match remote_job_id")
        manifest.job_id = job_id
        manifest.dispatch_receipt = dict(dispatch_receipt)
        manifest.status = "queued"
        return manifest

    @staticmethod
    def _validate_dispatch_state(manifest: JobManifest) -> None:
        if manifest.status != "queued":
            return
        if not str(manifest.job_id or "").strip() or not manifest.dispatch_receipt:
            raise ValueError("queued manifest requires remote job id and dispatch receipt")

    @staticmethod
    def generate_failure_review(
        task_id: str,
        run_id: str,
        attempt_count: int,
        reason: str,
        gap_analysis: str = "",
        next_strategy: str = "",
        artifacts: list[str] | None = None,
    ) -> FailureReview:
        return FailureReview(
            review_id=f"fr_{uuid4().hex[:10]}",
            task_id=task_id,
            run_id=run_id,
            attempt_count=attempt_count,
            failure_reason=reason,
            gap_analysis=gap_analysis,
            next_strategy=next_strategy,
            evidence_artifacts=artifacts or [],
        )

    @staticmethod
    def write_failure_review(review: FailureReview, output_dir: Path) -> Path:
        data = {
            "review_id": review.review_id,
            "task_id": review.task_id,
            "run_id": review.run_id,
            "attempt_count": review.attempt_count,
            "failure_reason": review.failure_reason,
            "gap_analysis": review.gap_analysis,
            "next_strategy": review.next_strategy,
            "evidence_artifacts": review.evidence_artifacts,
            "created_at": review.created_at,
        }
        path = output_dir / "failure_review.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def compute_artifact_sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
