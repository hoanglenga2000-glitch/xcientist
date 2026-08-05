#!/usr/bin/env python3
"""Persistently finish one governed SIIM Run after its bound GPU gate launches it.

The existing gate watcher owns the initial HOLD -> GO transition.  This process
starts no second initial watcher.  It waits for the immutable launch record, then
drives collection, freezing, the exactly-once terminal grader, Claim Audit, and
delivery.  Every completed stage is reconciled from hash-bound evidence so a
local restart does not repeat aggregation, grading, or delivery construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from research_agent_workstation.server.core.gpu_credentials import (  # noqa: E402
    CredentialError,
    RetryableTransportError,
)
from scripts import build_siim_claim_audit as claim_audit  # noqa: E402
from scripts import build_siim_delivery as delivery_builder  # noqa: E402
from scripts import build_siim_workflow_ingress as ingress  # noqa: E402
from scripts import manage_siim_job89508_campaign as campaign  # noqa: E402
from scripts import run_siim_private_grader_once as private_grader  # noqa: E402

SCHEMA = campaign.BINDING.schema("end_to_end_supervisor")
EVENT_SCHEMA = campaign.BINDING.schema("end_to_end_event")
DEFAULT_RUN_ID = campaign.DEFAULT_RUN_ID
DOWNLOAD_NAMES = tuple(delivery_builder.DOWNLOAD_NAMES)
ACTIVE_REMOTE_STATES = {
    "preflight",
    "ablation_training",
    "batch_probing",
    "formal_training",
}
TERMINAL_FAILURE_STATES = {"failed", "cancelled", "aborted"}
REQUIRED_FINAL_TASKS = (
    "request_setup",
    "hpc_data_preflight",
    "data_audit",
    "research_design",
    "preprocessing_ablation",
    "full_training",
    "independent_review_freeze",
    "terminal_private_grader",
    "claim_audit_delivery",
)


class EndToEndSupervisorError(RuntimeError):
    """A deterministic evidence or lifecycle contract failed closed."""


class EndToEndApi(Protocol):
    def final_complete(self) -> bool: ...

    def launch_ready(self) -> bool: ...

    def ensure_preflight(self) -> Mapping[str, Any]: ...

    def campaign_status(self) -> Mapping[str, Any]: ...

    def resume_campaign_if_ready(self, *, sample_interval_seconds: int) -> Mapping[str, Any]: ...

    def ensure_collection(self) -> Mapping[str, Any]: ...

    def ensure_aggregation(self) -> Mapping[str, Any]: ...

    def ensure_campaign_ingress(self) -> Mapping[str, Any]: ...

    def ensure_private_grader_execution(self) -> Mapping[str, Any]: ...

    def ensure_grader_ingress(self, execution: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def ensure_claim_audit(self) -> Mapping[str, Any]: ...

    def ensure_delivery_build(self) -> Mapping[str, Any]: ...

    def ensure_delivery_ingress(
        self,
        claim: Mapping[str, Any],
        delivery: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class SupervisorPaths:
    project_root: Path
    run_id: str
    campaign_dir: Path
    run_dir: Path
    candidate_dir: Path
    collected_root: Path
    delivery_dir: Path
    state: Path
    events: Path
    lock: Path

    @classmethod
    def build(cls, project_root: Path, run_id: str) -> "SupervisorPaths":
        root = project_root.expanduser().resolve()
        selected = ingress.validate_run_id(run_id)
        campaign_dir = root / "workspace" / "hpc" / f"{campaign.JOB_TAG}_siim_campaign" / selected
        return cls(
            project_root=root,
            run_id=selected,
            campaign_dir=campaign_dir,
            run_dir=root / "workspace" / "evomind_runs" / selected,
            candidate_dir=root / "workspace" / f"siim_{campaign.JOB_TAG}" / "candidates" / selected,
            collected_root=root / "workspace" / "hpc" / "mlebench_remote_ops" / "collected",
            delivery_dir=root / "workspace" / f"siim_{campaign.JOB_TAG}" / "deliveries" / selected,
            state=campaign_dir / "end_to_end_supervisor.json",
            events=campaign_dir / "end_to_end_supervisor.jsonl",
            lock=campaign_dir / ".end_to_end_supervisor.lock",
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise EndToEndSupervisorError(f"{label} is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EndToEndSupervisorError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise EndToEndSupervisorError(f"{label} is not a JSON object")
    return payload


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def exclusive_process_lock(path: Path) -> Iterator[None]:
    """Hold a non-blocking one-byte lock for the supervisor lifetime."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    acquired = False
    try:
        if path.stat().st_size == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as exc:
            raise EndToEndSupervisorError("a SIIM end-to-end supervisor is already running") from exc
        yield
    finally:
        try:
            if acquired:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _safe_relative(root: Path, raw: Any, label: str) -> Path:
    relative = Path(str(raw or ""))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise EndToEndSupervisorError(f"unsafe {label} path")
    resolved = (root / relative).resolve()
    if root.resolve() not in resolved.parents:
        raise EndToEndSupervisorError(f"{label} escaped its evidence root")
    return resolved


def _validate_record(path: Path, record: Mapping[str, Any], label: str) -> None:
    if not path.is_file():
        raise EndToEndSupervisorError(f"{label} is missing")
    if int(record.get("bytes") or -1) != path.stat().st_size:
        raise EndToEndSupervisorError(f"{label} byte count changed")
    expected = str(record.get("sha256") or "").lower()
    if len(expected) != 64 or sha256_file(path) != expected:
        raise EndToEndSupervisorError(f"{label} SHA-256 changed")


def _workflow_task_status(run_dir: Path, task_id: str) -> str:
    run = read_json(run_dir / "run.json", "EvoMind Run")
    tasks = run.get("tasks") if isinstance(run.get("tasks"), dict) else {}
    task = tasks.get(task_id) if isinstance(tasks.get(task_id), dict) else {}
    return str(task.get("status") or "")


def ingress_stage_verified(paths: SupervisorPaths, stage: str) -> bool:
    manifest_path = paths.run_dir / "ingress" / "workflow_ingress_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = read_json(manifest_path, "workflow ingress manifest")
    if manifest.get("schema") != "evomind.siim.workflow_ingress_manifest.v1":
        raise EndToEndSupervisorError("workflow ingress manifest schema changed")
    if manifest.get("run_id") != paths.run_id:
        raise EndToEndSupervisorError("workflow ingress manifest Run changed")
    stages = manifest.get("stages") if isinstance(manifest.get("stages"), dict) else {}
    record = stages.get(stage)
    if record is None:
        return False
    if not isinstance(record, dict) or record.get("status") != "verified_and_staged":
        raise EndToEndSupervisorError(f"workflow ingress stage {stage} is invalid")
    artifacts = record.get("staged_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise EndToEndSupervisorError(f"workflow ingress stage {stage} is empty")
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            raise EndToEndSupervisorError(f"workflow ingress stage {stage} record {index} is invalid")
        target = _safe_relative(paths.run_dir / "ingress", artifact.get("path"), "ingress")
        _validate_record(target, artifact, f"workflow ingress {stage} artifact {index}")
    return True


def validate_launch(paths: SupervisorPaths) -> bool:
    path = paths.campaign_dir / "launch.json"
    if not path.is_file():
        return False
    payload = read_json(path, "initial campaign launch")
    expected = {
        "schema": campaign.BINDING.schema("launch"),
        "run_id": paths.run_id,
        "job_id": campaign.HPC_JOB_ID,
        "credential_profile": campaign.CREDENTIAL_PROFILE,
        "status": "started",
        "signals_sent": 0,
        "other_processes_modified": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise EndToEndSupervisorError(f"initial campaign launch contract changed: {key}")
    if payload.get("action") not in {"new_supervisor_started", "existing_supervisor_reused"}:
        raise EndToEndSupervisorError("initial campaign launch action is invalid")
    if int(payload.get("supervisor_pid") or 0) <= 0:
        raise EndToEndSupervisorError("initial campaign launch PID is missing")
    launch_gate = payload.get("gate") if isinstance(payload.get("gate"), dict) else {}
    if launch_gate.get("passed") is not True or int(launch_gate.get("sample_count") or 0) < 5:
        raise EndToEndSupervisorError("initial campaign launch did not bind a clean five-sample GO gate")
    return True


def validate_collection(paths: SupervisorPaths) -> bool:
    path = paths.campaign_dir / "collection.json"
    if not path.is_file():
        return False
    payload = read_json(path, "campaign collection")
    expected = {
        "schema": campaign.BINDING.schema("collection"),
        "run_id": paths.run_id,
        "job_id": campaign.HPC_JOB_ID,
        "credential_profile": campaign.CREDENTIAL_PROFILE,
        "status": "collected",
        "signals_sent": 0,
        "other_processes_modified": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise EndToEndSupervisorError(f"campaign collection contract changed: {key}")
    plan = read_json(paths.campaign_dir / "campaign_plan.json", "campaign plan")
    expected_ids = {str(item["run_id"]) for item in plan.get("formal_runs") or []}
    reports = payload.get("seed_collections")
    if not isinstance(reports, list) or len(reports) != len(expected_ids) or len(reports) != 3:
        raise EndToEndSupervisorError("campaign collection does not contain three formal seeds")
    actual_ids: set[str] = set()
    for report_index, report in enumerate(reports):
        if not isinstance(report, dict):
            raise EndToEndSupervisorError("campaign collection contains an invalid seed report")
        formal_id = str(report.get("run_id") or "")
        actual_ids.add(formal_id)
        if report.get("schema") != "evomind.mlebench_remote_ops.collection.v1" or report.get("passed") is not True:
            raise EndToEndSupervisorError(f"formal collection {formal_id or report_index} did not pass")
        records = report.get("files")
        if not isinstance(records, list) or not records or int(report.get("file_count") or -1) != len(records):
            raise EndToEndSupervisorError(f"formal collection {formal_id} is incomplete")
        for record_index, record in enumerate(records):
            if not isinstance(record, dict):
                raise EndToEndSupervisorError(f"formal collection {formal_id} record is invalid")
            local = Path(str(record.get("local") or "")).resolve()
            if paths.collected_root.resolve() not in local.parents:
                raise EndToEndSupervisorError("formal collection file escaped the collected root")
            _validate_record(local, record, f"formal collection {formal_id} artifact {record_index}")
    if actual_ids != expected_ids:
        raise EndToEndSupervisorError("campaign collection formal Run IDs changed")
    evidence = Path(str(payload.get("remote_evidence_root") or "")).resolve()
    if paths.campaign_dir.resolve() not in evidence.parents or not evidence.is_dir():
        raise EndToEndSupervisorError("campaign remote evidence root is invalid")
    return True


def validate_delivery_bundle(paths: SupervisorPaths) -> bool:
    if not paths.delivery_dir.exists():
        return False
    if not paths.delivery_dir.is_dir():
        raise EndToEndSupervisorError("delivery output is not a directory")
    manifest = read_json(paths.delivery_dir / "artifact_manifest.json", "delivery manifest")
    reproducibility = (
        manifest.get("reproducibility")
        if isinstance(manifest.get("reproducibility"), dict)
        else {}
    )
    if (
        manifest.get("schema") != "evomind.siim.delivery_bundle_manifest.v1"
        or manifest.get("run_id") != paths.run_id
        or manifest.get("status") != "verified"
        or manifest.get("all_sha256_bound") is not True
        or manifest.get("official_submission") != "forbidden"
        or int(manifest.get("private_grader_execution_count") or 0) != 1
        or int(reproducibility.get("pdf_pages") or 0) != delivery_builder.PDF_PAGE_COUNT
    ):
        raise EndToEndSupervisorError("delivery manifest contract changed")
    records = manifest.get("deliverables")
    if not isinstance(records, list):
        raise EndToEndSupervisorError("delivery manifest artifact list is invalid")
    indexed = {str(item.get("path") or ""): item for item in records if isinstance(item, dict)}
    if set(indexed) != set(DOWNLOAD_NAMES):
        raise EndToEndSupervisorError("delivery manifest does not bind exactly four downloads")
    for name in DOWNLOAD_NAMES:
        _validate_record(paths.delivery_dir / name, indexed[name], f"delivery {name}")
    html_record = manifest.get("report_html")
    if not isinstance(html_record, dict):
        raise EndToEndSupervisorError("delivery HTML record is missing")
    _validate_record(paths.delivery_dir / "research_report.html", html_record, "delivery HTML report")
    try:
        html_text = (paths.delivery_dir / "research_report.html").read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise EndToEndSupervisorError("delivery HTML report is unreadable") from exc
    for marker in ("EvoMind 四轮进化", "R1", "R2", "R3", "R4"):
        if marker not in html_text:
            raise EndToEndSupervisorError(f"delivery HTML report omits evolution marker: {marker}")
    secret_scan = manifest.get("secret_scan") if isinstance(manifest.get("secret_scan"), dict) else {}
    if secret_scan.get("status") != "passed":
        raise EndToEndSupervisorError("delivery secret scan did not pass")
    qa = read_json(paths.delivery_dir / "qa" / "delivery-qa.json", "delivery QA")
    pdf_qa = qa.get("pdf") if isinstance(qa.get("pdf"), dict) else {}
    if (
        qa.get("run_id") != paths.run_id
        or qa.get("status") != "passed"
        or int(qa.get("download_file_count") or 0) != 4
        or tuple(qa.get("download_names") or ()) != DOWNLOAD_NAMES
        or qa.get("same_run_verified") is not True
        or qa.get("official_submission_executed") is not False
        or int(pdf_qa.get("page_count") or 0) != delivery_builder.PDF_PAGE_COUNT
    ):
        raise EndToEndSupervisorError("delivery QA contract changed")
    return True


def validate_workflow_grader(paths: SupervisorPaths) -> bool:
    result_path = paths.run_dir / "private_grader.json"
    ledger_path = paths.run_dir / "private_grader_ledger.json"
    freeze_path = paths.run_dir / "candidate_freeze.json"
    if not result_path.exists() and not ledger_path.exists():
        return False
    if not all(path.is_file() and not path.is_symlink() for path in (result_path, ledger_path, freeze_path)):
        raise EndToEndSupervisorError("workflow private-grader evidence is partial or unsafe")
    result = read_json(result_path, "workflow private grader")
    ledger = read_json(ledger_path, "workflow private grader ledger")
    freeze = read_json(freeze_path, "workflow candidate freeze")
    for label, payload in (
        ("private grader", result),
        ("private grader ledger", ledger),
        ("candidate freeze", freeze),
    ):
        if payload.get("run_id") != paths.run_id:
            raise EndToEndSupervisorError(f"workflow {label} Run changed")
    freeze_sha = sha256_file(freeze_path)
    if (
        freeze.get("status") != "frozen_before_private_grader"
        or freeze.get("tuning_closed") is not True
        or int(result.get("execution_index") or result.get("execution_count") or 0) != 1
        or int(ledger.get("execution_count") or 0) != 1
        or int(ledger.get("job_id") or 0) != campaign.HPC_JOB_ID
        or ledger.get("credential_profile") != campaign.CREDENTIAL_PROFILE
        or result.get("candidate_freeze_sha256") != freeze_sha
        or ledger.get("candidate_freeze_sha256") != freeze_sha
        or ledger.get("result_sha256") != sha256_file(result_path)
        or result.get("executed_after_freeze") is not True
        or result.get("feedback_used_for_tuning") is not False
        or result.get("official_submission_executed") is not False
    ):
        raise EndToEndSupervisorError("workflow private-grader exactly-once contract changed")
    return True


def validate_final_run(paths: SupervisorPaths) -> bool:
    run_path = paths.run_dir / "run.json"
    manifest_path = paths.run_dir / "artifact_manifest.json"
    if not run_path.is_file() or not manifest_path.is_file():
        return False
    run = read_json(run_path, "EvoMind Run")
    if run.get("run_id") != paths.run_id:
        raise EndToEndSupervisorError("final EvoMind Run ID changed")
    if run.get("status") != "completed":
        return False
    for task_id in REQUIRED_FINAL_TASKS:
        if _workflow_task_status(paths.run_dir, task_id) != "completed":
            raise EndToEndSupervisorError(f"final workflow task is not completed: {task_id}")
    if not validate_collection(paths):
        raise EndToEndSupervisorError("final Run has no verified campaign collection")
    ingress.validate_candidate_manifest(paths.candidate_dir, paths.run_id)
    if not validate_workflow_grader(paths):
        raise EndToEndSupervisorError("final Run has no verified exactly-once private grader")
    if not validate_delivery_bundle(paths):
        raise EndToEndSupervisorError("final Run has no verified delivery bundle")
    for stage in ("campaign", "independent_review", "private_grader", "delivery"):
        if not ingress_stage_verified(paths, stage):
            raise EndToEndSupervisorError(f"final Run ingress stage is not verified: {stage}")
    manifest = read_json(manifest_path, "final artifact manifest")
    if (
        manifest.get("schema") != "evomind.siim.artifact_manifest.v1"
        or manifest.get("run_id") != paths.run_id
        or manifest.get("status") != "verified"
        or manifest.get("official_submission") != "forbidden"
        or int(manifest.get("private_grader_execution_count") or 0) != 1
    ):
        raise EndToEndSupervisorError("final artifact manifest contract changed")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise EndToEndSupervisorError("final artifact manifest is empty")
    indexed = {str(item.get("path") or ""): item for item in artifacts if isinstance(item, dict)}
    for name in DOWNLOAD_NAMES:
        record = indexed.get(name)
        if not isinstance(record, dict):
            raise EndToEndSupervisorError(f"final artifact manifest omits {name}")
        _validate_record(paths.run_dir / name, record, f"final download {name}")
    delivery_manifest = read_json(paths.delivery_dir / "artifact_manifest.json", "delivery manifest")
    source_records = delivery_manifest.get("source_evidence")
    if not isinstance(source_records, list):
        raise EndToEndSupervisorError("delivery source evidence is missing")
    source_index = {
        str(record.get("path") or ""): record
        for record in source_records
        if isinstance(record, dict)
    }
    for name in ("candidate_freeze.json", "review.json", "private_grader.json", "claim_audit.json"):
        record = source_index.get(name)
        if not isinstance(record, dict):
            raise EndToEndSupervisorError(f"delivery source evidence omits {name}")
        path = paths.run_dir / name
        if name == "claim_audit.json" and (paths.run_dir / "claim_audit_source.json").is_file():
            path = paths.run_dir / "claim_audit_source.json"
        _validate_record(path, record, f"delivery source evidence {name}")
    return True


class ProductionApi:
    """Thin adapter over the already-tested SIIM campaign and ingress modules."""

    def __init__(self, paths: SupervisorPaths, *, require_profile: bool = True):
        self.paths = paths
        self.require_profile = require_profile

    def _require_profile(self) -> None:
        if (
            self.require_profile
            and os.environ.get("EVOMIND_SIIM_HPC_JOB_ID", "").strip()
            != str(campaign.HPC_JOB_ID)
        ):
            raise EndToEndSupervisorError(
                f"EVOMIND_SIIM_HPC_JOB_ID must be {campaign.HPC_JOB_ID}"
            )
        if (
            self.require_profile
            and os.environ.get("EVOMIND_HPC_CREDENTIAL_PROFILE", "").strip()
            != campaign.CREDENTIAL_PROFILE
        ):
            raise EndToEndSupervisorError(
                f"EVOMIND_HPC_CREDENTIAL_PROFILE must be {campaign.CREDENTIAL_PROFILE}"
            )

    def final_complete(self) -> bool:
        return validate_final_run(self.paths)

    def launch_ready(self) -> bool:
        return validate_launch(self.paths)

    def ensure_preflight(self) -> Mapping[str, Any]:
        done = ingress_stage_verified(self.paths, "preflight")
        if done and _workflow_task_status(self.paths.run_dir, "hpc_data_preflight") == "completed":
            return {"status": "reused", "stage": "preflight"}
        return ingress.stage_preflight_ingress(self.paths.project_root, self.paths.run_id)

    def campaign_status(self) -> Mapping[str, Any]:
        self._require_profile()
        return campaign.status(self.paths.run_id)

    def resume_campaign_if_ready(self, *, sample_interval_seconds: int) -> Mapping[str, Any]:
        self._require_profile()
        existing = sorted(self.paths.campaign_dir.glob("continuation_launch_*.json"))
        sequence = len(existing) + 1
        gate_path = self.paths.campaign_dir / f"continuation_gpu_gate_{sequence:03d}.json"
        launch_path = self.paths.campaign_dir / f"continuation_launch_{sequence:03d}.json"
        gate = campaign.gate(
            self.paths.run_id,
            interval_seconds=sample_interval_seconds,
            output_path=gate_path,
            publish_default=False,
        )
        if gate.get("passed") is not True:
            return {
                "status": "hold",
                "hold_reasons": list(gate.get("hold_reasons") or []),
                "gate_path": str(gate_path),
                "signals_sent": 0,
                "other_processes_modified": False,
            }
        launch = campaign.launch(
            self.paths.run_id,
            max_gate_age=600,
            gate_path=gate_path,
            launch_record_path=launch_path,
        )
        return {**launch, "status": "launched", "continuation_sequence": sequence}

    def ensure_collection(self) -> Mapping[str, Any]:
        if validate_collection(self.paths):
            return {"status": "reused", "stage": "collection"}
        self._require_profile()
        result = campaign.collect(self.paths.run_id)
        if not validate_collection(self.paths):
            raise EndToEndSupervisorError("campaign collection did not become complete")
        return result

    def ensure_aggregation(self) -> Mapping[str, Any]:
        if self.paths.candidate_dir.exists():
            ingress.validate_candidate_manifest(self.paths.candidate_dir, self.paths.run_id)
            return {"status": "reused", "stage": "aggregation"}
        result = campaign.aggregate(self.paths.run_id)
        ingress.validate_candidate_manifest(self.paths.candidate_dir, self.paths.run_id)
        return result

    def ensure_campaign_ingress(self) -> Mapping[str, Any]:
        campaign_done = ingress_stage_verified(self.paths, "campaign")
        review_done = ingress_stage_verified(self.paths, "independent_review")
        task_done = _workflow_task_status(self.paths.run_dir, "independent_review_freeze") == "completed"
        if campaign_done and review_done and task_done:
            private_grader.prepare_candidate(self.paths.project_root, self.paths.run_id)
            return {"status": "reused", "stage": "campaign_and_review"}
        return ingress.stage_campaign_ingress(self.paths.project_root, self.paths.run_id)

    def _workflow_grader_verified(self) -> bool:
        return validate_workflow_grader(self.paths)

    def ensure_private_grader_execution(self) -> Mapping[str, Any]:
        grader_staged = ingress_stage_verified(self.paths, "private_grader")
        task_done = _workflow_task_status(self.paths.run_dir, "terminal_private_grader") == "completed"
        result_path = self.paths.run_dir / "private_grader.json"
        if grader_staged and task_done and self._workflow_grader_verified():
            return {
                "status": "reused",
                "stage": "private_grader_once",
                "result_path": str(result_path),
                "execution_count": 1,
                "signals_sent": 0,
                "other_processes_modified": False,
            }
        self._require_profile()
        execution = private_grader.run_private_grader_once(self.paths.project_root, self.paths.run_id)
        if int(execution.get("execution_count") or 0) != 1:
            raise EndToEndSupervisorError("private grader execution count is not exactly one")
        result = Path(str(execution.get("result_path") or ""))
        if not result.is_file():
            raise EndToEndSupervisorError("private grader did not produce a local result")
        return execution

    def ensure_grader_ingress(self, execution: Mapping[str, Any]) -> Mapping[str, Any]:
        grader_staged = ingress_stage_verified(self.paths, "private_grader")
        task_done = _workflow_task_status(self.paths.run_dir, "terminal_private_grader") == "completed"
        result_path = self.paths.run_dir / "private_grader.json"
        if grader_staged and task_done and self._workflow_grader_verified():
            return {
                "status": "reused",
                "stage": "grader_ingress",
                "result_path": str(result_path),
                "execution_count": 1,
                "signals_sent": 0,
                "other_processes_modified": False,
            }
        source = Path(str(execution.get("result_path") or execution.get("ingress_path") or ""))
        if not source.is_file():
            raise EndToEndSupervisorError("private grader result is unavailable for ingress")
        staged = ingress.stage_private_grader_ingress(
            self.paths.project_root,
            self.paths.run_id,
            source,
        )
        return {"status": "staged", "stage": "grader_ingress", "ingress": staged}

    def ensure_claim_audit(self) -> Mapping[str, Any]:
        return claim_audit.build_claim_audit(
            self.paths.run_dir,
            output=self.paths.run_dir / "claim_audit_source.json",
        )

    def ensure_delivery_build(self) -> Mapping[str, Any]:
        if validate_delivery_bundle(self.paths):
            return {
                "status": "reused",
                "run_id": self.paths.run_id,
                "output_dir": str(self.paths.delivery_dir),
                "signals_sent": 0,
                "other_processes_modified": False,
            }
        return delivery_builder.build_delivery(
            self.paths.run_dir,
            self.paths.delivery_dir,
            repo_root=self.paths.project_root,
        )

    def ensure_delivery_ingress(
        self,
        claim: Mapping[str, Any],
        delivery: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if self.final_complete():
            return {
                "status": "completed",
                "run_id": self.paths.run_id,
                "stage": "delivery_ingress",
                "signals_sent": 0,
                "other_processes_modified": False,
            }
        claim_path = Path(str(claim.get("claim_audit") or ""))
        output_dir = Path(str(delivery.get("output_dir") or ""))
        if not claim_path.is_file() or output_dir.resolve() != self.paths.delivery_dir.resolve():
            raise EndToEndSupervisorError("delivery ingress source evidence changed")
        staged = ingress.stage_delivery_ingress(
            self.paths.project_root,
            self.paths.run_id,
            claim_path,
            output_dir,
        )
        if not self.final_complete():
            raise EndToEndSupervisorError("delivery ingress did not complete the EvoMind Run")
        return {"status": "completed", "stage": "delivery_ingress", "ingress": staged}


def _public_state(
    paths: SupervisorPaths,
    *,
    status: str,
    phase: str,
    iteration: int,
    started_at: str,
    detail: str = "",
    remote_status: str = "",
    transient_failures: int = 0,
    completed_stages: Iterable[str] = (),
    evidence_sha256: str = "",
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "captured_at": utc_now(),
        "started_at": started_at,
        "run_id": paths.run_id,
        "status": status,
        "phase": phase,
        "iteration": iteration,
        "detail": detail,
        "remote_status": remote_status,
        "transient_failures": transient_failures,
        "completed_stages": list(completed_stages),
        "evidence_sha256": evidence_sha256,
        "supervisor_pid": os.getpid(),
        "official_submission_executed": False,
        "kaggle_submission_executed": False,
        "signals_sent": 0,
        "other_processes_modified": False,
    }


def _event(state: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(state), "schema": EVENT_SCHEMA}


def _record_stage(
    paths: SupervisorPaths,
    *,
    phase: str,
    payload: Mapping[str, Any],
    iteration: int,
    started_at: str,
    completed_stages: list[str],
) -> None:
    if payload.get("run_id") not in {None, paths.run_id}:
        raise EndToEndSupervisorError(f"{phase} evidence belongs to another Run")
    if "signals_sent" in payload and int(payload.get("signals_sent") or 0) != 0:
        raise EndToEndSupervisorError(f"{phase} evidence reports process signals")
    if "other_processes_modified" in payload and payload.get("other_processes_modified") is not False:
        raise EndToEndSupervisorError(f"{phase} evidence reports another process was modified")
    for key in ("official_submission_executed", "kaggle_submission_executed"):
        if key in payload and payload.get(key) is not False:
            raise EndToEndSupervisorError(f"{phase} evidence reports a forbidden submission")
    if phase not in completed_stages:
        completed_stages.append(phase)
    state = _public_state(
        paths,
        status="running",
        phase=phase,
        iteration=iteration,
        started_at=started_at,
        detail="verified and reused" if payload.get("status") == "reused" else "completed",
        completed_stages=completed_stages,
        evidence_sha256=hashlib.sha256(
            json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    )
    atomic_json(paths.state, state)
    append_jsonl(paths.events, _event(state))


def advance_once(
    paths: SupervisorPaths,
    api: EndToEndApi,
    *,
    iteration: int,
    started_at: str,
    sample_interval_seconds: int,
) -> dict[str, Any]:
    completed_stages: list[str] = []
    if api.final_complete():
        return _public_state(
            paths,
            status="completed",
            phase="delivery_ingress",
            iteration=iteration,
            started_at=started_at,
            detail="same Run and four hash-bound downloads verified",
            completed_stages=(
                "waiting_launch",
                "preflight_ingress",
                "poll_campaign",
                "collect",
                "aggregate",
                "campaign_ingress_review_freeze",
                "private_grader_once",
                "grader_ingress",
                "claim_audit",
                "deterministic_delivery",
                "delivery_ingress",
            ),
        )
    if not api.launch_ready():
        return _public_state(
            paths,
            status="waiting",
            phase="waiting_launch",
            iteration=iteration,
            started_at=started_at,
            detail="waiting for the existing gate watcher to publish a fresh GO launch",
        )
    _record_stage(
        paths,
        phase="waiting_launch",
        payload={"status": "reused", "run_id": paths.run_id},
        iteration=iteration,
        started_at=started_at,
        completed_stages=completed_stages,
    )

    preflight = api.ensure_preflight()
    _record_stage(
        paths,
        phase="preflight_ingress",
        payload=preflight,
        iteration=iteration,
        started_at=started_at,
        completed_stages=completed_stages,
    )
    status_payload = api.campaign_status()
    remote = status_payload.get("state") if isinstance(status_payload.get("state"), dict) else {}
    remote_status = str(remote.get("status") or "missing")
    process_exists = status_payload.get("process_exists") is True

    if remote_status in TERMINAL_FAILURE_STATES:
        raise EndToEndSupervisorError(f"remote SIIM campaign failed closed in state {remote_status}")
    if remote_status == "needs_continuation":
        if process_exists:
            return _public_state(
                paths,
                status="running",
                phase="poll_campaign",
                iteration=iteration,
                started_at=started_at,
                remote_status=remote_status,
                detail="remote supervisor is finishing its checkpoint boundary",
                completed_stages=completed_stages,
            )
        resumed = api.resume_campaign_if_ready(sample_interval_seconds=sample_interval_seconds)
        resumed_status = str(resumed.get("status") or "")
        return _public_state(
            paths,
            status="running" if resumed_status == "launched" else "waiting",
            phase="poll_campaign",
            iteration=iteration,
            started_at=started_at,
            remote_status=remote_status,
            detail="same Run continuation launched" if resumed_status == "launched" else "continuation gate remains HOLD",
            completed_stages=completed_stages,
        )
    if remote_status != "awaiting_collection_and_freeze":
        if remote_status not in ACTIVE_REMOTE_STATES:
            raise EndToEndSupervisorError(
                f"remote SIIM campaign published unknown state {remote_status}"
            )
        if not process_exists:
            raise EndToEndSupervisorError(
                f"remote SIIM supervisor stopped unexpectedly in active state {remote_status}"
            )
        return _public_state(
            paths,
            status="running",
            phase="poll_campaign",
            iteration=iteration,
            started_at=started_at,
            remote_status=remote_status,
            detail="real A800 campaign is still progressing",
            completed_stages=completed_stages,
        )

    _record_stage(
        paths,
        phase="poll_campaign",
        payload={"status": "ready", "run_id": paths.run_id, "remote_status": remote_status},
        iteration=iteration,
        started_at=started_at,
        completed_stages=completed_stages,
    )
    collection = api.ensure_collection()
    _record_stage(
        paths,
        phase="collect",
        payload=collection,
        iteration=iteration,
        started_at=started_at,
        completed_stages=completed_stages,
    )
    aggregation = api.ensure_aggregation()
    _record_stage(
        paths,
        phase="aggregate",
        payload=aggregation,
        iteration=iteration,
        started_at=started_at,
        completed_stages=completed_stages,
    )
    campaign_ingress = api.ensure_campaign_ingress()
    _record_stage(
        paths,
        phase="campaign_ingress_review_freeze",
        payload=campaign_ingress,
        iteration=iteration,
        started_at=started_at,
        completed_stages=completed_stages,
    )
    grader_execution = api.ensure_private_grader_execution()
    _record_stage(
        paths,
        phase="private_grader_once",
        payload=grader_execution,
        iteration=iteration,
        started_at=started_at,
        completed_stages=completed_stages,
    )
    grader_ingress = api.ensure_grader_ingress(grader_execution)
    _record_stage(
        paths,
        phase="grader_ingress",
        payload=grader_ingress,
        iteration=iteration,
        started_at=started_at,
        completed_stages=completed_stages,
    )
    claim = api.ensure_claim_audit()
    _record_stage(
        paths,
        phase="claim_audit",
        payload=claim,
        iteration=iteration,
        started_at=started_at,
        completed_stages=completed_stages,
    )
    delivery = api.ensure_delivery_build()
    _record_stage(
        paths,
        phase="deterministic_delivery",
        payload=delivery,
        iteration=iteration,
        started_at=started_at,
        completed_stages=completed_stages,
    )
    delivery_ingress = api.ensure_delivery_ingress(claim, delivery)
    _record_stage(
        paths,
        phase="delivery_ingress",
        payload=delivery_ingress,
        iteration=iteration,
        started_at=started_at,
        completed_stages=completed_stages,
    )
    if not api.final_complete():
        raise EndToEndSupervisorError("end-to-end stages returned without a completed Run")
    return _public_state(
        paths,
        status="completed",
        phase="delivery_ingress",
        iteration=iteration,
        started_at=started_at,
        remote_status=remote_status,
        detail="training, terminal grader, Claim Audit, and four downloads completed",
        completed_stages=completed_stages,
    )


def is_transient_error(exc: BaseException) -> bool:
    """Return true only for bounded remote transport failures.

    Credential failures are fail-closed by default.  The SOCKS channel labels
    only handshake closure and target-connect failures with the explicit
    retryable subtype; auth, pinned-host, identity, and configuration failures
    remain deterministic.
    """
    if isinstance(exc, CredentialError):
        return isinstance(exc, RetryableTransportError)
    transient_names = {
        "ChannelException",
        "ConnectionError",
        "EOFError",
        "NoValidConnectionsError",
        "SSHException",
        "TimeoutError",
    }
    return isinstance(exc, (EOFError, TimeoutError, ConnectionError)) or type(exc).__name__ in transient_names


def supervise(
    paths: SupervisorPaths,
    *,
    poll_seconds: int,
    sample_interval_seconds: int,
    max_wait_seconds: int,
    max_transient_failures: int,
    once: bool,
    api: EndToEndApi | None = None,
    require_profile: bool = True,
) -> dict[str, Any]:
    if poll_seconds < 1 or sample_interval_seconds < 0 or max_wait_seconds < 0:
        raise ValueError("supervisor timing values are invalid")
    if max_transient_failures < 1:
        raise ValueError("max_transient_failures must be positive")
    selected_api = api or ProductionApi(paths, require_profile=require_profile)
    started_monotonic = time.monotonic()
    started_at = utc_now()
    transient_failures = 0
    iteration = 0
    with exclusive_process_lock(paths.lock):
        while True:
            iteration += 1
            try:
                state = advance_once(
                    paths,
                    selected_api,
                    iteration=iteration,
                    started_at=started_at,
                    sample_interval_seconds=sample_interval_seconds,
                )
                transient_failures = 0
            except Exception as exc:
                if not is_transient_error(exc):
                    state = _public_state(
                        paths,
                        status="failed_closed",
                        phase="evidence_validation",
                        iteration=iteration,
                        started_at=started_at,
                        detail=f"{type(exc).__name__}: deterministic evidence conflict",
                    )
                    atomic_json(paths.state, state)
                    append_jsonl(paths.events, _event(state))
                    raise
                transient_failures += 1
                state = _public_state(
                    paths,
                    status="retryable_error",
                    phase="remote_transport",
                    iteration=iteration,
                    started_at=started_at,
                    detail=f"{type(exc).__name__}: transient transport failure",
                    transient_failures=transient_failures,
                )
                if transient_failures > max_transient_failures:
                    state["status"] = "failed_closed"
                    state["detail"] = "bounded remote retry budget exhausted"
            atomic_json(paths.state, state)
            append_jsonl(paths.events, _event(state))
            print(json.dumps(state, ensure_ascii=False), flush=True)

            if state["status"] in {"completed", "failed_closed"} or once:
                return state
            elapsed = time.monotonic() - started_monotonic
            if max_wait_seconds and elapsed >= max_wait_seconds:
                exhausted = {
                    **state,
                    "captured_at": utc_now(),
                    "status": "wait_budget_exhausted",
                    "detail": "local supervisor wait budget exhausted without changing remote work",
                }
                atomic_json(paths.state, exhausted)
                append_jsonl(paths.events, _event(exhausted))
                return exhausted
            backoff = poll_seconds * min(4, max(1, transient_failures))
            if max_wait_seconds:
                backoff = min(backoff, max(0.0, max_wait_seconds - elapsed))
            time.sleep(backoff)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--sample-interval-seconds", type=int, default=15)
    parser.add_argument("--max-wait-seconds", type=int, default=0)
    parser.add_argument("--max-transient-failures", type=int, default=8)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    paths = SupervisorPaths.build(args.project_root, args.run_id)
    try:
        state = supervise(
            paths,
            poll_seconds=args.poll_seconds,
            sample_interval_seconds=args.sample_interval_seconds,
            max_wait_seconds=args.max_wait_seconds,
            max_transient_failures=args.max_transient_failures,
            once=args.once,
        )
    except (EndToEndSupervisorError, ValueError, OSError) as exc:
        print(
            json.dumps(
                {
                    "schema": campaign.BINDING.schema("end_to_end_failure"),
                    "captured_at": utc_now(),
                    "run_id": args.run_id,
                    "status": "failed_closed",
                    "error_type": type(exc).__name__,
                    "signals_sent": 0,
                    "other_processes_modified": False,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    if state["status"] == "completed":
        return 0
    if state["status"] in {"waiting", "running", "wait_budget_exhausted"}:
        return 4
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
