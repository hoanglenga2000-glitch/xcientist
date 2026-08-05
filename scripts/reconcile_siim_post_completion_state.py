#!/usr/bin/env python3
"""Reconcile post-completion SIIM metadata without mutating frozen evidence.

The job90353 campaign was governed by a corrected 78-hour campaign plan while
the later local workflow ingress emitted the original 24-hour template into the
hash-frozen research design.  The end-to-end supervisor also stopped on an
intermediate deterministic-ingress conflict before a separate recovery path
completed the same Run.  This tool preserves those historical facts and writes
an append-only supersession/reconciliation layer.

It never connects to HPC, starts training, executes a grader, submits to
Kaggle, creates a Run, or changes candidate/delivery artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")
SCHEMA = "evomind.siim.post_completion_reconciliation.v1"
BUDGET_SCHEMA = "evomind.siim.budget_supersession.v1"
EXPECTED_TASKS = {
    "request_setup",
    "hpc_data_preflight",
    "data_audit",
    "research_design",
    "preprocessing_ablation",
    "full_training",
    "independent_review_freeze",
    "terminal_private_grader",
    "claim_audit_delivery",
}
FROZEN_FILES = (
    "research_design.json",
    "workflow_contract.json",
    "candidate_freeze.json",
    "review.json",
    "private_grader.json",
    "private_grader_ledger.json",
    "claim_audit.json",
    "artifact_manifest.json",
    "evomind-siim-isic-report.pdf",
    "evomind-siim-isic-results.csv",
    "evomind-siim-isic-code.zip",
    "evomind-siim-isic-evidence.zip",
)
CURRENT_SNAPSHOT_FILES = (
    "status.json",
    "end_to_end_supervisor.json",
    "post_completion_delivery_watch.json",
    "narration_audio_watch.json",
    "published_status.json",
)
SNAPSHOT_ARCHIVE_STEMS = {
    "end_to_end_supervisor": "e2e",
    "post_completion_delivery_watch": "delivery_watch",
    "narration_audio_watch": "narration_watch",
}


class ReconciliationError(RuntimeError):
    """Raised when authoritative completion evidence is inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReconciliationError(message)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file() and path.stat().st_size > 0, f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReconciliationError(f"invalid {label}: {path}: {error}") from error
    _require(isinstance(payload, dict), f"{label} is not an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            Path(temporary).unlink(missing_ok=True)
        except OSError:
            pass


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, _canonical_json(payload))


def _append_jsonl_once(path: Path, payload: Mapping[str, Any], identity: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    if identity in existing:
        return
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _same_run(payload: Mapping[str, Any], run_id: str, label: str) -> None:
    _require(payload.get("run_id") == run_id, f"{label} belongs to a different Run")


def _budget(payload: Mapping[str, Any], label: str) -> dict[str, int]:
    value = payload.get("budget_hours")
    _require(isinstance(value, dict), f"{label} has no budget_hours")
    result = {str(key): int(number) for key, number in value.items()}
    _require(all(number >= 0 for number in result.values()), f"{label} budget is negative")
    return result


def _artifact_hash(manifest: Mapping[str, Any], relative: str) -> str:
    records = manifest.get("artifacts")
    _require(isinstance(records, list), "artifact manifest has no records")
    matches = [record for record in records if isinstance(record, dict) and record.get("path") == relative]
    _require(len(matches) == 1, f"artifact manifest does not bind exactly one {relative}")
    return str(matches[0].get("sha256") or "").lower()


def _iso_epoch(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


@dataclass(frozen=True)
class Layout:
    workspace: Path
    run_id: str
    run_dir: Path
    campaign_dir: Path
    video_path: Path
    video_qa_path: Path


def resolve_layout(
    workspace: str | Path,
    run_id: str,
    *,
    video_path: str | Path | None = None,
    video_qa_path: str | Path | None = None,
) -> Layout:
    workspace_path = Path(workspace).resolve()
    _require(bool(RUN_ID_RE.fullmatch(run_id)), f"invalid Run ID: {run_id!r}")
    run_dir = workspace_path / "workspace" / "evomind_runs" / run_id
    campaign_dir = workspace_path / "workspace" / "hpc" / "job90353_siim_campaign" / run_id
    default_video = (
        workspace_path
        / "video-production"
        / "siim-isic-melanoma-commercial-v2-user-journey"
        / "final"
        / "evomind-siim-isic-medical-research-user-journey-zh-92s.mp4"
    )
    resolved_video = Path(video_path).resolve() if video_path else default_video
    resolved_qa = (
        Path(video_qa_path).resolve()
        if video_qa_path
        else resolved_video.parent / "qa" / "final-qa-report.json"
    )
    _require(run_dir.is_dir(), f"Run directory is missing: {run_dir}")
    _require(campaign_dir.is_dir(), f"campaign directory is missing: {campaign_dir}")
    return Layout(workspace_path, run_id, run_dir, campaign_dir, resolved_video, resolved_qa)


def inspect(layout: Layout) -> dict[str, Any]:
    run = _read_json(layout.run_dir / "run.json", "Run snapshot")
    _same_run(run, layout.run_id, "Run snapshot")
    _require(run.get("status") == "completed", "Run is not completed")
    tasks = run.get("tasks")
    _require(isinstance(tasks, dict), "Run has no task graph")
    _require(set(tasks) == EXPECTED_TASKS, "Run task set is not the nine-node SIIM graph")
    _require(all(isinstance(task, dict) and task.get("status") == "completed" for task in tasks.values()), "not all nine nodes completed")

    design_path = layout.run_dir / "research_design.json"
    design = _read_json(design_path, "frozen research design")
    _same_run(design, layout.run_id, "frozen research design")
    _require(design.get("status") == "frozen_before_experiment", "research design is not frozen")
    original_budget = _budget(design, "research design")
    _require(original_budget.get("total") == 24, "research design no longer preserves the original 24-hour template")

    artifact_manifest = _read_json(layout.run_dir / "artifact_manifest.json", "artifact manifest")
    _same_run(artifact_manifest, layout.run_id, "artifact manifest")
    _require(artifact_manifest.get("status") == "verified", "artifact manifest is not verified")
    _require(int(artifact_manifest.get("private_grader_execution_count") or 0) == 1, "artifact manifest does not record one grader execution")
    _require(_artifact_hash(artifact_manifest, "research_design.json") == _sha256(design_path), "frozen research design hash changed")

    plan_path = layout.campaign_dir / "campaign_plan.json"
    plan = _read_json(plan_path, "corrected campaign plan")
    _same_run(plan, layout.run_id, "corrected campaign plan")
    effective_budget = _budget(plan, "campaign plan")
    _require(effective_budget == {"ablation": 4, "formal_training": 72, "delivery": 2, "total": 78}, "campaign plan is not corrected option A")
    plan_original = plan.get("original_budget_hours")
    _require(isinstance(plan_original, dict) and int(plan_original.get("total") or 0) == 24, "campaign plan lost original budget provenance")
    _require(plan.get("budget_policy") == "user_selected_A_corrected_full_closure", "campaign plan budget policy is wrong")

    supersession_path = layout.campaign_dir / "campaign_plan_supersession.json"
    supersession = _read_json(supersession_path, "campaign plan supersession")
    _same_run(supersession, layout.run_id, "campaign plan supersession")
    _require(supersession.get("new_plan_sha256") == _sha256(plan_path), "campaign plan supersession does not bind the corrected plan")

    state_path = layout.campaign_dir / "remote_evidence" / "campaign_state.json"
    state = _read_json(state_path, "collected campaign state")
    _same_run(state, layout.run_id, "collected campaign state")
    _require(state.get("status") == "awaiting_collection_and_freeze", "collected campaign state is not the terminal remote handoff")
    corrected_start = float(state.get("corrected_formal_budget_started_at_epoch") or 0.0)
    corrected_seconds = int(state.get("corrected_formal_budget_seconds") or 0)
    _require(corrected_start > 0 and corrected_seconds == 72 * 3600, "remote campaign did not use the 72-hour formal budget")
    finished = datetime.fromisoformat(str(state.get("updated_at")).replace("Z", "+00:00")).timestamp()
    elapsed_seconds = max(0.0, finished - corrected_start)
    _require(elapsed_seconds <= corrected_seconds, "formal training exceeded the corrected 72-hour budget")
    _require(set(int(seed) for seed in state.get("completed_seeds") or []) == {43, 44, 45}, "remote state does not contain all formal seeds")

    freeze = _read_json(layout.run_dir / "candidate_freeze.json", "candidate freeze")
    grader = _read_json(layout.run_dir / "private_grader.json", "private grader result")
    ledger = _read_json(layout.run_dir / "private_grader_ledger.json", "private grader ledger")
    claim = _read_json(layout.run_dir / "claim_audit.json", "Claim Audit")
    deliverables = _read_json(layout.run_dir / "deliverables.json", "deliverables")
    for label, payload in (("candidate freeze", freeze), ("private grader", grader), ("private grader ledger", ledger), ("Claim Audit", claim), ("deliverables", deliverables)):
        _same_run(payload, layout.run_id, label)
    _require(freeze.get("status") == "frozen_before_private_grader", "candidate is not frozen")
    _require(int(ledger.get("execution_count") or 0) == 1, "private grader execution count is not exactly one")
    _require(ledger.get("feedback_used_for_tuning") is False, "private grader feedback entered tuning")
    _require(grader.get("status") == "failed_closed" and grader.get("mle_private_grader_score") is None, "grader boundary changed")
    _require(claim.get("status") == "passed", "Claim Audit did not pass")
    _require(deliverables.get("status") == "ready", "deliverables are not ready")
    _require(all(payload.get("official_submission_executed") is False for payload in (grader, ledger)), "official submission was executed")
    _require(all(int(payload.get("signals_sent") or 0) == 0 for payload in (grader, ledger, claim)), "signals were sent")
    _require(all(payload.get("other_processes_modified") is False for payload in (grader, ledger, claim)), "another process was modified")

    _require(layout.video_path.is_file() and layout.video_path.stat().st_size > 0, "final video is missing")
    video_qa = _read_json(layout.video_qa_path, "video QA report")
    _require(video_qa.get("status") == "pass", "video QA did not pass")
    measured = video_qa.get("measured") if isinstance(video_qa.get("measured"), dict) else {}
    _require(measured.get("sha256") == _sha256(layout.video_path), "video QA is bound to another MP4")

    immutable_hashes = {name: _sha256(layout.run_dir / name) for name in FROZEN_FILES}
    # The delivery manifest is the candidate-freeze integrity ledger for every
    # frozen file except itself.  Comparing two hashes calculated from the same
    # current file would not detect post-freeze mutation, so bind every record
    # back to the pre-existing manifest here.
    for name, digest in immutable_hashes.items():
        if name == "artifact_manifest.json":
            continue
        _require(
            _artifact_hash(artifact_manifest, name) == digest,
            f"artifact manifest hash mismatch for {name}",
        )
    return {
        "run": run,
        "original_budget": original_budget,
        "effective_budget": effective_budget,
        "plan": plan,
        "plan_path": plan_path,
        "plan_sha256": _sha256(plan_path),
        "plan_supersession_path": supersession_path,
        "plan_supersession_sha256": _sha256(supersession_path),
        "remote_state_path": state_path,
        "remote_state_sha256": _sha256(state_path),
        "corrected_formal_started_at": _iso_epoch(corrected_start),
        "formal_training_handoff_at": datetime.fromtimestamp(finished, tz=timezone.utc).isoformat(),
        "formal_training_elapsed_seconds": elapsed_seconds,
        "immutable_hashes": immutable_hashes,
        "video_sha256": _sha256(layout.video_path),
        "video_qa_sha256": _sha256(layout.video_qa_path),
    }


def _archive_snapshot(layout: Layout, path: Path, digest: str) -> Path:
    stem = SNAPSHOT_ARCHIVE_STEMS.get(path.stem, path.stem)
    archive = layout.campaign_dir / ".reconciliation_backups" / f"{stem}.pre_{digest[:12]}{path.suffix}"
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, archive)
    _require(_sha256(archive) == digest, f"snapshot archive hash mismatch: {archive}")
    return archive


def _reconciled_at(layout: Layout) -> str:
    existing = layout.campaign_dir / "completion_reconciliation.json"
    if existing.is_file():
        payload = _read_json(existing, "existing reconciliation")
        _same_run(payload, layout.run_id, "existing reconciliation")
        value = str(payload.get("reconciled_at") or "")
        _require(bool(value), "existing reconciliation has no timestamp")
        return value
    return datetime.now(timezone.utc).isoformat()


def apply(layout: Layout) -> dict[str, Any]:
    facts = inspect(layout)
    before = dict(facts["immutable_hashes"])
    reconciled_at = _reconciled_at(layout)
    budget_path = layout.run_dir / "budget_supersession.json"
    reconciliation_path = layout.campaign_dir / "completion_reconciliation.json"
    run_reconciliation_path = layout.run_dir / "post_completion_reconciliation.json"

    snapshots: dict[str, dict[str, Any]] = {}
    for name in CURRENT_SNAPSHOT_FILES:
        path = layout.campaign_dir / name
        digest = _sha256(path)
        existing = _read_json(path, name)
        original_digest = str((existing.get("reconciliation") or {}).get("previous_snapshot_sha256") or digest) if isinstance(existing.get("reconciliation"), dict) else digest
        if digest == original_digest:
            archive = _archive_snapshot(layout, path, digest)
        else:
            stem = SNAPSHOT_ARCHIVE_STEMS.get(path.stem, path.stem)
            candidates = list((layout.campaign_dir / ".reconciliation_backups").glob(f"{stem}.pre_{original_digest[:12]}{path.suffix}"))
            _require(len(candidates) == 1, f"missing original snapshot archive for {name}")
            archive = candidates[0]
        snapshots[name] = {"path": str(path), "previous_snapshot_sha256": original_digest, "archive": str(archive)}

    budget_payload = {
        "schema": BUDGET_SCHEMA,
        "run_id": layout.run_id,
        "status": "effective",
        "policy": "append_only_supersession_preserves_frozen_research_design",
        "reason": "The local workflow ingress emitted the original 24-hour template after the corrected option-A campaign plan had already governed execution.",
        "frozen_research_design": {
            "path": str(layout.run_dir / "research_design.json"),
            "sha256": before["research_design.json"],
            "immutable": True,
            "budget_hours": facts["original_budget"],
        },
        "effective_campaign_plan": {
            "path": str(facts["plan_path"]),
            "sha256": facts["plan_sha256"],
            "supersession_path": str(facts["plan_supersession_path"]),
            "supersession_sha256": facts["plan_supersession_sha256"],
            "budget_policy": facts["plan"].get("budget_policy"),
            "budget_hours": facts["effective_budget"],
        },
        "runtime_evidence": {
            "remote_state_path": str(facts["remote_state_path"]),
            "remote_state_sha256": facts["remote_state_sha256"],
            "corrected_formal_budget_started_at": facts["corrected_formal_started_at"],
            "formal_training_handoff_at": facts["formal_training_handoff_at"],
            "formal_training_elapsed_seconds": round(facts["formal_training_elapsed_seconds"], 3),
            "formal_training_budget_seconds": 72 * 3600,
            "within_budget": True,
        },
        "effective_budget_hours": facts["effective_budget"],
        "reconciled_at": reconciled_at,
    }
    _atomic_json(budget_path, budget_payload)

    completion_payload = {
        "schema": SCHEMA,
        "run_id": layout.run_id,
        "job_id": 90353,
        "status": "completed",
        "phase": "post_completion_reconciliation",
        "reconciled_at": reconciled_at,
        "root_cause": {
            "budget": "workflow ingress retained the original 24-hour template instead of projecting the corrected option-A campaign plan",
            "supervisor": "the legacy supervisor stopped at an intermediate deterministic-ingress conflict; a later recovery path completed the same Run",
        },
        "authoritative_evidence": {
            "run_json": {"path": str(layout.run_dir / "run.json"), "status": "completed"},
            "candidate_freeze_sha256": before["candidate_freeze.json"],
            "private_grader_ledger_sha256": before["private_grader_ledger.json"],
            "claim_audit_sha256": before["claim_audit.json"],
            "artifact_manifest_sha256": before["artifact_manifest.json"],
            "video_path": str(layout.video_path),
            "video_sha256": facts["video_sha256"],
            "video_qa_path": str(layout.video_qa_path),
            "video_qa_sha256": facts["video_qa_sha256"],
        },
        "budget_supersession": {"path": str(budget_path), "sha256": _sha256(budget_path)},
        "legacy_snapshots": snapshots,
        "invariants": {
            "top_level_run_created": False,
            "training_restarted": False,
            "grader_reexecuted": False,
            "private_grader_execution_count": 1,
            "official_submission_executed": False,
            "kaggle_submission_executed": False,
            "signals_sent": 0,
            "other_processes_modified": False,
            "frozen_artifacts_modified": False,
        },
    }
    _atomic_json(reconciliation_path, completion_payload)
    reconciliation_hash = _sha256(reconciliation_path)

    status_path = layout.campaign_dir / "status.json"
    status = _read_json(status_path, "campaign status")
    status["status"] = "completed"
    status["authoritative_state"] = "completed_after_collection_freeze_grader_audit_delivery_and_video"
    status["reconciliation"] = {
        "path": str(reconciliation_path),
        "sha256": reconciliation_hash,
        "previous_snapshot_sha256": snapshots["status.json"]["previous_snapshot_sha256"],
        "legacy_remote_state_preserved": True,
        "reconciled_at": reconciled_at,
    }
    _atomic_json(status_path, status)

    supervisor_path = layout.campaign_dir / "end_to_end_supervisor.json"
    previous_supervisor = _read_json(supervisor_path, "end-to-end supervisor snapshot")
    supervisor = {
        "schema": previous_supervisor.get("schema", "evomind.siim.job90353.end_to_end_supervisor.v1"),
        "captured_at": reconciled_at,
        "started_at": previous_supervisor.get("started_at"),
        "run_id": layout.run_id,
        "status": "completed",
        "phase": "post_completion_reconciliation",
        "iteration": previous_supervisor.get("iteration", 1),
        "detail": "same Run completed through recovery ingress; frozen candidate, one grader record, Claim Audit, four downloads, UI and video QA verified",
        "remote_status": "awaiting_collection_and_freeze",
        "transient_failures": previous_supervisor.get("transient_failures", 0),
        "completed_stages": [
            "waiting_launch", "preflight_ingress", "poll_campaign", "collect", "aggregate",
            "campaign_ingress_review_freeze", "private_grader_once", "grader_ingress",
            "claim_audit", "deterministic_delivery", "delivery_ingress", "video_qa",
        ],
        "evidence_sha256": reconciliation_hash,
        "supervisor_pid": None,
        "official_submission_executed": False,
        "kaggle_submission_executed": False,
        "signals_sent": 0,
        "other_processes_modified": False,
        "reconciliation": {
            "path": str(reconciliation_path),
            "sha256": reconciliation_hash,
            "previous_snapshot_sha256": snapshots["end_to_end_supervisor.json"]["previous_snapshot_sha256"],
            "historical_failures_preserved_in": str(layout.campaign_dir / "end_to_end_supervisor.jsonl"),
        },
    }
    _atomic_json(supervisor_path, supervisor)
    supervisor_event = dict(supervisor)
    supervisor_event["schema"] = "evomind.siim.job90353.end_to_end_event.v1"
    supervisor_event["event_id"] = f"post_completion_reconciliation:{reconciliation_hash}"
    _append_jsonl_once(layout.campaign_dir / "end_to_end_supervisor.jsonl", supervisor_event, supervisor_event["event_id"])

    for name, schema in (
        ("post_completion_delivery_watch.json", "evomind.siim.post_completion_delivery_watch.v1"),
        ("narration_audio_watch.json", "evomind.siim.narration_audio_watch.v1"),
    ):
        path = layout.campaign_dir / name
        previous = _read_json(path, name)
        payload = dict(previous)
        payload.update(
            {
                "schema": previous.get("schema", schema),
                "captured_at": reconciled_at,
                "run_id": layout.run_id,
                "status": "completed",
                "delivery_watch_status": "completed",
                "supervisor_status": "completed",
                "supervisor_phase": "post_completion_reconciliation",
                "recording_started": True,
                "recording_completed": True,
                "video_path": str(layout.video_path),
                "video_sha256": facts["video_sha256"],
                "official_submission_executed": False,
                "signals_sent": 0,
                "other_processes_modified": False,
                "reconciliation": {
                    "path": str(reconciliation_path),
                    "sha256": reconciliation_hash,
                    "previous_snapshot_sha256": snapshots[name]["previous_snapshot_sha256"],
                },
            }
        )
        _atomic_json(path, payload)
        event_id = f"post_completion_reconciliation:{reconciliation_hash}:{name}"
        payload["event_id"] = event_id
        _append_jsonl_once(path.with_suffix(".jsonl"), payload, event_id)

    published_path = layout.campaign_dir / "published_status.json"
    published = _read_json(published_path, "published status")
    published.update(
        {
            "status": "published",
            "authoritative_run_status": "completed",
            "authoritative_campaign_status": "completed",
            "effective_budget_hours": facts["effective_budget"],
            "completion_reconciliation": {"path": str(reconciliation_path), "sha256": reconciliation_hash},
            "budget_supersession": {"path": str(budget_path), "sha256": _sha256(budget_path)},
            "video_path": str(layout.video_path),
            "video_sha256": facts["video_sha256"],
            "reconciled_at": reconciled_at,
            "signals_sent": 0,
            "other_processes_modified": False,
            "reconciliation": {
                "path": str(reconciliation_path),
                "sha256": reconciliation_hash,
                "previous_snapshot_sha256": snapshots["published_status.json"]["previous_snapshot_sha256"],
            },
        }
    )
    _atomic_json(published_path, published)

    run_path = layout.run_dir / "run.json"
    run = _read_json(run_path, "Run snapshot")
    run["effective_budget_hours"] = facts["effective_budget"]
    run["budget_policy"] = facts["plan"].get("budget_policy")
    run["budget_supersession"] = {"path": budget_path.name, "sha256": _sha256(budget_path)}
    run["post_completion_reconciliation"] = {"path": str(run_reconciliation_path), "campaign_path": str(reconciliation_path)}
    _atomic_json(run_path, run)

    run_reconciliation = dict(completion_payload)
    run_reconciliation["campaign_reconciliation"] = {"path": str(reconciliation_path), "sha256": reconciliation_hash}
    run_reconciliation["budget_supersession"] = {"path": str(budget_path), "sha256": _sha256(budget_path)}
    _atomic_json(run_reconciliation_path, run_reconciliation)
    event_id = f"post_completion_metadata_reconciled:{reconciliation_hash}"
    _append_jsonl_once(
        layout.run_dir / "events.jsonl",
        {
            "schema": "evomind.siim.post_completion_event.v1",
            "event_id": event_id,
            "run_id": layout.run_id,
            "status": "completed",
            "phase": "post_completion_reconciliation",
            "captured_at": reconciled_at,
            "budget_supersession_sha256": _sha256(budget_path),
            "completion_reconciliation_sha256": reconciliation_hash,
            "private_grader_execution_count": 1,
            "official_submission_executed": False,
            "signals_sent": 0,
            "other_processes_modified": False,
        },
        event_id,
    )

    after = {name: _sha256(layout.run_dir / name) for name in FROZEN_FILES}
    _require(after == before, "a frozen candidate or delivery artifact changed during reconciliation")
    verified = verify(layout, write_report=True)
    _require(verified["status"] == "passed", "post-reconciliation verification failed")
    return verified


def verify(layout: Layout, *, write_report: bool = False) -> dict[str, Any]:
    facts = inspect(layout)
    budget_path = layout.run_dir / "budget_supersession.json"
    reconciliation_path = layout.campaign_dir / "completion_reconciliation.json"
    run_reconciliation_path = layout.run_dir / "post_completion_reconciliation.json"
    budget = _read_json(budget_path, "budget supersession")
    reconciliation = _read_json(reconciliation_path, "completion reconciliation")
    run_reconciliation = _read_json(run_reconciliation_path, "Run completion reconciliation")
    for label, payload in (("budget supersession", budget), ("completion reconciliation", reconciliation), ("Run completion reconciliation", run_reconciliation)):
        _same_run(payload, layout.run_id, label)
    _require(budget.get("status") == "effective", "budget supersession is not effective")
    _require(budget.get("effective_budget_hours") == facts["effective_budget"], "effective budget mismatch")
    _require(reconciliation.get("status") == "completed", "completion reconciliation is not completed")
    budget_hash = _sha256(budget_path)
    reconciliation_hash = _sha256(reconciliation_path)
    reconciliation_budget_ref = reconciliation.get("budget_supersession")
    _require(
        isinstance(reconciliation_budget_ref, dict)
        and reconciliation_budget_ref.get("sha256") == budget_hash,
        "completion reconciliation does not bind the budget supersession",
    )
    run_budget_ref = run_reconciliation.get("budget_supersession")
    _require(
        isinstance(run_budget_ref, dict) and run_budget_ref.get("sha256") == budget_hash,
        "Run reconciliation does not bind the budget supersession",
    )
    campaign_ref = run_reconciliation.get("campaign_reconciliation")
    _require(
        isinstance(campaign_ref, dict) and campaign_ref.get("sha256") == reconciliation_hash,
        "Run reconciliation does not bind the campaign reconciliation",
    )
    invariants = reconciliation.get("invariants") if isinstance(reconciliation.get("invariants"), dict) else {}
    _require(invariants.get("grader_reexecuted") is False and int(invariants.get("private_grader_execution_count") or 0) == 1, "grader invariant failed")
    _require(invariants.get("top_level_run_created") is False, "Run uniqueness invariant failed")
    _require(invariants.get("official_submission_executed") is False, "submission invariant failed")
    _require(invariants.get("frozen_artifacts_modified") is False, "frozen artifact invariant failed")
    authoritative = reconciliation.get("authoritative_evidence")
    _require(isinstance(authoritative, dict), "completion reconciliation has no authoritative evidence")
    evidence_hashes = {
        "artifact_manifest_sha256": "artifact_manifest.json",
        "candidate_freeze_sha256": "candidate_freeze.json",
        "private_grader_ledger_sha256": "private_grader_ledger.json",
        "claim_audit_sha256": "claim_audit.json",
    }
    for field, name in evidence_hashes.items():
        _require(
            authoritative.get(field) == facts["immutable_hashes"][name],
            f"completion reconciliation evidence hash mismatch for {name}",
        )
    _require(authoritative.get("video_sha256") == facts["video_sha256"], "completion reconciliation video hash mismatch")
    _require(authoritative.get("video_qa_sha256") == facts["video_qa_sha256"], "completion reconciliation video QA hash mismatch")

    legacy_snapshots = reconciliation.get("legacy_snapshots")
    _require(isinstance(legacy_snapshots, dict), "completion reconciliation has no legacy snapshot ledger")
    for name in CURRENT_SNAPSHOT_FILES:
        payload = _read_json(layout.campaign_dir / name, name)
        if name == "status.json":
            _require(payload.get("status") == "completed", "campaign status was not reconciled")
        elif name == "published_status.json":
            _require(payload.get("authoritative_run_status") == "completed", "published status was not reconciled")
        else:
            _require(payload.get("status") == "completed", f"{name} was not reconciled")
        ref = payload.get("reconciliation")
        _require(isinstance(ref, dict) and ref.get("sha256") == reconciliation_hash, f"{name} is not bound to reconciliation")
        ledger_record = legacy_snapshots.get(name)
        _require(isinstance(ledger_record, dict), f"legacy snapshot ledger is missing {name}")
        archived = Path(str(ledger_record.get("archive") or ""))
        previous_hash = str(ledger_record.get("previous_snapshot_sha256") or "")
        _require(archived.is_file(), f"legacy snapshot archive is missing for {name}")
        _require(_sha256(archived) == previous_hash, f"legacy snapshot archive hash mismatch for {name}")
        _require(ref.get("previous_snapshot_sha256") == previous_hash, f"{name} lost its previous snapshot binding")
    run = _read_json(layout.run_dir / "run.json", "Run snapshot")
    _require(run.get("effective_budget_hours") == facts["effective_budget"], "Run snapshot has no effective budget")
    current_hashes = facts["immutable_hashes"]
    report = {
        "schema": "evomind.siim.post_completion_reconciliation_qa.v1",
        "run_id": layout.run_id,
        "status": "passed",
        "effective_budget_hours": facts["effective_budget"],
        "formal_training_elapsed_seconds": round(facts["formal_training_elapsed_seconds"], 3),
        "formal_training_budget_seconds": 72 * 3600,
        "tasks_completed": 9,
        "private_grader_execution_count": 1,
        "official_submission_executed": False,
        "signals_sent": 0,
        "other_processes_modified": False,
        "frozen_hashes": current_hashes,
        "budget_supersession_sha256": budget_hash,
        "completion_reconciliation_sha256": reconciliation_hash,
        "video_sha256": facts["video_sha256"],
    }
    report_path = layout.run_dir / "post_completion_reconciliation_qa.json"
    if write_report:
        _atomic_json(report_path, report)
        _require(_read_json(report_path, "reconciliation QA report") == report, "reconciliation QA report write verification failed")
    elif report_path.is_file():
        _require(_read_json(report_path, "reconciliation QA report") == report, "reconciliation QA report is stale or inconsistent")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--video-path")
    parser.add_argument("--video-qa-path")
    parser.add_argument("--apply", action="store_true", help="Write reconciliation artifacts; default is verify-only.")
    args = parser.parse_args()
    try:
        layout = resolve_layout(
            args.workspace,
            args.run_id,
            video_path=args.video_path,
            video_qa_path=args.video_qa_path,
        )
        result = apply(layout) if args.apply else verify(layout)
    except (OSError, ReconciliationError, ValueError) as error:
        print(json.dumps({"status": "failed_closed", "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
