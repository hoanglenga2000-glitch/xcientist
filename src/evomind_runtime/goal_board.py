"""Persistent, identity-bound goal records for the fixed G21 objective."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .competition_goal import (
    COMPETITIONS,
    FIXED_ALLOCATION,
    FIXED_GOAL_ID,
    FIXED_GOAL_KEY,
    FIXED_RUN_ID,
    GOAL_RECORD_SCHEMA,
    INITIAL_NEGATIVE_HUMAN_BASELINE_EVIDENCE_SHA256,
    OPEN_POLYMER_LEGACY_ADOPTION_SCHEMA,
    OPEN_POLYMER_LEGACY_COMPLETED_EVENT_SEQ,
    OPEN_POLYMER_LEGACY_IDEMPOTENCY_KEY,
    OPEN_POLYMER_LEGACY_MANIFEST_ARTIFACT_ID,
    OPEN_POLYMER_LEGACY_MANIFEST_RECEIPT_ARTIFACT_ID,
    OPEN_POLYMER_LEGACY_MANIFEST_RECEIPT_BYTES,
    OPEN_POLYMER_LEGACY_MANIFEST_RECEIPT_SHA256,
    OPEN_POLYMER_LEGACY_MANIFEST_SCHEMA,
    OPEN_POLYMER_LEGACY_MANIFEST_SHA256,
    OPEN_POLYMER_LEGACY_REVIEW_ARTIFACT_ID,
    OPEN_POLYMER_LEGACY_REVIEW_SHA256,
    OPEN_POLYMER_LEGACY_SOLUTION_ID,
    OPEN_POLYMER_LEGACY_SOURCE_CALL_ID,
    OPEN_POLYMER_LEGACY_SPLIT_ARTIFACT_ID,
    OPEN_POLYMER_LEGACY_SPLIT_SHA256,
    OPEN_POLYMER_LEGACY_STARTED_EVENT_SEQ,
    canonical_json_bytes,
    conditional_baseline_policy,
    sha256_json,
    validate_goal_board_identity,
    validate_goal_item,
    validate_goal_record,
    validate_goal_spec,
)
from .models import utc_now
from .store import RuntimeStore


class GoalRecordError(ValueError):
    """Raised when a Goal cannot be safely bound to an existing Run."""


def _json_text(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def _issue_text(prefix: str, issues: tuple[Any, ...]) -> str:
    codes = ",".join(f"{prefix}{item.code}" for item in issues)
    return codes or f"{prefix}invalid"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _parse_utc(value: Any, code: str) -> datetime:
    raw = str(value or "").strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise GoalRecordError(code) from exc
    if parsed.tzinfo is None:
        raise GoalRecordError(code)
    return parsed.astimezone(timezone.utc)


def _hpc_verify_complete(result: Mapping[str, Any]) -> bool:
    content = result.get("content") if isinstance(result.get("content"), Mapping) else {}
    evidence = content.get("hpc_identity") if isinstance(content.get("hpc_identity"), Mapping) else content
    samples = evidence.get("samples")
    return (
        result.get("ok") is True
        and evidence.get("samples_requested") == 5
        and evidence.get("samples_passed") == 5
        and isinstance(samples, list)
        and len(samples) == 5
        and all(isinstance(sample, Mapping) and sample.get("complete") is True for sample in samples)
    )


def _verified_deliverable(
    store: RuntimeStore,
    *,
    run_id: str,
    artifact_id: str,
    sha256: str,
    byte_count: int | None = None,
    expected_name: str = "",
) -> tuple[dict[str, Any], Path]:
    row = store.get_deliverable(artifact_id)
    if row is None:
        raise GoalRecordError(f"goal evidence artifact is not published: {artifact_id}")
    if str(row.get("run_id") or "") != run_id or str(row.get("session_id") or "") != run_id:
        raise GoalRecordError(f"goal evidence artifact is bound to another Run: {artifact_id}")
    if str(row.get("sha256") or "").lower() != sha256.lower():
        raise GoalRecordError(f"goal evidence artifact SHA mismatch: {artifact_id}")
    if byte_count is not None and int(row.get("bytes", -1)) != int(byte_count):
        raise GoalRecordError(f"goal evidence artifact byte count mismatch: {artifact_id}")
    if expected_name and str(row.get("name") or "") != expected_name:
        raise GoalRecordError(f"goal evidence artifact name mismatch: {artifact_id}")
    path = Path(str(row.get("path") or ""))
    if not path.is_file() or path.is_symlink():
        raise GoalRecordError(f"goal evidence artifact is not a regular file: {artifact_id}")
    if path.stat().st_size != int(row.get("bytes", -1)) or _sha256_file(path) != sha256.lower():
        raise GoalRecordError(f"goal evidence artifact current file mismatch: {artifact_id}")
    return row, path


def _read_json_file(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GoalRecordError(f"{label} artifact is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise GoalRecordError(f"{label} artifact is not a JSON object")
    return value


def _same_number(left: Any, right: Any) -> bool:
    try:
        a = float(left)
        b = float(right)
    except (TypeError, ValueError):
        return False
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def _all_events(store: RuntimeStore, session_id: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    cursor = 0
    while True:
        page = store.list_events(session_id, cursor, 1000)
        if not page:
            break
        events.extend(page)
        next_cursor = max(int(item.get("seq") or 0) for item in page)
        if next_cursor <= cursor:
            raise GoalRecordError("event pagination did not advance")
        cursor = next_cursor
        if len(page) < 1000:
            break
    return events


def _verify_open_polymer_legacy_artifacts(
    store: RuntimeStore,
    *,
    item: Mapping[str, Any],
    goal_created_at: str,
) -> None:
    adoption = item.get("existing_candidate_adoption")
    if not isinstance(adoption, Mapping) or adoption.get("schema") != OPEN_POLYMER_LEGACY_ADOPTION_SCHEMA:
        raise GoalRecordError("Open Polymer legacy adoption v2 is missing")
    if (
        adoption.get("source_tool_call_id") != OPEN_POLYMER_LEGACY_SOURCE_CALL_ID
        or adoption.get("source_idempotency_key") != OPEN_POLYMER_LEGACY_IDEMPOTENCY_KEY
        or adoption.get("source_solution_id") != OPEN_POLYMER_LEGACY_SOLUTION_ID
        or adoption.get("source_started_event_seq") != OPEN_POLYMER_LEGACY_STARTED_EVENT_SEQ
        or adoption.get("source_completed_event_seq") != OPEN_POLYMER_LEGACY_COMPLETED_EVENT_SEQ
        or adoption.get("source_artifact_manifest_artifact_id")
        != OPEN_POLYMER_LEGACY_MANIFEST_ARTIFACT_ID
        or str(adoption.get("source_artifact_manifest_sha256") or "").lower()
        != OPEN_POLYMER_LEGACY_MANIFEST_SHA256
        or adoption.get("source_split_manifest_artifact_id") != OPEN_POLYMER_LEGACY_SPLIT_ARTIFACT_ID
        or str(adoption.get("source_split_manifest_sha256") or "").lower()
        != OPEN_POLYMER_LEGACY_SPLIT_SHA256
        or adoption.get("source_independent_verification_artifact_id")
        != OPEN_POLYMER_LEGACY_REVIEW_ARTIFACT_ID
        or str(adoption.get("source_independent_verification_sha256") or "").lower()
        != OPEN_POLYMER_LEGACY_REVIEW_SHA256
    ):
        raise GoalRecordError("Open Polymer legacy adoption allowlist mismatch")

    source_call = store.get_tool_call(OPEN_POLYMER_LEGACY_SOURCE_CALL_ID)
    all_calls = store.list_tool_calls(FIXED_RUN_ID, limit=2000)
    idempotency_matches = [
        call
        for call in all_calls
        if str(call.get("idempotency_key") or "") == OPEN_POLYMER_LEGACY_IDEMPOTENCY_KEY
    ]
    source_arguments = source_call.get("arguments") if isinstance(source_call, Mapping) else None
    source_result = source_call.get("result") if isinstance(source_call, Mapping) else None
    source_content = source_result.get("content") if isinstance(source_result, Mapping) else None
    published = source_content.get("published_artifacts") if isinstance(source_content, Mapping) else None
    if (
        not isinstance(source_call, Mapping)
        or source_call.get("session_id") != FIXED_RUN_ID
        or source_call.get("tool_name") != "hpc_execute_solution"
        or source_call.get("status") != "completed"
        or source_call.get("idempotency_key") != OPEN_POLYMER_LEGACY_IDEMPOTENCY_KEY
        or len(idempotency_matches) != 1
        or idempotency_matches[0].get("id") != OPEN_POLYMER_LEGACY_SOURCE_CALL_ID
        or not isinstance(source_arguments, Mapping)
        or source_arguments.get("competition") != "open_polymer"
        or source_arguments.get("solution_id") != OPEN_POLYMER_LEGACY_SOLUTION_ID
        or not isinstance(source_result, Mapping)
        or source_result.get("ok") is not True
        or not isinstance(source_content, Mapping)
        or source_content.get("status") != "completed"
        or source_content.get("exit_code") != 0
        or not isinstance(published, list)
    ):
        raise GoalRecordError("Open Polymer source tool call binding is invalid")
    source_started_at = _parse_utc(
        source_call.get("started_at") or source_call.get("created_at"),
        "Open Polymer source tool call timestamp is invalid",
    )
    source_completed_at = _parse_utc(
        source_call.get("completed_at"),
        "Open Polymer source tool call timestamp is invalid",
    )
    goal_time = _parse_utc(goal_created_at, "goal.created timestamp is invalid")
    if not source_started_at < source_completed_at < goal_time:
        raise GoalRecordError("Open Polymer source tool call timestamp ordering is invalid")

    events = _all_events(store, FIXED_RUN_ID)
    started = [item for item in events if int(item.get("seq") or 0) == OPEN_POLYMER_LEGACY_STARTED_EVENT_SEQ]
    completed = [item for item in events if int(item.get("seq") or 0) == OPEN_POLYMER_LEGACY_COMPLETED_EVENT_SEQ]
    started_payload = started[0].get("payload") if len(started) == 1 else None
    completed_payload = completed[0].get("payload") if len(completed) == 1 else None
    if (
        len(started) != 1
        or len(completed) != 1
        or started[0].get("event_type") != "tool.started"
        or completed[0].get("event_type") != "tool.completed"
        or not isinstance(started_payload, Mapping)
        or not isinstance(completed_payload, Mapping)
        or started_payload.get("id") != OPEN_POLYMER_LEGACY_SOURCE_CALL_ID
        or completed_payload.get("tool_call_id") != OPEN_POLYMER_LEGACY_SOURCE_CALL_ID
        or completed_payload.get("ok") is not True
    ):
        raise GoalRecordError("Open Polymer source event binding is invalid")
    event_started_at = _parse_utc(started[0].get("created_at"), "Open Polymer source event timestamp is invalid")
    event_completed_at = _parse_utc(completed[0].get("created_at"), "Open Polymer source event timestamp is invalid")
    if event_started_at != source_started_at or event_completed_at != source_completed_at:
        raise GoalRecordError("Open Polymer source event timestamp binding is invalid")

    manifest = item.get("artifact_manifest")
    if not isinstance(manifest, Mapping) or manifest.get("schema") != OPEN_POLYMER_LEGACY_MANIFEST_SCHEMA:
        raise GoalRecordError("Open Polymer legacy manifest is invalid")
    _, manifest_path = _verified_deliverable(
        store,
        run_id=FIXED_RUN_ID,
        artifact_id=OPEN_POLYMER_LEGACY_MANIFEST_ARTIFACT_ID,
        sha256=OPEN_POLYMER_LEGACY_MANIFEST_SHA256,
        expected_name="artifact-manifest.json",
    )
    if _read_json_file(manifest_path, "Open Polymer manifest") != manifest:
        raise GoalRecordError("Open Polymer embedded legacy manifest differs from the published artifact")
    files = manifest.get("files") if isinstance(manifest.get("files"), list) else []
    by_name = {
        str(entry.get("name") or ""): entry
        for entry in files
        if isinstance(entry, Mapping) and str(entry.get("name") or "")
    }
    if len(by_name) != len(files):
        raise GoalRecordError("Open Polymer legacy manifest contains duplicate or invalid entries")
    bindings = item.get("legacy_artifact_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(by_name):
        raise GoalRecordError("Open Polymer legacy artifact bindings are incomplete")
    published_by_id: dict[str, Mapping[str, Any]] = {}
    for entry in published:
        if not isinstance(entry, Mapping) or not str(entry.get("id") or ""):
            raise GoalRecordError("Open Polymer source publication entry is invalid")
        artifact_id = str(entry.get("id") or "")
        previous = published_by_id.get(artifact_id)
        if previous is not None and (
            previous.get("name") != entry.get("name")
            or int(previous.get("bytes", -1)) != int(entry.get("bytes", -2))
            or str(previous.get("sha256") or "").lower()
            != str(entry.get("sha256") or "").lower()
        ):
            raise GoalRecordError("Open Polymer source publication contains a conflicting duplicate")
        published_by_id[artifact_id] = entry
    expected_published_ids = set(str(value) for value in bindings.values()) | {
        OPEN_POLYMER_LEGACY_MANIFEST_ARTIFACT_ID,
        OPEN_POLYMER_LEGACY_MANIFEST_RECEIPT_ARTIFACT_ID,
    }
    if set(published_by_id) != expected_published_ids:
        raise GoalRecordError("Open Polymer source publication set is not closed")
    artifact_paths: dict[str, Path] = {}
    for name, entry in by_name.items():
        artifact_id = str(bindings.get(name) or "")
        projected = published_by_id.get(artifact_id)
        if (
            not isinstance(projected, Mapping)
            or projected.get("name") != name
            or int(projected.get("bytes", -1)) != int(entry.get("bytes", -2))
            or str(projected.get("sha256") or "").lower()
            != str(entry.get("sha256") or "").lower()
        ):
            raise GoalRecordError(f"Open Polymer source publication mismatch: {name}")
        _, path = _verified_deliverable(
            store,
            run_id=FIXED_RUN_ID,
            artifact_id=artifact_id,
            sha256=str(entry.get("sha256") or ""),
            byte_count=int(entry.get("bytes", -1)),
            expected_name=name,
        )
        artifact_paths[name] = path
    manifest_projection = published_by_id.get(OPEN_POLYMER_LEGACY_MANIFEST_ARTIFACT_ID)
    if (
        not isinstance(manifest_projection, Mapping)
        or manifest_projection.get("name") != "artifact-manifest.json"
        or str(manifest_projection.get("sha256") or "").lower()
        != OPEN_POLYMER_LEGACY_MANIFEST_SHA256
    ):
        raise GoalRecordError("Open Polymer source manifest publication is missing")
    receipt_projection = published_by_id.get(OPEN_POLYMER_LEGACY_MANIFEST_RECEIPT_ARTIFACT_ID)
    if (
        not isinstance(receipt_projection, Mapping)
        or receipt_projection.get("name") != "artifact-manifest-receipt.json"
        or str(receipt_projection.get("sha256") or "").lower()
        != OPEN_POLYMER_LEGACY_MANIFEST_RECEIPT_SHA256
        or int(receipt_projection.get("bytes", -1)) != OPEN_POLYMER_LEGACY_MANIFEST_RECEIPT_BYTES
    ):
        raise GoalRecordError("Open Polymer source manifest receipt publication is invalid")
    _, receipt_path = _verified_deliverable(
        store,
        run_id=FIXED_RUN_ID,
        artifact_id=OPEN_POLYMER_LEGACY_MANIFEST_RECEIPT_ARTIFACT_ID,
        sha256=OPEN_POLYMER_LEGACY_MANIFEST_RECEIPT_SHA256,
        byte_count=OPEN_POLYMER_LEGACY_MANIFEST_RECEIPT_BYTES,
        expected_name="artifact-manifest-receipt.json",
    )
    receipt = _read_json_file(receipt_path, "Open Polymer manifest receipt")
    if (
        receipt.get("artifact") != "artifact-manifest.json"
        or int(receipt.get("bytes", -1)) != int(manifest_projection.get("bytes", -2))
        or str(receipt.get("sha256") or "").lower() != OPEN_POLYMER_LEGACY_MANIFEST_SHA256
    ):
        raise GoalRecordError("Open Polymer source manifest receipt binding is invalid")

    split = _read_json_file(artifact_paths["split-manifest.json"], "Open Polymer split manifest")
    review = _read_json_file(
        artifact_paths["independent-verification.json"], "Open Polymer independent verification"
    )
    task = _read_json_file(artifact_paths["task-contract-v2.json"], "Open Polymer task contract")
    dataset = _read_json_file(artifact_paths["dataset-audit.json"], "Open Polymer dataset audit")
    baseline = _read_json_file(artifact_paths["baseline-evidence.json"], "Open Polymer baseline evidence")
    reproduction = _read_json_file(
        artifact_paths["baseline-reproduction.json"], "Open Polymer baseline reproduction"
    )
    metrics = _read_json_file(artifact_paths["metrics.json"], "Open Polymer metrics")
    comparison = _read_json_file(
        artifact_paths["candidate-vs-baseline.json"], "Open Polymer candidate comparison"
    )
    training = _read_json_file(artifact_paths["training-config.json"], "Open Polymer training config")
    retrospective = _read_json_file(
        artifact_paths["retrospective-memory.json"], "Open Polymer retrospective"
    )
    primary = item.get("primary_baseline") if isinstance(item.get("primary_baseline"), Mapping) else {}
    primary_comparison = (
        item.get("primary_comparison") if isinstance(item.get("primary_comparison"), Mapping) else {}
    )
    bootstrap = metrics.get("bootstrap") if isinstance(metrics.get("bootstrap"), Mapping) else {}
    if (
        split.get("schema") != "evomind.open_polymer.split_manifest.v2"
        or split.get("holdout_blind_for_selection") is not True
        or split.get("group_overlap") != 0
        or task.get("schema") != "evomind.open_polymer.task_contract.v2"
        or task.get("competition") != "open_polymer"
        or task.get("metric") != item.get("official_metric")
        or task.get("direction") != "lower_is_better"
        or task.get("hidden_test_labels_used") is not False
        or dataset.get("schema") != "evomind.open_polymer.dataset_audit.v2"
        or dataset.get("test_labels_read") is not False
        or dataset.get("train_sha256") != manifest.get("data_train_sha256")
        or dataset.get("test_sha256") != manifest.get("data_test_sha256")
        or baseline.get("schema") != "evomind.open_polymer.baseline_evidence.v2"
        or baseline.get("classification") != "PUBLICLY_REPRODUCIBLE_STRONG_BASELINE"
        or baseline.get("implementation_sha256") != primary.get("implementation_sha256")
        or baseline.get("metric") != item.get("official_metric")
        or not _same_number(baseline.get("reproduction_value"), primary.get("value"))
        or reproduction.get("schema") != "evomind.open_polymer.baseline_evidence.v2"
        or not _same_number(reproduction.get("reproduction_value"), primary.get("value"))
        or metrics.get("schema") != "evomind.open_polymer.metrics.v2"
        or not _same_number(metrics.get("baseline_aggregate_normalized_rmse"), primary.get("value"))
        or not _same_number(
            metrics.get("candidate_aggregate_normalized_rmse"),
            primary_comparison.get("candidate_metric"),
        )
        or comparison.get("schema") != "evomind.open_polymer.candidate_vs_baseline.v2"
        or comparison.get("gate_passed") is not True
        or not _same_number(comparison.get("baseline"), primary.get("value"))
        or not _same_number(comparison.get("candidate"), primary_comparison.get("candidate_metric"))
        or review.get("schema") != "evomind.open_polymer.independent_verification.v2"
        or review.get("baseline_match") is not True
        or review.get("candidate_match") is not True
        or review.get("group_overlap") != 0
        or review.get("test_labels_used") is not False
        or review.get("gate_passed") is not True
        or not _same_number(review.get("baseline_recomputed"), primary.get("value"))
        or not _same_number(review.get("candidate_recomputed"), primary_comparison.get("candidate_metric"))
        or training.get("schema") != "evomind.open_polymer.training_config.v2"
        or training.get("single_gpu_solution") is not True
        or training.get("external_submission") is not False
        or retrospective.get("schema") != "evomind.retrospective_memory.v2"
        or retrospective.get("gate_passed") is not True
    ):
        raise GoalRecordError("Open Polymer legacy evidence content is inconsistent")
    expected_lower = float(primary["value"]) + float(bootstrap.get("difference_ci95_lower"))
    expected_upper = float(primary["value"]) + float(bootstrap.get("difference_ci95_upper"))
    if (
        not _same_number(primary_comparison.get("ci95_lower"), expected_lower)
        or not _same_number(primary_comparison.get("ci95_upper"), expected_upper)
        or float(bootstrap.get("difference_ci95_upper")) >= 0.0
        or not _same_number(
            comparison.get("difference_candidate_minus_baseline_ci95_upper"),
            bootstrap.get("difference_ci95_upper"),
        )
    ):
        raise GoalRecordError("Open Polymer paired confidence interval binding is invalid")

    for call in all_calls:
        arguments = call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {}
        if (
            arguments.get("competition") != "open_polymer"
            or call.get("tool_name") not in {"training_route", "hpc_verify", "hpc_execute_solution"}
        ):
            continue
        call_time = _parse_utc(
            call.get("created_at"),
            "Open Polymer tool call timestamp is invalid",
        )
        if call_time >= goal_time:
            raise GoalRecordError("Open Polymer adoption cannot retrain in the Goal window")


def _verify_completed_board_artifacts(
    store: RuntimeStore,
    *,
    board: Mapping[str, Any],
    current_board: Mapping[str, Any] | None,
    current_human_baseline_sha256: str,
    next_status: str,
    spec: Mapping[str, Any],
) -> str:
    is_conditional = conditional_baseline_policy(spec)
    rows = board.get("competitions") if isinstance(board.get("competitions"), list) else []
    completed = [item for item in rows if isinstance(item, Mapping) and item.get("goal_status") == "VERIFIED_COMPLETE"]
    current_rows = (
        current_board.get("competitions")
        if isinstance(current_board, Mapping) and isinstance(current_board.get("competitions"), list)
        else []
    )
    current_completed = {
        str(item.get("competition") or ""): item
        for item in current_rows
        if isinstance(item, Mapping) and item.get("goal_status") == "VERIFIED_COMPLETE"
    }
    next_by_competition = {
        str(item.get("competition") or ""): item
        for item in rows
        if isinstance(item, Mapping)
    }
    for competition, previous in current_completed.items():
        if next_by_competition.get(competition) != previous:
            raise GoalRecordError(f"verified competition evidence is immutable: {competition}")
    board_human_sha = str(board.get("human_baseline_evidence_sha256") or "").lower()
    board_human_artifact_id = str(board.get("human_baseline_evidence_artifact_id") or "")
    if not completed:
        if (
            current_human_baseline_sha256
            and current_human_baseline_sha256 != INITIAL_NEGATIVE_HUMAN_BASELINE_EVIDENCE_SHA256
            and board_human_sha != current_human_baseline_sha256
        ):
            raise GoalRecordError("verified human baseline evidence cannot be removed or downgraded")
        return current_human_baseline_sha256
    checked_completed = {
        str(item.get("competition") or ""): validate_goal_item(
            str(item.get("competition") or ""),
            item,
            conditional_policy=is_conditional,
        )
        for item in completed
    }
    requires_global_human = not is_conditional or any(
        checked.human_baseline_required for checked in checked_completed.values()
    )
    human_evidence: Mapping[str, Any] = {}
    evidence_competitions: Mapping[str, Any] = {}
    completion: Mapping[str, Any] = {}
    availability_evidence: dict[str, Mapping[str, Any]] = {}
    next_human_baseline_sha = current_human_baseline_sha256
    if requires_global_human:
        if board_human_sha == INITIAL_NEGATIVE_HUMAN_BASELINE_EVIDENCE_SHA256 and not is_conditional:
            raise GoalRecordError("initial negative human baseline evidence cannot complete a competition")
        _, evidence_path = _verified_deliverable(
            store,
            run_id=FIXED_RUN_ID,
            artifact_id=board_human_artifact_id,
            sha256=board_human_sha,
        )
        human_evidence = _read_json_file(evidence_path, "human baseline evidence")
        evidence_competitions_value = human_evidence.get("competitions")
        completion_value = human_evidence.get("completion")
        if (
            human_evidence.get("schema") != "evomind.five_competition_human_baseline_audit.v1"
            or human_evidence.get("run_id") != FIXED_RUN_ID
            or str(human_evidence.get("allocation") or "").casefold() != FIXED_ALLOCATION.casefold()
            or human_evidence.get("classification") not in (
                {"HUMAN_BASELINE_UNDEFINED", "HUMAN_BASELINE_PARTIAL", "HUMAN_BASELINE_VERIFIED"}
                if is_conditional
                else {"HUMAN_BASELINE_PARTIAL", "HUMAN_BASELINE_VERIFIED"}
            )
            or not isinstance(evidence_competitions_value, Mapping)
            or set(evidence_competitions_value) != set(COMPETITIONS)
            or not isinstance(completion_value, Mapping)
        ):
            raise GoalRecordError("human baseline evidence artifact is not completion eligible")
        evidence_competitions = evidence_competitions_value
        completion = completion_value
        next_human_baseline_sha = board_human_sha
    else:
        if board_human_sha and board_human_sha != current_human_baseline_sha256:
            raise GoalRecordError("conditional human-baseline evidence cannot drift without a verified global audit")
        for item in completed:
            competition = str(item.get("competition") or "")
            availability = (
                item.get("human_baseline_availability")
                if isinstance(item.get("human_baseline_availability"), Mapping)
                else {}
            )
            artifact_id = str(availability.get("evidence_artifact_id") or "")
            artifact_sha = str(availability.get("evidence_artifact_sha256") or "").lower()
            _, path = _verified_deliverable(
                store,
                run_id=FIXED_RUN_ID,
                artifact_id=artifact_id,
                sha256=artifact_sha,
            )
            value = _read_json_file(path, f"{competition} human-baseline availability")
            if (
                value.get("schema") != "evomind.human-baseline-evidence.v1"
                or value.get("competition") != competition
                or value.get("classification") != "HUMAN_BASELINE_UNDEFINED"
                or value.get("human_baseline") is not None
                or not str(value.get("unique_gate") or "").strip()
                or availability.get("status") != "NOT_VERIFIED_AVAILABLE"
                or availability.get("audit_complete") is not True
                or availability.get("protocol_search_complete") is not True
                or availability.get("human_baseline_verified") is not False
            ):
                raise GoalRecordError(f"human baseline availability does not match board: {competition}")
            availability_evidence[competition] = value
    if next_status.casefold() == "completed":
        if len(completed) != len(COMPETITIONS):
            raise GoalRecordError("completed Goal requires all five competitions")
        if requires_global_human and not is_conditional and (
            human_evidence.get("classification") != "HUMAN_BASELINE_VERIFIED"
            or completion.get("human_baseline_gate_passed") is not True
            or completion.get("all_five_verified") is not True
        ):
            raise GoalRecordError("completed Goal requires all five verified human baselines")
    if requires_global_human and human_evidence.get("classification") in {
        "HUMAN_BASELINE_UNDEFINED",
        "HUMAN_BASELINE_PARTIAL",
    } and (
        completion.get("human_baseline_gate_passed") is not False
        or completion.get("all_five_verified") is not False
    ):
        raise GoalRecordError("non-verified human baseline evidence has invalid completion flags")
    for item in completed:
        competition = str(item.get("competition") or "")
        checked = checked_completed[competition]
        if not checked.strict_verified_complete:
            raise GoalRecordError(_issue_text(f"{competition}:", checked.issues))
        evidence_item = evidence_competitions.get(competition) if requires_global_human else None
        evidence_baseline = (
            evidence_item.get("baseline") or evidence_item.get("human_baseline")
            if isinstance(evidence_item, Mapping)
            else None
        )
        if checked.human_baseline_required:
            if (
                not isinstance(evidence_item, Mapping)
                or
                evidence_item.get("status") != "VERIFIED"
                or evidence_baseline != item.get("human_baseline")
            ):
                raise GoalRecordError(f"human baseline evidence does not match board: {competition}")
        else:
            availability = (
                item.get("human_baseline_availability")
                if isinstance(item.get("human_baseline_availability"), Mapping)
                else {}
            )
            if requires_global_human:
                if (
                    not isinstance(evidence_item, Mapping)
                    or evidence_item.get("status") != "HUMAN_BASELINE_UNDEFINED"
                    or availability.get("status") != "NOT_VERIFIED_AVAILABLE"
                    or availability.get("evidence_artifact_id") != board_human_artifact_id
                    or str(availability.get("evidence_artifact_sha256") or "").lower()
                    != board_human_sha
                ):
                    raise GoalRecordError(f"human baseline availability does not match board: {competition}")
            elif competition not in availability_evidence:
                raise GoalRecordError(f"human baseline availability does not match board: {competition}")
        created_events = [
            event for event in _all_events(store, FIXED_RUN_ID)
            if event.get("event_type") == "goal.created"
        ]
        if len(created_events) != 1:
            raise GoalRecordError("completed evidence requires one goal.created event")
        goal_created_at = str(created_events[0].get("created_at") or "")
        adoption = item.get("existing_candidate_adoption")
        legacy_open_polymer = (
            competition == "open_polymer"
            and isinstance(adoption, Mapping)
            and adoption.get("schema") == OPEN_POLYMER_LEGACY_ADOPTION_SCHEMA
        )
        manifest = item.get("artifact_manifest")
        if not isinstance(manifest, Mapping):
            raise GoalRecordError(f"artifact manifest missing: {competition}")
        if legacy_open_polymer:
            manifest_sha = str(item.get("artifact_manifest_sha256") or "").lower()
            _verify_open_polymer_legacy_artifacts(
                store,
                item=item,
                goal_created_at=goal_created_at,
            )
        else:
            manifest_bytes = canonical_json_bytes(manifest)
            manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
            _verified_deliverable(
                store,
                run_id=FIXED_RUN_ID,
                artifact_id=str(item.get("artifact_manifest_artifact_id") or ""),
                sha256=manifest_sha,
                byte_count=len(manifest_bytes),
            )
            verified_by_role: dict[str, Path] = {}
            for entry in manifest.get("files") or ():
                if not isinstance(entry, Mapping):
                    raise GoalRecordError(f"artifact manifest entry invalid: {competition}")
                path = PurePosixPath(str(entry.get("path") or ""))
                _, artifact_path = _verified_deliverable(
                    store,
                    run_id=FIXED_RUN_ID,
                    artifact_id=str(entry.get("artifact_id") or ""),
                    sha256=str(entry.get("sha256") or ""),
                    byte_count=int(entry.get("bytes", -1)),
                    expected_name=path.name,
                )
                verified_by_role[str(entry.get("role") or "")] = artifact_path
            ledger = item.get("holdout_ledger") if isinstance(item.get("holdout_ledger"), Mapping) else {}
            for role, schema in (
                ("holdout_ledger", "evomind.holdout-ledger.v1"),
                ("holdout_claim", "evomind.holdout-claim.v1"),
            ):
                artifact_path = verified_by_role.get(role)
                value = _read_json_file(artifact_path, f"{competition} {role}") if artifact_path else None
                if (
                    not isinstance(value, Mapping)
                    or value.get("schema") != schema
                    or value.get("run_id") != FIXED_RUN_ID
                    or value.get("competition") != competition
                    or value.get("ledger_id") != ledger.get("ledger_id")
                    or (role == "holdout_claim" and value.get("solution_id") != item.get("solution_id"))
                    or (role == "holdout_claim" and value.get("claim_id") != ledger.get("claim_id"))
                    or (role == "holdout_claim" and value.get("consumed_once") is not True)
                    or (role == "holdout_claim" and value.get("unconsumed_at_claim") is not True)
                    or (role == "holdout_claim" and value.get("selection_data_scope") != "development_only")
                ):
                    raise GoalRecordError(f"{role} artifact binding is invalid: {competition}")
            review = item.get("independent_review") if isinstance(item.get("independent_review"), Mapping) else {}
            _, review_path = _verified_deliverable(
                store,
                run_id=FIXED_RUN_ID,
                artifact_id=str(review.get("artifact_id") or ""),
                sha256=str(review.get("sha256") or ""),
            )
            review_value = _read_json_file(review_path, f"{competition} independent review")
            if (
                review_value.get("schema") != "evomind.independent-review.v1"
                or review_value.get("run_id") != FIXED_RUN_ID
                or review_value.get("competition") != competition
                or review_value.get("solution_id") != item.get("solution_id")
                or str(review_value.get("artifact_manifest_sha256") or "").lower() != manifest_sha
                or review_value.get("verified") is not True
                or review_value.get("test_labels_used") is not False
                or review_value.get("manifest_closed") is not True
            ):
                raise GoalRecordError(f"independent review artifact binding is invalid: {competition}")
        if competition == "open_polymer" and not legacy_open_polymer:
            adoption = item.get("existing_candidate_adoption")
            if not isinstance(adoption, Mapping):
                raise GoalRecordError("Open Polymer existing-candidate adoption is missing")
            source_call = store.get_tool_call(str(adoption.get("source_tool_call_id") or ""))
            source_arguments = source_call.get("arguments") if isinstance(source_call, Mapping) else None
            source_result = source_call.get("result") if isinstance(source_call, Mapping) else None
            source_content = source_result.get("content") if isinstance(source_result, Mapping) else None
            published = source_content.get("published_artifacts") if isinstance(source_content, Mapping) else None
            source_manifest_sha = str(adoption.get("source_artifact_manifest_sha256") or "").lower()
            manifest_bound = (
                isinstance(source_content, Mapping)
                and str(source_content.get("artifact_manifest_sha256") or "").lower() == source_manifest_sha
            ) or (
                isinstance(published, list)
                and any(
                    isinstance(entry, Mapping)
                    and str(entry.get("sha256") or "").lower() == source_manifest_sha
                    for entry in published
                )
            )
            source_before_goal = False
            if isinstance(source_call, Mapping):
                source_before_goal = _parse_utc(
                    source_call.get("completed_at") or source_call.get("created_at"),
                    "Open Polymer source tool call timestamp is invalid",
                ) < _parse_utc(goal_created_at, "goal.created timestamp is invalid")
            if (
                not isinstance(source_call, Mapping)
                or source_call.get("session_id") != FIXED_RUN_ID
                or source_call.get("tool_name") != "hpc_execute_solution"
                or source_call.get("status") != "completed"
                or not isinstance(source_arguments, Mapping)
                or source_arguments.get("competition") != "open_polymer"
                or source_arguments.get("solution_id") != item.get("solution_id")
                or not source_before_goal
                or not isinstance(source_result, Mapping)
                or source_result.get("ok") is not True
                or not manifest_bound
            ):
                raise GoalRecordError("Open Polymer source tool call binding is invalid")
            window_calls = [
                call
                for call in store.list_tool_calls(FIXED_RUN_ID, limit=2000)
                if str(call.get("created_at") or "") >= goal_created_at
                and isinstance(call.get("arguments"), Mapping)
                and call["arguments"].get("competition") == "open_polymer"
            ]
            if any(
                call.get("tool_name") in {"training_route", "hpc_verify", "hpc_execute_solution"}
                for call in window_calls
            ):
                raise GoalRecordError("Open Polymer adoption cannot retrain in the Goal window")
        elif competition != "open_polymer":
            execution = item.get("managed_execution") if isinstance(item.get("managed_execution"), Mapping) else {}
            bound_calls = [
                store.get_tool_call(str(execution.get(field) or ""))
                for field in (
                    "training_route_tool_call_id",
                    "hpc_verify_tool_call_id",
                    "hpc_execute_tool_call_id",
                )
            ]
            if any(not isinstance(call, Mapping) for call in bound_calls):
                raise GoalRecordError(f"managed execution tool call is missing: {competition}")
            route_call, verify_call, execute_call = bound_calls
            call_times = [
                _parse_utc(call.get("created_at"), f"managed execution timestamp invalid: {competition}")
                for call in bound_calls
            ]
            goal_time = _parse_utc(goal_created_at, "goal.created timestamp is invalid")
            arguments = [call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {} for call in bound_calls]
            results = [call.get("result") if isinstance(call.get("result"), Mapping) else {} for call in bound_calls]
            execute_content = results[2].get("content") if isinstance(results[2].get("content"), Mapping) else {}
            published = execute_content.get("published_artifacts") if isinstance(execute_content.get("published_artifacts"), list) else []
            execute_manifest_bound = str(execute_content.get("artifact_manifest_sha256") or "").lower() == manifest_sha or any(
                isinstance(entry, Mapping) and str(entry.get("sha256") or "").lower() == manifest_sha
                for entry in published
            )
            if (
                [call.get("tool_name") for call in bound_calls]
                != ["training_route", "hpc_verify", "hpc_execute_solution"]
                or any(call.get("session_id") != FIXED_RUN_ID for call in bound_calls)
                or any(call.get("status") != "completed" for call in bound_calls)
                or not (goal_time <= call_times[0] < call_times[1] < call_times[2])
                or arguments[0].get("competition") != competition
                or "solution_id" in arguments[0]
                or arguments[1]
                or arguments[2].get("competition") != competition
                or arguments[2].get("solution_id") != item.get("solution_id")
                or not _hpc_verify_complete(results[1])
                or results[2].get("ok") is not True
                or not execute_manifest_bound
                or len({str(call.get("idempotency_key") or "") for call in bound_calls}) != 3
                or any(not str(call.get("idempotency_key") or "") for call in bound_calls)
            ):
                raise GoalRecordError(f"managed execution binding is invalid: {competition}")
    return next_human_baseline_sha


def ensure_fixed_goal(
    store: RuntimeStore,
    *,
    run_id: str,
    spec: Mapping[str, Any],
    board: Mapping[str, Any],
    human_baseline_sha256: str = "",
    status: str = "blocked",
    goal_id: str = FIXED_GOAL_ID,
    goal_key: str = FIXED_GOAL_KEY,
    allocation: str = FIXED_ALLOCATION,
) -> dict[str, Any]:
    """Create or reuse the one fixed G21 Goal for an already-existing Run."""

    human_baseline_sha256 = human_baseline_sha256 or str(spec.get("human_baseline_evidence_sha256") or "")
    if run_id != FIXED_RUN_ID:
        raise GoalRecordError("goal run must be the fixed G21 Run")
    if allocation.casefold() != FIXED_ALLOCATION.casefold():
        raise GoalRecordError("goal allocation must be G21")
    if goal_id != FIXED_GOAL_ID or goal_key != FIXED_GOAL_KEY:
        raise GoalRecordError("goal identity is not the fixed G21 goal")
    if str(status).casefold() != "blocked":
        raise GoalRecordError("the fixed Goal must be bootstrapped in blocked status")
    if any(
        isinstance(item, Mapping) and item.get("goal_status") == "VERIFIED_COMPLETE"
        for item in (board.get("competitions") if isinstance(board.get("competitions"), list) else [])
    ):
        raise GoalRecordError("the fixed Goal cannot be bootstrapped as completed")
    declared_human_baseline_sha = str(spec.get("human_baseline_evidence_sha256") or "")
    if human_baseline_sha256 != declared_human_baseline_sha:
        raise GoalRecordError("human baseline evidence hash is not bound to the goal spec")
    spec_ok, spec_issues = validate_goal_spec(spec)
    if not spec_ok:
        raise GoalRecordError(_issue_text("spec:", spec_issues))
    board_ok, board_issues = validate_goal_board_identity(
        board,
        run_id=run_id,
        allocation=allocation,
    )
    if not board_ok:
        raise GoalRecordError(_issue_text("board:", board_issues))
    now = utc_now()
    spec_hash = sha256_json(spec)
    board_hash = sha256_json(board)
    candidate = {
        "schema": GOAL_RECORD_SCHEMA,
        "goal_id": goal_id,
        "run_id": run_id,
        "allocation": allocation,
        "goal_key": goal_key,
        "status": status,
        "spec_sha256": spec_hash,
        "spec": dict(spec),
        "board_sha256": board_hash,
        "human_baseline_sha256": human_baseline_sha256,
        "board": dict(board),
    }
    record_check = validate_goal_record(candidate)
    if not record_check.valid:
        raise GoalRecordError(_issue_text("record:", record_check.issues))
    payload = {
        "id": goal_id,
        "run_id": run_id,
        "goal_key": goal_key,
        "schema": GOAL_RECORD_SCHEMA,
        "allocation": allocation,
        "spec_sha256": spec_hash,
        "spec_json": _json_text(spec),
        "board_sha256": board_hash,
        "human_baseline_sha256": human_baseline_sha256,
        "board_json": _json_text(board),
        "status": status,
        "created_at": now,
        "updated_at": now,
    }
    record, created = store.ensure_goal_record(
        payload,
        event_payload={
            "goal_id": goal_id,
            "goal_key": goal_key,
            "allocation": allocation,
            "spec_sha256": spec_hash,
            "board_sha256": board_hash,
        },
        session_metadata={
            "goal_id": goal_id,
            "goal_key": goal_key,
            "goal_spec_sha256": spec_hash,
            "goal_board_sha256": board_hash,
            "goal_human_baseline_sha256": human_baseline_sha256,
        },
    )
    return {"record": record, "created": created}


def update_fixed_goal_board(
    store: RuntimeStore,
    *,
    goal_id: str,
    board: Mapping[str, Any],
    status: str | None = None,
    session_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update only the hash-bound board/status of an existing fixed Goal."""

    current = store.get_goal_record(goal_id)
    if current is None:
        raise KeyError(goal_id)
    if str(current.get("goal_id") or current.get("id")) != FIXED_GOAL_ID:
        raise GoalRecordError("goal identity is not the fixed G21 goal")
    board_ok, board_issues = validate_goal_board_identity(
        board,
        run_id=FIXED_RUN_ID,
        allocation=FIXED_ALLOCATION,
    )
    if not board_ok:
        raise GoalRecordError(_issue_text("board:", board_issues))
    next_status = str(status or current.get("status") or "active")
    next_human_baseline_sha = _verify_completed_board_artifacts(
        store,
        board=board,
        current_board=current.get("board") if isinstance(current.get("board"), Mapping) else None,
        current_human_baseline_sha256=str(current.get("human_baseline_sha256") or ""),
        next_status=next_status,
        spec=current.get("spec") if isinstance(current.get("spec"), Mapping) else {},
    )
    candidate = {
        "schema": current.get("schema"),
        "goal_id": current.get("goal_id") or current.get("id"),
        "run_id": current.get("run_id"),
        "allocation": current.get("allocation"),
        "goal_key": current.get("goal_key"),
        "status": next_status,
        "spec_sha256": current.get("spec_sha256"),
        "spec": current.get("spec"),
        "board_sha256": sha256_json(board),
        "human_baseline_sha256": next_human_baseline_sha,
        "board": dict(board),
    }
    check = validate_goal_record(candidate)
    if not check.valid:
        raise GoalRecordError(_issue_text("record:", check.issues))
    next_metadata = session_metadata or {
        "goal_id": FIXED_GOAL_ID,
        "goal_key": FIXED_GOAL_KEY,
        "goal_spec_sha256": str(current.get("spec_sha256") or ""),
        "goal_board_sha256": check.board_sha256,
        "goal_human_baseline_sha256": next_human_baseline_sha,
    }
    return store.update_goal_record(
        goal_id,
        status=next_status,
        board=dict(board),
        board_sha256=check.board_sha256,
        human_baseline_sha256=next_human_baseline_sha,
        session_metadata=next_metadata,
    )


__all__ = ["GoalRecordError", "ensure_fixed_goal", "update_fixed_goal_board"]
