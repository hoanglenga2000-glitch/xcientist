from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evomind_runtime.competition_goal import (
    FIXED_GOAL_ID,
    FIXED_GOAL_KEY,
    FIXED_RUN_ID,
    GOAL_RECORD_SCHEMA,
    CONDITIONAL_BASELINE_POLICY_MODE,
    OPEN_POLYMER_LEGACY_ADOPTION_SCHEMA,
    OPEN_POLYMER_LEGACY_COMPLETED_EVENT_SEQ,
    OPEN_POLYMER_LEGACY_IDEMPOTENCY_KEY,
    OPEN_POLYMER_LEGACY_MANIFEST_ARTIFACT_ID,
    OPEN_POLYMER_LEGACY_MANIFEST_RECEIPT_BYTES,
    OPEN_POLYMER_LEGACY_MANIFEST_RECEIPT_ARTIFACT_ID,
    OPEN_POLYMER_LEGACY_MANIFEST_RECEIPT_SHA256,
    OPEN_POLYMER_LEGACY_MANIFEST_SHA256,
    OPEN_POLYMER_LEGACY_REVIEW_ARTIFACT_ID,
    OPEN_POLYMER_LEGACY_REVIEW_SHA256,
    OPEN_POLYMER_LEGACY_REQUIRED_FILES,
    OPEN_POLYMER_LEGACY_SOLUTION_ID,
    OPEN_POLYMER_LEGACY_SOURCE_CALL_ID,
    OPEN_POLYMER_LEGACY_SPLIT_ARTIFACT_ID,
    OPEN_POLYMER_LEGACY_SPLIT_SHA256,
    OPEN_POLYMER_LEGACY_STARTED_EVENT_SEQ,
    canonical_json_bytes,
    conditional_baseline_policy,
    sha256_json,
    validate_goal_board,
    validate_human_baseline,
    validate_goal_record,
    validate_serial_action_sequence,
    validate_goal_spec,
)


COMPETITIONS = ("cure_bench", "e2lmc", "mindgames", "open_polymer", "ariel_2025")
PROHIBITED_TOOLS = {
    "competition_data_prepare",
    "competition_data_accelerate",
    "kaggle_download",
    "kaggle_submit",
    "join_competition",
    "accept_terms",
}
UNSETTLED_STATUSES = frozenset({"requested", "starting", "started", "running", "waiting_approval", "pending", "paused"})
TERMINAL_RUN_STATUSES = frozenset({"completed", "blocked", "failed", "cancelled"})
MANAGED_SEQUENCE_TOOLS = frozenset(
    {"training_route", "hpc_verify", "hpc_execute_solution"}
)
NONEXISTENT_EVIDENCE_STAGE_TOOLS = frozenset({"artifact_collect", "independent_verify"})


def _hpc_verify_complete(result: dict[str, Any]) -> bool:
    content = result.get("content") if isinstance(result.get("content"), dict) else {}
    if result.get("ok") is not True and content.get("ok") is not True:
        return False
    evidence = content.get("hpc_identity") if isinstance(content.get("hpc_identity"), dict) else content
    samples = evidence.get("samples")
    return (
        evidence.get("samples_requested") == 5
        and evidence.get("samples_passed") == 5
        and isinstance(samples, list)
        and len(samples) == 5
        and all(isinstance(sample, dict) and sample.get("complete") is True for sample in samples)
    )


def parse_utc(value: str, *, field: str = "timestamp") -> datetime:
    raw = str(value or "").strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _timestamp_at_or_after(value: Any, boundary: datetime) -> bool:
    try:
        return parse_utc(str(value or ""), field="tool_call.created_at") >= boundary
    except ValueError:
        return False


def effective_active_tool_calls(
    calls: list[dict[str, Any]],
    *,
    approvals_by_id: dict[str, dict[str, Any]],
    latest_assistant_turn_at: str = "",
) -> list[dict[str, Any]]:
    """Mirror the managed Assistant active projection without raw-history inflation."""

    completed_by_name: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    completed_downloads: list[dict[str, Any]] = []
    for item in calls:
        if str(item.get("status") or "").casefold() != "completed":
            continue
        name = str(item.get("tool_name") or "")
        completed_by_name[name].append(item)
        if name == "kaggle_download":
            completed_downloads.append(item)
    active: list[dict[str, Any]] = []
    for item in calls:
        status = str(item.get("status") or "").casefold()
        if status not in {"running", "waiting_approval"}:
            continue
        if status == "waiting_approval":
            approval_id = str(item.get("approval_id") or "")
            approval = approvals_by_id.get(approval_id)
            if approval and str(approval.get("status") or "").casefold() in {"expired", "rejected"}:
                continue
            active.append(item)
            continue
        name = str(item.get("tool_name") or "")
        started_at = str(item.get("started_at") or item.get("created_at") or "")
        if latest_assistant_turn_at and latest_assistant_turn_at > started_at:
            continue
        successors = list(completed_by_name.get(name, ()))
        if name == "kaggle_list":
            successors.extend(completed_downloads)
        if any(str(candidate.get("completed_at") or "") > started_at for candidate in successors):
            continue
        active.append(item)
    return active


def _effective_prohibited_call(
    item: dict[str, Any], *, approvals_by_id: dict[str, dict[str, Any]]
) -> bool:
    status = str(item.get("status") or "").casefold()
    if status in {"cancelled", "canceled", "expired", "rejected"}:
        return False
    if status == "waiting_approval":
        approval = approvals_by_id.get(str(item.get("approval_id") or ""))
        return not approval or str(approval.get("status") or "").casefold() not in {"expired", "rejected"}
    return status in {"requested", "starting", "started", "running", "completed", "failed"}


def goal_unsettled_calls_since(
    calls: list[dict[str, Any]],
    *,
    since: datetime,
    approvals_by_id: dict[str, dict[str, Any]],
    latest_assistant_turn_at: str = "",
) -> list[dict[str, Any]]:
    """Project unresolved Goal-window calls while ignoring settled historical rows."""

    completed_by_name: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in calls:
        if str(item.get("status") or "").casefold() == "completed":
            completed_by_name[str(item.get("tool_name") or "")].append(item)
    unsettled: list[dict[str, Any]] = []
    for item in calls:
        if not _timestamp_at_or_after(item.get("created_at"), since):
            continue
        status = str(item.get("status") or "").casefold()
        if status not in UNSETTLED_STATUSES:
            continue
        approval = approvals_by_id.get(str(item.get("approval_id") or ""))
        if approval and str(approval.get("status") or "").casefold() in {"expired", "rejected"}:
            continue
        started_at = str(item.get("started_at") or item.get("created_at") or "")
        if latest_assistant_turn_at and latest_assistant_turn_at > started_at:
            continue
        successors = completed_by_name.get(str(item.get("tool_name") or ""), [])
        if any(str(candidate.get("completed_at") or "") > started_at for candidate in successors):
            continue
        unsettled.append(item)
    return unsettled


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def parse_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    return parsed if isinstance(parsed, dict) else {}


def load_json_file(path: Path) -> tuple[dict[str, Any], str]:
    data = path.read_bytes()
    parsed = json.loads(data.decode("utf-8-sig"))
    if not isinstance(parsed, dict):
        raise ValueError("JSON root must be an object")
    return parsed, hashlib.sha256(data).hexdigest()


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def _goal_record_projection(row: sqlite3.Row) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = dict(row)
    try:
        spec = json.loads(str(raw.get("spec_json") or "{}"))
        board = json.loads(str(raw.get("board_json") or "{}"))
    except json.JSONDecodeError:
        spec, board = {}, {}
    candidate = {
        "schema": raw.get("schema", ""),
        "goal_id": raw.get("id", ""),
        "run_id": raw.get("run_id", ""),
        "allocation": raw.get("allocation", ""),
        "goal_key": raw.get("goal_key", ""),
        "status": raw.get("status", ""),
        "spec_sha256": raw.get("spec_sha256", ""),
        "spec": spec,
        "board_sha256": raw.get("board_sha256", ""),
        "human_baseline_sha256": raw.get("human_baseline_sha256", ""),
        "board": board,
    }
    public = {
        "id": str(raw.get("id") or ""),
        "run_id": str(raw.get("run_id") or ""),
        "goal_key": str(raw.get("goal_key") or ""),
        "schema": str(raw.get("schema") or ""),
        "allocation": str(raw.get("allocation") or ""),
        "status": str(raw.get("status") or ""),
        "spec_sha256": str(raw.get("spec_sha256") or ""),
        "board_sha256": str(raw.get("board_sha256") or ""),
        "human_baseline_sha256": str(raw.get("human_baseline_sha256") or ""),
        "created_at": str(raw.get("created_at") or ""),
        "updated_at": str(raw.get("updated_at") or ""),
    }
    return candidate, public


def inspect_goal_records(connection: sqlite3.Connection, *, run_id: str) -> dict[str, Any]:
    """Read the fixed Goal record without mutating the runtime database."""

    if not _table_exists(connection, "goal_records"):
        return {
            "present": False,
            "valid": False,
            "count": 0,
            "record": None,
            "issues": [{"code": "GOAL_RECORD_TABLE_MISSING", "detail": ""}],
            "duplicate_goal_records": [],
            "unbound_goal_records": [],
        }
    schema_issues: list[dict[str, str]] = []
    try:
        table_info = connection.execute("PRAGMA table_info(goal_records)").fetchall()
        columns = {str(row[1]) for row in table_info}
        required_columns = {
            "id", "run_id", "goal_key", "schema", "allocation", "spec_sha256", "spec_json",
            "board_sha256", "human_baseline_sha256", "board_json", "status", "created_at", "updated_at",
        }
        if required_columns - columns:
            schema_issues.append({"code": "GOAL_RECORD_COLUMNS_INCOMPLETE", "detail": ""})
        by_name = {str(row[1]): row for row in table_info}
        if "id" in by_name and int(by_name["id"][5]) != 1:
            schema_issues.append({"code": "GOAL_RECORD_ID_NOT_PRIMARY_KEY", "detail": ""})
        if any(
            name in by_name and int(by_name[name][3]) != 1
            for name in required_columns - {"id"}
        ):
            schema_issues.append({"code": "GOAL_RECORD_REQUIRED_COLUMNS_NULLABLE", "detail": ""})
        unique_goal_key = False
        for index in connection.execute("PRAGMA index_list(goal_records)").fetchall():
            if int(index[2]) != 1:
                continue
            index_columns = [
                str(row[2])
                for row in connection.execute(f"PRAGMA index_info([{index[1]}])").fetchall()
            ]
            if index_columns == ["goal_key"]:
                unique_goal_key = True
        if not unique_goal_key:
            schema_issues.append({"code": "GOAL_RECORD_GOAL_KEY_NOT_UNIQUE", "detail": ""})
        foreign_keys = connection.execute("PRAGMA foreign_key_list(goal_records)").fetchall()
        if not any(
            str(row[2]) == "assistant_runs"
            and str(row[3]) == "run_id"
            and str(row[4]) == "id"
            for row in foreign_keys
        ):
            schema_issues.append({"code": "GOAL_RECORD_RUN_FOREIGN_KEY_MISSING", "detail": ""})
        rows = connection.execute("SELECT * FROM goal_records ORDER BY created_at").fetchall()
    except sqlite3.Error as exc:
        return {
            "present": False,
            "valid": False,
            "count": 0,
            "record": None,
            "issues": [{"code": "GOAL_RECORD_QUERY_FAILED", "detail": type(exc).__name__}],
            "duplicate_goal_records": [],
            "unbound_goal_records": [],
            "extra_goal_records": [],
            "schema_issues": schema_issues,
        }
    records: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = list(schema_issues)
    unbound: list[dict[str, Any]] = []
    extras: list[dict[str, Any]] = []
    for row in rows:
        candidate, public = _goal_record_projection(row)
        relevant = (
            public["id"] == FIXED_GOAL_ID
            or public["goal_key"] == FIXED_GOAL_KEY
            or public["run_id"] == run_id
        )
        if not relevant:
            continue
        checked = validate_goal_record(candidate, run_id=run_id)
        records.append({"record": public, "valid": checked.valid, "issues": [item.to_dict() for item in checked.issues]})
        exact = (
            public["id"] == FIXED_GOAL_ID
            and public["goal_key"] == FIXED_GOAL_KEY
            and public["run_id"] == run_id
        )
        if not exact:
            extras.append(public)
        if public["id"] == FIXED_GOAL_ID or public["goal_key"] == FIXED_GOAL_KEY:
            if public["run_id"] != run_id:
                unbound.append(public)
        if not checked.valid and exact:
            issues.extend({"code": item.code, "detail": f"{public['id']}:{item.detail}"} for item in checked.issues)
    fixed_records = [
        item for item in records
        if item["record"]["id"] == FIXED_GOAL_ID or item["record"]["goal_key"] == FIXED_GOAL_KEY
    ]
    duplicate = [item["record"] for item in fixed_records[1:]] if len(fixed_records) > 1 else []
    if duplicate:
        issues.append({"code": "GOAL_RECORD_DUPLICATE", "detail": str(len(fixed_records))})
    if extras:
        issues.append({"code": "GOAL_RECORD_WRONG_BINDING", "detail": str(len(extras))})
    valid_records = [
        item for item in fixed_records
        if item["valid"]
        and item["record"]["id"] == FIXED_GOAL_ID
        and item["record"]["goal_key"] == FIXED_GOAL_KEY
        and item["record"]["run_id"] == run_id
    ]
    return {
        "present": bool(records),
        "valid": (
            len(valid_records) == 1
            and not duplicate
            and not unbound
            and not extras
            and not schema_issues
            and not issues
        ),
        "count": len(records),
        "record": valid_records[0]["record"] if len(valid_records) == 1 else None,
        "issues": issues,
        "duplicate_goal_records": duplicate,
        "unbound_goal_records": unbound,
        "extra_goal_records": extras,
        "schema_issues": schema_issues,
    }


def human_baseline_gate_summary(
    value: Any,
    *,
    run_id: str,
    expected_metrics: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Project the source audit without treating negative evidence as a pass."""

    if not isinstance(value, dict):
        return {"valid": False, "issues": [{"code": "HUMAN_BASELINE_AUDIT_INVALID"}]}
    issues: list[dict[str, str]] = []
    if value.get("schema") != "evomind.five_competition_human_baseline_audit.v1":
        issues.append({"code": "HUMAN_BASELINE_AUDIT_SCHEMA_INVALID", "detail": ""})
    if str(value.get("run_id") or "") != run_id:
        issues.append({"code": "HUMAN_BASELINE_AUDIT_RUN_MISMATCH", "detail": ""})
    if str(value.get("allocation") or "").casefold() != "g21":
        issues.append({"code": "HUMAN_BASELINE_AUDIT_ALLOCATION_MISMATCH", "detail": ""})
    classification = value.get("classification")
    if classification not in {
        "HUMAN_BASELINE_UNDEFINED",
        "HUMAN_BASELINE_PARTIAL",
        "HUMAN_BASELINE_VERIFIED",
    }:
        issues.append({"code": "HUMAN_BASELINE_AUDIT_CLASSIFICATION_INVALID", "detail": ""})
    entries = value.get("competitions")
    if not isinstance(entries, dict) or set(entries) != set(COMPETITIONS):
        issues.append({"code": "HUMAN_BASELINE_AUDIT_SCOPE_INVALID", "detail": ""})
    else:
        statuses: list[str] = []
        for competition in COMPETITIONS:
            item = entries.get(competition)
            status = str(item.get("status") or "") if isinstance(item, dict) else ""
            statuses.append(status)
            allowed_statuses = (
                {"VERIFIED"}
                if classification == "HUMAN_BASELINE_VERIFIED"
                else {"HUMAN_BASELINE_UNDEFINED"}
                if classification == "HUMAN_BASELINE_UNDEFINED"
                else {"VERIFIED", "HUMAN_BASELINE_UNDEFINED"}
            )
            if not isinstance(item, dict) or status not in allowed_statuses:
                issues.append({"code": "HUMAN_BASELINE_AUDIT_ITEM_INVALID", "detail": competition})
            elif status == "HUMAN_BASELINE_UNDEFINED" and not str(item.get("unique_gate") or "").strip():
                issues.append({"code": "HUMAN_BASELINE_AUDIT_UNIQUE_GATE_MISSING", "detail": competition})
            elif status == "VERIFIED":
                baseline = item.get("baseline") or item.get("human_baseline")
                if not isinstance(baseline, dict):
                    issues.append({"code": "HUMAN_BASELINE_AUDIT_DETAIL_MISSING", "detail": competition})
                else:
                    expected = (expected_metrics or {}).get(competition, {})
                    item_metric = str(item.get("metric") or baseline.get("metric") or "")
                    item_direction = str(item.get("direction") or baseline.get("direction") or "")
                    if expected:
                        if item_metric != expected.get("metric") or item_direction != expected.get("direction"):
                            issues.append({"code": "HUMAN_BASELINE_AUDIT_METRIC_BINDING_MISMATCH", "detail": competition})
                    if not item_metric or not item_direction:
                        issues.append({"code": "HUMAN_BASELINE_AUDIT_METRIC_BINDING_MISSING", "detail": competition})
                    checked = validate_human_baseline(
                        baseline,
                        expected_metric=item_metric,
                        expected_direction=item_direction,
                    )
                    issues.extend(
                        {"code": f"{competition}:{issue.code}", "detail": issue.detail}
                        for issue in checked.issues
                    )
        if classification == "HUMAN_BASELINE_PARTIAL" and (
            "VERIFIED" not in statuses or "HUMAN_BASELINE_UNDEFINED" not in statuses
        ):
            issues.append({"code": "HUMAN_BASELINE_AUDIT_PARTIAL_STATUS_INVALID", "detail": ""})
    completion = value.get("completion") if isinstance(value.get("completion"), dict) else {}
    expected_completion = classification == "HUMAN_BASELINE_VERIFIED"
    if (
        completion.get("human_baseline_gate_passed") is not expected_completion
        or completion.get("all_five_verified") is not expected_completion
    ):
        issues.append({"code": "HUMAN_BASELINE_AUDIT_COMPLETION_FLAGS_INVALID", "detail": ""})
    return {
        "valid": not issues,
        "classification": value.get("classification"),
        "completion_eligible": (
            not issues
            and classification == "HUMAN_BASELINE_VERIFIED"
            and completion.get("human_baseline_gate_passed") is True
            and completion.get("all_five_verified") is True
        ),
        "issues": issues,
    }


def audit_goal_board_artifacts(
    board: dict[str, Any], deliverables: list[dict[str, Any]]
) -> dict[str, Any]:
    """Require every completion artifact reference to resolve to a current Run file."""

    by_id = {str(item.get("id") or ""): item for item in deliverables if str(item.get("id") or "")}
    issues: list[dict[str, str]] = []
    referenced: set[str] = set()

    def require(artifact_id: str, sha256: str, byte_count: int | None, detail: str) -> dict[str, Any] | None:
        referenced.add(artifact_id)
        row = by_id.get(artifact_id)
        if (
            not artifact_id
            or row is None
            or str(row.get("sha256") or "").lower() != str(sha256 or "").lower()
            or (byte_count is not None and int(row.get("bytes", -1)) != int(byte_count))
            or row.get("current_file_verified") is not True
        ):
            issues.append({"code": "GOAL_ARTIFACT_REFERENCE_INVALID", "detail": detail})
            return None
        return row

    completed = [
        item
        for item in board.get("competitions", [])
        if isinstance(item, dict) and item.get("goal_status") == "VERIFIED_COMPLETE"
    ]
    conditional_policy = board.get("baseline_policy") == CONDITIONAL_BASELINE_POLICY_MODE
    completed_requiring_human = [
        item for item in completed
        if isinstance(item.get("human_baseline"), dict) or isinstance(item.get("human_baseline_evidence"), dict)
    ]
    if completed and (not conditional_policy or completed_requiring_human):
        require(
            str(board.get("human_baseline_evidence_artifact_id") or ""),
            str(board.get("human_baseline_evidence_sha256") or ""),
            None,
            "human_baseline_evidence",
        )
    for item in completed:
        competition = str(item.get("competition") or "")
        manifest = item.get("artifact_manifest") if isinstance(item.get("artifact_manifest"), dict) else {}
        adoption = item.get("existing_candidate_adoption") if isinstance(item.get("existing_candidate_adoption"), dict) else {}
        legacy_open_polymer = (
            competition == "open_polymer"
            and adoption.get("schema") == OPEN_POLYMER_LEGACY_ADOPTION_SCHEMA
        )
        if conditional_policy and not (
            isinstance(item.get("human_baseline"), dict)
            or isinstance(item.get("human_baseline_evidence"), dict)
        ):
            availability = item.get("human_baseline_availability") if isinstance(item.get("human_baseline_availability"), dict) else {}
            require(
                str(availability.get("evidence_artifact_id") or ""),
                str(availability.get("evidence_artifact_sha256") or ""),
                None,
                f"{competition}:human_baseline_availability",
            )
        if legacy_open_polymer:
            manifest_row = require(
                str(item.get("artifact_manifest_artifact_id") or ""),
                str(item.get("artifact_manifest_sha256") or ""),
                None,
                f"{competition}:legacy_artifact_manifest",
            )
            if manifest_row is not None:
                try:
                    published_manifest = json.loads(
                        Path(str(manifest_row.get("path") or "")).read_text(encoding="utf-8-sig")
                    )
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    published_manifest = None
                if published_manifest != manifest:
                    issues.append({"code": "OPEN_POLYMER_LEGACY_MANIFEST_CONTENT_MISMATCH", "detail": competition})
            bindings = item.get("legacy_artifact_bindings") if isinstance(item.get("legacy_artifact_bindings"), dict) else {}
            for entry in manifest.get("files") or []:
                if not isinstance(entry, dict):
                    issues.append({"code": "OPEN_POLYMER_LEGACY_MANIFEST_ENTRY_INVALID", "detail": competition})
                    continue
                name = str(entry.get("name") or "")
                require(
                    str(bindings.get(name) or ""),
                    str(entry.get("sha256") or ""),
                    int(entry.get("bytes", -1)),
                    f"{competition}:{name or 'legacy_entry'}",
                )
            review = item.get("independent_review") if isinstance(item.get("independent_review"), dict) else {}
            review_row = require(
                str(review.get("artifact_id") or ""),
                str(review.get("sha256") or ""),
                None,
                f"{competition}:legacy_independent_review",
            )
            if review_row is not None:
                try:
                    review_value = json.loads(
                        Path(str(review_row.get("path") or "")).read_text(encoding="utf-8-sig")
                    )
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    review_value = None
                if (
                    not isinstance(review_value, dict)
                    or review_value.get("schema") != "evomind.open_polymer.independent_verification.v2"
                    or review_value.get("baseline_match") is not True
                    or review_value.get("candidate_match") is not True
                    or review_value.get("group_overlap") != 0
                    or review_value.get("test_labels_used") is not False
                    or review_value.get("gate_passed") is not True
                ):
                    issues.append({"code": "OPEN_POLYMER_LEGACY_REVIEW_INVALID", "detail": competition})
            continue
        manifest_bytes = canonical_json_bytes(manifest)
        require(
            str(item.get("artifact_manifest_artifact_id") or ""),
            hashlib.sha256(manifest_bytes).hexdigest(),
            len(manifest_bytes),
            f"{competition}:artifact_manifest",
        )
        for entry in manifest.get("files") or []:
            if not isinstance(entry, dict):
                issues.append({"code": "GOAL_ARTIFACT_MANIFEST_ENTRY_INVALID", "detail": competition})
                continue
            require(
                str(entry.get("artifact_id") or ""),
                str(entry.get("sha256") or ""),
                int(entry.get("bytes", -1)),
                f"{competition}:{entry.get('role') or entry.get('path') or 'entry'}",
            )
        review = item.get("independent_review") if isinstance(item.get("independent_review"), dict) else {}
        review_row = require(
            str(review.get("artifact_id") or ""),
            str(review.get("sha256") or ""),
            None,
            f"{competition}:independent_review",
        )
        if review_row is not None:
            try:
                review_value = json.loads(Path(str(review_row.get("path") or "")).read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                review_value = None
            if (
                not isinstance(review_value, dict)
                or review_value.get("schema") != "evomind.independent-review.v1"
                or review_value.get("run_id") != FIXED_RUN_ID
                or review_value.get("competition") != competition
                or review_value.get("solution_id") != item.get("solution_id")
                or str(review_value.get("artifact_manifest_sha256") or "").lower()
                != str(item.get("artifact_manifest_sha256") or "").lower()
                or review_value.get("verified") is not True
                or review_value.get("test_labels_used") is not False
                or review_value.get("manifest_closed") is not True
            ):
                issues.append({"code": "INDEPENDENT_REVIEW_ARTIFACT_INVALID", "detail": competition})
    return {
        "valid": not issues,
        "referenced_artifact_ids": sorted(referenced),
        "issues": issues,
    }


def validate_open_polymer_adoption(
    *,
    board_item: dict[str, Any],
    all_calls: list[dict[str, Any]],
    since: datetime,
    all_events: list[dict[str, Any]] | None = None,
) -> tuple[bool, list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    adoption = board_item.get("existing_candidate_adoption")
    if not isinstance(adoption, dict) or adoption.get("schema") not in {
        "evomind.existing-candidate-adoption.v1",
        OPEN_POLYMER_LEGACY_ADOPTION_SCHEMA,
    }:
        return False, [{"code": "OPEN_POLYMER_ADOPTION_SCHEMA_INVALID", "detail": ""}]
    solution_id = str(board_item.get("solution_id") or "")
    source_call_id = str(adoption.get("source_tool_call_id") or "")
    source_manifest_sha = str(adoption.get("source_artifact_manifest_sha256") or "").lower()
    legacy_v2 = adoption.get("schema") == OPEN_POLYMER_LEGACY_ADOPTION_SCHEMA
    if adoption.get("retraining_performed") is not False:
        issues.append({"code": "OPEN_POLYMER_RETRAINING_NOT_FALSE", "detail": ""})
    if str(adoption.get("source_solution_id") or "") != solution_id:
        issues.append({"code": "OPEN_POLYMER_SOURCE_SOLUTION_MISMATCH", "detail": ""})
    sources = [item for item in all_calls if str(item.get("id") or "") == source_call_id]
    if legacy_v2:
        if (
            source_call_id != OPEN_POLYMER_LEGACY_SOURCE_CALL_ID
            or adoption.get("source_idempotency_key") != OPEN_POLYMER_LEGACY_IDEMPOTENCY_KEY
            or adoption.get("mode") != "legacy_pre_goal_completed_candidate"
            or adoption.get("run_id") != FIXED_RUN_ID
            or str(adoption.get("allocation") or "").casefold() != "g21"
            or adoption.get("competition") != "open_polymer"
            or adoption.get("source_solution_id") != OPEN_POLYMER_LEGACY_SOLUTION_ID
            or adoption.get("source_started_event_seq") != OPEN_POLYMER_LEGACY_STARTED_EVENT_SEQ
            or adoption.get("source_completed_event_seq") != OPEN_POLYMER_LEGACY_COMPLETED_EVENT_SEQ
            or adoption.get("source_artifact_manifest_artifact_id")
            != OPEN_POLYMER_LEGACY_MANIFEST_ARTIFACT_ID
            or str(adoption.get("source_artifact_manifest_sha256") or "").lower()
            != OPEN_POLYMER_LEGACY_MANIFEST_SHA256
            or adoption.get("source_split_manifest_artifact_id")
            != OPEN_POLYMER_LEGACY_SPLIT_ARTIFACT_ID
            or str(adoption.get("source_split_manifest_sha256") or "").lower()
            != OPEN_POLYMER_LEGACY_SPLIT_SHA256
            or adoption.get("source_independent_verification_artifact_id")
            != OPEN_POLYMER_LEGACY_REVIEW_ARTIFACT_ID
            or str(adoption.get("source_independent_verification_sha256") or "").lower()
            != OPEN_POLYMER_LEGACY_REVIEW_SHA256
            or adoption.get("historical_atomic_holdout_claim_present") is not False
            or adoption.get("historical_ledger_fabricated") is not False
            or adoption.get("retraining_performed") is not False
            or board_item.get("official_metric_claimed") is not False
            or board_item.get("protocol_scope") != "publicly_reproducible_internal_protocol"
            or board_item.get("holdout_ledger")
            or board_item.get("holdout_claim")
        ):
            issues.append({"code": "OPEN_POLYMER_LEGACY_EXACT_ALLOWLIST_INVALID", "detail": source_call_id})
        idempotency_matches = [
            item
            for item in all_calls
            if str(item.get("idempotency_key") or "") == OPEN_POLYMER_LEGACY_IDEMPOTENCY_KEY
        ]
        if len(idempotency_matches) != 1 or str(idempotency_matches[0].get("id") or "") != source_call_id:
            issues.append({"code": "OPEN_POLYMER_SOURCE_IDEMPOTENCY_NOT_UNIQUE", "detail": source_call_id})
    if len(sources) != 1:
        issues.append({"code": "OPEN_POLYMER_SOURCE_CALL_COUNT_INVALID", "detail": source_call_id})
    else:
        source = sources[0]
        arguments = source.get("arguments") if isinstance(source.get("arguments"), dict) else {}
        result = source.get("result") if isinstance(source.get("result"), dict) else {}
        content = result.get("content") if isinstance(result.get("content"), dict) else {}
        published = content.get("published_artifacts") if isinstance(content.get("published_artifacts"), list) else []
        manifest_bound = str(content.get("artifact_manifest_sha256") or "").lower() == source_manifest_sha or any(
            isinstance(entry, dict) and str(entry.get("sha256") or "").lower() == source_manifest_sha
            for entry in published
        )
        source_started: datetime | None = None
        source_completed: datetime | None = None
        try:
            source_started = parse_utc(
                str(source.get("started_at") or source.get("created_at") or ""),
                field="open_polymer.source.started_at",
            )
            source_completed = parse_utc(
                str(source.get("completed_at") or ""),
                field="open_polymer.source.completed_at",
            )
            source_before_goal = source_started < source_completed < since
        except ValueError:
            source_before_goal = False
            issues.append({"code": "OPEN_POLYMER_SOURCE_TIMESTAMP_INVALID", "detail": source_call_id})
        if (
            source.get("tool_name") != "hpc_execute_solution"
            or (legacy_v2 and source.get("session_id") != FIXED_RUN_ID)
            or str(source.get("status") or "").casefold() != "completed"
            or arguments.get("competition") != "open_polymer"
            or arguments.get("solution_id") != solution_id
            or result.get("ok") is not True
            or (legacy_v2 and content.get("status") != "completed")
            or (legacy_v2 and content.get("run_id") not in {None, "", FIXED_RUN_ID})
            or (legacy_v2 and content.get("solution_id") not in {None, "", OPEN_POLYMER_LEGACY_SOLUTION_ID})
            or (legacy_v2 and content.get("exit_code") != 0)
            or (legacy_v2 and source.get("idempotency_key") != OPEN_POLYMER_LEGACY_IDEMPOTENCY_KEY)
            or not source_before_goal
            or not manifest_bound
        ):
            issues.append({"code": "OPEN_POLYMER_SOURCE_CALL_BINDING_INVALID", "detail": source_call_id})
        if legacy_v2:
            manifest = board_item.get("artifact_manifest") if isinstance(board_item.get("artifact_manifest"), dict) else {}
            files = manifest.get("files") if isinstance(manifest.get("files"), list) else []
            by_name = {
                str(entry.get("name") or ""): entry
                for entry in files
                if isinstance(entry, dict) and str(entry.get("name") or "")
            }
            bindings = board_item.get("legacy_artifact_bindings") if isinstance(board_item.get("legacy_artifact_bindings"), dict) else {}
            if (
                len(by_name) != len(files)
                or set(by_name) != set(OPEN_POLYMER_LEGACY_REQUIRED_FILES)
                or set(bindings) != set(by_name)
            ):
                issues.append({"code": "OPEN_POLYMER_LEGACY_MANIFEST_SET_INVALID", "detail": ""})
            published_by_id: dict[str, dict[str, Any]] = {}
            for entry in published:
                if not isinstance(entry, dict) or not str(entry.get("id") or ""):
                    issues.append({"code": "OPEN_POLYMER_LEGACY_PUBLICATION_ENTRY_INVALID", "detail": ""})
                    continue
                artifact_id = str(entry.get("id") or "")
                previous = published_by_id.get(artifact_id)
                if previous is not None and (
                    previous.get("name") != entry.get("name")
                    or int(previous.get("bytes", -1)) != int(entry.get("bytes", -2))
                    or str(previous.get("sha256") or "").lower()
                    != str(entry.get("sha256") or "").lower()
                ):
                    issues.append({"code": "OPEN_POLYMER_LEGACY_PUBLICATION_DUPLICATE_CONFLICT", "detail": artifact_id})
                published_by_id[artifact_id] = entry
            expected_ids = set(str(value) for value in bindings.values()) | {
                OPEN_POLYMER_LEGACY_MANIFEST_ARTIFACT_ID,
                OPEN_POLYMER_LEGACY_MANIFEST_RECEIPT_ARTIFACT_ID,
            }
            if set(published_by_id) != expected_ids:
                issues.append({"code": "OPEN_POLYMER_LEGACY_PUBLICATION_SET_INVALID", "detail": ""})
            manifest_projection = published_by_id.get(OPEN_POLYMER_LEGACY_MANIFEST_ARTIFACT_ID, {})
            receipt_projection = published_by_id.get(OPEN_POLYMER_LEGACY_MANIFEST_RECEIPT_ARTIFACT_ID, {})
            if (
                manifest_projection.get("name") != "artifact-manifest.json"
                or str(manifest_projection.get("sha256") or "").lower()
                != OPEN_POLYMER_LEGACY_MANIFEST_SHA256
                or receipt_projection.get("name") != "artifact-manifest-receipt.json"
                or str(receipt_projection.get("sha256") or "").lower()
                != OPEN_POLYMER_LEGACY_MANIFEST_RECEIPT_SHA256
                or int(receipt_projection.get("bytes", -1))
                != OPEN_POLYMER_LEGACY_MANIFEST_RECEIPT_BYTES
            ):
                issues.append({"code": "OPEN_POLYMER_LEGACY_MANIFEST_PUBLICATION_INVALID", "detail": ""})
            for name, entry in by_name.items():
                projected = published_by_id.get(str(bindings.get(name) or ""), {})
                if (
                    projected.get("name") != name
                    or int(projected.get("bytes", -1)) != int(entry.get("bytes", -2))
                    or str(projected.get("sha256") or "").lower()
                    != str(entry.get("sha256") or "").lower()
                ):
                    issues.append({"code": "OPEN_POLYMER_LEGACY_PUBLICATION_BINDING_INVALID", "detail": name})
            events = all_events if isinstance(all_events, list) else []
            started_events = [
                event for event in events
                if int(event.get("seq") or 0) == OPEN_POLYMER_LEGACY_STARTED_EVENT_SEQ
            ]
            completed_events = [
                event for event in events
                if int(event.get("seq") or 0) == OPEN_POLYMER_LEGACY_COMPLETED_EVENT_SEQ
            ]
            started_payload = started_events[0].get("payload") if len(started_events) == 1 else None
            completed_payload = completed_events[0].get("payload") if len(completed_events) == 1 else None
            event_binding_valid = (
                len(started_events) == 1
                and len(completed_events) == 1
                and started_events[0].get("event_type") == "tool.started"
                and completed_events[0].get("event_type") == "tool.completed"
                and isinstance(started_payload, dict)
                and isinstance(completed_payload, dict)
                and started_payload.get("id") == source_call_id
                and completed_payload.get("tool_call_id") == source_call_id
                and completed_payload.get("ok") is True
            )
            if event_binding_valid and source_started is not None and source_completed is not None:
                try:
                    event_binding_valid = (
                        parse_utc(str(started_events[0].get("created_at") or ""), field="open_polymer.event.started")
                        == source_started
                        and parse_utc(str(completed_events[0].get("created_at") or ""), field="open_polymer.event.completed")
                        == source_completed
                    )
                except ValueError:
                    event_binding_valid = False
            if not event_binding_valid:
                issues.append({"code": "OPEN_POLYMER_SOURCE_EVENT_BINDING_INVALID", "detail": source_call_id})
    for item in all_calls:
        arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        if (
            arguments.get("competition") != "open_polymer"
            or item.get("tool_name") not in {"training_route", "hpc_verify", "hpc_execute_solution"}
        ):
            continue
        try:
            item_time = parse_utc(str(item.get("created_at") or ""), field="open_polymer.window.created_at")
        except ValueError:
            issues.append({"code": "OPEN_POLYMER_WINDOW_TIMESTAMP_INVALID", "detail": str(item.get("id") or "")})
            continue
        if item_time >= since:
            issues.append({"code": "OPEN_POLYMER_WINDOW_RETRAINING_DETECTED", "detail": str(item.get("id") or "")})
    return not issues, issues


def validate_managed_candidate_execution(
    *, board_item: dict[str, Any], all_calls: list[dict[str, Any]], since: datetime
) -> tuple[bool, list[dict[str, str]]]:
    competition = str(board_item.get("competition") or "")
    solution_id = str(board_item.get("solution_id") or "")
    execution = board_item.get("managed_execution")
    if not isinstance(execution, dict) or execution.get("schema") != "evomind.managed-candidate-execution.v1":
        return False, [{"code": "MANAGED_EXECUTION_SCHEMA_INVALID", "detail": competition}]
    identifiers = [
        str(execution.get(field) or "")
        for field in (
            "training_route_tool_call_id", "hpc_verify_tool_call_id", "hpc_execute_tool_call_id"
        )
    ]
    bound: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    for identifier in identifiers:
        matches = [item for item in all_calls if str(item.get("id") or "") == identifier]
        if len(matches) != 1:
            issues.append({"code": "MANAGED_EXECUTION_CALL_COUNT_INVALID", "detail": identifier})
        else:
            bound.append(matches[0])
    if len(bound) != 3:
        return False, issues
    try:
        times = [parse_utc(str(item.get("created_at") or ""), field="managed_execution.created_at") for item in bound]
    except ValueError:
        issues.append({"code": "MANAGED_EXECUTION_TIMESTAMP_INVALID", "detail": competition})
        times = []
    arguments = [item.get("arguments") if isinstance(item.get("arguments"), dict) else {} for item in bound]
    results = [item.get("result") if isinstance(item.get("result"), dict) else {} for item in bound]
    execute_content = results[2].get("content") if isinstance(results[2].get("content"), dict) else {}
    published = execute_content.get("published_artifacts") if isinstance(execute_content.get("published_artifacts"), list) else []
    manifest_sha = str(board_item.get("artifact_manifest_sha256") or "").lower()
    manifest_bound = str(execute_content.get("artifact_manifest_sha256") or "").lower() == manifest_sha or any(
        isinstance(entry, dict) and str(entry.get("sha256") or "").lower() == manifest_sha
        for entry in published
    )
    if [item.get("tool_name") for item in bound] != ["training_route", "hpc_verify", "hpc_execute_solution"]:
        issues.append({"code": "MANAGED_EXECUTION_TOOL_ORDER_INVALID", "detail": competition})
    if any(str(item.get("status") or "").casefold() != "completed" for item in bound):
        issues.append({"code": "MANAGED_EXECUTION_NOT_COMPLETED", "detail": competition})
    if times and not (since <= times[0] < times[1] < times[2]):
        issues.append({"code": "MANAGED_EXECUTION_TIME_ORDER_INVALID", "detail": competition})
    if arguments[0].get("competition") != competition or "solution_id" in arguments[0]:
        issues.append({"code": "TRAINING_ROUTE_BINDING_INVALID", "detail": competition})
    if arguments[1]:
        issues.append({"code": "HPC_VERIFY_ARGUMENTS_NOT_EMPTY", "detail": competition})
    if arguments[2].get("competition") != competition or arguments[2].get("solution_id") != solution_id:
        issues.append({"code": "HPC_EXECUTE_BINDING_INVALID", "detail": competition})
    if not _hpc_verify_complete(results[1]):
        issues.append({"code": "HPC_VERIFY_5_OF_5_NOT_PROVEN", "detail": competition})
    if results[2].get("ok") is not True or not manifest_bound:
        issues.append({"code": "HPC_EXECUTE_ARTIFACTS_NOT_BOUND", "detail": competition})
    keys = [str(item.get("idempotency_key") or "") for item in bound]
    if any(not key for key in keys) or len(set(keys)) != 3:
        issues.append({"code": "MANAGED_EXECUTION_IDEMPOTENCY_INVALID", "detail": competition})
    return not issues, issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-db", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--since-utc", default="2026-08-29T15:30:00+00:00")
    parser.add_argument(
        "--goal-spec",
        default=str(ROOT / "configs" / "g21_five_competition_goal.json"),
    )
    parser.add_argument(
        "--human-baseline-evidence",
        default=str(ROOT / "configs" / "g21_five_competition_human_baseline_gate.json"),
    )
    args = parser.parse_args()

    if args.run_id != FIXED_RUN_ID:
        raise ValueError("audit is restricted to the fixed G21 Run")
    requested_since = parse_utc(args.since_utc, field="since_utc")

    database = Path(args.runtime_db).resolve(strict=True)
    output = Path(args.output).resolve()

    goal_spec: dict[str, Any] = {}
    goal_spec_sha256 = ""
    goal_spec_issues: list[dict[str, str]] = []
    try:
        goal_spec, goal_spec_sha256 = load_json_file(Path(args.goal_spec).resolve(strict=True))
        spec_ok, spec_issues = validate_goal_spec(goal_spec)
        if not spec_ok:
            goal_spec_issues = [item.to_dict() for item in spec_issues]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        goal_spec_issues = [{"code": "GOAL_SPEC_LOAD_FAILED", "detail": type(exc).__name__}]

    human_baseline_evidence: dict[str, Any] = {}
    human_baseline_evidence_sha256 = ""
    human_baseline_issues: list[dict[str, str]] = []
    human_baseline_completion_eligible = False
    try:
        human_baseline_evidence, human_baseline_evidence_sha256 = load_json_file(
            Path(args.human_baseline_evidence).resolve(strict=True)
        )
        human_summary = human_baseline_gate_summary(human_baseline_evidence, run_id=args.run_id)
        human_baseline_issues = list(human_summary.get("issues") or [])
        human_baseline_completion_eligible = bool(human_summary.get("completion_eligible"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
            human_baseline_issues = [{"code": "HUMAN_BASELINE_EVIDENCE_LOAD_FAILED", "detail": type(exc).__name__}]
    if goal_spec and human_baseline_evidence_sha256:
        if goal_spec.get("human_baseline_evidence_sha256") != human_baseline_evidence_sha256:
            goal_spec_issues.append({"code": "GOAL_SPEC_HUMAN_BASELINE_SOURCE_MISMATCH", "detail": ""})
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True, timeout=10)
    connection.row_factory = sqlite3.Row

    goal_record_audit = inspect_goal_records(connection, run_id=args.run_id)

    audit_since = requested_since
    goal_created_at = ""
    if _table_exists(connection, "events"):
        event_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(events)").fetchall()}
        if {"session_id", "event_type", "created_at"} <= event_columns:
            row = connection.execute(
                "SELECT created_at FROM events WHERE session_id=? AND event_type='goal.created' "
                "ORDER BY created_at DESC LIMIT 1",
                (args.run_id,),
            ).fetchone()
            if row is not None:
                goal_created_at = str(row[0] or "")
                try:
                    audit_since = max(audit_since, parse_utc(goal_created_at, field="goal.created"))
                except ValueError:
                    goal_record_audit["issues"].append(
                        {"code": "GOAL_CREATED_TIMESTAMP_INVALID", "detail": goal_created_at}
                    )
                    goal_record_audit["valid"] = False

    run = connection.execute("SELECT * FROM assistant_runs WHERE id=?", (args.run_id,)).fetchone()
    if run is None:
        raise ValueError("run not found")
    approval_rows = [
        dict(row)
        for row in connection.execute("SELECT * FROM approvals WHERE session_id=?", (args.run_id,)).fetchall()
    ]
    pending = sum(str(row.get("status") or "").casefold() == "pending" for row in approval_rows)
    approvals_by_id = {
        str(row.get("id") or ""): row
        for row in approval_rows
        if str(row.get("id") or "")
    }
    deliverables = [
        dict(row)
        for row in connection.execute(
            "SELECT * FROM deliverables WHERE run_id=? ORDER BY verified_at", (args.run_id,)
        ).fetchall()
    ]
    verified_deliverables: list[dict[str, Any]] = []
    artifact_evidence_rows: list[dict[str, Any]] = []
    artifact_failures: list[dict[str, str]] = []
    for row in deliverables:
        path = Path(str(row.get("path") or ""))
        regular_file = path.is_file() and not path.is_symlink()
        actual_bytes = path.stat().st_size if regular_file else -1
        actual_sha = sha256_file(path) if regular_file else ""
        declared_bytes = int(row["bytes"]) if row.get("bytes") is not None else -2
        ok = actual_bytes == declared_bytes and actual_sha == str(row.get("sha256") or "")
        public = {
            "id": str(row.get("id") or ""),
            "name": str(row.get("name") or ""),
            "bytes": int(row.get("bytes") or 0),
            "sha256": str(row.get("sha256") or ""),
            "source_tool_call": str(row.get("source_tool_call") or ""),
            "verified_at": str(row.get("verified_at") or ""),
            "current_file_verified": ok,
        }
        verified_deliverables.append(public)
        artifact_evidence_rows.append({**public, "path": str(row.get("path") or "")})
        if not ok:
            artifact_failures.append({"id": public["id"], "name": public["name"]})

    goal_board: dict[str, Any] = {}
    for row in reversed(deliverables):
        if row.get("name") == "goal-board-v2.json":
            path = Path(str(row.get("path") or ""))
            if path.is_file() and sha256_file(path) == row.get("sha256"):
                goal_board = json.loads(path.read_text(encoding="utf-8"))
                break

    goal_contract = validate_goal_board(goal_board, run_id=args.run_id)
    goal_artifact_contract = audit_goal_board_artifacts(goal_board, artifact_evidence_rows)
    expected_metrics = {
        str(item.get("competition")): {
            "metric": str(item.get("official_metric") or ""),
            "direction": str(item.get("metric_direction") or ""),
        }
        for item in goal_board.get("competitions", [])
        if isinstance(item, dict)
    }
    if human_baseline_evidence:
        human_summary = human_baseline_gate_summary(
            human_baseline_evidence,
            run_id=args.run_id,
            expected_metrics=expected_metrics,
        )
        human_baseline_issues = list(human_summary.get("issues") or [])
        human_baseline_completion_eligible = bool(human_summary.get("completion_eligible"))
    if goal_record_audit.get("record") is not None:
        record = goal_record_audit["record"]
        if goal_spec and record.get("spec_sha256") != sha256_json(goal_spec):
            goal_record_audit["issues"].append({"code": "GOAL_RECORD_SPEC_SOURCE_MISMATCH", "detail": ""})
        if goal_board and record.get("board_sha256") != sha256_json(goal_board):
            goal_record_audit["issues"].append({"code": "GOAL_RECORD_BOARD_SOURCE_MISMATCH", "detail": ""})
        if human_baseline_evidence_sha256 and record.get("human_baseline_sha256") != human_baseline_evidence_sha256:
            goal_record_audit["issues"].append({"code": "GOAL_RECORD_HUMAN_BASELINE_SOURCE_MISMATCH", "detail": ""})
        goal_record_audit["valid"] = bool(goal_record_audit.get("valid") and not goal_record_audit["issues"])
    goal_update_evidence = {"valid": False, "matching_event_count": 0, "issues": []}
    if _table_exists(connection, "events"):
        event_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(events)").fetchall()}
        if {"session_id", "event_type", "payload_json", "created_at"} <= event_columns:
            matches = 0
            for row in connection.execute(
                "SELECT payload_json,created_at FROM events WHERE session_id=? AND event_type='goal.updated' ORDER BY created_at",
                (args.run_id,),
            ).fetchall():
                try:
                    payload_value = json.loads(str(row[0] or "{}"))
                except json.JSONDecodeError:
                    continue
                if (
                    _timestamp_at_or_after(row[1], audit_since)
                    and isinstance(payload_value, dict)
                    and payload_value.get("goal_id") == FIXED_GOAL_ID
                    and payload_value.get("board_sha256") == sha256_json(goal_board)
                ):
                    matches += 1
            goal_update_evidence = {
                "valid": matches >= 1,
                "matching_event_count": matches,
                "issues": [] if matches >= 1 else [{"code": "GOAL_UPDATED_EVIDENCE_MISSING", "detail": ""}],
            }
        else:
            goal_update_evidence["issues"] = [{"code": "GOAL_EVENT_SCHEMA_INCOMPLETE", "detail": ""}]
    else:
        goal_update_evidence["issues"] = [{"code": "GOAL_EVENT_TABLE_MISSING", "detail": ""}]
    goal_items = {
        item.competition: item
        for item in goal_contract.items
    }

    solutions: list[dict[str, Any]] = []
    raw_running: list[str] = []
    raw_unsettled: list[dict[str, Any]] = []
    prohibited_calls: list[dict[str, Any]] = []
    managed_groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    nonexistent_evidence_stage_rows: list[dict[str, Any]] = []
    tool_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(tool_calls)").fetchall()}
    tool_schema_issues: list[dict[str, str]] = []
    if "id" not in tool_columns:
        tool_schema_issues.append({"code": "TOOL_CALL_ID_COLUMN_MISSING", "detail": ""})
    if "idempotency_key" not in tool_columns:
        tool_schema_issues.append({"code": "TOOL_CALL_IDEMPOTENCY_COLUMN_MISSING", "detail": ""})
    optional_tool_columns = {
        "id": "'' AS id",
        "started_at": "'' AS started_at",
        "completed_at": "'' AS completed_at",
        "approval_id": "'' AS approval_id",
        "idempotency_key": "'' AS idempotency_key",
    }
    projections = [
        name if name in tool_columns else fallback
        for name, fallback in optional_tool_columns.items()
    ]
    tool_rows = [
        dict(row)
        for row in connection.execute(
            "SELECT rowid AS audit_rowid,tool_name,status,created_at,arguments_json,result_json,"
            + ",".join(projections)
            + " FROM tool_calls WHERE session_id=? ORDER BY created_at,audit_rowid",
            (args.run_id,),
        ).fetchall()
    ]
    latest_assistant_turn_at = ""
    if _table_exists(connection, "turns"):
        turn_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(turns)").fetchall()}
        if {"session_id", "role", "created_at"} <= turn_columns:
            row = connection.execute(
                "SELECT MAX(created_at) FROM turns WHERE session_id=? AND role='assistant'",
                (args.run_id,),
            ).fetchone()
            latest_assistant_turn_at = str(row[0] or "") if row is not None else ""
    call_rows: list[dict[str, Any]] = []
    call_details: list[dict[str, Any]] = []
    for row in tool_rows:
        tool_name = str(row["tool_name"])
        arguments = parse_json(row["arguments_json"])
        result = parse_json(row["result_json"])
        status = str(row["status"] or "").casefold()
        call_projection = {
            "run_id": str(arguments.get("run_id") or args.run_id),
            "session_id": args.run_id,
            "tool_name": tool_name,
            "solution_id": str(arguments.get("solution_id") or ""),
            "idempotency_key": str(row["idempotency_key"] or arguments.get("idempotency_key") or ""),
            "status": status,
            "created_at": str(row["created_at"] or ""),
            "started_at": str(row.get("started_at") or row["created_at"] or ""),
            "completed_at": str(row.get("completed_at") or ""),
            "approval_id": str(row.get("approval_id") or ""),
            "id": str(row.get("id") or row.get("audit_rowid") or ""),
        }
        call_rows.append(dict(call_projection))
        call_details.append({**call_projection, "arguments": arguments, "result": result})
        if status == "running":
            raw_running.append(str(arguments.get("solution_id") or tool_name))
        if status in UNSETTLED_STATUSES:
            raw_unsettled.append(call_projection)
        in_window = _timestamp_at_or_after(row["created_at"], audit_since)
        if in_window and tool_name in NONEXISTENT_EVIDENCE_STAGE_TOOLS:
            nonexistent_evidence_stage_rows.append(call_projection)
        if (
            in_window
            and _effective_prohibited_call(call_projection, approvals_by_id=approvals_by_id)
            and (
                tool_name.casefold() in PROHIBITED_TOOLS
                or str(arguments.get("competition") or "").casefold() == "weather4cast"
            )
        ):
            prohibited_calls.append(
                {
                    "tool": tool_name,
                    "status": str(row["status"]),
                    "created_at": str(row["created_at"]),
                    "competition": str(arguments.get("competition") or ""),
                }
            )
        competition = str(arguments.get("competition") or "")
        solution_id = str(arguments.get("solution_id") or "")
        if in_window and tool_name in MANAGED_SEQUENCE_TOOLS and competition and solution_id:
            managed_groups[(competition, solution_id)].append({**call_projection, "result": result})
        if tool_name != "hpc_execute_solution":
            continue
        content = result.get("content") if isinstance(result.get("content"), dict) else {}
        published = content.get("published_artifacts") if isinstance(content.get("published_artifacts"), list) else []
        competition = str(arguments.get("competition") or "")
        solutions.append(
            {
                "competition": competition,
                "solution_id": str(arguments.get("solution_id") or ""),
                "tool_status": str(row["status"]),
                "created_at": str(row["created_at"]),
                "completed_at": str(row["completed_at"] or ""),
                "result_ok": result.get("ok"),
                "summary": str(result.get("summary") or ""),
                "error": str(result.get("error") or ""),
                "solution_status": str(content.get("status") or ""),
                "exit_code": content.get("exit_code"),
                "published_artifacts": [
                    {
                        "id": str(item.get("id") or ""),
                        "name": str(item.get("name") or ""),
                        "bytes": int(item.get("bytes") or 0),
                        "sha256": str(item.get("sha256") or ""),
                    }
                    for item in published
                    if isinstance(item, dict)
                ],
            }
        )

    event_details: list[dict[str, Any]] = []
    if _table_exists(connection, "events"):
        event_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(events)").fetchall()}
        if {"session_id", "seq", "event_type", "payload_json", "created_at"} <= event_columns:
            for row in connection.execute(
                "SELECT seq,event_type,payload_json,created_at FROM events WHERE session_id=? ORDER BY seq",
                (args.run_id,),
            ).fetchall():
                event_details.append(
                    {
                        "seq": int(row[0]),
                        "event_type": str(row[1] or ""),
                        "payload": parse_json(row[2]),
                        "created_at": str(row[3] or ""),
                    }
                )

    board_by_competition = {
        str(item.get("competition")): dict(item)
        for item in goal_board.get("competitions", [])
        if isinstance(item, dict)
    }
    sequence_results: list[dict[str, Any]] = []
    sequence_issues: list[dict[str, str]] = [
        *tool_schema_issues,
        *[
            {"code": "NONEXISTENT_EVIDENCE_STAGE_TOOL_ROW", "detail": str(item.get("tool_name") or "")}
            for item in nonexistent_evidence_stage_rows
        ],
    ]
    for competition in COMPETITIONS:
        item = board_by_competition.get(competition, {})
        if competition == "open_polymer":
            valid, issues = validate_open_polymer_adoption(
                board_item=item,
                all_calls=call_details,
                since=audit_since,
                all_events=event_details,
            )
            mode = "existing_candidate_adoption"
        else:
            valid, issues = validate_managed_candidate_execution(
                board_item=item,
                all_calls=call_details,
                since=audit_since,
            )
            mode = "managed_real_tools_plus_evidence_stages"
        entry = {
            "competition": competition,
            "solution_id": str(item.get("solution_id") or ""),
            "mode": mode,
            "valid": valid,
            "issues": issues,
        }
        sequence_results.append(entry)
        if not valid:
            sequence_issues.extend(
                {"code": f"{competition}:{issue['code']}", "detail": issue.get("detail", "")}
                for issue in issues
            )
    selected_intervals: list[tuple[datetime, datetime, str]] = []
    calls_by_id = {str(item.get("id") or ""): item for item in call_details}
    for competition in ("cure_bench", "e2lmc", "mindgames", "ariel_2025"):
        execution = board_by_competition.get(competition, {}).get("managed_execution")
        if not isinstance(execution, dict):
            continue
        route = calls_by_id.get(str(execution.get("training_route_tool_call_id") or ""))
        execute = calls_by_id.get(str(execution.get("hpc_execute_tool_call_id") or ""))
        if route and execute:
            try:
                selected_intervals.append(
                    (
                        parse_utc(str(route.get("created_at") or ""), field="serial.route"),
                        parse_utc(str(execute.get("completed_at") or execute.get("created_at") or ""), field="serial.execute"),
                        competition,
                    )
                )
            except ValueError:
                pass
    selected_intervals.sort()
    for previous, current in zip(selected_intervals, selected_intervals[1:]):
        if current[0] <= previous[1]:
            sequence_issues.append(
                {"code": "SINGLE_GPU_SELECTED_EXECUTION_OVERLAP", "detail": f"{previous[2]}:{current[2]}"}
            )
    sequence_valid = not sequence_issues and len(sequence_results) == len(COMPETITIONS)
    run_status = str(run["status"] or "").casefold()
    run_terminal = run_status in TERMINAL_RUN_STATUSES
    effective_active = effective_active_tool_calls(
        call_rows,
        approvals_by_id=approvals_by_id,
        latest_assistant_turn_at=latest_assistant_turn_at,
    )
    effective_active_count = len(effective_active)
    goal_unsettled = goal_unsettled_calls_since(
        call_rows,
        since=audit_since,
        approvals_by_id=approvals_by_id,
        latest_assistant_turn_at=latest_assistant_turn_at,
    )
    goal_record_completed = bool(
        goal_record_audit.get("valid")
        and isinstance(goal_record_audit.get("record"), dict)
        and str(goal_record_audit["record"].get("status") or "").casefold() == "completed"
    )

    requirements = []
    for competition in COMPETITIONS:
        item = board_by_competition.get(competition, {})
        status = str(item.get("goal_status") or "MISSING")
        contract_item = goal_items.get(competition)
        requirements.append(
            {
                "competition": competition,
                "goal_status": status,
                "data_status": str(item.get("data_status") or "MISSING"),
                "declared_verified_complete": status == "VERIFIED_COMPLETE",
                "verified_complete": bool(contract_item and contract_item.strict_verified_complete),
                "exact_gate": str(item.get("exact_gate") or ""),
                "solution_count": sum(row["competition"] == competition for row in solutions),
                "contract_issues": (
                    [issue.to_dict() for issue in contract_item.issues]
                    if contract_item
                    else [{"code": "GOAL_BOARD_ITEM_MISSING", "detail": competition}]
                ),
            }
        )

    payload = {
        "schema": "evomind.five_competition_goal_audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": {
            "id": args.run_id,
            "status": str(run["status"]),
            "terminal": run_terminal,
            "pending_approvals": int(pending),
            "effective_active_count": effective_active_count,
        },
        "audit_window": {
            "requested_since_utc": requested_since.isoformat(),
            "goal_created_at_utc": goal_created_at,
            "effective_since_utc": audit_since.isoformat(),
        },
        "goal_board_present": bool(goal_board),
        "goal_board": goal_board,
        "goal_contract": goal_contract.to_dict(),
        "goal_record": {
            **goal_record_audit,
            "completion_status_verified": goal_record_completed,
            "expected_goal_id": FIXED_GOAL_ID,
            "expected_goal_key": FIXED_GOAL_KEY,
            "expected_schema": GOAL_RECORD_SCHEMA,
        },
        "goal_update_evidence": goal_update_evidence,
        "goal_spec": {
            "path": str(Path(args.goal_spec).resolve()),
            "sha256": goal_spec_sha256,
            "valid": not goal_spec_issues,
            "issues": goal_spec_issues,
        },
        "human_baseline_evidence": {
            "path": str(Path(args.human_baseline_evidence).resolve()),
            "sha256": human_baseline_evidence_sha256,
            "valid": not human_baseline_issues,
            "classification": human_baseline_evidence.get("classification") if human_baseline_evidence else "",
            "completion_eligible": human_baseline_completion_eligible,
            "issues": human_baseline_issues,
        },
        "requirements": requirements,
        "declared_verified_complete_count": sum(item["declared_verified_complete"] for item in requirements),
        "verified_complete_count": sum(item["verified_complete"] for item in requirements),
        "target_count": len(COMPETITIONS),
        "solutions": solutions,
        "raw_running_tool_rows": raw_running,
        "raw_unsettled_tool_rows": raw_unsettled,
        "effective_active_tool_rows": effective_active,
        "goal_unsettled_calls_since_goal": goal_unsettled,
        "managed_sequence_results": sequence_results,
        "managed_sequence_issues": sequence_issues,
        "tool_schema_issues": tool_schema_issues,
        "nonexistent_evidence_stage_tool_rows": nonexistent_evidence_stage_rows,
        "deliverable_count": len(verified_deliverables),
        "deliverables_verified": len(verified_deliverables) - len(artifact_failures),
        "artifact_failures": artifact_failures,
        "goal_artifact_contract": goal_artifact_contract,
        "prohibited_calls_since_goal": prohibited_calls,
        "weather_actions_since_goal": sum(item["competition"] == "weather4cast" for item in prohibited_calls),
        "completion_proven": goal_contract.completion_proven
        and not goal_spec_issues
        and not human_baseline_issues
        and (
            human_baseline_completion_eligible
            or conditional_baseline_policy(goal_spec)
        )
        and all(item["verified_complete"] for item in requirements)
        and goal_record_audit["valid"]
        and goal_record_completed
        and goal_update_evidence["valid"]
        and run_terminal
        and pending == 0
        and effective_active_count == 0
        and not goal_unsettled
        and sequence_valid
        and goal_artifact_contract["valid"]
        and not prohibited_calls,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_bytes(data)
    temporary.replace(output)
    receipt = {
        "schema": "evomind.five_competition_goal_audit_receipt.v1",
        "artifact": output.name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "completion_proven": payload["completion_proven"],
        "verified_complete_count": payload["verified_complete_count"],
        "target_count": payload["target_count"],
    }
    receipt_path = output.with_name(f"{output.stem}-receipt.json")
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
