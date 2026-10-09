from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pytest

import evomind_runtime.competition_goal as goal_contract
import evomind_runtime.goal_board as goal_board_module
from evomind_runtime.competition_goal import (
    COMPETITIONS,
    CONDITIONAL_BASELINE_POLICY_EVIDENCE_SHA256,
    CONDITIONAL_BASELINE_POLICY_MODE,
    FIXED_ALLOCATION,
    FIXED_GOAL_ID,
    FIXED_RUN_ID,
    OPEN_POLYMER_LEGACY_ADOPTION_SCHEMA,
    OPEN_POLYMER_LEGACY_COMPLETED_EVENT_SEQ,
    OPEN_POLYMER_LEGACY_IDEMPOTENCY_KEY,
    OPEN_POLYMER_LEGACY_MANIFEST_ARTIFACT_ID,
    OPEN_POLYMER_LEGACY_MANIFEST_RECEIPT_ARTIFACT_ID,
    OPEN_POLYMER_LEGACY_MANIFEST_RECEIPT_BYTES,
    OPEN_POLYMER_LEGACY_MANIFEST_RECEIPT_SHA256,
    OPEN_POLYMER_LEGACY_MANIFEST_SHA256,
    OPEN_POLYMER_LEGACY_REQUIRED_FILES,
    OPEN_POLYMER_LEGACY_REVIEW_ARTIFACT_ID,
    OPEN_POLYMER_LEGACY_REVIEW_SHA256,
    OPEN_POLYMER_LEGACY_SOLUTION_ID,
    OPEN_POLYMER_LEGACY_SOURCE_CALL_ID,
    OPEN_POLYMER_LEGACY_SPLIT_ARTIFACT_ID,
    OPEN_POLYMER_LEGACY_SPLIT_SHA256,
    OPEN_POLYMER_LEGACY_STARTED_EVENT_SEQ,
    canonical_json_bytes,
    load_goal_spec,
    sha256_json,
    validate_goal_board,
    validate_goal_item,
)
from evomind_runtime.goal_board import ensure_fixed_goal, update_fixed_goal_board
from evomind_runtime.models import Session, ToolCall, ToolResult, utc_now
from evomind_runtime.store import RuntimeStore
from scripts import five_competition_goal_audit as goal_audit


METRIC = "mean per-target RMSE normalized by development-target standard deviation"
BASELINE = 0.7326585287082159
CANDIDATE = 0.6535338608202899
DIFF_LOWER = -0.11084084061270712
DIFF_UPPER = -0.050317963973942365


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _protocol() -> str:
    return (
        "The baseline and candidate use the same SHA-bound exact-SMILES group split, "
        "the same five targets, and mean per-target RMSE normalized by the development-target standard deviation."
    )


def _pure_item() -> dict:
    file_hashes = {name: _sha(name) for name in OPEN_POLYMER_LEGACY_REQUIRED_FILES}
    file_hashes["split-manifest.json"] = OPEN_POLYMER_LEGACY_SPLIT_SHA256
    file_hashes["independent-verification.json"] = OPEN_POLYMER_LEGACY_REVIEW_SHA256
    files = [
        {"name": name, "bytes": index + 1, "sha256": file_hashes[name]}
        for index, name in enumerate(sorted(file_hashes))
    ]
    bindings = {name: f"artifact-open-polymer-{index}" for index, name in enumerate(sorted(file_hashes), 1)}
    bindings["split-manifest.json"] = OPEN_POLYMER_LEGACY_SPLIT_ARTIFACT_ID
    bindings["independent-verification.json"] = OPEN_POLYMER_LEGACY_REVIEW_ARTIFACT_ID
    protocol = _protocol()
    human_sha = _sha("open-polymer-human-baseline-unavailable")
    manifest = {
        "schema": "evomind.open_polymer.artifact_manifest.v2",
        "source_sha256": file_hashes["solution.py"],
        "data_train_sha256": _sha("train"),
        "data_test_sha256": _sha("test"),
        "exit_code": 0,
        "gate_passed": True,
        "files": files,
    }
    return {
        "competition": "open_polymer",
        "solution_id": OPEN_POLYMER_LEGACY_SOLUTION_ID,
        "goal_status": "VERIFIED_COMPLETE",
        "official_metric": METRIC,
        "evaluation_metric": "mean_normalized_rmse",
        "metric_direction": "lower_is_better",
        "protocol_scope": "publicly_reproducible_internal_protocol",
        "official_metric_claimed": False,
        "primary_baseline": {
            "status": "VERIFIED",
            "baseline_type": "publicly_reproducible_strong",
            "participant_type": "model_baseline",
            "source_url": "https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025",
            "source_authority": "official_dataset",
            "source_sha256": file_hashes["baseline-evidence.json"],
            "evidence_artifact_id": bindings["baseline-evidence.json"],
            "evidence_artifact_sha256": file_hashes["baseline-evidence.json"],
            "title": "SHA-bound char-ngram Ridge strong baseline on Open Polymer data",
            "protocol_id": "open-polymer-internal-normalized-rmse-v2",
            "protocol": protocol,
            "protocol_sha256": hashlib.sha256(protocol.encode("utf-8")).hexdigest(),
            "protocol_comparable": True,
            "metric": METRIC,
            "direction": "lower_is_better",
            "value": BASELINE,
            "publicly_reproducible": True,
            "implementation_sha256": file_hashes["solution.py"],
            "data_manifest_sha256": file_hashes["split-manifest.json"],
            "environment_sha256": file_hashes["environment-lock.json"],
            "reproduction_receipt_sha256": file_hashes["baseline-reproduction.json"],
        },
        "primary_comparison": {
            "candidate_metric": CANDIDATE,
            "ci95_lower": BASELINE + DIFF_LOWER,
            "ci95_upper": BASELINE + DIFF_UPPER,
        },
        "human_baseline_availability": {
            "status": "NOT_VERIFIED_AVAILABLE",
            "audit_complete": True,
            "protocol_search_complete": True,
            "human_baseline_verified": False,
            "evidence_artifact_id": "artifact-open-polymer-human-unavailable",
            "evidence_artifact_sha256": human_sha,
        },
        "artifact_manifest": manifest,
        "artifact_manifest_artifact_id": OPEN_POLYMER_LEGACY_MANIFEST_ARTIFACT_ID,
        "artifact_manifest_sha256": OPEN_POLYMER_LEGACY_MANIFEST_SHA256,
        "legacy_artifact_bindings": bindings,
        "command_sha256": file_hashes["training-config.json"],
        "environment_sha256": file_hashes["environment-lock.json"],
        "data_manifest_sha256": file_hashes["split-manifest.json"],
        "source_code_sha256": file_hashes["solution.py"],
        "model_sha256": file_hashes["model-bundle.joblib"],
        "metrics_sha256": file_hashes["metrics.json"],
        "rollback_sha256": file_hashes["retrospective-memory.json"],
        "rollback_evidence_mode": "legacy_completed_candidate_not_deployed",
        "legacy_holdout_evidence": {
            "schema": "evomind.legacy-holdout-evidence.v1",
            "atomic_claim_available": False,
            "historical_claim_not_asserted": True,
            "posthoc_ledger_created": False,
            "selection_data_scope": "development_only",
            "split_manifest_artifact_id": OPEN_POLYMER_LEGACY_SPLIT_ARTIFACT_ID,
            "split_manifest_sha256": OPEN_POLYMER_LEGACY_SPLIT_SHA256,
            "holdout_blind_for_selection": True,
            "group_overlap": 0,
            "test_labels_used": False,
        },
        "leakage_audit": {
            "test_labels_used": False,
            "hidden_test_labels_used": False,
            "holdout_overlap": 0,
            "id_overlap": 0,
            "architecture_overlap": 0,
            "point_overlap": 0,
            "manifest_closed": True,
            "loader_smoke": True,
        },
        "side_effects": {
            "signals_sent": 0,
            "remote_writes": 0,
            "training_tool_calls": 0,
            "other_processes_modified": False,
        },
        "independent_review": {
            "schema": "evomind.legacy-independent-review-binding.v1",
            "source_schema": "evomind.open_polymer.independent_verification.v2",
            "artifact_id": OPEN_POLYMER_LEGACY_REVIEW_ARTIFACT_ID,
            "sha256": OPEN_POLYMER_LEGACY_REVIEW_SHA256,
            "verified": True,
        },
        "existing_candidate_adoption": {
            "schema": OPEN_POLYMER_LEGACY_ADOPTION_SCHEMA,
            "mode": "legacy_pre_goal_completed_candidate",
            "run_id": FIXED_RUN_ID,
            "allocation": FIXED_ALLOCATION,
            "competition": "open_polymer",
            "source_tool_call_id": OPEN_POLYMER_LEGACY_SOURCE_CALL_ID,
            "source_idempotency_key": OPEN_POLYMER_LEGACY_IDEMPOTENCY_KEY,
            "source_solution_id": OPEN_POLYMER_LEGACY_SOLUTION_ID,
            "source_started_event_seq": OPEN_POLYMER_LEGACY_STARTED_EVENT_SEQ,
            "source_completed_event_seq": OPEN_POLYMER_LEGACY_COMPLETED_EVENT_SEQ,
            "source_artifact_manifest_artifact_id": OPEN_POLYMER_LEGACY_MANIFEST_ARTIFACT_ID,
            "source_artifact_manifest_sha256": OPEN_POLYMER_LEGACY_MANIFEST_SHA256,
            "source_split_manifest_artifact_id": OPEN_POLYMER_LEGACY_SPLIT_ARTIFACT_ID,
            "source_split_manifest_sha256": OPEN_POLYMER_LEGACY_SPLIT_SHA256,
            "source_independent_verification_artifact_id": OPEN_POLYMER_LEGACY_REVIEW_ARTIFACT_ID,
            "source_independent_verification_sha256": OPEN_POLYMER_LEGACY_REVIEW_SHA256,
            "historical_atomic_holdout_claim_present": False,
            "historical_ledger_fabricated": False,
            "retraining_performed": False,
        },
    }


def _conditional_board(item: dict) -> dict:
    return {
        "schema": "evomind.goal-board.v2",
        "run_id": FIXED_RUN_ID,
        "allocation": FIXED_ALLOCATION,
        "baseline_policy": CONDITIONAL_BASELINE_POLICY_MODE,
        "policy_evidence_sha256": CONDITIONAL_BASELINE_POLICY_EVIDENCE_SHA256,
        "target_count": len(COMPETITIONS),
        "weather_actions": 0,
        "competitions": [
            item if name == "open_polymer" else {
                "competition": name,
                "goal_status": "WAITING_EXACT_GATE",
                "exact_gate": "pending",
            }
            for name in COMPETITIONS
        ],
    }


def test_legacy_adoption_v2_is_honest_and_does_not_require_a_fake_ledger() -> None:
    item = _pure_item()
    result = validate_goal_item("open_polymer", item, conditional_policy=True)
    assert result.strict_verified_complete is True, [issue.to_dict() for issue in result.issues]
    assert "holdout_ledger" not in item
    assert item["existing_candidate_adoption"]["historical_atomic_holdout_claim_present"] is False
    assert item["legacy_holdout_evidence"]["posthoc_ledger_created"] is False

    board_result = validate_goal_board(
        _conditional_board(item),
        spec=load_goal_spec("configs/g21_five_competition_goal.json"),
    )
    assert board_result.complete_count == 1
    assert board_result.completion_proven is False
    assert not any(issue.code.startswith("open_polymer:") for issue in board_result.issues)


@pytest.mark.parametrize(
    ("path", "value", "issue"),
    [
        (("existing_candidate_adoption", "historical_atomic_holdout_claim_present"), True, "LEGACY_ADOPTION_BINDING"),
        (("legacy_holdout_evidence", "posthoc_ledger_created"), True, "LEGACY_HOLDOUT_FACTS"),
        (("legacy_holdout_evidence", "group_overlap"), 1, "LEGACY_HOLDOUT_FACTS"),
        (("holdout_ledger",), {"unconsumed_at_claim": True}, "POSTHOC_HOLDOUT_CLAIM"),
        (("leakage_audit", "test_labels_used"), True, "TEST_LABEL_USAGE"),
        (("official_metric_claimed",), True, "LEGACY_ADOPTION_BINDING"),
        (("primary_comparison", "ci95_upper"), BASELINE, "PRIMARY_BASELINE_CI_LOWER"),
    ],
)
def test_legacy_adoption_v2_rejects_false_or_leaky_claims(path: tuple[str, ...], value, issue: str) -> None:
    item = _pure_item()
    target = item
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    result = validate_goal_item("open_polymer", item, conditional_policy=True)
    assert result.strict_verified_complete is False
    assert any(issue in entry.code for entry in result.issues)


def test_legacy_adoption_v2_rejects_extra_manifest_file_even_when_bound() -> None:
    item = _pure_item()
    item["artifact_manifest"]["files"].append(
        {"name": "unexpected-extra.bin", "bytes": 1, "sha256": _sha("unexpected")}
    )
    item["legacy_artifact_bindings"]["unexpected-extra.bin"] = "artifact-unexpected-extra"
    result = validate_goal_item("open_polymer", item, conditional_policy=True)
    assert result.strict_verified_complete is False
    assert any("LEGACY_MANIFEST_FILE_SET_INVALID" in issue.code for issue in result.issues)


def test_legacy_adoption_v2_rejects_duplicate_artifact_binding_ids() -> None:
    item = _pure_item()
    names = sorted(item["legacy_artifact_bindings"])
    item["legacy_artifact_bindings"][names[1]] = item["legacy_artifact_bindings"][names[0]]
    result = validate_goal_item("open_polymer", item, conditional_policy=True)
    assert result.strict_verified_complete is False
    assert any("ARTIFACT_BINDING_ID_INVALID" in issue.code for issue in result.issues)


def _audit_source_fixture(item: dict) -> tuple[list[dict], list[dict], datetime]:
    started = datetime(2026, 8, 29, 15, 50, 27, 166000, tzinfo=timezone.utc)
    completed = datetime(2026, 8, 29, 15, 56, 31, 85000, tzinfo=timezone.utc)
    published = [
        {
            "id": item["legacy_artifact_bindings"][entry["name"]],
            "name": entry["name"],
            "bytes": entry["bytes"],
            "sha256": entry["sha256"],
        }
        for entry in item["artifact_manifest"]["files"]
    ] + [
        {
            "id": OPEN_POLYMER_LEGACY_MANIFEST_ARTIFACT_ID,
            "name": "artifact-manifest.json",
            "bytes": 2985,
            "sha256": OPEN_POLYMER_LEGACY_MANIFEST_SHA256,
        },
        {
            "id": OPEN_POLYMER_LEGACY_MANIFEST_RECEIPT_ARTIFACT_ID,
            "name": "artifact-manifest-receipt.json",
            "bytes": OPEN_POLYMER_LEGACY_MANIFEST_RECEIPT_BYTES,
            "sha256": OPEN_POLYMER_LEGACY_MANIFEST_RECEIPT_SHA256,
        },
    ]
    calls = [
        {
            "id": OPEN_POLYMER_LEGACY_SOURCE_CALL_ID,
            "session_id": FIXED_RUN_ID,
            "tool_name": "hpc_execute_solution",
            "status": "completed",
            "created_at": started.isoformat(),
            "started_at": started.isoformat(),
            "completed_at": completed.isoformat(),
            "idempotency_key": OPEN_POLYMER_LEGACY_IDEMPOTENCY_KEY,
            "arguments": {"competition": "open_polymer", "solution_id": OPEN_POLYMER_LEGACY_SOLUTION_ID},
            "result": {
                "ok": True,
                "content": {
                    "status": "completed",
                    "exit_code": 0,
                    "published_artifacts": published,
                },
            },
        }
    ]
    events = [
        {
            "seq": OPEN_POLYMER_LEGACY_STARTED_EVENT_SEQ,
            "event_type": "tool.started",
            "created_at": started.isoformat(),
            "payload": {"id": OPEN_POLYMER_LEGACY_SOURCE_CALL_ID},
        },
        {
            "seq": OPEN_POLYMER_LEGACY_COMPLETED_EVENT_SEQ,
            "event_type": "tool.completed",
            "created_at": completed.isoformat(),
            "payload": {"tool_call_id": OPEN_POLYMER_LEGACY_SOURCE_CALL_ID, "ok": True},
        },
    ]
    return calls, events, datetime(2026, 8, 31, tzinfo=timezone.utc)


def test_independent_audit_requires_exact_v2_events_and_publication_closure() -> None:
    item = _pure_item()
    calls, events, since = _audit_source_fixture(item)
    valid, issues = goal_audit.validate_open_polymer_adoption(
        board_item=item,
        all_calls=calls,
        all_events=events,
        since=since,
    )
    assert valid is True, issues

    valid, issues = goal_audit.validate_open_polymer_adoption(
        board_item=item,
        all_calls=calls,
        all_events=[],
        since=since,
    )
    assert valid is False
    assert any(issue["code"] == "OPEN_POLYMER_SOURCE_EVENT_BINDING_INVALID" for issue in issues)

    extra = deepcopy(calls)
    extra[0]["result"]["content"]["published_artifacts"].append(
        {"id": "artifact-unexpected", "name": "unexpected.bin", "bytes": 1, "sha256": _sha("unexpected")}
    )
    valid, issues = goal_audit.validate_open_polymer_adoption(
        board_item=item,
        all_calls=extra,
        all_events=events,
        since=since,
    )
    assert valid is False
    assert any(issue["code"] == "OPEN_POLYMER_LEGACY_PUBLICATION_SET_INVALID" for issue in issues)

    duplicate_key = deepcopy(calls)
    duplicate_key.append(
        {
            **deepcopy(calls[0]),
            "id": "call-open-polymer-duplicate-idempotency",
            "status": "failed",
            "result": {"ok": False, "content": {}},
        }
    )
    valid, issues = goal_audit.validate_open_polymer_adoption(
        board_item=item,
        all_calls=duplicate_key,
        all_events=events,
        since=since,
    )
    assert valid is False
    assert any(issue["code"] == "OPEN_POLYMER_SOURCE_IDEMPOTENCY_NOT_UNIQUE" for issue in issues)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", "run_other"),
        ("allocation", "G20"),
        ("competition", "ariel_2025"),
        ("source_split_manifest_artifact_id", "artifact-wrong-split"),
        ("source_split_manifest_sha256", "0" * 64),
        ("source_independent_verification_artifact_id", "artifact-wrong-review"),
        ("source_independent_verification_sha256", "0" * 64),
        ("historical_atomic_holdout_claim_present", True),
        ("historical_ledger_fabricated", True),
    ],
)
def test_independent_audit_rejects_every_v2_identity_or_history_drift(field: str, value) -> None:
    item = _pure_item()
    item["existing_candidate_adoption"][field] = value
    calls, events, since = _audit_source_fixture(item)
    valid, issues = goal_audit.validate_open_polymer_adoption(
        board_item=item,
        all_calls=calls,
        all_events=events,
        since=since,
    )
    assert valid is False
    assert any(issue["code"] == "OPEN_POLYMER_LEGACY_EXACT_ALLOWLIST_INVALID" for issue in issues)


def test_independent_audit_rejects_cross_run_source_call() -> None:
    item = _pure_item()
    calls, events, since = _audit_source_fixture(item)
    calls[0]["session_id"] = "run_other"
    valid, issues = goal_audit.validate_open_polymer_adoption(
        board_item=item,
        all_calls=calls,
        all_events=events,
        since=since,
    )
    assert valid is False
    assert any(issue["code"] == "OPEN_POLYMER_SOURCE_CALL_BINDING_INVALID" for issue in issues)


def _seed_store(tmp_path: Path) -> RuntimeStore:
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    now = utc_now()
    store.create_session(
        Session(
            id=FIXED_RUN_ID,
            workspace_root=str(tmp_path),
            title="legacy adoption fixture",
            objective="legacy adoption fixture",
            created_at=now,
            updated_at=now,
        )
    )
    store.create_assistant_run(
        {
            "id": FIXED_RUN_ID,
            "session_id": FIXED_RUN_ID,
            "conversation_id": "legacy_fixture",
            "prompt": "legacy fixture",
            "task_root": str(tmp_path),
            "status": "cancelled",
            "plan": {},
            "attachment_ids": [],
            "retry_count": 0,
            "error_class": "",
            "error_message": "",
            "model_provider": "",
            "model": "",
            "created_at": now,
            "updated_at": now,
            "completed_at": now,
        }
    )
    return store


def _publish(store: RuntimeStore, root: Path, artifact_id: str, name: str, raw: bytes) -> dict:
    path = root / "published" / artifact_id / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return store.publish_deliverable(
        {
            "id": artifact_id,
            "run_id": FIXED_RUN_ID,
            "session_id": FIXED_RUN_ID,
            "name": name,
            "path": str(path),
            "media_type": "application/json",
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "source_tool_call": "hpc_execute_solution",
            "preview_kind": "structured_text",
            "verified_at": utc_now(),
        }
    )


def _materialize_legacy_fixture(
    store: RuntimeStore, root: Path, monkeypatch: pytest.MonkeyPatch, *, initial_extra_publication: bool = False,
) -> dict:
    item = _pure_item()
    payloads: dict[str, bytes] = {
        "solution.py": b"# fixture\n",
        "environment-lock.json": canonical_json_bytes({"python": "fixture"}),
        "dataset-audit.json": canonical_json_bytes(
            {
                "schema": "evomind.open_polymer.dataset_audit.v2",
                "train_sha256": _sha("train"),
                "test_sha256": _sha("test"),
                "test_labels_read": False,
            }
        ),
        "model-bundle.joblib": b"fixture-model",
        "metrics.json": canonical_json_bytes(
            {
                "schema": "evomind.open_polymer.metrics.v2",
                "baseline_aggregate_normalized_rmse": BASELINE,
                "candidate_aggregate_normalized_rmse": CANDIDATE,
                "bootstrap": {
                    "difference_ci95_lower": DIFF_LOWER,
                    "difference_ci95_upper": DIFF_UPPER,
                },
            }
        ),
        "baseline-evidence.json": canonical_json_bytes(
            {
                "schema": "evomind.open_polymer.baseline_evidence.v2",
                "classification": "PUBLICLY_REPRODUCIBLE_STRONG_BASELINE",
                "implementation_sha256": "",
                "metric": METRIC,
                "reproduction_value": BASELINE,
            }
        ),
        "baseline-reproduction.json": canonical_json_bytes(
            {
                "schema": "evomind.open_polymer.baseline_evidence.v2",
                "reproduction_value": BASELINE,
            }
        ),
        "candidate-vs-baseline.json": canonical_json_bytes(
            {
                "schema": "evomind.open_polymer.candidate_vs_baseline.v2",
                "baseline": BASELINE,
                "candidate": CANDIDATE,
                "difference_candidate_minus_baseline_ci95_upper": DIFF_UPPER,
                "gate_passed": True,
            }
        ),
        "split-manifest.json": canonical_json_bytes(
            {
                "schema": "evomind.open_polymer.split_manifest.v2",
                "holdout_blind_for_selection": True,
                "group_overlap": 0,
            }
        ),
        "independent-verification.json": canonical_json_bytes(
            {
                "schema": "evomind.open_polymer.independent_verification.v2",
                "baseline_recomputed": BASELINE,
                "candidate_recomputed": CANDIDATE,
                "baseline_match": True,
                "candidate_match": True,
                "group_overlap": 0,
                "test_labels_used": False,
                "gate_passed": True,
            }
        ),
        "task-contract-v2.json": canonical_json_bytes(
            {
                "schema": "evomind.open_polymer.task_contract.v2",
                "competition": "open_polymer",
                "metric": METRIC,
                "direction": "lower_is_better",
                "hidden_test_labels_used": False,
            }
        ),
        "training-config.json": canonical_json_bytes(
            {
                "schema": "evomind.open_polymer.training_config.v2",
                "single_gpu_solution": True,
                "external_submission": False,
            }
        ),
        "retrospective-memory.json": canonical_json_bytes(
            {"schema": "evomind.retrospective_memory.v2", "gate_passed": True}
        ),
        "holdout-predictions.csv": b"id,prediction\n1,0\n",
        "training.log": b"completed\n",
    }
    for required in OPEN_POLYMER_LEGACY_REQUIRED_FILES:
        payloads.setdefault(required, f"{required}\n".encode("utf-8"))
    source_sha = hashlib.sha256(payloads["solution.py"]).hexdigest()
    baseline_value = json.loads(payloads["baseline-evidence.json"])
    baseline_value["implementation_sha256"] = source_sha
    payloads["baseline-evidence.json"] = canonical_json_bytes(baseline_value)

    bindings: dict[str, str] = {}
    rows: dict[str, dict] = {}
    for index, name in enumerate(sorted(payloads), 1):
        artifact_id = f"artifact-legacy-{index}"
        if name == "split-manifest.json":
            artifact_id = OPEN_POLYMER_LEGACY_SPLIT_ARTIFACT_ID
        elif name == "independent-verification.json":
            artifact_id = OPEN_POLYMER_LEGACY_REVIEW_ARTIFACT_ID
        row = _publish(store, root, artifact_id, name, payloads[name])
        bindings[name] = artifact_id
        rows[name] = row
    manifest = {
        "schema": "evomind.open_polymer.artifact_manifest.v2",
        "source_sha256": source_sha,
        "data_train_sha256": _sha("train"),
        "data_test_sha256": _sha("test"),
        "exit_code": 0,
        "gate_passed": True,
        "files": [
            {"name": name, "bytes": rows[name]["bytes"], "sha256": rows[name]["sha256"]}
            for name in sorted(rows)
        ],
    }
    manifest_raw = canonical_json_bytes(manifest)
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    manifest_row = _publish(
        store,
        root,
        OPEN_POLYMER_LEGACY_MANIFEST_ARTIFACT_ID,
        "artifact-manifest.json",
        manifest_raw,
    )
    receipt_raw = canonical_json_bytes(
        {
            "artifact": "artifact-manifest.json",
            "bytes": manifest_row["bytes"],
            "sha256": manifest_row["sha256"],
        }
    )
    receipt_row = _publish(
        store,
        root,
        OPEN_POLYMER_LEGACY_MANIFEST_RECEIPT_ARTIFACT_ID,
        "artifact-manifest-receipt.json",
        receipt_raw,
    )
    split_sha = rows["split-manifest.json"]["sha256"]
    review_sha = rows["independent-verification.json"]["sha256"]
    for module in (goal_contract, goal_board_module):
        monkeypatch.setattr(module, "OPEN_POLYMER_LEGACY_MANIFEST_SHA256", manifest_sha)
        monkeypatch.setattr(module, "OPEN_POLYMER_LEGACY_SPLIT_SHA256", split_sha)
        monkeypatch.setattr(module, "OPEN_POLYMER_LEGACY_REVIEW_SHA256", review_sha)
    monkeypatch.setattr(
        goal_board_module,
        "OPEN_POLYMER_LEGACY_MANIFEST_RECEIPT_SHA256",
        receipt_row["sha256"],
    )
    monkeypatch.setattr(
        goal_board_module,
        "OPEN_POLYMER_LEGACY_MANIFEST_RECEIPT_BYTES",
        receipt_row["bytes"],
    )

    item["artifact_manifest"] = manifest
    item["artifact_manifest_sha256"] = manifest_sha
    item["legacy_artifact_bindings"] = bindings
    item["legacy_holdout_evidence"]["split_manifest_sha256"] = split_sha
    item["independent_review"]["sha256"] = review_sha
    adoption = item["existing_candidate_adoption"]
    adoption["source_artifact_manifest_sha256"] = manifest_sha
    adoption["source_split_manifest_sha256"] = split_sha
    adoption["source_independent_verification_sha256"] = review_sha
    item.update(
        command_sha256=rows["training-config.json"]["sha256"],
        environment_sha256=rows["environment-lock.json"]["sha256"],
        data_manifest_sha256=rows["split-manifest.json"]["sha256"],
        source_code_sha256=source_sha,
        model_sha256=rows["model-bundle.joblib"]["sha256"],
        metrics_sha256=rows["metrics.json"]["sha256"],
        rollback_sha256=rows["retrospective-memory.json"]["sha256"],
    )
    item["primary_baseline"].update(
        source_sha256=rows["baseline-evidence.json"]["sha256"],
        evidence_artifact_id=bindings["baseline-evidence.json"],
        evidence_artifact_sha256=rows["baseline-evidence.json"]["sha256"],
        implementation_sha256=source_sha,
        data_manifest_sha256=rows["split-manifest.json"]["sha256"],
        environment_sha256=rows["environment-lock.json"]["sha256"],
        reproduction_receipt_sha256=rows["baseline-reproduction.json"]["sha256"],
    )
    human_raw = canonical_json_bytes(
        {
            "schema": "evomind.human-baseline-evidence.v1",
            "competition": "open_polymer",
            "classification": "HUMAN_BASELINE_UNDEFINED",
            "human_baseline": None,
            "unique_gate": "same-protocol verified human participant average",
        }
    )
    human_row = _publish(
        store,
        root,
        "artifact-open-polymer-human-unavailable",
        "open-polymer-human-baseline-evidence.json",
        human_raw,
    )
    item["human_baseline_availability"]["evidence_artifact_sha256"] = human_row["sha256"]

    started_at = datetime(2026, 8, 29, 15, 50, tzinfo=timezone.utc)
    completed_at = started_at + timedelta(minutes=6)
    published = [
        {
            "id": row["id"],
            "name": row["name"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for row in rows.values()
    ] + [
        {
            "id": manifest_row["id"],
            "name": manifest_row["name"],
            "bytes": manifest_row["bytes"],
            "sha256": manifest_row["sha256"],
        },
        {
            "id": receipt_row["id"],
            "name": receipt_row["name"],
            "bytes": receipt_row["bytes"],
            "sha256": receipt_row["sha256"],
        },
    ]
    if initial_extra_publication:
        # Seed an independently bad synthetic archive before its first receipt;
        # never rewrite a completed receipt or any frozen historical artifact.
        extra = _publish(store, root, "artifact-unexpected-source-extra", "unexpected.bin", b"x")
        published.append({key: extra[key] for key in ("id", "name", "bytes", "sha256")})
    call = ToolCall(
        OPEN_POLYMER_LEGACY_SOURCE_CALL_ID,
        FIXED_RUN_ID,
        "hpc_execute_solution",
        {"competition": "open_polymer", "solution_id": OPEN_POLYMER_LEGACY_SOLUTION_ID},
        "completed",
        started_at.isoformat(),
        started_at.isoformat(),
        completed_at.isoformat(),
        "",
        "open-polymer-v2-baseline-candidate-1",
    )
    store.put_tool_call(
        call,
        ToolResult(
            call.id,
            True,
            {"status": "completed", "exit_code": 0, "published_artifacts": published},
            "completed",
        ),
    )
    with store._lock, store._connection:
        store._connection.execute(
            "INSERT INTO events VALUES (?,?,?,?,?,?)",
            (
                "evt-open-polymer-started",
                FIXED_RUN_ID,
                OPEN_POLYMER_LEGACY_STARTED_EVENT_SEQ,
                "tool.started",
                json.dumps({"id": call.id}, separators=(",", ":")),
                started_at.isoformat(),
            ),
        )
        store._connection.execute(
            "INSERT INTO events VALUES (?,?,?,?,?,?)",
            (
                "evt-open-polymer-completed",
                FIXED_RUN_ID,
                OPEN_POLYMER_LEGACY_COMPLETED_EVENT_SEQ,
                "tool.completed",
                json.dumps({"tool_call_id": call.id, "ok": True}, separators=(",", ":")),
                completed_at.isoformat(),
            ),
        )
    return item


def test_store_backed_legacy_adoption_is_idempotent_and_preserves_negative_human_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _seed_store(tmp_path)
    try:
        item = _materialize_legacy_fixture(store, tmp_path, monkeypatch)
        spec = load_goal_spec("configs/g21_five_competition_goal.json")
        bootstrap = _conditional_board(
            {
                "competition": "open_polymer",
                "goal_status": "WAITING_EXACT_GATE",
                "exact_gate": "legacy adoption pending",
            }
        )
        ensure_fixed_goal(store, run_id=FIXED_RUN_ID, spec=spec, board=bootstrap)
        board = _conditional_board(item)
        updated = update_fixed_goal_board(
            store,
            goal_id=FIXED_GOAL_ID,
            board=board,
            status="blocked",
        )
        assert updated["status"] == "blocked"
        assert updated["human_baseline_sha256"] == spec["human_baseline_evidence_sha256"]
        assert validate_goal_board(board, spec=spec).complete_count == 1
        replayed = update_fixed_goal_board(
            store,
            goal_id=FIXED_GOAL_ID,
            board=board,
            status="blocked",
        )
        assert replayed == updated
        updates = [
            event
            for event in goal_board_module._all_events(store, FIXED_RUN_ID)
            if event["event_type"] == "goal.updated"
        ]
        assert len(updates) == 1
    finally:
        store.close()


def test_store_backed_legacy_adoption_rejects_post_goal_retraining(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _seed_store(tmp_path)
    try:
        item = _materialize_legacy_fixture(store, tmp_path, monkeypatch)
        spec = load_goal_spec("configs/g21_five_competition_goal.json")
        bootstrap = _conditional_board(
            {
                "competition": "open_polymer",
                "goal_status": "WAITING_EXACT_GATE",
                "exact_gate": "legacy adoption pending",
            }
        )
        ensure_fixed_goal(store, run_id=FIXED_RUN_ID, spec=spec, board=bootstrap)
        call = ToolCall(
            "call-open-polymer-post-goal",
            FIXED_RUN_ID,
            "hpc_execute_solution",
            {"competition": "open_polymer", "solution_id": "forbidden-retrain"},
            "failed",
            "!",
            "!",
            "!",
            "",
            "forbidden-retrain",
        )
        store.put_tool_call(call, ToolResult(call.id, False, {}, "blocked", "forbidden"))
        with pytest.raises(goal_board_module.GoalRecordError, match="timestamp is invalid"):
            update_fixed_goal_board(
                store,
                goal_id=FIXED_GOAL_ID,
                board=_conditional_board(item),
                status="blocked",
            )
    finally:
        store.close()


def test_store_backed_legacy_adoption_rejects_extra_source_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _seed_store(tmp_path)
    try:
        _materialize_legacy_fixture(store, tmp_path, monkeypatch)
        source = store.get_tool_call(OPEN_POLYMER_LEGACY_SOURCE_CALL_ID)
        assert source is not None
        extra = _publish(store, tmp_path, "artifact-unexpected-source-extra", "unexpected.bin", b"x")
        content = deepcopy(source["result"]["content"])
        content["published_artifacts"].append(
            {"id": extra["id"], "name": extra["name"], "bytes": extra["bytes"], "sha256": extra["sha256"]}
        )
        call = ToolCall(
            source["id"], source["session_id"], source["tool_name"], source["arguments"],
            source["status"], source["created_at"], source["started_at"], source["completed_at"],
            source["approval_id"], source["idempotency_key"],
        )
        with pytest.raises(ValueError, match="tool_call_terminal_immutable"):
            store.put_tool_call(call, ToolResult(call.id, True, content, "completed"))
        assert store.get_tool_call(source["id"]) == source
    finally:
        store.close()


def test_initial_bad_legacy_archive_still_rejects_extra_source_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _seed_store(tmp_path)
    try:
        item = _materialize_legacy_fixture(store, tmp_path, monkeypatch, initial_extra_publication=True)
        spec = load_goal_spec("configs/g21_five_competition_goal.json")
        ensure_fixed_goal(
            store,
            run_id=FIXED_RUN_ID,
            spec=spec,
            board=_conditional_board(
                {"competition": "open_polymer", "goal_status": "WAITING_EXACT_GATE", "exact_gate": "pending"}
            ),
        )
        with pytest.raises(goal_board_module.GoalRecordError, match="publication set is not closed"):
            update_fixed_goal_board(
                store,
                goal_id=FIXED_GOAL_ID,
                board=_conditional_board(item),
                status="blocked",
            )
    finally:
        store.close()


def test_store_backed_legacy_adoption_rejects_duplicate_source_idempotency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _seed_store(tmp_path)
    try:
        item = _materialize_legacy_fixture(store, tmp_path, monkeypatch)
        duplicate = ToolCall(
            "call-open-polymer-duplicate-idempotency",
            FIXED_RUN_ID,
            "hpc_execute_solution",
            {"competition": "open_polymer", "solution_id": "duplicate"},
            "failed",
            "2026-08-29T16:00:00+00:00",
            "2026-08-29T16:00:00+00:00",
            "2026-08-29T16:00:01+00:00",
            "",
            OPEN_POLYMER_LEGACY_IDEMPOTENCY_KEY,
        )
        store.put_tool_call(duplicate, ToolResult(duplicate.id, False, {}, "failed", "duplicate"))
        spec = load_goal_spec("configs/g21_five_competition_goal.json")
        ensure_fixed_goal(
            store,
            run_id=FIXED_RUN_ID,
            spec=spec,
            board=_conditional_board(
                {"competition": "open_polymer", "goal_status": "WAITING_EXACT_GATE", "exact_gate": "pending"}
            ),
        )
        with pytest.raises(goal_board_module.GoalRecordError, match="source tool call binding is invalid"):
            update_fixed_goal_board(
                store,
                goal_id=FIXED_GOAL_ID,
                board=_conditional_board(item),
                status="blocked",
            )
    finally:
        store.close()
