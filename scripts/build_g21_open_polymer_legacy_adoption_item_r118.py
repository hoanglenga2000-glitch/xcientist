from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evomind_runtime.competition_goal import (  # noqa: E402
    FIXED_ALLOCATION,
    FIXED_RUN_ID,
    OPEN_POLYMER_LEGACY_ADOPTION_SCHEMA,
    OPEN_POLYMER_LEGACY_COMPLETED_EVENT_SEQ,
    OPEN_POLYMER_LEGACY_IDEMPOTENCY_KEY,
    OPEN_POLYMER_LEGACY_MANIFEST_ARTIFACT_ID,
    OPEN_POLYMER_LEGACY_MANIFEST_SHA256,
    OPEN_POLYMER_LEGACY_REVIEW_ARTIFACT_ID,
    OPEN_POLYMER_LEGACY_REVIEW_SHA256,
    OPEN_POLYMER_LEGACY_SOLUTION_ID,
    OPEN_POLYMER_LEGACY_SOURCE_CALL_ID,
    OPEN_POLYMER_LEGACY_SPLIT_ARTIFACT_ID,
    OPEN_POLYMER_LEGACY_SPLIT_SHA256,
    OPEN_POLYMER_LEGACY_STARTED_EVENT_SEQ,
    validate_goal_item,
)


METRIC = "mean per-target RMSE normalized by development-target standard deviation"
BASELINE = 0.7326585287082159
CANDIDATE = 0.6535338608202899
DIFF_CI95_LOWER = -0.11084084061270712
DIFF_CI95_UPPER = -0.050317963973942365
SOURCE_CODE_SHA256 = "2cba9c5dc9b28ac62c31ffee4d11e06a4f353157ea8807388c05d96ad044f1c4"
TRAIN_SHA256 = "1f79c85c785698e8c3499d99721adfe3be9660a487f137a923dd34eb7ef845e1"
TEST_SHA256 = "f62367095efa2beae8af3ddbf3a7a6063c7be65034bdfc1fce9b2def29203bf7"
HUMAN_AVAILABILITY_ARTIFACT_ID = "artifact_c331a54c3372496b8660138a0998e9ba"
HUMAN_AVAILABILITY_SHA256 = "b5f9920300106a92ff0dc9cff1f840953b2940ace5527d7988e6ea91bd03d461"


FILES: tuple[tuple[str, int, str, str], ...] = (
    ("baseline-evidence.json", 563, "f0e663fe52f147a9dea9d978d5201390d7c908f9babb941a846746918cfed56e", "artifact_bdef5370e5404974b57ce433e580f7d7"),
    ("baseline-reproduction.json", 4838, "23ab58532265e1c6e19a5ca27650a56417ca1aff4864f4f6608890607b90e9af", "artifact_dac8aa7a4c2e4972a8c8b9e4d22dacc5"),
    ("candidate-history.jsonl", 3207, "a76b3b96251e60935e71e6fac8604bc7699454d0b0b14ad207345efc2021f34e", "artifact_62380e0b147e4a1191632dccdcb777cd"),
    ("candidate-vs-baseline.json", 367, "f02ed5ff7b4ebda84ea449a89a6607873bcfc40b847f2d6bd1ae56d82216b49a", "artifact_657c74134a5940db937d9a11a2509b50"),
    ("dataset-audit.json", 1142, "ab641c6ec2d4b618611d2e18d614dbe4aa3d20b85cf4e7d843662f81603e60f4", "artifact_c9202384e4024853b87b10b91f8da63e"),
    ("environment-lock.json", 254, "e2ab2b6f6e2f389dceff5c92a5420379ac17a36bd4b27a6b63ffe373906aad9a", "artifact_86b6e5631e524df9a6ac21c5cd097c2b"),
    ("holdout-predictions.csv", 122651, "349a9b1fdd0407d5db03f133f86645cf7957f44ca5990acb60296110e8c80b4d", "artifact_031946dd81f34d95b668a5b0108c02d7"),
    ("independent-verification.json", 811, OPEN_POLYMER_LEGACY_REVIEW_SHA256, OPEN_POLYMER_LEGACY_REVIEW_ARTIFACT_ID),
    ("metrics.json", 4625, "b240500ee4a68e448d951bec983b1adfd7caea7ab907fa94602c15593cf77c27", "artifact_71aa82f3f6f54e899dbd77be4cdf62a2"),
    ("model-bundle.joblib", 81159442, "83431d47bbfdf79df79b2168cfbd3bbb87761d12ff91dd02cdd0be7958d9e8f2", "artifact_b07a2003012f49489864ec48a2346d83"),
    ("retrospective-memory.json", 402, "4e8ce97591ebf144411916c61333c3a293240e6ed04709bc285d4f477f7851ce", "artifact_8c849c2917424b5d8613a8666dd094d2"),
    ("solution.py", 20243, SOURCE_CODE_SHA256, "artifact_026618d3a79a4acdb58ce19003ad35c2"),
    ("split-manifest.json", 453, OPEN_POLYMER_LEGACY_SPLIT_SHA256, OPEN_POLYMER_LEGACY_SPLIT_ARTIFACT_ID),
    ("task-contract-v2.json", 442, "fa757be601aaa1d1efdd3811c9f4250e7d505280fcc128e544118d8fcceca111", "artifact_0a56a94c4a6f4535bc9de3933891aef5"),
    ("training-config.json", 357, "65ba71f2e124005a08baae58b53198c957b9e5dd36d30898043d48cba913bab4", "artifact_96e142a8bb304f50b19f62474ddd9fe6"),
    ("training.log", 914, "7834cba2336bf658cbc949788ab6a435e12a70b69998312150aed92fdfd26e9b", "artifact_6276fab622bf4ea69023fbb58b67c72c"),
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_item() -> dict[str, Any]:
    by_name = {name: {"bytes": size, "sha256": digest, "artifact_id": artifact_id} for name, size, digest, artifact_id in FILES}
    protocol = (
        "The baseline and candidate use the same SHA-bound exact-SMILES group split, "
        "the same five targets, and mean per-target RMSE normalized by the development-target standard deviation."
    )
    manifest = {
        "schema": "evomind.open_polymer.artifact_manifest.v2",
        "started_at_utc": "2026-08-29T15:50:56.754468+00:00",
        "completed_at_utc": "2026-08-29T15:54:09.666467+00:00",
        "source_sha256": SOURCE_CODE_SHA256,
        "data_train_sha256": TRAIN_SHA256,
        "data_test_sha256": TEST_SHA256,
        "exit_code": 0,
        "gate_passed": True,
        "files": [
            {"name": name, "bytes": size, "sha256": digest}
            for name, size, digest, _ in FILES
        ],
    }
    item: dict[str, Any] = {
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
            "source_sha256": by_name["baseline-evidence.json"]["sha256"],
            "evidence_artifact_id": by_name["baseline-evidence.json"]["artifact_id"],
            "evidence_artifact_sha256": by_name["baseline-evidence.json"]["sha256"],
            "title": "SHA-bound char-ngram Ridge strong baseline on Open Polymer data",
            "protocol_id": "open-polymer-internal-normalized-rmse-v2",
            "protocol": protocol,
            "protocol_sha256": sha256_bytes(protocol.encode("utf-8")),
            "protocol_comparable": True,
            "metric": METRIC,
            "direction": "lower_is_better",
            "value": BASELINE,
            "publicly_reproducible": True,
            "implementation_sha256": SOURCE_CODE_SHA256,
            "data_manifest_sha256": by_name["split-manifest.json"]["sha256"],
            "environment_sha256": by_name["environment-lock.json"]["sha256"],
            "reproduction_receipt_sha256": by_name["baseline-reproduction.json"]["sha256"],
        },
        "primary_comparison": {
            "candidate_metric": CANDIDATE,
            "ci95_lower": BASELINE + DIFF_CI95_LOWER,
            "ci95_upper": BASELINE + DIFF_CI95_UPPER,
        },
        "human_baseline_availability": {
            "status": "NOT_VERIFIED_AVAILABLE",
            "audit_complete": True,
            "protocol_search_complete": True,
            "human_baseline_verified": False,
            "evidence_artifact_id": HUMAN_AVAILABILITY_ARTIFACT_ID,
            "evidence_artifact_sha256": HUMAN_AVAILABILITY_SHA256,
        },
        "artifact_manifest": manifest,
        "artifact_manifest_artifact_id": OPEN_POLYMER_LEGACY_MANIFEST_ARTIFACT_ID,
        "artifact_manifest_sha256": OPEN_POLYMER_LEGACY_MANIFEST_SHA256,
        "legacy_artifact_bindings": {name: artifact_id for name, _, _, artifact_id in FILES},
        "command_sha256": by_name["training-config.json"]["sha256"],
        "environment_sha256": by_name["environment-lock.json"]["sha256"],
        "data_manifest_sha256": by_name["split-manifest.json"]["sha256"],
        "source_code_sha256": SOURCE_CODE_SHA256,
        "model_sha256": by_name["model-bundle.joblib"]["sha256"],
        "metrics_sha256": by_name["metrics.json"]["sha256"],
        "rollback_sha256": by_name["retrospective-memory.json"]["sha256"],
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
    result = validate_goal_item("open_polymer", item, conditional_policy=True)
    if not result.strict_verified_complete:
        raise ValueError(
            "OPEN_POLYMER_ADOPTION_ITEM_INVALID:"
            + ",".join(issue.code for issue in result.issues)
        )
    return item


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the fixed R118 Open Polymer adoption item")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("OUTPUT_EXISTS")
    item = build_item()
    raw = (json.dumps(item, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)
    print(
        json.dumps(
            {
                "schema": "evomind.open_polymer.legacy_adoption_item_build.r118.v1",
                "status": "built_and_validated",
                "path": str(output),
                "bytes": len(raw),
                "sha256": sha256_bytes(raw),
                "training_tool_calls": 0,
                "hpc_accessed": False,
                "gpu_touched": False,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
