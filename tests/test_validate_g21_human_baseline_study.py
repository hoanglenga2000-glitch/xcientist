"""Focused local-only tests for the G21 human-baseline study validator.

All packages are generated under pytest's temporary directory.  They are
synthetic contract fixtures only and are never checked in or represented as
human-participant evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import validate_g21_human_baseline_study as validator


RUN_ID = "run_7b1efb878afb40f396db431e91f093a5"
ALLOCATION = "G21"
COMPETITION = "cure_bench"
STUDY_ID = "study_g21_cure_human_v1"
METRIC = "accuracy"
DIRECTION = "higher_is_better"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_json_bytes(value))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(schema: str) -> dict[str, Any]:
    return {
        "schema": schema,
        "run_id": RUN_ID,
        "allocation": ALLOCATION,
        "competition": COMPETITION,
        "study_id": STUDY_ID,
    }


def _manifest_entry(root: Path, path: str, role: str) -> dict[str, Any]:
    target = root / path
    return {
        "path": path,
        "role": role,
        "artifact_id": f"artifact_{role}",
        "bytes": target.stat().st_size,
        "sha256": _sha(target),
        "regular_file": True,
        "symlink": False,
    }


def _reseal_manifest(root: Path) -> None:
    manifest_path = root / validator.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = []
    for old in manifest["files"]:
        files.append(_manifest_entry(root, old["path"], old["role"]))
    manifest["files"] = files
    manifest["file_count"] = len(files)
    manifest["total_bytes"] = sum(item["bytes"] for item in files)
    _write_json(manifest_path, manifest)


def _build_package(tmp_path: Path) -> Path:
    """Create an ephemeral structurally valid package; it is not real evidence."""

    root = tmp_path / "study-package"
    root.mkdir()

    scorer = {
        **_identity(validator.SCORER_SCHEMA),
        "scorer_id": "cure_official_accuracy_v1",
        "metric": METRIC,
        "direction": DIRECTION,
        "primary_aggregation_unit": "human_participant",
        "participant_score_field": "score",
        "aggregation": "arithmetic_mean",
    }
    _write_json(root / "scorer.json", scorer)

    task = {
        **_identity(validator.TASK_SCHEMA),
        "task_manifest_id": "cure_blind_task_v1",
        "metric": METRIC,
        "direction": DIRECTION,
        "scorer_sha256": _sha(root / "scorer.json"),
        "participant_blind_to_labels": True,
        "participant_blind_to_candidate_outputs": True,
        "test_labels_exposed": False,
    }
    _write_json(root / "task-manifest.json", task)

    ledger = {
        **_identity(validator.HOLDOUT_LEDGER_SCHEMA),
        "ledger_id": "ledger_cure_human_v1",
        "task_manifest_sha256": _sha(root / "task-manifest.json"),
    }
    _write_json(root / "holdout-ledger.json", ledger)

    claim = {
        **_identity(validator.HOLDOUT_CLAIM_SCHEMA),
        "claim_id": "claim_cure_human_v1",
        "ledger_id": ledger["ledger_id"],
        "task_manifest_sha256": _sha(root / "task-manifest.json"),
        "ledger_sha256": _sha(root / "holdout-ledger.json"),
        "unconsumed_at_claim": True,
        "consumed_once": True,
        "selection_data_scope": "development_only",
        "prior_holdout_overlap": 0,
        "test_labels_exposed": False,
    }
    _write_json(root / "holdout-claim.json", claim)

    # This is an inert synthetic byte fixture, not a publication or participant result.
    (root / "source-publication.txt").write_text("ephemeral pytest contract fixture\n", encoding="utf-8")

    protocol_text = "Frozen official CURE accuracy protocol with participant-level aggregation."
    protocol = {
        **_identity(validator.PROTOCOL_SCHEMA),
        "protocol_id": "protocol_cure_human_v1",
        "protocol_text": protocol_text,
        "participant_type": "human",
        "participant_id_scheme": "sha256_salted_pseudonym_v1",
        "metric": METRIC,
        "direction": DIRECTION,
        "task_manifest_sha256": _sha(root / "task-manifest.json"),
        "scorer_sha256": _sha(root / "scorer.json"),
        "source_sha256": _sha(root / "source-publication.txt"),
        "holdout_ledger_sha256": _sha(root / "holdout-ledger.json"),
        "holdout_claim_sha256": _sha(root / "holdout-claim.json"),
        "bootstrap": {
            "method": validator.BOOTSTRAP_METHOD,
            "seed": validator.BOOTSTRAP_SEED,
            "rounds": validator.BOOTSTRAP_ROUNDS,
            "confidence_level": 0.95,
            "resampling_unit": "human_participant",
        },
    }
    _write_json(root / "protocol.json", protocol)

    records = [
        {
            "participant_id": "p_" + hashlib.sha256(f"pytest-participant-{index}".encode()).hexdigest(),
            "score": 0.55 + (index % 10) * 0.01,
        }
        for index in range(30)
    ]
    metrics = {
        **_identity(validator.PARTICIPANT_METRICS_SCHEMA),
        "metric": METRIC,
        "direction": DIRECTION,
        "participant_count": len(records),
        "records": records,
    }
    _write_json(root / "participant-metrics.json", metrics)

    scores = [float(item["score"]) for item in records]
    mean = sum(scores) / len(scores)
    lower, upper = validator.deterministic_bootstrap_ci(scores)
    source_sha = _sha(root / "source-publication.txt")
    human_baseline = {
        **_identity("evomind.human-baseline.v1"),
        "status": "VERIFIED",
        "participant_type": "human",
        "source_url": "https://research.organizer.edu/cure/human-study-v1",
        "source_authority": "official_organizer",
        "source_sha256": source_sha,
        "evidence_artifact_id": "artifact_human_baseline_source",
        "evidence_artifact_sha256": source_sha,
        "title": "CURE human participant baseline study",
        "sample_size": len(records),
        "protocol_id": protocol["protocol_id"],
        "protocol": protocol_text,
        "protocol_sha256": hashlib.sha256(protocol_text.encode("utf-8")).hexdigest(),
        "protocol_comparable": True,
        "metric": METRIC,
        "direction": DIRECTION,
        "mean": mean,
        "uncertainty": {
            "lower": lower,
            "upper": upper,
            "confidence_level": 0.95,
            "method": validator.BOOTSTRAP_METHOD,
            "source_sha256": source_sha,
        },
        "protocol_file_sha256": _sha(root / "protocol.json"),
        "task_manifest_sha256": _sha(root / "task-manifest.json"),
        "scorer_sha256": _sha(root / "scorer.json"),
        "holdout_ledger_sha256": _sha(root / "holdout-ledger.json"),
        "holdout_claim_sha256": _sha(root / "holdout-claim.json"),
    }
    _write_json(root / "human-baseline.json", human_baseline)

    role_paths = {
        "protocol": "protocol.json",
        "task_manifest": "task-manifest.json",
        "scorer": "scorer.json",
        "human_baseline_source": "source-publication.txt",
        "holdout_ledger": "holdout-ledger.json",
        "holdout_claim": "holdout-claim.json",
        "participant_metrics": "participant-metrics.json",
        "human_baseline": "human-baseline.json",
    }
    files = [_manifest_entry(root, path, role) for role, path in role_paths.items()]
    manifest = {
        "schema": validator.MANIFEST_SCHEMA,
        "manifest_self_excluded": True,
        "run_id": RUN_ID,
        "allocation": ALLOCATION,
        "competition": COMPETITION,
        "study_id": STUDY_ID,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    _write_json(root / validator.MANIFEST_NAME, manifest)
    return root


def _validate(root: Path) -> dict[str, Any]:
    return validator.validate_study_package(
        root,
        competition=COMPETITION,
        expected_metric=METRIC,
        expected_direction=DIRECTION,
    )


def test_valid_ephemeral_package_recomputes_stats_calls_runtime_and_writes_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _build_package(tmp_path)
    original = validator.competition_goal.validate_human_baseline
    calls: list[tuple[str, str]] = []

    def recording_runtime_validator(value: Any, *, expected_metric: str, expected_direction: str):
        calls.append((expected_metric, expected_direction))
        return original(value, expected_metric=expected_metric, expected_direction=expected_direction)

    monkeypatch.setattr(validator.competition_goal, "validate_human_baseline", recording_runtime_validator)
    receipt = _validate(root)

    assert calls == [(METRIC, DIRECTION)]
    assert receipt["status"] == "VALIDATED"
    assert receipt["participant_count"] == 30
    assert receipt["runtime_human_baseline_validation"]["valid"] is True
    assert receipt["synthetic_evidence_accepted"] is False
    assert receipt["training_authorized_by_this_receipt"] is False
    assert receipt["side_effects"] == {
        "production_writes": 0,
        "hpc_accessed": False,
        "gpu_touched": False,
        "remote_writes": 0,
        "signals_sent": 0,
        "training_tool_calls": 0,
    }

    receipt_path = tmp_path / "receipt.json"
    validator.write_validation_receipt(receipt, receipt_path, package_dir=root)
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["package_tree_sha256"] == receipt["package_tree_sha256"]


def test_cli_emits_receipt_only_after_full_pass(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _build_package(tmp_path)
    receipt_path = tmp_path / "cli-receipt.json"
    common = [
        "--package-dir",
        str(root),
        "--competition",
        COMPETITION,
        "--expected-metric",
        METRIC,
        "--expected-direction",
        DIRECTION,
        "--receipt",
        str(receipt_path),
    ]

    assert validator.main(common) == 0
    assert receipt_path.is_file()
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "VALIDATED"
    assert "participant_id" not in json.dumps(output)

    (root / "source-publication.txt").write_text("tampered after valid receipt\n", encoding="utf-8")
    rejected_receipt = tmp_path / "rejected-receipt.json"
    rejected_args = common[:-1] + [str(rejected_receipt)]
    assert validator.main(rejected_args) == 2
    assert not rejected_receipt.exists()
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "REJECTED"


def test_bootstrap_is_deterministic() -> None:
    scores = [0.1 + index / 100 for index in range(30)]
    assert validator.deterministic_bootstrap_ci(scores) == validator.deterministic_bootstrap_ci(scores)


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("too_few", "PARTICIPANT_COUNT_INVALID"),
        ("duplicate", "PARTICIPANT_ID_DUPLICATE"),
        ("not_pseudonymous", "PARTICIPANT_ID_NOT_PSEUDONYMOUS"),
        ("pii_key", "PII_KEY_FORBIDDEN"),
        ("wrong_direction", "DIRECTION_MISMATCH"),
    ],
)
def test_participant_privacy_count_and_metric_contract_fail_closed(
    tmp_path: Path, mutation: str, expected_code: str
) -> None:
    root = _build_package(tmp_path)
    metrics_path = root / "participant-metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if mutation == "too_few":
        metrics["records"] = metrics["records"][:29]
        metrics["participant_count"] = 29
    elif mutation == "duplicate":
        metrics["records"][1]["participant_id"] = metrics["records"][0]["participant_id"]
    elif mutation == "not_pseudonymous":
        metrics["records"][0]["participant_id"] = "Alice Example"
    elif mutation == "pii_key":
        metrics["records"][0]["email"] = "redacted-at-source"
    elif mutation == "wrong_direction":
        metrics["direction"] = "lower_is_better"
    _write_json(metrics_path, metrics)
    _reseal_manifest(root)

    with pytest.raises(validator.StudyValidationError) as caught:
        _validate(root)
    assert caught.value.code == expected_code


def test_file_tamper_breaks_sha_closure_before_receipt(tmp_path: Path) -> None:
    root = _build_package(tmp_path)
    receipt_path = tmp_path / "must-not-exist.json"
    with (root / "source-publication.txt").open("a", encoding="utf-8") as handle:
        handle.write("tamper\n")

    with pytest.raises(validator.StudyValidationError) as caught:
        _validate(root)
    assert caught.value.code in {"FILE_BYTES_MISMATCH", "FILE_SHA256_MISMATCH"}
    assert not receipt_path.exists()


def test_manifest_synthetic_marker_is_never_accepted_as_real_evidence(tmp_path: Path) -> None:
    root = _build_package(tmp_path)
    manifest_path = root / validator.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["synthetic_fixture"] = True
    _write_json(manifest_path, manifest)

    with pytest.raises(validator.StudyValidationError) as caught:
        _validate(root)
    assert caught.value.code == "SYNTHETIC_FIXTURE_FORBIDDEN"


def test_wrong_declared_mean_or_ci_is_rejected(tmp_path: Path) -> None:
    root = _build_package(tmp_path)
    baseline_path = root / "human-baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline["mean"] += 0.01
    _write_json(baseline_path, baseline)
    _reseal_manifest(root)

    with pytest.raises(validator.StudyValidationError) as caught:
        _validate(root)
    assert caught.value.code == "HUMAN_BASELINE_MEAN_MISMATCH"


def test_receipt_must_be_new_and_outside_closed_package(tmp_path: Path) -> None:
    root = _build_package(tmp_path)
    receipt = _validate(root)
    inside = root / "receipt.json"
    with pytest.raises(validator.StudyValidationError) as caught:
        validator.write_validation_receipt(receipt, inside, package_dir=root)
    assert caught.value.code == "RECEIPT_INSIDE_PACKAGE_FORBIDDEN"
    assert not inside.exists()

    outside = tmp_path / "receipt.json"
    outside.write_text("existing\n", encoding="utf-8")
    with pytest.raises(validator.StudyValidationError) as caught:
        validator.write_validation_receipt(receipt, outside, package_dir=root)
    assert caught.value.code == "RECEIPT_ALREADY_EXISTS"


def test_holdout_claim_must_prove_single_consumption(tmp_path: Path) -> None:
    root = _build_package(tmp_path)
    claim_path = root / "holdout-claim.json"
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    claim["consumed_once"] = False
    _write_json(claim_path, claim)

    # Rebind the only two documents that intentionally carry the claim file SHA.
    claim_sha = _sha(claim_path)
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["holdout_claim_sha256"] = claim_sha
    _write_json(protocol_path, protocol)
    baseline_path = root / "human-baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline["holdout_claim_sha256"] = claim_sha
    baseline["protocol_file_sha256"] = _sha(protocol_path)
    _write_json(baseline_path, baseline)
    _reseal_manifest(root)

    with pytest.raises(validator.StudyValidationError) as caught:
        _validate(root)
    assert caught.value.code == "HOLDOUT_SINGLE_CONSUMPTION_NOT_PROVEN"
