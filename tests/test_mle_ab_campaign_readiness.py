from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from research_os.mle_ab_campaign import SCREEN_TASKS, build_config, load_campaign, preregister, sha256_file
from research_os.mlebench_phase_a import get_competition_spec, resolve_competition
from scripts import build_mle_prepared_contract as prepared_contract
from scripts import verify_mle_ab_campaign_readiness as readiness

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_COMMIT = "a" * 40
FIXTURE_VERSION = "1.0.0"
FIXTURE_SIIM_EXPECTED = {
    "manifest_sha256": "b" * 64,
    "file_count": 3,
    "total_bytes": 0,
    "train_jpeg_count": 0,
    "test_jpeg_count": 0,
}


def _campaign(tmp_path: Path) -> tuple[Path, Path]:
    canonical = tmp_path / "canonical.json"
    canonical.write_text('{"schema":"test"}\n', encoding="utf-8")
    source = ROOT / "src" / "research_os" / "experience_mcgs.py"
    config = build_config(
        namespace="evomind-mle-screen-readiness-test",
        phase="screen",
        tasks=SCREEN_TASKS,
        provider="local-model-gateway",
        source_hashes={"src/research_os/experience_mcgs.py": sha256_file(source)},
        canonical_manifest_sha256=sha256_file(canonical),
        created_at="2026-08-02T00:00:00+00:00",
    )
    return Path(preregister(tmp_path / "campaigns", config)["campaign_dir"]), canonical


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _create_contract(data_root: Path, upstream_root: Path, task_id: str) -> None:
    spec = get_competition_spec(task_id)
    resolved = resolve_competition(spec, data_root, require_exists=False, require_private=True)
    resolved.public_dir.mkdir(parents=True, exist_ok=True)
    resolved.private_dir.mkdir(parents=True, exist_ok=True)
    columns = list(spec.submission.id_columns) + list(spec.submission.passthrough_columns)
    columns += list(spec.submission.prediction_columns) or ["prediction"]
    resolved.sample_submission_path.parent.mkdir(parents=True, exist_ok=True)
    resolved.sample_submission_path.write_text(",".join(dict.fromkeys(columns)) + "\n", encoding="utf-8")
    resolved.answers_path.parent.mkdir(parents=True, exist_ok=True)
    resolved.answers_path.write_text("answer\n", encoding="utf-8")
    for candidate in (spec.train_candidates[0], spec.test_candidates[0]):
        path = resolved.competition_root.joinpath(*Path(candidate).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix:
            path.write_text("fixture\n", encoding="utf-8")
        else:
            path.mkdir(parents=True, exist_ok=True)

    zip_path = resolved.competition_root / f"{task_id}.zip"
    zip_path.write_bytes(f"official:{task_id}".encode())
    source = upstream_root / "mlebench" / "competitions" / task_id
    source.mkdir(parents=True, exist_ok=True)
    (source / "prepare.py").write_text("def prepare(raw, public, private):\n    pass\n", encoding="utf-8")
    (source / "config.yaml").write_text(f"id: {task_id}\n", encoding="utf-8")
    checksum_payload = {
        "zip": prepared_contract.md5_file(zip_path),
        "public": {
            path.name: prepared_contract.md5_file(path)
            for path in sorted(resolved.public_dir.glob("*.csv"))
        },
        "private": {
            path.name: prepared_contract.md5_file(path)
            for path in sorted(resolved.private_dir.glob("*.csv"))
        },
    }
    (source / "checksums.yaml").write_text(yaml.safe_dump(checksum_payload, sort_keys=True), encoding="utf-8")

    siim_expected = FIXTURE_SIIM_EXPECTED
    if task_id == prepared_contract.SIIM_TASK:
        public_files = sorted(resolved.public_dir.glob("*.csv"), key=lambda path: path.name)
        total_bytes = sum(path.stat().st_size for path in public_files)
        siim_expected = {**FIXTURE_SIIM_EXPECTED, "file_count": len(public_files), "total_bytes": total_bytes}
        _write_json(
            resolved.competition_root / "public_staging_inventory.json",
            {
                "schema": prepared_contract.SIIM_INVENTORY_SCHEMA,
                "competition_id": task_id,
                "manifest_sha256": siim_expected["manifest_sha256"],
                "file_count": len(public_files),
                "total_bytes": total_bytes,
                "train_jpeg_count": 0,
                "test_jpeg_count": 0,
                "entries": [{"path": path.name, "size": path.stat().st_size} for path in public_files],
            },
        )
    payload = prepared_contract.build_prepared_contract(
        resolved.competition_root,
        upstream_root,
        upstream_commit=FIXTURE_COMMIT,
        upstream_version=FIXTURE_VERSION,
        pinned_upstream_commit=FIXTURE_COMMIT,
        pinned_upstream_version=FIXTURE_VERSION,
        siim_expected=siim_expected,
    )
    assert payload["status"] == "verified"
    _write_json(resolved.competition_root / "prepared-contract.json", payload)


def _all_contracts(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    data_root = tmp_path / "prepared"
    upstream_root = tmp_path / "upstream"
    for task_id in SCREEN_TASKS:
        _create_contract(data_root, upstream_root, task_id)
    siim_root = data_root / prepared_contract.SIIM_TASK
    inventory = json.loads((siim_root / "public_staging_inventory.json").read_text(encoding="utf-8"))
    siim_expected = {
        "manifest_sha256": inventory["manifest_sha256"],
        "file_count": inventory["file_count"],
        "total_bytes": inventory["total_bytes"],
        "train_jpeg_count": 0,
        "test_jpeg_count": 0,
    }
    return data_root, upstream_root, siim_expected


def _environment(campaign: Path, path: Path) -> tuple[Path, dict[str, object]]:
    prereg, _schedule = load_campaign(campaign)
    host = {
        "schema": readiness.HOST_FINGERPRINT_SCHEMA,
        "machine_id_sha256": "c" * 64,
        "os_family": "windows",
        "cpu_architecture": "x86_64",
    }
    host_sha256 = readiness._canonical_sha256(host)
    payload = {
        "schema": readiness.EXECUTION_ENVIRONMENT_SCHEMA,
        "schema_version": 2,
        "namespace": prereg["namespace"],
        "campaign_lock_sha256": sha256_file(campaign / "campaign-lock.json"),
        "provider": "local-model-gateway",
        "model": "gpt-5.6-sol",
        "authenticated_no_completion_probe": {
            "schema": readiness.AUTH_NO_COMPLETION_PROBE_SCHEMA,
            "request_kind": "authenticated_metadata",
            "authentication_succeeded": True,
            "completion_requested": False,
            "http_status": 200,
            "response_sha256": "d" * 64,
            "observed_at": "2026-08-02T00:00:00+00:00",
        },
        "host_fingerprint": host,
        "host_fingerprint_sha256": host_sha256,
        "same_hardware_policy": {
            "schema": readiness.SAME_HARDWARE_POLICY_SCHEMA,
            "receipts_required_for_every_run": True,
            "receipt_schema": readiness.SAME_HARDWARE_RECEIPT_SCHEMA,
            "pair_key_fields": ["task_id", "seed"],
            "paired_arms": list(readiness.ARMS),
            "host_fingerprint_sha256": host_sha256,
        },
    }
    _write_json(path, payload)
    return path, host


def _evaluate(
    campaign: Path,
    canonical: Path,
    data_root: Path,
    upstream_root: Path,
    siim_expected: dict[str, object],
    *,
    environment: Path | None = None,
    receipts: Path | None = None,
) -> dict[str, object]:
    return readiness.evaluate_readiness(
        campaign,
        canonical_manifest=canonical,
        data_root=data_root,
        upstream_root=upstream_root,
        execution_environment=environment,
        same_hardware_receipts=receipts,
        expected_upstream_commit=FIXTURE_COMMIT,
        expected_upstream_version=FIXTURE_VERSION,
        upstream_commit_override=FIXTURE_COMMIT,
        upstream_version_override=FIXTURE_VERSION,
        siim_expected=siim_expected,
    )


def test_readiness_fails_closed_without_data_or_execution_identity(tmp_path: Path, monkeypatch) -> None:
    campaign, canonical = _campaign(tmp_path)
    monkeypatch.setattr(
        readiness,
        "_evaluator_info",
        lambda: {"available": True, "package": "mlebench", "version": FIXTURE_VERSION},
    )
    report = _evaluate(
        campaign,
        canonical,
        tmp_path / "missing-data",
        tmp_path / "missing-upstream",
        dict(FIXTURE_SIIM_EXPECTED),
    )
    assert report["status"] == "failed_closed"
    assert report["prepared_tasks"] == 0
    assert report["launched_task_runs"] == 0
    assert report["boundaries"] == {
        "llm_calls": 0,
        "completion_requests": 0,
        "grader_calls": 0,
        "task_runs_launched": 0,
        "private_siim_grader_calls": 0,
        "kaggle_submissions": 0,
    }
    assert all(row["reason"] == "prepared_contract_missing" for row in report["tasks"])


def test_readiness_passes_only_with_official_md5_inventory_and_bound_environment(tmp_path: Path, monkeypatch) -> None:
    campaign, canonical = _campaign(tmp_path)
    data_root, upstream_root, siim_expected = _all_contracts(tmp_path)
    environment, _host = _environment(campaign, tmp_path / "execution-environment.json")
    monkeypatch.setattr(
        readiness,
        "_evaluator_info",
        lambda: {"available": True, "package": "mlebench", "version": FIXTURE_VERSION},
    )
    report = _evaluate(
        campaign,
        canonical,
        data_root,
        upstream_root,
        siim_expected,
        environment=environment,
    )
    assert report["status"] == "ready"
    assert report["prepared_tasks"] == 6
    assert all(report["checks"].values())
    assert all(row["status"] == "ready" for row in report["tasks"])
    assert report["execution_environment"]["authenticated_no_completion_probe_verified"] is True


def test_official_md5_drift_blocks_even_when_all_paths_exist(tmp_path: Path, monkeypatch) -> None:
    campaign, canonical = _campaign(tmp_path)
    data_root, upstream_root, siim_expected = _all_contracts(tmp_path)
    environment, _host = _environment(campaign, tmp_path / "execution-environment.json")
    monkeypatch.setattr(
        readiness,
        "_evaluator_info",
        lambda: {"available": True, "package": "mlebench", "version": FIXTURE_VERSION},
    )
    target = data_root / "spooky-author-identification" / "prepared" / "public" / "train.csv"
    target.write_text("tampered\n", encoding="utf-8")
    report = _evaluate(
        campaign,
        canonical,
        data_root,
        upstream_root,
        siim_expected,
        environment=environment,
    )
    row = next(item for item in report["tasks"] if item["task_id"] == "spooky-author-identification")
    assert report["status"] == "failed_closed"
    assert row["reason"] == "official_md5_or_inventory_mismatch"
    assert row["checks"]["official_public_md5_matches"] is False


def test_siim_inventory_requires_zero_missing_wrong_size_and_part_files(tmp_path: Path, monkeypatch) -> None:
    campaign, canonical = _campaign(tmp_path)
    data_root, upstream_root, siim_expected = _all_contracts(tmp_path)
    environment, _host = _environment(campaign, tmp_path / "execution-environment.json")
    monkeypatch.setattr(
        readiness,
        "_evaluator_info",
        lambda: {"available": True, "package": "mlebench", "version": FIXTURE_VERSION},
    )
    public = data_root / prepared_contract.SIIM_TASK / "prepared" / "public"
    (public / "transfer.jpg.part").write_bytes(b"partial")
    report = _evaluate(
        campaign,
        canonical,
        data_root,
        upstream_root,
        siim_expected,
        environment=environment,
    )
    row = next(item for item in report["tasks"] if item["task_id"] == prepared_contract.SIIM_TASK)
    assert report["status"] == "failed_closed"
    assert row["siim_inventory"]["part_files"] == 1
    assert row["siim_inventory"]["missing_files"] == 0
    assert row["siim_inventory"]["wrong_size_files"] == 0


def test_environment_must_bind_namespace_lock_and_authenticated_no_completion_probe(tmp_path: Path, monkeypatch) -> None:
    campaign, canonical = _campaign(tmp_path)
    data_root, upstream_root, siim_expected = _all_contracts(tmp_path)
    environment, _host = _environment(campaign, tmp_path / "execution-environment.json")
    payload = json.loads(environment.read_text(encoding="utf-8"))
    payload["namespace"] = "evomind-wrong-namespace"
    payload["campaign_lock_sha256"] = "0" * 64
    payload["authenticated_no_completion_probe"]["completion_requested"] = True
    _write_json(environment, payload)
    monkeypatch.setattr(
        readiness,
        "_evaluator_info",
        lambda: {"available": True, "package": "mlebench", "version": FIXTURE_VERSION},
    )
    report = _evaluate(
        campaign,
        canonical,
        data_root,
        upstream_root,
        siim_expected,
        environment=environment,
    )
    assert report["status"] == "failed_closed"
    assert report["checks"]["execution_environment_namespace_and_lock_bound"] is False
    assert report["checks"]["model_gateway_authenticated_no_completion_probe_verified"] is False


def test_same_hardware_receipts_are_generated_and_bound_for_both_arms(tmp_path: Path) -> None:
    campaign, _canonical = _campaign(tmp_path)
    environment, host = _environment(campaign, tmp_path / "execution-environment.json")
    prereg, schedule = load_campaign(campaign)
    pair = [
        item
        for item in schedule["runs"]
        if item["task_id"] == SCREEN_TASKS[0] and item["seed"] == 42
    ]
    assert {item["arm"] for item in pair} == set(readiness.ARMS)
    receipts = tmp_path / "receipts"
    for item in pair:
        readiness.write_same_hardware_receipt_exclusive(
            campaign,
            target_run_id=item["run_id"],
            execution_environment=environment,
            observed_host_fingerprint=host,
            receipts_root=receipts,
            recorded_at="2026-08-02T00:00:00+00:00",
        )
        readiness.require_same_hardware_receipt_for_run(
            campaign,
            target_run_id=item["run_id"],
            execution_environment=environment,
            receipts_root=receipts,
        )
    environment_status = readiness._read_execution_environment(
        environment,
        prereg,
        campaign_lock_sha256=sha256_file(campaign / "campaign-lock.json"),
    )
    audit = readiness.verify_same_hardware_receipts(
        schedule,
        namespace=prereg["namespace"],
        campaign_lock_sha256=sha256_file(campaign / "campaign-lock.json"),
        execution_environment=environment_status,
        receipts_root=receipts,
        launched_run_ids=[item["run_id"] for item in pair],
    )
    assert audit["status"] == "verified"
    assert audit["paired_arms_same_host"] is True
    assert audit["observed_valid_receipts"] == 2

    different_host = {**host, "machine_id_sha256": "e" * 64}
    with pytest.raises(ValueError, match="observed host differs"):
        readiness.build_same_hardware_receipt(
            campaign,
            target_run_id=pair[0]["run_id"],
            execution_environment=environment,
            observed_host_fingerprint=different_host,
        )


def test_launched_run_without_same_hardware_receipt_fails_closed(tmp_path: Path, monkeypatch) -> None:
    campaign, canonical = _campaign(tmp_path)
    data_root, upstream_root, siim_expected = _all_contracts(tmp_path)
    environment, _host = _environment(campaign, tmp_path / "execution-environment.json")
    _prereg, schedule = load_campaign(campaign)
    run_id = schedule["runs"][0]["run_id"]
    run_dir = campaign / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "started.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        readiness,
        "_evaluator_info",
        lambda: {"available": True, "package": "mlebench", "version": FIXTURE_VERSION},
    )
    report = _evaluate(
        campaign,
        canonical,
        data_root,
        upstream_root,
        siim_expected,
        environment=environment,
    )
    assert report["status"] == "failed_closed"
    assert report["checks"]["same_hardware_receipts_verified_for_launched_runs"] is False
    assert report["same_hardware_receipts"]["missing_required_run_ids"] == [run_id]
