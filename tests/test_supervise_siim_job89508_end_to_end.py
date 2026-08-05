from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from research_agent_workstation.server.core.gpu_credentials import (
    CredentialError,
    RetryableTransportError,
)
from scripts import supervise_siim_job89508_end_to_end as supervisor

RUN_ID = "evomind_siim_supervisor_fixture"


class FakeApi:
    def __init__(
        self,
        *,
        launched: bool = True,
        campaign_state: str = "awaiting_collection_and_freeze",
        process_exists: bool = False,
        transient_grader_ingress_once: bool = False,
        conflict_stage: str | None = None,
        preflight_failure: Exception | None = None,
    ) -> None:
        self.launched = launched
        self.campaign_state = campaign_state
        self.process_exists = process_exists
        self.transient_grader_ingress_once = transient_grader_ingress_once
        self.conflict_stage = conflict_stage
        self.preflight_failure = preflight_failure
        self.completed = False
        self.preflight_done = False
        self.collection_done = False
        self.aggregation_done = False
        self.campaign_ingress_done = False
        self.grader_executed = False
        self.grader_ingress_done = False
        self.claim_done = False
        self.delivery_done = False
        self.calls: list[str] = []
        self.execution_counts: dict[str, int] = {}

    def _count(self, stage: str) -> None:
        self.execution_counts[stage] = self.execution_counts.get(stage, 0) + 1

    @staticmethod
    def _payload(status: str = "passed", **extra: Any) -> dict[str, Any]:
        return {
            "status": status,
            "run_id": RUN_ID,
            "signals_sent": 0,
            "other_processes_modified": False,
            "official_submission_executed": False,
            "kaggle_submission_executed": False,
            **extra,
        }

    def _conflict(self, stage: str) -> None:
        if self.conflict_stage == stage:
            raise supervisor.EndToEndSupervisorError(f"conflicting durable evidence at {stage}")

    def final_complete(self) -> bool:
        self.calls.append("final_complete")
        return self.completed

    def launch_ready(self) -> bool:
        self.calls.append("launch_ready")
        return self.launched

    def ensure_preflight(self) -> Mapping[str, Any]:
        self.calls.append("ensure_preflight")
        self._conflict("preflight_ingress")
        if self.preflight_failure is not None:
            raise self.preflight_failure
        if self.preflight_done:
            return self._payload("reused", stage="preflight_ingress")
        self._count("preflight_ingress")
        self.preflight_done = True
        return self._payload("staged", stage="preflight_ingress")

    def campaign_status(self) -> Mapping[str, Any]:
        self.calls.append("campaign_status")
        self._conflict("poll_campaign")
        return self._payload(
            "observed",
            state={"status": self.campaign_state},
            process_exists=self.process_exists,
        )

    def resume_campaign_if_ready(self, *, sample_interval_seconds: int) -> Mapping[str, Any]:
        self.calls.append(f"resume_campaign_if_ready:{sample_interval_seconds}")
        return self._payload("hold", hold_reasons=["fixture_hold"])

    def ensure_collection(self) -> Mapping[str, Any]:
        self.calls.append("ensure_collection")
        self._conflict("collect")
        if self.collection_done:
            return self._payload("reused", stage="collect")
        self._count("collect")
        self.collection_done = True
        return self._payload("collected", stage="collect")

    def ensure_aggregation(self) -> Mapping[str, Any]:
        self.calls.append("ensure_aggregation")
        self._conflict("aggregate")
        if self.aggregation_done:
            return self._payload("reused", stage="aggregate")
        self._count("aggregate")
        self.aggregation_done = True
        return self._payload("frozen_before_private_grader", stage="aggregate")

    def ensure_campaign_ingress(self) -> Mapping[str, Any]:
        self.calls.append("ensure_campaign_ingress")
        self._conflict("campaign_ingress_review_freeze")
        if self.campaign_ingress_done:
            return self._payload("reused", stage="campaign_ingress_review_freeze")
        self._count("campaign_ingress_review_freeze")
        self.campaign_ingress_done = True
        return self._payload("staged", stage="campaign_ingress_review_freeze")

    def ensure_private_grader_execution(self) -> Mapping[str, Any]:
        self.calls.append("ensure_private_grader_execution")
        self._conflict("private_grader_once")
        if self.grader_executed:
            return self._payload(
                "reused",
                stage="private_grader_once",
                result_path="C:/fixture/private_grader.json",
                execution_count=1,
            )
        self._count("private_grader_once")
        self.grader_executed = True
        return self._payload(
            "passed",
            stage="private_grader_once",
            result_path="C:/fixture/private_grader.json",
            execution_count=1,
        )

    def ensure_grader_ingress(self, execution: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls.append("ensure_grader_ingress")
        assert execution["execution_count"] == 1
        self._conflict("grader_ingress")
        if self.transient_grader_ingress_once:
            self.transient_grader_ingress_once = False
            raise TimeoutError("fixture SSH timeout")
        if self.grader_ingress_done:
            return self._payload("reused", stage="grader_ingress")
        self._count("grader_ingress")
        self.grader_ingress_done = True
        return self._payload("staged", stage="grader_ingress")

    def ensure_claim_audit(self) -> Mapping[str, Any]:
        self.calls.append("ensure_claim_audit")
        self._conflict("claim_audit")
        if self.claim_done:
            return self._payload("reused", claim_audit="C:/fixture/claim_audit.json")
        self._count("claim_audit")
        self.claim_done = True
        return self._payload("passed", claim_audit="C:/fixture/claim_audit.json")

    def ensure_delivery_build(self) -> Mapping[str, Any]:
        self.calls.append("ensure_delivery_build")
        self._conflict("deterministic_delivery")
        if self.delivery_done:
            return self._payload("reused", output_dir="C:/fixture/delivery")
        self._count("deterministic_delivery")
        self.delivery_done = True
        return self._payload("verified", output_dir="C:/fixture/delivery")

    def ensure_delivery_ingress(
        self,
        claim: Mapping[str, Any],
        delivery: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self.calls.append("ensure_delivery_ingress")
        assert claim["claim_audit"] and delivery["output_dir"]
        self._conflict("delivery_ingress")
        if not self.completed:
            self._count("delivery_ingress")
        self.completed = True
        return self._payload("completed", stage="delivery_ingress")


def _paths(root: Path) -> supervisor.SupervisorPaths:
    return supervisor.SupervisorPaths.build(root, RUN_ID)


def _run(root: Path, api: FakeApi) -> dict[str, Any]:
    return supervisor.supervise(
        _paths(root),
        poll_seconds=1,
        sample_interval_seconds=0,
        max_wait_seconds=0,
        max_transient_failures=3,
        once=True,
        api=api,
        require_profile=False,
    )


def _delivery_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": supervisor.sha256_file(path),
    }


def _delivery_fixture(root: Path, *, pdf_pages: int = 9, include_rounds: bool = True) -> supervisor.SupervisorPaths:
    paths = _paths(root)
    paths.delivery_dir.mkdir(parents=True)
    downloads = []
    for name in supervisor.DOWNLOAD_NAMES:
        path = paths.delivery_dir / name
        path.write_bytes((f"fixture:{name}\n" * 3).encode())
        downloads.append(_delivery_record(path))
    html = paths.delivery_dir / "research_report.html"
    markers = "EvoMind 四轮进化 R1 R2 R3 R4" if include_rounds else "legacy report"
    html.write_text(markers, encoding="utf-8")
    manifest = {
        "schema": "evomind.siim.delivery_bundle_manifest.v1",
        "run_id": RUN_ID,
        "status": "verified",
        "all_sha256_bound": True,
        "official_submission": "forbidden",
        "private_grader_execution_count": 1,
        "deliverables": downloads,
        "report_html": _delivery_record(html),
        "secret_scan": {"status": "passed"},
        "reproducibility": {"pdf_pages": pdf_pages},
    }
    (paths.delivery_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    qa_dir = paths.delivery_dir / "qa"
    qa_dir.mkdir()
    qa = {
        "run_id": RUN_ID,
        "status": "passed",
        "download_file_count": 4,
        "download_names": list(supervisor.DOWNLOAD_NAMES),
        "same_run_verified": True,
        "official_submission_executed": False,
        "pdf": {"page_count": pdf_pages},
    }
    (qa_dir / "delivery-qa.json").write_text(json.dumps(qa), encoding="utf-8")
    return paths


def test_delivery_validator_requires_nine_page_four_round_report(tmp_path: Path) -> None:
    assert supervisor.validate_delivery_bundle(_delivery_fixture(tmp_path)) is True


@pytest.mark.parametrize(
    ("pdf_pages", "include_rounds", "message"),
    [(8, True, "manifest contract"), (9, False, "evolution marker")],
)
def test_delivery_validator_blocks_stale_report(
    tmp_path: Path,
    pdf_pages: int,
    include_rounds: bool,
    message: str,
) -> None:
    paths = _delivery_fixture(tmp_path, pdf_pages=pdf_pages, include_rounds=include_rounds)
    with pytest.raises(supervisor.EndToEndSupervisorError, match=message):
        supervisor.validate_delivery_bundle(paths)


def test_complete_fake_chain_uses_the_locked_phase_order(tmp_path):
    api = FakeApi()

    state = _run(tmp_path, api)

    assert state["status"] == "completed"
    assert state["phase"] == "delivery_ingress"
    assert state["completed_stages"] == [
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
    ]
    expected = [
        "ensure_preflight",
        "campaign_status",
        "ensure_collection",
        "ensure_aggregation",
        "ensure_campaign_ingress",
        "ensure_private_grader_execution",
        "ensure_grader_ingress",
        "ensure_claim_audit",
        "ensure_delivery_build",
        "ensure_delivery_ingress",
    ]
    assert [call for call in api.calls if call.startswith("ensure_") or call == "campaign_status"] == expected
    assert api.execution_counts["private_grader_once"] == 1
    assert api.execution_counts["aggregate"] == 1
    assert api.execution_counts["deterministic_delivery"] == 1
    event_lines = _paths(tmp_path).events.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in event_lines]
    recorded = [event["phase"] for event in events if event["status"] == "running"]
    assert recorded == state["completed_stages"]
    assert all(event["signals_sent"] == 0 and event["other_processes_modified"] is False for event in events)


def test_missing_launch_only_waits_for_existing_watcher(tmp_path):
    api = FakeApi(launched=False)

    state = _run(tmp_path, api)

    assert state["status"] == "waiting"
    assert state["phase"] == "waiting_launch"
    assert api.calls == ["final_complete", "launch_ready"]
    assert not api.execution_counts
    assert not [call for call in api.calls if "resume_campaign" in call]


def test_production_adapter_needs_no_credentials_while_launch_is_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("EVOMIND_HPC_CREDENTIAL_PROFILE", raising=False)
    paths = _paths(tmp_path)

    state = supervisor.supervise(
        paths,
        poll_seconds=1,
        sample_interval_seconds=0,
        max_wait_seconds=0,
        max_transient_failures=3,
        once=True,
        require_profile=True,
    )

    assert state["status"] == "waiting"
    assert state["phase"] == "waiting_launch"


def test_restart_reuses_aggregate_grader_and_delivery(tmp_path):
    api = FakeApi()
    assert _run(tmp_path, api)["status"] == "completed"
    counts = dict(api.execution_counts)
    api.calls.clear()

    restarted = _run(tmp_path, api)

    assert restarted["status"] == "completed"
    assert api.execution_counts == counts
    assert api.calls == ["final_complete"]
    persisted = json.loads(_paths(tmp_path).state.read_text(encoding="utf-8"))
    assert persisted["status"] == "completed"


def test_transient_after_grader_never_executes_grader_twice(tmp_path):
    api = FakeApi(transient_grader_ingress_once=True)

    interrupted = _run(tmp_path, api)
    assert interrupted["status"] == "retryable_error"
    assert api.execution_counts["private_grader_once"] == 1

    completed = _run(tmp_path, api)

    assert completed["status"] == "completed"
    assert api.execution_counts["private_grader_once"] == 1
    assert api.execution_counts["grader_ingress"] == 1
    assert api.execution_counts["deterministic_delivery"] == 1


def test_conflicting_aggregate_evidence_fails_closed(tmp_path):
    api = FakeApi(conflict_stage="aggregate")

    with pytest.raises(supervisor.EndToEndSupervisorError, match="conflicting durable evidence"):
        _run(tmp_path, api)

    state = json.loads(_paths(tmp_path).state.read_text(encoding="utf-8"))
    assert state["status"] == "failed_closed"
    assert state["phase"] == "evidence_validation"
    assert api.execution_counts["collect"] == 1
    assert "aggregate" not in api.execution_counts


def test_exclusive_lock_rejects_second_supervisor(tmp_path):
    paths = _paths(tmp_path)
    paths.campaign_dir.mkdir(parents=True)
    api = FakeApi()

    with supervisor.exclusive_process_lock(paths.lock):
        with pytest.raises(supervisor.EndToEndSupervisorError, match="already running"):
            _run(tmp_path, api)

    assert not api.calls


def test_active_training_only_polls_and_does_not_collect(tmp_path):
    api = FakeApi(campaign_state="formal_training", process_exists=True)

    state = _run(tmp_path, api)

    assert state["status"] == "running"
    assert state["phase"] == "poll_campaign"
    assert "ensure_collection" not in api.calls
    assert "ensure_aggregation" not in api.calls


def test_continuation_hold_does_not_advance_or_bypass_gate(tmp_path):
    api = FakeApi(campaign_state="needs_continuation", process_exists=False)

    state = _run(tmp_path, api)

    assert state["status"] == "waiting"
    assert state["phase"] == "poll_campaign"
    assert "resume_campaign_if_ready:0" in api.calls
    assert "ensure_collection" not in api.calls
    assert api.execution_counts == {"preflight_ingress": 1}


@pytest.mark.parametrize("remote_status", ["missing", "future_completed", "typo_state"])
def test_unknown_remote_state_fails_closed(tmp_path, remote_status):
    api = FakeApi(campaign_state=remote_status, process_exists=False)

    with pytest.raises(supervisor.EndToEndSupervisorError, match="published unknown state"):
        _run(tmp_path, api)

    state = json.loads(_paths(tmp_path).state.read_text(encoding="utf-8"))
    assert state["status"] == "failed_closed"
    assert state["phase"] == "evidence_validation"
    assert "ensure_collection" not in api.calls


def test_active_remote_state_without_process_fails_closed(tmp_path):
    api = FakeApi(campaign_state="ablation_training", process_exists=False)

    with pytest.raises(supervisor.EndToEndSupervisorError, match="stopped unexpectedly"):
        _run(tmp_path, api)

    state = json.loads(_paths(tmp_path).state.read_text(encoding="utf-8"))
    assert state["status"] == "failed_closed"
    assert state["phase"] == "evidence_validation"
    assert "ensure_collection" not in api.calls



def test_retryable_socks_credential_transport_error_is_retried(tmp_path):
    api = FakeApi(
        preflight_failure=RetryableTransportError("SOCKS5 connection closed during handshake")
    )

    state = _run(tmp_path, api)

    assert state["status"] == "retryable_error"
    assert state["phase"] == "remote_transport"
    assert state["transient_failures"] == 1


def test_socks_authentication_credential_error_fails_closed(tmp_path):
    api = FakeApi(
        preflight_failure=CredentialError("SOCKS5 username/password authentication failed")
    )

    with pytest.raises(CredentialError, match="authentication failed"):
        _run(tmp_path, api)

    state = json.loads(_paths(tmp_path).state.read_text(encoding="utf-8"))
    assert state["status"] == "failed_closed"
    assert state["phase"] == "evidence_validation"


def test_ordinary_credential_error_fails_closed(tmp_path):
    api = FakeApi(preflight_failure=CredentialError("pinned known-hosts file is missing"))

    with pytest.raises(CredentialError, match="known-hosts"):
        _run(tmp_path, api)

    state = json.loads(_paths(tmp_path).state.read_text(encoding="utf-8"))
    assert state["status"] == "failed_closed"
    assert state["phase"] == "evidence_validation"


def _final_run_fixture(tmp_path: Path) -> supervisor.SupervisorPaths:
    paths = _paths(tmp_path)
    paths.run_dir.mkdir(parents=True)
    paths.campaign_dir.mkdir(parents=True)
    paths.delivery_dir.mkdir(parents=True)
    paths.candidate_dir.mkdir(parents=True)
    tasks = {task_id: {"status": "completed"} for task_id in supervisor.REQUIRED_FINAL_TASKS}
    (paths.run_dir / "run.json").write_text(
        json.dumps({"run_id": RUN_ID, "status": "completed", "tasks": tasks}),
        encoding="utf-8",
    )

    def record(path: Path, relative: str) -> dict[str, Any]:
        return {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": supervisor.sha256_file(path),
        }

    artifacts = []
    for name in supervisor.DOWNLOAD_NAMES:
        path = paths.run_dir / name
        path.write_bytes(f"fixture:{name}".encode())
        artifacts.append(record(path, name))
    (paths.run_dir / "artifact_manifest.json").write_text(
        json.dumps({
            "schema": "evomind.siim.artifact_manifest.v1",
            "run_id": RUN_ID,
            "status": "verified",
            "official_submission": "forbidden",
            "private_grader_execution_count": 1,
            "artifacts": artifacts,
        }),
        encoding="utf-8",
    )
    sources = []
    for name in ("candidate_freeze.json", "review.json", "private_grader.json"):
        path = paths.run_dir / name
        path.write_text(json.dumps({"run_id": RUN_ID, "name": name}), encoding="utf-8")
        sources.append(record(path, name))
    claim = paths.run_dir / "claim_audit_source.json"
    claim.write_text(json.dumps({"run_id": RUN_ID, "status": "passed"}), encoding="utf-8")
    sources.append(record(claim, "claim_audit.json"))
    (paths.delivery_dir / "artifact_manifest.json").write_text(
        json.dumps({"source_evidence": sources}),
        encoding="utf-8",
    )
    return paths


def test_final_validation_rechecks_all_critical_evidence_layers(tmp_path, monkeypatch):
    paths = _final_run_fixture(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(supervisor, "validate_collection", lambda _paths: calls.append("collection") or True)
    monkeypatch.setattr(
        supervisor.ingress,
        "validate_candidate_manifest",
        lambda _path, _run_id: calls.append("candidate"),
    )
    monkeypatch.setattr(
        supervisor,
        "validate_workflow_grader",
        lambda _paths: calls.append("grader") or True,
    )
    monkeypatch.setattr(
        supervisor,
        "validate_delivery_bundle",
        lambda _paths: calls.append("delivery_bundle") or True,
    )
    monkeypatch.setattr(
        supervisor,
        "ingress_stage_verified",
        lambda _paths, stage: calls.append(f"stage:{stage}") or True,
    )

    assert supervisor.validate_final_run(paths) is True
    assert calls == [
        "collection",
        "candidate",
        "grader",
        "delivery_bundle",
        "stage:campaign",
        "stage:independent_review",
        "stage:private_grader",
        "stage:delivery",
    ]


def test_final_validation_rejects_missing_delivery_ingress_seal(tmp_path, monkeypatch):
    paths = _final_run_fixture(tmp_path)
    monkeypatch.setattr(supervisor, "validate_collection", lambda _paths: True)
    monkeypatch.setattr(supervisor.ingress, "validate_candidate_manifest", lambda _path, _run_id: None)
    monkeypatch.setattr(supervisor, "validate_workflow_grader", lambda _paths: True)
    monkeypatch.setattr(supervisor, "validate_delivery_bundle", lambda _paths: True)
    monkeypatch.setattr(
        supervisor,
        "ingress_stage_verified",
        lambda _paths, stage: stage != "delivery",
    )

    with pytest.raises(supervisor.EndToEndSupervisorError, match="delivery"):
        supervisor.validate_final_run(paths)
