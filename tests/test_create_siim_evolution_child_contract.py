from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import create_siim_evolution_child_contract as contract
from research_os.agent.siim_hpc_workflow import run_siim_hpc_research
from xsci.user_request import parse_user_request


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _parent_fixture(root: Path, run_id: str) -> Path:
    parent = root / "workspace" / "evomind_runs" / run_id
    tasks = {f"task_{index}": {"status": "completed"} for index in range(9)}
    payloads = {
        "run.json": {
            "run_id": run_id,
            "status": "completed",
            "task_type": "image_classification",
            "dataset": "siim-isic-melanoma-classification",
            "tasks": tasks,
        },
        "request.json": {"run_id": run_id, "task_type": "image_classification"},
        "metrics.json": {"run_id": run_id, "roc_auc": 0.92, "pr_auc": 0.24},
        "review.json": {"run_id": run_id, "status": "review_passed"},
        "claim_audit.json": {"run_id": run_id, "status": "passed"},
        "candidate_freeze.json": {
            "run_id": run_id,
            "status": "frozen_before_private_grader",
            "tuning_closed": True,
        },
        "private_grader.json": {
            "run_id": run_id,
            "status": "failed_closed",
            "official_submission_executed": False,
            "score": None,
        },
        "private_grader_ledger.json": {
            "run_id": run_id,
            "execution_count": 1,
        },
        "artifact_manifest.json": {"run_id": run_id, "status": "verified"},
        "post_completion_reconciliation.json": {"run_id": run_id, "status": "passed"},
    }
    for name, payload in payloads.items():
        _write_json(parent / name, payload)
    (parent / "submission.csv").write_text("image_name,target\nfixture,0.5\n", encoding="utf-8")
    return parent


def test_reserve_and_verify_preserve_parent_and_keep_submission_closed(tmp_path):
    parent_id = "siim_parent"
    child_id = "siim_child_r2"
    parent = _parent_fixture(tmp_path, parent_id)
    before = {path.name: path.read_bytes() for path in parent.iterdir()}

    reservation = contract.reserve(
        tmp_path,
        parent_run_id=parent_id,
        child_run_id=child_id,
        purpose="evolution candidate",
    )
    verification = contract.verify(tmp_path, child_run_id=child_id)

    assert verification["passed"] is True
    assert verification["official_submission"] == "forbidden"
    assert verification["private_grader"] == "once_after_candidate_freeze"
    assert reservation["parent_file_count"] == len(before)
    assert {path.name: path.read_bytes() for path in parent.iterdir()} == before
    assert not (tmp_path / "workspace" / "evomind_runs" / child_id).exists()


def test_verify_fails_closed_after_parent_mutation(tmp_path):
    parent_id = "siim_parent"
    child_id = "siim_child_r2"
    parent = _parent_fixture(tmp_path, parent_id)
    contract.reserve(
        tmp_path,
        parent_run_id=parent_id,
        child_run_id=child_id,
        purpose="evolution candidate",
    )
    (parent / "metrics.json").write_text("{}", encoding="utf-8")

    with pytest.raises(contract.EvolutionContractError, match="changed after reservation"):
        contract.verify(tmp_path, child_run_id=child_id)


def test_reservation_rejects_second_child_control_directory(tmp_path):
    _parent_fixture(tmp_path, "siim_parent")
    contract.reserve(
        tmp_path,
        parent_run_id="siim_parent",
        child_run_id="siim_child_r2",
        purpose="evolution candidate",
    )
    with pytest.raises(contract.EvolutionContractError, match="already exists"):
        contract.reserve(
            tmp_path,
            parent_run_id="siim_parent",
            child_run_id="siim_child_r2",
            purpose="evolution candidate",
        )


def test_rescind_is_append_only_idempotent_and_blocks_verification(tmp_path):
    parent_id = "siim_parent"
    child_id = "siim_child_r2"
    parent = _parent_fixture(tmp_path, parent_id)
    before = {
        path.relative_to(parent).as_posix(): path.read_bytes()
        for path in parent.rglob("*")
        if path.is_file()
    }
    contract.reserve(
        tmp_path,
        parent_run_id=parent_id,
        child_run_id=child_id,
        purpose="evolution candidate",
    )

    first = contract.rescind(
        tmp_path,
        child_run_id=child_id,
        reason="single Run policy restored",
        goal_id="goal-fixture",
    )
    second = contract.rescind(
        tmp_path,
        child_run_id=child_id,
        reason="single Run policy restored",
        goal_id="goal-fixture",
    )

    assert first["status"] == "rescinded"
    assert second["status"] == "already_rescinded"
    assert first["rescission_sha256"] == second["rescission_sha256"]
    assert not (tmp_path / "workspace" / "evomind_runs" / child_id).exists()
    assert {
        path.relative_to(parent).as_posix(): path.read_bytes()
        for path in parent.rglob("*")
        if path.is_file()
    } == before
    with pytest.raises(contract.EvolutionContractError, match="was rescinded"):
        contract.verify(tmp_path, child_run_id=child_id)


def test_rescind_rejects_an_existing_formal_child_run(tmp_path):
    parent_id = "siim_parent"
    child_id = "siim_child_r2"
    _parent_fixture(tmp_path, parent_id)
    contract.reserve(
        tmp_path,
        parent_run_id=parent_id,
        child_run_id=child_id,
        purpose="evolution candidate",
    )
    (tmp_path / "workspace" / "evomind_runs" / child_id).mkdir(parents=True)

    with pytest.raises(contract.EvolutionContractError, match="already exists"):
        contract.rescind(
            tmp_path,
            child_run_id=child_id,
            reason="too late",
        )


def test_siim_child_run_binds_lineage_and_never_reuses_existing_directory(tmp_path):
    parent_id = "siim_parent"
    child_id = "siim_child_r2"
    parent = _parent_fixture(tmp_path, parent_id)
    before = {
        path.relative_to(parent).as_posix(): path.read_bytes()
        for path in parent.rglob("*")
        if path.is_file()
    }
    reservation = contract.reserve(
        tmp_path,
        parent_run_id=parent_id,
        child_run_id=child_id,
        purpose="evolution candidate",
    )
    control = Path(reservation["control_dir"])
    _write_json(
        control / "requested_change.json",
        {
            "schema": "evomind.siim.requested_change.v1",
            "run_id": child_id,
            "parent_run_id": parent_id,
            "status": "frozen",
            "changes": {"image_size": {"from": 384, "to": 512}},
        },
    )
    request = parse_user_request(
        "请对 SIIM-ISIC 黑色素瘤医学影像在 HPC 上继续分析、比较并训练，"
        "完成独立复核和报告，不提交 Kaggle。"
    )

    result = run_siim_hpc_research(
        tmp_path,
        request,
        run_id=child_id,
        parent_run_id=parent_id,
    )

    child = tmp_path / "workspace" / "evomind_runs" / child_id
    lineage = json.loads((child / "lineage.json").read_text(encoding="utf-8"))
    freeze_inputs = json.loads((child / "parent_preservation.json").read_text(encoding="utf-8"))
    after = {
        path.relative_to(parent).as_posix(): path.read_bytes()
        for path in parent.rglob("*")
        if path.is_file()
    }
    assert result.status == "needs_continuation"
    assert lineage["parent_run_id"] == parent_id
    assert lineage["official_submission"] == "forbidden"
    assert freeze_inputs["unchanged"] is True
    assert before == after
    with pytest.raises(FileExistsError):
        run_siim_hpc_research(
            tmp_path,
            request,
            run_id=child_id,
            parent_run_id=parent_id,
        )
