from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "video-production"
    / "siim-isic-melanoma-commercial-v1"
    / "scripts"
    / "verify_pre_recording_gate.py"
)
SPEC = importlib.util.spec_from_file_location("verify_pre_recording_gate", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def build_ready_fixture(root: Path) -> tuple[Path, Path, str]:
    run_id = "evomind_siim_video_gate_fixture"
    run_dir = root / "workspace" / "evomind_runs" / run_id
    run_dir.mkdir(parents=True)
    contract = write_json(root / "contract.json", {"run_id": run_id})
    write_json(run_dir / "run.json", {"run_id": run_id, "status": "completed"})
    write_json(
        run_dir / "task_graph.json",
        {
            "run_id": run_id,
            "nodes": [{"task_id": f"task_{index}", "status": "completed"} for index in range(9)],
        },
    )
    write_json(
        run_dir / "workflow_contract.json",
        {"run_id": run_id, "task_type": "image_classification", "dataset": "siim-isic-melanoma-classification"},
    )
    write_json(
        run_dir / "request.json",
        {
            "compute_policy": {"backend": "hpc", "local_gpu_allowed": False},
            "submission_policy": {"official_submission": "forbidden"},
        },
    )
    write_json(run_dir / "review.json", {"run_id": run_id, "status": "passed"})
    checks = {
        name: True
        for name in (
            "no_public_leaderboard_claim",
            "no_official_medal_claim",
            "no_clinical_diagnosis_claim",
            "private_grader_not_used_for_tuning",
            "candidate_hashes_unchanged",
            "official_submission_not_executed",
        )
    }
    write_json(run_dir / "claim_audit.json", {"run_id": run_id, "status": "passed", "checks": checks})
    write_json(run_dir / "private_grader_ledger.json", {"run_id": run_id, "execution_count": 1})
    write_json(
        run_dir / "private_grader.json",
        {
            "run_id": run_id,
            "status": "failed_closed",
            "mle_private_grader_score": None,
            "failure_reason": "fixture intentionally has no private label source",
            "official_submission_executed": False,
        },
    )
    artifacts = []
    deliveries = []
    html_path = run_dir / "research_report.html"
    html_path.write_text("EvoMind 四轮进化 R1 R2 R3 R4", encoding="utf-8")
    artifacts.append(
        {
            "path": "research_report.html",
            "bytes": html_path.stat().st_size,
            "sha256": gate.sha256_file(html_path),
        }
    )
    for index, name in enumerate(gate.DELIVERABLE_NAMES, 1):
        path = run_dir / name
        path.write_bytes(f"fixture-{index}-{name}".encode())
        digest = gate.sha256_file(path)
        record = {"name": name, "path": name, "bytes": path.stat().st_size, "sha256": digest}
        artifacts.append(record)
        deliveries.append(
            {
                **record,
                "download_url": f"/api/multi-agent/runs/{run_id}/download/{name}",
            }
        )
    write_json(
        run_dir / "deliverables.json",
        {"run_id": run_id, "status": "ready", "files": deliveries},
    )
    write_json(
        run_dir / "artifact_manifest.json",
        {
            "run_id": run_id,
            "status": "verified",
            "private_grader_execution_count": 1,
            "official_submission": "forbidden",
            "artifacts": artifacts,
        },
    )
    return contract, run_dir, run_id


def test_same_run_ready_delivery_passes_recording_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, _run_dir, run_id = build_ready_fixture(tmp_path)
    monkeypatch.setattr(
        gate,
        "verify_multiround_report",
        lambda _pdf, _html: {
            "pdf_pages": 9,
            "multiround_evolution_verified": True,
        },
    )

    result = gate.validate_gate(tmp_path, contract)

    assert result["status"] == "passed"
    assert result["recording_allowed"] is True
    assert result["run_id"] == run_id
    assert result["verified_download_count"] == 4
    assert result["private_grader_execution_count"] == 1
    assert result["official_submission_executed"] is False


def test_deliverable_mutation_blocks_recording_gate(tmp_path: Path) -> None:
    contract, run_dir, _run_id = build_ready_fixture(tmp_path)
    (run_dir / gate.DELIVERABLE_NAMES[0]).write_bytes(b"mutated")

    with pytest.raises(gate.PreRecordingGateError, match="hash mismatch"):
        gate.validate_gate(tmp_path, contract)


def test_incomplete_run_blocks_before_any_recording(tmp_path: Path) -> None:
    contract, run_dir, _run_id = build_ready_fixture(tmp_path)
    write_json(run_dir / "run.json", {"run_id": "evomind_siim_video_gate_fixture", "status": "needs_continuation"})

    with pytest.raises(gate.PreRecordingGateError, match="not completed"):
        gate.validate_gate(tmp_path, contract)


def test_task_graph_requires_exactly_nine_nodes(tmp_path: Path) -> None:
    contract, run_dir, run_id = build_ready_fixture(tmp_path)
    write_json(
        run_dir / "task_graph.json",
        {
            "run_id": run_id,
            "nodes": [
                {"task_id": f"task_{index}", "status": "completed"}
                for index in range(8)
            ],
        },
    )

    with pytest.raises(gate.PreRecordingGateError, match="nine governed nodes"):
        gate.validate_gate(tmp_path, contract)


def test_incomplete_task_graph_node_blocks_recording(tmp_path: Path) -> None:
    contract, run_dir, run_id = build_ready_fixture(tmp_path)
    nodes = [
        {"task_id": f"task_{index}", "status": "completed"}
        for index in range(9)
    ]
    nodes[-1]["status"] = "running"
    write_json(run_dir / "task_graph.json", {"run_id": run_id, "nodes": nodes})

    with pytest.raises(gate.PreRecordingGateError, match="incomplete nodes"):
        gate.validate_gate(tmp_path, contract)


def test_legacy_tasks_key_does_not_replace_real_nodes(tmp_path: Path) -> None:
    contract, run_dir, run_id = build_ready_fixture(tmp_path)
    write_json(
        run_dir / "task_graph.json",
        {
            "run_id": run_id,
            "tasks": [
                {"task_id": f"task_{index}", "status": "completed"}
                for index in range(9)
            ],
        },
    )

    with pytest.raises(gate.PreRecordingGateError, match="nine governed nodes"):
        gate.validate_gate(tmp_path, contract)
