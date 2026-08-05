from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "video-production"
    / "siim-isic-melanoma-commercial-v1"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))
import build_narration_manifest as narration  # noqa: E402
import verify_pre_recording_gate as gate  # noqa: E402


@pytest.fixture(autouse=True)
def stub_pre_recording_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    def _validated(_project_root: Path, contract_path: Path) -> dict:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        return {
            "run_id": contract["run_id"],
            "status": "passed",
            "recording_allowed": True,
        }

    monkeypatch.setattr(narration, "validate_gate", _validated)


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_ready_fixture(root: Path) -> tuple[Path, Path, str]:
    run_id = "evomind_siim_narration_fixture"
    run_dir = root / "workspace" / "evomind_runs" / run_id
    run_dir.mkdir(parents=True)
    contract = write_json(root / "production-contract.json", {"run_id": run_id})
    write_json(run_dir / "run.json", {"run_id": run_id, "status": "completed"})
    write_json(
        run_dir / "task_graph.json",
        {
            "run_id": run_id,
            "nodes": [
                {"task_id": f"task_{index}", "status": "completed"}
                for index in range(9)
            ],
        },
    )
    write_json(
        run_dir / "workflow_contract.json",
        {
            "run_id": run_id,
            "task_type": "image_classification",
            "dataset": "siim-isic-melanoma-classification",
        },
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
    write_json(
        run_dir / "claim_audit.json",
        {"run_id": run_id, "status": "passed", "checks": checks},
    )
    write_json(
        run_dir / "private_grader_ledger.json",
        {"run_id": run_id, "execution_count": 1},
    )
    write_json(
        run_dir / "private_grader.json",
        {
            "run_id": run_id,
            "mle_private_grader_score": 0.92345,
            "official_submission_executed": False,
        },
    )
    write_json(
        run_dir / "dataset_profile.json",
        {
            "run_id": run_id,
            "positive_rate": 513 / 28_984,
            "counts": {
                "files": 33_129,
                "train_images": 28_984,
                "test_images": 4_142,
                "positive_rows": 513,
                "patients": 2_056,
            },
        },
    )
    write_json(
        run_dir / "experiment_comparison.json",
        {"run_id": run_id, "selected_profile": "robust_multiview_v1"},
    )
    write_json(
        run_dir / "metrics.json",
        {
            "run_id": run_id,
            "roc_auc": 0.94123,
            "pr_auc": 0.31234,
            "historical_thresholds": {
                "bronze": 0.9370,
                "silver": 0.9401,
                "gold": 0.9455,
            },
            "patient_grouped_bootstrap_roc_auc_95ci": {
                "lower": 0.921,
                "upper": 0.958,
            },
        },
    )
    write_json(
        run_dir / "hpc_runtime.json",
        {
            "run_id": run_id,
            "training_progress": {
                "status": "completed",
                "completed_formal_seeds": 3,
                "completed_outer_folds": 15,
            },
        },
    )

    evidence_files = {
        "workflow_contract.json": json.dumps(
            {
                "run_id": run_id,
                "task_type": "image_classification",
                "dataset": "siim-isic-melanoma-classification",
            }
        ).encode(),
        "task_graph.json": json.dumps(
            {
                "run_id": run_id,
                "nodes": [
                    {"task_id": f"task_{index}", "status": "completed"}
                    for index in range(9)
                ],
            }
        ).encode(),
        "data_audit.json": b"{}",
        "preprocessing_ablation.json": b"{}",
        "training_history.json": b"{}",
        "hpc_telemetry.jsonl": b'{"gpu": {}}\n',
        "fold_metrics.csv": b"fold,roc_auc\n0,0.94\n",
        "candidate_freeze.json": b"{}",
        "research_report.html": b"<html>EvoMind four-round fixture</html>",
    }
    for name, content in evidence_files.items():
        (run_dir / name).write_bytes(content)

    artifacts = []
    deliveries = []
    html_path = run_dir / "research_report.html"
    artifacts.append(
        {
            "path": "research_report.html",
            "bytes": html_path.stat().st_size,
            "sha256": sha256_file(html_path),
        }
    )
    for index, name in enumerate(gate.DELIVERABLE_NAMES, 1):
        path = run_dir / name
        path.write_bytes(f"fixture-{index}-{name}".encode())
        digest = sha256_file(path)
        record = {
            "name": name,
            "path": name,
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
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


def test_builds_exact_92_second_same_run_narration(tmp_path: Path) -> None:
    contract, _run_dir, run_id = build_ready_fixture(tmp_path)

    result = narration.build_manifest(tmp_path, contract)

    assert result["run_id"] == run_id
    assert result["duration_seconds"] == 92.0
    assert result["same_run_evidence"] is True
    assert result["voice_count"] == 1
    assert result["background_music"] is False
    assert result["source_values"]["files"] == 33_129
    assert result["source_values"]["formal_seeds"] == 3
    assert result["source_values"]["outer_folds"] == 15
    assert result["source_values"]["selected_profile"] == "稳健组合预处理"
    assert result["source_values"]["historical_medal_reference"] == "仍低于历史铜牌线"
    assert "第一轮" in result["segments"][3]["text"]
    assert "第二轮" in result["segments"][3]["text"]
    assert "第三轮" in result["segments"][4]["text"]
    assert "第四轮" in result["segments"][5]["text"]
    assert len(result["segments"]) == 9
    assert result["segments"][0]["start"] == 0.0
    assert result["segments"][-1]["end"] == 92.0
    assert all(
        current["end"] == following["start"]
        for current, following in zip(
            result["segments"][:-1], result["segments"][1:], strict=True
        )
    )


def test_missing_real_evidence_blocks_narration(tmp_path: Path) -> None:
    contract, run_dir, _run_id = build_ready_fixture(tmp_path)
    (run_dir / "hpc_telemetry.jsonl").unlink()

    with pytest.raises(gate.PreRecordingGateError, match="missing narration evidence"):
        narration.build_manifest(tmp_path, contract)


def test_missing_training_progress_is_not_replaced_with_fabricated_defaults(
    tmp_path: Path,
) -> None:
    contract, run_dir, run_id = build_ready_fixture(tmp_path)
    write_json(run_dir / "hpc_runtime.json", {"run_id": run_id, "training_progress": {}})

    with pytest.raises(gate.PreRecordingGateError, match="missing narration count"):
        narration.build_manifest(tmp_path, contract)


def test_unknown_preprocessing_profile_blocks_narration(tmp_path: Path) -> None:
    contract, run_dir, run_id = build_ready_fixture(tmp_path)
    write_json(
        run_dir / "experiment_comparison.json",
        {"run_id": run_id, "selected_profile": "unverified_profile"},
    )

    with pytest.raises(gate.PreRecordingGateError, match="profile is unknown"):
        narration.build_manifest(tmp_path, contract)


def test_missing_historical_medal_thresholds_blocks_claim(tmp_path: Path) -> None:
    contract, run_dir, _run_id = build_ready_fixture(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.pop("historical_thresholds")
    write_json(run_dir / "metrics.json", metrics)

    with pytest.raises(gate.PreRecordingGateError, match="bronze_threshold"):
        narration.build_manifest(tmp_path, contract)
