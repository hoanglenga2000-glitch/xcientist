from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import refresh_siim_multiround_delivery as refresh

RUN_ID = "evomind_siim_refresh_fixture"
STABLE_TIME = "2026-07-30T12:00:00+00:00"


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def passed_claim() -> dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "status": "passed",
        "checks": {name: True for name in refresh.CLAIM_CHECKS},
    }


def artifact_bytes(root: Path, relative: str) -> bytes:
    return (root / relative).read_bytes()


def build_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, bytes]]:
    project = tmp_path / "project"
    run_dir = project / "workspace" / "evomind_runs" / RUN_ID
    delivery = project / "workspace" / "siim_fixture" / "deliveries" / RUN_ID
    run_dir.mkdir(parents=True)
    delivery.mkdir(parents=True)

    write_json(
        run_dir / "run.json",
        {"run_id": RUN_ID, "status": "completed"},
    )
    write_json(
        run_dir / "task_graph.json",
        {
            "run_id": RUN_ID,
            "nodes": [
                {"task_id": task_id, "status": "completed"}
                for task_id in refresh.REQUIRED_TASKS
            ],
        },
    )
    write_json(
        run_dir / "training_result.json",
        {"run_id": RUN_ID, "status": "completed", "formal_seeds": [43, 44, 45]},
    )
    write_json(run_dir / "review.json", {"run_id": RUN_ID, "status": "passed"})
    freeze = write_json(
        run_dir / "candidate_freeze.json",
        {
            "run_id": RUN_ID,
            "status": "frozen_before_private_grader",
            "tuning_closed": True,
            "private_grader_execution_count_before_freeze": 0,
        },
    )
    grader = write_json(
        run_dir / "private_grader.json",
        {
            "run_id": RUN_ID,
            "status": "passed",
            "execution_index": 1,
            "candidate_freeze_sha256": refresh.sha256_file(freeze),
            "feedback_used_for_tuning": False,
            "official_submission_executed": False,
        },
    )
    write_json(
        run_dir / "private_grader_ledger.json",
        {
            "run_id": RUN_ID,
            "execution_count": 1,
            "candidate_freeze_sha256": refresh.sha256_file(freeze),
            "result_sha256": refresh.sha256_file(grader),
        },
    )
    write_json(run_dir / "claim_audit_source.json", passed_claim())
    write_json(run_dir / "claim_audit.json", passed_claim())
    write_json(run_dir / "metrics.json", {"run_id": RUN_ID, "roc_auc": 0.91})

    for name in refresh.DOWNLOAD_NAMES:
        (delivery / name).write_bytes(f"old-delivery:{name}".encode())
    (delivery / "research_report.html").write_text("<html>old eight page report</html>", encoding="utf-8")
    write_json(delivery / "artifact_manifest.json", {"run_id": RUN_ID, "edition": 8})
    (delivery / "qa").mkdir()
    (delivery / "qa" / "page-01.png").write_bytes(b"old-page")

    for name in refresh.DOWNLOAD_NAMES:
        (run_dir / name).write_bytes(f"old-run:{name}".encode())
    (run_dir / "research_report.html").write_text("old root html", encoding="utf-8")
    original_ingress = run_dir / "ingress" / "campaign"
    original_ingress.mkdir(parents=True)
    (original_ingress / "sealed.json").write_bytes(b"immutable-ingress-seal")
    old_versioned = run_dir / "ingress" / refresh.VERSIONED_INGRESS
    old_versioned.mkdir(parents=True)
    (old_versioned / "research_report.html").write_text("old versioned html", encoding="utf-8")
    for name in refresh.DOWNLOAD_NAMES:
        (old_versioned / name).write_bytes(f"old-versioned:{name}".encode())
    write_json(run_dir / "deliverables.json", {"run_id": RUN_ID, "status": "ready", "old": True})
    write_json(run_dir / "delivery_multiround_refresh.json", {"run_id": RUN_ID, "old": True})
    write_json(run_dir / "artifact_manifest.json", {"run_id": RUN_ID, "status": "verified", "old": True})

    old = {
        relative: artifact_bytes(run_dir, relative)
        for relative in refresh.RUN_DERIVED_FILES
        if (run_dir / relative).is_file()
    }
    return project, run_dir, delivery, old


def fake_validated(run_dir: Path) -> SimpleNamespace:
    evidence_names = (
        "candidate_freeze.json",
        "private_grader.json",
        "private_grader_ledger.json",
        "metrics.json",
        "training_result.json",
        "review.json",
        "claim_audit_source.json",
    )
    paths = {name: run_dir / name for name in evidence_names}
    return SimpleNamespace(
        run_id=RUN_ID,
        stable_time=STABLE_TIME,
        evidence_paths=paths,
        source_hashes={name: refresh.sha256_file(path) for name, path in paths.items()},
    )


def fake_builder(run_dir: Path, output_dir: Path, *, repo_root: Path) -> dict[str, Any]:
    assert repo_root in run_dir.parents
    output_dir.mkdir(parents=True)
    for name in refresh.DOWNLOAD_NAMES:
        (output_dir / name).write_bytes(f"new-nine-page:{name}".encode())
    (output_dir / "research_report.html").write_text(
        "<html>EvoMind 四轮进化 R1 R2 R3 R4</html>", encoding="utf-8"
    )
    write_json(
        output_dir / "artifact_manifest.json",
        {
            "schema": "evomind.siim.delivery_bundle_manifest.v1",
            "run_id": RUN_ID,
            "status": "verified",
            "evidence_time": STABLE_TIME,
        },
    )
    (output_dir / "qa").mkdir()
    write_json(output_dir / "qa" / "delivery-qa.json", {"run_id": RUN_ID, "page_count": 9})
    return {"run_id": RUN_ID, "status": "verified", "output_dir": str(output_dir)}


def passing_gate(project_root: Path, run_id: str) -> dict[str, Any]:
    run_dir = project_root / "workspace" / "evomind_runs" / run_id
    manifest = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    records = {item["path"]: item for item in manifest["artifacts"]}
    delivery = json.loads((run_dir / "deliverables.json").read_text(encoding="utf-8"))
    assert delivery["status"] == "ready"
    for name in refresh.DOWNLOAD_NAMES:
        path = run_dir / name
        assert records[name]["sha256"] == refresh.sha256_file(path)
        assert records[name]["bytes"] == path.stat().st_size
        listed = next(item for item in delivery["files"] if item["name"] == name)
        assert listed["sha256"] == refresh.sha256_file(path)
    gate_html = run_dir / "research_report.html"
    assert records["research_report.html"]["sha256"] == refresh.sha256_file(gate_html)
    assert "R4" in gate_html.read_text(encoding="utf-8")
    versioned = run_dir / "ingress" / refresh.VERSIONED_INGRESS / "research_report.html"
    assert records[f"ingress/{refresh.VERSIONED_INGRESS}/research_report.html"]["sha256"] == refresh.sha256_file(versioned)
    assert versioned.read_bytes() == gate_html.read_bytes()
    return {
        "run_id": run_id,
        "status": "passed",
        "recording_allowed": True,
        "multiround_evolution_verified": True,
    }


def run_refresh(project: Path, run_dir: Path, delivery: Path, **kwargs: Any) -> dict[str, Any]:
    return refresh.refresh_delivery(
        project,
        RUN_ID,
        delivery_dir=delivery,
        backup_root=project / "backups" / RUN_ID,
        validate_run_fn=lambda path: fake_validated(path),
        build_fn=fake_builder,
        validate_fn=lambda _paths: True,
        staged_report_fn=lambda _path: None,
        gate_fn=passing_gate,
        **kwargs,
    )


def test_refresh_is_transactional_and_preserves_complete_old_derivations(tmp_path: Path) -> None:
    project, run_dir, delivery, old = build_fixture(tmp_path)
    protected_before = {
        name: refresh.sha256_file(run_dir / name)
        for name in ("metrics.json", "candidate_freeze.json", "private_grader.json", "private_grader_ledger.json")
    }

    result = run_refresh(project, run_dir, delivery)

    assert result["status"] == "verified"
    assert result["idempotent_reuse"] is False
    assert result["private_grader_execution_count"] == 1
    backup = Path(result["backup_dir"])
    assert backup.is_dir()
    for relative, content in old.items():
        assert (backup / "run_root" / relative).read_bytes() == content
    assert (backup / "delivery" / "qa" / "page-01.png").read_bytes() == b"old-page"
    assert "R4" in (run_dir / "research_report.html").read_text(encoding="utf-8")
    assert (run_dir / "research_report.html").read_bytes() == (
        run_dir / "ingress" / refresh.VERSIONED_INGRESS / "research_report.html"
    ).read_bytes()
    assert (run_dir / "ingress" / "campaign" / "sealed.json").read_bytes() == b"immutable-ingress-seal"
    assert all(
        (run_dir / name).read_bytes() == (delivery / name).read_bytes()
        for name in refresh.DOWNLOAD_NAMES
    )
    assert protected_before == {
        name: refresh.sha256_file(run_dir / name) for name in protected_before
    }


def test_same_content_is_reused_without_creating_another_backup(tmp_path: Path) -> None:
    project, run_dir, delivery, _old = build_fixture(tmp_path)
    first = run_refresh(project, run_dir, delivery)
    backups = project / "backups" / RUN_ID
    before = sorted(path.name for path in backups.iterdir() if path.is_dir())

    second = run_refresh(project, run_dir, delivery)

    assert first["idempotent_reuse"] is False
    assert second["idempotent_reuse"] is True
    assert second["backup_dir"] is None
    assert sorted(path.name for path in backups.iterdir() if path.is_dir()) == before


def test_failed_post_swap_gate_restores_every_old_artifact(tmp_path: Path) -> None:
    project, run_dir, delivery, old = build_fixture(tmp_path)
    old_delivery = refresh._inventory(delivery)

    with pytest.raises(refresh.DeliveryRefreshError, match="rolled back"):
        refresh.refresh_delivery(
            project,
            RUN_ID,
            delivery_dir=delivery,
            backup_root=project / "backups" / RUN_ID,
            validate_run_fn=lambda path: fake_validated(path),
            build_fn=fake_builder,
            validate_fn=lambda _paths: True,
            staged_report_fn=lambda _path: None,
            gate_fn=lambda _root, _run_id: (_ for _ in ()).throw(RuntimeError("gate injected failure")),
        )

    assert refresh._inventory(delivery) == old_delivery
    for relative, content in old.items():
        assert artifact_bytes(run_dir, relative) == content


def test_mid_swap_failure_rolls_back_partial_replacements(tmp_path: Path, monkeypatch) -> None:
    project, run_dir, delivery, old = build_fixture(tmp_path)
    old_delivery = refresh._inventory(delivery)
    backup_root = project / "backups" / RUN_ID
    backup_root.mkdir(parents=True)
    backup, _reused = refresh._copy_backup(run_dir, delivery, backup_root)
    staged_delivery = delivery.parent / ".staged"
    fake_builder(run_dir, staged_delivery, repo_root=project)
    staged_run = run_dir.parent / ".staged-run"
    staged_run.mkdir()
    for relative in refresh.RUN_DERIVED_FILES:
        source = staged_delivery / Path(relative).name
        if relative == "deliverables.json":
            source = write_json(staged_run / relative, {"run_id": RUN_ID, "status": "ready"})
        elif relative == "delivery_multiround_refresh.json":
            source = write_json(staged_run / relative, {"run_id": RUN_ID, "status": "verified"})
        elif relative == "artifact_manifest.json":
            source = write_json(staged_run / relative, {"run_id": RUN_ID, "status": "verified"})
        else:
            target = staged_run / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())

    real_replace = refresh.os.replace
    injected = False

    def failing_replace(source: str | Path, target: str | Path) -> None:
        nonlocal injected
        source_path = Path(source)
        target_path = Path(target)
        if (
            not injected
            and source_path.name.startswith(f".{target_path.name}.")
            and target_path.name == refresh.DOWNLOAD_NAMES[2]
        ):
            injected = True
            raise OSError("injected mid-swap failure")
        real_replace(source, target)

    monkeypatch.setattr(refresh.os, "replace", failing_replace)

    with pytest.raises(OSError, match="mid-swap"):
        refresh._swap_transaction(run_dir, delivery, staged_delivery, staged_run, backup)

    assert injected is True
    assert refresh._inventory(delivery) == old_delivery
    for relative, content in old.items():
        assert artifact_bytes(run_dir, relative) == content


def test_incomplete_three_seed_training_blocks_before_builder(tmp_path: Path) -> None:
    project, run_dir, delivery, _old = build_fixture(tmp_path)
    training = json.loads((run_dir / "training_result.json").read_text(encoding="utf-8"))
    training["formal_seeds"] = [43, 44]
    write_json(run_dir / "training_result.json", training)
    called = False

    def forbidden_builder(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        raise AssertionError("builder must not run")

    with pytest.raises(refresh.DeliveryRefreshError, match="three-seed training"):
        refresh.refresh_delivery(
            project,
            RUN_ID,
            delivery_dir=delivery,
            backup_root=project / "backups" / RUN_ID,
            validate_run_fn=lambda path: fake_validated(path),
            build_fn=forbidden_builder,
            validate_fn=lambda _paths: True,
            staged_report_fn=lambda _path: None,
            gate_fn=passing_gate,
        )
    assert called is False


def test_builder_source_mutation_is_detected_before_any_swap(tmp_path: Path) -> None:
    project, run_dir, delivery, old = build_fixture(tmp_path)
    old_delivery = refresh._inventory(delivery)

    def mutating_builder(run: Path, output: Path, *, repo_root: Path) -> dict[str, Any]:
        result = fake_builder(run, output, repo_root=repo_root)
        (run / "metrics.json").write_bytes(b"mutated")
        return result

    with pytest.raises(refresh.DeliveryRefreshError, match="non-derived evidence changed"):
        refresh.refresh_delivery(
            project,
            RUN_ID,
            delivery_dir=delivery,
            backup_root=project / "backups" / RUN_ID,
            validate_run_fn=lambda path: fake_validated(path),
            build_fn=mutating_builder,
            validate_fn=lambda _paths: True,
            staged_report_fn=lambda _path: None,
            gate_fn=passing_gate,
        )

    assert refresh._inventory(delivery) == old_delivery
    for relative, content in old.items():
        assert artifact_bytes(run_dir, relative) == content
