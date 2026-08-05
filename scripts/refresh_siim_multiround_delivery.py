#!/usr/bin/env python3
"""Transactionally refresh a completed SIIM Run to the nine-page R1-R4 delivery.

This script is deliberately downstream-only.  It never trains, grades, submits,
or changes candidate evidence.  The existing delivery ingress remains sealed;
the refreshed files are copied into a versioned ingress namespace and atomically
published to the user-facing Run root only after all source hashes stay stable.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import fitz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for _entry in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from research_os.agent.siim_hpc_workflow import (  # noqa: E402
    FORMAL_SEEDS,
    MUTABLE_FILES,
)
from scripts import build_siim_delivery as delivery_builder  # noqa: E402
from scripts import build_siim_workflow_ingress as ingress  # noqa: E402
from scripts import manage_siim_job89508_campaign as campaign  # noqa: E402
from scripts import supervise_siim_job89508_end_to_end as supervisor  # noqa: E402

DOWNLOAD_NAMES = tuple(delivery_builder.DOWNLOAD_NAMES)
HTML_NAME = "research_report.html"
VERSIONED_INGRESS = "deliverables_multiround_v2"
REFRESH_SCHEMA = "evomind.siim.multiround_delivery_refresh.v2"
EXPECTED_REPORT_PAGES = 9
EVOLUTION_MARKERS = ("EvoMind 四轮进化", "R1", "R2", "R3", "R4")
REQUIRED_TASKS = tuple(supervisor.REQUIRED_FINAL_TASKS)
CLAIM_CHECKS = (
    "no_public_leaderboard_claim",
    "no_official_medal_claim",
    "no_clinical_diagnosis_claim",
    "private_grader_not_used_for_tuning",
    "candidate_hashes_unchanged",
    "official_submission_not_executed",
)
RUN_DERIVED_FILES = (
    HTML_NAME,
    *DOWNLOAD_NAMES,
    "deliverables.json",
    "delivery_multiround_refresh.json",
    "artifact_manifest.json",
    f"ingress/{VERSIONED_INGRESS}/{HTML_NAME}",
    *(f"ingress/{VERSIONED_INGRESS}/{name}" for name in DOWNLOAD_NAMES),
)


class DeliveryRefreshError(RuntimeError):
    """The derived delivery could not be refreshed without changing evidence."""


BuildFn = Callable[..., Mapping[str, Any]]
ValidateFn = Callable[[supervisor.SupervisorPaths], bool]
GateFn = Callable[[Path, str], Mapping[str, Any]]
RunValidateFn = Callable[[Path], Any]
ReportFn = Callable[[Path], None]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DeliveryRefreshError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing or unsafe {label}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DeliveryRefreshError(f"invalid {label}") from exc
    require(isinstance(payload, dict), f"{label} is not a JSON object")
    return payload


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_copy(source: Path, destination: Path, *, compact_temporary_name: bool = False) -> None:
    require(source.is_file() and not source.is_symlink(), f"unsafe refresh source: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if compact_temporary_name:
        # Backup trees can be deeply nested on Windows test hosts; appending
        # the full destination basename to a dotfile can push otherwise-valid
        # paths over MAX_PATH and surface as FileNotFoundError.
        temporary = destination.parent / f".t{os.getpid()}"
    else:
        temporary = destination.parent / f".{destination.name}.{os.getpid()}.tmp"
    shutil.copy2(source, temporary)
    require(sha256_file(temporary) == sha256_file(source), f"refresh copy changed: {source.name}")
    os.replace(temporary, destination)


def _inventory(root: Path) -> dict[str, tuple[int, str]]:
    require(root.is_dir() and not root.is_symlink(), f"unsafe directory: {root}")
    records: dict[str, tuple[int, str]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise DeliveryRefreshError(f"symlink is forbidden in derived delivery: {path}")
        if path.is_file():
            records[path.relative_to(root).as_posix()] = (path.stat().st_size, sha256_file(path))
    return records


def _inventory_without_versioned_ingress(run_dir: Path) -> dict[str, tuple[int, str]]:
    root = run_dir / "ingress"
    if not root.exists():
        return {}
    require(root.is_dir() and not root.is_symlink(), "Run ingress is unsafe")
    excluded = root / VERSIONED_INGRESS
    records: dict[str, tuple[int, str]] = {}
    for path in sorted(root.rglob("*")):
        if excluded == path or excluded in path.parents:
            continue
        if path.is_symlink():
            raise DeliveryRefreshError(f"Run ingress contains a symlink: {path}")
        if path.is_file():
            records[path.relative_to(root).as_posix()] = (path.stat().st_size, sha256_file(path))
    return records


def _artifact_record(path: Path, run_dir: Path, *, kind: str) -> dict[str, Any]:
    return {
        "path": path.relative_to(run_dir).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "kind": kind,
    }


def _hash_evidence(paths: Mapping[str, Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, path in sorted(paths.items()):
        selected = Path(path)
        require(selected.is_file() and not selected.is_symlink(), f"protected evidence is unsafe: {name}")
        hashes[name] = sha256_file(selected)
    return hashes


def _validate_prerequisites(run_dir: Path, run_id: str, validated: Any) -> None:
    run = read_json(run_dir / "run.json", "run.json")
    graph = read_json(run_dir / "task_graph.json", "task graph")
    training = read_json(run_dir / "training_result.json", "training result")
    review = read_json(run_dir / "review.json", "Independent Review")
    claim = read_json(run_dir / "claim_audit.json", "Claim Audit")
    freeze = read_json(run_dir / "candidate_freeze.json", "candidate freeze")
    grader = read_json(run_dir / "private_grader.json", "private grader result")
    ledger = read_json(run_dir / "private_grader_ledger.json", "private grader ledger")
    for label, payload in (
        ("run", run),
        ("task graph", graph),
        ("training", training),
        ("review", review),
        ("claim", claim),
        ("freeze", freeze),
        ("grader", grader),
        ("ledger", ledger),
    ):
        require(payload.get("run_id") == run_id, f"{label} belongs to another Run")
    require(getattr(validated, "run_id", run_id) == run_id, "validated evidence belongs to another Run")
    require(run.get("status") == "completed", "Run is not completed")
    nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
    statuses = {
        str(node.get("task_id") or ""): node.get("status")
        for node in nodes
        if isinstance(node, dict)
    }
    require(
        all(statuses.get(task_id) == "completed" for task_id in REQUIRED_TASKS),
        "nine-node workflow is incomplete",
    )
    require(
        tuple(int(seed) for seed in training.get("formal_seeds") or ()) == tuple(FORMAL_SEEDS),
        "three-seed training contract is incomplete",
    )
    require(review.get("status") == "passed", "Independent Review is not passed")
    require(claim.get("status") == "passed", "Claim Audit is not passed")
    checks = claim.get("checks") if isinstance(claim.get("checks"), dict) else {}
    require(all(checks.get(name) is True for name in CLAIM_CHECKS), "Claim Audit checks changed")
    require(freeze.get("status") == "frozen_before_private_grader", "candidate is not frozen")
    require(
        int(freeze.get("private_grader_execution_count_before_freeze") or 0) == 0,
        "private grader ran before freeze",
    )
    require(int(ledger.get("execution_count") or 0) == 1, "private grader count is not exactly one")
    require(grader.get("feedback_used_for_tuning") is False, "private grader feedback boundary changed")
    require(grader.get("official_submission_executed") is False, "official submission boundary changed")


def _validate_staged_report(delivery_dir: Path) -> None:
    pdf_path = delivery_dir / DOWNLOAD_NAMES[0]
    html_path = delivery_dir / HTML_NAME
    require(pdf_path.is_file() and html_path.is_file(), "staged professional report is missing")
    try:
        with fitz.open(pdf_path) as document:
            require(document.page_count == EXPECTED_REPORT_PAGES, "staged PDF is not nine pages")
            pdf_text = "\n".join(page.get_text("text") for page in document)
        html_text = html_path.read_text(encoding="utf-8-sig")
    except (OSError, RuntimeError, UnicodeError, ValueError) as exc:
        raise DeliveryRefreshError("staged professional report is unreadable") from exc
    for marker in EVOLUTION_MARKERS:
        require(marker in pdf_text, f"staged PDF omits evolution marker: {marker}")
        require(marker in html_text, f"staged HTML omits evolution marker: {marker}")


def _overlay_manifest_entries(run_dir: Path, staged_run: Path) -> list[dict[str, Any]]:
    overlay = {Path(name).as_posix() for name in RUN_DERIVED_FILES}
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_symlink():
            raise DeliveryRefreshError(f"Run contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir).as_posix()
        if relative in overlay or path.name in MUTABLE_FILES or path.name == "artifact_manifest.json":
            continue
        if ".tmp" in path.name or "__pycache__" in path.parts or "results" in path.parts:
            continue
        records[relative] = _artifact_record(path, run_dir, kind=path.name)
    for path in sorted(staged_run.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        relative = path.relative_to(staged_run).as_posix()
        records[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "kind": path.name,
        }
    return [records[name] for name in sorted(records)]


def _stage_run_derivatives(
    run_dir: Path,
    staged_delivery: Path,
    staged_run: Path,
    *,
    run_id: str,
    stable_time: str,
    source_hashes: Mapping[str, str],
    original_ingress_inventory: Mapping[str, tuple[int, str]],
) -> None:
    for name in (HTML_NAME, *DOWNLOAD_NAMES):
        atomic_copy(staged_delivery / name, staged_run / name)
        atomic_copy(
            staged_delivery / name,
            staged_run / "ingress" / VERSIONED_INGRESS / name,
        )
    files = []
    for name in DOWNLOAD_NAMES:
        path = staged_run / name
        files.append(
            {
                **_artifact_record(path, staged_run, kind="user_download"),
                "name": name,
                "download_url": f"/api/multi-agent/runs/{run_id}/download/{name}",
            }
        )
    atomic_json(
        staged_run / "deliverables.json",
        {
            "schema": "evomind.siim.deliverables.v1",
            "run_id": run_id,
            "status": "ready",
            "files": files,
            "official_submission": "forbidden",
            "clinical_use": "not_claimed",
            "created_at": stable_time,
            "derived_delivery_version": "multiround_v2",
        },
    )
    refresh_record = {
        "schema": REFRESH_SCHEMA,
        "run_id": run_id,
        "status": "verified",
        "version": "multiround_v2",
        "created_at": stable_time,
        "source_delivery_manifest_sha256": sha256_file(staged_delivery / "artifact_manifest.json"),
        "candidate_freeze_sha256": source_hashes.get("candidate_freeze.json", ""),
        "private_grader_sha256": source_hashes.get("private_grader.json", ""),
        "private_grader_execution_count": 1,
        "original_ingress_inventory_sha256": _json_sha256(original_ingress_inventory),
        "versioned_ingress": f"ingress/{VERSIONED_INGRESS}",
        "official_submission_executed": False,
        "signals_sent": 0,
        "other_processes_modified": False,
    }
    atomic_json(staged_run / "delivery_multiround_refresh.json", refresh_record)
    atomic_json(
        staged_run / "artifact_manifest.json",
        {
            "schema": "evomind.siim.artifact_manifest.v1",
            "run_id": run_id,
            "task_id": delivery_builder.TASK_ID,
            "status": "verified",
            "artifacts": _overlay_manifest_entries(run_dir, staged_run),
            "mutable_ledgers_excluded_from_hash_seal": sorted(MUTABLE_FILES),
            "private_grader_execution_count": 1,
            "official_submission": "forbidden",
            "generated_at": stable_time,
            "derived_delivery_version": "multiround_v2",
        },
    )


def _backup_state_path(backup: Path) -> Path:
    return backup / "backup-state.json"


def _copy_backup(
    run_dir: Path,
    delivery_dir: Path,
    backup_root: Path,
) -> tuple[Path, bool]:
    backup_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    backup = backup_root / f"{stamp}-{os.getpid()}"
    suffix = 0
    while backup.exists():
        suffix += 1
        backup = backup_root / f"{stamp}-{os.getpid()}-{suffix}"
    backup.mkdir(parents=True)
    existing: list[str] = []
    for relative in RUN_DERIVED_FILES:
        source = run_dir / relative
        if source.is_file():
            require(not source.is_symlink(), f"unsafe derived Run file: {relative}")
            existing.append(relative)
            atomic_copy(source, backup / "run_root" / relative, compact_temporary_name=True)
    delivery_existed = delivery_dir.exists()
    if delivery_existed:
        require(delivery_dir.is_dir() and not delivery_dir.is_symlink(), "existing delivery is unsafe")
        shutil.copytree(delivery_dir, backup / "delivery", copy_function=shutil.copy2)
    atomic_json(
        _backup_state_path(backup),
        {
            "run_files_existing": existing,
            "delivery_existed": delivery_existed,
        },
    )
    return backup, False


def _restore_backup(run_dir: Path, delivery_dir: Path, backup: Path) -> None:
    state = read_json(_backup_state_path(backup), "delivery refresh backup state")
    existing = set(str(item) for item in state.get("run_files_existing") or [])
    if delivery_dir.exists():
        require(delivery_dir.is_dir() and not delivery_dir.is_symlink(), "rollback delivery is unsafe")
        shutil.rmtree(delivery_dir)
    if state.get("delivery_existed") is True:
        shutil.copytree(backup / "delivery", delivery_dir, copy_function=shutil.copy2)
    for relative in RUN_DERIVED_FILES:
        target = run_dir / relative
        saved = backup / "run_root" / relative
        if relative in existing:
            atomic_copy(saved, target)
        elif target.is_file():
            target.unlink()
    versioned = run_dir / "ingress" / VERSIONED_INGRESS
    if versioned.is_dir() and not any(versioned.iterdir()):
        versioned.rmdir()


def _swap_transaction(
    run_dir: Path,
    delivery_dir: Path,
    staged_delivery: Path,
    staged_run: Path,
    backup: Path,
) -> None:
    for relative in RUN_DERIVED_FILES:
        require((staged_run / relative).is_file(), f"staged Run derivative is missing: {relative}")
    delivery_dir.parent.mkdir(parents=True, exist_ok=True)
    displaced = delivery_dir.parent / f".{delivery_dir.name}.pre-multiround-{os.getpid()}"
    require(not displaced.exists(), "stale delivery swap directory exists")
    try:
        if delivery_dir.exists():
            os.replace(delivery_dir, displaced)
        os.replace(staged_delivery, delivery_dir)
        for relative in RUN_DERIVED_FILES:
            atomic_copy(staged_run / relative, run_dir / relative)
    except BaseException:
        if delivery_dir.exists():
            shutil.rmtree(delivery_dir)
        if displaced.exists():
            os.replace(displaced, delivery_dir)
        _restore_backup(run_dir, delivery_dir, backup)
        raise
    finally:
        if displaced.exists():
            shutil.rmtree(displaced, ignore_errors=True)


def _load_default_gate(project_root: Path, run_id: str) -> Mapping[str, Any]:
    gate_path = (
        project_root
        / "video-production"
        / "siim-isic-melanoma-commercial-v1"
        / "scripts"
        / "verify_pre_recording_gate.py"
    )
    contract_path = (
        project_root
        / "video-production"
        / "siim-isic-melanoma-commercial-v1"
        / "preproduction"
        / "production-contract.json"
    )
    contract = read_json(contract_path, "video production contract")
    require(contract.get("run_id") == run_id, "video production contract belongs to another Run")
    spec = importlib.util.spec_from_file_location("evomind_siim_pre_recording_gate", gate_path)
    require(spec is not None and spec.loader is not None, "pre-recording gate could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.validate_gate(project_root, contract_path)
    require(isinstance(result, dict), "pre-recording gate returned invalid evidence")
    return result


def _all_derivatives_match(
    run_dir: Path,
    delivery_dir: Path,
    staged_delivery: Path,
    staged_run: Path,
) -> bool:
    if not delivery_dir.is_dir() or _inventory(delivery_dir) != _inventory(staged_delivery):
        return False
    for relative in RUN_DERIVED_FILES:
        current = run_dir / relative
        staged = staged_run / relative
        if not current.is_file() or current.is_symlink():
            return False
        if current.stat().st_size != staged.stat().st_size or sha256_file(current) != sha256_file(staged):
            return False
    return True


def default_paths(project_root: Path, run_id: str) -> tuple[Path, Path, Path]:
    root = project_root.resolve()
    selected = ingress.validate_run_id(run_id)
    run_dir = root / "workspace" / "evomind_runs" / selected
    delivery_dir = root / "workspace" / f"siim_{campaign.JOB_TAG}" / "deliveries" / selected
    campaign_dir = root / "workspace" / "hpc" / f"{campaign.JOB_TAG}_siim_campaign" / selected
    return run_dir, delivery_dir, campaign_dir


def refresh_delivery(
    project_root: Path,
    run_id: str,
    *,
    delivery_dir: Path | None = None,
    backup_root: Path | None = None,
    validate_run_fn: RunValidateFn = delivery_builder.validate_run,
    build_fn: BuildFn = delivery_builder.build_delivery,
    validate_fn: ValidateFn = supervisor.validate_delivery_bundle,
    staged_report_fn: ReportFn = _validate_staged_report,
    gate_fn: GateFn | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    run_id = ingress.validate_run_id(run_id)
    run_dir, default_delivery, campaign_dir = default_paths(root, run_id)
    delivery_dir = Path(delivery_dir or default_delivery).resolve()
    backup_root = Path(backup_root or campaign_dir / "delivery_refresh_backups").resolve()
    require(run_dir.is_dir() and not run_dir.is_symlink(), "completed Run directory is unavailable")
    validated = validate_run_fn(run_dir)
    _validate_prerequisites(run_dir, run_id, validated)
    evidence_paths = {
        str(name): Path(path)
        for name, path in dict(getattr(validated, "evidence_paths", {})).items()
    }
    require(bool(evidence_paths), "validated Run exposes no protected evidence")
    evidence_before = _hash_evidence(evidence_paths)
    original_ingress_before = _inventory_without_versioned_ingress(run_dir)
    stable_time = str(getattr(validated, "stable_time", "") or utc_now())
    gate = gate_fn or _load_default_gate

    # Keep the transactional staging path short enough for Windows' legacy
    # CopyFile2 path limit while staying on the project volume for os.replace.
    work_parent = root / ".siim_delivery_refresh_work"
    work_parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="multiround-", dir=work_parent))
    staged_delivery = work / "delivery"
    staged_run = work / "run"
    staged_run.mkdir()
    backup: Path | None = None
    swapped = False
    try:
        build_fn(run_dir, staged_delivery, repo_root=root)
        staged_paths = replace(supervisor.SupervisorPaths.build(root, run_id), delivery_dir=staged_delivery)
        require(validate_fn(staged_paths) is True, "new multiround delivery did not validate")
        staged_report_fn(staged_delivery)
        require(
            _hash_evidence(evidence_paths) == evidence_before,
            "non-derived evidence changed while building the delivery",
        )
        require(
            _inventory_without_versioned_ingress(run_dir) == original_ingress_before,
            "original ingress seal changed while building the delivery",
        )
        _stage_run_derivatives(
            run_dir,
            staged_delivery,
            staged_run,
            run_id=run_id,
            stable_time=stable_time,
            source_hashes=evidence_before,
            original_ingress_inventory=original_ingress_before,
        )
        if _all_derivatives_match(run_dir, delivery_dir, staged_delivery, staged_run):
            gate_result = gate(root, run_id)
            require(gate_result.get("status") == "passed", "pre-recording gate did not pass")
            return {
                "schema": REFRESH_SCHEMA,
                "run_id": run_id,
                "status": "verified",
                "idempotent_reuse": True,
                "backup_dir": None,
                "delivery_dir": str(delivery_dir),
                "private_grader_execution_count": 1,
                "official_submission_executed": False,
                "signals_sent": 0,
                "other_processes_modified": False,
            }
        backup, _ = _copy_backup(run_dir, delivery_dir, backup_root)
        _swap_transaction(run_dir, delivery_dir, staged_delivery, staged_run, backup)
        swapped = True
        require(_hash_evidence(evidence_paths) == evidence_before, "non-derived evidence changed after swap")
        require(
            _inventory_without_versioned_ingress(run_dir) == original_ingress_before,
            "original ingress seal changed after swap",
        )
        gate_result = gate(root, run_id)
        require(gate_result.get("status") == "passed", "pre-recording gate did not pass after refresh")
        return {
            "schema": REFRESH_SCHEMA,
            "run_id": run_id,
            "status": "verified",
            "idempotent_reuse": False,
            "backup_dir": str(backup),
            "delivery_dir": str(delivery_dir),
            "private_grader_execution_count": 1,
            "official_submission_executed": False,
            "signals_sent": 0,
            "other_processes_modified": False,
        }
    except BaseException as exc:
        if swapped and backup is not None:
            try:
                _restore_backup(run_dir, delivery_dir, backup)
            except BaseException as rollback_exc:
                raise DeliveryRefreshError("delivery refresh failed and rollback also failed") from rollback_exc
            raise DeliveryRefreshError("delivery refresh failed and rolled back") from exc
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = refresh_delivery(args.project_root, args.run_id)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
