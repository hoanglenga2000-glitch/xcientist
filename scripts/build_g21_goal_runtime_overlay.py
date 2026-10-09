from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile


FIXED_RUN_ID = "run_7b1efb878afb40f396db431e91f093a5"
FIXED_ALLOCATION = "G21"
OVERLAY_SCHEMA = "evomind.g21_goal_runtime_overlay.v1"
MANIFEST_NAME = "goal-runtime-overlay-manifest.json"
REQUIRED_FILES = (
    "evomind_runtime/__init__.py",
    "evomind_runtime/assistant_runs.py",
    "evomind_runtime/competition_goal.py",
    "evomind_runtime/goal_board.py",
    "evomind_runtime/http_server.py",
    "evomind_runtime/runtime.py",
    "evomind_runtime/store.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def safe_zip_names(path: Path) -> tuple[list[str], list[str]]:
    unsafe: list[str] = []
    duplicates: list[str] = []
    seen: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name = info.filename.replace("\\", "/")
            if name in seen:
                duplicates.append(name)
            seen.add(name)
            parts = Path(name).parts
            if name.startswith("/") or Path(name).drive or ".." in parts:
                unsafe.append(name)
            mode = (info.external_attr >> 16) & 0xF000
            if mode == 0xA000:
                unsafe.append(name)
    return sorted(set(unsafe)), sorted(set(duplicates))


def regular_files(root: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        result[relative] = {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def smoke_candidate(root: Path, source_root: Path, round_number: int) -> dict[str, object]:
    smoke = root / f"goal-overlay-smoke-{round_number}.py"
    smoke.write_text(
        """
import json
import pathlib
import sys
import tempfile

from evomind_runtime.competition_goal import load_goal_spec
from evomind_runtime.goal_board import ensure_fixed_goal
from evomind_runtime.models import Session, utc_now
from evomind_runtime.store import RuntimeStore

source_root = pathlib.Path(sys.argv[1])
spec = load_goal_spec(str(source_root / "configs" / "g21_five_competition_goal.json"))
board = {
    "schema": "evomind.goal-board.v2",
    "run_id": "run_7b1efb878afb40f396db431e91f093a5",
    "allocation": "G21",
    "target_count": 5,
    "weather_actions": 0,
    "competitions": [
        {"competition": name, "goal_status": "WAITING_EXACT_GATE", "exact_gate": "pending"}
        for name in ("cure_bench", "e2lmc", "mindgames", "open_polymer", "ariel_2025")
    ],
}
root = pathlib.Path(tempfile.mkdtemp(prefix="goal-overlay-runtime-smoke-"))
store = RuntimeStore(root / "runtime.sqlite3")
now = utc_now()
store.create_session(Session(id=board["run_id"], workspace_root=str(root), created_at=now, updated_at=now))
store.create_assistant_run({
    "id": board["run_id"], "session_id": board["run_id"], "conversation_id": "goal-overlay-smoke",
    "prompt": "fixed goal smoke", "task_root": str(root), "status": "recovering", "plan": {},
    "attachment_ids": [], "retry_count": 0, "error_class": "", "error_message": "",
    "model_provider": "", "model": "", "created_at": now, "updated_at": now, "completed_at": "",
})
first = ensure_fixed_goal(store, run_id=board["run_id"], spec=spec, board=board)
second = ensure_fixed_goal(store, run_id=board["run_id"], spec=spec, board=board)
assert first["created"] is True and second["created"] is False
assert store.get_goal_for_run(board["run_id"])["id"] == "goal_g21_five_competition"
assert [item["event_type"] for item in store.list_events(board["run_id"]) if item["event_type"] == "goal.created"] == ["goal.created"]
store.close()
print(json.dumps({"status": "passed", "created": True, "replayed": True, "goal_id": "goal_g21_five_competition"}))
""",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [os.environ.get("PYTHON", "python"), str(smoke), str(source_root)],
        cwd=str(root),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    smoke.unlink(missing_ok=True)
    if completed.returncode != 0:
        raise RuntimeError(f"GOAL_OVERLAY_SMOKE_FAILED:{round_number}")
    return {"round": round_number, "status": "passed", "stdout": completed.stdout.strip()}


def build(args: argparse.Namespace) -> dict[str, object]:
    source_root = Path(args.source_root).resolve(strict=True)
    baseline_zip = Path(args.baseline_zip).resolve(strict=True)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    zip_path = output_root / args.zip_name
    manifest_path = output_root / args.manifest_name
    result_path = output_root / args.result_name
    if any(path.exists() for path in (zip_path, manifest_path, result_path)):
        raise RuntimeError("GOAL_OVERLAY_OUTPUT_EXISTS")
    baseline_sha = sha256_file(baseline_zip)
    expected_baseline_sha = args.expected_baseline_sha256.lower()
    if baseline_sha != expected_baseline_sha:
        raise RuntimeError("GOAL_OVERLAY_BASELINE_SHA_MISMATCH")
    baseline_unsafe, baseline_duplicates = safe_zip_names(baseline_zip)
    if baseline_unsafe or baseline_duplicates:
        raise RuntimeError("GOAL_OVERLAY_BASELINE_ZIP_SAFETY_REJECTED")
    source_files: dict[str, dict[str, object]] = {}
    for relative in REQUIRED_FILES:
        path = source_root / "src" / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"GOAL_OVERLAY_SOURCE_MISSING:{relative}")
        source_files[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    spec_path = source_root / "configs" / "g21_five_competition_goal.json"
    evidence_path = source_root / "configs" / "g21_five_competition_human_baseline_gate.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    evidence_sha = sha256_file(evidence_path)
    if str(spec.get("human_baseline_evidence_sha256") or "").lower() != evidence_sha:
        raise RuntimeError("GOAL_OVERLAY_HUMAN_BASELINE_SHA_MISMATCH")
    with tempfile.TemporaryDirectory(prefix="g21-goal-overlay-build-") as temp:
        temp_root = Path(temp)
        package_root = temp_root / "package"
        package_root.mkdir()
        with zipfile.ZipFile(baseline_zip) as archive:
            archive.extractall(package_root)
        baseline_files = regular_files(package_root)
        baseline_py = {key: value for key, value in baseline_files.items() if key.endswith(".py")}
        if len(baseline_py) != 30:
            raise RuntimeError("GOAL_OVERLAY_BASELINE_RUNTIME_COUNT_MISMATCH")
        (package_root / "runtime-hotfix-manifest.json").unlink(missing_ok=True)
        changed: list[dict[str, object]] = []
        for relative, source_info in source_files.items():
            target = package_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            baseline_info = baseline_files.get(relative)
            shutil.copy2(source_root / "src" / relative, target)
            candidate_info = {
                "path": relative,
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
            if baseline_info is None or baseline_info != candidate_info:
                changed.append({
                    "path": relative,
                    "baseline": baseline_info,
                    "candidate": candidate_info,
                })
        required_changed = {item["path"] for item in changed}
        if not {"evomind_runtime/competition_goal.py", "evomind_runtime/goal_board.py"} <= required_changed:
            raise RuntimeError("GOAL_OVERLAY_NEW_MODULES_NOT_PRESENT")
        candidate_files = [
            value
            for key, value in sorted(regular_files(package_root).items())
            if key != MANIFEST_NAME
        ]
        manifest = {
            "schema": OVERLAY_SCHEMA,
            "run_id": FIXED_RUN_ID,
            "allocation": FIXED_ALLOCATION,
            "baseline_zip_sha256": baseline_sha,
            "baseline_runtime_file_count": len(baseline_py),
            "candidate_runtime_file_count": len([p for p in regular_files(package_root) if p.endswith(".py")]),
            "target": "bundle/runtime/evomind_runtime",
            "stale_baseline_manifests_removed": ["runtime-hotfix-manifest.json"],
            "changed_files": changed,
            "changed_file_count": len(changed),
            "file_count": len(candidate_files),
            "files": candidate_files,
            "goal_spec_sha256": sha256_file(spec_path),
            "human_baseline_evidence_sha256": evidence_sha,
            "production_deployed": False,
            "hpc_accessed": False,
            "gpu_touched": False,
            "remote_writes": 0,
            "production_deployable": False,
            "requires_frozen_patch_review": True,
            "unreviewed_dependent_drift": [
                item["path"]
                for item in changed
                if item["baseline"] is not None
                and item["path"] not in {"evomind_runtime/competition_goal.py", "evomind_runtime/goal_board.py"}
            ],
        }
        write_json(package_root / MANIFEST_NAME, manifest)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(package_root.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    archive.write(path, path.relative_to(package_root).as_posix())
        unsafe, duplicates = safe_zip_names(zip_path)
        if unsafe or duplicates:
            raise RuntimeError("GOAL_OVERLAY_ZIP_SAFETY_REJECTED")
        verification_rounds: list[dict[str, object]] = []
        for round_number in (1, 2):
            verify_root = temp_root / f"verify-{round_number}"
            verify_root.mkdir()
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(verify_root)
            actual = regular_files(verify_root)
            expected = {key: value for key, value in actual.items()}
            if MANIFEST_NAME not in expected:
                raise RuntimeError("GOAL_OVERLAY_MANIFEST_MISSING")
            if "runtime-hotfix-manifest.json" in expected:
                raise RuntimeError("GOAL_OVERLAY_STALE_BASELINE_MANIFEST_PRESENT")
            py_files = [path for path in actual if path.endswith(".py")]
            if len(py_files) != len(baseline_py) + 2:
                raise RuntimeError("GOAL_OVERLAY_CANDIDATE_RUNTIME_COUNT_MISMATCH")
            for relative in py_files:
                completed = subprocess.run(
                    [os.environ.get("PYTHON", "python"), "-m", "py_compile", str(verify_root / relative)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if completed.returncode != 0:
                    raise RuntimeError(f"GOAL_OVERLAY_PYCOMPILE_FAILED:{round_number}:{relative}")
            verification_rounds.append(smoke_candidate(verify_root, source_root, round_number))
        result = {
            "schema": OVERLAY_SCHEMA,
            "status": "built_and_verified",
            "zip_path": str(zip_path),
            "zip_bytes": zip_path.stat().st_size,
            "zip_sha256": sha256_file(zip_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(package_root / MANIFEST_NAME),
            "baseline_zip_sha256": baseline_sha,
            "changed_file_count": len(changed),
            "candidate_runtime_file_count": len(baseline_py) + 2,
            "verification_rounds": verification_rounds,
            "production_deployed": False,
            "hpc_accessed": False,
            "gpu_touched": False,
            "remote_writes": 0,
            "production_deployable": False,
            "requires_frozen_patch_review": True,
        }
        write_json(manifest_path, manifest)
        write_json(result_path, result)
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--baseline-zip", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-baseline-sha256", required=True)
    parser.add_argument("--zip-name", default="g21-goal-runtime-overlay.zip")
    parser.add_argument("--manifest-name", default="g21-goal-runtime-overlay-source-manifest.json")
    parser.add_argument("--result-name", default="g21-goal-runtime-overlay-build-result.json")
    args = parser.parse_args()
    result = build(args)
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
