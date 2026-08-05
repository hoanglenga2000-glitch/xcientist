#!/usr/bin/env python3
"""Safely deploy and collect the frozen May 2022 CPU diagnostic on job 89941.

The tool is deliberately fail-closed.  It validates the complete local frozen
plan before connecting, performs a read-only remote input preflight before any
write, confines every created path to the operator's dedicated HPC root, and
uses an exclusive remote state file to prevent duplicate launches.  Process
state is inspected through ``/proc`` only; this tool never sends a process
signal.
"""

import argparse
import base64
import hashlib
import json
import os
import posixpath
import re
import secrets
import stat
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from research_agent_workstation.server.core.gpu_credentials import (  # noqa: E402
    ALLOWED_GPU_REMOTE_ROOT,
    connect_ssh,
)

PLAN_SCHEMA = "evomind.mlebench.may2022_cpu_feature_diagnostic_plan.v1"
RESULT_SCHEMA = "evomind.mlebench.may2022_cpu_feature_diagnostic.v1"
DEPLOYMENT_SCHEMA = "evomind.hpc.job89941_may2022_cpu_diagnostic_deployment.v1"
STATUS_SCHEMA = "evomind.hpc.job89941_may2022_cpu_diagnostic_status.v1"
COLLECTION_SCHEMA = "evomind.hpc.job89941_may2022_cpu_diagnostic_collection.v1"
REMOTE_STATE_SCHEMA = "evomind.hpc.job89941_may2022_cpu_diagnostic_run_state.v1"
EXPECTED_JOB_ID = 89941
EXPECTED_THREADS = 48
EXPECTED_COMPETITION = "tabular-playground-series-may-2022"
REMOTE_UNIFIED_SITE_PACKAGES = (
    f"{ALLOWED_GPU_REMOTE_ROOT}/mlebench_lite_runtime/"
    "unified-py310-sklearn1.7.2/site-packages"
)
EXPECTED_SOURCE_PATHS = {
    "scripts/diagnose_may2022_cpu_feature_model.py",
    "scripts/mlebench_medal_recovery_adapters.py",
    "scripts/mlebench_wave2_adapters.py",
    "scripts/russian_transliteration.py",
    "scripts/run_mlebench_lite_wave0.py",
    "src/research_os/mlebench_phase_a.py",
}
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_PLAN_GLOB = "may2022_cpu_feature_diagnostic_job89941_frozen_plan_*.json"
DEFAULT_EVIDENCE_DIR = (
    PROJECT_ROOT / "workspace" / "hpc" / "job89941_may2022_cpu_diagnostic"
)
DEFAULT_PROFILE_DIR = (
    Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    / "ResearchAgentWorkstation"
    / "profiles"
    / "job89941_cpu"
)
TERMINAL_STATUSES = {
    "completed",
    "failed",
    "artifact_invalid",
    "source_drift",
    "state_invalid",
}


class DeploymentError(RuntimeError):
    """Raised when a deployment or collection contract fails."""


class PlanValidationError(DeploymentError):
    """Raised when the frozen plan does not match its local bytes/contracts."""


@dataclass(frozen=True)
class SourceRecord:
    local_path: Path
    relative_path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class ValidatedPlan:
    path: Path
    payload: dict[str, Any]
    bytes: int
    sha256: str
    sources: tuple[SourceRecord, ...]

    @property
    def run_id(self) -> str:
        return str(self.payload["run_id"])


@dataclass(frozen=True)
class RemoteLayout:
    root: str
    base: str
    staged_root: str
    plan: str
    output: str
    model: str
    log: str
    state: str
    tmp: str

    def write_paths(self) -> tuple[str, ...]:
        return (
            self.base,
            self.staged_root,
            self.plan,
            self.output,
            self.model,
            self.log,
            self.state,
            self.tmp,
        )


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def find_default_plan(project_root: Path = PROJECT_ROOT) -> Path:
    plan_dir = project_root / "workspace" / "mlebench_plans"
    candidates = sorted(
        plan_dir.glob(DEFAULT_PLAN_GLOB),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No frozen Job 89941 May 2022 plan under {plan_dir}")
    return candidates[0]


def _require_sha256(value: object, field: str) -> str:
    digest = str(value or "").lower()
    if not SHA256_PATTERN.fullmatch(digest):
        raise PlanValidationError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def ensure_remote_path(path: str, *, root: str = ALLOWED_GPU_REMOTE_ROOT) -> str:
    """Return a normalized absolute POSIX path confined to ``root``."""

    root_path = PurePosixPath(root)
    candidate = PurePosixPath(path)
    if not root_path.is_absolute() or not candidate.is_absolute():
        raise ValueError(f"Remote path must be absolute: {path}")
    if any(part in {"", ".", ".."} for part in candidate.parts[1:]):
        raise ValueError(f"Remote path is not normalized: {path}")
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"Remote path escaped the dedicated root: {path}") from exc
    return candidate.as_posix()


def ensure_remote_write_path(path: str, *, root: str = ALLOWED_GPU_REMOTE_ROOT) -> str:
    normalized = ensure_remote_path(path, root=root)
    if PurePosixPath(normalized) == PurePosixPath(root):
        raise ValueError("Remote writes must target a subdirectory of the dedicated root")
    return normalized


def validate_relative_source_path(value: object) -> str:
    raw = str(value or "").replace("\\", "/")
    candidate = PurePosixPath(raw)
    if (
        not raw
        or candidate.is_absolute()
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise PlanValidationError(f"Unsafe source relative_path: {value}")
    return candidate.as_posix()


def _assert_exact_contract(payload: Mapping[str, Any]) -> None:
    expected = {
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_allowed": False,
        "writes_confined_to_remote_root": True,
        "diagnostic_not_medal": True,
    }
    contracts = payload.get("contracts")
    if not isinstance(contracts, Mapping):
        raise PlanValidationError("Frozen plan contracts must be an object")
    for key, required in expected.items():
        if contracts.get(key) is not required:
            raise PlanValidationError(f"Frozen plan contract {key} must be {required!r}")


def validate_frozen_plan(
    plan_path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
    allowed_root: str = ALLOWED_GPU_REMOTE_ROOT,
) -> ValidatedPlan:
    """Validate plan fields and every declared source against current bytes."""

    path = Path(plan_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise PlanValidationError("Frozen plan JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise PlanValidationError("Frozen plan root must be an object")
    exact_fields = {
        "schema": PLAN_SCHEMA,
        "status": "frozen",
        "job_id": EXPECTED_JOB_ID,
        "resource_mode": "cpu_only",
        "competition_id": EXPECTED_COMPETITION,
        "visibility_mode": "PUBLIC_ONLY",
    }
    for key, expected in exact_fields.items():
        if payload.get(key) != expected:
            raise PlanValidationError(f"Frozen plan {key} must equal {expected!r}")
    if str(payload.get("remote_root") or "").rstrip("/") != allowed_root.rstrip("/"):
        raise PlanValidationError("Frozen plan remote_root does not match the dedicated root")
    run_id = str(payload.get("run_id") or "")
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise PlanValidationError("Frozen plan run_id is unsafe")
    training = payload.get("training")
    if not isinstance(training, Mapping):
        raise PlanValidationError("Frozen plan training must be an object")
    integer_training = {
        "fold": 0,
        "seed": 42,
        "threads": EXPECTED_THREADS,
    }
    for key, expected in integer_training.items():
        if training.get(key) != expected:
            raise PlanValidationError(f"Frozen plan training.{key} must equal {expected}")
    for key in ("max_rounds", "num_leaves", "early_stopping_rounds"):
        if type(training.get(key)) is not int or int(training[key]) <= 0:
            raise PlanValidationError(f"Frozen plan training.{key} must be a positive integer")
    if isinstance(training.get("learning_rate"), bool) or not isinstance(
        training.get("learning_rate"), (int, float)
    ) or not (
        0 < float(training["learning_rate"]) <= 1
    ):
        raise PlanValidationError("Frozen plan training.learning_rate is invalid")
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise PlanValidationError("Frozen plan inputs must be an object")
    for key in ("data_root", "oof_bundle"):
        try:
            ensure_remote_path(str(inputs.get(key) or ""), root=allowed_root)
        except ValueError as exc:
            raise PlanValidationError(f"Frozen plan inputs.{key} escaped the root") from exc
    _require_sha256(inputs.get("oof_sha256"), "inputs.oof_sha256")
    if not isinstance(inputs.get("oof_bytes"), int) or int(inputs["oof_bytes"]) <= 0:
        raise PlanValidationError("Frozen plan inputs.oof_bytes must be a positive integer")
    _assert_exact_contract(payload)

    source_items = payload.get("sources")
    if not isinstance(source_items, list) or not source_items:
        raise PlanValidationError("Frozen plan sources must be a non-empty list")
    resolved_project = Path(project_root).resolve()
    records: list[SourceRecord] = []
    seen_relative: set[str] = set()
    for index, item in enumerate(source_items):
        if not isinstance(item, Mapping):
            raise PlanValidationError(f"Frozen plan sources[{index}] must be an object")
        relative = validate_relative_source_path(item.get("relative_path"))
        if relative in seen_relative:
            raise PlanValidationError(f"Duplicate source relative_path: {relative}")
        seen_relative.add(relative)
        expected_local = (resolved_project / Path(*PurePosixPath(relative).parts)).resolve()
        declared_local = Path(str(item.get("local_path") or "")).resolve()
        try:
            declared_local.relative_to(resolved_project)
        except ValueError as exc:
            raise PlanValidationError(f"Source escaped the project root: {declared_local}") from exc
        if declared_local != expected_local:
            raise PlanValidationError(
                f"Source local_path does not match project_root/relative_path: {relative}"
            )
        if not declared_local.is_file():
            raise PlanValidationError(f"Frozen source is missing: {relative}")
        expected_bytes = item.get("bytes")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise PlanValidationError(f"Source bytes are invalid: {relative}")
        expected_sha = _require_sha256(item.get("sha256"), f"sources[{index}].sha256")
        actual_bytes = declared_local.stat().st_size
        actual_sha = sha256_file(declared_local)
        if actual_bytes != expected_bytes or actual_sha != expected_sha:
            raise PlanValidationError(
                f"Frozen source drift: {relative}; expected {expected_bytes}/{expected_sha}, "
                f"got {actual_bytes}/{actual_sha}"
            )
        records.append(
            SourceRecord(
                local_path=declared_local,
                relative_path=relative,
                bytes=actual_bytes,
                sha256=actual_sha,
            )
        )
    actual_source_paths = {record.relative_path for record in records}
    if actual_source_paths != EXPECTED_SOURCE_PATHS:
        missing = sorted(EXPECTED_SOURCE_PATHS - actual_source_paths)
        extra = sorted(actual_source_paths - EXPECTED_SOURCE_PATHS)
        raise PlanValidationError(
            f"Frozen plan source set differs from the canonical five files; "
            f"missing={missing}, extra={extra}"
        )
    return ValidatedPlan(
        path=path,
        payload=payload,
        bytes=path.stat().st_size,
        sha256=sha256_file(path),
        sources=tuple(records),
    )


def build_remote_layout(
    plan: ValidatedPlan,
    *,
    allowed_root: str = ALLOWED_GPU_REMOTE_ROOT,
) -> RemoteLayout:
    base = ensure_remote_write_path(
        posixpath.join(
            allowed_root,
            "evomind_mle22",
            "job89941_may2022_cpu_diagnostic",
            plan.run_id,
        ),
        root=allowed_root,
    )
    staged_root = ensure_remote_write_path(posixpath.join(base, "frozen_source"), root=allowed_root)
    output = ensure_remote_write_path(
        posixpath.join(base, "artifacts", "may2022_cpu_feature_diagnostic.json"),
        root=allowed_root,
    )
    layout = RemoteLayout(
        root=allowed_root,
        base=base,
        staged_root=staged_root,
        plan=ensure_remote_write_path(
            posixpath.join(base, "plans", plan.path.name), root=allowed_root
        ),
        output=output,
        model=ensure_remote_write_path(
            str(PurePosixPath(output).with_suffix(".lightgbm.txt")), root=allowed_root
        ),
        log=ensure_remote_write_path(
            posixpath.join(base, "logs", f"{plan.run_id}.log"), root=allowed_root
        ),
        state=ensure_remote_write_path(
            posixpath.join(base, "state", f"{plan.run_id}.json"), root=allowed_root
        ),
        tmp=ensure_remote_write_path(posixpath.join(base, "tmp"), root=allowed_root),
    )
    for path in layout.write_paths():
        ensure_remote_write_path(path, root=allowed_root)
    return layout


def remote_source_records(
    plan: ValidatedPlan,
    layout: RemoteLayout,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in plan.sources:
        remote_path = ensure_remote_write_path(
            posixpath.join(layout.staged_root, record.relative_path), root=layout.root
        )
        records.append(
            {
                "local_path": str(record.local_path),
                "relative_path": record.relative_path,
                "remote_path": remote_path,
                "bytes": record.bytes,
                "sha256": record.sha256,
            }
        )
    return records


def build_runner_argv(plan: ValidatedPlan, layout: RemoteLayout) -> list[str]:
    training = plan.payload["training"]
    inputs = plan.payload["inputs"]
    runner = ensure_remote_write_path(
        posixpath.join(
            layout.staged_root,
            "scripts",
            "diagnose_may2022_cpu_feature_model.py",
        ),
        root=layout.root,
    )
    return [
        "python3",
        runner,
        "--data-root",
        ensure_remote_path(str(inputs["data_root"]), root=layout.root),
        "--oof-bundle",
        ensure_remote_path(str(inputs["oof_bundle"]), root=layout.root),
        "--output",
        layout.output,
        "--fold",
        str(training["fold"]),
        "--seed",
        str(training["seed"]),
        "--threads",
        str(training["threads"]),
        "--max-rounds",
        str(training["max_rounds"]),
        "--learning-rate",
        str(training["learning_rate"]),
        "--num-leaves",
        str(training["num_leaves"]),
        "--early-stopping-rounds",
        str(training["early_stopping_rounds"]),
    ]


def build_runtime_environment(plan: ValidatedPlan, layout: RemoteLayout) -> dict[str, str]:
    threads = int(plan.payload["training"]["threads"])
    if threads != EXPECTED_THREADS:
        raise PlanValidationError(f"CPU diagnostic must use exactly {EXPECTED_THREADS} threads")
    return {
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": f"{layout.staged_root}:{REMOTE_UNIFIED_SITE_PACKAGES}",
        "TMPDIR": layout.tmp,
        "OMP_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "NUMEXPR_NUM_THREADS": str(threads),
        "LIGHTGBM_NUM_THREADS": str(threads),
    }


def _python_command(source: str) -> str:
    encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
    return f"python3 -c \"import base64;exec(base64.b64decode('{encoded}'))\""


def render_readonly_preflight_command(plan: ValidatedPlan) -> str:
    inputs = plan.payload["inputs"]
    source = f"""from __future__ import annotations
import hashlib, json, os, pathlib, sys
root = pathlib.Path({plan.payload['remote_root']!r}).resolve(strict=True)
runtime_site = pathlib.Path({REMOTE_UNIFIED_SITE_PACKAGES!r}).resolve(strict=True)
sys.path.insert(0, str(runtime_site))
import lightgbm, numpy, pandas, sklearn
from sklearn.metrics import roc_auc_score
data_root = pathlib.Path({str(inputs['data_root'])!r}).resolve(strict=True)
oof = pathlib.Path({str(inputs['oof_bundle'])!r}).resolve(strict=True)
competition = {EXPECTED_COMPETITION!r}
expected_oof_bytes = {int(inputs['oof_bytes'])!r}
expected_oof_sha256 = {str(inputs['oof_sha256'])!r}
def inside(path):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()
public = (data_root / competition / 'prepared' / 'public').resolve(strict=True)
train = (public / 'train.csv').resolve(strict=True)
oof_bytes = oof.stat().st_size
oof_sha256 = digest(oof)
versions = {{
    'lightgbm': str(lightgbm.__version__),
    'numpy': str(numpy.__version__),
    'pandas': str(pandas.__version__),
    'scikit_learn': str(sklearn.__version__),
}}
payload = {{
    'schema': 'evomind.hpc.job89941_may2022_cpu_readonly_preflight.v1',
    'root': str(root),
    'root_is_directory': root.is_dir(),
    'root_writable_by_identity': os.access(root, os.W_OK),
    'runtime_site': str(runtime_site),
    'sklearn_metric_imported': callable(roc_auc_score),
    'data_root': str(data_root),
    'public_root': str(public),
    'train_path': str(train),
    'train_bytes': train.stat().st_size,
    'oof_path': str(oof),
    'oof_bytes': oof_bytes,
    'oof_sha256': oof_sha256,
    'cpu_count': os.cpu_count(),
    'python': sys.version.split()[0],
    'packages': versions,
}}
payload['passed'] = bool(
    payload['root_is_directory']
    and payload['root_writable_by_identity']
    and inside(data_root)
    and inside(runtime_site)
    and inside(public)
    and inside(train)
    and inside(oof)
    and data_root.is_dir()
    and runtime_site.is_dir()
    and public.is_dir()
    and train.is_file()
    and oof.is_file()
    and oof_bytes == expected_oof_bytes
    and oof_sha256 == expected_oof_sha256
    and payload['sklearn_metric_imported']
    and (os.cpu_count() or 0) >= {EXPECTED_THREADS!r}
    and all(versions.values())
)
print(json.dumps(payload, sort_keys=True))
"""
    return _python_command(source)


def render_remote_hash_command(
    records: Sequence[Mapping[str, Any]],
    *,
    root: str = ALLOWED_GPU_REMOTE_ROOT,
) -> str:
    expected: list[dict[str, Any]] = []
    for item in records:
        path = ensure_remote_write_path(str(item["remote_path"]), root=root)
        expected.append(
            {
                "path": path,
                "bytes": int(item["bytes"]),
                "sha256": _require_sha256(item["sha256"], f"remote record {path}"),
            }
        )
    source = f"""from __future__ import annotations
import hashlib, json, pathlib
root = pathlib.Path({root!r}).resolve(strict=True)
expected = {expected!r}
def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()
items = []
for record in expected:
    path = pathlib.Path(record['path'])
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
        confined = True
    except ValueError:
        confined = False
    size = path.stat().st_size
    sha256 = digest(path)
    items.append({{
        'path': str(path), 'resolved': str(resolved), 'confined': confined,
        'bytes': size, 'sha256': sha256,
        'matches': confined and size == record['bytes'] and sha256 == record['sha256'],
    }})
payload = {{
    'schema': 'evomind.hpc.job89941_may2022_staged_hashes.v1',
    'items': items,
    'passed': bool(items) and all(item['matches'] for item in items),
}}
print(json.dumps(payload, sort_keys=True))
"""
    return _python_command(source)


def render_staged_import_smoke_command(layout: RemoteLayout) -> str:
    """Import the exact staged runner dependency graph before launch."""

    source = f"""from __future__ import annotations
import json, pathlib, sys
staged_root = pathlib.Path({layout.staged_root!r}).resolve(strict=True)
runtime_site = pathlib.Path({REMOTE_UNIFIED_SITE_PACKAGES!r}).resolve(strict=True)
sys.path[:0] = [str(staged_root), str(runtime_site)]
from scripts import diagnose_may2022_cpu_feature_model as diagnostic
from scripts import mlebench_medal_recovery_adapters as recovery
from scripts import mlebench_wave2_adapters as wave2
from scripts import russian_transliteration
from scripts import run_mlebench_lite_wave0 as wave0
from research_os import mlebench_phase_a
modules = {{
    'diagnostic': diagnostic,
    'recovery': recovery,
    'wave2': wave2,
    'russian_transliteration': russian_transliteration,
    'wave0': wave0,
    'phase_a': mlebench_phase_a,
}}
paths = {{name: str(pathlib.Path(module.__file__).resolve(strict=True)) for name, module in modules.items()}}
passed = all(pathlib.Path(path).is_relative_to(staged_root) for path in paths.values())
print(json.dumps({{
    'schema': 'evomind.hpc.job89941_may2022_staged_import_smoke.v1',
    'paths': paths,
    'passed': passed,
}}, sort_keys=True))
"""
    return _python_command(source)


def render_launch_command(plan: ValidatedPlan, layout: RemoteLayout) -> str:
    argv = build_runner_argv(plan, layout)
    environment = build_runtime_environment(plan, layout)
    runner_sha = next(
        record.sha256
        for record in plan.sources
        if record.relative_path == "scripts/diagnose_may2022_cpu_feature_model.py"
    )
    source = f"""from __future__ import annotations
import json, os, pathlib, subprocess, time
state_path = pathlib.Path({layout.state!r})
result_path = pathlib.Path({layout.output!r})
log_path = pathlib.Path({layout.log!r})
runner_path = pathlib.Path({argv[1]!r})
argv = {argv!r}
environment_contract = {environment!r}
plan_sha256 = {plan.sha256!r}
runner_sha256 = {runner_sha!r}
def process_snapshot(pid):
    if not isinstance(pid, int) or pid <= 0:
        return {{'running': False, 'owned': False, 'pid': pid, 'state': None}}
    proc = pathlib.Path('/proc') / str(pid)
    try:
        command = (proc / 'cmdline').read_bytes().replace(b'\\x00', b' ').decode('utf-8', 'replace')
        fields = (proc / 'stat').read_text(encoding='utf-8', errors='replace').split()
        state = fields[2] if len(fields) > 2 else None
    except OSError:
        return {{'running': False, 'owned': False, 'pid': pid, 'state': None}}
    owned = str(runner_path) in command and str(result_path) in command
    return {{'running': bool(owned and state != 'Z'), 'owned': owned, 'pid': pid, 'state': state}}
def matching_processes():
    matches = []
    for proc in pathlib.Path('/proc').glob('[0-9]*'):
        try:
            command = (proc / 'cmdline').read_bytes().replace(b'\\x00', b' ').decode('utf-8', 'replace')
            fields = (proc / 'stat').read_text(encoding='utf-8', errors='replace').split()
        except OSError:
            continue
        if str(runner_path) in command and str(result_path) in command and len(fields) > 2 and fields[2] != 'Z':
            matches.append(int(proc.name))
    return sorted(matches)
state_path.parent.mkdir(parents=True, exist_ok=True)
log_path.parent.mkdir(parents=True, exist_ok=True)
pathlib.Path({layout.tmp!r}).mkdir(parents=True, exist_ok=True)
if state_path.exists():
    existing = json.loads(state_path.read_text(encoding='utf-8'))
    compatible = bool(
        existing.get('schema') == {REMOTE_STATE_SCHEMA!r}
        and existing.get('plan_sha256') == plan_sha256
        and existing.get('runner_sha256') == runner_sha256
        and existing.get('argv') == argv
        and existing.get('environment_contract') == environment_contract
    )
    if not compatible:
        raise SystemExit('EXISTING_STATE_CONTRACT_DRIFT')
    process = process_snapshot(existing.get('pid'))
    claim_age = max(0.0, time.time() - float(existing.get('created_at_epoch') or 0.0))
    recent_claim = existing.get('status') == 'launch_claimed' and claim_age <= 120.0
    status = 'finalizing' if result_path.is_file() and process['running'] else (
        'completed' if result_path.is_file() else (
            'running' if process['running'] else (
                'launching' if recent_claim else 'stale_failed'
            )
        )
    )
    print(json.dumps({{
        'status': status, 'idempotent_reuse': True,
        'pid': existing.get('pid'), 'process': process,
        'state_path': str(state_path), 'result_exists': result_path.is_file(),
        'claim_age_seconds': claim_age,
        'duplicate_processes_started': 0,
    }}, sort_keys=True))
    raise SystemExit(0)
unmanaged = matching_processes()
if unmanaged:
    print(json.dumps({{
        'status': 'existing_unmanaged_process', 'idempotent_reuse': True,
        'matching_pids': unmanaged, 'duplicate_processes_started': 0,
    }}, sort_keys=True))
    raise SystemExit(0)
claimed = {{
    'schema': {REMOTE_STATE_SCHEMA!r},
    'created_at_epoch': time.time(),
    'status': 'launch_claimed',
    'run_id': {plan.run_id!r},
    'plan_sha256': plan_sha256,
    'runner_sha256': runner_sha256,
    'argv': argv,
    'environment_contract': environment_contract,
    'pid': None,
    'process_signals_sent': 0,
}}
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
try:
    descriptor = os.open(state_path, flags, 0o600)
except FileExistsError:
    print(json.dumps({{
        'status': 'launch_claim_race_reused', 'idempotent_reuse': True,
        'duplicate_processes_started': 0,
    }}, sort_keys=True))
    raise SystemExit(0)
with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
    json.dump(claimed, handle, sort_keys=True)
    handle.write('\\n')
    handle.flush()
    os.fsync(handle.fileno())
child_environment = os.environ.copy()
child_environment.update(environment_contract)
nohup_argv = ['nohup', 'env'] + [f'{{key}}={{value}}' for key, value in sorted(environment_contract.items())] + argv
try:
    with log_path.open('ab', buffering=0) as log_handle:
        process = subprocess.Popen(
            nohup_argv,
            cwd={layout.staged_root!r},
            env=child_environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
except Exception as exc:
    claimed.update({{'status': 'launch_failed', 'error_type': type(exc).__name__}})
    state_path.write_text(json.dumps(claimed, sort_keys=True) + '\\n', encoding='utf-8')
    raise
claimed.update({{'status': 'running', 'pid': process.pid, 'launched_at_epoch': time.time()}})
temporary = state_path.with_suffix(state_path.suffix + '.tmp')
temporary.write_text(json.dumps(claimed, sort_keys=True) + '\\n', encoding='utf-8')
os.replace(temporary, state_path)
print(json.dumps({{
    'status': 'running', 'idempotent_reuse': False, 'pid': process.pid,
    'state_path': str(state_path), 'log_path': str(log_path),
    'nohup': True, 'non_blocking': True, 'duplicate_processes_started': 0,
}}, sort_keys=True))
"""
    return _python_command(source)


def derive_status(
    *,
    state_valid: bool,
    staged_integrity_passed: bool,
    result_present: bool,
    result_valid: bool,
    process_running: bool,
    launch_claimed: bool = False,
) -> str:
    if not state_valid:
        return "state_invalid"
    if not staged_integrity_passed:
        return "source_drift"
    if result_present and not result_valid:
        return "artifact_invalid"
    if result_valid and process_running:
        return "finalizing"
    if result_valid:
        return "completed"
    if process_running:
        return "running"
    if launch_claimed:
        return "launching"
    return "failed"


def render_status_command(
    plan: ValidatedPlan,
    layout: RemoteLayout,
    staged_records: Sequence[Mapping[str, Any]],
) -> str:
    expected_sources = [
        {
            "path": ensure_remote_write_path(str(item["remote_path"]), root=layout.root),
            "bytes": int(item["bytes"]),
            "sha256": str(item["sha256"]),
        }
        for item in staged_records
    ]
    expected_argv = build_runner_argv(plan, layout)
    expected_environment = build_runtime_environment(plan, layout)
    expected_runner_sha = next(
        record.sha256
        for record in plan.sources
        if record.relative_path == "scripts/diagnose_may2022_cpu_feature_model.py"
    )
    expected_frozen = [
        {
            "path": layout.plan,
            "bytes": plan.bytes,
            "sha256": plan.sha256,
        },
        *expected_sources,
    ]
    source = f"""from __future__ import annotations
import hashlib, json, pathlib, time
root = pathlib.Path({layout.root!r}).resolve(strict=True)
state_path = pathlib.Path({layout.state!r})
result_path = pathlib.Path({layout.output!r})
model_path = pathlib.Path({layout.model!r})
log_path = pathlib.Path({layout.log!r})
plan_path = pathlib.Path({layout.plan!r})
runner_path = pathlib.Path({build_runner_argv(plan, layout)[1]!r})
expected_frozen = {expected_frozen!r}
def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()
def record(path):
    if not path.is_file():
        return None
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError:
        return {{'path': str(path), 'confined': False}}
    return {{'path': str(path), 'confined': True, 'bytes': path.stat().st_size, 'sha256': digest(path)}}
state = None
state_errors = []
if state_path.is_file():
    try:
        state = json.loads(state_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        state_errors.append(type(exc).__name__)
else:
    state_errors.append('missing')
state_valid = bool(
    isinstance(state, dict)
    and state.get('schema') == {REMOTE_STATE_SCHEMA!r}
    and state.get('run_id') == {plan.run_id!r}
    and state.get('plan_sha256') == {plan.sha256!r}
    and state.get('runner_sha256') == {expected_runner_sha!r}
    and state.get('argv') == {expected_argv!r}
    and state.get('environment_contract') == {expected_environment!r}
    and state.get('process_signals_sent') == 0
)
process = {{'pid': None, 'running': False, 'owned': False, 'state': None}}
if isinstance(state, dict):
    pid = state.get('pid')
    process['pid'] = pid
    if isinstance(pid, int) and pid > 0:
        proc = pathlib.Path('/proc') / str(pid)
        try:
            command = (proc / 'cmdline').read_bytes().replace(b'\\x00', b' ').decode('utf-8', 'replace')
            fields = (proc / 'stat').read_text(encoding='utf-8', errors='replace').split()
            process['state'] = fields[2] if len(fields) > 2 else None
            process['owned'] = str(runner_path) in command and str(result_path) in command
            process['running'] = bool(process['owned'] and process['state'] != 'Z')
        except OSError:
            pass
source_checks = []
for expected in expected_frozen:
    actual = record(pathlib.Path(expected['path']))
    matches = bool(
        actual and actual.get('confined')
        and actual.get('bytes') == expected['bytes']
        and actual.get('sha256') == expected['sha256']
    )
    source_checks.append({{'expected': expected, 'actual': actual, 'matches': matches}})
staged_integrity_passed = bool(source_checks) and all(item['matches'] for item in source_checks)
result = None
result_errors = []
result_record = record(result_path)
model_record = record(model_path)
if result_record:
    try:
        result = json.loads(result_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        result_errors.append(type(exc).__name__)
if isinstance(result, dict):
    if result.get('schema') != {RESULT_SCHEMA!r}:
        result_errors.append('schema')
    if result.get('status') != 'completed':
        result_errors.append('status')
    if result.get('competition_id') != {EXPECTED_COMPETITION!r}:
        result_errors.append('competition_id')
    if result.get('visibility_mode') != 'PUBLIC_ONLY':
        result_errors.append('visibility_mode')
    if result.get('private_labels_used') is not False:
        result_errors.append('private_labels_used')
    if result.get('official_grader_executed') is not False:
        result_errors.append('official_grader_executed')
    if result.get('kaggle_submission_executed') is not False:
        result_errors.append('kaggle_submission_executed')
    if result.get('process_signals_sent') != 0:
        result_errors.append('process_signals_sent')
    if (result.get('input_oof_bundle') or {{}}).get('sha256') != {str(plan.payload['inputs']['oof_sha256'])!r}:
        result_errors.append('input_oof_sha256')
    model = result.get('model') or {{}}
    if model.get('path') != str(model_path):
        result_errors.append('model_path')
    if not model_record or model_record.get('sha256') != model.get('sha256'):
        result_errors.append('model_sha256')
result_present = bool(result_record)
result_valid = bool(isinstance(result, dict) and not result_errors and model_record and model_record.get('confined'))
claim_age = max(0.0, time.time() - float((state or {{}}).get('created_at_epoch') or 0.0))
launch_claimed = bool(
    isinstance(state, dict)
    and state.get('status') == 'launch_claimed'
    and claim_age <= 120.0
)
if not state_valid:
    status = 'state_invalid'
elif not staged_integrity_passed:
    status = 'source_drift'
elif result_present and not result_valid:
    status = 'artifact_invalid'
elif result_valid and process['running']:
    status = 'finalizing'
elif result_valid:
    status = 'completed'
elif process['running']:
    status = 'running'
elif launch_claimed:
    status = 'launching'
else:
    status = 'failed'
tail = ''
if log_path.is_file():
    with log_path.open('rb') as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - 16000))
        tail = handle.read().decode('utf-8', 'replace')[-16000:]
payload = {{
    'schema': {STATUS_SCHEMA!r},
    'status': status,
    'state_valid': state_valid,
    'state_errors': state_errors,
    'state': state,
    'process': process,
    'staged_integrity_passed': staged_integrity_passed,
    'source_checks': source_checks,
    'result_present': result_present,
    'result_valid': result_valid,
    'result_errors': result_errors,
    'launch_claimed': launch_claimed,
    'claim_age_seconds': claim_age,
    'artifacts': {{
        'result': result_record,
        'model': model_record,
        'log': record(log_path),
        'state': record(state_path),
        'plan': record(plan_path),
    }},
    'log_tail': tail,
    'process_signals_sent': 0,
}}
print(json.dumps(payload, sort_keys=True))
"""
    return _python_command(source)


def validate_collected_result(
    payload: Mapping[str, Any],
    plan: ValidatedPlan,
    layout: RemoteLayout,
    *,
    model_sha256: str,
) -> None:
    errors: list[str] = []
    exact = {
        "schema": RESULT_SCHEMA,
        "status": "completed",
        "competition_id": EXPECTED_COMPETITION,
        "visibility_mode": "PUBLIC_ONLY",
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
    }
    for key, expected in exact.items():
        if payload.get(key) != expected:
            errors.append(key)
    input_bundle = payload.get("input_oof_bundle")
    if not isinstance(input_bundle, Mapping) or input_bundle.get("sha256") != plan.payload[
        "inputs"
    ]["oof_sha256"]:
        errors.append("input_oof_bundle.sha256")
    model = payload.get("model")
    if not isinstance(model, Mapping):
        errors.append("model")
    else:
        if model.get("path") != layout.model:
            errors.append("model.path")
        if model.get("sha256") != model_sha256:
            errors.append("model.sha256")
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get("threads") != EXPECTED_THREADS:
        errors.append("runtime.threads")
    if errors:
        raise DeploymentError(f"Collected result contract failed: {sorted(errors)}")


def exec_json(client: Any, command: str, *, timeout: int = 300) -> dict[str, Any]:
    _stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    output = stdout.read().decode("utf-8", errors="replace").strip()
    error = stderr.read().decode("utf-8", errors="replace").strip()
    code = stdout.channel.recv_exit_status()
    if code:
        raise DeploymentError(
            f"Remote command failed with exit {code}: {(error or output)[-1000:]}"
        )
    try:
        payload = json.loads(output.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise DeploymentError("Remote command did not return a JSON object") from exc
    if not isinstance(payload, dict):
        raise DeploymentError("Remote command JSON root must be an object")
    return payload


def connect_job89941(profile_dir: Path = DEFAULT_PROFILE_DIR) -> Any:
    resolved = Path(profile_dir).resolve()
    if resolved.name != "job89941_cpu":
        raise DeploymentError("Only the job89941_cpu DPAPI profile is accepted")
    from workspace.hpc.probe_hpc_gpu_profile import _load_profile

    config = _load_profile(resolved)
    return connect_ssh(config)


def _sftp_lstat(sftp: Any, path: str) -> Any | None:
    try:
        return sftp.lstat(path)
    except OSError:
        return None


def sftp_mkdirs_confined(
    sftp: Any,
    path: str,
    *,
    root: str = ALLOWED_GPU_REMOTE_ROOT,
) -> None:
    target = PurePosixPath(ensure_remote_write_path(path, root=root))
    root_path = PurePosixPath(root)
    root_stat = _sftp_lstat(sftp, root_path.as_posix())
    if root_stat is None or not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
        raise DeploymentError("Dedicated remote root is missing, not a directory, or a symlink")
    current = root_path
    for part in target.relative_to(root_path).parts:
        current /= part
        current_path = current.as_posix()
        item_stat = _sftp_lstat(sftp, current_path)
        if item_stat is None:
            sftp.mkdir(current_path, mode=0o700)
            item_stat = sftp.lstat(current_path)
        if stat.S_ISLNK(item_stat.st_mode) or not stat.S_ISDIR(item_stat.st_mode):
            raise DeploymentError(f"Remote directory component is unsafe: {current_path}")


def _sftp_sha256(sftp: Any, path: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with sftp.open(path, "rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            total += len(block)
            digest.update(block)
    return total, digest.hexdigest()


def sftp_upload_exact(
    sftp: Any,
    local_path: Path,
    remote_path: str,
    *,
    expected_bytes: int,
    expected_sha256: str,
    root: str = ALLOWED_GPU_REMOTE_ROOT,
) -> str:
    remote_path = ensure_remote_write_path(remote_path, root=root)
    local_path = Path(local_path).resolve()
    if local_path.stat().st_size != expected_bytes or sha256_file(local_path) != expected_sha256:
        raise DeploymentError(f"Local file changed immediately before upload: {local_path}")
    sftp_mkdirs_confined(sftp, posixpath.dirname(remote_path), root=root)
    existing_stat = _sftp_lstat(sftp, remote_path)
    if existing_stat is not None:
        if stat.S_ISLNK(existing_stat.st_mode) or not stat.S_ISREG(existing_stat.st_mode):
            raise DeploymentError(f"Existing remote destination is unsafe: {remote_path}")
        remote_bytes, remote_sha = _sftp_sha256(sftp, remote_path)
        if remote_bytes != expected_bytes or remote_sha != expected_sha256:
            raise DeploymentError(f"Existing frozen remote file has drifted: {remote_path}")
        return "reused_exact"
    temporary = ensure_remote_write_path(
        remote_path + f".uploading.{os.getpid()}.{secrets.token_hex(4)}", root=root
    )
    try:
        sftp.put(str(local_path), temporary, confirm=True)
        uploaded_stat = sftp.lstat(temporary)
        if stat.S_ISLNK(uploaded_stat.st_mode) or not stat.S_ISREG(uploaded_stat.st_mode):
            raise DeploymentError(f"Uploaded temporary is not a regular file: {temporary}")
        remote_bytes, remote_sha = _sftp_sha256(sftp, temporary)
        if remote_bytes != expected_bytes or remote_sha != expected_sha256:
            raise DeploymentError(f"Uploaded bytes failed verification: {remote_path}")
        try:
            sftp.rename(temporary, remote_path)
        except OSError:
            existing_stat = _sftp_lstat(sftp, remote_path)
            if existing_stat is None:
                raise
            existing_bytes, existing_sha = _sftp_sha256(sftp, remote_path)
            if existing_bytes != expected_bytes or existing_sha != expected_sha256:
                raise DeploymentError(f"Concurrent remote destination drift: {remote_path}")
            sftp.remove(temporary)
            return "reused_concurrent_exact"
        final_bytes, final_sha = _sftp_sha256(sftp, remote_path)
        if final_bytes != expected_bytes or final_sha != expected_sha256:
            raise DeploymentError(f"Final remote bytes failed verification: {remote_path}")
        return "uploaded"
    except Exception:
        if _sftp_lstat(sftp, temporary) is not None:
            try:
                sftp.remove(temporary)
            except OSError:
                pass
        raise


def _plan_remote_record(plan: ValidatedPlan, layout: RemoteLayout) -> dict[str, Any]:
    return {
        "local_path": str(plan.path),
        "relative_path": f"plans/{plan.path.name}",
        "remote_path": layout.plan,
        "bytes": plan.bytes,
        "sha256": plan.sha256,
    }


def stage_frozen_files(
    client: Any,
    plan: ValidatedPlan,
    layout: RemoteLayout,
) -> dict[str, Any]:
    records = [_plan_remote_record(plan, layout), *remote_source_records(plan, layout)]
    uploaded: list[dict[str, Any]] = []
    with client.open_sftp() as sftp:
        for directory in (
            layout.base,
            layout.staged_root,
            posixpath.dirname(layout.output),
            posixpath.dirname(layout.log),
            posixpath.dirname(layout.state),
            layout.tmp,
        ):
            sftp_mkdirs_confined(sftp, directory, root=layout.root)
        for record in records:
            action = sftp_upload_exact(
                sftp,
                Path(str(record["local_path"])),
                str(record["remote_path"]),
                expected_bytes=int(record["bytes"]),
                expected_sha256=str(record["sha256"]),
                root=layout.root,
            )
            uploaded.append(
                {
                    "relative_path": record["relative_path"],
                    "remote_path": record["remote_path"],
                    "bytes": record["bytes"],
                    "sha256": record["sha256"],
                    "action": action,
                }
            )
    verification = exec_json(
        client,
        render_remote_hash_command(records, root=layout.root),
        timeout=300,
    )
    if verification.get("passed") is not True:
        raise DeploymentError("Independent remote staged-file SHA verification failed")
    import_smoke = exec_json(
        client,
        render_staged_import_smoke_command(layout),
        timeout=300,
    )
    if import_smoke.get("passed") is not True:
        raise DeploymentError("Remote staged dependency import smoke failed")
    return {
        "files": uploaded,
        "verification": verification,
        "import_smoke": import_smoke,
        "passed": True,
    }


def readonly_remote_preflight(client: Any, plan: ValidatedPlan) -> dict[str, Any]:
    payload = exec_json(client, render_readonly_preflight_command(plan), timeout=600)
    if payload.get("passed") is not True:
        raise DeploymentError("Remote read-only root/input/OOF preflight failed")
    if payload.get("oof_sha256") != plan.payload["inputs"]["oof_sha256"]:
        raise DeploymentError("Remote OOF digest differs from frozen plan")
    if payload.get("oof_bytes") != plan.payload["inputs"]["oof_bytes"]:
        raise DeploymentError("Remote OOF byte count differs from frozen plan")
    return payload


def launch_once(client: Any, plan: ValidatedPlan, layout: RemoteLayout) -> dict[str, Any]:
    payload = exec_json(client, render_launch_command(plan, layout), timeout=120)
    if payload.get("duplicate_processes_started") != 0:
        raise DeploymentError("Remote launcher reported a duplicate process")
    if payload.get("status") not in {"running", "finalizing", "completed", "launching"}:
        raise DeploymentError(f"Remote launch did not enter a usable state: {payload.get('status')}")
    return payload


def _validate_status_payload(payload: Mapping[str, Any]) -> None:
    expected = derive_status(
        state_valid=payload.get("state_valid") is True,
        staged_integrity_passed=payload.get("staged_integrity_passed") is True,
        result_present=payload.get("result_present") is True,
        result_valid=payload.get("result_valid") is True,
        process_running=(payload.get("process") or {}).get("running") is True,
        launch_claimed=payload.get("launch_claimed") is True,
    )
    if payload.get("schema") != STATUS_SCHEMA or payload.get("status") != expected:
        raise DeploymentError("Remote status payload failed local classification")
    if payload.get("process_signals_sent") != 0:
        raise DeploymentError("Remote status violated the no-signal contract")


def read_remote_status(
    plan: ValidatedPlan,
    layout: RemoteLayout,
    *,
    profile_dir: Path = DEFAULT_PROFILE_DIR,
    evidence_dir: Path = DEFAULT_EVIDENCE_DIR,
    client: Any | None = None,
) -> dict[str, Any]:
    own_client = client is None
    if client is None:
        client = connect_job89941(profile_dir)
    try:
        payload = exec_json(
            client,
            render_status_command(plan, layout, remote_source_records(plan, layout)),
            timeout=600,
        )
    finally:
        if own_client:
            client.close()
    _validate_status_payload(payload)
    report = {
        **payload,
        "observed_at": now_iso(),
        "job_id": EXPECTED_JOB_ID,
        "run_id": plan.run_id,
        "plan_sha256": plan.sha256,
        "profile": "job89941_cpu",
    }
    write_json_atomic(Path(evidence_dir) / "status_current.json", report)
    return report


def deploy(
    plan_path: Path,
    *,
    profile_dir: Path = DEFAULT_PROFILE_DIR,
    evidence_dir: Path = DEFAULT_EVIDENCE_DIR,
) -> tuple[ValidatedPlan, RemoteLayout, dict[str, Any]]:
    plan = validate_frozen_plan(plan_path)
    layout = build_remote_layout(plan)
    client = connect_job89941(profile_dir)
    try:
        preflight = readonly_remote_preflight(client, plan)
        staging = stage_frozen_files(client, plan, layout)
        launch = launch_once(client, plan, layout)
        status = read_remote_status(
            plan,
            layout,
            profile_dir=profile_dir,
            evidence_dir=evidence_dir,
            client=client,
        )
        if status["status"] not in {"running", "finalizing", "completed", "launching"}:
            raise DeploymentError(
                f"Initial remote status failed closed: {status['status']}"
            )
    finally:
        client.close()
    report = {
        "schema": DEPLOYMENT_SCHEMA,
        "created_at": now_iso(),
        "status": "deployed" if status["status"] in {"running", "finalizing", "completed"} else status["status"],
        "job_id": EXPECTED_JOB_ID,
        "profile": "job89941_cpu",
        "run_id": plan.run_id,
        "plan": {
            "local_path": str(plan.path),
            "remote_path": layout.plan,
            "bytes": plan.bytes,
            "sha256": plan.sha256,
        },
        "remote_root": layout.root,
        "remote_base": layout.base,
        "readonly_preflight": preflight,
        "staging": staging,
        "launch": launch,
        "initial_status": status,
        "runtime_contract": build_runtime_environment(plan, layout),
        "contracts": {
            "remote_preflight_before_writes": True,
            "writes_confined_to_remote_root": True,
            "process_signals_sent": 0,
            "nohup_non_blocking": launch.get("nohup") is True
            or launch.get("idempotent_reuse") is True,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        },
    }
    evidence_path = Path(evidence_dir) / "deployment_current.json"
    write_json_atomic(evidence_path, report)
    report["local_evidence"] = str(evidence_path.resolve())
    return plan, layout, report


def wait_for_completion(
    plan: ValidatedPlan,
    layout: RemoteLayout,
    *,
    profile_dir: Path = DEFAULT_PROFILE_DIR,
    evidence_dir: Path = DEFAULT_EVIDENCE_DIR,
    poll_seconds: float = 30.0,
    timeout_seconds: float = 21600.0,
) -> dict[str, Any]:
    if poll_seconds <= 0 or timeout_seconds <= 0:
        raise ValueError("Poll and timeout seconds must be positive")
    started = time.monotonic()
    observations = 0
    client = connect_job89941(profile_dir)
    try:
        while True:
            status = read_remote_status(
                plan,
                layout,
                profile_dir=profile_dir,
                evidence_dir=evidence_dir,
                client=client,
            )
            observations += 1
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "run_id": plan.run_id,
                        "status": status["status"],
                        "elapsed_seconds": round(elapsed, 1),
                        "observations": observations,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if status["status"] in TERMINAL_STATUSES:
                break
            if elapsed >= timeout_seconds:
                status = {**status, "status": "timeout"}
                break
            time.sleep(min(poll_seconds, max(0.0, timeout_seconds - elapsed)))
    finally:
        client.close()
    report = {
        "schema": "evomind.hpc.job89941_may2022_cpu_diagnostic_wait.v1",
        "created_at": now_iso(),
        "run_id": plan.run_id,
        "observations": observations,
        "elapsed_seconds": time.monotonic() - started,
        "final_status": status,
        "completed": status["status"] == "completed",
        "process_signals_sent": 0,
    }
    write_json_atomic(Path(evidence_dir) / "wait_current.json", report)
    return report


def _download_verified(
    sftp: Any,
    remote: Mapping[str, Any],
    local_path: Path,
    *,
    root: str,
) -> dict[str, Any]:
    remote_path = ensure_remote_write_path(str(remote.get("path") or ""), root=root)
    expected_bytes = int(remote["bytes"])
    expected_sha = _require_sha256(remote["sha256"], f"remote artifact {remote_path}")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = local_path.with_suffix(local_path.suffix + f".{os.getpid()}.tmp")
    try:
        sftp.get(remote_path, str(temporary))
        actual_bytes = temporary.stat().st_size
        actual_sha = sha256_file(temporary)
        if actual_bytes != expected_bytes or actual_sha != expected_sha:
            raise DeploymentError(f"Downloaded artifact hash mismatch: {remote_path}")
        os.replace(temporary, local_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "remote_path": remote_path,
        "local_path": str(local_path.resolve()),
        "bytes": expected_bytes,
        "sha256": expected_sha,
    }


def collect(
    plan: ValidatedPlan,
    layout: RemoteLayout,
    *,
    profile_dir: Path = DEFAULT_PROFILE_DIR,
    evidence_dir: Path = DEFAULT_EVIDENCE_DIR,
) -> dict[str, Any]:
    client = connect_job89941(profile_dir)
    try:
        status_payload = read_remote_status(
            plan,
            layout,
            profile_dir=profile_dir,
            evidence_dir=evidence_dir,
            client=client,
        )
        if status_payload["status"] != "completed":
            raise DeploymentError(
                f"Collection requires completed status, got {status_payload['status']}"
            )
        artifacts = status_payload["artifacts"]
        required = ("result", "model", "log", "state", "plan")
        if any(not isinstance(artifacts.get(name), Mapping) for name in required):
            raise DeploymentError("Completed status is missing a required artifact record")
        local_root = Path(evidence_dir) / "collected" / plan.run_id
        destinations = {
            "result": local_root / "may2022_cpu_feature_diagnostic.json",
            "model": local_root / "may2022_cpu_feature_diagnostic.lightgbm.txt",
            "log": local_root / "run.log",
            "state": local_root / "remote_state.json",
            "plan": local_root / "frozen_plan.json",
        }
        downloads: dict[str, dict[str, Any]] = {}
        with client.open_sftp() as sftp:
            for name in required:
                downloads[name] = _download_verified(
                    sftp,
                    artifacts[name],
                    destinations[name],
                    root=layout.root,
                )
    finally:
        client.close()
    result_payload = json.loads(destinations["result"].read_text(encoding="utf-8"))
    validate_collected_result(
        result_payload,
        plan,
        layout,
        model_sha256=downloads["model"]["sha256"],
    )
    collected_plan = validate_frozen_plan(destinations["plan"])
    if collected_plan.sha256 != plan.sha256:
        raise DeploymentError("Collected frozen plan digest differs from the local plan")
    report = {
        "schema": COLLECTION_SCHEMA,
        "created_at": now_iso(),
        "status": "verified",
        "job_id": EXPECTED_JOB_ID,
        "run_id": plan.run_id,
        "local_root": str(local_root.resolve()),
        "files": downloads,
        "result_contract_verified": True,
        "plan_contract_verified": True,
        "process_signals_sent": 0,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }
    manifest = local_root / "collection_manifest.json"
    write_json_atomic(manifest, report)
    report["local_manifest"] = str(manifest.resolve())
    write_json_atomic(Path(evidence_dir) / "collection_current.json", report)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    deploy_parser = subparsers.add_parser("deploy")
    deploy_parser.add_argument("--wait", action="store_true")
    deploy_parser.add_argument("--collect", action="store_true")
    deploy_parser.add_argument("--poll-seconds", type=float, default=30.0)
    deploy_parser.add_argument("--timeout-seconds", type=float, default=21600.0)
    subparsers.add_parser("status")
    wait_parser = subparsers.add_parser("wait")
    wait_parser.add_argument("--collect", action="store_true")
    wait_parser.add_argument("--poll-seconds", type=float, default=30.0)
    wait_parser.add_argument("--timeout-seconds", type=float, default=21600.0)
    subparsers.add_parser("collect")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    evidence_dir = Path(args.evidence_dir).resolve()
    plan_path = Path(args.plan).resolve() if args.plan else find_default_plan()
    try:
        if args.command == "validate":
            plan = validate_frozen_plan(plan_path)
            layout = build_remote_layout(plan)
            payload: dict[str, Any] = {
                "schema": "evomind.hpc.job89941_may2022_cpu_diagnostic_local_validation.v1",
                "created_at": now_iso(),
                "status": "passed",
                "plan_path": str(plan.path),
                "plan_sha256": plan.sha256,
                "run_id": plan.run_id,
                "source_count": len(plan.sources),
                "remote_base": layout.base,
                "runtime_contract": build_runtime_environment(plan, layout),
                "process_signals_sent": 0,
            }
            evidence = evidence_dir / "local_validation_current.json"
            write_json_atomic(evidence, payload)
        elif args.command == "deploy":
            plan, layout, payload = deploy(
                plan_path,
                profile_dir=args.profile_dir,
                evidence_dir=evidence_dir,
            )
            if args.wait or args.collect:
                wait_report = wait_for_completion(
                    plan,
                    layout,
                    profile_dir=args.profile_dir,
                    evidence_dir=evidence_dir,
                    poll_seconds=args.poll_seconds,
                    timeout_seconds=args.timeout_seconds,
                )
                payload["wait"] = wait_report
                if args.collect:
                    if wait_report["completed"] is not True:
                        raise DeploymentError("Deploy --collect requires a completed wait result")
                    payload["collection"] = collect(
                        plan,
                        layout,
                        profile_dir=args.profile_dir,
                        evidence_dir=evidence_dir,
                    )
            evidence = evidence_dir / "deployment_current.json"
            write_json_atomic(evidence, payload)
        else:
            plan = validate_frozen_plan(plan_path)
            layout = build_remote_layout(plan)
            if args.command == "status":
                payload = read_remote_status(
                    plan,
                    layout,
                    profile_dir=args.profile_dir,
                    evidence_dir=evidence_dir,
                )
                evidence = evidence_dir / "status_current.json"
            elif args.command == "wait":
                payload = wait_for_completion(
                    plan,
                    layout,
                    profile_dir=args.profile_dir,
                    evidence_dir=evidence_dir,
                    poll_seconds=args.poll_seconds,
                    timeout_seconds=args.timeout_seconds,
                )
                if args.collect:
                    if payload["completed"] is not True:
                        raise DeploymentError("Wait --collect requires completed status")
                    payload["collection"] = collect(
                        plan,
                        layout,
                        profile_dir=args.profile_dir,
                        evidence_dir=evidence_dir,
                    )
                evidence = evidence_dir / "wait_current.json"
                write_json_atomic(evidence, payload)
            else:
                payload = collect(
                    plan,
                    layout,
                    profile_dir=args.profile_dir,
                    evidence_dir=evidence_dir,
                )
                evidence = evidence_dir / "collection_current.json"
        print(
            json.dumps(
                {"evidence": str(evidence.resolve()), **payload},
                ensure_ascii=False,
                indent=2,
            )
        )
        if args.command == "wait" and payload.get("completed") is not True:
            return 3
        if (
            args.command == "deploy"
            and (args.wait or args.collect)
            and (payload.get("wait") or {}).get("completed") is not True
        ):
            return 3
        return 0
    except (DeploymentError, OSError, ValueError, json.JSONDecodeError) as exc:
        failure = {
            "schema": "evomind.hpc.job89941_may2022_cpu_diagnostic_failure.v1",
            "created_at": now_iso(),
            "command": args.command,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "process_signals_sent": 0,
            "passed": False,
        }
        evidence = evidence_dir / f"{args.command}_failure_current.json"
        write_json_atomic(evidence, failure)
        print(json.dumps({"evidence": str(evidence.resolve()), **failure}, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
