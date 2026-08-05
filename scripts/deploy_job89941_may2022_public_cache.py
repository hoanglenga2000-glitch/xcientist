#!/usr/bin/env python3
"""Deploy, monitor, and verify the frozen May-2022 CPU precompute on job 89941."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import posixpath
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from research_agent_workstation.server.core.gpu_credentials import (  # noqa: E402
    ALLOWED_GPU_REMOTE_ROOT,
)
from scripts.deploy_job89941_may2022_cpu_diagnostic import (  # noqa: E402
    DEFAULT_PROFILE_DIR,
    REMOTE_UNIFIED_SITE_PACKAGES,
    DeploymentError,
    connect_job89941,
    ensure_remote_path,
    ensure_remote_write_path,
    exec_json,
    sftp_mkdirs_confined,
    sftp_upload_exact,
)

PLAN_SCHEMA = "evomind.mlebench.may2022_public_cache_plan.v1"
STATUS_SCHEMA = "evomind.hpc.job89941_may2022_public_cache_status.v1"
DEPLOYMENT_SCHEMA = "evomind.hpc.job89941_may2022_public_cache_deployment.v1"
COLLECTION_SCHEMA = "evomind.hpc.job89941_may2022_public_cache_collection.v1"
REMOTE_STATE_SCHEMA = "evomind.hpc.job89941_may2022_public_cache_state.v1"
EXPECTED_JOB_ID = 89941
EXPECTED_COMPETITION = "tabular-playground-series-may-2022"
EXPECTED_SOURCE_PATHS = {
    "scripts/precompute_may2022_public_cache.py",
    "scripts/mlebench_medal_recovery_adapters.py",
    "scripts/mlebench_wave2_adapters.py",
    "scripts/russian_transliteration.py",
    "scripts/run_mlebench_lite_wave0.py",
    "src/research_os/mlebench_phase_a.py",
}
EXPECTED_INPUT_PATHS = {
    "prepared/public/train.csv",
    "prepared/public/test.csv",
    "prepared/public/sample_submission.csv",
}
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_PLAN_GLOB = "may2022_public_cache_job89941_frozen_plan_*.json"
DEFAULT_EVIDENCE_DIR = PROJECT_ROOT / "workspace" / "hpc" / "job89941_may2022_public_cache"


@dataclass(frozen=True)
class FileRecord:
    local_path: Path
    relative_path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class FrozenPlan:
    path: Path
    payload: dict[str, Any]
    bytes: int
    sha256: str
    sources: tuple[FileRecord, ...]

    @property
    def run_id(self) -> str:
        return str(self.payload["run_id"])


@dataclass(frozen=True)
class RemoteLayout:
    root: str
    base: str
    staged_root: str
    plan: str
    cache: str
    log: str
    state: str
    tmp: str


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


def _require_sha256(value: object, field: str) -> str:
    digest = str(value or "").lower()
    if not SHA256_PATTERN.fullmatch(digest):
        raise DeploymentError(f"{field} must be a lowercase SHA-256")
    return digest


def _relative_path(value: object) -> str:
    raw = str(value or "").replace("\\", "/")
    candidate = PurePosixPath(raw)
    if not raw or candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise DeploymentError(f"Unsafe relative path: {value}")
    return candidate.as_posix()


def find_default_plan() -> Path:
    candidates = sorted(
        (PROJECT_ROOT / "workspace" / "mlebench_plans").glob(DEFAULT_PLAN_GLOB),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No frozen job89941 May public-cache plan exists")
    return candidates[0]


def validate_plan(path: Path) -> FrozenPlan:
    plan_path = Path(path).resolve()
    payload = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    exact = {
        "schema": PLAN_SCHEMA,
        "status": "frozen",
        "job_id": EXPECTED_JOB_ID,
        "competition_id": EXPECTED_COMPETITION,
        "resource_mode": "cpu_only",
        "visibility_mode": "PUBLIC_ONLY",
        "seed": 42,
        "folds": 5,
        "threads": 48,
    }
    for key, expected in exact.items():
        if payload.get(key) != expected:
            raise DeploymentError(f"Frozen plan {key} must equal {expected!r}")
    run_id = str(payload.get("run_id") or "")
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise DeploymentError("Frozen plan run_id is unsafe")
    if str(payload.get("remote_root") or "").rstrip("/") != ALLOWED_GPU_REMOTE_ROOT.rstrip("/"):
        raise DeploymentError("Frozen plan remote root is invalid")
    contracts = payload.get("contracts")
    required_contracts = {
        "full_public_training_rows": True,
        "private_files_read": [],
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_allowed": False,
        "other_processes_may_be_modified": False,
        "gpu_used": False,
        "cache_manifest_written_last": True,
    }
    if not isinstance(contracts, dict):
        raise DeploymentError("Frozen plan contracts are missing")
    for key, expected in required_contracts.items():
        if contracts.get(key) != expected:
            raise DeploymentError(f"Frozen plan contract failed: {key}")

    data_root = ensure_remote_path(str(payload.get("data_root") or ""))
    expected_data_root = f"{ALLOWED_GPU_REMOTE_ROOT}/mlebench_official_data"
    if data_root != expected_data_root:
        raise DeploymentError("Frozen plan data root is not canonical")
    input_records = payload.get("inputs")
    if not isinstance(input_records, list):
        raise DeploymentError("Frozen plan input inventory is missing")
    actual_inputs: set[str] = set()
    for index, record in enumerate(input_records):
        if not isinstance(record, dict):
            raise DeploymentError(f"Frozen input {index} is invalid")
        relative = _relative_path(record.get("relative_path"))
        actual_inputs.add(relative)
        expected_path = posixpath.join(
            data_root,
            EXPECTED_COMPETITION,
            relative,
        )
        if ensure_remote_path(str(record.get("path") or "")) != expected_path:
            raise DeploymentError(f"Frozen input path mismatch: {relative}")
        if not isinstance(record.get("bytes"), int) or int(record["bytes"]) <= 0:
            raise DeploymentError(f"Frozen input byte count is invalid: {relative}")
        _require_sha256(record.get("sha256"), f"inputs[{index}].sha256")
    if actual_inputs != EXPECTED_INPUT_PATHS:
        raise DeploymentError("Frozen public input inventory differs from the exact allowlist")

    source_records = payload.get("sources")
    if not isinstance(source_records, list):
        raise DeploymentError("Frozen plan sources are missing")
    project = PROJECT_ROOT.resolve()
    records: list[FileRecord] = []
    actual_sources: set[str] = set()
    for index, record in enumerate(source_records):
        if not isinstance(record, dict):
            raise DeploymentError(f"Frozen source {index} is invalid")
        relative = _relative_path(record.get("relative_path"))
        actual_sources.add(relative)
        local = (project / Path(*PurePosixPath(relative).parts)).resolve()
        if Path(str(record.get("local_path") or "")).resolve() != local:
            raise DeploymentError(f"Frozen source local path mismatch: {relative}")
        try:
            local.relative_to(project)
        except ValueError as exc:
            raise DeploymentError(f"Frozen source escaped project: {relative}") from exc
        expected_bytes = record.get("bytes")
        expected_sha = _require_sha256(record.get("sha256"), f"sources[{index}].sha256")
        if (
            not local.is_file()
            or local.stat().st_size != expected_bytes
            or sha256_file(local) != expected_sha
        ):
            raise DeploymentError(f"Frozen source drifted: {relative}")
        records.append(FileRecord(local, relative, int(expected_bytes), expected_sha))
    if actual_sources != EXPECTED_SOURCE_PATHS:
        raise DeploymentError("Frozen source inventory differs from the exact runtime set")
    return FrozenPlan(
        path=plan_path,
        payload=payload,
        bytes=plan_path.stat().st_size,
        sha256=sha256_file(plan_path),
        sources=tuple(records),
    )


def build_layout(plan: FrozenPlan) -> RemoteLayout:
    base = ensure_remote_write_path(
        posixpath.join(
            ALLOWED_GPU_REMOTE_ROOT,
            "evomind_mle22",
            "job89941_may2022_public_cache",
            plan.run_id,
        )
    )
    return RemoteLayout(
        root=ALLOWED_GPU_REMOTE_ROOT,
        base=base,
        staged_root=ensure_remote_write_path(posixpath.join(base, "frozen_source")),
        plan=ensure_remote_write_path(posixpath.join(base, "plans", plan.path.name)),
        cache=ensure_remote_write_path(posixpath.join(base, "cache")),
        log=ensure_remote_write_path(posixpath.join(base, "logs", f"{plan.run_id}.log")),
        state=ensure_remote_write_path(posixpath.join(base, "state", f"{plan.run_id}.json")),
        tmp=ensure_remote_write_path(posixpath.join(base, "tmp")),
    )


def remote_records(plan: FrozenPlan, layout: RemoteLayout) -> list[dict[str, Any]]:
    records = [{
        "local_path": str(plan.path),
        "relative_path": f"plans/{plan.path.name}",
        "remote_path": layout.plan,
        "bytes": plan.bytes,
        "sha256": plan.sha256,
    }]
    for source in plan.sources:
        records.append({
            "local_path": str(source.local_path),
            "relative_path": source.relative_path,
            "remote_path": ensure_remote_write_path(
                posixpath.join(layout.staged_root, source.relative_path)
            ),
            "bytes": source.bytes,
            "sha256": source.sha256,
        })
    return records


def _python_command(source: str) -> str:
    encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
    return f"python3 -c \"import base64;exec(base64.b64decode('{encoded}'))\""


def build_runner_argv(plan: FrozenPlan, layout: RemoteLayout, *, verify_only: bool = False) -> list[str]:
    runner = ensure_remote_write_path(
        posixpath.join(layout.staged_root, "scripts", "precompute_may2022_public_cache.py")
    )
    argv = [
        "python3",
        runner,
        "--data-root",
        ensure_remote_path(str(plan.payload["data_root"])),
        "--output-dir",
        layout.cache,
        "--allowed-root",
        layout.root,
        "--seed",
        str(plan.payload["seed"]),
        "--folds",
        str(plan.payload["folds"]),
    ]
    if verify_only:
        argv.append("--verify-only")
    return argv


def _runtime_env(layout: RemoteLayout, threads: int) -> dict[str, str]:
    pythonpath = ":".join((
        posixpath.join(ALLOWED_GPU_REMOTE_ROOT, ".deps"),
        layout.staged_root,
        posixpath.join(layout.staged_root, "src"),
        posixpath.join(layout.staged_root, "scripts"),
        REMOTE_UNIFIED_SITE_PACKAGES,
    ))
    return {
        "CUDA_VISIBLE_DEVICES": "",
        "EVOMIND_MLEBENCH_PUBLIC_ONLY": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": pythonpath,
        "TMPDIR": layout.tmp,
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "NUMEXPR_NUM_THREADS": str(threads),
    }


def render_preflight(plan: FrozenPlan) -> str:
    source = f"""
import hashlib, json, os, pathlib
root = pathlib.Path({ALLOWED_GPU_REMOTE_ROOT!r}).resolve()
expected = {json.dumps(plan.payload['inputs'], sort_keys=True)!r}
expected = json.loads(expected)
actual = []
errors = []
for record in expected:
    path = pathlib.Path(record['path']).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        errors.append('escaped:' + record['relative_path'])
        continue
    if 'private' in {{part.lower() for part in path.parts}}:
        errors.append('private:' + record['relative_path'])
        continue
    digest = hashlib.sha256(); total = 0
    try:
        with path.open('rb') as handle:
            while True:
                block = handle.read(8 * 1024 * 1024)
                if not block: break
                total += len(block); digest.update(block)
    except OSError as exc:
        errors.append(type(exc).__name__ + ':' + record['relative_path'])
        continue
    item = {{'relative_path': record['relative_path'], 'path': str(path), 'bytes': total, 'sha256': digest.hexdigest()}}
    item['matches'] = total == record['bytes'] and item['sha256'] == record['sha256']
    if not item['matches']: errors.append('drift:' + record['relative_path'])
    actual.append(item)
root_ok = root.is_dir() and os.access(root, os.W_OK)
print(json.dumps({{'schema':'evomind.hpc.job89941_may2022_public_cache_preflight.v1','root_ok':root_ok,'inputs':actual,'errors':errors,'private_paths_read':[],'passed':root_ok and not errors and len(actual)==3}}, sort_keys=True))
"""
    return _python_command(source)


def render_hash_verification(records: Sequence[dict[str, Any]]) -> str:
    expected = [
        {key: record[key] for key in ("relative_path", "remote_path", "bytes", "sha256")}
        for record in records
    ]
    source = f"""
import hashlib, json, pathlib
expected = json.loads({json.dumps(json.dumps(expected, sort_keys=True))})
checks=[]
for item in expected:
    path=pathlib.Path(item['remote_path'])
    digest=hashlib.sha256(); total=0
    with path.open('rb') as handle:
        while True:
            block=handle.read(8*1024*1024)
            if not block: break
            total+=len(block); digest.update(block)
    checks.append({{'relative_path':item['relative_path'],'bytes':total,'sha256':digest.hexdigest(),'matches':total==item['bytes'] and digest.hexdigest()==item['sha256']}})
print(json.dumps({{'schema':'evomind.hpc.job89941_may2022_public_cache_staging.v1','checks':checks,'passed':bool(checks) and all(x['matches'] for x in checks)}},sort_keys=True))
"""
    return _python_command(source)


def render_import_smoke(layout: RemoteLayout) -> str:
    env = _runtime_env(layout, 1)
    source = """
import json, os
import numpy, pandas, sklearn
import mlebench_medal_recovery_adapters as recovery
import precompute_may2022_public_cache as runner
payload={'schema':'evomind.hpc.job89941_may2022_public_cache_import_smoke.v1','cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'cache_schema':recovery.MAY2022_PUBLIC_CACHE_SCHEMA,'runner':runner.__file__,'passed':os.environ.get('CUDA_VISIBLE_DEVICES')==''}
print(json.dumps(payload,sort_keys=True))
"""
    command = _python_command(source)
    prefix = " ".join(f"{key}={json.dumps(value)}" for key, value in env.items())
    return f"env {prefix} {command}"


def render_launch(plan: FrozenPlan, layout: RemoteLayout) -> str:
    source = f"""
import json, os, pathlib, subprocess, time
state_path=pathlib.Path({layout.state!r})
manifest_path=pathlib.Path({posixpath.join(layout.cache, 'cache_manifest.json')!r})
plan_sha={plan.sha256!r}
argv=json.loads({json.dumps(json.dumps(build_runner_argv(plan, layout)))})
env_update=json.loads({json.dumps(json.dumps(_runtime_env(layout, int(plan.payload['threads']))))})
state_path.parent.mkdir(parents=True,exist_ok=True)
try:
    fd=os.open(state_path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
except FileExistsError:
    state=json.loads(state_path.read_text(encoding='utf-8'))
    if state.get('plan_sha256') != plan_sha:
        raise SystemExit('EXISTING_STATE_PLAN_DRIFT')
    pid=int(state.get('pid') or 0)
    running=pid>0 and pathlib.Path(f'/proc/{{pid}}').is_dir()
    status='completed' if manifest_path.is_file() and not running else ('running' if running else 'failed')
    print(json.dumps({{'status':status,'action':'reused_existing_state','pid':pid,'duplicate_processes_started':0}},sort_keys=True))
else:
    claimed={{'schema':{REMOTE_STATE_SCHEMA!r},'status':'launch_claimed','plan_sha256':plan_sha,'pid':None,'created_at_epoch':time.time(),'argv':argv,'process_signals_sent':0,'other_processes_modified':False}}
    with os.fdopen(fd,'w',encoding='utf-8') as handle: json.dump(claimed,handle,sort_keys=True)
    log_path=pathlib.Path({layout.log!r}); log_path.parent.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy(); env.update(env_update)
    with log_path.open('ab',buffering=0) as log:
        process=subprocess.Popen(argv,cwd={layout.base!r},env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    state={{**claimed,'status':'running','pid':process.pid,'launched_at_epoch':time.time()}}
    temporary=state_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state,sort_keys=True),encoding='utf-8'); os.replace(temporary,state_path)
    print(json.dumps({{'status':'running','action':'launched','pid':process.pid,'duplicate_processes_started':0}},sort_keys=True))
"""
    return _python_command(source)


def render_status(plan: FrozenPlan, layout: RemoteLayout) -> str:
    source = f"""
import hashlib, json, pathlib
state_path=pathlib.Path({layout.state!r}); log_path=pathlib.Path({layout.log!r}); manifest_path=pathlib.Path({posixpath.join(layout.cache, 'cache_manifest.json')!r})
def record(path):
    if not path.is_file(): return None
    digest=hashlib.sha256(); total=0
    with path.open('rb') as handle:
        while True:
            block=handle.read(8*1024*1024)
            if not block: break
            total+=len(block); digest.update(block)
    return {{'path':str(path),'bytes':total,'sha256':digest.hexdigest()}}
state=None; errors=[]
try: state=json.loads(state_path.read_text(encoding='utf-8'))
except Exception as exc: errors.append(type(exc).__name__)
pid=int((state or {{}}).get('pid') or 0); running=False; cmdline=''
if pid>0 and pathlib.Path(f'/proc/{{pid}}').is_dir():
    try: cmdline=pathlib.Path(f'/proc/{{pid}}/cmdline').read_bytes().replace(b'\\0',b' ').decode('utf-8','replace')
    except OSError: cmdline=''
    running={posixpath.join(layout.staged_root, 'scripts', 'precompute_may2022_public_cache.py')!r} in cmdline and {layout.cache!r} in cmdline
manifest=record(manifest_path)
if errors or not isinstance(state,dict) or state.get('plan_sha256') != {plan.sha256!r}: status='state_invalid'
elif manifest and running: status='finalizing'
elif manifest: status='completed'
elif running: status='running'
else: status='failed'
tail=''
if log_path.is_file():
    with log_path.open('rb') as handle:
        handle.seek(0,2); size=handle.tell(); handle.seek(max(0,size-16000)); tail=handle.read().decode('utf-8','replace')[-16000:]
print(json.dumps({{'schema':{STATUS_SCHEMA!r},'status':status,'state':state,'state_errors':errors,'process':{{'pid':pid,'running':running,'cmdline_matches':running}},'manifest':manifest,'log':record(log_path),'log_tail':tail,'process_signals_sent':0,'other_processes_modified':False}},sort_keys=True))
"""
    return _python_command(source)


def stage_files(client: Any, plan: FrozenPlan, layout: RemoteLayout) -> dict[str, Any]:
    records = remote_records(plan, layout)
    uploaded = []
    with client.open_sftp() as sftp:
        for directory in (
            layout.base,
            layout.staged_root,
            posixpath.dirname(layout.plan),
            layout.cache,
            posixpath.dirname(layout.log),
            posixpath.dirname(layout.state),
            layout.tmp,
        ):
            sftp_mkdirs_confined(sftp, directory)
        for record in records:
            action = sftp_upload_exact(
                sftp,
                Path(record["local_path"]),
                record["remote_path"],
                expected_bytes=record["bytes"],
                expected_sha256=record["sha256"],
            )
            uploaded.append({**record, "action": action})
    verification = exec_json(client, render_hash_verification(records), timeout=300)
    if verification.get("passed") is not True:
        raise DeploymentError("Remote staged-file verification failed")
    smoke = exec_json(client, render_import_smoke(layout), timeout=300)
    if smoke.get("passed") is not True:
        raise DeploymentError("Remote public-cache import smoke failed")
    return {"files": uploaded, "verification": verification, "import_smoke": smoke, "passed": True}


def read_status(
    plan: FrozenPlan,
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
        status = exec_json(client, render_status(plan, layout), timeout=300)
        verification = None
        if status.get("status") in {"completed", "finalizing"}:
            argv = build_runner_argv(plan, layout, verify_only=True)
            env = _runtime_env(layout, 1)
            source = f"""
import json, os, subprocess
argv=json.loads({json.dumps(json.dumps(argv))}); env=os.environ.copy(); env.update(json.loads({json.dumps(json.dumps(env))}))
result=subprocess.run(argv,env=env,cwd={layout.base!r},capture_output=True,text=True,timeout=900)
payload={{'exit_code':result.returncode,'stdout_tail':result.stdout[-4000:],'stderr_tail':result.stderr[-4000:],'passed':result.returncode==0}}
if result.returncode==0:
    try: payload['verification']=json.loads(result.stdout.strip().splitlines()[-1])
    except Exception as exc: payload.update({{'passed':False,'parse_error':type(exc).__name__}})
print(json.dumps(payload,sort_keys=True))
"""
            verification = exec_json(client, _python_command(source), timeout=1_000)
            if status.get("status") == "completed" and verification.get("passed") is not True:
                status["status"] = "artifact_invalid"
        status["cache_verification"] = verification
    finally:
        if own_client:
            client.close()
    report = {
        **status,
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
) -> dict[str, Any]:
    plan = validate_plan(plan_path)
    layout = build_layout(plan)
    client = connect_job89941(profile_dir)
    try:
        preflight = exec_json(client, render_preflight(plan), timeout=600)
        if preflight.get("passed") is not True:
            raise DeploymentError("Read-only public input preflight failed")
        staging = stage_files(client, plan, layout)
        launch = exec_json(client, render_launch(plan, layout), timeout=120)
        if launch.get("duplicate_processes_started") != 0 or launch.get("status") not in {
            "running", "completed"
        }:
            raise DeploymentError("Public-cache launch failed closed")
        status = read_status(
            plan,
            layout,
            profile_dir=profile_dir,
            evidence_dir=evidence_dir,
            client=client,
        )
    finally:
        client.close()
    report = {
        "schema": DEPLOYMENT_SCHEMA,
        "created_at": now_iso(),
        "job_id": EXPECTED_JOB_ID,
        "run_id": plan.run_id,
        "plan": {"path": str(plan.path), "bytes": plan.bytes, "sha256": plan.sha256},
        "layout": layout.__dict__,
        "preflight": preflight,
        "staging": staging,
        "launch": launch,
        "status": status,
        "process_signals_sent": 0,
        "other_processes_modified": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }
    write_json_atomic(Path(evidence_dir) / "deployment_current.json", report)
    return report


def collect(
    plan_path: Path,
    *,
    profile_dir: Path = DEFAULT_PROFILE_DIR,
    evidence_dir: Path = DEFAULT_EVIDENCE_DIR,
) -> dict[str, Any]:
    plan = validate_plan(plan_path)
    layout = build_layout(plan)
    client = connect_job89941(profile_dir)
    try:
        status = read_status(
            plan,
            layout,
            profile_dir=profile_dir,
            evidence_dir=evidence_dir,
            client=client,
        )
        if status.get("status") != "completed":
            return status
        manifest = status.get("manifest")
        if not isinstance(manifest, dict):
            raise DeploymentError("Completed cache has no manifest record")
        collected_dir = Path(evidence_dir) / "collected" / plan.run_id
        collected_dir.mkdir(parents=True, exist_ok=True)
        downloads = []
        with client.open_sftp() as sftp:
            for name in ("cache_manifest.json", "feature_diagnostics.json", "feature_names.json"):
                remote = posixpath.join(layout.cache, name)
                local = collected_dir / name
                sftp.get(remote, str(local))
                downloads.append({
                    "name": name,
                    "path": str(local.resolve()),
                    "bytes": local.stat().st_size,
                    "sha256": sha256_file(local),
                })
    finally:
        client.close()
    local_manifest = json.loads((collected_dir / "cache_manifest.json").read_text(encoding="utf-8"))
    report = {
        "schema": COLLECTION_SCHEMA,
        "created_at": now_iso(),
        "job_id": EXPECTED_JOB_ID,
        "run_id": plan.run_id,
        "status": "completed_and_verified",
        "remote_status": status,
        "downloads": downloads,
        "cache_manifest": local_manifest,
        "large_arrays_remain_on_shared_hpc_storage": True,
        "process_signals_sent": 0,
        "other_processes_modified": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }
    write_json_atomic(Path(evidence_dir) / "collection_current.json", report)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("deploy", "status", "collect"))
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    plan_path = args.plan or find_default_plan()
    if args.action == "deploy":
        payload = deploy(plan_path, profile_dir=args.profile_dir, evidence_dir=args.evidence_dir)
    elif args.action == "status":
        plan = validate_plan(plan_path)
        payload = read_status(
            plan,
            build_layout(plan),
            profile_dir=args.profile_dir,
            evidence_dir=args.evidence_dir,
        )
    else:
        payload = collect(plan_path, profile_dir=args.profile_dir, evidence_dir=args.evidence_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
