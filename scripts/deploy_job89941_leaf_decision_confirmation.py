#!/usr/bin/env python3
"""Deploy, supervise, and collect the frozen Leaf CPU confirmation on job 89941."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import shlex
import stat
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts import deploy_job89941_may2022_cpu_diagnostic as hpc  # noqa: E402
from scripts import run_leaf_decision_logit_confirmation as runner  # noqa: E402

PLAN_SCHEMA = "evomind.leaf.decision_logit_confirmation_hpc_plan.v1"
STATUS_SCHEMA = "evomind.hpc.job89941_leaf_decision_confirmation_status.v1"
DEPLOYMENT_SCHEMA = "evomind.hpc.job89941_leaf_decision_confirmation_deployment.v1"
COLLECTION_SCHEMA = "evomind.hpc.job89941_leaf_decision_confirmation_collection.v1"
EXPECTED_SOURCE_PATHS = {
    "scripts/run_leaf_decision_logit_confirmation.py",
    "scripts/verify_leaf_decision_logit_confirmation.py",
    "scripts/run_leaf_multibackbone_oof.py",
    "scripts/mlebench_medal_recovery_adapters.py",
    "scripts/mlebench_wave2_adapters.py",
    "scripts/russian_transliteration.py",
    "src/research_os/mlebench_phase_a.py",
}
EXPECTED_INPUT_NAMES = {
    "reference_report",
    "image_manifest",
    "convnext_cache",
    "convnext_metadata",
    "efficientnet_cache",
    "efficientnet_metadata",
}
EXPECTED_PUBLIC_NAMES = {"train.csv", "test.csv", "sample_submission.csv"}
DEFAULT_PLAN = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "leaf_decision_logit_confirmation_job89941_frozen_plan_v3_20260727.json"
)
DEFAULT_EVIDENCE_DIR = (
    PROJECT_ROOT / "workspace" / "hpc" / "job89941_leaf_decision_confirmation"
)


class LeafDeploymentError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrozenPlan:
    path: Path
    payload: dict[str, Any]
    sha256: str
    bytes: int


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def file_record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hpc.sha256_file(path),
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LeafDeploymentError(message)


def _validate_record(
    record: Mapping[str, Any],
    *,
    local_required: bool,
    remote_root: str,
) -> None:
    _require(isinstance(record.get("bytes"), int) and int(record["bytes"]) >= 0, "Invalid byte count")
    digest = str(record.get("sha256") or "")
    _require(bool(hpc.SHA256_PATTERN.fullmatch(digest)), "Invalid SHA-256")
    hpc.ensure_remote_write_path(str(record["remote_path"]), root=remote_root)
    if local_required:
        path = Path(str(record["local_path"])).resolve()
        _require(path.is_file(), f"Frozen local file is missing: {path}")
        _require(path.stat().st_size == record["bytes"], f"Frozen local byte drift: {path}")
        _require(hpc.sha256_file(path) == digest, f"Frozen local hash drift: {path}")


def validate_plan(path: Path = DEFAULT_PLAN) -> FrozenPlan:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    _require(isinstance(payload, dict), "Frozen plan root must be an object")
    exact = {
        "schema": PLAN_SCHEMA,
        "status": "frozen",
        "job_id": 89941,
        "resource_mode": "cpu_only",
        "visibility_mode": "PUBLIC_ONLY",
        "remote_root": hpc.ALLOWED_GPU_REMOTE_ROOT,
    }
    for key, expected in exact.items():
        _require(payload.get(key) == expected, f"Frozen plan field differs: {key}")
    contracts = payload.get("contracts") or {}
    required_contracts = {
        "candidate_only": True,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_allowed": False,
        "writes_confined_to_remote_root": True,
        "new_training_on_local_gpu": False,
    }
    for key, expected in required_contracts.items():
        _require(contracts.get(key) is expected, f"Frozen execution contract differs: {key}")
    _require(payload.get("contract") == runner.fixed_contract(), "Frozen Leaf model contract differs")
    remote_root = str(payload["remote_root"])
    remote_base = hpc.ensure_remote_write_path(str(payload["remote_base"]), root=remote_root)
    for key in (
        "data_root",
        "output_root",
        "run_dir",
        "reference_report",
        "reference_image_manifest",
        "embedding_cache_dir",
    ):
        hpc.ensure_remote_path(str(payload[key]), root=remote_root)
    _require(str(payload["run_dir"]).startswith(remote_base + "/"), "Leaf run directory left frozen base")
    sources = payload.get("sources") or []
    _require(
        {str(value.get("relative_path")) for value in sources} == EXPECTED_SOURCE_PATHS,
        "Frozen Leaf source set differs",
    )
    for record in sources:
        _validate_record(record, local_required=True, remote_root=remote_root)
    inputs = payload.get("inputs") or []
    _require({str(value.get("name")) for value in inputs} == EXPECTED_INPUT_NAMES, "Frozen Leaf input set differs")
    for record in inputs:
        _validate_record(record, local_required=True, remote_root=remote_root)
    public_files = payload.get("public_files") or []
    _require({str(value.get("name")) for value in public_files} == EXPECTED_PUBLIC_NAMES, "Frozen Leaf public-file set differs")
    for record in public_files:
        _validate_record(record, local_required=False, remote_root=remote_root)
    runtime = payload.get("runtime") or {}
    _require(runtime.get("python") == "python3", "Frozen Leaf Python differs")
    _require(runtime.get("cuda_visible_devices") == "", "Leaf confirmation must hide CUDA")
    _require(runtime.get("site_packages") == hpc.REMOTE_UNIFIED_SITE_PACKAGES, "Leaf runtime site differs")
    return FrozenPlan(
        path=path,
        payload=payload,
        sha256=hpc.sha256_file(path),
        bytes=path.stat().st_size,
    )


def staged_records(plan: FrozenPlan) -> list[dict[str, Any]]:
    payload = plan.payload
    plan_record = {
        "local_path": str(plan.path),
        "remote_path": f"{payload['remote_base']}/plans/{plan.path.name}",
        "relative_path": f"plans/{plan.path.name}",
        "bytes": plan.bytes,
        "sha256": plan.sha256,
    }
    return [plan_record, *payload["sources"], *payload["inputs"]]


def runtime_environment(plan: FrozenPlan) -> dict[str, str]:
    payload = plan.payload
    runtime = payload["runtime"]
    return {
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": f"{payload['remote_base']}/frozen_source:{runtime['site_packages']}",
        "TMPDIR": f"{payload['remote_base']}/tmp",
        "OMP_NUM_THREADS": str(runtime["omp_num_threads"]),
        "MKL_NUM_THREADS": str(runtime["mkl_num_threads"]),
        "OPENBLAS_NUM_THREADS": str(runtime["openblas_num_threads"]),
        "NUMEXPR_NUM_THREADS": str(runtime["numexpr_num_threads"]),
    }


def runner_argv(plan: FrozenPlan) -> list[str]:
    p = plan.payload
    return [
        "python3",
        f"{p['remote_base']}/frozen_source/scripts/run_leaf_decision_logit_confirmation.py",
        "--data-root",
        str(p["data_root"]),
        "--reference-report",
        str(p["reference_report"]),
        "--reference-image-manifest",
        str(p["reference_image_manifest"]),
        "--embedding-cache-dir",
        str(p["embedding_cache_dir"]),
        "--output-root",
        str(p["output_root"]),
        "--run-id",
        str(p["run_id"]),
    ]


def verifier_argv(plan: FrozenPlan) -> list[str]:
    p = plan.payload
    return [
        "python3",
        f"{p['remote_base']}/frozen_source/scripts/verify_leaf_decision_logit_confirmation.py",
        "--run-dir",
        str(p["run_dir"]),
        "--data-root",
        str(p["data_root"]),
        "--output",
        f"{p['run_dir']}/independent_verification.json",
    ]


def render_readonly_preflight(plan: FrozenPlan) -> str:
    p = plan.payload
    source = f"""from __future__ import annotations
import hashlib,json,os,pathlib,sys
root=pathlib.Path({p['remote_root']!r}).resolve(strict=True)
site=pathlib.Path({p['runtime']['site_packages']!r}).resolve(strict=True)
sys.path.insert(0,str(site))
import joblib,numpy,pandas,scipy,sklearn
expected={p['public_files']!r}
def digest(path):
 value=hashlib.sha256()
 with path.open('rb') as handle:
  for block in iter(lambda:handle.read(8*1024*1024),b''): value.update(block)
 return value.hexdigest()
checks=[]
for record in expected:
 path=pathlib.Path(record['remote_path']).resolve(strict=True)
 checks.append({{'name':record['name'],'path':str(path),'bytes':path.stat().st_size,'sha256':digest(path),'matches':path.stat().st_size==record['bytes'] and digest(path)==record['sha256'] and path.is_relative_to(root)}})
payload={{'schema':'evomind.hpc.job89941_leaf_readonly_preflight.v1','root_writable':os.access(root,os.W_OK),'cpu_count':os.cpu_count(),'site':str(site),'versions':{{'numpy':numpy.__version__,'pandas':pandas.__version__,'scikit_learn':sklearn.__version__,'scipy':scipy.__version__,'joblib':joblib.__version__}},'public_files':checks}}
payload['passed']=bool(payload['root_writable'] and int(payload['cpu_count'] or 0)>=16 and site.is_relative_to(root) and all(item['matches'] for item in checks))
print(json.dumps(payload,sort_keys=True))
"""
    return hpc._python_command(source)


def render_staged_smoke(plan: FrozenPlan) -> str:
    p = plan.payload
    source = f"""from __future__ import annotations
import json,pathlib,sys
frozen=pathlib.Path({p['remote_base']!r})/'frozen_source'
site=pathlib.Path({p['runtime']['site_packages']!r})
sys.path[:0]=[str(frozen),str(site)]
import pandas as pd
from scripts import run_leaf_decision_logit_confirmation as runner
from scripts import verify_leaf_decision_logit_confirmation as verifier
public=runner.resolve_public_dir(pathlib.Path({p['data_root']!r}))
train=pd.read_csv(public/'train.csv').reset_index(drop=True)
test=pd.read_csv(public/'test.csv').reset_index(drop=True)
reference=runner.read_json(pathlib.Path({p['reference_report']!r}))
manifest_path,manifest=runner.validate_reference_manifest(reference,train,test,override_path=pathlib.Path({p['reference_image_manifest']!r}))
arrays,records=runner.resolve_cache_records(reference,cache_dir=pathlib.Path({p['embedding_cache_dir']!r}),expected_rows=len(train)+len(test))
payload={{'schema':'evomind.hpc.job89941_leaf_staged_smoke.v1','runner_contract':runner.fixed_contract(),'verifier_contract':verifier.fixed_contract(),'manifest_path':str(manifest_path),'manifest_rows':len(manifest),'cache_rows':{{key:len(value) for key,value in arrays.items()}},'cache_records':len(records)}}
payload['passed']=bool(payload['runner_contract']=={p['contract']!r} and payload['verifier_contract']=={p['contract']!r} and payload['manifest_rows']==990 and payload['cache_records']==2 and all(value==990 for value in payload['cache_rows'].values()))
print(json.dumps(payload,sort_keys=True))
"""
    return hpc._python_command(source)


def render_launch(plan: FrozenPlan) -> str:
    p = plan.payload
    state = f"{p['remote_base']}/state/{p['run_id']}.json"
    log = f"{p['remote_base']}/logs/{p['run_id']}.log"
    verification = f"{p['run_dir']}/independent_verification.json"
    run_command = shlex.join(runner_argv(plan))
    verify_command = shlex.join(verifier_argv(plan))
    shell = (
        f"set +e; {run_command} >>{shlex.quote(log)} 2>&1; rc=$?; "
        "if [ \"$rc\" -ne 0 ] && [ \"$rc\" -ne 3 ]; then exit \"$rc\"; fi; "
        f"{verify_command} >>{shlex.quote(log)} 2>&1"
    )
    environment = runtime_environment(plan)
    source = f"""from __future__ import annotations
import json,os,pathlib,subprocess,time
state_path=pathlib.Path({state!r})
verification_path=pathlib.Path({verification!r})
log_path=pathlib.Path({log!r})
remote_base={p['remote_base']!r}
plan_sha256={plan.sha256!r}
environment={environment!r}
shell={shell!r}
def process(pid):
 if not isinstance(pid,int) or pid<=0:return {{'running':False,'owned':False,'pid':pid}}
 proc=pathlib.Path('/proc')/str(pid)
 try: command=(proc/'cmdline').read_bytes().replace(b'\\x00',b' ').decode('utf-8','replace'); fields=(proc/'stat').read_text().split()
 except OSError:return {{'running':False,'owned':False,'pid':pid}}
 owned=remote_base in command
 return {{'running':bool(owned and len(fields)>2 and fields[2]!='Z'),'owned':owned,'pid':pid,'state':fields[2] if len(fields)>2 else None}}
state_path.parent.mkdir(parents=True,exist_ok=True);log_path.parent.mkdir(parents=True,exist_ok=True);pathlib.Path({environment['TMPDIR']!r}).mkdir(parents=True,exist_ok=True)
if state_path.exists():
 state=json.loads(state_path.read_text())
 if state.get('plan_sha256')!=plan_sha256 or state.get('environment')!=environment or state.get('shell')!=shell:raise SystemExit('EXISTING_STATE_CONTRACT_DRIFT')
 status='completed' if verification_path.is_file() else ('running' if process(state.get('pid'))['running'] else 'failed')
 print(json.dumps({{'status':status,'idempotent_reuse':True,'pid':state.get('pid'),'process':process(state.get('pid')),'duplicate_processes_started':0}},sort_keys=True));raise SystemExit(0)
flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL
descriptor=os.open(state_path,flags,0o600)
state={{'schema':'evomind.hpc.job89941_leaf_run_state.v1','created_at_epoch':time.time(),'status':'launch_claimed','run_id':{p['run_id']!r},'plan_sha256':plan_sha256,'environment':environment,'shell':shell,'pid':None,'process_signals_sent':0}}
with os.fdopen(descriptor,'w') as handle:json.dump(state,handle,sort_keys=True);handle.write('\\n');handle.flush();os.fsync(handle.fileno())
child_environment=os.environ.copy();child_environment.update(environment)
with log_path.open('ab',buffering=0) as handle:
 child=subprocess.Popen(['bash','-lc',shell],cwd={p['remote_base']!r}+'/frozen_source',env=child_environment,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
state.update({{'status':'running','pid':child.pid,'launched_at_epoch':time.time()}})
temporary=state_path.with_suffix('.tmp');temporary.write_text(json.dumps(state,sort_keys=True)+'\\n');os.replace(temporary,state_path)
print(json.dumps({{'status':'running','idempotent_reuse':False,'pid':child.pid,'duplicate_processes_started':0,'process_signals_sent':0}},sort_keys=True))
"""
    return hpc._python_command(source)


def render_status(plan: FrozenPlan) -> str:
    p = plan.payload
    state = f"{p['remote_base']}/state/{p['run_id']}.json"
    log = f"{p['remote_base']}/logs/{p['run_id']}.log"
    verification = f"{p['run_dir']}/independent_verification.json"
    source = f"""from __future__ import annotations
import json,pathlib
state_path=pathlib.Path({state!r});log_path=pathlib.Path({log!r});verification_path=pathlib.Path({verification!r});remote_base={p['remote_base']!r}
state=json.loads(state_path.read_text()) if state_path.is_file() else None
process={{'pid':None,'running':False,'owned':False,'state':None}}
if isinstance(state,dict) and isinstance(state.get('pid'),int):
 process['pid']=state['pid'];proc=pathlib.Path('/proc')/str(state['pid'])
 try: command=(proc/'cmdline').read_bytes().replace(b'\\x00',b' ').decode('utf-8','replace');fields=(proc/'stat').read_text().split();process.update({{'owned':remote_base in command,'state':fields[2] if len(fields)>2 else None}});process['running']=bool(process['owned'] and process['state']!='Z')
 except OSError:pass
verification=None
if verification_path.is_file():
 try:verification=json.loads(verification_path.read_text())
 except Exception:verification={{'ok':False,'status':'unreadable'}}
if verification_path.is_file() and isinstance(verification,dict) and verification.get('ok') is True:status='completed'
elif process['running']:status='running'
elif state is None:status='not_deployed'
else:status='failed'
tail=''
if log_path.is_file():
 with log_path.open('rb') as handle:handle.seek(0,2);size=handle.tell();handle.seek(max(0,size-16000));tail=handle.read().decode('utf-8','replace')[-16000:]
print(json.dumps({{'schema':{STATUS_SCHEMA!r},'status':status,'state':state,'process':process,'verification':verification,'verification_path':str(verification_path),'log_path':str(log_path),'log_tail':tail,'process_signals_sent':0}},sort_keys=True))
"""
    return hpc._python_command(source)


def _local_prefix_sha256(path: Path, size: int) -> str:
    digest = hashlib.sha256()
    remaining = int(size)
    with Path(path).open("rb") as handle:
        while remaining:
            block = handle.read(min(1024 * 1024, remaining))
            if not block:
                raise LeafDeploymentError("Local resumable source ended before remote prefix")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def sftp_upload_resumable_exact(
    sftp: Any,
    local_path: Path,
    remote_path: str,
    *,
    expected_bytes: int,
    expected_sha256: str,
    root: str,
) -> str:
    """Resume a large immutable upload using pipelined SFTP writes."""

    local_path = Path(local_path).resolve()
    remote_path = hpc.ensure_remote_write_path(remote_path, root=root)
    if local_path.stat().st_size != expected_bytes or hpc.sha256_file(local_path) != expected_sha256:
        raise LeafDeploymentError("Local resumable source changed before upload")
    hpc.sftp_mkdirs_confined(sftp, posixpath.dirname(remote_path), root=root)
    final_stat = hpc._sftp_lstat(sftp, remote_path)
    if final_stat is not None:
        if stat.S_ISLNK(final_stat.st_mode) or not stat.S_ISREG(final_stat.st_mode):
            raise LeafDeploymentError("Existing resumable destination is unsafe")
        if int(final_stat.st_size) != expected_bytes:
            raise LeafDeploymentError("Existing resumable destination byte count drifted")
        return "reused_size_pending_independent_hash"

    partial = hpc.ensure_remote_write_path(remote_path + ".resume.part", root=root)
    partial_stat = hpc._sftp_lstat(sftp, partial)
    offset = 0
    if partial_stat is not None:
        if stat.S_ISLNK(partial_stat.st_mode) or not stat.S_ISREG(partial_stat.st_mode):
            raise LeafDeploymentError("Resumable partial is unsafe")
        offset = int(partial_stat.st_size)
        if offset > expected_bytes:
            raise LeafDeploymentError("Resumable partial is larger than frozen input")

    mode = "r+b" if offset else "wb"
    with local_path.open("rb") as source, sftp.open(partial, mode) as destination:
        source.seek(offset)
        destination.seek(offset)
        destination.set_pipelined(True)
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            destination.write(block)
        destination.flush()

    completed_stat = sftp.lstat(partial)
    if int(completed_stat.st_size) != expected_bytes:
        raise LeafDeploymentError("Completed resumable upload byte count differs")
    try:
        sftp.rename(partial, remote_path)
    except OSError:
        final_stat = hpc._sftp_lstat(sftp, remote_path)
        if final_stat is None:
            raise
        if int(final_stat.st_size) != expected_bytes:
            raise LeafDeploymentError("Concurrent resumable destination byte count drifted")
        sftp.remove(partial)
        return "reused_concurrent_exact"
    return "uploaded_resumable"


def deploy(plan: FrozenPlan, evidence_dir: Path) -> dict[str, Any]:
    client = hpc.connect_job89941()
    try:
        preflight = hpc.exec_json(client, render_readonly_preflight(plan), timeout=300)
        _require(preflight.get("passed") is True, "Leaf read-only preflight failed")
        records = staged_records(plan)
        actions = []
        with client.open_sftp() as sftp:
            for record in records:
                upload = (
                    sftp_upload_resumable_exact
                    if str(record["remote_path"]).endswith(".npy")
                    else hpc.sftp_upload_exact
                )
                action = upload(
                    sftp,
                    Path(str(record["local_path"])),
                    str(record["remote_path"]),
                    expected_bytes=int(record["bytes"]),
                    expected_sha256=str(record["sha256"]),
                    root=str(plan.payload["remote_root"]),
                )
                actions.append({"path": record["remote_path"], "action": action})
        hashes = hpc.exec_json(
            client,
            hpc.render_remote_hash_command(records, root=str(plan.payload["remote_root"])),
            timeout=300,
        )
        _require(hashes.get("passed") is True, "Leaf staged hashes failed")
        smoke = hpc.exec_json(client, render_staged_smoke(plan), timeout=300)
        _require(smoke.get("passed") is True, "Leaf staged input/import smoke failed")
        launch = hpc.exec_json(client, render_launch(plan), timeout=120)
        _require(launch.get("status") in {"running", "completed"}, "Leaf launch failed")
    finally:
        client.close()
    report = {
        "schema": DEPLOYMENT_SCHEMA,
        "created_at": now_iso(),
        "status": launch["status"],
        "plan": file_record(plan.path),
        "remote_base": plan.payload["remote_base"],
        "preflight": preflight,
        "stage_actions": actions,
        "hash_verification": hashes,
        "staged_smoke": smoke,
        "launch": launch,
        "process_signals_sent": 0,
    }
    hpc.write_json_atomic(evidence_dir / "deployment_current.json", report)
    return report


def status(plan: FrozenPlan, evidence_dir: Path) -> dict[str, Any]:
    client = hpc.connect_job89941()
    try:
        payload = hpc.exec_json(client, render_status(plan), timeout=120)
    finally:
        client.close()
    _require(payload.get("schema") == STATUS_SCHEMA, "Leaf status schema differs")
    _require(payload.get("process_signals_sent") == 0, "Leaf status signal contract differs")
    payload["observed_at"] = now_iso()
    hpc.write_json_atomic(evidence_dir / "status_current.json", payload)
    return payload


def wait(plan: FrozenPlan, evidence_dir: Path, *, poll_seconds: float, timeout_seconds: float) -> dict[str, Any]:
    started = time.monotonic()
    while True:
        current = status(plan, evidence_dir)
        print(json.dumps({"status": current["status"], "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
        if current["status"] in {"completed", "failed", "not_deployed"}:
            return current
        if time.monotonic() - started > timeout_seconds:
            raise LeafDeploymentError("Leaf confirmation wait timed out")
        time.sleep(poll_seconds)


def _remote_snapshot(client: Any, plan: FrozenPlan) -> list[dict[str, Any]]:
    base = str(plan.payload["remote_base"])
    source = f"""from __future__ import annotations
import hashlib,json,pathlib
base=pathlib.Path({base!r}).resolve(strict=True)
def digest(path):
 value=hashlib.sha256()
 with path.open('rb') as handle:
  for block in iter(lambda:handle.read(8*1024*1024),b''):value.update(block)
 return value.hexdigest()
items=[]
for path in sorted(base.rglob('*')):
 if path.is_symlink():raise SystemExit('SYMLINK_IN_LEAF_REMOTE_BASE')
 if path.is_file():items.append({{'relative_path':path.relative_to(base).as_posix(),'bytes':path.stat().st_size,'sha256':digest(path)}})
print(json.dumps({{'items':items,'passed':bool(items)}},sort_keys=True))
"""
    payload = hpc.exec_json(client, hpc._python_command(source), timeout=600)
    _require(payload.get("passed") is True, "Leaf remote snapshot failed")
    return list(payload["items"])


def _collect_remote_file_resumable(
    *,
    plan: FrozenPlan,
    record: Mapping[str, Any],
    destination: Path,
    max_attempts: int = 12,
    chunk_bytes: int = 1024 * 1024,
) -> dict[str, Any]:
    """Download one immutable artifact with reconnect-and-resume semantics."""

    _require(max_attempts >= 1, "Leaf collection attempts must be positive")
    _require(chunk_bytes >= 64 * 1024, "Leaf collection chunk is too small")
    relative = PurePosixPath(str(record["relative_path"]))
    _require(
        not relative.is_absolute() and ".." not in relative.parts,
        "Unsafe Leaf snapshot path",
    )
    expected_bytes = int(record["bytes"])
    expected_sha256 = str(record["sha256"])
    remote = posixpath.join(
        str(plan.payload["remote_base"]), relative.as_posix()
    )
    local = destination.joinpath(*relative.parts)
    local.parent.mkdir(parents=True, exist_ok=True)
    _require(not local.is_symlink(), "Unsafe Leaf local artifact symlink")
    if local.is_file():
        _require(
            local.stat().st_size == expected_bytes
            and hpc.sha256_file(local) == expected_sha256,
            f"Existing collected Leaf artifact drift: {relative.as_posix()}",
        )
        return {
            "relative_path": relative.as_posix(),
            "status": "reused_verified",
            "attempts": 0,
            "resumed_from_bytes": expected_bytes,
        }
    _require(not local.exists(), "Unsafe non-file Leaf collection target")

    temporary = local.with_suffix(local.suffix + ".part")
    _require(not temporary.is_symlink(), "Unsafe Leaf partial artifact symlink")
    if temporary.exists():
        _require(temporary.is_file(), "Unsafe non-file Leaf partial artifact")
        _require(
            temporary.stat().st_size <= expected_bytes,
            f"Leaf partial artifact exceeds frozen size: {relative.as_posix()}",
        )
    initial_offset = temporary.stat().st_size if temporary.exists() else 0
    attempts = 0
    last_error: Exception | None = None
    while (temporary.stat().st_size if temporary.exists() else 0) < expected_bytes:
        attempts += 1
        if attempts > max_attempts:
            raise LeafDeploymentError(
                f"Leaf collection reconnect budget exhausted for {relative.as_posix()}"
            ) from last_error
        client = None
        try:
            client = hpc.connect_job89941()
            with client.open_sftp() as sftp:
                item_stat = sftp.lstat(remote)
                _require(
                    stat.S_ISREG(item_stat.st_mode)
                    and not stat.S_ISLNK(item_stat.st_mode),
                    "Unsafe Leaf remote artifact",
                )
                _require(
                    int(item_stat.st_size) == expected_bytes,
                    f"Leaf remote artifact size drift: {relative.as_posix()}",
                )
                offset = temporary.stat().st_size if temporary.exists() else 0
                with sftp.open(remote, "rb") as source, temporary.open("ab") as sink:
                    source.seek(offset)
                    remaining = expected_bytes - offset
                    while remaining:
                        block = source.read(min(chunk_bytes, remaining))
                        if not block:
                            raise EOFError(
                                f"Unexpected EOF while collecting {relative.as_posix()}"
                            )
                        sink.write(block)
                        remaining -= len(block)
                    sink.flush()
                    os.fsync(sink.fileno())
        except Exception as exc:  # reconnect after transport and SFTP failures
            last_error = exc
            if attempts >= max_attempts:
                raise LeafDeploymentError(
                    f"Leaf collection failed after {attempts} attempts: "
                    f"{relative.as_posix()}"
                ) from exc
            time.sleep(min(2.0**min(attempts, 3), 8.0))
        finally:
            if client is not None:
                client.close()

    _require(
        temporary.stat().st_size == expected_bytes,
        "Collected Leaf byte count differs",
    )
    _require(
        hpc.sha256_file(temporary) == expected_sha256,
        "Collected Leaf hash differs",
    )
    os.replace(temporary, local)
    return {
        "relative_path": relative.as_posix(),
        "status": "downloaded_verified",
        "attempts": attempts,
        "resumed_from_bytes": initial_offset,
    }


def collect(plan: FrozenPlan, evidence_dir: Path) -> dict[str, Any]:
    current = status(plan, evidence_dir)
    _require(current["status"] == "completed", "Leaf collection requires completed verification")
    destination = evidence_dir / "collected" / str(plan.payload["run_id"])
    destination.mkdir(parents=True, exist_ok=True)
    client = hpc.connect_job89941()
    try:
        snapshot = _remote_snapshot(client, plan)
    finally:
        client.close()
    transfer_records = [
        _collect_remote_file_resumable(
            plan=plan,
            record=record,
            destination=destination,
        )
        for record in snapshot
    ]
    verification = current.get("verification") or {}
    report = {
        "schema": COLLECTION_SCHEMA,
        "created_at": now_iso(),
        "status": "collected",
        "run_id": plan.payload["run_id"],
        "destination": str(destination.resolve()),
        "file_count": len(snapshot),
        "files": snapshot,
        "transfers": transfer_records,
        "reused_verified_files": sum(
            item["status"] == "reused_verified" for item in transfer_records
        ),
        "downloaded_verified_files": sum(
            item["status"] == "downloaded_verified" for item in transfer_records
        ),
        "candidate_ready": verification.get("candidate_ready"),
        "promotion_gate": verification.get("promotion_gate"),
        "ensemble_log_loss": verification.get("ensemble_log_loss"),
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
    }
    hpc.write_json_atomic(evidence_dir / "collection_current.json", report)
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    value.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    sub = value.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    deploy_parser = sub.add_parser("deploy")
    deploy_parser.add_argument("--wait", action="store_true")
    deploy_parser.add_argument("--collect", action="store_true")
    wait_parser = sub.add_parser("wait")
    wait_parser.add_argument("--collect", action="store_true")
    sub.add_parser("status")
    sub.add_parser("collect")
    for item in (deploy_parser, wait_parser):
        item.add_argument("--poll-seconds", type=float, default=15.0)
        item.add_argument("--timeout-seconds", type=float, default=3600.0)
    return value


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    evidence_dir = Path(args.evidence_dir).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    try:
        plan = validate_plan(args.plan)
        if args.command == "validate":
            result = {"status": "passed", "plan": file_record(plan.path), "run_id": plan.payload["run_id"], "sources": len(plan.payload["sources"]), "inputs": len(plan.payload["inputs"])}
        elif args.command == "deploy":
            result = deploy(plan, evidence_dir)
            if args.wait:
                result = wait(plan, evidence_dir, poll_seconds=args.poll_seconds, timeout_seconds=args.timeout_seconds)
                if args.collect:
                    _require(result["status"] == "completed", "Leaf deploy --collect requires completion")
                    result = collect(plan, evidence_dir)
        elif args.command == "status":
            result = status(plan, evidence_dir)
        elif args.command == "wait":
            result = wait(plan, evidence_dir, poll_seconds=args.poll_seconds, timeout_seconds=args.timeout_seconds)
            if args.collect:
                _require(result["status"] == "completed", "Leaf wait --collect requires completion")
                result = collect(plan, evidence_dir)
        else:
            result = collect(plan, evidence_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        failure = {"schema": "evomind.hpc.job89941_leaf_decision_confirmation_failure.v1", "created_at": now_iso(), "command": args.command, "error_type": type(exc).__name__, "error": str(exc), "process_signals_sent": 0, "passed": False}
        hpc.write_json_atomic(evidence_dir / "failure_current.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
