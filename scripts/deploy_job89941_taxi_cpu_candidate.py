#!/usr/bin/env python3
"""Deploy and monitor the frozen PUBLIC_ONLY Taxi CPU candidate on job 89941."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import stat
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.deploy_job89941_may2022_cpu_diagnostic import (
    ALLOWED_GPU_REMOTE_ROOT,
    DEFAULT_PROFILE_DIR,
    connect_job89941,
    exec_json,
    sha256_file,
    sftp_mkdirs_confined,
    sftp_upload_exact,
)

DEFAULT_PLAN = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "taxi_cpu_lightgbm_candidate_s43_job89941_v3_20260728.json"
)
DEFAULT_EVIDENCE_DIR = (
    PROJECT_ROOT / "workspace" / "hpc" / "job89941_taxi_cpu_candidate_s43_v3"
)
REMOTE_UNIFIED_SITE_PACKAGES = (
    f"{ALLOWED_GPU_REMOTE_ROOT}/.deps:"
    f"{ALLOWED_GPU_REMOTE_ROOT}/mlebench_lite_runtime/"
    "unified-py310-sklearn1.7.2/site-packages"
)
PLAN_SCHEMA = "evomind.hpc.job89941_taxi_cpu_candidate_plan.v1"
STATE_SCHEMA = "evomind.hpc.job89941_taxi_cpu_candidate_state.v1"
STATUS_SCHEMA = "evomind.hpc.job89941_taxi_cpu_candidate_status.v1"


class TaxiCpuDeploymentError(RuntimeError):
    """Raised when deployment evidence or a remote contract changes."""


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TaxiCpuDeploymentError(f"JSON object required: {path}")
    return payload


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def confined_remote(path: str) -> str:
    root = PurePosixPath(ALLOWED_GPU_REMOTE_ROOT)
    candidate = PurePosixPath(path)
    if not candidate.is_absolute() or candidate == root:
        raise TaxiCpuDeploymentError("remote path must be below the dedicated root")
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise TaxiCpuDeploymentError("remote path escaped the dedicated root") from exc
    if ".." in candidate.parts:
        raise TaxiCpuDeploymentError("remote path contains traversal")
    return candidate.as_posix()


def validate_plan(plan_path: Path) -> dict[str, Any]:
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    if plan.get("schema") != PLAN_SCHEMA or plan.get("status") != "approved_candidate_only":
        raise TaxiCpuDeploymentError("Taxi CPU plan schema/status changed")
    if plan.get("job_id") != 89941 or plan.get("profile") != "job89941_cpu":
        raise TaxiCpuDeploymentError("Taxi CPU plan target changed")
    boundary = plan.get("boundary") or {}
    expected_boundary = {
        "visibility_mode": "PUBLIC_ONLY",
        "candidate_only": True,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "gpu_used": False,
        "process_signals_allowed": False,
        "other_processes_may_be_modified": False,
    }
    if any(boundary.get(key) != value for key, value in expected_boundary.items()):
        raise TaxiCpuDeploymentError("Taxi CPU plan boundary changed")
    runtime = plan.get("runtime") or {}
    if (
        runtime.get("threads") != 60
        or runtime.get("iterations") != 1400
        or runtime.get("seed") != 43
        or runtime.get("cuda_visible_devices") != ""
        or runtime.get("pythonpath") != REMOTE_UNIFIED_SITE_PACKAGES
    ):
        raise TaxiCpuDeploymentError("Taxi CPU runtime contract changed")
    for key in (
        "remote_run_dir",
        "remote_source",
        "remote_plan",
        "remote_state",
        "remote_log",
        "remote_output_dir",
    ):
        confined_remote(str(plan.get(key) or ""))
    for name in ("candidate_source", "deployment_source"):
        record = plan.get(name) or {}
        path = Path(str(record.get("path") or "")).resolve()
        if not path.is_file():
            raise TaxiCpuDeploymentError(f"Taxi CPU source is missing: {name}")
        if path.stat().st_size != record.get("bytes") or sha256_file(path) != record.get("sha256"):
            raise TaxiCpuDeploymentError(f"Taxi CPU source hash changed: {name}")
    for name in ("base_cache", "route_stat_cache"):
        record = plan.get(name) or {}
        confined_remote(str(record.get("remote_path") or ""))
        local = Path(str(record.get("local_evidence_path") or "")).resolve()
        if not local.is_file() or sha256_file(local) != record.get("sha256"):
            raise TaxiCpuDeploymentError(f"Taxi CPU cache evidence changed: {name}")
    plan["_path"] = str(plan_path)
    plan["_sha256"] = sha256_file(plan_path)
    return plan


def validate_status_plan(plan_path: Path) -> dict[str, Any]:
    """Validate an already-launched run without coupling to a mutable checkout."""
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("status") != "approved_candidate_only"
        or plan.get("job_id") != 89941
        or plan.get("profile") != "job89941_cpu"
    ):
        raise TaxiCpuDeploymentError("Taxi CPU status plan identity changed")
    for key in (
        "remote_run_dir",
        "remote_source",
        "remote_plan",
        "remote_state",
        "remote_log",
        "remote_output_dir",
    ):
        confined_remote(str(plan.get(key) or ""))
    plan["_path"] = str(plan_path)
    plan["_sha256"] = sha256_file(plan_path)
    return plan


def encoded_python(source: str) -> str:
    encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
    return f"python3 -c {shlex.quote('import base64;exec(base64.b64decode(' + repr(encoded) + '))')}"


def remote_preflight(client: Any, plan: Mapping[str, Any]) -> dict[str, Any]:
    source = f'''import hashlib,importlib.util,json,os,pathlib,shutil
def rec(raw):
 p=pathlib.Path(raw); data=p.read_bytes() if p.is_file() else b""; return {{"path":raw,"exists":p.is_file(),"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest() if data else None}}
base=rec({str((plan['base_cache'])['remote_path'])!r})
route=rec({str((plan['route_stat_cache'])['remote_path'])!r})
mods={{m:bool(importlib.util.find_spec(m)) for m in ["numpy","lightgbm"]}}
mem={{}}
for line in open("/proc/meminfo"):
 k,v=line.split(":",1)
 if k in {{"MemTotal","MemAvailable"}}: mem[k]=v.strip()
print(json.dumps({{"base":base,"route":route,"modules":mods,"cpu_count":os.cpu_count(),"memory":mem,"root_writable":os.access({ALLOWED_GPU_REMOTE_ROOT!r},os.W_OK),"nvidia_smi":shutil.which("nvidia-smi") is not None}}))
'''
    command = (
        f"env PYTHONPATH={shlex.quote(REMOTE_UNIFIED_SITE_PACKAGES)} "
        f"{encoded_python(source)}"
    )
    report = exec_json(client, command, timeout=300)
    for name, key in (("base_cache", "base"), ("route_stat_cache", "route")):
        expected = plan[name]
        observed = report[key]
        if (
            observed.get("exists") is not True
            or observed.get("sha256") != expected.get("sha256")
            or observed.get("bytes") != expected.get("bytes")
        ):
            raise TaxiCpuDeploymentError(f"remote {name} manifest changed")
    if report.get("modules") != {"numpy": True, "lightgbm": True}:
        raise TaxiCpuDeploymentError("remote Taxi CPU runtime modules are incomplete")
    if report.get("cpu_count") != 64 or report.get("root_writable") is not True:
        raise TaxiCpuDeploymentError("remote Taxi CPU resource contract changed")
    return report


def upload_sources(client: Any, plan: Mapping[str, Any]) -> dict[str, Any]:
    with client.open_sftp() as sftp:
        for path in (
            str(plan["remote_run_dir"]),
            str(PurePosixPath(str(plan["remote_source"])).parent),
            str(plan["remote_output_dir"]),
            str(PurePosixPath(str(plan["remote_log"])).parent),
            str(PurePosixPath(str(plan["remote_run_dir"])) / "tmp"),
        ):
            sftp_mkdirs_confined(sftp, path)
        uploads = []
        for local_key, remote_key in (
            ("candidate_source", "remote_source"),
            ("_path", "remote_plan"),
        ):
            local_path = Path(str(plan[local_key] if local_key == "_path" else plan[local_key]["path"]))
            expected_bytes = local_path.stat().st_size
            expected_sha = sha256_file(local_path)
            sftp_upload_exact(
                sftp,
                local_path,
                str(plan[remote_key]),
                expected_bytes=expected_bytes,
                expected_sha256=expected_sha,
            )
            uploads.append(
                {"local": str(local_path.resolve()), "remote": str(plan[remote_key]), "bytes": expected_bytes, "sha256": expected_sha}
            )
    return {"uploads": uploads}


def launch(client: Any, plan: Mapping[str, Any]) -> dict[str, Any]:
    argv = [
        "python3",
        str(plan["remote_source"]),
        "--cache-dir",
        str(PurePosixPath(str(plan["base_cache"]["remote_path"])).parent),
        "--route-dir",
        str(PurePosixPath(str(plan["route_stat_cache"]["remote_path"])).parent),
        "--output-dir",
        str(plan["remote_output_dir"]),
        "--seed",
        str(plan["runtime"]["seed"]),
        "--iterations",
        str(plan["runtime"]["iterations"]),
        "--threads",
        str(plan["runtime"]["threads"]),
    ]
    source = f'''import json,os,pathlib,subprocess,time
state=pathlib.Path({str(plan['remote_state'])!r})
payload={{"schema":{STATE_SCHEMA!r},"created_at_epoch":time.time(),"status":"launching","run_id":{str(plan['run_id'])!r},"plan_sha256":{str(plan['_sha256'])!r},"argv":{argv!r},"process_signals_sent":0,"other_processes_modified":False}}
try:
 fd=os.open(state,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
except FileExistsError:
 print(json.dumps({{"status":"claim_exists","state":json.loads(state.read_text())}})); raise SystemExit(0)
with os.fdopen(fd,"w",encoding="utf-8") as h: json.dump(payload,h,sort_keys=True); h.write("\\n")
env=os.environ.copy(); env.update({{"CUDA_VISIBLE_DEVICES":"","OMP_NUM_THREADS":"60","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"60","PYTHONPATH":{REMOTE_UNIFIED_SITE_PACKAGES!r},"TMPDIR":{str(PurePosixPath(str(plan['remote_run_dir'])) / 'tmp')!r}}})
log=open({str(plan['remote_log'])!r},"ab",buffering=0)
p=subprocess.Popen({argv!r},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd={str(plan['remote_run_dir'])!r},env=env,start_new_session=True)
payload.update({{"status":"running","pid":p.pid,"launched_at_epoch":time.time()}})
tmp=state.with_suffix(".tmp"); tmp.write_text(json.dumps(payload,sort_keys=True)+"\\n",encoding="utf-8"); os.replace(tmp,state)
print(json.dumps({{"status":"launched","pid":p.pid,"state":payload}}))
'''
    return exec_json(client, encoded_python(source), timeout=300)


def status(client: Any, plan: Mapping[str, Any]) -> dict[str, Any]:
    result_path = str(PurePosixPath(str(plan["remote_output_dir"])) / "result.json")
    source = f'''import hashlib,json,pathlib
state_path=pathlib.Path({str(plan['remote_state'])!r}); log_path=pathlib.Path({str(plan['remote_log'])!r}); result_path=pathlib.Path({result_path!r})
state=json.loads(state_path.read_text()) if state_path.is_file() else None
pid=int((state or {{}}).get("pid") or 0); proc=pathlib.Path("/proc")/str(pid)
running=bool(pid and proc.is_dir()); cmdline=(proc/"cmdline").read_bytes().replace(b"\\0",b" ").decode("utf-8","replace") if running else ""
result=json.loads(result_path.read_text()) if result_path.is_file() else None
log_tail=""
if log_path.is_file():
 data=log_path.read_bytes(); log_tail=data[-12000:].decode("utf-8","replace")
print(json.dumps({{"schema":{STATUS_SCHEMA!r},"observed_at":__import__('datetime').datetime.now().astimezone().isoformat(),"state":state,"process":{{"pid":pid,"running":running,"cmdline_matches":{str(plan['remote_source'])!r} in cmdline}},"result":result,"log_tail":log_tail,"process_signals_sent":0,"other_processes_modified":False,"official_grader_executed":False,"kaggle_submission_executed":False}}))
'''
    return exec_json(client, encoded_python(source), timeout=300)


def deploy(plan: Mapping[str, Any], evidence_dir: Path) -> dict[str, Any]:
    client = connect_job89941(DEFAULT_PROFILE_DIR)
    try:
        preflight = remote_preflight(client, plan)
        uploads = upload_sources(client, plan)
        launch_report = launch(client, plan)
        observed = status(client, plan)
    finally:
        client.close()
    report = {
        "schema": "evomind.hpc.job89941_taxi_cpu_candidate_deployment.v1",
        "created_at": now_iso(),
        "plan_path": plan["_path"],
        "plan_sha256": plan["_sha256"],
        "preflight": preflight,
        "uploads": uploads,
        "launch": launch_report,
        "status": observed,
        "process_signals_sent": 0,
        "other_processes_modified": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }
    write_json_atomic(evidence_dir / "deployment_current.json", report)
    write_json_atomic(evidence_dir / "status_current.json", observed)
    return report


def collect_status(plan: Mapping[str, Any], evidence_dir: Path) -> dict[str, Any]:
    client = connect_job89941(DEFAULT_PROFILE_DIR)
    try:
        observed = status(client, plan)
    finally:
        client.close()
    write_json_atomic(evidence_dir / "status_current.json", observed)
    return observed


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "deploy", "status"))
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    plan = (
        validate_status_plan(args.plan)
        if args.action == "status"
        else validate_plan(args.plan)
    )
    if args.action == "validate":
        result = {"status": "validated", "plan_path": plan["_path"], "plan_sha256": plan["_sha256"]}
    elif args.action == "deploy":
        result = deploy(plan, args.evidence_dir.resolve())
    else:
        result = collect_status(plan, args.evidence_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
