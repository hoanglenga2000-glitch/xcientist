#!/usr/bin/env python3
"""Build and verify the isolated, content-addressed SIIM HPC runtime."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import posixpath
import shlex
import shutil
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for entry in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from research_agent_workstation.server.core.gpu_credentials import (  # noqa: E402
    ALLOWED_GPU_REMOTE_ROOT,
    connect_ssh,
    load_gpu_ssh_config,
    verify_job_container_identity,
)
from research_os.siim_hpc_binding import binding_from_environment  # noqa: E402

BINDING = binding_from_environment()
HPC_JOB_ID = BINDING.job_id
CREDENTIAL_PROFILE = BINDING.credential_profile
JOB_TAG = BINDING.job_tag
DEFAULT_REQUIREMENTS = PROJECT_ROOT / "workspace" / f"siim_{JOB_TAG}" / "runtime-requirements.txt"
LEGACY_REQUIREMENTS = PROJECT_ROOT / "workspace" / "siim_job89508" / "runtime-requirements.txt"
REMOTE_RUNTIME_ROOT = f"{ALLOWED_GPU_REMOTE_ROOT}/siim_{JOB_TAG}/runtime"
REMOTE_BASE_VENV = f"{ALLOWED_GPU_REMOTE_ROOT}/mlebench_lite_runtime/venv-uv"
LOCAL_RUNTIME_ROOT = PROJECT_ROOT / "workspace" / "hpc" / f"{JOB_TAG}_siim_runtime"
LOCAL_INSTALLER_ROOT = PROJECT_ROOT / "workspace" / f"siim_{JOB_TAG}"


class RuntimeErrorContract(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_requirements(path: Path) -> Path:
    """Create a job-scoped immutable copy of the shared requirements when needed."""

    selected = Path(path).resolve()
    if selected.is_file():
        return selected
    if selected != DEFAULT_REQUIREMENTS.resolve() or not LEGACY_REQUIREMENTS.is_file():
        raise FileNotFoundError(selected)
    selected.parent.mkdir(parents=True, exist_ok=True)
    temporary = selected.with_name(f".{selected.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(LEGACY_REQUIREMENTS, temporary)
        os.replace(temporary, selected)
    finally:
        temporary.unlink(missing_ok=True)
    if sha256_file(selected) != sha256_file(LEGACY_REQUIREMENTS):
        raise RuntimeErrorContract("job-scoped requirements copy changed")
    return selected


def connect_bound(*, timeout: int = 25) -> tuple[Any, Any]:
    config = load_gpu_ssh_config(strict_named_profile=True)
    if config.credential_profile != CREDENTIAL_PROFILE:
        raise RuntimeErrorContract(
            f"isolated SIIM runtime requires the {CREDENTIAL_PROFILE} credential profile"
        )
    client = connect_ssh(config, timeout=timeout)
    try:
        identity = verify_job_container_identity(
            client,
            config,
            expected_job_id=HPC_JOB_ID,
            expected_root=ALLOWED_GPU_REMOTE_ROOT,
            expected_gpu_name_fragment="A800",
            minimum_gpu_memory_mib=80_000,
        )
    except Exception:
        client.close()
        raise
    setattr(client, "_evomind_container_identity", identity)
    return client, config


def ensure_remote(path: str) -> str:
    candidate = PurePosixPath(path)
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeErrorContract("remote runtime path escaped the dedicated root")
    normalized = posixpath.normpath(candidate.as_posix())
    root = posixpath.normpath(ALLOWED_GPU_REMOTE_ROOT)
    if normalized != root and not normalized.startswith(root + "/"):
        raise RuntimeErrorContract("remote runtime path escaped the dedicated root")
    return normalized


def runtime_paths(requirements_sha256: str) -> dict[str, str]:
    if len(requirements_sha256) != 64 or any(char not in "0123456789abcdef" for char in requirements_sha256):
        raise RuntimeErrorContract("invalid requirements SHA-256")
    root = ensure_remote(f"{REMOTE_RUNTIME_ROOT}/{requirements_sha256}")
    return {
        "root": root,
        "venv": ensure_remote(f"{root}/venv"),
        "requirements": ensure_remote(f"{root}/requirements.txt"),
        "installer": ensure_remote(f"{root}/installer.py"),
        "state": ensure_remote(f"{root}/state.json"),
        "log": ensure_remote(f"{root}/install.log"),
        "verification": ensure_remote(f"{root}/verification.json"),
    }


def verify_remote_path_chain(sftp: Any, path: str) -> str:
    normalized = ensure_remote(path)
    root = PurePosixPath(ALLOWED_GPU_REMOTE_ROOT)
    candidate = PurePosixPath(normalized)
    current = root
    deepest = root
    for part in candidate.relative_to(root).parts:
        current /= part
        try:
            metadata = sftp.lstat(current.as_posix())
        except OSError:
            break
        deepest = current
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeErrorContract("remote runtime path contains a symbolic link")
        if current != candidate and not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeErrorContract("remote runtime parent is not a directory")
    if ensure_remote(str(sftp.normalize(deepest.as_posix()))) != deepest.as_posix():
        raise RuntimeErrorContract("remote runtime path realpath changed")
    return normalized


def _run(client: Any, command: str, *, timeout: int = 120) -> tuple[int, str, str]:
    _stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    output = stdout.read().decode("utf-8", "replace")
    error = stderr.read().decode("utf-8", "replace")
    return stdout.channel.recv_exit_status(), output, error


def _python_command(source: str) -> str:
    encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
    return f"python3 -c {shlex.quote(f'import base64;exec(base64.b64decode({encoded!r}))')}"


def _installer_source(paths: dict[str, str], requirements_sha256: str) -> str:
    return f'''from __future__ import annotations
import hashlib, json, os, pathlib, shutil, subprocess, time, traceback
ROOT = pathlib.Path({paths["root"]!r})
VENV = pathlib.Path({paths["venv"]!r})
REQUIREMENTS = pathlib.Path({paths["requirements"]!r})
STATE = pathlib.Path({paths["state"]!r})
LOG = pathlib.Path({paths["log"]!r})
BASE = pathlib.Path({REMOTE_BASE_VENV!r})
EXPECTED = {requirements_sha256!r}
ENV_ROOT = ROOT / "environment"
TEMP_ROOT = ROOT / "tmp"
CACHE_ROOT = ENV_ROOT / "cache"
ENVIRONMENT = {{
    "HOME": ENV_ROOT / "home",
    "TMPDIR": TEMP_ROOT,
    "TEMP": TEMP_ROOT,
    "TMP": TEMP_ROOT,
    "XDG_CACHE_HOME": CACHE_ROOT / "xdg",
    "HF_HOME": CACHE_ROOT / "huggingface",
    "TORCH_HOME": pathlib.Path({ALLOWED_GPU_REMOTE_ROOT!r}) / "mlebench_model_cache" / "torch",
    "PIP_CACHE_DIR": CACHE_ROOT / "pip",
    "MPLCONFIGDIR": CACHE_ROOT / "matplotlib",
    "NUMBA_CACHE_DIR": CACHE_ROOT / "numba",
    "CUDA_CACHE_PATH": CACHE_ROOT / "cuda",
    "TRITON_CACHE_DIR": CACHE_ROOT / "triton",
    "CUPY_CACHE_DIR": CACHE_ROOT / "cupy",
    "JOBLIB_TEMP_FOLDER": TEMP_ROOT,
    "PYTHONPYCACHEPREFIX": CACHE_ROOT / "python",
    "KAGGLE_CONFIG_DIR": ENV_ROOT / "kaggle_disabled",
}}
for path in dict.fromkeys(ENVIRONMENT.values()):
    path.mkdir(parents=True, exist_ok=True)
os.environ.update({{name:str(path) for name,path in ENVIRONMENT.items()}})
os.environ.pop("KAGGLE_USERNAME",None);os.environ.pop("KAGGLE_KEY",None)
def write_state(payload):
    payload.update({{"schema":"evomind.siim.isolated_runtime_state.v1","job_id":{HPC_JOB_ID},"credential_profile":{CREDENTIAL_PROFILE!r},"updated_at":time.time(),"signals_sent":0,"other_processes_modified":False}})
    temp=STATE.with_name("."+STATE.name+".tmp")
    temp.write_text(json.dumps(payload,indent=2)+"\\n",encoding="utf-8")
    os.replace(temp,STATE)
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()
try:
    os.nice(10)
except OSError:
    pass
try:
    if digest(REQUIREMENTS) != EXPECTED:
        raise RuntimeError("requirements hash changed")
    write_state({{"status":"copying_base_runtime","pid":os.getpid(),"requirements_sha256":EXPECTED}})
    if not (VENV / "bin" / "python").exists():
        shutil.copytree(BASE,VENV,symlinks=True,dirs_exist_ok=False)
    write_state({{"status":"installing","pid":os.getpid(),"requirements_sha256":EXPECTED}})
    environment=os.environ.copy();environment["PYTHONNOUSERSITE"]="1"
    command=[str(VENV/"bin"/"python"),"-m","pip","install","--disable-pip-version-check","--upgrade","--requirement",str(REQUIREMENTS)]
    with LOG.open("ab",buffering=0) as log:
        completed=subprocess.run(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=False)
    if completed.returncode:
        raise RuntimeError("pip install returned "+str(completed.returncode))
    write_state({{"status":"installed","pid":os.getpid(),"requirements_sha256":EXPECTED,"exit_code":0}})
except Exception as exc:
    with LOG.open("a",encoding="utf-8") as log:
        traceback.print_exc(file=log)
    write_state({{"status":"failed","pid":os.getpid(),"requirements_sha256":EXPECTED,"exit_code":1,"error_type":type(exc).__name__}})
    raise
'''


def start(requirements: Path) -> dict[str, Any]:
    requirements = resolve_requirements(requirements)
    requirements_sha256 = sha256_file(requirements)
    paths = runtime_paths(requirements_sha256)
    client, config = connect_bound()
    container_identity = dict(
        getattr(client, "_evomind_container_identity", {}) or {}
    )
    try:
        with client.open_sftp() as sftp:
            verify_remote_path_chain(sftp, paths["root"])
            try:
                sftp.stat(paths["root"])
            except OSError:
                code, output, error = _run(
                    client,
                    f"umask 077; mkdir -p -m 700 {shlex.quote(paths['root'])}",
                    timeout=30,
                )
                if code:
                    raise RuntimeErrorContract(f"remote runtime directory creation failed: {error[-300:] or output[-300:]}")
            verify_remote_path_chain(sftp, paths["root"])
            part = f"{paths['requirements']}.part.{os.getpid()}"
            sftp.put(str(requirements), part)
            sftp.chmod(part, 0o600)
            try:
                sftp.remove(paths["requirements"])
            except OSError:
                pass
            sftp.rename(part, paths["requirements"])
            installer_source = _installer_source(paths, requirements_sha256)
            LOCAL_INSTALLER_ROOT.mkdir(parents=True, exist_ok=True)
            installer_local = LOCAL_INSTALLER_ROOT / f"installer-{requirements_sha256}.py"
            installer_local.write_text(installer_source, encoding="utf-8", newline="\n")
            part = f"{paths['installer']}.part.{os.getpid()}"
            sftp.put(str(installer_local), part)
            sftp.chmod(part, 0o600)
            try:
                sftp.remove(paths["installer"])
            except OSError:
                pass
            sftp.rename(part, paths["installer"])
        launcher = f'''from __future__ import annotations
import json, os, pathlib, subprocess, time
state_path=pathlib.Path({paths["state"]!r})
root=pathlib.Path({paths["root"]!r});env_root=root/"environment";temp_root=root/"tmp";cache_root=env_root/"cache"
environment_paths={{"HOME":env_root/"home","TMPDIR":temp_root,"TEMP":temp_root,"TMP":temp_root,"XDG_CACHE_HOME":cache_root/"xdg","HF_HOME":cache_root/"huggingface","TORCH_HOME":pathlib.Path({ALLOWED_GPU_REMOTE_ROOT!r})/"mlebench_model_cache"/"torch","PIP_CACHE_DIR":cache_root/"pip","MPLCONFIGDIR":cache_root/"matplotlib","NUMBA_CACHE_DIR":cache_root/"numba","CUDA_CACHE_PATH":cache_root/"cuda","TRITON_CACHE_DIR":cache_root/"triton","CUPY_CACHE_DIR":cache_root/"cupy","JOBLIB_TEMP_FOLDER":temp_root,"PYTHONPYCACHEPREFIX":cache_root/"python","KAGGLE_CONFIG_DIR":env_root/"kaggle_disabled"}}
for path in dict.fromkeys(environment_paths.values()):path.mkdir(parents=True,exist_ok=True)
environment=os.environ.copy();environment.update({{name:str(path) for name,path in environment_paths.items()}});environment.pop("KAGGLE_USERNAME",None);environment.pop("KAGGLE_KEY",None);environment["PYTHONNOUSERSITE"]="1"
if state_path.is_file():
    state=json.loads(state_path.read_text(encoding="utf-8"))
    pid=int(state.get("pid") or -1)
    proc=pathlib.Path("/proc")/str(pid)
    cmdline=(proc/"cmdline").read_bytes().replace(b"\\0",b" ").decode("utf-8","replace") if (proc/"cmdline").is_file() else ""
    if state.get("job_id") not in {{None,{HPC_JOB_ID}}} or state.get("credential_profile") not in {{None,{CREDENTIAL_PROFILE!r}}}:
        raise RuntimeError("runtime allocation binding changed")
    if state.get("status") in {{"copying_base_runtime","installing"}} and {paths["installer"]!r} in cmdline:
        state["idempotent_reuse"]=True;print(json.dumps(state));raise SystemExit(0)
log=open({paths["log"]!r},"ab",buffering=0)
process=subprocess.Popen(["python3",{paths["installer"]!r}],cwd={paths["root"]!r},env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
payload={{"schema":"evomind.siim.isolated_runtime_launch.v1","status":"installing","pid":process.pid,"job_id":{HPC_JOB_ID},"credential_profile":{CREDENTIAL_PROFILE!r},"requirements_sha256":{requirements_sha256!r},"root":{paths["root"]!r},"created_at":time.time(),"signals_sent":0,"other_processes_modified":False}}
state_path.write_text(json.dumps(payload,indent=2)+"\\n",encoding="utf-8")
print(json.dumps(payload))
'''
        code, output, error = _run(client, _python_command(launcher), timeout=60)
        if code:
            raise RuntimeErrorContract(f"remote runtime launch failed: {error[-300:] or output[-300:]}")
        remote = json.loads(output.strip().splitlines()[-1])
    finally:
        client.close()
    return {
        "schema": "evomind.siim.isolated_runtime_launch.v1",
        "created_at": utc_now(),
        "credential_profile": config.credential_profile,
        "job_id": HPC_JOB_ID,
        "requirements": str(requirements),
        "requirements_sha256": requirements_sha256,
        "paths": paths,
        "remote": remote,
        "container_identity_gate": container_identity,
        "signals_sent": 0,
        "other_processes_modified": False,
    }


def status(requirements: Path) -> dict[str, Any]:
    requirements = resolve_requirements(requirements)
    requirements_sha256 = sha256_file(requirements)
    paths = runtime_paths(requirements_sha256)
    source = f'''from __future__ import annotations
import json, pathlib
state_path=pathlib.Path({paths["state"]!r});log_path=pathlib.Path({paths["log"]!r})
state=json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {{"status":"missing"}}
pid=int(state.get("pid") or -1);status_path=pathlib.Path("/proc")/str(pid)/"status"
process={{"exists":status_path.is_file(),"state_code":""}}
if status_path.is_file():
    line=next((line for line in status_path.read_text(encoding="utf-8",errors="replace").splitlines() if line.startswith("State:")),"")
    process["state_code"]=line.partition(":")[2].strip()[:1]
tail=""
if log_path.is_file():
    with log_path.open("rb") as f:f.seek(0,2);size=f.tell();f.seek(max(0,size-8000));tail=f.read().decode("utf-8","replace")
print(json.dumps({{"state":state,"process":process,"log_tail":tail,"signals_sent":0,"other_processes_modified":False}}))
'''
    client, _config = connect_bound()
    try:
        code, output, error = _run(client, _python_command(source), timeout=60)
    finally:
        client.close()
    if code:
        raise RuntimeErrorContract(f"remote runtime status failed: {error[-300:] or output[-300:]}")
    remote = json.loads(output.strip().splitlines()[-1])
    state = remote.get("state") if isinstance(remote.get("state"), dict) else {}
    if state.get("status") != "missing":
        if int(state.get("job_id") or 0) != HPC_JOB_ID:
            raise RuntimeErrorContract("remote runtime state job binding changed")
        if state.get("credential_profile") != CREDENTIAL_PROFILE:
            raise RuntimeErrorContract("remote runtime state credential profile changed")
    return {
        "job_id": HPC_JOB_ID,
        "credential_profile": CREDENTIAL_PROFILE,
        "requirements_sha256": requirements_sha256,
        "paths": paths,
        **remote,
    }


def verify(requirements: Path) -> dict[str, Any]:
    requirements = resolve_requirements(requirements)
    requirements_sha256 = sha256_file(requirements)
    paths = runtime_paths(requirements_sha256)
    probe = f'''from __future__ import annotations
import json, pathlib, sys
import numpy, pandas, scipy, sklearn, PIL, torch, torchvision, catboost
from catboost import CatBoostClassifier
if numpy.__version__ != "1.26.4" or pandas.__version__ != "2.2.3":
    raise SystemExit("NUMERIC_RUNTIME_VERSION_DRIFT")
if torch.__version__ != "2.5.1+cu118" or torchvision.__version__ != "0.20.1+cu118":
    raise SystemExit("TORCH_RUNTIME_VERSION_DRIFT")
if not torch.cuda.is_available() or "A800" not in torch.cuda.get_device_name(0):
    raise SystemExit("A800_CUDA_UNAVAILABLE")
model=CatBoostClassifier(iterations=2,depth=2,verbose=False,allow_writing_files=False,task_type="CPU")
model.fit([[0.0],[1.0],[2.0],[3.0]],[0,0,1,1])
payload={{"schema":"evomind.siim.isolated_runtime_verification.v1","status":"passed","job_id":{HPC_JOB_ID},"credential_profile":{CREDENTIAL_PROFILE!r},"python":sys.version.split()[0],"versions":{{"numpy":numpy.__version__,"pandas":pandas.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__,"pillow":PIL.__version__,"torch":torch.__version__,"torchvision":torchvision.__version__,"catboost":catboost.__version__}},"cuda":{{"available":True,"device":torch.cuda.get_device_name(0),"bf16":torch.cuda.is_bf16_supported()}},"catboost_smoke":True,"requirements_sha256":{requirements_sha256!r},"signals_sent":0,"other_processes_modified":False}}
path=pathlib.Path({paths["verification"]!r});temp=path.with_name("."+path.name+".tmp");temp.write_text(json.dumps(payload,indent=2)+"\\n",encoding="utf-8");temp.replace(path)
print(json.dumps(payload))
'''
    python = f"{paths['venv']}/bin/python"
    command = f"PYTHONNOUSERSITE=1 {shlex.quote(python)} -c {shlex.quote(probe)}"
    client, _config = connect_bound()
    try:
        code, output, error = _run(client, command, timeout=300)
    finally:
        client.close()
    if code:
        raise RuntimeErrorContract(f"isolated runtime verification failed: {error[-500:] or output[-500:]}")
    payload = json.loads(output.strip().splitlines()[-1])
    if payload.get("status") != "passed":
        raise RuntimeErrorContract("isolated runtime verification contract failed")
    if int(payload.get("job_id") or 0) != HPC_JOB_ID:
        raise RuntimeErrorContract("isolated runtime verification job binding changed")
    if payload.get("credential_profile") != CREDENTIAL_PROFILE:
        raise RuntimeErrorContract("isolated runtime verification credential profile changed")
    return {"requirements_sha256": requirements_sha256, "paths": paths, **payload}


def _write_local_evidence(name: str, payload: dict[str, Any]) -> Path:
    LOCAL_RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    target = LOCAL_RUNTIME_ROOT / name
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "status", "verify"))
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = (
            start(args.requirements)
            if args.command == "start"
            else status(args.requirements)
            if args.command == "status"
            else verify(args.requirements)
        )
        evidence = _write_local_evidence(f"{args.command}_current.json", payload)
        print(json.dumps({"evidence": str(evidence.resolve()), **payload}, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeErrorContract, json.JSONDecodeError) as exc:
        payload = {
            "schema": "evomind.siim.isolated_runtime_failure.v1",
            "created_at": utc_now(),
            "command": args.command,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "signals_sent": 0,
            "other_processes_modified": False,
        }
        evidence = _write_local_evidence(f"{args.command}_failure_current.json", payload)
        print(json.dumps({"evidence": str(evidence.resolve()), **payload}, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
