#!/usr/bin/env python3
"""Run one identity-bound, bounded CUDA driver smoke on the current HPC job.

The command accepts only a job ID and its matching named DPAPI profile.  SSH
endpoints, the allocation role account, proxy credentials, and authentication
material are resolved exclusively by ``gpu_credentials.py``.  A strict
read-only container identity gate runs before the smoke is allowed to create a
small job-scoped directory under the dedicated EvoMind remote root.

The smoke launches one deterministic PTX vector-add kernel, records five
``nvidia-smi`` samples, and collects three files smaller than 1 MiB.  It never
starts training, calls a grader, submits to Kaggle, signals another process, or
writes outside the profile-specific smoke directory.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_agent_workstation.server.core.gpu_credentials import (  # noqa: E402
    ALLOWED_GPU_REMOTE_ROOT,
    STRICT_NAMED_PROFILE_OVERRIDE_KEYS,
    connect_ssh,
    load_gpu_ssh_config,
    verify_job_container_identity,
)


POINTER_SCHEMA = "evomind.hpc.bounded_smoke_pointer.v1"
SAFE_PROFILE = re.compile(r"^job[1-9][0-9]*$")
MAX_EVIDENCE_FILE_BYTES = 1024 * 1024
REMOTE_REQUIRED_FILES = (
    "gpu_smoke.json",
    "nvidia_smi_samples.jsonl",
    "container_identity.json",
)


REMOTE_SOURCE_TEMPLATE = r'''from __future__ import annotations
import array
import ctypes
import ctypes.util
import json
import os
import pathlib
import socket
import subprocess
import time
from datetime import datetime, timezone

ROOT = pathlib.Path("__REMOTE_ROOT__")
PROFILE = "__PROFILE__"
JOB_ID = __JOB_ID__
RUN_ID = os.environ["EVOMIND_SMOKE_RUN_ID"]
RUN_DIR = ROOT / ("evomind_" + PROFILE + "_smoke") / RUN_ID


def atomic_json(path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def check(code, operation):
    if int(code) != 0:
        raise RuntimeError(operation + " failed: cuda_error_" + str(int(code)))


def gpu_sample(index):
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
        check=True,
    )
    rows = []
    for line in completed.stdout.strip().splitlines():
        parts = [value.strip() for value in line.split(",")]
        if len(parts) >= 5:
            rows.append({
                "name": parts[0],
                "uuid": parts[1],
                "memory_total_mib": int(parts[2]),
                "memory_free_mib": int(parts[3]),
                "utilization_gpu_percent": int(parts[4]),
            })
    if len(rows) != 1:
        raise RuntimeError("bounded smoke requires exactly one visible GPU")
    return {
        "sample_index": index,
        "sampled_at": datetime.now(timezone.utc).isoformat(),
        "gpus": rows,
    }


root_real = ROOT.resolve(strict=True)
if str(root_real) != "__REMOTE_ROOT__":
    raise RuntimeError("remote root mismatch")
RUN_DIR.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
RUN_DIR.mkdir(mode=0o700, parents=False, exist_ok=False)
if root_real not in RUN_DIR.resolve().parents:
    raise RuntimeError("smoke path escaped the dedicated root")

samples = []
for sample_index in range(1, 6):
    samples.append(gpu_sample(sample_index))
    time.sleep(0.2)
with (RUN_DIR / "nvidia_smi_samples.jsonl").open(
    "w", encoding="utf-8", newline="\n"
) as handle:
    for item in samples:
        handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
os.chmod(RUN_DIR / "nvidia_smi_samples.jsonl", 0o600)

host_uuid = ""
for candidate in ("/sys/class/dmi/id/product_uuid", "/etc/machine-id"):
    try:
        host_uuid = pathlib.Path(candidate).read_text(encoding="utf-8").strip().lower()
    except OSError:
        continue
    if host_uuid:
        break
identity = {
    "schema": "evomind.hpc.container_identity.v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "job_id": JOB_ID,
    "credential_profile": PROFILE,
    "hostname": socket.gethostname(),
    "host_uuid": host_uuid,
    "gpus": samples[0]["gpus"],
    "root_realpath": str(root_real),
    "run_dir": str(RUN_DIR),
    "signals_sent": 0,
    "other_processes_modified": False,
    "read_only_identity_probe": True,
}
atomic_json(RUN_DIR / "container_identity.json", identity)

started = time.perf_counter()
cuda = ctypes.CDLL(ctypes.util.find_library("cuda") or "libcuda.so.1")
cuda.cuInit.argtypes = [ctypes.c_uint]
cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
cuda.cuCtxCreate_v2.argtypes = [
    ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int,
]
cuda.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
cuda.cuModuleGetFunction.argtypes = [
    ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p,
]
cuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
cuda.cuMemcpyHtoD_v2.argtypes = [
    ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t,
]
cuda.cuMemcpyDtoH_v2.argtypes = [
    ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t,
]
cuda.cuLaunchKernel.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
    ctypes.c_uint, ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
]
cuda.cuMemFree_v2.argtypes = [ctypes.c_uint64]
cuda.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]

check(cuda.cuInit(0), "cuInit")
device = ctypes.c_int()
check(cuda.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
context = ctypes.c_void_p()
check(cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device), "cuCtxCreate")
try:
    ptx = b"""
.version 7.0
.target sm_70
.address_size 64
.visible .entry vadd(
    .param .u64 a,
    .param .u64 b,
    .param .u64 c,
    .param .u32 n
){
    .reg .pred %p;
    .reg .f32 %fa,%fb,%fc;
    .reg .b32 %r<6>;
    .reg .b64 %rd<10>;
    ld.param.u64 %rd1,[a];
    ld.param.u64 %rd2,[b];
    ld.param.u64 %rd3,[c];
    ld.param.u32 %r1,[n];
    mov.u32 %r2,%tid.x;
    mov.u32 %r3,%ctaid.x;
    mov.u32 %r4,%ntid.x;
    mad.lo.s32 %r5,%r3,%r4,%r2;
    setp.ge.u32 %p,%r5,%r1;
    @%p bra DONE;
    mul.wide.u32 %rd4,%r5,4;
    add.s64 %rd5,%rd1,%rd4;
    add.s64 %rd6,%rd2,%rd4;
    add.s64 %rd7,%rd3,%rd4;
    ld.global.f32 %fa,[%rd5];
    ld.global.f32 %fb,[%rd6];
    add.f32 %fc,%fa,%fb;
    st.global.f32 [%rd7],%fc;
DONE:
    ret;
}
"""
    module = ctypes.c_void_p()
    check(
        cuda.cuModuleLoadData(ctypes.byref(module), ctypes.c_char_p(ptx)),
        "cuModuleLoadData",
    )
    function = ctypes.c_void_p()
    check(
        cuda.cuModuleGetFunction(ctypes.byref(function), module, b"vadd"),
        "cuModuleGetFunction",
    )
    vector_length = 4096
    byte_count = vector_length * 4
    host_a = array.array("f", [float(index % 97) for index in range(vector_length)])
    host_b = array.array("f", [float((index * 3) % 101) for index in range(vector_length)])
    host_c = array.array("f", [0.0] * vector_length)
    device_a = ctypes.c_uint64()
    device_b = ctypes.c_uint64()
    device_c = ctypes.c_uint64()
    check(cuda.cuMemAlloc_v2(ctypes.byref(device_a), byte_count), "cuMemAllocA")
    check(cuda.cuMemAlloc_v2(ctypes.byref(device_b), byte_count), "cuMemAllocB")
    check(cuda.cuMemAlloc_v2(ctypes.byref(device_c), byte_count), "cuMemAllocC")
    try:
        check(
            cuda.cuMemcpyHtoD_v2(
                device_a,
                (ctypes.c_float * vector_length).from_buffer(host_a),
                byte_count,
            ),
            "copyA",
        )
        check(
            cuda.cuMemcpyHtoD_v2(
                device_b,
                (ctypes.c_float * vector_length).from_buffer(host_b),
                byte_count,
            ),
            "copyB",
        )
        arg_a = ctypes.c_uint64(device_a.value)
        arg_b = ctypes.c_uint64(device_b.value)
        arg_c = ctypes.c_uint64(device_c.value)
        arg_n = ctypes.c_uint(vector_length)
        parameters = (ctypes.c_void_p * 4)(
            ctypes.cast(ctypes.byref(arg_a), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(arg_b), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(arg_c), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(arg_n), ctypes.c_void_p),
        )
        check(
            cuda.cuLaunchKernel(
                function,
                16, 1, 1,
                256, 1, 1,
                0, None,
                parameters, None,
            ),
            "cuLaunchKernel",
        )
        check(cuda.cuCtxSynchronize(), "cuCtxSynchronize")
        check(
            cuda.cuMemcpyDtoH_v2(
                (ctypes.c_float * vector_length).from_buffer(host_c),
                device_c,
                byte_count,
            ),
            "copyC",
        )
        max_error = max(
            abs(host_c[index] - (host_a[index] + host_b[index]))
            for index in range(vector_length)
        )
        checksum = float(sum(host_c[:128]))
    finally:
        cuda.cuMemFree_v2(device_a)
        cuda.cuMemFree_v2(device_b)
        cuda.cuMemFree_v2(device_c)
    elapsed = time.perf_counter() - started
finally:
    cuda.cuCtxDestroy_v2(context)

children_path = pathlib.Path(
    "/proc/" + str(os.getpid()) + "/task/" + str(os.getpid()) + "/children"
)
children = children_path.read_text().split() if children_path.exists() else []
result = {
    "schema": "evomind.hpc.gpu_smoke.v1",
    "status": "passed" if max_error < 1e-6 and elapsed < 60 and not children else "failed",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "job_id": JOB_ID,
    "credential_profile": PROFILE,
    "device_name": samples[0]["gpus"][0]["name"],
    "gpu_uuid": samples[0]["gpus"][0]["uuid"],
    "driver_api": "libcuda",
    "kernel": "ptx_vector_add",
    "vector_length": vector_length,
    "elapsed_seconds": round(elapsed, 4),
    "max_error": max_error,
    "checksum": checksum,
    "run_dir": str(RUN_DIR),
    "remote_write_boundary_ok": root_real in RUN_DIR.resolve().parents,
    "residual_child_processes": children,
    "residual_running_processes": len(children),
    "background_processes_started": 0,
    "training_started": False,
    "grader_calls": 0,
    "kaggle_submissions": 0,
    "signals_sent": 0,
    "other_processes_modified": False,
}
atomic_json(RUN_DIR / "gpu_smoke.json", result)
print(json.dumps({"run_dir": str(RUN_DIR), "result": result}, separators=(",", ":")))
if result["status"] != "passed":
    raise SystemExit(1)
'''


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_remote_source(*, job_id: int, profile: str) -> str:
    if job_id <= 0 or profile != f"job{job_id}" or not SAFE_PROFILE.fullmatch(profile):
        raise ValueError("job ID and credential profile must have a one-to-one binding")
    return (
        REMOTE_SOURCE_TEMPLATE
        .replace("__REMOTE_ROOT__", ALLOWED_GPU_REMOTE_ROOT)
        .replace("__PROFILE__", profile)
        .replace("__JOB_ID__", str(job_id))
    )


def build_pointer(
    *,
    profile: str,
    job_id: int,
    evidence_root: Path,
    collection_path: Path,
    created_at: str,
) -> dict[str, Any]:
    return {
        "schema": POINTER_SCHEMA,
        "profile": profile,
        "job_id": job_id,
        "created_at": created_at,
        "evidence_root": str(evidence_root.resolve()),
        "collection_path": str(collection_path.resolve()),
        "collection_sha256": _hash_file(collection_path),
        "required_files": {
            name: _hash_file(evidence_root / name) for name in REMOTE_REQUIRED_FILES
        },
        "claim_boundary": (
            "bounded driver-api GPU smoke evidence; no Kaggle submission, "
            "no grader, no long training"
        ),
    }


def _read_channel(stream: Any) -> str:
    value = stream.read()
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def run(args: argparse.Namespace) -> dict[str, Any]:
    job_id = int(args.job_id)
    profile = str(args.credential_profile).strip()
    if job_id <= 0 or profile != f"job{job_id}" or not SAFE_PROFILE.fullmatch(profile):
        raise ValueError("job ID and credential profile must have a one-to-one binding")

    created_at = datetime.now(timezone.utc)
    run_id = f"{profile}-driver-{created_at.strftime('%Y%m%dT%H%M%SZ')}"
    remote_run_dir = (
        f"{ALLOWED_GPU_REMOTE_ROOT.rstrip('/')}/evomind_{profile}_smoke/{run_id}"
    )
    evidence_root = (
        Path(args.evidence_root).expanduser().resolve()
        if args.evidence_root
        else (ROOT / "workspace" / "hpc" / "bounded_smoke" / profile / run_id).resolve()
    )
    evidence_root.mkdir(parents=True, exist_ok=False)
    failure_path = evidence_root / "bounded_smoke_failure.json"
    pointer_path = ROOT / "workspace" / "hpc" / f"{profile}_bounded_smoke_current.json"

    previous_profile = os.environ.get("EVOMIND_HPC_CREDENTIAL_PROFILE")
    previous_overrides: dict[str, str] = {}
    client = None
    try:
        for name in STRICT_NAMED_PROFILE_OVERRIDE_KEYS:
            if name in os.environ:
                previous_overrides[name] = os.environ.pop(name)
        os.environ["EVOMIND_HPC_CREDENTIAL_PROFILE"] = profile
        config = load_gpu_ssh_config(strict_named_profile=True)
        if config.credential_profile != profile or config.job_id != job_id:
            raise RuntimeError("strict credential loader selected a different job binding")
        client = connect_ssh(config, timeout=args.timeout)
        identity = verify_job_container_identity(
            client,
            config,
            expected_job_id=job_id,
        )
        if identity.get("job_container_verified") is not True:
            raise RuntimeError("container identity gate failed before bounded smoke")

        source = build_remote_source(job_id=job_id, profile=profile)
        encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
        launcher = "import base64;exec(base64.b64decode(" + repr(encoded) + "))"
        command = (
            f"EVOMIND_SMOKE_RUN_ID={shlex.quote(run_id)} "
            f"python3 -c {shlex.quote(launcher)}"
        )
        _stdin, stdout, stderr = client.exec_command(command, timeout=args.timeout)
        output = _read_channel(stdout).strip()
        error = _read_channel(stderr).strip()
        exit_code = stdout.channel.recv_exit_status()
        if exit_code != 0 or not output:
            raise RuntimeError(
                f"driver GPU smoke failed closed (exit={exit_code}, "
                f"stderr_tail={error[-500:]!r})"
            )
        try:
            remote_summary = json.loads(output.splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError("driver GPU smoke returned invalid evidence") from exc
        remote_result = remote_summary.get("result")
        if not isinstance(remote_result, dict):
            raise RuntimeError("driver GPU smoke result is missing")
        if (
            remote_result.get("status") != "passed"
            or remote_result.get("remote_write_boundary_ok") is not True
            or remote_result.get("training_started") is not False
            or remote_result.get("signals_sent") != 0
            or remote_result.get("other_processes_modified") is not False
            or str(remote_result.get("run_dir") or "") != remote_run_dir
        ):
            raise RuntimeError("driver GPU smoke evidence violated its bounded contract")

        sftp = client.open_sftp()
        try:
            for name in REMOTE_REQUIRED_FILES:
                remote_path = f"{remote_run_dir}/{name}"
                size = int(sftp.stat(remote_path).st_size)
                if size <= 0 or size > MAX_EVIDENCE_FILE_BYTES:
                    raise RuntimeError(f"bounded smoke artifact size invalid: {name}")
                sftp.get(remote_path, str(evidence_root / name))
        finally:
            sftp.close()

        collection_path = evidence_root / "collection.json"
        collection = {
            "schema": "evomind.hpc.bounded_smoke_collection.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "profile": profile,
            "job_id": job_id,
            "run_id": run_id,
            "remote_run_dir": remote_run_dir,
            "identity_gate": identity,
            "gpu_smoke": remote_result,
            "files": [str((evidence_root / name).resolve()) for name in REMOTE_REQUIRED_FILES],
            "signals_sent": 0,
            "other_processes_modified": False,
            "secrets_emitted": False,
        }
        _atomic_json(collection_path, collection)
        pointer = build_pointer(
            profile=profile,
            job_id=job_id,
            evidence_root=evidence_root,
            collection_path=collection_path,
            created_at=collection["created_at"],
        )
        _atomic_json(pointer_path, pointer)
        return {
            "status": "passed",
            "schema": "evomind.hpc.bounded_smoke_execution.v1",
            "profile": profile,
            "job_id": job_id,
            "run_id": run_id,
            "job_container_verified": True,
            "gpu_name": remote_result.get("device_name"),
            "gpu_memory_total_mib": identity.get("gpu_memory_total_mib"),
            "driver_api": remote_result.get("driver_api"),
            "kernel": remote_result.get("kernel"),
            "max_error": remote_result.get("max_error"),
            "remote_run_dir": remote_run_dir,
            "collection_path": str(collection_path.relative_to(ROOT)).replace("\\", "/"),
            "pointer_path": str(pointer_path.relative_to(ROOT)).replace("\\", "/"),
            "training_started": False,
            "grader_calls": 0,
            "kaggle_submissions": 0,
            "signals_sent": 0,
            "other_processes_modified": False,
            "secrets_emitted": False,
        }
    except Exception as exc:
        _atomic_json(failure_path, {
            "schema": "evomind.hpc.bounded_smoke_failure.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "profile": profile,
            "job_id": job_id,
            "run_id": run_id,
            "error_type": type(exc).__name__,
            "error": str(exc)[:1200],
            "training_started": False,
            "grader_calls": 0,
            "kaggle_submissions": 0,
            "signals_sent": 0,
            "other_processes_modified": False,
            "secrets_emitted": False,
        })
        raise
    finally:
        if client is not None:
            client.close()
        if previous_profile is None:
            os.environ.pop("EVOMIND_HPC_CREDENTIAL_PROFILE", None)
        else:
            os.environ["EVOMIND_HPC_CREDENTIAL_PROFILE"] = previous_profile
        os.environ.update(previous_overrides)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--credential-profile", required=True)
    parser.add_argument("--evidence-root")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    if not 30 <= args.timeout <= 300:
        parser.error("--timeout must be between 30 and 300 seconds")
    try:
        result = run(args)
    except Exception as exc:
        print(json.dumps({
            "status": "failed_closed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:1200],
            "training_started": False,
            "grader_calls": 0,
            "kaggle_submissions": 0,
            "signals_sent": 0,
            "other_processes_modified": False,
            "secrets_emitted": False,
        }, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
