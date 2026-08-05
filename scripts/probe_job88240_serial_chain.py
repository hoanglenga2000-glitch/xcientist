#!/usr/bin/env python3
"""Capture the live, read-only job88240 serial chain and GPU occupancy."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import mlebench_remote_ops as ops  # noqa: E402

DEFAULT_OUTPUT = (
    PROJECT_ROOT / "workspace" / "hpc" / "job88240_serial_chain_readonly_current.json"
)
DEFAULT_LOCAL_SUCCESSOR_INDEX = (
    PROJECT_ROOT
    / "workspace"
    / "hpc"
    / "persistent_manifests"
    / "job88240_final_successor_chain"
    / "index.json"
)
REMOTE_ROOT = ops.ALLOWED_GPU_REMOTE_ROOT
STATUS_PATHS = (
    f"{REMOTE_ROOT}/evomind_mle22/job88240_leaf_s404142_20260727/full_leaf_status.json",
    f"{REMOTE_ROOT}/evomind_mle22/job88240_siim_s42_20260727/full_siim_status.json",
    f"{REMOTE_ROOT}/evomind_mle22/job88240_ranzcr_highres_s42_20260727/full_ranzcr_status.json",
    f"{REMOTE_ROOT}/evomind_mle22/job88240_cactus_after_ranzcr_s404142_20260727/full_cactus_status.json",
    f"{REMOTE_ROOT}/evomind_mle22/job88240_cactus_after_ranzcr_s404142_20260727/full_cactus_status_v2.json",
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_remote_command() -> str:
    source = f'''import hashlib,json,pathlib,subprocess
paths={list(STATUS_PATHS)!r}
statuses=[]
for raw in paths:
    path=pathlib.Path(raw)
    record={{"path":raw,"exists":path.is_file()}}
    if path.is_file():
        data=path.read_bytes()
        record.update({{"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}})
        try:
            payload=json.loads(data.decode("utf-8"))
            record["payload"]=payload
            pid=int(payload.get("wrapper_pid") or 0)
            proc=pathlib.Path("/proc")/str(pid)
            process={{"pid":pid,"exists":bool(pid and proc.is_dir())}}
            if process["exists"]:
                process["cmdline"]=(proc/"cmdline").read_bytes().replace(b"\\0",b" ").decode("utf-8","replace")
                process["state"]=(proc/"stat").read_text(encoding="utf-8").split()[2]
            record["wrapper_process"]=process
        except Exception as exc:
            record["payload_error"]=type(exc).__name__
    statuses.append(record)
gpu=subprocess.run(["nvidia-smi","--query-gpu=index,name,memory.used,utilization.gpu","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=30)
apps=subprocess.run(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=30)
print(json.dumps({{"statuses":statuses,"gpu_exit_code":gpu.returncode,"gpu_stdout":gpu.stdout.strip(),"gpu_stderr":gpu.stderr.strip(),"apps_exit_code":apps.returncode,"apps_stdout":apps.stdout.strip(),"apps_stderr":apps.stderr.strip()}},sort_keys=True))
'''
    encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
    return f'python3 -c "import base64;exec(base64.b64decode(\'{encoded}\'))"'


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _process_record(pid: object) -> dict[str, Any]:
    try:
        process_id = int(pid or 0)
    except (TypeError, ValueError):
        process_id = 0
    record: dict[str, Any] = {"pid": process_id, "exists": False}
    if process_id <= 0 or not psutil.pid_exists(process_id):
        return record
    try:
        process = psutil.Process(process_id)
        record.update(
            {
                "exists": process.is_running(),
                "status": process.status(),
                "name": process.name(),
                "cmdline": process.cmdline(),
            }
        )
    except (psutil.Error, OSError):
        record["inspection_error"] = "process_unavailable"
    return record


def inspect_local_successors(
    index_path: Path = DEFAULT_LOCAL_SUCCESSOR_INDEX,
) -> dict[str, Any]:
    """Inspect the persistent May/Taxi successor chain without mutating it."""

    resolved_index = Path(index_path).resolve()
    report: dict[str, Any] = {
        "index_path": str(resolved_index),
        "index_exists": resolved_index.is_file(),
        "successors": [],
    }
    if not resolved_index.is_file():
        report.update({"all_workers_live": False, "all_launchers_running": False})
        return report

    index = _read_json(resolved_index)
    if index.get("schema") != "evomind.job88240.final_successor_persistent_chain.v1":
        raise ValueError("Unexpected Job 88240 local successor index schema")
    for entry in index.get("manifests", []):
        if not isinstance(entry, dict):
            continue
        manifest_path = Path(str(entry.get("manifest") or "")).resolve()
        launcher_path = Path(str(entry.get("launcher_status") or "")).resolve()
        item: dict[str, Any] = {
            "name": entry.get("name"),
            "manifest_path": str(manifest_path),
            "manifest_exists": manifest_path.is_file(),
            "launcher_status_path": str(launcher_path),
            "launcher_status_exists": launcher_path.is_file(),
        }
        manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
        launcher = _read_json(launcher_path) if launcher_path.is_file() else {}
        arguments = [str(value) for value in manifest.get("arguments", [])]
        evidence_dir: Path | None = None
        if "--evidence-dir" in arguments:
            position = arguments.index("--evidence-dir") + 1
            if position < len(arguments):
                evidence_dir = Path(arguments[position])
                if not evidence_dir.is_absolute():
                    evidence_dir = PROJECT_ROOT / evidence_dir
                evidence_dir = evidence_dir.resolve()
        evidence_status_path = (
            evidence_dir / "status_current.json" if evidence_dir is not None else None
        )
        evidence = (
            _read_json(evidence_status_path)
            if evidence_status_path is not None and evidence_status_path.is_file()
            else {}
        )
        item.update(
            {
                "task_id": manifest.get("task_id"),
                "launcher_status": launcher.get("status"),
                "launcher_observed_at": launcher.get("created_at"),
                "wrapper_process": _process_record(launcher.get("wrapper_pid")),
                "worker_process": _process_record(launcher.get("worker_pid")),
                "evidence_status_path": (
                    str(evidence_status_path) if evidence_status_path is not None else None
                ),
                "evidence_status_exists": bool(
                    evidence_status_path is not None and evidence_status_path.is_file()
                ),
                "evidence_schema": evidence.get("schema"),
                "evidence_observed_at": evidence.get("created_at"),
                "evidence_status": evidence.get("status"),
                "plan_sha256": evidence.get("plan_sha256"),
            }
        )
        report["successors"].append(item)

    successors = report["successors"]
    report.update(
        {
            "expected_successor_count": 6,
            "observed_successor_count": len(successors),
            "all_launchers_running": bool(successors)
            and all(item.get("launcher_status") == "running" for item in successors),
            "all_wrappers_live": bool(successors)
            and all(item.get("wrapper_process", {}).get("exists") is True for item in successors),
            "all_workers_live": bool(successors)
            and all(item.get("worker_process", {}).get("exists") is True for item in successors),
            "all_evidence_statuses_present": bool(successors)
            and all(item.get("evidence_status_exists") is True for item in successors),
        }
    )
    report["passed"] = bool(
        len(successors) == report["expected_successor_count"]
        and report["all_launchers_running"]
        and report["all_wrappers_live"]
        and report["all_workers_live"]
        and report["all_evidence_statuses_present"]
    )
    return report


def probe(
    output: Path = DEFAULT_OUTPUT,
    local_successor_index: Path = DEFAULT_LOCAL_SUCCESSOR_INDEX,
) -> dict[str, Any]:
    client = ops._connect()
    try:
        code, stdout, stderr = ops._run_remote(client, build_remote_command(), timeout=120)
    finally:
        client.close()
    if code != 0:
        raise ops.RemoteOpsError(f"job88240 read-only serial probe failed: exit {code}")
    remote = json.loads(stdout.strip().splitlines()[-1])
    report = {
        "schema": "evomind.hpc88240.serial_chain_readonly_probe.v3",
        "created_at": now_iso(),
        **remote,
        "local_successor_chain": inspect_local_successors(local_successor_index),
        "remote_probe_exit_code": code,
        "remote_probe_stderr_type": "" if not stderr.strip() else "nonempty",
        "process_signals_sent": 0,
        "other_processes_modified": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }
    write_json_atomic(Path(output), report)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--local-successor-index",
        type=Path,
        default=DEFAULT_LOCAL_SUCCESSOR_INDEX,
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    report = probe(args.output, args.local_successor_index)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
