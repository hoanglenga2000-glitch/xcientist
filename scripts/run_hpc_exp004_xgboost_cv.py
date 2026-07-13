from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import paramiko
from hpc_connect import secure_ssh_client
from hpc_runtime_contract import add_hpc_runtime_arguments, validate_hpc_runtime_arguments

ROOT = Path(__file__).resolve().parents[1]


def socks5_connect(proxy_host: str, proxy_port: int, dest_host: str, dest_port: int, timeout: float = 30.0) -> socket.socket:
    sock = socket.create_connection((proxy_host, proxy_port), timeout=timeout)
    sock.settimeout(timeout)
    sock.sendall(b"\x05\x01\x00")
    if sock.recv(2) != b"\x05\x00":
        raise RuntimeError("SOCKS5 method negotiation failed")
    host_bytes = dest_host.encode("ascii")
    sock.sendall(b"\x05\x01\x00\x03" + bytes([len(host_bytes)]) + host_bytes + struct.pack("!H", dest_port))
    header = sock.recv(4)
    if len(header) != 4 or header[0] != 5 or header[1] != 0:
        raise RuntimeError(f"SOCKS5 connect failed with response {header!r}")
    if header[3] == 1:
        sock.recv(4)
    elif header[3] == 3:
        sock.recv(sock.recv(1)[0])
    elif header[3] == 4:
        sock.recv(16)
    sock.recv(2)
    return sock


def connect(args: argparse.Namespace) -> paramiko.SSHClient:
    password = os.environ.get(args.password_env, "")
    if not password:
        raise RuntimeError(f"{args.password_env} is not configured.")
    sock = socks5_connect(args.proxy_host, args.proxy_port, args.host, args.port) if args.proxy_host else None
    client = secure_ssh_client()
    client.connect(args.host, port=args.port, username=args.user, password=password, sock=sock, allow_agent=False, look_for_keys=False, timeout=30, banner_timeout=30, auth_timeout=30)
    return client


def sftp_mkdirs(sftp: paramiko.SFTPClient, remote_path: str) -> None:
    current = ""
    for part in remote_path.strip("/").split("/"):
        current += "/" + part
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def upload_file(sftp: paramiko.SFTPClient, local: Path, remote: str) -> None:
    sftp_mkdirs(sftp, str(Path(remote).parent).replace("\\", "/"))
    sftp.put(str(local), remote)


def download_tree(sftp: paramiko.SFTPClient, remote_dir: str, local_dir: Path) -> list[str]:
    downloaded: list[str] = []
    local_dir.mkdir(parents=True, exist_ok=True)
    for item in sftp.listdir_attr(remote_dir):
        remote_path = f"{remote_dir.rstrip('/')}/{item.filename}"
        local_path = local_dir / item.filename
        if str(item.longname).startswith("d"):
            downloaded.extend(download_tree(sftp, remote_path, local_path))
        else:
            sftp.get(remote_path, str(local_path))
            downloaded.append(str(local_path.relative_to(ROOT)).replace("\\", "/"))
    return downloaded


def remote_file_exists(sftp: paramiko.SFTPClient, path: str) -> bool:
    try:
        sftp.stat(path)
        return True
    except FileNotFoundError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EXP004 XGBoost CV on the verified HPC SSH path.")
    add_hpc_runtime_arguments(parser)
    parser.add_argument("--python-executable", default="python3")
    parser.add_argument("--remote-data-dir", default="")
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", default="42,2025,260612")
    parser.add_argument("--n-estimators", type=int, default=1800)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    validate_hpc_runtime_arguments(parser, args)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "sanity" if args.sample_rows else "full"
    remote_dir = f"{args.remote_root.rstrip('/')}/playground_series_s6e6/EXP004_{mode}_{run_id}"
    local_artifact_dir = ROOT / "workspace" / "hpc_experiments" / "playground_series_s6e6" / f"EXP004_{mode}_{run_id}"
    local_artifact_dir.mkdir(parents=True, exist_ok=True)

    client = connect(args)
    sftp = client.open_sftp()
    try:
        sftp_mkdirs(sftp, f"{remote_dir}/data")
        remote_existing_data = args.remote_data_dir.strip().rstrip("/")
        if remote_existing_data and all(remote_file_exists(sftp, f"{remote_existing_data}/{name}") for name in ["train.csv", "test.csv", "sample_submission.csv"]):
            copy_command = " && ".join([f"cp '{remote_existing_data}/{name}' '{remote_dir}/data/{name}'" for name in ["train.csv", "test.csv", "sample_submission.csv"]])
            _, stdout, stderr = client.exec_command(copy_command, timeout=600)
            exit_status = stdout.channel.recv_exit_status()
            if exit_status != 0:
                raise RuntimeError(stderr.read().decode("utf-8", "replace"))
        else:
            for name in ["train.csv", "test.csv", "sample_submission.csv"]:
                upload_file(sftp, ROOT / "tasks" / "playground_series_s6e6" / "data" / name, f"{remote_dir}/data/{name}")

        upload_file(sftp, ROOT / "notebooks_or_scripts" / "exp004_xgboost_cv.py", f"{remote_dir}/exp004_xgboost_cv.py")
        command = (
            f"cd '{remote_dir}' && '{args.python_executable}' exp004_xgboost_cv.py --data-dir data --out-dir outputs "
            f"--folds {args.folds} --seeds '{args.seeds}' --n-estimators {args.n_estimators} "
            f"--learning-rate {args.learning_rate} --sample-rows {args.sample_rows} --device {args.device}"
        )
        started = time.time()
        _, stdout, stderr = client.exec_command(command, timeout=args.timeout_seconds)
        exit_status = stdout.channel.recv_exit_status()
        stdout_text = stdout.read().decode("utf-8", "replace")
        stderr_text = stderr.read().decode("utf-8", "replace")
        (local_artifact_dir / "remote_stdout.log").write_text(stdout_text, encoding="utf-8")
        (local_artifact_dir / "remote_stderr.log").write_text(stderr_text, encoding="utf-8")
        try:
            downloaded = download_tree(sftp, f"{remote_dir}/outputs", local_artifact_dir)
        except FileNotFoundError:
            downloaded = []
        metrics: dict[str, Any] | None = None
        metrics_path = local_artifact_dir / "metrics.json"
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        manifest = {
            "status": "passed" if exit_status == 0 and metrics_path.is_file() else "failed",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "experiment_id": "EXP004",
            "mode": mode,
            "run_id": run_id,
            "remote_dir": remote_dir,
            "local_artifact_dir": str(local_artifact_dir.relative_to(ROOT)).replace("\\", "/"),
            "exit_status": exit_status,
            "seconds": round(time.time() - started, 3),
            "downloaded": downloaded,
            "metrics": metrics,
            "stdout_tail": stdout_text[-4000:],
            "stderr_tail": stderr_text[-4000:],
            "official_submission_run": False,
        }
        (local_artifact_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        if manifest["status"] != "passed":
            raise SystemExit(1)
    finally:
        sftp.close()
        client.close()


if __name__ == "__main__":
    main()
