from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import struct
from datetime import datetime
from pathlib import Path

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
    client.connect(
        args.host,
        port=args.port,
        username=args.user,
        password=password,
        sock=sock,
        allow_agent=False,
        look_for_keys=False,
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
    )
    return client


def sftp_mkdirs(sftp: paramiko.SFTPClient, remote_path: str) -> None:
    current = ""
    for part in remote_path.strip("/").split("/"):
        current += "/" + part
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def remote_script() -> str:
    return r'''
from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

for key in [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
]:
    os.environ.pop(key, None)

from kaggle.api.kaggle_api_extended import KaggleApi


def timeout_handler(signum, frame):
    raise TimeoutError("Kaggle submit timed out on remote host")


signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(540)

submission = Path(os.environ["KAGGLE_SUBMISSION_FILE"]).resolve()
message = os.environ["KAGGLE_SUBMISSION_MESSAGE"]
competition = os.environ["KAGGLE_COMPETITION"]
started = time.time()
api = KaggleApi()
api.authenticate()
response = api.competition_submit(
    str(submission),
    message,
    competition,
    quiet=True,
)
signal.alarm(0)
print(json.dumps({
    "status": "submitted",
    "seconds": round(time.time() - started, 3),
    "response": str(response)[:1000],
}, ensure_ascii=False, indent=2))
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a Kaggle file from the verified HPC host without storing secrets in the repo.")
    add_hpc_runtime_arguments(parser)
    parser.add_argument("--token-env", default="KAGGLE_API_TOKEN")
    parser.add_argument("--local-submission", default="workspace/gpu/playground_series_s6e6/20260614_183531/submission.zip")
    parser.add_argument("--competition", default="playground-series-s6e6")
    parser.add_argument("--message", default="Research Agent Workstation submission")
    args = parser.parse_args()
    validate_hpc_runtime_arguments(parser, args)

    token = os.environ.get(args.token_env, "")
    if not token:
        raise RuntimeError(f"{args.token_env} is not configured.")
    local_submission = ROOT / args.local_submission
    if not local_submission.is_file():
        raise FileNotFoundError(local_submission)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    remote_dir = f"{args.remote_root.rstrip('/')}/kaggle_submit/{run_id}"
    token_remote = f"{remote_dir}/access_token"
    script_remote = f"{remote_dir}/submit.py"
    submission_remote = f"{remote_dir}/{local_submission.name}"
    client = connect(args)
    sftp = client.open_sftp()
    try:
        sftp_mkdirs(sftp, remote_dir)
        with sftp.open(token_remote, "w") as handle:
            handle.write(token)
        sftp.chmod(token_remote, 0o600)
        with sftp.open(script_remote, "w") as handle:
            handle.write(remote_script())
        sftp.chmod(script_remote, 0o700)
        sftp.put(str(local_submission), submission_remote)
        command = (
            f"cd '{remote_dir}' && "
            f"KAGGLE_API_TOKEN={shlex.quote(token_remote)} "
            f"KAGGLE_SUBMISSION_FILE={shlex.quote(local_submission.name)} "
            f"KAGGLE_SUBMISSION_MESSAGE={shlex.quote(args.message)} "
            f"KAGGLE_COMPETITION={shlex.quote(args.competition)} "
            "python submit.py; "
            f"status=$?; rm -f '{token_remote}'; exit $status"
        )
        _, stdout, stderr = client.exec_command(command, timeout=700)
        exit_status = stdout.channel.recv_exit_status()
        stdout_text = stdout.read().decode("utf-8", "replace")
        stderr_text = stderr.read().decode("utf-8", "replace")
        report = {
            "status": "passed" if exit_status == 0 else "failed",
            "remote_dir": remote_dir,
            "exit_status": exit_status,
            "stdout": stdout_text[-4000:],
            "stderr": stderr_text[-4000:],
            "token_file_removed": True,
        }
        out = ROOT / "workspace" / "kaggle_submissions" / f"{run_id}_hpc_submit.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["artifact_path"] = str(out.relative_to(ROOT)).replace("\\", "/")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if exit_status != 0:
            raise SystemExit(1)
    finally:
        try:
            sftp.remove(token_remote)
        except Exception:
            pass
        sftp.close()
        client.close()


if __name__ == "__main__":
    main()
