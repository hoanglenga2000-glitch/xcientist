from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path


FIXTURE_ROOT = "/tmp/evomind-super-agent-sftp"
PORT = 22222


SETUP = r"""
import os, pathlib, shutil, subprocess, sys, time
root=pathlib.Path(sys.argv[1]); user=sys.argv[2]; port=int(sys.argv[3])
shutil.rmtree(root, ignore_errors=True); root.mkdir(mode=0o700, parents=True)
pathlib.Path('/run/sshd').mkdir(mode=0o755, parents=True, exist_ok=True)
data=root/'data'; data.mkdir(mode=0o700)
subprocess.run(['ssh-keygen','-q','-t','ed25519','-N','','-f',str(root/'host_key')],check=True)
subprocess.run(['ssh-keygen','-q','-t','ed25519','-N','','-f',str(root/'client_key')],check=True)
(root/'authorized_keys').write_text((root/'client_key.pub').read_text(),encoding='utf-8')
config=f'''Port {port}\nListenAddress 0.0.0.0\nHostKey {root}/host_key\nPidFile {root}/sshd.pid\nAuthorizedKeysFile {root}/authorized_keys\nPasswordAuthentication no\nKbdInteractiveAuthentication no\nPubkeyAuthentication yes\nPermitRootLogin no\nAllowUsers {user}\nUsePAM no\nStrictModes no\nLogLevel ERROR\nSubsystem sftp internal-sftp\n'''
(root/'sshd_config').write_text(config,encoding='utf-8')
uid=int(subprocess.check_output(['id','-u',user],text=True).strip()); gid=int(subprocess.check_output(['id','-g',user],text=True).strip())
os.chown(root,uid,gid)
for item in (data,root/'authorized_keys',root/'client_key',root/'client_key.pub'):
 os.chown(item,uid,gid)
os.chmod(root/'authorized_keys',0o600); os.chmod(root/'client_key',0o600)
subprocess.run(['/usr/sbin/sshd','-t','-f',str(root/'sshd_config')],check=True)
subprocess.Popen(['/usr/sbin/sshd','-D','-f',str(root/'sshd_config')],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
"""

TEARDOWN = r"""
import os,pathlib,shutil,signal,sys,time
root=pathlib.Path(sys.argv[1]); pid=root/'sshd.pid'
if pid.exists():
 try: os.kill(int(pid.read_text().strip()),signal.SIGTERM)
 except (ProcessLookupError,ValueError): pass
 time.sleep(.2)
shutil.rmtree(root,ignore_errors=True)
"""


def _wsl(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["wsl", "-d", "Ubuntu", *args],
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _root_python(source: str, *args: str) -> None:
    encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
    _wsl(
        "-u", "root", "--", "python3", "-c",
        "import base64;exec(base64.b64decode('" + encoded + "'))",
        *args,
    )


def _candidate_hosts() -> list[str]:
    values = ["127.0.0.1"]
    for value in _wsl("--", "hostname", "-I", check=False).stdout.split():
        if value and value not in values:
            values.append(value)
    return values


def _wait_port(timeout: float = 20.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for host in _candidate_hosts():
            try:
                with socket.create_connection((host, PORT), timeout=0.5):
                    return host
            except OSError:
                continue
        time.sleep(0.25)
    raise TimeoutError("isolated SFTP fixture did not open its localhost port")


def _unc(path: str) -> Path:
    return Path("\\\\wsl.localhost\\Ubuntu" + path.replace("/", "\\"))


def _key_from_public_line(line: str):
    import paramiko

    kind, payload, *_ = line.strip().split()
    return paramiko.PKey.from_type_string(kind, base64.b64decode(payload))


class _SftpSession:
    def __init__(self, *, host: str, username: str, key_path: Path, host_key) -> None:
        self.host = host
        self.username = username
        self.key_path = key_path
        self.host_key = host_key
        self.client = None
        self.sftp = None

    def __enter__(self):
        import paramiko

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        client.get_host_keys().add(f"[{self.host}]:{PORT}", self.host_key.get_name(), self.host_key)
        key = paramiko.Ed25519Key.from_private_key_file(str(self.key_path))
        client.connect(
            self.host,
            port=PORT,
            username=self.username,
            pkey=key,
            allow_agent=False,
            look_for_keys=False,
            timeout=5,
            banner_timeout=5,
            auth_timeout=5,
        )
        self.client = client
        self.sftp = client.open_sftp()
        return self.sftp

    def __exit__(self, *_exc):
        if self.sftp is not None:
            self.sftp.close()
        if self.client is not None:
            self.client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an isolated non-HPC SFTP capability canary.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve(strict=False)
    if output.exists():
        raise FileExistsError(output)
    username = _wsl("--", "id", "-un").stdout.strip()
    if not username or username == "root":
        raise RuntimeError("fixture requires a non-root Ubuntu user")
    try:
        _root_python(SETUP, FIXTURE_ROOT, username, str(PORT))
        host = _wait_port()
        host_key = _key_from_public_line(_unc(f"{FIXTURE_ROOT}/host_key.pub").read_text(encoding="utf-8"))
        key_path = _unc(f"{FIXTURE_ROOT}/client_key")

        from evomind_runtime.capabilities import DirectoryCapability
        from evomind_runtime.directory_broker import DirectoryBroker
        from evomind_runtime.remote_connectors import DockerSftpFixtureConnector

        connector = DockerSftpFixtureConnector(
            lambda _capability: _SftpSession(host=host, username=username, key_path=key_path, host_key=host_key)
        )
        capability = DirectoryCapability(
            "docker-sftp-canary",
            connector.connector_id,
            f"{FIXTURE_ROOT}/data",
            ("list", "stat", "read", "hash", "write", "mkdir", "copy", "sync", "delete"),
            tenant_id="canary",
            project_id="super-agent-v1",
            max_files=100,
            max_bytes=16 * 1024 * 1024,
        )
        broker = DirectoryBroker([connector])
        mount = broker.mount(capability)
        probe = b"evomind-super-agent-sftp-canary-v1"
        roundtrip = broker.verify_roundtrip(capability.directory_id, "canary/probe.bin", probe)
        copied = broker.copy(
            capability.directory_id,
            "canary/probe.bin",
            capability.directory_id,
            "canary/probe-copy.bin",
        )
        listing = broker.list(capability.directory_id, "canary", recursive=True)
        expected = hashlib.sha256(probe).hexdigest()
        passed = bool(
            mount.ok
            and roundtrip.ok
            and copied.ok
            and roundtrip.sha256 == copied.sha256 == expected
            and len([item for item in listing.entries if item["kind"] == "file"]) == 2
        )
        report = {
            "schema": "evomind.super_agent_sftp_canary.v1",
            "status": "passed" if passed else "failed",
            "connector_type": "docker-sftp-contract-on-wsl-openssh",
            "non_hpc": True,
            "fixed_host_key": True,
            "password_auth_used": False,
            "directory_capability_used": True,
            "roundtrip_sha256": expected if passed else "",
            "mount_receipt_sha256": hashlib.sha256(json.dumps(mount.to_dict(), sort_keys=True).encode()).hexdigest(),
            "roundtrip_receipt_sha256": hashlib.sha256(json.dumps(roundtrip.to_dict(), sort_keys=True).encode()).hexdigest(),
            "copy_receipt_sha256": hashlib.sha256(json.dumps(copied.to_dict(), sort_keys=True).encode()).hexdigest(),
            "files_verified": 2 if passed else 0,
            "fixture_cleaned": True,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": report["status"],
            "report_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "roundtrip_sha256": report["roundtrip_sha256"],
        }))
        return 0 if passed else 2
    finally:
        try:
            _root_python(TEARDOWN, FIXTURE_ROOT)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
