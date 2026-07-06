"""HPC Connect - Stable paramiko SSH via SOCKS5 proxy.

Passwords must be supplied through environment variables such as
HPC_87616_PASSWORD or GPU_SSH_PASSWORD. Do not hardcode rotating HPC secrets.
"""
import os
import paramiko, socket, struct

JOBS = {
    "87571": ("aimslab-fpgTDTSi", "HPC_87571_PASSWORD"),
    "87557": ("aimslab-zoeXIdNC", "HPC_87557_PASSWORD"),
    "87416": ("aimslab-IwkteXqP", "HPC_87416_PASSWORD"),
    "87384": ("aimslab-TTA-A800-1GPU", "HPC_87384_PASSWORD"),
    "87318": ("aimslab-kdd-ai4s", "HPC_87318_PASSWORD"),
    "87136": ("aimslab-lyudongxin", "HPC_87136_PASSWORD"),
    "87617": ("aimslab-HdLzYXoc", "HPC_87617_PASSWORD"),
    "87616": ("aimslab-wqx-SDBS", "HPC_87616_PASSWORD"),
    "87679": ("aimslab-RyBCioWD", "HPC_87679_PASSWORD"),
    "87739": ("aimslab-deOiwKsB", "HPC_87739_PASSWORD"),
}

def hpc_connect(job_id):
    user, password_env = JOBS[job_id]
    pwd = os.environ.get(password_env) or os.environ.get("GPU_SSH_PASSWORD")
    if not pwd:
        raise RuntimeError(f"Missing HPC password env: {password_env} or GPU_SSH_PASSWORD")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(30)
    sock.connect(('127.0.0.1', 7897))
    # SOCKS5 handshake (no-auth)
    sock.send(b'\x05\x01\x00')
    sock.recv(2)
    # SOCKS5 CONNECT to 100.85.169.63:1235 via IPv4
    sock.send(b'\x05\x01\x00\x01' + socket.inet_aton('100.85.169.63') + struct.pack('>H', 1235))
    sock.recv(10)
    # SSH via paramiko
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname='100.85.169.63', port=1235, username=user, password=pwd,
                sock=sock, timeout=30, allow_agent=False, look_for_keys=False,
                banner_timeout=30, auth_timeout=30)
    return ssh

def hpc_exec(job_id, command):
    ssh = hpc_connect(job_id)
    _, stdout, stderr = ssh.exec_command(command)
    out = stdout.read().decode()
    err = stderr.read().decode()
    ssh.close()
    return out, err
