import paramiko, socket, struct, time, os

SSHPIPER_HOST = "100.85.169.63"
SSHPIPER_PORT = 1235

INSTANCES = {
    "87384": ("aimslab-TTA-A800-1GPU", "HPC_87384_PASSWORD"),
    "87318": ("aimslab-kdd-ai4s", "HPC_87318_PASSWORD"),
}

SOCKS5_NOAUTH = bytes([0x05, 0x01, 0x00])

def _socks():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(15)
    sock.connect(("127.0.0.1", 7890))
    sock.send(SOCKS5_NOAUTH)
    if sock.recv(2) != bytes([0x05, 0x00]):
        raise Exception("SOCKS5 auth fail")
    req = bytes([0x05, 0x01, 0x00, 0x01]) + socket.inet_aton(SSHPIPER_HOST) + struct.pack(">H", SSHPIPER_PORT)
    sock.send(req)
    resp = sock.recv(10)
    if resp[1] != 0:
        raise Exception(f"SSHPiper unreachable: {resp[1]}")
    return sock

def connect(job_id, timeout=15):
    user, password_env = INSTANCES[job_id]
    pwd = os.environ.get(password_env) or os.environ.get("GPU_SSH_PASSWORD")
    if not pwd:
        raise RuntimeError(f"Missing SSH password env for job {job_id}")
    sock = _socks()
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=SSHPIPER_HOST, port=SSHPIPER_PORT, username=user, password=pwd,
                sock=sock, timeout=timeout, allow_agent=False, look_for_keys=False)
    return ssh

def exec_cmd(job_id, command, timeout=3600):
    ssh = connect(job_id)
    try:
        stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)
        stdout.channel.settimeout(15)
        stderr.channel.settimeout(15)
        out = b""
        err = b""
        try:
            while True:
                chunk = stdout.channel.recv(65536)
                if not chunk: break
                out += chunk
        except: pass
        try:
            while True:
                chunk = stderr.channel.recv(65536)
                if not chunk: break
                err += chunk
        except: pass
        exit_code = stdout.channel.recv_exit_status()
        return out.decode("utf-8", errors="replace"), err.decode("utf-8", errors="replace"), exit_code
    finally:
        try: ssh.close()
        except: pass
