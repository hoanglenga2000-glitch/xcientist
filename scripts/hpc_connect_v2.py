"""HPC Connect V2 — SSH via SOCKS5 bridge → upstream proxy → SSHPiper.
Supports parallel multi-instance connections. 9 GPU instances.
Connection chain: 127.0.0.1:7890 → 8.163.52.223:1080 → 100.85.169.63:1235 (SSHPiper)
"""
import paramiko, socket, struct, time, json, sys, os
from threading import Lock

# ── GPU Instance Registry ──────────────────────────────────────────────
# Only user-approved active allocations should remain here.
# All instances connect via SSHPiper (1:1 user→container mapping).
SSHPIPER_HOST = '100.85.169.63'
SSHPIPER_PORT = 1235

INSTANCES = {
    "87384": {"user": "aimslab-TTA-A800-1GPU","pwd_env": "HPC_87384_PASSWORD", "gpu": "1xA800 80GB", "cpu": 8,  "ram": "117GB"},
    "87318": {"user": "aimslab-kdd-ai4s",    "pwd_env": "HPC_87318_PASSWORD", "gpu": "1xA40 48GB",   "cpu": 8,  "ram": "117GB"},
}

# ── Default home dirs to try on HPC ────────────────────────────────────
HOME_CANDIDATES = [
    '/hpc2hdd/home/aimslab/research_agent_workstation',
    '/hpc2hdd/home/aimslab',
    '/home/aimslab',
    '/root',
]

# ── Connection pool (one SSH client per instance for reuse) ────────────
_pool: dict = {}
_pool_lock = Lock()

def _socks5_connect(timeout=15):
    """Create a SOCKS5 tunnel through local bridge → upstream proxy → SSHPiper."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect(('127.0.0.1', 7890))
    # SOCKS5 no-auth to local bridge
    sock.send(b'\x05\x01\x00')
    resp = sock.recv(2)
    if resp != b'\x05\x00':
        raise ConnectionError(f"Bridge SOCKS5 no-auth rejected: {resp.hex()}")
    # SOCKS5 CONNECT to SSHPiper via IPv4
    req = b'\x05\x01\x00\x01' + socket.inet_aton(SSHPIPER_HOST) + struct.pack('>H', SSHPIPER_PORT)
    sock.send(req)
    resp = b''
    while len(resp) < 10:
        chunk = sock.recv(10 - len(resp))
        if not chunk:
            raise ConnectionError("Bridge SOCKS5 response truncated")
        resp += chunk
    if resp[1] != 0:
        codes = {1:'general failure',2:'not allowed',3:'network unreachable',
                 4:'host unreachable',5:'refused',6:'TTL expired',7:'cmd not supported',8:'addr type not supported'}
        raise ConnectionError(f"Bridge SOCKS5 connect to SSHPiper failed: {codes.get(resp[1], resp[1])}")
    return sock

def connect(job_id, timeout=15):
    """Create SSH connection to HPC instance via SOCKS5 bridge → SSHPiper.
    SSHPiper uses 1:1 user→container mapping, no extra routing needed.
    Returns paramiko SSHClient. Each call creates a fresh connection
    (SSHPiper enforces single-session per connection — pooling is unreliable).
    """
    if job_id not in INSTANCES:
        raise ValueError(f"Unknown job_id: {job_id}. Available: {sorted(INSTANCES)}")

    instance = INSTANCES[job_id]
    password = os.environ.get(instance.get('pwd_env', '')) or os.environ.get('GPU_SSH_PASSWORD')
    if not password:
        raise RuntimeError(f"Missing HPC password for {job_id}")
    sock = _socks5_connect(timeout=timeout)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=SSHPIPER_HOST, port=SSHPIPER_PORT,
                username=instance['user'], password=password,
                sock=sock, timeout=timeout, allow_agent=False, look_for_keys=False)
    return ssh

def exec_cmd(job_id, command, timeout=3600):
    """Execute command on HPC instance. Returns (stdout, stderr, exit_code)."""
    ssh = connect(job_id)
    try:
        stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)
        # Use non-blocking read to avoid hang on nohup commands
        stdout.channel.settimeout(15)
        stderr.channel.settimeout(15)
        out = b''
        err = b''
        try:
            while True:
                chunk = stdout.channel.recv(65536)
                if not chunk: break
                out += chunk
        except:
            pass
        try:
            while True:
                chunk = stderr.channel.recv(65536)
                if not chunk: break
                err += chunk
        except:
            pass
        exit_code = stdout.channel.recv_exit_status()
        return out.decode('utf-8', errors='replace'), err.decode('utf-8', errors='replace'), exit_code
    finally:
        try: ssh.close()
        except: pass

def exec_async(job_id, command):
    """Execute command non-blocking, returns (ssh, stdin, stdout, stderr) tuple."""
    ssh = connect(job_id)
    stdin, stdout, stderr = ssh.exec_command(command, timeout=86400)
    return ssh, stdin, stdout, stderr

def health_check(job_id, timeout=20):
    """Quick check if instance is reachable and GPU is available."""
    try:
        out, err, code = exec_cmd(job_id, 'nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null; echo "HOME=$HOME"', timeout=timeout)
        return {
            "job_id": job_id,
            "reachable": True,
            "gpu_info": out.strip(),
            "home": [l.split('=')[1] for l in out.split('\n') if l.startswith('HOME=')],
            "exit_code": code,
        }
    except Exception as e:
        return {"job_id": job_id, "reachable": False, "error": str(e)[:200]}

def health_check_all():
    """Check all 9 instances."""
    results = {}
    for jid in INSTANCES:
        results[jid] = health_check(jid)
        status = "✓" if results[jid]["reachable"] else "✗"
        gpu = results[jid].get("gpu_info", "N/A")[:60]
        print(f"  [{jid}] {status} {INSTANCES[jid]['gpu']} | {gpu}")
    return results

def find_home(job_id):
    """Find the working home directory on the instance."""
    try:
        out, _, _ = exec_cmd(job_id, 'echo $HOME', timeout=10)
        home = out.strip()
        if home and home != '/':
            return home
    except: pass

    for candidate in HOME_CANDIDATES:
        try:
            out, _, code = exec_cmd(job_id, f'test -d {candidate} && echo EXISTS || echo NO', timeout=10)
            if 'EXISTS' in out:
                return candidate
        except: pass
    return None

def deploy_script(job_id, local_path, remote_name=None):
    """Deploy a Python script to the HPC instance."""
    if remote_name is None:
        remote_name = os.path.basename(local_path)

    home = find_home(job_id)
    if not home:
        raise RuntimeError(f"Cannot find home directory on {job_id}")

    remote_path = f"{home}/{remote_name}"
    with open(local_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Write via heredoc to avoid encoding issues
    escaped = content.replace('\\', '\\\\').replace('$', '\\$').replace('"', '\\"').replace('`', '\\`')
    cmd = f'cat > {remote_path} << "HEREDOC_END"\n{content}\nHEREDOC_END'
    out, err, code = exec_cmd(job_id, cmd, timeout=30)
    if code != 0:
        raise RuntimeError(f"Deploy failed on {job_id}: {err[:500]}")

    # Make executable
    exec_cmd(job_id, f'chmod +x {remote_path}', timeout=10)
    return remote_path

def close_all():
    """Close all pooled connections."""
    with _pool_lock:
        for jid, (ssh, _) in list(_pool.items()):
            try: ssh.close()
            except: pass
        _pool.clear()

def instance_summary():
    """Return a summary of all instances as JSON."""
    return [{
        "job_id": jid,
        "gpu": info["gpu"],
        "cpu_cores": info["cpu"],
        "ram": info["ram"],
    } for jid, info in INSTANCES.items()]

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'list'
    if cmd == 'list':
        print(json.dumps(instance_summary(), indent=2))
    elif cmd == 'health':
        health_check_all()
    elif cmd == 'connect':
        jid = sys.argv[2]
        ssh = connect(jid)
        print(f"[{jid}] connected OK, running nvidia-smi...")
        out, err, code = exec_cmd(jid, 'nvidia-smi', timeout=15)
        print(out)
    elif cmd == 'exec':
        jid = sys.argv[2]
        command = ' '.join(sys.argv[3:])
        out, err, code = exec_cmd(jid, command)
        print(out)
        if err: print(f"STDERR: {err}", file=sys.stderr)
        sys.exit(code)
    elif cmd == 'close':
        close_all()
        print("All connections closed.")
