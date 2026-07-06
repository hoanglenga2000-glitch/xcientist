"""Run on Aliyun - connect to HPC via SOCKS5 proxy, execute commands."""
import paramiko, socket, struct, sys, os, time

SOCKS5_HOST = "8.163.52.223"
SOCKS5_PORT = 1080
SSHPIPER_HOST = "100.85.169.63"
SSHPIPER_PORT = 1235

HPC_USER = "aimslab-deOiwKsB"
HPC_PASS = "31PFmLLb1f"

def socks5_connect(target_host, target_port):
    """Create a socket tunneled through SOCKS5 proxy to target."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(30)
    sock.connect((SOCKS5_HOST, SOCKS5_PORT))
    # SOCKS5 handshake (no auth)
    sock.send(b'\x05\x01\x00')
    resp = sock.recv(2)
    if resp != b'\x05\x00':
        raise Exception(f"SOCKS5 auth failed: {resp}")
    # SOCKS5 CONNECT to target
    addr = socket.inet_aton(target_host)
    req = b'\x05\x01\x00\x01' + addr + struct.pack('>H', target_port)
    sock.send(req)
    resp = sock.recv(10)
    if len(resp) < 10 or resp[1] != 0x00:
        raise Exception(f"SOCKS5 connect failed: {resp.hex()}")
    return sock

def hpc_ssh():
    """Connect to HPC via SOCKS5 -> SSHPiper."""
    sock = socks5_connect(SSHPIPER_HOST, SSHPIPER_PORT)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=SSHPIPER_HOST, port=SSHPIPER_PORT,
                username=HPC_USER, password=HPC_PASS,
                sock=sock, timeout=30, allow_agent=False, look_for_keys=False)
    return ssh

def hpc_exec_cmd(ssh, cmd, timeout=600):
    """Execute command via channel (more reliable than exec_command)."""
    chan = ssh.get_transport().open_session()
    chan.settimeout(timeout)
    chan.exec_command(cmd)
    stdout = b""
    stderr = b""
    while True:
        if chan.recv_ready():
            stdout += chan.recv(65536)
        if chan.recv_stderr_ready():
            stderr += chan.recv_stderr(65536)
        if chan.exit_status_ready():
            break
        time.sleep(0.5)
    # Drain remaining
    while chan.recv_ready():
        stdout += chan.recv(65536)
    while chan.recv_stderr_ready():
        stderr += chan.recv_stderr(65536)
    exit_code = chan.recv_exit_status()
    return stdout.decode(errors='replace'), stderr.decode(errors='replace'), exit_code

def hpc_sftp_put(ssh, local_path, remote_path):
    """Upload file to HPC via SFTP."""
    sftp = ssh.open_sftp()
    sftp.put(local_path, remote_path)
    sftp.close()

def hpc_sftp_get(ssh, remote_path, local_path):
    """Download file from HPC via SFTP."""
    sftp = ssh.open_sftp()
    sftp.get(remote_path, local_path)
    sftp.close()

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "nvidia-smi"
    print(f"Connecting to HPC 87739 via SOCKS5...")
    ssh = hpc_ssh()
    stdout, stderr, code = hpc_exec_cmd(ssh, cmd)
    print(f"[STDOUT]\n{stdout}")
    if stderr:
        print(f"[STDERR]\n{stderr}")
    print(f"[EXIT: {code}]")
    ssh.close()
