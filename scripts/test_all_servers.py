"""Test ALL GPU servers with updated credentials and get V3 data."""
import socket, struct, paramiko, time, os, subprocess, tarfile, base64

# All server credentials (UPDATED)
SERVERS = [
    ("S2", "aimslab-fpgTDTSi", "HPC_S2_PASSWORD", 4),
    ("S7", "aimslab-zoeXIdNC", "HPC_S7_PASSWORD", 1),
    ("S1", "aimslab-IwkteXqP", "HPC_S1_PASSWORD", 1),
    ("S6", "aimslab-TTA-A800-1GPU", "HPC_S6_PASSWORD", 1),
    ("S5", "aimslab-kdd-ai4s", "HPC_S5_PASSWORD", 1),
]

def test_server(label, user, pw_env):
    """Test connection to a server through SOCKS5 proxy."""
    pw = os.environ.get(pw_env) or os.environ.get("GPU_SSH_PASSWORD")
    if not pw:
        return False, f"Missing password env: {pw_env} or GPU_SSH_PASSWORD"
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(20)
        sock.connect(('127.0.0.1', 7890))
        sock.sendall(b'\x05\x01\x00')
        r = sock.recv(2)
        if r != b'\x05\x00':
            return False, "SOCKS5 auth failed"

        h = b'100.85.169.63'
        sock.sendall(b'\x05\x01\x00\x03' + bytes([len(h)]) + h + struct.pack('!H', 1235))
        r = sock.recv(4)
        if r[1] != 0:
            return False, "SOCKS5 connect failed: code %d" % r[1]
        if r[3] == 1: sock.recv(4)
        elif r[3] == 3: sock.recv(sock.recv(1)[0]); sock.recv(2)

        sock.setblocking(1)
        t = paramiko.Transport(sock)
        t.banner_timeout = 60
        t.start_client(timeout=20)
        t.auth_password(user, pw)

        # Get hostname + GPU info
        chan = t.open_session(); chan.settimeout(15)
        chan.exec_command("hostname && nvidia-smi --query-gpu=index,name,utilization.gpu,memory.free --format=csv,noheader 2>&1")
        time.sleep(4)
        out = b''
        while chan.recv_ready(): out += chan.recv(65536)
        result = out.decode('utf-8', errors='replace')
        chan.close()
        t.close()
        return True, result
    except Exception as e:
        return False, str(e)[:150]

# Test all servers ONE BY ONE (SSHPiper only allows one session per proxy restart)
print("=== TESTING ALL SERVERS ===")
working = []
for label, user, pw, gpus in SERVERS:
    ok, info = test_server(label, user, pw)
    status = "OK" if ok else "FAIL"
    print("\n%s (%s): %s" % (label, user, status))
    if ok:
        print(info[:300])
        working.append((label, user, pw))
    else:
        print("  Error: %s" % info[:150])

# If S2 works, use it to get V3 data
print("\n" + "="*60)
print("Working servers: %s" % ", ".join([w[0] for w in working]))

if working:
    # Use first working server to get V3 data
    label, user, pw_env = working[0]
    pw = os.environ.get(pw_env) or os.environ.get("GPU_SSH_PASSWORD")
    if not pw:
        raise RuntimeError(f"Missing password env: {pw_env} or GPU_SSH_PASSWORD")
    print("\nUsing %s to get V3 data..." % label)

    # Create tar.gz on server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM); sock.settimeout(30)
    sock.connect(('127.0.0.1', 7890))
    sock.sendall(b'\x05\x01\x00'); sock.recv(2)
    h = b'100.85.169.63'
    sock.sendall(b'\x05\x01\x00\x03' + bytes([len(h)]) + h + struct.pack('!H', 1235))
    r = sock.recv(4)
    if r[3] == 1: sock.recv(4)
    elif r[3] == 3: sock.recv(sock.recv(1)[0]); sock.recv(2)
    sock.setblocking(1)

    t = paramiko.Transport(sock); t.banner_timeout = 60; t.start_client(30)
    t.auth_password(user, pw)

    # Check what V3 files exist
    chan = t.open_session(); chan.settimeout(20)
    chan.exec_command("ls /hpc2hdd/home/aimslab/results/v3_result_*.json 2>/dev/null | wc -l && echo '---' && ls /hpc2hdd/home/aimslab/results/v3_submission_*.csv 2>/dev/null | wc -l && echo '---' && du -sh /hpc2hdd/home/aimslab/results/v3_submission_*.csv 2>/dev/null | sort -h | tail -5")
    time.sleep(5)
    out = b''
    while chan.recv_ready(): out += chan.recv(65536)
    print("V3 files:\n" + out.decode('utf-8', errors='replace'))
    chan.close()

    # Create tar.gz of V3 submissions (smaller ones first)
    chan2 = t.open_session(); chan2.settimeout(30)
    chan2.exec_command("cd /hpc2hdd/home/aimslab/results && tar czf /tmp/v3_subs_small.tar.gz $(ls v3_submission_*.csv 2>/dev/null | head -10) 2>/dev/null; ls -la /tmp/v3_subs_small.tar.gz")
    time.sleep(5)
    out2 = b''
    while chan2.recv_ready(): out2 += chan2.recv(65536)
    print("Small tar: " + out2.decode('utf-8', errors='replace').strip())
    chan2.close()

    # Download
    sftp = paramiko.SFTPClient.from_transport(t)
    local_dir = r'D:\桌面\codex\科研港科技\submissions_v3'
    os.makedirs(local_dir, exist_ok=True)

    local_tar = os.path.join(local_dir, 'v3_subs_small.tar.gz')
    sftp.get('/tmp/v3_subs_small.tar.gz', local_tar)
    sftp.close()
    t.close()

    print("Downloaded: %d bytes" % os.path.getsize(local_tar))

    # Extract
    with tarfile.open(local_tar, 'r:gz') as tar:
        tar.extractall(path=local_dir)

    import glob
    csvs = sorted(glob.glob(os.path.join(local_dir, 'v3_submission_*.csv')))
    print("Extracted %d CSV files:" % len(csvs))
    for c in csvs:
        print("  %s: %d bytes" % (os.path.basename(c), os.path.getsize(c)))

else:
    print("\nNO SERVERS AVAILABLE!")
