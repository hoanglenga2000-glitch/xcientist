"""One-shot V3 status check via fresh SOCKS5 + SSH connection."""
import socket, struct, paramiko, time, sys

# Fresh SOCKS5 connection
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(30)
sock.connect(('127.0.0.1', 7890))

# SOCKS5 no-auth handshake
sock.sendall(b'\x05\x01\x00')
r = sock.recv(2)
if r != b'\x05\x00':
    print(f"SOCKS5 auth rejected: {r.hex()}")
    sys.exit(1)

# Connect to SSHPiper
host = b'100.85.169.63'
sock.sendall(b'\x05\x01\x00\x03' + bytes([len(host)]) + host + struct.pack('!H', 1235))
r = sock.recv(4)
if r[1] != 0:
    print(f"SOCKS5 connect failed: code {r[1]}")
    sys.exit(1)
if r[3] == 1:
    sock.recv(4)
elif r[3] == 3:
    addr_len = sock.recv(1)[0]
    sock.recv(addr_len)
sock.recv(2)  # port

# SSH via paramiko — ensure blocking mode
sock.setblocking(True)
t = paramiko.Transport(sock)
t.banner_timeout = 30
try:
    t.start_client(timeout=20)
except paramiko.SSHException:
    # Retry once
    t.close()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(30)
    sock.connect(('127.0.0.1', 7890))
    sock.sendall(b'\x05\x01\x00'); sock.recv(2)
    sock.sendall(b'\x05\x01\x00\x03' + bytes([len(host)]) + host + struct.pack('!H', 1235))
    resp = sock.recv(4)
    if resp[3] == 1: sock.recv(4)
    elif resp[3] == 3: sock.recv(sock.recv(1)[0]); sock.recv(2)
    sock.setblocking(True)
    t = paramiko.Transport(sock)
    t.start_client(timeout=20)
t.auth_password('aimslab-fpgTDTSi', 'jXldSnFD6f')

# Run comprehensive check
chan = t.open_session()
chan.settimeout(30)

cmd = """
echo "=== GPU ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>&1
echo "=== V3_PROCS ==="
ps aux | grep gpu_train_v3 | grep -v grep | wc -l
echo "=== V3_RESULTS ==="
ls /hpc2hdd/home/aimslab/results/v3_result_*.json 2>/dev/null | wc -l
python3 -c "
import json, glob
fs = sorted(glob.glob('/hpc2hdd/home/aimslab/results/v3_result_*.json'))
for f in fs:
    d = json.load(open(f))
    print(d.get('task_id','?'), round(d.get('oof_score',0),4), 'GATE='+str(d.get('gate_passed','?')))
" 2>/dev/null
echo "=== V3_LOGS ==="
for f in /tmp/v3_batch_*.log; do
    echo "--- $(basename $f) ---"
    tail -3 "$f" 2>/dev/null
done
"""
chan.exec_command(cmd)
time.sleep(12)
out = b''
while chan.recv_ready():
    out += chan.recv(65536)
err = b''
while chan.recv_stderr_ready():
    err += chan.recv_stderr(65536)

print(out.decode('utf-8', errors='replace'))
if err:
    print("STDERR:", err.decode('utf-8', errors='replace')[:500])

chan.close()
t.close()
